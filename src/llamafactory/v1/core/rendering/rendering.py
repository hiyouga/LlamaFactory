# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rendering: turn a v1 ``Sample`` into a tokenized ``ModelInput``.

This module is the orchestration + public API (``Renderer``). The mechanical pieces live in
sibling modules:
  - ``format``  -- v1<->HF message conversion
  - ``escape``  -- special-token escaping (prompt-injection hardening)

Note: ``position_ids`` are assigned by ``process_samples`` (1-based); multimodal (mrope) position
ids are expected to be recomputed by the model/trainer.
"""

import json

import numpy as np
import torch

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer, is_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample
from ..utils.collation import _MULTIMODAL_PASSTHROUGH_KEYS
from .escape import _escape_special, _escape_special_in_messages, _special_token_strings
from .format import (
    _FALLBACK_CHATML_JINJA,
    _check_placeholder_counts,
    _count_media_in_messages,
    _extract_media_from_messages,
    _load_audios,
    _to_hf_messages,
)


def _render_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    **kwargs,
) -> ModelInput:
    r"""Render messages using the model's own chat template, locating supervision by a prompt/full diff.

    Note: ``position_ids`` are not produced here; ``process_samples`` assigns a 1-based range.
    """
    tokenizer = get_tokenizer(processor)
    is_multimodal = not is_tokenizer(processor)

    template_caller = processor if is_multimodal else tokenizer
    if not getattr(template_caller, "chat_template", None):
        template_caller.chat_template = _FALLBACK_CHATML_JINJA

    # 0. Neutralize special-token strings in user-controlled text (no-op for normal data).
    specials = _special_token_strings(tokenizer)
    special_ids = {tid for tid, t in tokenizer.added_tokens_decoder.items() if getattr(t, "special", False)}
    messages = _escape_special_in_messages(messages, specials, special_ids, tokenizer)

    hf_messages = _to_hf_messages(messages, is_multimodal=is_multimodal)

    tools_parsed = None
    if tools:
        tools = _escape_special(tools, specials, special_ids, tokenizer)  # E3: tools text is user-controlled
        try:
            tools_parsed = json.loads(tools)
        except json.JSONDecodeError as e:
            raise ValueError(f"tools is not valid JSON: {tools!r}") from e
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]

    if not is_generate and hf_messages and hf_messages[-1]["role"] == "assistant":
        kwargs["enable_thinking"] = bool(hf_messages[-1].get("reasoning_content"))

    def _encode(hf_msgs: list[dict], src_msgs: list[Message], add_generation_prompt: bool):
        """Render + tokenize, expanding media via the processor. Returns (input_ids, mm_outputs)."""
        text = template_caller.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools_parsed, **kwargs
        )
        if is_multimodal and _count_media_in_messages(src_msgs) != (0, 0, 0):
            images, videos, audios = _extract_media_from_messages(src_msgs)
            # Every placeholder must come from a media block (escaping broke any literal ones).
            _check_placeholder_counts(processor, text, len(images), len(videos), len(audios))
            proc_kwargs = {"return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
            if videos:
                proc_kwargs["videos"] = videos
            if audios:
                # Audio processors want decoded waveforms at the model's sampling rate, not paths.
                proc_kwargs["audio"] = _load_audios(audios, processor.feature_extractor.sampling_rate)
            mm_outputs = processor(text=text, **proc_kwargs)
            return mm_outputs["input_ids"][0].tolist(), mm_outputs
        return tokenizer(text, add_special_tokens=False)["input_ids"], None

    # 1. Full sequence (used verbatim), plus its multimodal feature outputs.
    input_ids, outputs = _encode(hf_messages, messages, add_generation_prompt=is_generate)
    n = len(input_ids)

    def _attach_multimodal(result: ModelInput) -> None:
        if outputs is None:
            return
        for key in _MULTIMODAL_PASSTHROUGH_KEYS:
            if key in outputs:
                result[key] = outputs[key]
        mm_type_ids = outputs["mm_token_type_ids"][0].tolist() if "mm_token_type_ids" in outputs else None

        for attr, marker in (("image_token_id", 1), ("video_token_id", 2), ("audio_token_id", 3)):
            token_id = getattr(processor, attr, None)
            if token_id is None:
                token_id = getattr(tokenizer, attr, None)
            if token_id is None or token_id not in input_ids:
                continue
            if mm_type_ids is not None and marker in mm_type_ids:
                continue
            if mm_type_ids is None:
                mm_type_ids = [0] * len(input_ids)
            mm_type_ids = [marker if tid == token_id else t for t, tid in zip(mm_type_ids, input_ids)]

        if mm_type_ids is not None:
            result["mm_token_type_ids"] = mm_type_ids

    if is_generate:
        # Generation prompt only -- nothing is supervised.
        result = ModelInput(
            input_ids=input_ids,
            attention_mask=[1] * n,
            labels=[IGNORE_INDEX] * n,
            loss_weights=[0.0] * n,
        )
        _attach_multimodal(result)
        return result

    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(
            "training render expects the last message to be the supervised assistant turn; "
            "multi-turn conversations are split per turn in process_samples."
        )

    prompt_ids, _ = _encode(hf_messages[:-1], messages[:-1], add_generation_prompt=True)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        # The prompt must be a token-prefix of the full sequence for the diff to be valid. If a
        # template re-renders earlier turns when the final turn is appended, fail loud rather than
        # mislabel.
        raise ValueError(
            "prompt is not a token-prefix of the full sequence; the chat template is not "
            "prefix-stable for this turn, so diff-based labeling is unsafe."
        )

    weight = messages[-1].get("loss_weight", 1.0)
    supervised = weight > 1e-6
    labels = [IGNORE_INDEX] * len(prompt_ids)
    loss_weights = [0.0] * len(prompt_ids)
    for tid in input_ids[len(prompt_ids) :]:
        labels.append(tid if supervised else IGNORE_INDEX)
        loss_weights.append(weight)

    result = ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * n,
        labels=labels,
        loss_weights=loss_weights,
    )
    _attach_multimodal(result)
    return result


class Renderer:
    def __init__(self, processor: Processor, config=None):
        # ``config`` is accepted for call-site compatibility (ModelEngine passes the model config)
        # but is no longer needed: supervision is located by a prompt/full diff, not a per-model
        # marker table, so the renderer is model-agnostic.
        self.processor = processor

    def render_messages(
        self,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        **kwargs,
    ) -> ModelInput:
        """Render messages to model input using apply_chat_template.

        Args:
            messages: The messages to render. For training the last message must be the supervised
                assistant turn (use ``process_samples`` to split multi-turn conversations).
            tools: JSON string of tool definitions.
            is_generate: Whether to render for generation (adds generation prompt, no supervision).
            **kwargs: Extra chat-template kwargs (e.g. ``enable_thinking``) forwarded verbatim to
                ``apply_chat_template``; unset ones fall back to the template's own defaults. A
                supervised assistant turn carrying reasoning forces ``enable_thinking=True``.

        Returns:
            ModelInput with input_ids, attention_mask, labels, and loss_weights.
        """
        return _render_messages(self.processor, messages, tools, is_generate, **kwargs)

    def get_dummy_media_fragment(self, modality: str) -> dict:
        """Build (and cache) a minimal valid media fragment for ``modality`` ("image"|"video"|"audio")."""
        if modality not in ("image", "video", "audio"):
            raise ValueError(f"Unsupported dummy media modality: {modality!r} (expected image/video/audio).")
        if is_tokenizer(self.processor):
            raise RuntimeError("Cannot build a dummy media fragment for a text-only processor.")

        if not hasattr(self, "_dummy_fragments"):
            self._dummy_fragments: dict[str, dict] = {}
        if modality in self._dummy_fragments:
            return self._dummy_fragments[modality]

        from PIL import Image as _PILImage

        if modality == "image":
            media_block = {"type": "image_url", "value": _PILImage.new("RGB", (64, 64))}
            target, presence_key = 1, "pixel_values"
        elif modality == "video":
            # A minimal clip: the temporal patch size is typically 2, so provide two frames.
            media_block = {"type": "video_url", "value": np.zeros((2, 64, 64, 3), dtype=np.uint8)}
            target, presence_key = 2, "pixel_values_videos"
        else:
            # A short synthetic waveform at the model's sampling rate; the feature extractor pads it.
            sr = self.processor.feature_extractor.sampling_rate
            media_block = {"type": "audio_url", "value": np.zeros(sr // 10, dtype=np.float32)}
            target, presence_key = 3, "input_features"

        messages: list[Message] = [
            {"role": "user", "content": [media_block]},
            {"role": "assistant", "content": [{"type": "text", "value": "ok"}]},
        ]
        rendered = self.render_messages(messages)

        mm_type_ids = rendered.get("mm_token_type_ids")
        if not mm_type_ids or target not in mm_type_ids or presence_key not in rendered:
            raise RuntimeError(f"Processor did not emit {modality} placeholder tokens for the dummy sample.")

        positions = [i for i, t in enumerate(mm_type_ids) if t == target]
        # Include the surrounding start/end delimiters (vision_start/end or audio_bos/eos) so the
        # fragment matches exactly what the template emits around real media.
        lo = max(positions[0] - 1, 0)
        hi = min(positions[-1] + 2, len(rendered["input_ids"]))

        fragment: dict = {
            "input_ids": list(rendered["input_ids"][lo:hi]),
            "mm_token_type_ids": list(mm_type_ids[lo:hi]),
        }

        for key in _MULTIMODAL_PASSTHROUGH_KEYS:
            if key in rendered:
                fragment[key] = rendered[key]

        self._dummy_fragments[modality] = fragment
        return fragment

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]:
        """Process samples to model input.

        Multi-turn SFT conversations are already prefix-split in the data layer (DataEngine), so each
        ``messages`` sample is rendered once -- the diff-based renderer supervises only its last
        assistant turn.

        Args:
            samples: The samples to process.

        Returns:
            List of processed model inputs.
        """
        model_inputs = []
        for sample in samples:
            rendered: list[ModelInput] = []
            if "messages" in sample:
                model_input = self.render_messages(sample["messages"], sample.get("tools"))
                model_input["position_ids"] = list(range(1, len(model_input["input_ids"]) + 1))
                rendered.append(model_input)
            elif "chosen_messages" in sample and "rejected_messages" in sample:
                chosen_input = self.render_messages(sample["chosen_messages"], sample.get("tools"))
                rejected_input = self.render_messages(sample["rejected_messages"], sample.get("tools"))
                chosen_input["token_type_ids"] = [1] * len(chosen_input["input_ids"])
                rejected_input["token_type_ids"] = [2] * len(rejected_input["input_ids"])
                model_input = ModelInput(
                    input_ids=chosen_input["input_ids"] + rejected_input["input_ids"],
                    attention_mask=chosen_input["attention_mask"] + rejected_input["attention_mask"],
                    labels=chosen_input["labels"] + rejected_input["labels"],
                    loss_weights=chosen_input["loss_weights"] + rejected_input["loss_weights"],
                    token_type_ids=chosen_input["token_type_ids"] + rejected_input["token_type_ids"],
                )
                # chosen and rejected are independent sequences; position ids must restart at 1 for
                # each (a single continuous range would offset rejected's positional embeddings).
                model_input["position_ids"] = list(range(1, len(chosen_input["input_ids"]) + 1)) + list(
                    range(1, len(rejected_input["input_ids"]) + 1)
                )

                for key in _MULTIMODAL_PASSTHROUGH_KEYS:
                    tensors = [inp[key] for inp in (chosen_input, rejected_input) if key in inp]
                    if tensors:
                        model_input[key] = torch.cat(tensors, dim=0)

                if "mm_token_type_ids" in chosen_input or "mm_token_type_ids" in rejected_input:
                    chosen_mm = chosen_input.get("mm_token_type_ids", [0] * len(chosen_input["input_ids"]))
                    rejected_mm = rejected_input.get("mm_token_type_ids", [0] * len(rejected_input["input_ids"]))
                    model_input["mm_token_type_ids"] = chosen_mm + rejected_mm

                rendered.append(model_input)
            else:
                raise ValueError("No valid messages or chosen_messages/rejected_messages found in sample.")

            for model_input in rendered:
                if "extra_info" in sample:
                    model_input["extra_info"] = sample["extra_info"]
                if "_dataset_name" in sample:
                    model_input["_dataset_name"] = sample["_dataset_name"]
                model_inputs.append(model_input)

        return model_inputs
