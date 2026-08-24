# Copyright 2025 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

import copy
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .template import Template


def _slice_mm_inputs_for_sample(
    mm_inputs: dict[str, Any],
    batch_imglens: list[int],
    batch_vidlens: list[int],
    batch_idx: int,
    images_per_subseq: Optional[list[int]] = None,
    videos_per_subseq: Optional[list[int]] = None,
    subseq_idx: Optional[int] = None,
) -> dict[str, Any]:
    r"""Slice mm_inputs for one batch sample, optionally for a single sub-sequence when packing.

    image_grid_thw / video_grid_thw have shape [num_items, 3]. Indices for sample batch_idx
    are batch_imglens[batch_idx] images and batch_vidlens[batch_idx] videos. When subseq_idx
    is given, further restrict to that sub-seq's counts via packed_*_counts.
    has_dummy_image=True means only batch[0] will be concated with fake image and no multimodal data.
    """
    image_start_idx = sum(batch_imglens[:batch_idx])
    image_end_idx = sum(batch_imglens[: batch_idx + 1])
    video_start_idx = sum(batch_vidlens[:batch_idx])
    video_end_idx = sum(batch_vidlens[: batch_idx + 1])

    if subseq_idx is not None and images_per_subseq is not None:
        image_start_idx += sum(images_per_subseq[:subseq_idx])
        image_end_idx = image_start_idx + images_per_subseq[subseq_idx]

    if subseq_idx is not None and videos_per_subseq is not None:
        video_start_idx += sum(videos_per_subseq[:subseq_idx])
        video_end_idx = video_start_idx + videos_per_subseq[subseq_idx]

    sliced_mm_inputs: dict[str, Any] = {}
    key_to_slice_meta = {
        "image_grid_thw": (image_start_idx, image_end_idx, True),
        "video_grid_thw": (video_start_idx, video_end_idx, True),
        "second_per_grid_ts": (video_start_idx, video_end_idx, False),  # qwen2.5vl
        "video_second_per_grid": (video_start_idx, video_end_idx, False),  # qwen omni
    }

    for key, (start_idx, end_idx, assign_none_when_empty) in key_to_slice_meta.items():
        if key not in mm_inputs:
            continue

        mm_value = mm_inputs[key]
        if mm_value is not None and end_idx > start_idx:
            sliced_mm_inputs[key] = mm_value[start_idx:end_idx]
        elif assign_none_when_empty:
            sliced_mm_inputs[key] = None

    return sliced_mm_inputs


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""Expand 2d attention mask to 4d attention mask.

    Expand the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    handle packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    # Create a non-padding mask.
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    # Create indices for comparison.
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)  # [bsz, 1, seq_len, 1]
    # Create a lower triangular mask.
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images, videos and audios.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model

        if getattr(getattr(self.model, "config", None), "model_type", None) == "moss_vl":
            self.get_rope_func = None  # MOSS-VL computes its own XRoPE positions in model.forward.
        elif self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            self.get_rope_func = self.model.get_rope_index  # transformers < 4.52.0 or qwen2.5 omni
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index  # transformers >= 4.52.0
        else:
            self.get_rope_func = None

    def _compute_rope_position_ids(self, features: dict[str, "torch.Tensor"], mm_inputs: dict[str, Any]) -> None:
        r"""Compute position_ids and rope_deltas via get_rope_func for VLMs."""
        rope_index_kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "video_grid_thw": mm_inputs.get("video_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        if features["attention_mask"].sum() == 0:  # for pad tokens
            seq_len = features["input_ids"].shape[-1]
            features["position_ids"] = (
                torch.arange(seq_len).view(1, 1, seq_len).expand(3, *features["input_ids"].shape).contiguous()
            )
            features["rope_deltas"] = torch.zeros(features["input_ids"].shape[0])
            return

        if "mm_token_type_ids" in inspect.signature(self.get_rope_func).parameters:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            video_token_id = getattr(self.model.config, "video_token_id", None)
            if image_token_id is not None or video_token_id is not None:
                mm_token_type_ids = torch.zeros_like(features["input_ids"])
                if image_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == image_token_id] = 1
                if video_token_id is not None:
                    mm_token_type_ids[features["input_ids"] == video_token_id] = 2
                rope_index_kwargs["mm_token_type_ids"] = mm_token_type_ids

        if "second_per_grid_ts" in mm_inputs:  # for qwen2vl
            rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")
        elif "video_second_per_grid" in mm_inputs:  # for qwen2.5 omni
            rope_index_kwargs["second_per_grids"] = mm_inputs.get("video_second_per_grid")

        if getattr(self.model.config, "model_type", None) in ["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]:
            rope_index_kwargs["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
            feature_attention_mask = mm_inputs.get("feature_attention_mask", None)
            if feature_attention_mask is not None:  # FIXME: need to get video image lengths
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                rope_index_kwargs["audio_seqlens"] = audio_feature_lengths  # prepare for input

            features["position_ids"], rope_deltas = self.get_rope_func(**rope_index_kwargs)
            features["rope_deltas"] = rope_deltas - (1 - rope_index_kwargs["attention_mask"]).sum(dim=-1).unsqueeze(-1)
        else:  # for qwen vl
            features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

    def _compute_rope_position_ids_with_packing(
        self,
        features: dict[str, "torch.Tensor"],
        mm_inputs: dict[str, Any],
        packing_params_list: list[dict[str, Any] | None],
        batch_imglens: list[int],
        batch_vidlens: list[int],
        batch_audlens: list[int],
        has_dummy_image: bool,
    ) -> None:
        r"""Compute position_ids and rope_deltas per sample (or per sub-sequence when packed), then merge and validate."""
        bsz = features["input_ids"].size(0)
        seq_len = features["input_ids"].size(1)
        all_position_ids: list[torch.Tensor] = []
        all_rope_deltas: list[torch.Tensor] = []

        if has_dummy_image:
            # for [0, seq_len] = [0, unpadded_length + right_padding_length + fake_input_ids_len + collator_padding_length]
            # FIXME: maybe right_padding_length is large, with improper max_cutoff_len
            unpadded_length = int(features["attention_mask"][0].bool().sum().item())
            right_padding_length = int((packing_params_list[0] or {}).get("right_padding_length") or 0)
            fake_input_padding_length = max(0, seq_len - unpadded_length - right_padding_length)
            # avoid continual cuseqlens breaking varlen attention @kuangdd
            # https://github.com/hiyouga/LlamaFactory/issues/10452
            dummy_image_right_padding_mrope = (
                torch.arange(fake_input_padding_length)
                .view(1, 1, fake_input_padding_length)
                .expand(3, bsz, fake_input_padding_length)
            )
            dummy_image_right_padding_attention_mask = torch.zeros((bsz, fake_input_padding_length))
            assert self.tokenizer.padding_side == "right", "padding_side should be right when fake image is injected"
            dummy_mm_inputs = copy.deepcopy(mm_inputs)

        for sample_idx in range(bsz):
            sample_packing = (packing_params_list[sample_idx] or {}) if sample_idx < len(packing_params_list) else {}
            sequence_boundaries = sample_packing.get("sequence_boundaries")
            num_sub_seqs = (
                (len(sequence_boundaries) - 1) if sequence_boundaries and len(sequence_boundaries) > 1 else 1
            )
            image_subseq_ids = sample_packing.get("image_subseq_ids") or []
            video_subseq_ids = sample_packing.get("video_subseq_ids") or []
            images_per_subseq = (
                [image_subseq_ids.count(i) for i in range(num_sub_seqs)]
                if image_subseq_ids and num_sub_seqs > 1
                else None
            )
            videos_per_subseq = (
                [video_subseq_ids.count(i) for i in range(num_sub_seqs)]
                if video_subseq_ids and num_sub_seqs > 1
                else None
            )
            if has_dummy_image:
                mm_inputs = {}

            if num_sub_seqs <= 1:
                sample_features = {
                    "input_ids": features["input_ids"],
                    "attention_mask": features["attention_mask"][sample_idx : sample_idx + 1],
                }
                mm_inputs_for_sample = _slice_mm_inputs_for_sample(
                    mm_inputs, batch_imglens, batch_vidlens, sample_idx=sample_idx
                )
                self._compute_rope_position_ids(sample_features, mm_inputs_for_sample)
                all_position_ids.append(sample_features["position_ids"])
                all_rope_deltas.append(sample_features["rope_deltas"])
            else:
                # when we do packing, don't need rope_deltas when training.
                sample_position_ids: list[torch.Tensor] = []
                for subseq_idx in range(num_sub_seqs):
                    subseq_start = sequence_boundaries[subseq_idx]
                    subseq_end = sequence_boundaries[subseq_idx + 1]
                    subseq_features = {
                        "input_ids": features["input_ids"][sample_idx : sample_idx + 1, subseq_start:subseq_end],
                        "attention_mask": features["attention_mask"][
                            sample_idx : sample_idx + 1, subseq_start:subseq_end
                        ],
                    }
                    mm_inputs_for_subseq = _slice_mm_inputs_for_sample(
                        mm_inputs,
                        batch_imglens,
                        batch_vidlens,
                        sample_idx,
                        images_per_subseq,
                        videos_per_subseq,
                        subseq_idx,
                    )
                    self._compute_rope_position_ids(subseq_features, mm_inputs_for_subseq)
                    sample_position_ids.append(subseq_features["position_ids"])

                all_position_ids.append(torch.cat(sample_position_ids, dim=-1))

        batch_dim_for_position_ids = 1 if all_position_ids[0].dim() == 3 else 0

        features["position_ids"] = torch.cat(all_position_ids, dim=batch_dim_for_position_ids)
        if has_dummy_image:
            mm_inputs = dummy_mm_inputs

        expected_position_ids_shape = (
            (bsz, seq_len)
            if all_position_ids[0].dim() == 2
            else (
                all_position_ids[0].size(0),
                bsz,
                seq_len,
            )
        )
        # Check if position_ids shape matches expected shape.
        # for further usage, we should padding to the right when some padding token on the right.
        if has_dummy_image:
            features["position_ids"] = torch.cat([features["position_ids"], dummy_image_right_padding_mrope], dim=-1)
            features["attention_mask"] = torch.cat(
                [features["attention_mask"], dummy_image_right_padding_attention_mask], dim=-1
            )

        # Mirror the non-FA2 packing fix (#10737) for the packed-mrope path. The merged position_ids
        # is built from per-subseq sequence_boundaries, which end at cutoff_len, while
        # `DataCollatorForSeq2Seq(pad_to_multiple_of=...)` right-pads input_ids/attention_mask past
        # cutoff_len. Right-pad the trailing (masked) positions with 0 so the merged position_ids
        # matches seq_len before validating. Works for both 2D and 3D (mrope) position_ids since the
        # sequence axis is last, and is idempotent with the has_dummy_image cat above.
        pad_len = seq_len - features["position_ids"].shape[-1]
        if pad_len > 0:
            features["position_ids"] = F.pad(features["position_ids"], (0, pad_len), value=0)

        if features["position_ids"].shape != expected_position_ids_shape:
            raise ValueError(
                "Merged position_ids shape mismatch: "
                f"got {features['position_ids'].shape}, expected {expected_position_ids_shape}."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        is_moss_vl = model_type == "moss_vl"
        batch_images, batch_videos, batch_audios = [], [], []
        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
        packing_params_list: list[dict[str, Any] | None] = []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            audios = feature.pop("audios", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_audios.extend(audios)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_audlens.append(len(audios))
            batch_input_ids.append(feature["input_ids"])
            packing_params_list.append(feature.pop("packing_params", None))

        fake_input_ids = []
        has_dummy_image = False
        if (
            self.template.mm_plugin.image_token is not None
            and sum(batch_imglens) == 0
            and sum(batch_vidlens) == 0
            and not is_moss_vl  # MOSS-VL builds one native zero-valued dummy per text-only sample in its plugin.
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, fake_images, [], [], self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, fake_images, [], [], self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_images = fake_images
            batch_imglens[0] = 1
            has_dummy_image = True

        if (
            self.template.mm_plugin.audio_token is not None and sum(batch_audlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": AUDIO_PLACEHOLDER}]
            fake_audios = [np.zeros(1600)]
            fake_messages = self.template.mm_plugin.process_messages(
                fake_messages, [], [], fake_audios, self.processor
            )
            _fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            _fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                _fake_input_ids, None, [], [], fake_audios, self.tokenizer, self.processor
            )
            fake_input_ids.extend(_fake_input_ids)
            batch_audios = fake_audios
            batch_audlens[0] = 1

        if len(fake_input_ids) != 0:
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images,
            batch_videos,
            batch_audios,
            batch_imglens,
            batch_vidlens,
            batch_audlens,
            batch_input_ids,
            self.processor,
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        if "mm_token_type_ids" in mm_inputs:  # need tensor-like for gemma4
            mm_token_type_ids = mm_inputs.pop("mm_token_type_ids")
            max_len = max(len(ids) for ids in mm_token_type_ids)
            padded = []
            for ids in mm_token_type_ids:
                pad_len = max_len - len(ids)
                if self.tokenizer.padding_side == "right":
                    padded.append(ids + [0] * pad_len)
                else:
                    padded.append([0] * pad_len + ids)

            mm_inputs["mm_token_type_ids"] = torch.tensor(padded, dtype=torch.long)

        features: dict[str, torch.Tensor] = super().__call__(features)

        bsz, seq_len = features["input_ids"].shape[:2]
        is_omni = model_type in [
            "qwen2_5_omni_thinker",
            "qwen3_omni_moe_thinker",
        ]

        if self.get_rope_func is not None:
            # for mmrope situation, we should calculate position_ids and rope_deltas per sample.
            # When neat_packing is on, each sample has packing_params; None means no packing for that sample.
            boundaries_list = [p.get("sequence_boundaries") if p is not None else None for p in packing_params_list]
            has_packing = any(b is not None and len(b) > 2 for b in boundaries_list)
            if has_dummy_image and has_packing:
                # FIXME: too tricky, need to be refactored @kuangdd
                features["has_dummy_image"] = True

            # When fake image/audio was injected, sequence_boundaries no longer match the tensor; use non-packing path.
            if not has_packing:
                self._compute_rope_position_ids(features, mm_inputs)
            else:
                if is_omni:  # TODO: support omni models for packed sequences @kuangdd
                    raise RuntimeError("Omni models are not supported for packed sequences for now.")

                self._compute_rope_position_ids_with_packing(
                    features,
                    mm_inputs,
                    packing_params_list,
                    batch_imglens,
                    batch_vidlens,
                    batch_audlens,
                    has_dummy_image,
                )

            # For transformers compatibility, after https://github.com/huggingface/transformers/issues/39400
            if features["position_ids"].dim() == 3:
                features["position_ids"] = torch.cat(
                    [features["position_ids"][0].unsqueeze(0), features["position_ids"]], dim=0
                )

        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None) in MROPE_MODELS
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError(f"{self.model.config.model_type} requires 3D position ids for mrope.")

        if (
            "cross_attention_mask" in mm_inputs and mm_inputs["cross_attention_mask"].dtype != torch.bool
        ):  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        if is_moss_vl:
            mm_inputs = self.template.mm_plugin.post_process_mossvl_inputs(features, mm_inputs, self.processor)

        features.update(mm_inputs)

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for 4d attention mask."""

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32
    neat_packing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            if self.model is not None and getattr(self.model.config, "model_type", None) in ["gemma4", "gpt_oss"]:
                raise ValueError("Neat packing is not supported for gemma4, gpt_oss models for now.")

    @staticmethod
    def _unpad_packed_features(features: dict[str, Any]) -> None:
        r"""Trim padded positions for packed FA2 batches."""
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
            return

        seq_len = attention_mask.size(1)
        non_padding_indices = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        if non_padding_indices.numel() == seq_len:
            return

        keys_on_seq_dim_1 = {"input_ids", "labels", "attention_mask", "token_type_ids"}
        for key, value in list(features.items()):
            if not torch.is_tensor(value):
                continue

            if key == "position_ids" and value.size(-1) == seq_len:
                features[key] = value.index_select(-1, non_padding_indices)
            elif (
                key == "cross_attention_mask" and value.dim() >= 2 and value.size(0) == 1 and value.size(1) == seq_len
            ):
                features[key] = value.index_select(1, non_padding_indices)
            elif key in keys_on_seq_dim_1 and value.dim() == 2 and value.size(0) == 1 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding_indices)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        has_dummy_image = features.pop("has_dummy_image", False)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        if self.neat_packing and self.attn_implementation == "flash_attention_2":  # FIXME compatibility fa3/fa4
            assert features["input_ids"].shape[0] == 1, "bsz should be 1 for neat packing"
            if not has_dummy_image:
                self._unpad_packed_features(features)

            features["attention_mask"] = None  # let transformers handle causal packed mask.
        else:
            # `DataCollatorForSeq2Seq(pad_to_multiple_of=...)` pads `input_ids`/`attention_mask`
            # but leaves `position_ids` untouched (it is not in `model_input_names`). On the
            # non-FA2 packing path we do not unpad, so `position_ids` stays shorter than
            # `input_ids`, which makes cos/sin shorter than query and crashes
            # `apply_rotary_pos_emb`. Right-pad `position_ids` to the padded length to match.
            position_ids = features.get("position_ids")
            if torch.is_tensor(position_ids):
                pad_len = features["input_ids"].shape[-1] - position_ids.shape[-1]
                if pad_len > 0:
                    features["position_ids"] = F.pad(position_ids, (0, pad_len), value=0)

        for key, value in features.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for pairwise data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        r"""Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                    "audios": feature["audios"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for KTO data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": feature["audios"],
            }
            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "cross_attention_mask" in kl_batch:  # for mllama inputs
            batch["kl_cross_attention_mask"] = kl_batch["cross_attention_mask"]

        if "token_type_ids" in kl_batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
