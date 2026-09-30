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

"""Message <-> HF-template plumbing for rendering.

Pure, stateless helpers: convert v1 ``Message`` to HF chat-template format, extract/count media, and
guard media placeholder counts. No tokenization policy decisions live here -- only mechanical
conversion used by ``rendering.py``.
"""

import json

from ...utils.helper import get_tokenizer
from ...utils.types import Message, Processor


_FALLBACK_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def _to_hf_messages(messages: list[Message], is_multimodal: bool = False) -> list[dict]:
    """Convert v1 Message format to HF format for apply_chat_template."""
    hf_messages = []
    for message in messages:
        tool_calls: list[dict] = []
        reasoning_content = ""

        if is_multimodal:
            hf_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    hf_content.append({"type": "text", "text": content["value"]})
                elif content["type"] == "reasoning":
                    reasoning_content += content["value"]
                elif content["type"] == "tool_call":
                    try:
                        tc = json.loads(content["value"])
                    except json.JSONDecodeError as e:
                        raise ValueError(f"tool_call value is not valid JSON: {content['value']!r}") from e
                    if not isinstance(tc, dict) or "name" not in tc or "arguments" not in tc:
                        raise ValueError(
                            f"tool_call must be a JSON object with 'name' and 'arguments' keys, got {tc!r}"
                        )
                    tool_calls.append(
                        {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    )
                elif content["type"] == "image_url":
                    hf_content.append({"type": "image", "image": content["value"]})
                elif content["type"] == "video_url":
                    hf_content.append({"type": "video", "video": content["value"]})
                elif content["type"] == "audio_url":
                    hf_content.append({"type": "audio", "audio": content["value"]})
            hf_msg = {"role": message["role"], "content": hf_content}
        else:
            text = ""
            for content in message["content"]:
                if content["type"] == "text":
                    text += content["value"]
                elif content["type"] == "reasoning":
                    reasoning_content += content["value"]
                elif content["type"] == "tool_call":
                    try:
                        tc = json.loads(content["value"])
                    except json.JSONDecodeError as e:
                        raise ValueError(f"tool_call value is not valid JSON: {content['value']!r}") from e
                    if not isinstance(tc, dict) or "name" not in tc or "arguments" not in tc:
                        raise ValueError(
                            f"tool_call must be a JSON object with 'name' and 'arguments' keys, got {tc!r}"
                        )
                    tool_calls.append(
                        {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    )
            hf_msg = {"role": message["role"], "content": text}

        if tool_calls:
            hf_msg["tool_calls"] = tool_calls
        if reasoning_content:
            hf_msg["reasoning_content"] = reasoning_content

        hf_messages.append(hf_msg)
    return hf_messages


def _extract_media_from_messages(messages: list[Message]) -> tuple[list, list, list]:
    """Extract image, video and audio paths/values from messages in order."""
    images, videos, audios = [], [], []
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images.append(content["value"])
            elif content["type"] == "video_url":
                videos.append(content["value"])
            elif content["type"] == "audio_url":
                audios.append(content["value"])
    return images, videos, audios


def _count_media_in_messages(messages: list[Message]) -> tuple[int, int, int]:
    """Count total images, videos and audios in messages."""
    n_images, n_videos, n_audios = 0, 0, 0
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                n_images += 1
            elif content["type"] == "video_url":
                n_videos += 1
            elif content["type"] == "audio_url":
                n_audios += 1
    return n_images, n_videos, n_audios


def _load_audios(values: list, sampling_rate: int) -> list:
    """Load audio inputs into mono waveforms resampled to ``sampling_rate``."""
    import numpy as np
    import torchaudio

    results = []
    for value in values:
        if isinstance(value, np.ndarray):
            results.append(value)
            continue

        waveform, sr = torchaudio.load(value)
        if waveform.shape[0] > 1:  # downmix to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
        results.append(waveform.squeeze(0).numpy())
    return results


def _check_placeholder_counts(
    processor: "Processor", full_text: str, n_images: int, n_videos: int, n_audios: int = 0
) -> None:
    """Guard: every media placeholder in the rendered text must originate from a media block."""
    tokenizer = get_tokenizer(processor)
    for attr, count, kind in (
        ("image_token_id", n_images, "image"),
        ("video_token_id", n_videos, "video"),
        ("audio_token_id", n_audios, "audio"),
    ):
        tid = getattr(processor, attr, None)
        if tid is None:
            tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        placeholder = tokenizer.convert_ids_to_tokens(tid)
        seen = full_text.count(placeholder)
        if seen != count:
            raise ValueError(
                f"{kind} placeholder count ({seen}) != number of {kind} blocks ({count}); "
                "media must be provided via image_url/video_url content blocks."
            )
