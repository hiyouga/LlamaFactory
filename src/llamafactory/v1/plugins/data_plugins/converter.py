# Copyright 2025 the LlamaFactory team.
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


import json
import re
from typing import Any, Literal, NotRequired, TypedDict

from ...utils import logging
from ...utils.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ...utils.plugin import BasePlugin
from ...utils.types import Content, DPOSample, Sample, SFTSample, ToolCall


logger = logging.get_logger(__name__)


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: str
    input: NotRequired[str]
    output: str
    images: NotRequired[list[str] | str]
    videos: NotRequired[list[str] | str]
    audios: NotRequired[list[str] | str]


SharegptMessage = TypedDict(
    "SharegptMessage",
    {"from": Literal["human", "gpt", "system", "function_call", "observation"], "value": str},
)


class SharegptSample(TypedDict, total=False):
    conversations: list[SharegptMessage]
    tools: NotRequired[str]
    images: NotRequired[list[str] | str]
    videos: NotRequired[list[str] | str]
    audios: NotRequired[list[str] | str]


class OpenaiMessage(TypedDict, total=False):
    role: Literal["user", "assistant", "tool"]
    content: str


class OpenaiSample(TypedDict, total=False):
    messages: list[OpenaiMessage]


class PairSample(TypedDict, total=False):
    chosen: list[OpenaiMessage]
    rejected: list[OpenaiMessage]
    images: NotRequired[list[str] | str]
    videos: NotRequired[list[str] | str]
    audios: NotRequired[list[str] | str]


# Inline media tag -> v1 content block type, and the raw-sample column holding the paths.
_MEDIA_SPECS: tuple[tuple[str, str, str], ...] = (
    (IMAGE_PLACEHOLDER, "image_url", "images"),
    (VIDEO_PLACEHOLDER, "video_url", "videos"),
    (AUDIO_PLACEHOLDER, "audio_url", "audios"),
)
_TAG_TO_BLOCK = {tag: block_type for tag, block_type, _col in _MEDIA_SPECS}
_TAG_PATTERN = re.compile("(" + "|".join(re.escape(tag) for tag, _b, _c in _MEDIA_SPECS) + ")")


def _as_media_list(value: Any) -> list:
    """Normalize a media column value into a list of paths/URLs (None -> [], scalar -> [scalar])."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_media_iters(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Build per-modality path iterators from a raw sample's media columns."""
    return {tag: iter(_as_media_list(raw_sample.get(col))) for tag, _block_type, col in _MEDIA_SPECS}


def _to_content_blocks(text: str, media_iters: dict[str, Any]) -> list[Content]:
    """Split ``text`` on inline media placeholders, interleaving media-url content blocks.

    Each placeholder consumes the next path from its modality iterator (in document order). Plain
    text with no placeholders yields a single text block (byte-identical to the legacy behavior).
    Raises on an unmatched placeholder (more tags than media files).
    """
    if not _TAG_PATTERN.search(text):
        return [{"type": "text", "value": text}]

    blocks: list[Content] = []
    for segment in _TAG_PATTERN.split(text):
        block_type = _TAG_TO_BLOCK.get(segment)
        if block_type is not None:
            try:
                path = next(media_iters[segment])
            except StopIteration:
                raise ValueError(f"More {segment} tags than provided media files.") from None
            blocks.append({"type": block_type, "value": path})
        elif segment:
            blocks.append({"type": "text", "value": segment})
    return blocks


def _assert_media_consumed(media_iters: dict[str, Any]) -> None:
    """Ensure every media file was referenced by a tag (fewer tags than media -> error)."""
    for tag, media_iter in media_iters.items():
        unused = len(list(media_iter))
        if unused:
            raise ValueError(f"Fewer {tag} tags than provided media files ({unused} unused).")


class DataConverterPlugin(BasePlugin):
    """Plugin for data converters."""

    def __call__(self, raw_sample: dict[str, Any]) -> Sample:
        return super().__call__(raw_sample)


@DataConverterPlugin("alpaca").register()
def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """Convert Alpaca sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en

    Args:
        raw_sample (AlpacaSample): Alpaca sample.

    Returns:
        SFTSample: SFT sample.
    """
    messages = []
    media_iters = _build_media_iters(raw_sample)
    if "system" in raw_sample:
        messages.append(
            {"role": "system", "content": [{"type": "text", "value": raw_sample["system"]}], "loss_weight": 0.0}
        )

    if "instruction" in raw_sample or "input" in raw_sample:
        messages.append(
            {
                "role": "user",
                "content": _to_content_blocks(
                    raw_sample.get("instruction", "") + raw_sample.get("input", ""), media_iters
                ),
                "loss_weight": 0.0,
            }
        )

    if "output" in raw_sample:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "value": raw_sample["output"]}], "loss_weight": 1.0}
        )

    _assert_media_consumed(media_iters)
    return {"messages": messages}


@DataConverterPlugin("sharegpt").register()
def sharegpt_converter(raw_sample: SharegptSample) -> SFTSample:
    """Convert ShareGPT sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/glaive_toolcall_en

    Args:
        raw_sample (SharegptSample): ShareGPT sample.

    Returns:
        SFTSample: SFT sample.
    """
    tag_mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
        "observation": "tool",
        "function_call": "assistant",
    }
    sample = {}
    messages = []
    media_iters = _build_media_iters(raw_sample)
    for message in raw_sample.get("conversations", []):
        tag = message["from"]
        if tag not in tag_mapping:
            logger.warning_rank0(f"Unsupported role tag {tag} in message: {message}")
        elif tag == "function_call":
            try:
                tool_calls: ToolCall | list[ToolCall] = json.loads(message["value"])
            except json.JSONDecodeError:
                logger.warning_rank0(f"Invalid tool call format: {str(message['value'])}")
                continue

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "value": json.dumps(tool_call)} for tool_call in tool_calls],
                    "loss_weight": 1.0,
                }
            )
        else:
            messages.append(
                {
                    "role": tag_mapping[tag],
                    "content": _to_content_blocks(message["value"], media_iters),
                    "loss_weight": 1.0 if tag == "gpt" else 0.0,
                }
            )

    _assert_media_consumed(media_iters)
    sample["messages"] = messages

    tools = raw_sample.get("tools")
    if tools:
        try:
            tools: list[dict[str, Any]] = json.loads(tools)
            sample["tools"] = json.dumps(tools)
        except json.JSONDecodeError:
            logger.warning_rank0(f"Invalid tools format: {str(tools)}")

    return sample


@DataConverterPlugin("pair").register()
def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to DPO sample.

    See raw example at: https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs

    Args:
        raw_sample (PairSample): pair sample with chosen, rejected fields.

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
    """

    def process_message(raw_messages: list[OpenaiMessage]):
        # chosen and rejected share the sample's media; each side consumes its own iterators.
        media_iters = _build_media_iters(raw_sample)
        messages = []
        for message in raw_messages:
            if message["role"] == "tool":
                try:
                    tool_calls: ToolCall | list[ToolCall] = json.loads(message["content"])
                except json.JSONDecodeError:
                    logger.warning_rank0(f"Invalid tool call format: {str(message['content'])}")
                    continue

                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]

                messages.append(
                    {
                        "role": message["role"],
                        "content": [{"type": "tool_call", "value": json.dumps(tool_call)} for tool_call in tool_calls],
                        "loss_weight": 1.0 if message["role"] == "assistant" else 0.0,
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": _to_content_blocks(message["content"], media_iters),
                        "loss_weight": 1.0 if message["role"] == "assistant" else 0.0,
                    }
                )

        _assert_media_consumed(media_iters)
        return messages

    sample = {}
    sample["chosen_messages"] = process_message(raw_sample.get("chosen", []))
    sample["rejected_messages"] = process_message(raw_sample.get("rejected", []))

    tools = raw_sample.get("tools")
    if tools:
        try:
            tools: list[dict[str, Any]] = json.loads(tools)
            sample["tools"] = json.dumps(tools)
        except json.JSONDecodeError:
            logger.warning_rank0(f"Invalid tools format: {str(tools)}")

    return sample
