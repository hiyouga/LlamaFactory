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

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.extras.packages import is_pyav_available, is_transformers_version_greater_than
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    from llamafactory.data.mm_plugin import BasePlugin
    from llamafactory.model.loader import TokenizerModule


HF_TOKEN = os.getenv("HF_TOKEN")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")
TINY_LLAMA4 = os.getenv("TINY_LLAMA4", "llamafactory/tiny-random-Llama-4")

MM_MESSAGES = [
    {"role": "user", "content": "<image>What is in this image?"},
    {"role": "assistant", "content": "A cat."},
]

OMNI_MESSAGES = [
    {"role": "user", "content": "<image>What is in this image?"},
    {"role": "assistant", "content": "A cat."},
    {"role": "user", "content": "<audio>What is in this audio?"},
    {"role": "assistant", "content": "Nothing."},
]

TEXT_MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
]

VIDEO_MESSAGES = [
    {"role": "user", "content": "<video>What is in this video?"},
    {"role": "assistant", "content": "A cat."},
]

AUDIOS = [np.zeros(1600)]

IMAGES = [Image.new("RGB", (32, 32), (255, 255, 255))]

VIDEOS = [[Image.new("RGB", (32, 32), (255, 255, 255))] * 4]

NO_IMAGES = []

NO_VIDEOS = []

NO_AUDIOS = []

IMGLENS = [1]

AUDLENS = [1]

NO_IMGLENS = [0]

NO_VIDLENS = [0]

NO_AUDLENS = [0]

INPUT_IDS = [0, 1, 2, 3, 4]

LABELS = [0, 1, 2, 3, 4]

BATCH_IDS = [[1] * 1024]


def _get_mm_inputs(processor: "ProcessorMixin") -> dict[str, "torch.Tensor"]:
    image_processor: BaseImageProcessor = getattr(processor, "image_processor")
    return image_processor(images=IMAGES, return_tensors="pt")


def _get_omni_inputs(processor: "ProcessorMixin") -> dict[str, "torch.Tensor"]:
    mm_inputs = {}
    image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
    feature_extractor = getattr(processor, "feature_extractor", None)

    mm_inputs.update(image_processor(IMAGES, return_tensors="pt"))
    mm_inputs.update(
        feature_extractor(
            AUDIOS,
            sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
    )
    mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")
    return mm_inputs


def _is_close(batch_a: dict[str, Any], batch_b: dict[str, Any]) -> None:
    assert batch_a.keys() == batch_b.keys()
    for key in batch_a.keys():
        if isinstance(batch_a[key], torch.Tensor):
            assert torch.allclose(batch_a[key], batch_b[key], rtol=1e-4, atol=1e-5)
        elif isinstance(batch_a[key], list) and all(isinstance(item, torch.Tensor) for item in batch_a[key]):
            assert len(batch_a[key]) == len(batch_b[key])
            for tensor_a, tensor_b in zip(batch_a[key], batch_b[key]):
                assert torch.allclose(tensor_a, tensor_b, rtol=1e-4, atol=1e-5)
        else:
            assert batch_a[key] == batch_b[key]


def _load_tokenizer_module(model_name_or_path: str) -> "TokenizerModule":
    model_args, *_ = get_infer_args({"model_name_or_path": model_name_or_path, "template": "default"})
    return load_tokenizer(model_args)


def _check_plugin(
    plugin: "BasePlugin",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    expected_mm_messages: list[dict[str, str]] = MM_MESSAGES,
    expected_input_ids: list[int] = INPUT_IDS,
    expected_labels: list[int] = LABELS,
    expected_mm_inputs: dict[str, Any] = {},
    expected_no_mm_inputs: dict[str, Any] = {},
) -> None:
    if plugin.__class__.__name__ == "Qwen2OmniPlugin":  # test omni_messages
        assert plugin.process_messages(OMNI_MESSAGES, IMAGES, NO_VIDEOS, AUDIOS, processor) == expected_mm_messages
        assert plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, NO_VIDEOS, AUDIOS, tokenizer, processor) == (
            expected_input_ids,
            expected_labels,
        )
        _is_close(
            plugin.get_mm_inputs(IMAGES, NO_VIDEOS, AUDIOS, IMGLENS, NO_VIDLENS, AUDLENS, BATCH_IDS, processor),
            expected_mm_inputs,
        )
    elif plugin.__class__.__name__ == "Qwen3VLPlugin":  # only check replacement
        assert plugin.process_messages(VIDEO_MESSAGES, NO_IMAGES, VIDEOS, NO_AUDIOS, processor) == expected_mm_messages
    elif plugin.__class__.__name__ != "BasePlugin":  # test mm_messages
        assert plugin.process_messages(MM_MESSAGES, IMAGES, NO_VIDEOS, NO_AUDIOS, processor) == expected_mm_messages
        assert plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, NO_VIDEOS, NO_AUDIOS, tokenizer, processor) == (
            expected_input_ids,
            expected_labels,
        )
        _is_close(
            plugin.get_mm_inputs(IMAGES, NO_VIDEOS, NO_AUDIOS, IMGLENS, NO_VIDLENS, NO_AUDLENS, BATCH_IDS, processor),
            expected_mm_inputs,
        )

    # test text_messages
    assert plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, NO_VIDEOS, NO_AUDIOS, processor) == TEXT_MESSAGES
    assert plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, NO_VIDEOS, NO_AUDIOS, tokenizer, processor) == (
        INPUT_IDS,
        LABELS,
    )
    _is_close(
        plugin.get_mm_inputs(
            NO_IMAGES, NO_VIDEOS, NO_AUDIOS, NO_IMGLENS, NO_VIDLENS, NO_AUDLENS, BATCH_IDS, processor
        ),
        expected_no_mm_inputs,
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_base_plugin():
    tokenizer_module = _load_tokenizer_module(model_name_or_path=TINY_LLAMA3)
    base_plugin = get_mm_plugin(name="base")
    check_inputs = {"plugin": base_plugin, **tokenizer_module}
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.skipif(not is_transformers_version_greater_than("4.50.0"), reason="Requires transformers>=4.50.0")
def test_gemma3_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="google/gemma-3-4b-it")
    gemma3_plugin = get_mm_plugin(name="gemma3", image_token="<image_soft_token>")
    image_tokens_expanded = "<image_soft_token>" * image_seqlen
    check_inputs = {"plugin": gemma3_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace("<image>", f"\n\n<start_of_image>{image_tokens_expanded}<end_of_image>\n\n")
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"].pop("num_crops")
    check_inputs["expected_mm_inputs"]["token_type_ids"] = [[0] * 1024]
    check_inputs["expected_no_mm_inputs"] = {"token_type_ids": [[0] * 1024]}
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("5.6.0"), reason="Requires transformers>=5.6.0")
def test_gemma4_plugin():
    tokenizer_module = _load_tokenizer_module(model_name_or_path="google/gemma-4-31B-it")
    processor = tokenizer_module["processor"]
    gemma4_plugin = get_mm_plugin(name="gemma4", image_token="<|image|>", video_token="<|video|>")
    check_inputs = {"plugin": gemma4_plugin, **tokenizer_module}
    # validate
    mm_inputs = gemma4_plugin._get_mm_inputs(IMAGES, NO_VIDEOS, NO_AUDIOS, processor)
    num_image_soft_tokens = 256  # when we use default max_soft_tokens=280
    image_token = getattr(processor, "image_token")
    boi_token = getattr(processor, "boi_token")
    eoi_token = getattr(processor, "eoi_token")

    expected_mm_type_ids = [
        [int(token_id == getattr(processor, "image_token_id")) for token_id in token_ids] for token_ids in BATCH_IDS
    ]
    check_inputs["expected_mm_messages"] = [
        {
            "role": "user",
            "content": f"{boi_token}{image_token * num_image_soft_tokens}{eoi_token}What is in this image?",
        },
        {"role": "assistant", "content": "A cat."},
    ]
    for key in ("num_soft_tokens_per_image",):
        mm_inputs.pop(key, None)

    mm_inputs["mm_token_type_ids"] = expected_mm_type_ids
    check_inputs["expected_mm_inputs"] = mm_inputs
    check_inputs["expected_no_mm_inputs"] = {"mm_token_type_ids": expected_mm_type_ids}
    _check_plugin(**check_inputs)


def test_gemma4_plugin_uses_audio_processor_fallback(monkeypatch):
    class AudioProcessor:
        sampling_rate = 16000

        def __call__(self, audios, **kwargs):
            return {}

    class Processor:
        image_token = "<|image|>"
        video_token = "<|video|>"
        audio_token = "<|audio|>"
        boi_token = "<start_of_image>"
        eoi_token = "<end_of_image>"
        boa_token = "<start_of_audio>"
        eoa_token = "<end_of_audio>"
        audio_processor = AudioProcessor()

        @staticmethod
        def _compute_audio_num_tokens(audio, sampling_rate):
            assert sampling_rate == 16000
            return 2

    gemma4_plugin = get_mm_plugin(name="gemma4", audio_token="<|audio|>")
    monkeypatch.setattr(
        gemma4_plugin,
        "_regularize_audios",
        lambda audios, **kwargs: {"audios": audios},
    )
    messages = [{"role": "user", "content": "<audio>"}]

    processed = gemma4_plugin.process_messages(messages, NO_IMAGES, NO_VIDEOS, AUDIOS, Processor())

    assert processed == [
        {
            "role": "user",
            "content": "<start_of_audio><|audio|><|audio|><end_of_audio>",
        }
    ]


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.52.0"), reason="Requires transformers>=4.52.0")
def test_internvl_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="OpenGVLab/InternVL3-1B-hf")
    internvl_plugin = get_mm_plugin("intern_vl", image_token="<image>", video_token="<video>")
    check_inputs = {"plugin": internvl_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace("<image>", f"<img>{'<IMG_CONTEXT>' * image_seqlen * 1}</img>")
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"].pop("num_patches", None)
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.51.0"), reason="Requires transformers>=4.51.0")
def test_llama4_plugin():
    tokenizer_module = _load_tokenizer_module(model_name_or_path=TINY_LLAMA4)
    processor = tokenizer_module["processor"]
    llama4_plugin = get_mm_plugin(name="llama4", image_token="<|image|>")
    check_inputs = {"plugin": llama4_plugin, **tokenizer_module}
    mm_inputs = _get_mm_inputs(tokenizer_module["processor"])
    image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
    num_patches_per_chunk = int(
        (image_height // processor.patch_size) * (image_width // processor.patch_size) // processor.downsample_ratio
    )
    aspect_ratios = mm_inputs.pop("aspect_ratios")
    tokens_for_this_image = processor._prompt_split_image(aspect_ratios[0], num_patches_per_chunk)
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", tokens_for_this_image) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = mm_inputs
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
def test_llava_plugin():
    image_seqlen = 576
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/llava-1.5-7b-hf")
    llava_plugin = get_mm_plugin(name="llava", image_token="<image>")
    check_inputs = {"plugin": llava_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
def test_llava_next_plugin():
    image_seqlen = 1176
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/llava-v1.6-vicuna-7b-hf")
    llava_next_plugin = get_mm_plugin(name="llava_next", image_token="<image>")
    check_inputs = {"plugin": llava_next_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
def test_llava_next_video_plugin():
    image_seqlen = 1176
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/LLaVA-NeXT-Video-7B-hf")
    llava_next_video_plugin = get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>")
    check_inputs = {"plugin": llava_next_video_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_paligemma_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="google/paligemma-3b-pt-224")
    paligemma_plugin = get_mm_plugin(name="paligemma", image_token="<image>")
    check_inputs = {"plugin": paligemma_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "") for key, value in message.items()} for message in MM_MESSAGES
    ]
    check_inputs["expected_input_ids"] = [
        tokenizer_module["tokenizer"].convert_tokens_to_ids(paligemma_plugin.image_token)
    ] * image_seqlen + INPUT_IDS
    check_inputs["expected_labels"] = [-100] * image_seqlen + LABELS
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"]["token_type_ids"] = [[0] * image_seqlen + [1] * (1024 - image_seqlen)]
    check_inputs["expected_no_mm_inputs"] = {"token_type_ids": [[1] * 1024]}
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.50.0"), reason="Requires transformers>=4.50.0")
def test_pixtral_plugin():
    image_slice_height, image_slice_width = 2, 2
    tokenizer_module = _load_tokenizer_module(model_name_or_path="mistral-community/pixtral-12b")
    pixtral_plugin = get_mm_plugin(name="pixtral", image_token="[IMG]")
    check_inputs = {"plugin": pixtral_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace(
                "<image>",
                ("{}[IMG_BREAK]".format("[IMG]" * image_slice_width) * image_slice_height).rsplit("[IMG_BREAK]", 1)[0]
                + "[IMG_END]",
            )
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"]["pixel_values"] = check_inputs["expected_mm_inputs"]["pixel_values"][0]
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.52.0"), reason="Requires transformers>=4.52.0")
def test_qwen2_omni_plugin():
    image_seqlen, audio_seqlen = 4, 2
    tokenizer_module = _load_tokenizer_module(model_name_or_path="Qwen/Qwen2.5-Omni-7B")
    qwen2_omni_plugin = get_mm_plugin(
        name="qwen2_omni",
        image_token="<|IMAGE|>",
        video_token="<|VIDEO|>",
        audio_token="<|AUDIO|>",
        vision_bos_token="<|vision_bos|>",
        vision_eos_token="<|vision_eos|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
    )
    check_inputs = {"plugin": qwen2_omni_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: (
                value.replace("<image>", f"<|vision_bos|>{'<|IMAGE|>' * image_seqlen}<|vision_eos|>").replace(
                    "<audio>", f"<|audio_bos|>{'<|AUDIO|>' * audio_seqlen}<|audio_eos|>"
                )
            )
            for key, value in message.items()
        }
        for message in OMNI_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_omni_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen2_vl_plugin():
    image_seqlen = 4
    tokenizer_module = _load_tokenizer_module(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    qwen2_vl_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>")
    check_inputs = {"plugin": qwen2_vl_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace("<image>", "<|vision_start|>{}<|vision_end|>".format("<|image_pad|>" * image_seqlen))
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


def test_moss_vl_plugin():
    messages = [
        {"role": "user", "content": "First <image>, finally <image>."},
        {"role": "assistant", "content": "Done."},
    ]
    expected_messages = [
        {"role": "user", "content": "First <|image_pad|>, finally <|image_pad|>."},
        {"role": "assistant", "content": "Done."},
    ]
    processor = SimpleNamespace(image_processor=object(), video_processor=object())
    plugin = get_mm_plugin(name="moss_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")

    processed_messages = plugin.process_messages(messages, [object(), object()], [], [], processor)

    assert processed_messages == expected_messages
    assert messages[0]["content"] == "First <image>, finally <image>."


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.57.0"), reason="Requires transformers>=4.57.0")
def test_qwen3_vl_plugin():
    frame_seqlen = 1
    tokenizer_module = _load_tokenizer_module(model_name_or_path="Qwen/Qwen3-VL-30B-A3B-Instruct")
    qwen3_vl_plugin = get_mm_plugin(name="qwen3_vl", video_token="<|video_pad|>")
    check_inputs = {"plugin": qwen3_vl_plugin, **tokenizer_module}
    video_token = "<|video_pad|>" * frame_seqlen
    first_video = (
        f"<0.2 seconds><|vision_start|>{video_token}<|vision_end|>"
        f"<1.2 seconds><|vision_start|>{video_token}<|vision_end|>"
    )
    second_video = first_video + f"<2.2 seconds><|vision_start|>{video_token}<|vision_end|>"
    videos = [
        [Image.new("RGB", (32, 32), (255, 255, 255))] * 4,
        [Image.new("RGB", (32, 32), (255, 255, 255))] * 6,
    ]
    messages = [
        {"role": "user", "content": "Compare these videos: <video> and <video>."},
        {"role": "assistant", "content": "They are different."},
    ]
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<video>", first_video) for key, value in message.items()} for message in VIDEO_MESSAGES
    ]
    _check_plugin(**check_inputs)
    assert qwen3_vl_plugin.process_messages(messages, NO_IMAGES, videos, NO_AUDIOS, tokenizer_module["processor"]) == [
        {
            "role": "user",
            "content": f"Compare these videos: {first_video} and {second_video}.",
        },
        {"role": "assistant", "content": "They are different."},
    ]


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.57.0"), reason="Requires transformers>=4.57.0")
@pytest.mark.skipif(not is_pyav_available(), reason="Requires pyav")
def test_qwen3_vl_plugin_video_path():
    video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "mllm_demo_data", "1.mp4")
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        pytest.skip(f"Video file not found: {video_path}")

    tokenizer_module = _load_tokenizer_module(model_name_or_path="Qwen/Qwen3-VL-30B-A3B-Instruct")
    processor = tokenizer_module["processor"]
    qwen3_vl_plugin = get_mm_plugin(name="qwen3_vl", video_token="<|video_pad|>")

    videos = [video_path]

    # fast path: metadata-only, no frame decoding
    fast_mm_inputs = qwen3_vl_plugin._get_mm_token_metadata([], videos, [], processor)
    assert fast_mm_inputs is not None, "_get_mm_token_metadata should succeed for a real video file"

    full_mm_inputs = qwen3_vl_plugin._get_mm_inputs([], videos, [], processor)

    # video_grid_thw must be identical between the two paths
    assert torch.equal(fast_mm_inputs["video_grid_thw"], full_mm_inputs["video_grid_thw"]), (
        f"video_grid_thw mismatch between fast path and full path: "
        f"fast={fast_mm_inputs['video_grid_thw']}, full={full_mm_inputs['video_grid_thw']}"
    )
    result = qwen3_vl_plugin.process_messages(VIDEO_MESSAGES, [], videos, [], processor)
    # This demo video duration is 9.72s, with video_fps=2, we extract 19 frames
    # 19 + 1 => temperoal compress => 10 video_sequence
    assert result[0]["content"].count("<|vision_start|>") == 10, (
        f"Expected 10 video tokens, got {result[0]['content'].count('<|vision_start|>')}"
    )


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_transformers_version_greater_than("4.47.0"), reason="Requires transformers>=4.47.0")
def test_video_llava_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="LanguageBind/Video-LLaVA-7B-hf")
    video_llava_plugin = get_mm_plugin(name="video_llava", image_token="<image>", video_token="<video>")
    check_inputs = {"plugin": video_llava_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.runs_on(["cpu", "mps"])
def test_lfm2_vl_plugin():
    """Test LFM2.5-VL plugin instantiation."""
    # Test plugin can be instantiated with correct tokens
    lfm2_vl_plugin = get_mm_plugin(name="lfm2_vl", image_token="<image>")
    assert lfm2_vl_plugin is not None
    assert lfm2_vl_plugin.image_token == "<image>"
    assert lfm2_vl_plugin.video_token is None
    assert lfm2_vl_plugin.audio_token is None
    assert lfm2_vl_plugin.__class__.__name__ == "LFMVLPlugin"
