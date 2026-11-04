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

"""Offline contract tests: no checkpoint weights or network requests are needed."""

import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, PreTrainedTokenizerFast

from llamafactory.data import TEMPLATES, MultiModalDataCollatorForSeq2Seq, get_template_and_fix_tokenizer
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.extras.constants import IGNORE_INDEX, SUPPORTED_MODELS
from llamafactory.hparams import DataArguments, FinetuningArguments
from llamafactory.model.model_utils.visual import patch_target_modules


@pytest.fixture
def processor():
    tokens = [
        "<unk>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<image>",
        "</image>",
        "<slice>",
        "</slice>",
        "<image_id>",
        "</image_id>",
        "<think>",
        "</think>",
        "user",
        "assistant",
        "Read",
        "ABC",
        "DEF",
        "0",
        "1",
    ]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=tokens[2:13],
    )

    def preprocess(images, **kwargs):
        # Each image has a global view and two slices, also when batched.
        count = len(images)
        return {
            "pixel_values": torch.zeros(1, 3, 14, 16 * 14 * count * 3),
            "target_sizes": torch.tensor([[4, 4]] * count * 3, dtype=torch.int32),
            "grids": torch.tensor([[1, 2]] * count),
            "num_patches_per_image": torch.tensor([3] * count),
        }

    return SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=Mock(side_effect=preprocess, downsample_mode="4x"),
        image_token="<|image_pad|>",
        image_start_token="<image>",
        image_end_token="</image>",
        slice_start_token="<slice>",
        slice_end_token="</slice>",
        image_id_start_token="<image_id>",
        image_id_end_token="</image_id>",
        default_use_image_id=True,
        slice_mode=True,
    )


def test_placeholder_and_batch_alignment(processor, monkeypatch):
    monkeypatch.delenv("DOWNSAMPLE_MODE", raising=False)
    plugin = TEMPLATES["minicpm_ocr"].mm_plugin
    images = [Image.new("RGB", (56, 56))] * 2
    messages = [{"role": "user", "content": "<image>Read <image>"}]
    original = deepcopy(messages)
    result = plugin.process_messages(messages, images, [], [], processor)
    assert messages == original
    assert result[0]["content"].count("<|image_pad|>") == 24
    assert "<image_id>0</image_id>" in result[0]["content"]
    assert "<image_id>1</image_id>" in result[0]["content"]
    assert result[0]["content"].count("<slice>") == 4
    batch = plugin.get_mm_inputs(images, [], [], [1, 1], [0, 0], [0, 0], [[1], [1]], processor)
    assert {"pixel_values", "target_sizes", "downsample_mode"} <= set(batch)
    assert batch["target_sizes"].shape == (6, 2)
    assert batch["downsample_mode"] == "4x"


def test_invalid_media(processor):
    plugin = TEMPLATES["minicpm_ocr"].mm_plugin
    with pytest.raises(ValueError):
        plugin.process_messages([{"role": "user", "content": "<image>"}], [], [], [], processor)
    with pytest.raises(ValueError):
        plugin.get_mm_inputs([], [[Image.new("RGB", (56, 56))]], [], [0], [1], [0], [[1]], processor)


def test_template_labels_and_padded_batch(processor, monkeypatch):
    monkeypatch.delenv("DOWNSAMPLE_MODE", raising=False)
    template = get_template_and_fix_tokenizer(
        processor.tokenizer, DataArguments(template="minicpm_ocr", enable_thinking=False)
    )
    features = []
    for answer in ["ABC", "ABC DEF"]:
        images = [Image.new("RGB", (56, 56))]
        messages = template.mm_plugin.process_messages(
            [{"role": "user", "content": "<image>Read"}, {"role": "assistant", "content": answer}],
            images,
            [],
            [],
            processor,
        )
        prompt, response = template.encode_oneturn(processor.tokenizer, messages)
        thought_ids = template.get_thought_word_ids(processor.tokenizer)
        assert prompt[-len(thought_ids) :] == thought_ids
        assert "<think>" not in processor.tokenizer.decode(response)
        features.append(
            {
                "input_ids": prompt + response,
                "attention_mask": [1] * (len(prompt) + len(response)),
                "labels": [IGNORE_INDEX] * len(prompt) + response,
                "images": images,
            }
        )
    batch = MultiModalDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        template=template,
        processor=processor,
        pad_to_multiple_of=8,
    )(features)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)
    image_ids = batch["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert torch.all(batch["labels"][image_ids] == IGNORE_INDEX)


def test_lora_freezes_visual_modules():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="minicpm_ocr")
    model.model = torch.nn.Module()
    for name in ["vision_tower", "merger", "language_model"]:
        module = torch.nn.Module()
        module.q_proj = torch.nn.Linear(2, 2)
        setattr(model.model, name, module)
    args = FinetuningArguments(freeze_vision_tower=True, freeze_multi_modal_projector=True)
    assert patch_target_modules(model, args, ["q_proj"]) == ["model.language_model.q_proj"]
    assert "MiniCPM-OCR-1.0" in SUPPORTED_MODELS


def test_native_processor_equivalence(monkeypatch, tmp_path):
    path = os.getenv("MINICPM_OCR_MODEL_PATH")
    if not path:
        pytest.skip("Set MINICPM_OCR_MODEL_PATH to test local processor assets without loading weights.")
    monkeypatch.delenv("DOWNSAMPLE_MODE", raising=False)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    plugin = TEMPLATES["minicpm_ocr"].mm_plugin
    images = [Image.new("RGB", (224, 112)), Image.new("RGB", (1200, 600))]
    image_path = tmp_path / "image.png"
    images[0].save(image_path)
    plugin_images = [str(image_path), images[1]]
    messages = [{"role": "user", "content": "<image>Read <image>"}]
    expanded = plugin.process_messages(messages, plugin_images, [], [], processor)[0]["content"]
    reference = processor(
        images=images,
        text="<|image_pad|>Read <|image_pad|>",
        add_special_tokens=False,
        return_tensors="pt",
    )
    assert processor.tokenizer.encode(expanded, add_special_tokens=False) == reference["input_ids"][0].tolist()
    batch = plugin.get_mm_inputs(plugin_images, [], [], [2], [0], [0], [[1]], processor)
    assert batch["downsample_mode"] == "4x"
    assert torch.equal(batch["target_sizes"], reference["target_sizes"])
    assert torch.equal(batch["pixel_values"], reference["pixel_values"])


def test_legacy_minicpm_returns_inputs():
    # The fork dropped this return relative to upstream; guard the restored behavior.
    plugin = get_mm_plugin("minicpm_v", image_token="<image>")
    processor = SimpleNamespace(
        image_processor=Mock(),
        tokenizer=SimpleNamespace(im_start_id=2, im_end_id=3, slice_start_id=4, slice_end_id=5),
    )
    plugin._get_mm_inputs = Mock(return_value={})
    batch = plugin.get_mm_inputs([], [], [], [0], [0], [0], [[0, 1]], processor)
    assert isinstance(batch, dict)
    assert batch["image_bound"][0].shape == (0, 2)
