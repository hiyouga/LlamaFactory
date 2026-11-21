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

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, ProcessorMixin

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.extras.packages import is_transformers_version_greater_than


@pytest.fixture
def processor():
    if not hasattr(ProcessorMixin, "create_mm_token_type_ids"):
        pytest.skip("Requires a processor with multimodal token type support.")
    processor = SimpleNamespace(image_processor=object(), video_processor=object())
    processor.image_ids, processor.video_ids, processor.audio_ids = [10], [11], []
    processor.create_mm_token_type_ids = ProcessorMixin.create_mm_token_type_ids.__get__(processor)
    return processor


@pytest.fixture
def plugin(monkeypatch):
    plugin = get_mm_plugin("qwen3_vl", image_token="<image>", video_token="<video>")
    monkeypatch.setattr(plugin, "_get_mm_inputs", lambda *args: {"image_grid_thw": torch.tensor([[1, 2, 2]])})
    return plugin


def _collator(plugin, processor, padding_side="right", pad_to_multiple_of=None, model=None):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<unk>": 1, "<image>": 10, "<video>": 11}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        unk_token="<unk>",
        padding_side=padding_side,
    )
    return MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        template=SimpleNamespace(mm_plugin=plugin),
        processor=processor,
        pad_to_multiple_of=pad_to_multiple_of,
    )


@pytest.mark.parametrize(
    "batch_ids, expected",
    [
        ([[1, 10, 2], [10, 2]], [[0, 1, 0], [1, 0]]),
        ([[1, 11, 11, 2], [11, 2]], [[0, 2, 2, 0], [2, 0]]),
        ([[10, 11, 1], [2, 10]], [[1, 2, 0], [0, 1]]),
        ([[1, 2], [3]], [[0, 0], [0]]),
    ],
)
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("pad_to_multiple_of", [None, 8])
def test_qwen3_vl_predict_collator(plugin, processor, batch_ids, expected, padding_side, pad_to_multiple_of):
    # Disable dummy-image injection for the pure-text case; it is tested separately.
    if all(10 not in ids and 11 not in ids for ids in batch_ids):
        plugin.image_token = None
        plugin.video_token = None
    collator = _collator(plugin, processor, padding_side, pad_to_multiple_of)
    features = [
        {
            "input_ids": ids[:],
            "attention_mask": [1] * len(ids),
            "labels": ids[:],
            "images": [object()] if 10 in ids else [],
            "videos": [object()] if 11 in ids else [],
        }
        for ids in batch_ids
    ]
    batch = collator(features)
    assert collator.get_rope_func is None
    assert "position_ids" not in batch
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    length = batch["input_ids"].shape[1]
    padded_expected = []
    for row in expected:
        padding = [0] * (length - len(row))
        padded_expected.append(padding + row if padding_side == "left" else row + padding)
    assert batch["mm_token_type_ids"].tolist() == padded_expected
    assert torch.equal(batch["image_grid_thw"], torch.tensor([[1, 2, 2]]))


def test_qwen3_vl_older_processor(plugin):
    processor = SimpleNamespace(image_processor=object(), video_processor=object())
    batch = _collator(plugin, processor)([{"input_ids": [1, 10], "attention_mask": [1, 1], "images": [object()]}])
    assert "mm_token_type_ids" not in batch


def test_qwen2_vl_unchanged(processor):
    plugin = get_mm_plugin("qwen2_vl", image_token=None)
    batch = _collator(plugin, processor)([{"input_ids": [1, 2], "attention_mask": [1, 1]}])
    assert "mm_token_type_ids" not in batch


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_qwen3_vl_dummy_image(plugin, processor, monkeypatch, padding_side):
    monkeypatch.setattr(plugin, "process_messages", lambda *args: [{"role": "user", "content": "<image>"}])
    batch = _collator(plugin, processor, padding_side)(
        [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}]
    )
    expected = [[1, 0, 0]] if padding_side == "left" else [[0, 0, 1]]
    assert batch["mm_token_type_ids"].tolist() == expected
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape


@pytest.mark.skipif(not is_transformers_version_greater_than("5.8.0"), reason="Requires transformers>=5.8.0")
@pytest.mark.parametrize("training", [False, True])
def test_qwen3_vl_predict_mrope(plugin, processor, training):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).eval()
    collator = _collator(plugin, processor, pad_to_multiple_of=8, model=model if training else None)
    batch = collator([{"input_ids": [5, 10, 6, 1], "attention_mask": [1, 1, 1, 1], "images": [object()]}])
    batch["pixel_values"] = torch.zeros(4, 24)
    assert ("mm_token_type_ids" in batch) is not training
    with torch.no_grad():
        output = model(**batch)
        if not training:
            generated = model.generate(**batch, max_new_tokens=2, do_sample=False, pad_token_id=0)
            assert generated.shape[1] > batch["input_ids"].shape[1]
    assert output.logits.shape == (1, 8, 32)
    assert torch.isfinite(output.logits).all()
    if not training:
        assert model.model.rope_deltas is not None
    else:
        assert "position_ids" in batch
