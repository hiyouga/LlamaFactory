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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments
from llamafactory.train.pt import workflow


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "a": 3, "b": 4, "c": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    )


@pytest.fixture
def make_collator(monkeypatch):
    """Exercise the collator selected by the PT workflow without loading model weights."""
    collators = []

    def capture_trainer(**kwargs):
        collators.append(kwargs["data_collator"])

    monkeypatch.setattr(workflow, "get_dataset", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "CustomTrainer", capture_trainer)
    monkeypatch.setattr(workflow, "create_modelcard_and_push", lambda *args, **kwargs: None)

    def make(tokenizer):
        monkeypatch.setattr(workflow, "load_tokenizer", lambda _: {"tokenizer": tokenizer, "processor": None})
        workflow.run_pt(None, DataArguments(template="default"), SimpleNamespace(do_train=False, do_eval=False), None)
        return collators[-1]

    return make


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("separate_pad_token", [False, True])
@pytest.mark.parametrize("pad_to_multiple_of", [None, 8])
def test_pretrain_padding(tokenizer, make_collator, padding_side, separate_pad_token, pad_to_multiple_of):
    tokenizer.padding_side = padding_side
    if separate_pad_token:
        tokenizer.pad_token = "[PAD]"
    collator = make_collator(tokenizer)
    collator.pad_to_multiple_of = pad_to_multiple_of
    features = [
        {"input_ids": [3, 1, 4, 1], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 1], "attention_mask": [1, 1]},
    ]
    original_features = deepcopy(features)

    batch = collator(features)

    for index, feature in enumerate(features):
        valid = batch["attention_mask"][index].bool()
        assert batch["labels"][index][valid].tolist() == feature["input_ids"]
        assert (batch["labels"][index][~valid] == IGNORE_INDEX).all()
    assert (batch["labels"] == tokenizer.eos_token_id).sum().item() == 3
    assert features == original_features
    assert batch["labels"].data_ptr() != batch["input_ids"].data_ptr()


@pytest.mark.parametrize("packing", [False, True])
def test_pretrain_processor_preserves_eos(tokenizer, make_collator, packing):
    data_args = DataArguments(template="default", packing=packing, cutoff_len=6)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = PretrainDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
    processed = processor.preprocess_dataset(
        {"_prompt": [[{"role": "user", "content": text}] for text in ["a b", "c", "a b c", "b"]]}
    )
    features = [{key: values[i] for key, values in processed.items()} for i in range(len(processed["input_ids"]))]

    batch = make_collator(tokenizer)(features)

    eos_positions = (batch["input_ids"] == tokenizer.eos_token_id) & batch["attention_mask"].bool()
    assert eos_positions.any()
    assert torch.all(batch["labels"][eos_positions] == tokenizer.eos_token_id)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)
