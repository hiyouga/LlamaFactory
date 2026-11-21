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

from copy import deepcopy

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import HfArgumentParser, PreTrainedTokenizerFast

from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


def make_examples(*conversations):
    return {
        "_prompt": [messages[:-1] for messages in conversations],
        "_response": [messages[-1:] for messages in conversations],
        "_system": [""] * len(conversations),
        "_tools": [""] * len(conversations),
        "_images": [None] * len(conversations),
        "_videos": [None] * len(conversations),
        "_audios": [None] * len(conversations),
    }


def conversation(*lengths):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": " ".join(["q" if i % 2 == 0 else "a"] * n)}
        for i, n in enumerate(lengths)
    ]


@pytest.fixture(params=["unpacked", "packed", "neat"])
def processor(request):
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2, "q": 3, "a": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    packing = request.param != "unpacked"
    data_args = DataArguments(cutoff_len=9 if packing else 8, packing=packing, neat_packing=request.param == "neat")
    data_args.drop_long_samples = True
    processor_class = PackedSupervisedDatasetProcessor if packing else SupervisedDatasetProcessor
    return processor_class(deepcopy(TEMPLATES["empty"]), tokenizer, None, data_args)


@pytest.mark.parametrize("efficient_eos", [False, True])
def test_drop_long_samples_keeps_complete_examples(processor, efficient_eos):
    processor.template.efficient_eos = efficient_eos
    short = conversation(2, 2)
    boundary = conversation(4 - int(efficient_eos), 4)
    overlong = conversation(5 - int(efficient_eos), 4)
    expected = processor.preprocess_dataset(make_examples(short, boundary))
    actual = processor.preprocess_dataset(make_examples(overlong, short, boundary))

    assert actual == expected
    assert len(actual["input_ids"]) == 2
    assert all(len(ids) == len(labels) for ids, labels in zip(actual["input_ids"], actual["labels"]))
    assert any(
        ids[:8] == [3] * (4 - int(efficient_eos)) + [4] * 4 + [2] * int(efficient_eos) for ids in actual["input_ids"]
    )


@pytest.mark.parametrize("mask_history", [False, True])
def test_drop_long_samples_counts_all_turns(processor, mask_history):
    processor.data_args.mask_history = mask_history
    short = conversation(1, 1, 2, 2)
    overlong = conversation(3, 3, 1, 2)
    expected = processor.preprocess_dataset(make_examples(short))
    actual = processor.preprocess_dataset(make_examples(overlong, short))

    assert actual == expected
    assert actual["input_ids"][0][:6] == [3, 4, 3, 3, 4, 4]
    assert actual["labels"][0][:6] == [
        IGNORE_INDEX,
        IGNORE_INDEX if mask_history else 4,
        IGNORE_INDEX,
        IGNORE_INDEX,
        4,
        4,
    ]


def test_drop_long_samples_disabled_preserves_truncation(processor):
    processor.data_args.drop_long_samples = False
    actual = processor.preprocess_dataset(make_examples(conversation(5, 4)))

    assert len(actual["input_ids"]) == 1
    assert actual["input_ids"][0][:8] == [3] * 5 + [4] * 3
    assert actual["labels"][0][:8] == [IGNORE_INDEX] * 5 + [4] * 3


def test_drop_long_samples_counts_multimodal_prefix(processor, monkeypatch):
    plugin = processor.template.mm_plugin
    monkeypatch.setattr(plugin, "process_messages", lambda messages, *args: messages)
    monkeypatch.setattr(plugin, "process_token_ids", lambda *args: ([10, 11], [IGNORE_INDEX, IGNORE_INDEX]))
    examples = make_examples(conversation(3, 4), conversation(2, 4))
    examples["_images"] = [["dropped.png"], ["kept.png"]]
    actual = processor.preprocess_dataset(examples)

    assert len(actual["input_ids"]) == 1
    assert actual["input_ids"][0][:8] == [10, 11, 3, 3, 4, 4, 4, 4]
    assert actual["images"] == [["kept.png"]]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("all_dropped", [False, True])
def test_drop_long_samples_handles_empty_batches(processor, streaming, all_dropped):
    middle = conversation(5, 5) if all_dropped else conversation(2, 2)
    dataset = Dataset.from_dict(make_examples(conversation(5, 4), middle, conversation(6, 6)))
    columns = dataset.column_names
    if streaming:
        dataset = dataset.to_iterable_dataset()
    mapped = dataset.map(processor.preprocess_dataset, batched=True, batch_size=1, remove_columns=columns)
    rows = list(mapped)

    if all_dropped:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["input_ids"][:4] == [3, 3, 4, 4]
        assert rows[0]["labels"][:4] == [IGNORE_INDEX, IGNORE_INDEX, 4, 4]


def test_drop_long_samples_argument():
    parser = HfArgumentParser(DataArguments)
    assert parser.parse_dict({})[0].drop_long_samples is False
    assert parser.parse_dict({"drop_long_samples": True})[0].drop_long_samples is True
