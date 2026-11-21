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
import random
from dataclasses import replace
from string import printable

import pytest
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llamafactory.data.formatter import StringFormatter
from llamafactory.data.processor import SupervisedDatasetProcessor
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import DataArguments
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.fixture
def role_tokenizer():
    special = ["<|im_start|>", "<|im_end|>", "<reserved_102>", "<reserved_103>"]
    vocab = {token: i for i, token in enumerate([*printable, "问", "答", "：", "x\n", *special])}
    backend = Tokenizer(BPE(vocab, [("x", "\n")]))
    backend.add_special_tokens(special)
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize(
    "slots",
    [
        TEMPLATES["qwen"].format_user.slots,
        ["[Round {{idx}}]\n\n问：{{content}}\n\n答："],
        [{"token": "<reserved_102>"}, "{{content}}", {"token": "<reserved_103>"}],
    ],
    ids=["qwen", "chatglm2-format", "baichuan-format"],
)
@pytest.mark.parametrize("mask_history", [False, True])
def test_truncated_user_prompt_retains_role_markers(role_tokenizer, slots, mask_history):
    template = replace(TEMPLATES["qwen"], format_user=StringFormatter(slots=slots))
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "two"},
    ]
    short_messages = [*messages[:2], {"role": "user", "content": "xxxxx"}, messages[-1]]
    expected = template.encode_multiturn(role_tokenizer, short_messages, system="sys")
    if mask_history:
        expected = expected[-1:]
    ids = [token for source, target in expected for token in source + target]
    labels = [token for source, target in expected for token in [IGNORE_INDEX] * len(source) + target]
    processor = SupervisedDatasetProcessor(
        template, role_tokenizer, None, DataArguments(cutoff_len=len(ids), mask_history=mask_history)
    )
    result = processor._encode_data_example(messages[:-1], messages[-1:], "sys", None, [], [], [])
    assert result == (ids, labels)


@pytest.mark.parametrize("budget_delta", [-1, 0, 1, 2])
def test_truncated_user_prompt_budget(role_tokenizer, budget_delta):
    template = TEMPLATES["qwen"]
    messages = [{"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "yes"}]
    source, target = template.encode_oneturn(role_tokenizer, messages, system="keep this")
    empty, _ = template.encode_oneturn(
        role_tokenizer, [dict(messages[0], content=""), messages[1]], system="keep this"
    )
    budget = len(empty) + budget_delta
    processor = SupervisedDatasetProcessor(
        template, role_tokenizer, None, DataArguments(cutoff_len=budget + len(target), train_on_prompt=True)
    )
    ids, labels = processor._encode_data_example(messages[:-1], messages[-1:], "keep this", None, [], [], [])
    if budget_delta < 0:
        assert ids == source[:budget] + target
    else:
        expected, _ = template.encode_oneturn(
            role_tokenizer, [dict(messages[0], content="x" * budget_delta), messages[1]], system="keep this"
        )
        assert ids == expected + target
    assert labels == ids


def test_truncated_user_prompt_preserves_multibyte_character_boundaries():
    backend = Tokenizer(BPE({byte: i for i, byte in enumerate(ByteLevel.alphabet())}, []))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    backend.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    template = TEMPLATES["qwen"]
    messages = [{"role": "user", "content": "🌙" * 20}, {"role": "assistant", "content": "yes"}]
    short_messages = [dict(messages[0], content="🌙"), messages[1]]
    expected, target = template.encode_oneturn(tokenizer, short_messages, system="sys")
    processor = SupervisedDatasetProcessor(
        template, tokenizer, None, DataArguments(cutoff_len=len(expected) + len(target) + 1)
    )
    ids, labels = processor._encode_data_example(messages[:1], messages[1:], "sys", None, [], [], [])
    assert ids == expected + target
    assert labels == [IGNORE_INDEX] * len(expected) + target


def test_truncated_user_prompt_with_reasoning_template(role_tokenizer):
    template = replace(TEMPLATES["qwen3"], enable_thinking=False)
    messages = [{"role": "user", "content": "x" * 100}, {"role": "assistant", "content": "two"}]
    [(source, target)] = template.encode_multiturn(role_tokenizer, messages, system="sys")
    budget = len(source) - 30
    processor = SupervisedDatasetProcessor(
        template, role_tokenizer, None, DataArguments(cutoff_len=budget + len(target))
    )
    ids, labels = processor._encode_data_example(messages[:1], messages[1:], "sys", None, [], [], [])
    assert ids == source[:budget] + target
    assert labels == [IGNORE_INDEX] * budget + target


def test_truncated_user_prompt_when_content_shortens_encoding():
    backend = Tokenizer(
        BPE({s: i for i, s in enumerate(["A", "B", "C", "x", "Y", "Ax", "AxB"])}, [("A", "x"), ("Ax", "B")])
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    template = replace(
        TEMPLATES["qwen"],
        default_system="",
        format_user=StringFormatter(slots=["A{{content}}BC"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
    )
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "YY"}]
    source, target = template.encode_oneturn(tokenizer, messages)
    empty, _ = template.encode_oneturn(tokenizer, [dict(messages[0], content=""), messages[1]])
    assert len(source) == 2 < len(empty)
    processor = SupervisedDatasetProcessor(template, tokenizer, None, DataArguments(cutoff_len=2))
    ids, labels = processor._encode_data_example(messages[:1], messages[1:], None, None, [], [], [])
    assert ids == source[:1] + target[:1]
    assert labels == [IGNORE_INDEX] + target[:1]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [16])
def test_supervised_single_turn(num_samples: int):
    train_dataset = load_dataset_module(dataset_dir="ONLINE", dataset=TINY_DATA, **TRAIN_ARGS)["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(TINY_DATA, split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        prompt = original_data["instruction"][index]
        if original_data["input"][index]:
            prompt += "\n" + original_data["input"][index]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": original_data["output"][index]},
        ]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        ref_prompt_ids = ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]
            ref_prompt_ids = ref_prompt_ids["input_ids"]

        prompt_len = len(ref_prompt_ids)
        ref_label_ids = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_label_ids


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [8])
def test_supervised_multi_turn(num_samples: int):
    train_dataset = load_dataset_module(dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", **TRAIN_ARGS)[
        "train_dataset"
    ]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_input_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]

        # cannot test the label ids in multi-turn case
        assert train_dataset["input_ids"][index] == ref_input_ids


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [4])
def test_supervised_train_on_prompt(num_samples: int):
    train_dataset = load_dataset_module(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", train_on_prompt=True, **TRAIN_ARGS
    )["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_input_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]

        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_input_ids


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [4])
def test_supervised_mask_history(num_samples: int):
    train_dataset = load_dataset_module(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", mask_history=True, **TRAIN_ARGS
    )["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        messages = original_data["messages"][index]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        ref_prompt_ids = ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]
            ref_prompt_ids = ref_prompt_ids["input_ids"]

        prompt_len = len(ref_prompt_ids)
        ref_label_ids = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_label_ids
