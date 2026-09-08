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

import pytest
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
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


@pytest.mark.runs_on(["cpu", "mps"])
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


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("template_name", ["bailing", "hy_dense_7b", "llama2"])
@pytest.mark.parametrize("num_turns", [1, 3])
@pytest.mark.parametrize("mask_history", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_supervised_eos_history_labels(template_name: str, num_turns: int, mask_history: bool, packed: bool):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({char: index + 3 for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>"
    )
    template = TEMPLATES[template_name]
    messages = []
    responses = []
    for turn in range(num_turns):
        pair = [{"role": "user", "content": f"Question {turn}"}, {"role": "assistant", "content": f"Answer {turn}"}]
        messages.extend(pair)
        _, target_ids = template.encode_oneturn(tokenizer, pair)
        responses.append(target_ids + ([tokenizer.eos_token_id] if template.efficient_eos else []))

    data_args = DataArguments(cutoff_len=512, mask_history=mask_history, packing=packed, neat_packing=packed)
    processor_cls = PackedSupervisedDatasetProcessor if packed else SupervisedDatasetProcessor
    processor = processor_cls(template, tokenizer, None, data_args)
    examples = {
        "_prompt": [messages[:-1]],
        "_response": [messages[-1:]],
        "_system": [None],
        "_tools": [None],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
    }
    result = processor.preprocess_dataset(examples)
    labels = result["labels"][0]
    assert len(labels) == len(result["input_ids"][0])
    # Compare exactly the target tokens consumed by shifted causal-LM loss.
    trained_tokens = [token for token in labels[1:] if token != IGNORE_INDEX]
    expected_responses = responses[-1:] if mask_history else responses
    assert trained_tokens == [token for response in expected_responses for token in response]


@pytest.mark.runs_on(["cpu", "mps"])
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


@pytest.mark.runs_on(["cpu", "mps"])
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


@pytest.mark.runs_on(["cpu", "mps"])
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
