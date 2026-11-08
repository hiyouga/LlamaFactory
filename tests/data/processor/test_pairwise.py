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
from copy import deepcopy
from types import SimpleNamespace

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
from llamafactory.data.template import TEMPLATES
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "rm",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "dpo_en_demo",
    "dataset_dir": "REMOTE:" + DEMO_DATA,
    "template": "llama3",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


class _CountingTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.calls = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append(text)
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token: str) -> int:
        return 3


def _convert_sharegpt_to_openai(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    role_mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    new_messages = []
    for message in messages:
        new_messages.append({"role": role_mapping[message["from"]], "content": message["value"]})

    return new_messages


def test_pairwise_reuses_prompt_tokenization():
    template = deepcopy(TEMPLATES["llama3"])
    prompt = [{"role": "user", "content": "Shared prompt"}]
    responses = [
        {"role": "assistant", "content": "Chosen response"},
        {"role": "assistant", "content": "Rejected response"},
    ]

    reference_tokenizer = _CountingTokenizer()
    prompt_ids, chosen_ids = template.encode_oneturn(reference_tokenizer, prompt + [responses[0]])
    _, rejected_ids = template.encode_oneturn(reference_tokenizer, prompt + [responses[1]])
    expected = (
        prompt_ids + chosen_ids,
        [IGNORE_INDEX] * len(prompt_ids) + chosen_ids,
        prompt_ids + rejected_ids,
        [IGNORE_INDEX] * len(prompt_ids) + rejected_ids,
    )

    tokenizer = _CountingTokenizer()
    processor = PairwiseDatasetProcessor(template, tokenizer, None, SimpleNamespace(cutoff_len=8192))
    actual = processor._encode_data_example(prompt, responses, None, None, [], [], [])

    assert actual == expected
    assert len(reference_tokenizer.calls) == 4
    assert len(tokenizer.calls) == 3


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [16])
def test_pairwise_data(num_samples: int):
    train_dataset = load_dataset_module(**TRAIN_ARGS)["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="dpo_en_demo", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        chosen_messages = original_data["conversations"][index] + [original_data["chosen"][index]]
        rejected_messages = original_data["conversations"][index] + [original_data["rejected"][index]]
        chosen_messages = _convert_sharegpt_to_openai(chosen_messages)
        rejected_messages = _convert_sharegpt_to_openai(rejected_messages)

        ref_chosen_input_ids = ref_tokenizer.apply_chat_template(chosen_messages)
        ref_chosen_prompt_ids = ref_tokenizer.apply_chat_template(chosen_messages[:-1], add_generation_prompt=True)
        ref_rejected_input_ids = ref_tokenizer.apply_chat_template(rejected_messages)
        ref_rejected_prompt_ids = ref_tokenizer.apply_chat_template(rejected_messages[:-1], add_generation_prompt=True)

        if is_transformers_version_greater_than("5.0.0"):
            ref_chosen_input_ids = ref_chosen_input_ids["input_ids"]
            ref_rejected_input_ids = ref_rejected_input_ids["input_ids"]
            ref_chosen_prompt_ids = ref_chosen_prompt_ids["input_ids"]
            ref_rejected_prompt_ids = ref_rejected_prompt_ids["input_ids"]

        chosen_prompt_len = len(ref_chosen_prompt_ids)
        rejected_prompt_len = len(ref_rejected_prompt_ids)
        ref_chosen_labels = [IGNORE_INDEX] * chosen_prompt_len + ref_chosen_input_ids[chosen_prompt_len:]
        ref_rejected_labels = [IGNORE_INDEX] * rejected_prompt_len + ref_rejected_input_ids[rejected_prompt_len:]
        assert train_dataset["chosen_input_ids"][index] == ref_chosen_input_ids
        assert train_dataset["chosen_labels"][index] == ref_chosen_labels
        assert train_dataset["rejected_input_ids"][index] == ref_rejected_input_ids
        assert train_dataset["rejected_labels"][index] == ref_rejected_labels
