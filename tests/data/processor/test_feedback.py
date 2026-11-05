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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory.data.processor.feedback import FeedbackDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "kto",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "kto_en_demo",
    "dataset_dir": "REMOTE:" + DEMO_DATA,
    "template": "llama3",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("num_samples", [16])
def test_feedback_data(num_samples: int):
    train_dataset = load_dataset_module(**TRAIN_ARGS)["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="kto_en_demo", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        messages = original_data["messages"][index]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        ref_prompt_ids = ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]
            ref_prompt_ids = ref_prompt_ids["input_ids"]

        prompt_len = len(ref_prompt_ids)
        ref_labels = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_labels
        assert train_dataset["kto_tags"][index] == original_data["label"][index]


def _make_feedback_processor():
    template = Mock(efficient_eos=False)
    template.mm_plugin.process_messages.side_effect = lambda messages, *args: messages
    template.mm_plugin.process_token_ids.side_effect = lambda ids, labels, *args: (ids, labels)
    template.encode_oneturn.side_effect = lambda tokenizer, messages, *args: (
        [int(messages[0]["content"])],
        [int(messages[-1]["content"])],
    )
    return FeedbackDatasetProcessor(template, None, None, SimpleNamespace(cutoff_len=16))


def _make_feedback_examples(rows):
    return {
        "_prompt": [prompt for prompt, response in rows],
        "_response": [response for prompt, response in rows],
        "_system": [""] * len(rows),
        "_tools": [""] * len(rows),
        "_images": [None] * len(rows),
        "_videos": [None] * len(rows),
        "_audios": [None] * len(rows),
    }


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("invalid_position", [None, 0, 1, 2])
@pytest.mark.parametrize("invalid_response", [[], [{"role": "assistant", "content": "99"}] * 2])
def test_feedback_kl_pairs_exclude_invalid_examples(invalid_position, invalid_response):
    rows = [
        (
            [{"role": "user", "content": "1"}],
            [{"role": "assistant", "content": "11"}, {"role": "assistant", "content": ""}],
        ),
        (
            [{"role": "user", "content": "2"}],
            [{"role": "assistant", "content": ""}, {"role": "assistant", "content": "22"}],
        ),
    ]
    if invalid_position is not None:
        rows.insert(invalid_position, ([], invalid_response))

    result = _make_feedback_processor().preprocess_dataset(_make_feedback_examples(rows))

    assert result["input_ids"] == [[1, 11], [2, 22]]
    assert result["labels"] == [[IGNORE_INDEX, 11], [IGNORE_INDEX, 22]]
    assert result["kl_input_ids"] == [[1, 22], [2, 11]]
    assert result["kl_labels"] == [[IGNORE_INDEX, 22], [IGNORE_INDEX, 11]]
    assert result["kto_tags"] == [True, False]
    assert all(len(values) == 2 for values in result.values())


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("rows", [[], [([], [])]])
def test_feedback_no_valid_examples(rows):
    result = _make_feedback_processor().preprocess_dataset(_make_feedback_examples(rows))

    assert all(not values for values in result.values())
