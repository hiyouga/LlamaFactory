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

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps"])
def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_merges_consecutive_observations():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "human", "value": "Check both symbols."},
            {"from": "function_call", "value": "<tool_call>lookup_a</tool_call>\n<tool_call>lookup_b</tool_call>"},
            {"from": "observation", "value": '{"symbol": "A", "price": 10}'},
            {"from": "observation", "value": '{"symbol": "B", "price": 20}'},
            {"from": "gpt", "value": "A is 10 and B is 20."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [
            {"role": Role.USER.value, "content": "Check both symbols."},
            {"role": Role.FUNCTION.value, "content": "<tool_call>lookup_a</tool_call>\n<tool_call>lookup_b</tool_call>"},
            {
                "role": Role.OBSERVATION.value,
                "content": '{"symbol": "A", "price": 10}\n</tool_response>\n<tool_response>\n{"symbol": "B", "price": 20}',
            },
        ],
        "_response": [{"role": Role.ASSISTANT.value, "content": "A is 10 and B is 20."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }
