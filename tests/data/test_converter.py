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

import json

import pytest
from datasets import IterableDataset
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import Role
from llamafactory.data.converter import align_dataset, get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
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


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
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


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("content_field", [{}, {"content": None}, {"content": ""}, {"content": "Let me check."}])
def test_openai_converter_accepts_tool_calls_without_content(content_field, tmp_path):
    dataset_attr = DatasetAttr("hf_hub", "tool-call-dataset")
    dataset_attr.join(
        {
            "formatting": "openai",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "tool",
                "function_tag": "function",
                "system_tag": "system",
            },
        }
    )
    function = {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
    example = {
        "messages": [
            {"role": "user", "content": "What is the weather in Beijing?"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": function}],
                **content_field,
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25 C."},
            {"role": "assistant", "content": "It is sunny and 25 C in Beijing."},
        ]
    }
    dataset = IterableDataset.from_generator(lambda: iter([example]))
    data_args = DataArguments(streaming=True)
    training_args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    output = next(iter(align_dataset(dataset, dataset_attr, data_args, training_args)))

    assert output["_prompt"][0] == {"role": Role.USER.value, "content": "What is the weather in Beijing?"}
    assert output["_prompt"][1]["role"] == Role.FUNCTION.value
    assert json.loads(output["_prompt"][1]["content"]) == [function]
    assert output["_prompt"][2] == {"role": Role.OBSERVATION.value, "content": "Sunny, 25 C."}
    assert output["_response"] == [{"role": Role.ASSISTANT.value, "content": "It is sunny and 25 C in Beijing."}]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("tool_calls", [None, []])
@pytest.mark.parametrize("role", ["gpt", "observation"])
def test_openai_converter_requires_content_without_tool_calls(tool_calls, role):
    dataset_attr = DatasetAttr("hf_hub", "tool-call-dataset")
    example = {
        "conversations": [
            {"from": "human", "value": "What is the weather?"},
            {"from": role, "tool_calls": tool_calls},
        ]
    }
    dataset_converter = get_dataset_converter("openai", dataset_attr, DataArguments())

    with pytest.raises(KeyError, match="value"):
        dataset_converter(example)
