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


def _openai_dataset_attr() -> DatasetAttr:
    dataset_attr = DatasetAttr("hf_hub", "x", formatting="openai")
    dataset_attr.messages = "messages"
    dataset_attr.role_tag = "role"
    dataset_attr.content_tag = "content"
    dataset_attr.user_tag = "user"
    dataset_attr.assistant_tag = "assistant"
    dataset_attr.observation_tag = "tool"
    dataset_attr.function_tag = "function"
    dataset_attr.system_tag = "system"
    return dataset_attr


@pytest.mark.runs_on(["cpu", "mps"])
def test_openai_converter_skips_conversation_ending_on_tool_response():
    converter = get_dataset_converter("openai", _openai_dataset_attr(), DataArguments())
    example = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Sunny, 25C"},
        ]
    }
    result = converter(example)
    # The trailing tool response makes this an incomplete tool cycle; it must be skipped,
    # not silently truncated into [user] -> [tool_call] (which dropped the tool response).
    assert result["_prompt"] == []
    assert result["_response"] == []


@pytest.mark.runs_on(["cpu", "mps"])
def test_openai_converter_keeps_completed_tool_cycle():
    converter = get_dataset_converter("openai", _openai_dataset_attr(), DataArguments())
    example = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Sunny, 25C"},
            {"role": "assistant", "content": "It is sunny, 25C."},
        ]
    }
    result = converter(example)
    all_content = str(result["_prompt"]) + str(result["_response"])
    assert "Sunny, 25C" in all_content
    assert result["_response"][-1] == {"role": Role.ASSISTANT.value, "content": "It is sunny, 25C."}
