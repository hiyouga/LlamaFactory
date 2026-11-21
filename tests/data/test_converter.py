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
def test_openai_converter():
    dataset_attr = DatasetAttr(
        "hf_hub",
        "custom-openai-dataset",
        formatting="openai",
        messages="messages",
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool",
        function_tag="function",
        system_tag="system",
    )
    data_args = DataArguments()
    example = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    dataset_converter = get_dataset_converter("openai", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Hello!"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "Hi there!"}],
        "_system": "detailed thinking off",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_openai_converter_tool_calls_omitted_content():
    dataset_attr = DatasetAttr(
        "hf_hub",
        "custom-openai-dataset",
        formatting="openai",
        messages="messages",
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool",
        function_tag="function",
        system_tag="system",
    )
    data_args = DataArguments()
    # Assistant message has tool_calls but content is omitted entirely (standard OpenAI streaming shape)
    example = {
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny."},
            {"role": "assistant", "content": "It is sunny."},
        ]
    }
    dataset_converter = get_dataset_converter("openai", dataset_attr, data_args)
    res = dataset_converter(example)
    assert len(res["_prompt"]) == 3
    assert res["_prompt"][0] == {"role": Role.USER.value, "content": "What is the weather?"}
    assert res["_prompt"][1] == {
        "role": Role.FUNCTION.value,
        "content": '[{"name": "weather", "arguments": "{}"}]',
    }
    assert res["_prompt"][2] == {"role": Role.OBSERVATION.value, "content": "Sunny."}
    assert res["_response"] == [{"role": Role.ASSISTANT.value, "content": "It is sunny."}]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_openai_converter_missing_content_raises():
    dataset_attr = DatasetAttr(
        "hf_hub",
        "custom-openai-dataset",
        formatting="openai",
        messages="messages",
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool",
        function_tag="function",
        system_tag="system",
    )
    data_args = DataArguments()
    # User message missing content field should still raise KeyError
    example = {
        "messages": [
            {"role": "user"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    dataset_converter = get_dataset_converter("openai", dataset_attr, data_args)
    with pytest.raises(KeyError):
        dataset_converter(example)
