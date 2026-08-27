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
    """Issue B regression: consecutive observation messages must be merged
    using the `\\n</tool_response>\\n<tool_response>\\n` separator so that the
    final encoded string matches what vLLM produces from the native jinja
    chat_template (which packs multiple tool_responses inside one user turn).
    """
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "human", "value": "What is the weather and stock price?"},
            {"from": "function_call", "value": '{"name": "get_weather", "arguments": {"city": "BJ"}}'},
            {"from": "observation", "value": '{"temperature": "25C"}'},
            {"from": "observation", "value": '{"price": 100}'},
            {"from": "observation", "value": '{"volume": 5000}'},
            {"from": "gpt", "value": "Weather is 25C and stock is at 100 with volume 5000."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    out = dataset_converter(example)
    # Three consecutive observations should be merged into one observation message.
    assert out["_prompt"] == [
        {"role": Role.USER.value, "content": "What is the weather and stock price?"},
        {"role": Role.FUNCTION.value, "content": '{"name": "get_weather", "arguments": {"city": "BJ"}}'},
        {
            "role": Role.OBSERVATION.value,
            "content": (
                '{"temperature": "25C"}'
                "\n</tool_response>\n<tool_response>\n"
                '{"price": 100}'
                "\n</tool_response>\n<tool_response>\n"
                '{"volume": 5000}'
            ),
        },
    ]
    assert out["_response"] == [
        {"role": Role.ASSISTANT.value, "content": "Weather is 25C and stock is at 100 with volume 5000."}
    ]


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_keeps_single_observation_unchanged():
    """A single observation between function_call and gpt must NOT be wrapped
    with the merge separator (would otherwise inject a stray <tool_response>).
    """
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "human", "value": "Weather?"},
            {"from": "function_call", "value": '{"name": "get_weather", "arguments": {"city": "BJ"}}'},
            {"from": "observation", "value": '{"temperature": "25C"}'},
            {"from": "gpt", "value": "25C."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    out = dataset_converter(example)
    obs_messages = [m for m in out["_prompt"] if m["role"] == Role.OBSERVATION.value]
    assert len(obs_messages) == 1
    assert obs_messages[0]["content"] == '{"temperature": "25C"}'
    assert "</tool_response>" not in obs_messages[0]["content"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_sharegpt_converter_tolerates_none_content():
    """Gemini high-priority comment: tool-call SFT datasets sometimes have
    `value: null` on observation/function_call rows. The converter must coerce
    None to "" instead of raising TypeError on the join.
    """
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "human", "value": "ping"},
            {"from": "function_call", "value": '{"name": "noop", "arguments": {}}'},
            {"from": "observation", "value": None},
            {"from": "observation", "value": '{"ok": true}'},
            {"from": "gpt", "value": "done"},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    # Must not raise.
    out = dataset_converter(example)
    obs_messages = [m for m in out["_prompt"] if m["role"] == Role.OBSERVATION.value]
    assert len(obs_messages) == 1
    # None coerced to "" then merged with the second observation.
    assert obs_messages[0]["content"] == "\n</tool_response>\n<tool_response>\n" + '{"ok": true}'
