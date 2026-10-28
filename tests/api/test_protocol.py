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

from llamafactory.api.protocol import ChatCompletionRequest, ChatMessage


TOOL_CALL_ID = "call_fcc8c62d539b47df94e042e511b7b9af"
TOOL_CALL = {
    "type": "function",
    "id": TOOL_CALL_ID,
    "function": {"name": "dark_style", "arguments": "{}"},
}


def test_tool_reply_preserves_call_id():
    message = {"role": "tool", "content": "切换夜间模式成功", "tool_call_id": TOOL_CALL_ID}

    assert ChatMessage.model_validate(message).model_dump(exclude_unset=True) == message


def test_chat_request_round_trip_preserves_tool_call_identity():
    payload = {
        "model": "function",
        "messages": [
            {"role": "user", "content": "切换夜间模式"},
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "content": "切换夜间模式成功", "tool_call_id": TOOL_CALL_ID},
        ],
    }
    request = ChatCompletionRequest.model_validate(payload)

    assert request.model_dump(exclude_unset=True) == payload
    assert ChatCompletionRequest.model_validate_json(request.model_dump_json(exclude_unset=True)) == request


def test_chat_request_schema_exposes_tool_reply_call_id():
    schema = ChatCompletionRequest.model_json_schema()
    message_schema = schema["$defs"]["ChatMessage"]

    assert schema["properties"]["messages"]["items"]["$ref"] == "#/$defs/ChatMessage"
    assert {"type": "string"} in message_schema["properties"]["tool_call_id"]["anyOf"]
    assert "tool_call_id" not in message_schema["required"]


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_messages_without_tool_call_id_remain_accepted(role):
    message = {"role": role, "content": "text"}

    assert ChatMessage.model_validate(message).model_dump(exclude_unset=True) == message


def test_explicit_null_tool_call_id():
    message = {"role": "tool", "content": "text", "tool_call_id": None}

    assert ChatMessage.model_validate(message).model_dump(exclude_unset=True) == message


@pytest.mark.parametrize("tool_calls", [None, [], [TOOL_CALL]])
def test_existing_tool_calls_remain_unchanged(tool_calls):
    message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

    assert ChatMessage.model_validate(message).model_dump(exclude_unset=True) == message
