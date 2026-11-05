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
"""Regression test for issue #9213.

The OpenAI-compatible API used to map ``repetition_penalty=request.presence_penalty``.
``presence_penalty`` (OpenAI default 0.0) and ``repetition_penalty`` (default 1.0) are
different controls, and no chat engine consumes ``presence_penalty`` at all, so the
misrouting silently changed decoding and made ``repetition_penalty`` unreachable from
the API (and could feed 0.0 into an engine that requires > 0). This test pins that the
request's ``repetition_penalty`` field is what flows to the engine.
"""

import asyncio

from llamafactory.api.chat import create_chat_completion_response
from llamafactory.api.protocol import ChatCompletionRequest, ChatMessage, Role
from llamafactory.chat.base_engine import Response


class _RecordingChatModel:
    """Minimal ChatModel stand-in that records the kwargs achat receives."""

    def __init__(self):
        self.captured_kwargs = None

    async def achat(self, messages, system=None, tools=None, images=None, videos=None, audios=None, **input_kwargs):
        self.captured_kwargs = input_kwargs
        return [Response(response_text="hi", response_length=1, prompt_length=1, finish_reason="stop")]


def _request(**overrides):
    base = {"model": "test", "messages": [ChatMessage(role=Role.USER, content="hello")]}
    base.update(overrides)
    return ChatCompletionRequest(**base)


def _run(request, model):
    return asyncio.run(create_chat_completion_response(request, model))


def test_repetition_penalty_field_flows_to_engine():
    model = _RecordingChatModel()
    _run(_request(repetition_penalty=1.3, presence_penalty=0.5), model)
    # repetition_penalty must come from the repetition_penalty field, not presence_penalty.
    assert model.captured_kwargs["repetition_penalty"] == 1.3


def test_presence_penalty_no_longer_leaks_into_repetition_penalty():
    model = _RecordingChatModel()
    # Only presence_penalty is set; it must NOT become repetition_penalty.
    _run(_request(presence_penalty=0.5), model)
    assert model.captured_kwargs["repetition_penalty"] is None


def test_repetition_penalty_defaults_to_none():
    model = _RecordingChatModel()
    _run(_request(), model)
    # None lets each engine fall back to its configured default (1.0).
    assert model.captured_kwargs["repetition_penalty"] is None
