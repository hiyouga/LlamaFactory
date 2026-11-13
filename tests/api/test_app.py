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
from fastapi.testclient import TestClient

from llamafactory.api.app import create_app
from llamafactory.extras.constants import EngineName


TOOLS = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]


class FakeEngine:
    name = EngineName.HF
    can_generate = True


class FakeChatModel:
    engine = FakeEngine()

    async def astream_chat(self, *args, **kwargs):
        yield "Hello"


@pytest.mark.parametrize(
    "extra_body, detail",
    [
        ({"tools": TOOLS}, "Cannot stream function calls."),
        ({"n": 2}, "Cannot stream multiple responses."),
        ({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}, "Only supports"),
    ],
)
def test_stream_invalid_request_returns_400(extra_body: dict, detail: str):
    client = TestClient(create_app(FakeChatModel()), raise_server_exceptions=False)
    body = {"model": "test", "stream": True, "messages": [{"role": "user", "content": "Hi"}], **extra_body}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["detail"].startswith(detail)


def test_stream_valid_request():
    client = TestClient(create_app(FakeChatModel()))
    body = {"model": "test", "stream": True, "messages": [{"role": "user", "content": "Hi"}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert '"content": "Hello"' in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
