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

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from llamafactory.chat import sglang_engine


MESSAGES = [{"role": "user", "content": "Hello"}]


def event(text):
    return (
        "data: "
        + json.dumps(
            {
                "text": text,
                "meta_info": {"completion_tokens": 2, "prompt_tokens": 2, "finish_reason": "stop"},
            },
            ensure_ascii=False,
        )
        + "\n\n"
    ).encode()


class ControlledStream(httpx.AsyncByteStream):
    def __init__(self, chunks, pause_at=None):
        self.chunks = chunks
        self.pause_at = pause_at
        self.read_started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False

    async def __aiter__(self):
        for index, chunk in enumerate(self.chunks):
            if index == self.pause_at:
                self.read_started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

            if isinstance(chunk, Exception):
                raise chunk

            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def engine():
    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.base_url = "http://sglang.test"
    engine.lora_request = False
    engine.tokenizer = object()
    engine.processor = None
    engine.template = SimpleNamespace(
        mm_plugin=SimpleNamespace(process_messages=lambda messages, *args: messages),
        encode_oneturn=lambda *args: ([1, 2], []),
        get_stop_token_ids=lambda tokenizer: [2],
    )
    engine.generating_args = {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 16,
        "repetition_penalty": 1.0,
        "skip_special_tokens": True,
    }
    return engine


@pytest.fixture
def install_transport(monkeypatch):
    client_type = httpx.AsyncClient

    def install(handler):
        clients = []

        def create_client(*args, **kwargs):
            client = client_type(*args, transport=httpx.MockTransport(handler), **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(sglang_engine.httpx, "AsyncClient", create_client)
        return clients

    return install


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
def test_stream_chat_preserves_deltas_and_request(engine, install_transport, lora, line_ending):
    async def run():
        # These Unicode characters are valid inside JSON strings, not SSE line endings.
        text = "你\u0085\u2028\u2029"
        first = event(text).replace(b"\n", line_ending)
        stream = ControlledStream(
            [b": keep-alive\n\n"]
            + [first[index : index + 1] for index in range(len(first))]
            + [event(text + "好"), b"data: [DONE]\n\n"]
        )
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, stream=stream)

        clients = install_transport(handle)
        engine.lora_request = lora
        deltas = [delta async for delta in engine.stream_chat(MESSAGES, seed=42, max_new_tokens=8, stop=["END"])]

        assert deltas == [text, "好"]
        assert len(requests) == 1
        assert requests[0].method == "POST"
        assert requests[0].url == "http://sglang.test/generate"
        # Inference may pause for longer than HTTPX's default five-second timeout.
        assert all(timeout is None for timeout in requests[0].extensions["timeout"].values())
        payload = json.loads(requests[0].content)
        assert payload["input_ids"] == [1, 2]
        assert payload["stream"] is True
        assert payload["sampling_params"] == {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "stop": ["END"],
            "stop_token_ids": [2],
            "max_new_tokens": 8,
            "repetition_penalty": 1.0,
            "skip_special_tokens": True,
            "seed": 42,
        }
        if lora:
            assert payload["lora_request"] == ["lora0"]
        else:
            assert "lora_request" not in payload

        assert stream.closed
        assert clients[0].is_closed

    asyncio.run(run())


def test_generation_follows_redirects(engine, install_transport):
    async def run():
        stream = ControlledStream([event("ab"), b"data: [DONE]\n\n"])
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path == "/generate":
                return httpx.Response(307, headers={"Location": "/redirected"})

            return httpx.Response(200, stream=stream)

        clients = install_transport(handle)
        responses = await engine.chat(MESSAGES)

        assert responses[0].response_text == "ab"
        assert [request.url.path for request in requests] == ["/generate", "/redirected"]
        assert all(request.method == "POST" for request in requests)
        assert requests[0].content == requests[1].content
        assert stream.closed
        assert clients[0].is_closed

    asyncio.run(run())


@pytest.mark.parametrize("with_done", [False, True])
def test_chat_returns_final_response_and_closes(engine, install_transport, with_done):
    async def run():
        chunks = [event("a"), event("ab")]
        if with_done:
            chunks.extend([b"data: [DONE]\n\n", b"data: invalid JSON after DONE\n\n"])
        else:
            chunks[-1] = chunks[-1].rstrip(b"\n")

        stream = ControlledStream(chunks)
        clients = install_transport(lambda request: httpx.Response(200, stream=stream))
        responses = await engine.chat(MESSAGES)

        assert len(responses) == 1
        assert responses[0].response_text == "ab"
        assert responses[0].response_length == 2
        assert responses[0].prompt_length == 2
        assert responses[0].finish_reason == "stop"
        assert stream.closed
        assert clients[0].is_closed

    asyncio.run(run())


@pytest.mark.parametrize("method", ["chat", "stream_chat"])
@pytest.mark.parametrize("phase", ["request", "first_token", "next_token", "error_body"])
def test_cancellation_interrupts_http_wait_and_closes(engine, install_transport, method, phase):
    async def run():
        request_started = asyncio.Event()
        request_cancelled = asyncio.Event()
        request_release = asyncio.Event()
        chunks = [b"backend unavailable"] if phase == "error_body" else [event("a"), event("ab")]
        stream = ControlledStream(chunks, pause_at=1 if phase == "next_token" else 0)

        async def handle(request):
            if phase == "request":
                request_started.set()
                try:
                    await request_release.wait()
                except asyncio.CancelledError:
                    request_cancelled.set()
                    raise

            return httpx.Response(503 if phase == "error_body" else 200, stream=stream)

        clients = install_transport(handle)

        async def consume():
            if method == "chat":
                await engine.chat(MESSAGES)
            else:
                async for _ in engine.stream_chat(MESSAGES):
                    pass

        consumer = asyncio.create_task(consume())
        entered = request_started if phase == "request" else stream.read_started
        await asyncio.wait_for(entered.wait(), timeout=2)
        # Another task can run while generation is waiting for network data.
        heartbeat = asyncio.create_task(asyncio.sleep(0, result="responsive"))
        assert await asyncio.wait_for(heartbeat, timeout=2) == "responsive"
        assert not consumer.done()

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=2)

        assert clients[0].is_closed
        if phase == "request":
            assert request_cancelled.is_set()
        else:
            assert stream.cancelled
            assert stream.closed

    asyncio.run(run())


def test_closing_stream_chat_closes_response_between_tokens(engine, install_transport):
    async def run():
        stream = ControlledStream([event("a"), event("ab")], pause_at=1)
        clients = install_transport(lambda request: httpx.Response(200, stream=stream))
        generator = engine.stream_chat(MESSAGES)
        assert await anext(generator) == "a"

        await generator.aclose()

        assert stream.closed
        assert clients[0].is_closed
        assert not stream.read_started.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["chat", "stream_chat"])
@pytest.mark.parametrize(
    ("status", "chunks", "error", "message"),
    [
        (503, [b"backend unavailable"], RuntimeError, "SGLang server error: 503, backend unavailable"),
        (200, [b"data: invalid JSON\n\n"], json.JSONDecodeError, None),
        (200, [httpx.ReadError("connection lost")], httpx.ReadError, "connection lost"),
    ],
)
def test_http_and_parse_errors_propagate_and_close(engine, install_transport, method, status, chunks, error, message):
    async def run():
        stream = ControlledStream(chunks)
        clients = install_transport(lambda request: httpx.Response(status, stream=stream))
        with pytest.raises(error, match=message):
            if method == "chat":
                await engine.chat(MESSAGES)
            else:
                async for _ in engine.stream_chat(MESSAGES):
                    pass

        assert stream.closed
        assert clients[0].is_closed

    asyncio.run(run())
