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

import time
from enum import StrEnum, unique
from typing import Any, Literal

from pydantic import BaseModel, Field


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@unique
class Finish(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["owner"] = "owner"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = []


class Function(BaseModel):
    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: FunctionDefinition | None = None


class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function


class URL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url", "video_url", "audio_url"]
    text: str | None = None
    image_url: URL | None = None
    video_url: URL | None = None
    audio_url: URL | None = None


class ChatMessage(BaseModel):
    role: Role
    content: str | list[MultimodalInputItem] | None = None
    tool_calls: list[FunctionCall] | None = None


class ChatCompletionMessage(BaseModel):
    role: Role | None = None
    content: str | None = None
    tool_calls: list[FunctionCall] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[FunctionAvailable] | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Finish | None = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]


class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: list[str]
    max_length: int | None = None


class ScoreEvaluationResponse(BaseModel):
    id: str
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: list[float]
