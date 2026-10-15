# Copyright 2026 the LlamaFactory team.
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

"""Prefix-stable text and reasoning rendering for DeepSeek V4."""

from ...core.rendering.rendering import Renderer
from ...utils.helper import get_tokenizer
from ...utils.types import Message, ModelInput


# The ordinary-chat branch matches DeepSeek's assistant/end-of-thinking markers.
# Supply it per render call: never replace or save the checkpoint's chat_template.
DEEPSEEK_V4_CHAT_TEMPLATE = """
{%- if bos_token %}{{ bos_token }}{%- else %}{{ '<｜begin▁of▁sentence｜>' }}{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}{{ message['content'] }}
    {%- elif message['role'] == 'user' %}{{ '<｜User｜>' }}{{ message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<｜Assistant｜>' }}
        {%- if message.get('reasoning_content') %}
            {{- '<think>' }}{{ message['reasoning_content'] }}{{ '</think>' }}
        {%- else %}{{ '</think>' }}{%- endif %}
        {{- message['content'] }}{{ '<｜end▁of▁sentence｜>' }}
    {%- else %}{{ raise_exception('DeepSeek V4 text rendering supports system, user, and assistant roles only.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<｜Assistant｜>' }}
    {%- if enable_thinking | default(false) %}{{ '<think>' }}{%- else %}{{ '</think>' }}{%- endif %}
{%- endif %}
""".strip()


class DeepseekV4Renderer(Renderer):
    """Use V4 token boundaries without rewriting model files or weakening label checks."""

    def __init__(self, processor):
        super().__init__(processor)
        tokenizer = get_tokenizer(processor)
        for token in (
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
        ):
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != token:
                raise ValueError(f"DeepSeek V4 rendering requires the checkpoint's existing special token {token!r}.")

    def render_messages(
        self, messages: list[Message], tools: str | None = None, is_generate: bool = False, **kwargs
    ) -> ModelInput:
        if tools or any(
            message["role"] not in {"system", "user", "assistant"}
            or any(block["type"] not in {"text", "reasoning"} for block in message["content"])
            for message in messages
        ):
            raise ValueError(
                "DeepSeek V4's built-in renderer supports text and reasoning only; use a custom template for tools or media."
            )
        if any(
            message["role"] != "assistant" and any(block["type"] == "reasoning" for block in message["content"])
            for message in messages
        ):
            raise ValueError("DeepSeek V4 reasoning content is only supported in assistant messages.")
        if "thinking_mode" in kwargs:
            if kwargs["thinking_mode"] not in {"chat", "thinking"}:
                raise ValueError("DeepSeek V4 thinking_mode must be 'chat' or 'thinking'.")
            kwargs.setdefault("enable_thinking", kwargs["thinking_mode"] == "thinking")
        kwargs.setdefault("chat_template", DEEPSEEK_V4_CHAT_TEMPLATE)
        return super().render_messages(messages, tools, is_generate, **kwargs)
