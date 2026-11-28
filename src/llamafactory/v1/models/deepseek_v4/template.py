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


# Text/reasoning protocol from DeepSeek-V4-Flash's official encoding_dsv4.py,
# revision 60d8d70770c6776ff598c94bb586a859a38244f1. Tools/media require a custom template.
DEEPSEEK_V4_CHAT_TEMPLATE = """
{%- set thinking = thinking_mode == 'thinking' if thinking_mode is defined else enable_thinking | default(false) %}
{%- set drop = drop_thinking | default(true) %}
{%- set ns = namespace(last_user=-1) %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}{%- set ns.last_user = loop.index0 %}{%- endif %}
{%- endfor %}
{%- if bos_token %}{{ bos_token }}{%- else %}{{ '<｜begin▁of▁sentence｜>' }}{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}{{ message['content'] or '' }}
    {%- elif message['role'] == 'user' %}
        {%- if not loop.first and messages[loop.index0 - 1]['role'] == 'user' %}
            {{- '\n\n' }}
        {%- else %}{{ '<｜User｜>' }}{%- endif %}
        {{- message['content'] or '' }}
        {%- if (loop.last and add_generation_prompt) or (not loop.last and messages[loop.index0 + 1]['role'] == 'assistant') %}
            {{- '<｜Assistant｜>' }}
            {%- if thinking and (not drop or loop.index0 >= ns.last_user) %}
                {{- '<think>' }}
            {%- else %}{{ '</think>' }}{%- endif %}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {%- if thinking and (not drop or loop.index0 > ns.last_user) %}
            {{- message.get('reasoning_content', '') or '' }}{{ '</think>' }}
        {%- endif %}
        {{- message['content'] or '' }}{{ '<｜end▁of▁sentence｜>' }}
    {%- else %}{{ raise_exception('DeepSeek V4 text rendering supports system, user, and assistant roles only.') }}
    {%- endif %}
{%- endfor %}
""".strip()


class DeepseekV4Renderer(Renderer):
    """Use V4 token boundaries without rewriting model files or weakening label checks."""

    position_start = 0

    def __init__(self, processor, *, use_model_template: bool = True):
        super().__init__(processor)
        self.use_model_template = use_model_template
        if not use_model_template:
            return
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
        if not self.use_model_template:
            return super().render_messages(messages, tools, is_generate, **kwargs)
        if kwargs.get("reasoning_effort") not in (None, "high"):
            raise ValueError("DeepSeek V4's built-in text renderer supports default/high reasoning effort only.")
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
