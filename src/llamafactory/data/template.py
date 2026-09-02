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

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from typing_extensions import override

from ..extras import logging
from .data_utils import Role
from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from .mm_plugin import get_mm_plugin


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .formatter import SLOTS, Formatter
    from .mm_plugin import BasePlugin
    from .tool_utils import FunctionCall


logger = logging.get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    tool_call_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    enable_thinking: Optional[bool]
    preserve_thinking: bool
    mm_plugin: "BasePlugin"

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        discarding_history_cot: bool = False,  # only effect reasoning template
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract tool message."""
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content

    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    if tools and not system:
                        # Tool prompts that separate themselves from the system message with a
                        # leading newline would otherwise emit a blank line when there is no
                        # system message to separate from.
                        tool_text = tool_text.lstrip("\n")

                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info_rank0(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info_rank0(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            try:
                num_added_tokens = tokenizer.add_special_tokens(
                    dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
                )
            except TypeError:
                num_added_tokens = tokenizer.add_special_tokens(dict(additional_special_tokens=stop_words))
            logger.info_rank0("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.warning_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + Template._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + Template._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return " + ".join(slot_items)

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(self.format_system.apply(), tokenizer, placeholder="system_message")
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template

    def fix_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info_rank0(f"Cannot add this chat template to tokenizer: {e}.")

    @staticmethod
    def _convert_slots_to_ollama(
        slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content"
    ) -> str:
        r"""Convert slots to ollama template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append(slot_pieces[0])
                if len(slot_pieces) > 1:
                    slot_items.append("{{ " + placeholder + " }}")
                    if slot_pieces[1]:
                        slot_items.append(slot_pieces[1])
            elif isinstance(slot, set):  # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append(tokenizer.bos_token)
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append(tokenizer.eos_token)
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return "".join(slot_items)

    def _get_ollama_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama template."""
        prefix = self._convert_slots_to_ollama(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_ollama(self.format_system.apply(), tokenizer, placeholder=".System")
        user = self._convert_slots_to_ollama(self.format_user.apply(), tokenizer, placeholder=".Content")
        assistant = self._convert_slots_to_ollama(self.format_assistant.apply(), tokenizer, placeholder=".Content")
        return (
            f"{prefix}{{{{ if .System }}}}{system}{{{{ end }}}}"
            f"""{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}{user}"""
            f"""{{{{ else if eq .Role "assistant" }}}}{assistant}{{{{ end }}}}{{{{ end }}}}"""
        )

    def get_ollama_modelfile(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the ollama modelfile.

        TODO: support function calling.
        """
        modelfile = "# ollama modelfile auto-generated by llamafactory\n\n"
        modelfile += f'FROM .\n\nTEMPLATE """{self._get_ollama_template(tokenizer)}"""\n\n'

        if self.default_system:
            modelfile += f'SYSTEM """{self.default_system}"""\n\n'

        for stop_token_id in self.get_stop_token_ids(tokenizer):
            modelfile += f'PARAMETER stop "{tokenizer.convert_ids_to_tokens(stop_token_id)}"\n'

        modelfile += "PARAMETER num_ctx 4096\n"
        return modelfile


@dataclass
class MossVLTemplate(Template):
    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    if tools and not system:
                        tool_text = tool_text.lstrip("\n")

                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"],
                    thought_words=self.thought_words,
                    tool_call_words=self.tool_call_words,
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class Llama2Template(Template):
    r"""A template that fuse the system message to first user message."""

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: str,
        tools: str,
    ) -> list[list[int]]:
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system_message = self._convert_slots_to_jinja(
            self.format_system.apply(), tokenizer, placeholder="system_message"
        )
        user_message = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant_message = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% for message in loop_messages %}"
            "{% if loop.index0 == 0 and system_message is defined %}"
            "{% set content = " + system_message + " + message['content'] %}"
            "{% else %}{% set content = message['content'] %}{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user_message + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant_message + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        if not self.preserve_thinking:
            for i in range(1, len(messages) - 2, 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0].strip() not in messages[-1]["content"]
            and self.thought_words[1].strip() not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        discarding_history_cot: bool = False,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if discarding_history_cot:
            for i in range(1, len(messages) - 2, 2):  # preserve the last cot
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        if discarding_history_cot:
            turn_indices = [len(messages) - 2]
        else:
            turn_indices = range(0, len(messages), 2)

        for i in turn_indices:
            if (
                self.thought_words[0].strip() not in messages[i + 1]["content"]
                and self.thought_words[1].strip() not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


@dataclass
class Qwen38ReasoningTemplate(ReasoningTemplate):
    r"""Qwen3.8 template with reasoning-effort instructions and official system ordering."""

    reasoning_effort: str = "xhigh"

    def _get_reasoning_instruction(self) -> str:
        if self.enable_thinking is False:
            return ""

        if self.reasoning_effort == "xhigh":
            return (
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
                "answer."
            )
        elif self.reasoning_effort == "medium":
            return ""
        elif self.reasoning_effort == "low":
            return (
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                "conclusion without unnecessary elaboration."
            )
        else:
            # Defensive validation for callers that configure the template without DataArguments.
            raise ValueError(
                f"Unexpected reasoning effort {self.reasoning_effort}. "
                "Supported types are xhigh (default), medium, and low."
            )

    @override
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        system = (system or self.default_system).strip()
        reasoning_instruction = self._get_reasoning_instruction()
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                system_parts = []
                if reasoning_instruction:
                    system_parts.append(reasoning_instruction)
                if tools:
                    system_parts.append(self.format_tools.apply(content=tools)[0].lstrip("\n"))
                if system:
                    system_parts.append(system)
                if system_parts:
                    elements += self.format_system.apply(content="\n\n".join(system_parts))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(
                    content=message["content"], thought_words=self.thought_words, tool_call_words=self.tool_call_words
                )
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages


@dataclass
class Glm47ReasoningTemplate(ReasoningTemplate):
    r"""GLM-4.7 uses only the closing </think> tag for empty thinking blocks."""

    @override
    def add_thought(self, content: str = "") -> str:
        if not content:
            return self.thought_words[1]

        return self.thought_words[0] + content + self.thought_words[1]


TEMPLATES: dict[str, "Template"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    tool_call_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    enable_thinking: Optional[bool] = True,
    preserve_thinking: bool = False,
    mm_plugin: "BasePlugin" = get_mm_plugin(name="base"),
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    if format_assistant is not None:
        default_function_formatter = FunctionFormatter(slots=format_assistant.slots, tool_format="default")
    else:
        default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")

    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        tool_call_words=tool_call_words or ("<tool_call>", "</tool_call>"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        mm_plugin=mm_plugin,
    )


def parse_template(tokenizer: "PreTrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]
    template_class = ReasoningTemplate if "<think>" in assistant_slot else Template
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")  # remove thought tags

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:  # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return template_class(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_function=FunctionFormatter(slots=[assistant_slot], tool_format="default"),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tools=ToolFormatter(tool_format="default"),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        tool_call_words=("<tool_call>", "</tool_call>"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
        preserve_thinking=False,
        mm_plugin=get_mm_plugin(name="base"),
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    if data_args.template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.warning_rank0("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.warning_rank0("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]  # placeholder
    else:
        if data_args.template not in TEMPLATES:
            raise ValueError(f"Template {data_args.template} does not exist.")

        template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if isinstance(template, Qwen38ReasoningTemplate) and data_args.tool_format not in {None, "qwen3_8"}:
        raise ValueError("Template `qwen3_8` uses its built-in tool format; remove the incompatible `tool_format`.")

    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    if data_args.default_system is not None:
        logger.info_rank0(f"Using default system message: {data_args.default_system}.")
        template.default_system = data_args.default_system

    if isinstance(template, ReasoningTemplate):
        logger.warning_rank0(
            "You are using reasoning template. "
            "If the base model is NOT a reasoning model (i.e., it has a separate Instruct variant), "
            "please add `_nothink` suffix to disable thinking. "
            "For reasoning-only model families (e.g., Qwen3.6), the suffix is not needed. "
            "e.g., qwen3_vl_nothink"
        )
        template.enable_thinking = data_args.enable_thinking
        if isinstance(template, Qwen38ReasoningTemplate):
            template.reasoning_effort = data_args.reasoning_effort
            template.preserve_thinking = True if data_args.preserve_thinking is None else data_args.preserve_thinking
        elif data_args.preserve_thinking is not None:
            template.preserve_thinking = data_args.preserve_thinking

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    return template


register_template(
    name="alpaca",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n\n### Response:\n"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n\n"]),
    default_system=(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    ),
    replace_jinja_template=True,
)


register_template(
    name="bailing",
    format_user=StringFormatter(slots=["<role>HUMAN</role>{{content}}<role>ASSISTANT</role>"]),
    format_system=StringFormatter(slots=["<role>SYSTEM</role>{{content}}"]),
    format_observation=StringFormatter(slots=["<role>OBSERVATION</role>{{content}}<role>ASSISTANT</role>"]),
    stop_words=["<|endoftext|>"],
    efficient_eos=True,
)


register_template(
    name="bailing_v2",
    format_user=StringFormatter(slots=["<role>HUMAN</role>{{content}}<|role_end|><role>ASSISTANT</role>"]),
    format_system=StringFormatter(slots=["<role>SYSTEM</role>{{content}}<|role_end|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|role_end|>"]),
    format_observation=StringFormatter(
        slots=[
            "<role>OBSERVATION</role>\n<tool_response>\n{{content}}\n</tool_response><|role_end|><role>ASSISTANT</role>"
        ]
    ),
    format_function=FunctionFormatter(slots=["{{content}}<|role_end|>"], tool_format="ling"),
    format_tools=ToolFormatter(tool_format="ling"),
    stop_words=["<|endoftext|>"],
    efficient_eos=True,
)


register_template(
    name="breeze",
    format_user=StringFormatter(slots=["[INST] {{content}} [/INST] "]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    efficient_eos=True,
)


register_template(
    name="chatglm3",
    format_user=StringFormatter(slots=[{"token": "<|user|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]),
    format_assistant=StringFormatter(slots=["\n", "{{content}}"]),
    format_system=StringFormatter(slots=[{"token": "<|system|>"}, "\n", "{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(
        slots=[{"token": "<|observation|>"}, "\n", "{{content}}", {"token": "<|assistant|>"}]
    ),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=[{"token": "[gMASK]"}, {"token": "sop"}]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


register_template(
    name="chatml",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
    replace_jinja_template=True,
)


# copied from chatml template
register_template(
    name="chatml_de",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="Du bist ein freundlicher und hilfsbereiter KI-Assistent.",
    stop_words=["<|im_end|>", "<|im_start|>"],
    replace_eos=True,
    replace_jinja_template=True,
)


register_template(
    name="cohere",
    format_user=StringFormatter(
        slots=[
            (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
        ]
    ),
    format_system=StringFormatter(slots=["<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{content}}<|END_OF_TURN_TOKEN|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


# copied from chatml template
register_template(
    name="cpm4",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
)


# copied from chatml template
register_template(
    name="dbrx",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system=(
        "You are DBRX, created by Databricks. You were last updated in December 2023. "
        "You answer questions based on information available up to that point.\n"
        "YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough "
        "responses to more complex and open-ended questions.\nYou assist with various tasks, "
        "from writing to coding (using markdown for code blocks — remember to use ``` with "
        "code, JSON, and tables).\n(You do not have real-time data access or code execution "
        "capabilities. You avoid stereotyping and provide balanced perspectives on "
        "controversial topics. You do not provide song lyrics, poems, or news articles and "
        "do not divulge details of your training data.)\nThis is your system prompt, "
        "guiding your responses. Do not reference it, just respond to the user. If you find "
        "yourself talking about this message, stop. You should be responding appropriately "
        "and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION "
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY."
    ),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


register_template(
    name="deepseek",
    format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="deepseek3",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


# copied from deepseek3 template
register_template(
    name="deepseekr1",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=ReasoningTemplate,
)


register_template(
    name="hy3",
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}<｜hy_Assistant｜>"]),
    format_assistant=StringFormatter(slots=["{{content}}<｜hy_eos｜>"]),
    format_system=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<｜hy_eos｜>"],
    replace_eos=True,
    thought_words=("<think>", "</think>"),
    template_class=ReasoningTemplate,
)


register_template(
    name="deepseekcoder",
    format_user=StringFormatter(slots=["### Instruction:\n{{content}}\n### Response:"]),
    format_assistant=StringFormatter(slots=["\n{{content}}\n<|EOT|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
        "developed by DeepSeek Company, and you only answer questions related to computer science. "
        "For politically sensitive questions, security and privacy issues, "
        "and other non-computer science questions, you will refuse to answer.\n"
    ),
)


register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}", {"eos_token"}, "\nAssistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["System: {{content}}", {"eos_token"}, "\n"]),
    replace_jinja_template=True,
)


register_template(
    name="dots_ocr",
    format_user=StringFormatter(slots=["<|user|>{{content}}<|endofuser|><|assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|endofassistant|>"]),
    format_system=StringFormatter(slots=["<|system|>{{content}}<|endofsystem|>\n"]),
    stop_words=["<|endofassistant|>"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(
        name="qwen2_vl",
        image_token="<|imgpad|>",
        video_token="<|vidpad|>",
        vision_bos_token="<|img|>",
        vision_eos_token="<|endofimg|>",
    ),
)


register_template(
    name="empty",
    format_assistant=StringFormatter(slots=["{{content}}"]),
)


# copied from chatml template
register_template(
    name="ernie",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n\n<|im_start|>assistant\n"]),
    default_system="<global_setting>\nthink_mode=True\n</global_setting>",
    stop_words=["<|im_end|>"],
)


register_template(
    name="ernie_nothink",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant: "]),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_sentence|>"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    stop_words=["<|end_of_sentence|>"],
)


register_template(
    name="ernie_vl",
    format_user=StringFormatter(slots=["User: {{content}}"]),
    format_assistant=StringFormatter(slots=["\nAssistant: {{content}}<|end_of_sentence|>"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    stop_words=["<|end_of_sentence|>"],
    replace_eos=True,
    replace_jinja_template=True,
    template_class=ReasoningTemplate,
    mm_plugin=get_mm_plugin(name="ernie_vl", image_token="<|IMAGE_PLACEHOLDER|>", video_token="<|VIDEO_PLACEHOLDER|>"),
)


register_template(
    name="exaone",
    format_user=StringFormatter(slots=["[|user|]{{content}}\n[|assistant|]"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}, "\n"]),
    format_system=StringFormatter(slots=["[|system|]{{content}}[|endofturn|]\n"]),
)


register_template(
    name="falcon",
    format_user=StringFormatter(slots=["User: {{content}}\nFalcon:"]),
    format_assistant=StringFormatter(slots=["{{content}}\n"]),
    efficient_eos=True,
)


# copied from chatml template
register_template(
    name="falcon_h1",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>", "<|end_of_text|>"],
)


register_template(
    name="fewshot",
    format_assistant=StringFormatter(slots=["{{content}}\n\n"]),
    efficient_eos=True,
    replace_jinja_template=True,
)


register_template(
    name="gemma",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    template_class=Llama2Template,
)


# copied from gemma template
register_template(
    name="gemma2",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<eos>", "<end_of_turn>"],
    efficient_eos=True,
    template_class=Llama2Template,
)


# copied from gemma template
register_template(
    name="gemma3",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>"),
    template_class=Llama2Template,
)


register_template(
    name="gemma3n",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin("gemma3n", image_token="<image_soft_token>", audio_token="<audio_soft_token>"),
    template_class=Llama2Template,
)


register_template(
    name="gemma4",
    format_user=StringFormatter(slots=["<|turn>user\n{{content}}<turn|>\n<|turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<turn|>\n"]),
    format_system=StringFormatter(
        slots=["<|turn>system\n<|think|>{{content}}<turn|>\n"]
    ),  #  default thought signal contained
    format_observation=StringFormatter(
        slots=["<|turn>tool\n{{content}}<turn|>\n<|turn>model\n"]
    ),  # seem not consistent with the chattemplate
    format_tools=ToolFormatter(tool_format="gemma4"),
    format_function=FunctionFormatter(slots=["<|tool>{{content}}<tool|>"], tool_format="gemma4"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<turn|>"],
    default_system="You are a helpful assistant.",  # important for thinking
    thought_words=("<|channel>thought\n", "<channel|>"),
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        "gemma4",
        image_token="<|image|>",
        video_token="<|video|>",
    ),
    template_class=ReasoningTemplate,
)


register_template(
    name="gemma4n",
    format_user=StringFormatter(slots=["<|turn>user\n{{content}}<turn|>\n<|turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<turn|>\n"]),
    format_system=StringFormatter(
        slots=["<|turn>system\n<|think|>{{content}}<turn|>\n"]
    ),  #  default thought signal contained
    format_observation=StringFormatter(slots=["<|turn>tool\n{{content}}<turn|>\n<|turn>model\n"]),
    format_tools=ToolFormatter(tool_format="gemma4"),
    format_function=FunctionFormatter(slots=["<|tool>{{content}}<tool|>"], tool_format="gemma4"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<turn|>"],
    default_system="You are a helpful assistant.",  # important for thinking
    thought_words=("<|channel>thought\n", "<channel|>"),
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        "gemma4",
        image_token="<|image|>",
        video_token="<|video|>",
        audio_token="<|audio|>",
    ),
    template_class=ReasoningTemplate,
)


register_template(
    name="glm4",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
)


# copied from glm4 template
register_template(
    name="glm4_moe",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glm4v",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>", "</answer>"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="glm4v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glm4_5v",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>", "</answer>"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="glm4v", image_token="<|image|>", video_token="<|video|>"),
    template_class=ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glm_ocr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="glm4v", image_token="<|image|>", video_token="<|video|>"),
)


# copied from glm4_moe template
register_template(
    name="glm4_7",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    thought_words=("<think>", "</think>"),
    efficient_eos=True,
    template_class=Glm47ReasoningTemplate,
)


# copied from glm4 template
register_template(
    name="glmz1",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    stop_words=["<|user|>", "<|observation|>"],
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="gpt_oss",
    format_user=StringFormatter(slots=["<|start|>user<|message|>{{content}}<|end|><|start|>assistant"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["<|start|>system<|message|>{{content}}<|end|>"]),
    default_system="You are ChatGPT, a large language model trained by OpenAI.",
    thought_words=("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"),
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="granite3",
    format_user=StringFormatter(
        slots=[
            "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_text|>\n"]),
    format_system=StringFormatter(slots=["<|start_of_role|>system<|end_of_role|>{{content}}<|end_of_text|>\n"]),
)


register_template(
    name="granite3_vision",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}\n<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}\n"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


register_template(
    name="granite4",
    format_user=StringFormatter(
        slots=[
            "<|start_of_role|>user<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_text|>\n"]),
    format_system=StringFormatter(slots=["<|start_of_role|>system<|end_of_role|>{{content}}<|end_of_text|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|end_of_text|>\n"], tool_format="default"),
    format_observation=StringFormatter(
        slots=["<|start_of_role|>tool<|end_of_role|>{{content}}<|end_of_text|>\n<|start_of_role|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="default"),
    stop_words=["<|end_of_text|>"],
    default_system="You are Granite, developed by IBM. You are a helpful AI assistant.",
)


register_template(
    name="index",
    format_user=StringFormatter(slots=["reserved_0{{content}}reserved_1"]),
    format_system=StringFormatter(slots=["<unk>{{content}}"]),
    efficient_eos=True,
)


register_template(
    name="hunyuan",
    format_user=StringFormatter(slots=["{{content}}<|extra_0|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|eos|>"]),
    format_system=StringFormatter(slots=["{{content}}<|extra_4|>"]),
    format_prefix=EmptyFormatter(slots=["<|startoftext|>"]),
    stop_words=["<|eos|>"],
)


register_template(
    name="hunyuan_small",
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}<｜hy_place▁holder▁no▁8｜>"]),
    format_assistant=StringFormatter(slots=["{{content}}<｜hy_place▁holder▁no▁2｜>"]),
    format_system=StringFormatter(slots=["{{content}}<｜hy_place▁holder▁no▁3｜>"]),
    format_prefix=EmptyFormatter(slots=["<｜hy_begin▁of▁sentence｜>"]),
    stop_words=["<｜hy_place▁holder▁no▁2｜>"],
)


# The following two templates are copied from the official Hy-MT2 chat templates:
# https://github.com/Tencent-Hunyuan/Hy-MT2/blob/main/train/llama_factory_support/hy_dense_template.py
register_template(
    name="hy_dense_1_8b",
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}"]),
    format_assistant=StringFormatter(slots=["<｜hy_Assistant｜>{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}<｜hy_place▁holder▁no▁3｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<｜hy_place▁holder▁no▁2｜>"],
    efficient_eos=True,
)


register_template(
    name="hy_dense_7b",
    format_user=StringFormatter(slots=["{{content}}<|extra_0|>"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}<|extra_4|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eos|>"],
    efficient_eos=True,
)


register_template(
    name="intern2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language "
        "chosen by the user such as English and 中文."
    ),
    stop_words=["<|im_end|>"],
)


register_template(
    name="intern_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    default_system=(
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    ),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="intern_vl", image_token="<image>", video_token="<video>"),
)


register_template(
    name="intern_s1",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="intern_vl", image_token="<image>", video_token="<video>"),
)


# copied from qwen template
register_template(
    name="keye_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


register_template(
    name="kimi_vl",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant",
    stop_words=["<|im_end|>"],
    thought_words=("◁think▷", "◁/think▷"),
    mm_plugin=get_mm_plugin("kimi_vl", image_token="<|media_pad|>"),
    template_class=ReasoningTemplate,
)


register_template(
    name="lfm2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="lfm2"),
    format_observation=StringFormatter(
        slots=[
            "<|im_start|>tool\n<|tool_response_start|>{{content}}<|tool_response_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="lfm2"),
    default_system="You are a helpful AI assistant.",
    stop_words=["<|im_end|>"],
    tool_call_words=("<|tool_call_start|>", "<|tool_call_end|>"),
    replace_eos=True,
)


register_template(
    name="lfm2_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="lfm2"),
    format_observation=StringFormatter(
        slots=[
            "<|im_start|>tool\n<|tool_response_start|>{{content}}<|tool_response_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="lfm2"),
    default_system="You are a helpful multimodal assistant by Liquid AI.",
    stop_words=["<|im_end|>"],
    tool_call_words=("<|tool_call_start|>", "<|tool_call_end|>"),
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="lfm2_vl", image_token="<image>"),
)


register_template(
    name="llama2",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    template_class=Llama2Template,
)


# copied from llama2 template
register_template(
    name="llama2_zh",
    format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
    format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
    default_system="You are a helpful assistant. 你是一个乐于助人的助手。",
    template_class=Llama2Template,
)


register_template(
    name="llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
)


register_template(
    name="llama4",
    format_user=StringFormatter(
        slots=["<|header_start|>user<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot|>"]),
    format_system=StringFormatter(slots=["<|header_start|>system<|header_end|>\n\n{{content}}<|eot|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            "<|header_start|>ipython<|header_end|>\n\n{{content}}<|eot|><|header_start|>assistant<|header_end|>\n\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot|>", "<|eom|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llama4", image_token="<|image|>"),
)


# copied from llama3 template
register_template(
    name="mllama",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="mllama", image_token="<|image|>"),
)


register_template(
    name="moonlight",
    format_user=StringFormatter(
        slots=["<|im_user|>user<|im_middle|>{{content}}<|im_end|><|im_assistant|>assistant<|im_middle|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_system|>system<|im_middle|>{{content}}<|im_end|>"]),
    default_system="You are a helpful assistant provided by Moonshot-AI.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from qwen template
register_template(
    name="moss_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="moss_vl",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        vision_bos_token="<|vision_start|>",
        vision_eos_token="<|vision_end|>",
        time_bos_token="<|time_start|>",
        time_eos_token="<|time_end|>",
    ),
    template_class=MossVLTemplate,
)


# copied from vicuna template
register_template(
    name="llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


# copied from vicuna template
register_template(
    name="llava_next",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from llama3 template
register_template(
    name="llava_next_llama3",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
    format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
    format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
    format_observation=StringFormatter(
        slots=[
            (
                "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_tools=ToolFormatter(tool_format="llama3"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>", "<|eom_id|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from mistral template
register_template(
    name="llava_next_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
    template_class=Llama2Template,
)


# copied from qwen template
register_template(
    name="llava_next_qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from chatml template
register_template(
    name="llava_next_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="llava_next", image_token="<image>"),
)


# copied from vicuna template
register_template(
    name="llava_next_video",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


# copied from mistral template
register_template(
    name="llava_next_video_mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
    template_class=Llama2Template,
)


# copied from chatml template
register_template(
    name="llava_next_video_yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    mm_plugin=get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>"),
)


# copied from qwen template
register_template(
    name="mimo",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


# copied from qwen template
register_template(
    name="mimo_v2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are MiMo, a helpful AI assistant engineered by Xiaomi.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    thought_words=("<think>", "</think>"),
    template_class=ReasoningTemplate,
)


# copied from qwen2vl
register_template(
    name="mimo_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are MiMo, an AI assistant developed by Xiaomi.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


# copied from chatml template
register_template(
    name="minicpm_v",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are a helpful assistant.",
    mm_plugin=get_mm_plugin(name="minicpm_v", image_token="<image>", video_token="<video>"),
)


register_template(
    name="minicpm_v_4_6",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are a helpful assistant.",
    mm_plugin=get_mm_plugin(name="minicpm_v_4_6", image_token="<image>", video_token="<video>"),
)


# copied from minicpm_v template
register_template(
    name="minicpm_o",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are a helpful assistant. You can accept audio and text input and output voice and text.",
    mm_plugin=get_mm_plugin(name="minicpm_v", image_token="<image>", video_token="<video>", audio_token="<audio>"),
)


register_template(
    name="minicpm5",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="minicpm5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="minicpm5"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="minimax1",
    format_user=StringFormatter(
        slots=[
            "<beginning_of_sentence>user name=user\n{{content}}<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
        ]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_sentence>\n"]),
    format_system=StringFormatter(
        slots=["<beginning_of_sentence>system ai_setting=assistant\n{{content}}<end_of_sentence>\n"]
    ),
    format_function=FunctionFormatter(slots=["{{content}}<end_of_sentence>\n"], tool_format="minimax1"),
    format_observation=StringFormatter(
        slots=[
            "<beginning_of_sentence>tool name=tools\n{{content}}<end_of_sentence>\n<beginning_of_sentence>ai name=assistant\n"
        ]
    ),
    format_tools=ToolFormatter(tool_format="minimax1"),
    default_system="You are a helpful assistant.",
    stop_words=["<end_of_sentence>"],
)


register_template(
    name="minimax2",
    format_user=StringFormatter(slots=["]~b]user\n{{content}}[e~[\n]~b]ai\n"]),
    format_assistant=StringFormatter(slots=["{{content}}[e~[\n"]),
    format_system=StringFormatter(slots=["]~!b[]~b]system\n{{content}}[e~[\n"]),
    format_function=FunctionFormatter(slots=["{{content}}[e~[\n"], tool_format="minimax2"),
    format_observation=StringFormatter(slots=["]~b]tool\n<response>{{content}}</response>[e~[\n]~b]ai\n"]),
    format_tools=ToolFormatter(tool_format="minimax2"),
    default_system="You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax.",
    stop_words=["[e~["],
    template_class=ReasoningTemplate,
)


# mistral tokenizer v3 tekken
register_template(
    name="ministral",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
)


# mistral tokenizer v3
register_template(
    name="mistral",
    format_user=StringFormatter(slots=["[INST] {{content}}[/INST]"]),
    format_assistant=StringFormatter(slots=[" {{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS] {"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
)


# mistral tokenizer v7 tekken (copied from ministral)
register_template(
    name="mistral_small",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["[SYSTEM_PROMPT]{{content}}[/SYSTEM_PROMPT]"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
)


register_template(
    name="ministral3",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    template_class=Llama2Template,
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
)


register_template(
    name="olmo",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"eos_token"}]),
)


register_template(
    name="openchat",
    format_user=StringFormatter(slots=["GPT4 Correct User: {{content}}", {"eos_token"}, "GPT4 Correct Assistant:"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
)


register_template(
    name="openchat-3.6",
    format_user=StringFormatter(
        slots=[
            (
                "<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{{content}}<|eot_id|>"
                "<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n"
            )
        ]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eot_id|>"],
)


# copied from chatml template
register_template(
    name="opencoder",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    default_system="You are OpenCoder, created by OpenCoder Team.",
    stop_words=["<|im_end|>"],
)


register_template(
    name="paligemma",
    format_user=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="paligemma", image_token="<image>"),
    template_class=Llama2Template,
)


# copied from gemma template
register_template(
    name="paligemma_chat",
    format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
    format_observation=StringFormatter(
        slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
    ),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<end_of_turn>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="paligemma", image_token="<image>"),
    template_class=Llama2Template,
)


register_template(
    name="phi",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


register_template(
    name="phi_small",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    format_prefix=EmptyFormatter(slots=[{"<|endoftext|>"}]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


register_template(
    name="phi4",
    format_user=StringFormatter(
        slots=["<|im_start|>user<|im_sep|>{{content}}<|im_end|><|im_start|>assistant<|im_sep|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>"]),
    format_system=StringFormatter(slots=["<|im_start|>system<|im_sep|>{{content}}<|im_end|>"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


register_template(
    name="phi4_mini",
    format_user=StringFormatter(slots=["<|user|>{{content}}<|end|><|assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>"]),
    format_system=StringFormatter(slots=["<|system|>{{content}}<|end|>"]),
    format_tools=StringFormatter(slots=["<|tool|>{{content}}<|/tool|>"]),
    stop_words=["<|end|>"],
    replace_eos=True,
)


# copied from ministral template
register_template(
    name="pixtral",
    format_user=StringFormatter(slots=["[INST]{{content}}[/INST]"]),
    format_system=StringFormatter(slots=["{{content}}\n\n"]),
    format_function=FunctionFormatter(slots=["[TOOL_CALLS]{{content}}", {"eos_token"}], tool_format="mistral"),
    format_observation=StringFormatter(slots=["""[TOOL_RESULTS]{"content": {{content}}}[/TOOL_RESULTS]"""]),
    format_tools=ToolFormatter(tool_format="mistral"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    mm_plugin=get_mm_plugin(name="pixtral", image_token="[IMG]"),
    template_class=Llama2Template,
)


# copied from chatml template
register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from qwen template
register_template(
    name="qwen3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


# copied from qwen template
register_template(
    name="qwen3_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
)


# copied from chatml template
register_template(
    name="qwen2_audio",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_audio", audio_token="<|AUDIO|>"),
)


# copied from qwen template
register_template(
    name="qwen2_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="qwen2_omni",
        image_token="<|IMAGE|>",
        video_token="<|VIDEO|>",
        audio_token="<|AUDIO|>",
        vision_bos_token="<|vision_bos|>",
        vision_eos_token="<|vision_eos|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
    ),
)


register_template(
    name="qwen3_omni",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="qwen2_omni", image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token="<|audio_pad|>"
    ),
    template_class=ReasoningTemplate,
)


register_template(
    name="qwen3_omni_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="qwen2_omni", image_token="<|image_pad|>", video_token="<|video_pad|>", audio_token="<|audio_pad|>"
    ),
)


# copied from qwen template
register_template(
    name="qwen2_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


# copied from qwen template
register_template(
    name="qwen3_vl",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


# copied from qwen template
register_template(
    name="qwen3_vl_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


register_template(
    name="qwen3_5",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_5"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


register_template(
    name="qwen3_5_nothink",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_5"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


# copied from qwen3_5 template
register_template(
    name="qwen3_6",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_5"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_5"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=ReasoningTemplate,
)


register_template(
    name="qwen3_8",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen3_8"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen3_8"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    preserve_thinking=True,
    mm_plugin=get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
    template_class=Qwen38ReasoningTemplate,
)


register_template(
    name="sailor",
    format_user=StringFormatter(slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    default_system=(
        "You are an AI assistant named Sailor created by Sea AI Lab. "
        "Your answer should be friendly, unbiased, faithful, informative and detailed."
    ),
    stop_words=["<|im_end|>"],
)


register_template(
    name="seed_coder",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "user\n{{content}}", {"eos_token"}, {"bos_token"}, "assistant\n"]
    ),
    format_system=StringFormatter(slots=[{"bos_token"}, "system\n{{content}}", {"eos_token"}]),
    default_system=(
        "You are an AI programming assistant, utilizing the Seed-Coder model, developed by ByteDance Seed, "
        "and you only answer questions related to computer science. For politically sensitive questions, "
        "security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n"
    ),
)


# copied from seed_coder
register_template(
    name="seed_oss",
    format_user=StringFormatter(
        slots=[{"bos_token"}, "user\n{{content}}", {"eos_token"}, {"bos_token"}, "assistant\n"]
    ),
    format_system=StringFormatter(slots=[{"bos_token"}, "system\n{{content}}", {"eos_token"}]),
    format_function=FunctionFormatter(slots=[{"bos_token"}, "\n{{content}}", {"eos_token"}], tool_format="seed_oss"),
    format_tools=ToolFormatter(tool_format="seed_oss"),
    template_class=ReasoningTemplate,
    thought_words=("<seed:think>", "</seed:think>"),
)


register_template(
    name="smollm",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
)


register_template(
    name="smollm2",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    default_system="You are a helpful AI assistant named SmolLM, trained by Hugging Face.",
)


register_template(
    name="solar",
    format_user=StringFormatter(slots=["### User:\n{{content}}\n\n### Assistant:\n"]),
    format_system=StringFormatter(slots=["### System:\n{{content}}\n\n"]),
    efficient_eos=True,
)


register_template(
    name="starchat",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|end|>\n<|assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}<|end|>\n"]),
    stop_words=["<|end|>"],
)


register_template(
    name="telechat2",
    format_user=StringFormatter(slots=["<_user>{{content}}<_bot>"]),
    format_system=StringFormatter(slots=["<_system>{{content}}"]),
    default_system=(
        "你是中国电信星辰语义大模型，英文名是TeleChat，你是由中电信人工智能科技有限公司和中国电信人工智能研究院（TeleAI）研发的人工智能助手。"
    ),
)


register_template(
    name="vicuna",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    replace_jinja_template=True,
)


register_template(
    name="video_llava",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=get_mm_plugin(name="video_llava", image_token="<image>", video_token="<video>"),
)


register_template(
    name="xuanyuan",
    format_user=StringFormatter(slots=["Human: {{content}} Assistant:"]),
    default_system=(
        "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，"
        "会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、"
        "不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
    ),
)


# copied from chatml template
register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
)


register_template(
    name="yi_vl",
    format_user=StringFormatter(slots=["### Human: {{content}}\n### Assistant:"]),
    format_assistant=StringFormatter(slots=["{{content}}\n"]),
    default_system=(
        "This is a chat between an inquisitive human and an AI assistant. "
        "Assume the role of the AI assistant. Read all the images carefully, "
        "and respond to the human's questions with informative, helpful, detailed and polite answers. "
        "这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。"
        "仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    ),
    stop_words=["###"],
    efficient_eos=True,
    mm_plugin=get_mm_plugin(name="llava", image_token="<image>"),
)


register_template(
    name="youtu",
    format_user=StringFormatter(slots=["<|User|>{{content}}<|Assistant|>"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_text|>"]),
    format_system=StringFormatter(slots=["{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="default"),
    format_observation=StringFormatter(slots=["<tool_response>\n{{content}}\n</tool_response><|Assistant|>"]),
    format_tools=ToolFormatter(tool_format="default"),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|end_of_text|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="youtu_vl",
    format_user=StringFormatter(
        slots=["<|begin_of_text|>user\n{{content}}<|end_of_text|>\n<|begin_of_text|>assistant\n"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|end_of_text|>\n"]),
    format_system=StringFormatter(slots=["<|begin_of_text|>system\n{{content}}<|end_of_text|>\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|end_of_text|>"],
    mm_plugin=get_mm_plugin(name="youtu_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
)


register_template(
    name="yuan",
    format_user=StringFormatter(slots=["{{content}}", {"token": "<sep>"}]),
    format_assistant=StringFormatter(slots=["{{content}}<eod>\n"]),
    stop_words=["<eod>"],
)


register_template(
    name="zephyr",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}", {"eos_token"}, "<|assistant|>\n"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}", {"eos_token"}]),
    default_system="You are Zephyr, a helpful assistant.",
)


# copied from glm4_7 template
register_template(
    name="aeva",
    format_user=StringFormatter(slots=["<|user|>\n{{content}}<|assistant|>"]),
    format_assistant=StringFormatter(slots=["\n{{content}}"]),
    format_system=StringFormatter(slots=["<|system|>\n{{content}}"]),
    format_function=FunctionFormatter(slots=["{{content}}"], tool_format="glm4_moe"),
    format_observation=StringFormatter(slots=["<|observation|>\n{{content}}<|assistant|>"]),
    format_tools=ToolFormatter(tool_format="glm4_moe"),
    format_prefix=EmptyFormatter(slots=["[gMASK]<sop>"]),
    default_system=(
        "You are an AI assistant named Aeva created by Zongzhi Lou. "
        "Your answer should be friendly, unbiased, faithful, informative and detailed."
    ),
    stop_words=["<|user|>", "<|observation|>"],
    thought_words=("<think>", "</think>"),
    efficient_eos=True,
    template_class=Glm47ReasoningTemplate,
)
