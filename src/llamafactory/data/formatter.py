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

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import override

from .data_utils import SLOTS
from .tool_utils import FunctionCall, get_tool_utils


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: str | None = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r"""Forms a list of slots according to the inputs to encode."""
        ...

    def extract(self, content: str) -> str | list["FunctionCall"]:
        r"""Extract a list of tuples from the response message if using tools.

        Each tuple consists of function name and function arguments.
        """
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}.")

        return elements


@dataclass
class FunctionFormatter(StringFormatter):
    def __post_init__(self):
        super().__post_init__()
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content: str = kwargs.pop("content")
        thought_words = kwargs.pop("thought_words", None)
        tool_call_words = kwargs.pop("tool_call_words", None)

        def _parse_functions(json_content: str) -> list["FunctionCall"]:
            try:
                tool_calls = json.loads(json_content)
                if not isinstance(tool_calls, list):  # parallel function call
                    tool_calls = [tool_calls]

                return [FunctionCall(tc["name"], json.dumps(tc["arguments"], ensure_ascii=False)) for tc in tool_calls]
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON format in function message: {str([content])}.")

        tool_call_match = None
        if tool_call_words and len(tool_call_words) == 2:
            tool_call_regex = re.compile(
                rf"{re.escape(tool_call_words[0])}(.*?){re.escape(tool_call_words[1])}", re.DOTALL
            )
            tool_call_match = re.search(tool_call_regex, content)

        if tool_call_match is None:
            thought_match = None
            if thought_words and len(thought_words) == 2:
                regex = re.compile(rf"{re.escape(thought_words[0])}(.*?){re.escape(thought_words[1])}", re.DOTALL)
                thought_match = re.search(regex, content)

            if thought_match:
                json_part = content.replace(thought_match.group(0), "")
            else:
                json_part = content

            functions = _parse_functions(json_part)
            function_str = self.tool_utils.function_formatter(functions)
            if thought_match:
                function_str = thought_match.group(0) + function_str
        else:
            thought_content = content.replace(tool_call_match.group(0), "")
            functions = _parse_functions(tool_call_match.group(1))
            function_str = self.tool_utils.function_formatter(functions)
            function_str = thought_content + function_str

        return super().apply(content=function_str)


@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return [self.tool_utils.tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}.")  # flat string

    @override
    def extract(self, content: str) -> str | list["FunctionCall"]:
        return self.tool_utils.tool_extractor(content)
