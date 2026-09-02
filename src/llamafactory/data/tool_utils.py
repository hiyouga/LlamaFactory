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

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Union

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{tool_text}"
)

GLM4_MOE_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:"
    "\n<tool_call>{{function-name}}"
    "\n<arg_key>{{arg-key-1}}</arg_key>"
    "\n<arg_value>{{arg-value-1}}</arg_value>"
    "\n<arg_key>{{arg-key-2}}</arg_key>"
    "\n<arg_value>{{arg-value-2}}</arg_value>"
    "\n...\n</tool_call>\n"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)

MINIMAX_M1_TOOL_PROMPT = (
    "You are provided with these tools:\n<tools>\n{tool_text}</tools>\n\n"
    "If you need to call tools, please respond with <tool_calls></tool_calls> XML tags, and provide tool-name and "
    "json-object of arguments, following the format below:\n<tool_calls>\n"
    """{{"name": <tool-name-1>, "arguments": <args-json-object-1>}}\n...\n</tool_calls>"""
)

MINIMAX_M2_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more tools to assist with the user query.\n"
    "Here are the tools available in JSONSchema format:\n\n<tools>\n{tool_text}</tools>\n\n"
    "When making tool calls, use XML format to invoke tools and pass parameters:\n"
    """\n<minimax:tool_call>\n<invoke name="tool-name-1">\n<parameter name="param-key-1">param-value-1</parameter>\n"""
    """<parameter name="param-key-2">param-value-2</parameter>\n...\n</invoke>\n</minimax:tool_call>"""
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

QWEN35_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>{tool_text}"
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: "
    "an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)

SEED_TOOL_PROMPT = (
    "system\nYou are Doubao, a helpful AI assistant. You may call one or more functions to assist with the user query."
    "Tool List:\nYou are authorized to use the following tools (described in JSON Schema format). Before performing "
    "any task, you must decide how to call them based on the descriptions and parameters of these tools.{tool_text}\n"
    "工具调用请遵循如下格式:\n<seed:tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>value_1"
    "</parameter>\n<parameter=example_parameter_2>This is the value for the second parameter\nthat can span\nmultiple "
    "lines</parameter>\n</function>\n</seed:tool_call>\n"
)

LING_TOOL_PROMPT = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)

LFM2_TOOL_PROMPT = "List of tools: <|tool_list_start|>{tool_text}<|tool_list_end|>"

MINICPM5_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>{tool_text}\n</tools>\n\nTool usage guidelines:\n"
    "- You may call zero or more functions. If no function calls are needed, just answer normally "
    "and do not include any <function ... </function>.\n"
    "- When calling a function, return an XML object within <function ... </function> using:\n"
    '<function name="function-name"><param name="param-name">param-value</param></function>\n'
    "- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a "
    'CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param>'
)


@dataclass
class ToolUtils(ABC):
    """Base class for tool utilities."""

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        r"""Generate the system message describing all the available tools."""
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        r"""Generate the assistant message including all the tool calls."""
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class DefaultToolUtils(ToolUtils):
    r"""Default tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return "\n".join([f"Action: {name}\nAction Input: {arguments}" for name, arguments in functions])

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: list[tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append(FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class Gemma4ToolUtils(ToolUtils):
    r"""Gemma-4 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        def _format_parameters(properties: dict[str, Any]) -> str:
            parts: list[str] = []
            for name, schema in properties.items():
                item_parts: list[str] = []
                if schema.get("description"):
                    item_parts.append(f'description:<|"|>{schema["description"]}<|"|>')
                if schema.get("type"):
                    item_parts.append(f'type:<|"|>{str(schema["type"]).upper()}<|"|>')
                parts.append(f"{name}:{{{','.join(item_parts)}}}")

            return ",".join(parts)

        declarations: list[str] = []
        for tool in tools:
            function_data = tool.get("function", tool) if tool.get("type") == "function" else tool
            declaration = (
                f"declaration:{function_data['name']}"
                + "{"
                + f'description:<|"|>{function_data.get("description", "")}<|"|>'
            )

            params = function_data.get("parameters")
            if params:
                param_parts: list[str] = []
                if params.get("properties"):
                    param_parts.append(f"properties:{{{_format_parameters(params['properties'])}}}")

                if params.get("required"):
                    required_text = ",".join(f'<|"|>{item}<|"|>' for item in params["required"])
                    param_parts.append(f"required:[{required_text}]")

                if params.get("type"):
                    param_parts.append(f'type:<|"|>{str(params["type"]).upper()}<|"|>')

                declaration += f",parameters:{{{','.join(param_parts)}}}"

            response_declaration = function_data.get("response")
            if response_declaration:
                response_parts: list[str] = []
                if response_declaration.get("description"):
                    response_parts.append(f'description:<|"|>{response_declaration["description"]}<|"|>')

                response_type = str(response_declaration.get("type", "")).upper()

                if response_type == "OBJECT":
                    response_parts.append(f'type:<|"|>{response_type}<|"|>')

                declaration += f",response:{{{','.join(response_parts)}}}"

            declarations.append(declaration + "}")

        return "\n".join(declarations)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<\|tool_call\>call:([^{\s]+)\{(.*?)\}<tool_call\|>", re.DOTALL)
        matches = re.findall(regex, content)
        if not matches:
            return content

        def _parse_arguments(arg_text: str) -> Any:
            text = arg_text.strip()
            if not text:
                return {}

            # `function_formatter` writes dict arguments as `k:v,...` inside `{...}`.
            # The extractor captures only the inner text, so re-wrap it to parse as JSON object.
            object_like_text = "{" + text + "}"
            # Convert Gemma string markers (<|"|>value<|"|>) to valid JSON strings.
            normalized = re.sub(
                r"<\|\"\|\>(.*?)<\|\"\|\>",
                lambda m: json.dumps(m.group(1), ensure_ascii=False),
                object_like_text,
                flags=re.DOTALL,
            )
            # Quote unquoted object keys so the payload can be parsed by json.loads.
            normalized = re.sub(r"(^|[{\s,])([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', normalized)
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                pass

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        results: list[FunctionCall] = []
        for name, arg_block in matches:
            parsed_arguments = _parse_arguments(arg_block)
            if isinstance(parsed_arguments, str):
                arguments = parsed_arguments
            else:
                arguments = json.dumps(parsed_arguments, ensure_ascii=False)
            results.append(FunctionCall(name.strip(), arguments))

        return results

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        def _format_argument(argument: Any, escape_keys: bool = True) -> str:
            if isinstance(argument, str):
                return f'<|"|>{argument}<|"|>'

            if isinstance(argument, bool):
                return "true" if argument else "false"

            if isinstance(argument, dict):
                items: list[str] = []
                for key in sorted(argument.keys()):
                    formatted_key = f'<|"|>{key}<|"|>' if escape_keys else str(key)
                    formatted_value = _format_argument(argument[key], escape_keys=escape_keys)
                    items.append(f"{formatted_key}:{formatted_value}")
                return "{" + ",".join(items) + "}"

            if isinstance(argument, (list, tuple)):
                return "[" + ",".join(_format_argument(item, escape_keys=escape_keys) for item in argument) + "]"

            if argument is None:
                return "null"

            return str(argument)

        function_texts: list[str] = []
        for function in functions:
            name = function.name
            raw_arguments = function.arguments

            try:
                parsed_arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                parsed_arguments = raw_arguments

            call_text = f"<|tool_call>call:{name}" + "{"
            if isinstance(parsed_arguments, dict):
                args_text = []
                for key in sorted(parsed_arguments.keys()):
                    value_text = _format_argument(parsed_arguments[key], escape_keys=False)
                    args_text.append(f"{key}:{value_text}")

                call_text += ",".join(args_text)
            elif isinstance(parsed_arguments, str):
                call_text += parsed_arguments
            else:
                call_text += _format_argument(parsed_arguments, escape_keys=False)

            call_text += "}<tool_call|>"
            function_texts.append(call_text)

        return "".join(function_texts)


class GLM4ToolUtils(ToolUtils):
    r"""GLM-4 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input.strip())
        except json.JSONDecodeError:
            return content

        return [FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False))]


class Llama3ToolUtils(ToolUtils):
    r"""Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_objects = [{"name": name, "parameters": json.loads(arguments)} for name, arguments in functions]
        return json.dumps(function_objects[0] if len(function_objects) == 1 else function_objects, ensure_ascii=False)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        tools = [tools] if not isinstance(tools, list) else tools
        try:
            return [FunctionCall(tool["name"], json.dumps(tool["parameters"], ensure_ascii=False)) for tool in tools]
        except KeyError:
            return content


class MiniMaxM1ToolUtils(ToolUtils):
    r"""MiniMax-M1 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            tool_text += json.dumps(tool, ensure_ascii=False) + "\n"

        return MINIMAX_M1_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for func in functions:
            name, arguments = func.name, json.loads(func.arguments)
            function_texts.append(json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False))

        return "<tool_calls>\n" + "\n".join(function_texts) + "\n</tool_calls>"

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_calls>\s*(.+?)\s*</tool_calls>", re.DOTALL)
        tool_match = re.search(regex, content)
        if not tool_match:
            return content

        tool_calls_content = tool_match.group(1)
        results = []
        for line in tool_calls_content.split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                tool_call = json.loads(line)
                results.append(FunctionCall(tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False)))
            except json.JSONDecodeError:
                continue

        return results


class MiniMaxM2ToolUtils(ToolUtils):
    r"""MiniMax-M2 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", "") if tool.get("type") == "function" else tool
            tool_text += "<tool>" + json.dumps(tool, ensure_ascii=False) + "</tool>\n"

        return MINIMAX_M2_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for func in functions:
            name, arguments = func.name, json.loads(func.arguments)
            prompt = f'<invoke name="{name}">'
            for key, value in arguments.items():
                prompt += f'\n<parameter name="{key}">'
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += value + "</parameter>"
            prompt += "\n</invoke>"
            function_texts.append(prompt)

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<minimax:tool_call>\s*(.+?)\s*</minimax:tool_call>", re.DOTALL)
        tool_match = re.search(regex, content)
        if not tool_match:
            return content

        tool_calls_content = tool_match.group(1)
        invoke_regex = re.compile(r"<invoke name=\"(.*?)\">(.*?)</invoke>", re.DOTALL)
        results = []

        for func_name, params_block in re.findall(invoke_regex, tool_calls_content):
            args_dict = {}
            param_pattern = re.compile(r"<parameter name=\"(.*?)\">(.*?)</parameter>", re.DOTALL)
            for key, raw_value in re.findall(param_pattern, params_block):
                value = raw_value.strip()
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = raw_value
                args_dict[key] = parsed_value

            results.append(FunctionCall(func_name.strip(), json.dumps(args_dict, ensure_ascii=False)))

        return results


class MiniCPM5ToolUtils(ToolUtils):
    r"""MiniCPM-5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            if tool.get("type") != "function":
                tool = {"type": "function", "function": tool}

            tool_text += "\n" + json.dumps(tool, ensure_ascii=False)

        return MINICPM5_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            prompt = f'<function name="{name}">'
            for key, value in json.loads(arguments).items():
                prompt += f'<param name="{key}">'
                if isinstance(value, str):
                    if "<" in value or "&" in value or "\n" in value:
                        prompt += f"<![CDATA[{value}]]>"
                    else:
                        prompt += value
                else:
                    prompt += str(value)

                prompt += "</param>"

            prompt += "</function>"
            function_texts.append(prompt)

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        results = []
        regex = re.compile(r'<function name="(.*?)">((?:<!\[CDATA\[.*?\]\]>|.)*?)</function>', re.DOTALL)
        for func_name, params_block in re.findall(regex, content):
            args_dict = {}
            param_pattern = re.compile(r'<param name="(.*?)">(<!\[CDATA\[.*?\]\]>|.*?)</param>', re.DOTALL)
            for key, raw_value in re.findall(param_pattern, params_block):
                cdata = re.fullmatch(r"<!\[CDATA\[(.*?)\]\]>", raw_value, re.DOTALL)
                if cdata:
                    args_dict[key] = cdata.group(1)
                    continue

                value = raw_value.strip()
                try:
                    args_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        args_dict[key] = ast.literal_eval(value)
                    except Exception:
                        args_dict[key] = raw_value

            results.append(FunctionCall(func_name, json.dumps(args_dict, ensure_ascii=False)))

        return results if results else content


class MistralToolUtils(ToolUtils):
    r"""Mistral v0.3 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append(tool if tool.get("type") == "function" else {"type": "function", "function": tool})

        return "[AVAILABLE_TOOLS] " + json.dumps(wrapped_tools, ensure_ascii=False) + "[/AVAILABLE_TOOLS]"

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        return json.dumps(
            [{"name": name, "arguments": json.loads(arguments)} for name, arguments in functions], ensure_ascii=False
        )

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        tools = [tools] if not isinstance(tools, list) else tools
        try:
            return [FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)) for tool in tools]
        except KeyError:
            return content


class QwenToolUtils(ToolUtils):
    r"""Qwen 2.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = [
            json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False)
            for name, arguments in functions
        ]
        return "\n".join([f"<tool_call>\n{text}\n</tool_call>" for text in function_texts])

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: list[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


class Qwen35ToolUtils(ToolUtils):
    r"""Qwen 3.5 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_text += "\n" + json.dumps(tool, ensure_ascii=False)

        return QWEN35_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_texts = []
        for func in functions:
            name, arguments = func.name, json.loads(func.arguments)
            prompt = f"<tool_call>\n<function={name}>"
            for key, value in arguments.items():
                prompt += f"\n<parameter={key}>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += f"\n{value}\n</parameter>"
            prompt += "\n</function>\n</tool_call>"
            function_texts.append(prompt)

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        results = []
        regex = re.compile(r"<tool_call>\s*<function=\s*([^\s<>]+)\s*(.*?)\s*</function>\s*</tool_call>", re.DOTALL)
        for func_name, params_block in re.findall(regex, content):
            args_dict = {}
            param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
            for key, raw_value in re.findall(param_pattern, params_block.strip()):
                value = raw_value.strip()
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = raw_value.strip()
                args_dict[key] = parsed_value

            results.append(FunctionCall(func_name.strip(), json.dumps(args_dict, ensure_ascii=False)))

        return results if results else content


class Qwen38ToolUtils(Qwen35ToolUtils):
    r"""Qwen 3.8 tool template preserving the OpenAI function wrapper."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN35_TOOL_PROMPT.format(tool_text=tool_text)


class GLM4MOEToolUtils(QwenToolUtils):
    r"""GLM-4-MOE tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return GLM4_MOE_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_json = [
            {"func_name": name, "func_key_values": json.loads(arguments)} for name, arguments in functions
        ]
        function_texts = []
        for func in function_json:
            prompt = "\n<tool_call>" + func["func_name"]
            for key, value in func["func_key_values"].items():
                prompt += "\n<arg_key>" + key + "</arg_key>"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += "\n<arg_value>" + value + "</arg_value>"
            function_texts.append(prompt)

        return "\n".join(function_texts)


class SeedToolUtils(ToolUtils):
    r"""Seed tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        return SEED_TOOL_PROMPT.format(tool_text="\n" + json.dumps(tools, ensure_ascii=False))

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        function_json = [
            {"func_name": name, "func_key_values": json.loads(arguments)} for name, arguments in functions
        ]
        function_texts = []
        for func in function_json:
            prompt = "\n<seed:tool_call>\n<function=" + func["func_name"]
            for key, value in func["func_key_values"].items():
                prompt += "\n<parameter=" + key + ">"
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                prompt += value + "</parameter>"
            prompt += "\n</function>\n</seed:tool_call>"
            function_texts.append(prompt)

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        results = []
        regex = re.compile(
            r"<seed:tool_call>\s*<function=\s*([^\s<]+)\s*(.*?)\s*</function>\s*</seed:tool_call>", re.DOTALL
        )
        for func_name, params_block in re.findall(regex, content):
            args_dict = {}
            param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
            for key, raw_value in re.findall(param_pattern, params_block.strip()):
                value = raw_value.strip()
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = raw_value
                args_dict[key] = parsed_value

            results.append(FunctionCall(func_name.strip(), json.dumps(args_dict, ensure_ascii=False)))

        return results if results else content


class LingToolUtils(QwenToolUtils):
    r"""Ling v2 tool using template."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return LING_TOOL_PROMPT.format(tool_text=tool_text) + "\n" + "detailed thinking off"


class LFM2ToolUtils(ToolUtils):
    r"""LFM2.5 tool using template with Pythonic function call syntax."""

    @override
    @staticmethod
    def tool_formatter(tools: list[dict[str, Any]]) -> str:
        tool_list = []
        for tool in tools:
            tool = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_list.append(tool)

        return LFM2_TOOL_PROMPT.format(tool_text=json.dumps(tool_list, ensure_ascii=False))

    @override
    @staticmethod
    def function_formatter(functions: list["FunctionCall"]) -> str:
        calls = []
        for name, args_json in functions:
            args = json.loads(args_json)
            kwargs_parts = []
            for key, value in args.items():
                if isinstance(value, str):
                    kwargs_parts.append(f'{key}="{value}"')
                else:
                    kwargs_parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")

            calls.append(f"{name}({', '.join(kwargs_parts)})")

        return f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"

    @staticmethod
    def _ast_to_value(node: ast.AST) -> Any:
        """Convert an AST node to a Python value, handling JSON-style booleans/null."""
        # Handle JSON-style true/false/null as Name nodes
        if isinstance(node, ast.Name):
            if node.id == "true":
                return True
            elif node.id == "false":
                return False
            elif node.id == "null":
                return None
            else:
                raise ValueError(f"Unknown identifier: {node.id}")

        # Use literal_eval for other cases (strings, numbers, lists, dicts)
        return ast.literal_eval(node)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, list["FunctionCall"]]:
        # Extract content between tool call markers
        start_marker = "<|tool_call_start|>"
        end_marker = "<|tool_call_end|>"

        start_idx = content.find(start_marker)
        if start_idx == -1:
            return content

        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            return content

        tool_call_str = content[start_idx + len(start_marker) : end_idx].strip()

        # Parse Pythonic function call syntax using AST
        try:
            tree = ast.parse(tool_call_str, mode="eval")
        except SyntaxError:
            return content

        # Handle both single call and list of calls
        if isinstance(tree.body, ast.List):
            call_nodes = tree.body.elts
        elif isinstance(tree.body, ast.Call):
            call_nodes = [tree.body]
        else:
            return content

        results = []
        for node in call_nodes:
            if not isinstance(node, ast.Call):
                return content

            # Extract function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                return content

            # Extract keyword arguments
            args_dict = {}
            for keyword in node.keywords:
                key = keyword.arg
                try:
                    value = LFM2ToolUtils._ast_to_value(keyword.value)
                except (ValueError, SyntaxError):
                    return content
                args_dict[key] = value

            results.append(FunctionCall(func_name, json.dumps(args_dict, ensure_ascii=False)))

        return results if results else content


TOOLS = {
    "default": DefaultToolUtils(),
    "gemma4": Gemma4ToolUtils(),
    "glm4": GLM4ToolUtils(),
    "llama3": Llama3ToolUtils(),
    "lfm2": LFM2ToolUtils(),
    "minicpm5": MiniCPM5ToolUtils(),
    "minimax1": MiniMaxM1ToolUtils(),
    "minimax2": MiniMaxM2ToolUtils(),
    "mistral": MistralToolUtils(),
    "qwen": QwenToolUtils(),
    "qwen3_5": Qwen35ToolUtils(),
    "qwen3_8": Qwen38ToolUtils(),
    "glm4_moe": GLM4MOEToolUtils(),
    "seed_oss": SeedToolUtils(),
    "ling": LingToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
