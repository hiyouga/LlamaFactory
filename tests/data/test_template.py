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
import os
from typing import TYPE_CHECKING

import pytest
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.template import TEMPLATES, parse_template
from llamafactory.extras.constants import (
    DEFAULT_TEMPLATE,
    MULTIMODAL_SUPPORTED_MODELS,
    SUPPORTED_MODELS,
    DownloadSource,
)
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import DataArguments


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


HF_TOKEN = os.getenv("HF_TOKEN")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")
TINY_LLAMA4 = os.getenv("TINY_LLAMA4", "llamafactory/tiny-random-Llama-4")

MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "很高兴认识你！"},
]

MESSAGES_WITH_THOUGHT = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "<think>\nModel thought here\n</think>\n\nI am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "<think>\n模型思考内容\n</think>\n\n很高兴认识你！"},
]


def _check_tokenization(
    tokenizer: "PreTrainedTokenizer", batch_input_ids: list[list[int]], batch_text: list[str]
) -> None:
    r"""Check token ids and texts.

    encode(text) == token_ids
    decode(token_ids) == text
    """
    for input_ids, text in zip(batch_input_ids, batch_text):
        assert tokenizer.encode(text, add_special_tokens=False) == input_ids
        assert tokenizer.decode(input_ids) == text


def _check_template(
    model_id: str,
    template_name: str,
    prompt_str: str,
    answer_str: str,
    messages: list[dict[str, str]] = MESSAGES,
) -> None:
    r"""Check template.

    Args:
        model_id: the model id on hugging face hub.
        template_name: the template name.
        prompt_str: the string corresponding to the prompt part.
        answer_str: the string corresponding to the answer part.
        messages: the list of messages.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    content_str = tokenizer.apply_chat_template(messages, tokenize=False)
    content_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if is_transformers_version_greater_than("5.0.0"):
        content_ids = content_ids["input_ids"]

    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=template_name))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, messages)
    assert content_str == prompt_str + answer_str
    assert content_ids == prompt_ids + answer_ids
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


def test_moss_vl_registration():
    model_name = "MOSS-VL-Instruct-0708"

    assert model_name in SUPPORTED_MODELS
    assert SUPPORTED_MODELS[model_name][DownloadSource.DEFAULT] == "OpenMOSS-Team/MOSS-VL-Instruct-0708"
    assert DEFAULT_TEMPLATE[model_name] == "moss_vl"
    assert model_name in MULTIMODAL_SUPPORTED_MODELS
    assert TEMPLATES["moss_vl"].mm_plugin.__class__.__name__ == "MossVLPlugin"
    assert TEMPLATES["moss_vl"].mm_plugin.image_token == "<|image_pad|>"
    assert TEMPLATES["moss_vl"].mm_plugin.video_token == "<|video_pad|>"
    assert TEMPLATES["moss_vl"].mm_plugin.vision_bos_token == "<|vision_start|>"
    assert TEMPLATES["moss_vl"].mm_plugin.vision_eos_token == "<|vision_end|>"
    assert TEMPLATES["moss_vl"].mm_plugin.time_bos_token == "<|time_start|>"
    assert TEMPLATES["moss_vl"].mm_plugin.time_eos_token == "<|time_end|>"


@pytest.mark.runs_on(["cpu", "mps"])
def test_encode_oneturn():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    prompt_str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nI am fine!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = "很高兴认识你！<|eot_id|>"
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.runs_on(["cpu", "mps"])
def test_encode_multiturn():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    encoded_pairs = template.encode_multiturn(tokenizer, MESSAGES)
    prompt_str_1 = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str_1 = "I am fine!<|eot_id|>"
    prompt_str_2 = (
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str_2 = "很高兴认识你！<|eot_id|>"
    _check_tokenization(
        tokenizer,
        (encoded_pairs[0][0], encoded_pairs[0][1], encoded_pairs[1][0], encoded_pairs[1][1]),
        (prompt_str_1, answer_str_1, prompt_str_2, answer_str_2),
    )


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("cot_messages", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_reasoning_encode_oneturn(cot_messages: bool, enable_thinking: bool):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data_args = DataArguments(template="qwen3", enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES_WITH_THOUGHT if cot_messages else MESSAGES)

    prompt_str = (
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        f"{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )
    if not cot_messages or enable_thinking is False:
        answer_str = f"{MESSAGES[3]['content']}<|im_end|>\n"
        if enable_thinking:
            answer_str = "<think>\n\n</think>\n\n" + answer_str
        else:
            prompt_str = prompt_str + "<think>\n\n</think>\n\n"
    else:
        answer_str = f"{MESSAGES_WITH_THOUGHT[3]['content']}<|im_end|>\n"

    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("cot_messages", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_reasoning_encode_multiturn(cot_messages: bool, enable_thinking: bool):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data_args = DataArguments(template="qwen3", enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoded_pairs = template.encode_multiturn(tokenizer, MESSAGES_WITH_THOUGHT if cot_messages else MESSAGES)

    messages = MESSAGES if not cot_messages or enable_thinking is False else MESSAGES_WITH_THOUGHT
    prompt_str_1 = f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    answer_str_1 = f"{messages[1]['content']}<|im_end|>\n"
    prompt_str_2 = f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n<|im_start|>assistant\n"
    answer_str_2 = f"{messages[3]['content']}<|im_end|>\n"
    if not cot_messages or enable_thinking is False:
        # last_query_index logic: only the last user turn (turn 2) gets think tokens
        if enable_thinking:
            answer_str_2 = "<think>\n\n</think>\n\n" + answer_str_2
        else:
            prompt_str_2 = prompt_str_2 + "<think>\n\n</think>\n\n"

    _check_tokenization(
        tokenizer,
        (encoded_pairs[0][0], encoded_pairs[0][1], encoded_pairs[1][0], encoded_pairs[1][1]),
        (prompt_str_1, answer_str_1, prompt_str_2, answer_str_2),
    )


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
@pytest.mark.parametrize("discarding_history_cot", [True, False])
def test_reasoning_encode_multiturn_discarding_history_cot(enable_thinking: bool, discarding_history_cot: bool):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data_args = DataArguments(template="qwen3", enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoded_pairs = template.encode_multiturn(
        tokenizer, MESSAGES_WITH_THOUGHT, discarding_history_cot=discarding_history_cot
    )

    prompt_str_1 = f"<|im_start|>user\n{MESSAGES_WITH_THOUGHT[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    prompt_str_2 = f"<|im_start|>user\n{MESSAGES_WITH_THOUGHT[2]['content']}<|im_end|>\n<|im_start|>assistant\n"

    if enable_thinking is False:
        answer_str_1 = f"{MESSAGES[1]['content']}<|im_end|>\n"
        answer_str_2 = f"{MESSAGES[3]['content']}<|im_end|>\n"
        if discarding_history_cot:
            prompt_str_2 = prompt_str_2 + "<think>\n\n</think>\n\n"
        else:
            # last_query_index logic: only the last user turn (turn 2) gets think tokens
            prompt_str_2 = prompt_str_2 + "<think>\n\n</think>\n\n"
    else:
        if discarding_history_cot:
            answer_str_1 = f"{MESSAGES[1]['content']}<|im_end|>\n"
        else:
            answer_str_1 = f"{MESSAGES_WITH_THOUGHT[1]['content']}<|im_end|>\n"
        answer_str_2 = f"{MESSAGES_WITH_THOUGHT[3]['content']}<|im_end|>\n"

    _check_tokenization(
        tokenizer,
        (encoded_pairs[0][0], encoded_pairs[0][1], encoded_pairs[1][0], encoded_pairs[1][1]),
        (prompt_str_1, answer_str_1, prompt_str_2, answer_str_2),
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_jinja_template():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    tokenizer.chat_template = template._get_jinja_template(tokenizer)  # llama3 template no replace
    assert tokenizer.chat_template != ref_tokenizer.chat_template
    assert tokenizer.apply_chat_template(MESSAGES) == ref_tokenizer.apply_chat_template(MESSAGES)


@pytest.mark.runs_on(["cpu", "mps"])
def test_ollama_modelfile():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert template.get_ollama_modelfile(tokenizer) == (
        "# ollama modelfile auto-generated by llamafactory\n\n"
        "FROM .\n\n"
        'TEMPLATE """<|begin_of_text|>'
        "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}"
        '{{ range .Messages }}{{ if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Content }}'
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        '{{ else if eq .Role "assistant" }}{{ .Content }}<|eot_id|>{{ end }}{{ end }}"""\n\n'
        'PARAMETER stop "<|eom_id|>"\n'
        'PARAMETER stop "<|eot_id|>"\n'
        "PARAMETER num_ctx 4096\n"
    )


@pytest.mark.runs_on(["cpu", "mps"])
def test_get_stop_token_ids():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert set(template.get_stop_token_ids(tokenizer)) == {128008, 128009}


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_gemma_template():
    prompt_str = (
        f"<bos><start_of_turn>user\n{MESSAGES[0]['content']}<end_of_turn>\n"
        f"<start_of_turn>model\n{MESSAGES[1]['content']}<end_of_turn>\n"
        f"<start_of_turn>user\n{MESSAGES[2]['content']}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<end_of_turn>\n"
    _check_template("google/gemma-3-4b-it", "gemma", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_gemma2_template():
    prompt_str = (
        f"<bos><start_of_turn>user\n{MESSAGES[0]['content']}<end_of_turn>\n"
        f"<start_of_turn>model\n{MESSAGES[1]['content']}<end_of_turn>\n"
        f"<start_of_turn>user\n{MESSAGES[2]['content']}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<end_of_turn>\n"
    _check_template("google/gemma-2-2b-it", "gemma2", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_llama3_template():
    prompt_str = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{MESSAGES[0]['content']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{MESSAGES[1]['content']}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{MESSAGES[2]['content']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|eot_id|>"
    _check_template("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
def test_llama4_template():
    prompt_str = (
        f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{MESSAGES[0]['content']}<|eot|>"
        f"<|header_start|>assistant<|header_end|>\n\n{MESSAGES[1]['content']}<|eot|>"
        f"<|header_start|>user<|header_end|>\n\n{MESSAGES[2]['content']}<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|eot|>"
    _check_template(TINY_LLAMA4, "llama4", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
def test_phi4_template():
    prompt_str = (
        f"<|im_start|>user<|im_sep|>{MESSAGES[0]['content']}<|im_end|>"
        f"<|im_start|>assistant<|im_sep|>{MESSAGES[1]['content']}<|im_end|>"
        f"<|im_start|>user<|im_sep|>{MESSAGES[2]['content']}<|im_end|>"
        "<|im_start|>assistant<|im_sep|>"
    )
    answer_str = f"{MESSAGES[3]['content']}<|im_end|>"
    _check_template("microsoft/phi-4", "phi4", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
def test_qwen2_5_template():
    prompt_str = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|im_end|>\n"
    _check_template("Qwen/Qwen2.5-7B-Instruct", "qwen", prompt_str, answer_str)


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("cot_messages", [True, False])
def test_qwen3_template(cot_messages: bool):
    prompt_str = (
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if not cot_messages:
        answer_str = f"<think>\n\n</think>\n\n{MESSAGES[3]['content']}<|im_end|>\n"
        messages = MESSAGES
    else:
        answer_str = f"{MESSAGES_WITH_THOUGHT[3]['content']}<|im_end|>\n"
        messages = MESSAGES_WITH_THOUGHT

    _check_template("Qwen/Qwen3-8B", "qwen3", prompt_str, answer_str, messages=messages)


@pytest.mark.runs_on(["cpu", "mps"])
def test_parse_llama3_template():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = parse_template(tokenizer)
    assert template.format_user.slots == [
        "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
    assert template.format_assistant.slots == ["{{content}}<|eot_id|>"]
    assert template.format_system.slots == ["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    assert template.format_prefix.slots == ["<|begin_of_text|>"]
    assert template.default_system == ""


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
def test_parse_qwen_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    template = parse_template(tokenizer)
    assert template.__class__.__name__ == "Template"
    assert template.format_user.slots == ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    assert template.format_assistant.slots == ["{{content}}<|im_end|>\n"]
    assert template.format_system.slots == ["<|im_start|>system\n{{content}}<|im_end|>\n"]
    assert template.format_prefix.slots == []
    assert template.default_system == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
def test_parse_qwen3_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    template = parse_template(tokenizer)
    assert template.__class__.__name__ == "ReasoningTemplate"
    assert template.format_user.slots == ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    assert template.format_assistant.slots == ["{{content}}<|im_end|>\n"]
    assert template.format_system.slots == ["<|im_start|>system\n{{content}}<|im_end|>\n"]
    assert template.format_prefix.slots == []
    assert template.default_system == ""


# === Tool-call regression tests for qwen3_5/qwen3_6 templates ===
# Verifies that tools_before_system produces tool_text BEFORE system in the encoded output,
# matching native jinja chat_template behavior for Qwen3.5/3.6 models.

TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Beijing?"},
    {"role": "function", "content": '{"name": "get_weather", "arguments": {"city": "Beijing"}}'},
    {"role": "observation", "content": '{"temperature": "25°C", "condition": "sunny"}'},
    {"role": "assistant", "content": "The weather in Beijing is sunny, 25°C."},
]

TOOL_CALL_MULTITURN_MESSAGES = [
    {"role": "user", "content": "What is the weather in Beijing?"},
    {"role": "function", "content": '{"name": "get_weather", "arguments": {"city": "Beijing"}}'},
    {"role": "observation", "content": '{"temperature": "25°C", "condition": "sunny"}'},
    {"role": "assistant", "content": "The weather in Beijing is sunny, 25°C."},
    {"role": "user", "content": "And in Shanghai?"},
    {"role": "function", "content": '{"name": "get_weather", "arguments": {"city": "Shanghai"}}'},
    {"role": "observation", "content": '{"temperature": "30°C", "condition": "cloudy"}'},
    {"role": "assistant", "content": "Shanghai is cloudy, 30°C."},
]

TOOLS_JSON = json.dumps([{
    "name": "get_weather",
    "description": "Get weather info",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}])

SYSTEM_PROMPT = "You are a helpful assistant."


def _build_qwen35_tool_text(tools_json: str) -> str:
    """Reproduce the tool_text that ToolFormatter(tool_format='qwen3_5') would generate."""
    tools = json.loads(tools_json)
    tool_text = ""
    for tool in tools:
        tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
    # QWEN35_TOOL_PROMPT with {tool_text} placeholder
    return (
        "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>" + tool_text
        + "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format"
        " with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
        "that can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: "
        "an inner <function=...></function> block must be nested within"
        " <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language "
        "BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal"
        " with your current knowledge "
        "and do not tell the user about function calls\n</IMPORTANT>"
    )


def _build_qwen35_function_call(name: str, arguments: dict) -> str:
    """Reproduce the function_formatter output for qwen3_5."""
    prompt = f"<tool_call>\n<function={name}>"
    for key, value in arguments.items():
        prompt += f"\n<parameter={key}>"
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        prompt += f"\n{value}\n</parameter>"
    prompt += "\n</function>\n</tool_call>"
    return prompt


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("template_name", ["qwen3_5_nothink", "qwen3_6"])
def test_qwen35_toolcall_oneturn(template_name: str):
    """Regression: tools_before_system puts tool_text before system in system block."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    # Use enable_thinking=False so <think> tokens go into prompt (not answer),
    # keeping expected strings simpler while still testing tools_before_system.
    data_args = DataArguments(template=template_name, enable_thinking=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    prompt_ids, answer_ids = template.encode_oneturn(
        tokenizer, TOOL_CALL_MESSAGES, system=SYSTEM_PROMPT, tools=TOOLS_JSON
    )

    # Build expected strings matching native jinja: tools BEFORE system
    tool_text = _build_qwen35_tool_text(TOOLS_JSON)
    system_content = tool_text.lstrip("\n") + "\n\n" + SYSTEM_PROMPT
    function_call_str = _build_qwen35_function_call("get_weather", {"city": "Beijing"})

    expected_prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{TOOL_CALL_MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{function_call_str}<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n"
        f"{TOOL_CALL_MESSAGES[2]['content']}"
        "\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    # For ReasoningTemplate (qwen3_6), <think>\n\n</think>\n\n is appended to prompt
    if template_name == "qwen3_6":
        expected_prompt += "<think>\n\n</think>\n\n"

    expected_answer = f"{TOOL_CALL_MESSAGES[3]['content']}<|im_end|>\n"

    actual_prompt = tokenizer.decode(prompt_ids)
    actual_answer = tokenizer.decode(answer_ids)

    assert actual_prompt == expected_prompt, (
        f"Prompt mismatch for {template_name}.\n"
        f"Expected:\n{expected_prompt}\n\nActual:\n{actual_prompt}"
    )
    assert actual_answer == expected_answer, (
        f"Answer mismatch for {template_name}.\n"
        f"Expected:\n{expected_answer}\n\nActual:\n{actual_answer}"
    )


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("template_name", ["qwen3_5_nothink", "qwen3_6"])
def test_qwen35_toolcall_oneturn_no_system(template_name: str):
    """Regression: tools_before_system=True with empty system must take the
    `elif self.tools_before_system and tool_text:` branch in template.py
    (line ~156-157), producing system block = tool_text only (no extra
    newlines, no leading empty system text).
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data_args = DataArguments(template=template_name, enable_thinking=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    prompt_ids, answer_ids = template.encode_oneturn(
        tokenizer, TOOL_CALL_MESSAGES, system="", tools=TOOLS_JSON
    )

    # No user-provided system => system block contains only tool_text (lstripped).
    tool_text = _build_qwen35_tool_text(TOOLS_JSON)
    system_content = tool_text.lstrip("\n")
    function_call_str = _build_qwen35_function_call("get_weather", {"city": "Beijing"})

    expected_prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{TOOL_CALL_MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{function_call_str}<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n"
        f"{TOOL_CALL_MESSAGES[2]['content']}"
        "\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if template_name == "qwen3_6":
        expected_prompt += "<think>\n\n</think>\n\n"

    expected_answer = f"{TOOL_CALL_MESSAGES[3]['content']}<|im_end|>\n"

    actual_prompt = tokenizer.decode(prompt_ids)
    actual_answer = tokenizer.decode(answer_ids)

    assert actual_prompt == expected_prompt, (
        f"Prompt mismatch for {template_name} (no-system branch).\n"
        f"Expected:\n{expected_prompt}\n\nActual:\n{actual_prompt}"
    )
    assert actual_answer == expected_answer


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("template_name", ["qwen3_5_nothink", "qwen3_6"])
def test_qwen35_toolcall_multiturn(template_name: str):
    """Regression: multi-turn tool-call encoding with tools_before_system."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    data_args = DataArguments(template=template_name, enable_thinking=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    encoded_pairs = template.encode_multiturn(
        tokenizer, TOOL_CALL_MULTITURN_MESSAGES, system=SYSTEM_PROMPT, tools=TOOLS_JSON
    )

    tool_text = _build_qwen35_tool_text(TOOLS_JSON)
    system_content = tool_text.lstrip("\n") + "\n\n" + SYSTEM_PROMPT
    fc1 = _build_qwen35_function_call("get_weather", {"city": "Beijing"})
    fc2 = _build_qwen35_function_call("get_weather", {"city": "Shanghai"})

    # For ReasoningTemplate (qwen3_6) with enable_thinking=False:
    # <think>\n\n</think>\n\n is appended to prompts in turn_indices.
    # turn_indices are turns >= last_query_index (index 4 = "And in Shanghai?").
    # So turns 3 and 4 (0-indexed pairs 2 and 3) get think tokens in prompt.
    think_suffix = "<think>\n\n</think>\n\n" if template_name == "qwen3_6" else ""

    # Turn 1: user question -> function_call
    expected_prompt_1 = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{TOOL_CALL_MULTITURN_MESSAGES[0]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    expected_answer_1 = f"{fc1}<|im_end|>\n"

    # Turn 2: observation -> assistant reply
    expected_prompt_2 = (
        "<|im_start|>user\n<tool_response>\n"
        f"{TOOL_CALL_MULTITURN_MESSAGES[2]['content']}"
        "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    expected_answer_2 = f"{TOOL_CALL_MULTITURN_MESSAGES[3]['content']}<|im_end|>\n"

    # Turn 3: user follow-up -> function_call (in turn_indices for qwen3_6)
    expected_prompt_3 = (
        f"<|im_start|>user\n{TOOL_CALL_MULTITURN_MESSAGES[4]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n" + think_suffix
    )
    expected_answer_3 = f"{fc2}<|im_end|>\n"

    # Turn 4: observation -> final reply (in turn_indices for qwen3_6)
    expected_prompt_4 = (
        "<|im_start|>user\n<tool_response>\n"
        f"{TOOL_CALL_MULTITURN_MESSAGES[6]['content']}"
        "\n</tool_response><|im_end|>\n<|im_start|>assistant\n" + think_suffix
    )
    expected_answer_4 = f"{TOOL_CALL_MULTITURN_MESSAGES[7]['content']}<|im_end|>\n"

    expected = [
        (expected_prompt_1, expected_answer_1),
        (expected_prompt_2, expected_answer_2),
        (expected_prompt_3, expected_answer_3),
        (expected_prompt_4, expected_answer_4),
    ]

    assert len(encoded_pairs) == 4, f"Expected 4 turn pairs, got {len(encoded_pairs)}"
    for idx, ((prompt_ids, answer_ids), (exp_prompt, exp_answer)) in enumerate(
        zip(encoded_pairs, expected)
    ):
        actual_prompt = tokenizer.decode(prompt_ids)
        actual_answer = tokenizer.decode(answer_ids)
        assert actual_prompt == exp_prompt, (
            f"Turn {idx + 1} prompt mismatch for {template_name}.\n"
            f"Expected:\n{exp_prompt}\n\nActual:\n{actual_prompt}"
        )
        assert actual_answer == exp_answer, (
            f"Turn {idx + 1} answer mismatch for {template_name}.\n"
            f"Expected:\n{exp_answer}\n\nActual:\n{actual_answer}"
        )
