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
from copy import deepcopy
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


def test_rendering_refactor_preserves_existing_template_boundaries():
    class ByteTokenizer:
        bos_token_id = 1000
        eos_token_id = 1001

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return list(text.encode())

        def convert_tokens_to_ids(self, token: str) -> int:
            raise AssertionError(f"Unexpected direct token conversion: {token}")

    tokenizer = ByteTokenizer()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    standard = deepcopy(TEMPLATES["falcon_h1"])
    prompt_ids, response_ids = standard.encode_oneturn(tokenizer, messages, system="system")
    assert prompt_ids == [tokenizer.bos_token_id] + tokenizer.encode(
        "<|im_start|>system\nsystem<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"
    )
    assert response_ids == tokenizer.encode("answer<|im_end|>\n")

    tools = json.dumps(
        [
            {
                "name": "search",
                "description": "Search documents.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
    )
    moss_vl = deepcopy(TEMPLATES["moss_vl"])
    prompt_ids, response_ids = moss_vl.encode_oneturn(tokenizer, messages, tools=tools)
    tool_text = moss_vl.format_tools.apply(content=tools)[0].lstrip("\n")
    assert prompt_ids == tokenizer.encode(
        moss_vl.format_system.apply(content=tool_text)[0]
        + "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"
    )
    assert response_ids == tokenizer.encode("answer<|im_end|>\n")

    llama2 = deepcopy(TEMPLATES["gemma"])
    prompt_ids, response_ids = llama2.encode_oneturn(tokenizer, messages, system="system")
    assert prompt_ids == [tokenizer.bos_token_id] + tokenizer.encode(
        "<start_of_turn>user\nsystem\n\nquestion<end_of_turn>\n<start_of_turn>model\n"
    )
    assert response_ids == tokenizer.encode("answer<end_of_turn>\n")


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
        if enable_thinking:
            answer_str_1 = "<think>\n\n</think>\n\n" + answer_str_1
            answer_str_2 = "<think>\n\n</think>\n\n" + answer_str_2
        else:
            prompt_str_1 = prompt_str_1 + "<think>\n\n</think>\n\n"
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
            prompt_str_1 = prompt_str_1 + "<think>\n\n</think>\n\n"
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
def test_yi_legacy_template_consistency():
    target_models = {
        "Yi-6B-Chat": "01-ai/Yi-6B-Chat",
        "Yi-34B-Chat": "01-ai/Yi-34B-Chat",
    }
    for model_name in target_models:
        assert DEFAULT_TEMPLATE[model_name] == "yi_legacy"

    unaffected_models = [
        "Yi-6B-Chat-4bits",
        "Yi-34B-Chat-8bits",
        "Yi-1.5-6B-Chat",
        "Yi-1.5-34B-Chat",
        "Yi-Coder-1.5B-Chat",
        "Yi-Coder-9B-Chat",
    ]
    for model_name in unaffected_models:
        assert DEFAULT_TEMPLATE[model_name] == "yi"

    assert TEMPLATES["yi_legacy"].__class__.__name__ == "YiLegacyTemplate"
    assert TEMPLATES["yi"].__class__.__name__ == "Template"

    test_cases = [
        (
            [
                {"role": "user", "content": "A box contains 12 red markers and 8 blue markers."},
                {"role": "assistant", "content": "There are 20 markers."},
            ],
            None,
        ),
        (
            [
                {"role": "user", "content": "盒子里有 12 支红笔和 8 支蓝笔，共有多少支？"},
                {"role": "assistant", "content": "一共有 20 支。"},
            ],
            "你是一名简洁的数学助手。",
        ),
        (
            [
                {"role": "user", "content": "A box contains 20 markers."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "移除 5 支后还剩多少支？"},
                {"role": "assistant", "content": "还剩 15 支。"},
            ],
            None,
        ),
    ]
    for model_id in target_models.values():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="yi_legacy"))
        for messages, system in test_cases:
            prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, system, tools=None)
            reference_prefix = ([{"role": "system", "content": system}] if system else []) + messages[:-1]
            reference_prompt_ids = tokenizer.apply_chat_template(
                reference_prefix,
                tokenize=True,
                add_generation_prompt=True,
            )
            reference_messages = ([{"role": "system", "content": system}] if system else []) + messages
            reference_full_ids = tokenizer.apply_chat_template(
                reference_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if hasattr(reference_prompt_ids, "input_ids"):
                reference_prompt_ids = reference_prompt_ids.input_ids
            if hasattr(reference_full_ids, "input_ids"):
                reference_full_ids = reference_full_ids.input_ids

            inference_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
            inference_prompt_ids, _ = template.encode_oneturn(tokenizer, inference_messages, system, tools=None)
            assert prompt_ids == inference_prompt_ids == reference_prompt_ids
            assert prompt_ids + response_ids == reference_full_ids


def test_yi_legacy_template_boundary_merge():
    class BoundaryMergingTokenizer:
        bos_token = None
        eos_token = None

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            token_ids = []
            index = 0
            while index < len(text):
                if text.startswith("\nT", index):
                    token_ids.append(0x110000)
                    index += 2
                else:
                    token_ids.append(ord(text[index]))
                    index += 1

            return token_ids

    tokenizer = BoundaryMergingTokenizer()
    template = TEMPLATES["yi_legacy"]
    messages = [
        {"role": "user", "content": "Count the markers."},
        {"role": "assistant", "content": "There are 20 markers."},
    ]
    rendered_messages = template._render(messages, system=None, tools=None)
    source_text = template._convert_elements_to_text(tokenizer, rendered_messages[0])
    target_text = template._convert_elements_to_text(tokenizer, rendered_messages[1])
    source_ids, target_ids = template._encode(tokenizer, messages, system=None, tools=None)

    assert source_ids == tokenizer.encode(source_text)[:-1]
    assert target_ids[0] == 0x110000
    assert source_ids + target_ids == tokenizer.encode(source_text + target_text)


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
