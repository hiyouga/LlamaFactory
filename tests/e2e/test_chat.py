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

import os

import pytest

from llamafactory.chat import ChatModel


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "do_sample": False,
    "max_new_tokens": 1,
}

MESSAGES = [
    {"role": "user", "content": "Hi"},
]

EXPECTED_RESPONSE = "_rho"


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_chat():
    chat_model = ChatModel(INFER_ARGS)
    assert chat_model.chat(MESSAGES)[0].response_text == EXPECTED_RESPONSE


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_stream_chat():
    chat_model = ChatModel(INFER_ARGS)
    response = ""
    for token in chat_model.stream_chat(MESSAGES):
        response += token

    assert response == EXPECTED_RESPONSE
