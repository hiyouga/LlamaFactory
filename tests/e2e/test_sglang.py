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

import sys

import pytest

from llamafactory.chat import ChatModel
from llamafactory.extras.packages import is_sglang_available


MODEL_NAME = "Qwen/Qwen2.5-0.5B"


INFER_ARGS = {
    "model_name_or_path": MODEL_NAME,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "infer_backend": "sglang",
    "do_sample": False,
    "max_new_tokens": 1,
}


MESSAGES = [
    {"role": "user", "content": "Hi"},
]


@pytest.mark.runs_on(["cuda", "xpu"])
@pytest.mark.skipif(not is_sglang_available(), reason="SGLang is not installed")
def test_chat():
    r"""Test the SGLang engine's basic chat functionality."""
    chat_model = ChatModel(INFER_ARGS)
    response = chat_model.chat(MESSAGES)[0]
    # TODO: Change to EXPECTED_RESPONSE
    print(response.response_text)


@pytest.mark.runs_on(["cuda", "xpu"])
@pytest.mark.skipif(not is_sglang_available(), reason="SGLang is not installed")
def test_stream_chat():
    r"""Test the SGLang engine's streaming chat functionality."""
    chat_model = ChatModel(INFER_ARGS)

    response = ""
    for token in chat_model.stream_chat(MESSAGES):
        response += token

    print("Complete response:", response)
    assert response, "Should receive a non-empty response"


# Run tests if executed directly
if __name__ == "__main__":
    if not is_sglang_available():
        print("SGLang is not available. Please install it.")
        sys.exit(1)

    test_chat()
    test_stream_chat()
