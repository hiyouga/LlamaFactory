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
from pathlib import Path

import pytest

from llamafactory.train.megatron_bridge.dataset_export import (
    _example_to_record,
    _inject_generation_block,
    export_dataset_for_megatron_bridge,
    get_sft_dataset_kwargs,
)


def test_inject_generation_block_into_llama_factory_template():
    template = (
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}"
        "{% endfor %}"
    )
    patched = _inject_generation_block(template)
    assert "{% generation %}" in patched
    assert "{% endgeneration %}" in patched
    assert _inject_generation_block(patched) == patched


def test_example_to_record_messages_format():
    example = {
        "_system": "You are helpful.",
        "_prompt": [{"role": "user", "content": "Hello"}],
        "_response": [{"role": "assistant", "content": "Hi there!"}],
    }
    record = _example_to_record(example, stage="sft", use_messages_format=True)
    assert record is not None
    assert record["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert record["messages"][-1] == {"role": "assistant", "content": "Hi there!"}


def test_example_to_record_pretrain_text():
    example = {"_prompt": [{"role": "user", "content": "plain text sample"}]}
    record = _example_to_record(example, stage="pt", use_messages_format=True)
    assert record == {"text": "plain text sample"}


def test_example_to_record_sharegpt_format():
    example = {
        "_system": "System prompt",
        "_prompt": [{"role": "user", "content": "Question"}],
        "_response": [{"role": "assistant", "content": "Answer"}],
    }
    record = _example_to_record(example, stage="sft", use_messages_format=False)
    assert record == {
        "system": "System prompt",
        "conversations": [
            {"from": "User", "value": "Question"},
            {"from": "Assistant", "value": "Answer"},
        ],
        "mask": "User",
    }


def test_export_dataset_for_megatron_bridge(mb_output_dir: Path, mb_model_path: str):
    dataset = [
        {
            "_prompt": [{"role": "user", "content": "Hello"}],
            "_response": [{"role": "assistant", "content": "World"}],
        }
    ]
    export_dir = mb_output_dir / "export"
    export_dataset_for_megatron_bridge(
        train_dataset=dataset,
        output_dir=str(export_dir),
        stage="sft",
        model_name_or_path=mb_model_path,
    )
    train_path = export_dir / "training.jsonl"
    assert train_path.is_file()
    record = json.loads(train_path.read_text(encoding="utf-8").strip())
    if "messages" in record:
        assert record["messages"][-1]["content"] == "World"
    else:
        assert record["conversations"][-1]["value"] == "World"


def test_get_sft_dataset_kwargs_enables_chat():
    kwargs = get_sft_dataset_kwargs()
    assert kwargs["chat"] is True


@pytest.mark.runs_on(["cuda"])
def test_tokenizer_supports_hf_chat_template_on_gpu(mb_model_path: str):
    from llamafactory.train.megatron_bridge.dataset_export import (
        supports_hf_chat_template,
        tokenizer_supports_hf_chat_template,
    )

    assert isinstance(supports_hf_chat_template(), bool)
    assert isinstance(tokenizer_supports_hf_chat_template(mb_model_path), bool)
