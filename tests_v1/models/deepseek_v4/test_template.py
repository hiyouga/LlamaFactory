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

"""Exercise the model adapter with a real byte-level BPE tokenizer, offline."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from llamafactory.v1.core.rendering import Renderer, create_renderer
from llamafactory.v1.models.deepseek_v4.template import DeepseekV4Renderer


SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"]
ORIGINAL_TEMPLATE = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}assistant: "


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["user: hi\nassistant: hello\nassistant: ", "system text analysis answer 中文问题你好 second turn"] * 10,
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=SPECIALS[0],
        eos_token=SPECIALS[1],
        pad_token=SPECIALS[1],
        additional_special_tokens=SPECIALS[2:],
        chat_template=ORIGINAL_TEMPLATE,
    )


def message(role, text, reasoning=None):
    blocks = [{"type": "text", "value": text}]
    if reasoning is not None:
        blocks.insert(0, {"type": "reasoning", "value": reasoning})
    return {"role": role, "content": blocks}


def test_chat_matches_experiment_template_without_editing_checkpoint(tokenizer, tmp_path):
    answer = " hello你好"
    tokenizer.save_pretrained(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    renderer = create_renderer(tokenizer, SimpleNamespace(model_type="deepseek_v4"))
    result = renderer.process_samples([{"messages": [message("user", "hi"), message("assistant", answer)]}])[0]
    prompt = "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
    full = prompt + answer + "<｜end▁of▁sentence｜>"
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    assert result["input_ids"] == tokenizer.encode(full, add_special_tokens=False)
    assert result["labels"][: len(prefix)] == [-100] * len(prefix)
    assert result["labels"][len(prefix) :] == result["input_ids"][len(prefix) :]
    assert tokenizer.chat_template == ORIGINAL_TEMPLATE
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_reasoning_is_preserved_and_supervised(tokenizer):
    renderer = DeepseekV4Renderer(tokenizer)
    result = renderer.render_messages([message("user", "hi"), message("assistant", "answer", reasoning="analysis")])
    prefix = tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜><think>", add_special_tokens=False)
    target = tokenizer.encode("analysis</think>answer<｜end▁of▁sentence｜>", add_special_tokens=False)
    assert result["input_ids"] == prefix + target
    assert result["labels"] == [-100] * len(prefix) + target


def test_multiturn_preserves_history_and_supervises_last_answer(tokenizer):
    renderer = DeepseekV4Renderer(tokenizer)
    result = renderer.render_messages(
        [
            message("system", "system text"),
            message("user", "hi"),
            message("assistant", "hello"),
            message("user", "second turn"),
            message("assistant", "answer", reasoning="analysis"),
        ]
    )
    target = tokenizer.encode("analysis</think>answer<｜end▁of▁sentence｜>", add_special_tokens=False)
    assert [x for x in result["labels"] if x != -100] == target
    assert tokenizer.decode(result["input_ids"]).startswith("<｜begin▁of▁sentence｜>system text<｜User｜>hi")


@pytest.mark.parametrize("mode,ending", [("chat", "</think>"), ("thinking", "<think>")])
def test_generation_mode(tokenizer, mode, ending):
    result = DeepseekV4Renderer(tokenizer).render_messages(
        [message("user", "hi")], is_generate=True, thinking_mode=mode
    )
    assert tokenizer.decode(result["input_ids"]).endswith("<｜Assistant｜>" + ending)


def test_missing_v4_token_is_not_added():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    before = len(tokenizer)
    with pytest.raises(ValueError, match="existing special token"):
        DeepseekV4Renderer(tokenizer)
    assert len(tokenizer) == before


def test_tools_are_not_silently_discarded(tokenizer):
    with pytest.raises(ValueError, match="text and reasoning only"):
        DeepseekV4Renderer(tokenizer).render_messages([message("user", "hi")], tools='[{"name":"tool"}]')


@pytest.mark.parametrize(
    ("model_type", "custom"), [("deepseek_v4", None), ("deepseek_v4", "explicit user template"), ("qwen3", None)]
)
def test_model_engine_selects_adapter_and_respects_custom_template(tokenizer, monkeypatch, model_type, custom):
    from llamafactory.v1.config.model_args import ModelArguments
    from llamafactory.v1.core.model_engine import ModelEngine

    monkeypatch.setattr(ModelEngine, "_init_processor", lambda self: tokenizer)
    monkeypatch.setattr(ModelEngine, "_init_model_config", lambda self: SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(ModelEngine, "_init_model", lambda self: object())
    engine = ModelEngine(ModelArguments(model="unused-local-model", custom_chat_template=custom))
    assert type(engine.renderer) is (DeepseekV4Renderer if model_type == "deepseek_v4" and not custom else Renderer)
    assert tokenizer.chat_template == (custom or ORIGINAL_TEMPLATE)


def test_missing_checkpoint_template_does_not_install_chatml(tokenizer):
    tokenizer.chat_template = None
    renderer = DeepseekV4Renderer(tokenizer)
    result = renderer.render_messages([message("user", "hi"), message("assistant", "hello")])
    assert result["input_ids"]
    assert tokenizer.chat_template is None
