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
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from llamafactory.v1.core.rendering import Renderer, create_renderer
from llamafactory.v1.models.deepseek_v4.template import DEEPSEEK_V4_CHAT_TEMPLATE, DeepseekV4Renderer


def message(role, text, reasoning=None):
    blocks = [{"type": "text", "value": text}]
    if reasoning is not None:
        blocks.insert(0, {"type": "reasoning", "value": reasoning})
    return {"role": role, "content": blocks}


def test_chat_matches_official_text_format_without_editing_checkpoint(tokenizer, tmp_path):
    answer = " hello你好"
    before_template = tokenizer.chat_template
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
    assert tokenizer.chat_template == before_template
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
    ("model_type", "custom", "attention"),
    [
        ("deepseek_v4", None, None),
        ("deepseek_v4", "explicit user template", None),
        ("qwen3", None, None),
        ("qwen3", None, "eager"),
        ("qwen3", None, "flash_attention_2"),
    ],
)
def test_model_engine_selects_adapter_and_respects_custom_template(
    tokenizer, monkeypatch, model_type, custom, attention
):
    from llamafactory.v1.config.model_args import ModelArguments

    original_template = tokenizer.chat_template
    from llamafactory.v1.core.model_engine import ModelEngine

    monkeypatch.setattr(ModelEngine, "_init_processor", lambda self: tokenizer)
    monkeypatch.setattr(ModelEngine, "_init_model_config", lambda self: SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(ModelEngine, "_init_model", lambda self: object())
    engine = ModelEngine(ModelArguments(model="unused-local-model", custom_chat_template=custom, flash_attn=attention))
    assert type(engine.renderer) is (DeepseekV4Renderer if model_type == "deepseek_v4" else Renderer)
    assert engine.renderer.position_start == (0 if model_type == "deepseek_v4" else 1)
    assert engine.attn_implementation == (attention or ("eager" if model_type == "deepseek_v4" else "sdpa"))
    assert tokenizer.chat_template == (custom or original_template)


def test_missing_checkpoint_template_does_not_install_chatml(tokenizer):
    tokenizer.chat_template = None
    renderer = DeepseekV4Renderer(tokenizer)
    result = renderer.render_messages([message("user", "hi"), message("assistant", "hello")])
    assert result["input_ids"]
    assert tokenizer.chat_template is None


@pytest.mark.parametrize(
    ("thinking_mode", "drop_thinking", "old_turn", "current_turn"),
    [
        ("chat", True, "</think>hello", "</think>answer"),
        ("thinking", True, "</think>hello", "<think>analysis</think>answer"),
        ("thinking", False, "<think>old analysis</think>hello", "<think>analysis</think>answer"),
    ],
)
def test_multiturn_tokens_match_official_encoding(tokenizer, thinking_mode, drop_thinking, old_turn, current_turn):
    # Golden text from encoding_dsv4.py at DeepSeek-V4-Flash revision 60d8d70770c6776ff598c94bb586a859a38244f1.
    messages = [
        message("system", "system text"),
        message("user", "hi"),
        message("assistant", "hello", reasoning="old analysis"),
        message("user", "second turn"),
        message("assistant", "answer", reasoning="analysis"),
    ]
    result = DeepseekV4Renderer(tokenizer).render_messages(
        messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking
    )
    expected = (
        "<｜begin▁of▁sentence｜>system text<｜User｜>hi<｜Assistant｜>"
        + old_turn
        + "<｜end▁of▁sentence｜><｜User｜>second turn<｜Assistant｜>"
        + current_turn
        + "<｜end▁of▁sentence｜>"
    )
    assert result["input_ids"] == tokenizer.encode(expected, add_special_tokens=False)
    target = ("analysis</think>" if thinking_mode == "thinking" else "") + "answer<｜end▁of▁sentence｜>"
    assert [v for v in result["labels"] if v != -100] == tokenizer.encode(target, add_special_tokens=False)


@pytest.mark.parametrize(("model_type", "custom"), [("deepseek_v4", False), ("deepseek_v4", True), ("qwen3", False)])
def test_position_origin_including_custom_templates_and_preference_pairs(tokenizer, model_type, custom):
    tokenizer.chat_template = DEEPSEEK_V4_CHAT_TEMPLATE
    renderer = create_renderer(tokenizer, SimpleNamespace(model_type=model_type), use_model_template=not custom)
    origin = 0 if model_type == "deepseek_v4" else 1
    messages = [message("user", "hi"), message("assistant", "answer")]
    single, pair = renderer.process_samples(
        [{"messages": messages}, {"chosen_messages": messages, "rejected_messages": messages}]
    )
    assert single["position_ids"] == list(range(origin, origin + len(single["input_ids"])))
    boundary = pair["token_type_ids"].index(2)
    assert pair["position_ids"] == list(range(origin, origin + boundary)) * 2


@pytest.mark.parametrize("attention", [None, "eager", "sdpa", "flash_attention_2"])
def test_v4_attention_capabilities_checked_before_model_loading(tokenizer, monkeypatch, attention):
    from llamafactory.v1.config.model_args import ModelArguments
    from llamafactory.v1.core import model_engine

    requirements = []
    loaded = []
    monkeypatch.setattr(model_engine.ModelEngine, "_init_processor", lambda self: tokenizer)
    monkeypatch.setattr(
        model_engine.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(model_type="deepseek_v4")
    )
    monkeypatch.setattr(model_engine, "check_version", lambda requirement, **kwargs: requirements.append(requirement))
    monkeypatch.setattr(model_engine.ModelEngine, "_init_model", lambda self: loaded.append(self.attn_implementation))
    args = ModelArguments(model="unused-local-model", flash_attn=attention)
    if attention not in (None, "eager"):
        with pytest.raises(ValueError, match="requires.*eager"):
            model_engine.ModelEngine(args)
        assert not loaded
    else:
        model_engine.ModelEngine(args)
        assert loaded == ["eager"]
    assert requirements == ["transformers>=5.16.0"]


def test_v4_old_transformers_is_rejected_during_configuration(tokenizer, monkeypatch):
    from llamafactory.v1.config.model_args import ModelArguments
    from llamafactory.v1.core import model_engine

    monkeypatch.setattr(model_engine.ModelEngine, "_init_processor", lambda self: tokenizer)
    monkeypatch.setattr(
        model_engine.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(model_type="deepseek_v4")
    )

    def reject(requirement, mandatory):
        assert requirement == "transformers>=5.16.0" and mandatory
        raise ImportError("transformers>=5.16.0 required")

    monkeypatch.setattr(model_engine, "check_version", reject)
    monkeypatch.setattr(
        model_engine.ModelEngine, "_init_model", lambda self: pytest.fail("must reject before loading weights")
    )
    with pytest.raises(ImportError, match="5.16"):
        model_engine.ModelEngine(ModelArguments(model="unused-local-model"))
