# Copyright 2026 the LlamaFactory team.
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
from types import SimpleNamespace

import pytest

from llamafactory.extras.constants import AttentionFunction
from llamafactory.model.model_utils.attention import configure_attn_implementation
from llamafactory.model.model_utils.context_parallel import configure_context_parallel_model


def test_configure_gemma4_triton_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    registrations = []
    monkeypatch.setitem(
        sys.modules,
        "gemma_triton_flash_attn",
        SimpleNamespace(register_triton_attention=lambda: registrations.append("triton_gqa")),
    )
    text_config = SimpleNamespace(model_type="gemma4_text", _attn_implementation="eager")
    config = SimpleNamespace(
        model_type="gemma4",
        text_config=text_config,
        _attn_implementation="eager",
        get_text_config=lambda: text_config,
    )
    model_args = SimpleNamespace(flash_attn=AttentionFunction.TRITON_GQA)

    configure_attn_implementation(config, model_args)

    assert registrations == ["triton_gqa"]
    assert config._attn_implementation == {"text_config": "triton_gqa"}
    assert text_config._attn_implementation == "triton_gqa"


def test_reject_triton_attention_for_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gemma_triton_flash_attn", SimpleNamespace())
    config = SimpleNamespace(model_type="llama", get_text_config=lambda: config)
    model_args = SimpleNamespace(flash_attn=AttentionFunction.TRITON_GQA)

    with pytest.raises(ValueError, match="Gemma 4"):
        configure_attn_implementation(config, model_args)


def test_gemma4_triton_attention_is_scoped_to_text_config(monkeypatch: pytest.MonkeyPatch) -> None:
    gemma4_module = pytest.importorskip(
        "transformers.models.gemma4.configuration_gemma4",
        reason="Gemma 4 requires a Transformers version that includes Gemma4Config.",
    )
    Gemma4Config = gemma4_module.Gemma4Config

    monkeypatch.setitem(
        sys.modules,
        "gemma_triton_flash_attn",
        SimpleNamespace(register_triton_attention=lambda: None),
    )
    config = Gemma4Config(vision_config={})

    configure_attn_implementation(config, SimpleNamespace(flash_attn=AttentionFunction.TRITON_GQA))

    assert config._attn_implementation is None
    assert config.text_config._attn_implementation == "triton_gqa"
    assert config.vision_config._attn_implementation is None


def test_configure_gemma4_context_parallel_model(monkeypatch: pytest.MonkeyPatch) -> None:
    registrations = []
    monkeypatch.setitem(
        sys.modules,
        "gemma_triton_flash_attn",
        SimpleNamespace(register_triton_attention_ulysses=registrations.append),
    )
    text_config = SimpleNamespace(
        model_type="gemma4_text",
        _attn_implementation="triton_gqa",
        num_attention_heads=32,
        num_key_value_heads=4,
        num_global_key_value_heads=8,
    )
    config = SimpleNamespace(model_type="gemma4", text_config=text_config, get_text_config=lambda: text_config)
    model = SimpleNamespace(
        config=config,
        set_attn_implementation=lambda implementation: setattr(model, "attn_implementation", implementation),
    )
    finetuning_args = SimpleNamespace(freeze_vision_tower=True, freeze_multi_modal_projector=True)
    process_group = object()

    configure_context_parallel_model(model, process_group, 4, finetuning_args)

    assert registrations == [process_group]
    assert model.attn_implementation == {"text_config": "triton_gqa_ulysses"}


@pytest.mark.parametrize(
    ("model_type", "attention", "freeze_vision_tower", "match"),
    [
        ("llama", "triton_gqa", True, "does not support model type"),
        ("gemma4", "eager", True, "flash_attn: triton_gqa"),
        ("gemma4", "triton_gqa", False, "freeze_vision_tower"),
    ],
)
def test_reject_unsupported_gemma4_context_parallel_model(
    model_type: str,
    attention: str,
    freeze_vision_tower: bool,
    match: str,
) -> None:
    text_config = SimpleNamespace(
        model_type="gemma4_text",
        _attn_implementation=attention,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_global_key_value_heads=8,
    )
    config = SimpleNamespace(model_type=model_type, text_config=text_config, get_text_config=lambda: text_config)
    model = SimpleNamespace(config=config)
    finetuning_args = SimpleNamespace(
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=True,
    )

    with pytest.raises(ValueError, match=match):
        configure_context_parallel_model(model, object(), 4, finetuning_args)
