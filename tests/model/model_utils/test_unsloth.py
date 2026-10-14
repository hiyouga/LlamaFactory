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

from unittest.mock import MagicMock

from llamafactory.model.model_utils.unsloth import _coerce_unsloth_target_modules


def _make_model(model_type: str):
    model = MagicMock()
    model.config.model_type = model_type
    return model


def test_coerce_composite_full_paths_to_leaf_names():
    model = _make_model("qwen2_vl")
    target_modules = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.q_proj",
    ]
    assert _coerce_unsloth_target_modules(model, target_modules) == ["k_proj", "q_proj"]


def test_coerce_excludes_vision_tower_paths():
    model = _make_model("qwen2_vl")
    target_modules = [
        "model.language_model.layers.0.self_attn.q_proj",
        "lm_head",
        "model.visual.blocks.0.attn.vision_only_proj",
    ]
    assert _coerce_unsloth_target_modules(model, target_modules) == ["lm_head", "q_proj"]


def test_coerce_uses_model_specific_language_root():
    model = _make_model("dots_ocr")
    target_modules = [
        "model.layers.0.self_attn.q_proj",
        "vision_tower.blocks.0.attn.vision_only_proj",
    ]
    assert _coerce_unsloth_target_modules(model, target_modules) == ["q_proj"]


def test_coerce_string_target_modules():
    model = _make_model("qwen2_vl")
    target_modules = "model.language_model.layers.0.self_attn.q_proj"
    assert _coerce_unsloth_target_modules(model, target_modules) == ["q_proj"]


def test_coerce_set_target_modules():
    model = _make_model("qwen2_vl")
    target_modules = {
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
    }
    assert _coerce_unsloth_target_modules(model, target_modules) == ["k_proj", "q_proj"]


def test_coerce_non_composite_is_noop():
    model = _make_model("llama")
    target_modules = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert _coerce_unsloth_target_modules(model, target_modules) == target_modules


def test_coerce_empty_target_modules_is_noop():
    model = _make_model("qwen2_vl")
    assert _coerce_unsloth_target_modules(model, []) == []


def test_coerce_already_leaf_names_is_noop():
    model = _make_model("qwen2_vl")
    target_modules = ["q_proj", "k_proj"]
    assert _coerce_unsloth_target_modules(model, target_modules) == target_modules
