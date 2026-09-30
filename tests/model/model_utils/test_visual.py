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
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageTextToText

from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.adapter import _setup_freeze_tuning, _setup_full_tuning, init_adapter
from llamafactory.model.model_utils.misc import find_all_linear_modules
from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, autocast_projector_dtype, patch_target_modules


class _MossVLFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="moss_vl",
            text_config=SimpleNamespace(num_hidden_layers=2),
        )
        self.model = torch.nn.Module()
        self.model.separator_token = torch.nn.Parameter(torch.empty(4))
        self.model.visual = torch.nn.Module()
        self.model.visual.pos_embed = torch.nn.Embedding(4, 4)
        self.model.visual.patch_embed = torch.nn.Module()
        self.model.visual.patch_embed.proj = torch.nn.Linear(4, 4)
        self.model.visual.blocks = torch.nn.ModuleList([self._make_block(), self._make_block()])
        self.model.visual.merger = torch.nn.Module()
        self.model.visual.merger.linear_fc1 = torch.nn.Linear(4, 4)
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([self._make_layer(), self._make_layer()])
        self.lm_head = torch.nn.Linear(4, 4)

    @staticmethod
    def _make_block() -> torch.nn.Module:
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv = torch.nn.Linear(4, 4)
        return block

    @staticmethod
    def _make_layer() -> torch.nn.Module:
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 4)
        return layer


@pytest.mark.parametrize("freeze_vision_tower", (False, True))
@pytest.mark.parametrize("freeze_multi_modal_projector", (False, True))
@pytest.mark.parametrize("freeze_language_model", (False, True))
def test_moss_vl_full(
    freeze_vision_tower: bool,
    freeze_multi_modal_projector: bool,
    freeze_language_model: bool,
):
    model = _MossVLFixture()
    finetuning_args = FinetuningArguments(
        finetuning_type="full",
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=freeze_multi_modal_projector,
        freeze_language_model=freeze_language_model,
    )

    _setup_full_tuning(model, finetuning_args, is_trainable=True, cast_trainable_params_to_fp32=False)

    for name, param in model.named_parameters():
        if name.startswith("model.visual.merger") or name == "model.separator_token":
            assert param.requires_grad != freeze_multi_modal_projector
        elif name.startswith("model.visual"):
            assert param.requires_grad != freeze_vision_tower
        else:
            assert param.requires_grad != freeze_language_model


@pytest.mark.parametrize("freeze_multi_modal_projector", (False, True))
def test_moss_vl_freeze(freeze_multi_modal_projector: bool):
    model = _MossVLFixture()
    finetuning_args = FinetuningArguments(
        finetuning_type="freeze",
        freeze_trainable_layers=1,
        freeze_vision_tower=True,
        freeze_multi_modal_projector=freeze_multi_modal_projector,
        freeze_language_model=False,
    )

    _setup_freeze_tuning(model, finetuning_args, is_trainable=True, cast_trainable_params_to_fp32=False)

    assert model.model.separator_token.requires_grad != freeze_multi_modal_projector
    assert model.model.visual.merger.linear_fc1.weight.requires_grad != freeze_multi_modal_projector
    assert model.model.visual.patch_embed.proj.weight.requires_grad is False
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.requires_grad is False
    assert model.model.language_model.layers[1].self_attn.q_proj.weight.requires_grad is True


@pytest.mark.parametrize("freeze_vision_tower", (False, True))
def test_moss_vl_lora_target_all(freeze_vision_tower: bool):
    model = _MossVLFixture()
    finetuning_args = FinetuningArguments(
        finetuning_type="lora",
        lora_target="all",
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
    )

    target_modules = find_all_linear_modules(model, freeze_vision_tower)
    target_modules = patch_target_modules(model, finetuning_args, target_modules)

    assert any(name.startswith("model.language_model") and name.endswith("q_proj") for name in target_modules)
    assert any(name.startswith("model.visual.blocks") and name.endswith("qkv") for name in target_modules) != (
        freeze_vision_tower
    )
    assert all("patch_embed" not in name for name in target_modules)
    assert all("merger" not in name for name in target_modules)
    assert all("lm_head" not in name for name in target_modules)


def test_moss_vl_projector_modules():
    model = _MossVLFixture()
    composite_model = COMPOSITE_MODELS["moss_vl"]

    assert composite_model.projector_keys == ["model.visual.merger", "model.separator_token"]
    assert composite_model.get_projectors(model) == [model.model.visual.merger]


def test_moss_vl_quantized_projector_hook_skips_parameter():
    model = _MossVLFixture()
    model.quantization_method = "bitsandbytes"

    autocast_projector_dtype(model, SimpleNamespace(compute_dtype=torch.float16))

    assert len(model.model.visual.merger._forward_hooks) == 1


@pytest.mark.parametrize("freeze_vision_tower", (False, True))
@pytest.mark.parametrize("freeze_multi_modal_projector", (False, True))
@pytest.mark.parametrize("freeze_language_model", (False, True))
def test_visual_full(freeze_vision_tower: bool, freeze_multi_modal_projector: bool, freeze_language_model: bool):
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-2B-Instruct")
    finetuning_args = FinetuningArguments(
        finetuning_type="full",
        freeze_vision_tower=freeze_vision_tower,
        freeze_multi_modal_projector=freeze_multi_modal_projector,
        freeze_language_model=freeze_language_model,
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable=True)
    for name, param in model.named_parameters():
        if any(key in name for key in ["visual.patch_embed", "visual.blocks"]):
            assert param.requires_grad != freeze_vision_tower
        elif "visual.merger" in name:
            assert param.requires_grad != freeze_multi_modal_projector
        else:
            assert param.requires_grad != freeze_language_model


@pytest.mark.parametrize("freeze_vision_tower,freeze_language_model", ((False, False), (False, True), (True, False)))
def test_visual_lora(freeze_vision_tower: bool, freeze_language_model: bool):
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-2B-Instruct")
    finetuning_args = FinetuningArguments(
        finetuning_type="lora", freeze_vision_tower=freeze_vision_tower, freeze_language_model=freeze_language_model
    )
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable=True)
    trainable_params, frozen_params = set(), set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.add(name)
        else:
            frozen_params.add(name)

    if is_transformers_version_greater_than("4.52.0"):
        visual_param_name = "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"
        language_param_name = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
        merger_param_name = "base_model.model.model.visual.merger.lora_A.default.weight"
    else:
        visual_param_name = "base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"
        language_param_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        merger_param_name = "base_model.model.visual.merger.lora_A.default.weight"

    assert (visual_param_name in trainable_params) != freeze_vision_tower
    assert (language_param_name in trainable_params) != freeze_language_model
    assert (merger_param_name in trainable_params) is False


def test_visual_model_save_load():
    # check VLM's state dict: https://github.com/huggingface/transformers/pull/38385
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-2B-Instruct")
    finetuning_args = FinetuningArguments(finetuning_type="full")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable=False)
    model.to_empty(device="cpu")
    loaded_model_weight = dict(model.named_parameters())

    model.save_pretrained(os.path.join("output", "qwen2_vl"), max_shard_size="10GB", safe_serialization=True)
    saved_model_weight = load_file(os.path.join("output", "qwen2_vl", "model.safetensors"))

    if is_transformers_version_greater_than("4.52.0"):
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in loaded_model_weight
    else:
        assert "model.layers.0.self_attn.q_proj.weight" in loaded_model_weight

    assert "model.layers.0.self_attn.q_proj.weight" in saved_model_weight
