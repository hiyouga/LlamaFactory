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
from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch.multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM


def _original_fla_op(*args, **kwargs):
    return args, kwargs


class _LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_gated_delta_rule = _original_fla_op
        self.recurrent_gated_delta_rule = _original_fla_op


class _FLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _LinearAttention()


def _apply_kernel(rank) -> None:
    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        with patch.dict(sys.modules, {"torch_npu": MagicMock()}):
            # Reload kernel modules so dependency checks use the mocked NPU environment.
            for k in list(sys.modules.keys()):
                if k.startswith("llamafactory.v1.plugins.model_plugins.kernels"):
                    del sys.modules[k]

            from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_kernels

            model = apply_kernels(model=model, config={"name": "npu_fused_rmsnorm"})

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is original_swiglu_forward.__func__


def _apply_all_kernels(rank) -> None:
    with patch("torch.accelerator.current_accelerator") as mock_get_accelerator:
        mock_device = MagicMock()
        setattr(mock_device, "type", "npu")
        mock_get_accelerator.return_value = mock_device

        model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3")
        original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward
        original_swiglu_forward = model.model.layers[0].mlp.forward

        with patch.dict(sys.modules, {"torch_npu": MagicMock()}):
            # Reload kernel modules so dependency checks use the mocked NPU environment.
            for k in list(sys.modules.keys()):
                if k.startswith("llamafactory.v1.plugins.model_plugins.kernels"):
                    del sys.modules[k]

            from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_kernels

            model = apply_kernels(model=model, config={"name": "auto"})

        assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__
        assert model.model.layers[0].mlp.forward.__func__ is not original_swiglu_forward.__func__


def test_apply_kernel():
    mp.spawn(_apply_kernel)


def test_apply_all_kernels():
    mp.spawn(_apply_all_kernels)


@pytest.mark.runs_on(["npu"])
def test_flash_linear_attention_kernels_compose_with_auto(monkeypatch):
    import fsdp_turbo.ops.fla  # noqa: F401
    from fsdp_turbo.ops import get_op

    from llamafactory.v1.plugins.model_plugins.kernels import interface
    from llamafactory.v1.plugins.model_plugins.kernels.ops.linear_attention.fla import (
        FlashLinearAttentionKernel,
    )

    model = _FLAModel()
    auto_calls = []

    monkeypatch.setattr(
        interface,
        "_apply_auto_kernels",
        lambda model, **kwargs: auto_calls.append((model, kwargs)) or model,
    )
    # FLA execution is outside this bridge test; its external runtime is not required.
    monkeypatch.setattr(FlashLinearAttentionKernel, "check_deps", staticmethod(lambda: None))

    config = {
        "name": "auto, flash-linear-attention",
        "include_kernels": "fused_recurrent_gated_delta_rule, chunk_gated_delta_rule",
        "chunk_size": 32,
    }
    assert interface.apply_kernels(model, config) is model
    assert auto_calls == [(model, {"config": config, "require_logits": False})]
    assert get_op("chunk_gated_delta_rule").__module__ == "fsdp_turbo.ops.fla"

    chunk_op = model.linear_attn.chunk_gated_delta_rule
    assert isinstance(chunk_op, partial)
    assert chunk_op.func.__module__ == "fsdp_turbo.ops.fla"
    assert chunk_op.keywords == {"chunk_size": 32}
    assert model.linear_attn.recurrent_gated_delta_rule.__module__ == "fsdp_turbo.ops.fla"

    with pytest.raises(RuntimeError, match="did not match any model module attributes"):
        FlashLinearAttentionKernel.apply(
            model=nn.Linear(2, 2),
            config={"include_kernels": "chunk_gated_delta_rule", "chunk_size": 32},
        )


def test_flash_linear_attention_kernel_validates_config(monkeypatch):
    from llamafactory.v1.plugins.model_plugins.kernels.ops.linear_attention.fla import (
        FlashLinearAttentionKernel,
    )

    model = nn.Sequential(nn.Linear(2, 2))
    monkeypatch.setattr(FlashLinearAttentionKernel, "check_device", staticmethod(lambda: None))
    monkeypatch.setattr(FlashLinearAttentionKernel, "check_deps", staticmethod(lambda: None))

    with pytest.raises(ValueError, match="chunk_size"):
        FlashLinearAttentionKernel.apply(model=model, config={"include_kernels": "auto", "chunk_size": 48})

    with pytest.raises(ValueError, match="Unsupported Flash Linear Attention kernels"):
        FlashLinearAttentionKernel.apply(model=model, config={"include_kernels": "not_a_kernel"})
