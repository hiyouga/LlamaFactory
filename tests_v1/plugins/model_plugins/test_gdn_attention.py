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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llamafactory.v1.plugins.model_plugins.parallelization import gdn_attention


class _KernelHolder(nn.Module):
    def forward(self, inputs):
        return inputs


def test_existing_gdn_kernel_overrides_are_preserved(monkeypatch):
    module = _KernelHolder()
    module.causal_conv1d_fn = None
    module.chunk_gated_delta_rule = lambda *args, **kwargs: None
    original_delta_rule = module.chunk_gated_delta_rule

    def unexpected_lookup(*args):
        pytest.fail("Existing per-instance kernels must not be replaced.")

    monkeypatch.setattr(gdn_attention.inspect, "getmodule", unexpected_lookup)
    gdn_attention._bind_gdn_kernels(module)

    assert module.causal_conv1d_fn is None
    assert module.chunk_gated_delta_rule is original_delta_rule


@pytest.mark.parametrize("fast_kernel", [True, False])
def test_global_gdn_kernels_preserve_fast_kernel_precedence(monkeypatch, fast_kernel):
    convolution = lambda *args, **kwargs: None
    fast_delta_rule = lambda *args, **kwargs: None
    torch_delta_rule = lambda *args, **kwargs: None
    modeling_module = SimpleNamespace(
        causal_conv1d_fn=convolution,
        chunk_gated_delta_rule=fast_delta_rule if fast_kernel else None,
        torch_chunk_gated_delta_rule=torch_delta_rule,
    )
    monkeypatch.setattr(gdn_attention.inspect, "getmodule", lambda _: modeling_module)
    module = _KernelHolder()
    gdn_attention._bind_gdn_kernels(module)

    assert module.causal_conv1d_fn.args[0] is convolution
    assert module.chunk_gated_delta_rule is (fast_delta_rule if fast_kernel else torch_delta_rule)


def test_global_torch_kernel_is_used_when_no_fast_symbol_exists(monkeypatch):
    modeling_module = SimpleNamespace(
        causal_conv1d_fn=lambda *args: None, torch_chunk_gated_delta_rule=lambda *args: None
    )
    monkeypatch.setattr(gdn_attention.inspect, "getmodule", lambda _: modeling_module)
    module = _KernelHolder()
    gdn_attention._bind_gdn_kernels(module)

    assert module.chunk_gated_delta_rule is modeling_module.torch_chunk_gated_delta_rule


def test_missing_kernel_does_not_partially_modify_the_module(monkeypatch):
    monkeypatch.setattr(
        gdn_attention.inspect, "getmodule", lambda _: SimpleNamespace(causal_conv1d_fn=lambda *args: None)
    )
    module = _KernelHolder()
    with pytest.raises(RuntimeError, match="chunk_gated_delta_rule"):
        gdn_attention._bind_gdn_kernels(module)

    assert not hasattr(module, "causal_conv1d_fn")
    assert not hasattr(module, "chunk_gated_delta_rule")


def _test_native_gdn_cp(local_rank, world_size, master_port, checkpoint):
    import copy

    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint as checkpoint_forward
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from llamafactory.v1.plugins.model_plugins.parallelization.ulysses import set_ulysses_sequence_parallel_group
    from llamafactory.v1.utils.pytest import dist_env

    with dist_env(local_rank, world_size, master_port):
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        torch.set_num_threads(2)
        torch.manual_seed(29)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        config = Qwen3_5TextConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        reference = Qwen3_5GatedDeltaNet(config, 0).cuda().float()
        reference.A_log.data.fill_(-5.0)
        reference.dt_bias.data.fill_(0.1)
        parallel = copy.deepcopy(reference)
        container = nn.Module()
        container.block_type = "linear_attention"
        container.linear_attn = parallel
        gdn_attention.apply_gdn_attention(container, world_size)

        inputs = torch.randn(1, 32, 128, device="cuda", requires_grad=True)
        target = torch.randn_like(inputs)
        expected = reference(inputs)
        (expected * target).sum().backward()
        local_inputs = inputs.detach().chunk(world_size, dim=1)[local_rank].contiguous().requires_grad_(True)
        output = (
            checkpoint_forward(parallel, local_inputs, use_reentrant=False) if checkpoint else parallel(local_inputs)
        )
        (output * target.chunk(world_size, dim=1)[local_rank]).sum().backward()

        torch.testing.assert_close(output, expected.chunk(world_size, dim=1)[local_rank], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            local_inputs.grad, inputs.grad.chunk(world_size, dim=1)[local_rank], rtol=3e-4, atol=1e-5
        )
        for name, parameter in parallel.named_parameters():
            gradient = parameter.grad
            dist.all_reduce(gradient)
            torch.testing.assert_close(gradient, reference.get_parameter(name).grad, rtol=3e-4, atol=1e-5)
        dist.destroy_process_group()


@pytest.mark.runs_on(["cuda"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_gdn_cp(checkpoint, monkeypatch):
    import torch.multiprocessing as mp

    from llamafactory.v1.utils.env import find_available_port
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.15.0"):
        pytest.skip("This regression exercises the Transformers >= 5.15 global-kernel API.")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(_test_native_gdn_cp, args=(2, find_available_port(), checkpoint), nprocs=2)


@pytest.mark.parametrize("packed", [False, True])
def test_global_convolution_receives_native_argument_names(monkeypatch, packed):
    captured = {}

    def native_convolution(hidden_states, weight, bias=None, activation=None, **kwargs):
        captured.update(hidden_states=hidden_states, weight=weight, bias=bias, activation=activation, **kwargs)
        return hidden_states

    modeling_module = SimpleNamespace(
        causal_conv1d_fn=native_convolution, torch_chunk_gated_delta_rule=lambda *args: None
    )
    monkeypatch.setattr(gdn_attention.inspect, "getmodule", lambda _: modeling_module)
    module = _KernelHolder()
    gdn_attention._bind_gdn_kernels(module)
    inputs = torch.randn(1, 4, 8)
    weight = torch.randn(4, 3)
    boundaries = torch.tensor([0, 4, 8], dtype=torch.int32) if packed else None
    result = module.causal_conv1d_fn(x=inputs, weight=weight, activation="silu", seq_idx=None, cu_seqlens=boundaries)

    assert result is inputs and captured["hidden_states"] is inputs
    assert captured["activation"] == "silu"
    assert "cu_seqlens" not in captured and "seq_idx" not in captured
    if packed:
        assert captured["cu_seq_lens_q"] is boundaries
    else:
        assert "cu_seq_lens_q" not in captured
