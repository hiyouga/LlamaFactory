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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llamafactory.v1.plugins.model_plugins.parallelization import gdn_state, hook
from llamafactory.v1.plugins.model_plugins.parallelization.sequence_parallel import SequenceParallelModelPlugin


@pytest.mark.parametrize("start", [0, 1, 19])
def test_contiguous_batch_rows_remain_independent(start):
    positions = torch.arange(start, start + 8).expand(2, -1)
    boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, None, cp_size=2)

    assert boundaries == [0, 8, 16]
    assert fragments == [[(0, 4), (8, 12)], [(4, 8), (12, 16)]]
    assert max_subsequences == 2


def test_packed_document_can_cross_a_rank_boundary():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, None, cp_size=2)

    assert boundaries == [0, 3, 8]
    assert fragments == [[(0, 4)], [(4, 8)]]
    assert max_subsequences == 2


def test_right_padding_is_one_dummy_sequence_per_tail():
    positions = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0]])
    boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, mask, cp_size=2)

    assert boundaries == [0, 8, 11, 16]
    assert len(fragments[0]) == len(fragments[1]) == 2
    assert max_subsequences == 3


def test_packed_documents_and_right_padding_keep_distinct_boundaries():
    positions = torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, mask, cp_size=2)

    assert boundaries == [0, 2, 5, 8]
    assert fragments == [[(0, 4)], [(4, 8)]]
    assert max_subsequences == 2


def test_all_padding_row_is_a_single_dummy_sequence():
    positions = torch.zeros(1, 8, dtype=torch.long)
    boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, positions, cp_size=2)

    assert boundaries == [0, 8]
    assert fragments == [[(0, 4)], [(4, 8)]]
    assert max_subsequences == 1


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        (torch.ones(3, 1, 4, dtype=torch.long), None, "2-D integer"),
        (torch.arange(4).float().unsqueeze(0), None, "2-D integer"),
        (torch.tensor([[0, 1, 3, 4]]), None, "contiguous text positions"),
        (torch.tensor([[0, 0, 1, 2]]), torch.tensor([[0, 1, 1, 1]]), "right padding"),
        (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 0, 1, 1]]), "right padding"),
        (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 1, 2, 2]]), "binary 2-D"),
        (torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 3), "binary 2-D"),
        (torch.tensor([[0, 1, 2]]), None, "divisible by cp_size"),
    ],
)
def test_unsupported_global_layout_fails_before_collectives(positions, mask, message, monkeypatch):
    def unexpected_collective(*args, **kwargs):
        pytest.fail("Global routing metadata must not use a distributed collective.")

    monkeypatch.setattr(gdn_state.dist, "all_gather", unexpected_collective)
    monkeypatch.setattr(gdn_state.dist, "all_to_all", unexpected_collective)
    with pytest.raises(ValueError, match=message):
        gdn_state._sequence_layout(positions, mask, cp_size=2)


class _RecordingLanguageModel(nn.Module):
    def forward(self, **kwargs):
        return kwargs


def test_routing_is_prepared_once_before_the_shared_split(monkeypatch):
    cp_mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    monkeypatch.setattr(
        hook, "DistributedInterface", lambda: SimpleNamespace(get_device_mesh=lambda _: {"cp": cp_mesh})
    )
    recorded = []

    def make_routing(**kwargs):
        routing = SimpleNamespace(**kwargs)
        recorded.append(routing)
        return routing

    monkeypatch.setattr(gdn_state, "_make_state_routing", make_routing)
    model = nn.Module()
    model.base_model = _RecordingLanguageModel()
    boundary = hook.install_sequence_parallel_hook(model)
    boundary.register_forward_pre_hook(
        partial(gdn_state._prepare_state_inputs, cp_size=2, cp_rank=1, conv_history=3),
        with_kwargs=True,
        prepend=True,
    )
    first = boundary(input_ids=torch.arange(8).unsqueeze(0), position_ids=torch.arange(8).unsqueeze(0))
    second = boundary(input_ids=torch.arange(8).unsqueeze(0), position_ids=torch.tensor([[0, 1, 0, 1, 2, 3, 4, 5]]))

    assert len(recorded) == 2
    assert first["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert first["gdn_cp_routing"].cu_seqlens_global == [0, 8]
    assert second["gdn_cp_routing"].cu_seqlens_global == [0, 2, 8]
    assert first["gdn_cp_routing"] is not second["gdn_cp_routing"]
    assert recorded[0].slots == recorded[1].slots == 1
    assert recorded[0].fragments == recorded[1].fragments == [[(0, 4)], [(4, 8)]]
    assert first["use_cache"] is False


def test_state_mode_with_cp_one_does_not_require_optional_dependencies():
    model = nn.Linear(4, 4)
    original_forward = model.forward
    SequenceParallelModelPlugin("ulysses_gdn_state")(model, 1)

    assert model.forward == original_forward


def test_transformers_decoder_forwards_routing_metadata():
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.9.0"):
        pytest.skip("State-routing CP requires Transformers >= 5.9.")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    class RecordingGDN(nn.Module):
        def forward(self, hidden_states, **kwargs):
            self.routing = kwargs.get("gdn_cp_routing")
            return hidden_states

    config = Qwen3_5TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    decoder = Qwen3_5DecoderLayer(config, 0)
    decoder.linear_attn = RecordingGDN()
    routing = object()
    decoder(
        torch.randn(1, 4, 128),
        position_embeddings=(torch.ones(1, 4, 128), torch.zeros(1, 4, 128)),
        gdn_cp_routing=routing,
    )

    assert decoder.linear_attn.routing is routing


def _assert_relative_close(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
    assert relative_error < 0.03, f"BF16 relative L2 error {relative_error.item()} exceeds 0.03."


def _local_gdn_reference(query, key, value, g, beta, *, cu_seqlens=None, **kwargs):
    from attn_gym.linear import chunk_gdn
    from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm

    options = {"autotune": False}
    query = l2norm(query, cu_seqlens=cu_seqlens, kernel_options=options)
    key = l2norm(key, cu_seqlens=cu_seqlens, kernel_options=options)
    return chunk_gdn(
        query,
        key,
        value,
        g.float(),
        beta.float(),
        scale=query.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        impl="fused",
        kernel_options=options,
    )


def _local_conv_reference(x, weight, bias=None, *, activation=None, **kwargs):
    from attn_gym.linear.short_conv import causal_conv1d

    assert bias is None
    return causal_conv1d(
        x.transpose(1, 2),
        weight.to(x.dtype).contiguous(),
        activation=activation,
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
    ).transpose(1, 2)


def _distributed_layout(case, device, length=190):
    if case == "dense":
        positions = torch.arange(length, device=device).unsqueeze(0)
        return positions, torch.ones_like(positions), [0, length]
    if case == "batch":
        positions = torch.arange(length, device=device).expand(2, -1).contiguous()
        return positions, torch.ones_like(positions), [0, length, 2 * length]
    if case == "packed":
        positions = torch.cat([torch.arange(size, device=device) for size in [19, 61, 110]]).unsqueeze(0)
        return positions, torch.ones_like(positions), [0, 19, 80, 190]
    if case in ("padded", "packed_padded"):
        positions = torch.zeros(2, length, device=device, dtype=torch.long)
        mask = torch.zeros_like(positions)
        for row, sizes in enumerate([[19, 61, 41], [17, 70]] if case == "packed_padded" else [[121], [87]]):
            tokens = sum(sizes)
            positions[row, :tokens] = torch.cat([torch.arange(size, device=device) for size in sizes])
            mask[row, :tokens] = 1
        boundaries = [0, 19, 80, 121, 190, 207, 277, 380] if case == "packed_padded" else [0, 121, 190, 277, 380]
        return positions, mask, boundaries
    raise ValueError(f"Unknown test layout {case!r}.")


def _test_state_gdn_distributed(
    local_rank,
    world_size,
    master_port,
    case,
    checkpoint,
    key_heads=2,
    value_heads=6,
    length=190,
    freeze_parameters=False,
    loss_on_earlier_rank=False,
    allocator_poison=None,
):
    import copy

    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint as checkpoint_forward
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from llamafactory.v1.utils.pytest import dist_env

    with dist_env(local_rank, world_size, master_port):
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        torch.set_num_threads(2)
        torch.manual_seed(73)
        device = torch.device("cuda", local_rank)
        config = Qwen3_5TextConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_num_key_heads=key_heads,
            linear_num_value_heads=value_heads,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        reference = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, 0).to(device).float()
        reference.A_log.data.fill_(-5.0)
        reference.dt_bias.data.fill_(0.1)
        if freeze_parameters:
            reference.requires_grad_(False)
        parallel = copy.deepcopy(reference)
        container = nn.Module()
        container.block_type = "linear_attention"
        container.linear_attn = parallel
        gdn_state.apply_gdn_state_attention(container, world_size, dist.group.WORLD, nn.Identity())

        # Keep native Transformers projections/norm and replace only its local kernels.
        modeling_qwen3_5.causal_conv1d_fn = _local_conv_reference
        modeling_qwen3_5.torch_chunk_gated_delta_rule = _local_gdn_reference
        if hasattr(reference, "causal_conv1d_fn"):
            reference.causal_conv1d_fn = _local_conv_reference
            reference.chunk_gated_delta_rule = _local_gdn_reference
        positions, mask, expected_boundaries = _distributed_layout(case, device, length)
        batch_size, length = positions.shape
        boundaries, fragments, max_subsequences, _ = gdn_state._sequence_layout(positions, mask, world_size)
        assert boundaries == expected_boundaries
        routing = gdn_state._make_state_routing(
            cu_seqlens_global=boundaries,
            fragments=fragments,
            cp_rank=local_rank,
            device=device,
            slots=batch_size,
            max_subsequences=max_subsequences,
            conv_history=3,
        )
        inputs = torch.randn(batch_size, length, 128, device=device, requires_grad=True)
        target = torch.zeros_like(inputs)
        loss_start = 0 if loss_on_earlier_rank else length // 2
        target[:, loss_start : loss_start + 16] = torch.randn_like(target[:, loss_start : loss_start + 16])
        target = target * mask.unsqueeze(-1)
        flat_inputs = inputs.reshape(1, batch_size * length, 128)
        flat_mask = mask.reshape(1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Independent native forwards reset both convolution and recurrent history.
            # The reference therefore does not rely on a kernel honoring packed offsets.
            expected = torch.cat(
                [
                    reference(flat_inputs[:, start:stop], attention_mask=flat_mask[:, start:stop])
                    for start, stop in zip(expected_boundaries[:-1], expected_boundaries[1:], strict=True)
                ],
                dim=1,
            ).reshape_as(inputs)
        (expected * target).sum().backward()

        if allocator_poison is not None:
            _poison_state_allocator(length // world_size, key_heads, value_heads, allocator_poison, device)
        local_inputs = inputs.detach().chunk(world_size, dim=1)[local_rank].contiguous().requires_grad_(True)
        local_mask = mask.chunk(world_size, dim=1)[local_rank].contiguous()
        forward = partial(parallel, attention_mask=local_mask, gdn_cp_routing=routing)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = (
                checkpoint_forward(forward, local_inputs, use_reentrant=False) if checkpoint else forward(local_inputs)
            )
        (output * target.chunk(world_size, dim=1)[local_rank]).sum().backward()

        valid = local_mask.bool()
        _assert_relative_close(output[valid], expected.chunk(world_size, dim=1)[local_rank][valid])
        _assert_relative_close(local_inputs.grad, inputs.grad.chunk(world_size, dim=1)[local_rank])
        if loss_on_earlier_rank and local_rank > 0:
            torch.testing.assert_close(local_inputs.grad, torch.zeros_like(local_inputs.grad), rtol=0, atol=0)
        if local_rank == 0:
            assert inputs.grad[:, : length // 2].norm() > 0, "The earlier rank must receive a nonzero gradient."
        actual_gradients, reference_gradients = [], []
        for parameter, reference_parameter in zip(parallel.parameters(), reference.parameters(), strict=True):
            if freeze_parameters:
                assert parameter.grad is None and reference_parameter.grad is None
                continue
            gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            dist.all_reduce(gradient)
            actual_gradients.append(gradient.flatten())
            reference_gradients.append(reference_parameter.grad.flatten())
        if not freeze_parameters:
            _assert_relative_close(torch.cat(actual_gradients), torch.cat(reference_gradients))
        if case in ("padded", "packed_padded"):
            torch.testing.assert_close(local_inputs.grad[1], torch.zeros_like(local_inputs.grad[1]), rtol=0, atol=0)
        if case in ("packed", "packed_padded") and local_rank == 0:
            torch.testing.assert_close(
                local_inputs.grad[0, :80], torch.zeros_like(local_inputs.grad[0, :80]), rtol=0, atol=0
            )
        if case == "packed" and not checkpoint:
            changed = local_inputs.detach().clone()
            if local_rank == 0:
                changed[:, :19] += 10
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                changed_output = forward(changed)
            local_start = max(80 - local_rank * (length // world_size), 0)
            torch.testing.assert_close(changed_output[:, local_start:], output[:, local_start:], rtol=0, atol=0)
        dist.destroy_process_group()


@pytest.mark.runs_on(["cuda"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize("case", ["dense", "batch", "packed", "padded", "packed_padded"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_state_gdn_distributed(case, checkpoint, monkeypatch):
    import torch.multiprocessing as mp

    from llamafactory.v1.utils.env import find_available_port
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.9.0"):
        pytest.skip("State-routing CP requires Transformers >= 5.9.")
    pytest.importorskip("attn_gym.linear.gdn.context_parallel")
    pytest.importorskip("attn_gym.linear.short_conv.cute")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(_test_state_gdn_distributed, args=(2, find_available_port(), case, checkpoint), nprocs=2)


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        (torch.tensor([[0, 1, 0, 0]]), torch.tensor([[1, 1, 1, 0]]), "padding mask"),
        (torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]), None, "flattened batch"),
    ],
)
def test_complete_mode_rejects_layouts_not_supported_by_full_attention(positions, mask, message, monkeypatch):
    def unexpected_routing(**kwargs):
        pytest.fail("Unsupported full-attention layouts must fail before routing or model collectives.")

    monkeypatch.setattr(gdn_state, "_make_state_routing", unexpected_routing)
    kwargs = {"input_ids": torch.ones_like(positions), "position_ids": positions, "attention_mask": mask}
    with pytest.raises(ValueError, match=message):
        gdn_state._prepare_state_inputs(nn.Identity(), (), kwargs, cp_size=2, cp_rank=0, conv_history=3)


def test_packed_all_ones_mask_is_removed_before_full_attention(monkeypatch):
    monkeypatch.setattr(gdn_state, "_make_state_routing", lambda **kwargs: SimpleNamespace(**kwargs))
    positions = torch.tensor([[0, 1, 0, 1]])
    kwargs = {
        "input_ids": torch.ones_like(positions),
        "position_ids": positions,
        "attention_mask": torch.ones_like(positions),
    }
    _, prepared = gdn_state._prepare_state_inputs(nn.Identity(), (), kwargs, cp_size=2, cp_rank=0, conv_history=3)

    assert prepared["attention_mask"] is None
    assert prepared["gdn_cp_routing"].cu_seqlens_global == [0, 2, 4]


def test_unpacked_right_padding_mask_is_retained(monkeypatch):
    monkeypatch.setattr(gdn_state, "_make_state_routing", lambda **kwargs: SimpleNamespace(**kwargs))
    positions = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    kwargs = {"input_ids": torch.ones_like(positions), "position_ids": positions, "attention_mask": mask}
    _, prepared = gdn_state._prepare_state_inputs(nn.Identity(), (), kwargs, cp_size=2, cp_rank=0, conv_history=3)

    assert prepared["attention_mask"] is mask
    assert prepared["gdn_cp_routing"].cu_seqlens_global == [0, 3, 4, 6, 8]


@pytest.mark.runs_on(["cuda"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize("freeze_parameters", [False, True])
def test_state_gdn_model_head_count(freeze_parameters, monkeypatch):
    import torch.multiprocessing as mp

    from llamafactory.v1.utils.env import find_available_port
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.9.0"):
        pytest.skip("State-routing CP requires Transformers >= 5.9.")
    pytest.importorskip("attn_gym.linear.gdn.context_parallel")
    pytest.importorskip("attn_gym.linear.short_conv.cute")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(
        _test_state_gdn_distributed,
        args=(2, find_available_port(), "dense", True, 16, 48, 2048, freeze_parameters),
        nprocs=2,
    )


def _test_hybrid_model_routing(local_rank, world_size, master_port):
    import copy

    import torch.distributed as dist
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from llamafactory.v1.accelerator.interface import DistributedInterface
    from llamafactory.v1.config.training_args import TrainingArguments
    from llamafactory.v1.utils.pytest import dist_env

    with dist_env(local_rank, world_size, master_port):
        torch.cuda.set_device(local_rank)
        torch.set_num_threads(2)
        torch.manual_seed(31)
        DistributedInterface(TrainingArguments(cp_size=world_size, dp_size=1))
        device = torch.device("cuda", local_rank)
        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            layer_types=["linear_attention", "linear_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        reference = modeling_qwen3_5.Qwen3_5ForCausalLM(config).to(device).float().train()
        parallel = copy.deepcopy(reference)
        for model in (reference, parallel):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # All-linear layers exercise the real HF model/hook/checkpoint chain without an FA2 dependency.
        SequenceParallelModelPlugin("ulysses_gdn_state")(parallel, world_size)
        modeling_qwen3_5.causal_conv1d_fn = _local_conv_reference
        modeling_qwen3_5.torch_chunk_gated_delta_rule = _local_gdn_reference
        for module in reference.modules():
            if hasattr(module, "causal_conv1d_fn"):
                module.causal_conv1d_fn = _local_conv_reference
                module.chunk_gated_delta_rule = _local_gdn_reference

        length = 128
        reference_inputs = [torch.randn(1, length, 128, device=device, requires_grad=True) for _ in range(2)]
        parallel_inputs = [inputs.detach().clone().requires_grad_(True) for inputs in reference_inputs]
        targets = [torch.randn(1, length, 64, device=device) for _ in range(2)]
        expected_outputs, actual_outputs = [], []
        for index, boundaries in enumerate(([0, length], [0, 32, length])):
            positions = torch.cat(
                [torch.arange(stop - start, device=device) for start, stop in zip(boundaries[:-1], boundaries[1:])]
            ).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = torch.cat(
                    [
                        reference(
                            inputs_embeds=reference_inputs[index][:, start:stop],
                            position_ids=torch.arange(stop - start, device=device).unsqueeze(0),
                            use_cache=False,
                        ).logits
                        for start, stop in zip(boundaries[:-1], boundaries[1:])
                    ],
                    dim=1,
                )
                actual = parallel(inputs_embeds=parallel_inputs[index], position_ids=positions, use_cache=False).logits
            expected_outputs.append(expected)
            actual_outputs.append(actual)
            _assert_relative_close(actual, expected.chunk(world_size, dim=1)[local_rank])

        # Two different routing objects are live before either checkpointed backward.
        sum((output * target).sum() for output, target in zip(expected_outputs, targets)).backward()
        sum(
            (output * target.chunk(world_size, dim=1)[local_rank]).sum()
            for output, target in zip(actual_outputs, targets)
        ).backward()
        for actual_input, reference_input in zip(parallel_inputs, reference_inputs):
            _assert_relative_close(
                actual_input.grad.chunk(world_size, dim=1)[local_rank],
                reference_input.grad.chunk(world_size, dim=1)[local_rank],
            )
        actual_gradients, reference_gradients = [], []
        for name, parameter in parallel.named_parameters():
            expected_gradient = reference.get_parameter(name).grad
            if expected_gradient is None:
                assert parameter.grad is None
                continue
            gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            dist.all_reduce(gradient)
            actual_gradients.append(gradient.flatten())
            reference_gradients.append(expected_gradient.flatten())
        _assert_relative_close(torch.cat(actual_gradients), torch.cat(reference_gradients))
        dist.destroy_process_group()


@pytest.mark.runs_on(["cuda"])
@pytest.mark.require_distributed(2)
def test_hf_model_routing_survives_checkpoint_recomputation(monkeypatch):
    import torch.multiprocessing as mp

    from llamafactory.v1.utils.env import find_available_port
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.9.0"):
        pytest.skip("State-routing CP requires Transformers >= 5.9.")
    pytest.importorskip("attn_gym.linear.gdn.context_parallel")
    pytest.importorskip("attn_gym.linear.short_conv.cute")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(_test_hybrid_model_routing, args=(2, find_available_port()), nprocs=2)


def test_state_routing_metadata_does_not_reach_the_full_attention_kernel(monkeypatch):
    from llamafactory.v1.plugins.model_plugins.parallelization import ulysses

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", lambda _, tensor, *__: tensor)
    monkeypatch.setattr(ulysses, "get_ulysses_sequence_parallel_world_size", lambda _: 2)

    def fake_all_gather(outputs, tensor, **kwargs):
        for rank, output in enumerate(outputs):
            output.copy_(tensor + rank * 3 if tensor.shape == (1, 3) else tensor)

    def native_attention(
        query,
        key,
        value,
        attention_mask,
        *,
        query_length,
        is_causal,
        dropout,
        position_ids,
        softmax_scale,
        deterministic,
        target_dtype,
    ):
        assert query_length == 6 and is_causal
        assert position_ids.tolist() == [[0, 1, 2, 3, 4, 5]]
        return query

    monkeypatch.setattr(ulysses.dist, "all_gather", fake_all_gather)
    routing = object()
    forward_kwargs = {"gdn_cp_routing": routing, "position_ids": torch.arange(3).unsqueeze(0)}
    query = torch.zeros(1, 3, 2, 4)
    output = ulysses.new_flash_attn_forward(
        query,
        query,
        query,
        None,
        sequence_parallel_size=2,
        group=object(),
        attn_fn=native_attention,
        **forward_kwargs,
    )

    torch.testing.assert_close(output, query)
    assert forward_kwargs["gdn_cp_routing"] is routing


def _poison_state_allocator(local_tokens, key_heads, value_heads, value, device):
    """Fill and release likely scratch-buffer sizes without changing model inputs."""
    sizes = [
        (value_heads * 256 * 128, torch.float32),
        (value_heads * 128 * 128, torch.float32),
        (local_tokens * value_heads, torch.float32),
        (local_tokens * key_heads * 128, torch.bfloat16),
        (local_tokens * value_heads * 128, torch.bfloat16),
        (local_tokens * value_heads * 64, torch.bfloat16),
    ]
    buffers = [torch.full((size,), value, dtype=dtype, device=device) for size, dtype in sizes for _ in range(2)]
    torch.cuda.synchronize()
    del buffers


@pytest.mark.runs_on(["cuda"])
@pytest.mark.require_distributed(2)
@pytest.mark.parametrize("allocator_poison", [None, float("nan"), 1e20], ids=["clean", "nan_pool", "large_pool"])
def test_state_gdn_zero_successor_cotangent(allocator_poison, monkeypatch):
    import torch.multiprocessing as mp

    from llamafactory.v1.utils.env import find_available_port
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.9.0"):
        pytest.skip("State-routing CP requires Transformers >= 5.9.")
    pytest.importorskip("attn_gym.linear.gdn.context_parallel")
    pytest.importorskip("attn_gym.linear.short_conv.cute")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(
        _test_state_gdn_distributed,
        args=(2, find_available_port(), "dense", True, 16, 48, 2048, True, True, allocator_poison),
        nprocs=2,
    )


@pytest.mark.parametrize("module_name", ["torch.distributed", "torch.distributed._functional_collectives"])
def test_state_routing_rejects_missing_collective_apis(module_name, monkeypatch):
    import importlib

    monkeypatch.setattr(gdn_state, "is_transformers_version_greater_than", lambda _: True)
    monkeypatch.delattr(importlib.import_module(module_name), "all_gather_single", raising=False)
    with pytest.raises(RuntimeError, match="PyTorch >= 2.13.*all_gather_single"):
        gdn_state.apply_gdn_state_attention(nn.Identity(), 2, None, nn.Identity())
