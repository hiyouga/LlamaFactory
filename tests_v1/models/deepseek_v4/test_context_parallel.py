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

import copy
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llamafactory.v1.models.deepseek_v4 import context_parallel as sequence_parallel


hf = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")


def test_deepseek_v4_cp_install_is_instance_scoped_without_backend(monkeypatch):
    import builtins

    from transformers import DeepseekV4Config
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf

    config = DeepseekV4Config(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        o_groups=2,
        o_lora_rank=8,
        num_hidden_layers=1,
        compress_ratios=[0],
    )
    config._attn_implementation = "eager"
    model = hf.DeepseekV4Attention(config, 0)
    other = hf.DeepseekV4Attention(config, 0)
    original_eager = hf.eager_attention_forward
    original_forward = hf.DeepseekV4Attention.forward
    original_import = builtins.__import__

    def import_without_backend(name, *args, **kwargs):
        if name.startswith("fsdp_turbo"):
            raise AssertionError("LF context parallelism must not import FSDPTurbo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_backend)
    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda _: 2)
    group = object()
    sequence_parallel.apply_deepseek_v4_sequence_parallel(model, group)
    installed = model.forward
    sequence_parallel.apply_deepseek_v4_sequence_parallel(model, group)
    assert model.forward is installed
    assert hf.eager_attention_forward is original_eager
    assert hf.DeepseekV4Attention.forward is original_forward
    assert other.forward.__func__ is original_forward
    with pytest.raises(ValueError, match="different group"):
        sequence_parallel.apply_deepseek_v4_sequence_parallel(model, object())


@pytest.fixture
def v4_cp_model(monkeypatch):
    from llamafactory.v1.plugins.model_plugins.parallelization import hook
    from llamafactory.v1.plugins.model_plugins.parallelization import sequence_parallel as routing

    class Mesh:
        def __getitem__(self, name):
            return self

        def size(self):
            return 2

        def get_local_rank(self):
            return 1

    class Core(torch.nn.Module):
        calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            return kwargs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="deepseek_v4", _attn_implementation="eager")
            self.base_model = Core()

    interface = SimpleNamespace(get_device_mesh=lambda _: Mesh(), get_group=lambda _: object())
    monkeypatch.setattr(hook, "DistributedInterface", lambda: interface)
    monkeypatch.setattr(routing, "DistributedInterface", lambda: interface)
    monkeypatch.setattr(sequence_parallel, "apply_deepseek_v4_sequence_parallel", lambda *args: None)
    model = Model()
    routing.apply_sequence_parallel(model, 2)
    return model


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        ([[0, 0, 1, 2]], None, "packed sequences"),
        ([[0, 1, 3, 4]], None, "contiguous zero-based positions"),
        ([[1, 2, 3, 4]], None, "zero-based positions"),
        ([[[0, 1, 2, 3]]], None, "2D position_ids"),
        ([[0, 1, 2, 3]], [[[[0.0] * 4] * 4]], "custom masks"),
        ([[0, 1, 2, 3]], [[1.0, 0.5, 1.0, 1.0]], "binary 2D padding mask"),
    ],
)
def test_deepseek_v4_rejects_unsupported_inputs_before_language_forward(v4_cp_model, positions, mask, message):
    model = v4_cp_model
    positions = torch.tensor(positions)
    with pytest.raises(ValueError, match=message):
        model.base_model(
            input_ids=torch.arange(positions.shape[-1]).unsqueeze(0),
            position_ids=positions,
            attention_mask=None if mask is None else torch.tensor(mask),
        )
    assert model.base_model.calls == 0


def test_deepseek_v4_rejects_packed_boundary_between_cp_ranks(v4_cp_model):
    from llamafactory.v1.plugins.trainer_plugins.batching import _pack_padding_free_samples

    sample = {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4], "loss_weights": [1.0] * 4}
    packed = _pack_padding_free_samples([sample, sample], cutoff_len=4)
    assert packed["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    model = v4_cp_model
    with pytest.raises(ValueError, match="packed sequences"):
        model.base_model(**{key: packed[key] for key in ("input_ids", "attention_mask", "position_ids")})
    assert model.base_model.calls == 0


@pytest.mark.parametrize(
    ("positions", "mask"),
    [
        ([[0, 1, 2, 3]], None),
        ([[0, 1, 0, 0]], [[1, 1, 0, 0]]),
        ([[0, 1, 2, 3]], [[0, 0, 1, 1]]),
        ([[0, 0, 0, 0]], [[0, 0, 0, 0]]),
        ([[0, 1, 2, 3]], [[1, 1, 1, 0], [0, 1, 1, 1]]),
    ],
)
def test_deepseek_v4_accepts_contiguous_positions_with_ordinary_padding(v4_cp_model, positions, mask):
    model = v4_cp_model
    positions = torch.tensor(positions)
    mask = None if mask is None else torch.tensor(mask)
    batch_size = positions.shape[0] if mask is None else mask.shape[0]
    batch = model.base_model(
        input_ids=torch.arange(4).expand(batch_size, -1), position_ids=positions, attention_mask=mask
    )
    torch.testing.assert_close(batch["position_ids"], positions[:, 2:])
    assert batch["input_ids"].shape == (batch_size, 2)
    assert batch["use_cache"] is False
    assert model.base_model.calls == 1


def _check_attention(rank, ratio, length):
    torch.manual_seed(42)
    config = hf.DeepseekV4Config(
        hidden_size=32,
        head_dim=16,
        q_lora_rank=16,
        num_attention_heads=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        partial_rotary_factor=0.5,
        num_hidden_layers=1,
        compress_ratios=[ratio],
        sliding_window=16,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference = hf.DeepseekV4Attention(config, 0).double()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, mean=0.1, std=0.02)
    candidate = copy.deepcopy(reference)
    hidden = torch.randn(2, length, 32, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(length).unsqueeze(0)
    rotary = hf.DeepseekV4RotaryEmbedding(config)

    def embeddings(states, pos):
        if hasattr(reference, "rope_layer_type"):
            return {kind: rotary(states, position_ids=pos, layer_type=kind) for kind in ("main", "compress")}
        return rotary(states, position_ids=pos, layer_type="main")

    query_pos = positions.view(-1, 1)
    key_pos = positions.view(1, -1)
    allowed = ((key_pos <= query_pos) & (key_pos > query_pos - 16)).view(1, 1, length, length).expand(2, -1, -1, -1)
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[1, -5:] = False
    mask = torch.zeros(2, 1, length, length, dtype=hidden.dtype).masked_fill(
        ~(allowed & valid[:, None, None, :]), torch.finfo(hidden.dtype).min
    )
    expected, _ = reference(hidden, embeddings(hidden, positions), positions, mask)
    expected.square().sum().backward()

    local_hidden = hidden.detach().chunk(2, dim=1)[rank].contiguous().requires_grad_()
    local_positions = positions.chunk(2, dim=-1)[rank].contiguous()
    local_mask = mask.chunk(2, dim=-2)[rank].chunk(2, dim=-1)[rank].contiguous()
    sequence_parallel.apply_deepseek_v4_sequence_parallel(candidate, dist.group.WORLD)
    actual, _ = candidate(local_hidden, embeddings(local_hidden, local_positions), local_positions, local_mask)
    torch.testing.assert_close(actual, expected.detach().chunk(2, dim=1)[rank], atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    torch.testing.assert_close(local_hidden.grad, hidden.grad.chunk(2, dim=1)[rank], atol=2e-6, rtol=2e-5)
    for (name, parameter), (_, ref_parameter) in zip(candidate.named_parameters(), reference.named_parameters()):
        if ref_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-5, rtol=2e-4, msg=name)

    if (ratio, length) in ((4, 132), (128, 132)):
        future = ratio - 1
        changed = hidden.detach().clone()
        changed[:, future] += torch.linspace(-3, 3, 32, dtype=hidden.dtype)
        with torch.no_grad():
            perturbed, _ = reference(changed, embeddings(changed, positions), positions, mask)
            torch.testing.assert_close(perturbed[:, :future], expected.detach()[:, :future], atol=0, rtol=0)
            local_changed = changed.chunk(2, dim=1)[rank].contiguous()
            cp_perturbed, _ = candidate(
                local_changed, embeddings(local_changed, local_positions), local_positions, local_mask
            )
            past_local = max(0, min(local_hidden.shape[1], future - rank * local_hidden.shape[1]))
            torch.testing.assert_close(cp_perturbed[:, :past_local], actual.detach()[:, :past_local], atol=0, rtol=0)
        # Past outputs must also have no differentiable path to the future token.
        causal_hidden = local_hidden.detach().clone().requires_grad_()
        causal, _ = candidate(causal_hidden, embeddings(causal_hidden, local_positions), local_positions, local_mask)
        future_grad = torch.autograd.grad(causal[:, :past_local].sum(), causal_hidden)[0]
        local_future = future - rank * local_hidden.shape[1]
        if 0 <= local_future < local_hidden.shape[1]:
            assert torch.count_nonzero(future_grad[:, local_future]) == 0


def _check_full_model(rank, cp_size, rendered):
    from types import SimpleNamespace

    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh

    from llamafactory.v1.plugins.model_plugins.parallelization import hook, sequence_parallel

    torch.manual_seed(42)
    config = hf.DeepseekV4Config(
        vocab_size=512,
        hidden_size=32,
        head_dim=16,
        q_lora_rank=16,
        num_attention_heads=4,
        o_groups=2,
        o_lora_rank=8,
        partial_rotary_factor=0.5,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_hash_layers=1,
        compress_ratios=[4, 128],
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        sliding_window=16,
        use_cache=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference = hf.DeepseekV4ForCausalLM(config).double()
    reference.model.layers[0].mlp.gate.tid2eid.copy_((torch.arange(512)[:, None] + torch.arange(2)[None, :]) % 4)
    candidate = copy.deepcopy(reference)
    reference.gradient_checkpointing_enable({"use_reentrant": False})
    candidate.gradient_checkpointing_enable({"use_reentrant": False})
    mesh = init_device_mesh("cpu", (cp_size,), mesh_dim_names=("cp",))
    interface = SimpleNamespace(
        current_device=torch.device("cpu"), get_device_mesh=lambda _: mesh, get_group=lambda _: mesh.get_group()
    )
    old_hook, old_routing = hook.DistributedInterface, sequence_parallel.DistributedInterface
    hook.DistributedInterface = sequence_parallel.DistributedInterface = lambda: interface
    try:
        sequence_parallel.apply_sequence_parallel(candidate, cp_size)
        ids = torch.tensor(rendered["input_ids"]).unsqueeze(0).expand(2, -1)
        positions = torch.tensor(rendered["position_ids"]).unsqueeze(0).expand(2, -1)
        labels = torch.tensor(rendered["labels"]).unsqueeze(0).expand(2, -1)
        weights = torch.tensor(rendered["loss_weights"]).unsqueeze(0).expand(2, -1)
        padding = (-ids.shape[-1]) % cp_size
        batch = {
            "input_ids": ids,
            "position_ids": positions,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
            "loss_weights": weights,
        }
        expected = reference(
            input_ids=F.pad(ids, (0, padding)),
            attention_mask=F.pad(torch.ones_like(ids), (0, padding)),
            labels=F.pad(labels, (0, padding), value=-100),
            use_cache=False,
        ).loss
        actual = sequence_parallel.sequence_parallel_loss(candidate, batch)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        expected.backward()
        actual.backward()
        for (name, param), (_, ref) in zip(candidate.named_parameters(), reference.named_parameters()):
            if ref.grad is None:
                assert param.grad is None, name
            else:
                dist.all_reduce(param.grad)
                param.grad.div_(cp_size)
                torch.testing.assert_close(param.grad, ref.grad, rtol=3e-4, atol=3e-5, msg=name)
    finally:
        hook.DistributedInterface, sequence_parallel.DistributedInterface = old_hook, old_routing


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for ratio, length in ((0, 132), (4, 132), (128, 126), (128, 132), (4, 256), (128, 256)):
            _check_attention(rank, ratio, length)
    finally:
        dist.destroy_process_group()


def test_native_attention_and_compressors_preserve_full_sequence_forward_and_backward():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_worker, args=(directory + "/group",), nprocs=2, join=True)


def _full_model_worker(rank, rendezvous, cp_size, rendered):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=cp_size)
    try:
        _check_full_model(rank, cp_size, rendered)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_rendered_zero_based_positions_match_native_full_model(tokenizer, cp_size):
    from llamafactory.v1.core.rendering import create_renderer
    from llamafactory.v1.utils.packages import is_transformers_version_greater_than

    if not is_transformers_version_greater_than("5.16.0"):
        pytest.skip("Native V4 full-model CP requires Transformers >=5.16")
    sample = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "value": "x " * 64}]},
            {"role": "assistant", "content": [{"type": "text", "value": "answer"}]},
        ]
    }
    batch = create_renderer(tokenizer, SimpleNamespace(model_type="deepseek_v4")).process_samples([sample])[0]
    assert batch["position_ids"] == list(range(len(batch["input_ids"])))
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_full_model_worker, args=(directory + "/group", cp_size, batch), nprocs=cp_size, join=True)
