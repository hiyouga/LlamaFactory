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

import copy

import pytest
import torch

from llamafactory.v1.plugins.trainer_plugins.optimizers.muon_optimizer import Muon, zeropower_via_newtonschulz5


@pytest.mark.parametrize("shape", [(7, 4), (4, 7), (4, 4)])
def test_batched_orthogonalization_normalizes_each_matrix(shape):
    torch.manual_seed(17)
    gradients = torch.randn(5, *shape) * torch.tensor([0.001, 1, 10, 100, 0]).view(-1, 1, 1)
    expected = torch.stack([zeropower_via_newtonschulz5(g, 5) for g in gradients])
    actual = zeropower_via_newtonschulz5(gradients, 5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def make_optimizer(batch_size, *, nesterov=True):
    # Mixed shapes, a partial final bucket, two parameter groups and AdamW fallback.
    parameters = [torch.nn.Parameter(torch.randn(*shape)) for shape in [(4, 7)] * 5 + [(7, 4)] * 2]
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = Muon(
        muon_params=parameters,
        adamw_params=[bias],
        lr=0.003,
        wd=0.02,
        nesterov=nesterov,
        ns_batch_size=batch_size,
    )
    # Optimizer must not combine matrices with different optimizer hyperparameters.
    group = optimizer.param_groups[0]
    group["params"] = parameters[:-1] + [bias]
    optimizer.add_param_group({"params": [parameters[-1]], "lr": 0.007, "momentum": 0.8})
    return parameters + [bias], optimizer


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_steps_match_serial_with_missing_gradients(nesterov, batch_size):
    torch.manual_seed(12)
    expected, serial = make_optimizer(1, nesterov=nesterov)
    torch.manual_seed(12)
    actual, batched = make_optimizer(batch_size, nesterov=nesterov)
    for step in range(4):
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            gradient = None if (i + step) % 3 == 0 else torch.randn_like(a)
            a.grad = gradient
            b.grad = None if gradient is None else gradient.clone()
        serial.step()
        batched.step()
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert serial.state[a].keys() == batched.state[b].keys()
            for key in serial.state[a]:
                torch.testing.assert_close(serial.state[a][key], batched.state[b][key], rtol=0, atol=0)


def test_batched_checkpoint_roundtrip_and_legacy_checkpoint():
    torch.manual_seed(5)
    parameters, optimizer = make_optimizer(4)
    for p in parameters:
        p.grad = torch.randn_like(p)
    optimizer.step()
    saved_parameters = [p.detach().clone() for p in parameters]
    saved_state = copy.deepcopy(optimizer.state_dict())
    restored, resumed = make_optimizer(1)
    for p, value in zip(restored, saved_parameters, strict=True):
        p.data.copy_(value)
    resumed.load_state_dict(saved_state)
    for a, b in zip(parameters, restored, strict=True):
        a.grad = torch.randn_like(a)
        b.grad = a.grad.clone()
    optimizer.step()
    resumed.step()
    for a, b in zip(parameters, restored, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Older Muon checkpoints have no batching option; restore with serial behavior.
    for group in saved_state["param_groups"]:
        group.pop("ns_batch_size")
    resumed.load_state_dict(saved_state)
    resumed.step()
    assert all(torch.isfinite(p).all() for p in restored)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_batch_size_rejected(batch_size):
    with pytest.raises(ValueError, match="ns_batch_size"):
        Muon(muon_params=[torch.nn.Parameter(torch.ones(2, 2))], ns_batch_size=batch_size)


def test_optimizer_plugin_validates_batch_size():
    from llamafactory.v1.plugins.trainer_plugins.optimizers.optimizer import create_muon_optimizer

    with pytest.raises(ValueError, match="ns_batch_size"):
        create_muon_optimizer(torch.nn.Linear(4, 4), {"ns_batch_size": 0})


def test_batch_working_set_is_bounded(monkeypatch):
    from llamafactory.v1.plugins.trainer_plugins.optimizers import muon_optimizer

    parameters, optimizer = make_optimizer(2)
    for p in parameters:
        p.grad = torch.randn_like(p)
    sizes = []
    original = muon_optimizer.zeropower_via_newtonschulz5

    def record_size(gradient, steps):
        sizes.append(gradient.size(0) if gradient.ndim == 3 else 1)
        return original(gradient, steps)

    monkeypatch.setattr(muon_optimizer, "zeropower_via_newtonschulz5", record_size)
    optimizer.step()
    assert max(sizes) == 2
    assert sum(sizes) == len(parameters) - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_batched_updates_remain_close_to_serial():
    torch.manual_seed(91)
    expected = [torch.nn.Parameter(torch.randn(128, 256, device="cuda")) for _ in range(5)]
    initial = [p.detach().clone() for p in expected]
    actual = [torch.nn.Parameter(p.detach().clone()) for p in expected]
    serial = Muon(muon_params=expected, lr=0.003)
    batched = Muon(muon_params=actual, lr=0.003, ns_batch_size=4)
    for _ in range(5):
        for a, b in zip(expected, actual, strict=True):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        serial.step()
        batched.step()
    # BF16 bmm/mm differ in rounding; compare complete parameter updates, not just loss.
    for a, b, start in zip(expected, actual, initial, strict=True):
        assert (a - b).norm() / (a - start).norm() < 0.06


def test_dtensor_parameters_keep_serial_path(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Replicate, distribute_tensor

    dist.init_process_group("gloo", init_method=(tmp_path / "init").as_uri(), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        torch.manual_seed(81)
        value = torch.randn(4, 7)
        gradient = torch.randn_like(value)
        parameters = [torch.nn.Parameter(distribute_tensor(value.clone(), mesh, [Replicate()])) for _ in range(2)]
        for p, batch_size in zip(parameters, [1, 4], strict=True):
            optimizer = Muon(muon_params=[p], ns_batch_size=batch_size)
            p.grad = distribute_tensor(gradient.clone(), mesh, [Replicate()])
            optimizer.step()
        torch.testing.assert_close(parameters[0].full_tensor(), parameters[1].full_tensor(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
