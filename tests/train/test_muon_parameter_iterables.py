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

from copy import deepcopy
from itertools import chain

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from llamafactory.third_party.muon import Muon
from llamafactory.v1.plugins.trainer_plugins.optimizers.muon_optimizer import Muon as MuonV1


@pytest.fixture(params=[Muon, MuonV1], ids=["legacy", "v1"])
def optimizer_cls(request):
    return request.param


def parameter_lists(model):
    matrices, fallback = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and not any(part in name for part in ("wte", "wpe", "lm_head")):
            matrices.append(parameter)
        else:
            fallback.append(parameter)
    return matrices, fallback


def iterable_parameters(parameters, kind):
    if kind == "generator":
        return (parameter for parameter in parameters)
    if kind == "filter":
        return filter(lambda parameter: parameter.requires_grad, parameters)
    if kind == "chain":
        return chain(parameters[:1], parameters[1:])
    return parameters


@pytest.mark.parametrize("kind", ["list", "generator", "filter", "chain"])
@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_iterables_match_list_training_and_checkpoint(optimizer_cls, kind, nesterov, tmp_path):
    torch.manual_seed(11)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    reference = deepcopy(model)
    matrices, fallback = parameter_lists(model)
    expected_matrices, expected_fallback = parameter_lists(reference)
    assert matrices and fallback
    original_order = matrices.copy()
    options = dict(lr=0.003, wd=0.02, momentum=0.8, nesterov=nesterov)
    optimizer = optimizer_cls(muon_params=iterable_parameters(matrices, kind), adamw_params=iter(fallback), **options)
    expected_optimizer = optimizer_cls(muon_params=expected_matrices, adamw_params=expected_fallback, **options)
    assert len(matrices) == len(original_order)
    assert all(a is b for a, b in zip(matrices, original_order))
    generator = torch.Generator().manual_seed(5)
    for step in range(4):
        tokens = torch.randint(0, 16, (2, 5), generator=generator)
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        actual = model(tokens, labels=tokens).loss
        expected = reference(tokens, labels=tokens).loss
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        expected.backward()
        for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0)
        if step == 1:
            checkpoint = tmp_path / "training.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            restored = GPT2LMHeadModel(model.config)
            restored.load_state_dict(saved["model"])
            matrices, fallback = parameter_lists(restored)
            resumed = optimizer_cls(
                muon_params=iterable_parameters(matrices, kind), adamw_params=iter(fallback), **options
            )
            resumed.load_state_dict(saved["optimizer"])
            model, optimizer = restored, resumed


def test_direct_module_parameters_support_gradless_and_training_steps(optimizer_cls):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6, bias=False), torch.nn.Linear(6, 3, bias=False))
    reference = deepcopy(model)
    optimizer = optimizer_cls(muon_params=model.parameters(), lr=0.01)
    expected_optimizer = optimizer_cls(muon_params=list(reference.parameters()), lr=0.01)
    optimizer.step()
    expected_optimizer.step()
    inputs = torch.randn(2, 4)
    for _ in range(2):
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        model(inputs).square().mean().backward()
        reference(inputs).square().mean().backward()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0)


def test_empty_muon_iterator_keeps_adamw_fallback(optimizer_cls):
    parameter = torch.nn.Parameter(torch.ones(3))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_cls(muon_params=iter(()), adamw_params=iter([parameter]), lr=0.01)
    expected_optimizer = optimizer_cls(muon_params=[], adamw_params=[reference], lr=0.01)
    for value in [0.2, -0.6, 0.4]:
        parameter.grad = torch.full_like(parameter, value)
        reference.grad = torch.full_like(reference, value)
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert optimizer.state[parameter]["use_muon"] is False


def test_iterator_keeps_the_constructor_dimension_check(optimizer_cls):
    parameter = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(AssertionError):
        optimizer_cls(muon_params=iter([parameter]))
