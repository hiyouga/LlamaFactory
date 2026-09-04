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
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llamafactory.model.patcher import (
    _install_npu_gdn_kernel_on_layer,
    _install_npu_gdn_kernel_on_modeling,
    _replace_gdn_kernel_for_npu,
)
from llamafactory.v1.plugins.model_plugins.parallelization import gdn_attention
from llamafactory.v1.plugins.model_plugins.parallelization.gdn_attention import (
    require_packed_conv1d_support,
    resolve_activation,
    resolve_gdn_kernel,
)


def _fake_gdn(module_name: str, **attributes):
    """A GDN stand-in whose class lives in a throwaway module, so the module-level lookup is ours."""
    modeling = ModuleType(module_name)
    sys.modules[module_name] = modeling

    layer_cls = type("FakeGatedDeltaNet", (), {"__module__": module_name})
    layer = layer_cls()
    for name, value in attributes.items():
        setattr(layer, name, value)

    return layer, modeling


@pytest.fixture
def modeling_module(request):
    """Register a throwaway modeling module and take it back out afterwards."""
    created: list[str] = []

    def factory(**attributes):
        name = f"_fake_modeling_{request.node.name}_{len(created)}"
        created.append(name)
        return _fake_gdn(name, **attributes)

    yield factory

    for name in created:
        sys.modules.pop(name, None)


def test_kernel_comes_from_the_layer_when_it_has_one(modeling_module):
    # transformers <= 5.14 hangs the kernels off the module, and it is also where the NPU
    # patch installs its replacement, so the layer has to win.
    layer, modeling = modeling_module(chunk_gated_delta_rule="on the layer")
    modeling.torch_chunk_gated_delta_rule = "on the module"

    assert resolve_gdn_kernel(layer, "chunk_gated_delta_rule", "torch_chunk_gated_delta_rule") == "on the layer"


def test_kernel_falls_back_to_the_module(modeling_module):
    # transformers 5.15 dropped the attributes and left module-level functions behind a
    # kernel-hub fallback decorator.
    layer, modeling = modeling_module()
    modeling.torch_chunk_gated_delta_rule = "on the module"

    assert resolve_gdn_kernel(layer, "chunk_gated_delta_rule", "torch_chunk_gated_delta_rule") == "on the module"


def test_missing_kernel_resolves_to_none(modeling_module):
    # A layer whose attribute is None because flash-linear-attention is not installed reads
    # the same as one that never had the attribute; both take the F.conv1d path.
    layer, _ = modeling_module(causal_conv1d_fn=None)

    assert resolve_gdn_kernel(layer, "causal_conv1d_fn", "causal_conv1d_fn") is None


def test_npu_kernel_installs_on_the_layer_when_the_forward_reads_it(modeling_module):
    layer, modeling = modeling_module(chunk_gated_delta_rule=lambda *_: "default")
    modeling.torch_chunk_gated_delta_rule = lambda *_: "default"

    assert _install_npu_gdn_kernel_on_layer(layer, "npu kernel") is True
    assert layer.chunk_gated_delta_rule == "npu kernel"
    # The per-layer write is the narrow one, so the shared module is left alone.
    assert modeling.torch_chunk_gated_delta_rule != "npu kernel"


def test_npu_kernel_installs_on_the_module_when_the_layer_has_no_attribute(modeling_module):
    # Assigning to the layer here would land on something the forward never reads, and the
    # run would silently keep the default kernel.
    layer, modeling = modeling_module()
    modeling.torch_chunk_gated_delta_rule = lambda *_: "default"

    installed_at = _install_npu_gdn_kernel_on_modeling(layer, "npu kernel")

    assert installed_at.endswith(".torch_chunk_gated_delta_rule")
    assert modeling.torch_chunk_gated_delta_rule == "npu kernel"
    assert not hasattr(layer, "chunk_gated_delta_rule")


def test_npu_kernel_reports_when_it_finds_nowhere_to_install(modeling_module):
    layer, _ = modeling_module()

    assert _install_npu_gdn_kernel_on_layer(layer, "npu kernel") is False
    assert _install_npu_gdn_kernel_on_modeling(layer, "npu kernel") == ""


def test_npu_module_level_kernel_is_written_once_not_once_per_layer(modeling_module):
    # One write covers every layer, and it mutates state shared by the whole process, so
    # doing it per layer is repeated global mutation for no gain.
    layer, modeling = modeling_module()
    modeling.torch_chunk_gated_delta_rule = lambda *_: "default"

    writes = []
    original_setattr = type(modeling).__setattr__

    def counting_setattr(self, name, value):
        if name == "torch_chunk_gated_delta_rule":
            writes.append(value)

        original_setattr(self, name, value)

    object.__setattr__(
        modeling, "__class__", type("CountingModule", (type(modeling),), {"__setattr__": counting_setattr})
    )

    layers = [SimpleNamespace(linear_attn=layer) for _ in range(4)]
    _replace_gdn_kernel_for_npu(layers, "npu kernel", "fake model")

    assert writes == ["npu kernel"]


def test_packed_run_without_causal_conv1d_is_refused(monkeypatch):
    # transformers 5.15 always exposes a module-level causal_conv1d_fn, and with no kernel
    # package installed its decorator resolves to an F.conv1d body that filters cu_seqlens
    # out of **kwargs. Without this the boundaries would be dropped and nothing would say so.
    monkeypatch.setattr(gdn_attention, "is_causal_conv1d_available", lambda: False)

    with pytest.raises(RuntimeError, match="cu_seqlens requires causal_conv1d"):
        require_packed_conv1d_support(cu_seqlens=torch.tensor([0, 4, 8]))


def test_unpacked_run_without_causal_conv1d_is_allowed(monkeypatch):
    # No boundaries to lose, so an unpacked run must not start demanding the kernels.
    monkeypatch.setattr(gdn_attention, "is_causal_conv1d_available", lambda: False)

    assert require_packed_conv1d_support(cu_seqlens=None) is None


def test_packed_run_with_causal_conv1d_is_allowed(monkeypatch):
    monkeypatch.setattr(gdn_attention, "is_causal_conv1d_available", lambda: True)

    assert require_packed_conv1d_support(cu_seqlens=torch.tensor([0, 4, 8])) is None


def test_activation_prefers_the_module_that_defines_its_own():
    # transformers <= 5.8 builds self.act in __init__, and a module that overrides it should
    # keep winning.
    own = torch.nn.Tanh()

    assert resolve_activation(SimpleNamespace(act=own, activation="silu")) is own


def test_activation_falls_back_to_the_name_and_is_not_rebuilt(monkeypatch):
    # 5.15 dropped self.act. ACT2FN is a ClassInstantier, so a direct lookup constructs a new
    # nn.Module every time, and this sits in the per-layer forward.
    module = SimpleNamespace(activation="silu")

    first = resolve_activation(module)
    second = resolve_activation(SimpleNamespace(activation="silu"))

    # Not asserting the concrete class: transformers has moved silu between its own
    # SiLUActivation and torch.nn.SiLU across the supported range. The point is the identity.
    assert isinstance(first, torch.nn.Module)
    assert first is second
