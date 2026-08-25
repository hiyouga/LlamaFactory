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
from types import ModuleType

import pytest

from llamafactory.model.patcher import _install_npu_gdn_kernel
from llamafactory.v1.plugins.model_plugins.parallelization.gdn_attention import resolve_gdn_kernel


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

    installed_at = _install_npu_gdn_kernel(layer, "npu kernel")

    assert installed_at == "layer attribute"
    assert layer.chunk_gated_delta_rule == "npu kernel"


def test_npu_kernel_installs_on_the_module_when_the_layer_has_no_attribute(modeling_module):
    # Assigning to the layer here would land on something the forward never reads, and the
    # run would silently keep the default kernel.
    layer, modeling = modeling_module()
    modeling.torch_chunk_gated_delta_rule = lambda *_: "default"

    installed_at = _install_npu_gdn_kernel(layer, "npu kernel")

    assert installed_at.endswith(".torch_chunk_gated_delta_rule")
    assert modeling.torch_chunk_gated_delta_rule == "npu kernel"
    assert not hasattr(layer, "chunk_gated_delta_rule")


def test_npu_kernel_reports_when_it_finds_nowhere_to_install(modeling_module):
    layer, _ = modeling_module()

    assert _install_npu_gdn_kernel(layer, "npu kernel") == ""
