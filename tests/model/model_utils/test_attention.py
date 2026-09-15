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

import builtins
import importlib.metadata
import inspect
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import transformers.utils
from transformers import GenerationMixin
from transformers.utils import is_flash_attn_2_available


# Compatible with Transformers v4 and Transformers v5
try:
    from transformers.utils import is_torch_sdpa_available
except ImportError:

    def is_torch_sdpa_available():
        return True


from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.model import patcher
from llamafactory.model.model_utils import attention
from llamafactory.train.test_utils import load_infer_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
}


@pytest.mark.parametrize("backend", ["fa2", "fa3", "fa4"])
def test_configure_flash_attention(backend, monkeypatch):
    monkeypatch.setattr(transformers.utils, f"is_flash_attn_{backend[-1]}_available", lambda: True, raising=False)
    check_version = Mock()
    monkeypatch.setattr(attention, "check_version", check_version)
    patch_varlen = Mock()
    monkeypatch.setattr(attention, "_patch_fa4_varlen", patch_varlen)
    config = SimpleNamespace(model_type="llama")
    attention.configure_attn_implementation(config, SimpleNamespace(flash_attn=backend))
    assert config._attn_implementation == f"flash_attention_{backend[-1]}"
    if backend == "fa4":
        check_version.assert_called_once_with("flash-attn-4>=4.0.0b30", mandatory=True)
        patch_varlen.assert_called_once_with()
    else:
        check_version.assert_not_called()
        patch_varlen.assert_not_called()


@pytest.mark.parametrize("model_type", ["qwen3_vl", "internlm2", "kimi_vl", "kimi_k25", "youtu_vl"])
def test_configure_fa4_model_routing(model_type, monkeypatch):
    monkeypatch.setattr(transformers.utils, "is_flash_attn_4_available", lambda: True, raising=False)
    monkeypatch.setattr(attention, "check_version", Mock())
    monkeypatch.setattr(attention, "_patch_fa4_varlen", Mock())
    config = SimpleNamespace(
        model_type=model_type,
        vision_config=SimpleNamespace(hidden_size=1152, num_heads=16),  # head_dim=72 is supported by b30
        text_config=SimpleNamespace(),
    )
    attention.configure_attn_implementation(config, SimpleNamespace(flash_attn="fa4"))
    if model_type == "internlm2":
        assert config.attn_implementation == "flash_attention_4"
    elif model_type != "kimi_vl":
        assert config._attn_implementation == "flash_attention_4"

    if model_type in ("kimi_vl", "kimi_k25", "youtu_vl"):
        assert config.vision_config._attn_implementation == "flash_attention_4"
        assert config.text_config._attn_implementation == "flash_attention_4"


def test_fa4_unavailable_preserves_config(monkeypatch):
    monkeypatch.setattr(transformers.utils, "is_flash_attn_4_available", lambda: False, raising=False)
    check_version = Mock()
    monkeypatch.setattr(attention, "check_version", check_version)
    config = SimpleNamespace(model_type="llama", _attn_implementation="sdpa")
    attention.configure_attn_implementation(config, SimpleNamespace(flash_attn="fa4"))
    assert config._attn_implementation == "sdpa"
    check_version.assert_not_called()


def test_fa4_without_transformers_support(monkeypatch):
    original_import = builtins.__import__

    def import_without_fa4(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.utils" and "is_flash_attn_4_available" in fromlist:
            raise ImportError("FlashAttention-4 availability helper does not exist")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_fa4)
    config = SimpleNamespace(model_type="llama", _attn_implementation="sdpa")
    attention.configure_attn_implementation(config, SimpleNamespace(flash_attn="fa4"))
    assert config._attn_implementation == "sdpa"


@pytest.mark.parametrize("package_version", ["4.0.0b29", "4.0.0b30"])
def test_fa4_minimum_version(package_version, monkeypatch):
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: package_version if name == "flash-attn-4" else original_version(name),
    )
    monkeypatch.setattr(transformers.utils, "is_flash_attn_4_available", lambda: True, raising=False)
    monkeypatch.setenv("DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(attention, "_patch_fa4_varlen", Mock())
    config = SimpleNamespace(model_type="llama")
    if package_version == "4.0.0b29":
        with pytest.raises(ImportError, match=r"flash-attn-4>=4.0.0b30"):
            attention.configure_attn_implementation(config, SimpleNamespace(flash_attn="fa4"))
        assert not hasattr(config, "_attn_implementation")
    else:
        attention.configure_attn_implementation(config, SimpleNamespace(flash_attn="fa4"))
        assert config._attn_implementation == "flash_attention_4"


@pytest.fixture
def fake_fa4_modules(monkeypatch):
    r"""Exercise the optional integration without installing or importing a CUDA package."""
    package = ModuleType("flash_attn")
    cute = ModuleType("flash_attn.cute")
    interface = ModuleType("flash_attn.cute.interface")
    package.cute = cute
    cute.interface = interface
    for module in (package, cute, interface):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return cute, interface


@pytest.mark.parametrize("positional", [False, True])
def test_fa4_varlen_signature_and_idempotence(fake_fa4_modules, positional):
    cute, interface = fake_fa4_modules
    calls = []

    # Deliberately place k before q and at different positions from FA4's current API.
    def original(q, k, v, max_seqlen_k=None, extra=None, max_seqlen_q=None):
        calls.append((q, k, v, max_seqlen_q, max_seqlen_k, extra))
        return "output"

    interface.flash_attn_varlen_func = cute.flash_attn_varlen_func = original
    attention._patch_fa4_varlen()
    wrapped = cute.flash_attn_varlen_func
    assert inspect.signature(wrapped) == inspect.signature(original)
    attention._patch_fa4_varlen()
    assert cute.flash_attn_varlen_func is wrapped is interface.flash_attn_varlen_func
    if positional:
        output = wrapped("q", "k", "v", torch.tensor(11), "extra", torch.tensor(7))
    else:
        output = wrapped("q", "k", "v", max_seqlen_q=torch.tensor(7), max_seqlen_k=torch.tensor(11), extra="extra")
    assert output == "output"
    assert calls == [("q", "k", "v", 7, 11, "extra")]
    assert isinstance(calls[0][3], int) and isinstance(calls[0][4], int)
    wrapped("q", "k", "v", max_seqlen_q=5, max_seqlen_k=None)
    assert calls[-1] == ("q", "k", "v", 5, None, None)


def test_fa4_varlen_propagates_backend_errors(fake_fa4_modules):
    cute, interface = fake_fa4_modules
    calls = []

    def original(q, k, v, max_seqlen_q=None, max_seqlen_k=None):
        calls.append(max_seqlen_q)
        raise TypeError("backend failure")

    interface.flash_attn_varlen_func = cute.flash_attn_varlen_func = original
    attention._patch_fa4_varlen()
    with pytest.raises(TypeError, match="backend failure"):
        cute.flash_attn_varlen_func("q", "k", "v", max_seqlen_q=torch.tensor(7))
    assert calls == [7]  # no retry that could hide errors or execute the kernel twice


@pytest.mark.parametrize("backend", ["fa2", "fa3", "fa4", "sdpa"])
@pytest.mark.parametrize("is_trainable", [False, True])
def test_qwen35_packing_patch_dispatch(backend, is_trainable, monkeypatch):
    class DummyModel(GenerationMixin):
        config = SimpleNamespace(model_type="qwen3_5")
        generation_config = SimpleNamespace(do_sample=True)

        def add_model_tags(self, tags):
            pass

    for name in ("prepare_model_for_training", "autocast_projector_dtype", "add_z3_leaf_module"):
        monkeypatch.setattr(patcher, name, Mock())
    monkeypatch.setattr(patcher, "is_torch_cuda_available", lambda: True)
    monkeypatch.setattr(patcher, "is_torch_npu_available", lambda: False)
    gpu_patch = Mock()
    monkeypatch.setattr(patcher, "patch_qwen3_5_forward_gpu", gpu_patch)
    model = DummyModel()
    model_args = SimpleNamespace(resize_vocab=False, use_unsloth=True, flash_attn=backend)
    patcher.patch_model(model, None, model_args, is_trainable=is_trainable, add_valuehead=False)
    if is_trainable and backend in ("fa2", "fa3", "fa4"):
        gpu_patch.assert_called_once_with(model)
    else:
        gpu_patch.assert_not_called()


@pytest.mark.parametrize("version", [3, 4])
def test_print_flash_attention(version, monkeypatch):
    log = Mock()
    monkeypatch.setattr(attention.logger, "info_rank0", log)
    attention.print_attn_implementation(SimpleNamespace(_attn_implementation=f"flash_attention_{version}"))
    log.assert_called_once_with(f"Using FlashAttention-{version} for faster training and inference.")


@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
def test_attention():
    attention_available = ["disabled"]
    if is_torch_sdpa_available():
        attention_available.append("sdpa")

    if is_flash_attn_2_available():
        attention_available.append("fa2")

    llama_attention_classes = {
        "disabled": "LlamaAttention",
        "sdpa": "LlamaSdpaAttention",
        "fa2": "LlamaFlashAttention2",
    }
    for requested_attention in attention_available:
        model = load_infer_model(flash_attn=requested_attention, **INFER_ARGS)
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                assert module.__class__.__name__ == llama_attention_classes[requested_attention]
