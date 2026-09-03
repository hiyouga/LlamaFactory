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

from typing import TYPE_CHECKING, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from ..data import get_dataset, get_template_and_fix_tokenizer
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer


if TYPE_CHECKING:
    from peft import LoraModel
    from transformers import PreTrainedModel

    from ..data.data_utils import DatasetModule


def compare_model(
    model_a: "torch.nn.Module",
    model_b: "torch.nn.Module",
    diff_keys: list[str] = [],
    skip_keys: list[str] = [],
) -> None:
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert not (set(skip_keys) & set(diff_keys)), "skip_keys and diff_keys must not overlap"
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        if any(key in name for key in skip_keys):
            continue
        a, b = state_dict_a[name].cpu(), state_dict_b[name].cpu()
        if any(key in name for key in diff_keys):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-5) is False
        else:
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-5) is True


def compare_model_pissa(model_a: "torch.nn.Module", model_b: "torch.nn.Module", adapter: str = "default") -> None:
    """Compare two PiSSA-initialized LoRA models by gauge-invariant quantities.

    SVD factorization is not unique: ``torch.linalg.svd`` on any backend (cuBLAS, MKL,
    Intel XPU, ...) may return sign-flipped columns of ``U``/``V``, or an arbitrary
    rotation within a degenerate (equal singular value) subspace. All such variants are
    equally valid, so the raw ``lora_A`` / ``lora_B`` factors legitimately differ
    across platforms/backends and must not be compared directly (that is why the
    old factor-level ``compare_model`` check was ``xfail``-flaky).

    What *is* invariant under every SVD gauge choice is the effective update
    ``B @ A * scaling``. This function asserts on that, plus the residual base weight
    ``W_orig - B @ A`` directly (both must match between the two models).
    """
    # lora_embedding_A/B are used by Embedding LoRA layers and are intentionally not checked here;
    # this function covers only Linear LoRA layers (lora_A/lora_B via get_delta_weight).
    modules_a = {n: m for n, m in model_a.named_modules() if hasattr(m, "lora_A") and adapter in m.lora_A}
    modules_b = {n: m for n, m in model_b.named_modules() if hasattr(m, "lora_A") and adapter in m.lora_A}
    assert modules_a.keys() == modules_b.keys()
    assert len(modules_a) > 0, "no LoRA layers found to compare"

    # PiSSA models load in fp16 and the invariant flows through an SVD + a B @ A matmul,
    # so the recovered quantities agree only to fp16 precision (~1e-4). A gauge violation,
    # by contrast, changes signs/magnitudes wholesale, so this tolerance still catches it.
    rtol, atol = 1e-3, 1e-4
    for name, module_a in modules_a.items():
        module_b = modules_b[name]

        # Gauge-invariant update: B @ A * scaling, in the base weight's orientation.
        delta_a = module_a.get_delta_weight(adapter).cpu()
        delta_b = module_b.get_delta_weight(adapter).cpu()
        assert torch.allclose(delta_a, delta_b, rtol=rtol, atol=atol), f"delta (B@A) mismatch at {name}"

        # Residual base weights (W_orig - B@A) must also match; since deltas are already
        # equal above, this is equivalent to checking the base weights directly.
        assert torch.allclose(
            module_a.base_layer.weight.cpu(), module_b.base_layer.weight.cpu(), rtol=rtol, atol=atol
        ), f"base_layer.weight mismatch at {name}"


def check_lora_model(model: "LoraModel") -> tuple[set[str], set[str]]:
    linear_modules, extra_modules = set(), set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        elif "modules_to_save" in name:
            extra_modules.add(name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    return linear_modules, extra_modules


def load_train_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    model_args, _, _, finetuning_args, _ = get_train_args(kwargs)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=add_valuehead)


def load_infer_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    model_args, _, finetuning_args, _ = get_infer_args(kwargs)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead)


def load_reference_model(
    model_path: str,
    lora_path: Optional[str] = None,
    use_lora: bool = False,
    use_pissa: bool = False,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> Union["PreTrainedModel", "LoraModel"]:
    if add_valuehead:
        model: AutoModelForCausalLMWithValueHead = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        return model

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    if use_lora or use_pissa:
        model = PeftModel.from_pretrained(
            model, lora_path, subfolder="pissa_init" if use_pissa else None, is_trainable=is_trainable
        )
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def load_dataset_module(**kwargs) -> "DatasetModule":
    model_args, data_args, training_args, _, _ = get_train_args(kwargs)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, kwargs["stage"], **tokenizer_module)
    return dataset_module


def patch_valuehead_model() -> None:
    def post_init(self: "AutoModelForCausalLMWithValueHead", state_dict: dict[str, "torch.Tensor"]) -> None:
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys() if k.startswith("v_head.")}
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

    AutoModelForCausalLMWithValueHead.post_init = post_init
