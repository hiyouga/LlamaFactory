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

"""The definition of NPU fused RoPE kernels.

Init Phase:
1. Define RoPE forward functions.
2. Register NPU fused RoPE kernel.

"""

import importlib

import torch

from ......accelerator.helper import DeviceType, get_current_accelerator
from ......utils.logging import get_logger
from ......utils.types import HFModel
from ...base import BaseKernel, KernelPlugin


logger = get_logger(__name__)

try:
    import torch_npu
except ImportError as exc:
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def _apply_npu_rotary_emb(q, k, cos, sin):
    """Apply NPU-accelerated rotary embedding with automatic Partial RoPE detection.

    Partial RoPE is detected when the ``cos/sin`` width is smaller than the ``q/k``
    head dimension. The leading rotary dimensions are transformed and any trailing
    dimensions are passed through unchanged.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        cos (Tensor): Cosine part of rotary embedding (already unsqueezed).
        sin (Tensor): Sine part of rotary embedding (already unsqueezed).

    Returns:
        tuple[Tensor, Tensor]: The embedded query and key tensors ``(q_embed, k_embed)``.
    """
    rotary_dim = cos.shape[-1]
    query_dim = q.shape[-1]

    if rotary_dim < query_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin, "half").to(q.dtype)
        k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin, "half").to(k.dtype)

        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    else:
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin, "half").to(q.dtype)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin, "half").to(k.dtype)

    return q_embed, k_embed


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to query and key tensors using NPU optimization.

    This function automatically supports both Full RoPE and Partial RoPE based on
    the dimension ratio between ``cos/sin`` and ``q/k`` tensors.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        cos (Tensor): Cosine part of embedding.
        sin (Tensor): Sine part of embedding.
        position_ids (Tensor | int, optional): Ignored Transformers v4 position IDs, or the Transformers v5
            ``unsqueeze_dim`` when supplied as the fifth positional argument.
        unsqueeze_dim (int): Dimension to unsqueeze cos and sin. Defaults to 1.

    Returns:
        tuple[Tensor, Tensor]: The embedded query and key tensors ``(q_embed, k_embed)``.
    """
    # In transformers v5, the fifth positional argument is ``unsqueeze_dim``.
    if isinstance(position_ids, int):
        unsqueeze_dim = position_ids

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    return _apply_npu_rotary_emb(q, k, cos, sin)


def _default_rope_patch(module_type: str):
    return (
        (
            f"transformers.models.{module_type}.modeling_{module_type}",
            (("apply_rotary_pos_emb", _apply_rotary_pos_emb),),
        ),
    )


_MODEL_TYPE_TO_PATCHES = {
    "qwen3": _default_rope_patch("qwen3"),
    "qwen3_moe": _default_rope_patch("qwen3_moe"),
    "qwen3_next": _default_rope_patch("qwen3_next"),
    "qwen3_omni_moe": _default_rope_patch("qwen3_omni_moe"),
    "qwen3_omni_moe_thinker": _default_rope_patch("qwen3_omni_moe"),
    "qwen3_vl": _default_rope_patch("qwen3_vl"),
    "qwen3_vl_moe": _default_rope_patch("qwen3_vl_moe"),
    "qwen3_5": _default_rope_patch("qwen3_5"),
    "qwen3_5_text": _default_rope_patch("qwen3_5"),
    "qwen3_5_moe": _default_rope_patch("qwen3_5_moe"),
    "qwen3_5_moe_text": _default_rope_patch("qwen3_5_moe"),
}


@KernelPlugin("npu_fused_rope").register()
class NpuRoPEKernel(BaseKernel):
    """NPU Kernel for Rotary Position Embedding."""

    @staticmethod
    def check_device() -> None:
        current = get_current_accelerator().type
        if current != DeviceType.NPU:
            raise RuntimeError(f"NpuRoPEKernel requires NPU, current accelerator is {current}.")

    @staticmethod
    def check_deps() -> None:
        if _TORCH_NPU_IMPORT_ERROR is not None:
            raise RuntimeError("NpuRoPEKernel requires torch_npu.") from _TORCH_NPU_IMPORT_ERROR

    @staticmethod
    def _apply_model_patches(model_type: str) -> int:
        patches = _MODEL_TYPE_TO_PATCHES.get(model_type)
        if patches is None:
            return 0

        patched_count = 0
        for module_name, replacements in patches:
            try:
                target_module = importlib.import_module(module_name)
            except Exception as e:
                logger.warning_rank0_once(f"Failed to import {module_name} for NPU RoPE kernel: {e}")
                continue

            for target_function_name, replacement in replacements:
                if not hasattr(target_module, target_function_name):
                    logger.warning_rank0_once(f"{module_name} has no {target_function_name}, skip NPU RoPE patch.")
                    continue

                if getattr(target_module, target_function_name) is replacement:
                    continue

                setattr(target_module, target_function_name, replacement)
                patched_count += 1

        return patched_count

    @staticmethod
    def _apply(**kwargs) -> "HFModel":
        """Apply RoPE acceleration by monkey-patching rotary embedding functions.

        Selects the target transformers modeling module from ``model.config.model_type``
        and replaces its rotary embedding helper with the NPU-accelerated version.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched RoPE functions.
        """
        model = kwargs["model"]

        model_type = getattr(model.config, "model_type", None)
        if model_type not in _MODEL_TYPE_TO_PATCHES:
            return model

        patched_count = NpuRoPEKernel._apply_model_patches(model_type)
        if patched_count:
            logger.info_rank0(f"Applied NPU RoPE kernel to {patched_count} functions for model type: {model_type}.")

        return model
