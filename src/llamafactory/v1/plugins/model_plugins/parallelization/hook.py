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

"""Common language-boundary hook for text-only and multimodal CP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from .batch import MULTIMODAL_ENCODER_INPUT_KEYS, split_sequence_tensor


if TYPE_CHECKING:
    from ....utils.types import HFModel


logger = logging.get_logger(__name__)

_MULTIMODAL_ENCODER_NAMES = ("visual", "audio_tower")


def _resolve_multimodal_boundary(model: HFModel) -> nn.Module | None:
    cores = (model, getattr(model, "model", None))
    for boundary_name in ("language_model", "model"):
        for core in cores:
            has_encoder = any(isinstance(getattr(core, name, None), nn.Module) for name in _MULTIMODAL_ENCODER_NAMES)
            boundary = getattr(core, boundary_name, None)
            if has_encoder and isinstance(boundary, nn.Module):
                return boundary

    return None


def _split_deepstack_inputs(kwargs: dict[str, Any], device_mesh) -> None:
    """Align optional DeepStack visual rows with the local language shard."""
    visual_pos_masks = kwargs.get("visual_pos_masks")
    if visual_pos_masks is None:
        return

    deepstack_visual_embeds = kwargs["deepstack_visual_embeds"]
    visual_ordinals = visual_pos_masks.reshape(-1).long().cumsum(dim=0) - 1
    visual_ordinals = visual_ordinals.view_as(visual_pos_masks)
    local_visual_pos_masks = split_sequence_tensor(visual_pos_masks, device_mesh)
    local_visual_ordinals = split_sequence_tensor(visual_ordinals, device_mesh)[local_visual_pos_masks]

    kwargs["visual_pos_masks"] = local_visual_pos_masks
    kwargs["deepstack_visual_embeds"] = [
        visual_embeds.index_select(0, local_visual_ordinals.to(visual_embeds.device))
        for visual_embeds in deepstack_visual_embeds
    ]


def install_sequence_parallel_hook(model: HFModel) -> None:
    """Install the common CP split at the model's language boundary."""
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        model = get_base_model()

    boundary = _resolve_multimodal_boundary(model)
    requires_fused_inputs = boundary is not None
    if boundary is None:
        boundary = model.base_model

    device_mesh = DistributedInterface().get_device_mesh(Dim.CP)

    def sequence_parallel_pre_hook(_module, args, kwargs):
        encoder_inputs = {key for key in MULTIMODAL_ENCODER_INPUT_KEYS if kwargs.get(key) is not None}
        if not requires_fused_inputs and encoder_inputs:
            raise ValueError(
                "Sequence parallelism reached a text language boundary with multimodal encoder inputs "
                f"{sorted(encoder_inputs)}; this model structure is not supported."
            )

        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        if requires_fused_inputs and input_ids is not None:
            raise ValueError(
                "Multimodal sequence parallelism must enter the language boundary through fused "
                "`inputs_embeds`; received non-null `input_ids`."
            )

        uses_inputs_embeds = inputs_embeds is not None
        sequence_tensor = inputs_embeds if uses_inputs_embeds else input_ids
        sequence_length = sequence_tensor.shape[1]

        attention_mask = kwargs.get("attention_mask")

        position_ids = kwargs["position_ids"]
        if position_ids.shape[-1] != sequence_length:
            raise ValueError("position_ids must match the global sequence length before CP splitting.")

        _split_deepstack_inputs(kwargs, device_mesh)

        sequence_name = "inputs_embeds" if uses_inputs_embeds else "input_ids"
        kwargs[sequence_name] = split_sequence_tensor(sequence_tensor, device_mesh, dim=1)
        if attention_mask is not None:
            kwargs["attention_mask"] = split_sequence_tensor(attention_mask, device_mesh)
        kwargs["position_ids"] = split_sequence_tensor(position_ids, device_mesh)

        kwargs["use_cache"] = False
        return args, kwargs

    boundary.register_forward_pre_hook(sequence_parallel_pre_hook, with_kwargs=True)
    logger.info_rank0("Installed sequence-parallel pre-hook at the language boundary.")
