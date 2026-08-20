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
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers

from ....accelerator.interface import Dim, DistributedInterface
from ....utils import logging
from ....utils.plugin import BasePlugin
from ....utils.types import ModelOutput
from .gdn_attention import _get_gdn_module, gdn_forward_with_cp, is_gdn_layer
from .ulysses import (
    UlyssesAttention,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
)


logger = logging.get_logger(__name__)


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, cp_size: int):
        return super().__call__(model, cp_size)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    target_dtype=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1] * sequence_parallel_size,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
            position_ids=kwargs.get("position_ids", None),
            target_dtype=target_dtype,
        )
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, cp_size: int):
    # Replace _flash_attention_forward with new_flash_attn_forward
    module = sys.modules[model.__module__]

    set_ulysses_sequence_parallel_group(DistributedInterface().get_group(Dim.CP))

    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    assert num_attention_heads % cp_size == 0, "num_attention_heads must be divisible by cp_size"
    assert num_key_value_heads % cp_size == 0 or cp_size % num_key_value_heads == 0, (
        "num_key_value_heads must be divisible by cp_size"
    )

    origin_attn = transformers.modeling_flash_attention_utils._flash_attention_forward
    new_flash_attention_forward = partial(
        new_flash_attn_forward,
        group=get_ulysses_sequence_parallel_group(),
        mode="ulysses",
        attn_fn=origin_attn,
        sequence_parallel_size=cp_size,
    )

    for module_name, module in list(sys.modules.items()):
        try:
            if (
                hasattr(module, "__file__")
                and "transformers" in module.__file__
                and getattr(module._flash_attention_forward, "__name__", "") == "_flash_attention_forward"
            ):
                module._flash_attention_forward = new_flash_attention_forward
                logger.info_rank0(
                    f"Replaced _flash_attention_forward in module {module_name} with new_flash_attn_forward for sequence parallel."
                )
        except (AttributeError, TypeError):
            continue

    # Register GDN forward for CP support
    if cp_size > 1:
        replaced_modules = set()
        for name, module in model.named_modules():
            if is_gdn_layer(module):
                gdn_module = _get_gdn_module(module)
                if id(gdn_module) in replaced_modules:
                    continue
                replaced_modules.add(id(gdn_module))
                gdn_module.original_forward = gdn_module.forward
                gdn_module.forward = gdn_forward_with_cp.__get__(gdn_module, type(gdn_module))
                gdn_name = name if gdn_module is module else f"{name}.linear_attn"
                logger.info_rank0(f"Replaced GDN forward in {gdn_name} with gdn_forward_with_cp for context parallel.")


def padding_and_split_data(data, device_mesh=None):
    if device_mesh is not None:
        cp_size = device_mesh["cp"].size()
        cp_rank = device_mesh["cp"].get_local_rank()
        cp_group = device_mesh["cp"].get_group()
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.ndim > 1:
                data_len = torch.tensor(v.shape[-1], device=v.device, dtype=torch.int64)
                global_data_len = [torch.empty_like(data_len) for _ in range(cp_size)]
                dist.all_gather(global_data_len, data_len, group=cp_group)
                max_data_len = max(global_data_len)
                pad_size = max_data_len - v.shape[-1] + (cp_size - max_data_len % cp_size) % cp_size
                if k == "labels":
                    pad_value = -100
                elif k == "loss_weights":
                    pad_value = 0.0
                else:
                    pad_value = 0
                pad_data = F.pad(v, (0, pad_size), value=pad_value)
                data[k] = torch.chunk(pad_data, chunks=cp_size, dim=-1)[cp_rank].contiguous()
    return data


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs):
    device_mesh = DistributedInterface().get_device_mesh(Dim.CP)

    model_inputs = {
        k: v.to(dist.get_rank(), non_blocking=True) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)
    }

    model_inputs = padding_and_split_data(model_inputs, device_mesh)

    batch_size, _ = model_inputs["labels"].shape

    outputs: ModelOutput = model(**model_inputs)

    logits = outputs.logits.float()

    labels = model_inputs["labels"]

    cp_group = get_ulysses_sequence_parallel_group()
    cp_world_size = get_ulysses_sequence_parallel_world_size(cp_group)
    cp_rank = get_ulysses_sequence_parallel_rank(cp_group)

    # use all_gather to collect labels from all sequence parallel processes
    global_labels = [torch.empty_like(labels) for _ in range(cp_world_size)]
    dist.all_gather(global_labels, labels, group=cp_group)
    labels = torch.cat(global_labels, dim=1).contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = F.pad(shift_labels, (0, 1), value=-100)
    shift_labels = torch.chunk(shift_labels, chunks=cp_world_size, dim=1)[cp_rank].contiguous()

    # use all_gather to collect loss_weights from all sequence parallel processes
    loss_weights = model_inputs["loss_weights"]
    global_loss_weights = [torch.empty_like(loss_weights) for _ in range(cp_world_size)]
    dist.all_gather(global_loss_weights, loss_weights, group=cp_group)
    shift_loss_weights = torch.cat(global_loss_weights, dim=1).contiguous()
    shift_loss_weights = shift_loss_weights[..., 1:].contiguous()

    shift_logits = logits.view(-1, logits.size(-1)).contiguous()
    shift_labels = shift_labels.view(-1).contiguous()

    # use all_gather to collect log_probs from all sequence parallel processes
    log_probs = -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(batch_size, -1)
    global_log_probs = dist.nn.all_gather(log_probs, group=cp_group)
    global_log_probs = torch.cat(global_log_probs, dim=1).contiguous()
    log_probs = global_log_probs[..., :-1].contiguous()

    loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)

    return loss
