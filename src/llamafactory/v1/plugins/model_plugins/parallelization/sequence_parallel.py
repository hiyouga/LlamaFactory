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

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ....accelerator.interface import Dim, DistributedInterface
from ....utils.constants import IGNORE_INDEX
from ....utils.plugin import BasePlugin
from .batch import prepare_sequence_parallel_batch
from .gdn_attention import apply_gdn_attention
from .ulysses import apply_ulysses_attention


class SequenceParallelModelPlugin(BasePlugin):
    def __call__(self, model, cp_size: int):
        return super().__call__(model, cp_size)


class SequenceParallelLossPlugin(BasePlugin):
    def __call__(self, model, inputs, *args, **kwargs):
        return super().__call__(model, inputs, *args, **kwargs)


@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, cp_size: int):
    from .hook import install_sequence_parallel_hook

    install_sequence_parallel_hook(model)
    group = DistributedInterface().get_group(Dim.CP)
    apply_ulysses_attention(model, cp_size, group)
    apply_gdn_attention(model, cp_size)


@SequenceParallelLossPlugin("sequence_parallel_loss").register()
def sequence_parallel_loss(model, model_inputs, loss_fn=None, *, uses_mrope: bool = False):
    """Prepare CP targets and aggregate weighted CE, optionally using a custom loss function.

    ``loss_fn`` receives ``(model, model_inputs, labels, loss_weights)``. Labels
    and weights are already shifted globally and sharded for the local CP rank.
    It must return a differentiable FP32 scalar weighted loss sum, without
    shifting targets again, normalizing, or performing CP collectives.
    """
    device_mesh = DistributedInterface().get_device_mesh(Dim.CP)

    prepared = prepare_sequence_parallel_batch(
        model_inputs,
        device=DistributedInterface().current_device,
        device_mesh=device_mesh,
        uses_mrope=uses_mrope,
    )
    labels = prepared.local_shift_labels
    loss_weights = prepared.local_shift_loss_weights
    if loss_fn is None:
        logits = model(**prepared.model_inputs).logits.float()
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none", ignore_index=IGNORE_INDEX
        )
        local_numerator = (token_loss * loss_weights.reshape(-1)).sum()
    else:
        local_numerator = loss_fn(model, prepared.model_inputs, labels, loss_weights)
    cp_group = device_mesh["cp"].get_group()

    # Do not average local mean losses: CP shards can own different supervised-token weights.
    # Gather the differentiable weighted numerators instead, reducing communication from
    # [batch, local_sequence] log probabilities to one scalar per CP rank.
    global_loss_numerators = dist.nn.all_gather(local_numerator.reshape(1), group=cp_group)
    global_loss_numerator = torch.cat(global_loss_numerators).sum()
    return global_loss_numerator / (prepared.global_loss_weight_sum + 1e-6)
