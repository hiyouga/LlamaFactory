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

from __future__ import annotations

from typing import TYPE_CHECKING

from ....utils import logging
from ....utils.plugin import BasePlugin


if TYPE_CHECKING:
    from ....config.arg_utils import PluginConfig
    from ....utils.types import HFModel


logger = logging.get_logger(__name__)


class OptimizerPlugin(BasePlugin):
    pass


@OptimizerPlugin("muon").register()
def create_muon_optimizer(model: HFModel, optim_config: PluginConfig):
    """Create a Muon optimizer.

    Muon is used for 2D "hidden" weight matrices; the remaining parameters (1D bias/LayerNorm,
    embeddings incl. GPT-2 ``wte``/``wpe``, the output ``lm_head``, and LoRA adapter factors) are
    optimized by the built-in AdamW.

    The Muon step is DTensor-aware: under FSDP2 it all-gathers the full gradient, runs Newton-Schulz
    on the full 2D matrix, then scatters the update back to the local shard. So it is correct under
    FSDP2 / sequence parallel (no longer approximate).
    """
    from .muon_optimizer import Muon

    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Muon is only appropriate for 2D "hidden" weight matrices. Route everything else to
            # the internal AdamW: 1D bias/norm, embeddings ("embed", GPT-2 "wte"/"wpe"), the output
            # head ("lm_head"), and LoRA adapter factors ("lora_A"/"lora_B"/"lora_embedding_*").
            if (
                param.ndim == 2
                and "embed" not in name
                and "lm_head" not in name
                and "wte" not in name
                and "wpe" not in name
                and "lora" not in name
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

    optimizer = Muon(
        lr=optim_config.get("lr", 1e-3),
        wd=optim_config.get("wd", 0.1),
        muon_params=muon_params,
        momentum=optim_config.get("momentum", 0.95),
        nesterov=optim_config.get("nesterov", True),
        ns_steps=optim_config.get("ns_steps", 5),
        adamw_params=adamw_params,
        adamw_betas=tuple(optim_config.get("adamw_betas", [0.9, 0.95])),
        adamw_eps=optim_config.get("adamw_eps", 1e-8),
    )
    logger.info_rank0(
        f"Using Muon optimizer with {len(muon_params)} Muon params and {len(adamw_params)} AdamW params."
    )
    return optimizer
