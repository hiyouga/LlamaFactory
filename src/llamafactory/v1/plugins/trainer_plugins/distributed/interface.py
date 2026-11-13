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

"""Distributed backend plugin definitions.

Backend-private params are parsed explicitly at ``shard_model``. ``DistributedInterface``
reads mesh topology from ``TrainingArguments`` and never puts it in backend params.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from ....utils.plugin import BasePlugin
from .base import BaseDistributed


if TYPE_CHECKING:
    from ....config.arg_utils import PluginConfig
    from ....utils.types import HFModel


@dataclass
class FSDP2Params:
    name: Literal["fsdp2"] = "fsdp2"
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None


@dataclass
class FSDPTurboParams:
    name: Literal["fsdpturbo"] = "fsdpturbo"
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None
    ep_size: int = 1
    ep_dispatcher: str = "eager"
    fsdp_ignored_modules: list[str] = field(default_factory=list)
    hook_modules: list[str] = field(default_factory=list)
    fsdp_implementation: str = "native"

    def __post_init__(self) -> None:
        if self.ep_size < 1:
            raise ValueError(f"ep_size must be positive, got {self.ep_size}.")


@dataclass
class DeepSpeedParams:
    name: Literal["deepspeed"] = "deepspeed"
    config_file: str = ""

    def __post_init__(self) -> None:
        if not self.config_file:
            raise ValueError("DeepSpeed config_file is required.")


class DistributedPlugin(BasePlugin):
    """Plugin family for distributed training backends."""


@DistributedPlugin("fsdp2").register()
class FSDP2Distributed(BaseDistributed):
    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | FSDP2Params, **kwargs) -> HFModel:
        dist_config = DistributedPlugin.parse_params(dist_config, FSDP2Params)
        from .fsdp2 import FSDP2Engine

        return FSDP2Engine(asdict(dist_config), bf16=bool(kwargs.get("bf16"))).shard_model(model)

    @staticmethod
    def save_model(model, output_dir, processor) -> None:
        from .fsdp2 import save_model

        save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .fsdp2 import save_checkpoint

        save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .fsdp2 import load_checkpoint

        load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("fsdpturbo").register()
class FSDPTurboDistributed(BaseDistributed):
    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | FSDPTurboParams, **kwargs) -> HFModel:
        dist_config = DistributedPlugin.parse_params(dist_config, FSDPTurboParams)
        from .fsdpturbo import FSDPTurboFSDP2Engine

        return FSDPTurboFSDP2Engine(asdict(dist_config), bf16=bool(kwargs.get("bf16"))).shard_model(model)

    @staticmethod
    def clip_grad_norm(model: HFModel, max_norm: float, **kwargs) -> float:
        from .fsdpturbo import clip_grad_norm_

        return clip_grad_norm_(model, max_norm, **kwargs)

    @staticmethod
    def save_model(model, output_dir, processor) -> None:
        from .fsdp2 import save_model

        save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .fsdp2 import save_checkpoint

        save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .fsdp2 import load_checkpoint

        load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("deepspeed").register()
class DeepSpeedDistributed(BaseDistributed):
    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | DeepSpeedParams, **kwargs) -> object:
        dist_config = DistributedPlugin.parse_params(dist_config, DeepSpeedParams)
        from .deepspeed import DeepSpeedEngine

        return DeepSpeedEngine(
            asdict(dist_config),
            num_micro_batch=kwargs.get("num_micro_batch"),
            micro_batch_size=kwargs.get("micro_batch_size"),
        ).shard_model(model)

    @staticmethod
    def save_model(model, output_dir, processor) -> None:
        from .deepspeed import save_model

        save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .deepspeed import save_checkpoint

        save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model, optimizer, ckpt_dir, **kwargs) -> None:
        from .deepspeed import load_checkpoint

        load_checkpoint(model, optimizer, ckpt_dir, **kwargs)
