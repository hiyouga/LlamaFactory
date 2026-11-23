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

from collections.abc import Callable

from .....utils.logging import get_logger
from .....utils.types import HFModel


logger = get_logger(__name__)


def _get_model_type(model: HFModel) -> str | None:
    return getattr(getattr(model, "config", None), "model_type", None)


class FSDPTurboEPModelSpec:
    _registry: dict[str, "FSDPTurboEPModelSpec"] = {}

    def __init__(
        self,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
        prepare_fn: Callable[[HFModel], HFModel] | None = None,
    ) -> None:
        self.ep_modules = ep_modules
        self.ep_fsdp_modules = ep_fsdp_modules
        self.prepare_fn = prepare_fn

    @classmethod
    def register(
        cls,
        model_type: str,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
    ):
        def decorator(fn):
            cls._registry[model_type] = cls(
                ep_modules=ep_modules,
                ep_fsdp_modules=ep_fsdp_modules,
                prepare_fn=fn,
            )
            return fn

        return decorator

    @classmethod
    def get(cls, model: HFModel) -> "FSDPTurboEPModelSpec | None":
        model_type = _get_model_type(model)
        if model_type is None:
            return None
        return cls._registry.get(model_type)

    def prepare(self, model: HFModel) -> HFModel:
        if self.prepare_fn is None:
            return model
        return self.prepare_fn(model)


@FSDPTurboEPModelSpec.register(
    "qwen3_moe",
    ep_modules=["model.layers.{*}.mlp.experts"],
    ep_fsdp_modules=["model.layers.{*}.mlp.experts"],
)
def _prepare_qwen3_moe_for_ep(model: HFModel) -> HFModel:
    prepared = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "hidden_dim", "num_experts")):
            continue

        # FSDPTurbo's eager EP dispatcher expects sparse expert blocks to expose `hidden_size`.
        if not hasattr(module, "hidden_size"):
            module.hidden_size = module.hidden_dim
        prepared += 1

    if prepared:
        logger.info_rank0(f"FSDPTurbo EP adapter: prepared {prepared} sparse expert modules for Transformers 5.x.")
    else:
        logger.info_rank0("FSDPTurbo EP adapter did not find a sparse expert module requiring preparation.")
    return model


@FSDPTurboEPModelSpec.register(
    "qwen3_5_moe",
    ep_modules=["model.language_model.layers.{*}.mlp.experts"],
    ep_fsdp_modules=["model.language_model.layers.{*}.mlp.experts"],
)
def _prepare_qwen3_5_moe_for_ep(model: HFModel) -> HFModel:
    return model
