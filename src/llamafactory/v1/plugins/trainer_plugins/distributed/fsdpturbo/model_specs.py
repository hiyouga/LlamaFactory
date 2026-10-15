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

from .....utils.types import HFModel


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


# These architectures already satisfy the pinned backend's expert interface.
# Intrinsic paths are data; no empty preparation callbacks or legacy aliases are needed.
FSDPTurboEPModelSpec._registry.update(
    {
        model_type: FSDPTurboEPModelSpec(ep_modules=[path], ep_fsdp_modules=[path])
        for model_type, path in {
            "qwen3_moe": "model.layers.{*}.mlp.experts",
            "qwen3_5_moe": "model.language_model.layers.{*}.mlp.experts",
            "deepseek_v4": "model.layers.{*}.mlp.experts",
        }.items()
    }
)
