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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ....utils.plugin import ensure_methods_implemented


if TYPE_CHECKING:
    import torch

    from ....utils.types import HFModel, Processor


class BaseDistributed(ABC):
    """Contract for distributed backend method groups."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ensure_methods_implemented(cls)

    @staticmethod
    @abstractmethod
    def shard_model(model: HFModel, dist_config: object, **kwargs) -> object: ...

    @staticmethod
    @abstractmethod
    def save_model(model: HFModel, output_dir: str, processor: Processor) -> None: ...

    @staticmethod
    @abstractmethod
    def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None: ...

    @staticmethod
    @abstractmethod
    def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None: ...
