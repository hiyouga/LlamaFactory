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

from abc import ABC, abstractmethod

from ....utils.plugin import BasePlugin, ensure_methods_implemented
from ....utils.types import HFModel


class KernelPlugin(BasePlugin):
    """Plugin family for model kernel optimization classes."""


class BaseKernel(ABC):
    """Template base for concrete kernel implementations."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ensure_methods_implemented(cls)

    @staticmethod
    @abstractmethod
    def check_device() -> None: ...

    @staticmethod
    def check_deps() -> None:
        pass

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        cls.check_device()
        cls.check_deps()
        if kwargs.get("model") is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")

        return cls._apply(**kwargs)

    @staticmethod
    @abstractmethod
    def _apply(**kwargs) -> HFModel: ...
