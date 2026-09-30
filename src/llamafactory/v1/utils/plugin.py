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

"""Lightweight plugin routing and shared parameter parsing helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from . import logging


logger = logging.get_logger(__name__)
ParamsT = TypeVar("ParamsT")


def ensure_methods_implemented(cls: type) -> None:
    """Raise when a static method-group implementation is incomplete."""
    required: set[str] = set()
    for base in cls.__mro__[1:]:
        required |= getattr(base, "__abstractmethods__", frozenset())

    missing = sorted(name for name in required if getattr(getattr(cls, name, None), "__isabstractmethod__", False))
    if missing:
        raise TypeError(f"{cls.__name__} does not implement all required methods: {missing}")


class BasePlugin:
    """Route a plugin name to one function or static method-group class.

    Every plugin family subclass owns an isolated registry. Parameter schemas
    deliberately do not live here; each plugin entrypoint parses its own config.
    """

    _registry: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def register(self):
        """Register one implementation object under this plugin name."""
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        cls = type(self)
        if self.name in cls._registry:
            logger.warning_rank0_once(f"Plugin {self.name!r} is already registered under {cls.__name__}.")

        def decorator(obj: Any) -> Any:
            cls._registry[self.name] = obj
            return obj

        return decorator

    @classmethod
    def parse_params(cls, config: Any, params_cls: type[ParamsT]) -> ParamsT:
        """Strictly convert config to the params dataclass used by one plugin entrypoint."""
        if not is_dataclass(params_cls):
            raise TypeError(f"{cls.__name__} params must be a dataclass type, got {params_cls!r}.")
        if isinstance(config, params_cls):
            return config
        if config is None:
            values = {}
        elif isinstance(config, dict):
            values = dict(config)
        else:
            raise TypeError(
                f"{cls.__name__} config must be a mapping or {params_cls.__name__}, got {type(config).__name__}."
            )

        known = {item.name for item in fields(params_cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Unknown params for {cls.__name__}.{params_cls.__name__}: {sorted(unknown)}. "
                f"Expected: {sorted(known)}"
            )

        return params_cls(**values)

    def _resolve(self) -> Any:
        cls = type(self)
        if self.name is None:
            raise ValueError(f"{cls.__name__} must be constructed with a name.")
        if self.name not in cls._registry:
            raise ValueError(f"Plugin {self.name!r} is not registered under {cls.__name__}.")
        return cls._registry[self.name]

    def __call__(self, *args, **kwargs) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._resolve(), attr)
