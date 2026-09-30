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

import json
from copy import deepcopy
from functools import lru_cache
from typing import Any


_registered_dist_config: Any | None = None


def register_deepspeed_dist_config(dist_config: Any | None) -> None:
    """Register backend config before model loading without involving the accelerator."""
    global _registered_dist_config
    _registered_dist_config = dist_config
    is_deepspeed_zero3_enabled.cache_clear()


@lru_cache(maxsize=1)
def is_deepspeed_zero3_enabled() -> bool:
    dist_config = _registered_dist_config
    if dist_config is None or getattr(dist_config, "name", None) != "deepspeed":
        return False

    config_file = dist_config.get("config_file")
    if not config_file:
        return False
    return _load_deepspeed_config(config_file).get("zero_optimization", {}).get("stage") == 3


def _normalize_precision_enabled(value: Any) -> bool | str:
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == "true":
            return True
        if value_lower == "false":
            return False
        if value_lower == "auto":
            return "auto"
    return value


def infer_deepspeed_mixed_precision(ds_config: dict[str, Any]) -> str:
    ds_config.setdefault("fp16", {})
    ds_config.setdefault("bf16", {})

    fp16_enabled = _normalize_precision_enabled(ds_config["fp16"].get("enabled", "auto"))
    bf16_enabled = _normalize_precision_enabled(ds_config["bf16"].get("enabled", "auto"))

    # This project only supports DeepSpeed bf16 or no mixed precision.
    if fp16_enabled is True:
        raise ValueError("DeepSpeed only supports bf16 mixed precision for now, fp16 is not supported.")

    if bf16_enabled is True:
        mixed_precision = "bf16"
    elif bf16_enabled is False:
        mixed_precision = "no"
    elif fp16_enabled is False:
        mixed_precision = "no"
    else:
        # When both bf16/fp16 are left as auto (or absent), default to bf16.
        mixed_precision = "bf16"

    ds_config["fp16"]["enabled"] = False
    ds_config["bf16"]["enabled"] = mixed_precision == "bf16"
    return mixed_precision


def _unset_hf_deepspeed_config() -> None:
    try:
        from transformers.integrations import unset_hf_deepspeed_config
    except ImportError:
        from transformers.deepspeed import unset_hf_deepspeed_config

    unset_hf_deepspeed_config()


def _load_deepspeed_config(config_file: str) -> dict[str, Any]:
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


def setup_deepspeed_zero3_model_loading():
    """Enable ZeRO-3-aware model loading for the registered backend config."""
    dist_config = _registered_dist_config
    config_file = dist_config.get("config_file") if dist_config is not None else None
    if not config_file:
        raise ValueError("DeepSpeed config_file is required in dist_config")

    from accelerate.utils import DeepSpeedPlugin

    try:
        from transformers.integrations import is_deepspeed_zero3_enabled as _hf_is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled as _hf_is_deepspeed_zero3_enabled

    # DeepSpeed configs often use "auto" placeholders that only make sense once
    # we know the current runtime batch settings and precision mode.
    ds_config = deepcopy(_load_deepspeed_config(config_file))
    if "gradient_accumulation_steps" not in ds_config or ds_config["gradient_accumulation_steps"] == "auto":
        ds_config["gradient_accumulation_steps"] = 1
    if "train_micro_batch_size_per_gpu" not in ds_config or ds_config["train_micro_batch_size_per_gpu"] == "auto":
        ds_config["train_micro_batch_size_per_gpu"] = 1
    if ds_config.get("train_batch_size") == "auto":
        ds_config.pop("train_batch_size")

    zero_stage = ds_config.get("zero_optimization", {}).get("stage")
    if zero_stage != 3:
        return None

    # ZeRO-3 model loading needs concrete fp16/bf16 flags, not "auto".
    mixed_precision = infer_deepspeed_mixed_precision(ds_config)

    plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=True)

    if not plugin.hf_ds_config.is_zero3():
        return None

    # Reuse the same precision inference rule as the training-time DeepSpeed path
    # so both model-loading and engine setup stay aligned.
    plugin.set_mixed_precision(mixed_precision)
    plugin.set_deepspeed_weakref()

    if not _hf_is_deepspeed_zero3_enabled():
        raise RuntimeError(
            "DeepSpeed ZeRO-3 model-loading bootstrap failed: transformers still reports zero3 disabled "
            "after constructing HfDeepSpeedConfig. This usually means the runtime is using a different transformers "
            "installation than expected, or the DeepSpeed global state was not established correctly."
        )
    return plugin


def teardown_deepspeed_zero3_model_loading(plugin) -> None:
    if plugin is not None:
        _unset_hf_deepspeed_config()
