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

import inspect
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from ...utils import logging


logger = logging.get_logger(__name__)


_SUPPORTED_ACTIVITIES = {"auto", "all", "cpu", "device"}
_SUPPORTED_RANK_MODES = {"all", "rank0"}
_NPU_PROFILER_LEVELS = {
    "none": "Level_none",
    "level0": "Level0",
    "level1": "Level1",
    "level2": "Level2",
}
_NPU_AIC_METRICS = {
    "none": "AiCoreNone",
    "pipe_utilization": "PipeUtilization",
    "arithmetic_utilization": "ArithmeticUtilization",
    "memory": "Memory",
    "memory_l0": "MemoryL0",
    "memory_ub": "MemoryUB",
    "l2_cache": "L2Cache",
    "memory_access": "MemoryAccess",
    "resource_conflict_ratio": "ResourceConflictRatio",
}
_NPU_HOST_SYS = {
    "cpu": "CPU",
    "mem": "MEM",
    "disk": "DISK",
    "network": "NETWORK",
    "osrt": "OSRT",
}
_NPU_BACKEND_OPTIONS = {
    "data_simplification",
    "host_sys",
    "sys_io",
    "sys_interconnection",
    "gc_detect_threshold",
}


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: Optional[str] = None
    skip_first: int = 0
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 1
    repeat: int = 1
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    activities: str = "auto"
    rank_mode: str = "all"
    level: str = "level0"
    aic_metrics: str = "auto"
    backend_options: Optional[dict[str, Any]] = None
    explicit_profile_kwargs: Optional[set[str]] = None

    @classmethod
    def from_args(cls, args: Any) -> "ProfilerConfig":
        enable_torch_profiler = bool(getattr(args, "enable_torch_profiler", False))
        record_shapes = getattr(args, "profiler_record_shapes", None)
        profile_memory = getattr(args, "profiler_profile_memory", None)
        with_stack = getattr(args, "profiler_with_stack", None)
        explicit_profile_kwargs = _get_explicit_profile_kwargs(
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )
        if getattr(args, "profiler_with_flops", False):
            explicit_profile_kwargs.add("with_flops")
        if getattr(args, "profiler_with_modules", False):
            explicit_profile_kwargs.add("with_modules")
        return cls(
            enabled=enable_torch_profiler,
            output_dir=getattr(args, "profiler_output_dir", None),
            skip_first=getattr(args, "profiler_skip_first", 0),
            wait_steps=getattr(args, "profiler_wait_steps", 1),
            warmup_steps=getattr(args, "profiler_warmup_steps", 1),
            active_steps=getattr(args, "profiler_active_steps", 1),
            repeat=getattr(args, "profiler_repeat", 1),
            record_shapes=_resolve_optional_bool(record_shapes, enable_torch_profiler),
            profile_memory=_resolve_optional_bool(profile_memory, enable_torch_profiler),
            with_stack=_resolve_optional_bool(with_stack, enable_torch_profiler),
            with_flops=getattr(args, "profiler_with_flops", False),
            with_modules=getattr(args, "profiler_with_modules", False),
            activities=getattr(args, "profiler_activities", "auto"),
            rank_mode=getattr(args, "profiler_rank_mode", "all"),
            level=getattr(args, "profiler_level", "level0"),
            aic_metrics=getattr(args, "profiler_aic_metrics", "auto"),
            backend_options=_parse_backend_options(getattr(args, "profiler_backend_options", None)),
            explicit_profile_kwargs=explicit_profile_kwargs,
        )

    def validate(self, backend_name: Optional[str] = None) -> None:
        if not self.enabled:
            return

        _validate_int_option("profiler_skip_first", self.skip_first, min_value=0)
        _validate_int_option("profiler_wait_steps", self.wait_steps, min_value=0)
        _validate_int_option("profiler_warmup_steps", self.warmup_steps, min_value=0)
        _validate_int_option("profiler_active_steps", self.active_steps, min_value=1)
        _validate_int_option("profiler_repeat", self.repeat, min_value=0)
        _validate_choice_option("profiler_activities", self.activities, _SUPPORTED_ACTIVITIES)
        _validate_choice_option("profiler_rank_mode", self.rank_mode, _SUPPORTED_RANK_MODES)

        if backend_name == "npu" and self.activities != "cpu":
            self.npu_profiler_level_name()
            self.npu_aic_metrics_name()
            self.npu_backend_options()
        self.schedule_kwargs()

    def npu_profiler_level_name(self) -> str:
        key = _validate_choice_option("profiler_level", self.level, _NPU_PROFILER_LEVELS)
        return _NPU_PROFILER_LEVELS[key]

    def npu_aic_metrics_name(self) -> str:
        key = _validate_choice_option("profiler_aic_metrics", self.aic_metrics, {"auto", *_NPU_AIC_METRICS})
        if key == "auto":
            if self.npu_profiler_level_name() in ("Level1", "Level2"):
                return "PipeUtilization"
            return "AiCoreNone"

        metric_name = _NPU_AIC_METRICS[key]
        if metric_name != "AiCoreNone" and self.npu_profiler_level_name() not in ("Level1", "Level2"):
            raise ValueError(
                "`profiler_aic_metrics` requires `profiler_level` to be `level1` or `level2` on NPU."
            )

        return metric_name

    def npu_backend_options(self) -> dict[str, Any]:
        if self.backend_options is None:
            return {}

        unsupported_backends = set(self.backend_options) - {"npu"}
        if unsupported_backends:
            raise ValueError(f"`profiler_backend_options` only supports `npu`, got {sorted(unsupported_backends)}.")

        if "npu" not in self.backend_options or self.backend_options["npu"] is None:
            return {}

        npu_options = self.backend_options["npu"]
        if not isinstance(npu_options, dict):
            raise ValueError("`profiler_backend_options.npu` must be a mapping.")

        unsupported_options = set(npu_options) - _NPU_BACKEND_OPTIONS
        if unsupported_options:
            raise ValueError(
                f"`profiler_backend_options.npu` only supports {sorted(_NPU_BACKEND_OPTIONS)}, "
                f"got {sorted(unsupported_options)}."
            )

        normalized_options = dict(npu_options)
        if "data_simplification" in normalized_options and not isinstance(
            normalized_options["data_simplification"], bool
        ):
            raise ValueError("`profiler_backend_options.npu.data_simplification` must be a boolean.")
        if "sys_io" in normalized_options and not isinstance(normalized_options["sys_io"], bool):
            raise ValueError("`profiler_backend_options.npu.sys_io` must be a boolean.")
        if "sys_interconnection" in normalized_options and not isinstance(
            normalized_options["sys_interconnection"], bool
        ):
            raise ValueError("`profiler_backend_options.npu.sys_interconnection` must be a boolean.")
        if normalized_options.get("gc_detect_threshold") is not None:
            threshold = normalized_options["gc_detect_threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold < 0:
                raise ValueError("`profiler_backend_options.npu.gc_detect_threshold` must be null or >= 0.")

        normalized_options["host_sys"] = _normalize_host_sys(normalized_options.get("host_sys", []))
        return normalized_options

    def explicit_npu_backend_options(self) -> set[str]:
        if self.backend_options is None:
            return set()

        npu_options = self.backend_options.get("npu")
        if not isinstance(npu_options, dict):
            return set()

        return set(npu_options)

    def schedule_kwargs(self) -> dict[str, int]:
        return dict(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
            skip_first=self.skip_first,
        )


class _ProfilerBackend:
    def __init__(self, name: str, profiler_module: Any, device_activity_name: Optional[str]) -> None:
        self.name = name
        self.profiler_module = profiler_module
        self.device_activity_name = device_activity_name

    def build_schedule(self, kwargs: dict[str, Any]) -> Any:
        if self.name == "npu":
            return self.profiler_module.schedule(
                wait=kwargs["wait"],
                active=kwargs["active"],
                warmup=kwargs["warmup"],
                repeat=kwargs["repeat"],
                skip_first=kwargs["skip_first"],
            )

        return self.profiler_module.schedule(**kwargs)

    def build_activities(self, config: ProfilerConfig) -> list[Any]:
        activity_cls = self.profiler_module.ProfilerActivity
        activities = []
        if config.activities in ("auto", "all", "cpu"):
            activities.append(activity_cls.CPU)
        if config.activities == "device" and self.device_activity_name is None:
            raise ValueError("`profiler_activities: device` requires a CUDA or NPU backend.")
        if config.activities in ("auto", "all", "device"):
            if self.device_activity_name is not None:
                activities.append(getattr(activity_cls, self.device_activity_name))
        return activities

    def build_experimental_config(self, config: ProfilerConfig, logger: Any = None) -> Optional[Any]:
        if self.name != "npu" or config.activities == "cpu":
            return None

        profiler = self.profiler_module
        npu_options = config.npu_backend_options()
        config_kwargs = dict(
            export_type=[profiler.ExportType.Text],
            profiler_level=_get_profiler_enum_value(
                profiler, "ProfilerLevel", config.npu_profiler_level_name(), "`profiler_level`"
            ),
            aic_metrics=_get_profiler_enum_value(
                profiler, "AiCMetrics", config.npu_aic_metrics_name(), "`profiler_aic_metrics`"
            ),
            l2_cache=False,
            op_attr=False,
            data_simplification=npu_options.get("data_simplification", True),
            record_op_args=False,
            gc_detect_threshold=npu_options.get("gc_detect_threshold"),
            host_sys=[
                _get_profiler_enum_value(
                    profiler, "HostSystem", item, "`profiler_backend_options.npu.host_sys`"
                )
                for item in npu_options.get("host_sys", [])
            ],
            sys_io=npu_options.get("sys_io", False),
            sys_interconnection=npu_options.get("sys_interconnection", False),
            mstx=False,
            mstx_domain_include=[],
            mstx_domain_exclude=[],
        )
        explicit_keys = {"profiler_level", "aic_metrics"}
        explicit_keys.update(config.explicit_npu_backend_options())
        return profiler._ExperimentalConfig(
            **_filter_supported_kwargs(
                profiler._ExperimentalConfig,
                config_kwargs,
                logger=logger,
                explicit_keys=explicit_keys,
            )
        )


def _get_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()

    return int(os.getenv("RANK", "0"))


def _get_current_accelerator_type() -> str:
    if is_torch_npu_available():
        return "npu"
    if is_torch_cuda_available():
        return "cuda"
    return "cpu"


def _parse_backend_options(options: Any) -> Optional[dict[str, Any]]:
    if options is None:
        return None

    if not isinstance(options, dict):
        raise ValueError("`profiler_backend_options` must be a mapping.")

    return options


def _resolve_optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default

    return bool(value)


def _validate_int_option(name: str, value: Any, min_value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
        if min_value == 1:
            raise ValueError(f"`{name}` must be a positive integer.")

        raise ValueError(f"`{name}` must be an integer greater than or equal to {min_value}.")


def _validate_choice_option(name: str, value: Any, supported: set[str] | dict[str, Any]) -> str:
    if not isinstance(value, str) or value not in supported:
        raise ValueError(f"`{name}` must be one of {sorted(supported)}.")

    return value


def _get_explicit_profile_kwargs(**kwargs: Any) -> set[str]:
    name_mapping = {
        "record_shapes": "record_shapes",
        "profile_memory": "profile_memory",
        "with_stack": "with_stack",
    }
    return {profile_key for arg_key, profile_key in name_mapping.items() if kwargs[arg_key] is not None}


def _normalize_host_sys(host_sys: Any) -> list[str]:
    if host_sys is None:
        return []

    if not isinstance(host_sys, list):
        raise ValueError("`profiler_backend_options.npu.host_sys` must be a list.")

    normalized = []
    for item in host_sys:
        key = _validate_choice_option("profiler_backend_options.npu.host_sys", item, _NPU_HOST_SYS)
        normalized.append(_NPU_HOST_SYS[key])

    return normalized


def _get_backend() -> _ProfilerBackend:
    accelerator_type = _get_current_accelerator_type()
    if accelerator_type == "cpu":
        return _ProfilerBackend("cpu", torch.profiler, None)

    if accelerator_type == "cuda":
        return _ProfilerBackend("cuda", torch.profiler, "CUDA")

    if accelerator_type == "npu":
        try:
            import torch_npu  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NPU profiler requires `torch_npu` to be installed.") from exc

        return _ProfilerBackend("npu", torch_npu.profiler, "NPU")

    raise RuntimeError(f"Profiler only supports CPU, CUDA and NPU devices, got current accelerator: {accelerator_type}.")


def _get_profiler_enum_value(profiler: Any, enum_name: str, member_name: str, option_name: str) -> Any:
    enum_cls = getattr(profiler, enum_name, None)
    if enum_cls is not None and hasattr(enum_cls, member_name):
        return getattr(enum_cls, member_name)

    supported = [] if enum_cls is None else [name for name in dir(enum_cls) if not name.startswith("_")]
    package_name = str(getattr(profiler, "__name__", "")).split(".", maxsplit=1)[0]
    torch_npu_version = getattr(sys.modules.get(package_name), "__version__", None)
    raise RuntimeError(
        f"The installed torch_npu profiler does not support {option_name}={member_name}. "
        f"Supported values: {supported}. torch_npu version: {torch_npu_version or 'unknown'}."
    )


def _filter_supported_kwargs(
    fn: Any,
    kwargs: dict[str, Any],
    logger: Any = None,
    explicit_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    parameters = inspect.signature(fn).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    supported = set(parameters)
    dropped = set(kwargs) - supported
    if dropped:
        explicit_dropped = dropped & (explicit_keys or set())
        message_level = "warning" if explicit_dropped else "debug"
        if explicit_dropped or logger is not None:
            _log(
                logger,
                message_level,
                f"Profiler backend ignored unsupported arguments: {sorted(dropped)}.",
            )

    return {key: value for key, value in kwargs.items() if key in supported}


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return

    rank_method = f"{level}_rank0"
    if hasattr(logger, rank_method):
        getattr(logger, rank_method)(message)
    else:
        getattr(logger, level)(message)


class ProfilerController:
    def __init__(self, args: Any, logger: Any = None) -> None:
        self.config = ProfilerConfig.from_args(args)
        self.logger = logger
        self.profiler = None
        self.backend_name: Optional[str] = None
        self.trace_dir: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self, output_dir: str, initial_step: int = 0) -> None:
        self.stop()
        if not self.config.enabled:
            return

        backend = _get_backend()
        self.backend_name = backend.name
        self.config.validate(backend.name)
        if backend.name == "npu" and os.getenv("PROF_CONFIG_PATH"):
            _log(
                self.logger,
                "warning",
                "`PROF_CONFIG_PATH` is set. Do not enable dynamic_profile together with "
                "`enable_torch_profiler`.",
            )
        schedule = self.config.schedule_kwargs()

        rank = _get_rank()
        if self.config.rank_mode == "rank0" and rank != 0:
            return

        trace_root = self.config.output_dir or os.path.join(output_dir, "profiler")
        self.trace_dir = os.path.realpath(os.path.join(trace_root, f"rank_{rank}"))
        os.makedirs(self.trace_dir, exist_ok=True)

        profile_kwargs = dict(
            activities=backend.build_activities(self.config),
            schedule=backend.build_schedule(schedule),
            on_trace_ready=backend.profiler_module.tensorboard_trace_handler(self.trace_dir),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
            experimental_config=backend.build_experimental_config(self.config, logger=self.logger),
        )
        profiler = backend.profiler_module.profile(
            **_filter_supported_kwargs(
                backend.profiler_module.profile,
                profile_kwargs,
                logger=self.logger,
                explicit_keys=set(self.config.explicit_profile_kwargs or set()),
            )
        )
        try:
            profiler.start()
        except Exception:
            with suppress(Exception):
                profiler.stop()
            raise

        self.profiler = profiler

        first_active_step = initial_step + schedule["skip_first"] + schedule["wait"] + schedule["warmup"] + 1
        schedule_message = (
            f"skip_first={schedule['skip_first']}, wait={schedule['wait']}, warmup={schedule['warmup']}, "
            f"active={schedule['active']}, repeat={schedule['repeat']}, first_active_step={first_active_step}"
        )
        _log(
            self.logger,
            "info",
            f"Profiler started on {backend.name}: {schedule_message}, initial_step={initial_step}. Traces -> {trace_root}",
        )

    def step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def stop(self) -> None:
        profiler, self.profiler = self.profiler, None
        if profiler is None:
            return

        profiler.stop()
        _log(self.logger, "info", "Profiler stopped.")
