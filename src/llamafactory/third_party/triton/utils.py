# Copyright 2025 the LlamaFactory team.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
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

import contextlib
import functools
import itertools
import logging
import os
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
import triton.runtime.driver as driver
from packaging import version


logger = logging.getLogger(__name__)

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


def tensor_cache(fn: Optional[Callable[..., torch.Tensor]] = None, *, maxsize: int = 1) -> Any:
    """A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the outputs of the decorated function for the most recent
    set of input tensors, up to `maxsize` entries. If the function is called again with
    the same input tensors, it will return the cached result.

    When maxsize=1 (default), the behavior is identical to caching only the most recent result.
    Can be used as @tensor_cache or @tensor_cache(maxsize=n).

    Args:
        fn (Callable[..., torch.Tensor], optional):
            The function to be decorated when used without parentheses.
        maxsize (int):
            Maximum number of input combinations to cache. Default is 1.

    Returns:
        Callable[..., torch.Tensor]:
        A wrapped version of the input function with caching.
    """
    if maxsize < 1:
        raise ValueError("maxsize must be at least 1")

    def _is_match(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a is b
        try:
            return a == b
        except Exception:
            return a is b

    def _make_wrapper(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        cache: list = []

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i, (cached_args, cached_kwargs, cached_result) in enumerate(cache):
                if len(args) == len(cached_args) and len(kwargs) == len(cached_kwargs):
                    if all(_is_match(a, b) for a, b in zip(args, cached_args)) and all(
                        k in cached_kwargs and _is_match(v, cached_kwargs[k]) for k, v in kwargs.items()
                    ):
                        if i != 0:
                            cache.insert(0, cache.pop(i))
                        return cached_result

            result = fn(*args, **kwargs)
            cache.insert(0, (args, kwargs, result))
            if len(cache) > maxsize:
                cache.pop()
            return result

        return wrapper

    if fn is not None:
        return _make_wrapper(fn)
    return _make_wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache(maxsize=3)
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return "cpu"


def map_triton_backend_to_torch_device() -> str:
    backend = get_available_device()  # 'cuda' | 'hip' | 'xpu' | 'cpu' | ...
    return {"cuda": "cuda", "hip": "cuda", "xpu": "xpu"}.get(backend, backend)


device = get_available_device() if get_available_device() != "hip" else "cuda"
device_torch_lib = getattr(torch, device)
device_platform = get_available_device()
is_amd = device_platform == "hip"
is_nvidia = device_platform == "cuda"
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9
)

is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
is_tma_supported = (
    (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9)
    and os.environ.get("FLA_NO_USE_TMA", "0") != "1"
    and (
        hasattr(triton.language, "_experimental_make_tensor_descriptor")
        or hasattr(triton.language, "make_tensor_descriptor")
    )
)

if is_nvidia and not is_tf32_supported:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


@functools.cache
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


if check_pytorch_version("2.4"):
    device = "cuda" if device == "cpu" else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == "cuda", "Only cuda device is supported for PyTorch version < 2.4.0."
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """A decorator to make sure all input tensors are contiguous and set the device based on input tensors."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def _cpu_device_warning():
    warnings.warn(("Triton is not supported on current platform, roll back to CPU."), stacklevel=1)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def get_autotune_config(
    multibuffer_list: tuple = (False,),
    unit_flag_list: tuple = (False,),
    limit_auto_multi_buffer_only_for_local_buffer_list: tuple = (False,),
    limit_auto_multi_buffer_of_local_buffer_list: tuple = ("no-l0c",),
    set_workspace_multibuffer_list: tuple = (2, 4),
    enable_hivm_auto_cv_balance_list: tuple = (True,),
    tile_mix_vector_loop_num_list: tuple = (2, 4),
    tile_mix_cube_loop_num_list: tuple = (2, 4),
):
    configs = []
    for (
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        list(multibuffer_list),
        list(unit_flag_list),
        list(limit_auto_multi_buffer_only_for_local_buffer_list),
        list(limit_auto_multi_buffer_of_local_buffer_list),
    ):
        base_config_dict = {
            "multibuffer": multibuffer,
            "unit_flag": unit_flag,
            "limit_auto_multi_buffer_only_for_local_buffer": limit_auto_multi_buffer_only_for_local_buffer,
            "limit_auto_multi_buffer_of_local_buffer": limit_auto_multi_buffer_of_local_buffer,
        }

        if limit_auto_multi_buffer_only_for_local_buffer:
            configs.append(triton.Config(base_config_dict))
        else:
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                list(set_workspace_multibuffer_list),
                list(enable_hivm_auto_cv_balance_list),
                list(tile_mix_vector_loop_num_list),
                list(tile_mix_cube_loop_num_list),
            ):
                full_config_dict = base_config_dict.copy()
                full_config_dict.update(
                    {
                        "set_workspace_multibuffer": set_workspace_multibuffer,
                        "enable_hivm_auto_cv_balance": enable_hivm_auto_cv_balance,
                        "tile_mix_vector_loop": tile_mix_vector_loop,
                        "tile_mix_cube_loop": tile_mix_cube_loop,
                    }
                )
                configs.append(triton.Config(full_config_dict))
    return configs


def get_npu_properties():
    return driver.active.utils.get_device_properties(torch.npu.current_device())
