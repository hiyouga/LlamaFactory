# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/import_utils.py
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

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

import transformers.utils.import_utils as import_utils
from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_pyav_available():
    return _is_package_available("av")


def is_librosa_available():
    return _is_package_available("librosa")


def is_fastapi_available():
    return _is_package_available("fastapi")


def is_galore_available():
    return _is_package_available("galore_torch")


def is_apollo_available():
    return _is_package_available("apollo_torch")


def is_jieba_available():
    return _is_package_available("jieba")


def is_gradio_available():
    return _is_package_available("gradio")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_hyper_parallel_available():
    return _is_package_available("hyper_parallel")


def is_mcore_adapter_available():
    return _is_package_available("mcore_adapter")


def is_megatron_bridge_available():
    try:
        return _is_package_available("megatron.bridge")
    except ModuleNotFoundError:
        return False


def is_pillow_available():
    return _is_package_available("PIL")


def is_ray_available():
    return _is_package_available("ray")


def is_kt_available():
    return _is_package_available("kt_kernel")


def is_requests_available():
    return _is_package_available("requests")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_safetensors_available():
    return _is_package_available("safetensors")


def is_sglang_available():
    return _is_package_available("sglang")


def is_starlette_available():
    return _is_package_available("sse_starlette")


@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str):
    return _get_package_version("torch") >= version.parse(content)


def is_uvicorn_available():
    return _is_package_available("uvicorn")


def is_vllm_available():
    return _is_package_available("vllm")


_orig_is_package_available = import_utils._is_package_available


class PackageAvailability(tuple):
    __slots__ = ()

    def __new__(cls, available: bool, pkg_version: str = "N/A"):
        return super().__new__(cls, (bool(available), pkg_version))

    def __bool__(self) -> bool:
        return self[0]


def _patched_is_package_available(pkg_name: str, return_version: bool = False):
    available, version = _orig_is_package_available(pkg_name, return_version=return_version)

    return PackageAvailability(available, version)


if is_transformers_version_greater_than("5.3.0"):
    import_utils._is_package_available = _patched_is_package_available
