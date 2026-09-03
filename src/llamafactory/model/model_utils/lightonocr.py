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

"""
Auto-patcher for LightOnOCR-2 model configs.

LightOnOCR-2 models on HuggingFace Hub ship with ``model_type: "mistral3"`` in
their ``config.json``, but transformers >= 5.1 has native ``lighton_ocr`` support
with the correct weight naming (``vision_encoder`` / ``vision_projection`` instead
of Mistral3's ``vision_tower`` / ``multi_modal_projector``).  Loading without this
patch causes **all vision weights to be randomly initialized** (MISSING in the
load report).

Additionally, the ``processor_config.json`` stores ``patch_size`` as a bare
integer (``14``), which triggers thousands of noisy INFO-level log messages from
``image_processing_utils.get_size_dict`` on every image processed.

This module provides :func:`patch_lightonocr_configs` which resolves and patches
both files **in-place on disk** (HuggingFace cache or local directory) so that
subsequent ``AutoConfig`` / ``AutoProcessor`` calls load the correct classes and
suppress the log spam.  The function is idempotent — it only writes when a change
is actually needed.
"""

import json
import os
from pathlib import Path
from typing import Optional

from ...extras import logging


logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Architecture / model_type mapping
# ---------------------------------------------------------------------------
_LIGHTONOCR_ARCH_OLD = "LightOnOCRForConditionalGeneration"
_LIGHTONOCR_ARCH_NEW = "LightOnOcrForConditionalGeneration"
_MODEL_TYPE_OLD = "mistral3"
_MODEL_TYPE_NEW = "lighton_ocr"


def _resolve_cached_file(model_name_or_path: str, filename: str, **hub_kwargs) -> Optional[str]:
    """Resolve a file inside a local directory or HuggingFace cache."""
    local_path = os.path.join(model_name_or_path, filename)
    if os.path.isfile(local_path):
        return local_path

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=model_name_or_path,
            filename=filename,
            cache_dir=hub_kwargs.get("cache_dir"),
            token=hub_kwargs.get("token"),
            revision=hub_kwargs.get("revision"),
            local_files_only=True,
        )
    except Exception:
        return None


def _needs_config_patch(data: dict) -> bool:
    """Check whether config.json needs the lighton_ocr model_type patch."""
    architectures = data.get("architectures", [])
    model_type = data.get("model_type", "")
    has_lightonocr_arch = any("lightonocr" in a.lower() or "light_on_ocr" in a.lower() for a in architectures)
    return has_lightonocr_arch and model_type == _MODEL_TYPE_OLD


def _needs_processor_patch(data: dict) -> bool:
    """Check whether processor_config.json needs the patch_size dict patch."""
    image_processor = data.get("image_processor", {})
    patch_size = image_processor.get("patch_size")
    return isinstance(patch_size, int)


def _patch_config_json(filepath: str) -> bool:
    """Patch config.json in-place.  Returns True if a change was made."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not _needs_config_patch(data):
        return False

    data["model_type"] = _MODEL_TYPE_NEW
    data["architectures"] = [
        _LIGHTONOCR_ARCH_NEW if a == _LIGHTONOCR_ARCH_OLD else a
        for a in data.get("architectures", [])
    ]

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return True


def _patch_processor_config_json(filepath: str) -> bool:
    """Patch processor_config.json in-place.  Returns True if a change was made."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not _needs_processor_patch(data):
        return False

    ps = data["image_processor"]["patch_size"]
    data["image_processor"]["patch_size"] = {"height": ps, "width": ps}

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return True


def patch_lightonocr_configs(model_name_or_path: str, **hub_kwargs) -> None:
    """Auto-detect and patch LightOnOCR-2 config files if necessary.

    Call this **before** ``AutoConfig.from_pretrained`` /
    ``AutoProcessor.from_pretrained`` to ensure the correct model class is
    loaded and the processor doesn't spam INFO logs.

    Parameters
    ----------
    model_name_or_path : str
        Local directory or HuggingFace Hub model id.
    **hub_kwargs
        Forwarded to ``hf_hub_download`` (``cache_dir``, ``token``,
        ``revision``).
    """
    config_path = _resolve_cached_file(model_name_or_path, "config.json", **hub_kwargs)
    if config_path is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if _needs_config_patch(data):
                if _patch_config_json(config_path):
                    logger.info_rank0(
                        "Patched LightOnOCR-2 config.json: "
                        f"model_type '{_MODEL_TYPE_OLD}' -> '{_MODEL_TYPE_NEW}', "
                        f"architecture -> '{_LIGHTONOCR_ARCH_NEW}'"
                    )
        except Exception as e:
            logger.warning_rank0(f"Failed to patch LightOnOCR-2 config.json: {e}")

    processor_path = _resolve_cached_file(model_name_or_path, "processor_config.json", **hub_kwargs)
    if processor_path is not None:
        try:
            with open(processor_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if _needs_processor_patch(data):
                if _patch_processor_config_json(processor_path):
                    logger.info_rank0(
                        "Patched LightOnOCR-2 processor_config.json: "
                        "patch_size int -> dict"
                    )
        except Exception as e:
            logger.warning_rank0(f"Failed to patch LightOnOCR-2 processor_config.json: {e}")
