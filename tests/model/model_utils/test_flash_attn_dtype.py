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

"""Tests for FlashAttention dtype validation.

Verifies that configure_attn_implementation raises ValueError when
FlashAttention (fa2/fa3) is used with float32 compute dtype.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llamafactory.extras.constants import AttentionFunction
from llamafactory.model.model_utils.attention import configure_attn_implementation


def _make_model_args(**overrides):
    """Create a mock ModelArguments with sensible defaults."""
    args = MagicMock()
    args.flash_attn = overrides.get("flash_attn", AttentionFunction.AUTO)
    args.compute_dtype = overrides.get("compute_dtype", None)
    args.infer_dtype = overrides.get("infer_dtype", "auto")
    return args


def _make_config(model_type="llama"):
    """Create a mock PretrainedConfig."""
    config = MagicMock()
    config.model_type = model_type
    return config


@patch("transformers.utils.is_flash_attn_2_available", return_value=True)
@patch("transformers.is_torch_npu_available", return_value=False)
def test_fa2_with_float32_raises(mock_npu, mock_fa2):
    """FA2 + float32 should raise ValueError."""
    model_args = _make_model_args(flash_attn=AttentionFunction.FA2, compute_dtype=torch.float32)
    config = _make_config()
    with pytest.raises(ValueError, match="incompatible with float32"):
        configure_attn_implementation(config, model_args)


@patch("transformers.utils.is_flash_attn_2_available", return_value=True)
@patch("transformers.is_torch_npu_available", return_value=False)
def test_fa2_with_bfloat16_ok(mock_npu, mock_fa2):
    """FA2 + bfloat16 should not raise."""
    model_args = _make_model_args(flash_attn=AttentionFunction.FA2, compute_dtype=torch.bfloat16)
    config = _make_config()
    configure_attn_implementation(config, model_args)


@patch("transformers.utils.is_flash_attn_2_available", return_value=True)
@patch("transformers.is_torch_npu_available", return_value=False)
def test_fa2_with_float16_ok(mock_npu, mock_fa2):
    """FA2 + float16 should not raise."""
    model_args = _make_model_args(flash_attn=AttentionFunction.FA2, compute_dtype=torch.float16)
    config = _make_config()
    configure_attn_implementation(config, model_args)


def test_sdpa_with_float32_ok():
    """SDPA + float32 should not raise (SDPA supports float32)."""
    model_args = _make_model_args(flash_attn=AttentionFunction.SDPA, compute_dtype=torch.float32)
    config = _make_config()
    configure_attn_implementation(config, model_args)


def test_disabled_with_float32_ok():
    """Disabled attention + float32 should not raise."""
    model_args = _make_model_args(flash_attn=AttentionFunction.DISABLED, compute_dtype=torch.float32)
    config = _make_config()
    configure_attn_implementation(config, model_args)


def test_auto_skips_validation():
    """Auto mode should skip entirely without validation."""
    model_args = _make_model_args(flash_attn=AttentionFunction.AUTO, compute_dtype=torch.float32)
    config = _make_config()
    configure_attn_implementation(config, model_args)
