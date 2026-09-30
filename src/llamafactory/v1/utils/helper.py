# Copyright 2026 the LlamaFactory team.
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


import random

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from transformers import set_seed as hf_set_seed

from ..accelerator.helper import is_torch_npu_available
from ..accelerator.interface import DistributedInterface
from .constants import IGNORE_INDEX
from .types import BatchInput, Processor


def enable_full_determinism(seed: int) -> None:
    """Enable full deterministic mode for reproducible distributed training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if is_torch_npu_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """Set seed for reproducibility.

    Args:
        seed: Random seed.
        full_determinism: Whether to enable full deterministic mode.
    """
    if full_determinism:
        enable_full_determinism(seed)
    else:
        hf_set_seed(seed)


def is_tokenizer(processor: Processor) -> bool:
    """Check if processor is tokenizer.

    Args:
        processor: Processor.

    Returns:
        Whether processor is tokenizer.
    """
    return not hasattr(processor, "tokenizer")


def get_tokenizer(processor: Processor) -> PreTrainedTokenizer:
    """Get tokenizer from processor.

    Args:
        processor: Processor.

    Returns:
        Tokenizer.
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def compute_valid_tokens(batches: list[BatchInput]) -> int:
    """Compute valid tokens in batches.

    Args:
        batches: Batches.

    Returns:
        Number of valid tokens.
    """
    device = DistributedInterface().current_device
    return sum(
        (batch["labels"].to(device, non_blocking=True) != IGNORE_INDEX).sum().item()
        for batch in batches
        if "labels" in batch
    )


def model_uses_mrope(config) -> bool:
    """Whether the model uses multimodal RoPE (3D position ids built from grid_thw).

    Detected from the (text) config's rope settings carrying an ``mrope_section`` (Qwen2.5-VL /
    Qwen3-VL / Qwen3.5 family). Such models compute their own multimodal position ids inside
    ``forward`` when ``position_ids`` is not provided.
    """
    text_config = getattr(config, "text_config", config)
    rope = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None)
    return isinstance(rope, dict) and "mrope_section" in rope
