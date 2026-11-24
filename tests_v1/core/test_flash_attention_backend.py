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

from types import SimpleNamespace

import pytest

from llamafactory.v1.config import BatchingStrategy
from llamafactory.v1.core import base_trainer


@pytest.mark.parametrize(
    ("implementation", "supported"),
    [
        ("flash_attention_2", True),
        ("kernels-community/flash-attn2", True),
        (None, False),
        ("sdpa", False),
        ("eager", False),
        ("flash_attention_3", False),
        ("kernels-community/flash-attn3", False),
        ("other/flash-attn2", False),
        ("kernels-community/flash-attn2-other", False),
    ],
)
def test_flash_attention_2_backend_identity(implementation, supported):
    assert base_trainer._is_flash_attention_2(implementation) is supported


@pytest.mark.parametrize(
    "implementation", ["flash_attention_2", "kernels-community/flash-attn2", "sdpa", "other/kernel"]
)
def test_padding_free_accepts_only_flash_attention_2(implementation, monkeypatch):
    trainer = base_trainer.BaseTrainer.__new__(base_trainer.BaseTrainer)
    trainer.args = SimpleNamespace(
        batching_strategy=BatchingStrategy.PADDING_FREE,
        micro_batch_size=1,
        global_batch_size=1,
        cutoff_len=128,
        batching_workers=0,
        seed=42,
    )
    trainer.model = SimpleNamespace(config=SimpleNamespace(_attn_implementation=implementation))
    trainer.train_dataset = object()
    trainer.renderer = object()
    generator = object()
    monkeypatch.setattr(base_trainer, "BatchGenerator", lambda **kwargs: generator)
    if implementation in ("flash_attention_2", "kernels-community/flash-attn2"):
        trainer._create_batch_generator()
        assert trainer.train_batch_generator is generator
    else:
        with pytest.raises(ValueError, match="padding_free.*flash_attention_2"):
            trainer._create_batch_generator()
