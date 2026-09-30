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

"""Unit tests: FSDP2 meta-device loading vs normal loading consistency.

Validates that the FSDP2 meta loading path behaves correctly for tied weights
and non-persistent buffers by comparing it with the standard non-meta path.
"""

from types import SimpleNamespace

import torch
from transformers import AutoConfig

from llamafactory.v1.accelerator.interface import DistributedInterface
from llamafactory.v1.config.arg_parser import get_args
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.plugins.trainer_plugins.distributed import fsdp2 as fsdp2_module
from llamafactory.v1.plugins.trainer_plugins.distributed.fsdp2 import FSDP2Engine


TINY_MODEL = "llamafactory/tiny-random-qwen3"


def collect_non_persistent_buffers(model):
    """Collect all non-persistent buffers from model."""
    result = {}
    for mod_name, module in model.named_modules():
        for buf_name in getattr(module, "_non_persistent_buffers_set", set()):
            fqn = f"{mod_name}.{buf_name}" if mod_name else buf_name
            buf = getattr(module, buf_name, None)
            if buf is not None:
                result[fqn] = buf.detach().cpu().clone()
    return result


def test_fsdp2_rank0_syncs_non_persistent_buffers(monkeypatch):
    expected = torch.tensor([1.0, 0.5, 0.25])

    class Rank0Model(torch.nn.Module):
        def __init__(self, rank):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False)
            self._init_mode = "init_on_rank0"
            self.weight = torch.nn.Parameter(torch.ones(1))
            buffer = expected.clone() if rank == 0 else torch.empty_like(expected, device="meta")
            self.register_buffer("inv_freq", buffer, persistent=False)

        def to_empty(self, *, device, recurse=True):
            self.inv_freq = torch.zeros_like(expected, device=device)
            return self

    def set_model_state_dict(model, state_dict, **kwargs):
        assert torch.count_nonzero(model.inv_freq) == 0

    monkeypatch.setattr(fsdp2_module, "get_current_accelerator", lambda: torch.device("cpu"))
    monkeypatch.setattr(fsdp2_module, "set_model_state_dict", set_model_state_dict)

    for rank in (0, 1):
        engine = object.__new__(FSDP2Engine)
        engine.rank = rank
        engine.world_size = 2
        monkeypatch.setattr(engine, "prepare_model", lambda model: model)
        monkeypatch.setattr(engine, "_warmup_grad_norm", lambda model: None)

        def broadcast(buffer, src):
            assert src == 0
            if rank == 0:
                assert torch.equal(buffer, expected)
            else:
                assert torch.count_nonzero(buffer) == 0
                buffer.copy_(expected)

        monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
        model = engine.shard_model(Rank0Model(rank))

        assert "inv_freq" not in model.state_dict()
        assert torch.equal(model.inv_freq, expected)


def test_fsdp2_meta_loading_buffers_and_tied_weights():
    """Verify non-persistent buffers and tied weights consistency after meta load."""
    # 1. Initialize DistributedInterface for single process
    DistributedInterface()

    # 2. Build FSDP2Engine config
    engine = FSDP2Engine(
        {
            "name": "fsdp2",
            "mixed_precision": "bf16",
            "reshard_after_forward": True,
            "offload_params": False,
            "pin_memory": False,
            "dcp_path": None,
        }
    )

    config = AutoConfig.from_pretrained(TINY_MODEL)

    # --- NORMAL PATH ---
    normal_args, *_ = get_args(dict(model=TINY_MODEL, init_config=None))
    normal_engine = ModelEngine(model_args=normal_args)
    normal_model = normal_engine.model.to(torch.bfloat16)

    normal_model = engine.shard_model(normal_model)
    normal_non_persistent = collect_non_persistent_buffers(normal_model)

    del normal_model

    # --- META PATH ---
    meta_args, *_ = get_args(dict(model=TINY_MODEL, init_config={"name": "init_on_meta"}))
    meta_model_engine = ModelEngine(model_args=meta_args)
    meta_model = meta_model_engine.model

    assert meta_model.device.type == "meta", "Model should be on meta device"

    # Process meta device: save buffers -> tie_weights -> load from checkpoint -> restore buffers
    meta_model = engine.shard_model(meta_model)
    meta_non_persistent = collect_non_persistent_buffers(meta_model)

    # 3. Tied weights (embed_tokens.weight and lm_head.weight)

    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
    if tie_word_embeddings:
        assert meta_model.lm_head.weight is meta_model.model.embed_tokens.weight, (
            "Weights should be tied after loading"
        )

    del meta_model

    # 4. Non-persistent buffers (e.g., inv_freq)
    normal_buf_keys = set(normal_non_persistent.keys())
    meta_buf_keys = set(meta_non_persistent.keys())
    assert normal_buf_keys == meta_buf_keys, "Non-persistent buffer keys mismatch"

    for key in sorted(normal_buf_keys & meta_buf_keys):
        nb = normal_non_persistent[key]
        mb = meta_non_persistent[key]
        assert nb.shape == mb.shape, f"Buffer shape mismatch: {key}"
        assert torch.allclose(nb.float(), mb.float(), atol=1e-5), f"Buffer value mismatch: {key}"
