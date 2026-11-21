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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from llamafactory.v1.core import base_trainer
from llamafactory.v1.core.base_trainer import BaseTrainer
from llamafactory.v1.plugins.trainer_plugins.distributed import interface as plugin_interface
from llamafactory.v1.utils import helper


def test_fit_finalizes_accumulated_gradients_before_clipping_and_step(monkeypatch):
    """Exercise the real optimizer boundary with two micro-batches per update."""
    events = []
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.ones_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.register_step_post_hook(lambda *_: events.append("step"))

    def loss(batch):
        result = model.weight.square().sum() * batch["scale"]
        result.register_hook(lambda grad: events.append("backward"))
        return result

    def synchronize(candidate):
        assert candidate is model
        assert events[-2:] == ["backward", "backward"]
        events.append("synchronize")
        # Simulate the final reduction's effect; real CP reduction is tested with Gloo meshes.
        model.weight.grad.div_(2)

    def clip(candidate, limit):
        assert candidate is model
        assert events[-1] == "synchronize"
        events.append("clip")
        return torch.nn.utils.clip_grad_norm_(candidate.parameters(), limit).item()

    plugin = SimpleNamespace(synchronize_gradients=synchronize, clip_grad_norm=clip)
    monkeypatch.setattr(plugin_interface, "DistributedPlugin", lambda _: plugin)
    context = SimpleNamespace(
        current_device=torch.device("cpu"), all_reduce=lambda value, **_: value, sync=lambda: None
    )
    monkeypatch.setattr(base_trainer, "DistributedInterface", lambda: context)
    monkeypatch.setattr(helper, "DistributedInterface", lambda: context)

    class Batches(list):
        def set_epoch(self, epoch):
            pass

    micro_batches = [{"labels": torch.ones(1, 1), "scale": scale} for scale in (1.0, 3.0)]
    trainer = SimpleNamespace(
        model=model,
        args=SimpleNamespace(
            dist_config=SimpleNamespace(name="fsdpturbo"), max_grad_norm=1.0, logging_steps=1, save_steps=None
        ),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        callback_handler=Mock(),
        state=SimpleNamespace(),
        global_step=0,
        num_training_steps=2,
        _resume_epoch=0,
        _deepspeed_engine=None,
        dp_size=1,
        compute_loss=loss,
        train_batch_generator=Batches([micro_batches, micro_batches]),
    )
    BaseTrainer.fit(trainer)
    assert events == ["backward", "backward", "synchronize", "clip", "step"] * 2
    assert model.weight.item() == pytest.approx(0.8, abs=1e-6)
    assert model.weight.grad is None


@pytest.mark.parametrize("attention", ["eager", "flash_attention_2"])
@pytest.mark.parametrize("strategy", ["normal", "dynamic_batching", "padding_free", "dynamic_padding_free"])
def test_padding_free_batching_requires_flash_attention(monkeypatch, attention, strategy):
    from llamafactory.v1.config.arg_utils import BatchingStrategy

    strategy = BatchingStrategy(strategy)
    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=attention)),
        train_dataset=[],
        renderer=None,
        args=SimpleNamespace(
            batching_strategy=strategy,
            micro_batch_size=1,
            global_batch_size=1,
            cutoff_len=8,
            batching_workers=0,
            seed=42,
        ),
    )
    created = []
    monkeypatch.setattr(base_trainer, "BatchGenerator", lambda **kwargs: created.append(kwargs) or kwargs)
    if (
        strategy in (BatchingStrategy.PADDING_FREE, BatchingStrategy.DYNAMIC_PADDING_FREE)
        and attention != "flash_attention_2"
    ):
        with pytest.raises(ValueError, match=strategy.value):
            BaseTrainer._create_batch_generator(trainer)
        assert not created
    else:
        BaseTrainer._create_batch_generator(trainer)
        assert trainer.train_batch_generator is created[0]


@pytest.mark.parametrize("backend", ["fsdp2", "fsdpturbo"])
def test_sft_entry_releases_completed_optimizer_state_before_export(monkeypatch, backend):
    from llamafactory.v1.trainers import sft_trainer

    events = []
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter]["exp_avg"] = torch.ones(1)

    def fit():
        assert optimizer.state
        events.append("fit")

    def save():
        assert not optimizer.state
        assert optimizer.param_groups[0]["params"][0] is parameter
        events.append("save")

    trainer = SimpleNamespace(optimizer=optimizer, fit=fit, save_model=save)
    arguments = SimpleNamespace(dist_config=SimpleNamespace(name=backend))
    interface = SimpleNamespace(destroy=lambda: events.append("destroy"))
    monkeypatch.setattr(sft_trainer, "get_args", lambda _: (None, SimpleNamespace(train_dataset=[]), arguments, None))
    monkeypatch.setattr(sft_trainer, "DistributedInterface", lambda *_: interface)
    monkeypatch.setattr(sft_trainer, "DataEngine", lambda _: [])
    monkeypatch.setattr(sft_trainer, "ModelEngine", lambda *_, **__: SimpleNamespace(model=None, renderer=None))
    monkeypatch.setattr(sft_trainer, "SFTTrainer", lambda **_: trainer)
    sft_trainer.run_sft({})
    assert events == ["fit", "save", "destroy"]
