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

import pytest
import torch

from llamafactory.train import trainer_utils


class _OptimizerFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="moss_vl")
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.numeric_adapter = torch.nn.ModuleDict({"0": torch.nn.Linear(4, 4)})
        self.model.visual = torch.nn.Module()
        self.model.visual.blocks = torch.nn.ModuleDict({"0": torch.nn.Module()})
        self.model.visual.blocks["0"].model = torch.nn.Module()
        self.model.visual.blocks["0"].model.language_model = torch.nn.Module()
        self.model.visual.blocks["0"].model.language_model.numeric_adapter = torch.nn.ModuleDict(
            {"0": torch.nn.Linear(4, 4)}
        )


class _CaptureOptimizer:
    def __init__(self, param_groups, **kwargs) -> None:
        self.param_groups = param_groups


@pytest.mark.parametrize(
    ("create_optimizer", "optimizer_name", "target_name", "algorithm_args"),
    (
        (
            trainer_utils._create_galore_optimizer,
            "GaLoreAdamW",
            "galore_target",
            {
                "galore_rank": 8,
                "galore_update_interval": 200,
                "galore_scale": 2.0,
                "galore_proj_type": "std",
                "galore_layerwise": False,
            },
        ),
        (
            trainer_utils._create_apollo_optimizer,
            "APOLLOAdamW",
            "apollo_target",
            {
                "apollo_rank": 8,
                "apollo_update_interval": 200,
                "apollo_scale": 2.0,
                "apollo_proj": "svd",
                "apollo_proj_type": "std",
                "apollo_scale_type": "channel",
                "apollo_scale_front": False,
                "apollo_layerwise": False,
            },
        ),
    ),
)
def test_optimizer_target_all_keeps_numeric_paths_exact(
    monkeypatch: pytest.MonkeyPatch,
    create_optimizer,
    optimizer_name: str,
    target_name: str,
    algorithm_args: dict,
):
    model = _OptimizerFixture()
    training_args = SimpleNamespace(
        optim="adamw_torch",
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
    )
    finetuning_args = SimpleNamespace(freeze_vision_tower=True, **{target_name: ["all"]}, **algorithm_args)
    monkeypatch.setattr(trainer_utils, optimizer_name, _CaptureOptimizer, raising=False)
    monkeypatch.setattr(
        trainer_utils.Trainer, "get_optimizer_cls_and_kwargs", staticmethod(lambda training_args: (None, {}))
    )

    optimizer = create_optimizer(model, training_args, finetuning_args)

    parameter_names = {id(param): name for name, param in model.named_parameters()}
    selected_names = {parameter_names[id(param)] for param in optimizer.param_groups[-1]["params"]}
    assert selected_names == {"model.language_model.numeric_adapter.0.weight"}
