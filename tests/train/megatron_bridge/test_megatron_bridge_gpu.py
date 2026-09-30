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

import pytest

from llamafactory.hparams.megatron_bridge_args import MegatronBridgeArguments
from llamafactory.train.megatron_bridge.config_builder import _apply_fusion_safety, _apply_model_parallelism
from llamafactory.train.megatron_bridge.workflow import _check_backend_available


@pytest.mark.runs_on(["cuda"])
def test_check_backend_available():
    _check_backend_available()


@pytest.mark.runs_on(["cuda"])
def test_auto_bridge_provider_creation(mb_model_path: str):
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(mb_model_path)
    provider = bridge.to_megatron_provider(load_weights=False)
    _apply_model_parallelism(provider, MegatronBridgeArguments(tensor_model_parallel_size=1))
    _apply_fusion_safety(provider)
    assert provider.tensor_model_parallel_size == 1


@pytest.mark.runs_on(["cuda"])
def test_build_sft_config_tp2_on_gpu(mb_training_args_factory, mb_output_dir):
    from llamafactory.train.megatron_bridge.config_builder import build_sft_config

    model_args, data_args, training_args, finetuning_args, mb_args, num_train_samples = mb_training_args_factory(
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        use_distributed_optimizer=True,
    )
    cfg = build_sft_config(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        mb_args=mb_args,
        dataset_root=str(mb_output_dir / "dataset"),
        pretrained_checkpoint=str(mb_output_dir / "pretrained"),
        num_train_samples=num_train_samples,
    )
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.sequence_parallel is True
    assert cfg.ddp.use_distributed_optimizer is True
