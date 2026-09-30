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

"""Fixtures for Megatron Bridge GPU unit tests."""

import os
from pathlib import Path
from unittest import mock

import pytest

from llamafactory.extras.packages import is_megatron_bridge_available


pytestmark = pytest.mark.skipif(
    not is_megatron_bridge_available(),
    reason="megatron-bridge is not installed",
)


@pytest.fixture(scope="session")
def mb_model_path() -> str:
    r"""Return the local Hugging Face model path used by Megatron Bridge tests."""
    model_path = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")
    return model_path


@pytest.fixture
def mb_output_dir(tmp_path: Path) -> Path:
    r"""Create a writable output directory for Megatron Bridge tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    return output_dir


@pytest.fixture
def mb_training_args_factory(mb_output_dir: Path, mb_model_path: str):
    r"""Build LLaMA-Factory dataclass arguments for Megatron Bridge tests."""
    patchers: list[mock._patch] = []

    def _factory(
        *,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = False,
        num_train_samples: int = 100,
        cutoff_len: int = 512,
        output_dir: Path | None = None,
    ):
        from llamafactory.hparams.data_args import DataArguments
        from llamafactory.hparams.finetuning_args import FinetuningArguments
        from llamafactory.hparams.megatron_bridge_args import MegatronBridgeArguments
        from llamafactory.hparams.model_args import ModelArguments
        from llamafactory.hparams.training_args import TrainingArguments

        resolved_output_dir = output_dir or mb_output_dir
        model_args = ModelArguments(model_name_or_path=mb_model_path)
        data_args = DataArguments(
            dataset="alpaca_en_demo",
            template="llama3",
            cutoff_len=cutoff_len,
            preprocessing_num_workers=1,
        )
        training_args = TrainingArguments(
            output_dir=str(resolved_output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=5e-6,
            logging_steps=1,
            save_steps=1000000,
            report_to="none",
        )
        world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        patcher = mock.patch.object(
            type(training_args),
            "world_size",
            new_callable=mock.PropertyMock,
            return_value=world_size,
        )
        patchers.append(patcher)
        patcher.start()
        finetuning_args = FinetuningArguments(stage="sft", finetuning_type="full")
        mb_args = MegatronBridgeArguments(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            export_hf_on_finish=False,
        )
        return model_args, data_args, training_args, finetuning_args, mb_args, num_train_samples

    yield _factory

    for patcher in patchers:
        patcher.stop()
