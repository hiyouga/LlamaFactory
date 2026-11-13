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

import os

import pytest

from llamafactory.hparams import get_train_args


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "finetuning_type": "lora",
    "output_dir": "dummy_dir",
    "report_to": "none",
}


def test_fp8_all_gather_enables_fp8():
    _, _, training_args, _, _ = get_train_args({**TRAIN_ARGS, "fp8_enable_fsdp_float8_all_gather": True})
    assert training_args.fp8 is True


def test_fp8_all_gather_rejects_quantization():
    with pytest.raises(ValueError, match="FP8 training is not compatible with quantization"):
        get_train_args({**TRAIN_ARGS, "fp8_enable_fsdp_float8_all_gather": True, "quantization_bit": 4})
