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

from llamafactory.train.tuner import export_model, run_exp


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_ADAPTER = os.getenv("TINY_LLAMA_ADAPTER", "llamafactory/tiny-random-Llama-3-lora")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "do_train": True,
    "finetuning_type": "lora",
    "dataset_dir": "REMOTE:" + DEMO_DATA,
    "template": "llama3",
    "cutoff_len": 1,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "adapter_name_or_path": TINY_LLAMA_ADAPTER,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
}

OS_NAME = os.getenv("OS_NAME", "")


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize(
    "stage,dataset",
    [
        ("pt", "c4_demo"),
        ("sft", "alpaca_en_demo"),
        ("dpo", "dpo_en_demo"),
        ("kto", "kto_en_demo"),
        pytest.param("rm", "dpo_en_demo", marks=pytest.mark.xfail(OS_NAME.startswith("windows"), reason="OS error.")),
    ],
)
def test_run_exp(stage: str, dataset: str):
    output_dir = os.path.join("output", f"train_{stage}")
    run_exp({"stage": stage, "dataset": dataset, "output_dir": output_dir, **TRAIN_ARGS})
    assert os.path.exists(output_dir)


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_export():
    export_dir = os.path.join("output", "llama3_export")
    export_model({"export_dir": export_dir, **INFER_ARGS})
    assert os.path.exists(export_dir)
