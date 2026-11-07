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

from llamafactory.train.test_utils import (
    compare_model,
    compare_model_pissa,
    load_infer_model,
    load_reference_model,
    load_train_model,
)


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_PISSA = os.getenv("TINY_LLAMA_ADAPTER", "llamafactory/tiny-random-Llama-3-pissa")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "pissa_init": True,
    "pissa_iter": -1,
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA_PISSA,
    "adapter_name_or_path": TINY_LLAMA_PISSA,
    "adapter_folder": "pissa_init",
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
}


def test_pissa_train():
    model = load_train_model(**TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_PISSA, TINY_LLAMA_PISSA, use_pissa=True, is_trainable=True)
    # Non-LoRA params (embeddings, norms, lm_head) must match exactly; skip LoRA-layer weights.
    compare_model(model, ref_model, skip_keys=["lora_A", "lora_B", "base_layer"])
    # SVD factors (lora_A/lora_B) are gauge-dependent and differ across platforms, so
    # compare the invariant B@A update instead of the raw factors.
    compare_model_pissa(model, ref_model)


@pytest.mark.xfail(reason="Known connection error.")
def test_pissa_inference():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_PISSA, TINY_LLAMA_PISSA, use_pissa=True, is_trainable=False)
    ref_model = ref_model.merge_and_unload()
    compare_model(model, ref_model)
