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
import torch

from llamafactory.hparams import get_infer_args
from llamafactory.train.test_utils import (
    check_lora_model,
    compare_model,
    load_infer_model,
    load_reference_model,
    load_train_model,
)


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_ADAPTER = os.getenv("TINY_LLAMA_ADAPTER", "llamafactory/tiny-random-Llama-3-lora")

TINY_LLAMA_VALUEHEAD = os.getenv("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "adapter_name_or_path": TINY_LLAMA_ADAPTER,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
}


def test_lora_train_qv_modules():
    model = load_train_model(lora_target="q_proj,v_proj", **TRAIN_ARGS)
    linear_modules, _ = check_lora_model(model)
    assert linear_modules == {"q_proj", "v_proj"}


def test_lora_train_all_modules():
    model = load_train_model(lora_target="all", **TRAIN_ARGS)
    linear_modules, _ = check_lora_model(model)
    assert linear_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}


def test_lora_train_extra_modules():
    model = load_train_model(additional_target="embed_tokens,lm_head", **TRAIN_ARGS)
    _, extra_modules = check_lora_model(model)
    assert extra_modules == {"embed_tokens", "lm_head"}


def test_lora_train_old_adapters():
    model = load_train_model(adapter_name_or_path=TINY_LLAMA_ADAPTER, create_new_adapter=False, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA3, TINY_LLAMA_ADAPTER, use_lora=True, is_trainable=True)
    compare_model(model, ref_model)


def test_lora_train_new_adapters():
    model = load_train_model(adapter_name_or_path=TINY_LLAMA_ADAPTER, create_new_adapter=True, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA3, TINY_LLAMA_ADAPTER, use_lora=True, is_trainable=True)
    compare_model(
        model, ref_model, diff_keys=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    )


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_lora_train_valuehead():
    model = load_train_model(add_valuehead=True, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, is_trainable=True, add_valuehead=True)
    state_dict = model.state_dict()
    ref_state_dict = ref_model.state_dict()
    assert torch.allclose(state_dict["v_head.summary.weight"], ref_state_dict["v_head.summary.weight"])
    assert torch.allclose(state_dict["v_head.summary.bias"], ref_state_dict["v_head.summary.bias"])


def test_lora_inference():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA3, TINY_LLAMA_ADAPTER, use_lora=True).merge_and_unload()
    compare_model(model, ref_model)


@pytest.mark.runs_on(["cpu"])
def test_lora_inference_cpu_no_keyerror():
    """Regression test for #10957: on CPU-only hosts, `llamafactory-cli chat` crashed with
    KeyError: 'base_model.model.model.model.model.embed_tokens' when loading a LoRA adapter,
    because device_map="auto" let accelerate disk-offload parts of the base model and
    peft<=0.18.1 mis-resolves the offloaded module prefix in _update_offload.

    The chat inference path must pin the model to CPU so the adapter loads the same way as a
    plain from_pretrained call. Qwen2.5 is the model reported in the issue; the repo has no tiny
    Qwen fixture, so this uses the tiny Llama-3 fixtures, which share the same "model."-prefixed
    checkpoint keys and exercise the identical load path.
    """
    model_args, _, _, _ = get_infer_args({**INFER_ARGS})
    assert model_args.device_map == {"": torch.device("cpu")}

    model_args_no_adapter, _, _, _ = get_infer_args(
        {key: value for key, value in INFER_ARGS.items() if key != "adapter_name_or_path"}
    )
    assert model_args_no_adapter.device_map == "auto"

    model = load_infer_model(**INFER_ARGS)  # must not raise KeyError
    assert any("lora_A" in name for name, _ in model.named_parameters())
