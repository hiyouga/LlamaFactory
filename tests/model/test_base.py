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
from types import SimpleNamespace

import pytest
import torch

from llamafactory.model.loader import _patch_missing_hf_device_map
from llamafactory.train.test_utils import compare_model, load_infer_model, load_reference_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_VALUEHEAD = os.getenv("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
    "infer_dtype": "float16",
}


def test_base():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA3)
    compare_model(model, ref_model)


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_valuehead():
    model = load_infer_model(add_valuehead=True, **INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, add_valuehead=True)
    compare_model(model, ref_model)


class DummyModel(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")


def test_patch_missing_hf_device_map():
    model = DummyModel()
    model_args = SimpleNamespace(export_dir="output/export", export_device="auto")

    _patch_missing_hf_device_map(model, model_args)

    assert model.hf_device_map == {"": "cpu"}
