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

import json
import os

import pytest
import torch
from safetensors import safe_open

from llamafactory.train.tuner import export_model


TINY_LLAMA4 = os.getenv("TINY_LLAMA4", "llamafactory/tiny-random-Llama-4")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA4,
    "template": "llama4",
    "export_size": 1,
}


@pytest.mark.parametrize("infer_dtype", ["float16", "bfloat16"])
def test_export_dtype_matches_sub_configs(tmp_path, infer_dtype: str):
    r"""The exported config should declare the export dtype for the sub-configs too.

    `save_pretrained` only refreshes the top-level dtype, so a composite (multimodal) config would
    otherwise keep the dtype of the source checkpoint and contradict the exported weights.
    """
    export_dir = tmp_path / "export"
    export_model({**INFER_ARGS, "infer_dtype": infer_dtype, "export_dir": str(export_dir)})

    with open(export_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    assert set(config.keys()) & {"text_config", "vision_config"}, "expected a composite config"
    assert config["dtype"] == infer_dtype
    for sub_config_key in ("text_config", "vision_config"):
        assert config[sub_config_key]["dtype"] == infer_dtype

    expected_dtype = getattr(torch, infer_dtype)
    with safe_open(export_dir / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.is_floating_point():
                assert tensor.dtype == expected_dtype, f"{key} was exported as {tensor.dtype}"
