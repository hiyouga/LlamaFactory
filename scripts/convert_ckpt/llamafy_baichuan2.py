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
from collections import OrderedDict
from typing import Any

import fire
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CONFIG_NAME = "config.json"


def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool):
    baichuan2_state_dict: dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".bin"):
            shard_weight = torch.load(os.path.join(input_dir, filepath), map_location="cpu", weights_only=True)
            baichuan2_state_dict.update(shard_weight)

    llama_state_dict: dict[str, torch.Tensor] = OrderedDict()
    for key, value in tqdm(baichuan2_state_dict.items(), desc="Convert format"):
        if "W_pack" in key:
            proj_size = value.size(0) // 3
            llama_state_dict[key.replace("W_pack", "q_proj")] = value[:proj_size, :]
            llama_state_dict[key.replace("W_pack", "k_proj")] = value[proj_size : 2 * proj_size, :]
            llama_state_dict[key.replace("W_pack", "v_proj")] = value[2 * proj_size :, :]
        elif "lm_head" in key:
            llama_state_dict[key] = torch.nn.functional.normalize(value)
        else:
            llama_state_dict[key] = value

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        llama_state_dict, filename_pattern=filename_pattern, max_shard_size=shard_size
    )
    for shard_file, tensors in tqdm(state_dict_split.filename_to_tensors.items(), desc="Save weights"):
        shard = {tensor: llama_state_dict[tensor].contiguous() for tensor in tensors}
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))

    if not state_dict_split.is_sharded:
        print(f"Model weights saved in {os.path.join(output_dir, weights_name)}.")
    else:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        print(f"Model weights saved in {output_dir}.")


def save_config(input_dir: str, output_dir: str):
    with open(os.path.join(input_dir, CONFIG_NAME), encoding="utf-8") as f:
        llama2_config_dict: dict[str, Any] = json.load(f)

    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict.pop("auto_map", None)
    llama2_config_dict.pop("tokenizer_class", None)
    llama2_config_dict["model_type"] = "llama"

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)

    print(f"Model config saved in {os.path.join(output_dir, CONFIG_NAME)}")


def llamafy_baichuan2(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = True,
):
    r"""Convert the Baichuan2-7B model in the same format as LLaMA2-7B.

    Usage: python llamafy_baichuan2.py --input_dir input --output_dir output
    Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied
    """
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise ValueError("Output dir already exists") from e

    save_weight(input_dir, output_dir, shard_size, save_safetensors)
    save_config(input_dir, output_dir)


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)
