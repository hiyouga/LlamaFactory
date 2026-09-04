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
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils import check_min_version


try:
    check_min_version("4.34.0")
except Exception:
    raise ValueError("Please upgrade `transformers` to 4.34.0")


CONFIG_NAME = "config.json"


def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool) -> str:
    qwen_state_dict: dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".safetensors"):
            with safe_open(os.path.join(input_dir, filepath), framework="pt", device="cpu") as f:
                for key in f.keys():
                    qwen_state_dict[key] = f.get_tensor(key)

    llama_state_dict: dict[str, torch.Tensor] = OrderedDict()
    torch_dtype = None
    for key, value in tqdm(qwen_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "wte" in key:
            llama_state_dict["model.embed_tokens.weight"] = value
        elif "ln_f" in key:
            llama_state_dict["model.norm.weight"] = value
        else:
            key = key.replace("transformer.h", "model.layers")
            if "attn.c_attn" in key:
                proj_size = value.size(0) // 3
                llama_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
                llama_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value[
                    proj_size : 2 * proj_size, ...
                ]
                llama_state_dict[key.replace("attn.c_attn", "self_attn.v_proj")] = value[2 * proj_size :, ...]
            elif "attn.c_proj" in key:
                llama_state_dict[key.replace("attn.c_proj", "self_attn.o_proj")] = value
                llama_state_dict[key.replace("attn.c_proj.weight", "self_attn.o_proj.bias")] = torch.zeros_like(
                    value[:, 0]
                ).squeeze()
            elif "ln_1" in key:
                llama_state_dict[key.replace("ln_1", "input_layernorm")] = value
            elif "ln_2" in key:
                llama_state_dict[key.replace("ln_2", "post_attention_layernorm")] = value
            elif "mlp.w1" in key:
                llama_state_dict[key.replace("mlp.w1", "mlp.up_proj")] = value
            elif "mlp.w2" in key:
                llama_state_dict[key.replace("mlp.w2", "mlp.gate_proj")] = value
            elif "mlp.c_proj" in key:
                llama_state_dict[key.replace("mlp.c_proj", "mlp.down_proj")] = value
            elif "lm_head" in key:
                llama_state_dict[key] = value
            else:
                raise KeyError(f"Unable to process key {key}")

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

    return str(torch_dtype).replace("torch.", "")


def save_config(input_dir: str, output_dir: str, torch_dtype: str):
    with open(os.path.join(input_dir, CONFIG_NAME), encoding="utf-8") as f:
        qwen_config_dict: dict[str, Any] = json.load(f)

    llama2_config_dict: dict[str, Any] = OrderedDict()
    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict["hidden_act"] = "silu"
    llama2_config_dict["hidden_size"] = qwen_config_dict["hidden_size"]
    llama2_config_dict["initializer_range"] = qwen_config_dict["initializer_range"]
    llama2_config_dict["intermediate_size"] = qwen_config_dict["intermediate_size"] // 2
    llama2_config_dict["max_position_embeddings"] = qwen_config_dict["max_position_embeddings"]
    llama2_config_dict["model_type"] = "llama"
    llama2_config_dict["num_attention_heads"] = qwen_config_dict["num_attention_heads"]
    llama2_config_dict["num_hidden_layers"] = qwen_config_dict["num_hidden_layers"]
    llama2_config_dict["num_key_value_heads"] = qwen_config_dict["hidden_size"] // qwen_config_dict["kv_channels"]
    llama2_config_dict["pretraining_tp"] = 1
    llama2_config_dict["rms_norm_eps"] = qwen_config_dict["layer_norm_epsilon"]
    llama2_config_dict["rope_scaling"] = None
    llama2_config_dict["tie_word_embeddings"] = qwen_config_dict["tie_word_embeddings"]
    llama2_config_dict["torch_dtype"] = torch_dtype
    llama2_config_dict["transformers_version"] = "4.34.0"
    llama2_config_dict["use_cache"] = True
    llama2_config_dict["vocab_size"] = qwen_config_dict["vocab_size"]
    llama2_config_dict["attention_bias"] = True

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)

    print(f"Model config saved in {os.path.join(output_dir, CONFIG_NAME)}")


def llamafy_qwen(
    input_dir: str,
    output_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = False,
):
    r"""Convert the Qwen models in the same format as LLaMA2.

    Usage: python llamafy_qwen.py --input_dir input --output_dir output
    Converted model: https://huggingface.co/hiyouga/Qwen-14B-Chat-LLaMAfied
    """
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise ValueError("Output dir already exists") from e

    torch_dtype = save_weight(input_dir, output_dir, shard_size, save_safetensors)
    save_config(input_dir, output_dir, torch_dtype)


if __name__ == "__main__":
    fire.Fire(llamafy_qwen)
