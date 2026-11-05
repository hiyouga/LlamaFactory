# Copyright 2026 Mingyang Wu (LlamaFactory contributor).
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

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llamafactory.train.tuner import export_model


def _prepare_export(tmp_path, generation_eos, use_adapter):
    base_dir = tmp_path / "base"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<|endoftext|>": 3, "<|im_end|>": 4, "hello": 5},
                unk_token="<unk>",
            )
        ),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(base_dir)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_positions=16,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
    )
    model.generation_config.eos_token_id = generation_eos
    model.save_pretrained(base_dir)
    args = {
        "model_name_or_path": str(base_dir),
        "export_dir": str(tmp_path / "export"),
        "infer_dtype": "float32",
    }
    if use_adapter:
        adapter_dir = tmp_path / "adapter"
        get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["c_attn"])).save_pretrained(
            adapter_dir
        )
        args["adapter_name_or_path"] = str(adapter_dir)

    return args


@pytest.mark.parametrize("use_adapter", [False, True])
@pytest.mark.parametrize(
    ("generation_eos", "expected_eos"),
    [(None, [4]), (3, [4, 3]), ([3, 5], [4, 3, 5]), (4, 4), ([4, 5], [4, 5])],
)
def test_export_replaced_eos(tmp_path, generation_eos, expected_eos, use_adapter):
    args = _prepare_export(tmp_path, generation_eos, use_adapter)
    export_model({"template": "qwen", **args})

    output_dir = tmp_path / "export"
    assert json.loads((output_dir / "config.json").read_text())["eos_token_id"] == 4
    assert json.loads((output_dir / "generation_config.json").read_text())["eos_token_id"] == expected_eos
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    assert tokenizer.eos_token_id == model.config.eos_token_id == 4

    input_ids = torch.tensor([[2, 5]])
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=3,
        prefix_allowed_tokens_fn=lambda batch_id, input_ids: [tokenizer.eos_token_id],
    )
    assert output[0, input_ids.shape[1] :].tolist() == [tokenizer.eos_token_id]


@pytest.mark.parametrize("use_adapter", [False, True])
@pytest.mark.parametrize("generation_eos", [3, [3, 5]])
def test_export_preserves_eos_without_replacement(tmp_path, generation_eos, use_adapter):
    args = _prepare_export(tmp_path, generation_eos, use_adapter)
    export_model({"template": "empty", **args})

    output_dir = tmp_path / "export"
    assert json.loads((output_dir / "config.json").read_text())["eos_token_id"] == 3
    assert json.loads((output_dir / "generation_config.json").read_text())["eos_token_id"] == generation_eos
