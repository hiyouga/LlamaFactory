# Copyright 2026 the LlamaFactory team.
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

from copy import deepcopy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.adapter import init_adapter


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
@pytest.mark.parametrize(
    ("modules_to_save", "update_embeddings"),
    [
        (None, False),
        (["embed_tokens"], True),
        (["lm_head"], True),
        (["embed_tokens", "lm_head"], True),
        (["embed_tokens", "lm_head"], False),
    ],
)
def test_lora_merge_preserves_embeddings_on_reload(tmp_path, tie_word_embeddings, modules_to_save, update_embeddings):
    torch.manual_seed(42)
    base_model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tie_word_embeddings,
        )
    )
    model = get_peft_model(
        deepcopy(base_model),
        LoraConfig(r=2, target_modules=["q_proj"], modules_to_save=modules_to_save),
    )
    # Give independently trained copies distinct values without a slow training loop.
    with torch.no_grad():
        if update_embeddings and "embed_tokens" in modules_to_save:
            model.get_input_embeddings().modules_to_save["default"].weight[0, 0] += 0.25
        if update_embeddings and "lm_head" in modules_to_save:
            model.get_output_embeddings().modules_to_save["default"].weight[0, 1] += 0.5

    input_ids = torch.tensor([[0, 1, 2, 3]])
    model.eval()
    with torch.no_grad():
        expected_logits = model(input_ids).logits
    expected_input = model.get_input_embeddings().weight.detach().clone()
    expected_output = model.get_output_embeddings().weight.detach().clone()
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)

    merged = init_adapter(
        base_model.config,
        base_model,
        ModelArguments(model_name_or_path="unused", adapter_name_or_path=str(adapter_dir)),
        FinetuningArguments(),
        is_trainable=False,
    )
    export_dir = tmp_path / "export"
    merged.save_pretrained(export_dir)
    reloaded = Qwen2ForCausalLM.from_pretrained(export_dir).eval()

    torch.testing.assert_close(reloaded.get_input_embeddings().weight, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(reloaded.get_output_embeddings().weight, expected_output, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids).logits, expected_logits)
    expected_tied = tie_word_embeddings and not update_embeddings
    assert merged.config.tie_word_embeddings == expected_tied
    assert reloaded.config.tie_word_embeddings == expected_tied
    assert (reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight) == expected_tied
