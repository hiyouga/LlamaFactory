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

import pytest
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM, TrainingArguments
from transformers.utils import SAFE_WEIGHTS_NAME
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from llamafactory.hparams import FinetuningArguments
from llamafactory.model.patcher import patch_valuehead_model
from llamafactory.train.callbacks import fix_valuehead_checkpoint
from llamafactory.train.rm.trainer import PairwiseTrainer


def get_tiny_reward_model(model_type):
    if model_type == "gpt2":
        base_model = GPT2LMHeadModel(GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=32, n_positions=16))
    else:
        base_model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                tie_word_embeddings=True,
            )
        )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    patch_valuehead_model(model)
    return model


def get_reward_trainer(model, output_dir, safe_serialization=True):
    return PairwiseTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir), use_cpu=True, report_to="none", save_safetensors=safe_serialization
        ),
        finetuning_args=FinetuningArguments(finetuning_type="full"),
        processor=None,
        tokenizer=None,
    )


@pytest.mark.parametrize("model_type", ["gpt2", "qwen3"])
@pytest.mark.parametrize("safe_serialization", [True, False])
def test_save_reward_model_with_tied_weights(tmp_path, model_type, safe_serialization):
    model = get_tiny_reward_model(model_type)
    trainer = get_reward_trainer(model, tmp_path, safe_serialization)
    inputs = {"input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]), "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.compute_loss(model, inputs).backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_values = model(**inputs)[2]

    trainer.save_model()
    fix_valuehead_checkpoint(model, str(tmp_path), safe_serialization)
    restored_base = type(model.pretrained_model).from_pretrained(tmp_path)
    for name, tensor in model.pretrained_model.state_dict().items():
        torch.testing.assert_close(restored_base.state_dict()[name], tensor, rtol=0, atol=0)

    restored = AutoModelForCausalLMWithValueHead.from_pretrained(restored_base)
    if safe_serialization:
        value_head = load_file(tmp_path / V_HEAD_SAFE_WEIGHTS_NAME)
    else:
        value_head = torch.load(tmp_path / V_HEAD_WEIGHTS_NAME, weights_only=True)
    restored.v_head.load_state_dict({name.removeprefix("v_head."): tensor for name, tensor in value_head.items()})
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs)[2], expected_values)


def test_save_reward_model_preserves_shared_storage_views(tmp_path):
    model = get_tiny_reward_model("gpt2")
    trainer = get_reward_trainer(model, tmp_path)
    storage = torch.arange(8, dtype=torch.float32)
    state_dict = {"left": storage[:4], "right": storage[4:], "alias": storage[:4]}
    expected = {name: tensor.clone() for name, tensor in state_dict.items()}

    trainer._save(str(tmp_path), state_dict=state_dict)

    saved = load_file(tmp_path / SAFE_WEIGHTS_NAME)
    assert saved.keys() == expected.keys()
    assert state_dict.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)
        torch.testing.assert_close(state_dict[name], tensor, rtol=0, atol=0)
