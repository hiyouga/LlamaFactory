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

from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.train.trainer_utils import restore_tied_weights_state_dict


def get_tiny_qwen3_model(tie_word_embeddings: bool):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=tie_word_embeddings,
        )
    )


def test_restore_tied_weights_state_dict(tmp_path):
    model = get_tiny_qwen3_model(tie_word_embeddings=True)
    state_dict = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    assert state_dict["model.embed_tokens.weight"].data_ptr() != state_dict["lm_head.weight"].data_ptr()

    restore_tied_weights_state_dict(model, state_dict)

    assert state_dict["model.embed_tokens.weight"].data_ptr() == state_dict["lm_head.weight"].data_ptr()
    model.save_pretrained(tmp_path, state_dict=state_dict)
    with safe_open(tmp_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())

    assert "model.embed_tokens.weight" in checkpoint_keys
    assert "lm_head.weight" not in checkpoint_keys


def test_restore_tied_weights_state_dict_ignores_untied_weights():
    model = get_tiny_qwen3_model(tie_word_embeddings=False)
    state_dict = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    restore_tied_weights_state_dict(model, state_dict)

    assert state_dict["model.embed_tokens.weight"].data_ptr() != state_dict["lm_head.weight"].data_ptr()
