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

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from llamafactory.v1.core.utils.checkpoint import _load_standard_training_states, _save_standard_training_states


def _model(kind, tied=True):
    torch.manual_seed(41)
    if kind == "gpt2":
        config = GPT2Config(
            vocab_size=19,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            tie_word_embeddings=tied,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config)
    config = Qwen3Config(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config)


def _optimizer(model, kind):
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.8)
    return torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)


def _processor():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "training": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _write_checkpoint(directory, model, optimizer, format):
    directory.mkdir()
    if format == "safe":
        _save_standard_training_states(str(directory), model, optimizer, _processor(), False)
    else:
        model.save_pretrained(directory / "model", safe_serialization=format != "bin", max_shard_size="1KB")
        (directory / "optimizer").mkdir()
        torch.save(optimizer.state_dict(), directory / "optimizer" / "state_dict.pt")


def _step(model, optimizer, offset):
    ids = (torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]) + offset) % model.config.vocab_size
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    optimizer.step()
    return loss.detach(), gradients


def _assert_equal(actual, expected, actual_optimizer, expected_optimizer):
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    for parameter, expected_parameter in zip(actual.parameters(), expected.parameters()):
        state = actual_optimizer.state[parameter]
        expected_state = expected_optimizer.state[expected_parameter]
        assert state.keys() == expected_state.keys()
        for key in state:
            torch.testing.assert_close(state[key], expected_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gpt2", "qwen3"])
@pytest.mark.parametrize("optimizer_kind", ["sgd", "adamw"])
@pytest.mark.parametrize("format", ["safe", "sharded_safe", "bin"])
def test_tied_checkpoint_restores_model_optimizer_and_next_updates(tmp_path, kind, optimizer_kind, format):
    model = _model(kind)
    optimizer = _optimizer(model, optimizer_kind)
    _step(model, optimizer, 0)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, model, optimizer, format)
    if format != "bin":
        saved = {}
        for path in (checkpoint / "model").glob("*.safetensors"):
            saved.update(load_file(path))
        assert "lm_head.weight" not in saved
    if format == "sharded_safe":
        assert len(list((checkpoint / "model").glob("*.safetensors"))) > 1

    restored = _model(kind)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.add_(0.5)
    restored_optimizer = _optimizer(restored, optimizer_kind)
    _load_standard_training_states(str(checkpoint), restored, restored_optimizer, torch.device("cpu"))
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    _assert_equal(restored, model, restored_optimizer, optimizer)

    for offset in (1, 2):
        expected_loss, expected_gradients = _step(model, optimizer, offset)
        actual_loss, actual_gradients = _step(restored, restored_optimizer, offset)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_equal(restored, model, restored_optimizer, optimizer)


@pytest.mark.parametrize("kind", ["gpt2", "qwen3"])
@pytest.mark.parametrize("missing", ["embedding_group", "independent_parameter", "untied_output", "buffer"])
def test_genuinely_missing_weights_still_raise(tmp_path, kind, missing):
    model = _model(kind, tied=missing != "untied_output")
    model.register_buffer("checkpoint_counter", torch.tensor(7))
    optimizer = _optimizer(model, "adamw")
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    embedding_names = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter is model.get_input_embeddings().weight
    ]
    if missing == "embedding_group":
        omitted = embedding_names
    elif missing == "independent_parameter":
        omitted = [next(name for name in state if name not in embedding_names and name != "checkpoint_counter")]
    elif missing == "untied_output":
        omitted = ["lm_head.weight"]
    else:
        omitted = ["checkpoint_counter"]
    for name in omitted:
        del state[name]
    save_file(state, str(model_dir / "model.safetensors"))

    with pytest.raises(RuntimeError, match="Unexpected missing keys"):
        _load_standard_training_states(str(checkpoint), model, optimizer, torch.device("cpu"))


@pytest.mark.parametrize("kind", ["gpt2", "qwen3"])
def test_a_present_output_alias_can_restore_the_shared_embedding(tmp_path, kind):
    model = _model(kind)
    optimizer = _optimizer(model, "sgd")
    _step(model, optimizer, 0)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, model, optimizer, "safe")
    path = checkpoint / "model" / "model.safetensors"
    state = load_file(path)
    embedding_name = next(
        name for name, parameter in model.named_parameters() if parameter is model.get_input_embeddings().weight
    )
    state["lm_head.weight"] = state.pop(embedding_name)
    save_file(state, str(path))

    restored = _model(kind)
    restored_optimizer = _optimizer(restored, "sgd")
    _load_standard_training_states(str(checkpoint), restored, restored_optimizer, torch.device("cpu"))
    _assert_equal(restored, model, restored_optimizer, optimizer)


@pytest.mark.parametrize("kind", ["gpt2", "qwen3"])
def test_equal_values_in_untied_parameters_do_not_count_as_an_alias(tmp_path, kind):
    model = _model(kind, tied=False)
    with torch.no_grad():
        model.get_output_embeddings().weight.copy_(model.get_input_embeddings().weight)
    assert model.get_output_embeddings().weight is not model.get_input_embeddings().weight
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    state = {name: value.clone() for name, value in model.state_dict().items() if name != "lm_head.weight"}
    save_file(state, str(checkpoint / "model" / "model.safetensors"))

    with pytest.raises(RuntimeError, match="lm_head.weight"):
        _load_standard_training_states(str(checkpoint), model, _optimizer(model, "sgd"), torch.device("cpu"))
