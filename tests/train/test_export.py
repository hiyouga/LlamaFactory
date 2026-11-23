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

from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llamafactory.train import tuner


@pytest.mark.parametrize("private", [None, False, True])
@pytest.mark.parametrize("with_processor", [False, True])
@pytest.mark.parametrize("hub_model_id", [None, "test-user/test-model"])
def test_export_hub_privacy(tmp_path, monkeypatch, private, with_processor, hub_model_id):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_layer=1, n_head=1, n_embd=8))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "<pad>": 2, "hello": 3}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    processor = Mock() if with_processor else None
    model_push = Mock()
    tokenizer_push = Mock()
    monkeypatch.setattr(model, "push_to_hub", model_push)
    monkeypatch.setattr(tokenizer, "push_to_hub", tokenizer_push)
    monkeypatch.setattr(tuner, "load_model", lambda *args: model)
    monkeypatch.setattr(tuner, "load_tokenizer", lambda *args: {"tokenizer": tokenizer, "processor": processor})

    export_dir = tmp_path / "export"
    args = {
        "model_name_or_path": "test-model",
        "template": "empty",
        "finetuning_type": "full",
        "infer_dtype": "float32",
        "export_dir": str(export_dir),
        "export_hub_model_id": hub_model_id,
    }
    if private is not None:
        args["export_hub_private"] = private

    tuner.export_model(args)

    reloaded = GPT2LMHeadModel.from_pretrained(export_dir)
    torch.testing.assert_close(reloaded.transformer.wte.weight, model.transformer.wte.weight)
    reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(export_dir)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert (export_dir / "Modelfile").is_file()

    uploads = [model_push, tokenizer_push]
    if with_processor:
        processor.save_pretrained.assert_called_once_with(str(export_dir))
        uploads.append(processor.push_to_hub)

    for upload in uploads:
        if hub_model_id is None:
            upload.assert_not_called()
        else:
            upload.assert_called_once()
            assert upload.call_args.args == (hub_model_id,)
            assert upload.call_args.kwargs.get("private") is (True if private else None)
