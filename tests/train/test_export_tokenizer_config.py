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
from types import SimpleNamespace

from llamafactory.train.tuner import _fix_tokenizer_config_for_compat


def _write_config(path, config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")


def _read_config(path):
    return json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))


def _model_args(model_name_or_path):
    return SimpleNamespace(
        model_name_or_path=str(model_name_or_path), cache_dir=None, model_revision="main", hf_hub_token=None
    )


def test_restore_tokenizer_class_and_special_tokens(tmp_path):
    base_dir, export_dir = tmp_path / "base", tmp_path / "export"
    _write_config(base_dir, {"tokenizer_class": "Qwen2Tokenizer"})
    _write_config(
        export_dir,
        {
            "tokenizer_class": "TokenizersBackend",
            "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
            "eos_token": "<|im_end|>",
        },
    )
    _fix_tokenizer_config_for_compat(str(export_dir), _model_args(base_dir))
    config = _read_config(export_dir)
    assert config["tokenizer_class"] == "Qwen2Tokenizer"
    assert config["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"]
    assert "extra_special_tokens" not in config
    assert config["eos_token"] == "<|im_end|>"


def test_fallback_tokenizer_class(tmp_path):
    base_dir, export_dir = tmp_path / "base", tmp_path / "export"
    _write_config(base_dir, {"tokenizer_class": "TokenizersBackend"})
    _write_config(export_dir, {"tokenizer_class": "TokenizersBackend"})
    _fix_tokenizer_config_for_compat(str(export_dir), _model_args(base_dir))
    assert _read_config(export_dir)["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_keep_compatible_config(tmp_path):
    base_dir, export_dir = tmp_path / "base", tmp_path / "export"
    _write_config(base_dir, {"tokenizer_class": "LlamaTokenizer"})
    original = {"tokenizer_class": "LlamaTokenizer", "extra_special_tokens": {"image_token": "<image>"}}
    _write_config(export_dir, original)
    _fix_tokenizer_config_for_compat(str(export_dir), _model_args(base_dir))
    assert _read_config(export_dir) == original
