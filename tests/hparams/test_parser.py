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

import importlib.metadata
import os

import pytest

from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.hparams.parser import _check_extra_dependencies


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def _mock_trl_version(monkeypatch: "pytest.MonkeyPatch", trl_version: str):
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda pkg: trl_version if pkg == "trl" else version(pkg))


def test_ppo_trl_version(monkeypatch: "pytest.MonkeyPatch"):
    monkeypatch.setenv("DISABLE_VERSION_CHECK", "1")
    model_args = ModelArguments(model_name_or_path=TINY_LLAMA3)
    finetuning_args = FinetuningArguments(stage="ppo", reward_model=TINY_LLAMA3)
    _mock_trl_version(monkeypatch, "0.9.6")
    _check_extra_dependencies(model_args, finetuning_args)

    _mock_trl_version(monkeypatch, "0.24.0")
    with pytest.raises(ImportError, match="trl==0.24.0"):
        _check_extra_dependencies(model_args, finetuning_args)


def test_sft_trl_version(monkeypatch: "pytest.MonkeyPatch"):
    model_args = ModelArguments(model_name_or_path=TINY_LLAMA3)
    finetuning_args = FinetuningArguments(stage="sft")
    _mock_trl_version(monkeypatch, "0.24.0")
    _check_extra_dependencies(model_args, finetuning_args)
