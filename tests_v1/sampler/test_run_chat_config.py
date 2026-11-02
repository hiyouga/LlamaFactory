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

from unittest.mock import MagicMock, patch

from llamafactory.v1.config import ModelArguments
from llamafactory.v1.samplers.cli_sampler import run_chat


def test_run_chat_sets_meta_init_config_for_non_hf_backend():
    """run_chat() must request the meta-device init through init_config.

    ModelEngine only reads init_config, so storing the plugin in another
    attribute silences the request and the model gets fully loaded before the
    vLLM backend takes over.
    """
    captured = {}

    class FakeModelEngine:
        def __init__(self, model_args):
            captured["args"] = model_args
            self.model = MagicMock()
            self.renderer = MagicMock()

    with (
        patch("llamafactory.v1.samplers.cli_sampler.ModelEngine", FakeModelEngine),
        patch("llamafactory.v1.samplers.cli_sampler.DataEngine"),
        patch("llamafactory.v1.samplers.cli_sampler.SyncSampler"),
    ):
        run_chat(["--model", "test-model", "--sample_backend", "vllm", "--train_dataset", "dummy"])

    model_args = captured["args"]
    assert isinstance(model_args, ModelArguments)
    assert model_args.init_config is not None
    assert model_args.init_config.name == "init_on_meta"
