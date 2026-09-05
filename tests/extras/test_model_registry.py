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

from llamafactory.extras.constants import SUPPORTED_MODELS, DownloadSource


def test_minimax_models_are_registered() -> None:
    expected_models = {
        "MiniMax-M2.7-Thinking": "MiniMaxAI/MiniMax-M2.7",
    }

    for model_name, model_path in expected_models.items():
        assert SUPPORTED_MODELS[model_name][DownloadSource.DEFAULT] == model_path
        assert SUPPORTED_MODELS[model_name][DownloadSource.MODELSCOPE] == model_path
