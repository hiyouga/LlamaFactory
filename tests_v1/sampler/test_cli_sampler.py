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

from llamafactory.v1.config import ModelArguments, SampleArguments
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.samplers.cli_sampler import SyncSampler


@pytest.mark.runs_on(["cuda", "npu", "xpu"])
def test_sync_sampler():
    model_args = ModelArguments(model="Qwen/Qwen3-4B-Instruct-2507")
    sample_args = SampleArguments()
    model_engine = ModelEngine(model_args)
    sampler = SyncSampler(sample_args, model_args, model_engine.model, model_engine.renderer)
    messages = [{"role": "user", "content": [{"type": "text", "value": "Say 'This is a test.'"}]}]
    response = ""
    for new_text in sampler.generate(messages):
        response += new_text

    print(response)
    assert "This is a test." in response
