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

import inspect

from llamafactory.chat.hf_engine import HuggingfaceEngine


def test_hf_engine_input_kwargs_default_is_not_mutable():
    methods = [
        HuggingfaceEngine._process_args,
        HuggingfaceEngine._chat,
        HuggingfaceEngine._stream_chat,
        HuggingfaceEngine._get_scores,
    ]

    for method in methods:
        assert inspect.signature(method).parameters["input_kwargs"].default is None
