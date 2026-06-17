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

import ast
from pathlib import Path


def test_huggingface_engine_input_kwargs_defaults_are_not_mutable():
    module_path = Path(__file__).parents[2] / "src" / "llamafactory" / "chat" / "hf_engine.py"
    module = ast.parse(module_path.read_text(encoding="utf-8"))

    expected_methods = {"_process_args", "_chat", "_stream_chat", "_get_scores"}
    methods = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name in expected_methods
    }

    assert set(methods) == expected_methods

    for method in methods.values():
        arguments = method.args.args
        defaults = method.args.defaults
        default_by_arg = dict(zip((arg.arg for arg in arguments[-len(defaults) :]), defaults))

        default = default_by_arg["input_kwargs"]
        assert isinstance(default, ast.Constant)
        assert default.value is None
