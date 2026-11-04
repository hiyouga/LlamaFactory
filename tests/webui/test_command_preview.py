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
import os
import shutil
import subprocess
from typing import Any

import pytest

from llamafactory.webui.common import gen_cmd


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="requires a POSIX shell")
@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("model_name_or_path", "models/my model", ["models/my model"]),
        ("label_names", ["loss label", "score"], ["loss label", "score"]),
        (
            "deepspeed",
            {"zero_optimization": {"stage": 2}},
            [json.dumps({"zero_optimization": {"stage": 2}}, ensure_ascii=False)],
        ),
        (
            "output_dir",
            "saves/model's weights; $HOME $(printf expanded)",
            ["saves/model's weights; $HOME $(printf expanded)"],
        ),
    ],
    ids=["scalar", "list", "json", "shell-characters"],
)
def test_gen_cmd_preserves_posix_arguments(name: str, value: Any, expected: list[str]) -> None:
    """Copied previews preserve the argument values supplied by the Web UI."""
    command = gen_cmd({name: value}).removeprefix("```bash\n").removesuffix("\n```")
    # Capture arguments with a shell function instead of launching training.
    capture = "llamafactory-cli() { printf '%s\\0' \"$@\"; }\n"
    result = subprocess.run(["bash", "-c", capture + command], check=True, capture_output=True, text=True)

    assert result.stdout.rstrip("\0").split("\0") == ["train", f"--{name}", *expected]
