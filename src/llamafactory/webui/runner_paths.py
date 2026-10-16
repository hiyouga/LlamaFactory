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

from pathlib import Path


def resolve_cli_executable(python_executable: str | Path, platform: str) -> str:
    r"""Resolve the LlamaFactory CLI executable next to the active Python installation."""
    python_dir = Path(python_executable).parent
    if platform == "nt":
        environment_cli = python_dir / "llamafactory-cli.exe"
        if environment_cli.exists():
            return str(environment_cli)

        scripts_cli = python_dir / "Scripts" / "llamafactory-cli.exe"
        if scripts_cli.exists():
            return str(scripts_cli)

    else:
        environment_cli = python_dir / "llamafactory-cli"
        if environment_cli.exists():
            return str(environment_cli)

    return "llamafactory-cli"
