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

from llamafactory.webui.runner_paths import resolve_cli_executable


def test_resolve_cli_executable_prefers_active_environment(tmp_path: Path) -> None:
    python_executable = tmp_path / "python.exe"
    cli_executable = tmp_path / "llamafactory-cli.exe"
    cli_executable.touch()

    assert resolve_cli_executable(python_executable, platform="nt") == str(cli_executable)


def test_resolve_cli_executable_finds_windows_scripts_directory(tmp_path: Path) -> None:
    python_executable = tmp_path / "python.exe"
    cli_executable = tmp_path / "Scripts" / "llamafactory-cli.exe"
    cli_executable.parent.mkdir()
    cli_executable.touch()

    assert resolve_cli_executable(python_executable, platform="nt") == str(cli_executable)


def test_resolve_cli_executable_prefers_active_unix_environment(tmp_path: Path) -> None:
    python_executable = tmp_path / "bin" / "python"
    cli_executable = python_executable.parent / "llamafactory-cli"
    cli_executable.parent.mkdir()
    cli_executable.touch()

    assert resolve_cli_executable(python_executable, platform="posix") == str(cli_executable)


def test_resolve_cli_executable_falls_back_to_path(tmp_path: Path) -> None:
    python_executable = tmp_path / "python.exe"

    assert resolve_cli_executable(python_executable, platform="nt") == "llamafactory-cli"
