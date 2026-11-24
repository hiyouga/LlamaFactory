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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from llamafactory.extras import env


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for the checkout fixture")


def _commit_in(directory: Path, message: str) -> str:
    directory.mkdir()
    git_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "init", "--quiet"], cwd=directory, env=git_env, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=LlamaFactory Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            message,
        ],
        cwd=directory,
        env=git_env,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def test_env_reports_loaded_source_checkout(tmp_path, monkeypatch, capsys):
    caller = tmp_path / "caller"
    caller_head = _commit_in(caller, "caller repository")
    checkout = tmp_path / "source"
    source_head = _commit_in(checkout, "LlamaFactory source")
    source = checkout / "src" / "llamafactory" / "extras" / "env.py"
    source.parent.mkdir(parents=True)
    source.touch()

    monkeypatch.chdir(caller)
    monkeypatch.setattr(env, "__file__", str(source))
    env.print_env()

    output = capsys.readouterr().out
    assert f"- Git commit: {source_head}" in output
    assert caller_head not in output


def test_env_ignores_callers_git_head_for_installed_package(tmp_path, monkeypatch, capsys):
    caller = tmp_path / "caller"
    caller_head = _commit_in(caller, "caller repository")
    installed = caller / ".venv" / "lib" / "site-packages" / "llamafactory" / "extras" / "env.py"
    installed.parent.mkdir(parents=True)
    installed.touch()

    monkeypatch.chdir(caller)
    monkeypatch.setattr(env, "__file__", str(installed))
    env.print_env()

    output = capsys.readouterr().out
    assert "Git commit" not in output
    assert caller_head not in output
