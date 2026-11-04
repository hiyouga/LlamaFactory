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

r"""Regression guard: the WebUI must not enable remote code execution by default.

CVE-class: unauthenticated RCE via ``trust_remote_code=True`` on an
attacker-controlled model path supplied through the (unauthenticated)
Gradio WebUI. See scan cmvgxjs4n00kzqn01fnzaqwf5.
"""

import re
from pathlib import Path

import pytest


WEBUI_DIR = Path(__file__).resolve().parents[2] / "src" / "llamafactory" / "webui"

# Matches a literal ``trust_remote_code=True`` argument, ignoring the
# environment-gated ``is_env_enabled(...)`` form.
_LITERAL_TRUE = re.compile(r"trust_remote_code\s*=\s*True")


@pytest.mark.parametrize("py_file", sorted(WEBUI_DIR.rglob("*.py")), ids=lambda p: p.name)
def test_webui_does_not_hardcode_trust_remote_code(py_file: Path) -> None:
    r"""No WebUI module may hardcode ``trust_remote_code=True``."""
    source = py_file.read_text(encoding="utf-8")
    matches = [i + 1 for i, line in enumerate(source.splitlines()) if _LITERAL_TRUE.search(line)]
    assert not matches, (
        f"{py_file.relative_to(WEBUI_DIR)} hardcodes trust_remote_code=True on line(s) {matches}; "
        "remote code execution must be an explicit server-side opt-in."
    )


def test_webui_remote_code_gate_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The opt-in flag must evaluate to False when unset."""
    from llamafactory.extras.misc import is_env_enabled

    monkeypatch.delenv("LLAMAFACTORY_WEBUI_TRUST_REMOTE_CODE", raising=False)
    assert is_env_enabled("LLAMAFACTORY_WEBUI_TRUST_REMOTE_CODE") is False
