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

"""Regression tests for API LFI/SSRF guards (issue #10944)."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from fastapi import HTTPException

from llamafactory.api import common as api_common
from llamafactory.api.common import check_lfi_path, check_ssrf_url


def test_check_lfi_path_rejects_prefix_sibling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling directory sharing a string prefix must not pass the allowlist."""
    safe = tmp_path / "safe_media"
    sibling = tmp_path / "safe_media_backup"
    safe.mkdir()
    sibling.mkdir()
    target = sibling / "secret.txt"
    target.write_text("nope", encoding="utf-8")

    monkeypatch.setattr(api_common, "SAFE_MEDIA_PATH", str(safe))
    monkeypatch.setattr(api_common, "ALLOW_LOCAL_FILES", True)

    with pytest.raises(HTTPException) as exc_info:
        check_lfi_path(str(target))
    assert exc_info.value.status_code == 403


def test_check_lfi_path_allows_file_under_safe_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    safe = tmp_path / "safe_media"
    safe.mkdir()
    target = safe / "ok.txt"
    target.write_text("yes", encoding="utf-8")

    monkeypatch.setattr(api_common, "SAFE_MEDIA_PATH", str(safe))
    monkeypatch.setattr(api_common, "ALLOW_LOCAL_FILES", True)

    check_lfi_path(str(target))


def test_check_lfi_path_preserves_403_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTPException(403) must not be rewritten as 400 by the broad except."""
    safe = tmp_path / "safe_media"
    outside = tmp_path / "outside"
    safe.mkdir()
    outside.mkdir()
    target = outside / "x.txt"
    target.write_text("x", encoding="utf-8")

    monkeypatch.setattr(api_common, "SAFE_MEDIA_PATH", str(safe))
    monkeypatch.setattr(api_common, "ALLOW_LOCAL_FILES", True)

    with pytest.raises(HTTPException) as exc_info:
        check_lfi_path(str(target))
    assert exc_info.value.status_code == 403
    assert "restricted" in str(exc_info.value.detail).lower()


def test_check_ssrf_url_rejects_when_any_resolved_address_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dual-stack / multi-A results must fail closed if any address is non-global."""

    def fake_getaddrinfo(host, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),  # public example
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),  # loopback
        ]

    monkeypatch.setattr(api_common.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("http://example.invalid/resource")
    assert exc_info.value.status_code == 403
