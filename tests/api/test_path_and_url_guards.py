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


def test_check_lfi_path_cross_drive_valueerror_returns_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows cross-drive ``commonpath`` ValueError must map to 403, not 400."""
    safe = tmp_path / "safe_media"
    safe.mkdir()
    target = safe / "ok.txt"
    target.write_text("yes", encoding="utf-8")

    monkeypatch.setattr(api_common, "SAFE_MEDIA_PATH", str(safe))
    monkeypatch.setattr(api_common, "ALLOW_LOCAL_FILES", True)

    def boom(_paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(api_common.os.path, "commonpath", boom)

    with pytest.raises(HTTPException) as exc_info:
        check_lfi_path(str(target))
    assert exc_info.value.status_code == 403
    assert "restricted" in str(exc_info.value.detail).lower()


@pytest.mark.parametrize(
    "url",
    [
        "HTTP://example.invalid/a.png",
        "HTTPS://example.invalid/a.png",
        "Ftp://example.invalid/a.bin",
    ],
)
def test_is_web_uri_matches_schemes_case_insensitively(url: str) -> None:
    assert api_common.is_web_uri(url) is True


def test_is_web_uri_false_for_local_paths() -> None:
    assert api_common.is_web_uri("/tmp/safe_media/x.png") is False
    assert api_common.is_web_uri("safe_media/x.png") is False


def test_check_ssrf_url_accepts_uppercase_http_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uppercase HTTP/HTTPS must reach SSRF checks, not fall through as LFI paths."""

    def fake_getaddrinfo(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(api_common.socket, "getaddrinfo", fake_getaddrinfo)
    check_ssrf_url("HTTP://example.invalid/resource")


def test_check_ssrf_url_rejects_ftp_after_web_uri_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ftp:// is routed as a web URI and rejected by the HTTP/HTTPS-only SSRF guard."""
    assert api_common.is_web_uri("ftp://example.invalid/file") is True
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("ftp://example.invalid/file")
    assert exc_info.value.status_code == 400
    assert "http" in str(exc_info.value.detail).lower()


def test_fetch_remote_media_rejects_redirect_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """With allow_redirects=False, a 302 body must not be returned to media loaders."""

    class FakeResp:
        status_code = 302
        raw = object()

    def fake_get(url, stream=True, allow_redirects=False, timeout=None):
        assert allow_redirects is False
        return FakeResp()

    def fake_getaddrinfo(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(api_common.socket, "getaddrinfo", fake_getaddrinfo)

    import requests as requests_mod

    monkeypatch.setattr(requests_mod, "get", fake_get)

    with pytest.raises(HTTPException) as exc_info:
        api_common.fetch_remote_media("http://example.invalid/img.png")
    assert exc_info.value.status_code == 400
    assert "302" in str(exc_info.value.detail)


def test_fetch_remote_media_returns_raw_on_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    class FakeResp:
        status_code = 200
        raw = sentinel

    def fake_get(url, stream=True, allow_redirects=False, timeout=None):
        return FakeResp()

    def fake_getaddrinfo(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(api_common.socket, "getaddrinfo", fake_getaddrinfo)

    import requests as requests_mod

    monkeypatch.setattr(requests_mod, "get", fake_get)

    assert api_common.fetch_remote_media("https://example.invalid/img.png") is sentinel
