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

import socket

import pytest
from fastapi import HTTPException

from llamafactory.api import common


def test_check_lfi_path_allows_safe_child(tmp_path, monkeypatch):
    safe_dir = tmp_path / "safe"
    media_file = safe_dir / "image.png"
    safe_dir.mkdir()
    media_file.write_bytes(b"image")

    monkeypatch.setattr(common, "ALLOW_LOCAL_FILES", True)
    monkeypatch.setattr(common, "SAFE_MEDIA_PATH", str(safe_dir))

    common.check_lfi_path(str(media_file))


def test_check_lfi_path_rejects_prefix_sibling(tmp_path, monkeypatch):
    safe_dir = tmp_path / "safe"
    sibling_dir = tmp_path / "safe_evil"
    sibling_file = sibling_dir / "image.png"
    sibling_dir.mkdir()
    sibling_file.write_bytes(b"image")

    monkeypatch.setattr(common, "ALLOW_LOCAL_FILES", True)
    monkeypatch.setattr(common, "SAFE_MEDIA_PATH", str(safe_dir))

    with pytest.raises(HTTPException) as exc_info:
        common.check_lfi_path(str(sibling_file))

    assert exc_info.value.status_code == 403


def test_check_ssrf_url_checks_all_resolved_addresses(monkeypatch):
    def fake_getaddrinfo(hostname, port, type=0):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port or 80)),
        ]

    monkeypatch.setattr(common.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException) as exc_info:
        common.check_ssrf_url("https://example.com/image.png")

    assert exc_info.value.status_code == 403


def test_check_ssrf_url_allows_global_addresses(monkeypatch):
    def fake_getaddrinfo(hostname, port, type=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 80))]

    monkeypatch.setattr(common.socket, "getaddrinfo", fake_getaddrinfo)

    common.check_ssrf_url("https://example.com/image.png")
