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

import ipaddress
import json
import os
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..extras.misc import is_env_enabled
from ..extras.packages import is_fastapi_available


if is_fastapi_available():
    from fastapi import HTTPException, status


if TYPE_CHECKING:
    from pydantic import BaseModel


SAFE_MEDIA_PATH = os.environ.get("SAFE_MEDIA_PATH", os.path.join(os.path.dirname(__file__), "safe_media"))
ALLOW_LOCAL_FILES = is_env_enabled("ALLOW_LOCAL_FILES", "1")


def dictify(data: "BaseModel") -> dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def check_lfi_path(path: str) -> None:
    """Checks if a given path is vulnerable to LFI. Raises HTTPException if unsafe."""
    if not ALLOW_LOCAL_FILES:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local file access is disabled.")

    try:
        os.makedirs(SAFE_MEDIA_PATH, exist_ok=True)
        real_path = os.path.realpath(path)
        safe_path = os.path.realpath(SAFE_MEDIA_PATH)
    except OSError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or inaccessible file path.")

    try:
        is_safe_path = os.path.commonpath([real_path, safe_path]) == safe_path
    except ValueError:
        is_safe_path = False

    if not is_safe_path:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="File access is restricted to the safe media directory."
        )


def check_ssrf_url(url: str) -> None:
    """Checks if a given URL is vulnerable to SSRF. Raises HTTPException if unsafe."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only HTTP/HTTPS URLs are allowed.")

        hostname = parsed_url.hostname
        if not hostname:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid URL hostname.")

        for ip_info in socket.getaddrinfo(hostname, parsed_url.port, type=socket.SOCK_STREAM):
            ip = ipaddress.ip_address(ip_info[4][0])
            if not ip.is_global:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access to private or reserved IP addresses is not allowed.",
                )

    except socket.gaierror:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not resolve hostname: {parsed_url.hostname}"
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL: {e}")
