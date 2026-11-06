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
from urllib.parse import urljoin, urlparse

from ..extras.misc import is_env_enabled
from ..extras.packages import is_fastapi_available, is_requests_available


if is_fastapi_available():
    from fastapi import HTTPException, status


if is_requests_available():
    import requests
    import urllib3
    from requests.adapters import HTTPAdapter


if TYPE_CHECKING:
    from pydantic import BaseModel
    from requests import Response


SAFE_MEDIA_PATH = os.environ.get("SAFE_MEDIA_PATH", os.path.join(os.path.dirname(__file__), "safe_media"))
ALLOW_LOCAL_FILES = is_env_enabled("ALLOW_LOCAL_FILES", "1")

# Maximum number of HTTP redirects to follow when fetching a remote media URL. Every hop's
# target is re-validated by check_ssrf_url before being followed, so this only bounds how many
# times we are willing to re-validate before giving up.
MAX_SAFE_REDIRECTS = 5


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

        if not real_path.startswith(safe_path):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="File access is restricted to the safe media directory."
            )
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or inaccessible file path.")


def check_ssrf_url(url: str) -> str:
    """Checks if a given URL is vulnerable to SSRF. Raises HTTPException if unsafe.

    Returns:
        The IP address that the URL's hostname resolved to. Callers that go on to fetch the URL
        MUST connect to this exact IP address (e.g. via `_PinnedIPAdapter`) instead of letting
        the HTTP client resolve the hostname again, otherwise a second DNS lookup could return a
        different, private address (DNS rebinding) and bypass this check entirely.
    """
    try:
        parsed_url = urlparse(url)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL: {e}")

    if parsed_url.scheme not in ["http", "https"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only HTTP/HTTPS URLs are allowed.")

    hostname = parsed_url.hostname
    if not hostname:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid URL hostname.")

    try:  # .port raises for an out-of-range port, e.g. http://example.com:99999/
        port = parsed_url.port
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid URL port: {e}")

    try:
        ip_info = socket.getaddrinfo(hostname, port)
    except socket.gaierror:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not resolve hostname: {hostname}")

    if not ip_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not resolve hostname: {hostname}")

    # Reject the hostname if ANY resolved address is private/reserved, not just the first one, since
    # a malicious DNS server can return multiple records and the HTTP client is free to pick any.
    resolved_ip = None
    for family_info in ip_info:
        ip_address_str = family_info[4][0]
        ip = ipaddress.ip_address(ip_address_str)
        if not ip.is_global:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to private or reserved IP addresses is not allowed.",
            )

        if resolved_ip is None:
            resolved_ip = ip_address_str

    return resolved_ip


class _PinnedIPConnectionMixin:
    """Connects to a pre-validated IP address instead of re-resolving the hostname.

    urllib3 opens the socket against `_dns_host` but keeps using `.host` for the Host header, SNI
    and certificate validation, so swapping only the former for the duration of the connect pins
    the connection to the address `check_ssrf_url` already validated while leaving the request
    otherwise untouched. This closes the DNS-rebinding TOCTOU window between the check and the
    connection: a second lookup cannot point the socket somewhere private, because there is no
    second lookup.

    The pin lives on the connection object, so unlike a `socket.getaddrinfo` monkeypatch it is
    invisible to concurrent fetches and to every other thread in the process, and needs no lock.
    """

    def __init__(self, *args: Any, pinned_ip: str | None = None, **kwargs: Any) -> None:
        self.pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> "socket.socket":
        if not self.pinned_ip:
            return super()._new_conn()

        dns_host, self._dns_host = self._dns_host, self.pinned_ip
        try:
            return super()._new_conn()
        finally:
            self._dns_host = dns_host


class _PinnedIPHTTPConnection(_PinnedIPConnectionMixin, urllib3.connection.HTTPConnection):
    pass


class _PinnedIPHTTPSConnection(_PinnedIPConnectionMixin, urllib3.connection.HTTPSConnection):
    pass


class _PinnedIPHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    ConnectionCls = _PinnedIPHTTPConnection


class _PinnedIPHTTPSConnectionPool(urllib3.connectionpool.HTTPSConnectionPool):
    ConnectionCls = _PinnedIPHTTPSConnection


class _PinnedIPPoolManager(urllib3.PoolManager):
    """A pool manager whose connections all target `pinned_ip`."""

    def __init__(self, pinned_ip: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pinned_ip = pinned_ip
        self.pool_classes_by_scheme = {
            "http": _PinnedIPHTTPConnectionPool,
            "https": _PinnedIPHTTPSConnectionPool,
        }

    def _new_pool(self, scheme: str, host: str, port: int, request_context=None):
        pool = super()._new_pool(scheme, host, port, request_context)
        # injected after construction rather than passed through connection_pool_kw, whose keys
        # must all be fields of urllib3's PoolKey
        pool.conn_kw["pinned_ip"] = self._pinned_ip
        return pool


class _PinnedIPAdapter(HTTPAdapter):
    """A requests adapter that connects to `pinned_ip` for every host it serves.

    A configured HTTP proxy is out of scope: requests routes proxied requests through a separate
    ProxyManager, and the proxy resolves the hostname itself, so the pin (like any client-side
    SSRF check) cannot constrain where the connection ends up.
    """

    def __init__(self, pinned_ip: str, **kwargs: Any) -> None:
        self._pinned_ip = pinned_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **pool_kwargs: Any) -> None:
        self.poolmanager = _PinnedIPPoolManager(
            self._pinned_ip, num_pools=connections, maxsize=maxsize, block=block, **pool_kwargs
        )


def fetch_safe_url(url: str, **kwargs) -> "Response":
    """Safely fetches a remote URL, guarding against SSRF via HTTP redirects and DNS rebinding.

    The hostname of `url`, and of every redirect hop encountered while following it, is validated
    with `check_ssrf_url` and the connection is pinned to the exact IP address that was just
    validated (see `_PinnedIPAdapter`). Redirects are therefore never auto-followed by the
    underlying HTTP client: each `Location` is re-validated from scratch, up to `MAX_SAFE_REDIRECTS`
    hops, before it is followed.
    """
    kwargs.setdefault("stream", True)
    kwargs.setdefault("timeout", 10)
    kwargs["allow_redirects"] = False

    current_url = url
    for _ in range(MAX_SAFE_REDIRECTS + 1):
        ip = check_ssrf_url(current_url)

        # mirrors requests' own top-level API: the session is closed once the response is built,
        # which leaves an already-checked-out streaming connection usable.
        with requests.Session() as session:
            adapter = _PinnedIPAdapter(ip)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.get(current_url, **kwargs)

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Redirect response is missing a Location header."
                )

            current_url = urljoin(current_url, location)
            continue

        return response

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Too many redirects.")
