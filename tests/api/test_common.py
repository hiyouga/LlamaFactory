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

import http.server
import ipaddress
import socket
import threading

import pytest
from fastapi import HTTPException

from llamafactory.api.common import MAX_SAFE_REDIRECTS, check_ssrf_url, fetch_safe_url


# A genuinely private address, used as the target an SSRF attempt is trying to reach. Nothing ever
# connects to it: every test that uses it asserts the fetch is refused before a connection is made.
PRIVATE_TARGET = "http://10.255.255.1:9/"


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    r"""Base test HTTP handler that records how many requests it served."""

    response_body = b"OK"
    hit_count = 0

    def do_GET(self):  # noqa: N802
        type(self).hit_count += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, *args):
        pass  # keep test output clean


@pytest.fixture
def start_server():
    r"""Start throwaway HTTP servers on 127.0.0.1, shut down even if the test fails.

    Everything binds 127.0.0.1: it is the only loopback address configured on macOS' lo0 by
    default, so a second one such as 127.0.0.2 would fail to bind on the macos-latest jobs.
    """
    servers = []

    def _start(handler_cls: type[http.server.BaseHTTPRequestHandler], port: int = 0):
        server = http.server.HTTPServer(("127.0.0.1", port), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    try:
        yield _start
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


@pytest.fixture
def pretend_loopback_is_global(monkeypatch):
    r"""Make `ipaddress.ip_address("127.0.0.1").is_global` return True.

    All of 127.0.0.0/8 is loopback and therefore never actually global, so real SSRF payloads
    can't be built against it directly. This fixture lets a local HTTP server stand in for what
    would, in a real attack, be a public IP address the attacker controls -- the first hop that
    `check_ssrf_url` is expected to allow.
    """
    original_is_global = ipaddress.IPv4Address.is_global

    def patched_is_global(self):
        if str(self) == "127.0.0.1":
            return True
        return original_is_global.fget(self)

    monkeypatch.setattr(ipaddress.IPv4Address, "is_global", property(patched_is_global))


def _redirect_handler(location: str) -> type[http.server.BaseHTTPRequestHandler]:
    class RedirectHandler(_CountingHandler):
        def do_GET(self):  # noqa: N802
            type(self).hit_count += 1
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

    return RedirectHandler


def test_check_ssrf_url_rejects_private_ip():
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("http://127.0.0.1/")
    assert exc_info.value.status_code == 403


def test_check_ssrf_url_rejects_non_http_scheme():
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("file:///etc/passwd")
    assert exc_info.value.status_code == 400


def test_check_ssrf_url_rejects_out_of_range_port():
    r"""An unparseable port is a bad request, not an unhandled ValueError (i.e. a 500)."""
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("http://example.com:99999/")
    assert exc_info.value.status_code == 400


def test_check_ssrf_url_returns_resolved_ip(pretend_loopback_is_global):
    ip = check_ssrf_url("http://127.0.0.1:1234/")
    assert ip == "127.0.0.1"


def test_fetch_safe_url_blocks_redirect_to_private_ip(start_server, pretend_loopback_is_global):
    r"""fetch_safe_url() must re-validate (and refuse to follow) a redirect into a private IP.

    This is the bug in #10646: validating only the first hop is not enough, because the HTTP
    client will happily follow a redirect straight into a private address afterwards.
    """
    handler_cls = _redirect_handler(PRIVATE_TARGET)
    handler_cls.hit_count = 0
    public = start_server(handler_cls)

    with pytest.raises(HTTPException) as exc_info:
        fetch_safe_url(f"http://127.0.0.1:{public.server_address[1]}/")

    assert exc_info.value.status_code == 403
    assert handler_cls.hit_count == 1  # stopped at the first hop, never followed the Location


def test_fetch_safe_url_follows_redirect_to_another_public_ip(start_server, pretend_loopback_is_global):
    r"""A redirect to a URL that also passes the SSRF check must still be followed."""

    class TargetHandler(_CountingHandler):
        response_body = b"FINAL_DESTINATION"
        hit_count = 0

    target = start_server(TargetHandler)
    redirector = start_server(_redirect_handler(f"http://127.0.0.1:{target.server_address[1]}/final"))

    response = fetch_safe_url(f"http://127.0.0.1:{redirector.server_address[1]}/")

    assert response.content == b"FINAL_DESTINATION"
    assert TargetHandler.hit_count == 1


def test_fetch_safe_url_refuses_a_redirect_chain_that_never_ends(start_server, pretend_loopback_is_global):
    r"""A server that redirects to itself forever must be cut off, not followed indefinitely."""
    handler_cls = _redirect_handler("/loop")  # relative: urljoin keeps pointing back at this server
    handler_cls.hit_count = 0
    server = start_server(handler_cls)

    with pytest.raises(HTTPException) as exc_info:
        fetch_safe_url(f"http://127.0.0.1:{server.server_address[1]}/")

    assert exc_info.value.status_code == 400
    assert "redirect" in exc_info.value.detail.lower()
    assert handler_cls.hit_count == MAX_SAFE_REDIRECTS + 1


def test_fetch_safe_url_pins_connection_against_dns_rebinding(monkeypatch, start_server, pretend_loopback_is_global):
    r"""fetch_safe_url() must connect to what it validated, immune to a second/different DNS answer.

    A hostname is resolved twice by the old pattern: once for the SSRF check, and again when the
    HTTP client connects. A malicious DNS server can answer differently the second time -- the
    "rebind" -- pointing the real connection somewhere the check never saw.

    The rebound answer here differs by port rather than by address. `socket.create_connection`
    connects to the whole sockaddr that `getaddrinfo` returns, port included, so this exercises
    exactly the same "the client re-resolved and got a different answer" path, while keeping every
    server on 127.0.0.1 so the test runs on macOS too.
    """

    class SafeHandler(_CountingHandler):
        response_body = b"SAFE_DATA"
        hit_count = 0

    class PrivateHandler(_CountingHandler):
        response_body = b"SECRET_INTERNAL_DATA"
        hit_count = 0

    safe_port = start_server(SafeHandler).server_address[1]
    private_port = start_server(PrivateHandler).server_address[1]

    real_getaddrinfo = socket.getaddrinfo
    calls = {"n": 0}

    def rebinding_getaddrinfo(host, port, *args, **kwargs):
        if host == "rebind.example.test":
            calls["n"] += 1
            # 1st resolution (the SSRF check) answers with the checked target; every subsequent
            # resolution -- i.e. what the HTTP client would do when it connects -- rebinds.
            port = safe_port if calls["n"] == 1 else private_port
            return real_getaddrinfo("127.0.0.1", port, *args, **kwargs)

        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)

    response = fetch_safe_url(f"http://rebind.example.test:{safe_port}/")

    assert response.content == b"SAFE_DATA"
    assert PrivateHandler.hit_count == 0  # the connection must never reach the rebound target
