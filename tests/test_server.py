"""Server startup tests, focused on the localhost/port-conflict failure.

A page loaded from http://localhost:PORT breaks when something else holds the
IPv6 wildcard on that port: the browser resolves localhost to ::1 first and the
foreign server answers /api/* with 404. These tests pin down that app.py
detects a half-occupied port.
"""

from __future__ import annotations

import socket

import pytest

from app import _listening_on, _port_is_free, _resolve_port


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Listener:
    """Hold a port the way the conflicting server did: a real listener."""

    def __init__(self, port, family=socket.AF_INET, address="127.0.0.1"):
        self.sock = socket.socket(family, socket.SOCK_STREAM)
        if family == socket.AF_INET6:
            self.sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        self.sock.bind((address, port))
        self.sock.listen(1)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_a_totally_free_port_is_reported_free():
    assert _port_is_free(_free_port()) is True


def test_a_bound_socket_is_reported_busy():
    port = _free_port()
    with _Listener(port):
        assert _port_is_free(port) is False


@pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6 support")
def test_ipv6_listener_is_reported_busy():
    """localhost may resolve to ::1, so an IPv6 listener must count as busy.

    This is the case that produced the 404s: a bind probe on Windows succeeds
    against a held wildcard, so only asking the OS for real listeners works.
    """
    port = _free_port()
    try:
        with _Listener(port, socket.AF_INET6, "::"):
            assert "::" in _listening_on(port)
            assert _port_is_free(port) is False
    except OSError as exc:
        pytest.skip(f"cannot hold an IPv6 listener here: {exc}")


def test_listening_on_reports_nothing_for_an_unused_port():
    assert _listening_on(_free_port()) == []


def test_resolve_port_prefers_the_requested_port_when_free():
    port = _free_port()
    assert _resolve_port(port) == port


def test_resolve_port_falls_back_past_a_blocked_port():
    blocked = _free_port()
    with _Listener(blocked):
        chosen = _resolve_port(blocked)
    assert chosen > blocked
    assert _port_is_free(chosen) is True


def test_resolve_port_reports_failure_when_the_range_is_exhausted():
    """No free port in the window must be signalled, not silently returned."""
    blocked = _free_port()
    with _Listener(blocked):
        assert _resolve_port(blocked, span=0) == -1
