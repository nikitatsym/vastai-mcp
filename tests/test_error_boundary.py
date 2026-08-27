from __future__ import annotations

import httpx
import pytest

from vastai_mcp import server, tools
from vastai_mcp.client import APIError


def _registered(name: str):
    return server.mcp._tool_manager._tools[name].fn


def _raise(exc: Exception):
    raise exc


def test_registered_root_returns_contextual_api_error(monkeypatch):
    class BrokenClient:
        def get(self, _path: str):
            raise APIError(503, "GET", "/api/v0/users/current/?api_key=secret", {"detail": "unavailable"})

    monkeypatch.setattr(tools, "_get_client", lambda: BrokenClient())

    result = _registered("vastai_version")()

    assert result["error"].startswith("GET /api/v0/users/current/ -> 503")
    assert "secret" not in result["error"]


def test_registered_tool_redacts_transport_query_values(monkeypatch):
    request = httpx.Request("GET", "https://vast.example/api/v0/instances/1?api_key=secret")
    monkeypatch.setattr(
        server,
        "_dispatch",
        lambda *_args: _raise(httpx.ConnectError("connection refused", request=request)),
    )

    result = _registered("vastai_read")("ShowInstance", {})

    assert "Vast.ai transport failure: GET /api/v0/instances/1: ConnectError" in result["error"]
    assert "secret" not in result["error"]
    assert "api_key=" not in result["error"]


def test_registered_tool_returns_missing_parameter_error():
    result = _registered("vastai_read")("ShowInstance", {})

    assert "Invalid params for ShowInstance" in result["error"]
    assert "id" in result["error"]


def test_registration_keeps_tools_sync():
    """MCP classifies tools with `iscoroutinefunction`; every op here is sync,
    so the wrapper must not make the SDK await them on the event loop."""
    assert all(not t.is_async for t in server.mcp._tool_manager._tools.values())


def test_registered_root_preserves_success_shape(monkeypatch):
    class OkClient:
        def get(self, _path: str):
            return {"id": 1}

    monkeypatch.setattr(tools, "_get_client", lambda: OkClient())

    result = _registered("vastai_version")()

    assert result["service"] == {"status": "ok"}
    assert "mcp" in result


def test_registered_tool_propagates_programming_error(monkeypatch):
    monkeypatch.setattr(
        server, "_dispatch", lambda *_args: _raise(AttributeError("programming error"))
    )

    with pytest.raises(AttributeError):
        _registered("vastai_read")("ShowInstance", {})


def test_error_text_redacts_secret_fields():
    """Container values are redacted whole: a partial match would leave the
    tail of a list or nested dict in the reported error."""
    text = server._redact_error_text(
        {"password": ["too short", "p@ssw0rd"], "api_secret": "zzz", "detail": "keep me"}
    )

    assert "p@ssw0rd" not in text
    assert "zzz" not in text
    assert "keep me" in text
