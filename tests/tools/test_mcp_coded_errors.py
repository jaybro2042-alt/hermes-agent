"""Coded MCP errors (Liminality D-ERRORS wire shape) are surfaced to the model.

Servers such as Liminality return every failure as a structured body::

    {"ok": false, "error": {"code": "LIM-ARGS-001", "message": "...", "hint": "...",
                            "retryable": false, "correlation_id": "lim-..."}}

Before this, ``_sanitize_error`` only regex-stripped credentials, so a coded body that
arrived through the ``isError`` transport path was shown to the model as a raw JSON blob
(or, worse, a bare "MCP tool returned an error"). Now the client parses
``error.code`` / ``message`` / ``hint`` and renders one readable line, while carrying
the stable code as its own key so the model can branch on it.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tools.mcp_tool import _coded_error_fields, _format_coded_error, _sanitize_error


_LIMINAL_BODY = {
    "ok": False,
    "error": {
        "code": "LIM-ARGS-001",
        "message": "unknown argument 'state' for tool handoff",
        "hint": "use session_state",
        "retryable": False,
        "correlation_id": "lim-8b6def97a52c",
        "suggestions": {"state": "session_state"},
    },
}


class TestCodedErrorFields:
    def test_parses_liminal_shape(self):
        fields = _coded_error_fields(json.dumps(_LIMINAL_BODY))
        assert fields["code"] == "LIM-ARGS-001"
        assert fields["message"] == "unknown argument 'state' for tool handoff"
        assert fields["hint"] == "use session_state"
        assert fields["retryable"] is False
        assert fields["correlation_id"] == "lim-8b6def97a52c"
        assert fields["suggestions"] == {"state": "session_state"}

    def test_parses_flat_code_message_shape(self):
        fields = _coded_error_fields('{"code": "E_BAD_REQUEST", "message": "nope"}')
        assert fields == {"code": "E_BAD_REQUEST", "message": "nope"}

    @pytest.mark.parametrize("text", [
        "",
        "plain transport error",
        "unhandled errors in a TaskGroup (1 sub-exception)",
        "{not json",
        '["a", "list"]',
        '{"ok": false, "error": "legacy string error"}',       # legacy string form
        '{"error": {"message": "no code here"}}',              # code missing
        '{"error": {"code": "X"}}',                            # message missing
        '{"error": {"code": 500, "message": "numeric code"}}', # code not a string
    ])
    def test_non_coded_inputs_return_none(self, text):
        assert _coded_error_fields(text) is None


class TestFormatCodedError:
    def test_full_line(self):
        line = _format_coded_error(_coded_error_fields(json.dumps(_LIMINAL_BODY)))
        assert line.startswith("[LIM-ARGS-001] unknown argument 'state' for tool handoff")
        assert "hint: use session_state" in line
        assert "did you mean: state → session_state" in line
        assert line.endswith("(ref lim-8b6def97a52c)")

    def test_minimal_line(self):
        assert _format_coded_error({"code": "LIM-INTERNAL-001", "message": "boom"}) == \
            "[LIM-INTERNAL-001] boom"


class TestSanitizeErrorCoded:
    def test_coded_body_becomes_readable_line(self):
        out = _sanitize_error(json.dumps(_LIMINAL_BODY))
        assert out.startswith("[LIM-ARGS-001] ")
        assert "hint: use session_state" in out
        assert "{" not in out  # no JSON blob leaks to the model

    def test_coded_body_still_redacts_credentials(self):
        body = {"error": {"code": "LIM-UPSTREAM-001", "message": "upstream rejected",
                          "detail": "Authorization: Bearer abc.def.ghi failed"}}
        out = _sanitize_error(json.dumps(body))
        assert out.startswith("[LIM-UPSTREAM-001] upstream rejected")
        assert "abc.def.ghi" not in out
        assert "[REDACTED]" in out

    def test_plain_text_unchanged(self):
        assert _sanitize_error("normal error message") == "normal error message"

    def test_legacy_string_error_body_unchanged(self):
        text = '{"ok": false, "error": "handoff_not_found"}'
        assert _sanitize_error(text) == text


class TestHandlerSurfacesCodedErrors:
    """End to end through the tool handler's isError branch."""

    @pytest.fixture()
    def _handler(self):
        from unittest.mock import AsyncMock, MagicMock, patch as mock_patch

        from tools import mcp_tool

        fake_session = MagicMock()
        fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)

        def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            loop = asyncio.new_event_loop()
            try:
                async def _install_lock_and_run():
                    for srv in list(mcp_tool._servers.values()):
                        if getattr(srv, "_rpc_lock", None) is None:
                            srv._rpc_lock = asyncio.Lock()
                    return await coro

                return loop.run_until_complete(_install_lock_and_run())
            finally:
                loop.close()

        mcp_tool._reset_server_error("liminal-test")
        try:
            with mock_patch.dict(mcp_tool._servers, {"liminal-test": fake_server}), \
                 mock_patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
                fake_session.call_tool = AsyncMock()
                yield fake_session, mcp_tool._make_tool_handler("liminal-test", "handoff", 30.0)
        finally:
            mcp_tool._reset_server_error("liminal-test")

    def test_is_error_coded_body(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(_LIMINAL_BODY))],
            isError=True, structuredContent=None,
        ))
        data = json.loads(handler({"task": "x", "state": {}}))
        assert data["error"].startswith("[LIM-ARGS-001] unknown argument 'state'")
        assert "hint: use session_state" in data["error"]
        assert data["code"] == "LIM-ARGS-001"
        assert data["hint"] == "use session_state"
        assert data["retryable"] is False
        assert data["correlation_id"] == "lim-8b6def97a52c"

    def test_is_error_plain_text_is_unchanged(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="tool failed")],
            isError=True, structuredContent=None,
        ))
        data = json.loads(handler({}))
        assert data == {"error": "tool failed"}

    def test_is_error_false_coded_result_passes_through_as_result(self, _handler):
        """Liminality returns coded errors with isError=false (normal structured
        result); that path is untouched — the model reads the body inline."""
        from unittest.mock import AsyncMock

        session, handler = _handler
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(_LIMINAL_BODY))],
            isError=False, structuredContent=None,
        ))
        data = json.loads(handler({}))
        assert json.loads(data["result"]) == _LIMINAL_BODY
