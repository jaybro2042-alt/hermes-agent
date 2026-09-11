"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key
from gateway.run import _clarify_delivery_channel_from_result


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def test_successful_text_fallback_ignores_unrelated_raw_response_fields():
    result = SendResult(
        success=True,
        message_id="text",
        raw_response={"message_id": "text"},
    )

    assert _clarify_delivery_channel_from_result(result, True) == "adapter"
    result.raw_response["clarify_delivery_channel"] = "numbered_text"
    assert _clarify_delivery_channel_from_result(result, True) == "numbered_text"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choices", "expected_channel"),
    [(["A", "B"], "numbered_text"), (None, "open_text")],
)
async def test_base_text_fallback_reports_the_channel_it_confirmed(
    choices,
    expected_channel,
):
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    cm.register("clarify-channel", "session-channel", "Pick", choices)

    result = await adapter.send_clarify_text_fallback(
        chat_id="12345",
        question="Pick",
        choices=choices,
        clarify_id="clarify-channel",
        session_key="session-channel",
    )

    assert result.success is True
    assert result.raw_response["clarify_delivery_channel"] == expected_channel
    assert _clarify_delivery_channel_from_result(result, bool(choices)) == expected_channel


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])
    assert cm.mark_delivery_confirmed("clarify-1", "native_buttons") is True

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}
