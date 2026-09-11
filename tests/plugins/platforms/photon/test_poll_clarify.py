"""Native-poll clarify tests for PhotonAdapter.

iMessage has a native poll bubble (spectrum-ts `poll()` builder). A
multiple-choice ``clarify`` renders as that poll; the user taps a choice and
the vote streams back inbound as a ``poll_option`` event. These tests cover
both directions without spawning the Node sidecar or binding ports:

  * outbound — ``send_clarify`` with choices POSTs ``/send-poll`` and flips the
    clarify into text-capture mode; with no choices it stays plain text;
  * inbound — a ``poll_option`` selection is dispatched as a plain-text message
    carrying the chosen option (so the gateway clarify-intercept resolves it),
    a deselection is dropped, and an empty-title vote is dropped.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _poll_option_event(
    *,
    title: str,
    selected: bool = True,
    msg_id: str = "spc-msg-vote",
    poll_guid: Optional[str] = "spc-msg-poll",
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {
            "type": "poll_option",
            "title": title,
            "selected": selected,
            "pollTitle": "Pick one",
            "pollGuid": poll_guid,
        },
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _poll_echo_event(
    *, title: str, options: list[str], msg_id: str = "spc-msg-poll"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+155****4567", "type": "dm"},
        "sender": {"id": "+155****4567"},
        "content": {"type": "poll", "title": title, "options": options},
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


def _register_clarify(clarify_id: str = "clar-1"):
    import tools.clarify_gateway as cg

    return cg.register(
        clarify_id,
        "sess-1",
        "Pick one",
        ["A", "B", "C"],
        platform="photon",
        chat_id="+155****4567",
    )


# ---------------------------------------------------------------------------
# Inbound: a poll vote becomes the clarify answer.


@pytest.mark.asyncio
async def test_poll_vote_dispatched_as_choice_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll selection is forwarded as a plain-text message carrying the
    chosen option, so the gateway clarify text-intercept can resolve it."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    import tools.clarify_gateway as cg

    _register_clarify()
    assert cg.mark_transport_accepted("clar-1", "spc-msg-poll") is True
    assert cg.mark_delivery_confirmed("clar-1", "native_poll") is True

    await adapter._dispatch_inbound(
        _poll_option_event(title="Yes — native tappable buttons")
    )

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text == "Yes — native tappable buttons"
    assert ev.message_type == MessageType.TEXT
    assert ev.source.chat_id == "+155****4567"


@pytest.mark.asyncio
async def test_stale_same_label_vote_for_old_poll_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    import tools.clarify_gateway as cg

    _register_clarify("clar-new")
    assert cg.mark_transport_accepted("clar-new", "new-poll-guid") is True
    assert cg.mark_delivery_confirmed("clar-new", "native_poll") is True

    await adapter._dispatch_inbound(
        _poll_option_event(title="A", poll_guid="old-poll-guid")
    )

    assert captured == []


# ---------------------------------------------------------------------------
# Outbound: send_clarify renders a native poll for choices.


def _stub_sidecar_poll(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch, *, ok: bool = True
) -> List[Tuple[str, str, list]]:
    calls: List[Tuple[str, str, list]] = []

    async def fake_send_poll(space_id: str, title: str, options: list):
        calls.append((space_id, title, list(options)))
        return SendResult(
            success=ok,
            message_id="spc-msg-poll" if ok else None,
            error=None if ok else "boom",
        )

    monkeypatch.setattr(adapter, "_sidecar_send_poll", fake_send_poll)
    return calls


def _stub_sidecar_text(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[Tuple[str, str]]:
    sends: List[Tuple[str, str]] = []

    async def fake_send(space_id: str, text: str):
        sends.append((space_id, text))
        return SendResult(success=True, message_id="spc-msg-text")

    monkeypatch.setattr(adapter, "_sidecar_send", fake_send)
    return sends


@pytest.mark.asyncio
async def test_send_clarify_with_choices_sends_native_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)
    import tools.clarify_gateway as cg
    entry = _register_clarify()
    confirmation = asyncio.create_task(
        adapter._dispatch_inbound(
            _poll_echo_event(title="Pick one", options=["A", "B", "C"])
        )
    )

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="Pick one",
        choices=["A", "B", "C"],
        clarify_id="clar-1",
        session_key="sess-1",
    )

    assert result.success
    assert len(poll_calls) == 1
    space_id, title, options = poll_calls[0]
    assert space_id == "+155****4567"
    assert title == "Pick one"
    assert options == ["A", "B", "C"]
    await confirmation
    assert entry.state == cg.STATE_EVENT_CONFIRMED
    assert entry.delivery_channel == "native_poll"
    assert entry.awaiting_text is False


@pytest.mark.asyncio
async def test_empty_title_confirmation_falls_back_to_numbered_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    text_sends = _stub_sidecar_text(adapter, monkeypatch)
    import tools.clarify_gateway as cg
    entry = _register_clarify("clar-empty")
    confirmation = asyncio.create_task(
        adapter._dispatch_inbound(
            _poll_echo_event(title="", options=["A", "B", "C"])
        )
    )

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="Pick one",
        choices=["A", "B", "C"],
        clarify_id="clar-empty",
        session_key="sess-1",
    )

    await confirmation
    assert result.success is True
    assert len(text_sends) == 1
    assert "1. A" in text_sends[0][1]
    assert entry.fallback_attempted is True
    assert entry.delivery_channel == "numbered_text"
    assert cg.claim_text_fallback("clar-empty") is False


@pytest.mark.asyncio
async def test_missing_confirmation_event_falls_back_to_numbered_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._poll_confirmation_timeout = 0.05
    _stub_sidecar_poll(adapter, monkeypatch)
    text_sends = _stub_sidecar_text(adapter, monkeypatch)
    import tools.clarify_gateway as cg
    entry = _register_clarify("clar-timeout")

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="Pick one",
        choices=["A", "B", "C"],
        clarify_id="clar-timeout",
        session_key="sess-1",
    )

    assert result.success is True
    assert len(text_sends) == 1
    assert "ignore it and reply to this numbered version" in text_sends[0][1]
    assert entry.fallback_attempted is True
    assert entry.delivery_channel == "numbered_text"
    assert cg.claim_text_fallback("clar-timeout") is False


def test_late_poll_confirmation_does_not_rearm_native_poll(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _make_adapter(monkeypatch)
    import tools.clarify_gateway as cg

    entry = _register_clarify("clar-late")
    assert cg.mark_transport_accepted("clar-late", "spc-msg-poll") is True
    assert cg.claim_text_fallback("clar-late") is True
    assert cg.mark_delivery_confirmed("clar-late", "numbered_text") is True

    adapter._record_poll_confirmation(
        _poll_echo_event(title="Pick one", options=["A", "B", "C"])
    )

    assert "spc-msg-poll" not in adapter._poll_confirmation_echoes
    assert entry.delivery_channel == "numbered_text"
    assert "numbered fallback" in caplog.text
