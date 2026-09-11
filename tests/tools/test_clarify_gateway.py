"""Tests for the gateway-side clarify primitive (tools/clarify_gateway.py).

The clarify tool needs to ask the user a question and block the agent
thread until they respond.  These tests cover the module-level state
machine: register, wait, resolve via button, resolve via text-fallback,
"Other"-button text-capture flip, timeout, session boundary cleanup.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _clear_clarify_state():
    """Reset module-level state between tests."""
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _confirm(cm, clarify_id: str, channel: str = "native_poll"):
    assert cm.mark_delivery_confirmed(clarify_id, channel) is True


class TestClarifyPrimitive:
    """Core register/wait/resolve mechanics."""

    def setup_method(self):
        _clear_clarify_state()

    def test_button_choice_resolves_wait(self):
        """resolve_gateway_clarify unblocks wait_for_response with the chosen string."""
        from tools import clarify_gateway as cm

        cm.register("id1", "sk1", "Pick one", ["A", "B", "C"])
        _confirm(cm, "id1")

        def resolver():
            time.sleep(0.05)
            cm.resolve_gateway_clarify("id1", "B")

        threading.Thread(target=resolver).start()
        result = cm.wait_for_response("id1", timeout=10.0)
        assert result == "B"

    def test_fast_native_click_before_confirmation_remains_successful(self):
        """A button callback is itself proof the prompt was visible."""
        from tools import clarify_gateway as cm

        cm.register("id1-fast", "sk1-fast", "Pick one", ["A", "B"])

        assert cm.resolve_gateway_clarify("id1-fast", "B") is True
        assert cm.mark_delivery_confirmed("id1-fast", "adapter") is True
        assert cm.wait_for_response("id1-fast", timeout=1.0) == "B"

    def test_open_ended_awaits_text_only_after_confirmed_delivery(self):
        """An undelivered prompt cannot consume an unrelated inbound message."""
        from tools import clarify_gateway as cm

        entry = cm.register("id2", "sk2", "Free form?", None)
        assert entry.awaiting_text is False
        assert cm.get_pending_for_session("sk2") is None

        _confirm(cm, "id2", "open_text")
        pending = cm.get_pending_for_session("sk2")
        assert pending is not None
        assert pending.clarify_id == "id2"

    def test_button_choice_does_not_auto_await(self):
        """Multi-choice clarify should NOT be in text-capture mode initially."""
        from tools import clarify_gateway as cm

        entry = cm.register("id3", "sk3", "Pick", ["X", "Y"])
        assert entry.awaiting_text is False
        assert cm.get_pending_for_session("sk3") is None

    def test_include_choice_prompts_returns_multi_choice_entry(self):
        """Gateway typed replies must see active choice prompts too."""
        from tools import clarify_gateway as cm

        cm.register("id3b", "sk3b", "Pick", ["X", "Y"])
        _confirm(cm, "id3b")
        pending = cm.get_pending_for_session("sk3b", include_choice_prompts=True)
        assert pending is not None
        assert pending.clarify_id == "id3b"


    def test_clear_session_cancels_pending_entries(self):
        """clear_session unblocks blocked threads with empty response."""
        from tools import clarify_gateway as cm

        cm.register("id7", "sk7", "Q?", ["A"])

        def waiter():
            return cm.wait_for_response("id7", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)
            cancelled = cm.clear_session("sk7")
            assert cancelled == 1
            result = fut.result(timeout=10.0)
            # clear_session sets response="" then the wait returns it
            assert result == ""

    def test_clear_session_preserves_answer_accepted_concurrently(self, monkeypatch):
        """Session cleanup cannot replace an accepted answer with cancellation."""
        from tools import clarify_gateway as cm

        entry = cm.register("id7-race", "sk7-race", "Q?", ["A", "B"])
        _confirm(cm, "id7-race")
        entered = threading.Event()
        release = threading.Event()
        original = cm._persist_transition

        def slow_answer(*args, **kwargs):
            if kwargs.get("state") == cm.STATE_ANSWERED:
                entered.set()
                assert release.wait(timeout=5)
            return original(*args, **kwargs)

        monkeypatch.setattr(cm, "_persist_transition", slow_answer)
        resolver = threading.Thread(
            target=lambda: cm.resolve_gateway_clarify("id7-race", "B")
        )
        resolver.start()
        assert entered.wait(timeout=5)
        clearer = threading.Thread(target=lambda: cm.clear_session("sk7-race"))
        clearer.start()
        time.sleep(0.05)
        assert clearer.is_alive()
        release.set()
        resolver.join(timeout=5)
        clearer.join(timeout=5)

        assert entry.state == cm.STATE_ANSWERED
        assert entry.response == "B"
        assert entry.event.is_set()

    def test_terminal_answer_rejects_timeout_regression(self):
        from tools import clarify_gateway as cm

        cm.register("id7-cas", "sk7-cas", "Q?", ["A", "B"])
        _confirm(cm, "id7-cas")
        assert cm.resolve_gateway_clarify("id7-cas", "A") is True

        assert cm._persist_transition(
            "id7-cas",
            expected_states=set(cm._RECOVERABLE_STATES),
            state=cm.STATE_TIMED_OUT,
        ) is False
        with cm._connect() as conn:
            row = conn.execute(
                "SELECT state, response FROM clarify_obligations WHERE clarify_id=?",
                ("id7-cas",),
            ).fetchone()
        assert row["state"] == cm.STATE_ANSWERED
        assert row["response"] == "A"


    def test_notify_register_unregister_clears_pending(self):
        """unregister_notify cancels any pending clarify so threads unwind."""
        from tools import clarify_gateway as cm

        cm.register("id9", "sk9", "Q?", ["A"])

        def waiter():
            return cm.wait_for_response("id9", timeout=10.0)

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(waiter)
            time.sleep(0.05)

            cm.register_notify("sk9", lambda entry: None)
            cm.unregister_notify("sk9")

            # unregister_notify calls clear_session; thread unwinds
            result = fut.result(timeout=10.0)
            assert result == ""

    def test_session_index_isolation(self):
        """Entries from different sessions don't leak across get_pending lookups."""
        from tools import clarify_gateway as cm

        cm.register("idA", "alpha", "Q?", None)
        cm.register("idB", "beta", "Q?", None)
        _confirm(cm, "idA", "open_text")
        _confirm(cm, "idB", "open_text")

        a = cm.get_pending_for_session("alpha")
        b = cm.get_pending_for_session("beta")
        assert a is not None and a.clarify_id == "idA"
        assert b is not None and b.clarify_id == "idB"

    def test_clarify_timeout_config_default(self):
        """get_clarify_timeout returns a positive int (default 3600)."""
        from tools import clarify_gateway as cm

        timeout = cm.get_clarify_timeout()
        # Default 3600s OR whatever is in the user's loaded config.
        # Floor check: must be a positive int, not crashed.
        assert isinstance(timeout, int)
        assert timeout > 0


class TestGatewayTextIntercept:
    """The gateway's _handle_message intercepts text replies to pending clarifies."""

    def setup_method(self):
        _clear_clarify_state()

    def test_get_pending_for_session_returns_oldest_text_awaiting(self):
        """When two clarifies are pending, get_pending_for_session returns the
        first that is awaiting_text (the older one if both)."""
        from tools import clarify_gateway as cm

        # Older multi-choice (not awaiting text)
        cm.register("first", "sk", "Q1?", ["A"])
        _confirm(cm, "first")
        # Newer open-ended (awaiting text)
        cm.register("second", "sk", "Q2?", None)
        _confirm(cm, "second", "open_text")

        pending = cm.get_pending_for_session("sk")
        # The newer one is awaiting text; the older isn't.
        assert pending is not None
        assert pending.clarify_id == "second"

        # Now flip the first to text mode too.  Both are awaiting text,
        # FIFO returns the older one.
        cm.mark_awaiting_text("first")
        pending2 = cm.get_pending_for_session("sk")
        assert pending2 is not None
        assert pending2.clarify_id == "first"
    def test_text_fallback_enables_awaiting_text_for_multi_choice(self):
        """When base send_clarify renders choices as text, mark_awaiting_text
        is called so the gateway text-intercept can capture the reply."""
        from tools import clarify_gateway as cm

        entry = cm.register("id-tf", "sk-tf", "Pick one", ["A", "B", "C"])
        # Initially, multi-choice does NOT await text (button path)
        assert entry.awaiting_text is False

        # The base text fallback confirms a numbered-text delivery.
        _confirm(cm, "id-tf", "numbered_text")

        # Now get_pending_for_session should find it
        pending = cm.get_pending_for_session("sk-tf")
        assert pending is not None
        assert pending.clarify_id == "id-tf"
        
        # Clean up
        cm.clear_session("sk-tf")


class TestCoverageGaps:
    """Cover remaining branches: signature(), get_entry miss, find_awaiting
    with deleted entry, cancel with None entry, timeout exception, get_notify."""

    def setup_method(self):
        _clear_clarify_state()

    def test_entry_signature(self):
        """_ClarifyEntry.signature() returns the expected dict."""
        from tools import clarify_gateway as cm

        entry = cm.register("sig1", "sk", "Q?", ["A", "B"])
        sig = entry.signature()
        assert sig["clarify_id"] == "sig1"
        assert sig["session_key"] == "sk"
        assert sig["question"] == "Q?"
        assert sig["choices"] == ["A", "B"]


    def test_wait_for_response_unknown_id_returns_none(self):
        """wait_for_response on a non-existent id returns None immediately."""
        from tools import clarify_gateway as cm

        assert cm.wait_for_response("nonexistent-id", timeout=0.1) is None


    def test_get_clarify_timeout_exception_returns_default(self, monkeypatch):
        """get_clarify_timeout returns 3600 when load_config raises."""
        from tools import clarify_gateway as cm

        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cm.get_clarify_timeout() == 3600


    def test_get_notify_returns_none_when_not_registered(self):
        """get_notify returns None for an unregistered session."""
        from tools import clarify_gateway as cm

        assert cm.get_notify("unregistered") is None


class TestClarifyTimeoutResolution:
    """resolve_clarify_timeout is the single source of truth for the clarify
    timeout, shared by the CLI, TUI/desktop, and messaging-gateway paths."""

    def test_canonical_agent_key(self):
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({"agent": {"clarify_timeout": 900}}) == 900


    def test_non_positive_preserved_as_unlimited_sentinel(self):
        """<= 0 is passed through verbatim — the waiting loops read it as
        'unlimited', so the resolver must not clamp it to a positive default."""
        from tools import clarify_gateway as cm

        assert cm.resolve_clarify_timeout({"agent": {"clarify_timeout": 0}}) == 0
        assert cm.resolve_clarify_timeout({"clarify": {"timeout": -1}}) == -1


class TestUnlimitedWait:
    """timeout <= 0 makes wait_for_response block until the answer arrives
    instead of auto-skipping."""

    def setup_method(self):
        _clear_clarify_state()

    def test_zero_timeout_waits_until_resolved(self):
        from tools import clarify_gateway as cm

        cm.register("u1", "sk", "Q?", ["A", "B"])
        _confirm(cm, "u1")
        result_box = {}

        def waiter():
            result_box["r"] = cm.wait_for_response("u1", timeout=0)

        t = threading.Thread(target=waiter)
        t.start()
        # An unlimited wait cannot finish while nothing resolves it: still
        # running after a comfortable margin (old code auto-skipped at once).
        t.join(timeout=1.5)
        assert t.is_alive()

        # Once resolved, the unlimited wait returns the real answer.
        cm.resolve_gateway_clarify("u1", "B")
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert result_box["r"] == "B"


class TestMultiSelectTextFallback:
    """Multi-select clarifies via the gateway text fallback.

    The adapter's numbered-list fallback asks the user to reply with
    comma/space-separated numbers; _coerce_text_response must map those to a
    JSON array of choice labels (which _parse_multi_select_response on the
    tool side decodes into a list).
    """

    def setup_method(self):
        _clear_clarify_state()

    def _register_multi(self, cid="m1", choices=("A", "B", "C")):
        from tools import clarify_gateway as cm
        entry = cm.register(cid, "sk", "Pick some", list(choices), multi_select=True)
        _confirm(cm, cid, "numbered_text")
        return entry

    def test_register_stores_multi_select_flag(self):
        entry = self._register_multi()
        assert entry.multi_select is True
        assert entry.signature()["multi_select"] is True


    def test_multi_select_without_choices_is_ignored(self):
        """multi_select on an open-ended clarify is meaningless — dropped."""
        from tools import clarify_gateway as cm
        entry = cm.register("s2", "sk", "Q?", None, multi_select=True)
        assert entry.multi_select is False


    def test_duplicate_selections_deduped(self):
        import json
        from tools import clarify_gateway as cm
        entry = self._register_multi()
        coerced = cm._coerce_text_response(entry, "1, 1, 2")
        assert json.loads(coerced) == ["A", "B"]

    def test_resolve_text_response_end_to_end(self):
        """resolve_text_response_for_session delivers the JSON array to the waiter."""
        import json
        from tools import clarify_gateway as cm
        self._register_multi(cid="m3")
        result_box = {}

        def waiter():
            result_box["r"] = cm.wait_for_response("m3", timeout=5)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert cm.resolve_text_response_for_session("sk", "1,2") is True
        t.join(timeout=5)
        assert json.loads(result_box["r"]) == ["A", "B"]

    def test_single_select_regression_numeric(self):
        """Single-select coercion unchanged: '2' maps to the choice label string."""
        from tools import clarify_gateway as cm
        entry = cm.register("s3", "sk", "Q?", ["A", "B", "C"])
        assert cm._coerce_text_response(entry, "2") == "B"

    def test_single_select_regression_label(self):
        from tools import clarify_gateway as cm
        entry = cm.register("s4", "sk", "Q?", ["A", "B"])
        assert cm._coerce_text_response(entry, "b") == "B"


class TestDurableClarifyRecovery:
    def setup_method(self):
        _clear_clarify_state()

    @staticmethod
    def _durable_state(cm, clarify_id: str) -> str:
        with cm._connect() as conn:
            row = conn.execute(
                "SELECT state FROM clarify_obligations WHERE clarify_id=?",
                (clarify_id,),
            ).fetchone()
        assert row is not None
        return str(row["state"])

    def test_unconfirmed_prompt_does_not_intercept_unrelated_text(self):
        from tools import clarify_gateway as cm

        cm.register("preconfirm", "session-a", "Pick", ["A", "B"])

        assert cm.resolve_text_response_for_session("session-a", "A") is False
        assert cm.resolve_text_response_for_session("different-session", "A") is False

    def test_valid_vote_resolves_after_delivery_confirmation(self):
        from tools import clarify_gateway as cm

        entry = cm.register("vote", "session-vote", "Pick", ["A", "B"])
        assert cm.mark_transport_accepted("vote", "poll-guid") is True
        _confirm(cm, "vote")

        assert cm.resolve_text_response_for_session("session-vote", "B") is True
        assert entry.response == "B"
        assert self._durable_state(cm, "vote") == cm.STATE_ANSWERED

    def test_confirmed_prompt_is_recovered_after_restart(self):
        from tools import clarify_gateway as cm

        cm.register(
            "recover-confirmed",
            "session-r",
            "Pick",
            ["A", "B"],
            platform="photon",
            chat_id="chat-r",
        )
        _confirm(cm, "recover-confirmed")
        _clear_clarify_state()

        recovered = cm.recover_pending(force=True)

        assert [entry.clarify_id for entry in recovered] == ["recover-confirmed"]
        assert recovered[0].recovered is True
        assert cm.get_pending_for_session(
            "session-r", include_choice_prompts=True
        ) is recovered[0]

    def test_recovered_process_local_buttons_get_one_numbered_fallback(self):
        from tools import clarify_gateway as cm

        cm.register(
            "recover-buttons",
            "session-buttons",
            "Pick",
            ["A", "B"],
            platform="telegram",
            chat_id="chat-buttons",
        )
        _confirm(cm, "recover-buttons", "adapter")
        _clear_clarify_state()

        entry = cm.recover_pending(force=True)[0]

        assert entry.needs_recovery_fallback is True
        assert cm.claim_text_fallback(entry.clarify_id) is True
        assert cm.mark_delivery_confirmed(entry.clarify_id, "numbered_text") is True
        assert entry.awaiting_text is True
        assert entry.needs_recovery_fallback is False

    def test_recovered_photon_poll_remains_correlated_without_resend(self):
        from tools import clarify_gateway as cm

        cm.register(
            "recover-photon",
            "session-photon",
            "Pick",
            ["A", "B"],
            platform="photon",
            chat_id="chat-photon",
        )
        assert cm.mark_transport_accepted("recover-photon", "poll-photon") is True
        _confirm(cm, "recover-photon", "native_poll")
        _clear_clarify_state()

        entry = cm.recover_pending(force=True)[0]

        assert entry.needs_recovery_fallback is False
        assert cm.get_pending_for_poll(
            platform="photon",
            chat_id="chat-photon",
            poll_guid="poll-photon",
        ) is entry

    def test_expired_prompt_is_timed_out_instead_of_recovered(self):
        from tools import clarify_gateway as cm

        cm.register(
            "expired",
            "session-expired",
            "Still needed?",
            None,
            expires_at=time.time() - 1,
        )
        _clear_clarify_state()

        assert cm.recover_pending(force=True) == []
        assert self._durable_state(cm, "expired") == cm.STATE_TIMED_OUT

    def test_ambiguous_fallback_restart_fails_without_duplicate_retry(self):
        from tools import clarify_gateway as cm

        cm.register("ambiguous", "session-f", "Pick", ["A", "B"])
        assert cm.mark_transport_accepted("ambiguous", "poll-guid") is True
        assert cm.claim_text_fallback("ambiguous") is True
        _clear_clarify_state()

        assert cm.recover_pending(force=True) == []
        assert self._durable_state(cm, "ambiguous") == cm.STATE_FAILED
