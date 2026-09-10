"""Offline continuity: actual DB, gateway lifecycle and AIAgent construction."""
import json
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tui_gateway import handoff, server

HANDOFF_ID = "ho-0123456789abcdef"
FRAME = {"handoff_id": HANDOFF_ID, "integrity": "verified_against_store",
         "objective": "Finish Hermes recovery", "decisions": ["ICP remains unproven"],
         "artifact_state": ["Changes isolated; live services untouched"],
         "next_action": "Test current conversation continuity", "evidence": ["test receipt"]}


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    import httpx
    from run_agent import AIAgent
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    (home / "config.yaml").write_text("model:\n  default: gpt-6-astra\n  provider: openai-codex\nagent:\n  reasoning_effort: high\nauxiliary:\n  background_review:\n    enabled: false\n")
    def no_network(*args, **kwargs):
        raise AssertionError("inference/network must not run in continuity tests")
    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    monkeypatch.setattr(server, "_sessions", {})
    db = SessionDB(db_path=home / "state.db")
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda: [])
    # Only the credential boundary is substituted. Gateway and actual AIAgent
    # constructors remain real; no chat/inference method is invoked.
    def resolve(kwargs):
        assert kwargs["requested"] == "openai-codex"
        return SimpleNamespace(used_fallback=False, runtime={"provider": "openai-codex",
            "api_mode": "codex_responses", "api_key": "offline-test-token",
            "base_url": "http://127.0.0.1:1/v1"})
    monkeypatch.setattr(server, "_resolve_runtime_with_fallback", resolve)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *a: None)
    monkeypatch.setattr(handoff, "_resume_resolver", lambda _: {"ok": True, "frame": dict(FRAME)})
    parent = server._methods["session.create"]("parent", {"source": "desktop",
        "cwd": str(tmp_path), "model": "gpt-6-astra", "provider": "openai-codex",
        "reasoning_effort": "high", "fast": False})["result"]
    source = server._sessions[parent["session_id"]]
    server._start_agent_build(parent["session_id"], source)
    assert source["agent_ready"].wait(15), "source constructor did not finish"
    assert isinstance(source.get("agent"), AIAgent), source.get("agent_error")
    server._ensure_session_db_row(source)
    monkeypatch.setattr(handoff, "_resume_resolver", lambda _: {
        "ok": True, "frame": {**FRAME, "source_sessions": [source["session_key"]]}})
    yield db, parent, source, emitted
    for session in list(server._sessions.values()):
        if session.get("agent"):
            session["agent"].close()
    db.close()


def request(source, **extra):
    return server._methods["handoff.successor.create"]("handoff", {
        "parent_runtime_id": source["session_key"], "handoff_id": HANDOFF_ID, **extra})


def test_actual_agent_ready_with_same_route_context_and_source_alive(runtime):
    from run_agent import AIAgent
    db, parent, source, emitted = runtime
    response = request(source)
    assert "result" in response, response
    result = response["result"]
    successor = server._sessions[result["successor_runtime_id"]]
    assert isinstance(successor["agent"], AIAgent)
    assert successor["agent"].provider == source["agent"].provider == "openai-codex"
    assert successor["agent"].reasoning_config == source["agent"].reasoning_config
    assert successor["agent"].service_tier == source["agent"].service_tier
    assert result["acknowledged"] is True
    assert db.get_session(source["session_key"])["ended_at"] is None
    assert db.get_session(result["successor_session_id"])["parent_session_id"] == source["session_key"]
    assert "ICP remains unproven" in db.get_messages(result["successor_session_id"])[0]["content"]
    assert emitted[-1][0] == "handoff.successor.created"
    again = request(source)["result"]
    assert again["successor_session_id"] == result["successor_session_id"]
    assert len(server._sessions) == 2
    blocked = source["agent"].run_conversation("Do not execute in the old source")
    assert blocked["error"] == "handoff_source_paused" and blocked["api_calls"] == 0
    source["agent"].clear_interrupt()
    assert source["agent"].run_conversation("Still blocked")["api_calls"] == 0
    resumed = server._methods["session.resume"]("resume-source", {"session_id": source["session_key"]})
    assert resumed["result"]["session_id"] == result["successor_runtime_id"]


def test_rpc_cannot_forge_receipt_or_context(runtime, monkeypatch):
    db, _, source, _ = runtime
    monkeypatch.setattr(handoff, "_resume_resolver", lambda _: {"ok": False, "frame": FRAME})
    response = request(source, receipt={"ok": True, "frame": FRAME}, context="forged")
    assert response["error"]["code"] == 4007
    assert db.get_acknowledged_successor(source["session_key"]) is None
    assert len(server._sessions) == 1


def test_failed_agent_build_cannot_ack_or_close_source(runtime, monkeypatch):
    db, _, source, _ = runtime
    def broken(*a, **k):
        raise RuntimeError("credential unavailable")
    monkeypatch.setattr(server, "_make_agent", broken)
    response = request(source)
    assert response["error"]["code"] == 4010
    assert db.get_acknowledged_successor(source["session_key"]) is None
    assert db.get_session(source["session_key"])["ended_at"] is None


def test_disconnected_conversation_retains_work(runtime):
    _, _, source, _ = runtime
    source["history"] = [{"role": "user", "content": "Keep my ongoing task"}]
    source["transport"] = server._detached_ws_transport
    assert not server._ws_session_is_orphaned(source)
    source["history"] = []
    source["queued_prompt"] = {"text": "My accepted follow-up"}
    assert not server._ws_session_is_orphaned(source)
    source.pop("queued_prompt")
    assert server._ws_session_is_orphaned(source)


def test_restart_rebuilds_same_durable_successor(runtime):
    db, _, source, _ = runtime
    first = request(source)["result"]
    child = server._sessions.pop(first["successor_runtime_id"])
    child["agent"].release_clients()
    source.pop("handoff_attempt", None)
    second = request(source)["result"]
    assert second["successor_session_id"] == first["successor_session_id"]
    assert second["successor_runtime_id"] != first["successor_runtime_id"]
    assert server._sessions[second["successor_runtime_id"]]["agent"] is not None
    assert len(db.get_messages(first["successor_session_id"])) == 1


def test_restart_before_ack_reuses_reserved_child_and_context(runtime, monkeypatch):
    db, _, source, _ = runtime
    real_ack = db.acknowledge_successor
    def fail_ack(*args, **kwargs):
        raise RuntimeError("simulated transaction unavailable")
    monkeypatch.setattr(db, "acknowledge_successor", fail_ack)
    assert request(source)["error"]["code"] == 4007
    pending = json.loads(db.get_meta("session_successor_pending:" + source["session_key"]))
    child_sid = source.pop("handoff_attempt")["runtime_id"]
    child = server._sessions.pop(child_sid)
    child["agent"].release_clients()
    monkeypatch.setattr(db, "acknowledge_successor", real_ack)
    result = request(source)["result"]
    assert result["successor_session_id"] == pending["successor_session_id"]
    assert len(db.get_messages(result["successor_session_id"])) == 1


def test_wrong_remote_source_cannot_cross_sessions(runtime, monkeypatch):
    db, _, source, _ = runtime
    monkeypatch.setattr(handoff, "_resume_resolver", lambda _: {
        "ok": True, "frame": {**FRAME, "source_sessions": ["another-user-session"]}})
    assert request(source)["error"]["code"] == 4007
    assert db.get_acknowledged_successor(source["session_key"]) is None


def test_compacted_source_uses_actual_agent_id_and_preserves_role(runtime, monkeypatch):
    db, _, source, _ = runtime
    old_key = source["session_key"]
    db.end_session(old_key, "compression")
    key = "post-compaction-source"
    db.create_session(key, "desktop", parent_session_id=old_key,
        model_config={"_session_role": "director", "_context_manifest": "context/manifest.md"})
    source["agent"].session_id = key
    monkeypatch.setattr(handoff, "_resume_resolver", lambda _: {
        "ok": True, "frame": {**FRAME, "source_sessions": [key]}})
    response = server._methods["handoff.successor.create"]("after-compact", {
        "parent_runtime_id": key, "handoff_id": HANDOFF_ID})
    assert "result" in response, response
    child = response["result"]["successor_session_id"]
    config = json.loads(db.get_session(child)["model_config"])
    assert config["_session_role"] == "director"
    assert config["_context_manifest"] == "context/manifest.md"
    assert db.get_acknowledged_successor(key)["successor_session_id"] == child
    assert db.get_acknowledged_successor(old_key) is None
    assert db.get_continuity_tip(old_key) == child
    resumed = server._methods["session.resume"]("old-compressed-id", {"session_id": old_key})
    assert resumed["result"]["session_id"] == response["result"]["successor_runtime_id"]


def test_paid_worker_cannot_bypass_supervisor(runtime):
    db, _, source, _ = runtime
    metadata = {"_run_budget_id": "same-tree", "_deadline_at": "2026-09-10T20:00:00Z", "_delegate_from": "director"}
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET model_config=? WHERE id=?",
        (json.dumps(metadata), source["session_key"])))
    assert request(source)["error"]["code"] == 4011
    assert len(server._sessions) == 1


def test_viewer_close_does_not_end_phone_work(runtime, monkeypatch):
    db, _, source, _ = runtime
    db.record_gateway_session_peer(source["session_key"], source="custom-phone-plugin",
        session_key="phone:user-a:thread-1", chat_id="thread-1", user_id="user-a")
    from tools.process_registry import process_registry
    killed = []
    monkeypatch.setattr(process_registry, "kill_all", lambda **kwargs: killed.append(kwargs))
    server._teardown_session(source, end_reason="ws_orphan_reap")
    assert db.get_session(source["session_key"])["ended_at"] is None
    assert killed == []


@pytest.mark.parametrize("field,value", [("profile_name", "other-profile"), ("user_id", "other-user"),
    ("chat_id", "other-chat"), ("thread_id", "other-thread")])
def test_atomic_ack_rejects_foreign_owner(runtime, field, value):
    db, _, source, _ = runtime
    child = "foreign-child"
    db.create_session(child, "desktop", parent_session_id=source["session_key"], **{field: value})
    db.append_message(child, role="assistant", content="restored")
    with pytest.raises(ValueError):
        db.acknowledge_successor(source["session_key"], child, HANDOFF_ID,
            runtime_id="foreign", model="gpt-6-astra", provider="openai-codex")
    assert db.get_acknowledged_successor(source["session_key"]) is None


def test_exact_phone_binding_adopts_once_and_survives_restart(runtime, tmp_path):
    from gateway.session import SessionStore, SessionSource
    from gateway.config import GatewayConfig, Platform
    db, _, source, _ = runtime
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    origin = SessionSource(platform=Platform.TELEGRAM, chat_id="phone-thread", user_id="user-a", chat_type="dm")
    entry = store.get_or_create_session(origin)
    original = entry.session_id
    # Exercise the exact same atomic link + real inbound routing path used by
    # a gateway session viewed in Desktop, without sending any actual message.
    child = "phone-successor"
    db.create_session(child, "telegram", parent_session_id=original)
    db.append_message(child, role="assistant", content="settled answer; unfinished task")
    db.acknowledge_successor(original, child, HANDOFF_ID,
        runtime_id="ready-child", model="model", provider="provider")
    assert store.get_or_create_session(origin).session_id == child
    assert store.get_or_create_session(origin).session_id == child
    other = SessionSource(platform=Platform.TELEGRAM, chat_id="other-thread", user_id="user-b", chat_type="dm")
    assert store.get_or_create_session(other).session_id != child
    store._db.close()
    resumed = SessionStore(tmp_path / "sessions", config)
    assert resumed.get_or_create_session(origin).session_id == child
    assert db.get_session(original)["ended_at"] is None
    resumed._db.close()
