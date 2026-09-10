"""A real durable successor remains inside the original paid worker admission."""
import json
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools.delegation_continuity import NativeWorkerContinuity
from tui_gateway import handoff

CHECKPOINT = "ho-0123456789abcdef"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    receipt = dict(budget_id="run-original", deadline_at="2099-01-01T00:00:00+00:00",
                   max_cost_usd="5.00", max_runtime_seconds="900")
    monkeypatch.setenv("PHYSEA_DISPATCH_PRINCIPAL", "autonomous")
    for name, field in {"PHYSEA_RUN_BUDGET_ID": "budget_id", "PHYSEA_RUN_DEADLINE_AT": "deadline_at",
                        "PHYSEA_RUN_MAX_COST_USD": "max_cost_usd", "PHYSEA_RUN_MAX_RUNTIME_SECONDS": "max_runtime_seconds"}.items():
        monkeypatch.setenv(name, receipt[field])
    db = SessionDB(db_path=tmp_path / "state.db")
    built, events = [], []
    def build(session_id, parent_id, source=None):
        agent = SimpleNamespace(session_id=session_id, model="sol", provider="custom", api_mode="codex_responses",
            base_url="http://127.0.0.1:1/v1", reasoning_config={"effort": "high"}, service_tier="priority",
            _session_init_model_config={"_run_budget_id": receipt["budget_id"], "_deadline_at": receipt["deadline_at"],
                "_delegate_from": "original-director", "_delegate_role": "senior"},
            _end_session_on_close=True, closed=False, paused=None, interrupted=False)
        agent.close = lambda: setattr(agent, "closed", True)
        agent.pause_for_handoff = lambda key: setattr(agent, "paused", key)
        agent.interrupt = lambda message: setattr(agent, "interrupted", True)
        built.append(agent)
        return agent
    source = build("worker-source", "original-director")
    db.create_session("original-director", "desktop")
    db.create_session(source.session_id, "subagent", model=source.model,
        parent_session_id="original-director", model_config=source._session_init_model_config)
    db.append_message(source.session_id, "user", "Finish the assigned change")
    frame = {"handoff_id": CHECKPOINT, "integrity": "verified_against_store",
             "objective": "Finish the assigned change", "artifact_state": ["tests remain"],
             "next_action": "Run the remaining test", "decisions": ["Keep original route"],
             "source_sessions": [source.session_id]}
    monkeypatch.setattr(handoff, "_resume_resolver", lambda key: {"ok": True, "frame": dict(frame)})
    owner = NativeWorkerContinuity(db, build, receipt, lambda: None, events.append)
    owner.active = source
    yield owner, source, db, frame, built, events
    db.close()


def create(owner, source, **kwargs):
    return owner.request_successor(handoff_id=CHECKPOINT, parent_runtime_id=source.session_id, **kwargs)


def test_handoff_preserves_receipt_route_source_and_resumes_exactly_once(worker):
    owner, source, db, frame, built, events = worker
    result = create(owner, source)
    assert result["ok"] and result["acknowledged"], result
    assert result["receipt"] == owner.receipt
    assert source.paused == result["successor_session_id"]
    assert db.get_session(source.session_id)["ended_at"] is None
    assert not source.closed and source._end_session_on_close is False
    assert create(owner, source)["successor_session_id"] == result["successor_session_id"]
    assert len(built) == 2 and len(events) == 1
    successor, history = owner.take_successor()
    assert owner._route(successor) == owner._route(source)
    assert "Keep original route" in history[0]["content"]
    metadata = json.loads(db.get_session(successor.session_id)["model_config"])
    assert metadata["_delegate_from"] == "original-director"
    assert metadata["_run_budget_id"] == "run-original"
    assert metadata["_deadline_at"] == owner.receipt["deadline_at"]
    assert owner.take_successor() is None
    assert owner.active is successor


@pytest.mark.parametrize("fault", ["wrong-source", "rejected", "changed-receipt", "expired", "changed-snapshot"])
def test_failed_verification_never_pauses_or_builds(worker, monkeypatch, fault):
    owner, source, db, frame, built, events = worker
    if fault == "wrong-source":
        frame["source_sessions"] = ["foreign-worker"]
    elif fault == "rejected":
        monkeypatch.setattr(handoff, "_resume_resolver", lambda key: {"ok": False, "frame": frame})
    elif fault == "changed-receipt":
        monkeypatch.setenv("PHYSEA_RUN_BUDGET_ID", "renewed")
    elif fault == "expired":
        owner.receipt["deadline_at"] = "2000-01-01T00:00:00+00:00"
        monkeypatch.setenv("PHYSEA_RUN_DEADLINE_AT", owner.receipt["deadline_at"])
    else:
        def changed():
            raise ValueError("request snapshot changed")
        owner.verify_snapshot = changed
    assert create(owner, source, receipt={"ok": True, "frame": frame})["ok"] is False
    assert source.paused is None and len(built) == 1 and not events
    assert db.get_acknowledged_successor(source.session_id) is None


def test_build_route_mismatch_retains_source_and_no_ack(worker):
    owner, source, db, _, built, _ = worker
    original = owner.build_agent
    def wrong(*args):
        result = original(*args)
        result.reasoning_config = {"effort": "low"}
        return result
    owner.build_agent = wrong
    assert create(owner, source)["ok"] is False
    assert source.paused is None
    assert built[-1].closed
    assert db.get_acknowledged_successor(source.session_id) is None


def test_stop_before_successor_execution_prevents_new_work(worker):
    owner, source, db, _, built, _ = worker
    assert create(owner, source)["ok"]
    owner.stop()
    with pytest.raises(ValueError, match="stopped"):
        owner.take_successor()
    assert all(agent.interrupted for agent in built)
    owner.close_pending()
    assert built[-1].closed
    assert db.get_session(source.session_id)["ended_at"] is None


def test_deadline_is_rechecked_after_acknowledgment(worker, monkeypatch):
    owner, source, _, _, _, _ = worker
    assert create(owner, source)["ok"]
    monkeypatch.setenv("PHYSEA_RUN_DEADLINE_AT", "2099-12-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="changed"):
        owner.take_successor()


def test_real_worker_runs_successor_under_original_meter(worker, monkeypatch, tmp_path):
    """Real AIAgent, provider HTTP and session DB; only remote storage is a fixture."""
    import io
    from contextlib import closing
    import signal
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import hermes_state
    from run_agent import AIAgent
    from tools import delegation_worker

    _, _, _, _, _, _ = worker  # supplies the immutable admission environment
    home = tmp_path / "runtime"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    monkeypatch.setenv("OFFLINE_KEY", "offline-value-never-a-provider-credential")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    with closing(SessionDB(db_path=home / "state.db")) as db:
        db.create_session("original-director", "desktop")
    requests = []
    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            usage = {"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44}
            payload = {"id": "offline", "object": "chat.completion", "created": 1, "model": "offline-model",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "Checked result."},
                                    "finish_reason": "stop"}], "usage": usage}
            content_type = "application/json"
            if body.get("stream"):
                chunks = [{"id": "offline", "object": "chat.completion.chunk", "created": 1, "model": "offline-model",
                           "choices": [{"index": 0, "delta": {"content": "Checked result."}, "finish_reason": None}]},
                          {"id": "offline", "object": "chat.completion.chunk", "created": 1, "model": "offline-model",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}]
                data = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    provider = HTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(delegation_worker.sys, "stdin", io.StringIO(
        '{"ok":true,"reservation_id":"first"}\n{"ok":true}\n'
        '{"ok":true,"reservation_id":"second"}\n{"ok":true}\n'))
    original = AIAgent.run_conversation
    sessions = []
    background = SimpleNamespace(closed=False)
    background.close = lambda: setattr(background, "closed", True)
    def run(self, *args, **kwargs):
        sessions.append(self.session_id)
        if len(sessions) == 2:
            assert not background.closed, "handoff killed the source's running leaf"
        result = original(self, *args, **kwargs)
        if len(sessions) == 1:
            self._active_children.append(background)
            frame = {"handoff_id": CHECKPOINT, "integrity": "verified_against_store",
                     "objective": "Finish assignment", "artifact_state": ["First check done"],
                     "next_action": "Perform final check", "source_sessions": [self.session_id]}
            monkeypatch.setattr(handoff, "_resume_resolver", lambda key: {"ok": True, "frame": frame})
            ack = handoff.request_successor(handoff_id=CHECKPOINT, parent_runtime_id=self.session_id)
            assert ack["ok"], ack
            result.update(completed=False, interrupted=True)
        return result
    monkeypatch.setattr(AIAgent, "run_conversation", run)
    request = dict(tier="fast", parent_tier="director", route_id="test", goal="Finish assignment",
                   context="Use only the supplied evidence.", workdir=str(tmp_path), depth=1,
                   toolsets=["file"], max_iterations=3, session_id="native-source",
                   parent_session_id="original-director", parent_subagent_id=None, subagent_id="worker-one")
    route = dict(model="offline-model", provider="custom", base_url=f"http://127.0.0.1:{provider.server_port}/v1",
                 api_mode="chat_completions", api_key_env="OFFLINE_KEY", tiers=["fast"])
    events = []
    previous_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert delegation_worker.worker_main(request, {"routes": {"test": route}}, events.append, lambda: None) == 0
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
    assert len(sessions) == 2 and len(set(sessions)) == 2
    assert background.closed, "whole-worker shutdown must still clean up leaves"
    assert len([event for event in events if event["type"] == "hermes.reserve"]) == 2
    assert len([event for event in events if event["type"] == "hermes.settle"]) == 2
    assert len([event for event in events if event["type"] == "hermes.receipt"]) == 1
    transition = next(event for event in events if event["type"] == "hermes.handoff")
    assert transition["receipt"]["budget_id"] == "run-original"
    assert transition["receipt"]["deadline_at"] == "2099-01-01T00:00:00+00:00"
    result = next(event for event in events if event["type"] == "hermes.result")["result"]
    assert result["status"] == "completed" and result["api_calls"] == 2
    assert sum("messages" in body for body in requests) == 2
    with closing(SessionDB(db_path=home / "state.db")) as db:
        assert db.get_session("native-source")["ended_at"] is None
        assert "First check done" in db.get_messages(sessions[1])[0]["content"]
