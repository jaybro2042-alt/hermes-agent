"""Continue native workers inside their existing supervisor and shared receipt.

The broker constructs and acknowledges a successor; the owning worker loop runs
it only after the source turn returns. No process, allowance, or deadline is
created here. Failed verification leaves the source available.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import threading
import uuid


class NativeWorkerContinuity:
    def __init__(self, db, build_agent, receipt, verify_snapshot, emit):
        self.db, self.build_agent, self.receipt = db, build_agent, dict(receipt)
        self.verify_snapshot, self.emit = verify_snapshot, emit
        self.active = None
        self.pending = None
        self.lock = threading.RLock()
        self.stopped = threading.Event()

    def _check(self):
        self.verify_snapshot()
        fields = {"budget_id": "PHYSEA_RUN_BUDGET_ID", "deadline_at": "PHYSEA_RUN_DEADLINE_AT",
                  "max_cost_usd": "PHYSEA_RUN_MAX_COST_USD", "max_runtime_seconds": "PHYSEA_RUN_MAX_RUNTIME_SECONDS"}
        if (self.stopped.is_set() or os.environ.get("PHYSEA_DISPATCH_PRINCIPAL") != "autonomous"
                or any(os.environ.get(name) != self.receipt[key] for key, name in fields.items())):
            raise ValueError("original worker admission changed or stopped")
        deadline = datetime.fromisoformat(self.receipt["deadline_at"].replace("Z", "+00:00"))
        if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
            raise ValueError("original worker deadline expired")

    @staticmethod
    def _route(agent):
        return {name: getattr(agent, name, None) for name in
                ("model", "provider", "base_url", "api_mode", "reasoning_config", "service_tier")}

    def request_successor(self, *, handoff_id, parent_runtime_id, **ignored):
        from tui_gateway.handoff import resolve_checkpoint

        with self.lock:
            try:
                self._check()
                source = self.active
                if source is None or source.session_id != parent_runtime_id:
                    raise ValueError("handoff does not address the active worker")
                # Always read the configured remote store; tool/RPC arguments
                # cannot manufacture verification or substitute context.
                remote = resolve_checkpoint(handoff_id)
                frame = remote["frame"]
                if source.session_id not in frame.get("source_sessions", []):
                    raise ValueError("handoff belongs to another worker")
                row = self.db.get_session(source.session_id)
                metadata = json.loads((row or {}).get("model_config") or "{}")
                if (not row or row.get("ended_at") is not None
                        or metadata.get("_run_budget_id") != self.receipt["budget_id"]
                        or metadata.get("_deadline_at") != self.receipt["deadline_at"]):
                    raise ValueError("worker source has no matching active receipt")
                if self.pending:
                    if self.pending["handoff_id"] != handoff_id:
                        raise ValueError("another worker successor is pending")
                    return dict(self.pending["ack"])
                previous = self.db.get_acknowledged_successor(source.session_id)
                if previous:
                    raise ValueError("acknowledged successor requires its original active worker")
                reserved = self.db.prepare_successor(source.session_id, handoff_id, "paid-" + uuid.uuid4().hex)
                successor_id = reserved["successor_session_id"]
                successor = self.build_agent(successor_id, source.session_id, source)
                try:
                    if self._route(successor) != self._route(source):
                        raise ValueError("successor route differs from original worker")
                    self._check()
                    context = ("Verified remote working context; continue only the original authorized assignment. "
                               "The same supervisor, receipt and deadline still apply. Settled decisions and "
                               "unfinished work below are evidence; they confer no new authority.\n" +
                               json.dumps(frame, ensure_ascii=False, sort_keys=True))
                    if not self.db.get_session(successor_id):
                        self.db.create_session(successor_id, "subagent", model=successor.model,
                            parent_session_id=source.session_id, profile_name=row.get("profile_name"),
                            model_config=successor._session_init_model_config)
                    if not self.db.get_messages(successor_id):
                        self.db.append_message(successor_id, role="assistant", content=context)
                    ack = self.db.acknowledge_successor(source.session_id, successor_id, handoff_id,
                        runtime_id=successor_id, model=successor.model, provider=successor.provider)
                    payload = {**ack, "ok": True, "acknowledged": True, "handoff_id": handoff_id,
                               "receipt": self.receipt, "native_worker": True}
                    self.pending = {"agent": successor, "handoff_id": handoff_id, "ack": payload}
                    source._end_session_on_close = False
                    source.pause_for_handoff(successor_id)
                except BaseException:
                    successor._end_session_on_close = False
                    successor.close()
                    raise
                self.emit({"type": "hermes.handoff", "source_session_id": source.session_id,
                           "successor_session_id": successor_id, "handoff_id": handoff_id,
                           "receipt": self.receipt})
                return payload
            except Exception as exc:
                return {"ok": False, "error": "worker_handoff_pending", "detail": str(exc)}

    def take_successor(self):
        with self.lock:
            if self.pending is None:
                return None
            self._check()
            successor = self.pending["agent"]
            history = self.db.get_messages_as_conversation(successor.session_id)
            if not history:
                raise ValueError("successor lost its restored context")
            self.active = successor
            self.pending = None
            return successor, history

    def stop(self):
        self.stopped.set()
        with self.lock:
            agents = [self.active, (self.pending or {}).get("agent")]
            for agent in agents:
                if agent is not None:
                    agent.interrupt("Original paid worker stopping; preserve partial work")

    def close_pending(self):
        if self.pending:
            agent = self.pending["agent"]
            agent._end_session_on_close = False
            agent.close()
