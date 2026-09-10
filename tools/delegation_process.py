"""Process-backed paid delegation; the canonical launcher owns admission/metering.

No credentials, prices, executable commands, or budget overrides come from tool
arguments. The operator's route catalog is resolved again by the launcher.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

TIERS = ("director", "senior", "medium", "fast")
DESKTOP_ENV = ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS")


def paid_config(config):
    value = config.get("paid_dispatch") or {}
    return value if value.get("enabled") is True else None


def catalog_for(config):
    cfg = paid_config(config)
    if not cfg:
        raise ValueError("Paid process delegation is not configured")
    path = Path(cfg["catalog_path"]).expanduser().resolve()
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1 or catalog.get("tiers") != list(TIERS):
        raise ValueError("Paid delegation requires the four-tier route catalog v1")
    return path, catalog


def select_tier(role, parent_tier="director"):
    default = "fast" if parent_tier == "medium" else "medium"
    tier = {None: default, "": default, "leaf": "fast", "orchestrator": "senior"}.get(role, role)
    if parent_tier not in TIERS or tier not in TIERS or TIERS.index(tier) <= TIERS.index(parent_tier):
        raise ValueError(f"{parent_tier} can delegate only to strictly lower tiers; requested {tier}")
    return tier


def worker_environment(source, *, host_desktop=False):
    env = dict(source)
    if not host_desktop:
        for name in DESKTOP_ENV:
            env.pop(name, None)
    return env


class ProcessChild:
    """The small child interface used by Hermes' existing background lifecycle."""

    def __init__(self, *, config, task_index, goal, context, parent_agent, role, max_iterations, task_count, **kwargs):
        from hermes_constants import get_hermes_home
        from tools.delegate_tool import _build_child_progress_callback, _resolve_workspace_hint, _expand_parent_toolsets

        self.config = config
        self.catalog_path, catalog = catalog_for(config)
        parent_tier = getattr(parent_agent, "_dispatch_tier", "director")
        self.tier = select_tier(role, parent_tier)
        route_id = catalog["defaults"][self.tier]
        route = catalog["routes"][route_id]
        allowed_toolsets = catalog.get("worker_toolsets", ["terminal", "file", "web", "browser", "skills", "delegation", "todo", "session_search"])
        parent_toolsets = getattr(parent_agent, "enabled_toolsets", None)
        inherited = _expand_parent_toolsets(set(parent_toolsets)) if parent_toolsets is not None else set(allowed_toolsets)
        self.model = route["model"]
        self.provider = route["provider"]
        self.session_id = "paid-" + uuid.uuid4().hex
        self._subagent_id = "sa-" + uuid.uuid4().hex[:12]
        self._parent_subagent_id = getattr(parent_agent, "_subagent_id", None)
        self._delegate_depth = getattr(parent_agent, "_delegate_depth", 0) + 1
        self._delegate_role = self.tier
        self._dispatch_tier = self.tier
        self._subagent_goal = goal
        self._parent_turn_id = getattr(parent_agent, "_current_turn_id", "") or ""
        self._process = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.tool_progress_callback = _build_child_progress_callback(
            task_index, goal, parent_agent, task_count,
            subagent_id=self._subagent_id, parent_id=self._parent_subagent_id,
            depth=self._delegate_depth - 1, model=self.model,
            session_ref={"session_id": self.session_id},
        )
        self.directory = get_hermes_home() / "cache" / "delegation" / "paid" / self.session_id
        self.request = {
            "schema_version": 1, "tier": self.tier, "parent_tier": parent_tier,
            "route_id": route_id, "goal": goal, "context": context or "",
            "session_id": self.session_id, "subagent_id": self._subagent_id,
            "parent_session_id": getattr(parent_agent, "session_id", None),
            "parent_subagent_id": self._parent_subagent_id,
            "parent_turn_id": self._parent_turn_id, "depth": self._delegate_depth,
            "max_iterations": max_iterations,
            "brief": {
                "goal": goal,
                "responsibility": "Complete this bounded assignment and report to the originating Director session.",
                "current_evidence_and_answered_decisions": "See the focused context field and its source references. Reuse answered decisions; do not reload full transcripts.",
                "source_session": getattr(parent_agent, "session_id", None),
                "result_contract": "Completed work, evidence paths, decisions, remaining work, verification, and exact next action.",
                "escalation": "Report missing authority, contradictory evidence, or reproducible failures to the Director. Do not repeat already answered questions.",
                "review": "Consequential changes require an independent second opinion before activation.",
            },
            "toolsets": [name for name in allowed_toolsets if name in inherited],
            "workdir": _resolve_workspace_hint(parent_agent) or os.getcwd(),
        }
        self.parent_agent = parent_agent
        if hasattr(parent_agent, "_active_children"):
            lock = getattr(parent_agent, "_active_children_lock", None) or self._lock
            with lock:
                parent_agent._active_children.append(self)
        self._emit("subagent.spawn_requested", preview=goal)

    def _emit(self, event, **data):
        if self.tool_progress_callback:
            try:
                self.tool_progress_callback(event, **data)
            except Exception:
                pass

    def interrupt(self, message=None):
        self._stop.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.send_signal(signal.SIGTERM)

    def close(self):
        self.interrupt()

    def run_entry(self, task_index):
        from tools.delegate_tool import _register_subagent, _unregister_subagent
        started = time.monotonic()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        request_path = self.directory / "request.json"
        request_path.write_text(json.dumps(self.request))
        request_path.chmod(0o600)
        cfg = paid_config(self.config)
        env = worker_environment(os.environ, host_desktop=cfg.get("allow_host_desktop") is True)
        env["PHYSEA_DISPATCH_TIER"] = self.request["parent_tier"]
        env["PHYSEA_SESSION_SOURCE"] = self.request["parent_session_id"] or self.session_id
        env["PHYSEA_DISPATCH_ORIGIN_PID"] = str(os.getpid())
        env["PHYSEA_DISPATCH_CATALOG"] = str(self.catalog_path)
        command = [cfg.get("launcher", "codex-bedrock"), "--hermes-worker-request", str(request_path)]
        result = None
        partial = ""
        receipt = {}
        meter = {}
        record = {"subagent_id": self._subagent_id, "parent_id": self._parent_subagent_id,
                  "depth": self._delegate_depth - 1, "goal": self._subagent_goal,
                  "model": self.model, "provider": self.provider, "tier": self.tier,
                  "started_at": time.time(), "status": "running", "tool_count": 0, "agent": self}
        _register_subagent(record)
        try:
            self._emit("subagent.start", preview=self._subagent_goal)
            with (self.directory / "events.jsonl").open("w") as transcript, (self.directory / "stderr.log").open("w") as errors:
                with self._lock:
                    if self._stop.is_set():
                        return {"task_index": task_index, "status": "interrupted", "summary": "Cancelled before launch"}
                    self._process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                                     stdout=subprocess.PIPE, stderr=errors,
                                                     text=True, start_new_session=True)
                for line in self._process.stdout:
                    transcript.write(line)
                    transcript.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "hermes.receipt":
                        receipt = event["receipt"]
                        record.update(receipt)
                    elif event.get("type") == "hermes.meter":
                        meter = event["meter"]
                    elif event.get("type") == "hermes.text":
                        partial = (partial + str(event.get("text", "")))[-24000:]
                    elif event.get("type") == "hermes.result":
                        result = event["result"]
                    elif event.get("type") == "hermes.progress" and self.tool_progress_callback:
                        self.tool_progress_callback(event["event"], **event.get("data", {}))
                rc = self._process.wait()
            result = result or {"summary": partial, "status": "timeout" if rc == 124 else "failed"}
            if rc != 0:
                result["status"] = "timeout" if rc == 124 else "interrupted" if self._stop.is_set() else "failed"
                reason = result.get("error") or "Paid worker stopped"
                result["error"] = f"{reason} (exit {rc}); preserved output: {self.directory}"
            result.update(task_index=task_index, model=self.model, provider=self.provider, tier=self.tier,
                          session_id=self.session_id, receipt=receipt, output_directory=str(self.directory),
                          duration_seconds=round(time.monotonic() - started, 2), _child_role=self.tier)
            result["meter"] = meter
            if meter:
                cost_key = "tree_category_estimate_usd" if self.request["parent_tier"] == "director" else "run_category_estimate_usd"
                result["_child_cost_usd"] = float(meter[cost_key])
            self._emit("subagent.complete", status=result["status"], summary=result.get("summary", ""),
                       duration_seconds=result["duration_seconds"], api_calls=result.get("api_calls", 0))
            return result
        except Exception as exc:
            self._emit("subagent.complete", status="failed", summary=str(exc))
            return {"task_index": task_index, "status": "failed", "summary": partial,
                    "error": str(exc), "output_directory": str(self.directory),
                    "receipt": receipt, "_child_role": self.tier,
                    "duration_seconds": round(time.monotonic() - started, 2)}
        finally:
            self.interrupt()
            _unregister_subagent(self._subagent_id)
            children = getattr(self.parent_agent, "_active_children", [])
            lock = getattr(self.parent_agent, "_active_children_lock", None) or self._lock
            with lock:
                if self in children:
                    children.remove(self)
