import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from tools.delegation_process import ProcessChild, select_tier, worker_environment
from tools.delegation_worker import UsageJournal, RequestReservations, PaidRequestStopped, read_snapshot, resolve_route_runtime


def test_worker_discovers_only_handoff_without_sampling(monkeypatch):
    from tools import mcp_tool
    from tools.delegation_worker import discover_worker_handoff_tools
    configs = []
    monkeypatch.setattr(mcp_tool, "_load_mcp_config", lambda: {
        "liminal": {"url": "http://127.0.0.1:1/mcp", "enabled": True},
        "unrelated": {"command": "must-not-launch"}})
    def register(cfg):
        configs.append(cfg)
        return ["mcp__liminal__handoff"]
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", register)
    discover_worker_handoff_tools(["file"])
    assert configs == []
    discover_worker_handoff_tools(["mcp-liminal"])
    assert list(configs[0]) == ["liminal"]
    assert configs[0]["liminal"]["tools"]["include"] == ["handoff"]
    assert configs[0]["liminal"]["sampling"]["enabled"] is False
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", lambda cfg: [])
    with pytest.raises(PaidRequestStopped, match="no worker inference"):
        discover_worker_handoff_tools(["mcp-liminal"])


def test_hierarchy_allows_only_lower_tiers():
    assert select_tier("fast", "director") == "fast"
    assert select_tier("fast", "senior") == "fast"
    assert select_tier("medium", "senior") == "medium"
    assert select_tier(None, "medium") == "fast"
    for parent, child in [("fast", "fast"), ("medium", "senior"), ("senior", "senior"), ("director", "missing")]:
        with pytest.raises(ValueError, match="strictly lower"):
            select_tier(child, parent)


def test_desktop_environment_requires_operator_opt_in():
    env = {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0", "XAUTHORITY": "/xauth",
           "DBUS_SESSION_BUS_ADDRESS": "private", "TERM": "xterm"}
    assert worker_environment(env) == {"TERM": "xterm"}
    assert worker_environment(env, host_desktop=True) == env


def test_usage_journal_keeps_main_aux_and_cache_categories_distinct():
    events = []
    journal = UsageJournal(events.append)
    journal.record("main", {"model": "actual-model", "input_tokens": 3, "cache_read_tokens": 10, "cache_write_tokens": 2, "output_tokens": 5})
    journal.record("main", {"model": "actual-model", "input_tokens": 4})
    journal.record("aux:compression", {"model": "other-model", "input_tokens": 8})
    assert events[1]["usage"] == dict(input_tokens=7, output_tokens=5, cache_read_tokens=10, cache_write_tokens=2)
    assert events[2]["usage"]["input_tokens"] == 8
    assert events[0]["stream"] != events[2]["stream"]


def test_denied_reservation_never_reaches_provider():
    events = []
    gate = RequestReservations(events.append, lambda: '{"ok":false,"error":"shared budget is reserved"}',
                               {"model": "route-model"}, lambda: None)
    agent = SimpleNamespace(model="route-model", _requested_output_cap_from_api_kwargs=lambda kwargs: 128)
    called = []
    with pytest.raises(PaidRequestStopped, match="shared budget"):
        gate.invoke(lambda: called.append(True), {"messages": [{"role": "user", "content": "bounded task"}]}, agent)
    assert not called
    assert events[0]["type"] == "hermes.reserve"


def test_changed_approved_snapshot_denies_before_request(tmp_path):
    path = tmp_path / "approved.json"
    path.write_text('{"model":"approved"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_snapshot(path, digest)["model"] == "approved"
    path.write_text('{"model":"changed"}')
    with pytest.raises(PaidRequestStopped, match="changed"):
        read_snapshot(path, digest)


@pytest.mark.parametrize("mode", ["transport-error", "wrong-model", "missing-model"])
def test_ambiguous_billing_keeps_reservation_without_settlement(mode):
    events = []
    gate = RequestReservations(events.append, lambda: '{"ok":true,"reservation_id":"held"}',
                               {"model": "approved"}, lambda: None)
    agent = SimpleNamespace(model="approved", provider="custom", api_mode="codex_responses",
                            _requested_output_cap_from_api_kwargs=lambda kwargs: 128)
    calls = []
    def provider_call():
        calls.append(True)
        if mode == "transport-error":
            raise ConnectionError("sensitive provider detail")
        return SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 1},
                               _wire_model="different" if mode == "wrong-model" else None)
    with pytest.raises(PaidRequestStopped) as error:
        gate.invoke(provider_call, {"input": "bounded evidence"}, agent)
    assert len(calls) == 1
    assert not any(event["type"] == "hermes.settle" for event in events)
    assert "sensitive" not in str(error.value)


@pytest.mark.parametrize("kwargs", [{"previous_response_id": "server-state"},
                                   {"conversation": "server-state"},
                                   {"tools": [{"type": "web_search"}]}])
def test_unbounded_provider_context_or_hosted_tools_denied_before_hold(kwargs):
    events = []
    gate = RequestReservations(events.append, lambda: '{}', {"model": "approved"}, lambda: None)
    with pytest.raises(PaidRequestStopped):
        gate.invoke(lambda: pytest.fail("must not call provider"), kwargs, SimpleNamespace(model="approved"))
    assert not events


def test_credential_helper_requires_admission_and_denies_without_leaking(monkeypatch, capsys):
    from tools import delegation_worker
    calls = []
    route = dict(model="test-model", provider="custom", base_url="http://127.0.0.1:1/v1",
                 api_mode="chat_completions", auth_helper="/operator/helper")
    def denied(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="sensitive-helper-output", stderr="sensitive-helper-error")
    monkeypatch.setattr(delegation_worker.subprocess, "run", denied)
    monkeypatch.delenv("PHYSEA_DISPATCH_PRINCIPAL", raising=False)
    with pytest.raises(PaidRequestStopped, match="admitted"):
        resolve_route_runtime(route)
    assert not calls
    for name, value in {"PHYSEA_DISPATCH_PRINCIPAL": "autonomous", "PHYSEA_RUN_BUDGET_ID": "run-test",
                        "PHYSEA_SESSION_SOURCE": "test-session", "PHYSEA_RUN_DEADLINE_AT": "2099-01-01T00:00:00+00:00"}.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(PaidRequestStopped, match="denied") as exc:
        resolve_route_runtime(route)
    assert len(calls) == 1
    assert calls[0][0] == (["/operator/helper"],)
    assert calls[0][1]["stdout"] == subprocess.PIPE
    assert calls[0][1]["stderr"] == subprocess.PIPE
    assert "sensitive" not in str(exc.value) + str(capsys.readouterr())


def _catalog(tmp_path, route=None):
    return dict(schema_version=1, tiers=["director", "senior", "medium", "fast"],
                defaults=dict(senior="careful", medium="careful", fast="careful"),
                routes={"careful": route or {"model": "independent-model-name", "provider": "operator-provider"}})


def test_native_delegate_builds_proxy_without_inference(tmp_path, monkeypatch):
    from tools import delegate_tool
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(_catalog(tmp_path)))
    cfg = {"paid_dispatch": {"enabled": True, "catalog_path": str(path)}}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)
    parent = SimpleNamespace(session_id="director", _active_children=[], enabled_toolsets=["file"],
                             _active_children_lock=threading.Lock(), _current_turn_id="turn-one")
    child = delegate_tool._build_child_agent(0, "Read evidence", "Already answered decision: inspect locally.",
                                             None, None, 5, 1, parent, role="senior")
    assert isinstance(child, ProcessChild)
    assert child.model == "independent-model-name"
    assert child._process is None
    assert child.request["parent_session_id"] == "director"
    assert child.request["parent_turn_id"] == "turn-one"
    assert "second opinion" in child.request["brief"]["review"]
    definitions = delegate_tool._build_dynamic_schema_overrides()
    assert definitions["parameters"]["properties"]["role"]["enum"] == ["senior", "medium", "fast"]
    invalid = json.loads(delegate_tool.delegate_task(goal="Try upwards", role="senior",
                                                    parent_agent=SimpleNamespace(_dispatch_tier="medium", _delegate_depth=2)))
    assert "lower" in str(invalid)


def test_process_wrapper_reports_verified_successor_identity(tmp_path, monkeypatch):
    from tools import delegation_process
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    catalog = tmp_path / "routes.json"
    catalog.write_text(json.dumps(_catalog(tmp_path)))
    parent = SimpleNamespace(session_id="director", enabled_toolsets=["file"],
                             _active_children=[], _active_children_lock=threading.Lock())
    selected_profile = tmp_path / "selected-profile"
    token = set_hermes_home_override(selected_profile)
    try:
        child = ProcessChild(config={"paid_dispatch": {"enabled": True, "catalog_path": str(catalog)}},
            task_index=0, goal="Finish assigned work", context="Preserve source", parent_agent=parent,
            role="fast", max_iterations=3, task_count=1)
    finally:
        reset_hermes_home_override(token)
    receipt = {"budget_id": "original-tree", "deadline_at": "original-deadline"}
    events = [{"type": "hermes.receipt", "receipt": receipt},
              {"type": "hermes.handoff", "receipt": receipt, "source_session_id": child.session_id,
               "successor_session_id": "paid-successor", "handoff_id": "ho-0123456789abcdef"},
              {"type": "hermes.result", "result": {"status": "completed", "summary": "Verified",
                                                     "session_id": "paid-successor"}}]
    process = SimpleNamespace(stdout=[json.dumps(event) + "\n" for event in events], wait=lambda: 0, poll=lambda: 0)
    launched = []
    def launch(*args, **kwargs):
        launched.append(kwargs)
        return process
    monkeypatch.setattr(delegation_process.subprocess, "Popen", launch)
    result = child.run_entry(0)
    assert result["status"] == "completed", result
    assert result["session_id"] == "paid-successor"
    assert result["source_session_id"] == child.session_id
    assert child._session_ref["session_id"] == "paid-successor"
    assert result["receipt"] == receipt
    assert launched[0]["env"]["HERMES_HOME"] == str(selected_profile)
    assert child.directory.is_relative_to(selected_profile)


@pytest.mark.parametrize("credential_source", ["environment", "helper"])
def test_real_hermes_worker_emits_usage_and_result_from_local_provider(tmp_path, credential_source):
    requests = []
    authorizations = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            authorizations.append(self.headers.get("Authorization"))
            payload = {"id": "offline-response", "object": "chat.completion", "created": 1,
                       "model": "offline-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Evidence collected; no open questions."}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 120, "completion_tokens": 10, "total_tokens": 130,
                                 "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10}}}
            if body.get("stream"):
                chunks = [
                    {"id": "offline-response", "object": "chat.completion.chunk", "created": 1, "model": "offline-model",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Evidence collected; no open questions."}, "finish_reason": None}]},
                    {"id": "offline-response", "object": "chat.completion.chunk", "created": 1, "model": "offline-model",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": payload["usage"]},
                ]
                data = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            else:
                data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = dict(model="offline-model", upstream_model="offline-model", provider="custom",
                     base_url=f"http://127.0.0.1:{server.server_port}/v1", api_mode="chat_completions",
                     api_key_env="OFFLINE_FIXTURE_KEY", tiers=["fast"])
        secret = "offline-fixture-secret-never-logged-00001"
        if credential_source == "helper":
            helper = tmp_path / "fixture-auth-helper"
            helper.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                              "assert os.environ['PHYSEA_RUN_BUDGET_ID'] == 'run-offline'\n"
                              "assert os.environ['PHYSEA_SESSION_SOURCE'] == 'offline-director'\n"
                              "print(" + repr(secret) + ")\nprint('private-helper-stderr', file=sys.stderr)\n")
            helper.chmod(0o700)
            route.pop("api_key_env")
            route["auth_helper"] = str(helper)
        catalog = tmp_path / "routes.json"
        catalog.write_text(json.dumps(_catalog(tmp_path, route)))
        request = tmp_path / "request.json"
        request.write_text(json.dumps(dict(schema_version=1, tier="fast", parent_tier="director", route_id="careful",
                         goal="Collect the existing evidence", context="No changes or network research needed.",
                         workdir=str(tmp_path), depth=1, toolsets=["file"], max_iterations=2,
                         session_id="offline-worker", subagent_id="sa-offline", parent_session_id="offline-director",
                         parent_subagent_id=None)))
        env = dict(os.environ, HERMES_HOME=str(tmp_path / "profile"), OFFLINE_FIXTURE_KEY=secret,
                   PHYSEA_DISPATCH_PRINCIPAL="autonomous", PHYSEA_RUN_BUDGET_ID="run-offline",
                   PHYSEA_SESSION_SOURCE="offline-director",
                   PHYSEA_RUN_DEADLINE_AT="2099-01-01T00:00:00+00:00", PHYSEA_RUN_MAX_RUNTIME_SECONDS="900",
                   PHYSEA_RUN_MAX_COST_USD="5.00", PYTHONPATH=str(Path(__file__).parents[2]),
                   PHYSEA_WORKER_CATALOG_SHA256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                   PHYSEA_WORKER_REQUEST_SHA256=hashlib.sha256(request.read_bytes()).hexdigest())
        result = subprocess.run([sys.executable, "-m", "tools.delegation_worker", "--request", str(request),
                                 "--catalog", str(catalog)], env=env, cwd=tmp_path,
                                input='{"ok":true,"reservation_id":"offline-request"}\n{"ok":true}\n',
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr[-3000:] + result.stdout[-3000:]
        assert secret not in result.stdout + result.stderr
        assert "private-helper-stderr" not in result.stdout + result.stderr
        assert "Bearer " + secret in authorizations
        events = [json.loads(line) for line in result.stdout.splitlines()]
        usage = next(e for e in events if e["type"] == "hermes.usage")
        assert usage["usage"] == dict(input_tokens=30, output_tokens=10, cache_read_tokens=80, cache_write_tokens=10)
        assert next(e for e in events if e["type"] == "hermes.reserve")["output_bound"] == 4096
        assert next(e for e in events if e["type"] == "hermes.settle")["reservation_id"] == "offline-request"
        assert next(e for e in events if e["type"] == "hermes.result")["result"]["status"] == "completed"
        assert "run-offline" in json.dumps(requests)
        assert "$5.00" in json.dumps(requests)
        # The local-provider metadata probe is not a model invocation.
        assert sum("messages" in request or "input" in request for request in requests) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
