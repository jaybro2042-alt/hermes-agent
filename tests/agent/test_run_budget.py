"""Tests for the optional per-run ceilings — wall-clock (agent.run_budget_seconds /
--run-budget) and estimated spend (agent.run_budget_usd / --run-budget-usd).

Covers:

1. Stale-timeout deadline scaling in
   ``run_agent.py:AIAgent._compute_non_stream_stale_timeout``:
   - an active run budget CAPS an implicit stale timeout (default 90s and
     reasoning floors like deepseek's 600s) at ``max(60, remaining * 0.5)``;
   - the cap never RAISES the timeout above what it would otherwise be;
   - explicit user configuration (provider ``stale_timeout_seconds`` or the
     ``HERMES_API_CALL_STALE_TIMEOUT`` env var) always wins untouched;
   - no budget => completely unchanged behavior.

2. One-time-ness of the 80% wrap-up notice injection in
   ``agent.conversation_loop._maybe_inject_run_budget_wrapup``:
   - fires once (latched), not repeatedly, on EITHER ceiling;
   - never fires when no budget is set or before the 80% threshold;
   - appended to the newest tool message (cache-safe /steer channel), no
     synthetic user message.

3. Normalization of the config/CLI value
   (``agent.agent_init._normalize_run_budget_seconds``): dormant on
   null/invalid/non-positive input.

4. The hard ceilings in ``agent.run_budget`` and their enforcement in
   ``agent.turn_iteration_prep.begin_iteration``:
   - spend is measured as a delta from this run's stamp, not session-cumulative;
   - a crossed ceiling breaks the loop BEFORE the next provider call and does not
     grow ``api_call_count``;
   - a subagent inherits the REMAINDER of both ceilings, never an "off" 0.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path, monkeypatch, config_body: str = "", **overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, config_body)

    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai",
        api_key="sk-dummy",
        base_url="https://api.openai.com/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── normalization ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    (None, None),
    (0, None),
    (-5, None),
    ("abc", None),
    (True, None),   # YAML `true` must not become a 1-second budget
    (False, None),
    (float("nan"), None),
    (900, 900.0),
    ("850", 850.0),
    (0.5, 0.5),
])
def test_normalize_run_budget_seconds(raw, expected):
    from agent.agent_init import _normalize_run_budget_seconds
    assert _normalize_run_budget_seconds(raw) == expected


# ── constructor / config plumbing ─────────────────────────────────────────


def test_no_budget_by_default(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path, monkeypatch)
    assert agent.run_budget_seconds is None
    assert agent._run_budget_started_at is None
    assert agent._run_budget_wrapup_injected is False


def test_constructor_arg_sets_budget(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    assert agent.run_budget_seconds == 900.0


def test_config_key_sets_budget(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path, monkeypatch,
        config_body="agent:\n  run_budget_seconds: 750\n",
    )
    assert agent.run_budget_seconds == 750.0


def test_constructor_arg_wins_over_config(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path, monkeypatch,
        config_body="agent:\n  run_budget_seconds: 750\n",
        run_budget_seconds=900,
    )
    assert agent.run_budget_seconds == 900.0


# ── stale-timeout deadline scaling ─────────────────────────────────────────


def test_no_budget_stale_timeout_unchanged(monkeypatch, tmp_path):
    """Without a run budget the implicit 90s default is untouched."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 90.0


def test_active_budget_caps_implicit_reasoning_floor(monkeypatch, tmp_path):
    """deepseek-v4-pro's 600s implicit floor yields to a tighter deadline cap.

    900s budget, 800s elapsed -> remaining 100s -> cap = max(60, 50) = 60s.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    # Sanity: implicit reasoning floor is 600s without a running clock.
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 600.0 and implicit is False

    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 60.0


def test_active_budget_cap_half_remaining(monkeypatch, tmp_path):
    """Cap is remaining * 0.5 when that exceeds the 60s floor."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = time.time() - 100  # remaining ~800 -> cap ~400
    timeout = agent._compute_non_stream_stale_timeout({"input": "hi"})
    assert 395.0 <= timeout <= 400.0


def test_active_budget_never_raises_timeout(monkeypatch, tmp_path):
    """The deadline cap NEVER loosens an already-tighter implicit timeout.

    90s default with lots of remaining budget (cap would be ~445s) -> stays 90s.
    """
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 10
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 90.0


def test_explicit_provider_config_wins_over_budget_cap(monkeypatch, tmp_path):
    """Explicit stale_timeout_seconds is never capped by the run budget."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: 1800.0)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


def test_explicit_env_var_wins_over_budget_cap(monkeypatch, tmp_path):
    """HERMES_API_CALL_STALE_TIMEOUT is explicit config — never capped."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "1200")
    agent._run_budget_started_at = time.time() - 800
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1200.0


def test_budget_without_started_clock_is_inert(monkeypatch, tmp_path):
    """A configured budget with no running turn clock changes nothing."""
    import run_agent
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", lambda *a, **k: None)
    agent = _make_agent(
        tmp_path, monkeypatch,
        model="deepseek/deepseek-v4-pro",
        run_budget_seconds=900,
    )
    agent._run_budget_started_at = None
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 600.0


# ── wrap-up injection one-time-ness ────────────────────────────────────────


class _StubAgent:
    def __init__(self, budget=None, started=None, usd_budget=None, cost_at_start=None, spent=0.0):
        self.run_budget_seconds = budget
        self._run_budget_started_at = started
        self._run_budget_wrapup_injected = False
        self.run_budget_usd = usd_budget
        self._run_budget_cost_at_start = cost_at_start
        self.session_estimated_cost_usd = spent


def _tool_messages():
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result one"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "result two"},
    ]


def test_wrapup_not_injected_when_unset():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=None, started=time.time() - 10_000)
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert messages == _tool_messages()


def test_wrapup_not_injected_before_threshold():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=900, started=time.time() - 100)  # 11% elapsed
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert agent._run_budget_wrapup_injected is False


def test_wrapup_injected_once_after_threshold():
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)  # 89% elapsed
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    assert agent._run_budget_wrapup_injected is True
    # Appended to the NEWEST tool message; earlier messages untouched.
    assert RUN_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]
    assert messages[-1]["content"].startswith("result two")
    assert messages[2]["content"] == "result one"
    # No synthetic user message was inserted.
    assert [m["role"] for m in messages] == [
        "user", "assistant", "tool", "assistant", "tool",
    ]

    # Second call: latched, no re-injection.
    snapshot = [dict(m) for m in messages]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert messages == snapshot
    assert messages[-1]["content"].count(RUN_BUDGET_WRAPUP_NOTICE) == 1


def test_wrapup_retries_when_no_tool_message_yet():
    """First iteration (no tool results) can't inject; the latch stays open
    so the next iteration with a tool result delivers the notice."""
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)
    messages = [{"role": "user", "content": "do the task"}]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False
    assert agent._run_budget_wrapup_injected is False

    messages += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    assert RUN_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]


def test_wrapup_not_injected_without_turn_clock():
    from agent.conversation_loop import _maybe_inject_run_budget_wrapup
    agent = _StubAgent(budget=900, started=None)
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is False


def test_wrapup_multimodal_tool_content():
    """Content-blocks tool results get a text block appended, not clobbered."""
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )
    agent = _StubAgent(budget=900, started=time.time() - 800)
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1",
         "content": [{"type": "text", "text": "block"}]},
    ]
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    blocks = messages[-1]["content"]
    assert blocks[0] == {"type": "text", "text": "block"}
    assert blocks[-1] == {"type": "text", "text": RUN_BUDGET_WRAPUP_NOTICE}


# ── turn clock stamping ────────────────────────────────────────────────────


def test_turn_clock_stamped_only_with_budget(monkeypatch, tmp_path):
    """prepare-turn stamps each ceiling iff that ceiling is configured."""
    from agent.run_budget import stamp_run

    agent_with = _make_agent(tmp_path, monkeypatch, run_budget_seconds=900)
    agent_without = _make_agent(tmp_path, monkeypatch)
    for agent in (agent_with, agent_without):
        stamp_run(agent)

    assert agent_with._run_budget_started_at is not None
    assert agent_with._run_budget_cost_at_start is None  # no spend ceiling configured
    assert agent_without._run_budget_started_at is None
    assert agent_without._run_budget_cost_at_start is None


# ── spend ceiling: stamping, deltas, breach ────────────────────────────────


def test_spend_ceiling_stamped_and_measured_as_a_delta(monkeypatch, tmp_path):
    """A session that already spent money starts this run at zero spent."""
    from agent.run_budget import spent_usd, stamp_run

    agent = _make_agent(tmp_path, monkeypatch, run_budget_usd=5)
    agent.session_estimated_cost_usd = 12.5  # earlier turns of a long-lived session
    stamp_run(agent)
    assert agent._run_budget_cost_at_start == 12.5
    assert spent_usd(agent) == 0.0

    agent.session_estimated_cost_usd = 13.25
    assert spent_usd(agent) == pytest.approx(0.75)


def test_spend_ceiling_config_and_constructor(monkeypatch, tmp_path):
    from_config = _make_agent(
        tmp_path, monkeypatch, config_body="agent:\n  run_budget_usd: 2.5\n"
    )
    assert from_config.run_budget_usd == 2.5
    override = _make_agent(
        tmp_path, monkeypatch,
        config_body="agent:\n  run_budget_usd: 2.5\n", run_budget_usd=9,
    )
    assert override.run_budget_usd == 9.0
    assert _make_agent(tmp_path, monkeypatch).run_budget_usd is None


@pytest.mark.parametrize("agent_kwargs,expected_kind", [
    (dict(budget=900, started=time.time() - 901), "time"),
    (dict(usd_budget=5, cost_at_start=0.0, spent=5.0), "cost"),
    (dict(usd_budget=5, cost_at_start=10.0, spent=15.5), "cost"),
])
def test_check_reports_the_crossed_ceiling(agent_kwargs, expected_kind):
    from agent.run_budget import check

    breach = check(_StubAgent(**agent_kwargs))
    assert breach is not None and breach.kind == expected_kind
    assert breach.exit_reason == f"run_budget_exhausted({expected_kind})"


@pytest.mark.parametrize("agent_kwargs", [
    dict(),                                                   # both ceilings off
    dict(budget=900, started=time.time() - 10),               # 1% elapsed
    dict(usd_budget=5, cost_at_start=0.0, spent=4.99),        # under the spend ceiling
    dict(usd_budget=5, cost_at_start=None, spent=99.0),       # configured but unstamped
])
def test_check_is_silent_below_the_ceiling(agent_kwargs):
    from agent.run_budget import check

    assert check(_StubAgent(**agent_kwargs)) is None


def test_unpriced_route_never_moves_the_spend_ceiling():
    """Providers with no pricing leave session cost at 0, so the ceiling stays dormant."""
    from agent.run_budget import check, spent_usd

    agent = _StubAgent(usd_budget=0.01, cost_at_start=0.0, spent=0.0)
    assert spent_usd(agent) == 0.0
    assert check(agent) is None


def test_wrapup_fires_on_the_spend_ceiling():
    """The 80% notice is ceiling-agnostic: spend alone can trigger it."""
    from agent.conversation_loop import (
        RUN_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_run_budget_wrapup,
    )

    agent = _StubAgent(usd_budget=5, cost_at_start=1.0, spent=5.25)  # 85% of $5
    messages = _tool_messages()
    assert _maybe_inject_run_budget_wrapup(agent, messages) is True
    assert RUN_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]


# ── enforcement at the iteration boundary ──────────────────────────────────


class _LoopStubAgent:
    """Minimum surface ``begin_iteration`` touches, plus the ceiling attributes."""

    def __init__(self, **budget_state):
        self.quiet_mode = True
        self.session_id = "sess-1"
        self._interrupt_requested = False
        self._budget_grace_call = False
        self._review_input_token_budget = None
        self.session_input_tokens = 0
        self._api_call_count = 0
        self.statuses: list = []
        self.touched: list = []
        from agent.iteration_budget import IterationBudget
        self.iteration_budget = IterationBudget(100)
        self._checkpoint_mgr = type("_Ckpt", (), {"new_turn": lambda self: None})()
        for name, value in dict(
            run_budget_seconds=None, _run_budget_started_at=None,
            run_budget_usd=None, _run_budget_cost_at_start=None,
            session_estimated_cost_usd=0.0,
        ).items():
            setattr(self, name, value)
        for name, value in budget_state.items():
            setattr(self, name, value)

    def _drain_pending_redirect(self):
        return None

    def _touch_activity(self, description):
        self.touched.append(description)

    def _emit_status(self, text):
        self.statuses.append(text)

    def _safe_print(self, text):  # pragma: no cover - quiet_mode keeps this unused
        pass


def _begin(agent):
    from agent.turn_iteration_prep import begin_iteration

    return begin_iteration(
        agent, messages=[], conversation_history=[], original_user_message="go",
        api_call_count=7, interrupted=False, _turn_exit_reason="unknown",
    )


def test_iteration_proceeds_under_the_ceiling():
    verdict = _begin(_LoopStubAgent(
        run_budget_usd=5, _run_budget_cost_at_start=0.0, session_estimated_cost_usd=1.0,
    ))
    assert verdict.action == "fallthrough"
    assert verdict.api_call_count == 8
    assert verdict._turn_exit_reason == "unknown"


@pytest.mark.parametrize("state,kind", [
    (dict(run_budget_seconds=60, _run_budget_started_at=time.time() - 61), "time"),
    (dict(run_budget_usd=5, _run_budget_cost_at_start=0.0,
          session_estimated_cost_usd=5.01), "cost"),
])
def test_crossed_ceiling_breaks_before_the_next_call(state, kind):
    """The refused call is the NEXT one: the counter does not grow and no activity
    is stamped for a request that never happens."""
    agent = _LoopStubAgent(**state)
    verdict = _begin(agent)
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == f"run_budget_exhausted({kind})"
    assert verdict.api_call_count == 7
    assert agent._api_call_count == 0
    assert agent.touched == []
    assert agent.statuses and "budget exhausted" in agent.statuses[0]
    # The iteration budget is untouched — this exit is not an iteration exhaustion.
    assert agent.iteration_budget.used == 0


def test_interrupt_still_wins_over_the_ceiling():
    agent = _LoopStubAgent(run_budget_seconds=60, _run_budget_started_at=time.time() - 999)
    agent._interrupt_requested = True
    verdict = _begin(agent)
    assert verdict.action == "break"
    assert verdict._turn_exit_reason == "interrupted_by_user"


def test_exit_reason_has_a_user_facing_explanation():
    from agent.turn_explainers import TurnExplainersMixin

    explanation = TurnExplainersMixin._format_turn_completion_explanation(
        "run_budget_exhausted(cost)"
    )
    assert "ceiling" in explanation
    assert "run_budget_usd" in explanation


# ── subagent inheritance ───────────────────────────────────────────────────


def test_child_inherits_the_remainder_of_both_ceilings():
    from agent.run_budget import child_budget_kwargs

    parent = _StubAgent(
        budget=900, started=time.time() - 300,
        usd_budget=5, cost_at_start=1.0, spent=3.0,
    )
    kwargs = child_budget_kwargs(parent)
    assert kwargs["run_budget_seconds"] == pytest.approx(600, abs=2)
    assert kwargs["run_budget_usd"] == pytest.approx(3.0)


def test_child_of_an_unbudgeted_parent_gets_nothing():
    from agent.run_budget import child_budget_kwargs

    assert child_budget_kwargs(_StubAgent()) == {}


def test_depleted_ceiling_is_never_handed_down_as_unlimited():
    """0 normalizes to "off", so a spent-out parent must pass a floor, not zero."""
    from agent.run_budget import child_budget_kwargs, normalize_budget

    parent = _StubAgent(
        budget=60, started=time.time() - 600, usd_budget=5, cost_at_start=0.0, spent=99.0
    )
    kwargs = child_budget_kwargs(parent)
    assert all(value > 0 for value in kwargs.values())
    assert all(normalize_budget(value) is not None for value in kwargs.values())
