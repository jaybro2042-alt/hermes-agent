"""Per-run ceilings: wall-clock time and spend.

``agent.run_budget_seconds`` shipped as a purely *advisory* signal — a one-time wrap-up
notice at 80% elapsed and a cap on implicit stale timeouts — so a run that keeps making
progress never stops on its own, and nothing watched spend at all. A single unattended
run could therefore bill for hours: the iteration budget is unlimited by default, the
clock only nudged, and the only cost signal in the loop was the empty-response guard.

This module is the enforcing half. Once elapsed time or run spend crosses its ceiling the
turn stops BEFORE the next provider call, with no extra summary request (a ceiling that
bills one more call to announce itself is not a ceiling).

Invariants:

* Both ceilings are off unless configured — unset means no clock read, no cost read.
* Stamped per run by ``turn_context.prepare_turn``; spend is a delta from that stamp, so
  earlier turns of a long-lived session never pre-exhaust this one.
* Checked BETWEEN iterations, never mid-request, so the crossing request completes and
  its usage/transcript writes land — the same contract as the review input budget.
* A subagent inherits what is LEFT of the parent's ceilings, so a delegated fan-out is
  bounded by the run that spawned it (the parent's own check only fires once the child's
  cost has folded back in).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Fraction of a ceiling at which the model gets its one-time wrap-up notice.
WRAPUP_FRACTION = 0.8

# A crossed ceiling must stay a ceiling when it is handed to a subagent: 0 normalizes to
# "off", so a depleted budget is passed down as this floor instead.
_DEPLETED_FLOOR = 0.01

__all__ = [
    "WRAPUP_FRACTION", "RunBudgetBreach", "check", "child_budget_kwargs", "elapsed_seconds",
    "normalize_budget", "remaining_seconds", "remaining_usd", "spent_usd", "stamp_run",
    "used_fraction",
]


def normalize_budget(value: Any) -> Optional[float]:
    """Positive float, or None when the ceiling is off. ``bool`` is rejected: YAML
    ``true`` must not become a 1-second / $1 budget."""
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None  # NaN compares False → None


def _number(value: Any) -> Optional[float]:
    """``float`` for a real number, else None (``bool`` and doubles are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _limit(agent: Any, attr: str) -> Optional[float]:
    return normalize_budget(getattr(agent, attr, None))


def stamp_run(agent: Any) -> None:
    """Start this run's clock and spend baseline; ``None`` on either means that ceiling
    is off, which is what keeps the feature dormant when unconfigured."""
    agent._run_budget_started_at = time.time() if _limit(agent, "run_budget_seconds") else None
    agent._run_budget_cost_at_start = (
        _number(getattr(agent, "session_estimated_cost_usd", 0.0)) or 0.0
        if _limit(agent, "run_budget_usd") else None
    )


def elapsed_seconds(agent: Any) -> Optional[float]:
    """Seconds since this run was stamped, or None when the clock is not running."""
    started = _number(getattr(agent, "_run_budget_started_at", None))
    return None if started is None else max(0.0, time.time() - started)


def spent_usd(agent: Any) -> Optional[float]:
    """Estimated spend since this run was stamped, or None when spend is not watched.

    Unpriced routes (``session_cost_status`` unknown, subscription/included providers)
    simply never move the counter, so the ceiling stays dormant rather than guessing.
    """
    baseline = _number(getattr(agent, "_run_budget_cost_at_start", None))
    if baseline is None:
        return None
    return max(0.0, (_number(getattr(agent, "session_estimated_cost_usd", 0.0)) or 0.0) - baseline)


@dataclass(frozen=True)
class RunBudgetBreach:
    """A ceiling this run has crossed. ``kind`` is ``"time"`` or ``"cost"``."""

    kind: str
    used: float
    limit: float

    @property
    def exit_reason(self) -> str:
        return f"run_budget_exhausted({self.kind})"

    @property
    def notice(self) -> str:
        if self.kind == "cost":
            return (
                f"Run cost budget exhausted (${self.used:.2f} of ${self.limit:.2f}) — "
                "stopping before the next provider call."
            )
        return (
            f"Run time budget exhausted ({self.used:.0f}s of {self.limit:.0f}s) — "
            "stopping before the next provider call."
        )


def _usage(agent: Any) -> tuple[tuple[str, Optional[float], Optional[float]], ...]:
    """``(kind, used, limit)`` for both ceilings; time first — it is the cheaper read and
    the one an external deadline is usually set against."""
    return (
        ("time", elapsed_seconds(agent), _limit(agent, "run_budget_seconds")),
        ("cost", spent_usd(agent), _limit(agent, "run_budget_usd")),
    )


def check(agent: Any) -> Optional[RunBudgetBreach]:
    """The first ceiling this run has crossed, or None."""
    for kind, used, limit in _usage(agent):
        if used is not None and limit is not None and used >= limit:
            return RunBudgetBreach(kind=kind, used=used, limit=limit)
    return None


def used_fraction(agent: Any) -> float:
    """How far into the tightest active ceiling this run is; 0.0 when none is active."""
    fractions = [
        used / limit for _kind, used, limit in _usage(agent)
        if used is not None and limit is not None
    ]
    return max(fractions) if fractions else 0.0


def remaining_seconds(agent: Any) -> Optional[float]:
    limit = _limit(agent, "run_budget_seconds")
    return None if limit is None else max(0.0, limit - (elapsed_seconds(agent) or 0.0))


def remaining_usd(agent: Any) -> Optional[float]:
    limit = _limit(agent, "run_budget_usd")
    return None if limit is None else max(0.0, limit - (spent_usd(agent) or 0.0))


def child_budget_kwargs(parent: Any) -> Dict[str, float]:
    """``AIAgent`` kwargs handing a subagent what is LEFT of the parent's ceilings.

    Empty when the parent has none. Siblings running in parallel each get the same
    remainder — the bound is per child, not a shared pool — but the parent stops at its
    next iteration once their cost folds in.
    """
    return {
        key: max(remaining, _DEPLETED_FLOOR)
        for key, remaining in (
            ("run_budget_seconds", remaining_seconds(parent)),
            ("run_budget_usd", remaining_usd(parent)),
        )
        if remaining is not None
    }
