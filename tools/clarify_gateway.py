"""Gateway-side clarify primitive (blocking event-based queue).

The ``clarify`` tool needs to ask the user a question and block the agent
thread until they respond.  In CLI mode this is trivial — ``input()`` is
synchronous.  In gateway mode the agent runs on a worker thread while the
event loop handles the user's reply, so we need a thread-safe primitive
that:

  * stores a pending clarify request (with a generated ``clarify_id``),
  * blocks the agent thread on an ``Event``,
  * resolves the wait when the gateway's button-callback or text-intercept
    fires ``resolve_gateway_clarify(clarify_id, response)``,
  * supports timeouts so a user who never responds does NOT hang the agent
    thread forever (which would also pin the gateway's running-agent guard).

State is module-level (same shape as ``tools.approval``) so platform
adapters can call ``resolve_gateway_clarify`` without holding a back-
reference to the ``GatewayRunner`` instance.

Two delivery paths from the adapter:

  1. **Button UI** — adapters override ``send_clarify`` to render inline
     buttons (e.g. Telegram ``InlineKeyboardMarkup``).  The button
     callback resolves with the chosen string.  A final "Other (type
     answer)" button enters text-capture mode for free-form responses.

  2. **Text fallback** — adapters without rich UI render a numbered list.
     The user replies with a number ("2") or with free text; the gateway's
     ``_handle_message`` intercepts the reply and resolves directly.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


STATE_CREATED = "created"
STATE_TRANSPORT_ACCEPTED = "transport_accepted"
STATE_EVENT_CONFIRMED = "event_confirmed"
STATE_ANSWERED = "answered"
STATE_TIMED_OUT = "timed_out"
STATE_FAILED = "failed"

_TERMINAL_STATES = {STATE_ANSWERED, STATE_TIMED_OUT, STATE_FAILED}
_RECOVERABLE_STATES = {
    STATE_CREATED,
    STATE_TRANSPORT_ACCEPTED,
    STATE_EVENT_CONFIRMED,
}
_TEXT_DELIVERY_CHANNELS = {"numbered_text", "open_text", "text"}


class ClarifyPersistenceError(RuntimeError):
    """A durable clarification transition could not be recorded."""

    def __init__(self, operation: str, cause: Exception):
        self.operation = operation
        self.cause = cause
        super().__init__(f"clarify persistence failed during {operation}: {cause}")


# =========================================================================
# Module-level state
# =========================================================================

@dataclass
class _ClarifyEntry:
    """One pending clarify request inside a gateway session."""
    clarify_id: str
    session_key: str
    question: str
    choices: Optional[List[str]]
    multi_select: bool = False
    platform: str = ""
    chat_id: str = ""
    thread_id: Optional[str] = None
    idempotency_key: str = ""
    expires_at: Optional[float] = None
    state: str = STATE_CREATED
    delivery_channel: Optional[str] = None
    poll_guid: Optional[str] = None
    fallback_attempted: bool = False
    recovered: bool = False
    home_key: str = ""
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None
    awaiting_text: bool = False

    def signature(self) -> Dict[str, object]:
        return {
            "clarify_id": self.clarify_id,
            "session_key": self.session_key,
            "question": self.question,
            "choices": list(self.choices) if self.choices else None,
            "multi_select": bool(self.multi_select),
        }

    @property
    def delivery_confirmed(self) -> bool:
        return self.state == STATE_EVENT_CONFIRMED

    @property
    def needs_recovery_fallback(self) -> bool:
        if not self.recovered or self.fallback_attempted:
            return False
        if self.state in {STATE_CREATED, STATE_TRANSPORT_ACCEPTED}:
            return True
        if self.state != STATE_EVENT_CONFIRMED:
            return False
        if self.delivery_channel in _TEXT_DELIVERY_CHANNELS:
            return False
        return not (
            self.platform == "photon"
            and self.delivery_channel == "native_poll"
            and bool(self.poll_guid)
        )


_lock = threading.RLock()
# clarify_id → _ClarifyEntry  (primary lookup for button callbacks)
_entries: Dict[str, _ClarifyEntry] = {}
# session_key → list[clarify_id]  (FIFO; for text-fallback intercept and session cleanup)
_session_index: Dict[str, List[str]] = {}


# =========================================================================
# Durable state
# =========================================================================

_DB_LOCK = threading.RLock()


def _db_path():
    return get_hermes_home() / "state.db"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (clarify_gateway)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS clarify_obligations (
            clarify_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            question TEXT NOT NULL,
            choices_json TEXT,
            multi_select INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL,
            delivery_channel TEXT,
            poll_guid TEXT,
            idempotency_key TEXT NOT NULL,
            awaiting_text INTEGER NOT NULL DEFAULT 0,
            fallback_attempted INTEGER NOT NULL DEFAULT 0,
            response TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        )"""
    )


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        _initialize_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid, started_at) -> bool:
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_started_at = get_process_start_time(pid)
    except Exception:
        current_started_at = None
    if current_started_at is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_started_at) == int(started_at)
    except (TypeError, ValueError):
        return True


def _persist_new(entry: _ClarifyEntry) -> None:
    now = time.time()
    owner_pid, owner_started_at = _owner_stamp()
    try:
        with _DB_LOCK, _transaction() as conn:
            conn.execute(
                """INSERT INTO clarify_obligations
                   (clarify_id, session_key, platform, chat_id, thread_id,
                    question, choices_json, multi_select, state,
                    delivery_channel, poll_guid, idempotency_key,
                    awaiting_text, fallback_attempted, response,
                    created_at, updated_at, expires_at, owner_pid,
                    owner_started_at, last_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.clarify_id,
                    entry.session_key,
                    entry.platform,
                    entry.chat_id,
                    entry.thread_id,
                    entry.question,
                    json.dumps(entry.choices, ensure_ascii=False)
                    if entry.choices
                    else None,
                    int(entry.multi_select),
                    entry.state,
                    entry.delivery_channel,
                    entry.poll_guid,
                    entry.idempotency_key,
                    int(entry.awaiting_text),
                    int(entry.fallback_attempted),
                    entry.response,
                    now,
                    now,
                    entry.expires_at,
                    owner_pid,
                    owner_started_at,
                    None,
                ),
            )
    except Exception as exc:
        raise ClarifyPersistenceError("register", exc) from exc


def _persist_transition(
    clarify_id: str,
    *,
    expected_states: Optional[set[str]] = None,
    state: Optional[str] = None,
    delivery_channel: Optional[str] = None,
    poll_guid: Optional[str] = None,
    awaiting_text: Optional[bool] = None,
    fallback_attempted: Optional[bool] = None,
    response: Optional[str] = None,
    last_error: Optional[str] = None,
) -> bool:
    assignments = ["updated_at=?"]
    values: List[object] = [time.time()]
    for column, value in (
        ("state", state),
        ("delivery_channel", delivery_channel),
        ("poll_guid", poll_guid),
        ("awaiting_text", None if awaiting_text is None else int(awaiting_text)),
        (
            "fallback_attempted",
            None if fallback_attempted is None else int(fallback_attempted),
        ),
        ("response", response),
        ("last_error", last_error),
    ):
        if value is not None:
            assignments.append(f"{column}=?")
            values.append(value)
    where = ["clarify_id=?"]
    values.append(clarify_id)
    if expected_states:
        ordered_states = sorted(expected_states)
        where.append(f"state IN ({', '.join('?' for _ in ordered_states)})")
        values.extend(ordered_states)
    try:
        with _DB_LOCK, _transaction() as conn:
            result = conn.execute(
                f"UPDATE clarify_obligations SET {', '.join(assignments)} "
                f"WHERE {' AND '.join(where)}",
                values,
            )
            return result.rowcount > 0
    except Exception as exc:
        raise ClarifyPersistenceError("transition", exc) from exc


def _row_to_entry(row: sqlite3.Row) -> _ClarifyEntry:
    choices = None
    if row["choices_json"]:
        parsed = json.loads(row["choices_json"])
        if isinstance(parsed, list):
            choices = [str(choice) for choice in parsed]
    return _ClarifyEntry(
        clarify_id=str(row["clarify_id"]),
        session_key=str(row["session_key"]),
        question=str(row["question"]),
        choices=choices,
        multi_select=bool(row["multi_select"]),
        platform=str(row["platform"] or ""),
        chat_id=str(row["chat_id"] or ""),
        thread_id=row["thread_id"],
        idempotency_key=str(row["idempotency_key"] or ""),
        expires_at=row["expires_at"],
        state=str(row["state"]),
        delivery_channel=row["delivery_channel"],
        poll_guid=row["poll_guid"],
        fallback_attempted=bool(row["fallback_attempted"]),
        recovered=True,
        home_key=str(get_hermes_home()),
        response=row["response"],
        awaiting_text=bool(row["awaiting_text"]),
    )


def _refresh_entry_from_durable(entry: _ClarifyEntry) -> bool:
    """Refresh one in-memory entry after a cross-process CAS miss."""
    try:
        with _DB_LOCK, _transaction() as conn:
            row = conn.execute(
                "SELECT * FROM clarify_obligations WHERE clarify_id=?",
                (entry.clarify_id,),
            ).fetchone()
    except Exception:
        logger.debug(
            "Failed to refresh clarify state for %s",
            entry.clarify_id,
            exc_info=True,
        )
        return False
    if row is None:
        return False
    entry.state = str(row["state"])
    entry.delivery_channel = row["delivery_channel"]
    entry.poll_guid = row["poll_guid"]
    entry.awaiting_text = bool(row["awaiting_text"])
    entry.fallback_attempted = bool(row["fallback_attempted"])
    entry.response = row["response"]
    if entry.state in _TERMINAL_STATES:
        entry.awaiting_text = False
        entry.event.set()
    return True


def recover_pending(*, force: bool = False) -> List[_ClarifyEntry]:
    """Claim recoverable clarifications left by a dead gateway process.

    Confirmed prompts resume text interception. Prompts whose native transport
    was only accepted are returned for one numbered-text recovery fallback.
    A crash during that fallback is ambiguous, so it is failed rather than sent
    again and risking an invisible duplicate loop.
    """
    now = time.time()
    owner_pid, owner_started_at = _owner_stamp()
    recovered: List[_ClarifyEntry] = []
    try:
        with _DB_LOCK, _transaction() as conn:
            rows = conn.execute(
                """SELECT * FROM clarify_obligations
                   WHERE state IN (?, ?, ?)
                   ORDER BY created_at ASC""",
                tuple(_RECOVERABLE_STATES),
            ).fetchall()
            for row in rows:
                clarify_id = str(row["clarify_id"])
                expires_at = row["expires_at"]
                if expires_at is not None and float(expires_at) <= now:
                    conn.execute(
                        """UPDATE clarify_obligations
                           SET state=?, awaiting_text=0, updated_at=?, last_error=?
                           WHERE clarify_id=? AND state=?""",
                        (
                            STATE_TIMED_OUT,
                            now,
                            "clarification expired",
                            clarify_id,
                            row["state"],
                        ),
                    )
                    continue
                if not force and _owner_alive(row["owner_pid"], row["owner_started_at"]):
                    continue
                if bool(row["fallback_attempted"]) and row["state"] != STATE_EVENT_CONFIRMED:
                    conn.execute(
                        """UPDATE clarify_obligations
                           SET state=?, awaiting_text=0, updated_at=?, last_error=?
                           WHERE clarify_id=? AND state=?""",
                        (
                            STATE_FAILED,
                            now,
                            "gateway restarted during ambiguous text fallback",
                            clarify_id,
                            row["state"],
                        ),
                    )
                    continue
                claimed = conn.execute(
                    """UPDATE clarify_obligations
                       SET owner_pid=?, owner_started_at=?, updated_at=?
                       WHERE clarify_id=? AND state=?
                         AND owner_pid IS ? AND owner_started_at IS ?""",
                    (
                        owner_pid,
                        owner_started_at,
                        now,
                        clarify_id,
                        row["state"],
                        row["owner_pid"],
                        row["owner_started_at"],
                    ),
                )
                if claimed.rowcount <= 0:
                    continue
                entry = _row_to_entry(row)
                if entry.state != STATE_EVENT_CONFIRMED:
                    entry.awaiting_text = False
                recovered.append(entry)
    except Exception as exc:
        raise ClarifyPersistenceError("recover", exc) from exc

    with _lock:
        for entry in recovered:
            if entry.clarify_id in _entries:
                continue
            _entries[entry.clarify_id] = entry
            _session_index.setdefault(entry.session_key, []).append(entry.clarify_id)
    return recovered


# =========================================================================
# Public API — agent-thread side
# =========================================================================

def register(
    clarify_id: str,
    session_key: str,
    question: str,
    choices: Optional[List[str]],
    multi_select: bool = False,
    *,
    platform: str = "",
    chat_id: str = "",
    thread_id: Optional[str] = None,
    expires_at: Optional[float] = None,
    idempotency_key: Optional[str] = None,
) -> _ClarifyEntry:
    """Register a pending clarify request and return the entry.

    The caller (gateway clarify_callback) will then send the prompt to the
    user and block on ``wait_for_response(clarify_id, timeout)``.
    """
    entry = _ClarifyEntry(
        clarify_id=clarify_id,
        session_key=session_key,
        question=question,
        choices=list(choices) if choices else None,
        multi_select=bool(multi_select) and bool(choices),
        platform=str(platform or ""),
        chat_id=str(chat_id or ""),
        thread_id=str(thread_id) if thread_id else None,
        expires_at=float(expires_at) if expires_at is not None else None,
        idempotency_key=(
            str(idempotency_key)
            if idempotency_key
            else f"clarify:{session_key}:{clarify_id}"
        ),
        # Interception is armed only after the adapter confirms delivery.
        awaiting_text=False,
        home_key=str(get_hermes_home()),
    )
    _persist_new(entry)
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def wait_for_response(clarify_id: str, timeout: float) -> Optional[str]:
    """Block on the entry's event until resolved or timeout fires.

    Polls in 1-second slices so the agent's inactivity heartbeat keeps
    firing — without this, ``Event.wait(timeout=600)`` blocks the thread
    for 10 minutes with zero activity touches and the gateway's inactivity
    watchdog kills the agent while the user is still typing.

    ``timeout <= 0`` means an unlimited wait (never auto-skip mid-think); the
    heartbeat still fires each slice so inactivity watchdogs don't kill a live
    prompt.

    Returns the resolved response string, or ``None`` on timeout.
    """
    with _lock:
        entry = _entries.get(clarify_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
        touch_activity_if_due = None

    # 0 / negative → unlimited: no deadline, poll forever in 1s slices.
    unlimited = timeout is None or float(timeout) <= 0.0
    deadline = None if unlimited else time.monotonic() + float(timeout)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        if deadline is None:
            slice_s = 1.0
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            slice_s = min(1.0, remaining)
        if entry.event.wait(timeout=slice_s):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user clarify response")

    with _lock:
        if entry.response is None and entry.state in _RECOVERABLE_STATES:
            try:
                transitioned = _persist_transition(
                    clarify_id,
                    expected_states=set(_RECOVERABLE_STATES),
                    state=STATE_TIMED_OUT,
                    awaiting_text=False,
                    last_error="clarification timed out",
                )
            except ClarifyPersistenceError:
                logger.warning(
                    "Failed to persist clarify timeout for %s",
                    clarify_id,
                    exc_info=True,
                )
                transitioned = False
            if transitioned:
                entry.state = STATE_TIMED_OUT
                entry.awaiting_text = False
            else:
                _refresh_entry_from_durable(entry)
        _remove_entry(entry)
        return entry.response


# =========================================================================
# Public API — gateway / adapter side
# =========================================================================

def resolve_gateway_clarify(clarify_id: str, response: str) -> bool:
    """Unblock the agent thread waiting on ``clarify_id``.

    Returns True if an entry was found and resolved, False otherwise
    (already resolved, expired, or never existed).
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        if entry.state in _TERMINAL_STATES:
            return False
        resolved = str(response) if response is not None else ""
        try:
            transitioned = _persist_transition(
                clarify_id,
                expected_states=set(_RECOVERABLE_STATES),
                state=STATE_ANSWERED,
                response=resolved,
                awaiting_text=False,
            )
        except ClarifyPersistenceError:
            logger.warning(
                "Failed to persist clarify answer for %s",
                clarify_id,
                exc_info=True,
            )
            return False
        if not transitioned:
            _refresh_entry_from_durable(entry)
            return False
        entry.state = STATE_ANSWERED
        entry.response = resolved
        entry.awaiting_text = False
        entry.event.set()
        if entry.recovered:
            _remove_entry(entry)
        return True


def get_pending_for_session(
    session_key: str,
    *,
    include_choice_prompts: bool = False,
) -> Optional[_ClarifyEntry]:
    """Return the oldest pending clarify entry for a session, or None.

    By default this only returns entries awaiting free-form text (open-ended
    clarifies, or a multi-choice clarify after the user picked ``Other``).
    Gateways may pass ``include_choice_prompts=True`` when the user has typed
    directly in response to an active multi-choice prompt; in that case the
    oldest unresolved clarify is returned so the text can resolve it instead
    of being queued as an unrelated follow-up turn.
    """
    with _lock:
        ids = _session_index.get(session_key) or []
        for cid in ids:
            entry = _entries.get(cid)
            if entry is None:
                continue
            if not entry.delivery_confirmed:
                continue
            if include_choice_prompts or entry.awaiting_text:
                return entry
        return None


def _coerce_text_response(entry: _ClarifyEntry, response: str) -> Optional[str]:
    """Map typed choice replies to canonical choice text, otherwise keep or reject custom text.

    For native interactive multi-choice clarifies (button UI, awaiting_text=False):
      - Accept numeric selections ("2" → choice[1])
      - Accept exact choice label matches (case-insensitive)
      - Reject arbitrary prose (return None) so the message continues as a normal turn

    For multi-select clarifies (entry.multi_select=True):
      - Accept several numbers separated by commas and/or spaces ("1,3" / "1 3")
      - Accept exact choice label matches (single or comma-separated)
      - Out-of-range numbers reject the whole reply (return None) so the user
        can retry instead of silently getting a partial selection
      - Selections are returned as a JSON array string, which the clarify
        tool's ``_parse_multi_select_response`` decodes back into a list

    For text fallback or awaiting_text mode:
      - Accept any text (numeric/label/custom) after passing through coercion

    For open-ended clarifies (no choices):
      - Accept any text

    Returns None when the response should be rejected (arbitrary prose for native multi-choice).
    """
    text = str(response).strip()

    if not entry.choices:
        # Open-ended: accept any text
        return text

    if entry.multi_select:
        coerced = _coerce_multi_select_text(entry, text)
        if coerced is not None:
            return coerced
        # Not a parseable selection — accept as custom text only in
        # awaiting_text mode (the "Other" path); otherwise reject.
        return text if entry.awaiting_text else None

    # Try numeric selection first (always valid for multi-choice)
    try:
        idx = int(text) - 1
    except ValueError:
        idx = -1

    if 0 <= idx < len(entry.choices):
        return entry.choices[idx]

    # Try exact choice label match (always valid for multi-choice)
    for choice in entry.choices:
        if text.casefold() == str(choice).strip().casefold():
            return str(choice).strip()

    # For text fallback or awaiting_text mode, accept custom text
    # For native interactive multi-choice mode, reject arbitrary prose
    if entry.awaiting_text:
        return text

    return None


def _coerce_multi_select_text(entry: _ClarifyEntry, text: str) -> Optional[str]:
    """Parse a typed multi-select reply into a JSON array of choice labels.

    Accepts numbers and/or exact labels separated by commas (and, for
    all-numeric replies, bare spaces): "1,3", "1 3", "staging, prod".
    Returns ``None`` when any token is out of range or unrecognised so the
    caller can reject the reply cleanly instead of resolving a partial or
    wrong selection.
    """
    import json as _json

    if not text:
        return None
    choices = entry.choices or []

    # Split on commas first; if no commas and every whitespace-separated
    # token is numeric, treat spaces as separators too ("1 3").
    if "," in text:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
    else:
        parts = text.split()
        if len(parts) > 1 and all(p.strip().isdigit() for p in parts):
            tokens = [p.strip() for p in parts]
        else:
            tokens = [text]

    selected: List[str] = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(choices):
                label = str(choices[idx]).strip()
                if label not in selected:
                    selected.append(label)
                continue
            return None  # out-of-range number → reject whole reply
        # Exact label match (case-insensitive)
        matched = None
        for choice in choices:
            if token.casefold() == str(choice).strip().casefold():
                matched = str(choice).strip()
                break
        if matched is None:
            return None
        if matched not in selected:
            selected.append(matched)

    if not selected:
        return None
    return _json.dumps(selected, ensure_ascii=False)


def resolve_text_response_for_session(session_key: str, response: str) -> bool:
    """Resolve the oldest pending clarify in ``session_key`` from typed text.

    Returns False if no pending clarify exists or if the response was rejected
    (arbitrary prose for native interactive multi-choice clarifies).
    """
    return resolve_text_response_for_session_with_entry(session_key, response) is not None


def resolve_text_response_for_session_with_entry(
    session_key: str,
    response: str,
) -> Optional[_ClarifyEntry]:
    """Resolve typed text and return the entry that accepted it."""
    entry = get_pending_for_session(session_key, include_choice_prompts=True)
    if entry is None:
        return None

    coerced = _coerce_text_response(entry, response)
    if coerced is None:
        # Response rejected: message should continue as a normal turn
        return None

    if not resolve_gateway_clarify(entry.clarify_id, coerced):
        return None
    return entry


def mark_transport_accepted(clarify_id: str, poll_guid: Optional[str]) -> bool:
    """Record native transport acceptance without claiming user-visible delivery."""
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        if entry.state == STATE_ANSWERED:
            return True
        if entry.state in _TERMINAL_STATES:
            return False
        if entry.state == STATE_EVENT_CONFIRMED:
            return True
        normalized_guid = str(poll_guid or "")
        if entry.state == STATE_TRANSPORT_ACCEPTED:
            return (entry.poll_guid or "") == normalized_guid
        try:
            transitioned = _persist_transition(
                clarify_id,
                expected_states={STATE_CREATED},
                state=STATE_TRANSPORT_ACCEPTED,
                delivery_channel="native_poll",
                poll_guid=normalized_guid,
            )
        except ClarifyPersistenceError:
            logger.warning(
                "Failed to persist clarify transport acceptance for %s",
                clarify_id,
                exc_info=True,
            )
            return False
        if not transitioned:
            _refresh_entry_from_durable(entry)
            return False
        entry.state = STATE_TRANSPORT_ACCEPTED
        entry.delivery_channel = "native_poll"
        entry.poll_guid = normalized_guid or None
        return True


def mark_delivery_confirmed(clarify_id: str, channel: str) -> bool:
    """Arm reply interception only after the prompt is confirmed visible."""
    normalized_channel = str(channel or "text")
    awaiting_text = normalized_channel in {
        "numbered_text",
        "open_text",
        "text",
    }
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        if entry.state == STATE_ANSWERED:
            return True
        if entry.state in _TERMINAL_STATES:
            return False
        if entry.state == STATE_EVENT_CONFIRMED:
            if entry.delivery_channel == normalized_channel:
                return True
            if not (
                entry.recovered
                and entry.fallback_attempted
                and normalized_channel in _TEXT_DELIVERY_CHANNELS
            ):
                return False
            expected_states = {STATE_EVENT_CONFIRMED}
        else:
            expected_states = {STATE_CREATED, STATE_TRANSPORT_ACCEPTED}
        try:
            transitioned = _persist_transition(
                clarify_id,
                expected_states=expected_states,
                state=STATE_EVENT_CONFIRMED,
                delivery_channel=normalized_channel,
                awaiting_text=awaiting_text,
            )
        except ClarifyPersistenceError:
            logger.warning(
                "Failed to persist clarify delivery confirmation for %s",
                clarify_id,
                exc_info=True,
            )
            return False
        if not transitioned:
            _refresh_entry_from_durable(entry)
            return False
        entry.state = STATE_EVENT_CONFIRMED
        entry.delivery_channel = normalized_channel
        entry.awaiting_text = awaiting_text
        return True


def claim_text_fallback(clarify_id: str) -> bool:
    """Atomically reserve the one allowed numbered-text fallback attempt."""
    with _lock:
        entry = _entries.get(clarify_id)
        if (
            entry is None
            or entry.state in _TERMINAL_STATES
            or entry.fallback_attempted
        ):
            return False
        expected_states = {STATE_CREATED, STATE_TRANSPORT_ACCEPTED}
        if entry.recovered and entry.state == STATE_EVENT_CONFIRMED:
            expected_states.add(STATE_EVENT_CONFIRMED)
        if entry.state not in expected_states:
            return False
        try:
            with _DB_LOCK, _transaction() as conn:
                result = conn.execute(
                    """UPDATE clarify_obligations
                       SET fallback_attempted=1, updated_at=?
                       WHERE clarify_id=? AND fallback_attempted=0
                         AND state IN ({})""".format(
                            ", ".join("?" for _ in sorted(expected_states))
                        ),
                    (time.time(), clarify_id, *sorted(expected_states)),
                )
                if result.rowcount <= 0:
                    _refresh_entry_from_durable(entry)
                    return False
        except Exception as exc:
            logger.warning(
                "Failed to claim clarify text fallback for %s: %s",
                clarify_id,
                exc,
            )
            return False
        entry.fallback_attempted = True
        return True


def mark_failed(clarify_id: str, error: str) -> bool:
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or entry.state in _TERMINAL_STATES:
            return False
        try:
            transitioned = _persist_transition(
                clarify_id,
                expected_states=set(_RECOVERABLE_STATES),
                state=STATE_FAILED,
                awaiting_text=False,
                last_error=str(error or "clarification delivery failed"),
            )
        except ClarifyPersistenceError:
            logger.warning(
                "Failed to persist clarify failure for %s",
                clarify_id,
                exc_info=True,
            )
            return False
        if not transitioned:
            _refresh_entry_from_durable(entry)
            return False
        entry.state = STATE_FAILED
        entry.awaiting_text = False
        entry.response = ""
        entry.event.set()
        _remove_entry(entry)
        return True


def recovered_answer_note(entry: _ClarifyEntry) -> str:
    """Build the API-only note used when a post-restart answer resumes work."""
    answer = entry.response or ""
    choices = ""
    if entry.choices:
        choices = " Options were: " + "; ".join(entry.choices) + "."
    return (
        "[System note: Before the gateway restarted, the assistant asked this "
        f"clarifying question: {entry.question!r}.{choices} The user's answer "
        f"to that recovered clarification is: {answer!r}. Continue from this "
        "answer and do not ask the same question again.]"
    )


def _remove_entry(entry: _ClarifyEntry) -> None:
    with _lock:
        _entries.pop(entry.clarify_id, None)
        ids = _session_index.get(entry.session_key)
        if ids and entry.clarify_id in ids:
            ids.remove(entry.clarify_id)
            if not ids:
                _session_index.pop(entry.session_key, None)


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked the 'Other' button).

    Returns True if the entry exists and was flipped, False otherwise.
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or not entry.delivery_confirmed:
            return False
        try:
            transitioned = _persist_transition(
                clarify_id,
                expected_states={STATE_EVENT_CONFIRMED},
                awaiting_text=True,
            )
        except ClarifyPersistenceError:
            logger.warning(
                "Failed to persist clarify text interception for %s",
                clarify_id,
                exc_info=True,
            )
            return False
        if not transitioned:
            _refresh_entry_from_durable(entry)
            return False
        entry.awaiting_text = True
        return True


def has_pending(session_key: str) -> bool:
    """Return True when this session has at least one pending clarify entry."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def recovered_entries(
    platform: Optional[str] = None,
    *,
    home_key: Optional[str] = None,
) -> List[_ClarifyEntry]:
    """Return the currently claimed restart-recovery entries."""
    wanted = str(platform or "")
    wanted_home = str(home_key or "")
    with _lock:
        return [
            entry
            for entry in _entries.values()
            if entry.recovered
            and (not wanted or entry.platform == wanted)
            and (not wanted_home or entry.home_key == wanted_home)
        ]


def get_pending_for_poll(
    *,
    platform: str,
    chat_id: str,
    poll_guid: str,
    require_confirmed: bool = True,
) -> Optional[_ClarifyEntry]:
    """Return the confirmed clarify owned by one exact native poll."""
    wanted_platform = str(platform or "")
    wanted_chat = str(chat_id or "")
    wanted_guid = str(poll_guid or "")
    wanted_home = str(get_hermes_home())
    if not wanted_guid:
        return None
    with _lock:
        for entry in _entries.values():
            if (
                entry.platform == wanted_platform
                and entry.chat_id == wanted_chat
                and (entry.poll_guid or "") == wanted_guid
                and entry.home_key == wanted_home
                and (
                    not require_confirmed
                    or (
                        entry.state == STATE_EVENT_CONFIRMED
                        and entry.delivery_channel == "native_poll"
                    )
                )
            ):
                return entry
    return None


def clear_session(session_key: str) -> int:
    """Resolve and drop every pending clarify for a session.

    Used by session-boundary cleanup (e.g. ``/new``, gateway shutdown,
    cached-agent eviction) so blocked agent threads don't hang past the
    end of their session.  Returns the number of entries cancelled.
    """
    cancelled = 0
    with _lock:
        ids = list(_session_index.get(session_key, []) or [])
        for clarify_id in ids:
            entry = _entries.get(clarify_id)
            if entry is None:
                continue
            if entry.state in _TERMINAL_STATES or entry.event.is_set():
                entry.event.set()
                _remove_entry(entry)
                continue
            try:
                transitioned = _persist_transition(
                    entry.clarify_id,
                    expected_states=set(_RECOVERABLE_STATES),
                    state=STATE_FAILED,
                    awaiting_text=False,
                    last_error="clarification cancelled at session boundary",
                )
            except ClarifyPersistenceError:
                logger.warning(
                    "Failed to persist clarify cancellation for %s",
                    entry.clarify_id,
                    exc_info=True,
                )
                transitioned = False
            if not transitioned:
                _refresh_entry_from_durable(entry)
                if entry.state in _TERMINAL_STATES or entry.event.is_set():
                    entry.event.set()
                    _remove_entry(entry)
                continue
            entry.state = STATE_FAILED
            entry.awaiting_text = False
            if entry.response is None:
                entry.response = ""
            entry.event.set()
            _remove_entry(entry)
            cancelled += 1
    return cancelled


# =========================================================================
# Config
# =========================================================================

def resolve_clarify_timeout(config: dict) -> int:
    """Resolve the clarify timeout (seconds) from an already-loaded config dict.

    Single source of truth shared by every surface (messaging gateway, CLI,
    TUI/desktop) so the timeout can't drift between them.  Resolution order:

    1. legacy top-level ``clarify.timeout`` if a user explicitly set it,
    2. else the canonical ``agent.clarify_timeout``,
    3. else 3600 (1 hour).

    ``<= 0`` is preserved verbatim and means *unlimited* to callers (never
    auto-skip while the user is still deciding); the waiting loops translate
    that into a null deadline.  A non-numeric value falls back to 3600.
    """
    raw = (config.get("clarify") or {}).get("timeout")
    if raw is None:
        raw = (config.get("agent") or {}).get("clarify_timeout", 3600)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3600


def get_clarify_timeout() -> int:
    """Read the clarify response timeout (seconds) from config.

    Defaults to 3600 (1 hour) — long enough that a user who steps away
    (meeting, AFK, slow to read) still finds a live entry when they tap
    the button, short enough that a genuinely abandoned prompt eventually
    unblocks the agent thread instead of pinning the running-agent guard
    forever.  The old 600s default evicted the entry mid-think, so a late
    tap landed on a dead entry and the agent hung on ``running: clarify``
    (#32762).

    Reads ``agent.clarify_timeout`` from config.yaml (see
    :func:`resolve_clarify_timeout` for the full resolution order).  Set to
    ``0`` (or negative) for an unlimited wait — never auto-skip while the user
    is still deciding.
    """
    try:
        from hermes_cli.config import load_config
        return resolve_clarify_timeout(load_config() or {})
    except Exception:
        return 3600


# =========================================================================
# Per-session notify hook (gateway → adapter bridge)
# =========================================================================
# Mirrors tools.approval's _gateway_notify_cbs: the gateway registers a
# per-session callback that sends the clarify prompt to the user.  The
# callback bridges sync→async (runs on the agent thread; schedules the
# adapter ``send_clarify`` call on the event loop).

_notify_cbs: Dict[str, Callable[[_ClarifyEntry], None]] = {}


def register_notify(session_key: str, cb: Callable[[_ClarifyEntry], None]) -> None:
    """Register a per-session notify callback used by ``clarify_callback``."""
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_notify(session_key: str) -> None:
    """Drop the per-session notify callback and cancel any pending clarify entries."""
    with _lock:
        _notify_cbs.pop(session_key, None)
    # Cancel any pending entries so blocked threads unwind when the run
    # ends (interrupt, completion, gateway shutdown).
    clear_session(session_key)


def get_notify(session_key: str) -> Optional[Callable[[_ClarifyEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)
