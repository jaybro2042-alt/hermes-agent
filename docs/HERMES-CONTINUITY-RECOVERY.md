# Hermes continuity recovery — 2026-09-10

Scope: core context semantics, Desktop reconnect/session lifecycle, verified
successor publication, and exact messaging-origin rebinding. No live service,
production profile, credential, paid model, or outbound message was changed by
the continuity implementation worker. OPS-14 parent owns deployment and review.

## Provenance

Base: Hermes main `77cf2f36df`, clean task worktree
`/mnt/physea/.worktrees/hermes-continuity-20260910`, branch
`ops-14-hermes-continuity-20260910`. The live source and old Desktop build remain
preserved evidence, not editing targets.

The broker registration shape and staged plugin were recovered from the prior
interrupted bootstrap effort in `/mnt/physea/director/worktrees/hermes-bootstrap`
and `/mnt/physea/director/worktrees/liminal-auto-handoff`. This change replaces
caller-trusted receipts and mocked-only readiness with a plugin-owned remote
store reader and actual agent construction. The plugin is isolated under the
parent platform worktree's `scripts/hermes/liminal_auto_handoff`; not installed.

## Contract

`handoff.successor.create` takes only an active source runtime/durable ID and
handoff ID. Caller-supplied receipt/context/route/address are not authority.
The registered reader runs under the source profile and must return a verified,
resumable, unfinished checkpoint naming that exact source session, including
objective, artifact state and next action. The successor restores that frame
as evidence, never transcript instructions. The receiver must subsequently
recheck referenced artifacts/criteria before consequential execution.

The actual source and successor agents must match provider, model, API mode,
base URL, reasoning and service tier. A real SessionDB transaction validates
parent, profile and peer ownership, then publishes one immutable
`session-successor/1` record in `state_meta`. Exact originating chat/thread/user
identity is copied from the source, never from a latest-session search.
Retries reuse the same child; process restart rebuilds the same durable child
and returns its fresh runtime ID. The original row and transcript survive.
A separate `session-successor-pending/1` reservation survives a restart before
acknowledgment; pending children never become routing targets. Resume validates
alternating compression and handoff links, so an older bookmarked source still
reaches its exact successor without searching unrelated recent sessions.

After publication, the source inference loop pauses without tearing down tool
state or child work. New source prompts are refused with the successor ID;
`session.resume` follows verified links. The messaging gateway adopts only the
exact linked child on the next inbound message for its existing routing key.
Desktop event payload includes source/child durable IDs, runtime IDs, profile,
route, checkpoint, and `acknowledged: true`; UI integration belongs to OPS-14.

An acknowledgment proves store integrity, identity, restored context and agent
construction. It is **not** proof of model-generated semantic verification,
artifact correctness, an executed next action, or live delivery acceptance.

## Context policy and migration

Core compaction retains still-authorized unfinished work and settled answers;
explicit cancellations/changes still override earlier objectives. Retired
summary wrappers are frozen and normalized on upgrade.

The isolated plugin selects roles via session metadata and profile
`continuity` configuration. Worker dispatch metadata outranks a Director
process setting. Defaults: Director at most 1 compaction, worker 0, personal
normal behavior. Counters live in profile SessionDB and follow compression
lineage, so reconnect/restart/row rotation does not reset the ceiling. A durable
reservation before summary means a crash can conservatively consume allowance;
it never grants more. Remote summary failure preserves source rather than
switching to an unbounded paid fallback. Example owner configuration:

```yaml
context:
  engine: liminal
continuity:
  role: director
  director_max_compactions: 1
  worker_max_compactions: 0
  handoff_tokens: 178500
  context_references:
    - /path/to/current-context-manifest.md
    - /path/to/answered-decisions/
```

Unclassified users/models are not inferred to be Directors from model name,
checkpoint subject, or a hardcoded workspace. Every handoff nudge requests goal,
state, answered decisions, responsibility, limits, next expected result,
escalation, unfinished tasks, known incidents and evidence/provenance pointers.
Paid workers preserve their whole-tree receipt/deadline; native worker successor
launch is denied until the canonical supervisor has a resume seam.

## Validation and activation

Use the canonical `scripts/run_tests.sh -j 2` with a test-capable Hermes Python.
The continuity test constructs actual AIAgent objects and real SQLite sessions
under temporary profiles. Only credentials and remote/fake model output are
substituted; no inference, external messages or desktop automation is performed.
Additional tests cover cross-owner/profile/thread rejection, forged caller
receipts, restart idempotence, failed construction/store read, compaction limits,
legacy state migration, and real inbound session routing.

The final offline gate passed 577 tests across 56 files in 19.1 seconds with two
workers, including mixed compression/handoff lineage and malformed remote
receipts. The 107-test gateway/protocol/continuity subset also passed separately.
See the parent session log for Desktop review. These are offline checks, not
live delivery acceptance.

Before claiming production success: parent independently reviews the diff,
selects a single pinned Python source for Desktop and gateway, preserves live
profile/plugin snapshots, installs the approved isolated plugin, and obtains
Jay-controlled service activation. Verify one live remote checkpoint, actual
app successor switch, one response on the originating phone thread, no response
on another peer, disconnect/reconnect during a task, receipt/deadline retention,
and representative settled-answer/unfinished-task recall. Monitor latency,
errors and context size; do not promise zero future degradation.

Rollback: revert only this task's code/plugin deployment to the preserved
versions; do not reset dirty live repos or delete sessions. New state_meta keys
are additive and older versions ignore them. Explicitly select the preserved
source session if rollback removes successor-aware routing. Never rewrite
phone bindings to an inferred latest session, and never restart the bridge as
a spend-control action.
