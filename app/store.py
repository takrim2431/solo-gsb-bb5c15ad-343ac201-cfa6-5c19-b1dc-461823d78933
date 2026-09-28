"""Merge-order persistence and state transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import audit, db, evidence, rules, undo
from .merge import MISSING, delete_at, merge3, parse_pointer, set_at
from .patch import generate_patch, sha256_hex
from .release_plan import build_preview
from .reuse import (
    build_reuse_plan,
    match_to_batch_item,
    match_to_view,
)
from .validation import check_result_schema, validate_result
from .rules import (
    InvalidPattern,
    InvalidRuleSet,
    RuleApplicationFailed,
    apply_plan,
    compile_rules,
    group_matches,
    plan_resolutions,
)

STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"

DRAFT_OPEN = "open"
DRAFT_COMMITTED = "committed"


class MergeNotFound(Exception):
    pass


class MergeNotResolved(Exception):
    """Raised when the deterministic patch is requested before resolution."""


class StaleRevision(Exception):
    def __init__(self, current_revision: int):
        self.current_revision = current_revision
        super().__init__(f"stale revision, current is {current_revision}")


class AlreadyResolved(Exception):
    pass


class UnknownConflictPointer(Exception):
    def __init__(self, pointer: str):
        self.pointer = pointer
        super().__init__(f"no open conflict at pointer {pointer!r}")


class InvalidResolution(Exception):
    pass


class DraftNotFound(Exception):
    pass


class DraftNotOpen(Exception):
    """The draft was already committed (or otherwise invalidated)."""


class DraftConflictSetChanged(Exception):
    """The merge's open conflicts no longer cover the draft's pointers."""


class SourceMergeInvalid(Exception):
    """The source merge of a reuse request cannot provide resolutions."""


class NothingToUndo(Exception):
    """No effective (non-compensated) adjudication event exists to undo."""


class AuditChainBroken(Exception):
    """The merge's audit chain does not verify; rebuilding it is unsafe."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return {
        "merge_id": row["id"],
        "status": row["status"],
        "revision": row["revision"],
        "document": json.loads(row["document"]),
        "conflicts": json.loads(row["conflicts"]),
        "conflict_count": len(json.loads(row["conflicts"])),
        "result": json.loads(row["result"]) if row["result"] is not None else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_merge(merge_id: str) -> dict:
    row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if row is None:
        raise MergeNotFound(merge_id)
    return _row_to_dict(row)


def get_merge_patch(merge_id: str) -> dict:
    """Return the RFC 6902 patch from base_doc to the immutable result.

    Raises MergeNotFound (404) for an unknown merge and MergeNotResolved
    (409) while the merge is still pending.
    """
    row = db.query_one(
        "SELECT status, base_doc, result FROM merges WHERE id = ?",
        (merge_id,),
    )
    if row is None:
        raise MergeNotFound(merge_id)
    # Availability follows status alone: SQL NULL means "no result yet" for
    # a pending merge, while a resolved merge always carries JSON text —
    # including a root-level null, stored as 'null'.
    if row["status"] != STATUS_RESOLVED:
        raise MergeNotResolved(merge_id)
    base = json.loads(row["base_doc"])
    result = json.loads(row["result"])
    return {
        "merge_id": merge_id,
        "status": STATUS_RESOLVED,
        "patch": generate_patch(base, result),
        "base_sha256": sha256_hex(base),
        "result_sha256": sha256_hex(result),
    }


def get_audit_trail(merge_id: str) -> dict:
    """Return the merge order's adjudication audit trail and chain verdict.

    Events are returned in sequence order, each re-verified against the
    per-merge SHA-256 chain; the response reports whether the whole chain
    verifies and, when it does not, the first anomalous sequence number.
    Raises MergeNotFound (404) for an unknown merge.
    """
    if db.query_one("SELECT id FROM merges WHERE id = ?", (merge_id,)) is None:
        raise MergeNotFound(merge_id)
    return audit.get_audit_trail(merge_id)


def export_evidence(merge_id: str) -> dict:
    """Build the signed portable evidence package for a resolved merge.

    The package carries the original base document, the immutable result,
    the deterministic base-to-result patch, the complete audit history and
    the generation time, signed with the configured HMAC key. Raises
    MergeNotFound (404) for an unknown merge and MergeNotResolved (409)
    while the merge is still pending.
    """
    row = db.query_one(
        "SELECT status, base_doc, result FROM merges WHERE id = ?",
        (merge_id,),
    )
    if row is None:
        raise MergeNotFound(merge_id)
    # Status alone decides availability, mirroring get_merge_patch: a
    # resolved merge always has a JSON result (a root null is stored as
    # 'null'), so the evidence package embeds that exact result.
    if row["status"] != STATUS_RESOLVED:
        raise MergeNotResolved(merge_id)
    return evidence.build_package(
        merge_id,
        json.loads(row["base_doc"]),
        json.loads(row["result"]),
        audit.list_event_dicts(merge_id),
    )


def create_merge(
    base: Any,
    left: Any,
    right: Any,
    idempotency_key: str | None,
    result_schema: Any = None,
) -> tuple[dict, bool]:
    """Create a merge order. Returns (merge_view, created).

    A repeated idempotency key replays the existing merge order instead of
    creating a new one. When ``result_schema`` is given it must itself be a
    valid Draft 2020-12 JSON Schema, and a conflict-free (immediately
    resolved) merge must produce a document satisfying it; either failure
    aborts creation before any row is inserted.
    """
    if idempotency_key:
        row = db.query_one(
            "SELECT * FROM merges WHERE idempotency_key = ?", (idempotency_key,)
        )
        if row is not None:
            return _row_to_dict(row), False

    # Gate 1: the schema itself must be a valid Draft 2020-12 schema.
    if result_schema is not None:
        check_result_schema(result_schema)

    document, conflicts = merge3(base, left, right)
    now = _now()
    resolved = not conflicts
    status = STATUS_RESOLVED if resolved else STATUS_PENDING
    # No conflicts: the result is final and immutable from the start, so it
    # must pass the result schema (gate 2) before anything is persisted.
    if resolved and result_schema is not None:
        validate_result(result_schema, document)

    merge_id = uuid.uuid4().hex
    try:
        db.execute(
            """
            INSERT INTO merges (
                id, idempotency_key, status, revision,
                base_doc, left_doc, right_doc,
                document, conflicts, source_conflicts, initial_document,
                result, result_schema,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                merge_id,
                idempotency_key,
                status,
                1,
                json.dumps(base),
                json.dumps(left),
                json.dumps(right),
                json.dumps(document),
                json.dumps(conflicts),
                # Immutable snapshot of the conflicts as first recorded:
                # resolutions shrink ``conflicts`` but reuse matching needs
                # the original base/left/right/missing of every conflict.
                json.dumps(conflicts),
                # Immutable initial working document: resolution undo
                # replays audit events from this snapshot instead of
                # reverse-editing the current document.
                json.dumps(document),
                # Persist the result by status, not by Python truthiness: a
                # resolved root-level JSON null is stored as the JSON text
                # 'null', while SQL NULL exclusively means "still pending,
                # no result produced yet" (including a JSON null result,
                # which ``json.dumps`` writes as 'null').
                json.dumps(document) if resolved else None,
                json.dumps(result_schema) if result_schema is not None else None,
                now,
                now,
            ),
        )
    except Exception:
        # Concurrent insert with the same idempotency key: replay wins.
        if idempotency_key:
            row = db.query_one(
                "SELECT * FROM merges WHERE idempotency_key = ?", (idempotency_key,)
            )
            if row is not None:
                return _row_to_dict(row), False
        raise
    return get_merge(merge_id), True


def _apply_batch(
    document: Any, conflicts: list[dict], resolutions: list[dict]
) -> tuple[Any, list[dict]]:
    """Validate and apply a batch of resolutions in memory.

    Returns (document, remaining_conflicts). Pure: the caller decides
    whether the outcome is persisted (apply_resolutions) or discarded
    (preview_resolutions). A repeated pointer fails because the first
    occurrence closes the conflict.
    """
    open_conflicts = {c["pointer"]: c for c in conflicts}
    for item in resolutions:
        pointer = item["pointer"]
        conflict = open_conflicts.pop(pointer, None)
        if conflict is None:
            raise UnknownConflictPointer(pointer)
        tokens = parse_pointer(pointer)

        if item.get("choice") is not None:
            side = item["choice"]
            if side in conflict["missing"]:
                # Chosen side deleted the key: honour the deletion.
                document = delete_at(document, tokens)
            else:
                document = set_at(document, tokens, conflict[side])
        elif item.get("value_provided"):
            document = set_at(document, tokens, item["value"])
        else:
            raise InvalidResolution(
                f"resolution for {pointer!r} needs 'choice' or 'value'"
            )

    remaining = [c for c in conflicts if c["pointer"] in open_conflicts]
    return document, remaining


def _resolution_detail_items(resolutions: list[dict]) -> list[dict]:
    """Compact per-pointer record of what a resolution batch decided.

    Mirrors the choice/value semantics of the batch items: an explicit
    null ``value`` is preserved as ``{"value": null}``.
    """
    items = []
    for item in resolutions:
        entry = {"pointer": item["pointer"]}
        if item.get("choice") is not None:
            entry["choice"] = item["choice"]
        else:
            entry["value"] = item["value"]
        items.append(entry)
    return items


# ---------------------------------------------------------------------------
# Shared adjudication commit phase
#
# Every state-changing adjudication entry point — explicit resolutions,
# rule resolutions, historical reuse and draft commits — runs the same
# commit phase: inside one transaction the merge row is read and guarded
# (status + revision), a pure in-memory adjudication plan is built, and the
# single committer persists it (final schema gate, merge update, revision
# bump, audit event, draft invalidation). A plan that matched nothing is
# never committed: no write, no revision bump, no audit event.
# ---------------------------------------------------------------------------


def _fetch_merge_row(conn, merge_id: str):
    """Fetch the merge row inside the caller's transaction.

    Raises MergeNotFound (404) for an unknown merge.
    """
    row = conn.execute(
        "SELECT * FROM merges WHERE id = ?", (merge_id,)
    ).fetchone()
    if row is None:
        raise MergeNotFound(merge_id)
    return row


def _require_pending_at_revision(row, expected_revision: int) -> None:
    """Guard a fetched merge row before any plan is built.

    Raises AlreadyResolved (409) once the result is final and
    StaleRevision (409) when the expected revision is not current.
    """
    if row["status"] == STATUS_RESOLVED:
        raise AlreadyResolved(row["id"])
    if row["revision"] != expected_revision:
        raise StaleRevision(row["revision"])


@dataclass(frozen=True)
class AdjudicationPlan:
    """Everything one adjudication batch needs to commit, as pure data.

    Built in memory by an entry point after its matching and validation;
    constructing the plan writes nothing. ``pointers`` lists the effective
    pointers in the exact order the audit event records them, ``detail``
    is the method-specific audit payload, and ``draft_id`` — when set —
    names the draft the committer invalidates in the same transaction.
    """

    document: Any
    remaining: list[dict]
    pointers: list[str]
    method: str
    detail: dict
    draft_id: str | None = None


def _commit_adjudication(
    conn, merge_id: str, row, plan: AdjudicationPlan
) -> None:
    """Persist one adjudication plan inside the caller's transaction.

    The single writer shared by every adjudication entry point: applies
    the final result-schema gate, updates the merge row (status, document,
    remaining conflicts, immutable result) with exactly one revision bump,
    invalidates the draft when the plan carries one, and appends the audit
    event. Any failure raises, and the caller's transaction rolls the
    merge update, the draft invalidation and the audit event back together.
    """
    resolved = not plan.remaining
    # Gate: only a batch that resolves the last conflict produces a final
    # document, so only that batch is validated. A partial batch still
    # carries conflict placeholders and is committed unchecked.
    if resolved and row["result_schema"] is not None:
        validate_result(json.loads(row["result_schema"]), plan.document)
    now = _now()
    conn.execute(
        """
        UPDATE merges
        SET status = ?, revision = revision + 1, document = ?,
            conflicts = ?, result = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            STATUS_RESOLVED if resolved else STATUS_PENDING,
            json.dumps(plan.document),
            json.dumps(plan.remaining),
            # Once written the result is never updated again: immutable.
            json.dumps(plan.document) if resolved else None,
            now,
            merge_id,
        ),
    )
    if plan.draft_id is not None:
        # The draft is invalidated in the same transaction as the merge
        # update that commits its resolutions.
        conn.execute(
            "UPDATE merge_drafts SET status = ?, updated_at = ? WHERE id = ?",
            (DRAFT_COMMITTED, now, plan.draft_id),
        )
    # Same transaction as the merge update (and the draft invalidation):
    # the audit event commits or rolls back together with the state change
    # it describes.
    audit.record_event(
        conn,
        merge_id,
        plan.method,
        row["revision"],
        row["revision"] + 1,
        plan.pointers,
        plan.detail,
    )


def apply_resolutions(
    merge_id: str, expected_revision: int, resolutions: list[dict]
) -> dict:
    """Apply a batch of conflict resolutions.

    Raises StaleRevision (409) if expected_revision does not match the
    current revision, and AlreadyResolved (409) once the result is final.
    A successful batch appends one audit event inside the same transaction
    as the merge update; a rejected batch leaves neither.
    """
    with db.transaction() as conn:
        row = _fetch_merge_row(conn, merge_id)
        _require_pending_at_revision(row, expected_revision)

        document, remaining = _apply_batch(
            json.loads(row["document"]), json.loads(row["conflicts"]), resolutions
        )
        plan = AdjudicationPlan(
            document=document,
            remaining=remaining,
            pointers=[item["pointer"] for item in resolutions],
            method=audit.METHOD_RESOLUTIONS,
            detail={"items": _resolution_detail_items(resolutions)},
        )
        _commit_adjudication(conn, merge_id, row, plan)
    return get_merge(merge_id)


def preview_resolutions(
    merge_id: str, expected_revision: int, resolutions: list[dict]
) -> dict:
    """Dry-run a resolution batch on an in-memory copy of the merge.

    Same revision/status guards and per-item validation as
    apply_resolutions, but stricter: an item carrying both 'choice' and
    'value' is rejected. A batch that would resolve the last conflict is
    also checked against the merge's result_schema with the same
    validate_result call the commit path uses, so a candidate the real
    submission would reject (422) fails the preview identically instead of
    reporting a false would_resolve. Nothing is persisted — no write, no
    revision bump, no status change.
    """
    row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] == STATUS_RESOLVED:
        raise AlreadyResolved(merge_id)
    if row["revision"] != expected_revision:
        raise StaleRevision(row["revision"])

    for item in resolutions:
        if item.get("choice") is not None and item.get("value_provided"):
            raise InvalidResolution(
                f"resolution for {item['pointer']!r} gives both 'choice' and 'value'"
            )

    document, remaining = _apply_batch(
        json.loads(row["document"]), json.loads(row["conflicts"]), resolutions
    )
    would_resolve = not remaining
    if would_resolve and row["result_schema"] is not None:
        # Same Draft 2020-12 gate as the committing endpoint: a batch that
        # would resolve the last conflict must yield a final document
        # satisfying result_schema, otherwise the preview fails with the
        # very 422 error list the real submission would return instead of
        # falsely reporting would_resolve. A partial batch (conflicts
        # remain) is never validated, exactly like a partial commit. The
        # preview writes nothing either way, so a failed gate leaves the
        # merge, its drafts and the audit trail untouched.
        validate_result(json.loads(row["result_schema"]), document)
    preview = {
        "merge_id": merge_id,
        "revision": row["revision"],
        "document": document,
        "remaining_conflicts": remaining,
        "would_resolve": would_resolve,
    }
    if would_resolve:
        # Same deterministic patch/digest the resolved merge would yield.
        base = json.loads(row["base_doc"])
        preview["patch"] = generate_patch(base, document)
        preview["result_sha256"] = sha256_hex(document)
    return preview


def apply_rule_resolutions(
    merge_id: str, expected_revision: int, rule_items: list[dict]
) -> dict:
    """Resolve open conflicts with ordered path-pattern rules.

    Every open conflict is matched against the ordered rules and the unique
    best match (more literal segments, then fewer '*', then declaration
    order) supplies its side. All hits are validated before any write: an
    invalid pattern, a rule without any pattern or a failure applying a
    hit rejects the whole request with no partial modification. The
    revision is bumped at most once, inside the single transaction that
    also appends the audit event; a run that matched nothing writes
    nothing and leaves no event.
    """
    with db.transaction() as conn:
        row = _fetch_merge_row(conn, merge_id)
        _require_pending_at_revision(row, expected_revision)

        # Compiling and planning happens entirely in memory; nothing is
        # touched until every selected conflict has been proven applicable.
        compiled_rules = compile_rules(rule_items)
        conflicts = json.loads(row["conflicts"])
        winners = plan_resolutions(conflicts, compiled_rules)
        rule_matches = group_matches(winners)
        unmatched = [c["pointer"] for c in conflicts if c["pointer"] not in {w.pointer for w in winners}]

        applied = False
        if winners:
            document, remaining = apply_plan(json.loads(row["document"]), conflicts, winners)
            # Winners apply in sorted pointer order (see apply_plan); the
            # event records that same effective order.
            ordered = sorted(winners, key=lambda w: w.pointer)
            plan = AdjudicationPlan(
                document=document,
                remaining=remaining,
                pointers=[w.pointer for w in ordered],
                method=audit.METHOD_RULE_RESOLUTIONS,
                detail={
                    "items": [
                        {
                            "pointer": w.pointer,
                            "side": w.side,
                            "rule_index": w.rule_index,
                        }
                        for w in ordered
                    ]
                },
            )
            _commit_adjudication(conn, merge_id, row, plan)
            applied = True
        # Nothing matched: nothing to validate or apply, no state change,
        # no revision bump and no audit event; the unmatched conflicts are
        # reported back below.

    view = get_merge(merge_id)
    view["rule_matches"] = rule_matches
    view["unmatched_conflicts"] = unmatched
    view["applied"] = applied
    return view


# ---------------------------------------------------------------------------
# Historical resolution reuse
# ---------------------------------------------------------------------------


def reuse_resolutions(
    merge_id: str, expected_revision: int, source_merge_id: str
) -> dict:
    """Reuse a resolved merge's adjudications on the current open conflicts.

    Every open conflict is matched by JSON Pointer against the source
    merge's conflict snapshot; a match requires base/left/right and the
    missing semantics to be identical after canonicalization (an absent
    key never matches an explicit null). Each match takes its final value
    from the source's immutable result — a path missing there deletes the
    target key as well. Matching and validation run entirely in memory;
    the single UPDATE either applies every match (one revision bump) or
    nothing at all. Zero matches is a successful no-op: no write and no
    revision bump.

    Raises StaleRevision (409) / AlreadyResolved (409) for the target and
    SourceMergeInvalid (400) when the source is unknown, still pending or
    has no conflict snapshot.
    """
    with db.transaction() as conn:
        row = _fetch_merge_row(conn, merge_id)
        _require_pending_at_revision(row, expected_revision)

        source_row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (source_merge_id,)
        ).fetchone()
        if source_row is None:
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} does not exist"
            )
        if source_row["status"] != STATUS_RESOLVED:
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} is not resolved yet"
            )
        if source_row["source_conflicts"] is None:
            # Rows created before conflict snapshots existed cannot serve as
            # a reuse source: their original conflict candidates are gone.
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} has no conflict snapshot"
            )
        if source_merge_id == merge_id:
            raise SourceMergeInvalid(
                "a merge cannot reuse resolutions from itself"
            )

        target_conflicts = json.loads(row["conflicts"])
        source_conflicts = json.loads(source_row["source_conflicts"])
        source_result = json.loads(source_row["result"])
        matches, skipped = build_reuse_plan(
            target_conflicts, source_conflicts, source_result
        )

        if not matches:
            # Nothing reusable: no write, no revision bump.
            applied = False
        else:
            conflict_by_pointer = {c["pointer"]: c for c in target_conflicts}
            batch = [
                match_to_batch_item(match, conflict_by_pointer[match.pointer])
                for match in matches
            ]
            document, remaining = _apply_batch(
                json.loads(row["document"]), target_conflicts, batch
            )
            plan = AdjudicationPlan(
                document=document,
                remaining=remaining,
                pointers=[match.pointer for match in matches],
                method=audit.METHOD_REUSE_RESOLUTIONS,
                detail={
                    "source_merge_id": source_merge_id,
                    "items": [match_to_view(match) for match in matches],
                },
            )
            # A zero-match reuse never reaches the committer and therefore
            # records no event.
            _commit_adjudication(conn, merge_id, row, plan)
            applied = True

    view = get_merge(merge_id)
    view["applied"] = applied
    view["source_merge_id"] = source_merge_id
    view["applied_resolutions"] = [match_to_view(m) for m in matches]
    view["skipped"] = skipped
    view["applied_count"] = len(matches)
    view["skipped_count"] = len(skipped)
    return view


# ---------------------------------------------------------------------------
# Resolution drafts: a persistent, editable set of conflict resolutions that
# pins the merge revision at creation time and commits atomically.
# ---------------------------------------------------------------------------


def _draft_row_to_dict(row) -> dict:
    stored = json.loads(row["resolutions"])
    resolutions = []
    for pointer, entry in stored.items():
        item = {"pointer": pointer}
        if "choice" in entry:
            item["choice"] = entry["choice"]
        else:
            item["value"] = entry["value"]
        resolutions.append(item)
    return {
        "draft_id": row["id"],
        "merge_id": row["merge_id"],
        "status": row["status"],
        "base_revision": row["base_revision"],
        "resolutions": resolutions,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _stored_to_batch(stored: dict) -> list[dict]:
    """Convert the draft's stored resolutions into _apply_batch items.

    ``stored`` maps pointer -> {"choice": side} or {"value": value}; a
    ``value`` key is always explicit (it may hold a JSON null), mirroring
    the ``value_provided`` semantics of the resolution endpoints.
    """
    return [
        {
            "pointer": pointer,
            "choice": entry.get("choice"),
            "value": entry.get("value"),
            "value_provided": "value" in entry,
        }
        for pointer, entry in stored.items()
    ]


def _get_draft_rows(merge_id: str, draft_id: str):
    """Fetch (merge_row, draft_row) or raise MergeNotFound / DraftNotFound.

    Both rows come from one read-only snapshot transaction, so a draft
    query or preview can never pair one version of the merge row with a
    different (concurrently committed) version of the draft row.
    """
    with db.read_snapshot() as conn:
        merge_row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        draft_row = conn.execute(
            "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
    if merge_row is None:
        raise MergeNotFound(merge_id)
    if draft_row is None or draft_row["merge_id"] != merge_id:
        raise DraftNotFound(draft_id)
    return merge_row, draft_row


def _fetch_draft_rows(conn, merge_id: str, draft_id: str):
    """Fetch (merge_row, draft_row) inside the caller's write transaction.

    Raises MergeNotFound (404) for an unknown merge and DraftNotFound (404)
    for an unknown draft or a draft reached through another merge's URL.
    """
    merge_row = conn.execute(
        "SELECT * FROM merges WHERE id = ?", (merge_id,)
    ).fetchone()
    if merge_row is None:
        raise MergeNotFound(merge_id)
    draft_row = conn.execute(
        "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
    ).fetchone()
    if draft_row is None or draft_row["merge_id"] != merge_id:
        raise DraftNotFound(draft_id)
    return merge_row, draft_row


def create_draft(merge_id: str) -> dict:
    """Open a resolution draft pinned to the merge's current revision.

    The draft snapshots the working document and the open conflicts, so
    later edits and previews are stable even if the merge moves on. The
    pending check, the snapshot copy and the insert run in one transaction,
    so a merge resolved by another request in between cannot be snapshotted
    as a draft. Only a pending merge accepts drafts; a resolved one yields
    AlreadyResolved.
    """
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if row is None:
            raise MergeNotFound(merge_id)
        if row["status"] == STATUS_RESOLVED:
            raise AlreadyResolved(merge_id)
        draft_id = uuid.uuid4().hex
        now = _now()
        conn.execute(
            """
            INSERT INTO merge_drafts (
                id, merge_id, status, base_revision,
                document, conflicts, resolutions, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft_id,
                merge_id,
                DRAFT_OPEN,
                row["revision"],
                row["document"],
                row["conflicts"],
                "{}",
                now,
                now,
            ),
        )
        # Build the response from the row just inserted, in the same
        # transaction, so it is exactly what was persisted.
        draft_row = conn.execute(
            "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        view = _draft_row_to_dict(draft_row)
    return view


def get_draft(merge_id: str, draft_id: str) -> dict:
    _, draft_row = _get_draft_rows(merge_id, draft_id)
    return _draft_row_to_dict(draft_row)


def update_draft(merge_id: str, draft_id: str, items: list[dict]) -> dict:
    """Append or overwrite conflict resolutions on the draft.

    Each item follows the same choice/value semantics as a resolution
    batch: exactly one of 'choice' / 'value' (an explicit null counts as
    'value'). A pointer already present in the draft is overwritten in
    place; a new one is appended. Only the draft row is written — the
    merge order is never touched by draft edits.

    The draft row is read, checked open, validated against its own
    conflict snapshot, merged and written back inside one transaction.
    Concurrent edits therefore serialize on the stored state: edits for
    different pointers all survive, and the same pointer resolves in
    transaction order with the later edit winning. A commit landing first
    flips the draft to 'committed' in the same transaction as the merge
    update, so an interleaved edit sees that status and is rejected.
    Every item in a batch is validated before the merge, so an invalid
    item rolls the whole batch back and the draft is left byte-for-byte
    unchanged.
    """
    with db.transaction() as conn:
        _, draft_row = _fetch_draft_rows(conn, merge_id, draft_id)
        if draft_row["status"] != DRAFT_OPEN:
            raise DraftNotOpen(draft_id)
        open_pointers = {c["pointer"] for c in json.loads(draft_row["conflicts"])}

        # Validate the whole batch against the transaction's snapshot
        # before touching stored: an unknown pointer or a malformed
        # choice/value rejects everything and nothing is written.
        for item in items:
            pointer = item["pointer"]
            if pointer not in open_pointers:
                raise UnknownConflictPointer(pointer)
            if item.get("choice") is not None and item.get("value_provided"):
                raise InvalidResolution(
                    f"resolution for {pointer!r} gives both 'choice' and 'value'"
                )
            if item.get("choice") is None and not item.get("value_provided"):
                raise InvalidResolution(
                    f"resolution for {pointer!r} needs 'choice' or 'value'"
                )

        stored = json.loads(draft_row["resolutions"])
        for item in items:
            pointer = item["pointer"]
            if item.get("choice") is not None:
                stored[pointer] = {"choice": item["choice"]}
            else:
                stored[pointer] = {"value": item["value"]}
        now = _now()
        conn.execute(
            "UPDATE merge_drafts SET resolutions = ?, updated_at = ? WHERE id = ?",
            (json.dumps(stored), now, draft_id),
        )
        # Re-read inside the transaction: the returned view is the exact
        # committed state of the draft (including the new updated_at).
        updated_row = conn.execute(
            "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        view = _draft_row_to_dict(updated_row)
    return view


def preview_draft(merge_id: str, draft_id: str) -> dict:
    """Dry-run the draft's resolutions against its creation-time snapshot.

    Nothing is written. Returns the would-be document and the conflicts
    still open after the draft (pointers may be filled in over several
    edits, so a partial draft simply reports the rest); when the draft
    would resolve everything, the candidate final document is checked
    against the merge's result_schema with the same validate_result call
    the commit uses — a draft the commit would reject (422) fails the
    preview identically — and the deterministic base-to-result patch and
    its SHA-256 are included, mirroring the resolution-preview endpoint.
    """
    merge_row, draft_row = _get_draft_rows(merge_id, draft_id)
    if draft_row["status"] != DRAFT_OPEN:
        raise DraftNotOpen(draft_id)
    document, remaining = _apply_batch(
        json.loads(draft_row["document"]),
        json.loads(draft_row["conflicts"]),
        _stored_to_batch(json.loads(draft_row["resolutions"])),
    )
    would_resolve = not remaining
    if would_resolve and merge_row["result_schema"] is not None:
        # Same Draft 2020-12 gate as the draft commit: a draft that would
        # resolve the last conflict must yield a final document satisfying
        # result_schema, otherwise the preview fails with the very 422
        # error list the commit would return instead of falsely reporting
        # would_resolve. A partial draft is never validated, exactly like
        # a partial commit. The preview writes nothing either way, so a
        # failed gate leaves the merge, the draft and the audit trail
        # untouched.
        validate_result(json.loads(merge_row["result_schema"]), document)
    preview = {
        "merge_id": merge_id,
        "draft_id": draft_id,
        "base_revision": draft_row["base_revision"],
        "current_revision": merge_row["revision"],
        "document": document,
        "remaining_conflicts": remaining,
        "would_resolve": would_resolve,
    }
    if would_resolve:
        base = json.loads(merge_row["base_doc"])
        preview["patch"] = generate_patch(base, document)
        preview["result_sha256"] = sha256_hex(document)
    return preview


def commit_draft(merge_id: str, draft_id: str) -> dict:
    """Apply the whole draft to the merge in a single transaction.

    Inside the transaction the merge row is re-read and re-validated: the
    draft must still be open, the merge still pending, the revision still
    equal to the draft's pinned base_revision, every drafted pointer still
    an open conflict, and — when the draft resolves the last conflict —
    the candidate result must satisfy result_schema. Any changed condition
    aborts with 409 (a schema violation with 422) and neither the merge
    nor the draft is partially written. On success all resolutions are
    applied with exactly one revision bump and the draft is invalidated.
    """
    with db.transaction() as conn:
        merge_row, draft_row = _fetch_draft_rows(conn, merge_id, draft_id)
        if draft_row["status"] != DRAFT_OPEN:
            raise DraftNotOpen(draft_id)
        _require_pending_at_revision(merge_row, draft_row["base_revision"])

        conflicts = json.loads(merge_row["conflicts"])
        open_pointers = {c["pointer"] for c in conflicts}
        stored = json.loads(draft_row["resolutions"])
        if not stored:
            raise InvalidResolution("draft has no resolutions to commit")
        if any(pointer not in open_pointers for pointer in stored):
            raise DraftConflictSetChanged(
                "the merge's open conflicts no longer cover the draft"
            )

        document, remaining = _apply_batch(
            json.loads(merge_row["document"]), conflicts, _stored_to_batch(stored)
        )
        plan = AdjudicationPlan(
            document=document,
            remaining=remaining,
            pointers=list(stored.keys()),
            method=audit.METHOD_DRAFT_COMMIT,
            detail={
                "draft_id": draft_id,
                "items": _resolution_detail_items(_stored_to_batch(stored)),
            },
            draft_id=draft_id,
        )
        # Commits the merge update, the draft invalidation and the audit
        # event together; any failure rolls all three back.
        _commit_adjudication(conn, merge_id, merge_row, plan)
    view = get_merge(merge_id)
    view["draft_id"] = draft_id
    return view


# ---------------------------------------------------------------------------
# Undo of the last effective adjudication batch
# ---------------------------------------------------------------------------


def undo_last_resolution(merge_id: str, expected_revision: int) -> dict:
    """Undo exactly the most recent still-effective adjudication event.

    The merge must be pending and the supplied revision current. The
    document and open conflicts are never reverse-edited: they are rebuilt
    from the immutable initial conflict snapshot by replaying the audit
    history with the target event (and every already compensated event)
    skipped. On success the merge row is replaced with the rebuilt state,
    the revision is bumped by one, and a ``resolution-undo`` event is
    appended in the same transaction; its detail names the compensated
    event's sequence number and the restored pointers. A later undo skips
    compensated events automatically, and the hash chain stays verifiable.

    Raises StaleRevision (409) for an outdated revision, AlreadyResolved
    (409) for a resolved (immutable) merge and NothingToUndo (409) when no
    effective adjudication remains. Any failure rolls the whole request
    back: no state change and no audit event.
    """
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if row is None:
            raise MergeNotFound(merge_id)
        if row["status"] == STATUS_RESOLVED:
            raise AlreadyResolved(merge_id)
        if row["revision"] != expected_revision:
            raise StaleRevision(row["revision"])

        events = conn.execute(
            "SELECT * FROM merge_audit_events WHERE merge_id = ? ORDER BY seq ASC",
            (merge_id,),
        ).fetchall()
        # Fail closed: the rebuilt state is only as trustworthy as the
        # history it is replayed from. A tampered or gapped chain makes the
        # recorded decisions unverifiable, so no state is rewritten and no
        # undo event is appended; the exception rolls the transaction back.
        if not audit.verify_chain(conn, merge_id):
            raise AuditChainBroken(merge_id)
        target = undo.find_undo_target(events)
        if target is None:
            raise NothingToUndo(merge_id)

        restored_pointers = json.loads(target["pointers"])
        # Rebuild from the immutable snapshot, skipping the target batch.
        # Replay raises AuditReplayError on an inconsistent history; that
        # propagates and rolls the transaction back, so no state changes.
        document, open_conflicts = undo.rebuild_state(
            row, events, skip_seq=target["seq"]
        )

        conn.execute(
            """
            UPDATE merges
            SET revision = revision + 1, document = ?,
                conflicts = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(document),
                json.dumps(open_conflicts),
                _now(),
                merge_id,
            ),
        )
        # The undo itself is a chained event. It does not change status
        # (a resolved merge cannot be undone) and records exactly which
        # event it compensates, so a subsequent undo skips both.
        audit.record_event(
            conn,
            merge_id,
            audit.METHOD_RESOLUTION_UNDO,
            row["revision"],
            row["revision"] + 1,
            restored_pointers,
            {
                undo.DETAIL_UNDONE_SEQ: target["seq"],
                undo.DETAIL_RESTORED_POINTERS: restored_pointers,
                "undone_method": target["method"],
            },
        )

    view = get_merge(merge_id)
    view["undone_seq"] = target["seq"]
    view["undone_method"] = target["method"]
    view["restored_pointers"] = restored_pointers
    view["restored_count"] = len(restored_pointers)
    return view


# ---------------------------------------------------------------------------
# Read-only release-plan preview
# ---------------------------------------------------------------------------


def preview_release_plan(nodes: list[dict]) -> dict:
    """Preview stable release batches for a dependency graph of merge orders.

    Read-only: the nodes are validated as an acyclic graph whose edges
    point at other listed nodes, their current status/revision is fetched
    from SQLite, and release batches are computed entirely in memory.
    Nothing is written and no merge row is updated, so the preview never
    bumps a revision. Raises release_plan.InvalidReleasePlan (400) for
    duplicate nodes, self-dependencies, dangling edges, cycles or merges
    that do not exist.
    """
    merge_ids = [node["merge_id"] for node in nodes]
    # SQLite caps host parameters at 999 on older builds, so fetch the rows
    # in 500-id chunks. All chunks run inside one read-only snapshot
    # transaction: adjudications committing between two chunk queries must
    # not mix an old prerequisite state with a new successor state (which
    # would falsely report a publishable node as dependency_blocked). The
    # snapshot connection writes nothing and bumps no revision; the next
    # request opens a fresh snapshot and observes any committed state.
    state: dict[str, dict] = {}
    with db.read_snapshot() as conn:
        for start in range(0, len(merge_ids), 500):
            chunk = merge_ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT id, status, revision FROM merges WHERE id IN ({placeholders})",
                tuple(chunk),
            ):
                state[row["id"]] = {
                    "status": row["status"],
                    "revision": row["revision"],
                }
    return build_preview(nodes, state)
