"""Concurrency regression tests for persistent resolution drafts.

Two requests may append different conflict pointers to the same open
draft at the same time. The edit flow must keep the draft row's read,
open-status check, pointer validation, merge and write inside one SQLite
transaction so that:

* valid edits for different pointers are all retained (no lost update);
* the same pointer is resolved in transaction order, the later edit
  winning, and the stored JSON stays well-formed;
* a batch containing an invalid item writes nothing at all;
* an edit interleaved with a successful commit is rejected once the
  draft is committed, and never mutates the committed draft — the
  committed resolutions, the merge result and the audit event stay
  mutually consistent.
"""

import os
import tempfile
import threading

os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_draft_concurrency.db")

import pytest  # noqa: E402

from app import db, store  # noqa: E402

# Importing db first avoids the audit <-> db import cycle (db reads the
# method list out of audit); the API test modules reach the same order
# transitively through app.main.

from app.db import init_schema  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _schema():
    init_schema()


def _value_item(pointer: str, value) -> dict:
    return {
        "pointer": pointer,
        "choice": None,
        "value": value,
        "value_provided": True,
    }


def _create_pending_merge(n: int) -> str:
    base = {f"k{i}": 0 for i in range(n)}
    left = {f"k{i}": 1 for i in range(n)}
    right = {f"k{i}": 2 for i in range(n)}
    merge, _created = store.create_merge(base, left, right, None)
    assert merge["status"] == "pending"
    assert merge["conflict_count"] == n
    return merge["merge_id"]


def _run_threads(target, count: int, **kwargs) -> list:
    """Start ``count`` threads simultaneously behind one barrier.

    All threads enter the update transaction at the same instant, so the
    read-modify-write window actually overlaps; a plain sequential
    start loop would let earlier threads commit before later ones read.
    """
    barrier = threading.Barrier(count)
    results: list = [None] * count
    errors: list = [None] * count

    def worker(index: int) -> None:
        try:
            barrier.wait()
            results[index] = target(index, **kwargs)
        except BaseException as exc:  # captured, not swallowed: re-raised below
            errors[index] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for exc in errors:
        if exc is not None:
            raise exc
    return results


def test_concurrent_edits_to_different_pointers_are_all_retained():
    n = 12
    for _ in range(5):
        mid = _create_pending_merge(n)
        did = store.create_draft(mid)["draft_id"]

        def edit(i: int) -> dict:
            return store.update_draft(mid, did, [_value_item(f"/k{i}", 100 + i)])

        _run_threads(edit, n)

        # After every transaction has serialized, the stored draft holds
        # all n distinct pointers with no lost adjudication.
        draft = store.get_draft(mid, did)
        assert draft["status"] == "open"
        assert {r["pointer"] for r in draft["resolutions"]} == {
            f"/k{i}" for i in range(n)
        }
        by_pointer = {r["pointer"]: r["value"] for r in draft["resolutions"]}
        assert by_pointer == {f"/k{i}": 100 + i for i in range(n)}

        # The preview is consistent with the stored draft: it resolves
        # everything to exactly the concurrent adjudications.
        preview = store.preview_draft(mid, did)
        assert preview["would_resolve"] is True
        assert preview["remaining_conflicts"] == []
        assert preview["document"] == {f"k{i}": 100 + i for i in range(n)}


def test_concurrent_edits_to_same_pointer_resolve_in_order():
    n = 8
    mid = _create_pending_merge(n)
    did = store.create_draft(mid)["draft_id"]

    def edit(i: int) -> dict:
        return store.update_draft(mid, did, [_value_item("/k0", i)])

    _run_threads(edit, n)

    # Exactly one value survives — the last transaction to commit.
    draft = store.get_draft(mid, did)
    assert len(draft["resolutions"]) == 1
    assert draft["resolutions"][0]["pointer"] == "/k0"
    assert draft["resolutions"][0]["value"] in set(range(n))

    # Same-pointer overwrite interleaved with distinct-pointer appends.
    def mixed(i: int) -> dict:
        if i % 2 == 0:
            items = [_value_item(f"/k{i}", i), _value_item("/k1", -i)]
        else:
            items = [_value_item(f"/k{i}", i)]
        return store.update_draft(mid, did, items)

    _run_threads(mixed, n - 2)
    draft = store.get_draft(mid, did)
    pointers = sorted(r["pointer"] for r in draft["resolutions"])
    # The appended distinct pointers survive alongside /k0 and /k1.
    assert pointers == sorted({f"/k{i}" for i in range(n - 2)} | {"/k0", "/k1"})


def test_invalid_batch_under_concurrency_writes_nothing():
    n = 6
    mid = _create_pending_merge(n)
    did = store.create_draft(mid)["draft_id"]

    def edit(i: int):
        if i == n - 1:
            # One invalid item in the batch must reject the whole batch.
            return store.update_draft(
                mid, did, [_value_item(f"/k{i}", i), _value_item("/missing", 1)]
            )
        return store.update_draft(mid, did, [_value_item(f"/k{i}", i)])

    with pytest.raises(store.UnknownConflictPointer):
        _run_threads(edit, n)

    draft = store.get_draft(mid, did)
    stored = {r["pointer"] for r in draft["resolutions"]}
    # The invalid batch left nothing of itself behind...
    assert "/missing" not in stored
    # ...and the merge order was never touched by a draft edit.
    merge = store.get_merge(mid)
    assert merge["revision"] == 1
    assert merge["status"] == "pending"


def test_edit_interleaved_with_commit_cannot_mutate_committed_draft():
    n = 10
    for _ in range(5):
        mid = _create_pending_merge(n)
        did = store.create_draft(mid)["draft_id"]
        # Pre-fill every pointer so the commit resolves the merge outright.
        store.update_draft(
            mid, did, [_value_item(f"/k{i}", 100 + i) for i in range(n)]
        )

        start = threading.Barrier(n + 1)
        outcomes: list = [None] * (n + 1)

        def editor(i: int) -> None:
            start.wait()
            try:
                store.update_draft(mid, did, [_value_item(f"/k{i}", 900 + i)])
                outcomes[i] = "edited"
            except store.DraftNotOpen:
                outcomes[i] = "rejected"

        def committer() -> None:
            start.wait()
            try:
                store.commit_draft(mid, did)
                outcomes[n] = "committed"
            except store.DraftNotOpen:
                outcomes[n] = "rejected"

        threads = [threading.Thread(target=editor, args=(i,)) for i in range(n)]
        threads.append(threading.Thread(target=committer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The commit happens exactly once; a committed draft never goes back.
        assert outcomes[n] == "committed"
        draft = store.get_draft(mid, did)
        assert draft["status"] == "committed"
        merge = store.get_merge(mid)
        assert merge["status"] == "resolved"
        assert merge["revision"] == 2

        # Serialization correspondence: an "edited" outcome means the edit
        # took the lock *before* the commit, so its value is frozen into the
        # committed draft and the immutable result; a "rejected" outcome
        # means it lost the race to the commit, and its value must be
        # absent everywhere. No value can ever land after the commit.
        stored = {r["pointer"]: r["value"] for r in draft["resolutions"]}
        for i in range(n):
            if outcomes[i] == "edited":
                assert stored[f"/k{i}"] == 900 + i
                assert merge["result"][f"k{i}"] == 900 + i
            else:
                assert outcomes[i] == "rejected"
                assert stored[f"/k{i}"] == 100 + i
                assert merge["result"][f"k{i}"] == 100 + i

        # A later edit is refused outright, again without any write.
        with pytest.raises(store.DraftNotOpen):
            store.update_draft(mid, did, [_value_item("/k0", 1)])
        assert store.get_draft(mid, did)["status"] == "committed"
        assert store.get_merge(mid)["result"] == merge["result"]

        # The one draft-commit audit event matches exactly what was
        # persisted: same pointers, same revision span, a valid chain.
        trail = store.get_audit_trail(mid)
        assert trail["chain_valid"] is True
        commit_events = [e for e in trail["events"] if e["method"] == "draft-commit"]
        assert len(commit_events) == 1
        event = commit_events[0]
        assert event["revision_before"] == 1
        assert event["revision_after"] == 2
        assert set(event["pointers"]) == {f"/k{i}" for i in range(n)}
        assert event["detail"]["draft_id"] == did
