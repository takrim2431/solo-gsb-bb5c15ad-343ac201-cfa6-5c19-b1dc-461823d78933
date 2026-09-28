"""Startup repair for resolved merges whose result was stored as SQL NULL.

Older code persisted the result with
``json.dumps(result) if result is not None else None``, so a conflict-free
merge whose final document was a root-level JSON null ended up
``status='resolved'`` with ``result IS NULL`` — the same SQL value that, by
contract, means "still pending, no result yet". Startup now repairs those
rows by restoring the result from the saved working document (a merge
resolved at creation time never mutates its document), while pending rows
keep their SQL NULL.
"""

import json
import os
import tempfile

os.environ.setdefault(
    "EVIDENCE_HMAC_KEY", "test-evidence-key-0123456789abcdef0123456789abcdef"
)
os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(), "test_migration.db"))

import sqlite3  # noqa: E402

from app import db, store  # noqa: E402

# The exact column set of an old database file, written without the repair
# ever having run.
_COLUMNS = (
    "id, idempotency_key, status, revision, base_doc, left_doc, right_doc, "
    "document, conflicts, source_conflicts, initial_document, result, "
    "result_schema, created_at, updated_at"
)
_PLACEHOLDERS = ",".join("?" * 15)


def _insert_old_row(conn, values):
    conn.execute(
        f"INSERT INTO merges ({_COLUMNS}) VALUES ({_PLACEHOLDERS})", values
    )


def _build_legacy_database(path):
    """Create a database file exactly as the pre-fix version would leave it."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE merges (
            id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,
            status TEXT NOT NULL CHECK (status IN ('pending', 'resolved')),
            revision INTEGER NOT NULL,
            base_doc TEXT NOT NULL,
            left_doc TEXT NOT NULL,
            right_doc TEXT NOT NULL,
            document TEXT NOT NULL,
            conflicts TEXT NOT NULL,
            source_conflicts TEXT,
            initial_document TEXT,
            result TEXT,
            result_schema TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Resolved conflict-free root-null merge, result wrongly SQL NULL.
    _insert_old_row(
        conn,
        (
            "bad-null", "k1", "resolved", 1,
            json.dumps({"a": 1}), json.dumps(None), json.dumps(None),
            "null", "[]", "[]", "null",
            None, None, "t", "t",
        ),
    )
    # A second bad row whose document is a non-trivial JSON value, to prove
    # the repair copies the document verbatim rather than hardcoding 'null'.
    _insert_old_row(
        conn,
        (
            "bad-doc", "k2", "resolved", 1,
            json.dumps({"a": 1}), json.dumps({"a": 2}), json.dumps({"a": 2}),
            json.dumps({"a": 2}), "[]", "[]", json.dumps({"a": 2}),
            None, None, "t", "t",
        ),
    )
    # Pending: SQL NULL genuinely means "no result yet" and must survive.
    _insert_old_row(
        conn,
        (
            "pending", "k3", "pending", 1,
            json.dumps({"a": 1}), json.dumps({"a": 2}), json.dumps({"a": 3}),
            json.dumps({"a": {"__merge_conflict__": "/a"}}),
            json.dumps(
                [{"pointer": "/a", "base": 1, "left": 2, "right": 3,
                  "missing": []}]
            ),
            "[]", json.dumps({"a": {"__merge_conflict__": "/a"}}),
            None, None, "t", "t",
        ),
    )
    # Healthy resolved row: must not be rewritten.
    _insert_old_row(
        conn,
        (
            "healthy", "k4", "resolved", 1,
            json.dumps({"a": 1}), json.dumps({"a": 9}), json.dumps({"a": 9}),
            json.dumps({"a": 9}), "[]", "[]", json.dumps({"a": 9}),
            json.dumps({"a": 9}), None, "t", "t",
        ),
    )
    conn.commit()
    conn.close()


def test_startup_repairs_resolved_null_results_from_saved_document(tmp_path):
    legacy = tmp_path / "legacy.db"
    _build_legacy_database(str(legacy))

    # Point the persistence layer at the legacy file for this test.
    original_path = db.DATABASE_PATH
    db._conn = None
    db.DATABASE_PATH = str(legacy)
    try:
        db.init_schema()

        direct = sqlite3.connect(str(legacy))
        try:
            bad_null = direct.execute(
                "SELECT result, document FROM merges WHERE id = 'bad-null'"
            ).fetchone()
            bad_doc = direct.execute(
                "SELECT result, document FROM merges WHERE id = 'bad-doc'"
            ).fetchone()
            pending = direct.execute(
                "SELECT result, status FROM merges WHERE id = 'pending'"
            ).fetchone()
            healthy = direct.execute(
                "SELECT result FROM merges WHERE id = 'healthy'"
            ).fetchone()
        finally:
            direct.close()

        # Resolved rows: result restored from the saved document...
        assert bad_null[0] == "null" == bad_null[1]
        assert bad_doc[0] == bad_doc[1] == json.dumps({"a": 2})
        # ...pending rows are not touched...
        assert pending[1] == "pending" and pending[0] is None
        # ...and already-correct rows stay byte-for-byte unchanged.
        assert healthy[0] == json.dumps({"a": 9})

        # The repaired rows now behave like normal resolved merges.
        patch_view = store.get_merge_patch("bad-null")
        assert patch_view["patch"] == [
            {"op": "replace", "path": "", "value": None}
        ]
        package = store.export_evidence("bad-null")
        assert package["payload"]["result"] is None

        from app import evidence

        assert evidence.verify_package(package)["valid"] is True
        assert store.get_merge("bad-doc")["result"] == {"a": 2}

        # The pending merge still reports unavailable patch/evidence (409).
        import pytest

        with pytest.raises(store.MergeNotResolved):
            store.get_merge_patch("pending")
        with pytest.raises(store.MergeNotResolved):
            store.export_evidence("pending")

        # The repair is idempotent: a second startup changes nothing.
        db.init_schema()
        direct = sqlite3.connect(str(legacy))
        try:
            assert direct.execute(
                "SELECT result FROM merges WHERE id = 'bad-null'"
            ).fetchone()[0] == "null"
            assert direct.execute(
                "SELECT result FROM merges WHERE id = 'pending'"
            ).fetchone()[0] is None
        finally:
            direct.close()
    finally:
        # Restore the shared layer so other test modules are unaffected.
        db._conn = None
        db.DATABASE_PATH = original_path
