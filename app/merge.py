"""Recursive three-way merge over JSON documents.

Merge rules (applied per JSON Pointer path):
  * only one side changed            -> adopt that side automatically
  * both sides changed identically   -> adopt the shared value
  * both sides changed differently   -> objects recurse key-by-key;
                                        anything else becomes a conflict
  * delete vs. unchanged             -> delete
  * delete vs. modify                -> conflict

"Changed" is decided by :func:`json_equal`, a strict JSON-type-aware
recursive equality: booleans never equal numbers (``true`` != ``1``),
integers never equal floats (``1`` != ``1.0``), objects and arrays compare
element-by-element under the same rules, and an absent key (``MISSING``)
never equals an explicit ``null``. Python's loose ``==`` (where
``True == 1``) is never used, so a bool-vs-number divergence on the two
surfaces as a conflict instead of being silently auto-merged.

Conflict locations keep a placeholder in the working document and both
candidates are returned to the client.
"""

from __future__ import annotations

from typing import Any

MISSING = object()  # sentinel: key absent on that side

PLACEHOLDER_KEY = "__merge_conflict__"


def placeholder(pointer: str) -> dict:
    """Marker left in the working document at a conflict location."""
    return {PLACEHOLDER_KEY: pointer}


def escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def parse_pointer(pointer: str) -> list[str]:
    """Parse an RFC 6901 JSON Pointer into tokens. '' -> root."""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer!r}")
    return [unescape(tok) for tok in pointer.split("/")[1:]]


def json_equal(a: Any, b: Any) -> bool:
    """Strict, MISSING-aware JSON equality, shared by every merge decision.

    Unlike Python's ``==`` this distinguishes ``True`` from ``1`` and ``1``
    from ``1.0`` (booleans, integers and floats are different JSON types),
    and applies the same rules recursively inside objects and arrays, so a
    nested ``true`` vs ``1`` divergence is never mistaken for an identical
    change. ``MISSING`` only equals ``MISSING``: an absent key stays
    distinct from an explicit JSON ``null``.
    """
    if a is MISSING or b is MISSING:
        return a is MISSING and b is MISSING
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            json_equal(x, y) for x, y in zip(a, b)
        )
    return a == b


def merge3(base: Any, left: Any, right: Any) -> tuple[Any, list[dict]]:
    """Merge left/right against base.

    Returns (document, conflicts): the working document with placeholders
    at conflict paths, and the conflict list with both candidates.
    """
    conflicts: list[dict] = []
    doc = _merge(base, left, right, "", conflicts)
    # The recursion visits object keys through a set union; sort on the way
    # out so conflict order (and thus snapshot/reuse order) is deterministic
    # regardless of Python's hash-seeded set iteration.
    conflicts.sort(key=lambda c: c["pointer"])
    return doc, conflicts


def _merge(base: Any, left: Any, right: Any, path: str, conflicts: list[dict]) -> Any:
    # Both sides agree: neither changed, or both made the same change.
    if json_equal(left, right):
        return left
    # Only right changed.
    if json_equal(left, base):
        return right
    # Only left changed.
    if json_equal(right, base):
        return left
    # Both changed differently: recurse into objects, conflict otherwise.
    if isinstance(left, dict) and isinstance(right, dict):
        base_obj = base if isinstance(base, dict) else {}
        out: dict[str, Any] = {}
        for key in base_obj.keys() | left.keys() | right.keys():
            value = _merge(
                base_obj.get(key, MISSING),
                left.get(key, MISSING),
                right.get(key, MISSING),
                path + "/" + escape(key),
                conflicts,
            )
            if value is not MISSING:
                out[key] = value
        return out
    conflicts.append(
        {
            "pointer": path,
            "base": None if base is MISSING else base,
            "left": None if left is MISSING else left,
            "right": None if right is MISSING else right,
            "missing": [
                name
                for name, v in (("base", base), ("left", left), ("right", right))
                if v is MISSING
            ],
        }
    )
    return placeholder(path)


def get_at(doc: Any, tokens: list[str]) -> Any:
    node = doc
    for tok in tokens:
        if isinstance(node, dict):
            node = node[tok]
        elif isinstance(node, list):
            node = node[int(tok)]
        else:
            raise KeyError(tok)
    return node


def set_at(doc: Any, tokens: list[str], value: Any) -> Any:
    """Set value at pointer path, returning the (possibly new) document."""
    if not tokens:
        return value
    parent = get_at(doc, tokens[:-1])
    last = tokens[-1]
    if isinstance(parent, dict):
        parent[last] = value
    elif isinstance(parent, list):
        parent[int(last)] = value
    else:
        raise KeyError(last)
    return doc


def delete_at(doc: Any, tokens: list[str]) -> Any:
    """Remove the key at pointer path (used when a missing side is chosen)."""
    if not tokens:
        raise ValueError("cannot delete the document root")
    parent = get_at(doc, tokens[:-1])
    last = tokens[-1]
    if isinstance(parent, dict):
        parent.pop(last, None)
    elif isinstance(parent, list):
        del parent[int(last)]
    else:
        raise KeyError(last)
    return doc
