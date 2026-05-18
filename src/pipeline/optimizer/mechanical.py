"""Pure-Python mechanical SQL optimizer.

Converts the large ROUND(CAST(lat/lon)) IN (...) predicate to a
VALUES-based subquery without any LLM calls.  Semantically identical:
  (expr1, expr2) IN (SELECT rlat, rlon FROM (VALUES ...) AS _geo_pts(rlat, rlon))
is equivalent to:
  (expr1, expr2) IN ((v1, v2), ...)

Saves ~180 s of LLM latency per query that has a large geo IN-list.
"""

from __future__ import annotations

import re
from typing import Optional

from .detector import _ROUND_IN_RE, _PAIR_RE


_LAST_WORD_RE = re.compile(r"\b(\w+)\s*$", re.IGNORECASE)

# SQL keywords that immediately precede the row-constructor where the
# VALUES-subquery rewrite is safe (i.e. not inside a CASE WHEN expression).
# In a CASE WHEN context, converting to IN (SELECT FROM VALUES) causes
# PostgreSQL to require the outer columns (lat, lon) in GROUP BY.
_SAFE_PRECEDING_WORDS = frozenset({"where", "and", "or", "on", "having", "join"})


def mechanical_optimize(sql: str) -> tuple[str, bool]:
    """Rewrite large ROUND(CAST(lat/lon)) IN (...) as a VALUES subquery.

    Returns (new_sql, was_transformed).  When no matching pattern is found,
    the pairs list is empty, or the IN predicate is inside a CASE WHEN
    expression, returns (original_sql, False).
    """
    m = _ROUND_IN_RE.search(sql)
    if not m:
        return sql, False

    # Guard: skip when the IN predicate is inside a CASE WHEN expression.
    # In that context, converting to IN (SELECT FROM VALUES) forces PostgreSQL
    # to require the outer lat/lon columns in GROUP BY, breaking the query.
    prefix_word = _LAST_WORD_RE.search(sql[: m.start()].rstrip())
    if prefix_word and prefix_word.group(1).lower() not in _SAFE_PRECEDING_WORDS:
        return sql, False

    # Walk depth to find the closing ')' of the IN list
    start = m.end() - 1  # position of the opening '(' of IN (...)
    depth = 0
    end = start
    for i in range(start, len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    in_body = sql[start:end]
    pairs = _PAIR_RE.findall(in_body)
    if not pairs:
        return sql, False

    # Build VALUES clause: (lat::numeric, lon::numeric), ...
    values_rows = ", ".join(f"({lat}::numeric, {lon}::numeric)" for lat, lon in pairs)
    new_in = f"(SELECT rlat, rlon FROM (VALUES {values_rows}) AS _geo_pts(rlat, rlon))"

    new_sql = sql[:start] + new_in + sql[end:]
    return new_sql, True


def verify_mechanical_transform(original: str, rewritten: str) -> bool:
    """Lightweight Python-side equivalence check (no LLM).

    Verifies that the rewritten SQL has the same coordinate pairs as the
    original IN-list by comparing extracted pairs from both sides.
    """
    original_pairs = _extract_in_pairs(original)
    rewritten_pairs = _extract_values_pairs(rewritten)
    return set(original_pairs) == set(rewritten_pairs)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_in_pairs(sql: str) -> list[tuple[str, str]]:
    m = _ROUND_IN_RE.search(sql)
    if not m:
        return []
    start = m.end() - 1
    depth = 0
    end = start
    for i in range(start, len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return _PAIR_RE.findall(sql[start:end])


_VALUES_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*::numeric\s*,\s*(-?\d+(?:\.\d+)?)\s*::numeric\s*\)"
)


def _extract_values_pairs(sql: str) -> list[tuple[str, str]]:
    return _VALUES_PAIR_RE.findall(sql)
