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
    """Rewrite all large ROUND(CAST(lat/lon)) IN (...) predicates as VALUES subqueries.

    Returns (new_sql, was_transformed).  When no matching pattern is found,
    the pairs list is empty, or the IN predicate is inside a CASE WHEN
    expression, returns (original_sql, False).

    Processes all occurrences left-to-right (each rewrite shifts positions, so
    we restart the search on the updated SQL after each substitution).
    """
    current = sql
    transformed = False
    search_pos = 0

    while True:
        m = _ROUND_IN_RE.search(current, search_pos)
        if not m:
            break

        # Guard: skip when the IN predicate is inside a CASE WHEN expression.
        prefix_word = _LAST_WORD_RE.search(current[: m.start()].rstrip())
        if prefix_word and prefix_word.group(1).lower() not in _SAFE_PRECEDING_WORDS:
            search_pos = m.end()
            continue

        # Walk depth to find the closing ')' of the IN list
        start = m.end() - 1  # position of the opening '(' of IN (...)
        depth = 0
        end = start
        for i in range(start, len(current)):
            if current[i] == "(":
                depth += 1
            elif current[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        in_body = current[start:end]
        pairs = _PAIR_RE.findall(in_body)
        if not pairs:
            # Already rewritten (VALUES subquery) or empty — skip this match
            search_pos = m.end()
            continue

        values_rows = ", ".join(f"({lat}::numeric, {lon}::numeric)" for lat, lon in pairs)
        new_in = f"(SELECT rlat, rlon FROM (VALUES {values_rows}) AS _geo_pts(rlat, rlon))"

        current = current[:start] + new_in + current[end:]
        transformed = True
        search_pos = 0  # positions shifted; restart scan from beginning

    return current, transformed


def verify_mechanical_transform(original: str, rewritten: str) -> bool:
    """Lightweight Python-side equivalence check (no LLM).

    Verifies that the rewritten SQL has the same coordinate pairs as the
    original IN-lists (all occurrences) by comparing extracted pair multisets.
    """
    original_pairs = _extract_in_pairs(original)
    rewritten_pairs = _extract_values_pairs(rewritten)
    return sorted(original_pairs) == sorted(rewritten_pairs)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_in_pairs(sql: str) -> list[tuple[str, str]]:
    """Extract all coordinate pairs from all ROUND...IN(...) occurrences."""
    all_pairs: list[tuple[str, str]] = []
    pos = 0
    while True:
        m = _ROUND_IN_RE.search(sql, pos)
        if not m:
            break
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
        all_pairs.extend(_PAIR_RE.findall(sql[start:end]))
        pos = end
    return all_pairs


_VALUES_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*::numeric\s*,\s*(-?\d+(?:\.\d+)?)\s*::numeric\s*\)"
)


def _extract_values_pairs(sql: str) -> list[tuple[str, str]]:
    return _VALUES_PAIR_RE.findall(sql)
