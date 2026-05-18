"""Pure-regex detection of large ROUND(CAST(lat/lon)) IN (...) patterns.

No LLM, no upstream context dependencies.  Self-contained.
"""

from __future__ import annotations

import re
from typing import Optional

POINT_COUNT_THRESHOLD = 20

# Matches the outer row-value constructor:
#   (ROUND(CAST(<col> AS NUMERIC/DECIMAL), 1),
#    ROUND(CAST(<col> AS NUMERIC/DECIMAL), 1)) IN (
# Inner CAST expressions are assumed to be simple column references (no nested parens).
_ROUND_IN_RE = re.compile(
    r'\(\s*'
    r'ROUND\s*\(\s*CAST\s*\([^)]+\s+AS\s+(?:NUMERIC|DECIMAL)\s*\)\s*,\s*1\s*\)'
    r'\s*,\s*'
    r'ROUND\s*\(\s*CAST\s*\([^)]+\s+AS\s+(?:NUMERIC|DECIMAL)\s*\)\s*,\s*1\s*\)'
    r'\s*\)\s+IN\s*\(',
    re.IGNORECASE,
)

# A single numeric pair inside the IN list: (35.9, 23.2)
_PAIR_RE = re.compile(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)')


def extract_points(sql: str) -> Optional[list[tuple[float, float]]]:
    """Return (lat, lon) pairs from the ROUND...IN clause, or None if pattern absent."""
    m = _ROUND_IN_RE.search(sql)
    if not m:
        return None
    # Walk parenthesis depth from the opening '(' of the IN list
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
    in_body = sql[start:end]
    pairs = _PAIR_RE.findall(in_body)
    return [(float(lat), float(lon)) for lat, lon in pairs] if pairs else []


def needs_optimization(sql: str, threshold: int = POINT_COUNT_THRESHOLD) -> bool:
    """True when SQL has the target pattern with more than *threshold* point pairs."""
    points = extract_points(sql)
    return points is not None and len(points) > threshold
