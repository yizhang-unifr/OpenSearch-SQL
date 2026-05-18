"""State model for the query-optimizer LangGraph."""

from __future__ import annotations

from typing import TypedDict


class OptimizerState(TypedDict):
    original_sql: str        # never mutated – used as reference in judge/self_debug
    current_sql: str         # evolves: optimize → self_debug cycles
    attempt: int             # number of completed judge calls
    is_equivalent: bool | None
    feedback: str            # latest judge verdict text
    trace: list[dict]        # full audit trail for logging / export
