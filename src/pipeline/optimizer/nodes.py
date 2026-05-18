"""LLM helper functions called by the optimizer graph nodes.

Each function wraps a single LLM call with a focused prompt.  They are kept
separate from graph.py so prompt text and parsing logic can be tested or
swapped independently of the graph wiring.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

# ── Prompt templates ──────────────────────────────────────────────────────────

_OPTIMIZE_PROMPT = """\
You are an expert PostgreSQL query optimizer.

The SQL below contains a large row-valued IN filter on rounded lat/lon coordinates:
  (ROUND(CAST(t.latitude AS NUMERIC), 1), ROUND(CAST(t.longitude AS NUMERIC), 1)) IN ((a1,b1), (a2,b2), ...)

Rewrite it to a VALUES-based JOIN for better query-plan performance.

Preferred form when the query has NO existing WITH clause:
  JOIN (VALUES (a1::numeric, b1::numeric), ...) AS _pts(rlat, rlon)
  ON ROUND(CAST(t.latitude AS NUMERIC), 1) = _pts.rlat
 AND ROUND(CAST(t.longitude AS NUMERIC), 1) = _pts.rlon

If the query already has a WITH clause, add a new CTE instead:
  _geo_pts(rlat, rlon) AS (VALUES (a1::numeric, b1::numeric), ...)
and replace the IN filter with a JOIN on _geo_pts.

Hard rules:
- Do NOT alter SELECT, aggregations, GROUP BY, ORDER BY, or any other filters.
- Preserve all table aliases and column names exactly as they appear.
- Return ONLY the rewritten SQL — no explanation, no markdown fences.

Original SQL:
{sql}
"""

_JUDGE_PROMPT = """\
You are a SQL semantic-equivalence judge.

Decide whether OPTIMIZED SQL is semantically equivalent to ORIGINAL SQL.
They are equivalent when the only change is converting a large lat/lon IN filter
to a VALUES-based JOIN that produces the same matching rows.

Return ONLY valid JSON (no markdown fences):
{{
  "is_equivalent": true or false,
  "feedback": "one sentence — what differs if not equivalent, or 'OK' if equivalent"
}}

ORIGINAL SQL:
{original}

OPTIMIZED SQL:
{optimized}
"""

_SELF_DEBUG_PROMPT = """\
You are an expert PostgreSQL query optimizer doing self-correction.

Your previous rewrite was rejected by a semantic-equivalence judge.
Produce a corrected version that is semantically equivalent to the original
while still converting the large lat/lon IN filter to a VALUES-based JOIN.

Judge feedback:
{feedback}

ORIGINAL SQL:
{original}

REJECTED OPTIMIZED SQL:
{optimized}

Hard rules:
- Do NOT alter SELECT, aggregations, GROUP BY, ORDER BY, or any other filters.
- Return ONLY the corrected SQL — no explanation, no markdown fences.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip(raw: str) -> str:
    return re.sub(r"```(?:sql|json)?|```", "", raw).strip()


# ── Public callables (used by graph.py via closures) ──────────────────────────

def call_optimize(sql: str, chat_model: Any) -> str:
    raw = chat_model.get_ans(_OPTIMIZE_PROMPT.format(sql=sql), temperature=0.0)
    return _strip(raw)


def call_judge(original: str, optimized: str, chat_model: Any) -> tuple[bool, str]:
    raw = chat_model.get_ans(
        _JUDGE_PROMPT.format(original=original, optimized=optimized),
        temperature=0.0,
    )
    try:
        result = json.loads(_strip(raw))
        return bool(result["is_equivalent"]), str(result.get("feedback", ""))
    except Exception as exc:
        logging.warning(
            "optimizer judge: JSON parse error (%s); treating as not equivalent. raw=%r",
            exc,
            raw[:300],
        )
        return False, f"parse_error: {exc}"


def call_self_debug(
    original: str, optimized: str, feedback: str, chat_model: Any
) -> str:
    raw = chat_model.get_ans(
        _SELF_DEBUG_PROMPT.format(
            original=original, optimized=optimized, feedback=feedback
        ),
        temperature=0.0,
    )
    return _strip(raw)
