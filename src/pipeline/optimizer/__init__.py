"""Query-optimizer pipeline node.

Sits between candidate_generate and align_correct.  For each candidate SQL
that contains a large ROUND(CAST(lat/lon)) IN (...) predicate (> 20 pairs),
rewrites it to a VALUES-based subquery using a pure-Python mechanical transform
(no LLM calls, saves ~180 s per query).  Falls back to the original SQL if the
mechanical transform fails verification.

This package is fully self-contained and does NOT read from any upstream
context nodes (geo_context, implicit_context, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager

from .detector import extract_points, needs_optimization, POINT_COUNT_THRESHOLD, _ROUND_IN_RE
from .mechanical import mechanical_optimize, verify_mechanical_transform

# Dedicated JSONL log for mechanical-transform failures.
# Written to the project root (cwd at run time) so it accumulates across runs
# and can be inspected with: jq . optimizer_failures.jsonl
_FAILURE_LOG = Path("optimizer_failures.jsonl")


def _log_failure(
    *,
    question_id: Any,
    db_id: str,
    question: str,
    candidate_index: int,
    sql: str,
    failure_type: str,  # "no_transform" | "verify_mismatch"
    rewritten: str | None,
) -> None:
    """Append one JSON record per failure to optimizer_failures.jsonl.

    failure_type values:
      no_transform   — mechanical_optimize could not parse the IN-list
                       (regex matched needs_optimization but pair extraction failed)
      verify_mismatch — transform succeeded but pair-set comparison disagreed
                        (suggests a bug in _PAIR_RE or _VALUES_PAIR_RE)
    """
    # Capture the raw IN-list fragment to make regex debugging easy
    in_list_snippet: str | None = None
    m = _ROUND_IN_RE.search(sql)
    if m:
        # Extract the first 300 chars of the IN body (enough to see the pattern)
        start = m.end() - 1
        in_list_snippet = sql[start : start + 300].replace("\n", " ")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question_id": question_id,
        "db_id": db_id,
        "question": question,
        "candidate_index": candidate_index,
        "failure_type": failure_type,
        # First 600 chars of original SQL (enough to see the full WHERE fragment)
        "sql_head": sql[:600],
        "in_list_snippet": in_list_snippet,
        # Only when verify_mismatch: show what the rewrite looked like
        "rewritten_head": rewritten[:600] if rewritten else None,
    }
    try:
        with _FAILURE_LOG.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logging.error("query_optimizer: could not write failure log: %s", exc)

    logging.warning(
        "query_optimizer: mechanical transform failure — type=%s question_id=%s candidate=%d "
        "in_list_snippet=%r",
        failure_type,
        question_id,
        candidate_index,
        (in_list_snippet or "")[:120],
    )


@node_decorator(check_schema_status=False)
def query_optimizer(task: Any, execution_history: list[dict]) -> dict:
    config, node_name = PipelineManager().get_model_para()

    cand_node = get_last_node_result(execution_history, "candidate_generate")
    if not cand_node or cand_node.get("status") != "success":
        raise RuntimeError(
            "query_optimizer: candidate_generate result missing or failed; "
            f"status={cand_node.get('status') if cand_node else 'missing'}"
        )

    sql_candidates: list[str] = cand_node.get("SQL", [])
    optimized_candidates: list[str] = []
    optimizer_trace: list[dict] = []

    question_id = getattr(task, "question_id", "unknown")
    db_id = getattr(task, "db_id", "unknown")
    question = getattr(task, "question", "")

    for i, sql in enumerate(sql_candidates):
        if needs_optimization(sql):
            points = extract_points(sql)
            logging.info(
                "query_optimizer: candidate %d — %d geo points (threshold=%d), mechanical transform",
                i,
                len(points),
                POINT_COUNT_THRESHOLD,
            )
            rewritten, transformed = mechanical_optimize(sql)

            if not transformed:
                _log_failure(
                    question_id=question_id,
                    db_id=db_id,
                    question=question,
                    candidate_index=i,
                    sql=sql,
                    failure_type="no_transform",
                    rewritten=None,
                )
                optimized_candidates.append(sql)
                optimizer_trace.append(
                    {
                        "candidate_index": i,
                        "triggered": True,
                        "method": "mechanical",
                        "point_count": len(points),
                        "transformed": False,
                        "failure_type": "no_transform",
                        "fallback": True,
                    }
                )
            elif not verify_mechanical_transform(sql, rewritten):
                _log_failure(
                    question_id=question_id,
                    db_id=db_id,
                    question=question,
                    candidate_index=i,
                    sql=sql,
                    failure_type="verify_mismatch",
                    rewritten=rewritten,
                )
                optimized_candidates.append(sql)
                optimizer_trace.append(
                    {
                        "candidate_index": i,
                        "triggered": True,
                        "method": "mechanical",
                        "point_count": len(points),
                        "transformed": True,
                        "failure_type": "verify_mismatch",
                        "fallback": True,
                    }
                )
            else:
                optimized_candidates.append(rewritten)
                optimizer_trace.append(
                    {
                        "candidate_index": i,
                        "triggered": True,
                        "method": "mechanical",
                        "point_count": len(points),
                        "transformed": True,
                        "verified": True,
                    }
                )
        else:
            optimized_candidates.append(sql)
            optimizer_trace.append({"candidate_index": i, "triggered": False})

    logging.info(
        "query_optimizer: done — triggered=%s/%d candidates",
        sum(1 for e in optimizer_trace if e["triggered"]),
        len(sql_candidates),
    )

    return {
        "SQL": optimized_candidates,
        "optimizer_trace": optimizer_trace,
    }
