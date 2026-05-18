"""Query-optimizer pipeline node.

Sits between candidate_generate and align_correct.  For each candidate SQL
that contains a large ROUND(CAST(lat/lon)) IN (...) predicate (> 20 pairs),
runs an LLM rewrite loop to convert the IN list to a VALUES-based JOIN,
verified by an independent LLM judge with self-debugging on mismatch.

This package is fully self-contained and does NOT read from any upstream
context nodes (geo_context, implicit_context, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from llm.model import model_chose

from .detector import extract_points, needs_optimization, POINT_COUNT_THRESHOLD
from .graph import run_optimizer


@node_decorator(check_schema_status=False)
def query_optimizer(task: Any, execution_history: list[dict]) -> dict:
    config, node_name = PipelineManager().get_model_para()
    chat_model = model_chose(node_name, config.get("engine", ""))

    cand_node = get_last_node_result(execution_history, "candidate_generate")
    if not cand_node or cand_node.get("status") != "success":
        raise RuntimeError(
            "query_optimizer: candidate_generate result missing or failed; "
            f"status={cand_node.get('status') if cand_node else 'missing'}"
        )

    sql_candidates: list[str] = cand_node.get("SQL", [])
    optimized_candidates: list[str] = []
    optimizer_trace: list[dict] = []

    for i, sql in enumerate(sql_candidates):
        if needs_optimization(sql):
            points = extract_points(sql)
            logging.info(
                "query_optimizer: candidate %d — %d geo points (threshold=%d), optimizing",
                i,
                len(points),
                POINT_COUNT_THRESHOLD,
            )
            final_sql, trace = run_optimizer(sql, chat_model)
            optimized_candidates.append(final_sql)
            optimizer_trace.append(
                {
                    "candidate_index": i,
                    "triggered": True,
                    "point_count": len(points),
                    "trace": trace,
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
