"""LangGraph workflow: optimize → judge → self_debug loop.

Graph shape
-----------
  optimize ──► judge ──► (is_equivalent or retries_exhausted) ──► END
                  │
                  └──► self_debug ──► judge  (cycles up to max_retries times)

The chat_model is bound at graph-creation time via closures so it does not
pollute the TypedDict state (which must be JSON-serialisable for LangGraph).

Usage
-----
  from pipeline.optimizer.graph import run_optimizer
  final_sql, trace = run_optimizer(sql, chat_model)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from .models import OptimizerState
from .nodes import call_optimize, call_judge, call_self_debug

MAX_DEBUG_RETRIES = 2


def create_optimizer_graph(chat_model: Any, max_retries: int = MAX_DEBUG_RETRIES):
    """Return a compiled LangGraph with *chat_model* bound in node closures."""

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def optimize_node(state: OptimizerState) -> dict:
        optimized = call_optimize(state["original_sql"], chat_model)
        return {
            "current_sql": optimized,
            "trace": state["trace"] + [
                {"attempt": 0, "action": "optimize", "sql": optimized}
            ],
        }

    def judge_node(state: OptimizerState) -> dict:
        is_eq, feedback = call_judge(
            state["original_sql"], state["current_sql"], chat_model
        )
        attempt = state["attempt"] + 1
        return {
            "attempt": attempt,
            "is_equivalent": is_eq,
            "feedback": feedback,
            "trace": state["trace"] + [
                {
                    "attempt": attempt,
                    "action": "judge",
                    "is_equivalent": is_eq,
                    "feedback": feedback,
                }
            ],
        }

    def self_debug_node(state: OptimizerState) -> dict:
        fixed = call_self_debug(
            state["original_sql"],
            state["current_sql"],
            state["feedback"],
            chat_model,
        )
        return {
            "current_sql": fixed,
            "trace": state["trace"] + [
                {"attempt": state["attempt"], "action": "self_debug", "sql": fixed}
            ],
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_after_judge(state: OptimizerState) -> Literal["debug", "accept"]:
        if state["is_equivalent"]:
            return "accept"
        if state["attempt"] >= max_retries:
            logging.warning(
                "optimizer: all %d retries exhausted — fallback will be applied",
                max_retries,
            )
            return "accept"  # caller checks is_equivalent to decide fallback
        return "debug"

    # ── Graph assembly ────────────────────────────────────────────────────────

    workflow = StateGraph(OptimizerState)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("self_debug", self_debug_node)

    workflow.set_entry_point("optimize")
    workflow.add_edge("optimize", "judge")
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {"accept": END, "debug": "self_debug"},
    )
    workflow.add_edge("self_debug", "judge")

    return workflow.compile()


def run_optimizer(
    sql: str, chat_model: Any, max_retries: int = MAX_DEBUG_RETRIES
) -> tuple[str, list[dict]]:
    """Run the full optimize–judge–self_debug loop for one SQL string.

    Returns (final_sql, trace).  Falls back to the original SQL when the judge
    never accepts the rewrite within *max_retries* attempts.
    """
    graph = create_optimizer_graph(chat_model, max_retries)
    initial_state: OptimizerState = {
        "original_sql": sql,
        "current_sql": sql,
        "attempt": 0,
        "is_equivalent": None,
        "feedback": "",
        "trace": [],
    }
    final_state = graph.invoke(initial_state)

    if final_state["is_equivalent"]:
        logging.info("optimizer: rewrite accepted after %d judge call(s)", final_state["attempt"])
        return final_state["current_sql"], final_state["trace"]

    trace = final_state["trace"] + [
        {"action": "fallback", "reason": final_state["feedback"]}
    ]
    logging.warning("optimizer: falling back to original SQL. reason=%s", final_state["feedback"])
    return sql, trace
