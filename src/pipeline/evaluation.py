import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.utils import get_last_node_result, node_decorator
from runner.check_and_correct import sql_raw_parse
from runner.database_manager import DatabaseManager
from runner.execution import compare_with_cached_gold, execute_sql
from runner.gold_sql_cache import GoldSqlCache
from runner.logger import Logger

# ---------------------------------------------------------------------------
# Module-level gold cache — set once at pipeline startup via set_gold_cache()
# ---------------------------------------------------------------------------
_gold_cache: Optional[GoldSqlCache] = None


def set_gold_cache(cache: GoldSqlCache) -> None:
    """Inject a pre-loaded GoldSqlCache for use by all evaluation calls."""
    global _gold_cache
    _gold_cache = cache


def load_gold_cache_from_env() -> None:
    """Attempt to load the gold cache from env vars / default path.

    Called automatically on first evaluation if no cache has been set.
    Uses GOLD_SQL_CACHE_PATH env var, or derives the default path from
    GOLD_SQL_DATASET and GOLD_SQL_DB_ID (both optional, default to
    'test_data_point' and 'meteo').
    """
    global _gold_cache
    if _gold_cache is not None:
        return

    explicit = os.environ.get("GOLD_SQL_CACHE_PATH")
    if explicit:
        path = Path(explicit)
    else:
        dataset = os.environ.get("GOLD_SQL_DATASET", "test_data_point")
        db_id   = os.environ.get("GOLD_SQL_DB_ID",   "meteo")
        # Walk up from this file to find the project root (contains pyproject.toml)
        here = Path(__file__).resolve()
        project_root = here
        for _ in range(10):
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent
        path = GoldSqlCache.cache_path(project_root, dataset, db_id)

    if path.exists():
        _gold_cache = GoldSqlCache(path)
        logging.info("Gold SQL cache loaded: %s (%d entries)",
                     path, len(_gold_cache._raw))
    else:
        logging.warning(
            "Gold SQL cache not found at %s — falling back to live gold execution.",
            path,
        )
        _gold_cache = GoldSqlCache.__new__(GoldSqlCache)
        _gold_cache._metadata = {}
        _gold_cache._raw = {}
        _gold_cache._lru = __import__("cachetools").LRUCache(maxsize=200)
        _gold_cache._lock = __import__("threading").Lock()


@node_decorator(check_schema_status=False)
def evaluation(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate predicted SQL queries against the ground-truth SQL.

    When a gold SQL cache is available (set via set_gold_cache() or auto-loaded
    from env), gold results are read from the cache instead of re-executing the
    gold SQL.  Only the *predicted* SQL is executed live, which eliminates the
    long gold-SQL timeouts that caused most eval errors.
    """
    # Ensure cache is initialised (no-op if already done)
    load_gold_cache_from_env()

    ground_truth_sql = task.SQL
    q_id             = task.question_id
    question         = getattr(task, "question", "") or ""

    # ------------------------------------------------------------------
    # Gold result: cache lookup by question text (stable across resamples)
    # ------------------------------------------------------------------
    cached_entry  = _gold_cache.get_entry_by_question(question) if _gold_cache is not None else None
    gold_cache_ok = cached_entry is not None and cached_entry.get("status") == "success"

    if gold_cache_ok:
        gold_result = cached_entry["result"]   # list of lists, JSON-serialisable
        gold_set    = _gold_cache.get_gold_set_by_question(question)
        t_gold      = _gold_cache.get_duration_by_question(question) or 0.0
    else:
        gold_result = None
        gold_set    = None
        t_gold      = None
        try:
            gold_result = list(execute_sql(ground_truth_sql, fetch="all", timeout_s=180))
        except Exception as _e:
            gold_result = f"gold_exec_error: {_e}"

    # ------------------------------------------------------------------
    # Predicted SQL sources
    # ------------------------------------------------------------------
    to_evaluate = {
        "candidate_generate": get_last_node_result(execution_history, "candidate_generate"),
        "align_correct":      get_last_node_result(execution_history, "align_correct"),
        "vote":               get_last_node_result(execution_history, "vote"),
    }

    result = {}
    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql    = "--"
        evaluation_result: Dict[str, Any] = {}

        try:
            if node_result["status"] == "success":
                if evaluation_for == "align":
                    predicted_sql = node_result["SQL_align_vote"]
                elif evaluation_for == "correct":
                    predicted_sql = node_result["SQL_correct_vote"]
                elif evaluation_for == "align_correct":
                    vote_all      = node_result["vote"]
                    predicted_sql = vote_all[0]["sql"]
                elif evaluation_for == "candidate_generate":
                    candidate_all = node_result["SQL"]
                    predicted_sql = sql_raw_parse(candidate_all[0], False)[0]
                elif evaluation_for == "vote":
                    predicted_sql = node_result["SQL"]

                # ----------------------------------------------------------
                # Comparison: use cache when available
                # ----------------------------------------------------------
                if gold_cache_ok and gold_set is not None:
                    response = compare_with_cached_gold(
                        predicted_sql=predicted_sql,
                        gold_set=gold_set,
                        t_gold=t_gold,
                        meta_time_out=600,
                    )
                else:
                    response = DatabaseManager().compare_sqls(
                        predicted_sql=predicted_sql,
                        ground_truth_sql=ground_truth_sql,
                        meta_time_out=180,
                    )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                    "ves":      response.get("ves", 0.0),
                })
            else:
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result["error"],
                    "ves":      0.0,
                })
        except Exception as e:
            Logger().log(
                f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                "error",
            )
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
                "ves":      0.0,
            })

        # ------------------------------------------------------------------
        # Execute predicted SQL for result inspection
        # ------------------------------------------------------------------
        predicted_result = None
        try:
            predicted_result = list(execute_sql(predicted_sql, fetch="all", timeout_s=30))
        except Exception as _pe:
            predicted_result = f"pred_exec_error: {_pe}"

        evaluation_result.update({
            "Question":        task.raw_question,
            "Evidence":        task.evidence,
            "GOLD_SQL":        ground_truth_sql,
            "GOLD_RESULT":     gold_result,
            "PREDICTED_SQL":   predicted_sql,
            "PREDICTED_RESULT": predicted_result,
            "gold_from_cache": gold_cache_ok,
        })
        result[evaluation_for] = evaluation_result

    logging.info("Evaluation completed (cache=%s)", gold_cache_ok)
    return result
