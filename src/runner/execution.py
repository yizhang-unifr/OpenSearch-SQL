"""PostgreSQL execution and comparison utilities.

Replaces the original SQLite-based execution with psycopg2 for PostgreSQL.
Connection parameters are read from environment variables (DB_HOST, DB_NAME,
DB_USER, DB_PORT, DB_PASS) loaded by the parent project's .env.
"""

import logging
import os
import re
import time
from decimal import Decimal
from typing import Any, Dict, List, Union

import psycopg2
from func_timeout import FunctionTimedOut, func_timeout

_NUMERIC_TYPES = (int, float, Decimal)
_EPS = 1e-6


def _val_approx_eq(a, b) -> bool:
    """True if a == b, or both are numeric and differ by less than _EPS."""
    if a == b:
        return True
    if isinstance(a, _NUMERIC_TYPES) and isinstance(b, _NUMERIC_TYPES):
        try:
            return abs(float(a) - float(b)) < _EPS
        except (TypeError, ValueError):
            pass
    return False


def _row_approx_eq(row_a, row_b) -> bool:
    """Column-order-independent row comparison with epsilon tolerance."""
    if len(row_a) != len(row_b):
        return False
    used = [False] * len(row_b)
    for a in row_a:
        matched = False
        for i, b in enumerate(row_b):
            if not used[i] and _val_approx_eq(a, b):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _results_approx_equal(pred_res: set, gold_res: set) -> bool:
    """Row-order-independent comparison with epsilon tolerance for numerics."""
    if len(pred_res) != len(gold_res):
        return False
    gold_list = list(gold_res)
    used = [False] * len(gold_list)
    for p_row in pred_res:
        matched = False
        for i, g_row in enumerate(gold_list):
            if not used[i] and _row_approx_eq(p_row, g_row):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _sql_statement_timeout_ms() -> int:
    """Read SQL_STATEMENT_TIMEOUT from env (seconds) and return milliseconds."""
    return int(os.environ.get("SQL_STATEMENT_TIMEOUT", "30")) * 1000


def _get_pg_connection(statement_timeout_ms: int | None = None):
    """Create a PostgreSQL connection from environment variables.

    Args:
        statement_timeout_ms: Per-statement timeout in milliseconds.
            Defaults to ``SQL_STATEMENT_TIMEOUT`` env var (seconds × 1000).
    """
    if statement_timeout_ms is None:
        statement_timeout_ms = _sql_statement_timeout_ms()
    
    # Set search_path to the configured schema (default: era5_land2)
    schema = os.environ.get("DB_SCHEMA", "era5_land2")
    options = f"-c statement_timeout={statement_timeout_ms} -c search_path={schema}"
    
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        port=os.environ.get("DB_PORT"),
        password=os.environ.get("DB_PASS"),
        options=options,
    )
    return conn


def _normalize_sql(sql: str) -> str:
    """Strip schema qualifications so SQL runs against the default search_path.

    Strips any ``<schema>.`` prefix before ``meteo_`` table names.
    """
    schema = os.environ.get("DB_SCHEMA", "public")
    if schema and schema != "public":
        sql = re.sub(rf"\b{re.escape(schema)}\.", "", sql)
    # Also strip legacy era5_land2 prefix
    sql = re.sub(r"\b\w+\.(meteo_)", r"\1", sql)
    return sql


def execute_sql(sql: str, fetch: Union[str, int] = "all", timeout_s: int | None = None) -> Any:
    """Execute SQL against PostgreSQL and fetch results.

    Args:
        sql: The SQL query to execute.
        fetch: How to fetch results - "all", "one", or an integer.
        timeout_s: Statement timeout in seconds.

    Returns:
        Fetched results based on the fetch argument.
    """
    if timeout_s is None:
        timeout_s = int(os.environ.get("SQL_STATEMENT_TIMEOUT", "30"))
    sql = _normalize_sql(sql)
    try:
        conn = _get_pg_connection()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{timeout_s * 1000}';")
            cur.execute(sql)
            if fetch == "all":
                return cur.fetchall()
            elif fetch == "one":
                return cur.fetchone()
            elif isinstance(fetch, int):
                return cur.fetchmany(fetch)
            else:
                raise ValueError(f"Invalid fetch argument: {fetch}")
    except Exception as e:
        logging.error(f"Error in execute_sql: {e}\nSQL: {sql}")
        raise
    finally:
        conn.close()


def _compare_sqls_outcomes(predicted_sql: str, ground_truth_sql: str) -> int:
    """Compare the outcomes of two SQL queries for equivalence.

    Returns:
        1 if the result sets are equivalent, 0 otherwise.
    """
    try:
        predicted_res = set(execute_sql(predicted_sql))
        ground_truth_res = set(execute_sql(ground_truth_sql))
        return int(
            predicted_res == ground_truth_res
            or _results_approx_equal(predicted_res, ground_truth_res)
        )
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise


def _compare_sqls_with_timing(
    predicted_sql: str, ground_truth_sql: str
) -> tuple[int, str, float]:
    """Execute both SQLs, compare result sets, and return (exec_res, exec_err, ves).

    VES (Valid Efficiency Score, BIRD benchmark):
        ves = sqrt(min(t_gold / t_pred, 1.0))  if result sets are equal
        ves = 0.0                               otherwise

    Returns:
        (exec_res, exec_err, ves)
    """
    import math

    try:
        gold_res, t_gold = sql_exec(ground_truth_sql)
    except Exception as e:
        return 0, f"gold_sql_error: {e}", 0.0

    try:
        pred_res, t_pred = sql_exec(predicted_sql)
    except Exception as e:
        return 0, str(e), 0.0

    correct = int(pred_res == gold_res or _results_approx_equal(pred_res, gold_res))
    error = "--" if correct else "incorrect answer"

    if correct and t_pred > 0:
        ratio = min(t_gold / t_pred, 1.0)
        ves = math.sqrt(ratio)
    elif correct:
        ves = 1.0
    else:
        ves = 0.0

    return correct, error, ves


def compare_sqls(
    predicted_sql: str,
    ground_truth_sql: str,
    meta_time_out: int | None = None,
) -> Dict[str, Union[int, str, float]]:
    """Compare predicted SQL with ground truth SQL within a timeout.

    Returns:
        Dict with 'exec_res' (1=correct, 0=incorrect), 'exec_err', and
        'ves' (Valid Efficiency Score per BIRD benchmark).
    """
    if meta_time_out is None:
        meta_time_out = int(os.environ.get("SQL_COMPARE_TIMEOUT", "60"))
    try:
        res, error, ves = func_timeout(
            meta_time_out,
            _compare_sqls_with_timing,
            args=(predicted_sql, ground_truth_sql),
        )
    except FunctionTimedOut:
        logging.warning("Comparison timed out.")
        error = "timeout"
        res = 0
        ves = 0.0
    except Exception as e:
        logging.error(f"Error in compare_sqls: {e}")
        error = str(e)
        res = 0
        ves = 0.0
    return {"exec_res": res, "exec_err": error, "ves": ves}


def validate_sql_query(sql: str, max_returned_rows: int = 30) -> Dict[str, Any]:
    """Validate an SQL query by executing it.

    Returns:
        Dict with 'SQL', 'RESULT', and 'STATUS'.
    """
    try:
        result = execute_sql(sql, fetch=max_returned_rows)
        return {"SQL": sql, "RESULT": result, "STATUS": "OK"}
    except Exception as e:
        logging.error(f"Error in validate_sql_query: {e}")
        return {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}


def aggregate_sqls(sqls: List[str]) -> str:
    """Aggregate multiple SQL queries by clustering on result sets.

    Picks the shortest SQL from the largest cluster of equivalent queries.
    """
    results = [validate_sql_query(sql) for sql in sqls]
    clusters: Dict[Any, List[str]] = {}

    for result in results:
        if result["STATUS"] == "OK":
            key = frozenset(tuple(row) for row in result["RESULT"])
            clusters.setdefault(key, []).append(result["SQL"])

    if clusters:
        largest_cluster = max(clusters.values(), key=len, default=[])
        if largest_cluster:
            return min(largest_cluster, key=len)

    logging.warning("No valid SQL clusters found. Returning the first SQL query.")
    return sqls[0]


def sql_exec(sql: str) -> tuple:
    """Execute SQL and return (result_set, time_cost)."""
    sql = _normalize_sql(sql)
    conn = _get_pg_connection()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{_sql_statement_timeout_ms()}';")
            s = time.time()
            cur.execute(sql)
            rows = cur.fetchall()
            ans = set(tuple(x) for x in rows)
            time_cost = time.time() - s
        return ans, time_cost
    finally:
        conn.close()
