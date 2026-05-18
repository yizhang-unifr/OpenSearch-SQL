"""Build natural-language usage contracts from PostgreSQL column metadata.

Queries pg_attribute + format_type() to detect columns with special types
(e.g. integer[][]) and emits hard usage rules for the LLM SQL reviewer.
"""

from __future__ import annotations

import os
from typing import Any

from database_process.table_whitelist import is_table_allowed
from runner.execution import _get_pg_connection

# Contracts for 2-D integer arrays (int4[][]).
# {col} is replaced with the actual column name at runtime.
_INT2D_CONTRACTS: list[str] = [
    "This column is an integer[][] (2-D integer array). NEVER use UNNEST() — it flattens to individual scalars, not (code, count) pairs.",
    "To iterate all entries use: CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(<alias>.{col}, 1) AS i",
    "Access element code with <alias>.{col}[i][1] and count/pixels with <alias>.{col}[i][2].",
    "For the single dominant (top-ranked) entry only, direct subscript <alias>.{col}[1][1] is correct.",
]


def _detect_decimal_precision(conn, table: str, col: str, schema: str) -> int | None:
    """Sample up to 200 non-null values and return the most common decimal digit count, or None."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT LENGTH(SPLIT_PART({col}::text, '.', 2))
                FROM {schema}.{table}
                WHERE {col} IS NOT NULL
                LIMIT 200
                """,
            )
            rows = cur.fetchall()
        if not rows:
            return None
        from collections import Counter
        counts = Counter(r[0] for r in rows if r[0] is not None)
        return counts.most_common(1)[0][0] if counts else None
    except Exception:
        return None


def build_column_contracts(conn=None, schema: str | None = None) -> dict[str, list[str]]:
    """Detect special column types and return hard usage contracts per column.

    Args:
        conn:   Optional existing psycopg2 connection (closed by caller).
                If None, a new connection is opened and closed internally.
        schema: PostgreSQL schema name; defaults to DB_SCHEMA env var or "public".

    Returns:
        dict mapping "table.column" to a list of natural-language constraint strings.
    """
    if schema is None:
        schema = os.environ.get("DB_SCHEMA", "public")

    own_conn = conn is None
    if own_conn:
        conn = _get_pg_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.relname  AS table_name,
                    a.attname  AS column_name,
                    format_type(a.atttypid, a.atttypmod) AS full_type,
                    a.attndims AS ndims
                FROM pg_attribute a
                JOIN pg_class     c ON c.oid  = a.attrelid
                JOIN pg_namespace n ON n.oid  = c.relnamespace
                WHERE n.nspname   = %s
                  AND a.attnum    > 0
                  AND NOT a.attisdropped
                ORDER BY c.relname, a.attnum;
                """,
                (schema,),
            )
            rows = cur.fetchall()

        contracts: dict[str, list[str]] = {}
        seen_precision: dict[str, int] = {}  # col_name → precision, detect once across tables

        for table_name, col_name, full_type, ndims in rows:
            if not is_table_allowed(table_name):
                continue

            # 2-D integer arrays: attndims=2; format_type() returns 'integer[]' for both 1D and 2D
            if ndims == 2 and "integer" in full_type:
                contracts[f"{table_name}.{col_name}"] = [
                    rule.replace("{col}", col_name) for rule in _INT2D_CONTRACTS
                ]

            elif col_name in ("latitude", "longitude") and any(t in full_type for t in ("real", "double", "numeric", "float")):
                if col_name not in seen_precision:
                    precision = _detect_decimal_precision(conn, table_name, col_name, schema)
                    if precision is not None:
                        seen_precision[col_name] = precision

        # Emit a single consolidated lat/lon precision note (not per-table) to avoid prompt bloat.
        for col_name, p in seen_precision.items():
            contracts[f"*.{col_name}"] = [
                f"All {col_name} columns are stored at {p} decimal place(s). "
                f"In WHERE clauses, filter with ROUND(CAST(col AS DECIMAL), {p}) when comparing to literal values. "
                f"For JOIN conditions between two table columns at the same stored precision, "
                f"direct equality (t1.{col_name} = t2.{col_name}) is correct and faster — "
                f"do NOT wrap JOIN keys in ROUND()."
            ]

    finally:
        if own_conn:
            conn.close()

    return contracts


def format_contracts_for_prompt(contracts: dict[str, list[str]]) -> str:
    """Render column contracts as a prompt-injectable string."""
    if not contracts:
        return "(none)"
    lines: list[str] = []
    for col_key, rules in contracts.items():
        lines.append(f"Column `{col_key}`:")
        for rule in rules:
            lines.append(f"  - {rule}")
    return "\n".join(lines)
