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
                    format_type(a.atttypid, a.atttypmod) AS full_type
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
    finally:
        if own_conn:
            conn.close()

    contracts: dict[str, list[str]] = {}
    for table_name, col_name, full_type in rows:
        if not is_table_allowed(table_name):
            continue
        if full_type == "integer[][]":
            contracts[f"{table_name}.{col_name}"] = [
                rule.replace("{col}", col_name) for rule in _INT2D_CONTRACTS
            ]
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
