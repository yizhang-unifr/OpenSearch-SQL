"""Database schema introspection – PostgreSQL version.

Replaces SQLite PRAGMA / sqlite_master queries with PostgreSQL
information_schema and system catalog queries.
"""

import os
import re

import pandas as pd

from runner.execution import _get_pg_connection


def _excluded_tables() -> set[str]:
    """Return the set of table names to exclude (from EXCLUDE_TABLES env var)."""
    raw = os.environ.get("EXCLUDE_TABLES", "")
    if not raw:
        return set()
    return {t.strip() for t in raw.split(",") if t.strip()}


def find_foreign_keys_pg() -> tuple:
    """Discover foreign keys from PostgreSQL system catalog.

    Returns:
        (foreign_keys_str, col_set) where foreign_keys_str is a human-readable
        string like ``table1.col = table2.col, …`` and col_set is the set of
        involved ``table.column`` identifiers.
    """
    sql = """
    SELECT
        tc.table_name AS from_table,
        kcu.column_name AS from_column,
        ccu.table_name AS to_table,
        ccu.column_name AS to_column
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = tc.constraint_name
        AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = %s;
    """
    schema = os.environ.get("DB_SCHEMA", "public")
    conn = _get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (schema,))
            rows = cur.fetchall()
    finally:
        conn.close()

    excluded = _excluded_tables()
    output = []
    col_set = set()
    for from_table, from_col, to_table, to_col in rows:
        if from_table in excluded or to_table in excluded:
            continue
        output.append(f"{from_table}.{from_col} = {to_table}.{to_col}")
        col_set.add(f"{from_table}.{from_col}")
        col_set.add(f"{to_table}.{to_col}")
    return ", ".join(output), col_set


def quote_field(field_name: str) -> str:
    if re.search(r"\W", field_name):
        return f'"{field_name}"'
    return field_name


class db_agent:

    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    def get_allinfo(self, db_name: str, model):
        """Build full schema info for the database.

        Args:
            db_name: Logical database name (e.g. 'meteo').
            model: SentenceTransformer model (for embedding similarity, kept for compatibility).
        """
        db_info, db_col = self.get_db_des()
        foreign_keys, _ = find_foreign_keys_pg()
        all_info = (
            f"Database Management System: PostgreSQL\n"
            f"#Database name: {db_name}\n"
            f"{db_info}\n"
            f"#Foreign keys:\n{foreign_keys}\n"
        )
        prompt = self.db_conclusion(all_info)
        db_all = self.chat_model.get_ans(prompt)
        all_info = f"{all_info}\n{db_all}\n"
        return all_info, db_col

    def get_complete_table_info(self, conn, table_name: str):
        """Get column-level metadata for a table from PostgreSQL."""
        schema = os.environ.get("DB_SCHEMA", "public")
        cur = conn.cursor()

        # Column info via information_schema
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table_name),
        )
        columns_info = cur.fetchall()

        # Sample rows (limited for large tables)
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT 100;')
        sample_rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(sample_rows, columns=col_names)

        contains_null = {col: df[col].isnull().any() for col in df.columns}
        contains_duplicates = {col: df[col].duplicated().any() for col in df.columns}

        # Build schema string
        schema_str = f"## Table {table_name}:\nColumn| Type| 3 Example Values\n"
        columns = {}

        for col_info in columns_info:
            column_name, data_type, is_nullable, default_val = col_info
            qname = quote_field(column_name)

            # Sample values
            if column_name in df.columns:
                vals = df[column_name].dropna().drop_duplicates().iloc[:3].tolist()
            else:
                vals = []

            include_null = "Include Null" if contains_null.get(column_name, False) else "Non-Null"
            unique = "Non-Unique" if contains_duplicates.get(column_name, False) else "Unique"

            if len(str(vals)) > 360:
                vals_str = "<Long text not displayed>"
            else:
                vals_str = str(vals)

            schema_str += f"{qname}| {data_type}| {vals_str}\n"
            columns[f"{table_name}.{qname}"] = ("", "", data_type, include_null, unique, vals_str)

        cur.close()
        return schema_str, columns

    def get_db_des(self):
        """Get full schema description from PostgreSQL."""
        schema = os.environ.get("DB_SCHEMA", "public")
        conn = _get_pg_connection()
        cur = conn.cursor()

        # List all tables
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """,
            (schema,),
        )
        excluded = _excluded_tables()
        tables = [row[0] for row in cur.fetchall() if row[0] not in excluded]
        cur.close()

        db_info = []
        db_col = {}
        for table_name in tables:
            table_info, columns = self.get_complete_table_info(conn, table_name)
            db_info.append(table_info)
            db_col.update(columns)

        conn.close()
        return "\n".join(db_info), db_col

    def db_conclusion(self, db_info):
        prompt = f"""/* Here is an example about describing a database */
#Foreign keys: 
Airlines.ORIGIN = Airports.Code, Airlines.DEST = Airports.Code
#Database Description: The database encompasses information related to flights.
#Tables Descriptions:
Air Carriers: Codes and descriptive information about airlines
Airports: IATA codes and descriptions of airports
Airlines: Detailed information about flights

/* Describe the following database */
{db_info}
Please conclude the database in the following format:
#Database Description:
#Tables Descriptions:
"""
        return prompt


class db_agent_string(db_agent):
    """Variant that produces richer, natural-language column descriptions."""

    def __init__(self, chat_model) -> None:
        super().__init__(chat_model)

    def get_complete_table_info(self, conn, table_name: str):
        schema = os.environ.get("DB_SCHEMA", "public")
        cur = conn.cursor()

        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table_name),
        )
        columns_info = cur.fetchall()

        cur.execute(f'SELECT * FROM "{table_name}" LIMIT 100;')
        sample_rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(sample_rows, columns=col_names)

        contains_null = {col: df[col].isnull().any() for col in df.columns}
        contains_duplicates = {col: df[col].duplicated().any() for col in df.columns}

        schema_str = f"## Table {table_name}:\n"
        columns = {}

        for col_info in columns_info:
            column_name, data_type, is_nullable, default_val = col_info
            qname = quote_field(column_name)
            schema_str_single = ""

            schema_str_single += f"The type is {data_type}, "
            if contains_null.get(column_name, False):
                schema_str_single += "which includes Null"
            else:
                schema_str_single += "which does not include Null"
            if contains_duplicates.get(column_name, False):
                schema_str_single += " and is Non-Unique. "
            else:
                schema_str_single += " and is Unique. "

            include_null = "Include Null" if contains_null.get(column_name, False) else "Non-Null"
            unique = "Non-Unique" if contains_duplicates.get(column_name, False) else "Unique"

            # Sample values
            if column_name in df.columns:
                df_tmp = df[column_name].dropna().drop_duplicates()
                if len(df_tmp) >= 3:
                    vals = df_tmp.sample(3).values.tolist()
                else:
                    vals = df_tmp.values.tolist()
            else:
                vals = []

            if len(str(vals)) > 360:
                vals_str = "<Long text>"
                schema_str_single += "Values format: <Long text>"
            elif not isinstance(vals, list) or len(vals) < 3:
                schema_str_single += f"Value of this column must in: {vals}"
            else:
                schema_str_single += f"Values format like: {vals}"

            schema_str += f"{qname}: {schema_str_single}\n"
            columns[f"{table_name}.{qname}"] = (
                schema_str_single,
                "",
                "",
                data_type,
                include_null,
                unique,
                str(vals),
            )

        cur.close()
        return schema_str, columns
