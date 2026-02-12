"""Data preprocessing – Meteo / PostgreSQL version.

Converts the Meteo dataset format to the OpenSearch-SQL pipeline format.
Generates data_preprocess/dev.json and a minimal tables.json from
PostgreSQL introspection.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_pg_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        port=os.environ.get("DB_PORT"),
        password=os.environ.get("DB_PASS"),
    )


def generate_tables_json(output_dir: Path, schema: str = "public"):
    """Generate a tables.json compatible with the pipeline from PostgreSQL.

    The format mirrors the BIRD/Spider tables.json structure:
        [{
            "db_id": "meteo",
            "table_names_original": [...],
            "column_names_original": [[table_idx, col_name], ...],
            "column_names": [[table_idx, col_name], ...],
            "foreign_keys": [[from_col_idx, to_col_idx], ...],
        }]
    """
    conn = _get_pg_connection()
    cur = conn.cursor()

    # Get tables
    cur.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """,
        (schema,),
    )
    tables = [row[0] for row in cur.fetchall()]

    table_names_original = tables
    column_names_original = [[-1, "*"]]  # First entry is always [-1, "*"]
    column_names = [[-1, "*"]]

    col_index_map = {}  # (table_name, col_name) -> index

    for table_idx, table_name in enumerate(tables):
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table_name),
        )
        for row in cur.fetchall():
            col_name = row[0]
            idx = len(column_names_original)
            column_names_original.append([table_idx, col_name])
            column_names.append([table_idx, col_name])
            col_index_map[(table_name, col_name)] = idx

    # Foreign keys
    cur.execute(
        """
        SELECT
            tc.table_name, kcu.column_name,
            ccu.table_name, ccu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s;
        """,
        (schema,),
    )
    foreign_keys = []
    for from_table, from_col, to_table, to_col in cur.fetchall():
        from_idx = col_index_map.get((from_table, from_col))
        to_idx = col_index_map.get((to_table, to_col))
        if from_idx is not None and to_idx is not None:
            foreign_keys.append([from_idx, to_idx])

    cur.close()
    conn.close()

    tables_data = [{
        "db_id": "meteo",
        "table_names_original": table_names_original,
        "table_names": table_names_original,
        "column_names_original": column_names_original,
        "column_names": column_names,
        "foreign_keys": foreign_keys,
    }]

    output_path = output_dir / "tables.json"
    with open(output_path, "w") as f:
        json.dump(tables_data, f, indent=4)
    logging.info(f"Generated tables.json with {len(tables)} tables at {output_path}")


def meteo_preprocess(data_dir: str, input_json: str):
    """Convert Meteo dataset to pipeline format.

    Args:
        data_dir: Root data directory.
        input_json: Path to the Meteo sample JSON file.
    """
    preprocessed_path = Path(data_dir) / "data_preprocess"
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to pipeline format
    pipeline_data = []
    for i, record in enumerate(raw_data):
        entry = {
            "question_id": i,
            "db_id": "meteo",
            "question": record.get("Natural language question", record.get("question", "")),
            "evidence": "",
            "SQL": record.get("SQL query", record.get("SQL", "")),
            "difficulty": record.get("Category", ""),
            "raw_question": record.get("Natural language question", record.get("question", "")),
        }
        pipeline_data.append(entry)

    output_path = preprocessed_path / "dev.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Preprocessed {len(pipeline_data)} questions to {output_path}")

    # Generate tables.json from PostgreSQL
    schema = os.environ.get("DB_SCHEMA", "public")
    generate_tables_json(preprocessed_path, schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Meteo dataset for OpenSearch-SQL pipeline.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory.")
    parser.add_argument("--input_json", type=str, required=True, help="Input Meteo JSON file.")
    parser.add_argument("--env_file", type=str, default=None, help="Path to .env file.")

    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    meteo_preprocess(args.data_dir, args.input_json)
