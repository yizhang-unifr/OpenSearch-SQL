"""Generate column-value embeddings – PostgreSQL version.

Replaces SQLite-based value extraction with PostgreSQL queries.
Embeddings are stored as gzip-compressed pickle files in the emb/ directory.
"""

import os
import re
import gzip
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import tqdm
import torch
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

uuid_pattern = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _get_pg_connection():
    """Create a PostgreSQL connection from environment variables."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        port=os.environ.get("DB_PORT"),
        password=os.environ.get("DB_PASS"),
    )


def filter_column(table_df, col, exclude_num=True, num_shold=6000):
    cols = table_df[col].unique()
    if len(cols) > num_shold and exclude_num:
        try:
            table_df[col].dropna().astype(float)
            return []
        except (ValueError, TypeError):
            pass
    col_vals = [
        item for item in cols
        if isinstance(item, str) and not uuid_pattern.match(item) and len(item) < 100
    ]
    return col_vals


def make_emb_pg(db_name, DB_emb, col_values, bert_model, exclude_int=True):
    """Generate embeddings for all string column values from PostgreSQL."""
    schema = os.environ.get("DB_SCHEMA", "public")
    conn = _get_pg_connection()
    cur = conn.cursor()

    # Get all tables
    cur.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """,
        (schema,),
    )
    excluded_raw = os.environ.get("EXCLUDE_TABLES", "")
    excluded = {t.strip() for t in excluded_raw.split(",") if t.strip()}
    tables = [row[0] for row in cur.fetchall() if row[0] not in excluded]
    if excluded:
        logging.info(f"Database: {db_name}, tables: {len(tables)} (excluded {len(excluded)}: {excluded})")
    else:
        logging.info(f"Database: {db_name}, tables: {len(tables)}")

    for table_name in tables:
        logging.info(f"Processing table: {table_name}")
        # Sample rows (limit to keep embedding manageable)
        try:
            df = pd.read_sql_query(
                f'SELECT * FROM "{table_name}" LIMIT 10000;', conn
            )
        except Exception as e:
            logging.warning(f"Error reading {table_name}: {e}")
            continue

        # Only string columns
        str_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in tqdm.tqdm(str_cols, desc=table_name):
            col_vals = filter_column(df, col, exclude_int)
            if len(col_vals) == 0:
                continue
            train_embeddings = bert_model.encode(col_vals, device=device)
            DB_emb[f"{table_name}.{col}"] = train_embeddings
            col_values[f"{table_name}.{col}"] = col_vals

    cur.close()
    conn.close()


def save_emb(dicts, dbname, emb_dir):
    os.makedirs(emb_dir, exist_ok=True)
    with gzip.open(os.path.join(emb_dir, f"{dbname}.pkl.gz"), "wb") as pkl_file:
        pickle.dump(dicts, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_emb(dbname, emb_dir):
    with gzip.open(os.path.join(emb_dir, f"{dbname}.pkl.gz"), "rb") as pkl_file:
        data = pickle.load(pkl_file)
    with gzip.open(os.path.join(emb_dir, f"{dbname}_value.pkl.gz"), "rb") as pkl_file:
        col_vs = pickle.load(pkl_file)
    return data, col_vs


def make_emb_all(emb_dir, bert_model_name, db_name="meteo"):
    """Generate embeddings for the Meteo database."""
    os.makedirs(emb_dir, exist_ok=True)
    bert_model = SentenceTransformer(bert_model_name, device=device)

    DB_emb = {}
    col_values = {}
    make_emb_pg(db_name, DB_emb, col_values, bert_model)
    save_emb(DB_emb, db_name, emb_dir)
    save_emb(col_values, f"{db_name}_value", emb_dir)
    logging.info(f"Embeddings saved to {emb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate column-value embeddings from PostgreSQL.")
    parser.add_argument("--emb_dir", type=str, required=True, help="Output directory for embeddings.")
    parser.add_argument("--bert_model", type=str, default="all-mpnet-base-v2", help="SentenceTransformer model name.")
    parser.add_argument("--db_name", type=str, default="meteo", help="Logical database name.")
    parser.add_argument("--env_file", type=str, default=None, help="Path to .env file.")

    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    make_emb_all(args.emb_dir, args.bert_model, args.db_name)
