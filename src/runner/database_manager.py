"""DatabaseManager singleton for PostgreSQL.

Replaces the original BIRD/SQLite path-based management with PostgreSQL
connection management using environment variables.
"""

import os
from pathlib import Path
from threading import Lock

import psycopg2

from runner.execution import compare_sqls


class DatabaseManager:
    """Singleton managing PostgreSQL connection and data paths for the Meteo dataset."""

    _instance = None
    _lock = Lock()

    def __new__(cls, db_root_path: str | None = None, db_id: str | None = None):
        with cls._lock:
            if cls._instance is None:
                if db_root_path is None:
                    raise ValueError("DatabaseManager has not been initialized.")
                cls._instance = super().__new__(cls)
                cls._instance._init(db_root_path, db_id or "meteo")
            elif db_root_path is not None and cls._instance.db_root_path != db_root_path:
                cls._instance._init(db_root_path, db_id or "meteo")
            return cls._instance

    def _init(self, db_root_path: str, db_id: str):
        self.db_root_path = db_root_path
        self.db_id = db_id
        self._set_paths()

    def _set_paths(self):
        root = Path(self.db_root_path)
        # Preprocessed data
        self.db_json = root / "data_preprocess" / "dev.json"
        self.db_tables = root / "data_preprocess" / "tables.json"
        # Fewshot
        self.db_fewshot_path = root / "fewshot" / "questions.json"
        self.db_fewshot2_path = root / "correct_fewshot2.json"
        # Embeddings
        self.emb_dir = root / "emb"
        # Schema cache
        self.db_schema_cache = root / "db_schema.json"

    # -- PostgreSQL connection ------------------------------------------------

    @staticmethod
    def get_connection():
        """Return a new psycopg2 connection from env vars with correct schema search_path."""
        # Set search_path to the configured schema (default: era5_land2)
        schema = os.environ.get("DB_SCHEMA", "era5_land2")
        options = f"-c search_path={schema}"
        
        return psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            port=os.environ.get("DB_PORT"),
            password=os.environ.get("DB_PASS"),
            options=options,
        )

    # -- Execution-based comparison -------------------------------------------

    @staticmethod
    def compare_sqls(predicted_sql: str, ground_truth_sql: str, meta_time_out: int = 180):
        """Compare predicted vs ground-truth SQL by executing both.

        Delegates to execution.compare_sqls().
        """
        return compare_sqls(
            predicted_sql=predicted_sql,
            ground_truth_sql=ground_truth_sql,
            meta_time_out=meta_time_out,
        )
