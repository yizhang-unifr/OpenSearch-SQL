"""Generate DB schema node – PostgreSQL version.

Introspects the PostgreSQL database directly instead of reading from
SQLite files and BIRD tables.json.
"""

import os
import json
import logging
from typing import Any, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import db_agent_string
from pipeline.review.schema_contract_builder import build_column_contracts


@node_decorator(check_schema_status=False)
def generate_db_schema(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()

    bert_model = SentenceTransformer(
        config["bert_model"],
        device=config["device"],
        local_files_only=True,
    )
    chat_model = model_chose(node_name, config["engine"])
    cache_file = paths.db_schema_cache

    # Check cache
    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    DB_info_agent = db_agent_string(chat_model)
    db = task.db_id

    existing_entry = data.get(db)
    if existing_entry and len(existing_entry) >= 3:
        all_info, db_col, column_contracts = existing_entry
    elif existing_entry:
        # Migrate old 2-element cache entries
        all_info, db_col = existing_entry
        column_contracts = _safe_build_contracts()
        data[db] = [all_info, db_col, column_contracts]
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:
        all_info, db_col = DB_info_agent.get_allinfo(db, bert_model)
        column_contracts = _safe_build_contracts()
        data[db] = [all_info, db_col, column_contracts]
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    response = {
        "db_list": all_info,
        "db_col_dic": db_col,
        "column_contracts": column_contracts,
    }
    return response


def _safe_build_contracts() -> dict:
    try:
        return build_column_contracts()
    except Exception as exc:
        logging.warning("schema_contract_builder failed, continuing without contracts: %s", exc)
        return {}
