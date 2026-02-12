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


@node_decorator(check_schema_status=False)
def generate_db_schema(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()

    bert_model = SentenceTransformer(config["bert_model"], device=config["device"])
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
    if existing_entry:
        all_info, db_col = existing_entry
    else:
        all_info, db_col = DB_info_agent.get_allinfo(db, bert_model)
        data[db] = [all_info, db_col]
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    response = {
        "db_list": all_info,
        "db_col_dic": db_col,
    }
    return response
