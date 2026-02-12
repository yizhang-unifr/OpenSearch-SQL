"""Alignment and correction node – PostgreSQL version.

Changed "SQLite" to "PostgreSQL" in the DB info string and removed
db_sqlite_path parameter (PostgreSQL connections are handled internally).
"""

import json
import logging
from typing import Any, Dict, List
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pipeline.utils import node_decorator, get_last_node_result, make_newprompt
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_pg
from llm.prompts import db_check_prompts
from runner.check_and_correct import muti_process_sql, soft_check, sql_raw_parse


@node_decorator(check_schema_status=False)
def align_correct(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path
    correct_fewshot_json = paths.db_fewshot2_path

    prompts_template = db_check_prompts()
    bert_model = SentenceTransformer(config["bert_model"], device=config["device"])

    if fewshot_path.exists():
        with open(fewshot_path) as f:
            df_fewshot = json.load(f)
    else:
        df_fewshot = {"questions": {}}

    chat_model = model_chose(node_name, config["engine"])

    if correct_fewshot_json.exists():
        with open(correct_fewshot_json) as f:
            correct_dic = json.load(f)
    else:
        correct_dic = {"default": ""}

    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"]
    column = get_last_node_result(execution_history, "column_retrieve_and_other_info")["column"]
    foreign_keys = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_keys"]
    foreign_set = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_set"]
    L_values = get_last_node_result(execution_history, "column_retrieve_and_other_info")["L_values"]
    q_order = get_last_node_result(execution_history, "column_retrieve_and_other_info")["q_order"]
    question = get_last_node_result(execution_history, "candidate_generate")["rewrite_question"]

    SQLs = get_last_node_result(execution_history, "candidate_generate")["SQL"]

    db = task.db_id
    hint = task.evidence
    foreign_set = set(foreign_set)
    fewshot = ""
    q_id_str = str(task.question_id)
    if q_id_str in df_fewshot.get("questions", {}):
        fewshot = df_fewshot["questions"][q_id_str].get("prompt", "")

    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    key_col_des = "#Values in Database:\n" + "\n".join(values)

    new_db_info = (
        f"Database Management System: PostgreSQL\n"
        f"#Database name: {db}\n"
        f"{column}\n\n"
        f"#Foreign keys:\n{foreign_keys}\n"
    )

    db_col = {x: all_db_col[x][0] for x in all_db_col}

    SQLs = [sql_raw_parse(x, False)[0] for x in SQLs]
    SQLs_dic = {}
    for x in SQLs:
        SQLs_dic.setdefault(x, 0)
        SQLs_dic[x] += 1

    tmp_prompt = make_newprompt(
        prompts_template.tmp_prompt,
        fewshot,
        key_col_des,
        new_db_info,
        question,
        task.evidence,
        q_order,
    )

    Dcheck = soft_check(
        bert_model,
        chat_model,
        prompts_template.soft_prompt,
        correct_dic,
        prompts_template.correct_prompt,
        prompts_template.vote_prompt,
    )

    vote, none_case = muti_process_sql(
        Dcheck,
        SQLs_dic,
        L_values,
        values,
        question,
        new_db_info,
        hint,
        key_col_des,
        tmp_prompt,
        db_col,
        foreign_set,
        config.get("align_methods", "style_align+function_align"),
        n=config.get("n", 1),
    )

    response = {
        "vote": vote,
        "none_case": none_case,
    }
    return response
