"""Candidate SQL generation node – PostgreSQL version.

Changed "SQLite" to "PostgreSQL" in the DB info string and updated
the division hint for PostgreSQL syntax.
"""

import json
import logging
from typing import Any, Dict, List

from pipeline.utils import node_decorator, get_last_node_result, make_newprompt
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_pg
from llm.prompts import db_check_prompts
from runner.check_and_correct import get_sql


@node_decorator(check_schema_status=False)
def candidate_generate(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path

    if fewshot_path.exists():
        with open(fewshot_path) as f:
            df_fewshot = json.load(f)
    else:
        df_fewshot = {"questions": {}}

    chat_model = model_chose(node_name, config["engine"])
    column = get_last_node_result(execution_history, "column_retrieve_and_other_info")["column"]
    foreign_keys = get_last_node_result(execution_history, "column_retrieve_and_other_info")["foreign_keys"]
    L_values = get_last_node_result(execution_history, "column_retrieve_and_other_info")["L_values"]
    q_order = get_last_node_result(execution_history, "column_retrieve_and_other_info")["q_order"]
    values = [f"{x[0]}: '{x[1]}'" for x in L_values]
    db = task.db_id

    key_col_des = "#Values in Database:\n" + "\n".join(values)

    new_db_info = (
        f"Database Management System: PostgreSQL\n"
        f"#Database name: {db}\n"
        f"{column}\n\n"
        f"#Foreign keys:\n{foreign_keys}\n"
    )

    question = task.question
    fewshot = ""
    q_id_str = str(task.question_id)
    if q_id_str in df_fewshot.get("questions", {}):
        fewshot = df_fewshot["questions"][q_id_str].get("prompt", "")

    prompts_template = db_check_prompts()
    new_prompt = make_newprompt(
        prompts_template.new_prompt,
        fewshot,
        key_col_des,
        new_db_info,
        question,
        task.evidence,
        q_order,
    )

    single = str(config.get("single", "True")).lower() == "true"
    return_question = str(config.get("return_question", "True")).lower() == "true"
    SQL, _ = get_sql(
        chat_model,
        new_prompt,
        config.get("temperature", 0.7),
        return_question=return_question,
        n=config.get("n", 1),
        single=single,
    )

    response = {
        "rewrite_question": question,
        "SQL": SQL,
    }
    return response


def rewrite_question(question):
    """Adapt division hint for PostgreSQL."""
    if question.find(" / ") != -1:
        question += ". For division operations, use CAST(xxx AS NUMERIC) or xxx::numeric to ensure precise decimal results"
    return question
