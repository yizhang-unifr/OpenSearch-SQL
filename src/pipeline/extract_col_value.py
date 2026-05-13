import logging
from typing import Any, Dict
from pathlib import Path
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from pipeline.sql_audit.col_value_parser import normalize_col_value_output, structured_output_suffix
from runner.database_manager import DatabaseManager
from llm.model import model_chose
import json
from llm.prompts import *


@node_decorator(check_schema_status=False)
def extract_col_value(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path
    fewshot_enabled = str(config.get("fewshot_enabled", "True")).lower() == "true"
    # Check node config first, then fall back to base LLM config (models.yaml)
    if "use_structured_output" in config:
        use_structured_output = str(config["use_structured_output"]).lower() == "true"
    else:
        from llm.model import _get_base_config
        use_structured_output = bool(_get_base_config().get("use_structured_output", False))
    chat_model = model_chose(node_name, config["engine"])

    if fewshot_enabled and fewshot_path.exists():
        with open(fewshot_path) as f:
            df_fewshot = json.load(f)
    else:
        df_fewshot = {"extract": {}}

    hint = task.evidence
    if hint == "":
        hint = "None"

    all_info = get_last_node_result(execution_history, "generate_db_schema")["db_list"]

    q_id_str = str(task.question_id)
    extract_fewshot = df_fewshot.get("extract", {}).get(q_id_str, {}).get("prompt", "")
    raw = get_des_ans(
        chat_model,
        db_check_prompts().extract_prompt,
        extract_fewshot,
        all_info,
        task.question,
        hint,
        False,
        temperature=config["temperature"],
        use_structured_output=use_structured_output,
    )
    key_col_des_raw = normalize_col_value_output(raw)

    return {"key_col_des_raw": key_col_des_raw}

def get_des_ans(chat_model, ext_prompt, fewshot, db, question, hint, debug,
                temperature=1.0, use_structured_output=False):
    if fewshot:
        fewshot_parts = fewshot.split("/* Answer the following:")[1:6]
        fewshot = "/* Answer the following:" + "/* Answer the following:".join(fewshot_parts)
    ext_prompt = ext_prompt.format(fewshot=fewshot, db_info=db, query=question, hint=hint)
    if use_structured_output:
        ext_prompt += structured_output_suffix()
    if debug:
        print(ext_prompt)
    return chat_model.get_ans(ext_prompt, temperature, debug=debug).replace('```', '')


