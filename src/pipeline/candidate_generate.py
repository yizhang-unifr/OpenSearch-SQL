"""Candidate SQL generation node – PostgreSQL version."""

import json
import logging
from typing import Any, Dict, List

from pipeline.utils import node_decorator, get_last_node_result, make_newprompt
from pipeline.context.implicit_context_utils import build_implicit_context_block, get_implicit_context_payload
from pipeline.context.landcover_semantic_hints import build_landcover_semantic_hint
from pipeline.context.landcover_entity_hint import build_landcover_entity_hint
from pipeline.sql_audit.sql_candidate_parser import parse_sql_candidates
from pipeline.sql_audit.constraint_validator import validate_sql_candidates
from pipeline.pipeline_manager import PipelineManager, get_bool
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
    fewshot_enabled = get_bool(config, "fewshot_enabled", default=True)

    # Ablation flags – all default True so existing "full" runs are unchanged
    enable_entity_hint   = get_bool(config, "enable_entity_hint",   default=True)
    enable_semantic_hint = get_bool(config, "enable_semantic_hint", default=True)
    enable_validator     = get_bool(config, "enable_validator",     default=True)

    if fewshot_enabled and fewshot_path.exists():
        with open(fewshot_path) as f:
            df_fewshot = json.load(f)
    else:
        df_fewshot = {"questions": {}}

    chat_model = model_chose(node_name, config["engine"])
    col_node = get_last_node_result(execution_history, "column_retrieve_and_other_info")
    if not col_node or col_node.get("status") != "success":
        raise RuntimeError(
            "Upstream node 'column_retrieve_and_other_info' failed; "
            f"cannot generate candidate SQL. upstream_error={col_node.get('error') if col_node else 'missing'}"
        )
    column = col_node["column"]
    foreign_keys = col_node["foreign_keys"]
    L_values = col_node["L_values"]
    q_order = col_node["q_order"]
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
    if not fewshot:
        question_key = (task.raw_question if hasattr(task, "raw_question") else question).strip().lower()
        fewshot = df_fewshot.get("by_question", {}).get(question_key, {}).get("prompt", "")

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
    new_prompt += build_implicit_context_block(execution_history)
    if enable_semantic_hint:
        new_prompt += build_landcover_semantic_hint(question)
    if enable_entity_hint:
        new_prompt += build_landcover_entity_hint(question)

    single = get_bool(config, "single", default=True)
    return_question = get_bool(config, "return_question", default=True)
    SQL, _ = get_sql(
        chat_model,
        new_prompt,
        config.get("temperature", 0.7),
        return_question=return_question,
        n=config.get("n", 1),
        single=single,
    )

    raw_candidates = [SQL] if isinstance(SQL, str) else list(SQL)
    parsed_candidates = parse_sql_candidates(raw_candidates)
    structured_candidates = [x.model_dump() for x in parsed_candidates]
    sql_candidates = [x.sql for x in parsed_candidates if x.sql]
    if not sql_candidates:
        logging.warning(
            "candidate_generate produced no parseable SQL candidates; "
            "question_id=%s raw_candidate_count=%s",
            getattr(task, "question_id", "unknown"),
            len(raw_candidates),
        )

    implicit_payload = get_implicit_context_payload(execution_history)
    schema_node = get_last_node_result(execution_history, "generate_db_schema")
    column_contracts = schema_node.get("column_contracts", {}) if schema_node else {}

    if enable_validator and sql_candidates:
        validated_candidates, validation_trace = validate_sql_candidates(
            chat_model,
            sql_candidates,
            column_contracts,
            implicit_payload,
            question,
        )
    else:
        validated_candidates = sql_candidates
        validation_trace = []

    return {
        "rewrite_question": question,
        "SQL": validated_candidates,
        "SQL_raw_candidates": raw_candidates,
        "SQL_structured_candidates": structured_candidates,
        "validation_trace": validation_trace,
        # legacy key kept for XLSX export compatibility
        "plugin_trace": validation_trace,
    }


def rewrite_question(question):
    """Adapt division hint for PostgreSQL."""
    if question.find(" / ") != -1:
        question += ". For division operations, use CAST(xxx AS NUMERIC) or xxx::numeric to ensure precise decimal results"
    return question
