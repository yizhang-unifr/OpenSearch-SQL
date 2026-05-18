"""Alignment and correction node – PostgreSQL version.

Changed "SQLite" to "PostgreSQL" in the DB info string and removed
db_sqlite_path parameter (PostgreSQL connections are handled internally).
"""

import json
import logging
import re
from typing import Any, Dict, List
from pathlib import Path

from sentence_transformers import SentenceTransformer

_bert_model_cache: dict = {}


def _get_bert_model(model_name: str, device: str) -> SentenceTransformer:
    key = (model_name, device)
    if key not in _bert_model_cache:
        _bert_model_cache[key] = SentenceTransformer(
            model_name, device=device, local_files_only=True
        )
    return _bert_model_cache[key]

from pipeline.utils import node_decorator, get_last_node_result, make_newprompt
from pipeline.context.implicit_context_utils import build_implicit_context_block, get_implicit_context_payload, geo_is_meaningful
from pipeline.context.landcover_semantic_hints import build_landcover_semantic_hint
from pipeline.context.landcover_entity_hint import build_landcover_entity_hint
from pipeline.pipeline_manager import PipelineManager, get_bool
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_pg
from llm.prompts import db_check_prompts
from runner.check_and_correct import muti_process_sql, soft_check, sql_raw_parse


def _extract_ogf_thresholds(ogf: Dict[str, Any]) -> set:
    """Extract explicit numeric threshold values from OGF function_specs pseudo_code_templates.

    These are the domain-specific values (e.g. 35 for a heat-day threshold) that
    the ontology dictates should appear in the SQL. Using them avoids the previous
    false-pass where any comparison operator in the SQL satisfied the check.
    """
    values: set = set()
    for spec in ogf.get("function_specs", []):
        template = spec.get("pseudo_code_template", "")
        for m in re.finditer(r"(?:>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", template):
            values.add(m.group(1))
    return values


def _implicit_consistency_check(execution_history: List[Dict[str, Any]], voted_sql: str) -> Dict[str, Any]:
    """Lightweight consistency check between final SQL and implicit payloads."""
    payload = get_implicit_context_payload(execution_history)
    geo_context = payload["geo_context"]
    ontology_grounded_function = payload["ontology_grounded_function"]
    if not geo_is_meaningful(geo_context) and not ontology_grounded_function:
        return {"implicit_check_status": "no_context", "implicit_check_warnings": []}
    warnings: List[str] = []
    vote_sql = (voted_sql or "").lower()

    if geo_is_meaningful(geo_context):
        has_geo_signal = any(k in vote_sql for k in ("latitude", "longitude", "st_", "bbox", "within", "intersect"))
        if not has_geo_signal:
            warnings.append("geo_context_present_but_sql_has_no_explicit_geo_signal")

    if ontology_grounded_function:
        thresholds = _extract_ogf_thresholds(ontology_grounded_function)
        if thresholds:
            if not any(t in vote_sql for t in thresholds):
                warnings.append(
                    f"ogf_thresholds_not_found_in_sql: expected one of {sorted(thresholds)}"
                )
        elif not re.search(r"(>=|<=|>|<)\s*-?\d+", vote_sql):
            warnings.append("ontology_grounded_function_present_but_sql_has_no_comparison_operator")

    status = "pass" if not warnings else "warn"
    return {"implicit_check_status": status, "implicit_check_warnings": warnings}


@node_decorator(check_schema_status=False)
def align_correct(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    fewshot_path = paths.db_fewshot_path
    correct_fewshot_json = paths.db_fewshot2_path
    fewshot_enabled = get_bool(config, "fewshot_enabled", default=True)

    prompts_template = db_check_prompts()
    bert_model = _get_bert_model(config["bert_model"], config["device"])

    if fewshot_enabled and fewshot_path.exists():
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

    schema_node = get_last_node_result(execution_history, "generate_db_schema")
    col_node = get_last_node_result(execution_history, "column_retrieve_and_other_info")
    cand_node = get_last_node_result(execution_history, "candidate_generate")
    if not schema_node or schema_node.get("status") != "success":
        raise RuntimeError(
            "Upstream node 'generate_db_schema' failed; "
            f"cannot run align_correct. upstream_error={schema_node.get('error') if schema_node else 'missing'}"
        )
    if not col_node or col_node.get("status") != "success":
        raise RuntimeError(
            "Upstream node 'column_retrieve_and_other_info' failed; "
            f"cannot run align_correct. upstream_error={col_node.get('error') if col_node else 'missing'}"
        )
    if not cand_node or cand_node.get("status") != "success":
        raise RuntimeError(
            "Upstream node 'candidate_generate' failed; "
            f"cannot run align_correct. upstream_error={cand_node.get('error') if cand_node else 'missing'}"
        )

    all_db_col = schema_node["db_col_dic"]
    column = col_node["column"]
    foreign_keys = col_node["foreign_keys"]
    foreign_set = col_node["foreign_set"]
    L_values = col_node["L_values"]
    q_order = col_node["q_order"]
    question = cand_node["rewrite_question"]

    # Use optimized SQL when query_optimizer ran successfully; fall back to candidate_generate.
    opt_node = get_last_node_result(execution_history, "query_optimizer")
    sql_source = (
        opt_node
        if opt_node and opt_node.get("status") == "success"
        else cand_node
    )
    SQLs = sql_source["SQL"]

    db = task.db_id
    hint = task.evidence
    foreign_set = set(foreign_set)
    fewshot = ""
    q_id_str = str(task.question_id)
    if q_id_str in df_fewshot.get("questions", {}):
        fewshot = df_fewshot["questions"][q_id_str].get("prompt", "")
    if not fewshot:
        question_key = task.raw_question.strip().lower()
        fewshot = df_fewshot.get("by_question", {}).get(question_key, {}).get("prompt", "")

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
    tmp_prompt += build_implicit_context_block(execution_history)
    tmp_prompt += build_landcover_semantic_hint(question)
    tmp_prompt += build_landcover_entity_hint(question)

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
    if not vote:
        logging.error(
            "align_correct vote list is empty; question_id=%s sql_candidates=%s candidate_sqls=%s align_methods=%s",
            getattr(task, "question_id", "unknown"),
            len(SQLs_dic),
            list(SQLs_dic.keys()),
            config.get("align_methods", "style_align+function_align"),
        )
    voted_sql = vote[0].get("sql", "") if vote else ""

    response = {
        "vote": vote,
        "none_case": none_case,
    }
    response.update(_implicit_consistency_check(execution_history, voted_sql))
    return response
