"""Column retrieval and info node – PostgreSQL version.

Uses find_foreign_keys_pg() instead of the BIRD tables.json-based
find_foreign_keys_MYSQL_like(). Column retrieval uses SentenceTransformer
embeddings over PostgreSQL column metadata.
"""

import json
import logging
import re
from typing import Any, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_pg
from llm.prompts import db_check_prompts
from runner.extract import DES_new
from database_process.make_emb import load_emb
from runner.column_update import ColumnUpdater


@node_decorator(check_schema_status=False)
def column_retrieve_and_other_info(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    emb_dir = paths.emb_dir
    chat_model = model_chose(node_name, config["engine"])
    bert_model = SentenceTransformer(config["bert_model"], device=config["device"])

    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"]
    origin_col = get_last_node_result(execution_history, "extract_query_noun")["col"]
    values = get_last_node_result(execution_history, "extract_query_noun")["values"]

    db = task.db_id

    # Load pre-computed embeddings
    emb_values_dic = {}
    if emb_values_dic.get(db):
        DB_emb, col_values = emb_values_dic[db]
    else:
        DB_emb, col_values = load_emb(db, emb_dir)
        emb_values_dic[db] = [DB_emb, col_values]

    db_col = {x: all_db_col[x][0] for x in all_db_col}
    db_keys_col = all_db_col.keys()

    # Simplified column retrieval (no tables.json needed for PostgreSQL)
    col_retrieve = _simple_col_retrieve(bert_model, task.question, db_keys_col)

    # Foreign keys from PostgreSQL
    foreign_keys, foreign_set = find_foreign_keys_pg()

    cols = ColumnUpdater(db_col).col_pre_update(origin_col, col_retrieve, foreign_set)

    des = DES_new(bert_model, DB_emb, col_values)
    cols_select, L_values = des.get_key_col_des(
        cols,
        values,
        debug=False,
        topk=config.get("top_k", 10),
        shold=0.65,
    )

    column = ColumnUpdater(db_col).col_suffix(cols_select)

    count = 0
    q_order = ""
    while count < 3:
        try:
            q_order = query_order(
                task.raw_question,
                chat_model,
                db_check_prompts().select_prompt,
                temperature=config.get("temperature", 0.3),
            )
            break
        except Exception:
            count += 1

    response = {
        "L_values": L_values,
        "column": column,
        "foreign_keys": foreign_keys,
        "foreign_set": list(foreign_set),
        "q_order": q_order,
    }
    return response


def _simple_col_retrieve(bert_model, question: str, db_keys_col) -> set:
    """Simplified column retrieval using embedding similarity.

    Replaces the ColumnRetriever that depended on BIRD tables.json.
    """
    import torch

    # Extract n-grams from question
    q_clean = re.sub(r"[^\s\w]", " ", question)
    q_clean = re.sub(r"\s+", " ", q_clean)
    words = q_clean.split(" ")
    ngrams = set()
    for k in range(1, min(5, len(words) + 1)):
        for j in range(len(words) - k + 1):
            ngrams.add(" ".join(words[j : j + k]))

    # Build column name list (strip table prefix)
    col_map = {}
    for full_col in db_keys_col:
        parts = full_col.split(".")
        if len(parts) == 2:
            col_name = parts[1].strip("`\"")
            col_map.setdefault(col_name, set())
            col_map[col_name].add(full_col)

    col_names = list(col_map.keys())
    if not col_names or not ngrams:
        return set()

    # Encode and find top matches
    col_embs = bert_model.encode(col_names, convert_to_tensor=True, show_progress_bar=False)
    ngram_list = list(ngrams)
    q_embs = bert_model.encode(ngram_list, convert_to_tensor=True, show_progress_bar=False)

    sims = q_embs @ col_embs.T
    num_pick = min(4, len(col_names))
    top_indices = set(torch.topk(sims.flatten(), min(num_pick * len(ngram_list), sims.numel())).indices.tolist())

    result = set()
    for idx in top_indices:
        col_idx = idx % len(col_names)
        sim_val = sims.flatten()[idx].item()
        if sim_val > 0.7:
            col_name = col_names[col_idx]
            result.update(col_map[col_name])
    return result


def query_order(question, chat_model, select_prompt, temperature):
    select_prompt = select_prompt.format(question=question)
    ans = chat_model.get_ans(select_prompt, temperature=temperature)
    ans = re.sub(r"```json|```", "", ans)
    select_json = json.loads(ans)
    res, judge = json_ext(select_json)
    return res


def json_ext(jsonf):
    ans = []
    judge = False
    for x in jsonf:
        if x["Type"] == "QIC":
            Q = x["Extract"]["Q"].lower()
            if Q in ["how many", "how much", "which", "how often"]:
                for item in x["Extract"]["I"]:
                    ans.append(x["Extract"]["Q"] + " " + item)
            elif Q in ["when", "who", "where"]:
                ans.append(x["Extract"]["Q"])
            else:
                ans.extend(x["Extract"]["I"])
        elif x["Type"] == "JC":
            ans.append(x["Extract"]["J"])
            judge = True
    return ans, judge
