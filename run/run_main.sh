#!/usr/bin/env bash
# Run the OpenSearch-SQL pipeline on the Meteo dataset (PostgreSQL).
#
# Prerequisites:
#   1. Set DB_HOST, DB_NAME, DB_USER, DB_PORT, DB_PASS in .env
#   2. Run preprocessing first: bash run/run_preprocess.sh
#
# Usage:
#   bash run/run_main.sh               # default (first question only)
#   START=0 END=10 bash run/run_main.sh  # first 10 questions

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Configuration ────────────────────────────────────────────────────
data_mode='dev'
db_root_path=data         # data directory (under OpenSearch-SQL/)
start=${START:-0}
end=${END:-1}

pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation'

# Use models.yaml from the parent project
bert_model=${BERT_MODEL:-"all-mpnet-base-v2"}

pipeline_setup='{
    "generate_db_schema": {
        "engine": "llm_factory",
        "bert_model": "'${bert_model}'",
        "device": "cpu"
    },
    "extract_col_value": {
        "engine": "llm_factory",
        "temperature": 0.0
    },
    "extract_query_noun": {
        "engine": "llm_factory",
        "temperature": 0.0
    },
    "column_retrieve_and_other_info": {
        "engine": "llm_factory",
        "bert_model": "'${bert_model}'",
        "device": "cpu",
        "temperature": 0.3,
        "top_k": 10
    },
    "candidate_generate": {
        "engine": "llm_factory",
        "temperature": 0.7,
        "n": 5,
        "return_question": "True",
        "single": "False"
    },
    "align_correct": {
        "engine": "llm_factory",
        "n": 5,
        "bert_model": "'${bert_model}'",
        "device": "cpu",
        "align_methods": "style_align+function_align"
    }
}'

python3 -u ./src/main.py \
    --data_mode "${data_mode}" \
    --db_root_path "${db_root_path}" \
    --pipeline_nodes "${pipeline_nodes}" \
    --pipeline_setup "${pipeline_setup}" \
    --start "${start}" \
    --end "${end}"
