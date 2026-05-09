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
#   ABLATION_MODE=baseline bash run/run_main.sh
#   ABLATION_MODE=geo_only bash run/run_main.sh
#   ABLATION_MODE=ontology_only bash run/run_main.sh
#   ABLATION_MODE=full bash run/run_main.sh
#   PLUGIN_MODE=native bash run/run_main.sh
#   PLUGIN_MODE=plugin_on bash run/run_main.sh

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-../../.venv/bin/python}

# ── Configuration ────────────────────────────────────────────────────
data_mode=${DATA_MODE:-dev}
db_root_path=${DB_ROOT_PATH:-data}         # data directory (under OpenSearch-SQL/)
start=${START:-0}
end=${END:-1}
fewshot_mode=${FEWSHOT_MODE:-with_few_shot}
sentence_transformer_model=${SENTENCE_TRANSFORMER_MODEL:-all-mpnet-base-v2}
disable_proxy=${DISABLE_PROXY:-1}

if [ "${disable_proxy}" = "1" ]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY="*"
  export no_proxy="*"
fi

# Force offline loading for sentence-transformers/huggingface stack.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME}

pipeline_nodes='generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+implicit_context_enhance+candidate_generate+align_correct+vote+evaluation'

# Use models.yaml from the parent project
bert_model=${BERT_MODEL:-"all-mpnet-base-v2"}
ablation_mode=${ABLATION_MODE:-full}
geo_anchor_mode=${GEO_ANCHOR_MODE:-points}
plugin_mode=${PLUGIN_MODE:-native}

case "${ablation_mode}" in
  baseline)
    enable_geo_context="False"
    enable_ontology_grounding="False"
    ;;
  geo_only)
    enable_geo_context="True"
    enable_ontology_grounding="False"
    ;;
  ontology_only)
    enable_geo_context="False"
    enable_ontology_grounding="True"
    ;;
  full)
    enable_geo_context="True"
    enable_ontology_grounding="True"
    ;;
  *)
    echo "Unsupported ABLATION_MODE='${ablation_mode}'. Use one of: baseline|geo_only|ontology_only|full" >&2
    exit 1
    ;;
esac

case "${plugin_mode}" in
  native)
    plugins_enabled="False"
    ;;
  plugin_on)
    plugins_enabled="True"
    ;;
  *)
    echo "Unsupported PLUGIN_MODE='${plugin_mode}'. Use one of: native|plugin_on" >&2
    exit 1
    ;;
esac

case "${fewshot_mode}" in
  with_few_shot)
    fewshot_enabled="True"
    ;;
  no_few_shot)
    fewshot_enabled="False"
    ;;
  *)
    echo "Unsupported FEWSHOT_MODE='${fewshot_mode}'. Use one of: with_few_shot|no_few_shot" >&2
    exit 1
    ;;
esac

echo "Running mode: ${ablation_mode} (geo=${enable_geo_context}, ontology=${enable_ontology_grounding}, anchor_mode=${geo_anchor_mode}, plugin_mode=${plugin_mode}, fewshot_mode=${fewshot_mode}, disable_proxy=${disable_proxy})"
echo "Offline mode: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}, SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME}"

# Fail fast if local sentence-transformer cache is missing/incomplete.
"${PYTHON_BIN}" - <<PY
from sentence_transformers import SentenceTransformer
model_name = "${sentence_transformer_model}"
SentenceTransformer(model_name, device="cpu", local_files_only=True)
print(f"[preflight] local sentence-transformer ready: {model_name}")
PY

pipeline_setup='{
    "generate_db_schema": {
        "engine": "llm_factory",
        "bert_model": "'${bert_model}'",
        "device": "cpu"
    },
    "extract_col_value": {
        "engine": "llm_factory",
        "temperature": 0.0,
        "fewshot_enabled": "'${fewshot_enabled}'"
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
    "implicit_context_enhance": {
        "enable_geo_context": "'${enable_geo_context}'",
        "enable_ontology_grounding": "'${enable_ontology_grounding}'",
        "geo_anchor_mode": "'${geo_anchor_mode}'"
    },
    "candidate_generate": {
        "engine": "llm_factory",
        "temperature": 0.7,
        "n": 5,
        "return_question": "True",
        "single": "False",
        "fewshot_enabled": "'${fewshot_enabled}'",
        "plugins": {
            "enabled": "'${plugins_enabled}'"
        }
    },
    "align_correct": {
        "engine": "llm_factory",
        "n": 5,
        "bert_model": "'${bert_model}'",
        "device": "cpu",
        "fewshot_enabled": "'${fewshot_enabled}'",
        "align_methods": "style_align+function_align"
    }
}'

"${PYTHON_BIN}" -u ./src/main.py \
    --data_mode "${data_mode}" \
    --db_root_path "${db_root_path}" \
    --pipeline_nodes "${pipeline_nodes}" \
    --pipeline_setup "${pipeline_setup}" \
    --ablation_mode "${ablation_mode}" \
    --fewshot_mode "${fewshot_mode}" \
    --start "${start}" \
    --end "${end}"
