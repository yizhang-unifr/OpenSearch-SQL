#!/usr/bin/env bash
# Preprocess the Meteo dataset for the OpenSearch-SQL pipeline.
#
# Steps:
#   1. Generate column-value embeddings from PostgreSQL
#
# Prerequisites:
#   - .env must be configured with DB_HOST, DB_NAME, DB_USER, DB_PORT, DB_PASS
#   - The eval/OpenSearch-SQL-eval.py script creates data_preprocess/dev.json
#     and fewshot/questions.json automatically.
#
# Usage:
#   bash run/run_preprocess.sh

set -euo pipefail
cd "$(dirname "$0")/.."

bert_model=${BERT_MODEL:-"all-mpnet-base-v2"}
data_dir="data"

echo "=== Generating column-value embeddings ==="
python3 -u src/database_process/make_emb.py \
    --emb_dir "${data_dir}/emb" \
    --bert_model "${bert_model}" \
    --db_name "meteo" \
    --env_file "$(cd .. && pwd)/.env"

echo "=== Preprocessing complete ==="
echo "Embeddings saved to ${data_dir}/emb/"
