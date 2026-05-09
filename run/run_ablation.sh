#!/usr/bin/env bash
# Batch runner for implicit-knowledge ablation experiments.
#
# Runs four configurations sequentially:
# - baseline
# - geo_only
# - ontology_only
# - full
#
# Usage:
#   bash run/run_ablation.sh
#   START=0 END=50 bash run/run_ablation.sh
#   GEO_ANCHOR_MODE=bbox bash run/run_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

modes=(baseline geo_only ontology_only full)

for mode in "${modes[@]}"; do
  echo "============================================================"
  echo "Ablation run start: ${mode}"
  echo "============================================================"
  ABLATION_MODE="${mode}" \
  START="${START:-0}" \
  END="${END:-1}" \
  GEO_ANCHOR_MODE="${GEO_ANCHOR_MODE:-points}" \
  bash run/run_main.sh
done

echo "All ablation runs completed."
