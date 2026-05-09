"""OpenSearch-SQL pipeline entry point – PostgreSQL / Meteo version.

Adapted from the original BIRD-benchmark main.py to work with
PostgreSQL and the Meteo dataset structure.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Paths
_SRC_DIR = Path(__file__).resolve().parent
_OPENSEARCH_ROOT = _SRC_DIR.parent
_PROJECT_ROOT = _OPENSEARCH_ROOT.parent.parent

# OpenSearch-SQL src dir needs a path entry for local runner/pipeline modules
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

load_dotenv(_PROJECT_ROOT / ".env")

from runner.run_manager import RunManager


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r") as file:
        dataset = json.load(file)
    return dataset


def main(args):
    db_json = os.path.join(args.db_root_path, "data_preprocess", f"{args.data_mode}.json")
    dataset = load_dataset(db_json)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(args.start, args.end, dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Run OpenSearch-SQL pipeline on Meteo dataset")
    args_parser.add_argument("--data_mode", type=str, required=True, help="Mode: 'dev' or 'train'")
    args_parser.add_argument("--db_root_path", type=str, required=True, help="Root path for data files")
    args_parser.add_argument("--pipeline_nodes", type=str, required=True, help="Pipeline nodes ('+' separated)")
    args_parser.add_argument("--pipeline_setup", type=str, required=True, help="Pipeline setup JSON string")
    args_parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpointing")
    args_parser.add_argument("--checkpoint_nodes", type=str, required=False, help="Checkpoint nodes")
    args_parser.add_argument("--checkpoint_dir", type=str, required=False, help="Checkpoint directory")
    args_parser.add_argument("--log_level", type=str, default="warning", help="Logging level")
    args_parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    args_parser.add_argument("--end", type=int, default=-1, help="End index (exclusive, -1=all)")
    args_parser.add_argument("--ablation_mode", type=str, default="full", help="Ablation mode label for run output grouping")
    args_parser.add_argument("--fewshot_mode", type=str, default="with_few_shot", help="Few-shot mode label for run output grouping")
    args = args_parser.parse_args()
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.use_checkpoint:
        print("Using checkpoint")
        if not args.checkpoint_nodes:
            raise ValueError("Please provide checkpoint_nodes")
        if not args.checkpoint_dir:
            raise ValueError("Please provide checkpoint_dir")

    main(args)
