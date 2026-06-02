#!/usr/bin/env python3
"""Run the OpenSearch-SQL evaluation pipeline for a single configuration.

Dataset resolution (--dataset):
  "landcover10"           →  src/database_process/landcover10.json
  "data/my.json"          →  relative to src/OpenSearch-SQL/
  "/abs/path/my.json"     →  absolute path

LLM config (--llm-config):
  Defaults to config/models.yaml (project root).
  Use --provider / --model for a quick override without creating a YAML file.

Examples:
  uv run run/run_eval.py
  uv run run/run_eval.py --dataset landcover10 --ablation a3 --end -1
  uv run run/run_eval.py --provider swiss_ai --model Qwen/Qwen3.5-27B --end -1
  uv run run/run_eval.py --ablation full --fewshot --export-xlsx
  uv run run/run_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT         = Path(__file__).resolve().parents[1]   # src/OpenSearch-SQL/
_PROJECT_ROOT = _ROOT.parents[1]                      # ontology_retriever/
_DB_PROCESS   = _ROOT / "src" / "database_process"
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "config" / "models.yaml")

ABLATION_MODES = ("baseline", "geo", "ogf", "hints", "full", "a1", "a2", "a3", "a4", "a5")
DEFAULT_ABLATION_MODES = ("baseline", "geo", "ogf", "hints", "full")

_FLAGS = {
    # ── Canonical 5-level ablation ────────────────────────────────────────────
    # Each step adds one complete contribution; OGF and validator are always
    # paired because the validator enforces OGF's unit constraints in SQL.
    "baseline": dict(geo="False", ogf="False", entity="False", semantic="False", validator="False", optimizer="False"),
    "geo":      dict(geo="True",  ogf="False", entity="False", semantic="False", validator="False", optimizer="False"),
    "ogf":      dict(geo="True",  ogf="True",  entity="False", semantic="False", validator="True",  optimizer="False"),
    "hints":    dict(geo="True",  ogf="True",  entity="True",  semantic="True",  validator="True",  optimizer="False"),
    "full":     dict(geo="True",  ogf="True",  entity="True",  semantic="True",  validator="True",  optimizer="True"),
    # ── Legacy fine-grained modes (kept for reference) ───────────────────────
    "a1":       dict(geo="True",  ogf="False", entity="False", semantic="False", validator="False", optimizer="False"),
    "a2":       dict(geo="True",  ogf="True",  entity="False", semantic="False", validator="False", optimizer="False"),
    "a3":       dict(geo="True",  ogf="True",  entity="True",  semantic="False", validator="False", optimizer="False"),
    "a4":       dict(geo="True",  ogf="True",  entity="True",  semantic="True",  validator="False", optimizer="False"),
    "a5":       dict(geo="True",  ogf="True",  entity="True",  semantic="True",  validator="True",  optimizer="False"),
}


def resolve_dataset(name: str) -> tuple[str, str]:
    """Return (data_mode, db_root_path) for src/main.py.

    data_mode    = stem of the JSON file (e.g. "landcover10")
    db_root_path = directory containing data_preprocess/ (e.g. "data")
    """
    p = Path(name)

    # Absolute path
    if p.is_absolute():
        return p.stem, str(p.parent.parent)

    # Name only (no directory separator, no .json) — resolve from project root.
    # Use an absolute path so the subprocess (cwd=src/OpenSearch-SQL/) doesn't
    # accidentally pick up a stale copy inside src/OpenSearch-SQL/data/.
    if "/" not in name and "\\" not in name and not name.endswith(".json"):
        return name, str(_PROJECT_ROOT / "data")

    # Relative or relative-with-extension
    full = (_ROOT / p) if not p.is_absolute() else p
    stem = full.stem
    # db_root_path is parent of data_preprocess/
    db_root = str(full.parent.parent)
    return stem, db_root


def build_pipeline_setup(
    flags: dict,
    bert_model: str,
    fewshot_enabled: str,
    geo_anchor: str,
    n_candidates: int,
    temperature: float,
) -> str:
    setup = {
        "generate_db_schema": {
            "engine": "llm_factory",
            "bert_model": bert_model,
            "device": "cpu",
        },
        "extract_col_value": {
            "engine": "llm_factory",
            "temperature": 0.0,
            "fewshot_enabled": fewshot_enabled,
        },
        "extract_query_noun": {
            "engine": "llm_factory",
            "temperature": 0.0,
        },
        "column_retrieve_and_other_info": {
            "engine": "llm_factory",
            "bert_model": bert_model,
            "device": "cpu",
            "temperature": 0.3,
            "top_k": 10,
        },
        "implicit_context_enhance": {
            "enable_geo_context":      flags["geo"],
            "enable_ontology_grounding": flags["ogf"],
            "geo_anchor_mode":         geo_anchor,
        },
        "candidate_generate": {
            "engine": "llm_factory",
            "temperature": temperature,
            "n": n_candidates,
            "return_question": "True",
            "single": "False",
            "fewshot_enabled": fewshot_enabled,
            "enable_entity_hint":   flags["entity"],
            "enable_semantic_hint": flags["semantic"],
            "enable_validator":     flags["validator"],
        },
        "align_correct": {
            "engine": "llm_factory",
            "n": n_candidates,
            "bert_model": bert_model,
            "device": "cpu",
            "fewshot_enabled": fewshot_enabled,
            "align_methods": "style_align+function_align",
        },
        "query_optimizer": {
            "engine": "llm_factory",
            "temperature": 0.0,
        },
    }
    return json.dumps(setup)


def make_llm_config(provider: str, model: str | None) -> str:
    """Write a temporary YAML config and return its path."""
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml is required for --provider shorthand. Run: uv add pyyaml")

    cfg: dict = {"provider": provider, "temperature": 0.0}
    if model:
        cfg["model"] = model

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"llm_{provider}_",
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def preflight_sentence_transformer(model_name: str):
    # Pre-load the sentence-transformer model to ensure it's cached and available before running the main pipeline.
    # Please use `uv run hf auth login` to authenticate if you haven't already, as the model may need to be downloaded on the first run.
    result = subprocess.run(
        ["uv", "run", "python", "-c",
         f"from sentence_transformers import SentenceTransformer; "
         f"SentenceTransformer('{model_name}', device='cpu'); "
         f"print('[preflight] sentence-transformer ready:', '{model_name}')"],
        cwd=_ROOT,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run OpenSearch-SQL evaluation for a single configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    p.add_argument("--dataset", default="landcover10",
                   help="Dataset name or path (default: landcover10)")
    p.add_argument("--start", type=int, default=0,
                   help="First question index (default: 0)")
    p.add_argument("--end", type=int, default=1,
                   help="Last question index exclusive; -1 = all (default: 1)")

    # Ablation
    p.add_argument("--ablation", default="full", choices=ABLATION_MODES,
                   help="Ablation configuration (default: full)")

    # LLM
    p.add_argument("--llm-config", default=None,
                   help=f"Path to models YAML (default: config/models.yaml)")
    p.add_argument("--provider", default=None,
                   choices=["openai", "bedrock", "swiss_ai", "ollama", "anthropic"],
                   help="LLM provider shorthand (creates a temp config; overrides --llm-config)")
    p.add_argument("--model", default=None,
                   help="Model name, used together with --provider")

    # Generation
    p.add_argument("--n-candidates", type=int, default=5,
                   help="Number of SQL candidates to generate (default: 5)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for candidate_generate (default: 0.7)")

    # Features
    p.add_argument("--fewshot", action="store_true", default=False,
                   help="Enable few-shot examples (default: off)")
    p.add_argument("--geo-anchor", default="points", choices=["points", "bbox"],
                   help="Geo anchor mode (default: points)")

    # Output
    p.add_argument("--export-xlsx", action="store_true",
                   help="Auto-export XLSX after run")

    # Execution control
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip run if result directory already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without running")
    p.add_argument("--bert-model", default="all-mpnet-base-v2",
                   help="Sentence-transformer model name (default: all-mpnet-base-v2)")

    return p.parse_args(argv)


def run(args, extra_env: dict | None = None):
    """Execute the pipeline. Called by run_ablation.py too."""
    flags = _FLAGS[args.ablation]
    fewshot_enabled = "True" if args.fewshot else "False"
    data_mode, db_root_path = resolve_dataset(args.dataset)

    # LLM config path
    tmp_config = None
    if args.provider:
        tmp_config = make_llm_config(args.provider, args.model)
        llm_config = tmp_config
    else:
        llm_config = args.llm_config or _DEFAULT_CONFIG
    # Resolve to absolute path — subprocess runs from _ROOT (src/OpenSearch-SQL/),
    # so relative paths would resolve incorrectly inside the child process.
    llm_config = str(Path(llm_config).resolve())

    _opt_node = "+query_optimizer" if flags.get("optimizer") == "True" else ""
    pipeline_nodes = (
        "generate_db_schema+extract_col_value+extract_query_noun"
        "+column_retrieve_and_other_info+implicit_context_enhance"
        f"+candidate_generate{_opt_node}+align_correct+vote+evaluation"
    )
    pipeline_setup = build_pipeline_setup(
        flags, args.bert_model, fewshot_enabled,
        args.geo_anchor, args.n_candidates, args.temperature,
    )

    end = str(args.end) if args.end != -1 else str(10_000)

    print(f"Ablation  : {args.ablation}")
    print(f"  geo={flags['geo']}, ogf={flags['ogf']}, entity={flags['entity']}, "
          f"semantic={flags['semantic']}, validator={flags['validator']}, "
          f"optimizer={flags['optimizer']}")
    print(f"Dataset   : {data_mode}  (start={args.start}, end={args.end})")
    print(f"LLM config: {llm_config}")
    print(f"Fewshot   : {fewshot_enabled}  |  geo_anchor={args.geo_anchor}")
    print(f"Candidates: n={args.n_candidates}, temp={args.temperature}")

    if args.dry_run:
        print("\n[dry-run] pipeline_setup:")
        import json as _json
        print(_json.dumps(_json.loads(pipeline_setup), indent=2))
        return 0

    # Env
    env = {**os.environ, "LLM_CONFIG_PATH": llm_config}
    if extra_env:
        env.update(extra_env)

    preflight_sentence_transformer(args.bert_model)

    cmd = [
        "uv", "run", "python", "-u", "src/main.py",
        "--data_mode",       data_mode,
        "--db_root_path",    db_root_path,
        "--pipeline_nodes",  pipeline_nodes,
        "--pipeline_setup",  pipeline_setup,
        "--ablation_mode",   args.ablation,
        "--fewshot_mode",    "with_few_shot" if args.fewshot else "no_few_shot",
        "--start",           str(args.start),
        "--end",             end,
    ]
    rc = subprocess.run(cmd, cwd=_ROOT, env=env).returncode

    if tmp_config:
        Path(tmp_config).unlink(missing_ok=True)

    if rc != 0:
        return rc

    if args.export_xlsx:
        fewshot_label = "with_few_shot" if args.fewshot else "no_few_shot"
        results_root = _ROOT / "results" / data_mode / fewshot_label / args.ablation
        print("Exporting XLSX...")
        subprocess.run(
            ["uv", "run", "python", "eval/export_results_xlsx.py",
             "--results_root", str(results_root)],
            cwd=_ROOT, env=env,
        )

    return 0


def main():
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
