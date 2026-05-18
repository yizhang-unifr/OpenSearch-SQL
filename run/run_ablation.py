#!/usr/bin/env python3
"""Run the full ablation suite (baseline → a1 → a2 → a3 → a4 → full) sequentially.

All flags from run_eval.py are accepted and forwarded to each mode run,
except --ablation (which is iterated automatically).

Examples:
  uv run run/run_ablation.py                                   # all modes, landcover10
  uv run run/run_ablation.py --dataset thesaly_sample          # different dataset
  uv run run/run_ablation.py --modes a3,a4,full                # subset of modes
  uv run run/run_ablation.py --provider swiss_ai --end -1      # Swiss AI, all questions
  uv run run/run_ablation.py --skip-existing --export-xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "run"))

from run_eval import parse_args as _eval_parse_args, run as _eval_run, ABLATION_MODES

ALL_MODES = list(ABLATION_MODES)   # baseline → a1 → a2 → a3 → a4 → a5 → full


def parse_args():
    p = argparse.ArgumentParser(
        description="Run all ablation modes sequentially via run_eval.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--modes", default=",".join(ALL_MODES),
                   help=f"Comma-separated modes to run (default: all). "
                        f"Choices: {', '.join(ALL_MODES)}")

    # Forward all run_eval flags (except --ablation which we override per mode)
    p.add_argument("--dataset",      default="landcover10")
    p.add_argument("--start",        type=int, default=0)
    p.add_argument("--end",          type=int, default=-1,
                   help="Default -1 (all questions) for ablation runs")
    p.add_argument("--llm-config",   default=None)
    p.add_argument("--provider",     default=None,
                   choices=["openai", "bedrock", "swiss_ai", "ollama", "anthropic"])
    p.add_argument("--model",        default=None)
    p.add_argument("--n-candidates", type=int, default=5)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--fewshot",      action="store_true", default=False)
    p.add_argument("--geo-anchor",   default="points", choices=["points", "bbox"])
    p.add_argument("--export-xlsx",  action="store_true")
    p.add_argument("--skip-existing",action="store_true")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--bert-model",   default="all-mpnet-base-v2")

    return p.parse_args()


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        sys.exit(f"Unknown mode(s): {invalid}. Valid: {ALL_MODES}")

    print("=" * 64)
    print(f"Ablation suite: {' → '.join(modes)}")
    print(f"Dataset: {args.dataset}  (start={args.start}, end={args.end})")
    print("=" * 64)

    for mode in modes:
        print(f"\n{'─' * 64}")
        print(f"Mode: {mode}")
        print("─" * 64)

        # Build an args namespace that run_eval.run() expects
        import argparse as _ap
        eval_args = _ap.Namespace(
            ablation=mode,
            dataset=args.dataset,
            start=args.start,
            end=args.end,
            llm_config=args.llm_config,
            provider=args.provider,
            model=args.model,
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            fewshot=args.fewshot,
            geo_anchor=args.geo_anchor,
            export_xlsx=args.export_xlsx,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            bert_model=args.bert_model,
        )
        rc = _eval_run(eval_args)
        if rc != 0:
            sys.exit(f"Mode '{mode}' failed with exit code {rc}")

    print("\n" + "=" * 64)
    print("All modes complete.")


if __name__ == "__main__":
    main()
