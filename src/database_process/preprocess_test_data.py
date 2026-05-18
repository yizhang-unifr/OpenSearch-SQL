"""Convert test_data.xlsx to pipeline-compatible eval JSON files.

test_data.xlsx schema:
  natural_language_question  — question text
  sql_query                  — gold SQL
  sql_variant                — 'point' | 'bbox'
  category                   — query category (A, B, C, …)
  template_index             — template source line

Outputs (in data/data_preprocess/):
  test_data_point.json       — rows where sql_variant == 'point'
  test_data_bbox.json        — rows where sql_variant == 'bbox'

Each JSON item matches the format expected by Task.__init__ and
build_fewshot_index.py:
  question_id, question, raw_question, db_id, evidence, SQL,
  category, geo_filter_mode

Usage (from src/OpenSearch-SQL/):
    uv run python src/database_process/preprocess_test_data.py [options]

Options:
    --test_xlsx   Path to test_data.xlsx   (default: auto-detected)
    --out_dir     Output directory          (default: data/data_preprocess)
    --limit       Only output first N rows per variant (for quick tests)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GEO_MODE_MAP = {"point": "points", "bbox": "bbox"}


def _find_test_xlsx(data_dir: Path) -> Path:
    candidates = [
        data_dir / "split_config_II_tiered" / "test_data.xlsx",
        data_dir.parent / "split_config_II_tiered" / "test_data.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"test_data.xlsx not found; tried: {[str(c) for c in candidates]}"
    )


def preprocess(test_xlsx: Path, out_dir: Path, limit: int | None = None) -> dict[str, Path]:
    log.info("Reading %s …", test_xlsx)
    df = pd.read_excel(test_xlsx)
    log.info("  total rows: %d, variants: %s", len(df), df["sql_variant"].value_counts().to_dict())

    df = df[df["augmentation_validation"] == True].copy()
    log.info("  after validation filter: %d rows", len(df))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for variant, geo_mode in GEO_MODE_MAP.items():
        subset = df[df["sql_variant"] == variant].reset_index(drop=True)
        if limit:
            subset = subset.head(limit)

        records = []
        for i, row in subset.iterrows():
            records.append({
                "question_id": int(i),
                "question": str(row["natural_language_question"]).strip(),
                "raw_question": str(row["natural_language_question"]).strip(),
                "db_id": "meteo",
                "evidence": "",
                "SQL": str(row["sql_query"]).strip(),
                "category": str(row.get("category", "")),
                "geo_filter_mode": geo_mode,
                "template_index": int(row.get("template_index", -1)),
            })

        out_path = out_dir / f"test_data_{variant}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        log.info("  wrote %d rows → %s", len(records), out_path)
        written[variant] = out_path

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess test_data.xlsx → eval JSON")
    parser.add_argument("--test_xlsx", default=None)
    parser.add_argument("--out_dir", default="data/data_preprocess")
    parser.add_argument("--limit", type=int, default=None,
                        help="Restrict to first N rows per variant (useful for quick tests)")
    args = parser.parse_args()

    data_dir = Path("data")
    test_xlsx = Path(args.test_xlsx) if args.test_xlsx else _find_test_xlsx(data_dir)
    out_dir = Path(args.out_dir)

    written = preprocess(test_xlsx, out_dir, args.limit)
    print("\nGenerated files:")
    for variant, p in written.items():
        with open(p) as f:
            n = len(json.load(f))
        print(f"  {p}  ({n} rows)")


if __name__ == "__main__":
    main()
