"""Convert test_data.xlsx to pipeline-compatible eval JSON files.

Supports two test_data.xlsx formats:

  Long format (split_config_II_tiered):
    Each row has sql_variant ('point'|'bbox') + sql_query.
    Rows are filtered by sql_variant to produce point/bbox JSON files.

  Wide format (split_config_III_tiered):
    Each row has both point_query and bbox_query.
    Each row expands to two records (one per variant).

Outputs (in data/data_preprocess/):
  test_data_point.json       — point-query eval records
  test_data_bbox.json        — bbox-query eval records

Each JSON item:
  question_id, question, raw_question, db_id, evidence, SQL,
  category, geo_filter_mode, template_index

Usage (from src/OpenSearch-SQL/):
    uv run python src/database_process/preprocess_test_data.py [options]

Options:
    --split       Split config name (default: split_config_III_tiered)
    --test_xlsx   Explicit path to test_data.xlsx (overrides --split)
    --out_dir     Output directory (default: data/data_preprocess)
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
_DEFAULT_SPLIT = "split_config_III_tiered"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # src/OpenSearch-SQL/src/database_process/ → project root


def _find_test_xlsx(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / split / "test_data.xlsx",
        data_dir.parent / split / "test_data.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"test_data.xlsx not found for split '{split}'; tried: {[str(c) for c in candidates]}"
    )


def _is_wide_format(df: pd.DataFrame) -> bool:
    return "point_query" in df.columns and "bbox_query" in df.columns


def _expand_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Expand wide-format rows (both queries per row) into per-variant subsets."""
    subsets = {}
    for variant, col in (("point", "point_query"), ("bbox", "bbox_query")):
        sub = df.copy()
        sub["sql_query"] = sub[col]
        subsets[variant] = sub.reset_index(drop=True)
    return subsets


def preprocess(test_xlsx: Path, out_dir: Path, limit: int | None = None) -> dict[str, Path]:
    log.info("Reading %s …", test_xlsx)
    df = pd.read_excel(test_xlsx)

    df = df[df["augmentation_validation"] == True].copy()
    log.info("  after validation filter: %d rows", len(df))

    wide = _is_wide_format(df)
    log.info("  format: %s", "wide (point_query + bbox_query)" if wide else "long (sql_variant)")

    if wide:
        variant_subsets = _expand_wide(df)
        log.info("  expanded to %d point + %d bbox records",
                 len(variant_subsets["point"]), len(variant_subsets["bbox"]))
    else:
        log.info("  variants: %s", df["sql_variant"].value_counts().to_dict())
        variant_subsets = {
            v: df[df["sql_variant"] == v].reset_index(drop=True)
            for v in ("point", "bbox")
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for variant, geo_mode in GEO_MODE_MAP.items():
        subset = variant_subsets[variant]
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
    parser.add_argument("--split", default=_DEFAULT_SPLIT,
                        help=f"Split config name under data/ (default: {_DEFAULT_SPLIT})")
    parser.add_argument("--test_xlsx", default=None,
                        help="Explicit path to test_data.xlsx (overrides --split)")
    parser.add_argument("--out_dir", default="data/data_preprocess")
    parser.add_argument("--limit", type=int, default=None,
                        help="Restrict to first N rows per variant (useful for quick tests)")
    args = parser.parse_args()

    if args.test_xlsx:
        test_xlsx = Path(args.test_xlsx)
    else:
        test_xlsx = _find_test_xlsx(_PROJECT_ROOT / "data", args.split)
    out_dir = Path(args.out_dir)

    written = preprocess(test_xlsx, out_dir, args.limit)
    print("\nGenerated files:")
    for variant, p in written.items():
        with open(p) as f:
            n = len(json.load(f))
        print(f"  {p}  ({n} rows)")


if __name__ == "__main__":
    main()
