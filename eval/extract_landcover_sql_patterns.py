#!/usr/bin/env python3
"""Extract canonical landcover_upscaled SQL patterns from golden datasets."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


PATTERNS = {
    "cross_join_lateral": re.compile(r"\bCROSS\s+JOIN\s+LATERAL\b", re.IGNORECASE),
    "unnest": re.compile(r"\bUNNEST\s*\(", re.IGNORECASE),
    "with_ordinality": re.compile(r"\bWITH\s+ORDINALITY\b", re.IGNORECASE),
    "join_landcover_type": re.compile(r"\bJOIN\s+landcover_type\b", re.IGNORECASE),
    "join_landcover_upscaled": re.compile(r"\b(?:FROM|JOIN)\s+landcover_upscaled\b", re.IGNORECASE),
    "round_function": re.compile(r"\bROUND\s*\(", re.IGNORECASE),
    "cast_function": re.compile(r"\bCAST\s*\(", re.IGNORECASE),
    "extract_function": re.compile(r"\bEXTRACT\s*\(", re.IGNORECASE),
    "tuple_in_filter": re.compile(r"\(\s*ROUND\s*\([^\)]*latitude[^\)]*\)\s*,\s*ROUND\s*\([^\)]*longitude[^\)]*\)\s*\)\s+IN", re.IGNORECASE),
    "array_subscript": re.compile(r"\[[0-9]+\]"),
}


@dataclass
class SqlRow:
    source_file: str
    row_index: int
    question: str
    sql_column: str
    sql: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract function usage from landcover_upscaled golden SQL rows.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/samples.xlsx",
            "data/v3_full.xlsx",
            "data/Thessaly_NOA.xlsx",
            "data/canton_of_zurich.xlsx",
        ],
        help="Input xlsx files to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/OpenSearch-SQL/eval",
        help="Directory for CSV/XLSX/summary outputs.",
    )
    parser.add_argument(
        "--top-samples-per-pattern",
        type=int,
        default=3,
        help="Number of representative SQL samples kept for each pattern.",
    )
    return parser.parse_args()


def _resolve_question_column(df: pd.DataFrame) -> str:
    candidates = [
        "Natural language question",
        "question_generated",
        "question_raw",
        "question",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return ""


def _iter_sql_columns(df: pd.DataFrame) -> Iterable[str]:
    for col in df.columns:
        lowered = col.lower()
        if "sql" in lowered or lowered in {"query", "sql query"}:
            yield col


def collect_landcover_rows(path: Path) -> List[SqlRow]:
    if not path.exists():
        return []
    df = pd.read_excel(path)
    question_col = _resolve_question_column(df)
    rows: List[SqlRow] = []
    for sql_col in _iter_sql_columns(df):
        sql_series = df[sql_col].fillna("").astype(str)
        mask = sql_series.str.contains("landcover_upscaled", case=False, regex=False)
        for idx in sql_series[mask].index.tolist():
            sql = sql_series.iloc[idx].strip()
            question = ""
            if question_col:
                question = str(df.at[idx, question_col] if idx in df.index else "").strip()
            rows.append(
                SqlRow(
                    source_file=path.name,
                    row_index=int(idx),
                    question=question,
                    sql_column=sql_col,
                    sql=sql,
                )
            )
    return rows


def build_pattern_columns(sql: str) -> dict:
    return {name: bool(pattern.search(sql)) for name, pattern in PATTERNS.items()}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[SqlRow] = []
    for raw_input in args.inputs:
        all_rows.extend(collect_landcover_rows(repo_root / raw_input))

    if not all_rows:
        raise SystemExit("No landcover_upscaled rows found in provided inputs.")

    records = []
    for row in all_rows:
        flags = build_pattern_columns(row.sql)
        records.append(
            {
                "source_file": row.source_file,
                "row_index": row.row_index,
                "sql_column": row.sql_column,
                "question": row.question,
                "sql": row.sql,
                **flags,
            }
        )
    df_rows = pd.DataFrame(records)

    pattern_counts = Counter()
    for name in PATTERNS:
        pattern_counts[name] = int(df_rows[name].sum())
    df_counts = pd.DataFrame(
        [{"pattern": k, "matched_rows": v, "ratio": v / len(df_rows)} for k, v in pattern_counts.items()]
    ).sort_values(["matched_rows", "pattern"], ascending=[False, True])

    sample_records = []
    for name in PATTERNS:
        subset = df_rows[df_rows[name]].head(args.top_samples_per_pattern)
        for _, row in subset.iterrows():
            sample_records.append(
                {
                    "pattern": name,
                    "source_file": row["source_file"],
                    "row_index": row["row_index"],
                    "question": row["question"],
                    "sql": row["sql"],
                }
            )
    df_samples = pd.DataFrame(sample_records)

    csv_rows = output_dir / "landcover_golden_sql_rows.csv"
    csv_counts = output_dir / "landcover_golden_sql_pattern_counts.csv"
    xlsx_path = output_dir / "landcover_golden_sql_patterns.xlsx"
    md_summary = output_dir / "landcover_golden_sql_patterns.md"

    df_rows.to_csv(csv_rows, index=False)
    df_counts.to_csv(csv_counts, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        df_rows.to_excel(writer, sheet_name="rows", index=False)
        df_counts.to_excel(writer, sheet_name="pattern_counts", index=False)
        df_samples.to_excel(writer, sheet_name="pattern_samples", index=False)

    core_skeleton = (
        "SELECT ... FROM landcover_upscaled lu\n"
        "CROSS JOIN LATERAL UNNEST(lu.ranks) WITH ORDINALITY AS rank_item(rank_pair, rank_idx)\n"
        "JOIN landcover_type lt ON lt.code = rank_item.rank_pair[1]\n"
        "WHERE ... -- optional temporal and geo filters"
    )
    top_lines = ["| pattern | matched_rows | ratio |", "|---|---:|---:|"]
    for _, row in df_counts.head(10).iterrows():
        top_lines.append(f"| {row['pattern']} | {int(row['matched_rows'])} | {float(row['ratio']):.3f} |")

    md_summary.write_text(
        "\n".join(
            [
                "# Landcover Golden SQL Pattern Summary",
                "",
                f"- total_rows: {len(df_rows)}",
                "- scanned_files: " + ", ".join(sorted(set(df_rows["source_file"].tolist()))),
                "",
                "## Top Pattern Counts",
                "",
                *top_lines,
                "",
                "## Canonical Join Skeleton",
                "",
                "```sql",
                core_skeleton,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved rows CSV: {csv_rows}")
    print(f"Saved pattern counts CSV: {csv_counts}")
    print(f"Saved workbook: {xlsx_path}")
    print(f"Saved markdown summary: {md_summary}")


if __name__ == "__main__":
    main()
