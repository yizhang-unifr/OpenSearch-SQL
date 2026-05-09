#!/usr/bin/env python3
"""Run OpenSearch-SQL on sampled questions (default: first 10 + last 10).

This script:
1) Reads `<repo>/data/questions.txt`
2) Builds `sample20.json` under `src/OpenSearch-SQL/data/data_preprocess/`
3) Runs `run_main.sh` for selected ablation modes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    from subprocess_utils import run_command_interruptible
except ModuleNotFoundError:
    from run.subprocess_utils import run_command_interruptible


ALL_MODES = ("baseline", "geo_only", "ontology_only", "full")
QUESTION_COLUMN_CANDIDATES = (
    "question_generated",
    "Natural language question",
    "question",
)
SQL_COLUMN_CANDIDATES = (
    "point_sql_generated",
    "bbox_sql_generated",
    "SQL query",
    "sql",
    "gold_sql",
    "query",
)
CATEGORY_COLUMN_CANDIDATES = ("category", "Category")
PLACE_COLUMN_CANDIDATES = ("places", "place")


def _resolve_column(
    columns: List[str], requested: str, candidates: Iterable[str], label: str
) -> str:
    lowered = {column.lower(): column for column in columns}
    if requested:
        for column in columns:
            if column == requested or column.lower() == requested.lower():
                return column
        raise ValueError(
            f"Cannot find requested {label} column '{requested}'. Available columns: {columns}"
        )

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    raise ValueError(
        f"Cannot find a {label} column in workbook. Available columns: {columns}"
    )


def _resolve_optional_column(
    columns: List[str], requested: str, candidates: Iterable[str]
) -> str:
    if requested:
        return _resolve_column(columns, requested, candidates, requested)

    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return ""


def read_source_rows(
    source_xlsx_path: Path,
    question_column: str,
    sql_column: str,
    category_column: str,
    place_column: str,
) -> List[dict]:
    df = pd.read_excel(source_xlsx_path)
    resolved_question_column = _resolve_column(
        df.columns.tolist(), question_column, QUESTION_COLUMN_CANDIDATES, "question"
    )
    resolved_sql_column = _resolve_column(
        df.columns.tolist(), sql_column, SQL_COLUMN_CANDIDATES, "SQL"
    )
    resolved_category_column = _resolve_optional_column(
        df.columns.tolist(), category_column, CATEGORY_COLUMN_CANDIDATES
    )
    resolved_place_column = _resolve_optional_column(
        df.columns.tolist(), place_column, PLACE_COLUMN_CANDIDATES
    )

    rows = []
    for source_row_no, (_, row) in enumerate(df.iterrows(), start=1):
        question = str(row[resolved_question_column]).strip()
        sql = str(row[resolved_sql_column]).strip()
        category = (
            str(row[resolved_category_column]).strip()
            if resolved_category_column
            else "sample20"
        )
        place = str(row[resolved_place_column]).strip() if resolved_place_column else ""
        if not question:
            raise ValueError(f"Empty question at workbook row {source_row_no}")
        if not sql:
            raise ValueError(f"Empty SQL at workbook row {source_row_no}")
        rows.append(
            {
                "source_row_no": source_row_no,
                "question": question,
                "sql": sql,
                "category": category,
                "place": place,
            }
        )
    return rows


def pick_first_last(rows: List[dict], head_rows: int, tail_rows: int) -> List[dict]:
    total_rows = head_rows + tail_rows
    if len(rows) < total_rows:
        raise ValueError(f"Need at least {total_rows} rows, got {len(rows)}")
    return rows[:head_rows] + rows[-tail_rows:]


def build_sample_dataset(selected: Iterable[dict]) -> list[dict]:
    data = []
    for new_id, row in enumerate(selected):
        data.append(
            {
                "question_id": new_id,
                "question": row["question"],
                "db_id": "meteo",
                "evidence": "None",
                "SQL": row["sql"],
                "category": row["category"] or "sample20",
                "source_line_no": row["source_row_no"],
            }
        )
    return data


def run_mode(open_search_root: Path, mode: str, plugin_mode: str) -> None:
    env = os.environ.copy()
    env["ABLATION_MODE"] = mode
    env["PLUGIN_MODE"] = plugin_mode
    env["DATA_MODE"] = "sample20"
    env["DB_ROOT_PATH"] = "data"
    env["START"] = "0"
    env["END"] = "-1"
    run_command_interruptible(
        ["bash", "run/run_main.sh"], cwd=open_search_root, env=env
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenSearch-SQL on sampled questions"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,geo_only,ontology_only,full",
        help="Comma-separated ablation modes",
    )
    parser.add_argument(
        "--plugin-mode",
        type=str,
        default="native",
        help="Plugin mode for run_main.sh: native|plugin_on",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate sample20 dataset JSON, do not run pipeline",
    )
    parser.add_argument(
        "--source-xlsx",
        type=str,
        default="",
        help="Path to the source workbook containing aligned question and SQL columns",
    )
    parser.add_argument(
        "--samples-xlsx",
        type=str,
        default="",
        help="Backward-compatible alias for --source-xlsx",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="",
        help="Override the question column name if auto-detection is ambiguous",
    )
    parser.add_argument(
        "--sql-column",
        type=str,
        default="",
        help="Override the SQL column name if auto-detection is ambiguous",
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="",
        help="Override the category column name if auto-detection is ambiguous",
    )
    parser.add_argument(
        "--place-column",
        type=str,
        default="",
        help="Override the place column name if auto-detection is ambiguous",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=10,
        help="Number of rows to take from the start of the workbook",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=10,
        help="Number of rows to take from the end of the workbook",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    open_search_root = script_path.parents[1]
    project_root = script_path.parents[3]
    source_xlsx_path = (
        Path(args.source_xlsx or args.samples_xlsx)
        if (args.source_xlsx or args.samples_xlsx)
        else (project_root / "data" / "samples.xlsx")
    )

    rows = read_source_rows(
        source_xlsx_path,
        question_column=args.question_column,
        sql_column=args.sql_column,
        category_column=args.category_column,
        place_column=args.place_column,
    )
    selected = pick_first_last(rows, head_rows=args.head_rows, tail_rows=args.tail_rows)
    sample_data = build_sample_dataset(selected)

    out_path = open_search_root / "data" / "data_preprocess" / "sample20.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(sample_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[sample20] wrote dataset: {out_path} (size={len(sample_data)}, source={source_xlsx_path})"
    )

    if args.prepare_only:
        return

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_MODES]
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}. Allowed: {ALL_MODES}")
    if args.plugin_mode not in {"native", "plugin_on"}:
        raise ValueError("Invalid --plugin-mode. Allowed: native|plugin_on")

    for mode in modes:
        print(f"[sample20] running mode={mode}, plugin_mode={args.plugin_mode}")
        run_mode(open_search_root, mode, args.plugin_mode)

    print("[sample20] done")


if __name__ == "__main__":
    main()
