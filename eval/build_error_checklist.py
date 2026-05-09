#!/usr/bin/env python3
"""Build semantic error checklist from OpenSearch-SQL run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
from eval_prompts import build_json_repair_prompt, build_semantic_mismatch_prompt


@dataclass
class CaseRecord:
    run_label: str
    run_path: str
    question_id: int
    question: str
    exec_err: str
    gold_sql: str
    predicted_sql: str
    error_types: str
    notes: str
    has_landcover_gold: bool
    has_landcover_pred: bool
    has_array_ops_pred: bool
    has_tuple_pair_gold: bool
    has_tuple_pair_pred: bool
    has_split_in_pred: bool
    has_bbox_pred: bool
    llm_primary_error_type: str
    llm_secondary_error_types: str
    llm_confidence: float
    llm_reasoning: str
    llm_analysis_json: str


def _text(x: Optional[str]) -> str:
    if not x:
        return ""
    return str(x)


def _contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def classify_errors(question: str, gold_sql: str, pred_sql: str) -> Tuple[List[str], List[str]]:
    q = question.lower()
    g = gold_sql.lower()
    p = pred_sql.lower()
    tags: List[str] = []
    notes: List[str] = []

    has_tuple_gold = "(round(cast" in g and ") in (" in g
    has_tuple_pred = "(round(" in p and ") in (" in p or "(latitude, longitude) in (" in p
    has_split_pred = "latitude in (" in p and "longitude in (" in p
    has_bbox_pred = "latitude between" in p or "longitude between" in p

    has_landcover_gold = _contains_any(g, ["landcover_upscaled", "landcover_type", "ranks", "unnest("])
    has_landcover_pred = _contains_any(p, ["landcover_upscaled", "landcover_type", "ranks", "unnest("])
    has_array_ops_pred = _contains_any(p, ["unnest(", "any(", "@>", "?|", "?&", "::int[]"])

    if has_tuple_gold and has_split_pred:
        tags.append("geo_pair_to_split_in")
        notes.append("Gold uses tuple/pair geo constraint, predicted SQL uses independent lat/lon IN.")
    if has_tuple_gold and has_bbox_pred:
        tags.append("geo_pair_to_bbox")
        notes.append("Gold uses pair points but predicted SQL relaxes to bbox range.")
    if has_landcover_gold and not has_landcover_pred:
        tags.append("landcover_filter_dropped")
        notes.append("Gold has landcover constraint but predicted SQL drops landcover tables.")
    if has_landcover_gold and has_landcover_pred and not has_array_ops_pred and "ranks" in g:
        tags.append("array_semantics_missing")
        notes.append("Gold requires array expansion (UNNEST/LATERAL) but predicted SQL lacks array operators.")
    if has_landcover_pred and "level1_label" in p and "landcover_upscaled" in p and "level1_label" not in g:
        tags.append("hallucinated_landcover_join")
        notes.append("Predicted SQL uses likely non-existent join key on landcover_upscaled.")

    asks_coord = "coordinate" in q or "coordinates" in q
    gold_has_temp_output = "min_temperature_in_celsius" in g or " - 273.15 as " in g
    pred_has_temp_output = " - 273.15 as " in p or "min_temperature_in_celsius" in p
    if asks_coord and gold_has_temp_output and not pred_has_temp_output:
        tags.append("output_projection_missing")
        notes.append("Gold includes temperature output but predicted SQL returns only coordinates.")

    if not tags:
        tags.append("other_semantic_mismatch")
        notes.append("Mismatch is not covered by current rule-based tags.")
    return tags, notes


def _extract_json_block(text: str) -> Dict:
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                try:
                    return json.loads(p)
                except json.JSONDecodeError:
                    continue
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(t[l : r + 1])
    raise ValueError("No JSON object found in LLM response")


def llm_analyze_mismatch(question: str, gold_sql: str, pred_sql: str, timeout_s: int = 45) -> Dict:
    # Reuse project LLM factory (AWS/Bedrock configured via .env).
    opensearch_root = Path(__file__).resolve().parent.parent
    project_root = opensearch_root.parent.parent
    load_dotenv(project_root / ".env")
    src_root = opensearch_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        from llm.model import model_chose
    except Exception as e:
        return {
            "primary_error_type": "llm_unavailable",
            "secondary_error_types": [],
            "confidence": 0.0,
            "reasoning": f"Failed to import llm.model/model_chose: {e}",
            "evidence": [],
        }

    prompt = build_semantic_mismatch_prompt(question, gold_sql, pred_sql)

    proxy_keys = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ]
    proxy_backup = {k: os.environ.get(k) for k in proxy_keys}
    for k in proxy_keys:
        os.environ.pop(k, None)
    try:
        chat_model = model_chose("eval_mismatch", "llm_factory")
        content = chat_model.get_ans(prompt, temperature=0.0)
        try:
            parsed = _extract_json_block(content)
        except Exception:
            repair_prompt = build_json_repair_prompt(content)
            repaired = chat_model.get_ans(repair_prompt, temperature=0.0)
            parsed = _extract_json_block(repaired)
        return {
            "primary_error_type": str(parsed.get("primary_error_type", "other_semantic_mismatch")),
            "secondary_error_types": parsed.get("secondary_error_types", []),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reasoning": str(parsed.get("reasoning", "")),
            "evidence": parsed.get("evidence", []),
        }
    except (TimeoutError, KeyError, ValueError, json.JSONDecodeError, Exception) as e:
        return {
            "primary_error_type": "llm_parse_or_request_error",
            "secondary_error_types": [],
            "confidence": 0.0,
            "reasoning": f"LLM analysis failed: {e}",
            "evidence": [],
        }
    finally:
        for k, v in proxy_backup.items():
            if v is not None:
                os.environ[k] = v


def parse_run(
    run_path: Path,
    run_label: str,
    enable_llm_analysis: bool = True,
    llm_max_cases: int = 200,
) -> List[CaseRecord]:
    rows: List[CaseRecord] = []
    llm_used = 0
    for fp in sorted(run_path.glob("*_meteo.json")):
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            continue

        qid = int(fp.name.split("_")[0])
        eval_nodes = [x for x in payload if x.get("node_type") == "evaluation"]
        if not eval_nodes:
            continue
        eval_node = eval_nodes[-1]
        vote_eval = eval_node.get("vote", {})

        exec_err = _text(vote_eval.get("exec_err"))
        if exec_err == "--":
            continue

        question = _text(vote_eval.get("Question"))
        gold_sql = _text(vote_eval.get("GOLD_SQL"))
        pred_sql = _text(vote_eval.get("PREDICTED_SQL"))

        tags, notes = classify_errors(question, gold_sql, pred_sql)
        llm_result = {
            "primary_error_type": "",
            "secondary_error_types": [],
            "confidence": 0.0,
            "reasoning": "",
            "evidence": [],
        }
        if (
            enable_llm_analysis
            and tags == ["other_semantic_mismatch"]
            and llm_used < llm_max_cases
        ):
            llm_result = llm_analyze_mismatch(question, gold_sql, pred_sql)
            llm_used += 1

        g = gold_sql.lower()
        p = pred_sql.lower()
        rows.append(
            CaseRecord(
                run_label=run_label,
                run_path=str(run_path),
                question_id=qid,
                question=question,
                exec_err=exec_err,
                gold_sql=gold_sql,
                predicted_sql=pred_sql,
                error_types=";".join(tags),
                notes=" | ".join(notes),
                has_landcover_gold=_contains_any(g, ["landcover_upscaled", "landcover_type", "ranks", "unnest("]),
                has_landcover_pred=_contains_any(p, ["landcover_upscaled", "landcover_type", "ranks", "unnest("]),
                has_array_ops_pred=_contains_any(p, ["unnest(", "any(", "@>", "?|", "?&", "::int[]"]),
                has_tuple_pair_gold="(round(cast" in g and ") in (" in g,
                has_tuple_pair_pred="(round(" in p and ") in (" in p or "(latitude, longitude) in (" in p,
                has_split_in_pred="latitude in (" in p and "longitude in (" in p,
                has_bbox_pred="latitude between" in p or "longitude between" in p,
                llm_primary_error_type=llm_result["primary_error_type"],
                llm_secondary_error_types=";".join(llm_result["secondary_error_types"]),
                llm_confidence=float(llm_result["confidence"]),
                llm_reasoning=llm_result["reasoning"],
                llm_analysis_json=json.dumps(llm_result, ensure_ascii=False),
            )
        )
    return rows


def discover_default_runs(results_root: Path) -> Dict[str, Path]:
    candidates = {
        "no_few_full_latest": results_root / "no_few_shot/full/pipe9_6672b56b/2026-05-08-00-35-46",
        "with_few_full_latest": results_root / "with_few_shot/full/pipe9_6672b56b/2026-05-08-00-51-28",
    }
    return {k: v for k, v in candidates.items() if v.exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checklist CSV/XLSX from result JSON files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sample20"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval"),
        help="Output directory for CSV/XLSX artifacts.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional run spec: label=relative/path/from/results-root",
    )
    parser.add_argument(
        "--enable-llm-analysis",
        action="store_true",
        default=False,
        help="Enable LLM-based structured analysis for other_semantic_mismatch cases.",
    )
    parser.add_argument(
        "--llm-max-cases",
        type=int,
        default=200,
        help="Max number of LLM-analyzed mismatch cases.",
    )
    args = parser.parse_args()

    run_map: Dict[str, Path] = {}
    if args.run:
        for spec in args.run:
            if "=" not in spec:
                raise ValueError(f"Invalid --run spec: {spec}")
            label, rel = spec.split("=", 1)
            run_map[label] = args.results_root / rel
    else:
        run_map = discover_default_runs(args.results_root)

    rows: List[CaseRecord] = []
    for label, path in run_map.items():
        if not path.exists():
            continue
        rows.extend(
            parse_run(
                path,
                label,
                enable_llm_analysis=args.enable_llm_analysis,
                llm_max_cases=args.llm_max_cases,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checklist_csv = args.out_dir / "semantic_error_checklist.csv"
    checklist_xlsx = args.out_dir / "semantic_error_checklist.xlsx"

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "run_label",
                "run_path",
                "question_id",
                "question",
                "exec_err",
                "error_types",
                "notes",
                "gold_sql",
                "predicted_sql",
            ]
        )
        empty.to_csv(checklist_csv, index=False)
        with pd.ExcelWriter(checklist_xlsx, engine="openpyxl") as writer:
            empty.to_excel(writer, index=False, sheet_name="checklist")
        print(f"Wrote empty checklist: {checklist_csv}")
        print(f"Wrote empty workbook: {checklist_xlsx}")
        return

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["run_label", "question_id"])
    exploded = (
        df.assign(error_type=df["error_types"].str.split(";"))
        .explode("error_type")
        .groupby(["run_label", "error_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["run_label", "count"], ascending=[True, False])
    )
    run_summary = (
        df.groupby("run_label", as_index=False)
        .agg(total_errors=("question_id", "count"), landcover_related=("has_landcover_gold", "sum"))
        .sort_values("run_label")
    )
    llm_summary = (
        df[df["llm_primary_error_type"].astype(str) != ""]
        .groupby(["run_label", "llm_primary_error_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["run_label", "count"], ascending=[True, False])
    )

    df.to_csv(checklist_csv, index=False)
    with pd.ExcelWriter(checklist_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="checklist")
        exploded.to_excel(writer, index=False, sheet_name="error_type_counts")
        run_summary.to_excel(writer, index=False, sheet_name="run_summary")
        llm_summary.to_excel(writer, index=False, sheet_name="llm_error_type_counts")

    print(f"Wrote checklist CSV: {checklist_csv}")
    print(f"Wrote checklist XLSX: {checklist_xlsx}")


if __name__ == "__main__":
    main()
