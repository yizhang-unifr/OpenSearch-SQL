"""LLM-based SQL constraint validator.

Given SQL candidates, column contracts, and OGF context, asks the LLM to
identify constraint violations and rewrite any offending SQL.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pipeline.sql_audit.column_contracts import format_contracts_for_prompt

_VALIDATOR_PROMPT = """\
You are an expert PostgreSQL SQL auditor. Check each SQL query against the \
schema column contracts and unit constraints below. Fix any violations and \
return every candidate unchanged if it already complies.

## Schema Column Contracts (hard rules — MUST be followed)
{column_contracts}

## Ontology Unit Constraints
{ogf_constraints}

## Original question
{question}

## SQL Candidates
{numbered_sqls}

Instructions:
- Evaluate each candidate independently.
- If no violations: return the SQL unchanged, "changed": false, "issues": [].
- If violations found: rewrite to fix them, "changed": true, list each fix in "issues".
- Do NOT alter anything that is not a constraint violation.

Return ONLY a valid JSON array — no prose, no markdown fences:
[
  {{"index": 0, "sql": "...", "changed": false, "issues": []}},
  {{"index": 1, "sql": "...", "changed": true, "issues": ["description of fix"]}}
]"""


def _ogf_unit_constraints(ogf: dict[str, Any]) -> str:
    lines: list[str] = []
    for spec in ogf.get("function_specs", []):
        if "unit conversion" in spec.get("semantics", "").lower():
            template = spec.get("pseudo_code_template", "")
            if "  # " in template:
                expr = template.split("  # ", 1)[1].strip()
                target = ogf.get("target_variable", "the target column")
                lines.append(
                    f"Column `{target}` stores values in raw units. "
                    f"Apply conversion `{expr}` before any numeric comparison or aggregation."
                )
    return "\n".join(lines) if lines else "(none)"


def validate_sql_candidates(
    chat_model,
    sql_candidates: list[str],
    column_contracts: dict[str, list[str]],
    implicit_payload: dict[str, Any],
    question: str,
) -> tuple[list[str], list[dict]]:
    """Validate SQL candidates for constraint violations and rewrite if needed.

    Returns:
        (corrected_sql_list, trace_list)
        Falls back to original SQL on any LLM or parse error.
    """
    if not sql_candidates:
        return sql_candidates, []

    prompt = _VALIDATOR_PROMPT.format(
        column_contracts=format_contracts_for_prompt(column_contracts),
        ogf_constraints=_ogf_unit_constraints(
            implicit_payload.get("ontology_grounded_function", {})
        ),
        question=question,
        numbered_sqls="\n".join(f"[{i}] {sql}" for i, sql in enumerate(sql_candidates)),
    )

    raw = chat_model.get_ans(prompt, temperature=0.0)

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        results: list[dict] = json.loads(cleaned)
        results_sorted = sorted(results, key=lambda x: x["index"])

        if len(results_sorted) != len(sql_candidates):
            raise ValueError(
                f"LLM returned {len(results_sorted)} results for {len(sql_candidates)} candidates"
            )

        corrected = [entry["sql"] for entry in results_sorted]
        trace = [
            {
                "validator": "constraint_validator",
                "candidate_index": entry["index"],
                "changed": entry.get("changed", False),
                "issues": entry.get("issues", []),
            }
            for entry in results_sorted
        ]
        return corrected, trace

    except Exception as exc:
        logging.warning(
            "constraint_validator: failed (%s); returning original SQL. raw=%r",
            exc,
            raw[:200],
        )
        return sql_candidates, [
            {"validator": "constraint_validator", "error": str(exc), "raw": raw[:500]}
        ]
