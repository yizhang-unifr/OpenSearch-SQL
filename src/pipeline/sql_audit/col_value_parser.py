"""Normalize extract_col_value LLM output to the #columns:/#values: text format.

Handles three output shapes in priority order:
1. JSON structured output  {"reason": "...", "columns": [...], "values": [...]}
2. Verbose text with #columns:/#values: markers embedded anywhere
3. Standard format already correct — pass through unchanged

Call normalize_col_value_output() on any raw LLM response before parse_des().
Call structured_output_suffix() to get the prompt suffix that requests JSON.
"""

from __future__ import annotations

import json
import logging
import re

_JSON_SUFFIX = (
    "\n\nRespond ONLY with a JSON object — no prose, no markdown fences:\n"
    '{"reason": "<brief reason>", "columns": ["table.col1", "table.col2"], "values": ["val1", "val2"]}'
)


def structured_output_suffix() -> str:
    """Prompt suffix that instructs the model to return structured JSON."""
    return _JSON_SUFFIX


def normalize_col_value_output(text: str) -> str:
    """Convert any model output to the standard #columns:/#values: text format.

    Tries in order:
    1. JSON parse — handles structured output responses
    2. Pass-through — if both markers already present
    3. Regex extraction — markers embedded inside verbose reasoning text
    4. Empty markers — last resort; downstream parse_des will return empty col/values
       rather than crashing with an unpack error

    Returns a string guaranteed to contain #columns: and #values: lines.
    """
    text = re.sub(r"^```\w*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    # 1. JSON structured output
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            cols = ", ".join(str(c) for c in data.get("columns", []))
            vals = ", ".join(f'"{v}"' for v in data.get("values", []))
            reason = data.get("reason", "")
            return f"#reason: {reason}\n#columns: {cols}\n#values: {vals}"
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # 2. Already well-formed
    if "#columns:" in text and "#values:" in text:
        return text

    # 3. Markers present but may be buried in reasoning
    col_match = re.search(r"#columns:\s*([^\n]+)", text)
    val_match = re.search(r"#values:\s*([^\n]*)", text)
    if col_match:
        cols = col_match.group(1).strip()
        vals = val_match.group(1).strip() if val_match else ""
        return f"#columns: {cols}\n#values: {vals}"

    # 4. Nothing usable found
    logging.warning(
        "col_value_parser: no structured markers found in LLM output; "
        "returning empty col/values. raw=%r",
        text[:300],
    )
    return "#columns: \n#values: "
