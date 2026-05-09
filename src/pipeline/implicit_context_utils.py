"""Shared helpers for implicit context payload handling."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pipeline.utils import get_last_node_result


def get_implicit_context_payload(execution_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return normalized implicit payload parts from `implicit_context_enhance`."""
    implicit_node = get_last_node_result(execution_history, "implicit_context_enhance") or {}
    return {
        "geo_context": implicit_node.get("geo_context", {}) or {},
        "ontology_grounded_function": implicit_node.get("ontology_grounded_function", {}) or {},
    }


def build_implicit_context_block(execution_history: List[Dict[str, Any]]) -> str:
    """Build prompt block containing GEO and ontology grounded function payloads."""
    payload = get_implicit_context_payload(execution_history)
    geo_context = payload["geo_context"]
    ontology_grounded_function = payload["ontology_grounded_function"]
    if not geo_context and not ontology_grounded_function:
        return ""
    return (
        "\n#GEO_CONTEXT:\n"
        f"{json.dumps(geo_context, ensure_ascii=False, indent=2)}\n"
        "#ONTOLOGY_GROUNDED_FUNCTION:\n"
        f"{json.dumps(ontology_grounded_function, ensure_ascii=False, indent=2)}\n"
    )

