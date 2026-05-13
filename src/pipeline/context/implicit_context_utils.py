"""Shared helpers for implicit context payload handling."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pipeline.utils import get_last_node_result


def geo_is_meaningful(geo_context: Dict[str, Any]) -> bool:
    """Return True only when geo_context carries actual spatial data.

    A payload with ``place: None`` (unresolved lookup) or with neither
    ``points`` nor ``bbox`` does not provide usable grounding for SQL and is
    treated as empty, even though the dict itself is non-empty.
    """
    return (
        bool(geo_context)
        and geo_context.get("place") is not None
        and (bool(geo_context.get("points")) or bool(geo_context.get("bbox")))
    )


def get_implicit_context_payload(execution_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return normalized implicit payload parts from `implicit_context_enhance`."""
    implicit_node = get_last_node_result(execution_history, "implicit_context_enhance") or {}
    return {
        "geo_context": implicit_node.get("geo_context", {}) or {},
        "ontology_grounded_function": implicit_node.get("ontology_grounded_function", {}) or {},
    }


def _condensed_geo_context(geo_context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact version of geo_context for prompt injection.

    When sql_filter is present it already encodes the full coordinate list,
    so the raw points/bbox arrays are redundant and bloat the prompt significantly.
    Keep only place + sql_filter in that case.
    """
    if geo_context.get("sql_filter"):
        return {k: v for k, v in geo_context.items() if k not in ("points", "bbox")}
    return geo_context


def build_implicit_context_block(execution_history: List[Dict[str, Any]]) -> str:
    """Build prompt block containing GEO and ontology grounded function payloads."""
    payload = get_implicit_context_payload(execution_history)
    geo_context = payload["geo_context"]
    ontology_grounded_function = payload["ontology_grounded_function"]
    if not geo_is_meaningful(geo_context) and not ontology_grounded_function:
        return ""
    prompt_geo = _condensed_geo_context(geo_context)
    return (
        "\n#GEO_CONTEXT:\n"
        f"{json.dumps(prompt_geo, ensure_ascii=False, indent=2)}\n"
        "#ONTOLOGY_GROUNDED_FUNCTION:\n"
        f"{json.dumps(ontology_grounded_function, ensure_ascii=False, indent=2)}\n"
    )

