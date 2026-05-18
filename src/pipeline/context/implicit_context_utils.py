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


def _geo_filter_block(geo_context: Dict[str, Any]) -> str:
    """Render a mandatory spatial filter block for prompt injection.

    Uses the pre-built sql_filter when available.
    Falls back to bbox or point-list construction when not.
    Switches to a BETWEEN bbox expression when the point list exceeds the threshold,
    to avoid bloating the prompt with hundreds of coordinate pairs.
    """
    place = geo_context.get("place", "")
    sql_filter = geo_context.get("sql_filter", "")
    points = geo_context.get("points") or []
    bbox = geo_context.get("bbox") or {}

    if sql_filter:
        filter_expr = sql_filter
    elif bbox:
        min_lat = bbox.get("min_lat") or bbox.get("lat_min")
        max_lat = bbox.get("max_lat") or bbox.get("lat_max")
        min_lon = bbox.get("min_lon") or bbox.get("lon_min")
        max_lon = bbox.get("max_lon") or bbox.get("lon_max")
        filter_expr = (
            f"ROUND(CAST(<alias>.latitude  AS DECIMAL), 1) BETWEEN {min_lat} AND {max_lat} "
            f"AND ROUND(CAST(<alias>.longitude AS DECIMAL), 1) BETWEEN {min_lon} AND {max_lon}"
        )
    elif points:
        pairs = ", ".join(
            f"({float(p['lat'])}, {float(p['lon'])})" for p in points
        )
        filter_expr = (
            f"(ROUND(CAST(<alias>.latitude AS DECIMAL), 1), "
            f"ROUND(CAST(<alias>.longitude AS DECIMAL), 1)) IN ({pairs})"
        )
    else:
        return ""

    return (
        "\n#MANDATORY_SPATIAL_FILTER\n"
        f"# Place: {place}\n"
        "# This filter MUST appear in the WHERE clause of your SQL.\n"
        "# Replace every <alias> with the actual table alias you use for the spatial table\n"
        "# (e.g. if you alias landcover_upscaled as 'u', replace <alias> with 'u').\n"
        "# latitude and longitude are stored at 1 decimal place — always use ROUND(..., 1).\n"
        f"{filter_expr}\n"
    )


def build_implicit_context_block(execution_history: List[Dict[str, Any]]) -> str:
    """Build prompt block containing mandatory spatial filter and ontology grounded function."""
    payload = get_implicit_context_payload(execution_history)
    geo_context = payload["geo_context"]
    ontology_grounded_function = payload["ontology_grounded_function"]
    if not geo_is_meaningful(geo_context) and not ontology_grounded_function:
        return ""

    parts: list[str] = []

    if geo_is_meaningful(geo_context):
        geo_block = _geo_filter_block(geo_context)
        if geo_block:
            parts.append(geo_block)

    if ontology_grounded_function:
        parts.append(
            "\n#ONTOLOGY_GROUNDED_FUNCTION:\n"
            f"{json.dumps(ontology_grounded_function, ensure_ascii=False, indent=2)}\n"
        )

    return "".join(parts)

