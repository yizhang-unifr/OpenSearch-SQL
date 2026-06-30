"""Implicit context enhancement node for OpenSearch-SQL.

This node adds two optional context payloads:
- GEO_CONTEXT (from geo_resolver)
- ONTOLOGY_GROUNDED_FUNCTION (from ontology retriever + parser)

The node is fail-soft by design and returns empty payloads on errors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager, get_bool
from pipeline.context.implicit_context_utils import geo_is_meaningful


def _inject_sql_filter(geo_context: dict, anchor_mode: str) -> None:
    """Add a pre-formatted sql_filter hint so the LLM knows the exact WHERE clause to use."""
    if anchor_mode == "points":
        points = geo_context.get("points") or []
        if not points:
            return
        pairs = ", ".join(
            f"({p['lat']}, {p['lon']})" for p in points
        )
        geo_context["sql_filter"] = (
            "(ROUND(CAST(<alias>.latitude AS DECIMAL), 1), "
            f"ROUND(CAST(<alias>.longitude AS DECIMAL), 1)) IN ({pairs})"
        )
    elif anchor_mode == "bbox":
        bbox = geo_context.get("bbox") or {}
        min_lat = bbox.get("min_lat") or bbox.get("lat_min") or bbox.get("minlat")
        max_lat = bbox.get("max_lat") or bbox.get("lat_max") or bbox.get("maxlat")
        min_lon = bbox.get("min_lon") or bbox.get("lon_min") or bbox.get("minlon")
        max_lon = bbox.get("max_lon") or bbox.get("lon_max") or bbox.get("maxlon")
        if None in (min_lat, max_lat, min_lon, max_lon):
            return
        # Source bbox corners are raw (unrounded) geographic extents; the gold SQL
        # cache rounds each corner to 1 decimal place (matching the DB's storage
        # grid), so the filter must do the same to have any chance of matching.
        min_lat, max_lat, min_lon, max_lon = (round(float(v), 1) for v in (min_lat, max_lat, min_lon, max_lon))
        geo_context["sql_filter"] = (
            f"<alias>.latitude BETWEEN {min_lat} AND {max_lat} "
            f"AND <alias>.longitude BETWEEN {min_lon} AND {max_lon}"
        )


def _ensure_project_src_in_path() -> None:
    """Ensure `<repo>/src` is importable for meteo/text2sql helpers."""
    here = Path(__file__).resolve()
    # .../src/OpenSearch-SQL/src/pipeline/implicit_context_enhance.py
    # repo src root is .../src
    repo_src = here.parents[3]
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))


@node_decorator(check_schema_status=False)
def implicit_context_enhance(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    # Pass the node name explicitly so get_model_para does not rely on stack inspection.
    config, _node_name = PipelineManager().get_model_para("implicit_context_enhance")
    enable_geo = get_bool(config, "enable_geo_context", default=True)
    enable_ontology = get_bool(config, "enable_ontology_grounding", default=True)
    # geo_anchor_mode is a run-level config (set via --geo-anchor); the dataset's
    # per-question geo_filter_mode field is a static "points" placeholder for every
    # row (not a meaningful per-question override), so it must not take priority.
    anchor_mode = str(config.get("geo_anchor_mode", "points"))

    geo_context: Dict[str, Any] = {}
    ontology_grounded_function: Dict[str, Any] = {}
    enhancement_status = "empty"
    errors: list[str] = []

    _ensure_project_src_in_path()

    if enable_geo:
        try:
            from text2sql.tools.places_geo import resolve_city_geo_for_question

            geo_raw = resolve_city_geo_for_question(task.question, anchor_mode=anchor_mode)
            geo_context = json.loads(geo_raw) if isinstance(geo_raw, str) else (geo_raw or {})
            _inject_sql_filter(geo_context, anchor_mode)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"geo_context_error: {exc}")
            geo_context = {}

    if enable_ontology:
        try:
            from text2sql.tools.kg_context_tool import build_kg_context

            kg_context = build_kg_context(task.question)
            ogf_raw = getattr(kg_context, "ontology_grounded_function_json", "{}")
            ontology_grounded_function = json.loads(ogf_raw) if isinstance(ogf_raw, str) else (ogf_raw or {})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ontology_grounded_function_error: {exc}")
            ontology_grounded_function = {}

    # A geo_context with place=None or no coordinates is not meaningful grounding.
    if geo_is_meaningful(geo_context) or ontology_grounded_function:
        enhancement_status = "success"
    elif errors:
        enhancement_status = "fallback"

    return {
        "geo_context": geo_context,
        "ontology_grounded_function": ontology_grounded_function,
        "enhancement_status": enhancement_status,
        "enhancement_errors": errors,
        "enhancement_config": {
            "enable_geo_context": enable_geo,
            "enable_ontology_grounding": enable_ontology,
            "geo_anchor_mode": anchor_mode,
        },
    }

