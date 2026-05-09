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
from pipeline.pipeline_manager import PipelineManager


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
    config, _node_name = PipelineManager().get_model_para()
    enable_geo = str(config.get("enable_geo_context", "True")).lower() == "true"
    enable_ontology = str(config.get("enable_ontology_grounding", "True")).lower() == "true"
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

    if geo_context or ontology_grounded_function:
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

