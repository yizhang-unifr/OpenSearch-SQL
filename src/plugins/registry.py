from __future__ import annotations

from typing import Dict, List, Tuple

from plugins.base import PluginContext, SQLPlugin
from plugins.geo_pair_constraint import GeoPairConstraintPlugin
from plugins.landcover_array_join import LandcoverArrayJoinPlugin
from plugins.ontology_unit_constraint import OntologyUnitConstraintPlugin


class PluginRegistry:
    def __init__(self, config: Dict | None = None):
        self.config = config or {}
        self.plugins: List[SQLPlugin] = [
            LandcoverArrayJoinPlugin(),
            GeoPairConstraintPlugin(),
            OntologyUnitConstraintPlugin(),
        ]

    def apply(self, sql_candidates: List[str], context: PluginContext) -> Tuple[List[str], List[Dict]]:
        enhanced: List[str] = []
        traces: List[Dict] = []
        enabled = str(self.config.get("enabled", "True")).lower() == "true"
        if not enabled:
            return sql_candidates, [{"plugin": "registry", "enabled": False}]

        for idx, sql in enumerate(sql_candidates):
            current = sql
            per_candidate_trace: List[Dict] = []
            for plugin in self.plugins:
                if not plugin.applies(context):
                    per_candidate_trace.append({"plugin": plugin.name(), "applied": False, "reason": "not_applicable"})
                    continue
                result = plugin.transform(current, context)
                current = result.sql
                per_candidate_trace.append(
                    {
                        "plugin": plugin.name(),
                        "applied": True,
                        "changed": result.changed,
                        "warnings": result.warnings,
                        "metadata": result.metadata,
                    }
                )
            enhanced.append(current)
            traces.append({"candidate_index": idx, "trace": per_candidate_trace})
        return enhanced, traces
