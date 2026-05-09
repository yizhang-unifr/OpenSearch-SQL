from __future__ import annotations

import re

from plugins.base import PluginContext, PluginResult


_AVG_EXPR_PATTERN = re.compile(r"AVG\((?P<expr>[^\)]+)\)", flags=re.IGNORECASE)


class OntologyUnitConstraintPlugin:
    def name(self) -> str:
        return "ontology_unit_constraint"

    def applies(self, context: PluginContext) -> bool:
        question = context.question.lower()
        return "temperature" in question and bool(context.ontology_grounded_function)

    def transform(self, sql: str, context: PluginContext) -> PluginResult:
        sql_lower = sql.lower()
        if "273.15" in sql_lower:
            return PluginResult(sql=sql, changed=False)
        if "tmean" not in sql_lower:
            return PluginResult(sql=sql, changed=False, warnings=["ontology_unit_constraint_non_tmean_query"])

        match = _AVG_EXPR_PATTERN.search(sql)
        if not match:
            return PluginResult(sql=sql, changed=False, warnings=["ontology_unit_constraint_no_avg_found"])

        avg_expr = match.group(0)
        replacement = f"({avg_expr} - 273.15)"
        rewritten = _AVG_EXPR_PATTERN.sub(replacement, sql, count=1)
        return PluginResult(
            sql=rewritten,
            changed=(rewritten != sql),
            metadata={"unit_conversion": "kelvin_to_celsius"},
        )
