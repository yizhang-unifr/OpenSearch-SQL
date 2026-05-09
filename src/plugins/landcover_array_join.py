from __future__ import annotations

import re

from plugins.base import PluginContext, PluginResult


_LANDCOVER_KEYWORD_RE = re.compile(r"\blandcover_upscaled\b|\blandcover_type\b|\branks\b", re.IGNORECASE)
_QUESTION_KEYWORD_RE = re.compile(r"\blandcover|forest|water|lake|urban|cropland|dominant\b", re.IGNORECASE)
_TABLE_ALIAS_RE = re.compile(
    r"(?:FROM|JOIN)\s+(?:[\w]+\.)?landcover_upscaled(?:\s+AS)?\s+([a-zA-Z_][\w]*)",
    re.IGNORECASE,
)
_JOIN_LT_RE = re.compile(
    r"JOIN\s+(?:[\w]+\.)?landcover_type(?:\s+AS)?\s+([a-zA-Z_][\w]*)\s+ON\s+",
    re.IGNORECASE,
)


class LandcoverArrayJoinPlugin:
    """Force canonical array-expansion join for landcover_upscaled.ranks."""

    def name(self) -> str:
        return "landcover_array_join"

    def applies(self, context: PluginContext) -> bool:
        return bool(_QUESTION_KEYWORD_RE.search((context.question or "").lower()))

    def transform(self, sql: str, context: PluginContext) -> PluginResult:
        sql_lower = sql.lower()
        if "landcover_upscaled" not in sql_lower:
            return PluginResult(sql=sql, changed=False)
        if "landcover_type" not in sql_lower:
            return PluginResult(
                sql=sql,
                changed=False,
                warnings=["landcover_array_join_missing_landcover_type_join"],
                metadata=self._constraints(),
            )
        if "unnest(" in sql_lower or "generate_subscripts(" in sql_lower or "ranks[" in sql_lower:
            return PluginResult(sql=sql, changed=False, metadata=self._constraints())
        if not _LANDCOVER_KEYWORD_RE.search(sql):
            return PluginResult(sql=sql, changed=False)

        u_alias = self._find_alias(sql, default="u")
        lt_alias = self._find_landcover_type_alias(sql, default="lt")
        rewrite_payload = self._rewrite_join(sql, u_alias=u_alias, lt_alias=lt_alias)
        if not rewrite_payload:
            return PluginResult(
                sql=sql,
                changed=False,
                warnings=["landcover_array_join_rewrite_pattern_not_found"],
                metadata=self._constraints(),
            )

        rewritten_sql = rewrite_payload
        return PluginResult(
            sql=rewritten_sql,
            changed=(rewritten_sql != sql),
            warnings=["landcover_array_join_applied_unnest_skeleton"],
            metadata={
                **self._constraints(),
                "skeleton": self._skeleton(u_alias=u_alias, lt_alias=lt_alias),
            },
        )

    def _rewrite_join(self, sql: str, u_alias: str, lt_alias: str) -> str | None:
        join_match = _JOIN_LT_RE.search(sql)
        if not join_match:
            return None
        original_join_start = join_match.start()
        original_join_text = self._consume_join_clause(sql, original_join_start)
        if not original_join_text:
            return None
        replacement = (
            f"CROSS JOIN LATERAL UNNEST({u_alias}.ranks) WITH ORDINALITY AS rank_item(level3_code, count, rank_idx) "
            f"JOIN landcover_type {lt_alias} ON {lt_alias}.level3_code = rank_item.level3_code "
        )
        return sql.replace(original_join_text, replacement, 1)

    def _consume_join_clause(self, sql: str, start: int) -> str:
        stop_tokens = (" JOIN ", " WHERE ", " GROUP BY ", " ORDER BY ", " HAVING ", " LIMIT ", ";")
        end = len(sql)
        tail = sql[start:]
        for token in stop_tokens:
            pos = tail.upper().find(token.strip().upper())
            if pos > 0:
                end = min(end, start + pos)
        return sql[start:end]

    def _find_alias(self, sql: str, default: str) -> str:
        match = _TABLE_ALIAS_RE.search(sql)
        return match.group(1) if match else default

    def _find_landcover_type_alias(self, sql: str, default: str) -> str:
        match = _JOIN_LT_RE.search(sql)
        return match.group(1) if match else default

    def _constraints(self) -> dict:
        return {
            "hard_constraints": [
                "Do not treat landcover_upscaled.ranks as scalar columns.",
                "When joining landcover_upscaled to landcover_type, expand ranks first.",
                "Use rank_item.level3_code as landcover level3_code join key.",
            ],
            "templates": {
                "expansion_template": "CROSS JOIN LATERAL UNNEST(<u_alias>.ranks) WITH ORDINALITY AS rank_item(level3_code, count, rank_idx)",
                "join_template": "JOIN landcover_type <lt_alias> ON <lt_alias>.level3_code = rank_item.level3_code",
            },
        }

    def _skeleton(self, u_alias: str, lt_alias: str) -> str:
        return (
            "SELECT ...\n"
            f"FROM landcover_upscaled {u_alias}\n"
            f"CROSS JOIN LATERAL UNNEST({u_alias}.ranks) WITH ORDINALITY AS rank_item(level3_code, count, rank_idx)\n"
            f"JOIN landcover_type {lt_alias} ON {lt_alias}.level3_code = rank_item.level3_code\n"
            "WHERE ...;"
        )
