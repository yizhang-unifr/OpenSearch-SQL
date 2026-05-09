from __future__ import annotations

import re
from typing import List, Tuple

from plugins.base import PluginContext, PluginResult


_LAT_LON_IN_PATTERN = re.compile(
    r"(?P<lat_col>\b[\w\.]*latitude\b)\s+IN\s*\([^\)]*\)\s+AND\s+"
    r"(?P<lon_col>\b[\w\.]*longitude\b)\s+IN\s*\([^\)]*\)",
    flags=re.IGNORECASE,
)


class GeoPairConstraintPlugin:
    def name(self) -> str:
        return "geo_pair_constraint"

    def applies(self, context: PluginContext) -> bool:
        return bool(context.geo_context.get("points"))

    def transform(self, sql: str, context: PluginContext) -> PluginResult:
        match = _LAT_LON_IN_PATTERN.search(sql)
        if not match:
            return PluginResult(
                sql=sql,
                changed=False,
                warnings=["geo_pair_constraint_not_applied_pattern_not_found"],
            )

        points = _normalize_points(context.geo_context.get("points", []))
        if not points:
            return PluginResult(sql=sql, changed=False, warnings=["geo_pair_constraint_no_valid_points"])

        lat_col = match.group("lat_col")
        lon_col = match.group("lon_col")
        pair_values = ", ".join(f"({lat}, {lon})" for lat, lon in points)
        replacement = (
            f"(ROUND(CAST({lat_col} AS DECIMAL), 1), ROUND(CAST({lon_col} AS DECIMAL), 1)) "
            f"IN ({pair_values})"
        )
        rewritten = _LAT_LON_IN_PATTERN.sub(replacement, sql, count=1)
        return PluginResult(
            sql=rewritten,
            changed=(rewritten != sql),
            metadata={"pair_count": len(points)},
        )


def _normalize_points(raw_points: List[dict]) -> List[Tuple[str, str]]:
    dedup = set()
    ordered: List[Tuple[str, str]] = []
    for item in raw_points:
        try:
            lat = f"{float(item['lat']):.1f}"
            lon = f"{float(item['lon']):.1f}"
        except Exception:
            continue
        key = (lat, lon)
        if key not in dedup:
            dedup.add(key)
            ordered.append(key)
    return ordered
