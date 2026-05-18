"""Semantic prompt hints for landcover array-join patterns.

Classification is LLM-based (structured output) to avoid brittle keyword matching.
Hints describe generic access paradigms, not specific SQL templates.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

LandcoverSemanticMode = Literal[
    "membership_filter",
    "aggregate_distribution",
    "ranking_by_metric",
    "group_comparison",
    "list_inventory",
    "dominant_lookup",
    "generic_landcover",
]

_CLASSIFY_PROMPT = """\
You are classifying a geospatial SQL question about landcover data stored as 2-D integer arrays.

Available access patterns:
- membership_filter: find grid locations containing a specific landcover type, then aggregate a metric per location
- aggregate_distribution: compute the share/percentage/count of each landcover type across a region
- ranking_by_metric: find the landcover type at the grid with the highest or lowest value of some external metric
- group_comparison: compare an external metric between two or more landcover categories (e.g. forest vs urban)
- list_inventory: enumerate all distinct landcover types in a region with their total pixel counts
- dominant_lookup: find or check the single dominant (most-prevalent) landcover type at a location

Question: {question}

Return ONLY valid JSON (no prose): {{"mode": "<pattern name>"}}"""


def _classify_mode_llm(question: str, chat_model) -> LandcoverSemanticMode:
    prompt = _CLASSIFY_PROMPT.format(question=question)
    try:
        raw = chat_model.get_ans(prompt, temperature=0.0)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        mode = data.get("mode", "")
        valid = LandcoverSemanticMode.__args__  # type: ignore[attr-defined]
        if mode in valid:
            return mode  # type: ignore[return-value]
    except Exception as exc:
        logging.debug("landcover mode classification failed (%s); using generic", exc)
    return "generic_landcover"


def _is_landcover_question(question: str) -> bool:
    landcover_tables = ("landcover_upscaled", "landcover_type")
    q = question.lower()
    return any(t in q for t in landcover_tables) or any(
        k in q for k in ("landcover", "land cover", "land-cover")
    )


def build_landcover_semantic_hint(question: str, chat_model=None) -> str:
    if not _is_landcover_question(question):
        return ""
    if chat_model is not None:
        mode = _classify_mode_llm(question, chat_model)
    else:
        mode = "generic_landcover"
    return _render_hint(mode)


# ── Array access fundamentals injected into every hint ────────────────────────

_ARRAY_RULES = (
    "landcover_upscaled.ranks is an int4[][] column: ranks[i][1]=level3_code (integer), ranks[i][2]=pixel_count (integer).\n"
    "To iterate entries: FROM landcover_upscaled AS u, GENERATE_SUBSCRIPTS(u.ranks, 1) AS i\n"
    "  then access u.ranks[i][1] (code) and u.ranks[i][2] (count).\n"
    "Direct subscript ranks[1][1] is valid ONLY for the single dominant entry.\n"
    "NEVER use plain UNNEST(ranks) without WITH ORDINALITY — it flattens to individual scalars.\n"
    "JOIN landcover_type ONLY on level3_code: landcover_type.level3_code = u.ranks[i][1]\n"
    "  NEVER join on level3_label, level2_label, level1_label, or any other label column.\n"
    "  level3_code is an integer; ranks[i][1] is an integer — no cast needed.\n"
    "Label hierarchy: level1 (coarsest) → level2 → level3 (finest).\n"
    "  Type-existence predicates (lakes, forests, etc.) should match level3_label — the finest level.\n"
    "  level2_label is too coarse and may include or exclude unintended categories.\n"
)


def _render_hint(mode: LandcoverSemanticMode) -> str:
    header = f"\n#LANDCOVER_SEMANTIC_HINT (mode={mode}):\n{_ARRAY_RULES}"

    if mode == "membership_filter":
        return header + (
            "Paradigm: find grid coordinates WHERE a landcover type satisfies a predicate;\n"
            "  then join those coordinates with a metric table and aggregate per grid.\n"
            "Step 1: CTE (type_grids) — SELECT DISTINCT latitude, longitude FROM landcover_upscaled\n"
            "  where the type predicate holds: expand ALL ranks entries (not just [1][1]) with\n"
            "  GENERATE_SUBSCRIPTS, JOIN landcover_type ON level3_code = ranks[i][1],\n"
            "  filter with WHERE lt.level3_label ILIKE '%<type>%'.\n"
            "  Use level3_label (finest granularity) — not level2_label — for specific type lookups\n"
            "  (e.g. lakes, forests, urban fabric are level3 concepts).\n"
            "  A grid may contain the target type as a non-dominant entry; scanning all ranks\n"
            "  is required to avoid missing such grids.\n"
            "Step 2: CTE (grid_stats) — JOIN type_grids with the metric table ON exact lat/lon equality\n"
            "  (no ROUND in the JOIN); apply time filter; aggregate metric per grid;\n"
            "  GROUP BY latitude, longitude.\n"
            "Output: SELECT latitude, longitude, aggregate_value FROM grid_stats ORDER BY latitude, longitude.\n"
            "  Return one row per qualifying grid — never collapse to a single aggregate.\n"
        )

    if mode == "aggregate_distribution":
        return header + (
            "Paradigm: compute pixel-count share or percentage for each landcover type in the region.\n"
            "Step 1: CTE (type_counts) — SUM pixel counts per level3_code across all grids.\n"
            "Step 2: CTE (grand_total) — SUM all type counts for the denominator.\n"
            "Step 3: join type_counts with landcover_type ON level3_code, CROSS JOIN grand_total;\n"
            "  compute percentage = ROUND(CAST(count AS DECIMAL) / total * 100, 2).\n"
            "REQUIRED output columns (all 8 — omitting any is wrong):\n"
            "  level1_code, level1_label, level2_code, level2_label, level3_code, level3_label,\n"
            "  total_count (raw pixel sum), percentage.\n"
            "ORDER BY percentage DESC.\n"
        )

    if mode == "ranking_by_metric":
        return header + (
            "Paradigm: find the landcover type at the grid with the extreme value of an external metric.\n"
            "Step 1: CTE — in the metric table, filter by region and time, aggregate metric per (lat, lon),\n"
            "  then select the single extremal grid: ORDER BY metric DESC/ASC LIMIT 1.\n"
            "Step 2: join that single grid's coordinates with landcover_upscaled on DIRECT lat/lon equality\n"
            "  (no ROUND in the JOIN — both tables store coordinates at the same precision).\n"
            "Step 3: expand ranks, join landcover_type, return the requested label.\n"
            "Never join the metric and landcover tables in a single flat query — always rank first.\n"
        )

    if mode == "group_comparison":
        return header + (
            "Paradigm: compare an external metric between two landcover categories (e.g. forest vs urban).\n"
            "Step 1: CTEs — for each category, SELECT DISTINCT level2_code WHERE level2_label ILIKE '%<name>%'.\n"
            "Step 2: CTE (dominant_grids) — for each grid, find its single dominant level2_code:\n"
            "  CROSS JOIN LATERAL (\n"
            "    SELECT lt.level2_code, SUM(u.ranks[i][2]) AS total_count\n"
            "    FROM GENERATE_SUBSCRIPTS(u.ranks, 1) AS i\n"
            "    JOIN landcover_type lt ON lt.level3_code = u.ranks[i][1]\n"
            "    GROUP BY lt.level2_code ORDER BY total_count DESC LIMIT 1\n"
            "  ) AS dom\n"
            "  Use CASE WHEN dom.level2_code IN (cat_A codes) THEN 'A' WHEN ... THEN 'B' ELSE 'Other'\n"
            "  Filter: AND dom.level2_code IN (cat_A UNION cat_B codes) — exclude 'Other' grids.\n"
            "Step 3: CTE — join dominant_grids with metric table on exact lat/lon equality;\n"
            "  GROUP BY landcover_group, time period; compute AVG metric.\n"
            "Output: pivot into one row — one column per category avg + difference column.\n"
            "  Use CASE WHEN / conditional aggregation to pivot categories into columns.\n"
        )

    if mode == "list_inventory":
        return header + (
            "Paradigm: enumerate all distinct landcover types present in a region with their aggregate pixel counts.\n"
            "Algorithm: expand ranks across all grids, SUM pixel counts per level3_code,\n"
            "  join with landcover_type to get all three hierarchy levels.\n"
            "REQUIRED output columns (all 7 — omitting any is wrong):\n"
            "  level1_code, level1_label, level2_code, level2_label, level3_code, level3_label, total_count.\n"
            "ORDER BY total_count DESC.\n"
        )

    if mode == "dominant_lookup":
        return header + (
            "Paradigm: access the single most-prevalent landcover type at each grid directly.\n"
            "ranks is sorted descending by pixel count: ranks[1][1] = dominant level3_code,\n"
            "  ranks[1][2] = dominant pixel count.\n"
            "Use direct subscript — no GENERATE_SUBSCRIPTS needed for dominant-only queries.\n"
        )

    # generic_landcover fallback
    return header + (
        "Choose the access pattern that fits the question semantics:\n"
        "A) membership / existence: expand ranks, JOIN landcover_type on level3_code, filter by label predicate\n"
        "B) aggregation / distribution: expand ranks, SUM(ranks[i][2]) per level3_code, GROUP BY code\n"
        "C) dominant entry: ranks[1][1] is the most-prevalent level3_code — direct subscript only\n"
    )
