"""Semantic prompt hints for landcover array-join patterns."""

from __future__ import annotations

from typing import Literal


LandcoverSemanticMode = Literal["simple_filter", "dominant_direct", "dominance_aggregation", "ranking_by_metric", "comparison", "generic_landcover"]


def build_landcover_semantic_hint(question: str) -> str:
    q = (question or "").lower()
    if not _is_landcover_question(q):
        return ""
    mode = _classify_mode(q)
    return _render_hint(mode)


def _is_landcover_question(q: str) -> bool:
    keys = ("landcover", "forest", "water", "lake", "urban", "cropland", "dominant")
    return any(k in q for k in keys)


def _classify_mode(q: str) -> LandcoverSemanticMode:
    # ranks is sorted by count descending: ranks[1][1]=dominant code, ranks[1][2]=dominant count.
    #
    # Six cases based on question semantics:
    #   dominant_direct       — "dominant landcover is X / is urban" → ranks[1][1] directly
    #   dominance_aggregation — "percentage/share/distribution" → GENERATE_SUBSCRIPTS + SUM
    #   ranking_by_metric     — "landcover type at the highest/lowest [temperature]" → CTE pattern
    #   comparison            — "compare ... between forest and urban" → CTE per group pattern
    #   simple_filter         — "areas with forest", "includes lake" → GENERATE_SUBSCRIPTS + label
    #   generic_landcover     — fallback
    distribution_keys = ("percentage", "share", "proportion", "distribution", "each type", "all types")
    dominant_keys = ("dominant",)
    membership_keys = ("includes", "include", "with ", "contains", "having", "areas that")
    comparison_keys = ("compare", "comparison", "difference between", "vs ")
    forest_urban_keys = ("forest", "urban")
    ranking_keys = ("highest", "lowest", "maximum", "minimum")
    temperature_keys = ("temperature", "temp", "tmean", "tmax", "tmin")

    has_dominant = any(k in q for k in dominant_keys)
    has_distribution = any(k in q for k in distribution_keys)
    has_comparison = any(k in q for k in comparison_keys)
    has_forest_urban = any(k in q for k in forest_urban_keys)
    has_ranking = any(k in q for k in ranking_keys)
    has_temperature = any(k in q for k in temperature_keys)

    # "compare ... between forest and urban" must be checked before dominant_direct
    # because Q3/Q4 contain "dominant" but are comparison queries, not single-type lookups.
    if (has_comparison or "between" in q) and has_forest_urban:
        return "comparison"
    if has_dominant and not has_distribution:
        return "dominant_direct"
    if has_distribution or (has_dominant and has_distribution):
        return "dominance_aggregation"
    # "first level landcover type of the highest [metric] location" → CTE ranking pattern
    if has_ranking and has_temperature and ("landcover" in q or "level" in q or "type" in q):
        return "ranking_by_metric"
    if any(k in q for k in ("top", "most")):
        return "dominance_aggregation"
    if any(k in q for k in membership_keys):
        return "simple_filter"
    if has_ranking or "first" in q:
        return "simple_filter"
    return "generic_landcover"


# Injected into modes that need full array iteration (simple_filter, dominance_aggregation, generic).
# NOT injected into dominant_direct mode where ranks[1][1] is intentionally correct.
_NEVER_SUBSCRIPT = (
    "NEVER use constant array subscript (ranks[1][1], ranks[1]) to iterate all entries.\n"
    "Exception: ranks[1][1]/ranks[1][2] are correct ONLY when the question asks for the single dominant entry.\n"
    "NEVER join landcover_type on level1_label or level2_label — always join on level3_code.\n"
    "ALWAYS expand the ranks array with GENERATE_SUBSCRIPTS before joining when iterating all entries.\n"
    "NEVER use UNNEST(ranks) — ranks is int4[][], UNNEST flattens all values to individual scalars, not (code, count) pairs.\n"
)

# For dominant_direct mode: only the non-contradictory rules.
_DOMINANT_NEVER = (
    "NEVER join landcover_type on level1_label or level2_label — always join on level3_code.\n"
    "NEVER use UNNEST(ranks) — ranks is int4[][], UNNEST flattens all values to individual scalars.\n"
)


def _render_hint(mode: LandcoverSemanticMode) -> str:
    common = (
        "Do not join landcover_upscaled directly to landcover_type by latitude/longitude.\n"
        "Treat ranks as array-encoded landcover entries and expand before joining landcover_type.\n"
        f"{_NEVER_SUBSCRIPT}"
    )
    dominant_common = (
        "Do not join landcover_upscaled directly to landcover_type by latitude/longitude.\n"
        f"{_DOMINANT_NEVER}"
    )
    if mode == "dominant_direct":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=dominant_direct\n"
            f"{dominant_common}"
            "ranks is sorted by count descending: ranks[1] = dominant (most prevalent) entry.\n"
            "ranks[1][1] = dominant level3_code, ranks[1][2] = dominant pixel count.\n"
            "For 'dominant landcover is X' queries, access ranks[1][1] directly — no array expansion needed.\n"
            "Recommended pattern:\n"
            "SELECT u.ranks[1][1] AS level3_code, u.ranks[1][2] AS total_count\n"
            "FROM landcover_upscaled AS u\n"
            "JOIN landcover_type lt ON lt.level3_code = u.ranks[1][1]\n"
            "WHERE lt.level3_label ILIKE '%<type>%' AND ...\n"
        )
    if mode == "simple_filter":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=simple_filter\n"
            f"{common}"
            "Use membership semantics: existence of a landcover label/code in a grid.\n"
            "Recommended pattern:\n"
            "CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(u.ranks, 1) AS i\n"
            "JOIN landcover_type lt ON lt.level3_code = u.ranks[i][1]\n"
            "u.ranks[i][1] = level3_code, u.ranks[i][2] = pixel count.\n"
            "Apply label/code predicates on lt.* (for example ILIKE '%forest%' or ILIKE '%lake%').\n"
        )
    if mode == "dominance_aggregation":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=dominance_aggregation\n"
            f"{common}"
            "Use dominance/share semantics: expand ranks, aggregate counts, then rank.\n"
            "Recommended pattern:\n"
            "CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(u.ranks, 1) AS i\n"
            "JOIN landcover_type lt ON lt.level3_code = u.ranks[i][1]\n"
            "u.ranks[i][1] = level3_code, u.ranks[i][2] = pixel count.\n"
            "Aggregate with SUM(u.ranks[i][2]) GROUP BY lt.level3_code and ORDER BY count DESC\n"
            "to identify the dominant landcover type.\n"
        )
    if mode == "ranking_by_metric":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=ranking_by_metric\n"
            f"{common}"
            "Query pattern: find the landcover type at the grid with the highest/lowest metric.\n"
            "Use a CTE to decompose the problem into steps:\n"
            "  Step 1 — CTE: filter the meteo table by time period AND geo filter, compute the metric per (lat, lon)\n"
            "  Step 2 — find the extremal (lat, lon) using ORDER BY metric DESC/ASC LIMIT 1\n"
            "  Step 3 — join that location with landcover_upscaled on exact lat/lon equality,\n"
            "            expand ranks with GENERATE_SUBSCRIPTS, join landcover_type on level3_code\n"
            "  Step 4 — return the landcover label (level1_label or as required)\n"
            "Do NOT join meteo and landcover tables directly using ROUND() in the JOIN condition — this is slow.\n"
            "Apply the geo filter (IN list) inside the meteo CTE, and join on exact latitude/longitude equality.\n"
        )
    if mode == "comparison":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=comparison\n"
            f"{common}"
            "Query pattern: compare a metric (e.g. temperature) between grid points classified as 'Forest' vs 'Urban'.\n"
            "Use CTEs to decompose:\n"
            "  Step 1 — CTE: get level2_codes for forest (level2_label ILIKE '%forest%')\n"
            "  Step 2 — CTE: get level2_codes for urban  (level2_label ILIKE '%urban%')\n"
            "  Step 3 — CTE: for each grid in geo_filter, determine the dominant level2 category\n"
            "            by expanding ranks with GENERATE_SUBSCRIPTS, grouping by level2_code,\n"
            "            ordering by SUM(ranks[i][2]) DESC LIMIT 1, then CASE WHEN the result\n"
            "            matches forest/urban codes → label 'Forest'/'Urban'\n"
            "  Step 4 — join labeled grids with the meteo table on exact lat/lon equality,\n"
            "            aggregate the metric per group, return the comparison\n"
            "Join meteo_tmean to dominant_grids on exact lat=lat AND lon=lon (no ROUND in JOIN).\n"
        )
    return (
        "\n#LANDCOVER_SEMANTIC_HINT:\n"
        "mode=generic_landcover\n"
        f"{common}"
        "Pick one of two GENERATE_SUBSCRIPTS-based templates:\n"
        "  ranks is int4[][]: ranks[i][1]=level3_code, ranks[i][2]=count.\n"
        "A) membership filter -> GENERATE_SUBSCRIPTS(u.ranks, 1) AS i + JOIN ON u.ranks[i][1] + label predicate on lt.*\n"
        "B) dominance/share  -> GENERATE_SUBSCRIPTS(u.ranks, 1) AS i + SUM(u.ranks[i][2]) aggregation, ORDER BY count DESC\n"
    )

