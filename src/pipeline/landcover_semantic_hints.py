"""Semantic prompt hints for landcover array-join patterns."""

from __future__ import annotations

from typing import Literal


LandcoverSemanticMode = Literal["simple_filter", "dominance_aggregation", "generic_landcover"]


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
    membership_keys = ("includes", "include", "with ", "contains", "having", "areas that")
    dominance_keys = (
        "dominant",
        "top",
        "percentage",
        "share",
        "proportion",
        "distribution",
        "highest",
        "lowest",
        "most",
    )
    if any(k in q for k in dominance_keys):
        return "dominance_aggregation"
    if any(k in q for k in membership_keys):
        return "simple_filter"
    return "generic_landcover"


def _render_hint(mode: LandcoverSemanticMode) -> str:
    common = (
        "Do not join landcover_upscaled directly to landcover_type by latitude/longitude.\n"
        "Treat ranks as array-encoded landcover entries and expand before joining landcover_type."
    )
    if mode == "simple_filter":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=simple_filter\n"
            f"{common}\n"
            "Use membership semantics: existence of a landcover label/code in a grid.\n"
            "Recommended pattern:\n"
            "CROSS JOIN LATERAL UNNEST(u.ranks) WITH ORDINALITY AS r(val, idx)\n"
            "JOIN landcover_type lt ON lt.level3_code = val\n"
            "Apply label/code predicates on lt.* (for example ILIKE '%forest%').\n"
        )
    if mode == "dominance_aggregation":
        return (
            "\n#LANDCOVER_SEMANTIC_HINT:\n"
            "mode=dominance_aggregation\n"
            f"{common}\n"
            "Use dominance/share semantics: aggregate counts from ranks entries.\n"
            "Recommended pattern:\n"
            "CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(u.ranks, 1) AS i\n"
            "JOIN landcover_type lt ON lt.level3_code = u.ranks[i][1]\n"
            "Use u.ranks[i][2] as count/weight and aggregate before ranking.\n"
        )
    return (
        "\n#LANDCOVER_SEMANTIC_HINT:\n"
        "mode=generic_landcover\n"
        f"{common}\n"
        "Pick one of two templates based on intent:\n"
        "A) simple filter -> UNNEST + label/code filter\n"
        "B) dominance/share -> GENERATE_SUBSCRIPTS + ranks[i][2] aggregation\n"
    )

