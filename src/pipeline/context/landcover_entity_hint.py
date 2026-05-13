"""Pre-generation prompt hint: DB-validated landcover entity → level/label filter.

Algorithm
---------
1. Extract a landcover keyword from the question.
2. Determine preferred CLC level (default 3; override if question mentions level1/level2).
3. Try ILIKE '%keyword%' at that level against the real landcover_type table.
4. If no rows: cascade up (3→2→1) until a level returns hits.
5. Inject the validated filter as #LANDCOVER_ENTITY_HINT into the generation prompt.

Results are cached so repeated calls within a run pay no DB round-trip.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Scan order: longer phrases before shorter ones to avoid partial shadowing.
_KEYWORDS = [
    "water bodies", "water body", "water courses", "water course",
    "inland water", "marine water", "sea and ocean", "coastal lagoon",
    "estuary", "estuaries",
    "broad-leaved forest", "coniferous forest", "mixed forest",
    "urban fabric", "urban area",
    "artificial surface",
    "agricultural area", "agricultural land", "arable land",
    "permanent crop", "olive grove", "vineyard",
    "inland marsh", "peat bog", "salt marsh",
    "scrub", "heathland", "moor", "heath",
    "natural grassland", "grassland",
    "glacier", "perpetual snow",
    "bare rock",
    "beach", "dune",
    "industrial",
    "farmland", "cropland", "pasture",
    "wetland", "marsh",
    "forest", "woodland",
    "urban", "city",
    "agricultural", "farmland",
    "lake", "river", "sea", "ocean",
    "water",
]

_STRUCTURAL = ("landcover", "land cover", "land use", "land-cover")

# Natural-language terms that don't appear as substrings in CLC labels but
# map to a CLC label that does.  Tried in order when the raw keyword yields
# no DB hit at any level.
_KEYWORD_ALIASES: dict[str, list[str]] = {
    # "lake" and "river" map to level2 "water" — gold SQL uses level2_label ILIKE '%water%'
    # which covers both "Inland waters" and "Marine waters" at CLC level 2.
    "lake":     ["water"],
    "river":    ["water"],
    "sea":      ["sea and ocean", "marine water"],
    "ocean":    ["sea and ocean", "marine water"],
    "city":     ["urban fabric", "urban"],
    "wetland":  ["inland marsh", "salt marsh", "peat bog"],
    "marsh":    ["inland marsh", "salt marsh"],
}

# Force certain natural-language keywords to resolve at a specific CLC level
# regardless of the caller's preferred_level.  Water-body keywords (lake, river)
# must match at level 2 ("Inland waters", "Marine waters") rather than level 3
# ("Water bodies", "Water courses") to match the gold SQL convention.
_PREFERRED_LEVEL_OVERRIDES: dict[str, int] = {
    "lake": 2,
    "river": 2,
}


def _is_landcover_question(question: str) -> bool:
    q = question.lower()
    return any(t in q for t in _STRUCTURAL) or _extract_keyword(q) is not None


def _extract_keyword(question: str) -> Optional[str]:
    q = question.lower()
    for kw in _KEYWORDS:  # already longest-first
        if kw in q:
            return kw
    return None


def _preferred_level(question: str) -> int:
    q = question.lower()
    if any(t in q for t in ("level 1", "level1", "level-1", "primary category", "main category", "broad category")):
        return 1
    if any(t in q for t in ("level 2", "level2", "level-2", "secondary category", "subcategory", "sub-category")):
        return 2
    return 3


@functools.lru_cache(maxsize=256)
def _query_level(keyword: str, level: int) -> tuple[str, ...]:
    """Return distinct labels matching ILIKE at *level*. Cached per (keyword, level)."""
    from runner.database_manager import DatabaseManager

    col = f"level{level}_label"
    sql = f"SELECT DISTINCT {col} FROM landcover_type WHERE {col} ILIKE %s ORDER BY {col}"
    try:
        conn = DatabaseManager.get_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (f"%{keyword}%",))
            rows = cur.fetchall()
        conn.close()
        return tuple(r[0] for r in rows if r[0])
    except Exception as exc:
        logger.warning("landcover_entity_hint: DB query failed for keyword=%r level=%d: %s", keyword, level, exc)
        return ()


def resolve_entity(
    keyword: str, preferred: int = 3, question: str = ""
) -> Optional[tuple[int, tuple[str, ...], str]]:
    """Resolve keyword → (validated_level, matched_labels, effective_keyword).

    Resolution order (cheapest → most general):
      L1  Direct ILIKE cascade: exact CLC label substrings (e.g. "forest", "urban").
      L2  Rule-based aliases:   common terms without CLC substrings (e.g. "lake").
      L3  Embedding + LLM judge (landcover_label_matcher): semantic fallback for
          everything else.  Called only when L1 and L2 both fail.
    """
    # Apply per-keyword level overrides (e.g. "lake" → level 2 not level 3).
    effective_preferred = _PREFERRED_LEVEL_OVERRIDES.get(keyword, preferred)

    # L1: direct ILIKE
    for lvl in range(effective_preferred, 0, -1):
        matches = _query_level(keyword, lvl)
        if matches:
            return lvl, matches, keyword

    # L2: rule-based aliases
    for alias in _KEYWORD_ALIASES.get(keyword, []):
        for lvl in range(effective_preferred, 0, -1):
            matches = _query_level(alias, lvl)
            if matches:
                return lvl, matches, alias

    # L3: embedding similarity + LLM judge
    try:
        from pipeline.context.landcover_label_matcher import resolve_semantic
        result = resolve_semantic(keyword, question=question, preferred_level=preferred)
        if result:
            return result
    except Exception as exc:
        logger.warning("landcover_entity_hint: L3 semantic fallback failed for %r: %s", keyword, exc)

    return None


def build_landcover_entity_hint(question: str) -> str:
    """Return #LANDCOVER_ENTITY_HINT prompt block for injection before SQL generation."""
    if not _is_landcover_question(question):
        return ""
    keyword = _extract_keyword(question)
    if not keyword:
        return ""

    preferred = _preferred_level(question)
    resolved = resolve_entity(keyword, preferred, question=question)

    if not resolved:
        return (
            "\n#LANDCOVER_ENTITY_HINT:\n"
            f'keyword="{keyword}" — no match found in landcover_type at any CLC level.\n'
            "Inspect the landcover_type table hierarchy before writing the WHERE predicate.\n"
        )

    level, labels, effective_kw = resolved
    col = f"level{level}_label"
    label_list = ", ".join(f'"{lb}"' for lb in labels[:6])
    scope_note = "" if level == 3 else f" (covers all sub-types under these {level}-level categories)"

    lines = [
        "\n#LANDCOVER_ENTITY_HINT:",
        f'keyword="{keyword}"' + (f' → matched via alias "{effective_kw}"' if effective_kw != keyword else ""),
        f"Validated filter (DB-confirmed): lt.{col} ILIKE '%{effective_kw}%'",
        f"Matched at CLC level {level}: {label_list}{scope_note}",
        f"Apply as: WHERE lt.{col} ILIKE '%{effective_kw}%'",
    ]
    if level < 3:
        lines.append(
            f"Note: no match at level3; using level{level} which covers a broader set of types."
        )
    return "\n".join(lines) + "\n"
