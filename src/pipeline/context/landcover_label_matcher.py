"""Semantic label matcher: embedding similarity + LLM judge for CLC entity resolution.

Architecture (two independent stages):

  Stage 1 — Embedding retrieval (find_top_k):
    Load all CLC labels from DB once, embed with SentenceTransformer (L2-normalised).
    Cosine similarity (dot product) ranks candidates.  Fully deterministic; no LLM.

  Stage 2 — LLM judge (judge_label):
    Prompt LLM with keyword, question context, and top-k candidates.
    LLM picks the single best CLC label.  Cached per (keyword, preferred_level).

  Entry point (resolve_semantic):
    Combines stages 1 + 2, then validates chosen label exists in DB.
    Returns (level, matched_labels, effective_kw) — same contract as
    landcover_entity_hint.resolve_entity() — or None on failure.

Called only as the L3 fallback in resolve_entity() after direct ILIKE and
rule-based aliases have both failed.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TOP_K = 5


# ── label corpus ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_clc_labels() -> tuple[tuple[str, int], ...]:
    """Return all (label, level) pairs from landcover_type, level3-first. Cached once."""
    from runner.database_manager import DatabaseManager
    rows: list[tuple[str, int]] = []
    try:
        conn = DatabaseManager.get_connection()
        with conn.cursor() as cur:
            for level in (3, 2, 1):
                col = f"level{level}_label"
                cur.execute(
                    f"SELECT DISTINCT {col} FROM landcover_type "
                    f"WHERE {col} IS NOT NULL ORDER BY {col}"
                )
                for (lbl,) in cur.fetchall():
                    rows.append((lbl, level))
        conn.close()
    except Exception as exc:
        logger.warning("landcover_label_matcher: failed to load CLC labels: %s", exc)
    return tuple(rows)


# ── embedding model & pre-computed label vectors ──────────────────────────────

@functools.lru_cache(maxsize=1)
def _embedding_model():
    """Lazy-load SentenceTransformer, reusing pipeline config when available."""
    from sentence_transformers import SentenceTransformer
    model_name = "all-mpnet-base-v2"
    device = "cpu"
    try:
        from pipeline.pipeline_manager import PipelineManager
        config, _ = PipelineManager().get_model_para()
        model_name = config.get("bert_model", model_name)
        device = config.get("device", device)
    except Exception:
        pass
    return SentenceTransformer(model_name, device=device, local_files_only=True)


@functools.lru_cache(maxsize=1)
def _label_vectors() -> tuple[list[tuple[str, int]], np.ndarray]:
    """Pre-compute L2-normalised embeddings for all CLC labels.  Computed once per process."""
    labels = list(_load_clc_labels())
    if not labels:
        return labels, np.empty((0, 0), dtype=np.float32)
    model = _embedding_model()
    texts = [lbl for lbl, _ in labels]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return labels, embs.astype(np.float32)


# ── stage 1: embedding retrieval ─────────────────────────────────────────────

def find_top_k(keyword: str, k: int = _TOP_K) -> list[tuple[float, str, int]]:
    """Return top-k (score, label, level) sorted descending by cosine similarity to keyword."""
    labels, embs = _label_vectors()
    if len(labels) == 0:
        return []
    model = _embedding_model()
    kw_vec = model.encode([keyword], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = (embs @ kw_vec.astype(np.float32)).tolist()
    ranked = sorted(
        zip(scores, (lbl for lbl, _ in labels), (lvl for _, lvl in labels)),
        key=lambda t: t[0],
        reverse=True,
    )
    return list(ranked[:k])


# ── stage 2: LLM judge ────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are a CORINE Land Cover (CLC) classification expert.

Task: select the single CLC label that best represents the concept below.

Keyword  : "{keyword}"
Context  : "{question}"
Preferred level: {preferred_level} (favour level-{preferred_level} entries when possible)

Top candidates by semantic similarity:
{candidates}

Rules:
- Return ONLY the exact label text as shown in the list above (copy it verbatim).
- Do not add quotes, explanation, or extra text.
- If none of the candidates is a reasonable match, return exactly: NONE
"""


@functools.lru_cache(maxsize=256)
def _judge_cached(
    keyword: str,
    preferred_level: int,
    candidates_repr: str,
) -> Optional[str]:
    """LLM judge result, cached per (keyword, preferred_level, candidates).

    Question context is intentionally excluded from the cache key: CLC entity
    mapping (lake → Water bodies) is stable across question phrasings.
    """
    try:
        from pipeline.pipeline_manager import PipelineManager
        from llm.model import model_chose
        config, node_name = PipelineManager().get_model_para()
        chat_model = model_chose(node_name, config["engine"])
    except Exception as exc:
        logger.warning("landcover_label_matcher: LLM init failed: %s", exc)
        return None

    prompt = _JUDGE_PROMPT.format(
        keyword=keyword,
        question="(not provided)",
        preferred_level=preferred_level,
        candidates=candidates_repr,
    )
    try:
        raw = chat_model.get_ans(prompt, 0.0).strip().strip("\"'")
        if not raw or raw.upper() == "NONE":
            return None
        return raw
    except Exception as exc:
        logger.warning("landcover_label_matcher: LLM call failed: %s", exc)
        return None


def judge_label(
    keyword: str,
    candidates: list[tuple[float, str, int]],
    preferred_level: int = 3,
    question: str = "",
) -> Optional[tuple[str, int]]:
    """Ask LLM to select best (label, level) from candidates.

    question is forwarded to the prompt for context but not used as a cache key
    (see _judge_cached docstring).
    """
    if not candidates:
        return None

    candidate_lines = "\n".join(
        f"  {i + 1}. [Level {lvl}] {lbl}  (similarity {score:.3f})"
        for i, (score, lbl, lvl) in enumerate(candidates)
    )

    chosen_label = _judge_cached(keyword, preferred_level, candidate_lines)
    if not chosen_label:
        return None

    chosen_lower = chosen_label.lower()

    # Exact match (case-insensitive)
    for _score, lbl, lvl in candidates:
        if lbl.lower() == chosen_lower:
            return lbl, lvl

    # Partial match fallback (LLM occasionally truncates)
    for _score, lbl, lvl in candidates:
        if chosen_lower in lbl.lower() or lbl.lower() in chosen_lower:
            logger.info(
                "landcover_label_matcher: partial match %r → %r (L%d)", chosen_label, lbl, lvl
            )
            return lbl, lvl

    logger.warning(
        "landcover_label_matcher: LLM returned %r which does not match any candidate %s",
        chosen_label,
        [lbl for _, lbl, _ in candidates],
    )
    return None


# ── DB validation ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def _db_exact_label(label: str, level: int) -> tuple[str, ...]:
    """Return tuple with *label* if it exists at *level* in landcover_type, else empty."""
    from runner.database_manager import DatabaseManager
    col = f"level{level}_label"
    sql = f"SELECT DISTINCT {col} FROM landcover_type WHERE {col} = %s"
    try:
        conn = DatabaseManager.get_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (label,))
            rows = cur.fetchall()
        conn.close()
        return tuple(r[0] for r in rows if r[0])
    except Exception as exc:
        logger.warning("landcover_label_matcher: DB validation failed for %r L%d: %s", label, level, exc)
        return ()


# ── entry point ───────────────────────────────────────────────────────────────

def resolve_semantic(
    keyword: str,
    question: str = "",
    preferred_level: int = 3,
) -> Optional[tuple[int, tuple[str, ...], str]]:
    """Semantic resolution: embed → judge → DB validate.

    Returns (validated_level, matched_labels, effective_keyword) or None.
    Contract matches landcover_entity_hint.resolve_entity().
    """
    candidates = find_top_k(keyword, k=_TOP_K)
    if not candidates:
        logger.warning("landcover_label_matcher: empty corpus, cannot resolve %r", keyword)
        return None

    best = judge_label(keyword, candidates, preferred_level=preferred_level, question=question)
    if not best:
        logger.warning("landcover_label_matcher: no judge match for keyword=%r", keyword)
        return None

    chosen_label, chosen_level = best

    # Try chosen level first, then other levels if DB has a miss
    for lvl in [chosen_level] + [l for l in (3, 2, 1) if l != chosen_level]:
        matches = _db_exact_label(chosen_label, lvl)
        if matches:
            if lvl != chosen_level:
                logger.info(
                    "landcover_label_matcher: %r found at level %d (judge said %d)",
                    chosen_label, lvl, chosen_level,
                )
            return lvl, matches, chosen_label

    logger.warning("landcover_label_matcher: %r not found in DB at any level", chosen_label)
    return None
