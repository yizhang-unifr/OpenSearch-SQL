"""Offline ChromaDB-based fewshot index builder for OpenSearch-SQL.

Reads training questions from train_data.xlsx, embeds them with
SentenceTransformer, stores in ChromaDB, then for each question in every
eval JSON file retrieves the top-K most similar training examples and writes
data/fewshot/questions.json (the file read by candidate_generate and
align_correct at inference time).

Usage (run from src/OpenSearch-SQL/):
    uv run python src/database_process/build_fewshot_index.py

Options:
    --data_dir       Root data directory                  (default: data)
    --train_xlsx     Path to training Excel file          (default: <project_root>/data/split_config_III_tiered/train_data.xlsx)
    --chroma_dir     Path for ChromaDB storage            (default: <data_dir>/chroma_fewshot)
    --top_k          Number of similar examples to inject (default: 3)
    --model          SentenceTransformer model name       (default: all-mpnet-base-v2)
    --device         Torch device                         (default: cpu)
    --rebuild        Force rebuild even if ChromaDB exists
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTION_NAME = "fewshot_questions"
PROMPT_HEADER = "/* Some SQL examples are provided based on similar problems: */"
EXAMPLE_HEADER = "/* Answer the following:"
EXAMPLE_FOOTER = "*/"


def _build_prompt(examples: list[tuple[str, str]]) -> str:
    """Build fewshot prompt from (question, sql) pairs."""
    if not examples:
        return ""
    parts = [PROMPT_HEADER]
    for q, sql in examples:
        parts.append(f"{EXAMPLE_HEADER}\nQuestion: {q}\n#SQL: {sql}\n{EXAMPLE_FOOTER}")
    return "\n".join(parts)


def build_chroma_collection(
    train_df: pd.DataFrame,
    chroma_dir: Path,
    model: SentenceTransformer,
    rebuild: bool,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not rebuild:
        log.info("Collection '%s' already exists — skipping rebuild", COLLECTION_NAME)
        return client.get_collection(COLLECTION_NAME)

    # Each training row produces TWO documents (point + bbox), giving 2× training
    # examples. Queries are then filtered by geo_mode so retrieved SQL always
    # matches the target variant.
    log.info("Embedding %d training questions → %d documents (point + bbox) …",
             len(train_df), len(train_df) * 2)
    questions = train_df["natural_language_question"].tolist()
    embeddings = model.encode(questions, show_progress_bar=True, batch_size=64)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500  # halved: each row → 2 docs
    for start in range(0, len(train_df), batch_size):
        batch = train_df.iloc[start : start + batch_size]
        batch_embs = embeddings[start : start + batch_size]

        ids, docs, embs, metas = [], [], [], []
        for local_i, (_, row) in enumerate(batch.iterrows()):
            global_i = start + local_i
            q = str(row["natural_language_question"])
            emb = batch_embs[local_i].tolist()
            for variant, sql_col in (("point", "point_query"), ("bbox", "bbox_query")):
                ids.append(f"{global_i}_{variant}")
                docs.append(q)
                embs.append(emb)
                metas.append({
                    "sql": str(row[sql_col] or ""),
                    "geo_mode": variant,
                    "category": str(row.get("category", "")),
                    "place": str(row.get("place", "")),
                })

        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        indexed = min(start + batch_size, len(train_df))
        log.info("  indexed %d / %d rows (%d docs)", indexed, len(train_df), indexed * 2)

    log.info("ChromaDB collection '%s' built with %d documents", COLLECTION_NAME, len(train_df) * 2)
    return collection


def retrieve_top_k(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    question: str,
    top_k: int,
    geo_mode: str,
) -> list[tuple[str, str]]:
    # Filter to only the matching geo_mode variant so retrieved SQL always
    # matches the target query style (point IN-list vs bbox lat/lon range).
    chroma_geo = "bbox" if geo_mode == "bbox" else "point"
    q_emb = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"geo_mode": chroma_geo},
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [(doc, meta["sql"]) for doc, meta in zip(docs, metas) if meta["sql"]]


def process_eval_file(
    eval_path: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
    by_question_out: dict,
) -> int:
    with open(eval_path) as f:
        items = json.load(f)

    added = 0
    for item in tqdm(items, desc=eval_path.name, unit="q"):
        question = item.get("question", "")
        raw_q = item.get("raw_question", question)
        geo_mode = item.get("geo_filter_mode", "points")
        if geo_mode not in ("points", "bbox"):
            geo_mode = "points"

        examples = retrieve_top_k(collection, model, question, top_k, geo_mode)
        prompt = _build_prompt(examples)

        # Key by lowercased raw question — question_id values are not unique
        # across eval files (all files start at 0), so by_question is the only
        # collision-free lookup the pipeline can rely on.
        by_question_out[raw_q.strip().lower()] = {"prompt": prompt}
        added += 1

    return added


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB fewshot index")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_xlsx", default=None)
    parser.add_argument("--chroma_dir", default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and recreate the ChromaDB collection")
    parser.add_argument("--eval_files", nargs="+", default=None,
                        help="Explicit list of eval JSON files to process. "
                             "If omitted, all *.json in data/data_preprocess/ are used.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    _project_root = Path(__file__).resolve().parents[4]
    if args.train_xlsx:
        train_xlsx = Path(args.train_xlsx)
    else:
        candidate = _project_root / "data" / "split_config_III_tiered" / "train_data.xlsx"
        train_xlsx = candidate if candidate.exists() else data_dir.parent / "split_config_III_tiered" / "train_data.xlsx"
    chroma_dir = Path(args.chroma_dir) if args.chroma_dir else data_dir / "chroma_fewshot"
    eval_dir = data_dir / "data_preprocess"
    fewshot_dir = data_dir / "fewshot"

    # ── Load and filter training data ────────────────────────────────────────
    log.info("Loading training data from %s …", train_xlsx)
    train_df = pd.read_excel(train_xlsx)
    log.info("  total rows: %d", len(train_df))
    train_df = train_df[train_df["augmentation_validation"] == True].reset_index(drop=True)
    train_df = train_df.dropna(subset=["natural_language_question", "point_query"])
    train_df = train_df.drop_duplicates(subset=["natural_language_question"]).reset_index(drop=True)
    log.info("  after filtering: %d unique validated questions", len(train_df))

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading SentenceTransformer '%s' on %s …", args.model, args.device)
    model = SentenceTransformer(args.model, device=args.device, local_files_only=True)

    # ── Build ChromaDB collection ─────────────────────────────────────────────
    chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = build_chroma_collection(train_df, chroma_dir, model, args.rebuild)

    # ── Resolve eval JSON files ───────────────────────────────────────────────
    if args.eval_files:
        all_eval = [Path(p) for p in args.eval_files]
        missing = [p for p in all_eval if not p.exists()]
        if missing:
            log.error("Eval files not found: %s", missing)
            sys.exit(1)
    else:
        all_eval = sorted(eval_dir.glob("*.json"))
        if not all_eval:
            log.error("No eval JSON files found in %s", eval_dir)
            sys.exit(1)

    # Process point files before bbox so bbox entries win when the same
    # question appears in both (bbox SQL is more directly useful for bbox eval).
    eval_files = [p for p in all_eval if "bbox" not in p.name] + \
                 [p for p in all_eval if "bbox" in p.name]

    by_question_out: dict = {}

    total = 0
    for eval_path in eval_files:
        n = process_eval_file(eval_path, collection, model, args.top_k, by_question_out)
        log.info("%s → %d entries", eval_path.name, n)
        total += n

    log.info("Total entries generated: %d (by_question=%d unique questions)",
             total, len(by_question_out))

    # ── Write questions.json ──────────────────────────────────────────────────
    # `questions` dict is intentionally empty: all eval files use overlapping
    # question_id ranges (each starts at 0), so keying by question_id would
    # silently return wrong fewshot for most runs. The pipeline falls back to
    # `by_question` (keyed by lowercased question text) which is always correct.
    fewshot_dir.mkdir(parents=True, exist_ok=True)
    out_path = fewshot_dir / "questions.json"
    output = {
        "questions": {},
        "by_question": by_question_out,
        "extract": {},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Written: %s", out_path)


if __name__ == "__main__":
    main()
