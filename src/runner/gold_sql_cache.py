"""Gold SQL result cache with LRU-backed in-memory access.

The cache file lives at ``<project_root>/data/gold_sql_cache/<dataset>_<db_id>_gold_sql.json``
and is shared by both OpenSearch-SQL and DAIL-SQL pipelines.

File format
-----------
{
  "metadata": {
    "dataset": "test_data_point",
    "db_id": "meteo",
    "timeout_s": 21600,
    "created_at": "...",
    "last_updated": "...",
    "total": 752,
    "completed": 700,
    "success": 680,
    "error": 20
  },
  "results": {
    "0": {
      "question_id": 0,
      "template_index": 0,
      "category": "A",
      "question": "...",
      "sql": "SELECT ...",
      "geo_filter_mode": "points",
      "result": [[val, ...], ...],   // null on error
      "duration_s": 1.23,
      "status": "success",           // "success" | "error"
      "error": null,
      "executed_at": "2026-05-21T10:00:00"
    }
  }
}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from cachetools import LRUCache


class GoldSqlCache:
    """Read-only accessor for pre-computed gold SQL results.

    Raw entry dicts are loaded once from disk into memory.  Converting
    each result list to a frozenset (needed for set-based comparison) is
    expensive for large result sets, so those conversions are cached with
    an LRU eviction policy.
    """

    def __init__(self, cache_path: Path, lru_maxsize: int = 200):
        self._path = cache_path
        self._metadata: dict = {}
        self._raw: dict[str, dict] = {}          # str(question_id) -> entry
        self._lru: LRUCache = LRUCache(maxsize=lru_maxsize)
        self._lock = threading.Lock()

        if cache_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        self._metadata = data.get("metadata", {})
        self._raw = data.get("results", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return bool(self._raw)

    def has(self, question_id: int) -> bool:
        return str(question_id) in self._raw

    def get_entry(self, question_id: int) -> Optional[dict]:
        """Return the raw cache entry for *question_id*, or None."""
        return self._raw.get(str(question_id))

    def get_gold_set(self, question_id: int) -> Optional[frozenset]:
        """Return a frozenset of gold result tuples (LRU-cached conversion).

        Returns None if question_id is absent or the cached status is 'error'.
        """
        key = str(question_id)
        with self._lock:
            if key in self._lru:
                return self._lru[key]

        entry = self._raw.get(key)
        if entry is None or entry.get("status") != "success":
            return None
        result = entry.get("result")
        if result is None:
            return None

        gold_set = frozenset(
            tuple(row) if not isinstance(row, tuple) else row
            for row in result
        )
        with self._lock:
            self._lru[key] = gold_set
        return gold_set

    def get_duration(self, question_id: int) -> Optional[float]:
        """Return gold SQL execution time in seconds, or None."""
        entry = self._raw.get(str(question_id))
        if entry and entry.get("status") == "success":
            return entry.get("duration_s")
        return None

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cache_path(project_root: Path, dataset: str, db_id: str) -> Path:
        """Return the canonical cache file path."""
        return project_root / "data" / "gold_sql_cache" / f"{dataset}_{db_id}_gold_sql.json"


class GoldSqlCacheWriter:
    """Incremental, thread-safe writer used exclusively by run_gold_sql.py.

    Multiple threads may call write() concurrently.  A single lock serialises
    all mutations and flushes so the on-disk file is always consistent.
    """

    def __init__(
        self,
        cache_path: Path,
        dataset: str,
        db_id: str,
        total: int,
        timeout_s: int,
    ):
        self._path = cache_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Load existing progress if resuming
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                existing = json.load(f)
            self._metadata: dict[str, Any] = existing.get("metadata", {})
            self._results: dict[str, dict] = existing.get("results", {})
        else:
            from datetime import datetime
            self._metadata = {
                "dataset": dataset,
                "db_id": db_id,
                "timeout_s": timeout_s,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "last_updated": "",
                "total": total,
                "completed": 0,
                "success": 0,
                "error": 0,
            }
            self._results = {}

    # ------------------------------------------------------------------

    def already_done(self, question_id: int) -> bool:
        with self._lock:
            return str(question_id) in self._results

    def write(self, entry: dict) -> None:
        """Append *entry* and flush to disk immediately (thread-safe)."""
        from datetime import datetime

        key = str(entry["question_id"])
        with self._lock:
            self._results[key] = entry
            success = sum(1 for e in self._results.values() if e["status"] == "success")
            error   = sum(1 for e in self._results.values() if e["status"] == "error")
            self._metadata["completed"]    = len(self._results)
            self._metadata["success"]      = success
            self._metadata["error"]        = error
            self._metadata["last_updated"] = datetime.now().isoformat(timespec="seconds")
            self._flush()

    def _flush(self) -> None:
        """Write cache to disk atomically.  Must be called under self._lock."""
        payload = {"metadata": self._metadata, "results": self._results}
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
            f.flush()
        tmp.replace(self._path)   # atomic rename

    @property
    def completed(self) -> int:
        with self._lock:
            return len(self._results)

    @property
    def skipped_ids(self) -> set[int]:
        with self._lock:
            return {int(k) for k in self._results}
