"""Unit tests for the gold SQL cache system.

Covers:
  - GoldSqlCacheWriter: incremental write, flush, resume
  - GoldSqlCache: load, has(), get_entry(), get_gold_set() (LRU), get_duration()
  - rescore_with_gold_cache: validate_run_dir(), _rescore_question() logic
  - run_gold_sql: argument parsing, dataset loading path resolution
  - execution.compare_with_cached_gold: correct / incorrect / timeout branches

These tests do NOT require a live PostgreSQL connection; all DB calls are mocked.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_OPENSEARCH   = _HERE.parent
_PROJECT_ROOT = _OPENSEARCH.parents[1]
_SRC          = _OPENSEARCH / "src"
_RUN          = _OPENSEARCH / "run"
_SCRIPTS      = _PROJECT_ROOT / "scripts"

for _p in (_SRC, _RUN, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runner.gold_sql_cache import GoldSqlCache, GoldSqlCacheWriter


# ===========================================================================
# GoldSqlCacheWriter
# ===========================================================================

class TestGoldSqlCacheWriter:

    def test_write_and_flush(self, tmp_path):
        path = tmp_path / "cache.json"
        w = GoldSqlCacheWriter(path, "test_ds", "meteo", total=3, timeout_s=21600)

        entry = {
            "question_id": 0,
            "template_index": 1,
            "category": "A",
            "question": "How warm?",
            "sql": "SELECT 1",
            "geo_filter_mode": "points",
            "result": [[42.0]],
            "duration_s": 0.5,
            "status": "success",
            "error": None,
            "executed_at": "2026-01-01T00:00:00",
        }
        w.write(entry)

        assert path.exists(), "Cache file must be written immediately after write()"
        data = json.loads(path.read_text())
        assert "0" in data["results"]
        assert data["metadata"]["success"] == 1
        assert data["metadata"]["completed"] == 1

    def test_write_error_entry(self, tmp_path):
        path = tmp_path / "cache.json"
        w = GoldSqlCacheWriter(path, "test_ds", "meteo", total=2, timeout_s=21600)

        w.write({
            "question_id": 1, "template_index": 0, "category": "B",
            "question": "q", "sql": "SELECT bad", "geo_filter_mode": "points",
            "result": None, "duration_s": 21600.0,
            "status": "error", "error": "timeout after 21600s",
            "executed_at": "2026-01-01T00:00:00",
        })
        data = json.loads(path.read_text())
        assert data["metadata"]["error"] == 1
        assert data["metadata"]["success"] == 0

    def test_resume_skips_done(self, tmp_path):
        path = tmp_path / "cache.json"
        w1 = GoldSqlCacheWriter(path, "ds", "meteo", total=5, timeout_s=3600)
        w1.write({"question_id": 0, "template_index": 0, "category": "A",
                  "question": "q", "sql": "SELECT 1", "geo_filter_mode": "points",
                  "result": [[1]], "duration_s": 0.1, "status": "success",
                  "error": None, "executed_at": "2026-01-01T00:00:00"})

        # Create new writer pointing to same file (resume)
        w2 = GoldSqlCacheWriter(path, "ds", "meteo", total=5, timeout_s=3600)
        assert w2.already_done(0), "Resumed writer must see previously written entry"
        assert not w2.already_done(1)
        assert w2.completed == 1

    def test_atomic_write_survives_simulated_interrupt(self, tmp_path):
        """The .tmp → rename pattern means partial writes never corrupt the file."""
        path = tmp_path / "cache.json"
        w = GoldSqlCacheWriter(path, "ds", "meteo", total=10, timeout_s=3600)
        for i in range(3):
            w.write({"question_id": i, "template_index": 0, "category": "A",
                     "question": f"q{i}", "sql": "SELECT 1", "geo_filter_mode": "points",
                     "result": [[i]], "duration_s": 0.1, "status": "success",
                     "error": None, "executed_at": "2026-01-01T00:00:00"})

        data = json.loads(path.read_text())
        assert len(data["results"]) == 3

    def test_no_tmp_file_left_after_write(self, tmp_path):
        path = tmp_path / "cache.json"
        w = GoldSqlCacheWriter(path, "ds", "meteo", total=1, timeout_s=3600)
        w.write({"question_id": 0, "template_index": 0, "category": "A",
                 "question": "q", "sql": "SELECT 1", "geo_filter_mode": "points",
                 "result": [[1]], "duration_s": 0.1, "status": "success",
                 "error": None, "executed_at": "2026-01-01T00:00:00"})
        assert not path.with_suffix(".tmp").exists(), ".tmp file must be removed after rename"

    def test_concurrent_writes_no_data_loss(self, tmp_path):
        """N threads writing distinct entries concurrently must all appear in the cache."""
        path = tmp_path / "cache.json"
        n = 40
        w = GoldSqlCacheWriter(path, "ds", "meteo", total=n, timeout_s=3600)

        def _write(i):
            w.write({"question_id": i, "template_index": 0, "category": "A",
                     "question": f"q{i}", "sql": "SELECT 1", "geo_filter_mode": "points",
                     "result": [[i]], "duration_s": 0.01, "status": "success",
                     "error": None, "executed_at": "2026-01-01T00:00:00"})

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()

        data = json.loads(path.read_text())
        assert len(data["results"]) == n, "All concurrent writes must be persisted"
        assert data["metadata"]["completed"] == n
        assert data["metadata"]["success"]   == n

    def test_already_done_thread_safe(self, tmp_path):
        """already_done() and write() called from multiple threads don't race."""
        path = tmp_path / "cache.json"
        w = GoldSqlCacheWriter(path, "ds", "meteo", total=20, timeout_s=3600)
        duplicates = []

        def _worker(i):
            if not w.already_done(i):
                w.write({"question_id": i, "template_index": 0, "category": "A",
                         "question": "q", "sql": "S", "geo_filter_mode": "points",
                         "result": [[i]], "duration_s": 0.01, "status": "success",
                         "error": None, "executed_at": "2026-01-01T00:00:00"})

        # Run each question twice from different threads to stress-test
        threads = []
        for i in range(10):
            for _ in range(2):
                threads.append(threading.Thread(target=_worker, args=(i,)))
        for t in threads: t.start()
        for t in threads: t.join()

        # Each question_id should appear at most once
        data = json.loads(path.read_text())
        assert len(data["results"]) == 10, "Duplicate writes must not corrupt the count"


# ===========================================================================
# GoldSqlCache (reader)
# ===========================================================================

def _make_cache_file(tmp_path: Path, results: dict) -> Path:
    """Write a minimal cache JSON and return its path."""
    data = {
        "metadata": {
            "dataset": "test_ds", "db_id": "meteo", "timeout_s": 21600,
            "created_at": "2026-01-01T00:00:00", "last_updated": "2026-01-01T00:00:01",
            "total": len(results), "completed": len(results),
            "success": sum(1 for e in results.values() if e["status"] == "success"),
            "error": sum(1 for e in results.values() if e["status"] == "error"),
        },
        "results": results,
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(data))
    return path


class TestGoldSqlCache:

    def test_load_empty_when_file_missing(self, tmp_path):
        cache = GoldSqlCache(tmp_path / "nonexistent.json")
        assert not cache.is_loaded
        assert not cache.has(0)

    def test_has_and_get_entry(self, tmp_path):
        results = {
            "0": {"question_id": 0, "status": "success", "result": [[1.0], [2.0]],
                  "duration_s": 0.5, "category": "A", "geo_filter_mode": "points",
                  "template_index": 1, "question": "q", "sql": "SELECT 1", "error": None,
                  "executed_at": "2026-01-01T00:00:00"},
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path)

        assert cache.is_loaded
        assert cache.has(0)
        assert not cache.has(99)
        entry = cache.get_entry(0)
        assert entry["category"] == "A"
        assert entry["duration_s"] == 0.5

    def test_get_gold_set_success(self, tmp_path):
        results = {
            "5": {"question_id": 5, "status": "success",
                  "result": [[10, 20], [30, 40]], "duration_s": 1.0,
                  "category": "B", "geo_filter_mode": "points",
                  "template_index": 2, "question": "q", "sql": "S", "error": None,
                  "executed_at": "2026-01-01T00:00:00"},
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path)
        gold  = cache.get_gold_set(5)
        assert gold == frozenset({(10, 20), (30, 40)})

    def test_get_gold_set_error_entry_returns_none(self, tmp_path):
        results = {
            "3": {"question_id": 3, "status": "error", "result": None,
                  "duration_s": 21600.0, "category": "A", "geo_filter_mode": "points",
                  "template_index": 0, "question": "q", "sql": "S", "error": "timeout",
                  "executed_at": "2026-01-01T00:00:00"},
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path)
        assert cache.get_gold_set(3) is None

    def test_lru_caches_conversion(self, tmp_path):
        results = {
            "7": {"question_id": 7, "status": "success",
                  "result": [[1]], "duration_s": 0.2,
                  "category": "A", "geo_filter_mode": "points",
                  "template_index": 0, "question": "q", "sql": "S", "error": None,
                  "executed_at": "2026-01-01T00:00:00"},
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path, lru_maxsize=5)
        g1 = cache.get_gold_set(7)
        g2 = cache.get_gold_set(7)
        assert g1 is g2, "LRU must return the same frozenset object on second call"

    def test_lru_eviction(self, tmp_path):
        """With maxsize=2, the third unique entry evicts the first."""
        results = {
            str(i): {"question_id": i, "status": "success", "result": [[i]],
                     "duration_s": 0.1, "category": "A", "geo_filter_mode": "points",
                     "template_index": 0, "question": "q", "sql": "S", "error": None,
                     "executed_at": "2026-01-01T00:00:00"}
            for i in range(5)
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path, lru_maxsize=2)
        _ = cache.get_gold_set(0)
        _ = cache.get_gold_set(1)
        _ = cache.get_gold_set(2)   # evicts 0
        # Cache still returns correct values (re-computed from raw)
        assert cache.get_gold_set(0) == frozenset({(0,)})

    def test_thread_safety(self, tmp_path):
        """Multiple threads may call get_gold_set concurrently without error."""
        results = {
            str(i): {"question_id": i, "status": "success", "result": [[i, i*2]],
                     "duration_s": 0.1, "category": "A", "geo_filter_mode": "points",
                     "template_index": 0, "question": "q", "sql": "S", "error": None,
                     "executed_at": "2026-01-01T00:00:00"}
            for i in range(20)
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path, lru_maxsize=5)
        errors = []

        def _worker(q_ids):
            for qid in q_ids:
                try:
                    cache.get_gold_set(qid)
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=_worker, args=(range(20),)) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Thread errors: {errors}"

    def test_cache_path_helper(self):
        root = Path("/project")
        p = GoldSqlCache.cache_path(root, "test_data_point", "meteo")
        assert p == Path("/project/data/gold_sql_cache/test_data_point_meteo_gold_sql.json")

    def test_get_duration(self, tmp_path):
        results = {
            "9": {"question_id": 9, "status": "success", "result": [[1]],
                  "duration_s": 3.14, "category": "A", "geo_filter_mode": "points",
                  "template_index": 0, "question": "q", "sql": "S", "error": None,
                  "executed_at": "2026-01-01T00:00:00"},
        }
        path  = _make_cache_file(tmp_path, results)
        cache = GoldSqlCache(path)
        assert cache.get_duration(9) == pytest.approx(3.14)
        assert cache.get_duration(99) is None


# ===========================================================================
# execution.compare_with_cached_gold
# ===========================================================================

class TestCompareWithCachedGold:

    def _import(self):
        from runner.execution import compare_with_cached_gold
        return compare_with_cached_gold

    def test_correct_result(self):
        fn = self._import()
        gold_set = frozenset({(1.0,), (2.0,)})

        with patch("runner.execution.func_timeout") as mock_ft:
            mock_ft.return_value = (frozenset({(1.0,), (2.0,)}), 0.3)
            result = fn("SELECT 1", gold_set, t_gold=1.0, meta_time_out=60)

        assert result["exec_res"] == 1
        assert result["exec_err"] == "--"
        assert 0.0 < result["ves"] <= 1.0

    def test_incorrect_result(self):
        fn = self._import()
        gold_set = frozenset({(1.0,)})

        with patch("runner.execution.func_timeout") as mock_ft:
            mock_ft.return_value = (frozenset({(99.0,)}), 0.3)
            result = fn("SELECT 99", gold_set, t_gold=1.0, meta_time_out=60)

        assert result["exec_res"] == 0
        assert result["exec_err"] == "incorrect answer"
        assert result["ves"] == 0.0

    def test_timeout(self):
        fn = self._import()
        from func_timeout import FunctionTimedOut
        gold_set = frozenset({(1,)})

        with patch("runner.execution.func_timeout", side_effect=FunctionTimedOut("", 1, None, ())):
            result = fn("SELECT SLOW", gold_set, t_gold=1.0, meta_time_out=1)

        assert result["exec_res"] == 0
        assert result["exec_err"] == "timeout"
        assert result["ves"] == 0.0

    def test_ves_formula_fast_pred(self):
        """When predicted is twice as fast as gold, VES = sqrt(0.5) ≈ 0.707."""
        fn = self._import()
        gold_set = frozenset({(42,)})
        t_gold = 2.0
        t_pred = 1.0   # twice as fast

        with patch("runner.execution.func_timeout") as mock_ft:
            mock_ft.return_value = (frozenset({(42,)}), t_pred)
            result = fn("SELECT 42", gold_set, t_gold=t_gold, meta_time_out=60)

        assert result["exec_res"] == 1
        expected_ves = math.sqrt(min(t_gold / t_pred, 1.0))
        assert result["ves"] == pytest.approx(expected_ves, rel=1e-6)

    def test_ves_capped_at_one_when_gold_slower(self):
        """When gold is slower than pred, ratio > 1 → capped at 1 → VES = 1."""
        fn = self._import()
        gold_set = frozenset({(1,)})

        with patch("runner.execution.func_timeout") as mock_ft:
            mock_ft.return_value = (frozenset({(1,)}), 10.0)   # pred very slow
            result = fn("SELECT 1", gold_set, t_gold=0.1, meta_time_out=60)

        assert result["ves"] == pytest.approx(math.sqrt(min(0.1 / 10.0, 1.0)))

    def test_execution_exception_returns_error(self):
        fn = self._import()
        gold_set = frozenset({(1,)})

        with patch("runner.execution.func_timeout", side_effect=RuntimeError("connection refused")):
            result = fn("SELECT 1", gold_set, t_gold=1.0, meta_time_out=60)

        assert result["exec_res"] == 0
        assert "connection refused" in result["exec_err"]


# ===========================================================================
# rescore_with_gold_cache helpers
# ===========================================================================

class TestRescoreHelpers:

    def _build_run_dir(self, tmp_path: Path, db_id: str = "meteo") -> Path:
        """Create a minimal run directory with -statistics.json and one question file."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "-statistics.json").write_text(json.dumps({"counts": {}, "ids": {}}))
        q_file = run_dir / f"0_{db_id}.json"
        nodes = [
            {"node_type": "candidate_generate", "status": "success", "duration_s": 1.0,
             "SQL": ["SELECT 1"]},
            {
                "node_type": "evaluation", "duration_s": 5.0,
                "candidate_generate": {
                    "exec_res": 0, "exec_err": "timeout", "ves": 0.0,
                    "Question": "test question", "Evidence": "None",
                    "GOLD_SQL": "SELECT 1", "GOLD_RESULT": None,
                    "PREDICTED_SQL": "SELECT 1", "PREDICTED_RESULT": [[1]],
                },
                "align_correct": {
                    "exec_res": 0, "exec_err": "timeout", "ves": 0.0,
                    "Question": "test question", "Evidence": "None",
                    "GOLD_SQL": "SELECT 1", "GOLD_RESULT": None,
                    "PREDICTED_SQL": "SELECT 1", "PREDICTED_RESULT": [[1]],
                },
                "vote": {
                    "exec_res": 0, "exec_err": "timeout", "ves": 0.0,
                    "Question": "test question", "Evidence": "None",
                    "GOLD_SQL": "SELECT 1", "GOLD_RESULT": None,
                    "PREDICTED_SQL": "SELECT 1", "PREDICTED_RESULT": [[1]],
                },
            },
        ]
        q_file.write_text(json.dumps(nodes))
        return run_dir

    def _build_cache(self, tmp_path: Path, q_id: int = 0) -> GoldSqlCache:
        results = {
            str(q_id): {
                "question_id": q_id, "status": "success",
                "result": [[1]], "duration_s": 0.5,
                "category": "A", "geo_filter_mode": "points",
                "template_index": 0, "question": "test", "sql": "SELECT 1",
                "error": None, "executed_at": "2026-01-01T00:00:00",
            }
        }
        path  = _make_cache_file(tmp_path, results)
        return GoldSqlCache(path)

    def test_validate_run_dir_ok(self, tmp_path):
        from rescore_with_gold_cache import validate_run_dir
        run_dir = self._build_run_dir(tmp_path)
        files = validate_run_dir(run_dir, "meteo")
        assert len(files) == 1
        assert files[0].name == "0_meteo.json"

    def test_validate_run_dir_missing_stats(self, tmp_path):
        from rescore_with_gold_cache import validate_run_dir
        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        (run_dir / "0_meteo.json").write_text("[]")
        with pytest.raises(ValueError, match="-statistics.json"):
            validate_run_dir(run_dir, "meteo")

    def test_validate_run_dir_missing_question_files(self, tmp_path):
        from rescore_with_gold_cache import validate_run_dir
        run_dir = tmp_path / "no_questions"
        run_dir.mkdir()
        (run_dir / "-statistics.json").write_text("{}")
        with pytest.raises(ValueError, match="No \\*_meteo.json"):
            validate_run_dir(run_dir, "meteo")

    def test_rescore_question_updates_fields(self, tmp_path):
        from rescore_with_gold_cache import _rescore_question

        run_dir = self._build_run_dir(tmp_path)
        cache   = self._build_cache(tmp_path)

        with open(run_dir / "0_meteo.json") as f:
            nodes = json.load(f)

        with patch("rescore_with_gold_cache.compare_with_cached_gold",
                   return_value={"exec_res": 1, "exec_err": "--", "ves": 0.9}):
            new_nodes, delta = _rescore_question(nodes, cache, q_id=0, dry_run=False)

        ev = next(n for n in new_nodes if n.get("node_type") == "evaluation")
        for stage in ("candidate_generate", "align_correct", "vote"):
            sd = ev[stage]
            assert sd["exec_res"] == 1,        f"{stage}: exec_res not updated"
            assert sd["exec_err"] == "--",     f"{stage}: exec_err not updated"
            assert sd["ves"] == pytest.approx(0.9), f"{stage}: ves not updated"
            assert sd["GOLD_RESULT"] == [[1]], f"{stage}: GOLD_RESULT not updated"
            assert sd["gold_from_cache"] is True, f"{stage}: gold_from_cache not added"
            # Verify no keys were removed
            assert "PREDICTED_SQL"   in sd
            assert "PREDICTED_RESULT" in sd
            assert "GOLD_SQL"        in sd
            assert "Question"        in sd

    def test_rescore_question_dry_run_no_changes(self, tmp_path):
        from rescore_with_gold_cache import _rescore_question

        run_dir = self._build_run_dir(tmp_path)
        cache   = self._build_cache(tmp_path)

        with open(run_dir / "0_meteo.json") as f:
            original_nodes = json.load(f)
        import copy
        original_snapshot = copy.deepcopy(original_nodes)

        new_nodes, delta = _rescore_question(original_nodes, cache, q_id=0, dry_run=True)

        # Dry run must not modify the nodes
        ev_orig = next(n for n in original_snapshot if n.get("node_type") == "evaluation")
        ev_new  = next(n for n in new_nodes         if n.get("node_type") == "evaluation")
        assert ev_orig["vote"]["exec_res"] == ev_new["vote"]["exec_res"]

    def test_rescore_question_missing_from_cache_no_change(self, tmp_path):
        from rescore_with_gold_cache import _rescore_question

        run_dir = self._build_run_dir(tmp_path)
        # Build cache with a DIFFERENT question id
        cache   = self._build_cache(tmp_path, q_id=999)

        with open(run_dir / "0_meteo.json") as f:
            nodes = json.load(f)

        new_nodes, delta = _rescore_question(nodes, cache, q_id=0, dry_run=False)
        # delta must be empty (no rescoring for q_id=0 since not in cache)
        assert delta == {}


# ===========================================================================
# run_gold_sql helpers
# ===========================================================================

class TestRunGoldSqlHelpers:

    def test_parse_args_defaults(self):
        from run_gold_sql import parse_args
        args = parse_args([])
        assert args.dataset == "test_data_point"
        assert args.db_id   == "meteo"
        assert args.timeout == 7200
        assert args.end     == -1
        assert not args.retry_errors
        assert not args.dry_run

    def test_parse_args_custom(self):
        from run_gold_sql import parse_args
        args = parse_args(["--dataset", "foo", "--start", "10", "--end", "50",
                           "--timeout", "3600", "--retry-errors"])
        assert args.dataset == "foo"
        assert args.start   == 10
        assert args.end     == 50
        assert args.timeout == 3600
        assert args.retry_errors

    def test_load_dataset_from_json_file(self, tmp_path):
        from run_gold_sql import _load_dataset

        data = [{"question_id": 0, "SQL": "SELECT 1", "question": "q",
                 "category": "A", "geo_filter_mode": "points"}]
        ds_file = tmp_path / "my_data.json"
        ds_file.write_text(json.dumps(data))
        loaded = _load_dataset(str(ds_file))
        assert len(loaded) == 1
        assert loaded[0]["question_id"] == 0
