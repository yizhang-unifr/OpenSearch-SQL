"""Gold-SQL regression test for mechanical_optimize.

Tests that the mechanical IN-list → VALUES-subquery rewrite produces
identical PostgreSQL results to the original SQL for all 376 Canton of
Zurich point_query rows in augmented_dataset.xlsx.

Usage:
    uv run python mechanical_optimize_gold_sql_test.py

Total wall-clock timeout: 10 minutes.
Per-SQL execution timeout: 30 s (via PostgreSQL statement_timeout).
Concurrency: up to 8 parallel SQL-pair comparisons.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

from pipeline.optimizer.mechanical import mechanical_optimize, verify_mechanical_transform
from pipeline.optimizer.detector import needs_optimization, extract_points, _ROUND_IN_RE
from runner.execution import sql_exec

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "augmented_dataset.xlsx"
PLACE_FILTER = "Canton of Zurich"
TOTAL_TIMEOUT_S = 600       # 10 minutes for the whole test suite
PER_SQL_TIMEOUT_S = 30      # PostgreSQL statement_timeout per SQL call
MAX_WORKERS = 8             # parallel SQL-pair comparisons


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    idx: int
    pair_count: int
    needs_opt: bool          # would trigger in production (pair_count > threshold)
    transformed: bool        # mechanical_optimize returned transformed=True
    verified: bool           # verify_mechanical_transform passed
    status: str              # ok | no_transform | verify_fail | result_mismatch
                             #   | orig_error | opt_error | timeout | skipped
    match: bool              # original result == optimized result
    orig_time_s: float = 0.0
    opt_time_s: float = 0.0
    error: str = ""
    orig_sql_head: str = ""
    opt_sql_head: str = ""


# ── Core comparison logic ─────────────────────────────────────────────────────

def _exec_sql(sql: str) -> tuple[set, float]:
    """Run SQL with per-SQL timeout; returns (result_set, elapsed_s)."""
    # statement_timeout is set inside sql_exec via _get_pg_connection
    return sql_exec(sql)


def compare_pair(idx: int, original: str, optimized: str) -> QueryResult:
    """Execute both SQLs and compare result sets. No outer timeout here —
    the ThreadPoolExecutor handles the overall deadline."""
    pair_count = len(extract_points(original) or [])
    needs_opt = needs_optimization(original)

    # Mechanical transform
    rewritten, transformed = mechanical_optimize(original)
    verified = transformed and verify_mechanical_transform(original, rewritten)

    result = QueryResult(
        idx=idx,
        pair_count=pair_count,
        needs_opt=needs_opt,
        transformed=transformed,
        verified=verified,
        status="",
        match=False,
        orig_sql_head=original[:200],
        opt_sql_head=(rewritten or original)[:200],
    )

    if not transformed:
        result.status = "no_transform"
        return result
    if not verified:
        result.status = "verify_fail"
        return result

    # Execute original
    try:
        orig_rows, orig_t = _exec_sql(original)
        result.orig_time_s = orig_t
    except Exception as exc:
        result.status = "orig_error"
        result.error = str(exc)[:300]
        return result

    # Execute optimized
    try:
        opt_rows, opt_t = _exec_sql(rewritten)
        result.opt_time_s = opt_t
    except Exception as exc:
        result.status = "opt_error"
        result.error = str(exc)[:300]
        return result

    result.match = (orig_rows == opt_rows)
    result.status = "ok" if result.match else "result_mismatch"
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_PATH} …")
    df = pd.read_excel(DATA_PATH)
    zurich = df[df["place"] == PLACE_FILTER].reset_index(drop=True)
    queries = zurich["point_query"].tolist()

    print(f"Canton of Zurich rows: {len(queries)}")
    print(f"Total timeout: {TOTAL_TIMEOUT_S}s  |  per-SQL timeout: {PER_SQL_TIMEOUT_S}s  "
          f"|  workers: {MAX_WORKERS}")
    print("-" * 70)

    results: list[QueryResult] = []
    wall_start = time.time()
    deadline = wall_start + TOTAL_TIMEOUT_S

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    future_map = {
        executor.submit(compare_pair, i, sql, sql): i
        for i, sql in enumerate(queries)
    }

    completed = 0
    try:
        for future in as_completed(future_map, timeout=TOTAL_TIMEOUT_S):
            completed += 1
            idx = future_map[future]
            try:
                res = future.result(timeout=0)
            except Exception as exc:
                res = QueryResult(
                    idx=idx, pair_count=0, needs_opt=False, transformed=False,
                    verified=False, status="timeout", match=False,
                    error=str(exc)[:200],
                )
            results.append(res)
            # Live progress every 20 queries
            if completed % 20 == 0 or completed == len(queries):
                elapsed = time.time() - wall_start
                print(f"  [{completed}/{len(queries)}] elapsed={elapsed:.1f}s  "
                      f"last_status={res.status}")
    except FuturesTimeout:
        print(f"\n⚠  Overall {TOTAL_TIMEOUT_S}s timeout hit — "
              f"{completed}/{len(queries)} completed.")
        for future, idx in future_map.items():
            if not future.done():
                results.append(QueryResult(
                    idx=idx, pair_count=0, needs_opt=False, transformed=False,
                    verified=False, status="timeout", match=False,
                    error="overall_timeout",
                ))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    results.sort(key=lambda r: r.idx)
    elapsed_total = time.time() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────────

    by_status: dict[str, list[QueryResult]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    ok_results = by_status.get("ok", [])
    mismatches = by_status.get("result_mismatch", [])
    no_transform = by_status.get("no_transform", [])
    verify_fail = by_status.get("verify_fail", [])
    errors = by_status.get("orig_error", []) + by_status.get("opt_error", [])
    timeouts = by_status.get("timeout", [])

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries      : {len(queries)}")
    print(f"Completed          : {len(results)}")
    print(f"Wall time          : {elapsed_total:.1f}s")
    print()
    print(f"  ✓ ok (match)     : {len(ok_results)}")
    print(f"  ✗ result mismatch: {len(mismatches)}")
    print(f"  - no_transform   : {len(no_transform)}")
    print(f"  - verify_fail    : {len(verify_fail)}")
    print(f"  - exec error     : {len(errors)}")
    print(f"  - timeout        : {len(timeouts)}")

    if ok_results:
        avg_orig = sum(r.orig_time_s for r in ok_results) / len(ok_results)
        avg_opt  = sum(r.opt_time_s  for r in ok_results) / len(ok_results)
        speedup  = avg_orig / avg_opt if avg_opt > 0 else float("inf")
        print()
        print(f"Execution time (matched queries):")
        print(f"  avg original   : {avg_orig*1000:.1f} ms")
        print(f"  avg optimized  : {avg_opt*1000:.1f} ms")
        print(f"  avg speedup    : {speedup:.2f}×")

    # ── Pair-count breakdown for ok queries ──────────────────────────────────
    if ok_results:
        print()
        print("Pair-count distribution (ok queries):")
        import collections
        cnt = collections.Counter(r.pair_count for r in ok_results)
        for k in sorted(cnt):
            print(f"  {k:3d} pairs: {cnt[k]} queries")

    # ── Failures ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FAILURE DETAILS")
    print("=" * 70)

    if mismatches:
        print(f"\n-- result_mismatch ({len(mismatches)}) --")
        for r in mismatches:
            print(f"  idx={r.idx}  pairs={r.pair_count}  "
                  f"orig_t={r.orig_time_s*1000:.0f}ms  opt_t={r.opt_time_s*1000:.0f}ms")
            print(f"    orig_sql : {r.orig_sql_head[:120]!r}")
            print(f"    opt_sql  : {r.opt_sql_head[:120]!r}")

    if no_transform:
        print(f"\n-- no_transform ({len(no_transform)}) --")
        for r in no_transform:
            print(f"  idx={r.idx}  pairs={r.pair_count}")
            print(f"    sql_head : {r.orig_sql_head[:120]!r}")

    if verify_fail:
        print(f"\n-- verify_fail ({len(verify_fail)}) --")
        for r in verify_fail:
            print(f"  idx={r.idx}  pairs={r.pair_count}")
            print(f"    orig     : {r.orig_sql_head[:120]!r}")
            print(f"    rewritten: {r.opt_sql_head[:120]!r}")

    if errors:
        print(f"\n-- exec errors ({len(errors)}) --")
        for r in errors:
            print(f"  idx={r.idx}  status={r.status}  error={r.error!r}")

    if timeouts:
        print(f"\n-- timeouts ({len(timeouts)}) --")
        for r in timeouts[:10]:
            print(f"  idx={r.idx}  error={r.error!r}")
        if len(timeouts) > 10:
            print(f"  … and {len(timeouts) - 10} more")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    pass_count = len(ok_results)
    total_done = len(results)
    pass_rate = pass_count / total_done * 100 if total_done else 0.0

    print(f"Pass rate: {pass_count}/{total_done} ({pass_rate:.1f}%)")

    if not mismatches and not no_transform and not verify_fail and not errors:
        print("✓ All completed queries matched — mechanical transform is correct.")
    else:
        if mismatches:
            print(f"✗ {len(mismatches)} result mismatches — transform produces different output.")
            print("  → Inspect opt_sql_head above; likely a pair-extraction or rewrite bug.")
        if no_transform:
            print(f"⚠  {len(no_transform)} queries had ROUND...IN pattern detected but "
                  "mechanical_optimize could not parse the IN-list.")
            print("  → Check _PAIR_RE against the sql_head patterns above.")
        if verify_fail:
            print(f"⚠  {len(verify_fail)} transforms passed but pair-set verification failed.")
            print("  → Asymmetry between _PAIR_RE and _VALUES_PAIR_RE — fix one of them.")

    # ── Save detailed JSONL ───────────────────────────────────────────────────
    out_path = Path("mechanical_optimize_test_results.jsonl")
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
