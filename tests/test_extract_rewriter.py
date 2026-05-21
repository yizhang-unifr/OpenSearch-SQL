"""Tests for extract_rewriter — EXTRACT equality → time range transform.

Run:
    uv run python -m pytest src/OpenSearch-SQL/tests/test_extract_rewriter.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.optimizer.extract_rewriter import rewrite_extract_to_range, verify_extract_rewrite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rewrite(sql: str) -> str:
    result, transformed = rewrite_extract_to_range(sql)
    assert transformed, f"Expected transformation but got none.\nSQL: {sql}"
    return result


def _no_rewrite(sql: str) -> None:
    result, transformed = rewrite_extract_to_range(sql)
    assert not transformed, f"Expected no transformation but got one.\nResult: {result}"


def _contains(sql: str, fragment: str) -> bool:
    return fragment.lower() in sql.lower()


# ---------------------------------------------------------------------------
# YEAR only
# ---------------------------------------------------------------------------

class TestYearOnly:
    def test_basic(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2020"
        out = _rewrite(sql)
        assert _contains(out, "2020-01-01")
        assert _contains(out, "2021-01-01")
        assert ">=" in out and "<" in out
        assert "EXTRACT" not in out.upper()

    def test_with_other_condition(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2018 AND t.val > 0"
        out = _rewrite(sql)
        assert _contains(out, "2018-01-01")
        assert _contains(out, "2019-01-01")
        assert _contains(out, "t.val > 0") or _contains(out, "val > 0")


# ---------------------------------------------------------------------------
# YEAR + MONTH
# ---------------------------------------------------------------------------

class TestYearMonth:
    def test_mid_year(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2015 AND EXTRACT(MONTH FROM t.time) = 3"
        out = _rewrite(sql)
        assert _contains(out, "2015-03-01")
        assert _contains(out, "2015-04-01")

    def test_december_wraps_year(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2019 AND EXTRACT(MONTH FROM t.time) = 12"
        out = _rewrite(sql)
        assert _contains(out, "2019-12-01")
        assert _contains(out, "2020-01-01")

    def test_order_irrelevant(self):
        # MONTH before YEAR in SQL
        sql = "SELECT * FROM t WHERE EXTRACT(MONTH FROM t.time) = 6 AND EXTRACT(YEAR FROM t.time) = 2021"
        out = _rewrite(sql)
        assert _contains(out, "2021-06-01")
        assert _contains(out, "2021-07-01")

    def test_real_zurich_q12(self):
        # Minimum wind speed in City of Zurich in March 2015
        sql = (
            "SELECT MIN(mws.windspeedmax) FROM meteo_windspeedmax AS mws "
            "WHERE EXTRACT(MONTH FROM mws.time) = 3 "
            "AND EXTRACT(YEAR FROM mws.time) = 2015 "
            "AND (ROUND(CAST(mws.latitude AS DECIMAL), 1), "
            "ROUND(CAST(mws.longitude AS DECIMAL), 1)) IN ((47.4, 8.5))"
        )
        out = _rewrite(sql)
        assert _contains(out, "2015-03-01")
        assert _contains(out, "2015-04-01")
        # IN list must still be present
        assert "47.4" in out


# ---------------------------------------------------------------------------
# YEAR + MONTH + DAY
# ---------------------------------------------------------------------------

class TestYearMonthDay:
    def test_basic(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2015 "
            "AND EXTRACT(MONTH FROM t.time) = 1 "
            "AND EXTRACT(DAY FROM t.time) = 15"
        )
        out = _rewrite(sql)
        assert _contains(out, "2015-01-15")
        assert _contains(out, "2015-01-16")

    def test_end_of_month(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2020 "
            "AND EXTRACT(MONTH FROM t.time) = 1 "
            "AND EXTRACT(DAY FROM t.time) = 31"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-01-31")
        assert _contains(out, "2020-02-01")

    def test_feb_28_non_leap(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2019 "
            "AND EXTRACT(MONTH FROM t.time) = 2 "
            "AND EXTRACT(DAY FROM t.time) = 28"
        )
        out = _rewrite(sql)
        assert _contains(out, "2019-02-28")
        assert _contains(out, "2019-03-01")

    def test_real_zurich_q38(self):
        # Lowest temp at lowest point in City of Zurich on Jan 15 2015
        sql = """
WITH min_elev_value AS (
  SELECT MIN(me.elevation) AS min_elev
  FROM meteo_elevation AS me
  WHERE (ROUND(CAST(me.latitude AS DECIMAL), 1),
         ROUND(CAST(me.longitude AS DECIMAL), 1)) IN ((47.4, 8.5))
), lowest_point AS (
  SELECT me1.latitude, me1.longitude
  FROM meteo_elevation AS me1
  WHERE me1.elevation = (SELECT min_elev FROM min_elev_value)
    AND (ROUND(CAST(me1.latitude AS DECIMAL), 1),
         ROUND(CAST(me1.longitude AS DECIMAL), 1)) IN ((47.4, 8.5))
  LIMIT 1
)
SELECT MIN(mt.tmin) - 273.15, lp.latitude, lp.longitude
FROM meteo_tmin AS mt
JOIN lowest_point AS lp
  ON (ROUND(CAST(mt.latitude AS DECIMAL), 1), ROUND(CAST(mt.longitude AS DECIMAL), 1))
   = (ROUND(CAST(lp.latitude AS DECIMAL), 1), ROUND(CAST(lp.longitude AS DECIMAL), 1))
WHERE EXTRACT(YEAR FROM mt.time) = 2015
  AND EXTRACT(MONTH FROM mt.time) = 1
  AND EXTRACT(DAY FROM mt.time) = 15
GROUP BY lp.latitude, lp.longitude
LIMIT 1
"""
        out = _rewrite(sql)
        assert _contains(out, "2015-01-15")
        assert _contains(out, "2015-01-16")
        # CTE structure preserved
        assert "min_elev_value" in out
        assert "lowest_point" in out
        assert verify_extract_rewrite(sql, out)


# ---------------------------------------------------------------------------
# YEAR + MONTH + DAY + HOUR
# ---------------------------------------------------------------------------

class TestYearMonthDayHour:
    def test_basic(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2020 "
            "AND EXTRACT(MONTH FROM t.time) = 6 "
            "AND EXTRACT(DAY FROM t.time) = 15 "
            "AND EXTRACT(HOUR FROM t.time) = 14"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-06-15 14:00:00")
        assert _contains(out, "2020-06-15 15:00:00")

    def test_hour_23_wraps_day(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2020 "
            "AND EXTRACT(MONTH FROM t.time) = 3 "
            "AND EXTRACT(DAY FROM t.time) = 31 "
            "AND EXTRACT(HOUR FROM t.time) = 23"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-03-31 23:00:00")
        assert _contains(out, "2020-04-01 00:00:00")


# ---------------------------------------------------------------------------
# YEAR + WEEK — must NOT be rewritten
# ---------------------------------------------------------------------------
# EXTRACT(YEAR FROM t) is the *calendar* year, not the ISO week year.
# ISO week 1 of Y+1 can start in late December of calendar year Y
# (e.g. 2018-12-31 has WEEK=1 and calendar YEAR=2018).
# Rewriting to a 7-day window would silently drop those rows.

class TestYearWeek:
    def test_year_week_not_rewritten(self):
        sql = (
            "SELECT AVG(mt.tmean) FROM meteo_tmean AS mt "
            "WHERE EXTRACT(WEEK FROM mt.time) = 15 "
            "AND EXTRACT(YEAR FROM mt.time) = 2017"
        )
        _no_rewrite(sql)

    def test_week_1_not_rewritten(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2020 AND EXTRACT(WEEK FROM t.time) = 1"
        _no_rewrite(sql)


# ---------------------------------------------------------------------------
# YEAR + DOY
# ---------------------------------------------------------------------------

class TestYearDoy:
    def test_basic(self):
        # Day 100 of 2015 = April 10
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2015 AND EXTRACT(DOY FROM t.time) = 100"
        out = _rewrite(sql)
        assert _contains(out, "2015-04-10")
        assert _contains(out, "2015-04-11")

    def test_year_month_doy(self):
        # Month is redundant but should still convert using year+doy
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.time) = 2020 "
            "AND EXTRACT(MONTH FROM t.time) = 3 "
            "AND EXTRACT(DOY FROM t.time) = 70"
        )
        out = _rewrite(sql)
        # DOY 70 of 2020 (leap year) = March 10
        assert _contains(out, "2020-03-10")
        assert _contains(out, "2020-03-11")


# ---------------------------------------------------------------------------
# YEAR + QUARTER
# ---------------------------------------------------------------------------

class TestYearQuarter:
    def test_q1(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2021 AND EXTRACT(QUARTER FROM t.time) = 1"
        out = _rewrite(sql)
        assert _contains(out, "2021-01-01")
        assert _contains(out, "2021-04-01")

    def test_q4_wraps_year(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2021 AND EXTRACT(QUARTER FROM t.time) = 4"
        out = _rewrite(sql)
        assert _contains(out, "2021-10-01")
        assert _contains(out, "2022-01-01")


# ---------------------------------------------------------------------------
# Cases that must NOT be rewritten
# ---------------------------------------------------------------------------

class TestNoRewrite:
    def test_month_only_no_year(self):
        sql = "SELECT * FROM t WHERE EXTRACT(MONTH FROM t.time) = 3"
        _no_rewrite(sql)

    def test_extract_between_not_eq(self):
        # BETWEEN is not EQ — must not be touched
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) BETWEEN 2015 AND 2020"
        _no_rewrite(sql)

    def test_extract_in_not_eq(self):
        sql = "SELECT * FROM t WHERE EXTRACT(MONTH FROM t.time) IN (12, 1, 2)"
        _no_rewrite(sql)

    def test_extract_in_select_only(self):
        # EXTRACT only in SELECT list, not WHERE
        sql = "SELECT EXTRACT(YEAR FROM t.time) FROM t WHERE t.val > 0"
        _no_rewrite(sql)

    def test_extract_in_case_when(self):
        # EXTRACT inside CASE WHEN — WHERE has IN not EQ
        sql = """
SELECT
  CASE WHEN EXTRACT(MONTH FROM t.time) = 12
       THEN EXTRACT(YEAR FROM t.time) + 1
       ELSE EXTRACT(YEAR FROM t.time) END AS yr
FROM t
WHERE EXTRACT(MONTH FROM t.time) IN (12, 1, 2)
"""
        _no_rewrite(sql)

    def test_no_extract_at_all(self):
        sql = "SELECT * FROM t WHERE t.val > 0"
        _no_rewrite(sql)

    def test_extract_dow_not_convertible(self):
        sql = "SELECT * FROM t WHERE EXTRACT(YEAR FROM t.time) = 2020 AND EXTRACT(DOW FROM t.time) = 1"
        # year + dow is not a convertible combo
        _no_rewrite(sql)


# ---------------------------------------------------------------------------
# OR conditions: each branch independently
# ---------------------------------------------------------------------------

class TestOrBranches:
    def test_or_with_extract_in_one_branch(self):
        sql = (
            "SELECT * FROM t "
            "WHERE (EXTRACT(YEAR FROM t.time) = 2020 AND EXTRACT(MONTH FROM t.time) = 1) "
            "OR t.val > 100"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-01-01")
        assert _contains(out, "2020-02-01")

    def test_two_or_branches_both_converted(self):
        sql = (
            "SELECT * FROM t "
            "WHERE (EXTRACT(YEAR FROM t.time) = 2020 AND EXTRACT(MONTH FROM t.time) = 1) "
            "OR (EXTRACT(YEAR FROM t.time) = 2021 AND EXTRACT(MONTH FROM t.time) = 6)"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-01-01") and _contains(out, "2020-02-01")
        assert _contains(out, "2021-06-01") and _contains(out, "2021-07-01")


# ---------------------------------------------------------------------------
# Multiple time columns in one query
# ---------------------------------------------------------------------------

class TestMultipleColumns:
    def test_two_columns_both_rewritten(self):
        sql = (
            "SELECT * FROM t "
            "WHERE EXTRACT(YEAR FROM t.start_time) = 2020 "
            "AND EXTRACT(YEAR FROM t.end_time) = 2021"
        )
        out = _rewrite(sql)
        assert _contains(out, "2020-01-01") and _contains(out, "2021-01-01")
        assert _contains(out, "2021-01-01") and _contains(out, "2022-01-01")


# ---------------------------------------------------------------------------
# Verify helper
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_passes_on_good_rewrite(self):
        sql = "SELECT * FROM meteo_tmin WHERE EXTRACT(YEAR FROM time) = 2020"
        out, _ = rewrite_extract_to_range(sql)
        assert verify_extract_rewrite(sql, out)

    def test_verify_fails_on_different_tables(self):
        original  = "SELECT * FROM meteo_tmin WHERE 1=1"
        rewritten = "SELECT * FROM other_table WHERE 1=1"
        assert not verify_extract_rewrite(original, rewritten)
