"""Tests for elevation_rewriter — meteo_elevation range-join → eligible_cells equi-join.

Run:
    uv run python -m pytest src/OpenSearch-SQL/tests/test_elevation_rewriter.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.optimizer.elevation_rewriter import rewrite_elevation_join, verify_elevation_rewrite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASIC_ELEV_SQL = """
WITH place_0 AS (
  SELECT me.latitude AS latitude, me.longitude AS longitude
  FROM meteo_elevation AS me
  WHERE me.latitude >= 47.3 AND me.latitude <= 47.4
    AND me.longitude >= 8.4 AND me.longitude <= 8.6
    AND me.elevation < 1500
), wind_data AS (
  SELECT wsm.time, wsm.windspeedmax, wsm.latitude, wsm.longitude
  FROM meteo_windspeedmax AS wsm
  INNER JOIN place_0 AS z
    ON wsm.latitude >= z.latitude AND wsm.latitude <= z.latitude + 0.1
    AND wsm.longitude >= z.longitude AND wsm.longitude <= z.longitude + 0.1
)
SELECT latitude, longitude, AVG(windspeedmax) FROM wind_data GROUP BY 1, 2
""".strip()


def _rewrite(sql: str) -> str:
    result, transformed = rewrite_elevation_join(sql)
    assert transformed, f"Expected elevation rewrite but got none.\nSQL: {sql}"
    return result


def _no_rewrite(sql: str) -> None:
    result, transformed = rewrite_elevation_join(sql)
    assert not transformed, f"Expected no transformation but got one.\nSQL: {sql}"


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetection:
    def test_basic_pattern_detected(self):
        result, transformed = rewrite_elevation_join(_BASIC_ELEV_SQL)
        assert transformed

    def test_no_elevation_table(self):
        sql = """
            SELECT wsm.latitude, wsm.longitude
            FROM meteo_windspeedmax AS wsm
            WHERE wsm.latitude >= 47.3 AND wsm.latitude <= 47.4
        """
        _no_rewrite(sql)

    def test_elevation_without_threshold(self):
        sql = """
            WITH place_0 AS (
              SELECT me.latitude, me.longitude
              FROM meteo_elevation AS me
              WHERE me.latitude >= 47.3 AND me.latitude <= 47.4
            ), wind AS (
              SELECT wsm.windspeedmax FROM meteo_windspeedmax AS wsm
              INNER JOIN place_0 AS z
                ON wsm.latitude >= z.latitude AND wsm.latitude <= z.latitude + 0.1
            )
            SELECT * FROM wind
        """
        _no_rewrite(sql)

    def test_no_non_elevation_meteo(self):
        sql = """
            WITH place_0 AS (
              SELECT me.latitude FROM meteo_elevation AS me
              WHERE me.elevation < 1000
            )
            SELECT * FROM place_0
        """
        _no_rewrite(sql)

    def test_elevation_without_join(self):
        sql = """
            WITH place_0 AS (
              SELECT me.latitude FROM meteo_elevation AS me
              WHERE me.latitude >= 47.0 AND me.elevation < 1500
            )
            SELECT wsm.windspeedmax FROM meteo_windspeedmax AS wsm
            WHERE wsm.latitude IN (SELECT latitude FROM place_0)
        """
        _no_rewrite(sql)


# ---------------------------------------------------------------------------
# Structural correctness tests
# ---------------------------------------------------------------------------

class TestRewriteStructure:
    def test_elevation_cte_removed(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        assert "place_0" not in rewritten

    def test_eligible_cells_cte_present(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        assert "eligible_cells" in rewritten.lower()
        assert "meteo_elevation" in rewritten.lower()
        # Two-branch UNION for boundary coverage
        assert "CEIL" in rewritten.upper()
        assert "FLOOR" in rewritten.upper()
        assert "UNION" in rewritten.upper()

    def test_inner_join_to_eligible_cells(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        # Equi-join replaces the proximity range join
        assert "inner join eligible_cells" in rewritten.lower().replace("\n", " ")
        assert "cell_lat" in rewritten.lower()
        assert "cell_lon" in rewritten.lower()
        # No correlated subquery remnants
        assert "EXISTS" not in rewritten.upper()

    def test_bbox_prefilter_added(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        # bbox pre-filter should restrict wind rows
        assert "47.3" in rewritten  # lat_min
        assert "8.4" in rewritten   # lon_min

    def test_original_elevation_condition_preserved(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        # The elevation threshold must appear inside eligible_cells CTE
        assert "me.elevation" in rewritten or "elevation < 1500" in rewritten

    def test_round_equijoin_present(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        # ROUND(wsm.latitude::NUMERIC, 1) = eligible_cells.cell_lat
        assert "ROUND" in rewritten.upper()
        assert "eligible_cells.cell_lat" in rewritten.lower()

    def test_non_elevation_tables_preserved(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        assert "meteo_windspeedmax" in rewritten.lower()

    def test_verify_passes(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        assert verify_elevation_rewrite(_BASIC_ELEV_SQL, rewritten)


# ---------------------------------------------------------------------------
# Verify function tests
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_passes_on_good_rewrite(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        assert verify_elevation_rewrite(_BASIC_ELEV_SQL, rewritten)

    def test_verify_fails_if_non_elevation_table_dropped(self):
        rewritten = _rewrite(_BASIC_ELEV_SQL)
        # Simulate a bad rewrite that drops meteo_windspeedmax
        bad = rewritten.replace("meteo_windspeedmax", "some_other_table")
        assert not verify_elevation_rewrite(_BASIC_ELEV_SQL, bad)


# ---------------------------------------------------------------------------
# Different delta values
# ---------------------------------------------------------------------------

class TestDeltaValues:
    def _make_sql(self, delta: float) -> str:
        return f"""
            WITH place_0 AS (
              SELECT me.latitude, me.longitude FROM meteo_elevation AS me
              WHERE me.latitude >= 47.0 AND me.latitude <= 48.0
                AND me.longitude >= 8.0 AND me.longitude <= 9.0
                AND me.elevation < 2000
            ), w AS (
              SELECT wsm.windspeedmax, wsm.latitude, wsm.longitude
              FROM meteo_windspeedmax AS wsm
              INNER JOIN place_0 AS z
                ON wsm.latitude >= z.latitude AND wsm.latitude <= z.latitude + {delta}
                AND wsm.longitude >= z.longitude AND wsm.longitude <= z.longitude + {delta}
            )
            SELECT * FROM w
        """

    def test_delta_0_1(self):
        sql = self._make_sql(0.1)
        result, ok = rewrite_elevation_join(sql)
        assert ok
        assert "0.1" in result

    def test_delta_0_25(self):
        sql = self._make_sql(0.25)
        result, ok = rewrite_elevation_join(sql)
        assert ok
        assert "0.25" in result


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------

class TestCTEOrderIndependence:
    def test_elevation_cte_last(self):
        sql = """
            WITH wind_data AS (
              SELECT wsm.windspeedmax, wsm.latitude, wsm.longitude, wsm.time
              FROM meteo_windspeedmax AS wsm
              INNER JOIN place_0 AS z
                ON wsm.latitude >= z.latitude AND wsm.latitude <= z.latitude + 0.1
                AND wsm.longitude >= z.longitude AND wsm.longitude <= z.longitude + 0.1
            ), place_0 AS (
              SELECT me.latitude, me.longitude FROM meteo_elevation AS me
              WHERE me.latitude >= 47.3 AND me.latitude <= 47.4
                AND me.longitude >= 8.4 AND me.longitude <= 8.6
                AND me.elevation < 1500
            )
            SELECT * FROM wind_data
        """
        result, transformed = rewrite_elevation_join(sql)
        assert transformed
        assert "eligible_cells" in result.lower()
