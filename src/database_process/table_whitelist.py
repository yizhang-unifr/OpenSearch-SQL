"""Table whitelist utilities for OpenSearch-SQL PostgreSQL pipeline."""

from __future__ import annotations


_EXTRA_ALLOWED_TABLES: set[str] = {
    "landcover_type",
    "landcover_upscaled",
}


def is_table_allowed(table_name: str) -> bool:
    """Allow `meteo_*` tables plus explicit extra tables."""
    if table_name.startswith("meteo_"):
        return True
    return table_name in _EXTRA_ALLOWED_TABLES


def filter_allowed_tables(table_names: list[str]) -> list[str]:
    """Return table names that pass whitelist policy, preserving order."""
    return [t for t in table_names if is_table_allowed(t)]

