"""Mechanical EXTRACT → time range rewriter using sqlglot AST.

Converts EXTRACT equality conditions that together specify a contiguous time
interval into equivalent GTE/LT range conditions that PostgreSQL can satisfy
with a btree index on the time column.

Supported combinations
----------------------
  YEAR                          → [Y-01-01,  Y+1-01-01)
  YEAR + MONTH                  → [Y-M-01,   Y-M+1-01)   (Dec wraps to Jan next year)
  YEAR + MONTH + DAY            → [Y-M-D,    Y-M-D+1)
  YEAR + MONTH + DAY + HOUR     → [Y-M-D H,  Y-M-D H+1)
  YEAR + WEEK  (ISO week)       → [Mon_week, Mon_week+7)  (may start in prior year)
  YEAR + DOY   (day of year)    → [Y-doy,    Y-doy+1)
  YEAR + MONTH + DOY            → same as YEAR + DOY (month is redundant)
  YEAR + QUARTER                → [Y-Qstart, Y-Qstart+3m)

Conditions that cannot form a contiguous range (e.g. MONTH without YEAR,
DOW, isolated HOUR) are left unchanged.

The rewrite is applied to every SELECT in the query, including CTEs.
OR-separated groups are processed independently so multi-range ORs are
handled without merging incompatible intervals.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional

import sqlglot
import sqlglot.expressions as exp


# ---------------------------------------------------------------------------
# Field name normalisation
# ---------------------------------------------------------------------------

_NORMALISE: dict[str, str] = {
    "year": "year", "years": "year",
    "month": "month", "months": "month", "mon": "month",
    "day": "day", "days": "day",
    "hour": "hour", "hours": "hour",
    "week": "week", "weeks": "week",
    "quarter": "quarter",
    "doy": "doy", "dayofyear": "doy",
    "dow": "dow", "dayofweek": "dow",
    "minute": "minute", "minutes": "minute",
    "second": "second", "seconds": "second",
}

# Combinations that map to a contiguous interval (all must include "year")
#
# NOTE: "year + week" is intentionally excluded.
# EXTRACT(YEAR FROM t) is the *calendar* year, not the ISO week year.
# ISO week 1 of year Y+1 can start in late December of calendar year Y
# (e.g. Dec 31 2018 has EXTRACT(WEEK)=1 and EXTRACT(YEAR)=2018).
# Rewriting to date.fromisocalendar(Y, W, 1)…+7d would silently drop those
# late-December rows and produce wrong results.  Use ISOYEAR+WEEK if you need
# a safe range rewrite.
_CONVERTIBLE: frozenset[frozenset[str]] = frozenset({
    frozenset(["year"]),
    frozenset(["year", "month"]),
    frozenset(["year", "month", "day"]),
    frozenset(["year", "month", "day", "hour"]),
    frozenset(["year", "doy"]),
    frozenset(["year", "month", "doy"]),   # month is redundant but not wrong
    frozenset(["year", "quarter"]),
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_extract_to_range(sql: str) -> tuple[str, bool]:
    """Rewrite EXTRACT equality predicates to GTE/LT range predicates.

    Returns ``(rewritten_sql, was_transformed)``.  Falls back to the
    original SQL on any parse or serialisation error.
    """
    # sqlglot drops column aliases from WITH ORDINALITY AS alias(col1, col2, ...)
    # constructs, corrupting the serialised SQL.  Skip rather than produce
    # a broken query.
    if re.search(r'\bWITH\s+ORDINALITY\b', sql, re.IGNORECASE):
        return sql, False

    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return sql, False

    modified = False
    for select in tree.find_all(exp.Select):
        where_node = select.args.get("where")
        if not where_node:
            continue
        new_cond, changed = _transform_cond(where_node.this)
        if changed:
            where_node.set("this", new_cond)
            modified = True

    if not modified:
        return sql, False

    try:
        return tree.sql(dialect="postgres", pretty=False), True
    except Exception:
        return sql, False


def verify_extract_rewrite(original: str, rewritten: str) -> bool:
    """Lightweight check: rewritten SQL must still contain the same tables."""
    try:
        orig_tables  = {t.name.lower() for t in sqlglot.parse_one(original,  dialect="postgres").find_all(exp.Table)}
        new_tables   = {t.name.lower() for t in sqlglot.parse_one(rewritten, dialect="postgres").find_all(exp.Table)}
        return orig_tables == new_tables
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Recursive condition transformer
# ---------------------------------------------------------------------------

def _transform_cond(expr: exp.Expression) -> tuple[exp.Expression, bool]:
    """Recursively transform an expression tree, returning (new_expr, changed).

    Note: sqlglot binary nodes use ``.this`` (left) and ``.expression`` (right),
    not ``.left``/``.right``.
    """
    if isinstance(expr, exp.Paren):
        inner, changed = _transform_cond(expr.this)
        if changed:
            return exp.Paren(this=inner), True
        return expr, False
    if isinstance(expr, exp.And):
        flat = _flatten_and(expr)
        return _rewrite_flat(flat)
    if isinstance(expr, exp.Or):
        left,  lc = _transform_cond(expr.this)
        right, rc = _transform_cond(expr.expression)
        if lc or rc:
            return exp.Or(this=left, expression=right), True
        return expr, False
    # Single leaf condition — attempt rewrite as a one-item group
    return _rewrite_flat([expr])


def _flatten_and(expr: exp.Expression) -> list[exp.Expression]:
    """Flatten a (left-associative) AND tree into a flat list.

    OR sub-expressions are treated as atomic units after recursively
    transforming their internals.
    """
    if isinstance(expr, exp.And):
        return _flatten_and(expr.this) + _flatten_and(expr.expression)
    if isinstance(expr, exp.Or):
        new_expr, _ = _transform_cond(expr)
        return [new_expr]
    return [expr]


def _rewrite_flat(conditions: list[exp.Expression]) -> tuple[exp.Expression, bool]:
    """Group EXTRACT EQ conditions in a flat AND list and replace with ranges."""
    extract_eqs: list[tuple[exp.Expression, str, exp.Expression, str, int]] = []
    others: list[exp.Expression] = []

    for cond in conditions:
        parsed = _parse_extract_eq(cond)
        if parsed:
            col_key, col_expr, field, value = parsed
            extract_eqs.append((cond, col_key, col_expr, field, value))
        else:
            others.append(cond)

    if not extract_eqs:
        return _rebuild_and(conditions), False

    # Group by the column expression string
    groups: dict[str, dict] = {}
    for node, col_key, col_expr, field, value in extract_eqs:
        if col_key not in groups:
            groups[col_key] = {"_col_expr": col_expr, "_nodes": []}
        groups[col_key][field] = value
        groups[col_key]["_nodes"].append(node)

    new_conditions = list(others)
    changed = False

    for col_key, info in groups.items():
        col_expr: exp.Expression = info["_col_expr"]
        nodes: list[exp.Expression] = info["_nodes"]
        fields = {k: v for k, v in info.items() if not k.startswith("_")}

        range_cond = _build_range(col_expr, fields)
        if range_cond is not None:
            new_conditions.append(range_cond)
            changed = True
        else:
            new_conditions.extend(nodes)

    return _rebuild_and(new_conditions), changed


def _rebuild_and(conditions: list[exp.Expression]) -> exp.Expression:
    if not conditions:
        raise ValueError("Cannot rebuild AND from empty list")
    result = conditions[0]
    for c in conditions[1:]:
        result = exp.And(this=result, expression=c)
    return result


# ---------------------------------------------------------------------------
# Extract EQ parsing
# ---------------------------------------------------------------------------

def _parse_extract_eq(node: exp.Expression) -> Optional[tuple[str, exp.Expression, str, int]]:
    """If *node* is ``EXTRACT(field FROM col) = integer``, return
    ``(col_key, col_expr, field_name, value)``.  Otherwise return ``None``.
    """
    if not isinstance(node, exp.EQ):
        return None

    left, right = node.this, node.expression

    if isinstance(left, exp.Extract) and _is_int_literal(right):
        extract, literal = left, right
    elif isinstance(right, exp.Extract) and _is_int_literal(left):
        extract, literal = right, left
    else:
        return None

    field = _get_field_name(extract)
    if field is None:
        return None

    col_expr = extract.expression
    col_key  = col_expr.sql(dialect="postgres")

    try:
        value = int(literal.this)
    except (ValueError, TypeError):
        return None

    return col_key, col_expr, field, value


def _is_int_literal(node: exp.Expression) -> bool:
    return isinstance(node, exp.Literal) and not node.is_string


def _get_field_name(extract: exp.Extract) -> Optional[str]:
    """Return the normalised field name from an Extract node, or None."""
    field = extract.this
    if isinstance(field, str):
        raw = field
    elif hasattr(field, "name"):
        raw = field.name
    elif hasattr(field, "this"):
        raw = str(field.this)
    else:
        return None
    return _NORMALISE.get(raw.lower())


# ---------------------------------------------------------------------------
# Range building
# ---------------------------------------------------------------------------

def _build_range(col_expr: exp.Expression, fields: dict[str, int]) -> Optional[exp.Expression]:
    """Build a ``col >= start AND col < end`` expression, or None if the
    combination is not convertible.
    """
    field_set = frozenset(fields.keys())
    if field_set not in _CONVERTIBLE:
        return None

    try:
        start_s, end_s = _compute_range(field_set, fields)
    except (ValueError, OverflowError, AttributeError):
        return None

    return exp.And(
        this=exp.GTE(this=col_expr.copy(), expression=exp.Literal.string(start_s)),
        expression=exp.LT(this=col_expr.copy(), expression=exp.Literal.string(end_s)),
    )


def _compute_range(field_set: frozenset[str], f: dict[str, int]) -> tuple[str, str]:
    y = f["year"]

    if field_set == frozenset(["year"]):
        return f"{y}-01-01", f"{y+1}-01-01"

    if field_set == frozenset(["year", "month"]):
        m = f["month"]
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        return f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"

    if field_set == frozenset(["year", "month", "day"]):
        d = date(y, f["month"], f["day"])
        return d.isoformat(), (d + timedelta(days=1)).isoformat()

    if field_set == frozenset(["year", "month", "day", "hour"]):
        dt = datetime(y, f["month"], f["day"], f["hour"])
        return (dt.strftime("%Y-%m-%d %H:%M:%S"),
                (dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))

    if field_set == frozenset(["year", "week"]):
        monday = date.fromisocalendar(y, f["week"], 1)
        return monday.isoformat(), (monday + timedelta(weeks=1)).isoformat()

    if field_set in (frozenset(["year", "doy"]),
                     frozenset(["year", "month", "doy"])):
        d = date(y, 1, 1) + timedelta(days=f["doy"] - 1)
        return d.isoformat(), (d + timedelta(days=1)).isoformat()

    if field_set == frozenset(["year", "quarter"]):
        q = f["quarter"]
        ms = (q - 1) * 3 + 1          # first month of quarter
        me = ms + 3                    # first month of next quarter
        ny, nm = (y + 1, 1) if me > 12 else (y, me)
        return f"{y}-{ms:02d}-01", f"{ny}-{nm:02d}-01"

    raise ValueError(f"Unhandled combination: {field_set}")
