"""A small AST interpreter. No eval/exec, Python attributes, indexing or imports."""

from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd

from .data import _aligned, _calendar, _panel

FIELDS = frozenset({"open", "high", "low", "close", "volume", "adj_close"})
WINDOW_FUNCTIONS = frozenset({"lag", "delta", "mean", "std", "ts_min", "ts_max", "ts_sum"})
UNARY_FUNCTIONS = frozenset({"cs_rank", "cs_zscore", "abs", "log", "sign"})
FUNCTIONS = WINDOW_FUNCTIONS | UNARY_FUNCTIONS | {"safe_div", "corr"}
MAX_WINDOW = 252


def validate_expression(expression: str) -> ast.Expression:
    """Validate all syntax and literal windows before any numeric operation."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 4096:
        raise ValueError("expression must be nonempty and <= 4096 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("expression is too complex")

    def window(node):
        if not isinstance(node, ast.Constant) or type(node.value) is not int or not 1 <= node.value <= MAX_WINDOW:
            raise ValueError(f"window must be a literal integer in [1, {MAX_WINDOW}]")

    def visit(node, depth=0):
        if depth > 32:
            raise ValueError("expression nesting is too deep")
        if isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise ValueError(f"unknown field: {node.id}")
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError("only finite bounded numeric constants are allowed")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            visit(node.operand, depth + 1)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            name = node.func.id
            expected = 3 if name == "corr" else 2 if name in WINDOW_FUNCTIONS or name == "safe_div" else 1
            if node.keywords or len(node.args) != expected:
                raise ValueError(f"{name} expects {expected} positional arguments")
            if name in WINDOW_FUNCTIONS or name == "corr":
                window(node.args[-1])
                for arg in node.args[:-1]:
                    visit(arg, depth + 1)
            else:
                for arg in node.args:
                    visit(arg, depth + 1)
        else:
            raise ValueError(f"syntax not permitted: {type(node).__name__}")

    visit(tree.body)
    return tree


def calculate(expression: str, panel: pd.DataFrame, calendar=None) -> pd.Series:
    tree = validate_expression(expression)
    panel = _panel(panel)
    dates = _calendar(panel, calendar)
    instruments = panel.index.get_level_values("instrument").unique().sort_values()
    if panel.empty:
        return pd.Series(index=panel.index, name="factor", dtype=float)
    template = pd.DataFrame(np.nan, index=dates, columns=instruments)
    cache = {}

    def wide(value):
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame(float(value), index=dates, columns=instruments)

    def finite(value):
        if isinstance(value, pd.DataFrame):
            return value.where(np.isfinite(value))
        return value if np.isfinite(value) else np.nan

    def run(node):
        if isinstance(node, ast.Name):
            if node.id not in panel:
                raise ValueError(f"panel has no field {node.id}")
            if node.id not in cache:
                cache[node.id] = panel[node.id].astype(float).unstack("instrument").reindex(index=dates, columns=instruments)
            return cache[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = run(node.left), run(node.right)
            with np.errstate(all="ignore"):
                if isinstance(node.op, ast.Add):
                    return finite(a + b)
                if isinstance(node.op, ast.Sub):
                    return finite(a - b)
                if isinstance(node.op, ast.Mult):
                    return finite(a * b)
                denominator = b.where(b.abs() > 1e-12) if isinstance(b, pd.DataFrame) else b if abs(b) > 1e-12 else np.nan
                return finite(a / denominator)
        name = node.func.id
        values = [run(arg) for arg in node.args]
        a = wide(values[0])
        if name in WINDOW_FUNCTIONS:
            window = int(values[1])
            if name == "lag":
                return a.shift(window)
            if name == "delta":
                return a - a.shift(window)
            rolling = a.rolling(window, min_periods=window)
            if name == "std":
                return rolling.std(ddof=1)
            operation = {"mean": "mean", "ts_min": "min", "ts_max": "max", "ts_sum": "sum"}[name]
            return getattr(rolling, operation)()
        if name == "safe_div":
            b = wide(values[1])
            return finite(a / b.where(b.abs() > 1e-12))
        if name == "corr":
            return finite(a.rolling(int(values[2]), min_periods=int(values[2])).corr(wide(values[1])))
        if name == "cs_rank":
            return a.rank(axis=1, method="average", pct=True)
        if name == "cs_zscore":
            scale = a.std(axis=1, ddof=1)
            return a.sub(a.mean(axis=1), axis=0).div(scale.where(scale.abs() > 1e-12), axis=0)
        if name == "abs":
            return a.abs()
        if name == "sign":
            return np.sign(a)
        if name == "log":
            return np.log(a.where(a > 0))
        raise ValueError("unsupported operator")

    with np.errstate(all="ignore"):
        result = finite(wide(run(tree.body))).reindex_like(template)
    return _aligned(result, panel.index, "factor")
