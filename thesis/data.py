"""Daily panel cleaning and close-to-close labels on one market calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
DATA_COLUMNS = (*PRICE_COLUMNS, "volume")


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["datetime", "instrument"]:
        raise ValueError("panel index must be (datetime, instrument)")
    result = frame.copy(deep=False)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("datetime")))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("panel dates must be finite timezone-naive daily dates")
    if frame.index.get_level_values("instrument").isna().any():
        raise ValueError("instrument must not be missing")
    result.index = pd.MultiIndex.from_arrays([dates, frame.index.get_level_values("instrument")], names=frame.index.names)
    if not result.index.is_unique:
        raise ValueError("panel index must be unique; call clean_panel first")
    return result


def _calendar(panel: pd.DataFrame, calendar=None) -> pd.DatetimeIndex:
    observed = pd.DatetimeIndex(panel.index.get_level_values("datetime").unique()).sort_values()
    supplied = calendar if calendar is not None else panel.attrs.get("calendar")
    if supplied is None:
        return observed
    result = pd.DatetimeIndex(pd.to_datetime(supplied)).unique().sort_values()
    if result.hasnans or result.tz is not None or not result.equals(result.normalize()):
        raise ValueError("calendar must contain finite daily dates")
    if not observed.isin(result).all():
        raise ValueError("calendar does not contain every observed trading date")
    return result


def _aligned(wide: pd.DataFrame, index: pd.MultiIndex, name: str) -> pd.Series:
    row = wide.index.get_indexer(index.get_level_values("datetime"))
    col = wide.columns.get_indexer(index.get_level_values("instrument"))
    return pd.Series(wide.to_numpy(dtype=float)[row, col], index=index, name=name, dtype=float)


def _horizon(h: int) -> int:
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 or h > 10000:
        raise ValueError("h must be a positive integer <= 10000")
    return int(h)


def clean_panel(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exclude ambiguous duplicate keys and invalid bars; never fill prices.

    ``duplicates_removed`` counts extra occurrences; ``duplicate_rows_excluded``
    counts ALL bars at duplicated keys, including the first, conservatively.
    Volume is kept in the source unit. Zero-volume bars are not tradable.
    """
    attrs = frame.attrs.copy()
    source = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    required = {"datetime", "instrument", "open", "high", "low", "close", "volume"}
    if not required.issubset(source.columns):
        raise ValueError(f"missing panel columns: {sorted(required - set(source.columns))}")
    source = source.copy()
    parsed_dates = pd.to_datetime(source["datetime"], errors="coerce")
    daily_date = parsed_dates.notna() & parsed_dates.eq(parsed_dates.dt.normalize())
    source["datetime"] = parsed_dates.dt.normalize()
    source["_daily_date_valid"] = daily_date
    if "adj_close" not in source:
        source["adj_close"] = source["close"]
    for column in DATA_COLUMNS:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    keys = ["datetime", "instrument"]
    duplicates = source.duplicated(keys, keep=False)
    extras = int(source.duplicated(keys).sum())
    conflict_rows = 0
    for _, group in source.loc[duplicates].groupby(keys, dropna=False):
        if len(group[list(DATA_COLUMNS)].drop_duplicates()) > 1:
            conflict_rows += len(group)
    unique = source.loc[~duplicates].copy()
    finite = np.isfinite(unique[list(DATA_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    positive = (unique[list(DATA_COLUMNS)] > 0).all(axis=1)
    valid_ohlc = (unique["low"] <= unique[["open", "close"]].min(axis=1)) & (unique["high"] >= unique[["open", "close"]].max(axis=1)) & (unique["high"] >= unique["low"])
    valid_keys = (unique["datetime"].notna() & unique["_daily_date_valid"] & unique["instrument"].notna()
                  & unique["instrument"].astype(str).str.strip().ne(""))
    valid = finite & positive & valid_ohlc & valid_keys
    result = unique.loc[valid, [*keys, *DATA_COLUMNS]].set_index(keys).sort_index()
    result.attrs = attrs
    report = {
        "input_rows": int(len(frame)), "valid_rows": int(len(result)),
        "duplicates_removed": extras, "duplicate_rows_excluded": int(duplicates.sum()),
        "conflicting_duplicate_rows": int(conflict_rows),
        "invalid_removed": int((~valid).sum()), "zero_volume_rows": int(unique["volume"].eq(0).sum()),
        "missing_values_filled": 0, "duplicate_policy": "exclude all bars at a duplicated key",
    }
    return result, report


def forward_labels(panel: pd.DataFrame, h: int = 1, calendar=None) -> pd.Series:
    """At t: buy at t+1 close, exit t+1+h close. Missing sessions stay missing."""
    h = _horizon(h)
    panel = _panel(panel)
    dates = _calendar(panel, calendar)
    column = "adj_close" if "adj_close" in panel else "close"
    if panel.empty:
        return pd.Series(index=panel.index, dtype=float, name="forward_return")
    prices = panel[column].astype(float).unstack("instrument").reindex(dates)
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    entry, exit_ = prices.shift(-1), prices.shift(-(h + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = (exit_ / entry - 1).replace([np.inf, -np.inf], np.nan)
    return _aligned(returns, panel.index, "forward_return")


def segment_mask(panel: pd.DataFrame, start, end, h: int = 1, calendar=None) -> pd.Series:
    """Select signal dates inside the segment whose EXIT also lies inside it."""
    h = _horizon(h)
    panel = _panel(panel)
    dates = _calendar(panel, calendar)
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(lower) or pd.isna(upper) or lower > upper:
        raise ValueError("invalid segment boundaries")
    signal = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    positions = dates.get_indexer(signal) + h + 1
    available = positions < len(dates)
    inside = np.zeros(len(panel), dtype=bool)
    inside[available] = dates.take(positions[available]) <= upper
    inside &= (signal >= lower) & (signal <= upper)
    return pd.Series(inside, index=panel.index, name="segment_mask")
