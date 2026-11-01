"""Daily cross-sectional IC and descriptive quantile returns, not a backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _calendar, _panel, forward_labels, segment_mask

PROTOCOL = {
    "id": "daily-close-entry1-exit2-purged-v1", "label": "t收盘因子 · t+1收盘进入 · t+2收盘退出",
    "signal": "t close", "entry": "t+1 close", "exit": "t+2 close", "horizon": 1,
    "return": "adj_close[t+2] / adj_close[t+1] - 1", "calendar": "shared exchange sessions; no filling",
    "purge": "signal >= segment start AND exit <= segment end",
    "ic": "daily cross-sectional Pearson", "rank_ic": "daily cross-sectional Spearman (average ties)",
    "ir": "mean daily IC / sample std (ddof=1), not annualized", "min_assets": 5,
    "quantiles": "five daily equal-count quantiles; ties stay together; equal weight daily mean returns",
    "coverage": "finite factor and label pairs / finite labels after segment exit purge",
    "costs_included": False, "tradability_checked": False,
}


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _summary(values):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return None, None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return _number(mean), _number(mean / std) if np.isfinite(std) and std > 1e-12 else None


def evaluate(factor: pd.Series, panel: pd.DataFrame, segment: str, *, start=None, end=None,
             h: int = 1, min_assets: int = 5, groups: int = 5, calendar=None) -> dict:
    panel = _panel(panel)
    if not isinstance(factor, pd.Series) or not factor.index.is_unique:
        raise ValueError("factor must be a Series with a unique panel index")
    if type(min_assets) is not int or min_assets < 2 or type(groups) is not int or groups < 2:
        raise ValueError("min_assets and groups must be integers >= 2")
    dates = _calendar(panel, calendar)
    if not len(dates):
        raise ValueError("cannot evaluate an empty panel")
    start = dates[0] if start is None else pd.Timestamp(start)
    end = dates[-1] if end is None else pd.Timestamp(end)
    # Restrict before creating labels: even an accidental caller cannot compute
    # labels from prices beyond the requested segment boundary.
    selected = panel.loc[(panel.index.get_level_values("datetime") <= end)].copy()
    selected_calendar = dates[dates <= end]
    selected.attrs["calendar"] = selected_calendar.strftime("%Y-%m-%d").tolist()
    mask = segment_mask(selected, start, end, h, calendar=selected_calendar)
    labels = forward_labels(selected, h, calendar=selected_calendar)
    f = pd.to_numeric(factor.reindex(selected.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    eligible = labels.notna() & mask
    paired = eligible & f.notna()
    denominator = int(eligible.sum())
    coverage = float(paired.sum() / denominator) if denominator else None
    table = pd.DataFrame({"factor": f, "label": labels}).loc[mask]
    daily, daily_quantiles = [], []
    pooled = {i: [] for i in range(1, groups + 1)}
    for date, day in table.groupby(level="datetime", sort=True):
        pairs = day.dropna()
        ic, rank_ic = None, None
        variable = len(pairs) >= min_assets and pairs["factor"].nunique() >= 2
        if variable and pairs["label"].nunique() >= 2:
            ic = _number(pairs["factor"].corr(pairs["label"]))
            rank_ic = _number(pairs["factor"].rank().corr(pairs["label"].rank()))
        day_string = pd.Timestamp(date).strftime("%Y-%m-%d")
        daily.append({"date": day_string, "ic": ic, "rank_ic": rank_ic,
                      "assets": int(len(pairs)), "eligible_assets": int(day["label"].notna().sum())})
        if variable and len(pairs) >= groups:
            # qcut on raw values leaves ties together instead of using instrument
            # order to manufacture returns for an otherwise constant factor.
            try:
                buckets = pd.qcut(pairs["factor"], groups, labels=False, duplicates="drop")
            except ValueError:
                continue
            if buckets.nunique() != groups:
                continue
            day_groups = []
            for group in range(1, groups + 1):
                members = pairs.loc[buckets.eq(group - 1), "label"]
                mean_return = _number(members.mean())
                day_groups.append({"group": group, "mean_return": mean_return, "assets": int(len(members))})
                pooled[group].append(mean_return)
            daily_quantiles.append({"date": day_string, "groups": day_groups})
    ic_mean, ic_ir = _summary([d["ic"] for d in daily])
    rank_ic_mean, rank_ic_ir = _summary([d["rank_ic"] for d in daily])
    valid_days = sum(d["ic"] is not None for d in daily)
    metrics = {"ic_mean": ic_mean, "rank_ic_mean": rank_ic_mean, "ic_ir": ic_ir, "rank_ic_ir": rank_ic_ir,
               "coverage": coverage, "valid_ic_days": valid_days,
               "valid_rank_ic_days": sum(d["rank_ic"] is not None for d in daily),
               "icir": ic_ir, "rank_icir": rank_ic_ir, "valid_days": valid_days}
    quantiles = [{"group": group, "mean_return": _number(np.mean(values)), "valid_days": len(values)}
                 for group, values in pooled.items() if values]
    values = f.loc[paired].to_numpy(dtype=float)
    distribution = []
    if len(values):
        counts, edges = np.histogram(values, bins=10)
        distribution = [{"bin": f"{edges[i]:.6g} – {edges[i+1]:.6g}", "count": int(count)} for i, count in enumerate(counts)]
    return {"segment": segment, "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(end).strftime("%Y-%m-%d"), "metrics": metrics, "daily": daily,
            "quantiles": quantiles, "daily_quantiles": daily_quantiles, "distribution": distribution,
            "eligible_rows": denominator, "paired_rows": int(paired.sum()),
            "purged_rows": int(((selected.index.get_level_values("datetime") >= start) & ~mask).sum()),
            "protocol_id": PROTOCOL["id"]}
