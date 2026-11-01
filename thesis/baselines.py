"""Real deterministic classic factors. Only validation is evaluated in v1.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .acquisition import _atomic_json, _now
from .evaluation import PROTOCOL, evaluate
from .factors import calculate

BASELINES = (
    {"id": "baseline-momentum-20", "name": "20日动量", "expression": "cs_rank(safe_div(close, lag(close, 20)) - 1)",
     "description": "按过去20个交易日收盘涨幅排序，较高值表示较强中期趋势。"},
    {"id": "baseline-reversal-5", "name": "5日反转", "expression": "cs_rank(1 - safe_div(close, lag(close, 5)))",
     "description": "过去5日涨幅的反向排序，较高值表示近期回落较多。"},
    {"id": "baseline-low-volatility-20", "name": "20日低波动", "expression": "cs_rank(-std(safe_div(close, lag(close, 1)) - 1, 20))",
     "description": "过去20个日收益率的样本标准差反向排序，较高值表示波动较低。"},
    {"id": "baseline-volume-ratio-5-20", "name": "5/20日量比", "expression": "cs_rank(safe_div(mean(volume, 5), mean(volume, 20)))",
     "description": "短期平均成交量相对中期成交量的排序，使用来源的手数单位。"},
    {"id": "baseline-price-ma-20", "name": "20日均线偏离", "expression": "cs_rank(safe_div(close, mean(close, 20)) - 1)",
     "description": "当日收盘价相对20日均价的偏离，较高值表示价格处于均线上方。"},
    {"id": "baseline-intraday-strength", "name": "日内强弱", "expression": "cs_rank(safe_div(close - open, high - low))",
     "description": "收开盘价差除以日内振幅，较高值表示日内收盘较强；零振幅记为缺失。"},
)


def build_baselines(root) -> dict:
    """Read the frozen manifest, calculate through validation end, archive atomically.

    The train interval supplies trailing history only. No factor, label, IC,
    quantile or histogram is calculated for the reserved final test interval.
    """
    root = Path(root)
    data = root / "runtime" / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in (None, "ready"):
        raise ValueError("dataset manifest is not ready")
    splits = manifest.get("splits", {})
    try:
        bounds = {key: tuple(pd.Timestamp(x) for x in splits[key]) for key in ("train", "validation", "test")}
        if any(len(value) != 2 or pd.isna(value[0]) or pd.isna(value[1]) or value[0] > value[1] for value in bounds.values()):
            raise ValueError("invalid split boundaries")
        if not bounds["train"][1] < bounds["validation"][0] <= bounds["validation"][1] < bounds["test"][0]:
            raise ValueError("splits overlap or are not chronological")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("train/validation/test splits are required") from exc
    panel_path = data / "panel.parquet"
    fingerprint = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != fingerprint:
        raise ValueError("dataset sha256 does not match its frozen manifest")
    # Predicate pushdown prevents reading test prices into the research panel.
    validation_start, validation_end = bounds["validation"]
    panel = pd.read_parquet(panel_path, filters=[("datetime", "<=", validation_end)])
    if panel.empty:
        raise ValueError("no data available through validation end")
    calendar = manifest.get("calendar") or panel.attrs.get("calendar")
    if calendar:
        panel.attrs["calendar"] = [str(pd.Timestamp(d).date()) for d in calendar if pd.Timestamp(d) <= validation_end]
    factors = []
    for definition in BASELINES:
        factor = calculate(definition["expression"], panel)
        evaluation = evaluate(factor, panel, "validation", start=validation_start, end=validation_end)
        factors.append({**definition, "origin": "baseline", "generator": "deterministic classic expression; no LLM",
                        **evaluation})
    dataset = {key: manifest.get(key) for key in ("id", "source", "start", "end", "rows", "instruments", "adjustment", "quality", "splits")}
    dataset["sha256"] = fingerprint
    dataset.setdefault("quality", {})
    dataset["start"] = dataset["start"] or str(bounds["train"][0].date())
    dataset["end"] = dataset["end"] or str(bounds["test"][1].date())
    if dataset["rows"] is None or dataset["instruments"] is None:
        import pyarrow.parquet as pq
        dataset["rows"] = pq.ParquetFile(panel_path).metadata.num_rows
        dataset["instruments"] = int(panel.index.get_level_values("instrument").nunique())
    dataset["adjustment"] = dataset["adjustment"] or "as supplied; fixture or external source"
    dataset["universe"] = manifest.get("universe", sorted(panel.index.get_level_values("instrument").unique().tolist()))
    dataset["manifest_path"] = "runtime/data/manifest.json"
    archive = {"schema_version": "1.2.0", "dataset": dataset, "protocol": dict(PROTOCOL),
               "factors": factors, "generated_at": _now(), "origin": "baseline",
               "evaluated_segments": ["validation"], "test_evaluated": False,
               "final_test": {"status": "reserved_not_run", "message": "最终测试已按日期保留，未计算因子、标签或指标。"},
               "selection_policy": "all six prespecified classic baselines retained; no validation-based direction reversal",
               "limitations": ["固定股票池存在幸存偏差，非历史时点成分。", "前复权为下载时点快照，非严格历史时点复权。",
                               "分层仅为统计性持有期收益，不含费用、滑点、涨跌停与交易可达性。",
                               "这是经典因子复算，未运行大语言模型生成或反馈实验。"]}
    _atomic_json(root / "runtime" / "artifacts" / "baselines.json", archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    archive = build_baselines(args.root)
    print(json.dumps({"dataset": archive["dataset"], "factors": [{"id": f["id"], "metrics": f["metrics"]} for f in archive["factors"]],
                      "test_evaluated": archive["test_evaluated"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
