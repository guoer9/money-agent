"""Download Tencent's 640-bar snapshots without splicing adjustment anchors.

Run: python -m thesis.acquisition --output runtime/data --end 2026-10-30
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import httpx
import pandas as pd

from .data import clean_panel

ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
# A deliberately fixed convenience universe, NOT historical index constituents.
DEFAULT_UNIVERSE = (
    "sh600000", "sh600009", "sh600016", "sh600028", "sh600030", "sh600036",
    "sh600048", "sh600050", "sh600104", "sh600111", "sh600276", "sh600309",
    "sh600519", "sh600585", "sh600690", "sh600887", "sh601088", "sh601166",
    "sh601318", "sh601398", "sh601668", "sh601857", "sz000001", "sz000002",
    "sz000063", "sz000333", "sz000651", "sz000858", "sz002415", "sz300750",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def parse_tencent_response(response: dict, code: str) -> pd.DataFrame:
    """qfqday order: date/open/close/high/low/volume (lots of 100 shares).

    Unadjusted ``day`` is intentionally not a stock-price fallback: a silent
    mixed-adjustment panel would not be comparable across assets or dates.
    """
    try:
        if response.get("code") != 0:
            raise ValueError(f"Tencent provider error: {response.get('msg')}")
        bars = response["data"][code]["qfqday"]
        if not isinstance(bars, list) or not bars:
            raise ValueError("empty qfqday response")
        rows = []
        for bar in bars:
            if not isinstance(bar, list) or len(bar) < 6:
                raise ValueError("malformed qfqday bar")
            date = pd.Timestamp(bar[0])
            if pd.isna(date) or date.tz is not None or date != date.normalize():
                raise ValueError("invalid daily bar date")
            open_, close, high, low, volume = map(float, bar[1:6])
            rows.append({"datetime": date, "instrument": code, "open": open_, "close": close,
                         "high": high, "low": low, "volume": volume, "adj_close": close})
        return pd.DataFrame(rows)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"missing or malformed qfqday for {code}") from exc


def normalized_prices_sha256(frame: pd.DataFrame) -> str:
    """Stable price-content hash, independent of quote wrappers/event metadata.

    v1 hashes parsed date, instrument, OHLC and source-unit volume, ordered by
    date/instrument. Duplicates are retained; this does not certify data quality.
    """
    columns = ["datetime", "instrument", "open", "high", "low", "close", "volume"]
    rows = frame[columns].sort_values(["datetime", "instrument"], kind="stable").copy()
    rows["datetime"] = rows["datetime"].dt.strftime("%Y-%m-%d")
    records = rows.to_dict("records")
    for row in records:
        for column in columns[2:]:
            number = row[column]
            if not math.isfinite(number):
                row[column] = "NaN" if math.isnan(number) else "Infinity" if number > 0 else "-Infinity"
    content = json.dumps({"schema": "alphalab-ohlcv-v1", "rows": records},
                         ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def fetch_market(codes, start, end, output, *, client=None, min_rows: int = 10000,
                 min_instruments: int = 2, retries: int = 2, calendar_code: str | None = None) -> dict:
    """One common 640-bar qfq query per stock, immutable raw responses + hashes.

    An optional broad-market index supplies the common trading calendar. Without
    it the union of observed sessions is used and the limitation is recorded.
    Failure bodies are retained; an insufficient snapshot raises, never fakes.
    """
    codes = list(dict.fromkeys(codes))
    if not codes or any(not isinstance(code, str) or re.fullmatch(r"(?:sh|sz)\d{6}", code) is None for code in codes):
        raise ValueError("codes must be nonempty Tencent sh/sz instrument identifiers")
    if calendar_code is not None and re.fullmatch(r"(?:sh|sz)\d{6}", calendar_code) is None:
        raise ValueError("invalid calendar index code")
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(lower) or pd.isna(upper) or lower > upper or retries < 0 or min_rows < 1
            or type(min_instruments) is not int or min_instruments < 1):
        raise ValueError("invalid download range, row threshold or retry count")
    start, end = lower.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
    session = client or httpx.Client(timeout=httpx.Timeout(25), follow_redirects=True)
    owned = client is None

    def download(code, *, calendar_request=False):
        records = []
        last_error = "unknown acquisition error"
        # Index prices are only used for dates; never mixed into the stock panel.
        param = f"{code},day,{start},{end},640,qfq"
        for attempt in range(retries + 1):
            record = {"instrument": code, "role": "calendar" if calendar_request else "prices",
                      "attempt": attempt + 1, "downloaded_at": _now(),
                      "url": str(httpx.URL(ENDPOINT, params={"param": param}))}
            try:
                response = session.get(ENDPOINT, params={"param": param})
                path = Path("raw") / run_id / f"{code}-{'calendar' if calendar_request else 'prices'}-{attempt + 1}.json"
                _atomic_bytes(output / path, response.content)
                record.update({"path": str(path), "sha256": hashlib.sha256(response.content).hexdigest(),
                               "bytes": len(response.content), "http_status": response.status_code,
                               "url": str(response.url), "content_type": response.headers.get("content-type")})
                response.raise_for_status()
                body = response.json()
                if calendar_request:
                    if body.get("code") != 0:
                        raise ValueError("calendar provider error")
                    block = body["data"][code]
                    bars = block.get("day") or block.get("qfqday")
                    if not bars:
                        raise ValueError("empty calendar response")
                    value = pd.DatetimeIndex([bar[0] for bar in bars]).unique().sort_values()
                    if value.hasnans:
                        raise ValueError("invalid calendar dates")
                else:
                    value = parse_tencent_response(body, code)
                    value = value.loc[value["datetime"].between(lower, upper)]
                    record["normalized_prices_sha256"] = normalized_prices_sha256(value)
                    record["normalized_prices_schema"] = "alphalab-ohlcv-v1"
                record["status"] = "ok"
                records.append(record)
                return value, records, None
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = str(exc)
                record.update({"status": "failed", "error": last_error})
                records.append(record)
                if attempt < retries:
                    time.sleep(0.25 * (2 ** attempt))
        return None, records, {"instrument": code, "role": "calendar" if calendar_request else "prices", "error": last_error}

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(codes))) as pool:
            downloaded = list(pool.map(download, codes))
        calendar_result = download(calendar_code, calendar_request=True) if calendar_code else None
    finally:
        if owned:
            session.close()
    frames = [item[0] for item in downloaded if item[0] is not None]
    raw = [record for item in downloaded for record in item[1]]
    failures = [item[2] for item in downloaded if item[2] is not None]
    if frames:
        panel, quality = clean_panel(pd.concat(frames, ignore_index=True))
    else:
        empty = pd.DataFrame(columns=["datetime", "instrument", "open", "close", "high", "low", "volume", "adj_close"])
        panel, quality = clean_panel(empty)
    observed = pd.DatetimeIndex(panel.index.get_level_values("datetime").unique()).sort_values()
    if calendar_result:
        raw.extend(calendar_result[1])
        if calendar_result[2] is not None:
            failures.append(calendar_result[2])
            calendar = observed
            calendar_source = "observed union (index request failed)"
        else:
            calendar = calendar_result[0]
            calendar = calendar[(calendar >= lower) & (calendar <= upper)]
            # The index's own 640-bar cap must cover all stock sessions.
            calendar_source = f"Tencent {calendar_code} daily index dates"
    else:
        calendar = observed
        calendar_source = "observed union; wholly missing market sessions cannot be inferred"
    mismatch = ~panel.index.get_level_values("datetime").isin(calendar)
    calendar_mismatch_rows_removed = int(mismatch.sum())
    if calendar_mismatch_rows_removed:
        panel = panel.loc[~mismatch].copy()
        panel.attrs["calendar"] = calendar.strftime("%Y-%m-%d").tolist()
    observed = pd.DatetimeIndex(panel.index.get_level_values("datetime").unique()).sort_values()
    calendar_ok = bool(len(calendar) and observed.isin(calendar).all())
    panel.attrs["calendar"] = calendar.strftime("%Y-%m-%d").tolist()
    universe = sorted(panel.index.get_level_values("instrument").unique().tolist())
    counts = panel.groupby(level="instrument").size()
    quality.update({
        "volume_unit": "lots (100 shares)", "volume_converted": False,
        "survivorship_bias": True, "universe_policy": "fixed convenience pool selected at acquisition; not point-in-time constituents",
        "raw_prices_available": False, "adjusted_ohlc": True,
        "adjustment_warning": "vendor qfq snapshot; adjustment factors and point-in-time corporate actions are not provided",
        "calendar_source": calendar_source, "calendar_covers_observed_dates": calendar_ok,
        "calendar_mismatch_rows_removed": calendar_mismatch_rows_removed,
        "valid_rows": int(len(panel)),
        "calendar_sessions": int(len(calendar)), "missing_instrument_sessions": int(len(calendar) * len(universe) - len(panel)),
        "requested_instruments": len(codes), "successful_instruments": len(universe),
        "minimum_instruments_required": min_instruments,
        "rows_by_instrument": {str(k): int(v) for k, v in counts.items()},
        "single_range_per_instrument": True, "anchor_splicing": False, "provider_bar_limit": 640,
        "requested_range_truncated_by_cap": bool(len(calendar) and calendar[0] > lower),
        "final_test_evaluated": False,
    })
    splits = {}
    if len(calendar) >= 10:
        train_end, validation_end = int(len(calendar) * 0.6), int(len(calendar) * 0.8)
        splits = {"train": [str(calendar[0].date()), str(calendar[train_end - 1].date())],
                  "validation": [str(calendar[train_end].date()), str(calendar[validation_end - 1].date())],
                  "test": [str(calendar[validation_end].date()), str(calendar[-1].date())]}
    sufficient = (len(panel) >= min_rows and len(universe) >= min_instruments and calendar_ok
                  and (not calendar_code or calendar_result[2] is None))
    manifest = {
        "id": f"tencent-qfq-{run_id}", "source": "Tencent fqkline HTTP / qfqday",
        "endpoint": ENDPOINT, "downloaded_at": _now(), "status": "ready" if sufficient else "insufficient_data",
        "start": str(observed[0].date()) if len(observed) else None,
        "end": str(observed[-1].date()) if len(observed) else None,
        "rows": int(len(panel)), "instruments": len(universe), "sha256": None,
        "adjustment": "qfq; one consistent 640-bar range per instrument; no anchor splicing",
        "requested_start": start, "requested_end": end, "requested_universe": codes, "universe": universe,
        "quality": quality, "splits": splits, "calendar": panel.attrs["calendar"],
        "calendar_path": "calendar.json", "panel_path": "panel.parquet",
        "raw_responses": raw, "failures": failures,
    }
    if sufficient:
        # Serializing to bytes avoids exposing a partially written Parquet file.
        _atomic_bytes(output / "panel.parquet", panel.to_parquet(index=True))
        manifest["sha256"] = hashlib.sha256((output / "panel.parquet").read_bytes()).hexdigest()
        _atomic_json(output / "calendar.json", {"source": calendar_source, "dates": manifest["calendar"]})
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "raw" / run_id / "manifest.json", manifest)
    if not sufficient:
        raise ValueError(f"only {len(panel)} valid rows across {len(universe)} instruments; require {min_rows} rows, "
                         f"{min_instruments} multi-instrument bars and a complete common calendar; see manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runtime/data"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    manifest = fetch_market(DEFAULT_UNIVERSE, args.start, args.end, args.output, calendar_code="sh000001")
    print(json.dumps({key: manifest[key] for key in ("id", "status", "start", "end", "rows", "instruments", "sha256", "splits", "failures")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
