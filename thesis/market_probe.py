"""Resumable, isolated A-share acquisition experiment; never publishes app data.

Run with the isolated AKShare environment for daily decoding. The network worker
uses one persistent requests session, one outstanding request and a hard deadline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def select_universe(document):
    items = document['items']
    codes = [item['code'] for item in items]
    if len(codes) != len(set(codes)) or len(codes) != int(document['expected_vendor_total']):
        raise ValueError('universe scan is incomplete or contains duplicate codes')
    selected = []
    for item in items:
        code = item['code']
        if not re.fullmatch(r'(?:sh6\d{5}|sz[03]\d{5}|bj[489]\d{5})', code):
            raise ValueError(f'unexpected A-share code: {code}')
        selected.append(dict(item, asset_class='cdr' if code.startswith('sh689') else 'stock'))
    return selected


def parse_qfq(payload, symbol):
    match = re.match(r'\s*var\s+' + re.escape(symbol) + r'qfq\s*=\s*', payload)
    if not match:
        raise ValueError('unexpected factor assignment')
    try:
        body, end = json.JSONDecoder().raw_decode(payload[match.end():])
        tail = payload[match.end() + end:].strip()
        if not re.fullmatch(r';?\s*(?:/\*.*?\*/\s*)?', tail, re.S):
            raise ValueError('unexpected executable suffix')
        rows = body['data']
        if not rows or int(body['total']) != len(rows):
            raise ValueError('empty or incomplete factor data')
        frame = pd.DataFrame({'datetime': [r['d'] for r in rows], 'qfq_factor': [r['f'] for r in rows]})
        frame['datetime'] = pd.to_datetime(frame['datetime'], errors='raise')
        frame['qfq_factor'] = pd.to_numeric(frame['qfq_factor'], errors='raise')
        if frame['datetime'].duplicated().any() or frame['datetime'].isna().any():
            raise ValueError('invalid factor dates')
        if not np.isfinite(frame['qfq_factor']).all() or not frame['qfq_factor'].gt(0).all():
            raise ValueError('factor must be finite and positive')
        return frame.sort_values('datetime').reset_index(drop=True)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('malformed factor data') from exc


def normalize_daily(rows, symbol):
    frame = pd.DataFrame(rows)
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    if frame.empty or any(c not in frame for c in required):
        raise ValueError('empty decoded daily data or missing fields')
    # The decoder labels daily sessions at UTC midnight. Retain the date label;
    # do not convert it to a market-local timestamp and shift trading sessions.
    frame['datetime'] = pd.to_datetime(frame.pop('date'), errors='coerce', utc=True).dt.tz_localize(None)
    frame['instrument'] = symbol
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce') if col in frame else np.nan
    ohlc = frame[['open', 'high', 'low', 'close']]
    invalid_price = (~np.isfinite(ohlc).all(axis=1) | ohlc.le(0).any(axis=1)
                     | frame['high'].lt(ohlc.max(axis=1)) | frame['low'].gt(ohlc.min(axis=1)))
    invalid_date = frame['datetime'].isna() | frame['datetime'].ne(frame['datetime'].dt.normalize())
    invalid_volume = ~np.isfinite(frame['volume']) | frame['volume'].lt(0)
    duplicates = frame['datetime'].duplicated(keep=False)
    quality = {'rows': len(frame), 'invalid_price_rows': int(invalid_price.sum()),
               'invalid_date_rows': int(invalid_date.sum()), 'invalid_volume_rows': int(invalid_volume.sum()),
               'duplicate_key_rows': int(duplicates.sum()), 'zero_volume_rows': int(frame['volume'].eq(0).sum()),
               'missing_amount_rows': int(frame['amount'].isna().sum()),
               'daily_quality_passed': not bool((invalid_price | invalid_date | invalid_volume | duplicates).any())}
    return frame.sort_values('datetime').reset_index(drop=True), quality


def verified_checkpoint(root, record):
    if record.get('status') != 'success' or len(record.get('artifacts', [])) < 2:
        return False
    root = Path(root).resolve()
    for artifact in record['artifacts']:
        path = (root / artifact['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path.read_bytes()) != artifact['sha256']:
            return False
    return True


def archived_transport(root, record):
    if record.get('http_status') != 200 or not record.get('artifacts'):
        return None
    artifact = record['artifacts'][0]
    root = Path(root).resolve()
    path = (root / artifact['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    content = path.read_bytes()
    if sha256(content) != artifact['sha256']:
        return None
    return {'http_status': 200, 'body': content, 'http_seconds': 0, 'from_archive': True,
            'original_fetched_at': record.get('fetched_at')}


def _http_worker(connection):
    import requests
    with requests.Session() as session:
        while True:
            url = connection.recv()
            if url is None:
                break
            started = time.monotonic()
            try:
                response = session.get(url, timeout=(4, 8))
                connection.send({'http_status': response.status_code, 'body': response.content,
                                 'final_url': response.url, 'content_type': response.headers.get('content-type'),
                                 'retry_after': response.headers.get('Retry-After'),
                                 'http_seconds': time.monotonic() - started})
            except requests.RequestException as exc:
                connection.send({'error_type': type(exc).__name__, 'message': str(exc),
                                 'http_seconds': time.monotonic() - started})


class BoundedFetcher:
    def __init__(self, deadline=12.0, min_interval=0.5):
        self.deadline, self.min_interval = deadline, min_interval
        self.worker = self.connection = None
        self.last_start = 0.0

    def get(self, url):
        time.sleep(max(0, self.min_interval - (time.monotonic() - self.last_start)))
        if self.worker is None:
            context = mp.get_context('spawn')
            self.connection, child = context.Pipe()
            self.worker = context.Process(target=_http_worker, args=(child,), daemon=True)
            self.worker.start()
            child.close()
        self.last_start = time.monotonic()
        try:
            self.connection.send(url)
            if self.connection.poll(self.deadline):
                return self.connection.recv()
            result = {'error_type': 'HardTimeout', 'message': 'network child exceeded hard deadline',
                      'http_seconds': time.monotonic() - self.last_start}
        except (EOFError, BrokenPipeError, OSError) as exc:
            result = {'error_type': type(exc).__name__, 'message': str(exc),
                      'http_seconds': time.monotonic() - self.last_start}
        self.close()
        return result

    def close(self):
        if self.worker is not None:
            self.worker.terminate()
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                self.worker.kill()
                self.worker.join(timeout=2)
            self.connection.close()
            self.worker = self.connection = None


def summarize(root, items, started_at, started_clock, halted=None):
    records = [json.loads(p.read_text()) for p in (root / 'records').glob('*.json')]
    states = Counter(r['status'] for r in records)
    success = [r for r in records if r['status'] == 'success']
    exchanges = {}
    for exchange in ['sh', 'sz', 'bj']:
        subset = [r for r in records if r['instrument'].startswith(exchange)]
        exchanges[exchange] = {'target': sum(i['code'].startswith(exchange) for i in items),
                               'attempted': len(subset), 'status_counts': dict(Counter(r['status'] for r in subset))}
    result = {'scope': 'Current vendor A-share stock universe; CDR excluded; not historical/delisted full universe',
              'started_at': started_at, 'updated_at': utc_now(), 'invocation_seconds': round(time.monotonic() - started_clock, 3),
              'target_instruments': len(items), 'recorded_instruments': len(records),
              'unattempted_instruments': len(items) - len(records), 'status_counts': dict(states),
              'normalized_rows': sum(r.get('rows', 0) for r in success),
              'decoded_raw_rows': sum(r.get('decoded_raw_rows', 0) for r in success),
              'daily_quality_failed_instruments': sum(not r.get('quality', {}).get('daily_quality_passed', True) for r in success),
              'latest_date_counts': dict(Counter(r.get('end') for r in success)), 'exchanges': exchanges,
              'halted': halted, 'research_ready': False, 'final_test_evaluated': False,
              'limits': 'Acquisition/decoding only. No adjusted panel, historical universe or strategy validity certified.'}
    atomic_json(root / 'progress.json', result)
    return result


def collect(args):
    root, universe_path = args.output.resolve(), args.universe.resolve()
    production = Path(__file__).resolve().parents[1] / 'runtime/data'
    if root == production or root.is_relative_to(production):
        raise ValueError('probe must not overwrite production data')
    document = json.loads(universe_path.read_text())
    all_items = select_universe(document)
    items = [i for i in all_items if i['asset_class'] == 'stock']
    root.mkdir(parents=True, exist_ok=True)
    metadata = {'universe_sha256': sha256(universe_path.read_bytes()), 'asset': args.asset,
                'start': args.start, 'end': args.end, 'source': 'Sina static realstock files',
                'min_interval_seconds': args.interval, 'network_concurrency': 1, 'hard_deadline_seconds': 12,
                'universe_vendor_rows': len(all_items), 'excluded_cdr': [i['code'] for i in all_items if i['asset_class'] == 'cdr'],
                'volume_unit': 'shares', 'amount_unit': 'CNY', 'adjustment': 'unadjusted daily; factors stored independently'}
    if (root / 'metadata.json').exists():
        previous = json.loads((root / 'metadata.json').read_text())
        if any(previous[k] != metadata[k] for k in ['universe_sha256', 'asset', 'start', 'end']):
            raise ValueError('resume metadata differs; choose another output directory')
    atomic_json(root / 'metadata.json', metadata)
    (root / 'records').mkdir(exist_ok=True)
    (root / 'raw').mkdir(exist_ok=True)
    (root / 'parquet').mkdir(exist_ok=True)
    decoder = None
    if args.asset == 'daily':
        from akshare.stock.cons import hk_js_decode
        from py_mini_racer import MiniRacer
        decoder = MiniRacer()
        decoder.eval(hk_js_decode)
    fetcher = BoundedFetcher(min_interval=args.interval)
    started_at, started_clock = utc_now(), time.monotonic()
    failures, halted, processed = 0, None, 0
    try:
        for item in items[:args.limit or None]:
            symbol = item['code']
            record_path = root / 'records' / f'{symbol}.json'
            if record_path.exists() and verified_checkpoint(root, json.loads(record_path.read_text())):
                continue
            suffix = 'hisdata_klc2/klc_kl.js' if args.asset == 'daily' else 'qfq.js'
            url = f'https://finance.sina.com.cn/realstock/company/{symbol}/{suffix}'
            record = {'instrument': symbol, 'name': item.get('name'), 'url': url, 'fetched_at': utc_now(),
                      'asset': args.asset, 'artifacts': [], 'status': 'failed'}
            seed = args.seed / f'pilot-{symbol}-{args.asset}.response' if args.seed else None
            replay = (archived_transport(root, json.loads(record_path.read_text()))
                      if args.replay_existing and record_path.exists() else None)
            transport = (replay or ({'http_status': 200, 'body': seed.read_bytes(), 'http_seconds': 0, 'from_seed': True}
                                   if seed and seed.is_file() else fetcher.get(url)))
            content = transport.pop('body', None)
            record.update(transport)
            if transport.get('original_fetched_at'):
                record['fetched_at'] = transport['original_fetched_at']
                record['processed_at'] = utc_now()
            if content is not None:
                raw = root / 'raw' / f'{symbol}.response'
                raw.write_bytes(content)
                record['artifacts'].append({'path': str(raw.relative_to(root)), 'sha256': sha256(content)})
                record['bytes'] = len(content)
            if 'error_type' in transport:
                record['status'] = 'transport_error'
                failures += 1
            elif transport['http_status'] != 200:
                record['status'] = 'http_error'
                failures = 0
                if transport['http_status'] in [403, 429]:
                    halted = f"HTTP {transport['http_status']}; stop source without bypass or immediate retry"
            else:
                failures = 0
                try:
                    text = content.decode('utf-8-sig')
                    if args.asset == 'daily':
                        match = re.match(r'\s*var\s+KLC_K2_' + re.escape(symbol) + r'\s*=\s*("[^";]+")', text)
                        if not match:
                            raise ValueError('unexpected K-line payload; no remote JavaScript evaluated')
                        decoded = decoder.call('d', json.loads(match[1]))
                        frame, quality = normalize_daily(decoded, symbol)
                        record['decoded_raw_rows'] = len(frame)
                        record['raw_start'] = str(frame['datetime'].min().date())
                        record['raw_end'] = str(frame['datetime'].max().date())
                        frame = frame.loc[frame['datetime'].between(args.start, args.end)].copy()
                        record['quality'] = quality
                    else:
                        frame = parse_qfq(text, symbol)
                        frame['instrument'] = symbol
                    path = root / 'parquet' / f'{symbol}.parquet'
                    temporary = path.with_suffix('.tmp')
                    frame.to_parquet(temporary, index=False)
                    os.replace(temporary, path)
                    record['artifacts'].append({'path': str(path.relative_to(root)), 'sha256': sha256(path.read_bytes())})
                    record.update(status='success', rows=len(frame),
                                  start=str(frame['datetime'].min().date()) if len(frame) else None,
                                  end=str(frame['datetime'].max().date()) if len(frame) else None)
                except Exception as exc:
                    record.update(status='parse_error', error_type=type(exc).__name__, message=str(exc)[:1500])
                    lowered = content[:10000].lower()
                    if b'captcha' in lowered or b'access denied' in lowered:
                        halted = 'access restriction response; stop source'
            atomic_json(record_path, record)
            processed += 1
            if failures >= 3:
                halted = 'three consecutive transport failures; source circuit opened'
            if processed % 25 == 0 or halted:
                summary = summarize(root, items, started_at, started_clock, halted)
                print(json.dumps({k: summary[k] for k in ['recorded_instruments', 'target_instruments', 'status_counts', 'normalized_rows', 'invocation_seconds', 'halted']}), flush=True)
            if halted:
                break
    finally:
        fetcher.close()
        result = summarize(root, items, started_at, started_clock, halted)
        atomic_json(root / 'report.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--asset', choices=['daily', 'qfq'], default='daily')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-10-30')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--replay-existing', action='store_true', help='reparse verified existing HTTP-200 raw assets before requesting again')
    args = parser.parse_args()
    if args.interval < 0.5 or args.limit < 0 or pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error('minimum interval is 0.5 seconds; limit and date range must be valid')
    collect(args)


if __name__ == '__main__':
    main()
