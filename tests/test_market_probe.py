"""Guard against corrupting a full-market acquisition with silent substitutions."""

import json
from pathlib import Path

import pandas as pd
import pytest

from thesis.market_probe import archived_transport, normalize_daily, parse_qfq, select_universe, verified_checkpoint


def test_factor_parser_accepts_json_assignment_but_never_javascript():
    text = 'var sh600000qfq={"total":2,"data":[{"d":"2020-01-02","f":"1.25"},{"d":"1900-01-01","f":"2"}]}\n/* checksum */'
    factors = parse_qfq(text, 'sh600000')
    assert factors['qfq_factor'].tolist() == [2.0, 1.25]
    assert factors['datetime'].dt.strftime('%Y-%m-%d').tolist() == ['1900-01-01', '2020-01-02']
    with pytest.raises(ValueError):
        parse_qfq('var sh600000qfq=__import__("os").system("false")', 'sh600000')
    with pytest.raises(ValueError):
        parse_qfq(text + '; alert(1)', 'sh600000')
    with pytest.raises(ValueError):
        parse_qfq(text, 'sz000001')


@pytest.mark.parametrize('factor', ['0', '-1', 'NaN', 'Infinity'])
def test_invalid_factor_cannot_become_a_valid_research_asset(factor):
    text = 'var sh600000qfq=' + json.dumps({'total': 1, 'data': [{'d': '2020-01-02', 'f': factor}]})
    with pytest.raises(ValueError):
        parse_qfq(text, 'sh600000')


def test_raw_daily_keeps_zero_volume_and_identical_prices_on_different_dates():
    rows = [dict(date=date, open=10, high=11, low=9, close=10, volume=0, amount=0)
            for date in ['2020-01-02T00:00:00Z', '2020-01-03T00:00:00Z']]
    frame, quality = normalize_daily(rows, 'sh600000')
    assert len(frame) == 2
    assert frame['datetime'].dt.strftime('%Y-%m-%d').tolist() == ['2020-01-02', '2020-01-03']
    assert frame['volume'].tolist() == [0, 0]
    assert 'adj_close' not in frame
    assert quality['zero_volume_rows'] == 2
    assert quality['invalid_price_rows'] == 0


def test_duplicate_dates_and_missing_prices_are_reported_without_filling():
    rows = [dict(date='2020-01-02', open=10, high=11, low=9, close=10, volume=100),
            dict(date='2020-01-02', open=10, high=11, low=9, close=None, volume=100)]
    frame, quality = normalize_daily(rows, 'sh600000')
    assert len(frame) == 2
    assert frame['close'].isna().sum() == 1
    assert quality['duplicate_key_rows'] == 2
    assert quality['invalid_price_rows'] == 1
    assert quality['daily_quality_passed'] is False


def test_universe_rejects_incomplete_scan_and_separately_labels_cdr():
    document = {'expected_vendor_total': 3, 'items': [
        {'code': 'sh600000', 'name': '浦发银行'},
        {'code': 'sh689009', 'name': '九号公司'},
        {'code': 'bj920001', 'name': '样本'},
    ]}
    selected = select_universe(document)
    assert [x['asset_class'] for x in selected] == ['stock', 'cdr', 'stock']
    document['items'].append({'code': 'sh600000', 'name': '重复'})
    with pytest.raises(ValueError):
        select_universe(document)
    document['items'].pop()
    document['expected_vendor_total'] = 4
    with pytest.raises(ValueError):
        select_universe(document)


def test_resume_rechecks_hashes_and_does_not_skip_a_failed_or_corrupt_asset(tmp_path):
    import hashlib
    raw = tmp_path / 'raw.response'
    normalized = tmp_path / 'daily.parquet'
    raw.write_bytes(b'original')
    normalized.write_bytes(b'normalized')
    record = {'status': 'success', 'artifacts': [
        {'path': str(raw.relative_to(tmp_path)), 'sha256': hashlib.sha256(b'original').hexdigest()},
        {'path': str(normalized.relative_to(tmp_path)), 'sha256': hashlib.sha256(b'normalized').hexdigest()},
    ]}
    assert verified_checkpoint(tmp_path, record)
    raw.write_bytes(b'changed')
    assert not verified_checkpoint(tmp_path, record)
    raw.write_bytes(b'original')
    assert not verified_checkpoint(tmp_path, dict(record, status='failed'))


def test_offline_replay_only_reuses_http_success_with_matching_raw_hash(tmp_path):
    import hashlib
    raw = tmp_path / 'raw.response'
    raw.write_bytes(b'previous raw data')
    record = {'http_status': 200, 'fetched_at': '2026-11-02T00:00:00Z', 'artifacts': [
        {'path': 'raw.response', 'sha256': hashlib.sha256(b'previous raw data').hexdigest()},
    ]}
    replay = archived_transport(tmp_path, record)
    assert replay['body'] == b'previous raw data'
    assert replay['original_fetched_at'] == '2026-11-02T00:00:00Z'
    raw.write_bytes(b'changed')
    assert archived_transport(tmp_path, record) is None
    raw.write_bytes(b'previous raw data')
    assert archived_transport(tmp_path, dict(record, http_status=403)) is None


def test_network_child_has_hard_deadline_and_recovers_for_next_request():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import time
    from thesis.market_probe import BoundedFetcher

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/slow':
                time.sleep(0.8)
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(b'ok')
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fetcher = BoundedFetcher(deadline=3, min_interval=0)
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        assert fetcher.get(url + '/ok')['body'] == b'ok'
        fetcher.deadline = 0.2
        before = time.monotonic()
        result = fetcher.get(url + '/slow')
        assert result['error_type'] == 'HardTimeout'
        assert time.monotonic() - before < 1.0
        fetcher.deadline = 3
        assert fetcher.get(url + '/ok')['body'] == b'ok'
    finally:
        fetcher.close()
        server.shutdown()
        server.server_close()
