import hashlib
import json

import httpx
import pandas as pd
import pytest

from thesis.acquisition import fetch_market, parse_tencent_response


def payload(code='sh600000'):
    return {'code': 0, 'msg': '', 'data': {code: {'qfqday': [
        ['2024-01-02', '10', '11', '12', '9', '123.45'],
        ['2024-01-03', '11', '12', '13', '10', '456.78'],
    ], 'qt': {code: ['1', '浦发银行', '600000']}}}}


def test_parser_preserves_vendor_units_and_adjustment():
    frame = parse_tencent_response(payload(), 'sh600000')
    assert frame.iloc[0]['open'] == 10
    assert frame.iloc[0]['close'] == 11
    assert frame.iloc[0]['high'] == 12
    assert frame.iloc[0]['low'] == 9
    assert frame.iloc[0]['volume'] == 123.45
    assert frame.iloc[0]['adj_close'] == 11
    assert frame.iloc[0]['instrument'] == 'sh600000'


@pytest.mark.parametrize('response', [{'code': -1, 'msg': 'failure'}, {'code': 0, 'data': {}},
    {'code': 0, 'data': {'sh600000': {'day': [['2024-01-02','1','1','1','1','1']]}}},
    {'code': 0, 'data': {'sh600000': {'qfqday': [['2024-01-02','bad','1']]}}}])
def test_parser_rejects_errors_unadjusted_fallback_and_malformed_bars(response):
    with pytest.raises(ValueError):
        parse_tencent_response(response, 'sh600000')


def test_fetch_archives_exact_response_hash_and_single_adjustment_range(tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=payload(), request=request)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path, client=client, min_rows=2, min_instruments=1)
    assert len(calls) == 1
    assert calls[0].url.params['param'] == 'sh600000,day,2024-01-01,2024-12-31,640,qfq'
    record = manifest['raw_responses'][0]
    raw = (tmp_path / record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    assert 'web.ifzq.gtimg.cn' in record['url']
    frame = pd.read_parquet(tmp_path / 'panel.parquet')
    assert len(frame) == 2
    assert manifest['rows'] == 2
    assert manifest['quality']['volume_unit'] == 'lots (100 shares)'
    assert manifest['quality']['survivorship_bias'] is True
    assert hashlib.sha256((tmp_path / 'panel.parquet').read_bytes()).hexdigest() == manifest['sha256']


def test_price_fingerprint_ignores_wrapper_but_detects_changed_price(tmp_path):
    bodies = [payload(), payload(), payload()]
    bodies[1]['data']['sh600000']['qt']['sh600000'][1] = '不同的报价包装'
    bodies[1]['data']['sh600000']['qfqday'][0].append({'FHcontent': '附属信息'})
    bodies[2]['data']['sh600000']['qfqday'][0][2] = '11.5'
    records = []
    for i, body in enumerate(bodies):
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body, request=req))) as client:
            manifest = fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path / str(i),
                                    client=client, min_rows=2, min_instruments=1)
        records.append(manifest['raw_responses'][0])
    assert records[0]['sha256'] != records[1]['sha256']
    assert records[0]['normalized_prices_sha256'] == records[1]['normalized_prices_sha256']
    assert records[0]['normalized_prices_sha256'] != records[2]['normalized_prices_sha256']


def test_nonfinite_bar_is_still_cleaned_without_rejecting_the_instrument(tmp_path):
    body = payload()
    body['data']['sh600000']['qfqday'].append(['2024-01-04', '12', 'NaN', '13', '10', '100'])
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body, request=req))) as client:
        manifest = fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path,
                                client=client, min_rows=2, min_instruments=1, retries=0)
    assert manifest['rows'] == 2
    assert manifest['quality']['invalid_removed'] == 1
    assert manifest['raw_responses'][0]['normalized_prices_sha256']


def test_fetch_fails_below_threshold_without_synthetic_replacement(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503, text='unavailable', request=req))) as client:
        with pytest.raises(ValueError, match='valid rows'):
            fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path, client=client, min_rows=10000, retries=0)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['status'] == 'insufficient_data'
    assert manifest['failures'][0]['instrument'] == 'sh600000'
    assert (tmp_path / manifest['raw_responses'][0]['path']).read_text() == 'unavailable'
    assert not (tmp_path / 'panel.parquet').exists()


def test_index_calendar_excludes_unmatched_stock_dates_and_records_quality(tmp_path):
    def respond(request):
        code = request.url.params['param'].split(',')[0]
        if code == 'sh000001':
            body = {'code': 0, 'data': {'sh000001': {'day': [['2024-01-02','1','1','1','1','1']]}}}
        else:
            body = payload(code)
        return httpx.Response(200, json=body, request=request)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        manifest = fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path,
                                client=client, min_rows=1, min_instruments=1, calendar_code='sh000001')
    panel = pd.read_parquet(tmp_path / 'panel.parquet')
    assert len(panel) == 1
    assert panel.index.get_level_values('datetime').unique().tolist() == [pd.Timestamp('2024-01-02')]
    assert manifest['quality']['calendar_covers_observed_dates'] is True
    assert manifest['quality']['calendar_mismatch_rows_removed'] == 1
    assert manifest['quality']['missing_instrument_sessions'] == 0


def test_default_threshold_rejects_single_instrument_as_not_multi_stock(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload(), request=req))) as client:
        with pytest.raises(ValueError, match='multi-instrument'):
            fetch_market(['sh600000'], '2024-01-01', '2024-12-31', tmp_path,
                         client=client, min_rows=1, retries=0)
