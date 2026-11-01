import numpy as np
import pandas as pd
import pytest

from thesis.data import clean_panel, forward_labels, segment_mask
from thesis.factors import calculate, validate_expression
from thesis.evaluation import evaluate


def panel(days=12, stocks=6):
    rows=[]
    for d,date in enumerate(pd.bdate_range('2022-01-03',periods=days)):
        for s in range(stocks):
            close=20+s+d*(s+1)*0.1
            rows.append(dict(datetime=date,instrument=f'S{s}',open=close,high=close+1,low=close-1,close=close,volume=1000+s*20,adj_close=close))
    return pd.DataFrame(rows).set_index(['datetime','instrument']).sort_index()


def test_labels_align_by_stock_and_exit_day():
    p=panel().sample(frac=1,random_state=2)
    y=forward_labels(p,1)
    expected=(20+0.2)/(20+0.1)-1
    assert y.loc[(pd.Timestamp('2022-01-03'),'S0')]==pytest.approx(expected)
    assert y.groupby(level='instrument').apply(lambda x:x.isna().sum()).eq(2).all()
    mask=segment_mask(p,'2022-01-03','2022-01-07',1)
    assert mask.sum()==18


def test_expression_is_causal_and_index_invariant():
    p=panel()
    expr='cs_rank(safe_div(close, lag(close, 3)) - 1)'
    full=calculate(expr,p)
    prefix=calculate(expr,p.loc[:'2022-01-12'])
    pd.testing.assert_series_equal(full.reindex(prefix.index),prefix)
    pd.testing.assert_series_equal(full,calculate(expr,p.sample(frac=1,random_state=5)).reindex(full.index))


@pytest.mark.parametrize('expr',['lag(close, -1)','close.shift(-1)','__import__("os")','mean(close, 0)','unknown(close)','close[0]'])
def test_unsafe_or_future_expression_rejected(expr):
    with pytest.raises(ValueError): validate_expression(expr)


def test_cleaning_records_exclusions_without_backfill():
    p=panel().reset_index(); p.loc[0,'close']=-1; p=pd.concat([p,p.iloc[[1]]])
    result,report=clean_panel(p)
    assert report['duplicates_removed']==1
    assert report['invalid_removed']==1
    assert len(result)==70
    assert result.index.is_unique


def test_daily_ic_and_groups_are_real():
    p=panel(days=20)
    f=calculate('cs_rank(close)',p)
    result=evaluate(f,p,'validation',min_assets=5)
    assert result['metrics']['rank_ic_mean']==pytest.approx(1)
    assert len(result['daily'])==18
    assert len(result['daily_quantiles'])==18
    assert len(result['quantiles'])==5
    assert result['metrics']['coverage']==1


def test_constant_factor_reports_null_instead_of_zero():
    p=panel(); f=pd.Series(1.0,index=p.index)
    result=evaluate(f,p,'validation',min_assets=5)
    assert result['metrics']['ic_mean'] is None
    assert result['metrics']['rank_ic_mean'] is None
    assert result['metrics']['icir'] is None
    assert result['metrics']['valid_days']==0


def test_missing_trading_date_does_not_shorten_horizon():
    p=panel(); p=p.drop((pd.Timestamp('2022-01-05'),'S0'))
    y=forward_labels(p,1)
    assert np.isnan(y.loc[(pd.Timestamp('2022-01-03'),'S0')])


def test_purge_uses_exit_date_for_each_segment():
    p = panel(days=12)
    mask = segment_mask(p, '2022-01-10', '2022-01-14', 1)
    assert sorted(p.index.get_level_values('datetime')[mask].unique()) == list(pd.to_datetime(['2022-01-10', '2022-01-11', '2022-01-12']))
    result = evaluate(calculate('close', p), p, 'validation', start='2022-01-10', end='2022-01-14')
    assert [x['date'] for x in result['daily']] == ['2022-01-10', '2022-01-11', '2022-01-12']


def test_rolling_and_lag_do_not_skip_a_missing_market_session():
    p = panel().drop((pd.Timestamp('2022-01-05'), 'S0'))
    lagged = calculate('lag(close, 1)', p)
    means = calculate('mean(close, 2)', p)
    assert np.isnan(lagged.loc[(pd.Timestamp('2022-01-06'), 'S0')])
    assert np.isnan(means.loc[(pd.Timestamp('2022-01-06'), 'S0')])
    assert means.loc[(pd.Timestamp('2022-01-04'), 'S1')] == pytest.approx(21.1)


def test_rank_is_per_day_and_future_mutations_do_not_change_past():
    p = panel()
    ranked = calculate('cs_rank(close)', p)
    assert ranked.loc[(pd.Timestamp('2022-01-03'), 'S0')] == pytest.approx(1 / 6)
    changed = p.copy()
    changed.loc[pd.IndexSlice['2022-01-12':, :], 'close'] *= 100
    left = calculate('std(close, 3) + delta(close, 2)', p).loc[:'2022-01-11']
    right = calculate('std(close, 3) + delta(close, 2)', changed).loc[:'2022-01-11']
    pd.testing.assert_series_equal(left, right)


@pytest.mark.parametrize('expr', ['lag(close, 0)', 'mean(close, 1.5)', 'lag(close, True)',
    'mean(close, 10001)', 'close ** 1000000', 'lambda: close', '(x for x in close)',
    'safe_div(close, 0, bad=1)', 'forward_return', '1e999', '"close"'])
def test_ast_rejects_resource_abuse_and_non_whitelisted_fields(expr):
    with pytest.raises(ValueError):
        validate_expression(expr)


def test_zero_division_and_missing_factors_never_become_fake_zero_metrics():
    p = panel()
    f = calculate('safe_div(close, close - close)', p)
    assert f.isna().all()
    result = evaluate(f, p, 'validation')
    assert result['metrics']['ic_ir'] is None
    assert result['metrics']['rank_ic_ir'] is None
    assert result['metrics']['coverage'] == 0
    assert result['metrics']['valid_ic_days'] == 0
    assert result['quantiles'] == []


def test_thin_cross_sections_and_constant_labels_are_invalid():
    p = panel()
    result = evaluate(calculate('close', p), p, 'validation', min_assets=7)
    assert result['metrics']['ic_mean'] is None
    assert result['metrics']['valid_ic_days'] == 0
    p['adj_close'] = 10.0
    result = evaluate(calculate('close', p), p, 'validation')
    assert result['metrics']['rank_ic_mean'] is None


def test_cleaning_rejects_nonfinite_ohlc_and_conflicting_duplicates():
    p = panel().reset_index()
    p.loc[0, 'high'] = np.inf
    p.loc[1, 'volume'] = 0
    p.loc[2, 'low'] = p.loc[2, 'close'] + 1
    conflict = p.iloc[[3]].copy()
    conflict['close'] += 0.2
    clean, report = clean_panel(pd.concat([p, conflict]))
    assert report['invalid_removed'] == 3
    assert report['conflicting_duplicate_rows'] == 2
    assert len(clean) == 68


def test_explicit_calendar_keeps_an_entirely_missing_session():
    p = panel()
    calendar = p.index.get_level_values('datetime').unique()
    p = p.drop(pd.Timestamp('2022-01-05'), level='datetime')
    y = forward_labels(p, 1, calendar=calendar)
    assert np.isnan(y.loc[(pd.Timestamp('2022-01-03'), 'S0')])


def test_daily_coverage_uses_only_purged_label_eligible_rows():
    p = panel(days=10)
    f = calculate('close', p)
    f.loc[pd.IndexSlice[:, 'S0']] = np.nan
    result = evaluate(f, p, 'validation', start='2022-01-03', end='2022-01-07')
    assert result['metrics']['coverage'] == pytest.approx(5 / 6)
    assert result['eligible_rows'] == 18
    assert len(result['daily']) == 3


def test_baseline_archive_is_honest_strict_json_and_does_not_evaluate_test(tmp_path):
    import json
    from thesis.baselines import build_baselines
    p = panel(days=90)
    data = tmp_path / 'runtime' / 'data'
    data.mkdir(parents=True)
    p.to_parquet(data / 'panel.parquet')
    days = p.index.get_level_values('datetime').unique()
    splits = {'train': [str(days[0].date()), str(days[49].date())],
              'validation': [str(days[50].date()), str(days[74].date())],
              'test': [str(days[75].date()), str(days[-1].date())]}
    (data / 'manifest.json').write_text(json.dumps({'id': 'fixture', 'source': 'test fixture', 'splits': splits, 'quality': {}}))
    archive = build_baselines(tmp_path)
    assert archive['test_evaluated'] is False
    assert archive['factors'] and all(f['origin'] == 'baseline' for f in archive['factors'])
    assert all(f['segment'] == 'validation' for f in archive['factors'])
    assert all(day['date'] <= splits['validation'][1] for f in archive['factors'] for day in f['daily'])
    assert all('test_metrics' not in f for f in archive['factors'])
    assert all('group' in q and 'mean_return' in q for f in archive['factors'] for q in f['quantiles'])
    json.dumps(archive, allow_nan=False)
    stored = json.loads((tmp_path / 'runtime' / 'artifacts' / 'baselines.json').read_text())
    assert stored['dataset']['splits'] == splits


def test_quantiles_ties_stay_together_and_aggregate_daily_means():
    p = panel(days=10, stocks=10)
    f = pd.Series(1.0, index=p.index)
    result = evaluate(f, p, 'validation')
    assert result['quantiles'] == []
    assert result['metrics']['coverage'] == 1
    assert result['metrics']['ic_mean'] is None
    f = calculate('cs_rank(close)', p)
    result = evaluate(f, p, 'validation')
    assert [q['group'] for q in result['quantiles']] == [1, 2, 3, 4, 5]
    first = result['daily_quantiles'][0]['groups'][0]
    assert first['mean_return'] == pytest.approx(((20.2 / 20.1 - 1) + (21.4 / 21.2 - 1)) / 2)
    assert result['quantiles'][0]['mean_return'] == pytest.approx(np.mean([x['groups'][0]['mean_return'] for x in result['daily_quantiles']]))


def test_ic_ir_uses_sample_standard_deviation_without_annualization():
    p = panel(days=5)
    f = calculate('close', p)
    f.loc[f.index.get_level_values('datetime') == pd.Timestamp('2022-01-04')] *= -1
    result = evaluate(f, p, 'validation')
    assert result['metrics']['rank_ic_mean'] == pytest.approx(1 / 3)
    assert result['metrics']['rank_ic_ir'] == pytest.approx((1 / 3) / np.std([1, -1, 1], ddof=1))


def test_baseline_result_does_not_depend_on_frozen_test_prices(tmp_path):
    import json
    from thesis.baselines import build_baselines
    p = panel(days=90)
    data = tmp_path / 'runtime' / 'data'
    data.mkdir(parents=True)
    days = p.index.get_level_values('datetime').unique()
    splits = {'train': [str(days[0].date()), str(days[49].date())],
              'validation': [str(days[50].date()), str(days[74].date())],
              'test': [str(days[75].date()), str(days[-1].date())]}
    (data / 'manifest.json').write_text(json.dumps({'id': 'fixture', 'source': 'test fixture', 'splits': splits, 'quality': {}}))
    p.to_parquet(data / 'panel.parquet')
    before = build_baselines(tmp_path)
    p.loc[pd.IndexSlice[days[75]:, :], 'adj_close'] = np.nan
    p.loc[pd.IndexSlice[days[75]:, :], 'close'] = np.nan
    p.to_parquet(data / 'panel.parquet')
    after = build_baselines(tmp_path)
    assert before['factors'] == after['factors']


def test_baselines_reject_overlapping_splits(tmp_path):
    import json
    from thesis.baselines import build_baselines
    data = tmp_path / 'runtime' / 'data'
    data.mkdir(parents=True)
    panel(days=20).to_parquet(data / 'panel.parquet')
    (data / 'manifest.json').write_text(json.dumps({'splits': {
        'train': ['2022-01-03','2022-01-10'], 'validation': ['2022-01-10','2022-01-20'], 'test': ['2022-01-21','2022-01-28']}}))
    with pytest.raises(ValueError, match='split'):
        build_baselines(tmp_path)


def test_cleaning_does_not_silently_normalize_intraday_timestamp():
    p = panel().reset_index()
    p.loc[0, 'datetime'] = pd.Timestamp('2022-01-03 09:30')
    clean, report = clean_panel(p)
    assert report['invalid_removed'] == 1
    assert pd.Timestamp('2022-01-03 09:30') not in clean.index.get_level_values('datetime')


def test_rolling_correlation_uses_only_prior_sessions_and_respects_missing_values():
    p = panel(days=8)
    p['volume'] = p['close'] * 10
    corr = calculate('corr(close, volume, 3)', p)
    assert corr.loc[(pd.Timestamp('2022-01-05'), 'S1')] == pytest.approx(1)
    p = p.drop((pd.Timestamp('2022-01-04'), 'S1'))
    corr = calculate('corr(close, volume, 3)', p)
    assert np.isnan(corr.loc[(pd.Timestamp('2022-01-05'), 'S1')])
