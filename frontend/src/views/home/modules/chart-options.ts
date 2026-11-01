import type { ECOption } from '@/hooks/common/echarts';
import type { ResearchFactor } from '@/service/api/research';

const grid = { left: 46, right: 18, top: 20, bottom: 30 };
const axis = { axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#8693a9', fontSize: 10 } };

export function icOption(factor: ResearchFactor | undefined, metric: 'ic' | 'rank_ic'): ECOption {
  return {
    animationDuration: 400,
    textStyle: { fontFamily: 'Inter, system-ui, sans-serif' },
    grid,
    tooltip: { trigger: 'axis', renderMode: 'richText', backgroundColor: '#1b2638', borderColor: '#344359', textStyle: { color: '#e3eaf5' }, valueFormatter: value => typeof value === 'number' ? value.toFixed(4) : '—' },
    xAxis: { type: 'category', boundaryGap: false, data: factor?.daily.map(d => d.date) || [], ...axis, axisLabel: { ...axis.axisLabel, formatter: (value: string) => value.slice(5), hideOverlap: true } },
    yAxis: { type: 'value', ...axis, splitLine: { lineStyle: { color: '#273143', type: 'dashed' } }, axisLabel: { ...axis.axisLabel, formatter: (v: number) => v.toFixed(2) } },
    series: [{ name: metric === 'ic' ? '日 IC' : '日 Rank IC', type: 'line', data: factor?.daily.map(d => d[metric]) || [], symbol: 'none', connectNulls: false, lineStyle: { color: '#b5eb82', width: 1.6 }, itemStyle: { color: '#b5eb82' }, areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#a6db673b' }, { offset: 1, color: '#a6db6700' }] } } }]
  };
}

export function quantileOption(factor: ResearchFactor | undefined): ECOption {
  return {
    grid: { ...grid, left: 48, right: 12, top: 20, bottom: 26 },
    tooltip: { trigger: 'axis', renderMode: 'richText', valueFormatter: value => typeof value === 'number' ? `${(value * 100).toFixed(3)}%` : '—' },
    xAxis: { type: 'category', data: factor?.quantiles.map(q => `Q${q.group}`) || [], ...axis },
    yAxis: { type: 'value', ...axis, splitLine: { lineStyle: { color: '#273143', type: 'dashed' } }, axisLabel: { ...axis.axisLabel, formatter: (v: number) => `${(v * 100).toFixed(1)}%` } },
    series: [{ type: 'bar', barWidth: '43%', data: factor?.quantiles.map((q, i) => ({ value: q.mean_return, itemStyle: { color: ['#44566f', '#586c84', '#759779', '#93c27b', '#b5eb82'][i % 5], borderRadius: [4, 4, 0, 0] } })) || [] }]
  };
}
