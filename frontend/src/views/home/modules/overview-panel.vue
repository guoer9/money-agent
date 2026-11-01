<script setup lang="ts">
import { computed, ref } from 'vue';
import { useResearchStore } from '@/store/modules/research';
import ResearchIcon from './research-icon.vue';
import ResearchChart from './research-chart.vue';
import { icOption, quantileOption } from './chart-options';
defineEmits<{ navigate: [view: string]; inspect: [id: string] }>();
const store = useResearchStore();
const metric = ref<'ic' | 'rank_ic'>('ic');
const icChart = computed(() => icOption(store.selected, metric.value));
const quantileChart = computed(() => quantileOption(store.selected));
const number = (value: number | null | undefined, digits = 4) => value == null ? '—' : value.toFixed(digits);
const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const dataset = computed(() => store.archive?.dataset);
const stages = computed(() => [
  { title: '行情数据', desc: dataset.value ? `${dataset.value.instruments} 只股票 · 可追溯快照` : '等待数据接入', done: Boolean(dataset.value) },
  { title: '经典因子评价', desc: store.factors.length ? `${store.factors.length} 个因子 · 逐日截面评价` : '等待基线实验', done: Boolean(store.factors.length) },
  { title: 'LLM 反馈进化', desc: store.overview?.environment.llm_configured ? '模型已配置 · 等待研究任务' : '模型待配置', done: false },
  { title: '独立测试', desc: store.overview?.final_test.message || '候选冻结后执行', done: store.overview?.final_test.status === 'complete' }
]);
</script>
<template>
  <div class="metrics-grid">
    <article class="metric-card"><div class="metric-top"><span>有效行情记录</span><ResearchIcon name="data" /></div><div class="metric-value">{{ dataset?.rows.toLocaleString() || '—' }}<span>条</span></div><div class="metric-foot"><span class="status-dot" :class="{ ready: dataset }" />{{ dataset ? `${dataset.instruments} 只股票 · 真实日频行情` : '等待真实行情入库' }}</div><div class="metric-decoration bars"><i v-for="i in 12" :key="i" :style="{ height: `${12 + (i % 4) * 6}px` }" /></div></article>
    <article class="metric-card"><div class="metric-top"><span>经典因子</span><ResearchIcon name="factor" /></div><div class="metric-value">{{ store.factors.length || '—' }}<span>个</span></div><div class="metric-foot"><span class="tiny-tag">BASELINE</span>验证集同口径评价</div></article>
    <article class="metric-card highlight"><div class="metric-top"><span>当前因子 · Rank IC</span><ResearchIcon name="factor" /></div><div class="metric-value accent">{{ number(store.selected?.metrics.rank_ic_mean) }}</div><div class="metric-foot"><span v-if="store.selected" class="status-dot ready" />{{ store.selected?.name || '选择因子后查看' }}<span class="muted">日均值</span></div></article>
    <article class="metric-card"><div class="metric-top"><span>评价覆盖率</span><ResearchIcon name="check" /></div><div class="metric-value">{{ percent(store.selected?.metrics.coverage) }}</div><div class="metric-foot"><span class="tiny-tag">VALIDATION</span>{{ store.selected?.metrics.valid_ic_days ?? '—' }} 个有效 IC 日</div></article>
  </div>
  <div class="dashboard-grid">
    <section class="panel prediction-panel">
      <div class="panel-head"><div><span class="eyebrow">PREDICTIVE SIGNAL</span><h2>因子预测力 <span class="soft-badge">验证集</span></h2></div><div class="segmented"><button :class="{ active: metric === 'ic' }" @click="metric = 'ic'">IC</button><button :class="{ active: metric === 'rank_ic' }" @click="metric = 'rank_ic'">Rank IC</button></div></div>
      <div class="chart-summary"><span class="legend-dot" /><strong>{{ store.selected?.name || '尚无实验结果' }}</strong><span>IC 均值 <b>{{ number(store.selected?.metrics.ic_mean) }}</b></span><span>ICIR <b>{{ number(store.selected?.metrics.ic_ir, 3) }}</b></span><select v-model="store.selectedId" aria-label="选择展示因子" :disabled="!store.factors.length"><option v-if="!store.factors.length" value="">暂无因子</option><option v-for="factor in store.factors" :key="factor.id" :value="factor.id">{{ factor.name }}</option></select></div>
      <div class="main-chart"><ResearchChart :option="icChart" :empty="!store.selected" label="验证集每日截面相关系数" /></div>
      <div class="chart-caption"><span><ResearchIcon name="info" :size="13" /> 逐日截面相关系数，不代表策略收益</span><span>{{ dataset?.splits.validation?.join(' — ') || '未设置评价区间' }}</span></div>
    </section>
    <section class="panel workflow-panel"><div class="panel-head"><div><span class="eyebrow">RESEARCH PIPELINE</span><h2>研究进程</h2></div><span class="soft-badge">RD-Agent</span></div><ol class="pipeline"><li v-for="(stage, index) in stages" :key="stage.title" :class="{ done: stage.done, pending: !stage.done }"><span class="pipeline-node"><ResearchIcon v-if="stage.done" name="check" :size="14" /><span v-else>{{ String(index + 1).padStart(2, '0') }}</span></span><div><strong>{{ stage.title }}</strong><p>{{ stage.desc }}</p></div><span class="pipeline-label">{{ stage.done ? '已完成' : index === 3 ? '待冻结' : '待执行' }}</span></li></ol><button class="workflow-action" @click="$emit('navigate', 'feedback')"><ResearchIcon name="feedback" :size="16" />查看研究链路<ResearchIcon name="arrow" :size="16" /></button></section>
    <section class="panel factor-panel"><div class="panel-head"><div><span class="eyebrow">FACTOR LIBRARY</span><h2>经典因子观察</h2></div><button class="text-button" @click="$emit('navigate', 'factors')">全部因子<ResearchIcon name="arrow" :size="15" /></button></div><div class="table-scroll"><table class="research-table"><thead><tr><th>因子名称 / 表达式</th><th>IC 均值</th><th>Rank IC</th><th>ICIR</th><th>来源</th><th /></tr></thead><tbody><tr v-for="(factor, i) in store.factors" :key="factor.id" tabindex="0" @click="$emit('inspect', factor.id)" @keydown.enter="$emit('inspect', factor.id)"><td><div class="factor-name"><span class="factor-index">{{ String(i + 1).padStart(2, '0') }}</span><div><strong>{{ factor.name }}</strong><code>{{ factor.expression }}</code></div></div></td><td :class="{ accent: (factor.metrics.ic_mean || 0) > 0 }">{{ number(factor.metrics.ic_mean) }}</td><td>{{ number(factor.metrics.rank_ic_mean) }}</td><td>{{ number(factor.metrics.ic_ir, 3) }}</td><td><span class="small-tag">经典基线</span></td><td><ResearchIcon name="chevron" :size="15" /></td></tr></tbody></table><div v-if="!store.factors.length" class="inline-empty">尚无真实因子归档。数据和评价完成后自动呈现。</div></div></section>
    <section class="panel quantile-panel"><div class="panel-head"><div><span class="eyebrow">QUANTILE ANALYSIS</span><h2>五分层收益</h2></div><span class="mini-note">单期均值</span></div><div class="quantile-chart"><ResearchChart :option="quantileChart" :empty="!store.selected" label="每日分层持有期收益均值" /></div><div class="chart-caption"><span>Q1 低因子值 → Q5 高因子值</span><button class="text-button" @click="$emit('navigate', 'factors')">评价详情<ResearchIcon name="arrow" :size="13" /></button></div></section>
  </div>
  <div class="integrity-strip"><div class="integrity-icon"><ResearchIcon name="lock" :size="18" /></div><div><strong>每一项结论，都可以回到实验现场。</strong><span>数据快照 · 固定评价协议 · 运行日志 · 独立测试隔离</span></div><button class="text-button" @click="$emit('navigate', 'reports')">查看实验材料<ResearchIcon name="arrow" :size="15" /></button></div>
</template>
