<script setup lang="ts">
import { computed, ref } from 'vue';
import { useResearchStore } from '@/store/modules/research';
import ResearchChart from './research-chart.vue';
import ResearchIcon from './research-icon.vue';
import EvidencePanel from './evidence-panel.vue';
import { icOption, quantileOption } from './chart-options';
defineProps<{ view: string }>();
defineEmits<{ create: []; inspectRun: [id: string]; export: [] }>();
const store = useResearchStore();
const search = ref('');
const filtered = computed(() => store.factors.filter(f => (f.name + f.expression).toLowerCase().includes(search.value.toLowerCase())));
const selected = computed(() => store.selected);
const num = (n: number | null | undefined) => n == null ? '—' : n.toFixed(4);
const statuses: Record<string, string> = { queued: '排队中', running: '研究中', complete: '已完成', completed: '已完成', failed: '失败', interrupted: '已中断' };
const qualityEntries = computed(() => Object.entries(store.archive?.dataset.quality || {}));
const qualitySummary = computed(() => qualityEntries.value.filter(([key]) =>
  ['input_rows', 'valid_rows', 'duplicates_removed', 'invalid_removed', 'missing_instrument_sessions', 'volume_unit'].includes(key)
));
const adjustmentTitle = computed(() => store.archive?.dataset.adjustment?.includes('qfq') ? '前复权快照 · qfq' : '复权口径见归档');
const qualityNames: Record<string, string> = {
  input_rows:'原始记录数', valid_rows:'有效记录数', output_rows:'输出记录数', duplicates_removed:'移除重复记录',
  duplicate_rows_excluded:'排除重复键记录', conflicting_duplicate_rows:'冲突重复记录', invalid_removed:'移除无效记录',
  zero_volume_rows:'零成交量记录', missing_values_filled:'填补缺失值记录', duplicate_policy:'重复记录处理策略',
  volume_unit:'成交量单位', volume_converted:'是否换算成交量', survivorship_bias:'固定股票池偏差',
  universe_policy:'股票池构成规则', raw_prices_available:'是否保留原始价格', adjusted_ohlc:'是否使用复权 OHLC',
  adjustment_warning:'复权数据提示', calendar_source:'交易日历来源', calendar_covers_observed_dates:'日历覆盖观测日期',
  calendar_mismatch_rows_removed:'因日历不匹配移除记录', calendar_sessions:'交易日数', missing_instrument_sessions:'缺失股票交易日记录',
  requested_instruments:'请求股票数', successful_instruments:'成功获取股票数', rows_by_instrument:'各股票记录数',
  single_range_per_instrument:'每只股票是否单一时间段', anchor_splicing:'是否拼接锚点数据', provider_bar_limit:'数据源单次条数上限',
  requested_range_truncated_by_cap:'请求区间是否被上限截断', final_test_evaluated:'是否评价最终测试集',
  rows:'有效记录数', instruments:'股票数量', calendar_days:'交易日数', missing_rows:'缺失记录'
};
const protocolNames: Record<string, string> = {
  id:'协议版本', label:'评价协议', signal:'信号时点', entry:'进入时点', exit:'退出时点', horizon:'持有期',
  return:'收益计算', calendar:'交易日历', purge:'分段清洗规则', ic:'IC口径', rank_ic:'Rank IC口径',
  ir:'IR口径', icir:'ICIR口径', min_assets:'最小截面股票数', quantiles:'分层方式', coverage:'覆盖率口径',
  costs:'交易费用', costs_included:'是否计入交易费用', tradability_checked:'是否检查可交易性', test:'最终测试'
};
function displayValue(value: unknown) {
  if (value === null) return '—';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (value === 'lots (100 shares)') return '手（100 股 / 手）';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}
</script>
<template>
  <template v-if="view === 'data'">
    <section class="panel data-hero">
      <div><span class="eyebrow">MARKET DATA ASSET</span><h2>让研究建立在可信的数据之上。</h2><p>原始行情、清洗质量与数据指纹，共同构成每一次实验的起点。</p><div class="tag-row"><span class="soft-badge">真实公开行情</span><span class="small-tag" :title="store.archive?.dataset.adjustment">{{ adjustmentTitle }}</span><span class="small-tag">日频 · A 股</span></div></div>
      <div class="data-count">{{ store.archive?.dataset.rows.toLocaleString() || '—' }}<span>条有效行情记录</span></div>
    </section>
    <EvidencePanel mode="data" />
    <div class="two-column">
      <section class="panel">
        <div class="panel-head"><div><span class="eyebrow">PROVENANCE & QUALITY</span><h2>来源与质量</h2></div><ResearchIcon name="data" /></div>
        <dl class="definition-list"><div><dt>数据来源</dt><dd>{{ store.archive?.dataset.source || '未接入' }}</dd></div><div><dt>时间范围</dt><dd>{{ store.archive ? store.archive.dataset.start + ' → ' + store.archive.dataset.end : '—' }}</dd></div><div><dt>股票数量</dt><dd>{{ store.archive?.dataset.instruments ?? '—' }}</dd></div><div><dt>SHA-256 数据指纹</dt><dd class="hash">{{ store.archive?.dataset.sha256 || '—' }}</dd></div></dl>
        <div class="quality-list"><div v-for="[key, value] in qualitySummary" :key="key"><span>{{ qualityNames[key] || key }}</span><b>{{ displayValue(value) }}</b></div></div>
        <details class="quality-details"><summary>完整清洗记录 · {{ qualityEntries.length }} 项</summary><div class="quality-list"><div v-for="[key, value] in qualityEntries" :key="key"><span>{{ qualityNames[key] || key }}</span><b>{{ displayValue(value) }}</b></div></div></details>
      </section>
      <section class="panel">
        <div class="panel-head"><div><span class="eyebrow">TIME SEPARATION</span><h2>实验时间隔离</h2></div><ResearchIcon name="lock" /></div>
        <div class="split-list"><div v-for="(dates, key, i) in store.archive?.dataset.splits || {}" :key="key" :class="{ frozen: key === 'test' }"><span class="split-number">0{{ Number(i) + 1 }}</span><div><h3>{{ key === 'train' ? '训练区间' : key === 'validation' ? '验证区间' : '最终测试区间' }}</h3><p>{{ dates.join(' — ') }}</p><small>{{ key === 'test' ? '候选冻结后独立执行，未参与反馈' : key === 'validation' ? '候选评价与反馈使用此区间' : '经典定义与研究准备' }}</small></div><ResearchIcon :name="key === 'test' ? 'lock' : 'check'" :size="17" /></div></div>
        <div class="notice"><ResearchIcon name="info" :size="17" /><p>当前使用固定股票池，存在幸存者偏差。结果用于课题方法验证，不宣称无偏复现历史指数成分股。</p></div>
      </section>
    </div>
  </template>
  <template v-else-if="view === 'factors'">
    <div class="factor-workspace">
      <section class="panel factor-list-panel"><div class="panel-head"><div><span class="eyebrow">FACTOR EXPLORER</span><h2>因子列表</h2></div><span class="soft-badge">{{ store.factors.length }}</span></div>
        <label class="search-field"><ResearchIcon name="search" :size="17" /><input v-model="search" placeholder="搜索因子或表达式" aria-label="搜索因子" /></label>
        <div class="factor-list"><button v-for="factor in filtered" :key="factor.id" :class="{ selected: selected?.id === factor.id }" @click="store.selectedId = factor.id"><span class="factor-list-icon"><ResearchIcon name="factor" /></span><span><strong>{{ factor.name }}</strong><small>{{ factor.expression }}</small></span><ResearchIcon name="chevron" :size="16" /></button><p v-if="!filtered.length" class="inline-empty">没有匹配的因子</p></div>
        <div class="list-footnote"><span class="legend-dot" />经典基线 · 统一验证协议</div>
      </section>
      <section class="panel factor-detail"><template v-if="selected">
        <div class="panel-head"><div><span class="eyebrow">CLASSICAL BASELINE · VALIDATION</span><h2>{{ selected.name }}</h2></div><span class="soft-badge">经典基线</span></div><p class="description">{{ selected.description }}</p>
        <div class="expression-block"><span>因子表达式</span><code>{{ selected.expression }}</code></div>
        <div class="detail-metrics"><div><span>IC 均值</span><b>{{ num(selected.metrics.ic_mean) }}</b></div><div><span>Rank IC</span><b class="accent">{{ num(selected.metrics.rank_ic_mean) }}</b></div><div><span>ICIR</span><b>{{ num(selected.metrics.ic_ir) }}</b></div><div><span>有效 IC 日</span><b>{{ selected.metrics.valid_ic_days }}</b></div></div>
        <div class="detail-charts"><div><h3>每日 Rank IC</h3><div class="detail-chart"><ResearchChart :option="icOption(selected, 'rank_ic')" /></div></div><div><h3>五分层持有期收益</h3><div class="detail-chart"><ResearchChart :option="quantileOption(selected)" /></div></div></div>
        <div class="code-block"><div><ResearchIcon name="terminal" :size="16" /><span>Python 计算入口</span><span>受控表达式</span></div><pre>from thesis.factors import calculate
factor = calculate({{ JSON.stringify(selected.expression) }}, panel)</pre></div>
        <div class="notice"><ResearchIcon name="info" :size="17" /><p>ICIR 使用日 IC 的样本标准差。分层收益为统计性评价，尚未计入交易费用、涨跌停和可交易性约束。</p></div>
      </template><div v-else class="honest-empty"><ResearchIcon name="factor" :size="34" /><h3>等待因子评价归档</h3><p>完成经典基线计算后，即可探索指标和表达式。</p></div></section>
    </div>
  </template>
  <template v-else-if="view === 'feedback'">
    <section class="panel evolution-hero"><div><span class="eyebrow">RD-AGENT · RESEARCH EVOLUTION</span><h2>让评价结果，成为下一次研究的起点。</h2><p>研究假设 → 因子表达式 → 受控计算 → 验证集评价 → 反馈改进</p><button class="primary-button" @click="$emit('create')"><ResearchIcon name="plus" :size="16" />创建研究任务</button></div><div class="evolution-symbol" aria-hidden="true">α<span>↗</span></div></section>
    <div class="evolution-flow"><div v-for="(step, i) in ['提出假设', '生成候选', '因果校验', '真实评价', '反馈改进']" :key="step"><span>0{{ i + 1 }}</span><ResearchIcon :name="['model','factor','lock','overview','feedback'][i]" :size="25" /><strong>{{ step }}</strong><ResearchIcon v-if="i < 4" name="arrow" :size="15" /></div></div>
    <div class="two-column"><section class="panel"><div class="panel-head"><div><span class="eyebrow">MODEL CONNECTION</span><h2>研究引擎状态</h2></div><span class="state-pill" :class="{ connected: store.overview?.environment.llm_configured }"><i />{{ store.overview?.environment.llm_configured ? '模型已配置' : '模型待配置' }}</span></div><dl class="definition-list"><div><dt>RD-Agent 源码</dt><dd>{{ store.overview?.environment.rdagent_available ? '固定版本接线可用' : '接线待验证' }}</dd></div><div><dt>生成模型</dt><dd>{{ store.overview?.environment.model || '尚未配置' }}</dd></div><div><dt>反馈区间</dt><dd>仅使用验证集</dd></div><div><dt>最终测试</dt><dd>候选冻结后单独运行</dd></div></dl><div class="notice"><ResearchIcon name="info" :size="17" /><p>真实调用完成后，才会记录生成结果、调用用量和父子因子对照。配置只在服务端管理。</p></div></section><section class="panel"><div class="panel-head"><div><span class="eyebrow">COMPARISON EVIDENCE</span><h2>反馈前后对照</h2></div><ResearchIcon name="feedback" /></div><div class="honest-empty"><div class="empty-emblem"><ResearchIcon name="feedback" :size="32" /></div><h3>等待一轮真实反馈实验</h3><p>模型接入后，将呈现父子因子的同口径评价、修改理由和调用记录。</p><span class="small-tag">真实调用完成后形成研究证据</span></div></section></div>
  </template>
  <template v-else-if="view === 'runs'">
    <section class="panel"><div class="panel-head"><div><span class="eyebrow">EXPERIMENT ARCHIVE</span><h2>每一次研究，都有迹可循。</h2></div><button class="primary-button" @click="$emit('create')"><ResearchIcon name="plus" :size="16" />新建研究</button></div><p class="description">独立记录任务状态、事件、评价协议与产物，刷新页面后仍可查看。</p>
      <div class="table-scroll"><table class="research-table"><thead><tr><th>实验 ID / 研究目标</th><th>运行模式</th><th>状态</th><th>创建时间</th><th /></tr></thead><tbody><tr v-for="run in store.overview?.runs || []" :key="run.id" tabindex="0" @click="$emit('inspectRun', run.id)" @keydown.enter="$emit('inspectRun', run.id)"><td><strong>{{ run.prompt || '经典因子复算' }}</strong><code>{{ run.id }}</code></td><td>{{ run.mode === 'llm' ? '真实 LLM' : '经典基线' }}</td><td><span class="run-status" :class="run.status">{{ statuses[run.status] || run.status }}</span></td><td>{{ new Date(run.created_at).toLocaleString('zh-CN') }}</td><td><ResearchIcon name="chevron" :size="16" /></td></tr></tbody></table>
      <div v-if="!store.overview?.runs.length" class="honest-empty"><div class="empty-emblem"><ResearchIcon name="archive" :size="32" /></div><h3>从第一份可追溯记录开始</h3><p>初始经典基线在总控中展示，新创建的研究任务将持续记录在这里。</p><button class="secondary-button" @click="$emit('create')">创建研究<ResearchIcon name="arrow" :size="16" /></button></div></div>
    </section>
  </template>
  <template v-else-if="view === 'reports'">
    <section class="panel report-hero"><div><span class="eyebrow">REPRODUCIBLE RESEARCH</span><h2>从系统展示，到论文证据。</h2><p>保存真实数据与评价结果，保留来源、表达式、日期区间和协议。</p></div><button class="primary-button" :disabled="!store.overview" @click="$emit('export')"><ResearchIcon name="download" :size="17" />导出实验材料</button></section>
    <div class="evidence-grid"><section class="panel" v-for="item in [{ icon:'data',title:'数据证据',desc:'原始快照、清洗报告、数据指纹',state:store.archive ? '已归档' : '等待数据' },{icon:'factor',title:'经典评价',desc:'日 IC、Rank IC、ICIR、分层收益',state:store.factors.length ? '已归档' : '等待评价'},{icon:'feedback',title:'反馈实验',desc:'模型响应、父子结果、调用用量',state:'待真实调用'},{icon:'lock',title:'独立测试',desc:'冻结候选、单次测试、隔离验证',state:'待候选冻结'}]" :key="item.title"><div class="evidence-icon"><ResearchIcon :name="item.icon" :size="23" /></div><h3>{{ item.title }}</h3><p>{{ item.desc }}</p><span class="small-tag">{{ item.state }}</span></section></div>
    <EvidencePanel mode="corpus" />
    <section class="panel" style="margin-top:20px"><div class="panel-head"><div><span class="eyebrow">EXPERIMENT PROTOCOL</span><h2>当前评价协议</h2></div><span class="soft-badge">{{ store.archive?.protocol.id || '待生成' }}</span></div><dl class="definition-list"><div v-for="(value, key) in store.archive?.protocol || {}" :key="key"><dt>{{ protocolNames[String(key)] || key }}</dt><dd>{{ displayValue(value) }}</dd></div></dl></section>
  </template>
</template>
