<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { useEventListener, useFullscreen, useIntervalFn, useNow } from '@vueuse/core';
import { NButton, NDrawer, NDrawerContent, NInput, NModal, NSelect, useMessage } from 'naive-ui';
import { downloadReport, fetchRun, submitResearch } from '@/service/api/research';
import type { ResearchRun } from '@/service/api/research';
import { useResearchStore } from '@/store/modules/research';
import { useThemeStore } from '@/store/modules/theme';
import OverviewPanel from './modules/overview-panel.vue';
import DetailPanels from './modules/detail-panels.vue';
import ResearchIcon from './modules/research-icon.vue';
import './research.css';
const store = useResearchStore();
const route = useRoute();
const router = useRouter();
const message = useMessage();
const appRef = ref<HTMLElement>();
const { isFullscreen, toggle } = useFullscreen(appRef);
const now = useNow();
const navigation = [
  { id: 'overview', icon: 'overview', label: '研究总控', en: 'RESEARCH OVERVIEW' },
  { id: 'data', icon: 'data', label: '数据资产', en: 'MARKET DATA' },
  { id: 'factors', icon: 'factor', label: '因子实验室', en: 'FACTOR EXPLORER' },
  { id: 'feedback', icon: 'feedback', label: '研究进化', en: 'RESEARCH EVOLUTION' },
  { id: 'runs', icon: 'archive', label: '实验档案', en: 'EXPERIMENT ARCHIVE' },
  { id: 'reports', icon: 'paper', label: '论文材料', en: 'RESEARCH EVIDENCE' }
];
const active = computed(() => navigation.find(n => n.id === route.query.view) || navigation[0]);
const displayMode = computed(() => isFullscreen.value || route.query.display === '1');
function navigate(view: string) { router.push({ path: '/home', query: { ...route.query, view } }); }
function inspect(id: string) { store.selectedId = id; navigate('factors'); }
watch(() => route.query.factor, id => { if (typeof id === 'string') store.selectedId = id; }, { immediate: true });
watch(() => store.factors, factors => {
  if (factors.length && !factors.some(f => f.id === store.selectedId)) store.selectedId = factors[0].id;
});
watch(() => store.selectedId, id => {
  if (id && route.query.factor !== id) router.replace({ query: { ...route.query, factor: id } });
});
const creation = ref(false);
const mode = ref('llm');
const prompt = ref('研究短期价格反转与成交量变化的关系，构建具有经济含义的量价因子。');
const iterations = ref(1);
const submitting = ref(false);
const requestKey = ref('');
const submitError = ref('');
watch([prompt, mode, iterations], () => { requestKey.value = ''; submitError.value = ''; });
const canLLM = computed(() => Boolean(store.overview?.environment.llm_configured && store.overview?.environment.rdagent_available));
const submitDisabled = computed(() => !store.overview || Boolean(store.error) || !store.overview.environment.worker_running || (mode.value === 'llm' && !canLLM.value) || !prompt.value.trim());
async function submit() {
  if (submitDisabled.value || submitting.value) return;
  submitting.value = true; submitError.value = '';
  requestKey.value ||= crypto.randomUUID();
  try {
    const run = await submitResearch({ prompt: prompt.value, mode: mode.value, iterations: iterations.value }, requestKey.value);
    creation.value = false; requestKey.value = '';
    await store.refresh(); navigate('runs'); await inspectRun(run.id);
    message.success('研究任务已进入队列');
  } catch (error) { submitError.value = error instanceof Error ? error.message : '任务提交失败'; }
  finally { submitting.value = false; }
}
const runDrawer = ref(false);
const selectedRun = ref<ResearchRun | null>(null);
const runError = ref('');
const runLoading = ref(false);
async function inspectRun(id: string) {
  selectedRun.value = null; runError.value = ''; runLoading.value = true; runDrawer.value = true;
  try { selectedRun.value = await fetchRun(id); }
  catch (error) { runError.value = error instanceof Error ? error.message : '读取实验失败'; }
  finally { runLoading.value = false; }
}
const statuses: Record<string,string> = {queued:'排队中',running:'研究中',complete:'已完成',completed:'已完成',failed:'失败',interrupted:'已中断'};
const settings = ref(false);
const searchOpen = ref(false);
const searchText = ref('');
const searchFactors = computed(() => store.factors.filter(f => (f.name + f.expression).toLowerCase().includes(searchText.value.toLowerCase())));
const searchRuns = computed(() => (store.overview?.runs || []).filter(r => (r.prompt + r.id).toLowerCase().includes(searchText.value.toLowerCase())));
useEventListener(document, 'keydown', event => {
  const element = event.target as HTMLElement | null;
  if (event.key === '/' && !element?.closest('input,textarea,[contenteditable=true]') && !creation.value) {
    event.preventDefault(); searchOpen.value = true;
  }
});
async function exportReport() {
  try { await downloadReport(); message.success('实验材料已导出'); }
  catch (error) { message.error(error instanceof Error ? error.message : '导出失败'); }
}
async function toggleDisplay() {
  if (route.query.display === '1') { const query = { ...route.query }; delete query.display; await router.replace({ query }); }
  else { try { await toggle(); } catch { await router.replace({ query: { ...route.query, display: '1' } }); } }
}
useIntervalFn(async () => {
  if (store.overview?.runs.some(r => ['running', 'queued'].includes(r.status))) {
    await store.refresh();
    if (selectedRun.value && runDrawer.value) {
      try { selectedRun.value = await fetchRun(selectedRun.value.id); } catch { /* Overview exposes connection errors. */ }
    }
  }
}, 3000);
onMounted(() => { useThemeStore().setThemeScheme('dark'); store.refresh(); });
</script>
<template>
  <div ref="appRef" class="research-app" :class="{ 'display-mode': displayMode }">
    <aside class="app-sidebar">
      <a class="brand-mark" href="/home" aria-label="AlphaLab 首页"><svg width="31" height="31" viewBox="0 0 32 32" fill="none"><path d="M4 25L14 6h4l10 19h-6l-6-12-6 12H4Z" fill="currentColor" /><path d="M11 21h13" stroke="#101722" stroke-width="3" /></svg></a>
      <div class="sidebar-divider" />
      <nav aria-label="研究导航"><button v-for="item in navigation" :key="item.id" :class="{ active: active.id === item.id }" :aria-label="item.label" :title="item.label" :aria-current="active.id === item.id ? 'page' : undefined" @click="navigate(item.id)"><ResearchIcon :name="item.icon" :size="21" /><span>{{ item.label }}</span></button></nav>
      <div class="sidebar-bottom"><button aria-label="运行环境" title="运行环境" @click="settings = true"><ResearchIcon name="settings" /></button><span class="avatar">CG</span></div>
    </aside>
    <div class="workspace">
      <header class="app-header">
        <div class="brand-title">ALPHA<span>LAB</span><span class="brand-separator" /><span class="workspace-name">智能量化因子研究平台</span><span class="version-tag">v1.2</span></div>
        <div class="header-actions"><button class="global-search" @click="searchOpen = true"><ResearchIcon name="search" :size="15" /><span>搜索因子、实验</span><kbd>/</kbd></button><span class="state-pill" :class="{ connected: store.overview && !store.error }"><i />{{ store.error ? '服务未连接' : store.loading && !store.overview ? '连接中' : store.overview ? '研究服务在线' : '等待连接' }}</span><button class="icon-button" aria-label="刷新研究数据" :disabled="store.loading" @click="store.refresh()"><ResearchIcon name="refresh" :size="18" /></button><button class="icon-button" :aria-label="displayMode ? '退出全屏展示' : '全屏展示'" @click="toggleDisplay"><ResearchIcon name="expand" :size="18" /></button><span class="header-clock">{{ now.toLocaleTimeString('zh-CN', { hour12: false, timeZone:'Asia/Shanghai' }) }}<small>CST</small></span></div>
      </header>
      <main class="main-content">
        <div class="page-heading"><div><div class="eyebrow"><span class="live-square" />{{ active.en }}</div><h1>{{ active.label }}<span v-if="active.id === 'overview'" class="heading-period">.</span></h1><p>{{ active.id === 'overview' ? '从研究假设到可复现实验，掌握每一步。' : '真实数据、清晰方法与可追溯的研究过程。' }}</p></div><div class="page-actions"><span class="archive-badge"><ResearchIcon name="clock" :size="14" />{{ store.archive ? '真实行情 · 经典归档' : '实验归档待生成' }}</span><button class="primary-button" @click="creation = true"><ResearchIcon name="plus" :size="17" />新建研究</button></div></div>
        <div v-if="store.error" class="error-banner" role="alert"><ResearchIcon name="info" :size="18" /><div><strong>研究服务暂时无法连接</strong><p>{{ store.error }}。{{ store.overview ? '以下保留上次读取的归档，状态可能已变化。' : '请启动本地研究服务后重新连接。' }}</p></div><button class="secondary-button" @click="store.refresh()">重新连接</button></div>
        <div v-if="store.loading && !store.overview" class="loading-banner" role="status">正在读取研究归档与服务状态…</div>
        <OverviewPanel v-if="active.id === 'overview'" @navigate="navigate" @inspect="inspect" />
        <DetailPanels v-else :view="active.id" @create="creation = true" @inspect-run="inspectRun" @export="exportReport" />
        <footer class="workspace-footer"><span>ALPHALAB <i /> 基于 RD-Agent 的智能因子研究</span><span>{{ store.lastUpdated ? '最后同步 ' + store.lastUpdated : '等待同步' }}<i />真实评价 · 结果可追溯</span></footer>
      </main>
    </div>
    <NModal v-model:show="creation" preset="card" title="新建因子研究" class="research-modal" :style="{ width: '600px', maxWidth: 'calc(100vw - 32px)' }" :mask-closable="!submitting">
      <div class="modal-content"><p>描述研究假设。候选将通过因果检查，并仅在验证集评价。</p>
        <label>运行模式<NSelect v-model:value="mode" :options="[{ label:'真实 LLM 研究',value:'llm' }, { label:'经典基线复算（不调用模型）', value:'baseline' }]" /></label>
        <div v-if="mode === 'llm' && !canLLM" class="notice"><ResearchIcon name="info" :size="18" /><p>真实研究引擎尚未就绪。请完成服务端模型配置与接线验证后刷新页面。经典基线复算可独立运行。</p></div>
        <label>研究目标<NInput v-model:value="prompt" type="textarea" placeholder="例如：研究短期反转与量能变化的关系…" :maxlength="2000" show-count :autosize="{ minRows: 4, maxRows: 7 }" /></label>
        <label v-if="mode === 'llm'">反馈轮数<NSelect v-model:value="iterations" :options="[{ label:'1 轮反馈',value:1 },{ label:'2 轮反馈',value:2 }]" /></label>
        <p class="muted">{{ mode === 'llm' ? '最终测试集保持隔离。调用上限由服务端预算控制。' : '按固定经典表达式复算归档，不会调用模型或生成新因子。' }}</p>
        <p v-if="!store.overview?.environment.worker_running" class="form-error">研究执行器尚未就绪，启动执行器后即可提交。</p><p v-if="submitError" class="form-error" role="alert">{{ submitError }}</p>
        <div class="modal-actions"><NButton :disabled="submitting" @click="creation = false">取消</NButton><NButton type="primary" :loading="submitting" :disabled="submitDisabled" @click="submit">创建研究任务</NButton></div>
      </div>
    </NModal>
    <NModal v-model:show="searchOpen" preset="card" title="搜索研究对象" :style="{ width:'600px',maxWidth:'calc(100vw - 32px)' }" class="research-modal">
      <NInput v-model:value="searchText" placeholder="因子名称、表达式、实验目标或 ID" autofocus />
      <div class="search-results"><span class="eyebrow">FACTORS · 因子</span><button v-for="factor in searchFactors" :key="factor.id" @click="inspect(factor.id); searchOpen = false"><ResearchIcon name="factor" /><span>{{ factor.name }}<small>{{ factor.expression }}</small></span><ResearchIcon name="arrow" :size="16" /></button><p v-if="!searchFactors.length">没有匹配的因子</p><span class="eyebrow">EXPERIMENTS · 实验</span><button v-for="run in searchRuns" :key="run.id" @click="navigate('runs'); inspectRun(run.id); searchOpen = false"><ResearchIcon name="archive" /><span>{{ run.prompt }}<small>{{ run.id }}</small></span><ResearchIcon name="arrow" :size="16" /></button><p v-if="!searchRuns.length">没有匹配的实验</p></div>
    </NModal>
    <NDrawer v-model:show="runDrawer" :width="600"><NDrawerContent title="实验运行详情" closable><div class="drawer-content"><p v-if="runLoading">正在读取实验…</p><p v-if="runError" role="alert">{{ runError }}</p><template v-if="selectedRun"><span class="small-tag">{{ statuses[selectedRun.status] || selectedRun.status }}</span><h2>{{ selectedRun.prompt }}</h2><code>{{ selectedRun.id }}</code><p v-if="selectedRun.error" class="form-error">{{ selectedRun.error }}</p><h3>运行事件</h3><div v-for="(event, i) in selectedRun.events || []" :key="i" class="event-item"><time>{{ new Date(event.time).toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai'}) }}</time><strong>{{ event.stage }}</strong><p>{{ event.message }}</p></div><details v-if="selectedRun.result"><summary>查看实验产物</summary><pre>{{ JSON.stringify(selectedRun.result, null, 2) }}</pre></details></template></div></NDrawerContent></NDrawer>
    <NDrawer v-model:show="settings" :width="440"><NDrawerContent title="运行环境" closable><div class="drawer-content"><span class="eyebrow">SYSTEM CONNECTION</span><h2>研究服务状态</h2><dl class="definition-list"><div><dt>产品版本</dt><dd>AlphaLab v1.2.0</dd></div><div><dt>前端架构</dt><dd>Vue 3 · SoybeanAdmin</dd></div><div><dt>后端服务</dt><dd>{{ store.overview && !store.error ? '已连接' : '未连接' }}</dd></div><div><dt>独立执行器</dt><dd>{{ store.overview?.environment.worker_running ? '运行中' : '未运行' }}</dd></div><div><dt>RD-Agent</dt><dd>{{ store.overview?.environment.rdagent_available ? '可用' : '待验证' }}</dd></div><div><dt>生成模型</dt><dd>{{ store.overview?.environment.model || '未配置' }}</dd></div></dl><p>凭据只由服务端读取，不会通过展示页面返回。</p><NButton @click="store.refresh()">重新检测</NButton></div></NDrawerContent></NDrawer>
  </div>
</template>
