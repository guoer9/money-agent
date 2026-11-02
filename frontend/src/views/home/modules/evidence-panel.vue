<script setup lang="ts">
import { computed, ref } from 'vue';
import { useResearchStore } from '@/store/modules/research';
import ResearchIcon from './research-icon.vue';

defineProps<{ mode: 'data' | 'corpus' }>();
const store = useResearchStore();
const evidence = computed(() => store.overview?.evidence);
const linked = computed(() => evidence.value?.archive_link?.status === 'matched');
const query = ref('');
const category = ref('全部主题');
const showAll = ref(false);
const categories = computed(() => ['全部主题', ...new Set(evidence.value?.corpus.notes.map(note => note.category) || [])]);
const filtered = computed(() => (evidence.value?.corpus.notes || []).filter(note =>
  (category.value === '全部主题' || note.category === category.value) &&
  `${note.title} ${note.path}`.toLowerCase().includes(query.value.trim().toLowerCase())
));
const visible = computed(() => showAll.value ? filtered.value : filtered.value.slice(0, 5));
const statusNames: Record<string, string> = {
  verified: '一致', mismatch: '不一致', missing: '文件缺失', unrecorded: '缺少原指纹',
  unavailable: '尚无归档', unreadable: '无法读取', incomplete: '存在缺口', invalid_path: '路径无效'
};
const auditState = computed(() => {
  if (evidence.value?.archive_link?.status === 'mismatch') return '评价归档与行情不一致';
  if (evidence.value?.dataset.status === 'verified') return linked.value ? '文件校验通过 · 已关联评价' : '文件一致 · 尚未关联评价';
  return statusNames[evidence.value?.dataset.status || 'unavailable'];
});
const localTime = (value: string | null | undefined) => value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '—';
</script>

<template>
  <section v-if="mode === 'data'" class="panel audit-panel">
    <div class="panel-head">
      <div><span class="eyebrow">EVIDENCE BEFORE CONCLUSIONS</span><h2>数据可信度</h2></div>
      <span class="audit-state" :class="{ verified: linked && evidence?.dataset.status === 'verified' }">
        <i />{{ auditState }}
      </span>
    </div>
    <div v-if="evidence?.archive_link?.status === 'mismatch'" class="corpus-gap"><ResearchIcon name="info" :size="17" /><p>当前采集的行情与页面展示的因子结果属于不同快照。完成对应基线复算后再使用二者的关联结论；本次文件检查仅针对当前采集目录。</p></div>
    <div class="audit-stats">
      <div><span>行情面板指纹</span><strong :class="{ accent: evidence?.dataset.panel.status === 'verified' }">{{ statusNames[evidence?.dataset.panel.status || 'missing'] }}</strong><small>与采集清单逐字节核对</small></div>
      <div><span>原始行情快照</span><strong>{{ evidence?.dataset.raw.verified ?? '—' }}<em>/ {{ evidence?.dataset.raw.total ?? '—' }}</em></strong><small>SHA-256 一致的响应文件</small></div>
      <div><span>历史时点数据 · PIT</span><strong class="audit-amber">尚未验证</strong><small>价格复权与股票池仍有边界</small></div>
      <div><span>已记录价格内容指纹</span><strong>{{ evidence?.dataset.normalized_fingerprints ?? '—' }}<em>份</em></strong><small>新增采集自动记录，旧快照未补写</small></div>
    </div>
    <div class="audit-boundaries">
      <article v-for="item in evidence?.boundaries || []" :key="item.title">
        <div><h3>{{ item.title }}</h3><span>{{ item.status }}</span></div><p>{{ item.detail }}</p>
      </article>
    </div>
    <details class="audit-proof"><summary>查看核验时间与清单指纹<ResearchIcon name="chevron" :size="14" /></summary>
      <dl class="definition-list"><div><dt>本次核验</dt><dd>{{ localTime(evidence?.checked_at) }}</dd></div><div><dt>行情采集</dt><dd>{{ localTime(evidence?.dataset.snapshot_time) }}</dd></div><div><dt>被检查的数据集</dt><dd>{{ evidence?.archive_link.checked_dataset_id || '—' }}</dd></div><div><dt>展示评价的数据集</dt><dd>{{ evidence?.archive_link.displayed_dataset_id || '—' }}</dd></div><div><dt>Manifest SHA-256</dt><dd class="hash">{{ evidence?.dataset.manifest_sha256 || '—' }}</dd></div></dl>
    </details>
  </section>

  <section v-else class="panel corpus-panel">
    <div class="panel-head"><div><span class="eyebrow">RESEARCH CORPUS · TRACEABLE SOURCES</span><h2>研究语料与证据</h2></div><ResearchIcon name="archive" :size="21" /></div>
    <p class="corpus-intro">让调研积累进入作品展示。笔记、原始文件与实现验证分别记录，每份资料均可回到原文。</p>
    <div class="corpus-stats">
      <div><strong>{{ evidence?.corpus.note_count ?? '—' }}</strong><span>份实质研究笔记</span></div>
      <div><strong>{{ evidence?.corpus.temporary_files.referenced ?? '—' }}</strong><span>个可解析的临时文件引用</span></div>
      <div><strong :class="{ 'audit-amber': evidence?.corpus.temporary_files.missing }">{{ evidence?.corpus.temporary_files.available ?? '—' }}</strong><span>个临时文件当前可访问</span></div>
    </div>
    <div v-if="evidence?.corpus.temporary_files.missing" class="corpus-gap"><ResearchIcon name="info" :size="17" /><p><strong>{{ evidence.corpus.temporary_files.missing }} 个临时文件当前未找到。</strong>笔记仍可阅读；这些引用需要补入持久证据库后，再复核原件内容与哈希。文件可访问也不代表原文已验证。</p></div>
    <div v-if="evidence?.corpus.note_count" class="corpus-tools">
      <label class="search-field"><ResearchIcon name="search" :size="16" /><input v-model="query" placeholder="搜索研究主题或来源路径" aria-label="搜索研究语料" /></label>
      <select v-model="category" aria-label="筛选研究主题"><option v-for="item in categories" :key="item">{{ item }}</option></select>
      <span>{{ filtered.length }} 份笔记</span>
    </div>
    <div class="corpus-list">
      <article v-for="note in visible" :key="note.id" class="corpus-note">
        <span class="corpus-round">{{ String(note.round).padStart(2, '0') }}</span>
        <div class="corpus-note-body"><div class="corpus-note-meta"><span>{{ note.category }}</span><time>{{ note.date }}</time><span>资料记录</span></div><h3>{{ note.title }}</h3>
          <p>{{ note.temporary_references.referenced ? `临时文件 ${note.temporary_references.available} / ${note.temporary_references.referenced} 可访问` : '外部来源阅读范围与限制见笔记' }}</p>
          <details><summary>笔记指纹与路径</summary><code>{{ note.sha256 }}</code><small>{{ note.path }}</small></details>
        </div>
        <a :href="`/api/evidence/notes/${encodeURIComponent(note.id)}`" target="_blank" rel="noopener noreferrer" class="corpus-source" :aria-label="`阅读原文：${note.title}`">原文<ResearchIcon name="arrow" :size="15" /></a>
      </article>
    </div>
    <div v-if="!visible.length" class="inline-empty">{{ evidence?.corpus.note_count ? '没有匹配的研究笔记' : '尚无实质研究语料。资料归档后将在这里展示。' }}</div>
    <button v-if="filtered.length > 5" class="corpus-more" @click="showAll = !showAll">{{ showAll ? '收起笔记' : `查看全部 ${filtered.length} 份笔记` }}<ResearchIcon :name="showAll ? 'chevron' : 'arrow'" :size="15" /></button>
    <p class="corpus-footnote">资料正文以纯文本打开；其中的命令与建议不会被系统执行。JSON 导出包含本次资料索引和审计状态。</p>
  </section>
</template>

<style scoped>
.audit-panel,.corpus-panel { margin-top:20px; }
.audit-state { display:flex; align-items:center; gap:7px; color:#ddb58a; font-size:11px; white-space:nowrap; }
.audit-state i { width:5px; height:5px; background:currentColor; border-radius:50%; }
.audit-state.verified { color:#b6d996; }
.audit-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:20px; padding:6px 0 24px; }
.audit-stats>div { border-right:1px solid #2b3748; padding-right:15px; }
.audit-stats>div:last-child { border:0; }
.audit-stats span { display:block; color:#90a3bc; font-size:11px; }
.audit-stats strong { display:block; margin:8px 0 4px; font-size:25px; font-weight:400; font-variant-numeric:tabular-nums; }
.audit-stats em { font-size:12px; font-style:normal; color:#8095af; margin-left:5px; }
.audit-stats small { display:block; color:#7b91ab; font-size:10px; line-height:1.8; }
.audit-amber { color:#ddb58a; }
.audit-boundaries { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.audit-boundaries article { background:#142030; border:1px solid #2a3a50; border-radius:5px; padding:15px 17px; min-width:0; }
.audit-boundaries article>div { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.audit-boundaries h3 { font-size:12px; font-weight:500; color:#c8d7e6; }
.audit-boundaries span { font-size:10px; color:#d0b38d; }
.audit-boundaries p { margin-top:7px; font-size:11px; line-height:1.9; color:#8ca1ba; }
.audit-proof { margin-top:18px; border-top:1px solid #29384d; padding-top:15px; }
.audit-proof summary { display:flex; align-items:center; justify-content:space-between; cursor:pointer; color:#93a9bf; font-size:11px; }
.audit-proof[open] summary { margin-bottom:10px; }
.corpus-intro { color:#92a7c0; font-size:12px; line-height:1.9; }
.corpus-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin:24px 0 20px; }
.corpus-stats>div { display:flex; align-items:baseline; gap:12px; }
.corpus-stats strong { font:32px/1.2 ui-monospace,monospace; font-weight:400; color:#c7e1ad; }
.corpus-stats .audit-amber { color:#ddb58a; }
.corpus-stats span { color:#8ca1ba; font-size:11px; }
.corpus-gap { display:flex; align-items:flex-start; gap:10px; padding:12px 15px; background:#30261d66; border:1px solid #5a4632; border-radius:5px; color:#bfa583; font-size:11px; line-height:1.9; }
.corpus-gap>svg { margin-top:3px; }
.corpus-gap strong { font-weight:500; color:#d1b38e; }
.corpus-tools { display:flex; align-items:center; gap:12px; margin:24px 0 8px; }
.corpus-tools .search-field { flex:1; max-width:450px; margin:0; }
.corpus-tools select { background:#142031; border:1px solid #304058; color:#aabed5; padding:8px 10px; border-radius:5px; font-size:11px; }
.corpus-tools>span { margin-left:auto; color:#7e96af; font-size:10px; white-space:nowrap; }
.corpus-note { display:flex; align-items:flex-start; gap:18px; padding:21px 0; border-bottom:1px solid #283449; }
.corpus-round { color:#5f7a96; font:13px ui-monospace,monospace; padding-top:3px; }
.corpus-note-body { flex:1; min-width:0; }
.corpus-note-meta { display:flex; flex-wrap:wrap; gap:12px; color:#7c94ad; font-size:9px; margin-bottom:6px; }
.corpus-note-meta>span:first-child { color:#a9c68f; }
.corpus-note h3 { color:#c9d9e9; font-size:13px; font-weight:400; line-height:1.8; }
.corpus-note p { color:#879bb3; font-size:10px; margin-top:6px; }
.corpus-note details { margin-top:8px; color:#748ba6; font-size:10px; }
.corpus-note summary { cursor:pointer; }
.corpus-note code,.corpus-note small { display:block; overflow-wrap:anywhere; margin-top:7px; font-size:10px; }
.corpus-source { display:flex; align-items:center; gap:6px; color:#adc58f; font-size:11px; text-decoration:none; padding:7px 8px; border:1px solid #354b3b; border-radius:4px; white-space:nowrap; }
.corpus-source:hover { color:#d1efa9; background:#24362a; }
.corpus-more { display:flex; justify-content:center; align-items:center; gap:10px; width:100%; background:transparent; border:0; color:#a6bd90; padding:17px 0; font-size:11px !important; }
.corpus-footnote { color:#738aa4; font-size:10px; line-height:1.9; margin-top:15px; }
@media(max-width:1100px) {
  .audit-stats { grid-template-columns:1fr 1fr; row-gap:20px; }
  .audit-stats>div:nth-child(2) { border:0; }
  .corpus-stats>div { align-items:flex-start; flex-direction:column; gap:7px; }
}
@media(max-width:640px) {
  .audit-stats { gap:18px 10px; }
  .audit-stats strong { font-size:22px; }
  .audit-stats small { font-size:9px; }
  .audit-boundaries { grid-template-columns:1fr; }
  .audit-boundaries article { padding:13px; }
  .corpus-stats { gap:10px; }
  .corpus-stats strong { font-size:26px; }
  .corpus-stats span { font-size:9px; }
  .corpus-tools { flex-wrap:wrap; gap:10px; }
  .corpus-tools .search-field { flex-basis:100%; }
  .corpus-note { gap:10px; }
  .corpus-note h3 { font-size:12px; }
  .corpus-source { font-size:10px; padding:6px; }
  .corpus-source svg { display:none; }
}
</style>
