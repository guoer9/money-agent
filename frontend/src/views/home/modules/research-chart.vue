<script setup lang="ts">
import { watch } from 'vue';
import { useEcharts } from '@/hooks/common/echarts';
import type { ECOption } from '@/hooks/common/echarts';

const props = defineProps<{ option: ECOption; empty?: boolean; label?: string }>();
const { domRef, updateOptions } = useEcharts(() => props.option, { onRender: chart => { chart.hideLoading(); } });
watch(() => props.option, option => { updateOptions(() => option); }, { deep: true });
</script>

<template>
  <div class="research-chart">
    <div ref="domRef" class="chart-canvas" :aria-label="label || '实验图表'" role="img" />
    <div v-if="empty" class="chart-empty"><span class="empty-orbit">∿</span><p>等待真实评价结果</p><small>实验完成后将在这里呈现</small></div>
  </div>
</template>

<style scoped>
.research-chart { position:relative; width:100%; height:100%; min-height:180px; }
.chart-canvas { width:100%; height:100%; }
.chart-empty { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#101725e8; color:#a4b0c4; gap:6px; }
.chart-empty small { color:#758298; }
.empty-orbit { font-size:40px; color:#b5eb82; }
</style>
