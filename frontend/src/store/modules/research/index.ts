import { computed, ref } from 'vue';
import { defineStore } from 'pinia';
import { fetchOverview } from '@/service/api/research';
import type { Overview } from '@/service/api/research';

export const useResearchStore = defineStore('research', () => {
  const overview = ref<Overview | null>(null);
  const loading = ref(false);
  const error = ref('');
  const selectedId = ref('');
  const lastUpdated = ref('');
  const archive = computed(() => overview.value?.archive);
  const factors = computed(() => archive.value?.factors || []);
  const selected = computed(() => factors.value.find(f => f.id === selectedId.value) || factors.value[0]);

  async function refresh() {
    if (loading.value) return;
    loading.value = true;
    error.value = '';
    try {
      overview.value = await fetchOverview();
      lastUpdated.value = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    } catch (cause) {
      error.value = cause instanceof Error ? cause.message : '研究服务暂时不可用';
    } finally {
      loading.value = false;
    }
  }

  return { overview, archive, factors, selected, selectedId, loading, error, lastUpdated, refresh };
});
