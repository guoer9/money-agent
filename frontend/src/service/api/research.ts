/** The research API uses plain JSON, independently of the upstream demo envelope. */
export interface FactorMetrics {
  ic_mean: number | null;
  rank_ic_mean: number | null;
  ic_ir: number | null;
  rank_ic_ir: number | null;
  coverage: number | null;
  valid_ic_days: number;
}

export interface ResearchFactor {
  id: string;
  name: string;
  expression: string;
  description: string;
  segment: string;
  origin?: string;
  metrics: FactorMetrics;
  daily: { date: string; ic: number | null; rank_ic: number | null }[];
  quantiles: { group: number | string; mean_return: number | null }[];
  distribution?: { bin: string; count: number }[];
}

export interface ResearchArchive {
  dataset: {
    id: string;
    source: string;
    start: string;
    end: string;
    rows: number;
    instruments: number;
    sha256: string;
    adjustment: string;
    quality: Record<string, unknown>;
    splits: Record<string, string[]>;
  };
  protocol: Record<string, unknown> & { id: string; label: string };
  factors: ResearchFactor[];
  generated_at: string;
}

export interface ResearchRun {
  id: string;
  mode: string;
  status: string;
  prompt: string;
  created_at: string;
  updated_at?: string;
  error?: string | null;
  events?: { time: string; stage: string; message: string }[];
  result?: Record<string, unknown> | null;
}

export interface Overview {
  archive: ResearchArchive | null;
  runs: ResearchRun[];
  environment: { llm_configured: boolean; rdagent_available: boolean; worker_running: boolean; model: string | null };
  final_test: { status: string; message: string };
  server_time: string;
}

async function researchRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/api${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...init.headers }
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `服务返回 ${response.status}`);
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

export const fetchOverview = () => researchRequest<Overview>('/overview');
export const fetchRun = (id: string) => researchRequest<ResearchRun>(`/runs/${encodeURIComponent(id)}`);
export const submitResearch = (request: { prompt: string; mode: string; iterations: number }, key: string) =>
  researchRequest<ResearchRun>('/runs', {
    method: 'POST',
    headers: { 'Idempotency-Key': key },
    body: JSON.stringify(request)
  });

export async function downloadReport() {
  const response = await fetch('/api/reports/export');
  if (!response.ok) throw new Error('导出失败，请确认已有实验归档');
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = '因子研究_实验材料.json';
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
