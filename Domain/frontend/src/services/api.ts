import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

export type MisbehaviorTask = 'toxicity' | 'jailbreak' | 'lie' | 'backdoor';

export interface CausalMapData {
  image?: string;
  tokens?: string[];
  layers?: string[];
  values?: number[][];
}

export interface MetricRow {
  model: string;
  task: MisbehaviorTask;
  auc?: number;
  accuracy?: number;
}

export interface RawScanResponse {
  baseline_response?: string;
  normal_response?: string;
  misbehavior_response?: string;
  unsafe_response?: string;
  baseline_scores?: Record<string, number | string>;
  score_summary?: Record<string, number | string>;
  hallucination_details?: {
    low_agreement_count?: number;
  };
  layer_analysis?: {
    influence_scores?: number[];
  };
  interventions?: Array<{
    new_response?: string;
  }>;
  plots?: Record<string, string | undefined>;
  causal_map?: string | CausalMapData;
  evaluation_metrics?: unknown;
  metrics?: unknown;
}

export async function runScan(prompt: string, model: string): Promise<RawScanResponse> {
  const response = await axios.post(`${API_BASE_URL}/scan`, { prompt, model });
  return response.data.data;
}

export async function fetchCausalMap(
  prompt: string,
  model: string,
  task: MisbehaviorTask,
  firstTokenOnly: boolean,
): Promise<CausalMapData | null> {
  try {
    const response = await axios.post(`${API_BASE_URL}/causal-map`, {
      prompt,
      model,
      task,
      first_token_only: firstTokenOnly,
    });

    return normalizeCausalMap(response.data.data ?? response.data) ?? null;
  } catch (error) {
    console.warn('Causal map endpoint unavailable, using scan result fallback.', error);
    return null;
  }
}

export async function fetchEvaluationMetrics(): Promise<MetricRow[]> {
  const response = await axios.get(`${API_BASE_URL}/metrics`);
  return normalizeMetricRows(response.data.data ?? response.data);
}

export function normalizeCausalMap(raw: unknown): CausalMapData | undefined {
  if (!raw) return undefined;
  if (typeof raw === 'string') return { image: raw };

  if (typeof raw === 'object') {
    const value = raw as Record<string, unknown>;
    return {
      image: typeof value.image === 'string'
        ? value.image
        : typeof value.base64 === 'string'
          ? value.base64
          : undefined,
      tokens: Array.isArray(value.tokens) ? value.tokens.map(String) : undefined,
      layers: Array.isArray(value.layers) ? value.layers.map(String) : undefined,
      values: Array.isArray(value.values) ? normalizeMatrix(value.values) : undefined,
    };
  }

  return undefined;
}

export function normalizeMetricRows(raw: unknown): MetricRow[] {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.flatMap((item) => normalizeMetricRow(item));
  }

  if (typeof raw !== 'object') return [];

  const source = raw as Record<string, unknown>;
  return Object.entries(source).flatMap(([model, taskValue]) => {
    if (!taskValue || typeof taskValue !== 'object') return [];

    return Object.entries(taskValue as Record<string, unknown>).flatMap(([task, metricValue]) => {
      if (!isTask(task) || !metricValue || typeof metricValue !== 'object') return [];
      const metrics = metricValue as Record<string, unknown>;
      return [{
        model,
        task,
        auc: toNumber(metrics.auc ?? metrics.AUC),
        accuracy: toNumber(metrics.accuracy ?? metrics.acc),
      }];
    });
  });
}

function normalizeMetricRow(raw: unknown): MetricRow[] {
  if (!raw || typeof raw !== 'object') return [];

  const item = raw as Record<string, unknown>;
  const task = String(item.task ?? '').toLowerCase();
  if (!isTask(task)) return [];

  return [{
    model: String(item.model ?? 'Unknown'),
    task,
    auc: toNumber(item.auc ?? item.AUC),
    accuracy: toNumber(item.accuracy ?? item.acc),
  }];
}

function normalizeMatrix(raw: unknown[]): number[][] {
  return raw.map((row) => (
    Array.isArray(row) ? row.map((value) => toNumber(value) ?? 0) : []
  ));
}

function isTask(task: string): task is MisbehaviorTask {
  return ['toxicity', 'jailbreak', 'lie', 'backdoor'].includes(task);
}

function toNumber(value: unknown): number | undefined {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}
