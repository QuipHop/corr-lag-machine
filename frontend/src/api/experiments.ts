// src/api/experiments.ts
const API_BASE =
  import.meta.env.VITE_API_BASE || 'http://localhost:3000';

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `API error ${res.status}: ${text || res.statusText}`,
    );
  }

  return (await res.json()) as T;
}

// ---------- Типи, що відповідають бекенду ----------

export type SeriesRole = 'target' | 'candidate' | 'ignored';

export type Frequency = 'M' | 'Q' | 'Y';

export type Imputation = 'none' | 'ffill' | 'bfill' | 'interp';

export interface RunExperimentPayload {
  name: string;
  context?: string;
  dates: string[];
  series: {
    name: string;
    role: SeriesRole;
    values: (number | null)[];
  }[];
  frequency: Frequency;
  horizon: number;
  imputation?: Imputation;
  maxLag?: number;
}

export interface Experiment {
  id: string;
  name: string;
  context: string | null;
  frequency: string;
  horizon: number;
  createdAt: string;
  finishedAt: string | null;
  diagnostics: any;
  correlations: any;
  factors: any;
}

export interface ExperimentMetric {
  id: string;
  experimentId: string;
  seriesName: string;
  modelType: string;
  horizon: number;
  mase: number;
  smape: number;
  rmse: number;
}

export interface Model {
  id: string;
  experimentId: string;
  seriesName: string;
  modelType: string;
  paramsJson: any;
  mase: number | null;
  smape: number | null;
  rmse: number | null;
  isSelected: boolean;
}

export interface Forecast {
  id: string;
  experimentId: string;
  seriesName: string;
  date: string;
  valueActual: number | null;
  valuePred: number | null;
  lowerPi: number | null;
  upperPi: number | null;
  setType: string; // 'train' | 'test' | 'future'
}

export interface ExperimentWithMetrics extends Experiment {
  metrics: ExperimentMetric[];
}

export interface ExperimentDetails extends Experiment {
  models: Model[];
  metrics: ExperimentMetric[];
}

export interface MlModel {
  series_name: string;
  model_type: string;
  params: any;
  mase: number | null;
  smape: number | null;
  rmse: number | null;
  is_selected: boolean;
}

export interface MlForecastPoint {
  series_name: string;
  date: string;
  value_actual: number | null;
  value_pred: number | null;
  lower_pi: number | null;
  upper_pi: number | null;
  set_type: string;
}

export interface MlMetric {
  series_name: string;
  model_type: string;
  horizon: number;
  mase: number;
  smape: number;
  rmse: number;
}

export interface MlExperimentResult {
  diagnostics: any;
  correlations: any;
  factors: any;
  models: MlModel[];
  forecasts: {
    base: MlForecastPoint[];
    macro: MlForecastPoint[];
  };
  metrics: MlMetric[];
}

export interface RunExperimentResponse {
  experiment: ExperimentWithMetrics | null;
  mlResult: MlExperimentResult;
}

// ---------- Виклики API ----------

export async function runExperiment(
  payload: RunExperimentPayload,
): Promise<RunExperimentResponse> {
  return request<RunExperimentResponse>('/experiments/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function listExperiments(): Promise<
  ExperimentWithMetrics[]
> {
  return request<ExperimentWithMetrics[]>('/experiments');
}

export async function getExperiment(
  id: string,
): Promise<ExperimentDetails> {
  return request<ExperimentDetails>(`/experiments/${id}`);
}

export async function getExperimentForecasts(
  id: string,
): Promise<Forecast[]> {
  return request<Forecast[]>(`/experiments/${id}/forecasts`);
}
