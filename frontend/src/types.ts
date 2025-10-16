export type DatasetLite = { id: string; name: string; freq: 'monthly' | 'quarterly' | 'annual' | string; createdAt: string; seriesCount?: number };
export type SeriesLite = { id: string; key: string; label?: string | null; units?: string | null; createdAt: string };

export type ObservationPoint = { date: string; value: number };

export type CreateDatasetReq = { name: string; freq?: 'monthly' | 'quarterly' | 'annual' };
export type CreateDatasetResp = DatasetLite;

export type Shape = 'long' | 'wide';
export type DecimalSep = 'auto' | '.' | ',';

export interface MappingDTO {
  shape: Shape;                  // long|wide
  dateColumn: string;            // назва колонки з датою
  valueColumns: string[];        // для wide: список колонок-значень; для long: одна колонка значень
  seriesKeyColumn?: string;      // для long: назва колонки з ключем серії
  dateFormat?: string;           // напр. "YYYY-MM" або autodetect
  decimal: DecimalSep;           // auto|.|,
  dropBlanks: boolean;           // знімати порожні
  frequency: 'M' | 'Q' | 'Y';    // бажана частота на виході (спочатку M)
}


export interface SaveMappingResp {
  datasetId: string;
  mappingHash: string; // sha1/json
}


export type UploadResp = { uploadId: string; columns: { name: string; typeGuess: 'date' | 'number'; examples: string[] }[]; sampleRows: number };
export type PreviewReq = {
  decimal: 'auto' | 'dot' | 'comma';
  dateFormat?: string;
  dropBlanks?: boolean;

  // long
  dateColumn?: string;
  valueColumns?: ValueColReq[];

  // wide
  shape?: 'long' | 'wide';
  seriesKeyColumn?: string;
  monthColumns?: string[]; // optional
  year?: number;           // optional
};

export type PreviewResp = {
  normalized: {
    freq: 'M';
    series: { key: string; label?: string; units?: string; rows: { date: string; value: number }[]; rowCount: number; warnings: string[] }[];
  };
  warnings: string[];
};
export type CommitReq = { saveMappingToDataset: boolean; createSeries: boolean; upsertMode: 'replace' | 'merge' };
export type CommitResp = { datasetId: string; seriesCreated: string[]; pointsUpserted: number };

export type CorrelateReq = {
  datasetId: string;
  series: string[];                  // series KEYS
  method?: 'spearman' | 'pearson';
  pearsonAlso?: boolean;
};
export type CorrelateResp = {
  freq: 'M';
  pairs: { x: string; y: string; n: number; spearman: number | null; pearson: number | null; overlapFrom: string; overlapTo: string }[];
};

export type PersistGraphReq = {
  datasetId: string;
  series?: string[];
  method?: 'spearman' | 'pearson';
  pearsonAlso?: boolean;
  minOverlap?: number;
  edgeMin?: number;
};
export type PersistGraphResp = {
  runId: number | null;
  datasetId: string;
  method: string;
  minOverlap: number;
  edgeMin: number;
  edgesInserted: number;
};


export type RunLite = {
  id: number;
  datasetId: string | null;
  createdAt: string;
  method: string;
  minOverlap: number;
  edgeMin: number;
  edgeCount: number;
};

export type RunDetail = {
  id: number;
  datasetId: string | null;
  createdAt: string;
  method: string;
  minOverlap: number;
  edgeMin: number;
  edges: { sourceId: string; targetId: string; sourceKey: string; targetKey: string; lag: number; weight: number }[];
};

export type ValueColReq = { name: string; key: string; label?: string; units?: string };

export type CorrMethod = "pearson" | "spearman";
export type TransformMode = "none" | "diff1" | "pct";

export interface Point { date: string; value: number; }
export interface SeriesIn { id: number; code: string; points: Point[]; }

export interface ResampleCfg {
  enabled?: boolean;
  freq?: "M";
  downsample?: "last" | "mean" | "sum";
  upsample?: "ffill" | "bfill" | "interpolate" | "none";
  winsorize_q?: number;
}
export interface LagCfg {
  min?: number; max?: number; ignoreZero?: boolean;
}

export interface CorrHeatmapReq {
  series: SeriesIn[];
  targetCode: string;
  candidateCodes?: string[];
  method: CorrMethod;
  minOverlap: number;
  resample: ResampleCfg;
  lag: LagCfg;
  topK?: number;
  returnStats?: boolean;
  transform: TransformMode;
  returnP?: boolean;
  fdrAlpha?: number;
}
export interface CorrHeatmapResp {
  matrix: { key: string; values: Record<string, number | null>; }[];
  bestByKey: any[]; // не використовується прямо у UI
  sortedTop: { key: string; lag: number; value: number; n: number; p?: number; passed_fdr?: boolean }[];
  meta: any;
  stats?: any[];
}

export interface CorrLagReq {
  series: SeriesIn[];
  maxLag?: number;
  method: CorrMethod;
  minOverlap: number;
  edgeMin: number;
  resample: ResampleCfg;
  normalizeOrientation?: boolean;
  dedupeOpposite?: boolean;
  topK?: number;
  perNodeTopK?: number;
  lag?: LagCfg;
  returnStats?: boolean;
  transform: TransformMode;
  returnP?: boolean;
  fdrAlpha?: number;
}
export interface CorrLagResp {
  nodes: { id: string }[];
  edges: { source: string; target: string; lag: number; weight: number; n: number; p?: number; passed_fdr?: boolean }[];
  meta: any;
  stats?: any[];
}

export interface FeaturesCfg {
  targetCode: string;
  features?: string[];
  lags?: Record<string, number>;
}
export interface AutoGridCfg {
  p?: [number, number]; d?: [number, number]; q?: [number, number];
  P?: [number, number]; D?: [number, number]; Q?: [number, number];
  s?: number; max_models?: number;
}
export interface TrainCfg {
  order?: { p?: number; d?: number; q?: number };
  seasonal_order?: { P?: number; D?: number; Q?: number; s?: number };
  trend?: "n" | "c" | "t" | "ct";
  enforce_stationarity?: boolean;
  enforce_invertibility?: boolean;
  auto_grid?: AutoGridCfg;
}
export interface SarimaxBacktestReq {
  series: SeriesIn[];
  resample: ResampleCfg;
  transform: TransformMode;
  features_cfg: FeaturesCfg;
  train: TrainCfg;
  backtest: { horizon: number; min_train: number; step: number; expanding: boolean; };
}
export interface SarimaxBacktestResp {
  metrics: { MAE: number; RMSE: number; sMAPE: number };
  meta: { n_obs: number; folds: number; timing_s: number };
}

export interface SarimaxForecastReq {
  series: SeriesIn[];
  resample: ResampleCfg;
  transform: TransformMode;
  features_cfg: FeaturesCfg;
  train: TrainCfg;
  horizon: number;
  return_pi?: boolean;
  alpha?: number;
}
export interface SarimaxForecastResp {
  order: [number, number, number];
  seasonal_order: [number, number, number, number];
  aic: number;
  fitted_end: string;
  forecast: { date: string; mean: number; lo?: number; hi?: number }[];
  meta: { timing_s: number };
}
