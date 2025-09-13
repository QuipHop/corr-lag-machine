export type DatasetLite = { id: string; name: string; freq: 'monthly' | 'quarterly' | 'annual' | string; createdAt: string; seriesCount?: number };
export type SeriesLite = { id: string; key: string; label?: string | null; units?: string | null; createdAt: string };

export type ObservationPoint = { date: string; value: number };

export type CreateDatasetReq = { name: string; freq?: 'monthly' | 'quarterly' | 'annual' };
export type CreateDatasetResp = DatasetLite;

export type UploadResp = { uploadId: string; columns: { name: string; typeGuess: 'date' | 'number'; examples: string[] }[]; sampleRows: number };
export type PreviewReq = {
  dateColumn: string;
  valueColumns: { name: string; key: string; label?: string; units?: string }[];
  decimal: 'auto' | 'dot' | 'comma';
  dateFormat?: string;
  dropBlanks?: boolean;
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
