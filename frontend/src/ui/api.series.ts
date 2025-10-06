import { getJSON, postFile, postJSON } from '../api';
import type {
    DatasetLite, SeriesLite, ObservationPoint,
    CreateDatasetReq, CreateDatasetResp,
    UploadResp, PreviewReq, PreviewResp, CommitReq, CommitResp,
    CorrelateReq, CorrelateResp, PersistGraphReq, PersistGraphResp
} from '../types';
import type { RunLite, RunDetail } from '../types';

// Datasets
export const listDatasets = () => getJSON<DatasetLite[]>('/datasets');
export const getDataset = (id: string) => getJSON<DatasetLite>(`/datasets/${id}`);
export const createDataset = (body: CreateDatasetReq) => postJSON<CreateDatasetResp>('/datasets', body);
export const listSeriesForDataset = (datasetId: string) => getJSON<SeriesLite[]>(`/datasets/${datasetId}/series`);
export const getSavedMapping = (datasetId: string) => getJSON<Record<string, unknown>>(`/datasets/${datasetId}/mapping`);

// Series & data
export const getSeries = (id: string) => getJSON<any>(`/series/${id}`);
export const getSeriesData = (id: string, from?: string, to?: string) => {
    const q = new URLSearchParams(); if (from) q.set('from', from); if (to) q.set('to', to);
    return getJSON<{ seriesId: string; points: ObservationPoint[] }>(`/series/${id}/data${q.toString() ? `?${q}` : ''}`);
};

// Uploads
export const uploadFile = (datasetId: string, file: File) => postFile<UploadResp>(`/datasets/${datasetId}/upload`, file);
export const previewUpload = (uploadId: string, body: PreviewReq) => postJSON<PreviewResp>(`/uploads/preview/${uploadId}`, body);
export const commitUpload = (uploadId: string, body: CommitReq) => postJSON<CommitResp>(`/uploads/commit/${uploadId}`, body, { 'Idempotency-Key': crypto.randomUUID() });

// Correlation
export const correlate = (body: CorrelateReq) => postJSON<CorrelateResp>('/correlate', body);
export const persistGraph = (body: PersistGraphReq) => postJSON<PersistGraphResp>('/analysis/graph', body);
export const listRuns = (datasetId?: string) =>
    getJSON<RunLite[]>(`/analysis/runs${datasetId ? `?datasetId=${datasetId}` : ''}`);

export const getRun = (id: number) =>
    getJSON<RunDetail>(`/analysis/run/${id}`);