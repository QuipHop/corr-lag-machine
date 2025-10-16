const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:3000';

export type Point = { date: string; value: number };
export type SeriesIn = { id: number; code: string; points: Point[] };

export type TransformMode = 'none' | 'diff1' | 'pct';
export type Method = 'pearson' | 'spearman';

export type LagCfg = { min: number; max: number; ignoreZero: boolean };
export type ResampleCfg = {
    enabled: boolean;
    downsample: 'last' | 'mean' | 'sum';
    upsample: 'ffill' | 'bfill' | 'interpolate' | 'none';
    winsorize_q: number;
};

export type MetaHeaders = {
    requestId: string;
    cache: string;
    cacheAge: string;
    attempts: string;
    rtMs: string;
};

function pickHeaders(res: Response): MetaHeaders {
    return {
        requestId: res.headers.get('x-request-id') || '',
        cache: res.headers.get('x-ml-cache') || '',
        cacheAge: res.headers.get('x-ml-cache-age') || '',
        attempts: res.headers.get('x-ml-attempts') || '',
        rtMs: res.headers.get('x-ml-rt-ms') || '',
    };
}

async function postJSON<TReq, TRes>(
    path: string,
    body: TReq,
    opts?: { refresh?: boolean; requestId?: string },
): Promise<{ status: number; headers: MetaHeaders; data: TRes }> {
    const url = `${BASE}${path}${opts?.refresh ? '?refresh=1' : ''}`;
    const res = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(opts?.requestId ? { 'x-request-id': opts.requestId } : {}),
        },
        body: JSON.stringify(body),
    });

    let data: any = null;
    try { data = await res.json(); } catch { }

    if (!res.ok) {
        const err: any = new Error((data && (data.message || data.error)) || `HTTP ${res.status}`);
        err.status = res.status;
        err.headers = pickHeaders(res);
        err.payload = data;
        throw err;
    }

    return { status: res.status, headers: pickHeaders(res), data };
}

/* ------------ Corr Heatmap -------------- */
export type HeatmapMatrixRow = { key: string; values: Record<string, number | null> };
export type HeatmapBest = { key: string; lag: number; value: number; n: number };
export type HeatmapMeta = {
    method: Method;
    minOverlap: number;
    lagRange: { min: number; max: number; ignoreZero: boolean };
    resample?: any;
    seriesCount: number;
    rowsCombined: number;
    target: string;
    periodStart?: string | null;
    periodEnd?: string | null;
    timing?: { total_s: number; compute_s: number; prep_s: number };
};

export type CorrHeatmapRequest = {
    series: SeriesIn[];
    targetCode: string;
    candidateCodes?: string[];
    method: Method;
    minOverlap: number;
    resample: ResampleCfg;
    lag: LagCfg;
    topK?: number;
    returnStats?: boolean;
    transform?: TransformMode;
};

export type CorrHeatmapResponse = {
    matrix: HeatmapMatrixRow[];
    bestByKey: HeatmapBest[];
    sortedTop: HeatmapBest[];
    meta: HeatmapMeta;
    stats?: Array<{
        code: string; n_total: number; n_notna: number; n_na: number;
        start?: string | null; end?: string | null; std?: number | null;
    }>;
};

export async function corrHeatmap(body: CorrHeatmapRequest, opts?: { refresh?: boolean; requestId?: string }) {
    return postJSON<CorrHeatmapRequest, CorrHeatmapResponse>('/analysis/corr-heatmap', body, opts);
}

/* --------------- Corr-Lag ---------------- */
export type CorrLagRequest = {
    series: SeriesIn[];
    method: Method;
    minOverlap: number;
    edgeMin: number;
    lag: LagCfg;
    normalizeOrientation?: boolean;
    dedupeOpposite?: boolean;
    topK?: number;
    perNodeTopK?: number;
    returnStats?: boolean;
    transform?: TransformMode;
};

export type CorrLagResponse = {
    nodes: { id: string }[];
    edges: { source: string; target: string; lag: number; weight: number; n: number }[];
    meta: {
        method: Method;
        minOverlap: number;
        edgeMin: number;
        lagRange: { min: number; max: number; ignoreZero: boolean };
        resample?: any;
        seriesCount: number;
        rowsCombined: number;
        periodStart?: string | null;
        periodEnd?: string | null;
        timing?: { total_s: number; compute_s: number; prep_s: number };
    };
    stats?: CorrHeatmapResponse['stats'];
};

export async function corrLag(body: CorrLagRequest, opts?: { refresh?: boolean; requestId?: string }) {
    return postJSON<CorrLagRequest, CorrLagResponse>('/analysis/corr-lag', body, opts);
}
