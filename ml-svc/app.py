from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from time import perf_counter

# --- NEW: для p-value/FDR та SARIMAX ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import t as t_dist, rankdata


# ===================== Helpers: базові трансформації =====================

def _apply_transform(s: pd.Series, mode: str) -> pd.Series:
    """
    none  : без змін
    diff1 : перша різниця (s_t - s_{t-1})
    pct   : (s_t / s_{t-1} - 1) * 100
    """
    if mode == 'diff1':
        return s.diff()
    if mode == 'pct':
        base = s.shift(1)
        out = (s / base - 1.0) * 100.0
        return out
    return s


def _asdict(model):
    # Pydantic v2
    if hasattr(model, "model_dump"):
        return model.model_dump()
    # Pydantic v1 fallback
    if hasattr(model, "dict"):
        return model.dict()
    return {}


# ===================== FastAPI & CORS =====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ===================== Pydantic I/O Models =====================

class Point(BaseModel):
    date: str
    value: float

class SeriesIn(BaseModel):
    id: int
    code: str
    points: List[Point]

class ResampleCfg(BaseModel):
    enabled: bool = True
    freq: Literal['M'] = 'M'
    downsample: Literal['last', 'mean', 'sum'] = 'last'
    upsample: Literal['ffill', 'bfill', 'interpolate', 'none'] = 'ffill'
    winsorize_q: float = 0.0  # 0..0.2

class LagCfg(BaseModel):
    min: int = -12
    max: int = 12
    ignoreZero: bool = False

class CorrLagRequest(BaseModel):
    series: List[SeriesIn]
    # legacy
    maxLag: int = 12

    method: Literal['pearson', 'spearman'] = 'pearson'
    minOverlap: int = 12
    edgeMin: float = 0.3
    resample: ResampleCfg = ResampleCfg()

    normalizeOrientation: bool = True
    dedupeOpposite: bool = True
    topK: Optional[int] = None
    perNodeTopK: Optional[int] = None

    lag: Optional[LagCfg] = None
    returnStats: bool = False
    transform: Literal['none', 'diff1', 'pct'] = 'none'

    # NEW:
    returnP: bool = False              # рахувати p-value для best лагів
    fdrAlpha: Optional[float] = None   # застосувати BH FDR, якщо є p

class CorrHeatmapRequest(BaseModel):
    series: List[SeriesIn]
    targetCode: str
    candidateCodes: Optional[List[str]] = None
    method: Literal['pearson', 'spearman'] = 'pearson'
    minOverlap: int = 12
    resample: ResampleCfg = ResampleCfg()
    lag: LagCfg = LagCfg()
    topK: Optional[int] = None
    returnStats: bool = False
    transform: Literal['none', 'diff1', 'pct'] = 'none'

    # NEW:
    returnP: bool = False
    fdrAlpha: Optional[float] = None

# --- SARIMAX config models ---

class SarimaxOrder(BaseModel):
    p: int = 1
    d: int = 0
    q: int = 0

class SarimaxSeasonalOrder(BaseModel):
    P: int = 0
    D: int = 0
    Q: int = 0
    s: int = 12

class AutoGridCfg(BaseModel):
    p: Tuple[int, int] = (0, 2)
    d: Tuple[int, int] = (0, 1)
    q: Tuple[int, int] = (0, 2)
    P: Tuple[int, int] = (0, 1)
    D: Tuple[int, int] = (0, 1)
    Q: Tuple[int, int] = (0, 1)
    s: int = 12
    max_models: int = 30

class FeaturesCfg(BaseModel):
    targetCode: str
    features: Optional[List[str]] = None
    lags: Optional[Dict[str, int]] = None  # {"CPI": 3, ...}

class TrainCfg(BaseModel):
    order: Optional[SarimaxOrder] = None
    seasonal_order: Optional[SarimaxSeasonalOrder] = None
    trend: Optional[Literal['n','c','t','ct']] = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    auto_grid: Optional[AutoGridCfg] = None

class BacktestCfg(BaseModel):
    horizon: int = 3
    min_train: int = 36
    step: int = 1
    expanding: bool = True

class SarimaxBacktestRequest(BaseModel):
    series: List[SeriesIn]
    resample: ResampleCfg = ResampleCfg()
    transform: Literal['none', 'diff1', 'pct'] = 'none'
    features_cfg: FeaturesCfg
    train: TrainCfg
    backtest: BacktestCfg

class SarimaxForecastRequest(BaseModel):
    series: List[SeriesIn]
    resample: ResampleCfg = ResampleCfg()
    transform: Literal['none', 'diff1', 'pct'] = 'none'
    features_cfg: FeaturesCfg
    train: TrainCfg
    horizon: int = 6
    return_pi: bool = True
    alpha: float = 0.05


# ===================== Corr-engine helpers =====================

MAX_LAG_SPAN = 60
MIN_MIN_OVERLAP = 3

def _winsorize(s: pd.Series, q: float) -> pd.Series:
    if q <= 0:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)

def _to_monthly(sr: pd.Series, cfg: ResampleCfg) -> pd.Series:
    s = sr.sort_index().copy()
    s.index = s.index.to_period('M').to_timestamp('M')
    if cfg.downsample == 'mean':
        s = s.groupby(level=0).mean()
    elif cfg.downsample == 'sum':
        s = s.groupby(level=0).sum()
    else:
        s = s.groupby(level=0).last()

    idx = pd.period_range(
        s.index.min().to_period('M'),
        s.index.max().to_period('M'),
        freq='M'
    ).to_timestamp('M')
    s = s.reindex(idx)

    if cfg.upsample == 'ffill':
        s = s.ffill()
    elif cfg.upsample == 'bfill':
        s = s.bfill()
    elif cfg.upsample == 'interpolate':
        s = s.interpolate(method='time')

    s = _winsorize(s, cfg.winsorize_q)
    return s

def _normalize_edges_orientation(edges: List[Dict]) -> List[Dict]:
    out = []
    for e in edges:
        L = e["lag"]
        ee = dict(e)
        if L < 0:
            ee["lag"] = -L
        elif L > 0:
            ee["source"], ee["target"] = ee["target"], ee["source"]
        out.append(ee)
    return out

def _dedupe_undirected(edges: List[Dict]) -> List[Dict]:
    sorted_edges = sorted(
        edges,
        key=lambda e: (-abs(e["weight"]), abs(e["lag"]), 0 if e["source"] <= e["target"] else 1)
    )
    seen = set()
    out = []
    for e in sorted_edges:
        key = tuple(sorted([e["source"], e["target"]]))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

def _apply_per_node_topk(edges: List[Dict], k: int) -> List[Dict]:
    if not k or k <= 0:
        return edges
    scored = sorted(edges, key=lambda e: (-abs(e["weight"]), abs(e["lag"])))
    keep = set()
    deg: Dict[str, int] = {}
    for e in scored:
        u, v = e["source"], e["target"]
        du = deg.get(u, 0)
        dv = deg.get(v, 0)
        if du < k or dv < k:
            keep.add((u, v, e["lag"], e["weight"]))
            deg[u] = du + 1
            deg[v] = dv + 1
    out = []
    for e in scored:
        key = (e["source"], e["target"], e["lag"], e["weight"])
        if key in keep:
            out.append(e)
    return out

def _validate_common(minOverlap: int):
    if minOverlap < MIN_MIN_OVERLAP:
        raise HTTPException(status_code=400, detail=f"minOverlap must be >= {MIN_MIN_OVERLAP}")

def _resolve_lag_bounds(length: int, minOverlap: int, lag_cfg: Optional[LagCfg], legacy_max_lag: Optional[int]):
    max_possible_lag = max(0, length - minOverlap)
    if lag_cfg is not None:
        Lmin, Lmax = int(lag_cfg.min), int(lag_cfg.max)
        ignore_zero = bool(lag_cfg.ignoreZero)
    else:
        Lmin, Lmax = -int(legacy_max_lag or 12), int(legacy_max_lag or 12)
        ignore_zero = False
    if (Lmax - Lmin) > MAX_LAG_SPAN:
        raise HTTPException(status_code=400, detail=f"lag span too wide (max {MAX_LAG_SPAN})")
    Lmin = max(Lmin, -max_possible_lag)
    Lmax = min(Lmax,  max_possible_lag)
    return Lmin, Lmax, ignore_zero

def _build_frames(series: List[SeriesIn], resample: ResampleCfg, transform: str):
    frames: Dict[str, pd.Series] = {}
    for s in series:
        if not s.points:
            continue
        df = pd.DataFrame([{"date": p.date, "value": p.value} for p in s.points])
        df["date"] = pd.to_datetime(df["date"])
        df = (df.groupby("date", as_index=False)["value"].mean()
                .sort_values("date").set_index("date"))
        sr = df["value"]
        if sr.dropna().nunique() < 2:
            continue
        if resample and resample.enabled:
            sr = _to_monthly(sr, resample)
        if transform and transform != 'none':
            sr = _apply_transform(sr, transform)
        frames[s.code] = sr
    return frames

def _frames_stats(frames: Dict[str, pd.Series]):
    stats = []
    for code, sr in frames.items():
        n_total = int(len(sr))
        n_notna = int(sr.notna().sum())
        n_na = n_total - n_notna
        first = sr.first_valid_index()
        last = sr.last_valid_index()
        stats.append({
            "code": code,
            "n_total": n_total,
            "n_notna": n_notna,
            "n_na": n_na,
            "start": None if first is None else str(first.date()),
            "end": None if last is None else str(last.date()),
            "std": None if n_notna < 2 else float(sr.dropna().std()),
        })
    return stats


# ===================== p-value & FDR helpers =====================

def _pearson_pvalue(r: float, n: int) -> float:
    if n < 3 or pd.isna(r):
        return np.nan
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(2 * (1 - t_dist.cdf(abs(t), df=n - 2)))

def _spearman_via_ranks(x: pd.Series, y: pd.Series) -> Tuple[float, int, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < 3:
        return (np.nan, n, np.nan)
    rx = rankdata(pair.iloc[:, 0].values)
    ry = rankdata(pair.iloc[:, 1].values)
    r = np.corrcoef(rx, ry)[0, 1]
    p = _pearson_pvalue(r, n)  # наближення p через Пірсона над рангами
    return (float(r), n, float(p))

def _bh_fdr(pvals: List[float], alpha: float) -> List[bool]:
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = np.array([(i + 1) * alpha / m for i in range(m)])
    passed = np.zeros(m, dtype=bool)
    max_i = -1
    for rank, idx in enumerate(order):
        if pvals[idx] <= thresh[rank]:
            max_i = rank
    if max_i >= 0:
        passed[order[:max_i + 1]] = True
    return passed.tolist()


# ===================== SARIMAX helpers =====================

def _build_exog(frames: Dict[str, pd.Series], target: str,
                features: Optional[List[str]], lags: Optional[Dict[str,int]]):
    if target not in frames:
        raise HTTPException(status_code=400, detail=f"Target '{target}' not found in frames.")

    if features is None:
        features = [c for c in frames.keys() if c != target]

    cols = {}
    for f in features:
        if f not in frames:
            continue
        lag = int(lags.get(f, 0)) if lags else 0
        s = frames[f].shift(lag)
        cols[f] = s

    X = pd.DataFrame(cols).sort_index() if cols else None
    y = frames[target].sort_index()

    if X is not None:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        if df.empty:
            raise HTTPException(status_code=400, detail="No overlap between target and features after shifting/resample.")
        y = df["y"]
        X = df.drop(columns=["y"])
    else:
        y = y.dropna()
        X = None

    return y, X

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def _rmse(y_true, y_pred):
    e = np.array(y_true) - np.array(y_pred)
    return float(np.sqrt(np.mean(e*e)))

def _smape(y_true, y_pred):
    yt = np.array(y_true); yp = np.array(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(yt - yp) / denom)) * 100.0

def _fit_one(y, X, cfg: TrainCfg):
    if cfg.auto_grid:
        grid = []
        rg = cfg.auto_grid
        for p in range(rg.p[0], rg.p[1]+1):
            for d in range(rg.d[0], rg.d[1]+1):
                for q in range(rg.q[0], rg.q[1]+1):
                    for P in range(rg.P[0], rg.P[1]+1):
                        for D in range(rg.D[0], rg.D[1]+1):
                            for Q in range(rg.Q[0], rg.Q[1]+1):
                                grid.append(((p,d,q),(P,D,Q,rg.s)))
        best = None
        tried = 0
        for (order, seas) in grid:
            if tried >= rg.max_models:
                break
            try:
                model = SARIMAX(y, exog=X, order=order, seasonal_order=seas,
                                trend=cfg.trend, enforce_stationarity=cfg.enforce_stationarity,
                                enforce_invertibility=cfg.enforce_invertibility)
                res = model.fit(disp=False)
                score = res.aic
                if (best is None) or (score < best[0]):
                    best = (score, order, seas, res)
                tried += 1
            except Exception:
                continue
        if best is None:
            raise HTTPException(status_code=422, detail="Auto-grid could not fit any model.")
        return {"res": best[3], "order": best[1], "seasonal_order": best[2], "aic": float(best[0])}
    else:
        order = (cfg.order.p, cfg.order.d, cfg.order.q) if cfg.order else (1,0,0)
        seas = (0,0,0,12)
        if cfg.seasonal_order:
            seas = (cfg.seasonal_order.P, cfg.seasonal_order.D, cfg.seasonal_order.Q, cfg.seasonal_order.s)
        model = SARIMAX(y, exog=X, order=order, seasonal_order=seas,
                        trend=cfg.trend, enforce_stationarity=cfg.enforce_stationarity,
                        enforce_invertibility=cfg.enforce_invertibility)
        res = model.fit(disp=False)
        return {"res": res, "order": order, "seasonal_order": seas, "aic": float(res.aic)}


# ===================== ROUTES =====================

@app.get('/health')
async def health():
    return {"status": "ok"}


# ---------- /corr-lag ----------

@app.post('/corr-lag')
async def corr_lag(req: CorrLagRequest):
    t0 = perf_counter()
    _validate_common(req.minOverlap)

    frames = _build_frames(req.series, req.resample, req.transform)
    if not frames:
        return {"nodes": [], "edges": [], "meta": {"reason": "no_valid_series"}}

    combined = pd.DataFrame(frames).sort_index().dropna(how="all")
    nodes = [{"id": code} for code in combined.columns]

    Lmin, Lmax, ignore_zero = _resolve_lag_bounds(len(combined), req.minOverlap, req.lag, req.maxLag)
    if Lmin > Lmax:
        meta = {
            "method": req.method, "minOverlap": req.minOverlap, "edgeMin": req.edgeMin,
            "lagRange": {"min": Lmin, "max": Lmax, "ignoreZero": ignore_zero},
            "resample": _asdict(req.resample) if req.resample else None,
            "seriesCount": len(frames), "rowsCombined": int(len(combined)),
            "timing": {"total_s": round(perf_counter() - t0, 6)},
            "transform": req.transform,
        }
        return {"nodes": nodes, "edges": [], "meta": meta}

    t1 = perf_counter()
    edges: List[Dict] = []
    cols = list(combined.columns)
    shift_cache: Dict[Tuple[str, int], pd.Series] = {}

    for i, src_code in enumerate(cols):
        src = combined[src_code]
        if src.dropna().nunique() < 2:
            continue
        for j, tgt_code in enumerate(cols):
            if i == j:
                continue
            tgt = combined[tgt_code]
            if tgt.dropna().nunique() < 2:
                continue

            best_corr = float("nan")
            best_lag = 0
            best_n = 0

            for lag in range(Lmin, Lmax + 1):
                if ignore_zero and lag == 0:
                    continue
                key = (tgt_code, lag)
                if key not in shift_cache:
                    shift_cache[key] = tgt.shift(lag)

                pair = pd.concat([src, shift_cache[key]], axis=1, join="inner").dropna()
                n = len(pair)
                if n < req.minOverlap:
                    continue

                corr = pair.corr(method=("spearman" if req.method == "spearman" else "pearson")).iloc[0, 1]
                if pd.isna(corr):
                    continue

                if (pd.isna(best_corr)
                    or abs(corr) > abs(best_corr)
                    or (abs(corr) == abs(best_corr) and abs(lag) < abs(best_lag))):
                    best_corr, best_lag, best_n = float(corr), int(lag), int(n)

            if pd.notna(best_corr) and abs(best_corr) >= req.edgeMin:
                # p-value для best лагу (за бажанням)
                pval = None
                if req.returnP:
                    best_pair = pd.concat([src, tgt.shift(best_lag)], axis=1, join="inner").dropna()
                    if req.method == "spearman":
                        r, n_eff, pval = _spearman_via_ranks(best_pair.iloc[:,0], best_pair.iloc[:,1])
                        pval = float(pval)
                    else:
                        n_eff = len(best_pair)
                        pval = _pearson_pvalue(best_corr, n_eff)

                edge = {
                    "source": src_code, "target": tgt_code,
                    "lag": best_lag, "weight": round(best_corr, 6), "n": best_n,
                }
                if req.returnP:
                    edge["p"] = None if pval is None else float(pval)
                edges.append(edge)

    edges.sort(key=lambda e: abs(e["weight"]), reverse=True)
    if req.normalizeOrientation:
        edges = _normalize_edges_orientation(edges)
    if req.dedupeOpposite:
        edges = _dedupe_undirected(edges)
    if req.perNodeTopK is not None and req.perNodeTopK > 0:
        edges = _apply_per_node_topk(edges, req.perNodeTopK)
    if req.topK is not None and req.topK > 0:
        edges = edges[:req.topK]

    # FDR (BH)
    if req.fdrAlpha is not None and req.returnP:
        pvals = [e["p"] for e in edges if e.get("p") is not None]
        if pvals:
            mask = _bh_fdr([float(p) for p in pvals], float(req.fdrAlpha))
            k = 0
            for e in edges:
                if e.get("p") is not None:
                    e["passed_fdr"] = bool(mask[k])
                    k += 1

    t2 = perf_counter()
    first_idx = combined.first_valid_index()
    last_idx = combined.last_valid_index()
    meta = {
        "method": req.method, "minOverlap": req.minOverlap, "edgeMin": req.edgeMin,
        "lagRange": {"min": Lmin, "max": Lmax, "ignoreZero": ignore_zero},
        "resample": _asdict(req.resample) if req.resample else None,
        "seriesCount": len(frames), "rowsCombined": int(len(combined)),
        "periodStart": None if first_idx is None else str(first_idx.date()),
        "periodEnd": None if last_idx is None else str(last_idx.date()),
        "timing": {
            "total_s": round(perf_counter() - t0, 6),
            "compute_s": round(t2 - t1, 6),
            "prep_s": round(t1 - t0, 6),
        },
        "transform": req.transform
    }

    resp = {"nodes": nodes, "edges": edges, "meta": meta}
    if req.returnStats:
        resp["stats"] = _frames_stats(frames)
    return resp


# ---------- /corr-heatmap ----------

@app.post('/corr-heatmap')
async def corr_heatmap(req: CorrHeatmapRequest):
    t0 = perf_counter()
    _validate_common(req.minOverlap)

    frames = _build_frames(req.series, req.resample, req.transform)
    if not frames:
        return {"matrix": [], "bestByKey": [], "sortedTop": [], "meta": {"reason": "no_valid_series"}}

    combined = pd.DataFrame(frames).sort_index().dropna(how="all")

    if req.targetCode not in combined.columns:
        raise HTTPException(status_code=400, detail=f"targetCode '{req.targetCode}' not found")

    target = combined[req.targetCode]
    if req.candidateCodes:
        candidates = [c for c in req.candidateCodes if c in combined.columns and c != req.targetCode]
    else:
        candidates = [c for c in combined.columns if c != req.targetCode]

    if not candidates:
        return {
            "matrix": [], "bestByKey": [], "sortedTop": [],
            "meta": {"reason": "no_candidates", "target": req.targetCode}
        }

    Lmin, Lmax, ignore_zero = _resolve_lag_bounds(len(combined), req.minOverlap, req.lag, None)
    if Lmin > Lmax:
        meta = {
            "method": req.method, "minOverlap": req.minOverlap,
            "lagRange": {"min": Lmin, "max": Lmax, "ignoreZero": ignore_zero},
            "resample": _asdict(req.resample) if req.resample else None,
            "seriesCount": len(frames), "rowsCombined": int(len(combined)),
            "target": req.targetCode,
            "timing": {"total_s": round(perf_counter() - t0, 6)},
            "transform": req.transform
        }
        return {"matrix": [], "bestByKey": [], "sortedTop": [], "meta": meta}

    t1 = perf_counter()
    target_shifts: Dict[int, pd.Series] = {
        lag: target.shift(lag) for lag in range(Lmin, Lmax + 1)
        if not (ignore_zero and lag == 0)
    }

    matrix = []
    best = []

    for key in candidates:
        cand = combined[key]
        if cand.dropna().nunique() < 2:
            continue

        vals: Dict[str, Optional[float]] = {}
        best_corr = float("nan")
        best_lag_internal = 0
        best_n = 0
        best_p = None

        for lag, t_shift in target_shifts.items():
            pair = pd.concat([cand, t_shift], axis=1, join="inner").dropna()
            n = len(pair)
            out_lag = -lag  # зовнішня семантика: позитивний => кандидат веде target

            if n < req.minOverlap:
                vals[str(out_lag)] = None
                continue

            if req.method == "spearman":
                r, _, pval = _spearman_via_ranks(pair.iloc[:,0], pair.iloc[:,1])
                corr = r
                p_here = pval
            else:
                corr = pair.corr(method="pearson").iloc[0,1]
                p_here = _pearson_pvalue(corr, n)

            vals[str(out_lag)] = None if pd.isna(corr) else float(corr)

            if pd.isna(corr):
                continue
            if (pd.isna(best_corr)
                or abs(corr) > abs(best_corr)
                or (abs(corr) == abs(best_corr) and abs(lag) < abs(best_lag_internal))):
                best_corr, best_lag_internal, best_n, best_p = float(corr), int(lag), int(n), float(p_here) if p_here is not None else None

        matrix.append({"key": key, "values": vals})
        if pd.notna(best_corr):
            entry = {
                "key": key,
                "lag": -best_lag_internal,
                "value": round(best_corr, 6),
                "n": best_n
            }
            if req.returnP:
                entry["p"] = None if best_p is None else float(best_p)
            best.append(entry)

    sortedTop = sorted(best, key=lambda x: abs(x["value"]), reverse=True)
    if req.topK:
        sortedTop = sortedTop[:req.topK]

    # FDR (BH) поверх sortedTop
    if req.fdrAlpha is not None and req.returnP:
        pvals = [x.get("p") for x in sortedTop if x.get("p") is not None]
        if pvals:
            mask = _bh_fdr([float(p) for p in pvals], float(req.fdrAlpha))
            k = 0
            for x in sortedTop:
                if x.get("p") is not None:
                    x["passed_fdr"] = bool(mask[k])
                    k += 1

    t2 = perf_counter()
    first_idx = combined.first_valid_index()
    last_idx = combined.last_valid_index()
    meta = {
        "method": req.method, "minOverlap": req.minOverlap,
        "lagRange": {"min": Lmin, "max": Lmax, "ignoreZero": ignore_zero},
        "resample": _asdict(req.resample) if req.resample else None,
        "seriesCount": len(frames), "rowsCombined": int(len(combined)),
        "target": req.targetCode,
        "periodStart": None if first_idx is None else str(first_idx.date()),
        "periodEnd": None if last_idx is None else str(last_idx.date()),
        "timing": {
            "total_s": round(perf_counter() - t0, 6),
            "compute_s": round(t2 - t1, 6),
            "prep_s": round(t1 - t0, 6),
        },
        "transform": req.transform
    }

    resp = {"matrix": matrix, "bestByKey": best, "sortedTop": sortedTop, "meta": meta}
    if req.returnStats:
        resp["stats"] = _frames_stats(frames)
    return resp


# ---------- /sarimax/backtest ----------

@app.post('/sarimax/backtest')
async def sarimax_backtest(req: SarimaxBacktestRequest):
    t0 = perf_counter()

    frames = _build_frames(req.series, req.resample, req.transform)
    if not frames:
        raise HTTPException(status_code=400, detail="No valid series")

    y_full, X_full = _build_exog(frames, req.features_cfg.targetCode,
                                 req.features_cfg.features, req.features_cfg.lags)

    n = len(y_full)
    if n < req.backtest.min_train + req.backtest.horizon:
        raise HTTPException(status_code=400, detail="Not enough data for backtest")

    preds = []
    trues = []
    folds = 0

    start = req.backtest.min_train
    while start + req.backtest.horizon <= n:
        if req.backtest.expanding:
            tr_slice = slice(0, start)
        else:
            tr_slice = slice(max(0, start - req.backtest.min_train), start)
        te_slice = slice(start, start + req.backtest.horizon)

        y_tr = y_full.iloc[tr_slice]
        X_tr = None if X_full is None else X_full.iloc[tr_slice]
        y_te = y_full.iloc[te_slice]
        X_te = None if X_full is None else X_full.iloc[te_slice]

        fit = _fit_one(y_tr, X_tr, req.train)
        res = fit["res"]

        fc = res.get_forecast(steps=len(y_te), exog=X_te)
        yhat = fc.predicted_mean
        preds.extend(yhat.values.tolist())
        trues.extend(y_te.values.tolist())

        folds += 1
        start += req.backtest.step

    metrics = {
        "MAE": _mae(trues, preds),
        "RMSE": _rmse(trues, preds),
        "sMAPE": _smape(trues, preds)
    }

    meta = {"n_obs": n, "folds": folds, "timing_s": round(perf_counter()-t0, 6)}
    return {"metrics": metrics, "meta": meta}


# ---------- /sarimax/forecast ----------

@app.post('/sarimax/forecast')
async def sarimax_forecast(req: SarimaxForecastRequest):
    t0 = perf_counter()

    frames = _build_frames(req.series, req.resample, req.transform)
    if not frames:
        raise HTTPException(status_code=400, detail="No valid series")

    y, X = _build_exog(frames, req.features_cfg.targetCode,
                       req.features_cfg.features, req.features_cfg.lags)

    fit = _fit_one(y, X, req.train)
    res = fit["res"]

    steps = int(req.horizon)

    # Якщо модель тренувалась з exog, для forecast потрібне future exog.
    # Використаємо простий hold-last як наївний варіант (можеш замінити на власні future features).
    if X is not None:
        last_row = X.iloc[[-1]]
        future_index = pd.period_range(y.index[-1].to_period('M') + 1, periods=steps, freq='M').to_timestamp('M')
        X_future = pd.concat([last_row] * steps, ignore_index=True)
        X_future.index = future_index
        fc = res.get_forecast(steps=steps, exog=X_future)
    else:
        fc = res.get_forecast(steps=steps)

    mean = fc.predicted_mean

    out = {
        "order": fit["order"],
        "seasonal_order": fit["seasonal_order"],
        "aic": fit["aic"],
        "fitted_end": str(y.index[-1].date()),
    }

    if req.return_pi:
        conf = fc.conf_int(alpha=req.alpha)
        out["forecast"] = [
            {
                "date": str(mean.index[i].date()),
                "mean": float(mean.iloc[i]),
                "lo": float(conf.iloc[i, 0]),
                "hi": float(conf.iloc[i, 1]),
            }
            for i in range(len(mean))
        ]
    else:
        out["forecast"] = [
            {"date": str(mean.index[i].date()), "mean": float(mean.iloc[i])}
            for i in range(len(mean))
        ]

    out["meta"] = {"timing_s": round(perf_counter()-t0, 6)}
    return out
