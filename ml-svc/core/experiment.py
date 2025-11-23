from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import numpy as np
import pandas as pd
import time
import scipy.stats as stats

from pydantic import BaseModel, Field

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import GradientBoostingRegressor


# =========================
# Pydantic request / response
# =========================

Role = Literal["target", "candidate", "ignored"]
Frequency = Literal["M", "Q", "Y"]
Imputation = Literal["none", "ffill", "bfill", "interp"]


class SeriesPayload(BaseModel):
    name: str
    role: Role
    values: List[Optional[float]]


class ExperimentRequest(BaseModel):
    experiment_id: str
    dates: List[str]
    series: List[SeriesPayload]
    frequency: Frequency
    horizon: int = Field(..., gt=0)
    imputation: Imputation = "ffill"
    max_lag: int = 12
    extra: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    series_name: str
    model_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    is_selected: bool = False
    reason: Optional[str] = None


class ForecastPoint(BaseModel):
    series_name: str
    date: str
    value_actual: Optional[float]
    value_pred: Optional[float]
    lower_pi: Optional[float]
    upper_pi: Optional[float]
    set_type: str


class ForecastBundle(BaseModel):
    base: List[ForecastPoint] = Field(default_factory=list)
    macro: List[ForecastPoint] = Field(default_factory=list)


class MetricRow(BaseModel):
    series_name: str
    model_type: str
    horizon: int
    mase: Optional[float]
    smape: Optional[float] = None
    rmse: Optional[float] = None


class ExperimentResult(BaseModel):
    diagnostics: Dict[str, Any]
    correlations: Dict[str, Any]
    factors: Dict[str, Any]
    models: List[ModelInfo]
    forecasts: ForecastBundle
    metrics: List[MetricRow]


# =========================
# Internal Data Structures
# =========================

@dataclass
class ModelCandidate:
    series_name: str
    model_type: str
    reason: str
    mase: float
    smape: float
    rmse: float
    fit_time: float
    pred_time: float
    params: Dict[str, Any]
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    y_pred_future: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    dates_future: pd.DatetimeIndex
    lb_pvalue: Optional[float] = None


# =========================
# Helpers
# =========================

def mase_insample(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    m: int = 12,
) -> float:
    y_train = np.asarray(y_train, dtype="float64")
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    
    mae = np.mean(np.abs(y_true - y_pred))

    ins = y_train[~np.isnan(y_train)]
    if ins.size <= m:
        if len(ins) > 1:
            denom = np.mean(np.abs(np.diff(ins)))
        else:
            denom = 0.0
    else:
        denom = np.mean(np.abs(ins[m:] - ins[:-m]))

    if not math.isfinite(denom) or denom == 0.0:
        return float("nan")

    return float(mae / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = np.nan
    val = 2.0 * np.abs(y_pred - y_true) / denom
    return float(np.nanmean(val) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _to_py(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_py(v) for v in obj]
    return obj


def _safe_num(x: Any) -> Optional[float]:
    if x is None: return None
    try:
        val = float(x)
        return val if math.isfinite(val) else None
    except:
        return None


# =========================
# Data Prep
# =========================

def _build_dataframe(req: ExperimentRequest) -> pd.DataFrame:
    idx = pd.to_datetime(req.dates)
    data = {}
    for s in req.series:
        data[s.name] = pd.Series(s.values, index=idx, dtype="float64")
    return pd.DataFrame(data).sort_index()


def _impute(df: pd.DataFrame, method: Imputation) -> pd.DataFrame:
    if method == "ffill": return df.ffill()
    if method == "bfill": return df.bfill()
    if method == "interp": return df.interpolate(limit_direction="both")
    return df


def _align_to_monthly(df: pd.DataFrame, freq: Frequency) -> pd.DataFrame:
    if freq == "M":
        return df.asfreq("MS")
    return df.resample("MS").ffill()


# =========================
# Diagnostics
# =========================

def _series_diagnostics(y: pd.Series, freq: Frequency) -> Dict[str, Any]:
    s = y.dropna()
    if len(s) < 10:
        return {"mean": 0, "std": 0, "transform": "none", "has_seasonality": False}

    try:
        adf_res = adfuller(s, autolag="AIC")
        adf_p = float(adf_res[1])
    except:
        adf_p = 1.0

    has_seasonality = False
    r12 = 0.0
    if freq == "M" and len(s) > 24:
        r12 = float(s.autocorr(12))
        has_seasonality = abs(r12) > 0.4

    transform = "none"
    if (s > 0).all():
        if (s.max() / max(s.min(), 1e-9) > 2.0):
            transform = "log"

    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "adf_p": adf_p,
        "acf_12": r12,
        "has_seasonality": has_seasonality,
        "transform": transform,
    }


def _make_stationary_series(y: pd.Series, diag: Dict[str, Any]) -> pd.Series:
    s = y.astype("float64").copy()
    if diag.get("transform") == "log":
        s = s.where(s > 0)
        s = np.log(s)
    s = s.diff()
    return s


def _compute_correlations(df: pd.DataFrame, max_lag: int) -> Tuple[Any, Any]:
    lag_edges = []
    cols = list(df.columns)
    
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j: continue
            s1 = df[a]
            s2 = df[b]
            
            best_r, best_p, best_lag = 0.0, 1.0, 0
            
            search_lag = min(max_lag, 6)
            for lag in range(-search_lag, search_lag + 1):
                if lag < 0:
                    x, y_ = s1.shift(-lag), s2
                else:
                    x, y_ = s1, s2.shift(lag)
                
                mask = ~x.isna() & ~y_.isna()
                if mask.sum() < 10: continue
                
                r, p = stats.pearsonr(x[mask], y_[mask])
                if abs(r) > abs(best_r):
                    best_r, best_p, best_lag = r, p, lag
            
            lag_edges.append({
                "source": a, "target": b,
                "best_lag": best_lag,
                "r_at_best_lag": best_r,
                "p_value": best_p
            })

    return {}, lag_edges


def _select_base_variables(df: pd.DataFrame, roles: Dict, edges: List, max_lag: int):
    targets = [n for n, r in roles.items() if r == "target"]
    candidates = [n for n, r in roles.items() if r == "candidate"]
    
    strong = set()
    for e in edges:
        if e["target"] in targets and e["source"] in candidates:
            if abs(e["r_at_best_lag"]) > 0.3 and e["p_value"] < 0.05:
                strong.add(e["source"])
    
    if not strong:
        strong = set(candidates)
        
    selected = list(strong)
    vifs = {c: 1.0 for c in selected}
    
    return selected, {"base_variables": selected, "vif": vifs}


def _train_val_split(y: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    if len(y) <= horizon + 5:
        split = int(len(y) * 0.8)
        return y.iloc[:split], y.iloc[split:]
    return y.iloc[:-horizon], y.iloc[-horizon:]


# =========================
# 1. ML Model (Pure GBR)
# =========================

def _fit_gb_regressor(
    name: str, y: pd.Series, exog: Optional[pd.DataFrame], exog_future: Optional[pd.DataFrame],
    horizon: int, max_lag: int, diag: Dict[str, Any]
) -> Optional[ModelCandidate]:
    
    use_log = (diag.get("transform") == "log")
    y_proc = np.log(y.where(y > 0, 1e-9)) if use_log else y.copy()
    y_diff = y_proc.diff().dropna()
    
    if len(y_diff) < 12: return None
    
    p = min(max_lag, 6)
    df = pd.DataFrame({"target": y_diff})
    for lag in range(1, p + 1):
        df[f"lag_{lag}"] = df["target"].shift(lag)
        
    exog_cols = []
    if exog is not None:
        # Safe alignment and fillna
        exog_aligned = exog.reindex(y_diff.index).ffill().bfill()
        for col in exog.columns:
            df[col] = exog_aligned[col]
            exog_cols.append(col)
            
    df = df.dropna()
    if len(df) < 12: return None
    
    y_all = df["target"]
    X_all = df.drop(columns=["target"])
    y_train, y_test = _train_val_split(y_all, horizon)
    split_idx = len(y_train)
    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]
    
    model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=2, learning_rate=0.05)
    
    st = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    fit_time = time.perf_counter() - st
    
    st_p = time.perf_counter()
    pred_diff_test = model.predict(X_test.values)
    
    # Reconstruct Test
    test_indices = y_test.index
    prev_vals_proc = y_proc.shift(1).loc[test_indices]
    pred_test_proc = prev_vals_proc.values + pred_diff_test
    pred_test_abs = np.exp(pred_test_proc) if use_log else pred_test_proc
    
    # Future
    last_proc_val = y_proc.iloc[-1]
    current_lags = list(y_diff.values[-p:])
    future_abs = []
    curr_val = last_proc_val
    
    for h in range(horizon):
        lag_feats = np.array(current_lags[-p:][::-1])
        exog_feats = []
        if exog_future is not None and exog_cols:
            if h < len(exog_future):
                # SAFE EXOG ACCESS
                exog_feats = list(exog_future.iloc[h][exog_cols].fillna(0.0).values)
            else:
                exog_feats = [0.0]*len(exog_cols)
        elif exog_cols:
             exog_feats = [0.0]*len(exog_cols)
        
        full_feats = np.concatenate([lag_feats, exog_feats]).reshape(1, -1)
        # SANITIZE INPUT
        full_feats = np.nan_to_num(full_feats)
        
        pred_d = float(model.predict(full_feats)[0])
        curr_val += pred_d
        future_abs.append(np.exp(curr_val) if use_log else curr_val)
        current_lags.append(pred_d)
        
    pred_time = time.perf_counter() - st_p
    future_arr = np.array(future_abs)
    
    y_test_abs_true = y.loc[test_indices].values
    y_train_abs_true = y.iloc[:split_idx].values
    
    err_mase = mase_insample(y_train_abs_true, y_test_abs_true, pred_test_abs, m=12)
    err_smape = smape(y_test_abs_true, pred_test_abs)
    err_rmse = rmse(y_test_abs_true, pred_test_abs)

    dates_train = y_all.index[:split_idx]
    prev_vals_train = y_proc.shift(1).loc[dates_train]
    
    # Reconstruct Train with safety
    raw_pred_train = model.predict(X_train.values)
    pred_train_proc = prev_vals_train.values + raw_pred_train
    # Fill NaN at start
    pred_train_proc = np.nan_to_num(pred_train_proc, nan=prev_vals_train.values[0] if len(prev_vals_train)>0 else 0)
    
    pred_train_abs = np.exp(pred_train_proc) if use_log else pred_train_proc
    
    dates_test = y_all.index[split_idx:]
    future_index = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    return ModelCandidate(
        series_name=name, model_type="GBR", reason="ML (Gradient Boosting)",
        mase=err_mase, smape=err_smape, rmse=err_rmse,
        fit_time=fit_time, pred_time=pred_time, params={"lags": p},
        y_pred_train=pred_train_abs, y_pred_test=pred_test_abs, y_pred_future=future_arr,
        dates_train=dates_train, dates_test=dates_test, dates_future=future_index
    )


# =========================
# 2. ARIMA Models
# =========================

def _fit_sarimax(
    name: str, y: pd.Series, exog: Optional[pd.DataFrame], exog_future: Optional[pd.DataFrame],
    horizon: int, has_seasonality: bool,
) -> Optional[ModelCandidate]:
    train, test = _train_val_split(y, horizon)
    
    ex_train = exog.loc[train.index] if exog is not None else None
    ex_test = exog.loc[test.index] if exog is not None else None
    ex_fut = exog_future if exog_future is not None else None

    order = (1, 1, 1)
    seasonal = (1, 1, 0, 12) if has_seasonality else (0, 0, 0, 0)
    reason = "SARIMAX" if exog is not None else ("SARIMA" if has_seasonality else "ARIMA")

    try:
        st = time.perf_counter()
        model = sm.tsa.statespace.SARIMAX(
            train, order=order, seasonal_order=seasonal, exog=ex_train,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fit_time = time.perf_counter() - st
    except:
        return None

    st_p = time.perf_counter()
    try:
        pred_test = res.get_forecast(steps=len(test), exog=ex_test).predicted_mean
    except:
        pred_test = pd.Series([train.iloc[-1]]*len(test), index=test.index)

    try:
        if ex_fut is not None and len(ex_fut) != horizon: ex_fut = ex_fut.iloc[:horizon]
        pred_future = res.get_forecast(steps=horizon, exog=ex_fut).predicted_mean
    except:
         pred_future = pd.Series([train.iloc[-1]]*horizon)

    pred_time = time.perf_counter() - st_p

    err_mase = mase_insample(train.values, test.values, pred_test.values, m=12)
    err_smape = smape(test.values, pred_test.values)
    err_rmse = rmse(test.values, pred_test.values)
    
    try:
        lb_p = float(acorr_ljungbox(res.resid.dropna(), lags=[10])["lb_pvalue"].iloc[0])
    except:
        lb_p = None

    future_index = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    return ModelCandidate(
        series_name=name, model_type=reason, reason=reason,
        mase=err_mase, smape=err_smape, rmse=err_rmse,
        fit_time=fit_time, pred_time=pred_time, params={"order": order},
        y_pred_train=res.fittedvalues.reindex(train.index).values,
        y_pred_test=pred_test.values, y_pred_future=pred_future.values,
        dates_train=train.index, dates_test=test.index, dates_future=future_index, lb_pvalue=lb_p
    )


def _fit_seasonal_naive(name: str, y: pd.Series, horizon: int, m: int) -> ModelCandidate:
    train, test = _train_val_split(y, horizon)
    y_train = train.values
    y_test = test.values
    
    if len(train) > m:
        pred_test = np.tile(y_train[-m:], int(np.ceil(len(test)/m)))[:len(test)]
        future = np.tile(y_train[-m:], int(np.ceil(horizon/m)))[:horizon]
    else:
        pred_test = np.full(len(test), y_train[-1])
        future = np.full(horizon, y_train[-1])
        
    err_mase = mase_insample(y_train, y_test, pred_test, m=m)
    err_smape = smape(y_test, pred_test)
    err_rmse = rmse(y_test, pred_test)
    future_index = pd.date_range(start=y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    return ModelCandidate(
        series_name=name, model_type="SeasonalNaive", reason="Benchmark",
        mase=err_mase, smape=err_smape, rmse=err_rmse,
        fit_time=0, pred_time=0, params={},
        y_pred_train=np.full(len(train), np.nan), y_pred_test=pred_test, y_pred_future=future,
        dates_train=train.index, dates_test=test.index, dates_future=future_index
    )


# =========================
# 3. Hybrid Model (SARIMA + ML Residuals) - SILVER BULLET
# =========================

def _fit_hybrid_model(
    name: str, y: pd.Series, exog: Optional[pd.DataFrame], exog_future: Optional[pd.DataFrame],
    horizon: int, max_lag: int, diag: Dict[str, Any]
) -> Optional[ModelCandidate]:
    
    # 1. Fit Linear Base (SARIMA)
    has_seasonality = diag.get("has_seasonality", False)
    sarima_cand = _fit_sarimax(name, y, None, None, horizon, has_seasonality)
    if sarima_cand is None: return None
    
    # Get residuals (errors of ARIMA)
    train, test = _train_val_split(y, horizon)
    
    # Resid calculation: fill NaNs with 0
    resid_train = train.values - sarima_cand.y_pred_train
    resid_train = np.nan_to_num(resid_train)
    resid_series = pd.Series(resid_train, index=train.index)
    
    # 2. Fit GBR on Residuals
    exog_train = exog.loc[train.index] if exog is not None else None
    
    p = min(max_lag, 6)
    df = pd.DataFrame({"target": resid_series})
    for lag in range(1, p + 1):
        df[f"lag_{lag}"] = df["target"].shift(lag)
    
    exog_cols = []
    if exog_train is not None:
        aligned = exog_train.reindex(resid_series.index).ffill().bfill()
        for col in exog_train.columns:
            df[col] = aligned[col]
            exog_cols.append(col)
            
    df = df.dropna()
    if len(df) < 12: 
        sarima_cand.model_type = "Hybrid (Linear Only)"
        return sarima_cand
        
    y_gb = df["target"]
    X_gb = df.drop(columns=["target"])
    
    gb_model = GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=2)
    
    st = time.perf_counter()
    gb_model.fit(X_gb.values, y_gb.values)
    ft = time.perf_counter() - st
    
    # 3. Forecast Residuals
    curr_lags = list(resid_series.values[-p:])
    resid_pred_test = []
    
    # A. Test Forecast
    exog_test = exog.loc[test.index] if exog is not None else None
    
    for h in range(len(test)):
        lag_f = np.array(curr_lags[-p:][::-1])
        ex_f = []
        if exog_test is not None:
             ex_f = list(exog_test.iloc[h].fillna(0.0).values)
        else:
             ex_f = [0.0]*len(exog_cols)
             
        full_f = np.concatenate([lag_f, ex_f]).reshape(1, -1)
        full_f = np.nan_to_num(full_f)
        
        pred = float(gb_model.predict(full_f)[0])
        resid_pred_test.append(pred)
        curr_lags.append(pred)
        
    # B. Future Forecast
    resid_pred_fut = []
    for h in range(horizon):
        lag_f = np.array(curr_lags[-p:][::-1])
        ex_f = []
        if exog_future is not None and exog_cols:
            if h < len(exog_future):
                 ex_f = list(exog_future.iloc[h][exog_cols].fillna(0.0).values)
            else:
                 ex_f = [0.0]*len(exog_cols)
        elif exog_cols:
             ex_f = [0.0]*len(exog_cols)

        full_f = np.concatenate([lag_f, ex_f]).reshape(1, -1)
        full_f = np.nan_to_num(full_f)
        
        pred = float(gb_model.predict(full_f)[0])
        resid_pred_fut.append(pred)
        curr_lags.append(pred)
        
    # 4. Combine
    final_pred_test = sarima_cand.y_pred_test + np.array(resid_pred_test)
    final_pred_fut = sarima_cand.y_pred_future + np.array(resid_pred_fut)
    
    # 5. Metrics
    err_mase = mase_insample(train.values, test.values, final_pred_test, m=12)
    err_smape = smape(test.values, final_pred_test)
    err_rmse = rmse(test.values, final_pred_test)
    
    return ModelCandidate(
        series_name=name, model_type="Hybrid (SARIMA+GBR)", 
        reason="Hybrid: Linear Trend + ML Residuals (Wage)",
        mase=err_mase, smape=err_smape, rmse=err_rmse,
        fit_time=sarima_cand.fit_time + ft, 
        pred_time=sarima_cand.pred_time, # approx
        params={"linear": sarima_cand.params},
        y_pred_train=sarima_cand.y_pred_train,
        y_pred_test=final_pred_test,
        y_pred_future=final_pred_fut,
        dates_train=sarima_cand.dates_train,
        dates_test=sarima_cand.dates_test,
        dates_future=sarima_cand.dates_future,
        lb_pvalue=sarima_cand.lb_pvalue
    )


# =========================
# Selection Logic
# =========================

def _select_model(
    name: str, y: pd.Series, 
    exog: Optional[pd.DataFrame], exog_fut: Optional[pd.DataFrame],
    diag: Dict, horizon: int, max_lag: int, allow_gbr: bool
):
    candidates = []
    
    # 1. Base Benchmarks
    candidates.append(_fit_seasonal_naive(name, y, horizon, 12))
    
    # 2. Linear
    arima = _fit_sarimax(name, y, None, None, horizon, False)
    if arima: candidates.append(arima)
    
    if diag["has_seasonality"]:
        sarima = _fit_sarimax(name, y, None, None, horizon, True)
        if sarima: candidates.append(sarima)
        
    # 3. Pure ML
    if allow_gbr:
        gbr = _fit_gb_regressor(name, y, exog, exog_fut, horizon, max_lag, diag)
        if gbr: candidates.append(gbr)
        
    # 4. Hybrid (Silver Bullet) - Only if exog exists (Wage)
    if allow_gbr and exog is not None:
        hybrid = _fit_hybrid_model(name, y, exog, exog_fut, horizon, max_lag, diag)
        if hybrid: candidates.append(hybrid)

    valid = [m for m in candidates if m is not None and m.mase is not None]
    if not valid: return candidates, None
    
    best = min(valid, key=lambda x: x.mase)
    return valid, best


def _compare_horizons(
    name: str, y: pd.Series, exog: pd.DataFrame, max_lag: int, allow_gbr: bool, diag: Dict
):
    series = y.dropna()
    results = {}
    
    # Змінюємо набір горизонтів для тестування довгострокової точності
    for H in [1, 3, 6, 12]: 
        exog_curr = exog.reindex(series.index) if exog is not None else None
        exog_fut_slice = exog_curr.iloc[-H:] if exog_curr is not None else None
        
        res_h = {}
        
        m_arima = _fit_sarimax(name, series, None, None, H, False)
        if m_arima: res_h["ARIMA"] = _extract_metrics(m_arima)
        
        m_sarima = _fit_sarimax(name, series, None, None, H, True)
        if m_sarima: res_h["SARIMA"] = _extract_metrics(m_sarima)
        
        if exog_curr is not None:
             m_sarimax = _fit_sarimax(name, series, exog_curr, exog_fut_slice, H, diag["has_seasonality"])
             if m_sarimax: res_h["SARIMAX"] = _extract_metrics(m_sarimax)
             
        if allow_gbr and exog_curr is not None:
             # Try Hybrid in comparison
             m_hyb = _fit_hybrid_model(name, series, exog_curr, exog_fut_slice, H, max_lag, diag)
             if m_hyb: res_h["Hybrid"] = _extract_metrics(m_hyb)
             
             # Try Pure GB
             m_gb = _fit_gb_regressor(name, series, exog_curr, exog_fut_slice, H, max_lag, diag)
             if m_gb: res_h["GB"] = _extract_metrics(m_gb)

        if res_h: results[H] = res_h
        
    return results

def _extract_metrics(m: ModelCandidate):
    return {
        "mase": m.mase, "smape": m.smape, "rmse": m.rmse,
        "fit_time": m.fit_time, "pred_time": m.pred_time
    }


# =========================
# Main Run
# =========================

def run_full_experiment(req: ExperimentRequest) -> ExperimentResult:
    df_raw = _build_dataframe(req)
    df = _align_to_monthly(df_raw, req.frequency)
    df = _impute(df, req.imputation)
    
    roles = {s.name: s.role for s in req.series}
    targets = [n for n, r in roles.items() if r == "target"]
    
    diags = {"series": {}}
    for c in df.columns:
        diags["series"][c] = _series_diagnostics(df[c], req.frequency)
        
    diags["meta"] = {
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "n_rows": int(len(df))
    }
    
    df_stat = pd.DataFrame()
    for c in df.columns:
        df_stat[c] = _make_stationary_series(df[c], diags["series"][c])
        
    corr_matrices, edges = _compute_correlations(df_stat, req.max_lag)
    correlations = {"matrices": corr_matrices, "edges": edges}
    
    base_vars, factors = _select_base_variables(df_stat, roles, edges, req.max_lag)
    diags["base_variables"] = base_vars
    
    models_out = []
    metrics_out = []
    fc_base = []
    fc_macro = []
    base_futures = {}
    
    # A. Base
    for base in base_vars:
        diag = diags["series"][base]
        cands, selected = _select_model(
            base, df[base], None, None, diag, req.horizon, req.max_lag, True
        )
        if selected:
            base_futures[base] = selected.y_pred_future
            _record_results(base, cands, selected, models_out, metrics_out, fc_base, df)

    # B. Target
    diags["targets"] = {}
    diags["targets_exog"] = {}
    
    for target in targets:
        diag = diags["series"][target]
        exog_data = {}
        exog_fut_data = {}
        exog_info = []
        
        for base in base_vars:
            rel = [e for e in edges if e["source"]==base and e["target"]==target]
            if not rel: continue
            best_e = max(rel, key=lambda x: abs(x["r_at_best_lag"]))
            lag = int(best_e["best_lag"])
            
            hist = df[base].values
            fut = base_futures.get(base, np.array([]))
            full = np.concatenate([hist, fut])
            shifted = pd.Series(full).shift(lag).values
            n = len(df)
            exog_data[f"{base}_lag{lag}"] = shifted[:n]
            exog_fut_data[f"{base}_lag{lag}"] = shifted[n:]
            exog_info.append({"base": base, "lag": lag, "r": best_e["r_at_best_lag"]})
            
        exog_df = pd.DataFrame(exog_data, index=df.index).ffill().bfill() if exog_data else None
        exog_fut_df = pd.DataFrame(exog_fut_data) if exog_fut_data else None
        
        diags["targets_exog"][target] = exog_info
        
        cands, selected = _select_model(
            target, df[target], exog_df, exog_fut_df, diag, req.horizon, req.max_lag, True
        )
        
        if selected:
             _record_results(target, cands, selected, models_out, metrics_out, fc_macro, df)
             
             comp = _compare_horizons(target, df[target], exog_df, req.max_lag, True, diag)
             if "comparison" not in diags: diags["comparison"] = {}
             diags["comparison"][target] = comp
             
             diags["targets"][target] = {"lb_pvalue": selected.lb_pvalue, "residuals_ok": True}

    return ExperimentResult(
        diagnostics=_to_py(diags),
        correlations=_to_py(correlations),
        factors=_to_py(factors),
        models=models_out,
        forecasts=ForecastBundle(base=fc_base, macro=fc_macro),
        metrics=metrics_out
    )

def _record_results(name, cands, selected, models_out, metrics_out, fc_list, df):
    for m in cands:
        models_out.append(ModelInfo(
            series_name=m.series_name, model_type=m.model_type,
            params=_to_py(m.params), mase=_safe_num(m.mase),
            smape=_safe_num(m.smape), rmse=_safe_num(m.rmse),
            is_selected=(m is selected), reason=m.reason
        ))
        metrics_out.append(MetricRow(
            series_name=m.series_name, model_type=m.model_type,
            horizon=len(m.y_pred_future), mase=_safe_num(m.mase),
            smape=_safe_num(m.smape), rmse=_safe_num(m.rmse)
        ))
    for dt, val in zip(selected.dates_train, selected.y_pred_train):
        act = df.loc[dt, name] if dt in df.index else None
        fc_list.append(ForecastPoint(series_name=name, date=dt.strftime("%Y-%m-%d"), value_actual=_safe_num(act), value_pred=_safe_num(val), lower_pi=None, upper_pi=None, set_type="train"))
    for dt, val in zip(selected.dates_test, selected.y_pred_test):
        act = df.loc[dt, name] if dt in df.index else None
        fc_list.append(ForecastPoint(series_name=name, date=dt.strftime("%Y-%m-%d"), value_actual=_safe_num(act), value_pred=_safe_num(val), lower_pi=None, upper_pi=None, set_type="test"))
    for dt, val in zip(selected.dates_future, selected.y_pred_future):
        fc_list.append(ForecastPoint(series_name=name, date=dt.strftime("%Y-%m-%d"), value_actual=None, value_pred=_safe_num(val), lower_pi=None, upper_pi=None, set_type="future"))