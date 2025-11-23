from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import numpy as np
import pandas as pd
import time

from pydantic import BaseModel, Field

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


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
    reason: Optional[str] = None  # чому ця модель обрана


class ForecastPoint(BaseModel):
    series_name: str
    date: str  # ISO YYYY-MM-DD
    value_actual: Optional[float]
    value_pred: Optional[float]
    lower_pi: Optional[float]
    upper_pi: Optional[float]
    set_type: str  # "train" | "test" | "future"


class ForecastBundle(BaseModel):
    base: List[ForecastPoint] = Field(default_factory=list)
    macro: List[ForecastPoint] = Field(default_factory=list)


class MetricRow(BaseModel):
    series_name: str
    model_type: str
    horizon: int
    mase: Optional[float]
    smape: Optional[float]
    rmse: Optional[float]


class ExperimentResult(BaseModel):
    diagnostics: Dict[str, Any]
    correlations: Dict[str, Any]
    factors: Dict[str, Any]
    models: List[ModelInfo]
    forecasts: ForecastBundle
    metrics: List[MetricRow]


# =========================
# Helpers: JSON-safe conversion
# =========================
def _make_lagged_ml(series: pd.Series, max_lag: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Лагові ознаки для ML-моделей:
    повертає (X, y), де X містить lag_1..lag_p.
    """
    s = pd.Series(series).copy()
    df_lag = pd.DataFrame({"y": s})
    for lag in range(1, max_lag + 1):
        df_lag[f"lag_{lag}"] = df_lag["y"].shift(lag)
    df_lag = df_lag.dropna()
    if df_lag.empty:
        return np.empty((0, max_lag)), np.empty((0,))
    X = df_lag.drop(columns=["y"]).values
    y = df_lag["y"].values
    return X, y

def _evaluate_target_horizons(
    name: str,
    y: pd.Series,
    exog_df: Optional[pd.DataFrame],
    max_lag: int,
    allow_gbr: bool,
    horizons: List[int],
) -> Dict[str, Any]:
    """
    Colab-style порівняння моделей на горизонтах H=1,2,3.
    Для кожного H:
      - ARIMA
      - SARIMA
      - GB (Gradient Boosting)
      - RF (Random Forest)
      - SARIMAX (комбінований спосіб з exog)
    Рахує MASE, sMAPE, RMSE, час навчання та прогнозу.
    Повертає dict: {H: {family: {mase, smape, rmse, fit_time, pred_time}}}
    """
    series = y.dropna()
    results: Dict[int, Any] = {}
    if len(series) < 30:
        return results

    exog_aligned: Optional[pd.DataFrame] = None
    if exog_df is not None:
        exog_aligned = exog_df.reindex(series.index)

    for H in horizons:
        if H <= 0:
            continue
        # щоб було нормальне розділення train/test
        if len(series) <= H + max_lag + 5:
            continue

        y_train_full = series.iloc[:-H]
        y_test = series.iloc[-H:]
        insample = y_train_full.values

        if exog_aligned is not None:
            ex_train = exog_aligned.iloc[:-H]
            ex_future = exog_aligned.iloc[-H:]
        else:
            ex_train = None
            ex_future = None

        fam_res: Dict[str, Any] = {}

        # === ARIMA ===
        try:
            start = time.perf_counter()
            model_arima = sm.tsa.statespace.SARIMAX(
                y_train_full,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                exog=None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fit_t = time.perf_counter() - start

            start = time.perf_counter()
            fc = model_arima.get_forecast(steps=H)
            pred = np.asarray(fc.predicted_mean, dtype="float64")
            pred_t = time.perf_counter() - start

            fam_res["ARIMA"] = {
                "mase": mase_insample(insample, y_test.values, pred, m=1),
                "smape": smape(y_test.values, pred),
                "rmse": rmse(y_test.values, pred),
                "fit_time": fit_t,
                "pred_time": pred_t,
            }
        except Exception:
            pass

        # === SARIMA ===
        try:
            start = time.perf_counter()
            model_sarima = sm.tsa.statespace.SARIMAX(
                y_train_full,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                exog=None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fit_t = time.perf_counter() - start

            start = time.perf_counter()
            fc = model_sarima.get_forecast(steps=H)
            pred = np.asarray(fc.predicted_mean, dtype="float64")
            pred_t = time.perf_counter() - start

            fam_res["SARIMA"] = {
                "mase": mase_insample(insample, y_test.values, pred, m=1),
                "smape": smape(y_test.values, pred),
                "rmse": rmse(y_test.values, pred),
                "fit_time": fit_t,
                "pred_time": pred_t,
            }
        except Exception:
            pass

        # === GB (Gradient Boosting) ===
        if allow_gbr:
            try:
                X_ml, y_ml = _make_lagged_ml(y_train_full, max_lag=max_lag)
                if y_ml.size >= 30:
                    gb_model = GradientBoostingRegressor(
                        random_state=42,
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                    )
                    start = time.perf_counter()
                    gb_model.fit(X_ml, y_ml)
                    fit_t = time.perf_counter() - start

                    start = time.perf_counter()
                    pred = _recursive_forecast_ml(
                        gb_model, y_train_full, horizon=H, max_lag=max_lag
                    )
                    pred_t = time.perf_counter() - start

                    fam_res["GB"] = {
                        "mase": mase_insample(insample, y_test.values, pred, m=1),
                        "smape": smape(y_test.values, pred),
                        "rmse": rmse(y_test.values, pred),
                        "fit_time": fit_t,
                        "pred_time": pred_t,
                    }
            except Exception:
                pass

            # === RF (Random Forest) ===
            try:
                X_ml, y_ml = _make_lagged_ml(y_train_full, max_lag=max_lag)
                if y_ml.size >= 30:
                    rf_model = RandomForestRegressor(
                        random_state=42,
                        n_estimators=400,
                        max_depth=None,
                        n_jobs=-1,
                    )
                    start = time.perf_counter()
                    rf_model.fit(X_ml, y_ml)
                    fit_t = time.perf_counter() - start

                    start = time.perf_counter()
                    pred = _recursive_forecast_ml(
                        rf_model, y_train_full, horizon=H, max_lag=max_lag
                    )
                    pred_t = time.perf_counter() - start

                    fam_res["RF"] = {
                        "mase": mase_insample(insample, y_test.values, pred, m=1),
                        "smape": smape(y_test.values, pred),
                        "rmse": rmse(y_test.values, pred),
                        "fit_time": fit_t,
                        "pred_time": pred_t,
                    }
            except Exception:
                pass

        # === SARIMAX (комбінований спосіб з exog) ===
        if ex_train is not None and ex_future is not None:
            try:
                start = time.perf_counter()
                model_sarimax = sm.tsa.statespace.SARIMAX(
                    y_train_full,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    exog=ex_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                fit_t = time.perf_counter() - start

                start = time.perf_counter()
                fc = model_sarimax.get_forecast(steps=H, exog=ex_future)
                pred = np.asarray(fc.predicted_mean, dtype="float64")
                pred_t = time.perf_counter() - start

                fam_res["SARIMAX"] = {
                    "mase": mase_insample(insample, y_test.values, pred, m=1),
                    "smape": smape(y_test.values, pred),
                    "rmse": rmse(y_test.values, pred),
                    "fit_time": fit_t,
                    "pred_time": pred_t,
                }
            except Exception:
                pass

        if fam_res:
            results[H] = fam_res

    return results


def _recursive_forecast_ml(
    model,
    history_series: pd.Series,
    horizon: int,
    max_lag: int = 12,
) -> np.ndarray:
    """
    Рекурсивний багатокроковий прогноз для ML-моделей:
    на кожному кроці беремо останні max_lag значень (з урахуванням уже спрогнозованих).
    """
    hist_vals = np.asarray(history_series, dtype="float64").copy()
    preds: List[float] = []
    if len(hist_vals) < max_lag:
        raise ValueError("Not enough history for given max_lag")

    for _ in range(horizon):
        lags = [hist_vals[-lag] for lag in range(1, max_lag + 1)]
        X_new = np.array(lags).reshape(1, -1)
        y_hat = float(model.predict(X_new)[0])
        preds.append(y_hat)
        hist_vals = np.append(hist_vals, y_hat)

    return np.asarray(preds, dtype="float64")

def mase_insample(
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    m: int = 1,
) -> float:
    """
    MASE зі шкалою, порахованою на in-sample (train):
    - y_train — повний тренувальний ряд;
    - y_true / y_pred — значення на тесті.
    """
    y_train = np.asarray(y_train, dtype="float64")
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    # чисельник: MAE на тесті
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    num = np.mean(np.abs(y_true - y_pred))

    # знаменник: середня абсолютна зміна на train з лагом m
    ins = y_train[~np.isnan(y_train)]
    if ins.size <= m:
        return float("nan")

    denom = np.mean(np.abs(ins[m:] - ins[:-m]))
    if not math.isfinite(denom) or denom == 0.0:
        return float("nan")

    return float(num / denom)

def _make_stationary_series(
    y: pd.Series,
    diag: Dict[str, Any],
    freq: Frequency,
) -> pd.Series:
    """
    Стаціонаризація ряду для кроків 2–3:
    - логарифмування, якщо обрано transform='log';
    - різницювання за трендом (d=1), якщо has_trend=True;
    - сезонне різницювання (D=1) при наявності сезонності.
    """
    s = y.astype("float64").copy()

    # лог-трансформація
    if diag.get("transform") == "log":
        s = s.where(s > 0)
        s = np.log(s)

    # трендова різниця
    if diag.get("has_trend"):
        s = s.diff()

    # сезонна різниця
    if diag.get("has_seasonality"):
        if freq == "M":
            m = 12
        elif freq == "Q":
            m = 4
        else:
            m = 1
        if m > 1:
            s = s.diff(m)

    return s

def _to_py(obj: Any) -> Any:
    """
    Рекурсивно перетворює numpy-типи на стандартні Python-типи
    й прибирає невалідні числа (NaN, ±inf -> None).
    """
    # numpy-скаляри
    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return val if math.isfinite(val) else None
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val

    # звичайні float’и
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # dict
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}

    # списки / тьюпли / множини
    if isinstance(obj, (list, tuple, set)):
        return [_to_py(v) for v in obj]

    return obj


def _safe_num(x: Any) -> Optional[float]:
    """Повертає звичайний finite float або None."""
    if x is None:
        return None
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


# =========================
# Error metrics
# =========================

def mase(y_true: np.ndarray, y_pred: np.ndarray, m: int = 1) -> float:
    """Mean Absolute Scaled Error (без ±inf)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    num = np.mean(np.abs(y_true - y_pred))
    if len(y_true) <= m:
        return float("nan")
    denom = np.mean(np.abs(y_true[m:] - y_true[:-m]))
    if denom == 0:
        return float("nan")
    return float(num / denom)


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


# =========================
# Data preparation
# =========================

def _build_dataframe(req: ExperimentRequest) -> pd.DataFrame:
    idx = pd.to_datetime(req.dates)
    data: Dict[str, pd.Series] = {}
    for s in req.series:
        arr = pd.Series(s.values, index=idx, dtype="float64")
        data[s.name] = arr
    df = pd.DataFrame(data).sort_index()
    return df


def _impute(df: pd.DataFrame, method: Imputation) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "ffill":
        return df.ffill()
    if method == "bfill":
        return df.bfill()
    if method == "interp":
        return df.interpolate(limit_direction="both")
    return df


def _disagg_sum_preserving(ts: pd.Series, freq_in: Frequency) -> pd.Series:
    """
    Перетворює квартальні / річні сумарні значення у місячні
    зі збереженням підсумків (розмазування рівними частинами).
    """
    ts = ts.dropna()
    if ts.empty:
        return ts.asfreq("MS")

    monthly_idx = pd.date_range(
        ts.index.min().to_period("M").start_time,
        ts.index.max().to_period("M").end_time,
        freq="MS",
    )
    monthly = ts.resample("MS").ffill().reindex(monthly_idx)

    if freq_in == "M":
        return monthly

    if freq_in == "Q":
        periods = ts.index.to_period("Q")
    elif freq_in == "Y":
        periods = ts.index.to_period("Y")
    else:
        return monthly

    for dt, val, per in zip(ts.index, ts.values, periods):
        m_start = per.start_time.to_period("M").to_timestamp()
        m_end = per.end_time.to_period("M").to_timestamp()
        mask = (monthly.index >= m_start) & (monthly.index <= m_end)
        n = mask.sum()
        if n > 0:
            monthly.loc[mask] = float(val) / float(n)

    return monthly


def _align_to_monthly(df: pd.DataFrame, freq: Frequency) -> pd.DataFrame:
    if freq == "M":
        return df.asfreq("MS")
    cols: Dict[str, pd.Series] = {}
    for name in df.columns:
        cols[name] = _disagg_sum_preserving(df[name], freq)
    out = pd.DataFrame(cols)
    return out


# =========================
# Diagnostics & correlations
# =========================

def _series_diagnostics(
    y: pd.Series,
    freq: Frequency,
    horizon: int,
) -> Dict[str, Any]:
    s = y.dropna()
    if len(s) < 10:
        return {
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
            "adf_p": None,
            "kpss_p": None,
            "has_trend": None,
            "has_seasonality": None,
            "transform": "none",
            "is_nonlinear": False,
        }

    # ADF
    try:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    except Exception:
        adf_p = None

    # KPSS
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_p = None

    # простий індикатор сезонності = автокореляція на лаг 12
    has_seasonality = False
    r12 = None
    if freq == "M" and len(s) > 24:
        r12 = float(s.autocorr(12))
        has_seasonality = abs(r12) > 0.4

    has_trend = None
    if adf_p is not None and kpss_p is not None:
        # грубий критерій: ADF не відкидає H0 (є одиничний корінь) + KPSS відкидає H0
        has_trend = (adf_p > 0.1) and (kpss_p < 0.05)

    # евристика трансформації
    transform = "none"
    if (s > 0).all() and (s.max() / max(s.min(), 1e-9) > 3.0):
        transform = "log"

    # груба оцінка нелінійності через різницю Пірсон / Спірман для лагу 1
    is_nonlinear = False
    try:
        lag1 = s.shift(1)
        mask = ~lag1.isna() & ~s.isna()
        if mask.sum() >= 10:
            x = lag1[mask]
            y2 = s[mask]
            pearson = float(np.corrcoef(x, y2)[0, 1])
            xr = x.rank()
            yr = y2.rank()
            spearman = float(np.corrcoef(xr, yr)[0, 1])
            if not (math.isnan(pearson) or math.isnan(spearman)):
                is_nonlinear = abs(spearman) - abs(pearson) > 0.2
    except Exception:
        is_nonlinear = False

    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "adf_p": float(adf_p) if adf_p is not None else None,
        "kpss_p": float(kpss_p) if kpss_p is not None else None,
        "acf_12": r12,
        "has_trend": has_trend,
        "has_seasonality": has_seasonality,
        "transform": transform,
        "is_nonlinear": is_nonlinear,
    }


def _compute_correlations(
    df: pd.DataFrame,
    max_lag: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    lag_edges: List[Dict[str, Any]] = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                continue
            s1 = df[a]
            s2 = df[b]
            best_r = 0.0
            best_lag = 0
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    x = s1.shift(-lag)
                    y = s2
                else:
                    x = s1
                    y = s2.shift(lag)
                mask = ~x.isna() & ~y.isna()
                if mask.sum() < 6:
                    continue
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                if abs(r) > abs(best_r):
                    best_r = r
                    best_lag = lag
            lag_edges.append(
                {
                    "source": a,
                    "target": b,
                    "best_lag": best_lag,
                    "r_at_best_lag": float(best_r) if not np.isnan(best_r) else None,
                }
            )

    corr_struct = {
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }
    return corr_struct, lag_edges


# =========================
# Base variables (VIF + PCA)
# =========================

def _compute_vif(df: pd.DataFrame) -> Dict[str, float]:
    if df.shape[1] <= 1:
        return {c: 1.0 for c in df.columns}
    X = sm.add_constant(df.values)
    vifs: Dict[str, float] = {}
    for i, col in enumerate(df.columns, start=1):  # пропускаємо константу
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = float("nan")
        vifs[col] = float(v)
    return vifs


def _select_base_variables(
    df: pd.DataFrame,
    roles: Dict[str, Role],
    lag_edges: List[Dict[str, Any]],
    max_lag: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Формування базових змінних згідно з методом:
    - беремо candidate-ряди, що мають суттєву кореляцію з таргетами;
    - усуваємо мультиколінеарність за VIF;
    - застосовуємо PCA для зменшення розмірності (зберігаємо лише інфо для діагностики).
    """
    targets = [n for n, r in roles.items() if r == "target"]
    candidates = [n for n, r in roles.items() if r == "candidate"]

    strong_candidates = set()
    for e in lag_edges:
        if e["target"] in targets and e["source"] in candidates:
            r = e["r_at_best_lag"]
            if r is not None and abs(r) >= 0.4:
                strong_candidates.add(e["source"])

    if not strong_candidates:
        strong_candidates = set(candidates)

    base_df = df[list(strong_candidates)].dropna()
    if base_df.empty:
        return [], {"base_variables": [], "vif": {}, "pca": {}}

    # VIF-ітерація
    selected = list(base_df.columns)
    vifs = _compute_vif(base_df[selected])
    while True:
        worst = max(selected, key=lambda c: vifs.get(c, 0.0))
        if vifs.get(worst, 0.0) <= 10.0:
            break
        if len(selected) <= 2:
            break
        selected.remove(worst)
        vifs = _compute_vif(base_df[selected])

    stability_counts = {col: 0 for col in base_df.columns}
    n_iter = 20
    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        sample = base_df.sample(
            frac=0.8,
            replace=False,
            random_state=int(rng.randint(0, 1e9)),
        )
        if sample.shape[0] < 10:
            continue

        sub_selected = list(sample.columns)
        sub_vifs = _compute_vif(sample[sub_selected])
        while True:
            worst_sub = max(sub_selected, key=lambda c: sub_vifs.get(c, 0.0))
            if sub_vifs.get(worst_sub, 0.0) <= 10.0 or len(sub_selected) <= 2:
                break
            sub_selected.remove(worst_sub)
            sub_vifs = _compute_vif(sample[sub_selected])

        for col in sub_selected:
            stability_counts[col] += 1

    stability = {
        col: stability_counts[col] / max(1, n_iter) for col in stability_counts
    }

    # PCA (на стандартизованих даних)
    X = (base_df[selected] - base_df[selected].mean()) / base_df[selected].std(ddof=0)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    pca_res: Dict[str, Any] = {}
    if not X.empty:
        pca = PCA()
        pca.fit(X.values)
        pca_res = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": [
                {col: float(val) for col, val in zip(selected, comp)}
                for comp in pca.components_.tolist()
            ],
        }

    factors_info = {
        "base_variables": selected,
        "vif": vifs,
        "pca": pca_res,
        "stability": stability,
    }
    return selected, factors_info


# =========================
# Model candidates
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
    lb_pvalue: Optional[float] = None  # тільки для ARIMA-подібних



def _train_val_split(
    y: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    if len(y) <= horizon + 5:
        split = int(len(y) * 0.67)
        return y.iloc[:split], y.iloc[split:]
    return y.iloc[:-horizon], y.iloc[-horizon:]

def _fit_rf_regressor(
    name: str,
    y: pd.Series,
    horizon: int,
    max_lag: int,
) -> Optional[ModelCandidate]:
    """
    Random Forest із лагами цільового ряду:
    X_t = [y_{t-1},...,y_{t-p}], p = min(max_lag, 12).
    Використовується як baseline-модель.
    """
    p = min(max_lag, 12)
    df = pd.DataFrame({"y": y})
    for lag in range(1, p + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()

    if len(df) < 30:
        return None

    y_all = df["y"]
    X_all = df.drop(columns=["y"])

    y_train, y_test = _train_val_split(y_all, horizon)
    split_idx = len(y_train)
    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]

    model = RandomForestRegressor(
        random_state=42,
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
    )

    # Час навчання
    start_fit = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    fit_time = time.perf_counter() - start_fit

    # Час прогнозу (test + future)
    start_pred = time.perf_counter()
    pred_test = model.predict(X_test.values)

    # autorecursive прогноз у майбутнє
    last_vals = list(y.values[-p:])
    future: List[float] = []
    for _ in range(horizon):
        features = np.array(last_vals[-p:][::-1])  # [y_{t-1},...,y_{t-p}]
        pred = float(model.predict(features.reshape(1, -1))[0])
        future.append(pred)
        last_vals.append(pred)
    future_arr = np.array(future, dtype="float64")

    pred_time = time.perf_counter() - start_pred

    err_mase = mase_insample(y_train.values, y_test.values, pred_test, m=12)
    err_smape = smape(y_test.values, pred_test)
    err_rmse = rmse(y_test.values, pred_test)

    idx_all = y_all.index
    dates_train = idx_all[:split_idx]
    dates_test = idx_all[split_idx:]

    last_date = y.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    return ModelCandidate(
        series_name=name,
        model_type="RF",
        reason="baseline: Random Forest з лагами цільового ряду",
        mase=err_mase,
        smape=err_smape,
        rmse=err_rmse,
        fit_time=fit_time,
        pred_time=pred_time,
        params={"lags": p},
        y_pred_train=model.predict(X_train.values),
        y_pred_test=pred_test,
        y_pred_future=future_arr,
        dates_train=dates_train,
        dates_test=dates_test,
        dates_future=future_index,
    )

def _fit_seasonal_naive(
    name: str,
    y: pd.Series,
    horizon: int,
    m: int,
) -> ModelCandidate:
    """
    Бенчмарк-модель (сезонний наївний прогноз).
    НЕ входить до комбінованого методу, використовується тільки для порівняння.
    """
    train, test = _train_val_split(y, horizon)
    y_train = train.values
    y_test = test.values

    # backtest
    if len(train) > m:
        pred_test = y_train[-m:].tolist()
        while len(pred_test) < len(test):
            pred_test.extend(pred_test[-m:])
        pred_test = np.array(pred_test[: len(test)])
    else:
        pred_test = np.repeat(y_train[-1], len(test))

    # майбутнє (для повноти, зараз у backtest не використовуємо)
    future_vals = pred_test[-m:].tolist()
    while len(future_vals) < horizon:
        future_vals.extend(future_vals[-m:])
    future = np.array(future_vals[:horizon])

    err_mase = mase_insample(train.values, y_test, pred_test, m=m)
    err_smape = smape(y_test, pred_test)
    err_rmse = rmse(y_test, pred_test)

    last_date = y.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    return ModelCandidate(
        series_name=name,
        model_type="SeasonalNaive",
        reason="benchmark: seasonal naive",
        mase=err_mase,
        smape=err_smape,
        rmse=err_rmse,
        fit_time=0.0,
        pred_time=0.0,
        params={"m": m},
        y_pred_train=np.full(len(train), np.nan),
        y_pred_test=pred_test,
        y_pred_future=future,
        dates_train=train.index,
        dates_test=test.index,
        dates_future=future_index,
        lb_pvalue=None,
    )


def _fit_sarimax(
    name: str,
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    horizon: int,
    has_seasonality: bool,
) -> Optional[ModelCandidate]:
    train, test = _train_val_split(y, horizon)

    if exog is not None:
        ex_train = exog.loc[train.index]
        ex_test = exog.loc[test.index]
        # для майбутнього – беремо останні horizon значень exog (для реального forecast)
        ex_future = exog.iloc[-horizon:]
        if len(ex_future) < horizon:
            ex_future = None
    else:
        ex_train = ex_test = ex_future = None

    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)
    reason = "ARIMA (лінійна динаміка)"

    if has_seasonality:
        seasonal_order = (1, 0, 1, 12)
        reason = "SARIMA (регулярна сезонність)"
        if exog is not None:
            reason = "SARIMAX з екзогенними змінними"

    try:
        start_fit = time.perf_counter()
        model = sm.tsa.statespace.SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            exog=ex_train,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fit_time = time.perf_counter() - start_fit
    except Exception:
        return None

    # backtest (прогноз на тестовий відрізок)
    start_pred = time.perf_counter()
    pred_test = res.get_forecast(steps=len(test), exog=ex_test)
    mean_test = pred_test.predicted_mean

    # майбутнє (не відображаємо у backtest, але залишаємо в структурі)
    last_date = y.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    if ex_future is not None and len(ex_future) == horizon:
        pred_future = res.get_forecast(steps=horizon, exog=ex_future)
    else:
        pred_future = res.get_forecast(steps=horizon)
    mean_future = pred_future.predicted_mean
    pred_time = time.perf_counter() - start_pred

    err_mase = mase_insample(train.values, test.values, mean_test.values, m=12)
    err_smape = smape(test.values, mean_test.values)
    err_rmse = rmse(test.values, mean_test.values)

    # Ljung–Box по залишках
    try:
        resid = res.resid
        lb = acorr_ljungbox(resid.dropna(), lags=[min(12, len(resid) - 1)])
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = None

    model_type = (
        "SARIMAX" if exog is not None
        else "SARIMA" if has_seasonality
        else "ARIMA"
    )

    return ModelCandidate(
        series_name=name,
        model_type=model_type,
        reason=reason,
        mase=err_mase,
        smape=err_smape,
        rmse=err_rmse,
        fit_time=fit_time,
        pred_time=pred_time,
        params={
            "order": order,
            "seasonal_order": seasonal_order,
        },
        y_pred_train=res.fittedvalues.reindex(train.index).values,
        y_pred_test=mean_test.values,
        y_pred_future=mean_future.values,
        dates_train=train.index,
        dates_test=test.index,
        dates_future=future_index,
        lb_pvalue=lb_p,
    )

def _fit_gb_regressor(
    name: str,
    y: pd.Series,
    horizon: int,
    max_lag: int,
) -> Optional[ModelCandidate]:
    """
    GradientBoosting з лагами:
    X_t = [y_{t-1},...,y_{t-p}], p = min(max_lag, 12).
    Використовується тільки за вираженої нелінійності (is_nonlinear=True).
    """
    p = min(max_lag, 12)
    df = pd.DataFrame({"y": y})
    for lag in range(1, p + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()

    if len(df) < 30:
        return None

    y_all = df["y"]
    X_all = df.drop(columns=["y"])

    y_train, y_test = _train_val_split(y_all, horizon)
    split_idx = len(y_train)
    X_train = X_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]

    model = GradientBoostingRegressor(random_state=42)

    start_fit = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    fit_time = time.perf_counter() - start_fit

    start_pred = time.perf_counter()
    pred_test = model.predict(X_test.values)

    # прогноз у майбутнє (autorecursive)
    last_vals = list(y.values[-p:])
    future = []
    for _ in range(horizon):
        features = np.array(last_vals[-p:][::-1])  # [y_{t-1},...,y_{t-p}]
        pred = model.predict(features.reshape(1, -1))[0]
        future.append(pred)
        last_vals.append(pred)
    future = np.array(future)
    pred_time = time.perf_counter() - start_pred

    err_mase = mase_insample(y_train.values, y_test.values, pred_test, m=12)
    err_smape = smape(y_test.values, pred_test)
    err_rmse = rmse(y_test.values, pred_test)

    idx_all = y_all.index
    dates_train = idx_all[:split_idx]
    dates_test = idx_all[split_idx:]

    last_date = y.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    return ModelCandidate(
        series_name=name,
        model_type="GBR",
        reason="GradientBoosting (нелінійна модель з лагами)",
        mase=err_mase,
        smape=err_smape,
        rmse=err_rmse,
        fit_time=fit_time,
        pred_time=pred_time,
        params={"lags": p},
        y_pred_train=model.predict(X_train.values),
        y_pred_test=pred_test,
        y_pred_future=future,
        dates_train=dates_train,
        dates_test=dates_test,
        dates_future=future_index,
    )


def _select_model_family_and_candidates(
    name: str,
    y: pd.Series,
    exog_for_sarimax: Optional[pd.DataFrame],
    diag: Dict[str, Any],
    horizon: int,
    max_lag: int,
    allow_gbr: bool,
) -> Tuple[List[ModelCandidate], Optional[ModelCandidate]]:
    """
    Реалізація кроків 5–6 методу для одного ряду:
    - будуємо бенчмарк seasonal naive;
    - за діагностикою обираємо клас (ARIMA/SARIMA/SARIMAX або GBR);
    - MASE / sMAPE / RMSE рахуються для оцінки, але клас задає діагностика.
    """
    models: List[ModelCandidate] = []

    # 0) Бенчмарк
    m_season = 12
    seasonal_naive = _fit_seasonal_naive(name, y, horizon, m_season)
    models.append(seasonal_naive)

    # 1) Лінійна модель
    has_seasonality = bool(diag.get("has_seasonality"))
    linear_model = _fit_sarimax(
        name=name,
        y=y,
        exog=exog_for_sarimax,
        horizon=horizon,
        has_seasonality=has_seasonality,
    )
    if linear_model is not None:
        models.append(linear_model)

    # 2) Нелінійна модель (за потреби)
    nonlinear_model: Optional[ModelCandidate] = None
    is_nonlinear = bool(diag.get("is_nonlinear"))
    if allow_gbr and is_nonlinear:
        nonlinear_model = _fit_gb_regressor(
            name=name,
            y=y,
            horizon=horizon,
            max_lag=max_lag,
        )
        if nonlinear_model is not None:
            models.append(nonlinear_model)

    # ---- Вибір сімейства згідно з методом ----
    selected: Optional[ModelCandidate] = None

    if is_nonlinear and nonlinear_model is not None:
        # Виражена нелінійність → GBR, якщо він зійшовся
        selected = nonlinear_model
    elif linear_model is not None:
        # Лінійна динаміка → ARIMA/SARIMA/SARIMAX
        selected = linear_model
    elif nonlinear_model is not None:
        # fallback: хоч щось
        selected = nonlinear_model
    else:
        # зовсім поганий кейс → бенчмарк
        selected = seasonal_naive

    # SeasonalNaive – тільки якщо взагалі немає інших
    if selected is seasonal_naive and (linear_model or nonlinear_model):
        selected = linear_model or nonlinear_model

    return models, selected

def _log_target_comparison(
    target_name: str,
    horizon: int,
    candidates: List[ModelCandidate],
    our_model: Optional[ModelCandidate],
) -> None:
    """
    Консольний лог порівняння моделей для цільового ряду:
    SeasonalNaive, baseline ARIMA/SARIMA, нелінійні, НАШ МЕТОД (SARIMAX + exog).
    Аналог таблички з Colab.
    """
    if not candidates:
        return

    print(f"\n=== Цільовий ряд: {target_name} (горизонт {horizon}) ===")
    print("{:<22s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s}".format(
        "Модель", "RMSE", "sMAPE", "MASE", "fit[s]", "pred[s]"
    ))

    for m in candidates:
        rmse_val = m.rmse if m.rmse is not None else float("nan")
        smape_val = m.smape if m.smape is not None else float("nan")
        mase_val = m.mase if m.mase is not None else float("nan")
        print("{:<22s} {:>8.3f} {:>8.2f} {:>8.3f} {:>9.3f} {:>9.3f}".format(
            m.model_type,
            rmse_val,
            smape_val,
            mase_val,
            m.fit_time,
            m.pred_time,
        ))

    if our_model is None or our_model.mase is None or not math.isfinite(our_model.mase):
        return

    print(f"\nПоліпшення нашого методу ({our_model.model_type}) за MASE:")
    for m in candidates:
        if m is our_model or m.mase is None or not math.isfinite(m.mase) or m.mase <= 0:
            continue
        improvement = 100.0 * (m.mase - our_model.mase) / m.mase
        print(f"  vs {m.model_type:18s}: {improvement:6.2f} %")



# =========================
# Main experiment procedure
# =========================

def run_full_experiment(req: ExperimentRequest) -> ExperimentResult:
    """
    Повна реалізація комбінованого методу (8 кроків), узгоджена з текстом дисертації.
    Повертаємо:
      - діагностику рядів та базових змінних;
      - кореляції та лаги;
      - факторний блок (VIF, PCA, стабільність);
      - моделі (усі кандидати + позначена обрана);
      - backtest-прогнози (train/test) для базових і таргетних рядів;
      - colab-style порівняння моделей для таргетів на горизонтах H=1,2,3.
    """

    # 1. Приведення до єдиної періодичності (місячна шкала)
    df_raw = _build_dataframe(req)
    df_m = _align_to_monthly(df_raw, req.frequency)
    df_m = _impute(df_m, req.imputation)

    roles: Dict[str, Role] = {s.name: s.role for s in req.series}
    targets = [n for n, r in roles.items() if r == "target"]

    # 2. Діагностика рядів у вихідній шкалі
    diagnostics: Dict[str, Any] = {"series": {}}
    for name in df_m.columns:
        diagnostics["series"][name] = _series_diagnostics(
            df_m[name],
            req.frequency,
            req.horizon,
        )

    # 2b. Стаціонарні ряди для кореляцій / факторного аналізу
    df_stat = pd.DataFrame(
        {
            name: _make_stationary_series(
                df_m[name],
                diagnostics["series"][name],
                req.frequency,
            )
            for name in df_m.columns
        }
    )

    # meta-інформація про період даних
    if len(df_m.index) > 0:
        diagnostics["meta"] = {
            "start": df_m.index.min().strftime("%Y-%m-%d"),
            "end": df_m.index.max().strftime("%Y-%m-%d"),
            "n_rows": int(len(df_m)),
        }
    else:
        diagnostics["meta"] = {
            "start": None,
            "end": None,
            "n_rows": 0,
        }

    # 2c. Кореляції + лаги (на df_stat)
    corr_struct, lag_edges = _compute_correlations(df_stat, req.max_lag)
    correlations = {
        "matrices": corr_struct,
        "edges": lag_edges,
    }

    # 3. Вибір базових змінних (candidate → base) + VIF/PCA/стабільність
    base_vars, factors_info = _select_base_variables(
        df_stat,
        roles,
        lag_edges,
        req.max_lag,
    )

    # 4. Статистична обробка базових змінних (зберігаємо список)
    diagnostics["base_variables"] = base_vars

    models: List[ModelInfo] = []
    metrics: List[MetricRow] = []
    forecast_base: List[ForecastPoint] = []
    forecast_macro: List[ForecastPoint] = []

    allow_gbr = bool(req.extra.get("allow_gbr", True))

    # 5–6. Прогнозні моделі для базових змінних (без exog), тільки backtest
    base_df = df_m[base_vars].copy() if base_vars else pd.DataFrame(index=df_m.index)

    for name in base_vars:
        y = base_df[name].astype("float64")
        diag = diagnostics["series"][name]

        cand_models, selected = _select_model_family_and_candidates(
            name=name,
            y=y,
            exog_for_sarimax=None,  # базові ряди без екзогенних
            diag=diag,
            horizon=req.horizon,
            max_lag=req.max_lag,
            allow_gbr=allow_gbr,
        )
        if not cand_models or selected is None:
            continue

        # 5–6a. Зберігаємо всі кандидатні моделі + позначаємо обрану
        for m in cand_models:
            models.append(
                ModelInfo(
                    series_name=m.series_name,
                    model_type=m.model_type,
                    params=_to_py(m.params),
                    mase=_safe_num(m.mase),
                    smape=_safe_num(m.smape),
                    rmse=_safe_num(m.rmse),
                    is_selected=(m is selected),
                    reason=m.reason,
                )
            )
            metrics.append(
                MetricRow(
                    series_name=m.series_name,
                    model_type=m.model_type,
                    horizon=req.horizon,
                    mase=_safe_num(m.mase),
                    smape=_safe_num(m.smape),
                    rmse=_safe_num(m.rmse),
                )
            )

        # 5–6b. Backtest-прогнози для базових (train + test)
        # train
        for date, val_pred in zip(selected.dates_train, selected.y_pred_train):
            actual = float(df_m.loc[date, name]) if date in df_m.index else None
            forecast_base.append(
                ForecastPoint(
                    series_name=name,
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=actual,
                    value_pred=_safe_num(val_pred),
                    lower_pi=None,
                    upper_pi=None,
                    set_type="train",
                )
            )
        # test
        for date, val_pred in zip(selected.dates_test, selected.y_pred_test):
            actual = float(df_m.loc[date, name]) if date in df_m.index else None
            forecast_base.append(
                ForecastPoint(
                    series_name=name,
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=actual,
                    value_pred=_safe_num(val_pred),
                    lower_pi=None,
                    upper_pi=None,
                    set_type="test",
                )
            )

    # 7. Прогноз цільових рядів із використанням базових змінних як exog
    for t_name in targets:
        y = df_m[t_name].astype("float64")
        diag_t = diagnostics["series"][t_name]
        has_seasonality = bool(diag_t.get("has_seasonality"))

        # 7.1. Формуємо екзогенні регресори з базових змінних (з урахуванням лагів)
        exog_cols: Dict[str, pd.Series] = {}
        for base in base_vars:
            rel_edges = [
                e for e in lag_edges
                if e["source"] == base and e["target"] == t_name
            ]
            if not rel_edges:
                continue
            best_edge = max(
                rel_edges,
                key=lambda e: abs(e["r_at_best_lag"] or 0.0),
            )
            lag = best_edge["best_lag"]
            s = df_m[base]
            if lag > 0:
                exog_cols[f"{base}_lag{lag}"] = s.shift(lag)
            else:
                # lag <= 0 — використовуємо поточне значення
                exog_cols[f"{base}_lag0"] = s

        exog_df = (
            pd.DataFrame(exog_cols).astype("float64") if exog_cols else None
        )

        # 7.2. Кандидатні моделі для таргетного ряду
        cand_models: List[ModelCandidate] = []

        # (a) Бенчмарк: сезонний наївний
        benchmark = _fit_seasonal_naive(t_name, y, req.horizon, m=12)
        cand_models.append(benchmark)

        # (b) Базові лінійні моделі без exog: ARIMA та SARIMA
        arima_baseline = _fit_sarimax(
            name=t_name,
            y=y,
            exog=None,
            horizon=req.horizon,
            has_seasonality=False,
        )
        if arima_baseline is not None:
            arima_baseline.model_type = "ARIMA"
            arima_baseline.reason = "baseline: ARIMA без сезонності та exog"
            cand_models.append(arima_baseline)

        sarima_baseline: Optional[ModelCandidate] = None
        if has_seasonality:
            sarima_baseline = _fit_sarimax(
                name=t_name,
                y=y,
                exog=None,
                horizon=req.horizon,
                has_seasonality=True,
            )
            if sarima_baseline is not None:
                sarima_baseline.model_type = "SARIMA"
                sarima_baseline.reason = "baseline: SARIMA без exog"
                cand_models.append(sarima_baseline)

        # (c) Дерева рішень із лагами цільового ряду: GBR та RF
        gb_model: Optional[ModelCandidate] = None
        rf_model: Optional[ModelCandidate] = None
        if allow_gbr:
            gb_model = _fit_gb_regressor(
                name=t_name,
                y=y,
                horizon=req.horizon,
                max_lag=req.max_lag,
            )
            if gb_model is not None:
                gb_model.model_type = "GBR"
                gb_model.reason = (
                    "baseline: Gradient Boosting з лагами цільового ряду"
                )
                cand_models.append(gb_model)

            rf_model = _fit_rf_regressor(
                name=t_name,
                y=y,
                horizon=req.horizon,
                max_lag=req.max_lag,
            )
            if rf_model is not None:
                # reason вже заданий у _fit_rf_regressor
                cand_models.append(rf_model)

        # (d) Комбінований спосіб: SARIMAX із відібраними базовими змінними (наш пайплайн)
        method_model = _fit_sarimax(
            name=t_name,
            y=y,
            exog=exog_df,
            horizon=req.horizon,
            has_seasonality=has_seasonality,
        )
        if method_model is not None:
            # ВАЖЛИВО: model_type лишається "SARIMAX".
            # Комбінований спосіб = увесь пайплайн, а не одна модель.
            method_model.model_type = "SARIMAX"
            method_model.reason = (
                "комбінований спосіб: таргет із базовими змінними як exogenous"
            )
            cand_models.append(method_model)

        if not cand_models:
            continue

        # 7.3. Обрана модель у термінах методу:
        # комбінований спосіб → SARIMA → ARIMA → дерева → бенчмарк
        selected = (
            method_model
            or sarima_baseline
            or arima_baseline
            or gb_model
            or rf_model
            or benchmark
        )

        # 7.4. Зберігаємо всі моделі + позначаємо обрану
        for m in cand_models:
            models.append(
                ModelInfo(
                    series_name=m.series_name,
                    model_type=m.model_type,
                    params=_to_py(m.params),
                    mase=_safe_num(m.mase),
                    smape=_safe_num(m.smape),
                    rmse=_safe_num(m.rmse),
                    is_selected=(m is selected),
                    reason=m.reason,
                )
            )
            metrics.append(
                MetricRow(
                    series_name=m.series_name,
                    model_type=m.model_type,
                    horizon=req.horizon,
                    mase=_safe_num(m.mase),
                    smape=_safe_num(m.smape),
                    rmse=_safe_num(m.rmse),
                )
            )

        # 7.5. Ljung–Box для обраної моделі
        if "targets" not in diagnostics:
            diagnostics["targets"] = {}
        diagnostics["targets"][t_name] = {
            "lb_pvalue": _safe_num(selected.lb_pvalue),
            "residuals_ok": bool(
                selected.lb_pvalue is not None and selected.lb_pvalue > 0.05
            ),
        }

        # 7.6. Backtest-прогнози таргета (train + test)
        for date, val_pred in zip(selected.dates_train, selected.y_pred_train):
            actual = float(df_m.loc[date, t_name]) if date in df_m.index else None
            forecast_macro.append(
                ForecastPoint(
                    series_name=t_name,
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=actual,
                    value_pred=_safe_num(val_pred),
                    lower_pi=None,
                    upper_pi=None,
                    set_type="train",
                )
            )

        for date, val_pred in zip(selected.dates_test, selected.y_pred_test):
            actual = float(df_m.loc[date, t_name]) if date in df_m.index else None
            forecast_macro.append(
                ForecastPoint(
                    series_name=t_name,
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=actual,
                    value_pred=_safe_num(val_pred),
                    lower_pi=None,
                    upper_pi=None,
                    set_type="test",
                )
            )

        # 7.7. Colab-style порівняння моделей на горизонтах H = 1,2,3
        comparison = _evaluate_target_horizons(
            name=t_name,
            y=y,
            exog_df=exog_df,
            max_lag=req.max_lag,
            allow_gbr=allow_gbr,
            horizons=[1, 2, 3],
        )
        if comparison:
            if "comparison" not in diagnostics:
                diagnostics["comparison"] = {}
            diagnostics["comparison"][t_name] = comparison

        # (опційно) консольний лог: табличка як у Colab
        _log_target_comparison(
            target_name=t_name,
            horizon=req.horizon,
            candidates=cand_models,
            our_model=method_model,
        )

    forecasts = ForecastBundle(base=forecast_base, macro=forecast_macro)

    # 8. JSON-safe конвертація
    diagnostics_py = _to_py(diagnostics)
    correlations_py = _to_py(correlations)
    factors_py = _to_py(factors_info)

    return ExperimentResult(
        diagnostics=diagnostics_py,
        correlations=correlations_py,
        factors=factors_py,
        models=models,
        forecasts=forecasts,
        metrics=metrics,
    )
