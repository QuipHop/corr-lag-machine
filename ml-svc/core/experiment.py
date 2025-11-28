from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import math
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =========================
# Типи запиту/відповіді
# =========================

Role = Literal["target", "candidate", "ignored"]
Frequency = Literal["M", "Q", "Y"]
Imputation = Literal["none", "ffill", "bfill", "interp"]


class SeriesPayload(BaseModel):
    name: str
    role: Role
    values: List[Optional[float]]


class ExperimentRequest(BaseModel):
    # від Nest
    experiment_id: Optional[str] = None
    dates: Optional[List[str]] = None

    series: List[SeriesPayload]
    frequency: Frequency = "M"
    horizon: int = 12
    imputation: Imputation = "ffill"
    max_lag: int = 12
    start_date: Optional[str] = None  # якщо dates немає
    extra: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    # camelCase для прямої роботи з Nest/React
    seriesName: str
    modelType: str
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    fit_time: Optional[float] = None
    pred_time: Optional[float] = None
    isSelected: bool = False


class ForecastPoint(BaseModel):
    seriesName: str
    date: str
    setType: Literal["train", "test", "future"]
    valueActual: Optional[float] = None
    valuePred: Optional[float] = None


class MetricInfo(BaseModel):
    seriesName: str
    modelType: str
    horizon: int
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None


class ExperimentResult(BaseModel):
    id: str = Field(default_factory=lambda: "exp-" + str(int(time.time())))
    diagnostics: Dict[str, Any]
    correlations: Dict[str, Any]
    factors: Dict[str, Any]
    models: List[ModelInfo]
    forecasts: List[ForecastPoint]
    metrics: List[MetricInfo] = Field(default_factory=list)

def _to_python(obj: Any) -> Any:
    """
    Рекурсивно перетворює numpy/pandas-типи на звичайні Python-скаляри,
    щоб Pydantic міг їх серіалізувати.
    """
    # --- скаляри numpy / python ---
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    # --- numpy масиви ---
    if isinstance(obj, np.ndarray):
        return [_to_python(v) for v in obj.tolist()]

    # --- pandas дати / періоди ---
    if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Period)):
        return obj.isoformat()

    # --- pandas Index (включно з RangeIndex, Int64Index, тощо у pandas 2.x) ---
    if isinstance(obj, pd.Index):
        return [_to_python(v) for v in obj.tolist()]

    # --- dict / list / tuple рекурсивно ---
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_python(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_to_python(v) for v in obj)

    # все інше лишаємо як є (str, float, int, None, bool, ...)
    return obj

def _detect_nonlinearity(x: pd.Series) -> bool:
    """
    Грубий детектор монотонної нелінійності.
    Ідея: якщо Spearman по лагу 1 суттєво відрізняється від Pearson – є нелінійність.
    """
    x = x.dropna()
    if len(x) < 30:
        return False

    x_lag = x.shift(1).dropna()
    x_cur = x.iloc[1:len(x)]
    x_cur = x_cur.loc[x_lag.index]

    if len(x_cur) < 20:
        return False

    try:
        pearson = x_cur.corr(x_lag, method="pearson")
        spearman = x_cur.corr(x_lag, method="spearman")
    except Exception:
        return False

    if pearson is None or spearman is None:
        return False
    if math.isnan(pearson) or math.isnan(spearman):
        return False

    # якщо Spearman значно більший за Pearson — монотонна нелінійність
    return abs(spearman - pearson) > 0.15 and abs(spearman) > abs(pearson)


# =========================
# 1–2. Dataframe + діагностика
# =========================

def _build_dataframe(req: ExperimentRequest) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    max_len = max(len(s.values) for s in req.series)
    data: Dict[str, List[Optional[float]]] = {}
    for sp in req.series:
        vals = list(sp.values)
        if len(vals) < max_len:
            vals += [None] * (max_len - len(vals))
        data[sp.name] = vals

    # Якщо прийшли дати з фронту — юзаємо їх
    if req.dates:
        idx = pd.to_datetime(req.dates[:max_len])
    else:
        if req.start_date:
            idx = pd.date_range(start=req.start_date, periods=max_len, freq=req.frequency)
        else:
            idx = pd.date_range(start="2000-01-01", periods=max_len, freq=req.frequency)

    df = pd.DataFrame(data, index=idx)

    meta = {
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
        "n_rows": int(len(df)),
        "frequency": req.frequency,
    }

    return df, meta


def _impute_df(df: pd.DataFrame, mode: Imputation) -> pd.DataFrame:
    if mode == "none":
        return df
    if mode == "ffill":
        return df.ffill()
    if mode == "bfill":
        return df.bfill()
    if mode == "interp":
        return df.interpolate(limit_direction="both")
    return df


def _adf_test(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    try:
        res = adfuller(x, autolag="AIC")
        return float(res[1])
    except Exception:
        return float("nan")


def _kpss_test(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 10:
        return float("nan")
    try:
        res = kpss(x, regression="c", nlags="auto")
        return float(res[1])
    except Exception:
        return float("nan")


def _detect_seasonality(x: pd.Series, freq: Frequency) -> bool:
    x = x.dropna()
    if len(x) < 24:
        return False
    acf_vals = sm.tsa.stattools.acf(x, nlags=24, fft=True)
    if freq == "M":
        s_lag = 12
    elif freq == "Q":
        s_lag = 4
    else:
        s_lag = 1
    if s_lag < len(acf_vals) and abs(acf_vals[s_lag]) > 0.3:
        return True
    return False


def _acf_at_lag(x: pd.Series, lag: int) -> float:
    x = x.dropna()
    if len(x) <= lag:
        return float("nan")
    acf_vals = sm.tsa.stattools.acf(x, nlags=lag, fft=True)
    return float(acf_vals[lag])


def _format_transform(info: Dict[str, Any]) -> str:
    """Людське представлення трансформацій: log/diff/seas."""
    parts: List[str] = []
    if info.get("log"):
        parts.append("log")
    if info.get("diff", 0) > 0:
        parts.append(f"diff({info['diff']})")
    if info.get("seas_diff", 0) > 0:
        parts.append(f"seas({info['seas_diff']})")
    if not parts:
        return "none"
    return " + ".join(parts)


def _compute_series_diagnostics(
    df: pd.DataFrame,
    freq: Frequency,
    trans_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Базова діагностика рядів:
    - mean, std
    - ADF/KPSS
    - наявність сезонності
    - ACF(12)
    - skew/kurtosis
    - індикатор нелінійності (форма розподілу + Spearman vs Pearson)
    - трансформації (log/diff/seas) з _make_stationary
    """
    out: Dict[str, Any] = {}
    for name in df.columns:
        s = df[name].dropna()
        if s.empty:
            out[name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "adf_p": float("nan"),
                "kpss_p": float("nan"),
                "has_seasonality": False,
                "acf_12": float("nan"),
                "skew": float("nan"),
                "kurtosis": float("nan"),
                "transform": "none",
                "is_nonlinear": False,
            }
            continue

        mean = float(s.mean())
        std = float(s.std())
        adf_p = _adf_test(s)
        kpss_p = _kpss_test(s)
        has_seas = _detect_seasonality(s, freq)
        acf12 = _acf_at_lag(s, 12) if freq == "M" else float("nan")

        skew = float(s.skew())
        kurt = float(s.kurtosis())  # ексцес

        # 1) форма розподілу
        nonlinear_shape = (abs(skew) > 1.0) or (kurt > 3.5)
        # 2) Spearman vs Pearson по лагу 1
        nonlinear_spearman = _detect_nonlinearity(s)

        is_nonlinear = nonlinear_shape or nonlinear_spearman

        transform_label = "none"
        if trans_info and name in trans_info:
            transform_label = _format_transform(trans_info[name])

        out[name] = {
            "mean": mean,
            "std": std,
            "adf_p": adf_p,
            "kpss_p": kpss_p,
            "has_seasonality": has_seas,
            "acf_12": acf12,
            "skew": skew,
            "kurtosis": kurt,
            "transform": transform_label,
            "is_nonlinear": is_nonlinear,
            # Якщо хочеш дебажити — можна також вивести:
            # "nonlinear_shape": nonlinear_shape,
            # "nonlinear_spearman": nonlinear_spearman,
        }
    return out


def _make_stationary(df: pd.DataFrame, freq: Frequency) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Проста схема: log (якщо всі > 0) + перша різниця при нестабільності рівня +
    сезонна різниця при сезонності.
    """
    trans_info: Dict[str, Any] = {}
    df_tr = df.copy()

    for name in df.columns:
        s = df[name]
        info = {
            "log": False,
            "diff": 0,
            "seas_diff": 0,
        }

        s_tr = s.astype(float)

        if (s_tr > 0).all():
            s_tr = np.log(s_tr)
            info["log"] = True

        adf_p = _adf_test(s_tr)
        kpss_p = _kpss_test(s_tr)

        # Перша різниця за потреби
        if (not math.isnan(adf_p) and adf_p > 0.05) or (not math.isnan(kpss_p) and kpss_p < 0.05):
            s_tr = s_tr.diff()
            info["diff"] = 1

        # Сезонна різниця
        has_seas = _detect_seasonality(s_tr.dropna(), freq)
        if has_seas:
            if freq == "M":
                lag = 12
            elif freq == "Q":
                lag = 4
            else:
                lag = 1
            if lag > 1:
                s_tr = s_tr.diff(lag)
                info["seas_diff"] = lag

        df_tr[name] = s_tr
        trans_info[name] = info

    return df_tr, trans_info


# =========================
# 3–4. Кореляції, базові змінні, VIF
# =========================

def _cross_correlation(
    x: pd.Series, y: pd.Series, max_lag: int = 12
) -> Dict[str, Any]:
    """
    Рахуємо Pearson та Spearman на різних лагах.
    Для вибору "силу зв'язку" беремо ту метрику (r чи rho), де |кореляція| більша.
    """
    x = x.dropna()
    y = y.dropna()
    if len(x) < 10 or len(y) < 10:
        return {
            "best_lag": 0,
            "r_pearson": float("nan"),
            "rho_spearman": float("nan"),
            "r_at_best_lag": float("nan"),
            "metric": "pearson",
        }

    best_lag = 0
    best_val = 0.0
    best_r = float("nan")
    best_rho = float("nan")
    best_metric = "pearson"

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xs = x.iloc[lag:]
            ys = y.iloc[:-lag]
        elif lag < 0:
            xs = x.iloc[:lag]
            ys = y.iloc[-lag:]
        else:
            xs = x
            ys = y
        if len(xs) < 10 or len(ys) < 10:
            continue

        r = xs.corr(ys, method="pearson")
        rho = xs.corr(ys, method="spearman")

        if r is None or math.isnan(r):
            r = float("nan")
        if rho is None or math.isnan(rho):
            rho = float("nan")

        # беремо ту, де |.| більше
        cand_val = 0.0
        cand_metric = "pearson"
        if not math.isnan(r) and (math.isnan(rho) or abs(r) >= abs(rho)):
            cand_val = float(r)
            cand_metric = "pearson"
        elif not math.isnan(rho):
            cand_val = float(rho)
            cand_metric = "spearman"
        else:
            continue

        if abs(cand_val) > abs(best_val):
            best_val = cand_val
            best_metric = cand_metric
            best_lag = lag
            best_r = float(r)
            best_rho = float(rho)

    return {
        "best_lag": best_lag,
        "r_pearson": best_r,
        "rho_spearman": best_rho,
        "r_at_best_lag": best_val,
        "metric": best_metric,
    }


def _compute_correlations(
    df_st: pd.DataFrame,
    targets: List[str],
    candidates: List[str],
    max_lag: int = 12,
) -> Dict[str, Any]:
    edges: List[Dict[str, Any]] = []

    for t in targets:
        for c in candidates:
            if c == t:
                continue
            res = _cross_correlation(df_st[c], df_st[t], max_lag=max_lag)
            if math.isnan(res["r_at_best_lag"]):
                continue
            edges.append(
                {
                    "source": c,
                    "target": t,
                    "best_lag": res["best_lag"],
                    "r_pearson": res["r_pearson"],
                    "rho_spearman": res["rho_spearman"],
                    "r_at_best_lag": res["r_at_best_lag"],
                    "metric": res["metric"],
                }
            )

    return {"edges": edges}


def _select_base_variables(corr: Dict[str, Any], threshold: float = 0.3) -> List[str]:
    edges = corr.get("edges", [])
    base_vars = set()
    for e in edges:
        if abs(e["r_at_best_lag"]) >= threshold:
            base_vars.add(e["source"])
    return sorted(base_vars)


def _compute_vif(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(cols) < 2:
        for c in cols:
            out[c] = float("nan")
        return out

    X = df[cols].dropna()
    if X.empty:
        for c in cols:
            out[c] = float("nan")
        return out

    X = sm.add_constant(X)
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = float("nan")
        out[c] = float(vif)

    return out


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray, m: int = 1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae_model = np.mean(np.abs(y_true - y_pred))

    if len(y_insample) <= m:
        return float("inf")
    naive_diff = np.abs(y_insample[m:] - y_insample[:-m])
    mae_naive = np.mean(naive_diff)
    if mae_naive == 0:
        return float("inf")
    return float(mae_model / mae_naive)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


# =========================
# 5. Walk-forward backtest
# =========================

@dataclass
class FamilyResult:
    mase: float
    smape: float
    rmse: float
    fit_time: float
    pred_time: float


def _seasonal_naive_forecast(
    y: pd.Series, horizon: int, seasonal_period: int
) -> np.ndarray:
    if len(y) < seasonal_period:
        return np.repeat(y.iloc[-1], horizon)
    history = y.values
    out = []
    for h in range(horizon):
        out.append(history[-seasonal_period + (h % seasonal_period)])
    return np.array(out)


def _walk_forward_backtest(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    freq: Frequency,
    horizons: List[int],
) -> Dict[int, Dict[str, FamilyResult]]:
    """
    {horizon: {family: FamilyResult}}
    families: SeasonalNaive, ARIMA, SARIMA, SARIMAX, RF, GB
    """
    res: Dict[int, Dict[str, FamilyResult]] = {h: {} for h in horizons}
    y = y.dropna()
    if len(y) < 40:
        return res

    if freq == "M":
        season = 12
    elif freq == "Q":
        season = 4
    else:
        season = 1

    n = len(y)
    split_idx = int(n * 0.7)
    if split_idx + max(horizons) + 5 >= n:
        split_idx = n - max(horizons) - 5

    y_train_full = y.iloc[:split_idx]
    y_test_full = y.iloc[split_idx:]

    if exog is not None:
        exog = exog.loc[y.index]
    y_insample = y_train_full.values

    for h in horizons:
        fam_err: Dict[str, Dict[str, List[float]]] = {
            "SeasonalNaive": {"y": [], "yhat": []},
            "ARIMA": {"y": [], "yhat": []},
            "SARIMA": {"y": [], "yhat": []},
            "SARIMAX": {"y": [], "yhat": []},
            "RF": {"y": [], "yhat": []},
            "GB": {"y": [], "yhat": []},
        }
        fam_fit_time: Dict[str, float] = {k: 0.0 for k in fam_err}
        fam_pred_time: Dict[str, float] = {k: 0.0 for k in fam_err}

        max_steps = min(5, len(y_test_full) - h)
        if max_steps <= 0:
            continue

        for step in range(max_steps):
            end_train = split_idx + step
            start_test = end_train
            end_test = end_train + h

            y_train = y.iloc[:end_train]
            y_test = y.iloc[start_test:end_test]

            if exog is not None:
                exog_train = exog.iloc[:end_train]
                exog_test = exog.iloc[start_test:end_test]
            else:
                exog_train = None
                exog_test = None

            # SeasonalNaive
            t0 = time.time()
            yhat_sn = _seasonal_naive_forecast(y_train, h, season)
            fam_fit_time["SeasonalNaive"] += time.time() - t0
            fam_pred_time["SeasonalNaive"] += 0.0
            fam_err["SeasonalNaive"]["y"].extend(y_test.values.tolist())
            fam_err["SeasonalNaive"]["yhat"].extend(yhat_sn.tolist())

            # ARIMA
            try:
                t0 = time.time()
                arima_model = SARIMAX(
                    y_train,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    exog=exog_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                fam_fit_time["ARIMA"] += time.time() - t0

                t1 = time.time()
                yhat_arima = arima_model.forecast(steps=h, exog=exog_test)
                fam_pred_time["ARIMA"] += time.time() - t1
                fam_err["ARIMA"]["y"].extend(y_test.values.tolist())
                fam_err["ARIMA"]["yhat"].extend(yhat_arima.tolist())
            except Exception:
                pass

            # SARIMA
            try:
                t0 = time.time()
                sarima_model = SARIMAX(
                    y_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, season),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                fam_fit_time["SARIMA"] += time.time() - t0

                t1 = time.time()
                yhat_sarima = sarima_model.forecast(steps=h)
                fam_pred_time["SARIMA"] += time.time() - t1
                fam_err["SARIMA"]["y"].extend(y_test.values.tolist())
                fam_err["SARIMA"]["yhat"].extend(yhat_sarima.tolist())
            except Exception:
                pass

            # SARIMAX
            if exog_train is not None:
                try:
                    t0 = time.time()
                    sarimax_model = SARIMAX(
                        y_train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, season),
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                    fam_fit_time["SARIMAX"] += time.time() - t0

                    t1 = time.time()
                    yhat_sarimax = sarimax_model.forecast(steps=h, exog=exog_test)
                    fam_pred_time["SARIMAX"] += time.time() - t1
                    fam_err["SARIMAX"]["y"].extend(y_test.values.tolist())
                    fam_err["SARIMAX"]["yhat"].extend(yhat_sarimax.tolist())
                except Exception:
                    pass

            # RF / GB на лагових ознаках
            max_lag_rf = min(12, len(y_train) - 1)
            if max_lag_rf <= 0:
                continue

            lagged: Dict[str, pd.Series] = {}
            for l in range(1, max_lag_rf + 1):
                lagged[f"lag_{l}"] = y.shift(l)

            X_all = pd.DataFrame(lagged, index=y.index)
            if exog is not None:
                X_all = pd.concat([X_all, exog], axis=1)

            # тренувальна частина: до end_train, потім дропаємо NaN
            X_train = X_all.iloc[:end_train].dropna()
            y_train_rf = y.loc[X_train.index]

            X_test = X_all.iloc[start_test:end_test]
            if X_test.isna().any().any() or X_test.empty:
                continue

            y_test_rf = y.iloc[start_test:end_test]

            # RF
            try:
                t0 = time.time()
                rf = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    random_state=0,
                )
                rf.fit(X_train, y_train_rf)
                fam_fit_time["RF"] += time.time() - t0

                t1 = time.time()
                yhat_rf = rf.predict(X_test)
                fam_pred_time["RF"] += time.time() - t1

                fam_err["RF"]["y"].extend(y_test_rf.values.tolist())
                fam_err["RF"]["yhat"].extend(yhat_rf.tolist())
            except Exception:
                pass

            # GB
            try:
                t0 = time.time()
                gb = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=0,
                )
                gb.fit(X_train, y_train_rf)
                fam_fit_time["GB"] += time.time() - t0

                t1 = time.time()
                yhat_gb = gb.predict(X_test)
                fam_pred_time["GB"] += time.time() - t1

                fam_err["GB"]["y"].extend(y_test_rf.values.tolist())
                fam_err["GB"]["yhat"].extend(yhat_gb.tolist())
            except Exception:
                pass

        # агрегація метрик
        for fam, store in fam_err.items():
            if len(store["y"]) == 0:
                continue
            y_true = np.asarray(store["y"])
            y_hat = np.asarray(store["yhat"])
            mse = mean_squared_error(y_true, y_hat)
            rmse_val = float(math.sqrt(mse))
            mase_val = mase(y_true, y_hat, y_insample, m=season)
            smape_val = smape(y_true, y_hat)

            res[h][fam] = FamilyResult(
                mase=mase_val,
                smape=smape_val,
                rmse=rmse_val,
                fit_time=fam_fit_time[fam],
                pred_time=fam_pred_time[fam],
            )

    return res


# =========================
# 6–7. Повний експеримент
# =========================

def run_full_experiment(req: ExperimentRequest) -> ExperimentResult:
    df_raw, meta = _build_dataframe(req)
    df_imp = _impute_df(df_raw, req.imputation)

    targets = [s.name for s in req.series if s.role == "target"]
    candidates = [s.name for s in req.series if s.role == "candidate"]
    ignored = [s.name for s in req.series if s.role == "ignored"]

    # 1–2. Діагностика + стаціонаризація
    df_st, trans_info = _make_stationary(df_imp, req.frequency)
    series_diag = _compute_series_diagnostics(df_imp, req.frequency, trans_info)

    # 3. Кореляції й базові змінні
    correlations = _compute_correlations(df_st, targets, candidates)
    base_vars = _select_base_variables(correlations, threshold=0.3)
    vif = _compute_vif(df_imp, base_vars) if base_vars else {}

    factors = {"vif": vif}

    # 5. Walk-forward backtest для порівняння сімейств (але НЕ для вибору класу)
    horizons = [1, 2, 3]
    comparison: Dict[str, Any] = {}
    models: List[ModelInfo] = []
    targets_diag: Dict[str, Any] = {}
    targets_exog: Dict[str, Any] = {}
    forecasts: List[ForecastPoint] = []
    selection_info: Dict[str, Any] = {}  # для пояснення вибору на ЮІ

    for t in targets:
        # --- Визначаємо exog для таргету ---
        exog_cols = [b for b in base_vars if b != t]
        exog_df = df_imp[exog_cols] if exog_cols else None
        has_exog = bool(exog_cols)

        # --- Беремо діагностику ряду ---
        s_info = series_diag.get(t, {})
        has_seasonality = bool(s_info.get("has_seasonality", False))
        is_nonlinear = bool(s_info.get("is_nonlinear", False))

        # --- Backtest для всіх сімейств (для порівняння й метрик) ---
        fam_results = _walk_forward_backtest(
            y=df_imp[t],
            exog=exog_df,
            freq=req.frequency,
            horizons=horizons,
        )

        comparison[t] = {}
        for h in horizons:
            comparison[t][h] = {}
            for fam, fr in fam_results.get(h, {}).items():
                comparison[t][h][fam] = {
                    "mase": fr.mase,
                    "smape": fr.smape,
                    "rmse": fr.rmse,
                    "fit_time": fr.fit_time,
                    "pred_time": fr.pred_time,
                }

        # --- 4. Середні MASE по сімействам (для override) ---
        avg_mase: Dict[str, float] = {}
        for fam in ["SeasonalNaive", "ARIMA", "SARIMA", "SARIMAX", "RF", "GB"]:
            vals = []
            for h in horizons:
                fr = fam_results.get(h, {}).get(fam)
                if fr is not None and not math.isinf(fr.mase):
                    vals.append(fr.mase)
            if vals:
                avg_mase[fam] = float(np.mean(vals))

        # --- 5. RULE-BASED вибір (як було) ---
        # 5.1. За замовчуванням – ARIMA-сімейство
        chosen_family = "ARIMA"
        chosen_rule = "linear_arima"  # для ЮІ

        if is_nonlinear:
            # Клас дерев рішень. Конкретну модель беремо за MASE з backtest.
            tree_candidates: Dict[str, float] = {}
            for fam in ["RF", "GB"]:
                if fam in avg_mase:
                    tree_candidates[fam] = avg_mase[fam]

            if tree_candidates:
                chosen_family = min(tree_candidates.items(), key=lambda kv: kv[1])[0]
                chosen_rule = "nonlinear_trees"
            else:
                # якщо дерева чомусь не порахувались – падаємо назад у лінійну гілку нижче
                is_nonlinear = False  # щоб спрацювали лінійні правила

        # Лінійна динаміка / fallback
        if not is_nonlinear:
            if has_exog:
                chosen_family = "SARIMAX"
                chosen_rule = "linear_with_exog_sarimax"
            elif has_seasonality:
                chosen_family = "SARIMA"
                chosen_rule = "linear_seasonal_sarima"
            else:
                chosen_family = "ARIMA"
                chosen_rule = "linear_arima"

        # --- 6. Якщо для chosen_family немає жодного результату backtest ---
        if not any(fam_results.get(h, {}).get(chosen_family) is not None for h in horizons):
            if avg_mase:
                # fallback: будь-яке сімейство з мінімальним середнім MASE
                chosen_family = min(avg_mase.items(), key=lambda kv: kv[1])[0]
                chosen_rule = "fallback_best_mase"

        # --- 7. OVERRIDE: якщо backtest явно каже, що інше сімейство кращe ---
        if avg_mase:
            chosen_mase = avg_mase.get(chosen_family, float("inf"))
            best_fam, best_mase_val = min(avg_mase.items(), key=lambda kv: kv[1])

            # якщо хтось кращий за поточного хоча б на 20%
            if math.isfinite(chosen_mase) and best_mase_val < chosen_mase * 0.8:
                chosen_family = best_fam
                if best_fam in ["RF", "GB"]:
                    chosen_rule = "override_backtest_trees"
                    is_nonlinear = True  # щоб у діагностиці відображалось чесно
                else:
                    chosen_rule = "override_backtest_linear"

        # --- 8. Зберігаємо інфу для ЮІ (чому так) ---
        selection_info[t] = {
            "has_exog": has_exog,
            "has_seasonality": has_seasonality,
            "is_nonlinear": is_nonlinear,
            "chosen_family": chosen_family,
            "rule": chosen_rule,
        }

        # СИНХРОНІЗУЄМО діагностику для таблиці 1:
        if chosen_family in ["RF", "GB"]:
            if t in series_diag:
                series_diag[t]["is_nonlinear"] = True

    
        # Перетворюємо chosen_family у modelType (як у БД)
        if chosen_family == "GB":
            model_type = "GBR"
        else:
            model_type = chosen_family

        # Беремо середній MASE по горизонтах для вибраного сімейства (для таблиці 5)
        mase_vals = []
        for h in horizons:
            fr = fam_results.get(h, {}).get(chosen_family)
            if fr is not None and not math.isinf(fr.mase):
                mase_vals.append(fr.mase)
        best_mase = float(np.mean(mase_vals)) if mase_vals else float("inf")

        fr_h1 = fam_results.get(1, {}).get(chosen_family)
        smape_best = fr_h1.smape if fr_h1 else None
        rmse_best = fr_h1.rmse if fr_h1 else None
        fit_time_best = fr_h1.fit_time if fr_h1 else None
        pred_time_best = fr_h1.pred_time if fr_h1 else None

        models.append(
            ModelInfo(
                seriesName=t,
                modelType=model_type,
                mase=best_mase,
                smape=smape_best,
                rmse=rmse_best,
                fit_time=fit_time_best,
                pred_time=pred_time_best,
                isSelected=True,
            )
        )

        # --- 7. Фінальна модель + Ljung–Box + прогнози (як у тебе було) ---
        y = df_imp[t].dropna()
        n = len(y)
        horizon = req.horizon

        train_end = int(n * 0.8)
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:]

        if exog_df is not None:
            exog_all = exog_df.loc[y.index]
            exog_train = exog_all.iloc[:train_end]
            exog_test = exog_all.iloc[train_end:]
            exog_future = exog_all.iloc[-horizon:]
        else:
            exog_all = None
            exog_train = None
            exog_test = None
            exog_future = None

        lb_pvalue = None
        resid_ok = None

        if model_type in ["ARIMA", "SARIMA", "SARIMAX"]:
            if req.frequency == "M":
                season = 12
            elif req.frequency == "Q":
                season = 4
            else:
                season = 1

            if model_type == "ARIMA":
                order = (1, 1, 1)
                seas_order = (0, 0, 0, 0)
            elif model_type == "SARIMA":
                order = (1, 1, 1)
                seas_order = (1, 1, 1, season)
            else:  # SARIMAX
                order = (1, 1, 1)
                seas_order = (1, 1, 1, season)

            try:
                final_model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seas_order,
                    exog=exog_train if model_type == "SARIMAX" else None,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                resid = final_model.resid
                lb = acorr_ljungbox(resid.dropna(), lags=[min(10, len(resid) // 2)])
                lb_pvalue = float(lb["lb_pvalue"].iloc[0])
                resid_ok = bool(lb_pvalue > 0.05)
            except Exception:
                lb_pvalue = None
                resid_ok = None

        targets_diag[t] = {
            "lb_pvalue": lb_pvalue,
            "residuals_ok": resid_ok,
        }

        targets_exog[t] = []
        if exog_cols:
            for col in exog_cols:
                targets_exog[t].append({"base": col, "lag": 0})

        # Прогнози для ARIMA/SARIMA/SARIMAX (дерева як forecast на future можемо додати окремо)
        if model_type in ["ARIMA", "SARIMA", "SARIMAX"]:
            if req.frequency == "M":
                s = 12
            elif req.frequency == "Q":
                s = 4
            else:
                s = 1
            if model_type == "ARIMA":
                order = (1, 1, 1)
                seas_order = (0, 0, 0, 0)
                ex_all = None
            elif model_type == "SARIMA":
                order = (1, 1, 1)
                seas_order = (1, 1, 1, s)
                ex_all = None
            else:  # SARIMAX
                order = (1, 1, 1)
                seas_order = (1, 1, 1, s)
                ex_all = exog_all if exog_all is not None else None

            try:
                final_model_all = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seas_order,
                    exog=ex_all,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                # backtest на тесті
                if ex_all is not None and model_type == "SARIMAX":
                    yhat_test = final_model_all.get_prediction(
                        start=y_test.index[0], end=y_test.index[-1], exog=exog_test
                    ).predicted_mean
                else:
                    yhat_test = final_model_all.get_prediction(
                        start=y_test.index[0], end=y_test.index[-1]
                    ).predicted_mean

                for dt, y_true_val in y_test.items():
                    y_pred_val = float(yhat_test.loc[dt]) if dt in yhat_test.index else None
                    forecasts.append(
                        ForecastPoint(
                            seriesName=t,
                            date=str(dt.date()),
                            setType="test",
                            valueActual=float(y_true_val),
                            valuePred=y_pred_val,
                        )
                    )

                # future
                if ex_all is not None and model_type == "SARIMAX":
                    yhat_future = final_model_all.forecast(steps=horizon, exog=exog_future)
                else:
                    yhat_future = final_model_all.forecast(steps=horizon)
                if req.frequency == "M":
                    freq_future = "ME"  # month-end
                else:
                    freq_future = req.frequency

                future_idx = pd.date_range(
                    start=y.index[-1] + (y.index[-1] - y.index[-2]),
                    periods=horizon,
                    freq=freq_future,
                )
                for dt, v in zip(future_idx, yhat_future):
                    forecasts.append(
                        ForecastPoint(
                            seriesName=t,
                            date=str(dt.date()),
                            setType="future",
                            valueActual=None,
                            valuePred=float(v),
                        )
                    )
            except Exception:
                pass
        else:
            # Для RF/GB зараз не будуємо future (можемо додати окремо пізніше, якщо реально треба)
            pass

    diagnostics: Dict[str, Any] = {
        "meta": meta,
        "series": series_diag,
        "transforms": trans_info,
        "base_variables": base_vars,
        "targets": targets_diag,
        "targets_exog": targets_exog,
        "comparison": comparison,
        "selection": selection_info,
    }

    # важливо: чистимо все, що йде в Dict[str, Any]
    diagnostics_clean = _to_python(diagnostics)
    correlations_clean = _to_python(correlations)
    factors_clean = _to_python(factors)

    resp = ExperimentResult(
        diagnostics=diagnostics_clean,
        correlations=correlations_clean,
        factors=factors_clean,
        models=models,
        forecasts=forecasts,
    )
    return resp



