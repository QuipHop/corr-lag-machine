# ml-svc/core/pipeline.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

from .schemas import (
    ExperimentRequest,
    ExperimentResult,
    ModelInfo,
    ForecastPoint,
    ExperimentMetric,
)


# ===========================
# ВСПОМОГАТЕЛЬНІ ФУНКЦІЇ
# ===========================

def _freq_to_alias(freq: str) -> str:
    if freq == "M":
        return "MS"   # MonthStart
    if freq == "Q":
        return "QS"   # QuarterStart
    if freq == "Y":
        return "YS"   # YearStart
    raise ValueError(f"Unsupported frequency: {freq}")


def _seasonal_period(freq: str) -> int:
    if freq == "M":
        return 12
    if freq == "Q":
        return 4
    if freq == "Y":
        return 1
    return 1


def _build_dataframe(req: ExperimentRequest) -> pd.DataFrame:
    """
    Створює DataFrame з dates + series[].values,
    приводить до частоти й робить імпутацію.
    """
    if not req.dates:
        raise ValueError("dates array is empty")

    idx = pd.to_datetime(req.dates)
    data: Dict[str, pd.Series] = {}

    for s in req.series:
        if s.role == "ignored":
            continue
        if len(s.values) != len(idx):
            raise ValueError(
                f"Series '{s.name}' length {len(s.values)} "
                f"does not match dates length {len(idx)}"
            )
        ser = pd.Series(s.values, index=idx, dtype="float64")
        data[s.name] = ser

    if not data:
        raise ValueError("No usable series (all are 'ignored')")

    df = pd.DataFrame(data, index=idx).sort_index()

    # Приведення до заданої частоти
    alias = _freq_to_alias(req.frequency)
    df = df.asfreq(alias)

    # Імпутація пропусків
    if req.imputation == "ffill":
        df = df.ffill()
    elif req.imputation == "bfill":
        df = df.bfill()
    elif req.imputation == "interp":
        df = df.interpolate()
    elif req.imputation == "none":
        pass
    else:
        raise ValueError(f"Unsupported imputation: {req.imputation}")

    return df


def _train_test_split(df: pd.Series | pd.DataFrame, horizon: int) -> Tuple[Any, Any]:
    if len(df) <= horizon + 5:
        raise ValueError(
            f"Not enough observations ({len(df)}) for horizon={horizon}. "
            "Need at least horizon + 5."
        )
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return train, test


# ===========================
# КОРЕЛЯЦІЇ
# ===========================

def _cross_corr(a: pd.Series, b: pd.Series, lag: int) -> float:
    """
    Крос-кореляція при довільному лагу.
    lag > 0: a_t з b_{t-lag}
    lag < 0: a_t з b_{t+|lag|}
    """
    if lag > 0:
        return a.iloc[lag:].corr(b.iloc[:-lag])
    elif lag < 0:
        lag = -lag
        return a.iloc[:-lag].corr(b.iloc[lag:])
    else:
        return a.corr(b)


def _compute_correlations(
    df: pd.DataFrame,
    target_names: List[str],
    candidate_names: List[str],
    max_lag: int,
) -> Dict[str, Any]:
    pearson = df.corr().round(4).to_dict()

    lag_corr: Dict[str, Dict[str, Dict[int, float]]] = {}
    for target in target_names:
        lag_corr[target] = {}
        for cand in candidate_names:
            pair: Dict[int, float] = {}
            for lag in range(-max_lag, max_lag + 1):
                val = _cross_corr(df[target], df[cand], lag)
                if pd.isna(val):
                    continue
                pair[lag] = float(round(val, 4))
            lag_corr[target][cand] = pair

    return {
        "pearson": pearson,
        "lag": lag_corr,
    }


# ===========================
# ФАКТОРНИЙ АНАЛІЗ (PCA)
# ===========================

def _run_factor_analysis(
    df: pd.DataFrame,
    candidate_names: List[str],
) -> Dict[str, Any]:
    """
    Простий PCA по кандидатам. Обираємо базові змінні як ті,
    що мають найбільше |loading| у перших компонентах.
    """
    if not candidate_names:
        return {
            "base_variables": [],
            "explained_variance_ratio": [],
            "loadings": {},
        }

    X = df[candidate_names].dropna()
    if X.shape[0] < 5:
        # занадто мало спостережень — беремо всі кандидати як базові
        return {
            "base_variables": candidate_names,
            "explained_variance_ratio": [],
            "loadings": {},
        }

    # стандартизуємо
    X_std = (X - X.mean()) / (X.std(ddof=0) + 1e-8)

    n_components = min(3, X_std.shape[1], X_std.shape[0])
    pca = PCA(n_components=n_components)
    _ = pca.fit_transform(X_std)

    loadings = pca.components_.T  # shape: [n_features, n_components]
    explained = pca.explained_variance_ratio_.tolist()

    base_vars: List[str] = []
    used_idx: set[int] = set()

    for comp_idx in range(n_components):
        col = loadings[:, comp_idx]
        order = np.argsort(-np.abs(col))  # сортуємо за |loading|
        for idx in order:
            if idx not in used_idx:
                used_idx.add(idx)
                base_vars.append(candidate_names[idx])
                break

    loadings_dict: Dict[str, Dict[int, float]] = {}
    for i, name in enumerate(candidate_names):
        loadings_dict[name] = {
            int(j): float(loadings[i, j]) for j in range(n_components)
        }

    return {
        "base_variables": base_vars,
        "explained_variance_ratio": [float(x) for x in explained],
        "loadings": loadings_dict,
    }


# ===========================
# ДІАГНОСТИКА РЯДІВ
# ===========================

def _safe_adf(s: pd.Series) -> Dict[str, Any]:
    try:
        res = adfuller(s.dropna(), autolag="AIC")
        return {
            "statistic": float(res[0]),
            "pvalue": float(res[1]),
            "is_stationary": res[1] < 0.05,
        }
    except Exception:
        return {"statistic": None, "pvalue": None, "is_stationary": None}


def _safe_kpss(s: pd.Series) -> Dict[str, Any]:
    try:
        res = kpss(s.dropna(), regression="c", nlags="auto")
        # H0: стаціонарність
        return {
            "statistic": float(res[0]),
            "pvalue": float(res[1]),
            "is_stationary": res[1] > 0.05,
        }
    except Exception:
        return {"statistic": None, "pvalue": None, "is_stationary": None}


def _diagnose_series(
    df: pd.DataFrame,
    base_vars: List[str],
    targets: List[str],
    freq: str,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    s_period = _seasonal_period(freq)

    for col in df.columns:
        s = df[col]
        desc = {
            "role": (
                "target"
                if col in targets
                else ("base" if col in base_vars else "candidate")
            ),
            "is_target": col in targets,
            "is_base_variable": col in base_vars,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "n": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
        }

        adf_info = _safe_adf(s)
        kpss_info = _safe_kpss(s)

        desc["adf"] = adf_info
        desc["kpss"] = kpss_info

        # груба ознака сезонності: кореляція з самим собою на сезонному лагу
        if s_period > 1 and len(s) > s_period + 1:
            corr_seasonal = s.iloc[s_period:].corr(s.iloc[:-s_period])
            desc["seasonal_period"] = s_period
            desc["seasonal_corr"] = float(corr_seasonal) if not pd.isna(corr_seasonal) else None
            desc["has_seasonality"] = (
                corr_seasonal is not None and not pd.isna(corr_seasonal) and abs(corr_seasonal) > 0.5
            )
        else:
            desc["seasonal_period"] = 1
            desc["seasonal_corr"] = None
            desc["has_seasonality"] = False

        info[col] = desc

    return info


# ===========================
# ПРОСТІ МОДЕЛІ ДЛЯ ТАЙМ-СЕРІЙ
# ===========================

def _seasonal_naive_forecast(train: pd.Series, horizon: int, s: int) -> pd.Series:
    """
    Сезонний наївний прогноз: y_{t+h} = y_{t+h-s}.
    Якщо даних замало — падаємо до простого наївного y_{t+h} = y_t.
    """
    if s < 1 or len(train) <= s:
        last = train.iloc[-1]
        idx = pd.date_range(start=train.index[-1] + (train.index[-1] - train.index[-2]),
                            periods=horizon, freq=train.index.freq)
        return pd.Series([last] * horizon, index=idx)

    idx = pd.date_range(start=train.index[-1] + train.index.freq, periods=horizon, freq=train.index.freq)
    values = []
    for i in range(horizon):
        # беремо значення s періодів тому; якщо не вистачає — останнє
        src_idx = len(train) - s + i
        if 0 <= src_idx < len(train):
            values.append(train.iloc[src_idx])
        else:
            values.append(train.iloc[-1])
    return pd.Series(values, index=idx)


def _fit_sarima_forecast(
    series: pd.Series,
    horizon: int,
    freq: str,
    exog_full: pd.DataFrame | None = None,
    use_exog: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Фітимо SARIMAX(1,1,1)(*,*,*,s).
    Повертаємо (forecast_test, forecast_future),
    де forecast_test має довжину horizon (останній відрізок),
    а forecast_future — прогнози на horizon кроків після кінця ряду.
    """
    s = _seasonal_period(freq)

    train, test = _train_test_split(series, horizon)

    if use_exog and exog_full is not None:
        exog_train, exog_test = _train_test_split(exog_full, horizon)
    else:
        exog_train = exog_test = None

    if s > 1:
        seasonal_order = (1, 1, 0, s)
    else:
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        train,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    fc_test = res.get_forecast(steps=len(test), exog=exog_test)
    pred_test = fc_test.predicted_mean
    pred_test.index = test.index

    # майбутні прогнози
    if use_exog and exog_full is not None:
        last_exog = exog_full.iloc[-1:]
        future_exog = pd.concat(
            [last_exog] * horizon
        ).set_index(
            pd.date_range(start=series.index[-1] + series.index.freq,
                          periods=horizon, freq=series.index.freq)
        )
    else:
        future_exog = None

    fc_future = res.get_forecast(steps=horizon, exog=future_exog)
    pred_future = fc_future.predicted_mean

    return pred_test, pred_future


# ===========================
# МЕТРИКИ
# ===========================

def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))

    if seasonality < 1 or len(y_true) <= seasonality:
        denom = np.mean(np.abs(np.diff(y_true)))
    else:
        denom = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))

    if denom == 0 or np.isnan(denom):
        return float("inf")
    return float(mae / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(frac) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ===========================
# ОСНОВНИЙ ПАЙПЛАЙН (8 кроків)
# ===========================

def run_full_pipeline(req: ExperimentRequest) -> ExperimentResult:
    # 1. Приведення рядів до єдиної часової шкали та частоти
    df = _build_dataframe(req)

    target_names = [s.name for s in req.series if s.role == "target"]
    candidate_names = [s.name for s in req.series if s.role == "candidate"]

    if not target_names:
        raise ValueError("At least one series with role='target' is required")

    # 2. Кореляційні залежності
    correlations = _compute_correlations(df, target_names, candidate_names, req.max_lag)

    # 3. Факторний аналіз + вибір базових змінних
    factors = _run_factor_analysis(df, candidate_names)
    base_vars: List[str] = factors.get("base_variables", [])

    # 4. Діагностика (стаціонарність, сезонність, роль)
    per_series_diag = _diagnose_series(df, base_vars, target_names, req.frequency)

    diagnostics: Dict[str, Any] = {
        "meta": {
            "start": df.index.min().strftime("%Y-%m-%d"),
            "end": df.index.max().strftime("%Y-%m-%d"),
            "n_rows": int(len(df)),
            "frequency": req.frequency,
        },
        "targets": target_names,
        "candidates": candidate_names,
        "base_variables": base_vars,
        "series": per_series_diag,
    }

    s_period = _seasonal_period(req.frequency)

    # 5–6. Вибір моделі для базових змінних:
    #    порівнюємо SeasonalNaive vs SARIMA за MASE
    base_models: List[ModelInfo] = []
    base_forecast_points: List[ForecastPoint] = []

    for var in base_vars:
        series = df[var]
        train, test = _train_test_split(series, req.horizon)

        # Seasonal naive
        sn_test = _seasonal_naive_forecast(train, len(test), s_period)
        sn_test = sn_test.reindex(test.index)

        mase_sn = mase(test.values, sn_test.values, seasonality=s_period)
        smape_sn = smape(test.values, sn_test.values)
        rmse_sn = rmse(test.values, sn_test.values)

        # SARIMA
        sarima_test, sarima_future = _fit_sarima_forecast(series, req.horizon, req.frequency)
        mase_sarima = mase(test.values, sarima_test.values, seasonality=s_period)
        smape_sarima = smape(test.values, sarima_test.values)
        rmse_sarima = rmse(test.values, sarima_test.values)

        # вибір
        if mase_sarima <= mase_sn:
            chosen_type = "SARIMA(1,1,1)"
            chosen_mase, chosen_smape, chosen_rmse = mase_sarima, smape_sarima, rmse_sarima
            chosen_pred_test = sarima_test
            chosen_future = sarima_future
        else:
            chosen_type = "SeasonalNaive"
            chosen_mase, chosen_smape, chosen_rmse = mase_sn, smape_sn, rmse_sn
            chosen_pred_test = sn_test
            chosen_future = _seasonal_naive_forecast(series, req.horizon, s_period)

        # два кандидати в models[]
        base_models.append(
            ModelInfo(
                series_name=var,
                model_type="SeasonalNaive",
                params={"seasonal_period": s_period},
                mase=mase_sn,
                smape=smape_sn,
                rmse=rmse_sn,
                is_selected=(chosen_type == "SeasonalNaive"),
            )
        )
        base_models.append(
            ModelInfo(
                series_name=var,
                model_type="SARIMA(1,1,1)",
                params={"seasonal_period": s_period},
                mase=mase_sarima,
                smape=smape_sarima,
                rmse=rmse_sarima,
                is_selected=(chosen_type == "SARIMA(1,1,1)"),
            )
        )

        # дописуємо інформацію в diagnostics
        diagnostics["series"][var]["selected_model_type"] = chosen_type
        diagnostics["series"][var]["selected_model_mase"] = chosen_mase

        # точки test + future
        for date, actual, pred_val in zip(test.index, test.values, chosen_pred_test.values):
            base_forecast_points.append(
                ForecastPoint(
                    series_name=var,   # ← базова змінна
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=float(actual),
                    value_pred=float(pred_val),
                    set_type="test",
                )
            )
        for date, pred_val in zip(chosen_future.index, chosen_future.values):
            base_forecast_points.append(
                ForecastPoint(
                    series_name=var,   # ← та ж сама базова змінна
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=None,
                    value_pred=float(pred_val),
                    set_type="future",
                )
            )


    # 7. Прогноз макропоказників (таргетів) з урахуванням базових змінних
    macro_models: List[ModelInfo] = []
    macro_forecast_points: List[ForecastPoint] = []
    macro_metrics: List[ExperimentMetric] = []

    base_exog = df[base_vars] if base_vars else None

    for target in target_names:
        series = df[target]
        train_y, test_y = _train_test_split(series, req.horizon)
        if base_exog is not None:
            exog_train, exog_test = _train_test_split(base_exog, req.horizon)
        else:
            exog_train = exog_test = None

        # Seasonal naive baseline
        sn_test = _seasonal_naive_forecast(train_y, len(test_y), s_period)
        sn_test = sn_test.reindex(test_y.index)

        mase_sn = mase(test_y.values, sn_test.values, seasonality=s_period)
        smape_sn = smape(test_y.values, sn_test.values)
        rmse_sn = rmse(test_y.values, sn_test.values)

        # SARIMAX з exog (якщо є базові змінні)
        sarima_test, sarima_future = _fit_sarima_forecast(
            series,
            req.horizon,
            req.frequency,
            exog_full=base_exog,
            use_exog=base_exog is not None,
        )
        mase_sarima = mase(test_y.values, sarima_test.values, seasonality=s_period)
        smape_sarima = smape(test_y.values, sarima_test.values)
        rmse_sarima = rmse(test_y.values, sarima_test.values)

        if mase_sarima <= mase_sn:
            chosen_type = "SARIMAX(1,1,1)"
            chosen_mase, chosen_smape, chosen_rmse = mase_sarima, smape_sarima, rmse_sarima
            chosen_pred_test = sarima_test
            chosen_future = sarima_future
        else:
            chosen_type = "SeasonalNaive"
            chosen_mase, chosen_smape, chosen_rmse = mase_sn, smape_sn, rmse_sn
            chosen_pred_test = sn_test
            chosen_future = _seasonal_naive_forecast(series, req.horizon, s_period)

        # два кандидати у models[]
        macro_models.append(
            ModelInfo(
                series_name=target,
                model_type="SeasonalNaive",
                params={"seasonal_period": s_period},
                mase=mase_sn,
                smape=smape_sn,
                rmse=rmse_sn,
                is_selected=(chosen_type == "SeasonalNaive"),
            )
        )
        macro_models.append(
            ModelInfo(
                series_name=target,
                model_type="SARIMAX(1,1,1)",
                params={"seasonal_period": s_period, "uses_exog": base_exog is not None},
                mase=mase_sarima,
                smape=smape_sarima,
                rmse=rmse_sarima,
                is_selected=(chosen_type == "SARIMAX(1,1,1)"),
            )
        )

        macro_metrics.append(
            ExperimentMetric(
                series_name=target,
                model_type=chosen_type,
                horizon=req.horizon,
                mase=chosen_mase,
                smape=chosen_smape,
                rmse=chosen_rmse,
            )
        )

        # diagnostics для таргетів
        diagnostics["series"][target]["selected_model_type"] = chosen_type
        diagnostics["series"][target]["selected_model_mase"] = chosen_mase

        # точки test + future
        for date, actual, pred_val in zip(test_y.index, test_y.values, chosen_pred_test.values):
            macro_forecast_points.append(
                ForecastPoint(
                    series_name=target,   # ← назва таргетного ряду
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=float(actual),
                    value_pred=float(pred_val),
                    set_type="test",
                )
            )
        for date, pred_val in zip(chosen_future.index, chosen_future.values):
            macro_forecast_points.append(
                ForecastPoint(
                    series_name=target,   # ← та сама ціль
                    date=date.strftime("%Y-%m-%d"),
                    value_actual=None,
                    value_pred=float(pred_val),
                    set_type="future",
                )
            )

    all_models = base_models + macro_models

    return ExperimentResult(
        diagnostics=diagnostics,
        correlations=correlations,
        factors=factors,
        models=all_models,
        forecasts={
            "base": base_forecast_points,
            "macro": macro_forecast_points,
        },
        metrics=macro_metrics,
    )
