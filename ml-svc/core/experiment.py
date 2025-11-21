# ml-svc/core/experiment.py
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .schemas import (
    CorrelationEntry,
    Correlations,
    DiagnosticSeriesInfo,
    Diagnostics,
    ExperimentRequest,
    ExperimentResponse,
    FactorInfo,
    Factors,
    ForecastPoint,
    Forecasts,
    MetricRow,
    ModelInfo,
)


# --------- Допоміжні речі ---------


def _freq_to_offset(freq: str) -> str:
    if freq == "M":
        return "MS"
    if freq == "Q":
        return "QS"
    if freq == "Y":
        return "YS"
    return "D"


def _seasonal_period(freq: str) -> int:
    if freq == "M":
        return 12
    if freq == "Q":
        return 4
    if freq == "Y":
        return 1
    return 1


def _detect_seasonality(series: pd.Series, freq: str) -> bool:
    m = _seasonal_period(freq)
    if m <= 1:
        return False
    s = series.dropna()
    if len(s) < m * 2:
        return False
    ac = s.autocorr(lag=m)
    return bool(ac is not None and abs(ac) > 0.3)


def _safe_adf(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 10:
        return None
    try:
        res = adfuller(s, autolag="AIC")
        return float(res[1])
    except Exception:
        return None


def _safe_kpss(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 10:
        return None
    try:
        res = kpss(s, regression="c", nlags="auto")
        return float(res[1])
    except Exception:
        return None


def _mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, m: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)

    num = np.mean(np.abs(y_true - y_pred))
    if len(insample) <= m:
        return float("nan")
    denom = np.mean(np.abs(insample[m:] - insample[:-m]))
    if denom == 0:
        return float("nan")
    return float(num / denom)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = np.nan
    val = 2.0 * np.abs(y_true - y_pred) / denom
    return float(np.nanmean(val) * 100.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# --------- Основний пайплайн ---------


def run_full_experiment(req: ExperimentRequest) -> Dict:
    """
    Повний алгоритм:
    1) формування DataFrame
    2) імпутація
    3) діагностика
    4) лагові кореляції target vs candidates
    5) відбір драйверів
    6) SARIMAX для таргетів (train/test + фінальний future)
    7) формування прогнозів і метрик
    """
    # 1. DataFrame
    idx = pd.to_datetime(req.dates)
    data = {}
    for s in req.series:
        data[s.name] = pd.to_numeric(pd.Series(s.values, index=idx), errors="coerce")
    df = pd.DataFrame(data, index=idx).sort_index()

    # 2. Імпутація
    if req.imputation == "ffill":
        df = df.ffill()
    elif req.imputation == "bfill":
        df = df.bfill()
    elif req.imputation == "interp":
        df = df.interpolate()

    # 3. Діагностика
    diag_items: List[DiagnosticSeriesInfo] = []
    for name, s in df.items():
        diag_items.append(
            DiagnosticSeriesInfo(
                name=name,
                n=int(s.count()),
                mean=float(s.mean()) if s.count() > 0 else None,
                std=float(s.std()) if s.count() > 1 else None,
                has_seasonality=_detect_seasonality(s, req.frequency),
                adf_pvalue=_safe_adf(s),
                kpss_pvalue=_safe_kpss(s),
            )
        )

    diagnostics = Diagnostics(series=diag_items, frequency=req.frequency)

    # 4. Лагові кореляції (target vs candidates)
    targets = [s.name for s in req.series if s.role == "target"]
    candidates = [s.name for s in req.series if s.role == "candidate"]

    corr_pairs: List[CorrelationEntry] = []
    max_lag = req.max_lag

    for tgt in targets:
        ts_target = df[tgt]
        for cand in candidates:
            ts_cand = df[cand]
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    x = ts_cand.shift(lag)
                    y = ts_target
                elif lag < 0:
                    x = ts_cand
                    y = ts_target.shift(-lag)
                else:
                    x = ts_cand
                    y = ts_target
                aligned = pd.concat([x, y], axis=1).dropna()
                n = len(aligned)
                if n < 8:
                    continue
                val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                corr_pairs.append(
                    CorrelationEntry(
                        source=cand,
                        target=tgt,
                        lag=lag,
                        value=val,
                        abs=abs(val),
                        n=n,
                    )
                )

    correlations = Correlations(pairs=corr_pairs, max_lag=max_lag)

    # 5. Драйвери: для кожного target беремо топ-3 candidate за |corr|
    factors_items: List[FactorInfo] = []
    for tgt in targets:
        relevant = [p for p in corr_pairs if p.target == tgt]
        by_cand: Dict[str, float] = {}
        for p in relevant:
            prev = by_cand.get(p.source)
            if prev is None or (p.abs or 0) > prev:
                by_cand[p.source] = p.abs or 0.0
        top = sorted(by_cand.items(), key=lambda kv: kv[1], reverse=True)[:3]
        drivers = [name for name, _ in top]
        factors_items.append(FactorInfo(target=tgt, drivers=drivers))

    factors = Factors(items=factors_items)

    # 6. Моделювання: SARIMAX для кожного target
    models: List[ModelInfo] = []
    metrics: List[MetricRow] = []
    forecast_points: List[ForecastPoint] = []

    m = _seasonal_period(req.frequency)
    offset = _freq_to_offset(req.frequency)

    for tgt in targets:
        y_full = df[tgt].astype(float)
        drivers = next((f.drivers for f in factors_items if f.target == tgt), [])

        exog_full = df[drivers] if drivers else None

        # довжина тесту (hold-out для backtest)
        horizon_test = min(req.horizon, max(4, len(y_full) // 4))

        train = y_full.iloc[:-horizon_test]
        test = y_full.iloc[-horizon_test:]
        exog_train = exog_full.iloc[:-horizon_test] if exog_full is not None else None
        exog_test = exog_full.iloc[-horizon_test:] if exog_full is not None else None

        # ---- 6.1. Перша модель: train -> in-sample + forecast на test ----
        try:
            model1 = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, m),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res1 = model1.fit(disp=False)
        except Exception:
            model1 = SARIMAX(
                train,
                order=(1, 1, 1),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res1 = model1.fit(disp=False)

        # In-sample (train) – без exog у get_prediction, бо він знає їх з фіту
        pred_train = res1.get_prediction()
        pred_train_mean = pred_train.predicted_mean
        pred_train_ci = pred_train.conf_int(alpha=0.2)

        # Out-of-sample (test) – через get_forecast з exog_test
        if exog_test is not None:
            fc_test = res1.get_forecast(steps=horizon_test, exog=exog_test)
        else:
            fc_test = res1.get_forecast(steps=horizon_test)

        test_mean = fc_test.predicted_mean
        test_ci = fc_test.conf_int(alpha=0.2)

        # метрики на test
        y_test = test.values
        y_pred_test = test_mean.values

        mase_val = _mase(y_test, y_pred_test, train.values, m=max(m, 1))
        smape_val = _smape(y_test, y_pred_test)
        rmse_val = _rmse(y_test, y_pred_test)

        models.append(
            ModelInfo(
                series_name=tgt,
                model_type=f"SARIMAX(1,1,1)x(1,0,1)[{m}]",
                params={"drivers": drivers},
                mase=mase_val,
                smape=smape_val,
                rmse=rmse_val,
                is_selected=True,
            )
        )

        metrics.append(
            MetricRow(
                series_name=tgt,
                model_type="SARIMAX",
                horizon=len(test),
                mase=mase_val,
                smape=smape_val,
                rmse=rmse_val,
            )
        )

        # ---- 6.2. Формуємо ForecastPoint для train/test ----

        # train
        for idx_date in train.index:
            mu = float(pred_train_mean.loc[idx_date])
            ci_row = pred_train_ci.loc[idx_date]
            lower = float(ci_row.iloc[0])
            upper = float(ci_row.iloc[1])
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=float(y_full.loc[idx_date]),
                    value_pred=mu,
                    lower_pi=lower,
                    upper_pi=upper,
                    set_type="train",
                )
            )

        # test – індекси беремо з test.index, значення з test_mean/test_ci по порядку
        for i, idx_date in enumerate(test.index):
            mu = float(test_mean.iloc[i])
            ci_row = test_ci.iloc[i]
            lower = float(ci_row.iloc[0])
            upper = float(ci_row.iloc[1])
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=float(y_full.loc[idx_date]),
                    value_pred=mu,
                    lower_pi=lower,
                    upper_pi=upper,
                    set_type="test",
                )
            )

        # ---- 6.3. Друга модель: full fit -> future forecast ----
        try:
            model2 = SARIMAX(
                y_full,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, m),
                exog=exog_full,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res2 = model2.fit(disp=False)
        except Exception:
            model2 = SARIMAX(
                y_full,
                order=(1, 1, 1),
                exog=exog_full,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res2 = model2.fit(disp=False)

        # future index
        last_date = y_full.index[-1]
        future_idx = pd.date_range(
            last_date + pd.tseries.frequencies.to_offset(offset),
            periods=req.horizon,
            freq=offset,
        )

        # exog_future: тримаємо останнє значення драйверів як константу
        if exog_full is not None and len(exog_full.columns) > 0:
            last_row = exog_full.iloc[-1]
            exog_future = pd.DataFrame(
                np.tile(last_row.values, (req.horizon, 1)),
                index=future_idx,
                columns=exog_full.columns,
            )
            fc_fut = res2.get_forecast(steps=req.horizon, exog=exog_future)
        else:
            fc_fut = res2.get_forecast(steps=req.horizon)

        fut_mean = fc_fut.predicted_mean
        fut_ci = fc_fut.conf_int(alpha=0.2)

        for i, idx_date in enumerate(future_idx):
            mu = float(fut_mean.iloc[i])
            ci_row = fut_ci.iloc[i]
            lower = float(ci_row.iloc[0])
            upper = float(ci_row.iloc[1])
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=None,
                    value_pred=mu,
                    lower_pi=lower,
                    upper_pi=upper,
                    set_type="future",
                )
            )

    forecasts = Forecasts(base=forecast_points, macro=[])

    # ---- Фінальна відповідь ----

    response = ExperimentResponse(
        diagnostics=diagnostics,
        correlations=correlations,
        factors=factors,
        models=models,
        forecasts=forecasts,
        metrics=metrics,
    )

    # Pydantic-модель -> dict, далі її вже to_native+serialize в app.py
    return response.model_dump()
