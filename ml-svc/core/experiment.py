# ml-svc/core/experiment.py
from __future__ import annotations

from typing import Dict, List, Tuple

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
    Реалізація повного алгоритму:
    1) формування DataFrame
    2) імпутація
    3) діагностика
    4) лагові кореляції target vs candidates
    5) відбір драйверів (factors)
    6) побудова SARIMAX для таргетів
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

    # 4. Лагові кореляції (тільки target vs candidates)
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

    # 5. Відбір драйверів: для кожного target беремо топ-3 candidate за |corr|
    factors_items: List[FactorInfo] = []
    for tgt in targets:
        relevant = [p for p in corr_pairs if p.target == tgt]
        # max по модулю для кожного candidate
        by_cand: Dict[str, float] = {}
        for p in relevant:
            prev = by_cand.get(p.source)
            if prev is None or (p.abs or 0) > prev:
                by_cand[p.source] = p.abs or 0.0
        top = sorted(by_cand.items(), key=lambda kv: kv[1], reverse=True)[:3]
        drivers = [name for name, _ in top]
        factors_items.append(FactorInfo(target=tgt, drivers=drivers))

    factors = Factors(items=factors_items)

    # 6. Моделювання: SARIMAX для кожного target з exog = drivers
    models: List[ModelInfo] = []
    metrics: List[MetricRow] = []
    forecast_points: List[ForecastPoint] = []

    m = _seasonal_period(req.frequency)
    offset = _freq_to_offset(req.frequency)

    for tgt in targets:
        y = df[tgt].astype(float)
        drivers = next((f.drivers for f in factors_items if f.target == tgt), [])

        exog = df[drivers] if drivers else None

        # train/test split
        horizon = min(req.horizon, max(4, len(y) // 4))
        train = y.iloc[:-horizon]
        test = y.iloc[-horizon:]
        exog_train = exog.iloc[:-horizon] if exog is not None else None
        exog_test = exog.iloc[-horizon:] if exog is not None else None

        # SARIMAX(p,d,q)(P,D,Q)m – беремо просту схему (1,1,1) + сезонний (1,0,1)
        try:
            model = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, m),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
        except Exception:
            # fallback: без сезонності / без exog
            model = SARIMAX(
                train,
                order=(1, 1, 1),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

        # in-sample прогноз на train+test
        pred_all = res.get_prediction(
            start=train.index[0],
            end=test.index[-1],
            exog=exog if exog is not None else None,
        )
        pred_mean = pred_all.predicted_mean
        pred_ci = pred_all.conf_int(alpha=0.2)  # 80% PI як приклад

        # майбутній прогноз
        last_date = y.index[-1]
        future_idx = pd.date_range(
            last_date + pd.tseries.frequencies.to_offset(offset),
            periods=req.horizon,
            freq=offset,
        )
        exog_future = df[drivers].reindex(future_idx) if drivers else None

        fut_res = res.get_forecast(steps=req.horizon, exog=exog_future)
        fut_mean = fut_res.predicted_mean
        fut_ci = fut_res.conf_int(alpha=0.2)

        # метрики на test
        y_test = test.values
        y_pred_test = pred_mean.loc[test.index].values
        mase_val = _mase(y_test, y_pred_test, train.values, m=max(m, 1))
        smape_val = _smape(y_test, y_pred_test)
        rmse_val = _rmse(y_test, y_pred_test)

        models.append(
            ModelInfo(
                series_name=tgt,
                model_type="SARIMAX(1,1,1)x(1,0,1)[{}]".format(m),
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

        # формуємо ForecastPoint для train/test
        for idx_date in train.index:
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=float(y.loc[idx_date]),
                    value_pred=float(pred_mean.loc[idx_date]),
                    lower_pi=float(pred_ci.loc[idx_date].iloc[0]),
                    upper_pi=float(pred_ci.loc[idx_date].iloc[1]),
                    set_type="train",
                )
            )

        for idx_date in test.index:
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=float(y.loc[idx_date]),
                    value_pred=float(pred_mean.loc[idx_date]),
                    lower_pi=float(pred_ci.loc[idx_date].iloc[0]),
                    upper_pi=float(pred_ci.loc[idx_date].iloc[1]),
                    set_type="test",
                )
            )

        # future
        for idx_date in future_idx:
            ci_row = fut_ci.loc[idx_date]
            forecast_points.append(
                ForecastPoint(
                    date=idx_date.date(),
                    series_name=tgt,
                    value_actual=None,
                    value_pred=float(fut_mean.loc[idx_date]),
                    lower_pi=float(ci_row.iloc[0]),
                    upper_pi=float(ci_row.iloc[1]),
                    set_type="future",
                )
            )

    forecasts = Forecasts(base=forecast_points, macro=[])

    # ---- Формуємо фінальну відповідь як dict, щоб to_native міг його пройти ----

    response = ExperimentResponse(
        diagnostics=diagnostics,
        correlations=correlations,
        factors=factors,
        models=models,
        forecasts=forecasts,
        metrics=metrics,
    )

    # Pydantic-модель -> dict, далі вже to_native + serialize у app.py
    return response.model_dump()
