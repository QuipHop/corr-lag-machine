# ml-svc/core/schemas.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------- Вхідний payload від бекенду ---------


class SeriesPayload(BaseModel):
    name: str
    role: Literal["target", "candidate", "ignored"]
    values: List[Optional[float]]


class ExperimentRequest(BaseModel):
    experiment_id: str
    dates: List[str]
    series: List[SeriesPayload]
    frequency: Literal["M", "Q", "Y"]
    horizon: int
    imputation: Literal["none", "ffill", "bfill", "interp"] = "ffill"
    max_lag: int = 12
    extra: Dict[str, Any] = Field(default_factory=dict)


# --------- Діагностика ---------


class DiagnosticSeriesInfo(BaseModel):
    name: str
    role: Literal["target", "candidate", "ignored"]
    n: int
    mean: Optional[float] = None
    std: Optional[float] = None
    adf_pvalue: Optional[float] = None
    adf_stat: Optional[float] = None
    kpss_pvalue: Optional[float] = None
    kpss_stat: Optional[float] = None
    has_seasonality: Optional[bool] = None
    season_label: Optional[str] = None


class Diagnostics(BaseModel):
    frequency: Literal["M", "Q", "Y"]
    series: List[DiagnosticSeriesInfo]


# --------- Кореляції / фактори ---------


class CorrelationEntry(BaseModel):
    source: str
    target: str
    lag: int
    value: Optional[float] = None
    abs: Optional[float] = None
    n: int


class Correlations(BaseModel):
    max_lag: int
    pairs: List[CorrelationEntry]


class FactorInfo(BaseModel):
    target: str
    drivers: List[str]


class Factors(BaseModel):
    items: List[FactorInfo]


# --------- Моделі / метрики ---------


class ModelInfo(BaseModel):
    series_name: str
    model_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    is_selected: bool = False


class MetricRow(BaseModel):
    series_name: str
    model_type: str
    horizon: int
    mase: float
    smape: float
    rmse: float


# --------- Прогнози ---------


class ForecastPoint(BaseModel):
    date: date
    series_name: str
    value_actual: Optional[float] = None
    value_pred: Optional[float] = None
    lower_pi: Optional[float] = None
    upper_pi: Optional[float] = None
    set_type: Literal["train", "test", "future"]


class Forecasts(BaseModel):
    base: List[ForecastPoint]
    macro: List[ForecastPoint] = Field(default_factory=list)


# --------- Повна відповідь експерименту ---------


class ExperimentResponse(BaseModel):
    diagnostics: Diagnostics
    correlations: Correlations
    factors: Factors
    models: List[ModelInfo]
    forecasts: Forecasts
    metrics: List[MetricRow]
