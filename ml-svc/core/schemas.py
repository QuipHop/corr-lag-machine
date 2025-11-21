# ml-svc/core/schemas.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Request ----------

class SeriesPayload(BaseModel):
    name: str
    role: Literal["target", "candidate", "ignored"]
    values: List[Optional[float]]


class ExperimentRequest(BaseModel):
    experiment_id: str
    dates: List[date]  # бек шле строки, pydantic сам конвертить у date
    series: List[SeriesPayload]

    frequency: Literal["M", "Q", "Y"]
    horizon: int = Field(gt=0)

    imputation: Literal["none", "ffill", "bfill", "interp"] = "ffill"
    max_lag: int = 12

    extra: Dict[str, Any] = Field(default_factory=dict)


# ---------- Diagnostics ----------

class DiagnosticSeriesInfo(BaseModel):
    name: str
    n: int
    mean: Optional[float]
    std: Optional[float]
    has_seasonality: Optional[bool] = None
    adf_pvalue: Optional[float] = None
    kpss_pvalue: Optional[float] = None


class Diagnostics(BaseModel):
    series: List[DiagnosticSeriesInfo]
    frequency: str


# ---------- Correlations ----------

class CorrelationEntry(BaseModel):
    source: str
    target: str
    lag: int
    value: Optional[float]
    abs: Optional[float]
    n: int


class Correlations(BaseModel):
    pairs: List[CorrelationEntry]
    max_lag: int


# ---------- Factors (drivers) ----------

class FactorInfo(BaseModel):
    target: str
    drivers: List[str]


class Factors(BaseModel):
    items: List[FactorInfo]


# ---------- Models / Forecasts / Metrics ----------

class ModelInfo(BaseModel):
    series_name: str
    model_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    is_selected: bool = False


class ForecastPoint(BaseModel):
    date: date
    series_name: Optional[str] = None  # для зручності, бек зараз це ігнорує
    value_actual: Optional[float] = None
    value_pred: Optional[float] = None
    lower_pi: Optional[float] = None
    upper_pi: Optional[float] = None
    set_type: Literal["train", "test", "future"]


class Forecasts(BaseModel):
    base: List[ForecastPoint] = Field(default_factory=list)
    macro: List[ForecastPoint] = Field(default_factory=list)


class MetricRow(BaseModel):
    series_name: str
    model_type: str
    horizon: int
    mase: float
    smape: float
    rmse: float


class ExperimentResponse(BaseModel):
    diagnostics: Diagnostics
    correlations: Correlations
    factors: Factors
    models: List[ModelInfo]
    forecasts: Forecasts
    metrics: List[MetricRow]
