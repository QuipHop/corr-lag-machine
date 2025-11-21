# ml-svc/core/schemas.py
from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class SeriesPayload(BaseModel):
    """
    Один часовий ряд: ім'я, роль і значення.
    values йдуть у тому ж порядку, що й масив dates у запиті.
    """
    name: str
    role: Literal["target", "candidate", "ignored"] = "candidate"
    values: List[Optional[float]] = Field(
        ..., description="Time-ordered values aligned with dates[]"
    )


class ExperimentRequest(BaseModel):
    """
    Вхід у ml-svc: жодних файлів, тільки масив дат і рядів.
    """
    experiment_id: str

    dates: List[str]  # ISO-рядки 'YYYY-MM-DD'
    series: List[SeriesPayload]

    frequency: Literal["M", "Q", "Y"] = "M"   # місяць / квартал / рік
    horizon: int = 12                        # довжина тестового періоду

    imputation: Literal["none", "ffill", "bfill", "interp"] = "ffill"
    max_lag: int = 12                        # макс. лаг для крос-кореляцій

    extra: Dict[str, Any] = {}               # запас на майбутні налаштування


class ModelInfo(BaseModel):
    """
    Інформація про МОДЕЛЬ (кандидат або обрана) для конкретного ряду.
    is_selected = True для тієї, яка вважена найкращою за MASE.
    """
    series_name: str
    model_type: str              # "SARIMA", "SeasonalNaive", "SARIMAX" тощо
    params: Dict[str, Any]
    mase: Optional[float] = None
    smape: Optional[float] = None
    rmse: Optional[float] = None
    is_selected: bool = True


class ForecastPoint(BaseModel):
    """
    Одна точка прогнозу / факту.
    """
    date: str
    value_actual: Optional[float] = None
    value_pred: Optional[float] = None
    lower_pi: Optional[float] = None
    upper_pi: Optional[float] = None
    set_type: Literal["train", "test", "future"]


class ExperimentMetric(BaseModel):
    """
    Агреговані метрики для таргетного ряду (обраної моделі).
    """
    series_name: str
    model_type: str
    horizon: int
    mase: float
    smape: float
    rmse: float


class ExperimentResult(BaseModel):
    """
    Повна відповідь пайплайна.
    """
    diagnostics: Dict[str, Any]              # стаціонарність, сезонність, базові змінні
    correlations: Dict[str, Any]             # кореляції, крос-кореляції
    factors: Dict[str, Any]                  # факторний аналіз, пояснена дисперсія
    models: List[ModelInfo]                  # ВСІ моделі (кандидати + обрані)

    # Прогнози:
    #  - "base": базові змінні (test + future)
    #  - "macro": таргетні макропоказники
    forecasts: Dict[str, List[ForecastPoint]]

    metrics: List[ExperimentMetric]          # MASE/sMAPE/RMSE для таргетів (обрані моделі)
