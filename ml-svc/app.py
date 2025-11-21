from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import ExperimentRequest, ExperimentResponse
from core.experiment import run_full_experiment
from core.utils import to_native

app = FastAPI(title="Corr-Lag ML Service")

# CORS на всяк випадок
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/experiment/run", response_model=ExperimentResponse)
async def experiment_run(req: ExperimentRequest) -> ExperimentResponse:
    """
    Головний ендпойнт повного експерименту.
    Бере вже підготовлені ряди, ганяє повний пайплайн і повертає
    діагностику, кореляції, моделі, прогнози та метрики.
    """
    raw_result = run_full_experiment(req)

    # Чистимо numpy-типи (np.bool_, np.int64, np.float64) -> звичайні python-типи
    cleaned = to_native(raw_result)

    # Валідуємо/перетворюємо у pydantic-модель для стабільного JSON
    return ExperimentResponse.model_validate(cleaned)
