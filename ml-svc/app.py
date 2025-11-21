# ml-svc/app.py
from fastapi import FastAPI, HTTPException

from core.schemas import ExperimentRequest, ExperimentResult
from core.pipeline import run_full_pipeline

app = FastAPI(title="Macro Forecast ML Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/experiment/run", response_model=ExperimentResult)
def run_experiment(req: ExperimentRequest):
    """
    Головний ендпоінт: приймає dates + series, повертає повний результат алгоритму.
    """
    try:
        return run_full_pipeline(req)
    except Exception as e:
        # на проді додаси нормальне логування
        raise HTTPException(status_code=500, detail=str(e))
