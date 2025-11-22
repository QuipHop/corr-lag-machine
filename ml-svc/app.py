# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from core.experiment import ExperimentRequest, ExperimentResult, run_full_experiment

app = FastAPI()


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/experiment/run", response_model=ExperimentResult)
def run_experiment(req: ExperimentRequest) -> ExperimentResult:
    return run_full_experiment(req)
