# ml-svc/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import ExperimentRequest
from core.experiment import run_full_experiment

app = FastAPI(title="Corr Lag ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/experiment/run")
def experiment_run(req: ExperimentRequest):
    return run_full_experiment(req)
