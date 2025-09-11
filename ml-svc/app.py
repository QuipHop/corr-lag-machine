from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd

app = FastAPI()

class Point(BaseModel):
    date: str
    value: float

class SeriesIn(BaseModel):
    id: int
    code: str
    points: List[Point]

class CorrLagRequest(BaseModel):
    series: List[SeriesIn]
    maxLag: int = 12
    method: Literal['pearson', 'spearman'] = 'pearson'
    minOverlap: int = 12
    edgeMin: float = 0.3 

@app.get('/health')
async def health():
    return {"status": "ok"}

@app.post('/corr-lag')
async def corr_lag(req: CorrLagRequest):
    frames = {}
    for s in req.series:
        if not s.points:
            continue
        df = pd.DataFrame([{"date": p.date, "value": p.value} for p in s.points])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        frames[s.code] = df["value"]

    if not frames:
        return {"nodes": [], "edges": []}

    combined = pd.DataFrame(frames).sort_index()
    nodes = [{"id": code} for code in combined.columns]
    edges = []

    # For each ordered pair, find the best lag by shifting the TARGET series
    # Convention: lag > 0 means SOURCE leads TARGET by `lag` periods
    cols = list(combined.columns)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue

            src = combined[cols[i]]
            tgt = combined[cols[j]]
            best = {"lag": 0, "corr": float("nan")}

            # Helpful cap: don't iterate lags that can never reach minOverlap
            max_possible_lag = max(0, len(combined) - req.minOverlap)
            L = min(req.maxLag, max_possible_lag)

            for lag in range(-L, L + 1):
                # Shift TARGET by +lag (positive = move target forward; source leads)
                shifted = pd.concat([src, tgt.shift(lag)], axis=1, join="inner").dropna()
                if len(shifted) < req.minOverlap:
                    continue

                corr = shifted.corr(
                    method='spearman' if req.method == 'spearman' else 'pearson'
                ).iloc[0, 1]

                # Track the highest absolute correlation
                if pd.notna(corr):
                    if pd.isna(best["corr"]) or abs(corr) > abs(best["corr"]):
                        best = {"lag": lag, "corr": float(corr)}

            # Keep the best edge if it passes threshold
            if pd.notna(best["corr"]) and abs(best["corr"]) >= req.edgeMin:
                edges.append({
                    "source": cols[i],
                    "target": cols[j],
                    "lag": best["lag"],
                    "weight": best["corr"],
                })

    return {"nodes": nodes, "edges": edges}
