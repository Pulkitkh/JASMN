"""FastAPI service (pipeline stage 10: dashboard/API).

Endpoints:
  GET /health                     liveness + model status
  GET /universe                   configured symbols
  GET /predict/{symbol}           full prediction with explanation & confidence
  GET /predictions                recent prediction audit log
  GET /models                     model registry versions & metrics

Run with: `jasmin serve` or `uvicorn jasmin.api.app:app`.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException

from jasmin import __version__
from jasmin.config import PipelineConfig
from jasmin.models.registry import ModelRegistry
from jasmin.prediction.predict import PREDICTIONS_PATH, predict

app = FastAPI(
    title="JASMIN AI",
    description="Explainable market intelligence for Indian equities",
    version=__version__,
)

config = PipelineConfig()
registry = ModelRegistry()


@app.get("/health")
def health() -> dict:
    try:
        version, _ = registry.latest_approved()
        model = {"status": "ready", "version": version}
    except FileNotFoundError:
        model = {"status": "no_model", "version": None}
    return {"service": "jasmin-ai", "version": __version__, "model": model}


@app.get("/universe")
def universe() -> dict:
    return {"symbols": config.universe}


@app.get("/predict/{symbol}")
def predict_symbol(symbol: str) -> dict:
    try:
        return predict(symbol.upper(), config=config, registry=registry).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/predictions")
def recent_predictions(limit: int = 20) -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    df = pd.read_csv(PREDICTIONS_PATH).tail(limit)
    records = df.drop(columns=["detail_json"]).to_dict(orient="records")
    for record, detail in zip(records, df["detail_json"]):
        record["detail"] = json.loads(detail)
    return records


@app.get("/models")
def models() -> list[dict]:
    return registry.list_versions()
