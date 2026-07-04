"""FastAPI service (pipeline stage 10: dashboard/API).

Endpoints:
  GET /health                     liveness + model status
  GET /universe                   configured symbols
  GET /predict/{symbol}           full prediction with explanation & confidence
  GET /analyze?q={name or ticker} analyze ANY stock by free-text name/ticker
  GET /market-summary             aggregated view across the whole universe
  GET /market-status              NSE calendar status (IST)
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


@app.get("/analyze")
def analyze_stock(q: str) -> dict:
    from jasmin.prediction.analyze import analyze

    try:
        return analyze(q, config=config, registry=registry)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, ConnectionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/market-summary")
def market_summary() -> dict:
    from jasmin.prediction.summary import build_market_summary

    predictions = []
    errors = {}
    for symbol in config.universe:
        try:
            predictions.append(predict(symbol, config=config, registry=registry))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            errors[symbol] = str(exc)
    if not predictions:
        raise HTTPException(status_code=503, detail={"errors": errors})
    body = build_market_summary(predictions)
    if errors:
        body["errors"] = errors
    return body


@app.get("/market-status")
def market_status_endpoint() -> dict:
    from jasmin.utils.market_calendar import market_status

    return market_status()


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
