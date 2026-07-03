"""Prediction pipeline (design doc §14) — end-to-end inference for one symbol.

Steps: gather latest engineered features -> validate completeness -> load
latest approved model -> infer probability & expected movement -> rank
factors -> attach explanation and confidence. Results are appended to
data/master/predictions.csv so accuracy can be audited later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from jasmin.config import MASTER_DIR, PipelineConfig
from jasmin.dataset import load_master_dataset
from jasmin.models.registry import ModelRegistry
from jasmin.prediction.confidence import confidence_score
from jasmin.prediction.explain import explain
from jasmin.utils.logging import get_logger

log = get_logger("predict")

PREDICTIONS_PATH = MASTER_DIR / "predictions.csv"


@dataclass
class Prediction:
    symbol: str
    as_of: str
    horizon_days: int
    direction: str
    probability_up: float
    expected_move_pct: float
    confidence: dict
    explanation: dict
    model_version: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def predict(symbol: str, config: PipelineConfig | None = None,
            registry: ModelRegistry | None = None) -> Prediction:
    config = config or PipelineConfig()
    registry = registry or ModelRegistry()

    version, bundle = registry.latest_approved()
    features: list[str] = bundle["features"]

    master = load_master_dataset()
    rows = master[master["symbol"] == symbol].sort_values("date")
    if rows.empty:
        raise KeyError(f"symbol {symbol} not present in master dataset")
    latest = rows.iloc[-1]

    # Validate completeness before inferring (design doc §14).
    completeness = float(latest[features].notna().mean())
    if completeness < config.min_feature_completeness:
        raise ValueError(
            f"feature completeness {completeness:.0%} below required "
            f"{config.min_feature_completeness:.0%} for {symbol}; refresh collectors"
        )

    X = latest[features].to_frame().T.astype(float)
    medians = pd.Series(bundle["feature_stats"]["median"])
    X = X.fillna(medians)

    member_probas = [
        float(c.predict_proba(X)[0, 1]) for c in bundle["classifiers"].values()
    ]
    proba_up = float(np.mean(member_probas))
    expected_move = float(bundle["regressor"].predict(X)[0])

    staleness = (pd.Timestamp.today().normalize() - latest["date"].normalize()).days
    conf = confidence_score(
        proba_up=proba_up,
        member_probas=member_probas,
        completeness=completeness,
        staleness_days=max(staleness - 1, 0),  # data as of last close is fresh
        validation_accuracy=bundle["metrics"]["ensemble_accuracy"],
        india_vix=float(latest["india_vix"]) if pd.notna(latest.get("india_vix")) else None,
    )
    explanation = explain(bundle["classifiers"], X.iloc[0], bundle["feature_stats"])

    prediction = Prediction(
        symbol=symbol,
        as_of=str(latest["date"].date()),
        horizon_days=bundle["config"]["horizon_days"],
        direction="UP" if proba_up >= 0.5 else "DOWN",
        probability_up=round(proba_up, 4),
        expected_move_pct=round(expected_move, 3),
        confidence=conf,
        explanation=explanation,
        model_version=version,
    )
    _store(prediction)
    log.info("%s: %s (p=%.2f, move=%.2f%%, confidence=%.0f)", symbol,
             prediction.direction, proba_up, expected_move, conf["score"])
    return prediction


def _store(prediction: Prediction) -> None:
    """Append to the prediction audit log (stage 10: store results)."""
    record = prediction.to_dict()
    flat = {
        **{k: record[k] for k in (
            "symbol", "as_of", "horizon_days", "direction", "probability_up",
            "expected_move_pct", "model_version", "generated_at",
        )},
        "confidence_score": record["confidence"]["score"],
        "explanation_summary": record["explanation"]["summary"],
        "detail_json": json.dumps(
            {"confidence": record["confidence"], "explanation": record["explanation"]}
        ),
    }
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([flat])
    header = not PREDICTIONS_PATH.exists()
    row.to_csv(PREDICTIONS_PATH, mode="a", header=header, index=False)
