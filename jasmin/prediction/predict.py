"""Prediction pipeline (design doc §14) — end-to-end inference for one symbol.

Steps: gather latest engineered features -> validate completeness -> load
latest approved model -> infer probability, expected movement and the
likely price range -> rank factors -> attach explanation and confidence.
Results are appended to data/master/predictions.csv for later auditing.

`infer_row` is the shared core: `predict` feeds it the latest master-dataset
row for a universe symbol, while `analyze` (on-demand flow) feeds it a
feature row built live for any symbol the user asks about.
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
    price: dict
    confidence: dict
    explanation: dict
    model_version: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def infer_row(bundle: dict, version: str, symbol: str, row: pd.Series,
              as_of: str, last_close: float, completeness: float,
              staleness_days: int) -> Prediction:
    """Run the loaded model bundle over one prepared feature row."""
    features: list[str] = bundle["features"]
    X = row[features].to_frame().T.astype(float)
    medians = pd.Series(bundle["feature_stats"]["median"])
    X = X.fillna(medians).fillna(0.0)

    member_probas = [
        float(c.predict_proba(X)[0, 1]) for c in bundle["classifiers"].values()
    ]
    proba_up = float(np.mean(member_probas))
    expected_move = float(bundle["regressor"].predict(X)[0])

    # Likely touch range from the quantile models, kept internally
    # consistent: high >= max(move, 0), low <= min(move, 0).
    high_pct = float(bundle["high_regressor"].predict(X)[0])
    low_pct = float(bundle["low_regressor"].predict(X)[0])
    high_pct = max(high_pct, expected_move, 0.0)
    low_pct = min(low_pct, expected_move, 0.0)

    # Conditional accuracy: judge the model by how it has performed in
    # market conditions like today's, not by its global average.
    from jasmin.models.train import regime_key

    vix = float(row["india_vix"]) if pd.notna(row.get("india_vix")) else None
    nifty_5d = float(row["nifty_return_5d"]) if pd.notna(row.get("nifty_return_5d")) else float("nan")
    regime = regime_key(vix if vix is not None else float("nan"), nifty_5d)
    accuracy = bundle["metrics"]["ensemble_accuracy"]
    regime_stats = bundle.get("regime_accuracy", {}).get(regime)
    if regime_stats and regime_stats["n"] >= 50:
        accuracy = regime_stats["accuracy"]

    conf = confidence_score(
        proba_up=proba_up,
        member_probas=member_probas,
        completeness=completeness,
        staleness_days=staleness_days,
        validation_accuracy=accuracy,
        india_vix=vix,
    )
    conf["regime"] = {"key": regime, "accuracy_basis": round(float(accuracy), 4)}
    explanation = explain(bundle["classifiers"], X.iloc[0], bundle["feature_stats"])

    prediction = Prediction(
        symbol=symbol,
        as_of=as_of,
        horizon_days=bundle["config"]["horizon_days"],
        direction="UP" if proba_up >= 0.5 else "DOWN",
        probability_up=round(proba_up, 4),
        expected_move_pct=round(expected_move, 3),
        price={
            "last_close": round(last_close, 2),
            "expected_close": round(last_close * (1 + expected_move / 100), 2),
            "likely_high_touch": round(last_close * (1 + high_pct / 100), 2),
            "likely_low_touch": round(last_close * (1 + low_pct / 100), 2),
        },
        confidence=conf,
        explanation=explanation,
        model_version=version,
    )
    _store(prediction)
    log.info("%s: %s (p=%.2f, move=%.2f%%, range %.2f..%.2f, confidence=%.0f)",
             symbol, prediction.direction, proba_up, expected_move,
             prediction.price["likely_low_touch"],
             prediction.price["likely_high_touch"], conf["score"])
    return prediction


def predict(symbol: str, config: PipelineConfig | None = None,
            registry: ModelRegistry | None = None) -> Prediction:
    """Predict a universe symbol from its latest master-dataset row."""
    config = config or PipelineConfig()
    registry = registry or ModelRegistry()

    version, bundle = registry.latest_approved()
    features: list[str] = bundle["features"]

    master = load_master_dataset()
    rows = master[master["symbol"] == symbol].sort_values("date")
    if rows.empty:
        raise KeyError(f"symbol {symbol} not present in master dataset")
    latest = rows.iloc[-1]

    completeness = float(latest[features].notna().mean())
    if completeness < config.min_feature_completeness:
        raise ValueError(
            f"feature completeness {completeness:.0%} below required "
            f"{config.min_feature_completeness:.0%} for {symbol}; refresh collectors"
        )

    staleness = (pd.Timestamp.today().normalize() - latest["date"].normalize()).days
    return infer_row(
        bundle, version, symbol, latest,
        as_of=str(latest["date"].date()),
        last_close=float(latest["close"]),
        completeness=completeness,
        staleness_days=max(staleness - 1, 0),  # data as of last close is fresh
    )


def _store(prediction: Prediction) -> None:
    """Append to the prediction audit log (stage 10: store results)."""
    record = prediction.to_dict()
    flat = {
        **{k: record[k] for k in (
            "symbol", "as_of", "horizon_days", "direction", "probability_up",
            "expected_move_pct", "model_version", "generated_at",
        )},
        "last_close": record["price"]["last_close"],
        "expected_close": record["price"]["expected_close"],
        "likely_high_touch": record["price"]["likely_high_touch"],
        "likely_low_touch": record["price"]["likely_low_touch"],
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
