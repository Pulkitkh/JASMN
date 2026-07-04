"""Confidence scoring (design doc §16).

Blends: prediction probability strength, agreement among ensemble members,
feature completeness, data freshness, the model's own validation accuracy,
and current market volatility (high India VIX reduces confidence).
Output is a 0-100 score plus its component breakdown.
"""

from __future__ import annotations

import numpy as np

_WEIGHTS = {
    "probability_strength": 0.27,
    "model_agreement": 0.22,
    "feature_completeness": 0.13,
    "data_freshness": 0.09,
    "historical_accuracy": 0.09,
    "volatility_regime": 0.09,
    # Direction classifier and expected-move regressor pointing the same
    # way. When they conflict, the honest stance is "no edge" and the
    # score drops accordingly.
    "signal_alignment": 0.11,
}


def confidence_score(
    proba_up: float,
    member_probas: list[float],
    completeness: float,
    staleness_days: int,
    validation_accuracy: float,
    india_vix: float | None,
    signal_aligned: bool = True,
) -> dict:
    components = {
        "signal_alignment": 1.0 if signal_aligned else 0.0,
        # 0.5 proba -> 0, 0/1 proba -> 1
        "probability_strength": abs(proba_up - 0.5) * 2,
        # members on the same side of 0.5 and close together -> high agreement
        "model_agreement": max(0.0, 1.0 - float(np.std(member_probas)) * 4)
        * (1.0 if len({p > 0.5 for p in member_probas}) == 1 else 0.5),
        "feature_completeness": float(np.clip(completeness, 0, 1)),
        "data_freshness": float(np.clip(1 - staleness_days / 5, 0, 1)),
        # map 50% acc -> 0, 70%+ acc -> 1
        "historical_accuracy": float(np.clip((validation_accuracy - 0.5) / 0.2, 0, 1)),
        # VIX 12 or below -> calm (1), 30+ -> stressed (0)
        "volatility_regime": float(np.clip((30 - (india_vix or 18)) / 18, 0, 1)),
    }
    score = sum(_WEIGHTS[k] * v for k, v in components.items()) * 100
    return {
        "score": round(score, 1),
        "components": {k: round(v, 3) for k, v in components.items()},
    }
