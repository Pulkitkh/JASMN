"""Explainability engine (design doc §15).

For each prediction, ranks feature contributions and renders them as
plain-English positive/negative factors ("bullish MACD crossover",
"improving FII activity") instead of a bare BUY/SELL.

Contribution heuristic: ensemble feature importance x signed z-score of the
current value against the training distribution. This is model-agnostic,
fast, and directionally faithful for tree ensembles on tabular data; SHAP
can be swapped in later behind the same `explain()` signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.features.engineering import FEATURE_DESCRIPTIONS

# Features whose *high* value is typically bearish, so the sign of the
# z-score flips when converting to a directional contribution.
_INVERTED = {"debt_equity", "pe", "india_vix", "promoter_pledge_pct", "atr_pct",
             "bb_width", "volatility_20d", "cpi_delta_20d", "bond_yield_10y"}


def _ensemble_importance(classifiers: dict) -> pd.Series:
    # Not every member exposes importances (HistGradientBoosting doesn't);
    # average over those that do.
    imps = [
        pd.Series(c.feature_importances_, index=c.feature_names_in_)
        for c in classifiers.values()
        if hasattr(c, "feature_importances_")
    ]
    return pd.concat(imps, axis=1).mean(axis=1)


def explain(
    classifiers: dict,
    row: pd.Series,
    feature_stats: dict,
    top_n: int = 5,
) -> dict:
    """Return top positive and negative contributors for one feature row."""
    importance = _ensemble_importance(classifiers)
    mean = pd.Series(feature_stats["mean"])
    std = pd.Series(feature_stats["std"]).replace(0, 1.0)

    z = ((row[importance.index] - mean) / std).clip(-3, 3)
    sign = pd.Series(
        [-1.0 if f in _INVERTED else 1.0 for f in importance.index], index=importance.index
    )
    contribution = (importance * z * sign).dropna()

    def _render(features: pd.Series, direction: str) -> list[dict]:
        out = []
        for name, value in features.items():
            if abs(value) < 1e-4:
                continue
            out.append(
                {
                    "feature": name,
                    "description": FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ")),
                    "value": round(float(row[name]), 4) if pd.notna(row[name]) else None,
                    "contribution": round(float(value), 4),
                    "direction": direction,
                }
            )
        return out

    positive = _render(contribution.nlargest(top_n), "positive")
    negative = _render(contribution.nsmallest(top_n), "negative")

    summary_bits = [p["description"] for p in positive[:3]]
    drag_bits = [n["description"] for n in negative[:2]]
    summary = ""
    if summary_bits:
        summary += "Supported by " + ", ".join(summary_bits)
    if drag_bits:
        summary += ("; held back by " if summary_bits else "Held back by ") + ", ".join(drag_bits)

    return {"positive_factors": positive, "negative_factors": negative, "summary": summary or "No dominant factors."}
