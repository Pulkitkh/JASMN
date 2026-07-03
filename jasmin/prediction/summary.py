"""Aggregated market view: accumulate per-stock predictions into one response.

Combines every symbol's prediction with market-level context (VIX regime,
NIFTY trend, FII flows) into a single summary: overall bias, conviction,
sector rollup and ranked picks — the "accumulated response" layer on top of
individual predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.config import SECTOR_MAP
from jasmin.dataset import load_master_dataset
from jasmin.prediction.predict import Prediction


def build_market_summary(predictions: list[Prediction]) -> dict:
    if not predictions:
        return {"status": "no_predictions"}

    rows = [
        {
            "symbol": p.symbol,
            "sector": SECTOR_MAP.get(p.symbol, "OTHER"),
            "direction": p.direction,
            "probability_up": p.probability_up,
            "expected_move_pct": p.expected_move_pct,
            "confidence": p.confidence["score"],
            "top_driver": (
                p.explanation["positive_factors"][0]["description"]
                if p.direction == "UP" and p.explanation["positive_factors"]
                else p.explanation["negative_factors"][0]["description"]
                if p.explanation["negative_factors"]
                else ""
            ),
        }
        for p in predictions
    ]
    df = pd.DataFrame(rows)

    # Confidence-weighted breadth: +1 fully-confident UP, -1 fully-confident DOWN.
    signed = np.where(df["direction"] == "UP", 1, -1) * df["confidence"] / 100
    breadth = float(signed.mean())
    bias = "BULLISH" if breadth > 0.15 else "BEARISH" if breadth < -0.15 else "NEUTRAL"

    sectors = (
        df.groupby("sector")
        .apply(
            lambda g: {
                "symbols": g["symbol"].tolist(),
                "up": int((g["direction"] == "UP").sum()),
                "down": int((g["direction"] == "DOWN").sum()),
                "avg_expected_move_pct": round(float(g["expected_move_pct"].mean()), 3),
            },
            include_groups=False,
        )
        .to_dict()
    )

    ranked = df.sort_values(
        ["direction", "confidence"], ascending=[False, False]
    )
    top_bullish = ranked[ranked["direction"] == "UP"].head(3).to_dict(orient="records")
    top_bearish = (
        df[df["direction"] == "DOWN"]
        .sort_values("confidence", ascending=False)
        .head(3)
        .to_dict(orient="records")
    )

    summary = {
        "as_of": predictions[0].as_of,
        "horizon_days": predictions[0].horizon_days,
        "market_bias": bias,
        "breadth_score": round(breadth, 3),
        "n_up": int((df["direction"] == "UP").sum()),
        "n_down": int((df["direction"] == "DOWN").sum()),
        "avg_confidence": round(float(df["confidence"].mean()), 1),
        "market_context": _market_context(),
        "sectors": sectors,
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
        "predictions": rows,
    }
    return summary


def _market_context() -> dict:
    """Latest market-wide readings from the master dataset."""
    try:
        master = load_master_dataset()
        latest = master.sort_values("date").iloc[-1]
        vix = float(latest["india_vix"])
        return {
            "india_vix": round(vix, 2),
            "vix_regime": "calm" if vix < 14 else "elevated" if vix < 22 else "stressed",
            "nifty_5d_return_pct": round(float(latest["nifty_return_5d"]) * 100, 2),
            "usdinr": round(float(latest["usdinr"]), 2),
            "fii_5d_net_cr": round(float(latest["fii_net_5d"]), 0)
            if pd.notna(latest.get("fii_net_5d"))
            else None,
        }
    except Exception:
        return {}
