"""Cleaning & normalization (pipeline stage 3).

Deduplicates, sorts chronologically, fills small gaps, winsorizes extreme
outliers and derives the base return/gap columns described in design doc §6.
Cleaned frames are written to data/clean/ by the pipeline driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    subset = [c for c in ("date", "symbol") if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="last")
    return df.sort_values(subset).reset_index(drop=True)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV and derive returns, gaps and rolling returns."""
    df = _dedupe_sort(df.copy())
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    out = []
    for sym, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        # Small gaps (missing sessions) -> forward-fill prices, zero volume.
        price_cols = ["open", "high", "low", "close", "adj_close"]
        grp[price_cols] = grp[price_cols].ffill()
        grp["volume"] = grp["volume"].fillna(0)

        # Winsorize insane single-day returns (bad ticks), keep real moves.
        ret = grp["close"].pct_change()
        clipped = ret.clip(lower=ret.quantile(0.001), upper=ret.quantile(0.999))
        grp["return_1d"] = clipped.fillna(0.0)

        grp["gap_pct"] = ((grp["open"] - grp["close"].shift(1)) / grp["close"].shift(1)).fillna(0.0)
        for w in (5, 20):
            grp[f"return_{w}d"] = grp["close"].pct_change(w).fillna(0.0)
        grp["log_return_1d"] = np.log1p(grp["return_1d"])
        out.append(grp)

    return pd.concat(out, ignore_index=True)


def clean_daily_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a daily-frequency domain (macro, institutional, news aggregates)."""
    df = _dedupe_sort(df.copy())
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    numeric = df.select_dtypes(include=[np.number]).columns
    group_key = "symbol" if "symbol" in df.columns else None
    if group_key:
        df[numeric] = df.groupby(group_key)[numeric].transform(lambda s: s.ffill())
    else:
        df[numeric] = df[numeric].ffill()
    return df


def clean_quarterly_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Clean quarterly snapshots (fundamentals): dedupe and order only —
    as-of merging onto the daily grid happens in feature engineering."""
    df = _dedupe_sort(df.copy())
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df
