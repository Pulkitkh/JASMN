"""Macroeconomic signal collector (design doc §10).

Daily series for repo rate, CPI, USD/INR, crude, 10Y yield, India VIX and
the NIFTY index. Live mode pulls market-linked series (NIFTY, India VIX,
USD/INR, Brent crude) from Yahoo; policy variables (repo rate, CPI, 10Y
yield) have no reliable free API, so they come from configured values —
POLICY_MACRO_DEFAULTS, overridable in data/config/macro.json. The synthetic
fallback covers offline mode. Values are market-wide (no symbol column).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector
from jasmin.config import DATA_DIR, POLICY_MACRO_DEFAULTS

_LIVE_TICKERS = {
    "nifty_index": "^NSEI",
    "india_vix": "^INDIAVIX",
    "usdinr": "INR=X",
    "crude_usd": "BZ=F",
}


def _policy_values() -> dict:
    values = dict(POLICY_MACRO_DEFAULTS)
    override = DATA_DIR / "config" / "macro.json"
    if override.exists():
        values.update(json.loads(override.read_text()))
    return values


class MacroCollector(BaseCollector):
    name = "macro"

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        if not self.offline:
            live = self._fetch_live(days)
            if live is not None:
                return live
        return self._fetch_synthetic(days)

    def _fetch_live(self, days: int) -> pd.DataFrame | None:
        try:
            from jasmin.live.yahoo import YahooClient

            yahoo = YahooClient()
            merged: pd.DataFrame | None = None
            for column, ticker in _LIVE_TICKERS.items():
                series = yahoo.daily_history(ticker, days)[["date", "close"]].rename(
                    columns={"close": column}
                )
                merged = series if merged is None else merged.merge(
                    series, on="date", how="outer"
                )
            merged = merged.sort_values("date").reset_index(drop=True)
            numeric = merged.columns.drop("date")
            merged[numeric] = merged[numeric].ffill()
            for column, value in _policy_values().items():
                merged[column] = value
            merged["source"] = "yahoo+policy"
            self.log.info("live macro: %d rows", len(merged))
            return merged
        except Exception as exc:
            self.log.warning("live macro fetch failed (%s); using synthetic macro", exc)
            return None

    def _fetch_synthetic(self, days: int) -> pd.DataFrame:
        rng = np.random.default_rng(20260703)
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        n = len(dates)

        def walk(start: float, vol: float, drift: float = 0.0) -> np.ndarray:
            return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))

        # Policy rates step every ~4 months rather than drifting daily.
        repo = np.repeat(
            rng.choice([6.0, 6.25, 6.5], size=n // 85 + 1), 85
        )[:n]
        cpi = np.clip(walk(5.0, 0.004), 2.5, 8.0)

        return pd.DataFrame(
            {
                "date": dates,
                "repo_rate": repo,
                "cpi_inflation": cpi.round(2),
                "usdinr": walk(84.0, 0.002).round(2),
                "crude_usd": walk(78.0, 0.012).round(2),
                "bond_yield_10y": np.clip(walk(7.0, 0.004), 5.5, 8.5).round(3),
                "india_vix": np.clip(walk(13.5, 0.03), 8, 40).round(2),
                "nifty_index": walk(24_000.0, 0.008, drift=0.0003).round(1),
            }
        )
