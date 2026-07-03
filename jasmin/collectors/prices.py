"""Historical OHLCV price collector (design doc §6).

Live source: yfinance with NSE suffix (RELIANCE -> RELIANCE.NS), used when
the optional dependency is installed and offline mode is not forced.
Fallback: a deterministic geometric-random-walk generator seeded per symbol,
so the whole pipeline runs (and tests pass) with no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector
from jasmin.utils.seeds import stable_seed

TRADING_DAYS_PER_YEAR = 252

# Base price levels so synthetic series look plausible per symbol.
_BASE_PRICE = {
    "RELIANCE": 2900.0,
    "TCS": 4100.0,
    "HDFCBANK": 1650.0,
    "INFY": 1850.0,
    "ICICIBANK": 1200.0,
}


def _synthetic_ohlcv(symbol: str, days: int) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed("prices", symbol))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    base = _BASE_PRICE.get(symbol, 500.0 + stable_seed("base", symbol) % 3000)
    drift = rng.normal(0.0004, 0.0002)
    vol = rng.uniform(0.012, 0.022)
    # Mild regime cycles so indicators have real trends to detect.
    t = np.arange(days)
    regime = 0.003 * np.sin(2 * np.pi * t / 90)
    returns = rng.normal(drift, vol, days) + regime * vol
    close = base * np.exp(np.cumsum(returns))

    spread = np.abs(rng.normal(0, vol / 2, days))
    open_ = close * (1 + rng.normal(0, vol / 3, days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=13.5, sigma=0.4, size=days) * (1 + np.abs(returns) * 25)
    delivery_pct = np.clip(rng.normal(52, 10, days), 15, 95)

    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "adj_close": close.round(2),
            "volume": volume.astype(int),
            "delivery_pct": delivery_pct.round(2),
        }
    )


class PriceCollector(BaseCollector):
    name = "prices"

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        if not self.offline:
            live = self._fetch_live(symbols, days)
            if live is not None:
                return live
        frames = [_synthetic_ohlcv(sym, days) for sym in symbols]
        return pd.concat(frames, ignore_index=True)

    def _fetch_live(self, symbols: list[str], days: int) -> pd.DataFrame | None:
        try:
            import yfinance as yf  # optional dependency
        except ImportError:
            self.log.info("yfinance not installed; using synthetic prices")
            return None
        try:
            frames = []
            for sym in symbols:
                hist = yf.Ticker(f"{sym}.NS").history(period=f"{days}d", auto_adjust=False)
                if hist.empty:
                    raise ValueError(f"empty history for {sym}")
                hist = hist.reset_index()
                frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(hist["Date"]).dt.tz_localize(None),
                            "symbol": sym,
                            "open": hist["Open"],
                            "high": hist["High"],
                            "low": hist["Low"],
                            "close": hist["Close"],
                            "adj_close": hist.get("Adj Close", hist["Close"]),
                            "volume": hist["Volume"],
                            "delivery_pct": float("nan"),  # not exposed by yfinance
                        }
                    )
                )
            return pd.concat(frames, ignore_index=True)
        except Exception as exc:  # network/source failure -> synthetic fallback
            self.log.warning("live price fetch failed (%s); using synthetic prices", exc)
            return None
