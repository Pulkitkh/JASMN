"""Yahoo Finance client: daily OHLCV history and fundamentals snapshot.

Uses the public chart API (no auth) for history and the cookie+crumb flow
for quoteSummary fundamentals. NSE symbols take the .NS suffix; index/macro
tickers (^NSEI, ^INDIAVIX, INR=X, BZ=F) are passed through unchanged.
"""

from __future__ import annotations

import json
import time
import urllib.parse

import numpy as np
import pandas as pd

from jasmin.config import DATA_DIR
from jasmin.live.http import HttpClient
from jasmin.utils.logging import get_logger

log = get_logger("live.yahoo")

_CACHE_DIR = DATA_DIR / "cache" / "yahoo"
_CACHE_TTL_SECONDS = 30 * 60

_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval=1d&events=div%2Csplit"
_QUOTE_SUMMARY = (
    "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    "?modules=defaultKeyStatistics,financialData,summaryDetail&crumb={crumb}"
)


def nse_ticker(symbol: str) -> str:
    return symbol if symbol.startswith("^") or "=" in symbol or "." in symbol else f"{symbol}.NS"


class YahooClient:
    def __init__(self, client: HttpClient | None = None):
        self.http = client or HttpClient()
        self._crumb: str | None = None

    def daily_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Daily OHLCV for one ticker as a tidy frame (date ascending)."""
        if days > 2600:
            rng = "max"
        elif days > 1300:
            rng = "10y"
        elif days > 600:
            rng = "5y"
        elif days > 250:
            rng = "2y"
        else:
            rng = "1y" if days > 120 else "6mo"
        url = _CHART.format(symbol=urllib.parse.quote(nse_ticker(symbol)), range=rng)
        data = self._cached_get_json(url, f"chart_{nse_ticker(symbol)}_{rng}")
        result = data["chart"]["result"][0]
        ts = result.get("timestamp")
        if not ts:
            raise ValueError(f"no price data returned for {symbol}")
        quote = result["indicators"]["quote"][0]
        adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        tz_offset = result["meta"].get("gmtoffset", 19800)

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(np.array(ts) + tz_offset, unit="s").normalize(),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "adj_close": adj if adj is not None else quote["close"],
                "volume": quote["volume"],
            }
        )
        df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")
        return df.sort_values("date").tail(days).reset_index(drop=True)

    def _cached_get_json(self, url: str, key: str):
        """Disk cache with a short TTL: pre-market reruns and multi-command
        sessions reuse responses instead of re-tripping Yahoo rate limits."""
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        path = _CACHE_DIR / f"{safe}.json"
        if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # corrupt/unreadable cache entry: refetch
        data = self.http.get_json(url)
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        except OSError:
            pass
        return data

    def _get_crumb(self) -> str:
        if self._crumb is None:
            # Visiting fc.yahoo.com sets the auth cookie (the 404 is expected);
            # getcrumb then returns the API token tied to that cookie.
            self.http.get("https://fc.yahoo.com", allow_error=True)
            self._crumb = self.http.get(
                "https://query1.finance.yahoo.com/v1/test/getcrumb"
            ).decode("utf-8")
        return self._crumb

    def fundamentals_snapshot(self, symbol: str) -> dict:
        """Current fundamentals for one NSE symbol (raw floats, NaN if absent)."""
        url = _QUOTE_SUMMARY.format(
            symbol=urllib.parse.quote(nse_ticker(symbol)),
            crumb=urllib.parse.quote(self._get_crumb()),
        )
        result = self.http.get_json(url)["quoteSummary"]["result"][0]

        def raw(section: str, field: str, scale: float = 1.0) -> float:
            value = result.get(section, {}).get(field, {})
            return round(value["raw"] * scale, 4) if isinstance(value, dict) and "raw" in value else float("nan")

        return {
            "pe": raw("summaryDetail", "trailingPE"),
            "pb": raw("defaultKeyStatistics", "priceToBook"),
            "eps": raw("defaultKeyStatistics", "trailingEps"),
            "roe": raw("financialData", "returnOnEquity", 100),
            "roce": float("nan"),  # not exposed by Yahoo
            "debt_equity": raw("financialData", "debtToEquity", 0.01),
            "current_ratio": raw("financialData", "currentRatio"),
            "operating_margin": raw("financialData", "operatingMargins", 100),
            "net_margin": raw("financialData", "profitMargins", 100),
            "promoter_holding": raw("defaultKeyStatistics", "heldPercentInsiders", 100),
            "institutional_holding": raw("defaultKeyStatistics", "heldPercentInstitutions", 100),
            "dividend_yield": raw("summaryDetail", "dividendYield", 100),
            "market_cap_cr": raw("summaryDetail", "marketCap", 1e-7),  # INR -> crores
            "revenue_growth_qoq": raw("financialData", "revenueGrowth", 100),
            "earnings_surprise_pct": float("nan"),  # needs earningsHistory module
        }
