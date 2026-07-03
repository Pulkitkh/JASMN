"""Company fundamentals collector (design doc §8).

Live mode pulls the current snapshot per symbol from Yahoo quoteSummary
(PE, PB, EPS, ROE, margins, holdings, market cap, revenue growth). Yahoo
only exposes present values, so the snapshot is stamped at the start of
history and treated as constant across the daily grid — a documented
approximation until a filings source provides true quarterly history.
The synthetic source emits one snapshot per quarter for offline mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector
from jasmin.utils.seeds import stable_seed


class FundamentalsCollector(BaseCollector):
    name = "fundamentals"

    FIELDS = [
        "pe", "pb", "eps", "roe", "roce", "debt_equity", "current_ratio",
        "operating_margin", "net_margin", "promoter_holding",
        "institutional_holding", "dividend_yield", "market_cap_cr",
        "revenue_growth_qoq", "earnings_surprise_pct",
    ]

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        if not self.offline:
            live = self._fetch_live(symbols)
            if live is not None:
                return live
        return self._fetch_synthetic(symbols, days)

    def _fetch_live(self, symbols: list[str]) -> pd.DataFrame | None:
        try:
            from jasmin.live.yahoo import YahooClient

            yahoo = YahooClient()
            # Stamp at epoch-of-history so merge_asof(backward) applies the
            # snapshot to every daily row, not just today's.
            start = pd.Timestamp("2000-01-01")
            rows = []
            for sym in symbols:
                snap = yahoo.fundamentals_snapshot(sym)
                rows.append({"date": start, "symbol": sym, **snap})
            df = pd.DataFrame(rows)
            df["source"] = "yahoo"
            self.log.info("live fundamentals: %d symbols", len(df))
            return df
        except Exception as exc:
            self.log.warning(
                "live fundamentals fetch failed (%s); using synthetic fundamentals", exc
            )
            return None

    def _fetch_synthetic(self, symbols: list[str], days: int) -> pd.DataFrame:
        quarters = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=max(days // 90 + 1, 2), freq="QS"
        )
        rows = []
        for sym in symbols:
            rng = np.random.default_rng(stable_seed("fund", sym))
            eps = rng.uniform(20, 120)
            roe = rng.uniform(10, 26)
            for q in quarters:
                eps *= 1 + rng.normal(0.02, 0.03)  # earnings drift quarter over quarter
                rows.append(
                    {
                        "date": q,
                        "symbol": sym,
                        "pe": round(rng.uniform(15, 35), 2),
                        "pb": round(rng.uniform(2, 8), 2),
                        "eps": round(eps, 2),
                        "roe": round(roe + rng.normal(0, 1), 2),
                        "roce": round(roe + rng.normal(2, 1.5), 2),
                        "debt_equity": round(abs(rng.normal(0.5, 0.3)), 2),
                        "current_ratio": round(rng.uniform(0.9, 2.5), 2),
                        "operating_margin": round(rng.uniform(12, 32), 2),
                        "net_margin": round(rng.uniform(8, 24), 2),
                        "promoter_holding": round(rng.uniform(30, 60), 2),
                        "institutional_holding": round(rng.uniform(15, 45), 2),
                        "dividend_yield": round(rng.uniform(0.3, 2.5), 2),
                        "market_cap_cr": round(rng.uniform(50_000, 1_900_000), 0),
                        "revenue_growth_qoq": round(rng.normal(3, 4), 2),
                        "earnings_surprise_pct": round(rng.normal(1, 4), 2),
                    }
                )
        return pd.DataFrame(rows)
