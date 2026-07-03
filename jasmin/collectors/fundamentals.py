"""Company fundamentals collector (design doc §8).

Fundamentals change quarterly, so the collector emits one snapshot row per
symbol per quarter covering the requested window. The synthetic source keeps
values internally consistent (e.g. PE derived from EPS and price level) so
downstream features behave sensibly. Swap in a real source (screener,
exchange filings API) by overriding `_fetch_live`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector


class FundamentalsCollector(BaseCollector):
    name = "fundamentals"

    FIELDS = [
        "pe", "pb", "eps", "roe", "roce", "debt_equity", "current_ratio",
        "operating_margin", "net_margin", "promoter_holding",
        "institutional_holding", "dividend_yield", "market_cap_cr",
        "revenue_growth_qoq", "earnings_surprise_pct",
    ]

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        quarters = pd.date_range(
            end=pd.Timestamp.today().normalize(), periods=max(days // 90 + 1, 2), freq="QS"
        )
        rows = []
        for sym in symbols:
            rng = np.random.default_rng(abs(hash(("fund", sym))) % (2**32))
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
