"""Institutional & smart-money collector (design doc §11).

Tracks FII/DII net flows (market-wide, in crores), plus per-symbol bulk/block
deal counts and promoter pledge levels. The premise from the design document:
informed capital often precedes visible price movement, so these become
leading features downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector


class InstitutionalCollector(BaseCollector):
    name = "institutional"

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        rng_mkt = np.random.default_rng(9001)
        n = len(dates)
        # FII flows trend in persistent regimes; DII often takes the other side.
        fii_regime = np.repeat(rng_mkt.normal(0, 800, n // 20 + 1), 20)[:n]
        fii = fii_regime + rng_mkt.normal(0, 500, n)
        dii = -0.6 * fii + rng_mkt.normal(200, 400, n)

        rows = []
        for sym in symbols:
            rng = np.random.default_rng(abs(hash(("inst", sym))) % (2**32))
            pledge = np.clip(rng.normal(2, 2), 0, 30)
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": sym,
                        "fii_net_cr": fii.round(1),
                        "dii_net_cr": dii.round(1),
                        "bulk_deals": rng.poisson(0.05, n),
                        "block_deals": rng.poisson(0.03, n),
                        "promoter_pledge_pct": np.round(
                            np.clip(pledge + np.cumsum(rng.normal(0, 0.02, n)), 0, 40), 2
                        ),
                        "insider_net_trades": rng.poisson(0.08, n) - rng.poisson(0.08, n),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)
