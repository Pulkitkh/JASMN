"""Institutional & smart-money collector (design doc §11).

Tracks FII/DII net flows (market-wide, in crores), plus per-symbol bulk/block
deal counts and promoter pledge levels. The premise from the design document:
informed capital often precedes visible price movement, so these become
leading features downstream.

Live mode overlays real FII/DII flows from the official NSE API onto the
series. NSE only serves the latest session, so each fetch is appended to a
persistent log — real coverage grows day by day while the synthetic series
backfills dates from before the log started. Per-symbol deal/pledge data
stays synthetic until a bulk-deals source is integrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector
from jasmin.utils.seeds import stable_seed


class InstitutionalCollector(BaseCollector):
    name = "institutional"

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        df = self._fetch_synthetic(symbols, days)
        if not self.offline:
            df = self._overlay_live(df)
        return df

    def _overlay_live(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from jasmin.live.nse import fetch_fii_dii

            real = fetch_fii_dii().set_index("date")
            df = df.set_index("date")
            for col in ("fii_net_cr", "dii_net_cr"):
                if col in real.columns:
                    overlap = df.index.intersection(real.index)
                    df.loc[overlap, col] = real.loc[overlap, col]
            df["source"] = "synthetic+nse"
            self.log.info(
                "overlaid %d real FII/DII sessions from NSE log", len(real)
            )
            return df.reset_index()
        except Exception as exc:
            self.log.warning("NSE FII/DII fetch failed (%s); synthetic flows only", exc)
            return df

    def _fetch_synthetic(self, symbols: list[str], days: int) -> pd.DataFrame:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        rng_mkt = np.random.default_rng(9001)
        n = len(dates)
        # FII flows trend in persistent regimes; DII often takes the other side.
        fii_regime = np.repeat(rng_mkt.normal(0, 800, n // 20 + 1), 20)[:n]
        fii = fii_regime + rng_mkt.normal(0, 500, n)
        dii = -0.6 * fii + rng_mkt.normal(200, 400, n)

        rows = []
        for sym in symbols:
            rng = np.random.default_rng(stable_seed("inst", sym))
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
