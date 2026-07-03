"""Collector contract.

Every collector is independent: it fetches one data domain, stamps it with
collection metadata, and writes raw CSVs under data/raw/<name>/. Collectors
never depend on each other, so any of them can be upgraded or replaced
without touching the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd

from jasmin.config import RAW_DIR
from jasmin.utils.logging import get_logger


class BaseCollector(ABC):
    """Fetch one data domain and persist it as raw CSV."""

    #: subdirectory under data/raw/ and the key used in the master dataset
    name: str = "base"

    def __init__(self, offline: bool = False):
        # offline=True forces the deterministic synthetic source; live
        # sources are only attempted when their optional dependency exists.
        self.offline = offline
        self.log = get_logger(f"collector.{self.name}")

    @abstractmethod
    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        """Return a tidy DataFrame with at least a `date` column (and
        `symbol` for per-stock domains)."""

    def collect(self, symbols: list[str], days: int = 400) -> pd.DataFrame:
        df = self.fetch(symbols, days)
        df = df.copy()
        df["collected_at"] = datetime.now(timezone.utc).isoformat()
        out_dir = RAW_DIR / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.csv"
        df.to_csv(path, index=False)
        self.log.info("collected %d rows -> %s", len(df), path)
        return df

    @staticmethod
    def load_raw(name: str) -> pd.DataFrame:
        path = RAW_DIR / name / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No raw data for '{name}'. Run `jasmin collect` first."
            )
        return pd.read_csv(path, parse_dates=["date"])
