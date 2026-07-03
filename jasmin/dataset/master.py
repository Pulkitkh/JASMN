"""Master dataset construction (pipeline stage 5).

Runs validation -> cleaning -> feature engineering over the raw collector
outputs, attaches forward-looking labels, and appends the result to the
master dataset in data/master/. Labels:

  - target_up:       close in `horizon_days` is above today's close
  - target_move_pct: percentage move over the horizon (regression target)

The last `horizon_days` rows per symbol have no label yet ("new labels
become available after market movement", design doc §17); they are kept
with NaN labels and used only for inference.
"""

from __future__ import annotations

import pandas as pd

from jasmin.cleaning import clean_daily_domain, clean_prices, clean_quarterly_domain
from jasmin.collectors.base import BaseCollector
from jasmin.config import CLEAN_DIR, MASTER_DIR, PipelineConfig
from jasmin.features.engineering import build_feature_frame
from jasmin.utils.logging import get_logger
from jasmin.validation import validate_domain, validate_prices

log = get_logger("dataset")

MASTER_PATH = MASTER_DIR / "master.csv"


def _add_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    future_close = df.groupby("symbol")["close"].shift(-horizon)
    df["target_move_pct"] = (future_close / df["close"] - 1) * 100
    df["target_up"] = (df["target_move_pct"] > 0).astype(float)
    df.loc[df["target_move_pct"].isna(), "target_up"] = float("nan")
    return df


def build_master_dataset(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Validate, clean and feature-engineer raw data into the master dataset."""
    config = config or PipelineConfig()

    prices = BaseCollector.load_raw("prices")
    fundamentals = BaseCollector.load_raw("fundamentals")
    macro = BaseCollector.load_raw("macro")
    institutional = BaseCollector.load_raw("institutional")
    news = BaseCollector.load_raw("news")

    # Stage 2: validation (errors abort, warnings pass through).
    validate_prices(prices).raise_if_failed()
    validate_domain(fundamentals, "fundamentals", ["date", "symbol", "pe", "eps"]).raise_if_failed()
    validate_domain(macro, "macro", ["date", "repo_rate", "usdinr"]).raise_if_failed()
    validate_domain(
        institutional, "institutional", ["date", "symbol", "fii_net_cr"]
    ).raise_if_failed()
    validate_domain(news, "news", ["date", "symbol", "headline", "sentiment"]).raise_if_failed()

    # Stage 3: cleaning; cleaned frames persisted for inspection/debugging.
    prices = clean_prices(prices)
    fundamentals = clean_quarterly_domain(fundamentals)
    macro = clean_daily_domain(macro)
    institutional = clean_daily_domain(institutional)
    news = clean_daily_domain(news)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("prices", prices), ("fundamentals", fundamentals), ("macro", macro),
        ("institutional", institutional), ("news", news),
    ]:
        frame.to_csv(CLEAN_DIR / f"{name}.csv", index=False)

    # Stage 4: feature engineering.
    features = build_feature_frame(prices, fundamentals, macro, institutional, news)

    # Stage 5: labels + master dataset.
    master = _add_labels(features, config.horizon_days)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)
    log.info("master dataset: %d rows, %d columns -> %s", *master.shape, MASTER_PATH)
    return master


def load_master_dataset() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            "Master dataset not built. Run `jasmin build-dataset` first."
        )
    return pd.read_csv(MASTER_PATH, parse_dates=["date"])
