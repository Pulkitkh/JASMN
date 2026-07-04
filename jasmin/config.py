"""Central configuration: paths, universe, and pipeline settings.

Every module resolves its input/output locations through this file so that
collectors, feature generators and models stay independent of each other
(the "modular philosophy" from the design document).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("JASMIN_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
MASTER_DIR = DATA_DIR / "master"
MODELS_DIR = PROJECT_ROOT / "models" / "store"
LOGS_DIR = PROJECT_ROOT / "logs"

ALL_DIRS = [RAW_DIR, CLEAN_DIR, FEATURES_DIR, MASTER_DIR, MODELS_DIR, LOGS_DIR]

# Training universe: the full NIFTY 50. A broad, diverse universe is what
# lets the model generalize to stocks a user asks about that were never
# trained on — features are symbol-relative (returns, z-scores,
# sector-relative strength), not absolute prices.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH", "NTPC",
    "POWERGRID", "TATASTEEL", "BAJFINANCE", "ADANIENT", "TECHM",
    "M&M", "ADANIPORTS", "BAJAJFINSV", "BAJAJ-AUTO", "BEL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "JSWSTEEL",
    "NESTLEIND", "ONGC", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
    "APOLLOHOSP", "BRITANNIA", "TRENT", "JIOFIN", "DIVISLAB",
]

SECTOR_MAP = {
    "RELIANCE": "ENERGY", "ADANIENT": "ENERGY", "ONGC": "ENERGY",
    "COALINDIA": "ENERGY", "NTPC": "POWER", "POWERGRID": "POWER",
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING",
    "KOTAKBANK": "BANKING", "AXISBANK": "BANKING", "INDUSINDBK": "BANKING",
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE", "JIOFIN": "FINANCE",
    "SHRIRAMFIN": "FINANCE", "HDFCLIFE": "INSURANCE", "SBILIFE": "INSURANCE",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "ASIANPAINT": "CONSUMER", "TITAN": "CONSUMER", "TRENT": "CONSUMER",
    "MARUTI": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "LT": "INFRA", "ADANIPORTS": "INFRA", "ULTRACEMCO": "CEMENT",
    "GRASIM": "CEMENT", "TATASTEEL": "METALS", "JSWSTEEL": "METALS",
    "HINDALCO": "METALS", "BEL": "DEFENCE",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "HEALTHCARE",
    "BHARTIARTL": "TELECOM",
}

# Aliases used to match news headlines to universe symbols (lowercase).
COMPANY_ALIASES = {
    "RELIANCE": ["reliance", "ril", "reliance industries", "ambani"],
    "TCS": ["tcs", "tata consultancy"],
    "INFY": ["infosys", "infy"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "ICICIBANK": ["icici bank", "icicibank"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "ITC": ["itc"],
    "SBIN": ["sbi", "state bank"],
    "BHARTIARTL": ["airtel", "bharti"],
    "KOTAKBANK": ["kotak"],
    "LT": ["larsen", "l&t"],
    "AXISBANK": ["axis bank"],
    "ASIANPAINT": ["asian paints"],
    "MARUTI": ["maruti", "suzuki"],
    "SUNPHARMA": ["sun pharma"],
    "TITAN": ["titan"],
    "ULTRACEMCO": ["ultratech"],
    "WIPRO": ["wipro"],
    "HCLTECH": ["hcl tech", "hcltech", "hcl technologies"],
    "NTPC": ["ntpc"],
    "POWERGRID": ["power grid", "powergrid"],
    "TATASTEEL": ["tata steel"],
    "BAJFINANCE": ["bajaj finance"],
    "ADANIENT": ["adani enterprises", "adani ent"],
    "TECHM": ["tech mahindra"],
    "M&M": ["mahindra", "m&m"],
    "ADANIPORTS": ["adani ports"],
    "BAJAJFINSV": ["bajaj finserv"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "BEL": ["bharat electronics"],
    "CIPLA": ["cipla"],
    "COALINDIA": ["coal india"],
    "DRREDDY": ["dr reddy", "dr. reddy"],
    "EICHERMOT": ["eicher", "royal enfield"],
    "GRASIM": ["grasim"],
    "HDFCLIFE": ["hdfc life"],
    "HEROMOTOCO": ["hero motocorp", "hero moto"],
    "HINDALCO": ["hindalco"],
    "INDUSINDBK": ["indusind"],
    "JSWSTEEL": ["jsw steel"],
    "NESTLEIND": ["nestle"],
    "ONGC": ["ongc"],
    "SBILIFE": ["sbi life"],
    "SHRIRAMFIN": ["shriram finance"],
    "TATACONSUM": ["tata consumer"],
    "APOLLOHOSP": ["apollo hospitals"],
    "BRITANNIA": ["britannia"],
    "TRENT": ["trent", "westside"],
    "JIOFIN": ["jio financial"],
    "DIVISLAB": ["divis", "divi's"],
}

# Slow-moving policy variables that have no reliable free API. Update these
# when RBI/MoSPI publish new numbers, or override via data/config/macro.json.
POLICY_MACRO_DEFAULTS = {
    "repo_rate": 5.5,        # RBI repo rate
    "cpi_inflation": 2.1,    # CPI YoY %
    "bond_yield_10y": 6.3,   # 10Y G-Sec yield %
}


@dataclass
class PipelineConfig:
    """Tunable knobs for the end-to-end pipeline."""

    # Labeling: predict direction of the close `horizon_days` ahead.
    horizon_days: int = 1
    # Trading days of history to collect (~10 years). More history is the
    # single cheapest accuracy lever: it multiplies training rows for free.
    history_days: int = 2500
    # Movement larger than this (in %) counts as a directional move; smaller
    # moves are treated as flat and excluded from classifier training.
    flat_threshold_pct: float = 0.15
    # Chronological train/validation split ratio (no shuffling - time series).
    train_fraction: float = 0.8
    # Minimum validation accuracy a candidate model must reach to be
    # auto-approved into the registry.
    approval_min_accuracy: float = 0.5
    # Rolling windows used across feature engineering.
    windows: tuple = (5, 10, 20)
    # Random seed for reproducibility.
    seed: int = 42
    # Feature-completeness below this ratio blocks a prediction.
    min_feature_completeness: float = 0.7
    universe: list = field(default_factory=lambda: list(DEFAULT_UNIVERSE))


def ensure_dirs() -> None:
    """Create the on-disk folder layout described in the design document."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
