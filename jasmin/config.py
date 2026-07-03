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

# Default NSE universe used for demos and tests. Extend freely; every
# pipeline stage takes an explicit symbol list, so nothing hardcodes this.
DEFAULT_UNIVERSE = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
]

SECTOR_MAP = {
    "RELIANCE": "ENERGY",
    "TCS": "IT",
    "INFY": "IT",
    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
}

# Aliases used to match news headlines to universe symbols (lowercase).
COMPANY_ALIASES = {
    "RELIANCE": ["reliance", "ril", "reliance industries", "ambani"],
    "TCS": ["tcs", "tata consultancy"],
    "INFY": ["infosys", "infy"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "ICICIBANK": ["icici bank", "icicibank"],
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
