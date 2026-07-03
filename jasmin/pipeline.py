"""Pipeline orchestration: the end-to-end flow from the design document.

collect -> validate -> clean -> features -> master dataset -> train ->
predict -> explain -> store. Each stage is also runnable on its own via
the CLI; this module just chains them.
"""

from __future__ import annotations

from jasmin.collectors import (
    FundamentalsCollector,
    InstitutionalCollector,
    MacroCollector,
    NewsCollector,
    PriceCollector,
)
from jasmin.config import PipelineConfig, ensure_dirs
from jasmin.dataset import build_master_dataset
from jasmin.models.train import train_models
from jasmin.prediction import predict
from jasmin.utils.logging import get_logger

log = get_logger("pipeline")


def run_collectors(config: PipelineConfig, days: int = 400, offline: bool = False) -> None:
    """Stage 1: run every collector independently."""
    ensure_dirs()
    for collector_cls in (
        PriceCollector, FundamentalsCollector, MacroCollector,
        InstitutionalCollector, NewsCollector,
    ):
        collector_cls(offline=offline).collect(config.universe, days=days)


def run_cycle(config: PipelineConfig | None = None, days: int = 400,
              offline: bool = False) -> dict:
    """One full continuous-learning cycle."""
    config = config or PipelineConfig()
    run_collectors(config, days=days, offline=offline)
    build_master_dataset(config)
    training = train_models(config)
    predictions = []
    for symbol in config.universe:
        try:
            predictions.append(predict(symbol, config=config).to_dict())
        except (ValueError, KeyError) as exc:
            log.warning("prediction skipped for %s: %s", symbol, exc)
    return {"training": training, "predictions": predictions}
