"""Test fixtures: redirect all JASMIN data/model paths into a temp dir so
tests never touch the real data folders, and run one shared offline
pipeline (collect -> dataset -> train) that the whole suite reuses."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="session")
def workspace(tmp_path_factory, monkeypatch_session=None):
    root = tmp_path_factory.mktemp("jasmin_ws")

    import os
    os.environ["JASMIN_ROOT"] = str(root)

    # Reload config-dependent modules so path constants pick up JASMIN_ROOT.
    import jasmin.config
    importlib.reload(jasmin.config)
    for mod in (
        "jasmin.utils.logging", "jasmin.collectors.base", "jasmin.collectors.prices",
        "jasmin.collectors.fundamentals", "jasmin.collectors.macro",
        "jasmin.collectors.institutional", "jasmin.collectors.news",
        "jasmin.collectors", "jasmin.validation.validators", "jasmin.validation",
        "jasmin.cleaning.cleaners", "jasmin.cleaning", "jasmin.features.technical",
        "jasmin.features.engineering", "jasmin.features", "jasmin.dataset.master",
        "jasmin.dataset", "jasmin.models.registry", "jasmin.models.train",
        "jasmin.models", "jasmin.prediction.confidence", "jasmin.prediction.explain",
        "jasmin.prediction.predict", "jasmin.prediction", "jasmin.pipeline",
    ):
        importlib.reload(importlib.import_module(mod))

    jasmin.config.ensure_dirs()
    return root


@pytest.fixture(scope="session")
def trained_pipeline(workspace):
    """Run one offline end-to-end cycle shared across the test session."""
    from jasmin.config import PipelineConfig
    from jasmin.pipeline import run_collectors
    from jasmin.dataset import build_master_dataset
    from jasmin.models.train import train_models

    config = PipelineConfig(universe=["RELIANCE", "TCS", "HDFCBANK"])
    run_collectors(config, days=420, offline=True)
    master = build_master_dataset(config)
    training = train_models(config)
    return {"config": config, "master": master, "training": training}
