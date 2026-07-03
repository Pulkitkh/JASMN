"""Model training (design doc §13, §17) — pipeline stage 6.

Ensemble of gradient boosting + random forest classifiers for direction
(both from scikit-learn, so the base install stays light; XGBoost/LightGBM
drop in behind the same interface), plus a gradient boosting regressor for
expected % movement. Chronological train/validation split — never shuffled.

A candidate is auto-approved only if it beats the approval threshold AND
does not underperform the previously approved model by more than 2pp
(design doc: "validation against previous models before deployment").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from jasmin.config import PipelineConfig
from jasmin.dataset import load_master_dataset
from jasmin.features.engineering import FEATURE_COLUMNS
from jasmin.models.registry import ModelRegistry
from jasmin.utils.logging import get_logger

log = get_logger("train")


def _prepare(master: pd.DataFrame, config: PipelineConfig):
    labeled = master.dropna(subset=["target_up"]).copy()
    # Drop the warm-up window where long indicators are still NaN.
    features = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    labeled = labeled.dropna(subset=features, thresh=int(len(features) * 0.9))
    labeled[features] = labeled[features].fillna(labeled[features].median(numeric_only=True))

    # Flat moves teach the classifier nothing about direction.
    directional = labeled[labeled["target_move_pct"].abs() >= config.flat_threshold_pct]

    directional = directional.sort_values("date")
    split_date = directional["date"].quantile(config.train_fraction)
    train = directional[directional["date"] <= split_date]
    valid = directional[directional["date"] > split_date]
    return train, valid, features


def train_models(config: PipelineConfig | None = None, registry: ModelRegistry | None = None) -> dict:
    config = config or PipelineConfig()
    registry = registry or ModelRegistry()

    master = load_master_dataset()
    train, valid, features = _prepare(master, config)
    if len(train) < 100 or len(valid) < 20:
        raise ValueError(
            f"Not enough labeled data to train (train={len(train)}, valid={len(valid)}). "
            "Collect a longer history."
        )
    log.info("training on %d rows, validating on %d rows, %d features",
             len(train), len(valid), len(features))

    X_tr, y_tr = train[features], train["target_up"]
    X_va, y_va = valid[features], valid["target_up"]

    classifiers = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=config.seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            random_state=config.seed, n_jobs=-1
        ),
    }
    per_model_acc = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        per_model_acc[name] = round(accuracy_score(y_va, clf.predict(X_va)), 4)

    # Ensemble = mean of member probabilities.
    proba_va = np.mean([c.predict_proba(X_va)[:, 1] for c in classifiers.values()], axis=0)
    ensemble_acc = accuracy_score(y_va, (proba_va > 0.5).astype(int))
    try:
        auc = roc_auc_score(y_va, proba_va)
    except ValueError:
        auc = float("nan")

    regressor = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=config.seed
    )
    regressor.fit(X_tr, train["target_move_pct"])
    mae = mean_absolute_error(valid["target_move_pct"], regressor.predict(X_va))

    metrics = {
        "ensemble_accuracy": round(float(ensemble_acc), 4),
        "roc_auc": round(float(auc), 4),
        "move_mae_pct": round(float(mae), 4),
        "per_model_accuracy": per_model_acc,
        "n_train": len(train),
        "n_valid": len(valid),
        "valid_from": str(valid["date"].min().date()),
        "valid_to": str(valid["date"].max().date()),
    }

    # Feature stats from training data feed the explanation engine (z-scores).
    feature_stats = {
        "mean": X_tr.mean().to_dict(),
        "std": X_tr.std().replace(0, 1.0).to_dict(),
        "median": X_tr.median().to_dict(),
    }

    approved = ensemble_acc >= config.approval_min_accuracy
    try:
        _, previous = registry.latest_approved()
        prev_acc = previous["metrics"]["ensemble_accuracy"]
        if ensemble_acc < prev_acc - 0.02:
            approved = False
            log.warning(
                "candidate accuracy %.3f is worse than approved model (%.3f); not auto-approving",
                ensemble_acc, prev_acc,
            )
    except FileNotFoundError:
        pass  # first model ever: threshold check alone decides

    bundle = {
        "classifiers": classifiers,
        "regressor": regressor,
        "features": features,
        "feature_stats": feature_stats,
        "metrics": metrics,
        "config": {"horizon_days": config.horizon_days},
    }
    version = registry.register(bundle, metrics, approved)
    return {"version": version, "approved": approved, **metrics}
