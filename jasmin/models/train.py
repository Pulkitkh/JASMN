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
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from jasmin.config import PipelineConfig
from jasmin.dataset import load_master_dataset
from jasmin.features.engineering import FEATURE_COLUMNS
from jasmin.models.registry import ModelRegistry
from jasmin.utils.logging import get_logger

log = get_logger("train")


def regime_key(vix: float, nifty_5d: float) -> str:
    """Bucket market conditions: VIX level x short-term index trend.

    Used for conditional accuracy — "how good has the model been in
    conditions like today's?" (design doc §16: historical accuracy on
    similar conditions).
    """
    if pd.isna(vix):
        vix = 18.0
    vol = "calm" if vix < 14 else ("elevated" if vix < 22 else "stressed")
    trend = "up" if (not pd.isna(nifty_5d) and nifty_5d >= 0) else "down"
    return f"{vol}_{trend}"


def _prepare(master: pd.DataFrame, config: PipelineConfig):
    labeled = master.dropna(subset=["target_up"]).copy()
    # Drop the warm-up window where long indicators are still NaN.
    features = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    labeled = labeled.dropna(subset=features, thresh=int(len(features) * 0.7))
    labeled[features] = labeled[features].fillna(labeled[features].median(numeric_only=True))
    # Columns that are entirely NaN (live sources that expose no history,
    # e.g. earnings surprises) have no median either — neutral-fill so the
    # sklearn ensembles never see NaN.
    labeled[features] = labeled[features].fillna(0.0)

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
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=config.seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            max_features="sqrt", random_state=config.seed, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=config.seed,
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

    # Range models: how far up/down the stock is likely to touch over the
    # horizon. Quantile loss gives "likely extreme" rather than the mean,
    # which is what a realistic touch estimate needs.
    high_regressor = GradientBoostingRegressor(
        loss="quantile", alpha=0.75, n_estimators=200, max_depth=3,
        learning_rate=0.05, random_state=config.seed,
    )
    low_regressor = GradientBoostingRegressor(
        loss="quantile", alpha=0.25, n_estimators=200, max_depth=3,
        learning_rate=0.05, random_state=config.seed,
    )
    range_train = train.dropna(subset=["target_high_pct", "target_low_pct"])
    high_regressor.fit(range_train[features], range_train["target_high_pct"])
    low_regressor.fit(range_train[features], range_train["target_low_pct"])
    range_valid = valid.dropna(subset=["target_high_pct", "target_low_pct"])
    high_mae = mean_absolute_error(
        range_valid["target_high_pct"], high_regressor.predict(range_valid[features])
    )
    low_mae = mean_absolute_error(
        range_valid["target_low_pct"], low_regressor.predict(range_valid[features])
    )

    metrics = {
        "ensemble_accuracy": round(float(ensemble_acc), 4),
        "roc_auc": round(float(auc), 4),
        "move_mae_pct": round(float(mae), 4),
        "high_touch_mae_pct": round(float(high_mae), 4),
        "low_touch_mae_pct": round(float(low_mae), 4),
        "per_model_accuracy": per_model_acc,
        "n_train": len(train),
        "n_valid": len(valid),
        "valid_from": str(valid["date"].min().date()),
        "valid_to": str(valid["date"].max().date()),
    }

    # Conditional accuracy per market regime on the validation window: at
    # prediction time, confidence uses the accuracy for today's regime
    # rather than the global average (falls back when a bucket is thin).
    regime_df = pd.DataFrame(
        {
            "regime": [
                regime_key(v, n)
                for v, n in zip(valid["india_vix"], valid["nifty_return_5d"])
            ],
            "correct": (proba_va > 0.5).astype(int) == y_va.to_numpy().astype(int),
        }
    )
    regime_accuracy = {
        regime: {"accuracy": round(float(grp["correct"].mean()), 4), "n": int(len(grp))}
        for regime, grp in regime_df.groupby("regime")
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
        # First model ever: the system needs a live model to serve, so the
        # bootstrap model is always approved; the gate governs replacements.
        if not approved:
            log.warning(
                "first model below approval threshold (%.3f); bootstrap-approving",
                ensemble_acc,
            )
        approved = True

    bundle = {
        "classifiers": classifiers,
        "regressor": regressor,
        "high_regressor": high_regressor,
        "low_regressor": low_regressor,
        "features": features,
        "feature_stats": feature_stats,
        "regime_accuracy": regime_accuracy,
        "metrics": metrics,
        "config": {"horizon_days": config.horizon_days},
    }
    version = registry.register(bundle, metrics, approved)
    return {"version": version, "approved": approved, **metrics}
