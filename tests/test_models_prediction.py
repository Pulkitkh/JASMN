import pytest


def test_training_registers_model(trained_pipeline):
    training = trained_pipeline["training"]
    assert training["version"].startswith("v")
    assert 0 <= training["ensemble_accuracy"] <= 1
    assert training["n_train"] > training["n_valid"] > 0


def test_registry_versioning_and_rollback(trained_pipeline):
    from jasmin.models.registry import ModelRegistry
    from jasmin.models.train import train_models

    registry = ModelRegistry()
    first = registry.list_versions()[-1]["version"]

    # Train a second version, force-approve both, then roll back.
    train_models(trained_pipeline["config"], registry=registry)
    versions = registry.list_versions()
    assert len(versions) >= 2
    second = versions[-1]["version"]
    registry.set_approval(first, True)
    registry.set_approval(second, True)

    live_before, _ = registry.latest_approved()
    assert live_before == second
    registry.rollback()
    live_after, _ = registry.latest_approved()
    assert live_after == first
    registry.set_approval(second, True)  # restore for other tests


def test_prediction_shape_and_explainability(trained_pipeline):
    from jasmin.prediction import predict

    pred = predict("RELIANCE", config=trained_pipeline["config"])
    assert pred.direction in ("UP", "DOWN")
    assert 0 <= pred.probability_up <= 1
    assert 0 <= pred.confidence["score"] <= 100
    assert set(pred.confidence["components"]) == {
        "probability_strength", "model_agreement", "feature_completeness",
        "data_freshness", "historical_accuracy", "volatility_regime",
    }
    assert pred.explanation["positive_factors"] or pred.explanation["negative_factors"]
    for factor in pred.explanation["positive_factors"]:
        assert factor["contribution"] > 0
        assert factor["description"]
    for factor in pred.explanation["negative_factors"]:
        assert factor["contribution"] < 0


def test_prediction_unknown_symbol(trained_pipeline):
    from jasmin.prediction import predict

    with pytest.raises(KeyError):
        predict("NOTASTOCK", config=trained_pipeline["config"])


def test_confidence_monotonicity(workspace):
    from jasmin.prediction.confidence import confidence_score

    kwargs = dict(member_probas=[0.7, 0.72], completeness=1.0, staleness_days=0,
                  validation_accuracy=0.6, india_vix=14)
    strong = confidence_score(proba_up=0.9, **kwargs)
    weak = confidence_score(proba_up=0.52, **kwargs)
    assert strong["score"] > weak["score"]

    calm = confidence_score(proba_up=0.7, **kwargs)
    stressed = confidence_score(proba_up=0.7, **{**kwargs, "india_vix": 35})
    assert calm["score"] > stressed["score"]
