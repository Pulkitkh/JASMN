import pandas as pd
import pytest


def test_master_has_range_labels(trained_pipeline):
    master = trained_pipeline["master"]
    assert {"target_high_pct", "target_low_pct"} <= set(master.columns)
    labeled = master.dropna(subset=["target_high_pct", "target_low_pct"])
    # The touched high can't be below the touched low, and the close move
    # must lie inside the touched range.
    assert (labeled["target_high_pct"] >= labeled["target_low_pct"]).all()
    assert (labeled["target_move_pct"] <= labeled["target_high_pct"] + 1e-9).all()
    assert (labeled["target_move_pct"] >= labeled["target_low_pct"] - 1e-9).all()


def test_prediction_price_targets(trained_pipeline):
    from jasmin.prediction import predict

    pred = predict("TCS", config=trained_pipeline["config"])
    price = pred.price
    assert set(price) == {"last_close", "expected_close", "likely_high_touch", "likely_low_touch"}
    assert price["likely_low_touch"] <= price["expected_close"] <= price["likely_high_touch"]
    assert price["likely_low_touch"] <= price["last_close"] <= price["likely_high_touch"]
    assert price["last_close"] > 0


def test_training_reports_range_metrics(trained_pipeline):
    metrics = trained_pipeline["training"]
    assert metrics["high_touch_mae_pct"] >= 0
    assert metrics["low_touch_mae_pct"] >= 0


def test_resolve_symbol_passthrough(workspace):
    from jasmin.live.search import resolve_symbol

    # Exact tickers and universe members resolve without any network call.
    assert resolve_symbol("TCS", universe=["TCS"])["symbol"] == "TCS"
    assert resolve_symbol("RELIANCE.NS", universe=[])["symbol"] == "RELIANCE"
    assert resolve_symbol(" hdfcbank ", universe=["HDFCBANK"])["symbol"] == "HDFCBANK"


def test_analyze_uses_fast_path_for_universe_symbol(trained_pipeline, monkeypatch):
    from jasmin.prediction import analyze as analyze_mod

    # Network-dependent pieces must not be touched for a fresh universe symbol.
    monkeypatch.setattr(
        analyze_mod, "analyze_on_demand",
        lambda *a, **k: pytest.fail("on-demand path should not run"),
    )
    result = analyze_mod.analyze("RELIANCE", config=trained_pipeline["config"])
    assert result["resolved"]["symbol"] == "RELIANCE"
    assert result["prediction"]["direction"] in ("UP", "DOWN")
    assert "price" in result["prediction"]


def test_empty_news_aggregation(workspace):
    from jasmin.features.engineering import _aggregate_news

    empty = pd.DataFrame(columns=["date", "symbol", "headline", "weighted_sentiment", "event_type"])
    out = _aggregate_news(empty)
    assert out.empty
    assert str(out["date"].dtype) == "datetime64[ns]"


def test_regime_key_buckets(workspace):
    from jasmin.models.train import regime_key

    assert regime_key(10.0, 0.01) == "calm_up"
    assert regime_key(18.0, -0.01) == "elevated_down"
    assert regime_key(30.0, 0.0) == "stressed_up"
    assert regime_key(float("nan"), float("nan")) == "elevated_down"


def test_bundle_has_regime_accuracy_and_prediction_uses_it(trained_pipeline):
    from jasmin.models.registry import ModelRegistry
    from jasmin.prediction import predict

    _, bundle = ModelRegistry().latest_approved()
    assert "regime_accuracy" in bundle
    for stats in bundle["regime_accuracy"].values():
        assert 0.0 <= stats["accuracy"] <= 1.0
        assert stats["n"] > 0

    pred = predict("HDFCBANK", config=trained_pipeline["config"])
    assert "regime" in pred.confidence
    assert pred.confidence["regime"]["key"].count("_") == 1
    assert 0.0 <= pred.confidence["regime"]["accuracy_basis"] <= 1.0


def test_ensemble_has_three_members(trained_pipeline):
    from jasmin.models.registry import ModelRegistry

    _, bundle = ModelRegistry().latest_approved()
    assert set(bundle["classifiers"]) == {
        "gradient_boosting", "random_forest", "hist_gradient_boosting"
    }
