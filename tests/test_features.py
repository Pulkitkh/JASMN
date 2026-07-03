import numpy as np
import pandas as pd


def _ohlcv(days=120, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-01", periods=days)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, days)))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "open": close * (1 + rng.normal(0, 0.005, days)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, days),
        }
    )


def test_indicator_ranges(workspace):
    from jasmin.features.technical import add_technical_indicators

    out = add_technical_indicators(_ohlcv())
    tail = out.iloc[60:]  # past indicator warm-up
    assert tail["rsi_14"].between(0, 100).all()
    assert tail["stoch_k"].between(0, 100).all()
    assert tail["williams_r_14"].between(-100, 0).all()
    assert tail["adx_14"].between(0, 100).all()
    assert (tail["atr_14"] > 0).all()
    assert (tail["bb_upper"] >= tail["bb_lower"]).all()


def test_macd_crossover_flag_is_binary(workspace):
    from jasmin.features.technical import add_technical_indicators

    out = add_technical_indicators(_ohlcv())
    assert set(out["macd_bullish_cross"].unique()) <= {0, 1}


def test_feature_frame_has_all_model_features(trained_pipeline):
    from jasmin.features.engineering import FEATURE_COLUMNS

    master = trained_pipeline["master"]
    missing = [c for c in FEATURE_COLUMNS if c not in master.columns]
    assert missing == []


def test_labels_align_with_future_close(trained_pipeline):
    master = trained_pipeline["master"]
    grp = master[master["symbol"] == "RELIANCE"].sort_values("date").reset_index(drop=True)
    horizon = trained_pipeline["config"].horizon_days
    expected = (grp["close"].shift(-horizon) / grp["close"] - 1) * 100
    pd.testing.assert_series_equal(
        grp["target_move_pct"], expected, check_names=False, atol=1e-6
    )
    # Unlabeled tail rows exist for inference only.
    assert grp["target_up"].tail(horizon).isna().all()
