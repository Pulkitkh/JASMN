import pandas as pd
import pytest


def _price_frame():
    dates = pd.bdate_range("2026-01-01", periods=30)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 101.0,
            "adj_close": 101.0,
            "volume": 1000,
        }
    )


def test_validate_prices_accepts_good_data(workspace):
    from jasmin.validation import validate_prices

    report = validate_prices(_price_frame())
    assert report.ok


def test_validate_prices_rejects_bad_ohlc(workspace):
    from jasmin.validation import validate_prices

    df = _price_frame()
    df.loc[3, "high"] = 50.0  # high below low
    report = validate_prices(df)
    assert not report.ok
    with pytest.raises(ValueError):
        report.raise_if_failed()


def test_validate_missing_columns(workspace):
    from jasmin.validation import validate_prices

    report = validate_prices(_price_frame().drop(columns=["volume"]))
    assert not report.ok


def test_clean_prices_derives_returns_and_gaps(workspace):
    from jasmin.cleaning import clean_prices

    df = _price_frame()
    cleaned = clean_prices(df)
    assert {"return_1d", "gap_pct", "return_5d", "return_20d"} <= set(cleaned.columns)
    assert cleaned["return_1d"].isna().sum() == 0


def test_clean_prices_dedupes(workspace):
    from jasmin.cleaning import clean_prices

    df = pd.concat([_price_frame(), _price_frame()], ignore_index=True)
    cleaned = clean_prices(df)
    assert len(cleaned) == 30
