import pandas as pd


def test_price_collector_offline(workspace):
    from jasmin.collectors import PriceCollector

    df = PriceCollector(offline=True).collect(["RELIANCE", "TCS"], days=100)
    assert set(["date", "symbol", "open", "high", "low", "close", "volume"]) <= set(df.columns)
    assert df["symbol"].nunique() == 2
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()


def test_price_collector_deterministic(workspace):
    from jasmin.collectors import PriceCollector

    a = PriceCollector(offline=True).fetch(["INFY"], days=50)
    b = PriceCollector(offline=True).fetch(["INFY"], days=50)
    pd.testing.assert_frame_equal(a, b)


def test_news_collector_enrichment(workspace):
    from jasmin.collectors import NewsCollector

    df = NewsCollector(offline=True).collect(["RELIANCE"], days=200)
    assert {"event_type", "sentiment", "event_weight", "confidence", "weighted_sentiment"} <= set(df.columns)
    assert df["sentiment"].between(-1, 1).all()
    assert df["event_type"].isin(
        ["earnings", "merger_acquisition", "regulatory", "management_change",
         "product_launch", "analyst_rating", "macro", "general"]
    ).all()


def test_sentiment_scorer_direction(workspace):
    from jasmin.collectors.news import SentimentScorer

    scorer = SentimentScorer()
    assert scorer.score("Company beats estimates with record profit growth") > 0
    assert scorer.score("Company misses estimates as losses surge and shares plunge") < 0
    assert scorer.score("Board meeting scheduled on Tuesday") == 0.0


def test_macro_and_institutional_shapes(workspace):
    from jasmin.collectors import InstitutionalCollector, MacroCollector

    macro = MacroCollector(offline=True).fetch([], days=60)
    assert {"repo_rate", "cpi_inflation", "usdinr", "india_vix"} <= set(macro.columns)
    assert "symbol" not in macro.columns

    inst = InstitutionalCollector(offline=True).fetch(["TCS"], days=60)
    assert {"fii_net_cr", "dii_net_cr", "promoter_pledge_pct"} <= set(inst.columns)
