from datetime import datetime

import pandas as pd


def test_trading_day_rules(workspace):
    from jasmin.utils.market_calendar import IST, is_market_open, is_trading_day

    saturday = datetime(2026, 7, 4, 11, 0, tzinfo=IST)
    assert not is_trading_day(saturday)
    republic_day = datetime(2026, 1, 26, 11, 0, tzinfo=IST)
    assert not is_trading_day(republic_day)

    weekday = datetime(2026, 7, 2, 11, 0, tzinfo=IST)
    assert is_trading_day(weekday)
    assert is_market_open(weekday)
    before_open = datetime(2026, 7, 2, 9, 0, tzinfo=IST)
    assert not is_market_open(before_open)


def test_next_premarket_run_is_future_trading_day(workspace):
    from jasmin.utils.market_calendar import IST, PREMARKET_RUN, is_trading_day, next_premarket_run

    friday_noon = datetime(2026, 7, 3, 12, 0, tzinfo=IST)
    nxt = next_premarket_run(friday_noon)
    assert nxt > friday_noon
    assert nxt.time().hour == PREMARKET_RUN.hour
    assert is_trading_day(nxt)
    assert nxt.weekday() == 0  # Saturday/Sunday skipped -> Monday


_SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Markets</title>
<item><title>Infosys wins large deal, shares surge</title>
<pubDate>Thu, 02 Jul 2026 10:30:00 +0530</pubDate></item>
<item><title>HDFC Bank Q1 results beat estimates</title>
<pubDate>Thu, 02 Jul 2026 09:00:00 +0530</pubDate></item>
<item><title>Weather update for Mumbai</title>
<pubDate>Thu, 02 Jul 2026 08:00:00 +0530</pubDate></item>
</channel></rss>"""


def test_parse_rss_and_symbol_matching(workspace):
    from jasmin.live.rss import match_symbol, parse_rss

    items = parse_rss(_SAMPLE_RSS)
    assert len(items) == 3
    assert items[0]["headline"].startswith("Infosys")
    assert items[0]["date"] == pd.Timestamp("2026-07-02")

    universe = ["INFY", "HDFCBANK", "TCS"]
    assert match_symbol(items[0]["headline"], universe) == "INFY"
    assert match_symbol(items[1]["headline"], universe) == "HDFCBANK"
    assert match_symbol(items[2]["headline"], universe) is None
    # Alias must match whole words: "ril" should not fire inside "thriller".
    assert match_symbol("A thriller finish to the trading day", ["RELIANCE"]) is None
    assert match_symbol("RIL announces expansion", ["RELIANCE"]) == "RELIANCE"


def test_market_summary_aggregation(trained_pipeline):
    from jasmin.prediction import predict
    from jasmin.prediction.summary import build_market_summary

    preds = [predict(s, config=trained_pipeline["config"]) for s in ("RELIANCE", "TCS", "HDFCBANK")]
    summary = build_market_summary(preds)

    assert summary["market_bias"] in ("BULLISH", "BEARISH", "NEUTRAL")
    assert summary["n_up"] + summary["n_down"] + summary["n_neutral"] == 3
    assert len(summary["predictions"]) == 3
    assert set(summary["sectors"]) <= {"ENERGY", "IT", "BANKING", "OTHER"}
    for pick in summary["top_bullish"]:
        assert pick["direction"] == "UP"
    for pick in summary["top_bearish"]:
        assert pick["direction"] == "DOWN"


def test_market_status_shape(workspace):
    from jasmin.utils.market_calendar import market_status

    status = market_status()
    assert set(status) == {"now_ist", "trading_day", "market_open", "next_open_ist"}
