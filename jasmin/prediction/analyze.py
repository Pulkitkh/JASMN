"""On-demand analysis: any stock name/ticker -> full prediction vs the market.

This is the "type a stock, get an answer" flow:

  1. Resolve the query to an NSE symbol (Yahoo search; exact tickers pass
     straight through).
  2. Universe symbols with a fresh master-dataset row reuse `predict`.
  3. Anything else is built live: 2 years of the stock's prices and its
     fundamentals snapshot are fetched on the spot, then merged with the
     same market-wide context every universe stock uses (macro series,
     FII/DII flows, news log) through the identical cleaning and feature
     engineering — so the trained model scores an unseen stock exactly the
     way it scores a trained one. Features are symbol-relative (returns,
     ratios, z-scores), which is what makes cross-symbol generalization
     legitimate.

The response adds the resolved company name, the market backdrop, and the
price block (expected close, likely high/low touch) from the range models.
"""

from __future__ import annotations

import pandas as pd

from jasmin.cleaning import clean_daily_domain, clean_prices, clean_quarterly_domain
from jasmin.collectors.fundamentals import FundamentalsCollector
from jasmin.collectors.institutional import InstitutionalCollector
from jasmin.collectors.macro import MacroCollector
from jasmin.collectors.news import NewsCollector
from jasmin.config import RAW_DIR, PipelineConfig
from jasmin.features.engineering import build_feature_frame
from jasmin.live.yahoo import YahooClient
from jasmin.models.registry import ModelRegistry
from jasmin.prediction.predict import Prediction, infer_row, predict
from jasmin.utils.logging import get_logger

log = get_logger("analyze")


def analyze(query: str, config: PipelineConfig | None = None,
            registry: ModelRegistry | None = None) -> dict:
    """Analyze any stock by name or ticker against the market."""
    from jasmin.live.search import resolve_symbol
    from jasmin.prediction.summary import _market_context

    config = config or PipelineConfig()
    registry = registry or ModelRegistry()

    resolved = resolve_symbol(query, universe=config.universe)
    symbol = resolved["symbol"]

    prediction = _predict_known_or_ondemand(symbol, resolved, config, registry)
    return {
        "query": query,
        "resolved": resolved,
        "prediction": prediction.to_dict(),
        "market_context": _market_context(),
    }


def _predict_known_or_ondemand(symbol: str, resolved: dict,
                               config: PipelineConfig,
                               registry: ModelRegistry) -> Prediction:
    # Universe symbols with a fresh dataset row take the fast path.
    try:
        from jasmin.dataset import load_master_dataset

        master = load_master_dataset()
        rows = master[master["symbol"] == symbol]
        if not rows.empty:
            staleness = (
                pd.Timestamp.today().normalize() - rows["date"].max().normalize()
            ).days
            if staleness <= 5:
                return predict(symbol, config=config, registry=registry)
    except FileNotFoundError:
        pass
    return analyze_on_demand(symbol, resolved, config, registry)


def analyze_on_demand(symbol: str, resolved: dict, config: PipelineConfig,
                      registry: ModelRegistry, days: int = 400) -> Prediction:
    """Build features live for a symbol outside the master dataset."""
    version, bundle = registry.latest_approved()
    yahoo = YahooClient()

    log.info("on-demand analysis for %s (%s)", symbol, resolved.get("name", ""))
    prices = yahoo.daily_history(symbol, days)  # raises if the ticker is bogus
    prices.insert(1, "symbol", symbol)
    prices["delivery_pct"] = float("nan")
    prices = clean_prices(prices)

    fundamentals = _fundamentals_frame(symbol)
    macro = clean_daily_domain(_market_macro(days))
    institutional = clean_daily_domain(
        InstitutionalCollector(offline=False).fetch([symbol], days)
    )
    news = _symbol_news(symbol, resolved, days)

    features = build_feature_frame(prices, fundamentals, macro, institutional, news)
    latest = features.sort_values("date").iloc[-1]

    completeness = float(latest[bundle["features"]].notna().mean())
    staleness = (pd.Timestamp.today().normalize() - latest["date"].normalize()).days
    return infer_row(
        bundle, version, symbol, latest,
        as_of=str(latest["date"].date()),
        last_close=float(latest["close"]),
        completeness=completeness,
        staleness_days=max(staleness - 1, 0),
    )


def _fundamentals_frame(symbol: str) -> pd.DataFrame:
    try:
        snap = YahooClient().fundamentals_snapshot(symbol)
    except Exception as exc:
        log.warning("fundamentals unavailable for %s (%s)", symbol, exc)
        snap = {f: float("nan") for f in FundamentalsCollector.FIELDS}
    frame = pd.DataFrame([{"date": pd.Timestamp("2000-01-01"), "symbol": symbol, **snap}])
    return clean_quarterly_domain(frame)


def _market_macro(days: int) -> pd.DataFrame:
    """Reuse the universe's raw macro file when fresh; refetch otherwise."""
    path = RAW_DIR / "macro" / "macro.csv"
    if path.exists():
        macro = pd.read_csv(path, parse_dates=["date"])
        if (pd.Timestamp.today().normalize() - macro["date"].max()).days <= 4:
            return macro
    return MacroCollector(offline=False).fetch([], days)


def _symbol_news(symbol: str, resolved: dict, days: int) -> pd.DataFrame:
    """Live headlines matched against the symbol's resolved company name."""
    from jasmin.live.rss import fetch_news

    name = str(resolved.get("name", "")).lower()
    # "TATA MOTORS PASS VEH LTD" -> "tata motors pass veh" -> useful prefixes.
    words = [w for w in name.split() if w not in ("ltd", "limited", "the", "of", "india")]
    aliases = [symbol.lower()]
    if len(words) >= 2:
        aliases.append(" ".join(words[:2]))
    elif words:
        aliases.append(words[0])
    try:
        matched = fetch_news([symbol], extra_aliases={symbol: aliases})
        matched = matched[matched["symbol"] == symbol]
    except Exception as exc:
        log.warning("news unavailable for %s (%s)", symbol, exc)
        matched = pd.DataFrame(columns=["date", "symbol", "headline"])
    return NewsCollector(offline=True)._enrich(matched) if not matched.empty \
        else _empty_news()


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "headline", "event_type", "sentiment",
                 "event_weight", "confidence", "weighted_sentiment"]
    )
