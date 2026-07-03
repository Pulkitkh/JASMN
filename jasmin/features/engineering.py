"""Feature engineering (design doc §12) — pipeline stage 4.

Merges every cleaned domain onto a per-symbol daily grid and derives the
engineered features the model consumes: rolling stats, momentum, volatility,
lags, relative strength vs sector and index, weighted sentiment windows,
macro deltas, institutional flow trends, event flags, temporal/seasonality
features and interaction terms.

`FEATURE_DESCRIPTIONS` doubles as the human-readable vocabulary used by the
explanation engine, so every model feature has a plain-English meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jasmin.config import SECTOR_MAP
from jasmin.features.technical import add_technical_indicators

FEATURE_DESCRIPTIONS = {
    # technical
    "rsi_14": "14-day RSI (momentum)",
    "macd_hist": "MACD histogram (trend momentum)",
    "macd_bullish_cross": "bullish MACD crossover",
    "adx_14": "ADX trend strength",
    "bb_pct_b": "position inside Bollinger Bands",
    "bb_width": "Bollinger Band width (volatility)",
    "stoch_k": "stochastic %K (overbought/oversold)",
    "cci_20": "commodity channel index",
    "williams_r_14": "Williams %R",
    "roc_10": "10-day rate of change",
    "ma_cross_10_50": "10/50-day moving-average spread",
    "price_vs_vwap": "price premium over 20-day VWAP",
    "atr_pct": "ATR relative to price (volatility)",
    # price behaviour
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "return_20d": "20-day return",
    "gap_pct": "opening gap",
    "volatility_20d": "20-day realized volatility",
    "volume_ratio_20d": "volume vs 20-day average (unusual volume)",
    "return_lag_1": "previous day's return",
    "return_lag_2": "return two days ago",
    "dist_from_high_20d": "distance below 20-day high",
    # relative strength
    "rel_strength_index_20d": "20-day relative strength vs NIFTY",
    "rel_strength_sector_20d": "20-day return vs own sector",
    # sentiment
    "news_sentiment_3d": "3-day weighted news sentiment",
    "news_sentiment_7d": "7-day weighted news sentiment",
    "news_volume_7d": "news flow intensity (7 days)",
    "earnings_event_flag": "earnings news in the last 3 days",
    # fundamentals
    "pe": "price-to-earnings ratio",
    "roe": "return on equity",
    "debt_equity": "debt-to-equity ratio",
    "revenue_growth_qoq": "quarterly revenue growth",
    "earnings_surprise_pct": "latest earnings surprise",
    "promoter_holding": "promoter shareholding",
    # macro
    "repo_rate": "RBI repo rate",
    "cpi_delta_20d": "20-day CPI inflation change",
    "usdinr_return_5d": "5-day USD/INR move",
    "crude_return_5d": "5-day crude oil move",
    "india_vix": "India VIX (market fear)",
    "bond_yield_10y": "10-year bond yield",
    "nifty_return_5d": "5-day NIFTY return",
    # institutional
    "fii_net_5d": "5-day cumulative FII flows",
    "dii_net_5d": "5-day cumulative DII flows",
    "fii_trend_20d": "20-day FII flow trend",
    "promoter_pledge_pct": "promoter pledge level",
    "insider_net_5d": "5-day net insider trades",
    "deal_activity_5d": "bulk/block deal activity (5 days)",
    # temporal
    "day_of_week": "day of the week",
    "month": "calendar month",
    "is_expiry_week": "F&O expiry week",
    # interactions
    "sentiment_x_volume": "news sentiment amplified by unusual volume",
    "rsi_x_vix": "RSI conditioned on market fear",
}

FEATURE_COLUMNS = list(FEATURE_DESCRIPTIONS.keys())


def _is_expiry_week(dates: pd.Series) -> pd.Series:
    """NSE monthly F&O expiry: last Thursday of the month."""
    def last_thursday(d: pd.Timestamp) -> pd.Timestamp:
        end = d + pd.offsets.MonthEnd(0)
        return end - pd.Timedelta(days=(end.weekday() - 3) % 7)

    lt = dates.map(last_thursday)
    return ((lt - dates).dt.days.between(0, 4)).astype(int)


def _aggregate_news(news: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-headline rows into per-symbol daily sentiment metrics."""
    if news.empty:
        return pd.DataFrame(columns=["date", "symbol", "daily_sentiment", "news_count", "earnings_flag"])
    daily = (
        news.groupby(["date", "symbol"])
        .agg(
            daily_sentiment=("weighted_sentiment", "mean"),
            news_count=("headline", "count"),
            earnings_flag=("event_type", lambda s: int((s == "earnings").any())),
        )
        .reset_index()
    )
    return daily


def build_feature_frame(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    macro: pd.DataFrame,
    institutional: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all cleaned domains and compute the model feature matrix."""
    # Bookkeeping columns (collection metadata) must not enter the merges.
    _meta = ["collected_at", "source"]
    prices = prices.drop(columns=_meta, errors="ignore")
    fundamentals = fundamentals.drop(columns=_meta, errors="ignore")
    macro = macro.drop(columns=_meta, errors="ignore")
    institutional = institutional.drop(columns=_meta, errors="ignore")

    news_daily = _aggregate_news(news)
    macro = macro.sort_values("date")

    frames = []
    for sym, grp in prices.groupby("symbol", sort=False):
        grp = add_technical_indicators(grp)

        # --- price-derived features ---
        grp["volatility_20d"] = grp["return_1d"].rolling(20).std() * np.sqrt(252)
        grp["volume_ratio_20d"] = grp["volume"] / grp["volume"].rolling(20).mean().replace(0, np.nan)
        grp["return_lag_1"] = grp["return_1d"].shift(1)
        grp["return_lag_2"] = grp["return_1d"].shift(2)
        grp["dist_from_high_20d"] = grp["close"] / grp["high"].rolling(20).max() - 1
        grp["atr_pct"] = grp["atr_14"] / grp["close"]

        # --- fundamentals: as-of merge of the latest quarterly snapshot ---
        fund_sym = fundamentals[fundamentals["symbol"] == sym].sort_values("date")
        if not fund_sym.empty:
            grp = pd.merge_asof(
                grp.sort_values("date"),
                fund_sym.drop(columns=["symbol", "collected_at"], errors="ignore"),
                on="date",
                direction="backward",
            )

        # --- macro: daily join + deltas ---
        grp = pd.merge_asof(
            grp.sort_values("date"),
            macro.drop(columns=["collected_at"], errors="ignore"),
            on="date",
            direction="backward",
        )
        grp["cpi_delta_20d"] = grp["cpi_inflation"].diff(20)
        grp["usdinr_return_5d"] = grp["usdinr"].pct_change(5)
        grp["crude_return_5d"] = grp["crude_usd"].pct_change(5)
        grp["nifty_return_5d"] = grp["nifty_index"].pct_change(5)

        # --- relative strength vs index ---
        grp["rel_strength_index_20d"] = (
            grp["close"].pct_change(20) - grp["nifty_index"].pct_change(20)
        )

        # --- institutional flows ---
        inst_sym = institutional[institutional["symbol"] == sym].sort_values("date")
        if not inst_sym.empty:
            grp = pd.merge_asof(
                grp,
                inst_sym.drop(columns=["symbol", "collected_at"], errors="ignore"),
                on="date",
                direction="backward",
            )
            grp["fii_net_5d"] = grp["fii_net_cr"].rolling(5).sum()
            grp["dii_net_5d"] = grp["dii_net_cr"].rolling(5).sum()
            grp["fii_trend_20d"] = grp["fii_net_cr"].rolling(20).mean()
            grp["insider_net_5d"] = grp["insider_net_trades"].rolling(5).sum()
            grp["deal_activity_5d"] = (grp["bulk_deals"] + grp["block_deals"]).rolling(5).sum()

        # --- news sentiment windows ---
        nd = news_daily[news_daily["symbol"] == sym].drop(columns=["symbol"])
        grp = grp.merge(nd, on="date", how="left")
        grp["daily_sentiment"] = grp["daily_sentiment"].fillna(0.0)
        grp["news_count"] = grp["news_count"].fillna(0)
        grp["earnings_flag"] = grp["earnings_flag"].fillna(0)
        grp["news_sentiment_3d"] = grp["daily_sentiment"].rolling(3, min_periods=1).mean()
        grp["news_sentiment_7d"] = grp["daily_sentiment"].rolling(7, min_periods=1).mean()
        grp["news_volume_7d"] = grp["news_count"].rolling(7, min_periods=1).sum()
        grp["earnings_event_flag"] = (
            grp["earnings_flag"].rolling(3, min_periods=1).max().astype(int)
        )

        # --- temporal ---
        grp["day_of_week"] = grp["date"].dt.dayofweek
        grp["month"] = grp["date"].dt.month
        grp["is_expiry_week"] = _is_expiry_week(grp["date"])

        grp["sector"] = SECTOR_MAP.get(sym, "OTHER")
        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)

    # --- sector-relative strength (needs all symbols together) ---
    df["ret_20d_tmp"] = df.groupby("symbol")["close"].pct_change(20)
    sector_ret = df.groupby(["date", "sector"])["ret_20d_tmp"].transform("mean")
    df["rel_strength_sector_20d"] = df["ret_20d_tmp"] - sector_ret
    df = df.drop(columns=["ret_20d_tmp"])

    # --- interaction features ---
    df["sentiment_x_volume"] = df["news_sentiment_3d"] * df["volume_ratio_20d"]
    df["rsi_x_vix"] = (df["rsi_14"] - 50) * df["india_vix"] / 100

    return df
