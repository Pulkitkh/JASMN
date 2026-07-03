"""Technical indicators (design doc §7).

Pure-pandas implementations of the planned indicator set: SMA, EMA, RSI,
MACD (+signal/histogram), ATR, ADX, Bollinger Bands, VWAP, OBV, Stochastic,
ROC, Momentum, CCI, Williams %R, Ichimoku components and MA crossovers.
All functions operate on a single-symbol OHLCV frame sorted by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(line, 9)
    return line, signal, line - signal


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, width, pct_b


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).rolling(period).sum()
    vol = df["volume"].rolling(period).sum().replace(0, np.nan)
    return pv / vol


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def stochastic(df: pd.DataFrame, period: int = 14):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    return k, k.rolling(3).mean()


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def ichimoku(df: pd.DataFrame):
    conv = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    base = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
    return conv, base, span_a, span_b


def add_technical_indicators(grp: pd.DataFrame) -> pd.DataFrame:
    """Append the full indicator set to one symbol's OHLCV frame."""
    grp = grp.sort_values("date").reset_index(drop=True)
    close = grp["close"]

    for w in (10, 20, 50):
        grp[f"sma_{w}"] = close.rolling(w).mean()
    for w in (12, 26):
        grp[f"ema_{w}"] = _ema(close, w)

    grp["rsi_14"] = rsi(close)
    grp["macd"], grp["macd_signal"], grp["macd_hist"] = macd(close)
    grp["atr_14"] = atr(grp)
    grp["adx_14"] = adx(grp)
    grp["bb_upper"], grp["bb_lower"], grp["bb_width"], grp["bb_pct_b"] = bollinger(close)
    grp["vwap_20"] = vwap(grp)
    grp["obv"] = obv(grp)
    grp["stoch_k"], grp["stoch_d"] = stochastic(grp)
    grp["roc_10"] = close.pct_change(10) * 100
    grp["momentum_10"] = close - close.shift(10)
    grp["cci_20"] = cci(grp)
    grp["williams_r_14"] = williams_r(grp)
    (grp["ichimoku_conv"], grp["ichimoku_base"],
     grp["ichimoku_span_a"], grp["ichimoku_span_b"]) = ichimoku(grp)

    # Crossover signals (normalized to price so they compare across symbols).
    grp["ma_cross_10_50"] = (grp["sma_10"] - grp["sma_50"]) / close
    grp["price_vs_vwap"] = (close - grp["vwap_20"]) / grp["vwap_20"]
    grp["macd_bullish_cross"] = (
        (grp["macd"] > grp["macd_signal"])
        & (grp["macd"].shift(1) <= grp["macd_signal"].shift(1))
    ).astype(int)
    return grp
