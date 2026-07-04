# JASMIN AI

An explainable, AI-driven market intelligence platform for Indian equities. JASMIN combines technical behaviour, company fundamentals, macroeconomics, institutional activity, news sentiment and corporate events into one prediction pipeline — and every prediction ships with a plain-English explanation and a confidence score, making it a decision-support tool rather than blind automation.

> Implements the [JASMIN AI Comprehensive System Design](docs/DESIGN.md): India-first architecture, modular collectors, explainable predictions, continuous learning.

## Architecture

```
Data Sources → Collectors → Validation → Cleaning → Feature Engineering
     → Master Dataset → Model Training → Prediction → Explanation → API
```

Every stage is an independent module — any collector, feature generator or model can be upgraded without touching the rest of the system.

| Module | Purpose |
|---|---|
| `jasmin/collectors/` | Prices (OHLCV), fundamentals, macro, institutional (FII/DII, deals, pledges), news intelligence. Live by default, with a deterministic synthetic fallback when the network is unavailable. |
| `jasmin/live/` | Standard-library HTTP clients for the live sources: Yahoo Finance (prices, fundamentals, macro tickers), the official NSE API (FII/DII flows) and Indian financial RSS feeds — with rate-limit rotation, request pacing and a short-TTL disk cache. |
| `jasmin/validation/` | Integrity checks: schema, OHLC consistency, duplicates, missing sessions. Errors block the pipeline; warnings pass through. |
| `jasmin/cleaning/` | Dedupe, normalization, gap filling, outlier winsorization, derived returns/gaps. |
| `jasmin/features/` | 20+ technical indicators (RSI, MACD, ADX, Bollinger, Ichimoku, VWAP, OBV, …) plus engineered features: relative strength, weighted sentiment windows, macro deltas, FII flow trends, expiry-week seasonality, interaction terms. |
| `jasmin/dataset/` | Builds the labeled master dataset (direction + expected % move over the horizon). |
| `jasmin/models/` | Gradient-boosting + random-forest ensemble (direction) and a regressor (expected move), with a versioned registry, auto-approval gates and rollback. |
| `jasmin/prediction/` | Inference pipeline: completeness validation → ensemble probability → expected move → ranked factor explanation → confidence score. |
| `jasmin/api/` | FastAPI service exposing predictions, the audit log and the model registry. |
| `jasmin/scheduler/` | Continuous-learning daemons: fixed-interval, plus a market-aware live daemon keyed to the NSE calendar. |

## Quick start

```bash
pip install -e .
jasmin premarket            # live data -> train -> predict all -> market summary
jasmin analyze tata steel   # analyze ANY stock by name or ticker
jasmin cycle --offline      # same pipeline on synthetic data (no network needed)
```

## Analyze any stock

`jasmin analyze <name or ticker>` (or `GET /analyze?q=...`) takes free text — `"tata steel"`, `"INFY"`, `"asian paints"` — resolves it to an NSE listing, and analyzes it against the market. Stocks in the trained universe use their master-dataset features; anything else is built live on the spot: 2 years of the stock's prices and its current fundamentals are fetched, merged with the same market context (macro, FII/DII flows, news), and scored by the trained model. This works because every feature is symbol-relative (returns, ratios, relative strength) rather than absolute.

The answer includes **price targets** from dedicated quantile range models trained on how far stocks actually travel in a session:

```json
{
  "direction": "UP",
  "probability_up": 0.58,
  "expected_move_pct": 0.31,
  "price": {
    "last_close": 3450.10,
    "expected_close": 3460.80,
    "likely_high_touch": 3492.55,
    "likely_low_touch": 3421.30
  },
  "confidence": {"score": 61.2},
  "explanation": {"summary": "Supported by ..."}
}
```

Or stage by stage:

```bash
jasmin collect --days 400   # 1. run all collectors (add --offline to force synthetic)
jasmin build-dataset        # 2-5. validate, clean, engineer features, label
jasmin train                # 6. train & register a model version
jasmin predict RELIANCE TCS # 7-10. predict with explanation + confidence
jasmin serve                # start the API on :8000
```

## Ready at market open

`jasmin live-daemon` sleeps until 08:30 IST on each NSE trading day (weekends and exchange holidays skipped automatically), refreshes every live source, retrains, and has predictions plus the aggregated market summary ready before the 09:15 open:

```bash
jasmin live-daemon                        # pre-market run every trading day
jasmin live-daemon --intraday-minutes 30  # plus intraday refresh while the market is open
jasmin market-status                      # current NSE session status (IST)
```

The market summary accumulates every symbol's prediction into one response — overall bias (confidence-weighted breadth), sector rollups, top bullish/bearish picks, and the macro backdrop (India VIX regime, NIFTY trend, USD/INR, FII flows).

Example prediction:

```json
{
  "symbol": "INFY",
  "direction": "UP",
  "probability_up": 0.63,
  "expected_move_pct": 0.49,
  "confidence": {"score": 60.6, "components": {"model_agreement": 0.45, "...": "..."}},
  "explanation": {
    "summary": "Supported by 20-day return vs own sector, MACD histogram (trend momentum); held back by Bollinger Band width (volatility)",
    "positive_factors": ["..."],
    "negative_factors": ["..."]
  }
}
```

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Service liveness + live model version |
| `GET /universe` | Configured symbol universe |
| `GET /predict/{symbol}` | Full prediction with explanation & confidence |
| `GET /analyze?q=...` | Analyze any stock by free-text name or ticker, with price targets |
| `GET /market-summary` | Aggregated view: bias, sectors, top picks, macro context |
| `GET /market-status` | NSE calendar status (IST) |
| `GET /predictions` | Recent prediction audit log |
| `GET /models` | Model registry versions & metrics |

## Model governance

Each training run registers a new version under `models/store/`. A candidate is auto-approved only if it clears the accuracy threshold **and** doesn't underperform the currently approved model. `jasmin models`, `jasmin approve <version>` and `jasmin rollback` manage which version serves predictions — nothing is ever deleted.

## Confidence score

Confidence (0–100) blends probability strength, agreement among ensemble members, feature completeness, data freshness, the model's validation accuracy, and the current volatility regime (India VIX). Low-confidence output means "the data doesn't support conviction," which is itself information.

## Live vs offline data

Live mode is the default and needs no API keys — all sources are free and fetched with the standard library:

| Domain | Live source | Notes |
|---|---|---|
| Prices (OHLCV) | Yahoo Finance chart API | 2-year daily history per symbol |
| Fundamentals | Yahoo quoteSummary | current snapshot (PE, ROE, margins, holdings, mcap) treated as constant over history |
| Macro | Yahoo (NIFTY, India VIX, USD/INR, Brent) + configured policy rates | update repo rate/CPI in `data/config/macro.json` when RBI/MoSPI publish |
| FII/DII flows | Official NSE API | latest session per fetch; a persistent log accumulates real history day by day |
| News | Economic Times + LiveMint RSS | headlines matched to the universe via company aliases; persistent log accumulates |

Every collector falls back to its deterministic synthetic source if a live fetch fails, so the pipeline never dies on a network hiccup — it logs the fallback and continues. `--offline` forces synthetic mode (used by the test suite).

Remaining upgrade points: true quarterly fundamentals history (filings source), per-symbol bulk/block deals, and a FinBERT-backed `SentimentScorer` (same one-method interface as the lexicon scorer).

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Disclaimer

Predictions are probabilistic decision support, not financial advice. Markets are affected by events no model can foresee; JASMIN reports confidence, not certainty.
