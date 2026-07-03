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
| `jasmin/collectors/` | Prices (OHLCV), fundamentals, macro, institutional (FII/DII, deals, pledges), news intelligence. Each falls back to a deterministic synthetic source when live dependencies/network are unavailable. |
| `jasmin/validation/` | Integrity checks: schema, OHLC consistency, duplicates, missing sessions. Errors block the pipeline; warnings pass through. |
| `jasmin/cleaning/` | Dedupe, normalization, gap filling, outlier winsorization, derived returns/gaps. |
| `jasmin/features/` | 20+ technical indicators (RSI, MACD, ADX, Bollinger, Ichimoku, VWAP, OBV, …) plus engineered features: relative strength, weighted sentiment windows, macro deltas, FII flow trends, expiry-week seasonality, interaction terms. |
| `jasmin/dataset/` | Builds the labeled master dataset (direction + expected % move over the horizon). |
| `jasmin/models/` | Gradient-boosting + random-forest ensemble (direction) and a regressor (expected move), with a versioned registry, auto-approval gates and rollback. |
| `jasmin/prediction/` | Inference pipeline: completeness validation → ensemble probability → expected move → ranked factor explanation → confidence score. |
| `jasmin/api/` | FastAPI service exposing predictions, the audit log and the model registry. |
| `jasmin/scheduler/` | Continuous-learning daemon: collect → rebuild → retrain → re-predict on an interval. |

## Quick start

```bash
pip install -e .            # add ".[live]" for yfinance/feedparser live sources
jasmin cycle --offline      # full pipeline on synthetic data, no network needed
```

Or stage by stage:

```bash
jasmin collect --days 400   # 1. run all collectors (add --offline to force synthetic)
jasmin build-dataset        # 2-5. validate, clean, engineer features, label
jasmin train                # 6. train & register a model version
jasmin predict RELIANCE TCS # 7-10. predict with explanation + confidence
jasmin serve                # start the API on :8000
```

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
| `GET /predictions` | Recent prediction audit log |
| `GET /models` | Model registry versions & metrics |

## Model governance

Each training run registers a new version under `models/store/`. A candidate is auto-approved only if it clears the accuracy threshold **and** doesn't underperform the currently approved model. `jasmin models`, `jasmin approve <version>` and `jasmin rollback` manage which version serves predictions — nothing is ever deleted.

## Confidence score

Confidence (0–100) blends probability strength, agreement among ensemble members, feature completeness, data freshness, the model's validation accuracy, and the current volatility regime (India VIX). Low-confidence output means "the data doesn't support conviction," which is itself information.

## Offline vs live data

The repository runs fully offline out of the box: every collector has a deterministic synthetic source, so the pipeline, tests and demos work with zero network access or API keys. To go live:

- `pip install yfinance` → real NSE OHLCV history
- `pip install feedparser` and pass RSS feed URLs to `NewsCollector` → real headlines
- Swap the lexicon `SentimentScorer` for a FinBERT-backed implementation (same one-method interface)
- Point `FundamentalsCollector`/`InstitutionalCollector` at real filing/exchange sources by overriding their fetch logic

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Disclaimer

Predictions are probabilistic decision support, not financial advice. Markets are affected by events no model can foresee; JASMIN reports confidence, not certainty.
