# JASMIN AI — Comprehensive System Design

*Version: Concept Design. Prepared for internal project documentation.*

> Recurring implementation note from the original document: every module is intended to remain independent so that individual collectors, feature generators or models can be upgraded without affecting the remainder of the system. This modular philosophy is central to the overall design and supports testing, maintenance and future expansion.

## 1. Executive Summary

JASMIN AI is an AI-driven market intelligence platform focused on Indian equities. The goal is not merely to forecast stock prices, but to understand the market as a combination of technical behaviour, company fundamentals, macroeconomics, institutional activity, news, and corporate events. Every prediction is accompanied by an explanation and confidence score, making the system useful for decision support rather than blind automation.

## 2. Problem Statement

Most retail prediction systems rely almost entirely on historical prices. Such systems often fail when sudden news, policy changes, earnings, or institutional buying shifts market direction. JASMIN addresses this by combining structured and unstructured data into one prediction pipeline.

## 3. Design Goals

- India-first architecture
- Modular collectors
- Explainable predictions
- Continuous learning
- Robust data cleaning
- Scalable feature engineering
- Support for conversational AI in future

## 4. Overall Architecture

Pipeline: Data Sources → Collectors → Validation → Cleaning → Feature Engineering → Master Dataset → Model Training → Prediction → Explanation → Dashboard/API.

## 5. Data Collection Modules

Historical prices (OHLCV), company fundamentals, technical indicators, macro indicators, sector performance, institutional activity, news sentiment, corporate actions, market status, unusual volume, insider activity, regulatory announcements and future options-chain data.

## 6. Historical Market Data

Fields: Open, High, Low, Close, Adjusted Close, Volume, Delivery %, Returns, Gaps, Rolling Returns. Data should be normalized, timestamped, checked for missing sessions and corporate actions.

## 7. Technical Indicators

Indicators planned: SMA, EMA, RSI, MACD, Signal Line, Histogram, ATR, ADX, Bollinger Bands, VWAP, OBV, Stochastic Oscillator, ROC, Momentum, CCI, Williams %R, Ichimoku components and moving-average crossovers. These capture trend, momentum, volatility and accumulation.

## 8. Fundamental Analysis

PE, PB, EPS, ROE, ROCE, debt/equity, current ratio, operating margin, net margin, promoter holding, institutional holding, dividend yield, market capitalization, quarterly growth, earnings surprises and valuation ratios.

## 9. News Intelligence

RSS feeds and trusted financial sources ingested continuously. Headlines are deduplicated, classified into event types, scored using a financial sentiment model (FinBERT), assigned confidence and event weights, then aggregated over different time windows.

## 10. Macroeconomic Signals

Repo rate, CPI inflation, GDP, industrial production, unemployment (where available), USD/INR, crude oil, bond yields, India VIX and major global indices. These variables provide market context and sector-specific sensitivity.

## 11. Institutional & Smart Money

Track FII/DII flows, bulk deals, block deals, promoter pledging, insider trades and large ownership changes. The philosophy: informed capital often precedes visible price movement.

## 12. Feature Engineering

Rather than feeding raw values into the model, engineered features are created: rolling averages, momentum, volatility, trend strength, lagged variables, relative strength, sector-relative returns, weighted sentiment, macro deltas, earnings flags and interaction features.

## 13. Machine Learning Strategy

Candidate models include Gradient Boosting, Random Forest, XGBoost and LightGBM. Ensemble methods are preferred because they perform well on heterogeneous tabular data. Future experimentation could include temporal neural networks for sequence modelling.

## 14. Prediction Pipeline

Each prediction gathers the latest engineered features, validates completeness, loads the latest approved model, infers probabilities, estimates expected percentage movement and generates explanatory factors.

## 15. Explainability

Instead of saying "BUY", JASMIN explains the reasoning. Example contributors: bullish MACD crossover, positive weighted earnings sentiment, improving FII activity, strong quarterly revenue growth and supportive sector momentum. Negative contributors are also reported.

## 16. Confidence Score

Confidence depends on prediction probability, agreement among models, feature completeness, data freshness, historical accuracy on similar conditions and current market volatility.

## 17. Continuous Learning

Collectors run on schedules. Clean data is appended to a master dataset. New labels become available after market movement. Retraining occurs periodically, with validation against previous models before deployment. Model versioning and rollback are integral.

## 18. Folder Structure Philosophy

Separate folders for raw data, cleaned data, engineered datasets, trained models, collectors, utilities, prediction scripts, scheduler/daemon and logs. Each module operates independently to simplify debugging.

## 19. Risks & Limitations

Predictions cannot guarantee market outcomes. Unexpected geopolitical events, data outages, manipulation, black swan events and unavailable information limit model certainty. Therefore the system reports confidence instead of certainty.

## 20. Future Vision

Future versions may include portfolio optimization, AI chat assistant, anomaly detection, options analytics, graph neural networks for company relationships, reinforcement learning experiments, multilingual interface, broker integration and real-time alerts.

## Representative Feature Categories

| Category | Examples | Purpose |
|---|---|---|
| Technical | RSI, EMA, MACD | Trend and momentum |
| Price | OHLCV, returns | Base market behaviour |
| Fundamental | EPS, PE, ROE | Business quality |
| Sentiment | FinBERT scores | Market psychology |
| Macro | CPI, Repo, USDINR | Economic backdrop |
| Institutional | FII/DII | Smart money |
| Sector | Sector index | Relative strength |
| Events | Earnings, mergers | Catalysts |
| Temporal | Weekday, expiry | Seasonality |

## End-to-End Flow Summary

1. Collect fresh data.
2. Validate integrity.
3. Clean and normalize.
4. Generate engineered features.
5. Build inference dataset.
6. Load latest approved model.
7. Predict probability and expected movement.
8. Rank important features.
9. Produce explanation.
10. Store results and update dashboard.
