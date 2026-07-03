"""News intelligence collector (design doc §9).

Pipeline per headline: dedupe -> event-type classification -> sentiment
scoring -> confidence & event weight -> daily aggregation happens later in
feature engineering.

Sentiment uses a financial lexicon scorer by default (fast, dependency-free).
The `SentimentScorer` interface is the swap point for FinBERT: implement
`score(text) -> float in [-1, 1]` and pass it to the collector.

Live mode reads RSS feeds with feedparser when available; otherwise a seeded
synthetic headline stream keeps the pipeline running offline.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from jasmin.collectors.base import BaseCollector
from jasmin.utils.seeds import stable_seed

# Event taxonomy with weights: how strongly each event type tends to move a
# stock (design doc: "assigned confidence and event weights").
EVENT_WEIGHTS = {
    "earnings": 1.0,
    "merger_acquisition": 0.9,
    "regulatory": 0.8,
    "management_change": 0.6,
    "product_launch": 0.5,
    "analyst_rating": 0.5,
    "macro": 0.4,
    "general": 0.3,
}

_EVENT_PATTERNS = {
    "earnings": r"\b(results?|earnings|profit|revenue|q[1-4]|quarterly|net income)\b",
    "merger_acquisition": r"\b(merger|acquisition|acquire[sd]?|takeover|stake)\b",
    "regulatory": r"\b(sebi|rbi|regulat|penalty|probe|investigation|compliance)\b",
    "management_change": r"\b(ceo|cfo|resign|appoint|director|board)\b",
    "product_launch": r"\b(launch|unveil|new product|expansion|plant|contract|order win)\b",
    "analyst_rating": r"\b(upgrade[sd]?|downgrade[sd]?|target price|rating|overweight|underweight)\b",
    "macro": r"\b(inflation|gdp|repo|fed|rate cut|rate hike|budget|fiscal)\b",
}

_POSITIVE = {
    "beats", "beat", "surge", "surges", "record", "strong", "growth", "wins",
    "win", "upgrade", "upgrades", "profit", "gain", "gains", "expansion",
    "jump", "jumps", "rally", "bullish", "outperform", "robust", "raises",
}
_NEGATIVE = {
    "misses", "miss", "falls", "fall", "drop", "drops", "weak", "loss",
    "losses", "downgrade", "downgrades", "penalty", "probe", "fraud",
    "decline", "declines", "plunge", "plunges", "bearish", "cuts", "cut",
    "slump", "layoffs", "warns",
}


class SentimentScorer:
    """Lexicon-based financial sentiment in [-1, 1]. FinBERT swap point."""

    def score(self, text: str) -> float:
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return 0.0
        pos = sum(w in _POSITIVE for w in words)
        neg = sum(w in _NEGATIVE for w in words)
        if pos == neg:
            return 0.0
        return (pos - neg) / (pos + neg)


def classify_event(headline: str) -> str:
    text = headline.lower()
    for event, pattern in _EVENT_PATTERNS.items():
        if re.search(pattern, text):
            return event
    return "general"


def dedupe_key(headline: str) -> str:
    """Normalization hash so near-identical headlines collapse together."""
    normalized = re.sub(r"[^a-z0-9 ]", "", headline.lower()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


_HEADLINE_TEMPLATES = [
    ("{sym} Q{q} results: profit beats estimates on strong revenue growth", 0.028),
    ("{sym} misses quarterly profit estimates as margins decline", -0.030),
    ("{sym} wins large multi-year contract, shares jump", 0.022),
    ("Brokerage upgrades {sym}, raises target price", 0.018),
    ("Brokerage downgrades {sym} on weak demand outlook", -0.018),
    ("{sym} announces expansion of new plant capacity", 0.012),
    ("SEBI probe into {sym} disclosure practices", -0.024),
    ("{sym} board appoints new CEO", 0.004),
    ("{sym} in talks for acquisition of overseas rival", 0.010),
    ("RBI policy: rate decision in focus for banks including {sym}", 0.0),
]


class NewsCollector(BaseCollector):
    name = "news"

    def __init__(self, offline: bool = False, scorer: SentimentScorer | None = None,
                 feeds: list[str] | None = None):
        super().__init__(offline=offline)
        self.scorer = scorer or SentimentScorer()
        self.feeds = feeds or []

    def fetch(self, symbols: list[str], days: int) -> pd.DataFrame:
        items = None
        if not self.offline and self.feeds:
            items = self._fetch_rss(symbols)
        if items is None or items.empty:
            items = self._synthetic_headlines(symbols, days)
        return self._enrich(items)

    def _fetch_rss(self, symbols: list[str]) -> pd.DataFrame | None:
        try:
            import feedparser  # optional dependency
        except ImportError:
            self.log.info("feedparser not installed; using synthetic headlines")
            return None
        rows = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries:
                    title = entry.get("title", "")
                    matched = [s for s in symbols if s.lower() in title.lower()]
                    if not matched:
                        continue
                    rows.append(
                        {
                            "date": pd.to_datetime(
                                entry.get("published", pd.Timestamp.today())
                            ).normalize(),
                            "symbol": matched[0],
                            "headline": title,
                        }
                    )
            except Exception as exc:
                self.log.warning("RSS fetch failed for %s: %s", url, exc)
        return pd.DataFrame(rows) if rows else None

    def _synthetic_headlines(self, symbols: list[str], days: int) -> pd.DataFrame:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        rows = []
        for sym in symbols:
            rng = np.random.default_rng(stable_seed("news", sym))
            for date in dates:
                # ~1 headline every 3 sessions per symbol.
                if rng.random() > 0.33:
                    continue
                template, _ = _HEADLINE_TEMPLATES[rng.integers(len(_HEADLINE_TEMPLATES))]
                rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "headline": template.format(sym=sym, q=rng.integers(1, 5)),
                    }
                )
        return pd.DataFrame(rows)

    def _enrich(self, items: pd.DataFrame) -> pd.DataFrame:
        items = items.copy()
        items["dedupe_key"] = items["headline"].map(dedupe_key)
        items = items.drop_duplicates(subset=["date", "symbol", "dedupe_key"])
        items["event_type"] = items["headline"].map(classify_event)
        items["sentiment"] = items["headline"].map(self.scorer.score).round(3)
        items["event_weight"] = items["event_type"].map(EVENT_WEIGHTS)
        # Confidence: strong lexicon signal -> higher confidence.
        items["confidence"] = (0.5 + 0.5 * items["sentiment"].abs()).round(3)
        items["weighted_sentiment"] = (
            items["sentiment"] * items["event_weight"] * items["confidence"]
        ).round(4)
        return items.drop(columns=["dedupe_key"]).reset_index(drop=True)
