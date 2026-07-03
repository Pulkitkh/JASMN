"""RSS news ingestion for Indian financial media.

Parses RSS 2.0 with the standard library (no feedparser dependency) and
matches headlines to universe symbols through the company-alias map in
config. Because feeds only expose recent items, every fetch is appended to
a persistent log so news history accumulates across days.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

from jasmin.config import COMPANY_ALIASES, RAW_DIR
from jasmin.live.http import HttpClient
from jasmin.utils.logging import get_logger

log = get_logger("live.rss")

DEFAULT_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/companies",
]

_LOG_PATH = RAW_DIR / "news" / "news_log.csv"


def parse_rss(content: bytes) -> list[dict]:
    """Extract (title, published) pairs from an RSS 2.0 document."""
    items = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        log.warning("unparseable feed: %s", exc)
        return items
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        pub = item.findtext("pubDate")
        try:
            date = pd.Timestamp(parsedate_to_datetime(pub)).tz_localize(None).normalize()
        except (TypeError, ValueError):
            date = pd.Timestamp.today().normalize()
        items.append({"date": date, "headline": title})
    return items


def match_symbol(headline: str, symbols: list[str]) -> str | None:
    """Match a headline to the first universe symbol whose alias appears."""
    text = f" {headline.lower()} "
    for sym in symbols:
        for alias in COMPANY_ALIASES.get(sym, [sym.lower()]):
            if re.search(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", text):
                return sym
    return None


def fetch_news(symbols: list[str], feeds: list[str] | None = None,
               client: HttpClient | None = None) -> pd.DataFrame:
    """Fetch feeds, match to symbols, and merge into the accumulated log."""
    http = client or HttpClient()
    rows = []
    for url in feeds or DEFAULT_FEEDS:
        try:
            for item in parse_rss(http.get(url)):
                sym = match_symbol(item["headline"], symbols)
                if sym:
                    rows.append({**item, "symbol": sym})
        except Exception as exc:
            log.warning("feed failed %s: %s", url, exc)
    latest = pd.DataFrame(rows, columns=["date", "symbol", "headline"])
    log.info("matched %d live headlines", len(latest))
    return _append_log(latest)


def _append_log(latest: pd.DataFrame) -> pd.DataFrame:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LOG_PATH.exists():
        history = pd.read_csv(_LOG_PATH, parse_dates=["date"])
        merged = pd.concat([history, latest], ignore_index=True)
    else:
        merged = latest
    merged = (
        merged.drop_duplicates(subset=["date", "symbol", "headline"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged.to_csv(_LOG_PATH, index=False)
    return merged
