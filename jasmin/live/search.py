"""Symbol resolution: free-text stock name/ticker -> NSE symbol.

Exact NSE tickers pass through without a network call. Anything else goes
to Yahoo's search API, preferring NSE (NSI) listings, then BSE. Raises
LookupError with suggestions when nothing matches an Indian exchange.
"""

from __future__ import annotations

import urllib.parse

from jasmin.live.http import HttpClient
from jasmin.utils.logging import get_logger

log = get_logger("live.search")

_SEARCH = (
    "https://query1.finance.yahoo.com/v1/finance/search"
    "?q={query}&quotesCount=8&newsCount=0"
)
_INDIAN_EXCHANGES = ("NSI", "BSE")


def resolve_symbol(query: str, universe: list[str] | None = None,
                   client: HttpClient | None = None) -> dict:
    """Return {"symbol", "name", "exchange"} for a user query."""
    cleaned = query.strip().upper().removesuffix(".NS").removesuffix(".BO")
    if universe and cleaned in universe:
        return {"symbol": cleaned, "name": cleaned, "exchange": "NSI"}
    # Ticker-looking input (no spaces, short) is trusted as-is; the price
    # fetch will fail loudly if it doesn't exist on NSE.
    if cleaned.isalnum() and " " not in query.strip() and len(cleaned) <= 12 \
            and not query.strip().islower():
        return {"symbol": cleaned, "name": cleaned, "exchange": "NSI"}

    http = client or HttpClient()
    data = http.get_json(_SEARCH.format(query=urllib.parse.quote(query.strip())))
    equities = [q for q in data.get("quotes", []) if q.get("quoteType") == "EQUITY"]
    for exchange in _INDIAN_EXCHANGES:
        for hit in equities:
            if hit.get("exchange") == exchange:
                symbol = hit["symbol"].removesuffix(".NS").removesuffix(".BO")
                result = {
                    "symbol": symbol,
                    "name": hit.get("shortname") or hit.get("longname") or symbol,
                    "exchange": exchange,
                }
                log.info("resolved %r -> %s (%s)", query, symbol, result["name"])
                return result
    suggestions = [f"{q.get('symbol')} ({q.get('shortname')}, {q.get('exchange')})"
                   for q in equities[:5]]
    raise LookupError(
        f"No NSE/BSE listing found for {query!r}."
        + (f" Closest matches: {', '.join(suggestions)}" if suggestions else "")
    )
