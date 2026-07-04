"""NSE official API client: FII/DII daily provisional flows.

The fiidiiTradeReact endpoint returns only the latest session, so every
fetch is appended to a persistent log under data/raw/institutional/ —
real history accumulates day by day and backfills the synthetic series.
"""

from __future__ import annotations

import pandas as pd

from jasmin.config import RAW_DIR
from jasmin.live.http import HttpClient
from jasmin.utils.logging import get_logger

log = get_logger("live.nse")

_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
_LOG_PATH = RAW_DIR / "institutional" / "fii_dii_log.csv"

_NSE_HEADERS = {
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/reports/fii-dii",
}


def fetch_fii_dii(client: HttpClient | None = None) -> pd.DataFrame:
    """Latest FII/DII net flows in crores; one row per session in the log."""
    http = client or HttpClient()
    rows = http.get_json(_FII_DII_URL, headers=_NSE_HEADERS)
    by_date: dict[str, dict] = {}
    for row in rows:
        date = pd.to_datetime(row["date"], format="%d-%b-%Y")
        entry = by_date.setdefault(str(date.date()), {"date": date})
        key = "fii_net_cr" if row["category"].upper().startswith("FII") else "dii_net_cr"
        entry[key] = float(row["netValue"])
    latest = pd.DataFrame(list(by_date.values()))
    log.info("NSE FII/DII: %s", latest.to_dict(orient="records"))
    return _append_log(latest)


def _append_log(latest: pd.DataFrame) -> pd.DataFrame:
    """Merge the latest snapshot into the accumulated history log."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LOG_PATH.exists():
        history = pd.read_csv(_LOG_PATH, parse_dates=["date"])
        merged = pd.concat([history, latest], ignore_index=True)
    else:
        merged = latest
    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged.to_csv(_LOG_PATH, index=False)
    return merged
