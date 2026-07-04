"""NSE market calendar: IST sessions, open/close times, pre-market windows.

Trading hours: 09:15-15:30 IST, Monday-Friday, excluding exchange holidays.
The holiday list ships with known dates and can be extended in
data/config/nse_holidays.txt (one YYYY-MM-DD per line).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from jasmin.config import DATA_DIR

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
# When the pre-market pipeline should run so predictions are ready at open.
PREMARKET_RUN = time(8, 30)

# NSE trading holidays (extend via data/config/nse_holidays.txt).
_KNOWN_HOLIDAYS = {
    # 2026
    "2026-01-26", "2026-03-04", "2026-03-21", "2026-04-01", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26", "2026-08-15",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-09", "2026-11-10",
    "2026-11-24", "2026-12-25",
}


def _holidays() -> set[str]:
    extra = DATA_DIR / "config" / "nse_holidays.txt"
    days = set(_KNOWN_HOLIDAYS)
    if extra.exists():
        days |= {line.strip() for line in extra.read_text().splitlines() if line.strip()}
    return days


def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    return dt.weekday() < 5 and dt.strftime("%Y-%m-%d") not in _holidays()


def is_market_open(dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    return is_trading_day(dt) and MARKET_OPEN <= dt.time() <= MARKET_CLOSE


def next_session_open(dt: datetime | None = None) -> datetime:
    """Next market open at or after `dt`."""
    dt = dt or now_ist()
    candidate = dt
    if not is_trading_day(candidate) or candidate.time() > MARKET_CLOSE:
        candidate = (candidate + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    open_dt = candidate.replace(
        hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0
    )
    return open_dt if open_dt >= dt else dt


def next_premarket_run(dt: datetime | None = None) -> datetime:
    """Next time the pre-market pipeline should fire (PREMARKET_RUN on the
    next trading day, or today if we're before it)."""
    dt = dt or now_ist()
    candidate = dt.replace(
        hour=PREMARKET_RUN.hour, minute=PREMARKET_RUN.minute, second=0, microsecond=0
    )
    while candidate <= dt or not is_trading_day(candidate):
        candidate = (candidate + timedelta(days=1)).replace(
            hour=PREMARKET_RUN.hour, minute=PREMARKET_RUN.minute
        )
    return candidate


def market_status() -> dict:
    dt = now_ist()
    return {
        "now_ist": dt.isoformat(timespec="seconds"),
        "trading_day": is_trading_day(dt),
        "market_open": is_market_open(dt),
        "next_open_ist": next_session_open(dt).isoformat(timespec="seconds"),
    }
