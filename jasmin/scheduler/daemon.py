"""Continuous-learning daemons (design doc §17).

`run_daemon`   - fixed-interval loop: collect -> rebuild -> retrain -> predict.
`run_live_daemon` - market-aware loop: sleeps until the pre-market window
(08:30 IST) of each NSE trading day, refreshes everything so predictions are
ready when the market opens at 09:15, then optionally re-predicts on an
intraday interval while the session is open (prices update intraday; the
model stays fixed during the day).

The registry decides whether each retrained model is approved, so a bad
retrain never silently replaces a good model. For production, cron/systemd
pointing at `jasmin premarket` is equally valid; the daemons exist for
simple long-running deployments.
"""

from __future__ import annotations

import time

from jasmin.config import PipelineConfig, ensure_dirs
from jasmin.pipeline import run_cycle, run_premarket
from jasmin.utils.logging import get_logger
from jasmin.utils import market_calendar as cal

log = get_logger("daemon")


def run_daemon(interval_hours: float = 24, offline: bool = False,
               config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    ensure_dirs()
    while True:
        try:
            result = run_cycle(config=config, offline=offline)
            log.info("cycle complete: model=%s approved=%s",
                     result["training"]["version"], result["training"]["approved"])
        except Exception:
            log.exception("cycle failed; will retry next interval")
        time.sleep(interval_hours * 3600)


def run_live_daemon(config: PipelineConfig | None = None,
                    intraday_minutes: int = 0) -> None:
    """Market-aware loop keyed to the NSE calendar (IST)."""
    config = config or PipelineConfig()
    ensure_dirs()
    while True:
        next_run = cal.next_premarket_run()
        wait = (next_run - cal.now_ist()).total_seconds()
        log.info("sleeping until pre-market run at %s IST (%.1f h)",
                 next_run.isoformat(timespec="minutes"), wait / 3600)
        time.sleep(max(wait, 0))
        try:
            result = run_premarket(config)
            summary = result["summary"]
            log.info("pre-market ready: bias=%s up=%d down=%d avg_conf=%.1f",
                     summary.get("market_bias"), summary.get("n_up", 0),
                     summary.get("n_down", 0), summary.get("avg_confidence", 0))
        except Exception:
            log.exception("pre-market run failed")

        # Intraday refresh: re-collect prices/news and re-predict with the
        # already-approved model while the market is open.
        while intraday_minutes > 0 and cal.is_market_open():
            time.sleep(intraday_minutes * 60)
            if not cal.is_market_open():
                break
            try:
                run_premarket(config, retrain=False)
                log.info("intraday refresh complete")
            except Exception:
                log.exception("intraday refresh failed")
