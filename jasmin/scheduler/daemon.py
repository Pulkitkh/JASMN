"""Continuous-learning daemon (design doc §17).

Runs the full cycle on an interval: collect -> rebuild master dataset ->
retrain -> re-predict the universe. The registry decides whether each new
model is approved, so a bad retrain never silently replaces a good model.

For production, point cron/systemd at `jasmin cycle` instead of keeping
this process alive; the daemon exists for simple long-running deployments.
"""

from __future__ import annotations

import time

from jasmin.config import PipelineConfig, ensure_dirs
from jasmin.pipeline import run_cycle
from jasmin.utils.logging import get_logger

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
