"""Shared logging setup: console + rotating file under logs/."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from jasmin.config import LOGS_DIR

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(LOGS_DIR / "jasmin.log", maxBytes=2_000_000, backupCount=3)
        fileh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fileh)
    except OSError:
        pass  # read-only filesystem: console logging still works

    logger.propagate = False
    return logger
