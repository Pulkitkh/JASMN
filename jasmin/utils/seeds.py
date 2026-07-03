"""Stable seeding for synthetic data sources.

Python's built-in hash() is randomized per process (PYTHONHASHSEED), so it
cannot seed reproducible synthetic data. CRC32 is stable across processes
and platforms, keeping the offline sources deterministic run to run.
"""

from __future__ import annotations

import zlib


def stable_seed(*parts: object) -> int:
    key = "|".join(map(str, parts))
    return zlib.crc32(key.encode("utf-8"))
