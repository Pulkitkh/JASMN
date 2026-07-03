"""Minimal HTTP client for live data sources.

Uses only the standard library (urllib + cookiejar) so live mode adds zero
dependencies. Browser-like headers are required by both Yahoo Finance and
NSE; requests without them get 401/403/429.
"""

from __future__ import annotations

import http.cookiejar
import json
import time
import urllib.error
import urllib.request

from jasmin.utils.logging import get_logger

log = get_logger("live.http")

# Yahoo rate-limits per (IP, User-Agent); rotating to the next UA on a 429
# restores service immediately instead of waiting out the throttle window.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
]


class HttpClient:
    """Cookie-aware GET client with retries, rate-limit backoff and spacing."""

    def __init__(self, timeout: int = 20, retries: int = 4,
                 min_interval: float = 1.0):
        self.timeout = timeout
        self.retries = retries
        # Minimum seconds between any two requests from this client, so a
        # burst over many symbols doesn't trip source rate limits.
        self.min_interval = min_interval
        self._last_request = 0.0
        self.jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        self._ua_index = 0

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str, headers: dict | None = None,
            allow_error: bool = False) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            merged = {"User-Agent": USER_AGENTS[self._ua_index], "Accept": "*/*"}
            merged.update(headers or {})
            self._pace()
            try:
                req = urllib.request.Request(url, headers=merged)
                with self._opener.open(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if allow_error:
                    return b""
                if exc.code == 429:
                    # Rate limited for this UA: rotate identity and retry.
                    last_exc = exc
                    self._ua_index = (self._ua_index + 1) % len(USER_AGENTS)
                    log.info("429 on %s; rotating user-agent", url.split("?")[0])
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if exc.code < 500:  # other 4xx won't heal on retry
                    raise
                last_exc = exc
            except Exception as exc:  # URLError, timeout, reset
                last_exc = exc
            time.sleep(1.5 * (attempt + 1))
        raise ConnectionError(f"GET {url} failed after {self.retries} tries: {last_exc}")

    def get_json(self, url: str, headers: dict | None = None):
        return json.loads(self.get(url, headers=headers).decode("utf-8"))
