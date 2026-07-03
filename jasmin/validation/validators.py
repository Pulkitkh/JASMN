"""Data integrity validation (pipeline stage 2).

Checks raw collector output before it is allowed into cleaning: required
columns, OHLC consistency, non-negative volume, duplicate rows and missing
trading sessions. Issues are collected into a report; `errors` block the
pipeline while `warnings` are logged and passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from jasmin.utils.logging import get_logger

log = get_logger("validation")

PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


@dataclass
class ValidationReport:
    domain: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(f"validation failed for {self.domain}: {self.errors}")


def validate_domain(df: pd.DataFrame, domain: str, required: list[str]) -> ValidationReport:
    """Generic checks shared by every data domain."""
    report = ValidationReport(domain=domain)
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report
    if df.empty:
        report.errors.append("no rows collected")
        return report

    subset = [c for c in ("date", "symbol") if c in df.columns]
    dupes = int(df.duplicated(subset=subset).sum())
    if dupes:
        report.warnings.append(f"{dupes} duplicate rows (will be dropped in cleaning)")

    null_ratio = df[required].isna().mean()
    for col, ratio in null_ratio.items():
        if ratio > 0.5:
            report.errors.append(f"column {col} is {ratio:.0%} null")
        elif ratio > 0.05:
            report.warnings.append(f"column {col} is {ratio:.0%} null")

    for w in report.warnings:
        log.warning("[%s] %s", domain, w)
    return report


def validate_prices(df: pd.DataFrame) -> ValidationReport:
    """Price-specific checks on top of the generic ones."""
    report = validate_domain(df, "prices", PRICE_COLUMNS)
    if not report.ok:
        return report

    bad_ohlc = df[(df["high"] < df["low"]) | (df["close"] <= 0) | (df["open"] <= 0)]
    if not bad_ohlc.empty:
        report.errors.append(f"{len(bad_ohlc)} rows with inconsistent OHLC values")

    if (df["volume"] < 0).any():
        report.errors.append("negative volume values found")

    # Missing-session check: compare each symbol's dates to business days.
    for sym, grp in df.groupby("symbol"):
        dates = pd.to_datetime(grp["date"]).dt.normalize()
        expected = pd.bdate_range(dates.min(), dates.max())
        # NSE holidays make small gaps normal; flag only substantial ones.
        missing_ratio = 1 - len(dates.unique()) / max(len(expected), 1)
        if missing_ratio > 0.1:
            report.warnings.append(
                f"{sym}: {missing_ratio:.0%} of expected sessions missing"
            )

    for w in report.warnings:
        log.warning("[prices] %s", w)
    return report
