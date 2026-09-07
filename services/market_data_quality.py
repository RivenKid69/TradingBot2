# -*- coding: utf-8 -*-
"""
services/market_data_quality.py
===============================

Market-data quality assurance + vendor failover (P1 #11).

The existing ``data_validation`` is a rigid fail-fast schema checker and
``service_data_quality`` covers PIT-leakage. Neither does the quant-desk data-QA a
live feed needs: **robust spike detection, staleness watchdog, frozen-feed
detection, session-aware gap detection, and cross-vendor reconciliation**, plus a
**MarketDataRouter** that fails over from a primary vendor to a secondary on outage
or bad data (with a per-source circuit breaker).

Methods
-------
* **Spike** — robust modified z-score on log-returns: z = 0.6745·|r−median|/MAD
  (Iglewicz & Hoaglin, 1993; the Hampel identifier). Robust to the very outliers it
  detects, unlike a mean/σ z-score.
* **Staleness** — last-bar age vs a heartbeat TTL (a feed that stopped updating).
* **Frozen** — N identical consecutive prices (a stuck/again-replayed feed).
* **Gaps** — timestamp irregularities vs the modal bar interval (session-aware: a
  gap up to ``session_gap_factor`` is tolerated for overnight/weekend breaks).
* **Cross-vendor** — per-symbol close divergence in bps between two sources.

Pure pandas/numpy. Layer ``services``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAD_SCALE = 0.6745  # makes MAD a consistent estimator of σ for normal data


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------
@dataclass
class QualityIssue:
    type: str  # spike | stale | frozen | gap | ohlc | nonpositive | nan
    severity: str  # LOW | MEDIUM | HIGH
    detail: str
    count: int = 1
    locations: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "detail": self.detail,
            "count": self.count,
            "locations": self.locations[:20],
        }


@dataclass
class DataQualityReport:
    symbol: str
    n_bars: int
    clean: bool
    issues: List[QualityIssue]
    last_ts_ms: Optional[int] = None
    staleness_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "clean": self.clean,
            "last_ts_ms": self.last_ts_ms,
            "staleness_seconds": self.staleness_seconds,
            "n_issues": len(self.issues),
            "max_severity": (
                max(
                    (i.severity for i in self.issues),
                    key=lambda s: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[s],
                )
                if self.issues
                else None
            ),
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def modified_zscores(x: np.ndarray) -> np.ndarray:
    """Hampel/Iglewicz–Hoaglin robust modified z-scores (MAD-based)."""
    x = np.asarray(x, dtype="float64")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 1e-15:
        # fall back to mean/std if MAD degenerate (all-equal returns)
        sd = np.std(x)
        return np.zeros_like(x) if sd <= 1e-15 else (x - np.mean(x)) / sd
    return _MAD_SCALE * (x - med) / mad


class DataQualityMonitor:
    """Robust QC over a single symbol's OHLC(V) bars."""

    def __init__(
        self,
        *,
        spike_threshold: float = 8.0,  # |modified z| on log-returns
        staleness_seconds: Optional[float] = 300.0,
        frozen_run: int = 6,  # N identical consecutive closes ⇒ frozen
        session_gap_factor: float = 6.0,  # tolerate gaps up to N× the modal interval
        price_col: str = "close",
        ts_col: str = "timestamp",  # epoch ms (or pandas datetime index)
    ) -> None:
        self.spike_threshold = float(spike_threshold)
        self.staleness_seconds = staleness_seconds
        self.frozen_run = int(frozen_run)
        self.session_gap_factor = float(session_gap_factor)
        self.price_col = price_col
        self.ts_col = ts_col

    def check(
        self, df: pd.DataFrame, *, symbol: str = "?", now_ms: Optional[int] = None
    ) -> DataQualityReport:
        issues: List[QualityIssue] = []
        n = int(len(df))
        if n == 0:
            return DataQualityReport(
                symbol, 0, False, [QualityIssue("nan", "HIGH", "empty series")]
            )

        px = pd.to_numeric(df[self.price_col], errors="coerce").to_numpy(dtype="float64")

        # NaN / non-positive prices
        n_nan = int(np.sum(~np.isfinite(px)))
        if n_nan:
            issues.append(QualityIssue("nan", "HIGH", f"{n_nan} NaN/inf prices", n_nan))
        n_nonpos = int(np.sum(px[np.isfinite(px)] <= 0))
        if n_nonpos:
            issues.append(
                QualityIssue("nonpositive", "HIGH", f"{n_nonpos} non-positive prices", n_nonpos)
            )

        valid = px[np.isfinite(px) & (px > 0)]
        # Spikes (robust modified z on log-returns)
        if valid.size >= 5:
            logret = np.diff(np.log(valid))
            mz = modified_zscores(logret)
            spike_idx = np.where(np.abs(mz) > self.spike_threshold)[0]
            if spike_idx.size:
                issues.append(
                    QualityIssue(
                        "spike",
                        "HIGH" if spike_idx.size > 1 else "MEDIUM",
                        f"{spike_idx.size} return spikes (|mod-z|>{self.spike_threshold})",
                        int(spike_idx.size),
                        [int(i + 1) for i in spike_idx],
                    )
                )

        # Frozen feed (consecutive identical closes)
        if valid.size >= self.frozen_run:
            run = 1
            max_run = 1
            for i in range(1, valid.size):
                run = run + 1 if valid[i] == valid[i - 1] else 1
                max_run = max(max_run, run)
            if max_run >= self.frozen_run:
                issues.append(
                    QualityIssue(
                        "frozen",
                        "MEDIUM",
                        f"{max_run} identical consecutive prices (stuck feed)",
                        max_run,
                    )
                )

        # Gap detection (session-aware) on timestamps
        last_ts_ms = None
        staleness = None
        ts = self._timestamps_ms(df)
        if ts is not None and ts.size >= 3:
            dt = np.diff(ts)
            dt = dt[dt > 0]
            if dt.size:
                modal = float(np.median(dt))
                big = np.where(dt > self.session_gap_factor * modal)[0]
                if big.size:
                    issues.append(
                        QualityIssue(
                            "gap",
                            "LOW",
                            f"{big.size} timestamp gaps > {self.session_gap_factor}× modal interval",
                            int(big.size),
                            [int(i) for i in big],
                        )
                    )
            last_ts_ms = int(ts[-1])
            ref = now_ms if now_ms is not None else int(time.time() * 1000)
            staleness = max(0.0, (ref - last_ts_ms) / 1000.0)
            if self.staleness_seconds is not None and staleness > self.staleness_seconds:
                issues.append(
                    QualityIssue(
                        "stale",
                        "HIGH",
                        f"last bar {staleness:.0f}s old (> {self.staleness_seconds:.0f}s TTL)",
                    )
                )

        # OHLC invariants (if columns present)
        cols = set(df.columns)
        if {"high", "low"} <= cols:
            hi = pd.to_numeric(df["high"], errors="coerce").to_numpy()
            lo = pd.to_numeric(df["low"], errors="coerce").to_numpy()
            bad = int(np.sum(np.isfinite(hi) & np.isfinite(lo) & (hi < lo)))
            if bad:
                issues.append(QualityIssue("ohlc", "HIGH", f"{bad} bars with high < low", bad))
            for c in ("open", "close"):
                if c in cols:
                    v = pd.to_numeric(df[c], errors="coerce").to_numpy()
                    viol = int(
                        np.sum(
                            np.isfinite(v)
                            & np.isfinite(hi)
                            & np.isfinite(lo)
                            & ((v > hi) | (v < lo))
                        )
                    )
                    if viol:
                        issues.append(
                            QualityIssue("ohlc", "MEDIUM", f"{viol} {c} outside [low, high]", viol)
                        )

        clean = not any(i.severity == "HIGH" for i in issues)
        return DataQualityReport(symbol, n, clean, issues, last_ts_ms, staleness)

    def _timestamps_ms(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.ts_col in df.columns:
            s = df[self.ts_col]
            if np.issubdtype(s.dtype, np.number):
                return s.to_numpy(dtype="float64")
            try:
                return (pd.to_datetime(s).astype("int64") // 1_000_000).to_numpy(dtype="float64")
            except Exception:
                return None
        if isinstance(df.index, pd.DatetimeIndex):
            return (df.index.astype("int64") // 1_000_000).to_numpy(dtype="float64")
        return None


def cross_source_reconcile(
    a: Dict[str, float],
    b: Dict[str, float],
    *,
    tolerance_bps: float = 50.0,
) -> Dict[str, Any]:
    """Compare two vendors' prices per symbol; flag divergences beyond ``tolerance_bps``."""
    syms = sorted(set(a) & set(b))
    divergences: List[Dict[str, Any]] = []
    for s in syms:
        pa, pb = float(a[s]), float(b[s])
        if pa <= 0 or pb <= 0:
            continue
        mid = 0.5 * (pa + pb)
        bps = abs(pa - pb) / mid * 1e4
        if bps > tolerance_bps:
            divergences.append({"symbol": s, "a": pa, "b": pb, "divergence_bps": round(bps, 2)})
    return {
        "reconciled": len(divergences) == 0,
        "n_compared": len(syms),
        "tolerance_bps": tolerance_bps,
        "divergences": divergences,
    }


# ---------------------------------------------------------------------------
# Vendor failover router
# ---------------------------------------------------------------------------
@dataclass
class _SourceState:
    name: str
    fetch: Callable[..., pd.DataFrame]
    consecutive_failures: int = 0
    tripped: bool = False
    last_ok_ms: Optional[int] = None
    ok_count: int = 0
    fail_count: int = 0


class MarketDataRouter:
    """Primary→secondary market-data router with QC-gated failover + circuit breaker.

    ``sources`` is an ordered priority list of (name, fetch_fn). ``fetch_fn(symbol,
    **kw) -> DataFrame`` returns OHLC bars. On each request the router tries sources
    in order, **skipping tripped ones**, and accepts the first whose data passes the
    quality monitor (no HIGH-severity issue). A source that errors or returns bad data
    ``failure_threshold`` times in a row is tripped (circuit-broken) until
    ``reset_after_seconds`` elapses.
    """

    def __init__(
        self,
        sources: Sequence[Tuple[str, Callable[..., pd.DataFrame]]],
        *,
        monitor: Optional[DataQualityMonitor] = None,
        failure_threshold: int = 3,
        reset_after_seconds: float = 60.0,
        require_clean: bool = True,
    ) -> None:
        if not sources:
            raise ValueError("at least one source required")
        self._sources = [_SourceState(name=n, fetch=f) for n, f in sources]
        self.monitor = monitor or DataQualityMonitor()
        self.failure_threshold = int(failure_threshold)
        self.reset_after_seconds = float(reset_after_seconds)
        self.require_clean = bool(require_clean)

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _maybe_reset(self, s: _SourceState) -> None:
        if s.tripped and s.last_ok_ms is not None:
            # nothing to reset from last_ok; use a separate trip clock
            pass

    def get_bars(self, symbol: str, **kw) -> Dict[str, Any]:
        """Return {bars, source, failover, attempts} using the first healthy source."""
        attempts: List[Dict[str, Any]] = []
        now = self._now_ms()
        for s in self._sources:
            if s.tripped:
                # auto half-open after reset window
                if (
                    s.last_ok_ms is not None
                    and (now - s.last_ok_ms) > self.reset_after_seconds * 1000
                ):
                    s.tripped = False
                    s.consecutive_failures = 0
                else:
                    attempts.append({"source": s.name, "status": "tripped"})
                    continue
            try:
                df = s.fetch(symbol, **kw)
            except Exception as exc:
                s.consecutive_failures += 1
                s.fail_count += 1
                attempts.append({"source": s.name, "status": "error", "error": str(exc)})
                if s.consecutive_failures >= self.failure_threshold:
                    s.tripped = True
                    s.last_ok_ms = now
                continue
            rep = self.monitor.check(df, symbol=symbol, now_ms=kw.get("now_ms"))
            if self.require_clean and not rep.clean:
                s.consecutive_failures += 1
                s.fail_count += 1
                attempts.append({"source": s.name, "status": "bad_data", "report": rep.to_dict()})
                if s.consecutive_failures >= self.failure_threshold:
                    s.tripped = True
                    s.last_ok_ms = now
                continue
            # success
            s.consecutive_failures = 0
            s.ok_count += 1
            s.last_ok_ms = now
            attempts.append({"source": s.name, "status": "ok"})
            return {
                "bars": df,
                "source": s.name,
                "failover": s is not self._sources[0],
                "quality": rep.to_dict(),
                "attempts": attempts,
            }
        return {
            "bars": None,
            "source": None,
            "failover": True,
            "error": "all sources unavailable",
            "attempts": attempts,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "sources": [
                {
                    "name": s.name,
                    "tripped": s.tripped,
                    "ok": s.ok_count,
                    "fail": s.fail_count,
                    "consecutive_failures": s.consecutive_failures,
                }
                for s in self._sources
            ]
        }

    def reset(self, name: Optional[str] = None) -> None:
        for s in self._sources:
            if name is None or s.name == name:
                s.tripped = False
                s.consecutive_failures = 0


__all__ = [
    "QualityIssue",
    "DataQualityReport",
    "DataQualityMonitor",
    "modified_zscores",
    "cross_source_reconcile",
    "MarketDataRouter",
]
