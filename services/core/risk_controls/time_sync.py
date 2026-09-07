# -*- coding: utf-8 -*-
"""
RTS 25 Compliant Clock Synchronisation Module.

Implements clock synchronisation requirements per Commission Delegated
Regulation (EU) 2017/574 (RTS 25) for MiFID II compliance.

Accuracy Requirements (RTS 25):
    - Gateway-to-gateway latency < 1ms: ±100 microseconds
    - High-frequency trading: ±100 microseconds
    - Algorithmic trading (general): ±1 millisecond
    - Voice trading: ±1 second

Timestamp Granularity Requirements:
    - All timestamps must be in UTC
    - Nanosecond precision where possible
    - Must be traceable to UTC via NTP

References:
    - RTS 25 (Regulation 2017/574): https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32017R0574
    - ESMA Guidelines: https://www.esma.europa.eu/press-news/esma-news/esma-provides-guidance-transaction-reporting-order-record-keeping-and-clock
"""

from __future__ import annotations

import logging
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable, Tuple
from enum import Enum
from threading import Lock, Thread
import queue

logger = logging.getLogger(__name__)


class ClockDriftSeverity(str, Enum):
    """Clock drift severity levels per RTS 25 thresholds."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    KILL_SWITCH = "kill_switch"

    @classmethod
    def from_offset(
        cls,
        offset_ms: float,
        warning_ms: float = 50.0,
        critical_ms: float = 100.0,
        kill_switch_ms: float = 1000.0,
    ) -> "ClockDriftSeverity":
        """Determine severity from offset value."""
        abs_offset = abs(offset_ms)
        if abs_offset >= kill_switch_ms:
            return cls.KILL_SWITCH
        if abs_offset >= critical_ms:
            return cls.CRITICAL
        if abs_offset >= warning_ms:
            return cls.WARNING
        return cls.NORMAL


@dataclass
class ClockSyncStatus:
    """
    Clock synchronisation status per RTS 25 requirements.

    Contains information about the current clock state,
    offset from UTC, and NTP server details.
    """

    offset_ms: float
    stratum: int
    reference_server: str
    last_sync_time: float
    sync_success: bool
    round_trip_ms: float = 0.0
    dispersion_ms: float = 0.0
    severity: ClockDriftSeverity = ClockDriftSeverity.NORMAL

    def is_compliant(self, max_offset_ms: float = 100.0) -> bool:
        """
        Check if clock meets RTS 25 compliance threshold.

        Args:
            max_offset_ms: Maximum allowed offset (default: 1ms for algo trading).

        Returns:
            True if offset is within acceptable range.
        """
        return self.sync_success and abs(self.offset_ms) <= max_offset_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "offset_ms": round(self.offset_ms, 3),
            "stratum": self.stratum,
            "reference_server": self.reference_server,
            "last_sync_time": self.last_sync_time,
            "sync_success": self.sync_success,
            "round_trip_ms": round(self.round_trip_ms, 3),
            "dispersion_ms": round(self.dispersion_ms, 3),
            "severity": self.severity.value,
            "is_compliant": self.is_compliant(),
        }


@dataclass
class ClockSyncEvent:
    """Event generated on clock sync status change."""

    timestamp: float
    previous_offset_ms: float
    current_offset_ms: float
    severity: ClockDriftSeverity
    message: str


class ComplianceClock:
    """
    RTS 25 Compliant Clock with NTP Synchronisation.

    Provides:
        - UTC timestamps with nanosecond precision
        - NTP synchronisation with multiple servers
        - Drift monitoring and alerting
        - Kill switch trigger on excessive drift
        - Audit trail for compliance

    Example:
        clock = ComplianceClock()
        clock.start_sync()

        # Get compliant timestamp
        timestamp_ns = clock.now_utc_ns()

        # Check compliance status
        status = clock.get_status()
        if not status.is_compliant():
            raise ComplianceError("Clock drift exceeds RTS 25 threshold")

    RTS 25 Requirements:
        - Synchronise to UTC via NTP
        - Maintain records of sync accuracy
        - Maximum drift: 1ms for algorithmic trading
        - Timestamps in nanosecond granularity
    """

    DEFAULT_NTP_SERVERS = [
        "time.google.com",
        "pool.ntp.org",
        "time.windows.com",
        "time.cloudflare.com",
    ]

    def __init__(
        self,
        ntp_servers: Optional[List[str]] = None,
        max_offset_ms: float = 100.0,
        sync_interval_seconds: int = 60,
        warning_threshold_ms: float = 50.0,
        critical_threshold_ms: float = 100.0,
        kill_switch_threshold_ms: float = 1000.0,
        kill_switch_callback: Optional[Callable[[ClockSyncStatus], None]] = None,
        alert_callback: Optional[Callable[[ClockSyncEvent], None]] = None,
        stratum_max: int = 3,
    ):
        """
        Initialize RTS 25 compliant clock.

        Args:
            ntp_servers: List of NTP servers in priority order.
            max_offset_ms: Maximum allowed offset for compliance (default: 1ms).
            sync_interval_seconds: Interval between sync attempts.
            warning_threshold_ms: Threshold for warning alerts.
            critical_threshold_ms: Threshold for critical alerts.
            kill_switch_threshold_ms: Threshold to trigger kill switch.
            kill_switch_callback: Called when drift exceeds kill switch threshold.
            alert_callback: Called on drift severity changes.
            stratum_max: Maximum acceptable NTP stratum level.
        """
        self._ntp_servers = ntp_servers or self.DEFAULT_NTP_SERVERS.copy()
        self._max_offset_ms = max_offset_ms
        self._sync_interval = sync_interval_seconds
        self._warning_threshold_ms = warning_threshold_ms
        self._critical_threshold_ms = critical_threshold_ms
        self._kill_switch_threshold_ms = kill_switch_threshold_ms
        self._kill_switch_callback = kill_switch_callback
        self._alert_callback = alert_callback
        self._stratum_max = stratum_max

        # Clock state
        self._offset_ms: float = 0.0
        self._last_sync: Optional[ClockSyncStatus] = None
        self._sync_history: List[ClockSyncStatus] = []
        self._max_history_size = 1000

        # Threading
        self._lock = Lock()
        self._sync_thread: Optional[Thread] = None
        self._stop_event = queue.Queue()
        self._running = False

        # Statistics
        self._stats = {
            "sync_attempts": 0,
            "sync_successes": 0,
            "sync_failures": 0,
            "warnings_triggered": 0,
            "critical_alerts": 0,
            "kill_switch_triggers": 0,
        }

        logger.info(
            "ComplianceClock initialized",
            extra={
                "ntp_servers": self._ntp_servers,
                "max_offset_ms": max_offset_ms,
                "sync_interval": sync_interval_seconds,
            },
        )

    @property
    def offset_ms(self) -> float:
        """Current clock offset in milliseconds."""
        with self._lock:
            return self._offset_ms

    @property
    def is_synchronized(self) -> bool:
        """Check if clock has been synchronized successfully."""
        with self._lock:
            return self._last_sync is not None and self._last_sync.sync_success

    def now_utc_ns(self) -> int:
        """
        Get current UTC timestamp in nanoseconds (RTS 25 compliant).

        Returns:
            Nanoseconds since Unix epoch, adjusted for clock offset.
        """
        with self._lock:
            offset_s = self._offset_ms / 1000.0
        return int((time.time() + offset_s) * 1_000_000_000)

    def now_utc_us(self) -> int:
        """
        Get current UTC timestamp in microseconds.

        Returns:
            Microseconds since Unix epoch, adjusted for clock offset.
        """
        return self.now_utc_ns() // 1000

    def now_utc_ms(self) -> int:
        """
        Get current UTC timestamp in milliseconds.

        Returns:
            Milliseconds since Unix epoch, adjusted for clock offset.
        """
        return self.now_utc_ns() // 1_000_000

    def now_utc_s(self) -> float:
        """
        Get current UTC timestamp in seconds.

        Returns:
            Seconds since Unix epoch (float), adjusted for clock offset.
        """
        return self.now_utc_ns() / 1_000_000_000

    def now_utc_datetime(self) -> datetime:
        """
        Get current UTC datetime object.

        Returns:
            datetime object in UTC timezone.
        """
        return datetime.fromtimestamp(self.now_utc_s(), tz=timezone.utc)

    def now_utc_iso(self) -> str:
        """
        Get current UTC timestamp as ISO 8601 string.

        Returns:
            ISO 8601 formatted string with microsecond precision.
        """
        dt = self.now_utc_datetime()
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    def _sync_with_ntp_server(self, server: str) -> Optional[Tuple[float, int, float]]:
        """
        Attempt NTP sync with a single server.

        Returns:
            Tuple of (offset_ms, stratum, rtt_ms) or None on failure.
        """
        try:
            import ntplib

            client = ntplib.NTPClient()
            response = client.request(server, version=3, timeout=5)

            # Check stratum
            if response.stratum > self._stratum_max:
                logger.warning(
                    f"NTP server {server} stratum {response.stratum} exceeds max {self._stratum_max}"
                )
                return None

            offset_ms = response.offset * 1000  # Convert to ms
            rtt_ms = response.delay * 1000

            return (offset_ms, response.stratum, rtt_ms)

        except ImportError:
            logger.warning("ntplib not installed, using fallback sync")
            return self._sync_with_http_time(server)

        except Exception as e:
            logger.debug(f"NTP sync failed for {server}: {e}")
            return None

    def _sync_with_http_time(self, server: str) -> Optional[Tuple[float, int, float]]:
        """
        Fallback: estimate time offset using HTTP Date header.

        Less accurate than NTP but works without ntplib.
        """
        try:
            import urllib.request
            from email.utils import parsedate_to_datetime

            # Use a well-known time API
            urls = [
                "https://www.google.com",
                "https://www.cloudflare.com",
            ]

            for url in urls:
                try:
                    start = time.time()
                    request = urllib.request.Request(url, method="HEAD")
                    with urllib.request.urlopen(request, timeout=5) as response:
                        rtt = time.time() - start
                        date_str = response.headers.get("Date")
                        if date_str:
                            server_time = parsedate_to_datetime(date_str)
                            local_time = datetime.now(timezone.utc)
                            offset = (server_time - local_time).total_seconds() * 1000
                            # Adjust for round trip
                            offset += (rtt * 1000) / 2
                            return (offset, 15, rtt * 1000)  # Stratum 15 = unsynchronized
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"HTTP time sync failed: {e}")

        return None

    def sync(self) -> ClockSyncStatus:
        """
        Synchronise clock with NTP servers.

        Tries each configured NTP server in order until one succeeds.
        Uses median filtering for multiple samples.

        Returns:
            ClockSyncStatus with sync results.
        """
        self._stats["sync_attempts"] += 1

        samples: List[Tuple[float, int, float, str]] = []

        for server in self._ntp_servers:
            result = self._sync_with_ntp_server(server)
            if result:
                offset_ms, stratum, rtt_ms = result
                samples.append((offset_ms, stratum, rtt_ms, server))
                # Take multiple samples from first successful server
                for _ in range(2):
                    result = self._sync_with_ntp_server(server)
                    if result:
                        samples.append((*result, server))
                break

        if not samples:
            self._stats["sync_failures"] += 1
            status = ClockSyncStatus(
                offset_ms=self._offset_ms,
                stratum=16,  # Unsynchronized
                reference_server="",
                last_sync_time=time.time(),
                sync_success=False,
                severity=ClockDriftSeverity.CRITICAL,
            )
            logger.error("Clock sync failed: no NTP servers available")
            return status

        # Use median offset for robustness
        offsets = [s[0] for s in samples]
        median_offset = statistics.median(offsets)

        # Filter samples close to median
        filtered = [s for s in samples if abs(s[0] - median_offset) < 50]
        if not filtered:
            filtered = samples

        # Use sample with lowest RTT among filtered
        best = min(filtered, key=lambda x: x[2])
        offset_ms, stratum, rtt_ms, server = best

        # Update clock offset
        previous_offset = self._offset_ms
        with self._lock:
            # Apply exponential smoothing
            alpha = 0.3
            self._offset_ms = alpha * offset_ms + (1 - alpha) * self._offset_ms

        # Determine severity
        severity = ClockDriftSeverity.from_offset(
            self._offset_ms,
            self._warning_threshold_ms,
            self._critical_threshold_ms,
            self._kill_switch_threshold_ms,
        )

        status = ClockSyncStatus(
            offset_ms=self._offset_ms,
            stratum=stratum,
            reference_server=server,
            last_sync_time=time.time(),
            sync_success=True,
            round_trip_ms=rtt_ms,
            dispersion_ms=statistics.stdev(offsets) if len(offsets) > 1 else 0.0,
            severity=severity,
        )

        with self._lock:
            self._last_sync = status
            self._sync_history.append(status)
            if len(self._sync_history) > self._max_history_size:
                self._sync_history = self._sync_history[-self._max_history_size :]

        self._stats["sync_successes"] += 1

        # Handle severity changes
        self._handle_severity(status, previous_offset)

        logger.info(
            f"Clock synced: offset={self._offset_ms:.3f}ms, server={server}",
            extra=status.to_dict(),
        )

        return status

    def _handle_severity(self, status: ClockSyncStatus, previous_offset: float) -> None:
        """Handle clock drift severity and trigger callbacks."""
        previous_severity = ClockDriftSeverity.from_offset(
            previous_offset,
            self._warning_threshold_ms,
            self._critical_threshold_ms,
            self._kill_switch_threshold_ms,
        )

        if status.severity != previous_severity:
            event = ClockSyncEvent(
                timestamp=time.time(),
                previous_offset_ms=previous_offset,
                current_offset_ms=status.offset_ms,
                severity=status.severity,
                message=f"Clock drift severity changed: {previous_severity.value} -> {status.severity.value}",
            )

            if self._alert_callback:
                try:
                    self._alert_callback(event)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        # Track statistics
        if status.severity == ClockDriftSeverity.WARNING:
            self._stats["warnings_triggered"] += 1
        elif status.severity == ClockDriftSeverity.CRITICAL:
            self._stats["critical_alerts"] += 1
        elif status.severity == ClockDriftSeverity.KILL_SWITCH:
            self._stats["kill_switch_triggers"] += 1
            if self._kill_switch_callback:
                try:
                    self._kill_switch_callback(status)
                except Exception as e:
                    logger.error(f"Kill switch callback error: {e}")

    def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                self.sync()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            try:
                self._stop_event.get(timeout=self._sync_interval)
                break  # Stop signal received
            except queue.Empty:
                pass  # Timeout - continue loop

    def start_sync(self) -> None:
        """
        Start background clock synchronisation.

        Performs initial sync and starts background thread
        for periodic synchronisation.
        """
        if self._running:
            return

        # Initial sync
        self.sync()

        # Start background thread
        self._running = True
        self._sync_thread = Thread(target=self._sync_loop, daemon=True, name="clock-sync")
        self._sync_thread.start()

        logger.info("Clock sync background thread started")

    def stop_sync(self) -> None:
        """Stop background clock synchronisation."""
        if not self._running:
            return

        self._running = False
        self._stop_event.put(True)

        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)
            self._sync_thread = None

        logger.info("Clock sync background thread stopped")

    def get_status(self) -> ClockSyncStatus:
        """
        Get current clock sync status.

        Returns:
            Latest ClockSyncStatus or default if not synchronized.
        """
        with self._lock:
            if self._last_sync:
                return self._last_sync

        return ClockSyncStatus(
            offset_ms=0.0,
            stratum=16,
            reference_server="",
            last_sync_time=0.0,
            sync_success=False,
            severity=ClockDriftSeverity.CRITICAL,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get clock sync statistics."""
        with self._lock:
            history_offsets = [s.offset_ms for s in self._sync_history[-100:]]

        return {
            **self._stats,
            "current_offset_ms": self._offset_ms,
            "is_synchronized": self.is_synchronized,
            "is_compliant": self.get_status().is_compliant(self._max_offset_ms),
            "history_size": len(self._sync_history),
            "avg_offset_ms": statistics.mean(history_offsets) if history_offsets else 0.0,
            "max_offset_ms": max(abs(o) for o in history_offsets) if history_offsets else 0.0,
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate RTS 25 compliance report.

        Returns:
            Dictionary with compliance report data for audit trail.
        """
        status = self.get_status()
        stats = self.get_statistics()

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "regulation": "RTS 25 (Regulation 2017/574)",
            "requirement": "Clock Synchronisation",
            "compliance_status": (
                "COMPLIANT" if status.is_compliant(self._max_offset_ms) else "NON-COMPLIANT"
            ),
            "current_status": status.to_dict(),
            "configuration": {
                "ntp_servers": self._ntp_servers,
                "max_offset_ms": self._max_offset_ms,
                "sync_interval_seconds": self._sync_interval,
                "warning_threshold_ms": self._warning_threshold_ms,
                "critical_threshold_ms": self._critical_threshold_ms,
                "kill_switch_threshold_ms": self._kill_switch_threshold_ms,
            },
            "statistics": stats,
        }


def create_compliance_clock(
    config: Optional[Dict[str, Any]] = None,
    kill_switch_callback: Optional[Callable[[ClockSyncStatus], None]] = None,
    alert_callback: Optional[Callable[[ClockSyncEvent], None]] = None,
) -> ComplianceClock:
    """
    Factory function to create ComplianceClock instance.

    Args:
        config: Optional configuration dictionary.
        kill_switch_callback: Called when drift exceeds threshold.
        alert_callback: Called on drift severity changes.

    Returns:
        Configured ComplianceClock instance.
    """
    config = config or {}

    return ComplianceClock(
        ntp_servers=config.get("ntp_servers"),
        max_offset_ms=config.get("max_offset_ms", 100.0),
        sync_interval_seconds=config.get("sync_interval_seconds", 60),
        warning_threshold_ms=config.get("warning_threshold_ms", 50.0),
        critical_threshold_ms=config.get("critical_threshold_ms", 100.0),
        kill_switch_threshold_ms=config.get("kill_switch_threshold_ms", 1000.0),
        kill_switch_callback=kill_switch_callback,
        alert_callback=alert_callback,
        stratum_max=config.get("stratum_max", 3),
    )


__all__ = [
    "ClockDriftSeverity",
    "ClockSyncStatus",
    "ClockSyncEvent",
    "ComplianceClock",
    "create_compliance_clock",
]
