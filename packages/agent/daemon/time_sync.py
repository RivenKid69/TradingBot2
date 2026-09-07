# -*- coding: utf-8 -*-
"""
Time Sync Checker - Time synchronization verification.

Design Doc D1/Phase 5:
- Verify time sync (допустимый drift)
- Verify корректность timestamps/idempotency

CCEA Phase 5 Component.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Final, List, Optional, Tuple


# NTP Constants
NTP_SERVERS: Final[List[str]] = [
    "time.google.com",
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.apple.com",
]
NTP_PORT: Final[int] = 123
NTP_PACKET_FORMAT: Final[str] = "!12I"
NTP_TIMEOUT: Final[float] = 5.0


class TimeSyncError(Exception):
    """Time synchronization error."""

    pass


class TimeDriftError(TimeSyncError):
    """Time drift exceeds acceptable threshold."""

    pass


@dataclass
class TimeSyncResult:
    """
    Result of time synchronization check.
    """

    synchronized: bool = False
    drift_seconds: float = 0.0
    drift_ms: int = 0
    server_time: Optional[datetime] = None
    local_time: Optional[datetime] = None
    ntp_server: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def is_acceptable(self) -> bool:
        """Check if drift is within acceptable range."""
        return self.synchronized and abs(self.drift_seconds) < 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "synchronized": self.synchronized,
            "drift_seconds": self.drift_seconds,
            "drift_ms": self.drift_ms,
            "server_time": self.server_time.isoformat() if self.server_time else None,
            "local_time": self.local_time.isoformat() if self.local_time else None,
            "ntp_server": self.ntp_server,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class TimeSyncConfig:
    """
    Time sync configuration.
    """

    # Drift tolerance
    max_drift_seconds: float = 5.0  # Design Doc D1: допустимый drift
    warning_drift_seconds: float = 2.0

    # Check intervals
    check_interval_seconds: int = 60
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    # NTP settings
    ntp_servers: List[str] = field(default_factory=lambda: list(NTP_SERVERS))
    ntp_timeout: float = NTP_TIMEOUT

    # Behavior
    halt_on_drift: bool = True
    allow_degraded_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_drift_seconds": self.max_drift_seconds,
            "warning_drift_seconds": self.warning_drift_seconds,
            "check_interval_seconds": self.check_interval_seconds,
            "ntp_servers": self.ntp_servers,
            "halt_on_drift": self.halt_on_drift,
        }


class TimeSyncChecker:
    """
    Verifies time synchronization with NTP servers.

    Design Doc D1 Pre-flight check:
    - Verify time sync (допустимый drift)
    - Verify корректность timestamps/idempotency

    Usage:
        checker = TimeSyncChecker()

        # One-time check
        result = checker.check()
        if not result.is_acceptable:
            # Handle drift

        # Continuous monitoring
        checker.start_monitoring(on_drift=handle_drift)
    """

    def __init__(
        self,
        config: Optional[TimeSyncConfig] = None,
        on_drift: Optional[Callable[[TimeSyncResult], None]] = None,
    ):
        """
        Initialize time sync checker.

        Args:
            config: Configuration
            on_drift: Callback when drift exceeds threshold
        """
        self.config = config or TimeSyncConfig()
        self._on_drift = on_drift

        # State
        self._last_check: Optional[TimeSyncResult] = None
        self._check_history: List[TimeSyncResult] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Lock
        self._lock = threading.RLock()

    @property
    def last_result(self) -> Optional[TimeSyncResult]:
        """Get last check result."""
        return self._last_check

    @property
    def is_synchronized(self) -> bool:
        """Check if time is synchronized."""
        if self._last_check is None:
            return False
        return self._last_check.synchronized

    @property
    def current_drift_seconds(self) -> float:
        """Get current drift in seconds."""
        if self._last_check is None:
            return float("inf")
        return self._last_check.drift_seconds

    def check(self) -> TimeSyncResult:
        """
        Perform time synchronization check.

        Tries multiple NTP servers until one succeeds.

        Returns:
            TimeSyncResult with drift information
        """
        errors = []

        for server in self.config.ntp_servers:
            for attempt in range(self.config.retry_count):
                try:
                    result = self._query_ntp(server)
                    with self._lock:
                        self._last_check = result
                        self._check_history.append(result)
                        if len(self._check_history) > 100:
                            self._check_history = self._check_history[-100:]

                    # Check drift
                    if abs(result.drift_seconds) >= self.config.max_drift_seconds:
                        if self._on_drift:
                            self._on_drift(result)

                    return result

                except Exception as e:
                    errors.append(f"{server}: {str(e)}")
                    if attempt < self.config.retry_count - 1:
                        time.sleep(self.config.retry_delay_seconds)

        # All servers failed
        result = TimeSyncResult(
            synchronized=False,
            error=f"All NTP servers failed: {'; '.join(errors)}",
        )
        with self._lock:
            self._last_check = result
        return result

    def check_timestamp_valid(
        self,
        timestamp: datetime,
        max_age_seconds: float = 60.0,
        max_future_seconds: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Verify timestamp is valid (not too old or in future).

        Used for idempotency and replay attack prevention.

        Args:
            timestamp: Timestamp to check
            max_age_seconds: Maximum age
            max_future_seconds: Maximum time in future

        Returns:
            (is_valid, reason)
        """
        # Ensure timezone aware
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        diff = (now - timestamp).total_seconds()

        if diff > max_age_seconds:
            return False, f"Timestamp too old: {diff:.1f}s ago (max {max_age_seconds}s)"

        if diff < -max_future_seconds:
            return False, f"Timestamp in future: {-diff:.1f}s ahead (max {max_future_seconds}s)"

        return True, "valid"

    def start_monitoring(
        self,
        on_drift: Optional[Callable[[TimeSyncResult], None]] = None,
    ) -> None:
        """
        Start continuous time monitoring.

        Args:
            on_drift: Optional callback for drift events
        """
        if self._monitoring:
            return

        if on_drift:
            self._on_drift = on_drift

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check()
            except Exception:
                pass

            self._stop_event.wait(self.config.check_interval_seconds)

    def _query_ntp(self, server: str) -> TimeSyncResult:
        """
        Query NTP server for time.

        Args:
            server: NTP server hostname

        Returns:
            TimeSyncResult with drift information
        """
        # Create NTP request packet
        # LI=0, VN=3, Mode=3 (client), Stratum=0, Poll=0, Precision=0
        packet = bytearray(48)
        packet[0] = 0x1B  # LI=0, VN=3, Mode=3

        # Record send time
        send_time = time.time()
        local_time = datetime.now(timezone.utc)

        # Send request
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.config.ntp_timeout)

        try:
            sock.sendto(packet, (server, NTP_PORT))
            response, _ = sock.recvfrom(48)
        finally:
            sock.close()

        # Record receive time
        recv_time = time.time()
        latency_ms = (recv_time - send_time) * 1000

        # Parse response
        unpacked = struct.unpack(NTP_PACKET_FORMAT, response)

        # Get transmit timestamp (seconds since 1900-01-01)
        ntp_time_int = unpacked[10]
        ntp_time_frac = unpacked[11]

        # Convert to Unix timestamp (subtract 70 years in seconds)
        ntp_epoch = 2208988800
        server_timestamp = ntp_time_int - ntp_epoch + ntp_time_frac / (2**32)

        # Calculate drift
        # Account for network latency by using midpoint
        local_timestamp = (send_time + recv_time) / 2
        drift_seconds = server_timestamp - local_timestamp

        server_time = datetime.fromtimestamp(server_timestamp, tz=timezone.utc)

        return TimeSyncResult(
            synchronized=True,
            drift_seconds=drift_seconds,
            drift_ms=int(drift_seconds * 1000),
            server_time=server_time,
            local_time=local_time,
            ntp_server=server,
            latency_ms=latency_ms,
        )

    def get_corrected_time(self) -> datetime:
        """
        Get current time corrected for drift.

        Returns:
            Drift-corrected UTC time
        """
        now = datetime.now(timezone.utc)
        if self._last_check and self._last_check.synchronized:
            # Apply drift correction
            corrected = now.timestamp() + self._last_check.drift_seconds
            return datetime.fromtimestamp(corrected, tz=timezone.utc)
        return now

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get time sync statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            if not self._check_history:
                return {
                    "checks": 0,
                    "avg_drift_ms": 0,
                    "max_drift_ms": 0,
                    "successful_checks": 0,
                }

            successful = [r for r in self._check_history if r.synchronized]
            drifts = [r.drift_ms for r in successful]

            return {
                "checks": len(self._check_history),
                "successful_checks": len(successful),
                "avg_drift_ms": sum(drifts) / len(drifts) if drifts else 0,
                "max_drift_ms": max(abs(d) for d in drifts) if drifts else 0,
                "min_drift_ms": min(drifts) if drifts else 0,
                "last_check": self._last_check.to_dict() if self._last_check else None,
            }


def verify_time_sync(max_drift_seconds: float = 5.0) -> Tuple[bool, str]:
    """
    Quick utility to verify time synchronization.

    Args:
        max_drift_seconds: Maximum acceptable drift

    Returns:
        (is_ok, message)
    """
    checker = TimeSyncChecker(config=TimeSyncConfig(max_drift_seconds=max_drift_seconds))
    result = checker.check()

    if not result.synchronized:
        return False, f"Time sync check failed: {result.error}"

    if abs(result.drift_seconds) >= max_drift_seconds:
        return (
            False,
            f"Time drift ({result.drift_seconds:.2f}s) exceeds maximum ({max_drift_seconds}s)",
        )

    return True, f"Time synchronized (drift: {result.drift_ms}ms)"
