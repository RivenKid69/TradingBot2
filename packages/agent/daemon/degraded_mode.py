# -*- coding: utf-8 -*-
"""
Degraded Mode Manager - Safe degradation handling.

Design Doc Phase 5:
- Degraded safe режимы: cloud down / network down / data feed invalid
- → halt или ограничение по локальной политике

CCEA Phase 5 Component.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


class DegradedMode(Enum):
    """
    Types of degraded modes.

    Each mode has specific behavior and restrictions.
    """
    NORMAL = auto()  # Full functionality
    CLOUD_UNREACHABLE = auto()  # Cloud connection lost
    NETWORK_DEGRADED = auto()  # Partial network issues
    DATA_FEED_STALE = auto()  # Market data is stale
    DATA_FEED_INVALID = auto()  # Market data is invalid
    BROKER_UNREACHABLE = auto()  # Broker connection lost
    HIGH_LATENCY = auto()  # Latency exceeds threshold
    RATE_LIMITED = auto()  # Hit rate limits
    MANUAL_RESTRICT = auto()  # Manual restriction mode


class DegradedModeAction(Enum):
    """Actions to take in degraded mode."""
    CONTINUE = "continue"  # Continue normally
    RESTRICT = "restrict"  # Restrict new orders
    CLOSE_ONLY = "close_only"  # Only allow position closing
    PAUSE = "pause"  # Pause trading
    HALT = "halt"  # Full halt


@dataclass
class DegradedModeConfig:
    """
    Configuration for degraded mode handling.
    """
    # Thresholds
    cloud_timeout_seconds: int = 60
    data_stale_threshold_seconds: int = 30
    latency_threshold_ms: int = 2000
    broker_timeout_seconds: int = 30

    # Actions by mode
    cloud_unreachable_action: DegradedModeAction = DegradedModeAction.CONTINUE
    data_feed_stale_action: DegradedModeAction = DegradedModeAction.PAUSE
    data_feed_invalid_action: DegradedModeAction = DegradedModeAction.HALT
    broker_unreachable_action: DegradedModeAction = DegradedModeAction.HALT
    high_latency_action: DegradedModeAction = DegradedModeAction.RESTRICT

    # Recovery
    auto_recover: bool = True
    recovery_check_interval_seconds: int = 10
    min_recovery_duration_seconds: int = 30

    # Notifications
    notify_on_enter: bool = True
    notify_on_exit: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cloud_timeout_seconds": self.cloud_timeout_seconds,
            "data_stale_threshold_seconds": self.data_stale_threshold_seconds,
            "latency_threshold_ms": self.latency_threshold_ms,
            "broker_timeout_seconds": self.broker_timeout_seconds,
            "cloud_unreachable_action": self.cloud_unreachable_action.value,
            "data_feed_stale_action": self.data_feed_stale_action.value,
            "data_feed_invalid_action": self.data_feed_invalid_action.value,
            "broker_unreachable_action": self.broker_unreachable_action.value,
            "high_latency_action": self.high_latency_action.value,
        }


@dataclass
class DegradedModeEvent:
    """
    Record of degraded mode event.
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    mode: DegradedMode = DegradedMode.NORMAL
    action_taken: DegradedModeAction = DegradedModeAction.CONTINUE
    entered_at: datetime = field(default_factory=datetime.utcnow)
    exited_at: Optional[datetime] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    auto_recovered: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get duration in degraded mode."""
        end = self.exited_at or datetime.utcnow()
        return (end - self.entered_at).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if mode is still active."""
        return self.exited_at is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "mode": self.mode.name,
            "action_taken": self.action_taken.value,
            "entered_at": self.entered_at.isoformat(),
            "exited_at": self.exited_at.isoformat() if self.exited_at else None,
            "reason": self.reason,
            "details": self.details,
            "duration_seconds": self.duration_seconds,
            "is_active": self.is_active,
            "auto_recovered": self.auto_recovered,
        }


class DegradedModeManager:
    """
    Manages degraded mode states and transitions.

    Design Doc Phase 5:
    - Detects degraded conditions (cloud down, network down, data feed invalid)
    - Applies appropriate restrictions per local policy
    - Supports auto-recovery when conditions improve
    - Agent автономно работает без cloud (safe degraded mode)

    Usage:
        manager = DegradedModeManager(config)

        # Report conditions
        manager.report_cloud_status(connected=False)
        manager.report_data_feed(last_update=datetime.utcnow() - timedelta(seconds=60))

        # Check if action is allowed
        if manager.can_submit_order():
            # Submit order
        else:
            # Skip or queue
    """

    def __init__(
        self,
        config: Optional[DegradedModeConfig] = None,
        on_mode_change: Optional[Callable[[DegradedMode, DegradedModeAction], None]] = None,
    ):
        """
        Initialize degraded mode manager.

        Args:
            config: Configuration
            on_mode_change: Callback when mode changes
        """
        self.config = config or DegradedModeConfig()
        self._on_mode_change = on_mode_change

        # State
        self._current_modes: Set[DegradedMode] = {DegradedMode.NORMAL}
        self._current_action = DegradedModeAction.CONTINUE
        self._events: List[DegradedModeEvent] = []
        self._active_events: Dict[DegradedMode, DegradedModeEvent] = {}

        # Timestamps
        self._last_cloud_contact: Optional[datetime] = None
        self._last_data_update: Optional[datetime] = None
        self._last_broker_contact: Optional[datetime] = None

        # Thread safety
        self._lock = threading.RLock()

        # Recovery monitor
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def current_mode(self) -> DegradedMode:
        """Get highest priority degraded mode."""
        # Priority order (worst first)
        priority = [
            DegradedMode.BROKER_UNREACHABLE,
            DegradedMode.DATA_FEED_INVALID,
            DegradedMode.DATA_FEED_STALE,
            DegradedMode.CLOUD_UNREACHABLE,
            DegradedMode.HIGH_LATENCY,
            DegradedMode.RATE_LIMITED,
            DegradedMode.NETWORK_DEGRADED,
            DegradedMode.MANUAL_RESTRICT,
            DegradedMode.NORMAL,
        ]
        for mode in priority:
            if mode in self._current_modes:
                return mode
        return DegradedMode.NORMAL

    @property
    def current_action(self) -> DegradedModeAction:
        """Get current action based on modes."""
        return self._current_action

    @property
    def is_degraded(self) -> bool:
        """Check if in any degraded mode."""
        return self._current_modes != {DegradedMode.NORMAL}

    @property
    def active_modes(self) -> Set[DegradedMode]:
        """Get all active degraded modes."""
        return self._current_modes.copy()

    def can_submit_order(self) -> bool:
        """Check if new orders can be submitted."""
        return self._current_action in (
            DegradedModeAction.CONTINUE,
            DegradedModeAction.RESTRICT,
        )

    def can_close_position(self) -> bool:
        """Check if positions can be closed."""
        return self._current_action in (
            DegradedModeAction.CONTINUE,
            DegradedModeAction.RESTRICT,
            DegradedModeAction.CLOSE_ONLY,
        )

    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._current_action == DegradedModeAction.HALT

    def is_paused(self) -> bool:
        """Check if trading is paused."""
        return self._current_action in (
            DegradedModeAction.PAUSE,
            DegradedModeAction.HALT,
        )

    # ===== Status Reporting =====

    def report_cloud_status(
        self,
        connected: bool,
        latency_ms: Optional[int] = None,
    ) -> Optional[DegradedModeEvent]:
        """
        Report cloud connection status.

        Args:
            connected: Whether connected to cloud
            latency_ms: Optional latency measurement

        Returns:
            DegradedModeEvent if mode changed
        """
        with self._lock:
            if connected:
                self._last_cloud_contact = datetime.utcnow()
                return self._exit_mode(DegradedMode.CLOUD_UNREACHABLE)
            else:
                # Check if timeout exceeded
                if self._last_cloud_contact:
                    elapsed = (datetime.utcnow() - self._last_cloud_contact).total_seconds()
                    if elapsed > self.config.cloud_timeout_seconds:
                        return self._enter_mode(
                            DegradedMode.CLOUD_UNREACHABLE,
                            self.config.cloud_unreachable_action,
                            f"Cloud unreachable for {elapsed:.0f}s",
                            {"elapsed_seconds": elapsed},
                        )
                else:
                    # Never connected
                    return self._enter_mode(
                        DegradedMode.CLOUD_UNREACHABLE,
                        self.config.cloud_unreachable_action,
                        "Cloud connection not established",
                    )
        return None

    def report_data_feed(
        self,
        last_update: Optional[datetime],
        is_valid: bool = True,
    ) -> Optional[DegradedModeEvent]:
        """
        Report data feed status.

        Args:
            last_update: Time of last data update
            is_valid: Whether data is valid

        Returns:
            DegradedModeEvent if mode changed
        """
        with self._lock:
            if not is_valid:
                return self._enter_mode(
                    DegradedMode.DATA_FEED_INVALID,
                    self.config.data_feed_invalid_action,
                    "Data feed marked as invalid",
                )

            # Exit invalid mode if valid
            self._exit_mode(DegradedMode.DATA_FEED_INVALID)

            if last_update is None:
                return self._enter_mode(
                    DegradedMode.DATA_FEED_STALE,
                    self.config.data_feed_stale_action,
                    "No data feed updates received",
                )

            self._last_data_update = last_update
            elapsed = (datetime.utcnow() - last_update).total_seconds()

            if elapsed > self.config.data_stale_threshold_seconds:
                return self._enter_mode(
                    DegradedMode.DATA_FEED_STALE,
                    self.config.data_feed_stale_action,
                    f"Data feed stale for {elapsed:.0f}s",
                    {"elapsed_seconds": elapsed},
                )
            else:
                return self._exit_mode(DegradedMode.DATA_FEED_STALE)

        return None

    def report_broker_status(
        self,
        connected: bool,
    ) -> Optional[DegradedModeEvent]:
        """
        Report broker connection status.

        Args:
            connected: Whether connected to broker

        Returns:
            DegradedModeEvent if mode changed
        """
        with self._lock:
            if connected:
                self._last_broker_contact = datetime.utcnow()
                return self._exit_mode(DegradedMode.BROKER_UNREACHABLE)
            else:
                if self._last_broker_contact:
                    elapsed = (datetime.utcnow() - self._last_broker_contact).total_seconds()
                    if elapsed > self.config.broker_timeout_seconds:
                        return self._enter_mode(
                            DegradedMode.BROKER_UNREACHABLE,
                            self.config.broker_unreachable_action,
                            f"Broker unreachable for {elapsed:.0f}s",
                            {"elapsed_seconds": elapsed},
                        )
                else:
                    return self._enter_mode(
                        DegradedMode.BROKER_UNREACHABLE,
                        self.config.broker_unreachable_action,
                        "Broker connection not established",
                    )
        return None

    def report_latency(
        self,
        latency_ms: int,
    ) -> Optional[DegradedModeEvent]:
        """
        Report latency measurement.

        Args:
            latency_ms: Latency in milliseconds

        Returns:
            DegradedModeEvent if mode changed
        """
        with self._lock:
            if latency_ms > self.config.latency_threshold_ms:
                return self._enter_mode(
                    DegradedMode.HIGH_LATENCY,
                    self.config.high_latency_action,
                    f"High latency: {latency_ms}ms",
                    {"latency_ms": latency_ms},
                )
            else:
                return self._exit_mode(DegradedMode.HIGH_LATENCY)
        return None

    def enter_manual_restrict(self, reason: str = "Manual restriction") -> DegradedModeEvent:
        """
        Manually enter restricted mode.

        Args:
            reason: Reason for restriction

        Returns:
            DegradedModeEvent
        """
        with self._lock:
            return self._enter_mode(
                DegradedMode.MANUAL_RESTRICT,
                DegradedModeAction.RESTRICT,
                reason,
            )

    def exit_manual_restrict(self) -> Optional[DegradedModeEvent]:
        """Exit manual restriction mode."""
        with self._lock:
            return self._exit_mode(DegradedMode.MANUAL_RESTRICT)

    # ===== Internal Mode Management =====

    def _enter_mode(
        self,
        mode: DegradedMode,
        action: DegradedModeAction,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> DegradedModeEvent:
        """Enter a degraded mode."""
        if mode in self._current_modes and mode != DegradedMode.NORMAL:
            # Already in this mode, update event
            if mode in self._active_events:
                event = self._active_events[mode]
                event.details.update(details or {})
                return event

        # Create event
        event = DegradedModeEvent(
            mode=mode,
            action_taken=action,
            reason=reason,
            details=details or {},
        )

        # Update state
        self._current_modes.discard(DegradedMode.NORMAL)
        self._current_modes.add(mode)
        self._active_events[mode] = event
        self._events.append(event)

        # Trim events
        if len(self._events) > 1000:
            self._events = self._events[-1000:]

        # Update action
        self._update_current_action()

        # Callback
        if self.config.notify_on_enter and self._on_mode_change:
            try:
                self._on_mode_change(mode, action)
            except Exception:
                pass

        return event

    def _exit_mode(self, mode: DegradedMode) -> Optional[DegradedModeEvent]:
        """Exit a degraded mode."""
        if mode not in self._current_modes:
            return None

        # Update event
        if mode in self._active_events:
            event = self._active_events[mode]
            event.exited_at = datetime.utcnow()
            event.auto_recovered = self.config.auto_recover
            del self._active_events[mode]
        else:
            event = None

        # Update state
        self._current_modes.discard(mode)
        if not self._current_modes:
            self._current_modes.add(DegradedMode.NORMAL)

        # Update action
        self._update_current_action()

        # Callback
        if self.config.notify_on_exit and self._on_mode_change:
            try:
                self._on_mode_change(self.current_mode, self._current_action)
            except Exception:
                pass

        return event

    def _update_current_action(self) -> None:
        """Update current action based on active modes."""
        # Get most restrictive action
        action_priority = [
            DegradedModeAction.HALT,
            DegradedModeAction.PAUSE,
            DegradedModeAction.CLOSE_ONLY,
            DegradedModeAction.RESTRICT,
            DegradedModeAction.CONTINUE,
        ]

        for action in action_priority:
            for event in self._active_events.values():
                if event.action_taken == action:
                    self._current_action = action
                    return

        self._current_action = DegradedModeAction.CONTINUE

    # ===== Monitoring =====

    def start_monitoring(self) -> None:
        """Start background monitoring for auto-recovery."""
        if self._monitoring:
            return

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_auto_recovery()
            except Exception:
                pass

            self._stop_event.wait(self.config.recovery_check_interval_seconds)

    def _check_auto_recovery(self) -> None:
        """Check if any modes can auto-recover."""
        with self._lock:
            now = datetime.utcnow()

            # Check cloud recovery
            if DegradedMode.CLOUD_UNREACHABLE in self._current_modes:
                if self._last_cloud_contact:
                    elapsed = (now - self._last_cloud_contact).total_seconds()
                    if elapsed < self.config.cloud_timeout_seconds:
                        self._exit_mode(DegradedMode.CLOUD_UNREACHABLE)

            # Check data feed recovery
            if DegradedMode.DATA_FEED_STALE in self._current_modes:
                if self._last_data_update:
                    elapsed = (now - self._last_data_update).total_seconds()
                    if elapsed < self.config.data_stale_threshold_seconds:
                        self._exit_mode(DegradedMode.DATA_FEED_STALE)

            # Check broker recovery
            if DegradedMode.BROKER_UNREACHABLE in self._current_modes:
                if self._last_broker_contact:
                    elapsed = (now - self._last_broker_contact).total_seconds()
                    if elapsed < self.config.broker_timeout_seconds:
                        self._exit_mode(DegradedMode.BROKER_UNREACHABLE)

    # ===== Status =====

    def get_status(self) -> Dict[str, Any]:
        """
        Get current degraded mode status.

        Returns:
            Status dictionary
        """
        with self._lock:
            return {
                "is_degraded": self.is_degraded,
                "current_mode": self.current_mode.name,
                "current_action": self._current_action.value,
                "active_modes": [m.name for m in self._current_modes],
                "can_submit_order": self.can_submit_order(),
                "can_close_position": self.can_close_position(),
                "is_halted": self.is_halted(),
                "is_paused": self.is_paused(),
                "active_events": [e.to_dict() for e in self._active_events.values()],
                "last_cloud_contact": self._last_cloud_contact.isoformat() if self._last_cloud_contact else None,
                "last_data_update": self._last_data_update.isoformat() if self._last_data_update else None,
                "last_broker_contact": self._last_broker_contact.isoformat() if self._last_broker_contact else None,
            }

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get degraded mode history.

        Args:
            limit: Maximum events to return

        Returns:
            List of event dictionaries
        """
        with self._lock:
            return [e.to_dict() for e in self._events[-limit:]]
