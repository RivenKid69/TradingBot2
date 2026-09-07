# -*- coding: utf-8 -*-
"""
services/futures_margin_monitor.py
Real-time margin monitoring service for futures trading.

Provides:
- Continuous margin ratio monitoring
- Multi-level alerts (warning, danger, critical)
- Margin call notifications
- Auto-reduction suggestions
- Historical margin tracking
- Integration with risk guards

Architecture:
    FuturesMarginMonitor
    ├── MarginLevelTracker (threshold tracking)
    ├── MarginCallDetector (call detection)
    ├── MarginHistoryRecorder (historical data)
    └── AlertManager (notifications)

References:
- Binance Margin Risk: https://www.binance.com/en/support/faq/360033525031
- CME SPAN Margin: https://www.cmegroup.com/clearing/risk-management/span-methodology.html

Author: Trading Bot Team
Date: 2025-12-02
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from core_futures import (
    FuturesPosition,
    FuturesType,
    MarginMode,
    PositionSide,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default thresholds (margin ratio = equity / margin_used)
DEFAULT_WARNING_RATIO = Decimal("1.5")  # 150%
DEFAULT_DANGER_RATIO = Decimal("1.2")  # 120%
DEFAULT_CRITICAL_RATIO = Decimal("1.1")  # 110%
DEFAULT_LIQUIDATION_RATIO = Decimal("1.0")  # 100%

# Monitoring intervals
DEFAULT_MONITOR_INTERVAL_SEC = 5.0
DEFAULT_HISTORY_MAX_SIZE = 1000
DEFAULT_ALERT_COOLDOWN_SEC = 300.0  # 5 minutes

# Auto-reduction parameters
DEFAULT_TARGET_MARGIN_RATIO = Decimal("2.0")  # Target 200% after reduction
DEFAULT_REDUCTION_STEP_PCT = Decimal("0.1")  # 10% position reduction per step


# ============================================================================
# ENUMS
# ============================================================================


class MarginLevel(Enum):
    """Margin level classifications."""

    HEALTHY = auto()  # > warning threshold
    WARNING = auto()  # between warning and danger
    DANGER = auto()  # between danger and critical
    CRITICAL = auto()  # between critical and liquidation
    LIQUIDATION = auto()  # at or below liquidation threshold


class MarginAlertType(Enum):
    """Types of margin alerts."""

    LEVEL_CHANGE = auto()
    APPROACHING_WARNING = auto()
    APPROACHING_DANGER = auto()
    APPROACHING_CRITICAL = auto()
    MARGIN_CALL = auto()
    LIQUIDATION_RISK = auto()
    RECOVERED = auto()


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class MarginDataProvider(Protocol):
    """Protocol for margin data providers."""

    def get_account_equity(self) -> Decimal:
        """Get total account equity."""
        ...

    def get_total_margin_used(self) -> Decimal:
        """Get total margin used."""
        ...

    def get_available_margin(self) -> Decimal:
        """Get available margin."""
        ...

    def get_maintenance_margin(self) -> Decimal:
        """Get maintenance margin requirement."""
        ...

    def get_positions(self) -> Dict[str, FuturesPosition]:
        """Get all positions."""
        ...


@runtime_checkable
class AlertCallback(Protocol):
    """Protocol for alert callbacks."""

    def __call__(
        self,
        alert_type: MarginAlertType,
        level: MarginLevel,
        data: Dict[str, Any],
    ) -> None:
        """Handle margin alert."""
        ...


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MarginSnapshot:
    """Point-in-time margin snapshot."""

    timestamp_ms: int
    equity: Decimal
    margin_used: Decimal
    available_margin: Decimal
    maintenance_margin: Decimal
    margin_ratio: Decimal
    level: MarginLevel
    position_count: int
    total_position_value: Decimal

    @property
    def utilization_pct(self) -> Decimal:
        """Margin utilization percentage."""
        if self.equity == 0:
            return Decimal("0")
        return (self.margin_used / self.equity) * 100

    @property
    def buffer_to_liquidation_pct(self) -> Decimal:
        """Buffer to liquidation as percentage."""
        if self.margin_ratio == 0:
            return Decimal("0")
        return (self.margin_ratio - Decimal("1")) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "equity": float(self.equity),
            "margin_used": float(self.margin_used),
            "available_margin": float(self.available_margin),
            "maintenance_margin": float(self.maintenance_margin),
            "margin_ratio": float(self.margin_ratio),
            "level": self.level.name,
            "position_count": self.position_count,
            "total_position_value": float(self.total_position_value),
            "utilization_pct": float(self.utilization_pct),
            "buffer_to_liquidation_pct": float(self.buffer_to_liquidation_pct),
        }


@dataclass
class MarginAlert:
    """Margin alert event."""

    timestamp_ms: int
    alert_type: MarginAlertType
    level: MarginLevel
    previous_level: Optional[MarginLevel]
    margin_ratio: Decimal
    equity: Decimal
    margin_used: Decimal
    message: str
    recommended_action: Optional[str] = None
    severity: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "alert_type": self.alert_type.name,
            "level": self.level.name,
            "previous_level": self.previous_level.name if self.previous_level else None,
            "margin_ratio": float(self.margin_ratio),
            "equity": float(self.equity),
            "margin_used": float(self.margin_used),
            "message": self.message,
            "recommended_action": self.recommended_action,
            "severity": self.severity,
        }


@dataclass
class ReductionSuggestion:
    """Suggestion for position reduction."""

    symbol: str
    current_qty: Decimal
    suggested_reduction_qty: Decimal
    suggested_reduction_pct: Decimal
    reason: str
    priority: int  # 1 = highest priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "current_qty": float(self.current_qty),
            "suggested_reduction_qty": float(self.suggested_reduction_qty),
            "suggested_reduction_pct": float(self.suggested_reduction_pct),
            "reason": self.reason,
            "priority": self.priority,
        }


@dataclass
class MarginMonitorConfig:
    """Configuration for margin monitor."""

    # Thresholds
    warning_ratio: Decimal = DEFAULT_WARNING_RATIO
    danger_ratio: Decimal = DEFAULT_DANGER_RATIO
    critical_ratio: Decimal = DEFAULT_CRITICAL_RATIO
    liquidation_ratio: Decimal = DEFAULT_LIQUIDATION_RATIO

    # Monitoring
    monitor_interval_sec: float = DEFAULT_MONITOR_INTERVAL_SEC
    history_max_size: int = DEFAULT_HISTORY_MAX_SIZE

    # Alerts
    alert_cooldown_sec: float = DEFAULT_ALERT_COOLDOWN_SEC
    enable_alerts: bool = True

    # Auto-reduction
    enable_auto_reduction_suggestions: bool = True
    target_margin_ratio: Decimal = DEFAULT_TARGET_MARGIN_RATIO
    reduction_step_pct: Decimal = DEFAULT_REDUCTION_STEP_PCT

    # Futures type specific
    futures_type: FuturesType = FuturesType.CRYPTO_PERPETUAL

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarginMonitorConfig":
        """Create from dictionary."""
        futures_type_str = d.get("futures_type", "CRYPTO_PERPETUAL")
        if isinstance(futures_type_str, str):
            futures_type = FuturesType[futures_type_str.upper()]
        else:
            futures_type = futures_type_str

        return cls(
            warning_ratio=Decimal(str(d.get("warning_ratio", DEFAULT_WARNING_RATIO))),
            danger_ratio=Decimal(str(d.get("danger_ratio", DEFAULT_DANGER_RATIO))),
            critical_ratio=Decimal(str(d.get("critical_ratio", DEFAULT_CRITICAL_RATIO))),
            liquidation_ratio=Decimal(str(d.get("liquidation_ratio", DEFAULT_LIQUIDATION_RATIO))),
            monitor_interval_sec=float(d.get("monitor_interval_sec", DEFAULT_MONITOR_INTERVAL_SEC)),
            history_max_size=int(d.get("history_max_size", DEFAULT_HISTORY_MAX_SIZE)),
            alert_cooldown_sec=float(d.get("alert_cooldown_sec", DEFAULT_ALERT_COOLDOWN_SEC)),
            enable_alerts=bool(d.get("enable_alerts", True)),
            enable_auto_reduction_suggestions=bool(
                d.get("enable_auto_reduction_suggestions", True)
            ),
            target_margin_ratio=Decimal(
                str(d.get("target_margin_ratio", DEFAULT_TARGET_MARGIN_RATIO))
            ),
            reduction_step_pct=Decimal(
                str(d.get("reduction_step_pct", DEFAULT_REDUCTION_STEP_PCT))
            ),
            futures_type=futures_type,
        )

    @classmethod
    def for_crypto(cls, **kwargs) -> "MarginMonitorConfig":
        """Create config for crypto futures."""
        return cls(futures_type=FuturesType.CRYPTO_PERPETUAL, **kwargs)

    @classmethod
    def for_cme(cls, **kwargs) -> "MarginMonitorConfig":
        """Create config for CME futures with SPAN margin."""
        return cls(
            futures_type=FuturesType.INDEX_FUTURES,
            # CME has stricter margin requirements
            warning_ratio=Decimal("1.3"),
            danger_ratio=Decimal("1.15"),
            critical_ratio=Decimal("1.05"),
            **kwargs,
        )


# ============================================================================
# MARGIN LEVEL TRACKER
# ============================================================================


class MarginLevelTracker:
    """Tracks margin levels and detects transitions."""

    def __init__(self, config: MarginMonitorConfig):
        """
        Initialize level tracker.

        Args:
            config: Monitor configuration
        """
        self._config = config
        self._current_level = MarginLevel.HEALTHY
        self._level_lock = threading.Lock()

    def classify_level(self, margin_ratio: Decimal) -> MarginLevel:
        """
        Classify margin ratio into level.

        Args:
            margin_ratio: Current margin ratio

        Returns:
            MarginLevel classification
        """
        if margin_ratio <= self._config.liquidation_ratio:
            return MarginLevel.LIQUIDATION
        elif margin_ratio <= self._config.critical_ratio:
            return MarginLevel.CRITICAL
        elif margin_ratio <= self._config.danger_ratio:
            return MarginLevel.DANGER
        elif margin_ratio <= self._config.warning_ratio:
            return MarginLevel.WARNING
        else:
            return MarginLevel.HEALTHY

    def update_level(self, margin_ratio: Decimal) -> Tuple[MarginLevel, bool]:
        """
        Update current level and detect transition.

        Args:
            margin_ratio: Current margin ratio

        Returns:
            (new_level, level_changed)
        """
        new_level = self.classify_level(margin_ratio)

        with self._level_lock:
            changed = new_level != self._current_level
            self._current_level = new_level

        return new_level, changed

    @property
    def current_level(self) -> MarginLevel:
        """Get current level."""
        with self._level_lock:
            return self._current_level

    def get_threshold_for_level(self, level: MarginLevel) -> Decimal:
        """Get threshold for level."""
        thresholds = {
            MarginLevel.HEALTHY: self._config.warning_ratio,
            MarginLevel.WARNING: self._config.danger_ratio,
            MarginLevel.DANGER: self._config.critical_ratio,
            MarginLevel.CRITICAL: self._config.liquidation_ratio,
            MarginLevel.LIQUIDATION: Decimal("0"),
        }
        return thresholds.get(level, Decimal("0"))

    def get_buffer_to_next_level(
        self,
        margin_ratio: Decimal,
    ) -> Tuple[MarginLevel, Decimal]:
        """
        Get buffer to next worse level.

        Args:
            margin_ratio: Current margin ratio

        Returns:
            (next_worse_level, buffer_amount)
        """
        current = self.classify_level(margin_ratio)

        next_level_map = {
            MarginLevel.HEALTHY: MarginLevel.WARNING,
            MarginLevel.WARNING: MarginLevel.DANGER,
            MarginLevel.DANGER: MarginLevel.CRITICAL,
            MarginLevel.CRITICAL: MarginLevel.LIQUIDATION,
            MarginLevel.LIQUIDATION: MarginLevel.LIQUIDATION,
        }

        next_level = next_level_map[current]
        threshold = self.get_threshold_for_level(current)
        buffer = margin_ratio - threshold

        return next_level, buffer


# ============================================================================
# ALERT MANAGER
# ============================================================================


class MarginAlertManager:
    """Manages margin alerts with cooldowns."""

    def __init__(
        self,
        config: MarginMonitorConfig,
        callbacks: Optional[List[AlertCallback]] = None,
    ):
        """
        Initialize alert manager.

        Args:
            config: Monitor configuration
            callbacks: Alert callbacks
        """
        self._config = config
        self._callbacks = callbacks or []
        self._last_alert_times: Dict[MarginAlertType, float] = {}
        self._alert_history: Deque[MarginAlert] = deque(maxlen=100)
        self._lock = threading.Lock()

    def add_callback(self, callback: AlertCallback) -> None:
        """Add alert callback."""
        self._callbacks.append(callback)

    def should_alert(self, alert_type: MarginAlertType) -> bool:
        """Check if alert should be sent (respecting cooldown)."""
        with self._lock:
            last_time = self._last_alert_times.get(alert_type, 0.0)
            return time.time() - last_time >= self._config.alert_cooldown_sec

    def send_alert(self, alert: MarginAlert) -> None:
        """Send alert to callbacks."""
        if not self._config.enable_alerts:
            return

        if not self.should_alert(alert.alert_type):
            logger.debug(f"Alert suppressed (cooldown): {alert.alert_type.name}")
            return

        with self._lock:
            self._last_alert_times[alert.alert_type] = time.time()
            self._alert_history.append(alert)

        # Log alert
        log_level = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(alert.severity, logging.INFO)

        logger.log(log_level, f"Margin alert: {alert.message}")

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(alert.alert_type, alert.level, alert.to_dict())
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_alert_history(self, limit: int = 50) -> List[MarginAlert]:
        """Get recent alert history."""
        with self._lock:
            return list(self._alert_history)[-limit:]

    def clear_cooldowns(self) -> None:
        """Clear all cooldowns."""
        with self._lock:
            self._last_alert_times.clear()


# ============================================================================
# MAIN MONITOR CLASS
# ============================================================================


class FuturesMarginMonitor:
    """
    Real-time margin monitoring service.

    Provides continuous monitoring of margin levels with:
    - Multi-level threshold tracking
    - Alert management with cooldowns
    - Historical margin data
    - Position reduction suggestions

    Thread-safe with background monitoring support.

    Example:
        >>> config = MarginMonitorConfig.for_crypto()
        >>> monitor = FuturesMarginMonitor(config, margin_provider)
        >>>
        >>> # Register alert callback
        >>> monitor.register_alert_callback(my_alert_handler)
        >>>
        >>> # Start background monitoring
        >>> monitor.start_background_monitoring()
        >>>
        >>> # Get current status
        >>> snapshot = monitor.get_current_snapshot()
        >>> print(f"Margin ratio: {snapshot.margin_ratio}")
    """

    def __init__(
        self,
        config: MarginMonitorConfig,
        margin_provider: MarginDataProvider,
    ):
        """
        Initialize margin monitor.

        Args:
            config: Monitor configuration
            margin_provider: Margin data provider
        """
        self._config = config
        self._margin_provider = margin_provider

        # Components
        self._level_tracker = MarginLevelTracker(config)
        self._alert_manager = MarginAlertManager(config)

        # State
        self._current_snapshot: Optional[MarginSnapshot] = None
        self._history: Deque[MarginSnapshot] = deque(maxlen=config.history_max_size)
        self._previous_level: Optional[MarginLevel] = None
        self._snapshot_lock = threading.Lock()

        # Background monitoring
        self._monitoring = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(
            f"FuturesMarginMonitor initialized: "
            f"warn={config.warning_ratio}, danger={config.danger_ratio}, "
            f"critical={config.critical_ratio}"
        )

    # ------------------------------------------------------------------------
    # SNAPSHOT CREATION
    # ------------------------------------------------------------------------

    def take_snapshot(self) -> MarginSnapshot:
        """
        Take current margin snapshot.

        Returns:
            MarginSnapshot
        """
        try:
            equity = self._margin_provider.get_account_equity()
            margin_used = self._margin_provider.get_total_margin_used()
            available = self._margin_provider.get_available_margin()
            maintenance = self._margin_provider.get_maintenance_margin()
            positions = self._margin_provider.get_positions()

            # Calculate margin ratio
            if margin_used > 0:
                margin_ratio = equity / margin_used
            else:
                margin_ratio = Decimal("999")  # No margin used

            # Calculate total position value
            total_value = Decimal("0")
            for pos in positions.values():
                if pos.position_value:
                    total_value += abs(pos.position_value)

            # Classify level
            level = self._level_tracker.classify_level(margin_ratio)

            snapshot = MarginSnapshot(
                timestamp_ms=int(time.time() * 1000),
                equity=equity,
                margin_used=margin_used,
                available_margin=available,
                maintenance_margin=maintenance,
                margin_ratio=margin_ratio,
                level=level,
                position_count=len(positions),
                total_position_value=total_value,
            )

            return snapshot

        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
            # Return a safe default snapshot
            return MarginSnapshot(
                timestamp_ms=int(time.time() * 1000),
                equity=Decimal("0"),
                margin_used=Decimal("0"),
                available_margin=Decimal("0"),
                maintenance_margin=Decimal("0"),
                margin_ratio=Decimal("0"),
                level=MarginLevel.LIQUIDATION,
                position_count=0,
                total_position_value=Decimal("0"),
            )

    # ------------------------------------------------------------------------
    # MONITORING
    # ------------------------------------------------------------------------

    def check_once(self) -> Tuple[MarginSnapshot, Optional[MarginAlert]]:
        """
        Perform single monitoring check.

        Returns:
            (snapshot, alert) - alert is None if no alert triggered
        """
        snapshot = self.take_snapshot()
        alert = None

        # Update level and check for transition
        new_level, level_changed = self._level_tracker.update_level(snapshot.margin_ratio)

        # Generate alert if needed
        if level_changed:
            alert = self._create_level_change_alert(snapshot, new_level)

        # Store snapshot
        with self._snapshot_lock:
            self._current_snapshot = snapshot
            self._history.append(snapshot)
            self._previous_level = new_level

        # Send alert
        if alert:
            self._alert_manager.send_alert(alert)

        return snapshot, alert

    def _create_level_change_alert(
        self,
        snapshot: MarginSnapshot,
        new_level: MarginLevel,
    ) -> MarginAlert:
        """Create alert for level change."""
        # Determine severity and type
        if new_level == MarginLevel.LIQUIDATION:
            severity = "CRITICAL"
            alert_type = MarginAlertType.LIQUIDATION_RISK
            message = f"LIQUIDATION RISK: Margin ratio {snapshot.margin_ratio:.2f}"
            action = "Immediately reduce positions or add margin"
        elif new_level == MarginLevel.CRITICAL:
            severity = "ERROR"
            alert_type = MarginAlertType.MARGIN_CALL
            message = f"MARGIN CALL: Margin ratio {snapshot.margin_ratio:.2f}"
            action = "Reduce positions or add margin urgently"
        elif new_level == MarginLevel.DANGER:
            severity = "WARNING"
            alert_type = MarginAlertType.APPROACHING_CRITICAL
            message = f"DANGER: Margin ratio {snapshot.margin_ratio:.2f}"
            action = "Consider reducing positions"
        elif new_level == MarginLevel.WARNING:
            severity = "WARNING"
            alert_type = MarginAlertType.APPROACHING_DANGER
            message = f"WARNING: Margin ratio {snapshot.margin_ratio:.2f}"
            action = "Monitor closely"
        else:
            # Recovered to healthy
            severity = "INFO"
            alert_type = MarginAlertType.RECOVERED
            message = f"RECOVERED: Margin ratio {snapshot.margin_ratio:.2f}"
            action = None

        return MarginAlert(
            timestamp_ms=snapshot.timestamp_ms,
            alert_type=alert_type,
            level=new_level,
            previous_level=self._previous_level,
            margin_ratio=snapshot.margin_ratio,
            equity=snapshot.equity,
            margin_used=snapshot.margin_used,
            message=message,
            recommended_action=action,
            severity=severity,
        )

    # ------------------------------------------------------------------------
    # BACKGROUND MONITORING
    # ------------------------------------------------------------------------

    def start_background_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            logger.warning("Background monitoring already running")
            return

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._background_monitor_loop,
            name="MarginMonitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Background margin monitoring started")

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Background margin monitoring stopped")

    def _background_monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring and not self._stop_event.is_set():
            try:
                self.check_once()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            self._stop_event.wait(self._config.monitor_interval_sec)

    async def start_background_monitoring_async(self) -> None:
        """Start async background monitoring."""
        self._monitoring = True

        while self._monitoring:
            try:
                self.check_once()
            except Exception as e:
                logger.error(f"Error in async monitoring: {e}")

            await asyncio.sleep(self._config.monitor_interval_sec)

    # ------------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------------

    def get_current_snapshot(self) -> Optional[MarginSnapshot]:
        """Get most recent snapshot."""
        with self._snapshot_lock:
            return self._current_snapshot

    def get_current_level(self) -> MarginLevel:
        """Get current margin level."""
        return self._level_tracker.current_level

    def get_history(self, limit: Optional[int] = None) -> List[MarginSnapshot]:
        """Get snapshot history."""
        with self._snapshot_lock:
            history = list(self._history)
        if limit:
            return history[-limit:]
        return history

    def get_margin_ratio(self) -> Optional[Decimal]:
        """Get current margin ratio."""
        snapshot = self.get_current_snapshot()
        return snapshot.margin_ratio if snapshot else None

    def is_healthy(self) -> bool:
        """Check if margin is healthy."""
        return self.get_current_level() == MarginLevel.HEALTHY

    def is_at_risk(self) -> bool:
        """Check if margin is at risk (danger or worse)."""
        return self.get_current_level() in (
            MarginLevel.DANGER,
            MarginLevel.CRITICAL,
            MarginLevel.LIQUIDATION,
        )

    # ------------------------------------------------------------------------
    # REDUCTION SUGGESTIONS
    # ------------------------------------------------------------------------

    def get_reduction_suggestions(
        self,
        positions: Dict[str, FuturesPosition],
        target_ratio: Optional[Decimal] = None,
    ) -> List[ReductionSuggestion]:
        """
        Get position reduction suggestions to reach target margin ratio.

        Args:
            positions: Current positions
            target_ratio: Target margin ratio (default from config)

        Returns:
            List of reduction suggestions
        """
        if not self._config.enable_auto_reduction_suggestions:
            return []

        snapshot = self.get_current_snapshot()
        if not snapshot or snapshot.level == MarginLevel.HEALTHY:
            return []

        target = target_ratio or self._config.target_margin_ratio
        suggestions = []

        # Sort positions by margin impact (larger first)
        sorted_positions = sorted(
            positions.items(),
            key=lambda x: abs(x[1].margin) if x[1].margin else Decimal("0"),
            reverse=True,
        )

        current_ratio = snapshot.margin_ratio
        equity = snapshot.equity
        margin_used = snapshot.margin_used

        if margin_used == 0:
            return []

        # Calculate how much margin needs to be freed
        margin_to_free = margin_used - (equity / target)
        if margin_to_free <= 0:
            return []

        priority = 1
        accumulated_freed = Decimal("0")

        for symbol, position in sorted_positions:
            if position.qty == 0 or not position.margin:
                continue

            if accumulated_freed >= margin_to_free:
                break

            # Calculate reduction needed
            remaining = margin_to_free - accumulated_freed
            position_margin = abs(position.margin)

            if position_margin <= remaining:
                # Close entire position
                reduction_pct = Decimal("100")
                reduction_qty = abs(position.qty)
                freed = position_margin
            else:
                # Partial reduction
                reduction_pct = (remaining / position_margin) * 100
                # Round up to step size
                step = self._config.reduction_step_pct * 100
                reduction_pct = ((reduction_pct // step) + 1) * step
                reduction_pct = min(reduction_pct, Decimal("100"))
                reduction_qty = abs(position.qty) * (reduction_pct / 100)
                freed = position_margin * (reduction_pct / 100)

            accumulated_freed += freed

            reason = self._get_reduction_reason(snapshot.level)

            suggestions.append(
                ReductionSuggestion(
                    symbol=symbol,
                    current_qty=abs(position.qty),
                    suggested_reduction_qty=reduction_qty,
                    suggested_reduction_pct=reduction_pct,
                    reason=reason,
                    priority=priority,
                )
            )

            priority += 1

        return suggestions

    def _get_reduction_reason(self, level: MarginLevel) -> str:
        """Get reason for reduction based on level."""
        reasons = {
            MarginLevel.LIQUIDATION: "Immediate liquidation risk",
            MarginLevel.CRITICAL: "Margin call triggered",
            MarginLevel.DANGER: "Approaching margin call",
            MarginLevel.WARNING: "Margin utilization too high",
        }
        return reasons.get(level, "Margin management")

    # ------------------------------------------------------------------------
    # ALERT MANAGEMENT
    # ------------------------------------------------------------------------

    def register_alert_callback(self, callback: AlertCallback) -> None:
        """Register alert callback."""
        self._alert_manager.add_callback(callback)

    def get_alert_history(self, limit: int = 50) -> List[MarginAlert]:
        """Get alert history."""
        return self._alert_manager.get_alert_history(limit)

    # ------------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------------

    def get_statistics(self, lookback_snapshots: int = 100) -> Dict[str, Any]:
        """
        Get margin statistics over lookback period.

        Args:
            lookback_snapshots: Number of snapshots to analyze

        Returns:
            Statistics dictionary
        """
        history = self.get_history(lookback_snapshots)
        if not history:
            return {}

        ratios = [float(s.margin_ratio) for s in history]
        levels = [s.level for s in history]

        import statistics

        level_counts = {}
        for level in MarginLevel:
            level_counts[level.name] = sum(1 for l in levels if l == level)

        return {
            "snapshot_count": len(history),
            "time_range_ms": (
                history[-1].timestamp_ms - history[0].timestamp_ms if len(history) > 1 else 0
            ),
            "current_ratio": ratios[-1] if ratios else 0,
            "min_ratio": min(ratios) if ratios else 0,
            "max_ratio": max(ratios) if ratios else 0,
            "mean_ratio": statistics.mean(ratios) if ratios else 0,
            "std_ratio": statistics.stdev(ratios) if len(ratios) > 1 else 0,
            "level_distribution": level_counts,
            "time_at_risk_pct": (
                (
                    level_counts.get("DANGER", 0)
                    + level_counts.get("CRITICAL", 0)
                    + level_counts.get("LIQUIDATION", 0)
                )
                / len(levels)
                * 100
                if levels
                else 0
            ),
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_margin_monitor(
    margin_provider: MarginDataProvider,
    config: Optional[MarginMonitorConfig] = None,
    alert_callbacks: Optional[List[AlertCallback]] = None,
) -> FuturesMarginMonitor:
    """
    Create margin monitor.

    Args:
        margin_provider: Margin data provider
        config: Monitor configuration
        alert_callbacks: Alert callbacks

    Returns:
        FuturesMarginMonitor instance
    """
    if config is None:
        config = MarginMonitorConfig()

    monitor = FuturesMarginMonitor(config, margin_provider)

    if alert_callbacks:
        for callback in alert_callbacks:
            monitor.register_alert_callback(callback)

    return monitor


def create_crypto_margin_monitor(
    margin_provider: MarginDataProvider,
    alert_callbacks: Optional[List[AlertCallback]] = None,
    **kwargs,
) -> FuturesMarginMonitor:
    """Create margin monitor for crypto futures."""
    config = MarginMonitorConfig.for_crypto(**kwargs)
    return create_margin_monitor(margin_provider, config, alert_callbacks)


def create_cme_margin_monitor(
    margin_provider: MarginDataProvider,
    alert_callbacks: Optional[List[AlertCallback]] = None,
    **kwargs,
) -> FuturesMarginMonitor:
    """Create margin monitor for CME futures."""
    config = MarginMonitorConfig.for_cme(**kwargs)
    return create_margin_monitor(margin_provider, config, alert_callbacks)
