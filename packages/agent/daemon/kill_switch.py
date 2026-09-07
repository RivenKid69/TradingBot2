# -*- coding: utf-8 -*-
"""
Kill Switch Manager - Enhanced kill switch with halt reasons.

Design Doc 9.4: Kill Switch + halt reasons
- Triggers: max daily loss, broker errors burst, latency spike, order spam, state divergence, data feed invalid
- Actions: cancel open orders -> optional flatten -> halt run; report reason in telemetry

CCEA Phase 5 Component.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional
from uuid import uuid4


# Constants
MAX_HALT_HISTORY: Final[int] = 1000


class HaltReasonType(Enum):
    """
    Types of halt reasons.

    Design Doc 9.4: Categorized halt triggers.
    """

    MAX_DAILY_LOSS = auto()
    MAX_DRAWDOWN = auto()
    BROKER_ERROR_BURST = auto()
    LATENCY_SPIKE = auto()
    ORDER_SPAM = auto()
    STATE_DIVERGENCE = auto()
    DATA_FEED_INVALID = auto()
    POSITION_MISMATCH = auto()
    HARD_CAP_EXCEEDED = auto()
    MANUAL_TRIGGER = auto()
    NETWORK_DOWN = auto()
    CLOUD_UNREACHABLE = auto()
    TIME_SYNC_DRIFT = auto()
    SIGNATURE_INVALID = auto()
    POLICY_VIOLATION = auto()
    RECONCILIATION_FAILED = auto()
    PREFLIGHT_FAILED = auto()


class HaltSeverity(Enum):
    """Severity levels for halt events."""

    CRITICAL = "critical"  # Immediate halt, cancel all orders
    HIGH = "high"  # Halt after current orders complete
    MEDIUM = "medium"  # Pause, require manual intervention
    LOW = "low"  # Warning, continue but alert


class HaltAction(Enum):
    """Actions to take on halt."""

    HALT_ONLY = "halt_only"  # Stop trading, keep positions
    CANCEL_ORDERS = "cancel_orders"  # Cancel open orders, keep positions
    FLATTEN_LOCAL = "flatten_local"  # Flatten if local policy allows
    FLATTEN_EMERGENCY = "flatten_emergency"  # Emergency flatten (enterprise only)


@dataclass
class HaltReason:
    """
    Detailed halt reason with context.

    Contains all information about why a halt occurred.
    """

    reason_id: str = field(default_factory=lambda: str(uuid4()))
    reason_type: HaltReasonType = HaltReasonType.MANUAL_TRIGGER
    severity: HaltSeverity = HaltSeverity.CRITICAL
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trigger_source: str = ""  # Component that triggered
    recovery_action: Optional[str] = None
    acknowledgment_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reason_id": self.reason_id,
            "reason_type": self.reason_type.name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "trigger_source": self.trigger_source,
            "recovery_action": self.recovery_action,
            "acknowledgment_required": self.acknowledgment_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HaltReason:
        """Create from dictionary."""
        return cls(
            reason_id=data["reason_id"],
            reason_type=HaltReasonType[data["reason_type"]],
            severity=HaltSeverity(data["severity"]),
            message=data["message"],
            details=data.get("details", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            trigger_source=data.get("trigger_source", ""),
            recovery_action=data.get("recovery_action"),
            acknowledgment_required=data.get("acknowledgment_required", True),
        )

    def get_telemetry_safe_details(self) -> Dict[str, Any]:
        """
        Get details safe for telemetry (redacted).

        Removes any potentially sensitive information.
        """
        safe = {}
        for key, value in self.details.items():
            # Redact sensitive keys
            if any(s in key.lower() for s in ["key", "secret", "password", "token", "credential"]):
                safe[key] = "[REDACTED]"
            elif isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif isinstance(value, Decimal):
                safe[key] = str(value)
            else:
                safe[key] = str(type(value).__name__)
        return safe


@dataclass
class HaltEvent:
    """
    Record of a halt event.

    Stored for audit and evidence purposes.
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    halt_reason: HaltReason = field(default_factory=HaltReason)
    action_taken: HaltAction = HaltAction.HALT_ONLY
    orders_cancelled: int = 0
    positions_flattened: bool = False
    recovery_timestamp: Optional[datetime] = None
    recovery_approval_code: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    evidence_hash: str = ""

    def __post_init__(self):
        """Compute evidence hash."""
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash for tamper evidence."""
        data = {
            "event_id": self.event_id,
            "reason": self.halt_reason.to_dict(),
            "action": self.action_taken.value,
            "orders_cancelled": self.orders_cancelled,
            "positions_flattened": self.positions_flattened,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "halt_reason": self.halt_reason.to_dict(),
            "action_taken": self.action_taken.value,
            "orders_cancelled": self.orders_cancelled,
            "positions_flattened": self.positions_flattened,
            "recovery_timestamp": (
                self.recovery_timestamp.isoformat() if self.recovery_timestamp else None
            ),
            "recovery_approval_code": self.recovery_approval_code,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class KillSwitchConfig:
    """
    Kill switch configuration.

    Defines thresholds and behaviors.
    """

    # Loss thresholds
    max_daily_loss_pct: Decimal = Decimal("0.30")  # 30%
    max_drawdown_pct: Decimal = Decimal("0.50")  # 50%

    # Error thresholds
    max_broker_errors_per_minute: int = 5
    max_broker_errors_per_hour: int = 20
    max_consecutive_errors: int = 3

    # Latency thresholds
    max_latency_ms: int = 5000  # 5 seconds
    latency_spike_window_seconds: int = 60
    max_latency_spikes_in_window: int = 3

    # Order spam thresholds
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100

    # State divergence
    position_mismatch_tolerance_pct: Decimal = Decimal("0.01")  # 1%

    # Data feed
    max_data_gap_seconds: int = 30
    stale_price_threshold_seconds: int = 60

    # Time sync
    max_time_drift_seconds: int = 5

    # Recovery
    cooldown_seconds: int = 300  # 5 minutes
    require_manual_reset: bool = True
    allow_flatten_on_halt: bool = False  # Only local flatten, not remote

    # Storage
    history_file: Path = field(default_factory=lambda: Path.home() / ".ccea" / "halt_history.json")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_daily_loss_pct": str(self.max_daily_loss_pct),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "max_broker_errors_per_minute": self.max_broker_errors_per_minute,
            "max_broker_errors_per_hour": self.max_broker_errors_per_hour,
            "max_consecutive_errors": self.max_consecutive_errors,
            "max_latency_ms": self.max_latency_ms,
            "max_orders_per_second": self.max_orders_per_second,
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_data_gap_seconds": self.max_data_gap_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "require_manual_reset": self.require_manual_reset,
            "allow_flatten_on_halt": self.allow_flatten_on_halt,
        }


class KillSwitchManager:
    """
    Manages kill switch logic and state.

    Design Doc 9.4:
    - Monitors all halt trigger conditions
    - Executes halt actions (cancel orders, optionally flatten)
    - Records halt events for audit
    - Requires manual approval for recovery

    Usage:
        manager = KillSwitchManager()

        # Check conditions
        if manager.check_daily_loss(pnl, equity):
            manager.trigger(reason, action=HaltAction.CANCEL_ORDERS)

        # Recovery
        if manager.acknowledge("operator", "APPROVE_RESET_12345"):
            manager.reset()
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        on_trigger: Optional[Callable[[HaltReason], None]] = None,
        cancel_orders_fn: Optional[Callable[[], int]] = None,
        flatten_fn: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize kill switch manager.

        Args:
            config: Kill switch configuration
            on_trigger: Callback when kill switch triggers
            cancel_orders_fn: Function to cancel open orders
            flatten_fn: Function to flatten positions (local only)
        """
        self.config = config or KillSwitchConfig()
        self._on_trigger = on_trigger
        self._cancel_orders_fn = cancel_orders_fn
        self._flatten_fn = flatten_fn

        # State
        self._triggered = False
        self._current_halt: Optional[HaltEvent] = None
        self._halt_history: List[HaltEvent] = []
        self._last_trigger_time: Optional[datetime] = None

        # Tracking
        self._errors: List[datetime] = []
        self._latency_spikes: List[datetime] = []
        self._orders: List[datetime] = []
        self._last_data_time: Optional[datetime] = None

        # Thread safety
        self._lock = threading.RLock()

        # Load history
        self._load_history()

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self._triggered

    @property
    def current_halt(self) -> Optional[HaltEvent]:
        """Get current halt event."""
        return self._current_halt

    @property
    def halt_history(self) -> List[HaltEvent]:
        """Get halt history."""
        return list(self._halt_history)

    def trigger(
        self,
        reason: HaltReason,
        action: HaltAction = HaltAction.CANCEL_ORDERS,
    ) -> HaltEvent:
        """
        Trigger kill switch.

        Args:
            reason: Halt reason
            action: Action to take

        Returns:
            Halt event record
        """
        with self._lock:
            if self._triggered:
                # Already triggered, update reason if more severe
                if reason.severity.value < self._current_halt.halt_reason.severity.value:
                    self._current_halt.halt_reason = reason
                return self._current_halt

            self._triggered = True
            self._last_trigger_time = datetime.utcnow()

            # Execute action
            orders_cancelled = 0
            positions_flattened = False

            if action in (
                HaltAction.CANCEL_ORDERS,
                HaltAction.FLATTEN_LOCAL,
                HaltAction.FLATTEN_EMERGENCY,
            ):
                if self._cancel_orders_fn:
                    try:
                        orders_cancelled = self._cancel_orders_fn()
                    except Exception:
                        pass  # Log but continue

            if action in (HaltAction.FLATTEN_LOCAL, HaltAction.FLATTEN_EMERGENCY):
                if self.config.allow_flatten_on_halt and self._flatten_fn:
                    try:
                        positions_flattened = self._flatten_fn()
                    except Exception:
                        pass  # Log but continue

            # Create event
            event = HaltEvent(
                halt_reason=reason,
                action_taken=action,
                orders_cancelled=orders_cancelled,
                positions_flattened=positions_flattened,
            )

            self._current_halt = event
            self._halt_history.append(event)

            # Trim history
            if len(self._halt_history) > MAX_HALT_HISTORY:
                self._halt_history = self._halt_history[-MAX_HALT_HISTORY:]

            # Save history
            self._save_history()

            # Callback
            if self._on_trigger:
                try:
                    self._on_trigger(reason)
                except Exception:
                    pass

            return event

    def acknowledge(
        self,
        acknowledged_by: str,
        approval_code: str,
    ) -> bool:
        """
        Acknowledge halt event.

        Args:
            acknowledged_by: Who acknowledged
            approval_code: Approval code

        Returns:
            True if acknowledged
        """
        with self._lock:
            if not self._triggered or not self._current_halt:
                return False

            # Validate approval code format
            if not approval_code.startswith("APPROVE_RESET_"):
                return False

            self._current_halt.acknowledged_by = acknowledged_by
            self._current_halt.acknowledged_at = datetime.utcnow()
            self._current_halt.recovery_approval_code = approval_code

            self._save_history()
            return True

    def reset(self) -> bool:
        """
        Reset kill switch after acknowledgment.

        Returns:
            True if reset successful
        """
        with self._lock:
            if not self._triggered:
                return True

            if self._current_halt and not self._current_halt.acknowledged_by:
                if self.config.require_manual_reset:
                    return False

            # Check cooldown
            if self._last_trigger_time:
                cooldown_end = self._last_trigger_time + timedelta(
                    seconds=self.config.cooldown_seconds
                )
                if datetime.utcnow() < cooldown_end:
                    return False

            # Record recovery
            if self._current_halt:
                self._current_halt.recovery_timestamp = datetime.utcnow()
                self._save_history()

            # Reset state
            self._triggered = False
            self._current_halt = None
            self._errors.clear()
            self._latency_spikes.clear()

            return True

    # ===== Check Methods =====

    def check_daily_loss(
        self,
        daily_pnl: Decimal,
        equity: Decimal,
    ) -> Optional[HaltReason]:
        """
        Check if daily loss exceeds threshold.

        Args:
            daily_pnl: Current daily P&L
            equity: Current equity

        Returns:
            HaltReason if threshold exceeded
        """
        if equity <= 0:
            return None

        loss_pct = abs(daily_pnl / equity) if daily_pnl < 0 else Decimal("0")

        if loss_pct >= self.config.max_daily_loss_pct:
            return HaltReason(
                reason_type=HaltReasonType.MAX_DAILY_LOSS,
                severity=HaltSeverity.CRITICAL,
                message=f"Daily loss {loss_pct:.2%} exceeds maximum {self.config.max_daily_loss_pct:.2%}",
                details={
                    "daily_pnl": str(daily_pnl),
                    "equity": str(equity),
                    "loss_pct": str(loss_pct),
                    "threshold": str(self.config.max_daily_loss_pct),
                },
                trigger_source="KillSwitchManager.check_daily_loss",
            )
        return None

    def check_drawdown(
        self,
        current_equity: Decimal,
        peak_equity: Decimal,
    ) -> Optional[HaltReason]:
        """
        Check if drawdown exceeds threshold.

        Args:
            current_equity: Current equity
            peak_equity: Peak equity

        Returns:
            HaltReason if threshold exceeded
        """
        if peak_equity <= 0:
            return None

        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown >= self.config.max_drawdown_pct:
            return HaltReason(
                reason_type=HaltReasonType.MAX_DRAWDOWN,
                severity=HaltSeverity.CRITICAL,
                message=f"Drawdown {drawdown:.2%} exceeds maximum {self.config.max_drawdown_pct:.2%}",
                details={
                    "current_equity": str(current_equity),
                    "peak_equity": str(peak_equity),
                    "drawdown": str(drawdown),
                    "threshold": str(self.config.max_drawdown_pct),
                },
                trigger_source="KillSwitchManager.check_drawdown",
            )
        return None

    def record_error(self) -> Optional[HaltReason]:
        """
        Record an error and check thresholds.

        Returns:
            HaltReason if threshold exceeded
        """
        with self._lock:
            now = datetime.utcnow()
            self._errors.append(now)

            # Clean old errors
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            self._errors = [e for e in self._errors if e > cutoff_hour]

            # Check per-minute
            errors_per_minute = sum(1 for e in self._errors if e > cutoff_minute)
            if errors_per_minute >= self.config.max_broker_errors_per_minute:
                return HaltReason(
                    reason_type=HaltReasonType.BROKER_ERROR_BURST,
                    severity=HaltSeverity.CRITICAL,
                    message=f"Broker errors ({errors_per_minute}/min) exceeds maximum ({self.config.max_broker_errors_per_minute}/min)",
                    details={
                        "errors_per_minute": errors_per_minute,
                        "threshold": self.config.max_broker_errors_per_minute,
                    },
                    trigger_source="KillSwitchManager.record_error",
                )

            # Check per-hour
            errors_per_hour = len(self._errors)
            if errors_per_hour >= self.config.max_broker_errors_per_hour:
                return HaltReason(
                    reason_type=HaltReasonType.BROKER_ERROR_BURST,
                    severity=HaltSeverity.HIGH,
                    message=f"Broker errors ({errors_per_hour}/hr) exceeds maximum ({self.config.max_broker_errors_per_hour}/hr)",
                    details={
                        "errors_per_hour": errors_per_hour,
                        "threshold": self.config.max_broker_errors_per_hour,
                    },
                    trigger_source="KillSwitchManager.record_error",
                )

            return None

    def record_latency(self, latency_ms: int) -> Optional[HaltReason]:
        """
        Record latency and check for spikes.

        Args:
            latency_ms: Latency in milliseconds

        Returns:
            HaltReason if threshold exceeded
        """
        with self._lock:
            now = datetime.utcnow()

            if latency_ms >= self.config.max_latency_ms:
                self._latency_spikes.append(now)

                # Clean old spikes
                cutoff = now - timedelta(seconds=self.config.latency_spike_window_seconds)
                self._latency_spikes = [s for s in self._latency_spikes if s > cutoff]

                if len(self._latency_spikes) >= self.config.max_latency_spikes_in_window:
                    return HaltReason(
                        reason_type=HaltReasonType.LATENCY_SPIKE,
                        severity=HaltSeverity.HIGH,
                        message=f"Latency spikes ({len(self._latency_spikes)}) in {self.config.latency_spike_window_seconds}s exceeds maximum",
                        details={
                            "latency_ms": latency_ms,
                            "spikes_count": len(self._latency_spikes),
                            "threshold_ms": self.config.max_latency_ms,
                            "window_seconds": self.config.latency_spike_window_seconds,
                        },
                        trigger_source="KillSwitchManager.record_latency",
                    )

            return None

    def record_order(self) -> Optional[HaltReason]:
        """
        Record order and check for spam.

        Returns:
            HaltReason if threshold exceeded
        """
        with self._lock:
            now = datetime.utcnow()
            self._orders.append(now)

            # Clean old orders
            cutoff_second = now - timedelta(seconds=1)
            cutoff_minute = now - timedelta(minutes=1)
            self._orders = [o for o in self._orders if o > cutoff_minute]

            # Check per-second
            orders_per_second = sum(1 for o in self._orders if o > cutoff_second)
            if orders_per_second > self.config.max_orders_per_second:
                return HaltReason(
                    reason_type=HaltReasonType.ORDER_SPAM,
                    severity=HaltSeverity.HIGH,
                    message=f"Orders per second ({orders_per_second}) exceeds maximum ({self.config.max_orders_per_second})",
                    details={
                        "orders_per_second": orders_per_second,
                        "threshold": self.config.max_orders_per_second,
                    },
                    trigger_source="KillSwitchManager.record_order",
                )

            # Check per-minute
            orders_per_minute = len(self._orders)
            if orders_per_minute > self.config.max_orders_per_minute:
                return HaltReason(
                    reason_type=HaltReasonType.ORDER_SPAM,
                    severity=HaltSeverity.MEDIUM,
                    message=f"Orders per minute ({orders_per_minute}) exceeds maximum ({self.config.max_orders_per_minute})",
                    details={
                        "orders_per_minute": orders_per_minute,
                        "threshold": self.config.max_orders_per_minute,
                    },
                    trigger_source="KillSwitchManager.record_order",
                )

            return None

    def check_data_feed(self, last_data_time: datetime) -> Optional[HaltReason]:
        """
        Check if data feed is stale.

        Args:
            last_data_time: Time of last data

        Returns:
            HaltReason if data is stale
        """
        self._last_data_time = last_data_time
        gap = (datetime.utcnow() - last_data_time).total_seconds()

        if gap > self.config.max_data_gap_seconds:
            return HaltReason(
                reason_type=HaltReasonType.DATA_FEED_INVALID,
                severity=HaltSeverity.HIGH,
                message=f"Data feed gap ({gap:.1f}s) exceeds maximum ({self.config.max_data_gap_seconds}s)",
                details={
                    "gap_seconds": gap,
                    "threshold": self.config.max_data_gap_seconds,
                    "last_data_time": last_data_time.isoformat(),
                },
                trigger_source="KillSwitchManager.check_data_feed",
            )

        return None

    def check_position_mismatch(
        self,
        local_position: Decimal,
        broker_position: Decimal,
        symbol: str,
    ) -> Optional[HaltReason]:
        """
        Check for position mismatch.

        Args:
            local_position: Local position
            broker_position: Broker position
            symbol: Symbol

        Returns:
            HaltReason if mismatch exceeds tolerance
        """
        if local_position == 0 and broker_position == 0:
            return None

        base = max(abs(local_position), abs(broker_position), Decimal("1"))
        diff_pct = abs(local_position - broker_position) / base

        if diff_pct > self.config.position_mismatch_tolerance_pct:
            return HaltReason(
                reason_type=HaltReasonType.POSITION_MISMATCH,
                severity=HaltSeverity.HIGH,
                message=f"Position mismatch for {symbol}: local={local_position}, broker={broker_position}",
                details={
                    "symbol": symbol,
                    "local_position": str(local_position),
                    "broker_position": str(broker_position),
                    "diff_pct": str(diff_pct),
                    "tolerance": str(self.config.position_mismatch_tolerance_pct),
                },
                trigger_source="KillSwitchManager.check_position_mismatch",
            )

        return None

    # ===== Factory Methods =====

    @staticmethod
    def create_halt_reason(
        reason_type: HaltReasonType,
        message: str,
        severity: HaltSeverity = HaltSeverity.CRITICAL,
        details: Optional[Dict[str, Any]] = None,
        trigger_source: str = "external",
    ) -> HaltReason:
        """
        Factory method to create halt reason.

        Args:
            reason_type: Type of reason
            message: Human readable message
            severity: Severity level
            details: Additional details
            trigger_source: Source component

        Returns:
            HaltReason instance
        """
        return HaltReason(
            reason_type=reason_type,
            severity=severity,
            message=message,
            details=details or {},
            trigger_source=trigger_source,
        )

    # ===== Persistence =====

    def _load_history(self) -> None:
        """Load halt history from disk."""
        if not self.config.history_file.exists():
            return

        try:
            with open(self.config.history_file, "r") as f:
                data = json.load(f)

            for item in data.get("history", []):
                reason_data = item.get("halt_reason", {})
                reason = HaltReason.from_dict(reason_data)

                event = HaltEvent(
                    event_id=item["event_id"],
                    halt_reason=reason,
                    action_taken=HaltAction(item["action_taken"]),
                    orders_cancelled=item.get("orders_cancelled", 0),
                    positions_flattened=item.get("positions_flattened", False),
                    evidence_hash=item.get("evidence_hash", ""),
                )

                if item.get("recovery_timestamp"):
                    event.recovery_timestamp = datetime.fromisoformat(item["recovery_timestamp"])
                if item.get("acknowledged_by"):
                    event.acknowledged_by = item["acknowledged_by"]
                if item.get("acknowledged_at"):
                    event.acknowledged_at = datetime.fromisoformat(item["acknowledged_at"])

                self._halt_history.append(event)

        except Exception:
            pass  # Start fresh on error

    def _save_history(self) -> None:
        """Save halt history to disk."""
        try:
            self.config.history_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0.0",
                "updated_at": datetime.utcnow().isoformat(),
                "history": [e.to_dict() for e in self._halt_history[-MAX_HALT_HISTORY:]],
            }

            with open(self.config.history_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception:
            pass  # Log but continue

    def export_for_evidence(self) -> Dict[str, Any]:
        """
        Export halt data for evidence pack.

        Returns:
            Evidence-ready data structure
        """
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "config": self.config.to_dict(),
            "current_triggered": self._triggered,
            "current_halt": self._current_halt.to_dict() if self._current_halt else None,
            "history_count": len(self._halt_history),
            "history": [e.to_dict() for e in self._halt_history],
        }
