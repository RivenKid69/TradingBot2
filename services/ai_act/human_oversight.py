# -*- coding: utf-8 -*-
"""
AI Act Human Oversight System Implementation.

EU AI Act Article 14 - Human Oversight for High-Risk AI Systems.

This module implements human oversight capabilities required by the EU AI Act,
including stop/pause functions, manual override, anomaly detection, and
automation bias monitoring.

Article 14 Requirements:
    (1) High-risk AI systems shall be designed and developed in such a way,
        including with appropriate human-machine interface tools, that they
        can be effectively overseen by natural persons during the period in
        which the AI system is in use.

    (2) Human oversight shall aim to prevent or minimise the risks to health,
        safety or fundamental rights that may emerge when a high-risk AI system
        is used in accordance with its intended purpose or under conditions of
        reasonably foreseeable misuse.

References:
    - EU AI Act Article 14: Human Oversight
    - NIST AI RMF: Human-AI Interaction Guidelines
    - ISO/IEC 22989:2022 AI Concepts and Terminology

Example:
    >>> from services.ai_act.human_oversight import (
    ...     HumanOversightSystem,
    ...     create_human_oversight_system,
    ...     OversightLevel,
    ... )
    >>> oversight = create_human_oversight_system()
    >>> oversight.emergency_stop(reason="Market anomaly detected")
    >>> oversight.get_status()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque


logger = logging.getLogger(__name__)


class OversightLevel(IntEnum):
    """
    Human oversight capability levels per Article 14.

    Higher levels provide more granular control over AI operations.
    """

    L1_STOP = 1
    """Immediate halt of all AI operations (emergency stop)."""

    L2_PAUSE = 2
    """Temporary suspension with state preservation."""

    L3_OVERRIDE = 3
    """Manual override of specific AI decisions."""

    L4_AUDIT = 4
    """Complete decision audit and review capability."""


class OversightCapability(Enum):
    """Specific oversight capabilities available to operators."""

    EMERGENCY_STOP = "emergency_stop"
    """Immediately halt all AI trading operations."""

    TRADING_PAUSE = "trading_pause"
    """Temporarily pause trading while preserving state."""

    POSITION_OVERRIDE = "position_override"
    """Manually override AI position recommendations."""

    SIGNAL_VETO = "signal_veto"
    """Veto specific AI trading signals."""

    RISK_LIMIT_ADJUST = "risk_limit_adjust"
    """Dynamically adjust risk limits."""

    MODEL_DISABLE = "model_disable"
    """Disable specific AI model components."""

    FULL_AUDIT = "full_audit"
    """Access complete decision audit trail."""


class SystemState(Enum):
    """Current state of the AI trading system."""

    ACTIVE = "active"
    """System is fully operational."""

    PAUSED = "paused"
    """System is paused but state preserved."""

    STOPPED = "stopped"
    """System is stopped (emergency stop activated)."""

    OVERRIDE = "override"
    """System is under manual override."""

    MAINTENANCE = "maintenance"
    """System is in maintenance mode."""


class AlertSeverity(Enum):
    """Severity levels for anomaly alerts."""

    INFO = "info"
    """Informational alert, no action required."""

    WARNING = "warning"
    """Warning alert, operator attention recommended."""

    CRITICAL = "critical"
    """Critical alert, immediate operator action required."""

    EMERGENCY = "emergency"
    """Emergency alert, automatic protective actions triggered."""


@dataclass
class OversightAction:
    """
    Record of a human oversight action.

    Attributes:
        action_id: Unique identifier for the action.
        capability: Type of oversight capability exercised.
        operator: Identifier of the human operator.
        timestamp: When the action was taken.
        reason: Reason for the action.
        details: Additional action details.
        system_state_before: System state before action.
        system_state_after: System state after action.
    """

    action_id: str
    capability: OversightCapability
    operator: str
    timestamp: datetime
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    system_state_before: SystemState = SystemState.ACTIVE
    system_state_after: SystemState = SystemState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "capability": self.capability.value,
            "operator": self.operator,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "details": self.details,
            "system_state_before": self.system_state_before.value,
            "system_state_after": self.system_state_after.value,
        }


@dataclass
class AnomalyAlert:
    """
    Anomaly detection alert for operator notification.

    Attributes:
        alert_id: Unique identifier.
        severity: Alert severity level.
        source: Source component that generated the alert.
        message: Human-readable alert message.
        timestamp: When the anomaly was detected.
        metrics: Relevant metrics at time of anomaly.
        recommended_action: Suggested operator action.
        acknowledged: Whether operator has acknowledged.
        acknowledged_by: Operator who acknowledged.
        acknowledged_at: When acknowledged.
    """

    alert_id: str
    severity: AlertSeverity
    source: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def acknowledge(self, operator: str) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = operator
        self.acknowledged_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "recommended_action": self.recommended_action,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class HumanOversightConfig:
    """
    Configuration for Human Oversight System.

    Attributes:
        enabled_capabilities: List of enabled oversight capabilities.
        anomaly_detection_enabled: Whether anomaly detection is active.
        automation_bias_monitoring: Whether to monitor for automation bias.
        alert_cooldown_seconds: Minimum time between similar alerts.
        max_alerts_per_hour: Maximum alerts per hour (spam prevention).
        require_acknowledgment: Whether alerts require acknowledgment.
        auto_pause_on_critical: Automatically pause on critical alerts.
        auto_stop_on_emergency: Automatically stop on emergency alerts.
        audit_retention_days: Days to retain audit records.
        state_persistence_path: Path for state persistence.
    """

    enabled_capabilities: List[OversightCapability] = field(
        default_factory=lambda: list(OversightCapability)
    )
    anomaly_detection_enabled: bool = True
    automation_bias_monitoring: bool = True
    alert_cooldown_seconds: float = 60.0
    max_alerts_per_hour: int = 100
    require_acknowledgment: bool = True
    auto_pause_on_critical: bool = True
    auto_stop_on_emergency: bool = True
    audit_retention_days: int = 365
    state_persistence_path: Optional[Path] = None


class AnomalyDetector:
    """
    Anomaly detection for human oversight alerting.

    Monitors system metrics for anomalous behavior that requires
    human operator attention.

    Attributes:
        config: Oversight configuration.
    """

    def __init__(self, config: HumanOversightConfig) -> None:
        """
        Initialize AnomalyDetector.

        Args:
            config: Oversight configuration.
        """
        self._config = config
        self._lock = threading.Lock()
        self._metric_history: Dict[str, deque] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._last_alert_time: Dict[str, float] = {}

        # Anomaly thresholds
        self._thresholds = {
            "pnl_drawdown_pct": {"warning": 0.05, "critical": 0.10, "emergency": 0.20},
            "position_deviation_pct": {"warning": 0.20, "critical": 0.50, "emergency": 0.80},
            "execution_latency_ms": {"warning": 1000, "critical": 5000, "emergency": 10000},
            "signal_confidence": {"warning": 0.3, "critical": 0.2, "emergency": 0.1},
            "market_volatility_zscore": {"warning": 2.0, "critical": 3.0, "emergency": 4.0},
            "model_uncertainty": {"warning": 0.5, "critical": 0.7, "emergency": 0.9},
        }

    def configure_threshold(
        self,
        metric: str,
        warning: float,
        critical: float,
        emergency: float,
    ) -> None:
        """Configure anomaly thresholds for a metric."""
        self._thresholds[metric] = {
            "warning": warning,
            "critical": critical,
            "emergency": emergency,
        }

    def record_metric(self, metric: str, value: float) -> Optional[AnomalyAlert]:
        """
        Record a metric value and check for anomalies.

        Args:
            metric: Metric name.
            value: Metric value.

        Returns:
            AnomalyAlert if anomaly detected, None otherwise.
        """
        with self._lock:
            # Store in history
            if metric not in self._metric_history:
                self._metric_history[metric] = deque(maxlen=1000)
            self._metric_history[metric].append(
                {
                    "value": value,
                    "timestamp": time.time(),
                }
            )

            # Check thresholds
            if metric not in self._thresholds:
                return None

            thresholds = self._thresholds[metric]
            severity = None
            threshold_value = None

            # Determine severity (check from highest to lowest)
            if value >= thresholds["emergency"]:
                severity = AlertSeverity.EMERGENCY
                threshold_value = thresholds["emergency"]
            elif value >= thresholds["critical"]:
                severity = AlertSeverity.CRITICAL
                threshold_value = thresholds["critical"]
            elif value >= thresholds["warning"]:
                severity = AlertSeverity.WARNING
                threshold_value = thresholds["warning"]

            if severity is None:
                return None

            # Check cooldown
            alert_key = f"{metric}_{severity.value}"
            last_time = self._last_alert_time.get(alert_key, 0)
            if time.time() - last_time < self._config.alert_cooldown_seconds:
                return None

            # Check rate limit
            recent_alerts = sum(
                1 for a in self._alert_history if time.time() - a["timestamp"] < 3600
            )
            if recent_alerts >= self._config.max_alerts_per_hour:
                return None

            # Create alert
            alert = AnomalyAlert(
                alert_id=f"ANM-{int(time.time() * 1000)}",
                severity=severity,
                source="anomaly_detector",
                message=f"Metric {metric} exceeded {severity.value} threshold: {value:.4f} >= {threshold_value:.4f}",
                timestamp=datetime.now(timezone.utc),
                metrics={metric: value, "threshold": threshold_value},
                recommended_action=self._get_recommended_action(metric, severity),
            )

            self._alert_history.append(
                {
                    "alert_id": alert.alert_id,
                    "timestamp": time.time(),
                }
            )
            self._last_alert_time[alert_key] = time.time()

            logger.warning(f"Anomaly detected: {alert.message}")
            return alert

    def _get_recommended_action(
        self,
        metric: str,
        severity: AlertSeverity,
    ) -> str:
        """Get recommended action for an anomaly."""
        actions = {
            "pnl_drawdown_pct": {
                AlertSeverity.WARNING: "Monitor position sizes and consider reducing exposure",
                AlertSeverity.CRITICAL: "Reduce position sizes immediately",
                AlertSeverity.EMERGENCY: "Execute emergency stop and review strategy",
            },
            "position_deviation_pct": {
                AlertSeverity.WARNING: "Review position sync and execution quality",
                AlertSeverity.CRITICAL: "Halt new trades until deviation resolved",
                AlertSeverity.EMERGENCY: "Emergency stop and manual position reconciliation",
            },
            "execution_latency_ms": {
                AlertSeverity.WARNING: "Monitor network connectivity",
                AlertSeverity.CRITICAL: "Pause trading and investigate latency source",
                AlertSeverity.EMERGENCY: "Emergency stop due to execution risk",
            },
            "signal_confidence": {
                AlertSeverity.WARNING: "Review model predictions",
                AlertSeverity.CRITICAL: "Reduce position sizes due to low confidence",
                AlertSeverity.EMERGENCY: "Disable model and switch to safe mode",
            },
            "market_volatility_zscore": {
                AlertSeverity.WARNING: "Increase monitoring frequency",
                AlertSeverity.CRITICAL: "Reduce exposure and tighten risk limits",
                AlertSeverity.EMERGENCY: "Execute volatility protection protocol",
            },
            "model_uncertainty": {
                AlertSeverity.WARNING: "Log predictions for review",
                AlertSeverity.CRITICAL: "Apply uncertainty-based position scaling",
                AlertSeverity.EMERGENCY: "Disable AI recommendations",
            },
        }

        metric_actions = actions.get(metric, {})
        return metric_actions.get(
            severity,
            f"Review {metric} and take appropriate action based on {severity.value} severity",
        )


class ManualOverrideController:
    """
    Controller for manual override of AI decisions.

    Allows human operators to override specific AI trading decisions
    while maintaining audit trail.

    Attributes:
        active_overrides: Currently active overrides.
    """

    def __init__(self) -> None:
        """Initialize ManualOverrideController."""
        self._lock = threading.Lock()
        self._active_overrides: Dict[str, Dict[str, Any]] = {}
        self._override_history: List[Dict[str, Any]] = []

    def set_position_override(
        self,
        symbol: str,
        target_position: float,
        operator: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> str:
        """
        Set a manual position override.

        Args:
            symbol: Trading symbol.
            target_position: Target position (fraction of max).
            operator: Operator identifier.
            reason: Reason for override.
            duration_minutes: Override duration in minutes.

        Returns:
            Override ID.
        """
        with self._lock:
            override_id = f"OVR-{symbol}-{int(time.time())}"
            expiry = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

            override = {
                "override_id": override_id,
                "symbol": symbol,
                "target_position": target_position,
                "operator": operator,
                "reason": reason,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expiry.isoformat(),
                "active": True,
            }

            self._active_overrides[override_id] = override
            self._override_history.append(override.copy())

            logger.info(
                f"Position override set: {symbol} -> {target_position} "
                f"by {operator} (expires {expiry})"
            )

            return override_id

    def set_signal_veto(
        self,
        symbol: str,
        direction: str,
        operator: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> str:
        """
        Veto AI signals for a symbol/direction.

        Args:
            symbol: Trading symbol.
            direction: Signal direction to veto ("buy", "sell", "both").
            operator: Operator identifier.
            reason: Reason for veto.
            duration_minutes: Veto duration.

        Returns:
            Veto ID.
        """
        with self._lock:
            veto_id = f"VETO-{symbol}-{direction}-{int(time.time())}"
            expiry = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

            veto = {
                "veto_id": veto_id,
                "symbol": symbol,
                "direction": direction,
                "operator": operator,
                "reason": reason,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expiry.isoformat(),
                "active": True,
            }

            self._active_overrides[veto_id] = veto
            self._override_history.append(veto.copy())

            logger.info(
                f"Signal veto set: {symbol} {direction} by {operator} " f"(expires {expiry})"
            )

            return veto_id

    def cancel_override(self, override_id: str, operator: str) -> bool:
        """
        Cancel an active override.

        Args:
            override_id: Override to cancel.
            operator: Operator canceling.

        Returns:
            True if canceled successfully.
        """
        with self._lock:
            if override_id not in self._active_overrides:
                return False

            self._active_overrides[override_id]["active"] = False
            self._active_overrides[override_id]["canceled_by"] = operator
            self._active_overrides[override_id]["canceled_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            logger.info(f"Override {override_id} canceled by {operator}")
            return True

    def get_active_overrides(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active overrides, optionally filtered by symbol."""
        with self._lock:
            now = datetime.now(timezone.utc)
            active = []

            for override in self._active_overrides.values():
                if not override.get("active", False):
                    continue

                expiry = datetime.fromisoformat(override["expires_at"])
                if expiry < now:
                    override["active"] = False
                    continue

                if symbol and override.get("symbol") != symbol:
                    continue

                active.append(override.copy())

            return active

    def check_signal_vetoed(self, symbol: str, direction: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a signal is vetoed.

        Args:
            symbol: Trading symbol.
            direction: Signal direction ("buy" or "sell").

        Returns:
            Tuple of (is_vetoed, veto_reason).
        """
        overrides = self.get_active_overrides(symbol)
        for override in overrides:
            if "veto_id" not in override.get("veto_id", override.get("override_id", "")):
                continue
            veto_dir = override.get("direction", "")
            if veto_dir == "both" or veto_dir == direction:
                return True, override.get("reason", "Signal vetoed by operator")
        return False, None


class AutomationBiasMonitor:
    """
    Monitor for automation bias in human-AI interaction.

    Tracks operator behavior to detect potential automation bias
    (over-reliance on AI recommendations).

    EU AI Act requires measures to prevent automation bias in
    human oversight of high-risk AI systems.

    Attributes:
        config: Oversight configuration.
    """

    def __init__(self, config: HumanOversightConfig) -> None:
        """
        Initialize AutomationBiasMonitor.

        Args:
            config: Oversight configuration.
        """
        self._config = config
        self._lock = threading.Lock()

        # Track AI recommendations and human decisions
        self._recommendations: deque = deque(maxlen=1000)
        self._human_decisions: deque = deque(maxlen=1000)
        self._agreement_rate_history: deque = deque(maxlen=100)

        # Bias thresholds
        self._high_agreement_threshold = 0.95  # Over-reliance warning
        self._low_disagreement_threshold = 0.02  # Never questioning AI
        self._review_interval_seconds = 3600  # 1 hour

    def record_recommendation(
        self,
        recommendation_id: str,
        ai_action: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> None:
        """Record an AI recommendation."""
        with self._lock:
            self._recommendations.append(
                {
                    "id": recommendation_id,
                    "action": ai_action,
                    "confidence": confidence,
                    "context": context,
                    "timestamp": time.time(),
                }
            )

    def record_human_decision(
        self,
        recommendation_id: str,
        human_action: str,
        followed_ai: bool,
        operator: str,
        reason: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """
        Record human decision on AI recommendation.

        Args:
            recommendation_id: Related AI recommendation.
            human_action: Action taken by human.
            followed_ai: Whether human followed AI recommendation.
            operator: Operator identifier.
            reason: Reason if disagreeing with AI.

        Returns:
            AnomalyAlert if automation bias detected.
        """
        with self._lock:
            self._human_decisions.append(
                {
                    "recommendation_id": recommendation_id,
                    "human_action": human_action,
                    "followed_ai": followed_ai,
                    "operator": operator,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            )

            # Calculate agreement rate
            return self._check_automation_bias()

    def _check_automation_bias(self) -> Optional[AnomalyAlert]:
        """Check for automation bias indicators."""
        recent_decisions = [
            d
            for d in self._human_decisions
            if time.time() - d["timestamp"] < self._review_interval_seconds
        ]

        if len(recent_decisions) < 10:
            return None

        agreement_rate = sum(1 for d in recent_decisions if d["followed_ai"]) / len(
            recent_decisions
        )

        self._agreement_rate_history.append(
            {
                "rate": agreement_rate,
                "sample_size": len(recent_decisions),
                "timestamp": time.time(),
            }
        )

        # Check for over-reliance (following AI too much)
        if agreement_rate >= self._high_agreement_threshold:
            return AnomalyAlert(
                alert_id=f"BIAS-{int(time.time() * 1000)}",
                severity=AlertSeverity.WARNING,
                source="automation_bias_monitor",
                message=(
                    f"Potential automation bias detected: {agreement_rate:.1%} "
                    f"agreement rate with AI recommendations over last "
                    f"{len(recent_decisions)} decisions"
                ),
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "agreement_rate": agreement_rate,
                    "sample_size": len(recent_decisions),
                    "threshold": self._high_agreement_threshold,
                },
                recommended_action=(
                    "Review recent AI recommendations critically. Consider "
                    "independent analysis before following AI suggestions."
                ),
            )

        return None

    def get_bias_metrics(self) -> Dict[str, Any]:
        """Get automation bias metrics summary."""
        with self._lock:
            recent_decisions = [
                d
                for d in self._human_decisions
                if time.time() - d["timestamp"] < self._review_interval_seconds
            ]

            if not recent_decisions:
                return {
                    "agreement_rate": None,
                    "sample_size": 0,
                    "bias_risk": "insufficient_data",
                }

            agreement_rate = sum(1 for d in recent_decisions if d["followed_ai"]) / len(
                recent_decisions
            )

            # Determine bias risk level
            if agreement_rate >= self._high_agreement_threshold:
                bias_risk = "high"
            elif agreement_rate >= 0.85:
                bias_risk = "moderate"
            else:
                bias_risk = "low"

            return {
                "agreement_rate": agreement_rate,
                "sample_size": len(recent_decisions),
                "bias_risk": bias_risk,
                "review_interval_hours": self._review_interval_seconds / 3600,
                "high_agreement_threshold": self._high_agreement_threshold,
            }


class HumanOversightSystem:
    """
    Complete Human Oversight System for EU AI Act Compliance.

    Implements Article 14 requirements including:
    - Emergency stop capability (L1)
    - Trading pause with state preservation (L2)
    - Manual override of AI decisions (L3)
    - Complete audit trail (L4)
    - Anomaly detection and alerting
    - Automation bias monitoring

    Thread-safe implementation for production use.

    Example:
        >>> oversight = HumanOversightSystem()
        >>> oversight.emergency_stop("Market anomaly", "operator1")
        >>> status = oversight.get_status()
        >>> oversight.resume("operator1", "Anomaly resolved")
    """

    def __init__(
        self,
        config: Optional[HumanOversightConfig] = None,
    ) -> None:
        """
        Initialize HumanOversightSystem.

        Args:
            config: Configuration options.
        """
        self._config = config or HumanOversightConfig()
        self._lock = threading.RLock()

        # System state
        self._state = SystemState.ACTIVE
        self._state_changed_at = datetime.now(timezone.utc)
        self._state_changed_by = "system"

        # Components
        self._anomaly_detector = AnomalyDetector(self._config)
        self._override_controller = ManualOverrideController()
        self._bias_monitor = AutomationBiasMonitor(self._config)

        # Action history
        self._action_history: List[OversightAction] = []
        self._alerts: List[AnomalyAlert] = []

        # Callbacks
        self._on_state_change_callbacks: List[Callable[[SystemState, SystemState], None]] = []
        self._on_alert_callbacks: List[Callable[[AnomalyAlert], None]] = []

        # Persistence
        if self._config.state_persistence_path is None:
            self._persistence_path = Path("logs/ai_act/human_oversight")
        else:
            self._persistence_path = self._config.state_persistence_path
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        logger.info("HumanOversightSystem initialized")

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if system is active."""
        return self._state == SystemState.ACTIVE

    @property
    def is_stopped(self) -> bool:
        """Check if system is stopped."""
        return self._state == SystemState.STOPPED

    def emergency_stop(
        self,
        reason: str,
        operator: str = "system",
    ) -> OversightAction:
        """
        Execute emergency stop (L1 capability).

        Immediately halts all AI trading operations.

        Args:
            reason: Reason for emergency stop.
            operator: Operator executing the stop.

        Returns:
            OversightAction record.
        """
        with self._lock:
            old_state = self._state
            action = OversightAction(
                action_id=f"STOP-{int(time.time() * 1000)}",
                capability=OversightCapability.EMERGENCY_STOP,
                operator=operator,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
                system_state_before=old_state,
                system_state_after=SystemState.STOPPED,
            )

            self._state = SystemState.STOPPED
            self._state_changed_at = action.timestamp
            self._state_changed_by = operator

            self._action_history.append(action)
            self._persist_state()
            self._trigger_state_change_callbacks(old_state, SystemState.STOPPED)

            logger.critical(f"EMERGENCY STOP executed by {operator}: {reason}")
            return action

    def pause_trading(
        self,
        reason: str,
        operator: str = "system",
    ) -> OversightAction:
        """
        Pause trading operations (L2 capability).

        Temporarily suspends trading while preserving state.

        Args:
            reason: Reason for pause.
            operator: Operator executing the pause.

        Returns:
            OversightAction record.
        """
        with self._lock:
            old_state = self._state
            action = OversightAction(
                action_id=f"PAUSE-{int(time.time() * 1000)}",
                capability=OversightCapability.TRADING_PAUSE,
                operator=operator,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
                system_state_before=old_state,
                system_state_after=SystemState.PAUSED,
            )

            self._state = SystemState.PAUSED
            self._state_changed_at = action.timestamp
            self._state_changed_by = operator

            self._action_history.append(action)
            self._persist_state()
            self._trigger_state_change_callbacks(old_state, SystemState.PAUSED)

            logger.warning(f"Trading PAUSED by {operator}: {reason}")
            return action

    def resume(
        self,
        operator: str,
        reason: str = "Normal operations resumed",
    ) -> OversightAction:
        """
        Resume trading operations.

        Args:
            operator: Operator authorizing resume.
            reason: Reason for resume.

        Returns:
            OversightAction record.
        """
        with self._lock:
            old_state = self._state
            action = OversightAction(
                action_id=f"RESUME-{int(time.time() * 1000)}",
                capability=OversightCapability.TRADING_PAUSE,
                operator=operator,
                timestamp=datetime.now(timezone.utc),
                reason=reason,
                details={"action": "resume"},
                system_state_before=old_state,
                system_state_after=SystemState.ACTIVE,
            )

            self._state = SystemState.ACTIVE
            self._state_changed_at = action.timestamp
            self._state_changed_by = operator

            self._action_history.append(action)
            self._persist_state()
            self._trigger_state_change_callbacks(old_state, SystemState.ACTIVE)

            logger.info(f"Trading RESUMED by {operator}: {reason}")
            return action

    def set_override(
        self,
        symbol: str,
        target_position: float,
        operator: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> str:
        """
        Set manual position override (L3 capability).

        Args:
            symbol: Trading symbol.
            target_position: Target position.
            operator: Operator setting override.
            reason: Reason for override.
            duration_minutes: Override duration.

        Returns:
            Override ID.
        """
        if OversightCapability.POSITION_OVERRIDE not in self._config.enabled_capabilities:
            raise ValueError("POSITION_OVERRIDE capability not enabled")

        override_id = self._override_controller.set_position_override(
            symbol=symbol,
            target_position=target_position,
            operator=operator,
            reason=reason,
            duration_minutes=duration_minutes,
        )

        action = OversightAction(
            action_id=override_id,
            capability=OversightCapability.POSITION_OVERRIDE,
            operator=operator,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            details={
                "symbol": symbol,
                "target_position": target_position,
                "duration_minutes": duration_minutes,
            },
            system_state_before=self._state,
            system_state_after=SystemState.OVERRIDE,
        )

        self._action_history.append(action)
        return override_id

    def veto_signal(
        self,
        symbol: str,
        direction: str,
        operator: str,
        reason: str,
        duration_minutes: int = 60,
    ) -> str:
        """
        Veto AI trading signal (L3 capability).

        Args:
            symbol: Trading symbol.
            direction: Direction to veto ("buy", "sell", "both").
            operator: Operator setting veto.
            reason: Reason for veto.
            duration_minutes: Veto duration.

        Returns:
            Veto ID.
        """
        if OversightCapability.SIGNAL_VETO not in self._config.enabled_capabilities:
            raise ValueError("SIGNAL_VETO capability not enabled")

        veto_id = self._override_controller.set_signal_veto(
            symbol=symbol,
            direction=direction,
            operator=operator,
            reason=reason,
            duration_minutes=duration_minutes,
        )

        action = OversightAction(
            action_id=veto_id,
            capability=OversightCapability.SIGNAL_VETO,
            operator=operator,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            details={
                "symbol": symbol,
                "direction": direction,
                "duration_minutes": duration_minutes,
            },
            system_state_before=self._state,
            system_state_after=self._state,
        )

        self._action_history.append(action)
        return veto_id

    def record_metric(self, metric: str, value: float) -> Optional[AnomalyAlert]:
        """
        Record a metric for anomaly detection.

        Args:
            metric: Metric name.
            value: Metric value.

        Returns:
            AnomalyAlert if anomaly detected.
        """
        if not self._config.anomaly_detection_enabled:
            return None

        alert = self._anomaly_detector.record_metric(metric, value)

        if alert:
            self._alerts.append(alert)
            self._trigger_alert_callbacks(alert)

            # Auto-actions based on severity
            if alert.severity == AlertSeverity.EMERGENCY and self._config.auto_stop_on_emergency:
                self.emergency_stop(
                    reason=f"Auto-stop triggered by emergency alert: {alert.message}",
                    operator="system_auto",
                )
            elif alert.severity == AlertSeverity.CRITICAL and self._config.auto_pause_on_critical:
                self.pause_trading(
                    reason=f"Auto-pause triggered by critical alert: {alert.message}",
                    operator="system_auto",
                )

        return alert

    def record_ai_recommendation(
        self,
        recommendation_id: str,
        ai_action: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> None:
        """Record AI recommendation for bias monitoring."""
        if self._config.automation_bias_monitoring:
            self._bias_monitor.record_recommendation(
                recommendation_id, ai_action, confidence, context
            )

    def record_human_decision(
        self,
        recommendation_id: str,
        human_action: str,
        followed_ai: bool,
        operator: str,
        reason: Optional[str] = None,
    ) -> Optional[AnomalyAlert]:
        """
        Record human decision for bias monitoring.

        Returns AnomalyAlert if automation bias detected.
        """
        if not self._config.automation_bias_monitoring:
            return None

        alert = self._bias_monitor.record_human_decision(
            recommendation_id, human_action, followed_ai, operator, reason
        )

        if alert:
            self._alerts.append(alert)
            self._trigger_alert_callbacks(alert)

        return alert

    def acknowledge_alert(self, alert_id: str, operator: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge(operator)
                logger.info(f"Alert {alert_id} acknowledged by {operator}")
                return True
        return False

    def get_unacknowledged_alerts(self) -> List[AnomalyAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]

    def check_signal_allowed(
        self,
        symbol: str,
        direction: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trading signal is allowed.

        Args:
            symbol: Trading symbol.
            direction: Signal direction.

        Returns:
            Tuple of (is_allowed, reason_if_blocked).
        """
        # Check system state
        if self._state == SystemState.STOPPED:
            return False, "System is stopped"
        if self._state == SystemState.PAUSED:
            return False, "Trading is paused"

        # Check signal veto
        vetoed, veto_reason = self._override_controller.check_signal_vetoed(symbol, direction)
        if vetoed:
            return False, veto_reason

        return True, None

    def get_position_override(self, symbol: str) -> Optional[float]:
        """Get active position override for symbol."""
        overrides = self._override_controller.get_active_overrides(symbol)
        for override in overrides:
            if "target_position" in override:
                return override["target_position"]
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            return {
                "state": self._state.value,
                "state_changed_at": self._state_changed_at.isoformat(),
                "state_changed_by": self._state_changed_by,
                "is_active": self.is_active,
                "enabled_capabilities": [c.value for c in self._config.enabled_capabilities],
                "active_overrides": len(self._override_controller.get_active_overrides()),
                "unacknowledged_alerts": len(self.get_unacknowledged_alerts()),
                "total_actions": len(self._action_history),
                "automation_bias_metrics": self._bias_monitor.get_bias_metrics(),
            }

    def get_audit_trail(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail of oversight actions (L4 capability).

        Args:
            start_time: Filter start time.
            end_time: Filter end time.

        Returns:
            List of action records.
        """
        with self._lock:
            actions = self._action_history

            if start_time:
                actions = [a for a in actions if a.timestamp >= start_time]
            if end_time:
                actions = [a for a in actions if a.timestamp <= end_time]

            return [a.to_dict() for a in actions]

    def export_audit_report(self, output_path: Optional[Path] = None) -> Path:
        """Export complete audit report."""
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self._persistence_path / f"audit_report_{timestamp}.json"

        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "ai_act_article": "Article 14 - Human Oversight",
            "system_status": self.get_status(),
            "action_history": [a.to_dict() for a in self._action_history],
            "alerts": [a.to_dict() for a in self._alerts],
            "automation_bias_metrics": self._bias_monitor.get_bias_metrics(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Audit report exported to {output_path}")
        return output_path

    def register_state_change_callback(
        self,
        callback: Callable[[SystemState, SystemState], None],
    ) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def register_alert_callback(
        self,
        callback: Callable[[AnomalyAlert], None],
    ) -> None:
        """Register callback for alerts."""
        self._on_alert_callbacks.append(callback)

    def _trigger_state_change_callbacks(
        self,
        old_state: SystemState,
        new_state: SystemState,
    ) -> None:
        """Trigger state change callbacks."""
        for callback in self._on_state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def _trigger_alert_callbacks(self, alert: AnomalyAlert) -> None:
        """Trigger alert callbacks."""
        for callback in self._on_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _persist_state(self) -> None:
        """Persist current state to disk."""
        state_file = self._persistence_path / "oversight_state.json"
        state_data = {
            "state": self._state.value,
            "state_changed_at": self._state_changed_at.isoformat(),
            "state_changed_by": self._state_changed_by,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)


def create_human_oversight_system(
    config: Optional[HumanOversightConfig] = None,
) -> HumanOversightSystem:
    """
    Factory function to create HumanOversightSystem.

    Args:
        config: Configuration options.

    Returns:
        Configured HumanOversightSystem instance.
    """
    return HumanOversightSystem(config=config)
