# -*- coding: utf-8 -*-
"""
Dashboard Alerts - CLOUD ZONE ONLY.

Alert management for dashboard notifications.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import UUID


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

ALERT_RETENTION_HOURS: Final[int] = 168  # 7 days
MAX_ALERTS_PER_WORKSPACE: Final[int] = 10000
ALERT_EVAL_INTERVAL_SECONDS: Final[int] = 60


# ============================================================================
# Enums
# ============================================================================


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    """Alert lifecycle state."""

    PENDING = "pending"  # Just triggered
    FIRING = "firing"  # Active
    RESOLVED = "resolved"  # Condition cleared
    ACKNOWLEDGED = "acknowledged"  # User acknowledged
    SILENCED = "silenced"  # Temporarily suppressed


class ComparisonOperator(str, Enum):
    """Comparison operators for alert conditions."""

    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal


class AggregationType(str, Enum):
    """Aggregation types for alert conditions."""

    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AlertCondition:
    """
    Condition that triggers an alert.

    Example: CPU > 90% for 5 minutes
    """

    metric_name: str
    operator: ComparisonOperator
    threshold: float

    # Aggregation settings
    aggregation: AggregationType = AggregationType.AVG
    window_minutes: int = 5

    # Optional filters
    filters: Dict[str, str] = field(default_factory=dict)

    def evaluate(self, value: float) -> bool:
        """Evaluate condition against a value."""
        if self.operator == ComparisonOperator.GT:
            return value > self.threshold
        elif self.operator == ComparisonOperator.GTE:
            return value >= self.threshold
        elif self.operator == ComparisonOperator.LT:
            return value < self.threshold
        elif self.operator == ComparisonOperator.LTE:
            return value <= self.threshold
        elif self.operator == ComparisonOperator.EQ:
            return value == self.threshold
        elif self.operator == ComparisonOperator.NE:
            return value != self.threshold
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "aggregation": self.aggregation.value,
            "window_minutes": self.window_minutes,
            "filters": self.filters,
        }


@dataclass
class AlertRule:
    """
    Alert rule definition.

    Defines when and how alerts should be triggered.
    """

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    workspace_id: Optional[UUID] = None  # None = global rule

    # Conditions (all must be true = AND logic)
    conditions: List[AlertCondition] = field(default_factory=list)

    # Alert settings
    severity: AlertSeverity = AlertSeverity.WARNING

    # Notification settings
    notify_channels: List[str] = field(default_factory=list)  # email, slack, webhook
    notify_interval_minutes: int = 60  # Min time between notifications

    # State
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "conditions": [c.to_dict() for c in self.conditions],
            "severity": self.severity.value,
            "notify_channels": self.notify_channels,
            "notify_interval_minutes": self.notify_interval_minutes,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "labels": self.labels,
        }


@dataclass
class Alert:
    """
    Alert instance.

    Represents an active or historical alert.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    workspace_id: Optional[UUID] = None

    # Alert details
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    state: AlertState = AlertState.PENDING

    # Trigger information
    triggered_value: float = 0.0
    threshold: float = 0.0
    metric_name: str = ""

    # Timestamps
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # Notification tracking
    last_notified_at: Optional[datetime] = None
    notification_count: int = 0

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def acknowledge(self, user_id: str) -> None:
        """Acknowledge the alert."""
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = user_id

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "state": self.state.value,
            "triggered_value": self.triggered_value,
            "threshold": self.threshold,
            "metric_name": self.metric_name,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "labels": self.labels,
        }


@dataclass
class SilenceRule:
    """
    Silence rule to suppress alerts.

    Temporarily suppresses alerts matching certain criteria.
    """

    silence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: Optional[UUID] = None

    # Matching criteria
    matchers: Dict[str, str] = field(default_factory=dict)  # label -> value

    # Time window
    starts_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ends_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )

    # Metadata
    created_by: str = ""
    comment: str = ""

    def is_active(self) -> bool:
        """Check if silence is currently active."""
        now = datetime.now(timezone.utc)
        return self.starts_at <= now <= self.ends_at

    def matches_alert(self, alert: Alert) -> bool:
        """Check if alert matches this silence rule."""
        if not self.is_active():
            return False

        for key, value in self.matchers.items():
            if alert.labels.get(key) != value:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silence_id": self.silence_id,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "matchers": self.matchers,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": self.ends_at.isoformat(),
            "created_by": self.created_by,
            "comment": self.comment,
            "is_active": self.is_active(),
        }


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """
    Manages alerts, rules, and notifications.

    SECURITY: Respects workspace isolation.
    """

    def __init__(self, metrics_provider: Optional[Any] = None):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._silences: Dict[str, SilenceRule] = {}
        self._metrics_provider = metrics_provider
        self._running = False
        self._notification_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

        logger.info("AlertManager initialized")

    # ========================================================================
    # Rule Management
    # ========================================================================

    async def create_rule(self, rule: AlertRule) -> AlertRule:
        """Create alert rule."""
        async with self._lock:
            self._rules[rule.rule_id] = rule
        logger.info(f"Created alert rule: {rule.name}")
        return rule

    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[AlertRule]:
        """Update alert rule."""
        async with self._lock:
            rule = self._rules.get(rule_id)
            if not rule:
                return None

            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            rule.updated_at = datetime.now(timezone.utc)

        logger.info(f"Updated alert rule: {rule_id}")
        return rule

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete alert rule."""
        async with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Deleted alert rule: {rule_id}")
                return True
        return False

    async def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID."""
        return self._rules.get(rule_id)

    async def list_rules(
        self,
        workspace_id: Optional[UUID] = None,
        enabled_only: bool = False,
    ) -> List[AlertRule]:
        """List alert rules."""
        rules = []
        for rule in self._rules.values():
            if workspace_id and rule.workspace_id != workspace_id:
                continue
            if enabled_only and not rule.enabled:
                continue
            rules.append(rule)
        return rules

    # ========================================================================
    # Alert Management
    # ========================================================================

    async def trigger_alert(
        self,
        rule: AlertRule,
        triggered_value: float,
        condition: AlertCondition,
    ) -> Alert:
        """Trigger a new alert from a rule."""
        # Check for existing firing alert from same rule
        for alert in self._alerts.values():
            if alert.rule_id == rule.rule_id and alert.state == AlertState.FIRING:
                # Update existing alert
                alert.triggered_value = triggered_value
                return alert

        alert = Alert(
            rule_id=rule.rule_id,
            workspace_id=rule.workspace_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            state=AlertState.FIRING,
            triggered_value=triggered_value,
            threshold=condition.threshold,
            metric_name=condition.metric_name,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
        )

        async with self._lock:
            self._alerts[alert.alert_id] = alert

        logger.warning(f"Alert triggered: {alert.name} ({alert.severity.value})")

        # Send notifications
        await self._send_notifications(alert, rule)

        return alert

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        alert.resolve()
        logger.info(f"Alert resolved: {alert.name}")
        return True

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        alert.acknowledge(user_id)
        logger.info(f"Alert acknowledged: {alert.name} by {user_id}")
        return True

    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    async def list_alerts(
        self,
        workspace_id: Optional[UUID] = None,
        state: Optional[AlertState] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """List alerts with filtering."""
        alerts = []
        for alert in sorted(
            self._alerts.values(),
            key=lambda a: a.triggered_at,
            reverse=True,
        ):
            if workspace_id and alert.workspace_id != workspace_id:
                continue
            if state and alert.state != state:
                continue
            if severity and alert.severity != severity:
                continue
            alerts.append(alert)
            if len(alerts) >= limit:
                break
        return alerts

    async def get_alert_counts(
        self,
        workspace_id: Optional[UUID] = None,
    ) -> Dict[str, int]:
        """Get alert counts by severity and state."""
        counts = {
            "total": 0,
            "by_severity": {s.value: 0 for s in AlertSeverity},
            "by_state": {s.value: 0 for s in AlertState},
        }

        for alert in self._alerts.values():
            if workspace_id and alert.workspace_id != workspace_id:
                continue
            counts["total"] += 1
            counts["by_severity"][alert.severity.value] += 1
            counts["by_state"][alert.state.value] += 1

        return counts

    # ========================================================================
    # Silence Management
    # ========================================================================

    async def create_silence(self, silence: SilenceRule) -> SilenceRule:
        """Create silence rule."""
        async with self._lock:
            self._silences[silence.silence_id] = silence
        logger.info(f"Created silence rule: {silence.silence_id}")
        return silence

    async def delete_silence(self, silence_id: str) -> bool:
        """Delete silence rule."""
        async with self._lock:
            if silence_id in self._silences:
                del self._silences[silence_id]
                return True
        return False

    async def list_silences(
        self,
        workspace_id: Optional[UUID] = None,
        active_only: bool = False,
    ) -> List[SilenceRule]:
        """List silence rules."""
        silences = []
        for silence in self._silences.values():
            if workspace_id and silence.workspace_id != workspace_id:
                continue
            if active_only and not silence.is_active():
                continue
            silences.append(silence)
        return silences

    def _is_silenced(self, alert: Alert) -> bool:
        """Check if alert should be silenced."""
        for silence in self._silences.values():
            if silence.matches_alert(alert):
                return True
        return False

    # ========================================================================
    # Notification
    # ========================================================================

    def register_notification_handler(
        self,
        channel: str,
        handler: Callable[[Alert, AlertRule], Any],
    ) -> None:
        """Register notification handler for a channel."""
        self._notification_handlers[channel] = handler
        logger.info(f"Registered notification handler for: {channel}")

    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        # Check if silenced
        if self._is_silenced(alert):
            alert.state = AlertState.SILENCED
            logger.info(f"Alert silenced: {alert.name}")
            return

        # Check notification interval
        if alert.last_notified_at:
            elapsed = datetime.now(timezone.utc) - alert.last_notified_at
            if elapsed < timedelta(minutes=rule.notify_interval_minutes):
                return

        # Send to each channel
        for channel in rule.notify_channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    (
                        await handler(alert, rule)
                        if asyncio.iscoroutinefunction(handler)
                        else handler(alert, rule)
                    )
                    logger.info(f"Sent alert notification to {channel}")
                except Exception as e:
                    logger.exception(f"Failed to send notification to {channel}: {e}")

        alert.last_notified_at = datetime.now(timezone.utc)
        alert.notification_count += 1

    # ========================================================================
    # Evaluation Loop
    # ========================================================================

    async def start(self) -> None:
        """Start alert evaluation loop."""
        if self._running:
            return

        self._running = True
        logger.info("AlertManager started")

        while self._running:
            try:
                await self._evaluate_rules()
                await self._cleanup_old_alerts()
            except Exception as e:
                logger.exception(f"Alert evaluation error: {e}")

            await asyncio.sleep(ALERT_EVAL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        """Stop alert evaluation loop."""
        self._running = False
        logger.info("AlertManager stopped")

    async def _evaluate_rules(self) -> None:
        """Evaluate all enabled rules."""
        if not self._metrics_provider:
            return

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.exception(f"Failed to evaluate rule {rule.name}: {e}")

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single rule."""
        # All conditions must be true
        all_triggered = True
        triggered_condition = None
        triggered_value = 0.0

        for condition in rule.conditions:
            value = await self._get_metric_value(
                condition.metric_name,
                condition.aggregation,
                condition.window_minutes,
                condition.filters,
                rule.workspace_id,
            )

            if value is None:
                all_triggered = False
                break

            if not condition.evaluate(value):
                all_triggered = False
                break

            triggered_condition = condition
            triggered_value = value

        if all_triggered and triggered_condition:
            await self.trigger_alert(rule, triggered_value, triggered_condition)
        else:
            # Check if we should resolve existing alerts
            for alert in list(self._alerts.values()):
                if alert.rule_id == rule.rule_id and alert.state == AlertState.FIRING:
                    await self.resolve_alert(alert.alert_id)

    async def _get_metric_value(
        self,
        metric_name: str,
        aggregation: AggregationType,
        window_minutes: int,
        filters: Dict[str, str],
        workspace_id: Optional[UUID],
    ) -> Optional[float]:
        """Get aggregated metric value from provider."""
        if not self._metrics_provider:
            return None

        try:
            # This would integrate with the telemetry service
            return await self._metrics_provider.get_aggregated_value(
                metric_name=metric_name,
                aggregation=aggregation.value,
                window_minutes=window_minutes,
                filters=filters,
                workspace_id=workspace_id,
            )
        except Exception as e:
            logger.warning(f"Failed to get metric {metric_name}: {e}")
            return None

    async def _cleanup_old_alerts(self) -> None:
        """Remove old resolved alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ALERT_RETENTION_HOURS)

        to_remove = []
        for alert_id, alert in self._alerts.items():
            if alert.state == AlertState.RESOLVED:
                if alert.resolved_at and alert.resolved_at < cutoff:
                    to_remove.append(alert_id)

        async with self._lock:
            for alert_id in to_remove:
                del self._alerts[alert_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old alerts")


# ============================================================================
# Default Alert Rules
# ============================================================================


def create_default_rules(workspace_id: Optional[UUID] = None) -> List[AlertRule]:
    """Create default alert rules for a workspace."""
    return [
        AlertRule(
            name="High CPU Usage",
            description="CPU usage exceeded 90% for 5 minutes",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="system.cpu.usage",
                    operator=ComparisonOperator.GT,
                    threshold=90.0,
                    window_minutes=5,
                )
            ],
            severity=AlertSeverity.WARNING,
            labels={"category": "resource", "resource": "cpu"},
        ),
        AlertRule(
            name="High Memory Usage",
            description="Memory usage exceeded 85%",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="system.memory.usage",
                    operator=ComparisonOperator.GT,
                    threshold=85.0,
                    window_minutes=5,
                )
            ],
            severity=AlertSeverity.WARNING,
            labels={"category": "resource", "resource": "memory"},
        ),
        AlertRule(
            name="High Error Rate",
            description="Error rate exceeded 5% for 10 minutes",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="api.error_rate",
                    operator=ComparisonOperator.GT,
                    threshold=5.0,
                    window_minutes=10,
                )
            ],
            severity=AlertSeverity.ERROR,
            labels={"category": "api", "metric": "errors"},
        ),
        AlertRule(
            name="Job Queue Backlog",
            description="More than 100 jobs pending for 15 minutes",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="jobs.pending_count",
                    operator=ComparisonOperator.GT,
                    threshold=100.0,
                    window_minutes=15,
                )
            ],
            severity=AlertSeverity.WARNING,
            labels={"category": "jobs", "metric": "backlog"},
        ),
        AlertRule(
            name="Agent Offline",
            description="Agent has not reported heartbeat for 5 minutes",
            workspace_id=workspace_id,
            conditions=[
                AlertCondition(
                    metric_name="agent.heartbeat_age_seconds",
                    operator=ComparisonOperator.GT,
                    threshold=300.0,
                    window_minutes=1,
                )
            ],
            severity=AlertSeverity.CRITICAL,
            labels={"category": "agent", "metric": "health"},
        ),
    ]
