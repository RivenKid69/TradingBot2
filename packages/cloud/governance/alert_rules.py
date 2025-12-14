# -*- coding: utf-8 -*-
"""
Alert Rules Engine.

CCEA Phase 8 Implementation.

This engine provides event-based alerting:
    - Configurable alert rules
    - Multiple trigger conditions
    - Action execution (notify, escalate, etc.)
    - Alert deduplication and throttling

Design Doc Reference:
    - Phase 8 (4.1/3.1): "Alerts for basic events"
    - "Kill switch, broker errors burst, data feed invalid, order spam"

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Pattern, Set, Union
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Default throttle window
DEFAULT_THROTTLE_SECONDS: Final[int] = 300

# Alert severities
class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertAction(Enum):
    """Actions to take when alert triggers."""
    NOTIFY = auto()        # Send notification
    LOG = auto()           # Log only
    ESCALATE = auto()      # Escalate to on-call
    WEBHOOK = auto()       # Call webhook
    SLACK = auto()         # Send to Slack
    EMAIL = auto()         # Send email
    PAGERDUTY = auto()     # Page on-call


class AlertConditionType(Enum):
    """Types of alert conditions."""
    THRESHOLD = auto()     # Value exceeds threshold
    EQUALS = auto()        # Value equals target
    CONTAINS = auto()      # Value contains string
    REGEX = auto()         # Value matches regex
    RATE = auto()          # Rate of events exceeds threshold
    COUNT = auto()         # Count of events exceeds threshold
    ABSENT = auto()        # Event absent for duration


@dataclass
class AlertCondition:
    """
    Condition for triggering an alert.

    Attributes:
        type: Condition type
        field: Field to evaluate
        operator: Comparison operator (<, >, ==, !=, etc.)
        threshold: Threshold value
        window_seconds: Time window for rate/count conditions
    """
    type: AlertConditionType = AlertConditionType.THRESHOLD
    field: str = ""
    operator: str = ">"
    threshold: Union[int, float, str] = 0
    window_seconds: int = 60
    regex_pattern: Optional[str] = None

    def evaluate(self, value: Any) -> bool:
        """Evaluate condition against value."""
        if self.type == AlertConditionType.THRESHOLD:
            return self._evaluate_threshold(value)
        elif self.type == AlertConditionType.EQUALS:
            return value == self.threshold
        elif self.type == AlertConditionType.CONTAINS:
            return str(self.threshold) in str(value)
        elif self.type == AlertConditionType.REGEX:
            if self.regex_pattern:
                return bool(re.search(self.regex_pattern, str(value)))
            return False
        return False

    def _evaluate_threshold(self, value: Any) -> bool:
        """Evaluate threshold condition."""
        try:
            v = float(value)
            t = float(self.threshold)

            if self.operator == ">":
                return v > t
            elif self.operator == ">=":
                return v >= t
            elif self.operator == "<":
                return v < t
            elif self.operator == "<=":
                return v <= t
            elif self.operator == "==":
                return v == t
            elif self.operator == "!=":
                return v != t
        except (TypeError, ValueError):
            pass
        return False


@dataclass
class AlertTrigger:
    """
    Alert trigger event.

    Attributes:
        id: Trigger identifier
        rule_id: Rule that triggered
        alert_type: Type of alert
        severity: Severity level
        message: Alert message
        context: Additional context
        timestamp: When triggered
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    alert_type: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    workspace_id: str = ""
    agent_id: Optional[str] = None

    # Status
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "agent_id": self.agent_id,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertRule:
    """
    Alert rule definition.

    Attributes:
        id: Rule identifier
        name: Rule name
        description: Rule description
        alert_type: Type of alert this generates
        conditions: Conditions that trigger the rule
        severity: Severity when triggered
        actions: Actions to take
        enabled: Whether rule is active
        throttle_seconds: Minimum time between triggers
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    alert_type: str = ""
    workspace_id: str = ""
    conditions: List[AlertCondition] = field(default_factory=list)
    severity: AlertSeverity = AlertSeverity.WARNING
    actions: Set[AlertAction] = field(default_factory=lambda: {AlertAction.NOTIFY})
    enabled: bool = True
    throttle_seconds: int = DEFAULT_THROTTLE_SECONDS

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered_at: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "alert_type": self.alert_type,
            "workspace_id": self.workspace_id,
            "conditions": [
                {
                    "type": c.type.name,
                    "field": c.field,
                    "operator": c.operator,
                    "threshold": c.threshold,
                }
                for c in self.conditions
            ],
            "severity": self.severity.value,
            "actions": [a.name for a in self.actions],
            "enabled": self.enabled,
            "throttle_seconds": self.throttle_seconds,
            "trigger_count": self.trigger_count,
        }

    def evaluate(self, event: Dict[str, Any]) -> bool:
        """
        Evaluate event against all conditions.

        Args:
            event: Event to evaluate

        Returns:
            True if all conditions match
        """
        if not self.enabled or not self.conditions:
            return False

        # All conditions must match
        for condition in self.conditions:
            value = event.get(condition.field)
            if not condition.evaluate(value):
                return False

        return True

    def is_throttled(self) -> bool:
        """Check if rule is in throttle period."""
        if not self.last_triggered_at:
            return False
        elapsed = (datetime.utcnow() - self.last_triggered_at).total_seconds()
        return elapsed < self.throttle_seconds


class AlertRulesEngine:
    """
    Alert rules evaluation engine.

    Evaluates events against rules and triggers alerts.

    Usage:
        engine = AlertRulesEngine()

        # Create rules
        engine.add_rule(AlertRule(
            name="kill_switch_triggered",
            alert_type="kill_switch",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="kill_switch",
                ),
            ],
            severity=AlertSeverity.CRITICAL,
            actions={AlertAction.NOTIFY, AlertAction.ESCALATE},
        ))

        # Process event
        triggers = engine.process_event(event_data)

        for trigger in triggers:
            handle_alert(trigger)

    MONITORING:
        - Kill switch alerts
        - Broker errors burst
        - Data feed invalid
        - Order spam detection
    """

    def __init__(
        self,
        on_trigger: Optional[Callable[[AlertTrigger], None]] = None,
    ):
        """
        Initialize engine.

        Args:
            on_trigger: Callback when alert triggers
        """
        self._on_trigger = on_trigger
        self._rules: Dict[str, AlertRule] = {}
        self._triggers: List[AlertTrigger] = []
        self._event_counts: Dict[str, List[datetime]] = {}  # For rate limiting
        self._lock = threading.Lock()

        # Initialize built-in rules
        self._init_builtin_rules()

    def _init_builtin_rules(self) -> None:
        """Initialize built-in alert rules."""
        # Kill switch triggered
        self.add_rule(AlertRule(
            name="kill_switch_triggered",
            description="Alert when kill switch is triggered",
            alert_type="kill_switch",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="kill_switch",
                ),
            ],
            severity=AlertSeverity.CRITICAL,
            actions={AlertAction.NOTIFY, AlertAction.ESCALATE},
            throttle_seconds=60,  # Alert once per minute
        ))

        # Broker errors burst
        self.add_rule(AlertRule(
            name="broker_errors_burst",
            description="Alert on burst of broker errors",
            alert_type="broker_errors",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="broker_error",
                ),
                AlertCondition(
                    type=AlertConditionType.RATE,
                    field="error_rate",
                    operator=">",
                    threshold=5,  # More than 5 errors
                    window_seconds=60,  # Per minute
                ),
            ],
            severity=AlertSeverity.ERROR,
            actions={AlertAction.NOTIFY},
            throttle_seconds=300,
        ))

        # Data feed invalid
        self.add_rule(AlertRule(
            name="data_feed_invalid",
            description="Alert when data feed is invalid",
            alert_type="data_feed",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="data_feed_invalid",
                ),
            ],
            severity=AlertSeverity.ERROR,
            actions={AlertAction.NOTIFY},
            throttle_seconds=120,
        ))

        # Order spam detection
        self.add_rule(AlertRule(
            name="order_spam_detected",
            description="Alert on excessive order rate",
            alert_type="order_spam",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="order_submitted",
                ),
                AlertCondition(
                    type=AlertConditionType.RATE,
                    field="order_rate",
                    operator=">",
                    threshold=100,  # More than 100 orders
                    window_seconds=60,  # Per minute
                ),
            ],
            severity=AlertSeverity.WARNING,
            actions={AlertAction.NOTIFY},
            throttle_seconds=300,
        ))

        # Agent offline
        self.add_rule(AlertRule(
            name="agent_offline",
            description="Alert when agent goes offline",
            alert_type="agent_status",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.EQUALS,
                    field="event_type",
                    threshold="agent_offline",
                ),
            ],
            severity=AlertSeverity.CRITICAL,
            actions={AlertAction.NOTIFY},
            throttle_seconds=60,
        ))

        # High error rate
        self.add_rule(AlertRule(
            name="high_error_rate",
            description="Alert on high error rate",
            alert_type="error_rate",
            conditions=[
                AlertCondition(
                    type=AlertConditionType.THRESHOLD,
                    field="error_count",
                    operator=">",
                    threshold=10,
                ),
            ],
            severity=AlertSeverity.WARNING,
            actions={AlertAction.NOTIFY},
            throttle_seconds=300,
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        with self._lock:
            self._rules[rule.id] = rule

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def get_rules(self, workspace_id: Optional[str] = None) -> List[AlertRule]:
        """Get all rules, optionally filtered by workspace."""
        with self._lock:
            rules = list(self._rules.values())
            if workspace_id:
                rules = [r for r in rules if not r.workspace_id or r.workspace_id == workspace_id]
            return rules

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[AlertRule]:
        """Update existing rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if not rule:
                return None

            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            rule.updated_at = datetime.utcnow()
            return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def process_event(
        self,
        event: Dict[str, Any],
        workspace_id: str = "",
        agent_id: Optional[str] = None,
    ) -> List[AlertTrigger]:
        """
        Process event against all rules.

        Args:
            event: Event to process
            workspace_id: Workspace context
            agent_id: Agent context

        Returns:
            List of triggered alerts
        """
        triggers = []

        with self._lock:
            for rule in self._rules.values():
                # Skip disabled rules
                if not rule.enabled:
                    continue

                # Check workspace scope
                if rule.workspace_id and rule.workspace_id != workspace_id:
                    continue

                # Check throttling
                if rule.is_throttled():
                    continue

                # Evaluate conditions
                if rule.evaluate(event):
                    trigger = self._create_trigger(rule, event, workspace_id, agent_id)
                    triggers.append(trigger)

                    # Update rule stats
                    rule.last_triggered_at = datetime.utcnow()
                    rule.trigger_count += 1

                    # Store trigger
                    self._triggers.append(trigger)

                    # Execute callback
                    if self._on_trigger:
                        try:
                            self._on_trigger(trigger)
                        except Exception:
                            pass

        return triggers

    def _create_trigger(
        self,
        rule: AlertRule,
        event: Dict[str, Any],
        workspace_id: str,
        agent_id: Optional[str],
    ) -> AlertTrigger:
        """Create alert trigger."""
        return AlertTrigger(
            rule_id=rule.id,
            rule_name=rule.name,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=f"{rule.name}: {rule.description}",
            context=event,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

    def acknowledge_trigger(
        self,
        trigger_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge alert trigger."""
        with self._lock:
            for trigger in self._triggers:
                if trigger.id == trigger_id:
                    trigger.acknowledged = True
                    trigger.acknowledged_by = acknowledged_by
                    trigger.acknowledged_at = datetime.utcnow()
                    return True
            return False

    def resolve_trigger(self, trigger_id: str) -> bool:
        """Resolve alert trigger."""
        with self._lock:
            for trigger in self._triggers:
                if trigger.id == trigger_id:
                    trigger.resolved = True
                    trigger.resolved_at = datetime.utcnow()
                    return True
            return False

    def get_triggers(
        self,
        workspace_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> List[AlertTrigger]:
        """Get alert triggers."""
        with self._lock:
            triggers = list(self._triggers)

            if workspace_id:
                triggers = [t for t in triggers if t.workspace_id == workspace_id]
            if severity:
                triggers = [t for t in triggers if t.severity == severity]
            if unresolved_only:
                triggers = [t for t in triggers if not t.resolved]

            return sorted(triggers, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_trigger_stats(self, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """Get trigger statistics."""
        with self._lock:
            triggers = list(self._triggers)
            if workspace_id:
                triggers = [t for t in triggers if t.workspace_id == workspace_id]

            return {
                "total": len(triggers),
                "unresolved": len([t for t in triggers if not t.resolved]),
                "unacknowledged": len([t for t in triggers if not t.acknowledged]),
                "by_severity": {
                    "critical": len([t for t in triggers if t.severity == AlertSeverity.CRITICAL]),
                    "error": len([t for t in triggers if t.severity == AlertSeverity.ERROR]),
                    "warning": len([t for t in triggers if t.severity == AlertSeverity.WARNING]),
                    "info": len([t for t in triggers if t.severity == AlertSeverity.INFO]),
                },
                "by_type": dict(self._count_by_field(triggers, "alert_type")),
            }

    def _count_by_field(
        self,
        triggers: List[AlertTrigger],
        field: str,
    ) -> Dict[str, int]:
        """Count triggers by field value."""
        counts: Dict[str, int] = {}
        for trigger in triggers:
            value = getattr(trigger, field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def cleanup_old_triggers(self, max_age_days: int = 30) -> int:
        """Clean up old triggers."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        with self._lock:
            before = len(self._triggers)
            self._triggers = [t for t in self._triggers if t.timestamp > cutoff]
            return before - len(self._triggers)
