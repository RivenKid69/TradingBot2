# -*- coding: utf-8 -*-
"""
Telemetry Contracts.

Defines telemetry data structures shared between Cloud and Agent.
Telemetry is collected by Agent and sent to Cloud for monitoring.

IMPORTANT: All telemetry MUST pass through redaction middleware before
being sent to Cloud. Sensitive data (API keys, account IDs, etc.)
must NEVER appear in telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional
from uuid import UUID, uuid4


class TelemetryLevel(str, Enum):
    """
    Telemetry detail level.

    From Design Doc 13.1:
    - AGGREGATED: Default for retail, minimal data
    - DETAILED_NON_SENSITIVE: More detail, still safe
    - RAW_ORDER_EVENTS: Enterprise only, opt-in
    """

    AGGREGATED = "aggregated"
    DETAILED_NON_SENSITIVE = "detailed_non_sensitive"
    RAW_ORDER_EVENTS = "raw_order_events"


class MetricType(str, Enum):
    """Type of telemetry metric."""

    # Performance metrics
    PNL = "pnl"
    RETURN = "return"
    SHARPE = "sharpe"
    DRAWDOWN = "drawdown"

    # Risk metrics
    EXPOSURE = "exposure"
    VAR = "var"
    POSITION_COUNT = "position_count"

    # Execution metrics
    ORDER_COUNT = "order_count"
    FILL_RATE = "fill_rate"
    SLIPPAGE = "slippage"
    LATENCY = "latency"

    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ERROR_COUNT = "error_count"

    # State metrics
    RUN_STATE = "run_state"
    HEARTBEAT = "heartbeat"


class EventType(str, Enum):
    """Type of telemetry event."""

    # Lifecycle events
    HEARTBEAT = "heartbeat"
    RUN_STARTED = "run_started"
    RUN_STOPPED = "run_stopped"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"

    # Trading events (aggregated, not individual orders)
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_ADJUSTED = "position_adjusted"

    # Risk events
    RISK_LIMIT_HIT = "risk_limit_hit"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    DRAWDOWN_WARNING = "drawdown_warning"

    # System events
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"
    CONFIG_UPDATED = "config_updated"
    ARTIFACT_UPGRADED = "artifact_upgraded"

    # Approval events
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


# Fields that must NEVER appear in telemetry
PROHIBITED_TELEMETRY_FIELDS: Final[frozenset] = frozenset(
    [
        "api_key",
        "api_secret",
        "secret_key",
        "private_key",
        "password",
        "token",
        "credential",
        "auth",
        "broker_key",
        "account_number",
        "ssn",
        "tax_id",
    ]
)


@dataclass
class TelemetryMetric:
    """
    Single telemetry metric.

    Represents a point-in-time measurement.
    """

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class TelemetryEvent:
    """
    Telemetry event for Cloud monitoring.

    Events are discrete occurrences with associated data.
    All data MUST be redacted before sending to Cloud.
    """

    # Event identification
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Source identification (safe, non-sensitive)
    agent_id: str = ""
    run_id: str = ""
    strategy_id: str = ""

    # Event data (must be redacted)
    data: Dict[str, Any] = field(default_factory=dict)

    # Metrics (optional)
    metrics: List[TelemetryMetric] = field(default_factory=list)

    # Severity
    severity: str = "info"  # debug, info, warning, error, critical

    # Telemetry level
    level: TelemetryLevel = TelemetryLevel.AGGREGATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "strategy_id": self.strategy_id,
            "data": self.data,
            "metrics": [m.to_dict() for m in self.metrics],
            "severity": self.severity,
            "level": self.level.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TelemetryEvent:
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]) if "event_id" in data else uuid4(),
            event_type=EventType(data.get("event_type", "heartbeat")),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.utcnow(),
            agent_id=data.get("agent_id", ""),
            run_id=data.get("run_id", ""),
            strategy_id=data.get("strategy_id", ""),
            data=data.get("data", {}),
            metrics=[],  # Simplified - metrics parsing can be added
            severity=data.get("severity", "info"),
            level=TelemetryLevel(data.get("level", "aggregated")),
        )

    def contains_prohibited_fields(self) -> List[str]:
        """Check if event contains prohibited fields."""
        violations = []
        self._check_dict_for_prohibited(self.data, violations, "data")
        return violations

    def _check_dict_for_prohibited(
        self, obj: Any, violations: List[str], path: str
    ) -> None:
        """Recursively check for prohibited fields."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in PROHIBITED_TELEMETRY_FIELDS:
                    violations.append(f"{path}.{key}")
                self._check_dict_for_prohibited(value, violations, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._check_dict_for_prohibited(item, violations, f"{path}[{i}]")


@dataclass
class TelemetryBatch:
    """
    Batch of telemetry events for efficient transmission.
    """

    events: List[TelemetryEvent] = field(default_factory=list)
    batch_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    level: TelemetryLevel = TelemetryLevel.AGGREGATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": str(self.batch_id),
            "created_at": self.created_at.isoformat(),
            "agent_id": self.agent_id,
            "level": self.level.value,
            "events": [e.to_dict() for e in self.events],
            "event_count": len(self.events),
        }

    def add_event(self, event: TelemetryEvent) -> None:
        """Add event to batch."""
        self.events.append(event)
