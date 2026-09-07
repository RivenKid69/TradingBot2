# -*- coding: utf-8 -*-
"""
CCEA Telemetry Collector.

Collects telemetry events with mandatory redaction.
Supports multiple telemetry levels per Design Doc.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from ccea.telemetry.redaction import RedactionMiddleware, redact_data


logger = logging.getLogger(__name__)


class TelemetryLevel(str, Enum):
    """
    Telemetry detail levels.

    AGGREGATED: Default for retail - no trade details
    DETAILED_NON_SENSITIVE: Opt-in - trade counts, timing
    RAW_ORDER_EVENTS: Enterprise-only - full order details
    """

    AGGREGATED = "AGGREGATED"
    DETAILED_NON_SENSITIVE = "DETAILED_NON_SENSITIVE"
    RAW_ORDER_EVENTS = "RAW_ORDER_EVENTS"


class EventType(str, Enum):
    """Standard event types."""

    # System events
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_STOPPED = "AGENT_STOPPED"
    HEARTBEAT = "HEARTBEAT"

    # Run events
    RUN_STARTED = "RUN_STARTED"
    RUN_STOPPED = "RUN_STOPPED"
    RUN_PAUSED = "RUN_PAUSED"
    RUN_HALTED = "RUN_HALTED"

    # Strategy events
    STRATEGY_ITERATION = "STRATEGY_ITERATION"
    STRATEGY_ERROR = "STRATEGY_ERROR"

    # Performance events (AGGREGATED level)
    PERFORMANCE_SUMMARY = "PERFORMANCE_SUMMARY"
    RISK_METRICS = "RISK_METRICS"

    # Detailed events (DETAILED_NON_SENSITIVE level)
    TRADE_COUNT = "TRADE_COUNT"
    LATENCY_METRICS = "LATENCY_METRICS"

    # Raw events (RAW_ORDER_EVENTS level - enterprise only)
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"


@dataclass
class TelemetryEvent:
    """
    Telemetry event.

    All events are redacted before storage/export.
    """

    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    level: TelemetryLevel = TelemetryLevel.AGGREGATED
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "level": self.level.value,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryEvent":
        """Create from dictionary."""
        return cls(
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            level=TelemetryLevel(data.get("level", "AGGREGATED")),
            agent_id=data.get("agent_id"),
            run_id=data.get("run_id"),
            sequence=data.get("sequence", 0),
        )


class TelemetryCollector:
    """
    Telemetry event collector.

    Collects events with MANDATORY redaction.
    """

    def __init__(
        self,
        agent_id: str,
        level: TelemetryLevel = TelemetryLevel.AGGREGATED,
        buffer_size: int = 1000,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize collector.

        Args:
            agent_id: Agent identifier
            level: Maximum telemetry level to collect
            buffer_size: Event buffer size
            persistence_path: Optional path for durable storage
        """
        self.agent_id = agent_id
        self.level = level
        self.buffer_size = buffer_size
        self.persistence_path = persistence_path

        # MANDATORY redaction middleware
        self._redaction = RedactionMiddleware()

        # Event buffer (thread-safe deque)
        self._events: Deque[TelemetryEvent] = deque(maxlen=buffer_size)
        self._sequence = 0
        self._lock = threading.Lock()

        # Current run context
        self._current_run_id: Optional[str] = None

        # Callbacks for event handlers
        self._callbacks: List[Callable[[TelemetryEvent], None]] = []

        # Load persisted events if available
        if persistence_path:
            self._load_persisted()

    @property
    def redaction_applied(self) -> bool:
        """
        Check if redaction is applied.

        Always returns True - redaction cannot be disabled.
        """
        return True

    def set_run_id(self, run_id: str) -> None:
        """Set current run ID for events."""
        self._current_run_id = run_id

    def clear_run_id(self) -> None:
        """Clear current run ID."""
        self._current_run_id = None

    def add_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Add event callback."""
        self._callbacks.append(callback)

    def collect(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        level: Optional[TelemetryLevel] = None,
        run_id: Optional[str] = None,
    ) -> TelemetryEvent:
        """
        Collect a telemetry event.

        Data is ALWAYS redacted before storage.

        Args:
            event_type: Event type
            data: Event data
            level: Event level (default: collector level)
            run_id: Run ID (default: current run)

        Returns:
            Redacted TelemetryEvent
        """
        event_level = level or TelemetryLevel.AGGREGATED

        # Skip if level exceeds collector level
        if self._level_value(event_level) > self._level_value(self.level):
            # Return a placeholder event without storing
            return TelemetryEvent(
                event_type=event_type,
                level=event_level,
                agent_id=self.agent_id,
            )

        # MANDATORY: Redact data
        redacted_data = self._redaction.process(data or {})

        with self._lock:
            self._sequence += 1
            event = TelemetryEvent(
                event_type=event_type,
                data=redacted_data,
                level=event_level,
                agent_id=self.agent_id,
                run_id=run_id or self._current_run_id,
                sequence=self._sequence,
            )

            self._events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass

        # Persist if enabled
        if self.persistence_path:
            self._persist_event(event)

        return event

    def collect_performance(
        self,
        pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        drawdown: Optional[float] = None,
        sharpe: Optional[float] = None,
    ) -> TelemetryEvent:
        """Collect performance summary (AGGREGATED level)."""
        return self.collect(
            event_type=EventType.PERFORMANCE_SUMMARY.value,
            data={
                "pnl_pct": pnl,
                "win_rate_pct": win_rate,
                "max_drawdown_pct": drawdown,
                "sharpe_ratio": sharpe,
            },
            level=TelemetryLevel.AGGREGATED,
        )

    def collect_trade_count(
        self,
        total_trades: int,
        buys: int,
        sells: int,
    ) -> TelemetryEvent:
        """Collect trade counts (DETAILED level)."""
        return self.collect(
            event_type=EventType.TRADE_COUNT.value,
            data={
                "total_trades": total_trades,
                "buy_count": buys,
                "sell_count": sells,
            },
            level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        )

    def collect_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Collect error event."""
        return self.collect(
            event_type=EventType.STRATEGY_ERROR.value,
            data={
                "error_type": error_type,
                "message": message,  # Will be redacted
                "details": details or {},
            },
            level=TelemetryLevel.AGGREGATED,
        )

    def get_events(
        self,
        since: Optional[datetime] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[TelemetryEvent]:
        """Get collected events."""
        with self._lock:
            events = list(self._events)

        if since:
            events = [e for e in events if e.timestamp >= since]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def flush(self) -> List[TelemetryEvent]:
        """Flush and return all events."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _level_value(self, level: TelemetryLevel) -> int:
        """Get numeric value for level comparison."""
        return {
            TelemetryLevel.AGGREGATED: 1,
            TelemetryLevel.DETAILED_NON_SENSITIVE: 2,
            TelemetryLevel.RAW_ORDER_EVENTS: 3,
        }.get(level, 1)

    def _persist_event(self, event: TelemetryEvent) -> None:
        """Persist event to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            events_file = self.persistence_path / "events.jsonl"

            with open(events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")

    def _load_persisted(self) -> None:
        """Load persisted events."""
        if not self.persistence_path:
            return

        events_file = self.persistence_path / "events.jsonl"
        if not events_file.exists():
            return

        try:
            with open(events_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        event = TelemetryEvent.from_dict(data)
                        self._events.append(event)
                        self._sequence = max(self._sequence, event.sequence)
        except Exception as e:
            logger.warning(f"Failed to load persisted events: {e}")
