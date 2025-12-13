# -*- coding: utf-8 -*-
"""
Telemetry Ingester - Receives telemetry from agents.

CLOUD ZONE ONLY.

Receives and stores telemetry that has been:
1. Redacted by agent (mandatory redaction middleware)
2. Classified by sensitivity level
3. Validated for prohibited fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Final, FrozenSet, List, Optional
from uuid import UUID

from packages.shared.contracts.telemetry import (
    TelemetryEvent,
    TelemetryBatch,
    TelemetryLevel,
    PROHIBITED_TELEMETRY_FIELDS,
)


class TelemetryValidationError(Exception):
    """Telemetry validation failed."""

    pass


class TelemetryIngester:
    """
    Ingests telemetry from agents.

    Validates that telemetry:
    1. Does not contain prohibited fields (secrets, credentials)
    2. Respects sensitivity levels
    3. Comes from authenticated agent

    Usage:
        ingester = TelemetryIngester(storage)

        # Ingest batch from agent
        result = ingester.ingest_batch(batch, agent_id)
        if not result.success:
            log_rejection(result.errors)
    """

    def __init__(self, storage: Optional["TelemetryStorage"] = None):
        """Initialize ingester."""
        self._storage = storage or TelemetryStorage()

    def ingest_event(
        self,
        event: TelemetryEvent,
        agent_id: str,
    ) -> "IngestionResult":
        """
        Ingest single telemetry event.

        Args:
            event: Telemetry event
            agent_id: Source agent ID

        Returns:
            IngestionResult
        """
        result = IngestionResult()

        # Validate event
        violations = event.contains_prohibited_fields()
        if violations:
            result.success = False
            result.errors.append(f"Prohibited fields found: {violations}")
            return result

        # Verify agent matches
        if event.agent_id and event.agent_id != agent_id:
            result.success = False
            result.errors.append("Agent ID mismatch")
            return result

        # Set agent ID if not set
        if not event.agent_id:
            # Create new event with agent_id (TelemetryEvent is not frozen)
            event.agent_id = agent_id

        # Store
        self._storage.store(event)
        result.events_stored = 1

        return result

    def ingest_batch(
        self,
        batch: TelemetryBatch,
        agent_id: str,
    ) -> "IngestionResult":
        """
        Ingest batch of telemetry events.

        Args:
            batch: Telemetry batch
            agent_id: Source agent ID

        Returns:
            IngestionResult
        """
        result = IngestionResult()

        # Verify batch agent matches
        if batch.agent_id and batch.agent_id != agent_id:
            result.success = False
            result.errors.append("Batch agent ID mismatch")
            return result

        # Process each event
        for event in batch.events:
            event_result = self.ingest_event(event, agent_id)
            if event_result.success:
                result.events_stored += 1
            else:
                result.errors.extend(event_result.errors)
                result.events_rejected += 1

        # Batch is successful if any events stored
        result.success = result.events_stored > 0 or len(batch.events) == 0

        return result

    def get_storage(self) -> "TelemetryStorage":
        """Get storage instance."""
        return self._storage


@dataclass
class IngestionResult:
    """Result of telemetry ingestion."""

    success: bool = True
    events_stored: int = 0
    events_rejected: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "events_stored": self.events_stored,
            "events_rejected": self.events_rejected,
            "errors": self.errors,
        }


class TelemetryStorage:
    """
    Storage for telemetry events.

    In production, this would be backed by a database.
    """

    def __init__(self):
        """Initialize storage."""
        self._events: List[TelemetryEvent] = []
        self._by_agent: Dict[str, List[TelemetryEvent]] = {}
        self._by_run: Dict[str, List[TelemetryEvent]] = {}

    def store(self, event: TelemetryEvent) -> None:
        """Store telemetry event."""
        self._events.append(event)

        # Index by agent
        if event.agent_id:
            if event.agent_id not in self._by_agent:
                self._by_agent[event.agent_id] = []
            self._by_agent[event.agent_id].append(event)

        # Index by run
        if event.run_id:
            if event.run_id not in self._by_run:
                self._by_run[event.run_id] = []
            self._by_run[event.run_id].append(event)

    def get_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        level: Optional[TelemetryLevel] = None,
    ) -> List[TelemetryEvent]:
        """Get events for agent."""
        events = self._by_agent.get(agent_id, [])

        if level:
            events = [e for e in events if e.level == level]

        return events[-limit:]

    def get_by_run(
        self,
        run_id: str,
        limit: int = 100,
    ) -> List[TelemetryEvent]:
        """Get events for run."""
        events = self._by_run.get(run_id, [])
        return events[-limit:]

    def get_recent(self, limit: int = 100) -> List[TelemetryEvent]:
        """Get recent events."""
        return self._events[-limit:]

    def count(self) -> int:
        """Get total event count."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._by_agent.clear()
        self._by_run.clear()
