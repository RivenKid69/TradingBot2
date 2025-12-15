# -*- coding: utf-8 -*-
"""
Telemetry Service - Stores and manages agent telemetry events.

Design Doc Phase 5: Agent telemetry storage in Cloud.

This service handles:
- Agent telemetry event storage
- Event querying and retrieval
- Retention policy enforcement
- Aggregation for dashboards

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, insert, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class TelemetryService:
    """
    Service for storing and managing agent telemetry events.

    Design Doc Phase 5:
    - Stores telemetry events from agents
    - Applies retention policies
    - Provides query interface for dashboards
    - Enforces workspace isolation

    Usage:
        service = TelemetryService(session)

        # Store agent event
        await service.store_agent_event(
            agent_id=agent_id,
            workspace_id=workspace_id,
            event_data={"event_type": "HEARTBEAT", ...}
        )

        # Query events
        events = await service.query_events(
            workspace_id=workspace_id,
            event_type="HEARTBEAT",
            limit=100,
        )
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize telemetry service.

        Args:
            session: Database session
        """
        self._session = session
        self._batch_buffer: List[Dict[str, Any]] = []
        self._batch_size = 100

    async def store_agent_event(
        self,
        agent_id: UUID,
        workspace_id: UUID,
        event_data: Dict[str, Any],
    ) -> str:
        """
        Store a single agent telemetry event.

        Args:
            agent_id: Agent identifier
            workspace_id: Workspace identifier
            event_data: Event data dictionary

        Returns:
            Event ID
        """
        from ..models import TelemetryEvent

        event_id = event_data.get("event_id", str(UUID()))
        event_type = event_data.get("event_type", "UNKNOWN")
        timestamp_str = event_data.get("timestamp")

        # Parse timestamp
        if timestamp_str:
            try:
                event_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                event_timestamp = datetime.now(timezone.utc)
        else:
            event_timestamp = datetime.now(timezone.utc)

        try:
            # Try to use database model
            event = TelemetryEvent(
                id=UUID(event_id) if isinstance(event_id, str) and len(event_id) == 36 else UUID(),
                workspace_id=workspace_id,
                agent_id=agent_id,
                event_type=event_type,
                event_data=event_data,
                timestamp=event_timestamp,
                level=event_data.get("level", "aggregated"),
                run_id=UUID(event_data["run_id"]) if event_data.get("run_id") else None,
            )
            self._session.add(event)
            await self._session.commit()

            logger.debug(
                "Stored telemetry event",
                extra={
                    "event_id": event_id,
                    "event_type": event_type,
                    "agent_id": str(agent_id),
                }
            )
            return event_id

        except Exception as e:
            # If database model not available, log the event
            logger.info(
                "Telemetry event received (no DB model)",
                extra={
                    "event_id": event_id,
                    "event_type": event_type,
                    "agent_id": str(agent_id),
                    "workspace_id": str(workspace_id),
                    "data": event_data,
                }
            )
            return event_id

    async def store_events_batch(
        self,
        agent_id: UUID,
        workspace_id: UUID,
        events: List[Dict[str, Any]],
    ) -> int:
        """
        Store multiple agent telemetry events.

        Args:
            agent_id: Agent identifier
            workspace_id: Workspace identifier
            events: List of event data dictionaries

        Returns:
            Number of events stored
        """
        stored = 0
        for event_data in events:
            try:
                await self.store_agent_event(agent_id, workspace_id, event_data)
                stored += 1
            except Exception as e:
                logger.warning(
                    f"Failed to store telemetry event: {e}",
                    extra={"event_id": event_data.get("event_id")},
                )
        return stored

    async def query_events(
        self,
        workspace_id: UUID,
        event_type: Optional[str] = None,
        agent_id: Optional[UUID] = None,
        run_id: Optional[UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query telemetry events with filters.

        Args:
            workspace_id: Workspace identifier (required for isolation)
            event_type: Filter by event type
            agent_id: Filter by agent
            run_id: Filter by run
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum events to return
            offset: Offset for pagination

        Returns:
            List of event dictionaries
        """
        try:
            from ..models import TelemetryEvent

            query = select(TelemetryEvent).where(
                TelemetryEvent.workspace_id == workspace_id
            )

            if event_type:
                query = query.where(TelemetryEvent.event_type == event_type)
            if agent_id:
                query = query.where(TelemetryEvent.agent_id == agent_id)
            if run_id:
                query = query.where(TelemetryEvent.run_id == run_id)
            if start_time:
                query = query.where(TelemetryEvent.timestamp >= start_time)
            if end_time:
                query = query.where(TelemetryEvent.timestamp <= end_time)

            query = query.order_by(TelemetryEvent.timestamp.desc())
            query = query.limit(limit).offset(offset)

            result = await self._session.execute(query)
            events = result.scalars().all()

            return [
                {
                    "event_id": str(e.id),
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "agent_id": str(e.agent_id) if e.agent_id else None,
                    "run_id": str(e.run_id) if e.run_id else None,
                    "level": e.level,
                    "data": e.event_data,
                }
                for e in events
            ]

        except Exception as e:
            logger.warning(f"Query telemetry events failed (model may not exist): {e}")
            return []

    async def get_event_counts(
        self,
        workspace_id: UUID,
        group_by: str = "event_type",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Get event counts grouped by field.

        Args:
            workspace_id: Workspace identifier
            group_by: Field to group by (event_type, agent_id, level)
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            Dictionary of counts by group
        """
        try:
            from ..models import TelemetryEvent

            if group_by == "event_type":
                group_col = TelemetryEvent.event_type
            elif group_by == "agent_id":
                group_col = TelemetryEvent.agent_id
            elif group_by == "level":
                group_col = TelemetryEvent.level
            else:
                group_col = TelemetryEvent.event_type

            query = select(group_col, func.count()).where(
                TelemetryEvent.workspace_id == workspace_id
            ).group_by(group_col)

            if start_time:
                query = query.where(TelemetryEvent.timestamp >= start_time)
            if end_time:
                query = query.where(TelemetryEvent.timestamp <= end_time)

            result = await self._session.execute(query)
            rows = result.fetchall()

            return {str(row[0]): row[1] for row in rows}

        except Exception as e:
            logger.warning(f"Get event counts failed: {e}")
            return {}

    async def delete_old_events(
        self,
        workspace_id: UUID,
        before: datetime,
    ) -> int:
        """
        Delete events older than specified time.

        Args:
            workspace_id: Workspace identifier
            before: Delete events before this time

        Returns:
            Number of events deleted
        """
        try:
            from ..models import TelemetryEvent

            result = await self._session.execute(
                delete(TelemetryEvent).where(
                    TelemetryEvent.workspace_id == workspace_id,
                    TelemetryEvent.timestamp < before,
                )
            )
            await self._session.commit()

            deleted = result.rowcount
            logger.info(
                f"Deleted {deleted} old telemetry events",
                extra={
                    "workspace_id": str(workspace_id),
                    "before": before.isoformat(),
                }
            )
            return deleted

        except Exception as e:
            logger.warning(f"Delete old events failed: {e}")
            return 0
