# -*- coding: utf-8 -*-
"""
Data Purge Scheduler for Cloud Control Plane.

Phase 2 Implementation (2.3): Auto-purge jobs based on retention policies.

This module provides:
    - PurgeScheduler: Background scheduler for automatic data purge
    - PurgeJob: Individual purge job definition
    - PurgeHistory: Track purge execution history

Design Doc Reference:
    - Phase 2.3: "Retention/Deletion механизм, не декларация: auto-purge jobs, DSAR deletion"
    - Automatic purge based on retention policies
    - Legal hold enforcement
    - DSAR deletion workflow integration

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from ..database import get_session
from ..models import (
    AccessAudit,
    Alert,
    AuditAction,
    Command,
    DataRetentionPolicy,
    TelemetryEvent,
)


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class PurgeJobStatus(str, Enum):
    """Status of a purge job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Legal hold or disabled


class PurgeDataType(str, Enum):
    """Data types that can be purged."""
    TELEMETRY_EVENTS = "telemetry_events"
    ALERTS = "alerts"
    COMMANDS = "commands"
    ACCESS_AUDITS = "access_audits"
    APPROVAL_RECORDS = "approval_records"
    CONFIG_BLOBS = "config_blobs"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PurgeJobResult:
    """Result of a single purge job execution."""
    job_id: UUID
    workspace_id: UUID
    data_type: PurgeDataType
    status: PurgeJobStatus
    records_purged: int
    started_at: datetime
    completed_at: datetime
    error_message: Optional[str] = None
    legal_hold_blocked: bool = False
    retention_days: int = 0


@dataclass
class PurgeSchedulerConfig:
    """Configuration for purge scheduler."""
    enabled: bool = True
    interval_hours: int = 24  # Run every 24 hours
    batch_size: int = 1000    # Records to delete per batch
    max_runtime_minutes: int = 60  # Max runtime per workspace
    dry_run: bool = False     # If True, don't actually delete
    notify_before_days: int = 7  # Notify before purge (if enabled)


@dataclass
class PurgeSchedulerState:
    """Current state of the scheduler."""
    is_running: bool = False
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    total_runs: int = 0
    total_records_purged: int = 0
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Purge Scheduler
# ============================================================================

class PurgeScheduler:
    """
    Background scheduler for automatic data purge.

    Runs periodically to purge data based on retention policies.
    Respects legal holds and DSAR requests.
    """

    def __init__(
        self,
        config: Optional[PurgeSchedulerConfig] = None,
        on_purge_complete: Optional[Callable[[PurgeJobResult], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ):
        """
        Initialize scheduler.

        Args:
            config: Scheduler configuration
            on_purge_complete: Callback when a purge job completes
            on_error: Callback when an error occurs
        """
        self.config = config or PurgeSchedulerConfig()
        self.on_purge_complete = on_purge_complete
        self.on_error = on_error

        self._state = PurgeSchedulerState()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Purge handlers per data type
        self._purge_handlers: Dict[PurgeDataType, Callable] = {
            PurgeDataType.TELEMETRY_EVENTS: self._purge_telemetry_events,
            PurgeDataType.ALERTS: self._purge_alerts,
            PurgeDataType.COMMANDS: self._purge_commands,
            PurgeDataType.ACCESS_AUDITS: self._purge_access_audits,
        }

    @property
    def state(self) -> PurgeSchedulerState:
        """Get current scheduler state."""
        return self._state

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._task is not None:
            logger.warning("Scheduler already running")
            return

        if not self.config.enabled:
            logger.info("Scheduler disabled in config")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Purge scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._task is None:
            return

        self._stop_event.set()
        self._task.cancel()

        try:
            await self._task
        except asyncio.CancelledError:
            pass

        self._task = None
        self._state.is_running = False
        logger.info("Purge scheduler stopped")

    async def run_once(self, workspace_id: Optional[UUID] = None) -> List[PurgeJobResult]:
        """
        Run purge for all workspaces or a specific workspace.

        Args:
            workspace_id: Optional specific workspace to purge

        Returns:
            List of purge job results
        """
        results = []
        self._state.is_running = True
        self._state.last_run_at = datetime.now(timezone.utc)

        try:
            async with get_session() as session:
                # Get all workspaces with retention policies
                if workspace_id:
                    workspace_ids = [workspace_id]
                else:
                    workspace_ids = await self._get_workspaces_with_policies(session)

                for ws_id in workspace_ids:
                    ws_results = await self._purge_workspace(session, ws_id)
                    results.extend(ws_results)

                await session.commit()

        except Exception as e:
            logger.error(f"Purge scheduler error: {e}")
            self._state.errors.append(str(e))
            if self.on_error:
                self.on_error("scheduler_run", e)

        finally:
            self._state.is_running = False
            self._state.total_runs += 1
            self._state.total_records_purged += sum(r.records_purged for r in results)

        return results

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                # Calculate next run time
                self._state.next_run_at = datetime.now(timezone.utc) + timedelta(
                    hours=self.config.interval_hours
                )

                # Run purge
                await self.run_once()

                # Wait for next interval or stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.interval_hours * 3600,
                    )
                    break  # Stop signal received
                except asyncio.TimeoutError:
                    pass  # Continue to next run

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self._state.errors.append(str(e))
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def _get_workspaces_with_policies(
        self,
        session: AsyncSession,
    ) -> List[UUID]:
        """Get all workspace IDs with retention policies."""
        result = await session.execute(
            select(DataRetentionPolicy.workspace_id).distinct()
        )
        return [row[0] for row in result.fetchall()]

    async def _purge_workspace(
        self,
        session: AsyncSession,
        workspace_id: UUID,
    ) -> List[PurgeJobResult]:
        """Purge data for a single workspace based on its policies."""
        results = []

        # Get retention policies for this workspace
        policies_result = await session.execute(
            select(DataRetentionPolicy).where(
                DataRetentionPolicy.workspace_id == workspace_id
            )
        )
        policies = list(policies_result.scalars().all())

        for policy in policies:
            job_id = uuid4()
            started_at = datetime.now(timezone.utc)

            try:
                # Check if purge is enabled
                if not policy.auto_purge_enabled:
                    results.append(PurgeJobResult(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        data_type=PurgeDataType(policy.data_type),
                        status=PurgeJobStatus.SKIPPED,
                        records_purged=0,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        legal_hold_blocked=not policy.dsar_export_enabled,
                    ))
                    continue

                # Check legal hold
                if not policy.dsar_export_enabled:
                    results.append(PurgeJobResult(
                        job_id=job_id,
                        workspace_id=workspace_id,
                        data_type=PurgeDataType(policy.data_type),
                        status=PurgeJobStatus.SKIPPED,
                        records_purged=0,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        legal_hold_blocked=True,
                    ))
                    continue

                # Get purge handler
                data_type = PurgeDataType(policy.data_type)
                handler = self._purge_handlers.get(data_type)

                if handler is None:
                    logger.warning(f"No handler for data type: {data_type}")
                    continue

                # Calculate cutoff date
                cutoff = datetime.now(timezone.utc) - timedelta(days=policy.retention_days)

                # Execute purge
                records_purged = await handler(
                    session,
                    workspace_id,
                    cutoff,
                    self.config.batch_size,
                    self.config.dry_run,
                )

                # Update last purge time
                policy.last_purge_at = datetime.now(timezone.utc)

                result = PurgeJobResult(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    data_type=data_type,
                    status=PurgeJobStatus.COMPLETED,
                    records_purged=records_purged,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    retention_days=policy.retention_days,
                )
                results.append(result)

                # Notify callback
                if self.on_purge_complete:
                    self.on_purge_complete(result)

                # Log audit
                await self._log_purge_audit(session, result)

            except Exception as e:
                logger.error(f"Purge job failed for {workspace_id}/{policy.data_type}: {e}")
                results.append(PurgeJobResult(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    data_type=PurgeDataType(policy.data_type),
                    status=PurgeJobStatus.FAILED,
                    records_purged=0,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    error_message=str(e),
                ))

                if self.on_error:
                    self.on_error(f"purge_{policy.data_type}", e)

        return results

    async def _purge_telemetry_events(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        cutoff: datetime,
        batch_size: int,
        dry_run: bool,
    ) -> int:
        """Purge old telemetry events."""
        if dry_run:
            result = await session.execute(
                select(func.count(TelemetryEvent.id)).where(
                    TelemetryEvent.workspace_id == workspace_id,
                    TelemetryEvent.created_at < cutoff,
                )
            )
            return result.scalar() or 0

        total_deleted = 0
        while True:
            # Get batch of IDs to delete
            result = await session.execute(
                select(TelemetryEvent.id).where(
                    TelemetryEvent.workspace_id == workspace_id,
                    TelemetryEvent.created_at < cutoff,
                ).limit(batch_size)
            )
            ids = [row[0] for row in result.fetchall()]

            if not ids:
                break

            # Delete batch
            await session.execute(
                delete(TelemetryEvent).where(TelemetryEvent.id.in_(ids))
            )
            total_deleted += len(ids)

            # Flush after each batch
            await session.flush()

        return total_deleted

    async def _purge_alerts(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        cutoff: datetime,
        batch_size: int,
        dry_run: bool,
    ) -> int:
        """Purge old alerts (only resolved ones)."""
        if dry_run:
            result = await session.execute(
                select(func.count(Alert.id)).where(
                    Alert.workspace_id == workspace_id,
                    Alert.created_at < cutoff,
                    Alert.resolved == True,
                )
            )
            return result.scalar() or 0

        total_deleted = 0
        while True:
            result = await session.execute(
                select(Alert.id).where(
                    Alert.workspace_id == workspace_id,
                    Alert.created_at < cutoff,
                    Alert.resolved == True,
                ).limit(batch_size)
            )
            ids = [row[0] for row in result.fetchall()]

            if not ids:
                break

            await session.execute(delete(Alert).where(Alert.id.in_(ids)))
            total_deleted += len(ids)
            await session.flush()

        return total_deleted

    async def _purge_commands(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        cutoff: datetime,
        batch_size: int,
        dry_run: bool,
    ) -> int:
        """Purge old commands (only terminal state)."""
        terminal_states = ["executed", "failed", "expired", "rejected"]

        if dry_run:
            result = await session.execute(
                select(func.count(Command.id)).where(
                    Command.workspace_id == workspace_id,
                    Command.created_at < cutoff,
                    Command.status.in_(terminal_states),
                )
            )
            return result.scalar() or 0

        total_deleted = 0
        while True:
            result = await session.execute(
                select(Command.id).where(
                    Command.workspace_id == workspace_id,
                    Command.created_at < cutoff,
                    Command.status.in_(terminal_states),
                ).limit(batch_size)
            )
            ids = [row[0] for row in result.fetchall()]

            if not ids:
                break

            await session.execute(delete(Command).where(Command.id.in_(ids)))
            total_deleted += len(ids)
            await session.flush()

        return total_deleted

    async def _purge_access_audits(
        self,
        session: AsyncSession,
        workspace_id: UUID,
        cutoff: datetime,
        batch_size: int,
        dry_run: bool,
    ) -> int:
        """Purge old access audits (7 year minimum for compliance)."""
        # Enforce 7 year minimum for audits
        min_retention = datetime.now(timezone.utc) - timedelta(days=2555)
        effective_cutoff = min(cutoff, min_retention)

        if dry_run:
            result = await session.execute(
                select(func.count(AccessAudit.id)).where(
                    AccessAudit.workspace_id == workspace_id,
                    AccessAudit.created_at < effective_cutoff,
                )
            )
            return result.scalar() or 0

        total_deleted = 0
        while True:
            result = await session.execute(
                select(AccessAudit.id).where(
                    AccessAudit.workspace_id == workspace_id,
                    AccessAudit.created_at < effective_cutoff,
                ).limit(batch_size)
            )
            ids = [row[0] for row in result.fetchall()]

            if not ids:
                break

            await session.execute(delete(AccessAudit).where(AccessAudit.id.in_(ids)))
            total_deleted += len(ids)
            await session.flush()

        return total_deleted

    async def _log_purge_audit(
        self,
        session: AsyncSession,
        result: PurgeJobResult,
    ) -> None:
        """Log audit entry for purge operation."""
        audit = AccessAudit(
            workspace_id=result.workspace_id,
            actor_type="system",
            action=AuditAction.DELETE.value,
            resource_type=result.data_type.value,
            resource_id=str(result.job_id),
            success=result.status == PurgeJobStatus.COMPLETED,
            error_message=result.error_message,
            context={
                "records_purged": result.records_purged,
                "retention_days": result.retention_days,
                "legal_hold_blocked": result.legal_hold_blocked,
                "duration_seconds": (
                    result.completed_at - result.started_at
                ).total_seconds(),
            },
        )
        session.add(audit)


# ============================================================================
# DSAR Deletion Integration
# ============================================================================

class DSARDeletionWorkflow:
    """
    DSAR deletion workflow integration.

    Handles GDPR Article 17 (Right to Erasure) requests.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def execute_erasure(
        self,
        user_id: UUID,
        workspace_id: UUID,
        data_categories: Set[str],
        skip_legal_hold: bool = False,
    ) -> Dict[str, int]:
        """
        Execute DSAR erasure request.

        Args:
            user_id: User whose data should be deleted
            workspace_id: Workspace scope
            data_categories: Categories to delete
            skip_legal_hold: If True, delete even if legal hold (requires admin)

        Returns:
            Dict of data_type -> records_deleted
        """
        results = {}

        # Check legal hold unless explicitly overridden
        if not skip_legal_hold:
            legal_holds = await self._get_legal_holds(workspace_id, data_categories)
            if legal_holds:
                logger.warning(f"DSAR deletion blocked by legal hold on: {legal_holds}")
                return {dt: 0 for dt in data_categories}

        # Delete telemetry events
        if "telemetry_events" in data_categories:
            result = await self.session.execute(
                delete(TelemetryEvent).where(
                    TelemetryEvent.workspace_id == workspace_id,
                )
            )
            results["telemetry_events"] = result.rowcount

        # Delete alerts
        if "alerts" in data_categories:
            result = await self.session.execute(
                delete(Alert).where(
                    Alert.workspace_id == workspace_id,
                )
            )
            results["alerts"] = result.rowcount

        # Delete commands
        if "commands" in data_categories:
            result = await self.session.execute(
                delete(Command).where(
                    Command.workspace_id == workspace_id,
                )
            )
            results["commands"] = result.rowcount

        # Log audit
        await self._log_dsar_audit(user_id, workspace_id, results)

        await self.session.commit()
        return results

    async def _get_legal_holds(
        self,
        workspace_id: UUID,
        data_categories: Set[str],
    ) -> Set[str]:
        """Get data types under legal hold."""
        result = await self.session.execute(
            select(DataRetentionPolicy).where(
                DataRetentionPolicy.workspace_id == workspace_id,
                DataRetentionPolicy.data_type.in_(data_categories),
                DataRetentionPolicy.dsar_export_enabled == False,
            )
        )
        policies = result.scalars().all()
        return {p.data_type for p in policies}

    async def _log_dsar_audit(
        self,
        user_id: UUID,
        workspace_id: UUID,
        results: Dict[str, int],
    ) -> None:
        """Log DSAR deletion audit."""
        audit = AccessAudit(
            workspace_id=workspace_id,
            user_id=user_id,
            actor_type="system",
            action=AuditAction.DELETE.value,
            resource_type="dsar_erasure",
            resource_id=str(uuid4()),
            success=True,
            context={
                "deletion_results": results,
                "total_deleted": sum(results.values()),
            },
        )
        self.session.add(audit)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "PurgeJobStatus",
    "PurgeDataType",
    # Data classes
    "PurgeJobResult",
    "PurgeSchedulerConfig",
    "PurgeSchedulerState",
    # Classes
    "PurgeScheduler",
    "DSARDeletionWorkflow",
]
