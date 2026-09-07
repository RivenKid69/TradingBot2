# -*- coding: utf-8 -*-
"""
Staged Rollout Manager - Managed deployment with progressive rollout.

CLOUD ZONE ONLY.

Implements staged rollout for:
- Canary → Early Adopters → General progression
- Automatic health monitoring
- Automatic rollback on failure
- Manual promotion/demotion
- Percentage-based targeting

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 15.2: Agent updates - Staged rollout + rollback
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CANARY_PERCENTAGE: Final[float] = 5.0
DEFAULT_EARLY_ADOPTER_PERCENTAGE: Final[float] = 25.0
DEFAULT_GENERAL_PERCENTAGE: Final[float] = 100.0


class RolloutState(Enum):
    """State of a rollout."""

    CREATED = auto()
    PAUSED = auto()
    ROLLING_OUT = auto()
    COMPLETED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()
    CANCELLED = auto()


class RolloutStageState(Enum):
    """State of a rollout stage."""

    PENDING = auto()
    ACTIVE = auto()
    MONITORING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class RolloutStage:
    """Definition of a rollout stage."""

    name: str
    percentage: float
    duration_hours: float = 24.0  # Time to wait before next stage
    min_success_rate: float = 0.99  # 99% success required
    min_healthy_agents: int = 1  # Minimum agents that must be healthy
    auto_promote: bool = True  # Automatically promote to next stage
    require_approval: bool = False  # Require manual approval

    # Runtime state
    state: RolloutStageState = RolloutStageState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agents_targeted: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "percentage": self.percentage,
            "duration_hours": self.duration_hours,
            "min_success_rate": self.min_success_rate,
            "min_healthy_agents": self.min_healthy_agents,
            "auto_promote": self.auto_promote,
            "require_approval": self.require_approval,
            "state": self.state.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agents_targeted": self.agents_targeted,
            "agents_succeeded": self.agents_succeeded,
            "agents_failed": self.agents_failed,
        }

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        total = self.agents_succeeded + self.agents_failed
        if total == 0:
            return 1.0
        return self.agents_succeeded / total


@dataclass
class RolloutProgress:
    """Progress tracking for a rollout."""

    rollout_id: UUID
    current_stage: str
    current_percentage: float
    total_agents: int = 0
    targeted_agents: int = 0
    pending_agents: int = 0
    installing_agents: int = 0
    succeeded_agents: int = 0
    failed_agents: int = 0
    rolled_back_agents: int = 0

    # Health metrics
    overall_success_rate: float = 0.0
    current_stage_success_rate: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    time_in_current_stage: timedelta = timedelta()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rollout_id": str(self.rollout_id),
            "current_stage": self.current_stage,
            "current_percentage": self.current_percentage,
            "total_agents": self.total_agents,
            "targeted_agents": self.targeted_agents,
            "pending_agents": self.pending_agents,
            "installing_agents": self.installing_agents,
            "succeeded_agents": self.succeeded_agents,
            "failed_agents": self.failed_agents,
            "rolled_back_agents": self.rolled_back_agents,
            "overall_success_rate": self.overall_success_rate,
            "current_stage_success_rate": self.current_stage_success_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "estimated_completion": (
                self.estimated_completion.isoformat() if self.estimated_completion else None
            ),
            "time_in_current_stage_seconds": self.time_in_current_stage.total_seconds(),
        }


@dataclass
class Rollout:
    """A staged rollout instance."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Target
    update_id: Optional[UUID] = None
    artifact_digest: str = ""
    target_version: str = ""

    # Configuration
    stages: List[RolloutStage] = field(default_factory=list)
    rollback_on_failure: bool = True
    pause_on_failure: bool = True
    failure_threshold: float = 0.05  # 5% failures trigger action

    # State
    state: RolloutState = RolloutState.CREATED
    current_stage_index: int = 0
    current_percentage: float = 0.0

    # Targeting
    workspace_id: Optional[UUID] = None
    agent_filter: Optional[Dict[str, Any]] = None
    targeted_agent_ids: Set[UUID] = field(default_factory=set)
    excluded_agent_ids: Set[UUID] = field(default_factory=set)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None

    # Rollback info
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None
    rollback_at: Optional[datetime] = None

    # Creator
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "update_id": str(self.update_id) if self.update_id else None,
            "artifact_digest": self.artifact_digest,
            "target_version": self.target_version,
            "stages": [s.to_dict() for s in self.stages],
            "rollback_on_failure": self.rollback_on_failure,
            "pause_on_failure": self.pause_on_failure,
            "failure_threshold": self.failure_threshold,
            "state": self.state.name,
            "current_stage_index": self.current_stage_index,
            "current_percentage": self.current_percentage,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "targeted_agent_count": len(self.targeted_agent_ids),
            "excluded_agent_count": len(self.excluded_agent_ids),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "rollback_triggered": self.rollback_triggered,
            "rollback_reason": self.rollback_reason,
            "created_by": self.created_by,
        }

    @property
    def current_stage(self) -> Optional[RolloutStage]:
        """Get current stage."""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None


@dataclass
class RolloutConfig:
    """Configuration for rollout manager."""

    # Default stages
    default_stages: List[RolloutStage] = field(
        default_factory=lambda: [
            RolloutStage(
                name="canary",
                percentage=DEFAULT_CANARY_PERCENTAGE,
                duration_hours=24,
                min_success_rate=0.99,
                auto_promote=True,
            ),
            RolloutStage(
                name="early_adopters",
                percentage=DEFAULT_EARLY_ADOPTER_PERCENTAGE,
                duration_hours=48,
                min_success_rate=0.98,
                auto_promote=True,
            ),
            RolloutStage(
                name="general",
                percentage=DEFAULT_GENERAL_PERCENTAGE,
                duration_hours=0,  # No wait for final stage
                min_success_rate=0.95,
                auto_promote=False,  # Final stage
            ),
        ]
    )

    # Monitoring
    health_check_interval_seconds: int = 60
    monitor_after_completion_hours: float = 24.0

    # Rollback
    auto_rollback: bool = True
    rollback_timeout_seconds: int = 300

    # Notifications
    notify_on_stage_change: bool = True
    notify_on_failure: bool = True
    notify_on_rollback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_stages": [s.to_dict() for s in self.default_stages],
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "monitor_after_completion_hours": self.monitor_after_completion_hours,
            "auto_rollback": self.auto_rollback,
            "rollback_timeout_seconds": self.rollback_timeout_seconds,
        }


class StagedRolloutManager:
    """
    Staged Rollout Manager for progressive deployments.

    Implements phased rollout with automatic monitoring:
    - Canary (5%) → Early Adopters (25%) → General (100%)
    - Health monitoring at each stage
    - Automatic rollback on failure threshold
    - Manual promotion/demotion controls

    Usage:
        config = RolloutConfig()
        manager = StagedRolloutManager(config)

        # Create rollout
        rollout = manager.create_rollout(
            name="Agent v1.1.0 Rollout",
            update_id=update_id,
            target_version="1.1.0",
        )

        # Start rollout
        await manager.start_rollout(rollout.id)

        # Monitor progress
        progress = manager.get_progress(rollout.id)

        # Manual promotion (if needed)
        await manager.promote_stage(rollout.id)
    """

    def __init__(
        self,
        config: Optional[RolloutConfig] = None,
        db_session: Optional[Any] = None,
        on_stage_change: Optional[Callable[[Rollout, RolloutStage], None]] = None,
        on_rollback: Optional[Callable[[Rollout, str], None]] = None,
        on_complete: Optional[Callable[[Rollout], None]] = None,
    ):
        """
        Initialize rollout manager.

        Args:
            config: Rollout configuration
            db_session: Database session
            on_stage_change: Callback on stage change
            on_rollback: Callback on rollback
            on_complete: Callback on completion
        """
        self.config = config or RolloutConfig()
        self._db = db_session

        # Callbacks
        self._on_stage_change = on_stage_change
        self._on_rollback = on_rollback
        self._on_complete = on_complete

        # State
        self._rollouts: Dict[UUID, Rollout] = {}
        self._agent_assignments: Dict[UUID, UUID] = {}  # agent_id -> rollout_id
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    def create_rollout(
        self,
        name: str,
        update_id: Optional[UUID] = None,
        artifact_digest: str = "",
        target_version: str = "",
        description: str = "",
        stages: Optional[List[RolloutStage]] = None,
        workspace_id: Optional[UUID] = None,
        agent_filter: Optional[Dict[str, Any]] = None,
        exclude_agents: Optional[List[UUID]] = None,
        rollback_on_failure: bool = True,
        failure_threshold: float = 0.05,
        created_by: str = "system",
    ) -> Rollout:
        """
        Create a new rollout.

        Args:
            name: Rollout name
            update_id: Associated update ID
            artifact_digest: Artifact digest to deploy
            target_version: Target version
            description: Description
            stages: Custom stages (uses default if not provided)
            workspace_id: Target workspace
            agent_filter: Filter for agent selection
            exclude_agents: Agents to exclude
            rollback_on_failure: Enable auto rollback
            failure_threshold: Failure threshold for rollback
            created_by: Creator identifier

        Returns:
            Created rollout
        """
        # Use custom stages or defaults
        rollout_stages = stages or [
            RolloutStage(
                name=s.name,
                percentage=s.percentage,
                duration_hours=s.duration_hours,
                min_success_rate=s.min_success_rate,
                min_healthy_agents=s.min_healthy_agents,
                auto_promote=s.auto_promote,
                require_approval=s.require_approval,
            )
            for s in self.config.default_stages
        ]

        rollout = Rollout(
            name=name,
            description=description,
            update_id=update_id,
            artifact_digest=artifact_digest,
            target_version=target_version,
            stages=rollout_stages,
            rollback_on_failure=rollback_on_failure,
            failure_threshold=failure_threshold,
            workspace_id=workspace_id,
            agent_filter=agent_filter,
            excluded_agent_ids=set(exclude_agents or []),
            created_by=created_by,
        )

        self._rollouts[rollout.id] = rollout
        logger.info(f"Created rollout: {rollout.name} ({rollout.id})")

        return rollout

    async def start_rollout(self, rollout_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Start a rollout.

        Args:
            rollout_id: Rollout ID

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state not in (RolloutState.CREATED, RolloutState.PAUSED):
            return False, f"Cannot start from state: {rollout.state.name}"

        if not rollout.stages:
            return False, "No stages defined"

        # Initialize first stage
        first_stage = rollout.stages[0]
        first_stage.state = RolloutStageState.ACTIVE
        first_stage.started_at = datetime.utcnow()

        rollout.state = RolloutState.ROLLING_OUT
        rollout.started_at = datetime.utcnow()
        rollout.current_stage_index = 0
        rollout.current_percentage = first_stage.percentage
        rollout.paused_at = None

        # Start monitoring
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"Started rollout: {rollout.name} at {first_stage.percentage}% " f"({first_stage.name})"
        )

        # Notify
        if self._on_stage_change:
            try:
                self._on_stage_change(rollout, first_stage)
            except Exception as e:
                logger.error(f"Stage change callback error: {e}")

        return True, None

    async def pause_rollout(self, rollout_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Pause a rollout.

        Args:
            rollout_id: Rollout ID

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state != RolloutState.ROLLING_OUT:
            return False, f"Cannot pause from state: {rollout.state.name}"

        rollout.state = RolloutState.PAUSED
        rollout.paused_at = datetime.utcnow()

        logger.info(f"Paused rollout: {rollout.name}")
        return True, None

    async def resume_rollout(self, rollout_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Resume a paused rollout.

        Args:
            rollout_id: Rollout ID

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state != RolloutState.PAUSED:
            return False, f"Cannot resume from state: {rollout.state.name}"

        rollout.state = RolloutState.ROLLING_OUT
        rollout.paused_at = None

        logger.info(f"Resumed rollout: {rollout.name}")
        return True, None

    async def cancel_rollout(
        self,
        rollout_id: UUID,
        reason: str = "Cancelled by user",
    ) -> Tuple[bool, Optional[str]]:
        """
        Cancel a rollout.

        Args:
            rollout_id: Rollout ID
            reason: Cancellation reason

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state in (
            RolloutState.COMPLETED,
            RolloutState.ROLLED_BACK,
            RolloutState.CANCELLED,
        ):
            return False, f"Cannot cancel from state: {rollout.state.name}"

        rollout.state = RolloutState.CANCELLED
        rollout.completed_at = datetime.utcnow()

        # Mark current stage as skipped
        if rollout.current_stage:
            rollout.current_stage.state = RolloutStageState.SKIPPED

        logger.info(f"Cancelled rollout: {rollout.name} - {reason}")
        return True, None

    async def promote_stage(self, rollout_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Manually promote to next stage.

        Args:
            rollout_id: Rollout ID

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state != RolloutState.ROLLING_OUT:
            return False, f"Cannot promote from state: {rollout.state.name}"

        return await self._advance_stage(rollout)

    async def demote_stage(self, rollout_id: UUID) -> Tuple[bool, Optional[str]]:
        """
        Demote to previous stage (reduce percentage).

        Args:
            rollout_id: Rollout ID

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state != RolloutState.ROLLING_OUT:
            return False, f"Cannot demote from state: {rollout.state.name}"

        if rollout.current_stage_index <= 0:
            return False, "Already at first stage"

        # Move to previous stage
        rollout.current_stage.state = RolloutStageState.SKIPPED
        rollout.current_stage_index -= 1

        prev_stage = rollout.current_stage
        prev_stage.state = RolloutStageState.ACTIVE
        prev_stage.started_at = datetime.utcnow()
        rollout.current_percentage = prev_stage.percentage

        logger.info(
            f"Demoted rollout: {rollout.name} to {prev_stage.name} " f"({prev_stage.percentage}%)"
        )

        return True, None

    async def trigger_rollback(
        self,
        rollout_id: UUID,
        reason: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Trigger rollback for a rollout.

        Args:
            rollout_id: Rollout ID
            reason: Rollback reason

        Returns:
            (success, error_message)
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False, "Rollout not found"

        if rollout.state in (
            RolloutState.COMPLETED,
            RolloutState.ROLLED_BACK,
            RolloutState.CANCELLED,
        ):
            return False, f"Cannot rollback from state: {rollout.state.name}"

        rollout.state = RolloutState.ROLLED_BACK
        rollout.rollback_triggered = True
        rollout.rollback_reason = reason
        rollout.rollback_at = datetime.utcnow()
        rollout.completed_at = datetime.utcnow()

        # Mark current stage as failed
        if rollout.current_stage:
            rollout.current_stage.state = RolloutStageState.FAILED

        logger.warning(f"Rollback triggered for: {rollout.name} - {reason}")

        # Notify
        if self._on_rollback:
            try:
                self._on_rollback(rollout, reason)
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")

        return True, None

    def report_agent_result(
        self,
        rollout_id: UUID,
        agent_id: UUID,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Report agent update result.

        Args:
            rollout_id: Rollout ID
            agent_id: Agent ID
            success: Whether update succeeded
            error_message: Error message if failed
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout or not rollout.current_stage:
            return

        stage = rollout.current_stage

        if success:
            stage.agents_succeeded += 1
        else:
            stage.agents_failed += 1
            logger.warning(
                f"Agent {agent_id} failed update in rollout {rollout.name}: " f"{error_message}"
            )

        # Check failure threshold
        if stage.success_rate < (1 - rollout.failure_threshold):
            if rollout.rollback_on_failure:
                asyncio.create_task(
                    self.trigger_rollback(
                        rollout_id,
                        f"Failure threshold exceeded: {stage.success_rate:.1%} success rate",
                    )
                )
            elif rollout.pause_on_failure:
                asyncio.create_task(self.pause_rollout(rollout_id))

    def assign_agent_to_rollout(
        self,
        agent_id: UUID,
        rollout_id: UUID,
    ) -> bool:
        """
        Assign agent to rollout targeting.

        Args:
            agent_id: Agent ID
            rollout_id: Rollout ID

        Returns:
            True if assigned
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False

        if agent_id in rollout.excluded_agent_ids:
            return False

        rollout.targeted_agent_ids.add(agent_id)
        self._agent_assignments[agent_id] = rollout_id

        if rollout.current_stage:
            rollout.current_stage.agents_targeted += 1

        return True

    def is_agent_in_rollout(
        self,
        agent_id: UUID,
        rollout_id: UUID,
    ) -> bool:
        """
        Check if agent is targeted by rollout.

        Uses deterministic hash for percentage-based targeting.

        Args:
            agent_id: Agent ID
            rollout_id: Rollout ID

        Returns:
            True if agent should receive update
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return False

        if agent_id in rollout.excluded_agent_ids:
            return False

        # Use hash for deterministic targeting
        hash_input = f"{agent_id}:{rollout_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100

        return percentage < rollout.current_percentage

    def get_rollout(self, rollout_id: UUID) -> Optional[Rollout]:
        """Get rollout by ID."""
        return self._rollouts.get(rollout_id)

    def get_progress(self, rollout_id: UUID) -> Optional[RolloutProgress]:
        """
        Get rollout progress.

        Args:
            rollout_id: Rollout ID

        Returns:
            Progress information
        """
        rollout = self._rollouts.get(rollout_id)
        if not rollout:
            return None

        stage = rollout.current_stage

        # Calculate totals across all stages
        total_targeted = sum(s.agents_targeted for s in rollout.stages)
        total_succeeded = sum(s.agents_succeeded for s in rollout.stages)
        total_failed = sum(s.agents_failed for s in rollout.stages)

        progress = RolloutProgress(
            rollout_id=rollout_id,
            current_stage=stage.name if stage else "",
            current_percentage=rollout.current_percentage,
            total_agents=len(rollout.targeted_agent_ids),
            targeted_agents=total_targeted,
            succeeded_agents=total_succeeded,
            failed_agents=total_failed,
            started_at=rollout.started_at,
        )

        # Calculate success rates
        if total_targeted > 0:
            progress.overall_success_rate = total_succeeded / total_targeted

        if stage:
            progress.current_stage_success_rate = stage.success_rate

            if stage.started_at:
                progress.time_in_current_stage = datetime.utcnow() - stage.started_at

        return progress

    def list_rollouts(
        self,
        state: Optional[RolloutState] = None,
        workspace_id: Optional[UUID] = None,
    ) -> List[Rollout]:
        """
        List rollouts.

        Args:
            state: Filter by state
            workspace_id: Filter by workspace

        Returns:
            List of rollouts
        """
        rollouts = list(self._rollouts.values())

        if state:
            rollouts = [r for r in rollouts if r.state == state]

        if workspace_id:
            rollouts = [r for r in rollouts if r.workspace_id == workspace_id]

        return sorted(rollouts, key=lambda r: r.created_at, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get rollout statistics."""
        state_counts: Dict[str, int] = {}
        for rollout in self._rollouts.values():
            state = rollout.state.name
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_rollouts": len(self._rollouts),
            "state_counts": state_counts,
            "active_rollouts": sum(
                1 for r in self._rollouts.values() if r.state == RolloutState.ROLLING_OUT
            ),
            "assigned_agents": len(self._agent_assignments),
        }

    async def shutdown(self) -> None:
        """Shutdown manager."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    # ===== Private Methods =====

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                for rollout in self._rollouts.values():
                    if rollout.state == RolloutState.ROLLING_OUT:
                        await self._check_stage_progress(rollout)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    async def _check_stage_progress(self, rollout: Rollout) -> None:
        """Check if stage should advance."""
        stage = rollout.current_stage
        if not stage:
            return

        # Check if stage duration elapsed
        if stage.started_at and stage.duration_hours > 0:
            elapsed = datetime.utcnow() - stage.started_at
            if elapsed < timedelta(hours=stage.duration_hours):
                return

        # Check success rate
        if stage.success_rate < stage.min_success_rate:
            logger.warning(
                f"Rollout {rollout.name} stage {stage.name} has low success rate: "
                f"{stage.success_rate:.1%}"
            )
            return

        # Check minimum healthy agents
        if stage.agents_succeeded < stage.min_healthy_agents:
            return

        # Stage is complete
        stage.state = RolloutStageState.COMPLETED
        stage.completed_at = datetime.utcnow()

        # Auto-promote if configured
        if stage.auto_promote and not stage.require_approval:
            await self._advance_stage(rollout)

    async def _advance_stage(self, rollout: Rollout) -> Tuple[bool, Optional[str]]:
        """Advance to next stage."""
        next_index = rollout.current_stage_index + 1

        if next_index >= len(rollout.stages):
            # Rollout complete
            rollout.state = RolloutState.COMPLETED
            rollout.completed_at = datetime.utcnow()

            logger.info(f"Completed rollout: {rollout.name}")

            # Notify
            if self._on_complete:
                try:
                    self._on_complete(rollout)
                except Exception as e:
                    logger.error(f"Complete callback error: {e}")

            return True, None

        # Move to next stage
        rollout.current_stage_index = next_index
        next_stage = rollout.stages[next_index]
        next_stage.state = RolloutStageState.ACTIVE
        next_stage.started_at = datetime.utcnow()
        rollout.current_percentage = next_stage.percentage

        logger.info(
            f"Advanced rollout: {rollout.name} to {next_stage.name} " f"({next_stage.percentage}%)"
        )

        # Notify
        if self._on_stage_change:
            try:
                self._on_stage_change(rollout, next_stage)
            except Exception as e:
                logger.error(f"Stage change callback error: {e}")

        return True, None
