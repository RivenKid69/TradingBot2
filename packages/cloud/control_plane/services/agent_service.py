# -*- coding: utf-8 -*-
"""
Agent Lifecycle Service - Manages agent registration and health.

Phase 7 (WI-CLOUD-02): Implements agent-auth lifecycle operations.

CLOUD ZONE ONLY.

This service provides:
- Agent enrollment via tokens
- Heartbeat processing with pending command counts
- Trust state management
- Agent health tracking

References:
- Design Doc: Agent enrollment and trust lifecycle
- WI-CLOUD-02: Agent-auth lifecycle endpoints
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import (
    Agent,
    AgentEnrollmentToken,
    Command,
    CommandStatus,
    TrustState,
    Workspace,
)


# ============================================================================
# Exceptions
# ============================================================================


class AgentServiceError(Exception):
    """Base exception for agent service errors."""

    pass


class AgentNotFoundError(AgentServiceError):
    """Agent not found."""

    pass


class AgentNotEnrolledError(AgentServiceError):
    """Agent is not enrolled."""

    pass


class EnrollmentTokenError(AgentServiceError):
    """Enrollment token error."""

    pass


class TrustStateError(AgentServiceError):
    """Trust state transition error."""

    pass


# ============================================================================
# DTOs
# ============================================================================


@dataclass
class AgentEnrollmentRequest:
    """Request to enroll a new agent."""

    enrollment_token: str
    agent_name: str
    public_key: str
    agent_version: str
    capabilities: List[str]
    attestation: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None
    os_info: Optional[Dict[str, Any]] = None


@dataclass
class AgentEnrollmentResult:
    """Result of agent enrollment."""

    agent_id: UUID
    workspace_id: UUID
    organization_id: UUID
    trust_state: TrustState
    capabilities: List[str]


@dataclass
class HeartbeatRequest:
    """Agent heartbeat request."""

    agent_id: UUID
    agent_version: str
    current_state: str
    health_metrics: Dict[str, Any]
    last_run_id: Optional[UUID] = None


@dataclass
class HeartbeatResult:
    """Agent heartbeat response."""

    server_time: datetime
    trust_state: TrustState
    pending_commands: int
    next_heartbeat_sec: int


# ============================================================================
# Agent Service
# ============================================================================


class AgentService:
    """
    Agent lifecycle service - manages enrollment, heartbeat, and trust.

    Phase 7 (WI-CLOUD-02): Implements agent-auth lifecycle.

    Usage:
        service = AgentService(session)

        # Enroll new agent
        result = await service.enroll(request)

        # Process heartbeat
        hb_result = await service.heartbeat(hb_request)

        # Revoke agent
        await service.revoke(agent_id, workspace_id, reason)
    """

    # Default heartbeat interval in seconds
    DEFAULT_HEARTBEAT_INTERVAL = 60

    # Stale threshold - agent considered stale if no heartbeat for this duration
    STALE_THRESHOLD_MINUTES = 5

    def __init__(self, session: AsyncSession):
        """Initialize service with database session."""
        self.session = session

    # ========================================================================
    # Agent Enrollment
    # ========================================================================

    async def enroll(
        self,
        request: AgentEnrollmentRequest,
    ) -> AgentEnrollmentResult:
        """
        Enroll a new agent using enrollment token.

        Args:
            request: Enrollment request

        Returns:
            Enrollment result with agent details

        Raises:
            EnrollmentTokenError: If token invalid/expired/used
        """
        # Hash the provided token
        token_hash = hashlib.sha256(request.enrollment_token.encode()).hexdigest()

        # Find and validate enrollment token
        token_result = await self.session.execute(
            select(AgentEnrollmentToken).where(
                AgentEnrollmentToken.token_hash == token_hash,
                AgentEnrollmentToken.is_used == False,
                AgentEnrollmentToken.expires_at > datetime.now(timezone.utc),
            )
        )
        enrollment_token = token_result.scalar_one_or_none()

        if enrollment_token is None:
            raise EnrollmentTokenError("Invalid or expired enrollment token")

        # Get workspace for org_id
        ws_result = await self.session.execute(
            select(Workspace).where(Workspace.id == enrollment_token.workspace_id)
        )
        workspace = ws_result.scalar_one_or_none()

        if workspace is None:
            raise EnrollmentTokenError("Workspace not found")

        # Create agent
        now = datetime.now(timezone.utc)
        agent = Agent(
            workspace_id=enrollment_token.workspace_id,
            name=request.agent_name,
            public_key=request.public_key,
            agent_version=request.agent_version,
            capabilities=request.capabilities or enrollment_token.capabilities,
            trust_state=TrustState.ENROLLED.value,
            trust_state_changed_at=now,
            last_seen_at=now,
            last_heartbeat_at=now,
            device_id=request.device_id,
            os_info=request.os_info,
        )

        if request.attestation:
            agent.attestation = request.attestation

        self.session.add(agent)
        await self.session.flush()

        # Mark token as used
        enrollment_token.is_used = True
        enrollment_token.used_at = now
        enrollment_token.used_by_agent_id = agent.id

        await self.session.flush()
        await self.session.refresh(agent)

        return AgentEnrollmentResult(
            agent_id=agent.id,
            workspace_id=agent.workspace_id,
            organization_id=workspace.organization_id,
            trust_state=TrustState(agent.trust_state),
            capabilities=agent.capabilities or [],
        )

    # ========================================================================
    # Heartbeat Processing
    # ========================================================================

    async def heartbeat(
        self,
        request: HeartbeatRequest,
    ) -> HeartbeatResult:
        """
        Process agent heartbeat.

        Updates last_seen, returns pending command count.

        Args:
            request: Heartbeat request

        Returns:
            Heartbeat result with server time and pending commands

        Raises:
            AgentNotFoundError: If agent not found
            AgentNotEnrolledError: If agent not enrolled
        """
        # Get agent
        result = await self.session.execute(
            select(Agent).where(
                Agent.id == request.agent_id,
                Agent.deleted_at.is_(None),
            )
        )
        agent = result.scalar_one_or_none()

        if agent is None:
            raise AgentNotFoundError(f"Agent {request.agent_id} not found")

        # Check trust state
        if agent.trust_state == TrustState.REVOKED.value:
            raise AgentNotEnrolledError(f"Agent {request.agent_id} has been revoked")

        if agent.trust_state == TrustState.SUSPENDED.value:
            raise AgentNotEnrolledError(f"Agent {request.agent_id} is suspended")

        # Update agent status
        now = datetime.now(timezone.utc)
        agent.last_seen_at = now
        agent.last_heartbeat_at = now
        agent.agent_version = request.agent_version
        agent.health_status = request.health_metrics

        await self.session.flush()

        # Get pending command count
        pending_count = await self._get_pending_command_count(
            agent.id,
            agent.workspace_id,
        )

        return HeartbeatResult(
            server_time=now,
            trust_state=TrustState(agent.trust_state),
            pending_commands=pending_count,
            next_heartbeat_sec=self.DEFAULT_HEARTBEAT_INTERVAL,
        )

    # ========================================================================
    # Trust State Management
    # ========================================================================

    async def revoke(
        self,
        agent_id: UUID,
        workspace_id: UUID,
        reason: str,
    ) -> Agent:
        """
        Revoke an agent's trust.

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID
            reason: Revocation reason

        Returns:
            Updated agent

        Raises:
            AgentNotFoundError: If agent not found
        """
        agent = await self._get_agent(agent_id, workspace_id)

        if agent.trust_state == TrustState.REVOKED.value:
            return agent  # Already revoked

        now = datetime.now(timezone.utc)
        agent.trust_state = TrustState.REVOKED.value
        agent.trust_state_changed_at = now
        agent.revocation_reason = reason

        await self.session.flush()
        await self.session.refresh(agent)

        return agent

    async def suspend(
        self,
        agent_id: UUID,
        workspace_id: UUID,
        reason: Optional[str] = None,
    ) -> Agent:
        """
        Suspend an agent.

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID
            reason: Suspension reason

        Returns:
            Updated agent
        """
        agent = await self._get_agent(agent_id, workspace_id)

        if agent.trust_state != TrustState.ENROLLED.value:
            raise TrustStateError(f"Cannot suspend agent in state {agent.trust_state}")

        now = datetime.now(timezone.utc)
        agent.trust_state = TrustState.SUSPENDED.value
        agent.trust_state_changed_at = now
        if reason:
            agent.revocation_reason = reason

        await self.session.flush()
        await self.session.refresh(agent)

        return agent

    async def reinstate(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> Agent:
        """
        Reinstate a suspended agent.

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID

        Returns:
            Updated agent
        """
        agent = await self._get_agent(agent_id, workspace_id)

        if agent.trust_state != TrustState.SUSPENDED.value:
            raise TrustStateError(f"Cannot reinstate agent in state {agent.trust_state}")

        now = datetime.now(timezone.utc)
        agent.trust_state = TrustState.ENROLLED.value
        agent.trust_state_changed_at = now
        agent.revocation_reason = None

        await self.session.flush()
        await self.session.refresh(agent)

        return agent

    # ========================================================================
    # Agent Queries
    # ========================================================================

    async def get_agent(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> Agent:
        """Get agent by ID."""
        return await self._get_agent(agent_id, workspace_id)

    async def get_stale_agents(
        self,
        workspace_id: UUID,
        threshold_minutes: int = STALE_THRESHOLD_MINUTES,
    ) -> List[Agent]:
        """
        Get agents that haven't sent heartbeat recently.

        Args:
            workspace_id: Workspace ID
            threshold_minutes: Minutes since last heartbeat

        Returns:
            List of stale agents
        """
        threshold = datetime.now(timezone.utc) - timedelta(minutes=threshold_minutes)

        result = await self.session.execute(
            select(Agent).where(
                Agent.workspace_id == workspace_id,
                Agent.trust_state == TrustState.ENROLLED.value,
                Agent.deleted_at.is_(None),
                or_(
                    Agent.last_heartbeat_at.is_(None),
                    Agent.last_heartbeat_at < threshold,
                ),
            )
        )

        return list(result.scalars().all())

    async def list_agents(
        self,
        workspace_id: UUID,
        trust_state: Optional[TrustState] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[List[Agent], int]:
        """
        List agents in workspace.

        Args:
            workspace_id: Workspace ID
            trust_state: Optional filter by trust state
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (agents, total_count)
        """
        base_query = select(Agent).where(
            Agent.workspace_id == workspace_id,
            Agent.deleted_at.is_(None),
        )

        if trust_state:
            base_query = base_query.where(Agent.trust_state == trust_state.value)

        # Get count
        count_query = select(func.count()).select_from(base_query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get agents
        query = base_query.order_by(Agent.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(query)
        agents = list(result.scalars().all())

        return agents, total

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _get_agent(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> Agent:
        """Get agent with workspace validation."""
        result = await self.session.execute(
            select(Agent).where(
                Agent.id == agent_id,
                Agent.workspace_id == workspace_id,
                Agent.deleted_at.is_(None),
            )
        )
        agent = result.scalar_one_or_none()

        if agent is None:
            raise AgentNotFoundError(f"Agent {agent_id} not found")

        return agent

    async def _get_pending_command_count(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> int:
        """Get count of pending commands for agent."""
        now = datetime.now(timezone.utc)

        result = await self.session.execute(
            select(func.count(Command.id)).where(
                Command.agent_id == agent_id,
                Command.workspace_id == workspace_id,
                Command.status.in_(
                    [
                        CommandStatus.PENDING.value,
                        CommandStatus.APPROVED.value,
                        CommandStatus.SENT.value,
                    ]
                ),
                or_(
                    Command.expires_at.is_(None),
                    Command.expires_at > now,
                ),
            )
        )

        return result.scalar() or 0
