# -*- coding: utf-8 -*-
"""
CCEA Enrollment Service.

Handles agent enrollment process:
1. Create enrollment token with TTL
2. Agent sends enrollment request with public key
3. Cloud validates token and registers agent
4. Agent receives agent_id and endpoints

Security:
- Tokens have TTL and max uses
- Agent generates keypair locally
- Cloud only stores public key
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ccea.models.enrollment import (
    EnrollmentToken,
    EnrollmentRequest,
    EnrollmentResponse,
    AgentInfo,
    TrustState,
    AgentCapability,
)
from ccea.crypto.tokens import (
    generate_enrollment_token,
    generate_agent_id,
    is_valid_enrollment_token,
)
from ccea.crypto.keys import load_public_key


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class EnrollmentError(Exception):
    """Base enrollment error."""
    pass


class TokenExpiredError(EnrollmentError):
    """Token has expired."""
    pass


class TokenInvalidError(EnrollmentError):
    """Token is invalid."""
    pass


class TokenRevokedError(EnrollmentError):
    """Token has been revoked."""
    pass


class TokenExhaustedError(EnrollmentError):
    """Token max uses reached."""
    pass


class AgentAlreadyEnrolledError(EnrollmentError):
    """Agent is already enrolled."""
    pass


# ============================================================================
# Token Store Interface
# ============================================================================

class TokenStore(ABC):
    """Abstract token store interface."""

    @abstractmethod
    def save(self, token: EnrollmentToken) -> None:
        """Save a token."""
        pass

    @abstractmethod
    def get(self, token_id: str) -> Optional[EnrollmentToken]:
        """Get a token by ID."""
        pass

    @abstractmethod
    def delete(self, token_id: str) -> None:
        """Delete a token."""
        pass

    @abstractmethod
    def increment_use(self, token_id: str) -> None:
        """Increment token use count."""
        pass


class InMemoryTokenStore(TokenStore):
    """In-memory token store for development/testing."""

    def __init__(self):
        self._tokens: Dict[str, EnrollmentToken] = {}

    def save(self, token: EnrollmentToken) -> None:
        self._tokens[token.token_id] = token

    def get(self, token_id: str) -> Optional[EnrollmentToken]:
        return self._tokens.get(token_id)

    def delete(self, token_id: str) -> None:
        self._tokens.pop(token_id, None)

    def increment_use(self, token_id: str) -> None:
        token = self._tokens.get(token_id)
        if token:
            token.uses_count += 1


# ============================================================================
# Agent Store Interface
# ============================================================================

class AgentStore(ABC):
    """Abstract agent store interface."""

    @abstractmethod
    def save(self, agent: AgentInfo) -> None:
        """Save an agent."""
        pass

    @abstractmethod
    def get(self, agent_id: str) -> Optional[AgentInfo]:
        """Get an agent by ID."""
        pass

    @abstractmethod
    def get_by_public_key(self, public_key: str) -> Optional[AgentInfo]:
        """Get an agent by public key."""
        pass

    @abstractmethod
    def update(self, agent: AgentInfo) -> None:
        """Update an agent."""
        pass

    @abstractmethod
    def list_by_workspace(self, workspace_id: str) -> List[AgentInfo]:
        """List agents in a workspace."""
        pass


class InMemoryAgentStore(AgentStore):
    """In-memory agent store for development/testing."""

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._by_public_key: Dict[str, str] = {}

    def save(self, agent: AgentInfo) -> None:
        self._agents[agent.agent_id] = agent
        self._by_public_key[agent.public_key] = agent.agent_id

    def get(self, agent_id: str) -> Optional[AgentInfo]:
        return self._agents.get(agent_id)

    def get_by_public_key(self, public_key: str) -> Optional[AgentInfo]:
        agent_id = self._by_public_key.get(public_key)
        if agent_id:
            return self._agents.get(agent_id)
        return None

    def update(self, agent: AgentInfo) -> None:
        if agent.agent_id in self._agents:
            self._agents[agent.agent_id] = agent

    def list_by_workspace(self, workspace_id: str) -> List[AgentInfo]:
        return [
            a for a in self._agents.values()
            if a.workspace_id == workspace_id
        ]


# ============================================================================
# Enrollment Service
# ============================================================================

class EnrollmentService:
    """
    Agent enrollment service.

    Handles the full enrollment lifecycle:
    - Token creation with TTL
    - Token validation
    - Agent registration
    - Agent lookup/management
    """

    def __init__(
        self,
        token_store: Optional[TokenStore] = None,
        agent_store: Optional[AgentStore] = None,
        base_url: str = "http://localhost:8000",
    ):
        """
        Initialize enrollment service.

        Args:
            token_store: Token storage backend
            agent_store: Agent storage backend
            base_url: Base URL for endpoints
        """
        self.token_store = token_store or InMemoryTokenStore()
        self.agent_store = agent_store or InMemoryAgentStore()
        self.base_url = base_url.rstrip("/")

    def create_token(
        self,
        workspace_id: str,
        created_by: str,
        ttl_hours: int = 24,
        max_uses: int = 1,
        capabilities: Optional[List[AgentCapability]] = None,
    ) -> EnrollmentToken:
        """
        Create a new enrollment token.

        Args:
            workspace_id: Target workspace
            created_by: User creating the token
            ttl_hours: Token TTL in hours (default: 24)
            max_uses: Maximum uses (default: 1)
            capabilities: Agent capabilities to grant

        Returns:
            New EnrollmentToken
        """
        token_id = generate_enrollment_token()
        token = EnrollmentToken.create(
            token_id=token_id,
            workspace_id=workspace_id,
            created_by=created_by,
            ttl_hours=ttl_hours,
            max_uses=max_uses,
            capabilities=capabilities,
        )

        self.token_store.save(token)

        logger.info(
            "Created enrollment token",
            extra={
                "token_id": token_id[:20] + "...",
                "workspace_id": workspace_id,
                "ttl_hours": ttl_hours,
                "max_uses": max_uses,
            }
        )

        return token

    def validate_token(self, token_id: str) -> EnrollmentToken:
        """
        Validate an enrollment token.

        Args:
            token_id: Token to validate

        Returns:
            Valid EnrollmentToken

        Raises:
            TokenInvalidError: If token doesn't exist
            TokenExpiredError: If token has expired
            TokenRevokedError: If token is revoked
            TokenExhaustedError: If max uses reached
        """
        if not is_valid_enrollment_token(token_id):
            raise TokenInvalidError("Invalid token format")

        token = self.token_store.get(token_id)
        if token is None:
            raise TokenInvalidError("Token not found")

        if token.revoked:
            raise TokenRevokedError("Token has been revoked")

        if datetime.utcnow() > token.expires_at:
            raise TokenExpiredError("Token has expired")

        if token.uses_count >= token.max_uses:
            raise TokenExhaustedError("Token max uses reached")

        return token

    def enroll_agent(self, request: EnrollmentRequest) -> EnrollmentResponse:
        """
        Enroll a new agent.

        Args:
            request: Enrollment request from agent

        Returns:
            EnrollmentResponse with agent_id and endpoints
        """
        try:
            # Validate token
            token = self.validate_token(request.token)

            # Validate public key format
            try:
                load_public_key(request.public_key)
            except ValueError as e:
                return EnrollmentResponse.error_response(
                    "INVALID_PUBLIC_KEY",
                    str(e),
                )

            # Check if agent already enrolled with this key
            existing = self.agent_store.get_by_public_key(request.public_key)
            if existing and existing.trust_state == TrustState.ENROLLED:
                return EnrollmentResponse.error_response(
                    "ALREADY_ENROLLED",
                    f"Agent already enrolled: {existing.agent_id}",
                )

            # Generate agent ID
            agent_id = generate_agent_id()

            # Create agent info
            agent = AgentInfo(
                agent_id=agent_id,
                workspace_id=token.workspace_id,
                public_key=request.public_key,
                public_key_algorithm=request.public_key_algorithm,
                agent_version=request.agent_version,
                hostname=request.hostname,
                capabilities=list(set(request.capabilities) & set(token.capabilities)),
                trust_state=TrustState.ENROLLED,
                enrolled_at=datetime.utcnow(),
                enrolled_by_token=token.token_id,
            )

            # Save agent
            self.agent_store.save(agent)

            # Increment token use
            self.token_store.increment_use(token.token_id)

            logger.info(
                "Agent enrolled successfully",
                extra={
                    "agent_id": agent_id,
                    "workspace_id": token.workspace_id,
                    "capabilities": [c.value for c in agent.capabilities],
                }
            )

            return EnrollmentResponse.success_response(
                agent_id=agent_id,
                workspace_id=token.workspace_id,
                heartbeat_endpoint=f"{self.base_url}/api/v1/agents/{agent_id}/heartbeat",
                commands_endpoint=f"{self.base_url}/api/v1/agents/{agent_id}/commands",
                telemetry_endpoint=f"{self.base_url}/api/v1/agents/{agent_id}/telemetry",
            )

        except TokenInvalidError as e:
            return EnrollmentResponse.error_response("TOKEN_INVALID", str(e))
        except TokenExpiredError as e:
            return EnrollmentResponse.error_response("TOKEN_EXPIRED", str(e))
        except TokenRevokedError as e:
            return EnrollmentResponse.error_response("TOKEN_REVOKED", str(e))
        except TokenExhaustedError as e:
            return EnrollmentResponse.error_response("TOKEN_EXHAUSTED", str(e))
        except Exception as e:
            logger.exception("Enrollment failed")
            return EnrollmentResponse.error_response("INTERNAL_ERROR", str(e))

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID."""
        return self.agent_store.get(agent_id)

    def revoke_agent(
        self,
        agent_id: str,
        revoked_by: str,
        reason: str,
    ) -> bool:
        """
        Revoke an agent's trust.

        Args:
            agent_id: Agent to revoke
            revoked_by: User performing revocation
            reason: Reason for revocation

        Returns:
            True if revoked, False if agent not found
        """
        agent = self.agent_store.get(agent_id)
        if agent is None:
            return False

        agent.revoke(revoked_by, reason)
        self.agent_store.update(agent)

        logger.warning(
            "Agent revoked",
            extra={
                "agent_id": agent_id,
                "revoked_by": revoked_by,
                "reason": reason,
            }
        )

        return True

    def revoke_token(
        self,
        token_id: str,
        revoked_by: str,
    ) -> bool:
        """
        Revoke an enrollment token.

        Args:
            token_id: Token to revoke
            revoked_by: User performing revocation

        Returns:
            True if revoked, False if token not found
        """
        token = self.token_store.get(token_id)
        if token is None:
            return False

        token.revoked = True
        token.revoked_at = datetime.utcnow()
        token.revoked_by = revoked_by
        self.token_store.save(token)

        logger.info(
            "Enrollment token revoked",
            extra={
                "token_id": token_id[:20] + "...",
                "revoked_by": revoked_by,
            }
        )

        return True
