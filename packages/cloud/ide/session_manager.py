# -*- coding: utf-8 -*-
"""
IDE Session Manager - CLOUD ZONE ONLY.

Manages user sessions for the Strategy Workspace IDE.

Design Doc Reference:
    - Section 4.1: "Strategy Workspace"
    - Multi-tenant isolation per workspace
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SESSION_TIMEOUT_MINUTES: Final[int] = 120
MAX_SESSIONS_PER_USER: Final[int] = 5
MAX_SESSIONS_PER_WORKSPACE: Final[int] = 50


class SessionState(str, Enum):
    """Session state."""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    """IDE session configuration."""
    timeout_minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES
    max_notebooks: int = 10
    max_editor_sessions: int = 20
    max_file_size_mb: int = 50
    enable_collaboration: bool = True
    enable_terminal: bool = False  # Disabled in Cloud Zone
    theme: str = "dark"
    font_size: int = 14
    tab_size: int = 4
    auto_save_interval_seconds: int = 30


@dataclass
class IDESession:
    """
    IDE session for a user in a workspace.

    Tracks all active notebooks, editors, and resources.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: UUID = field(default_factory=uuid.uuid4)
    user_id: UUID = field(default_factory=uuid.uuid4)
    state: SessionState = SessionState.ACTIVE
    config: SessionConfig = field(default_factory=SessionConfig)

    # Active resources
    notebook_sessions: Set[str] = field(default_factory=set)
    editor_sessions: Set[str] = field(default_factory=set)
    open_files: Set[str] = field(default_factory=set)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metrics
    total_executions: int = 0
    total_saves: int = 0

    def is_expired(self) -> bool:
        """Check if session expired."""
        timeout = timedelta(minutes=self.config.timeout_minutes)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def touch(self) -> None:
        """Update activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        if self.state == SessionState.IDLE:
            self.state = SessionState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "workspace_id": str(self.workspace_id),
            "user_id": str(self.user_id),
            "state": self.state.value,
            "notebook_count": len(self.notebook_sessions),
            "editor_count": len(self.editor_sessions),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


class IDESessionManager:
    """
    Manages IDE sessions with tenant isolation.

    SECURITY: Enforces workspace boundaries and resource limits.
    """

    def __init__(self, default_config: Optional[SessionConfig] = None):
        self._sessions: Dict[str, IDESession] = {}
        self._user_sessions: Dict[UUID, Set[str]] = {}
        self._workspace_sessions: Dict[UUID, Set[str]] = {}
        self._default_config = default_config or SessionConfig()
        self._lock = asyncio.Lock()
        logger.info("IDESessionManager initialized")

    async def create_session(
        self,
        workspace_id: UUID,
        user_id: UUID,
        config: Optional[SessionConfig] = None,
    ) -> IDESession:
        """Create new IDE session."""
        async with self._lock:
            # Check limits
            user_sessions = self._user_sessions.get(user_id, set())
            if len(user_sessions) >= MAX_SESSIONS_PER_USER:
                raise ValueError(f"Maximum sessions per user ({MAX_SESSIONS_PER_USER}) exceeded")

            workspace_sessions = self._workspace_sessions.get(workspace_id, set())
            if len(workspace_sessions) >= MAX_SESSIONS_PER_WORKSPACE:
                raise ValueError(f"Maximum sessions per workspace ({MAX_SESSIONS_PER_WORKSPACE}) exceeded")

            session = IDESession(
                workspace_id=workspace_id,
                user_id=user_id,
                config=config or SessionConfig(**vars(self._default_config)),
            )

            self._sessions[session.session_id] = session

            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session.session_id)

            if workspace_id not in self._workspace_sessions:
                self._workspace_sessions[workspace_id] = set()
            self._workspace_sessions[workspace_id].add(session.session_id)

        logger.info(f"Created IDE session {session.session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[IDESession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            await self.close_session(session_id)
            return None
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close session and cleanup resources."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            session.state = SessionState.CLOSED

            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(session_id)
            if session.workspace_id in self._workspace_sessions:
                self._workspace_sessions[session.workspace_id].discard(session_id)

        logger.info(f"Closed IDE session {session_id}")
        return True

    async def list_sessions(
        self,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> List[IDESession]:
        """List active sessions."""
        sessions = []
        for session in list(self._sessions.values()):
            if session.is_expired():
                await self.close_session(session.session_id)
                continue
            if workspace_id and session.workspace_id != workspace_id:
                continue
            if user_id and session.user_id != user_id:
                continue
            sessions.append(session)
        return sessions

    async def cleanup_expired(self) -> int:
        """Cleanup expired sessions."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            await self.close_session(sid)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired IDE sessions")
        return len(expired)
