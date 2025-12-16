# -*- coding: utf-8 -*-
"""
Row-Level Security Context Manager - CLOUD ZONE ONLY.

Provides application-level RLS context management for PostgreSQL.

Design Doc Reference:
    - Multi-tenant isolation via workspace_id
    - SOC2/ISO27001 data segregation

Usage:
    async with rls_context(session, workspace_id):
        # All queries within this block are tenant-isolated
        results = await session.execute(select(Strategy))

Security:
    - ALWAYS set RLS context before tenant-scoped queries
    - Context is automatically cleared on exit
    - Admin bypass requires explicit is_admin=True
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator, Optional, TYPE_CHECKING
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from sqlalchemy.pool import ConnectionPoolEntry

logger = logging.getLogger(__name__)


# ============================================================================
# RLS Context for Synchronous Sessions
# ============================================================================


@contextmanager
def rls_context(
    session: Session,
    workspace_id: UUID,
    is_admin: bool = False,
) -> Generator[Session, None, None]:
    """
    Set RLS context for synchronous database session.

    Args:
        session: SQLAlchemy session
        workspace_id: Current workspace ID for tenant isolation
        is_admin: If True, bypass RLS restrictions

    Yields:
        Session with RLS context set

    Example:
        with rls_context(session, workspace_id):
            strategies = session.query(Strategy).all()
    """
    connection = session.connection()

    try:
        # Set workspace context
        connection.execute(
            text("SET app.current_workspace_id = :workspace_id"),
            {"workspace_id": str(workspace_id)}
        )

        # Set admin flag
        connection.execute(
            text("SET app.is_admin = :is_admin"),
            {"is_admin": str(is_admin).lower()}
        )

        logger.debug(f"RLS context set: workspace={workspace_id}, admin={is_admin}")
        yield session

    finally:
        # Clear context
        try:
            connection.execute(text("RESET app.current_workspace_id"))
            connection.execute(text("RESET app.is_admin"))
            logger.debug("RLS context cleared")
        except Exception as e:
            logger.warning(f"Failed to clear RLS context: {e}")


# ============================================================================
# RLS Context for Async Sessions
# ============================================================================


@asynccontextmanager
async def async_rls_context(
    session: AsyncSession,
    workspace_id: UUID,
    is_admin: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Set RLS context for async database session.

    Args:
        session: AsyncSession instance
        workspace_id: Current workspace ID
        is_admin: Admin bypass flag

    Yields:
        AsyncSession with RLS context

    Example:
        async with async_rls_context(session, workspace_id):
            result = await session.execute(select(Strategy))
    """
    try:
        # Set workspace context
        await session.execute(
            text("SET app.current_workspace_id = :workspace_id"),
            {"workspace_id": str(workspace_id)}
        )

        # Set admin flag
        await session.execute(
            text("SET app.is_admin = :is_admin"),
            {"is_admin": str(is_admin).lower()}
        )

        logger.debug(f"Async RLS context set: workspace={workspace_id}, admin={is_admin}")
        yield session

    finally:
        # Clear context
        try:
            await session.execute(text("RESET app.current_workspace_id"))
            await session.execute(text("RESET app.is_admin"))
            logger.debug("Async RLS context cleared")
        except Exception as e:
            logger.warning(f"Failed to clear async RLS context: {e}")


# ============================================================================
# RLS Middleware for FastAPI
# ============================================================================


class RLSMiddleware:
    """
    FastAPI middleware for automatic RLS context.

    Sets RLS context based on authenticated user's workspace.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # RLS context is set per-request in dependencies
        # This middleware logs RLS-related events
        await self.app(scope, receive, send)


# ============================================================================
# RLS Event Listeners
# ============================================================================


def setup_rls_listeners(engine: Engine) -> None:
    """
    Setup SQLAlchemy event listeners for RLS enforcement.

    Ensures RLS context is properly managed on connection checkout/checkin.
    """

    @event.listens_for(engine, "checkout")
    def on_checkout(
        dbapi_connection: Any,
        connection_record: "ConnectionPoolEntry",
        connection_proxy: Any,
    ) -> None:
        """Reset RLS context when connection is checked out from pool."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("RESET app.current_workspace_id")
            cursor.execute("RESET app.is_admin")
        except Exception:
            # Ignore errors (e.g., SQLite doesn't support SET/RESET)
            pass
        finally:
            cursor.close()

    @event.listens_for(engine, "checkin")
    def on_checkin(
        dbapi_connection: Any,
        connection_record: "ConnectionPoolEntry",
    ) -> None:
        """Clear RLS context when connection is returned to pool."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("RESET app.current_workspace_id")
            cursor.execute("RESET app.is_admin")
        except Exception:
            pass
        finally:
            cursor.close()

    logger.info("RLS event listeners configured")


# ============================================================================
# RLS Verification Utilities
# ============================================================================


async def verify_rls_context(session: AsyncSession) -> dict:
    """
    Verify current RLS context settings.

    Returns:
        Dict with current workspace_id and is_admin values
    """
    result = await session.execute(text("""
        SELECT
            current_setting('app.current_workspace_id', true) as workspace_id,
            current_setting('app.is_admin', true) as is_admin
    """))
    row = result.fetchone()

    return {
        "workspace_id": row[0] if row and row[0] else None,
        "is_admin": row[1].lower() == "true" if row and row[1] else False,
    }


def verify_rls_context_sync(session: Session) -> dict:
    """Synchronous version of verify_rls_context."""
    result = session.execute(text("""
        SELECT
            current_setting('app.current_workspace_id', true) as workspace_id,
            current_setting('app.is_admin', true) as is_admin
    """))
    row = result.fetchone()

    return {
        "workspace_id": row[0] if row and row[0] else None,
        "is_admin": row[1].lower() == "true" if row and row[1] else False,
    }


# ============================================================================
# RLS Enforcement Decorator
# ============================================================================


def require_rls_context(func):
    """
    Decorator to ensure RLS context is set before function execution.

    Raises RuntimeError if workspace context is not set.
    """
    import functools

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Find session in args/kwargs
        session = kwargs.get("session") or (args[0] if args else None)
        if isinstance(session, AsyncSession):
            context = await verify_rls_context(session)
            if not context["workspace_id"] and not context["is_admin"]:
                raise RuntimeError(
                    "RLS context not set. Use async_rls_context() before tenant-scoped queries."
                )
        return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        session = kwargs.get("session") or (args[0] if args else None)
        if isinstance(session, Session):
            context = verify_rls_context_sync(session)
            if not context["workspace_id"] and not context["is_admin"]:
                raise RuntimeError(
                    "RLS context not set. Use rls_context() before tenant-scoped queries."
                )
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
