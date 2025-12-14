# -*- coding: utf-8 -*-
"""
Database Configuration for Cloud Control Plane.

CLOUD ZONE ONLY.

Provides async SQLAlchemy engine and session management for PostgreSQL.
Implements connection pooling and tenant context for Row Level Security.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool


# Environment-based configuration
# WI-DEPS-01: For dev/test, default to SQLite to avoid requiring PostgreSQL.
# Production MUST set CCEA_DATABASE_URL to a real PostgreSQL connection.
#
# Valid URL formats:
#   - SQLite (dev/test): sqlite+aiosqlite:///./ccea_control_plane.db
#   - PostgreSQL (prod): postgresql+asyncpg://user:pass@host:5432/ccea_control_plane
#
# NOTE: asyncpg is required for PostgreSQL. Install with: pip install asyncpg
# NOTE: aiosqlite is required for SQLite async. Install with: pip install aiosqlite
_DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./ccea_control_plane.db"
_PROD_DATABASE_URL_EXAMPLE = "postgresql+asyncpg://postgres:postgres@localhost:5432/ccea_control_plane"

DATABASE_URL = os.getenv("CCEA_DATABASE_URL", _DEFAULT_DATABASE_URL)

# Validate database URL at import time
def _validate_database_url(url: str) -> None:
    """Validate database URL and check for required drivers."""
    if "asyncpg" in url:
        try:
            import asyncpg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgreSQL async connections. "
                "Install with: pip install asyncpg\n"
                f"Or use SQLite for dev/test: CCEA_DATABASE_URL={_DEFAULT_DATABASE_URL}"
            ) from e
    elif "aiosqlite" in url:
        try:
            import aiosqlite  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "aiosqlite is required for SQLite async connections. "
                "Install with: pip install aiosqlite"
            ) from e

# Validate on import (fail fast)
_validate_database_url(DATABASE_URL)

# Connection pool settings
POOL_SIZE = int(os.getenv("CCEA_DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("CCEA_DB_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("CCEA_DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("CCEA_DB_POOL_RECYCLE", "3600"))

# Test mode uses NullPool (no connection pooling)
TEST_MODE = os.getenv("CCEA_TEST_MODE", "false").lower() == "true"


def create_engine(
    url: Optional[str] = None,
    pool_size: int = POOL_SIZE,
    max_overflow: int = MAX_OVERFLOW,
    pool_timeout: int = POOL_TIMEOUT,
    pool_recycle: int = POOL_RECYCLE,
    test_mode: bool = TEST_MODE,
    echo: bool = False,
) -> AsyncEngine:
    """
    Create async SQLAlchemy engine.

    Args:
        url: Database URL (defaults to environment variable)
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        pool_timeout: Pool timeout in seconds
        pool_recycle: Connection recycle time in seconds
        test_mode: Use NullPool for testing
        echo: Enable SQL logging

    Returns:
        AsyncEngine instance

    Note:
        - For SQLite, connection pooling is disabled (StaticPool used instead)
        - For PostgreSQL, QueuePool is used in production, NullPool in test mode
    """
    db_url = url or DATABASE_URL

    # Validate the URL before creating engine
    _validate_database_url(db_url)

    # SQLite requires special handling - no connection pooling
    is_sqlite = db_url.startswith("sqlite")

    if is_sqlite:
        # SQLite: Use StaticPool for single-connection (thread-safe for async)
        from sqlalchemy.pool import StaticPool

        engine = create_async_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=echo,
            future=True,
        )
    else:
        # PostgreSQL: Use connection pooling
        pool_class = NullPool if test_mode else QueuePool

        engine = create_async_engine(
            db_url,
            poolclass=pool_class,
            pool_size=pool_size if not test_mode else None,
            max_overflow=max_overflow if not test_mode else None,
            pool_timeout=pool_timeout if not test_mode else None,
            pool_recycle=pool_recycle if not test_mode else None,
            echo=echo,
            future=True,
        )

    return engine


# Global engine instance (lazy initialization)
_engine: Optional[AsyncEngine] = None


def get_engine() -> AsyncEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def set_engine(engine: AsyncEngine) -> None:
    """Set the global engine instance (for testing)."""
    global _engine
    _engine = engine


async def close_engine() -> None:
    """Close the global engine instance."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


# Session factory
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


def set_session_factory(factory: async_sessionmaker[AsyncSession]) -> None:
    """Set the session factory (for testing)."""
    global _session_factory
    _session_factory = factory


class TenantContext:
    """
    Tenant context for Row Level Security.

    Sets the current workspace_id as a session variable for RLS policies.
    """

    def __init__(self, workspace_id: Optional[UUID] = None):
        """
        Initialize tenant context.

        Args:
            workspace_id: Current workspace ID for RLS
        """
        self.workspace_id = workspace_id

    async def set_tenant(self, session: AsyncSession) -> None:
        """
        Set tenant context on the database session.

        This sets a session variable that RLS policies use to filter rows.

        Args:
            session: Database session
        """
        if self.workspace_id is not None:
            await session.execute(
                text(f"SET app.current_workspace_id = '{self.workspace_id}'")
            )
        else:
            # Reset to empty string for superuser/admin operations
            await session.execute(text("SET app.current_workspace_id = ''"))


@asynccontextmanager
async def get_session(
    tenant_context: Optional[TenantContext] = None,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with optional tenant context.

    Usage:
        async with get_session(TenantContext(workspace_id)) as session:
            result = await session.execute(query)

    Args:
        tenant_context: Optional tenant context for RLS

    Yields:
        Database session
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            if tenant_context is not None:
                await tenant_context.set_tenant(session)
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session_dependency(
    workspace_id: Optional[UUID] = None,
) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Usage:
        @app.get("/items")
        async def list_items(session: AsyncSession = Depends(get_session_dependency)):
            ...

    Args:
        workspace_id: Optional workspace ID for tenant context

    Yields:
        Database session
    """
    tenant_context = TenantContext(workspace_id) if workspace_id else None
    async with get_session(tenant_context) as session:
        yield session


async def init_db() -> None:
    """
    Initialize database schema.

    Creates all tables defined in models.
    For production, use Alembic migrations instead.
    """
    from .models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This is destructive and should only be used in testing.
    """
    from .models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def setup_rls() -> None:
    """
    Set up Row Level Security policies.

    Executes the RLS setup SQL from models.
    Should be run once during database initialization.
    """
    from .models import POSTGRES_RLS_SETUP_SQL

    engine = get_engine()
    async with engine.begin() as conn:
        # Split into individual statements and execute
        statements = [
            s.strip()
            for s in POSTGRES_RLS_SETUP_SQL.split(";")
            if s.strip() and not s.strip().startswith("--")
        ]
        for stmt in statements:
            if stmt:
                await conn.execute(text(stmt))


# Health check
async def check_db_health() -> bool:
    """
    Check database connectivity.

    Returns:
        True if database is healthy, False otherwise
    """
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
