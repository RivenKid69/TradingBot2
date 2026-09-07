# -*- coding: utf-8 -*-
"""
Alembic Migration Environment.

Phase 7 (WI-CLOUD-04): Async SQLAlchemy migrations for CCEA Cloud Control Plane.

This environment supports:
- Async migrations with asyncpg/aiosqlite
- Multi-tenant RLS policy setup
- Automatic model detection from models.py

Usage:
    # Generate migration
    cd packages/cloud/control_plane
    alembic revision --autogenerate -m "description"

    # Apply migrations
    alembic upgrade head

    # Rollback
    alembic downgrade -1
"""

import asyncio
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models for autogenerate support
from models import Base

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata


# Get database URL from environment or config
def get_database_url() -> str:
    """Get database URL from environment or config."""
    # Priority: environment variable > alembic.ini
    url = os.getenv("CCEA_DATABASE_URL")
    if url:
        return url
    return config.get_main_option("sqlalchemy.url", "sqlite+aiosqlite:///./ccea_control_plane.db")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to context.execute() will emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Include schema changes
        include_schemas=True,
        # Render column types properly
        render_as_batch=True,  # For SQLite support
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.

    Creates an async Engine and associates a connection with the context.
    """
    # Create configuration dict
    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}

    # Override URL from environment
    configuration["sqlalchemy.url"] = get_database_url()

    # Determine pool class based on database type
    url = configuration["sqlalchemy.url"]
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        poolclass = StaticPool
        connect_args = {"check_same_thread": False}
    else:
        from sqlalchemy.pool import NullPool

        poolclass = NullPool
        connect_args = {}

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=poolclass,
        connect_args=connect_args,
        future=True,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
