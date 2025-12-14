# -*- coding: utf-8 -*-
"""
FastAPI Cloud Control Plane Application.

CLOUD ZONE ONLY.

Main application entry point for the CCEA Cloud Control Plane.
Provides REST API for:
- Organization and Workspace management
- User and RBAC management
- Agent enrollment and trust management
- Deployment and Run management
- Command dispatch and approval
- Telemetry ingestion
- Config blob storage
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import check_db_health, close_engine, init_db
from .routers import (
    agent_lifecycle,
    agents,
    auth,
    commands,
    config_blobs,
    deployments,
    health,
    organizations,
    strategies,
    telemetry,
    users,
    workspaces,
)

logger = logging.getLogger(__name__)


# Application metadata
APP_TITLE = "CCEA Cloud Control Plane"
APP_DESCRIPTION = """
Cloud Control Plane for the Cloud-Centric Execution Architecture (CCEA).

## Overview

The Cloud Control Plane manages:
- **Organizations & Workspaces**: Multi-tenant isolation
- **Users & RBAC**: Role-based access control
- **Agents**: Enrollment, trust management, heartbeat
- **Strategies & Artifacts**: Build registry and versioning
- **Deployments & Runs**: Lifecycle management
- **Commands**: Dispatch with approval workflow
- **Telemetry**: Metrics and alerts ingestion
- **Config Blobs**: Immutable configuration storage

## Security

- All endpoints require authentication (JWT)
- Tenant isolation via workspace context
- Row Level Security (RLS) on database
- Approval workflow for trading-impacting changes
- Audit logging for all sensitive operations

## Design Principles

Cloud CANNOT send order-like payloads. All trading decisions
are made locally by the Agent. Cloud only sends lifecycle commands.
"""
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes database on startup and closes connections on shutdown.
    """
    logger.info("Starting Cloud Control Plane...")

    # Initialize database (in production, use migrations)
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Cloud Control Plane...")
    await close_engine()
    logger.info("Database connections closed")


def create_app(
    debug: bool = False,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    Create FastAPI application instance.

    Args:
        debug: Enable debug mode
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "type": "internal_error",
            },
        )

    # Include routers
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(
        organizations.router, prefix="/api/v1/organizations", tags=["Organizations"]
    )
    app.include_router(
        workspaces.router, prefix="/api/v1/workspaces", tags=["Workspaces"]
    )
    app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(
        strategies.router, prefix="/api/v1/strategies", tags=["Strategies"]
    )
    app.include_router(
        deployments.router, prefix="/api/v1/deployments", tags=["Deployments"]
    )
    app.include_router(commands.router, prefix="/api/v1/commands", tags=["Commands"])
    app.include_router(
        config_blobs.router, prefix="/api/v1/config-blobs", tags=["Config Blobs"]
    )
    app.include_router(
        telemetry.router, prefix="/api/v1/telemetry", tags=["Telemetry"]
    )
    # Phase 7 (WI-CLOUD-02): Agent lifecycle endpoints with AgentDep auth
    app.include_router(
        agent_lifecycle.router,
        prefix="/api/v1/agent",
        tags=["Agent Lifecycle"],
    )

    return app


# Default application instance
app = create_app()


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health/health",
    }
