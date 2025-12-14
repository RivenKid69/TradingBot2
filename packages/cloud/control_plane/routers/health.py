# -*- coding: utf-8 -*-
"""
Health Check Router.

CLOUD ZONE ONLY.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, status
from pydantic import BaseModel

from ..database import check_db_health

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    database: str
    details: Dict[str, Any] = {}


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health of the Cloud Control Plane service.",
)
async def health_check() -> HealthResponse:
    """
    Perform health check.

    Returns:
        Health status with component details
    """
    db_healthy = await check_db_health()

    overall_status = "healthy" if db_healthy else "degraded"
    db_status = "connected" if db_healthy else "disconnected"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        database=db_status,
        details={
            "components": {
                "database": {"status": db_status, "type": "postgresql"},
            }
        },
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the service is ready to accept requests.",
)
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe for Kubernetes.

    Returns:
        Ready status
    """
    db_healthy = await check_db_health()

    if not db_healthy:
        return {"status": "not_ready", "reason": "database_unavailable"}

    return {"status": "ready"}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the service is alive.",
)
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.

    Returns:
        Alive status
    """
    return {"status": "alive"}
