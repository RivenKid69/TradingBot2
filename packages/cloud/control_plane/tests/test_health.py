# -*- coding: utf-8 -*-
"""Tests for Health Router."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    async def test_health_check_returns_200(self, client: AsyncClient) -> None:
        """Health check should return 200 OK."""
        response = await client.get("/api/v1/health/health")
        assert response.status_code == 200

    async def test_health_check_response_structure(self, client: AsyncClient) -> None:
        """Health check response should have correct structure."""
        response = await client.get("/api/v1/health/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "database" in data
        assert "details" in data

    async def test_health_check_status_healthy(self, client: AsyncClient) -> None:
        """Health check status should be healthy when DB is connected."""
        response = await client.get("/api/v1/health/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["database"] == "connected"

    async def test_health_check_version(self, client: AsyncClient) -> None:
        """Health check should return version."""
        response = await client.get("/api/v1/health/health")
        data = response.json()

        assert data["version"] == "1.0.0"

    async def test_health_check_details_components(self, client: AsyncClient) -> None:
        """Health check details should include database component."""
        response = await client.get("/api/v1/health/health")
        data = response.json()

        assert "components" in data["details"]
        assert "database" in data["details"]["components"]
        assert data["details"]["components"]["database"]["status"] == "connected"


class TestReadinessEndpoint:
    """Tests for /ready endpoint."""

    async def test_readiness_check_returns_200(self, client: AsyncClient) -> None:
        """Readiness check should return 200 OK."""
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200

    async def test_readiness_check_ready_status(self, client: AsyncClient) -> None:
        """Readiness check should return ready status."""
        response = await client.get("/api/v1/health/ready")
        data = response.json()

        assert data["status"] == "ready"


class TestLivenessEndpoint:
    """Tests for /live endpoint."""

    async def test_liveness_check_returns_200(self, client: AsyncClient) -> None:
        """Liveness check should return 200 OK."""
        response = await client.get("/api/v1/health/live")
        assert response.status_code == 200

    async def test_liveness_check_alive_status(self, client: AsyncClient) -> None:
        """Liveness check should return alive status."""
        response = await client.get("/api/v1/health/live")
        data = response.json()

        assert data["status"] == "alive"

    async def test_liveness_check_no_db_dependency(self, client: AsyncClient) -> None:
        """Liveness check should not depend on database."""
        # This test verifies that /live endpoint doesn't require DB
        response = await client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
