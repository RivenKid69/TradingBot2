# -*- coding: utf-8 -*-
"""Tests for Authentication Router."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    AgentEnrollmentToken,
    Organization,
    Permission,
    Role,
    TrustState,
    User,
    Workspace,
    compute_token_hash,
    generate_enrollment_token,
)

pytestmark = pytest.mark.asyncio


def hash_password(password: str) -> str:
    """Hash password with SHA256 (mirrors auth.py)."""
    return hashlib.sha256(password.encode()).hexdigest()


class TestLoginEndpoint:
    """Tests for POST /auth/login endpoint."""

    async def test_login_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Successful login should return JWT token."""
        # Create organization
        org = Organization(name="login-test-org", display_name="Login Test Org")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        # Create user with known password
        password = "testpassword123"
        user = User(
            email="login@example.com",
            password_hash=hash_password(password),
            display_name="Login User",
            organization_id=org.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "login@example.com", "password": password},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["email"] == "login@example.com"
        assert data["user_id"] == str(user.id)

    async def test_login_invalid_email(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Login with non-existent email should return 401."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "anypassword"},
        )

        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    async def test_login_invalid_password(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Login with wrong password should return 401."""
        # Create organization
        org = Organization(name="password-test-org", display_name="Password Test Org")
        db_session.add(org)
        await db_session.commit()

        # Create user
        user = User(
            email="wrongpwd@example.com",
            password_hash=hash_password("correctpassword"),
            organization_id=org.id,
        )
        db_session.add(user)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "wrongpwd@example.com", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    async def test_login_inactive_user(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Login as inactive user should return 401."""
        # Create organization
        org = Organization(name="inactive-test-org", display_name="Inactive Test Org")
        db_session.add(org)
        await db_session.commit()

        # Create inactive user
        user = User(
            email="inactive@example.com",
            password_hash=hash_password("password123"),
            organization_id=org.id,
            is_active=False,
        )
        db_session.add(user)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "inactive@example.com", "password": "password123"},
        )

        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    async def test_login_invalid_email_format(
        self,
        client: AsyncClient,
    ) -> None:
        """Login with invalid email format should return 422."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "notanemail", "password": "password123"},
        )

        assert response.status_code == 422

    async def test_login_missing_password(
        self,
        client: AsyncClient,
    ) -> None:
        """Login without password should return 422."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com"},
        )

        assert response.status_code == 422

    async def test_login_updates_last_login(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ) -> None:
        """Successful login should update last_login timestamp."""
        # Create organization
        org = Organization(name="lastlogin-test-org", display_name="Last Login Test")
        db_session.add(org)
        await db_session.commit()

        # Create user with no last_login
        password = "testpassword123"
        user = User(
            email="lastlogin@example.com",
            password_hash=hash_password(password),
            organization_id=org.id,
            last_login=None,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        assert user.last_login is None

        before_login = datetime.now(timezone.utc)
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "lastlogin@example.com", "password": password},
        )

        assert response.status_code == 200

        # Refresh user to see updated last_login
        await db_session.refresh(user)
        assert user.last_login is not None
        # SQLite stores timezone-naive datetimes, so compare naive datetimes
        before_login_naive = before_login.replace(tzinfo=None)
        assert user.last_login >= before_login_naive


class TestLogoutEndpoint:
    """Tests for POST /auth/logout endpoint."""

    async def test_logout_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """Logout should return 204 No Content."""
        response = await client.post(
            "/api/v1/auth/logout",
            headers=auth_headers,
        )

        assert response.status_code == 204

    async def test_logout_requires_authentication(
        self,
        client: AsyncClient,
    ) -> None:
        """Logout without authentication should return 401."""
        response = await client.post("/api/v1/auth/logout")

        assert response.status_code == 401


class TestGetMeEndpoint:
    """Tests for GET /auth/me endpoint."""

    async def test_get_me_returns_user_info(
        self,
        client: AsyncClient,
        auth_headers: dict,
        user_id,
        workspace_id,
        org_id,
    ) -> None:
        """Get me should return current user information."""
        response = await client.get(
            "/api/v1/auth/me",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(user_id)
        assert data["email"] == "testuser@example.com"
        assert data["workspace_id"] == str(workspace_id)
        assert data["org_id"] == str(org_id)
        assert "permissions" in data
        assert data["is_superuser"] is False

    async def test_get_me_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Superuser should see is_superuser=True."""
        response = await client.get(
            "/api/v1/auth/me",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_superuser"] is True

    async def test_get_me_requires_authentication(
        self,
        client: AsyncClient,
    ) -> None:
        """Get me without authentication should return 401."""
        response = await client.get("/api/v1/auth/me")

        assert response.status_code == 401


class TestAgentEnrollEndpoint:
    """Tests for POST /auth/agent/enroll endpoint."""

    async def test_agent_enroll_success(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Agent enrollment with valid token should succeed."""
        # Create enrollment token
        raw_token = generate_enrollment_token()
        token_hash = compute_token_hash(raw_token)

        token_record = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="enroll-test-token",
            token_hash=token_hash,
            capabilities=["trading", "backtest"],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        db_session.add(token_record)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": raw_token,
                "agent_name": "test-agent-001",
                "public_key": "a" * 64,  # 64 char minimum
                "agent_version": "1.0.0",
                "capabilities": ["trading"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "agent_id" in data
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["workspace_id"] == str(sample_workspace.id)

    async def test_agent_enroll_invalid_token(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent enrollment with invalid token should return 401."""
        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": "invalid-token-xyz",
                "agent_name": "test-agent",
                "public_key": "a" * 64,
                "agent_version": "1.0.0",
            },
        )

        assert response.status_code == 401
        assert "Invalid or expired enrollment token" in response.json()["detail"]

    async def test_agent_enroll_expired_token(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Agent enrollment with expired token should return 401."""
        # Create expired enrollment token
        raw_token = generate_enrollment_token()
        token_hash = compute_token_hash(raw_token)

        token_record = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="expired-token",
            token_hash=token_hash,
            capabilities=["trading"],
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired!
            created_by_user_id=sample_user.id,
        )
        db_session.add(token_record)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": raw_token,
                "agent_name": "test-agent",
                "public_key": "a" * 64,
                "agent_version": "1.0.0",
            },
        )

        assert response.status_code == 401
        assert "Invalid or expired enrollment token" in response.json()["detail"]

    async def test_agent_enroll_used_token(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Agent enrollment with already used token should return 401."""
        # Create used enrollment token
        raw_token = generate_enrollment_token()
        token_hash = compute_token_hash(raw_token)

        token_record = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="used-token",
            token_hash=token_hash,
            capabilities=["trading"],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
            is_used=True,  # Already used!
            used_at=datetime.now(timezone.utc),
        )
        db_session.add(token_record)
        await db_session.commit()

        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": raw_token,
                "agent_name": "test-agent",
                "public_key": "a" * 64,
                "agent_version": "1.0.0",
            },
        )

        assert response.status_code == 401
        assert "Invalid or expired enrollment token" in response.json()["detail"]

    async def test_agent_enroll_short_public_key(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent enrollment with short public key should return 422."""
        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": "some-token",
                "agent_name": "test-agent",
                "public_key": "short",  # Too short (< 64 chars)
                "agent_version": "1.0.0",
            },
        )

        assert response.status_code == 422

    async def test_agent_enroll_invalid_version_format(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent enrollment with invalid version format should return 422."""
        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": "some-token",
                "agent_name": "test-agent",
                "public_key": "a" * 64,
                "agent_version": "invalid",  # Not semver format
            },
        )

        assert response.status_code == 422

    async def test_agent_enroll_marks_token_used(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_user: User,
    ) -> None:
        """Agent enrollment should mark the token as used."""
        # Create enrollment token
        raw_token = generate_enrollment_token()
        token_hash = compute_token_hash(raw_token)

        token_record = AgentEnrollmentToken(
            workspace_id=sample_workspace.id,
            name="single-use-token",
            token_hash=token_hash,
            capabilities=["trading"],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by_user_id=sample_user.id,
        )
        db_session.add(token_record)
        await db_session.commit()
        await db_session.refresh(token_record)
        assert token_record.is_used is False

        # First enrollment should succeed
        response = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": raw_token,
                "agent_name": "first-agent",
                "public_key": "a" * 64,
                "agent_version": "1.0.0",
            },
        )
        assert response.status_code == 201

        # Refresh token record
        await db_session.refresh(token_record)
        assert token_record.is_used is True
        assert token_record.used_at is not None

        # Second enrollment with same token should fail
        response2 = await client.post(
            "/api/v1/auth/agent/enroll",
            json={
                "enrollment_token": raw_token,
                "agent_name": "second-agent",
                "public_key": "b" * 64,
                "agent_version": "1.0.0",
            },
        )
        assert response2.status_code == 401


class TestAgentHeartbeatEndpoint:
    """Tests for POST /auth/agent/heartbeat endpoint."""

    async def test_agent_heartbeat_success(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent heartbeat should return server time and status."""
        response = await client.post(
            "/api/v1/auth/agent/heartbeat",
            json={
                "agent_version": "1.0.0",
                "current_state": "running",
                "health_metrics": {"cpu": 50.0, "memory": 60.0},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "server_time" in data
        assert "trust_state" in data
        assert data["trust_state"] == TrustState.ENROLLED.value
        assert "pending_commands" in data
        assert "next_heartbeat_sec" in data
        assert data["next_heartbeat_sec"] == 60

    async def test_agent_heartbeat_with_run_id(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent heartbeat can include last_run_id."""
        run_id = uuid4()
        response = await client.post(
            "/api/v1/auth/agent/heartbeat",
            json={
                "agent_version": "1.0.0",
                "current_state": "running",
                "last_run_id": str(run_id),
                "health_metrics": {},
            },
        )

        assert response.status_code == 200

    async def test_agent_heartbeat_missing_fields(
        self,
        client: AsyncClient,
    ) -> None:
        """Agent heartbeat with missing required fields should return 422."""
        response = await client.post(
            "/api/v1/auth/agent/heartbeat",
            json={
                "current_state": "running",
                # Missing agent_version
            },
        )

        assert response.status_code == 422
