# -*- coding: utf-8 -*-
"""Tests for Telemetry Router."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Agent,
    Alert,
    AlertSeverity,
    Organization,
    TelemetryEvent,
    TelemetryLevel,
    Workspace,
)

pytestmark = pytest.mark.asyncio


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def sample_telemetry_event(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> TelemetryEvent:
    """Create a sample telemetry event."""
    event = TelemetryEvent(
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        run_id=uuid4(),
        event_type="strategy_signal",
        event_timestamp=datetime.now(timezone.utc),
        telemetry_level=TelemetryLevel.AGGREGATED.value,
        payload={"signal": "buy", "confidence": 0.85},
        redaction_applied=True,
        redaction_version="1.0.0",
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest.fixture
async def multiple_telemetry_events(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> list[TelemetryEvent]:
    """Create multiple telemetry events."""
    events = []
    base_time = datetime.now(timezone.utc)
    event_types = ["order_placed", "order_filled", "strategy_signal", "risk_check", "heartbeat"]

    for i, event_type in enumerate(event_types):
        event = TelemetryEvent(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            run_id=uuid4(),
            event_type=event_type,
            event_timestamp=base_time - timedelta(minutes=i * 10),
            telemetry_level=TelemetryLevel.AGGREGATED.value,
            payload={"index": i, "type": event_type},
            redaction_applied=True,
            redaction_version="1.0.0",
        )
        db_session.add(event)
        events.append(event)

    await db_session.commit()
    for event in events:
        await db_session.refresh(event)

    return events


@pytest.fixture
async def sample_alert(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> Alert:
    """Create a sample alert."""
    alert = Alert(
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        run_id=uuid4(),
        alert_type="margin_warning",
        severity=AlertSeverity.WARNING.value,
        title="Margin Level Low",
        message="Margin level dropped below 30%",
        context={"current_margin": 0.28, "threshold": 0.30},
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)
    return alert


@pytest.fixture
async def multiple_alerts(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> list[Alert]:
    """Create multiple alerts with different severities."""
    alerts = []
    severities = [
        AlertSeverity.INFO.value,
        AlertSeverity.WARNING.value,
        AlertSeverity.ERROR.value,
        AlertSeverity.CRITICAL.value,
    ]

    for i, severity in enumerate(severities):
        alert = Alert(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            alert_type=f"alert_type_{i}",
            severity=severity,
            title=f"Alert {i}",
            message=f"Alert message {i}",
            acknowledged=(i % 2 == 0),  # 0 and 2 are acknowledged
            resolved=(i == 0),  # only first is resolved
        )
        if i % 2 == 0:
            alert.acknowledged_by = "operator@example.com"
            alert.acknowledged_at = datetime.now(timezone.utc)
        if i == 0:
            alert.resolved_at = datetime.now(timezone.utc)
        db_session.add(alert)
        alerts.append(alert)

    await db_session.commit()
    for alert in alerts:
        await db_session.refresh(alert)

    return alerts


@pytest.fixture
async def telemetry_permission_headers(
    db_session: AsyncSession,
    sample_user,
    org_id,
    workspace_id,
) -> dict:
    """Create auth headers with telemetry permissions."""
    from ..models import Role, User
    from ..routers.auth import create_access_token
    from .conftest import get_or_create_permission
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    # Get the user with roles eagerly loaded
    result = await db_session.execute(
        select(User).where(User.id == sample_user.id).options(selectinload(User.roles))
    )
    user = result.scalar_one()

    # Create permissions (using get_or_create to avoid duplicates)
    read_perm = await get_or_create_permission(
        db_session, "telemetry:read", description="Read telemetry"
    )
    write_perm = await get_or_create_permission(
        db_session, "telemetry:write", description="Write telemetry"
    )

    # Create role with permissions
    role = Role(
        name="telemetry-role-test",
        description="Telemetry Role",
        organization_id=org_id,
    )
    role.permissions.append(read_perm)
    role.permissions.append(write_perm)
    db_session.add(role)
    await db_session.commit()
    await db_session.refresh(role)

    user.roles.append(role)
    await db_session.commit()

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        org_id=org_id,
        workspace_id=workspace_id,
        is_superuser=False,
        permissions=["telemetry:read", "telemetry:write"],
    )

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def alert_permission_headers(
    db_session: AsyncSession,
    sample_user,
    org_id,
    workspace_id,
) -> dict:
    """Create auth headers with alert permissions."""
    from ..models import Permission, Role, User
    from ..routers.auth import create_access_token
    from sqlalchemy import select

    # Get the user
    result = await db_session.execute(select(User).where(User.id == sample_user.id))
    user = result.scalar_one()

    # Create permissions
    perms = []
    for perm_name in ["alert:read", "alert:create", "alert:acknowledge", "alert:resolve"]:
        perm = Permission(name=perm_name, description=f"Permission {perm_name}")
        db_session.add(perm)
        perms.append(perm)

    await db_session.commit()
    for perm in perms:
        await db_session.refresh(perm)

    # Create role with permissions
    role = Role(
        name="alert-role-test",
        description="Alert Role",
        organization_id=org_id,
    )
    for perm in perms:
        role.permissions.append(perm)
    db_session.add(role)
    await db_session.commit()
    await db_session.refresh(role)

    user.roles.append(role)
    await db_session.commit()

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        org_id=org_id,
        workspace_id=workspace_id,
        is_superuser=False,
        permissions=["alert:read", "alert:create", "alert:acknowledge", "alert:resolve"],
    )

    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Telemetry Event Tests
# ============================================================================


class TestListTelemetryEvents:
    """Tests for GET /telemetry/events endpoint."""

    async def test_list_events_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_telemetry_event: TelemetryEvent,
    ) -> None:
        """Superuser can list all telemetry events."""
        response = await client.get(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_events_with_workspace_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_telemetry_event: TelemetryEvent,
        sample_workspace: Workspace,
    ) -> None:
        """Can filter events by workspace."""
        response = await client.get(
            f"/api/v1/telemetry/events?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["workspace_id"] == str(sample_workspace.id)

    async def test_list_events_with_agent_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_telemetry_event: TelemetryEvent,
        sample_agent: Agent,
    ) -> None:
        """Can filter events by agent."""
        response = await client.get(
            f"/api/v1/telemetry/events?agent_id={sample_agent.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["agent_id"] == str(sample_agent.id)

    async def test_list_events_with_event_type_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_telemetry_events: list[TelemetryEvent],
    ) -> None:
        """Can filter events by event_type."""
        response = await client.get(
            "/api/v1/telemetry/events?event_type=strategy_signal",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["event_type"] == "strategy_signal"

    async def test_list_events_with_time_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_telemetry_events: list[TelemetryEvent],
    ) -> None:
        """Can filter events by time range."""
        from urllib.parse import quote

        start_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        end_time = datetime.now(timezone.utc).isoformat()

        # URL-encode to handle the '+' in timezone offset ('+00:00' -> '%2B00:00')
        response = await client.get(
            f"/api/v1/telemetry/events?start_time={quote(start_time)}&end_time={quote(end_time)}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 0

    async def test_list_events_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """User without permission cannot list events."""
        response = await client.get(
            "/api/v1/telemetry/events",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "telemetry:read" in response.json()["detail"]

    async def test_list_events_unauthenticated(
        self,
        client: AsyncClient,
    ) -> None:
        """Unauthenticated request returns 401."""
        response = await client.get("/api/v1/telemetry/events")
        assert response.status_code == 401


class TestCreateTelemetryEvent:
    """Tests for POST /telemetry/events endpoint."""

    async def test_create_event_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Superuser can create telemetry event."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
            json={
                "agent_id": str(sample_agent.id),
                "event_type": "test_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "telemetry_level": TelemetryLevel.AGGREGATED.value,
                "payload": {"test": "data"},
                "redaction_applied": True,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == str(sample_agent.id)
        assert data["event_type"] == "test_event"
        assert data["payload"] == {"test": "data"}

    async def test_create_event_with_run_id(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Can create event with run_id."""
        run_id = uuid4()
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
            json={
                "agent_id": str(sample_agent.id),
                "run_id": str(run_id),
                "event_type": "run_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {"run": "data"},
                "redaction_applied": True,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["run_id"] == str(run_id)

    async def test_create_event_invalid_telemetry_level(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Cannot create event with invalid telemetry level."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
            json={
                "agent_id": str(sample_agent.id),
                "event_type": "test_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "telemetry_level": "invalid_level",
                "payload": {"test": "data"},
                "redaction_applied": True,
            },
        )

        assert response.status_code == 400
        assert "Invalid telemetry_level" in response.json()["detail"]

    async def test_create_event_raw_requires_superuser(
        self,
        client: AsyncClient,
        telemetry_permission_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """RAW_ORDER_EVENTS level requires enterprise tier (superuser)."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=telemetry_permission_headers,
            json={
                "agent_id": str(sample_agent.id),
                "event_type": "order_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "telemetry_level": TelemetryLevel.RAW_ORDER_EVENTS.value,
                "payload": {"order": "data"},
                "redaction_applied": True,
            },
        )

        assert response.status_code == 403
        assert "enterprise tier" in response.json()["detail"]

    async def test_create_event_detailed_requires_redaction(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Non-aggregated events require redaction_applied=True."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
            json={
                "agent_id": str(sample_agent.id),
                "event_type": "test_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "telemetry_level": TelemetryLevel.DETAILED_NON_SENSITIVE.value,
                "payload": {"test": "data"},
                "redaction_applied": False,
            },
        )

        assert response.status_code == 400
        assert "redaction_applied" in response.json()["detail"]

    async def test_create_event_agent_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot create event for non-existent agent."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=superuser_headers,
            json={
                "agent_id": str(uuid4()),
                "event_type": "test_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {"test": "data"},
                "redaction_applied": True,
            },
        )

        assert response.status_code == 404
        assert "Agent not found" in response.json()["detail"]

    async def test_create_event_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """User without permission cannot create event."""
        response = await client.post(
            "/api/v1/telemetry/events",
            headers=auth_headers,
            json={
                "agent_id": str(sample_agent.id),
                "event_type": "test_event",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {"test": "data"},
                "redaction_applied": True,
            },
        )

        # This will fail at workspace access or permission check
        assert response.status_code == 403


class TestCreateTelemetryEventsBatch:
    """Tests for POST /telemetry/events/batch endpoint."""

    async def test_batch_create_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Can create multiple events in batch."""
        events = [
            {
                "agent_id": str(sample_agent.id),
                "event_type": f"batch_event_{i}",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {"index": i},
                "redaction_applied": True,
            }
            for i in range(3)
        ]

        response = await client.post(
            "/api/v1/telemetry/events/batch",
            headers=superuser_headers,
            json={"events": events},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["created_count"] == 3
        assert len(data["event_ids"]) == 3

    async def test_batch_create_different_workspace_fails(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Batch events must belong to same workspace."""
        # Create another workspace with another agent
        ws2 = Workspace(
            name="workspace-2-batch",
            organization_id=sample_organization.id,
        )
        db_session.add(ws2)
        await db_session.commit()
        await db_session.refresh(ws2)

        agent2 = Agent(
            workspace_id=ws2.id,
            name="agent-2-batch",
            public_key="b" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent2)
        await db_session.commit()
        await db_session.refresh(agent2)

        events = [
            {
                "agent_id": str(sample_agent.id),
                "event_type": "event_1",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {},
                "redaction_applied": True,
            },
            {
                "agent_id": str(agent2.id),
                "event_type": "event_2",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {},
                "redaction_applied": True,
            },
        ]

        response = await client.post(
            "/api/v1/telemetry/events/batch",
            headers=superuser_headers,
            json={"events": events},
        )

        assert response.status_code == 400
        assert "same workspace" in response.json()["detail"]

    async def test_batch_create_invalid_agent(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_agent: Agent,
    ) -> None:
        """Batch fails if any agent is invalid."""
        events = [
            {
                "agent_id": str(sample_agent.id),
                "event_type": "event_1",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {},
                "redaction_applied": True,
            },
            {
                "agent_id": str(uuid4()),
                "event_type": "event_2",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {},
                "redaction_applied": True,
            },
        ]

        response = await client.post(
            "/api/v1/telemetry/events/batch",
            headers=superuser_headers,
            json={"events": events},
        )

        assert response.status_code == 404
        assert "Agent" in response.json()["detail"]


class TestGetTelemetryEvent:
    """Tests for GET /telemetry/events/{event_id} endpoint."""

    async def test_get_event_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_telemetry_event: TelemetryEvent,
    ) -> None:
        """Can get telemetry event by ID."""
        response = await client.get(
            f"/api/v1/telemetry/events/{sample_telemetry_event.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_telemetry_event.id)
        assert data["event_type"] == sample_telemetry_event.event_type
        assert "payload" in data

    async def test_get_event_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Getting non-existent event returns 404."""
        response = await client.get(
            f"/api/v1/telemetry/events/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    async def test_get_event_different_workspace_forbidden(
        self,
        client: AsyncClient,
        telemetry_permission_headers: dict,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """User cannot get event from different org's workspace."""
        # Create different org and workspace
        other_org = Organization(name="other-org-telemetry", display_name="Other")
        db_session.add(other_org)
        await db_session.commit()
        await db_session.refresh(other_org)

        other_ws = Workspace(
            name="other-ws-telemetry",
            organization_id=other_org.id,
        )
        db_session.add(other_ws)
        await db_session.commit()
        await db_session.refresh(other_ws)

        other_agent = Agent(
            workspace_id=other_ws.id,
            name="other-agent",
            public_key="x" * 64,
            agent_version="1.0.0",
        )
        db_session.add(other_agent)
        await db_session.commit()
        await db_session.refresh(other_agent)

        other_event = TelemetryEvent(
            workspace_id=other_ws.id,
            agent_id=other_agent.id,
            event_type="other_event",
            event_timestamp=datetime.now(timezone.utc),
            telemetry_level=TelemetryLevel.AGGREGATED.value,
            payload={"other": "data"},
            redaction_applied=True,
        )
        db_session.add(other_event)
        await db_session.commit()
        await db_session.refresh(other_event)

        response = await client.get(
            f"/api/v1/telemetry/events/{other_event.id}",
            headers=telemetry_permission_headers,
        )

        assert response.status_code == 403


class TestListTelemetryLevels:
    """Tests for GET /telemetry/events/types/list endpoint."""

    async def test_list_telemetry_levels(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Can get list of valid telemetry levels."""
        response = await client.get(
            "/api/v1/telemetry/events/types/list",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert TelemetryLevel.AGGREGATED.value in data
        assert TelemetryLevel.DETAILED_NON_SENSITIVE.value in data
        assert TelemetryLevel.RAW_ORDER_EVENTS.value in data

    async def test_list_telemetry_levels_sorted(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Telemetry levels are returned sorted."""
        response = await client.get(
            "/api/v1/telemetry/events/types/list",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data == sorted(data)


# ============================================================================
# Alert Tests
# ============================================================================


class TestListAlerts:
    """Tests for GET /telemetry/alerts endpoint."""

    async def test_list_alerts_superuser(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """Superuser can list all alerts."""
        response = await client.get(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    async def test_list_alerts_with_workspace_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
        sample_workspace: Workspace,
    ) -> None:
        """Can filter alerts by workspace."""
        response = await client.get(
            f"/api/v1/telemetry/alerts?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["workspace_id"] == str(sample_workspace.id)

    async def test_list_alerts_with_severity_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
    ) -> None:
        """Can filter alerts by severity."""
        response = await client.get(
            f"/api/v1/telemetry/alerts?severity={AlertSeverity.CRITICAL.value}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["severity"] == AlertSeverity.CRITICAL.value

    async def test_list_alerts_with_acknowledged_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
    ) -> None:
        """Can filter alerts by acknowledged status."""
        response = await client.get(
            "/api/v1/telemetry/alerts?acknowledged=false",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["acknowledged"] is False

    async def test_list_alerts_with_resolved_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
    ) -> None:
        """Can filter alerts by resolved status."""
        response = await client.get(
            "/api/v1/telemetry/alerts?resolved=true",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["resolved"] is True

    async def test_list_alerts_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """User without permission cannot list alerts."""
        response = await client.get(
            "/api/v1/telemetry/alerts",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "alert:read" in response.json()["detail"]


class TestCreateAlert:
    """Tests for POST /telemetry/alerts endpoint."""

    async def test_create_alert_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
        sample_agent: Agent,
    ) -> None:
        """Superuser can create alert."""
        response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "agent_id": str(sample_agent.id),
                "alert_type": "test_alert",
                "severity": AlertSeverity.WARNING.value,
                "title": "Test Alert",
                "message": "This is a test alert message",
                "context": {"key": "value"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["alert_type"] == "test_alert"
        assert data["severity"] == AlertSeverity.WARNING.value
        assert data["title"] == "Test Alert"
        assert data["acknowledged"] is False
        assert data["resolved"] is False

    async def test_create_alert_without_agent(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Can create alert without agent_id."""
        response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "alert_type": "system_alert",
                "severity": AlertSeverity.INFO.value,
                "title": "System Alert",
                "message": "System alert message",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] is None

    async def test_create_alert_invalid_severity(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """Cannot create alert with invalid severity."""
        response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "alert_type": "test_alert",
                "severity": "invalid_severity",
                "title": "Test Alert",
                "message": "Test message",
            },
        )

        assert response.status_code == 400
        assert "Invalid severity" in response.json()["detail"]

    async def test_create_alert_agent_wrong_workspace(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
        db_session: AsyncSession,
        sample_organization: Organization,
    ) -> None:
        """Cannot create alert with agent from different workspace."""
        # Create another workspace with agent
        ws2 = Workspace(
            name="workspace-2-alert",
            organization_id=sample_organization.id,
        )
        db_session.add(ws2)
        await db_session.commit()
        await db_session.refresh(ws2)

        agent2 = Agent(
            workspace_id=ws2.id,
            name="agent-2-alert",
            public_key="c" * 64,
            agent_version="1.0.0",
        )
        db_session.add(agent2)
        await db_session.commit()
        await db_session.refresh(agent2)

        response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "agent_id": str(agent2.id),  # Agent from different workspace
                "alert_type": "test_alert",
                "severity": AlertSeverity.WARNING.value,
                "title": "Test Alert",
                "message": "Test message",
            },
        )

        assert response.status_code == 404
        assert "Agent not found in this workspace" in response.json()["detail"]

    async def test_create_alert_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_workspace: Workspace,
    ) -> None:
        """User without permission cannot create alert."""
        response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=auth_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "alert_type": "test_alert",
                "severity": AlertSeverity.INFO.value,
                "title": "Test Alert",
                "message": "Test message",
            },
        )

        assert response.status_code == 403


class TestGetAlert:
    """Tests for GET /telemetry/alerts/{alert_id} endpoint."""

    async def test_get_alert_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """Can get alert by ID."""
        response = await client.get(
            f"/api/v1/telemetry/alerts/{sample_alert.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_alert.id)
        assert data["title"] == sample_alert.title
        assert data["message"] == sample_alert.message
        assert "context" in data

    async def test_get_alert_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Getting non-existent alert returns 404."""
        response = await client.get(
            f"/api/v1/telemetry/alerts/{uuid4()}",
            headers=superuser_headers,
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestAcknowledgeAlert:
    """Tests for POST /telemetry/alerts/{alert_id}/acknowledge endpoint."""

    async def test_acknowledge_alert_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """Can acknowledge an alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{sample_alert.id}/acknowledge",
            headers=superuser_headers,
            json={"acknowledged_by": "operator@example.com"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True
        assert data["acknowledged_by"] == "operator@example.com"
        assert data["acknowledged_at"] is not None

    async def test_acknowledge_alert_already_acknowledged(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_agent: Agent,
    ) -> None:
        """Cannot acknowledge already acknowledged alert."""
        # Create already acknowledged alert
        alert = Alert(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            alert_type="test",
            severity=AlertSeverity.INFO.value,
            title="Test",
            message="Test",
            acknowledged=True,
            acknowledged_by="someone",
            acknowledged_at=datetime.now(timezone.utc),
        )
        db_session.add(alert)
        await db_session.commit()
        await db_session.refresh(alert)

        response = await client.post(
            f"/api/v1/telemetry/alerts/{alert.id}/acknowledge",
            headers=superuser_headers,
            json={"acknowledged_by": "another@example.com"},
        )

        assert response.status_code == 400
        assert "already acknowledged" in response.json()["detail"]

    async def test_acknowledge_alert_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot acknowledge non-existent alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{uuid4()}/acknowledge",
            headers=superuser_headers,
            json={"acknowledged_by": "operator@example.com"},
        )

        assert response.status_code == 404

    async def test_acknowledge_alert_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """User without permission cannot acknowledge alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{sample_alert.id}/acknowledge",
            headers=auth_headers,
            json={"acknowledged_by": "operator@example.com"},
        )

        assert response.status_code == 403
        assert "alert:acknowledge" in response.json()["detail"]


class TestResolveAlert:
    """Tests for POST /telemetry/alerts/{alert_id}/resolve endpoint."""

    async def test_resolve_alert_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """Can resolve an alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{sample_alert.id}/resolve",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["resolved"] is True
        assert data["resolved_at"] is not None

    async def test_resolve_alert_already_resolved(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        db_session: AsyncSession,
        sample_workspace: Workspace,
        sample_agent: Agent,
    ) -> None:
        """Cannot resolve already resolved alert."""
        # Create already resolved alert
        alert = Alert(
            workspace_id=sample_workspace.id,
            agent_id=sample_agent.id,
            alert_type="test",
            severity=AlertSeverity.INFO.value,
            title="Test",
            message="Test",
            resolved=True,
            resolved_at=datetime.now(timezone.utc),
        )
        db_session.add(alert)
        await db_session.commit()
        await db_session.refresh(alert)

        response = await client.post(
            f"/api/v1/telemetry/alerts/{alert.id}/resolve",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 400
        assert "already resolved" in response.json()["detail"]

    async def test_resolve_alert_not_found(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Cannot resolve non-existent alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{uuid4()}/resolve",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 404

    async def test_resolve_alert_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """User without permission cannot resolve alert."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{sample_alert.id}/resolve",
            headers=auth_headers,
            json={},
        )

        assert response.status_code == 403
        assert "alert:resolve" in response.json()["detail"]


class TestGetAlertStats:
    """Tests for GET /telemetry/alerts/stats endpoint."""

    async def test_get_alert_stats_success(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
    ) -> None:
        """Can get alert statistics."""
        response = await client.get(
            "/api/v1/telemetry/alerts/stats",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "by_severity" in data
        assert "unacknowledged" in data
        assert "unresolved" in data
        assert data["total"] >= 4  # We created 4 alerts

    async def test_get_alert_stats_by_severity(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
    ) -> None:
        """Stats include count by severity."""
        response = await client.get(
            "/api/v1/telemetry/alerts/stats",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        by_severity = data["by_severity"]
        # We have one alert of each severity
        assert AlertSeverity.INFO.value in by_severity
        assert AlertSeverity.WARNING.value in by_severity
        assert AlertSeverity.ERROR.value in by_severity
        assert AlertSeverity.CRITICAL.value in by_severity

    async def test_get_alert_stats_with_workspace_filter(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        multiple_alerts: list[Alert],
        sample_workspace: Workspace,
    ) -> None:
        """Can filter stats by workspace."""
        response = await client.get(
            f"/api/v1/telemetry/alerts/stats?workspace_id={sample_workspace.id}",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 4

    async def test_get_alert_stats_without_permission(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ) -> None:
        """User without permission cannot get alert stats."""
        response = await client.get(
            "/api/v1/telemetry/alerts/stats",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "alert:read" in response.json()["detail"]


class TestListAlertSeverities:
    """Tests for GET /telemetry/alerts/severities/list endpoint."""

    async def test_list_alert_severities(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Can get list of valid alert severities."""
        response = await client.get(
            "/api/v1/telemetry/alerts/severities/list",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert AlertSeverity.INFO.value in data
        assert AlertSeverity.WARNING.value in data
        assert AlertSeverity.ERROR.value in data
        assert AlertSeverity.CRITICAL.value in data

    async def test_list_alert_severities_sorted(
        self,
        client: AsyncClient,
        superuser_headers: dict,
    ) -> None:
        """Alert severities are returned sorted."""
        response = await client.get(
            "/api/v1/telemetry/alerts/severities/list",
            headers=superuser_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data == sorted(data)


# ============================================================================
# Alert Workflow Tests
# ============================================================================


class TestAlertWorkflow:
    """Tests for complete alert workflow."""

    async def test_full_alert_lifecycle(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_workspace: Workspace,
        sample_agent: Agent,
    ) -> None:
        """Test complete alert lifecycle: create -> acknowledge -> resolve."""
        # Create alert
        create_response = await client.post(
            "/api/v1/telemetry/alerts",
            headers=superuser_headers,
            json={
                "workspace_id": str(sample_workspace.id),
                "agent_id": str(sample_agent.id),
                "alert_type": "lifecycle_test",
                "severity": AlertSeverity.ERROR.value,
                "title": "Lifecycle Test Alert",
                "message": "Testing complete lifecycle",
            },
        )
        assert create_response.status_code == 201
        alert_id = create_response.json()["id"]

        # Verify initial state
        get_response = await client.get(
            f"/api/v1/telemetry/alerts/{alert_id}",
            headers=superuser_headers,
        )
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["acknowledged"] is False
        assert data["resolved"] is False

        # Acknowledge
        ack_response = await client.post(
            f"/api/v1/telemetry/alerts/{alert_id}/acknowledge",
            headers=superuser_headers,
            json={"acknowledged_by": "operator@example.com"},
        )
        assert ack_response.status_code == 200
        assert ack_response.json()["acknowledged"] is True

        # Resolve
        resolve_response = await client.post(
            f"/api/v1/telemetry/alerts/{alert_id}/resolve",
            headers=superuser_headers,
            json={},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["resolved"] is True

        # Verify final state
        final_response = await client.get(
            f"/api/v1/telemetry/alerts/{alert_id}",
            headers=superuser_headers,
        )
        assert final_response.status_code == 200
        final_data = final_response.json()
        assert final_data["acknowledged"] is True
        assert final_data["resolved"] is True

    async def test_resolve_without_acknowledge(
        self,
        client: AsyncClient,
        superuser_headers: dict,
        sample_alert: Alert,
    ) -> None:
        """Can resolve alert without acknowledging first."""
        response = await client.post(
            f"/api/v1/telemetry/alerts/{sample_alert.id}/resolve",
            headers=superuser_headers,
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["resolved"] is True
        # acknowledged should still be False
        assert data["acknowledged"] is False
