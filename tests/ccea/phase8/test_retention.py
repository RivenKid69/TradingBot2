# -*- coding: utf-8 -*-
"""
Tests for RetentionService.

CCEA Phase 8 - Data retention and auto-purge tests.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from packages.cloud.governance.retention import (
    RetentionService,
    RetentionConfig,
    RetentionPolicy,
    RetentionAction,
    PurgeResult,
    DEFAULT_RETENTION_PERIODS,
    MIN_RETENTION_PERIODS,
)


@pytest.fixture
def retention_service():
    """Create retention service with mock purger."""
    purge_counts = {}

    def mock_purger(workspace_id, data_type, cutoff, action):
        key = f"{workspace_id}:{data_type}"
        purge_counts[key] = purge_counts.get(key, 0) + 10
        return 10  # Purged 10 records

    return RetentionService(purger=mock_purger)


class TestRetentionServiceBasic:
    """Basic retention service tests."""

    def test_create_service(self):
        """Test creating retention service."""
        service = RetentionService()
        assert service is not None
        assert service.config is not None

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = RetentionConfig(
            auto_purge_enabled=True,
            purge_batch_size=500,
        )
        service = RetentionService(config)

        assert service.config.purge_batch_size == 500


class TestPolicyCreation:
    """Policy creation tests."""

    def test_create_policy(self, retention_service):
        """Test creating retention policy."""
        policy = retention_service.create_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=30,
        )

        assert policy.id is not None
        assert policy.workspace_id == "ws-123"
        assert policy.data_type == "telemetry_events"
        assert policy.retention_days == 30

    def test_policy_uses_default_retention(self, retention_service):
        """Test policy uses default retention if not specified."""
        policy = retention_service.create_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
        )

        expected = DEFAULT_RETENTION_PERIODS.get("telemetry_events", 90)
        assert policy.retention_days == expected

    def test_policy_enforces_minimum(self, retention_service):
        """Test policy enforces minimum retention."""
        policy = retention_service.create_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=1,  # Below minimum
        )

        min_days = MIN_RETENTION_PERIODS.get("telemetry_events", 7)
        assert policy.retention_days >= min_days

    def test_create_policy_with_action(self, retention_service):
        """Test creating policy with specific action."""
        policy = retention_service.create_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            action=RetentionAction.ARCHIVE,
        )

        assert policy.action == RetentionAction.ARCHIVE


class TestPolicyManagement:
    """Policy management tests."""

    def test_get_policy(self, retention_service):
        """Test getting policy."""
        created = retention_service.create_policy("ws-123", "telemetry_events")
        retrieved = retention_service.get_policy("ws-123", "telemetry_events")

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_workspace_policies(self, retention_service):
        """Test getting all policies for workspace."""
        retention_service.create_policy("ws-123", "telemetry_events")
        retention_service.create_policy("ws-123", "alerts")

        policies = retention_service.get_workspace_policies("ws-123")

        assert len(policies) == 2

    def test_update_policy(self, retention_service):
        """Test updating policy."""
        retention_service.create_policy("ws-123", "telemetry_events", 30)

        updated = retention_service.update_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=60,
        )

        assert updated is not None
        assert updated.retention_days == 60

    def test_update_enforces_minimum(self, retention_service):
        """Test update enforces minimum retention."""
        retention_service.create_policy("ws-123", "telemetry_events")

        updated = retention_service.update_policy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=1,  # Below minimum
        )

        min_days = MIN_RETENTION_PERIODS.get("telemetry_events", 7)
        assert updated.retention_days >= min_days


class TestLegalHold:
    """Legal hold tests."""

    def test_set_legal_hold(self, retention_service):
        """Test setting legal hold."""
        retention_service.create_policy("ws-123", "telemetry_events")

        result = retention_service.set_legal_hold(
            workspace_id="ws-123",
            data_type="telemetry_events",
            reason="Legal investigation",
        )

        assert result is True
        policy = retention_service.get_policy("ws-123", "telemetry_events")
        assert policy.legal_hold is True

    def test_set_legal_hold_creates_policy(self, retention_service):
        """Test setting legal hold creates policy if needed."""
        result = retention_service.set_legal_hold(
            workspace_id="ws-new",
            data_type="alerts",
            reason="Compliance audit",
        )

        assert result is True
        policy = retention_service.get_policy("ws-new", "alerts")
        assert policy is not None
        assert policy.legal_hold is True

    def test_legal_hold_with_expiration(self, retention_service):
        """Test legal hold with expiration date."""
        until = datetime.utcnow() + timedelta(days=90)

        retention_service.set_legal_hold(
            workspace_id="ws-123",
            data_type="alerts",
            reason="Investigation",
            until=until,
        )

        policy = retention_service.get_policy("ws-123", "alerts")
        assert policy.legal_hold_until == until

    def test_release_legal_hold(self, retention_service):
        """Test releasing legal hold."""
        retention_service.create_policy("ws-123", "telemetry_events")
        retention_service.set_legal_hold("ws-123", "telemetry_events", "reason")

        result = retention_service.release_legal_hold(
            workspace_id="ws-123",
            data_type="telemetry_events",
            released_by="admin",
        )

        assert result is True
        policy = retention_service.get_policy("ws-123", "telemetry_events")
        assert policy.legal_hold is False


class TestIsOnLegalHold:
    """Legal hold status tests."""

    def test_is_on_legal_hold_active(self, retention_service):
        """Test is_on_legal_hold when active."""
        retention_service.create_policy("ws-123", "alerts")
        retention_service.set_legal_hold("ws-123", "alerts", "reason")

        policy = retention_service.get_policy("ws-123", "alerts")
        assert policy.is_on_legal_hold is True

    def test_is_on_legal_hold_expired(self, retention_service):
        """Test is_on_legal_hold when expired."""
        past = datetime.utcnow() - timedelta(days=1)

        retention_service.set_legal_hold(
            workspace_id="ws-123",
            data_type="alerts",
            reason="reason",
            until=past,
        )

        policy = retention_service.get_policy("ws-123", "alerts")
        assert policy.is_on_legal_hold is False


class TestPurgeExecution:
    """Purge execution tests."""

    def test_run_purge(self, retention_service):
        """Test running purge."""
        retention_service.create_policy("ws-123", "telemetry_events", 30)

        results = asyncio.run(retention_service.run_purge("ws-123"))

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].records_purged > 0

    def test_purge_respects_legal_hold(self, retention_service):
        """Test purge respects legal hold."""
        retention_service.create_policy("ws-123", "telemetry_events")
        retention_service.set_legal_hold("ws-123", "telemetry_events", "hold")

        results = asyncio.run(retention_service.run_purge("ws-123"))

        assert results[0].error == "Data on legal hold"
        assert results[0].records_purged == 0

    def test_purge_respects_disabled_policy(self, retention_service):
        """Test purge respects disabled policy."""
        retention_service.create_policy("ws-123", "telemetry_events")
        retention_service.update_policy("ws-123", "telemetry_events", enabled=False)

        results = asyncio.run(retention_service.run_purge("ws-123"))

        assert results[0].error == "Policy disabled"

    def test_purge_all_workspaces(self, retention_service):
        """Test purge across all workspaces."""
        retention_service.create_policy("ws-1", "telemetry_events")
        retention_service.create_policy("ws-2", "telemetry_events")

        results = asyncio.run(retention_service.run_purge())

        assert len(results) == 2


class TestDryRunMode:
    """Dry run mode tests."""

    def test_dry_run_no_deletion(self):
        """Test dry run doesn't delete data."""
        delete_called = []

        def mock_purger(workspace_id, data_type, cutoff, action):
            delete_called.append(True)
            return 10

        config = RetentionConfig(dry_run_mode=True)
        service = RetentionService(config=config, purger=mock_purger)
        service.create_policy("ws-123", "telemetry_events")

        results = asyncio.run(service.run_purge("ws-123"))

        assert len(delete_called) == 0  # Purger not called
        assert results[0].was_dry_run is True


class TestRetentionReport:
    """Retention report tests."""

    def test_get_retention_report(self, retention_service):
        """Test getting retention report."""
        retention_service.create_policy("ws-123", "telemetry_events", 30)
        retention_service.create_policy("ws-123", "alerts", 90)

        report = retention_service.get_retention_report("ws-123")

        assert report["workspace_id"] == "ws-123"
        assert len(report["policies"]) == 2
        assert "upcoming_purges" in report

    def test_report_includes_legal_holds(self, retention_service):
        """Test report includes legal holds."""
        retention_service.create_policy("ws-123", "alerts")
        retention_service.set_legal_hold("ws-123", "alerts", "investigation")

        report = retention_service.get_retention_report("ws-123")

        assert len(report["legal_holds"]) == 1


class TestCutoffDate:
    """Cutoff date tests."""

    def test_cutoff_date_calculation(self):
        """Test cutoff date calculation."""
        policy = RetentionPolicy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=30,
        )

        cutoff = policy.cutoff_date
        expected = datetime.utcnow() - timedelta(days=30)

        # Should be within a second of expected
        delta = abs((cutoff - expected).total_seconds())
        assert delta < 1


class TestAuditLog:
    """Audit log tests."""

    def test_audit_log_on_policy_create(self, retention_service):
        """Test audit log on policy creation."""
        retention_service.create_policy("ws-123", "telemetry_events")

        log = retention_service.get_audit_log(workspace_id="ws-123")

        assert len(log) > 0
        assert log[0]["action"] == "policy_created"

    def test_audit_log_on_legal_hold(self, retention_service):
        """Test audit log on legal hold."""
        retention_service.create_policy("ws-123", "alerts")
        retention_service.set_legal_hold("ws-123", "alerts", "reason")

        log = retention_service.get_audit_log(workspace_id="ws-123")

        actions = [e["action"] for e in log]
        assert "legal_hold_set" in actions


class TestPolicySerialization:
    """Policy serialization tests."""

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = RetentionPolicy(
            workspace_id="ws-123",
            data_type="telemetry_events",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        data = policy.to_dict()

        assert data["workspace_id"] == "ws-123"
        assert data["retention_days"] == 30
        assert data["action"] == "DELETE"

    def test_purge_result_to_dict(self):
        """Test purge result serialization."""
        result = PurgeResult(
            success=True,
            workspace_id="ws-123",
            data_type="telemetry_events",
            records_purged=100,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["records_purged"] == 100


class TestRetentionConstants:
    """Retention constants tests."""

    def test_default_periods_defined(self):
        """Test default retention periods are defined."""
        assert "telemetry_events" in DEFAULT_RETENTION_PERIODS
        assert "alerts" in DEFAULT_RETENTION_PERIODS
        assert "access_audits" in DEFAULT_RETENTION_PERIODS

    def test_minimum_periods_defined(self):
        """Test minimum retention periods are defined."""
        assert "telemetry_events" in MIN_RETENTION_PERIODS
        assert "access_audits" in MIN_RETENTION_PERIODS

    def test_compliance_data_long_retention(self):
        """Test compliance data has long retention."""
        # Access audits and approval records should have 7+ years
        assert DEFAULT_RETENTION_PERIODS.get("access_audits", 0) >= 365
        assert DEFAULT_RETENTION_PERIODS.get("approval_records", 0) >= 365


class TestServiceWithoutPurger:
    """Tests for service without purger."""

    def test_purge_fails_without_purger(self):
        """Test purge fails gracefully without purger."""
        service = RetentionService()
        service.create_policy("ws-123", "telemetry_events")

        results = asyncio.run(service.run_purge("ws-123"))

        assert results[0].success is False
        assert "not configured" in results[0].error
