# -*- coding: utf-8 -*-
"""
Comprehensive tests for GDPR Phase 4: Data Retention Service.

Tests cover:
- Retention Policy Registry (create, update, validate, delete)
- Legal Hold Service (create, release, expire, block)
- Auto-Purge Scheduler (run, skip, audit)
- Integration tests (purge + legal hold + audit)

GDPR Compliance:
- Art. 5(1)(e): Storage Limitation
- Minimum retention for compliance data (7 years)
- Legal hold prevents deletion
- Full audit trail

Run with: pytest -v packages/cloud/governance/tests/test_retention_service.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
from uuid import uuid4

import pytest

from ..retention_service import (
    # Constants
    MIN_RETENTION_DAYS,
    DEFAULT_RETENTION_DAYS,
    MAX_RETENTION_DAYS,
    COMPLIANCE_DATA_CATEGORIES,
    ALL_DATA_CATEGORIES,
    # Enums
    RetentionAction,
    PurgeStatus,
    LegalHoldStatus,
    # Data classes
    RetentionPolicy,
    LegalHold,
    PurgeEvent,
    RetentionValidationResult,
    PurgeSchedulerConfig,
    PurgeSchedulerState,
    # Services
    RetentionPolicyRegistry,
    LegalHoldService,
    AutoPurgeScheduler,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def workspace_id() -> str:
    """Generate a workspace ID."""
    return str(uuid4())


@pytest.fixture
def user_id() -> str:
    """Generate a user ID."""
    return str(uuid4())


@pytest.fixture
def audit_log() -> List[Dict[str, Any]]:
    """Capture audit events."""
    return []


@pytest.fixture
def audit_callback(audit_log: List[Dict[str, Any]]):
    """Create audit callback that captures events."""

    def callback(action: str, details: Dict[str, Any]):
        # Store action separately to avoid overwrite by nested "action" keys
        audit_log.append({"_action": action, **details})

    return callback


@pytest.fixture
def policy_registry(audit_callback) -> RetentionPolicyRegistry:
    """Create a policy registry with audit callback."""
    return RetentionPolicyRegistry(audit_callback=audit_callback)


@pytest.fixture
def legal_hold_service(audit_callback) -> LegalHoldService:
    """Create a legal hold service with audit callback."""
    return LegalHoldService(audit_callback=audit_callback)


@pytest.fixture
def purge_results() -> Dict[str, int]:
    """Track purge results."""
    return {"count": 0}


@pytest.fixture
def purge_handler(purge_results: Dict[str, int]):
    """Create a mock purge handler."""

    def handler(
        workspace_id: str,
        data_type: str,
        cutoff: datetime,
        action: RetentionAction,
        batch_size: int,
    ) -> int:
        # Simulate purging some records
        count = 100  # Simulated count
        purge_results["count"] += count
        return count

    return handler


@pytest.fixture
def purge_events() -> List[PurgeEvent]:
    """Capture purge events."""
    return []


@pytest.fixture
def purge_callback(purge_events: List[PurgeEvent]):
    """Create purge callback that captures events."""

    def callback(event: PurgeEvent):
        purge_events.append(event)

    return callback


@pytest.fixture
def scheduler(
    policy_registry: RetentionPolicyRegistry,
    legal_hold_service: LegalHoldService,
    purge_handler,
    purge_callback,
) -> AutoPurgeScheduler:
    """Create auto-purge scheduler."""
    config = PurgeSchedulerConfig(
        enabled=True,
        dry_run=False,
        batch_size=100,
    )
    return AutoPurgeScheduler(
        policy_registry=policy_registry,
        legal_hold_service=legal_hold_service,
        config=config,
        purge_handler=purge_handler,
        audit_callback=purge_callback,
    )


# ============================================================================
# RetentionPolicy Data Class Tests
# ============================================================================


class TestRetentionPolicy:
    """Tests for RetentionPolicy data class."""

    def test_policy_creation_with_defaults(self, workspace_id: str):
        """Test policy creation with default values."""
        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
        )

        assert policy.workspace_id == workspace_id
        assert policy.data_type == "telemetry_aggregated"
        assert policy.retention_days == 90  # Default
        assert policy.action == RetentionAction.DELETE
        assert policy.auto_purge_enabled is True
        assert policy.id is not None
        assert policy.created_at is not None

    def test_policy_enforces_minimum_retention(self, workspace_id: str):
        """Test that minimum retention is enforced."""
        # Try to create policy with less than minimum
        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="access_audits",
            retention_days=30,  # Less than 7-year minimum
        )

        # Should be adjusted to minimum
        assert policy.retention_days == MIN_RETENTION_DAYS["access_audits"]
        assert policy.retention_days == 2555  # 7 years

    def test_policy_enforces_maximum_retention(self, workspace_id: str):
        """Test that maximum retention is enforced."""
        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="telemetry_raw_order_events",
            retention_days=365,  # More than 30-day max
        )

        # Should be adjusted to maximum
        assert policy.retention_days == MAX_RETENTION_DAYS["telemetry_raw_order_events"]
        assert policy.retention_days == 30

    def test_policy_cutoff_date(self, workspace_id: str):
        """Test cutoff date calculation."""
        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            retention_days=90,
        )

        cutoff = policy.cutoff_date
        expected = datetime.now(timezone.utc) - timedelta(days=90)

        # Allow 1 second tolerance
        assert abs((cutoff - expected).total_seconds()) < 1

    def test_policy_is_compliance_data(self, workspace_id: str):
        """Test compliance data detection."""
        compliance_policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="access_audits",
        )
        assert compliance_policy.is_compliance_data is True

        regular_policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
        )
        assert regular_policy.is_compliance_data is False

    def test_policy_to_dict(self, workspace_id: str):
        """Test policy serialization."""
        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=365,
            action=RetentionAction.ARCHIVE,
        )

        data = policy.to_dict()

        assert data["workspace_id"] == workspace_id
        assert data["data_type"] == "alerts"
        assert data["retention_days"] == 365
        assert data["action"] == "archive"
        assert "cutoff_date" in data
        assert "is_compliance_data" in data


# ============================================================================
# LegalHold Data Class Tests
# ============================================================================


class TestLegalHold:
    """Tests for LegalHold data class."""

    def test_hold_creation(self, workspace_id: str, user_id: str):
        """Test legal hold creation."""
        hold = LegalHold(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            reason="Litigation case #12345",
            created_by=user_id,
        )

        assert hold.workspace_id == workspace_id
        assert hold.data_type == "telemetry_aggregated"
        assert hold.reason == "Litigation case #12345"
        assert hold.created_by == user_id
        assert hold.status == LegalHoldStatus.ACTIVE
        assert hold.is_active is True
        assert hold.hold_until is None  # Indefinite

    def test_hold_is_active_with_expiry(self, workspace_id: str, user_id: str):
        """Test hold active status with expiry date."""
        # Future expiry - active
        future_hold = LegalHold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Pending investigation",
            created_by=user_id,
            hold_until=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert future_hold.is_active is True

        # Past expiry - inactive
        past_hold = LegalHold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Old investigation",
            created_by=user_id,
            hold_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert past_hold.is_active is False

    def test_hold_released_is_inactive(self, workspace_id: str, user_id: str):
        """Test that released holds are inactive."""
        hold = LegalHold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation",
            created_by=user_id,
            status=LegalHoldStatus.RELEASED,
        )
        assert hold.is_active is False

    def test_hold_duration_days(self, workspace_id: str, user_id: str):
        """Test hold duration calculation."""
        # Create hold 10 days ago
        hold = LegalHold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation",
            created_by=user_id,
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )

        assert hold.duration_days == 10


# ============================================================================
# PurgeEvent Data Class Tests
# ============================================================================


class TestPurgeEvent:
    """Tests for PurgeEvent data class."""

    def test_event_creation(self, workspace_id: str):
        """Test purge event creation."""
        event = PurgeEvent(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            retention_days=90,
            records_deleted=1000,
        )

        assert event.workspace_id == workspace_id
        assert event.data_type == "telemetry_aggregated"
        assert event.records_deleted == 1000
        assert event.total_records_processed == 1000
        assert event.event_type == "purge_completed"

    def test_event_total_records(self, workspace_id: str):
        """Test total records calculation."""
        event = PurgeEvent(
            workspace_id=workspace_id,
            data_type="alerts",
            records_deleted=100,
            records_archived=50,
            records_anonymized=25,
            records_aggregated=10,
        )

        assert event.total_records_processed == 185

    def test_event_to_dict(self, workspace_id: str):
        """Test event serialization."""
        event = PurgeEvent(
            workspace_id=workspace_id,
            data_type="commands",
            retention_days=180,
            records_deleted=500,
            status=PurgeStatus.COMPLETED,
        )

        data = event.to_dict()

        assert data["workspace_id"] == workspace_id
        assert data["data_type"] == "commands"
        assert data["status"] == "completed"
        assert data["results"]["records_deleted"] == 500
        assert data["retention_config"]["retention_days"] == 180

    def test_event_compute_hash(self, workspace_id: str):
        """Test event integrity hash."""
        event = PurgeEvent(
            workspace_id=workspace_id,
            data_type="alerts",
            records_deleted=100,
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()

        assert hash1 == hash2
        assert hash1.startswith("sha256:")


# ============================================================================
# RetentionPolicyRegistry Tests
# ============================================================================


class TestRetentionPolicyRegistry:
    """Tests for RetentionPolicyRegistry."""

    def test_create_policy(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test policy creation."""
        policy, validation = policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            retention_days=90,
            created_by=user_id,
        )

        assert policy.workspace_id == workspace_id
        assert policy.data_type == "telemetry_aggregated"
        assert policy.retention_days == 90
        assert validation.is_valid is True
        assert validation.adjusted is False

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "policy_created"

    def test_create_policy_adjusts_to_minimum(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that policy is adjusted to minimum."""
        policy, validation = policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="access_audits",
            retention_days=30,  # Below 7-year minimum
        )

        assert policy.retention_days == 2555  # 7 years
        assert validation.is_valid is False
        assert validation.adjusted is True
        assert "Increased to minimum" in validation.reason

    def test_create_policy_adjusts_to_maximum(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that policy is adjusted to maximum."""
        policy, validation = policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="telemetry_raw_order_events",
            retention_days=365,  # Above 30-day max
        )

        assert policy.retention_days == 30
        assert validation.is_valid is False
        assert validation.adjusted is True
        assert "Reduced to maximum" in validation.reason

    def test_create_policy_invalid_data_type(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test error on invalid data type."""
        with pytest.raises(ValueError, match="Invalid data type"):
            policy_registry.create_policy(
                workspace_id=workspace_id,
                data_type="invalid_type",
            )

    def test_create_policy_uses_default_retention(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that default retention is used if not specified."""
        policy, _ = policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
        )

        assert policy.retention_days == DEFAULT_RETENTION_DAYS["alerts"]
        assert policy.retention_days == 365

    def test_get_policy(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test getting a policy."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="commands",
        )

        retrieved = policy_registry.get_policy(workspace_id, "commands")
        assert retrieved is not None
        assert retrieved.data_type == "commands"

        missing = policy_registry.get_policy(workspace_id, "nonexistent")
        assert missing is None

    def test_get_workspace_policies(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test getting all policies for a workspace."""
        policy_registry.create_policy(workspace_id, "alerts")
        policy_registry.create_policy(workspace_id, "commands")
        policy_registry.create_policy(workspace_id, "telemetry_aggregated")

        policies = policy_registry.get_workspace_policies(workspace_id)
        assert len(policies) == 3

    def test_update_policy(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test updating a policy."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=365,
        )
        audit_log.clear()

        updated, validation = policy_registry.update_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=180,
            action=RetentionAction.ARCHIVE,
            updated_by=user_id,
        )

        assert updated is not None
        assert updated.retention_days == 180
        assert updated.action == RetentionAction.ARCHIVE

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "policy_updated"

    def test_update_nonexistent_policy(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test updating nonexistent policy returns None."""
        result, _ = policy_registry.update_policy(
            workspace_id=workspace_id,
            data_type="nonexistent",
            retention_days=90,
        )
        assert result is None

    def test_delete_policy(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test deleting a policy."""
        policy_registry.create_policy(workspace_id, "alerts")
        audit_log.clear()

        deleted = policy_registry.delete_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            deleted_by=user_id,
        )

        assert deleted is True
        assert policy_registry.get_policy(workspace_id, "alerts") is None

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "policy_deleted"

    def test_delete_compliance_policy_blocked(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that compliance data policies cannot be deleted."""
        policy_registry.create_policy(workspace_id, "access_audits")

        deleted = policy_registry.delete_policy(
            workspace_id=workspace_id,
            data_type="access_audits",
        )

        assert deleted is False
        assert policy_registry.get_policy(workspace_id, "access_audits") is not None

    def test_validate_retention(
        self,
        policy_registry: RetentionPolicyRegistry,
    ):
        """Test retention validation."""
        # Valid retention
        result = policy_registry.validate_retention("alerts", 365)
        assert result.is_valid is True
        assert result.adjusted is False

        # Below minimum (compliance data)
        result = policy_registry.validate_retention("access_audits", 30)
        assert result.is_valid is False
        assert result.adjusted is True
        assert result.effective_days == 2555

        # Above maximum
        result = policy_registry.validate_retention("telemetry_raw_order_events", 100)
        assert result.is_valid is False
        assert result.effective_days == 30

    def test_update_last_purge(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test updating last purge timestamp."""
        policy_registry.create_policy(workspace_id, "alerts")

        policy_registry.update_last_purge(workspace_id, "alerts", 500)

        policy = policy_registry.get_policy(workspace_id, "alerts")
        assert policy.last_purge_at is not None
        assert policy.last_purge_count == 500

    def test_get_audit_log(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test getting audit log."""
        policy_registry.create_policy(workspace_id, "alerts")
        policy_registry.create_policy(workspace_id, "commands")

        log = policy_registry.get_audit_log(workspace_id)
        assert len(log) == 2

        all_log = policy_registry.get_audit_log()
        assert len(all_log) == 2


# ============================================================================
# LegalHoldService Tests
# ============================================================================


class TestLegalHoldService:
    """Tests for LegalHoldService."""

    def test_create_hold(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test creating a legal hold."""
        hold = legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            reason="Litigation case #12345 requires data preservation",
            created_by=user_id,
        )

        assert hold.workspace_id == workspace_id
        assert hold.data_type == "telemetry_aggregated"
        assert hold.is_active is True
        assert hold.hold_until is None  # Indefinite

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "legal_hold_created"

    def test_create_hold_with_expiry(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test creating a hold with expiry date."""
        expiry = datetime.now(timezone.utc) + timedelta(days=180)

        hold = legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation expected to conclude in 6 months",
            created_by=user_id,
            hold_until=expiry,
        )

        assert hold.hold_until == expiry
        assert hold.is_active is True

    def test_create_hold_requires_reason(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test that hold requires meaningful reason."""
        with pytest.raises(ValueError, match="at least 10 characters"):
            legal_hold_service.create_hold(
                workspace_id=workspace_id,
                data_type="alerts",
                reason="Short",  # Too short
                created_by=user_id,
            )

    def test_create_duplicate_hold_blocked(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test that duplicate active holds are blocked."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="First investigation case",
            created_by=user_id,
        )

        with pytest.raises(ValueError, match="already exists"):
            legal_hold_service.create_hold(
                workspace_id=workspace_id,
                data_type="alerts",
                reason="Second investigation case",
                created_by=user_id,
            )

    def test_get_active_hold(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test getting active hold."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation active",
            created_by=user_id,
        )

        hold = legal_hold_service.get_active_hold(workspace_id, "alerts")
        assert hold is not None
        assert hold.data_type == "alerts"

        missing = legal_hold_service.get_active_hold(workspace_id, "commands")
        assert missing is None

    def test_is_data_held(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test checking if data is held."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation active",
            created_by=user_id,
        )

        assert legal_hold_service.is_data_held(workspace_id, "alerts") is True
        assert legal_hold_service.is_data_held(workspace_id, "commands") is False

    def test_release_hold(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test releasing a legal hold."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case #12345",
            created_by=user_id,
        )
        audit_log.clear()

        released = legal_hold_service.release_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            released_by=user_id,
            release_reason="Investigation concluded with no action required",
        )

        assert released is not None
        assert released.status == LegalHoldStatus.RELEASED
        assert released.is_active is False
        assert released.released_by == user_id
        assert released.released_at is not None

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "legal_hold_released"

    def test_release_hold_requires_reason(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test that release requires meaningful reason."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case",
            created_by=user_id,
        )

        with pytest.raises(ValueError, match="at least 10 characters"):
            legal_hold_service.release_hold(
                workspace_id=workspace_id,
                data_type="alerts",
                released_by=user_id,
                release_reason="Short",  # Too short
            )

    def test_check_and_expire_holds(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test automatic expiry of holds."""
        # Create hold that has already expired
        hold = legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Short-term investigation",
            created_by=user_id,
            hold_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        audit_log.clear()

        expired = legal_hold_service.check_and_expire_holds()

        assert len(expired) == 1
        assert expired[0].status == LegalHoldStatus.EXPIRED
        assert expired[0].is_active is False

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "legal_hold_expired"

    def test_extend_hold(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
        audit_log: List[Dict[str, Any]],
    ):
        """Test extending a hold's duration."""
        original_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case",
            created_by=user_id,
            hold_until=original_expiry,
        )
        audit_log.clear()

        new_expiry = datetime.now(timezone.utc) + timedelta(days=90)
        extended = legal_hold_service.extend_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            new_hold_until=new_expiry,
            extended_by=user_id,
            extension_reason="Investigation ongoing, extension required",
        )

        assert extended is not None
        assert extended.hold_until == new_expiry

        # Check audit
        assert len(audit_log) == 1
        assert audit_log[0]["_action"] == "legal_hold_extended"

    def test_get_workspace_holds(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test getting all holds for a workspace."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation A",
            created_by=user_id,
        )
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="commands",
            reason="Investigation B",
            created_by=user_id,
        )

        holds = legal_hold_service.get_workspace_holds(workspace_id)
        assert len(holds) == 2

    def test_get_all_active_holds(
        self,
        legal_hold_service: LegalHoldService,
        user_id: str,
    ):
        """Test getting all active holds across workspaces."""
        ws1 = str(uuid4())
        ws2 = str(uuid4())

        legal_hold_service.create_hold(
            workspace_id=ws1,
            data_type="alerts",
            reason="Investigation 1",
            created_by=user_id,
        )
        legal_hold_service.create_hold(
            workspace_id=ws2,
            data_type="commands",
            reason="Investigation 2",
            created_by=user_id,
        )

        holds = legal_hold_service.get_all_active_holds()
        assert len(holds) == 2

    def test_get_hold_history(
        self,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test getting hold history."""
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case",
            created_by=user_id,
        )
        legal_hold_service.release_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            released_by=user_id,
            release_reason="Investigation concluded",
        )

        history = legal_hold_service.get_hold_history(workspace_id)
        assert len(history) == 1
        assert history[0].status == LegalHoldStatus.RELEASED


# ============================================================================
# AutoPurgeScheduler Tests
# ============================================================================


class TestAutoPurgeScheduler:
    """Tests for AutoPurgeScheduler."""

    def test_run_purge_basic(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
        purge_events: List[PurgeEvent],
    ):
        """Test basic purge execution."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=30,
        )

        events = scheduler.run_purge(workspace_id)

        assert len(events) == 1
        assert events[0].status == PurgeStatus.COMPLETED
        assert events[0].records_deleted == 100  # From mock handler
        assert events[0].workspace_id == workspace_id
        assert events[0].data_type == "alerts"

        # Check callback was called
        assert len(purge_events) == 1

    def test_run_purge_blocked_by_legal_hold(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
        purge_events: List[PurgeEvent],
    ):
        """Test that purge is blocked by legal hold."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=30,
        )
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation requires data preservation",
            created_by=user_id,
        )

        events = scheduler.run_purge(workspace_id)

        assert len(events) == 1
        assert events[0].status == PurgeStatus.SKIPPED
        assert events[0].legal_hold_blocked is True
        assert events[0].records_deleted == 0

    def test_run_purge_multiple_policies(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test purging multiple data types."""
        policy_registry.create_policy(workspace_id, "alerts")
        policy_registry.create_policy(workspace_id, "commands")
        policy_registry.create_policy(workspace_id, "telemetry_aggregated")

        events = scheduler.run_purge(workspace_id)

        assert len(events) == 3
        assert all(e.status == PurgeStatus.COMPLETED for e in events)

    def test_run_purge_skips_disabled_policies(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that disabled policies are skipped."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            auto_purge_enabled=True,
        )
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="commands",
            auto_purge_enabled=False,
        )

        events = scheduler.run_purge(workspace_id)

        # Only one event for enabled policy
        assert len(events) == 1
        assert events[0].data_type == "alerts"

    def test_run_purge_updates_state(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that scheduler state is updated."""
        policy_registry.create_policy(workspace_id, "alerts")

        assert scheduler.state.total_runs == 0

        scheduler.run_purge(workspace_id)

        state = scheduler.state
        assert state.total_runs == 1
        assert state.last_run_at is not None
        assert state.total_records_purged == 100
        assert state.is_running is False

    def test_preview_purge(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test purge preview."""
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
            retention_days=30,
        )

        preview = scheduler.preview_purge(workspace_id, "alerts")

        assert preview["workspace_id"] == workspace_id
        assert preview["data_type"] == "alerts"
        assert preview["legal_hold_active"] is False
        assert preview["would_be_blocked"] is False
        assert "cutoff_date" in preview

    def test_preview_purge_with_legal_hold(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        workspace_id: str,
        user_id: str,
    ):
        """Test purge preview with active legal hold."""
        policy_registry.create_policy(workspace_id, "alerts")
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case",
            created_by=user_id,
        )

        preview = scheduler.preview_purge(workspace_id, "alerts")

        assert preview["legal_hold_active"] is True
        assert preview["would_be_blocked"] is True
        assert preview["legal_hold"] is not None

    def test_get_purge_statistics(
        self,
        scheduler: AutoPurgeScheduler,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test purge statistics."""
        policy_registry.create_policy(workspace_id, "alerts")
        policy_registry.create_policy(workspace_id, "commands")

        scheduler.run_purge(workspace_id)
        scheduler.run_purge(workspace_id)

        stats = scheduler.get_purge_statistics(workspace_id, days=30)

        assert stats["total_runs"] == 4  # 2 policies x 2 runs
        assert stats["successful_runs"] == 4
        assert stats["failed_runs"] == 0
        assert stats["total_records_deleted"] == 400
        assert "by_data_type" in stats

    def test_dry_run_mode(
        self,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        purge_handler,
        purge_callback,
        workspace_id: str,
        purge_results: Dict[str, int],
    ):
        """Test dry run mode doesn't delete data."""
        config = PurgeSchedulerConfig(dry_run=True)
        scheduler = AutoPurgeScheduler(
            policy_registry=policy_registry,
            legal_hold_service=legal_hold_service,
            config=config,
            purge_handler=purge_handler,
            audit_callback=purge_callback,
        )

        policy_registry.create_policy(workspace_id, "alerts")
        events = scheduler.run_purge(workspace_id)

        assert len(events) == 1
        assert events[0].status == PurgeStatus.COMPLETED
        assert events[0].event_type == "purge_completed_dry_run"
        assert purge_results["count"] == 0  # Handler not called


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the retention system."""

    def test_full_workflow_purge_with_audit(
        self,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        purge_handler,
        workspace_id: str,
        user_id: str,
    ):
        """Test full workflow: create policy, run purge, verify audit."""
        purge_events: List[PurgeEvent] = []

        scheduler = AutoPurgeScheduler(
            policy_registry=policy_registry,
            legal_hold_service=legal_hold_service,
            config=PurgeSchedulerConfig(),
            purge_handler=purge_handler,
            audit_callback=lambda e: purge_events.append(e),
        )

        # 1. Create retention policy
        policy, _ = policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="telemetry_aggregated",
            retention_days=90,
            created_by=user_id,
        )

        # 2. Run purge
        events = scheduler.run_purge(workspace_id)

        # 3. Verify results
        assert len(events) == 1
        assert events[0].status == PurgeStatus.COMPLETED
        assert events[0].records_deleted == 100

        # 4. Verify audit
        assert len(purge_events) == 1
        audit_event = purge_events[0]
        assert audit_event.workspace_id == workspace_id
        assert audit_event.data_type == "telemetry_aggregated"
        assert audit_event.event_id is not None
        assert audit_event.completed_at is not None
        assert audit_event.duration_seconds > 0

        # 5. Verify policy updated
        updated_policy = policy_registry.get_policy(workspace_id, "telemetry_aggregated")
        assert updated_policy.last_purge_at is not None
        assert updated_policy.last_purge_count == 100

    def test_legal_hold_blocks_purge_then_allows_after_release(
        self,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        purge_handler,
        workspace_id: str,
        user_id: str,
    ):
        """Test that legal hold blocks purge, then allows after release."""
        scheduler = AutoPurgeScheduler(
            policy_registry=policy_registry,
            legal_hold_service=legal_hold_service,
            config=PurgeSchedulerConfig(),
            purge_handler=purge_handler,
        )

        # 1. Create policy
        policy_registry.create_policy(
            workspace_id=workspace_id,
            data_type="alerts",
        )

        # 2. Create legal hold
        legal_hold_service.create_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            reason="Investigation case #12345 requires data preservation",
            created_by=user_id,
        )

        # 3. Attempt purge - should be blocked
        events = scheduler.run_purge(workspace_id)
        assert len(events) == 1
        assert events[0].status == PurgeStatus.SKIPPED
        assert events[0].legal_hold_blocked is True
        assert events[0].records_deleted == 0

        # 4. Release hold
        legal_hold_service.release_hold(
            workspace_id=workspace_id,
            data_type="alerts",
            released_by=user_id,
            release_reason="Investigation concluded, hold no longer required",
        )

        # 5. Attempt purge again - should succeed
        events = scheduler.run_purge(workspace_id)
        assert len(events) == 1
        assert events[0].status == PurgeStatus.COMPLETED
        assert events[0].legal_hold_blocked is False
        assert events[0].records_deleted == 100

    def test_compliance_data_minimum_retention_enforced(
        self,
        policy_registry: RetentionPolicyRegistry,
        workspace_id: str,
    ):
        """Test that compliance data categories enforce minimum retention."""
        for category in COMPLIANCE_DATA_CATEGORIES:
            policy, validation = policy_registry.create_policy(
                workspace_id=workspace_id,
                data_type=category,
                retention_days=30,  # Try to set below minimum
            )

            # All should be adjusted to 7 years
            assert policy.retention_days >= 2555
            assert validation.adjusted is True

    def test_all_data_categories_valid(self):
        """Test that all data categories are valid."""
        # Ensure MIN_RETENTION_DAYS has entries for all compliance categories
        for category in COMPLIANCE_DATA_CATEGORIES:
            assert category in MIN_RETENTION_DAYS, f"Missing min retention for {category}"
            assert MIN_RETENTION_DAYS[category] >= 2555, f"{category} should have 7-year minimum"

        # Ensure DEFAULT_RETENTION_DAYS has sensible defaults
        for category in ALL_DATA_CATEGORIES:
            if category in DEFAULT_RETENTION_DAYS:
                assert DEFAULT_RETENTION_DAYS[category] > 0

    def test_purge_event_integrity_hash(self, workspace_id: str):
        """Test that purge event hash provides integrity."""
        event = PurgeEvent(
            workspace_id=workspace_id,
            data_type="alerts",
            records_deleted=1000,
            status=PurgeStatus.COMPLETED,
        )

        hash1 = event.compute_hash()

        # Same event should produce same hash
        hash2 = event.compute_hash()
        assert hash1 == hash2

        # Modified event should produce different hash
        event.records_deleted = 2000
        hash3 = event.compute_hash()
        assert hash1 != hash3

    def test_multiple_workspaces_isolation(
        self,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        purge_handler,
        user_id: str,
    ):
        """Test that workspaces are properly isolated."""
        ws1 = str(uuid4())
        ws2 = str(uuid4())

        scheduler = AutoPurgeScheduler(
            policy_registry=policy_registry,
            legal_hold_service=legal_hold_service,
            config=PurgeSchedulerConfig(),
            purge_handler=purge_handler,
        )

        # Create policies in both workspaces
        policy_registry.create_policy(ws1, "alerts")
        policy_registry.create_policy(ws2, "alerts")

        # Create hold only in ws1
        legal_hold_service.create_hold(
            workspace_id=ws1,
            data_type="alerts",
            reason="Investigation in workspace 1",
            created_by=user_id,
        )

        # Run purge for ws1 - should be blocked
        events1 = scheduler.run_purge(ws1)
        assert events1[0].status == PurgeStatus.SKIPPED
        assert events1[0].legal_hold_blocked is True

        # Run purge for ws2 - should succeed
        events2 = scheduler.run_purge(ws2)
        assert events2[0].status == PurgeStatus.COMPLETED
        assert events2[0].legal_hold_blocked is False


# ============================================================================
# Constant Validation Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_compliance_categories_have_7_year_minimum(self):
        """Test that all compliance categories have 7-year minimum."""
        seven_years = 2555

        for category in COMPLIANCE_DATA_CATEGORIES:
            assert category in MIN_RETENTION_DAYS
            assert (
                MIN_RETENTION_DAYS[category] >= seven_years
            ), f"{category} should have at least 7-year minimum"

    def test_all_categories_in_constants(self):
        """Test that all categories are properly defined."""
        for category in ALL_DATA_CATEGORIES:
            # Each category should have at least a default or minimum
            has_default = category in DEFAULT_RETENTION_DAYS
            has_min = category in MIN_RETENTION_DAYS

            # Customer-controlled data might not have min
            if category not in {"strategy_versions", "backtest_results", "research_artifacts"}:
                assert has_default or has_min, f"{category} missing from retention constants"

    def test_max_greater_than_min(self):
        """Test that max retention is always greater than min."""
        for category, max_days in MAX_RETENTION_DAYS.items():
            if max_days is None:
                continue  # Indefinite is always valid

            min_days = MIN_RETENTION_DAYS.get(category, 1)
            assert (
                max_days >= min_days
            ), f"{category}: max ({max_days}) should be >= min ({min_days})"

    def test_default_within_bounds(self):
        """Test that defaults are within min/max bounds."""
        for category, default in DEFAULT_RETENTION_DAYS.items():
            min_days = MIN_RETENTION_DAYS.get(category, 1)
            max_days = MAX_RETENTION_DAYS.get(category)

            assert (
                default >= min_days
            ), f"{category}: default ({default}) should be >= min ({min_days})"

            if max_days is not None:
                assert (
                    default <= max_days
                ), f"{category}: default ({default}) should be <= max ({max_days})"
