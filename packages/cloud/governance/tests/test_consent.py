# -*- coding: utf-8 -*-
"""
Tests for Support Consent Service.

GDPR Phase 1 Implementation - Support-with-Consent.

Tests cover:
    - Consent request creation
    - Consent approval/denial
    - Consent verification
    - Consent revocation
    - Expiry handling
    - Audit trail

Design Doc Reference:
    - Phase 1: "Transparency + legal artifacts aligned to CCEA"
    - Support-with-consent enforcement

CLOUD ZONE ONLY.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from ..consent import (
    SupportConsentService,
    ConsentRequest,
    ConsentRecord,
    ConsentType,
    ConsentStatus,
    ConsentScope,
    ConsentVerificationResult,
    get_consent_service,
    DEFAULT_CONSENT_DURATION_HOURS,
    MAX_CONSENT_DURATION_DAYS,
    CONSENT_REQUEST_EXPIRY_DAYS,
    VALID_SCOPES,
)


class TestConsentRequest:
    """Tests for ConsentRequest dataclass."""

    def test_create_request_with_defaults(self):
        """Test creating a consent request with default values."""
        request = ConsentRequest()
        assert request.id is not None
        assert request.consent_type == ConsentType.SUPPORT_DATA_ACCESS
        assert request.status == ConsentStatus.PENDING
        assert ConsentScope.FULL in request.scopes

    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        request = ConsentRequest(
            consent_type=ConsentType.SUPPORT_DATA_ACCESS,
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.TELEMETRY, ConsentScope.LOGS},
            purpose="Debug issue",
        )
        result = request.to_dict()

        assert result["consent_type"] == "support_data_access"
        assert result["support_agent_id"] == "agent_001"
        assert result["support_ticket_id"] == "TICKET-123"
        assert result["user_id"] == "user_123"
        assert result["workspace_id"] == "ws_456"
        assert set(result["scopes"]) == {"telemetry", "logs"}
        assert result["purpose"] == "Debug issue"
        assert result["status"] == "pending"

    def test_request_is_expired(self):
        """Test request expiry detection."""
        # Not expired
        request = ConsentRequest(
            expires_at=datetime.utcnow() + timedelta(days=1)
        )
        assert not request.is_expired

        # Expired
        request_expired = ConsentRequest(
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        assert request_expired.is_expired


class TestConsentRecord:
    """Tests for ConsentRecord dataclass."""

    def test_create_record_with_defaults(self):
        """Test creating a consent record with default values."""
        record = ConsentRecord()
        assert record.id is not None
        assert record.consent_type == ConsentType.SUPPORT_DATA_ACCESS
        assert record.status == ConsentStatus.ACTIVE
        assert record.access_count == 0

    def test_record_is_active(self):
        """Test consent active status check."""
        # Active consent
        record = ConsentRecord(
            status=ConsentStatus.ACTIVE,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        assert record.is_active

        # Revoked consent
        record_revoked = ConsentRecord(
            status=ConsentStatus.REVOKED,
            revoked_at=datetime.utcnow(),
        )
        assert not record_revoked.is_active

        # Expired consent
        record_expired = ConsentRecord(
            status=ConsentStatus.ACTIVE,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert not record_expired.is_active

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = ConsentRecord(
            consent_type=ConsentType.SUPPORT_DATA_ACCESS,
            user_id="user_123",
            workspace_id="ws_456",
            support_agent_id="agent_001",
            scopes={ConsentScope.TELEMETRY},
            access_count=5,
        )
        result = record.to_dict()

        assert result["user_id"] == "user_123"
        assert result["workspace_id"] == "ws_456"
        assert result["support_agent_id"] == "agent_001"
        assert result["scopes"] == ["telemetry"]
        assert result["access_count"] == 5


class TestSupportConsentService:
    """Tests for SupportConsentService."""

    @pytest.fixture
    def service(self):
        """Create a fresh consent service for each test."""
        return SupportConsentService()

    def test_create_request(self, service):
        """Test creating a consent request."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            organization_id="org_789",
            scopes={ConsentScope.TELEMETRY, ConsentScope.LOGS},
            purpose="Debug performance issue",
            duration_hours=48,
        )

        assert request.id is not None
        assert request.support_agent_id == "agent_001"
        assert request.support_ticket_id == "TICKET-123"
        assert request.user_id == "user_123"
        assert request.workspace_id == "ws_456"
        assert request.organization_id == "org_789"
        assert request.scopes == {ConsentScope.TELEMETRY, ConsentScope.LOGS}
        assert request.purpose == "Debug performance issue"
        assert request.requested_duration_hours == 48
        assert request.status == ConsentStatus.PENDING

    def test_create_request_with_invalid_scopes(self, service):
        """Test that invalid scopes raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            service.create_request(
                support_agent_id="agent_001",
                support_ticket_id="TICKET-123",
                user_id="user_123",
                workspace_id="ws_456",
                consent_type=ConsentType.SUPPORT_LOG_ACCESS,
                scopes={ConsentScope.TELEMETRY},  # Invalid for LOG_ACCESS
            )
        assert "Invalid scopes" in str(exc_info.value)

    def test_create_request_enforces_max_duration(self, service):
        """Test that maximum duration is enforced."""
        max_hours = MAX_CONSENT_DURATION_DAYS * 24
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            duration_hours=max_hours + 100,  # Exceeds max
        )
        assert request.requested_duration_hours == max_hours

    def test_get_request(self, service):
        """Test retrieving a request by ID."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )

        retrieved = service.get_request(request.id)
        assert retrieved is not None
        assert retrieved.id == request.id

        # Non-existent request
        assert service.get_request("non-existent") is None

    def test_get_pending_requests(self, service):
        """Test retrieving pending requests for a user."""
        # Create multiple requests
        service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-1",
            user_id="user_123",
            workspace_id="ws_1",
        )
        service.create_request(
            support_agent_id="agent_002",
            support_ticket_id="TICKET-2",
            user_id="user_123",
            workspace_id="ws_2",
        )
        service.create_request(
            support_agent_id="agent_003",
            support_ticket_id="TICKET-3",
            user_id="user_456",  # Different user
            workspace_id="ws_3",
        )

        pending = service.get_pending_requests("user_123")
        assert len(pending) == 2
        assert all(r.user_id == "user_123" for r in pending)

    def test_approve_request(self, service):
        """Test approving a consent request."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.TELEMETRY},
            purpose="Debug issue",
            duration_hours=48,
        )

        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        assert consent is not None
        assert consent.request_id == request.id
        assert consent.user_id == "user_123"
        assert consent.workspace_id == "ws_456"
        assert consent.support_agent_id == "agent_001"
        assert consent.granted_by_email == "user@example.com"
        assert consent.status == ConsentStatus.ACTIVE
        assert consent.is_active

        # Request status should be updated
        updated_request = service.get_request(request.id)
        assert updated_request.status == ConsentStatus.ACTIVE

    def test_approve_request_with_custom_duration(self, service):
        """Test approving with custom duration."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            duration_hours=72,
        )

        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
            duration_hours=24,  # Override to shorter duration
        )

        # Expiry should be ~24 hours from now
        expected_expiry = datetime.utcnow() + timedelta(hours=24)
        assert abs((consent.expires_at - expected_expiry).total_seconds()) < 5

    def test_approve_nonexistent_request(self, service):
        """Test approving a non-existent request."""
        consent = service.approve_request(
            request_id="non-existent",
            approved_by="user_123",
            approved_by_email="user@example.com",
        )
        assert consent is None

    def test_approve_already_approved_request(self, service):
        """Test approving an already-approved request."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )

        # First approval
        consent1 = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )
        assert consent1 is not None

        # Second approval should fail
        consent2 = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )
        assert consent2 is None

    def test_deny_request(self, service):
        """Test denying a consent request."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )

        result = service.deny_request(
            request_id=request.id,
            denied_by="user_123",
            reason="Not comfortable sharing data",
        )

        assert result is True
        updated_request = service.get_request(request.id)
        assert updated_request.status == ConsentStatus.DENIED

    def test_deny_nonexistent_request(self, service):
        """Test denying a non-existent request."""
        result = service.deny_request(
            request_id="non-existent",
            denied_by="user_123",
        )
        assert result is False

    def test_verify_consent_with_active_consent(self, service):
        """Test verifying consent when active consent exists."""
        # Create and approve request
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.TELEMETRY, ConsentScope.LOGS},
        )
        service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Verify consent for authorized scope
        result = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )

        assert result.has_consent is True
        assert result.consent_id is not None
        assert ConsentScope.TELEMETRY in result.scopes

    def test_verify_consent_with_full_scope(self, service):
        """Test that FULL scope covers all scopes."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.FULL},  # Full access
        )
        service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Should have access to any scope
        for scope in [ConsentScope.TELEMETRY, ConsentScope.LOGS, ConsentScope.CONFIG]:
            result = service.verify_consent(
                support_agent_id="agent_001",
                workspace_id="ws_456",
                required_scope=scope,
            )
            assert result.has_consent is True

    def test_verify_consent_without_consent(self, service):
        """Test verifying consent when no consent exists."""
        result = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )

        assert result.has_consent is False
        assert result.error is not None
        assert "No active consent found" in result.error

    def test_verify_consent_wrong_agent(self, service):
        """Test that consent is agent-specific."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Different agent should not have access
        result = service.verify_consent(
            support_agent_id="agent_002",  # Different agent
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )

        assert result.has_consent is False

    def test_verify_consent_wrong_workspace(self, service):
        """Test that consent is workspace-specific."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Different workspace should not have access
        result = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_789",  # Different workspace
            required_scope=ConsentScope.TELEMETRY,
        )

        assert result.has_consent is False

    def test_verify_consent_wrong_scope(self, service):
        """Test that consent is scope-specific."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            scopes={ConsentScope.LOGS},  # Only logs
        )
        service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Telemetry scope should not be authorized
        result = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,  # Not authorized
        )

        assert result.has_consent is False

    def test_record_access(self, service):
        """Test recording data access under consent."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Record multiple accesses
        assert service.record_access(consent.id) is True
        assert service.record_access(consent.id) is True
        assert service.record_access(consent.id) is True

        updated_consent = service.get_consent(consent.id)
        assert updated_consent.access_count == 3
        assert updated_consent.last_access_at is not None

    def test_record_access_nonexistent_consent(self, service):
        """Test recording access for non-existent consent."""
        result = service.record_access("non-existent")
        assert result is False

    def test_revoke_consent(self, service):
        """Test revoking consent."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Revoke consent
        result = service.revoke_consent(
            consent_id=consent.id,
            revoked_by="user_123",
            reason="No longer needed",
        )

        assert result is True
        updated_consent = service.get_consent(consent.id)
        assert updated_consent.status == ConsentStatus.REVOKED
        assert updated_consent.revoked_at is not None
        assert updated_consent.revoked_by == "user_123"
        assert updated_consent.revocation_reason == "No longer needed"
        assert not updated_consent.is_active

    def test_revoke_consent_blocks_access(self, service):
        """Test that revoking consent blocks further access verification."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Verify access before revocation
        result_before = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )
        assert result_before.has_consent is True

        # Revoke
        service.revoke_consent(consent.id, revoked_by="user_123")

        # Verify access after revocation
        result_after = service.verify_consent(
            support_agent_id="agent_001",
            workspace_id="ws_456",
            required_scope=ConsentScope.TELEMETRY,
        )
        assert result_after.has_consent is False

    def test_revoke_already_revoked(self, service):
        """Test revoking already revoked consent."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # First revocation
        assert service.revoke_consent(consent.id, revoked_by="user_123") is True

        # Second revocation should fail
        assert service.revoke_consent(consent.id, revoked_by="user_123") is False

    def test_get_active_consents(self, service):
        """Test retrieving active consents for a workspace."""
        # Create multiple consents
        for i in range(3):
            request = service.create_request(
                support_agent_id=f"agent_{i}",
                support_ticket_id=f"TICKET-{i}",
                user_id="user_123",
                workspace_id="ws_456",
            )
            service.approve_request(
                request_id=request.id,
                approved_by="user_123",
                approved_by_email="user@example.com",
            )

        # One for different workspace
        request_other = service.create_request(
            support_agent_id="agent_other",
            support_ticket_id="TICKET-other",
            user_id="user_123",
            workspace_id="ws_789",  # Different workspace
        )
        service.approve_request(
            request_id=request_other.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        active = service.get_active_consents("ws_456")
        assert len(active) == 3
        assert all(c.workspace_id == "ws_456" for c in active)

    def test_get_user_consents(self, service):
        """Test retrieving all consents for a user."""
        # Create and approve
        request1 = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-1",
            user_id="user_123",
            workspace_id="ws_1",
        )
        service.approve_request(
            request_id=request1.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Create and approve, then revoke
        request2 = service.create_request(
            support_agent_id="agent_002",
            support_ticket_id="TICKET-2",
            user_id="user_123",
            workspace_id="ws_2",
        )
        consent2 = service.approve_request(
            request_id=request2.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )
        service.revoke_consent(consent2.id, revoked_by="user_123")

        # Different user
        request3 = service.create_request(
            support_agent_id="agent_003",
            support_ticket_id="TICKET-3",
            user_id="user_456",
            workspace_id="ws_3",
        )
        service.approve_request(
            request_id=request3.id,
            approved_by="user_456",
            approved_by_email="other@example.com",
        )

        consents = service.get_user_consents("user_123")
        assert len(consents) == 2  # Both active and revoked
        assert all(c.user_id == "user_123" for c in consents)

    def test_audit_log(self, service):
        """Test audit log is created for consent actions."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )
        service.record_access(consent.id)
        service.revoke_consent(consent.id, revoked_by="user_123")

        audit_log = service.get_audit_log()
        actions = [entry["action"] for entry in audit_log]

        assert "consent_request_created" in actions
        assert "consent_granted" in actions
        assert "consent_access_recorded" in actions
        assert "consent_revoked" in actions

    def test_audit_log_filter_by_consent_id(self, service):
        """Test filtering audit log by consent ID."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        # Create another consent
        request2 = service.create_request(
            support_agent_id="agent_002",
            support_ticket_id="TICKET-456",
            user_id="user_123",
            workspace_id="ws_456",
        )
        service.approve_request(
            request_id=request2.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        filtered = service.get_audit_log(consent_id=consent.id)
        assert all(entry.get("consent_id") == consent.id for entry in filtered)

    def test_cleanup_expired(self, service):
        """Test cleanup of expired consents."""
        # Create and approve with very short expiry
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )
        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
            duration_hours=0,  # Will be expired immediately
        )

        # Manually set expiry to past
        consent.expires_at = datetime.utcnow() - timedelta(hours=1)

        # Run cleanup
        count = service.cleanup_expired()
        assert count >= 1

        updated = service.get_consent(consent.id)
        assert updated.status == ConsentStatus.EXPIRED


class TestConsentTypes:
    """Tests for different consent types and their valid scopes."""

    @pytest.fixture
    def service(self):
        return SupportConsentService()

    def test_support_data_access_allows_all_scopes(self, service):
        """Test SUPPORT_DATA_ACCESS allows all standard scopes."""
        for scope in VALID_SCOPES[ConsentType.SUPPORT_DATA_ACCESS]:
            request = service.create_request(
                support_agent_id="agent_001",
                support_ticket_id="TICKET-123",
                user_id="user_123",
                workspace_id="ws_456",
                consent_type=ConsentType.SUPPORT_DATA_ACCESS,
                scopes={scope},
            )
            assert request is not None

    def test_support_log_access_only_allows_logs(self, service):
        """Test SUPPORT_LOG_ACCESS only allows logs scope."""
        # Valid
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            consent_type=ConsentType.SUPPORT_LOG_ACCESS,
            scopes={ConsentScope.LOGS},
        )
        assert request is not None

        # Invalid
        with pytest.raises(ValueError):
            service.create_request(
                support_agent_id="agent_001",
                support_ticket_id="TICKET-123",
                user_id="user_123",
                workspace_id="ws_456",
                consent_type=ConsentType.SUPPORT_LOG_ACCESS,
                scopes={ConsentScope.TELEMETRY},
            )

    def test_raw_telemetry_access_only_allows_telemetry(self, service):
        """Test RAW_TELEMETRY_ACCESS only allows telemetry scope."""
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
            consent_type=ConsentType.RAW_TELEMETRY_ACCESS,
            scopes={ConsentScope.TELEMETRY},
        )
        assert request is not None

        with pytest.raises(ValueError):
            service.create_request(
                support_agent_id="agent_001",
                support_ticket_id="TICKET-123",
                user_id="user_123",
                workspace_id="ws_456",
                consent_type=ConsentType.RAW_TELEMETRY_ACCESS,
                scopes={ConsentScope.LOGS},
            )


class TestGetConsentService:
    """Tests for the module-level get_consent_service function."""

    def test_get_consent_service_returns_singleton(self):
        """Test that get_consent_service returns the same instance."""
        service1 = get_consent_service()
        service2 = get_consent_service()
        assert service1 is service2

    def test_get_consent_service_creates_service(self):
        """Test that get_consent_service creates a service instance."""
        service = get_consent_service()
        assert isinstance(service, SupportConsentService)


class TestConcurrency:
    """Tests for thread safety."""

    def test_concurrent_consent_operations(self):
        """Test that consent operations are thread-safe."""
        import threading

        service = SupportConsentService()
        errors = []

        def create_and_approve():
            try:
                request = service.create_request(
                    support_agent_id=f"agent_{threading.current_thread().name}",
                    support_ticket_id=f"TICKET-{threading.current_thread().name}",
                    user_id="user_123",
                    workspace_id="ws_456",
                )
                service.approve_request(
                    request_id=request.id,
                    approved_by="user_123",
                    approved_by_email="user@example.com",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_approve, name=f"t{i}") for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestExpiredRequestHandling:
    """Tests for expired request handling."""

    def test_approve_expired_request(self):
        """Test that expired requests cannot be approved."""
        service = SupportConsentService()
        request = service.create_request(
            support_agent_id="agent_001",
            support_ticket_id="TICKET-123",
            user_id="user_123",
            workspace_id="ws_456",
        )

        # Manually expire the request
        request.expires_at = datetime.utcnow() - timedelta(days=1)

        consent = service.approve_request(
            request_id=request.id,
            approved_by="user_123",
            approved_by_email="user@example.com",
        )

        assert consent is None
        assert service.get_request(request.id).status == ConsentStatus.EXPIRED
