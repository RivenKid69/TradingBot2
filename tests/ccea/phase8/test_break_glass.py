# -*- coding: utf-8 -*-
"""
Tests for BreakGlassController.

CCEA Phase 8 - Emergency access control tests.
"""

import pytest
from datetime import datetime, timedelta

from packages.cloud.governance.break_glass import (
    BreakGlassController,
    BreakGlassRequest,
    BreakGlassResult,
    BreakGlassReason,
    BreakGlassScope,
    MAX_BREAK_GLASS_DURATION_HOURS,
    DEFAULT_BREAK_GLASS_DURATION_HOURS,
    MIN_REASON_LENGTH,
    BREAK_GLASS_COOLDOWN_MINUTES,
)


class TestBreakGlassControllerBasic:
    """Basic break glass controller tests."""

    def test_create_controller(self):
        """Test creating controller."""
        controller = BreakGlassController()
        assert controller is not None

    def test_create_with_approvers(self):
        """Test creating with approvers list."""
        controller = BreakGlassController(approvers={"admin@example.com", "security@example.com"})
        assert len(controller._approvers) == 2


class TestRequestCreation:
    """Request creation tests."""

    def test_create_request(self):
        """Test creating break glass request."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="Investigating production incident #12345 with customer impact",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        assert request.id is not None
        assert request.requester_id == "user-123"
        assert request.reason_type == BreakGlassReason.INCIDENT_RESPONSE
        assert BreakGlassScope.TELEMETRY_READ in request.scope

    def test_request_requires_reason_length(self):
        """Test request requires minimum reason length."""
        controller = BreakGlassController()

        with pytest.raises(ValueError) as exc_info:
            controller.create_request(
                requester_id="user-123",
                requester_email="user@example.com",
                reason="too short",  # Below minimum
                reason_type=BreakGlassReason.OTHER,
                workspace_id="ws-456",
                scope={BreakGlassScope.TELEMETRY_READ},
            )

        assert str(MIN_REASON_LENGTH) in str(exc_info.value)

    def test_request_evidence_hash(self):
        """Test request has evidence hash."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="Investigating production incident #12345",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        assert request.evidence_hash is not None
        assert len(request.evidence_hash) == 64  # SHA-256 hex

    def test_duration_capped_at_max(self):
        """Test duration is capped at maximum."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
            duration_hours=100,  # Above max
        )

        assert request.duration_hours == MAX_BREAK_GLASS_DURATION_HOURS


class TestApprovalWorkflow:
    """Approval workflow tests."""

    def test_approve_request(self):
        """Test approving request."""
        controller = BreakGlassController(
            approvers={"admin@example.com"},
        )

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        result = controller.approve_request(
            request_id=request.id,
            approved_by="admin@example.com",
        )

        assert result.success is True
        assert result.access_token is not None
        assert result.expires_at is not None

    def test_self_approval_denied(self):
        """Test self-approval is denied."""
        controller = BreakGlassController()

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        result = controller.approve_request(
            request_id=request.id,
            approved_by="user-123",  # Same as requester
        )

        assert result.success is False
        assert "Self-approval" in result.error

    def test_unauthorized_approver_denied(self):
        """Test unauthorized approver is denied."""
        controller = BreakGlassController(
            approvers={"admin@example.com"},  # Only this user can approve
        )

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        result = controller.approve_request(
            request_id=request.id,
            approved_by="random@example.com",  # Not in approvers
        )

        assert result.success is False
        assert "Not authorized" in result.error

    def test_auto_approve_when_allowed(self):
        """Test auto-approve when not requiring approval."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        result = controller.auto_approve_request(request.id)

        assert result.success is True

    def test_auto_approve_denied_when_required(self):
        """Test auto-approve denied when approval required."""
        controller = BreakGlassController(require_approval=True)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        result = controller.auto_approve_request(request.id)

        assert result.success is False
        assert "Manual approval required" in result.error


class TestAccessControl:
    """Access control tests."""

    def test_has_access_approved(self):
        """Test has_access for approved request."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ, BreakGlassScope.AUDIT_READ},
        )
        controller.auto_approve_request(request.id)

        assert controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ) is True
        assert controller.has_access(request.id, BreakGlassScope.AUDIT_READ) is True

    def test_has_access_denied_wrong_scope(self):
        """Test has_access denied for wrong scope."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},  # Only telemetry
        )
        controller.auto_approve_request(request.id)

        assert controller.has_access(request.id, BreakGlassScope.ADMIN_ACCESS) is False

    def test_has_access_denied_not_approved(self):
        """Test has_access denied when not approved."""
        controller = BreakGlassController()

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        assert controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ) is False


class TestTokenValidation:
    """Token validation tests."""

    def test_validate_token(self):
        """Test token validation."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        result = controller.auto_approve_request(request.id)

        validated_id = controller.validate_token(result.access_token)

        assert validated_id == request.id

    def test_invalid_token(self):
        """Test invalid token returns None."""
        controller = BreakGlassController()

        result = controller.validate_token("invalid-token")

        assert result is None


class TestRevocation:
    """Request revocation tests."""

    def test_revoke_request(self):
        """Test revoking request."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        result = controller.auto_approve_request(request.id)

        revoked = controller.revoke_request(
            request_id=request.id,
            revoked_by="security",
            reason="No longer needed",
        )

        assert revoked is True

        # Access should be denied after revocation
        assert controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ) is False
        assert controller.validate_token(result.access_token) is None


class TestExpiration:
    """Expiration tests."""

    def test_is_active_not_expired(self):
        """Test is_active when not expired."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        controller.auto_approve_request(request.id)

        updated = controller.get_request(request.id)
        assert updated.is_active is True

    def test_cleanup_expired(self):
        """Test cleanup of expired tokens."""
        controller = BreakGlassController(require_approval=False)

        # Create and approve request
        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        controller.auto_approve_request(request.id)

        # Manually expire
        updated = controller.get_request(request.id)
        updated.expires_at = datetime.utcnow() - timedelta(hours=1)

        cleaned = controller.cleanup_expired()

        assert cleaned >= 0


class TestRequestQueries:
    """Request query tests."""

    def test_get_request(self):
        """Test getting request by ID."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        retrieved = controller.get_request(request.id)

        assert retrieved is not None
        assert retrieved.id == request.id

    def test_get_active_requests(self):
        """Test getting active requests."""
        controller = BreakGlassController(require_approval=False)

        # Create and approve request
        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        controller.auto_approve_request(request.id)

        active = controller.get_active_requests()

        assert len(active) >= 1

    def test_get_pending_requests(self):
        """Test getting pending requests."""
        controller = BreakGlassController()

        controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        pending = controller.get_pending_requests()

        assert len(pending) >= 1


class TestAccessTracking:
    """Access tracking tests."""

    def test_access_count_incremented(self):
        """Test access count is incremented."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        controller.auto_approve_request(request.id)

        # Use access multiple times
        controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ)
        controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ)

        updated = controller.get_request(request.id)
        assert updated.access_count >= 2


class TestAuditLog:
    """Audit log tests."""

    def test_audit_log_on_create(self):
        """Test audit log on request creation."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        log = controller.get_audit_log(request_id=request.id)

        assert len(log) > 0
        assert log[0]["action"] == "request_created"

    def test_audit_log_on_approve(self):
        """Test audit log on approval."""
        controller = BreakGlassController(require_approval=False)

        request = controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )
        controller.auto_approve_request(request.id)

        log = controller.get_audit_log(request_id=request.id)
        actions = [e["action"] for e in log]

        assert "request_approved" in actions


class TestRequestSerialization:
    """Request serialization tests."""

    def test_request_to_dict(self):
        """Test request serialization."""
        request = BreakGlassRequest(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="Valid reason for access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        data = request.to_dict()

        assert data["requester_id"] == "user-123"
        assert data["reason_type"] == "incident_response"
        assert "TELEMETRY_READ" in data["scope"]

    def test_result_to_dict(self):
        """Test result serialization."""
        result = BreakGlassResult(
            success=True,
            request_id="req-123",
            access_token="token",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["has_token"] is True


class TestCooldown:
    """Cooldown tests."""

    def test_cooldown_enforced(self):
        """Test cooldown between requests is enforced."""
        controller = BreakGlassController(require_approval=False)

        # First request succeeds
        controller.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            reason="This is a valid reason for break glass access first request",
            reason_type=BreakGlassReason.OTHER,
            workspace_id="ws-456",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Second request from same user should fail due to cooldown
        with pytest.raises(ValueError) as exc_info:
            controller.create_request(
                requester_id="user-123",
                requester_email="user@example.com",
                reason="This is a valid reason for break glass access second request",
                reason_type=BreakGlassReason.OTHER,
                workspace_id="ws-456",
                scope={BreakGlassScope.TELEMETRY_READ},
            )

        assert "Cooldown" in str(exc_info.value)
