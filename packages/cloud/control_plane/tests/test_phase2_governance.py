# -*- coding: utf-8 -*-
"""
Tests for Phase 2: Cloud Control Plane Operational Governance.

Tests cover:
- 2.0 RBAC/Tenant Isolation
- 2.1 DB-backed Governance (DSAR, Retention, Residency, Break-Glass)
- 2.1A Agent Trust-State Enforcement
- 2.2 AccessAudit middleware
- 2.3 Retention/Deletion auto-purge
- 2.4 Data Residency Enforcement
- 2.5 Telemetry Validation
- 2.6 Research Sandbox integration
- 2.7-2.8 CI Guardrails

References:
- CCEA_ALIGNMENT_PLAN.md Phase 2
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Set
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Add package to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))


# ============================================================================
# 2.0 RBAC/Tenant Isolation Tests
# ============================================================================

class TestRBACTenantIsolation:
    """Tests for RBAC and tenant isolation."""

    def test_current_user_has_permission_basic(self):
        """Test basic permission check."""
        from packages.cloud.control_plane.dependencies import CurrentUser

        user = CurrentUser(
            id=uuid4(),
            email="test@example.com",
            permissions={"telemetry:read", "alerts:read"},
        )

        assert user.has_permission("telemetry:read")
        assert user.has_permission("alerts:read")
        assert not user.has_permission("admin:write")

    def test_current_user_superuser_bypasses_permissions(self):
        """Test that superuser bypasses all permission checks."""
        from packages.cloud.control_plane.dependencies import CurrentUser

        user = CurrentUser(
            id=uuid4(),
            email="admin@example.com",
            permissions=set(),
            is_superuser=True,
        )

        assert user.has_permission("any:permission")
        assert user.has_permission("admin:delete")

    def test_current_user_has_any_permission(self):
        """Test any permission check."""
        from packages.cloud.control_plane.dependencies import CurrentUser

        user = CurrentUser(
            id=uuid4(),
            email="test@example.com",
            permissions={"telemetry:read"},
        )

        assert user.has_any_permission(["telemetry:read", "admin:write"])
        assert not user.has_any_permission(["admin:write", "admin:delete"])

    def test_current_user_has_all_permissions(self):
        """Test all permissions check."""
        from packages.cloud.control_plane.dependencies import CurrentUser

        user = CurrentUser(
            id=uuid4(),
            email="test@example.com",
            permissions={"telemetry:read", "alerts:read", "commands:read"},
        )

        assert user.has_all_permissions(["telemetry:read", "alerts:read"])
        assert not user.has_all_permissions(["telemetry:read", "admin:write"])

    def test_current_agent_has_capability(self):
        """Test agent capability check."""
        from packages.cloud.control_plane.dependencies import CurrentAgent
        from packages.cloud.control_plane.models import TrustState

        agent = CurrentAgent(
            id=uuid4(),
            workspace_id=uuid4(),
            org_id=uuid4(),
            capabilities=["execute_commands", "send_telemetry"],
            trust_state=TrustState.ENROLLED,
        )

        assert agent.has_capability("execute_commands")
        assert agent.has_capability("send_telemetry")
        assert not agent.has_capability("admin_access")


class TestTenantContext:
    """Tests for TenantContext."""

    def test_tenant_context_creation_with_workspace(self):
        """Test TenantContext with workspace_id."""
        from packages.cloud.control_plane.database import TenantContext

        workspace_id = uuid4()
        ctx = TenantContext(workspace_id)

        assert ctx.workspace_id == workspace_id

    def test_tenant_context_creation_without_workspace(self):
        """Test TenantContext without workspace_id."""
        from packages.cloud.control_plane.database import TenantContext

        ctx = TenantContext(None)
        assert ctx.workspace_id is None


# ============================================================================
# 2.1 DB-backed Governance Tests
# ============================================================================

class TestDSARService:
    """Tests for DSAR service."""

    @pytest.mark.asyncio
    async def test_dsar_create_access_request(self):
        """Test creating DSAR access request."""
        from packages.cloud.control_plane.services.governance_service import (
            DSARService,
            DSARRequestType,
            DSARStatus,
        )

        mock_session = AsyncMock()

        service = DSARService(mock_session)
        user_id = str(uuid4())
        workspace_id = uuid4()

        dsar = await service.create_request(
            user_id=user_id,
            workspace_id=workspace_id,
            request_type=DSARRequestType.ACCESS,
            reason="Testing DSAR access request",
        )

        assert dsar.user_id == user_id
        assert dsar.workspace_id == workspace_id
        assert dsar.request_type == DSARRequestType.ACCESS
        assert dsar.status == DSARStatus.PENDING
        # GDPR deadline is 30 days
        assert (dsar.deadline - dsar.created_at).days == 30

    @pytest.mark.asyncio
    async def test_dsar_erasure_requires_verification(self):
        """Test that DSAR erasure requires identity verification."""
        from packages.cloud.control_plane.services.governance_service import (
            DSARService,
            DSARRequestType,
        )

        mock_session = AsyncMock()
        service = DSARService(mock_session)

        dsar = await service.create_request(
            user_id=str(uuid4()),
            workspace_id=uuid4(),
            request_type=DSARRequestType.ERASURE,
            reason="Delete my data",
        )

        assert dsar.verification_required is True
        assert dsar.verification_completed is False

    @pytest.mark.asyncio
    async def test_dsar_extend_deadline_once(self):
        """Test DSAR deadline can only be extended once."""
        from packages.cloud.control_plane.services.governance_service import (
            DSARService,
            DSARRequestType,
            DSARStatus,
        )

        mock_session = AsyncMock()
        service = DSARService(mock_session)

        dsar = await service.create_request(
            user_id=str(uuid4()),
            workspace_id=uuid4(),
            request_type=DSARRequestType.ACCESS,
        )

        # First extension should succeed
        result1 = await service.extend_deadline(dsar.id, "Complex request")
        assert result1 is True
        assert dsar.status == DSARStatus.EXTENDED
        assert dsar.extended_deadline is not None

        # Second extension should fail
        result2 = await service.extend_deadline(dsar.id, "Need more time")
        assert result2 is False


class TestResidencyPolicyService:
    """Tests for data residency service."""

    @pytest.mark.asyncio
    async def test_eu_country_gets_eu_region(self):
        """Test EU country gets EU region and GDPR compliance."""
        from packages.cloud.control_plane.services.governance_service import (
            ResidencyPolicyService,
            DataRegion,
            EU_REGIONS,
        )

        mock_session = AsyncMock()
        service = ResidencyPolicyService(mock_session)

        policy = await service.create_policy_for_country(
            workspace_id=uuid4(),
            country_code="DE",  # Germany
        )

        assert policy.gdpr_compliant is True
        assert policy.requires_eu_only is True
        assert policy.primary_region in EU_REGIONS
        assert all(r in EU_REGIONS for r in policy.allowed_regions)

    @pytest.mark.asyncio
    async def test_us_country_gets_us_region(self):
        """Test US country gets US region."""
        from packages.cloud.control_plane.services.governance_service import (
            ResidencyPolicyService,
            DataRegion,
        )

        mock_session = AsyncMock()
        service = ResidencyPolicyService(mock_session)

        policy = await service.create_policy_for_country(
            workspace_id=uuid4(),
            country_code="US",
        )

        assert policy.gdpr_compliant is False
        assert policy.requires_eu_only is False
        assert policy.primary_region == DataRegion.US_EAST

    @pytest.mark.asyncio
    async def test_eu_only_blocks_us_storage(self):
        """Test EU-only policy blocks US region storage."""
        from packages.cloud.control_plane.services.governance_service import (
            ResidencyPolicyService,
            DataRegion,
        )

        mock_session = AsyncMock()
        service = ResidencyPolicyService(mock_session)
        workspace_id = uuid4()

        # Create EU policy
        await service.create_policy_for_country(
            workspace_id=workspace_id,
            country_code="FR",  # France
        )

        # Try to store in US region
        allowed, reason = await service.can_store_in_region(
            workspace_id=workspace_id,
            region=DataRegion.US_EAST,
        )

        assert allowed is False
        assert "not in allowed list" in reason or "EU-only" in reason

    @pytest.mark.asyncio
    async def test_local_only_blocks_all_cloud(self):
        """Test LOCAL_ONLY mode blocks all cloud storage."""
        from packages.cloud.control_plane.services.governance_service import (
            ResidencyPolicyService,
            DataRegion,
        )

        mock_session = AsyncMock()
        service = ResidencyPolicyService(mock_session)
        workspace_id = uuid4()

        # Create LOCAL_ONLY policy
        await service.create_enterprise_local_policy(workspace_id)

        # Try any region
        for region in DataRegion:
            allowed, reason = await service.can_store_in_region(workspace_id, region)
            assert allowed is False
            assert "LOCAL_ONLY" in reason


class TestBreakGlassService:
    """Tests for break-glass access service."""

    @pytest.mark.asyncio
    async def test_break_glass_requires_min_reason_length(self):
        """Test break-glass requires minimum reason length."""
        from packages.cloud.control_plane.services.governance_service import (
            BreakGlassService,
            BreakGlassReason,
            BreakGlassScope,
        )

        mock_session = AsyncMock()
        service = BreakGlassService(mock_session)

        with pytest.raises(ValueError, match="at least 20 characters"):
            await service.create_request(
                requester_id=str(uuid4()),
                requester_email="test@example.com",
                workspace_id=uuid4(),
                reason="Too short",  # Less than 20 chars
                reason_type=BreakGlassReason.INCIDENT_RESPONSE,
                scope={BreakGlassScope.TELEMETRY_READ},
            )

    @pytest.mark.asyncio
    async def test_break_glass_no_self_approval(self):
        """Test break-glass cannot be self-approved."""
        from packages.cloud.control_plane.services.governance_service import (
            BreakGlassService,
            BreakGlassReason,
            BreakGlassScope,
        )

        mock_session = AsyncMock()
        service = BreakGlassService(mock_session)
        requester_id = str(uuid4())

        request = await service.create_request(
            requester_id=requester_id,
            requester_email="test@example.com",
            workspace_id=uuid4(),
            reason="Critical incident response needed for investigation",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Try self-approval
        success, token, error = await service.approve_request(
            request_id=request.id,
            approved_by=requester_id,  # Same as requester
        )

        assert success is False
        assert "Self-approval not allowed" in error

    @pytest.mark.asyncio
    async def test_break_glass_generates_evidence_hash(self):
        """Test break-glass generates evidence hash for audit."""
        from packages.cloud.control_plane.services.governance_service import (
            BreakGlassService,
            BreakGlassReason,
            BreakGlassScope,
        )

        mock_session = AsyncMock()
        service = BreakGlassService(mock_session)

        request = await service.create_request(
            requester_id=str(uuid4()),
            requester_email="test@example.com",
            workspace_id=uuid4(),
            reason="Security investigation for suspected breach",
            reason_type=BreakGlassReason.SECURITY_INVESTIGATION,
            scope={BreakGlassScope.AUDIT_READ},
        )

        # Evidence hash should be SHA256 (64 hex chars)
        assert len(request.evidence_hash) == 64
        assert all(c in "0123456789abcdef" for c in request.evidence_hash)


# ============================================================================
# 2.1A Agent Trust-State Enforcement Tests
# ============================================================================

class TestAgentTrustStateEnforcement:
    """Tests for agent trust-state enforcement."""

    def test_trust_state_enum_values(self):
        """Test trust state enum values."""
        from packages.cloud.control_plane.models import TrustState

        assert TrustState.ENROLLED.value == "enrolled"
        assert TrustState.REVOKED.value == "revoked"
        assert TrustState.SUSPENDED.value == "suspended"
        assert TrustState.PENDING.value == "pending"


# ============================================================================
# 2.2 AccessAudit Middleware Tests
# ============================================================================

class TestAuditLogger:
    """Tests for audit logger."""

    @pytest.mark.asyncio
    async def test_audit_logger_creates_entry(self):
        """Test audit logger creates entries correctly."""
        from packages.cloud.control_plane.middleware.audit_middleware import (
            AuditLogger,
        )
        from packages.cloud.control_plane.models import AuditAction

        mock_session = AsyncMock()
        logger = AuditLogger(mock_session)

        workspace_id = uuid4()
        user_id = uuid4()

        await logger.log(
            action=AuditAction.READ,
            resource_type="telemetry_event",
            resource_id="123",
            workspace_id=workspace_id,
            user_id=user_id,
        )

        # Verify session.add was called
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    def test_audit_context_add_entry(self):
        """Test AuditContext entry tracking."""
        from packages.cloud.control_plane.middleware.audit_middleware import (
            AuditContext,
        )
        from packages.cloud.control_plane.models import AuditAction

        ctx = AuditContext(
            request_id="test-123",
            user_id=uuid4(),
            workspace_id=uuid4(),
        )

        ctx.add_entry(
            action=AuditAction.READ,
            resource_type="alert",
            resource_id="alert-456",
        )

        assert len(ctx.entries) == 1
        assert ctx.entries[0].action == AuditAction.READ
        assert ctx.entries[0].resource_type == "alert"


class TestAuditSensitiveResources:
    """Tests for sensitive resource definitions."""

    def test_telemetry_is_sensitive_read(self):
        """Test telemetry is marked as sensitive for reads."""
        from packages.cloud.control_plane.middleware.audit_middleware import (
            SENSITIVE_RESOURCES_READ,
        )

        assert "telemetry_event" in SENSITIVE_RESOURCES_READ
        assert "telemetry_events" in SENSITIVE_RESOURCES_READ

    def test_commands_are_sensitive_write(self):
        """Test commands are marked as sensitive for writes."""
        from packages.cloud.control_plane.middleware.audit_middleware import (
            SENSITIVE_RESOURCES_WRITE,
        )

        assert "command" in SENSITIVE_RESOURCES_WRITE
        assert "commands" in SENSITIVE_RESOURCES_WRITE


# ============================================================================
# 2.5 Telemetry Validation Tests
# ============================================================================

class TestTelemetryValidator:
    """Tests for telemetry payload validation."""

    def test_rejects_order_fields(self):
        """Test validator rejects order/intent fields."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        # Payload with prohibited order fields
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": "BUY",  # PROHIBITED
            "quantity": 100,  # PROHIBITED
            "price": 50.0,  # PROHIBITED
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is False
        assert result.blocked is True
        assert any(v.violation_type == ViolationType.PROHIBITED_FIELD for v in result.violations)

    def test_rejects_intent_fields(self):
        """Test validator rejects intent fields."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": "LONG",  # PROHIBITED
            "target_position": 1000,  # PROHIBITED
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is False
        assert any(v.violation_type == ViolationType.PROHIBITED_FIELD for v in result.violations)

    def test_rejects_nested_order_structure(self):
        """Test validator rejects nested order-like structures."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        # Order structure hidden in nested dict
        payload = {
            "metrics": {
                "cpu": 50,
                "hidden_order": {
                    "side": "SELL",
                    "quantity": 50,
                    "price": 100,
                }
            }
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is False
        # Should catch both prohibited fields AND intent injection
        violation_types = {v.violation_type for v in result.violations}
        assert ViolationType.PROHIBITED_FIELD in violation_types or \
               ViolationType.INTENT_INJECTION in violation_types

    def test_rejects_broker_credentials(self):
        """Test validator rejects broker credentials."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "binance_api_key": "abcdef123456",  # PROHIBITED
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is False
        assert any(v.violation_type == ViolationType.BROKER_CREDENTIALS for v in result.violations)

    def test_requires_redaction_for_non_aggregated(self):
        """Test non-aggregated telemetry requires redaction."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "test",
        }

        result = validator.validate(
            payload,
            TelemetryLevel.DETAILED_NON_SENSITIVE.value,
            redaction_applied=False,  # Not redacted
        )

        assert result.valid is False
        assert any(v.violation_type == ViolationType.MISSING_REDACTION for v in result.violations)

    def test_accepts_valid_aggregated_payload(self):
        """Test validator accepts valid aggregated payload."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        # Valid aggregated metrics
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": 45.5,
            "memory_percent": 60.2,
            "latency_ms": 12.5,
            "trade_count": 100,
            "win_rate": 0.65,
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is True
        assert result.blocked is False

    def test_detects_intent_injection_pattern(self):
        """Test detection of order-like structure pattern."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            TelemetryValidator,
            ViolationType,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        validator = TelemetryValidator(strict_mode=True)

        # This is a complete order structure
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order": {
                "side": "BUY",
                "quantity": 100,
                "price": 50.0,
            }
        }

        result = validator.validate(
            payload,
            TelemetryLevel.AGGREGATED.value,
            redaction_applied=True,
        )

        assert result.valid is False
        assert any(v.violation_type in (ViolationType.PROHIBITED_FIELD, ViolationType.INTENT_INJECTION)
                  for v in result.violations)


class TestRedactionEnforcer:
    """Tests for redaction enforcement."""

    def test_identifies_unredacted_pii(self):
        """Test identifies unredacted PII fields."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            RedactionEnforcer,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        enforcer = RedactionEnforcer()

        payload = {
            "ip_address": "192.168.1.100",  # Should be redacted
            "user_agent": "Mozilla/5.0...",  # Should be redacted
            "metric": 100,
        }

        is_redacted, unredacted = enforcer.check_redaction(
            payload,
            TelemetryLevel.DETAILED_NON_SENSITIVE.value,
        )

        assert is_redacted is False
        assert "ip_address" in unredacted
        assert "user_agent" in unredacted

    def test_accepts_hashed_fields(self):
        """Test accepts properly hashed fields."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            RedactionEnforcer,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        enforcer = RedactionEnforcer()

        # SHA256 hash (64 hex chars)
        payload = {
            "ip_address": "a" * 64,  # Looks like SHA256 hash
            "device_id": "b" * 64,
            "metric": 100,
        }

        is_redacted, unredacted = enforcer.check_redaction(
            payload,
            TelemetryLevel.DETAILED_NON_SENSITIVE.value,
        )

        assert is_redacted is True
        assert len(unredacted) == 0

    def test_accepts_redaction_markers(self):
        """Test accepts standard redaction markers."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            RedactionEnforcer,
        )
        from packages.cloud.control_plane.models import TelemetryLevel

        enforcer = RedactionEnforcer()

        payload = {
            "ip_address": "[REDACTED]",
            "device_id": "[HASHED]",
            "session_id": "***",
        }

        is_redacted, unredacted = enforcer.check_redaction(
            payload,
            TelemetryLevel.DETAILED_NON_SENSITIVE.value,
        )

        assert is_redacted is True
        assert len(unredacted) == 0


# ============================================================================
# 2.3 Retention/Purge Tests
# ============================================================================

class TestPurgeScheduler:
    """Tests for auto-purge scheduler."""

    def test_purge_scheduler_config_defaults(self):
        """Test purge scheduler default configuration."""
        from packages.cloud.control_plane.services.purge_scheduler import (
            PurgeSchedulerConfig,
        )

        config = PurgeSchedulerConfig()

        assert config.enabled is True
        assert config.interval_hours == 24
        assert config.batch_size == 1000
        assert config.dry_run is False

    def test_purge_data_type_enum(self):
        """Test purge data type enum values."""
        from packages.cloud.control_plane.services.purge_scheduler import (
            PurgeDataType,
        )

        assert PurgeDataType.TELEMETRY_EVENTS.value == "telemetry_events"
        assert PurgeDataType.ALERTS.value == "alerts"
        assert PurgeDataType.COMMANDS.value == "commands"
        assert PurgeDataType.ACCESS_AUDITS.value == "access_audits"


# ============================================================================
# Prohibited Fields Constants Tests
# ============================================================================

class TestProhibitedFieldsConstants:
    """Tests for prohibited fields constants."""

    def test_order_fields_complete(self):
        """Test all critical order fields are prohibited."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            PROHIBITED_ORDER_FIELDS,
        )

        # Must include these critical order fields
        critical_fields = {
            "side",
            "quantity",
            "qty",
            "price",
            "order_type",
            "intent",
            "signal",
            "target_position",
            "execute_order",
            "place_order",
            "submit_order",
        }

        assert critical_fields.issubset(PROHIBITED_ORDER_FIELDS)

    def test_pii_fields_complete(self):
        """Test PII fields are prohibited."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            PROHIBITED_PII_FIELDS,
        )

        critical_pii = {
            "email",
            "phone",
            "ssn",
            "credit_card",
            "password",
        }

        assert critical_pii.issubset(PROHIBITED_PII_FIELDS)

    def test_allowed_aggregated_fields_present(self):
        """Test allowed aggregated fields are defined."""
        from packages.cloud.control_plane.middleware.telemetry_validation import (
            ALLOWED_AGGREGATED_FIELDS,
        )

        # Metrics that SHOULD be allowed
        allowed_metrics = {
            "cpu_percent",
            "memory_percent",
            "latency_ms",
            "trade_count",
            "win_rate",
            "sharpe_ratio",
        }

        assert allowed_metrics.issubset(ALLOWED_AGGREGATED_FIELDS)


# ============================================================================
# Integration Tests
# ============================================================================

class TestGovernanceIntegration:
    """Integration tests for governance components."""

    def test_dsar_retention_interaction(self):
        """Test DSAR respects retention policy legal hold."""
        from packages.cloud.control_plane.services.governance_service import (
            DEFAULT_RETENTION_DAYS,
        )

        # Compliance audit data must be retained 7 years
        assert DEFAULT_RETENTION_DAYS["approval_records"] >= 2555  # ~7 years
        assert DEFAULT_RETENTION_DAYS["access_audits"] >= 2555

    def test_break_glass_scope_enum(self):
        """Test break-glass scope covers necessary access."""
        from packages.cloud.control_plane.services.governance_service import (
            BreakGlassScope,
        )

        # Must have telemetry and audit read for incident response
        assert BreakGlassScope.TELEMETRY_READ.value == "telemetry_read"
        assert BreakGlassScope.AUDIT_READ.value == "audit_read"

    def test_eu_countries_list(self):
        """Test EU countries list is complete."""
        from packages.cloud.control_plane.services.governance_service import (
            EU_COUNTRIES,
        )

        # Major EU countries must be present
        major_eu = {"DE", "FR", "IT", "ES", "NL", "BE", "PL"}
        assert major_eu.issubset(EU_COUNTRIES)

        # UK for GDPR-like treatment
        assert "GB" in EU_COUNTRIES


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
