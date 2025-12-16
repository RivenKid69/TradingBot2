# -*- coding: utf-8 -*-
"""
Tests for DSAR Phase 5 Service.

Comprehensive test coverage for GDPR Data Subject Access Requests including:
- Request lifecycle management
- Identity verification
- Deadline management (30 days + 60 days extension)
- Export generation with CCEA boundary
- Erasure with legal hold integration
- Exemption handling
- Metrics and statistics
- Audit trail

Design Doc Reference:
    - Phase 5: "DSAR: access/export/delete (Cloud data) with CCEA boundary clarity"
    - DoD: End-to-end tests, deadline computation, state transitions

CLOUD ZONE ONLY.
"""

import asyncio
import hashlib
import json
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from ..dsar_phase5 import (
    # Constants
    DSAR_RESPONSE_DEADLINE_DAYS,
    DSAR_EXTENSION_DAYS,
    DSAR_MAX_TOTAL_DAYS,
    DSAR_DATA_CATEGORIES,
    DSAR_OUT_OF_SCOPE_CATEGORIES,
    CCEA_BOUNDARY_NOTICE,
    ERASURE_EXEMPTIONS,
    COMPLIANCE_DATA_CATEGORIES,
    # Enums
    DSARRequestType,
    DSARStatus,
    VerificationMethod,
    VerificationStatus,
    ExportFormat,
    AuditAction,
    # Data classes
    VerificationToken,
    DSARRequest,
    DSARResult,
    DSARMetrics,
    AuditEntry,
    DataCategory,
    # Registry
    DATA_CATEGORIES,
    # Service
    DSARPhase5Service,
)
from ..retention_service import LegalHoldService, LegalHold, LegalHoldStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def export_dir():
    """Create temporary export directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_data_fetcher():
    """Mock data fetcher that returns sample data."""
    def fetcher(user_id: str, workspace_id: str, categories: Set[str]) -> List[Dict]:
        records = []
        for category in categories:
            for i in range(10):
                records.append({
                    "category": category,
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "record_id": f"{category}_{i}",
                    "data": f"Sample data for {category}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
        return records
    return fetcher


@pytest.fixture
def mock_data_deleter():
    """Mock data deleter that returns count."""
    def deleter(user_id: str, workspace_id: str, categories: Set[str]) -> int:
        return len(categories) * 10  # 10 records per category
    return deleter


@pytest.fixture
def legal_hold_service():
    """Create legal hold service for testing."""
    return LegalHoldService()


@pytest.fixture
def service(export_dir, mock_data_fetcher, mock_data_deleter, legal_hold_service):
    """Create DSAR Phase 5 service with mocks."""
    return DSARPhase5Service(
        legal_hold_service=legal_hold_service,
        data_fetcher=mock_data_fetcher,
        data_deleter=mock_data_deleter,
        export_dir=export_dir,
        base_url="https://api.example.com",
    )


@pytest.fixture
def service_no_fetcher(export_dir, legal_hold_service):
    """Create DSAR service without data fetcher."""
    return DSARPhase5Service(
        legal_hold_service=legal_hold_service,
        export_dir=export_dir,
    )


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Tests for DSAR constants."""

    def test_response_deadline_days(self):
        """Test standard response deadline is 30 days."""
        assert DSAR_RESPONSE_DEADLINE_DAYS == 30

    def test_extension_days(self):
        """Test extension period is 60 days."""
        assert DSAR_EXTENSION_DAYS == 60

    def test_max_total_days(self):
        """Test maximum total period is 90 days."""
        assert DSAR_MAX_TOTAL_DAYS == 90

    def test_in_scope_categories(self):
        """Test in-scope data categories are defined."""
        expected = {
            "telemetry_events",
            "alerts",
            "commands",
            "approval_records",
            "access_audits",
            "user_settings",
            "agent_data",
            "run_data",
            "deployment_data",
            "consent_records",
            "break_glass_requests",
            "session_data",
            "billing_records",
        }
        assert DSAR_DATA_CATEGORIES == expected

    def test_out_of_scope_categories(self):
        """Test out-of-scope data categories (Agent-controlled)."""
        expected = {
            "broker_credentials",
            "local_execution_logs",
            "order_fill_data",
            "local_vault_contents",
            "position_data_local",
            "local_strategy_source",
            "local_config_files",
        }
        assert DSAR_OUT_OF_SCOPE_CATEGORIES == expected

    def test_ccea_boundary_notice_content(self):
        """Test CCEA boundary notice contains required information."""
        assert "CCEA Architecture Data Boundary Notice" in CCEA_BOUNDARY_NOTICE
        assert "Cloud systems" in CCEA_BOUNDARY_NOTICE
        assert "Agent environment" in CCEA_BOUNDARY_NOTICE
        assert "Broker API credentials" in CCEA_BOUNDARY_NOTICE
        assert "RAW_ORDER_EVENTS" in CCEA_BOUNDARY_NOTICE
        assert "GDPR Article 20(2)" in CCEA_BOUNDARY_NOTICE

    def test_erasure_exemptions(self):
        """Test erasure exemption categories per Art. 17(3)."""
        assert "legal_obligation" in ERASURE_EXEMPTIONS
        assert "public_interest" in ERASURE_EXEMPTIONS
        assert "legal_claims" in ERASURE_EXEMPTIONS
        assert "regulatory_retention" in ERASURE_EXEMPTIONS

    def test_compliance_data_categories(self):
        """Test compliance data categories (7-year retention)."""
        expected = {
            "approval_records",
            "access_audits",
            "break_glass_requests",
            "governance_audit_logs",
            "billing_records",
        }
        assert COMPLIANCE_DATA_CATEGORIES == expected


# ============================================================================
# Data Category Registry Tests
# ============================================================================

class TestDataCategoryRegistry:
    """Tests for data category registry."""

    def test_all_in_scope_categories_registered(self):
        """Test all in-scope categories are in registry."""
        for category in DSAR_DATA_CATEGORIES:
            assert category in DATA_CATEGORIES, f"Missing category: {category}"

    def test_compliance_data_not_deletable(self):
        """Test compliance data categories are not deletable."""
        for category in COMPLIANCE_DATA_CATEGORIES:
            if category in DATA_CATEGORIES:
                cat_info = DATA_CATEGORIES[category]
                assert not cat_info.is_deletable, f"{category} should not be deletable"

    def test_approval_records_has_exemption(self):
        """Test approval_records has regulatory retention exemption."""
        cat_info = DATA_CATEGORIES["approval_records"]
        assert cat_info.exemption == "regulatory_retention"
        assert not cat_info.is_deletable
        assert cat_info.retention_days == 2555  # 7 years


# ============================================================================
# Request Creation Tests
# ============================================================================

class TestRequestCreation:
    """Tests for DSAR request creation."""

    @pytest.mark.asyncio
    async def test_create_access_request(self, service):
        """Test creating an ACCESS request."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        assert request.id is not None
        assert request.user_id == "user-123"
        assert request.workspace_id == "workspace-456"
        assert request.request_type == DSARRequestType.ACCESS
        assert request.status == DSARStatus.PENDING
        assert request.data_categories == DSAR_DATA_CATEGORIES

    @pytest.mark.asyncio
    async def test_create_erasure_request(self, service):
        """Test creating an ERASURE request."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            reason="User requested account deletion",
        )

        assert request.request_type == DSARRequestType.ERASURE
        assert request.reason == "User requested account deletion"
        assert request.requires_verification is True

    @pytest.mark.asyncio
    async def test_create_portability_request(self, service):
        """Test creating a PORTABILITY request."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.PORTABILITY,
        )

        assert request.request_type == DSARRequestType.PORTABILITY

    @pytest.mark.asyncio
    async def test_create_request_with_custom_categories(self, service):
        """Test creating request with specific data categories."""
        custom_categories = {"telemetry_events", "alerts"}
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
            data_categories=custom_categories,
        )

        assert request.data_categories == custom_categories

    @pytest.mark.asyncio
    async def test_create_request_filters_invalid_categories(self, service):
        """Test that invalid categories are filtered out."""
        categories = {"telemetry_events", "invalid_category", "alerts"}
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
            data_categories=categories,
        )

        assert "invalid_category" not in request.data_categories
        assert "telemetry_events" in request.data_categories
        assert "alerts" in request.data_categories

    @pytest.mark.asyncio
    async def test_create_request_sets_deadline(self, service):
        """Test that deadline is set to 30 days."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        expected_deadline = request.created_at + timedelta(days=30)
        assert abs((request.deadline - expected_deadline).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_rate_limiting(self, service):
        """Test rate limiting (max 12 requests per month)."""
        # Create 12 requests
        for i in range(12):
            await service.create_request(
                user_id="rate-limited-user",
                workspace_id=f"workspace-{i}",
                request_type=DSARRequestType.ACCESS,
            )

        # 13th request should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            await service.create_request(
                user_id="rate-limited-user",
                workspace_id="workspace-13",
                request_type=DSARRequestType.ACCESS,
            )


# ============================================================================
# Identity Verification Tests
# ============================================================================

class TestIdentityVerification:
    """Tests for identity verification."""

    @pytest.mark.asyncio
    async def test_initiate_verification(self, service):
        """Test initiating identity verification."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
            send_to="user@example.com",
        )

        assert token is not None
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.AWAITING_VERIFICATION
        assert updated.verification_method == VerificationMethod.EMAIL_LINK
        assert updated.verification_token is not None

    @pytest.mark.asyncio
    async def test_verify_identity_with_token(self, service):
        """Test identity verification with valid token."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        result = await service.verify_identity(
            request_id=request.id,
            token=token,
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.VERIFIED
        assert updated.verified_at is not None

    @pytest.mark.asyncio
    async def test_verify_identity_invalid_token(self, service):
        """Test verification with invalid token."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        result = await service.verify_identity(
            request_id=request.id,
            token="invalid_token",
        )

        assert result is False
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.AWAITING_VERIFICATION

    @pytest.mark.asyncio
    async def test_verify_identity_expired_token(self, service):
        """Test verification with expired token."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        # Manually expire the token
        request_obj = service.get_request(request.id)
        request_obj.verification_token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        result = await service.verify_identity(
            request_id=request.id,
            token=token,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_identity_max_attempts(self, service):
        """Test verification lockout after max attempts."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        # Try invalid tokens 5 times
        for _ in range(5):
            await service.verify_identity(
                request_id=request.id,
                token="wrong_token",
            )

        # 6th attempt should fail due to lockout
        result = await service.verify_identity(
            request_id=request.id,
            token="any_token",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_manual_verification(self, service):
        """Test manual verification by support."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = await service.verify_identity_manual(
            request_id=request.id,
            verification_method="document_review",
            verified_by="support-agent-1",
            notes="ID verified via passport",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.VERIFIED
        assert updated.verified_by == "support-agent-1"


# ============================================================================
# Deadline Management Tests
# ============================================================================

class TestDeadlineManagement:
    """Tests for deadline management per GDPR Art. 12(3)."""

    @pytest.mark.asyncio
    async def test_extend_deadline(self, service):
        """Test extending deadline once."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = await service.extend_deadline(
            request_id=request.id,
            reason="Complex request requiring additional time",
            extended_by="support-agent-1",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.extended_deadline is not None
        assert updated.status == DSARStatus.EXTENDED
        assert updated.extension_reason == "Complex request requiring additional time"

    @pytest.mark.asyncio
    async def test_cannot_extend_twice(self, service):
        """Test that deadline can only be extended once (GDPR requirement)."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # First extension succeeds
        result1 = await service.extend_deadline(
            request_id=request.id,
            reason="First reason",
            extended_by="agent",
        )
        assert result1 is True

        # Second extension fails
        result2 = await service.extend_deadline(
            request_id=request.id,
            reason="Second reason",
            extended_by="agent",
        )
        assert result2 is False

    @pytest.mark.asyncio
    async def test_extension_respects_max_deadline(self, service):
        """Test that extension doesn't exceed 90 days total."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.extend_deadline(
            request_id=request.id,
            reason="Complex request",
            extended_by="agent",
        )

        updated = service.get_request(request.id)
        max_deadline = request.created_at + timedelta(days=90)

        # Extended deadline should not exceed max
        assert updated.extended_deadline <= max_deadline

    @pytest.mark.asyncio
    async def test_effective_deadline(self, service):
        """Test effective deadline calculation."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Before extension
        assert request.effective_deadline == request.deadline

        # After extension
        await service.extend_deadline(
            request_id=request.id,
            reason="Complex",
            extended_by="agent",
        )

        updated = service.get_request(request.id)
        assert updated.effective_deadline == updated.extended_deadline

    @pytest.mark.asyncio
    async def test_days_until_deadline(self, service):
        """Test days until deadline calculation."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Should be approximately 30 days
        assert 29 <= request.days_until_deadline <= 30

    @pytest.mark.asyncio
    async def test_is_overdue(self, service):
        """Test overdue detection."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Not overdue initially
        assert request.is_overdue is False

        # Manually set deadline to past
        request.deadline = datetime.now(timezone.utc) - timedelta(days=1)
        assert request.is_overdue is True


# ============================================================================
# Request Processing Tests (ACCESS/PORTABILITY)
# ============================================================================

class TestAccessProcessing:
    """Tests for ACCESS and PORTABILITY request processing."""

    @pytest.mark.asyncio
    async def test_process_access_request(self, service, export_dir):
        """Test processing ACCESS request (full workflow)."""
        # Create request
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Verify identity
        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )
        await service.verify_identity(request_id=request.id, token=token)

        # Process
        result = await service.process_request(request.id)

        assert result.success is True
        assert result.status == DSARStatus.COMPLETED
        assert result.records_processed > 0
        assert result.export_path is not None
        assert result.export_checksum is not None
        assert result.export_checksum.startswith("sha256:")
        assert result.download_url is not None

    @pytest.mark.asyncio
    async def test_export_file_format(self, service, export_dir):
        """Test export file contains correct structure."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="manual",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        # Read export file
        with open(result.export_path, "r") as f:
            export_data = json.load(f)

        # Verify structure
        assert "metadata" in export_data
        assert "ccea_boundary" in export_data
        assert "data" in export_data

        # Verify metadata
        metadata = export_data["metadata"]
        assert metadata["request_id"] == request.id
        assert metadata["request_type"] == "access"
        assert metadata["user_id"] == "user-123"
        assert "gdpr_article" in metadata

        # Verify CCEA boundary
        boundary = export_data["ccea_boundary"]
        assert "notice" in boundary
        assert "in_scope_categories" in boundary
        assert "out_of_scope_categories" in boundary
        assert "explanation" in boundary

    @pytest.mark.asyncio
    async def test_export_checksum_valid(self, service, export_dir):
        """Test export checksum is valid."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="manual",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        # Verify checksum
        with open(result.export_path, "rb") as f:
            computed = hashlib.sha256(f.read()).hexdigest()

        assert result.export_checksum == f"sha256:{computed}"

    @pytest.mark.asyncio
    async def test_portability_request(self, service, export_dir):
        """Test PORTABILITY request processing."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.PORTABILITY,
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="manual",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        assert result.success is True
        assert result.request_type == DSARRequestType.PORTABILITY

    @pytest.mark.asyncio
    async def test_process_without_data_fetcher(self, service_no_fetcher):
        """Test processing fails without data fetcher."""
        request = await service_no_fetcher.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service_no_fetcher.verify_identity_manual(
            request_id=request.id,
            verification_method="manual",
            verified_by="admin",
        )

        result = await service_no_fetcher.process_request(request.id)

        assert result.success is False
        assert "Data fetcher not configured" in result.error


# ============================================================================
# Erasure Processing Tests
# ============================================================================

class TestErasureProcessing:
    """Tests for ERASURE request processing with legal hold."""

    @pytest.mark.asyncio
    async def test_erasure_requires_verification(self, service):
        """Test that erasure requires identity verification."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
        )

        # Process without verification
        result = await service.process_request(request.id)

        assert result.success is False
        assert result.status == DSARStatus.AWAITING_VERIFICATION
        assert "verification required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_erasure_with_verification(self, service):
        """Test erasure after verification."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            data_categories={"telemetry_events", "alerts", "session_data"},
        )

        # Verify identity
        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="document",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        assert result.success is True
        assert result.records_deleted > 0

    @pytest.mark.asyncio
    async def test_erasure_exempts_compliance_data(self, service):
        """Test that compliance data is exempt from erasure."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            data_categories={"approval_records", "access_audits", "telemetry_events"},
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="document",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        # Compliance data should be in exemptions
        assert "approval_records" in result.exemptions_applied
        assert "access_audits" in result.exemptions_applied
        # Status should be partially completed due to exemptions
        assert result.status == DSARStatus.PARTIALLY_COMPLETED

    @pytest.mark.asyncio
    async def test_erasure_blocked_by_legal_hold(self, service, legal_hold_service):
        """Test erasure is blocked by legal hold."""
        # Create legal hold (using category from ALL_DATA_CATEGORIES in retention_service)
        legal_hold_service.create_hold(
            workspace_id="workspace-456",
            data_type="alerts",  # alerts is in both DSAR_DATA_CATEGORIES and ALL_DATA_CATEGORIES
            reason="Ongoing investigation - do not delete",
            created_by="legal-team",
        )

        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            data_categories={"alerts", "session_data"},
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="document",
            verified_by="admin",
        )

        result = await service.process_request(request.id)

        # alerts should be blocked by legal hold
        assert "alerts" in result.exemptions_applied
        # Verify request was marked with legal hold info
        updated = service.get_request(request.id)
        assert updated.legal_hold_blocked is True


# ============================================================================
# Request Management Tests
# ============================================================================

class TestRequestManagement:
    """Tests for request management operations."""

    @pytest.mark.asyncio
    async def test_reject_request(self, service):
        """Test rejecting a DSAR request."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = await service.reject_request(
            request_id=request.id,
            reason="Manifestly unfounded request",
            rejected_by="dpo",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.REJECTED
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_request(self, service):
        """Test cancelling a DSAR request by user."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = await service.cancel_request(
            request_id=request.id,
            cancelled_by="user-123",
            reason="No longer needed",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed_request(self, service):
        """Test that completed requests cannot be cancelled."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="manual",
            verified_by="admin",
        )

        await service.process_request(request.id)

        result = await service.cancel_request(
            request_id=request.id,
            cancelled_by="user-123",
        )

        assert result is False


# ============================================================================
# Query and Monitoring Tests
# ============================================================================

class TestQueryAndMonitoring:
    """Tests for query and monitoring methods."""

    @pytest.mark.asyncio
    async def test_get_user_requests(self, service):
        """Test getting all requests for a user."""
        # Create multiple requests
        await service.create_request(
            user_id="user-123",
            workspace_id="ws-1",
            request_type=DSARRequestType.ACCESS,
        )
        await service.create_request(
            user_id="user-123",
            workspace_id="ws-2",
            request_type=DSARRequestType.ERASURE,
        )
        await service.create_request(
            user_id="user-456",
            workspace_id="ws-3",
            request_type=DSARRequestType.ACCESS,
        )

        requests = service.get_user_requests("user-123")
        assert len(requests) == 2
        assert all(r.user_id == "user-123" for r in requests)

    @pytest.mark.asyncio
    async def test_get_workspace_requests(self, service):
        """Test getting all requests for a workspace."""
        await service.create_request(
            user_id="user-1",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )
        await service.create_request(
            user_id="user-2",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
        )

        requests = service.get_workspace_requests("workspace-456")
        assert len(requests) == 2

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, service):
        """Test getting all pending requests."""
        r1 = await service.create_request(
            user_id="user-1",
            workspace_id="ws-1",
            request_type=DSARRequestType.ACCESS,
        )
        r2 = await service.create_request(
            user_id="user-2",
            workspace_id="ws-2",
            request_type=DSARRequestType.ACCESS,
        )

        # Complete one request
        await service.verify_identity_manual(r1.id, "manual", "admin")
        await service.process_request(r1.id)

        pending = service.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].id == r2.id

    @pytest.mark.asyncio
    async def test_get_overdue_requests(self, service):
        """Test getting overdue requests."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Manually make request overdue
        request.deadline = datetime.now(timezone.utc) - timedelta(days=1)

        overdue = service.get_overdue_requests()
        assert len(overdue) == 1
        assert overdue[0].id == request.id

    @pytest.mark.asyncio
    async def test_get_requests_by_deadline(self, service):
        """Test getting requests due within specified days."""
        r1 = await service.create_request(
            user_id="user-1",
            workspace_id="ws-1",
            request_type=DSARRequestType.ACCESS,
        )

        # Requests due within 7 days
        due_soon = service.get_requests_by_deadline(days=7)
        assert len(due_soon) == 0  # 30-day deadline, not due within 7 days

        # Manually adjust deadline
        r1.deadline = datetime.now(timezone.utc) + timedelta(days=5)
        due_soon = service.get_requests_by_deadline(days=7)
        assert len(due_soon) == 1

    @pytest.mark.asyncio
    async def test_check_expired_requests(self, service):
        """Test checking and marking expired requests."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Make request overdue
        request.deadline = datetime.now(timezone.utc) - timedelta(days=1)

        expired = service.check_expired_requests()
        assert len(expired) == 1
        assert expired[0].status == DSARStatus.EXPIRED


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Tests for DSAR metrics and statistics."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, service):
        """Test getting DSAR metrics."""
        # Create some requests
        for i in range(5):
            r = await service.create_request(
                user_id=f"user-{i}",
                workspace_id="workspace-456",
                request_type=DSARRequestType.ACCESS,
            )
            await service.verify_identity_manual(r.id, "manual", "admin")
            await service.process_request(r.id)

        for i in range(3):
            await service.create_request(
                user_id=f"user-{i+10}",
                workspace_id="workspace-456",
                request_type=DSARRequestType.ERASURE,
            )

        metrics = service.get_metrics(workspace_id="workspace-456")

        assert metrics.total_requests == 8
        assert metrics.by_type["access"] == 5
        assert metrics.by_type["erasure"] == 3
        assert metrics.by_status["completed"] == 5
        assert metrics.by_status["pending"] == 3

    @pytest.mark.asyncio
    async def test_metrics_completion_rate(self, service):
        """Test on-time completion rate calculation."""
        # Create and complete requests
        for i in range(10):
            r = await service.create_request(
                user_id=f"user-{i}",
                workspace_id="workspace-456",
                request_type=DSARRequestType.ACCESS,
            )
            await service.verify_identity_manual(r.id, "manual", "admin")
            await service.process_request(r.id)

        metrics = service.get_metrics()
        assert metrics.completed_on_time_rate == 1.0  # All completed on time

    @pytest.mark.asyncio
    async def test_metrics_to_dict(self, service):
        """Test metrics serialization."""
        await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        metrics = service.get_metrics()
        metrics_dict = metrics.to_dict()

        assert "total_requests" in metrics_dict
        assert "by_type" in metrics_dict
        assert "by_status" in metrics_dict
        assert "average_completion_days" in metrics_dict
        assert "completed_on_time_rate" in metrics_dict


# ============================================================================
# Audit Trail Tests
# ============================================================================

class TestAuditTrail:
    """Tests for audit trail."""

    @pytest.mark.asyncio
    async def test_audit_log_on_request_creation(self, service):
        """Test audit log entry on request creation."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        audit_log = service.get_audit_log(request_id=request.id)
        assert len(audit_log) > 0
        assert audit_log[0].action == AuditAction.REQUEST_CREATED

    @pytest.mark.asyncio
    async def test_audit_log_on_verification(self, service):
        """Test audit log entries for verification."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )

        audit_log = service.get_audit_log(request_id=request.id)
        actions = [e.action for e in audit_log]
        assert AuditAction.VERIFICATION_SENT in actions

    @pytest.mark.asyncio
    async def test_audit_log_on_completion(self, service):
        """Test audit log entries on completion."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(request.id, "manual", "admin")
        await service.process_request(request.id)

        audit_log = service.get_audit_log(request_id=request.id)
        actions = [e.action for e in audit_log]

        assert AuditAction.PROCESSING_STARTED in actions
        assert AuditAction.DATA_COLLECTED in actions
        assert AuditAction.EXPORT_GENERATED in actions
        assert AuditAction.REQUEST_COMPLETED in actions

    @pytest.mark.asyncio
    async def test_audit_entry_integrity_hash(self, service):
        """Test audit entry has integrity hash."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        audit_log = service.get_audit_log(request_id=request.id)
        for entry in audit_log:
            assert entry.integrity_hash is not None
            assert entry.integrity_hash.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_get_audit_trail(self, service):
        """Test getting complete audit trail for request."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(request.id, "manual", "admin")
        await service.process_request(request.id)

        trail = service.get_audit_trail(request.id)
        assert len(trail) >= 4  # Created, verified, started, completed
        assert all("timestamp" in entry for entry in trail)

    @pytest.mark.asyncio
    async def test_audit_callback(self, service_no_fetcher, export_dir, mock_data_fetcher):
        """Test audit callback is called."""
        callback_entries = []

        def callback(entry):
            callback_entries.append(entry)

        service = DSARPhase5Service(
            data_fetcher=mock_data_fetcher,
            export_dir=export_dir,
            audit_callback=callback,
        )

        await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        assert len(callback_entries) > 0
        assert callback_entries[0].action == AuditAction.REQUEST_CREATED


# ============================================================================
# CCEA Boundary Tests
# ============================================================================

class TestCCEABoundary:
    """Tests for CCEA boundary handling."""

    @pytest.mark.asyncio
    async def test_result_includes_ccea_boundary(self, service):
        """Test that result includes CCEA boundary information."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(request.id, "manual", "admin")
        result = await service.process_request(request.id)

        assert result.ccea_boundary_notice == CCEA_BOUNDARY_NOTICE
        assert result.in_scope_categories == DSAR_DATA_CATEGORIES
        assert result.out_of_scope_categories == DSAR_OUT_OF_SCOPE_CATEGORIES

    @pytest.mark.asyncio
    async def test_result_to_dict_includes_boundary(self, service):
        """Test that to_dict includes boundary information."""
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        await service.verify_identity_manual(request.id, "manual", "admin")
        result = await service.process_request(request.id)
        result_dict = result.to_dict()

        assert "ccea_boundary" in result_dict
        assert "notice" in result_dict["ccea_boundary"]
        assert "in_scope_categories" in result_dict["ccea_boundary"]
        assert "out_of_scope_categories" in result_dict["ccea_boundary"]
        assert "explanation" in result_dict["ccea_boundary"]


# ============================================================================
# Data Class Tests
# ============================================================================

class TestDataClasses:
    """Tests for data class functionality."""

    def test_verification_token_is_expired(self):
        """Test VerificationToken expiry check."""
        token = VerificationToken(
            token_hash="test",
            method=VerificationMethod.EMAIL_LINK,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert token.is_expired is True

    def test_verification_token_is_locked(self):
        """Test VerificationToken lockout."""
        token = VerificationToken(
            token_hash="test",
            method=VerificationMethod.EMAIL_LINK,
            attempts=5,
        )
        assert token.is_locked is True

    def test_dsar_request_to_dict(self):
        """Test DSARRequest serialization."""
        request = DSARRequest(
            request_type=DSARRequestType.ACCESS,
            user_id="user-123",
            workspace_id="workspace-456",
        )

        data = request.to_dict()
        assert data["id"] == request.id
        assert data["request_type"] == "access"
        assert data["user_id"] == "user-123"
        assert "timing" in data
        assert "verification" in data

    def test_dsar_result_to_dict(self):
        """Test DSARResult serialization."""
        result = DSARResult(
            success=True,
            request_id="req-123",
            request_type=DSARRequestType.ACCESS,
            records_processed=100,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["request_type"] == "access"
        assert data["processing"]["records_processed"] == 100
        assert "ccea_boundary" in data

    def test_audit_entry_to_dict(self):
        """Test AuditEntry serialization."""
        entry = AuditEntry(
            action=AuditAction.REQUEST_CREATED,
            request_id="req-123",
            user_id="user-123",
            workspace_id="ws-456",
        )

        data = entry.to_dict()
        assert data["action"] == "request_created"
        assert data["request_id"] == "req-123"
        assert "integrity_hash" in data

    def test_dsar_metrics_to_dict(self):
        """Test DSARMetrics serialization."""
        metrics = DSARMetrics(
            total_requests=100,
            by_type={"access": 70, "erasure": 30},
            overdue_count=2,
        )

        data = metrics.to_dict()
        assert data["total_requests"] == 100
        assert data["overdue_count"] == 2


# ============================================================================
# End-to-End Tests (DoD Requirements)
# ============================================================================

class TestEndToEnd:
    """End-to-end tests per DoD requirements."""

    @pytest.mark.asyncio
    async def test_full_access_workflow(self, service):
        """
        DoD: End-to-end test: create DSAR → verify identity → export → audit record exists.
        """
        # 1. Create request
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )
        assert request.status == DSARStatus.PENDING

        # 2. Initiate verification
        token = await service.initiate_verification(
            request_id=request.id,
            method=VerificationMethod.EMAIL_LINK,
        )
        assert token is not None

        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.AWAITING_VERIFICATION

        # 3. Complete verification
        verified = await service.verify_identity(
            request_id=request.id,
            token=token,
        )
        assert verified is True

        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.VERIFIED

        # 4. Process request
        result = await service.process_request(request.id)
        assert result.success is True
        assert result.status == DSARStatus.COMPLETED
        assert result.export_path is not None
        assert result.export_checksum is not None

        # 5. Verify audit trail
        audit_log = service.get_audit_log(request_id=request.id)
        actions = [e.action for e in audit_log]

        assert AuditAction.REQUEST_CREATED in actions
        assert AuditAction.VERIFICATION_SENT in actions
        assert AuditAction.VERIFICATION_COMPLETED in actions
        assert AuditAction.PROCESSING_STARTED in actions
        assert AuditAction.REQUEST_COMPLETED in actions

    @pytest.mark.asyncio
    async def test_full_erasure_workflow(self, service):
        """
        DoD: End-to-end test: create DSAR → verify identity → delete → audit record exists.
        """
        # 1. Create erasure request
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            data_categories={"telemetry_events", "alerts"},
            reason="Account deletion request",
        )

        # 2. Verify identity (MANDATORY for erasure)
        await service.verify_identity_manual(
            request_id=request.id,
            verification_method="document_review",
            verified_by="support-agent",
            notes="Passport verified",
        )

        # 3. Process erasure
        result = await service.process_request(request.id)
        assert result.success is True
        assert result.records_deleted > 0

        # 4. Verify audit trail
        audit_log = service.get_audit_log(request_id=request.id)
        actions = [e.action for e in audit_log]

        assert AuditAction.REQUEST_CREATED in actions
        assert AuditAction.ERASURE_STARTED in actions
        assert AuditAction.ERASURE_COMPLETED in actions

    @pytest.mark.asyncio
    async def test_deadline_computation(self, service):
        """
        DoD: Tests prove deadline computation and state transitions.
        """
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Standard deadline is 30 days
        assert request.days_until_deadline == 30 or request.days_until_deadline == 29

        # Extend deadline
        await service.extend_deadline(
            request_id=request.id,
            reason="Complex request",
            extended_by="agent",
        )

        updated = service.get_request(request.id)

        # State should be EXTENDED
        assert updated.status == DSARStatus.EXTENDED

        # Extended deadline should be ~60 days from now (max 90 from creation)
        max_days = (updated.created_at + timedelta(days=90) - datetime.now(timezone.utc)).days
        assert updated.days_until_deadline <= max_days

        # Cannot extend again
        second_extension = await service.extend_deadline(
            request_id=request.id,
            reason="More time needed",
            extended_by="agent",
        )
        assert second_extension is False

    @pytest.mark.asyncio
    async def test_erasure_with_legal_hold_and_exemptions(self, service, legal_hold_service):
        """
        DoD: Tests legal hold and exemption handling in erasure workflow.
        """
        # Create legal hold (using category from ALL_DATA_CATEGORIES in retention_service)
        legal_hold_service.create_hold(
            workspace_id="workspace-456",
            data_type="commands",  # commands is in both DSAR and retention categories
            reason="Legal investigation - preserve evidence",
            created_by="legal-team",
        )

        # Create erasure request including held data and compliance data
        request = await service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            data_categories={
                "alerts",  # Should be deleted
                "commands",  # Legal hold
                "approval_records",  # Compliance exemption
                "session_data",  # Should be deleted
            },
        )

        await service.verify_identity_manual(request.id, "manual", "admin")
        result = await service.process_request(request.id)

        # Verify exemptions
        assert "commands" in result.exemptions_applied  # Legal hold
        assert "approval_records" in result.exemptions_applied  # Compliance
        assert result.records_deleted > 0  # Some data deleted
        assert result.status == DSARStatus.PARTIALLY_COMPLETED

        # Verify audit trail captured legal hold check
        audit_log = service.get_audit_log(request_id=request.id)
        actions = [e.action for e in audit_log]
        assert AuditAction.LEGAL_HOLD_CHECK in actions


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_request(self, service):
        """Test getting non-existent request."""
        request = service.get_request("non-existent-id")
        assert request is None

    @pytest.mark.asyncio
    async def test_verify_nonexistent_request(self, service):
        """Test verifying non-existent request."""
        result = await service.verify_identity(
            request_id="non-existent-id",
            token="any-token",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_process_nonexistent_request(self, service):
        """Test processing non-existent request."""
        result = await service.process_request("non-existent-id")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_extend_nonexistent_request(self, service):
        """Test extending non-existent request."""
        result = await service.extend_deadline(
            request_id="non-existent-id",
            reason="Test",
            extended_by="agent",
        )
        assert result is False
