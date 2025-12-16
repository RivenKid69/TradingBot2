# -*- coding: utf-8 -*-
"""
Tests for DSAR Service - GDPR Phase 1 Updates.

Tests cover:
    - CCEA boundary notice inclusion
    - In-scope vs out-of-scope categories
    - Export format compliance
    - Boundary text in responses

Design Doc Reference:
    - Phase 1: "Transparency + legal artifacts aligned to CCEA"
    - DSAR scope boundaries

CLOUD ZONE ONLY.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from ..dsar import (
    DSARService,
    DSARRequest,
    DSARRequestType,
    DSARStatus,
    DSARResult,
    DSAR_DATA_CATEGORIES,
    DSAR_OUT_OF_SCOPE_CATEGORIES,
    CCEA_BOUNDARY_NOTICE,
)


class TestDSARCCEABoundary:
    """Tests for CCEA boundary handling in DSAR."""

    @pytest.fixture
    def export_dir(self):
        """Create temporary export directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_data_fetcher(self):
        """Mock data fetcher that returns sample data."""
        def fetcher(user_id: str, workspace_id: str, categories: set):
            return [
                {"type": "telemetry", "user_id": user_id, "data": "sample"},
                {"type": "commands", "user_id": user_id, "data": "sample"},
            ]
        return fetcher

    @pytest.fixture
    def service(self, export_dir, mock_data_fetcher):
        """Create DSAR service with mock fetcher."""
        return DSARService(
            data_fetcher=mock_data_fetcher,
            export_dir=export_dir,
        )

    def test_dsar_result_includes_ccea_boundary(self):
        """Test that DSARResult includes CCEA boundary information."""
        result = DSARResult(
            success=True,
            request_id="test-123",
            request_type=DSARRequestType.ACCESS,
        )

        assert result.ccea_boundary_notice == CCEA_BOUNDARY_NOTICE
        assert result.in_scope_categories == DSAR_DATA_CATEGORIES
        assert result.out_of_scope_categories == DSAR_OUT_OF_SCOPE_CATEGORIES

    def test_dsar_result_to_dict_includes_boundary(self):
        """Test that to_dict includes boundary information."""
        result = DSARResult(
            success=True,
            request_id="test-123",
            request_type=DSARRequestType.ACCESS,
        )
        result_dict = result.to_dict()

        assert "ccea_boundary_notice" in result_dict
        assert "in_scope_categories" in result_dict
        assert "out_of_scope_categories" in result_dict
        assert result_dict["ccea_boundary_notice"] == CCEA_BOUNDARY_NOTICE

    def test_in_scope_categories(self):
        """Test that in-scope categories are correctly defined."""
        expected_in_scope = {
            "telemetry_events",
            "alerts",
            "commands",
            "approval_records",
            "access_audits",
            "user_settings",
            "agent_data",
            "run_data",
            "deployment_data",
        }
        assert DSAR_DATA_CATEGORIES == expected_in_scope

    def test_out_of_scope_categories(self):
        """Test that out-of-scope categories are correctly defined."""
        expected_out_of_scope = {
            "broker_credentials",
            "local_execution_logs",
            "order_fill_data",
            "local_vault_contents",
            "position_data_local",
        }
        assert DSAR_OUT_OF_SCOPE_CATEGORIES == expected_out_of_scope

    def test_ccea_boundary_notice_content(self):
        """Test that CCEA boundary notice contains required information."""
        assert "Cloud systems" in CCEA_BOUNDARY_NOTICE
        assert "Agent environment" in CCEA_BOUNDARY_NOTICE
        assert "Broker API credentials" in CCEA_BOUNDARY_NOTICE
        assert "Local execution logs" in CCEA_BOUNDARY_NOTICE
        assert "Order and fill data" in CCEA_BOUNDARY_NOTICE
        assert "RAW_ORDER_EVENTS" in CCEA_BOUNDARY_NOTICE
        assert "Local vault contents" in CCEA_BOUNDARY_NOTICE


class TestDSARExportBoundary:
    """Tests for CCEA boundary in DSAR exports."""

    @pytest.fixture
    def export_dir(self):
        """Create temporary export directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_data_fetcher(self):
        """Mock data fetcher that returns sample data."""
        def fetcher(user_id: str, workspace_id: str, categories: set):
            return [
                {"type": "telemetry", "user_id": user_id, "workspace_id": workspace_id},
                {"type": "commands", "user_id": user_id, "workspace_id": workspace_id},
            ]
        return fetcher

    @pytest.fixture
    def service(self, export_dir, mock_data_fetcher):
        """Create DSAR service with mock fetcher."""
        return DSARService(
            data_fetcher=mock_data_fetcher,
            export_dir=export_dir,
        )

    @pytest.mark.asyncio
    async def test_export_includes_ccea_boundary(self, service, export_dir):
        """Test that export JSON includes CCEA boundary section."""
        # Create request
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        # Verify identity
        service.verify_identity(
            request_id=request.id,
            verification_method="email_link",
            verified_by="system",
        )

        # Process request
        result = await service.process_request(request.id)

        assert result.success
        assert result.export_path is not None

        # Read export file
        with open(result.export_path, "r") as f:
            export_data = json.load(f)

        # Verify CCEA boundary section exists
        assert "ccea_boundary" in export_data
        boundary = export_data["ccea_boundary"]

        assert "notice" in boundary
        assert "in_scope_categories" in boundary
        assert "out_of_scope_categories" in boundary
        assert "explanation" in boundary

    @pytest.mark.asyncio
    async def test_export_boundary_notice_matches_constant(self, service, export_dir):
        """Test that export boundary notice matches the constant."""
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )
        service.verify_identity(
            request_id=request.id,
            verification_method="email_link",
            verified_by="system",
        )
        result = await service.process_request(request.id)

        with open(result.export_path, "r") as f:
            export_data = json.load(f)

        assert export_data["ccea_boundary"]["notice"] == CCEA_BOUNDARY_NOTICE

    @pytest.mark.asyncio
    async def test_export_includes_correct_categories(self, service, export_dir):
        """Test that export includes correct category lists."""
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )
        service.verify_identity(
            request_id=request.id,
            verification_method="email_link",
            verified_by="system",
        )
        result = await service.process_request(request.id)

        with open(result.export_path, "r") as f:
            export_data = json.load(f)

        boundary = export_data["ccea_boundary"]
        assert set(boundary["in_scope_categories"]) == DSAR_DATA_CATEGORIES
        assert set(boundary["out_of_scope_categories"]) == DSAR_OUT_OF_SCOPE_CATEGORIES

    @pytest.mark.asyncio
    async def test_export_explanation_is_clear(self, service, export_dir):
        """Test that export explanation is clear about scope."""
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )
        service.verify_identity(
            request_id=request.id,
            verification_method="email_link",
            verified_by="system",
        )
        result = await service.process_request(request.id)

        with open(result.export_path, "r") as f:
            export_data = json.load(f)

        explanation = export_data["ccea_boundary"]["explanation"]
        assert "Cloud-controlled" in explanation
        assert "Agent-zone" in explanation
        assert "broker credentials" in explanation.lower()


class TestDSARRequestCreation:
    """Tests for DSAR request creation."""

    def test_create_access_request(self):
        """Test creating an access request."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        assert request.id is not None
        assert request.user_id == "user_123"
        assert request.workspace_id == "ws_456"
        assert request.request_type == DSARRequestType.ACCESS
        assert request.status == DSARStatus.PENDING
        assert request.data_categories == DSAR_DATA_CATEGORIES

    def test_create_erasure_request(self):
        """Test creating an erasure request."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ERASURE,
            reason="User requested account deletion",
        )

        assert request.request_type == DSARRequestType.ERASURE
        assert request.reason == "User requested account deletion"

    def test_create_request_with_custom_categories(self):
        """Test creating request with specific data categories."""
        service = DSARService()
        custom_categories = {"telemetry_events", "commands"}
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
            data_categories=custom_categories,
        )

        assert request.data_categories == custom_categories


class TestDSARVerification:
    """Tests for DSAR identity verification."""

    def test_verify_identity(self):
        """Test identity verification."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        result = service.verify_identity(
            request_id=request.id,
            verification_method="email_link",
            verified_by="support_agent",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.verified_at is not None
        assert updated.verification_method == "email_link"
        assert updated.status == DSARStatus.IN_PROGRESS

    def test_verify_nonexistent_request(self):
        """Test verifying non-existent request."""
        service = DSARService()
        result = service.verify_identity(
            request_id="non-existent",
            verification_method="email_link",
            verified_by="support_agent",
        )
        assert result is False


class TestDSARDeadlines:
    """Tests for DSAR deadline handling."""

    def test_request_has_deadline(self):
        """Test that request has a 30-day deadline."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        # Deadline should be ~30 days from now
        expected_deadline = request.created_at + pytest.importorskip("datetime").timedelta(days=30)
        assert abs((request.deadline - expected_deadline).total_seconds()) < 5

    def test_extend_deadline(self):
        """Test extending the deadline."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        result = service.extend_deadline(
            request_id=request.id,
            reason="Complex request requiring more time",
            extended_by="support_agent",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.extended_deadline is not None
        assert updated.status == DSARStatus.EXTENDED

    def test_cannot_extend_twice(self):
        """Test that deadline can only be extended once."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        # First extension
        assert service.extend_deadline(request.id, "First reason", "agent") is True

        # Second extension should fail
        assert service.extend_deadline(request.id, "Second reason", "agent") is False


class TestDSARAudit:
    """Tests for DSAR audit logging."""

    def test_audit_log_created_on_request(self):
        """Test that audit log is created on request creation."""
        service = DSARService()
        service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        audit_log = service.get_audit_log()
        assert len(audit_log) > 0
        assert audit_log[0]["action"] == "request_created"

    def test_audit_log_filter_by_request_id(self):
        """Test filtering audit log by request ID."""
        service = DSARService()
        request1 = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )
        request2 = service.create_request(
            user_id="user_456",
            workspace_id="ws_789",
            request_type=DSARRequestType.ERASURE,
        )

        filtered = service.get_audit_log(request_id=request1.id)
        assert all(entry["request_id"] == request1.id for entry in filtered)


class TestDSARRejection:
    """Tests for DSAR request rejection."""

    def test_reject_request(self):
        """Test rejecting a DSAR request."""
        service = DSARService()
        request = service.create_request(
            user_id="user_123",
            workspace_id="ws_456",
            request_type=DSARRequestType.ACCESS,
        )

        result = service.reject_request(
            request_id=request.id,
            reason="Manifestly unfounded request",
            rejected_by="dpo",
        )

        assert result is True
        updated = service.get_request(request.id)
        assert updated.status == DSARStatus.REJECTED
        assert updated.completed_at is not None

    def test_reject_nonexistent_request(self):
        """Test rejecting non-existent request."""
        service = DSARService()
        result = service.reject_request(
            request_id="non-existent",
            reason="Test",
            rejected_by="dpo",
        )
        assert result is False


class TestDSARQueries:
    """Tests for DSAR query methods."""

    def test_get_user_requests(self):
        """Test getting all requests for a user."""
        service = DSARService()

        # Create multiple requests
        service.create_request(
            user_id="user_123",
            workspace_id="ws_1",
            request_type=DSARRequestType.ACCESS,
        )
        service.create_request(
            user_id="user_123",
            workspace_id="ws_2",
            request_type=DSARRequestType.ERASURE,
        )
        service.create_request(
            user_id="user_456",
            workspace_id="ws_3",
            request_type=DSARRequestType.ACCESS,
        )

        user_requests = service.get_user_requests("user_123")
        assert len(user_requests) == 2
        assert all(r.user_id == "user_123" for r in user_requests)

    def test_get_pending_requests(self):
        """Test getting all pending requests."""
        service = DSARService()

        request1 = service.create_request(
            user_id="user_123",
            workspace_id="ws_1",
            request_type=DSARRequestType.ACCESS,
        )
        service.create_request(
            user_id="user_456",
            workspace_id="ws_2",
            request_type=DSARRequestType.ACCESS,
        )

        # Reject one
        service.reject_request(request1.id, "Test", "dpo")

        pending = service.get_pending_requests()
        assert len(pending) == 1
