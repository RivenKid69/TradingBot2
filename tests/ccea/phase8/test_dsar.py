# -*- coding: utf-8 -*-
"""
Tests for GDPR DSAR (Data Subject Access Request) Service.

CCEA Phase 8 - GDPR compliance tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from packages.cloud.governance.dsar import (
    DSARService,
    DSARRequest,
    DSARResult,
    DSARRequestType,
    DSARStatus,
    DSAR_DATA_CATEGORIES,
    DSAR_RESPONSE_DEADLINE_DAYS,
    DSAR_EXTENSION_DAYS,
)


@pytest.fixture
def dsar_service():
    """Create DSAR service with mock data handlers."""
    mock_data = [
        {"id": 1, "user_id": "user-123", "data": "record1"},
        {"id": 2, "user_id": "user-123", "data": "record2"},
    ]

    def data_fetcher(user_id, workspace_id, categories):
        return [d for d in mock_data if d["user_id"] == user_id]

    def data_deleter(user_id, workspace_id, categories):
        count = len([d for d in mock_data if d["user_id"] == user_id])
        return count

    with tempfile.TemporaryDirectory() as tmpdir:
        yield DSARService(
            data_fetcher=data_fetcher,
            data_deleter=data_deleter,
            export_dir=Path(tmpdir),
        )


class TestDSARRequestCreation:
    """DSAR request creation tests."""

    def test_create_access_request(self, dsar_service):
        """Test creating data access request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        assert request.id is not None
        assert request.user_id == "user-123"
        assert request.request_type == DSARRequestType.ACCESS
        assert request.status == DSARStatus.PENDING

    def test_create_erasure_request(self, dsar_service):
        """Test creating data erasure request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
            reason="User requested account deletion",
        )

        assert request.request_type == DSARRequestType.ERASURE
        assert request.reason == "User requested account deletion"

    def test_request_has_deadline(self, dsar_service):
        """Test request has GDPR-compliant deadline."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # Deadline should be ~30 days from now
        expected_deadline = datetime.utcnow() + timedelta(days=DSAR_RESPONSE_DEADLINE_DAYS)
        delta = abs((request.deadline - expected_deadline).total_seconds())
        assert delta < 60  # Within a minute

    def test_request_data_categories(self, dsar_service):
        """Test specifying data categories."""
        categories = {"telemetry_events", "alerts"}
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
            data_categories=categories,
        )

        assert request.data_categories == categories


class TestIdentityVerification:
    """Identity verification tests."""

    def test_verify_identity(self, dsar_service):
        """Test identity verification."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = dsar_service.verify_identity(
            request_id=request.id,
            verification_method="email_confirmation",
            verified_by="support-agent",
        )

        assert result is True
        updated = dsar_service.get_request(request.id)
        assert updated.verified_at is not None
        assert updated.verification_method == "email_confirmation"
        assert updated.status == DSARStatus.IN_PROGRESS

    def test_erasure_requires_verification(self, dsar_service):
        """Test erasure requires identity verification."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
        )

        # Process without verification
        result = asyncio.run(dsar_service.process_request(request.id))

        assert result.success is False
        assert result.status == DSARStatus.AWAITING_VERIFICATION


class TestDeadlineExtension:
    """Deadline extension tests."""

    def test_extend_deadline(self, dsar_service):
        """Test extending deadline."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )
        original_deadline = request.deadline

        result = dsar_service.extend_deadline(
            request_id=request.id,
            reason="Complex request requiring more time",
            extended_by="admin",
        )

        assert result is True
        updated = dsar_service.get_request(request.id)
        assert updated.extended_deadline > original_deadline
        assert updated.status == DSARStatus.EXTENDED

    def test_can_only_extend_once(self, dsar_service):
        """Test can only extend deadline once per GDPR."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        # First extension succeeds
        dsar_service.extend_deadline(request.id, "Reason 1", "admin")

        # Second extension fails
        result = dsar_service.extend_deadline(request.id, "Reason 2", "admin")
        assert result is False


class TestAccessRequestProcessing:
    """Access request processing tests."""

    def test_process_access_request(self, dsar_service):
        """Test processing access request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = asyncio.run(dsar_service.process_request(request.id))

        assert result.success is True
        assert result.status == DSARStatus.COMPLETED
        assert result.records_processed > 0
        assert result.export_path is not None
        assert result.export_checksum is not None

    def test_export_file_created(self, dsar_service):
        """Test export file is created."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = asyncio.run(dsar_service.process_request(request.id))

        assert Path(result.export_path).exists()


class TestErasureRequestProcessing:
    """Erasure request processing tests."""

    def test_process_erasure_request(self, dsar_service):
        """Test processing erasure request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ERASURE,
        )

        # Verify identity first
        dsar_service.verify_identity(request.id, "email", "admin")

        result = asyncio.run(dsar_service.process_request(request.id))

        assert result.success is True
        assert result.status == DSARStatus.COMPLETED
        assert result.records_deleted > 0


class TestPortabilityRequest:
    """Portability request tests."""

    def test_process_portability_request(self, dsar_service):
        """Test processing portability request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.PORTABILITY,
        )

        result = asyncio.run(dsar_service.process_request(request.id))

        assert result.success is True
        assert result.export_path is not None


class TestRequestRejection:
    """Request rejection tests."""

    def test_reject_request(self, dsar_service):
        """Test rejecting a request."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        result = dsar_service.reject_request(
            request_id=request.id,
            reason="Unable to verify identity",
            rejected_by="admin",
        )

        assert result is True
        updated = dsar_service.get_request(request.id)
        assert updated.status == DSARStatus.REJECTED
        assert updated.completed_at is not None


class TestRequestQueries:
    """Request query tests."""

    def test_get_request(self, dsar_service):
        """Test getting request by ID."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="workspace-456",
            request_type=DSARRequestType.ACCESS,
        )

        retrieved = dsar_service.get_request(request.id)
        assert retrieved.id == request.id

    def test_get_user_requests(self, dsar_service):
        """Test getting all requests for a user."""
        dsar_service.create_request("user-123", "ws-1", DSARRequestType.ACCESS)
        dsar_service.create_request("user-123", "ws-2", DSARRequestType.ERASURE)
        dsar_service.create_request("user-456", "ws-1", DSARRequestType.ACCESS)

        user_requests = dsar_service.get_user_requests("user-123")
        assert len(user_requests) == 2

    def test_get_pending_requests(self, dsar_service):
        """Test getting pending requests."""
        dsar_service.create_request("user-123", "ws-1", DSARRequestType.ACCESS)
        dsar_service.create_request("user-456", "ws-1", DSARRequestType.ACCESS)

        pending = dsar_service.get_pending_requests()
        assert len(pending) >= 2


class TestOverdueDetection:
    """Overdue request detection tests."""

    def test_is_overdue_property(self, dsar_service):
        """Test is_overdue property."""
        request = DSARRequest(
            user_id="user-123",
            workspace_id="ws-1",
            deadline=datetime.utcnow() - timedelta(days=1),  # Past deadline
        )

        assert request.is_overdue is True

    def test_get_overdue_requests(self, dsar_service):
        """Test getting overdue requests."""
        # Create request with past deadline
        request = dsar_service.create_request("user-123", "ws-1", DSARRequestType.ACCESS)
        # Manually set past deadline
        request.deadline = datetime.utcnow() - timedelta(days=1)

        overdue = dsar_service.get_overdue_requests()
        assert len(overdue) >= 1


class TestAuditLog:
    """Audit log tests."""

    def test_audit_log_created(self, dsar_service):
        """Test audit log entries are created."""
        request = dsar_service.create_request(
            user_id="user-123",
            workspace_id="ws-1",
            request_type=DSARRequestType.ACCESS,
        )

        log = dsar_service.get_audit_log(request_id=request.id)
        assert len(log) > 0
        assert log[0]["action"] == "request_created"

    def test_audit_log_filtering(self, dsar_service):
        """Test audit log filtering by request ID."""
        req1 = dsar_service.create_request("user-1", "ws-1", DSARRequestType.ACCESS)
        req2 = dsar_service.create_request("user-2", "ws-1", DSARRequestType.ACCESS)

        log1 = dsar_service.get_audit_log(request_id=req1.id)
        log2 = dsar_service.get_audit_log(request_id=req2.id)

        assert all(e["request_id"] == req1.id for e in log1)
        assert all(e["request_id"] == req2.id for e in log2)


class TestDSARRequestSerialization:
    """Request serialization tests."""

    def test_request_to_dict(self):
        """Test request serialization."""
        request = DSARRequest(
            user_id="user-123",
            workspace_id="ws-456",
            request_type=DSARRequestType.ACCESS,
            status=DSARStatus.PENDING,
        )

        data = request.to_dict()

        assert data["user_id"] == "user-123"
        assert data["request_type"] == "ACCESS"
        assert data["status"] == "pending"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = DSARResult(
            success=True,
            request_id="req-123",
            request_type=DSARRequestType.ACCESS,
            records_processed=10,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["request_type"] == "ACCESS"
        assert data["records_processed"] == 10


class TestDataCategoriesConstant:
    """Data categories constant tests."""

    def test_data_categories_defined(self):
        """Test data categories are properly defined."""
        assert "telemetry_events" in DSAR_DATA_CATEGORIES
        assert "alerts" in DSAR_DATA_CATEGORIES
        assert "commands" in DSAR_DATA_CATEGORIES
        assert "access_audits" in DSAR_DATA_CATEGORIES


class TestServiceWithoutHandlers:
    """Tests for service without data handlers."""

    def test_export_without_fetcher(self):
        """Test export fails gracefully without data fetcher."""
        service = DSARService()
        request = service.create_request("user", "ws", DSARRequestType.ACCESS)

        result = asyncio.run(service.process_request(request.id))

        assert result.success is False
        assert "not configured" in result.error

    def test_erasure_without_deleter(self):
        """Test erasure fails gracefully without data deleter."""
        service = DSARService(data_fetcher=lambda *args: [])
        request = service.create_request("user", "ws", DSARRequestType.ERASURE)
        service.verify_identity(request.id, "email", "admin")

        result = asyncio.run(service.process_request(request.id))

        assert result.success is False
        assert "not configured" in result.error
