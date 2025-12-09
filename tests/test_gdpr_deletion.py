"""
Tests for GDPR Article 17 - Right to Erasure implementation.

References:
    - GDPR Regulation (EU) 2016/679 Article 17
    - EDPB Guidelines on the Right to Erasure
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from services.gdpr.data_deletion import (
    GDPRDeletionService,
    DeletionRequest,
    DeletionStatus,
    DataCategory,
    DeletionError,
    DeletionInProgressError,
    DeletionNotFoundError,
    InMemoryDeletionRequestStorage,
    InMemoryDataRepository,
    RetentionException,
)


class TestDeletionStatus:
    """Tests for DeletionStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all required statuses exist."""
        assert DeletionStatus.PENDING.value == "pending"
        assert DeletionStatus.IN_PROGRESS.value == "in_progress"
        assert DeletionStatus.COMPLETED.value == "completed"
        assert DeletionStatus.PARTIALLY_COMPLETED.value == "partially_completed"
        assert DeletionStatus.FAILED.value == "failed"
        assert DeletionStatus.CANCELLED.value == "cancelled"


class TestDataCategory:
    """Tests for DataCategory enum."""

    def test_all_categories_defined(self):
        """Verify all required data categories exist."""
        expected = [
            "account", "profile", "strategies", "backtests",
            "execution_logs", "broker_credentials", "analytics",
            "notifications", "sessions", "disclaimers", "audit_logs"
        ]
        actual = [c.value for c in DataCategory]
        for exp in expected:
            assert exp in actual, f"Missing category: {exp}"


class TestDeletionRequest:
    """Tests for DeletionRequest dataclass."""

    def test_creation(self):
        """Test DeletionRequest creation."""
        request = DeletionRequest(
            request_id="del_123",
            user_id="user_001",
            user_email="user@example.com",
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT, DataCategory.STRATEGIES],
            status=DeletionStatus.PENDING,
        )
        assert request.request_id == "del_123"
        assert request.user_id == "user_001"
        assert request.status == DeletionStatus.PENDING
        assert len(request.categories) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        request = DeletionRequest(
            request_id="del_123",
            user_id="user_001",
            user_email="user@example.com",
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT],
            status=DeletionStatus.PENDING,
        )
        data = request.to_dict()
        assert data["request_id"] == "del_123"
        assert data["user_id"] == "user_001"
        assert data["status"] == "pending"
        assert "account" in data["categories"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "request_id": "del_456",
            "user_id": "user_002",
            "user_email": "user2@example.com",
            "requested_at": "2024-01-01T00:00:00+00:00",
            "categories": ["account", "strategies"],
            "status": "completed",
            "completed_at": "2024-01-02T00:00:00+00:00",
            "retention_exceptions": ["audit_logs: Legal requirement"],
            "deleted_categories": ["account", "strategies"],
            "anonymized_categories": [],
        }
        request = DeletionRequest.from_dict(data)
        assert request.request_id == "del_456"
        assert request.status == DeletionStatus.COMPLETED
        assert DataCategory.ACCOUNT in request.categories


class TestInMemoryDeletionRequestStorage:
    """Tests for InMemoryDeletionRequestStorage."""

    @pytest.fixture
    def storage(self):
        return InMemoryDeletionRequestStorage()

    def test_save_and_get(self, storage):
        """Test saving and retrieving a request."""
        request = DeletionRequest(
            request_id="del_123",
            user_id="user_001",
            user_email=None,
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT],
            status=DeletionStatus.PENDING,
        )
        storage.save(request)
        retrieved = storage.get("del_123")
        assert retrieved is not None
        assert retrieved.request_id == "del_123"

    def test_get_nonexistent(self, storage):
        """Test getting a nonexistent request."""
        assert storage.get("nonexistent") is None

    def test_get_by_user(self, storage):
        """Test getting all requests for a user."""
        for i in range(3):
            storage.save(DeletionRequest(
                request_id=f"del_{i}",
                user_id="user_001",
                user_email=None,
                requested_at=datetime.now(timezone.utc),
                categories=[DataCategory.ACCOUNT],
                status=DeletionStatus.COMPLETED,
            ))
        storage.save(DeletionRequest(
            request_id="del_other",
            user_id="user_002",
            user_email=None,
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT],
            status=DeletionStatus.COMPLETED,
        ))

        user_requests = storage.get_by_user("user_001")
        assert len(user_requests) == 3

    def test_get_pending(self, storage):
        """Test getting pending requests."""
        storage.save(DeletionRequest(
            request_id="del_pending",
            user_id="user_001",
            user_email=None,
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT],
            status=DeletionStatus.PENDING,
        ))
        storage.save(DeletionRequest(
            request_id="del_completed",
            user_id="user_002",
            user_email=None,
            requested_at=datetime.now(timezone.utc),
            categories=[DataCategory.ACCOUNT],
            status=DeletionStatus.COMPLETED,
        ))

        pending = storage.get_pending()
        assert len(pending) == 1
        assert pending[0].request_id == "del_pending"


class TestInMemoryDataRepository:
    """Tests for InMemoryDataRepository."""

    @pytest.fixture
    def repo(self):
        return InMemoryDataRepository()

    def test_add_and_delete(self, repo):
        """Test adding and deleting records."""
        repo.add_record("user_001", {"id": 1, "data": "test"})
        repo.add_record("user_001", {"id": 2, "data": "test2"})

        count = repo.delete_by_user("user_001")
        assert count == 2

    def test_anonymize(self, repo):
        """Test anonymizing records."""
        repo.add_record("user_001", {
            "id": 1,
            "user_id": "user_001",
            "email": "user@example.com",
            "ip_address": "1.2.3.4",
            "name": "John Doe",
        })

        count = repo.anonymize_by_user("user_001")
        assert count == 1


class TestGDPRDeletionService:
    """Tests for GDPRDeletionService."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        return {
            "deletion_requests": InMemoryDeletionRequestStorage(),
            "account": InMemoryDataRepository(),
            "strategies": InMemoryDataRepository(),
            "backtests": InMemoryDataRepository(),
            "broker_credentials": InMemoryDataRepository(),
            "audit_logs": InMemoryDataRepository(),
        }

    @pytest.fixture
    def service(self, mock_repos):
        """Create service with mock repositories."""
        return GDPRDeletionService(mock_repos)

    def test_create_request(self, service):
        """Test creating a deletion request."""
        request = service.create_request("user_001", "user@example.com")

        assert request is not None
        assert request.user_id == "user_001"
        assert request.status == DeletionStatus.PENDING
        assert request.deadline is not None
        assert len(request.categories) == len(DataCategory)

    def test_create_request_with_specific_categories(self, service):
        """Test creating a request for specific categories."""
        categories = [DataCategory.ACCOUNT, DataCategory.STRATEGIES]
        request = service.create_request(
            "user_001",
            categories=categories,
        )

        assert len(request.categories) == 2
        assert DataCategory.ACCOUNT in request.categories
        assert DataCategory.STRATEGIES in request.categories

    def test_create_request_duplicate_raises(self, service):
        """Test that duplicate requests raise error."""
        service.create_request("user_001")

        with pytest.raises(DeletionInProgressError):
            service.create_request("user_001")

    def test_deadline_is_30_days(self, service):
        """Test GDPR 30-day deadline is set."""
        request = service.create_request("user_001")

        expected_deadline = request.requested_at + timedelta(days=30)
        assert request.deadline.date() == expected_deadline.date()

    def test_execute_deletion_completes(self, service, mock_repos):
        """Test successful deletion execution."""
        # Add some data
        mock_repos["account"].add_record("user_001", {"data": "test"})
        mock_repos["strategies"].add_record("user_001", {"data": "test"})

        request = service.create_request("user_001")
        result = service.execute_deletion(request)

        assert result.status == DeletionStatus.COMPLETED
        assert result.completed_at is not None
        assert "account" in result.deleted_categories

    def test_execute_deletion_within_deadline(self, service):
        """Test deletion completes within 30-day deadline."""
        request = service.create_request("user_001")
        result = service.execute_deletion(request)

        assert result.completed_at <= result.deadline

    def test_audit_logs_anonymized_not_deleted(self, service, mock_repos):
        """Test audit logs are anonymized, not deleted (retention exception)."""
        mock_repos["audit_logs"].add_record("user_001", {
            "user_id": "user_001",
            "action": "login",
        })

        request = service.create_request("user_001")
        result = service.execute_deletion(request)

        assert DataCategory.AUDIT_LOGS.value in result.anonymized_categories
        assert len(result.retention_exceptions) > 0

    def test_broker_credentials_deleted(self, service, mock_repos):
        """Test broker credentials are deleted."""
        mock_repos["broker_credentials"].add_record("user_001", {
            "api_key": "encrypted_key",
        })

        request = service.create_request("user_001")
        result = service.execute_deletion(request)

        assert "broker_credentials" in result.deleted_categories

    def test_get_deletion_report(self, service):
        """Test generating a deletion report."""
        request = service.create_request("user_001")
        service.execute_deletion(request)

        report = service.get_deletion_report(request.request_id)

        assert "request_id" in report
        assert "deleted_categories" in report
        assert "retention_exceptions" in report
        assert "compliance" in report
        assert report["compliance"]["gdpr_article"] == "Article 17 - Right to Erasure"

    def test_get_deletion_report_not_found(self, service):
        """Test report for nonexistent request raises error."""
        with pytest.raises(DeletionNotFoundError):
            service.get_deletion_report("nonexistent")

    def test_cancel_request(self, service):
        """Test canceling a pending request."""
        request = service.create_request("user_001")
        cancelled = service.cancel_request(request.request_id)

        assert cancelled.status == DeletionStatus.CANCELLED

    def test_cancel_completed_request_raises(self, service):
        """Test canceling a completed request raises error."""
        request = service.create_request("user_001")
        service.execute_deletion(request)

        with pytest.raises(DeletionError):
            service.cancel_request(request.request_id)

    def test_get_pending_requests(self, service):
        """Test getting all pending requests."""
        service.create_request("user_001")
        service.create_request("user_002")
        req3 = service.create_request("user_003")
        service.execute_deletion(req3)

        pending = service.get_pending_requests()
        assert len(pending) == 2

    def test_get_overdue_requests(self, service):
        """Test getting overdue requests."""
        # Create a request and manually set overdue deadline
        request = service.create_request("user_001")
        request.deadline = datetime.now(timezone.utc) - timedelta(days=1)
        service._request_storage.update(request)

        overdue = service.get_overdue_requests()
        assert len(overdue) == 1

    def test_process_pending_deletions(self, service):
        """Test batch processing of pending deletions."""
        service.create_request("user_001")
        service.create_request("user_002")

        results = service.process_pending_deletions()

        assert len(results) == 2
        assert all(r.status == DeletionStatus.COMPLETED for r in results)

    def test_callback_on_completion(self, mock_repos):
        """Test callback is called on deletion completion."""
        callback = MagicMock()
        service = GDPRDeletionService(mock_repos, on_deletion_complete=callback)

        request = service.create_request("user_001")
        service.execute_deletion(request)

        callback.assert_called_once()

    def test_user_deletion_history(self, service):
        """Test getting deletion history for a user."""
        service.create_request("user_001")
        service.process_pending_deletions()

        # Create another request after a bit
        service.create_request("user_001")

        history = service.get_user_deletion_history("user_001")
        assert len(history) == 2


class TestRetentionException:
    """Tests for RetentionException dataclass."""

    def test_creation(self):
        """Test RetentionException creation."""
        exception = RetentionException(
            category=DataCategory.AUDIT_LOGS,
            reason="Legal requirement",
            retention_period_years=5,
            legal_basis="GDPR Art. 17(3)(b)",
            anonymization_required=True,
        )
        assert exception.category == DataCategory.AUDIT_LOGS
        assert exception.retention_period_years == 5
        assert exception.anonymization_required is True


class TestGDPRDeletionServiceCompliance:
    """Compliance-focused tests for GDPRDeletionService."""

    @pytest.fixture
    def service(self):
        repos = {
            "deletion_requests": InMemoryDeletionRequestStorage(),
            "account": InMemoryDataRepository(),
            "audit_logs": InMemoryDataRepository(),
        }
        return GDPRDeletionService(repos)

    def test_deadline_within_30_days_per_gdpr(self, service):
        """GDPR Art. 12(3) - respond within one month."""
        request = service.create_request("user_001")
        delta = request.deadline - request.requested_at
        assert delta.days <= 30

    def test_retention_exception_for_legal_obligation(self, service):
        """GDPR Art. 17(3)(b) - retention for legal obligation."""
        assert DataCategory.AUDIT_LOGS in service.RETENTION_EXCEPTIONS
        exception = service.RETENTION_EXCEPTIONS[DataCategory.AUDIT_LOGS]
        assert "legal" in exception.reason.lower()
        assert "17(3)" in exception.legal_basis

    def test_report_includes_compliance_info(self, service):
        """Test report includes GDPR compliance information."""
        request = service.create_request("user_001")
        service.execute_deletion(request)
        report = service.get_deletion_report(request.request_id)

        assert report["compliance"]["gdpr_article"] == "Article 17 - Right to Erasure"
        assert report["compliance"]["deadline_days"] == 30
