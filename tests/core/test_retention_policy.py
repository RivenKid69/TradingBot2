# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 4 Retention Policy.

Tests retention policy management for MiFIR Article 25 compliance including:
- Default 5-year retention period
- Extended 7-year retention on NCA request
- Legal hold management
- Archival operations
- NCA request handling

Coverage target: 100%
"""

import pytest
import time
from datetime import datetime, timedelta
from decimal import Decimal

from services.core.risk_controls.audit_models import (
    AuditRecord,
    AuditRecordBuilder,
    AuditEventType,
    AuditExportRequest,
    OrderSide,
)
from services.core.risk_controls.audit_storage import (
    MemoryAuditStorage,
    AuditStorageConfig,
    StorageBackendType,
)
from services.core.risk_controls.retention_policy import (
    RetentionPeriod,
    ArchiveStatus,
    NCARequestType,
    RetentionPolicyConfig,
    NCARequest,
    RetentionRecord,
    RetentionMetrics,
    ArchiveOperation,
    RetentionManager,
    create_retention_manager,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def retention_config():
    """Create retention policy configuration."""
    return RetentionPolicyConfig(
        default_retention_years=5,
        extended_retention_years=7,
        archive_after_days=365,
        cold_storage_path="state/test_archive",
        enable_compression=True,
        deletion_requires_approval=True,
    )


@pytest.fixture
def hot_storage():
    """Create hot storage for testing."""
    config = AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
    return MemoryAuditStorage(config)


@pytest.fixture
def cold_storage():
    """Create cold storage for testing."""
    config = AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
    return MemoryAuditStorage(config)


@pytest.fixture
def retention_manager(hot_storage, cold_storage, retention_config):
    """Create retention manager with hot and cold storage."""
    manager = RetentionManager(
        hot_storage=hot_storage,
        config=retention_config,
        cold_storage=cold_storage,
    )
    return manager


@pytest.fixture
def sample_records():
    """Create sample records for testing."""
    records = []
    for i in range(10):
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id(f"order-{i:03d}")
            .algorithm(f"algo-{i % 3:03d}")
            .instrument("US0378331005")
            .side(OrderSide.BUY)
            .quantity(100)
            .price(150)
            .sequence(i + 1)
            .build_unsafe()
        )
        records.append(record)
    return records


@pytest.fixture
def nca_request():
    """Create sample NCA request."""
    return NCARequest(
        request_id="nca-001",
        request_type=NCARequestType.DATA_EXPORT,
        nca_identifier="FCA",
        request_date=datetime.utcnow(),
        effective_date=datetime.utcnow(),
        scope_start_date=datetime.utcnow() - timedelta(days=365),
        scope_end_date=datetime.utcnow(),
        reference_number="FCA/2024/12345",
        contact_person="compliance@fca.gov.uk",
    )


# =============================================================================
# Enum Tests
# =============================================================================

class TestRetentionPeriod:
    """Tests for RetentionPeriod enum."""

    def test_standard_period(self):
        """Test standard retention period."""
        assert RetentionPeriod.STANDARD.value == 5

    def test_extended_period(self):
        """Test extended retention period."""
        assert RetentionPeriod.EXTENDED.value == 7

    def test_minimum_audit_period(self):
        """Test minimum audit period."""
        assert RetentionPeriod.MINIMUM_AUDIT.value == 2

    def test_maximum_period(self):
        """Test maximum retention period."""
        assert RetentionPeriod.MAXIMUM.value == 10


class TestArchiveStatus:
    """Tests for ArchiveStatus enum."""

    def test_all_statuses(self):
        """Test all archive statuses exist."""
        assert ArchiveStatus.ACTIVE.value == "active"
        assert ArchiveStatus.PENDING_ARCHIVE.value == "pending_archive"
        assert ArchiveStatus.ARCHIVED.value == "archived"
        assert ArchiveStatus.PENDING_DELETION.value == "pending_deletion"
        assert ArchiveStatus.DELETED.value == "deleted"
        assert ArchiveStatus.HELD.value == "held"


class TestNCARequestType:
    """Tests for NCARequestType enum."""

    def test_all_request_types(self):
        """Test all NCA request types exist."""
        assert NCARequestType.DATA_EXPORT.value == "data_export"
        assert NCARequestType.RETENTION_EXTENSION.value == "retention_extension"
        assert NCARequestType.LEGAL_HOLD.value == "legal_hold"
        assert NCARequestType.INVESTIGATION.value == "investigation"
        assert NCARequestType.AUDIT.value == "audit"


# =============================================================================
# Config Tests
# =============================================================================

class TestRetentionPolicyConfig:
    """Tests for RetentionPolicyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetentionPolicyConfig()

        assert config.default_retention_years == 5
        assert config.extended_retention_years == 7
        assert config.archive_after_days == 365
        assert config.deletion_requires_approval is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetentionPolicyConfig(
            default_retention_years=6,
            extended_retention_years=8,
            archive_after_days=180,
        )

        assert config.default_retention_years == 6
        assert config.extended_retention_years == 8
        assert config.archive_after_days == 180


# =============================================================================
# Data Class Tests
# =============================================================================

class TestNCARequest:
    """Tests for NCARequest dataclass."""

    def test_nca_request_creation(self, nca_request):
        """Test NCA request creation."""
        assert nca_request.request_id == "nca-001"
        assert nca_request.request_type == NCARequestType.DATA_EXPORT
        assert nca_request.nca_identifier == "FCA"
        assert nca_request.status == "active"

    def test_nca_request_with_filters(self):
        """Test NCA request with specific filters."""
        request = NCARequest(
            request_id="nca-002",
            request_type=NCARequestType.INVESTIGATION,
            nca_identifier="BaFin",
            request_date=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            order_ids=["order-001", "order-002"],
            algorithm_ids=["algo-001"],
            instrument_isins=["US0378331005"],
        )

        assert len(request.order_ids) == 2
        assert len(request.algorithm_ids) == 1


class TestRetentionRecord:
    """Tests for RetentionRecord dataclass."""

    def test_retention_record_creation(self):
        """Test retention record creation."""
        record = RetentionRecord(
            record_id="record-001",
            created_date=datetime.utcnow(),
            retention_period_years=5,
        )

        assert record.record_id == "record-001"
        assert record.status == ArchiveStatus.ACTIVE
        assert record.retention_period_years == 5
        assert record.legal_hold is False

    def test_retention_record_with_legal_hold(self):
        """Test retention record with legal hold."""
        record = RetentionRecord(
            record_id="record-002",
            created_date=datetime.utcnow(),
            legal_hold=True,
            legal_hold_reference="NCA/2024/001",
        )

        assert record.legal_hold is True
        assert record.legal_hold_reference == "NCA/2024/001"


class TestRetentionMetrics:
    """Tests for RetentionMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = RetentionMetrics()

        assert metrics.total_records == 0
        assert metrics.active_records == 0
        assert metrics.archived_records == 0
        assert metrics.deleted_records == 0


class TestArchiveOperation:
    """Tests for ArchiveOperation dataclass."""

    def test_archive_operation_creation(self):
        """Test archive operation creation."""
        operation = ArchiveOperation(
            operation_id="op-001",
            operation_type="archive",
            start_time=datetime.utcnow(),
        )

        assert operation.operation_id == "op-001"
        assert operation.operation_type == "archive"
        assert operation.records_processed == 0
        assert operation.success is True


# =============================================================================
# RetentionManager Tests
# =============================================================================

class TestRetentionManager:
    """Tests for RetentionManager class."""

    def test_initialization(self, retention_manager):
        """Test retention manager initialization."""
        assert retention_manager.hot_storage is not None
        assert retention_manager.cold_storage is not None
        assert retention_manager.config.default_retention_years == 5

    def test_get_retention_expiry_default(self, retention_manager):
        """Test default retention expiry calculation."""
        record_date = datetime.utcnow()
        expiry = retention_manager.get_retention_expiry(record_date)

        expected = record_date + timedelta(days=5 * 365)
        assert abs((expiry - expected).total_seconds()) < 1

    def test_get_retention_expiry_extended(self, retention_manager):
        """Test extended retention expiry calculation."""
        record_date = datetime.utcnow()
        expiry = retention_manager.get_retention_expiry(record_date, extended=True)

        expected = record_date + timedelta(days=7 * 365)
        assert abs((expiry - expected).total_seconds()) < 1

    def test_get_archive_date(self, retention_manager):
        """Test archive date calculation."""
        record_date = datetime.utcnow()
        archive_date = retention_manager.get_archive_date(record_date)

        expected = record_date + timedelta(days=365)
        assert abs((archive_date - expected).total_seconds()) < 1

    def test_is_under_retention_recent(self, retention_manager):
        """Test recent records are under retention."""
        record_date = datetime.utcnow()
        assert retention_manager.is_under_retention(record_date) is True

    def test_is_under_retention_old(self, retention_manager):
        """Test old records are not under retention."""
        record_date = datetime.utcnow() - timedelta(days=6 * 365)
        assert retention_manager.is_under_retention(record_date) is False

    def test_is_under_retention_extended(self, retention_manager):
        """Test extended retention period."""
        # 6 years old - beyond standard but within extended
        record_date = datetime.utcnow() - timedelta(days=6 * 365)
        assert retention_manager.is_under_retention(record_date, extended=True) is True

        # 8 years old - beyond extended
        record_date = datetime.utcnow() - timedelta(days=8 * 365)
        assert retention_manager.is_under_retention(record_date, extended=True) is False

    def test_is_ready_for_archive_recent(self, retention_manager):
        """Test recent records are not ready for archive."""
        record_date = datetime.utcnow()
        assert retention_manager.is_ready_for_archive(record_date) is False

    def test_is_ready_for_archive_old(self, retention_manager):
        """Test old records are ready for archive."""
        record_date = datetime.utcnow() - timedelta(days=400)
        assert retention_manager.is_ready_for_archive(record_date) is True

    def test_is_eligible_for_deletion_under_retention(self, retention_manager, sample_records):
        """Test records under retention are not eligible for deletion."""
        record = sample_records[0]
        can_delete, reason = retention_manager.is_eligible_for_deletion(record)

        assert can_delete is False
        assert "Under retention" in reason

    def test_is_eligible_for_deletion_with_legal_hold(self, retention_manager, sample_records):
        """Test records with legal hold are not eligible for deletion."""
        record = sample_records[0]
        retention_record = RetentionRecord(
            record_id=record.record_id,
            created_date=datetime.utcnow() - timedelta(days=6 * 365),
            legal_hold=True,
            legal_hold_reference="NCA/2024/001",
        )

        can_delete, reason = retention_manager.is_eligible_for_deletion(record, retention_record)

        assert can_delete is False
        assert "Legal hold" in reason


class TestNCARequestManagement:
    """Tests for NCA request management."""

    def test_register_nca_request(self, retention_manager, nca_request):
        """Test registering NCA request."""
        result = retention_manager.register_nca_request(nca_request)

        assert result is True
        assert nca_request.request_id in retention_manager._nca_requests

    def test_close_nca_request(self, retention_manager, nca_request):
        """Test closing NCA request."""
        retention_manager.register_nca_request(nca_request)

        result = retention_manager.close_nca_request(nca_request.request_id, "Completed")

        assert result is True
        assert retention_manager._nca_requests[nca_request.request_id].status == "closed"

    def test_close_nonexistent_request(self, retention_manager):
        """Test closing non-existent request."""
        result = retention_manager.close_nca_request("non-existent")
        assert result is False

    def test_register_legal_hold_request(self, retention_manager):
        """Test registering legal hold request."""
        request = NCARequest(
            request_id="hold-001",
            request_type=NCARequestType.LEGAL_HOLD,
            nca_identifier="FCA",
            request_date=datetime.utcnow(),
            effective_date=datetime.utcnow(),
        )

        result = retention_manager.register_nca_request(request)

        assert result is True


class TestLegalHoldManagement:
    """Tests for legal hold management."""

    def test_set_legal_hold(self, retention_manager):
        """Test setting legal hold."""
        result = retention_manager.set_legal_hold(
            record_id="record-001",
            reference="NCA/2024/001",
            reason="Investigation",
        )

        assert result is True
        assert "record-001" in retention_manager._retention_records
        assert retention_manager._retention_records["record-001"].legal_hold is True

    def test_release_legal_hold(self, retention_manager):
        """Test releasing legal hold."""
        # Set hold first
        retention_manager.set_legal_hold("record-001", "NCA/2024/001")

        # Release it
        result = retention_manager.release_legal_hold("record-001")

        assert result is True
        assert retention_manager._retention_records["record-001"].legal_hold is False

    def test_release_nonexistent_hold(self, retention_manager):
        """Test releasing non-existent hold."""
        result = retention_manager.release_legal_hold("non-existent")
        assert result is False


class TestArchiveOperations:
    """Tests for archive operations."""

    def test_archive_old_records(self, retention_manager, hot_storage, sample_records):
        """Test archiving old records."""
        # Add records to hot storage
        for record in sample_records:
            hot_storage.append(record)

        # Archive with past cutoff (so all records are archived)
        operation = retention_manager.archive_old_records(
            cutoff_date=datetime.utcnow() + timedelta(days=1)
        )

        assert operation.operation_type == "archive"
        assert operation.success

    def test_archive_with_no_cold_storage(self, hot_storage, retention_config):
        """Test archive fails without cold storage."""
        manager = RetentionManager(hot_storage=hot_storage, config=retention_config)
        manager.cold_storage = None

        operation = manager.archive_old_records()

        assert operation.success is False
        assert "Cold storage not configured" in operation.error_message

    def test_delete_expired_records_dry_run(self, retention_manager, hot_storage, sample_records):
        """Test deleting expired records (dry run)."""
        # Add records to hot storage
        for record in sample_records:
            hot_storage.append(record)

        operation = retention_manager.delete_expired_records(dry_run=True)

        assert operation.operation_type == "delete"
        assert operation.success


class TestNCAExport:
    """Tests for NCA export functionality."""

    def test_prepare_nca_export(self, retention_manager, hot_storage, sample_records, nca_request):
        """Test preparing NCA export."""
        # Add records to hot storage
        for record in sample_records:
            hot_storage.append(record)

        result = retention_manager.prepare_nca_export(nca_request)

        assert result.success
        assert result.records_exported >= 0


class TestMetrics:
    """Tests for retention metrics."""

    def test_get_metrics(self, retention_manager, hot_storage, sample_records):
        """Test getting retention metrics."""
        for record in sample_records:
            hot_storage.append(record)

        metrics = retention_manager.get_metrics()

        assert metrics.total_records == len(sample_records)


class TestRetentionReport:
    """Tests for retention report generation."""

    def test_generate_retention_report(self, retention_manager, hot_storage, sample_records, nca_request):
        """Test generating retention report."""
        for record in sample_records:
            hot_storage.append(record)

        retention_manager.register_nca_request(nca_request)

        report = retention_manager.generate_retention_report()

        assert "report_date" in report
        assert "policy_version" in report
        assert "default_retention_years" in report
        assert report["default_retention_years"] == 5
        assert "metrics" in report
        assert "active_nca_requests" in report


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateRetentionManager:
    """Tests for create_retention_manager factory function."""

    def test_create_with_defaults(self, hot_storage):
        """Test creating manager with defaults."""
        manager = create_retention_manager(hot_storage)

        assert manager.hot_storage is hot_storage
        assert manager.config.default_retention_years == 5

    def test_create_with_custom_config(self, hot_storage):
        """Test creating manager with custom config."""
        config = RetentionPolicyConfig(
            default_retention_years=6,
            archive_after_days=180,
        )

        manager = create_retention_manager(hot_storage, config=config)

        assert manager.config.default_retention_years == 6

    def test_create_with_cold_storage_path(self, hot_storage, tmp_path):
        """Test creating manager with cold storage path."""
        cold_path = str(tmp_path / "custom_archive")
        manager = create_retention_manager(
            hot_storage,
            cold_storage_path=cold_path,
        )

        assert manager.config.cold_storage_path == cold_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
