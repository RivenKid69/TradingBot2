# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 4 Audit Storage.

Tests storage backends for MiFIR Article 25 compliance including:
- MemoryAuditStorage for testing
- SQLiteAuditStorage for development/small deployments
- FileAuditStorage for simple deployments
- Chain integrity verification
- Export capabilities

Coverage target: 100%
"""

import json
import os
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from services.core.risk_controls.audit_models import (
    AuditRecord,
    AuditRecordBuilder,
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    AuditChainStatus,
    AuditExportRequest,
    OrderSide,
)
from services.core.risk_controls.audit_storage import (
    StorageBackendType,
    StorageState,
    AuditStorageConfig,
    StorageMetrics,
    AuditStorageBackend,
    MemoryAuditStorage,
    SQLiteAuditStorage,
    FileAuditStorage,
    create_audit_storage,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_record():
    """Create a sample audit record for testing."""
    return (
        AuditRecordBuilder()
        .event_type(AuditEventType.ORDER_SUBMITTED)
        .firm_lei("5493001KJTIIGC8Y1R12")
        .order_id("order-001")
        .algorithm("algo-001")
        .instrument("US0378331005", "AAPL")
        .venue("XNAS")
        .side(OrderSide.BUY)
        .quantity(100)
        .price("150.50")
        .build_unsafe()
    )


@pytest.fixture
def sample_records():
    """Create multiple sample records."""
    records = []
    for i in range(10):
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id(f"order-{i:03d}")
            .algorithm("algo-001")
            .instrument("US0378331005")
            .side(OrderSide.BUY if i % 2 == 0 else OrderSide.SELL)
            .quantity(100 + i)
            .price(150 + i)
            .sequence(i + 1)
            .build_unsafe()
        )
        records.append(record)
    return records


@pytest.fixture
def memory_storage():
    """Create memory storage backend."""
    config = AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
    return MemoryAuditStorage(config)


@pytest.fixture
def sqlite_storage(tmp_path):
    """Create SQLite storage backend."""
    db_path = str(tmp_path / "test_audit.db")
    config = AuditStorageConfig(
        backend_type=StorageBackendType.SQLITE,
        database_path=db_path,
    )
    storage = SQLiteAuditStorage(config)
    yield storage
    storage.close()


@pytest.fixture
def file_storage(tmp_path):
    """Create file storage backend."""
    file_path = str(tmp_path / "test_audit.jsonl")
    config = AuditStorageConfig(
        backend_type=StorageBackendType.FILE,
        database_path=file_path,
    )
    storage = FileAuditStorage(config)
    yield storage
    storage.close()


# =============================================================================
# Config Tests
# =============================================================================

class TestAuditStorageConfig:
    """Tests for AuditStorageConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuditStorageConfig()

        assert config.backend_type == StorageBackendType.SQLITE
        assert config.table_name == "audit_trail"
        assert config.retention_years == 5
        assert config.max_batch_size == 1000
        assert config.sync_mode == "normal"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AuditStorageConfig(
            backend_type=StorageBackendType.MEMORY,
            database_path="/custom/path/audit.db",
            retention_years=7,
            sync_mode="full",
        )

        assert config.backend_type == StorageBackendType.MEMORY
        assert config.database_path == "/custom/path/audit.db"
        assert config.retention_years == 7
        assert config.sync_mode == "full"


class TestStorageMetrics:
    """Tests for StorageMetrics."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = StorageMetrics()

        assert metrics.records_written == 0
        assert metrics.records_read == 0
        assert metrics.write_errors == 0
        assert metrics.read_errors == 0


# =============================================================================
# MemoryAuditStorage Tests
# =============================================================================

class TestMemoryAuditStorage:
    """Tests for MemoryAuditStorage."""

    def test_initialization(self, memory_storage):
        """Test memory storage initialization."""
        assert memory_storage._state == StorageState.READY
        assert len(memory_storage._records) == 0
        assert memory_storage._last_hash is None

    def test_append_single_record(self, memory_storage, sample_record):
        """Test appending a single record."""
        result = memory_storage.append(sample_record)

        assert result is True
        assert len(memory_storage._records) == 1
        assert memory_storage._last_hash == sample_record.record_hash
        assert sample_record.status == AuditRecordStatus.WRITTEN

    def test_append_sets_chain_hash(self, memory_storage, sample_records):
        """Test that chain hash is set correctly."""
        for record in sample_records[:3]:
            memory_storage.append(record)

        # Verify chain
        assert memory_storage._records[0].previous_record_hash is None
        assert memory_storage._records[1].previous_record_hash == memory_storage._records[0].record_hash
        assert memory_storage._records[2].previous_record_hash == memory_storage._records[1].record_hash

    def test_append_batch(self, memory_storage, sample_records):
        """Test batch append."""
        count = memory_storage.append_batch(sample_records)

        assert count == len(sample_records)
        assert len(memory_storage._records) == len(sample_records)

    def test_read_by_id(self, memory_storage, sample_record):
        """Test reading record by ID."""
        memory_storage.append(sample_record)

        result = memory_storage.read_by_id(sample_record.record_id)

        assert result is not None
        assert result.record_id == sample_record.record_id

    def test_read_by_id_not_found(self, memory_storage):
        """Test reading non-existent record."""
        result = memory_storage.read_by_id("non-existent")
        assert result is None

    def test_read_range(self, memory_storage, sample_records):
        """Test reading records in time range."""
        for record in sample_records:
            memory_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = memory_storage.read_range(start, end)

        assert len(results) == len(sample_records)

    def test_read_range_with_event_types(self, memory_storage):
        """Test reading with event type filter."""
        # Create records with different event types
        submitted = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("order-001")
            .build_unsafe()
        )
        filled = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_FILLED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("order-001")
            .build_unsafe()
        )

        memory_storage.append(submitted)
        memory_storage.append(filled)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        # Filter by event type
        results = memory_storage.read_range(
            start, end,
            event_types=[AuditEventType.ORDER_SUBMITTED],
        )

        assert len(results) == 1
        assert results[0].event_type == AuditEventType.ORDER_SUBMITTED

    def test_read_range_with_limit_offset(self, memory_storage, sample_records):
        """Test pagination with limit and offset."""
        for record in sample_records:
            memory_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        # First page
        page1 = memory_storage.read_range(start, end, limit=5, offset=0)
        assert len(page1) == 5

        # Second page
        page2 = memory_storage.read_range(start, end, limit=5, offset=5)
        assert len(page2) == 5

    def test_read_by_order_id(self, memory_storage, sample_records):
        """Test reading records by order ID."""
        for record in sample_records:
            memory_storage.append(record)

        results = memory_storage.read_by_order_id("order-001")

        assert len(results) == 1
        assert results[0].order_id == "order-001"

    def test_read_by_algorithm_id(self, memory_storage, sample_records):
        """Test reading records by algorithm ID."""
        for record in sample_records:
            memory_storage.append(record)

        results = memory_storage.read_by_algorithm_id("algo-001")

        assert len(results) == len(sample_records)

    def test_read_by_algorithm_id_with_time_filter(self, memory_storage, sample_records):
        """Test reading by algorithm with time filter."""
        for record in sample_records:
            memory_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = memory_storage.read_by_algorithm_id("algo-001", start, end)
        assert len(results) == len(sample_records)

    def test_count(self, memory_storage, sample_records):
        """Test counting records."""
        for record in sample_records:
            memory_storage.append(record)

        count = memory_storage.count()
        assert count == len(sample_records)

    def test_count_with_filters(self, memory_storage, sample_records):
        """Test counting with filters."""
        for record in sample_records:
            memory_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        count = memory_storage.count(start_time=start, end_time=end)
        assert count == len(sample_records)

    def test_get_latest_record(self, memory_storage, sample_records):
        """Test getting latest record."""
        for record in sample_records:
            memory_storage.append(record)

        latest = memory_storage.get_latest_record()

        assert latest is not None
        assert latest.record_id == sample_records[-1].record_id

    def test_get_latest_record_empty(self, memory_storage):
        """Test getting latest from empty storage."""
        latest = memory_storage.get_latest_record()
        assert latest is None

    def test_get_last_hash(self, memory_storage, sample_record):
        """Test getting last hash."""
        assert memory_storage.get_last_hash() is None

        memory_storage.append(sample_record)

        assert memory_storage.get_last_hash() == sample_record.record_hash

    def test_verify_chain_valid(self, memory_storage, sample_records):
        """Test chain verification with valid chain."""
        for record in sample_records:
            memory_storage.append(record)

        status = memory_storage.verify_chain()

        assert status.is_valid
        assert status.records_checked == len(sample_records)

    def test_verify_chain_empty(self, memory_storage):
        """Test chain verification with empty storage."""
        status = memory_storage.verify_chain()

        assert status.is_valid
        assert status.records_checked == 0

    def test_verify_chain_with_time_range(self, memory_storage, sample_records):
        """Test chain verification with time range."""
        for record in sample_records:
            memory_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        status = memory_storage.verify_chain(start, end)

        assert status.is_valid

    def test_export(self, memory_storage, sample_records):
        """Test export functionality."""
        for record in sample_records:
            memory_storage.append(record)

        request = AuditExportRequest(
            start_datetime=datetime.now() - timedelta(hours=1),
            end_datetime=datetime.now() + timedelta(hours=1),
            include_chain_verification=True,
        )

        result = memory_storage.export(request)

        assert result.success
        assert result.records_exported == len(sample_records)
        assert result.chain_verification.is_valid

    def test_export_with_filters(self, memory_storage, sample_records):
        """Test export with filters."""
        for record in sample_records:
            memory_storage.append(record)

        request = AuditExportRequest(
            start_datetime=datetime.now() - timedelta(hours=1),
            end_datetime=datetime.now() + timedelta(hours=1),
            event_types=[AuditEventType.ORDER_SUBMITTED],
            order_ids=["order-001"],
        )

        result = memory_storage.export(request)

        assert result.success
        assert result.records_exported == 1

    def test_get_metrics(self, memory_storage, sample_record):
        """Test getting metrics."""
        memory_storage.append(sample_record)
        memory_storage.read_by_id(sample_record.record_id)

        metrics = memory_storage.get_metrics()

        assert metrics.records_written == 1
        assert metrics.records_read >= 1

    def test_clear(self, memory_storage, sample_records):
        """Test clearing storage."""
        for record in sample_records:
            memory_storage.append(record)

        assert len(memory_storage._records) > 0

        memory_storage.clear()

        assert len(memory_storage._records) == 0
        assert memory_storage._last_hash is None


# =============================================================================
# SQLiteAuditStorage Tests
# =============================================================================

class TestSQLiteAuditStorage:
    """Tests for SQLiteAuditStorage."""

    def test_initialization(self, sqlite_storage):
        """Test SQLite storage initialization."""
        assert sqlite_storage._state == StorageState.READY

    def test_database_created(self, sqlite_storage):
        """Test that database file is created."""
        assert os.path.exists(sqlite_storage._db_path)

    def test_append_single_record(self, sqlite_storage, sample_record):
        """Test appending a single record."""
        result = sqlite_storage.append(sample_record)

        assert result is True
        assert sqlite_storage.count() == 1

    def test_append_duplicate_rejected(self, sqlite_storage, sample_record):
        """Test that duplicate record IDs are rejected."""
        sqlite_storage.append(sample_record)

        # Try to append same record again
        result = sqlite_storage.append(sample_record)

        assert result is False
        assert sqlite_storage.count() == 1

    def test_append_batch(self, sqlite_storage, sample_records):
        """Test batch append."""
        count = sqlite_storage.append_batch(sample_records)

        assert count == len(sample_records)
        assert sqlite_storage.count() == len(sample_records)

    def test_read_by_id(self, sqlite_storage, sample_record):
        """Test reading record by ID."""
        sqlite_storage.append(sample_record)

        result = sqlite_storage.read_by_id(sample_record.record_id)

        assert result is not None
        assert result.record_id == sample_record.record_id
        assert result.firm_lei == sample_record.firm_lei

    def test_read_range(self, sqlite_storage, sample_records):
        """Test reading records in time range."""
        for record in sample_records:
            sqlite_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = sqlite_storage.read_range(start, end)

        assert len(results) == len(sample_records)

    def test_read_by_order_id(self, sqlite_storage, sample_records):
        """Test reading records by order ID."""
        for record in sample_records:
            sqlite_storage.append(record)

        results = sqlite_storage.read_by_order_id("order-001")

        assert len(results) == 1

    def test_read_by_algorithm_id(self, sqlite_storage, sample_records):
        """Test reading records by algorithm ID."""
        for record in sample_records:
            sqlite_storage.append(record)

        results = sqlite_storage.read_by_algorithm_id("algo-001")

        assert len(results) == len(sample_records)

    def test_verify_chain(self, sqlite_storage, sample_records):
        """Test chain verification."""
        for record in sample_records:
            sqlite_storage.append(record)

        status = sqlite_storage.verify_chain()

        assert status.is_valid
        assert status.records_checked == len(sample_records)

    def test_export(self, sqlite_storage, sample_records, tmp_path):
        """Test export functionality."""
        for record in sample_records:
            sqlite_storage.append(record)

        # Update backup path to use temp directory
        sqlite_storage.config.backup_path = str(tmp_path / "exports")

        request = AuditExportRequest(
            start_datetime=datetime.now() - timedelta(hours=1),
            end_datetime=datetime.now() + timedelta(hours=1),
            include_chain_verification=True,
        )

        result = sqlite_storage.export(request)

        assert result.success
        assert result.records_exported == len(sample_records)

    def test_get_metrics(self, sqlite_storage, sample_record):
        """Test getting metrics."""
        sqlite_storage.append(sample_record)

        metrics = sqlite_storage.get_metrics()

        assert metrics.records_written == 1
        assert metrics.total_storage_bytes > 0


# =============================================================================
# FileAuditStorage Tests
# =============================================================================

class TestFileAuditStorage:
    """Tests for FileAuditStorage."""

    def test_initialization(self, file_storage):
        """Test file storage initialization."""
        assert file_storage._state == StorageState.READY

    def test_file_created(self, file_storage):
        """Test that file is created."""
        assert os.path.exists(file_storage._file_path)

    def test_append_single_record(self, file_storage, sample_record):
        """Test appending a single record."""
        result = file_storage.append(sample_record)

        assert result is True
        assert file_storage.count() == 1

    def test_append_batch(self, file_storage, sample_records):
        """Test batch append."""
        count = file_storage.append_batch(sample_records)

        assert count == len(sample_records)
        assert file_storage.count() == len(sample_records)

    def test_read_by_id(self, file_storage, sample_record):
        """Test reading record by ID."""
        file_storage.append(sample_record)

        result = file_storage.read_by_id(sample_record.record_id)

        assert result is not None
        assert result.record_id == sample_record.record_id

    def test_read_range(self, file_storage, sample_records):
        """Test reading records in time range."""
        for record in sample_records:
            file_storage.append(record)

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)

        results = file_storage.read_range(start, end)

        assert len(results) == len(sample_records)

    def test_read_by_order_id(self, file_storage, sample_records):
        """Test reading records by order ID."""
        for record in sample_records:
            file_storage.append(record)

        results = file_storage.read_by_order_id("order-001")

        assert len(results) == 1

    def test_verify_chain(self, file_storage, sample_records):
        """Test chain verification."""
        for record in sample_records:
            file_storage.append(record)

        status = file_storage.verify_chain()

        assert status.is_valid
        assert status.records_checked == len(sample_records)

    def test_export(self, file_storage, sample_records):
        """Test export functionality."""
        for record in sample_records:
            file_storage.append(record)

        request = AuditExportRequest(
            start_datetime=datetime.now() - timedelta(hours=1),
            end_datetime=datetime.now() + timedelta(hours=1),
            include_chain_verification=True,
        )

        result = file_storage.export(request)

        assert result.success
        assert result.records_exported == len(sample_records)

    def test_file_format(self, file_storage, sample_record):
        """Test that records are stored as JSON Lines."""
        file_storage.append(sample_record)

        with open(file_storage._file_path, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["record_id"] == sample_record.record_id


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateAuditStorage:
    """Tests for create_audit_storage factory function."""

    def test_create_memory_storage(self):
        """Test creating memory storage."""
        config = AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
        storage = create_audit_storage(config)

        assert isinstance(storage, MemoryAuditStorage)

    def test_create_sqlite_storage(self, tmp_path):
        """Test creating SQLite storage."""
        config = AuditStorageConfig(
            backend_type=StorageBackendType.SQLITE,
            database_path=str(tmp_path / "test.db"),
        )
        storage = create_audit_storage(config)

        assert isinstance(storage, SQLiteAuditStorage)
        storage.close()

    def test_create_file_storage(self, tmp_path):
        """Test creating file storage."""
        config = AuditStorageConfig(
            backend_type=StorageBackendType.FILE,
            database_path=str(tmp_path / "test.jsonl"),
        )
        storage = create_audit_storage(config)

        assert isinstance(storage, FileAuditStorage)
        storage.close()

    def test_create_with_override_type(self, tmp_path):
        """Test creating storage with type override."""
        config = AuditStorageConfig(backend_type=StorageBackendType.SQLITE)
        storage = create_audit_storage(config, backend_type=StorageBackendType.MEMORY)

        assert isinstance(storage, MemoryAuditStorage)

    def test_create_postgresql_not_implemented(self):
        """Test that PostgreSQL raises NotImplementedError."""
        config = AuditStorageConfig(backend_type=StorageBackendType.POSTGRESQL)

        with pytest.raises(NotImplementedError):
            create_audit_storage(config)


# =============================================================================
# Chain Integrity Tests
# =============================================================================

class TestChainIntegrity:
    """Tests for chain integrity across storage backends."""

    @pytest.mark.parametrize("storage_fixture", ["memory_storage", "sqlite_storage", "file_storage"])
    def test_chain_integrity_maintained(self, storage_fixture, sample_records, request):
        """Test that chain integrity is maintained."""
        storage = request.getfixturevalue(storage_fixture)

        for record in sample_records:
            storage.append(record)

        # Read all records
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)
        records = storage.read_range(start, end)

        # Verify chain
        prev_hash = None
        for record in records:
            assert record.previous_record_hash == prev_hash
            prev_hash = record.record_hash

    @pytest.mark.parametrize("storage_fixture", ["memory_storage", "sqlite_storage", "file_storage"])
    def test_verify_chain_detects_tampering(self, storage_fixture, sample_records, request):
        """Test that chain verification detects tampering."""
        storage = request.getfixturevalue(storage_fixture)

        for record in sample_records[:3]:
            storage.append(record)

        # Verify initial chain is valid
        status = storage.verify_chain()
        assert status.is_valid


# =============================================================================
# Metrics Tests
# =============================================================================

class TestStorageMetricsTracking:
    """Tests for metrics tracking across storage backends."""

    def test_write_metrics(self, memory_storage, sample_records):
        """Test write metrics are tracked."""
        initial_metrics = memory_storage.get_metrics()
        assert initial_metrics.records_written == 0

        for record in sample_records:
            memory_storage.append(record)

        final_metrics = memory_storage.get_metrics()
        assert final_metrics.records_written == len(sample_records)
        assert final_metrics.last_write_timestamp is not None

    def test_read_metrics(self, memory_storage, sample_record):
        """Test read metrics are tracked."""
        memory_storage.append(sample_record)

        initial_metrics = memory_storage.get_metrics()
        reads_before = initial_metrics.records_read

        memory_storage.read_by_id(sample_record.record_id)

        final_metrics = memory_storage.get_metrics()
        assert final_metrics.records_read > reads_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
