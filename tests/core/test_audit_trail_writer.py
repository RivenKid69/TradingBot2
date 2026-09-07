# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 4 - Audit Trail Writer.

Tests cover the audit trail writer with chain integrity verification
per MiFIR Article 25 requirements.

Coverage target: 100%
"""

import os
import pytest
import tempfile
import threading
import time
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, call

from services.core.risk_controls.audit_trail_writer import (
    WriterMode,
    WriterState,
    AuditTrailWriterConfig,
    WriterMetrics,
    AuditTrailWriter,
    create_audit_trail_writer,
)
from services.core.risk_controls.audit_models import (
    AuditRecord,
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
    OrderSide,
    AuditRecordBuilder,
)
from services.core.risk_controls.audit_storage import (
    AuditStorageConfig,
    StorageBackendType,
    StorageMetrics,
    MemoryAuditStorage,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_lei():
    """Sample LEI for testing."""
    return "5493001KJTIIGC8Y1R12"


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_audit.db")


@pytest.fixture
def memory_storage():
    """Create memory storage backend."""
    config = AuditStorageConfig(backend_type=StorageBackendType.MEMORY)
    return MemoryAuditStorage(config)


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = Mock()
    storage.get_last_hash.return_value = None
    storage.get_latest_record.return_value = None
    storage.append.return_value = True
    storage.append_batch.return_value = 0
    storage.read_by_id.return_value = None
    storage.verify_chain.return_value = AuditChainStatus(is_valid=True, records_checked=0)
    storage.export.return_value = AuditExportResult(
        request_id="test-request", success=True, records_exported=0
    )
    storage.get_metrics.return_value = StorageMetrics()
    storage.close.return_value = None
    return storage


@pytest.fixture
def sync_config(sample_lei):
    """Create sync mode config."""
    return AuditTrailWriterConfig(
        firm_lei=sample_lei,
        mode=WriterMode.SYNC,
        storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
        enable_retention_management=False,
    )


@pytest.fixture
def async_config(sample_lei):
    """Create async mode config."""
    return AuditTrailWriterConfig(
        firm_lei=sample_lei,
        mode=WriterMode.ASYNC,
        storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
        enable_retention_management=False,
        queue_size=100,
    )


@pytest.fixture
def batch_config(sample_lei):
    """Create batch mode config."""
    return AuditTrailWriterConfig(
        firm_lei=sample_lei,
        mode=WriterMode.BATCH,
        storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
        enable_retention_management=False,
        batch_size=10,
        batch_interval_ms=100,
        queue_size=100,
    )


# =============================================================================
# Test WriterMode Enum
# =============================================================================


class TestWriterMode:
    """Tests for WriterMode enum."""

    def test_writer_mode_values(self):
        """Test all writer mode values exist."""
        assert WriterMode.SYNC.value == "sync"
        assert WriterMode.ASYNC.value == "async"
        assert WriterMode.BATCH.value == "batch"

    def test_writer_mode_count(self):
        """Test writer mode has expected number of values."""
        assert len(WriterMode) == 3


# =============================================================================
# Test WriterState Enum
# =============================================================================


class TestWriterState:
    """Tests for WriterState enum."""

    def test_writer_state_values(self):
        """Test all writer state values exist."""
        assert WriterState.INITIALIZING.value == "initializing"
        assert WriterState.RUNNING.value == "running"
        assert WriterState.PAUSED.value == "paused"
        assert WriterState.STOPPING.value == "stopping"
        assert WriterState.STOPPED.value == "stopped"
        assert WriterState.ERROR.value == "error"

    def test_writer_state_count(self):
        """Test writer state has expected number of values."""
        assert len(WriterState) == 6


# =============================================================================
# Test AuditTrailWriterConfig
# =============================================================================


class TestAuditTrailWriterConfig:
    """Tests for AuditTrailWriterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AuditTrailWriterConfig()
        assert config.firm_lei == ""
        assert config.mode == WriterMode.ASYNC
        assert config.storage_config is None
        assert config.retention_config is None
        assert config.batch_size == 100
        assert config.batch_interval_ms == 1000
        assert config.queue_size == 10000
        assert config.verify_on_write is False
        assert config.auto_sequence is True
        assert config.enable_retention_management is True
        assert config.on_write_error is None
        assert config.on_chain_break is None

    def test_custom_config(self, sample_lei):
        """Test custom configuration values."""
        storage_config = AuditStorageConfig()

        def on_error(record, exc):
            pass

        def on_break(status):
            pass

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.BATCH,
            storage_config=storage_config,
            batch_size=50,
            batch_interval_ms=500,
            queue_size=5000,
            verify_on_write=True,
            auto_sequence=False,
            enable_retention_management=False,
            on_write_error=on_error,
            on_chain_break=on_break,
        )

        assert config.firm_lei == sample_lei
        assert config.mode == WriterMode.BATCH
        assert config.storage_config == storage_config
        assert config.batch_size == 50
        assert config.batch_interval_ms == 500
        assert config.queue_size == 5000
        assert config.verify_on_write is True
        assert config.auto_sequence is False
        assert config.enable_retention_management is False
        assert config.on_write_error == on_error
        assert config.on_chain_break == on_break


# =============================================================================
# Test WriterMetrics
# =============================================================================


class TestWriterMetrics:
    """Tests for WriterMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = WriterMetrics()
        assert metrics.records_written == 0
        assert metrics.records_queued == 0
        assert metrics.records_failed == 0
        assert metrics.batches_written == 0
        assert metrics.chain_verifications == 0
        assert metrics.chain_failures == 0
        assert metrics.last_write_timestamp_ns is None
        assert metrics.last_sequence_number == 0
        assert metrics.queue_depth == 0
        assert metrics.write_latency_ms == 0.0
        assert metrics.avg_batch_size == 0.0

    def test_updated_metrics(self):
        """Test updated metrics values."""
        metrics = WriterMetrics(
            records_written=100,
            records_queued=50,
            records_failed=5,
            batches_written=10,
            chain_verifications=20,
            chain_failures=1,
            last_write_timestamp_ns=1234567890,
            last_sequence_number=100,
            queue_depth=25,
            write_latency_ms=1.5,
            avg_batch_size=10.0,
        )

        assert metrics.records_written == 100
        assert metrics.records_queued == 50
        assert metrics.records_failed == 5
        assert metrics.batches_written == 10
        assert metrics.chain_verifications == 20
        assert metrics.chain_failures == 1


# =============================================================================
# Test AuditTrailWriter Initialization
# =============================================================================


class TestAuditTrailWriterInit:
    """Tests for AuditTrailWriter initialization."""

    def test_init_requires_firm_lei(self):
        """Test that firm_lei is required."""
        config = AuditTrailWriterConfig(firm_lei="")
        with pytest.raises(ValueError, match="firm_lei is required"):
            AuditTrailWriter(config)

    def test_init_with_storage(self, sample_lei, mock_storage):
        """Test initialization with provided storage."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, enable_retention_management=False
        )
        writer = AuditTrailWriter(config, storage=mock_storage)

        assert writer.state == WriterState.STOPPED
        assert writer._storage == mock_storage
        mock_storage.get_last_hash.assert_called_once()
        mock_storage.get_latest_record.assert_called_once()

    def test_init_creates_storage(self, sync_config):
        """Test initialization creates storage if not provided."""
        writer = AuditTrailWriter(sync_config)
        assert writer._storage is not None
        assert writer.state == WriterState.STOPPED

    def test_init_with_existing_records(self, sample_lei):
        """Test initialization with existing records in storage."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = "abc123hash"

        # Create a mock record with sequence number
        mock_record = Mock()
        mock_record.sequence_number = 42
        mock_storage.get_latest_record.return_value = mock_record

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, enable_retention_management=False
        )
        writer = AuditTrailWriter(config, storage=mock_storage)

        assert writer._last_hash == "abc123hash"
        assert writer._sequence_number == 42

    def test_init_with_retention_management(self, sample_lei, mock_storage):
        """Test initialization with retention management enabled."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, enable_retention_management=True
        )

        with patch(
            "services.core.risk_controls.audit_trail_writer.create_retention_manager"
        ) as mock_create:
            mock_retention = Mock()
            mock_create.return_value = mock_retention

            writer = AuditTrailWriter(config, storage=mock_storage)

            mock_create.assert_called_once()
            assert writer._retention_manager is not None

    def test_init_error_handling(self, sample_lei):
        """Test initialization error handling."""
        mock_storage = Mock()
        mock_storage.get_last_hash.side_effect = RuntimeError("Storage error")

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, enable_retention_management=False
        )

        with pytest.raises(RuntimeError):
            AuditTrailWriter(config, storage=mock_storage)


# =============================================================================
# Test AuditTrailWriter Start/Stop
# =============================================================================


class TestAuditTrailWriterStartStop:
    """Tests for AuditTrailWriter start/stop methods."""

    def test_start_sync_mode(self, sync_config):
        """Test starting writer in sync mode."""
        writer = AuditTrailWriter(sync_config)
        assert writer.state == WriterState.STOPPED

        writer.start()
        assert writer.state == WriterState.RUNNING
        assert writer._worker_thread is None

        writer.stop()

    def test_start_async_mode(self, async_config):
        """Test starting writer in async mode."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        assert writer.state == WriterState.RUNNING
        assert writer._worker_thread is not None
        assert writer._worker_thread.is_alive()

        writer.stop()
        assert writer.state == WriterState.STOPPED

    def test_start_batch_mode(self, batch_config):
        """Test starting writer in batch mode."""
        writer = AuditTrailWriter(batch_config)
        writer.start()

        assert writer.state == WriterState.RUNNING
        assert writer._worker_thread is not None
        assert writer._worker_thread.is_alive()

        writer.stop()
        assert writer.state == WriterState.STOPPED

    def test_start_idempotent(self, sync_config):
        """Test that start is idempotent."""
        writer = AuditTrailWriter(sync_config)
        writer.start()
        writer.start()  # Second call should be no-op
        assert writer.state == WriterState.RUNNING
        writer.stop()

    def test_stop_idempotent(self, sync_config):
        """Test that stop is idempotent."""
        writer = AuditTrailWriter(sync_config)
        writer.start()
        writer.stop()
        writer.stop()  # Second call should be no-op
        assert writer.state == WriterState.STOPPED

    def test_stop_with_flush(self, async_config):
        """Test stop with flush."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        # Add some records
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(async_config.firm_lei)
            .build_unsafe()
        )
        writer.write(record)

        writer.stop(flush=True)
        assert writer.state == WriterState.STOPPED

    def test_stop_without_flush(self, async_config):
        """Test stop without flush."""
        writer = AuditTrailWriter(async_config)
        writer.start()
        writer.stop(flush=False, timeout=1.0)
        assert writer.state == WriterState.STOPPED

    def test_start_with_retention_manager(self, sample_lei, mock_storage):
        """Test start with retention manager."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, enable_retention_management=True, mode=WriterMode.SYNC
        )

        with patch(
            "services.core.risk_controls.audit_trail_writer.create_retention_manager"
        ) as mock_create:
            mock_retention = Mock()
            mock_create.return_value = mock_retention

            writer = AuditTrailWriter(config, storage=mock_storage)
            writer.start()

            mock_retention.start.assert_called_once()

            writer.stop()
            mock_retention.stop.assert_called_once()


# =============================================================================
# Test AuditTrailWriter Sync Mode Writing
# =============================================================================


class TestAuditTrailWriterSyncWrite:
    """Tests for AuditTrailWriter sync write operations."""

    def test_write_requires_running(self, sync_config):
        """Test write requires running state."""
        writer = AuditTrailWriter(sync_config)
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sync_config.firm_lei)
            .build_unsafe()
        )

        with pytest.raises(RuntimeError, match="not running"):
            writer.write(record)

    def test_write_sync_success(self, sync_config):
        """Test successful sync write."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sync_config.firm_lei)
            .build_unsafe()
        )

        result = writer.write(record)
        assert result is True

        metrics = writer.get_metrics()
        assert metrics.records_written == 1
        assert metrics.last_sequence_number == 1

        writer.stop()

    def test_write_sets_firm_lei(self, sync_config):
        """Test write sets firm_lei if not set."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .build_unsafe()
        )
        record.firm_lei = ""

        writer.write(record)
        assert record.firm_lei == sync_config.firm_lei

        writer.stop()

    def test_write_auto_sequence(self, sync_config):
        """Test write auto-increments sequence number."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        for i in range(5):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(sync_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)
            assert record.sequence_number == i + 1

        writer.stop()

    def test_write_chain_hash(self, sync_config):
        """Test write maintains hash chain."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        hashes = []
        for i in range(3):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(sync_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)
            hashes.append(record.record_hash)

            if i > 0:
                assert record.previous_record_hash == hashes[i - 1]

        writer.stop()

    def test_write_sync_failure(self, sample_lei):
        """Test sync write failure handling."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.return_value = False

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, mode=WriterMode.SYNC, enable_retention_management=False
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        result = writer.write(record)
        assert result is False

        metrics = writer.get_metrics()
        assert metrics.records_failed == 1

        writer.stop(flush=False)

    def test_write_with_error_callback(self, sample_lei):
        """Test write error callback."""
        errors = []

        def on_error(record, exc):
            errors.append((record, exc))

        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.return_value = False
        mock_storage.close.return_value = None

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            enable_retention_management=False,
            on_write_error=on_error,
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        writer.write(record)
        assert len(errors) == 1
        assert errors[0][0] == record

        writer.stop(flush=False)

    def test_write_with_exception(self, sample_lei):
        """Test write exception handling."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.side_effect = RuntimeError("Storage error")
        mock_storage.close.return_value = None

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei, mode=WriterMode.SYNC, enable_retention_management=False
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        result = writer.write(record)
        assert result is False

        metrics = writer.get_metrics()
        assert metrics.records_failed == 1

        writer.stop(flush=False)

    def test_write_with_verify_on_write(self, sample_lei):
        """Test write with verification enabled."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.return_value = True
        mock_storage.close.return_value = None

        # Mock read_by_id to return the record with matching hash
        def read_by_id_side_effect(record_id):
            mock_record = Mock()
            mock_record.record_hash = "matching_hash"
            return mock_record

        mock_storage.read_by_id.side_effect = read_by_id_side_effect

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            enable_retention_management=False,
            verify_on_write=True,
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        # Note: hash won't match since we're using mock
        writer.write(record)

        metrics = writer.get_metrics()
        assert metrics.chain_verifications == 1

        writer.stop(flush=False)

    def test_write_verify_record_not_found(self, sample_lei):
        """Test write verification when record not found."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.return_value = True
        mock_storage.read_by_id.return_value = None
        mock_storage.close.return_value = None

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            enable_retention_management=False,
            verify_on_write=True,
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        writer.write(record)

        metrics = writer.get_metrics()
        assert metrics.chain_failures == 1

        writer.stop(flush=False)

    def test_write_verify_hash_mismatch(self, sample_lei):
        """Test write verification with hash mismatch."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.return_value = True
        mock_storage.close.return_value = None

        # Return record with different hash
        mock_stored = Mock()
        mock_stored.record_hash = "different_hash"
        mock_storage.read_by_id.return_value = mock_stored

        chain_breaks = []

        def on_chain_break(status):
            chain_breaks.append(status)

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            enable_retention_management=False,
            verify_on_write=True,
            on_chain_break=on_chain_break,
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        writer.write(record)

        metrics = writer.get_metrics()
        assert metrics.chain_failures == 1
        assert len(chain_breaks) == 1
        assert chain_breaks[0].is_valid is False

        writer.stop(flush=False)


# =============================================================================
# Test AuditTrailWriter Async Mode Writing
# =============================================================================


class TestAuditTrailWriterAsyncWrite:
    """Tests for AuditTrailWriter async write operations."""

    def test_write_async_queues_record(self, async_config):
        """Test async write queues record."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(async_config.firm_lei)
            .build_unsafe()
        )

        result = writer.write(record)
        assert result is True

        metrics = writer.get_metrics()
        assert metrics.records_queued == 1

        # Wait for async write
        time.sleep(0.2)

        metrics = writer.get_metrics()
        assert metrics.records_written == 1

        writer.stop()

    def test_write_async_queue_full(self, sample_lei):
        """Test async write when queue is full."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.ASYNC,
            storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
            enable_retention_management=False,
            queue_size=1,
        )

        writer = AuditTrailWriter(config)
        # Don't start worker - queue will fill up
        writer._state = WriterState.RUNNING

        record1 = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )
        record2 = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )

        # First should succeed
        result1 = writer.write(record1)
        assert result1 is True

        # Second should fail (queue full)
        result2 = writer.write(record2)
        assert result2 is False

        metrics = writer.get_metrics()
        assert metrics.records_failed == 1

    def test_write_async_multiple_records(self, async_config):
        """Test async write multiple records."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        for i in range(10):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(async_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)

        # Wait for all writes
        writer.flush()

        metrics = writer.get_metrics()
        assert metrics.records_written == 10

        writer.stop()


# =============================================================================
# Test AuditTrailWriter Batch Mode Writing
# =============================================================================


class TestAuditTrailWriterBatchWrite:
    """Tests for AuditTrailWriter batch write operations."""

    def test_write_batch_accumulates(self, batch_config):
        """Test batch write accumulates records."""
        writer = AuditTrailWriter(batch_config)
        writer.start()

        # Write fewer than batch_size records
        for i in range(5):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(batch_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)

        # Records should be queued but not written yet
        metrics = writer.get_metrics()
        assert metrics.records_queued == 5

        # Flush to write
        writer.flush()

        metrics = writer.get_metrics()
        assert metrics.batches_written >= 1

        writer.stop()

    def test_write_batch_auto_flush_on_size(self, batch_config):
        """Test batch auto-flushes when reaching batch_size."""
        writer = AuditTrailWriter(batch_config)
        writer.start()

        # Write batch_size records
        for i in range(batch_config.batch_size):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(batch_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)

        # Wait for batch to flush
        time.sleep(0.3)

        metrics = writer.get_metrics()
        assert metrics.batches_written >= 1

        writer.stop()

    def test_write_batch_auto_flush_on_interval(self, sample_lei):
        """Test batch auto-flushes on interval."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.BATCH,
            storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
            enable_retention_management=False,
            batch_size=1000,  # Large batch size
            batch_interval_ms=50,  # Short interval
        )

        writer = AuditTrailWriter(config)
        writer.start()

        # Write a few records
        for i in range(3):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(sample_lei)
                .build_unsafe()
            )
            writer.write(record)

        # Wait for interval flush
        time.sleep(0.2)

        metrics = writer.get_metrics()
        assert metrics.batches_written >= 1

        writer.stop()


# =============================================================================
# Test AuditTrailWriter Flush
# =============================================================================


class TestAuditTrailWriterFlush:
    """Tests for AuditTrailWriter flush method."""

    def test_flush_empty_queue(self, async_config):
        """Test flush with empty queue."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        result = writer.flush(timeout=1.0)
        assert result is True

        writer.stop()

    def test_flush_timeout(self, sample_lei):
        """Test flush timeout."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.ASYNC,
            storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
            enable_retention_management=False,
            queue_size=100,
        )

        writer = AuditTrailWriter(config)
        # Set running but don't start worker
        writer._state = WriterState.RUNNING

        # Fill queue
        for i in range(10):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(sample_lei)
                .build_unsafe()
            )
            writer._queue.put_nowait(record)

        # Flush should timeout
        result = writer.flush(timeout=0.1)
        assert result is False

    def test_flush_batch_mode(self, batch_config):
        """Test flush in batch mode."""
        writer = AuditTrailWriter(batch_config)
        writer.start()

        # Add records to batch buffer
        for i in range(5):
            record = (
                AuditRecordBuilder()
                .event_type(AuditEventType.SYSTEM_STARTUP)
                .firm_lei(batch_config.firm_lei)
                .build_unsafe()
            )
            writer.write(record)

        # Flush
        result = writer.flush()
        assert result is True

        writer.stop()


# =============================================================================
# Test High-Level Write Methods
# =============================================================================


class TestHighLevelWriteMethods:
    """Tests for high-level write methods."""

    def test_write_order_submitted(self, sync_config):
        """Test write_order_submitted."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_submitted(
            order_id="ORD-12345",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=100,
            price=150.50,
            currency="USD",
            venue_mic="XNAS",
            algorithm_id="ALGO-001",
            trader_id="TRADER-001",
            client_order_id="CLIENT-12345",
            details={"strategy": "VWAP"},
        )

        assert result is True
        metrics = writer.get_metrics()
        assert metrics.records_written == 1

        writer.stop()

    def test_write_order_submitted_with_decimal(self, sync_config):
        """Test write_order_submitted with Decimal values."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_submitted(
            order_id="ORD-12346",
            instrument_isin="US0378331005",
            side=OrderSide.SELL,
            quantity=Decimal("100.5"),
            price=Decimal("150.75"),
        )

        assert result is True
        writer.stop()

    def test_write_order_filled(self, sync_config):
        """Test write_order_filled."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_filled(
            order_id="ORD-12345",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=50,
            fill_price=150.55,
            currency="USD",
            venue_mic="XNAS",
            is_partial=False,
            algorithm_id="ALGO-001",
            details={"execution_id": "EXEC-001"},
        )

        assert result is True
        writer.stop()

    def test_write_order_filled_partial(self, sync_config):
        """Test write_order_filled for partial fill."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_filled(
            order_id="ORD-12345",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=30,
            fill_price=150.55,
            is_partial=True,
            remaining_quantity=70,
        )

        assert result is True
        writer.stop()

    def test_write_order_cancelled(self, sync_config):
        """Test write_order_cancelled."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_cancelled(
            order_id="ORD-12345",
            reason="User requested cancellation",
            algorithm_id="ALGO-001",
            details={"source": "UI"},
        )

        assert result is True
        writer.stop()

    def test_write_order_cancelled_without_details(self, sync_config):
        """Test write_order_cancelled without optional parameters."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_cancelled(order_id="ORD-12345")

        assert result is True
        writer.stop()

    def test_write_order_rejected(self, sync_config):
        """Test write_order_rejected."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_rejected(
            order_id="ORD-12345",
            reason="Insufficient funds",
            rejection_code="INSUFF_FUNDS",
            algorithm_id="ALGO-001",
            details={"available_balance": 1000},
        )

        assert result is True
        writer.stop()

    def test_write_order_rejected_without_code(self, sync_config):
        """Test write_order_rejected without rejection code."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_order_rejected(
            order_id="ORD-12345", reason="Unknown error"
        )

        assert result is True
        writer.stop()

    def test_write_risk_event(self, sync_config):
        """Test write_risk_event."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_risk_event(
            event_type=AuditEventType.PRE_TRADE_CHECK_FAILED,
            reason="Price collar breach: price 200 exceeds limit 150",
            algorithm_id="ALGO-001",
            order_id="ORD-12345",
            details={"limit": 150, "actual": 200},
        )

        assert result is True
        writer.stop()

    def test_write_system_event(self, sync_config):
        """Test write_system_event."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_system_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            message="Trading system started",
            details={"version": "1.0.0"},
        )

        assert result is True
        writer.stop()

    def test_write_algorithm_event(self, sync_config):
        """Test write_algorithm_event."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_algorithm_event(
            event_type=AuditEventType.ALGO_STARTED,
            algorithm_id="ALGO-001",
            version="2.0.0",
            parameters={"aggression": 0.5, "max_participation": 10},
            details={"instruments": ["AAPL", "MSFT"]},
        )

        assert result is True
        writer.stop()

    def test_write_algorithm_event_minimal(self, sync_config):
        """Test write_algorithm_event with minimal parameters."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_algorithm_event(
            event_type=AuditEventType.ALGO_STOPPED, algorithm_id="ALGO-001"
        )

        assert result is True
        writer.stop()

    def test_write_kill_switch_event(self, sync_config):
        """Test write_kill_switch_event."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        result = writer.write_kill_switch_event(
            triggered_by="COMPLIANCE_OFFICER",
            reason="Position limit breach detected",
            scope="ALGORITHM",
            orders_cancelled=15,
            details={"algorithm_id": "ALGO-001", "position": 1500000},
        )

        assert result is True
        writer.stop()


# =============================================================================
# Test Query Methods
# =============================================================================


class TestQueryMethods:
    """Tests for query methods."""

    def test_verify_chain(self, sync_config):
        """Test verify_chain method."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        # Write some records
        for i in range(5):
            writer.write_system_event(
                AuditEventType.SYSTEM_STARTUP, f"Event {i}"
            )

        # Verify chain
        status = writer.verify_chain()
        assert status.is_valid is True

        writer.stop()

    def test_verify_chain_with_time_range(self, sync_config):
        """Test verify_chain with time range."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        # Write records
        writer.write_system_event(AuditEventType.SYSTEM_STARTUP, "Test")

        start_time = datetime(2020, 1, 1)
        end_time = datetime(2030, 12, 31)

        status = writer.verify_chain(start_time=start_time, end_time=end_time)
        assert isinstance(status, AuditChainStatus)

        writer.stop()

    def test_export(self, sync_config):
        """Test export method."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        # Write records
        writer.write_system_event(AuditEventType.SYSTEM_STARTUP, "Test")

        # Export
        request = AuditExportRequest(export_format="json")
        result = writer.export(request)

        assert isinstance(result, AuditExportResult)

        writer.stop()

    def test_get_metrics(self, sync_config):
        """Test get_metrics method."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        writer.write_system_event(AuditEventType.SYSTEM_STARTUP, "Test")

        metrics = writer.get_metrics()
        assert isinstance(metrics, WriterMetrics)
        assert metrics.records_written == 1

        writer.stop()

    def test_get_storage_metrics(self, sync_config):
        """Test get_storage_metrics method."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        metrics = writer.get_storage_metrics()
        assert isinstance(metrics, StorageMetrics)

        writer.stop()


# =============================================================================
# Test Properties
# =============================================================================


class TestProperties:
    """Tests for properties."""

    def test_state_property(self, sync_config):
        """Test state property."""
        writer = AuditTrailWriter(sync_config)
        assert writer.state == WriterState.STOPPED

        writer.start()
        assert writer.state == WriterState.RUNNING

        writer.stop()
        assert writer.state == WriterState.STOPPED

    def test_is_running_property(self, sync_config):
        """Test is_running property."""
        writer = AuditTrailWriter(sync_config)
        assert writer.is_running is False

        writer.start()
        assert writer.is_running is True

        writer.stop()
        assert writer.is_running is False


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateAuditTrailWriter:
    """Tests for create_audit_trail_writer factory function."""

    def test_create_default(self, sample_lei, temp_db_path):
        """Test create with default parameters."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei, storage_path=temp_db_path
        )

        assert writer.config.firm_lei == sample_lei
        assert writer.config.mode == WriterMode.ASYNC
        assert writer.is_running is True

        writer.stop()

    def test_create_sync_mode(self, sample_lei, temp_db_path):
        """Test create with sync mode."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            storage_path=temp_db_path,
            auto_start=True,
        )

        assert writer.config.mode == WriterMode.SYNC
        assert writer.is_running is True

        writer.stop()

    def test_create_batch_mode(self, sample_lei, temp_db_path):
        """Test create with batch mode."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei, mode=WriterMode.BATCH, storage_path=temp_db_path
        )

        assert writer.config.mode == WriterMode.BATCH
        assert writer.is_running is True

        writer.stop()

    def test_create_memory_storage(self, sample_lei):
        """Test create with memory storage."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei, storage_type=StorageBackendType.MEMORY
        )

        assert writer.is_running is True
        writer.stop()

    def test_create_without_retention(self, sample_lei, temp_db_path):
        """Test create without retention management."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei,
            storage_path=temp_db_path,
            enable_retention=False,
        )

        assert writer._retention_manager is None
        writer.stop()

    def test_create_no_auto_start(self, sample_lei, temp_db_path):
        """Test create without auto-start."""
        writer = create_audit_trail_writer(
            firm_lei=sample_lei, storage_path=temp_db_path, auto_start=False
        )

        assert writer.is_running is False
        assert writer.state == WriterState.STOPPED


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes_sync(self, sync_config):
        """Test concurrent writes in sync mode."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        errors = []
        written = []

        def write_records(thread_id: int, count: int):
            try:
                for i in range(count):
                    result = writer.write_system_event(
                        AuditEventType.SYSTEM_STARTUP, f"Thread {thread_id} Event {i}"
                    )
                    if result:
                        written.append((thread_id, i))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_records, args=(i, 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(written) == 100

        metrics = writer.get_metrics()
        assert metrics.records_written == 100

        writer.stop()

    def test_concurrent_writes_async(self, async_config):
        """Test concurrent writes in async mode."""
        writer = AuditTrailWriter(async_config)
        writer.start()

        errors = []

        def write_records(thread_id: int, count: int):
            try:
                for i in range(count):
                    writer.write_system_event(
                        AuditEventType.SYSTEM_STARTUP, f"Thread {thread_id} Event {i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_records, args=(i, 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Flush to ensure all records are written
        writer.flush(timeout=5.0)

        assert len(errors) == 0

        metrics = writer.get_metrics()
        assert metrics.records_queued == 100

        writer.stop()


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_write_without_auto_sequence(self, sample_lei):
        """Test write without auto-sequence."""
        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.SYNC,
            storage_config=AuditStorageConfig(backend_type=StorageBackendType.MEMORY),
            enable_retention_management=False,
            auto_sequence=False,
        )

        writer = AuditTrailWriter(config)
        writer.start()

        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )
        record.sequence_number = 999

        writer.write(record)

        # Sequence should remain as set
        assert record.sequence_number == 999

        writer.stop()

    def test_empty_batch_flush(self, batch_config):
        """Test flushing empty batch buffer."""
        writer = AuditTrailWriter(batch_config)
        writer._flush_batch()  # Should not raise

    def test_writer_worker_handles_errors(self, sample_lei):
        """Test worker thread handles errors gracefully."""
        mock_storage = Mock()
        mock_storage.get_last_hash.return_value = None
        mock_storage.get_latest_record.return_value = None
        mock_storage.append.side_effect = RuntimeError("Test error")
        mock_storage.close.return_value = None

        config = AuditTrailWriterConfig(
            firm_lei=sample_lei,
            mode=WriterMode.ASYNC,
            enable_retention_management=False,
        )

        writer = AuditTrailWriter(config, storage=mock_storage)
        writer.start()

        # Write record - should be queued
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei(sample_lei)
            .build_unsafe()
        )
        writer.write(record)

        # Wait for worker to process
        time.sleep(0.2)

        # Worker should continue running despite error
        assert writer._worker_thread.is_alive()

        writer.stop(flush=False, timeout=1.0)

    def test_latency_tracking(self, sync_config):
        """Test write latency tracking."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        # Write multiple records
        for i in range(10):
            writer.write_system_event(AuditEventType.SYSTEM_STARTUP, f"Event {i}")

        metrics = writer.get_metrics()
        assert metrics.write_latency_ms >= 0

        writer.stop()

    def test_avg_batch_size_tracking(self, batch_config):
        """Test average batch size tracking."""
        writer = AuditTrailWriter(batch_config)
        writer.start()

        # Write enough records to trigger batch flush
        for i in range(batch_config.batch_size):
            writer.write_system_event(AuditEventType.SYSTEM_STARTUP, f"Event {i}")

        # Wait for batch flush
        writer.flush()

        metrics = writer.get_metrics()
        assert metrics.batches_written >= 1

        writer.stop()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for audit trail writer."""

    def test_full_trading_scenario(self, sync_config):
        """Test full trading scenario audit trail."""
        writer = AuditTrailWriter(sync_config)
        writer.start()

        # System startup
        writer.write_system_event(
            AuditEventType.SYSTEM_STARTUP, "Trading system started"
        )

        # Algorithm started
        writer.write_algorithm_event(
            AuditEventType.ALGO_STARTED,
            algorithm_id="VWAP-001",
            version="1.0.0",
            parameters={"aggression": 0.5},
        )

        # Order submitted
        writer.write_order_submitted(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=1000,
            price=150.00,
            algorithm_id="VWAP-001",
        )

        # Partial fill
        writer.write_order_filled(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=500,
            fill_price=150.05,
            is_partial=True,
            remaining_quantity=500,
            algorithm_id="VWAP-001",
        )

        # Full fill
        writer.write_order_filled(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=500,
            fill_price=150.10,
            is_partial=False,
            algorithm_id="VWAP-001",
        )

        # Risk event
        writer.write_risk_event(
            AuditEventType.POSITION_LIMIT_WARNING,
            reason="Position approaching limit",
            algorithm_id="VWAP-001",
        )

        # Kill switch
        writer.write_kill_switch_event(
            triggered_by="RISK_SYSTEM",
            reason="Position limit exceeded",
            scope="ALGORITHM",
            orders_cancelled=5,
        )

        # Algorithm stopped
        writer.write_algorithm_event(
            AuditEventType.ALGO_STOPPED, algorithm_id="VWAP-001"
        )

        # System shutdown
        writer.write_system_event(
            AuditEventType.SYSTEM_SHUTDOWN, "Trading system stopped"
        )

        # Verify
        metrics = writer.get_metrics()
        assert metrics.records_written == 9

        # Verify chain integrity
        status = writer.verify_chain()
        assert status.is_valid is True

        writer.stop()

    def test_export_with_records(self, sync_config, temp_db_path):
        """Test export with actual records."""
        # Use SQLite for export test
        config = AuditTrailWriterConfig(
            firm_lei=sync_config.firm_lei,
            mode=WriterMode.SYNC,
            storage_config=AuditStorageConfig(
                backend_type=StorageBackendType.SQLITE, database_path=temp_db_path
            ),
            enable_retention_management=False,
        )

        writer = AuditTrailWriter(config)
        writer.start()

        # Write records
        for i in range(5):
            writer.write_system_event(AuditEventType.SYSTEM_STARTUP, f"Event {i}")

        # Export
        request = AuditExportRequest(export_format="json")
        result = writer.export(request)

        assert result.success is True
        assert result.records_exported == 5

        writer.stop()
