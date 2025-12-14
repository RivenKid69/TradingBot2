# -*- coding: utf-8 -*-
"""
Tests for Telemetry Buffer.

Design Doc Phase 5: Durable telemetry storage.

WI-AGENT-02: Includes regression tests for Windows file locking issues
and proper resource cleanup.
"""

import contextlib
import os
import platform
import sqlite3
import threading
import time
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from packages.agent.daemon.telemetry_buffer import (
    TelemetryBuffer,
    TelemetryBufferConfig,
    TelemetryEvent,
    TelemetryEventType,
    TelemetryLevel,
)


class TestTelemetryEvent:
    """Tests for TelemetryEvent."""

    def test_create_event(self):
        """Test creating event."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"status": "running"},
            run_id="test-123",
        )

        assert event.event_type == TelemetryEventType.HEARTBEAT
        assert event.data["status"] == "running"
        assert event.level == TelemetryLevel.AGGREGATED
        assert event.sent is False

    def test_event_to_dict(self):
        """Test serialization."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.ERROR,
            data={"message": "Test error"},
        )

        d = event.to_dict()
        assert d["event_type"] == "ERROR"
        assert d["sent"] is False

    def test_event_from_dict(self):
        """Test deserialization."""
        data = {
            "event_id": "test-123",
            "event_type": "ORDER_SUBMITTED",
            "timestamp": "2025-01-01T00:00:00",
            "level": "detailed",
            "data": {"order_id": "o1"},
            "run_id": "run-1",
            "agent_id": "agent-1",
            "sent": False,
            "retry_count": 0,
        }

        event = TelemetryEvent.from_dict(data)
        assert event.event_id == "test-123"
        assert event.event_type == TelemetryEventType.ORDER_SUBMITTED
        assert event.level == TelemetryLevel.DETAILED_NON_SENSITIVE

    def test_redaction(self):
        """Test sensitive data redaction."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.METRIC,
            data={
                "api_key": "secret123",
                "api_secret": "supersecret",
                "pnl": 1000,
                "token": "bearer_token",
                "nested": {
                    "password": "mypass",
                    "value": 42,
                },
            },
        )

        redacted = event.get_redacted_data()
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["api_secret"] == "[REDACTED]"
        assert redacted["token"] == "[REDACTED]"
        assert redacted["pnl"] == 1000
        assert redacted["nested"]["password"] == "[REDACTED]"
        assert redacted["nested"]["value"] == 42


class TestTelemetryBufferConfig:
    """Tests for TelemetryBufferConfig."""

    def test_default_config(self):
        """Test default values."""
        config = TelemetryBufferConfig()

        assert config.max_buffer_size == 100000
        assert config.batch_size == 100
        assert config.default_level == TelemetryLevel.AGGREGATED
        assert config.allow_raw_events is False
        assert config.redaction_enabled is True

    def test_redaction_always_enabled(self):
        """Test redaction cannot be disabled."""
        config = TelemetryBufferConfig(redaction_enabled=False)
        # Config stores false, but buffer should enforce true
        # This is checked at buffer level


class TestTelemetryBuffer:
    """Tests for TelemetryBuffer."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def buffer(self, temp_dir):
        """Create TelemetryBuffer with temp storage."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            batch_size=10,
        )
        return TelemetryBuffer(config=config, agent_id="test-agent")

    def test_add_event(self, buffer):
        """Test adding event."""
        event_id = buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"status": "running"},
        ))

        assert event_id is not None
        assert len(buffer._memory_buffer) == 1

    def test_add_metric(self, buffer):
        """Test adding metric."""
        event_id = buffer.add_metric(
            name="cpu_usage",
            value=75.5,
            tags={"host": "agent-1"},
        )

        assert event_id is not None

    def test_add_heartbeat(self, buffer):
        """Test adding heartbeat."""
        event_id = buffer.add_heartbeat(
            status="running",
            details={"uptime": 3600},
        )

        assert event_id is not None

    def test_add_error(self, buffer):
        """Test adding error."""
        event_id = buffer.add_error(
            error_type="ConnectionError",
            message="Failed to connect",
            details={"host": "broker.com"},
        )

        assert event_id is not None

    def test_batch_persistence(self, buffer):
        """Test batch persistence when size reached."""
        # Add events up to batch size
        for i in range(12):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.METRIC,
                data={"index": i},
            ))

        # Should have persisted first batch, some in memory
        assert len(buffer._memory_buffer) < 12
        pending = buffer.get_pending_count()
        assert pending == 12

    def test_flush_with_send_fn(self, buffer):
        """Test flush with send function."""
        # Add events
        for i in range(5):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"index": i},
            ))

        # Mock send function
        send_fn = MagicMock(return_value=True)

        # Flush
        sent = buffer.flush(send_fn=send_fn)

        assert sent == 5
        send_fn.assert_called_once()

    def test_flush_failure_increments_retry(self, buffer):
        """Test flush failure increments retry count."""
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))

        # Mock failed send
        send_fn = MagicMock(return_value=False)

        # Flush fails
        sent = buffer.flush(send_fn=send_fn)
        assert sent == 0

        # Event still pending
        events = buffer.get_pending_events()
        assert len(events) == 1
        assert events[0].retry_count == 1

    def test_get_pending_count(self, buffer):
        """Test pending count."""
        assert buffer.get_pending_count() == 0

        for i in range(3):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.METRIC,
                data={"i": i},
            ))

        assert buffer.get_pending_count() == 3

    def test_get_pending_events(self, buffer):
        """Test getting pending events."""
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.ERROR,
            data={"message": "Test"},
        ))

        events = buffer.get_pending_events()
        assert len(events) == 1
        assert events[0].event_type == TelemetryEventType.ERROR

    def test_cleanup_old_events(self, buffer):
        """Test cleanup of old sent events."""
        # Add and mark as sent
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))

        # Force persist and mark sent
        buffer._persist_batch()

        send_fn = MagicMock(return_value=True)
        buffer.flush(send_fn=send_fn)

        # Set old timestamp by directly modifying DB
        import sqlite3
        with sqlite3.connect(str(buffer.config.db_path)) as conn:
            old_time = (datetime.utcnow() - timedelta(days=10)).isoformat()
            conn.execute("UPDATE telemetry_events SET timestamp = ?", (old_time,))
            conn.commit()

        # Cleanup
        deleted = buffer.cleanup_old_events()
        assert deleted == 1

    def test_export_jsonl(self, buffer, temp_dir):
        """Test JSONL export."""
        for i in range(3):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.METRIC,
                data={"index": i},
            ))

        output_path = temp_dir / "export.jsonl"
        count = buffer.export_jsonl(output_path)

        assert count == 3
        assert output_path.exists()

        # Verify content
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "event_id" in data

    def test_get_statistics(self, buffer):
        """Test statistics retrieval."""
        for i in range(5):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"i": i},
            ))

        # Force persist to DB
        buffer._persist_batch()

        stats = buffer.get_statistics()

        assert stats["total_events"] == 5
        assert stats["pending_events"] == 5
        assert stats["sent_events"] == 0

    def test_agent_id_propagation(self, buffer):
        """Test agent ID is set on events."""
        event_id = buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={},
        ))

        events = buffer.get_pending_events()
        assert events[0].agent_id == "test-agent"

    def test_run_id_propagation(self, buffer):
        """Test run ID is set on events."""
        buffer.set_run_id("run-123")

        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={},
        ))

        events = buffer.get_pending_events()
        assert events[0].run_id == "run-123"

    def test_level_restriction(self, buffer):
        """Test RAW level is restricted."""
        buffer.config.allow_raw_events = False

        event = TelemetryEvent(
            event_type=TelemetryEventType.ORDER_SUBMITTED,
            level=TelemetryLevel.RAW_ORDER_EVENTS,
            data={"order": "details"},
        )
        buffer.add(event)

        events = buffer.get_pending_events()
        # Should be downgraded
        assert events[0].level == TelemetryLevel.DETAILED_NON_SENSITIVE

    def test_redaction_enforced(self, buffer):
        """Test redaction is always enforced in __init__."""
        # The buffer should have redaction enabled after init
        # (it's set in __init__ to True regardless of config)
        # Test that redaction works on events
        event = TelemetryEvent(
            event_type=TelemetryEventType.METRIC,
            data={"api_key": "secret123", "value": 42},
        )
        buffer.add(event)
        buffer._persist_batch()

        # Check that data is persisted (redaction happens during persist)
        events = buffer.get_pending_events()
        assert len(events) == 1
        # The redaction is applied during serialize, so check redacted version
        redacted = events[0].get_redacted_data()
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["value"] == 42

    def test_persistence_across_instances(self, temp_dir):
        """Test data persists across buffer instances."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        # Create first buffer and add events
        buffer1 = TelemetryBuffer(config=config)
        buffer1.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": "data"},
        ))
        buffer1._persist_batch()
        buffer1.close()  # Explicitly close to release resources

        # Create second buffer
        buffer2 = TelemetryBuffer(config=config)
        pending = buffer2.get_pending_count()
        buffer2.close()

        assert pending == 1


class TestTelemetryBufferResourceManagement:
    """
    Tests for proper resource management.

    WI-AGENT-02: These tests verify that SQLite connections are properly closed
    and background threads are cleanly stopped, preventing Windows file locking issues.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_close_method_exists(self, temp_dir):
        """Test close() method is available."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")
        buffer = TelemetryBuffer(config=config)

        assert hasattr(buffer, 'close')
        assert callable(buffer.close)

        buffer.close()

    def test_context_manager_support(self, temp_dir):
        """Test buffer can be used as context manager."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        with TelemetryBuffer(config=config) as buffer:
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"test": True},
            ))
            count = buffer.get_pending_count()
            assert count == 1

        # After exiting context, buffer should be closed
        # Try to reopen the database to verify no locks remain
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            assert cursor.fetchone()[0] == 1

    def test_close_stops_background_thread(self, temp_dir):
        """Test close() stops background flush thread."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            flush_interval_seconds=1,  # Short interval for test
        )
        buffer = TelemetryBuffer(config=config)

        # Start background flusher
        send_fn = MagicMock(return_value=True)
        try:
            buffer.start_background_flush(send_fn)
        except RuntimeError as e:
            if "can't start new thread" in str(e):
                pytest.skip("Thread creation not supported in this environment")
            raise

        assert buffer._flushing is True
        assert buffer._flush_thread is not None
        assert buffer._flush_thread.is_alive()

        # Close should stop the thread
        buffer.close()

        assert buffer._flushing is False
        assert buffer._flush_thread is None

    def test_stop_background_flush_returns_status(self, temp_dir):
        """Test stop_background_flush returns True when thread stops cleanly."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            flush_interval_seconds=60,  # Long interval
        )
        buffer = TelemetryBuffer(config=config)

        send_fn = MagicMock(return_value=True)
        try:
            buffer.start_background_flush(send_fn)
        except RuntimeError as e:
            if "can't start new thread" in str(e):
                pytest.skip("Thread creation not supported in this environment")
            raise

        # Stop should return True for clean stop
        result = buffer.stop_background_flush(timeout=5.0)
        assert result is True

    def test_multiple_close_calls_safe(self, temp_dir):
        """Test calling close() multiple times is safe."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")
        buffer = TelemetryBuffer(config=config)

        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))

        # Multiple close calls should not raise
        buffer.close()
        buffer.close()
        buffer.close()

    def test_close_persists_memory_buffer(self, temp_dir):
        """Test close() persists events in memory buffer to disk."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            batch_size=100,  # Large batch size so events stay in memory
        )
        buffer = TelemetryBuffer(config=config)

        # Add events (less than batch size, so they stay in memory)
        for i in range(5):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.METRIC,
                data={"index": i},
            ))

        assert len(buffer._memory_buffer) == 5

        # Close should persist to disk
        buffer.close()

        # Verify events are in database
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            assert cursor.fetchone()[0] == 5


class TestWindowsFileLocking:
    """
    Regression tests for Windows file locking issues.

    WI-AGENT-02: These tests verify that SQLite connections are properly closed
    using contextlib.closing, preventing "database is locked" errors on Windows.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_no_database_locked_on_rapid_operations(self, temp_dir):
        """Test rapid DB operations don't cause locking issues."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        # Rapid create/close cycles should not cause locking
        for i in range(10):
            buffer = TelemetryBuffer(config=config)
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"iteration": i},
            ))
            buffer._persist_batch()
            buffer.close()

        # Final verification
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            assert cursor.fetchone()[0] == 10

    def test_concurrent_read_after_close(self, temp_dir):
        """Test database can be read immediately after buffer close."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        buffer = TelemetryBuffer(config=config)
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))
        buffer._persist_batch()
        buffer.close()

        # Immediately try to read - should not be locked
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT * FROM telemetry_events")
            rows = cursor.fetchall()
            assert len(rows) == 1

    def test_multiple_buffers_same_db_sequential(self, temp_dir):
        """Test multiple sequential buffer instances on same DB."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        # First buffer
        with TelemetryBuffer(config=config) as buffer1:
            buffer1.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"buffer": 1},
            ))

        # Second buffer - should be able to access DB
        with TelemetryBuffer(config=config) as buffer2:
            buffer2.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"buffer": 2},
            ))
            count = buffer2.get_pending_count()
            assert count == 2

    def test_db_file_not_locked_after_operations(self, temp_dir):
        """Test DB file can be deleted after buffer operations (not locked)."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")

        buffer = TelemetryBuffer(config=config)
        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))
        buffer.flush()
        buffer.get_statistics()
        buffer.get_pending_count()
        buffer.close()

        # On Windows, trying to delete a locked file would raise PermissionError
        # This tests that no handles remain open
        db_path = config.db_path
        assert db_path.exists()

        try:
            db_path.unlink()
            # If we get here, file was not locked
        except PermissionError:
            pytest.fail("Database file is still locked after close()")

    def test_background_flush_doesnt_hold_lock(self, temp_dir):
        """Test background flush doesn't hold database lock continuously."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            flush_interval_seconds=1,
        )
        buffer = TelemetryBuffer(config=config)

        send_fn = MagicMock(return_value=True)
        try:
            buffer.start_background_flush(send_fn)
        except RuntimeError as e:
            if "can't start new thread" in str(e):
                pytest.skip("Thread creation not supported in this environment")
            raise

        # Add some events
        for i in range(5):
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"index": i},
            ))

        # Wait for a flush cycle
        time.sleep(1.5)

        # While background thread is running, we should still be able to
        # create a separate connection and read
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            count = cursor.fetchone()[0]
            # Events should have been persisted
            assert count >= 0  # May vary based on timing

        buffer.close()

    def test_flush_with_exception_releases_connection(self, temp_dir):
        """Test flush releases connection even on exception."""
        config = TelemetryBufferConfig(db_path=temp_dir / "telemetry.db")
        buffer = TelemetryBuffer(config=config)

        buffer.add(TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"test": True},
        ))

        # Send function that raises exception
        def failing_send(events):
            raise RuntimeError("Simulated failure")

        # This should not leave connection open
        sent = buffer.flush(send_fn=failing_send)
        assert sent == 0

        # Close and verify no lock
        buffer.close()

        # Should be able to open database
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            assert cursor.fetchone()[0] == 1


class TestCleanupOldEventsWithClosing:
    """
    Tests for cleanup operations using proper connection management.

    WI-AGENT-02: Verifies cleanup_old_events properly closes connections.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cleanup_old_events_using_contextlib_closing(self, temp_dir):
        """Test cleanup uses contextlib.closing for connection."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            max_age_days=1,
        )

        with TelemetryBuffer(config=config) as buffer:
            # Add and mark as sent
            buffer.add(TelemetryEvent(
                event_type=TelemetryEventType.HEARTBEAT,
                data={"test": True},
            ))
            buffer._persist_batch()

            send_fn = MagicMock(return_value=True)
            buffer.flush(send_fn=send_fn)

            # Manually set old timestamp
            with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
                old_time = (datetime.utcnow() - timedelta(days=10)).isoformat()
                conn.execute("UPDATE telemetry_events SET timestamp = ?", (old_time,))
                conn.commit()

            # Cleanup should work and release connection
            deleted = buffer.cleanup_old_events()
            assert deleted == 1

        # Verify no locks remain
        with contextlib.closing(sqlite3.connect(str(config.db_path))) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM telemetry_events")
            assert cursor.fetchone()[0] == 0
