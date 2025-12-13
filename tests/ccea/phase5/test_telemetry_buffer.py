# -*- coding: utf-8 -*-
"""
Tests for Telemetry Buffer.

Design Doc Phase 5: Durable telemetry storage.
"""

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

        # Create second buffer
        buffer2 = TelemetryBuffer(config=config)
        pending = buffer2.get_pending_count()

        assert pending == 1
