# -*- coding: utf-8 -*-
"""
Tests for Telemetry Buffer - Phase 4.8

Tests cover:
- Telemetry levels (AGGREGATED, DETAILED, RAW)
- Durable buffer storage
- Mandatory redaction
- Flush to cloud
- Recovery after restart
"""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Any

import pytest

from packages.agent.daemon.telemetry_buffer import (
    TelemetryBuffer,
    TelemetryBufferConfig,
    TelemetryEvent,
    TelemetryEventType,
    TelemetryLevel,
)


class TestTelemetryEvent:
    """Test TelemetryEvent dataclass."""

    def test_creation_with_defaults(self):
        """Test event creation with defaults."""
        event = TelemetryEvent()

        assert event.event_type == TelemetryEventType.METRIC
        assert event.level == TelemetryLevel.AGGREGATED
        assert event.sent is False
        assert event.event_id is not None

    def test_creation_with_data(self):
        """Test event creation with data."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.ORDER_SUBMITTED,
            level=TelemetryLevel.DETAILED_NON_SENSITIVE,
            data={"symbol": "BTCUSDT", "quantity": "1.0"},
        )

        assert event.event_type == TelemetryEventType.ORDER_SUBMITTED
        assert event.data["symbol"] == "BTCUSDT"

    def test_to_dict(self):
        """Test serialization."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"status": "running"},
            run_id="run-123",
            agent_id="agent-456",
        )

        data = event.to_dict()

        assert data["event_type"] == "HEARTBEAT"
        assert data["run_id"] == "run-123"
        assert data["agent_id"] == "agent-456"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "event_id": "test-123",
            "event_type": "ERROR",
            "timestamp": datetime.utcnow().isoformat(),
            "level": "detailed",
            "data": {"error": "test"},
        }

        event = TelemetryEvent.from_dict(data)

        assert event.event_id == "test-123"
        assert event.event_type == TelemetryEventType.ERROR

    def test_get_redacted_data(self):
        """Test mandatory sensitive data redaction."""
        event = TelemetryEvent(
            data={
                "message": "public info",
                "api_key": "secret123",
                "api_secret": "secret456",
                "password": "pass123",
                "token": "tok123",
                "credential": "cred123",
                "bearer": "bear123",
                "authorization": "auth123",
                "amount": 100,
            }
        )

        redacted = event.get_redacted_data()

        # Public data preserved
        assert redacted["message"] == "public info"
        assert redacted["amount"] == 100

        # Sensitive data redacted
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["api_secret"] == "[REDACTED]"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["token"] == "[REDACTED]"
        assert redacted["credential"] == "[REDACTED]"
        assert redacted["bearer"] == "[REDACTED]"
        assert redacted["authorization"] == "[REDACTED]"

    def test_get_redacted_data_nested(self):
        """Test redaction of nested data."""
        event = TelemetryEvent(
            data={
                "config": {
                    "api_key": "secret",
                    "endpoint": "https://api.example.com",
                }
            }
        )

        redacted = event.get_redacted_data()

        assert redacted["config"]["api_key"] == "[REDACTED]"
        assert redacted["config"]["endpoint"] == "https://api.example.com"


class TestTelemetryBufferConfig:
    """Test TelemetryBufferConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = TelemetryBufferConfig()

        assert config.default_level == TelemetryLevel.AGGREGATED
        assert config.allow_raw_events is False  # Enterprise only
        assert config.redaction_enabled is True  # Cannot be disabled

    def test_to_dict(self):
        """Test serialization."""
        config = TelemetryBufferConfig(
            max_buffer_size=50000,
            batch_size=50,
        )

        data = config.to_dict()

        assert data["max_buffer_size"] == 50000
        assert data["batch_size"] == 50


class TestTelemetryBuffer:
    """Test TelemetryBuffer operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def buffer(self, temp_dir):
        """Create telemetry buffer."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "telemetry.db",
            batch_size=5,  # Small batch for testing
        )
        buf = TelemetryBuffer(
            config=config,
            agent_id="test-agent",
            run_id="test-run",
        )
        yield buf
        buf.close()

    def test_add_event(self, buffer):
        """Test adding event to buffer."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.HEARTBEAT,
            data={"status": "running"},
        )

        event_id = buffer.add(event)

        assert event_id is not None
        assert event.agent_id == "test-agent"
        assert event.run_id == "test-run"

    def test_add_metric(self, buffer):
        """Test adding metric event."""
        event_id = buffer.add_metric(
            name="cpu_usage",
            value=75.5,
            tags={"host": "agent-1"},
        )

        assert event_id is not None

    def test_add_heartbeat(self, buffer):
        """Test adding heartbeat event."""
        event_id = buffer.add_heartbeat(
            status="running",
            details={"positions": 5},
        )

        assert event_id is not None

    def test_add_error(self, buffer):
        """Test adding error event."""
        event_id = buffer.add_error(
            error_type="ConnectionError",
            message="Failed to connect to broker",
            details={"retry_count": 3},
        )

        assert event_id is not None

    def test_get_pending_count(self, buffer):
        """Test getting pending event count."""
        # Add events
        for i in range(10):
            buffer.add_heartbeat(status=f"status-{i}")

        count = buffer.get_pending_count()

        assert count == 10

    def test_get_pending_events(self, buffer):
        """Test getting pending events."""
        # Add events
        for i in range(5):
            buffer.add_metric(name=f"metric-{i}", value=i)

        events = buffer.get_pending_events(limit=10)

        assert len(events) == 5

    def test_flush_with_send_function(self, buffer):
        """Test flushing events with send function."""
        sent_events = []

        def mock_send(events: List[Dict[str, Any]]) -> bool:
            sent_events.extend(events)
            return True

        # Add events
        for i in range(3):
            buffer.add_heartbeat(status=f"status-{i}")

        # Flush
        sent_count = buffer.flush(send_fn=mock_send)

        assert sent_count == 3
        assert len(sent_events) == 3

    def test_flush_marks_sent(self, buffer):
        """Test flush marks events as sent."""

        def mock_send(events: List[Dict[str, Any]]) -> bool:
            return True

        # Add events
        buffer.add_heartbeat(status="test")

        # Before flush
        assert buffer.get_pending_count() == 1

        # After flush
        buffer.flush(send_fn=mock_send)
        assert buffer.get_pending_count() == 0

    def test_flush_retry_on_failure(self, buffer):
        """Test retry count incremented on failure."""

        def failing_send(events: List[Dict[str, Any]]) -> bool:
            return False

        # Add event
        buffer.add_heartbeat(status="test")

        # Flush fails
        buffer.flush(send_fn=failing_send)

        # Event still pending (will retry)
        assert buffer.get_pending_count() == 1

    def test_redaction_always_enabled(self, temp_dir):
        """Test redaction is always enabled in buffer init."""
        # Create config with redaction disabled
        config = TelemetryBufferConfig(
            db_path=temp_dir / "redact_test.db",
            redaction_enabled=False,
        )
        buffer = TelemetryBuffer(config=config)

        # Should be forced to True during init
        assert buffer.config.redaction_enabled is True

        buffer.close()

    def test_level_restrictions(self, buffer):
        """Test RAW level restricted to enterprise."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.ORDER_SUBMITTED,
            level=TelemetryLevel.RAW_ORDER_EVENTS,  # Raw level
        )

        buffer.add(event)

        # Should be downgraded since allow_raw_events is False
        assert event.level == TelemetryLevel.DETAILED_NON_SENSITIVE

    def test_get_statistics(self, buffer):
        """Test getting buffer statistics."""
        # Add some events
        for i in range(5):
            buffer.add_heartbeat(status=f"status-{i}")

        stats = buffer.get_statistics()

        assert stats["pending_events"] == 5
        assert "db_path" in stats

    def test_context_manager(self, temp_dir):
        """Test using buffer as context manager."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "context_test.db",
        )

        with TelemetryBuffer(config=config) as buffer:
            buffer.add_heartbeat(status="test")

        # Buffer should be closed

    def test_set_run_id(self, buffer):
        """Test setting run ID."""
        buffer.set_run_id("new-run-123")

        event = TelemetryEvent(event_type=TelemetryEventType.HEARTBEAT)
        buffer.add(event)

        assert event.run_id == "new-run-123"

    def test_set_agent_id(self, buffer):
        """Test setting agent ID."""
        buffer.set_agent_id("new-agent-456")

        event = TelemetryEvent(event_type=TelemetryEventType.HEARTBEAT)
        buffer.add(event)

        assert event.agent_id == "new-agent-456"


class TestTelemetryPersistence:
    """Test telemetry persistence across restarts."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_events_persist_across_restart(self, temp_dir):
        """Test events persist across restarts."""
        db_path = temp_dir / "persistent.db"
        config = TelemetryBufferConfig(db_path=db_path)

        # First buffer instance
        buffer1 = TelemetryBuffer(config=config)
        buffer1.add_heartbeat(status="before-restart")
        buffer1.close()

        # Second buffer instance (simulating restart)
        buffer2 = TelemetryBuffer(config=config)

        pending = buffer2.get_pending_events()
        assert len(pending) == 1
        assert pending[0].data["status"] == "before-restart"

        buffer2.close()

    def test_cleanup_old_events(self, temp_dir):
        """Test cleaning up old sent events."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "cleanup.db",
            max_age_days=0,  # Clean immediately
        )
        buffer = TelemetryBuffer(config=config)

        # Add and send event
        buffer.add_heartbeat(status="old")
        buffer.flush(send_fn=lambda x: True)

        # Cleanup
        deleted = buffer.cleanup_old_events()

        # Old sent events should be cleaned
        stats = buffer.get_statistics()
        assert stats["sent_events"] == 0

        buffer.close()


class TestTelemetryLevels:
    """Test telemetry level behavior."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_aggregated_level_default(self, temp_dir):
        """Test AGGREGATED is default level."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "levels.db",
            default_level=TelemetryLevel.AGGREGATED,
        )
        buffer = TelemetryBuffer(config=config)

        event_id = buffer.add_metric(name="test", value=1)
        events = buffer.get_pending_events()

        assert events[0].level == TelemetryLevel.AGGREGATED

        buffer.close()

    def test_detailed_level_opt_in(self, temp_dir):
        """Test DETAILED level is opt-in."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "detailed.db",
            default_level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        )
        buffer = TelemetryBuffer(config=config)

        event_id = buffer.add_metric(name="test", value=1)
        events = buffer.get_pending_events()

        assert events[0].level == TelemetryLevel.DETAILED_NON_SENSITIVE

        buffer.close()

    def test_raw_level_enterprise_only(self, temp_dir):
        """Test RAW level only for enterprise."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "raw.db",
            allow_raw_events=False,  # Default
        )
        buffer = TelemetryBuffer(config=config)

        event = TelemetryEvent(
            event_type=TelemetryEventType.ORDER_SUBMITTED,
            level=TelemetryLevel.RAW_ORDER_EVENTS,
        )
        buffer.add(event)

        # Should be downgraded
        assert event.level == TelemetryLevel.DETAILED_NON_SENSITIVE

        buffer.close()

    def test_raw_level_allowed_for_enterprise(self, temp_dir):
        """Test RAW level allowed when enabled."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "enterprise.db",
            allow_raw_events=True,  # Enterprise enabled
        )
        buffer = TelemetryBuffer(config=config)

        event = TelemetryEvent(
            event_type=TelemetryEventType.ORDER_SUBMITTED,
            level=TelemetryLevel.RAW_ORDER_EVENTS,
        )
        buffer.add(event)

        # Should keep raw level
        assert event.level == TelemetryLevel.RAW_ORDER_EVENTS

        buffer.close()


class TestMandatoryRedaction:
    """Test mandatory redaction per Design Doc."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_redaction_cannot_be_disabled(self, temp_dir):
        """Test redaction cannot be disabled."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "redaction.db",
            redaction_enabled=False,  # Try to disable
        )
        buffer = TelemetryBuffer(config=config)

        # Should be forced to True
        assert buffer.config.redaction_enabled is True

        buffer.close()

    def test_sensitive_data_redacted_on_flush(self, temp_dir):
        """Test sensitive data is redacted when flushing."""
        config = TelemetryBufferConfig(
            db_path=temp_dir / "flush_redact.db",
        )
        buffer = TelemetryBuffer(config=config)

        sent_events = []

        def capture_send(events: List[Dict[str, Any]]) -> bool:
            sent_events.extend(events)
            return True

        # Add event with sensitive data
        event = TelemetryEvent(
            event_type=TelemetryEventType.CONFIG_CHANGE,
            data={
                "setting_name": "some_value",  # Non-sensitive key
                "api_key": "should_be_redacted",
                "amount": 100,  # Numeric value preserved
            },
        )
        buffer.add(event)

        # Flush
        buffer.flush(send_fn=capture_send)

        # Check sent data is redacted
        assert len(sent_events) == 1
        assert sent_events[0]["data"]["api_key"] == "[REDACTED]"
        assert sent_events[0]["data"]["setting_name"] == "some_value"
        assert sent_events[0]["data"]["amount"] == 100

        buffer.close()
