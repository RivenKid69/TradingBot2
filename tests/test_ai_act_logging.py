# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Logging System (Article 12).

Tests cover:
- AIActLogger - Main logging functionality
- AIActLogEvent - Event data structures
- LogSession - Session management
- LogRetentionManager - Retention policies
- LogIntegrityVerifier - Tamper detection
- AIActEventBus - Extended event bus
- Factory functions
- Thread safety
"""

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.ai_act.logging_system import (
    # Event types
    AIActLogEventType,
    AIActLogCategory,
    # Data structures
    AIActLogEvent,
    LogSession,
    # Configuration
    AIActLoggingConfig,
    # Main classes
    AIActLogger,
    LogRetentionManager,
    LogIntegrityVerifier,
    AIActEventBus,
    # Factory functions
    create_ai_act_logger,
    create_ai_act_event_bus,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def logging_config(temp_log_dir):
    """Create a test logging configuration."""
    return AIActLoggingConfig(
        log_path=temp_log_dir,
        retention_days=180,
        enable_compression=True,
        compression_after_days=7,
        enable_integrity_verification=True,
        enable_async_logging=False,  # Disable async for testing
        buffer_size=10,
        flush_interval_seconds=1.0,
        log_all_decisions=True,
        log_all_risk_events=True,
        log_all_oversight_events=True,
        log_data_inputs=True,
        log_system_events=True,
    )


@pytest.fixture
def ai_act_logger(logging_config):
    """Create a test logger instance."""
    logger = AIActLogger(logging_config)
    yield logger
    logger.close()


# =============================================================================
# AIActLogEventType Tests
# =============================================================================


class TestAIActLogEventType:
    """Tests for AIActLogEventType enum."""

    def test_system_event_types(self):
        """Test system event type values."""
        assert AIActLogEventType.SYSTEM_START.value == "system_start"
        assert AIActLogEventType.SYSTEM_STOP.value == "system_stop"
        assert AIActLogEventType.SYSTEM_ERROR.value == "system_error"

    def test_decision_event_types(self):
        """Test decision event type values."""
        assert AIActLogEventType.TRADING_DECISION.value == "trading_decision"
        assert AIActLogEventType.PREDICTION.value == "prediction"
        assert AIActLogEventType.ORDER_PLACED.value == "order_placed"
        assert AIActLogEventType.ORDER_EXECUTED.value == "order_executed"

    def test_risk_event_types(self):
        """Test risk event type values."""
        assert AIActLogEventType.RISK_TRIGGER.value == "risk_trigger"
        assert AIActLogEventType.KILL_SWITCH_ACTIVATED.value == "kill_switch_activated"

    def test_oversight_event_types(self):
        """Test oversight event type values."""
        assert AIActLogEventType.HUMAN_OVERRIDE.value == "human_override"
        assert AIActLogEventType.OVERSIGHT_ALERT.value == "oversight_alert"

    def test_data_event_types(self):
        """Test data event type values."""
        assert AIActLogEventType.DATA_INPUT.value == "data_input"
        assert AIActLogEventType.DATA_QUALITY_CHECK.value == "data_quality_check"


class TestAIActLogCategory:
    """Tests for AIActLogCategory enum."""

    def test_category_values(self):
        """Test all category values."""
        assert AIActLogCategory.SYSTEM.value == "system"
        assert AIActLogCategory.DECISION.value == "decision"
        assert AIActLogCategory.RISK.value == "risk"
        assert AIActLogCategory.OVERSIGHT.value == "oversight"
        assert AIActLogCategory.DATA.value == "data"
        assert AIActLogCategory.EXPLAINABILITY.value == "explainability"
        assert AIActLogCategory.AUDIT.value == "audit"


# =============================================================================
# AIActLogEvent Tests
# =============================================================================


class TestAIActLogEvent:
    """Tests for AIActLogEvent dataclass."""

    def test_create_event(self):
        """Test creating a log event."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.TRADING_DECISION.value,
            category=AIActLogCategory.DECISION.value,
            session_id="SES-TEST",
        )

        assert event.event_id.startswith("EVT-")
        assert event.event_type == "trading_decision"
        assert event.category == "decision"
        assert event.session_id == "SES-TEST"

    def test_event_auto_timestamp(self):
        """Test event auto-generates timestamp."""
        event = AIActLogEvent()

        assert event.timestamp_utc != ""
        # Should be a valid ISO format
        datetime.fromisoformat(event.timestamp_utc.replace("Z", "+00:00"))

    def test_event_auto_id(self):
        """Test event auto-generates ID."""
        event = AIActLogEvent()

        assert event.event_id.startswith("EVT-")
        assert len(event.event_id) == 16  # EVT- + 12 hex chars

    def test_event_integrity_hash(self):
        """Test event integrity hash computation."""
        event = AIActLogEvent(
            event_type="test",
            category="test",
            session_id="SES-TEST",
        )

        assert event.integrity_hash != ""
        assert len(event.integrity_hash) == 32  # SHA-256 truncated

    def test_event_integrity_verification(self):
        """Test event integrity verification."""
        event = AIActLogEvent(
            event_type="test",
            category="test",
        )

        # Should verify successfully
        assert event.verify_integrity() is True

        # Tamper with the event
        original_hash = event.integrity_hash
        event.event_type = "tampered"

        # Should fail verification
        assert event.verify_integrity() is False

    def test_event_to_dict(self):
        """Test event serialization."""
        event = AIActLogEvent(
            event_type="test",
            category="test",
            inputs={"key": "value"},
        )

        data = event.to_dict()

        assert isinstance(data, dict)
        assert data["event_type"] == "test"
        assert data["inputs"]["key"] == "value"

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "event_id": "EVT-TEST123",
            "timestamp_utc": "2025-12-08T00:00:00+00:00",
            "event_type": "test",
            "category": "test",
            "session_id": "SES-TEST",
        }

        event = AIActLogEvent.from_dict(data)

        assert event.event_id == "EVT-TEST123"
        assert event.event_type == "test"

    def test_event_with_all_fields(self):
        """Test event with all fields populated."""
        event = AIActLogEvent(
            event_type=AIActLogEventType.TRADING_DECISION.value,
            category=AIActLogCategory.DECISION.value,
            system_state={"active": True},
            inputs={"observation": [1, 2, 3]},
            outputs={"action": "BUY"},
            confidence_metrics={"confidence": 0.95},
            human_oversight_state="active",
            human_operator_id="operator_001",
            session_id="SES-TEST",
            correlation_id="COR-TEST",
            causation_id="EVT-CAUSE",
            metadata={"extra": "data"},
            natural_person_involved="trader_001",
            verification_type="manual",
        )

        assert event.system_state["active"] is True
        assert event.inputs["observation"] == [1, 2, 3]
        assert event.outputs["action"] == "BUY"
        assert event.confidence_metrics["confidence"] == 0.95


# =============================================================================
# LogSession Tests
# =============================================================================


class TestLogSession:
    """Tests for LogSession dataclass."""

    def test_create_session(self):
        """Test creating a log session."""
        session = LogSession(
            system_version="1.0.0",
            operator_id="operator_001",
        )

        assert session.session_id.startswith("SES-")
        assert session.system_version == "1.0.0"
        assert session.is_active is True

    def test_session_auto_start_time(self):
        """Test session auto-generates start time."""
        session = LogSession()

        assert session.start_time != ""
        datetime.fromisoformat(session.start_time.replace("Z", "+00:00"))

    def test_session_default_values(self):
        """Test session default values."""
        session = LogSession()

        assert session.end_time is None
        assert session.is_active is True
        assert session.event_count == 0


# =============================================================================
# AIActLoggingConfig Tests
# =============================================================================


class TestAIActLoggingConfig:
    """Tests for AIActLoggingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AIActLoggingConfig()

        assert config.log_path == "logs/ai_act"
        assert config.retention_days == 180  # Article 19: 6 months
        assert config.enable_compression is True
        assert config.enable_integrity_verification is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AIActLoggingConfig(
            log_path="/custom/path",
            retention_days=365,
            enable_async_logging=False,
        )

        assert config.log_path == "/custom/path"
        assert config.retention_days == 365


# =============================================================================
# AIActLogger Tests
# =============================================================================


class TestAIActLogger:
    """Tests for AIActLogger class."""

    def test_logger_initialization(self, ai_act_logger, temp_log_dir):
        """Test logger initialization."""
        assert ai_act_logger is not None
        assert Path(temp_log_dir).exists()

    def test_logger_creates_subdirectories(self, temp_log_dir, logging_config):
        """Test logger creates category subdirectories."""
        logger = AIActLogger(logging_config)

        expected_dirs = ["decisions", "risk_events", "human_overrides",
                        "system_events", "data_events", "audit"]

        for dir_name in expected_dirs:
            assert (Path(temp_log_dir) / dir_name).exists()

        logger.close()

    def test_start_session(self, ai_act_logger):
        """Test starting a session."""
        session = ai_act_logger.start_session(
            system_version="1.0.0",
            operator_id="operator_001",
        )

        assert session is not None
        assert session.session_id.startswith("SES-")
        assert session.system_version == "1.0.0"
        assert session.is_active is True

    def test_end_session(self, ai_act_logger):
        """Test ending a session."""
        ai_act_logger.start_session(system_version="1.0.0")
        session = ai_act_logger.end_session()

        assert session is not None
        assert session.is_active is False
        assert session.end_time is not None

    def test_get_current_session(self, ai_act_logger):
        """Test getting current session."""
        assert ai_act_logger.get_current_session() is None

        ai_act_logger.start_session()
        session = ai_act_logger.get_current_session()

        assert session is not None
        assert session.is_active is True

    def test_log_trading_decision(self, ai_act_logger):
        """Test logging a trading decision."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_trading_decision(
            inputs={"observation": [1.0, 2.0, 3.0]},
            outputs={"action": "BUY", "position_size": 0.1},
            confidence_metrics={"value_estimate": 0.05},
            contributing_features=[("rsi", 0.3), ("momentum", 0.25)],
        )

        assert event is not None
        assert event.event_type == "trading_decision"
        assert event.category == "decision"

    def test_log_prediction(self, ai_act_logger):
        """Test logging a prediction."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_prediction(
            model_name="distributional_ppo",
            inputs={"features": [1.0, 2.0]},
            prediction=0.05,
            confidence=0.85,
            uncertainty_bounds=(0.02, 0.08),
        )

        assert event.event_type == "prediction"
        assert event.outputs["confidence"] == 0.85

    def test_log_order(self, ai_act_logger):
        """Test logging an order."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_order(
            order_id="ORD-001",
            order_type="MARKET",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            status="placed",
        )

        assert event.event_type == "order_placed"
        assert event.outputs["order_id"] == "ORD-001"

    def test_log_risk_event(self, ai_act_logger):
        """Test logging a risk event."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_risk_event(
            risk_type="threshold_breach",
            risk_id="RISK-001",
            severity="high",
            description="Drawdown limit exceeded",
            trigger_value=0.15,
            threshold_value=0.10,
        )

        assert event.event_type == "risk_trigger"
        assert event.category == "risk"

    def test_log_kill_switch(self, ai_act_logger):
        """Test logging kill switch activation."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_kill_switch(
            reason="Emergency stop",
            trigger_source="operator",
            positions_affected=5,
            orders_cancelled=3,
        )

        assert event.event_type == "kill_switch_activated"
        assert event.outputs["positions_affected"] == 5

    def test_log_human_override(self, ai_act_logger):
        """Test logging human override."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_human_override(
            action="position_closure",
            reason="Market volatility",
            operator_id="trader_001",
            override_type="manual",
        )

        assert event.event_type == "human_override"
        assert event.natural_person_involved == "trader_001"

    def test_log_oversight_alert(self, ai_act_logger):
        """Test logging oversight alert."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_oversight_alert(
            alert_type="anomaly_detected",
            message="Unusual trading pattern",
            severity="medium",
        )

        assert event.event_type == "oversight_alert"

    def test_log_data_input(self, ai_act_logger):
        """Test logging data input."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_data_input(
            data_source="binance",
            data_type="klines",
            record_count=1000,
            timestamp_range=("2025-01-01", "2025-12-01"),
        )

        assert event.event_type == "data_input"
        assert event.inputs["data_source"] == "binance"

    def test_log_data_quality_check(self, ai_act_logger):
        """Test logging data quality check."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_data_quality_check(
            check_name="completeness",
            passed=True,
            details={"score": 0.99},
        )

        assert event.event_type == "data_quality_check"
        assert event.outputs["passed"] is True

    def test_log_data_anomaly(self, ai_act_logger):
        """Test logging data anomaly."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_data_anomaly(
            anomaly_type="outlier",
            description="Price spike detected",
            affected_records=10,
            severity="medium",
        )

        assert event.event_type == "data_anomaly_detected"

    def test_log_system_event(self, ai_act_logger):
        """Test logging system event."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_system_event(
            event_type=AIActLogEventType.SYSTEM_CONFIG_CHANGE,
            description="Configuration updated",
            details={"param": "value"},
        )

        assert event.event_type == "system_config_change"

    def test_log_error(self, ai_act_logger):
        """Test logging error."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_error(
            error_type="ValueError",
            error_message="Invalid input",
            stack_trace="...",
            component="data_loader",
        )

        assert event.event_type == "system_error"
        assert event.outputs["error_type"] == "ValueError"

    def test_log_explanation(self, ai_act_logger):
        """Test logging explanation."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_explanation(
            decision_id="EVT-DEC001",
            explanation_type="feature_attribution",
            explanation_content={"top_features": ["rsi", "momentum"]},
        )

        assert event.event_type == "explanation_generated"
        assert event.causation_id == "EVT-DEC001"

    def test_log_audit_access(self, ai_act_logger):
        """Test logging audit access."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_audit_access(
            accessor_id="auditor_001",
            access_type="read",
            records_accessed=100,
        )

        assert event.event_type == "audit_access"
        assert event.natural_person_involved == "auditor_001"

    def test_correlation_tracking(self, ai_act_logger):
        """Test correlation ID tracking."""
        ai_act_logger.start_session()

        correlation_id = ai_act_logger.start_correlation()
        assert correlation_id.startswith("COR-")

        event = ai_act_logger.log_trading_decision(
            inputs={},
            outputs={},
        )

        assert event.correlation_id == correlation_id

        ai_act_logger.end_correlation()

    def test_get_log_statistics(self, ai_act_logger):
        """Test getting log statistics."""
        ai_act_logger.start_session()
        ai_act_logger.log_trading_decision(inputs={}, outputs={})

        stats = ai_act_logger.get_log_statistics()

        assert "log_path" in stats
        assert "current_session" in stats
        assert "retention_status" in stats

    def test_context_manager(self, logging_config):
        """Test logger as context manager."""
        with AIActLogger(logging_config) as logger:
            logger.start_session()
            logger.log_trading_decision(inputs={}, outputs={})

        # Logger should be closed after exiting context


# =============================================================================
# LogRetentionManager Tests
# =============================================================================


class TestLogRetentionManager:
    """Tests for LogRetentionManager class."""

    def test_retention_status(self, logging_config, temp_log_dir):
        """Test getting retention status."""
        manager = LogRetentionManager(logging_config)
        status = manager.get_retention_status()

        assert "retention_days" in status
        assert status["retention_days"] == 180
        assert "total_files" in status

    def test_compress_old_logs(self, logging_config, temp_log_dir):
        """Test compressing old logs."""
        # Create a test log file
        log_file = Path(temp_log_dir) / "test.jsonl"
        log_file.write_text('{"test": "data"}\n')

        # Set modification time to past
        old_time = time.time() - (8 * 24 * 3600)  # 8 days ago
        os.utime(log_file, (old_time, old_time))

        manager = LogRetentionManager(logging_config)
        compressed = manager.compress_old_logs()

        assert compressed >= 0

    def test_run_maintenance(self, logging_config):
        """Test running maintenance."""
        manager = LogRetentionManager(logging_config)
        result = manager.run_maintenance()

        assert "compressed" in result
        assert "deleted" in result


# =============================================================================
# LogIntegrityVerifier Tests
# =============================================================================


class TestLogIntegrityVerifier:
    """Tests for LogIntegrityVerifier class."""

    def test_verify_log_file(self, temp_log_dir):
        """Test verifying a log file."""
        verifier = LogIntegrityVerifier(Path(temp_log_dir))

        # Create a valid log file
        log_file = Path(temp_log_dir) / "test.jsonl"
        event = AIActLogEvent(event_type="test", category="test")
        log_file.write_text(json.dumps(event.to_dict()) + "\n")

        is_valid, errors = verifier.verify_log_file(log_file)

        assert is_valid is True
        assert len(errors) == 0

    def test_verify_invalid_log_file(self, temp_log_dir):
        """Test verifying an invalid log file."""
        verifier = LogIntegrityVerifier(Path(temp_log_dir))

        # Create an invalid log file
        log_file = Path(temp_log_dir) / "invalid.jsonl"
        log_file.write_text('not valid json\n')

        is_valid, errors = verifier.verify_log_file(log_file)

        assert is_valid is False
        assert len(errors) > 0

    def test_verify_all_logs(self, temp_log_dir):
        """Test verifying all log files."""
        verifier = LogIntegrityVerifier(Path(temp_log_dir))
        report = verifier.verify_all_logs()

        assert "timestamp" in report
        assert "total_files" in report
        assert "valid_files" in report
        assert "invalid_files" in report

    def test_chain_hash(self, temp_log_dir):
        """Test chain hash computation."""
        verifier = LogIntegrityVerifier(Path(temp_log_dir))

        event1 = AIActLogEvent(event_type="test1", category="test")
        event2 = AIActLogEvent(event_type="test2", category="test")

        hash1 = verifier.compute_chain_hash("GENESIS", event1)
        hash2 = verifier.compute_chain_hash(hash1, event2)

        assert hash1 != hash2
        assert len(hash1) == 32


# =============================================================================
# AIActEventBus Tests
# =============================================================================


class TestAIActEventBus:
    """Tests for AIActEventBus class."""

    @pytest.mark.asyncio
    async def test_event_bus_put_get(self):
        """Test event bus put and get."""
        bus = AIActEventBus(queue_size=10)

        event = {"type": "test", "data": "value"}
        accepted = await bus.put(event)

        assert accepted is True

        received = await bus.get()
        assert received["type"] == "test"
        assert "_ai_act_event_id" in received

        bus.close()

    @pytest.mark.asyncio
    async def test_event_bus_correlation_id(self):
        """Test event bus correlation ID."""
        bus = AIActEventBus(queue_size=10)
        bus.set_correlation_id("COR-TEST123")

        await bus.put({"type": "test"})
        received = await bus.get()

        assert received["_ai_act_correlation_id"] == "COR-TEST123"

        bus.close()

    def test_event_bus_depth(self):
        """Test event bus depth property."""
        bus = AIActEventBus(queue_size=10)

        assert bus.depth == 0

        bus.close()


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_ai_act_logger_default(self, temp_log_dir):
        """Test creating logger with defaults."""
        logger = create_ai_act_logger({"log_path": temp_log_dir})

        assert logger is not None
        logger.close()

    def test_create_ai_act_logger_with_config(self, logging_config):
        """Test creating logger with config object."""
        logger = create_ai_act_logger(logging_config)

        assert logger is not None
        logger.close()

    def test_create_ai_act_logger_with_dict(self, temp_log_dir):
        """Test creating logger with dict config."""
        config = {
            "log_path": temp_log_dir,
            "retention_days": 365,
        }
        logger = create_ai_act_logger(config)

        assert logger is not None
        logger.close()

    def test_create_ai_act_event_bus(self):
        """Test creating event bus."""
        bus = create_ai_act_event_bus(queue_size=100)

        assert bus is not None
        bus.close()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_logging(self, ai_act_logger):
        """Test concurrent logging from multiple threads."""
        ai_act_logger.start_session()
        events_logged = []
        errors = []

        def log_events(count):
            try:
                for i in range(count):
                    event = ai_act_logger.log_trading_decision(
                        inputs={"thread": threading.current_thread().name, "i": i},
                        outputs={"action": "BUY"},
                    )
                    events_logged.append(event)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=log_events, args=(10,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(events_logged) == 50

    def test_concurrent_sessions(self, logging_config):
        """Test concurrent session operations."""
        errors = []

        def session_operations(logger):
            try:
                session = logger.start_session()
                logger.log_trading_decision(inputs={}, outputs={})
                logger.end_session()
            except Exception as e:
                errors.append(e)

        logger = AIActLogger(logging_config)

        threads = []
        for _ in range(3):
            t = threading.Thread(target=session_operations, args=(logger,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        logger.close()

        # Some overlap is expected, but no crashes
        assert len([e for e in errors if "RuntimeError" in str(type(e))]) == 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_log_without_session(self, ai_act_logger):
        """Test logging without an active session."""
        # Should not raise, just log with empty session_id
        event = ai_act_logger.log_trading_decision(
            inputs={},
            outputs={},
        )

        assert event.session_id == ""

    def test_multiple_session_start(self, ai_act_logger):
        """Test starting multiple sessions (should end previous)."""
        session1 = ai_act_logger.start_session()
        session2 = ai_act_logger.start_session()

        # First session should be ended
        assert session1.session_id != session2.session_id

    def test_end_nonexistent_session(self, ai_act_logger):
        """Test ending when no session exists."""
        result = ai_act_logger.end_session()

        assert result is None

    def test_large_input_serialization(self, ai_act_logger):
        """Test serialization of large inputs."""
        import numpy as np

        ai_act_logger.start_session()

        # Large numpy array
        large_array = np.random.randn(1000, 100)

        event = ai_act_logger.log_trading_decision(
            inputs={"observation": large_array},
            outputs={},
        )

        # Should be serialized as summary
        assert "_type" in event.inputs.get("observation", {}) or isinstance(event.inputs.get("observation"), dict)


# =============================================================================
# Article 12 Compliance Tests
# =============================================================================


class TestArticle12Compliance:
    """Tests for Article 12 compliance requirements."""

    def test_period_of_use_tracking(self, ai_act_logger):
        """Test Article 12(2)(a) - period of use tracking."""
        session = ai_act_logger.start_session(
            system_version="1.0.0",
            operator_id="operator_001",
        )

        assert session.start_time is not None

        ended_session = ai_act_logger.end_session()

        assert ended_session.end_time is not None

    def test_natural_person_identification(self, ai_act_logger):
        """Test Article 12(2)(d) - natural person identification."""
        ai_act_logger.start_session()

        event = ai_act_logger.log_human_override(
            action="test",
            reason="test",
            operator_id="trader_john_doe",
        )

        assert event.natural_person_involved == "trader_john_doe"
        assert event.verification_type == "manual_override"

    def test_retention_period(self, logging_config):
        """Test Article 19 - minimum 6 month retention."""
        manager = LogRetentionManager(logging_config)
        status = manager.get_retention_status()

        assert status["retention_days"] >= 180  # 6 months

    def test_event_traceability(self, ai_act_logger):
        """Test event traceability with correlation IDs."""
        ai_act_logger.start_session()
        correlation_id = ai_act_logger.start_correlation()

        event1 = ai_act_logger.log_data_input(
            data_source="binance",
            data_type="klines",
            record_count=100,
        )

        event2 = ai_act_logger.log_trading_decision(
            inputs={},
            outputs={},
            causation_id=event1.event_id,
        )

        assert event1.correlation_id == correlation_id
        assert event2.correlation_id == correlation_id
        assert event2.causation_id == event1.event_id
