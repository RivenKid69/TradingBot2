# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 4 Audit Models.

Tests audit record models for MiFIR Article 25 compliance including:
- AuditEventType enum completeness
- AuditRecord creation, serialization, and validation
- AuditRecordBuilder fluent interface
- Hash computation and chain integrity
- Factory functions for common record types

Coverage target: 100%
"""

import json
import pytest
import time
from datetime import datetime, timedelta
from decimal import Decimal

from services.core.risk_controls.audit_models import (
    # Enums
    AuditEventType,
    AuditRecordPriority,
    AuditRecordStatus,
    OrderSide,
    # Data classes
    AuditRecord,
    AuditRecordBuilder,
    AuditChainStatus,
    AuditExportRequest,
    AuditExportResult,
    # Factory functions
    create_order_submitted_record,
    create_order_filled_record,
    create_risk_event_record,
    create_system_event_record,
)


# =============================================================================
# AuditEventType Tests
# =============================================================================

class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_all_event_types_defined(self):
        """Verify all expected event types exist."""
        # Order lifecycle
        assert AuditEventType.ORDER_SUBMITTED.value == "order_submitted"
        assert AuditEventType.ORDER_ACKNOWLEDGED.value == "order_acknowledged"
        assert AuditEventType.ORDER_MODIFIED.value == "order_modified"
        assert AuditEventType.ORDER_CANCELLED.value == "order_cancelled"
        assert AuditEventType.ORDER_REJECTED.value == "order_rejected"
        assert AuditEventType.ORDER_FILLED.value == "order_filled"
        assert AuditEventType.ORDER_PARTIALLY_FILLED.value == "order_partially_filled"
        assert AuditEventType.ORDER_EXPIRED.value == "order_expired"

        # Risk events
        assert AuditEventType.RISK_CHECK_PASSED.value == "risk_check_passed"
        assert AuditEventType.RISK_CHECK_FAILED.value == "risk_check_failed"
        assert AuditEventType.PRE_TRADE_CHECK_PASSED.value == "pre_trade_check_passed"
        assert AuditEventType.PRE_TRADE_CHECK_FAILED.value == "pre_trade_check_failed"
        assert AuditEventType.KILL_SWITCH_TRIGGERED.value == "kill_switch_triggered"

        # Algorithm events
        assert AuditEventType.ALGO_STARTED.value == "algo_started"
        assert AuditEventType.ALGO_STOPPED.value == "algo_stopped"
        assert AuditEventType.ALGO_PARAMETER_CHANGED.value == "algo_parameter_changed"

        # System events
        assert AuditEventType.SYSTEM_STARTUP.value == "system_startup"
        assert AuditEventType.SYSTEM_SHUTDOWN.value == "system_shutdown"
        assert AuditEventType.CONNECTION_ESTABLISHED.value == "connection_established"
        assert AuditEventType.CONNECTION_LOST.value == "connection_lost"

    def test_event_type_count(self):
        """Verify minimum number of event types for compliance."""
        # MiFIR requires comprehensive audit trail
        assert len(AuditEventType) >= 30

    def test_event_type_from_value(self):
        """Test enum creation from string value."""
        event = AuditEventType("order_submitted")
        assert event == AuditEventType.ORDER_SUBMITTED


class TestAuditRecordPriority:
    """Tests for AuditRecordPriority enum."""

    def test_priority_levels(self):
        """Test all priority levels exist."""
        assert AuditRecordPriority.CRITICAL.value == "critical"
        assert AuditRecordPriority.HIGH.value == "high"
        assert AuditRecordPriority.NORMAL.value == "normal"
        assert AuditRecordPriority.LOW.value == "low"

    def test_priority_count(self):
        """Test expected number of priority levels."""
        assert len(AuditRecordPriority) == 4


class TestAuditRecordStatus:
    """Tests for AuditRecordStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert AuditRecordStatus.PENDING.value == "pending"
        assert AuditRecordStatus.WRITTEN.value == "written"
        assert AuditRecordStatus.REPLICATED.value == "replicated"
        assert AuditRecordStatus.ARCHIVED.value == "archived"
        assert AuditRecordStatus.VERIFIED.value == "verified"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_order_sides(self):
        """Test order side values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


# =============================================================================
# AuditRecord Tests
# =============================================================================

class TestAuditRecord:
    """Tests for AuditRecord dataclass."""

    @pytest.fixture
    def sample_record(self):
        """Create a sample audit record for testing."""
        return AuditRecord(
            record_id="test-record-001",
            event_type=AuditEventType.ORDER_SUBMITTED,
            event_timestamp_ns=time.time_ns(),
            record_timestamp_ns=time.time_ns(),
            firm_lei="5493001KJTIIGC8Y1R12",
            algorithm_id="algo-001",
            trader_id="trader-001",
            order_id="order-001",
            client_order_id="client-001",
            instrument_isin="US0378331005",
            instrument_symbol="AAPL",
            venue_mic="XNAS",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            currency="USD",
            notional_value=Decimal("15050.00"),
            details={"strategy": "TWAP"},
            sequence_number=1,
            priority=AuditRecordPriority.NORMAL,
            status=AuditRecordStatus.PENDING,
        )

    def test_record_creation_with_defaults(self):
        """Test record creation with default values."""
        record = AuditRecord()
        assert record.record_id is not None
        assert len(record.record_id) == 36  # UUID format
        assert record.event_type == AuditEventType.SYSTEM_STARTUP
        assert record.event_timestamp_ns > 0
        assert record.record_timestamp_ns > 0
        assert record.firm_lei == ""
        assert record.details == {}

    def test_record_creation_with_values(self, sample_record):
        """Test record creation with explicit values."""
        assert sample_record.record_id == "test-record-001"
        assert sample_record.event_type == AuditEventType.ORDER_SUBMITTED
        assert sample_record.firm_lei == "5493001KJTIIGC8Y1R12"
        assert sample_record.order_id == "order-001"
        assert sample_record.quantity == Decimal("100")
        assert sample_record.side == OrderSide.BUY

    def test_to_dict(self, sample_record):
        """Test serialization to dictionary."""
        data = sample_record.to_dict()

        assert data["record_id"] == "test-record-001"
        assert data["event_type"] == "order_submitted"
        assert data["firm_lei"] == "5493001KJTIIGC8Y1R12"
        assert data["order_id"] == "order-001"
        assert data["quantity"] == "100"
        assert data["price"] == "150.50"
        assert data["side"] == "BUY"
        assert data["details"] == {"strategy": "TWAP"}

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes optional None values."""
        record = AuditRecord(
            record_id="test-001",
            firm_lei="5493001KJTIIGC8Y1R12",
        )
        data = record.to_dict()

        assert "order_id" not in data
        assert "algorithm_id" not in data
        assert "quantity" not in data

    def test_to_json(self, sample_record):
        """Test serialization to JSON string."""
        json_str = sample_record.to_json()

        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["record_id"] == "test-record-001"

    def test_to_json_with_indent(self, sample_record):
        """Test JSON serialization with indentation."""
        json_str = sample_record.to_json(indent=2)
        assert "\n" in json_str

    def test_compute_hash(self, sample_record):
        """Test hash computation."""
        hash1 = sample_record.compute_hash()

        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex length

        # Hash should be deterministic
        hash2 = sample_record.compute_hash()
        assert hash1 == hash2

    def test_hash_changes_with_data(self, sample_record):
        """Test that hash changes when data changes."""
        hash1 = sample_record.compute_hash()

        # Modify record
        sample_record.quantity = Decimal("200")
        hash2 = sample_record.compute_hash()

        assert hash1 != hash2

    def test_validate_valid_record(self, sample_record):
        """Test validation of valid record."""
        errors = sample_record.validate()
        assert errors == []
        assert sample_record.is_valid()

    def test_validate_missing_record_id(self):
        """Test validation catches missing record_id."""
        record = AuditRecord(record_id="", firm_lei="5493001KJTIIGC8Y1R12")
        errors = record.validate()
        assert "Missing record_id" in errors

    def test_validate_missing_firm_lei(self):
        """Test validation catches missing firm_lei."""
        record = AuditRecord(firm_lei="")
        errors = record.validate()
        assert "Missing firm_lei (required per MiFIR)" in errors

    def test_validate_invalid_timestamp(self):
        """Test validation catches invalid timestamps."""
        record = AuditRecord(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_timestamp_ns=0,
        )
        errors = record.validate()
        assert "Invalid event_timestamp_ns" in errors

    def test_validate_future_timestamp(self):
        """Test validation catches future timestamps."""
        future_ns = time.time_ns() + 60_000_000_000  # 60 seconds in future
        record = AuditRecord(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_timestamp_ns=future_ns,
        )
        errors = record.validate()
        assert "event_timestamp_ns is in the future" in errors

    def test_validate_hash_mismatch(self, sample_record):
        """Test validation catches hash mismatch."""
        sample_record.record_hash = "invalid_hash"
        errors = sample_record.validate()
        assert any("Hash mismatch" in e for e in errors)

    def test_validate_order_event_requires_order_id(self):
        """Test that order events require order_id."""
        record = AuditRecord(
            event_type=AuditEventType.ORDER_SUBMITTED,
            firm_lei="5493001KJTIIGC8Y1R12",
            order_id=None,
        )
        errors = record.validate()
        assert any("requires order_id" in e for e in errors)

    def test_event_datetime_property(self, sample_record):
        """Test event_datetime property."""
        dt = sample_record.event_datetime
        assert isinstance(dt, datetime)
        assert dt.year >= 2020

    def test_record_datetime_property(self, sample_record):
        """Test record_datetime property."""
        dt = sample_record.record_datetime
        assert isinstance(dt, datetime)

    def test_get_retention_expiry(self, sample_record):
        """Test retention expiry calculation."""
        # Default 5 years
        expiry = sample_record.get_retention_expiry()
        expected = sample_record.record_datetime + timedelta(days=5 * 365)
        assert abs((expiry - expected).total_seconds()) < 1

        # Extended 7 years
        expiry_extended = sample_record.get_retention_expiry(years=7)
        expected_extended = sample_record.record_datetime + timedelta(days=7 * 365)
        assert abs((expiry_extended - expected_extended).total_seconds()) < 1

    def test_from_dict(self, sample_record):
        """Test deserialization from dictionary."""
        data = sample_record.to_dict()
        restored = AuditRecord.from_dict(data)

        assert restored.record_id == sample_record.record_id
        assert restored.event_type == sample_record.event_type
        assert restored.firm_lei == sample_record.firm_lei
        assert restored.order_id == sample_record.order_id
        assert restored.quantity == sample_record.quantity
        assert restored.side == sample_record.side

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "record_id": "test-001",
            "event_type": "order_submitted",
            "firm_lei": "5493001KJTIIGC8Y1R12",
        }
        record = AuditRecord.from_dict(data)
        assert record.record_id == "test-001"
        assert record.event_type == AuditEventType.ORDER_SUBMITTED

    def test_from_json(self, sample_record):
        """Test deserialization from JSON string."""
        json_str = sample_record.to_json()
        restored = AuditRecord.from_json(json_str)

        assert restored.record_id == sample_record.record_id
        assert restored.event_type == sample_record.event_type


# =============================================================================
# AuditRecordBuilder Tests
# =============================================================================

class TestAuditRecordBuilder:
    """Tests for AuditRecordBuilder."""

    def test_basic_build(self):
        """Test basic record building."""
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("order-001")
            .build_unsafe()
        )

        assert record.event_type == AuditEventType.ORDER_SUBMITTED
        assert record.firm_lei == "5493001KJTIIGC8Y1R12"
        assert record.order_id == "order-001"
        assert record.record_hash is not None

    def test_full_order_build(self):
        """Test building a complete order record."""
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("order-001", "client-001")
            .instrument("US0378331005", "AAPL")
            .venue("XNAS")
            .side(OrderSide.BUY)
            .quantity(100)
            .price("150.50")
            .currency("USD")
            .notional(15050)
            .algorithm("algo-001")
            .trader("trader-001")
            .priority(AuditRecordPriority.HIGH)
            .build()
        )

        assert record.order_id == "order-001"
        assert record.client_order_id == "client-001"
        assert record.instrument_isin == "US0378331005"
        assert record.instrument_symbol == "AAPL"
        assert record.venue_mic == "XNAS"
        assert record.side == OrderSide.BUY
        assert record.quantity == Decimal("100")
        assert record.price == Decimal("150.50")
        assert record.currency == "USD"
        assert record.notional_value == Decimal("15050")
        assert record.algorithm_id == "algo-001"
        assert record.trader_id == "trader-001"
        assert record.priority == AuditRecordPriority.HIGH

    def test_details_methods(self):
        """Test details setting methods."""
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.SYSTEM_STARTUP)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .details({"key1": "value1"})
            .add_detail("key2", "value2")
            .build()
        )

        assert record.details["key1"] == "value1"
        assert record.details["key2"] == "value2"

    def test_timestamp_methods(self):
        """Test timestamp setting methods."""
        ts_ns = time.time_ns()
        ts_ms = int(time.time() * 1000)
        dt = datetime.now()

        # Test nanoseconds
        record1 = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .timestamp(ts_ns)
            .build()
        )
        assert record1.event_timestamp_ns == ts_ns

        # Test milliseconds
        record2 = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .timestamp_ms(ts_ms)
            .build()
        )
        assert abs(record2.event_timestamp_ns - ts_ms * 1_000_000) < 1_000_000

        # Test datetime
        record3 = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .timestamp_datetime(dt)
            .build()
        )
        expected_ns = int(dt.timestamp() * 1e9)
        assert abs(record3.event_timestamp_ns - expected_ns) < 1_000_000

    def test_chain_hash_setting(self):
        """Test setting previous record hash."""
        prev_hash = "abc123def456"
        record = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .previous_hash(prev_hash)
            .build()
        )
        assert record.previous_record_hash == prev_hash

    def test_sequence_setting(self):
        """Test sequence number setting."""
        record = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .sequence(42)
            .build()
        )
        assert record.sequence_number == 42

    def test_build_with_validation(self):
        """Test build with validation enabled."""
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("order-001")
            .build(validate=True)
        )
        assert record.is_valid()

    def test_build_validation_failure(self):
        """Test build fails validation with invalid data."""
        with pytest.raises(ValueError) as excinfo:
            (
                AuditRecordBuilder()
                .event_type(AuditEventType.ORDER_SUBMITTED)
                .firm_lei("")  # Missing LEI
                .build(validate=True)
            )
        assert "validation failed" in str(excinfo.value)

    def test_build_unsafe_skips_validation(self):
        """Test build_unsafe skips validation."""
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("")  # Invalid but should not raise
            .build_unsafe()
        )
        assert record.firm_lei == ""

    def test_build_computes_hash(self):
        """Test build automatically computes hash."""
        record = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .build()
        )
        assert record.record_hash is not None
        assert len(record.record_hash) == 64

    def test_build_sets_record_timestamp(self):
        """Test build sets record timestamp."""
        before = time.time_ns()
        record = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .build()
        )
        after = time.time_ns()

        assert before <= record.record_timestamp_ns <= after


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_order_submitted_record(self):
        """Test order submitted factory function."""
        record = create_order_submitted_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            order_id="order-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
            currency="USD",
            venue_mic="XNAS",
            algorithm_id="algo-001",
            trader_id="trader-001",
            client_order_id="client-001",
            details={"strategy": "TWAP"},
        )

        assert record.event_type == AuditEventType.ORDER_SUBMITTED
        assert record.firm_lei == "5493001KJTIIGC8Y1R12"
        assert record.order_id == "order-001"
        assert record.instrument_isin == "US0378331005"
        assert record.side == OrderSide.BUY
        assert record.quantity == Decimal("100")
        assert record.price == Decimal("150.50")
        assert record.algorithm_id == "algo-001"
        assert record.is_valid()

    def test_create_order_submitted_minimal(self):
        """Test order submitted with minimal params."""
        record = create_order_submitted_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            order_id="order-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.50"),
        )
        assert record.event_type == AuditEventType.ORDER_SUBMITTED
        assert record.is_valid()

    def test_create_order_filled_record(self):
        """Test order filled factory function."""
        record = create_order_filled_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            order_id="order-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=Decimal("100"),
            fill_price=Decimal("150.55"),
            currency="USD",
            venue_mic="XNAS",
            is_partial=False,
            algorithm_id="algo-001",
        )

        assert record.event_type == AuditEventType.ORDER_FILLED
        assert record.quantity == Decimal("100")
        assert record.price == Decimal("150.55")
        assert record.notional_value == Decimal("15055.00")
        assert record.is_valid()

    def test_create_order_partially_filled(self):
        """Test partial fill record creation."""
        record = create_order_filled_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            order_id="order-001",
            instrument_isin="US0378331005",
            side=OrderSide.BUY,
            fill_quantity=Decimal("50"),
            fill_price=Decimal("150.50"),
            is_partial=True,
            remaining_quantity=Decimal("50"),
        )

        assert record.event_type == AuditEventType.ORDER_PARTIALLY_FILLED
        assert record.quantity == Decimal("50")
        assert record.details["remaining_quantity"] == "50"

    def test_create_risk_event_record(self):
        """Test risk event factory function."""
        record = create_risk_event_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_type=AuditEventType.PRE_TRADE_CHECK_FAILED,
            reason="Price collar breach",
            algorithm_id="algo-001",
            order_id="order-001",
            details={"collar_limit": "5%"},
        )

        assert record.event_type == AuditEventType.PRE_TRADE_CHECK_FAILED
        assert record.details["reason"] == "Price collar breach"
        assert record.priority == AuditRecordPriority.HIGH
        assert record.is_valid()

    def test_create_risk_event_critical_priority(self):
        """Test risk event with critical priority."""
        record = create_risk_event_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_type=AuditEventType.KILL_SWITCH_TRIGGERED,
            reason="Emergency",
        )
        assert record.priority == AuditRecordPriority.CRITICAL

    def test_create_system_event_record(self):
        """Test system event factory function."""
        record = create_system_event_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_type=AuditEventType.SYSTEM_STARTUP,
            message="Trading system started",
            details={"version": "1.0.0"},
        )

        assert record.event_type == AuditEventType.SYSTEM_STARTUP
        assert record.details["message"] == "Trading system started"
        assert record.details["version"] == "1.0.0"
        assert record.priority == AuditRecordPriority.CRITICAL
        assert record.is_valid()

    def test_create_system_event_connection_lost(self):
        """Test connection lost event priority."""
        record = create_system_event_record(
            firm_lei="5493001KJTIIGC8Y1R12",
            event_type=AuditEventType.CONNECTION_LOST,
            message="Market data feed disconnected",
        )
        assert record.priority == AuditRecordPriority.HIGH


# =============================================================================
# AuditChainStatus Tests
# =============================================================================

class TestAuditChainStatus:
    """Tests for AuditChainStatus dataclass."""

    def test_valid_chain_status(self):
        """Test valid chain status."""
        status = AuditChainStatus(
            is_valid=True,
            records_checked=1000,
        )
        assert status.is_valid
        assert status.records_checked == 1000
        assert status.first_invalid_record_id is None

    def test_invalid_chain_status(self):
        """Test invalid chain status."""
        status = AuditChainStatus(
            is_valid=False,
            records_checked=500,
            first_invalid_record_id="record-501",
            first_invalid_index=500,
            error_message="Hash mismatch",
        )
        assert not status.is_valid
        assert status.first_invalid_record_id == "record-501"
        assert status.first_invalid_index == 500
        assert status.error_message == "Hash mismatch"

    def test_verification_timestamp(self):
        """Test verification timestamp is set."""
        before = time.time_ns()
        status = AuditChainStatus(is_valid=True, records_checked=0)
        after = time.time_ns()

        assert before <= status.verification_timestamp_ns <= after


# =============================================================================
# AuditExportRequest Tests
# =============================================================================

class TestAuditExportRequest:
    """Tests for AuditExportRequest dataclass."""

    def test_export_request_creation(self):
        """Test export request creation."""
        request = AuditExportRequest(
            request_id="export-001",
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 12, 31),
            event_types=[AuditEventType.ORDER_SUBMITTED, AuditEventType.ORDER_FILLED],
            order_ids=["order-001", "order-002"],
            export_format="json",
            include_chain_verification=True,
            requested_by="compliance@firm.com",
            request_reason="NCA FCA audit request",
        )

        assert request.request_id == "export-001"
        assert request.start_datetime == datetime(2024, 1, 1)
        assert len(request.event_types) == 2
        assert len(request.order_ids) == 2
        assert request.include_chain_verification

    def test_export_request_defaults(self):
        """Test export request default values."""
        request = AuditExportRequest()

        assert request.request_id is not None
        assert request.export_format == "json"
        assert request.include_chain_verification is True


# =============================================================================
# AuditExportResult Tests
# =============================================================================

class TestAuditExportResult:
    """Tests for AuditExportResult dataclass."""

    def test_successful_export_result(self):
        """Test successful export result."""
        result = AuditExportResult(
            request_id="export-001",
            success=True,
            records_exported=5000,
            export_path="/path/to/export.json",
            chain_verification=AuditChainStatus(is_valid=True, records_checked=5000),
        )

        assert result.success
        assert result.records_exported == 5000
        assert result.export_path is not None
        assert result.chain_verification.is_valid

    def test_failed_export_result(self):
        """Test failed export result."""
        result = AuditExportResult(
            request_id="export-002",
            success=False,
            records_exported=0,
            error_message="Database connection failed",
        )

        assert not result.success
        assert result.records_exported == 0
        assert result.error_message is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAuditRecordChaining:
    """Integration tests for record chain integrity."""

    def test_record_chain_creation(self):
        """Test creating a chain of records."""
        records = []
        prev_hash = None

        for i in range(5):
            builder = (
                AuditRecordBuilder()
                .event_type(AuditEventType.ORDER_SUBMITTED)
                .firm_lei("5493001KJTIIGC8Y1R12")
                .order_id(f"order-{i:03d}")
                .sequence(i + 1)
            )

            if prev_hash:
                builder.previous_hash(prev_hash)

            record = builder.build()
            records.append(record)
            prev_hash = record.record_hash

        # Verify chain
        for i in range(1, len(records)):
            assert records[i].previous_record_hash == records[i - 1].record_hash

    def test_chain_tamper_detection(self):
        """Test that tampering is detected."""
        # Create chain
        record1 = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .sequence(1)
            .build()
        )

        record2 = (
            AuditRecordBuilder()
            .firm_lei("5493001KJTIIGC8Y1R12")
            .sequence(2)
            .previous_hash(record1.record_hash)
            .build()
        )

        # Tamper with record1
        record1.details["tampered"] = True

        # Hash no longer matches
        assert record1.compute_hash() != record1.record_hash
        errors = record1.validate()
        assert any("Hash mismatch" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
