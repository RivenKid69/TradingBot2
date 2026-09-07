# -*- coding: utf-8 -*-
"""
MiFID II Audit Trail Models - MiFIR Article 25 Compliance.

This module provides data models for maintaining a tamper-proof audit trail
of all trading activities per MiFIR Article 25 requirements:

- All orders (submitted, modified, cancelled)
- All transactions (executed trades)
- Algorithm parameters and changes
- Risk control decisions
- Timestamps with microsecond/nanosecond precision (RTS 25)

Key Requirements:
    - Records must be stored for 5 years minimum (7 years if requested by NCA)
    - Records must be tamper-proof (chain integrity via hashes)
    - Records must be retrievable within specified timeframes
    - Timestamps must be RTS 25 compliant

References:
    - MiFIR Article 25: Obligation to maintain records
        https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifir/article-25-obligation-maintain-records
    - RTS 24 (Regulation 2017/580): Order book data
    - RTS 25 (Regulation 2017/574): Clock synchronisation
    - ESMA Guidelines on transaction reporting
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Union


class AuditEventType(Enum):
    """
    Types of auditable events per MiFIR Article 25.

    All order lifecycle events, risk decisions, algorithm events,
    and system events must be recorded in the audit trail.
    """

    # === Order Lifecycle Events ===
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_EXPIRED = "order_expired"
    ORDER_REPLACED = "order_replaced"
    ORDER_SUSPENDED = "order_suspended"
    ORDER_REACTIVATED = "order_reactivated"

    # === Risk Control Events ===
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    PRE_TRADE_CHECK_PASSED = "pre_trade_check_passed"
    PRE_TRADE_CHECK_FAILED = "pre_trade_check_failed"
    POSITION_LIMIT_WARNING = "position_limit_warning"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    PNL_LIMIT_WARNING = "pnl_limit_warning"
    PNL_LIMIT_BREACH = "pnl_limit_breach"
    MESSAGE_RATE_WARNING = "message_rate_warning"
    MESSAGE_RATE_BREACH = "message_rate_breach"

    # === Kill Switch Events ===
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    KILL_SWITCH_RECOVERED = "kill_switch_recovered"
    KILL_SWITCH_MANUAL = "kill_switch_manual"
    KILL_SWITCH_AUTOMATIC = "kill_switch_automatic"

    # === Algorithm Events ===
    ALGO_STARTED = "algo_started"
    ALGO_STOPPED = "algo_stopped"
    ALGO_PAUSED = "algo_paused"
    ALGO_RESUMED = "algo_resumed"
    ALGO_PARAMETER_CHANGED = "algo_parameter_changed"
    ALGO_REGISTERED = "algo_registered"
    ALGO_DEREGISTERED = "algo_deregistered"
    ALGO_VERSION_CHANGED = "algo_version_changed"

    # === System Events ===
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_RECOVERY = "system_recovery"

    # === Connectivity Events ===
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    RECONNECTION_ATTEMPT = "reconnection_attempt"

    # === Market Events ===
    MARKET_DATA_RECEIVED = "market_data_received"
    MARKET_DATA_STALE = "market_data_stale"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    TRADING_HALT = "trading_halt"
    TRADING_RESUMED = "trading_resumed"

    # === Compliance Events ===
    TRANSACTION_REPORTED = "transaction_reported"
    TRANSACTION_REPORT_FAILED = "transaction_report_failed"
    CLOCK_SYNC_SUCCESS = "clock_sync_success"
    CLOCK_SYNC_FAILED = "clock_sync_failed"
    CLOCK_DRIFT_WARNING = "clock_drift_warning"
    LEI_VALIDATED = "lei_validated"
    LEI_VALIDATION_FAILED = "lei_validation_failed"

    # === OTR (Order-to-Trade Ratio) Events ===
    OTR_WARNING = "otr_warning"
    OTR_BREACH = "otr_breach"
    OTR_THROTTLE_APPLIED = "otr_throttle_applied"

    # === Audit Trail Events ===
    AUDIT_CHAIN_VERIFIED = "audit_chain_verified"
    AUDIT_CHAIN_BREACH = "audit_chain_breach"
    AUDIT_EXPORT_REQUESTED = "audit_export_requested"
    AUDIT_EXPORT_COMPLETED = "audit_export_completed"


class AuditRecordPriority(Enum):
    """Priority levels for audit records (affects processing order)."""

    CRITICAL = "critical"  # Kill switch, system errors
    HIGH = "high"  # Risk breaches, order rejections
    NORMAL = "normal"  # Standard order events
    LOW = "low"  # Market data, routine events


class AuditRecordStatus(Enum):
    """Status of an audit record in the storage pipeline."""

    PENDING = "pending"  # Not yet written to storage
    WRITTEN = "written"  # Written to primary storage
    REPLICATED = "replicated"  # Written to backup storage
    ARCHIVED = "archived"  # Moved to cold storage
    VERIFIED = "verified"  # Integrity verified


class OrderSide(Enum):
    """Order side for trading events."""

    BUY = "BUY"
    SELL = "SELL"


def _default_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def _current_timestamp_ns() -> int:
    """Get current timestamp in nanoseconds."""
    return time.time_ns()


@dataclass
class AuditRecord:
    """
    Immutable audit record per MiFIR Article 25.

    This record captures all relevant data for a single auditable event.
    Records are chained via cryptographic hashes for tamper detection.

    Retention Requirements:
        - Minimum: 5 years from record creation
        - Extended (on NCA request): 7 years
        - Records must be retrievable within reasonable timeframes

    Attributes:
        record_id: Unique identifier for this record (UUID)
        event_type: Type of event being recorded
        event_timestamp_ns: When the event occurred (nanoseconds since epoch)
        record_timestamp_ns: When the record was created (nanoseconds since epoch)
        firm_lei: Legal Entity Identifier of the firm
        algorithm_id: ID of the algorithm (if applicable)
        trader_id: ID of the trader/desk (if applicable)
        order_id: Order identifier (if applicable)
        client_order_id: Client-assigned order ID (if applicable)
        instrument_isin: ISIN of the instrument (if applicable)
        instrument_symbol: Trading symbol (if applicable)
        venue_mic: Market Identifier Code (if applicable)
        side: Order side (BUY/SELL)
        quantity: Order/trade quantity
        price: Order/trade price
        currency: Price/notional currency
        notional_value: Total notional value
        details: Additional event-specific details
        sequence_number: Sequential record number (for ordering)
        priority: Record priority level
        status: Current record status
        previous_record_hash: Hash of the previous record (chain integrity)
        record_hash: Hash of this record
    """

    # === Identification ===
    record_id: str = field(default_factory=_default_uuid)
    event_type: AuditEventType = AuditEventType.SYSTEM_STARTUP

    # === Timestamps (RTS 25 compliant) ===
    event_timestamp_ns: int = field(default_factory=_current_timestamp_ns)
    record_timestamp_ns: int = field(default_factory=_current_timestamp_ns)

    # === Entity Identification ===
    firm_lei: str = ""
    algorithm_id: Optional[str] = None
    trader_id: Optional[str] = None

    # === Order Details (if applicable) ===
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    instrument_isin: Optional[str] = None
    instrument_symbol: Optional[str] = None
    venue_mic: Optional[str] = None
    side: Optional[OrderSide] = None
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None
    currency: Optional[str] = None
    notional_value: Optional[Decimal] = None

    # === Event Details ===
    details: Dict[str, Any] = field(default_factory=dict)

    # === Record Metadata ===
    sequence_number: int = 0
    priority: AuditRecordPriority = AuditRecordPriority.NORMAL
    status: AuditRecordStatus = AuditRecordStatus.PENDING

    # === Chain Integrity ===
    previous_record_hash: Optional[str] = None
    record_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to dictionary for serialization.

        Returns:
            Dictionary representation with all fields.
        """
        result = {}

        # Core fields
        result["record_id"] = self.record_id
        result["event_type"] = self.event_type.value
        result["event_timestamp_ns"] = self.event_timestamp_ns
        result["record_timestamp_ns"] = self.record_timestamp_ns
        result["firm_lei"] = self.firm_lei

        # Optional entity fields
        if self.algorithm_id is not None:
            result["algorithm_id"] = self.algorithm_id
        if self.trader_id is not None:
            result["trader_id"] = self.trader_id

        # Order fields
        if self.order_id is not None:
            result["order_id"] = self.order_id
        if self.client_order_id is not None:
            result["client_order_id"] = self.client_order_id
        if self.instrument_isin is not None:
            result["instrument_isin"] = self.instrument_isin
        if self.instrument_symbol is not None:
            result["instrument_symbol"] = self.instrument_symbol
        if self.venue_mic is not None:
            result["venue_mic"] = self.venue_mic
        if self.side is not None:
            result["side"] = self.side.value
        if self.quantity is not None:
            result["quantity"] = str(self.quantity)
        if self.price is not None:
            result["price"] = str(self.price)
        if self.currency is not None:
            result["currency"] = self.currency
        if self.notional_value is not None:
            result["notional_value"] = str(self.notional_value)

        # Details and metadata
        result["details"] = self.details
        result["sequence_number"] = self.sequence_number
        result["priority"] = self.priority.value
        result["status"] = self.status.value

        # Chain integrity
        if self.previous_record_hash is not None:
            result["previous_record_hash"] = self.previous_record_hash
        if self.record_hash is not None:
            result["record_hash"] = self.record_hash

        return result

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize to JSON string.

        Args:
            indent: JSON indentation level (None for compact).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash for integrity verification.

        The hash is computed over all record fields except the hash itself.
        This is used to detect tampering.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        # Create dict without hash field
        data = self.to_dict()
        data.pop("record_hash", None)

        # Compute hash of canonical JSON
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def validate(self) -> List[str]:
        """
        Validate record integrity and required fields.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Required fields
        if not self.record_id:
            errors.append("Missing record_id")

        if not self.firm_lei:
            errors.append("Missing firm_lei (required per MiFIR)")

        # Timestamp validation
        if self.event_timestamp_ns <= 0:
            errors.append("Invalid event_timestamp_ns")

        if self.record_timestamp_ns <= 0:
            errors.append("Invalid record_timestamp_ns")

        # Event timestamp should not be in the future (allow 5s tolerance)
        future_tolerance_ns = 5_000_000_000  # 5 seconds
        if self.event_timestamp_ns > time.time_ns() + future_tolerance_ns:
            errors.append("event_timestamp_ns is in the future")

        # Hash validation (if set)
        if self.record_hash:
            computed = self.compute_hash()
            if computed != self.record_hash:
                errors.append(f"Hash mismatch: stored={self.record_hash}, computed={computed}")

        # Order event validation
        order_events = {
            AuditEventType.ORDER_SUBMITTED,
            AuditEventType.ORDER_ACKNOWLEDGED,
            AuditEventType.ORDER_MODIFIED,
            AuditEventType.ORDER_CANCELLED,
            AuditEventType.ORDER_REJECTED,
            AuditEventType.ORDER_FILLED,
            AuditEventType.ORDER_PARTIALLY_FILLED,
            AuditEventType.ORDER_EXPIRED,
            AuditEventType.ORDER_REPLACED,
        }

        if self.event_type in order_events:
            if not self.order_id:
                errors.append(f"Order event {self.event_type.value} requires order_id")

        return errors

    def is_valid(self) -> bool:
        """Check if record is valid."""
        return len(self.validate()) == 0

    @property
    def event_datetime(self) -> datetime:
        """Get event timestamp as datetime."""
        return datetime.fromtimestamp(self.event_timestamp_ns / 1e9)

    @property
    def record_datetime(self) -> datetime:
        """Get record creation timestamp as datetime."""
        return datetime.fromtimestamp(self.record_timestamp_ns / 1e9)

    def get_retention_expiry(self, years: int = 5) -> datetime:
        """
        Calculate retention expiry date.

        Args:
            years: Retention period in years (default 5 per MiFIR).

        Returns:
            Datetime when record can be deleted.
        """
        from datetime import timedelta

        return self.record_datetime + timedelta(days=years * 365)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """
        Create AuditRecord from dictionary.

        Args:
            data: Dictionary with record fields.

        Returns:
            New AuditRecord instance.
        """
        # Convert enum strings back to enums
        event_type = AuditEventType(data.get("event_type", "system_startup"))
        priority = AuditRecordPriority(data.get("priority", "normal"))
        status = AuditRecordStatus(data.get("status", "pending"))

        # Handle optional side
        side_str = data.get("side")
        side = OrderSide(side_str) if side_str else None

        # Handle Decimal fields
        quantity = Decimal(data["quantity"]) if data.get("quantity") else None
        price = Decimal(data["price"]) if data.get("price") else None
        notional_value = Decimal(data["notional_value"]) if data.get("notional_value") else None

        return cls(
            record_id=data.get("record_id", _default_uuid()),
            event_type=event_type,
            event_timestamp_ns=data.get("event_timestamp_ns", _current_timestamp_ns()),
            record_timestamp_ns=data.get("record_timestamp_ns", _current_timestamp_ns()),
            firm_lei=data.get("firm_lei", ""),
            algorithm_id=data.get("algorithm_id"),
            trader_id=data.get("trader_id"),
            order_id=data.get("order_id"),
            client_order_id=data.get("client_order_id"),
            instrument_isin=data.get("instrument_isin"),
            instrument_symbol=data.get("instrument_symbol"),
            venue_mic=data.get("venue_mic"),
            side=side,
            quantity=quantity,
            price=price,
            currency=data.get("currency"),
            notional_value=notional_value,
            details=data.get("details", {}),
            sequence_number=data.get("sequence_number", 0),
            priority=priority,
            status=status,
            previous_record_hash=data.get("previous_record_hash"),
            record_hash=data.get("record_hash"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AuditRecord":
        """
        Create AuditRecord from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New AuditRecord instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class AuditRecordBuilder:
    """
    Builder for creating AuditRecord instances with validation.

    This builder provides a fluent interface for constructing audit records
    with proper validation and hash computation.

    Example:
        record = (
            AuditRecordBuilder()
            .event_type(AuditEventType.ORDER_SUBMITTED)
            .firm_lei("5493001KJTIIGC8Y1R12")
            .order_id("ORD-12345")
            .instrument("US0378331005", "AAPL")
            .side(OrderSide.BUY)
            .quantity(Decimal("100"))
            .price(Decimal("150.50"))
            .build()
        )
    """

    _record: AuditRecord = field(default_factory=AuditRecord)

    def event_type(self, event_type: AuditEventType) -> "AuditRecordBuilder":
        """Set the event type."""
        self._record.event_type = event_type
        return self

    def firm_lei(self, lei: str) -> "AuditRecordBuilder":
        """Set the firm LEI."""
        self._record.firm_lei = lei
        return self

    def algorithm(self, algorithm_id: str) -> "AuditRecordBuilder":
        """Set the algorithm ID."""
        self._record.algorithm_id = algorithm_id
        return self

    def trader(self, trader_id: str) -> "AuditRecordBuilder":
        """Set the trader ID."""
        self._record.trader_id = trader_id
        return self

    def order_id(self, order_id: str, client_order_id: Optional[str] = None) -> "AuditRecordBuilder":
        """Set order identifiers."""
        self._record.order_id = order_id
        self._record.client_order_id = client_order_id
        return self

    def instrument(
        self,
        isin: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> "AuditRecordBuilder":
        """Set instrument identification."""
        self._record.instrument_isin = isin
        self._record.instrument_symbol = symbol
        return self

    def venue(self, mic: str) -> "AuditRecordBuilder":
        """Set venue MIC code."""
        self._record.venue_mic = mic
        return self

    def side(self, side: OrderSide) -> "AuditRecordBuilder":
        """Set order side."""
        self._record.side = side
        return self

    def quantity(self, qty: Union[Decimal, float, int, str]) -> "AuditRecordBuilder":
        """Set order quantity."""
        self._record.quantity = Decimal(str(qty))
        return self

    def price(self, price: Union[Decimal, float, int, str]) -> "AuditRecordBuilder":
        """Set order price."""
        self._record.price = Decimal(str(price))
        return self

    def currency(self, currency: str) -> "AuditRecordBuilder":
        """Set price/notional currency."""
        self._record.currency = currency
        return self

    def notional(self, notional: Union[Decimal, float, int, str]) -> "AuditRecordBuilder":
        """Set notional value."""
        self._record.notional_value = Decimal(str(notional))
        return self

    def details(self, details: Dict[str, Any]) -> "AuditRecordBuilder":
        """Set additional event details."""
        self._record.details = details
        return self

    def add_detail(self, key: str, value: Any) -> "AuditRecordBuilder":
        """Add a single detail field."""
        self._record.details[key] = value
        return self

    def priority(self, priority: AuditRecordPriority) -> "AuditRecordBuilder":
        """Set record priority."""
        self._record.priority = priority
        return self

    def timestamp(self, timestamp_ns: int) -> "AuditRecordBuilder":
        """Set event timestamp (nanoseconds)."""
        self._record.event_timestamp_ns = timestamp_ns
        return self

    def timestamp_ms(self, timestamp_ms: int) -> "AuditRecordBuilder":
        """Set event timestamp (milliseconds)."""
        self._record.event_timestamp_ns = timestamp_ms * 1_000_000
        return self

    def timestamp_datetime(self, dt: datetime) -> "AuditRecordBuilder":
        """Set event timestamp from datetime."""
        self._record.event_timestamp_ns = int(dt.timestamp() * 1e9)
        return self

    def previous_hash(self, hash_value: str) -> "AuditRecordBuilder":
        """Set previous record hash for chain integrity."""
        self._record.previous_record_hash = hash_value
        return self

    def sequence(self, seq: int) -> "AuditRecordBuilder":
        """Set sequence number."""
        self._record.sequence_number = seq
        return self

    def build(self, compute_hash: bool = True, validate: bool = True) -> AuditRecord:
        """
        Build the audit record.

        Args:
            compute_hash: Whether to compute record hash.
            validate: Whether to validate the record.

        Returns:
            Completed AuditRecord.

        Raises:
            ValueError: If validation fails.
        """
        # Set record timestamp
        self._record.record_timestamp_ns = time.time_ns()

        # Compute hash if requested
        if compute_hash:
            self._record.record_hash = self._record.compute_hash()

        # Validate if requested
        if validate:
            errors = self._record.validate()
            if errors:
                raise ValueError(f"Record validation failed: {errors}")

        return self._record

    def build_unsafe(self) -> AuditRecord:
        """Build without validation (for testing only)."""
        return self.build(compute_hash=True, validate=False)


@dataclass
class AuditChainStatus:
    """Status of audit chain integrity verification."""

    is_valid: bool
    records_checked: int
    first_invalid_record_id: Optional[str] = None
    first_invalid_index: Optional[int] = None
    error_message: Optional[str] = None
    verification_timestamp_ns: int = field(default_factory=_current_timestamp_ns)


@dataclass
class AuditExportRequest:
    """Request for exporting audit records (e.g., for NCA request)."""

    request_id: str = field(default_factory=_default_uuid)
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    order_ids: Optional[List[str]] = None
    algorithm_ids: Optional[List[str]] = None
    instrument_isins: Optional[List[str]] = None
    export_format: str = "json"  # json, csv, xml
    include_chain_verification: bool = True
    requested_by: str = ""
    request_reason: str = ""
    request_timestamp_ns: int = field(default_factory=_current_timestamp_ns)


@dataclass
class AuditExportResult:
    """Result of an audit export operation."""

    request_id: str
    success: bool
    records_exported: int
    export_path: Optional[str] = None
    error_message: Optional[str] = None
    chain_verification: Optional[AuditChainStatus] = None
    export_timestamp_ns: int = field(default_factory=_current_timestamp_ns)


def create_order_submitted_record(
    firm_lei: str,
    order_id: str,
    instrument_isin: str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    currency: str = "EUR",
    venue_mic: str = "XOFF",
    algorithm_id: Optional[str] = None,
    trader_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """
    Factory function for ORDER_SUBMITTED audit records.

    Args:
        firm_lei: Firm's LEI.
        order_id: Unique order identifier.
        instrument_isin: Instrument ISIN.
        side: Order side (BUY/SELL).
        quantity: Order quantity.
        price: Order price.
        currency: Price currency (default EUR).
        venue_mic: Trading venue MIC (default XOFF).
        algorithm_id: Algorithm ID (if algorithmic order).
        trader_id: Trader ID.
        client_order_id: Client-assigned order ID.
        details: Additional details.

    Returns:
        AuditRecord for order submission.
    """
    builder = (
        AuditRecordBuilder()
        .event_type(AuditEventType.ORDER_SUBMITTED)
        .firm_lei(firm_lei)
        .order_id(order_id, client_order_id)
        .instrument(isin=instrument_isin)
        .side(side)
        .quantity(quantity)
        .price(price)
        .currency(currency)
        .venue(venue_mic)
        .priority(AuditRecordPriority.NORMAL)
    )

    if algorithm_id:
        builder.algorithm(algorithm_id)
    if trader_id:
        builder.trader(trader_id)
    if details:
        builder.details(details)

    return builder.build()


def create_order_filled_record(
    firm_lei: str,
    order_id: str,
    instrument_isin: str,
    side: OrderSide,
    fill_quantity: Decimal,
    fill_price: Decimal,
    currency: str = "EUR",
    venue_mic: str = "XOFF",
    is_partial: bool = False,
    remaining_quantity: Optional[Decimal] = None,
    algorithm_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """
    Factory function for ORDER_FILLED/ORDER_PARTIALLY_FILLED audit records.

    Args:
        firm_lei: Firm's LEI.
        order_id: Order identifier.
        instrument_isin: Instrument ISIN.
        side: Order side.
        fill_quantity: Filled quantity.
        fill_price: Fill price.
        currency: Price currency.
        venue_mic: Trading venue MIC.
        is_partial: Whether this is a partial fill.
        remaining_quantity: Remaining quantity (for partial fills).
        algorithm_id: Algorithm ID.
        details: Additional details.

    Returns:
        AuditRecord for order fill.
    """
    event_type = AuditEventType.ORDER_PARTIALLY_FILLED if is_partial else AuditEventType.ORDER_FILLED

    fill_details = details.copy() if details else {}
    if is_partial and remaining_quantity is not None:
        fill_details["remaining_quantity"] = str(remaining_quantity)

    builder = (
        AuditRecordBuilder()
        .event_type(event_type)
        .firm_lei(firm_lei)
        .order_id(order_id)
        .instrument(isin=instrument_isin)
        .side(side)
        .quantity(fill_quantity)
        .price(fill_price)
        .currency(currency)
        .venue(venue_mic)
        .notional(fill_quantity * fill_price)
        .details(fill_details)
        .priority(AuditRecordPriority.NORMAL)
    )

    if algorithm_id:
        builder.algorithm(algorithm_id)

    return builder.build()


def create_risk_event_record(
    firm_lei: str,
    event_type: AuditEventType,
    reason: str,
    algorithm_id: Optional[str] = None,
    order_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """
    Factory function for risk-related audit records.

    Args:
        firm_lei: Firm's LEI.
        event_type: Risk event type.
        reason: Reason for the risk event.
        algorithm_id: Algorithm ID.
        order_id: Related order ID.
        details: Additional details.

    Returns:
        AuditRecord for risk event.
    """
    event_details = details.copy() if details else {}
    event_details["reason"] = reason

    # Determine priority based on event type
    priority = AuditRecordPriority.NORMAL
    if event_type in {
        AuditEventType.KILL_SWITCH_TRIGGERED,
        AuditEventType.POSITION_LIMIT_BREACH,
        AuditEventType.PNL_LIMIT_BREACH,
    }:
        priority = AuditRecordPriority.CRITICAL
    elif event_type in {
        AuditEventType.RISK_CHECK_FAILED,
        AuditEventType.PRE_TRADE_CHECK_FAILED,
        AuditEventType.POSITION_LIMIT_WARNING,
    }:
        priority = AuditRecordPriority.HIGH

    builder = (
        AuditRecordBuilder()
        .event_type(event_type)
        .firm_lei(firm_lei)
        .details(event_details)
        .priority(priority)
    )

    if algorithm_id:
        builder.algorithm(algorithm_id)
    if order_id:
        builder.order_id(order_id)

    return builder.build()


def create_system_event_record(
    firm_lei: str,
    event_type: AuditEventType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    """
    Factory function for system-level audit records.

    Args:
        firm_lei: Firm's LEI.
        event_type: System event type.
        message: Event message/description.
        details: Additional details.

    Returns:
        AuditRecord for system event.
    """
    event_details = details.copy() if details else {}
    event_details["message"] = message

    priority = AuditRecordPriority.NORMAL
    if event_type in {
        AuditEventType.SYSTEM_ERROR,
        AuditEventType.CONNECTION_LOST,
    }:
        priority = AuditRecordPriority.HIGH
    elif event_type in {
        AuditEventType.SYSTEM_STARTUP,
        AuditEventType.SYSTEM_SHUTDOWN,
    }:
        priority = AuditRecordPriority.CRITICAL

    return (
        AuditRecordBuilder()
        .event_type(event_type)
        .firm_lei(firm_lei)
        .details(event_details)
        .priority(priority)
        .build()
    )


__all__ = [
    # Enums
    "AuditEventType",
    "AuditRecordPriority",
    "AuditRecordStatus",
    "OrderSide",
    # Data classes
    "AuditRecord",
    "AuditRecordBuilder",
    "AuditChainStatus",
    "AuditExportRequest",
    "AuditExportResult",
    # Factory functions
    "create_order_submitted_record",
    "create_order_filled_record",
    "create_risk_event_record",
    "create_system_event_record",
]
