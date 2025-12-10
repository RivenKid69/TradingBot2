# -*- coding: utf-8 -*-
"""
MiFIR Article 26 Transaction Report Data Model (RTS 22).

Implements the complete 65-field transaction report format as specified
in Commission Delegated Regulation (EU) 2017/590 (RTS 22).

Transaction reports must be submitted to National Competent Authorities (NCAs)
via Approved Reporting Mechanisms (ARMs) within T+1.

Report Structure (RTS 22 Annex I Table 2):
    - Fields 1-6: Report identification
    - Fields 7-16: Buyer identification
    - Fields 17-26: Seller identification
    - Fields 27-29: Transmission
    - Fields 30-35: Trading decision/execution
    - Fields 36-40: Order details
    - Fields 41-56: Transaction details
    - Fields 57-65: Instrument identification

References:
    - RTS 22 (Regulation 2017/590): Transaction Reporting
    - ESMA Guidelines: https://www.esma.europa.eu/sites/default/files/library/esma65-8-2356_mifir_transaction_reporting_technical_reporting_instructions.pdf
    - MiFIR Article 26: Obligation to report transactions
"""

from __future__ import annotations

import re
import logging
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


# ============================================================================
# RTS 22 Enumerations
# ============================================================================


class BuySellIndicator(str, Enum):
    """
    Transaction side indicator per RTS 22 Field 63.

    Values:
        BUYI: Buy transaction
        SELL: Sell transaction
    """
    BUY = "BUYI"
    SELL = "SELL"


class TradingCapacity(str, Enum):
    """
    Trading capacity per RTS 22 Field 29.

    Values:
        DEAL: Dealing on own account (principal)
        MTCH: Matched principal trading
        AOTC: Any other trading capacity (agency)
    """
    DEAL = "DEAL"  # Dealing on own account
    MTCH = "MTCH"  # Matched principal
    AOTC = "AOTC"  # Any other trading capacity


class IdentifierType(str, Enum):
    """
    Party identifier type per RTS 22.

    Values:
        LEI: Legal Entity Identifier (ISO 17442)
        MIC: Market Identifier Code (ISO 10383)
        NATIONAL_ID: National ID with country code
        INTC: Internal client code
        NONE: No identifier available
    """
    LEI = "LEI"
    MIC = "MIC"
    NATIONAL_ID = "NATIONAL_ID"
    INTC = "INTC"  # Internal client
    NONE = "NONE"


class InstrumentIdentifierType(str, Enum):
    """
    Instrument identifier type per RTS 22 Field 41.

    Values:
        ISIN: International Securities Identification Number (ISO 6166)
    """
    ISIN = "ISIN"


class PriceType(str, Enum):
    """
    Price type per RTS 22 Field 33.

    Values:
        MONE: Monetary value (actual price)
        PERC: Percentage
        YIEL: Yield
        BPNT: Basis points
    """
    MONETARY = "MONE"
    PERCENTAGE = "PERC"
    YIELD = "YIEL"
    BASIS_POINTS = "BPNT"


class QuantityType(str, Enum):
    """
    Quantity type per RTS 22 Field 30.

    Values:
        UNIT: Units/shares
        NOML: Nominal value
        MONE: Monetary value
    """
    UNITS = "UNIT"
    NOMINAL = "NOML"
    MONETARY = "MONE"


class TransactionType(str, Enum):
    """
    Transaction type indicator.
    """
    NEW = "NEW"
    CANCEL = "CANC"
    AMENDMENT = "AMND"


class ReportStatus(str, Enum):
    """
    Report lifecycle status.
    """
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


# ============================================================================
# Validation Helpers
# ============================================================================


class ISINValidator:
    """
    ISIN (International Securities Identification Number) validator per ISO 6166.

    ISIN Format:
        - 2-letter country code (ISO 3166-1 alpha-2)
        - 9 alphanumeric characters (NSIN)
        - 1 check digit (Luhn algorithm)
    """

    ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

    @classmethod
    def validate_format(cls, isin: str) -> bool:
        """Validate ISIN format."""
        if not isin or not isinstance(isin, str):
            return False
        return bool(cls.ISIN_PATTERN.match(isin.upper()))

    @classmethod
    def validate_checksum(cls, isin: str) -> bool:
        """
        Validate ISIN check digit using Luhn algorithm.

        The algorithm converts letters to numbers (A=10, B=11, ..., Z=35)
        then applies the Luhn checksum.
        """
        if not cls.validate_format(isin):
            return False

        isin = isin.upper()

        # Convert letters to numbers
        digits = ""
        for char in isin:
            if char.isalpha():
                digits += str(ord(char) - ord('A') + 10)
            else:
                digits += char

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(digits)):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    @classmethod
    def validate(cls, isin: str) -> Tuple[bool, str]:
        """
        Full ISIN validation.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not isin:
            return (False, "ISIN is required")

        isin = isin.upper().strip()

        if not cls.validate_format(isin):
            return (False, f"Invalid ISIN format: {isin}")

        if not cls.validate_checksum(isin):
            return (False, f"Invalid ISIN checksum: {isin}")

        return (True, "")


class MICValidator:
    """
    MIC (Market Identifier Code) validator per ISO 10383.

    MIC Format:
        - 4 alphanumeric characters
        - First character must be a letter

    Special Values:
        - XOFF: Off-exchange transaction
        - XXXX: No venue (systematic internaliser)
    """

    MIC_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{3}$")

    # Common valid MIC codes for EU venues
    KNOWN_MICS = {
        # Major EU exchanges
        "XLON", "XPAR", "XETR", "XAMS", "XBRU", "XLIS", "XMIL", "XMAD",
        "XWBO", "XHEL", "XCSE", "XSTO", "XOSL", "XWAR", "XPRA", "XBUD",
        "XATH", "XDUB", "XLUX", "XICE",
        # MTFs
        "BATE", "CHIX", "TRQX", "BOAT", "TQEX",
        # Special codes
        "XOFF",  # Off-exchange
        "XXXX",  # No venue (SI)
        "SINT",  # Systematic internaliser
    }

    @classmethod
    def validate_format(cls, mic: str) -> bool:
        """Validate MIC format."""
        if not mic or not isinstance(mic, str):
            return False
        return bool(cls.MIC_PATTERN.match(mic.upper()))

    @classmethod
    def validate(cls, mic: str, strict: bool = False) -> Tuple[bool, str]:
        """
        Validate MIC code.

        Args:
            mic: MIC code to validate.
            strict: If True, only accept known MICs.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not mic:
            return (False, "MIC is required")

        mic = mic.upper().strip()

        if not cls.validate_format(mic):
            return (False, f"Invalid MIC format: {mic}")

        if strict and mic not in cls.KNOWN_MICS:
            return (False, f"Unknown MIC code: {mic}")

        return (True, "")


class CFIValidator:
    """
    CFI (Classification of Financial Instruments) validator per ISO 10962.

    CFI Format:
        - 6 characters
        - Each position represents a classification category
    """

    CFI_PATTERN = re.compile(r"^[A-Z]{6}$")

    # First character categories
    CATEGORIES = {
        "E": "Equities",
        "D": "Debt instruments",
        "R": "Entitlements (rights)",
        "O": "Options",
        "F": "Futures",
        "S": "Swaps",
        "H": "Non-listed complex instruments",
        "I": "Spot",
        "J": "Forwards",
        "K": "Strategies",
        "L": "Financing",
        "T": "Referential instruments",
        "M": "Others",
    }

    @classmethod
    def validate_format(cls, cfi: str) -> bool:
        """Validate CFI format."""
        if not cfi or not isinstance(cfi, str):
            return False
        return bool(cls.CFI_PATTERN.match(cfi.upper()))

    @classmethod
    def validate(cls, cfi: str) -> Tuple[bool, str]:
        """
        Validate CFI code.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not cfi:
            return (True, "")  # CFI is optional

        cfi = cfi.upper().strip()

        if not cls.validate_format(cfi):
            return (False, f"Invalid CFI format: {cfi}")

        if cfi[0] not in cls.CATEGORIES:
            return (False, f"Unknown CFI category: {cfi[0]}")

        return (True, "")


# ============================================================================
# Transaction Report Data Model
# ============================================================================


@dataclass
class TransactionReportParty:
    """
    Party identification for transaction report (buyer/seller).

    Per RTS 22, parties can be identified by:
        - LEI (for legal entities)
        - National ID (for natural persons with country code)
        - Internal code (for non-EEA clients)
    """

    identifier: str = ""
    identifier_type: IdentifierType = IdentifierType.LEI
    country: str = ""  # ISO 3166-1 alpha-2

    # Natural person details (if applicable)
    first_name: str = ""
    surname: str = ""
    date_of_birth: Optional[date] = None

    # Decision maker (if different from party)
    decision_maker_id: str = ""
    decision_maker_id_type: IdentifierType = IdentifierType.LEI

    def is_legal_entity(self) -> bool:
        """Check if party is a legal entity (vs natural person)."""
        return self.identifier_type == IdentifierType.LEI

    def validate(self) -> List[str]:
        """Validate party data. Returns list of errors."""
        errors = []

        if not self.identifier and self.identifier_type != IdentifierType.NONE:
            errors.append("Party identifier is required")

        if self.identifier_type == IdentifierType.LEI:
            # LEI format: 20 characters
            if self.identifier and len(self.identifier) != 20:
                errors.append(f"Invalid LEI length: {len(self.identifier)}")

        if self.identifier_type == IdentifierType.NATIONAL_ID and not self.country:
            errors.append("Country code required for national ID")

        if self.country and len(self.country) != 2:
            errors.append(f"Invalid country code: {self.country}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "identifier": self.identifier,
            "identifier_type": self.identifier_type.value,
            "country": self.country,
            "first_name": self.first_name,
            "surname": self.surname,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "decision_maker_id": self.decision_maker_id,
            "decision_maker_id_type": self.decision_maker_id_type.value,
        }


@dataclass
class TransactionReport:
    """
    MiFIR Article 26 Transaction Report per RTS 22.

    Complete 65-field transaction report for submission to NCAs via ARM.

    Field Groups:
        1. Report Identification (Fields 1-6)
        2. Buyer Details (Fields 7-16)
        3. Seller Details (Fields 17-26)
        4. Transmission (Fields 27-29)
        5. Trading Decision/Execution (Fields 30-35)
        6. Order Details (Fields 36-40)
        7. Transaction Details (Fields 41-56)
        8. Instrument Identification (Fields 57-65)

    Example:
        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("100"),
            price=Decimal("150.25"),
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
            buy_sell_indicator=BuySellIndicator.BUY,
        )

        errors = report.validate()
        if not errors:
            xml_str = report.to_xml()
    """

    # === Report Identification (Fields 1-6) ===
    transaction_reference_number: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    """Field 1: Unique transaction reference number."""

    trading_venue_transaction_id: str = ""
    """Field 2: Trading venue's transaction identifier."""

    executing_entity_lei: str = ""
    """Field 3: LEI of the executing investment firm (required)."""

    submitting_entity_lei: str = ""
    """Field 4: LEI of the submitting entity (if different from executing)."""

    investment_firm_covered: bool = True
    """Field 5: Whether the transaction is covered by MiFIR."""

    transaction_type: TransactionType = TransactionType.NEW
    """Field 6: NEW, CANCEL, or AMENDMENT."""

    # === Buyer Identification (Fields 7-16) ===
    buyer: TransactionReportParty = field(default_factory=TransactionReportParty)
    """Fields 7-16: Buyer identification and details."""

    # === Seller Identification (Fields 17-26) ===
    seller: TransactionReportParty = field(default_factory=TransactionReportParty)
    """Fields 17-26: Seller identification and details."""

    # === Transmission (Fields 27-29) ===
    transmission_indicator: bool = False
    """Field 27: Order transmitted to another firm for execution."""

    transmitting_firm_lei: str = ""
    """Field 28: LEI of firm transmitting the order."""

    trading_capacity: TradingCapacity = TradingCapacity.DEAL
    """Field 29: Trading capacity (DEAL/MTCH/AOTC)."""

    # === Trading Decision/Execution (Fields 30-35) ===
    quantity: Decimal = Decimal("0")
    """Field 30: Quantity of units traded."""

    quantity_type: QuantityType = QuantityType.UNITS
    """Field 30 (qualifier): Type of quantity."""

    quantity_currency: str = ""
    """Field 30 (qualifier): Currency if quantity is monetary."""

    derivative_notional_increase: Optional[Decimal] = None
    """Field 31: Notional increase for derivative contracts."""

    derivative_notional_decrease: Optional[Decimal] = None
    """Field 32: Notional decrease for derivative contracts."""

    price: Decimal = Decimal("0")
    """Field 33: Price per unit."""

    price_type: PriceType = PriceType.MONETARY
    """Field 33 (qualifier): Type of price."""

    price_currency: str = ""
    """Field 34: Currency of the price (ISO 4217)."""

    net_amount: Decimal = Decimal("0")
    """Field 35: Net amount of the transaction (price * quantity)."""

    # === Order Details (Fields 36-40) ===
    venue_mic: str = ""
    """Field 36: MIC of the execution venue."""

    country_of_branch: str = ""
    """Field 37: Country of the branch supervising the trader."""

    upfront_payment: Optional[Decimal] = None
    """Field 38: Upfront payment for derivative contracts."""

    upfront_payment_currency: str = ""
    """Field 39: Currency of upfront payment."""

    complex_trade_component_id: str = ""
    """Field 40: ID linking components of complex trades."""

    # === Transaction Details (Fields 41-56) ===
    instrument_isin: str = ""
    """Field 41: ISIN of the traded instrument (required)."""

    instrument_full_name: str = ""
    """Field 42: Full name of the instrument."""

    instrument_classification: str = ""
    """Field 43: CFI code (ISO 10962)."""

    notional_currency_1: str = ""
    """Field 44: First notional currency for derivatives."""

    notional_currency_2: str = ""
    """Field 45: Second notional currency for FX derivatives."""

    price_multiplier: Decimal = Decimal("1")
    """Field 46: Price multiplier for derivatives."""

    underlying_instrument_isin: str = ""
    """Field 47: ISIN of the underlying instrument."""

    underlying_index_name: str = ""
    """Field 48: Name of the underlying index."""

    underlying_index_term_value: Optional[int] = None
    """Field 49: Term of the underlying index (value)."""

    underlying_index_term_unit: str = ""
    """Field 50: Term of the underlying index (unit)."""

    option_type: str = ""
    """Field 51: Option type (CALL/PUT)."""

    strike_price: Optional[Decimal] = None
    """Field 52: Strike price for options."""

    strike_price_currency: str = ""
    """Field 53: Currency of strike price."""

    option_exercise_style: str = ""
    """Field 54: EUROPEAN/AMERICAN/ASIAN/BERMUDAN."""

    maturity_date: Optional[date] = None
    """Field 55: Maturity date of the instrument."""

    expiry_date: Optional[date] = None
    """Field 56: Expiry date for options/warrants."""

    # === Instrument Identification (Fields 57-65) ===
    delivery_type: str = ""
    """Field 57: CASH/PHYSICAL for derivatives."""

    trading_datetime: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    """Field 58: Date and time of trade execution (UTC, microsecond precision)."""

    report_status: ReportStatus = ReportStatus.PENDING
    """Field 59 (internal): Status of the report."""

    trading_decision_maker_id: str = ""
    """Field 60: ID of trading decision maker (if algorithm: ALGO)."""

    trading_decision_maker_id_type: str = "ALGO"
    """Field 60 (qualifier): Type of decision maker ID."""

    execution_decision_maker_id: str = ""
    """Field 61: ID of execution decision maker."""

    execution_decision_maker_id_type: str = "ALGO"
    """Field 61 (qualifier): Type of execution decision maker ID."""

    buy_sell_indicator: BuySellIndicator = BuySellIndicator.BUY
    """Field 62: Buy or sell transaction."""

    waiver_indicator: str = ""
    """Field 63: Pre-trade waiver indicator."""

    short_selling_indicator: str = ""
    """Field 64: SESH (short)/SSEX (short exempt)/SELL/UNDI."""

    otc_post_trade_indicator: str = ""
    """Field 65: Post-trade transparency flags."""

    # === Internal Fields (not submitted) ===
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Internal: Report creation timestamp."""

    submitted_at: Optional[datetime] = None
    """Internal: Report submission timestamp."""

    arm_confirmation_id: str = ""
    """Internal: Confirmation ID from ARM."""

    rejection_reason: str = ""
    """Internal: Reason for rejection (if rejected)."""

    amendment_reason: str = ""
    """Internal: Reason for amendment (if amended)."""

    original_report_id: str = ""
    """Internal: ID of original report (for cancellations/amendments)."""

    def validate(self) -> List[str]:
        """
        Validate all report fields per RTS 22 requirements.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # === Required Fields ===

        # Field 1: Transaction reference
        if not self.transaction_reference_number:
            errors.append("Field 1: Transaction reference number is required")

        # Field 3: Executing entity LEI
        if not self.executing_entity_lei:
            errors.append("Field 3: Executing entity LEI is required")
        elif len(self.executing_entity_lei) != 20:
            errors.append(f"Field 3: Invalid LEI length: {len(self.executing_entity_lei)}")

        # Field 41: Instrument ISIN
        if not self.instrument_isin:
            errors.append("Field 41: Instrument ISIN is required")
        else:
            isin_valid, isin_error = ISINValidator.validate(self.instrument_isin)
            if not isin_valid:
                errors.append(f"Field 41: {isin_error}")

        # Field 30: Quantity
        if self.quantity <= 0:
            errors.append("Field 30: Quantity must be positive")

        # Field 33: Price
        if self.price < 0:
            errors.append("Field 33: Price cannot be negative")

        # Field 36: Venue MIC
        if not self.venue_mic:
            errors.append("Field 36: Venue MIC is required")
        else:
            mic_valid, mic_error = MICValidator.validate(self.venue_mic)
            if not mic_valid:
                errors.append(f"Field 36: {mic_error}")

        # Field 58: Trading datetime
        if not self.trading_datetime:
            errors.append("Field 58: Trading datetime is required")

        # Field 34: Price currency
        if self.price_type == PriceType.MONETARY and not self.price_currency:
            errors.append("Field 34: Price currency required for monetary price")
        elif self.price_currency and len(self.price_currency) != 3:
            errors.append(f"Field 34: Invalid currency code: {self.price_currency}")

        # === Conditional Validations ===

        # CFI validation (if provided)
        if self.instrument_classification:
            cfi_valid, cfi_error = CFIValidator.validate(self.instrument_classification)
            if not cfi_valid:
                errors.append(f"Field 43: {cfi_error}")

        # Buyer/Seller validation
        buyer_errors = self.buyer.validate()
        for err in buyer_errors:
            errors.append(f"Buyer: {err}")

        seller_errors = self.seller.validate()
        for err in seller_errors:
            errors.append(f"Seller: {err}")

        # Transmission consistency
        if self.transmission_indicator and not self.transmitting_firm_lei:
            errors.append("Field 28: Transmitting firm LEI required when transmission indicator is true")

        # Cancellation/amendment must reference original
        if self.transaction_type in (TransactionType.CANCEL, TransactionType.AMENDMENT):
            if not self.original_report_id:
                errors.append("Original report ID required for cancellation/amendment")

        return errors

    def is_valid(self) -> bool:
        """Check if report is valid."""
        return len(self.validate()) == 0

    def calculate_net_amount(self) -> Decimal:
        """Calculate net amount from price and quantity."""
        return self.price * self.quantity * self.price_multiplier

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary for JSON serialization.

        Returns:
            Dictionary representation of all fields.
        """
        def decimal_to_str(d: Optional[Decimal]) -> Optional[str]:
            return str(d) if d is not None else None

        return {
            # Report identification
            "transaction_reference_number": self.transaction_reference_number,
            "trading_venue_transaction_id": self.trading_venue_transaction_id,
            "executing_entity_lei": self.executing_entity_lei,
            "submitting_entity_lei": self.submitting_entity_lei,
            "investment_firm_covered": self.investment_firm_covered,
            "transaction_type": self.transaction_type.value,

            # Buyer/Seller
            "buyer": self.buyer.to_dict(),
            "seller": self.seller.to_dict(),

            # Transmission
            "transmission_indicator": self.transmission_indicator,
            "transmitting_firm_lei": self.transmitting_firm_lei,
            "trading_capacity": self.trading_capacity.value,

            # Trading decision
            "quantity": decimal_to_str(self.quantity),
            "quantity_type": self.quantity_type.value,
            "quantity_currency": self.quantity_currency,
            "derivative_notional_increase": decimal_to_str(self.derivative_notional_increase),
            "derivative_notional_decrease": decimal_to_str(self.derivative_notional_decrease),
            "price": decimal_to_str(self.price),
            "price_type": self.price_type.value,
            "price_currency": self.price_currency,
            "net_amount": decimal_to_str(self.net_amount),

            # Order details
            "venue_mic": self.venue_mic,
            "country_of_branch": self.country_of_branch,
            "upfront_payment": decimal_to_str(self.upfront_payment),
            "upfront_payment_currency": self.upfront_payment_currency,
            "complex_trade_component_id": self.complex_trade_component_id,

            # Instrument
            "instrument_isin": self.instrument_isin,
            "instrument_full_name": self.instrument_full_name,
            "instrument_classification": self.instrument_classification,
            "notional_currency_1": self.notional_currency_1,
            "notional_currency_2": self.notional_currency_2,
            "price_multiplier": decimal_to_str(self.price_multiplier),
            "underlying_instrument_isin": self.underlying_instrument_isin,
            "underlying_index_name": self.underlying_index_name,
            "underlying_index_term_value": self.underlying_index_term_value,
            "underlying_index_term_unit": self.underlying_index_term_unit,
            "option_type": self.option_type,
            "strike_price": decimal_to_str(self.strike_price),
            "strike_price_currency": self.strike_price_currency,
            "option_exercise_style": self.option_exercise_style,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,

            # Additional
            "delivery_type": self.delivery_type,
            "trading_datetime": self.trading_datetime.isoformat() if self.trading_datetime else None,
            "report_status": self.report_status.value,
            "trading_decision_maker_id": self.trading_decision_maker_id,
            "trading_decision_maker_id_type": self.trading_decision_maker_id_type,
            "execution_decision_maker_id": self.execution_decision_maker_id,
            "execution_decision_maker_id_type": self.execution_decision_maker_id_type,
            "buy_sell_indicator": self.buy_sell_indicator.value,
            "waiver_indicator": self.waiver_indicator,
            "short_selling_indicator": self.short_selling_indicator,
            "otc_post_trade_indicator": self.otc_post_trade_indicator,

            # Internal
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "arm_confirmation_id": self.arm_confirmation_id,
            "rejection_reason": self.rejection_reason,
            "amendment_reason": self.amendment_reason,
            "original_report_id": self.original_report_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionReport":
        """
        Create TransactionReport from dictionary.

        Args:
            data: Dictionary with report fields.

        Returns:
            TransactionReport instance.
        """
        def parse_decimal(value: Any) -> Decimal:
            if value is None:
                return Decimal("0")
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return Decimal("0")

        def parse_optional_decimal(value: Any) -> Optional[Decimal]:
            if value is None:
                return None
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None

        def parse_datetime(value: Any) -> Optional[datetime]:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return None

        def parse_date(value: Any) -> Optional[date]:
            if isinstance(value, date):
                return value
            if isinstance(value, str):
                return date.fromisoformat(value[:10])
            return None

        def parse_enum(enum_cls, value: Any, default):
            try:
                return enum_cls(value)
            except (ValueError, TypeError):
                return default

        # Parse buyer/seller
        buyer_data = data.get("buyer", {})
        buyer = TransactionReportParty(
            identifier=buyer_data.get("identifier", ""),
            identifier_type=parse_enum(IdentifierType, buyer_data.get("identifier_type"), IdentifierType.LEI),
            country=buyer_data.get("country", ""),
            first_name=buyer_data.get("first_name", ""),
            surname=buyer_data.get("surname", ""),
            date_of_birth=parse_date(buyer_data.get("date_of_birth")),
            decision_maker_id=buyer_data.get("decision_maker_id", ""),
            decision_maker_id_type=parse_enum(IdentifierType, buyer_data.get("decision_maker_id_type"), IdentifierType.LEI),
        )

        seller_data = data.get("seller", {})
        seller = TransactionReportParty(
            identifier=seller_data.get("identifier", ""),
            identifier_type=parse_enum(IdentifierType, seller_data.get("identifier_type"), IdentifierType.LEI),
            country=seller_data.get("country", ""),
            first_name=seller_data.get("first_name", ""),
            surname=seller_data.get("surname", ""),
            date_of_birth=parse_date(seller_data.get("date_of_birth")),
            decision_maker_id=seller_data.get("decision_maker_id", ""),
            decision_maker_id_type=parse_enum(IdentifierType, seller_data.get("decision_maker_id_type"), IdentifierType.LEI),
        )

        return cls(
            transaction_reference_number=data.get("transaction_reference_number", str(uuid.uuid4())),
            trading_venue_transaction_id=data.get("trading_venue_transaction_id", ""),
            executing_entity_lei=data.get("executing_entity_lei", ""),
            submitting_entity_lei=data.get("submitting_entity_lei", ""),
            investment_firm_covered=data.get("investment_firm_covered", True),
            transaction_type=parse_enum(TransactionType, data.get("transaction_type"), TransactionType.NEW),
            buyer=buyer,
            seller=seller,
            transmission_indicator=data.get("transmission_indicator", False),
            transmitting_firm_lei=data.get("transmitting_firm_lei", ""),
            trading_capacity=parse_enum(TradingCapacity, data.get("trading_capacity"), TradingCapacity.DEAL),
            quantity=parse_decimal(data.get("quantity")),
            quantity_type=parse_enum(QuantityType, data.get("quantity_type"), QuantityType.UNITS),
            quantity_currency=data.get("quantity_currency", ""),
            derivative_notional_increase=parse_optional_decimal(data.get("derivative_notional_increase")),
            derivative_notional_decrease=parse_optional_decimal(data.get("derivative_notional_decrease")),
            price=parse_decimal(data.get("price")),
            price_type=parse_enum(PriceType, data.get("price_type"), PriceType.MONETARY),
            price_currency=data.get("price_currency", ""),
            net_amount=parse_decimal(data.get("net_amount")),
            venue_mic=data.get("venue_mic", ""),
            country_of_branch=data.get("country_of_branch", ""),
            upfront_payment=parse_optional_decimal(data.get("upfront_payment")),
            upfront_payment_currency=data.get("upfront_payment_currency", ""),
            complex_trade_component_id=data.get("complex_trade_component_id", ""),
            instrument_isin=data.get("instrument_isin", ""),
            instrument_full_name=data.get("instrument_full_name", ""),
            instrument_classification=data.get("instrument_classification", ""),
            notional_currency_1=data.get("notional_currency_1", ""),
            notional_currency_2=data.get("notional_currency_2", ""),
            price_multiplier=parse_decimal(data.get("price_multiplier", "1")),
            underlying_instrument_isin=data.get("underlying_instrument_isin", ""),
            underlying_index_name=data.get("underlying_index_name", ""),
            underlying_index_term_value=data.get("underlying_index_term_value"),
            underlying_index_term_unit=data.get("underlying_index_term_unit", ""),
            option_type=data.get("option_type", ""),
            strike_price=parse_optional_decimal(data.get("strike_price")),
            strike_price_currency=data.get("strike_price_currency", ""),
            option_exercise_style=data.get("option_exercise_style", ""),
            maturity_date=parse_date(data.get("maturity_date")),
            expiry_date=parse_date(data.get("expiry_date")),
            delivery_type=data.get("delivery_type", ""),
            trading_datetime=parse_datetime(data.get("trading_datetime")) or datetime.now(timezone.utc),
            report_status=parse_enum(ReportStatus, data.get("report_status"), ReportStatus.PENDING),
            trading_decision_maker_id=data.get("trading_decision_maker_id", ""),
            trading_decision_maker_id_type=data.get("trading_decision_maker_id_type", "ALGO"),
            execution_decision_maker_id=data.get("execution_decision_maker_id", ""),
            execution_decision_maker_id_type=data.get("execution_decision_maker_id_type", "ALGO"),
            buy_sell_indicator=parse_enum(BuySellIndicator, data.get("buy_sell_indicator"), BuySellIndicator.BUY),
            waiver_indicator=data.get("waiver_indicator", ""),
            short_selling_indicator=data.get("short_selling_indicator", ""),
            otc_post_trade_indicator=data.get("otc_post_trade_indicator", ""),
            created_at=parse_datetime(data.get("created_at")) or datetime.now(timezone.utc),
            submitted_at=parse_datetime(data.get("submitted_at")),
            arm_confirmation_id=data.get("arm_confirmation_id", ""),
            rejection_reason=data.get("rejection_reason", ""),
            amendment_reason=data.get("amendment_reason", ""),
            original_report_id=data.get("original_report_id", ""),
        )

    def to_xml(self, namespace: str = "urn:iso:std:iso:20022:tech:xsd:auth.016.001.01") -> str:
        """
        Convert report to ISO 20022 XML format for ARM submission.

        Per RTS 22, transaction reports must be submitted in ISO 20022 format.

        Args:
            namespace: XML namespace for the document.

        Returns:
            XML string representation of the report.
        """
        # Root element
        root = ET.Element("TxRpt", xmlns=namespace)

        # Transaction element
        tx = ET.SubElement(root, "Tx")

        # New/Cancel/Amendment
        ET.SubElement(tx, "TxTp").text = self.transaction_type.value

        # Report identification
        rpt_id = ET.SubElement(tx, "RptId")
        ET.SubElement(rpt_id, "TxRefNb").text = self.transaction_reference_number
        if self.trading_venue_transaction_id:
            ET.SubElement(rpt_id, "TradgVnTxId").text = self.trading_venue_transaction_id

        # Executing entity
        exec_enty = ET.SubElement(tx, "ExctgPty")
        ET.SubElement(exec_enty, "LEI").text = self.executing_entity_lei

        # Investment firm covered
        ET.SubElement(tx, "InvstmtFirmCovrd").text = "true" if self.investment_firm_covered else "false"

        # Buyer
        if self.buyer.identifier:
            buyr = ET.SubElement(tx, "Buyr")
            buyr_id = ET.SubElement(buyr, "Id")
            if self.buyer.identifier_type == IdentifierType.LEI:
                ET.SubElement(buyr_id, "LEI").text = self.buyer.identifier
            else:
                prsn = ET.SubElement(buyr_id, "Prsn")
                if self.buyer.first_name:
                    ET.SubElement(prsn, "FrstNm").text = self.buyer.first_name
                if self.buyer.surname:
                    ET.SubElement(prsn, "Nm").text = self.buyer.surname
                if self.buyer.date_of_birth:
                    ET.SubElement(prsn, "BirthDt").text = self.buyer.date_of_birth.isoformat()

        # Seller
        if self.seller.identifier:
            sellr = ET.SubElement(tx, "Sellr")
            sellr_id = ET.SubElement(sellr, "Id")
            if self.seller.identifier_type == IdentifierType.LEI:
                ET.SubElement(sellr_id, "LEI").text = self.seller.identifier
            else:
                prsn = ET.SubElement(sellr_id, "Prsn")
                if self.seller.first_name:
                    ET.SubElement(prsn, "FrstNm").text = self.seller.first_name
                if self.seller.surname:
                    ET.SubElement(prsn, "Nm").text = self.seller.surname

        # Trading capacity
        ET.SubElement(tx, "TradgCpcty").text = self.trading_capacity.value

        # Quantity
        qty = ET.SubElement(tx, "Qty")
        ET.SubElement(qty, "Unit").text = str(self.quantity)

        # Price
        pric = ET.SubElement(tx, "Pric")
        if self.price_type == PriceType.MONETARY:
            mn = ET.SubElement(pric, "MntryVal")
            ET.SubElement(mn, "Amt", Ccy=self.price_currency).text = str(self.price)
        else:
            ET.SubElement(pric, self.price_type.value).text = str(self.price)

        # Net amount
        if self.net_amount:
            ET.SubElement(tx, "NetAmt", Ccy=self.price_currency).text = str(self.net_amount)

        # Venue
        vnue = ET.SubElement(tx, "TradgVn")
        ET.SubElement(vnue, "MIC").text = self.venue_mic

        # Trading datetime
        ET.SubElement(tx, "TradDt").text = self.trading_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Instrument
        finInstrm = ET.SubElement(tx, "FinInstrm")
        instrmId = ET.SubElement(finInstrm, "Id")
        ET.SubElement(instrmId, "ISIN").text = self.instrument_isin

        if self.instrument_full_name:
            ET.SubElement(finInstrm, "FullNm").text = self.instrument_full_name

        if self.instrument_classification:
            ET.SubElement(finInstrm, "ClssfctnTp").text = self.instrument_classification

        if self.price_multiplier != Decimal("1"):
            ET.SubElement(finInstrm, "PricMltplr").text = str(self.price_multiplier)

        # Buy/Sell indicator
        ET.SubElement(tx, "BuySellInd").text = self.buy_sell_indicator.value

        # Trading decision maker
        if self.trading_decision_maker_id:
            tdm = ET.SubElement(tx, "TradgDcsnMakr")
            if self.trading_decision_maker_id_type == "ALGO":
                ET.SubElement(tdm, "Algo").text = self.trading_decision_maker_id
            else:
                ET.SubElement(tdm, "Prsn").text = self.trading_decision_maker_id

        # Execution decision maker
        if self.execution_decision_maker_id:
            edm = ET.SubElement(tx, "ExctnDcsnMakr")
            if self.execution_decision_maker_id_type == "ALGO":
                ET.SubElement(edm, "Algo").text = self.execution_decision_maker_id
            else:
                ET.SubElement(edm, "Prsn").text = self.execution_decision_maker_id

        # Short selling indicator
        if self.short_selling_indicator:
            ET.SubElement(tx, "ShrtSellgInd").text = self.short_selling_indicator

        # Waiver indicator
        if self.waiver_indicator:
            ET.SubElement(tx, "WvrInd").text = self.waiver_indicator

        # OTC post-trade indicator
        if self.otc_post_trade_indicator:
            ET.SubElement(tx, "OTCPstTradInd").text = self.otc_post_trade_indicator

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of report for integrity verification.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        import json
        data = self.to_dict()
        # Exclude mutable fields
        for key in ["created_at", "submitted_at", "arm_confirmation_id", "rejection_reason", "report_status"]:
            data.pop(key, None)
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def create_cancellation(self, reason: str = "") -> "TransactionReport":
        """
        Create a cancellation report for this transaction.

        Args:
            reason: Reason for cancellation.

        Returns:
            New TransactionReport with CANCEL type.
        """
        cancellation = TransactionReport.from_dict(self.to_dict())
        cancellation.transaction_reference_number = str(uuid.uuid4())
        cancellation.transaction_type = TransactionType.CANCEL
        cancellation.original_report_id = self.transaction_reference_number
        cancellation.amendment_reason = reason
        cancellation.report_status = ReportStatus.PENDING
        cancellation.created_at = datetime.now(timezone.utc)
        cancellation.submitted_at = None
        cancellation.arm_confirmation_id = ""
        return cancellation

    def create_amendment(self, reason: str = "") -> "TransactionReport":
        """
        Create an amendment report for this transaction.

        Args:
            reason: Reason for amendment.

        Returns:
            New TransactionReport with AMENDMENT type.
        """
        amendment = TransactionReport.from_dict(self.to_dict())
        amendment.transaction_reference_number = str(uuid.uuid4())
        amendment.transaction_type = TransactionType.AMENDMENT
        amendment.original_report_id = self.transaction_reference_number
        amendment.amendment_reason = reason
        amendment.report_status = ReportStatus.PENDING
        amendment.created_at = datetime.now(timezone.utc)
        amendment.submitted_at = None
        amendment.arm_confirmation_id = ""
        return amendment


# ============================================================================
# Report Builder
# ============================================================================


class TransactionReportBuilder:
    """
    Builder for creating TransactionReport instances from trade data.

    Simplifies the creation of transaction reports by providing
    sensible defaults and automatic field population.

    Example:
        builder = TransactionReportBuilder(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
        )

        report = builder.from_trade({
            "order_id": "12345",
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150.25,
            "side": "BUY",
            "venue": "XNAS",
            "timestamp": datetime.now(timezone.utc),
        })
    """

    def __init__(
        self,
        executing_entity_lei: str,
        submitting_entity_lei: str = "",
        default_currency: str = "EUR",
        trading_capacity: TradingCapacity = TradingCapacity.DEAL,
        country_of_branch: str = "",
    ):
        """
        Initialize report builder.

        Args:
            executing_entity_lei: LEI of the executing firm.
            submitting_entity_lei: LEI of the submitting firm (if different).
            default_currency: Default currency for prices.
            trading_capacity: Default trading capacity.
            country_of_branch: Country code of supervising branch.
        """
        self._executing_lei = executing_entity_lei
        self._submitting_lei = submitting_entity_lei or executing_entity_lei
        self._default_currency = default_currency
        self._trading_capacity = trading_capacity
        self._country_of_branch = country_of_branch

    def from_trade(
        self,
        trade: Dict[str, Any],
        algorithm_id: str = "",
    ) -> TransactionReport:
        """
        Create TransactionReport from trade dictionary.

        Args:
            trade: Trade data dictionary.
            algorithm_id: ID of the algorithm that made the trading decision.

        Returns:
            TransactionReport instance.

        Trade Dictionary Fields:
            - order_id: Order/trade identifier
            - isin: Instrument ISIN
            - quantity: Trade quantity
            - price: Execution price
            - side: "BUY" or "SELL"
            - venue/mic: Execution venue MIC
            - timestamp/datetime: Execution timestamp
            - currency: Price currency (optional)
            - counterparty_lei: Counterparty LEI (optional)
        """
        # Determine buy/sell
        side = trade.get("side", "BUY").upper()
        buy_sell = BuySellIndicator.BUY if side == "BUY" else BuySellIndicator.SELL

        # Parse quantity and price
        quantity = Decimal(str(trade.get("quantity", 0)))
        price = Decimal(str(trade.get("price", 0)))

        # Get currency
        currency = trade.get("currency", self._default_currency)

        # Get venue
        venue = trade.get("venue") or trade.get("mic", "XOFF")

        # Get timestamp
        timestamp = trade.get("timestamp") or trade.get("datetime") or datetime.now(timezone.utc)
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Build buyer/seller based on side
        counterparty_lei = trade.get("counterparty_lei", "")

        if buy_sell == BuySellIndicator.BUY:
            buyer = TransactionReportParty(
                identifier=self._executing_lei,
                identifier_type=IdentifierType.LEI,
            )
            seller = TransactionReportParty(
                identifier=counterparty_lei,
                identifier_type=IdentifierType.LEI if counterparty_lei else IdentifierType.NONE,
            )
        else:
            seller = TransactionReportParty(
                identifier=self._executing_lei,
                identifier_type=IdentifierType.LEI,
            )
            buyer = TransactionReportParty(
                identifier=counterparty_lei,
                identifier_type=IdentifierType.LEI if counterparty_lei else IdentifierType.NONE,
            )

        return TransactionReport(
            transaction_reference_number=trade.get("order_id") or str(uuid.uuid4()),
            trading_venue_transaction_id=trade.get("venue_trade_id", ""),
            executing_entity_lei=self._executing_lei,
            submitting_entity_lei=self._submitting_lei,
            buyer=buyer,
            seller=seller,
            trading_capacity=self._trading_capacity,
            quantity=quantity,
            price=price,
            price_currency=currency,
            net_amount=quantity * price,
            venue_mic=venue,
            country_of_branch=self._country_of_branch,
            instrument_isin=trade.get("isin", ""),
            instrument_full_name=trade.get("instrument_name", ""),
            instrument_classification=trade.get("cfi", ""),
            trading_datetime=timestamp,
            buy_sell_indicator=buy_sell,
            trading_decision_maker_id=algorithm_id,
            trading_decision_maker_id_type="ALGO" if algorithm_id else "",
            execution_decision_maker_id=algorithm_id,
            execution_decision_maker_id_type="ALGO" if algorithm_id else "",
        )


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Enums
    "BuySellIndicator",
    "TradingCapacity",
    "IdentifierType",
    "InstrumentIdentifierType",
    "PriceType",
    "QuantityType",
    "TransactionType",
    "ReportStatus",
    # Validators
    "ISINValidator",
    "MICValidator",
    "CFIValidator",
    # Data classes
    "TransactionReportParty",
    "TransactionReport",
    # Builder
    "TransactionReportBuilder",
]
