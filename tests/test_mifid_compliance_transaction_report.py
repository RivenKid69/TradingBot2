# -*- coding: utf-8 -*-
"""
Tests for MiFID II Transaction Report Module (RTS 22).

Tests cover:
    - ISIN validation (ISO 6166)
    - MIC validation (ISO 10383)
    - CFI validation (ISO 10962)
    - TransactionReport data model
    - TransactionReportParty model
    - Report validation
    - XML generation
    - JSON serialization/deserialization
    - TransactionReportBuilder
    - Cancellation and amendment creation

References:
    - RTS 22 (Regulation 2017/590)
    - ISO 6166 - ISIN
    - ISO 10383 - MIC
    - ISO 10962 - CFI
"""

import pytest
from datetime import datetime, date, timezone
from decimal import Decimal
import uuid

from services.compliance.transaction_report import (
    # Enums
    BuySellIndicator,
    TradingCapacity,
    IdentifierType,
    InstrumentIdentifierType,
    PriceType,
    QuantityType,
    TransactionType,
    ReportStatus,
    # Validators
    ISINValidator,
    MICValidator,
    CFIValidator,
    # Data classes
    TransactionReportParty,
    TransactionReport,
    # Builder
    TransactionReportBuilder,
)


# ============================================================================
# ISIN Validator Tests
# ============================================================================


class TestISINValidator:
    """Tests for ISIN validation per ISO 6166."""

    def test_validate_format_valid_isins(self):
        """Test validation of correctly formatted ISINs."""
        valid_isins = [
            "US0378331005",  # Apple Inc.
            "GB00B0SWJX34",  # Royal Dutch Shell
            "DE0007236101",  # Siemens AG
            "FR0000120271",  # TotalEnergies
            "CH0012032048",  # Roche
            "JP3633400001",  # Toyota
            "NL0000009165",  # Heineken
        ]
        for isin in valid_isins:
            assert ISINValidator.validate_format(isin), f"Should accept valid ISIN: {isin}"

    def test_validate_format_invalid_isins(self):
        """Test rejection of incorrectly formatted ISINs."""
        invalid_isins = [
            "",  # Empty
            "US037833100",  # Too short
            "US03783310051",  # Too long
            "1S0378331005",  # First char not letter
            "U50378331005",  # Second char not letter
            # Note: lowercase is valid since validate_format converts to upper
            "US037833100A",  # Last char not digit
            "US 378331005",  # Contains space
        ]
        for isin in invalid_isins:
            assert not ISINValidator.validate_format(isin), f"Should reject invalid ISIN: {isin}"

    def test_validate_format_none_and_non_string(self):
        """Test rejection of None and non-string inputs."""
        assert not ISINValidator.validate_format(None)
        assert not ISINValidator.validate_format(12345)
        assert not ISINValidator.validate_format([])

    def test_validate_checksum_valid(self):
        """Test checksum validation for real ISINs."""
        valid_isins = [
            "US0378331005",  # Apple - checksum 05
            "GB00B0SWJX34",  # Shell - checksum 34
            "DE0007236101",  # Siemens - checksum 01
        ]
        for isin in valid_isins:
            assert ISINValidator.validate_checksum(isin), f"Checksum should pass for: {isin}"

    def test_validate_checksum_invalid(self):
        """Test checksum validation fails for wrong check digits."""
        invalid_isins = [
            "US0378331000",  # Wrong check digit
            "US0378331001",  # Wrong check digit
            "GB00B0SWJX30",  # Wrong check digit
        ]
        for isin in invalid_isins:
            assert not ISINValidator.validate_checksum(isin), f"Checksum should fail for: {isin}"

    def test_validate_full_validation(self):
        """Test complete validation including format and checksum."""
        # Valid ISIN
        is_valid, error = ISINValidator.validate("US0378331005")
        assert is_valid
        assert error == ""

        # Empty ISIN
        is_valid, error = ISINValidator.validate("")
        assert not is_valid
        assert "required" in error.lower()

        # Invalid format
        is_valid, error = ISINValidator.validate("INVALID")
        assert not is_valid
        assert "format" in error.lower()

        # Invalid checksum
        is_valid, error = ISINValidator.validate("US0378331000")
        assert not is_valid
        assert "checksum" in error.lower()

    def test_validate_case_insensitive(self):
        """Test that validation handles lowercase input."""
        is_valid, _ = ISINValidator.validate("us0378331005")
        assert is_valid


# ============================================================================
# MIC Validator Tests
# ============================================================================


class TestMICValidator:
    """Tests for MIC validation per ISO 10383."""

    def test_validate_format_valid_mics(self):
        """Test validation of correctly formatted MICs."""
        valid_mics = [
            "XLON",  # London Stock Exchange
            "XPAR",  # Euronext Paris
            "XETR",  # Deutsche Börse Xetra
            "XNAS",  # NASDAQ
            "XNYS",  # NYSE
            "XOFF",  # Off-exchange
            "XXXX",  # No venue
        ]
        for mic in valid_mics:
            assert MICValidator.validate_format(mic), f"Should accept valid MIC: {mic}"

    def test_validate_format_invalid_mics(self):
        """Test rejection of incorrectly formatted MICs."""
        invalid_mics = [
            "",  # Empty
            "XLO",  # Too short
            "XLONG",  # Too long
            "1LON",  # First char not letter
            "X_ON",  # Contains underscore
            "X ON",  # Contains space
        ]
        for mic in invalid_mics:
            assert not MICValidator.validate_format(mic), f"Should reject invalid MIC: {mic}"

    def test_validate_format_none_and_non_string(self):
        """Test rejection of None and non-string inputs."""
        assert not MICValidator.validate_format(None)
        assert not MICValidator.validate_format(12345)

    def test_validate_full_validation(self):
        """Test complete validation."""
        # Valid MIC
        is_valid, error = MICValidator.validate("XLON")
        assert is_valid
        assert error == ""

        # Empty MIC
        is_valid, error = MICValidator.validate("")
        assert not is_valid
        assert "required" in error.lower()

        # Invalid format
        is_valid, error = MICValidator.validate("X")
        assert not is_valid
        assert "format" in error.lower()

    def test_validate_strict_mode(self):
        """Test strict validation only accepts known MICs."""
        # Known MIC should pass
        is_valid, _ = MICValidator.validate("XLON", strict=True)
        assert is_valid

        # Unknown MIC should fail in strict mode
        is_valid, error = MICValidator.validate("ZZZZ", strict=True)
        assert not is_valid
        assert "unknown" in error.lower()

        # Unknown MIC should pass in non-strict mode
        is_valid, _ = MICValidator.validate("ZZZZ", strict=False)
        assert is_valid

    def test_validate_case_insensitive(self):
        """Test that validation handles lowercase input."""
        is_valid, _ = MICValidator.validate("xlon")
        assert is_valid


# ============================================================================
# CFI Validator Tests
# ============================================================================


class TestCFIValidator:
    """Tests for CFI validation per ISO 10962."""

    def test_validate_format_valid_cfis(self):
        """Test validation of correctly formatted CFIs."""
        valid_cfis = [
            "ESXXXX",  # Equity share
            "DBXXXX",  # Debt bond
            "OCXXXX",  # Option call
            "OPXXXX",  # Option put
            "FFXXXX",  # Future
            "SXXXXX",  # Swap
        ]
        for cfi in valid_cfis:
            assert CFIValidator.validate_format(cfi), f"Should accept valid CFI: {cfi}"

    def test_validate_format_invalid_cfis(self):
        """Test rejection of incorrectly formatted CFIs."""
        invalid_cfis = [
            "ESXXX",  # Too short
            "ESXXXXX",  # Too long
            "1SXXXX",  # Starts with digit
            "ES1234",  # Contains digits
        ]
        for cfi in invalid_cfis:
            assert not CFIValidator.validate_format(cfi), f"Should reject invalid CFI: {cfi}"

    def test_validate_empty_cfi_allowed(self):
        """Test that empty CFI is allowed (optional field)."""
        is_valid, error = CFIValidator.validate("")
        assert is_valid
        assert error == ""

    def test_validate_valid_category(self):
        """Test validation of known categories."""
        is_valid, _ = CFIValidator.validate("ESXXXX")
        assert is_valid

    def test_validate_unknown_category(self):
        """Test validation fails for unknown category."""
        is_valid, error = CFIValidator.validate("ZSXXXX")
        assert not is_valid
        assert "category" in error.lower()


# ============================================================================
# TransactionReportParty Tests
# ============================================================================


class TestTransactionReportParty:
    """Tests for TransactionReportParty data class."""

    def test_create_party_with_lei(self):
        """Test creating a party with LEI identifier."""
        party = TransactionReportParty(
            identifier="5493001KJTIIGC8Y1R12",
            identifier_type=IdentifierType.LEI,
        )
        assert party.identifier == "5493001KJTIIGC8Y1R12"
        assert party.identifier_type == IdentifierType.LEI
        assert party.is_legal_entity()

    def test_create_party_with_national_id(self):
        """Test creating a party with national ID."""
        party = TransactionReportParty(
            identifier="123456789",
            identifier_type=IdentifierType.NATIONAL_ID,
            country="DE",
            first_name="John",
            surname="Doe",
            date_of_birth=date(1990, 1, 15),
        )
        assert party.identifier == "123456789"
        assert party.identifier_type == IdentifierType.NATIONAL_ID
        assert party.country == "DE"
        assert party.first_name == "John"
        assert not party.is_legal_entity()

    def test_validate_party_valid(self):
        """Test validation passes for valid party."""
        party = TransactionReportParty(
            identifier="5493001KJTIIGC8Y1R12",
            identifier_type=IdentifierType.LEI,
        )
        errors = party.validate()
        assert len(errors) == 0

    def test_validate_party_missing_identifier(self):
        """Test validation fails for missing identifier."""
        party = TransactionReportParty(
            identifier="",
            identifier_type=IdentifierType.LEI,
        )
        errors = party.validate()
        assert len(errors) > 0
        assert any("identifier" in e.lower() for e in errors)

    def test_validate_party_invalid_lei_length(self):
        """Test validation fails for invalid LEI length."""
        party = TransactionReportParty(
            identifier="SHORT",
            identifier_type=IdentifierType.LEI,
        )
        errors = party.validate()
        assert len(errors) > 0
        assert any("lei" in e.lower() for e in errors)

    def test_validate_party_national_id_requires_country(self):
        """Test validation fails for national ID without country."""
        party = TransactionReportParty(
            identifier="123456789",
            identifier_type=IdentifierType.NATIONAL_ID,
            country="",  # Missing country
        )
        errors = party.validate()
        assert len(errors) > 0
        assert any("country" in e.lower() for e in errors)

    def test_validate_party_invalid_country_code(self):
        """Test validation fails for invalid country code."""
        party = TransactionReportParty(
            identifier="123456789",
            identifier_type=IdentifierType.NATIONAL_ID,
            country="INVALID",  # Too long
        )
        errors = party.validate()
        assert len(errors) > 0

    def test_party_to_dict(self):
        """Test party serialization to dictionary."""
        party = TransactionReportParty(
            identifier="5493001KJTIIGC8Y1R12",
            identifier_type=IdentifierType.LEI,
            country="GB",
        )
        data = party.to_dict()
        assert data["identifier"] == "5493001KJTIIGC8Y1R12"
        assert data["identifier_type"] == "LEI"
        assert data["country"] == "GB"


# ============================================================================
# TransactionReport Tests
# ============================================================================


class TestTransactionReport:
    """Tests for TransactionReport data class."""

    @pytest.fixture
    def valid_report(self):
        """Create a valid transaction report for testing."""
        return TransactionReport(
            transaction_reference_number="TEST-001",
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("100"),
            price=Decimal("150.25"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
            buy_sell_indicator=BuySellIndicator.BUY,
            buyer=TransactionReportParty(
                identifier="5493001KJTIIGC8Y1R12",
                identifier_type=IdentifierType.LEI,
            ),
            seller=TransactionReportParty(
                identifier="",
                identifier_type=IdentifierType.NONE,
            ),
        )

    def test_create_report_defaults(self):
        """Test creating report with default values."""
        report = TransactionReport()
        assert report.transaction_reference_number  # UUID generated
        assert report.investment_firm_covered is True
        assert report.transaction_type == TransactionType.NEW
        assert report.trading_capacity == TradingCapacity.DEAL
        assert report.quantity == Decimal("0")

    def test_create_valid_report(self, valid_report):
        """Test creating a fully valid report."""
        assert valid_report.executing_entity_lei == "5493001KJTIIGC8Y1R12"
        assert valid_report.instrument_isin == "US0378331005"
        assert valid_report.quantity == Decimal("100")
        assert valid_report.buy_sell_indicator == BuySellIndicator.BUY

    def test_validate_valid_report(self, valid_report):
        """Test validation passes for valid report."""
        errors = valid_report.validate()
        assert len(errors) == 0
        assert valid_report.is_valid()

    def test_validate_missing_executing_entity_lei(self, valid_report):
        """Test validation fails without executing entity LEI."""
        valid_report.executing_entity_lei = ""
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("lei" in e.lower() for e in errors)

    def test_validate_invalid_lei_length(self, valid_report):
        """Test validation fails for invalid LEI length."""
        valid_report.executing_entity_lei = "SHORT"
        errors = valid_report.validate()
        assert len(errors) > 0

    def test_validate_missing_isin(self, valid_report):
        """Test validation fails without ISIN."""
        valid_report.instrument_isin = ""
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("isin" in e.lower() for e in errors)

    def test_validate_invalid_isin(self, valid_report):
        """Test validation fails for invalid ISIN."""
        valid_report.instrument_isin = "INVALID"
        errors = valid_report.validate()
        assert len(errors) > 0

    def test_validate_zero_quantity(self, valid_report):
        """Test validation fails for zero quantity."""
        valid_report.quantity = Decimal("0")
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("quantity" in e.lower() for e in errors)

    def test_validate_negative_price(self, valid_report):
        """Test validation fails for negative price."""
        valid_report.price = Decimal("-100")
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("price" in e.lower() for e in errors)

    def test_validate_missing_venue_mic(self, valid_report):
        """Test validation fails without venue MIC."""
        valid_report.venue_mic = ""
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("mic" in e.lower() for e in errors)

    def test_validate_invalid_mic(self, valid_report):
        """Test validation fails for invalid MIC."""
        valid_report.venue_mic = "X"
        errors = valid_report.validate()
        assert len(errors) > 0

    def test_validate_missing_price_currency(self, valid_report):
        """Test validation fails without price currency for monetary price."""
        valid_report.price_currency = ""
        valid_report.price_type = PriceType.MONETARY
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("currency" in e.lower() for e in errors)

    def test_validate_invalid_currency_code(self, valid_report):
        """Test validation fails for invalid currency code."""
        valid_report.price_currency = "INVALID"
        errors = valid_report.validate()
        assert len(errors) > 0

    def test_validate_transmission_without_lei(self, valid_report):
        """Test validation fails for transmission without transmitting LEI."""
        valid_report.transmission_indicator = True
        valid_report.transmitting_firm_lei = ""
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("transmitting" in e.lower() for e in errors)

    def test_validate_cancellation_without_original_id(self, valid_report):
        """Test validation fails for cancellation without original report ID."""
        valid_report.transaction_type = TransactionType.CANCEL
        valid_report.original_report_id = ""
        errors = valid_report.validate()
        assert len(errors) > 0
        assert any("original" in e.lower() for e in errors)

    def test_validate_invalid_cfi(self, valid_report):
        """Test validation fails for invalid CFI."""
        valid_report.instrument_classification = "INVALID"
        errors = valid_report.validate()
        assert len(errors) > 0

    def test_calculate_net_amount(self, valid_report):
        """Test net amount calculation."""
        valid_report.quantity = Decimal("100")
        valid_report.price = Decimal("10.50")
        valid_report.price_multiplier = Decimal("1")
        assert valid_report.calculate_net_amount() == Decimal("1050.00")

    def test_calculate_net_amount_with_multiplier(self, valid_report):
        """Test net amount calculation with price multiplier."""
        valid_report.quantity = Decimal("100")
        valid_report.price = Decimal("10")
        valid_report.price_multiplier = Decimal("100")  # Options multiplier
        assert valid_report.calculate_net_amount() == Decimal("100000")

    def test_to_dict(self, valid_report):
        """Test report serialization to dictionary."""
        data = valid_report.to_dict()
        assert data["transaction_reference_number"] == "TEST-001"
        assert data["executing_entity_lei"] == "5493001KJTIIGC8Y1R12"
        assert data["instrument_isin"] == "US0378331005"
        assert data["quantity"] == "100"
        assert data["buy_sell_indicator"] == "BUYI"

    def test_from_dict(self, valid_report):
        """Test report deserialization from dictionary."""
        data = valid_report.to_dict()
        restored = TransactionReport.from_dict(data)

        assert restored.transaction_reference_number == valid_report.transaction_reference_number
        assert restored.executing_entity_lei == valid_report.executing_entity_lei
        assert restored.instrument_isin == valid_report.instrument_isin
        assert restored.quantity == valid_report.quantity
        assert restored.buy_sell_indicator == valid_report.buy_sell_indicator

    def test_to_xml(self, valid_report):
        """Test XML generation."""
        xml = valid_report.to_xml()
        assert "<?xml" in xml
        assert "TxRpt" in xml
        assert valid_report.executing_entity_lei in xml
        assert valid_report.instrument_isin in xml
        assert "BUYI" in xml

    def test_calculate_hash(self, valid_report):
        """Test hash calculation."""
        hash1 = valid_report.calculate_hash()
        assert len(hash1) == 64  # SHA-256 hex

        # Same report should have same hash
        hash2 = valid_report.calculate_hash()
        assert hash1 == hash2

        # Modified report should have different hash
        valid_report.quantity = Decimal("200")
        hash3 = valid_report.calculate_hash()
        assert hash1 != hash3

    def test_create_cancellation(self, valid_report):
        """Test cancellation report creation."""
        cancellation = valid_report.create_cancellation("Error in original report")

        assert cancellation.transaction_type == TransactionType.CANCEL
        assert cancellation.original_report_id == valid_report.transaction_reference_number
        assert cancellation.amendment_reason == "Error in original report"
        assert cancellation.transaction_reference_number != valid_report.transaction_reference_number
        assert cancellation.report_status == ReportStatus.PENDING

    def test_create_amendment(self, valid_report):
        """Test amendment report creation."""
        amendment = valid_report.create_amendment("Price correction")

        assert amendment.transaction_type == TransactionType.AMENDMENT
        assert amendment.original_report_id == valid_report.transaction_reference_number
        assert amendment.amendment_reason == "Price correction"
        assert amendment.transaction_reference_number != valid_report.transaction_reference_number


# ============================================================================
# TransactionReportBuilder Tests
# ============================================================================


class TestTransactionReportBuilder:
    """Tests for TransactionReportBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a builder for testing."""
        return TransactionReportBuilder(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            default_currency="EUR",
            trading_capacity=TradingCapacity.DEAL,
            country_of_branch="GB",
        )

    def test_builder_initialization(self, builder):
        """Test builder initialization."""
        assert builder._executing_lei == "5493001KJTIIGC8Y1R12"
        assert builder._default_currency == "EUR"
        assert builder._trading_capacity == TradingCapacity.DEAL

    def test_from_trade_buy(self, builder):
        """Test creating report from buy trade."""
        trade = {
            "order_id": "ORDER-001",
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150.25,
            "side": "BUY",
            "venue": "XNAS",
            "currency": "USD",
        }
        report = builder.from_trade(trade)

        assert report.transaction_reference_number == "ORDER-001"
        assert report.instrument_isin == "US0378331005"
        assert report.quantity == Decimal("100")
        assert report.price == Decimal("150.25")
        assert report.buy_sell_indicator == BuySellIndicator.BUY
        assert report.venue_mic == "XNAS"
        assert report.price_currency == "USD"

    def test_from_trade_sell(self, builder):
        """Test creating report from sell trade."""
        trade = {
            "order_id": "ORDER-002",
            "isin": "GB00B0SWJX34",
            "quantity": 50,
            "price": 25.50,
            "side": "SELL",
            "venue": "XLON",
        }
        report = builder.from_trade(trade)

        assert report.buy_sell_indicator == BuySellIndicator.SELL
        assert report.venue_mic == "XLON"

    def test_from_trade_with_algorithm(self, builder):
        """Test creating report with algorithm ID."""
        trade = {
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150,
            "side": "BUY",
            "venue": "XNAS",
        }
        report = builder.from_trade(trade, algorithm_id="ALGO-001")

        assert report.trading_decision_maker_id == "ALGO-001"
        assert report.trading_decision_maker_id_type == "ALGO"
        assert report.execution_decision_maker_id == "ALGO-001"

    def test_from_trade_with_counterparty(self, builder):
        """Test creating report with counterparty LEI."""
        trade = {
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150,
            "side": "BUY",
            "venue": "XNAS",
            "counterparty_lei": "549300EPNV3WGMPHFX08",
        }
        report = builder.from_trade(trade)

        # For buy, seller is counterparty
        assert report.seller.identifier == "549300EPNV3WGMPHFX08"
        assert report.seller.identifier_type == IdentifierType.LEI

    def test_from_trade_generates_order_id(self, builder):
        """Test that builder generates order ID if not provided."""
        trade = {
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150,
            "side": "BUY",
            "venue": "XNAS",
        }
        report = builder.from_trade(trade)

        assert report.transaction_reference_number  # Should be generated UUID

    def test_from_trade_default_currency(self, builder):
        """Test that builder uses default currency."""
        trade = {
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150,
            "side": "BUY",
            "venue": "XNAS",
            # No currency specified
        }
        report = builder.from_trade(trade)

        assert report.price_currency == "EUR"  # Default from builder

    def test_from_trade_with_timestamp(self, builder):
        """Test creating report with explicit timestamp."""
        test_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        trade = {
            "isin": "US0378331005",
            "quantity": 100,
            "price": 150,
            "side": "BUY",
            "venue": "XNAS",
            "timestamp": test_time,
        }
        report = builder.from_trade(trade)

        assert report.trading_datetime == test_time


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for RTS 22 enumerations."""

    def test_buy_sell_indicator_values(self):
        """Test BuySellIndicator enum values."""
        assert BuySellIndicator.BUY.value == "BUYI"
        assert BuySellIndicator.SELL.value == "SELL"

    def test_trading_capacity_values(self):
        """Test TradingCapacity enum values."""
        assert TradingCapacity.DEAL.value == "DEAL"
        assert TradingCapacity.MTCH.value == "MTCH"
        assert TradingCapacity.AOTC.value == "AOTC"

    def test_identifier_type_values(self):
        """Test IdentifierType enum values."""
        assert IdentifierType.LEI.value == "LEI"
        assert IdentifierType.MIC.value == "MIC"
        assert IdentifierType.NATIONAL_ID.value == "NATIONAL_ID"

    def test_price_type_values(self):
        """Test PriceType enum values."""
        assert PriceType.MONETARY.value == "MONE"
        assert PriceType.PERCENTAGE.value == "PERC"
        assert PriceType.YIELD.value == "YIEL"

    def test_quantity_type_values(self):
        """Test QuantityType enum values."""
        assert QuantityType.UNITS.value == "UNIT"
        assert QuantityType.NOMINAL.value == "NOML"
        assert QuantityType.MONETARY.value == "MONE"

    def test_transaction_type_values(self):
        """Test TransactionType enum values."""
        assert TransactionType.NEW.value == "NEW"
        assert TransactionType.CANCEL.value == "CANC"
        assert TransactionType.AMENDMENT.value == "AMND"

    def test_report_status_values(self):
        """Test ReportStatus enum values."""
        assert ReportStatus.PENDING.value == "pending"
        assert ReportStatus.SUBMITTED.value == "submitted"
        assert ReportStatus.ACCEPTED.value == "accepted"
        assert ReportStatus.REJECTED.value == "rejected"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_report_with_all_optional_fields(self):
        """Test report with all optional derivative fields."""
        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("100"),
            price=Decimal("150"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
            buyer=TransactionReportParty(identifier="", identifier_type=IdentifierType.NONE),
            seller=TransactionReportParty(identifier="", identifier_type=IdentifierType.NONE),
            # Optional derivative fields
            derivative_notional_increase=Decimal("1000000"),
            derivative_notional_decrease=Decimal("500000"),
            underlying_instrument_isin="US0000000000",
            underlying_index_name="S&P 500",
            underlying_index_term_value=3,
            underlying_index_term_unit="MNTH",
            option_type="CALL",
            strike_price=Decimal("160"),
            strike_price_currency="USD",
            option_exercise_style="EUROPEAN",
            maturity_date=date(2025, 12, 31),
            expiry_date=date(2025, 12, 31),
            delivery_type="CASH",
        )
        errors = report.validate()
        assert len(errors) == 0

    def test_from_dict_with_missing_fields(self):
        """Test from_dict handles missing fields gracefully."""
        minimal_data = {
            "executing_entity_lei": "5493001KJTIIGC8Y1R12",
            "instrument_isin": "US0378331005",
            "quantity": "100",
            "price": "150",
            "venue_mic": "XNAS",
        }
        report = TransactionReport.from_dict(minimal_data)
        assert report.executing_entity_lei == "5493001KJTIIGC8Y1R12"
        assert report.quantity == Decimal("100")

    def test_from_dict_with_invalid_decimal(self):
        """Test from_dict handles invalid decimal gracefully."""
        data = {
            "quantity": "invalid",
            "price": "also_invalid",
        }
        report = TransactionReport.from_dict(data)
        assert report.quantity == Decimal("0")
        assert report.price == Decimal("0")

    def test_from_dict_with_invalid_enum(self):
        """Test from_dict handles invalid enum gracefully."""
        data = {
            "buy_sell_indicator": "INVALID",
            "transaction_type": "INVALID",
        }
        report = TransactionReport.from_dict(data)
        assert report.buy_sell_indicator == BuySellIndicator.BUY  # Default
        assert report.transaction_type == TransactionType.NEW  # Default

    def test_large_quantity(self):
        """Test report with very large quantity."""
        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            quantity=Decimal("999999999999999"),
            price=Decimal("0.00001"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
            buyer=TransactionReportParty(identifier="", identifier_type=IdentifierType.NONE),
            seller=TransactionReportParty(identifier="", identifier_type=IdentifierType.NONE),
        )
        errors = report.validate()
        assert len(errors) == 0

    def test_xml_special_characters(self):
        """Test XML generation with special characters."""
        report = TransactionReport(
            executing_entity_lei="5493001KJTIIGC8Y1R12",
            instrument_isin="US0378331005",
            instrument_full_name="Test & Company <Holdings>",
            quantity=Decimal("100"),
            price=Decimal("150"),
            price_currency="USD",
            venue_mic="XNAS",
            trading_datetime=datetime.now(timezone.utc),
        )
        xml = report.to_xml()
        # Special characters should be escaped
        assert "&amp;" in xml or "Test & Company" not in xml
