# -*- coding: utf-8 -*-
"""
Tests for corporate actions data models (adapters/models.py extensions).

Tests cover:
- CorporateActionType enum
- DividendType enum
- CorporateAction dataclass
- Dividend dataclass
- StockSplit dataclass
- EarningsEvent dataclass
- AdjustmentFactors dataclass
"""

import pytest
from decimal import Decimal
from datetime import datetime

from adapters.models import (
    CorporateActionType,
    DividendType,
    CorporateAction,
    Dividend,
    StockSplit,
    EarningsEvent,
    AdjustmentFactors,
    ExchangeVendor,
)


class TestCorporateActionType:
    """Tests for CorporateActionType enum."""

    def test_all_action_types_exist(self):
        """Verify all expected action types are defined."""
        expected = {
            "dividend",
            "stock_dividend",
            "split",
            "merger",
            "spinoff",
            "rights",
            "symbol_change",
            "delisting",
        }
        actual = {t.value for t in CorporateActionType}
        assert actual == expected

    def test_enum_string_values(self):
        """Verify enum values are strings."""
        assert CorporateActionType.DIVIDEND.value == "dividend"
        assert CorporateActionType.SPLIT.value == "split"
        assert CorporateActionType.MERGER.value == "merger"

    def test_enum_is_str_subclass(self):
        """Verify enum inherits from str for JSON serialization."""
        assert isinstance(CorporateActionType.DIVIDEND, str)


class TestDividendType:
    """Tests for DividendType enum."""

    def test_all_dividend_types_exist(self):
        """Verify all expected dividend types are defined."""
        expected = {"regular", "special", "stock", "qualified", "unqualified"}
        actual = {t.value for t in DividendType}
        assert actual == expected

    def test_enum_is_str_subclass(self):
        """Verify enum inherits from str."""
        assert isinstance(DividendType.REGULAR, str)


class TestCorporateAction:
    """Tests for CorporateAction dataclass."""

    def test_create_basic_action(self):
        """Test creating a basic corporate action."""
        action = CorporateAction(
            action_type=CorporateActionType.DIVIDEND,
            symbol="AAPL",
            ex_date="2024-01-15",
        )
        assert action.symbol == "AAPL"
        assert action.ex_date == "2024-01-15"
        assert action.action_type == CorporateActionType.DIVIDEND
        assert action.description == ""  # Default is empty string
        assert action.adjustment_factor is None

    def test_create_action_with_all_fields(self):
        """Test creating action with all optional fields."""
        action = CorporateAction(
            action_type=CorporateActionType.SPLIT,
            symbol="MSFT",
            ex_date="2024-02-01",
            description="4-for-1 stock split",
            adjustment_factor=0.25,
            record_date="2024-01-25",
            pay_date="2024-02-05",  # Note: pay_date, not payment_date
            raw_data={"ratio": "4:1"},
        )
        assert action.description == "4-for-1 stock split"
        assert action.adjustment_factor == 0.25
        assert action.record_date == "2024-01-25"
        assert action.pay_date == "2024-02-05"
        assert action.raw_data == {"ratio": "4:1"}

    def test_dataclass_is_frozen(self):
        """Verify dataclass is immutable."""
        action = CorporateAction(
            action_type=CorporateActionType.DIVIDEND,
            symbol="AAPL",
            ex_date="2024-01-15",
        )
        with pytest.raises(AttributeError):
            action.symbol = "MSFT"

    def test_adjustment_factor_computed_from_ratio(self):
        """Test that adjustment_factor is computed from ratio if not provided."""
        action = CorporateAction(
            action_type=CorporateActionType.SPLIT,
            symbol="AAPL",
            ex_date="2024-01-15",
            ratio=(4, 1),  # 4-for-1 split
        )
        # For 4:1 split, adjustment_factor = 1/4 = 0.25
        assert action.adjustment_factor == pytest.approx(0.25)

    def test_is_price_adjusting(self):
        """Test is_price_adjusting property."""
        split_action = CorporateAction(
            action_type=CorporateActionType.SPLIT,
            symbol="AAPL",
            ex_date="2024-01-15",
        )
        assert split_action.is_price_adjusting is True

        symbol_change = CorporateAction(
            action_type=CorporateActionType.SYMBOL_CHANGE,
            symbol="FB",
            ex_date="2024-01-15",
        )
        assert symbol_change.is_price_adjusting is False


class TestDividend:
    """Tests for Dividend dataclass."""

    def test_create_basic_dividend(self):
        """Test creating a basic dividend."""
        div = Dividend(
            symbol="AAPL",
            ex_date="2024-01-15",
            amount=Decimal("0.24"),
        )
        assert div.symbol == "AAPL"
        assert div.ex_date == "2024-01-15"
        assert div.amount == Decimal("0.24")
        assert div.dividend_type == DividendType.REGULAR
        assert div.currency == "USD"
        assert div.is_adjusted is False  # Default is False

    def test_create_special_dividend(self):
        """Test creating a special dividend."""
        div = Dividend(
            symbol="MSFT",
            ex_date="2024-03-01",
            amount=Decimal("3.00"),
            dividend_type=DividendType.SPECIAL,
        )
        assert div.dividend_type == DividendType.SPECIAL

    def test_to_corporate_action(self):
        """Test converting Dividend to CorporateAction."""
        div = Dividend(
            symbol="AAPL",
            ex_date="2024-01-15",
            amount=Decimal("0.24"),
            record_date="2024-01-12",
            pay_date="2024-01-20",
        )
        action = div.to_corporate_action()

        assert isinstance(action, CorporateAction)
        assert action.symbol == "AAPL"
        assert action.ex_date == "2024-01-15"
        assert action.action_type == CorporateActionType.DIVIDEND
        assert "0.24" in action.description
        assert action.record_date == "2024-01-12"
        assert action.pay_date == "2024-01-20"

    def test_dataclass_is_frozen(self):
        """Verify dataclass is immutable."""
        div = Dividend(
            symbol="AAPL",
            ex_date="2024-01-15",
            amount=Decimal("0.24"),
        )
        with pytest.raises(AttributeError):
            div.amount = Decimal("0.50")

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        div = Dividend(
            symbol="AAPL",
            ex_date="2024-01-15",
            amount=Decimal("0.24"),
            dividend_type=DividendType.REGULAR,
        )
        d = div.to_dict()
        restored = Dividend.from_dict(d)

        assert restored.symbol == div.symbol
        assert restored.ex_date == div.ex_date
        assert restored.amount == div.amount


class TestStockSplit:
    """Tests for StockSplit dataclass."""

    def test_create_forward_split(self):
        """Test creating a forward split (4:1)."""
        split = StockSplit(
            symbol="AAPL",
            ex_date="2024-01-15",
            ratio=(4, 1),
        )
        assert split.symbol == "AAPL"
        assert split.ratio == (4, 1)
        assert split.is_reverse is False

    def test_create_reverse_split(self):
        """Test creating a reverse split (1:10)."""
        split = StockSplit(
            symbol="XYZ",
            ex_date="2024-02-01",
            ratio=(1, 10),
            is_reverse=True,
        )
        assert split.ratio == (1, 10)
        assert split.is_reverse is True

    def test_adjustment_factor_forward_split(self):
        """Test adjustment factor for forward split."""
        split = StockSplit(
            symbol="AAPL",
            ex_date="2024-01-15",
            ratio=(4, 1),
        )
        # For 4:1 split, prices before should be divided by 4
        assert split.adjustment_factor == 0.25

    def test_adjustment_factor_reverse_split(self):
        """Test adjustment factor for reverse split."""
        split = StockSplit(
            symbol="XYZ",
            ex_date="2024-02-01",
            ratio=(1, 10),
            is_reverse=True,
        )
        # For 1:10 reverse split, prices before should be multiplied by 10
        assert split.adjustment_factor == 10.0

    def test_to_corporate_action(self):
        """Test converting StockSplit to CorporateAction."""
        split = StockSplit(
            symbol="AAPL",
            ex_date="2024-01-15",
            ratio=(4, 1),
        )
        action = split.to_corporate_action()

        assert isinstance(action, CorporateAction)
        assert action.symbol == "AAPL"
        assert action.ex_date == "2024-01-15"
        assert action.action_type == CorporateActionType.SPLIT
        # Description format is "4-for-1 split"
        assert "4-for-1" in action.description
        assert action.adjustment_factor == 0.25

    def test_share_multiplier(self):
        """Test share_multiplier property."""
        split = StockSplit(
            symbol="AAPL",
            ex_date="2024-01-15",
            ratio=(4, 1),
        )
        assert split.share_multiplier == 4.0


class TestEarningsEvent:
    """Tests for EarningsEvent dataclass."""

    def test_create_basic_earnings(self):
        """Test creating basic earnings event."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
        )
        assert event.symbol == "AAPL"
        assert event.report_date == "2024-01-25"
        assert event.fiscal_quarter == 1
        assert event.fiscal_year == 2024
        assert event.is_confirmed is False

    def test_create_earnings_with_estimates(self):
        """Test creating earnings with estimates and actuals."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_estimate=Decimal("2.10"),
            eps_actual=Decimal("2.18"),
            revenue_estimate=Decimal("118000000000"),
            revenue_actual=Decimal("119500000000"),
            is_confirmed=True,
        )
        assert event.eps_estimate == Decimal("2.10")
        assert event.eps_actual == Decimal("2.18")
        assert event.is_confirmed is True

    def test_surprise_pct_is_field(self):
        """Test surprise_pct is a field (not computed property)."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_estimate=Decimal("2.00"),
            eps_actual=Decimal("2.20"),
            surprise_pct=10.0,  # Must be passed as field
            is_confirmed=True,
        )
        assert event.surprise_pct == 10.0

    def test_surprise_pct_no_estimate(self):
        """Test surprise_pct when not provided."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_actual=Decimal("2.20"),
            is_confirmed=True,
        )
        assert event.surprise_pct is None

    def test_beat_estimates_property(self):
        """Test beat_estimates property."""
        # Beat case
        beat_event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_estimate=Decimal("2.00"),
            eps_actual=Decimal("2.10"),
            is_confirmed=True,
        )
        assert beat_event.beat_estimates is True

        # Miss case
        miss_event = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_estimate=Decimal("2.00"),
            eps_actual=Decimal("1.90"),
            is_confirmed=True,
        )
        assert miss_event.beat_estimates is False

    def test_has_reported_property(self):
        """Test has_reported property."""
        # Not reported
        upcoming = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
        )
        assert upcoming.has_reported is False

        # Reported
        reported = EarningsEvent(
            symbol="AAPL",
            report_date="2024-01-25",
            fiscal_quarter=1,
            fiscal_year=2024,
            eps_actual=Decimal("2.10"),
            is_confirmed=True,
        )
        assert reported.has_reported is True


class TestAdjustmentFactors:
    """Tests for AdjustmentFactors dataclass."""

    def test_create_basic_factors(self):
        """Test creating basic adjustment factors."""
        factors = AdjustmentFactors(
            symbol="AAPL",
            date="2024-01-15",
            split_factor=0.25,
            dividend_factor=0.98,
        )
        assert factors.symbol == "AAPL"
        assert factors.date == "2024-01-15"
        assert factors.split_factor == 0.25
        assert factors.dividend_factor == 0.98

    def test_combined_factor(self):
        """Test combined adjustment factor."""
        factors = AdjustmentFactors(
            symbol="AAPL",
            date="2024-01-15",
            split_factor=0.25,
            dividend_factor=0.98,
        )
        expected = 0.25 * 0.98
        assert factors.combined_factor == pytest.approx(expected)

    def test_no_adjustment(self):
        """Test factors when no adjustment needed."""
        factors = AdjustmentFactors(
            symbol="AAPL",
            date="2024-01-15",
            split_factor=1.0,
            dividend_factor=1.0,
        )
        assert factors.combined_factor == 1.0


class TestExchangeVendorYahoo:
    """Tests for Yahoo ExchangeVendor."""

    def test_yahoo_vendor_exists(self):
        """Verify YAHOO vendor is defined."""
        assert hasattr(ExchangeVendor, "YAHOO")
        assert ExchangeVendor.YAHOO.value == "yahoo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
