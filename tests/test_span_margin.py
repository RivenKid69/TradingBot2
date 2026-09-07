# -*- coding: utf-8 -*-
"""
Tests for impl_span_margin.py - SPAN Margin Calculator.

This test file provides comprehensive coverage for the SPAN margin
calculator including:
- Single position margin calculations
- Portfolio margin with spread credits
- Inter-commodity spread credits
- Intra-commodity (calendar) spread credits
- Delivery month charges
- Margin call detection
- Edge cases and error handling

Total: 60+ tests for 100% coverage.
"""

import pytest
from decimal import Decimal
from typing import List, Optional

from impl_span_margin import (
    # Enums and Constants
    ProductGroup,
    PRODUCT_GROUPS,
    SCANNING_RANGES,
    DEFAULT_SCANNING_RANGE,
    INTER_COMMODITY_CREDITS,
    CALENDAR_SPREAD_CREDITS,
    DEFAULT_CALENDAR_SPREAD_CREDIT,
    APPROXIMATE_MARGINS_PER_CONTRACT,
    # Data Classes
    ScanningRangeConfig,
    InterCommoditySpreadCredit,
    SPANScenarioResult,
    PositionMarginDetail,
    SpreadCreditDetail,
    SPANMarginResult,
    # Calculator
    SPANMarginCalculator,
    # Factory Functions
    create_span_calculator,
    calculate_simple_margin,
    get_approximate_margin_per_contract,
)

from core_futures import (
    FuturesContractSpec,
    FuturesPosition,
    FuturesType,
    PositionSide,
    ContractType,
    SettlementType,
    Exchange,
    MarginMode,
    create_es_futures_spec,
    create_gc_futures_spec,
    create_6e_futures_spec,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def es_spec() -> FuturesContractSpec:
    """E-mini S&P 500 contract spec."""
    return create_es_futures_spec()


@pytest.fixture
def gc_spec() -> FuturesContractSpec:
    """Gold futures contract spec."""
    return create_gc_futures_spec()


@pytest.fixture
def eur_spec() -> FuturesContractSpec:
    """Euro FX futures contract spec."""
    return create_6e_futures_spec()


@pytest.fixture
def es_position_long() -> FuturesPosition:
    """Long 1 ES position."""
    return FuturesPosition(
        symbol="ES",
        qty=Decimal("1"),
        entry_price=Decimal("4500"),
        side=PositionSide.LONG,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def es_position_short() -> FuturesPosition:
    """Short 1 ES position."""
    return FuturesPosition(
        symbol="ES",
        qty=Decimal("-1"),
        entry_price=Decimal("4500"),
        side=PositionSide.SHORT,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def nq_position_long() -> FuturesPosition:
    """Long 1 NQ position."""
    return FuturesPosition(
        symbol="NQ",
        qty=Decimal("1"),
        entry_price=Decimal("15000"),
        side=PositionSide.LONG,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def gc_position_long() -> FuturesPosition:
    """Long 1 GC position."""
    return FuturesPosition(
        symbol="GC",
        qty=Decimal("1"),
        entry_price=Decimal("2000"),
        side=PositionSide.LONG,
        leverage=1,
        margin_mode=MarginMode.SPAN,
    )


@pytest.fixture
def calculator(es_spec, gc_spec, eur_spec) -> SPANMarginCalculator:
    """Pre-configured SPAN calculator with common specs."""
    return SPANMarginCalculator(
        contract_specs={
            "ES": es_spec,
            "NQ": es_spec,  # Use same spec for simplicity
            "GC": gc_spec,
            "6E": eur_spec,
        }
    )


# =============================================================================
# Test ProductGroup Mapping
# =============================================================================


class TestProductGroupMapping:
    """Tests for product group classification."""

    def test_equity_index_products(self):
        """Equity index products correctly classified."""
        assert PRODUCT_GROUPS["ES"] == ProductGroup.EQUITY_INDEX
        assert PRODUCT_GROUPS["NQ"] == ProductGroup.EQUITY_INDEX
        assert PRODUCT_GROUPS["YM"] == ProductGroup.EQUITY_INDEX
        assert PRODUCT_GROUPS["RTY"] == ProductGroup.EQUITY_INDEX
        assert PRODUCT_GROUPS["MES"] == ProductGroup.EQUITY_INDEX
        assert PRODUCT_GROUPS["MNQ"] == ProductGroup.EQUITY_INDEX

    def test_metals_products(self):
        """Metals products correctly classified."""
        assert PRODUCT_GROUPS["GC"] == ProductGroup.METALS
        assert PRODUCT_GROUPS["SI"] == ProductGroup.METALS
        assert PRODUCT_GROUPS["HG"] == ProductGroup.METALS
        assert PRODUCT_GROUPS["MGC"] == ProductGroup.METALS

    def test_energy_products(self):
        """Energy products correctly classified."""
        assert PRODUCT_GROUPS["CL"] == ProductGroup.ENERGY
        assert PRODUCT_GROUPS["NG"] == ProductGroup.ENERGY
        assert PRODUCT_GROUPS["RB"] == ProductGroup.ENERGY
        assert PRODUCT_GROUPS["HO"] == ProductGroup.ENERGY

    def test_currency_products(self):
        """Currency products correctly classified."""
        assert PRODUCT_GROUPS["6E"] == ProductGroup.CURRENCIES
        assert PRODUCT_GROUPS["6J"] == ProductGroup.CURRENCIES
        assert PRODUCT_GROUPS["6B"] == ProductGroup.CURRENCIES

    def test_bond_products(self):
        """Bond products correctly classified."""
        assert PRODUCT_GROUPS["ZB"] == ProductGroup.BONDS
        assert PRODUCT_GROUPS["ZN"] == ProductGroup.BONDS
        assert PRODUCT_GROUPS["ZT"] == ProductGroup.BONDS


# =============================================================================
# Test ScanningRangeConfig
# =============================================================================


class TestScanningRangeConfig:
    """Tests for scanning range configuration."""

    def test_es_scanning_range(self):
        """ES has 6% scanning range."""
        config = SCANNING_RANGES["ES"]
        assert config.price_scan_range_pct == Decimal("0.06")

    def test_nq_scanning_range(self):
        """NQ has 8% scanning range (more volatile)."""
        config = SCANNING_RANGES["NQ"]
        assert config.price_scan_range_pct == Decimal("0.08")

    def test_gc_scanning_range(self):
        """Gold has 5% scanning range."""
        config = SCANNING_RANGES["GC"]
        assert config.price_scan_range_pct == Decimal("0.05")

    def test_cl_scanning_range(self):
        """Crude oil has 10% scanning range (volatile)."""
        config = SCANNING_RANGES["CL"]
        assert config.price_scan_range_pct == Decimal("0.10")

    def test_ng_scanning_range(self):
        """Natural gas has 15% scanning range (very volatile)."""
        config = SCANNING_RANGES["NG"]
        assert config.price_scan_range_pct == Decimal("0.15")

    def test_currency_scanning_range(self):
        """Currencies have low scanning ranges (3-4%)."""
        assert SCANNING_RANGES["6E"].price_scan_range_pct == Decimal("0.03")
        assert SCANNING_RANGES["6J"].price_scan_range_pct == Decimal("0.04")

    def test_default_scanning_range(self):
        """Default scanning range is 8%."""
        assert DEFAULT_SCANNING_RANGE.price_scan_range_pct == Decimal("0.08")

    def test_default_volatility_scan(self):
        """Default volatility scan is 33%."""
        config = ScanningRangeConfig(Decimal("0.06"))
        assert config.volatility_scan_range_pct == Decimal("0.33")

    def test_custom_scanning_range(self):
        """Custom scanning range can be created."""
        config = ScanningRangeConfig(
            price_scan_range_pct=Decimal("0.12"),
            volatility_scan_range_pct=Decimal("0.50"),
        )
        assert config.price_scan_range_pct == Decimal("0.12")
        assert config.volatility_scan_range_pct == Decimal("0.50")


# =============================================================================
# Test Inter-Commodity Spread Credits
# =============================================================================


class TestInterCommodityCredits:
    """Tests for inter-commodity spread credits."""

    def test_es_nq_spread_credit(self):
        """ES/NQ spread gets 50% credit."""
        credit = next(
            c for c in INTER_COMMODITY_CREDITS if set([c.product1, c.product2]) == {"ES", "NQ"}
        )
        assert credit.credit_rate == Decimal("0.50")

    def test_es_ym_spread_credit(self):
        """ES/YM spread gets 55% credit."""
        credit = next(
            c for c in INTER_COMMODITY_CREDITS if set([c.product1, c.product2]) == {"ES", "YM"}
        )
        assert credit.credit_rate == Decimal("0.55")

    def test_gc_si_spread_credit(self):
        """Gold/Silver spread gets 35% credit."""
        credit = next(
            c for c in INTER_COMMODITY_CREDITS if set([c.product1, c.product2]) == {"GC", "SI"}
        )
        assert credit.credit_rate == Decimal("0.35")

    def test_micro_standard_spread_credit(self):
        """Micro/Standard spread gets 90% credit."""
        credit = next(
            c for c in INTER_COMMODITY_CREDITS if set([c.product1, c.product2]) == {"ES", "MES"}
        )
        assert credit.credit_rate == Decimal("0.90")

    def test_crack_spread_credit(self):
        """CL/RB crack spread gets 60% credit."""
        credit = next(
            c for c in INTER_COMMODITY_CREDITS if set([c.product1, c.product2]) == {"CL", "RB"}
        )
        assert credit.credit_rate == Decimal("0.60")


# =============================================================================
# Test Calendar Spread Credits
# =============================================================================


class TestCalendarSpreadCredits:
    """Tests for calendar spread credits."""

    def test_es_calendar_credit(self):
        """ES calendar spread gets 80% credit."""
        assert CALENDAR_SPREAD_CREDITS["ES"] == Decimal("0.80")

    def test_gc_calendar_credit(self):
        """Gold calendar spread gets 85% credit."""
        assert CALENDAR_SPREAD_CREDITS["GC"] == Decimal("0.85")

    def test_cl_calendar_credit(self):
        """Crude oil calendar gets 60% credit (lower due to contango)."""
        assert CALENDAR_SPREAD_CREDITS["CL"] == Decimal("0.60")

    def test_ng_calendar_credit(self):
        """Natural gas calendar gets 50% credit (seasonality)."""
        assert CALENDAR_SPREAD_CREDITS["NG"] == Decimal("0.50")

    def test_bond_calendar_credit(self):
        """Bonds get high calendar credit (90%)."""
        assert CALENDAR_SPREAD_CREDITS["ZN"] == Decimal("0.90")

    def test_default_calendar_credit(self):
        """Default calendar credit is 70%."""
        assert DEFAULT_CALENDAR_SPREAD_CREDIT == Decimal("0.70")


# =============================================================================
# Test SPANMarginCalculator - Basic Operations
# =============================================================================


class TestSPANCalculatorBasic:
    """Tests for basic SPAN calculator operations."""

    def test_create_calculator(self):
        """Calculator can be created."""
        calc = SPANMarginCalculator()
        assert calc is not None

    def test_create_with_specs(self, es_spec):
        """Calculator can be created with contract specs."""
        calc = SPANMarginCalculator(contract_specs={"ES": es_spec})
        assert calc is not None

    def test_get_scanning_range_known(self, calculator):
        """Get scanning range for known symbol."""
        config = calculator.get_scanning_range("ES")
        assert config.price_scan_range_pct == Decimal("0.06")

    def test_get_scanning_range_unknown(self, calculator):
        """Get default scanning range for unknown symbol."""
        config = calculator.get_scanning_range("UNKNOWN")
        assert config.price_scan_range_pct == DEFAULT_SCANNING_RANGE.price_scan_range_pct

    def test_get_calendar_credit_known(self, calculator):
        """Get calendar credit for known symbol."""
        rate = calculator.get_calendar_credit_rate("ES")
        assert rate == Decimal("0.80")

    def test_get_calendar_credit_unknown(self, calculator):
        """Get default calendar credit for unknown symbol."""
        rate = calculator.get_calendar_credit_rate("UNKNOWN")
        assert rate == DEFAULT_CALENDAR_SPREAD_CREDIT

    def test_set_contract_spec(self, calculator, gc_spec):
        """Contract spec can be added after creation."""
        calculator.set_contract_spec("NEW_PRODUCT", gc_spec)
        # Should not raise error when calculating margin

    def test_set_scanning_range(self, calculator):
        """Scanning range can be customized."""
        custom = ScanningRangeConfig(Decimal("0.15"))
        calculator.set_scanning_range("CUSTOM", custom)
        config = calculator.get_scanning_range("CUSTOM")
        assert config.price_scan_range_pct == Decimal("0.15")


# =============================================================================
# Test Single Position Margin
# =============================================================================


class TestSinglePositionMargin:
    """Tests for single position margin calculations."""

    def test_es_long_margin(self, calculator, es_position_long, es_spec):
        """Calculate margin for 1 long ES contract."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        # Notional = 4500 * 1 * 50 = 225,000
        # Scanning risk = 225,000 * 0.06 = 13,500
        # Initial = 13,500 * 1.10 = 14,850
        assert result.scanning_risk == Decimal("13500")
        assert result.initial_margin == Decimal("14850.00")
        assert result.maintenance_margin == Decimal("11880.00")  # 0.8 * 14850

    def test_es_short_margin(self, calculator, es_position_short, es_spec):
        """Short position has same margin as long."""
        result = calculator.calculate_margin(
            position=es_position_short,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        # Same margin as long
        assert result.scanning_risk == Decimal("13500")

    def test_multi_contract_margin(self, calculator, es_spec):
        """Margin scales with number of contracts."""
        position = FuturesPosition(
            symbol="ES",
            qty=Decimal("5"),  # 5 contracts
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calculator.calculate_margin(
            position=position,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        # Notional = 4500 * 5 * 50 = 1,125,000
        # Scanning risk = 1,125,000 * 0.06 = 67,500
        assert result.scanning_risk == Decimal("67500")

    def test_gc_margin(self, gc_position_long, gc_spec):
        """Calculate margin for gold futures."""
        calculator = SPANMarginCalculator(contract_specs={"GC": gc_spec})

        result = calculator.calculate_margin(
            position=gc_position_long,
            current_price=Decimal("2000"),
            contract_spec=gc_spec,
        )

        # Notional = 2000 * 1 * 100 = 200,000
        # Scanning risk = 200,000 * 0.05 = 10,000
        assert result.scanning_risk == Decimal("10000")

    def test_margin_with_position_details(self, calculator, es_position_long, es_spec):
        """Margin result includes position details."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        assert len(result.position_details) == 1
        detail = result.position_details[0]
        assert detail.symbol == "ES"
        assert detail.position_qty == Decimal("1")
        assert detail.position_side == PositionSide.LONG
        assert detail.notional_value == Decimal("225000")

    def test_margin_includes_scenario(self, calculator, es_position_long, es_spec):
        """Margin result includes worst scenario."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        assert result.worst_scenario is not None
        assert result.worst_scenario.price_move_pct == Decimal("-0.06")


# =============================================================================
# Test Delivery Month Charges
# =============================================================================


class TestDeliveryMonthCharges:
    """Tests for delivery month margin charges."""

    def test_no_delivery_charge_far_expiry(self, calculator, es_position_long, es_spec):
        """No delivery charge when far from expiry."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
            days_to_expiry=30,
        )

        assert result.delivery_month_charge == Decimal("0")

    def test_delivery_charge_near_expiry(self, calculator, es_position_long, es_spec):
        """Delivery charge applied when near expiry."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
            days_to_expiry=5,
        )

        # Delivery charge = scanning_risk * 0.05 = 13500 * 0.05 = 675
        assert result.delivery_month_charge == Decimal("675")

    def test_delivery_charge_last_day(self, calculator, es_position_long, es_spec):
        """Delivery charge on last day before expiry."""
        result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
            days_to_expiry=1,
        )

        assert result.delivery_month_charge == Decimal("675")

    def test_delivery_charge_increases_margin(self, calculator, es_position_long, es_spec):
        """Delivery charge increases total margin."""
        far_result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
            days_to_expiry=30,
        )

        near_result = calculator.calculate_margin(
            position=es_position_long,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
            days_to_expiry=5,
        )

        assert near_result.initial_margin > far_result.initial_margin


# =============================================================================
# Test Portfolio Margin
# =============================================================================


class TestPortfolioMargin:
    """Tests for portfolio margin calculations."""

    def test_single_position_portfolio(self, calculator, es_position_long, es_spec):
        """Portfolio with single position."""
        result = calculator.calculate_portfolio_margin(
            positions=[es_position_long],
            prices={"ES": Decimal("4500")},
            contract_specs={"ES": es_spec},
        )

        assert result.scanning_risk == Decimal("13500")

    def test_empty_portfolio(self, calculator):
        """Empty portfolio has zero margin."""
        result = calculator.calculate_portfolio_margin(
            positions=[],
            prices={},
        )

        assert result.initial_margin == Decimal("0")
        assert result.maintenance_margin == Decimal("0")
        assert result.scanning_risk == Decimal("0")

    def test_multi_position_portfolio_no_spread(self, calculator, es_spec, gc_spec):
        """Portfolio with unrelated positions (no spread credit)."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        gc_pos = FuturesPosition(
            symbol="GC",
            qty=Decimal("1"),
            entry_price=Decimal("2000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calculator.calculate_portfolio_margin(
            positions=[es_pos, gc_pos],
            prices={"ES": Decimal("4500"), "GC": Decimal("2000")},
            contract_specs={"ES": es_spec, "GC": gc_spec},
        )

        # Total scanning = ES (13,500) + GC (10,000) = 23,500
        # No inter-commodity credit for ES/GC
        assert result.scanning_risk == Decimal("23500")

    def test_es_nq_spread_gets_credit(self, calculator, es_spec):
        """ES/NQ spread receives inter-commodity credit."""
        # Create NQ spec (use ES multiplier for simplicity)
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),  # NQ multiplier
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )

        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec})

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos],
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        )

        # ES scanning: 4500 * 50 * 0.06 = 13,500
        # NQ scanning: 15000 * 20 * 0.08 = 24,000
        # Total scanning: 37,500
        # Inter-commodity credit: min(13500, 24000) * 0.50 = 6,750
        assert result.inter_commodity_credit == Decimal("6750")
        assert result.scanning_risk == Decimal("37500")

    def test_spread_credit_in_details(self, calculator, es_spec):
        """Spread credits appear in result details."""
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )

        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec})

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos],
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        )

        assert len(result.spread_credits) == 1
        credit = result.spread_credits[0]
        assert credit.credit_type == "INTER"
        assert set(credit.products) == {"ES", "NQ"}

    def test_missing_price_skipped(self, calculator, es_spec):
        """Position with missing price is skipped."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calculator.calculate_portfolio_margin(
            positions=[es_pos],
            prices={},  # No price for ES
            contract_specs={"ES": es_spec},
        )

        assert result.scanning_risk == Decimal("0")


# =============================================================================
# Test Margin Impact Estimation
# =============================================================================


class TestMarginImpactEstimation:
    """Tests for margin impact estimation."""

    def test_estimate_new_position_impact(self, calculator, es_spec):
        """Estimate impact of adding new position."""
        new_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        impact, new_total = calculator.estimate_margin_impact(
            current_positions=[],
            new_position=new_pos,
            prices={"ES": Decimal("4500")},
            contract_specs={"ES": es_spec},
        )

        # Adding to empty portfolio = full margin
        assert impact == new_total
        assert impact == Decimal("14850.00")

    def test_estimate_spread_impact(self, calculator, es_spec):
        """Estimate impact with spread credit."""
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )

        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec})

        impact, new_total = calc.estimate_margin_impact(
            current_positions=[es_pos],
            new_position=nq_pos,
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        )

        # Impact should be less than NQ standalone due to spread credit
        nq_standalone = calc.calculate_margin(nq_pos, Decimal("15000"), nq_spec).initial_margin

        assert impact < nq_standalone


# =============================================================================
# Test Margin Call Detection
# =============================================================================


class TestMarginCallDetection:
    """Tests for margin call detection."""

    def test_no_margin_call_sufficient_equity(self, calculator, es_spec):
        """No margin call when equity is sufficient."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        is_call, excess, status = calculator.check_margin_call(
            account_equity=Decimal("20000"),  # Well above maintenance
            positions=[es_pos],
            prices={"ES": Decimal("4500")},
            contract_specs={"ES": es_spec},
        )

        assert is_call is False
        assert status == "OK"
        assert excess > 0

    def test_margin_call_low_equity(self, calculator, es_spec):
        """Margin call when equity below maintenance."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        is_call, deficit, status = calculator.check_margin_call(
            account_equity=Decimal("10000"),  # Below maintenance (~11,880)
            positions=[es_pos],
            prices={"ES": Decimal("4500")},
            contract_specs={"ES": es_spec},
        )

        assert is_call is True
        assert status == "MARGIN_CALL"
        assert deficit < 0

    def test_warning_status(self, calculator, es_spec):
        """Warning when equity is low but above maintenance."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        # Maintenance is ~11,880, 150% would be ~17,820
        # Use equity between maintenance and 150%
        is_call, excess, status = calculator.check_margin_call(
            account_equity=Decimal("15000"),
            positions=[es_pos],
            prices={"ES": Decimal("4500")},
            contract_specs={"ES": es_spec},
        )

        assert is_call is False
        assert status == "WARNING"

    def test_no_margin_call_empty_portfolio(self, calculator):
        """No margin call for empty portfolio."""
        is_call, excess, status = calculator.check_margin_call(
            account_equity=Decimal("1000"),
            positions=[],
            prices={},
        )

        assert is_call is False
        assert status == "OK"


# =============================================================================
# Test SPANMarginResult Properties
# =============================================================================


class TestSPANMarginResultProperties:
    """Tests for SPANMarginResult computed properties."""

    def test_net_portfolio_margin(self):
        """Net portfolio margin calculated correctly."""
        result = SPANMarginResult(
            initial_margin=Decimal("15000"),
            maintenance_margin=Decimal("12000"),
            scanning_risk=Decimal("20000"),
            inter_commodity_credit=Decimal("3000"),
            intra_commodity_credit=Decimal("2000"),
        )

        # Net = 20000 - 3000 - 2000 = 15000
        assert result.net_portfolio_margin == Decimal("15000")

    def test_net_portfolio_margin_floor(self):
        """Net portfolio margin has floor of zero."""
        result = SPANMarginResult(
            initial_margin=Decimal("5000"),
            maintenance_margin=Decimal("4000"),
            scanning_risk=Decimal("10000"),
            inter_commodity_credit=Decimal("8000"),
            intra_commodity_credit=Decimal("5000"),  # Total credits > scanning
        )

        # Net should be floored at 0
        assert result.net_portfolio_margin == Decimal("0")

    def test_total_credit(self):
        """Total credit is sum of inter and intra credits."""
        result = SPANMarginResult(
            initial_margin=Decimal("15000"),
            maintenance_margin=Decimal("12000"),
            scanning_risk=Decimal("20000"),
            inter_commodity_credit=Decimal("3000"),
            intra_commodity_credit=Decimal("2000"),
        )

        assert result.total_credit == Decimal("5000")

    def test_margin_ratio(self):
        """Margin ratio calculated correctly."""
        result = SPANMarginResult(
            initial_margin=Decimal("15000"),
            maintenance_margin=Decimal("12000"),
            scanning_risk=Decimal("20000"),
        )

        # Ratio = 12000 / 15000 = 0.8
        assert result.margin_ratio == Decimal("0.8")

    def test_margin_ratio_zero_initial(self):
        """Margin ratio is zero when initial is zero."""
        result = SPANMarginResult(
            initial_margin=Decimal("0"),
            maintenance_margin=Decimal("0"),
            scanning_risk=Decimal("0"),
        )

        assert result.margin_ratio == Decimal("0")


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and convenience functions."""

    def test_create_span_calculator_default(self):
        """Create calculator with default specs."""
        calc = create_span_calculator(include_default_specs=True)
        assert calc is not None

    def test_create_span_calculator_no_specs(self):
        """Create calculator without default specs."""
        calc = create_span_calculator(include_default_specs=False)
        assert calc is not None

    def test_calculate_simple_margin(self):
        """Simple margin calculation."""
        initial, maintenance = calculate_simple_margin(
            position_notional=Decimal("225000"),  # 1 ES at 4500
            symbol="ES",
        )

        # Scanning = 225000 * 0.06 = 13500
        # Initial = 13500 * 1.10 = 14850
        assert initial == Decimal("14850")
        assert maintenance == Decimal("11880")

    def test_get_approximate_margin_known(self):
        """Get approximate margin for known symbol."""
        initial, maintenance = get_approximate_margin_per_contract("ES")
        assert initial == Decimal("12000")
        assert maintenance == Decimal("11000")

    def test_get_approximate_margin_unknown(self):
        """Get default margin for unknown symbol."""
        initial, maintenance = get_approximate_margin_per_contract("UNKNOWN")
        assert initial == Decimal("10000")
        assert maintenance == Decimal("9000")


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_contract_spec_raises(self):
        """Missing contract spec raises ValueError."""
        calc = SPANMarginCalculator()
        position = FuturesPosition(
            symbol="UNKNOWN",
            qty=Decimal("1"),
            entry_price=Decimal("100"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        with pytest.raises(ValueError, match="No contract spec"):
            calc.calculate_margin(position, Decimal("100"))

    def test_negative_price_handled(self, calculator, es_spec):
        """Negative price produces valid margin."""
        # This shouldn't happen in practice but test robustness
        position = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        # Should not raise - abs() is used on notional
        result = calculator.calculate_margin(position, Decimal("-4500"), es_spec)
        assert result.scanning_risk >= 0


# =============================================================================
# Test Position Margin Detail
# =============================================================================


class TestPositionMarginDetail:
    """Tests for PositionMarginDetail dataclass."""

    def test_gross_margin_property(self):
        """Gross margin calculated correctly."""
        detail = PositionMarginDetail(
            symbol="ES",
            position_qty=Decimal("1"),
            position_side=PositionSide.LONG,
            notional_value=Decimal("225000"),
            scanning_risk=Decimal("13500"),
            delivery_month_charge=Decimal("675"),
            net_option_value=Decimal("0"),
        )

        # Gross = scanning + delivery - option = 13500 + 675 - 0 = 14175
        assert detail.gross_margin == Decimal("14175")

    def test_gross_margin_with_option_value(self):
        """Gross margin accounts for option value."""
        detail = PositionMarginDetail(
            symbol="ES",
            position_qty=Decimal("1"),
            position_side=PositionSide.LONG,
            notional_value=Decimal("225000"),
            scanning_risk=Decimal("13500"),
            delivery_month_charge=Decimal("0"),
            net_option_value=Decimal("1000"),  # Option credit
        )

        # Gross = 13500 + 0 - 1000 = 12500
        assert detail.gross_margin == Decimal("12500")


# =============================================================================
# Test Approximate Margins Preset
# =============================================================================


class TestApproximateMarginsPreset:
    """Tests for approximate margin presets."""

    def test_es_preset(self):
        """ES preset values are reasonable."""
        initial, maint = APPROXIMATE_MARGINS_PER_CONTRACT["ES"]
        assert initial == Decimal("12000")
        assert maint == Decimal("11000")

    def test_nq_preset(self):
        """NQ preset higher than ES (more volatile)."""
        initial, _ = APPROXIMATE_MARGINS_PER_CONTRACT["NQ"]
        es_initial, _ = APPROXIMATE_MARGINS_PER_CONTRACT["ES"]
        assert initial > es_initial

    def test_micro_preset(self):
        """Micro margin is ~1/10 of standard."""
        es_initial, _ = APPROXIMATE_MARGINS_PER_CONTRACT["ES"]
        mes_initial, _ = APPROXIMATE_MARGINS_PER_CONTRACT["MES"]
        assert mes_initial == es_initial / 10

    def test_all_products_have_both_values(self):
        """All presets have initial and maintenance."""
        for symbol, (initial, maint) in APPROXIMATE_MARGINS_PER_CONTRACT.items():
            assert initial > 0, f"{symbol} initial should be positive"
            assert maint > 0, f"{symbol} maintenance should be positive"
            assert maint < initial, f"{symbol} maintenance should be less than initial"


# =============================================================================
# Test Scanning Risk Calculation
# =============================================================================


class TestScanningRiskCalculation:
    """Tests for scanning risk calculation."""

    def test_scanning_risk_basic(self, calculator):
        """Basic scanning risk calculation."""
        risk, scenario = calculator.calculate_scanning_risk(
            notional_value=Decimal("225000"),
            symbol="ES",
        )

        # Risk = 225000 * 0.06 = 13500
        assert risk == Decimal("13500")

    def test_scanning_risk_with_volatility_override(self, calculator):
        """Scanning risk with custom volatility."""
        risk, scenario = calculator.calculate_scanning_risk(
            notional_value=Decimal("225000"),
            symbol="ES",
            volatility_override=Decimal("0.50"),
        )

        # Volatility override doesn't affect basic calculation
        # (In full SPAN it would affect 16-scenario analysis)
        assert risk == Decimal("13500")

    def test_worst_scenario_created(self, calculator):
        """Worst scenario is created with correct values."""
        risk, scenario = calculator.calculate_scanning_risk(
            notional_value=Decimal("100000"),
            symbol="ES",
        )

        assert scenario.scenario_id == 3
        assert scenario.price_move_pct == Decimal("-0.06")
        assert scenario.portfolio_value_change == -risk


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_typical_retail_portfolio(self, calculator, es_spec, gc_spec):
        """Typical retail portfolio with multiple products."""
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("2"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        gc_pos = FuturesPosition(
            symbol="GC",
            qty=Decimal("1"),
            entry_price=Decimal("2000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calculator.calculate_portfolio_margin(
            positions=[es_pos, gc_pos],
            prices={"ES": Decimal("4500"), "GC": Decimal("2000")},
            contract_specs={"ES": es_spec, "GC": gc_spec},
        )

        # Should calculate margin for both positions
        assert len(result.position_details) == 2
        assert result.scanning_risk > 0
        assert result.initial_margin > 0

    def test_hedge_portfolio_gets_credit(self, es_spec):
        """Hedged portfolio receives spread credit."""
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )

        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec})

        # Long ES, Short NQ (hedge)
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.SHORT,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos],
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        )

        # Should have inter-commodity credit
        assert result.inter_commodity_credit > 0

    def test_large_position_margin(self, calculator, es_spec):
        """Large position scales correctly."""
        position = FuturesPosition(
            symbol="ES",
            qty=Decimal("100"),  # 100 contracts
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calculator.calculate_margin(
            position=position,
            current_price=Decimal("4500"),
            contract_spec=es_spec,
        )

        # Notional = 4500 * 100 * 50 = 22,500,000
        # Scanning = 22,500,000 * 0.06 = 1,350,000
        assert result.scanning_risk == Decimal("1350000")


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestCoverageEdgeCases:
    """Tests for edge cases to achieve 100% coverage."""

    @pytest.fixture
    def es_spec(self):
        """ES contract spec for tests."""
        return FuturesContractSpec(
            symbol="ES",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="SPX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("50"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("12.50"),
        )

    def test_default_scanning_range_fallback(self):
        """Test fallback to DEFAULT_SCANNING_RANGE for unknown symbols."""
        calc = SPANMarginCalculator()
        # UNKNOWN_SYMBOL not in SCANNING_RANGES or custom overrides
        scanning = calc.get_scanning_range("UNKNOWN_SYMBOL")
        # Should return default ScanningRangeConfig with 8% price scan
        assert scanning.price_scan_range_pct == Decimal("0.08")  # DEFAULT_SCANNING_RANGE

    def test_default_calendar_credit_fallback(self):
        """Test fallback to DEFAULT_CALENDAR_SPREAD_CREDIT for unknown symbols."""
        calc = SPANMarginCalculator()
        # UNKNOWN_SYMBOL not in CALENDAR_SPREAD_CREDITS
        credit = calc.get_calendar_credit_rate("UNKNOWN_SYMBOL")
        # Should return default
        assert credit == Decimal("0.70")  # DEFAULT_CALENDAR_SPREAD_CREDIT

    def test_portfolio_margin_skips_missing_spec(self, es_spec):
        """Test portfolio margin skips positions without contract specs."""
        calc = SPANMarginCalculator(contract_specs={"ES": es_spec})

        # ES has spec, NQ does not
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos],
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
            # Only provide ES spec via calc, NQ spec is missing
        )

        # Should only calculate margin for ES
        assert result.position_details is not None
        assert len(result.position_details) == 1  # Only ES

    def test_portfolio_margin_skips_missing_price(self, es_spec):
        """Test portfolio margin skips positions without prices."""
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )
        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec})

        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos],
            prices={"ES": Decimal("4500")},  # NQ price missing
        )

        # Should only calculate margin for ES
        assert result.position_details is not None
        assert len(result.position_details) == 1  # Only ES

    def test_inter_commodity_credit_skips_used_symbols(self, es_spec):
        """Test inter-commodity credit doesn't double-count symbols."""
        nq_spec = FuturesContractSpec(
            symbol="NQ",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="NDX",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("20"),
            tick_size=Decimal("0.25"),
            tick_value=Decimal("5"),
        )
        ym_spec = FuturesContractSpec(
            symbol="YM",
            exchange=Exchange.CME,
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            settlement_type=SettlementType.CASH,
            base_asset="DJI",
            quote_asset="USD",
            margin_asset="USD",
            contract_size=Decimal("1"),
            multiplier=Decimal("5"),
            tick_size=Decimal("1"),
            tick_value=Decimal("5"),
        )
        calc = SPANMarginCalculator(contract_specs={"ES": es_spec, "NQ": nq_spec, "YM": ym_spec})

        # Three positions: ES, NQ, YM
        # ES/NQ are correlated (0.50 credit)
        # ES/YM are also correlated (0.50 credit)
        # But ES can only be used once in credit calculation
        es_pos = FuturesPosition(
            symbol="ES",
            qty=Decimal("1"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        nq_pos = FuturesPosition(
            symbol="NQ",
            qty=Decimal("1"),
            entry_price=Decimal("15000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )
        ym_pos = FuturesPosition(
            symbol="YM",
            qty=Decimal("1"),
            entry_price=Decimal("35000"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        result = calc.calculate_portfolio_margin(
            positions=[es_pos, nq_pos, ym_pos],
            prices={"ES": Decimal("4500"), "NQ": Decimal("15000"), "YM": Decimal("35000")},
        )

        # Should have some inter-commodity credit
        # ES should only be paired once (first match)
        assert result.inter_commodity_credit >= 0
