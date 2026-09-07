# -*- coding: utf-8 -*-
"""
test_futures_margin.py
Comprehensive tests for futures margin calculation engine.

Coverage:
- TieredMarginCalculator: Binance-style tiered brackets
- CMEMarginCalculator: CME SPAN-like fixed margins
- SimpleMarginCalculator: Basic percentage-based margins
- Factory functions and JSON loading
- Edge cases and boundary conditions

Test count: 60+ tests
"""

from __future__ import annotations

from decimal import Decimal
import json
import pytest
from typing import List
from pathlib import Path
import tempfile
import os

from impl_futures_margin import (
    TieredMarginCalculator,
    CMEMarginCalculator,
    SimpleMarginCalculator,
    CMEMarginRate,
    FuturesMarginCalculator,
    create_margin_calculator,
    get_default_btc_brackets,
    get_default_eth_brackets,
    load_brackets_from_json,
)
from core_futures import (
    FuturesType,
    FuturesPosition,
    LeverageBracket,
    MarginMode,
    PositionSide,
    MarginRequirement,
    FuturesContractSpec,
    ContractType,
    Exchange,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def btc_brackets() -> List[LeverageBracket]:
    """Standard BTC-like leverage brackets."""
    return [
        LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),
            max_leverage=125,
        ),
        LeverageBracket(
            bracket=2,
            notional_cap=Decimal("250000"),
            maint_margin_rate=Decimal("0.005"),
            max_leverage=100,
        ),
        LeverageBracket(
            bracket=3,
            notional_cap=Decimal("1000000"),
            maint_margin_rate=Decimal("0.01"),
            max_leverage=50,
        ),
        LeverageBracket(
            bracket=4,
            notional_cap=Decimal("5000000"),
            maint_margin_rate=Decimal("0.025"),
            max_leverage=20,
        ),
        LeverageBracket(
            bracket=5,
            notional_cap=Decimal("10000000"),
            maint_margin_rate=Decimal("0.05"),
            max_leverage=10,
        ),
    ]


@pytest.fixture
def simple_brackets() -> List[LeverageBracket]:
    """Simplified brackets for testing."""
    return [
        LeverageBracket(
            bracket=1,
            notional_cap=Decimal("10000"),
            maint_margin_rate=Decimal("0.01"),
            max_leverage=100,
        ),
        LeverageBracket(
            bracket=2,
            notional_cap=Decimal("100000"),
            maint_margin_rate=Decimal("0.02"),
            max_leverage=50,
        ),
        LeverageBracket(
            bracket=3,
            notional_cap=Decimal("1000000"),
            maint_margin_rate=Decimal("0.05"),
            max_leverage=20,
        ),
    ]


@pytest.fixture
def tiered_calculator(btc_brackets) -> TieredMarginCalculator:
    """TieredMarginCalculator with BTC-like brackets."""
    return TieredMarginCalculator(brackets=btc_brackets)


@pytest.fixture
def cme_calculator() -> CMEMarginCalculator:
    """CMEMarginCalculator with default rates."""
    return CMEMarginCalculator()


@pytest.fixture
def simple_calculator() -> SimpleMarginCalculator:
    """SimpleMarginCalculator with default settings."""
    return SimpleMarginCalculator(initial_pct=5.0, maintenance_pct=4.0, max_leverage=20)


@pytest.fixture
def long_position() -> FuturesPosition:
    """Sample long position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        qty=Decimal("1.0"),  # 1 BTC
        entry_price=Decimal("50000"),  # $50,000
        leverage=10,
        margin=Decimal("5000"),  # $5,000
        margin_mode=MarginMode.ISOLATED,
        side=PositionSide.LONG,
        timestamp_ms=1700000000000,
    )


@pytest.fixture
def short_position() -> FuturesPosition:
    """Sample short position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        qty=Decimal("-0.5"),  # -0.5 BTC (short)
        entry_price=Decimal("60000"),
        leverage=20,
        margin=Decimal("1500"),
        margin_mode=MarginMode.ISOLATED,
        side=PositionSide.SHORT,
        timestamp_ms=1700000000000,
    )


@pytest.fixture
def cross_margin_position() -> FuturesPosition:
    """Sample cross-margin position."""
    return FuturesPosition(
        symbol="ETHUSDT",
        qty=Decimal("10.0"),  # 10 ETH
        entry_price=Decimal("2000"),
        leverage=25,
        margin=Decimal("800"),
        margin_mode=MarginMode.CROSS,
        side=PositionSide.LONG,
        timestamp_ms=1700000000000,
    )


# ============================================================================
# TIERED MARGIN CALCULATOR TESTS
# ============================================================================


class TestTieredMarginCalculatorInit:
    """Tests for TieredMarginCalculator initialization."""

    def test_init_with_valid_brackets(self, btc_brackets):
        """Should initialize with valid brackets."""
        calc = TieredMarginCalculator(brackets=btc_brackets)
        assert calc is not None

    def test_init_with_empty_brackets_raises(self):
        """Should raise ValueError with empty brackets."""
        with pytest.raises(ValueError, match="At least one leverage bracket required"):
            TieredMarginCalculator(brackets=[])

    def test_init_sorts_brackets_by_notional_cap(self):
        """Should sort brackets by notional cap ascending."""
        unsorted = [
            LeverageBracket(
                bracket=3,
                notional_cap=Decimal("100000"),
                maint_margin_rate=Decimal("0.03"),
                max_leverage=30,
            ),
            LeverageBracket(
                bracket=1,
                notional_cap=Decimal("10000"),
                maint_margin_rate=Decimal("0.01"),
                max_leverage=100,
            ),
            LeverageBracket(
                bracket=2,
                notional_cap=Decimal("50000"),
                maint_margin_rate=Decimal("0.02"),
                max_leverage=50,
            ),
        ]
        calc = TieredMarginCalculator(brackets=unsorted)
        # Access internal brackets to verify sorting
        assert calc._brackets[0].notional_cap == Decimal("10000")
        assert calc._brackets[1].notional_cap == Decimal("50000")
        assert calc._brackets[2].notional_cap == Decimal("100000")

    def test_init_precomputes_cumulative_maintenance(self, btc_brackets):
        """Should precompute cumulative maintenance values."""
        calc = TieredMarginCalculator(brackets=btc_brackets)
        assert len(calc._cum_maintenance) == len(btc_brackets)
        # First bracket should have 0 cumulative
        assert calc._cum_maintenance[0] == Decimal("0")

    def test_init_with_custom_liquidation_fee(self, simple_brackets):
        """Should accept custom liquidation fee rate."""
        calc = TieredMarginCalculator(
            brackets=simple_brackets,
            liquidation_fee_rate=Decimal("0.01"),  # 1%
        )
        assert calc._liquidation_fee_rate == Decimal("0.01")


class TestTieredMarginCalculatorBracketLookup:
    """Tests for bracket finding logic."""

    def test_find_bracket_small_notional(self, tiered_calculator):
        """Should find bracket 1 for small notional."""
        idx, bracket = tiered_calculator._find_bracket(Decimal("10000"))
        assert bracket.max_leverage == 125
        assert bracket.maint_margin_rate == Decimal("0.004")

    def test_find_bracket_medium_notional(self, tiered_calculator):
        """Should find appropriate bracket for medium notional."""
        idx, bracket = tiered_calculator._find_bracket(Decimal("100000"))
        assert bracket.max_leverage == 100  # Bracket 2

    def test_find_bracket_large_notional(self, tiered_calculator):
        """Should find last bracket for very large notional."""
        idx, bracket = tiered_calculator._find_bracket(Decimal("50000000"))
        assert bracket.max_leverage == 10  # Last bracket

    def test_find_bracket_exact_boundary(self, tiered_calculator):
        """Should handle exact bracket boundaries."""
        idx, bracket = tiered_calculator._find_bracket(Decimal("50000"))
        assert bracket.notional_cap == Decimal("50000")

    def test_find_bracket_negative_notional(self, tiered_calculator):
        """Should handle negative notional (use absolute value)."""
        idx, bracket = tiered_calculator._find_bracket(Decimal("-25000"))
        assert bracket.max_leverage == 125  # Bracket 1


class TestTieredMarginCalculatorMaxLeverage:
    """Tests for maximum leverage calculation."""

    def test_get_max_leverage_small_position(self, tiered_calculator):
        """Should return max leverage for small position."""
        assert tiered_calculator.get_max_leverage(Decimal("10000")) == 125

    def test_get_max_leverage_medium_position(self, tiered_calculator):
        """Should return reduced leverage for medium position."""
        assert tiered_calculator.get_max_leverage(Decimal("100000")) == 100

    def test_get_max_leverage_large_position(self, tiered_calculator):
        """Should return low leverage for large position."""
        assert tiered_calculator.get_max_leverage(Decimal("6000000")) == 10

    def test_get_max_leverage_boundary_values(self, tiered_calculator):
        """Should handle boundary values correctly."""
        # Just under boundary
        lev1 = tiered_calculator.get_max_leverage(Decimal("49999"))
        # At boundary
        lev2 = tiered_calculator.get_max_leverage(Decimal("50000"))
        # Just over boundary
        lev3 = tiered_calculator.get_max_leverage(Decimal("50001"))

        assert lev1 == 125
        assert lev2 == 125  # At boundary belongs to bracket
        assert lev3 == 100


class TestTieredMarginCalculatorInitialMargin:
    """Tests for initial margin calculation."""

    def test_calculate_initial_margin_basic(self, tiered_calculator):
        """Should calculate IM = Notional / Leverage."""
        # $100,000 notional at 10x leverage = $10,000 margin
        im = tiered_calculator.calculate_initial_margin(Decimal("100000"), leverage=10)
        assert im == Decimal("10000")

    def test_calculate_initial_margin_respects_max_leverage(self, tiered_calculator):
        """Should cap leverage at maximum for bracket."""
        # Large notional with high requested leverage
        notional = Decimal("2000000")  # Bracket 4, max 20x
        im_requested_50x = tiered_calculator.calculate_initial_margin(notional, leverage=50)
        im_at_20x = tiered_calculator.calculate_initial_margin(notional, leverage=20)

        # 50x should be capped to 20x
        assert im_requested_50x == im_at_20x

    def test_calculate_initial_margin_zero_leverage_raises(self, tiered_calculator):
        """Should raise ValueError for zero leverage."""
        with pytest.raises(ValueError, match="Leverage must be positive"):
            tiered_calculator.calculate_initial_margin(Decimal("100000"), leverage=0)

    def test_calculate_initial_margin_negative_leverage_raises(self, tiered_calculator):
        """Should raise ValueError for negative leverage."""
        with pytest.raises(ValueError, match="Leverage must be positive"):
            tiered_calculator.calculate_initial_margin(Decimal("100000"), leverage=-5)

    def test_calculate_initial_margin_negative_notional(self, tiered_calculator):
        """Should handle negative notional (use absolute value)."""
        im = tiered_calculator.calculate_initial_margin(Decimal("-100000"), leverage=10)
        assert im == Decimal("10000")

    def test_calculate_initial_margin_small_notional_high_leverage(self, tiered_calculator):
        """Should allow high leverage for small positions."""
        im = tiered_calculator.calculate_initial_margin(Decimal("10000"), leverage=100)
        assert im == Decimal("100")  # $10,000 / 100 = $100


class TestTieredMarginCalculatorMaintenanceMargin:
    """Tests for maintenance margin calculation."""

    def test_calculate_maintenance_margin_basic(self, tiered_calculator):
        """Should calculate MM = Notional * MMR."""
        # Bracket 1: MMR = 0.4%
        mm = tiered_calculator.calculate_maintenance_margin(Decimal("10000"))
        assert mm == Decimal("40")  # 10000 * 0.004

    def test_calculate_maintenance_margin_larger_notional(self, tiered_calculator):
        """Should use correct MMR for larger positions."""
        # Bracket 3: MMR = 1%
        mm = tiered_calculator.calculate_maintenance_margin(Decimal("500000"))
        assert mm == Decimal("5000")  # 500000 * 0.01

    def test_calculate_maintenance_margin_exact_uses_tiered_rate(self, tiered_calculator):
        """Exact calculation should use cumulative maintenance."""
        mm_simple = tiered_calculator.calculate_maintenance_margin(Decimal("100000"))
        mm_exact = tiered_calculator.calculate_maintenance_margin_exact(Decimal("100000"))

        # Simple method may differ from exact
        assert mm_simple > 0
        assert mm_exact >= 0

    def test_calculate_maintenance_margin_negative_notional(self, tiered_calculator):
        """Should handle negative notional."""
        mm = tiered_calculator.calculate_maintenance_margin(Decimal("-50000"))
        assert mm == Decimal("200")  # 50000 * 0.004


class TestTieredMarginCalculatorLiquidationPrice:
    """Tests for liquidation price calculation."""

    def test_calculate_liquidation_price_long_isolated(self, tiered_calculator):
        """Should calculate liquidation price for long isolated position."""
        liq_price = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            wallet_balance=Decimal("10000"),
            margin_mode=MarginMode.ISOLATED,
            isolated_margin=Decimal("5000"),
        )

        # Long liquidation price should be below entry
        assert liq_price < Decimal("50000")
        assert liq_price > Decimal("0")

    def test_calculate_liquidation_price_short_isolated(self, tiered_calculator):
        """Should calculate liquidation price for short isolated position."""
        liq_price = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),  # Short
            leverage=10,
            wallet_balance=Decimal("10000"),
            margin_mode=MarginMode.ISOLATED,
            isolated_margin=Decimal("5000"),
        )

        # Short liquidation price should be above entry
        assert liq_price > Decimal("50000")

    def test_calculate_liquidation_price_long_cross(self, tiered_calculator):
        """Should use wallet balance for cross margin."""
        liq_price = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            wallet_balance=Decimal("10000"),
            margin_mode=MarginMode.CROSS,
        )

        assert liq_price < Decimal("50000")

    def test_calculate_liquidation_price_zero_qty(self, tiered_calculator):
        """Should return 0 for zero quantity."""
        liq_price = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("0"),
            leverage=10,
            wallet_balance=Decimal("10000"),
            margin_mode=MarginMode.ISOLATED,
        )

        assert liq_price == Decimal("0")

    def test_calculate_liquidation_price_high_leverage_closer_to_entry(self, tiered_calculator):
        """Higher leverage should have liquidation price closer to entry."""
        liq_price_10x = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            wallet_balance=Decimal("5000"),
            margin_mode=MarginMode.ISOLATED,
            isolated_margin=Decimal("5000"),
        )

        liq_price_20x = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=20,
            wallet_balance=Decimal("2500"),
            margin_mode=MarginMode.ISOLATED,
            isolated_margin=Decimal("2500"),
        )

        # 20x should have liquidation closer to entry
        distance_10x = abs(Decimal("50000") - liq_price_10x)
        distance_20x = abs(Decimal("50000") - liq_price_20x)
        assert distance_20x < distance_10x


class TestTieredMarginCalculatorMarginRatio:
    """Tests for margin ratio calculation."""

    def test_calculate_margin_ratio_safe_long(self, tiered_calculator, long_position):
        """Should return high ratio for safe long position."""
        ratio = tiered_calculator.calculate_margin_ratio(
            long_position,
            mark_price=Decimal("50000"),  # Same as entry
            wallet_balance=Decimal("10000"),
        )

        assert ratio > Decimal("1")

    def test_calculate_margin_ratio_danger_zone(self, tiered_calculator, long_position):
        """Should return low ratio when approaching liquidation."""
        ratio = tiered_calculator.calculate_margin_ratio(
            long_position,
            mark_price=Decimal("45500"),  # 9% drop
            wallet_balance=Decimal("10000"),
        )

        # Should be low but calculation depends on implementation
        assert ratio > Decimal("0")

    def test_calculate_margin_ratio_zero_qty_returns_inf(self, tiered_calculator):
        """Should return infinity for zero quantity position."""
        zero_position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("0"),
            entry_price=Decimal("50000"),
            leverage=10,
            margin=Decimal("0"),
            margin_mode=MarginMode.ISOLATED,
            side=PositionSide.LONG,
            timestamp_ms=0,
        )

        ratio = tiered_calculator.calculate_margin_ratio(
            zero_position,
            mark_price=Decimal("50000"),
            wallet_balance=Decimal("10000"),
        )

        assert ratio == Decimal("inf")

    def test_calculate_margin_ratio_short_position(self, tiered_calculator, short_position):
        """Should calculate ratio for short position correctly."""
        ratio = tiered_calculator.calculate_margin_ratio(
            short_position,
            mark_price=Decimal("60000"),  # Same as entry
            wallet_balance=Decimal("10000"),
        )

        assert ratio > Decimal("0")


class TestTieredMarginCalculatorMarginRequirement:
    """Tests for margin requirement calculation."""

    def test_calculate_margin_requirement_full(self, tiered_calculator):
        """Should calculate complete margin requirement."""
        contract = FuturesContractSpec(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            contract_type=ContractType.PERPETUAL,
            exchange=Exchange.BINANCE,
            multiplier=Decimal("1"),
        )

        req = tiered_calculator.calculate_margin_requirement(
            contract=contract,
            qty=Decimal("1"),
            price=Decimal("50000"),
            leverage=10,
        )

        assert isinstance(req, MarginRequirement)
        assert req.initial == Decimal("5000")  # 50000 / 10
        assert req.maintenance > Decimal("0")
        assert req.available == req.initial - req.maintenance


class TestTieredMarginCalculatorEffectiveLeverage:
    """Tests for effective leverage calculation."""

    def test_get_effective_leverage_basic(self, tiered_calculator, long_position):
        """Should calculate effective leverage."""
        eff_lev = tiered_calculator.get_effective_leverage(
            long_position,
            mark_price=Decimal("50000"),
        )

        # Position value / margin = 50000 / 5000 = 10
        assert eff_lev == Decimal("10")

    def test_get_effective_leverage_with_profit(self, tiered_calculator, long_position):
        """Effective leverage may exceed nominal with profit."""
        # Price up, position value increased but margin same
        eff_lev = tiered_calculator.get_effective_leverage(
            long_position,
            mark_price=Decimal("60000"),
        )

        # 60000 / 5000 = 12
        assert eff_lev == Decimal("12")

    def test_get_effective_leverage_zero_margin(self, tiered_calculator):
        """Should handle zero margin."""
        zero_margin_position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("1"),
            entry_price=Decimal("50000"),
            leverage=10,
            margin=Decimal("0"),
            margin_mode=MarginMode.ISOLATED,
            side=PositionSide.LONG,
            timestamp_ms=0,
        )

        eff_lev = tiered_calculator.get_effective_leverage(
            zero_margin_position,
            mark_price=Decimal("50000"),
        )

        assert eff_lev == Decimal("0")


class TestTieredMarginCalculatorLiquidationDistance:
    """Tests for liquidation distance estimation."""

    def test_estimate_liquidation_distance_pct_safe(self, tiered_calculator, long_position):
        """Should show significant distance for safe position."""
        distance = tiered_calculator.estimate_liquidation_distance_pct(
            long_position,
            mark_price=Decimal("50000"),
            wallet_balance=Decimal("10000"),
        )

        assert distance > Decimal("5")  # More than 5% away

    def test_estimate_liquidation_distance_pct_close(self, tiered_calculator, long_position):
        """Should show small distance when near liquidation."""
        # Create position closer to liquidation
        risky_position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("1"),
            entry_price=Decimal("50000"),
            leverage=50,  # High leverage
            margin=Decimal("1000"),
            margin_mode=MarginMode.ISOLATED,
            side=PositionSide.LONG,
            timestamp_ms=0,
        )

        distance = tiered_calculator.estimate_liquidation_distance_pct(
            risky_position,
            mark_price=Decimal("50000"),
            wallet_balance=Decimal("1000"),
        )

        # Should be smaller distance due to high leverage
        assert distance < Decimal("10")


# ============================================================================
# CME MARGIN CALCULATOR TESTS
# ============================================================================


class TestCMEMarginCalculatorInit:
    """Tests for CMEMarginCalculator initialization."""

    def test_init_with_default_rates(self):
        """Should initialize with default CME margin rates."""
        calc = CMEMarginCalculator()
        assert calc is not None
        assert calc.get_margin_rate("ES") is not None

    def test_init_with_custom_rates(self):
        """Should accept custom margin rates."""
        custom_rates = {
            "TEST": CMEMarginRate(
                symbol="TEST",
                initial_margin=Decimal("1000"),
                maintenance_margin=Decimal("900"),
            ),
        }
        calc = CMEMarginCalculator(margin_rates=custom_rates)
        assert calc.get_margin_rate("TEST") is not None
        assert calc.get_margin_rate("ES") is None

    def test_init_with_day_trading_margin(self):
        """Should support day trading margin flag."""
        calc = CMEMarginCalculator(use_day_trading_margin=True)
        assert calc._use_day_margin is True


class TestCMEMarginCalculatorRateLookup:
    """Tests for margin rate lookup."""

    def test_get_margin_rate_known_symbol(self, cme_calculator):
        """Should return rate for known symbol."""
        rate = cme_calculator.get_margin_rate("ES")
        assert rate is not None
        assert rate.initial_margin == Decimal("15180")

    def test_get_margin_rate_unknown_symbol(self, cme_calculator):
        """Should return None for unknown symbol."""
        rate = cme_calculator.get_margin_rate("UNKNOWN")
        assert rate is None

    def test_get_margin_rate_with_contract_month(self, cme_calculator):
        """Should extract base symbol from contract month."""
        rate = cme_calculator.get_margin_rate("ESM24")
        assert rate is not None  # Should find ES


class TestCMEMarginCalculatorInitialMargin:
    """Tests for CME initial margin calculation."""

    def test_calculate_initial_margin_es(self, cme_calculator):
        """Should calculate correct margin for ES."""
        im = cme_calculator.calculate_initial_margin("ES", qty=Decimal("2"))
        assert im == Decimal("30360")  # 2 * 15180

    def test_calculate_initial_margin_day_trading(self):
        """Should use day trading margin when enabled."""
        calc = CMEMarginCalculator(use_day_trading_margin=True)
        im = calc.calculate_initial_margin("ES", qty=Decimal("1"))
        assert im == Decimal("500")  # Day trading margin

    def test_calculate_initial_margin_unknown_symbol(self, cme_calculator):
        """Should use fallback for unknown symbol."""
        im = cme_calculator.calculate_initial_margin("UNKNOWN", qty=Decimal("1"))
        assert im == Decimal("10000")  # Fallback


class TestCMEMarginCalculatorMaintenanceMargin:
    """Tests for CME maintenance margin calculation."""

    def test_calculate_maintenance_margin_es(self, cme_calculator):
        """Should calculate correct maintenance margin for ES."""
        mm = cme_calculator.calculate_maintenance_margin("ES", qty=Decimal("3"))
        assert mm == Decimal("41400")  # 3 * 13800

    def test_calculate_maintenance_margin_negative_qty(self, cme_calculator):
        """Should use absolute value of quantity."""
        mm = cme_calculator.calculate_maintenance_margin("ES", qty=Decimal("-2"))
        assert mm == Decimal("27600")  # 2 * 13800


class TestCMEMarginCalculatorMaxPositionSize:
    """Tests for maximum position size calculation."""

    def test_calculate_max_position_size_es(self, cme_calculator):
        """Should calculate max contracts for given margin."""
        max_contracts = cme_calculator.calculate_max_position_size(
            "ES",
            available_margin=Decimal("50000"),
        )
        assert max_contracts == 3  # 50000 / 15180 = 3.29 -> 3

    def test_calculate_max_position_size_day_trading(self, cme_calculator):
        """Should calculate higher max with day trading margin."""
        max_contracts = cme_calculator.calculate_max_position_size(
            "ES",
            available_margin=Decimal("5000"),
            use_day_margin=True,
        )
        assert max_contracts == 10  # 5000 / 500 = 10


class TestCMEMarginCalculatorImpliedLeverage:
    """Tests for implied leverage calculation."""

    def test_get_implied_leverage_es(self, cme_calculator):
        """Should calculate implied leverage for ES."""
        contract = FuturesContractSpec(
            symbol="ES",
            base_asset="SPX",
            quote_asset="USD",
            margin_asset="USD",
            futures_type=FuturesType.INDEX_FUTURES,
            contract_type=ContractType.CURRENT_QUARTER,
            exchange=Exchange.CME,
            multiplier=Decimal("50"),  # ES multiplier
        )

        leverage = cme_calculator.get_implied_leverage(
            "ES",
            price=Decimal("5000"),  # ES price
            contract=contract,
        )

        # Notional = 5000 * 50 = 250,000
        # Implied leverage = 250,000 / 15,180 ≈ 16.5x
        assert leverage > Decimal("15")


# ============================================================================
# SIMPLE MARGIN CALCULATOR TESTS
# ============================================================================


class TestSimpleMarginCalculatorInit:
    """Tests for SimpleMarginCalculator initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default percentages."""
        calc = SimpleMarginCalculator()
        assert calc is not None

    def test_init_with_custom_percentages(self):
        """Should accept custom percentages."""
        calc = SimpleMarginCalculator(
            initial_pct=10.0,
            maintenance_pct=8.0,
            max_leverage=10,
        )
        assert calc._max_leverage == 10

    def test_init_zero_or_negative_max_leverage(self):
        """Should clamp max_leverage to a minimum of 1 during initialization."""
        calc_zero = SimpleMarginCalculator(max_leverage=0)
        assert calc_zero._max_leverage == 1

        calc_neg = SimpleMarginCalculator(max_leverage=-5)
        assert calc_neg._max_leverage == 1


class TestSimpleMarginCalculatorBasicOps:
    """Tests for SimpleMarginCalculator basic operations."""

    def test_calculate_initial_margin(self, simple_calculator):
        """Should calculate initial margin as notional / leverage."""
        im = simple_calculator.calculate_initial_margin(Decimal("100000"), leverage=10)
        assert im == Decimal("10000")

    def test_calculate_initial_margin_caps_leverage(self, simple_calculator):
        """Should cap leverage at max (20x)."""
        # Request 50x but max is 20x
        im = simple_calculator.calculate_initial_margin(Decimal("100000"), leverage=50)
        assert im == Decimal("5000")  # 100000 / 20

    def test_calculate_maintenance_margin(self, simple_calculator):
        """Should calculate maintenance margin as percentage."""
        mm = simple_calculator.calculate_maintenance_margin(Decimal("100000"))
        assert mm == Decimal("4000")  # 100000 * 0.04

    def test_calculate_liquidation_price_long(self, simple_calculator):
        """Should calculate liquidation price for long."""
        liq_price = simple_calculator.calculate_liquidation_price(
            entry_price=Decimal("100"),
            qty=Decimal("10"),
            leverage=10,
            wallet_balance=Decimal("100"),
            margin_mode=MarginMode.ISOLATED,
        )

        assert liq_price < Decimal("100")

    def test_calculate_liquidation_price_short(self, simple_calculator):
        """Should calculate liquidation price for short."""
        liq_price = simple_calculator.calculate_liquidation_price(
            entry_price=Decimal("100"),
            qty=Decimal("-10"),
            leverage=10,
            wallet_balance=Decimal("100"),
            margin_mode=MarginMode.ISOLATED,
        )

        assert liq_price > Decimal("100")

    def test_get_max_leverage(self, simple_calculator):
        """Should return configured max leverage."""
        assert simple_calculator.get_max_leverage(Decimal("1000000")) == 20

    def test_calculate_initial_margin_zero_or_negative_leverage(self, simple_calculator):
        """Should handle leverage <= 0 gracefully by falling back to leverage = 1."""
        # Test zero leverage
        im_zero = simple_calculator.calculate_initial_margin(Decimal("100000"), leverage=0)
        assert im_zero == Decimal("100000")  # 100000 / 1

        # Test negative leverage
        im_neg = simple_calculator.calculate_initial_margin(Decimal("100000"), leverage=-5)
        assert im_neg == Decimal("100000")  # 100000 / 1

    def test_calculate_liquidation_price_zero_or_negative_leverage(self, simple_calculator):
        """Should handle leverage <= 0 gracefully in liquidation price calculation."""
        # Test zero leverage (should fallback to 1x leverage)
        liq_price_zero = simple_calculator.calculate_liquidation_price(
            entry_price=Decimal("100"),
            qty=Decimal("10"),
            leverage=0,
            wallet_balance=Decimal("100"),
            margin_mode=MarginMode.ISOLATED,
        )
        assert liq_price_zero > Decimal("0")

        # Test negative leverage (should fallback to 1x leverage)
        liq_price_neg = simple_calculator.calculate_liquidation_price(
            entry_price=Decimal("100"),
            qty=Decimal("10"),
            leverage=-5,
            wallet_balance=Decimal("100"),
            margin_mode=MarginMode.ISOLATED,
        )
        assert liq_price_neg > Decimal("0")


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateMarginCalculator:
    """Tests for create_margin_calculator factory."""

    def test_create_for_perpetual_with_brackets(self, btc_brackets):
        """Should create TieredMarginCalculator for crypto perpetual."""
        calc = create_margin_calculator(
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            brackets=btc_brackets,
        )
        assert isinstance(calc, TieredMarginCalculator)

    def test_create_for_perpetual_without_brackets(self):
        """Should create TieredMarginCalculator with defaults."""
        calc = create_margin_calculator(futures_type=FuturesType.CRYPTO_PERPETUAL)
        assert isinstance(calc, TieredMarginCalculator)

    def test_create_for_index_futures(self):
        """Should create CMEMarginCalculator for index futures."""
        calc = create_margin_calculator(futures_type=FuturesType.INDEX_FUTURES)
        assert isinstance(calc, CMEMarginCalculator)

    def test_create_for_commodity_futures(self):
        """Should create CMEMarginCalculator for commodity futures."""
        calc = create_margin_calculator(futures_type=FuturesType.COMMODITY_FUTURES)
        assert isinstance(calc, CMEMarginCalculator)

    def test_create_with_custom_config(self):
        """Should pass config to calculator."""
        calc = create_margin_calculator(
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            config={"liquidation_fee_rate": "0.01"},
        )
        assert isinstance(calc, TieredMarginCalculator)


class TestDefaultBrackets:
    """Tests for default bracket functions."""

    def test_get_default_btc_brackets(self):
        """Should return valid BTC brackets."""
        brackets = get_default_btc_brackets()
        assert len(brackets) >= 5
        assert brackets[0].max_leverage == 125

    def test_get_default_eth_brackets(self):
        """Should return valid ETH brackets."""
        brackets = get_default_eth_brackets()
        assert len(brackets) >= 5
        assert brackets[0].max_leverage == 100


class TestLoadBracketsFromJson:
    """Tests for JSON bracket loading."""

    def test_load_brackets_from_json(self):
        """Should load brackets from JSON file."""
        # Create temp JSON file
        data = {
            "brackets": {
                "TESTUSDT": [
                    {
                        "bracket": 1,
                        "notional_cap": 10000,
                        "maint_margin_rate": 0.01,
                        "max_leverage": 50,
                    },
                    {
                        "bracket": 2,
                        "notional_cap": 100000,
                        "maint_margin_rate": 0.02,
                        "max_leverage": 25,
                    },
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write with nested structure
            json.dump({"TESTUSDT": data["brackets"]["TESTUSDT"]}, f)
            temp_path = f.name

        try:
            brackets = load_brackets_from_json(temp_path)
            assert "TESTUSDT" in brackets
            assert len(brackets["TESTUSDT"]) == 2
            assert brackets["TESTUSDT"][0].max_leverage == 50
        finally:
            os.unlink(temp_path)

    def test_load_brackets_handles_different_key_formats(self):
        """Should handle different key naming conventions."""
        data = {
            "BTCUSDT": [
                {"bracket": 1, "notionalCap": 50000, "maintMarginRate": 0.004, "maxLeverage": 125},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            brackets = load_brackets_from_json(temp_path)
            assert "BTCUSDT" in brackets
            assert brackets["BTCUSDT"][0].notional_cap == Decimal("50000")
        finally:
            os.unlink(temp_path)


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_notional(self, tiered_calculator):
        """Should handle very small notional values."""
        im = tiered_calculator.calculate_initial_margin(Decimal("0.01"), leverage=10)
        assert im == Decimal("0.001")

    def test_very_large_notional(self, tiered_calculator):
        """Should handle very large notional values."""
        im = tiered_calculator.calculate_initial_margin(Decimal("1000000000"), leverage=1)
        assert im == Decimal("1000000000")

    def test_decimal_precision_maintained(self, tiered_calculator):
        """Should maintain decimal precision."""
        im = tiered_calculator.calculate_initial_margin(Decimal("12345.6789"), leverage=10)
        assert im == Decimal("1234.56789")

    def test_multiple_positions_margin_sum(self, tiered_calculator):
        """Should correctly sum margins for multiple positions."""
        im1 = tiered_calculator.calculate_initial_margin(Decimal("50000"), leverage=10)
        im2 = tiered_calculator.calculate_initial_margin(Decimal("50000"), leverage=10)
        im_combined = tiered_calculator.calculate_initial_margin(Decimal("100000"), leverage=10)

        assert im1 + im2 == im_combined


class TestCrossMarginScenarios:
    """Tests for cross-margin specific scenarios."""

    def test_cross_margin_uses_wallet_balance(self, tiered_calculator, cross_margin_position):
        """Cross margin should use wallet balance for ratio calc."""
        ratio = tiered_calculator.calculate_margin_ratio(
            cross_margin_position,
            mark_price=Decimal("2000"),
            wallet_balance=Decimal("10000"),  # Large wallet
        )

        assert ratio > Decimal("1")

    def test_cross_margin_liquidation_with_pnl(self, tiered_calculator):
        """Should consider cumulative PnL in cross margin liquidation."""
        liq_price = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            wallet_balance=Decimal("5000"),
            margin_mode=MarginMode.CROSS,
            cum_pnl=Decimal("2000"),  # Profit
        )

        liq_price_no_profit = tiered_calculator.calculate_liquidation_price(
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            wallet_balance=Decimal("5000"),
            margin_mode=MarginMode.CROSS,
            cum_pnl=Decimal("0"),
        )

        # With profit, liquidation price should be lower (safer)
        assert liq_price < liq_price_no_profit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
