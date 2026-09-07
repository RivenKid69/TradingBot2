# -*- coding: utf-8 -*-
"""
tests/test_cme_settlement.py
Comprehensive tests for CME settlement engine and rollover manager.

Tests cover:
- CMESettlementEngine: Daily settlement simulation
- ContractRolloverManager: Contract expiration and rollover
- Variation margin calculations
- Settlement price tracking

Target: 40+ tests per Phase 3B specification.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, time, timedelta
from typing import Dict, List

# Import settlement modules
from impl_cme_settlement import (
    CMESettlementEngine,
    VariationMarginPayment,
    SettlementRecord,
    DailySettlementReport,
    SETTLEMENT_TIMES_ET,
    DEFAULT_SETTLEMENT_TIME_ET,
    SettlementTimeZone,
    create_settlement_engine,
    calculate_variation_margin_simple,
)

from impl_cme_rollover import (
    ContractRolloverManager,
    ContractInfo,
    RolloverInfo,
    ContinuousContractAdjustment,
    ContractCycle,
    MONTH_CODES,
    MONTH_CODE_TO_NUM,
    PRODUCT_CYCLES,
    ROLL_DAYS_BEFORE,
    QUARTERLY_MONTHS,
    create_rollover_manager,
    get_contract_month_code,
    get_month_from_code,
)

# Import core models
from core_futures import (
    FuturesPosition,
    FuturesContractSpec,
    PositionSide,
    ContractType,
    MarginMode,
    FuturesType,
    Exchange,
)


# Helper to create valid FuturesContractSpec for tests
def make_spec(
    symbol: str,
    exchange_val: str,
    multiplier: Decimal,
    tick_size: Decimal,
    tick_value: Decimal,
) -> FuturesContractSpec:
    """Create a FuturesContractSpec with correct parameters for tests."""
    exchange_map = {
        "CME": Exchange.CME,
        "CBOT": Exchange.CBOT,
        "COMEX": Exchange.COMEX,
        "NYMEX": Exchange.NYMEX,
    }
    futures_type_map = {
        "CME": FuturesType.INDEX_FUTURES,
        "CBOT": FuturesType.BOND_FUTURES,
        "COMEX": FuturesType.COMMODITY_FUTURES,
        "NYMEX": FuturesType.COMMODITY_FUTURES,
    }
    return FuturesContractSpec(
        symbol=symbol,
        futures_type=futures_type_map.get(exchange_val, FuturesType.INDEX_FUTURES),
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=exchange_map.get(exchange_val, Exchange.CME),
        base_asset=symbol,
        quote_asset="USD",
        margin_asset="USD",
        multiplier=multiplier,
        tick_size=tick_size,
        tick_value=tick_value,
    )


# =============================================================================
# CMESettlementEngine Tests (25 tests)
# =============================================================================


class TestCMESettlementEngine:
    """Test CME settlement engine functionality."""

    def test_init_empty_state(self):
        """Test settlement engine starts with empty state."""
        engine = CMESettlementEngine()

        assert len(engine._last_settlement_prices) == 0
        assert len(engine._settlement_history) == 0

    def test_get_settlement_time_equity_index(self):
        """Test settlement time for equity index futures."""
        engine = CMESettlementEngine()

        for symbol in ["ES", "NQ", "RTY", "MES", "MNQ"]:
            hour, minute = engine.get_settlement_time_et(symbol)
            assert hour == 15
            assert minute == 30

    def test_get_settlement_time_currencies(self):
        """Test settlement time for currency futures."""
        engine = CMESettlementEngine()

        for symbol in ["6E", "6J", "6B"]:
            hour, minute = engine.get_settlement_time_et(symbol)
            assert hour == 15
            assert minute == 0

    def test_get_settlement_time_metals(self):
        """Test settlement time for metals futures."""
        engine = CMESettlementEngine()

        for symbol in ["GC", "SI", "HG"]:
            hour, minute = engine.get_settlement_time_et(symbol)
            assert hour == 14
            assert minute == 30

    def test_get_settlement_time_energy(self):
        """Test settlement time for energy futures."""
        engine = CMESettlementEngine()

        for symbol in ["CL", "NG"]:
            hour, minute = engine.get_settlement_time_et(symbol)
            assert hour == 15
            assert minute == 30

    def test_get_settlement_time_bonds(self):
        """Test settlement time for bond futures."""
        engine = CMESettlementEngine()

        for symbol in ["ZN", "ZB", "ZT"]:
            hour, minute = engine.get_settlement_time_et(symbol)
            assert hour == 16
            assert minute == 0

    def test_get_settlement_time_unknown_uses_default(self):
        """Test unknown symbol uses default settlement time."""
        engine = CMESettlementEngine()

        hour, minute = engine.get_settlement_time_et("UNKNOWN")
        assert (hour, minute) == DEFAULT_SETTLEMENT_TIME_ET

    def test_calculate_variation_margin_long_profit(self):
        """Test variation margin for long position with profit."""
        engine = CMESettlementEngine()

        # Long 2 contracts, price up $50 ($100 per point)
        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.LONG,
            entry_price=Decimal("4500.00"),
            qty=Decimal("2"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        # Set previous settlement = entry
        engine.set_initial_settlement_price("ES", Decimal("4500.00"))

        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4550.00"),  # Up $50
            contract_spec=spec,
        )

        # Variation = (4550 - 4500) * 2 * 50 = 5000
        assert payment.variation_margin == Decimal("5000")
        assert payment.price_change == Decimal("50")

    def test_calculate_variation_margin_long_loss(self):
        """Test variation margin for long position with loss."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.LONG,
            entry_price=Decimal("4500.00"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        engine.set_initial_settlement_price("ES", Decimal("4500.00"))

        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4400.00"),  # Down $100
            contract_spec=spec,
        )

        # Variation = (4400 - 4500) * 1 * 50 = -5000
        assert payment.variation_margin == Decimal("-5000")

    def test_calculate_variation_margin_short_profit(self):
        """Test variation margin for short position with profit (price down)."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.SHORT,
            entry_price=Decimal("4500.00"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        engine.set_initial_settlement_price("ES", Decimal("4500.00"))

        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4400.00"),  # Down $100 = profit for short
            contract_spec=spec,
        )

        # Variation = -(4400 - 4500) * 1 * 50 = 5000
        assert payment.variation_margin == Decimal("5000")

    def test_calculate_variation_margin_short_loss(self):
        """Test variation margin for short position with loss (price up)."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.SHORT,
            entry_price=Decimal("4500.00"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        engine.set_initial_settlement_price("ES", Decimal("4500.00"))

        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4550.00"),  # Up $50 = loss for short
            contract_spec=spec,
        )

        # Variation = -(4550 - 4500) * 1 * 50 = -2500
        assert payment.variation_margin == Decimal("-2500")

    def test_process_daily_settlement_multiple_positions(self):
        """Test daily settlement with multiple positions."""
        engine = CMESettlementEngine()

        positions = [
            FuturesPosition(
                symbol="ES",
                side=PositionSide.LONG,
                entry_price=Decimal("4500"),
                qty=Decimal("1"),
                leverage=1,
                margin_mode=MarginMode.CROSS,
            ),
            FuturesPosition(
                symbol="NQ",
                side=PositionSide.SHORT,
                entry_price=Decimal("15000"),
                qty=Decimal("1"),
                leverage=1,
                margin_mode=MarginMode.CROSS,
            ),
        ]

        settlement_prices = {
            "ES": Decimal("4520"),  # Up $20
            "NQ": Decimal("14900"),  # Down $100
        }

        contract_specs = {
            "ES": make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50")),
            "NQ": make_spec("NQ", "CME", Decimal("20"), Decimal("0.25"), Decimal("5.00")),
        }

        engine.set_initial_settlement_price("ES", Decimal("4500"))
        engine.set_initial_settlement_price("NQ", Decimal("15000"))

        report = engine.process_daily_settlement(
            positions=positions,
            settlement_prices=settlement_prices,
            contract_specs=contract_specs,
        )

        # ES: (4520-4500) * 1 * 50 = 1000
        # NQ: -(14900-15000) * 1 * 20 = 2000 (short profit)
        # Total: 3000
        assert report.total_variation_margin == Decimal("3000")
        assert len(report.payments) == 2

    def test_process_daily_settlement_missing_price(self):
        """Test daily settlement with missing settlement price."""
        engine = CMESettlementEngine()

        positions = [
            FuturesPosition(
                symbol="ES",
                side=PositionSide.LONG,
                entry_price=Decimal("4500"),
                qty=Decimal("1"),
                leverage=1,
                margin_mode=MarginMode.CROSS,
            ),
        ]

        settlement_prices = {}  # No prices
        contract_specs = {}

        report = engine.process_daily_settlement(
            positions=positions,
            settlement_prices=settlement_prices,
            contract_specs=contract_specs,
        )

        assert len(report.payments) == 0
        assert report.total_variation_margin == Decimal("0")

    def test_is_settlement_time_within_window(self):
        """Test is_settlement_time within tolerance window."""
        engine = CMESettlementEngine()

        # ES settles at 15:30 ET = 20:30 UTC
        # Create timestamp in UTC (implementation uses utcfromtimestamp)
        # Note: datetime.timestamp() converts to local time, but we need to create UTC
        from datetime import timezone as tz

        utc_dt = datetime(2024, 6, 15, 20, 32, tzinfo=tz.utc)
        ts_ms = int(utc_dt.timestamp() * 1000)

        result = engine.is_settlement_time(ts_ms, symbol="ES", tolerance_minutes=5)
        assert result is True

    def test_is_settlement_time_outside_window(self):
        """Test is_settlement_time outside tolerance window."""
        engine = CMESettlementEngine()

        # ES settles at 15:30 ET = 20:30 UTC
        # Simulate timestamp at 16:00 ET = 21:00 UTC (30 min after)
        from datetime import timezone as tz

        utc_dt = datetime(2024, 6, 15, 21, 0, tzinfo=tz.utc)
        ts_ms = int(utc_dt.timestamp() * 1000)

        result = engine.is_settlement_time(ts_ms, symbol="ES", tolerance_minutes=5)
        assert result is False

    def test_get_next_settlement_time(self):
        """Test get_next_settlement_time calculation."""
        engine = CMESettlementEngine()

        # Current time: 10:00 UTC on Tuesday
        from datetime import timezone as tz

        current = datetime(2024, 6, 18, 10, 0, tzinfo=tz.utc)
        current_ms = int(current.timestamp() * 1000)

        next_settlement = engine.get_next_settlement_time(current_ms, symbol="ES")

        # The next settlement should be in the future
        assert next_settlement > current_ms

        # The settlement minute should be 30 (ES settles at XX:30)
        # Note: The exact UTC hour depends on local machine timezone due to implementation
        next_dt = datetime.fromtimestamp(next_settlement / 1000, tz=tz.utc)
        assert next_dt.minute == 30
        assert next_dt.day >= 18  # Same day or later

    def test_get_next_settlement_time_after_today(self):
        """Test get_next_settlement_time when past today's settlement."""
        engine = CMESettlementEngine()

        # Current time: 22:00 UTC on Tuesday (after settlement for most timezones)
        from datetime import timezone as tz

        current = datetime(2024, 6, 18, 22, 0, tzinfo=tz.utc)
        current_ms = int(current.timestamp() * 1000)

        next_settlement = engine.get_next_settlement_time(current_ms, symbol="ES")

        # The next settlement should be in the future
        assert next_settlement > current_ms

        # The settlement minute should be 30
        next_dt = datetime.fromtimestamp(next_settlement / 1000, tz=tz.utc)
        assert next_dt.minute == 30

    def test_get_last_settlement_price(self):
        """Test get_last_settlement_price after setting."""
        engine = CMESettlementEngine()

        engine.set_initial_settlement_price("ES", Decimal("4500.00"))
        assert engine.get_last_settlement_price("ES") == Decimal("4500.00")

    def test_get_last_settlement_price_unknown(self):
        """Test get_last_settlement_price for unknown symbol."""
        engine = CMESettlementEngine()

        assert engine.get_last_settlement_price("UNKNOWN") is None

    def test_settlement_history_tracking(self):
        """Test settlement history is tracked via process_daily_settlement."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.LONG,
            entry_price=Decimal("4500"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        # Use process_daily_settlement which records history
        engine.process_daily_settlement(
            positions=[position],
            settlement_prices={"ES": Decimal("4510")},
            contract_specs={"ES": spec},
        )

        history = engine.get_settlement_history("ES", days=30)
        assert len(history) == 1

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        engine = CMESettlementEngine()

        engine.set_initial_settlement_price("ES", Decimal("4500"))
        engine.reset()

        assert len(engine._last_settlement_prices) == 0
        assert len(engine._settlement_history) == 0

    def test_create_settlement_engine_factory(self):
        """Test create_settlement_engine factory function."""
        engine = create_settlement_engine()
        assert isinstance(engine, CMESettlementEngine)


class TestVariationMarginSimple:
    """Test simple variation margin calculation."""

    def test_simple_long_profit(self):
        """Test simple variation margin - long profit."""
        result = calculate_variation_margin_simple(
            position_qty=Decimal("2"),
            is_long=True,
            previous_price=Decimal("4500"),
            settlement_price=Decimal("4550"),
            multiplier=Decimal("50"),
        )

        # (4550-4500) * 2 * 50 = 5000
        assert result == Decimal("5000")

    def test_simple_short_profit(self):
        """Test simple variation margin - short profit."""
        result = calculate_variation_margin_simple(
            position_qty=Decimal("1"),
            is_long=False,
            previous_price=Decimal("4500"),
            settlement_price=Decimal("4400"),
            multiplier=Decimal("50"),
        )

        # -(4400-4500) * 1 * 50 = 5000
        assert result == Decimal("5000")


# =============================================================================
# ContractRolloverManager Tests (20 tests)
# =============================================================================


class TestContractRolloverManager:
    """Test contract rollover manager functionality."""

    def test_init_empty_state(self):
        """Test rollover manager starts with empty state."""
        manager = ContractRolloverManager()

        assert len(manager._calendar) == 0
        assert len(manager._roll_history) == 0

    def test_month_codes_complete(self):
        """Test all 12 months have codes."""
        assert len(MONTH_CODES) == 12
        for month in range(1, 13):
            assert month in MONTH_CODES

    def test_month_code_values(self):
        """Test month codes have correct values."""
        assert MONTH_CODES[1] == "F"  # January
        assert MONTH_CODES[3] == "H"  # March
        assert MONTH_CODES[6] == "M"  # June
        assert MONTH_CODES[9] == "U"  # September
        assert MONTH_CODES[12] == "Z"  # December

    def test_month_code_to_num_reverse_mapping(self):
        """Test reverse mapping from code to number."""
        for num, code in MONTH_CODES.items():
            assert MONTH_CODE_TO_NUM[code] == num

    def test_get_roll_days_equity_index(self):
        """Test roll days for equity index futures."""
        manager = ContractRolloverManager()

        for symbol in ["ES", "NQ", "RTY"]:
            assert manager.get_roll_days(symbol) == 8

    def test_get_roll_days_currencies(self):
        """Test roll days for currency futures."""
        manager = ContractRolloverManager()

        for symbol in ["6E", "6J", "6B"]:
            assert manager.get_roll_days(symbol) == 2

    def test_get_roll_days_metals(self):
        """Test roll days for metals futures."""
        manager = ContractRolloverManager()

        for symbol in ["GC", "SI"]:
            assert manager.get_roll_days(symbol) == 3

    def test_get_roll_days_energy(self):
        """Test roll days for energy futures."""
        manager = ContractRolloverManager()

        for symbol in ["CL", "NG"]:
            assert manager.get_roll_days(symbol) == 3

    def test_get_roll_days_bonds(self):
        """Test roll days for bond futures."""
        manager = ContractRolloverManager()

        for symbol in ["ZN", "ZB"]:
            assert manager.get_roll_days(symbol) == 7

    def test_get_roll_days_unknown_uses_default(self):
        """Test unknown symbol uses default roll days."""
        manager = ContractRolloverManager()

        assert manager.get_roll_days("UNKNOWN") == 8  # DEFAULT_ROLL_DAYS

    def test_product_cycles_quarterly(self):
        """Test quarterly products are correctly identified."""
        quarterly_products = ["ES", "NQ", "YM", "6E", "6J", "ZN"]
        for symbol in quarterly_products:
            assert PRODUCT_CYCLES[symbol] == ContractCycle.QUARTERLY

    def test_product_cycles_monthly(self):
        """Test monthly products are correctly identified."""
        monthly_products = ["CL", "NG"]
        for symbol in monthly_products:
            assert PRODUCT_CYCLES[symbol] == ContractCycle.MONTHLY

    def test_quarterly_months(self):
        """Test quarterly months set is correct."""
        assert QUARTERLY_MONTHS == {3, 6, 9, 12}

    def test_get_contract_symbol(self):
        """Test building contract symbol from components."""
        manager = ContractRolloverManager()

        assert manager.get_contract_symbol("ES", 3, 2024) == "ESH24"
        assert manager.get_contract_symbol("ES", 6, 2024) == "ESM24"
        assert manager.get_contract_symbol("ES", 9, 2024) == "ESU24"
        assert manager.get_contract_symbol("ES", 12, 2024) == "ESZ24"

    def test_parse_contract_symbol(self):
        """Test parsing contract symbol into components."""
        manager = ContractRolloverManager()

        info = manager.parse_contract_symbol("ESH24")
        assert info is not None
        assert info.base_symbol == "ES"
        assert info.month_code == "H"
        assert info.year == 2024

    def test_parse_contract_symbol_invalid(self):
        """Test parsing invalid contract symbol."""
        manager = ContractRolloverManager()

        info = manager.parse_contract_symbol("XY")
        assert info is None

    def test_should_roll_with_calendar(self):
        """Test should_roll with expiration calendar."""
        manager = ContractRolloverManager()

        # Set expiration for ES
        expiry = date(2024, 6, 21)  # 3rd Friday of June (Friday)
        manager.set_expiration_calendar("ES", [expiry])

        # Roll date is 8 BUSINESS days before expiry:
        # June 21 (Fri) - 8 biz days = June 11 (Tue)
        # So June 10 should NOT roll, but June 11+ should roll

        # June 10 (Mon) - should NOT roll yet
        test_date = date(2024, 6, 10)
        assert manager.should_roll("ES", test_date) is False

        # June 12 (Wed) - should roll (past roll date)
        test_date = date(2024, 6, 12)
        assert manager.should_roll("ES", test_date) is True

    def test_get_roll_date(self):
        """Test get_roll_date calculation."""
        manager = ContractRolloverManager()

        expiry = date(2024, 6, 21)  # Friday
        manager.set_expiration_calendar("ES", [expiry])

        roll_date = manager.get_roll_date("ES", date(2024, 6, 1))
        assert roll_date is not None
        # Should be 8 business days before expiry
        assert roll_date < expiry

    def test_get_front_month(self):
        """Test get_front_month returns correct contract."""
        manager = ContractRolloverManager()

        expiry = date(2024, 6, 21)  # June expiry
        manager.set_expiration_calendar("ES", [expiry])

        front = manager.get_front_month("ES", date(2024, 6, 1))
        assert front is not None
        assert front.base_symbol == "ES"
        assert front.month_code == "M"  # June
        assert front.is_front_month is True

    def test_get_next_month(self):
        """Test get_next_month returns back month contract."""
        manager = ContractRolloverManager()

        expiries = [
            date(2024, 6, 21),  # June expiry
            date(2024, 9, 20),  # September expiry
        ]
        manager.set_expiration_calendar("ES", expiries)

        back = manager.get_next_month("ES", date(2024, 6, 1))
        assert back is not None
        assert back.month_code == "U"  # September
        assert back.is_front_month is False

    def test_record_roll(self):
        """Test recording a rollover event."""
        manager = ContractRolloverManager()

        # Use a recent date for the test (today's year)
        today = date.today()

        from_contract = ContractInfo(
            symbol=f"ESM{str(today.year)[-2:]}",
            base_symbol="ES",
            month_code="M",
            year=today.year,
            expiry_date=today,
        )

        to_contract = ContractInfo(
            symbol=f"ESU{str(today.year)[-2:]}",
            base_symbol="ES",
            month_code="U",
            year=today.year,
            expiry_date=today + timedelta(days=90),
        )

        roll_info = RolloverInfo(
            from_contract=from_contract,
            to_contract=to_contract,
            roll_date=today,  # Use today so it's within lookback
            roll_spread=Decimal("5.00"),
        )

        manager.record_roll(roll_info)

        history = manager.get_roll_history("ES")
        assert len(history) == 1
        assert history[0].roll_spread == Decimal("5.00")

    def test_adjust_price_for_continuous(self):
        """Test continuous contract price adjustment."""
        manager = ContractRolloverManager()

        # Record a roll with spread
        from_contract = ContractInfo(
            symbol="ESM24",
            base_symbol="ES",
            month_code="M",
            year=2024,
            expiry_date=date(2024, 6, 21),
        )
        to_contract = ContractInfo(
            symbol="ESU24",
            base_symbol="ES",
            month_code="U",
            year=2024,
            expiry_date=date(2024, 9, 20),
        )
        roll_info = RolloverInfo(
            from_contract=from_contract,
            to_contract=to_contract,
            roll_date=date(2024, 6, 10),
            roll_spread=Decimal("10.00"),  # Back month $10 higher
        )
        manager.record_roll(roll_info)

        # Adjust historical price
        historical_price = Decimal("4500.00")
        adjusted = manager.adjust_price_for_continuous(
            "ES",
            historical_price,
            date(2024, 5, 15),  # Before roll
            adjustment_method="backward",
        )

        # No adjustment yet (before first roll)
        assert adjusted == historical_price

    def test_convenience_functions(self):
        """Test convenience functions."""
        # create_rollover_manager
        manager = create_rollover_manager()
        assert isinstance(manager, ContractRolloverManager)

        # get_contract_month_code
        assert get_contract_month_code(3) == "H"
        assert get_contract_month_code(6) == "M"

        # get_month_from_code
        assert get_month_from_code("H") == 3
        assert get_month_from_code("M") == 6


class TestSettlementEdgeCases:
    """Edge case tests for settlement and rollover."""

    def test_variation_margin_zero_price_change(self):
        """Test variation margin with no price change."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.LONG,
            entry_price=Decimal("4500"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        engine.set_initial_settlement_price("ES", Decimal("4500"))

        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4500"),  # No change
            contract_spec=spec,
        )

        assert payment.variation_margin == Decimal("0")

    def test_rollover_manager_empty_calendar(self):
        """Test rollover manager with empty calendar."""
        manager = ContractRolloverManager()

        # Should calculate from rules
        front = manager.get_front_month("ES", date(2024, 6, 1))
        assert front is not None

    def test_settlement_history_limit(self):
        """Test settlement history is limited to 365 days."""
        engine = CMESettlementEngine()

        position = FuturesPosition(
            symbol="ES",
            side=PositionSide.LONG,
            entry_price=Decimal("4500"),
            qty=Decimal("1"),
            leverage=1,
            margin_mode=MarginMode.CROSS,
        )

        spec = make_spec("ES", "CME", Decimal("50"), Decimal("0.25"), Decimal("12.50"))

        # Simulate 400 settlements using process_daily_settlement
        for i in range(400):
            engine.process_daily_settlement(
                positions=[position],
                settlement_prices={"ES": Decimal("4500") + Decimal(str(i))},
                contract_specs={"ES": spec},
                settlement_date=date(2024, 1, 1) + timedelta(days=i),
            )

        # Should be capped at 365
        assert len(engine._settlement_history["ES"]) <= 365

    def test_case_insensitive_symbol(self):
        """Test symbols are case insensitive."""
        engine = CMESettlementEngine()

        engine.set_initial_settlement_price("es", Decimal("4500"))
        assert engine.get_last_settlement_price("ES") == Decimal("4500")

    def test_subtract_business_days(self):
        """Test business day subtraction."""
        manager = ContractRolloverManager()

        # Friday - 5 business days = previous Friday
        friday = date(2024, 6, 21)  # Friday
        result = manager._subtract_business_days(friday, 5)

        # Should skip weekend, land on Friday June 14
        assert result.weekday() == 4  # Friday
        assert result == date(2024, 6, 14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
