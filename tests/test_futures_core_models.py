# -*- coding: utf-8 -*-
"""
tests/test_futures_core_models.py
Comprehensive tests for core_futures.py models.

Tests cover:
- All enum types and their values
- All dataclass models and their methods
- Serialization (to_dict/from_dict)
- Edge cases and validation
- Factory functions

50+ tests ensuring full coverage of futures core models.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from typing import Dict, Any
import json

from core_futures import (
    # Enums
    FuturesType,
    ContractType,
    SettlementType,
    MarginMode,
    PositionSide,
    Exchange,
    OrderSide,
    OrderType,
    TimeInForce,
    WorkingType,
    # Dataclasses
    LeverageBracket,
    FuturesContractSpec,
    MarkPriceTick,
    FundingRateInfo,
    FundingPayment,
    OpenInterestInfo,
    LiquidationEvent,
    FuturesPosition,
    FuturesAccountState,
    FuturesOrder,
    FuturesFill,
    ContractRollover,
    MarginRequirement,
    # Factory functions
    create_btc_perpetual_spec,
    create_eth_perpetual_spec,
    create_es_futures_spec,
    create_gc_futures_spec,
    create_6e_futures_spec,
)


# ============================================================================
# ENUM TESTS
# ============================================================================


class TestFuturesTypeEnum:
    """Tests for FuturesType enum."""

    def test_futures_type_values(self):
        """Test all FuturesType enum values exist."""
        assert FuturesType.CRYPTO_PERPETUAL.value == "CRYPTO_PERPETUAL"
        assert FuturesType.CRYPTO_QUARTERLY.value == "CRYPTO_QUARTERLY"
        assert FuturesType.INDEX_FUTURES.value == "INDEX_FUTURES"
        assert FuturesType.COMMODITY_FUTURES.value == "COMMODITY_FUTURES"
        assert FuturesType.CURRENCY_FUTURES.value == "CURRENCY_FUTURES"
        assert FuturesType.BOND_FUTURES.value == "BOND_FUTURES"

    def test_futures_type_is_string_enum(self):
        """Test FuturesType inherits from str."""
        assert isinstance(FuturesType.CRYPTO_PERPETUAL, str)
        assert FuturesType.CRYPTO_PERPETUAL == "CRYPTO_PERPETUAL"

    def test_futures_type_count(self):
        """Test correct number of futures types."""
        assert len(FuturesType) == 6


class TestContractTypeEnum:
    """Tests for ContractType enum."""

    def test_contract_type_values(self):
        """Test all ContractType enum values exist."""
        assert ContractType.PERPETUAL.value == "PERPETUAL"
        assert ContractType.CURRENT_MONTH.value == "CURRENT_MONTH"
        assert ContractType.CURRENT_QUARTER.value == "CURRENT_QUARTER"
        assert ContractType.NEXT_QUARTER.value == "NEXT_QUARTER"
        assert ContractType.BACK_MONTH.value == "BACK_MONTH"
        assert ContractType.CONTINUOUS.value == "CONTINUOUS"

    def test_contract_type_is_string_enum(self):
        """Test ContractType inherits from str."""
        assert isinstance(ContractType.PERPETUAL, str)
        assert ContractType.PERPETUAL == "PERPETUAL"

    def test_contract_type_count(self):
        """Test correct number of contract types."""
        assert len(ContractType) == 6


class TestSettlementTypeEnum:
    """Tests for SettlementType enum."""

    def test_settlement_type_values(self):
        """Test all SettlementType enum values exist."""
        assert SettlementType.CASH.value == "CASH"
        assert SettlementType.PHYSICAL.value == "PHYSICAL"
        assert SettlementType.FUNDING.value == "FUNDING"

    def test_settlement_type_is_string_enum(self):
        """Test SettlementType inherits from str."""
        assert isinstance(SettlementType.CASH, str)
        assert SettlementType.CASH == "CASH"

    def test_settlement_type_count(self):
        """Test correct number of settlement types."""
        assert len(SettlementType) == 3


class TestMarginModeEnum:
    """Tests for MarginMode enum."""

    def test_margin_mode_values(self):
        """Test all MarginMode enum values exist."""
        assert MarginMode.CROSS.value == "CROSS"
        assert MarginMode.ISOLATED.value == "ISOLATED"
        assert MarginMode.SPAN.value == "SPAN"

    def test_margin_mode_is_string_enum(self):
        """Test MarginMode inherits from str."""
        assert isinstance(MarginMode.CROSS, str)
        assert MarginMode.CROSS == "CROSS"

    def test_margin_mode_count(self):
        """Test correct number of margin modes."""
        assert len(MarginMode) == 3


class TestPositionSideEnum:
    """Tests for PositionSide enum."""

    def test_position_side_values(self):
        """Test all PositionSide enum values exist."""
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"
        assert PositionSide.BOTH.value == "BOTH"

    def test_position_side_is_string_enum(self):
        """Test PositionSide inherits from str."""
        assert isinstance(PositionSide.LONG, str)
        assert PositionSide.LONG == "LONG"

    def test_position_side_count(self):
        """Test correct number of position sides."""
        assert len(PositionSide) == 3


class TestOrderSideEnum:
    """Tests for OrderSide enum."""

    def test_order_side_values(self):
        """Test all OrderSide enum values exist."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_side_is_string_enum(self):
        """Test OrderSide inherits from str."""
        assert isinstance(OrderSide.BUY, str)
        assert OrderSide.BUY == "BUY"

    def test_order_side_count(self):
        """Test correct number of order sides."""
        assert len(OrderSide) == 2


class TestOrderTypeEnum:
    """Tests for OrderType enum."""

    def test_order_type_values(self):
        """Test all OrderType enum values exist."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_MARKET.value == "STOP_MARKET"
        assert OrderType.TAKE_PROFIT.value == "TAKE_PROFIT"
        assert OrderType.TAKE_PROFIT_MARKET.value == "TAKE_PROFIT_MARKET"
        assert OrderType.TRAILING_STOP_MARKET.value == "TRAILING_STOP_MARKET"

    def test_order_type_is_string_enum(self):
        """Test OrderType inherits from str."""
        assert isinstance(OrderType.MARKET, str)
        assert OrderType.MARKET == "MARKET"


class TestTimeInForceEnum:
    """Tests for TimeInForce enum."""

    def test_time_in_force_values(self):
        """Test all TimeInForce enum values exist."""
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.DAY.value == "DAY"

    def test_time_in_force_is_string_enum(self):
        """Test TimeInForce inherits from str."""
        assert isinstance(TimeInForce.GTC, str)
        assert TimeInForce.GTC == "GTC"


class TestWorkingTypeEnum:
    """Tests for WorkingType enum."""

    def test_working_type_values(self):
        """Test WorkingType enum values exist."""
        assert WorkingType.MARK_PRICE.value == "MARK_PRICE"
        assert WorkingType.CONTRACT_PRICE.value == "CONTRACT_PRICE"

    def test_working_type_is_string_enum(self):
        """Test WorkingType inherits from str."""
        assert isinstance(WorkingType.MARK_PRICE, str)
        assert WorkingType.MARK_PRICE == "MARK_PRICE"


class TestExchangeEnum:
    """Tests for Exchange enum."""

    def test_exchange_values(self):
        """Test major Exchange enum values exist."""
        assert Exchange.BINANCE.value == "BINANCE"
        assert Exchange.CME.value == "CME"
        assert Exchange.COMEX.value == "COMEX"
        assert Exchange.ICE.value == "ICE"
        assert Exchange.NYMEX.value == "NYMEX"
        assert Exchange.CBOT.value == "CBOT"
        assert Exchange.BYBIT.value == "BYBIT"

    def test_exchange_is_string_enum(self):
        """Test Exchange inherits from str."""
        assert isinstance(Exchange.BINANCE, str)
        assert Exchange.BINANCE == "BINANCE"


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestLeverageBracket:
    """Tests for LeverageBracket dataclass."""

    def test_leverage_bracket_creation(self):
        """Test creating a LeverageBracket."""
        bracket = LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),
            max_leverage=125,
            cum_maintenance=Decimal("0"),
        )
        assert bracket.bracket == 1
        assert bracket.notional_cap == Decimal("50000")
        assert bracket.maint_margin_rate == Decimal("0.004")
        assert bracket.max_leverage == 125
        assert bracket.cum_maintenance == Decimal("0")

    def test_leverage_bracket_frozen(self):
        """Test LeverageBracket is immutable."""
        bracket = LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),
            max_leverage=125,
            cum_maintenance=Decimal("0"),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            bracket.bracket = 2

    def test_leverage_bracket_to_dict(self):
        """Test LeverageBracket serialization."""
        bracket = LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),
            max_leverage=125,
            cum_maintenance=Decimal("0"),
        )
        d = bracket.to_dict()
        assert d["bracket"] == 1
        assert d["notional_cap"] == "50000"
        assert d["maint_margin_rate"] == "0.004"
        assert d["max_leverage"] == 125
        assert d["cum_maintenance"] == "0"

    def test_leverage_bracket_from_dict(self):
        """Test LeverageBracket deserialization."""
        d = {
            "bracket": 2,
            "notional_cap": "250000",
            "maint_margin_rate": "0.005",
            "max_leverage": 100,
            "cum_maintenance": "50",
        }
        bracket = LeverageBracket.from_dict(d)
        assert bracket.bracket == 2
        assert bracket.notional_cap == Decimal("250000")
        assert bracket.maint_margin_rate == Decimal("0.005")
        assert bracket.max_leverage == 100
        assert bracket.cum_maintenance == Decimal("50")

    def test_leverage_bracket_roundtrip(self):
        """Test LeverageBracket roundtrip serialization."""
        original = LeverageBracket(
            bracket=3,
            notional_cap=Decimal("1000000"),
            maint_margin_rate=Decimal("0.01"),
            max_leverage=50,
            cum_maintenance=Decimal("2500"),
        )
        roundtrip = LeverageBracket.from_dict(original.to_dict())
        assert roundtrip == original


class TestFuturesContractSpec:
    """Tests for FuturesContractSpec dataclass."""

    def test_btc_perpetual_spec(self):
        """Test BTC perpetual contract spec."""
        spec = create_btc_perpetual_spec()
        assert spec.symbol == "BTCUSDT"
        assert spec.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert spec.contract_type == ContractType.PERPETUAL
        assert spec.exchange == Exchange.BINANCE
        assert spec.base_asset == "BTC"
        assert spec.quote_asset == "USDT"
        assert spec.margin_asset == "USDT"
        assert spec.max_leverage == 125
        assert spec.settlement_type == SettlementType.FUNDING
        assert spec.trading_hours == "24/7"

    def test_eth_perpetual_spec(self):
        """Test ETH perpetual contract spec."""
        spec = create_eth_perpetual_spec()
        assert spec.symbol == "ETHUSDT"
        assert spec.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert spec.max_leverage == 100

    def test_es_futures_spec(self):
        """Test E-mini S&P 500 futures spec."""
        spec = create_es_futures_spec()
        assert spec.symbol == "ES"
        assert spec.futures_type == FuturesType.INDEX_FUTURES
        assert spec.contract_type == ContractType.CURRENT_QUARTER
        assert spec.exchange == Exchange.CME
        assert spec.multiplier == Decimal("50")
        assert spec.tick_size == Decimal("0.25")
        assert spec.tick_value == Decimal("12.50")
        assert spec.settlement_type == SettlementType.CASH

    def test_gc_futures_spec(self):
        """Test Gold futures spec."""
        spec = create_gc_futures_spec()
        assert spec.symbol == "GC"
        assert spec.futures_type == FuturesType.COMMODITY_FUTURES
        assert spec.contract_type == ContractType.CURRENT_MONTH
        assert spec.exchange == Exchange.COMEX
        assert spec.contract_size == Decimal("100")
        assert spec.settlement_type == SettlementType.PHYSICAL

    def test_6e_futures_spec(self):
        """Test Euro FX futures spec."""
        spec = create_6e_futures_spec()
        assert spec.symbol == "6E"
        assert spec.futures_type == FuturesType.CURRENCY_FUTURES
        assert spec.contract_size == Decimal("125000")
        assert spec.tick_size == Decimal("0.00005")
        assert spec.tick_value == Decimal("6.25")

    def test_contract_spec_frozen(self):
        """Test FuturesContractSpec is immutable."""
        spec = create_btc_perpetual_spec()
        with pytest.raises(Exception):
            spec.symbol = "ETHUSDT"

    def test_contract_spec_to_dict(self):
        """Test FuturesContractSpec serialization."""
        spec = create_btc_perpetual_spec()
        d = spec.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["futures_type"] == "CRYPTO_PERPETUAL"
        assert d["contract_type"] == "PERPETUAL"
        assert d["exchange"] == "BINANCE"
        assert d["max_leverage"] == 125

    def test_contract_spec_from_dict(self):
        """Test FuturesContractSpec deserialization."""
        spec = create_btc_perpetual_spec()
        d = spec.to_dict()
        restored = FuturesContractSpec.from_dict(d)
        assert restored.symbol == spec.symbol
        assert restored.futures_type == spec.futures_type
        assert restored.max_leverage == spec.max_leverage

    def test_contract_spec_is_perpetual(self):
        """Test is_perpetual property."""
        perp_spec = create_btc_perpetual_spec()
        assert perp_spec.is_perpetual is True

        es_spec = create_es_futures_spec()
        assert es_spec.is_perpetual is False

    def test_contract_spec_is_crypto(self):
        """Test is_crypto property."""
        btc_spec = create_btc_perpetual_spec()
        assert btc_spec.is_crypto is True

        es_spec = create_es_futures_spec()
        assert es_spec.is_crypto is False


class TestMarkPriceTick:
    """Tests for MarkPriceTick dataclass."""

    def test_mark_price_tick_creation(self):
        """Test creating a MarkPriceTick."""
        tick = MarkPriceTick(
            symbol="BTCUSDT",
            mark_price=Decimal("50000.50"),
            index_price=Decimal("50005.00"),
            estimated_settle_price=Decimal("50002.00"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700000000000,
            timestamp_ms=1699999000000,
        )
        assert tick.symbol == "BTCUSDT"
        assert tick.mark_price == Decimal("50000.50")
        assert tick.index_price == Decimal("50005.00")
        assert tick.funding_rate == Decimal("0.0001")

    def test_mark_price_tick_to_dict(self):
        """Test MarkPriceTick serialization."""
        tick = MarkPriceTick(
            symbol="ETHUSDT",
            mark_price=Decimal("2500.00"),
            index_price=Decimal("2501.00"),
            estimated_settle_price=Decimal("2500.50"),
            funding_rate=Decimal("0.0002"),
            next_funding_time_ms=1700000000000,
            timestamp_ms=1699999000000,
        )
        d = tick.to_dict()
        assert d["symbol"] == "ETHUSDT"
        assert d["mark_price"] == "2500.00"
        assert d["funding_rate"] == "0.0002"


class TestFundingRateInfo:
    """Tests for FundingRateInfo dataclass."""

    def test_funding_rate_info_creation(self):
        """Test creating a FundingRateInfo."""
        info = FundingRateInfo(
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("50000"),
            estimated_rate=Decimal("0.00015"),
        )
        assert info.symbol == "BTCUSDT"
        assert info.funding_rate == Decimal("0.0001")
        assert info.estimated_rate == Decimal("0.00015")

    def test_funding_rate_info_optional_next(self):
        """Test FundingRateInfo with optional estimated_rate."""
        info = FundingRateInfo(
            symbol="ETHUSDT",
            funding_rate=Decimal("-0.0002"),
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("2500"),
        )
        assert info.estimated_rate is None

    def test_funding_rate_info_to_dict(self):
        """Test FundingRateInfo serialization."""
        info = FundingRateInfo(
            symbol="BTCUSDT",
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("50000"),
        )
        d = info.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["funding_rate"] == "0.0001"


class TestFundingPayment:
    """Tests for FundingPayment dataclass."""

    def test_funding_payment_creation(self):
        """Test creating a FundingPayment."""
        payment = FundingPayment(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            position_qty=Decimal("1.5"),
            payment_amount=Decimal("-5.50"),
        )
        assert payment.symbol == "BTCUSDT"
        assert payment.payment_amount == Decimal("-5.50")
        assert payment.position_qty == Decimal("1.5")

    def test_funding_payment_to_dict(self):
        """Test FundingPayment serialization."""
        payment = FundingPayment(
            symbol="ETHUSDT",
            timestamp_ms=1700000000000,
            funding_rate=Decimal("0.0002"),
            mark_price=Decimal("2500"),
            position_qty=Decimal("-5.0"),
            payment_amount=Decimal("10.00"),
        )
        d = payment.to_dict()
        assert d["payment_amount"] == "10.00"
        assert d["position_qty"] == "-5.0"


class TestOpenInterestInfo:
    """Tests for OpenInterestInfo dataclass."""

    def test_open_interest_creation(self):
        """Test creating an OpenInterestInfo."""
        oi = OpenInterestInfo(
            symbol="BTCUSDT",
            open_interest=Decimal("50000.5"),
            open_interest_value=Decimal("2500000000"),
            timestamp_ms=1700000000000,
        )
        assert oi.symbol == "BTCUSDT"
        assert oi.open_interest == Decimal("50000.5")
        assert oi.open_interest_value == Decimal("2500000000")

    def test_open_interest_to_dict(self):
        """Test OpenInterestInfo serialization."""
        oi = OpenInterestInfo(
            symbol="ETHUSDT",
            open_interest=Decimal("1000000"),
            open_interest_value=Decimal("2500000000"),
            timestamp_ms=1700000000000,
        )
        d = oi.to_dict()
        assert d["open_interest"] == "1000000"


class TestLiquidationEvent:
    """Tests for LiquidationEvent dataclass."""

    def test_liquidation_event_creation(self):
        """Test creating a LiquidationEvent."""
        event = LiquidationEvent(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            side="SELL",  # Liquidation sell (closing long)
            qty=Decimal("0.5"),
            price=Decimal("48000"),
            liquidation_type="full",
            loss_amount=Decimal("1000"),
        )
        assert event.symbol == "BTCUSDT"
        assert event.side == "SELL"
        assert event.qty == Decimal("0.5")
        assert event.price == Decimal("48000")

    def test_liquidation_event_to_dict(self):
        """Test LiquidationEvent serialization."""
        event = LiquidationEvent(
            symbol="ETHUSDT",
            timestamp_ms=1700000000000,
            side="BUY",  # Short liquidation (closing short)
            qty=Decimal("10"),
            price=Decimal("2200"),
            liquidation_type="partial",
            loss_amount=Decimal("500"),
        )
        d = event.to_dict()
        assert d["side"] == "BUY"
        assert d["qty"] == "10"


class TestFuturesPosition:
    """Tests for FuturesPosition dataclass."""

    def test_futures_position_creation(self):
        """Test creating a FuturesPosition."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1.5"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=Decimal("1500"),
            mark_price=Decimal("51000"),
            margin=Decimal("7500"),
            liquidation_price=Decimal("45000"),
        )
        assert pos.symbol == "BTCUSDT"
        assert pos.side == PositionSide.LONG
        assert pos.qty == Decimal("1.5")
        assert pos.entry_price == Decimal("50000")
        assert pos.leverage == 10
        assert pos.margin_mode == MarginMode.CROSS

    def test_futures_position_is_long(self):
        """Test is_long property."""
        long_pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("5000"),
        )
        assert long_pos.is_long is True
        assert long_pos.is_short is False

    def test_futures_position_is_short(self):
        """Test is_short property."""
        short_pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            margin=Decimal("5000"),
        )
        assert short_pos.is_short is True
        assert short_pos.is_long is False

    def test_futures_position_notional(self):
        """Test notional property."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("2"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        # notional = abs_qty * entry_price = 2 * 50000 = 100000
        assert pos.notional == Decimal("100000")

    def test_futures_position_roe_pct(self):
        """Test roe_pct property (Return on Equity)."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("5000"),
            unrealized_pnl=Decimal("1000"),
        )
        # ROE = (unrealized_pnl / margin) * 100 = (1000 / 5000) * 100 = 20%
        assert pos.roe_pct == Decimal("20")

    def test_futures_position_calculate_pnl(self):
        """Test calculate_pnl method."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        # Long position: PnL = qty * (current_price - entry_price)
        pnl = pos.calculate_pnl(Decimal("52000"))
        assert pnl == Decimal("2000")  # (52000 - 50000) * 1

    def test_futures_position_calculate_pnl_short(self):
        """Test calculate_pnl for short position."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        # Short position: PnL = qty * (current_price - entry_price)
        # = -1 * (48000 - 50000) = -1 * -2000 = 2000
        pnl = pos.calculate_pnl(Decimal("48000"))
        assert pnl == Decimal("2000")

    def test_futures_position_to_dict(self):
        """Test FuturesPosition serialization."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        d = pos.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["side"] == "LONG"
        assert d["leverage"] == 10
        assert d["margin_mode"] == "CROSS"


class TestFuturesAccountState:
    """Tests for FuturesAccountState dataclass."""

    def test_account_state_creation(self):
        """Test creating a FuturesAccountState."""
        state = FuturesAccountState(
            timestamp_ms=1700000000000,
            total_wallet_balance=Decimal("10000"),
            total_margin_balance=Decimal("10500"),
            total_unrealized_pnl=Decimal("500"),
            available_balance=Decimal("5000"),
            total_initial_margin=Decimal("4500"),
            total_maint_margin=Decimal("2250"),
        )
        assert state.total_wallet_balance == Decimal("10000")
        assert state.available_balance == Decimal("5000")

    def test_account_state_margin_ratio(self):
        """Test margin_ratio property."""
        state = FuturesAccountState(
            timestamp_ms=1700000000000,
            total_wallet_balance=Decimal("10000"),
            total_margin_balance=Decimal("10000"),
            total_unrealized_pnl=Decimal("0"),
            available_balance=Decimal("5000"),
            total_initial_margin=Decimal("5000"),
            total_maint_margin=Decimal("2500"),
        )
        # margin_ratio = maint_margin / margin_balance = 2500 / 10000 = 0.25
        assert state.margin_ratio == Decimal("0.25")

    def test_account_state_with_positions(self):
        """Test FuturesAccountState with positions."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        state = FuturesAccountState(
            timestamp_ms=1700000000000,
            total_wallet_balance=Decimal("10000"),
            total_margin_balance=Decimal("10500"),
            total_unrealized_pnl=Decimal("500"),
            available_balance=Decimal("5000"),
            total_initial_margin=Decimal("5000"),
            total_maint_margin=Decimal("2500"),
            positions={"BTCUSDT": pos},
        )
        assert "BTCUSDT" in state.positions
        assert state.positions["BTCUSDT"].qty == Decimal("1")

    def test_account_state_to_dict(self):
        """Test FuturesAccountState serialization."""
        state = FuturesAccountState(
            timestamp_ms=1700000000000,
            total_wallet_balance=Decimal("10000"),
            total_margin_balance=Decimal("10000"),
            total_unrealized_pnl=Decimal("0"),
            available_balance=Decimal("5000"),
            total_initial_margin=Decimal("0"),
            total_maint_margin=Decimal("0"),
        )
        d = state.to_dict()
        assert d["total_wallet_balance"] == "10000"
        assert d["available_balance"] == "5000"


class TestFuturesOrder:
    """Tests for FuturesOrder dataclass."""

    def test_futures_order_creation(self):
        """Test creating a FuturesOrder."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("1"),
            price=Decimal("49000"),
        )
        assert order.symbol == "BTCUSDT"
        assert order.order_type == OrderType.LIMIT
        assert order.time_in_force == TimeInForce.GTC  # Default

    def test_futures_order_is_market_order(self):
        """Test is_market_order property."""
        market_order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
        )
        assert market_order.is_market_order is True
        assert market_order.is_limit_order is False

    def test_futures_order_is_limit_order(self):
        """Test is_limit_order property."""
        limit_order = FuturesOrder(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            qty=Decimal("5"),
            price=Decimal("2500"),
        )
        assert limit_order.is_limit_order is True
        assert limit_order.is_market_order is False

    def test_futures_order_is_stop_order(self):
        """Test is_stop_order property."""
        stop_order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            qty=Decimal("1"),
            stop_price=Decimal("48000"),
            reduce_only=True,
        )
        assert stop_order.is_stop_order is True

    def test_futures_order_to_dict(self):
        """Test FuturesOrder serialization."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("1"),
            price=Decimal("50000"),
        )
        d = order.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["order_type"] == "LIMIT"
        assert d["side"] == "BUY"


class TestFuturesFill:
    """Tests for FuturesFill dataclass."""

    def test_futures_fill_creation(self):
        """Test creating a FuturesFill."""
        fill = FuturesFill(
            order_id="order_1",
            client_order_id="client_1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            filled_qty=Decimal("0.5"),
            avg_price=Decimal("50000"),
            commission=Decimal("10"),
            commission_asset="USDT",
            realized_pnl=Decimal("0"),
            timestamp_ms=1700000000000,
            is_maker=False,
            liquidity="TAKER",
        )
        assert fill.order_id == "order_1"
        assert fill.avg_price == Decimal("50000")
        assert fill.filled_qty == Decimal("0.5")
        assert fill.is_maker is False

    def test_futures_fill_notional(self):
        """Test notional property."""
        fill = FuturesFill(
            order_id="order_2",
            client_order_id=None,
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            filled_qty=Decimal("10"),
            avg_price=Decimal("2500"),
            commission=Decimal("5"),
            commission_asset="USDT",
            realized_pnl=Decimal("100"),
            timestamp_ms=1700000000000,
            is_maker=True,
            liquidity="MAKER",
        )
        # notional = filled_qty * avg_price = 10 * 2500 = 25000
        assert fill.notional == Decimal("25000")

    def test_futures_fill_to_dict(self):
        """Test FuturesFill serialization."""
        fill = FuturesFill(
            order_id="o1",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            filled_qty=Decimal("1"),
            avg_price=Decimal("50000"),
            commission=Decimal("20"),
            commission_asset="USDT",
            realized_pnl=Decimal("0"),
            timestamp_ms=1700000000000,
            is_maker=False,
            liquidity="TAKER",
        )
        d = fill.to_dict()
        assert d["order_id"] == "o1"
        assert d["avg_price"] == "50000"
        assert d["is_maker"] is False


class TestMarginRequirement:
    """Tests for MarginRequirement dataclass."""

    def test_margin_requirement_creation(self):
        """Test creating a MarginRequirement."""
        req = MarginRequirement(
            initial=Decimal("5000"),
            maintenance=Decimal("2500"),
            variation=Decimal("100"),
            available=Decimal("7500"),
        )
        assert req.initial == Decimal("5000")
        assert req.maintenance == Decimal("2500")
        assert req.variation == Decimal("100")
        assert req.available == Decimal("7500")

    def test_margin_requirement_margin_ratio(self):
        """Test margin_ratio property."""
        req = MarginRequirement(
            initial=Decimal("5000"),
            maintenance=Decimal("2500"),
        )
        # ratio = maintenance / initial = 2500 / 5000 = 0.5
        assert req.margin_ratio == Decimal("0.5")

    def test_margin_requirement_to_dict(self):
        """Test MarginRequirement serialization."""
        req = MarginRequirement(
            initial=Decimal("5000"),
            maintenance=Decimal("2500"),
        )
        d = req.to_dict()
        assert d["initial"] == "5000"
        assert d["maintenance"] == "2500"


class TestContractRollover:
    """Tests for ContractRollover dataclass."""

    def test_contract_rollover_creation(self):
        """Test creating a ContractRollover."""
        rollover = ContractRollover(
            from_contract="ESH24",
            to_contract="ESM24",
            roll_date="2024-03-15",
            price_adjustment=Decimal("5.25"),
            volume_adjustment=Decimal("1.0"),
        )
        assert rollover.from_contract == "ESH24"
        assert rollover.to_contract == "ESM24"
        assert rollover.roll_date == "2024-03-15"
        assert rollover.price_adjustment == Decimal("5.25")

    def test_contract_rollover_defaults(self):
        """Test ContractRollover default values."""
        rollover = ContractRollover(
            from_contract="GCJ24",
            to_contract="GCM24",
            roll_date="2024-04-01",
        )
        assert rollover.price_adjustment == Decimal("0")
        assert rollover.volume_adjustment == Decimal("1")

    def test_contract_rollover_to_dict(self):
        """Test ContractRollover serialization."""
        rollover = ContractRollover(
            from_contract="6EH24",
            to_contract="6EM24",
            roll_date="2024-03-18",
            price_adjustment=Decimal("0.0015"),
        )
        d = rollover.to_dict()
        assert d["from_contract"] == "6EH24"
        assert d["to_contract"] == "6EM24"
        assert d["price_adjustment"] == "0.0015"

    def test_contract_rollover_from_dict(self):
        """Test ContractRollover deserialization."""
        d = {
            "from_contract": "CLG24",
            "to_contract": "CLH24",
            "roll_date": "2024-01-15",
            "price_adjustment": "0.50",
            "volume_adjustment": "1.1",
        }
        rollover = ContractRollover.from_dict(d)
        assert rollover.from_contract == "CLG24"
        assert rollover.price_adjustment == Decimal("0.50")
        assert rollover.volume_adjustment == Decimal("1.1")


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_btc_perpetual_spec(self):
        """Test create_btc_perpetual_spec factory."""
        spec = create_btc_perpetual_spec()
        assert spec.symbol == "BTCUSDT"
        assert spec.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert spec.max_leverage == 125
        assert spec.settlement_type == SettlementType.FUNDING

    def test_create_eth_perpetual_spec(self):
        """Test create_eth_perpetual_spec factory."""
        spec = create_eth_perpetual_spec()
        assert spec.symbol == "ETHUSDT"
        assert spec.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert spec.max_leverage == 100

    def test_create_es_futures_spec(self):
        """Test create_es_futures_spec factory."""
        spec = create_es_futures_spec()
        assert spec.symbol == "ES"
        assert spec.futures_type == FuturesType.INDEX_FUTURES
        assert spec.exchange == Exchange.CME
        assert spec.multiplier == Decimal("50")

    def test_create_gc_futures_spec(self):
        """Test create_gc_futures_spec factory."""
        spec = create_gc_futures_spec()
        assert spec.symbol == "GC"
        assert spec.futures_type == FuturesType.COMMODITY_FUTURES
        assert spec.exchange == Exchange.COMEX
        assert spec.settlement_type == SettlementType.PHYSICAL

    def test_create_6e_futures_spec(self):
        """Test create_6e_futures_spec factory."""
        spec = create_6e_futures_spec()
        assert spec.symbol == "6E"
        assert spec.futures_type == FuturesType.CURRENCY_FUTURES
        assert spec.contract_size == Decimal("125000")


# ============================================================================
# EDGE CASES AND VALIDATION
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and validation."""

    def test_leverage_bracket_zero_cum_maintenance(self):
        """Test LeverageBracket with zero cum_maintenance."""
        bracket = LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),
            max_leverage=125,
            cum_maintenance=Decimal("0"),
        )
        assert bracket.cum_maintenance == Decimal("0")

    def test_position_zero_margin_roe(self):
        """Test ROE when margin is zero (edge case)."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("0"),  # Zero margin
        )
        # Should handle division by zero gracefully
        roe = pos.roe_pct
        assert roe == Decimal("0")

    def test_funding_rate_negative(self):
        """Test negative funding rate (shorts pay longs)."""
        info = FundingRateInfo(
            symbol="BTCUSDT",
            funding_rate=Decimal("-0.0005"),  # -0.05%
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("50000"),
        )
        assert info.funding_rate < 0

    def test_liquidation_price_calculation_long(self):
        """Test liquidation price for long position."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1"),
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            margin=Decimal("5000"),
            liquidation_price=Decimal("45500"),  # ~9% below entry
        )
        # Liquidation price should be below entry for long
        assert pos.liquidation_price < pos.entry_price

    def test_liquidation_price_calculation_short(self):
        """Test liquidation price for short position."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            qty=Decimal("-1"),
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            margin=Decimal("5000"),
            liquidation_price=Decimal("54500"),  # ~9% above entry
        )
        # Liquidation price should be above entry for short
        assert pos.liquidation_price > pos.entry_price

    def test_contract_spec_json_serializable(self):
        """Test that contract spec can be JSON serialized."""
        spec = create_btc_perpetual_spec()
        d = spec.to_dict()
        json_str = json.dumps(d)
        assert "BTCUSDT" in json_str
        # And deserialize
        restored = json.loads(json_str)
        assert restored["symbol"] == "BTCUSDT"

    def test_very_small_funding_rate(self):
        """Test very small funding rates (e.g., 0.0001%)."""
        info = FundingRateInfo(
            symbol="BTCUSDT",
            funding_rate=Decimal("0.000001"),  # 0.0001%
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("50000"),
        )
        assert info.funding_rate == Decimal("0.000001")

    def test_very_large_position(self):
        """Test very large position values."""
        pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1000"),  # 1000 BTC
            leverage=10,
            margin_mode=MarginMode.CROSS,
            margin=Decimal("5000000"),
            unrealized_pnl=Decimal("1000000"),  # $1M profit
        )
        # notional = 1000 * 50000 = 50M
        assert pos.notional == Decimal("50000000")


class TestDecimalPrecision:
    """Tests for Decimal precision handling."""

    def test_tick_size_precision(self):
        """Test tick size precision in contract specs."""
        es_spec = create_es_futures_spec()
        assert es_spec.tick_size == Decimal("0.25")

        gc_spec = create_gc_futures_spec()
        assert gc_spec.tick_size == Decimal("0.10")

        fx_spec = create_6e_futures_spec()
        assert fx_spec.tick_size == Decimal("0.00005")

    def test_margin_rate_precision(self):
        """Test margin rate precision."""
        bracket = LeverageBracket(
            bracket=1,
            notional_cap=Decimal("50000"),
            maint_margin_rate=Decimal("0.004"),  # 0.4%
            max_leverage=125,
            cum_maintenance=Decimal("0"),
        )
        assert bracket.maint_margin_rate == Decimal("0.004")

    def test_funding_rate_precision(self):
        """Test funding rate precision (8 decimal places)."""
        info = FundingRateInfo(
            symbol="BTCUSDT",
            funding_rate=Decimal("0.00010000"),
            next_funding_time_ms=1700000000000,
            mark_price=Decimal("50000"),
        )
        assert info.funding_rate == Decimal("0.00010000")
