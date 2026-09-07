# -*- coding: utf-8 -*-
"""
test_execution_providers_futures_base.py
Tests for futures execution provider base classes and protocols.

Covers:
- FuturesMarketState dataclass
- ExecutionCostEstimate dataclass
- Protocol interface contracts
- L2FuturesExecutionProvider execution logic
- L3FuturesExecutionProvider fallback behavior
- Factory function
"""

import pytest
from decimal import Decimal
from typing import Optional, Dict, Any

from execution_providers_futures_base import (
    # Data Models
    FuturesMarketState,
    ExecutionCostEstimate,
    # Protocol Interfaces
    FuturesMarginProvider,
    FuturesSlippageProvider,
    FuturesFeeProvider,
    FuturesFundingProvider,
    # Abstract Base Classes
    BaseFuturesExecutionProvider,
    L2FuturesExecutionProvider,
    L3FuturesExecutionProvider,
    # Factory Function
    create_futures_execution_provider,
)

from core_futures import (
    FuturesType,
    FuturesContractSpec,
    FuturesPosition,
    FuturesOrder,
    FuturesFill,
    MarginRequirement,
    MarginMode,
    PositionSide,
    OrderSide,
    OrderType,
    ContractType,
    SettlementType,
    Exchange,
    LeverageBracket,
)


# ═══════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def btc_contract() -> FuturesContractSpec:
    """Create BTC perpetual contract spec."""
    return FuturesContractSpec(
        symbol="BTCUSDT",
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        contract_type=ContractType.PERPETUAL,
        exchange=Exchange.BINANCE,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_size=Decimal("1"),
        multiplier=Decimal("1"),
        tick_size=Decimal("0.1"),
        min_qty=Decimal("0.001"),
        max_leverage=125,
    )


@pytest.fixture
def es_contract() -> FuturesContractSpec:
    """Create ES (E-mini S&P 500) contract spec."""
    return FuturesContractSpec(
        symbol="ES",
        futures_type=FuturesType.INDEX_FUTURES,
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=Exchange.CME,
        base_asset="SPX",
        quote_asset="USD",
        margin_asset="USD",
        contract_size=Decimal("1"),
        multiplier=Decimal("50"),  # $50 per point
        tick_size=Decimal("0.25"),
        min_qty=Decimal("1"),
        max_leverage=20,
    )


@pytest.fixture
def market_state() -> FuturesMarketState:
    """Create sample market state."""
    return FuturesMarketState(
        timestamp_ms=1700000000000,
        bid=Decimal("40000.0"),
        ask=Decimal("40005.0"),
        bid_size=Decimal("10.0"),
        ask_size=Decimal("8.0"),
        mark_price=Decimal("40002.5"),
        index_price=Decimal("40001.0"),
        last_price=Decimal("40003.0"),
        volume_24h=Decimal("50000.0"),
        open_interest=Decimal("100000.0"),
        funding_rate=Decimal("0.0001"),
        next_funding_time_ms=1700003600000,
    )


@pytest.fixture
def market_state_no_funding() -> FuturesMarketState:
    """Create market state without funding (CME-style)."""
    return FuturesMarketState(
        timestamp_ms=1700000000000,
        bid=Decimal("4500.00"),
        ask=Decimal("4500.25"),
        bid_size=Decimal("100"),
        ask_size=Decimal("80"),
        mark_price=Decimal("4500.125"),
        index_price=Decimal("4500.0"),
        last_price=Decimal("4500.10"),
        volume_24h=Decimal("500000"),
        open_interest=Decimal("2000000"),
        settlement_price=Decimal("4499.50"),
        days_to_expiry=45,
    )


@pytest.fixture
def long_position() -> FuturesPosition:
    """Create sample long position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=Decimal("39000.0"),
        qty=Decimal("1.0"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        timestamp_ms=1699900000000,
    )


@pytest.fixture
def short_position() -> FuturesPosition:
    """Create sample short position."""
    return FuturesPosition(
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        entry_price=Decimal("41000.0"),
        qty=Decimal("-1.0"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        timestamp_ms=1699900000000,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MOCK PROVIDERS FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════


class MockMarginProvider:
    """Mock margin provider for testing."""

    def __init__(
        self,
        im_rate: Decimal = Decimal("0.01"),
        mm_rate: Decimal = Decimal("0.005"),
    ):
        self.im_rate = im_rate
        self.mm_rate = mm_rate

    def calculate_initial_margin(
        self,
        contract: FuturesContractSpec,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        return notional / Decimal(str(leverage))

    def calculate_maintenance_margin(
        self,
        contract: FuturesContractSpec,
        notional: Decimal,
    ) -> Decimal:
        return notional * self.mm_rate

    def calculate_liquidation_price(
        self,
        position: FuturesPosition,
        wallet_balance: Decimal,
    ) -> Decimal:
        if position.qty == Decimal("0"):
            return Decimal("0")
        notional = abs(position.qty) * position.entry_price
        im = notional / Decimal(str(position.leverage))
        mm = notional * self.mm_rate
        if position.qty > 0:  # Long
            return position.entry_price * (Decimal("1") - im / notional + mm / notional)
        else:  # Short
            return position.entry_price * (Decimal("1") + im / notional - mm / notional)

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        if position.qty == Decimal("0"):
            return Decimal("999")
        notional = abs(position.qty) * mark_price
        mm = notional * self.mm_rate
        pnl = (mark_price - position.entry_price) * position.qty
        return (wallet_balance + pnl) / mm

    def get_max_leverage(self, notional: Decimal) -> int:
        if notional < Decimal("50000"):
            return 125
        elif notional < Decimal("250000"):
            return 100
        elif notional < Decimal("1000000"):
            return 50
        else:
            return 20


class MockSlippageProvider:
    """Mock slippage provider for testing."""

    def __init__(self, base_slippage_bps: Decimal = Decimal("2.0")):
        self.base_slippage_bps = base_slippage_bps

    def estimate_slippage_bps(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        participation_rate: Optional[Decimal] = None,
    ) -> Decimal:
        return self.base_slippage_bps + market.spread_bps / Decimal("2")


class MockFeeProvider:
    """Mock fee provider for testing."""

    def __init__(
        self,
        maker_rate: Decimal = Decimal("0.0002"),
        taker_rate: Decimal = Decimal("0.0004"),
    ):
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate

    def calculate_fee(
        self,
        notional: Decimal,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate

    def get_fee_rate(
        self,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        return self.maker_rate if is_maker else self.taker_rate


class MockFundingProvider:
    """Mock funding provider for testing."""

    def __init__(self, funding_rate: Decimal = Decimal("0.0001")):
        self._funding_rate = funding_rate

    def get_current_funding_rate(self, symbol: str) -> Decimal:
        return self._funding_rate

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        if position.qty == Decimal("0"):
            return Decimal("0")
        notional = abs(position.qty) * mark_price
        if position.qty > 0:  # Long pays
            return notional * funding_rate
        else:  # Short receives
            return -notional * funding_rate

    def get_next_funding_time(self, symbol: str) -> int:
        return 1700003600000


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: FuturesMarketState
# ═══════════════════════════════════════════════════════════════════════════


class TestFuturesMarketState:
    """Tests for FuturesMarketState dataclass."""

    def test_creation_with_funding(self, market_state):
        """Test creation with funding rate."""
        assert market_state.bid == Decimal("40000.0")
        assert market_state.ask == Decimal("40005.0")
        assert market_state.funding_rate == Decimal("0.0001")
        assert market_state.is_crypto is True
        assert market_state.has_funding is True

    def test_creation_without_funding(self, market_state_no_funding):
        """Test creation without funding rate (CME-style)."""
        assert market_state_no_funding.funding_rate is None
        assert market_state_no_funding.settlement_price == Decimal("4499.50")
        assert market_state_no_funding.days_to_expiry == 45
        assert market_state_no_funding.is_crypto is False
        assert market_state_no_funding.has_funding is False

    def test_mid_price_calculation(self, market_state):
        """Test mid price calculation."""
        expected_mid = (Decimal("40000.0") + Decimal("40005.0")) / Decimal("2")
        assert market_state.mid_price == expected_mid

    def test_spread_calculation(self, market_state):
        """Test spread calculation."""
        assert market_state.spread == Decimal("5.0")

    def test_spread_bps_calculation(self, market_state):
        """Test spread in basis points."""
        mid = market_state.mid_price
        expected_bps = Decimal("5.0") / mid * Decimal("10000")
        assert abs(market_state.spread_bps - expected_bps) < Decimal("0.0001")

    def test_spread_bps_zero_mid(self):
        """Test spread_bps with zero mid price."""
        state = FuturesMarketState(
            timestamp_ms=0,
            bid=Decimal("0"),
            ask=Decimal("0"),
            bid_size=Decimal("0"),
            ask_size=Decimal("0"),
            mark_price=Decimal("0"),
            index_price=Decimal("0"),
            last_price=Decimal("0"),
            volume_24h=Decimal("0"),
            open_interest=Decimal("0"),
        )
        assert state.spread_bps == Decimal("0")

    def test_to_dict(self, market_state):
        """Test conversion to dictionary."""
        d = market_state.to_dict()
        assert d["bid"] == "40000.0"
        assert d["ask"] == "40005.0"
        assert d["funding_rate"] == "0.0001"
        assert d["timestamp_ms"] == 1700000000000

    def test_from_dict(self, market_state):
        """Test creation from dictionary."""
        d = market_state.to_dict()
        restored = FuturesMarketState.from_dict(d)
        assert restored.bid == market_state.bid
        assert restored.ask == market_state.ask
        assert restored.funding_rate == market_state.funding_rate

    def test_immutability(self, market_state):
        """Test that FuturesMarketState is immutable."""
        with pytest.raises(AttributeError):
            market_state.bid = Decimal("50000")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: ExecutionCostEstimate
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionCostEstimate:
    """Tests for ExecutionCostEstimate dataclass."""

    def test_creation(self):
        """Test creation with all fields."""
        estimate = ExecutionCostEstimate(
            slippage_bps=Decimal("2.5"),
            fee_bps=Decimal("4.0"),
            total_cost_bps=Decimal("6.5"),
            impact_cost=Decimal("100.0"),
            estimated_fill_price=Decimal("40010.0"),
            funding_cost=Decimal("10.0"),
            participation_rate=Decimal("0.01"),
        )
        assert estimate.slippage_bps == Decimal("2.5")
        assert estimate.total_cost_bps == Decimal("6.5")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        estimate = ExecutionCostEstimate(
            slippage_bps=Decimal("2.5"),
            fee_bps=Decimal("4.0"),
            total_cost_bps=Decimal("6.5"),
            impact_cost=Decimal("100.0"),
            estimated_fill_price=Decimal("40010.0"),
        )
        d = estimate.to_dict()
        assert d["slippage_bps"] == "2.5"
        assert d["funding_cost"] is None


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Protocol Compliance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolCompliance:
    """Tests that mock providers comply with protocols."""

    def test_margin_provider_protocol(self, btc_contract, long_position):
        """Test MockMarginProvider implements FuturesMarginProvider."""
        provider = MockMarginProvider()

        # Test all protocol methods
        im = provider.calculate_initial_margin(btc_contract, Decimal("40000"), 10)
        assert im == Decimal("4000")

        mm = provider.calculate_maintenance_margin(btc_contract, Decimal("40000"))
        assert mm == Decimal("200")

        liq_price = provider.calculate_liquidation_price(long_position, Decimal("10000"))
        assert liq_price > Decimal("0")

        mr = provider.calculate_margin_ratio(long_position, Decimal("40000"), Decimal("10000"))
        assert mr > Decimal("0")

        max_lev = provider.get_max_leverage(Decimal("100000"))
        assert max_lev == 100

    def test_slippage_provider_protocol(self, market_state):
        """Test MockSlippageProvider implements FuturesSlippageProvider."""
        provider = MockSlippageProvider()
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        slippage = provider.estimate_slippage_bps(order, market_state)
        assert slippage > Decimal("0")

    def test_fee_provider_protocol(self):
        """Test MockFeeProvider implements FuturesFeeProvider."""
        provider = MockFeeProvider()

        fee = provider.calculate_fee(Decimal("40000"), is_maker=True)
        assert fee == Decimal("8.0")

        rate = provider.get_fee_rate(is_maker=False)
        assert rate == Decimal("0.0004")

    def test_funding_provider_protocol(self, long_position):
        """Test MockFundingProvider implements FuturesFundingProvider."""
        provider = MockFundingProvider()

        rate = provider.get_current_funding_rate("BTCUSDT")
        assert rate == Decimal("0.0001")

        payment = provider.calculate_funding_payment(
            long_position, Decimal("0.0001"), Decimal("40000")
        )
        assert payment > Decimal("0")  # Long pays

        next_time = provider.get_next_funding_time("BTCUSDT")
        assert next_time > 0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: L2FuturesExecutionProvider
# ═══════════════════════════════════════════════════════════════════════════


class TestL2FuturesExecutionProvider:
    """Tests for L2 parametric execution provider."""

    @pytest.fixture
    def l2_provider(self):
        """Create L2 provider with mock dependencies."""
        return L2FuturesExecutionProvider(
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            margin_provider=MockMarginProvider(),
            slippage_provider=MockSlippageProvider(),
            fee_provider=MockFeeProvider(),
            funding_provider=MockFundingProvider(),
        )

    @pytest.fixture
    def l2_provider_cme(self):
        """Create L2 provider for CME futures."""
        return L2FuturesExecutionProvider(
            futures_type=FuturesType.INDEX_FUTURES,
            margin_provider=MockMarginProvider(),
            slippage_provider=MockSlippageProvider(),
            fee_provider=MockFeeProvider(),
        )

    def test_creation_crypto_with_funding(self, l2_provider):
        """Test creation for crypto with funding provider."""
        assert l2_provider.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert l2_provider.requires_funding is True

    def test_creation_crypto_without_funding_raises(self):
        """Test that crypto perpetual without funding raises error."""
        with pytest.raises(ValueError, match="FundingProvider required"):
            L2FuturesExecutionProvider(
                futures_type=FuturesType.CRYPTO_PERPETUAL,
                margin_provider=MockMarginProvider(),
                slippage_provider=MockSlippageProvider(),
                fee_provider=MockFeeProvider(),
            )

    def test_creation_cme_without_funding(self, l2_provider_cme):
        """Test creation for CME without funding provider."""
        assert l2_provider_cme.futures_type == FuturesType.INDEX_FUTURES
        assert l2_provider_cme.requires_funding is False

    def test_execute_market_buy(self, l2_provider, market_state):
        """Test market buy execution."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        fill = l2_provider.execute(order, market_state)

        assert fill.symbol == "BTCUSDT"
        assert fill.side == OrderSide.BUY
        assert fill.filled_qty == Decimal("1.0")
        assert fill.avg_price > market_state.mid_price  # Slippage
        assert fill.commission > Decimal("0")
        assert fill.is_maker is False
        assert fill.liquidity == "TAKER"

    def test_execute_market_sell(self, l2_provider, market_state):
        """Test market sell execution."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        fill = l2_provider.execute(order, market_state)

        assert fill.side == OrderSide.SELL
        assert fill.avg_price < market_state.mid_price  # Slippage

    def test_execute_limit_maker(self, l2_provider, market_state):
        """Test limit order execution as maker."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=Decimal("1.0"),
            price=Decimal("39950.0"),
            post_only=True,
        )

        fill = l2_provider.execute(order, market_state)

        assert fill.is_maker is True
        assert fill.liquidity == "MAKER"
        # Maker fee is lower
        expected_taker_fee = fill.avg_price * Decimal("0.0004")
        assert fill.commission < expected_taker_fee

    def test_execute_closing_long_position(self, l2_provider, market_state, long_position):
        """Test closing a long position."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
            reduce_only=True,
        )

        fill = l2_provider.execute(order, market_state, long_position)

        # Should have realized PnL (profit since entry=39000, now ~40000)
        assert fill.realized_pnl > Decimal("0")

    def test_execute_closing_short_position(self, l2_provider, market_state, short_position):
        """Test closing a short position."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
            reduce_only=True,
        )

        fill = l2_provider.execute(order, market_state, short_position)

        # Should have realized PnL (profit since entry=41000, now ~40000)
        assert fill.realized_pnl > Decimal("0")

    def test_estimate_execution_cost(self, l2_provider, market_state):
        """Test pre-trade cost estimation."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        estimate = l2_provider.estimate_execution_cost(order, market_state)

        assert estimate.slippage_bps > Decimal("0")
        assert estimate.fee_bps > Decimal("0")
        assert estimate.total_cost_bps == estimate.slippage_bps + estimate.fee_bps
        assert estimate.estimated_fill_price > market_state.mid_price
        assert estimate.funding_cost is not None  # Crypto has funding

    def test_estimate_execution_cost_cme(self, l2_provider_cme, market_state_no_funding):
        """Test cost estimation for CME (no funding)."""
        order = FuturesOrder(
            symbol="ES",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
        )

        estimate = l2_provider_cme.estimate_execution_cost(order, market_state_no_funding)

        assert estimate.funding_cost is None  # CME has no funding

    def test_calculate_pnl_long(self, l2_provider, long_position):
        """Test PnL calculation for long position."""
        pnl = l2_provider.calculate_pnl(long_position, Decimal("40000"))
        # Entry: 39000, Current: 40000, Qty: 1 => PnL = +1000
        assert pnl == Decimal("1000")

    def test_calculate_pnl_short(self, l2_provider, short_position):
        """Test PnL calculation for short position."""
        pnl = l2_provider.calculate_pnl(short_position, Decimal("40000"))
        # Entry: 41000, Current: 40000, Qty: -1 => PnL = +1000
        assert pnl == Decimal("1000")

    def test_calculate_pnl_zero_position(self, l2_provider):
        """Test PnL for zero position."""
        zero_pos = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.BOTH,
            entry_price=Decimal("40000"),
            qty=Decimal("0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            timestamp_ms=0,
        )
        pnl = l2_provider.calculate_pnl(zero_pos, Decimal("50000"))
        assert pnl == Decimal("0")

    def test_calculate_funding_payment_long(self, l2_provider, market_state, long_position):
        """Test funding payment for long position."""
        payment = l2_provider.calculate_funding_payment(long_position, market_state)
        # Long pays when funding > 0
        assert payment > Decimal("0")

    def test_calculate_funding_payment_short(self, l2_provider, market_state, short_position):
        """Test funding payment for short position."""
        payment = l2_provider.calculate_funding_payment(short_position, market_state)
        # Short receives when funding > 0
        assert payment < Decimal("0")

    def test_get_margin_requirement(self, l2_provider, btc_contract):
        """Test margin requirement calculation."""
        margin_req = l2_provider.get_margin_requirement(
            contract=btc_contract,
            qty=Decimal("1.0"),
            price=Decimal("40000"),
            leverage=10,
        )

        assert margin_req.initial == Decimal("4000")  # 40000/10
        assert margin_req.maintenance == Decimal("200")  # 40000 * 0.005


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: L3FuturesExecutionProvider
# ═══════════════════════════════════════════════════════════════════════════


class TestL3FuturesExecutionProvider:
    """Tests for L3 LOB execution provider."""

    @pytest.fixture
    def l3_provider(self):
        """Create L3 provider with mock dependencies."""
        return L3FuturesExecutionProvider(
            futures_type=FuturesType.CRYPTO_PERPETUAL,
            margin_provider=MockMarginProvider(),
            slippage_provider=MockSlippageProvider(),
            fee_provider=MockFeeProvider(),
            funding_provider=MockFundingProvider(),
            lob_config={"impact_model": "almgren_chriss"},
        )

    def test_creation(self, l3_provider):
        """Test L3 provider creation."""
        assert l3_provider.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert l3_provider._initialized is False

    def test_fallback_to_l2_when_lob_unavailable(self, l3_provider, market_state):
        """Test fallback to L2 when LOB components not available."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        # Should fall back to L2 execution
        fill = l3_provider.execute(order, market_state)

        assert fill.filled_qty == Decimal("1.0")
        assert fill.avg_price > Decimal("0")

    def test_estimate_fallback_to_l2(self, l3_provider, market_state):
        """Test cost estimation fallback to L2."""
        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        estimate = l3_provider.estimate_execution_cost(order, market_state)

        assert estimate.slippage_bps > Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Factory Function
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryFunction:
    """Tests for create_futures_execution_provider factory."""

    def test_create_l2_crypto_perpetual(self):
        """Test creating L2 provider for crypto perpetual."""
        provider = create_futures_execution_provider(
            FuturesType.CRYPTO_PERPETUAL,
            level="L2",
        )

        assert isinstance(provider, L2FuturesExecutionProvider)
        assert provider.futures_type == FuturesType.CRYPTO_PERPETUAL

    def test_create_l2_index_futures(self):
        """Test creating L2 provider for index futures."""
        provider = create_futures_execution_provider(
            FuturesType.INDEX_FUTURES,
            level="L2",
        )

        assert isinstance(provider, L2FuturesExecutionProvider)
        assert provider.futures_type == FuturesType.INDEX_FUTURES

    def test_create_l3_provider(self):
        """Test creating L3 provider."""
        provider = create_futures_execution_provider(
            FuturesType.CRYPTO_PERPETUAL,
            level="L3",
        )

        assert isinstance(provider, L3FuturesExecutionProvider)

    def test_invalid_level_raises(self):
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid execution level"):
            create_futures_execution_provider(
                FuturesType.CRYPTO_PERPETUAL,
                level="L4",
            )

    def test_create_with_custom_config(self):
        """Test creating provider with custom config."""
        provider = create_futures_execution_provider(
            FuturesType.INDEX_FUTURES,
            level="L2",
            config={
                "base_slippage_bps": "5.0",
                "initial_margin": "6000",
            },
        )

        assert isinstance(provider, L2FuturesExecutionProvider)

    def test_create_commodity_futures(self):
        """Test creating provider for commodity futures."""
        provider = create_futures_execution_provider(
            FuturesType.COMMODITY_FUTURES,
            level="L2",
        )

        assert provider.futures_type == FuturesType.COMMODITY_FUTURES
        assert provider.requires_funding is False

    def test_create_currency_futures(self):
        """Test creating provider for currency futures."""
        provider = create_futures_execution_provider(
            FuturesType.CURRENCY_FUTURES,
            level="L2",
        )

        assert provider.futures_type == FuturesType.CURRENCY_FUTURES


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_trade_cycle(self):
        """Test full trade cycle: open, hold, close."""
        provider = create_futures_execution_provider(
            FuturesType.CRYPTO_PERPETUAL,
            level="L2",
        )

        # Create market state
        market = FuturesMarketState(
            timestamp_ms=1700000000000,
            bid=Decimal("40000"),
            ask=Decimal("40010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            mark_price=Decimal("40005"),
            index_price=Decimal("40000"),
            last_price=Decimal("40005"),
            volume_24h=Decimal("50000"),
            open_interest=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700003600000,
        )

        # Open long position
        open_order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )
        open_fill = provider.execute(open_order, market)
        entry_price = open_fill.avg_price

        # Close position (price moved up)
        close_market = FuturesMarketState(
            timestamp_ms=1700010000000,
            bid=Decimal("41000"),
            ask=Decimal("41010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            mark_price=Decimal("41005"),
            index_price=Decimal("41000"),
            last_price=Decimal("41005"),
            volume_24h=Decimal("50000"),
            open_interest=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700013600000,
        )

        close_order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        # Create position for close
        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=entry_price,
            qty=Decimal("1.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            timestamp_ms=1700000000000,
        )

        close_fill = provider.execute(close_order, close_market, position)

        # Should have profit (price went up)
        assert close_fill.realized_pnl > Decimal("0")

    def test_cost_estimation_accuracy(self):
        """Test that execution matches estimation."""
        provider = create_futures_execution_provider(
            FuturesType.CRYPTO_PERPETUAL,
            level="L2",
        )

        market = FuturesMarketState(
            timestamp_ms=1700000000000,
            bid=Decimal("40000"),
            ask=Decimal("40010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            mark_price=Decimal("40005"),
            index_price=Decimal("40000"),
            last_price=Decimal("40005"),
            volume_24h=Decimal("50000"),
            open_interest=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700003600000,
        )

        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0"),
        )

        # Get estimation
        estimate = provider.estimate_execution_cost(order, market)

        # Execute
        fill = provider.execute(order, market)

        # Prices should be close
        price_diff = abs(fill.avg_price - estimate.estimated_fill_price)
        assert price_diff < Decimal("1")  # Within $1


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_quantity_order(self):
        """Test handling of zero quantity order."""
        provider = create_futures_execution_provider(
            FuturesType.INDEX_FUTURES,
            level="L2",
        )

        market = FuturesMarketState(
            timestamp_ms=1700000000000,
            bid=Decimal("4500"),
            ask=Decimal("4500.25"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            mark_price=Decimal("4500.125"),
            index_price=Decimal("4500"),
            last_price=Decimal("4500.10"),
            volume_24h=Decimal("500000"),
            open_interest=Decimal("2000000"),
        )

        order = FuturesOrder(
            symbol="ES",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0"),
        )

        fill = provider.execute(order, market)
        assert fill.filled_qty == Decimal("0")

    def test_very_large_order(self):
        """Test handling of very large order."""
        provider = create_futures_execution_provider(
            FuturesType.CRYPTO_PERPETUAL,
            level="L2",
        )

        market = FuturesMarketState(
            timestamp_ms=1700000000000,
            bid=Decimal("40000"),
            ask=Decimal("40010"),
            bid_size=Decimal("10"),
            ask_size=Decimal("10"),
            mark_price=Decimal("40005"),
            index_price=Decimal("40000"),
            last_price=Decimal("40005"),
            volume_24h=Decimal("50000"),
            open_interest=Decimal("100000"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700003600000,
        )

        order = FuturesOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1000"),  # Large order
        )

        fill = provider.execute(order, market)
        assert fill.filled_qty == Decimal("1000")
        # Large order should have some commission
        assert fill.commission >= Decimal("0")

    def test_all_futures_types(self):
        """Test that all futures types can be created."""
        futures_types = [
            FuturesType.CRYPTO_PERPETUAL,
            FuturesType.CRYPTO_QUARTERLY,
            FuturesType.INDEX_FUTURES,
            FuturesType.COMMODITY_FUTURES,
            FuturesType.CURRENCY_FUTURES,
            FuturesType.BOND_FUTURES,
        ]

        for ft in futures_types:
            provider = create_futures_execution_provider(ft, level="L2")
            assert provider.futures_type == ft


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
