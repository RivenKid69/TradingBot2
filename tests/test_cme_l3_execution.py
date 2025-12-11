# -*- coding: utf-8 -*-
"""
tests/test_cme_l3_execution.py
Comprehensive test suite for Phase 5B: L3 CME Execution Provider.

This test suite covers:
1. GlobexMatchingEngine - CME Globex matching mechanics
2. CMEL3ExecutionProvider - Full L3 execution simulation
3. CMEL3SlippageProvider - L3 slippage with LOB walk-through
4. CMEL3FillProvider - Globex-style order filling
5. DailySettlementSimulator - Mark-to-market settlement
6. Session detection - RTH/ETH/maintenance

Target: 55+ tests with 100% coverage of critical paths.

References:
- CME Globex Matching Algorithm
- CME Rule 80B (Circuit Breakers)
- CME Velocity Logic
"""

import pytest
pytest.importorskip("sortedcontainers")
import math
import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

# Module under test - GlobexMatchingEngine
from lob.cme_matching import (
    GlobexMatchingEngine,
    GlobexOrderType,
    StopOrder,
    StopTriggerType,
    AuctionOrder,
    AuctionResult,
    AuctionState,
    VelocityLogicResult,
    create_globex_matching_engine,
    DEFAULT_PROTECTION_POINTS,
)

from lob.data_structures import (
    Side,
    LimitOrder,
    OrderType as LOBOrderType,
)

from lob.matching_engine import (
    MatchResult,
    MatchType,
)

# Module under test - L3 Execution Provider
from execution_providers_cme_l3 import (
    CMESession,
    SettlementEvent,
    DailySettlementState,
    get_cme_session,
    is_rth_session,
    get_minutes_to_settlement,
    DailySettlementSimulator,
    CMEL3SlippageProvider,
    CMEL3FillProvider,
    CMEL3ExecutionProvider,
    create_cme_l3_execution_provider,
    create_cme_l3_slippage_provider,
    create_cme_l3_fill_provider,
    SETTLEMENT_TIMES,
    RTH_START_HOUR,
    RTH_END_HOUR,
)

# Dependencies
from execution_providers import (
    Order,
    MarketState,
    BarData,
    Fill,
    AssetClass,
)

from execution_providers_cme import (
    CMESlippageConfig,
    CMEFeeConfig,
    CMEFeeProvider,
    TICK_SIZES,
)

from impl_circuit_breaker import (
    CMECircuitBreaker,
    CircuitBreakerLevel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def es_matching_engine():
    """Create E-mini S&P 500 matching engine."""
    return create_globex_matching_engine(
        symbol="ES",
        tick_size=0.25,
        protection_points=6,
    )


@pytest.fixture
def nq_matching_engine():
    """Create E-mini NASDAQ matching engine."""
    return create_globex_matching_engine(
        symbol="NQ",
        tick_size=0.25,
        protection_points=10,
    )


@pytest.fixture
def gc_matching_engine():
    """Create Gold futures matching engine."""
    return create_globex_matching_engine(
        symbol="GC",
        tick_size=0.10,
        protection_points=30,
    )


@pytest.fixture
def sample_market_state():
    """Create sample market state for ES."""
    return MarketState(
        timestamp=int(time.time() * 1000),
        bid=4500.00,
        ask=4500.25,
        adv=2_000_000_000,  # $2B ADV
        volatility=0.015,
        bid_depth=[(4500.00, 500), (4499.75, 300), (4499.50, 200)],
        ask_depth=[(4500.25, 500), (4500.50, 300), (4500.75, 200)],
    )


@pytest.fixture
def sample_bar_data():
    """Create sample bar data for ES."""
    return BarData(
        open=4500.00,
        high=4505.00,
        low=4495.00,
        close=4502.00,
        volume=50000.0,
    )


@pytest.fixture
def settlement_simulator():
    """Create settlement simulator for ES."""
    return DailySettlementSimulator(symbol="ES", contract_multiplier=Decimal("50"))


# =============================================================================
# Test: GlobexMatchingEngine Basic Operations
# =============================================================================

class TestGlobexMatchingEngineBasic:
    """Test basic GlobexMatchingEngine functionality."""

    def test_create_engine_es(self, es_matching_engine):
        """Test engine creation for E-mini S&P."""
        assert es_matching_engine is not None
        assert es_matching_engine.symbol == "ES"
        assert es_matching_engine.tick_size == 0.25
        assert es_matching_engine.protection_points == 6

    def test_create_engine_gc(self, gc_matching_engine):
        """Test engine creation for Gold."""
        assert gc_matching_engine is not None
        assert gc_matching_engine.symbol == "GC"
        assert gc_matching_engine.tick_size == 0.10
        assert gc_matching_engine.protection_points == 30

    def test_add_resting_order(self, es_matching_engine):
        """Test adding a resting limit order."""
        resting = LimitOrder(
            order_id="rest_1",
            price=4500.00,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.add_order(resting)
        assert result.resting_order is not None
        assert result.total_filled_qty == 0.0

    def test_match_aggressive_order(self, es_matching_engine):
        """Test matching an aggressive order."""
        # Add resting ask
        resting = LimitOrder(
            order_id="rest_1",
            price=4500.25,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
            order_type=LOBOrderType.LIMIT,
        )
        es_matching_engine.add_order(resting)

        # Send aggressive buy
        aggressive = LimitOrder(
            order_id="aggr_1",
            price=4500.50,  # Crosses the spread
            qty=5.0,
            remaining_qty=5.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.add_order(aggressive)
        assert result.total_filled_qty == 5.0
        assert len(result.fills) == 1

    def test_fifo_priority(self, es_matching_engine):
        """Test FIFO price-time priority."""
        # Add two resting orders at same price
        order1 = LimitOrder(
            order_id="order_1",
            price=4500.00,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=1000,
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )
        order2 = LimitOrder(
            order_id="order_2",
            price=4500.00,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=2000,  # Later timestamp
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        es_matching_engine.add_order(order1)
        es_matching_engine.add_order(order2)

        # Aggressive sell should match order1 first (FIFO)
        aggressive = LimitOrder(
            order_id="aggr_1",
            price=4499.75,
            qty=5.0,
            remaining_qty=5.0,
            timestamp_ns=3000,
            side=Side.SELL,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.add_order(aggressive)
        assert result.total_filled_qty == 5.0
        assert len(result.fills) == 1
        # First fill should be against order1

    def test_get_best_bid_ask(self, es_matching_engine):
        """Test getting best bid/ask."""
        # Initially empty
        assert es_matching_engine.get_best_bid() is None
        assert es_matching_engine.get_best_ask() is None

        # Add bid
        bid = LimitOrder(
            order_id="bid_1",
            price=4500.00,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )
        es_matching_engine.add_order(bid)

        # Add ask
        ask = LimitOrder(
            order_id="ask_1",
            price=4500.25,
            qty=10.0,
            remaining_qty=10.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
            order_type=LOBOrderType.LIMIT,
        )
        es_matching_engine.add_order(ask)

        assert es_matching_engine.get_best_bid() == 4500.00
        assert es_matching_engine.get_best_ask() == 4500.25

    def test_get_spread(self, es_matching_engine):
        """Test spread calculation."""
        # Add bid and ask
        es_matching_engine.add_order(LimitOrder(
            order_id="bid_1", price=4500.00, qty=10.0, remaining_qty=10.0,
            timestamp_ns=time.time_ns(), side=Side.BUY, order_type=LOBOrderType.LIMIT,
        ))
        es_matching_engine.add_order(LimitOrder(
            order_id="ask_1", price=4500.25, qty=10.0, remaining_qty=10.0,
            timestamp_ns=time.time_ns(), side=Side.SELL, order_type=LOBOrderType.LIMIT,
        ))

        spread = es_matching_engine.get_spread()
        assert spread == pytest.approx(0.25, abs=0.001)

    def test_clear_book(self, es_matching_engine):
        """Test clearing the order book."""
        # Add some orders
        es_matching_engine.add_order(LimitOrder(
            order_id="bid_1", price=4500.00, qty=10.0, remaining_qty=10.0,
            timestamp_ns=time.time_ns(), side=Side.BUY, order_type=LOBOrderType.LIMIT,
        ))

        assert es_matching_engine.get_best_bid() is not None

        # Clear
        es_matching_engine.clear()
        assert es_matching_engine.get_best_bid() is None


# =============================================================================
# Test: Market with Protection (MWP)
# =============================================================================

class TestGlobexMatchingEngineMWP:
    """Test Market with Protection order handling."""

    def test_mwp_within_protection(self, es_matching_engine):
        """Test MWP execution within protection limit."""
        # Add some liquidity
        for i in range(5):
            es_matching_engine.add_order(LimitOrder(
                order_id=f"ask_{i}",
                price=4500.25 + i * 0.25,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=time.time_ns(),
                side=Side.SELL,
                order_type=LOBOrderType.LIMIT,
            ))

        # MWP buy order
        mwp_order = LimitOrder(
            order_id="mwp_1",
            price=4600.00,  # High price
            qty=50.0,
            remaining_qty=50.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.match_with_protection(mwp_order, protection_points=6)
        # Protection limit = 4500.25 + 6 * 0.25 = 4501.75
        assert result.total_filled_qty > 0

    def test_mwp_no_liquidity(self, es_matching_engine):
        """Test MWP with no liquidity - order rejected."""
        mwp_order = LimitOrder(
            order_id="mwp_1",
            price=4600.00,
            qty=50.0,
            remaining_qty=50.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.match_with_protection(mwp_order)
        assert result.total_filled_qty == 0

    def test_protection_limit_calculation(self, es_matching_engine):
        """Test protection limit price calculation."""
        # Add ask at 4500.25
        es_matching_engine.add_order(LimitOrder(
            order_id="ask_1",
            price=4500.25,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
            order_type=LOBOrderType.LIMIT,
        ))

        # Get protection limit
        limit = es_matching_engine.get_protection_limit(Side.BUY, protection_points=6)
        # 4500.25 + 6 * 0.25 = 4501.75
        assert limit == pytest.approx(4501.75, abs=0.01)


# =============================================================================
# Test: Stop Orders
# =============================================================================

class TestGlobexMatchingEngineStops:
    """Test stop order handling."""

    def test_submit_stop_order(self, es_matching_engine):
        """Test submitting a stop order."""
        stop = StopOrder(
            order_id="stop_1",
            symbol="ES",
            side=Side.BUY,
            qty=10.0,
            stop_price=4510.00,
            trigger_type=StopTriggerType.LAST_TRADE,
        )

        result = es_matching_engine.submit_stop_order(stop)
        assert result is True  # Returns True if accepted

    def test_stop_order_triggers(self, es_matching_engine):
        """Test stop order triggering."""
        # Submit stop
        stop = StopOrder(
            order_id="stop_1",
            symbol="ES",
            side=Side.BUY,
            qty=10.0,
            stop_price=4510.00,
            trigger_type=StopTriggerType.LAST_TRADE,
        )
        es_matching_engine.submit_stop_order(stop)

        # Add some liquidity first
        es_matching_engine.add_order(LimitOrder(
            order_id="ask_1",
            price=4510.25,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
            order_type=LOBOrderType.LIMIT,
        ))

        # Trigger via trade at 4510.00
        triggered_orders = es_matching_engine.check_stops(last_trade_price=4510.00)
        assert len(triggered_orders) == 1

    def test_stop_limit_order(self, es_matching_engine):
        """Test stop-limit order."""
        stop = StopOrder(
            order_id="stop_lmt_1",
            symbol="ES",
            side=Side.BUY,
            qty=10.0,
            stop_price=4510.00,
            limit_price=4511.00,  # Stop-limit
            trigger_type=StopTriggerType.LAST_TRADE,
        )

        assert stop.is_stop_limit is True
        es_matching_engine.submit_stop_order(stop)

    def test_stop_should_trigger_buy(self):
        """Test stop trigger logic for buy."""
        stop = StopOrder(
            order_id="stop_1",
            symbol="ES",
            side=Side.BUY,
            qty=10.0,
            stop_price=4510.00,
            trigger_type=StopTriggerType.LAST_TRADE,
        )

        # Should not trigger below
        assert stop.should_trigger(last_trade_price=4505.00) is False

        # Should trigger at or above
        assert stop.should_trigger(last_trade_price=4510.00) is True
        assert stop.should_trigger(last_trade_price=4515.00) is True

    def test_stop_should_trigger_sell(self):
        """Test stop trigger logic for sell."""
        stop = StopOrder(
            order_id="stop_1",
            symbol="ES",
            side=Side.SELL,
            qty=10.0,
            stop_price=4490.00,
            trigger_type=StopTriggerType.LAST_TRADE,
        )

        # Should not trigger above
        assert stop.should_trigger(last_trade_price=4495.00) is False

        # Should trigger at or below
        assert stop.should_trigger(last_trade_price=4490.00) is True
        assert stop.should_trigger(last_trade_price=4485.00) is True


# =============================================================================
# Test: Session Detection
# =============================================================================

class TestSessionDetection:
    """Test CME session detection."""

    def test_rth_detection(self):
        """Test RTH session detection."""
        # 10am ET on a Tuesday
        ts = datetime(2025, 12, 2, 10, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)

        session = get_cme_session(ts_ms)
        # Note: 10am UTC = 5am ET during EST, so it's actually ETH
        # Let's use 15:00 UTC = 10am ET
        ts = datetime(2025, 12, 2, 15, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)
        session = get_cme_session(ts_ms)
        assert session == CMESession.RTH

    def test_eth_detection(self):
        """Test ETH session detection."""
        # 3am ET on a Tuesday (8am UTC)
        ts = datetime(2025, 12, 2, 8, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)

        session = get_cme_session(ts_ms)
        assert session == CMESession.ETH

    def test_weekend_closed(self):
        """Test weekend detection."""
        # Saturday
        ts = datetime(2025, 11, 29, 12, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)

        session = get_cme_session(ts_ms)
        assert session == CMESession.CLOSED

    def test_is_rth_session_helper(self):
        """Test is_rth_session helper function."""
        # RTH time (10am ET = 15:00 UTC)
        ts = datetime(2025, 12, 2, 15, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)
        assert is_rth_session(ts_ms) is True

        # ETH time (3am ET = 8:00 UTC)
        ts = datetime(2025, 12, 2, 8, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)
        assert is_rth_session(ts_ms) is False

    def test_minutes_to_settlement(self):
        """Test minutes to settlement calculation."""
        # 3pm ET (ES settlement at 4:00pm ET = 16:00)
        ts = datetime(2025, 12, 2, 20, 0, 0, tzinfo=timezone.utc)  # 3pm ET = 20:00 UTC
        ts_ms = int(ts.timestamp() * 1000)

        minutes = get_minutes_to_settlement(ts_ms, "ES")
        assert minutes == pytest.approx(60, abs=5)


# =============================================================================
# Test: Daily Settlement Simulator
# =============================================================================

class TestDailySettlementSimulator:
    """Test daily settlement simulation."""

    def test_create_simulator(self, settlement_simulator):
        """Test simulator creation."""
        assert settlement_simulator is not None
        assert settlement_simulator._symbol == "ES"
        assert settlement_simulator._multiplier == Decimal("50")

    def test_process_first_settlement(self, settlement_simulator):
        """Test first settlement (no prior)."""
        event = settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("1"),
        )

        assert event is not None
        assert event.settlement_price == Decimal("4500.00")
        assert event.variation_margin == Decimal("0")  # First settlement

    def test_process_subsequent_settlement(self, settlement_simulator):
        """Test subsequent settlement with variation margin."""
        # First settlement
        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("1"),
        )

        # Second settlement - price went up
        event = settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000) + 86400000,
            settlement_price=Decimal("4510.00"),
            position_qty=Decimal("1"),
        )

        # VM = (4510 - 4500) * 1 * 50 = 500
        assert event.variation_margin == Decimal("500.00")

    def test_variation_margin_loss(self, settlement_simulator):
        """Test variation margin for loss."""
        # First settlement
        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("1"),
        )

        # Second settlement - price went down
        event = settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000) + 86400000,
            settlement_price=Decimal("4490.00"),
            position_qty=Decimal("1"),
        )

        # VM = (4490 - 4500) * 1 * 50 = -500
        assert event.variation_margin == Decimal("-500.00")

    def test_short_position_variation_margin(self, settlement_simulator):
        """Test variation margin for short position."""
        # First settlement
        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("-1"),
        )

        # Second settlement - price went up (loss for short)
        event = settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000) + 86400000,
            settlement_price=Decimal("4510.00"),
            position_qty=Decimal("-1"),
        )

        # VM = (4510 - 4500) * (-1) * 50 = -500 (loss)
        assert event.variation_margin == Decimal("-500.00")

    def test_get_last_settlement_price(self, settlement_simulator):
        """Test getting last settlement price."""
        assert settlement_simulator.get_last_settlement_price() is None

        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("1"),
        )

        assert settlement_simulator.get_last_settlement_price() == Decimal("4500.00")

    def test_clear_variation_margin(self, settlement_simulator):
        """Test clearing accumulated variation margin."""
        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000),
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("1"),
        )
        settlement_simulator.process_settlement(
            timestamp_ms=int(time.time() * 1000) + 86400000,
            settlement_price=Decimal("4510.00"),
            position_qty=Decimal("1"),
        )

        total_vm = settlement_simulator.get_pending_variation_margin()
        assert total_vm == Decimal("500.00")

        settlement_simulator.clear_variation_margin()
        assert settlement_simulator.get_pending_variation_margin() == Decimal("0")


# =============================================================================
# Test: CMEL3SlippageProvider
# =============================================================================

class TestCMEL3SlippageProvider:
    """Test CMEL3 slippage provider."""

    def test_create_provider(self):
        """Test provider creation."""
        provider = CMEL3SlippageProvider(symbol="ES")
        assert provider is not None
        assert provider._symbol == "ES"

    def test_compute_slippage_without_lob(self, sample_market_state):
        """Test slippage computation without LOB."""
        provider = CMEL3SlippageProvider(symbol="ES")
        order = Order(symbol="ES", side="BUY", qty=5.0, order_type="MARKET")

        slippage = provider.compute_slippage_bps(
            order=order,
            market=sample_market_state,
            participation_ratio=0.001,
            is_rth=True,
        )

        assert slippage > 0
        assert slippage < 100  # Reasonable range

    def test_eth_spread_multiplier(self, sample_market_state):
        """Test ETH spread multiplier."""
        provider = CMEL3SlippageProvider(symbol="ES")
        order = Order(symbol="ES", side="BUY", qty=5.0, order_type="MARKET")

        # RTH slippage
        rth_slippage = provider.compute_slippage_bps(
            order=order,
            market=sample_market_state,
            participation_ratio=0.001,
            is_rth=True,
        )

        # ETH slippage (should be higher)
        eth_slippage = provider.compute_slippage_bps(
            order=order,
            market=sample_market_state,
            participation_ratio=0.001,
            is_rth=False,
        )

        assert eth_slippage >= rth_slippage

    def test_settlement_premium(self, sample_market_state):
        """Test settlement time premium."""
        provider = CMEL3SlippageProvider(symbol="ES")
        order = Order(symbol="ES", side="BUY", qty=5.0, order_type="MARKET")

        # Normal slippage
        normal_slippage = provider.compute_slippage_bps(
            order=order,
            market=sample_market_state,
            participation_ratio=0.001,
            is_rth=True,
            minutes_to_settlement=60,
        )

        # Near settlement (higher)
        settlement_slippage = provider.compute_slippage_bps(
            order=order,
            market=sample_market_state,
            participation_ratio=0.001,
            is_rth=True,
            minutes_to_settlement=5,
        )

        assert settlement_slippage >= normal_slippage


# =============================================================================
# Test: CMEL3FillProvider
# =============================================================================

class TestCMEL3FillProvider:
    """Test CMEL3 fill provider."""

    def test_create_provider(self):
        """Test provider creation."""
        slippage = CMEL3SlippageProvider(symbol="ES")
        fees = CMEFeeProvider()
        provider = CMEL3FillProvider(
            symbol="ES",
            slippage_provider=slippage,
            fee_provider=fees,
        )
        assert provider is not None

    def test_fill_market_order(self, sample_market_state, sample_bar_data):
        """Test market order filling."""
        slippage = CMEL3SlippageProvider(symbol="ES")
        fees = CMEFeeProvider()
        provider = CMEL3FillProvider(
            symbol="ES",
            slippage_provider=slippage,
            fee_provider=fees,
        )

        order = Order(symbol="ES", side="BUY", qty=1.0, order_type="MARKET")
        fill = provider.try_fill(
            order=order,
            market=sample_market_state,
            bar=sample_bar_data,
            is_rth=True,
        )

        assert fill is not None
        assert fill.qty == 1.0
        assert fill.price > 0


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_cme_l3_execution_provider(self):
        """Test creating full L3 provider."""
        provider = create_cme_l3_execution_provider(symbol="ES")
        assert provider is not None
        assert provider._symbol == "ES"

    def test_create_with_profile(self):
        """Test creating with slippage profile."""
        provider = create_cme_l3_execution_provider(
            symbol="ES",
            profile="conservative",
        )
        assert provider is not None

    def test_create_slippage_provider(self):
        """Test creating just slippage provider."""
        provider = create_cme_l3_slippage_provider(symbol="NQ")
        assert provider is not None
        assert provider._symbol == "NQ"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_order_book(self, es_matching_engine):
        """Test operations on empty order book."""
        assert es_matching_engine.get_best_bid() is None
        assert es_matching_engine.get_best_ask() is None
        assert es_matching_engine.get_spread() is None

    def test_zero_qty_order(self, es_matching_engine):
        """Test handling zero quantity order."""
        order = LimitOrder(
            order_id="zero_1",
            price=4500.00,
            qty=0.0,
            remaining_qty=0.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=LOBOrderType.LIMIT,
        )

        result = es_matching_engine.add_order(order)
        # Should handle gracefully
        assert result is not None

    def test_various_symbols(self):
        """Test with various symbols."""
        for symbol in ["ES", "NQ", "GC", "CL", "6E", "ZN"]:
            engine = create_globex_matching_engine(symbol=symbol)
            assert engine.symbol == symbol
            assert engine.protection_points > 0


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_execution_flow(self, sample_market_state, sample_bar_data):
        """Test full execution flow."""
        provider = create_cme_l3_execution_provider(symbol="ES")

        order = Order(symbol="ES", side="BUY", qty=1.0, order_type="MARKET")
        fill = provider.execute(
            order=order,
            market=sample_market_state,
            bar=sample_bar_data,
        )

        assert fill is not None
        assert fill.qty == 1.0
        assert fill.price > 0
        assert fill.fee > 0

    def test_settlement_flow(self):
        """Test settlement flow."""
        simulator = DailySettlementSimulator(symbol="ES", contract_multiplier=Decimal("50"))

        # Day 1
        simulator.process_settlement(
            timestamp_ms=1000,
            settlement_price=Decimal("4500.00"),
            position_qty=Decimal("2"),
        )

        # Day 2
        simulator.process_settlement(
            timestamp_ms=86401000,
            settlement_price=Decimal("4520.00"),
            position_qty=Decimal("2"),
        )

        # Total VM = (4520-4500) * 2 * 50 = 2000
        assert simulator.get_pending_variation_margin() == Decimal("2000.00")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
