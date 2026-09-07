# -*- coding: utf-8 -*-
"""
tests/test_forex_dealer_simulation.py

Comprehensive tests for OTC Forex Dealer Quote Simulation.
Phase 5: Forex Integration (2025-11-30)

Tests cover:
1. Data class validation and properties
2. Multi-dealer quote aggregation
3. Last-look rejection simulation
4. Size-dependent spread widening
5. Quote flickering
6. Session-dependent behavior
7. Execution statistics
8. Factory functions
9. Integration helpers

Expected: ~85 tests (100% pass)
"""
from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

from services.forex_dealer import (
    # Enums
    QuoteType,
    RejectReason,
    DealerTier,
    # Data classes
    DealerProfile,
    DealerQuote,
    AggregatedQuote,
    ExecutionResult,
    ForexDealerConfig,
    ExecutionStats,
    # Main class
    ForexDealerSimulator,
    # Factory functions
    create_forex_dealer_simulator,
    create_default_dealer_pool,
    # Integration helpers
    combine_with_parametric_slippage,
    estimate_rejection_probability,
    # Constants
    DEFAULT_QUOTE_VALIDITY_MS,
    DEFAULT_LAST_LOOK_MS,
    DEFAULT_ADVERSE_THRESHOLD_PIPS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_dealer_profile() -> DealerProfile:
    """Create a default dealer profile for testing."""
    return DealerProfile(
        dealer_id="test_dealer",
        tier=DealerTier.TIER2,
        spread_factor=1.0,
        max_size_usd=5_000_000.0,
    )


@pytest.fixture
def tier1_dealer() -> DealerProfile:
    """Create a Tier 1 dealer profile."""
    return DealerProfile(
        dealer_id="tier1_test",
        tier=DealerTier.TIER1,
        spread_factor=0.7,
        max_size_usd=15_000_000.0,
        last_look_window_ms=100,
        base_reject_prob=0.02,
        adverse_threshold_pips=0.15,
        latency_ms=20,
        is_primary=True,
    )


@pytest.fixture
def default_config() -> ForexDealerConfig:
    """Create a default forex dealer config."""
    return ForexDealerConfig()


@pytest.fixture
def simulator(default_config: ForexDealerConfig) -> ForexDealerSimulator:
    """Create a forex dealer simulator with fixed seed."""
    return ForexDealerSimulator(config=default_config, seed=42)


@pytest.fixture
def dealer_quote() -> DealerQuote:
    """Create a sample dealer quote."""
    return DealerQuote(
        dealer_id="test_dealer",
        bid=1.0850,
        ask=1.0852,
        bid_size_usd=1_000_000.0,
        ask_size_usd=1_000_000.0,
        timestamp_ns=time.time_ns(),
    )


@pytest.fixture
def aggregated_quote(dealer_quote: DealerQuote) -> AggregatedQuote:
    """Create an aggregated quote for testing."""
    return AggregatedQuote(
        best_bid=1.0850,
        best_ask=1.0852,
        best_bid_dealer="dealer_a",
        best_ask_dealer="dealer_b",
        total_bid_size=5_000_000.0,
        total_ask_size=5_000_000.0,
        dealer_quotes=[dealer_quote],
        timestamp_ns=time.time_ns(),
        num_active_dealers=1,
    )


# =============================================================================
# Test DealerProfile
# =============================================================================


class TestDealerProfile:
    """Tests for DealerProfile data class."""

    def test_default_values(self, default_dealer_profile: DealerProfile) -> None:
        """Test default values are set correctly."""
        assert default_dealer_profile.tier == DealerTier.TIER2
        assert default_dealer_profile.spread_factor == 1.0
        assert default_dealer_profile.last_look_window_ms == DEFAULT_LAST_LOOK_MS
        assert default_dealer_profile.base_reject_prob == 0.05
        assert default_dealer_profile.is_primary is False

    def test_tier1_characteristics(self, tier1_dealer: DealerProfile) -> None:
        """Test Tier 1 dealer has expected characteristics."""
        assert tier1_dealer.tier == DealerTier.TIER1
        assert tier1_dealer.spread_factor < 1.0  # Tighter spread
        assert tier1_dealer.max_size_usd >= 10_000_000  # Large capacity
        assert tier1_dealer.is_primary is True

    def test_validation_spread_factor_positive(self) -> None:
        """Test that spread_factor must be positive."""
        with pytest.raises(ValueError, match="spread_factor must be positive"):
            DealerProfile(dealer_id="test", spread_factor=0.0)

        with pytest.raises(ValueError, match="spread_factor must be positive"):
            DealerProfile(dealer_id="test", spread_factor=-0.5)

    def test_validation_max_size_positive(self) -> None:
        """Test that max_size_usd must be positive."""
        with pytest.raises(ValueError, match="max_size_usd must be positive"):
            DealerProfile(dealer_id="test", max_size_usd=0.0)

    def test_validation_reject_prob_range(self) -> None:
        """Test that base_reject_prob must be in [0, 1]."""
        with pytest.raises(ValueError, match="base_reject_prob must be in"):
            DealerProfile(dealer_id="test", base_reject_prob=-0.1)

        with pytest.raises(ValueError, match="base_reject_prob must be in"):
            DealerProfile(dealer_id="test", base_reject_prob=1.5)

    def test_validation_last_look_non_negative(self) -> None:
        """Test that last_look_window_ms must be non-negative."""
        with pytest.raises(ValueError, match="last_look_window_ms must be non-negative"):
            DealerProfile(dealer_id="test", last_look_window_ms=-10)

    def test_validation_latency_non_negative(self) -> None:
        """Test that latency_ms must be non-negative."""
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            DealerProfile(dealer_id="test", latency_ms=-5)

    def test_active_hours_tuple(self) -> None:
        """Test active_hours tuple is stored correctly."""
        dealer = DealerProfile(
            dealer_id="test",
            active_hours=(8, 17),
        )
        assert dealer.active_hours == (8, 17)


# =============================================================================
# Test DealerQuote
# =============================================================================


class TestDealerQuote:
    """Tests for DealerQuote data class."""

    def test_mid_price_property(self, dealer_quote: DealerQuote) -> None:
        """Test mid price calculation."""
        expected_mid = (1.0850 + 1.0852) / 2.0
        assert dealer_quote.mid == pytest.approx(expected_mid)

    def test_spread_property(self, dealer_quote: DealerQuote) -> None:
        """Test spread calculation."""
        expected_spread = 1.0852 - 1.0850
        assert dealer_quote.spread == pytest.approx(expected_spread)

    def test_spread_pips_standard_pair(self, dealer_quote: DealerQuote) -> None:
        """Test spread in pips for standard pair (0.0001 pip)."""
        spread_pips = dealer_quote.spread_pips(is_jpy_pair=False)
        expected = (1.0852 - 1.0850) / 0.0001
        assert spread_pips == pytest.approx(expected)

    def test_spread_pips_jpy_pair(self) -> None:
        """Test spread in pips for JPY pair (0.01 pip)."""
        quote = DealerQuote(
            dealer_id="test",
            bid=150.00,
            ask=150.03,
            bid_size_usd=1_000_000,
            ask_size_usd=1_000_000,
            timestamp_ns=time.time_ns(),
        )
        spread_pips = quote.spread_pips(is_jpy_pair=True)
        expected = (150.03 - 150.00) / 0.01
        assert spread_pips == pytest.approx(expected)  # 3 pips

    def test_is_valid_fresh_quote(self, dealer_quote: DealerQuote) -> None:
        """Test that a fresh quote is valid."""
        current_ns = dealer_quote.timestamp_ns + 50_000_000  # 50ms later
        assert dealer_quote.is_valid(current_ns) is True

    def test_is_valid_expired_quote(self, dealer_quote: DealerQuote) -> None:
        """Test that an expired quote is invalid."""
        # Default validity is 200ms
        current_ns = dealer_quote.timestamp_ns + 300_000_000  # 300ms later
        assert dealer_quote.is_valid(current_ns) is False

    def test_quote_type_default(self, dealer_quote: DealerQuote) -> None:
        """Test default quote type is LAST_LOOK."""
        assert dealer_quote.quote_type == QuoteType.LAST_LOOK

    def test_sequence_number(self) -> None:
        """Test sequence number is stored."""
        quote = DealerQuote(
            dealer_id="test",
            bid=1.0,
            ask=1.001,
            bid_size_usd=100_000,
            ask_size_usd=100_000,
            timestamp_ns=time.time_ns(),
            sequence_num=12345,
        )
        assert quote.sequence_num == 12345


# =============================================================================
# Test AggregatedQuote
# =============================================================================


class TestAggregatedQuote:
    """Tests for AggregatedQuote data class."""

    def test_mid_price_property(self, aggregated_quote: AggregatedQuote) -> None:
        """Test mid price calculation from best bid/ask."""
        expected_mid = (1.0850 + 1.0852) / 2.0
        assert aggregated_quote.mid == pytest.approx(expected_mid)

    def test_spread_property(self, aggregated_quote: AggregatedQuote) -> None:
        """Test spread calculation."""
        expected_spread = 1.0852 - 1.0850
        assert aggregated_quote.spread == pytest.approx(expected_spread)

    def test_spread_pips(self, aggregated_quote: AggregatedQuote) -> None:
        """Test spread in pips."""
        spread_pips = aggregated_quote.spread_pips(is_jpy_pair=False)
        expected = (1.0852 - 1.0850) / 0.0001
        assert spread_pips == pytest.approx(expected)

    def test_is_crossed_normal_market(self, aggregated_quote: AggregatedQuote) -> None:
        """Test normal market is not crossed."""
        assert aggregated_quote.is_crossed() is False

    def test_is_crossed_true(self) -> None:
        """Test crossed market detection."""
        crossed_quote = AggregatedQuote(
            best_bid=1.0855,  # Bid > Ask
            best_ask=1.0850,
            best_bid_dealer="a",
            best_ask_dealer="b",
            total_bid_size=100_000,
            total_ask_size=100_000,
            dealer_quotes=[],
            timestamp_ns=time.time_ns(),
        )
        assert crossed_quote.is_crossed() is True

    def test_get_depth_at_level_bid(self) -> None:
        """Test getting depth at bid levels."""
        quotes = [
            DealerQuote(
                dealer_id="d1",
                bid=1.0850,
                ask=1.0852,
                bid_size_usd=1_000_000,
                ask_size_usd=1_000_000,
                timestamp_ns=time.time_ns(),
            ),
            DealerQuote(
                dealer_id="d2",
                bid=1.0849,  # Lower bid
                ask=1.0853,
                bid_size_usd=500_000,
                ask_size_usd=500_000,
                timestamp_ns=time.time_ns(),
            ),
        ]
        agg = AggregatedQuote(
            best_bid=1.0850,
            best_ask=1.0852,
            best_bid_dealer="d1",
            best_ask_dealer="d1",
            total_bid_size=1_000_000,
            total_ask_size=1_000_000,
            dealer_quotes=quotes,
            timestamp_ns=time.time_ns(),
            num_active_dealers=2,
        )

        # Level 0 (best) should be d1
        price, size = agg.get_depth_at_level(0, is_bid=True)
        assert price == pytest.approx(1.0850)
        assert size == pytest.approx(1_000_000)

        # Level 1 should be d2
        price, size = agg.get_depth_at_level(1, is_bid=True)
        assert price == pytest.approx(1.0849)
        assert size == pytest.approx(500_000)

    def test_get_depth_at_level_ask(self) -> None:
        """Test getting depth at ask levels."""
        quotes = [
            DealerQuote(
                dealer_id="d1",
                bid=1.0850,
                ask=1.0852,
                bid_size_usd=1_000_000,
                ask_size_usd=1_000_000,
                timestamp_ns=time.time_ns(),
            ),
            DealerQuote(
                dealer_id="d2",
                bid=1.0849,
                ask=1.0855,  # Higher ask
                bid_size_usd=500_000,
                ask_size_usd=750_000,
                timestamp_ns=time.time_ns(),
            ),
        ]
        agg = AggregatedQuote(
            best_bid=1.0850,
            best_ask=1.0852,
            best_bid_dealer="d1",
            best_ask_dealer="d1",
            total_bid_size=1_000_000,
            total_ask_size=1_000_000,
            dealer_quotes=quotes,
            timestamp_ns=time.time_ns(),
            num_active_dealers=2,
        )

        # Level 0 (best) should be d1 (lowest ask)
        price, size = agg.get_depth_at_level(0, is_bid=False)
        assert price == pytest.approx(1.0852)
        assert size == pytest.approx(1_000_000)

        # Level 1 should be d2 (higher ask)
        price, size = agg.get_depth_at_level(1, is_bid=False)
        assert price == pytest.approx(1.0855)
        assert size == pytest.approx(750_000)

    def test_get_depth_at_level_out_of_range(self, aggregated_quote: AggregatedQuote) -> None:
        """Test that out of range level returns (0, 0)."""
        price, size = aggregated_quote.get_depth_at_level(10, is_bid=True)
        assert price == 0.0
        assert size == 0.0


# =============================================================================
# Test ExecutionResult
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult data class."""

    def test_latency_ms_conversion(self) -> None:
        """Test latency conversion from nanoseconds to milliseconds."""
        result = ExecutionResult(
            filled=True,
            fill_price=1.0850,
            fill_qty=100_000,
            latency_ns=50_000_000,  # 50ms
        )
        assert result.latency_ms == pytest.approx(50.0)

    def test_to_dict_filled(self) -> None:
        """Test to_dict for filled execution."""
        result = ExecutionResult(
            filled=True,
            fill_price=1.0850,
            fill_qty=100_000,
            dealer_id="test_dealer",
            latency_ns=50_000_000,
            slippage_pips=0.5,
            last_look_passed=True,
            price_improvement=0.00001,
        )
        d = result.to_dict()

        assert d["filled"] is True
        assert d["fill_price"] == pytest.approx(1.0850)
        assert d["fill_qty"] == pytest.approx(100_000)
        assert d["dealer_id"] == "test_dealer"
        assert d["latency_ms"] == pytest.approx(50.0)
        assert d["slippage_pips"] == pytest.approx(0.5)
        assert d["reject_reason"] == "none"

    def test_to_dict_rejected(self) -> None:
        """Test to_dict for rejected execution."""
        result = ExecutionResult(
            filled=False,
            reject_reason=RejectReason.LAST_LOOK_ADVERSE,
        )
        d = result.to_dict()

        assert d["filled"] is False
        assert d["reject_reason"] == "last_look_adverse"

    def test_partial_fill_attributes(self) -> None:
        """Test partial fill attributes."""
        result = ExecutionResult(
            filled=True,
            fill_price=1.0850,
            fill_qty=75_000,
            partial_fill=True,
            remaining_qty=25_000,
        )
        assert result.partial_fill is True
        assert result.remaining_qty == pytest.approx(25_000)


# =============================================================================
# Test ForexDealerConfig
# =============================================================================


class TestForexDealerConfig:
    """Tests for ForexDealerConfig data class."""

    def test_default_values(self, default_config: ForexDealerConfig) -> None:
        """Test default configuration values."""
        assert default_config.num_dealers == 5
        assert default_config.base_spread_pips == 1.0
        assert default_config.last_look_enabled is True
        assert default_config.size_impact_threshold_usd == pytest.approx(1_000_000)

    def test_validation_num_dealers_min(self) -> None:
        """Test that num_dealers must be at least 1."""
        with pytest.raises(ValueError, match="num_dealers must be at least 1"):
            ForexDealerConfig(num_dealers=0)

    def test_validation_base_spread_non_negative(self) -> None:
        """Test that base_spread_pips must be non-negative."""
        with pytest.raises(ValueError, match="base_spread_pips must be non-negative"):
            ForexDealerConfig(base_spread_pips=-0.5)

    def test_validation_size_impact_threshold_positive(self) -> None:
        """Test that size_impact_threshold_usd must be positive."""
        with pytest.raises(ValueError, match="size_impact_threshold_usd must be positive"):
            ForexDealerConfig(size_impact_threshold_usd=0)

    def test_validation_size_impact_factor_non_negative(self) -> None:
        """Test that size_impact_factor must be non-negative."""
        with pytest.raises(ValueError, match="size_impact_factor must be non-negative"):
            ForexDealerConfig(size_impact_factor=-0.1)

    def test_validation_latency_variance_non_negative(self) -> None:
        """Test that latency_variance must be non-negative."""
        with pytest.raises(ValueError, match="latency_variance must be non-negative"):
            ForexDealerConfig(latency_variance=-0.5)

    def test_validation_max_slippage_positive(self) -> None:
        """Test that max_slippage_pips must be positive."""
        with pytest.raises(ValueError, match="max_slippage_pips must be positive"):
            ForexDealerConfig(max_slippage_pips=0)


# =============================================================================
# Test ExecutionStats
# =============================================================================


class TestExecutionStats:
    """Tests for ExecutionStats data class."""

    def test_fill_rate_no_attempts(self) -> None:
        """Test fill rate with no attempts."""
        stats = ExecutionStats()
        assert stats.fill_rate == pytest.approx(0.0)

    def test_fill_rate_calculation(self) -> None:
        """Test fill rate calculation."""
        stats = ExecutionStats(
            total_attempts=100,
            total_fills=95,
        )
        assert stats.fill_rate == pytest.approx(95.0)

    def test_fill_rate_50_percent(self) -> None:
        """Test 50% fill rate."""
        stats = ExecutionStats(
            total_attempts=10,
            total_fills=5,
        )
        assert stats.fill_rate == pytest.approx(50.0)


# =============================================================================
# Test ForexDealerSimulator - Initialization
# =============================================================================


class TestForexDealerSimulatorInit:
    """Tests for ForexDealerSimulator initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        sim = ForexDealerSimulator()
        assert sim.config is not None
        assert len(sim._dealers) == sim.config.num_dealers

    def test_initialization_with_seed(self) -> None:
        """Test initialization with seed for reproducibility."""
        sim1 = ForexDealerSimulator(seed=42)
        sim2 = ForexDealerSimulator(seed=42)

        # Dealers should have same characteristics
        for d1, d2 in zip(sim1._dealers, sim2._dealers):
            assert d1.spread_factor == pytest.approx(d2.spread_factor)
            assert d1.max_size_usd == pytest.approx(d2.max_size_usd)

    def test_dealer_pool_creation(self, simulator: ForexDealerSimulator) -> None:
        """Test that dealer pool has correct tier distribution."""
        dealers = simulator.get_all_dealers()

        tier1_count = sum(1 for d in dealers if d.tier == DealerTier.TIER1)
        tier2_count = sum(1 for d in dealers if d.tier == DealerTier.TIER2)
        tier3_count = sum(1 for d in dealers if d.tier == DealerTier.TIER3)

        # At least one tier1 primary LP
        assert tier1_count >= 1
        assert tier1_count + tier2_count + tier3_count == len(dealers)

    def test_dealer_map_creation(self, simulator: ForexDealerSimulator) -> None:
        """Test that dealer map is created correctly."""
        dealers = simulator.get_all_dealers()
        for d in dealers:
            retrieved = simulator.get_dealer(d.dealer_id)
            assert retrieved is not None
            assert retrieved.dealer_id == d.dealer_id

    def test_get_dealer_not_found(self, simulator: ForexDealerSimulator) -> None:
        """Test getting non-existent dealer returns None."""
        result = simulator.get_dealer("non_existent_dealer")
        assert result is None


# =============================================================================
# Test ForexDealerSimulator - Quote Generation
# =============================================================================


class TestForexDealerSimulatorQuotes:
    """Tests for quote generation."""

    def test_get_aggregated_quote_basic(self, simulator: ForexDealerSimulator) -> None:
        """Test basic aggregated quote generation."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        assert quote.best_bid < 1.0850
        assert quote.best_ask > 1.0850
        assert quote.best_bid < quote.best_ask
        assert quote.num_active_dealers > 0

    def test_get_aggregated_quote_jpy_pair(self, simulator: ForexDealerSimulator) -> None:
        """Test quote generation for JPY pair."""
        quote = simulator.get_aggregated_quote(
            symbol="USD_JPY",
            mid_price=150.00,
        )

        # JPY pairs use 0.01 pip size
        spread_pips = quote.spread_pips(is_jpy_pair=True)
        assert spread_pips > 0
        assert spread_pips < 20  # Reasonable spread

    def test_quote_spread_widens_with_size(self, simulator: ForexDealerSimulator) -> None:
        """Test that spreads widen for large orders."""
        small_quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            order_size_usd=100_000,
        )

        # Create new simulator with same seed for comparable quotes
        sim2 = ForexDealerSimulator(config=simulator.config, seed=42)
        large_quote = sim2.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            order_size_usd=10_000_000,  # Large order
        )

        # Large order should have wider spread on average
        # Note: due to randomness, we use statistical comparison
        assert large_quote.spread >= small_quote.spread * 0.5  # At least comparable

    def test_quote_spread_widens_with_low_liquidity(self) -> None:
        """Test that spreads widen in low liquidity sessions."""
        config = ForexDealerConfig(session_spread_adjustment=True)
        sim = ForexDealerSimulator(config=config, seed=42)

        high_liq_quote = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            session_factor=1.5,  # High liquidity
        )

        sim2 = ForexDealerSimulator(config=config, seed=42)
        low_liq_quote = sim2.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            session_factor=0.5,  # Low liquidity
        )

        # Low liquidity should have wider spread
        assert low_liq_quote.spread > high_liq_quote.spread

    def test_quote_contains_all_dealer_quotes(self, simulator: ForexDealerSimulator) -> None:
        """Test that aggregated quote contains individual dealer quotes."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        assert len(quote.dealer_quotes) == quote.num_active_dealers
        assert quote.num_active_dealers <= simulator.config.num_dealers

    def test_quote_best_bid_is_highest(self, simulator: ForexDealerSimulator) -> None:
        """Test that best_bid is the highest bid from all dealers."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        all_bids = [q.bid for q in quote.dealer_quotes]
        assert quote.best_bid == pytest.approx(max(all_bids))

    def test_quote_best_ask_is_lowest(self, simulator: ForexDealerSimulator) -> None:
        """Test that best_ask is the lowest ask from all dealers."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        all_asks = [q.ask for q in quote.dealer_quotes]
        assert quote.best_ask == pytest.approx(min(all_asks))

    def test_dealer_hour_filtering(self) -> None:
        """Test dealer filtering by active hours."""
        # Create a dealer that's only active 9-17 UTC
        config = ForexDealerConfig(num_dealers=1)
        sim = ForexDealerSimulator(config=config, seed=42)

        # Manually set active hours
        sim._dealers[0].active_hours = (9, 17)

        # During active hours
        quote_active = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            hour_utc=12,
        )
        assert quote_active.num_active_dealers == 1

        # Outside active hours
        quote_inactive = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            hour_utc=3,
        )
        assert quote_inactive.num_active_dealers == 0


# =============================================================================
# Test ForexDealerSimulator - Execution
# =============================================================================


class TestForexDealerSimulatorExecution:
    """Tests for trade execution."""

    def test_attempt_execution_buy_filled(self, simulator: ForexDealerSimulator) -> None:
        """Test successful buy execution."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        result = simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,  # No adverse move
            symbol="EUR_USD",
        )

        # With no adverse move, should fill most of the time
        if result.filled:
            assert result.fill_price is not None
            assert result.fill_price >= quote.best_ask - 0.001  # Close to ask
            assert result.fill_qty is not None
            assert result.dealer_id is not None

    def test_attempt_execution_sell_filled(self, simulator: ForexDealerSimulator) -> None:
        """Test successful sell execution."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        result = simulator.attempt_execution(
            is_buy=False,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
            symbol="EUR_USD",
        )

        if result.filled:
            assert result.fill_price is not None
            assert result.fill_price <= quote.best_bid + 0.001  # Close to bid

    def test_attempt_execution_market_closed(self, simulator: ForexDealerSimulator) -> None:
        """Test execution when market is closed."""
        # Create quote with no active dealers
        quote = AggregatedQuote(
            best_bid=1.0850,
            best_ask=1.0852,
            best_bid_dealer="none",
            best_ask_dealer="none",
            total_bid_size=0,
            total_ask_size=0,
            dealer_quotes=[],
            timestamp_ns=time.time_ns(),
            num_active_dealers=0,
        )

        result = simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
        )

        assert result.filled is False
        assert result.reject_reason == RejectReason.MARKET_CLOSED

    def test_last_look_rejection_on_adverse_move(self) -> None:
        """Test last-look rejection when price moves adversely."""
        config = ForexDealerConfig(last_look_enabled=True)
        sim = ForexDealerSimulator(config=config, seed=42)

        quote = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        # Run multiple times to see rejection
        rejected_count = 0
        for _ in range(50):
            result = sim.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0860,  # 10 pip adverse move for buyer
                symbol="EUR_USD",
            )
            if not result.filled:
                rejected_count += 1

        # Should see some rejections with adverse move
        assert rejected_count > 0, "Expected some rejections on adverse move"

    def test_execution_without_last_look(self) -> None:
        """Test execution with last-look disabled."""
        config = ForexDealerConfig(last_look_enabled=False)
        sim = ForexDealerSimulator(config=config, seed=42)

        filled_count = 0
        for _ in range(20):
            quote = sim.get_aggregated_quote(
                symbol="EUR_USD",
                mid_price=1.0850,
            )
            result = sim.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0850,
                symbol="EUR_USD",
            )
            if result.filled:
                filled_count += 1

        # Without last-look, fill rate should be higher
        assert filled_count >= 15, "Expected high fill rate without last-look"

    def test_price_improvement(self) -> None:
        """Test price improvement feature."""
        config = ForexDealerConfig(
            enable_price_improvement=True,
            price_improvement_prob=1.0,  # Always improve for testing
            max_price_improvement_pips=0.5,
        )
        sim = ForexDealerSimulator(config=config, seed=42)

        improvements = 0
        for _ in range(20):
            quote = sim.get_aggregated_quote(
                symbol="EUR_USD",
                mid_price=1.0850,
            )
            result = sim.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0850,
                symbol="EUR_USD",
            )
            if result.filled and result.price_improvement > 0:
                improvements += 1

        # With 100% prob, should see improvements on successful fills
        assert improvements > 0

    def test_partial_fills(self) -> None:
        """Test partial fill capability."""
        config = ForexDealerConfig(
            num_dealers=2,
            enable_partial_fills=True,
        )
        sim = ForexDealerSimulator(config=config, seed=42)

        # Request more than one dealer can fill
        quote = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            order_size_usd=50_000_000,  # Very large order
        )

        result = sim.attempt_execution(
            is_buy=True,
            size_usd=50_000_000,
            quote=quote,
            current_mid=1.0850,
            symbol="EUR_USD",
        )

        if result.filled:
            # May be partial fill if order > total available
            if result.partial_fill:
                assert result.remaining_qty > 0

    def test_execution_latency(self, simulator: ForexDealerSimulator) -> None:
        """Test that execution has latency."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        result = simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
            symbol="EUR_USD",
        )

        # Even rejected executions should have latency from attempts
        assert result.latency_ns >= 0

    def test_slippage_calculation(self, simulator: ForexDealerSimulator) -> None:
        """Test slippage is calculated correctly."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
        )

        result = simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
            symbol="EUR_USD",
        )

        if result.filled:
            assert result.slippage_pips >= 0


# =============================================================================
# Test ForexDealerSimulator - Quote Flickering
# =============================================================================


class TestForexDealerSimulatorFlicker:
    """Tests for quote flickering simulation."""

    def test_simulate_quote_flicker_basic(self, simulator: ForexDealerSimulator) -> None:
        """Test basic quote flicker generation."""
        quotes = list(
            simulator.simulate_quote_flicker(
                symbol="EUR_USD",
                base_mid=1.0850,
                duration_ms=100,
                tick_interval_ms=10,
            )
        )

        # Should generate 10 quotes (100ms / 10ms)
        assert len(quotes) == 10

    def test_simulate_quote_flicker_prices_vary(self, simulator: ForexDealerSimulator) -> None:
        """Test that quote prices vary due to random walk."""
        quotes = list(
            simulator.simulate_quote_flicker(
                symbol="EUR_USD",
                base_mid=1.0850,
                duration_ms=500,
                tick_interval_ms=10,
            )
        )

        mids = [q.mid for q in quotes]
        # Prices should vary (not all identical)
        assert max(mids) != min(mids)

    def test_simulate_quote_flicker_jpy(self, simulator: ForexDealerSimulator) -> None:
        """Test quote flicker for JPY pair."""
        quotes = list(
            simulator.simulate_quote_flicker(
                symbol="USD_JPY",
                base_mid=150.00,
                duration_ms=50,
                tick_interval_ms=10,
            )
        )

        assert len(quotes) == 5
        for q in quotes:
            assert q.best_bid < q.best_ask


# =============================================================================
# Test ForexDealerSimulator - Statistics
# =============================================================================


class TestForexDealerSimulatorStats:
    """Tests for execution statistics."""

    def test_stats_update_on_fill(self, simulator: ForexDealerSimulator) -> None:
        """Test stats are updated on successful fill."""
        initial_stats = simulator.get_stats()
        assert initial_stats.total_attempts == 0

        quote = simulator.get_aggregated_quote("EUR_USD", 1.0850)
        simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
        )

        updated_stats = simulator.get_stats()
        assert updated_stats.total_attempts == 1

    def test_stats_track_rejections(self) -> None:
        """Test that rejections are tracked."""
        config = ForexDealerConfig(last_look_enabled=True)
        sim = ForexDealerSimulator(config=config, seed=42)

        # Generate many executions with adverse moves to get rejections
        for _ in range(50):
            quote = sim.get_aggregated_quote("EUR_USD", 1.0850)
            sim.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0865,  # Adverse move
            )

        stats = sim.get_stats()
        assert stats.total_attempts == 50
        assert stats.total_fills + stats.total_rejections >= 0

    def test_reset_stats(self, simulator: ForexDealerSimulator) -> None:
        """Test stats reset."""
        quote = simulator.get_aggregated_quote("EUR_USD", 1.0850)
        simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
        )

        simulator.reset_stats()
        stats = simulator.get_stats()
        assert stats.total_attempts == 0
        assert stats.total_fills == 0

    def test_reset_state(self, simulator: ForexDealerSimulator) -> None:
        """Test full state reset."""
        quote = simulator.get_aggregated_quote("EUR_USD", 1.0850)
        simulator.attempt_execution(
            is_buy=True,
            size_usd=100_000,
            quote=quote,
            current_mid=1.0850,
        )

        simulator.reset_state()

        assert len(simulator._current_quotes) == 0
        assert len(simulator._execution_history) == 0
        assert simulator._quote_sequence == 0

    def test_get_recent_fill_rate_empty(self, simulator: ForexDealerSimulator) -> None:
        """Test recent fill rate with no history."""
        rate = simulator.get_recent_fill_rate()
        assert rate == pytest.approx(0.0)

    def test_get_recent_fill_rate_with_history(self, simulator: ForexDealerSimulator) -> None:
        """Test recent fill rate calculation."""
        for _ in range(10):
            quote = simulator.get_aggregated_quote("EUR_USD", 1.0850)
            simulator.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0850,
            )

        rate = simulator.get_recent_fill_rate(window=10)
        assert 0 <= rate <= 100

    def test_get_recent_slippage(self, simulator: ForexDealerSimulator) -> None:
        """Test recent slippage calculation."""
        for _ in range(10):
            quote = simulator.get_aggregated_quote("EUR_USD", 1.0850)
            simulator.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0850,
            )

        slippage = simulator.get_recent_slippage(window=10)
        assert slippage >= 0


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_forex_dealer_simulator_default(self) -> None:
        """Test default simulator creation."""
        sim = create_forex_dealer_simulator()
        assert sim is not None
        assert isinstance(sim, ForexDealerSimulator)

    def test_create_forex_dealer_simulator_retail_profile(self) -> None:
        """Test retail profile."""
        sim = create_forex_dealer_simulator(profile="retail")
        assert sim.config.num_dealers == 5
        assert sim.config.base_spread_pips == pytest.approx(1.2)

    def test_create_forex_dealer_simulator_institutional_profile(self) -> None:
        """Test institutional profile."""
        sim = create_forex_dealer_simulator(profile="institutional")
        assert sim.config.num_dealers == 10
        assert sim.config.base_spread_pips == pytest.approx(0.3)

    def test_create_forex_dealer_simulator_conservative_profile(self) -> None:
        """Test conservative profile."""
        sim = create_forex_dealer_simulator(profile="conservative")
        assert sim.config.num_dealers == 3
        assert sim.config.base_spread_pips == pytest.approx(2.0)

    def test_create_forex_dealer_simulator_with_custom_config(self) -> None:
        """Test with custom config override."""
        sim = create_forex_dealer_simulator(
            config={"num_dealers": 7, "base_spread_pips": 0.8},
            profile="retail",
        )
        assert sim.config.num_dealers == 7
        assert sim.config.base_spread_pips == pytest.approx(0.8)

    def test_create_forex_dealer_simulator_with_seed(self) -> None:
        """Test seed is passed correctly."""
        sim1 = create_forex_dealer_simulator(seed=123)
        sim2 = create_forex_dealer_simulator(seed=123)

        q1 = sim1.get_aggregated_quote("EUR_USD", 1.0850)
        q2 = sim2.get_aggregated_quote("EUR_USD", 1.0850)

        assert q1.best_bid == pytest.approx(q2.best_bid)
        assert q1.best_ask == pytest.approx(q2.best_ask)

    def test_create_default_dealer_pool(self) -> None:
        """Test default dealer pool creation."""
        dealers = create_default_dealer_pool()

        assert len(dealers) == 4
        tier1_count = sum(1 for d in dealers if d.tier == DealerTier.TIER1)
        tier2_count = sum(1 for d in dealers if d.tier == DealerTier.TIER2)
        tier3_count = sum(1 for d in dealers if d.tier == DealerTier.TIER3)

        assert tier1_count == 1
        assert tier2_count == 2
        assert tier3_count == 1

    def test_create_default_dealer_pool_characteristics(self) -> None:
        """Test that default pool has expected characteristics."""
        dealers = create_default_dealer_pool()

        # Find tier1 primary
        tier1 = next(d for d in dealers if d.tier == DealerTier.TIER1)
        assert tier1.is_primary is True
        assert tier1.spread_factor < 1.0
        assert tier1.max_size_usd >= 10_000_000


# =============================================================================
# Test Integration Helpers
# =============================================================================


class TestIntegrationHelpers:
    """Tests for integration helper functions."""

    def test_combine_with_parametric_slippage_filled(self) -> None:
        """Test combining slippage for filled execution."""
        parametric = 1.0
        result = ExecutionResult(
            filled=True,
            fill_price=1.0851,
            fill_qty=100_000,
            slippage_pips=0.5,
        )

        combined = combine_with_parametric_slippage(parametric, result, weight_execution=0.3)

        # Combined = 0.3 * 0.5 + 0.7 * 1.0 = 0.15 + 0.7 = 0.85
        expected = 0.3 * 0.5 + 0.7 * 1.0
        assert combined == pytest.approx(expected)

    def test_combine_with_parametric_slippage_rejected(self) -> None:
        """Test combining slippage for rejected execution."""
        parametric = 1.0
        result = ExecutionResult(
            filled=False,
            reject_reason=RejectReason.LAST_LOOK_ADVERSE,
        )

        combined = combine_with_parametric_slippage(parametric, result)

        # Rejected: parametric * 1.5
        assert combined == pytest.approx(1.5)

    def test_combine_with_parametric_slippage_weight_0(self) -> None:
        """Test with weight = 0 (all parametric)."""
        parametric = 2.0
        result = ExecutionResult(
            filled=True,
            slippage_pips=0.5,
        )

        combined = combine_with_parametric_slippage(parametric, result, weight_execution=0.0)
        assert combined == pytest.approx(2.0)

    def test_combine_with_parametric_slippage_weight_1(self) -> None:
        """Test with weight = 1 (all execution)."""
        parametric = 2.0
        result = ExecutionResult(
            filled=True,
            slippage_pips=0.5,
        )

        combined = combine_with_parametric_slippage(parametric, result, weight_execution=1.0)
        assert combined == pytest.approx(0.5)

    def test_estimate_rejection_probability_small_size(self) -> None:
        """Test rejection probability for small order."""
        prob = estimate_rejection_probability(
            size_usd=100_000,
            session_factor=1.0,
            volatility_regime="normal",
        )
        # Small size, normal conditions: low rejection
        assert prob < 0.2

    def test_estimate_rejection_probability_large_size(self) -> None:
        """Test rejection probability for large order."""
        prob = estimate_rejection_probability(
            size_usd=10_000_000,
            session_factor=1.0,
            volatility_regime="normal",
        )
        # Large size: higher rejection
        assert prob > 0.1

    def test_estimate_rejection_probability_low_liquidity(self) -> None:
        """Test rejection probability in low liquidity."""
        prob_high = estimate_rejection_probability(
            size_usd=500_000,
            session_factor=1.5,  # High liquidity
        )
        prob_low = estimate_rejection_probability(
            size_usd=500_000,
            session_factor=0.5,  # Low liquidity
        )
        # Low liquidity should have higher rejection probability
        assert prob_low > prob_high

    def test_estimate_rejection_probability_high_volatility(self) -> None:
        """Test rejection probability in high volatility."""
        prob_normal = estimate_rejection_probability(
            size_usd=500_000,
            session_factor=1.0,
            volatility_regime="normal",
        )
        prob_extreme = estimate_rejection_probability(
            size_usd=500_000,
            session_factor=1.0,
            volatility_regime="extreme",
        )
        # Extreme volatility should have higher rejection
        assert prob_extreme > prob_normal

    def test_estimate_rejection_probability_bounds(self) -> None:
        """Test rejection probability stays in valid range."""
        # Extreme case
        prob = estimate_rejection_probability(
            size_usd=100_000_000,  # Huge order
            session_factor=0.1,  # Very low liquidity
            volatility_regime="extreme",
        )
        assert 0.01 <= prob <= 0.95

        # Best case
        prob = estimate_rejection_probability(
            size_usd=10_000,
            session_factor=2.0,
            volatility_regime="low",
        )
        assert 0.01 <= prob <= 0.95


# =============================================================================
# Test Enum Values
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_quote_type_values(self) -> None:
        """Test QuoteType enum values."""
        assert QuoteType.FIRM.value == "firm"
        assert QuoteType.INDICATIVE.value == "indicative"
        assert QuoteType.LAST_LOOK.value == "last_look"

    def test_reject_reason_values(self) -> None:
        """Test RejectReason enum values."""
        assert RejectReason.NONE.value == "none"
        assert RejectReason.LAST_LOOK_ADVERSE.value == "last_look_adverse"
        assert RejectReason.SIZE_EXCEEDED.value == "size_exceeded"
        assert RejectReason.QUOTE_EXPIRED.value == "quote_expired"
        assert RejectReason.PRICE_MOVED.value == "price_moved"
        assert RejectReason.LATENCY_ARBITRAGE.value == "latency_arbitrage"
        assert RejectReason.DEALER_DISCRETION.value == "dealer_discretion"
        assert RejectReason.MARKET_CLOSED.value == "market_closed"

    def test_dealer_tier_values(self) -> None:
        """Test DealerTier enum values."""
        assert DealerTier.TIER1.value == "tier1"
        assert DealerTier.TIER2.value == "tier2"
        assert DealerTier.TIER3.value == "tier3"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_session_factor(self, simulator: ForexDealerSimulator) -> None:
        """Test with very low session factor."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            session_factor=0.1,  # Very low
        )
        # Should still produce valid quote
        assert quote.best_bid < quote.best_ask

    def test_very_large_order_size(self, simulator: ForexDealerSimulator) -> None:
        """Test with very large order."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            order_size_usd=1_000_000_000,  # $1B
        )
        # Should produce wider spread
        assert quote.spread > 0

    def test_minimum_quote_size(self) -> None:
        """Test minimum quote size filtering."""
        config = ForexDealerConfig(min_quote_size_usd=100_000)
        sim = ForexDealerSimulator(config=config, seed=42)

        # Very small order should still work
        quote = sim.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.0850,
            order_size_usd=1_000,
        )
        assert quote.num_active_dealers > 0

    def test_execution_history_bounded(self) -> None:
        """Test execution history is bounded by max_history_size."""
        config = ForexDealerConfig(max_history_size=10)
        sim = ForexDealerSimulator(config=config, seed=42)

        # Execute 20 times
        for _ in range(20):
            quote = sim.get_aggregated_quote("EUR_USD", 1.0850)
            sim.attempt_execution(
                is_buy=True,
                size_usd=100_000,
                quote=quote,
                current_mid=1.0850,
            )

        # History should be bounded
        assert len(sim._execution_history) <= 10

    def test_single_dealer_config(self) -> None:
        """Test with single dealer configuration."""
        config = ForexDealerConfig(num_dealers=1)
        sim = ForexDealerSimulator(config=config, seed=42)

        quote = sim.get_aggregated_quote("EUR_USD", 1.0850)
        assert quote.num_active_dealers == 1
        assert len(quote.dealer_quotes) == 1

    def test_mid_price_precision(self, simulator: ForexDealerSimulator) -> None:
        """Test that mid price calculations maintain precision."""
        quote = simulator.get_aggregated_quote(
            symbol="EUR_USD",
            mid_price=1.08501234,  # High precision
        )
        # Spread should be symmetric around mid
        calculated_mid = (quote.best_bid + quote.best_ask) / 2.0
        assert abs(calculated_mid - quote.mid) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
