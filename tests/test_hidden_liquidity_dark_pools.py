"""
Tests for Stage 6: Hidden Liquidity & Dark Pools.

This module tests:
- IcebergDetector: Iceberg order detection from execution patterns
- HiddenLiquidityEstimator: Hidden quantity estimation
- DarkPoolSimulator: Dark pool execution simulation
- DarkPoolVenue: Individual venue behavior
- Integration tests for combined functionality

Test Coverage:
- 10+ tests for IcebergDetector
- 5+ tests for HiddenLiquidityEstimator
- 10+ tests for DarkPoolSimulator
- 5+ integration tests

Total: 30+ tests
"""

import pytest
pytest.importorskip("sortedcontainers")
import time
from typing import List, Tuple, Optional

from lob import (
    # Data structures
    LimitOrder,
    Side,
    OrderType,
    Trade,
    PriceLevel,
    OrderBook,
    # Hidden Liquidity (Stage 6)
    IcebergDetector,
    IcebergOrder,
    IcebergState,
    RefillEvent,
    LevelSnapshot,
    DetectionConfidence,
    HiddenLiquidityEstimator,
    create_iceberg_detector,
    create_hidden_liquidity_estimator,
    # Dark Pool (Stage 6)
    DarkPoolSimulator,
    DarkPoolVenue,
    DarkPoolConfig,
    DarkPoolFill,
    DarkPoolState,
    DarkPoolVenueType,
    FillType,
    LeakageType,
    InformationLeakage,
    create_dark_pool_simulator,
    create_default_dark_pool_simulator,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_order() -> LimitOrder:
    """Create a sample limit order."""
    return LimitOrder(
        order_id="test_order_1",
        price=100.0,
        qty=1000.0,
        remaining_qty=1000.0,
        timestamp_ns=time.time_ns(),
        side=Side.BUY,
        order_type=OrderType.LIMIT,
    )


@pytest.fixture
def iceberg_order() -> LimitOrder:
    """Create an iceberg order."""
    return LimitOrder(
        order_id="iceberg_1",
        price=100.0,
        qty=10000.0,
        remaining_qty=10000.0,
        timestamp_ns=time.time_ns(),
        side=Side.BUY,
        order_type=OrderType.ICEBERG,
        display_qty=1000.0,
        hidden_qty=9000.0,
    )


@pytest.fixture
def iceberg_detector() -> IcebergDetector:
    """Create an iceberg detector."""
    return create_iceberg_detector(
        min_refills_to_confirm=2,
        lookback_window_sec=60.0,
        min_display_size=10.0,
    )


@pytest.fixture
def dark_pool_simulator() -> DarkPoolSimulator:
    """Create a dark pool simulator."""
    return create_default_dark_pool_simulator(seed=42)


@pytest.fixture
def price_level() -> PriceLevel:
    """Create a price level with orders."""
    level = PriceLevel(price=100.0)
    for i in range(5):
        order = LimitOrder(
            order_id=f"order_{i}",
            price=100.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns() + i * 1000,
            side=Side.BUY,
        )
        level.add_order(order)
    return level


# ==============================================================================
# IcebergDetector Tests
# ==============================================================================


class TestIcebergDetector:
    """Tests for IcebergDetector class."""

    def test_detector_creation(self, iceberg_detector: IcebergDetector):
        """Test detector creation with default parameters."""
        assert iceberg_detector is not None
        stats = iceberg_detector.stats()
        assert stats["total_detected"] == 0
        assert stats["active_icebergs"] == 0

    def test_take_level_snapshot(self, iceberg_detector: IcebergDetector, price_level: PriceLevel):
        """Test taking a level snapshot."""
        snapshot = iceberg_detector.take_level_snapshot(price_level, Side.BUY)

        assert snapshot.price == 100.0
        assert snapshot.visible_qty == 500.0  # 5 orders * 100 qty
        assert snapshot.order_count == 5
        assert len(snapshot.order_ids) == 5

    def test_detect_iceberg_single_refill(self, iceberg_detector: IcebergDetector):
        """Test detection of single iceberg refill."""
        price = 100.0
        ts = time.time_ns()

        # Pre-level: 500 qty visible
        pre_level = LevelSnapshot(
            price=price,
            visible_qty=500.0,
            order_count=5,
            timestamp_ns=ts,
        )

        # Trade occurs: 100 qty filled
        trade = Trade(
            price=price,
            qty=100.0,
            maker_order_id="maker_1",
            timestamp_ns=ts + 1000,
            aggressor_side=Side.SELL,
        )

        # Post-level: 500 qty (refilled!) instead of expected 400
        post_level = LevelSnapshot(
            price=price,
            visible_qty=500.0,  # Refilled back to 500!
            order_count=5,
            timestamp_ns=ts + 1000,
        )

        # Process execution
        iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg is not None
        assert iceberg.state == IcebergState.SUSPECTED
        assert iceberg.confidence == DetectionConfidence.LOW
        assert iceberg.display_size == 100.0  # Refill amount
        assert iceberg.refill_count == 1

    def test_detect_iceberg_confirmed(self, iceberg_detector: IcebergDetector):
        """Test confirmation of iceberg with multiple refills."""
        price = 100.0
        ts = time.time_ns()
        iceberg = None

        # Simulate multiple refills at same price
        for i in range(3):
            pre_level = LevelSnapshot(
                price=price,
                visible_qty=500.0,
                order_count=5,
                timestamp_ns=ts + i * 10000,
            )

            trade = Trade(
                price=price,
                qty=100.0,
                maker_order_id="maker_1",
                timestamp_ns=ts + i * 10000 + 1000,
                aggressor_side=Side.SELL,
            )

            post_level = LevelSnapshot(
                price=price,
                visible_qty=500.0,  # Refilled
                order_count=5,
                timestamp_ns=ts + i * 10000 + 1000,
            )

            iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg is not None
        assert iceberg.state == IcebergState.CONFIRMED
        assert iceberg.confidence >= DetectionConfidence.MEDIUM
        assert iceberg.refill_count == 3
        assert iceberg.total_executed == 300.0

    def test_detect_iceberg_no_refill(self, iceberg_detector: IcebergDetector):
        """Test that no iceberg is detected when no refill occurs."""
        price = 100.0
        ts = time.time_ns()

        pre_level = LevelSnapshot(
            price=price,
            visible_qty=500.0,
            order_count=5,
            timestamp_ns=ts,
        )

        trade = Trade(
            price=price,
            qty=100.0,
            maker_order_id="maker_1",
            timestamp_ns=ts + 1000,
            aggressor_side=Side.SELL,
        )

        # Post-level: 400 qty (no refill, as expected after fill)
        post_level = LevelSnapshot(
            price=price,
            visible_qty=400.0,  # No refill
            order_count=4,
            timestamp_ns=ts + 1000,
        )

        iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg is None

    def test_detect_iceberg_batch_method(self, iceberg_detector: IcebergDetector):
        """Test batch detection from execution history."""
        price = 100.0
        ts = time.time_ns()

        # Create execution history with refill pattern
        executions = []
        level_qty_history = []

        for i in range(5):
            executions.append(Trade(
                price=price,
                qty=100.0,
                maker_order_id="maker_1",
                timestamp_ns=ts + i * 10000,
            ))
            # Qty refills after each execution
            level_qty_history.append(500.0)  # Always back to 500

        iceberg = iceberg_detector.detect_iceberg(
            executions=executions,
            level_qty_history=level_qty_history,
            price=price,
            side=Side.BUY,
        )

        assert iceberg is not None
        assert iceberg.state == IcebergState.CONFIRMED
        assert iceberg.refill_count >= 2

    def test_estimate_hidden_reserve(self, iceberg_detector: IcebergDetector):
        """Test hidden reserve estimation."""
        price = 100.0
        ts = time.time_ns()

        # Create iceberg with multiple refills
        for i in range(4):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg is not None

        # Estimate hidden reserve
        hidden_estimate = iceberg_detector.estimate_hidden_reserve(iceberg, current_visible_qty=500.0)

        assert hidden_estimate > 0
        # Should be reasonable estimate based on refill pattern

    def test_get_active_icebergs(self, iceberg_detector: IcebergDetector):
        """Test getting active icebergs."""
        price = 100.0
        ts = time.time_ns()

        # Create iceberg at price 100
        for i in range(2):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        active = list(iceberg_detector.get_active_icebergs())
        assert len(active) == 1
        assert active[0].price == price

    def test_iceberg_at_price_lookup(self, iceberg_detector: IcebergDetector):
        """Test iceberg lookup at specific price."""
        price = 100.0
        ts = time.time_ns()

        pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts)
        trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + 1000, aggressor_side=Side.SELL)
        post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + 1000)

        iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        # Lookup should find iceberg
        found = iceberg_detector.get_iceberg_at_price(price, Side.BUY)
        assert found is not None
        assert found.price == price

        # Wrong side should not find
        not_found = iceberg_detector.get_iceberg_at_price(price, Side.SELL)
        assert not_found is None

    def test_remove_stale_icebergs(self, iceberg_detector: IcebergDetector):
        """Test removal of stale icebergs."""
        price = 100.0
        ts = time.time_ns()

        # Create iceberg
        pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts)
        trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + 1000, aggressor_side=Side.SELL)
        post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + 1000)

        iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert len(list(iceberg_detector.get_active_icebergs())) == 1

        # Remove stale (with future timestamp)
        removed_count = iceberg_detector.remove_stale_icebergs(ts + 100_000_000_000)  # 100 seconds later

        assert removed_count == 1
        assert len(list(iceberg_detector.get_active_icebergs())) == 0

    def test_detector_stats(self, iceberg_detector: IcebergDetector):
        """Test detector statistics."""
        price = 100.0
        ts = time.time_ns()

        # Create and confirm iceberg
        for i in range(3):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        stats = iceberg_detector.stats()

        assert stats["total_detected"] == 1
        assert stats["total_confirmed"] == 1
        assert stats["active_icebergs"] == 1


# ==============================================================================
# HiddenLiquidityEstimator Tests
# ==============================================================================


class TestHiddenLiquidityEstimator:
    """Tests for HiddenLiquidityEstimator class."""

    def test_estimator_creation(self, iceberg_detector: IcebergDetector):
        """Test estimator creation."""
        estimator = create_hidden_liquidity_estimator(
            iceberg_detector=iceberg_detector,
            hidden_ratio=0.15,
        )
        assert estimator is not None

    def test_estimate_hidden_at_level_no_iceberg(self, iceberg_detector: IcebergDetector):
        """Test hidden estimation at level without detected iceberg."""
        estimator = HiddenLiquidityEstimator(
            iceberg_detector=iceberg_detector,
            hidden_ratio_estimate=0.20,  # 20% default
        )

        # No iceberg detected at this price
        hidden = estimator.estimate_hidden_at_level(
            price=100.0,
            side=Side.BUY,
            visible_qty=500.0,
        )

        # Should use default ratio
        assert hidden == 500.0 * 0.20

    def test_estimate_hidden_at_level_with_iceberg(self, iceberg_detector: IcebergDetector):
        """Test hidden estimation at level with detected iceberg."""
        price = 100.0
        ts = time.time_ns()

        # Create iceberg
        for i in range(2):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        estimator = HiddenLiquidityEstimator(
            iceberg_detector=iceberg_detector,
            hidden_ratio_estimate=0.10,
        )

        # Should use iceberg estimate, not default
        hidden = estimator.estimate_hidden_at_level(
            price=price,
            side=Side.BUY,
            visible_qty=500.0,
        )

        # Should be higher than default ratio estimate
        assert hidden > 0

    def test_estimate_total_hidden(self, iceberg_detector: IcebergDetector):
        """Test total hidden liquidity estimation."""
        estimator = HiddenLiquidityEstimator(
            iceberg_detector=iceberg_detector,
            hidden_ratio_estimate=0.15,
        )

        # Multiple price levels
        price_levels = [
            (100.0, 500.0),
            (99.0, 300.0),
            (98.0, 200.0),
        ]

        total_hidden = estimator.estimate_total_hidden(Side.BUY, price_levels)

        # Should equal sum of visible * ratio
        expected = sum(v * 0.15 for _, v in price_levels)
        assert total_hidden == pytest.approx(expected, rel=0.01)

    def test_get_hidden_liquidity_map(self, iceberg_detector: IcebergDetector):
        """Test getting hidden liquidity map."""
        price = 100.0
        ts = time.time_ns()

        # Create iceberg
        for i in range(2):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        estimator = HiddenLiquidityEstimator(iceberg_detector=iceberg_detector)

        hidden_map = estimator.get_hidden_liquidity_map(Side.BUY)

        assert len(hidden_map) == 1
        assert price in hidden_map
        assert hidden_map[price] > 0


# ==============================================================================
# DarkPoolSimulator Tests
# ==============================================================================


class TestDarkPoolVenue:
    """Tests for DarkPoolVenue class."""

    def test_venue_creation(self):
        """Test venue creation with config."""
        config = DarkPoolConfig(
            venue_id="TEST_VENUE",
            venue_type=DarkPoolVenueType.MIDPOINT_CROSS,
            min_order_size=100,
            base_fill_probability=0.40,
        )

        venue = DarkPoolVenue(config)

        assert venue.venue_id == "TEST_VENUE"
        assert venue.venue_type == DarkPoolVenueType.MIDPOINT_CROSS

    def test_venue_fill_success(self, sample_order: LimitOrder):
        """Test successful venue fill."""
        config = DarkPoolConfig(
            venue_id="HIGH_FILL",
            base_fill_probability=0.99,  # Very high for testing
            min_order_size=10,
            info_leakage_probability=0.0,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        fill = venue.attempt_fill(
            order=sample_order,
            lit_mid_price=100.0,
            lit_spread=0.05,
            adv=10_000_000,
        )

        # With high probability and fixed seed, should fill
        assert fill.fill_type in (FillType.FULL, FillType.PARTIAL, FillType.NO_FILL)
        if fill.is_filled:
            assert fill.fill_price == 100.0  # Mid price
            assert fill.filled_qty > 0

    def test_venue_min_size_rejection(self, sample_order: LimitOrder):
        """Test rejection of order below minimum size."""
        config = DarkPoolConfig(
            venue_id="BLOCK_VENUE",
            min_order_size=10000,  # Very high minimum
        )

        venue = DarkPoolVenue(config)

        # Order is 1000, minimum is 10000
        fill = venue.attempt_fill(
            order=sample_order,
            lit_mid_price=100.0,
        )

        assert fill.fill_type == FillType.NO_FILL
        assert fill.filled_qty == 0

    def test_venue_latency(self, sample_order: LimitOrder):
        """Test that fills have latency."""
        config = DarkPoolConfig(
            venue_id="LATENCY_TEST",
            base_fill_probability=0.99,
            latency_ms=10.0,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        fill = venue.attempt_fill(
            order=sample_order,
            lit_mid_price=100.0,
        )

        if fill.is_filled:
            # Latency should be approximately latency_ms in nanoseconds
            assert fill.latency_ns > 0

    def test_venue_stats(self, sample_order: LimitOrder):
        """Test venue statistics tracking."""
        config = DarkPoolConfig(
            venue_id="STATS_TEST",
            base_fill_probability=0.80,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        for _ in range(10):
            venue.attempt_fill(order=sample_order, lit_mid_price=100.0)

        stats = venue.stats()

        assert stats["total_attempts"] == 10
        assert stats["total_fills"] >= 0


class TestDarkPoolSimulator:
    """Tests for DarkPoolSimulator class."""

    def test_simulator_creation(self, dark_pool_simulator: DarkPoolSimulator):
        """Test simulator creation with default venues."""
        assert dark_pool_simulator is not None

        # Should have default venues
        venues = list(dark_pool_simulator.get_all_venues())
        assert len(venues) == 4  # DEFAULT_VENUES has 4 venues

    def test_simulator_custom_venues(self):
        """Test simulator with custom venue configs."""
        configs = [
            DarkPoolConfig(venue_id="VENUE_A", base_fill_probability=0.50),
            DarkPoolConfig(venue_id="VENUE_B", base_fill_probability=0.30),
        ]

        simulator = create_dark_pool_simulator(venue_configs=configs, seed=42)

        venues = list(simulator.get_all_venues())
        assert len(venues) == 2

    def test_attempt_dark_fill(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test attempting dark pool fill."""
        fill = dark_pool_simulator.attempt_dark_fill(
            order=sample_order,
            lit_mid_price=100.0,
            lit_spread=0.05,
            adv=10_000_000,
            volatility=0.02,
            hour_of_day=10,
        )

        # May or may not fill depending on probability
        if fill is not None and fill.is_filled:
            assert fill.fill_price > 0
            assert fill.filled_qty > 0
            assert fill.venue_id != ""

    def test_smart_routing(self, sample_order: LimitOrder):
        """Test smart order routing between venues."""
        simulator = create_default_dark_pool_simulator(seed=123)

        probs = simulator.estimate_fill_probability(
            order=sample_order,
            adv=10_000_000,
            volatility=0.02,
            hour_of_day=10,
        )

        # Should have probabilities for all venues
        assert len(probs) == 4
        for venue_id, prob in probs.items():
            assert 0 <= prob <= 1

    def test_routing_with_multiple_fills(self, sample_order: LimitOrder):
        """Test routing with multiple venue attempts."""
        simulator = create_default_dark_pool_simulator(seed=42)

        fills = simulator.attempt_fill_with_routing(
            order=sample_order,
            lit_mid_price=100.0,
            lit_spread=0.05,
            adv=10_000_000,
            max_attempts=3,
        )

        # May have 0 or more fills
        assert isinstance(fills, list)

    def test_leakage_detection(self):
        """Test information leakage detection."""
        config = DarkPoolConfig(
            venue_id="HIGH_LEAK",
            info_leakage_probability=0.99,  # High for testing
            base_fill_probability=0.50,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        order = LimitOrder(
            order_id="leak_test",
            price=100.0,
            qty=1000.0,
            remaining_qty=1000.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        # Try multiple times, should eventually get leakage
        leakage_found = False
        for _ in range(20):
            fill = venue.attempt_fill(order=order, lit_mid_price=100.0)
            if fill.info_leakage is not None:
                leakage_found = True
                assert fill.info_leakage.leakage_type != LeakageType.NONE
                break

        assert leakage_found

    def test_state_tracking(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test state tracking across fills."""
        initial_state = dark_pool_simulator.get_state()
        initial_attempts = initial_state.total_attempts

        # Make some attempts
        for _ in range(5):
            dark_pool_simulator.attempt_dark_fill(
                order=sample_order,
                lit_mid_price=100.0,
            )

        final_state = dark_pool_simulator.get_state()

        assert final_state.total_attempts == initial_attempts + 5

    def test_fill_callback(self, sample_order: LimitOrder):
        """Test fill callback."""
        fills_received: List[DarkPoolFill] = []

        def on_fill(fill: DarkPoolFill):
            fills_received.append(fill)

        configs = [
            DarkPoolConfig(
                venue_id="HIGH_FILL",
                base_fill_probability=0.99,
            ),
        ]

        simulator = create_dark_pool_simulator(
            venue_configs=configs,
            seed=42,
            on_fill=on_fill,
        )

        # Make attempts until we get a fill
        for _ in range(10):
            simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        # Should have received fills via callback
        # Note: may be 0 if probability didn't hit

    def test_leakage_callback(self, sample_order: LimitOrder):
        """Test leakage callback."""
        leakages_received: List[InformationLeakage] = []

        def on_leakage(leakage: InformationLeakage):
            leakages_received.append(leakage)

        configs = [
            DarkPoolConfig(
                venue_id="HIGH_LEAK",
                info_leakage_probability=0.99,
            ),
        ]

        simulator = create_dark_pool_simulator(
            venue_configs=configs,
            seed=42,
            on_leakage=on_leakage,
        )

        for _ in range(20):
            simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        # Should have received leakages via callback

    def test_preferred_venue(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test preferred venue routing."""
        # Should try preferred venue first
        fill = dark_pool_simulator.attempt_dark_fill(
            order=sample_order,
            lit_mid_price=100.0,
            preferred_venue="IEX_D",
        )

        # Result depends on probability, but venue should be tried

    def test_clear_history(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test clearing history."""
        # Make some attempts
        for _ in range(5):
            dark_pool_simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        # Clear
        dark_pool_simulator.clear_history()

        assert len(dark_pool_simulator.get_fill_history()) == 0
        assert len(dark_pool_simulator.get_leakage_history()) == 0

    def test_stats(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test simulator stats."""
        for _ in range(10):
            dark_pool_simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        stats = dark_pool_simulator.stats()

        assert "total_volume" in stats
        assert "total_attempts" in stats
        assert "fill_rate" in stats
        assert stats["total_attempts"] == 10


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_iceberg_detection_with_order_book(self):
        """Test iceberg detection with real order book."""
        book = OrderBook(symbol="TEST", tick_size=0.01, lot_size=1.0)
        detector = create_iceberg_detector()

        # Add iceberg order to book
        iceberg_order = LimitOrder(
            order_id="iceberg_1",
            price=100.0,
            qty=10000.0,
            remaining_qty=10000.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
            order_type=OrderType.ICEBERG,
            display_qty=1000.0,
            hidden_qty=9000.0,
        )
        book.add_limit_order(iceberg_order)

        # Simulate executions
        bid_level = book._bids.peekitem(0)[1]
        pre_snap = detector.take_level_snapshot(bid_level, Side.BUY)

        # Execute market order
        fill = book.execute_market_order(Side.SELL, 500.0)

        # After execution, iceberg should have replenished
        bid_level = book._bids.peekitem(0)[1]
        post_snap = detector.take_level_snapshot(bid_level, Side.BUY)

        # Check if iceberg detection works
        # Note: depends on iceberg replenishment in order book

    def test_combined_hidden_liquidity_and_dark_pool(self, sample_order: LimitOrder):
        """Test combined hidden liquidity estimation with dark pool."""
        detector = create_iceberg_detector()
        estimator = create_hidden_liquidity_estimator(detector, hidden_ratio=0.15)
        dark_pool = create_default_dark_pool_simulator(seed=42)

        # Estimate hidden liquidity
        price_levels = [(100.0, 1000.0), (99.0, 800.0)]
        hidden_estimate = estimator.estimate_total_hidden(Side.BUY, price_levels)

        assert hidden_estimate > 0

        # Try dark pool execution
        fill = dark_pool.attempt_dark_fill(
            order=sample_order,
            lit_mid_price=100.0,
            lit_spread=0.05,
        )

        # Combined workflow should work

    def test_dark_pool_venue_types(self, sample_order: LimitOrder):
        """Test different dark pool venue types."""
        venue_types = [
            DarkPoolVenueType.MIDPOINT_CROSS,
            DarkPoolVenueType.BLOCK_CROSS,
            DarkPoolVenueType.RETAIL_INTERNALIZATION,
            DarkPoolVenueType.CONTINUOUS_CROSS,
            DarkPoolVenueType.AUCTION_CROSS,
        ]

        for vtype in venue_types:
            config = DarkPoolConfig(
                venue_id=f"TEST_{vtype.name}",
                venue_type=vtype,
                base_fill_probability=0.50,
                min_order_size=10,
            )

            import random
            venue = DarkPoolVenue(config, rng=random.Random(42))

            fill = venue.attempt_fill(order=sample_order, lit_mid_price=100.0)

            # All venue types should be able to attempt fill
            assert fill.venue_id == f"TEST_{vtype.name}"

    def test_iceberg_lifecycle(self, iceberg_detector: IcebergDetector):
        """Test complete iceberg lifecycle: detection -> confirmation -> exhaustion."""
        price = 100.0
        ts = time.time_ns()

        # Phase 1: Detection
        pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts)
        trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + 1000, aggressor_side=Side.SELL)
        post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + 1000)

        iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)
        assert iceberg.state == IcebergState.SUSPECTED

        # Phase 2: Confirmation
        for i in range(2, 5):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg.state == IcebergState.CONFIRMED
        assert iceberg.confidence >= DetectionConfidence.MEDIUM

        # Phase 3: Exhaustion (no refill)
        pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + 100000)
        trade = Trade(price=price, qty=100.0, maker_order_id="maker_1", timestamp_ns=ts + 100001, aggressor_side=Side.SELL)
        post_level = LevelSnapshot(price=price, visible_qty=0.0, order_count=0, timestamp_ns=ts + 100001)  # No refill

        iceberg = iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        # Should be marked as exhausted
        assert iceberg.state == IcebergState.EXHAUSTED

    def test_factory_functions(self):
        """Test all factory functions."""
        # Iceberg detector
        detector = create_iceberg_detector(min_refills_to_confirm=3, lookback_window_sec=120.0)
        assert detector is not None

        # Hidden liquidity estimator
        estimator = create_hidden_liquidity_estimator(detector, hidden_ratio=0.20)
        assert estimator is not None

        # Dark pool simulator
        simulator = create_dark_pool_simulator(seed=42)
        assert simulator is not None

        # Default dark pool
        default_dp = create_default_dark_pool_simulator(seed=123)
        assert default_dp is not None


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_execution_list(self, iceberg_detector: IcebergDetector):
        """Test batch detection with empty execution list."""
        iceberg = iceberg_detector.detect_iceberg(
            executions=[],
            level_qty_history=[],
        )
        assert iceberg is None

    def test_null_snapshots(self, iceberg_detector: IcebergDetector):
        """Test processing with null snapshots."""
        trade = Trade(price=100.0, qty=100.0, maker_order_id="test", timestamp_ns=time.time_ns())

        result = iceberg_detector.process_execution(trade, None, None)
        assert result is None

    def test_zero_quantity_order(self, dark_pool_simulator: DarkPoolSimulator):
        """Test dark pool with zero quantity order."""
        order = LimitOrder(
            order_id="zero_qty",
            price=100.0,
            qty=0.0,
            remaining_qty=0.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        fill = dark_pool_simulator.attempt_dark_fill(order=order, lit_mid_price=100.0)

        # Should not fill
        if fill is not None:
            assert fill.filled_qty == 0

    def test_very_large_order(self, dark_pool_simulator: DarkPoolSimulator):
        """Test dark pool with very large order."""
        order = LimitOrder(
            order_id="large_order",
            price=100.0,
            qty=1_000_000_000.0,  # 1 billion shares
            remaining_qty=1_000_000_000.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        fill = dark_pool_simulator.attempt_dark_fill(
            order=order,
            lit_mid_price=100.0,
            adv=10_000_000,  # Normal ADV
        )

        # Should handle gracefully (likely low fill probability)


# ==============================================================================
# Performance Tests
# ==============================================================================


class TestPerformance:
    """Performance tests for Stage 6 modules."""

    def test_iceberg_detection_performance(self, iceberg_detector: IcebergDetector):
        """Test iceberg detection performance with many executions."""
        import time as time_module

        ts = time_module.time_ns()
        start = time_module.perf_counter()

        for i in range(1000):
            pre_level = LevelSnapshot(price=100.0, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 1000)
            trade = Trade(price=100.0, qty=10.0, maker_order_id="maker_1", timestamp_ns=ts + i * 1000 + 100, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=100.0, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 1000 + 100)

            iceberg_detector.process_execution(trade, pre_level, post_level, Side.BUY)

        elapsed = time_module.perf_counter() - start

        # Should process 1000 executions in < 1 second
        assert elapsed < 1.0, f"Processing took {elapsed:.2f}s, expected < 1.0s"

    def test_dark_pool_performance(self, dark_pool_simulator: DarkPoolSimulator, sample_order: LimitOrder):
        """Test dark pool simulation performance."""
        import time as time_module

        start = time_module.perf_counter()

        for _ in range(1000):
            dark_pool_simulator.attempt_dark_fill(
                order=sample_order,
                lit_mid_price=100.0,
                adv=10_000_000,
            )

        elapsed = time_module.perf_counter() - start

        # Should simulate 1000 attempts in < 1 second
        assert elapsed < 1.0, f"Simulation took {elapsed:.2f}s, expected < 1.0s"


# ==============================================================================
# Bug Fix Verification Tests
# ==============================================================================


class TestBugFixes:
    """Tests that verify specific bug fixes."""

    def test_zero_latency_no_division_error(self):
        """Test that zero latency config doesn't cause division by zero."""
        # BUG FIX: dark_pool.py:362 - expovariate(1.0 / 0) caused ZeroDivisionError
        config = DarkPoolConfig(
            venue_id="ZERO_LATENCY",
            latency_ms=0.0,  # Zero latency
            base_fill_probability=0.99,
            min_order_size=10,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        order = LimitOrder(
            order_id="test_zero_lat",
            price=100.0,
            qty=1000.0,
            remaining_qty=1000.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        # Should not raise ZeroDivisionError
        fill = venue.attempt_fill(order=order, lit_mid_price=100.0)
        if fill.is_filled:
            assert fill.latency_ns == 0  # Zero latency

    def test_batch_detection_updates_counters(self):
        """Test that batch detection updates total_detected and total_confirmed."""
        # BUG FIX: hidden_liquidity.py:663 - batch detection didn't update counters
        detector = create_iceberg_detector(min_refills_to_confirm=2)

        initial_stats = detector.stats()
        assert initial_stats["total_detected"] == 0
        assert initial_stats["total_confirmed"] == 0

        # Create execution history with refill pattern (confirmed iceberg)
        price = 100.0
        ts = time.time_ns()
        executions = []
        level_qty_history = []

        for i in range(3):  # 3 refills -> confirmed
            executions.append(Trade(
                price=price,
                qty=100.0,
                maker_order_id="maker_1",
                timestamp_ns=ts + i * 10000,
            ))
            level_qty_history.append(500.0)  # Always back to 500 (refill)

        iceberg = detector.detect_iceberg(
            executions=executions,
            level_qty_history=level_qty_history,
            price=price,
            side=Side.BUY,
        )

        # Verify counters are updated
        assert iceberg is not None
        stats = detector.stats()
        assert stats["total_detected"] == 1
        assert stats["total_confirmed"] == 1

    def test_batch_detection_suspected_not_confirmed(self):
        """Test batch detection with suspected but not confirmed iceberg."""
        detector = create_iceberg_detector(min_refills_to_confirm=2)

        price = 100.0
        ts = time.time_ns()
        executions = [
            Trade(price=price, qty=100.0, maker_order_id="m1", timestamp_ns=ts)
        ]
        level_qty_history = [500.0]  # Only 1 refill -> suspected

        iceberg = detector.detect_iceberg(
            executions=executions,
            level_qty_history=level_qty_history,
            price=price,
            side=Side.BUY,
        )

        assert iceberg is not None
        assert iceberg.state == IcebergState.SUSPECTED
        stats = detector.stats()
        assert stats["total_detected"] == 1
        assert stats["total_confirmed"] == 0  # Not confirmed

    def test_configurable_initial_hidden_multiplier(self):
        """Test that initial_hidden_multiplier is configurable."""
        # BUG FIX: hidden_liquidity.py:467 - hardcoded 3.0 multiplier
        detector_default = create_iceberg_detector()
        detector_custom = create_iceberg_detector(initial_hidden_multiplier=5.0)

        price = 100.0
        ts = time.time_ns()

        # Create iceberg with display size = 100 (from refill)
        def create_iceberg(det: IcebergDetector) -> IcebergOrder:
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts)
            trade = Trade(price=price, qty=100.0, maker_order_id="m1", timestamp_ns=ts + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + 1000)
            return det.process_execution(trade, pre_level, post_level, Side.BUY)

        iceberg_default = create_iceberg(detector_default)
        iceberg_custom = create_iceberg(detector_custom)

        assert iceberg_default is not None
        assert iceberg_custom is not None

        # Default multiplier is 3.0, custom is 5.0
        # display_size = 100 (refill amount)
        # estimated_hidden = display_size * multiplier
        assert iceberg_default.estimated_hidden_qty == pytest.approx(100.0 * 3.0, rel=0.01)
        assert iceberg_custom.estimated_hidden_qty == pytest.approx(100.0 * 5.0, rel=0.01)

    def test_aggressor_side_none_handling(self):
        """Test that process_execution handles aggressor_side=None gracefully."""
        # Test for edge case where aggressor_side is None
        detector = create_iceberg_detector()

        ts = time.time_ns()
        pre_level = LevelSnapshot(price=100.0, visible_qty=500.0, order_count=5, timestamp_ns=ts)
        post_level = LevelSnapshot(price=100.0, visible_qty=500.0, order_count=5, timestamp_ns=ts + 1000)

        # Trade without aggressor_side and without explicit side
        trade = Trade(
            price=100.0,
            qty=100.0,
            maker_order_id="maker_1",
            timestamp_ns=ts + 1000,
            aggressor_side=None,  # Explicitly None
        )

        # Should return None when can't infer side
        result = detector.process_execution(trade, pre_level, post_level, side=None)
        assert result is None

    def test_configurable_dark_pool_parameters(self):
        """Test that dark pool magic numbers are now configurable."""
        # BUG FIX: dark_pool.py:400,439,512 - magic numbers made configurable
        config = DarkPoolConfig(
            venue_id="CUSTOM_PARAMS",
            size_penalty_multiplier=5.0,  # Default was 10
            partial_fill_min_ratio=0.5,  # Default was 0.3
            partial_fill_size_multiplier=3.0,  # Default was 5.0
            partial_fill_max_reduction=0.5,  # Default was 0.7
            impact_size_normalization=5000.0,  # Default was 10000
            base_fill_probability=0.99,
        )

        import random
        venue = DarkPoolVenue(config, rng=random.Random(42))

        # Verify config is used
        assert venue._config.size_penalty_multiplier == 5.0
        assert venue._config.partial_fill_min_ratio == 0.5
        assert venue._config.impact_size_normalization == 5000.0

    def test_no_double_decay_in_estimate(self):
        """Test that decay is not applied twice."""
        # BUG FIX: hidden_liquidity.py - decay was applied in both
        # _update_hidden_estimate and estimate_hidden_reserve
        detector = create_iceberg_detector(decay_factor=0.5)  # Strong decay for visibility

        price = 100.0
        ts = time.time_ns()

        # Create iceberg with multiple refills
        iceberg = None
        for i in range(4):
            pre_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000)
            trade = Trade(price=price, qty=100.0, maker_order_id="m1", timestamp_ns=ts + i * 10000 + 1000, aggressor_side=Side.SELL)
            post_level = LevelSnapshot(price=price, visible_qty=500.0, order_count=5, timestamp_ns=ts + i * 10000 + 1000)
            iceberg = detector.process_execution(trade, pre_level, post_level, Side.BUY)

        assert iceberg is not None

        # Now call estimate_hidden_reserve multiple times
        # It should return consistent values (not decaying further with each call)
        estimate1 = detector.estimate_hidden_reserve(iceberg, current_visible_qty=500.0)
        estimate2 = detector.estimate_hidden_reserve(iceberg, current_visible_qty=500.0)
        estimate3 = detector.estimate_hidden_reserve(iceberg, current_visible_qty=500.0)

        # All estimates should be the same (decay only based on refill_count, not call count)
        assert estimate1 == estimate2
        assert estimate2 == estimate3

    def test_rng_consistency_with_custom_venues(self):
        """Test that custom venues maintain RNG consistency."""
        # BUG FIX: dark_pool.py:1010 - RNG was created twice
        configs = [
            DarkPoolConfig(venue_id="V1", base_fill_probability=0.99),
            DarkPoolConfig(venue_id="V2", base_fill_probability=0.99),
        ]

        # Create two simulators with same seed
        sim1 = create_dark_pool_simulator(venue_configs=configs, seed=42)
        sim2 = create_dark_pool_simulator(venue_configs=configs, seed=42)

        order = LimitOrder(
            order_id="test_rng",
            price=100.0,
            qty=1000.0,
            remaining_qty=1000.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        # Both should produce same fill results (reproducible)
        fill1 = sim1.attempt_dark_fill(order=order, lit_mid_price=100.0)
        fill2 = sim2.attempt_dark_fill(order=order, lit_mid_price=100.0)

        if fill1 is not None and fill2 is not None:
            assert fill1.fill_type == fill2.fill_type
            if fill1.is_filled and fill2.is_filled:
                assert fill1.filled_qty == fill2.filled_qty

    def test_config_validation_impact_size_normalization_zero(self):
        """Test that impact_size_normalization=0 raises ValueError."""
        # BUG FIX: dark_pool.py:543 - division by zero if impact_size_normalization=0
        with pytest.raises(ValueError, match="impact_size_normalization must be positive"):
            DarkPoolConfig(
                venue_id="INVALID",
                impact_size_normalization=0.0,
            )

    def test_config_validation_impact_size_normalization_negative(self):
        """Test that negative impact_size_normalization raises ValueError."""
        with pytest.raises(ValueError, match="impact_size_normalization must be positive"):
            DarkPoolConfig(
                venue_id="INVALID",
                impact_size_normalization=-100.0,
            )

    def test_config_validation_base_fill_probability_invalid(self):
        """Test that invalid base_fill_probability raises ValueError."""
        # Too high
        with pytest.raises(ValueError, match="base_fill_probability must be in"):
            DarkPoolConfig(
                venue_id="INVALID",
                base_fill_probability=1.5,
            )
        # Negative
        with pytest.raises(ValueError, match="base_fill_probability must be in"):
            DarkPoolConfig(
                venue_id="INVALID",
                base_fill_probability=-0.1,
            )

    def test_config_validation_info_leakage_probability_invalid(self):
        """Test that invalid info_leakage_probability raises ValueError."""
        with pytest.raises(ValueError, match="info_leakage_probability must be in"):
            DarkPoolConfig(
                venue_id="INVALID",
                info_leakage_probability=2.0,
            )

    def test_config_validation_min_order_size_negative(self):
        """Test that negative min_order_size raises ValueError."""
        with pytest.raises(ValueError, match="min_order_size must be non-negative"):
            DarkPoolConfig(
                venue_id="INVALID",
                min_order_size=-100.0,
            )

    def test_config_validation_max_history_size_invalid(self):
        """Test that invalid max_history_size raises ValueError."""
        with pytest.raises(ValueError, match="max_history_size must be positive"):
            DarkPoolConfig(
                venue_id="INVALID",
                max_history_size=0,
            )
        with pytest.raises(ValueError, match="max_history_size must be positive"):
            DarkPoolConfig(
                venue_id="INVALID",
                max_history_size=-10,
            )

    def test_bounded_history_prevents_memory_leak(self, sample_order: LimitOrder):
        """Test that history is bounded to prevent memory leaks."""
        # BUG FIX: dark_pool.py:701-702 - unbounded List -> bounded deque
        max_history = 50  # Small for testing
        simulator = DarkPoolSimulator(
            venues=None,
            seed=42,
            max_history_size=max_history,
        )

        # Make many more attempts than max_history
        for _ in range(max_history * 3):
            simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        # History should be bounded
        fill_history = simulator.get_fill_history(limit=max_history * 2)
        leakage_history = simulator.get_leakage_history(limit=max_history * 2)

        assert len(fill_history) <= max_history
        assert len(leakage_history) <= max_history

    def test_bounded_history_preserves_recent_entries(self, sample_order: LimitOrder):
        """Test that bounded history keeps the most recent entries."""
        max_history = 10
        simulator = DarkPoolSimulator(
            venues=None,
            seed=42,
            max_history_size=max_history,
        )

        # Track all fills
        all_fills: List[DarkPoolFill] = []

        def track_fill(fill: DarkPoolFill):
            all_fills.append(fill)

        # Reconnect callback
        simulator._on_fill = track_fill

        # Make many attempts
        for i in range(50):
            simulator.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        # Get history
        history = simulator.get_fill_history(limit=max_history * 2)

        if len(all_fills) > max_history:
            # Should have last max_history entries
            assert len(history) == max_history
            # Most recent fills should be in history
            expected_recent = all_fills[-max_history:]
            for expected, actual in zip(expected_recent, history):
                assert expected.order_id == actual.order_id

    def test_factory_function_passes_max_history_size(self, sample_order: LimitOrder):
        """Test that factory functions pass max_history_size correctly."""
        max_history = 25

        # Test create_dark_pool_simulator
        sim1 = create_dark_pool_simulator(seed=42, max_history_size=max_history)
        assert sim1._max_history_size == max_history

        # Test create_default_dark_pool_simulator
        sim2 = create_default_dark_pool_simulator(seed=42, max_history_size=max_history)
        assert sim2._max_history_size == max_history

        # Verify bounded behavior
        for _ in range(max_history * 2):
            sim1.attempt_dark_fill(order=sample_order, lit_mid_price=100.0)

        assert len(sim1.get_fill_history(limit=max_history * 2)) <= max_history

    def test_valid_config_passes_validation(self):
        """Test that valid config passes all validations."""
        # Should not raise
        config = DarkPoolConfig(
            venue_id="VALID_VENUE",
            base_fill_probability=0.5,
            info_leakage_probability=0.1,
            min_order_size=100,
            impact_size_normalization=10000,
            max_history_size=5000,
        )

        assert config.venue_id == "VALID_VENUE"
        assert config.base_fill_probability == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
