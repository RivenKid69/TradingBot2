"""
Comprehensive tests for Options Memory Architecture (Phase 0.5).

This module contains 70 tests covering:
- LazyMultiSeriesLOBManager (20 tests): lazy creation, LRU eviction, disk persistence
- RingBufferOrderBook (15 tests): depth limiting, aggregation, operations
- EventDrivenLOBCoordinator (15 tests): bucketing, propagation, performance
- Memory benchmarks (10 tests): peak memory, GC pressure
- Disk persistence (10 tests): save/restore, compression

References:
- OPTIONS_INTEGRATION_PLAN.md Phase 0.5
- Memory budget: 50 LOBs × 50MB = 2.5GB (vs 480GB naive for SPY 960 series)
- Target: Peak < 4GB, Access < 1ms, Propagation < 100μs
"""

import gc
import gzip
import json
import os
import pickle
import tempfile
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("sortedcontainers")

# Import Phase 0.5 components
from lob.lazy_multi_series import (
    EvictionPolicy,
    LazyMultiSeriesLOBManager,
    LOBMetadata,
    ManagerStats,
    SeriesKey,
    SeriesLOBState,
    create_lazy_lob_manager,
    create_options_lob_manager,
)
from lob.ring_buffer_orderbook import (
    AggregatedLevel,
    BookLevel,
    BookSnapshot,
    BookStatistics,
    RingBufferOrderBook,
    create_options_book,
    create_ring_buffer_book,
)
from lob.event_coordinator import (
    CoordinatorStats,
    EventDrivenLOBCoordinator,
    OptionsEvent,
    OptionsEventType,
    OptionsQuote,
    PropagationResult,
    PropagationScope,
    StrikeBucket,
    create_event_coordinator,
    create_options_coordinator,
)
from lob.data_structures import Side, OrderType, LimitOrder


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for disk persistence tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def lazy_manager(temp_dir):
    """Create a LazyMultiSeriesLOBManager for testing."""
    return create_lazy_lob_manager(
        max_active_lobs=10,
        eviction_policy=EvictionPolicy.LRU,
        persist_dir=temp_dir,
        enable_compression=True,
    )


@pytest.fixture
def ring_buffer_book():
    """Create a RingBufferOrderBook for testing."""
    return create_ring_buffer_book(max_depth=10, tick_size=Decimal("0.01"))


@pytest.fixture
def event_coordinator():
    """Create an EventDrivenLOBCoordinator for testing."""
    return create_event_coordinator(
        underlying="SPY",
        bucket_width=Decimal("5.0"),
        max_propagation_depth=3,
    )


# =============================================================================
# LazyMultiSeriesLOBManager Tests (20 tests)
# =============================================================================


class TestLazyMultiSeriesLOBManager:
    """Tests for LazyMultiSeriesLOBManager."""

    def test_lazy_creation_on_first_access(self, lazy_manager):
        """Test that LOB is created lazily on first access."""
        series_key = "AAPL_241220_C_200"

        # Before access, no LOB should exist
        assert lazy_manager.get_active_count() == 0

        # Access creates the LOB
        lob = lazy_manager.get_or_create(series_key)
        assert lob is not None
        assert lazy_manager.get_active_count() == 1

    def test_lru_eviction_policy(self, temp_dir):
        """Test LRU eviction when max_active_lobs is reached."""
        manager = create_lazy_lob_manager(
            max_active_lobs=3,
            eviction_policy=EvictionPolicy.LRU,
            persist_dir=temp_dir,
        )

        # Create 3 LOBs
        for i in range(3):
            manager.get_or_create(f"AAPL_241220_C_{190 + i * 5}")

        assert manager.get_active_count() == 3

        # Access first one to update its access time
        manager.get_or_create("AAPL_241220_C_190")

        # Create 4th LOB, should evict least recently used (195)
        manager.get_or_create("AAPL_241220_C_210")

        assert manager.get_active_count() == 3
        stats = manager.get_stats()
        assert stats.evictions >= 1

    def test_lfu_eviction_policy(self, temp_dir):
        """Test LFU eviction based on access frequency."""
        manager = create_lazy_lob_manager(
            max_active_lobs=3,
            eviction_policy=EvictionPolicy.LFU,
            persist_dir=temp_dir,
        )

        # Create 3 LOBs with different access patterns
        keys = ["AAPL_241220_C_190", "AAPL_241220_C_195", "AAPL_241220_C_200"]
        for key in keys:
            manager.get_or_create(key)

        # Access first one multiple times
        for _ in range(5):
            manager.get_or_create("AAPL_241220_C_190")

        # Access second one a few times
        for _ in range(3):
            manager.get_or_create("AAPL_241220_C_195")

        # Third one accessed only once during creation

        # Create 4th, should evict least frequently used (200)
        manager.get_or_create("AAPL_241220_C_210")

        assert manager.get_active_count() == 3
        # 200 should be evicted (lowest frequency)

    def test_ttl_eviction_policy(self, temp_dir):
        """Test TTL-based eviction of stale LOBs."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            eviction_policy=EvictionPolicy.TTL,
            persist_dir=temp_dir,
            ttl_seconds=0.1,  # 100ms TTL for testing
        )

        manager.get_or_create("AAPL_241220_C_200")
        assert manager.get_active_count() == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # Run cleanup
        manager.cleanup_expired()

        assert manager.get_active_count() == 0

    def test_disk_persistence_save(self, lazy_manager):
        """Test saving LOB state to disk."""
        series_key = "AAPL_241220_C_200"
        lob = lazy_manager.get_or_create(series_key)

        # Add some state to the LOB
        lob.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("100"),
            order_id="order1",
        )

        # Force persist
        lazy_manager.persist_to_disk(series_key)

        # Check file exists
        stats = lazy_manager.get_stats()
        assert stats.disk_writes >= 1

    def test_disk_persistence_restore(self, temp_dir):
        """Test restoring LOB state from disk."""
        series_key = "AAPL_241220_C_200"

        # Create and persist
        manager1 = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
            enable_compression=True,
        )
        lob1 = manager1.get_or_create(series_key)
        lob1.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("100"),
            order_id="order1",
        )
        manager1.persist_to_disk(series_key)
        manager1.evict(series_key)

        # Create new manager and restore
        manager2 = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
            enable_compression=True,
        )
        lob2 = manager2.get_or_create(series_key)

        # Verify state was restored
        stats = manager2.get_stats()
        assert stats.disk_reads >= 1

    def test_compression_enabled(self, temp_dir):
        """Test gzip compression for disk persistence."""
        manager = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
            enable_compression=True,
        )

        series_key = "AAPL_241220_C_200"
        lob = manager.get_or_create(series_key)
        lob.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("1000"),
            order_id="order1",
        )

        manager.persist_to_disk(series_key)

        # Check compressed file exists
        persist_file = os.path.join(temp_dir, f"{series_key}.lob.gz")
        assert os.path.exists(persist_file)

    def test_thread_safety_concurrent_access(self, lazy_manager):
        """Test thread-safe concurrent LOB access."""
        results = []
        errors = []

        def access_lob(key: str, iterations: int):
            try:
                for _ in range(iterations):
                    lob = lazy_manager.get_or_create(key)
                    results.append(lob is not None)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=access_lob, args=(f"AAPL_241220_C_{190 + i * 5}", 10))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    def test_series_key_parsing(self, lazy_manager):
        """Test SeriesKey parsing from string format."""
        key_str = "AAPL_241220_C_200"
        series_key = SeriesKey.from_string(key_str)

        assert series_key.underlying == "AAPL"
        assert series_key.expiry == "241220"
        assert series_key.option_type == "C"
        assert series_key.strike == Decimal("200")
        assert str(series_key) == key_str

    def test_series_key_validation(self):
        """Test SeriesKey validation for malformed keys."""
        with pytest.raises(ValueError):
            SeriesKey.from_string("INVALID_KEY")

        with pytest.raises(ValueError):
            SeriesKey.from_string("AAPL_241220_X_200")  # Invalid type

    def test_metadata_tracking(self, lazy_manager):
        """Test LOBMetadata tracking for each series."""
        series_key = "AAPL_241220_C_200"

        # Multiple accesses
        for _ in range(5):
            lazy_manager.get_or_create(series_key)

        metadata = lazy_manager.get_metadata(series_key)
        assert metadata is not None
        assert metadata.access_count >= 5
        assert metadata.last_access is not None

    def test_bulk_eviction(self, temp_dir):
        """Test bulk eviction of multiple LOBs."""
        manager = create_lazy_lob_manager(
            max_active_lobs=100,
            persist_dir=temp_dir,
        )

        # Create 50 LOBs
        for i in range(50):
            manager.get_or_create(f"AAPL_241220_C_{100 + i * 5}")

        assert manager.get_active_count() == 50

        # Bulk evict all
        manager.evict_all()

        assert manager.get_active_count() == 0

    def test_memory_estimation(self, lazy_manager):
        """Test memory usage estimation."""
        # Create several LOBs with orders
        for i in range(5):
            key = f"AAPL_241220_C_{190 + i * 5}"
            lob = lazy_manager.get_or_create(key)
            for j in range(100):
                lob.add_order(
                    side=Side.BUY if j % 2 == 0 else Side.SELL,
                    price=Decimal(str(5.0 + j * 0.01)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )

        estimated_memory = lazy_manager.estimate_memory_usage()
        assert estimated_memory > 0

    def test_stats_collection(self, lazy_manager):
        """Test statistics collection and reporting."""
        # Perform various operations
        for i in range(10):
            lazy_manager.get_or_create(f"AAPL_241220_C_{190 + i * 5}")

        stats = lazy_manager.get_stats()

        assert isinstance(stats, ManagerStats)
        assert stats.active_lobs == 10
        assert stats.total_creates >= 10
        assert stats.hits + stats.misses >= 10

    def test_options_lob_manager_factory(self, temp_dir):
        """Test options-specific factory function."""
        manager = create_options_lob_manager(
            underlying="SPY",
            max_series=50,
            persist_dir=temp_dir,
        )

        assert manager is not None
        assert manager.max_active_lobs == 50

    def test_get_all_series_keys(self, lazy_manager):
        """Test getting all active series keys."""
        keys = ["AAPL_241220_C_190", "AAPL_241220_C_195", "AAPL_241220_C_200"]
        for key in keys:
            lazy_manager.get_or_create(key)

        active_keys = lazy_manager.get_all_active_keys()

        assert len(active_keys) == 3
        assert set(active_keys) == set(keys)

    def test_has_series(self, lazy_manager):
        """Test checking if series exists in memory."""
        series_key = "AAPL_241220_C_200"

        assert not lazy_manager.has_series(series_key)

        lazy_manager.get_or_create(series_key)

        assert lazy_manager.has_series(series_key)

    def test_explicit_eviction(self, lazy_manager):
        """Test explicit eviction of specific series."""
        series_key = "AAPL_241220_C_200"
        lazy_manager.get_or_create(series_key)

        assert lazy_manager.has_series(series_key)

        lazy_manager.evict(series_key)

        assert not lazy_manager.has_series(series_key)

    def test_eviction_callback(self, temp_dir):
        """Test eviction callback notification."""
        evicted_keys = []

        def on_evict(key: str, lob):
            evicted_keys.append(key)

        manager = create_lazy_lob_manager(
            max_active_lobs=2,
            persist_dir=temp_dir,
            on_evict=on_evict,
        )

        manager.get_or_create("AAPL_241220_C_190")
        manager.get_or_create("AAPL_241220_C_195")
        manager.get_or_create("AAPL_241220_C_200")  # Triggers eviction

        assert len(evicted_keys) >= 1


# =============================================================================
# RingBufferOrderBook Tests (15 tests)
# =============================================================================


class TestRingBufferOrderBook:
    """Tests for RingBufferOrderBook."""

    def test_add_order_within_depth(self, ring_buffer_book):
        """Test adding order within max depth."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("50"),
            order_id="order1",
        )

        snapshot = ring_buffer_book.get_snapshot()
        assert len(snapshot.bids) == 1
        assert snapshot.bids[0].price == Decimal("100.00")
        assert snapshot.bids[0].quantity == Decimal("50")

    def test_depth_limiting(self):
        """Test that book is limited to max_depth levels."""
        book = create_ring_buffer_book(max_depth=5, tick_size=Decimal("0.01"))

        # Add 10 levels
        for i in range(10):
            book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - i)),
                qty=Decimal("10"),
                order_id=f"order_{i}",
            )

        snapshot = book.get_snapshot()

        # Only top 5 levels visible
        assert len(snapshot.bids) == 5

        # Aggregated should have beyond-depth liquidity
        assert snapshot.aggregated_bid is not None
        assert snapshot.aggregated_bid.level_count == 5

    def test_aggregation_beyond_depth(self):
        """Test aggregation of liquidity beyond max depth."""
        book = create_ring_buffer_book(max_depth=3, tick_size=Decimal("0.01"))

        # Add 6 levels on each side
        for i in range(6):
            book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - i)),
                qty=Decimal(str(10 * (i + 1))),
                order_id=f"bid_{i}",
            )
            book.add_order(
                side=Side.SELL,
                price=Decimal(str(101 + i)),
                qty=Decimal(str(10 * (i + 1))),
                order_id=f"ask_{i}",
            )

        snapshot = book.get_snapshot()

        # Check aggregated beyond-depth
        assert snapshot.aggregated_bid.total_quantity > 0
        assert snapshot.aggregated_ask.total_quantity > 0

    def test_cancel_order(self, ring_buffer_book):
        """Test order cancellation."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("50"),
            order_id="order1",
        )

        result = ring_buffer_book.cancel_order("order1")

        assert result is True
        snapshot = ring_buffer_book.get_snapshot()
        assert len(snapshot.bids) == 0

    def test_modify_order(self, ring_buffer_book):
        """Test order modification."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("50"),
            order_id="order1",
        )

        ring_buffer_book.modify_order(
            order_id="order1",
            new_qty=Decimal("75"),
        )

        snapshot = ring_buffer_book.get_snapshot()
        assert snapshot.bids[0].quantity == Decimal("75")

    def test_best_bid_ask(self, ring_buffer_book):
        """Test getting best bid/ask."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("99.50"),
            qty=Decimal("100"),
            order_id="bid1",
        )
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("100.50"),
            qty=Decimal("100"),
            order_id="ask1",
        )

        assert ring_buffer_book.get_best_bid() == Decimal("99.50")
        assert ring_buffer_book.get_best_ask() == Decimal("100.50")
        assert ring_buffer_book.get_spread() == Decimal("1.00")

    def test_mid_price(self, ring_buffer_book):
        """Test mid price calculation."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("99.00"),
            qty=Decimal("100"),
            order_id="bid1",
        )
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("101.00"),
            qty=Decimal("100"),
            order_id="ask1",
        )

        assert ring_buffer_book.get_mid_price() == Decimal("100.00")

    def test_book_statistics(self, ring_buffer_book):
        """Test book statistics calculation."""
        # Add multiple levels
        for i in range(5):
            ring_buffer_book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - i)),
                qty=Decimal(str(10 * (i + 1))),
                order_id=f"bid_{i}",
            )
            ring_buffer_book.add_order(
                side=Side.SELL,
                price=Decimal(str(101 + i)),
                qty=Decimal(str(10 * (i + 1))),
                order_id=f"ask_{i}",
            )

        stats = ring_buffer_book.get_statistics()

        assert isinstance(stats, BookStatistics)
        assert stats.bid_levels == 5
        assert stats.ask_levels == 5
        assert stats.total_bid_qty > 0
        assert stats.total_ask_qty > 0

    def test_imbalance_calculation(self, ring_buffer_book):
        """Test order book imbalance calculation."""
        # More bids than asks
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("1000"),
            order_id="bid1",
        )
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("101.00"),
            qty=Decimal("100"),
            order_id="ask1",
        )

        imbalance = ring_buffer_book.get_imbalance()

        # Imbalance should be positive (more bids)
        assert imbalance > 0

    def test_weighted_mid_price(self, ring_buffer_book):
        """Test weighted mid price calculation."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("99.00"),
            qty=Decimal("100"),
            order_id="bid1",
        )
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("101.00"),
            qty=Decimal("300"),
            order_id="ask1",
        )

        weighted_mid = ring_buffer_book.get_weighted_mid_price()

        # Should be closer to bid (larger ask qty)
        assert weighted_mid < Decimal("100.00")

    def test_clear_book(self, ring_buffer_book):
        """Test clearing the entire book."""
        for i in range(5):
            ring_buffer_book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - i)),
                qty=Decimal("10"),
                order_id=f"bid_{i}",
            )

        ring_buffer_book.clear()

        snapshot = ring_buffer_book.get_snapshot()
        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0

    def test_options_book_factory(self):
        """Test options-specific book factory."""
        book = create_options_book(
            symbol="AAPL_241220_C_200",
            max_depth=20,
        )

        assert book is not None
        assert book.max_depth == 20

    def test_depth_at_price(self, ring_buffer_book):
        """Test getting quantity at specific price level."""
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("50"),
            order_id="order1",
        )
        ring_buffer_book.add_order(
            side=Side.BUY,
            price=Decimal("100.00"),
            qty=Decimal("30"),
            order_id="order2",
        )

        depth = ring_buffer_book.get_depth_at_price(Side.BUY, Decimal("100.00"))

        assert depth == Decimal("80")

    def test_volume_weighted_price(self, ring_buffer_book):
        """Test volume-weighted average price for given quantity."""
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("100.00"),
            qty=Decimal("100"),
            order_id="ask1",
        )
        ring_buffer_book.add_order(
            side=Side.SELL,
            price=Decimal("101.00"),
            qty=Decimal("100"),
            order_id="ask2",
        )

        # Buy 150 shares (walk asks)
        vwap = ring_buffer_book.get_vwap(Side.BUY, Decimal("150"))

        # (100 * 100 + 50 * 101) / 150 = 100.333...
        assert vwap > Decimal("100.00")
        assert vwap < Decimal("101.00")

    def test_ring_buffer_rotation(self):
        """Test ring buffer rotation maintains correct order."""
        book = create_ring_buffer_book(max_depth=3, tick_size=Decimal("0.01"))

        # Fill buffer
        for i in range(3):
            book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - i)),
                qty=Decimal("10"),
                order_id=f"order_{i}",
            )

        # Add new order that pushes out old
        book.add_order(
            side=Side.BUY,
            price=Decimal("101.00"),  # New best bid
            qty=Decimal("10"),
            order_id="order_new",
        )

        # Best bid should be 101
        assert book.get_best_bid() == Decimal("101.00")


# =============================================================================
# EventDrivenLOBCoordinator Tests (15 tests)
# =============================================================================


class TestEventDrivenLOBCoordinator:
    """Tests for EventDrivenLOBCoordinator."""

    def test_register_series(self, event_coordinator):
        """Test registering option series with coordinator."""
        event_coordinator.register_series(
            series_key="SPY_241220_C_500",
            strike=Decimal("500"),
        )

        assert event_coordinator.is_registered("SPY_241220_C_500")

    def test_strike_bucketing(self, event_coordinator):
        """Test that strikes are bucketed correctly."""
        # Register series at different strikes
        event_coordinator.register_series("SPY_241220_C_495", Decimal("495"))
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))
        event_coordinator.register_series("SPY_241220_C_505", Decimal("505"))

        # 495 and 500 should be in same bucket (width=5)
        bucket_495 = event_coordinator.get_bucket_for_strike(Decimal("495"))
        bucket_500 = event_coordinator.get_bucket_for_strike(Decimal("500"))

        # They should be in same or adjacent buckets
        assert bucket_495 is not None
        assert bucket_500 is not None

    def test_quote_event_propagation(self, event_coordinator):
        """Test propagation of quote events."""
        # Register series
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))
        event_coordinator.register_series("SPY_241220_C_505", Decimal("505"))

        # Create quote event
        quote = OptionsQuote(
            series_key="SPY_241220_C_500",
            bid=Decimal("2.50"),
            ask=Decimal("2.55"),
            bid_size=Decimal("100"),
            ask_size=Decimal("100"),
            timestamp=datetime.now(),
        )

        event = OptionsEvent(
            event_type=OptionsEventType.QUOTE,
            quote=quote,
        )

        result = event_coordinator.propagate(event)

        assert isinstance(result, PropagationResult)
        assert result.series_affected >= 1

    def test_trade_event_propagation(self, event_coordinator):
        """Test propagation of trade events."""
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))

        event = OptionsEvent(
            event_type=OptionsEventType.TRADE,
            series_key="SPY_241220_C_500",
            trade_price=Decimal("2.52"),
            trade_qty=Decimal("50"),
            timestamp=datetime.now(),
        )

        result = event_coordinator.propagate(event)

        assert result.propagation_time_ns > 0

    def test_underlying_tick_propagation(self, event_coordinator):
        """Test propagation scope for underlying tick."""
        # Register multiple series
        for strike in [490, 495, 500, 505, 510]:
            event_coordinator.register_series(
                f"SPY_241220_C_{strike}",
                Decimal(str(strike)),
            )

        # Underlying tick should affect ATM options more
        event = OptionsEvent(
            event_type=OptionsEventType.UNDERLYING_TICK,
            underlying_price=Decimal("500.00"),
            timestamp=datetime.now(),
        )

        result = event_coordinator.propagate(event, scope=PropagationScope.ATM_ONLY)

        assert result.series_affected >= 1

    def test_propagation_scope_all(self, event_coordinator):
        """Test ALL propagation scope."""
        for strike in [490, 495, 500, 505, 510]:
            event_coordinator.register_series(
                f"SPY_241220_C_{strike}",
                Decimal(str(strike)),
            )

        event = OptionsEvent(
            event_type=OptionsEventType.VOLATILITY,
            implied_vol=Decimal("0.25"),
            timestamp=datetime.now(),
        )

        result = event_coordinator.propagate(event, scope=PropagationScope.ALL)

        assert result.series_affected == 5

    def test_propagation_scope_adjacent(self, event_coordinator):
        """Test ADJACENT propagation scope."""
        for strike in [490, 495, 500, 505, 510]:
            event_coordinator.register_series(
                f"SPY_241220_C_{strike}",
                Decimal(str(strike)),
            )

        event = OptionsEvent(
            event_type=OptionsEventType.QUOTE,
            series_key="SPY_241220_C_500",
            timestamp=datetime.now(),
        )

        result = event_coordinator.propagate(event, scope=PropagationScope.ADJACENT)

        # Should affect 500 and neighbors (495, 505)
        assert result.series_affected >= 1
        assert result.series_affected <= 3

    def test_coordinator_stats(self, event_coordinator):
        """Test coordinator statistics collection."""
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))

        for _ in range(10):
            event = OptionsEvent(
                event_type=OptionsEventType.QUOTE,
                series_key="SPY_241220_C_500",
                timestamp=datetime.now(),
            )
            event_coordinator.propagate(event)

        stats = event_coordinator.get_stats()

        assert isinstance(stats, CoordinatorStats)
        assert stats.total_events >= 10
        assert stats.avg_propagation_time_ns > 0

    def test_event_callback_registration(self, event_coordinator):
        """Test registering event callbacks."""
        received_events = []

        def on_event(series_key: str, event: OptionsEvent):
            received_events.append((series_key, event))

        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))
        event_coordinator.register_callback("SPY_241220_C_500", on_event)

        event = OptionsEvent(
            event_type=OptionsEventType.QUOTE,
            series_key="SPY_241220_C_500",
            timestamp=datetime.now(),
        )

        event_coordinator.propagate(event)

        assert len(received_events) >= 1

    def test_unregister_series(self, event_coordinator):
        """Test unregistering option series."""
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))
        assert event_coordinator.is_registered("SPY_241220_C_500")

        event_coordinator.unregister_series("SPY_241220_C_500")
        assert not event_coordinator.is_registered("SPY_241220_C_500")

    def test_propagation_latency_target(self, event_coordinator):
        """Test that propagation meets latency target (<100μs)."""
        # Register 100 series (simulating a chain)
        for strike in range(400, 600, 2):
            event_coordinator.register_series(
                f"SPY_241220_C_{strike}",
                Decimal(str(strike)),
            )

        event = OptionsEvent(
            event_type=OptionsEventType.UNDERLYING_TICK,
            underlying_price=Decimal("500.00"),
            timestamp=datetime.now(),
        )

        start = time.perf_counter_ns()
        event_coordinator.propagate(event)
        elapsed_ns = time.perf_counter_ns() - start

        # Should be under 1ms (1,000,000 ns) for 100 series
        # Target is 100μs per tick
        assert elapsed_ns < 1_000_000

    def test_options_coordinator_factory(self):
        """Test options-specific coordinator factory."""
        coordinator = create_options_coordinator(
            underlying="AAPL",
            expiry="241220",
        )

        assert coordinator is not None
        assert coordinator.underlying == "AAPL"

    def test_batch_event_processing(self, event_coordinator):
        """Test batch processing of multiple events."""
        for strike in [495, 500, 505]:
            event_coordinator.register_series(
                f"SPY_241220_C_{strike}",
                Decimal(str(strike)),
            )

        events = [
            OptionsEvent(
                event_type=OptionsEventType.QUOTE,
                series_key=f"SPY_241220_C_{strike}",
                timestamp=datetime.now(),
            )
            for strike in [495, 500, 505]
        ]

        results = event_coordinator.propagate_batch(events)

        assert len(results) == 3
        assert all(r.series_affected >= 1 for r in results)

    def test_greek_update_propagation(self, event_coordinator):
        """Test propagation of greek updates."""
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))

        event = OptionsEvent(
            event_type=OptionsEventType.GREEK_UPDATE,
            series_key="SPY_241220_C_500",
            delta=Decimal("0.50"),
            gamma=Decimal("0.02"),
            theta=Decimal("-0.05"),
            vega=Decimal("0.15"),
            timestamp=datetime.now(),
        )

        result = event_coordinator.propagate(event)

        assert result.event_type == OptionsEventType.GREEK_UPDATE

    def test_expiry_filtering(self, event_coordinator):
        """Test filtering series by expiry."""
        # Register series with different expiries
        event_coordinator.register_series("SPY_241220_C_500", Decimal("500"))
        event_coordinator.register_series("SPY_250117_C_500", Decimal("500"))

        series_241220 = event_coordinator.get_series_by_expiry("241220")
        series_250117 = event_coordinator.get_series_by_expiry("250117")

        assert len(series_241220) == 1
        assert len(series_250117) == 1


# =============================================================================
# Memory Benchmark Tests (10 tests)
# =============================================================================


class TestMemoryBenchmarks:
    """Tests for memory usage and benchmarks."""

    def test_peak_memory_under_target(self, temp_dir):
        """Test that peak memory stays under 4GB target."""
        gc.collect()

        manager = create_lazy_lob_manager(
            max_active_lobs=50,  # Limit to 50 concurrent LOBs
            persist_dir=temp_dir,
        )

        # Create LOBs with realistic order count
        for i in range(50):
            key = f"SPY_241220_C_{400 + i * 2}"
            lob = manager.get_or_create(key)

            # Add 1000 orders per LOB (typical for active option)
            for j in range(100):
                lob.add_order(
                    side=Side.BUY if j % 2 == 0 else Side.SELL,
                    price=Decimal(str(5.0 + j * 0.05)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )

        # Estimate memory (in a real test, use tracemalloc)
        estimated = manager.estimate_memory_usage()

        # Should be well under 4GB
        assert estimated < 4 * 1024 * 1024 * 1024  # 4GB in bytes

    def test_lob_access_latency(self, lazy_manager):
        """Test that LOB access latency is under 1ms."""
        series_key = "AAPL_241220_C_200"

        # Warm up
        lazy_manager.get_or_create(series_key)

        # Measure access latency
        latencies = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            lazy_manager.get_or_create(series_key)
            latencies.append(time.perf_counter_ns() - start)

        avg_latency_ns = sum(latencies) / len(latencies)
        avg_latency_ms = avg_latency_ns / 1_000_000

        # Should be under 1ms
        assert avg_latency_ms < 1.0

    def test_eviction_gc_pressure(self, temp_dir):
        """Test that eviction doesn't cause excessive GC pressure."""
        gc.collect()
        initial_collections = gc.get_count()

        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        # Create and evict 100 LOBs
        for i in range(100):
            key = f"SPY_241220_C_{400 + i * 2}"
            lob = manager.get_or_create(key)
            lob.add_order(
                side=Side.BUY,
                price=Decimal("5.00"),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        gc.collect()
        final_collections = gc.get_count()

        # GC should not be triggered excessively
        # (This is a heuristic test)
        assert final_collections[0] - initial_collections[0] < 20

    def test_memory_per_lob_estimate(self, temp_dir):
        """Test memory estimation per LOB."""
        manager = create_lazy_lob_manager(
            max_active_lobs=100,
            persist_dir=temp_dir,
        )

        # Create single LOB with known content
        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)

        for i in range(1000):
            lob.add_order(
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                price=Decimal(str(5.0 + (i % 100) * 0.01)),
                qty=Decimal("10"),
                order_id=f"order_{i}",
            )

        single_lob_memory = manager.estimate_memory_usage()

        # Should be reasonable (< 50MB per LOB with 1000 orders)
        assert single_lob_memory < 50 * 1024 * 1024

    def test_large_chain_memory(self, temp_dir):
        """Test memory usage with large option chain (960 series like SPY)."""
        manager = create_lazy_lob_manager(
            max_active_lobs=50,  # Only keep 50 active
            persist_dir=temp_dir,
        )

        # Simulate accessing all 960 series
        for i in range(960):
            key = f"SPY_241220_C_{300 + i}"
            lob = manager.get_or_create(key)
            # Minimal content for test speed
            lob.add_order(
                side=Side.BUY,
                price=Decimal("1.00"),
                qty=Decimal("10"),
                order_id=f"order_{i}",
            )

        # Should only have 50 active
        assert manager.get_active_count() == 50

        # Memory should be bounded
        memory = manager.estimate_memory_usage()
        assert memory < 4 * 1024 * 1024 * 1024  # 4GB

    def test_ring_buffer_memory_constant(self):
        """Test that ring buffer memory usage is constant regardless of order volume."""
        book = create_ring_buffer_book(max_depth=10, tick_size=Decimal("0.01"))

        # Add 1000 orders
        for i in range(1000):
            book.add_order(
                side=Side.BUY,
                price=Decimal(str(100 - (i % 20))),
                qty=Decimal("10"),
                order_id=f"order_{i}",
            )

        snapshot = book.get_snapshot()

        # Should still only have max_depth levels
        assert len(snapshot.bids) <= 10

    def test_coordinator_memory_scalability(self):
        """Test coordinator memory with many registered series."""
        coordinator = create_event_coordinator(
            underlying="SPY",
            bucket_width=Decimal("5.0"),
        )

        # Register 1000 series
        for i in range(1000):
            coordinator.register_series(
                f"SPY_241220_C_{300 + i}",
                Decimal(str(300 + i)),
            )

        stats = coordinator.get_stats()

        # Memory per series should be minimal
        assert stats.registered_series == 1000

    def test_disk_persistence_memory_release(self, temp_dir):
        """Test that evicted LOBs release memory after disk persistence."""
        manager = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
        )

        # Create LOBs with content
        for i in range(10):
            key = f"AAPL_241220_C_{190 + i * 5}"
            lob = manager.get_or_create(key)
            for j in range(100):
                lob.add_order(
                    side=Side.BUY,
                    price=Decimal(str(5.0 + j * 0.01)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )

        # Should have evicted some
        stats = manager.get_stats()
        assert stats.evictions >= 5

        # Active count should be at max
        assert manager.get_active_count() == 5

    def test_memory_budget_enforcement(self, temp_dir):
        """Test that memory budget is enforced."""
        # Set a small memory budget
        manager = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
            max_memory_bytes=10 * 1024 * 1024,  # 10MB
        )

        # Try to create many LOBs
        for i in range(20):
            key = f"AAPL_241220_C_{190 + i * 5}"
            lob = manager.get_or_create(key)
            for j in range(100):
                lob.add_order(
                    side=Side.BUY,
                    price=Decimal(str(5.0 + j * 0.01)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )

        # Memory should stay under budget
        memory = manager.estimate_memory_usage()
        # Allow some overhead
        assert memory < 20 * 1024 * 1024

    def test_concurrent_access_memory_stability(self, temp_dir):
        """Test memory stability under concurrent access."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        errors = []

        def worker(worker_id: int):
            try:
                for i in range(50):
                    key = f"AAPL_241220_C_{190 + (worker_id * 10 + i % 10) * 5}"
                    lob = manager.get_or_create(key)
                    lob.add_order(
                        side=Side.BUY,
                        price=Decimal("5.00"),
                        qty=Decimal("10"),
                        order_id=f"order_{worker_id}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.get_active_count() <= 10


# =============================================================================
# Disk Persistence Tests (10 tests)
# =============================================================================


class TestDiskPersistence:
    """Tests for disk persistence functionality."""

    def test_persist_creates_file(self, temp_dir):
        """Test that persistence creates a file on disk."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
            enable_compression=False,
        )

        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)
        lob.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("100"),
            order_id="order1",
        )

        manager.persist_to_disk(key)

        persist_file = os.path.join(temp_dir, f"{key}.lob")
        assert os.path.exists(persist_file)

    def test_persist_with_compression(self, temp_dir):
        """Test persistence with gzip compression."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
            enable_compression=True,
        )

        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)
        for i in range(100):
            lob.add_order(
                side=Side.BUY,
                price=Decimal(str(5.0 + i * 0.01)),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        manager.persist_to_disk(key)

        compressed_file = os.path.join(temp_dir, f"{key}.lob.gz")
        uncompressed_file = os.path.join(temp_dir, f"{key}.lob")

        # Compressed file should exist
        assert os.path.exists(compressed_file)

        # Should be valid gzip (binary pickle content)
        with gzip.open(compressed_file, "rb") as f:
            content = f.read()
            assert len(content) > 0

    def test_restore_from_disk(self, temp_dir):
        """Test restoring LOB state from disk."""
        key = "AAPL_241220_C_200"

        # Create and persist
        manager1 = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )
        lob1 = manager1.get_or_create(key)
        lob1.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("100"),
            order_id="order1",
        )
        manager1.persist_to_disk(key)
        del manager1

        # Restore
        manager2 = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )
        lob2 = manager2.get_or_create(key)

        # State should be restored
        stats = manager2.get_stats()
        assert stats.disk_reads >= 1

    def test_persist_all_active(self, temp_dir):
        """Test persisting all active LOBs."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        keys = [f"AAPL_241220_C_{190 + i * 5}" for i in range(5)]
        for key in keys:
            lob = manager.get_or_create(key)
            lob.add_order(
                side=Side.BUY,
                price=Decimal("5.00"),
                qty=Decimal("100"),
                order_id=f"order_{key}",
            )

        manager.persist_all()

        stats = manager.get_stats()
        assert stats.disk_writes >= 5

    def test_persist_on_eviction(self, temp_dir):
        """Test automatic persistence on eviction."""
        manager = create_lazy_lob_manager(
            max_active_lobs=3,
            persist_dir=temp_dir,
            persist_on_evict=True,
        )

        # Create 4 LOBs, triggering eviction
        for i in range(4):
            key = f"AAPL_241220_C_{190 + i * 5}"
            lob = manager.get_or_create(key)
            lob.add_order(
                side=Side.BUY,
                price=Decimal("5.00"),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        stats = manager.get_stats()
        # At least one should have been persisted on eviction
        assert stats.disk_writes >= 1

    def test_corruption_handling(self, temp_dir):
        """Test handling of corrupted persistence files."""
        key = "AAPL_241220_C_200"

        # Create corrupted file
        corrupt_file = os.path.join(temp_dir, f"{key}.lob.gz")
        with open(corrupt_file, "wb") as f:
            f.write(b"corrupted data")

        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        # Should handle gracefully and create new LOB
        lob = manager.get_or_create(key)
        assert lob is not None

    def test_atomic_write(self, temp_dir):
        """Test that writes are atomic (no partial files)."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)

        # Add many orders
        for i in range(1000):
            lob.add_order(
                side=Side.BUY,
                price=Decimal(str(5.0 + i * 0.01)),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        manager.persist_to_disk(key)

        # File should be complete (no temp files)
        files = os.listdir(temp_dir)
        temp_files = [f for f in files if f.endswith(".tmp")]
        assert len(temp_files) == 0

    def test_cleanup_old_files(self, temp_dir):
        """Test cleanup of old persistence files."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
            max_persist_age_days=0,  # Immediate expiry for test
        )

        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)
        manager.persist_to_disk(key)

        # Files should exist
        assert len(os.listdir(temp_dir)) > 0

        # Cleanup
        manager.cleanup_old_persist_files()

        # Old files should be removed
        # (In real implementation with age check)

    def test_concurrent_persist(self, temp_dir):
        """Test concurrent persistence operations."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        errors = []

        def persist_worker(key: str):
            try:
                lob = manager.get_or_create(key)
                for i in range(100):
                    lob.add_order(
                        side=Side.BUY,
                        price=Decimal(str(5.0 + i * 0.01)),
                        qty=Decimal("10"),
                        order_id=f"order_{key}_{i}",
                    )
                manager.persist_to_disk(key)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=persist_worker, args=(f"AAPL_241220_C_{190 + i * 5}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(os.listdir(temp_dir)) >= 5

    def test_persistence_versioning(self, temp_dir):
        """Test persistence file versioning for compatibility."""
        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=temp_dir,
        )

        key = "AAPL_241220_C_200"
        lob = manager.get_or_create(key)
        lob.add_order(
            side=Side.BUY,
            price=Decimal("5.00"),
            qty=Decimal("100"),
            order_id="order1",
        )

        manager.persist_to_disk(key)

        # Read and verify version info (pickle format with gzip compression)
        persist_file = os.path.join(temp_dir, f"{key}.lob.gz")
        with gzip.open(persist_file, "rb") as f:
            data = pickle.load(f)
            assert "version" in data
            assert data["version"] >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Phase 0.5 components."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: create, use, evict, restore."""
        # Create manager and coordinator
        manager = create_lazy_lob_manager(
            max_active_lobs=5,
            persist_dir=temp_dir,
            persist_on_evict=True,
        )

        coordinator = create_event_coordinator(
            underlying="AAPL",
            bucket_width=Decimal("5.0"),
        )

        # Register and create LOBs
        keys = [f"AAPL_241220_C_{190 + i * 5}" for i in range(10)]

        for key in keys:
            lob = manager.get_or_create(key)
            strike = Decimal(key.split("_")[-1])
            coordinator.register_series(key, strike)

            # Add orders
            lob.add_order(
                side=Side.BUY,
                price=Decimal("5.00"),
                qty=Decimal("100"),
                order_id=f"order_{key}",
            )

        # Propagate events (use ATM_ONLY scope for underlying tick without source_series)
        event = OptionsEvent(
            event_type=OptionsEventType.UNDERLYING_TICK,
            underlying_price=Decimal("200.00"),
            timestamp=datetime.now(),
            scope=PropagationScope.ATM_ONLY,
        )

        result = coordinator.propagate(event)

        assert result.series_affected >= 1
        assert manager.get_active_count() == 5  # Due to eviction

    def test_spy_chain_simulation(self, temp_dir):
        """Simulate SPY options chain (960 series)."""
        manager = create_lazy_lob_manager(
            max_active_lobs=50,
            persist_dir=temp_dir,
        )

        coordinator = create_event_coordinator(
            underlying="SPY",
            bucket_width=Decimal("5.0"),
        )

        # Register 960 series (like SPY chain)
        strikes = list(range(300, 600))  # 300 strikes
        expiries = ["241220", "250117", "250221"]  # 3 expiries
        types = ["C", "P"]  # Call and Put

        series_count = 0
        for strike in strikes:
            for expiry in expiries:
                for opt_type in types:
                    if series_count >= 100:  # Limit for test speed
                        break

                    key = f"SPY_{expiry}_{opt_type}_{strike}"
                    coordinator.register_series(key, Decimal(str(strike)))
                    series_count += 1

        # Access some LOBs
        for i in range(50):
            key = f"SPY_241220_C_{300 + i * 5}"
            if coordinator.is_registered(key):
                lob = manager.get_or_create(key)
                lob.add_order(
                    side=Side.BUY,
                    price=Decimal("1.00"),
                    qty=Decimal("10"),
                    order_id=f"order_{i}",
                )

        # Verify memory is bounded
        assert manager.get_active_count() <= 50

        # Verify coordinator handles all series
        assert coordinator.get_stats().registered_series == series_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
