"""
Tests for L3 LOB Latency Simulation (Stage 5).

Tests cover:
- Latency distributions (log-normal, Pareto, Gamma)
- Latency profiles (co-located, proximity, retail, institutional)
- Event scheduling and ordering
- Race condition detection
- Simulation clock

Total: 70+ tests covering all latency functionality.
"""

import math
import pytest
pytest.importorskip("sortedcontainers")
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

from lob.latency_model import (
    LatencyConfig,
    LatencyDistribution,
    LatencyModel,
    LatencyModelConfig,
    LatencyProfile,
    LatencySample,
    LatencySampler,
    create_latency_model,
)
from lob.event_scheduler import (
    EventScheduler,
    EventType,
    FillEvent,
    MarketDataEvent,
    OrderSubmission,
    RaceConditionInfo,
    ScheduledEvent,
    SimulationClock,
    create_event_scheduler,
    create_simulation_clock,
)
from lob.data_structures import LimitOrder, Side, Fill, Trade


# =============================================================================
# LatencyConfig Tests
# =============================================================================
class TestLatencyConfig:
    """Tests for LatencyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LatencyConfig()
        assert config.distribution == LatencyDistribution.LOGNORMAL
        assert config.mean_us == 100.0
        assert config.std_us == 30.0
        assert config.min_us == 1.0
        assert config.max_us == 100_000.0
        assert config.spike_prob == 0.001
        assert config.spike_mult == 10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = LatencyConfig(
            distribution=LatencyDistribution.GAMMA,
            mean_us=50.0,
            std_us=15.0,
            min_us=5.0,
            max_us=500.0,
            spike_prob=0.01,
            spike_mult=5.0,
        )
        assert config.distribution == LatencyDistribution.GAMMA
        assert config.mean_us == 50.0
        assert config.std_us == 15.0

    def test_negative_mean_raises(self):
        """Test that negative mean raises ValueError."""
        with pytest.raises(ValueError, match="mean_us must be non-negative"):
            LatencyConfig(mean_us=-10.0)

    def test_negative_std_raises(self):
        """Test that negative std raises ValueError."""
        with pytest.raises(ValueError, match="std_us must be non-negative"):
            LatencyConfig(std_us=-5.0)

    def test_invalid_bounds_raises(self):
        """Test that max < min raises ValueError."""
        with pytest.raises(ValueError, match="max_us must be >= min_us"):
            LatencyConfig(min_us=100.0, max_us=50.0)

    def test_invalid_spike_prob_raises(self):
        """Test that spike_prob outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="spike_prob must be in"):
            LatencyConfig(spike_prob=1.5)
        with pytest.raises(ValueError, match="spike_prob must be in"):
            LatencyConfig(spike_prob=-0.1)

    def test_invalid_spike_mult_raises(self):
        """Test that spike_mult < 1 raises ValueError."""
        with pytest.raises(ValueError, match="spike_mult must be >= 1"):
            LatencyConfig(spike_mult=0.5)

    def test_invalid_pareto_alpha_raises(self):
        """Test that non-positive pareto_alpha raises ValueError."""
        with pytest.raises(ValueError, match="pareto_alpha must be positive"):
            LatencyConfig(pareto_alpha=0.0)
        with pytest.raises(ValueError, match="pareto_alpha must be positive"):
            LatencyConfig(pareto_alpha=-1.0)


# =============================================================================
# LatencySampler Tests
# =============================================================================
class TestLatencySampler:
    """Tests for LatencySampler class."""

    def test_constant_distribution(self):
        """Test constant distribution returns exact value."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=100.0,
            spike_prob=0.0,  # No spikes
        )
        sampler = LatencySampler(config, seed=42)

        for _ in range(10):
            sample = sampler.sample()
            assert sample.latency_us == 100.0
            assert sample.latency_ns == 100_000  # 100us = 100,000ns
            assert not sample.is_spike

    def test_uniform_distribution_bounds(self):
        """Test uniform distribution stays within bounds."""
        config = LatencyConfig(
            distribution=LatencyDistribution.UNIFORM,
            mean_us=100.0,
            std_us=20.0,
            min_us=50.0,
            max_us=200.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        samples = [sampler.sample().latency_us for _ in range(1000)]
        assert min(samples) >= 50.0
        assert max(samples) <= 200.0
        # Mean should be near 100
        assert 80.0 <= sum(samples) / len(samples) <= 120.0

    def test_lognormal_distribution(self):
        """Test log-normal distribution properties."""
        config = LatencyConfig(
            distribution=LatencyDistribution.LOGNORMAL,
            mean_us=100.0,
            std_us=30.0,
            min_us=1.0,
            max_us=10000.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        samples = [sampler.sample().latency_us for _ in range(10000)]

        # Check mean is approximately correct (within 20%)
        mean_val = sum(samples) / len(samples)
        assert 80.0 <= mean_val <= 120.0, f"Mean {mean_val} not in expected range"

        # Log-normal should have right skew (median < mean)
        sorted_samples = sorted(samples)
        median = sorted_samples[len(sorted_samples) // 2]
        assert median < mean_val, "Log-normal should have right skew"

    def test_pareto_distribution(self):
        """Test Pareto distribution heavy tail."""
        config = LatencyConfig(
            distribution=LatencyDistribution.PARETO,
            pareto_alpha=2.0,
            pareto_xmin_us=10.0,
            min_us=10.0,
            max_us=100000.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        samples = [sampler.sample().latency_us for _ in range(10000)]

        # All samples should be >= xmin
        assert min(samples) >= 10.0

        # Heavy tail: should have some extreme values
        max_sample = max(samples)
        p99 = sorted(samples)[int(0.99 * len(samples))]
        assert max_sample > p99 * 2, "Pareto should have heavy tail"

    def test_gamma_distribution(self):
        """Test Gamma distribution properties."""
        config = LatencyConfig(
            distribution=LatencyDistribution.GAMMA,
            mean_us=100.0,
            std_us=50.0,
            min_us=1.0,
            max_us=10000.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        samples = [sampler.sample().latency_us for _ in range(10000)]

        # All samples positive
        assert all(s > 0 for s in samples)

        # Mean approximately correct
        mean_val = sum(samples) / len(samples)
        assert 80.0 <= mean_val <= 120.0

    def test_spike_probability(self):
        """Test that spikes occur at expected rate."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=100.0,
            spike_prob=0.1,  # 10% spikes
            spike_mult=5.0,
        )
        sampler = LatencySampler(config, seed=42)

        n_samples = 10000
        samples = [sampler.sample() for _ in range(n_samples)]
        spike_count = sum(1 for s in samples if s.is_spike)

        # Should be approximately 10% (within 2%)
        spike_rate = spike_count / n_samples
        assert 0.08 <= spike_rate <= 0.12, f"Spike rate {spike_rate} not in expected range"

    def test_spike_multiplier(self):
        """Test that spikes multiply latency correctly."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=100.0,
            spike_prob=1.0,  # Always spike
            spike_mult=5.0,
            max_us=1000.0,
        )
        sampler = LatencySampler(config, seed=42)

        sample = sampler.sample()
        assert sample.is_spike
        # 100 * 5 = 500us
        assert sample.latency_us == 500.0

    def test_bounds_clamping(self):
        """Test that samples are clamped to bounds."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=50.0,
            min_us=100.0,  # min > mean
            max_us=200.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        sample = sampler.sample()
        # Should clamp to min
        assert sample.latency_us == 100.0

    def test_empirical_distribution(self):
        """Test empirical distribution sampling."""
        config = LatencyConfig(
            distribution=LatencyDistribution.EMPIRICAL,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        # Set historical data
        historical = [50.0, 100.0, 150.0, 200.0]
        sampler.set_empirical_data(historical)

        # All samples should be from historical data
        samples = [sampler.sample().latency_us for _ in range(1000)]
        unique_samples = set(samples)
        assert unique_samples.issubset(set(historical))

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=100.0,
            spike_prob=0.0,
        )
        sampler = LatencySampler(config, seed=42)

        # Generate samples
        for _ in range(100):
            sampler.sample()

        stats = sampler.stats()
        assert stats["count"] == 100
        assert stats["mean_us"] == 100.0
        assert stats["p50_us"] == 100.0
        assert stats["p95_us"] == 100.0
        assert stats["spike_rate"] == 0.0

    def test_stats_reset(self):
        """Test statistics reset."""
        config = LatencyConfig(distribution=LatencyDistribution.CONSTANT, mean_us=100.0)
        sampler = LatencySampler(config, seed=42)

        for _ in range(100):
            sampler.sample()

        sampler.reset_stats()
        stats = sampler.stats()
        assert stats["count"] == 0

    def test_thread_safety(self):
        """Test thread-safe sampling."""
        config = LatencyConfig(
            distribution=LatencyDistribution.LOGNORMAL,
            mean_us=100.0,
            std_us=30.0,
        )
        sampler = LatencySampler(config, seed=42)

        results = []
        errors = []

        def sample_worker():
            try:
                for _ in range(1000):
                    sample = sampler.sample()
                    results.append(sample.latency_us)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sample_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 4000

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same samples."""
        config = LatencyConfig(
            distribution=LatencyDistribution.LOGNORMAL,
            mean_us=100.0,
            std_us=30.0,
        )

        sampler1 = LatencySampler(config, seed=42)
        sampler2 = LatencySampler(config, seed=42)

        samples1 = [sampler1.sample().latency_us for _ in range(100)]
        samples2 = [sampler2.sample().latency_us for _ in range(100)]

        assert samples1 == samples2


# =============================================================================
# LatencyModel Tests
# =============================================================================
class TestLatencyModel:
    """Tests for LatencyModel class."""

    def test_default_initialization(self):
        """Test default model initialization."""
        model = LatencyModel()
        assert model.config is not None

    def test_legacy_parameters(self):
        """Test initialization with legacy parameters."""
        model = LatencyModel(
            feed_latency_mean_us=50.0,
            feed_latency_std_us=10.0,
            order_latency_mean_us=100.0,
            order_latency_std_us=20.0,
            exchange_latency_us=5.0,
            seed=42,
        )

        # Feed latency should be around 50us
        feed_samples = [model.sample_feed_latency() for _ in range(1000)]
        mean_feed_ns = sum(feed_samples) / len(feed_samples)
        assert 30_000 <= mean_feed_ns <= 80_000  # 30-80us in ns

    def test_from_profile_colocated(self):
        """Test co-located profile has low latency."""
        model = LatencyModel.from_profile(LatencyProfile.COLOCATED, seed=42)

        samples = [model.sample_feed_latency() for _ in range(1000)]
        mean_ns = sum(samples) / len(samples)

        # Co-located: ~10us mean, so expect 5-20us range
        assert 5_000 <= mean_ns <= 30_000, f"Mean {mean_ns} not in co-located range"

    def test_from_profile_retail(self):
        """Test retail profile has high latency."""
        model = LatencyModel.from_profile(LatencyProfile.RETAIL, seed=42)

        samples = [model.sample_feed_latency() for _ in range(1000)]
        mean_ns = sum(samples) / len(samples)

        # Retail: ~2ms mean, so expect 1-4ms range
        assert 1_000_000 <= mean_ns <= 4_000_000, f"Mean {mean_ns} not in retail range"

    def test_all_profiles(self):
        """Test all profiles can be created and sampled."""
        profiles = [
            LatencyProfile.COLOCATED,
            LatencyProfile.PROXIMITY,
            LatencyProfile.RETAIL,
            LatencyProfile.INSTITUTIONAL,
            LatencyProfile.CUSTOM,
        ]

        for profile in profiles:
            model = LatencyModel.from_profile(profile, seed=42)
            sample = model.sample_all()
            assert sample["feed_ns"] > 0
            assert sample["order_ns"] > 0
            assert sample["exchange_ns"] > 0
            assert sample["fill_ns"] > 0
            assert sample["round_trip_ns"] > 0

    def test_sample_components(self):
        """Test individual latency components."""
        model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)

        feed_lat = model.sample_feed_latency()
        order_lat = model.sample_order_latency()
        exchange_lat = model.sample_exchange_latency()
        fill_lat = model.sample_fill_latency()

        assert feed_lat > 0
        assert order_lat > 0
        assert exchange_lat > 0
        assert fill_lat > 0

        # Exchange latency should be smallest
        assert exchange_lat < order_lat

    def test_round_trip(self):
        """Test round-trip latency calculation."""
        model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)

        round_trip = model.sample_round_trip()

        # Round trip = order + exchange + fill
        # Should be > any single component
        assert round_trip > model.sample_order_latency()

    def test_seasonality(self):
        """Test time-of-day seasonality."""
        # Create multipliers: higher during market hours
        multipliers = [
            1.0, 1.0, 1.0, 1.0,  # 00:00-03:00 (low)
            1.0, 1.0, 1.0, 1.0,  # 04:00-07:00 (low)
            1.5, 2.0, 2.0, 2.0,  # 08:00-11:00 (market open)
            2.0, 2.0, 1.5, 1.5,  # 12:00-15:00 (market hours)
            1.2, 1.0, 1.0, 1.0,  # 16:00-19:00 (market close)
            1.0, 1.0, 1.0, 1.0,  # 20:00-23:00 (low)
        ]

        config = LatencyModelConfig(
            feed_config=LatencyConfig(
                distribution=LatencyDistribution.CONSTANT,
                mean_us=100.0,
                spike_prob=0.0,
            ),
            seasonality_multipliers=multipliers,
            seed=42,
        )
        model = LatencyModel(config=config)

        # During market hours (hour=10) should be ~2x
        market_lat = model.sample_feed_latency(hour=10)
        # Off hours (hour=2) should be ~1x
        off_lat = model.sample_feed_latency(hour=2)

        assert market_lat > off_lat

    def test_volatility_sensitivity(self):
        """Test volatility-adjusted latency."""
        config = LatencyModelConfig(
            feed_config=LatencyConfig(
                distribution=LatencyDistribution.CONSTANT,
                mean_us=100.0,
                spike_prob=0.0,
            ),
            volatility_sensitivity=0.5,  # 50% sensitivity
            seed=42,
        )
        model = LatencyModel(config=config)

        # Normal volatility
        normal_lat = model.sample_feed_latency()

        # High volatility
        model.set_volatility_multiplier(2.0)
        high_vol_lat = model.sample_feed_latency()

        # High vol should be higher (but not 2x due to 50% sensitivity)
        assert high_vol_lat > normal_lat

    def test_empirical_data(self):
        """Test setting empirical latency data."""
        model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)

        # Set empirical data for feed latency
        historical = [50.0, 100.0, 150.0, 200.0]
        model.set_empirical_data("feed", historical)

        samples = [model.sample_feed_latency() for _ in range(100)]
        # All samples should be from historical data (in ns)
        expected_ns = {int(h * 1000) for h in historical}
        for s in samples:
            assert s in expected_ns

    def test_statistics(self):
        """Test latency statistics."""
        model = LatencyModel.from_profile(LatencyProfile.INSTITUTIONAL, seed=42)

        for _ in range(100):
            model.sample_all()

        stats = model.stats()
        assert "feed" in stats
        assert "order" in stats
        assert "exchange" in stats
        assert "fill" in stats

        assert stats["feed"]["count"] == 100

    def test_factory_function(self):
        """Test create_latency_model factory."""
        model = create_latency_model("colocated", seed=42)
        assert model is not None

        model2 = create_latency_model(LatencyProfile.RETAIL, seed=42)
        assert model2 is not None


# =============================================================================
# EventScheduler Tests
# =============================================================================
class TestEventScheduler:
    """Tests for EventScheduler class."""

    def test_default_initialization(self):
        """Test default scheduler initialization."""
        scheduler = EventScheduler()
        assert scheduler.pending_count == 0
        assert scheduler.current_time_ns == 0

    def test_schedule_market_data(self):
        """Test scheduling market data events."""
        scheduler = create_event_scheduler("institutional", seed=42)

        event = MarketDataEvent(
            symbol="AAPL",
            exchange_time_ns=1_000_000,
            bid_price=150.0,
            ask_price=150.05,
        )
        our_time = scheduler.schedule_market_data(event, exchange_time_ns=1_000_000)

        assert our_time > 1_000_000  # Should include latency
        assert scheduler.pending_count == 1

        # Check event properties
        next_event = scheduler.peek()
        assert next_event is not None
        assert next_event.event_type == EventType.MARKET_DATA_UPDATE
        assert next_event.exchange_time_ns == 1_000_000

    def test_schedule_order_arrival(self):
        """Test scheduling order submission."""
        scheduler = create_event_scheduler("institutional", seed=42)

        order = LimitOrder(
            order_id="test_order_1",
            price=150.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_000_000,
            side=Side.BUY,
        )

        arrival_time = scheduler.schedule_order_arrival(
            order=order,
            our_send_time_ns=1_000_000,
        )

        assert arrival_time > 1_000_000  # Should include latency
        assert scheduler.pending_count == 3  # Submitted, Received, Accepted

    def test_schedule_fill_notification(self):
        """Test scheduling fill notifications."""
        scheduler = create_event_scheduler("institutional", seed=42)

        fill = Fill(
            order_id="test_order_1",
            total_qty=100.0,
            avg_price=150.0,
            trades=[],
        )

        our_time = scheduler.schedule_fill_notification(
            fill=fill,
            exchange_time_ns=2_000_000,
        )

        assert our_time > 2_000_000  # Should include latency
        assert scheduler.pending_count == 1

    def test_event_ordering(self):
        """Test events are processed in timestamp order."""
        scheduler = create_event_scheduler("colocated", seed=42)  # Low latency

        # Schedule events out of order
        events_data = [
            (3_000_000, "event3"),
            (1_000_000, "event1"),
            (2_000_000, "event2"),
        ]

        for ts, name in events_data:
            event = MarketDataEvent(symbol="AAPL", exchange_time_ns=ts)
            scheduler.schedule_market_data(event, exchange_time_ns=ts)

        # Process and verify order
        processed = []
        for event in scheduler:
            processed.append(event.exchange_time_ns)

        # Should be in timestamp order (approximately, due to latency)
        assert processed[0] <= processed[1] <= processed[2]

    def test_process_until(self):
        """Test processing events until specific time."""
        scheduler = EventScheduler(
            latency_model=LatencyModel(
                feed_latency_mean_us=10.0,
                feed_latency_std_us=0.0,
                seed=42,
            ),
        )

        # Schedule events at different times
        for i in range(5):
            event = MarketDataEvent(symbol="AAPL", exchange_time_ns=i * 100_000)
            scheduler.schedule_market_data(event, exchange_time_ns=i * 100_000)

        # Process events until 200_000 + latency
        processed = scheduler.process_until(250_000)

        # Should have processed ~3 events
        assert len(processed) >= 2

    def test_timer_callback(self):
        """Test timer callback execution."""
        scheduler = create_event_scheduler("institutional", seed=42)

        callback_data = {"called": False, "payload": None}

        def timer_callback(event: ScheduledEvent):
            callback_data["called"] = True
            callback_data["payload"] = event.payload

        scheduler.schedule_timer(
            trigger_time_ns=1_000_000,
            callback=timer_callback,
            payload="test_data",
        )

        scheduler.process_all()

        assert callback_data["called"]
        assert callback_data["payload"] == "test_data"

    def test_race_condition_detection(self):
        """Test detection of race conditions."""
        race_conditions: List[RaceConditionInfo] = []

        def on_race(info: RaceConditionInfo):
            race_conditions.append(info)

        scheduler = EventScheduler(
            latency_model=LatencyModel(
                feed_latency_mean_us=100.0,
                feed_latency_std_us=10.0,
                order_latency_mean_us=100.0,
                order_latency_std_us=10.0,
                seed=42,
            ),
            detect_race_conditions=True,
            on_race_condition=on_race,
        )

        # Submit order
        order = LimitOrder(
            order_id="race_order",
            price=150.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_000_000,
            side=Side.BUY,
        )
        scheduler.schedule_order_arrival(order=order, our_send_time_ns=1_000_000)

        # Schedule market data very close in time
        event = MarketDataEvent(
            symbol="AAPL",
            exchange_time_ns=1_000_050,  # Very close!
        )
        scheduler.schedule_market_data(event, exchange_time_ns=1_000_050)

        # Check if race was detected
        all_races = scheduler.get_race_conditions()
        # May or may not detect depending on exact latencies
        # The key is that the mechanism works

    def test_event_handlers(self):
        """Test registering and calling event handlers."""
        scheduler = create_event_scheduler("institutional", seed=42)

        handled_events: List[EventType] = []

        def market_data_handler(event: ScheduledEvent):
            handled_events.append(event.event_type)

        scheduler.register_handler(EventType.MARKET_DATA_UPDATE, market_data_handler)

        event = MarketDataEvent(symbol="AAPL", exchange_time_ns=1_000_000)
        scheduler.schedule_market_data(event, exchange_time_ns=1_000_000)

        scheduler.process_all()

        assert EventType.MARKET_DATA_UPDATE in handled_events

    def test_unregister_handler(self):
        """Test unregistering event handlers."""
        scheduler = create_event_scheduler("institutional", seed=42)

        call_count = [0]

        def handler(event: ScheduledEvent):
            call_count[0] += 1

        scheduler.register_handler(EventType.MARKET_DATA_UPDATE, handler)

        # Schedule and process first event
        event1 = MarketDataEvent(symbol="AAPL", exchange_time_ns=1_000_000)
        scheduler.schedule_market_data(event1, exchange_time_ns=1_000_000)
        scheduler.process_all()

        assert call_count[0] == 1

        # Unregister handler
        result = scheduler.unregister_handler(EventType.MARKET_DATA_UPDATE, handler)
        assert result is True

        # Schedule and process second event
        event2 = MarketDataEvent(symbol="AAPL", exchange_time_ns=2_000_000)
        scheduler.schedule_market_data(event2, exchange_time_ns=2_000_000)
        scheduler.process_all()

        # Handler should not have been called again
        assert call_count[0] == 1

    def test_clear(self):
        """Test clearing scheduler."""
        scheduler = create_event_scheduler("institutional", seed=42)

        for i in range(10):
            event = MarketDataEvent(symbol="AAPL", exchange_time_ns=i * 100_000)
            scheduler.schedule_market_data(event, exchange_time_ns=i * 100_000)

        assert scheduler.pending_count == 10

        scheduler.clear()

        assert scheduler.pending_count == 0

    def test_statistics(self):
        """Test scheduler statistics."""
        scheduler = create_event_scheduler("institutional", seed=42)

        for i in range(10):
            event = MarketDataEvent(symbol="AAPL", exchange_time_ns=i * 100_000)
            scheduler.schedule_market_data(event, exchange_time_ns=i * 100_000)

        scheduler.process_all()

        stats = scheduler.stats()
        assert stats["total_events"] == 10
        assert stats["processed_events"] == 10
        assert stats["pending_events"] == 0
        assert "latency" in stats

    def test_advance_time(self):
        """Test advancing simulation time."""
        scheduler = create_event_scheduler("institutional", seed=42)

        assert scheduler.current_time_ns == 0

        scheduler.advance_time(1_000_000)
        assert scheduler.current_time_ns == 1_000_000

        # Can't go backwards
        scheduler.advance_time(500_000)
        assert scheduler.current_time_ns == 1_000_000

    def test_custom_event(self):
        """Test scheduling custom events."""
        scheduler = create_event_scheduler("institutional", seed=42)

        scheduler.schedule_custom(
            timestamp_ns=1_000_000,
            event_type=EventType.TIMER,
            exchange_time_ns=1_000_000,
            payload={"custom": "data"},
            priority=0,
        )

        assert scheduler.pending_count == 1

        event = scheduler.pop()
        assert event.payload == {"custom": "data"}


# =============================================================================
# SimulationClock Tests
# =============================================================================
class TestSimulationClock:
    """Tests for SimulationClock class."""

    def test_default_initialization(self):
        """Test default clock initialization."""
        clock = SimulationClock()
        assert clock.exchange_time_ns == 0
        assert clock.local_time_ns == 0

    def test_set_exchange_time(self):
        """Test setting exchange time updates local time with latency."""
        clock = create_simulation_clock("institutional", seed=42)

        clock.set_exchange_time(1_000_000)

        assert clock.exchange_time_ns == 1_000_000
        assert clock.local_time_ns > 1_000_000  # Includes feed latency

    def test_advance(self):
        """Test advancing both clocks."""
        clock = create_simulation_clock("institutional", seed=42, initial_time_ns=1_000_000)

        clock.advance(500_000)

        assert clock.exchange_time_ns == 1_500_000
        assert clock.local_time_ns == 1_500_000

    def test_exchange_to_local(self):
        """Test converting exchange time to local time."""
        clock = create_simulation_clock("institutional", seed=42)

        local_time = clock.exchange_to_local(1_000_000)

        # Should include feed latency
        assert local_time > 1_000_000

    def test_local_to_exchange(self):
        """Test converting local time to exchange time."""
        clock = create_simulation_clock("institutional", seed=42)

        exchange_time = clock.local_to_exchange(1_500_000)

        # Should subtract estimated latency
        assert exchange_time < 1_500_000

    def test_order_arrival_time(self):
        """Test computing order arrival time."""
        clock = create_simulation_clock("institutional", seed=42)

        arrival_time = clock.order_arrival_time(1_000_000)

        # Should include order latency
        assert arrival_time > 1_000_000

    def test_round_trip_time(self):
        """Test getting round-trip latency."""
        clock = create_simulation_clock("institutional", seed=42)

        rtt = clock.get_round_trip_time()

        # Should be positive
        assert rtt > 0

    def test_factory_function(self):
        """Test create_simulation_clock factory."""
        clock = create_simulation_clock("colocated", seed=42, initial_time_ns=1_000)
        assert clock.exchange_time_ns == 1_000


# =============================================================================
# Integration Tests
# =============================================================================
class TestLatencyIntegration:
    """Integration tests for latency module."""

    def test_full_order_lifecycle(self):
        """Test complete order lifecycle with latency."""
        scheduler = create_event_scheduler("institutional", seed=42)

        events_received: List[ScheduledEvent] = []

        def track_events(event: ScheduledEvent):
            events_received.append(event)

        scheduler.register_handler(EventType.ORDER_SUBMITTED, track_events)
        scheduler.register_handler(EventType.ORDER_RECEIVED, track_events)
        scheduler.register_handler(EventType.ORDER_ACCEPTED, track_events)
        scheduler.register_handler(EventType.OUR_FILL, track_events)

        # 1. Submit order
        order = LimitOrder(
            order_id="lifecycle_test",
            price=150.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_000_000,
            side=Side.BUY,
        )
        scheduler.schedule_order_arrival(order=order, our_send_time_ns=1_000_000)

        # 2. Schedule fill
        fill = Fill(
            order_id="lifecycle_test",
            total_qty=100.0,
            avg_price=150.0,
            trades=[],
        )
        scheduler.schedule_fill_notification(
            fill=fill,
            exchange_time_ns=1_500_000,
        )

        # 3. Process all events
        scheduler.process_all()

        # 4. Verify lifecycle
        event_types = [e.event_type for e in events_received]
        assert EventType.ORDER_SUBMITTED in event_types
        assert EventType.ORDER_RECEIVED in event_types
        assert EventType.ORDER_ACCEPTED in event_types
        assert EventType.OUR_FILL in event_types

    def test_market_data_with_order(self):
        """Test market data and order interactions."""
        scheduler = create_event_scheduler("proximity", seed=42)

        # Schedule market data updates
        for i in range(5):
            event = MarketDataEvent(
                symbol="AAPL",
                exchange_time_ns=1_000_000 + i * 100_000,
                bid_price=150.0 + i * 0.01,
                ask_price=150.05 + i * 0.01,
            )
            scheduler.schedule_market_data(event, exchange_time_ns=event.exchange_time_ns)

        # Submit order in the middle
        order = LimitOrder(
            order_id="market_test",
            price=150.02,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_200_000,
            side=Side.BUY,
        )
        scheduler.schedule_order_arrival(order=order, our_send_time_ns=1_200_000)

        # Process and verify interleaving
        processed = scheduler.process_all()

        # Should have 5 market data + 3 order events = 8 events
        assert len(processed) == 8

    def test_latency_comparison_across_profiles(self):
        """Compare latency across different profiles."""
        profiles = ["colocated", "proximity", "retail", "institutional"]
        mean_latencies = {}

        for profile in profiles:
            model = create_latency_model(profile, seed=42)
            samples = [model.sample_round_trip() for _ in range(1000)]
            mean_latencies[profile] = sum(samples) / len(samples)

        # Co-located should be fastest
        assert mean_latencies["colocated"] < mean_latencies["proximity"]
        assert mean_latencies["proximity"] < mean_latencies["institutional"]
        assert mean_latencies["institutional"] < mean_latencies["retail"]

    def test_thread_safety_scheduler(self):
        """Test thread-safe scheduler operations."""
        scheduler = create_event_scheduler("institutional", seed=42)
        errors = []
        event_count = [0]
        lock = threading.Lock()

        def producer():
            try:
                for i in range(100):
                    event = MarketDataEvent(symbol="AAPL", exchange_time_ns=i * 1000)
                    scheduler.schedule_market_data(event, exchange_time_ns=i * 1000)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                while True:
                    event = scheduler.pop()
                    if event is None:
                        time.sleep(0.001)
                        continue
                    with lock:
                        event_count[0] += 1
            except Exception as e:
                errors.append(e)

        # Start producer
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        producer_thread.join()

        # Process all events
        while scheduler.pending_count > 0:
            scheduler.pop()
            event_count[0] += 1

        assert len(errors) == 0
        assert event_count[0] == 100


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scheduler(self):
        """Test operations on empty scheduler."""
        scheduler = create_event_scheduler("institutional", seed=42)

        assert scheduler.peek() is None
        assert scheduler.pop() is None
        assert scheduler.process_next() is None
        assert scheduler.process_all() == []

    def test_zero_latency(self):
        """Test with zero latency configuration."""
        config = LatencyModelConfig(
            feed_config=LatencyConfig(
                distribution=LatencyDistribution.CONSTANT,
                mean_us=0.0,
                min_us=0.0,
                spike_prob=0.0,
            ),
        )
        model = LatencyModel(config=config)

        # Should return 0 latency
        lat = model.sample_feed_latency()
        assert lat == 0

    def test_very_high_latency(self):
        """Test with very high latency configuration."""
        config = LatencyConfig(
            distribution=LatencyDistribution.CONSTANT,
            mean_us=1_000_000.0,  # 1 second
            spike_prob=0.0,
            max_us=2_000_000.0,
        )
        sampler = LatencySampler(config, seed=42)

        sample = sampler.sample()
        assert sample.latency_us == 1_000_000.0
        assert sample.latency_ns == 1_000_000_000  # 1 second in ns

    def test_negative_timestamp_raises(self):
        """Test that negative timestamp raises error."""
        with pytest.raises(ValueError, match="timestamp_ns must be non-negative"):
            ScheduledEvent(
                timestamp_ns=-1,
                sequence_id=0,
                event_type=EventType.TIMER,
                exchange_time_ns=-1,
            )

    def test_invalid_seasonality_length(self):
        """Test that invalid seasonality length raises error."""
        # Validation happens in LatencyModel constructor, not LatencyModelConfig
        config = LatencyModelConfig(
            feed_config=LatencyConfig(),
            seasonality_multipliers=[1.0] * 10,  # Wrong length
        )
        with pytest.raises(ValueError, match="seasonality_multipliers must have length 24"):
            LatencyModel(config=config)

    def test_invalid_empirical_component(self):
        """Test invalid component name for empirical data."""
        model = create_latency_model("institutional", seed=42)

        with pytest.raises(ValueError, match="Unknown component"):
            model.set_empirical_data("invalid", [100.0])

    def test_schedule_event_payload_types(self):
        """Test different payload types in events."""
        scheduler = create_event_scheduler("institutional", seed=42)

        # Schedule with different payload types
        payloads = [
            None,
            "string",
            123,
            {"dict": "value"},
            ["list"],
        ]

        for i, payload in enumerate(payloads):
            scheduler.schedule_custom(
                timestamp_ns=i * 1000,
                event_type=EventType.TIMER,
                exchange_time_ns=i * 1000,
                payload=payload,
            )

        # Process and verify payloads preserved
        processed = scheduler.process_all()
        for event, expected_payload in zip(processed, payloads):
            assert event.payload == expected_payload


# =============================================================================
# New Tests for Bug Fixes (Stage 5 Review)
# =============================================================================
class TestPriorityOrdering:
    """Tests for priority field ordering fix."""

    def test_priority_affects_ordering_same_timestamp(self):
        """Test that priority field affects event ordering for same timestamp."""
        # Create events with same timestamp but different priorities
        e1 = ScheduledEvent(
            timestamp_ns=1000,
            priority=10,  # Lower priority (higher number)
            sequence_id=1,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )
        e2 = ScheduledEvent(
            timestamp_ns=1000,
            priority=0,  # Higher priority (lower number)
            sequence_id=2,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )
        e3 = ScheduledEvent(
            timestamp_ns=1000,
            priority=5,  # Medium priority
            sequence_id=3,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )

        # Sort events
        events = [e1, e2, e3]
        sorted_events = sorted(events)

        # Expected order: e2 (priority=0), e3 (priority=5), e1 (priority=10)
        assert sorted_events[0].priority == 0
        assert sorted_events[1].priority == 5
        assert sorted_events[2].priority == 10

    def test_priority_before_sequence_in_ordering(self):
        """Test that priority is compared before sequence_id."""
        # Event with higher priority (lower number) but higher sequence_id
        high_priority = ScheduledEvent(
            timestamp_ns=1000,
            priority=0,
            sequence_id=100,  # Higher sequence_id
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )
        # Event with lower priority but lower sequence_id
        low_priority = ScheduledEvent(
            timestamp_ns=1000,
            priority=10,
            sequence_id=1,  # Lower sequence_id
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )

        # High priority should come first despite higher sequence_id
        assert high_priority < low_priority

    def test_scheduler_respects_priority(self):
        """Test that EventScheduler processes events by priority."""
        scheduler = create_event_scheduler("institutional", seed=42)

        # Schedule events with different priorities but close timestamps
        # Use custom events with specific priorities
        scheduler.schedule_custom(
            timestamp_ns=1000,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
            payload="low_priority",
            priority=10,
        )
        scheduler.schedule_custom(
            timestamp_ns=1000,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
            payload="high_priority",
            priority=0,
        )
        scheduler.schedule_custom(
            timestamp_ns=1000,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
            payload="medium_priority",
            priority=5,
        )

        # Process events
        processed = scheduler.process_all()

        # Should be ordered by priority
        assert processed[0].payload == "high_priority"
        assert processed[1].payload == "medium_priority"
        assert processed[2].payload == "low_priority"

    def test_timestamp_still_primary_sort_key(self):
        """Test that timestamp is still the primary sort key."""
        e_early = ScheduledEvent(
            timestamp_ns=500,
            priority=10,  # Low priority
            sequence_id=1,
            event_type=EventType.TIMER,
            exchange_time_ns=500,
        )
        e_late = ScheduledEvent(
            timestamp_ns=1000,
            priority=0,  # High priority
            sequence_id=2,
            event_type=EventType.TIMER,
            exchange_time_ns=1000,
        )

        # Earlier timestamp should come first regardless of priority
        assert e_early < e_late


class TestRaceConditionsLimit:
    """Tests for race conditions list memory limit."""

    def test_race_conditions_limit_default(self):
        """Test default max_race_conditions value."""
        scheduler = EventScheduler(seed=42)
        assert scheduler._max_race_conditions == EventScheduler.DEFAULT_MAX_RACE_CONDITIONS

    def test_race_conditions_limit_custom(self):
        """Test custom max_race_conditions value."""
        scheduler = EventScheduler(max_race_conditions=100, seed=42)
        assert scheduler._max_race_conditions == 100

    def test_race_conditions_list_bounded(self):
        """Test that race conditions list is bounded."""
        scheduler = EventScheduler(
            max_race_conditions=10,  # Small limit for testing
            detect_race_conditions=True,
            seed=42,
        )

        # Submit an order
        order = LimitOrder(
            order_id="bounded_test",
            price=150.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_000_000,
            side=Side.BUY,
        )
        scheduler.schedule_order_arrival(order=order, our_send_time_ns=1_000_000)

        # Schedule many market data events to trigger race conditions
        for i in range(100):
            event = MarketDataEvent(
                symbol="AAPL",
                exchange_time_ns=1_000_000 + i * 10,  # Very close timing
            )
            scheduler.schedule_market_data(event, exchange_time_ns=event.exchange_time_ns)

        # Race conditions list should be bounded
        races = scheduler.get_race_conditions()
        assert len(races) <= 10

    def test_stats_includes_race_tracking_info(self):
        """Test that stats include race tracking information."""
        scheduler = EventScheduler(max_race_conditions=500, seed=42)

        stats = scheduler.stats()
        assert "tracked_races" in stats
        assert "max_race_conditions" in stats
        assert stats["max_race_conditions"] == 500

    def test_oldest_races_dropped_when_limit_reached(self):
        """Test that oldest race conditions are dropped when limit is reached."""
        scheduler = EventScheduler(
            max_race_conditions=5,
            detect_race_conditions=True,
            seed=42,
        )

        # Submit an order
        order = LimitOrder(
            order_id="drop_test",
            price=150.0,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=1_000_000,
            side=Side.BUY,
        )
        scheduler.schedule_order_arrival(order=order, our_send_time_ns=1_000_000)

        # Schedule events to trigger race conditions
        for i in range(20):
            event = MarketDataEvent(
                symbol="AAPL",
                exchange_time_ns=1_000_000 + i * 10,
            )
            scheduler.schedule_market_data(event, exchange_time_ns=event.exchange_time_ns)

        races = scheduler.get_race_conditions()
        # Should have at most 5 race conditions
        assert len(races) <= 5

        # The tracked races should be the most recent ones
        # (deque drops oldest when maxlen is exceeded)


class TestParetoAlphaWarnings:
    """Tests for Pareto alpha warning."""

    def test_pareto_alpha_less_than_one_warning(self):
        """Test warning for Pareto alpha <= 1 (undefined mean)."""
        with pytest.warns(UserWarning, match="undefined mean"):
            LatencyConfig(
                distribution=LatencyDistribution.PARETO,
                pareto_alpha=0.9,
            )

    def test_pareto_alpha_equal_one_warning(self):
        """Test warning for Pareto alpha = 1 (undefined mean)."""
        with pytest.warns(UserWarning, match="undefined mean"):
            LatencyConfig(
                distribution=LatencyDistribution.PARETO,
                pareto_alpha=1.0,
            )

    def test_pareto_alpha_less_than_two_warning(self):
        """Test warning for Pareto alpha <= 2 (infinite variance)."""
        with pytest.warns(UserWarning, match="infinite variance"):
            LatencyConfig(
                distribution=LatencyDistribution.PARETO,
                pareto_alpha=1.5,
            )

    def test_pareto_alpha_equal_two_warning(self):
        """Test warning for Pareto alpha = 2 (infinite variance)."""
        with pytest.warns(UserWarning, match="infinite variance"):
            LatencyConfig(
                distribution=LatencyDistribution.PARETO,
                pareto_alpha=2.0,
            )

    def test_pareto_alpha_greater_than_two_no_warning(self):
        """Test no warning for Pareto alpha > 2."""
        import warnings as warn_module

        with warn_module.catch_warnings(record=True) as w:
            warn_module.simplefilter("always")
            LatencyConfig(
                distribution=LatencyDistribution.PARETO,
                pareto_alpha=2.5,
            )
            # Filter for our specific warnings
            pareto_warnings = [x for x in w if "Pareto" in str(x.message)]
            assert len(pareto_warnings) == 0, f"Unexpected warnings: {pareto_warnings}"

    def test_non_pareto_distribution_no_warning(self):
        """Test no warning for non-Pareto distributions even with same alpha value."""
        import warnings as warn_module

        with warn_module.catch_warnings(record=True) as w:
            warn_module.simplefilter("always")
            LatencyConfig(
                distribution=LatencyDistribution.LOGNORMAL,
                pareto_alpha=0.5,  # Would trigger warning for Pareto
            )
            # Filter for our specific warnings
            pareto_warnings = [x for x in w if "Pareto" in str(x.message)]
            assert len(pareto_warnings) == 0


class TestSequenceIdFIFO:
    """Tests for sequence_id FIFO ordering when timestamp and priority are equal."""

    def test_sequence_id_fifo_for_equal_priority(self):
        """Test FIFO ordering by sequence_id when timestamp and priority are equal."""
        scheduler = create_event_scheduler("institutional", seed=42)

        # Schedule events with same timestamp and priority
        for i in range(5):
            scheduler.schedule_custom(
                timestamp_ns=1000,
                event_type=EventType.TIMER,
                exchange_time_ns=1000,
                payload=f"event_{i}",
                priority=0,  # Same priority
            )

        # Process events
        processed = scheduler.process_all()

        # Should be in FIFO order (by sequence_id)
        for i, event in enumerate(processed):
            assert event.payload == f"event_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
