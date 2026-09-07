# -*- coding: utf-8 -*-
"""
tests/test_futures_funding.py
Comprehensive tests for futures funding rate mechanics.

Tests cover:
- FundingRateTracker: Core tracking and calculation logic
- FundingRateSimulator: Simulation modes (historical, constant, random_walk)
- FundingTrackerService: High-level service functionality
- Edge cases: Pro-rata, timing, sign conventions

Target: 50+ tests for Phase 3A validation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List
import tempfile
import os

from impl_futures_funding import (
    FundingRateTracker,
    FundingRateSimulator,
    FundingRateRecord,
    FundingStatistics,
    FUNDING_PERIOD_MS,
    FUNDING_PERIODS_PER_DAY,
    FUNDING_TIMES_UTC,
    DEFAULT_NEUTRAL_RATE,
    FUNDING_RATE_MAX,
    FUNDING_RATE_MIN,
    create_funding_tracker,
    create_funding_simulator,
    calculate_funding_payment_simple,
    annualize_funding_rate,
    funding_rate_to_bps,
    bps_to_funding_rate,
)
from core_futures import (
    FundingPayment,
    FuturesPosition,
    PositionSide,
    MarginMode,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tracker():
    """Create a fresh funding rate tracker."""
    return create_funding_tracker(max_history=1000)


@pytest.fixture
def long_position():
    """Create a standard long position for testing."""
    return FuturesPosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=Decimal("50000"),
        qty=Decimal("1.0"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
    )


@pytest.fixture
def short_position():
    """Create a standard short position for testing."""
    return FuturesPosition(
        symbol="BTCUSDT",
        side=PositionSide.SHORT,
        entry_price=Decimal("50000"),
        qty=Decimal("-1.0"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
    )


@pytest.fixture
def funding_time_utc():
    """Get a standard funding time (00:00 UTC today)."""
    now = datetime.now(tz=timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@pytest.fixture
def sample_funding_history():
    """Create sample funding history data."""
    base_ts = int(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    records = []
    for i in range(10):
        ts = base_ts + i * FUNDING_PERIOD_MS
        rate = Decimal("0.0001") + Decimal(str(i * 0.00001))
        records.append(
            FundingRateRecord(
                symbol="BTCUSDT",
                timestamp_ms=ts,
                funding_rate=rate,
                mark_price=Decimal("50000") + Decimal(str(i * 100)),
            )
        )
    return records


# ============================================================================
# FUNDING RATE TRACKER TESTS
# ============================================================================


class TestFundingRateTrackerBasic:
    """Basic functionality tests for FundingRateTracker."""

    def test_create_tracker(self):
        """Test tracker creation."""
        tracker = create_funding_tracker()
        assert tracker is not None
        assert tracker.get_symbols() == []

    def test_add_funding_rate(self, tracker):
        """Test adding a single funding rate."""
        ts = 1704067200000  # 2024-01-01 00:00 UTC
        tracker.add_funding_rate(
            symbol="BTCUSDT",
            timestamp_ms=ts,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
        )
        assert tracker.get_history_count("BTCUSDT") == 1
        assert "BTCUSDT" in tracker.get_symbols()

    def test_add_multiple_funding_rates(self, tracker):
        """Test adding multiple funding rates."""
        base_ts = 1704067200000
        for i in range(5):
            tracker.add_funding_rate(
                symbol="BTCUSDT",
                timestamp_ms=base_ts + i * FUNDING_PERIOD_MS,
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal("50000"),
            )
        assert tracker.get_history_count("BTCUSDT") == 5

    def test_add_funding_records_batch(self, tracker, sample_funding_history):
        """Test batch adding funding records."""
        count = tracker.add_funding_records(sample_funding_history)
        assert count == 10
        assert tracker.get_history_count("BTCUSDT") == 10

    def test_get_funding_rate_exact(self, tracker):
        """Test retrieving funding rate at exact timestamp."""
        ts = 1704067200000
        rate = Decimal("0.0001")
        tracker.add_funding_rate("BTCUSDT", ts, rate, Decimal("50000"))

        record = tracker.get_funding_rate("BTCUSDT", ts)
        assert record is not None
        assert record.funding_rate == rate

    def test_get_funding_rate_not_found(self, tracker):
        """Test retrieving funding rate at non-existent timestamp."""
        tracker.add_funding_rate("BTCUSDT", 1704067200000, Decimal("0.0001"), Decimal("50000"))
        record = tracker.get_funding_rate("BTCUSDT", 1704067200001)
        assert record is None

    def test_get_funding_rate_at_or_before(self, tracker):
        """Test retrieving funding rate at or before timestamp."""
        ts1 = 1704067200000
        ts2 = 1704067200000 + FUNDING_PERIOD_MS
        tracker.add_funding_rate("BTCUSDT", ts1, Decimal("0.0001"), Decimal("50000"))
        tracker.add_funding_rate("BTCUSDT", ts2, Decimal("0.0002"), Decimal("50100"))

        # Query between two funding times
        query_ts = ts1 + FUNDING_PERIOD_MS // 2
        record = tracker.get_funding_rate_at_or_before("BTCUSDT", query_ts)
        assert record is not None
        assert record.funding_rate == Decimal("0.0001")

    def test_get_funding_rates_range(self, tracker, sample_funding_history):
        """Test retrieving funding rates within time range."""
        tracker.add_funding_records(sample_funding_history)

        start_ms = sample_funding_history[2].timestamp_ms
        end_ms = sample_funding_history[7].timestamp_ms

        records = tracker.get_funding_rates_range("BTCUSDT", start_ms, end_ms)
        assert len(records) == 6

    def test_clear_history_symbol(self, tracker):
        """Test clearing history for specific symbol."""
        tracker.add_funding_rate("BTCUSDT", 1704067200000, Decimal("0.0001"), Decimal("50000"))
        tracker.add_funding_rate("ETHUSDT", 1704067200000, Decimal("0.0001"), Decimal("2500"))

        tracker.clear_history("BTCUSDT")
        assert tracker.get_history_count("BTCUSDT") == 0
        assert tracker.get_history_count("ETHUSDT") == 1

    def test_clear_history_all(self, tracker):
        """Test clearing all history."""
        tracker.add_funding_rate("BTCUSDT", 1704067200000, Decimal("0.0001"), Decimal("50000"))
        tracker.add_funding_rate("ETHUSDT", 1704067200000, Decimal("0.0001"), Decimal("2500"))

        tracker.clear_history()
        assert tracker.get_symbols() == []

    def test_max_history_limit(self):
        """Test that history respects max limit."""
        tracker = create_funding_tracker(max_history=5)
        base_ts = 1704067200000

        for i in range(10):
            tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                Decimal("0.0001"),
                Decimal("50000"),
            )

        assert tracker.get_history_count("BTCUSDT") == 5

    def test_symbol_case_insensitive(self, tracker):
        """Test symbol names are case-insensitive."""
        tracker.add_funding_rate("btcusdt", 1704067200000, Decimal("0.0001"), Decimal("50000"))
        assert tracker.get_history_count("BTCUSDT") == 1
        assert tracker.get_history_count("btcusdt") == 1


# ============================================================================
# FUNDING PAYMENT CALCULATION TESTS
# ============================================================================


class TestFundingPaymentCalculation:
    """Tests for funding payment calculation."""

    def test_long_pays_positive_funding(self, tracker, long_position):
        """Long position pays when funding rate is positive."""
        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),  # 0.01%
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        # Position value = 50000 * 1 = 50000
        # Payment = -50000 * 0.0001 = -5 (pays $5)
        assert payment.payment_amount == Decimal("-5")

    def test_short_receives_positive_funding(self, tracker, short_position):
        """Short position receives when funding rate is positive."""
        payment = tracker.calculate_funding_payment(
            position=short_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        # Short receives $5
        assert payment.payment_amount == Decimal("5")

    def test_long_receives_negative_funding(self, tracker, long_position):
        """Long position receives when funding rate is negative."""
        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("-0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        # Long receives $5 when funding is negative
        assert payment.payment_amount == Decimal("5")

    def test_short_pays_negative_funding(self, tracker, short_position):
        """Short position pays when funding rate is negative."""
        payment = tracker.calculate_funding_payment(
            position=short_position,
            funding_rate=Decimal("-0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        # Short pays $5 when funding is negative
        assert payment.payment_amount == Decimal("-5")

    def test_zero_position_zero_payment(self, tracker):
        """Zero position results in zero payment."""
        zero_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.BOTH,
            entry_price=Decimal("50000"),
            qty=Decimal("0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        payment = tracker.calculate_funding_payment(
            position=zero_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        assert payment.payment_amount == Decimal("0")

    def test_payment_scales_with_position_size(self, tracker):
        """Payment scales linearly with position size."""
        small_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0.5"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        large_position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("2.0"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        small_payment = tracker.calculate_funding_payment(
            small_position, Decimal("0.0001"), Decimal("50000"), 0
        )
        large_payment = tracker.calculate_funding_payment(
            large_position, Decimal("0.0001"), Decimal("50000"), 0
        )

        # Large position should pay 4x more than small
        assert large_payment.payment_amount == small_payment.payment_amount * 4

    def test_payment_attributes(self, tracker, long_position):
        """Test that payment contains correct attributes."""
        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=1704067200000,
        )
        assert payment.symbol == "BTCUSDT"
        assert payment.timestamp_ms == 1704067200000
        assert payment.funding_rate == Decimal("0.0001")
        assert payment.mark_price == Decimal("50000")
        assert payment.position_qty == Decimal("1.0")


# ============================================================================
# PRO-RATA FUNDING TESTS
# ============================================================================


class TestProRataFunding:
    """Tests for pro-rata funding calculation."""

    def test_full_period_prorate_factor_one(self, tracker, long_position):
        """Position held for full period has prorate factor of 1."""
        funding_time = 1704067200000

        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=funding_time,
            entry_time_ms=None,  # Held before period start
            exit_time_ms=None,  # Still open
        )
        # Full payment
        assert payment.payment_amount == Decimal("-5")

    def test_half_period_prorate_factor(self, tracker, long_position):
        """Position held for half period pays half."""
        funding_time = 1704067200000
        period_start = funding_time - FUNDING_PERIOD_MS
        half_way = period_start + FUNDING_PERIOD_MS // 2

        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=funding_time,
            entry_time_ms=half_way,  # Entered halfway through
        )
        # Should pay approximately half
        assert abs(payment.payment_amount - Decimal("-2.5")) < Decimal("0.01")

    def test_position_opened_after_funding_no_payment(self, tracker, long_position):
        """Position opened after funding time receives no payment."""
        funding_time = 1704067200000

        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=funding_time,
            entry_time_ms=funding_time + 1000,  # Opened after funding
        )
        assert payment.payment_amount == Decimal("0")

    def test_position_closed_before_period_no_payment(self, tracker, long_position):
        """Position closed before funding period receives no payment."""
        funding_time = 1704067200000
        period_start = funding_time - FUNDING_PERIOD_MS

        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            timestamp_ms=funding_time,
            entry_time_ms=period_start - FUNDING_PERIOD_MS,  # Opened way before
            exit_time_ms=period_start - 1000,  # Closed before period
        )
        assert payment.payment_amount == Decimal("0")

    def test_should_apply_funding_true(self, tracker):
        """Test should_apply_funding returns true for valid position."""
        funding_time = 1704067200000
        entry_time = funding_time - 1000  # Before funding

        result = tracker.should_apply_funding(
            position_entry_ms=entry_time,
            position_exit_ms=None,
            funding_time_ms=funding_time,
        )
        assert result is True

    def test_should_apply_funding_false_opened_at_funding(self, tracker):
        """Position opened at funding time should not receive payment."""
        funding_time = 1704067200000

        result = tracker.should_apply_funding(
            position_entry_ms=funding_time,  # Opened exactly at funding
            position_exit_ms=None,
            funding_time_ms=funding_time,
        )
        assert result is False

    def test_should_apply_funding_false_closed_before(self, tracker):
        """Position closed before funding time should not receive payment."""
        funding_time = 1704067200000

        result = tracker.should_apply_funding(
            position_entry_ms=funding_time - FUNDING_PERIOD_MS,
            position_exit_ms=funding_time - 1000,  # Closed before
            funding_time_ms=funding_time,
        )
        assert result is False


# ============================================================================
# FUNDING TIME UTILITIES TESTS
# ============================================================================


class TestFundingTimeUtilities:
    """Tests for funding time calculation utilities."""

    def test_get_next_funding_time_before_first(self, tracker):
        """Test next funding when before first funding of day."""
        # 07:30 UTC -> next is 08:00 UTC
        dt = datetime(2024, 1, 1, 7, 30, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        next_funding = tracker.get_next_funding_time(ts_ms)
        next_dt = datetime.fromtimestamp(next_funding / 1000, tz=timezone.utc)

        assert next_dt.hour == 8
        assert next_dt.minute == 0

    def test_get_next_funding_time_after_last(self, tracker):
        """Test next funding when after last funding of day."""
        # 17:30 UTC -> next is 00:00 next day
        dt = datetime(2024, 1, 1, 17, 30, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        next_funding = tracker.get_next_funding_time(ts_ms)
        next_dt = datetime.fromtimestamp(next_funding / 1000, tz=timezone.utc)

        assert next_dt.day == 2
        assert next_dt.hour == 0
        assert next_dt.minute == 0

    def test_get_previous_funding_time(self, tracker):
        """Test previous funding time calculation."""
        # 10:30 UTC -> previous is 08:00 UTC
        dt = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        prev_funding = tracker.get_previous_funding_time(ts_ms)
        prev_dt = datetime.fromtimestamp(prev_funding / 1000, tz=timezone.utc)

        assert prev_dt.hour == 8
        assert prev_dt.minute == 0

    def test_is_funding_time_exact(self, tracker):
        """Test is_funding_time with exact match."""
        dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        assert tracker.is_funding_time(ts_ms) is True

    def test_is_funding_time_within_tolerance(self, tracker):
        """Test is_funding_time within tolerance."""
        dt = datetime(2024, 1, 1, 8, 0, 30, tzinfo=timezone.utc)  # 30 seconds after
        ts_ms = int(dt.timestamp() * 1000)

        assert tracker.is_funding_time(ts_ms, tolerance_ms=60_000) is True

    def test_is_funding_time_outside_tolerance(self, tracker):
        """Test is_funding_time outside tolerance."""
        dt = datetime(2024, 1, 1, 8, 5, 0, tzinfo=timezone.utc)  # 5 minutes after
        ts_ms = int(dt.timestamp() * 1000)

        assert tracker.is_funding_time(ts_ms, tolerance_ms=60_000) is False

    def test_get_funding_times_in_range(self, tracker):
        """Test getting all funding times within range."""
        # One full day
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 23, 59, tzinfo=timezone.utc)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        funding_times = tracker.get_funding_times_in_range(start_ms, end_ms)

        # Should have 3 funding times: 00:00, 08:00, 16:00
        assert len(funding_times) == 3

    def test_time_to_next_funding_ms(self, tracker):
        """Test time until next funding."""
        dt = datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        time_to_funding = tracker.time_to_next_funding_ms(ts_ms)

        # 7:00 to 8:00 = 1 hour = 3,600,000 ms
        assert time_to_funding == 3600000


# ============================================================================
# FUNDING COST ESTIMATION TESTS
# ============================================================================


class TestFundingCostEstimation:
    """Tests for funding cost estimation."""

    def test_estimate_daily_funding_cost(self, tracker, long_position):
        """Test daily funding cost estimation."""
        daily_cost = tracker.estimate_daily_funding_cost(
            position=long_position,
            avg_funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
        )
        # 3 funding periods per day
        # Each period: -5 USDT
        # Daily: -15 USDT
        assert daily_cost == Decimal("-15")

    def test_estimate_funding_cost_hours(self, tracker, long_position):
        """Test funding cost estimation over hours."""
        cost_24h = tracker.estimate_funding_cost(
            position=long_position,
            avg_funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            hours=24,
        )
        # 24 hours = 3 periods = -15 USDT
        assert cost_24h == Decimal("-15")

        cost_8h = tracker.estimate_funding_cost(
            position=long_position,
            avg_funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
            hours=8,
        )
        # 8 hours = 1 period = -5 USDT
        assert cost_8h == Decimal("-5")

    def test_get_average_funding_rate(self, tracker, sample_funding_history):
        """Test average funding rate calculation."""
        tracker.add_funding_records(sample_funding_history)

        # Use a current_ts_ms after the test data timestamps
        last_ts = sample_funding_history[-1].timestamp_ms
        current_ts_ms = last_ts + 1000  # Just after last record

        avg_rate = tracker.get_average_funding_rate(
            "BTCUSDT", lookback_hours=1000, current_ts_ms=current_ts_ms
        )

        # Average of rates from sample history
        expected_avg = sum(r.funding_rate for r in sample_funding_history) / len(
            sample_funding_history
        )
        assert abs(avg_rate - expected_avg) < Decimal("0.000001")

    def test_get_funding_statistics(self, tracker, sample_funding_history):
        """Test funding statistics calculation."""
        tracker.add_funding_records(sample_funding_history)

        # Use a current_ts_ms after the test data timestamps
        last_ts = sample_funding_history[-1].timestamp_ms
        current_ts_ms = last_ts + 1000  # Just after last record

        stats = tracker.get_funding_statistics(
            "BTCUSDT", lookback_hours=1000, current_ts_ms=current_ts_ms
        )

        assert stats.symbol == "BTCUSDT"
        assert stats.count == 10
        assert stats.avg_rate > 0
        assert stats.positive_count == 10


# ============================================================================
# FUNDING RATE PREDICTION TESTS
# ============================================================================


class TestFundingRatePrediction:
    """Tests for funding rate prediction."""

    def test_predict_last(self, tracker, sample_funding_history):
        """Test prediction using last rate method."""
        tracker.add_funding_records(sample_funding_history)

        predicted = tracker.predict_next_funding_rate(
            symbol="BTCUSDT",
            method="last",
            lookback_periods=8,
        )
        # Should return last rate
        assert predicted == sample_funding_history[-1].funding_rate

    def test_predict_avg(self, tracker, sample_funding_history):
        """Test prediction using average method."""
        tracker.add_funding_records(sample_funding_history)

        predicted = tracker.predict_next_funding_rate(
            symbol="BTCUSDT",
            method="avg",
            lookback_periods=8,
        )
        # Should be average of last 8 rates
        last_8 = sample_funding_history[-8:]
        expected = sum(r.funding_rate for r in last_8) / 8
        assert abs(predicted - expected) < Decimal("0.000001")

    def test_predict_ewma(self, tracker, sample_funding_history):
        """Test prediction using EWMA method."""
        tracker.add_funding_records(sample_funding_history)

        predicted = tracker.predict_next_funding_rate(
            symbol="BTCUSDT",
            method="ewma",
            lookback_periods=8,
        )
        # EWMA should be between min and max
        min_rate = min(r.funding_rate for r in sample_funding_history[-8:])
        max_rate = max(r.funding_rate for r in sample_funding_history[-8:])
        assert min_rate <= predicted <= max_rate

    def test_predict_no_history_returns_default(self, tracker):
        """Test prediction with no history returns default rate."""
        predicted = tracker.predict_next_funding_rate(
            symbol="BTCUSDT",
            method="last",
        )
        assert predicted == DEFAULT_NEUTRAL_RATE


# ============================================================================
# FUNDING RATE SIMULATOR TESTS
# ============================================================================


class TestFundingRateSimulator:
    """Tests for FundingRateSimulator."""

    def test_constant_mode(self):
        """Test constant rate simulation mode."""
        constant_rate = Decimal("0.0002")
        sim = create_funding_simulator(
            mode="constant",
            constant_rate=constant_rate,
        )

        rate = sim.get_funding_rate("BTCUSDT", 1704067200000)
        assert rate == constant_rate

    def test_historical_mode(self, tracker, sample_funding_history):
        """Test historical simulation mode."""
        tracker.add_funding_records(sample_funding_history)

        sim = create_funding_simulator(
            mode="historical",
            tracker=tracker,
        )

        ts = sample_funding_history[5].timestamp_ms
        rate = sim.get_funding_rate("BTCUSDT", ts)
        assert rate == sample_funding_history[5].funding_rate

    def test_random_walk_mode(self):
        """Test random walk simulation mode."""
        sim = FundingRateSimulator(
            mode="random_walk",
            mean_rate=Decimal("0.0001"),
            volatility=Decimal("0.0001"),
            seed=42,
        )

        rates = [sim.get_funding_rate("BTCUSDT", i * 1000) for i in range(100)]

        # Rates should vary
        unique_rates = len(set(rates))
        assert unique_rates > 1

        # Rates should be within bounds
        for rate in rates:
            assert FUNDING_RATE_MIN <= rate <= FUNDING_RATE_MAX

    def test_simulator_reset(self):
        """Test simulator state reset."""
        sim = FundingRateSimulator(
            mode="random_walk",
            seed=42,
        )

        # Generate some rates
        rate1 = sim.get_funding_rate("BTCUSDT", 1000)
        sim.get_funding_rate("BTCUSDT", 2000)

        # Reset
        sim.reset("BTCUSDT")

        # First rate after reset should not depend on previous state
        # (but will still be deterministic due to seed)


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_funding_payment_simple_long(self):
        """Test simple payment calculation for long."""
        payment = calculate_funding_payment_simple(
            position_qty=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
        )
        assert payment == Decimal("-5")

    def test_calculate_funding_payment_simple_short(self):
        """Test simple payment calculation for short."""
        payment = calculate_funding_payment_simple(
            position_qty=Decimal("-1"),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50000"),
        )
        assert payment == Decimal("5")

    def test_annualize_funding_rate(self):
        """Test annualization of funding rate."""
        rate = Decimal("0.0001")  # 0.01% per 8h
        annualized = annualize_funding_rate(rate)
        # 0.0001 * 3 * 365 * 100 = 10.95%
        assert abs(annualized - Decimal("10.95")) < Decimal("0.01")

    def test_funding_rate_to_bps(self):
        """Test conversion to basis points."""
        rate = Decimal("0.0001")  # 0.01%
        bps = funding_rate_to_bps(rate)
        assert bps == Decimal("1")

    def test_bps_to_funding_rate(self):
        """Test conversion from basis points."""
        bps = Decimal("5")  # 5 bps
        rate = bps_to_funding_rate(bps)
        assert rate == Decimal("0.0005")


# ============================================================================
# EXPORT/IMPORT TESTS
# ============================================================================


class TestExportImport:
    """Tests for export/import functionality."""

    def test_export_history(self, tracker, sample_funding_history):
        """Test exporting funding history."""
        tracker.add_funding_records(sample_funding_history)

        exported = tracker.export_history("BTCUSDT")

        assert len(exported) == 10
        assert exported[0]["symbol"] == "BTCUSDT"
        assert "funding_rate" in exported[0]

    def test_import_history(self, tracker):
        """Test importing funding history."""
        records = [
            {
                "symbol": "BTCUSDT",
                "timestamp_ms": 1704067200000,
                "funding_rate": "0.0001",
                "mark_price": "50000",
            },
            {
                "symbol": "BTCUSDT",
                "timestamp_ms": 1704067200000 + FUNDING_PERIOD_MS,
                "funding_rate": "0.0002",
                "mark_price": "50100",
            },
        ]

        count = tracker.import_history(records)
        assert count == 2
        assert tracker.get_history_count("BTCUSDT") == 2


# ============================================================================
# FUNDING RATE RECORD TESTS
# ============================================================================


class TestFundingRateRecord:
    """Tests for FundingRateRecord dataclass."""

    def test_record_properties(self):
        """Test record property calculations."""
        record = FundingRateRecord(
            symbol="BTCUSDT",
            timestamp_ms=1704067200000,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50100"),
            index_price=Decimal("50000"),
        )

        assert record.rate_bps == Decimal("1")
        assert record.rate_pct == Decimal("0.01")
        assert record.annualized_rate == Decimal("0.0001") * 3 * 365 * 100
        # Premium = (50100 - 50000) / 50000 * 10000 = 20 bps
        assert record.premium_bps == Decimal("20")


# ============================================================================
# FUNDING STATISTICS TESTS
# ============================================================================


class TestFundingStatistics:
    """Tests for FundingStatistics dataclass."""

    def test_statistics_properties(self):
        """Test statistics property calculations."""
        stats = FundingStatistics(
            symbol="BTCUSDT",
            period_hours=24,
            count=10,
            avg_rate=Decimal("0.0001"),
            min_rate=Decimal("-0.0001"),
            max_rate=Decimal("0.0003"),
            std_rate=Decimal("0.00005"),
            positive_count=7,
            negative_count=2,
            total_annualized_rate=Decimal("10.95"),
            avg_mark_price=Decimal("50000"),
        )

        assert stats.positive_pct == Decimal("70")
        assert stats.negative_pct == Decimal("20")
        # Daily cost bps = avg_rate * 3 * 10000 = 0.0001 * 3 * 10000 = 3 bps
        assert stats.estimated_daily_cost_bps == Decimal("3")


# ============================================================================
# POSITION FUNDING CALCULATION TESTS
# ============================================================================


class TestPositionFundingCalculation:
    """Tests for calculating funding over position lifetime."""

    def test_calculate_position_funding_multiple_periods(self, tracker, long_position):
        """Test calculating funding for multiple periods."""
        base_ts = 1704067200000  # 00:00 UTC

        # Add 5 funding records
        for i in range(5):
            tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                Decimal("0.0001"),
                Decimal("50000"),
            )

        # Entry at least 8 hours before first funding to receive full first period
        entry_time = base_ts - FUNDING_PERIOD_MS  # 8 hours before first funding

        payments = tracker.calculate_position_funding(
            position=long_position,
            start_ms=base_ts,
            end_ms=base_ts + 4 * FUNDING_PERIOD_MS,
            entry_time_ms=entry_time,
        )

        assert len(payments) == 5
        total = sum(p.payment_amount for p in payments)
        assert total == Decimal("-25")  # 5 * -5 USDT

    def test_calculate_position_funding_with_entry_time(self, tracker, long_position):
        """Test funding calculation respecting entry time."""
        base_ts = 1704067200000

        # Add 3 funding records
        for i in range(3):
            tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                Decimal("0.0001"),
                Decimal("50000"),
            )

        # Position entered after first funding
        payments = tracker.calculate_position_funding(
            position=long_position,
            start_ms=base_ts,
            end_ms=base_ts + 2 * FUNDING_PERIOD_MS,
            entry_time_ms=base_ts + 1000,  # After first funding
        )

        # Should only get funding for periods 2 and 3
        assert len(payments) == 2


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_position(self, tracker):
        """Test with very small position size."""
        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0.001"),  # Tiny position
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        payment = tracker.calculate_funding_payment(
            position, Decimal("0.0001"), Decimal("50000"), 0
        )
        # 50000 * 0.001 * 0.0001 = 0.005
        assert abs(payment.payment_amount + Decimal("0.005")) < Decimal("0.0001")

    def test_very_large_position(self, tracker):
        """Test with very large position size."""
        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("1000"),  # Large position
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )
        payment = tracker.calculate_funding_payment(
            position, Decimal("0.0001"), Decimal("50000"), 0
        )
        # 50000 * 1000 * 0.0001 = 5000
        assert payment.payment_amount == Decimal("-5000")

    def test_extreme_positive_funding(self, tracker, long_position):
        """Test with maximum positive funding rate."""
        payment = tracker.calculate_funding_payment(
            position=long_position,
            funding_rate=FUNDING_RATE_MAX,  # 0.375%
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )
        # 50000 * 0.00375 = 187.5
        assert payment.payment_amount == Decimal("-187.5")

    def test_extreme_negative_funding(self, tracker, short_position):
        """Test with maximum negative funding rate."""
        payment = tracker.calculate_funding_payment(
            position=short_position,
            funding_rate=FUNDING_RATE_MIN,  # -0.375%
            mark_price=Decimal("50000"),
            timestamp_ms=0,
        )
        # Short pays when funding is negative
        # 50000 * 0.00375 = 187.5
        assert payment.payment_amount == Decimal("-187.5")

    def test_duplicate_timestamp_update(self, tracker):
        """Test that duplicate timestamps update the record."""
        ts = 1704067200000
        tracker.add_funding_rate("BTCUSDT", ts, Decimal("0.0001"), Decimal("50000"))
        tracker.add_funding_rate("BTCUSDT", ts, Decimal("0.0002"), Decimal("50100"))

        # Should still have only 1 record, but with updated value
        assert tracker.get_history_count("BTCUSDT") == 1
        record = tracker.get_funding_rate("BTCUSDT", ts)
        assert record.funding_rate == Decimal("0.0002")

    def test_multi_symbol_tracking(self, tracker):
        """Test tracking multiple symbols independently."""
        tracker.add_funding_rate("BTCUSDT", 1704067200000, Decimal("0.0001"), Decimal("50000"))
        tracker.add_funding_rate("ETHUSDT", 1704067200000, Decimal("0.0002"), Decimal("2500"))
        tracker.add_funding_rate("BNBUSDT", 1704067200000, Decimal("0.0003"), Decimal("300"))

        assert tracker.get_history_count("BTCUSDT") == 1
        assert tracker.get_history_count("ETHUSDT") == 1
        assert tracker.get_history_count("BNBUSDT") == 1
        assert len(tracker.get_symbols()) == 3


# ============================================================================
# FUNDING TRACKER SERVICE TESTS
# ============================================================================


class TestFundingTrackerService:
    """Tests for FundingTrackerService."""

    def test_create_service(self):
        """Test service creation."""
        from services.futures_funding_tracker import (
            FundingTrackerService,
            FundingTrackerConfig,
        )

        config = FundingTrackerConfig(auto_load=False)
        service = FundingTrackerService(config)
        assert service is not None

    def test_service_with_dataframe(self):
        """Test loading data via DataFrame."""
        import pandas as pd
        from services.futures_funding_tracker import (
            FundingTrackerService,
            FundingTrackerConfig,
        )

        config = FundingTrackerConfig(auto_load=False)
        service = FundingTrackerService(config)

        df = pd.DataFrame(
            {
                "ts_ms": [1704067200000, 1704067200000 + FUNDING_PERIOD_MS],
                "funding_rate": [0.0001, 0.0002],
                "mark_price": [50000, 50100],
            }
        )

        count = service.load_funding_dataframe(df, "BTCUSDT")
        assert count == 2
        assert service.is_data_loaded("BTCUSDT")

    def test_service_estimate_daily_cost(self, long_position):
        """Test daily cost estimation via service."""
        from services.futures_funding_tracker import (
            FundingTrackerService,
            FundingTrackerConfig,
        )

        config = FundingTrackerConfig(auto_load=False)
        service = FundingTrackerService(config)

        # Add some funding data
        for i in range(3):
            service.tracker.add_funding_rate(
                "BTCUSDT",
                1704067200000 + i * FUNDING_PERIOD_MS,
                Decimal("0.0001"),
                Decimal("50000"),
            )

        cost = service.estimate_daily_funding_cost(
            position=long_position,
            mark_price=Decimal("50000"),
            use_recent_avg=True,
        )
        assert cost == Decimal("-15")

    def test_service_get_statistics(self):
        """Test statistics via service."""
        from services.futures_funding_tracker import (
            FundingTrackerService,
            FundingTrackerConfig,
        )

        config = FundingTrackerConfig(auto_load=False)
        service = FundingTrackerService(config)

        # Add funding data
        base_ts = 1704067200000
        for i in range(5):
            service.tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                Decimal("0.0001") + Decimal(str(i * 0.00001)),
                Decimal("50000"),
            )

        # Use current_ts_ms after test data timestamps
        current_ts_ms = base_ts + 5 * FUNDING_PERIOD_MS

        stats = service.get_funding_statistics(
            "BTCUSDT", lookback_hours=1000, current_ts_ms=current_ts_ms
        )
        assert stats.count == 5
        assert stats.positive_count == 5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_funding_calculation_flow(self):
        """Test complete flow: add history -> calculate payments -> estimate costs."""
        from services.futures_funding_tracker import create_funding_service

        service = create_funding_service(data_dir="data/futures")

        # Add realistic funding history
        base_ts = int(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

        for i in range(30):  # 10 days of funding
            # Vary the funding rate realistically
            rate = Decimal("0.0001") + Decimal(str((i % 5) * 0.00005))
            service.tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                rate,
                Decimal("50000") + Decimal(str(i * 50)),
            )

        # Create a position
        position = FuturesPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            qty=Decimal("0.5"),
            leverage=10,
            margin_mode=MarginMode.CROSS,
        )

        # Calculate funding payments
        # Entry before first funding to receive all 30 payments
        entry_before_first = base_ts - 1000

        payments = service.calculate_position_funding(
            position,
            base_ts,
            base_ts + 29 * FUNDING_PERIOD_MS,
            entry_time_ms=entry_before_first,
        )

        assert len(payments) == 30

        # Calculate total cost
        total = service.get_total_funding_cost(
            position,
            base_ts,
            base_ts + 29 * FUNDING_PERIOD_MS,
        )

        # Should be negative (long pays positive funding)
        assert total < 0

        # Estimate future cost
        daily_estimate = service.estimate_daily_funding_cost(
            position,
            Decimal("50000"),
            use_recent_avg=True,
        )

        assert daily_estimate < 0

    def test_simulator_integration(self):
        """Test simulator integration with tracker."""
        tracker = create_funding_tracker()

        # Add history
        base_ts = 1704067200000
        for i in range(10):
            tracker.add_funding_rate(
                "BTCUSDT",
                base_ts + i * FUNDING_PERIOD_MS,
                Decimal("0.0001"),
                Decimal("50000"),
            )

        # Create simulator using historical data
        sim = create_funding_simulator(mode="historical", tracker=tracker)

        # Get rate at known timestamp
        rate = sim.get_funding_rate("BTCUSDT", base_ts + 5 * FUNDING_PERIOD_MS)
        assert rate == Decimal("0.0001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
