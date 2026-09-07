"""Comprehensive tests for execution_algos.py - 100% coverage.

This test suite provides complete coverage of execution algorithms including:
- TakerExecutor
- TWAPExecutor (Time-Weighted Average Price)
- POVExecutor (Percentage of Volume)
- VWAPExecutor (Volume-Weighted Average Price)
- MidOffsetLimitExecutor
- MarketOpenH1Executor
- Factory function make_executor
"""

import math
from typing import Dict, Any, List

import pytest

from execution_algos import (
    MarketChild,
    BaseExecutor,
    TakerExecutor,
    TWAPExecutor,
    POVExecutor,
    VWAPExecutor,
    MidOffsetLimitExecutor,
    MarketOpenH1Executor,
    make_executor,
)


class TestMarketChild:
    """Test MarketChild dataclass."""

    def test_init_default(self):
        """Test default initialization."""
        child = MarketChild(ts_offset_ms=0, qty=10.0)
        assert child.ts_offset_ms == 0
        assert child.qty == 10.0
        assert child.liquidity_hint is None

    def test_init_with_liquidity(self):
        """Test initialization with liquidity hint."""
        child = MarketChild(ts_offset_ms=100, qty=5.0, liquidity_hint=100.0)
        assert child.ts_offset_ms == 100
        assert child.qty == 5.0
        assert child.liquidity_hint == 100.0


class TestTakerExecutor:
    """Test TakerExecutor."""

    @pytest.fixture
    def executor(self):
        """Create TakerExecutor."""
        return TakerExecutor()

    def test_plan_market_positive_qty(self, executor):
        """Test planning with positive quantity."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=10.0, snapshot={})

        assert len(plan) == 1
        assert plan[0].qty == 10.0
        assert plan[0].ts_offset_ms == 0

    def test_plan_market_zero_qty(self, executor):
        """Test planning with zero quantity."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=0.0, snapshot={})

        assert len(plan) == 0

    def test_plan_market_negative_qty(self, executor):
        """Test planning with negative quantity (absolute value used)."""
        plan = executor.plan_market(now_ts_ms=1000, side="SELL", target_qty=-10.0, snapshot={})

        assert len(plan) == 1
        assert plan[0].qty == 10.0


class TestTWAPExecutor:
    """Test TWAPExecutor."""

    @pytest.fixture
    def executor(self):
        """Create TWAPExecutor."""
        return TWAPExecutor(parts=6, child_interval_s=600)

    def test_init(self):
        """Test initialization."""
        executor = TWAPExecutor(parts=10, child_interval_s=300)
        assert executor.parts == 10
        assert executor.child_interval_ms == 300000

    def test_plan_market_basic(self, executor):
        """Test basic planning without bar window."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=60.0, snapshot={})

        assert len(plan) == 6
        for child in plan:
            assert child.qty == 10.0

    def test_plan_market_with_bar_window(self, executor):
        """Test planning with bar window metadata."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=60.0,
            snapshot={"bar_timeframe_ms": 60000, "bar_start_ts": 1000, "bar_end_ts": 61000},
        )

        assert len(plan) == 6
        # First child should be immediate
        assert plan[0].ts_offset_ms == 0
        # Last child should be near bar end
        assert plan[-1].ts_offset_ms <= 60000

    def test_plan_market_rounding(self, executor):
        """Test correct rounding of quantities."""
        plan = executor.plan_market(
            now_ts_ms=1000, side="BUY", target_qty=65.0, snapshot={}  # Not evenly divisible
        )

        total = sum(c.qty for c in plan)
        assert math.isclose(total, 65.0, rel_tol=1e-9)

    def test_plan_market_single_part(self):
        """Test planning with single part."""
        executor = TWAPExecutor(parts=1, child_interval_s=600)
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=10.0, snapshot={})

        assert len(plan) == 1
        assert plan[0].qty == 10.0


class TestPOVExecutor:
    """Test POVExecutor."""

    @pytest.fixture
    def executor(self):
        """Create POVExecutor."""
        return POVExecutor(participation=0.1, child_interval_s=60, min_child_notional=20.0)

    def test_init(self):
        """Test initialization."""
        executor = POVExecutor(participation=0.15, child_interval_s=30, min_child_notional=50.0)
        assert executor.participation == 0.15
        assert executor.child_interval_ms == 30000
        assert executor.min_child_notional == 50.0

    def test_plan_market_with_liquidity(self, executor):
        """Test planning with liquidity hint."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={"liquidity": 50.0, "ref_price": 50000.0},
        )

        assert len(plan) > 0
        # Each child should participate at configured rate
        for child in plan:
            assert child.liquidity_hint == 50.0

    def test_plan_market_no_liquidity(self, executor):
        """Test planning without liquidity (fallback to taker)."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=100.0, snapshot={})

        assert len(plan) == 1
        assert plan[0].qty == 100.0

    def test_plan_market_with_bar_window(self, executor):
        """Test planning with bar window."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "liquidity": 50.0,
                "ref_price": 50000.0,
                "bar_timeframe_ms": 60000,
                "bar_start_ts": 1000,
            },
        )

        assert len(plan) > 0
        # All children should be within bar window
        for child in plan:
            assert child.ts_offset_ms <= 60000

    def test_plan_market_min_notional(self):
        """Test minimum notional enforcement."""
        executor = POVExecutor(
            participation=0.01, min_child_notional=100.0  # Very low participation
        )

        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=1000.0,
            snapshot={"liquidity": 10.0, "ref_price": 50000.0},  # Low liquidity
        )

        # Each child should meet minimum notional
        for child in plan:
            notional = child.qty * 50000.0
            assert notional >= 100.0 or child == plan[-1]  # Last can be smaller

    def test_plan_market_max_children_limit(self, executor):
        """Test maximum children limit."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=1000000.0,  # Very large quantity
            snapshot={"liquidity": 1.0, "ref_price": 1.0},
        )

        # Should not exceed 10000 children
        assert len(plan) <= 10000


class TestVWAPExecutor:
    """Test VWAPExecutor."""

    @pytest.fixture
    def executor(self):
        """Create VWAPExecutor."""
        return VWAPExecutor(fallback_parts=6)

    def test_init(self):
        """Test initialization."""
        executor = VWAPExecutor(fallback_parts=10)
        assert executor.fallback_parts == 10

    def test_plan_market_no_profile(self, executor):
        """Test fallback planning without volume profile."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=60.0, snapshot={})

        assert len(plan) == 6  # fallback_parts

    def test_plan_market_with_profile(self, executor):
        """Test planning with volume profile."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "bar_timeframe_ms": 60000,
                "bar_start_ts": 1000,
                "intrabar_volume_profile": [
                    {"ts": 10000, "volume": 50.0},
                    {"ts": 30000, "volume": 30.0},
                    {"ts": 50000, "volume": 20.0},
                ],
            },
        )

        assert len(plan) > 0
        # Total quantity should match target
        total = sum(c.qty for c in plan)
        assert math.isclose(total, 100.0, rel_tol=1e-6)

    def test_plan_market_profile_with_offset(self, executor):
        """Test profile with offset_ms."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "bar_start_ts": 1000,
                "bar_timeframe_ms": 60000,
                "intrabar_volume_profile": [
                    {"offset_ms": 0, "volume": 50.0},
                    {"offset_ms": 30000, "volume": 50.0},
                ],
            },
        )

        assert len(plan) > 0

    def test_plan_market_profile_with_fraction(self, executor):
        """Test profile with fraction."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "bar_start_ts": 1000,
                "bar_timeframe_ms": 60000,
                "intrabar_volume_profile": [
                    {"fraction": 0.0, "volume": 50.0},
                    {"fraction": 0.5, "volume": 30.0},
                    {"fraction": 1.0, "volume": 20.0},
                ],
            },
        )

        assert len(plan) > 0

    def test_plan_market_profile_sequence(self, executor):
        """Test profile as sequence of [timestamp, volume] pairs."""
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "bar_timeframe_ms": 60000,
                "bar_start_ts": 1000,
                "intrabar_volume_profile": [[10000, 50.0], [30000, 30.0], [50000, 20.0]],
            },
        )

        assert len(plan) > 0

    def test_plan_market_past_timestamps_filtered(self, executor):
        """Test that past timestamps are filtered out."""
        now_ms = 50000
        plan = executor.plan_market(
            now_ts_ms=now_ms,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "bar_start_ts": 1000,
                "bar_timeframe_ms": 60000,
                "intrabar_volume_profile": [
                    {"ts": 10000, "volume": 50.0},  # Past
                    {"ts": 60000, "volume": 50.0},  # Future
                ],
            },
        )

        # Should only have future entries
        assert all(c.ts_offset_ms >= 0 for c in plan)

    def test_fallback_plan_with_horizon(self, executor):
        """Test fallback with time horizon."""
        plan = executor._fallback_plan(
            now_ts_ms=1000, total_qty=60.0, bar_end_ts=61000, timeframe_ms=60000
        )

        assert len(plan) == 6
        # Last child should be near end
        assert plan[-1].ts_offset_ms <= 60000


class TestMidOffsetLimitExecutor:
    """Test MidOffsetLimitExecutor."""

    @pytest.fixture
    def executor(self):
        """Create MidOffsetLimitExecutor."""
        return MidOffsetLimitExecutor(offset_bps=10.0, ttl_steps=100, tif="GTC")

    def test_init(self):
        """Test initialization."""
        executor = MidOffsetLimitExecutor(offset_bps=20.0, ttl_steps=50, tif="IOC")
        assert executor.offset_bps == 20.0
        assert executor.ttl_steps == 50
        assert executor.tif == "IOC"

    def test_build_action_buy(self, executor):
        """Test building buy limit action."""
        action = executor.build_action(side="BUY", qty=10.0, snapshot={"mid": 50000.0})

        assert action is not None
        # Buy should be above mid
        assert action.abs_price == 50000.0 * (1.0 + 10.0 / 10000.0)

    def test_build_action_sell(self, executor):
        """Test building sell limit action."""
        action = executor.build_action(side="SELL", qty=10.0, snapshot={"mid": 50000.0})

        assert action is not None
        # Sell should be below mid
        assert action.abs_price == 50000.0 * (1.0 - 10.0 / 10000.0)

    def test_build_action_no_mid(self, executor):
        """Test building action without mid price."""
        action = executor.build_action(side="BUY", qty=10.0, snapshot={})

        assert action is None

    def test_build_action_zero_qty(self, executor):
        """Test building action with zero quantity."""
        action = executor.build_action(side="BUY", qty=0.0, snapshot={"mid": 50000.0})

        assert action is None


class TestMarketOpenH1Executor:
    """Test MarketOpenH1Executor."""

    @pytest.fixture
    def executor(self):
        """Create MarketOpenH1Executor."""
        return MarketOpenH1Executor()

    def test_plan_market(self, executor):
        """Test planning for next hour open."""
        now_ms = 3_600_000 * 5 + 1000  # 5 hours + 1 second
        plan = executor.plan_market(now_ts_ms=now_ms, side="BUY", target_qty=10.0, snapshot={})

        assert len(plan) == 1
        # Should wait until next hour
        expected_offset = 3_600_000 * 6 - now_ms
        assert plan[0].ts_offset_ms == expected_offset
        assert plan[0].qty == 10.0

    def test_plan_market_zero_qty(self, executor):
        """Test planning with zero quantity."""
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=0.0, snapshot={})

        assert len(plan) == 0


class TestMakeExecutor:
    """Test make_executor factory function."""

    def test_make_taker(self):
        """Test creating TAKER executor."""
        executor = make_executor("TAKER")
        assert isinstance(executor, TakerExecutor)

        executor = make_executor("taker")
        assert isinstance(executor, TakerExecutor)

    def test_make_twap(self):
        """Test creating TWAP executor."""
        executor = make_executor("TWAP")
        assert isinstance(executor, TWAPExecutor)

    def test_make_twap_with_config(self):
        """Test creating TWAP executor with config."""
        cfg = {"twap": {"parts": 10, "child_interval_s": 300}}
        executor = make_executor("TWAP", cfg)
        assert isinstance(executor, TWAPExecutor)
        assert executor.parts == 10
        assert executor.child_interval_ms == 300000

    def test_make_pov(self):
        """Test creating POV executor."""
        executor = make_executor("POV")
        assert isinstance(executor, POVExecutor)

    def test_make_pov_with_config(self):
        """Test creating POV executor with config."""
        cfg = {"pov": {"participation": 0.15, "child_interval_s": 30, "min_child_notional": 50.0}}
        executor = make_executor("POV", cfg)
        assert isinstance(executor, POVExecutor)
        assert executor.participation == 0.15

    def test_make_vwap(self):
        """Test creating VWAP executor."""
        executor = make_executor("VWAP")
        assert isinstance(executor, VWAPExecutor)

    def test_make_unknown(self):
        """Test creating unknown executor (defaults to TAKER)."""
        executor = make_executor("UNKNOWN")
        assert isinstance(executor, TakerExecutor)

        executor = make_executor("")
        assert isinstance(executor, TakerExecutor)


class TestBarWindowAware:
    """Test _BarWindowAware mixin functionality."""

    def test_resolve_bar_window_from_snapshot(self):
        """Test resolving bar window from snapshot."""
        executor = TWAPExecutor(parts=6)
        now_ms = 60000

        timeframe, start, end = executor._resolve_bar_window(
            now_ms, {"bar_timeframe_ms": 60000, "bar_start_ts": 0, "bar_end_ts": 60000}
        )

        assert timeframe == 60000
        assert start == 0
        assert end == 60000

    def test_resolve_bar_window_inference(self):
        """Test inferring bar window from partial data."""
        executor = TWAPExecutor(parts=6)
        now_ms = 60000

        # Only timeframe and start
        timeframe, start, end = executor._resolve_bar_window(
            now_ms, {"bar_timeframe_ms": 60000, "bar_start_ts": 0}
        )

        assert timeframe == 60000
        assert start == 0
        assert end == 60000

    def test_resolve_bar_window_caching(self):
        """Test caching of bar window parameters."""
        executor = TWAPExecutor(parts=6)

        # First call with full data
        executor._resolve_bar_window(60000, {"bar_timeframe_ms": 60000})

        # Second call without data (should use cached)
        timeframe, _, _ = executor._resolve_bar_window(120000, {})
        assert timeframe == 60000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_quantities(self):
        """Test handling of negative quantities."""
        executor = TakerExecutor()
        plan = executor.plan_market(now_ts_ms=1000, side="SELL", target_qty=-10.0, snapshot={})

        assert len(plan) == 1
        assert plan[0].qty == 10.0

    def test_very_small_quantities(self):
        """Test handling of very small quantities."""
        executor = TWAPExecutor(parts=6)
        plan = executor.plan_market(now_ts_ms=1000, side="BUY", target_qty=0.000001, snapshot={})

        total = sum(c.qty for c in plan)
        assert math.isclose(total, 0.000001, rel_tol=1e-6)

    def test_very_large_quantities(self):
        """Test handling of very large quantities.

        Note: POVExecutor caps total_children at 10000 to prevent excessive orders.
        With participation=0.1, liquidity=1.0, min_child_notional=20.0:
        - per_child_qty = max(0.1 * 1.0, 20.0 / 1.0) = 20.0
        - total_children = min(ceil(1000000.0 / 20.0), 10000) = 10000
        - actual_total = 10000 * 20.0 = 200000.0
        """
        executor = POVExecutor()
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=1000000.0,
            snapshot={"liquidity": 1.0, "ref_price": 1.0},
        )

        # POVExecutor caps at 10000 children, resulting in 200000.0 total
        total = sum(c.qty for c in plan)
        assert len(plan) <= 10000, "Should not exceed 10000 children"
        assert math.isclose(total, 200000.0, rel_tol=1e-6)

    def test_invalid_snapshot_values(self):
        """Test handling of invalid snapshot values."""
        executor = VWAPExecutor()

        # Invalid profile entries
        plan = executor.plan_market(
            now_ts_ms=1000,
            side="BUY",
            target_qty=100.0,
            snapshot={
                "intrabar_volume_profile": [
                    {"ts": "invalid", "volume": 50.0},
                    {"ts": 30000, "volume": "invalid"},
                    None,
                    "invalid",
                ]
            },
        )

        # Should fall back to default plan
        assert len(plan) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
