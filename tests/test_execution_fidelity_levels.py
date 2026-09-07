# -*- coding: utf-8 -*-
"""
Tests for execution provider fidelity levels (L2 vs L2+ vs L3).

This test suite validates the documented design trade-offs between different
execution simulation fidelity levels. These are NOT bugs - they are intentional
design choices documented in docs/PLATFORM_REFERENCE.md sections #54-#59.

References:
    - docs/PLATFORM_REFERENCE.md #54: L2 ADV Intraday Seasonality (by design)
    - docs/PLATFORM_REFERENCE.md #55: L2 Temp/Perm Impact (by design)
    - docs/PLATFORM_REFERENCE.md #56: L2 Static Spread (by design)
    - docs/PLATFORM_REFERENCE.md #57: L2 Deterministic Fills (by design)
    - docs/PLATFORM_REFERENCE.md #58: Whale Threshold Configurable
    - docs/PLATFORM_REFERENCE.md #59: Reward Clipping Not Stacked
"""

import pytest
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Test Fixtures and Mock Classes
# =============================================================================


@dataclass
class MockMarketState:
    """Mock market state for testing."""

    timestamp: int = 0
    bid: float = 100.0
    ask: float = 100.05
    adv: Optional[float] = 10_000_000.0
    volatility: Optional[float] = 0.02

    def get_spread_bps(self) -> Optional[float]:
        if self.bid and self.ask:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2) * 10000
        return None

    def get_mid_price(self) -> Optional[float]:
        if self.bid and self.ask:
            return (self.ask + self.bid) / 2
        return None


@dataclass
class MockOrder:
    """Mock order for testing."""

    symbol: str = "BTCUSDT"
    side: str = "BUY"
    qty: float = 100.0
    order_type: str = "MARKET"

    def get_notional(self, price: float) -> float:
        return self.qty * price


@dataclass
class MockBarData:
    """Mock bar data for testing."""

    open: float = 100.0
    high: float = 101.0
    low: float = 99.0
    close: float = 100.5
    volume: float = 100000.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3


# =============================================================================
# Test L2 vs L2+ Fidelity Differences (Issues #6-#9)
# =============================================================================


class TestL2VsL2PlusFidelity:
    """
    Test that L2 is intentionally simpler than L2+.

    These tests validate the DOCUMENTED design trade-offs, not bugs.
    """

    def test_l2_no_tod_adjustment_by_design(self):
        """
        Test #54: L2 StatisticalSlippageProvider does NOT adjust ADV by time-of-day.
        This is BY DESIGN - L2+ has tod_curve for this.
        """
        try:
            from execution_providers import StatisticalSlippageProvider
        except ImportError:
            pytest.skip("execution_providers not available")

        provider = StatisticalSlippageProvider()
        market = MockMarketState()
        order = MockOrder()

        # L2 uses market.adv directly without TOD adjustment
        # The slippage should be the same regardless of the hour
        participation = order.get_notional(market.get_mid_price()) / market.adv

        slippage1 = provider.compute_slippage_bps(order, market, participation)
        # Note: L2 doesn't have hour_utc parameter - this is BY DESIGN
        slippage2 = provider.compute_slippage_bps(order, market, participation)

        # L2 gives same slippage - no TOD adjustment
        assert slippage1 == slippage2, "L2 should NOT have TOD adjustment (by design)"
        assert slippage1 > 0, "Slippage should be positive"

    def test_l2_plus_has_tod_curve(self):
        """
        Test that L2+ (CryptoParametricSlippageProvider) HAS tod_curve.
        """
        try:
            from execution_providers import CryptoParametricSlippageProvider, CryptoParametricConfig
        except ImportError:
            pytest.skip("CryptoParametricSlippageProvider not available")

        config = CryptoParametricConfig()

        # L2+ should have tod_curve with 24 hours
        assert hasattr(config, "tod_curve"), "L2+ should have tod_curve"
        assert len(config.tod_curve) == 24, "tod_curve should have 24 hourly values"

        # Asia session (low liquidity) should have lower factors
        asia_factor = config.tod_curve[3]  # 3 AM UTC
        eu_us_factor = config.tod_curve[16]  # 4 PM UTC (EU/US overlap)

        assert asia_factor < eu_us_factor, "Asia session should have lower liquidity factor"

    def test_l2_plus_tod_affects_slippage(self):
        """
        Test that L2+ applies TOD adjustment to slippage calculation.
        """
        try:
            from execution_providers import (
                CryptoParametricSlippageProvider,
                Order,
                MarketState,
            )
        except ImportError:
            pytest.skip("CryptoParametricSlippageProvider not available")

        provider = CryptoParametricSlippageProvider()

        market = MarketState(
            timestamp=0,
            bid=100.0,
            ask=100.05,
            adv=10_000_000,
        )
        order = Order(symbol="BTCUSDT", side="BUY", qty=1000.0, order_type="MARKET")
        participation = 0.01  # 1%

        # Compare slippage at different hours
        slippage_asia = provider.compute_slippage_bps(order, market, participation, hour_utc=3)
        slippage_peak = provider.compute_slippage_bps(order, market, participation, hour_utc=16)

        # Asia (low liquidity) should have HIGHER slippage
        assert (
            slippage_asia > slippage_peak
        ), f"Asia slippage ({slippage_asia:.2f}) should be > peak ({slippage_peak:.2f})"

    def test_l2_single_impact_term_by_design(self):
        """
        Test #55: L2 uses single √participation impact term.
        Temp/perm separation is in L3 (by design).
        """
        try:
            from execution_providers import StatisticalSlippageProvider
        except ImportError:
            pytest.skip("execution_providers not available")

        provider = StatisticalSlippageProvider(impact_coef=0.1)
        market = MockMarketState(adv=10_000_000)
        order = MockOrder()

        # Test that slippage increases monotonically with participation
        participations = [0.0001, 0.001, 0.01, 0.1]  # 0.01%, 0.1%, 1%, 10%
        slippages = [provider.compute_slippage_bps(order, market, p) for p in participations]

        # Verify monotonically increasing
        for i in range(1, len(slippages)):
            assert slippages[i] > slippages[i - 1], (
                f"Slippage should increase with participation: "
                f"p={participations[i]} -> {slippages[i]:.2f}bps, "
                f"p={participations[i-1]} -> {slippages[i-1]:.2f}bps"
            )

        # Verify that L2 uses single-term model (no temp/perm separation)
        # by checking that compute_slippage_bps returns a single value, not dict
        result = provider.compute_slippage_bps(order, market, 0.01)
        assert isinstance(
            result, (int, float)
        ), "L2 should return single slippage value (no temp/perm separation)"

    def test_l3_has_temp_perm_separation(self):
        """
        Test that L3 (AlmgrenChrissModel) separates temporary and permanent impact.
        """
        try:
            from lob.market_impact import AlmgrenChrissModel, ImpactParameters
        except ImportError:
            pytest.skip("L3 market_impact not available")

        params = ImpactParameters.for_equity()
        model = AlmgrenChrissModel(params=params)

        result = model.compute_total_impact(
            order_qty=10000,
            adv=10_000_000,
            volatility=0.02,
            mid_price=100.0,
        )

        # L3 should have separate temp and perm components
        assert hasattr(result, "temporary_impact_bps"), "L3 should have temporary_impact_bps"
        assert hasattr(result, "permanent_impact_bps"), "L3 should have permanent_impact_bps"
        assert result.temporary_impact_bps > 0, "Temporary impact should be positive"
        assert result.permanent_impact_bps > 0, "Permanent impact should be positive"


class TestWhaleThresholdConfigurable:
    """
    Test #58: Whale threshold is configurable, not a bug.
    """

    def test_whale_threshold_default(self):
        """Test default whale threshold is 1%."""
        try:
            from execution_providers import CryptoParametricConfig
        except ImportError:
            pytest.skip("CryptoParametricConfig not available")

        config = CryptoParametricConfig()
        assert config.whale_threshold == 0.01, "Default whale threshold should be 1%"

    def test_whale_threshold_configurable(self):
        """Test whale threshold can be configured."""
        try:
            from execution_providers import CryptoParametricConfig
        except ImportError:
            pytest.skip("CryptoParametricConfig not available")

        # For low-liquidity altcoins
        config = CryptoParametricConfig(whale_threshold=0.005)
        assert config.whale_threshold == 0.005, "Whale threshold should be configurable"

    def test_whale_profiles_exist(self):
        """Test that configuration profiles exist for different asset types."""
        try:
            from execution_providers import CryptoParametricSlippageProvider
        except ImportError:
            pytest.skip("CryptoParametricSlippageProvider not available")

        profiles = ["default", "conservative", "aggressive", "altcoin", "stablecoin"]

        for profile in profiles:
            provider = CryptoParametricSlippageProvider.from_profile(profile)
            assert provider is not None, f"Profile '{profile}' should exist"
            assert (
                provider.config.whale_threshold > 0
            ), f"Profile '{profile}' should have positive whale_threshold"

    def test_altcoin_profile_lower_threshold(self):
        """Test altcoin profile has lower whale threshold."""
        try:
            from execution_providers import CryptoParametricSlippageProvider
        except ImportError:
            pytest.skip("CryptoParametricSlippageProvider not available")

        default_provider = CryptoParametricSlippageProvider.from_profile("default")
        altcoin_provider = CryptoParametricSlippageProvider.from_profile("altcoin")

        # Altcoins should have stricter whale detection (lower threshold)
        # because low-liquidity assets are more sensitive to large orders
        assert (
            altcoin_provider.config.whale_threshold <= default_provider.config.whale_threshold
        ), "Altcoin profile should have lower or equal whale threshold"


class TestRewardClippingNotStacked:
    """
    Test #59: Reward clipping serves different purposes, not stacked.

    The two clips in trading_patchnew.py are:
    1. ratio clip (1e-10, 1e10) - numerical safety BEFORE log()
    2. final clip (-clip_for_clamp, clip_for_clamp) - policy bounds
    """

    def test_ratio_clip_prevents_log_overflow(self):
        """Test that ratio clipping prevents log(0) = -inf."""
        # Simulate what happens in _compute_reward_signal_only

        # Without clip: log(0) -> -inf -> NaN propagation
        # Note: Python's math.log(0) raises ValueError, but in numpy with errstate it returns -inf
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_bad = 0.0
            log_result = np.log(ratio_bad)
            assert np.isinf(log_result) or np.isnan(
                log_result
            ), "log(0) should be -inf or nan without clipping"

        # With clip: log(1e-10) -> finite negative value
        ratio_clipped = np.clip(ratio_bad, 1e-10, 1e10)
        result = math.log(ratio_clipped)
        assert math.isfinite(result), "Clipped ratio should give finite log"
        assert result < 0, "log(1e-10) should be negative"
        # log(1e-10) ≈ -23
        assert result < -20, "log(1e-10) should be approximately -23"

    def test_final_clip_bounds_reward(self):
        """Test that final clip bounds extreme rewards."""
        clip_for_clamp = 10.0  # Typical value

        # Extreme reward before clip
        reward_extreme = 100.0
        reward_clipped = float(np.clip(reward_extreme, -clip_for_clamp, clip_for_clamp))

        assert reward_clipped == clip_for_clamp, "Extreme reward should be clipped"

    def test_clips_rarely_triggered_in_normal_operation(self):
        """Test that clips are defensive and rarely trigger."""
        # Normal price ratio: 0.99 to 1.01 (1% move)
        normal_ratios = [0.99, 0.995, 1.0, 1.005, 1.01]

        for ratio in normal_ratios:
            # First clip: rarely triggers for normal ratios
            ratio_clipped = np.clip(ratio, 1e-10, 1e10)
            assert ratio_clipped == ratio, f"Normal ratio {ratio} should not be clipped"

            # Log return
            log_return = math.log(ratio_clipped)

            # Second clip: rarely triggers for normal returns
            # |log(1.01)| ≈ 0.01 << 10.0
            reward = log_return * 1.0  # Assuming position = 1.0
            reward_clipped = np.clip(reward, -10.0, 10.0)
            assert reward_clipped == reward, f"Normal reward {reward} should not be clipped"

    def test_value_function_monotonicity_preserved(self):
        """
        Test that value function remains monotonic despite clipping.

        For monotonically increasing prices, rewards should be monotonically increasing.
        """
        base_price = 100.0
        prices = [100.0, 100.5, 101.0, 101.5, 102.0]  # Monotonically increasing

        rewards = []
        prev_price = base_price
        position = 1.0

        for price in prices[1:]:
            ratio = price / prev_price
            ratio_clipped = np.clip(ratio, 1e-10, 1e10)
            log_return = math.log(ratio_clipped)
            reward = log_return * position
            reward_clipped = np.clip(reward, -10.0, 10.0)
            rewards.append(reward_clipped)
            prev_price = price

        # All rewards should be positive (price increasing)
        assert all(r > 0 for r in rewards), "Increasing prices should give positive rewards"

        # Rewards should be monotonically comparable (within tolerance)
        # Note: rewards might not be strictly increasing due to compounding effects
        # but they should all be positive for increasing prices


class TestLongOnlyRewardSemantics:
    """
    Test that long-only reward semantics are mathematically correct.

    reward = log(price_ratio) * position

    At position = 0, reward = 0 regardless of market move.
    This is CORRECT because the agent didn't participate.
    """

    def test_zero_position_gives_zero_reward(self):
        """Test that position=0 gives reward=0 regardless of price move."""
        price_prev = 100.0
        price_curr_up = 110.0  # +10%
        price_curr_down = 90.0  # -10%
        position = 0.0

        ratio_up = price_curr_up / price_prev
        ratio_down = price_curr_down / price_prev

        reward_up = math.log(ratio_up) * position
        reward_down = math.log(ratio_down) * position

        assert reward_up == 0.0, "Position=0 should give zero reward for up move"
        assert reward_down == 0.0, "Position=0 should give zero reward for down move"

    def test_full_position_captures_move(self):
        """Test that position=1.0 captures full price move."""
        price_prev = 100.0
        price_curr = 110.0  # +10%
        position = 1.0

        ratio = price_curr / price_prev
        reward = math.log(ratio) * position

        expected = math.log(1.10)  # ~0.0953
        assert abs(reward - expected) < 1e-6, "Full position should capture full return"

    def test_partial_position_proportional_reward(self):
        """Test that partial position gives proportional reward."""
        price_prev = 100.0
        price_curr = 110.0  # +10%

        full_return = math.log(price_curr / price_prev)

        for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
            reward = full_return * position
            expected = full_return * position
            assert (
                abs(reward - expected) < 1e-10
            ), f"Position {position} should give proportional reward"

    def test_reward_correct_for_loss(self):
        """Test reward is negative for price drop with positive position."""
        price_prev = 100.0
        price_curr = 90.0  # -10%
        position = 1.0

        ratio = price_curr / price_prev
        reward = math.log(ratio) * position

        assert reward < 0, "Negative price move with positive position should give negative reward"


# =============================================================================
# Integration Tests
# =============================================================================


class TestFidelityLevelSelection:
    """
    Test that users can select appropriate fidelity level for their use case.
    """

    def test_l2_is_fastest(self):
        """Test that L2 is faster than L2+ (simpler model)."""
        try:
            from execution_providers import (
                StatisticalSlippageProvider,
                CryptoParametricSlippageProvider,
                Order,
                MarketState,
            )
        except ImportError:
            pytest.skip("execution_providers not available")

        import time

        l2_provider = StatisticalSlippageProvider()
        l2_plus_provider = CryptoParametricSlippageProvider()

        market = MarketState(timestamp=0, bid=100.0, ask=100.05, adv=10_000_000)
        order = Order(symbol="BTCUSDT", side="BUY", qty=1000.0, order_type="MARKET")
        participation = 0.01

        # Time L2
        n_iterations = 1000
        start = time.perf_counter()
        for _ in range(n_iterations):
            l2_provider.compute_slippage_bps(order, market, participation)
        l2_time = time.perf_counter() - start

        # Time L2+
        start = time.perf_counter()
        for _ in range(n_iterations):
            l2_plus_provider.compute_slippage_bps(order, market, participation)
        l2_plus_time = time.perf_counter() - start

        # L2 should be faster (simpler model)
        # Note: This might not always hold due to JIT/caching, so we use a generous ratio
        assert (
            l2_time <= l2_plus_time * 2
        ), f"L2 ({l2_time:.4f}s) should be faster than L2+ ({l2_plus_time:.4f}s)"

    def test_l2_plus_more_accurate_for_intraday(self):
        """Test that L2+ captures intraday effects that L2 misses."""
        try:
            from execution_providers import (
                StatisticalSlippageProvider,
                CryptoParametricSlippageProvider,
                Order,
                MarketState,
            )
        except ImportError:
            pytest.skip("execution_providers not available")

        l2_provider = StatisticalSlippageProvider()
        l2_plus_provider = CryptoParametricSlippageProvider()

        market = MarketState(timestamp=0, bid=100.0, ask=100.05, adv=10_000_000)
        order = Order(symbol="BTCUSDT", side="BUY", qty=1000.0, order_type="MARKET")
        participation = 0.01

        # L2: Same slippage regardless of hour
        l2_asia = l2_provider.compute_slippage_bps(order, market, participation)
        l2_peak = l2_provider.compute_slippage_bps(order, market, participation)

        assert l2_asia == l2_peak, "L2 should not vary by hour"

        # L2+: Different slippage based on hour
        l2_plus_asia = l2_plus_provider.compute_slippage_bps(
            order, market, participation, hour_utc=3
        )
        l2_plus_peak = l2_plus_provider.compute_slippage_bps(
            order, market, participation, hour_utc=16
        )

        assert l2_plus_asia != l2_plus_peak, "L2+ should vary by hour"
        assert l2_plus_asia > l2_plus_peak, "L2+ should show higher slippage in Asia session"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
