"""
Core tests for CRITICAL fixes (no external dependencies).

Tests cover:
- CRITICAL FIX #2: Yang-Zhang Bessel's correction
- CRITICAL FIX #3: Log vs Linear returns consistency
- CRITICAL FIX #4: EWMA robust initialization (direct testing)

These tests use only standard library + numpy/pandas.
"""

import math
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Direct EWMA Testing (without importing transformers)
# =============================================================================


def _test_calculate_ewma_robust(prices, lambda_decay=0.94):
    """
    Direct implementation matching the FIXED EWMA logic.

    CRITICAL FIX #4: Robust initialization.
    """
    if not prices or len(prices) < 2:
        return None

    price_array = np.array(prices, dtype=float)

    if np.any(price_array <= 0) or np.any(~np.isfinite(price_array)):
        return None

    log_returns = np.log(price_array[1:] / price_array[:-1])

    if not np.all(np.isfinite(log_returns)):
        return None

    # CRITICAL FIX #4: Robust initialization
    if len(log_returns) >= 10:
        variance = np.var(log_returns, ddof=1)
    elif len(log_returns) >= 3:
        variance = float(np.median(log_returns**2))
    else:
        variance = float(np.mean(log_returns**2))

    for ret in log_returns:
        variance = lambda_decay * variance + (1 - lambda_decay) * (ret**2)

    volatility = np.sqrt(variance)

    if not np.isfinite(volatility) or volatility <= 0:
        return None

    return float(volatility)


class TestEWMARobustInitialization:
    """Test EWMA robust initialization fix."""

    def test_median_filters_spike(self):
        """
        CRITICAL FIX #4: Median initialization should filter first-return spikes.
        """
        # Spike scenario: first return is 10x larger
        prices = [100.0, 120.0]  # +20% spike
        for _ in range(8):
            prices.append(prices[-1] * 1.01)  # +1% normal moves

        vol = _test_calculate_ewma_robust(prices)

        # Calculate what old approach would give
        log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        old_init = log_returns[0] ** 2
        new_init = float(np.median(log_returns**2))

        # New init should be much smaller (spike filtered)
        assert (
            new_init < old_init * 0.5
        ), f"Median should filter spike: old={old_init:.6f}, new={new_init:.6f}"
        assert vol < 0.05, f"Final volatility should be reasonable: {vol:.4f}"

    def test_mean_fallback_for_2_returns(self):
        """With only 2 returns, use mean of squared returns."""
        prices = [100.0, 105.0, 110.0]  # 2 returns

        vol = _test_calculate_ewma_robust(prices)

        assert vol is not None, "Should work with 2 returns"
        assert 0.02 < vol < 0.10, f"Volatility should be reasonable: {vol:.4f}"

    def test_sample_variance_for_sufficient_data(self):
        """With >=10 returns, use sample variance."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(20):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))

        vol = _test_calculate_ewma_robust(prices)

        assert vol is not None
        assert 0.01 < vol < 0.05, f"Volatility should match input: {vol:.4f}"


# =============================================================================
# Log vs Linear Returns Consistency
# =============================================================================


class TestLogLinearReturnsConsistency:
    """Test that log returns are used consistently."""

    def test_log_return_formula(self):
        """Verify log return calculation."""
        price_old = 100.0
        price_new = 110.0

        log_ret = math.log(price_new / price_old)
        linear_ret = (price_new / price_old) - 1.0

        # Log return should be ln(1.10) ≈ 0.0953
        assert abs(log_ret - 0.09531) < 0.0001, f"Log return: {log_ret:.6f}"

        # Linear return is 0.10
        assert abs(linear_ret - 0.10) < 0.0001, f"Linear return: {linear_ret:.6f}"

        # They should differ
        assert abs(log_ret - linear_ret) > 0.003, "Log and linear returns differ for 10% move"

    def test_large_returns_difference(self):
        """For large returns, log vs linear diverges significantly."""
        price_old = 100.0
        price_new = 150.0  # 50% increase

        log_ret = math.log(price_new / price_old)
        linear_ret = (price_new / price_old) - 1.0

        # Log: ln(1.5) ≈ 0.4055
        # Linear: 0.5
        assert abs(log_ret - 0.4055) < 0.001, f"Log return: {log_ret:.6f}"
        assert abs(linear_ret - 0.5) < 0.001, f"Linear return: {linear_ret:.6f}"

        # 19% difference!
        difference = abs(linear_ret - log_ret)
        assert difference > 0.09, f"Should have large difference: {difference:.4f}"

    def test_small_returns_similarity(self):
        """For small returns (<5%), log ≈ linear."""
        price_old = 100.0
        price_new = 102.0  # 2% increase

        log_ret = math.log(price_new / price_old)
        linear_ret = (price_new / price_old) - 1.0

        # Should be very close for small returns
        difference = abs(linear_ret - log_ret)
        assert difference < 0.0002, f"Small returns should be similar: diff={difference:.6f}"


# =============================================================================
# Yang-Zhang Bessel's Correction
# =============================================================================


def _test_calculate_yang_zhang_fixed(ohlc_bars, n):
    """
    Direct implementation of Yang-Zhang with FIXED Bessel's correction.

    CRITICAL FIX #2: Rogers-Satchell uses (n-1) not n.
    """
    if not ohlc_bars or len(ohlc_bars) < n or n < 2:
        return None

    bars = list(ohlc_bars)[-n:]
    k = 0.34

    # Overnight returns
    overnight_returns = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1]["close"]
        curr_open = bars[i]["open"]
        if prev_close > 0 and curr_open > 0:
            overnight_returns.append(math.log(curr_open / prev_close))

    if len(overnight_returns) < 2:
        return None

    mean_overnight = sum(overnight_returns) / len(overnight_returns)
    sigma_o_sq = sum((r - mean_overnight) ** 2 for r in overnight_returns) / (
        len(overnight_returns) - 1
    )

    # Open-close returns
    oc_returns = []
    for bar in bars:
        open_price = bar["open"]
        close_price = bar["close"]
        if open_price > 0 and close_price > 0:
            oc_returns.append(math.log(close_price / open_price))

    if len(oc_returns) < 2:
        return None

    mean_oc = sum(oc_returns) / len(oc_returns)
    sigma_c_sq = sum((r - mean_oc) ** 2 for r in oc_returns) / (len(oc_returns) - 1)

    # Rogers-Satchell
    rs_sum = 0.0
    rs_count = 0
    for bar in bars:
        high = bar["high"]
        low = bar["low"]
        open_price = bar["open"]
        close_price = bar["close"]

        if high > 0 and low > 0 and open_price > 0 and close_price > 0:
            term1 = math.log(high / close_price) * math.log(high / open_price)
            term2 = math.log(low / close_price) * math.log(low / open_price)
            rs_sum += term1 + term2
            rs_count += 1

    if rs_count < 2:  # CRITICAL FIX #2: minimum 2 for (n-1)
        return None

    sigma_rs_sq = rs_sum / (rs_count - 1)  # CRITICAL FIX #2: Bessel's correction

    # Combined
    sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq

    if sigma_yz_sq < 0:
        return None

    return math.sqrt(sigma_yz_sq)


class TestYangZhangBesselCorrection:
    """Test Yang-Zhang Bessel's correction fix."""

    def test_minimum_sample_size_2(self):
        """After Bessel's correction, rs_count < 2 should fail."""
        # Single bar
        ohlc_bars = [
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        ]

        result = _test_calculate_yang_zhang_fixed(ohlc_bars, n=1)

        assert result is None, "Single bar should fail with rs_count < 2"

    def test_three_bars_succeed(self):
        """
        Three bars should succeed (minimum for Yang-Zhang).

        Yang-Zhang needs:
        - overnight_returns: requires 2+ transitions (3 bars)
        - oc_returns: requires 2+ bars
        - rs: requires 2+ bars
        """
        ohlc_bars = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0},
            {"open": 101.0, "high": 102.5, "low": 100.5, "close": 101.5},
        ]

        result = _test_calculate_yang_zhang_fixed(ohlc_bars, n=3)

        assert result is not None, "Three bars should succeed"
        assert result > 0, "Volatility should be positive"

    def test_reasonable_volatility_estimate(self):
        """Yang-Zhang should give reasonable estimates."""
        np.random.seed(42)
        n = 20
        base_price = 100.0
        volatility = 0.02  # 2% daily vol

        ohlc_bars = []
        for i in range(n):
            close_prev = base_price if i == 0 else ohlc_bars[i - 1]["close"]
            open_price = close_prev * (1 + np.random.normal(0, volatility / 2))
            close = open_price * (1 + np.random.normal(0, volatility))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility / 4)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility / 4)))

            ohlc_bars.append(
                {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )

        yz_vol = _test_calculate_yang_zhang_fixed(ohlc_bars, n)

        assert yz_vol is not None
        assert 0.01 < yz_vol < 0.05, f"Volatility should be ~2%: {yz_vol:.4f}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for all fixes."""

    def test_returns_time_additivity(self):
        """
        Log returns are time-additive: r(t1→t3) = r(t1→t2) + r(t2→t3).

        This is why log returns are preferred for features.
        """
        prices = [100.0, 110.0, 121.0]  # 10% each step

        # Log returns
        log_ret_1 = math.log(prices[1] / prices[0])
        log_ret_2 = math.log(prices[2] / prices[1])
        log_ret_total = math.log(prices[2] / prices[0])

        # Time-additive property
        assert abs(log_ret_total - (log_ret_1 + log_ret_2)) < 1e-10, "Log returns are additive"

        # Linear returns are NOT time-additive
        linear_ret_1 = (prices[1] / prices[0]) - 1
        linear_ret_2 = (prices[2] / prices[1]) - 1
        linear_ret_total = (prices[2] / prices[0]) - 1

        # This should NOT equal sum (it equals product minus 1)
        assert (
            abs(linear_ret_total - (linear_ret_1 + linear_ret_2)) > 0.001
        ), "Linear returns not additive"

    def test_volatility_estimation_robustness(self):
        """
        Test that volatility estimation is robust across different scenarios.
        """
        np.random.seed(42)

        # Generate realistic price paths using cumulative product
        def generate_prices(n, vol):
            returns = np.random.normal(0, vol, n - 1)
            prices = [100.0]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            return prices

        scenarios = [
            # Normal volatility (2% daily)
            (generate_prices(30, 0.02), 0.010, 0.030),
            # Low volatility (0.5% daily)
            (generate_prices(30, 0.005), 0.002, 0.010),
            # High volatility (5% daily)
            (generate_prices(30, 0.05), 0.035, 0.065),
        ]

        for prices, vol_min, vol_max in scenarios:
            vol = _test_calculate_ewma_robust(prices)
            assert vol is not None, "Should estimate volatility"
            assert (
                vol_min < vol < vol_max
            ), f"Volatility {vol:.4f} should be in range [{vol_min}, {vol_max}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
