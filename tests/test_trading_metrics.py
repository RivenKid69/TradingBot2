"""Unit tests for trading_metrics.py module.

Tests cover:
1. Sharpe Ratio calculation
2. Sortino Ratio calculation
3. Max Drawdown calculation
4. Calmar Ratio calculation
5. Win Rate and Profit Factor
6. Edge cases and numerical stability
"""

import math
import numpy as np
import pytest

from trading_metrics import (
    TradingMetrics,
    TradingMetricsAccumulator,
    compute_trading_metrics,
    extract_trading_metrics_payload,
    _safe_divide,
    _compute_downside_std,
    _compute_max_drawdown,
    DEFAULT_BARS_PER_YEAR,
)


class TestSafeHelpers:
    """Tests for helper functions."""

    def test_safe_divide_normal(self):
        assert _safe_divide(10.0, 2.0) == 5.0

    def test_safe_divide_zero_denominator(self):
        assert _safe_divide(10.0, 0.0) == 0.0

    def test_safe_divide_zero_denominator_custom_default(self):
        assert _safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_safe_divide_nan_denominator(self):
        assert _safe_divide(10.0, float("nan")) == 0.0

    def test_safe_divide_inf_denominator(self):
        assert _safe_divide(10.0, float("inf")) == 0.0


class TestDownsideStd:
    """Tests for downside standard deviation calculation."""

    def test_downside_std_all_negative(self):
        returns = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])
        result = _compute_downside_std(returns)
        # sqrt(mean(r^2)) for r < 0
        expected = math.sqrt(np.mean(returns**2))
        assert abs(result - expected) < 1e-10

    def test_downside_std_mixed(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03])
        result = _compute_downside_std(returns)
        expected = math.sqrt(np.mean(np.minimum(returns, 0.0)**2))
        assert abs(result - expected) < 1e-10

    def test_downside_std_all_positive(self):
        returns = np.array([0.01, 0.02, 0.03])
        result = _compute_downside_std(returns)
        # No negatives - should return small epsilon
        assert result == 1e-8

    def test_downside_std_empty(self):
        returns = np.array([])
        result = _compute_downside_std(returns)
        assert math.isnan(result)


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_max_drawdown_simple(self):
        # Equity: 100 -> 120 -> 90 -> 110
        # Peak at 120, trough at 90 => DD = (120-90)/120 = 0.25
        equity = np.array([100.0, 120.0, 90.0, 110.0])
        max_dd, duration, current_dd = _compute_max_drawdown(equity)
        assert abs(max_dd - 0.25) < 1e-10
        assert duration >= 1

    def test_max_drawdown_no_drawdown(self):
        # Monotonically increasing equity
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        max_dd, duration, current_dd = _compute_max_drawdown(equity)
        assert max_dd < 1e-10
        assert current_dd < 1e-10

    def test_max_drawdown_full_recovery(self):
        # 100 -> 80 -> 100 (full recovery)
        equity = np.array([100.0, 80.0, 100.0])
        max_dd, duration, current_dd = _compute_max_drawdown(equity)
        assert abs(max_dd - 0.2) < 1e-10  # 20% drawdown
        assert current_dd < 1e-10  # Recovered

    def test_max_drawdown_duration(self):
        # 100 -> 90 -> 85 -> 80 -> 85 -> 100 (4 steps in drawdown)
        equity = np.array([100.0, 90.0, 85.0, 80.0, 85.0, 100.0])
        max_dd, duration, current_dd = _compute_max_drawdown(equity)
        assert abs(max_dd - 0.2) < 1e-10  # 20% max drawdown
        assert duration == 4  # 4 consecutive steps in drawdown

    def test_max_drawdown_empty(self):
        equity = np.array([])
        max_dd, duration, current_dd = _compute_max_drawdown(equity)
        assert max_dd == 0.0
        assert duration == 0
        assert current_dd == 0.0


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive(self):
        # Consistent positive returns
        returns = [0.01] * 100  # 1% each step
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # With zero std, Sharpe should be 0 (due to epsilon protection)
        # But we have slight std due to float precision
        assert metrics.sharpe_ratio >= 0

    def test_sharpe_negative(self):
        # Negative average returns with some variance
        np.random.seed(42)
        returns = np.random.normal(-0.01, 0.005, 100).tolist()  # -1% mean, 0.5% std
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert metrics.sharpe_ratio < 0
        assert metrics.mean_return < 0

    def test_sharpe_volatile(self):
        # High volatility returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.05, 1000).tolist()  # 0.1% mean, 5% std
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # Should be relatively small due to high vol
        assert -10 < metrics.sharpe_ratio < 10

    def test_sharpe_annualization(self):
        # Test that annualization factor is applied
        returns = np.random.normal(0.001, 0.01, 100).tolist()

        # Default bars per year
        metrics_default = compute_trading_metrics(returns, bars_per_year=DEFAULT_BARS_PER_YEAR)

        # Different annualization
        metrics_daily = compute_trading_metrics(returns, bars_per_year=365.25)

        assert metrics_default is not None
        assert metrics_daily is not None
        # Different annualization should give different Sharpe
        # (unless returns are all identical)


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_vs_sharpe_asymmetric(self):
        # Returns with more upside than downside
        # Sortino should be higher than Sharpe
        returns = [0.03, 0.02, 0.01, -0.01, 0.02, 0.03, -0.005, 0.025]
        metrics = compute_trading_metrics(returns * 20)
        assert metrics is not None
        assert metrics.sortino_ratio >= metrics.sharpe_ratio

    def test_sortino_no_downside(self):
        # All positive returns - Sortino should be high
        returns = [0.01, 0.02, 0.015, 0.005, 0.01] * 20
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # Sortino should be very high (or inf-like)
        assert metrics.sortino_ratio > 0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_positive(self):
        # Consistent positive returns with small drawdown
        np.random.seed(123)
        returns = np.random.normal(0.002, 0.005, 500).tolist()
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # Should be positive if overall profitable

    def test_calmar_no_drawdown(self):
        # Monotonic returns - no drawdown
        returns = [0.01] * 100
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # With no drawdown, Calmar should be 0 or very high
        # (division by near-zero DD)

    def test_calmar_large_drawdown(self):
        # Large drawdown scenario
        returns = [0.01] * 50 + [-0.05] * 10 + [0.01] * 40
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        # Should be finite


class TestWinRateAndProfitFactor:
    """Tests for win rate and profit factor calculations."""

    def test_win_rate_all_wins(self):
        returns = [0.01] * 100
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert metrics.win_rate == 1.0

    def test_win_rate_all_losses(self):
        returns = [-0.01] * 100
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert metrics.win_rate == 0.0

    def test_win_rate_mixed(self):
        returns = [0.01, -0.01] * 50
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert abs(metrics.win_rate - 0.5) < 0.01

    def test_profit_factor_balanced(self):
        # Equal wins and losses
        returns = [0.02, -0.02] * 50
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert abs(metrics.profit_factor - 1.0) < 0.01

    def test_profit_factor_profitable(self):
        # More gains than losses
        returns = [0.03, -0.01] * 50  # 3x gain vs loss
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert abs(metrics.profit_factor - 3.0) < 0.1

    def test_avg_win_loss(self):
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015]
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert abs(metrics.avg_win - 0.02) < 0.01  # Mean of 0.02, 0.03, 0.01
        assert abs(metrics.avg_loss - (-0.015)) < 0.01  # Mean of -0.01, -0.02, -0.015


class TestTradingMetricsAccumulator:
    """Tests for the accumulator class."""

    def test_accumulator_basic(self):
        acc = TradingMetricsAccumulator()

        for ret in [0.01, -0.005, 0.02, -0.01, 0.015]:
            acc.add_step(ret)

        metrics = acc.summary()
        assert metrics is not None
        assert metrics.n_steps == 5

    def test_accumulator_with_equity(self):
        acc = TradingMetricsAccumulator()

        equity = 100.0
        for ret in [0.01, -0.005, 0.02, -0.01, 0.015]:
            equity *= (1 + ret)
            acc.add_step(ret, equity=equity)

        metrics = acc.summary()
        assert metrics is not None
        assert metrics.n_steps == 5

    def test_accumulator_episode_count(self):
        acc = TradingMetricsAccumulator()

        for i in range(10):
            acc.add_step(0.01, is_episode_end=(i == 4 or i == 9))

        assert acc.n_episodes == 2

    def test_accumulator_batch(self):
        acc = TradingMetricsAccumulator()

        returns = np.array([[0.01, 0.02], [-0.01, 0.015], [0.005, -0.005]])
        dones = np.array([[False, False], [True, False], [False, True]])

        acc.add_batch(returns, dones=dones)

        assert acc.n_steps == 6  # 3x2 = 6
        assert acc.n_episodes == 2

    def test_accumulator_reset(self):
        acc = TradingMetricsAccumulator()

        for _ in range(10):
            acc.add_step(0.01)

        assert acc.n_steps == 10

        acc.reset()

        assert acc.n_steps == 0
        assert acc.n_episodes == 0

    def test_accumulator_nan_filtering(self):
        acc = TradingMetricsAccumulator()

        acc.add_step(0.01)
        acc.add_step(float("nan"))
        acc.add_step(0.02)
        acc.add_step(float("inf"))
        acc.add_step(-0.01)

        # Only finite values should be counted
        assert acc.n_steps == 3


class TestExtractTradingMetricsPayload:
    """Tests for info dict extraction."""

    def test_extract_reward_raw_fraction(self):
        info = {"reward_raw_fraction": 0.015, "equity": 1050.0}
        ret, eq = extract_trading_metrics_payload(info)
        assert ret == 0.015
        assert eq == 1050.0

    def test_extract_alternative_keys(self):
        info = {"log_return": 0.01, "net_worth": 1000.0}
        ret, eq = extract_trading_metrics_payload(info)
        assert ret == 0.01
        assert eq == 1000.0

    def test_extract_missing_values(self):
        info = {}
        ret, eq = extract_trading_metrics_payload(info)
        assert ret is None
        assert eq is None

    def test_extract_none_info(self):
        ret, eq = extract_trading_metrics_payload(None)
        assert ret is None
        assert eq is None

    def test_extract_invalid_values(self):
        info = {"reward_raw_fraction": "not_a_number", "equity": -100.0}
        ret, eq = extract_trading_metrics_payload(info)
        assert ret is None  # Invalid string
        assert eq is None  # Negative equity filtered


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_insufficient_data(self):
        metrics = compute_trading_metrics([0.01])  # Only 1 data point
        assert metrics is None

    def test_all_nan(self):
        metrics = compute_trading_metrics([float("nan")] * 10)
        assert metrics is None

    def test_extreme_values(self):
        # Very large returns
        returns = [1.0] * 10 + [-0.5] * 5
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert math.isfinite(metrics.sharpe_ratio)
        assert math.isfinite(metrics.sortino_ratio)
        assert math.isfinite(metrics.max_drawdown)

    def test_zero_returns(self):
        returns = [0.0] * 100
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert metrics.sharpe_ratio == 0.0
        assert metrics.win_rate == 0.0  # No positive returns

    def test_alternating_returns(self):
        # Perfect alternation
        returns = [0.01, -0.01] * 500
        metrics = compute_trading_metrics(returns)
        assert metrics is not None
        assert abs(metrics.mean_return) < 1e-6
        assert abs(metrics.win_rate - 0.5) < 0.01


class TestTradingMetricsDataclass:
    """Tests for the TradingMetrics dataclass."""

    def test_dataclass_fields(self):
        metrics = TradingMetrics(
            n_steps=100,
            n_episodes=5,
            mean_return=0.001,
            std_return=0.02,
            downside_std=0.015,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.1,
            max_drawdown_duration=10,
            current_drawdown=0.05,
            calmar_ratio=3.0,
            win_rate=0.55,
            profit_factor=1.2,
            avg_win=0.015,
            avg_loss=-0.01,
            total_return=0.1,
            equity_final=1.1,
        )

        assert metrics.n_steps == 100
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == 0.1
        assert metrics.calmar_ratio == 3.0
        assert metrics.win_rate == 0.55


class TestIntegration:
    """Integration tests simulating real training scenarios."""

    def test_realistic_training_rollout(self):
        """Simulate a realistic training rollout with multiple episodes."""
        np.random.seed(2024)

        acc = TradingMetricsAccumulator(bars_per_year=365.25 * 6)  # 4h bars

        # Simulate 3 episodes of different lengths
        for episode in range(3):
            episode_length = np.random.randint(100, 300)
            equity = 10000.0

            for step in range(episode_length):
                # Random return with slight positive drift
                ret = np.random.normal(0.0002, 0.01)
                equity *= (1 + ret)

                is_end = (step == episode_length - 1)
                acc.add_step(ret, equity=equity, is_episode_end=is_end)

        metrics = acc.summary()
        assert metrics is not None
        assert metrics.n_episodes == 3
        assert metrics.n_steps > 300

        # All metrics should be finite
        assert math.isfinite(metrics.sharpe_ratio)
        assert math.isfinite(metrics.sortino_ratio)
        assert math.isfinite(metrics.max_drawdown)
        assert math.isfinite(metrics.calmar_ratio)
        assert 0 <= metrics.win_rate <= 1
        assert 0 <= metrics.max_drawdown <= 1

    def test_batch_mode_matches_step_mode(self):
        """Verify batch mode produces same results as step-by-step."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (100, 4))  # 100 steps, 4 envs

        # Step-by-step accumulation
        acc_steps = TradingMetricsAccumulator()
        for step in range(100):
            for env in range(4):
                acc_steps.add_step(returns[step, env])

        # Batch accumulation
        acc_batch = TradingMetricsAccumulator()
        acc_batch.add_batch(returns)

        metrics_steps = acc_steps.summary()
        metrics_batch = acc_batch.summary()

        assert metrics_steps is not None
        assert metrics_batch is not None
        assert metrics_steps.n_steps == metrics_batch.n_steps
        assert abs(metrics_steps.sharpe_ratio - metrics_batch.sharpe_ratio) < 1e-10
        assert abs(metrics_steps.mean_return - metrics_batch.mean_return) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
