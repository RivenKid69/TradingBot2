"""
Tests for TradingEnv reset() observation and reward price fixes (2025-11-25).

These tests verify three issues fixed in trading_patchnew.py:

Issue #1: Zero observation at reset() [CRITICAL]
    Problem: reset() returned np.zeros(...), violating Gymnasium semantics.
    Impact: LSTM received zeros as first input; policy made first decision on garbage.
    Fix: Build actual observation from row 0 using _mediator._build_observation().
    Location: trading_patchnew.py:830-868

Issue #2: Redundant _last_signal_position assignment [CODE SMELL]
    Problem: In signal_only mode, _last_signal_position was set twice with same value.
    Impact: No functional bug - values were identical (next_signal_pos == agent_signal_pos).
    Fix: Removed redundant assignment, kept info dict updates.
    Location: trading_patchnew.py:2101-2109

Issue #3: Reward=0 stuck with invalid price data [MEDIUM]
    Problem: If data starts with NaN close prices, _last_reward_price = 0.0 and
             subsequent steps couldn't fallback (prev_price=0 fails >0 check).
    Impact: reward=0 for several steps until first valid price appears.
    Fix: Try multiple fallbacks: close → open → scan first 10 rows.
    Location: trading_patchnew.py:791-828

Reference: docs/PLATFORM_REFERENCE.md sections "Troubleshooting" and "История критических исправлений"
Test count: 9 tests (2 for Issue #1, 1 for Issue #2, 3 for Issue #3, 3 regression)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Minimal stub classes for testing without Cython dependencies
# ============================================================================

class _EnvState:
    """Minimal state stub for testing."""
    def __init__(self):
        self.cash = 1000.0
        self.units = 0.0
        self.net_worth = 1000.0
        self.step_idx = 0
        self.peak_value = 1000.0
        self.max_position = 1.0
        self.max_position_risk_on = 1.0
        self.is_bankrupt = False


class _MediatorStub:
    """Minimal mediator stub for testing."""

    def __init__(self, env):
        self._env = env
        self.calls = []
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0
        self._context_row_idx = None

    def reset(self):
        self.calls.clear()
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0

    def _build_observation(self, *, row, state, mark_price):
        """Build a non-zero observation for testing."""
        obs_shape = getattr(self._env.observation_space, "shape", (10,))
        obs = np.zeros(obs_shape, dtype=np.float32)
        # Fill with some recognizable values based on row data
        if row is not None:
            pos = 0
            for key in ("open", "high", "low", "close", "price"):
                if key in row.index and pos < len(obs):
                    val = float(row.get(key, 0.0))
                    if math.isfinite(val):
                        obs[pos] = val
                    pos += 1
        # Add mark price
        if len(obs) > 0 and math.isfinite(mark_price):
            obs[0] = mark_price
        return obs

    def step(self, proto):
        """Minimal step for testing."""
        state = self._env.state
        self.calls.append(proto)
        obs = self._build_observation(
            row=self._env.df.iloc[state.step_idx] if len(self._env.df) > state.step_idx else None,
            state=state,
            mark_price=100.0
        )
        state.step_idx += 1
        return obs, 0.0, False, False, {}


# ============================================================================
# Test fixtures
# ============================================================================

def _create_test_df_valid() -> pd.DataFrame:
    """Create test dataframe with valid data."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.0, 102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [100.5, 101.5, 102.5, 103.5],
        "price": [100.5, 101.5, 102.5, 103.5],
        "quote_asset_volume": [10.0, 10.0, 10.0, 10.0],
        "ts_ms": [0, 60000, 120000, 180000],
    })


def _create_test_df_nan_close_row0() -> pd.DataFrame:
    """Create test dataframe with NaN close at row 0."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.0, 102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [float("nan"), 101.5, 102.5, 103.5],  # NaN at row 0
        "price": [float("nan"), 101.5, 102.5, 103.5],
        "quote_asset_volume": [10.0, 10.0, 10.0, 10.0],
        "ts_ms": [0, 60000, 120000, 180000],
    })


def _create_test_df_nan_close_multiple() -> pd.DataFrame:
    """Create test dataframe with NaN close at first several rows."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        "close": [float("nan"), float("nan"), float("nan"), 103.5, 104.5, 105.5],
        "price": [float("nan"), float("nan"), float("nan"), 103.5, 104.5, 105.5],
        "quote_asset_volume": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "ts_ms": [0, 60000, 120000, 180000, 240000, 300000],
    })


# ============================================================================
# Test Issue #1: reset() should return actual observation, not zeros
# ============================================================================

class TestResetObservationNotZeros:
    """Tests for Issue #1: reset() should return actual observation from row 0."""

    def test_reset_observation_not_all_zeros_with_valid_data(self):
        """Reset should return non-zero observation when data is valid."""
        # This test verifies the FIX by checking that:
        # 1. reset() calls _build_observation with row 0
        # 2. The returned observation is not all zeros

        df = _create_test_df_valid()

        # Mock TradingEnv to avoid Cython dependencies
        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            # Set up minimal required attributes
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False

            # Call _init_state directly
            obs, info = env._init_state()

            # Verify: observation should NOT be all zeros
            # At minimum, the mark_price (close or open) should be populated
            assert obs is not None
            assert obs.shape == (10,)
            # With valid data, obs should have non-zero elements
            assert np.count_nonzero(obs) > 0, (
                f"Expected non-zero observation from row 0, got all zeros. "
                f"obs={obs}"
            )

    def test_reset_observation_reflects_row0_data(self):
        """Reset observation should contain data from row 0."""
        df = _create_test_df_valid()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False

            obs, info = env._init_state()

            # The mediator stub puts mark_price in obs[0]
            # mark_price should be close from row 0 = 100.5
            expected_price = df.iloc[0]["close"]
            assert obs[0] == pytest.approx(expected_price, rel=0.01), (
                f"Expected obs[0] to be close to {expected_price}, got {obs[0]}"
            )


# ============================================================================
# Test Issue #2: Redundant _last_signal_position assignment removed
# ============================================================================

class TestRedundantSignalPositionRemoved:
    """Tests for Issue #2: Redundant _last_signal_position assignment."""

    def test_signal_only_next_signal_pos_equals_agent_signal_pos(self):
        """In signal_only mode, next_signal_pos should equal agent_signal_pos."""
        # This test verifies the precondition for the fix:
        # In signal_only mode, next_signal_pos = agent_signal_pos (line 1621-1622)
        # Therefore the second assignment at line 2088-2093 was redundant

        # Verify the code logic by checking the assignment pattern:
        # 1. Line 1621-1622: next_signal_pos = agent_signal_pos if self._reward_signal_only else ...
        # 2. Line 2000: self._last_signal_position = float(next_signal_pos)
        # 3. Old line 2088-2093 (REMOVED): if self._reward_signal_only: self._last_signal_position = float(agent_signal_pos)

        # The fix removed the redundant code, so we verify it's gone:
        import inspect
        from trading_patchnew import TradingEnv

        source = inspect.getsource(TradingEnv.step)

        # Check that there's only ONE assignment of _last_signal_position in step()
        # that's not in a comment
        lines = source.split('\n')
        assignment_count = 0
        for line in lines:
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if 'self._last_signal_position = float(next_signal_pos)' in stripped:
                assignment_count += 1
            if 'self._last_signal_position = float(agent_signal_pos)' in stripped:
                assignment_count += 1

        # After the fix, there should be exactly 1 assignment (line 2000)
        # The redundant assignment at line 2088-2093 should be removed
        assert assignment_count == 1, (
            f"Expected exactly 1 _last_signal_position assignment in step(), "
            f"found {assignment_count}. The redundant assignment may not have been removed."
        )


# ============================================================================
# Test Issue #3: Improved _last_reward_price initialization
# ============================================================================

class TestImprovedRewardPriceInitialization:
    """Tests for Issue #3: Better _last_reward_price initialization."""

    def test_fallback_to_open_price_when_close_is_nan(self):
        """When close is NaN at row 0, should fallback to open price."""
        df = _create_test_df_nan_close_row0()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False
            # No close_actual so it will use close column
            if hasattr(env, '_close_actual'):
                delattr(env, '_close_actual')

            env._init_state()

            # _last_reward_price should be set to open price (100.0) as fallback
            expected_price = df.iloc[0]["open"]  # 100.0
            assert env._last_reward_price == pytest.approx(expected_price), (
                f"Expected _last_reward_price={expected_price} (open price), "
                f"got {env._last_reward_price}"
            )

    def test_scan_multiple_rows_for_valid_price(self):
        """When first several rows have invalid close, scan for valid price."""
        df = _create_test_df_nan_close_multiple()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False
            if hasattr(env, '_close_actual'):
                delattr(env, '_close_actual')

            # Since open is valid (100.0), it should be used as fallback
            # even before scanning for other rows
            env._init_state()

            # open[0] = 100.0 is valid, so that should be used
            expected_price = df.iloc[0]["open"]  # 100.0
            assert env._last_reward_price > 0.0, (
                f"Expected _last_reward_price > 0, got {env._last_reward_price}"
            )
            assert env._last_reward_price == pytest.approx(expected_price, rel=0.1), (
                f"Expected _last_reward_price near {expected_price}, "
                f"got {env._last_reward_price}"
            )

    def test_valid_data_uses_close_price(self):
        """With valid data, should use close price directly."""
        df = _create_test_df_valid()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False
            if hasattr(env, '_close_actual'):
                delattr(env, '_close_actual')

            env._init_state()

            # With valid data, close[0] = 100.5 should be used
            expected_price = df.iloc[0]["close"]  # 100.5
            assert env._last_reward_price == pytest.approx(expected_price), (
                f"Expected _last_reward_price={expected_price} (close price), "
                f"got {env._last_reward_price}"
            )


# ============================================================================
# Integration test: Full flow verification
# ============================================================================

class TestResetObservationIntegration:
    """Integration tests for the reset observation flow."""

    def test_reset_then_step_observation_difference(self):
        """Reset observation should be similar to step 0 observation (same data source)."""
        df = _create_test_df_valid()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False

            # Get reset observation
            obs_reset, _ = env._init_state()

            # The observation from reset should be based on row 0
            # obs[0] should contain mark_price from row 0
            assert obs_reset[0] > 0, "Reset observation should have non-zero mark_price"

            # Both reset and first step use row 0 data
            # With the fix, reset observation should be meaningful
            assert np.count_nonzero(obs_reset) > 0, (
                "Reset observation should have non-zero elements"
            )


# ============================================================================
# Regression tests: Ensure existing behavior is preserved
# ============================================================================

class TestResetObservationRegression:
    """Regression tests to ensure fixes don't break existing functionality."""

    def test_empty_dataframe_returns_zeros(self):
        """With empty dataframe, should safely return zeros."""
        df = pd.DataFrame()

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False

            obs, info = env._init_state()

            # With empty df, should return zeros (safe fallback)
            assert obs is not None
            assert obs.shape == (10,)
            assert np.all(obs == 0.0), "Empty df should return zeros observation"

    def test_last_reward_price_zero_with_all_nan_data(self):
        """With all NaN data, _last_reward_price should be 0 (safe fallback)."""
        df = pd.DataFrame({
            "open": [float("nan")] * 4,
            "high": [float("nan")] * 4,
            "low": [float("nan")] * 4,
            "close": [float("nan")] * 4,
            "price": [float("nan")] * 4,
            "quote_asset_volume": [10.0] * 4,
            "ts_ms": [0, 60000, 120000, 180000],
        })

        from trading_patchnew import TradingEnv

        with patch.object(TradingEnv, '__init__', lambda self, *a, **k: None):
            env = TradingEnv.__new__(TradingEnv)
            env.df = df
            env.initial_cash = 1000.0
            env._rng = np.random.default_rng(0)
            env.observation_space = MagicMock()
            env.observation_space.shape = (10,)
            env._mediator = _MediatorStub(env)
            env._signal_long_only_default = False
            env._reward_signal_only_default = True
            env._equity_floor_norm = 1.0
            env._equity_floor_log = 10.0
            env._reward_price_fallback_count = 0
            env._diag_metric_heaps = {}
            env._pending_action = None
            env._action_queue = []
            env._bar_interval_ms = 60000
            env.bar_interval_ms = 60000  # Public attribute used by get_bar_interval_seconds
            env._bar_interval_updated = False
            if hasattr(env, '_close_actual'):
                delattr(env, '_close_actual')

            env._init_state()

            # With all NaN, _last_reward_price should be 0 (safe fallback)
            # This is expected behavior - data is completely invalid
            assert env._last_reward_price == 0.0, (
                f"Expected _last_reward_price=0.0 with all NaN data, "
                f"got {env._last_reward_price}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
