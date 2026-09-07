"""
Tests for step() returning observation from NEXT row (2025-11-25).

Issue: step() was returning observation from the SAME row as the action was based on,
violating Gymnasium semantics. After reset() returned obs from row[0], the first step()
was also returning obs from row[0], creating duplicate observations.

Fix: Build observation from next_idx row instead of current_idx row.
This follows Gymnasium semantics: step() returns the NEXT state observation.

Reference: https://gymnasium.farama.org/api/env/#gymnasium.Env.step
Test count: 6 tests
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
    """Mediator stub that records which row_idx was used for observation."""

    def __init__(self, env):
        self._env = env
        self.calls = []
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0
        self._context_row_idx = None
        self.last_mtm_price = 100.0
        self.observation_row_indices = []  # Track which rows were used

    def reset(self):
        self.calls.clear()
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0
        self.observation_row_indices.clear()

    def _build_observation(self, *, row, state, mark_price):
        """Build observation and record which row was used."""
        obs_shape = getattr(self._env.observation_space, "shape", (10,))
        obs = np.zeros(obs_shape, dtype=np.float32)

        # Record the row index used
        row_idx = None
        if row is not None and hasattr(row, "name"):
            row_idx = int(row.name)
        self.observation_row_indices.append(row_idx)

        # Fill observation with data from the row
        if row is not None:
            pos = 0
            for key in ("open", "high", "low", "close", "price"):
                if key in row.index and pos < len(obs):
                    val = float(row.get(key, 0.0))
                    if math.isfinite(val):
                        obs[pos] = val
                    pos += 1
        if len(obs) > 0 and math.isfinite(mark_price):
            obs[0] = mark_price
        return obs

    def step(self, proto):
        """Minimal step stub - not used directly in signal_only mode."""
        self.calls.append(proto)
        state = self._env.state
        obs = self._build_observation(
            row=self._env.df.iloc[state.step_idx] if len(self._env.df) > state.step_idx else None,
            state=state,
            mark_price=100.0,
        )
        state.step_idx += 1
        return obs, 0.0, False, False, {}


# ============================================================================
# Test fixtures
# ============================================================================


def _create_test_df_distinguishable() -> pd.DataFrame:
    """Create test dataframe with clearly distinguishable rows."""
    # Each row has unique values to verify which row observation comes from
    return pd.DataFrame(
        {
            "ts_ms": [0, 60000, 120000, 180000, 240000],
            "open": [100.0, 200.0, 300.0, 400.0, 500.0],
            "high": [110.0, 210.0, 310.0, 410.0, 510.0],
            "low": [90.0, 190.0, 290.0, 390.0, 490.0],
            "close": [105.0, 205.0, 305.0, 405.0, 505.0],
            "price": [105.0, 205.0, 305.0, 405.0, 505.0],
            "quote_asset_volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )


def _setup_mock_env(df):
    """Set up a mock TradingEnv for testing."""
    from trading_patchnew import TradingEnv

    with patch.object(TradingEnv, "__init__", lambda self, *a, **k: None):
        env = TradingEnv.__new__(TradingEnv)
        env.df = df.copy()
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
        env.bar_interval_ms = 60000
        env._bar_interval_updated = False
        env._max_steps = len(df)
        env.state = None
        return env


# ============================================================================
# Tests for step() returning NEXT row observation
# ============================================================================


class TestStepReturnsNextRowObservation:
    """Tests verifying step() returns observation from NEXT row."""

    def test_first_step_returns_row1_not_row0(self):
        """First step should return observation from row[1], not row[0]."""
        df = _create_test_df_distinguishable()
        env = _setup_mock_env(df)

        # Initialize environment
        obs_reset, _ = env._init_state()

        # Verify reset returned row[0] data
        reset_obs_row = env._mediator.observation_row_indices[-1]
        assert reset_obs_row == 0, f"Expected reset to use row 0, got {reset_obs_row}"

        # Reset tracking
        env._mediator.observation_row_indices.clear()

        # Simulate first step
        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        # Call _signal_only_step
        row_idx = env.state.step_idx  # Should be 0
        row = env.df.iloc[row_idx]
        mark_price = 100.0

        obs, reward, term, trunc, info = env._signal_only_step(
            proto, row_idx, row, mark_price, next_signal_pos=0.0
        )

        # Verify step returned row[1] observation (NEXT row)
        step_obs_row = env._mediator.observation_row_indices[-1]
        assert step_obs_row == 1, (
            f"Expected first step to use row 1 for observation, got row {step_obs_row}. "
            f"This violates Gymnasium semantics - step() should return NEXT state."
        )

    def test_step_observations_are_sequential(self):
        """Each step should return observation from the next row."""
        df = _create_test_df_distinguishable()
        env = _setup_mock_env(df)

        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        # Initialize
        env._init_state()
        env._mediator.observation_row_indices.clear()

        expected_rows = []
        actual_rows = []

        # Run several steps
        for step_num in range(3):
            row_idx = env.state.step_idx
            row = env.df.iloc[row_idx]

            env._signal_only_step(proto, row_idx, row, 100.0, next_signal_pos=0.0)

            # Expected: observation from row_idx + 1 (next row)
            expected_next = min(row_idx + 1, len(df) - 1)
            expected_rows.append(expected_next)
            actual_rows.append(env._mediator.observation_row_indices[-1])

        assert (
            actual_rows == expected_rows
        ), f"Expected observation row sequence {expected_rows}, got {actual_rows}"

    def test_reset_and_first_step_observations_differ(self):
        """Reset observation and first step observation should be different."""
        df = _create_test_df_distinguishable()
        env = _setup_mock_env(df)

        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        # Get reset observation
        obs_reset, _ = env._init_state()

        # Get first step observation
        row_idx = env.state.step_idx
        row = env.df.iloc[row_idx]
        obs_step, _, _, _, _ = env._signal_only_step(
            proto, row_idx, row, 100.0, next_signal_pos=0.0
        )

        # Observations should differ (from different rows)
        assert not np.allclose(obs_reset, obs_step, atol=1e-6), (
            "Reset and first step should return different observations! "
            f"reset_obs[:5]={obs_reset[:5]}, step_obs[:5]={obs_step[:5]}"
        )

    def test_terminal_step_uses_last_row(self):
        """At terminal step, observation should still be from last valid row."""
        df = _create_test_df_distinguishable()
        env = _setup_mock_env(df)

        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        env._init_state()
        env._mediator.observation_row_indices.clear()

        # Advance to near end
        env.state.step_idx = len(df) - 2  # Second to last

        row_idx = env.state.step_idx
        row = env.df.iloc[row_idx]
        obs, _, term, trunc, _ = env._signal_only_step(
            proto, row_idx, row, 100.0, next_signal_pos=0.0
        )

        # Should use last row (index len(df)-1 = 4)
        obs_row = env._mediator.observation_row_indices[-1]
        expected = len(df) - 1
        assert (
            obs_row == expected
        ), f"Expected terminal observation from row {expected}, got {obs_row}"

    def test_observation_mark_price_from_next_row(self):
        """Observation mark price should be from next row."""
        df = _create_test_df_distinguishable()
        env = _setup_mock_env(df)

        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        env._init_state()

        # First step: should use row[1] price (205.0) for observation
        row_idx = env.state.step_idx  # 0
        row = env.df.iloc[row_idx]
        obs, _, _, _, _ = env._signal_only_step(proto, row_idx, row, 100.0, next_signal_pos=0.0)

        # obs[0] should be mark_price from row[1]
        expected_price = df.iloc[1]["close"]  # 205.0
        assert obs[0] == pytest.approx(
            expected_price, rel=0.1
        ), f"Expected obs mark_price ~{expected_price}, got {obs[0]}"


class TestStepObservationEdgeCases:
    """Edge case tests for step observation fix."""

    def test_single_row_dataframe(self):
        """With single row dataframe, all observations should use row[0]."""
        df = pd.DataFrame(
            {
                "ts_ms": [0],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "price": [105.0],
                "quote_asset_volume": [1000.0],
            }
        )
        env = _setup_mock_env(df)

        from action_proto import ActionProto, ActionType

        proto = ActionProto(ActionType.HOLD, 0.0)

        env._init_state()
        env._mediator.observation_row_indices.clear()

        row_idx = env.state.step_idx
        row = env.df.iloc[row_idx]
        obs, _, term, trunc, _ = env._signal_only_step(
            proto, row_idx, row, 100.0, next_signal_pos=0.0
        )

        # With single row, next_idx=1 but capped to len(df)-1=0
        obs_row = env._mediator.observation_row_indices[-1]
        assert obs_row == 0, f"Expected row 0 for single-row df, got {obs_row}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
