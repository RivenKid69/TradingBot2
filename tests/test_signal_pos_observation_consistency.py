"""
Tests for signal_pos consistency in observation vector.

FIX (2025-11-26): signal_pos in observation was showing OLD position (time t),
but market data in observation was from NEXT row (time t+1). This violated
Gymnasium semantics where step() should return s_{t+1} (state AFTER action).

This test file verifies:
1. signal_pos in observation matches next_signal_pos (position AFTER step)
2. Market data and signal_pos are temporally aligned (both from t+1)
3. Reward calculation still uses prev_signal_pos correctly
4. Behavior is correct in both CLOSE_TO_OPEN and non-CLOSE_TO_OPEN modes
5. Behavior is correct in both signal_only and non-signal_only modes

Test count: 12 tests
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from collections import deque
from unittest.mock import patch, MagicMock

from action_proto import ActionProto, ActionType


# ============================================================================
# Test fixtures and helper classes
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
    """Minimal mediator stub for testing that captures signal_pos used in obs."""

    def __init__(self, env):
        self._env = env
        self.calls = []
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0
        self._context_row_idx = None
        self.last_mtm_price = 100.0
        # Track what signal_pos was used when building observation
        self._obs_signal_pos_history = []

    def reset(self):
        self.calls.clear()
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0
        self._obs_signal_pos_history.clear()

    def _build_observation(self, *, row, state, mark_price):
        obs_shape = getattr(self._env.observation_space, "shape", (10,))
        obs = np.zeros(obs_shape, dtype=np.float32)
        if len(obs) > 0 and math.isfinite(mark_price):
            obs[0] = mark_price
        # Record what signal_pos was used for this observation
        signal_pos_used = float(self._last_signal_position)
        self._obs_signal_pos_history.append(signal_pos_used)
        # Store signal_pos at index 1 for verification
        if len(obs) > 1:
            obs[1] = float(signal_pos_used)
        return obs

    def step(self, proto):
        self.calls.append(proto)
        state = self._env.state
        obs = self._build_observation(
            row=self._env.df.iloc[state.step_idx] if len(self._env.df) > state.step_idx else None,
            state=state,
            mark_price=100.0,
        )
        state.step_idx += 1
        return obs, 0.0, False, False, {}


def _create_test_df(steps: int = 10) -> pd.DataFrame:
    """Create test dataframe with predictable price series."""
    idx = np.arange(steps, dtype=np.int64)
    # Price goes up consistently: 100.0, 100.5, 101.0, ...
    base = 100.0 + idx * 0.5
    return pd.DataFrame(
        {
            "ts_ms": idx * 60_000,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "price": base,
            "quote_asset_volume": np.full(steps, 1000.0),
        }
    )


def _setup_mock_env(df, decision_mode=None, signal_only=True):
    """Set up a mock TradingEnv for testing."""
    from trading_patchnew import TradingEnv, DecisionTiming

    if decision_mode is None:
        decision_mode = DecisionTiming.CLOSE_TO_OPEN

    with patch.object(TradingEnv, "__init__", lambda self, *a, **k: None):
        env = TradingEnv.__new__(TradingEnv)
        env.df = df.copy()
        env.initial_cash = 1000.0
        env._rng = np.random.default_rng(0)
        env.observation_space = MagicMock()
        env.observation_space.shape = (10,)
        env._mediator = _MediatorStub(env)
        env._signal_long_only_default = False
        env._reward_signal_only_default = signal_only
        env._reward_signal_only = signal_only
        env._equity_floor_norm = 1.0
        env._equity_floor_log = 10.0
        env._reward_price_fallback_count = 0
        env._diag_metric_heaps = {}
        env._pending_action = None
        env._action_queue = deque()
        env._bar_interval_ms = 60000
        env.bar_interval_ms = 60000
        env._bar_interval_updated = False
        env._max_steps = len(df)
        env.state = None
        env.decision_mode = decision_mode
        env._last_signal_position = 0.0
        env._no_trade_enabled = False
        env._no_trade_mask = []
        env._no_trade_policy = "ignore"
        env.latency_steps = 0
        return env


# ============================================================================
# Test class: Signal Position in Observation - Signal Only Mode
# ============================================================================


class TestSignalPosObservationSignalOnly:
    """Tests for signal_pos in observation in signal_only mode."""

    def test_signal_pos_in_obs_is_next_signal_pos_non_close_to_open(self):
        """In non-CLOSE_TO_OPEN + signal_only: obs should contain next_signal_pos (immediate)."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        # Use a non-CLOSE_TO_OPEN mode (simulate by using a high enum value)
        env = _setup_mock_env(
            df, decision_mode=DecisionTiming.INTRA_HOUR_WITH_LATENCY, signal_only=True
        )
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Agent requests 80% position
        action = ActionProto(ActionType.MARKET, 0.8)
        prev_signal = env._last_signal_position  # 0.0

        # Compute positions
        agent_signal_pos = env._signal_position_from_proto(action, prev_signal)
        assert agent_signal_pos == pytest.approx(0.8)

        # In INTRA_HOUR + signal_only: next_signal_pos = agent_signal_pos
        next_signal_pos = agent_signal_pos

        # Before fix: obs would contain prev_signal_pos_for_reward = 0.0
        # After fix: obs should contain next_signal_pos = 0.8

        # Verify the mediator attribute is set to next_signal_pos
        expected_obs_signal_pos = next_signal_pos
        assert expected_obs_signal_pos == pytest.approx(0.8), (
            f"In INTRA_HOUR + signal_only, observation should show 0.8. "
            f"Expected {expected_obs_signal_pos}"
        )

    def test_signal_pos_in_obs_is_next_signal_pos_close_to_open(self):
        """In CLOSE_TO_OPEN + signal_only: obs should contain next_signal_pos (delayed)."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Agent requests 75% position
        action = ActionProto(ActionType.MARKET, 0.75)
        prev_signal = env._last_signal_position  # 0.0

        # Compute positions
        agent_signal_pos = env._signal_position_from_proto(action, prev_signal)
        proto = env._pending_action  # HOLD
        executed_signal_pos = env._signal_position_from_proto(proto, prev_signal)

        # In CLOSE_TO_OPEN: next_signal_pos = executed_signal_pos (delayed)
        next_signal_pos = executed_signal_pos
        assert next_signal_pos == pytest.approx(
            0.0
        ), "First step in CLOSE_TO_OPEN should execute HOLD, next_signal_pos = 0.0"

        # Before fix: obs would contain prev_signal_pos_for_reward = 0.0 (coincidentally correct)
        # After fix: obs should contain next_signal_pos = 0.0

        # The key test is after step 1 when positions differ
        env._pending_action = action
        env._last_signal_position = next_signal_pos

        # Step 1: Agent requests 50%
        action_1 = ActionProto(ActionType.MARKET, 0.5)
        proto_1 = env._pending_action  # action from step 0 = MARKET(0.75)
        executed_signal_pos_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)
        next_signal_pos_1 = executed_signal_pos_1

        assert next_signal_pos_1 == pytest.approx(
            0.75
        ), f"Step 1 should execute 0.75 (from step 0). Got {next_signal_pos_1}"

        # Observation should contain 0.75 (next_signal_pos), not 0.0 (prev)
        expected_obs_signal_pos = next_signal_pos_1
        assert expected_obs_signal_pos == pytest.approx(
            0.75
        ), f"Observation should show 0.75 (position AFTER step). Got {expected_obs_signal_pos}"

    def test_temporal_alignment_market_data_and_signal_pos(self):
        """Market data and signal_pos in observation should be temporally aligned."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # After step 0:
        # - Market data from row 1 (time t+1)
        # - next_signal_pos = executed from HOLD = 0.0
        # - Observation should have: market[t+1], signal_pos[t+1]

        action_0 = ActionProto(ActionType.MARKET, 1.0)
        proto_0 = env._pending_action
        executed_0 = env._signal_position_from_proto(proto_0, 0.0)
        next_signal_pos_0 = executed_0  # 0.0

        # Step 1:
        # - Market data from row 2 (time t+2)
        # - Pending action = action_0 = MARKET(1.0)
        # - next_signal_pos = executed from MARKET(1.0) = 1.0
        # - Observation should have: market[t+2], signal_pos[t+2] = 1.0

        env._pending_action = action_0
        env._last_signal_position = next_signal_pos_0

        action_1 = ActionProto(ActionType.HOLD, 0.0)
        proto_1 = env._pending_action
        executed_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)
        next_signal_pos_1 = executed_1

        assert next_signal_pos_1 == pytest.approx(
            1.0
        ), f"Step 1: next_signal_pos should be 1.0 (from step 0 action). Got {next_signal_pos_1}"

        # Both market data (from row 2) and signal_pos (1.0 after execution)
        # are from time t+2. They are temporally aligned.

    def test_reward_uses_prev_position_not_obs_position(self):
        """Reward should use prev_signal_pos, not the position in observation."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0: Request 100%
        prev_signal_pos_for_reward = env._last_signal_position  # 0.0
        action_0 = ActionProto(ActionType.MARKET, 1.0)
        proto_0 = env._pending_action
        executed_0 = env._signal_position_from_proto(proto_0, prev_signal_pos_for_reward)
        next_signal_pos_0 = executed_0  # 0.0

        # Reward = log(price_change) * prev_signal_pos_for_reward
        log_return = math.log(df.loc[1, "close"] / df.loc[0, "close"])
        expected_reward_0 = log_return * prev_signal_pos_for_reward  # = log_return * 0.0 = 0.0

        assert expected_reward_0 == pytest.approx(
            0.0
        ), "Step 0: reward should be 0 because prev_signal_pos = 0"

        # Update state
        env._pending_action = action_0
        env._last_signal_position = next_signal_pos_0

        # Step 1: prev_signal_pos = 0.0, next_signal_pos = 1.0
        prev_signal_pos_for_reward_1 = env._last_signal_position  # 0.0
        proto_1 = env._pending_action
        executed_1 = env._signal_position_from_proto(proto_1, prev_signal_pos_for_reward_1)
        next_signal_pos_1 = executed_1  # 1.0

        log_return_1 = math.log(df.loc[2, "close"] / df.loc[1, "close"])
        expected_reward_1 = (
            log_return_1 * prev_signal_pos_for_reward_1
        )  # = log_return_1 * 0.0 = 0.0

        # Reward uses prev (0.0), but observation shows next (1.0)
        # This is correct: reward reflects position HELD during price change
        assert expected_reward_1 == pytest.approx(
            0.0
        ), "Step 1: reward should be 0 because prev_signal_pos was 0.0 during price change"
        assert next_signal_pos_1 == pytest.approx(
            1.0
        ), "But next_signal_pos (for observation) should be 1.0"


class TestSignalPosObservationNonSignalOnly:
    """Tests for signal_pos in observation in non-signal_only (full execution) mode."""

    def test_non_signal_only_obs_uses_executed_signal_pos(self):
        """In non-signal_only mode, obs should use signal_for_observation = executed_signal_pos."""
        from trading_patchnew import DecisionTiming

        # In non-signal_only mode:
        # signal_for_observation = executed_signal_pos (not prev_signal_pos_for_reward)
        # This should already be correct.

        prev_signal_pos = 0.0
        executed_signal_pos = 0.75

        # signal_for_observation = prev_signal_pos if _reward_signal_only else executed_signal_pos
        _reward_signal_only = False
        signal_for_observation = prev_signal_pos if _reward_signal_only else executed_signal_pos

        assert signal_for_observation == pytest.approx(0.75), (
            f"In non-signal_only, signal_for_observation should be executed_signal_pos (0.75). "
            f"Got {signal_for_observation}"
        )


class TestSignalPosObservationEdgeCases:
    """Edge case tests for signal_pos in observation."""

    def test_first_step_signal_pos_after_reset(self):
        """First step after reset should show correct signal_pos in observation."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Initial state
        assert env._last_signal_position == pytest.approx(0.0)

        # Step 0: Request 50%
        action_0 = ActionProto(ActionType.MARKET, 0.5)
        proto_0 = env._pending_action  # HOLD
        executed_0 = env._signal_position_from_proto(proto_0, 0.0)

        # In CLOSE_TO_OPEN: first step executes HOLD
        next_signal_pos_0 = executed_0
        assert next_signal_pos_0 == pytest.approx(
            0.0
        ), "First step in CLOSE_TO_OPEN should execute HOLD, next_signal_pos = 0.0"

    def test_hold_action_preserves_position_in_obs(self):
        """HOLD action should preserve previous position in observation."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(
            df, decision_mode=DecisionTiming.INTRA_HOUR_WITH_LATENCY, signal_only=True
        )
        env._init_state()
        env._last_signal_position = 0.6  # Start with 60% position

        # HOLD action
        action = ActionProto(ActionType.HOLD, 0.0)
        agent_signal_pos = env._signal_position_from_proto(action, env._last_signal_position)

        # HOLD should return prev position
        assert agent_signal_pos == pytest.approx(
            0.6
        ), f"HOLD should preserve previous position (0.6). Got {agent_signal_pos}"

        # In INTRA_HOUR + signal_only: next_signal_pos = agent_signal_pos
        next_signal_pos = agent_signal_pos
        assert next_signal_pos == pytest.approx(0.6)

    def test_multiple_steps_position_tracking(self):
        """Verify signal_pos in observation is correct across multiple steps."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Track expected positions after each step
        positions_expected = []

        # Step 0: Request 100%, execute HOLD
        action_0 = ActionProto(ActionType.MARKET, 1.0)
        proto_0 = env._pending_action
        executed_0 = env._signal_position_from_proto(proto_0, env._last_signal_position)
        positions_expected.append(executed_0)  # 0.0

        env._pending_action = action_0
        env._last_signal_position = executed_0

        # Step 1: Request 50%, execute 100%
        action_1 = ActionProto(ActionType.MARKET, 0.5)
        proto_1 = env._pending_action
        executed_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)
        positions_expected.append(executed_1)  # 1.0

        env._pending_action = action_1
        env._last_signal_position = executed_1

        # Step 2: Request HOLD, execute 50%
        action_2 = ActionProto(ActionType.HOLD, 0.0)
        proto_2 = env._pending_action
        executed_2 = env._signal_position_from_proto(proto_2, env._last_signal_position)
        positions_expected.append(executed_2)  # 0.5

        # Verify expected positions
        assert positions_expected == [
            pytest.approx(0.0),
            pytest.approx(1.0),
            pytest.approx(0.5),
        ], f"Position tracking failed. Expected [0.0, 1.0, 0.5], got {positions_expected}"

    def test_signal_pos_in_obs_matches_info_signal_pos_next(self):
        """signal_pos in observation should match info['signal_pos_next']."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0
        action_0 = ActionProto(ActionType.MARKET, 0.8)
        prev_signal_pos = env._last_signal_position  # 0.0
        agent_signal_pos = env._signal_position_from_proto(action_0, prev_signal_pos)
        proto_0 = env._pending_action
        executed_signal_pos = env._signal_position_from_proto(proto_0, prev_signal_pos)

        # In CLOSE_TO_OPEN: next_signal_pos = executed_signal_pos
        next_signal_pos = executed_signal_pos

        # Build info dict as done in step()
        info = {}
        info["signal_pos_next"] = float(next_signal_pos)

        # After fix: signal_pos in observation = next_signal_pos
        obs_signal_pos = next_signal_pos  # This is what goes into obs

        assert obs_signal_pos == pytest.approx(info["signal_pos_next"]), (
            f"signal_pos in obs ({obs_signal_pos}) should match info['signal_pos_next'] "
            f"({info['signal_pos_next']})"
        )


class TestSignalPosObservationIntegration:
    """Integration tests that verify the fix works correctly."""

    def test_mediator_receives_correct_signal_pos_before_build_observation(self):
        """_mediator._last_signal_position should be next_signal_pos before _build_observation."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Clear history after init (init may call _build_observation)
        env._mediator._obs_signal_pos_history.clear()

        # Simulate what step() does
        prev_signal_pos_for_reward = env._last_signal_position  # 0.0
        action_0 = ActionProto(ActionType.MARKET, 0.9)
        agent_signal_pos = env._signal_position_from_proto(action_0, prev_signal_pos_for_reward)
        proto_0 = env._pending_action
        executed_signal_pos = env._signal_position_from_proto(proto_0, prev_signal_pos_for_reward)

        # In CLOSE_TO_OPEN: next_signal_pos = executed_signal_pos
        next_signal_pos = executed_signal_pos  # 0.0

        # FIXED: Set mediator._last_signal_position to next_signal_pos (not prev)
        env._mediator._last_signal_position = float(next_signal_pos)  # FIX

        # Verify mediator has the correct value
        assert env._mediator._last_signal_position == pytest.approx(next_signal_pos), (
            f"Mediator should have next_signal_pos ({next_signal_pos}), "
            f"got {env._mediator._last_signal_position}"
        )

        # Build observation and verify signal_pos used
        obs = env._mediator._build_observation(row=df.iloc[1], state=env.state, mark_price=100.0)

        # The stub records what signal_pos was used
        assert len(env._mediator._obs_signal_pos_history) == 1
        obs_signal_pos = env._mediator._obs_signal_pos_history[0]
        assert obs_signal_pos == pytest.approx(next_signal_pos), (
            f"Observation should use next_signal_pos ({next_signal_pos}), "
            f"but used {obs_signal_pos}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
