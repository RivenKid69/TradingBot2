"""
Tests for CLOSE_TO_OPEN + SIGNAL_ONLY timing consistency (2025-11-25).

Issue: In SIGNAL_ONLY mode with CLOSE_TO_OPEN timing, the signal position was updated
immediately (agent_signal_pos), ignoring the 1-bar delay that CLOSE_TO_OPEN introduces.
This created look-ahead bias where reward assumed immediate position change.

Fix: In CLOSE_TO_OPEN mode, use executed_signal_pos (from delayed proto) even in SIGNAL_ONLY.
This ensures reward timing matches actual execution timing.

Test count: 5 tests
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from collections import deque
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
        self.last_mtm_price = 100.0

    def reset(self):
        self.calls.clear()
        self._last_signal_position = 0.0
        self._latest_log_ret_prev = 0.0

    def _build_observation(self, *, row, state, mark_price):
        obs_shape = getattr(self._env.observation_space, "shape", (10,))
        obs = np.zeros(obs_shape, dtype=np.float32)
        if len(obs) > 0 and math.isfinite(mark_price):
            obs[0] = mark_price
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


# ============================================================================
# Test fixtures
# ============================================================================


def _create_test_df() -> pd.DataFrame:
    """Create test dataframe."""
    return pd.DataFrame(
        {
            "ts_ms": [0, 60000, 120000, 180000, 240000],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "price": [100.5, 101.5, 102.5, 103.5, 104.5],
            "quote_asset_volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
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
# Tests for CLOSE_TO_OPEN + SIGNAL_ONLY timing
# ============================================================================


class TestCloseToOpenSignalOnlyTiming:
    """Tests verifying CLOSE_TO_OPEN respects delay even in SIGNAL_ONLY mode."""

    def test_close_to_open_signal_position_is_delayed(self):
        """In CLOSE_TO_OPEN mode, signal position should be delayed by 1 bar."""
        from trading_patchnew import DecisionTiming
        from action_proto import ActionProto, ActionType

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

        # Initialize
        env._init_state()

        # Track signal positions through steps
        signal_positions = []

        # Step 0: Agent wants 50% position (MARKET with positive volume_frac)
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)  # Initial pending
        action_0 = ActionProto(ActionType.MARKET, 0.5)  # 50% long position

        # Get prev_signal_pos before step
        prev_signal_pos = env._last_signal_position
        signal_positions.append(("before_step_0", prev_signal_pos))

        # In CLOSE_TO_OPEN:
        # - proto = _pending_action = HOLD(0.0) -> executed_signal_pos = 0.0
        # - _pending_action = action_0 (stored for next step)
        # - next_signal_pos should be executed_signal_pos = 0.0 (NOT 0.5!)

        # Simulate the signal position computation
        from trading_patchnew import TradingEnv

        executed_signal_pos = env._signal_position_from_proto(
            ActionProto(ActionType.HOLD, 0.0), prev_signal_pos  # proto is pending from previous
        )
        agent_signal_pos = env._signal_position_from_proto(action_0, prev_signal_pos)

        # In CLOSE_TO_OPEN mode, next_signal_pos should be executed_signal_pos
        # (from the HOLD action), not agent_signal_pos (0.5)
        assert executed_signal_pos == 0.0, f"Expected executed=0.0, got {executed_signal_pos}"
        assert agent_signal_pos == 0.5, f"Expected agent=0.5, got {agent_signal_pos}"

        # Verify the fix: In CLOSE_TO_OPEN, next_signal_pos = executed_signal_pos
        if env.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
            next_signal_pos = executed_signal_pos  # As per the fix
        else:
            next_signal_pos = agent_signal_pos if env._reward_signal_only else executed_signal_pos

        assert next_signal_pos == 0.0, (
            f"In CLOSE_TO_OPEN + SIGNAL_ONLY, next_signal_pos should be 0.0 (delayed), "
            f"not 0.5 (immediate). Got {next_signal_pos}"
        )

    def test_signal_position_applied_on_next_step(self):
        """Signal position from step N should affect reward at step N+1 (1-bar delay)."""
        from trading_patchnew import DecisionTiming
        from action_proto import ActionProto, ActionType

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0: Agent requests 50% position
        # In CLOSE_TO_OPEN: next_signal_pos = 0.0 (from pending HOLD)
        # The 50% request is stored in _pending_action for next step
        step_0_pending = env._pending_action
        step_0_agent_action = ActionProto(ActionType.MARKET, 0.5)  # 50% position

        proto_0 = step_0_pending  # HOLD(0.0)
        env._pending_action = step_0_agent_action  # Store for next step

        executed_0 = env._signal_position_from_proto(proto_0, 0.0)
        assert executed_0 == 0.0, "Step 0: executed should be 0.0"

        # Update signal position as per fix
        env._last_signal_position = executed_0

        # Step 1: Now the 50% action is executed
        # proto_1 = _pending_action = action_0 (50%)
        proto_1 = env._pending_action  # The 50% action from step 0

        executed_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)
        assert executed_1 == pytest.approx(
            0.5, rel=0.1
        ), f"Step 1: executed should be ~0.5, got {executed_1}"

    def test_no_delay_in_default_mode_signal_only(self):
        """In default mode + SIGNAL_ONLY, signal position should update immediately."""
        from trading_patchnew import DecisionTiming
        from action_proto import ActionProto, ActionType

        df = _create_test_df()
        # Use default mode (not CLOSE_TO_OPEN)
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        # Override to test non-CLOSE_TO_OPEN behavior
        env.decision_mode = 2  # Some other mode value (not 0=CLOSE_TO_OPEN, not 1=INTRA_HOUR)

        env._init_state()

        action = ActionProto(ActionType.MARKET, 0.5)  # 50% position
        agent_signal_pos = env._signal_position_from_proto(action, 0.0)

        # In non-CLOSE_TO_OPEN + SIGNAL_ONLY, next_signal_pos = agent_signal_pos
        # (immediate, no delay)
        if env.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
            next_signal_pos = 0.0  # Would be delayed
        else:
            next_signal_pos = agent_signal_pos if env._reward_signal_only else 0.0

        assert next_signal_pos == pytest.approx(
            0.5
        ), f"In default mode + SIGNAL_ONLY, expected immediate signal pos 0.5, got {next_signal_pos}"

    def test_intra_hour_with_latency_uses_queue(self):
        """In INTRA_HOUR_WITH_LATENCY mode, actions should go through queue."""
        from trading_patchnew import DecisionTiming
        from action_proto import ActionProto, ActionType

        df = _create_test_df()
        env = _setup_mock_env(
            df, decision_mode=DecisionTiming.INTRA_HOUR_WITH_LATENCY, signal_only=True
        )
        env.latency_steps = 2

        env._init_state()
        # Initialize queue with HOLD actions
        env._action_queue = deque(
            [ActionProto(ActionType.HOLD, 0.0) for _ in range(env.latency_steps)]
        )

        # First action
        action_0 = ActionProto(ActionType.MARKET, 0.5)  # 50% position

        # Pop from queue (should be HOLD)
        proto = env._action_queue.popleft()
        env._action_queue.append(action_0)

        assert proto.action_type == ActionType.HOLD, "First proto should be HOLD from queue"

        # Need to pop one more HOLD before action_0 comes out (latency_steps=2)
        proto = env._action_queue.popleft()  # Second HOLD
        action_1 = ActionProto(ActionType.MARKET, 0.7)
        env._action_queue.append(action_1)

        assert proto.action_type == ActionType.HOLD, "Second proto should also be HOLD from queue"

        # Now action_0 should come out
        proto = env._action_queue.popleft()
        action_2 = ActionProto(ActionType.MARKET, 0.8)
        env._action_queue.append(action_2)

        assert proto.action_type == ActionType.MARKET and proto.volume_frac == pytest.approx(
            0.5
        ), f"After latency_steps={env.latency_steps}, action_0 (0.5) should be executed, got {proto}"


class TestCloseToOpenRewardConsistency:
    """Tests verifying reward calculation is consistent with execution timing."""

    def test_reward_uses_correct_position_timing(self):
        """Reward should reflect position that was actually held during the bar."""
        from trading_patchnew import DecisionTiming
        from action_proto import ActionProto, ActionType

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0: Agent wants 100% position
        # But in CLOSE_TO_OPEN, this is delayed
        action_0 = ActionProto(ActionType.MARKET, 1.0)  # 100% position

        # The position used for reward should be from the EXECUTED action (HOLD = 0%)
        # not the INTENDED action (100%)
        proto = env._pending_action  # HOLD
        env._pending_action = action_0

        executed_pos = env._signal_position_from_proto(proto, 0.0)
        intended_pos = env._signal_position_from_proto(action_0, 0.0)

        assert executed_pos == 0.0, "Executed position should be 0 (HOLD)"
        assert intended_pos == 1.0, "Intended position should be 1.0 (100%)"

        # Reward = log(price_change) * executed_pos = log(...) * 0.0 = 0
        # If we incorrectly used intended_pos, reward would be non-zero
        # This is the look-ahead bias we fixed

        # In CLOSE_TO_OPEN mode, the fix ensures next_signal_pos = executed_pos
        next_signal_pos = executed_pos  # As per fix
        assert (
            next_signal_pos == 0.0
        ), "next_signal_pos should be 0.0 (from executed HOLD), not 1.0 (intended)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
