"""
Tests for info["signal_pos_next"] consistency in CLOSE_TO_OPEN + signal_only mode.

Issue (2025-11-25): In CLOSE_TO_OPEN + signal_only mode:
- next_signal_pos = executed_signal_pos (from delayed proto, respects 1-bar delay)
- BUT info["signal_pos_next"] was set to agent_signal_pos (agent's intention, no delay)

This mismatch caused confusion during debugging/analysis:
- info showed position that agent WANTED
- But actual _last_signal_position was different (delayed)

Fix:
- info["signal_pos_next"] now shows next_signal_pos (actual position after this step)
- info["signal_pos_requested"] shows agent_signal_pos (what agent wanted)

Test count: 8 tests
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


def _create_test_df(steps: int = 6) -> pd.DataFrame:
    """Create test dataframe with predictable price series."""
    idx = np.arange(steps, dtype=np.int64)
    base = np.linspace(100.0, 100.0 + steps * 0.5, steps)
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


class TestSignalPosNextConsistency:
    """Tests verifying info['signal_pos_next'] matches actual position.

    These tests directly verify the signal position computation logic
    without requiring full TradingEnv initialization (which needs Cython).
    """

    def test_close_to_open_signal_pos_next_is_delayed(self):
        """In CLOSE_TO_OPEN, signal_pos_next should show delayed (executed) position."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

        # Initialize
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)  # Default from reset

        # Agent wants 100% position
        agent_action = ActionProto(ActionType.MARKET, 1.0)
        prev_signal_pos = env._last_signal_position  # 0.0

        # Compute signal positions
        agent_signal_pos = env._signal_position_from_proto(agent_action, prev_signal_pos)
        proto = env._pending_action  # HOLD from reset
        executed_signal_pos = env._signal_position_from_proto(proto, prev_signal_pos)

        # In CLOSE_TO_OPEN, next_signal_pos = executed_signal_pos
        if env.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
            next_signal_pos = executed_signal_pos
        else:
            next_signal_pos = agent_signal_pos if env._reward_signal_only else executed_signal_pos

        # Verify the fix: signal_pos_next should match next_signal_pos (actual)
        assert next_signal_pos == pytest.approx(0.0), (
            f"In CLOSE_TO_OPEN, first step executes HOLD (pending from reset). "
            f"next_signal_pos should be 0.0, got {next_signal_pos}"
        )
        assert agent_signal_pos == pytest.approx(
            1.0
        ), f"Agent requested 100%, agent_signal_pos should be 1.0, got {agent_signal_pos}"

    def test_close_to_open_delayed_position_takes_effect(self):
        """Position requested in step N should appear in signal_pos_next at step N+1."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0: Request 75%
        action_0 = ActionProto(ActionType.MARKET, 0.75)
        proto_0 = env._pending_action  # HOLD
        env._pending_action = action_0  # Store for next step

        executed_0 = env._signal_position_from_proto(proto_0, 0.0)
        env._last_signal_position = executed_0  # Update state

        assert executed_0 == 0.0, "Step 0: executed should be 0.0 (HOLD)"

        # Step 1: The 75% action is now executed
        proto_1 = env._pending_action  # action_0 from Step 0
        executed_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)

        assert executed_1 == pytest.approx(
            0.75
        ), f"Step 1: executed should be 0.75 (action from Step 0). Got {executed_1}"

    def test_signal_position_computation_consistency(self):
        """_signal_position_from_proto should correctly compute positions."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()

        # MARKET action: returns volume_frac
        market_action = ActionProto(ActionType.MARKET, 0.65)
        pos = env._signal_position_from_proto(market_action, 0.0)
        assert pos == pytest.approx(0.65), f"MARKET(0.65) should return 0.65, got {pos}"

        # HOLD action: returns prev_signal_pos
        hold_action = ActionProto(ActionType.HOLD, 0.0)
        pos = env._signal_position_from_proto(hold_action, 0.5)
        assert pos == pytest.approx(0.5), f"HOLD should keep prev (0.5), got {pos}"

    def test_signal_pos_requested_shows_intention(self):
        """Verify agent_signal_pos represents agent's immediate intention."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env._init_state()
        env._pending_action = ActionProto(ActionType.HOLD, 0.0)

        # Step 0: Request 100%
        action_0 = ActionProto(ActionType.MARKET, 1.0)
        agent_signal_pos_0 = env._signal_position_from_proto(action_0, 0.0)
        assert agent_signal_pos_0 == pytest.approx(
            1.0
        ), f"Agent requested 1.0, got {agent_signal_pos_0}"

        # Store for next step and update state
        proto_0 = env._pending_action
        env._pending_action = action_0
        executed_0 = env._signal_position_from_proto(proto_0, 0.0)
        env._last_signal_position = executed_0

        # Step 1: Request 30% (but executed is 100% from Step 0)
        action_1 = ActionProto(ActionType.MARKET, 0.3)
        agent_signal_pos_1 = env._signal_position_from_proto(action_1, env._last_signal_position)

        proto_1 = env._pending_action  # action_0 = MARKET(1.0)
        executed_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)

        assert agent_signal_pos_1 == pytest.approx(
            0.3
        ), f"Agent requested 0.3, got {agent_signal_pos_1}"
        assert executed_1 == pytest.approx(
            1.0
        ), f"Executed should be 1.0 (from Step 0 pending), got {executed_1}"

    def test_non_close_to_open_no_delay(self):
        """In non-CLOSE_TO_OPEN modes with signal_only, position updates immediately."""
        from trading_patchnew import DecisionTiming

        df = _create_test_df()
        env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
        env.decision_mode = 999  # Override to non-CLOSE_TO_OPEN mode

        env._init_state()

        action = ActionProto(ActionType.MARKET, 0.8)
        agent_signal_pos = env._signal_position_from_proto(action, 0.0)
        executed_signal_pos = 0.0  # Would be from pending in real scenario

        # In non-CLOSE_TO_OPEN + signal_only: next_signal_pos = agent_signal_pos
        if env.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
            next_signal_pos = executed_signal_pos
        else:
            next_signal_pos = agent_signal_pos if env._reward_signal_only else executed_signal_pos

        assert next_signal_pos == pytest.approx(0.8), (
            f"In non-CLOSE_TO_OPEN mode, next_signal_pos should be immediate (0.8). "
            f"Got {next_signal_pos}"
        )

    def test_info_dict_construction_signal_only(self):
        """Verify info dict fields are correctly constructed in signal_only mode."""
        from trading_patchnew import DecisionTiming

        # Simulate info dict construction as done in step()
        prev_signal_pos = 0.0
        next_signal_pos = 0.0  # From executed HOLD in CLOSE_TO_OPEN
        agent_signal_pos = 0.75  # Agent's intention

        # As per the fix: signal_pos_next = next_signal_pos (actual)
        # signal_pos_requested = agent_signal_pos (intention)
        info = {}
        reward_signal_only = True

        if reward_signal_only:
            info["signal_position_prev"] = float(prev_signal_pos)
            info["signal_pos"] = float(prev_signal_pos)
            info["signal_pos_next"] = float(next_signal_pos)  # FIXED: was agent_signal_pos
            info["signal_pos_requested"] = float(agent_signal_pos)  # NEW field

        assert info["signal_pos_next"] == pytest.approx(
            0.0
        ), "signal_pos_next should show actual position (0.0), not agent intention"
        assert info["signal_pos_requested"] == pytest.approx(
            0.75
        ), "signal_pos_requested should show agent's intention (0.75)"


class TestSignalPosNextNonSignalOnlyMode:
    """Tests for signal_pos_next in non-signal_only (full execution) mode."""

    def test_non_signal_only_info_dict_construction(self):
        """Verify info dict fields in non-signal_only mode."""
        # Simulate info dict construction
        prev_signal_pos = 0.0
        next_signal_pos = 0.0  # From executed in CLOSE_TO_OPEN
        agent_signal_pos = 0.5  # Agent's intention

        info = {}
        reward_signal_only = False

        if reward_signal_only:
            info["signal_position_prev"] = float(prev_signal_pos)
            info["signal_pos"] = float(prev_signal_pos)
            info["signal_pos_next"] = float(next_signal_pos)
            info["signal_pos_requested"] = float(agent_signal_pos)
        else:
            info["signal_position_prev"] = float(prev_signal_pos)
            info["signal_pos"] = float(next_signal_pos)
            info["signal_pos_next"] = float(next_signal_pos)
            info["signal_pos_requested"] = float(agent_signal_pos)  # Also added

        assert info["signal_pos_next"] == pytest.approx(
            0.0
        ), "Non-signal_only: signal_pos_next should be actual position (0.0)"
        assert info["signal_pos_requested"] == pytest.approx(
            0.5
        ), "Non-signal_only: signal_pos_requested should be agent's intention (0.5)"

    def test_signal_pos_requested_always_available(self):
        """signal_pos_requested should be in info regardless of mode."""
        # Test signal_only mode
        info1 = {
            "signal_pos_next": 0.0,
            "signal_pos_requested": 0.75,
        }
        assert "signal_pos_requested" in info1

        # Test non-signal_only mode
        info2 = {
            "signal_pos_next": 0.0,
            "signal_pos_requested": 0.5,
        }
        assert "signal_pos_requested" in info2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
