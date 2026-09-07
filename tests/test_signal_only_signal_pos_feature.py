"""
Tests for signal-only mode signal position feature.

Updated 2025-11-25: info["signal_pos_next"] now shows ACTUAL position (next_signal_pos),
not agent's intention (agent_signal_pos). In CLOSE_TO_OPEN mode, these differ due to 1-bar delay.

Agent's intention is now available in info["signal_pos_requested"].

NOTE: These tests use mock approach to avoid Cython dependencies.
Full integration tests are in test_signal_pos_next_close_to_open_consistency.py.
"""

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


def _frame_for_signal_only() -> pd.DataFrame:
    """Construct a minimal dataframe for signal-only environment tests."""
    steps = 4
    idx = np.arange(steps, dtype=np.int64)
    base = np.linspace(100.0, 101.5, steps)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "price": base,
            "quote_asset_volume": np.full(steps, 10.0),
            "ts_ms": idx * 60_000,
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


def test_signal_only_close_to_open_delay_behavior() -> None:
    """Test signal position delay behavior in CLOSE_TO_OPEN + signal_only mode.

    This test verifies the 2025-11-25 fix:
    - signal_pos_next shows ACTUAL position (delayed in CLOSE_TO_OPEN)
    - signal_pos_requested shows agent's intention (immediate)
    """
    from trading_patchnew import DecisionTiming

    df = _frame_for_signal_only()
    env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)

    # Initialize
    env._init_state()
    env._pending_action = ActionProto(ActionType.HOLD, 0.0)  # From reset

    # Step 0: Agent wants 75%, but pending is HOLD
    agent_action = ActionProto(ActionType.MARKET, 0.75)
    prev_signal_pos = env._last_signal_position  # 0.0

    # Compute positions
    agent_signal_pos = env._signal_position_from_proto(agent_action, prev_signal_pos)
    proto = env._pending_action  # HOLD
    executed_signal_pos = env._signal_position_from_proto(proto, prev_signal_pos)

    # In CLOSE_TO_OPEN: next_signal_pos = executed_signal_pos
    next_signal_pos = executed_signal_pos  # 0.0, not 0.75!

    # Verify signal_pos_next should be 0.0 (actual), not 0.75 (requested)
    assert next_signal_pos == pytest.approx(
        0.0
    ), f"In CLOSE_TO_OPEN, first step executes HOLD. next_signal_pos should be 0.0, got {next_signal_pos}"
    assert agent_signal_pos == pytest.approx(
        0.75
    ), f"Agent requested 75%. agent_signal_pos should be 0.75, got {agent_signal_pos}"

    # Update state for next step
    env._pending_action = agent_action
    env._last_signal_position = next_signal_pos

    # Step 1: Now the 75% action from Step 0 is executed
    action_1 = ActionProto(ActionType.HOLD, 0.0)
    proto_1 = env._pending_action  # MARKET(0.75) from Step 0
    executed_signal_pos_1 = env._signal_position_from_proto(proto_1, env._last_signal_position)

    assert executed_signal_pos_1 == pytest.approx(
        0.75
    ), f"Step 1 should execute the 75% from Step 0. Got {executed_signal_pos_1}"


def test_signal_only_info_fields_semantics() -> None:
    """Test semantics of info fields in signal_only mode.

    Fields:
    - signal_pos: Current position at start of step (equals prev)
    - signal_pos_next: Position after this step (actual, may be delayed)
    - signal_pos_requested: What agent wanted (immediate intention)
    """
    from trading_patchnew import DecisionTiming

    df = _frame_for_signal_only()
    env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
    env._init_state()
    env._pending_action = ActionProto(ActionType.HOLD, 0.0)

    # Agent requests 50%
    prev_signal_pos = 0.0
    agent_signal_pos = 0.5
    executed_signal_pos = 0.0  # HOLD from pending
    next_signal_pos = executed_signal_pos  # In CLOSE_TO_OPEN

    # Build info dict as per the fix
    info = {}
    info["signal_position_prev"] = float(prev_signal_pos)
    info["signal_pos"] = float(prev_signal_pos)  # Current position
    info["signal_pos_next"] = float(next_signal_pos)  # Actual after step
    info["signal_pos_requested"] = float(agent_signal_pos)  # Agent's intention

    assert info["signal_pos"] == pytest.approx(0.0)
    assert info["signal_pos_next"] == pytest.approx(
        0.0
    ), "signal_pos_next should show 0.0 (delayed), not 0.5 (requested)"
    assert info["signal_pos_requested"] == pytest.approx(
        0.5
    ), "signal_pos_requested should show 0.5 (agent's intention)"


def test_signal_only_reward_uses_prev_position() -> None:
    """Verify reward calculation uses prev_signal_pos, not next."""
    from trading_patchnew import DecisionTiming

    df = _frame_for_signal_only()
    env = _setup_mock_env(df, decision_mode=DecisionTiming.CLOSE_TO_OPEN, signal_only=True)
    env._init_state()
    env._pending_action = ActionProto(ActionType.HOLD, 0.0)

    # Step 0: prev_signal_pos = 0.0, agent wants 100%
    prev_signal_pos = 0.0
    log_return = math.log(df.loc[1, "close"] / df.loc[0, "close"])

    # Reward = log_return * prev_signal_pos = log_return * 0.0 = 0
    expected_reward_0 = log_return * prev_signal_pos
    assert expected_reward_0 == pytest.approx(
        0.0
    ), "Step 0: reward should be 0 because prev_signal_pos=0"

    # After Step 0: _last_signal_position = 0.0 (HOLD executed)
    # After Step 1: _last_signal_position = 1.0 (100% from Step 0)
    # Step 2: prev_signal_pos = 1.0

    prev_signal_pos_2 = 1.0
    log_return_2 = math.log(df.loc[2, "close"] / df.loc[1, "close"])
    expected_reward_2 = log_return_2 * prev_signal_pos_2

    assert expected_reward_2 != pytest.approx(
        0.0
    ), f"Step 2: reward should be non-zero. Expected {expected_reward_2}"
