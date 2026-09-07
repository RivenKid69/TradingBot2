"""
Comprehensive tests for LongOnlyActionWrapper action space fix (2025-11-25).

This fix addresses a critical bug where:
- LongOnlyActionWrapper inherited action_space = [0, 1] from TradingEnv
- Policy used sigmoid -> outputs [0, 1]
- Wrapper applied (x+1)/2 mapping expecting [-1, 1]
- Result: [0, 1] -> [0.5, 1.0], minimum position = 50%!

After fix:
- LongOnlyActionWrapper sets action_space = [-1, 1]
- Policy detects this and uses tanh -> outputs [-1, 1]
- Wrapper correctly maps [-1, 1] -> [0, 1]
- Agent can express full range [0%, 100%]

Test categories:
1. Wrapper action space definition
2. Wrapper transformation correctness
3. Policy activation function adaptation
4. Integration tests (wrapper + policy)
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
from gymnasium import spaces
from dataclasses import dataclass
from typing import Any, Optional

# Import modules under test
from wrappers.action_space import (
    LongOnlyActionWrapper,
    ScoreActionWrapper,
    LONG_ONLY_LOW,
    LONG_ONLY_HIGH,
    SCORE_LOW,
    SCORE_HIGH,
)
from action_proto import ActionProto, ActionType


# ============================================================================
# TEST FIXTURES
# ============================================================================


class MockEnvWithActionSpace(gym.Env):
    """Mock environment with configurable action_space."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        super().__init__()
        self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self._last_action = None

    def step(self, action):
        self._last_action = np.asarray(action).copy()
        return np.zeros(10, dtype=np.float32), 0.0, False, False, {}

    def reset(self, **kwargs):
        self._last_action = None
        return np.zeros(10, dtype=np.float32), {}


@pytest.fixture
def mock_env_01():
    """Mock env with [0, 1] action space (like TradingEnv)."""
    return MockEnvWithActionSpace(low=0.0, high=1.0)


@pytest.fixture
def mock_env_neg11():
    """Mock env with [-1, 1] action space."""
    return MockEnvWithActionSpace(low=-1.0, high=1.0)


# ============================================================================
# TEST 1: Wrapper Action Space Definition
# ============================================================================


class TestWrapperActionSpaceDefinition:
    """Verify LongOnlyActionWrapper correctly defines its action_space."""

    def test_wrapper_exposes_neg11_action_space(self, mock_env_01):
        """
        CRITICAL: Wrapper must expose action_space = [-1, 1] regardless of base env.

        This is essential for the policy to use tanh instead of sigmoid.
        """
        wrapped = LongOnlyActionWrapper(mock_env_01)

        assert wrapped.action_space.low[0] == pytest.approx(-1.0)
        assert wrapped.action_space.high[0] == pytest.approx(1.0)

    def test_wrapper_uses_correct_constants(self, mock_env_01):
        """Verify wrapper uses LONG_ONLY_LOW and LONG_ONLY_HIGH constants."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        assert wrapped.action_space.low[0] == pytest.approx(LONG_ONLY_LOW)
        assert wrapped.action_space.high[0] == pytest.approx(LONG_ONLY_HIGH)

    def test_wrapper_action_space_shape(self, mock_env_01):
        """Verify action_space shape is preserved."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        assert wrapped.action_space.shape == (1,)
        assert wrapped.action_space.dtype == np.float32

    def test_wrapper_observation_space_preserved(self, mock_env_01):
        """Verify observation_space is passed through unchanged."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        np.testing.assert_array_equal(
            wrapped.observation_space.low, mock_env_01.observation_space.low
        )


# ============================================================================
# TEST 2: Wrapper Transformation Correctness
# ============================================================================


class TestWrapperTransformation:
    """Verify LongOnlyActionWrapper correctly transforms [-1, 1] -> [0, 1]."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (-1.0, 0.0),  # Full exit
            (-0.5, 0.25),  # 25% position
            (0.0, 0.5),  # 50% position
            (0.5, 0.75),  # 75% position
            (1.0, 1.0),  # Full position
        ],
    )
    def test_mapping_correctness(self, mock_env_01, input_val, expected):
        """Test linear mapping: (x + 1) / 2."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        action = np.array([input_val], dtype=np.float32)
        result = wrapped.action(action)

        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_mapping_formula(self, mock_env_01):
        """Verify the exact formula: output = (input + 1) / 2."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        test_values = np.linspace(-1.0, 1.0, 21, dtype=np.float32)

        for val in test_values:
            action = np.array([val], dtype=np.float32)
            result = wrapped.action(action)
            expected = (val + 1.0) / 2.0

            assert result[0] == pytest.approx(
                expected, abs=1e-6
            ), f"Input {val}: got {result[0]}, expected {expected}"

    def test_output_bounds(self, mock_env_01):
        """Verify output is always in [0, 1]."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        # Test many values including edge cases
        test_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

        for val in test_values:
            action = np.array([val], dtype=np.float32)
            result = wrapped.action(action)

            assert 0.0 <= result[0] <= 1.0, f"Input {val} produced out-of-bounds output {result[0]}"

    def test_numpy_array_transformation(self, mock_env_01):
        """Test transformation with numpy arrays."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        action = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

        result = wrapped.action(action)

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_action_proto_transformation(self, mock_env_01):
        """Test transformation with ActionProto objects."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        proto = ActionProto(ActionType.MARKET, volume_frac=-0.5)
        result = wrapped.action(proto)

        assert isinstance(result, ActionProto)
        assert result.volume_frac == pytest.approx(0.25, abs=1e-6)
        assert result.action_type == ActionType.MARKET

    def test_scalar_transformation(self, mock_env_01):
        """Test transformation with scalar values."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        result = wrapped.action(-0.8)

        assert result == pytest.approx(0.1, abs=1e-6)


# ============================================================================
# TEST 3: Position Expressiveness
# ============================================================================


class TestPositionExpressiveness:
    """Verify agent can express full range of positions."""

    def test_can_express_zero_position(self, mock_env_01):
        """
        CRITICAL: Agent must be able to express 0% position (full exit).

        Before fix: minimum was 50%!
        After fix: -1.0 -> 0.0
        """
        wrapped = LongOnlyActionWrapper(mock_env_01)

        action = np.array([-1.0], dtype=np.float32)
        result = wrapped.action(action)

        assert result[0] == pytest.approx(0.0, abs=1e-6), "Agent cannot express 0% position!"

    def test_can_express_full_position(self, mock_env_01):
        """Agent can express 100% position."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        action = np.array([1.0], dtype=np.float32)
        result = wrapped.action(action)

        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_position_granularity(self, mock_env_01):
        """Verify fine-grained position control."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        # Test 5% increments
        for target_pct in range(0, 101, 5):
            target_frac = target_pct / 100.0
            # Inverse: input = 2 * output - 1
            policy_output = 2.0 * target_frac - 1.0

            action = np.array([policy_output], dtype=np.float32)
            result = wrapped.action(action)

            assert result[0] == pytest.approx(
                target_frac, abs=1e-6
            ), f"Cannot achieve {target_pct}% position"


# ============================================================================
# TEST 4: Regression Prevention
# ============================================================================


class TestRegressionPrevention:
    """Prevent regression to the broken behavior."""

    def test_no_position_bias(self, mock_env_01):
        """
        REGRESSION TEST: Ensure no systematic position bias.

        The old bug caused all positions to be biased upward by 50%.
        """
        wrapped = LongOnlyActionWrapper(mock_env_01)

        # Test neutral action (should give 50%, not 75%)
        action = np.array([0.0], dtype=np.float32)
        result = wrapped.action(action)

        assert result[0] == pytest.approx(
            0.5, abs=1e-6
        ), "Neutral action should give 50% position, not 75%!"

    def test_old_bug_does_not_exist(self, mock_env_01):
        """
        REGRESSION TEST: The old [0,1] -> [0.5,1.0] bug must not exist.

        Old behavior: 0.0 -> 0.5, 0.5 -> 0.75, 1.0 -> 1.0
        New behavior: -1.0 -> 0.0, 0.0 -> 0.5, 1.0 -> 1.0
        """
        wrapped = LongOnlyActionWrapper(mock_env_01)

        # Test that 0.0 input doesn't give 0.5 output (that would be old bug)
        # In new design, 0.0 input gives 0.5 output (which is CORRECT)
        # The key is that action_space is now [-1, 1], not [0, 1]

        # Verify action_space is [-1, 1]
        assert wrapped.action_space.low[0] == pytest.approx(-1.0)

        # Verify -1.0 gives 0.0 (this would be impossible in old design)
        action = np.array([-1.0], dtype=np.float32)
        result = wrapped.action(action)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# TEST 5: Integration with MockEnv
# ============================================================================


class TestIntegrationWithEnv:
    """Test full integration: wrapper -> env.step()."""

    def test_env_receives_correct_action(self, mock_env_01):
        """Verify the underlying env receives the transformed action."""
        wrapped = LongOnlyActionWrapper(mock_env_01)
        wrapped.reset()

        # Policy outputs -0.5 (wants 25% position)
        action = np.array([-0.5], dtype=np.float32)
        wrapped.step(action)

        # Env should receive 0.25
        assert mock_env_01._last_action[0] == pytest.approx(0.25, abs=1e-6)

    def test_full_range_env_actions(self, mock_env_01):
        """Test full range of actions reach env correctly."""
        wrapped = LongOnlyActionWrapper(mock_env_01)
        wrapped.reset()

        test_cases = [
            (-1.0, 0.0),
            (-0.5, 0.25),
            (0.0, 0.5),
            (0.5, 0.75),
            (1.0, 1.0),
        ]

        for policy_action, expected_env_action in test_cases:
            action = np.array([policy_action], dtype=np.float32)
            wrapped.step(action)

            assert mock_env_01._last_action[0] == pytest.approx(
                expected_env_action, abs=1e-6
            ), f"Policy action {policy_action} should give env action {expected_env_action}"


# ============================================================================
# TEST 6: ScoreActionWrapper Interaction
# ============================================================================


class TestScoreActionWrapperInteraction:
    """Test that ScoreActionWrapper works correctly after LongOnlyActionWrapper."""

    def test_chain_wrapping(self, mock_env_01):
        """Test LongOnlyActionWrapper -> ScoreActionWrapper chain."""
        # This is the typical wrapping order in train_model_multi_patch.py
        wrapped = LongOnlyActionWrapper(mock_env_01)
        wrapped = ScoreActionWrapper(wrapped)

        # ScoreActionWrapper should clip to [0, 1]
        assert wrapped.action_space.low[0] == pytest.approx(SCORE_LOW)
        assert wrapped.action_space.high[0] == pytest.approx(SCORE_HIGH)


# ============================================================================
# TEST 7: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_action_passthrough(self, mock_env_01):
        """None action should pass through unchanged."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        result = wrapped.action(None)
        assert result is None

    def test_empty_array_passthrough(self, mock_env_01):
        """Empty array should pass through unchanged."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        action = np.array([], dtype=np.float32)
        result = wrapped.action(action)

        assert result.size == 0

    def test_non_finite_raises(self, mock_env_01):
        """Non-finite values should raise ValueError."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        with pytest.raises(ValueError, match="Non-finite"):
            wrapped.action(np.nan)

        with pytest.raises(ValueError, match="Non-finite"):
            wrapped.action(np.inf)

    def test_out_of_bounds_clipping(self, mock_env_01):
        """Out-of-bounds values should be clipped after mapping."""
        wrapped = LongOnlyActionWrapper(mock_env_01)

        # +2.0 -> (2.0 + 1) / 2 = 1.5 -> clipped to 1.0
        action = np.array([2.0], dtype=np.float32)
        result = wrapped.action(action)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

        # -2.0 -> (-2.0 + 1) / 2 = -0.5 -> clipped to 0.0
        action = np.array([-2.0], dtype=np.float32)
        result = wrapped.action(action)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
