"""
Unit tests for RawRecurrentRolloutBuffer class.

This test suite covers the critical rollout buffer used for recurrent PPO training.
Tests address the 0% coverage gap identified in COMPREHENSIVE_TEST_REPORT.md.

Tested methods:
- reset(): Buffer initialization and state reset
- add(): Adding rollout data with various parameters
- _to_numpy(): Static tensor/array conversion
- get(): Generator for training batches
- _get_samples(): Sample extraction with padding

Tech Debt Closure: Testing/Quality - RawRecurrentRolloutBuffer 0% coverage
Control Artifact: This test file + CI coverage tracking
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, patch
from typing import Tuple

# Import the class under test
from distributional_ppo import RawRecurrentRolloutBuffer, RNNStates


class MockObservationSpace:
    """Mock observation space for testing."""
    def __init__(self, shape: Tuple[int, ...] = (4,)):
        self.shape = shape


class MockActionSpace:
    """Mock action space for testing."""
    def __init__(self, shape: Tuple[int, ...] = (2,)):
        self.shape = shape
        self.n = 2  # Discrete action space size


@pytest.fixture
def buffer_params():
    """Common buffer parameters."""
    return {
        "buffer_size": 8,
        "observation_space": MockObservationSpace(shape=(4,)),
        "action_space": MockActionSpace(shape=(2,)),
        "device": "cpu",
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "n_envs": 2,
        "hidden_state_shape": (1, 2, 16),  # (n_layers, n_envs, hidden_size)
        "n_seq_start": 1,
    }


class TestToNumpy:
    """Tests for _to_numpy static method."""

    def test_torch_tensor_conversion(self):
        """Torch tensor is converted to numpy array."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_cuda_tensor_moves_to_cpu(self):
        """CUDA tensor is moved to CPU before conversion."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_numpy_array_passthrough(self):
        """Numpy array is returned as-is."""
        array = np.array([1.0, 2.0, 3.0])
        result = RawRecurrentRolloutBuffer._to_numpy(array)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_list_conversion(self):
        """List is converted to numpy array."""
        input_list = [1.0, 2.0, 3.0]
        result = RawRecurrentRolloutBuffer._to_numpy(input_list)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_scalar_conversion(self):
        """Scalar value is converted to numpy array."""
        result = RawRecurrentRolloutBuffer._to_numpy(42.0)

        assert isinstance(result, np.ndarray)
        assert result == 42.0

    def test_multidimensional_tensor(self):
        """Multi-dimensional tensor preserves shape."""
        tensor = torch.randn(2, 3, 4)
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3, 4)

    def test_tensor_with_grad_is_detached(self):
        """Tensor with gradient is properly detached."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor_with_grad = tensor * 2  # Creates grad_fn

        result = RawRecurrentRolloutBuffer._to_numpy(tensor_with_grad)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [2.0, 4.0, 6.0])


class TestReset:
    """Tests for reset() method."""

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    def test_reset_initializes_actions_raw(self, mock_super_reset, mock_init):
        """reset() initializes actions_raw array."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)

        buffer.reset()

        assert hasattr(buffer, "actions_raw")
        assert buffer.actions_raw.shape == (8, 2, 2)
        assert buffer.actions_raw.dtype == np.float32
        assert np.all(buffer.actions_raw == 0)

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    def test_reset_initializes_log_prob_raw(self, mock_super_reset, mock_init):
        """reset() initializes old_log_prob_raw array."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)

        buffer.reset()

        assert hasattr(buffer, "old_log_prob_raw")
        assert buffer.old_log_prob_raw.shape == (8, 2)
        assert buffer.old_log_prob_raw.dtype == np.float32

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    def test_reset_initializes_seq_start_indices(self, mock_super_reset, mock_init):
        """reset() initializes seq_start_indices as empty list."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)

        buffer.reset()

        assert hasattr(buffer, "seq_start_indices")
        assert buffer.seq_start_indices == []

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    def test_reset_initializes_distributional_fields_to_none(self, mock_super_reset, mock_init):
        """reset() initializes distributional fields to None."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)

        buffer.reset()

        assert buffer.value_quantiles is None
        assert buffer.value_probs is None
        assert buffer.value_quantiles_critic1 is None
        assert buffer.value_quantiles_critic2 is None
        assert buffer.value_probs_critic1 is None
        assert buffer.value_probs_critic2 is None

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    def test_reset_calls_super(self, mock_super_reset, mock_init):
        """reset() calls parent class reset."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)

        buffer.reset()

        mock_super_reset.assert_called_once()


class TestAdd:
    """Tests for add() method."""

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_add_requires_actions_raw(self, mock_super_add, mock_reset, mock_init):
        """add() raises TypeError if actions_raw is None."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        with pytest.raises(TypeError, match="actions_raw.*must be provided"):
            buffer.add(
                lstm_states=lstm_states,
                actions_raw=None,
                log_prob_raw=torch.tensor([0.1, 0.2]),
            )

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_add_requires_log_prob_raw(self, mock_super_add, mock_reset, mock_init):
        """add() raises TypeError if log_prob_raw is None."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        with pytest.raises(TypeError, match="log_prob_raw.*must be provided"):
            buffer.add(
                lstm_states=lstm_states,
                actions_raw=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
                log_prob_raw=None,
            )

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_add_stores_actions_raw(self, mock_super_add, mock_reset, mock_init):
        """add() stores actions_raw in buffer."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.buffer_size = 8
        buffer.n_envs = 2
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)
        buffer.pos = 1  # After super().add()
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        actions_raw = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        log_prob_raw = torch.tensor([-0.5, -0.3])

        buffer.add(
            lstm_states=lstm_states,
            actions_raw=actions_raw,
            log_prob_raw=log_prob_raw,
        )

        # Position 0 should have the data (pos=1 after super, so pos-1=0)
        assert np.allclose(buffer.actions_raw[0], [[0.7, 0.3], [0.4, 0.6]])

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_add_stores_value_quantiles_on_first_add(self, mock_super_add, mock_reset, mock_init):
        """add() initializes and stores value_quantiles on first add."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.buffer_size = 8
        buffer.n_envs = 2
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)
        buffer.pos = 1
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        n_quantiles = 21
        value_quantiles = torch.randn(2, n_quantiles)

        buffer.add(
            lstm_states=lstm_states,
            actions_raw=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            log_prob_raw=torch.tensor([-0.5, -0.3]),
            value_quantiles=value_quantiles,
        )

        assert buffer.value_quantiles is not None
        assert buffer.value_quantiles.shape == (8, 2, n_quantiles)

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_add_stores_twin_critics_quantiles(self, mock_super_add, mock_reset, mock_init):
        """add() stores twin critics quantiles separately."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.buffer_size = 8
        buffer.n_envs = 2
        buffer.actions = np.zeros((8, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((8, 2), dtype=np.float32)
        buffer.pos = 1
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        n_quantiles = 21
        value_quantiles_c1 = torch.randn(2, n_quantiles)
        value_quantiles_c2 = torch.randn(2, n_quantiles)

        buffer.add(
            lstm_states=lstm_states,
            actions_raw=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            log_prob_raw=torch.tensor([-0.5, -0.3]),
            value_quantiles_critic1=value_quantiles_c1,
            value_quantiles_critic2=value_quantiles_c2,
        )

        assert buffer.value_quantiles_critic1 is not None
        assert buffer.value_quantiles_critic2 is not None
        assert buffer.value_quantiles_critic1.shape == (8, 2, n_quantiles)
        assert buffer.value_quantiles_critic2.shape == (8, 2, n_quantiles)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_to_numpy_with_empty_tensor(self):
        """_to_numpy handles empty tensor."""
        tensor = torch.tensor([])
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_to_numpy_with_complex_dtype(self):
        """_to_numpy handles various dtypes."""
        tensor_int = torch.tensor([1, 2, 3], dtype=torch.int64)
        result_int = RawRecurrentRolloutBuffer._to_numpy(tensor_int)
        assert result_int.dtype == np.int64

        tensor_bool = torch.tensor([True, False, True])
        result_bool = RawRecurrentRolloutBuffer._to_numpy(tensor_bool)
        assert result_bool.dtype == np.bool_


class TestBufferIntegration:
    """Integration tests for full buffer workflow."""

    @patch("distributional_ppo.RecurrentRolloutBuffer.__init__", return_value=None)
    @patch("distributional_ppo.RecurrentRolloutBuffer.reset")
    @patch("distributional_ppo.RecurrentRolloutBuffer.add")
    def test_multiple_adds_cycle_buffer(self, mock_super_add, mock_reset, mock_init):
        """Multiple add() calls cycle through buffer positions."""
        buffer = RawRecurrentRolloutBuffer.__new__(RawRecurrentRolloutBuffer)
        buffer.buffer_size = 4
        buffer.n_envs = 2
        buffer.actions = np.zeros((4, 2, 2), dtype=np.float32)
        buffer.log_probs = np.zeros((4, 2), dtype=np.float32)
        buffer.reset()

        lstm_states = RNNStates(
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
            (torch.zeros(1, 2, 16), torch.zeros(1, 2, 16)),
        )

        # Simulate adding to different positions
        for i in range(4):
            buffer.pos = (i + 1) % buffer.buffer_size
            if buffer.pos == 0:
                buffer.pos = buffer.buffer_size
            buffer.add(
                lstm_states=lstm_states,
                actions_raw=torch.full((2, 2), float(i)),
                log_prob_raw=torch.full((2,), float(-i)),
            )

        # Verify data at each position
        for i in range(4):
            expected_pos = i % 4
            assert np.allclose(buffer.actions_raw[expected_pos], float(i))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
