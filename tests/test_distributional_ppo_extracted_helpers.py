"""
Unit tests for extracted helper methods in DistributionalPPO.

Tests cover the following methods refactored from the train() method:
- _concat_tensor_batches: Static method for concatenating tensor batches
- _concat_string_keys: Static method for flattening string key batches
- _prepare_minibatch_iterator: Instance method for grouped micro-batch iteration

These tests verify that the extraction maintained correct functionality.
"""

import pytest
torch = pytest.importorskip("torch")
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Optional

# Import the class under test
from distributional_ppo import DistributionalPPO


class TestConcatTensorBatches:
    """Tests for _concat_tensor_batches static method."""

    def test_single_tensor(self):
        """Single tensor is reshaped to (-1, 1) and returned."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = DistributionalPPO._concat_tensor_batches([tensor])

        assert result is not None
        assert result.shape == (3, 1)
        assert torch.allclose(result.flatten(), tensor)

    def test_multiple_tensors(self):
        """Multiple tensors are concatenated along dim 0."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0, 5.0])
        result = DistributionalPPO._concat_tensor_batches([t1, t2])

        assert result is not None
        assert result.shape == (5, 1)
        expected = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        assert torch.allclose(result, expected)

    def test_2d_tensors(self):
        """2D tensors are flattened then reshaped to (-1, 1)."""
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        result = DistributionalPPO._concat_tensor_batches([t1])

        assert result is not None
        assert result.shape == (4, 1)  # Flattened to 4 elements

    def test_with_none_values(self):
        """None values in sequence are filtered out."""
        t1 = torch.tensor([1.0, 2.0])
        result = DistributionalPPO._concat_tensor_batches([t1, None, t1])

        assert result is not None
        assert result.shape == (4, 1)  # Only two valid tensors

    def test_with_empty_tensors(self):
        """Empty tensors (numel=0) are filtered out."""
        t1 = torch.tensor([1.0, 2.0])
        empty = torch.tensor([])
        result = DistributionalPPO._concat_tensor_batches([t1, empty, t1])

        assert result is not None
        assert result.shape == (4, 1)

    def test_all_none_returns_none(self):
        """Sequence of all None values returns None."""
        result = DistributionalPPO._concat_tensor_batches([None, None])
        assert result is None

    def test_all_empty_returns_none(self):
        """Sequence of all empty tensors returns None."""
        result = DistributionalPPO._concat_tensor_batches([torch.tensor([]), torch.tensor([])])
        assert result is None

    def test_empty_sequence_returns_none(self):
        """Empty sequence returns None."""
        result = DistributionalPPO._concat_tensor_batches([])
        assert result is None

    def test_preserves_dtype(self):
        """Output preserves the dtype of input tensors."""
        t = torch.tensor([1, 2, 3], dtype=torch.float64)
        result = DistributionalPPO._concat_tensor_batches([t])

        assert result is not None
        assert result.dtype == torch.float64

    def test_mixed_none_and_empty(self):
        """Mixed None and empty tensors are all filtered."""
        t1 = torch.tensor([1.0])
        result = DistributionalPPO._concat_tensor_batches([None, torch.tensor([]), t1, None])

        assert result is not None
        assert result.shape == (1, 1)


class TestConcatStringKeys:
    """Tests for _concat_string_keys static method."""

    def test_single_batch(self):
        """Single batch of keys is returned as list."""
        keys = ["BTCUSDT", "ETHUSDT"]
        result = DistributionalPPO._concat_string_keys([keys])

        assert result is not None
        assert result == ["BTCUSDT", "ETHUSDT"]

    def test_multiple_batches(self):
        """Multiple batches are flattened into single list."""
        batch1 = ["BTCUSDT", "ETHUSDT"]
        batch2 = ["BNBUSDT"]
        result = DistributionalPPO._concat_string_keys([batch1, batch2])

        assert result is not None
        assert result == ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    def test_with_empty_batches(self):
        """Empty batches are filtered out."""
        batch1 = ["BTCUSDT"]
        result = DistributionalPPO._concat_string_keys([batch1, [], batch1])

        assert result is not None
        assert result == ["BTCUSDT", "BTCUSDT"]

    def test_all_empty_returns_none(self):
        """All empty batches returns None."""
        result = DistributionalPPO._concat_string_keys([[], []])
        assert result is None

    def test_empty_sequence_returns_none(self):
        """Empty sequence returns None."""
        result = DistributionalPPO._concat_string_keys([])
        assert result is None

    def test_converts_non_string_items(self):
        """Non-string items are converted to strings."""
        items = [1, 2, 3]
        result = DistributionalPPO._concat_string_keys([items])

        assert result is not None
        assert result == ["1", "2", "3"]

    def test_mixed_types(self):
        """Mixed types are all converted to strings."""
        batch1 = ["BTCUSDT", 123]
        batch2 = [45.6, True]
        result = DistributionalPPO._concat_string_keys([batch1, batch2])

        assert result is not None
        assert result == ["BTCUSDT", "123", "45.6", "True"]

    def test_preserves_order(self):
        """Order of items is preserved across batches."""
        batch1 = ["A", "B"]
        batch2 = ["C"]
        batch3 = ["D", "E", "F"]
        result = DistributionalPPO._concat_string_keys([batch1, batch2, batch3])

        assert result is not None
        assert result == ["A", "B", "C", "D", "E", "F"]


class TestPrepareMinibatchIterator:
    """Tests for _prepare_minibatch_iterator instance method."""

    @pytest.fixture
    def mock_ppo(self):
        """Create a mock PPO instance with mocked rollout buffer."""
        # Create a minimal mock - don't use spec to allow rollout_buffer attribute
        mock = MagicMock()
        mock.rollout_buffer = MagicMock()
        # Bind the method to our mock instance
        mock._prepare_minibatch_iterator = DistributionalPPO._prepare_minibatch_iterator.__get__(mock, DistributionalPPO)
        return mock

    def test_returns_none_for_empty_buffer(self, mock_ppo):
        """Empty rollout buffer returns (None, effective_batch_size)."""
        mock_ppo.rollout_buffer.get.return_value = iter([])  # Empty iterator

        iterator, batch_size = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=32,
            effective_batch_size=128,
            grad_accum_steps=4,
        )

        assert iterator is None
        assert batch_size == 128

    def test_groups_microbatches_correctly(self, mock_ppo):
        """Microbatches are grouped by grad_accum_steps."""
        # 8 microbatches, grad_accum_steps=2 -> 4 groups of 2
        microbatches = [f"batch_{i}" for i in range(8)]
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        iterator, batch_size = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=32,
            effective_batch_size=128,
            grad_accum_steps=2,
        )

        assert iterator is not None
        groups = list(iterator)
        assert len(groups) == 4
        assert groups[0] == ("batch_0", "batch_1")
        assert groups[1] == ("batch_2", "batch_3")
        assert groups[2] == ("batch_4", "batch_5")
        assert groups[3] == ("batch_6", "batch_7")

    def test_returns_correct_expected_batch_size(self, mock_ppo):
        """Expected batch size is microbatch_size * grad_accum_steps."""
        microbatches = ["batch"] * 4
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        iterator, batch_size = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=32,
            effective_batch_size=999,  # Not used when buffer not empty
            grad_accum_steps=4,
        )

        assert batch_size == 128  # 32 * 4

    def test_raises_on_incomplete_microbatch_bucket(self, mock_ppo):
        """Raises RuntimeError if microbatch count not divisible by grad_accum_steps."""
        # 7 microbatches, grad_accum_steps=4 -> error (7 % 4 != 0)
        microbatches = ["batch"] * 7
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        with pytest.raises(RuntimeError, match="incomplete micro-batch bucket"):
            mock_ppo._prepare_minibatch_iterator(
                microbatch_size=32,
                effective_batch_size=128,
                grad_accum_steps=4,
            )

    def test_single_group(self, mock_ppo):
        """Single group when microbatch count equals grad_accum_steps."""
        microbatches = ["a", "b", "c", "d"]
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        iterator, batch_size = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=16,
            effective_batch_size=64,
            grad_accum_steps=4,
        )

        groups = list(iterator)
        assert len(groups) == 1
        assert groups[0] == ("a", "b", "c", "d")
        assert batch_size == 64

    def test_grad_accum_steps_one(self, mock_ppo):
        """Each microbatch is its own group when grad_accum_steps=1."""
        microbatches = ["x", "y", "z"]
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        iterator, batch_size = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=10,
            effective_batch_size=30,
            grad_accum_steps=1,
        )

        groups = list(iterator)
        assert len(groups) == 3
        assert groups[0] == ("x",)
        assert groups[1] == ("y",)
        assert groups[2] == ("z",)
        assert batch_size == 10

    def test_iterator_is_generator(self, mock_ppo):
        """Returned iterator is a generator (lazy evaluation)."""
        microbatches = ["batch"] * 4
        mock_ppo.rollout_buffer.get.return_value = iter(microbatches)

        iterator, _ = mock_ppo._prepare_minibatch_iterator(
            microbatch_size=32,
            effective_batch_size=128,
            grad_accum_steps=2,
        )

        # Check it's iterable but not a list (lazy)
        assert hasattr(iterator, "__iter__")
        assert hasattr(iterator, "__next__")

    def test_calls_rollout_buffer_with_microbatch_size(self, mock_ppo):
        """rollout_buffer.get is called with microbatch_size argument."""
        mock_ppo.rollout_buffer.get.return_value = iter([])

        mock_ppo._prepare_minibatch_iterator(
            microbatch_size=64,
            effective_batch_size=256,
            grad_accum_steps=4,
        )

        mock_ppo.rollout_buffer.get.assert_called_once_with(64)


class TestIntegration:
    """Integration tests verifying methods work together correctly."""

    def test_concat_batches_typical_training_scenario(self):
        """Test typical training scenario with value predictions."""
        # Simulate collecting value predictions across minibatches
        batch1 = torch.randn(32, 1)  # 32 samples
        batch2 = torch.randn(32, 1)
        batch3 = None  # Skipped batch
        batch4 = torch.randn(32, 1)

        result = DistributionalPPO._concat_tensor_batches([batch1, batch2, batch3, batch4])

        assert result is not None
        assert result.shape == (96, 1)  # 32 * 3 valid batches

    def test_concat_keys_typical_training_scenario(self):
        """Test typical training scenario with symbol keys."""
        # Simulate collecting symbol keys across minibatches
        symbols_batch1 = ["BTCUSDT"] * 10 + ["ETHUSDT"] * 10
        symbols_batch2 = ["BNBUSDT"] * 5
        symbols_batch3 = []  # Empty batch

        result = DistributionalPPO._concat_string_keys([symbols_batch1, symbols_batch2, symbols_batch3])

        assert result is not None
        assert len(result) == 25
        assert result.count("BTCUSDT") == 10
        assert result.count("ETHUSDT") == 10
        assert result.count("BNBUSDT") == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
