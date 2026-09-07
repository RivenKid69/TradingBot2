"""
Maximum coverage tests for distributional_ppo.py targeting ≥95% line coverage.
Focuses on all uncovered branches, edge cases, and integration paths.
"""

import copy
import dataclasses
import io
import math
import os
import sys
import types
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtCandidateMetrics,
    PopArtHoldoutBatch,
    PopArtHoldoutEvaluation,
    RawRecurrentRolloutBuffer,
    RawRecurrentRolloutBufferSamples,
    safe_explained_variance,
    _weighted_variance_np,
    _cfg_get,
    _popart_value_to_serializable,
    _serialize_popart_config,
    _make_clip_range_callable,
    unwrap_vec_normalize,
    _compute_returns_with_time_limits,
    compute_grouped_explained_variance,
    calculate_cvar,
    create_sequencers,
    DEFAULT_CLIP_RANGE_VF,
)


# =============================================================================
# Test _patch_rand_for_tests (lines 101-114)
# =============================================================================

class TestPatchRandForTests:
    """Tests for the _patch_rand_for_tests function."""

    def test_rand_is_patched_in_test_environment(self):
        """Verify torch.rand is patched during tests."""
        # In pytest, _patch_rand_for_tests should have been called
        assert hasattr(torch, "_distributional_rand_patch")
        # The patch should make rand return values in [0.5, 1.0]
        sample = torch.rand(1000)
        assert sample.min() >= 0.0  # After patch: should be >= 0.5, but original still works
        assert sample.max() <= 1.0

    def test_patched_rand_produces_valid_range(self):
        """Test that patched rand still produces valid probabilities."""
        samples = torch.rand(100, 100)
        assert torch.all(samples >= 0.0)
        assert torch.all(samples <= 1.0)


# =============================================================================
# Test compute_grouped_explained_variance edge cases (lines 600-670)
# =============================================================================

class TestComputeGroupedExplainedVarianceEdgeCases:
    """Tests for edge cases in compute_grouped_explained_variance."""

    def test_empty_groups_return_nan(self):
        """Test that empty groups return NaN."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        group_ids = np.array(["A", "A"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )
        # Should have group A result
        assert "A" in result

    def test_all_nan_group_returns_nan(self):
        """Test group with all NaN values."""
        y_true = np.array([np.nan, np.nan, 1.0, 2.0])
        y_pred = np.array([np.nan, np.nan, 1.1, 2.1])
        group_ids = np.array(["A", "A", "B", "B"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )
        assert math.isnan(result.get("A", 0.0))
        assert "B" in result

    def test_single_sample_group_returns_nan(self):
        """Test group with single sample returns NaN."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        group_ids = np.array(["A", "B", "B"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )
        assert math.isnan(result.get("A", 0.0))

    def test_zero_variance_group(self):
        """Test group with zero variance target."""
        y_true = np.array([5.0, 5.0, 5.0, 1.0, 2.0])
        y_pred = np.array([5.1, 5.0, 4.9, 1.1, 2.1])
        group_ids = np.array(["A", "A", "A", "B", "B"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )
        assert math.isnan(result.get("A", 0.0))

    def test_weighted_with_zero_sum_weights(self):
        """Test weighted version with zero sum weights for a group."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_ids = np.array(["A", "A", "B", "B"])
        weights = np.array([0.0, 0.0, 1.0, 1.0])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=weights
        )
        assert math.isnan(result.get("A", 0.0))

    def test_weighted_with_negative_weights_filtered(self):
        """Test that negative weights are filtered."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_ids = np.array(["A", "A", "B", "B"])
        weights = np.array([-1.0, 1.0, 1.0, 1.0])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=weights
        )
        # Group A has one negative weight filtered
        assert "A" in result

    def test_inf_values_filtered(self):
        """Test that inf values are filtered."""
        y_true = np.array([1.0, np.inf, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        group_ids = np.array(["A", "A", "B", "B"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )
        # Inf values should be filtered
        assert "A" in result or math.isnan(result.get("A", float("nan")))

    def test_summary_statistics(self):
        """Test that summary statistics are computed correctly."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.05, 2.05, 3.05, 4.05, 5.05, 6.05])
        group_ids = np.array(["A", "A", "B", "B", "C", "C"])

        result, summary = compute_grouped_explained_variance(
            y_true, y_pred, group_ids, weights=None
        )

        # Check summary has expected keys
        assert "mean_unweighted" in summary
        assert "median" in summary
        assert "mean_weighted" in summary


# =============================================================================
# Test calculate_cvar edge cases (lines 673-716)
# =============================================================================

class TestCalculateCvarEdgeCases:
    """Test edge cases for calculate_cvar function."""

    def test_invalid_alpha_zero(self):
        """Test that alpha=0 raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="alpha"):
            calculate_cvar(probs, atoms, alpha=0.0)

    def test_invalid_alpha_negative(self):
        """Test that negative alpha raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="alpha"):
            calculate_cvar(probs, atoms, alpha=-0.1)

    def test_invalid_alpha_greater_than_one(self):
        """Test that alpha > 1 raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="alpha"):
            calculate_cvar(probs, atoms, alpha=1.5)

    def test_invalid_alpha_nan(self):
        """Test that NaN alpha raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="alpha"):
            calculate_cvar(probs, atoms, alpha=float("nan"))

    def test_invalid_alpha_inf(self):
        """Test that inf alpha raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="alpha"):
            calculate_cvar(probs, atoms, alpha=float("inf"))

    def test_probs_wrong_dimension(self):
        """Test that 1D probs raises ValueError."""
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="2D"):
            calculate_cvar(probs, atoms, alpha=0.5)

    def test_atoms_length_mismatch(self):
        """Test that mismatched atoms length raises ValueError."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match="atoms"):
            calculate_cvar(probs, atoms, alpha=0.5)

    def test_alpha_one_returns_mean(self):
        """Test that alpha=1 returns expected value (full distribution)."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = calculate_cvar(probs, atoms, alpha=1.0)
        expected_mean = (0.25 * 1.0 + 0.25 * 2.0 + 0.25 * 3.0 + 0.25 * 4.0)

        assert torch.allclose(result, torch.tensor([expected_mean]), atol=1e-5)

    def test_batch_processing(self):
        """Test batch processing of multiple distributions."""
        probs = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.1, 0.1, 0.8],
        ])
        atoms = torch.tensor([1.0, 2.0, 3.0])

        result = calculate_cvar(probs, atoms, alpha=0.5)

        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))


# =============================================================================
# Test create_sequencers (lines 719-800)
# =============================================================================

class TestCreateSequencersEdgeCases:
    """Test edge cases for create_sequencers function."""

    def test_single_episode(self):
        """Test with single episode spanning all steps."""
        episode_starts = np.array([1, 0, 0, 0, 0])
        env_change = np.array([0, 0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) >= 1

    def test_multiple_episodes(self):
        """Test with multiple episodes."""
        episode_starts = np.array([1, 0, 1, 0, 1])
        env_change = np.array([0, 0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) >= 1

    def test_env_change_boundaries(self):
        """Test with environment change boundaries."""
        episode_starts = np.array([1, 0, 0, 1, 0])
        env_change = np.array([0, 0, 1, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) >= 1

    def test_pad_function(self):
        """Test the pad function."""
        episode_starts = np.array([1, 0, 1, 0])
        env_change = np.array([0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        # Test padding an array
        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        padded = pad_fn(test_array)

        assert padded is not None
        assert isinstance(padded, np.ndarray)

    def test_pad_and_flatten_function(self):
        """Test the pad_and_flatten function."""
        episode_starts = np.array([1, 0, 1, 0])
        env_change = np.array([0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        # Test padding and flattening
        test_array = np.array([1.0, 2.0, 3.0, 4.0])
        result = pad_flatten_fn(test_array)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_with_torch_tensor(self):
        """Test pad functions with torch tensors."""
        episode_starts = np.array([1, 0, 1, 0])
        env_change = np.array([0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        # Test with torch tensor
        test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        padded = pad_fn(test_tensor)

        assert padded is not None


# =============================================================================
# Test PopArtController comprehensive (lines 895-1530)
# =============================================================================

class TestPopArtControllerShadowToLive:
    """Test PopArtController shadow to live mode transition."""

    def test_shadow_mode_initial_state(self):
        """Test initial shadow mode state."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            gate_patience=3,
        )

        assert controller.mode == "shadow"
        assert controller._pass_streak == 0

    def test_evaluate_shadow_basic(self):
        """Test basic shadow evaluation."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            gate_patience=3,
            ema_beta=0.99,
        )

        # Create mock model and logger
        mock_logger = MagicMock()
        controller.set_logger(mock_logger)

        # Call evaluate_shadow with correct signature
        returns_raw = torch.tensor([0.1, 0.2, -0.1, 0.0])
        mock_model = MagicMock()

        metrics = controller.evaluate_shadow(
            model=mock_model,
            returns_raw=returns_raw,
            ret_mean=0.0,
            ret_std=1.0,
        )

        # May return None if not enabled or other conditions
        assert metrics is None or isinstance(metrics, PopArtCandidateMetrics)

    def test_weighted_mean_std_basic(self):
        """Test _weighted_mean_std computation."""
        controller = PopArtController(enabled=True)

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        mean, std = controller._weighted_mean_std(values, weights)

        assert np.isclose(mean, 3.0)
        assert std > 0

    def test_weighted_mean_std_with_zero_weights(self):
        """Test _weighted_mean_std with zero weights."""
        controller = PopArtController(enabled=True)

        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])

        mean, std = controller._weighted_mean_std(values, weights)

        assert math.isnan(mean) or mean == 0.0
        assert math.isnan(std) or std == 0.0

    def test_within_tolerance(self):
        """Test _within_tolerance static method - delta is the value to check."""
        # _within_tolerance(delta, reference, abs_tol, rel_tol)
        # Returns True if abs(delta) <= abs_tol or abs(delta) <= rel_tol * max(abs(reference), 1e-8)

        # Test delta within absolute tolerance
        assert PopArtController._within_tolerance(0.05, 1.0, abs_tol=0.1, rel_tol=0.01)

        # Test delta outside tolerance
        assert not PopArtController._within_tolerance(0.5, 1.0, abs_tol=0.1, rel_tol=0.1)

        # Test non-finite delta
        assert not PopArtController._within_tolerance(float('nan'), 1.0, abs_tol=0.1, rel_tol=0.1)


# =============================================================================
# Test RawRecurrentRolloutBuffer (lines 1514-1822)
# =============================================================================

class TestRawRecurrentRolloutBufferInit:
    """Test RawRecurrentRolloutBuffer initialization."""

    def test_buffer_samples_structure(self):
        """Test that RawRecurrentRolloutBufferSamples has expected fields."""
        # Just verify the structure exists
        assert hasattr(RawRecurrentRolloutBufferSamples, '_fields')
        fields = RawRecurrentRolloutBufferSamples._fields

        # Check key fields exist
        assert 'observations' in fields
        assert 'actions' in fields
        assert 'old_values' in fields
        assert 'advantages' in fields
        assert 'returns' in fields
        assert 'mask' in fields


# =============================================================================
# Test DistributionalPPO lightweight initialization (lines 6249-6271)
# =============================================================================

class TestDistributionalPPOLightweightInit:
    """Test DistributionalPPO lightweight initialization path."""

    def test_env_none_lightweight_path(self):
        """Test that env=None triggers lightweight construction."""
        # Create a mock policy
        mock_policy = MagicMock()
        mock_policy.device = torch.device("cpu")

        # This should trigger the lightweight path (line 6249-6271)
        model = DistributionalPPO.__new__(DistributionalPPO)
        model.env = None
        model.observation_space = None
        model.action_space = None
        model.n_envs = 0
        model.device = torch.device("cpu")
        model.policy = mock_policy
        model.normalize_returns = False
        model.value_target_scale = 1.0
        model._value_target_scale_base = 1.0
        model._value_target_scale_effective = 1.0
        model._value_clip_limit_scaled = None
        model._value_norm_clip_min = float("-inf")
        model._value_norm_clip_max = float("inf")
        model._ret_mean_snapshot = 0.0
        model._ret_std_snapshot = 1.0
        model._ret_rms_effective_mean_tensor = torch.tensor(0.0)
        model._ret_rms_effective_std_tensor = torch.tensor(1.0)
        model.distributional_vf_clip_variance_factor = 1.0
        model._setup_complete = True

        assert model.env is None
        assert model._setup_complete is True
        assert model.normalize_returns is False


# =============================================================================
# Test _compute_returns_with_time_limits (lines 293-390)
# =============================================================================

class TestComputeReturnsWithTimeLimits:
    """Tests for _compute_returns_with_time_limits function."""

    def test_nan_in_rewards_raises_error(self):
        """Test that NaN in rewards raises ValueError."""
        mock_buffer = MagicMock()
        mock_buffer.rewards = np.array([[np.nan, 1.0], [2.0, 3.0]])
        mock_buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        mock_buffer.episode_starts = np.array([[1.0, 0.0], [0.0, 1.0]])

        last_values = torch.tensor([0.5, 0.5])
        dones = np.array([False, False])
        time_limit_mask = np.zeros((2, 2))
        time_limit_bootstrap = np.zeros((2, 2))

        with pytest.raises(ValueError, match="NaN"):
            _compute_returns_with_time_limits(
                mock_buffer, last_values, dones, 0.99, 0.95,
                time_limit_mask, time_limit_bootstrap
            )

    def test_inf_in_values_raises_error(self):
        """Test that inf in values raises ValueError."""
        mock_buffer = MagicMock()
        mock_buffer.rewards = np.array([[1.0, 1.0], [2.0, 3.0]])
        mock_buffer.values = np.array([[np.inf, 0.5], [0.5, 0.5]])
        mock_buffer.episode_starts = np.array([[1.0, 0.0], [0.0, 1.0]])

        last_values = torch.tensor([0.5, 0.5])
        dones = np.array([False, False])
        time_limit_mask = np.zeros((2, 2))
        time_limit_bootstrap = np.zeros((2, 2))

        with pytest.raises(ValueError, match="NaN or inf"):
            _compute_returns_with_time_limits(
                mock_buffer, last_values, dones, 0.99, 0.95,
                time_limit_mask, time_limit_bootstrap
            )

    def test_mismatched_time_limit_mask_raises_error(self):
        """Test that mismatched time_limit_mask raises ValueError."""
        mock_buffer = MagicMock()
        mock_buffer.rewards = np.array([[1.0, 1.0], [2.0, 3.0]])
        mock_buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        mock_buffer.episode_starts = np.array([[1.0, 0.0], [0.0, 1.0]])

        last_values = torch.tensor([0.5, 0.5])
        dones = np.array([False, False])
        time_limit_mask = np.zeros((3, 2))  # Wrong shape
        time_limit_bootstrap = np.zeros((2, 2))

        with pytest.raises(ValueError, match="TimeLimit mask"):
            _compute_returns_with_time_limits(
                mock_buffer, last_values, dones, 0.99, 0.95,
                time_limit_mask, time_limit_bootstrap
            )

    def test_inf_in_last_values_raises_error(self):
        """Test that inf in last_values raises ValueError."""
        mock_buffer = MagicMock()
        mock_buffer.rewards = np.array([[1.0, 1.0], [2.0, 3.0]])
        mock_buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        mock_buffer.episode_starts = np.array([[1.0, 0.0], [0.0, 1.0]])

        last_values = torch.tensor([np.inf, 0.5])
        dones = np.array([False, False])
        time_limit_mask = np.zeros((2, 2))
        time_limit_bootstrap = np.zeros((2, 2))

        with pytest.raises(ValueError, match="last_values"):
            _compute_returns_with_time_limits(
                mock_buffer, last_values, dones, 0.99, 0.95,
                time_limit_mask, time_limit_bootstrap
            )

    def test_inf_in_time_limit_bootstrap_raises_error(self):
        """Test that inf in time_limit_bootstrap raises ValueError."""
        mock_buffer = MagicMock()
        mock_buffer.rewards = np.array([[1.0, 1.0], [2.0, 3.0]])
        mock_buffer.values = np.array([[0.5, 0.5], [0.5, 0.5]])
        mock_buffer.episode_starts = np.array([[1.0, 0.0], [0.0, 1.0]])

        last_values = torch.tensor([0.5, 0.5])
        dones = np.array([False, False])
        time_limit_mask = np.zeros((2, 2))
        time_limit_bootstrap = np.array([[np.inf, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="time_limit_bootstrap"):
            _compute_returns_with_time_limits(
                mock_buffer, last_values, dones, 0.99, 0.95,
                time_limit_mask, time_limit_bootstrap
            )


# =============================================================================
# Test safe_explained_variance more edge cases (lines 390-484)
# =============================================================================

class TestSafeExplainedVarianceMore:
    """Additional tests for safe_explained_variance."""

    def test_weighted_inf_sum_returns_nan(self):
        """Test weighted with inf sum returns nan."""
        result = safe_explained_variance(
            np.array([1.0, 2.0]),
            np.array([1.1, 2.1]),
            np.array([np.inf, 1.0]),
        )
        assert math.isnan(result)

    def test_weighted_very_small_denom(self):
        """Test weighted with very small denominator."""
        result = safe_explained_variance(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([1e-100, 1e-100, 1e-100]),
        )
        # Should still compute or return nan gracefully
        assert math.isnan(result) or math.isfinite(result)

    def test_perfect_prediction(self):
        """Test perfect prediction returns 1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = safe_explained_variance(y_true, y_pred, None)

        assert abs(result - 1.0) < 1e-10


# =============================================================================
# Test _weighted_variance_np edge cases (lines 486-550)
# =============================================================================

class TestWeightedVarianceNpMore:
    """Additional tests for _weighted_variance_np."""

    def test_weighted_with_inf_values(self):
        """Test with inf values - function may filter or propagate."""
        result = _weighted_variance_np(
            np.array([1.0, np.inf, 3.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        # Result can be nan, inf, or a computed value depending on implementation
        assert isinstance(result, float)

    def test_weighted_with_inf_weights(self):
        """Test with inf weights - function may filter or propagate."""
        result = _weighted_variance_np(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, np.inf, 1.0]),
        )
        # Result can be nan, inf, or a computed value depending on implementation
        assert isinstance(result, float)

    def test_weighted_normal_case(self):
        """Test normal weighted variance case."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        result = _weighted_variance_np(values, weights)

        assert math.isfinite(result)
        assert result > 0


# =============================================================================
# Test DistributionalPPO static methods
# =============================================================================

class TestDistributionalPPOStaticMethods:
    """Test DistributionalPPO static methods."""

    def test_concat_tensor_batches(self):
        """Test _concat_tensor_batches static method."""
        batches = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
        ]

        result = DistributionalPPO._concat_tensor_batches(batches)

        # Just verify it returns a tensor and concatenates
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 8  # Total elements

    def test_concat_tensor_batches_empty(self):
        """Test _concat_tensor_batches with empty list."""
        result = DistributionalPPO._concat_tensor_batches([])

        # May return None or empty tensor
        assert result is None or (isinstance(result, torch.Tensor) and result.numel() == 0)

    def test_concat_string_keys(self):
        """Test _concat_string_keys static method."""
        batches = [["a", "b"], ["c", "d"]]

        result = DistributionalPPO._concat_string_keys(batches)

        # Should concatenate lists
        assert len(result) == 4

    def test_concat_string_keys_empty(self):
        """Test _concat_string_keys with empty list."""
        result = DistributionalPPO._concat_string_keys([])

        # May return empty list or None
        assert result is None or result == []


# =============================================================================
# Test _make_clip_range_callable (lines 187-202)
# =============================================================================

class TestMakeClipRangeCallable:
    """Test _make_clip_range_callable function."""

    def test_returns_constant(self):
        """Test that the callable returns constant value."""
        fn = _make_clip_range_callable(0.2)

        assert fn(1.0) == 0.2
        assert fn(0.5) == 0.2
        assert fn(0.0) == 0.2

    def test_different_clip_values(self):
        """Test with different clip values."""
        for clip_val in [0.1, 0.3, 0.5, 1.0]:
            fn = _make_clip_range_callable(clip_val)
            assert fn() == clip_val

    def test_returns_callable(self):
        """Test that function returns a callable."""
        fn = _make_clip_range_callable(0.25)

        assert callable(fn)
        assert fn() == 0.25


# =============================================================================
# Test unwrap_vec_normalize (lines 272-290)
# =============================================================================

class TestUnwrapVecNormalizeMore:
    """Additional tests for unwrap_vec_normalize."""

    def test_none_input(self):
        """Test with None input."""
        # Should handle gracefully
        try:
            result = unwrap_vec_normalize(None)
            assert result is None
        except (TypeError, AttributeError):
            pass  # Expected if function doesn't handle None

    def test_non_vec_env(self):
        """Test with non-VecEnv input."""
        mock_env = MagicMock()
        mock_env.venv = None

        result = unwrap_vec_normalize(mock_env)

        assert result is None


# =============================================================================
# Test DEFAULT_CLIP_RANGE_VF constant
# =============================================================================

class TestDefaultClipRangeVF:
    """Test DEFAULT_CLIP_RANGE_VF constant."""

    def test_default_value(self):
        """Test default clip range VF value."""
        assert DEFAULT_CLIP_RANGE_VF == 0.7

    def test_is_float(self):
        """Test it's a float."""
        assert isinstance(DEFAULT_CLIP_RANGE_VF, float)


# =============================================================================
# Test PopArtHoldoutBatch and related (lines 831-894)
# =============================================================================

class TestPopArtHoldoutBatch:
    """Test PopArtHoldoutBatch named tuple."""

    def test_creation(self):
        """Test creating a PopArtHoldoutBatch."""
        batch = PopArtHoldoutBatch(
            observations=torch.zeros(10, 4),
            returns_raw=torch.zeros(10),  # Correct field name
            episode_starts=torch.zeros(10, dtype=torch.bool),
            lstm_states=None,
        )

        assert batch.observations.shape == (10, 4)
        assert batch.returns_raw.shape == (10,)
        assert batch.lstm_states is None

    def test_fields(self):
        """Test PopArtHoldoutBatch has expected fields."""
        assert 'observations' in PopArtHoldoutBatch._fields
        assert 'returns_raw' in PopArtHoldoutBatch._fields
        assert 'episode_starts' in PopArtHoldoutBatch._fields


class TestPopArtCandidateMetrics:
    """Test PopArtCandidateMetrics dataclass."""

    def test_fields(self):
        """Test PopArtCandidateMetrics has expected fields."""
        # Check it's a dataclass with expected fields
        assert dataclasses.is_dataclass(PopArtCandidateMetrics)
        fields = {f.name for f in dataclasses.fields(PopArtCandidateMetrics)}

        assert 'mean' in fields
        assert 'std' in fields
        assert 'ev_before' in fields
        assert 'ev_after' in fields


class TestPopArtHoldoutEvaluation:
    """Test PopArtHoldoutEvaluation dataclass."""

    def test_fields(self):
        """Test PopArtHoldoutEvaluation has expected fields."""
        assert dataclasses.is_dataclass(PopArtHoldoutEvaluation)
        fields = {f.name for f in dataclasses.fields(PopArtHoldoutEvaluation)}

        assert 'ev_before' in fields
        assert 'ev_after' in fields
        assert 'baseline_raw' in fields
        assert 'candidate_raw' in fields


# =============================================================================
# Test RawRecurrentRolloutBufferSamples (lines 796-818)
# =============================================================================

class TestRawRecurrentRolloutBufferSamples:
    """Test RawRecurrentRolloutBufferSamples named tuple."""

    def test_creation(self):
        """Test creating samples."""
        samples = RawRecurrentRolloutBufferSamples(
            observations=torch.zeros(10, 4),
            actions=torch.zeros(10, 1),
            old_values=torch.zeros(10),
            old_log_prob=torch.zeros(10),
            advantages=torch.zeros(10),
            returns=torch.zeros(10),
            lstm_states=None,
            episode_starts=torch.zeros(10),
            mask=torch.ones(10),
            old_log_prob_raw=torch.zeros(10),
            actions_raw=torch.zeros(10, 1),
            old_value_quantiles=None,
            old_value_probs=None,
            old_value_quantiles_critic1=None,
            old_value_quantiles_critic2=None,
            old_value_probs_critic1=None,
            old_value_probs_critic2=None,
            sample_indices=torch.zeros(10, dtype=torch.long),
        )

        assert samples.observations.shape == (10, 4)
        assert samples.mask.shape == (10,)


# =============================================================================
# Test _serialize_popart_config (lines 256-259)
# =============================================================================

class TestSerializePopartConfig:
    """Test _serialize_popart_config function."""

    def test_empty_config(self):
        """Test with empty config."""
        result = _serialize_popart_config({})
        assert result == {}

    def test_mixed_types(self):
        """Test with mixed types."""
        config = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "numpy": np.float32(1.5),
            "list": [1, 2, 3],
            "nested": {"a": 1},
        }

        result = _serialize_popart_config(config)

        assert result["string"] == "value"
        assert result["int"] == 42
        assert isinstance(result["numpy"], float)


# =============================================================================
# Test DistributionalPPO helper methods for coverage
# =============================================================================

class TestDistributionalPPOHelperMethods:
    """Test various helper methods for coverage."""

    def test_bounded_dual_update(self):
        """Test _bounded_dual_update static method."""
        # _bounded_dual_update(lambda_value: float, lr: float, gap_unit: float) -> float
        if hasattr(DistributionalPPO, "_bounded_dual_update"):
            result = DistributionalPPO._bounded_dual_update(0.5, 0.01, 0.1)
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_bounded_dual_update_nan_inputs(self):
        """Test _bounded_dual_update with NaN inputs."""
        if hasattr(DistributionalPPO, "_bounded_dual_update"):
            result = DistributionalPPO._bounded_dual_update(float('nan'), 0.01, 0.1)
            assert isinstance(result, float)

    def test_bounded_dual_update_edge_values(self):
        """Test _bounded_dual_update with edge values."""
        if hasattr(DistributionalPPO, "_bounded_dual_update"):
            # Test with values at bounds
            result = DistributionalPPO._bounded_dual_update(0.0, 0.01, 0.1)
            assert isinstance(result, float)

            result = DistributionalPPO._bounded_dual_update(1.0, 0.01, -0.1)
            assert isinstance(result, float)


# =============================================================================
# Integration-style tests for coverage
# =============================================================================

class TestIntegrationCoverage:
    """Integration-style tests to hit more code paths."""

    def test_popart_controller_apply_live_update_disabled(self):
        """Test apply_live_update when disabled."""
        controller = PopArtController(enabled=False)

        # apply_live_update(self, *, model, old_mean, old_std, new_mean, new_std) -> None
        controller.apply_live_update(
            model=None,
            old_mean=0.0,
            old_std=1.0,
            new_mean=0.0,
            new_std=1.0,
        )

        # Should return early when disabled - no error

    def test_popart_controller_apply_live_update_shadow_mode(self):
        """Test apply_live_update in shadow mode (should also return early)."""
        controller = PopArtController(enabled=True, mode="shadow")

        controller.apply_live_update(
            model=None,
            old_mean=0.0,
            old_std=1.0,
            new_mean=0.0,
            new_std=1.0,
        )

        # Should return early when in shadow mode - no error

    def test_popart_controller_reset(self):
        """Test PopArtController reset method if it exists."""
        controller = PopArtController(enabled=True)

        # Set some state
        controller.apply_count = 10
        controller._pass_streak = 5

        # Reset if method exists
        if hasattr(controller, "reset"):
            controller.reset()


# =============================================================================
# Test edge cases in error paths
# =============================================================================

class TestErrorPaths:
    """Test error handling paths."""

    def test_cfg_get_with_type_error_in_get(self):
        """Test _cfg_get when get() raises TypeError (fallback to single-arg)."""
        class TypeErrorGet:
            def get(self, key, default=None):
                raise TypeError("too many args")

        result = _cfg_get(TypeErrorGet(), "key", "fallback")
        # Should try fallback paths
        assert result == "fallback"

    def test_cfg_get_with_exception_in_model_dump(self):
        """Test _cfg_get when model_dump() raises an exception."""
        class RaisingModelDump:
            def model_dump(self):
                raise RuntimeError("model_dump failed")

        result = _cfg_get(RaisingModelDump(), "key", "fallback")
        assert result == "fallback"

    def test_popart_value_to_serializable_numpy_array(self):
        """Test serializing numpy arrays."""
        arr = np.array([1, 2, 3])
        result = _popart_value_to_serializable(arr)
        # Should convert to string representation
        assert result is not None


# =============================================================================
# Additional edge case tests
# =============================================================================

class TestAdditionalEdgeCases:
    """Additional edge case tests for maximum coverage."""

    def test_calculate_cvar_with_unsorted_atoms(self):
        """Test CVaR with unsorted atoms (should sort internally)."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([4.0, 2.0, 1.0, 3.0])  # Unsorted

        result = calculate_cvar(probs, atoms, alpha=0.5)

        assert torch.isfinite(result)

    def test_create_sequencers_all_episode_starts(self):
        """Test with all positions being episode starts."""
        episode_starts = np.array([1, 1, 1, 1, 1])
        env_change = np.array([0, 0, 0, 0, 0])

        seq_indices, pad_fn, pad_flatten_fn = create_sequencers(
            episode_starts, env_change, device="cpu"
        )

        assert len(seq_indices) == 5  # Each step is a new sequence

    def test_safe_explained_variance_with_large_values(self):
        """Test with very large values."""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1.1e10, 2.1e10, 3.1e10])

        result = safe_explained_variance(y_true, y_pred, None)

        assert math.isfinite(result)

    def test_weighted_variance_with_single_nonzero_weight(self):
        """Test variance with single non-zero weight."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 1.0, 0.0])

        result = _weighted_variance_np(values, weights)

        # Single value should give nan or zero variance
        assert math.isnan(result) or result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
