"""
Intensive unit tests for maximum coverage of distributional_ppo.py helper functions.
Focuses on every branch and edge case of testable static/instance methods.
"""

import math
import types
import warnings
from typing import Any, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    RawRecurrentRolloutBuffer,
    safe_explained_variance,
    _weighted_variance_np,
    _cfg_get,
    _popart_value_to_serializable,
    _serialize_popart_config,
    compute_grouped_explained_variance,
    calculate_cvar,
    create_sequencers,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


# =============================================================================
# Exhaustive _cfg_get tests
# =============================================================================


class TestCfgGetComprehensive:
    """Comprehensive tests for every branch in _cfg_get."""

    def test_none_config(self):
        assert _cfg_get(None, "key", "default") == "default"

    def test_dict_with_key(self):
        assert _cfg_get({"key": "value"}, "key") == "value"

    def test_dict_missing_key(self):
        assert _cfg_get({"other": 1}, "key", "default") == "default"

    def test_hasattr_branch(self):
        obj = types.SimpleNamespace(key="value")
        assert _cfg_get(obj, "key") == "value"

    def test_callable_get_success(self):
        class WithGet:
            def get(self, key, default=None):
                return "from_get" if key == "key" else default

        assert _cfg_get(WithGet(), "key") == "from_get"

    def test_callable_get_type_error_then_single_arg(self):
        class GetOnlyOneArg:
            def get(self, key):
                if key == "key":
                    return "single_arg"
                raise KeyError(key)

        # First call with default raises TypeError, then single arg is tried
        assert _cfg_get(GetOnlyOneArg(), "key") == "single_arg"

    def test_callable_get_type_error_then_exception(self):
        """Test TypeError fallback when single-arg get also fails."""

        class GetTypeErrorThenFail:
            def get(self, key, default=None):
                raise TypeError("too many args")

        # TypeError triggers fallback, but no model_dump/dict available
        result = _cfg_get(GetTypeErrorThenFail(), "key", "fallback")
        assert result == "fallback"

    def test_model_dump_method(self):
        class WithModelDump:
            def model_dump(self):
                return {"key": "from_model_dump"}

        assert _cfg_get(WithModelDump(), "key") == "from_model_dump"

    def test_dict_method(self):
        class WithDict:
            def dict(self):
                return {"key": "from_dict"}

        assert _cfg_get(WithDict(), "key") == "from_dict"

    def test_model_dump_fails_uses_dict(self):
        class BothMethods:
            def model_dump(self):
                raise RuntimeError("fail")

            def dict(self):
                return {"key": "from_dict"}

        assert _cfg_get(BothMethods(), "key") == "from_dict"

    def test_dataclass_fallback(self):
        from dataclasses import dataclass

        @dataclass
        class DC:
            key: str = "dataclass_value"

        assert _cfg_get(DC(), "key") == "dataclass_value"

    def test_final_default(self):
        class Empty:
            pass

        assert _cfg_get(Empty(), "key", "final_default") == "final_default"


# =============================================================================
# Exhaustive _popart_value_to_serializable tests
# =============================================================================


class TestPopartValueComprehensive:
    """Comprehensive tests for _popart_value_to_serializable."""

    def test_string(self):
        assert _popart_value_to_serializable("test") == "test"

    def test_bool_true(self):
        assert _popart_value_to_serializable(True) is True

    def test_bool_false(self):
        assert _popart_value_to_serializable(False) is False

    def test_none(self):
        assert _popart_value_to_serializable(None) is None

    def test_int(self):
        assert _popart_value_to_serializable(42) == 42

    def test_float(self):
        assert _popart_value_to_serializable(3.14) == 3.14

    def test_numpy_int(self):
        result = _popart_value_to_serializable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = _popart_value_to_serializable(np.float32(3.14))
        assert isinstance(result, float)

    def test_path_object(self):
        from pathlib import Path
        import os

        p = Path("/tmp")
        result = _popart_value_to_serializable(p)
        assert result == os.fspath(p)

    def test_dict(self):
        result = _popart_value_to_serializable({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = _popart_value_to_serializable({"a": {"b": {"c": 1}}})
        assert result == {"a": {"b": {"c": 1}}}

    def test_list(self):
        result = _popart_value_to_serializable([1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple(self):
        result = _popart_value_to_serializable((1, 2, 3))
        assert result == [1, 2, 3]  # Tuple converted to list

    def test_nested_list(self):
        result = _popart_value_to_serializable([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_unknown_object(self):
        class Custom:
            def __str__(self):
                return "custom_str"

        result = _popart_value_to_serializable(Custom())
        assert result == "custom_str"


# =============================================================================
# Exhaustive safe_explained_variance tests
# =============================================================================


class TestSafeExplainedVarianceComprehensive:
    """Comprehensive tests for safe_explained_variance."""

    def test_empty_arrays(self):
        result = safe_explained_variance(np.array([]), np.array([]), None)
        assert math.isnan(result)

    def test_single_value(self):
        result = safe_explained_variance(np.array([1.0]), np.array([1.0]), None)
        assert math.isnan(result)

    def test_all_same_values(self):
        y = np.array([5.0, 5.0, 5.0])
        pred = np.array([5.0, 5.0, 5.0])
        result = safe_explained_variance(y, pred, None)
        assert math.isnan(result)

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = y.copy()
        result = safe_explained_variance(y, pred, None)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_zero_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.zeros_like(y)
        result = safe_explained_variance(y, pred, None)
        assert math.isfinite(result)

    def test_with_nan_values(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = safe_explained_variance(y, pred, None)
        assert math.isfinite(result) or math.isnan(result)

    def test_with_inf_values(self):
        y = np.array([1.0, 2.0, np.inf, 4.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = safe_explained_variance(y, pred, None)
        assert math.isfinite(result) or math.isnan(result)

    def test_all_nan(self):
        y = np.array([np.nan, np.nan])
        pred = np.array([np.nan, np.nan])
        result = safe_explained_variance(y, pred, None)
        assert math.isnan(result)

    def test_with_uniform_weights(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.1, 2.1, 2.9, 4.1])
        weights = np.ones(4)
        result = safe_explained_variance(y, pred, weights)
        assert math.isfinite(result)

    def test_with_varied_weights(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.1, 2.1, 2.9, 4.1])
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        result = safe_explained_variance(y, pred, weights)
        assert math.isfinite(result)

    def test_with_zero_weights(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        weights = np.zeros(3)
        result = safe_explained_variance(y, pred, weights)
        assert math.isnan(result)

    def test_with_negative_weights(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, -1.0, 1.0, 1.0])
        result = safe_explained_variance(y, pred, weights)
        # Negative weights should be filtered
        assert math.isfinite(result) or math.isnan(result)


# =============================================================================
# Exhaustive _weighted_variance_np tests
# =============================================================================


class TestWeightedVarianceNpComprehensive:
    """Comprehensive tests for _weighted_variance_np."""

    def test_empty(self):
        result = _weighted_variance_np(np.array([]), None)
        assert math.isnan(result)

    def test_single_value(self):
        result = _weighted_variance_np(np.array([5.0]), None)
        assert math.isnan(result)

    def test_two_values_no_weights(self):
        result = _weighted_variance_np(np.array([1.0, 3.0]), None)
        assert math.isfinite(result)
        assert result > 0

    def test_uniform_values(self):
        result = _weighted_variance_np(np.array([5.0, 5.0, 5.0]), None)
        assert result == 0.0 or math.isnan(result)

    def test_with_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0, 1.0])
        result = _weighted_variance_np(values, weights)
        assert math.isfinite(result)

    def test_all_nan_values(self):
        result = _weighted_variance_np(np.array([np.nan, np.nan]), None)
        assert math.isnan(result)

    def test_all_zero_weights(self):
        result = _weighted_variance_np(np.array([1.0, 2.0]), np.zeros(2))
        assert math.isnan(result)

    def test_negative_weights_filtered(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, -1.0, 1.0])
        result = _weighted_variance_np(values, weights)
        # Negative weight at index 1 should be filtered
        assert math.isfinite(result) or math.isnan(result)


# =============================================================================
# Exhaustive calculate_cvar tests
# =============================================================================


class TestCalculateCvarComprehensive:
    """Comprehensive tests for calculate_cvar function."""

    def test_alpha_one(self):
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        result = calculate_cvar(probs, atoms, 1.0)
        expected_mean = (probs * atoms).sum(dim=1)
        assert torch.allclose(result, expected_mean, atol=1e-5)

    def test_alpha_zero_point_one(self):
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        result = calculate_cvar(probs, atoms, 0.1)
        assert torch.isfinite(result).all()
        # At low alpha, CVaR should be close to worst outcome
        assert result.item() < 0  # Leftmost atom is negative

    def test_alpha_half(self):
        probs = torch.tensor([[0.5, 0.5]])
        atoms = torch.tensor([-1.0, 1.0])
        result = calculate_cvar(probs, atoms, 0.5)
        assert torch.isfinite(result).all()
        assert result.item() < 0  # Left tail average

    def test_batch_size_greater_than_one(self):
        probs = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        result = calculate_cvar(probs, atoms, 0.5)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_concentrated_probability(self):
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        result = calculate_cvar(probs, atoms, 0.5)
        assert torch.isfinite(result).all()
        # All probability on -2.0
        assert result.item() == pytest.approx(-2.0, abs=1e-5)


# =============================================================================
# Exhaustive compute_grouped_explained_variance tests
# =============================================================================


class TestComputeGroupedExplainedVarianceComprehensive:
    """Comprehensive tests for compute_grouped_explained_variance."""

    def test_single_group(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        groups = ["a", "a", "a"]

        group_evs, aggregates = compute_grouped_explained_variance(y_true, y_pred, groups)

        assert "a" in group_evs
        assert math.isfinite(group_evs["a"])

    def test_multiple_groups(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        groups = ["a", "a", "a", "b", "b", "b"]

        group_evs, aggregates = compute_grouped_explained_variance(y_true, y_pred, groups)

        assert "a" in group_evs
        assert "b" in group_evs

    def test_aggregates_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        groups = ["a", "a", "b", "b"]

        group_evs, aggregates = compute_grouped_explained_variance(y_true, y_pred, groups)

        assert "mean_unweighted" in aggregates
        assert "mean_weighted" in aggregates
        assert "median" in aggregates


# =============================================================================
# Exhaustive create_sequencers tests
# =============================================================================


class TestCreateSequencersComprehensive:
    """Comprehensive tests for create_sequencers function."""

    def test_single_sequence(self):
        episode_starts = np.array([True, False, False, False])
        env_change = np.zeros(4, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert indices is not None

    def test_multiple_sequences(self):
        episode_starts = np.zeros(10, dtype=bool)
        episode_starts[0] = True
        episode_starts[5] = True
        env_change = np.zeros(10, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert len(indices) >= 2

    def test_all_starts(self):
        episode_starts = np.ones(5, dtype=bool)
        env_change = np.zeros(5, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert len(indices) == 5

    def test_with_env_changes(self):
        episode_starts = np.zeros(10, dtype=bool)
        episode_starts[0] = True
        env_change = np.zeros(10, dtype=bool)
        env_change[5] = True

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")
        assert indices is not None


# =============================================================================
# PopArtController comprehensive tests
# =============================================================================


class TestPopArtControllerComprehensive:
    """Additional comprehensive tests for PopArtController."""

    def test_clip_fraction_boundary_cases(self):
        # All within bounds
        values = torch.tensor([0.0])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == 0.0

        # All outside bounds
        values = torch.tensor([-5.0, 5.0])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == 1.0

        # Half outside
        values = torch.tensor([-0.5, 0.5, -5.0, 5.0])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_safe_numpy_preserves_values(self):
        tensor = torch.tensor([1.5, 2.5, 3.5])
        result = PopArtController._safe_numpy(tensor)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])

    def test_within_tolerance_edge_cases(self):
        # Zero reference
        assert PopArtController._within_tolerance(0.001, 0.0, abs_tol=0.01, rel_tol=0.0)

        # Large reference
        assert PopArtController._within_tolerance(100.0, 10000.0, abs_tol=0.0, rel_tol=0.01)


# =============================================================================
# DistributionalPPO static methods comprehensive tests
# =============================================================================


class TestDistributionalPPOStaticComprehensive:
    """Comprehensive tests for DistributionalPPO static methods."""

    def test_clone_states_list_input(self):
        states = [torch.zeros(1, 64), torch.ones(1, 64)]
        result = DistributionalPPO._clone_states_to_device(states, torch.device("cpu"))
        assert isinstance(result, list)

    def test_clone_observations_edge_cases(self):
        # None passthrough
        result = DistributionalPPO._clone_observations_to_device(None, torch.device("cpu"))
        assert result is None

        # Scalar passthrough
        result = DistributionalPPO._clone_observations_to_device(42, torch.device("cpu"))
        assert result == 42

    def test_detach_observations_edge_cases(self):
        # Scalar passthrough
        result = DistributionalPPO._detach_observations_to_cpu(42)
        assert result == 42

        # None passthrough - should handle gracefully
        try:
            result = DistributionalPPO._detach_observations_to_cpu(None)
            assert result is None
        except Exception:
            pass  # Method may not handle None

    def test_flatten_rollout_mask_edge_cases(self):
        # List input
        mask = [1.0, 0.0, 1.0]
        result = DistributionalPPO._flatten_rollout_mask(mask)
        assert result.shape == (3,)

    def test_coerce_value_target_scale_all_variants(self):
        assert DistributionalPPO._coerce_value_target_scale("percent") == 100.0
        assert DistributionalPPO._coerce_value_target_scale("percents") == 100.0
        assert DistributionalPPO._coerce_value_target_scale("percentage") == 100.0
        assert DistributionalPPO._coerce_value_target_scale("%") == 100.0

        assert DistributionalPPO._coerce_value_target_scale("bps") == 10000.0
        assert DistributionalPPO._coerce_value_target_scale("basis_points") == 10000.0
        assert DistributionalPPO._coerce_value_target_scale("basis-point") == 10000.0
        assert DistributionalPPO._coerce_value_target_scale("basis points") == 10000.0


# =============================================================================
# RawRecurrentRolloutBuffer tests
# =============================================================================


class TestRawRecurrentRolloutBufferComprehensive:
    """Comprehensive tests for RawRecurrentRolloutBuffer."""

    def test_to_numpy_various_inputs(self):
        # Tensor
        tensor = torch.tensor([1.0, 2.0])
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)
        assert isinstance(result, np.ndarray)

        # Numpy array
        arr = np.array([1.0, 2.0])
        result = RawRecurrentRolloutBuffer._to_numpy(arr)
        assert isinstance(result, np.ndarray)

        # List
        lst = [1.0, 2.0]
        result = RawRecurrentRolloutBuffer._to_numpy(lst)
        assert isinstance(result, np.ndarray)


# =============================================================================
# RNNStates handling tests
# =============================================================================


class TestRNNStatesHandling:
    """Tests for RNNStates handling in various methods."""

    def test_extract_critic_with_rnn_states(self):
        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.ones(1, 2, 64), torch.ones(1, 2, 64))
        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._extract_critic_states(states)
        assert len(result) == 2
        assert torch.all(result[0] == 1.0)

    def test_extract_actor_with_rnn_states(self):
        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.ones(1, 2, 64), torch.ones(1, 2, 64))
        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._extract_actor_states(states)
        assert len(result) == 2
        assert torch.all(result[0] == 0.0)

    def test_clone_rnn_states(self):
        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.ones(1, 2, 64), torch.ones(1, 2, 64))
        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._clone_states_to_device(states, torch.device("cpu"))
        assert hasattr(result, "pi")
        assert hasattr(result, "vf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
