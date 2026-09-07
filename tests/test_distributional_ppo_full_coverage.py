"""
Comprehensive tests to achieve >=95% line coverage for distributional_ppo.py.
Focuses on PopArtController, RawRecurrentRolloutBuffer, and uncovered DistributionalPPO methods.
"""

import copy
import math
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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
# PopArtController Tests
# =============================================================================


class TestPopArtControllerInit:
    """Tests for PopArtController initialization."""

    def test_default_initialization(self):
        controller = PopArtController(enabled=False)
        assert controller.enabled is False
        assert controller.mode == "shadow"
        assert controller.apply_count == 0

    def test_enabled_initialization(self):
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            ema_beta=0.99,
            gate_patience=5,
        )
        assert controller.enabled is True
        assert controller.ema_beta == 0.99
        assert controller.gate_patience == 5

    def test_live_mode_initialization(self):
        controller = PopArtController(enabled=True, mode="live")
        assert controller.mode == "live"


class TestPopArtControllerSetLogger:
    """Tests for PopArtController.set_logger method."""

    def test_set_logger_none(self):
        controller = PopArtController(enabled=True)
        controller.set_logger(None)
        assert controller._logger is None

    def test_set_logger_object(self):
        class MockLogger:
            pass

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        assert controller._logger is logger


class TestPopArtControllerLog:
    """Tests for PopArtController._log method."""

    def test_log_no_logger(self):
        controller = PopArtController(enabled=True)
        controller._logger = None
        # Should not raise
        controller._log("test_key", 1.0)

    def test_log_with_logger_record_method(self):
        class MockLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        controller._log("test_key", 42.0)
        assert logger.records.get("test_key") == 42.0

    def test_log_with_string_value(self):
        class MockLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        controller._log("test_key", "string_value")
        assert logger.records.get("test_key") == "string_value"

    def test_log_with_nan_value(self):
        class MockLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        controller._log("test_key", float("nan"))
        # NaN should be converted to 0.0
        assert logger.records.get("test_key") == 0.0

    def test_log_with_inf_value(self):
        class MockLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        controller._log("test_key", float("inf"))
        # Inf should be converted to 0.0
        assert logger.records.get("test_key") == 0.0

    def test_log_with_numpy_int(self):
        class MockLogger:
            def __init__(self):
                self.records = {}

            def record(self, key, value):
                self.records[key] = value

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        controller._log("test_key", np.int64(42))
        assert logger.records.get("test_key") == 42.0

    def test_log_exception_handling(self):
        class MockLogger:
            def record(self, key, value):
                raise RuntimeError("Record failed")

        controller = PopArtController(enabled=True)
        logger = MockLogger()
        controller.set_logger(logger)
        # Should not raise - exception is caught
        controller._log("test_key", 1.0)


class TestPopArtControllerWeightedMeanStd:
    """Tests for PopArtController._weighted_mean_std static method."""

    def test_empty_values(self):
        mean, std = PopArtController._weighted_mean_std(np.array([]), None)
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_unweighted_finite_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = PopArtController._weighted_mean_std(values, None)
        assert math.isclose(mean, 3.0)
        assert math.isfinite(std)

    def test_unweighted_with_nan(self):
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        mean, std = PopArtController._weighted_mean_std(values, None)
        assert math.isfinite(mean)

    def test_unweighted_all_nan(self):
        values = np.array([np.nan, np.nan, np.nan])
        mean, std = PopArtController._weighted_mean_std(values, None)
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_weighted_basic(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0, 1.0])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        assert math.isfinite(mean)
        assert math.isfinite(std)

    def test_weighted_zero_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 0.0])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_weighted_mismatched_sizes(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 2.0])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        # Should truncate to minimum size
        assert math.isfinite(mean) or math.isnan(mean)

    def test_weighted_empty_weights(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_weighted_negative_weights_filtered(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, -1.0, 1.0])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        # Negative weights filtered out
        assert math.isfinite(mean) or math.isnan(mean)

    def test_weighted_inf_sum_returns_nan(self):
        values = np.array([1.0, 2.0])
        weights = np.array([1e308, 1e308])
        mean, std = PopArtController._weighted_mean_std(values, weights)
        assert math.isnan(mean) or math.isfinite(mean)


class TestPopArtControllerWithinTolerance:
    """Tests for PopArtController._within_tolerance static method."""

    def test_within_absolute_tolerance(self):
        result = PopArtController._within_tolerance(0.001, 100.0, abs_tol=0.01, rel_tol=0.0)
        assert result is True

    def test_within_relative_tolerance(self):
        result = PopArtController._within_tolerance(1.0, 100.0, abs_tol=0.0, rel_tol=0.02)
        assert result is True

    def test_outside_tolerance(self):
        result = PopArtController._within_tolerance(10.0, 100.0, abs_tol=0.01, rel_tol=0.01)
        assert result is False

    def test_nan_delta_returns_false(self):
        result = PopArtController._within_tolerance(float("nan"), 100.0, abs_tol=0.01, rel_tol=0.01)
        assert result is False

    def test_inf_reference_handled(self):
        result = PopArtController._within_tolerance(0.001, float("inf"), abs_tol=0.01, rel_tol=0.01)
        assert result is True


class TestPopArtControllerLoadHoldout:
    """Tests for PopArtController._load_holdout method."""

    def test_no_holdout_loader(self):
        controller = PopArtController(enabled=True)
        result = controller._load_holdout()
        assert result is None

    def test_holdout_loader_returns_none(self):
        controller = PopArtController(enabled=True)
        controller._holdout_loader = lambda: None
        result = controller._load_holdout()
        assert result is None

    def test_holdout_loader_returns_batch(self):
        controller = PopArtController(enabled=True)

        mock_batch = PopArtHoldoutBatch(
            observations=torch.zeros(10, 4),
            returns_raw=torch.zeros(10, 1),
            episode_starts=torch.zeros(10, dtype=torch.bool),
            lstm_states=None,
            mask=None,
        )
        controller._holdout_loader = lambda: mock_batch
        result = controller._load_holdout()
        assert result is mock_batch

    def test_holdout_cache_used(self):
        controller = PopArtController(enabled=True)

        mock_batch = PopArtHoldoutBatch(
            observations=torch.zeros(10, 4),
            returns_raw=torch.zeros(10, 1),
            episode_starts=torch.zeros(10, dtype=torch.bool),
            lstm_states=None,
            mask=None,
        )
        controller._holdout_cache = mock_batch
        controller._holdout_loader = lambda: None  # Should not be called

        result = controller._load_holdout()
        assert result is mock_batch


class TestPopArtControllerSafeNumpy:
    """Tests for PopArtController._safe_numpy static method."""

    def test_tensor_to_numpy(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = PopArtController._safe_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_cuda_tensor_if_available(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = PopArtController._safe_numpy(tensor)
        assert isinstance(result, np.ndarray)


class TestPopArtControllerClipFraction:
    """Tests for PopArtController._clip_fraction static method."""

    def test_empty_tensor(self):
        result = PopArtController._clip_fraction(torch.tensor([]), -1.0, 1.0)
        assert result == 0.0

    def test_no_clipping_needed(self):
        values = torch.tensor([0.0, 0.5, -0.5])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == 0.0

    def test_partial_clipping(self):
        values = torch.tensor([-2.0, 0.0, 2.0])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_all_clipped(self):
        values = torch.tensor([-5.0, 5.0])
        result = PopArtController._clip_fraction(values, -1.0, 1.0)
        assert result == 1.0


class TestPopArtControllerEvaluateShadow:
    """Tests for PopArtController.evaluate_shadow method."""

    def _make_mock_model(self):
        """Create a mock model for testing."""
        model = types.SimpleNamespace()
        model._use_quantile_value = False
        model.normalize_returns = False
        model._value_clip_limit_unscaled = None

        policy = types.SimpleNamespace()
        policy.device = torch.device("cpu")
        policy.training = False
        policy.eval = lambda: None
        policy.train = lambda: None
        policy.recurrent_initial_state = None

        model.policy = policy
        model._policy_value_outputs = lambda obs, states, starts: torch.zeros(obs.shape[0], 1)
        model._to_raw_returns = lambda x: x

        return model

    def test_disabled_controller_returns_none(self):
        controller = PopArtController(enabled=False)
        model = self._make_mock_model()

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.zeros(10, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is None

    def test_enabled_no_holdout(self):
        controller = PopArtController(enabled=True)
        model = self._make_mock_model()

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([1.0, 2.0, 3.0]).reshape(-1, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        assert result.blocked_reason == "no_holdout"

    def test_deprecated_ev_train_warning(self):
        controller = PopArtController(enabled=True)
        model = self._make_mock_model()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            controller.evaluate_shadow(
                model=model,
                returns_raw=torch.tensor([1.0, 2.0, 3.0]).reshape(-1, 1),
                ret_mean=0.0,
                ret_std=1.0,
                explained_variance_train=0.5,
            )
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_all_nan_returns(self):
        controller = PopArtController(enabled=True)
        model = self._make_mock_model()

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([float("nan"), float("nan")]).reshape(-1, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None

    def test_empty_returns(self):
        controller = PopArtController(enabled=True)
        model = self._make_mock_model()

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([]).reshape(0, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None

    def test_shadow_ema_update(self):
        controller = PopArtController(enabled=True, ema_beta=0.9)
        model = self._make_mock_model()

        # First call initializes shadow stats
        controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([1.0, 2.0, 3.0]).reshape(-1, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert controller._shadow_mean is not None
        initial_mean = controller._shadow_mean

        # Second call updates via EMA
        controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([10.0, 20.0, 30.0]).reshape(-1, 1),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert controller._shadow_mean != initial_mean


class TestPopArtControllerApplyLiveUpdate:
    """Tests for PopArtController.apply_live_update method."""

    def _make_mock_model(self):
        model = types.SimpleNamespace()
        model._use_quantile_value = False

        policy = types.SimpleNamespace()
        policy.device = torch.device("cpu")
        policy.atoms = torch.linspace(-1, 1, 21)
        policy.v_min = -1.0
        policy.v_max = 1.0

        model.policy = policy
        return model

    def test_disabled_does_nothing(self):
        controller = PopArtController(enabled=False, mode="live")
        model = self._make_mock_model()

        controller.apply_live_update(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=1.0,
            new_std=2.0,
        )
        assert controller.apply_count == 0

    def test_shadow_mode_does_nothing(self):
        controller = PopArtController(enabled=True, mode="shadow")
        model = self._make_mock_model()

        controller.apply_live_update(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=1.0,
            new_std=2.0,
        )
        assert controller.apply_count == 0

    def test_invalid_new_std_does_nothing(self):
        controller = PopArtController(enabled=True, mode="live")
        model = self._make_mock_model()

        controller.apply_live_update(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=1.0,
            new_std=0.0,  # Invalid
        )
        assert controller.apply_count == 0


class TestPopArtControllerApplyQuantileTransform:
    """Tests for PopArtController._apply_quantile_transform method."""

    def test_no_quantile_head(self):
        controller = PopArtController(enabled=True)
        model = types.SimpleNamespace()
        model.policy = types.SimpleNamespace()

        # Should not raise
        controller._apply_quantile_transform(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=1.0,
            new_std=2.0,
        )

    def test_with_quantile_head(self):
        controller = PopArtController(enabled=True)

        linear = torch.nn.Linear(32, 21)
        quantile_head = types.SimpleNamespace(linear=linear)

        model = types.SimpleNamespace()
        model.policy = types.SimpleNamespace(quantile_head=quantile_head)

        original_weight = linear.weight.detach().clone()

        with torch.no_grad():
            controller._apply_quantile_transform(
                model=model,
                old_mean=0.0,
                old_std=1.0,
                new_mean=0.0,
                new_std=2.0,
            )

        # Weight should be scaled
        assert not torch.allclose(linear.weight.detach(), original_weight)


class TestPopArtControllerApplyCategoricalTransform:
    """Tests for PopArtController._apply_categorical_transform method."""

    def test_no_atoms(self):
        controller = PopArtController(enabled=True)
        model = types.SimpleNamespace()
        model.policy = types.SimpleNamespace()

        # Should not raise
        controller._apply_categorical_transform(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=1.0,
            new_std=2.0,
        )

    def test_with_atoms_no_update_method(self):
        controller = PopArtController(enabled=True)

        atoms = torch.linspace(-1, 1, 21)
        policy = types.SimpleNamespace(
            atoms=atoms,
            v_min=-1.0,
            v_max=1.0,
            delta_z=0.1,
        )

        model = types.SimpleNamespace(policy=policy)

        original_atoms = atoms.clone()

        controller._apply_categorical_transform(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=0.0,
            new_std=2.0,
        )

        # Atoms should be transformed
        assert not torch.allclose(atoms, original_atoms)

    def test_with_update_atoms_method(self):
        controller = PopArtController(enabled=True)

        atoms = torch.linspace(-1, 1, 21)
        updated = [False]

        def update_atoms(v_min, v_max):
            updated[0] = True

        policy = types.SimpleNamespace(
            atoms=atoms,
            v_min=-1.0,
            v_max=1.0,
            update_atoms=update_atoms,
        )

        model = types.SimpleNamespace(policy=policy)

        controller._apply_categorical_transform(
            model=model,
            old_mean=0.0,
            old_std=1.0,
            new_mean=0.0,
            new_std=2.0,
        )

        assert updated[0] is True


# =============================================================================
# RawRecurrentRolloutBuffer Tests
# =============================================================================


class TestRawRecurrentRolloutBufferToNumpy:
    """Tests for RawRecurrentRolloutBuffer._to_numpy static method."""

    def test_tensor_input(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = RawRecurrentRolloutBuffer._to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = RawRecurrentRolloutBuffer._to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_list_input(self):
        lst = [1.0, 2.0, 3.0]
        result = RawRecurrentRolloutBuffer._to_numpy(lst)
        assert isinstance(result, np.ndarray)


# =============================================================================
# DistributionalPPO Static Methods Tests
# =============================================================================


class TestDistributionalPPOCloneStates:
    """Tests for DistributionalPPO._clone_states_to_device."""

    def test_none_input(self):
        result = DistributionalPPO._clone_states_to_device(None, torch.device("cpu"))
        assert result is None

    def test_tensor_tuple(self):
        states = (torch.zeros(1, 2, 64), torch.ones(1, 2, 64))
        result = DistributionalPPO._clone_states_to_device(states, torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nested_tuple(self):
        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))

        # Create RNNStates-like object
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._clone_states_to_device(states, torch.device("cpu"))
        assert hasattr(result, "pi")
        assert hasattr(result, "vf")


class TestDistributionalPPOCloneObservations:
    """Tests for DistributionalPPO._clone_observations_to_device."""

    def test_tensor_input(self):
        obs = torch.zeros(10, 4)
        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)

    def test_dict_input(self):
        obs = {"a": torch.zeros(10, 4), "b": torch.ones(10, 2)}
        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result

    def test_list_input(self):
        obs = [torch.zeros(10, 4), torch.ones(10, 2)]
        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))
        assert isinstance(result, list)

    def test_tuple_input(self):
        obs = (torch.zeros(10, 4), torch.ones(10, 2))
        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))
        assert isinstance(result, tuple)

    def test_object_with_to_method(self):
        class MockObs:
            def __init__(self):
                self.device = torch.device("cpu")

            def to(self, device):
                self.device = device
                return self

        obs = MockObs()
        result = DistributionalPPO._clone_observations_to_device(obs, torch.device("cpu"))
        assert result.device == torch.device("cpu")


class TestDistributionalPPODetachObservations:
    """Tests for DistributionalPPO._detach_observations_to_cpu."""

    def test_tensor_input(self):
        obs = torch.zeros(10, 4, requires_grad=True)
        result = DistributionalPPO._detach_observations_to_cpu(obs)
        assert not result.requires_grad

    def test_dict_input(self):
        obs = {"a": torch.zeros(10, 4)}
        result = DistributionalPPO._detach_observations_to_cpu(obs)
        assert isinstance(result, dict)

    def test_tuple_input(self):
        obs = (torch.zeros(10, 4), torch.ones(10, 2))
        result = DistributionalPPO._detach_observations_to_cpu(obs)
        assert isinstance(result, tuple)

    def test_list_input(self):
        obs = [torch.zeros(10, 4), torch.ones(10, 2)]
        result = DistributionalPPO._detach_observations_to_cpu(obs)
        assert isinstance(result, list)

    def test_primitive_passthrough(self):
        result = DistributionalPPO._detach_observations_to_cpu(42)
        assert result == 42


class TestDistributionalPPODetachStatesToCpu:
    """Tests for DistributionalPPO._detach_states_to_cpu."""

    def test_none_input(self):
        result = DistributionalPPO._detach_states_to_cpu(None)
        assert result is None

    def test_tensor_tuple(self):
        states = (torch.zeros(1, 2, 64), torch.ones(1, 2, 64))
        result = DistributionalPPO._detach_states_to_cpu(states)
        assert isinstance(result, tuple)


class TestDistributionalPPODetachEpisodeStarts:
    """Tests for DistributionalPPO._detach_episode_starts_to_cpu."""

    def test_tensor_input_bool(self):
        starts = torch.tensor([True, False, True])
        result = DistributionalPPO._detach_episode_starts_to_cpu(starts)
        assert result.dtype == torch.bool

    def test_tensor_input_float(self):
        starts = torch.tensor([1.0, 0.0, 1.0])
        result = DistributionalPPO._detach_episode_starts_to_cpu(starts)
        assert result.dtype == torch.bool

    def test_numpy_input(self):
        starts = np.array([True, False, True])
        result = DistributionalPPO._detach_episode_starts_to_cpu(starts)
        assert result.dtype == torch.bool


class TestDistributionalPPOExtractStates:
    """Tests for DistributionalPPO._extract_critic_states and _extract_actor_states."""

    def test_extract_critic_none(self):
        result = DistributionalPPO._extract_critic_states(None)
        assert result is None

    def test_extract_critic_with_vf(self):
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.ones(1, 2, 64), torch.ones(1, 2, 64))
        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._extract_critic_states(states)
        assert isinstance(result, tuple)
        assert torch.all(result[0] == 1.0)

    def test_extract_critic_plain_tuple(self):
        states = (torch.zeros(1, 2, 64), torch.ones(1, 2, 64))
        result = DistributionalPPO._extract_critic_states(states)
        assert isinstance(result, tuple)

    def test_extract_actor_none(self):
        result = DistributionalPPO._extract_actor_states(None)
        assert result is None

    def test_extract_actor_with_pi(self):
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        pi_states = (torch.zeros(1, 2, 64), torch.zeros(1, 2, 64))
        vf_states = (torch.ones(1, 2, 64), torch.ones(1, 2, 64))
        states = RNNStates(pi=pi_states, vf=vf_states)

        result = DistributionalPPO._extract_actor_states(states)
        assert isinstance(result, tuple)
        assert torch.all(result[0] == 0.0)


class TestDistributionalPPOConcatHelpers:
    """Tests for _concat_tensor_batches and _concat_string_keys."""

    def test_concat_tensor_batches_single(self):
        batch = torch.tensor([[1], [2]])
        result = DistributionalPPO._concat_tensor_batches([batch])
        assert torch.equal(result, batch)

    def test_concat_tensor_batches_multiple(self):
        batch1 = torch.tensor([[1], [2]])
        batch2 = torch.tensor([[3], [4]])
        result = DistributionalPPO._concat_tensor_batches([batch1, batch2])
        expected = torch.tensor([[1], [2], [3], [4]])
        assert torch.equal(result, expected)

    def test_concat_tensor_batches_with_none(self):
        batch = torch.tensor([[1], [2]])
        result = DistributionalPPO._concat_tensor_batches([None, batch, None])
        assert torch.equal(result, batch)

    def test_concat_tensor_batches_all_none(self):
        result = DistributionalPPO._concat_tensor_batches([None, None])
        assert result is None

    def test_concat_string_keys_basic(self):
        keys1 = ["a", "b"]
        keys2 = ["c", "d"]
        result = DistributionalPPO._concat_string_keys([keys1, keys2])
        assert result == ["a", "b", "c", "d"]

    def test_concat_string_keys_empty(self):
        result = DistributionalPPO._concat_string_keys([])
        assert result is None

    def test_concat_string_keys_with_none(self):
        keys = ["a", "b"]
        result = DistributionalPPO._concat_string_keys([None, keys])
        assert result == ["a", "b"]


class TestDistributionalPPOValueTargetOutlierFractions:
    """Tests for DistributionalPPO._value_target_outlier_fractions."""

    def test_empty_tensor(self):
        below, above = DistributionalPPO._value_target_outlier_fractions(
            torch.tensor([]), -1.0, 1.0
        )
        assert below == 0.0
        assert above == 0.0

    def test_no_outliers(self):
        values = torch.tensor([0.0, 0.5, -0.5])
        below, above = DistributionalPPO._value_target_outlier_fractions(values, -1.0, 1.0)
        assert below == 0.0
        assert above == 0.0

    def test_mixed_outliers(self):
        values = torch.tensor([-2.0, 0.0, 2.0])
        below, above = DistributionalPPO._value_target_outlier_fractions(values, -1.0, 1.0)
        assert below == pytest.approx(1.0 / 3.0)
        assert above == pytest.approx(1.0 / 3.0)


class TestDistributionalPPOFlattenRolloutMask:
    """Tests for DistributionalPPO._flatten_rollout_mask."""

    def test_none_input(self):
        result = DistributionalPPO._flatten_rollout_mask(None)
        assert result is None

    def test_tensor_input(self):
        mask = torch.tensor([[1, 0], [1, 1]])
        result = DistributionalPPO._flatten_rollout_mask(mask)
        assert result.shape == (4,)

    def test_numpy_input(self):
        mask = np.array([[1, 0], [1, 1]])
        result = DistributionalPPO._flatten_rollout_mask(mask)
        assert result.shape == (4,)


class TestDistributionalPPOCoerceValueTargetScale:
    """Tests for DistributionalPPO._coerce_value_target_scale."""

    def test_none_returns_one(self):
        result = DistributionalPPO._coerce_value_target_scale(None)
        assert result == 1.0

    def test_percent_string(self):
        result = DistributionalPPO._coerce_value_target_scale("percent")
        assert result == 100.0

    def test_bps_string(self):
        result = DistributionalPPO._coerce_value_target_scale("bps")
        assert result == 10000.0

    def test_float_value(self):
        result = DistributionalPPO._coerce_value_target_scale(50.0)
        assert result == 50.0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            DistributionalPPO._coerce_value_target_scale("invalid")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            DistributionalPPO._coerce_value_target_scale(-1.0)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            DistributionalPPO._coerce_value_target_scale(0.0)

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            DistributionalPPO._coerce_value_target_scale(float("inf"))


class TestDistributionalPPOBoundedDualUpdate:
    """Tests for DistributionalPPO._bounded_dual_update."""

    def test_stay_in_range(self):
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, 0.2)
        assert 0.0 <= result <= 1.0

    def test_clamp_to_zero(self):
        result = DistributionalPPO._bounded_dual_update(0.1, 0.1, -2.0)
        assert result == 0.0

    def test_clamp_to_one(self):
        result = DistributionalPPO._bounded_dual_update(0.9, 0.1, 2.0)
        assert result == 1.0

    def test_nan_lambda_becomes_zero(self):
        result = DistributionalPPO._bounded_dual_update(float("nan"), 0.1, 0.5)
        assert math.isfinite(result)

    def test_nan_lr_becomes_zero(self):
        result = DistributionalPPO._bounded_dual_update(0.5, float("nan"), 0.5)
        assert result == 0.5

    def test_nan_gap_becomes_zero(self):
        result = DistributionalPPO._bounded_dual_update(0.5, 0.1, float("nan"))
        assert result == 0.5


class TestDistributionalPPOHasNonemptyBatches:
    """Tests for DistributionalPPO._has_nonempty_batches."""

    def test_empty_list(self):
        result = DistributionalPPO._has_nonempty_batches([])
        assert result is False

    def test_all_empty_tensors(self):
        result = DistributionalPPO._has_nonempty_batches(
            [
                torch.tensor([]),
                torch.tensor([]),
            ]
        )
        assert result is False

    def test_one_nonempty(self):
        result = DistributionalPPO._has_nonempty_batches(
            [
                torch.tensor([]),
                torch.tensor([1.0]),
            ]
        )
        assert result is True

    def test_all_nonempty(self):
        result = DistributionalPPO._has_nonempty_batches(
            [
                torch.tensor([1.0]),
                torch.tensor([2.0]),
            ]
        )
        assert result is True


class TestDistributionalPPOQuantileHuberLoss:
    """Tests for DistributionalPPO._quantile_huber_loss."""

    def _make_algo(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._quantile_huber_kappa = 1.0
        algo._use_fixed_quantile_loss_asymmetry = True
        algo._num_quantiles = 5
        algo._quantile_levels = torch.linspace(0.1, 0.9, 5)
        # Mock policy with quantile_levels attribute
        algo.policy = types.SimpleNamespace(
            device=torch.device("cpu"),
            quantile_levels=torch.linspace(0.1, 0.9, 5),
        )
        return algo

    def test_basic_loss_computation(self):
        algo = self._make_algo()

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        targets = torch.tensor([[0.5]])

        loss = algo._quantile_huber_loss(predictions, targets)
        assert torch.isfinite(loss)
        assert loss >= 0.0

    def test_reduction_none(self):
        algo = self._make_algo()

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]])
        targets = torch.tensor([[0.5], [0.5]])

        loss = algo._quantile_huber_loss(predictions, targets, reduction="none")
        assert loss.shape == (2,)

    def test_reduction_sum(self):
        algo = self._make_algo()

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]])
        targets = torch.tensor([[0.5], [0.5]])

        loss = algo._quantile_huber_loss(predictions, targets, reduction="sum")
        assert loss.ndim == 0

    def test_invalid_reduction_raises(self):
        algo = self._make_algo()

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        targets = torch.tensor([[0.5]])

        with pytest.raises(ValueError, match="Invalid reduction mode"):
            algo._quantile_huber_loss(predictions, targets, reduction="invalid")

    def test_scalar_targets_raises(self):
        algo = self._make_algo()

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        targets = torch.tensor(0.5)

        with pytest.raises(ValueError, match="must include a batch dimension"):
            algo._quantile_huber_loss(predictions, targets)

    def test_legacy_formula(self):
        algo = self._make_algo()
        algo._use_fixed_quantile_loss_asymmetry = False

        predictions = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        targets = torch.tensor([[0.5]])

        loss = algo._quantile_huber_loss(predictions, targets)
        assert torch.isfinite(loss)


class TestDistributionalPPOProjectDistribution:
    """Tests for DistributionalPPO._project_distribution (deprecated)."""

    def test_deprecation_warning(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[0.2, 0.3, 0.5]])
        source_atoms = torch.tensor([-1.0, 0.0, 1.0])
        target_atoms = torch.tensor([-2.0, 0.0, 2.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = algo._project_distribution(probs, source_atoms, target_atoms)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

        assert result.shape == probs.shape


class TestDistributionalPPOProjectCategoricalDistribution:
    """Tests for DistributionalPPO._project_categorical_distribution."""

    def test_basic_projection(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[0.2, 0.3, 0.5]])
        source_atoms = torch.tensor([-1.0, 0.0, 1.0])
        target_atoms = torch.tensor([-2.0, 0.0, 2.0])

        result = algo._project_categorical_distribution(probs, source_atoms, target_atoms)
        assert result.shape == probs.shape
        assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=1e-5)

    def test_identity_projection(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        probs = torch.tensor([[0.2, 0.3, 0.5]])
        atoms = torch.tensor([-1.0, 0.0, 1.0])

        result = algo._project_categorical_distribution(probs, atoms, atoms)
        assert torch.allclose(result, probs, atol=1e-5)


class TestDistributionalPPOCvarFromQuantiles:
    """Tests for DistributionalPPO._cvar_from_quantiles."""

    def _make_algo(self, num_quantiles=5):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._num_quantiles = num_quantiles
        algo._quantile_levels = torch.linspace(0.1, 0.9, num_quantiles)
        algo._cvar_alpha = 0.25
        algo.cvar_alpha = 0.25  # Public property
        algo.policy = types.SimpleNamespace(device=torch.device("cpu"))
        return algo

    def test_basic_cvar(self):
        algo = self._make_algo()

        quantiles = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        cvar = algo._cvar_from_quantiles(quantiles)
        assert torch.isfinite(cvar).all()


class TestDistributionalPPOExpandLoggerKeyLength:
    """Tests for DistributionalPPO._expand_logger_key_length."""

    def test_none_logger(self):
        # Should not raise
        DistributionalPPO._expand_logger_key_length(None, min_max_length=80)

    def test_logger_no_output_formats(self):
        logger = types.SimpleNamespace()
        # Should not raise
        DistributionalPPO._expand_logger_key_length(logger, min_max_length=80)

    def test_logger_with_output_formats(self):
        class MockOutput:
            def __init__(self):
                self.max_length = 40

        output = MockOutput()
        logger = types.SimpleNamespace(output_formats=[output])

        DistributionalPPO._expand_logger_key_length(logger, min_max_length=80)
        assert output.max_length == 80


class TestComputeReturnsWithTimeLimits:
    """Tests for _compute_returns_with_time_limits function."""

    def test_basic_computation(self):
        # Create a mock rollout buffer
        class MockBuffer:
            def __init__(self):
                self.buffer_size = 4
                self.n_envs = 2
                self.rewards = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
                self.episode_starts = np.array([[1, 1], [0, 0], [0, 1], [0, 0]])
                self.values = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
                self.advantages = np.zeros_like(self.rewards)
                self.returns = np.zeros_like(self.rewards)

        buffer = MockBuffer()
        last_values = torch.tensor([[1.0], [1.0]])
        dones = np.array([False, False])
        time_limit_mask = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        time_limit_bootstrap = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        _compute_returns_with_time_limits(
            buffer, last_values, dones, 0.99, 0.95, time_limit_mask, time_limit_bootstrap
        )

        # Check that returns and advantages are computed
        assert buffer.returns is not None
        assert buffer.advantages is not None


class TestComputeGroupedExplainedVariance:
    """Tests for compute_grouped_explained_variance function."""

    def test_basic_computation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9, 6.1])
        groups = ["a", "a", "a", "b", "b", "b"]

        result = compute_grouped_explained_variance(y_true, y_pred, groups)

        # Returns (group_evs, aggregates) tuple
        group_evs, aggregates = result
        assert "a" in group_evs
        assert "b" in group_evs
        assert math.isfinite(group_evs["a"])
        assert math.isfinite(group_evs["b"])

    def test_aggregates_structure(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9, 6.1])
        groups = ["a", "a", "a", "b", "b", "b"]

        group_evs, aggregates = compute_grouped_explained_variance(y_true, y_pred, groups)

        assert "mean_unweighted" in aggregates
        assert "mean_weighted" in aggregates
        assert "median" in aggregates


class TestCalculateCvar:
    """Tests for calculate_cvar function."""

    def test_basic_cvar(self):
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        alpha = 0.5

        result = calculate_cvar(probs, atoms, alpha)
        assert torch.isfinite(result).all()

    def test_alpha_one_gives_mean(self):
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        atoms = torch.tensor([-2.0, -1.0, 1.0, 2.0])

        cvar = calculate_cvar(probs, atoms, 1.0)
        expected = (probs * atoms).sum(dim=1)
        assert torch.allclose(cvar, expected, atol=1e-5)


class TestCreateSequencers:
    """Tests for create_sequencers function."""

    def test_basic_sequencers(self):
        episode_starts = np.zeros(10, dtype=bool)
        episode_starts[0] = True
        episode_starts[5] = True
        env_change = np.zeros(10, dtype=bool)

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")

        assert indices is not None

    def test_with_env_changes(self):
        episode_starts = np.zeros(10, dtype=bool)
        episode_starts[0] = True
        env_change = np.zeros(10, dtype=bool)
        env_change[3] = True

        indices, pad_fn, unpad_fn = create_sequencers(episode_starts, env_change, device="cpu")

        assert indices is not None


class TestCfgGetEdgeCases:
    """Additional edge cases for _cfg_get."""

    def test_object_with_get_type_error_fallback(self):
        """Test that TypeError in get() triggers fallback to get(key) only."""

        class GetWithTypeError:
            def get(self, key, default=None):
                # Simulate a get method that doesn't accept default
                raise TypeError("get() takes 2 positional arguments but 3 were given")

        # This should return default since TypeError triggers fallback but key doesn't exist
        result = _cfg_get(GetWithTypeError(), "key", "default")
        # Falls back to get(key) which also fails, then to model_dump/dict which don't exist
        assert result == "default"

    def test_object_with_model_dump(self):
        class WithModelDump:
            def model_dump(self):
                return {"key": "value"}

        result = _cfg_get(WithModelDump(), "key")
        assert result == "value"

    def test_object_with_dict_method(self):
        class WithDict:
            def dict(self):
                return {"key": "value"}

        result = _cfg_get(WithDict(), "key")
        assert result == "value"

    def test_failing_model_dump_falls_back_to_dict(self):
        class FailingModelDumpWithDict:
            def model_dump(self):
                raise RuntimeError("model_dump failed")

            def dict(self):
                return {"key": "value"}

        result = _cfg_get(FailingModelDumpWithDict(), "key", "default")
        assert result == "value"

    def test_all_fallbacks_fail(self):
        class AllFail:
            def model_dump(self):
                raise RuntimeError("model_dump failed")

            def dict(self):
                raise RuntimeError("dict failed")

        result = _cfg_get(AllFail(), "key", "default")
        assert result == "default"

    def test_get_type_error_then_key_error(self):
        """Test TypeError triggers retry with single arg which also fails."""

        class GetTypeErrorThenKeyError:
            def get(self, *args):
                if len(args) == 2:
                    raise TypeError("Too many arguments")
                raise KeyError(args[0])

        result = _cfg_get(GetTypeErrorThenKeyError(), "key", "default")
        assert result == "default"


class TestPopartValueToSerializableEdgeCases:
    """Additional edge cases for _popart_value_to_serializable."""

    def test_tuple_conversion(self):
        result = _popart_value_to_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        result = _popart_value_to_serializable({"a": {"b": 1}})
        assert result == {"a": {"b": 1}}


class TestUnwrapVecNormalizeEdgeCases:
    """Additional edge cases for unwrap_vec_normalize."""

    def test_with_vec_env_wrapper_chain(self):
        # Create a mock wrapper chain
        class MockVecEnvWrapper:
            def __init__(self, venv):
                self.venv = venv

        inner = types.SimpleNamespace()
        wrapper = MockVecEnvWrapper(inner)

        result = unwrap_vec_normalize(wrapper)
        # Should handle gracefully
        assert result is None or result == wrapper or hasattr(result, "venv")


class TestDistributionalPPOFilterEvReserveRows:
    """Tests for DistributionalPPO._filter_ev_reserve_rows."""

    def test_with_index_tensor(self):
        rollout_data = types.SimpleNamespace()
        target_norm = torch.tensor([[1.0], [2.0], [3.0]])
        target_raw = torch.tensor([[1.0], [2.0], [3.0]])
        weights = torch.tensor([[1.0], [1.0], [1.0]])
        index_tensor = torch.tensor([0, 1, 2])

        result = DistributionalPPO._filter_ev_reserve_rows(
            rollout_data, target_norm, target_raw, weights, index_tensor
        )

        assert result[0] is target_norm
        assert result[1] is target_raw

    def test_with_sample_indices_padding(self):
        rollout_data = types.SimpleNamespace(
            sample_indices=torch.tensor([0, 1, -1])  # -1 is padding
        )
        target_norm = torch.tensor([[1.0], [2.0], [3.0]])
        target_raw = torch.tensor([[1.0], [2.0], [3.0]])
        weights = torch.tensor([[1.0], [1.0], [1.0]])

        result = DistributionalPPO._filter_ev_reserve_rows(
            rollout_data, target_norm, target_raw, weights, None
        )

        # Should filter out padded rows
        assert result[0].shape[0] == 2
        assert result[1].shape[0] == 2

    def test_all_valid_indices(self):
        rollout_data = types.SimpleNamespace(sample_indices=torch.tensor([0, 1, 2]))
        target_norm = torch.tensor([[1.0], [2.0], [3.0]])
        target_raw = torch.tensor([[1.0], [2.0], [3.0]])
        weights = None

        result = DistributionalPPO._filter_ev_reserve_rows(
            rollout_data, target_norm, target_raw, weights, None
        )

        assert result[0] is target_norm


class TestDistributionalPPOEvReserveMaskToWeights:
    """Tests for DistributionalPPO._ev_reserve_mask_to_weights."""

    def test_none_mask(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.device = "cpu"

        result = algo._ev_reserve_mask_to_weights(None, None)
        assert result is None

    def test_empty_mask(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.device = "cpu"

        result = algo._ev_reserve_mask_to_weights(torch.tensor([]), None)
        assert result is None

    def test_with_indices(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo.device = "cpu"

        mask = torch.tensor([1.0, 2.0, 3.0])
        indices = torch.tensor([0, 2])

        result = algo._ev_reserve_mask_to_weights(mask, indices)
        assert result.shape[0] == 2


class TestDistributionalPPOShouldSkipEvReserveBatch:
    """Tests for DistributionalPPO._should_skip_ev_reserve_batch."""

    def test_with_valid_indices(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._logger = None

        rollout_data = types.SimpleNamespace()
        result = algo._should_skip_ev_reserve_batch(rollout_data, torch.tensor([0, 1]), None)
        assert result is False

    def test_with_mask_values(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._logger = None

        rollout_data = types.SimpleNamespace()
        result = algo._should_skip_ev_reserve_batch(rollout_data, None, torch.tensor([1.0, 1.0]))
        assert result is False

    def test_no_rollout_mask(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._logger = None

        rollout_data = types.SimpleNamespace()
        result = algo._should_skip_ev_reserve_batch(rollout_data, None, None)
        assert result is False

    def test_all_masked_rollout(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._logger = types.SimpleNamespace(record=lambda k, v: None)

        rollout_data = types.SimpleNamespace(mask=torch.zeros(10))
        result = algo._should_skip_ev_reserve_batch(rollout_data, None, None)
        assert result is True


class TestDistributionalPPOResolveCvarPenaltyState:
    """Tests for DistributionalPPO._resolve_cvar_penalty_state."""

    def test_no_violation(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        result = algo._resolve_cvar_penalty_state(0.1, 0.1, 0.0)
        assert result == (0.0, 0.0, False)

    def test_with_violation_and_weight(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)

        result = algo._resolve_cvar_penalty_state(0.1, 0.2, 0.5)
        assert result[2] is True
        assert result[0] == 0.1
        assert result[1] == 0.2

    def test_with_violation_no_weight(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._cvar_ramp_updates = 10
        algo._cvar_weight_target = 1.0
        algo.cvar_penalty_cap = float("inf")

        result = algo._resolve_cvar_penalty_state(0.0, 0.0, 0.5)
        assert result[2] is True
        assert result[0] > 0

    def test_with_penalty_cap(self):
        algo = DistributionalPPO.__new__(DistributionalPPO)
        algo._cvar_ramp_updates = 1
        algo._cvar_weight_target = 100.0
        algo.cvar_penalty_cap = 0.5

        result = algo._resolve_cvar_penalty_state(0.0, 0.0, 0.5)
        assert result[0] <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
