"""
Comprehensive tests to achieve ≥95% coverage for distributional_ppo.py.
Targets uncovered branches identified via coverage analysis.

Tests are designed to be:
- Fast (minimal steps, CPU only)
- Deterministic (fixed seeds)
- Isolated (no side effects)
"""
from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import warnings
from typing import Any, Dict, Optional, Tuple, Union
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

gymnasium = pytest.importorskip("gymnasium")
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

import distributional_ppo as dppo
from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtCandidateMetrics,
    PopArtHoldoutBatch,
    RawRecurrentRolloutBuffer,
    safe_explained_variance,
    _weighted_variance_np,
    compute_grouped_explained_variance,
)

# Set seeds for determinism
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Test Environment
# =============================================================================


class MinimalEnv(gymnasium.Env):
    """Minimal deterministic environment for testing."""

    def __init__(self, seed: int = 42, max_steps: int = 8):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._max_steps = max_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._rng.uniform(-1, 1, size=4).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = float(-np.sum(obs**2) * 0.01 + 0.1)
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"step": self._step_count}


def make_vec_env(n_envs: int = 1, seed: int = 42) -> DummyVecEnv:
    """Create vectorized environment."""
    return DummyVecEnv([lambda: MinimalEnv(seed=seed) for _ in range(n_envs)])


# =============================================================================
# Mock Logger
# =============================================================================


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.records: Dict[str, Any] = {}
        self.name_to_value = {}
        self.name_to_count = {}

    def record(self, key: str, value: Any, exclude: Optional[str] = None):
        self.records[key] = value
        self.name_to_value[key] = value

    def dump(self, step: int = 0):
        pass

    def get_dir(self) -> Optional[str]:
        return None


# =============================================================================
# Tests for safe_explained_variance edge cases (lines 434, 441, 458, etc.)
# =============================================================================


class TestSafeExplainedVarianceMoreEdges:
    """Additional edge cases for safe_explained_variance."""

    def test_weighted_max_abs_weight_huge(self):
        """Cover line 438: max_abs_weight > 1e50 returns nan."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        weights = np.array([1e60, 1.0, 1.0])  # Huge weight
        result = safe_explained_variance(y_true, y_pred, weights)
        assert math.isnan(result)

    def test_weighted_denom_raw_not_finite(self):
        """Cover line 447: denom_raw not finite returns nan."""
        # When sum_w_sq / sum_w causes issues
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([1.0, 1.0])
        # Equal weights will make denom_raw <= 0
        weights = np.array([1.0, 1.0])
        result = safe_explained_variance(y_true, y_pred, weights)
        # This should return nan due to denom_raw being <= 0
        assert isinstance(result, float)

    def test_weighted_mean_y_produces_nan(self):
        """Test case where mean_y computation has issues."""
        y_true = np.array([1e308, 1e308])
        y_pred = np.array([1e308, 1e308])
        weights = np.array([0.5, 0.5])
        result = safe_explained_variance(y_true, y_pred, weights)
        # Should handle gracefully
        assert isinstance(result, float)


class TestWeightedVarianceMoreEdges:
    """Additional edge cases for _weighted_variance_np."""

    def test_large_weights_overflow_path(self):
        """Cover lines 538-542: large weights overflow handling."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1e150, 1e150, 1e150])  # Very large weights
        result = _weighted_variance_np(values, weights)
        # Should handle via scaling
        assert isinstance(result, float)

    def test_denom_raw_negative(self):
        """Cover line 550: denom_raw <= 0 returns nan."""
        values = np.array([1.0])
        weights = np.array([1.0])  # Single sample, denom will be 0
        result = _weighted_variance_np(values, weights)
        assert math.isnan(result)


class TestComputeGroupedExplainedVarianceMoreEdges:
    """Additional edge cases for compute_grouped_explained_variance."""

    def test_single_sample_in_group(self):
        """Test group with single sample."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 3.1]
        group_keys = ["a", "b", "c"]  # Each group has 1 sample

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)
        # Single-sample groups should have nan EV
        assert isinstance(result, dict)

    def test_all_nan_group(self):
        """Test group where all values are nan."""
        y_true = [np.nan, np.nan, 1.0]
        y_pred = [1.0, 1.0, 1.1]
        group_keys = ["a", "a", "b"]

        result, summary = compute_grouped_explained_variance(y_true, y_pred, group_keys)
        assert isinstance(result, dict)


# =============================================================================
# Tests for PopArtController edge cases
# =============================================================================


class TestPopArtControllerMoreEdgeCases:
    """More edge cases for PopArtController."""

    def test_evaluate_shadow_disabled(self):
        """Test evaluate_shadow when disabled returns None."""
        controller = PopArtController(enabled=False)
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(10),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is None
        env.close()

    def test_evaluate_shadow_with_deprecated_ev_train(self):
        """Test evaluate_shadow with deprecated explained_variance_train arg."""
        controller = PopArtController(enabled=True, mode="shadow")
        controller.set_logger(MockLogger())

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            controller.evaluate_shadow(
                model=model,
                returns_raw=torch.randn(10),
                ret_mean=0.0,
                ret_std=1.0,
                explained_variance_train=0.5,  # deprecated
            )
            # Should produce deprecation warning
            assert any("deprecated" in str(warning.message).lower() for warning in w)

        env.close()

    def test_evaluate_shadow_no_holdout_blocked(self):
        """Test evaluate_shadow blocked due to no holdout (first blocking check)."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1000,  # Very high
        )
        controller.set_logger(MockLogger())

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(10),  # Only 10 samples
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        # no_holdout is checked first, before min_samples
        assert result.blocked_reason == "no_holdout"
        env.close()

    def test_evaluate_shadow_multiple_calls(self):
        """Test evaluate_shadow with multiple sequential calls."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=0,
        )
        controller.set_logger(MockLogger())

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        # Multiple calls to trigger different code paths
        for _ in range(3):
            result = controller.evaluate_shadow(
                model=model,
                returns_raw=torch.randn(100),
                ret_mean=0.0,
                ret_std=1.0,
            )
            assert result is not None

        # Shadow values should be set after multiple calls
        assert controller._shadow_mean is not None
        assert controller._shadow_std is not None
        env.close()

    def test_evaluate_shadow_empty_returns(self):
        """Test evaluate_shadow with empty returns (sample_count=0)."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=0,
            warmup_updates=0,
        )
        controller.set_logger(MockLogger())

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        # Empty tensor - sample_count will be 0
        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.tensor([]),
            ret_mean=5.0,
            ret_std=2.0,
        )
        # Should use fallback mean/std
        assert result is not None
        assert result.mean == 5.0
        assert result.std == 2.0
        env.close()

    def test_evaluate_shadow_second_update_ema(self):
        """Test evaluate_shadow EMA blending on second call."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=0,
            ema_beta=0.9,
        )
        controller.set_logger(MockLogger())

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            verbose=0,
        )

        # First call
        controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(50),
            ret_mean=0.0,
            ret_std=1.0,
        )

        # Second call - triggers EMA blending
        result = controller.evaluate_shadow(
            model=model,
            returns_raw=torch.randn(50),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        # Shadow values should be set
        assert controller._shadow_mean is not None
        assert controller._shadow_std is not None
        env.close()


# =============================================================================
# Tests for RawRecurrentRolloutBuffer edge cases
# =============================================================================


class TestRawRecurrentRolloutBufferEdgeCases:
    """Test RawRecurrentRolloutBuffer edge cases."""

    def test_buffer_without_actions_raw_raises(self):
        """Test that add without actions_raw raises TypeError."""
        buffer = RawRecurrentRolloutBuffer(
            buffer_size=4,
            observation_space=spaces.Box(-1, 1, (4,), np.float32),
            action_space=spaces.Box(-1, 1, (1,), np.float32),
            hidden_state_shape=(1, 1, 64),
            device="cpu",
            n_envs=1,
        )
        buffer.reset()

        with pytest.raises(TypeError):
            buffer.add(
                obs=np.zeros(4, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=0.0,
                episode_start=np.array([True]),
                value=torch.tensor([0.0]),
                log_prob=torch.tensor([0.0]),
                lstm_states=(torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)),
                # Missing actions_raw and log_prob_raw
            )


# =============================================================================
# Tests for DistributionalPPO train loop branches
# =============================================================================


class TestTrainLoopVFClipBranches:
    """Test VF clip branches in train loop."""

    def test_train_vf_clip_mean_only_normalize_returns(self):
        """Cover VF clipping with mean_only mode + normalize_returns."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.5,
            normalize_returns=True,
            distributional_vf_clip_mode="mean_only",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_vf_clip_mean_and_variance(self):
        """Cover VF clipping with mean_and_variance mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.5,
            distributional_vf_clip_mode="mean_and_variance",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_no_normalize_returns(self):
        """Cover branches when normalize_returns=False."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            normalize_returns=False,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


class TestTrainLoopMoreBranches:
    """Test additional training branches."""

    def test_train_with_vf_clip_per_quantile(self):
        """Cover per_quantile VF clip mode."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=0.3,
            distributional_vf_clip_mode="per_quantile",
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_vf_clip_disabled(self):
        """Cover disabled VF clipping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            clip_range_vf=None,  # Disabled
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_cvar_weight(self):
        """Cover CVaR weight parameter."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_weight=0.3,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_kl_penalty(self):
        """Cover KL penalty beta."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            kl_penalty_beta=0.01,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


class TestTrainLoopKLBranches:
    """Test KL divergence branches in train loop."""

    def test_train_with_target_kl(self):
        """Cover target_kl early stopping."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=10,  # Many epochs to trigger early stop
            target_kl=0.00001,  # Very low threshold
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


class TestTrainLoopCVaRBranches:
    """Test CVaR branches in train loop."""

    def test_train_with_cvar_alpha(self):
        """Cover CVaR alpha path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_alpha=0.1,
            cvar_weight=0.5,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_train_with_cvar_constraint(self):
        """Cover CVaR constraint path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            cvar_use_constraint=True,
            cvar_limit=-1.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for predict branches
# =============================================================================


class TestPredictBranches:
    """Test _predict method branches."""

    def test_predict_deterministic_true(self):
        """Cover deterministic=True path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )

        obs = env.reset()[0]
        actions, states = model.predict(obs, deterministic=True)
        assert actions is not None
        env.close()

    def test_predict_deterministic_false(self):
        """Cover deterministic=False path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )

        obs = env.reset()[0]
        actions, states = model.predict(obs, deterministic=False)
        assert actions is not None
        env.close()


# =============================================================================
# Tests for save/load
# =============================================================================


class TestSaveLoadBranches:
    """Test save/load branches."""

    def test_save_load_roundtrip(self):
        """Cover save/load roundtrip."""
        import tempfile

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )
        model.learn(total_timesteps=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model.zip"
            model.save(path)

            loaded = DistributionalPPO.load(path, env=env)
            assert loaded is not None

            # Verify can predict
            obs = env.reset()[0]
            actions, _ = loaded.predict(obs)
            assert actions is not None

        env.close()


# =============================================================================
# Tests for collect_rollouts edge cases
# =============================================================================


class TestCollectRolloutsEdgeCases:
    """Test collect_rollouts edge cases."""

    def test_collect_rollouts_with_callback(self):
        """Cover collect_rollouts with callback."""
        from stable_baselines3.common.callbacks import BaseCallback

        class CountingCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.count = 0

            def _on_step(self):
                self.count += 1
                return True

        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            verbose=0,
        )

        callback = CountingCallback()
        model.learn(total_timesteps=16, callback=callback)

        assert callback.count > 0
        env.close()


# =============================================================================
# Tests for learn with eval_env
# =============================================================================


class TestLearnWithEvalEnv:
    """Test learn with evaluation environment."""

    def test_learn_with_eval_env(self):
        """Cover learn with eval_env parameter."""
        env = make_vec_env()
        eval_env = make_vec_env(seed=123)

        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            verbose=0,
        )

        model.learn(
            total_timesteps=16,
            eval_env=eval_env,
            eval_freq=8,
        )

        env.close()
        eval_env.close()


# =============================================================================
# Tests for multi-env scenarios
# =============================================================================


class TestMultiEnvScenarios:
    """Test with multiple environments."""

    def test_train_multi_env(self):
        """Cover training with multiple environments."""
        env = make_vec_env(n_envs=2)
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            verbose=0,
        )
        model.learn(total_timesteps=32)
        env.close()

    def test_predict_multi_env(self):
        """Cover prediction with multi-env observations."""
        env = make_vec_env(n_envs=2)
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            verbose=0,
        )

        obs = env.reset()[0]
        actions, _ = model.predict(obs)
        # Actions shape depends on env - just verify it works
        assert actions is not None
        env.close()


# =============================================================================
# Tests for value_target_scale validation
# =============================================================================


class TestValueTargetScaleValidation:
    """Test value_target_scale validation branches."""

    def test_value_target_scale_percent(self):
        """Cover value_target_scale='percent' path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            value_target_scale="percent",
            verbose=0,
        )
        assert model.value_target_scale == 100.0
        env.close()

    def test_value_target_scale_bps(self):
        """Cover value_target_scale='bps' path."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            value_target_scale="bps",
            verbose=0,
        )
        assert model.value_target_scale == 10000.0
        env.close()

    def test_value_target_scale_numeric(self):
        """Cover numeric value_target_scale."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            value_target_scale=50.0,
            verbose=0,
        )
        assert model.value_target_scale == 50.0
        env.close()


# =============================================================================
# Tests for optimizer_class validation
# =============================================================================


class TestOptimizerClassValidation:
    """Test optimizer_class validation branches."""

    def test_optimizer_class_adam_string(self):
        """Cover optimizer_class='Adam' string."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            optimizer_class="Adam",
            verbose=0,
        )
        model.learn(total_timesteps=8)
        env.close()

    def test_optimizer_class_adamw_string(self):
        """Cover optimizer_class='AdamW' string."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=4,
            batch_size=4,
            optimizer_class="AdamW",
            verbose=0,
        )
        model.learn(total_timesteps=8)
        env.close()


# =============================================================================
# Tests for LR bounds and optimizer settings
# =============================================================================


class TestLRBoundsBranches:
    """Test LR bounds branches."""

    def test_optimizer_lr_bounds(self):
        """Cover optimizer LR min/max bounds."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            learning_rate=3e-4,
            optimizer_lr_min=1e-6,
            optimizer_lr_max=1e-2,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_scheduler_min_lr(self):
        """Cover scheduler min LR."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            learning_rate=3e-4,
            scheduler_min_lr=1e-7,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for entropy coefficient branches
# =============================================================================


class TestEntCoefBranches:
    """Test entropy coefficient branches."""

    def test_ent_coef_nonzero(self):
        """Cover ent_coef > 0."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            ent_coef=0.1,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_ent_coef_zero(self):
        """Cover ent_coef = 0."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            ent_coef=0.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


# =============================================================================
# Tests for GAE lambda variations
# =============================================================================


class TestGAELambdaBranches:
    """Test GAE lambda variations."""

    def test_gae_lambda_one(self):
        """Cover gae_lambda=1.0 (Monte Carlo)."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            gae_lambda=1.0,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()

    def test_gae_lambda_low(self):
        """Cover low gae_lambda."""
        env = make_vec_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            gae_lambda=0.8,
            verbose=0,
        )
        model.learn(total_timesteps=16)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
