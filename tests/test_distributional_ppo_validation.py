"""
Tests for validation error paths in distributional_ppo.py.
Targets the 40+ raise ValueError and 26+ raise RuntimeError branches.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from distributional_ppo import DistributionalPPO, safe_explained_variance


np.random.seed(42)
torch.manual_seed(42)


def _make_env(max_steps: int = 8, n_envs: int = 1) -> DummyVecEnv:
    """Create minimal test environment."""
    def _env_fn():
        import gymnasium

        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
                self.observation_space = spaces.Box(-10.0, 10.0, (4,), np.float32)
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.random.randn(4).astype(np.float32) * 0.1, {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                obs = np.random.randn(4).astype(np.float32) * 0.1
                reward = np.random.randn() * 0.1
                return obs, reward, done, False, {}

        return _Env()

    return DummyVecEnv([_env_fn for _ in range(n_envs)])


# =============================================================================
# __init__ Validation Errors
# =============================================================================

class TestInitValidationErrors:
    """Test validation errors in __init__."""

    def test_clip_range_vf_negative_raises(self):
        """Negative clip_range_vf raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                clip_range_vf=-0.5,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_clip_range_vf_zero_raises(self):
        """Zero clip_range_vf raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                clip_range_vf=0.0,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_clip_range_vf_inf_raises(self):
        """Infinite clip_range_vf raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                clip_range_vf=float('inf'),
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_clip_range_vf_nan_raises(self):
        """NaN clip_range_vf raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                clip_range_vf=float('nan'),
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_vf_clip_threshold_ev_inf_raises(self):
        """Infinite vf_clip_threshold_ev raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                vf_clip_threshold_ev=float('inf'),
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_vf_clip_threshold_ev_too_high_raises(self):
        """vf_clip_threshold_ev > 1 raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                vf_clip_threshold_ev=1.5,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_vf_clip_threshold_ev_too_low_raises(self):
        """vf_clip_threshold_ev < -1 raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                vf_clip_threshold_ev=-1.5,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_distributional_vf_clip_mode_invalid_raises(self):
        """Invalid distributional_vf_clip_mode raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                distributional_vf_clip_mode="invalid_mode",
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_distributional_vf_clip_variance_factor_low_raises(self):
        """distributional_vf_clip_variance_factor < 1 raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                distributional_vf_clip_variance_factor=0.5,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_distributional_vf_clip_variance_factor_inf_raises(self):
        """Infinite distributional_vf_clip_variance_factor raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                distributional_vf_clip_variance_factor=float('inf'),
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_value_target_scale_invalid_string_raises(self):
        """Invalid value_target_scale string raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                value_target_scale="invalid_scale",
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_value_target_scale_negative_raises(self):
        """Negative value_target_scale raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                value_target_scale=-100.0,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_value_target_scale_fixed_negative_raises(self):
        """Negative value_target_scale_fixed raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                value_target_scale_fixed=-50.0,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_optimizer_lr_min_negative_raises(self):
        """Negative optimizer_lr_min raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                optimizer_lr_min=-1e-5,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_optimizer_lr_max_negative_raises(self):
        """Negative optimizer_lr_max raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                optimizer_lr_max=-1e-3,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_scheduler_min_lr_negative_raises(self):
        """Negative scheduler_min_lr raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                scheduler_min_lr=-1e-6,
                device="cpu",
                verbose=0,
            )
        env.close()


# =============================================================================
# Additional Validation Errors
# =============================================================================

class TestAdditionalValidationErrors:
    """Additional validation error tests."""

    def test_optimizer_class_invalid_string_raises(self):
        """Invalid optimizer class string raises ValueError."""
        env = _make_env()
        with pytest.raises(ValueError):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                optimizer_class="NonExistentOptimizer",
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_clip_range_negative_raises(self):
        """Negative clip_range raises ValueError."""
        env = _make_env()
        with pytest.raises((ValueError, AssertionError)):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                clip_range=-0.1,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_n_steps_negative_raises(self):
        """n_steps=-1 raises ValueError."""
        env = _make_env()
        with pytest.raises((ValueError, AssertionError)):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=-1,
                device="cpu",
                verbose=0,
            )
        env.close()

    def test_batch_size_zero_raises(self):
        """batch_size=0 raises ValueError."""
        env = _make_env()
        with pytest.raises((ValueError, AssertionError)):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                batch_size=0,
                device="cpu",
                verbose=0,
            )
        env.close()


# =============================================================================
# safe_explained_variance edge cases
# =============================================================================

class TestSafeExplainedVarianceEdgeCases:
    """Test safe_explained_variance edge cases."""

    def test_safe_ev_empty_arrays(self):
        """Test with empty arrays."""
        y_pred = np.array([])
        y_true = np.array([])
        result = safe_explained_variance(y_pred, y_true)
        # Should handle gracefully
        assert np.isnan(result) or isinstance(result, (int, float))

    def test_safe_ev_single_element(self):
        """Test with single element arrays."""
        y_pred = np.array([1.0])
        y_true = np.array([1.0])
        result = safe_explained_variance(y_pred, y_true)
        assert np.isfinite(result) or np.isnan(result)

    def test_safe_ev_identical_arrays(self):
        """Test with identical arrays."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        result = safe_explained_variance(y_pred, y_true)
        # Perfect prediction = EV of 1.0
        assert result == 1.0 or np.isclose(result, 1.0)

    def test_safe_ev_with_nan(self):
        """Test with NaN values."""
        y_pred = np.array([1.0, np.nan, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        result = safe_explained_variance(y_pred, y_true)
        # Should handle NaN gracefully
        assert isinstance(result, (int, float))

    def test_safe_ev_with_inf(self):
        """Test with infinite values."""
        y_pred = np.array([1.0, np.inf, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])
        result = safe_explained_variance(y_pred, y_true)
        # Should handle inf gracefully
        assert isinstance(result, (int, float))

    def test_safe_ev_zero_variance_true(self):
        """Test when true values have zero variance."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 1.0, 1.0])  # All same
        result = safe_explained_variance(y_pred, y_true)
        # Should handle zero variance gracefully
        assert isinstance(result, (int, float))

    def test_safe_ev_very_large_values(self):
        """Test with very large values."""
        y_pred = np.array([1e10, 2e10, 3e10])
        y_true = np.array([1.1e10, 2.1e10, 2.9e10])
        result = safe_explained_variance(y_pred, y_true)
        assert np.isfinite(result)

    def test_safe_ev_very_small_values(self):
        """Test with very small values."""
        y_pred = np.array([1e-10, 2e-10, 3e-10])
        y_true = np.array([1.1e-10, 2.1e-10, 2.9e-10])
        result = safe_explained_variance(y_pred, y_true)
        assert np.isfinite(result) or np.isnan(result)


# =============================================================================
# Valid configurations that should NOT raise
# =============================================================================

class TestValidConfigurations:
    """Test valid configurations that should not raise."""

    def test_valid_vf_clip_threshold_ev_range(self):
        """Test valid vf_clip_threshold_ev in valid range."""
        env = _make_env()
        for threshold in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                vf_clip_threshold_ev=threshold,
                device="cpu",
                verbose=0,
            )
            assert model is not None
        env.close()

    def test_valid_distributional_vf_clip_modes(self):
        """Test valid distributional_vf_clip_mode values."""
        env = _make_env()
        for mode in ["disable", "mean_only", "mean_and_variance", "per_quantile", None]:
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                distributional_vf_clip_mode=mode,
                device="cpu",
                verbose=0,
            )
            assert model is not None
        env.close()

    def test_valid_optimizer_classes(self):
        """Test valid optimizer class strings."""
        env = _make_env()
        for opt_class in ["Adam", "AdamW", "SGD"]:
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                optimizer_class=opt_class,
                device="cpu",
                verbose=0,
            )
            assert model is not None
        env.close()

    def test_valid_value_target_scale_strings(self):
        """Test valid value_target_scale strings."""
        env = _make_env()
        for scale in ["percent", "bps"]:
            model = DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                value_target_scale=scale,
                device="cpu",
                verbose=0,
            )
            assert model is not None
        env.close()


# =============================================================================
# Edge case numerical values
# =============================================================================

class TestEdgeCaseNumericalValues:
    """Test edge case numerical values."""

    def test_very_small_learning_rate(self):
        """Test with very small learning rate."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            learning_rate=1e-10,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=8, progress_bar=False)
        env.close()

    def test_very_large_learning_rate(self):
        """Test with very large learning rate."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            learning_rate=1.0,
            device="cpu",
            verbose=0,
        )
        # May produce warnings but should not crash
        model.learn(total_timesteps=8, progress_bar=False)
        env.close()

    def test_gamma_zero_raises_during_train(self):
        """Test with gamma=0 raises RuntimeError during train."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            gamma=0.0,
            device="cpu",
            verbose=0,
        )
        # gamma=0 raises during train(), not __init__
        with pytest.raises(RuntimeError, match="Invalid discount factor"):
            model.learn(total_timesteps=8, progress_bar=False)
        env.close()

    def test_gamma_one(self):
        """Test with gamma=1."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            gamma=1.0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=8, progress_bar=False)
        env.close()

    def test_very_small_clip_range_vf(self):
        """Test with very small clip_range_vf."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            clip_range_vf=0.001,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=8, progress_bar=False)
        env.close()

    def test_very_large_clip_range_vf(self):
        """Test with very large clip_range_vf."""
        env = _make_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            clip_range_vf=10.0,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=8, progress_bar=False)
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
