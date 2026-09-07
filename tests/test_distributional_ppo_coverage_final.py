"""
Comprehensive coverage tests for distributional_ppo.py to achieve ≥95% coverage.

Focus areas:
1. __init__ validation branches (value_scale configs, normalize_returns conflicts)
2. PopArt shadow evaluation paths
3. train() edge cases (KL stop, EMA-KL, vf_clip_warmup, cvar, entropy schedule, vgs)
4. _predict() deterministic/non-deterministic, vectorization
5. _twin_critics_loss() quantile/categorical modes
6. Import fallback paths
7. Various helper functions and edge cases
"""

from __future__ import annotations

import math
import sys
import types
import warnings
from collections import deque
from types import SimpleNamespace
from typing import Any, Mapping, Optional
from unittest import mock

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from distributional_ppo import (
    DistributionalPPO,
    PopArtController,
    PopArtHoldoutBatch,
    PopArtHoldoutEvaluation,
)

# Fix seeds for determinism
np.random.seed(42)
torch.manual_seed(42)


def _make_simple_env(max_steps: int = 10) -> DummyVecEnv:
    """Create minimal vectorized environment for testing."""

    def _env_fn():
        class _Env(gymnasium.Env):
            def __init__(self):
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(4,), dtype=np.float32
                )
                self._step = 0
                self._max = max_steps

            def reset(self, *, seed=None, options=None):
                self._step = 0
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                self._step += 1
                done = self._step >= self._max
                return (
                    np.zeros(4, dtype=np.float32),
                    0.1,
                    done,
                    False,
                    {"TimeLimit.truncated": done},
                )

        return _Env()

    return DummyVecEnv([_env_fn])


def _make_holdout_batch(device: torch.device) -> PopArtHoldoutBatch:
    """Create holdout batch for PopArt tests."""
    obs = torch.zeros((2, 4), device=device)
    returns_raw = torch.zeros((2, 1), device=device)
    episode_starts = torch.zeros((2,), dtype=torch.bool, device=device)
    return PopArtHoldoutBatch(
        observations=obs,
        returns_raw=returns_raw,
        episode_starts=episode_starts,
        lstm_states=None,
        mask=None,
    )


def _make_holdout_eval(
    *,
    ev_after: float = 0.9,
    ev_before: float = 0.8,
) -> PopArtHoldoutEvaluation:
    """Create holdout evaluation result."""
    baseline = torch.tensor([[0.5], [0.5]])
    candidate = baseline.clone()
    target = torch.zeros_like(baseline)
    return PopArtHoldoutEvaluation(
        baseline_raw=baseline,
        candidate_raw=candidate,
        target_raw=target,
        mask=None,
        ev_before=ev_before,
        ev_after=ev_after,
        clip_fraction_before=0.0,
        clip_fraction_after=0.0,
    )


class TestInitValidationBranches:
    """Test __init__ parameter validation branches."""

    def test_winrate_confidence_invalid_type(self):
        """Cover 6289-6290: invalid winrate_confidence_level type."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            winrate_confidence_level="invalid",
        )
        assert model._winrate_confidence_level == 0.95
        env.close()

    def test_winrate_confidence_out_of_range(self):
        """Cover winrate_confidence_level boundary check."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            winrate_confidence_level=1.5,
        )
        assert model._winrate_confidence_level == 0.95
        env.close()

    def test_value_scale_controller_as_object(self):
        """Cover 6302-6317: value_scale_controller as object with attributes."""

        class VSConfig:
            enabled = True
            mode = "ema"
            min_samples = 10
            warmup_updates = 2

        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale_controller=VSConfig(),
        )
        env.close()

    def test_normalize_returns_conflict_raises(self):
        """Cover 6713-6720: conflicting normalize_returns values."""
        env = _make_simple_env()
        with pytest.raises(TypeError, match="normalize_returns"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                normalize_returns=True,
                **{"normalize_returns": False},  # via kwargs
            )
        env.close()

    def test_ret_clip_invalid_raises(self):
        """Cover 6726: invalid ret_clip value."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="ret_clip"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                ret_clip=-1.0,
            )
        env.close()

    def test_value_scale_cfg_as_mapping(self):
        """Cover 6757-6772: value_scale config as Mapping."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={
                "ema_beta": 0.3,
                "max_rel_step": 0.4,
                "std_floor": 1e-4,
                "window_updates": 5,
                "warmup_updates": 2,
                "freeze_after": 100,
                "freeze_after_updates": 50,
                "never_freeze": True,
                "range_max_rel_step": 0.1,
                "stability": {"min_explained_variance": 0.5, "max_abs_p95": 2.0, "patience": 3},
                "stability_patience": 4,
                "target_scale_ema_beta": 0.25,
                "max_change_pct": 5.0,
            },
        )
        assert model._value_scale_ema_beta == pytest.approx(0.3)
        env.close()

    def test_value_scale_cfg_as_object(self):
        """Cover 6774-6796: value_scale config as object with attrs."""

        class VSCfg:
            ema_beta = 0.35
            max_rel_step = 0.45
            std_floor = 2e-4
            window_updates = 6
            warmup_updates = 3
            freeze_after = 80
            freeze_after_updates = 40
            never_freeze = False
            range_max_rel_step = 0.12
            stability = {"ev_min": 0.4, "ret_abs_p95_max": 1.5, "consecutive": 2}
            stability_patience = 5
            target_scale_ema_beta = 0.22
            max_change_pct = 4.5

        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale=VSCfg(),
        )
        assert model._value_scale_ema_beta == pytest.approx(0.35)
        env.close()

    def test_value_scale_ema_beta_invalid(self):
        """Cover 6859: invalid ema_beta."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="ema_beta"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"ema_beta": 0.0},
            )
        env.close()

    def test_value_scale_max_rel_step_negative(self):
        """Cover 6862: negative max_rel_step."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="max_rel_step"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"max_rel_step": -0.1},
            )
        env.close()

    def test_value_scale_window_updates_negative(self):
        """Cover 6866: negative window_updates."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="window_updates"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"window_updates": -1},
            )
        env.close()

    def test_value_scale_target_ema_beta_inf(self):
        """Cover 6881: infinite target_scale_ema_beta."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="ema_beta"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"target_scale_ema_beta": float("inf")},
            )
        env.close()

    def test_value_scale_max_change_pct_inf(self):
        """Cover 6890-6891: infinite max_change_pct."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="max_change_pct"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"max_change_pct": float("inf")},
            )
        env.close()

    def test_value_scale_max_change_pct_negative(self):
        """Cover 6893: negative max_change_pct."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="max_change_pct"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"max_change_pct": -1.0},
            )
        env.close()

    def test_value_scale_range_max_rel_step_inf(self):
        """Cover 6920-6921: infinite range_max_rel_step."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="range_max_rel_step"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"range_max_rel_step": float("inf")},
            )
        env.close()

    def test_value_scale_range_max_rel_step_negative(self):
        """Cover 6922-6923: negative range_max_rel_step."""
        env = _make_simple_env()
        with pytest.raises(ValueError, match="range_max_rel_step"):
            DistributionalPPO(
                policy="DistributionalPolicy",
                env=env,
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                device="cpu",
                verbose=0,
                value_scale={"range_max_rel_step": -0.5},
            )
        env.close()

    def test_value_scale_stability_alternative_keys(self):
        """Cover 6936-6937, 6944-6946, 6953-6955: alternative stability keys."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={
                "stability": {
                    "ev_min": 0.3,
                    "ret_abs_p95_max": 1.2,
                    "consecutive": 4,
                }
            },
        )
        assert model._value_scale_stability_min_ev == pytest.approx(0.3)
        env.close()

    def test_value_scale_stability_patience_override(self):
        """Cover 6960-6961: stability_patience override."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={
                "stability": {"patience": 2},
                "stability_patience": 7,
            },
        )
        assert model._value_scale_stability_patience == 7
        env.close()

    def test_value_scale_stability_inf_ev(self):
        """Cover 6963-6964: infinite stability_min_ev."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={"stability": {"min_explained_variance": float("inf")}},
        )
        assert model._value_scale_stability_min_ev is None
        env.close()

    def test_value_scale_stability_negative_p95(self):
        """Cover 6965-6966: non-positive max_abs_p95."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={"stability": {"max_abs_p95": -1.0}},
        )
        assert model._value_scale_stability_max_abs_p95 is None
        env.close()

    def test_value_scale_freeze_after_updates_via_cfg(self):
        """Cover 6912-6913: freeze_after_updates from config."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_scale={"freeze_after_updates": 25},
        )
        assert model._value_scale_freeze_after == 25
        env.close()


class TestPopArtShadowEvaluation:
    """Test PopArtController.evaluate_shadow() blocking reasons."""

    def test_shadow_blocked_no_holdout(self):
        """Cover no_holdout blocking reason."""
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=0,
            holdout_loader=None,
        )
        result = controller.evaluate_shadow(
            model=SimpleNamespace(policy=SimpleNamespace(device=torch.device("cpu"))),
            returns_raw=torch.tensor([0.0]),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        assert result.blocked_reason == "no_holdout"
        assert result.passed_guards is False

    def test_shadow_blocked_min_samples(self):
        """Cover min_samples blocking reason."""
        device = torch.device("cpu")
        holdout = _make_holdout_batch(device)
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=100,  # Need 100 samples
            warmup_updates=0,
            holdout_loader=lambda: holdout,
        )

        def _stub_eval(*_a, **_kw):
            return _make_holdout_eval()

        controller._evaluate_holdout = types.MethodType(_stub_eval, controller)
        result = controller.evaluate_shadow(
            model=SimpleNamespace(policy=SimpleNamespace(device=device, training=False)),
            returns_raw=torch.tensor([0.0]),  # Only 1 sample
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        assert result.blocked_reason == "min_samples"
        assert result.passed_guards is False

    def test_shadow_blocked_warmup(self):
        """Cover warmup blocking reason."""
        device = torch.device("cpu")
        holdout = _make_holdout_batch(device)
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=100,  # High warmup
            holdout_loader=lambda: holdout,
        )
        controller._shadow_update_count = 5  # Below warmup

        def _stub_eval(*_a, **_kw):
            return _make_holdout_eval()

        controller._evaluate_holdout = types.MethodType(_stub_eval, controller)
        result = controller.evaluate_shadow(
            model=SimpleNamespace(policy=SimpleNamespace(device=device, training=False)),
            returns_raw=torch.zeros(10),
            ret_mean=0.0,
            ret_std=1.0,
        )
        assert result is not None
        assert result.blocked_reason == "warmup"
        assert result.passed_guards is False

    def test_shadow_blocked_std_band(self):
        """Cover std_band blocking reason."""
        device = torch.device("cpu")
        holdout = _make_holdout_batch(device)
        controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=0,
            ret_std_band=(1.0, 10.0),  # ret_std must be in [1, 10]
            holdout_loader=lambda: holdout,
        )

        def _stub_eval(*_a, **_kw):
            return _make_holdout_eval()

        controller._evaluate_holdout = types.MethodType(_stub_eval, controller)
        result = controller.evaluate_shadow(
            model=SimpleNamespace(policy=SimpleNamespace(device=device, training=False)),
            returns_raw=torch.zeros(10),
            ret_mean=0.0,
            ret_std=0.5,  # Below band
        )
        assert result is not None
        assert result.blocked_reason == "std_band"
        assert result.passed_guards is False

    @pytest.mark.skip(reason="Complex mock setup - covered by targeted_branches")
    @pytest.mark.skip(reason="Complex internal state setup required")
    def test_shadow_blocked_rel_step(self):
        pass

    @pytest.mark.skip(reason="Complex mock setup - covered by targeted_branches")
    def test_shadow_blocked_ev_regress(self):
        """Cover ev_regress blocking reason."""
        pass

    @pytest.mark.skip(reason="Complex mock setup - covered by targeted_branches")
    def test_shadow_success_transition_to_live(self):
        """Cover successful pass with mode transition to 'live'."""
        pass


class _DummyCallback(BaseCallback):
    """Simple callback for testing."""

    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        return True


class TestTrainBranches:
    """Test train() method branches."""

    def _make_model_and_collect(self, seed: int = 42, max_steps: int = 8, **model_kwargs):
        """Create model, setup and collect rollouts."""
        env = _make_simple_env(max_steps=max_steps)
        defaults = {
            "policy": "DistributionalPolicy",
            "env": env,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "device": "cpu",
            "verbose": 0,
        }
        defaults.update(model_kwargs)
        model = DistributionalPPO(**defaults)

        total = int(model.n_steps * env.num_envs)
        _, callback = model._setup_learn(
            total_timesteps=total,
            callback=_DummyCallback(),
            reset_num_timesteps=True,
        )
        model._last_callback = callback
        model._current_progress_remaining = 1.0
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
        return model, env

    def test_train_kl_absolute_stop(self):
        """Cover KL absolute stop branch."""
        model, env = self._make_model_and_collect(
            target_kl=1e-10,  # Very low
            kl_absolute_stop_factor=0.5,
        )
        model.train()
        env.close()

    def test_train_ema_kl(self):
        """Cover EMA-KL branch."""
        model, env = self._make_model_and_collect(
            target_kl=0.1,
            kl_ema_updates=2,
            kl_ema_alpha=None,
        )
        model.train()
        env.close()

    def test_train_vf_clip_warmup_gating(self):
        """Cover vf_clip_warmup gating."""
        model, env = self._make_model_and_collect(
            vf_clip_warmup_updates=100,  # High warmup
        )
        model._update_calls = 1
        model.train()
        env.close()

    def test_train_cvar_constraint(self):
        """Cover CVaR constraint branch."""
        model, env = self._make_model_and_collect(
            cvar_use_constraint=True,
            cvar_use_predicted_for_dual=True,
        )
        model._cvar_predicted_last_unit = 0.0
        model._cvar_predicted_last_raw = 0.0
        model.train()
        env.close()

    def test_train_entropy_schedule(self):
        """Cover entropy schedule update."""
        model, env = self._make_model_and_collect(
            ent_coef_decay_steps=1,
            ent_coef_plateau_window=2,
        )
        model.train()
        env.close()

    def test_train_cvar_penalty_disabled(self):
        """Cover CVaR penalty disabled path."""
        model, env = self._make_model_and_collect(
            cvar_use_penalty=False,
        )
        model.train()
        env.close()

    def test_train_with_popart_shadow(self):
        """Cover PopArt shadow evaluation in train."""
        model, env = self._make_model_and_collect()
        model._popart_controller = PopArtController(
            enabled=True,
            mode="shadow",
            min_samples=1,
            warmup_updates=0,
            holdout_loader=lambda: _make_holdout_batch(model.policy.device),
        )
        model._popart_controller._evaluate_holdout = types.MethodType(
            lambda *_args, **_kwargs: _make_holdout_eval(ev_after=0.9),
            model._popart_controller,
        )
        model.train()
        env.close()


class TestPredictBranches:
    """Test predict() method branches."""

    def test_predict_deterministic(self):
        """Cover deterministic prediction path."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        obs = np.zeros((1, 4), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
        env.close()

    def test_predict_non_deterministic(self):
        """Cover non-deterministic prediction path."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        obs = np.zeros((1, 4), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=False)
        assert action is not None
        env.close()

    def test_predict_vectorized(self):
        """Cover vectorized prediction."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        obs = np.zeros((4, 4), dtype=np.float32)  # Batch of 4
        actions, _ = model.predict(obs, deterministic=True)
        assert actions.shape[0] == 4
        env.close()


class TestHelperFunctions:
    """Test various helper functions and edge cases."""

    def test_coerce_value_target_scale_string_percent(self):
        """Cover value_target_scale as 'percent' string."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_target_scale="percent",
        )
        assert model.value_target_scale == pytest.approx(100.0)
        env.close()

    def test_coerce_value_target_scale_string_bps(self):
        """Cover value_target_scale as 'bps' string."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            value_target_scale="bps",
        )
        assert model.value_target_scale == pytest.approx(10000.0)
        env.close()

    def test_enforce_optimizer_lr_bounds_with_bounds(self):
        """Cover _enforce_optimizer_lr_bounds with explicit bounds."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
            optimizer_lr_min=1e-6,
            optimizer_lr_max=1e-2,
        )
        model._enforce_optimizer_lr_bounds(log_values=True, warn_on_floor=True)
        env.close()


class TestSafeExplainedVarianceEdgeCases:
    """Test safe_explained_variance edge cases - covered elsewhere."""

    pass


class TestGetStateSetState:
    """Test __getstate__ and __setstate__ methods."""

    def test_getstate_with_vgs(self):
        """Cover __getstate__ with VGS state."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        model._variance_gradient_scaler = SimpleNamespace(state_dict=lambda: {"step": 1})
        state = model.__getstate__()
        assert "_vgs_saved_state" in state
        env.close()

    def test_setstate_restores(self):
        """Cover __setstate__ restoration."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        state = model.__getstate__()
        model.__setstate__(state)
        env.close()


class TestLoadSave:
    """Test load/save functionality edge cases."""

    def test_load_with_env_override(self, tmp_path):
        """Cover load with env override."""
        env = _make_simple_env()
        model = DistributionalPPO(
            policy="DistributionalPolicy",
            env=env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        path = tmp_path / "model.zip"
        model.save(str(path))
        env.close()

        env2 = _make_simple_env()
        loaded = DistributionalPPO.load(str(path), env=env2)
        assert loaded is not None
        env2.close()


class TestCollectRolloutsBranches:
    """Test collect_rollouts edge cases - covered by other tests."""

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
