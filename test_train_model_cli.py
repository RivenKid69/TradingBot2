import sys
import types

import pandas as pd
import pytest


def _install_sb3_stub():
    if "sb3_contrib" in sys.modules:
        return

    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.__path__ = []  # mark as package
    sys.modules["sb3_contrib"] = sb3_contrib

    common = types.ModuleType("sb3_contrib.common")
    common.__path__ = []
    sys.modules["sb3_contrib.common"] = common
    sb3_contrib.common = common  # type: ignore[attr-defined]

    recurrent = types.ModuleType("sb3_contrib.common.recurrent")
    recurrent.__path__ = []
    sys.modules["sb3_contrib.common.recurrent"] = recurrent
    common.recurrent = recurrent  # type: ignore[attr-defined]

    policies = types.ModuleType("sb3_contrib.common.recurrent.policies")
    class _DummyPolicy:  # pragma: no cover - simple placeholder
        pass

    policies.RecurrentActorCriticPolicy = _DummyPolicy
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies
    recurrent.policies = policies  # type: ignore[attr-defined]

    type_aliases = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")
    type_aliases.RNNStates = object  # pragma: no cover - placeholder type
    sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases
    recurrent.type_aliases = type_aliases  # type: ignore[attr-defined]


_install_sb3_stub()

import train_model_multi_patch as train_script  # noqa: E402  (import after stub)


class _DummyTrial:
    def __init__(self):
        self.suggested: list[str] = []
        self.number = 0

    def suggest_float(self, name, *_args, **_kwargs):
        self.suggested.append(name)
        # return mid value when bounds provided
        if _args:
            low = _args[0]
            high = _args[1] if len(_args) > 1 else low
            return (low + high) / 2 if high is not None else low
        return 0.0

    def suggest_categorical(self, name, choices, **_kwargs):
        self.suggested.append(name)
        return choices[0]

    def suggest_int(self, name, low, high=None, **_kwargs):
        self.suggested.append(name)
        return low if high is None else low


def test_dataset_split_none_skips_offline_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path):
    seasonality = tmp_path / "seasonality.json"
    seasonality.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: train\n", encoding="utf-8")

    called = False

    def fake_resolve(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("resolve_split_bundle should not be called when dataset-split is none")

    def stop_after_config(_path):
        raise RuntimeError("stop after config load")

    monkeypatch.setattr(train_script, "resolve_split_bundle", fake_resolve)
    monkeypatch.setattr(train_script, "load_config", stop_after_config)

    argv = [
        "train_model_multi_patch.py",
        "--config",
        str(config_path),
        "--dataset-split",
        "none",
        "--liquidity-seasonality",
        str(seasonality),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="stop after config load"):
        train_script.main()

    assert called is False


def test_config_params_override_optuna(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.15,
                "ent_coef": 0.001,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 1024,
                "batch_size": 256,
            }
        ),
        algo=types.SimpleNamespace(
            actions={},
            action_wrapper=types.SimpleNamespace(bins_vol=2),
        ),
    )

    trial = _DummyTrial()

    class _StopWatchdogVecEnv:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("stop after params")

    monkeypatch.setattr(train_script, "WatchdogVecEnv", _StopWatchdogVecEnv)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(RuntimeError, match="stop after params"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=1024,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
        )

    overridden_keys = {
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "n_steps",
        "batch_size",
    }

    assert overridden_keys.isdisjoint(trial.suggested)


def test_invalid_batch_size_config_raises(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.15,
                "ent_coef": 0.001,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 1024,
                "batch_size": 257,
            }
        ),
        algo=types.SimpleNamespace(
            actions={},
            action_wrapper=types.SimpleNamespace(bins_vol=2),
        ),
    )

    trial = _DummyTrial()

    class _FailWatchdogVecEnv:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover - should never be called
            raise AssertionError("WatchdogVecEnv should not be constructed for invalid batch_size")

    monkeypatch.setattr(train_script, "WatchdogVecEnv", _FailWatchdogVecEnv)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(ValueError, match="batch_size.*divide"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=1024,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
        )


def test_scheduler_disabled_uses_constant_lr(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            params={
                "learning_rate": 3.0e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.15,
                "ent_coef": 0.001,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 1,
                "v_range_ema_alpha": 0.005,
                "cql_alpha": 0.0,
                "cql_beta": 5.0,
                "cvar_alpha": 0.05,
                "cvar_weight": 0.0,
                "cvar_cap": 0.5,
                "num_atoms": 5,
                "v_min": -1.0,
                "v_max": 1.0,
                "hidden_dim": 32,
                "trade_frequency_penalty": 0.0,
                "turnover_penalty_coef": 0.0,
            }
        ),
        algo=types.SimpleNamespace(
            actions={},
            action_wrapper=types.SimpleNamespace(bins_vol=2),
        ),
        optimization={"scheduler": {"enabled": False}},
    )

    trial = _DummyTrial()

    class _StubTradingEnv:
        def __init__(self, *_args, **_kwargs):
            self.action_space = None

    class _StubWatchdogVecEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

    class _StubVecMonitor:
        def __init__(self, env):
            self.env = env

    class _StubDummyVecEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

    class _StubVecNormalize:
        def __init__(self, env, **_kwargs):
            self.env = env
            self.training = True
            self.norm_reward = _kwargs.get("norm_reward")
            self.clip_reward = _kwargs.get("clip_reward")

        def save(self, _path):
            return None

        @classmethod
        def load(cls, _path, env):
            inst = cls(env, norm_obs=False, norm_reward=False, clip_reward=None, gamma=1.0)
            inst.training = False
            inst.norm_reward = False
            inst.clip_reward = None
            return inst

    captured_policy_kwargs: dict[str, object] = {}

    class _StubAlgo:
        def __init__(self, *_, policy_kwargs=None, **__):
            captured_policy_kwargs.clear()
            if policy_kwargs:
                captured_policy_kwargs.update(policy_kwargs)
            raise RuntimeError("stop before training")

    def _fail_one_cycle_lr(*_args, **_kwargs):
        raise AssertionError("scheduler should be disabled")

    monkeypatch.setattr(train_script, "TradingEnv", _StubTradingEnv)
    monkeypatch.setattr(train_script, "_wrap_action_space_if_needed", lambda env, **_: env)
    monkeypatch.setattr(train_script, "WatchdogVecEnv", _StubWatchdogVecEnv)
    monkeypatch.setattr(train_script, "VecMonitor", _StubVecMonitor)
    monkeypatch.setattr(train_script, "DummyVecEnv", _StubDummyVecEnv)
    monkeypatch.setattr(train_script, "VecNormalize", _StubVecNormalize)
    monkeypatch.setattr(train_script, "save_sidecar_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_script, "check_model_compat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_script, "_get_distributional_ppo", lambda: _StubAlgo)
    monkeypatch.setattr(train_script, "OneCycleLR", _fail_one_cycle_lr)

    df = pd.DataFrame({"price": [1.0], "ts_ms": [0]})

    with pytest.raises(RuntimeError, match="stop before training"):
        train_script.objective(
            trial,
            cfg,
            total_timesteps=32,
            train_data_by_token={"BTCUSDT": df},
            train_obs_by_token={},
            val_data_by_token={"BTCUSDT": df},
            val_obs_by_token={},
            test_data_by_token={},
            test_obs_by_token={},
            norm_stats={},
            sim_config={},
            timing_env_kwargs={},
            leak_guard_kwargs={},
            trials_dir=tmp_path,
            tensorboard_log_dir=None,
        )

    assert "optimizer_scheduler_fn" not in captured_policy_kwargs
