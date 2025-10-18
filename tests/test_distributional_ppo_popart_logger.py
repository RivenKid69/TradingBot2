import types

import pytest

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed

import distributional_ppo as distributional_ppo_module
from distributional_ppo import DistributionalPPO


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, object] = {}

    def record(self, key: str, value, **_: object) -> None:
        self.records[key] = value


def test_popart_logger_rebound_after_setup_learn(monkeypatch: pytest.MonkeyPatch) -> None:
    old_logger = _CaptureLogger()
    new_logger = _CaptureLogger()

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = old_logger
    algo._logger = old_logger
    algo._popart_holdout_loader = None
    algo._popart_config_logs = {}
    algo._use_quantile_value = False
    algo.normalize_returns = False
    algo._to_raw_returns = lambda tensor: tensor
    algo._value_clip_limit_unscaled = None
    algo.policy = types.SimpleNamespace()
    algo.device = None
    algo.gamma = 0.99
    algo.gae_lambda = 0.97
    algo.n_envs = 1
    algo.n_steps = 1
    algo.observation_space = None
    algo.action_space = None

    cfg = {
        "enabled": True,
        "mode": "live",
        "ema_beta": 0.9,
        "min_samples": 8,
        "warmup_updates": 0,
        "max_rel_step": 0.5,
        "ev_floor": 0.2,
        "ret_std_band": (0.1, 2.0),
        "gate_patience": 1,
        "replay_path": "",
        "replay_seed": 123,
        "replay_batch_size": 16,
    }

    algo._initialise_popart_controller(cfg)
    controller = algo._popart_controller
    assert controller is not None

    def _fake_parent_setup_learn(
        self,
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="run",
        progress_bar=False,
    ):
        self._logger = new_logger
        self.logger = new_logger
        return total_timesteps, callback

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "_setup_learn",
        _fake_parent_setup_learn,
        raising=False,
    )

    def _fake_parent_learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        return self

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "learn",
        _fake_parent_learn,
        raising=False,
    )

    result = DistributionalPPO.learn(algo, total_timesteps=4)
    assert result is algo
    assert controller._logger is new_logger

    assert "config/popart/enabled" in new_logger.records
    assert "config/popart/mode" in new_logger.records

    controller._apply_categorical_transform = lambda *_, **__: None
    controller._apply_quantile_transform = lambda *_, **__: None
    controller._last_holdout_eval = None

    controller.apply_live_update(
        model=algo,
        old_mean=0.0,
        old_std=1.0,
        new_mean=0.0,
        new_std=1.0,
    )

    assert new_logger.records.get("popart/apply_count") == pytest.approx(1.0)
