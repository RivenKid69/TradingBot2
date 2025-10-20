import math
import types

import pytest
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed

import distributional_ppo as distributional_ppo_module
from distributional_ppo import DistributionalPPO


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float, **_: object) -> None:
        self.records.append((key, float(value)))


class _PolicyStub:
    def __init__(self) -> None:
        self.uses_quantile_value_head = False
        self.quantile_huber_kappa = 1.0
        self.device = torch.device("cpu")

    def named_parameters(self):  # pragma: no cover - simple stub returning empty iterable
        return []


def test_clip_range_vf_none_disables_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_super_init(self, *args, **kwargs):
        logger = getattr(self, "logger", _CaptureLogger())
        self.logger = logger
        self._logger = logger
        self.policy = _PolicyStub()
        self.device = torch.device("cpu")
        self.n_steps = 1
        self.n_envs = 1
        self.gae_lambda = 0.97
        self.n_epochs = 1
        self.lr_schedule = lambda _: 0.001
        self.normalize_returns = True
        self._value_scale_updates_enabled = True
        self.ent_coef = 0.01
        self.action_space = types.SimpleNamespace()
        self.observation_space = types.SimpleNamespace()

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "__init__",
        _fake_super_init,
        raising=False,
    )
    monkeypatch.setattr(DistributionalPPO, "_rebuild_scheduler_if_needed", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_ensure_score_action_space", lambda self: None)
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_loss_head_weights",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DistributionalPPO,
        "_configure_gradient_accumulation",
        lambda self, **kwargs: None,
    )

    algo = DistributionalPPO.__new__(DistributionalPPO)
    logger = _CaptureLogger()
    algo.logger = logger
    algo._logger = logger

    DistributionalPPO.__init__(
        algo,
        policy=_PolicyStub(),
        env=object(),
        clip_range_vf=None,
        value_scale={
            "ema_beta": 0.9,
            "max_rel_step": 0.15,
            "std_floor": 0.003,
            "window_updates": 16,
            "warmup_updates": 6,
        },
    )

    assert algo.clip_range_vf is None
    clip_range_logs = [value for key, value in logger.records if key == "config/clip_range_vf"]
    assert clip_range_logs, "clip_range_vf should be logged"
    assert math.isnan(clip_range_logs[-1])
