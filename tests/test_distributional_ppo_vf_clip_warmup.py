import types
from typing import Any

import pytest
import torch

import test_distributional_ppo_raw_outliers  # noqa: F401  # ensure RL stubs are installed

import distributional_ppo as distributional_ppo_module
from distributional_ppo import DistributionalPPO


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: dict[str, list[Any]] = {}

    def record(self, key: str, value: Any, **_: object) -> None:
        if isinstance(value, (list, tuple)):
            converted = [float(v) for v in value]
            self.records.setdefault(key, []).append(converted)
            return
        if isinstance(value, (int, float)):
            self.records.setdefault(key, []).append(float(value))
            return
        self.records.setdefault(key, []).append(value)


class _GaussianDistribution:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.distribution = torch.distributions.Normal(mean, std)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1, keepdim=True)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)


class _PolicyStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 5, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(1))
        self.uses_quantile_value_head = False
        self.quantile_huber_kappa = 1.0
        self.v_min = -2.0
        self.v_max = 2.0
        self.num_atoms = 5
        self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.num_atoms)
        self.device = torch.device("cpu")
        self.last_value_logits = None
        self._last_value_logits = None
        with torch.no_grad():
            self.actor.weight.fill_(0.1)
            self.value_head.weight.copy_(
                torch.tensor([[0.5], [0.25], [0.0], [-0.25], [-0.5]], dtype=torch.float32)
            )

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: object,
        episode_starts: torch.Tensor,
        *,
        actions_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del lstm_states, episode_starts, actions_raw
        obs_fp32 = obs.to(dtype=torch.float32)
        mean = self.actor(obs_fp32)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value_logits = self.value_head(obs_fp32)
        self.last_value_logits = value_logits
        self._last_value_logits = value_logits
        return value_logits.mean(dim=1, keepdim=True), log_prob, entropy

    def get_distribution(
        self,
        obs: torch.Tensor,
        actor_states: object,
        episode_starts: torch.Tensor,
    ) -> _GaussianDistribution:
        del actor_states, episode_starts
        obs_fp32 = obs.to(dtype=torch.float32)
        mean = self.actor(obs_fp32)
        std = torch.exp(self.log_std)
        return _GaussianDistribution(mean, std)

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: object,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        del lstm_states, episode_starts
        logits = self.value_head(obs.to(dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)
        return (probs * self.atoms.to(device=probs.device, dtype=probs.dtype)).sum(dim=1, keepdim=True)

    def _log_prob_raw_only(self, dist: _GaussianDistribution, actions: torch.Tensor) -> torch.Tensor:
        raw = dist.distribution.log_prob(actions)
        if raw.ndim > 1:
            raw = raw.sum(dim=-1, keepdim=True)
        return raw

    def update_atoms(self, v_min: float, v_max: float) -> None:
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atoms = torch.linspace(self.v_min, self.v_max, steps=self.num_atoms)


class _DummyRolloutBuffer:
    def __init__(self) -> None:
        base_obs = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
        self._observations = base_obs
        self._actions = torch.zeros_like(base_obs)
        self._advantages = torch.tensor([[0.2], [-0.15], [0.25], [-0.05]], dtype=torch.float32)
        self._returns = torch.tensor([[0.4], [0.3], [0.2], [0.1]], dtype=torch.float32)
        self._old_values = torch.zeros_like(self._returns)
        self._old_log_prob = torch.zeros_like(self._returns)
        self._sample_indices = torch.arange(self._returns.shape[0], dtype=torch.int64)
        self.rewards = self._returns.clone()
        self.returns = self._returns.clone()
        self.buffer_size = int(self._returns.shape[0])
        self.n_envs = 1

    def _make_sample(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            observations=self._observations.clone(),
            actions=self._actions.clone(),
            actions_raw=self._actions.clone(),
            lstm_states=None,
            episode_starts=torch.zeros(self._returns.shape[0], dtype=torch.bool),
            advantages=self._advantages.clone(),
            returns=self._returns.clone(),
            old_values=self._old_values.clone(),
            old_log_prob=self._old_log_prob.clone(),
            old_log_prob_raw=self._old_log_prob.clone(),
            mask=None,
            sample_indices=self._sample_indices.clone(),
        )

    def get(self, batch_size: int):  # noqa: D401 - simple generator interface
        del batch_size
        yield self._make_sample()


def test_vf_clip_warmup_allows_ev_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _CaptureLogger()
    policy = _PolicyStub()

    def _fake_super_init(self, policy: object, env: object, *args: object, **kwargs: object) -> None:
        del env, args, kwargs
        self.logger = logger
        self._logger = logger
        self.policy = policy  # type: ignore[assignment]
        self.device = torch.device("cpu")
        self.policy.to(self.device)
        self.n_steps = 4
        self.n_envs = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.n_epochs = 1
        self.batch_size = 4
        self.lr_schedule = lambda _: 0.001
        self.normalize_returns = False
        self._value_scale_updates_enabled = True
        self.ent_coef = 0.01
        self.action_space = types.SimpleNamespace()
        self.observation_space = types.SimpleNamespace()
        self._grad_accumulation_steps = 1
        self._microbatch_size = 4
        self._critic_ce_normalizer = 1.0
        self._current_progress_remaining = 1.0
        self.max_grad_norm = 0.5
        self.clip_range = 0.2
        self._n_updates = 0
        self._global_update_step = 0
        self._global_step = 0
        self.target_kl = None
        self.kl_early_stop = False
        self.kl_exceed_stop_fraction = 0.0
        self._kl_consec_minibatches = 0

    monkeypatch.setattr(
        distributional_ppo_module.RecurrentPPO,
        "__init__",
        _fake_super_init,
        raising=False,
    )
    monkeypatch.setattr(DistributionalPPO, "_rebuild_scheduler_if_needed", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_ensure_score_action_space", lambda self: None)
    monkeypatch.setattr(DistributionalPPO, "_configure_loss_head_weights", lambda self, *a, **k: None)
    monkeypatch.setattr(DistributionalPPO, "_configure_gradient_accumulation", lambda self, **_: None)

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.logger = logger
    algo._logger = logger

    DistributionalPPO.__init__(
        algo,
        policy=policy,
        env=object(),
        clip_range_vf=0.05,
        vf_clip_warmup_updates=2,
        value_scale_max_rel_step=0.1,
    )

    algo.rollout_buffer = _DummyRolloutBuffer()

    algo.train()
    first_ev = logger.records.get("train/ev/on_train_batch", [])[-1]
    first_warmup = logger.records.get("train/vf_clip_warmup_active", [])[-1]

    algo.train()
    second_ev = logger.records.get("train/ev/on_train_batch", [])[-1]
    second_warmup = logger.records.get("train/vf_clip_warmup_active", [])[-1]

    algo.train()
    warmup_flags = logger.records.get("train/vf_clip_warmup_active", [])

    assert abs(first_ev) > 1e-6
    assert abs(second_ev) > 1e-6
    assert first_warmup == pytest.approx(1.0)
    assert second_warmup == pytest.approx(1.0)
    assert warmup_flags[-1] == pytest.approx(0.0)
