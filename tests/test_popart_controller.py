from __future__ import annotations

import math
from typing import Optional

import pytest
import torch

pytest.importorskip("sb3_contrib")

from distributional_ppo import PopArtController, PopArtHoldoutBatch


class DummyLogger:
    def __init__(self) -> None:
        self.records: dict[str, list[float | str]] = {}

    def record(self, key: str, value: float | str) -> None:
        self.records.setdefault(key, []).append(value)


class DummyQuantilePolicy(torch.nn.Module):
    def __init__(self, input_dim: int, num_quantiles: int) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.quantile_head = torch.nn.Module()
        self.quantile_head.linear = torch.nn.Linear(input_dim, num_quantiles, bias=True)
        torch.nn.init.xavier_uniform_(self.quantile_head.linear.weight)
        torch.nn.init.zeros_(self.quantile_head.linear.bias)
        self.value_net = self.quantile_head.linear
        self.uses_quantile_value_head = True
        self.recurrent_initial_state = None

    def value_quantiles(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, ...]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quantile_head.linear(obs)


class DummyCategoricalPolicy(torch.nn.Module):
    def __init__(self, input_dim: int, num_atoms: int) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.dist_head = torch.nn.Linear(input_dim, num_atoms, bias=True)
        torch.nn.init.xavier_uniform_(self.dist_head.weight)
        torch.nn.init.zeros_(self.dist_head.bias)
        self.atoms = torch.linspace(-1.0, 1.0, num_atoms)
        self.v_min = float(self.atoms[0])
        self.v_max = float(self.atoms[-1])
        self.delta_z = float((self.v_max - self.v_min) / max(num_atoms - 1, 1))
        self.uses_quantile_value_head = False
        self.recurrent_initial_state = None

    def value_quantiles(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, ...]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.dist_head(obs)

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, ...]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.dist_head(obs)
        probs = torch.softmax(logits, dim=-1)
        expectation = (probs * self.atoms.to(dtype=logits.dtype)).sum(dim=-1, keepdim=True)
        return expectation

    @torch.no_grad()
    def update_atoms(self, v_min: float, v_max: float) -> None:
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atoms = torch.linspace(v_min, v_max, self.atoms.numel())


class DummyModel:
    def __init__(self, policy: torch.nn.Module, ret_mean: float, ret_std: float, *, use_quantile: bool) -> None:
        self.policy = policy
        self.device = torch.device("cpu")
        self.normalize_returns = True
        self._use_quantile_value = use_quantile
        self.raw_mean = float(ret_mean)
        self.raw_std = float(ret_std)

    def _to_raw_returns(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.new_tensor(self.raw_mean)
        std = tensor.new_tensor(self.raw_std)
        return tensor * std + mean


def _make_holdout(batch_size: int, input_dim: int, model: DummyModel) -> PopArtHoldoutBatch:
    obs = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        preds = model.policy.value_quantiles(obs, None, None)
        if model._use_quantile_value:
            baseline_norm = preds.mean(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(preds, dim=-1)
            atoms = model.policy.atoms.to(dtype=torch.float32)
            baseline_norm = (probs * atoms).sum(dim=-1, keepdim=True)
    returns_raw = model._to_raw_returns(baseline_norm)
    episode_starts = torch.zeros(batch_size, dtype=torch.float32)
    return PopArtHoldoutBatch(
        observations=obs,
        returns_raw=returns_raw,
        episode_starts=episode_starts,
        lstm_states=None,
        mask=None,
    )


def test_quantile_live_update_preserves_raw_predictions() -> None:
    torch.manual_seed(0)
    policy = DummyQuantilePolicy(input_dim=3, num_quantiles=5)
    with torch.no_grad():
        policy.quantile_head.linear.weight.copy_(torch.ones_like(policy.quantile_head.linear.weight))
        policy.quantile_head.linear.bias.fill_(0.25)
    model = DummyModel(policy, ret_mean=0.5, ret_std=0.7, use_quantile=True)
    holdout_batch = _make_holdout(batch_size=32, input_dim=3, model=model)
    logger = DummyLogger()
    controller = PopArtController(
        enabled=True,
        mode="live",
        ema_beta=0.5,
        min_samples=1,
        warmup_updates=0,
        max_rel_step=1.0,
        ev_floor=0.0,
        ret_std_band=(0.1, 5.0),
        gate_patience=1,
        holdout_loader=lambda: holdout_batch,
        logger=logger,
    )

    returns_raw = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    metrics = controller.evaluate_shadow(
        model=model,
        returns_raw=returns_raw,
        ret_mean=model.raw_mean,
        ret_std=model.raw_std,
        explained_variance_train=1.0,
    )
    assert metrics is not None
    assert metrics.passed_guards

    baseline_raw = controller._last_holdout_eval.baseline_raw.clone()
    old_mean, old_std = model.raw_mean, model.raw_std
    new_mean = metrics.mean
    new_std = metrics.std
    controller.apply_live_update(
        model=model,
        old_mean=old_mean,
        old_std=old_std,
        new_mean=new_mean,
        new_std=new_std,
    )
    model.raw_mean = new_mean
    model.raw_std = new_std

    with torch.no_grad():
        updated_preds = policy.value_quantiles(
            holdout_batch.observations, None, holdout_batch.episode_starts
        )
        updated_mean = updated_preds.mean(dim=-1, keepdim=True)
        updated_raw = model._to_raw_returns(updated_mean)

    assert torch.allclose(updated_raw, baseline_raw, atol=1e-6)

    scale_expected = old_std / new_std
    shift_expected = (old_mean - new_mean) / new_std
    weight_expected = torch.ones_like(policy.quantile_head.linear.weight) * scale_expected
    bias_expected = torch.full_like(policy.quantile_head.linear.bias, 0.25 * scale_expected + shift_expected)
    assert torch.allclose(policy.quantile_head.linear.weight, weight_expected, atol=1e-6)
    assert torch.allclose(policy.quantile_head.linear.bias, bias_expected, atol=1e-6)


def test_shadow_evaluation_does_not_mutate_policy() -> None:
    torch.manual_seed(1)
    policy = DummyQuantilePolicy(input_dim=2, num_quantiles=3)
    model = DummyModel(policy, ret_mean=0.0, ret_std=1.0, use_quantile=True)
    holdout_batch = _make_holdout(batch_size=8, input_dim=2, model=model)
    logger = DummyLogger()
    controller = PopArtController(
        enabled=True,
        mode="shadow",
        ema_beta=0.9,
        min_samples=4,
        warmup_updates=0,
        max_rel_step=1.0,
        ev_floor=0.0,
        ret_std_band=(0.1, 5.0),
        gate_patience=2,
        holdout_loader=lambda: holdout_batch,
        logger=logger,
    )

    weight_before = policy.quantile_head.linear.weight.detach().clone()
    bias_before = policy.quantile_head.linear.bias.detach().clone()

    controller.evaluate_shadow(
        model=model,
        returns_raw=torch.ones(6),
        ret_mean=0.0,
        ret_std=1.0,
        explained_variance_train=1.0,
    )

    assert torch.allclose(policy.quantile_head.linear.weight, weight_before)
    assert torch.allclose(policy.quantile_head.linear.bias, bias_before)


def test_guard_blocks_when_min_samples_not_met() -> None:
    policy = DummyQuantilePolicy(input_dim=2, num_quantiles=2)
    model = DummyModel(policy, ret_mean=0.0, ret_std=0.8, use_quantile=True)
    holdout_batch = _make_holdout(batch_size=4, input_dim=2, model=model)
    controller = PopArtController(
        enabled=True,
        mode="shadow",
        ema_beta=0.8,
        min_samples=10,
        warmup_updates=0,
        max_rel_step=1.0,
        ev_floor=0.0,
        ret_std_band=(0.1, 5.0),
        gate_patience=2,
        holdout_loader=lambda: holdout_batch,
        logger=DummyLogger(),
    )

    metrics = controller.evaluate_shadow(
        model=model,
        returns_raw=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        ret_mean=0.0,
        ret_std=0.8,
        explained_variance_train=0.9,
    )
    assert metrics is not None
    assert not metrics.passed_guards
    assert metrics.blocked_reason == "min_samples"


def test_gate_switches_to_live_after_consecutive_passes() -> None:
    policy = DummyQuantilePolicy(input_dim=1, num_quantiles=3)
    model = DummyModel(policy, ret_mean=0.0, ret_std=0.6, use_quantile=True)
    holdout_batch = _make_holdout(batch_size=6, input_dim=1, model=model)
    controller = PopArtController(
        enabled=True,
        mode="shadow",
        ema_beta=0.5,
        min_samples=2,
        warmup_updates=0,
        max_rel_step=1.0,
        ev_floor=0.0,
        ret_std_band=(0.1, 5.0),
        gate_patience=2,
        holdout_loader=lambda: holdout_batch,
        logger=DummyLogger(),
    )

    returns_raw = torch.tensor([0.0, 0.6, -0.6, 0.3], dtype=torch.float32)
    for _ in range(2):
        metrics = controller.evaluate_shadow(
            model=model,
            returns_raw=returns_raw,
            ret_mean=model.raw_mean,
            ret_std=model.raw_std,
            explained_variance_train=1.0,
        )
        assert metrics is not None and metrics.passed_guards
    assert controller.mode == "live"


def test_categorical_live_update_preserves_expectation() -> None:
    torch.manual_seed(3)
    policy = DummyCategoricalPolicy(input_dim=2, num_atoms=5)
    model = DummyModel(policy, ret_mean=0.1, ret_std=0.9, use_quantile=False)
    holdout_batch = _make_holdout(batch_size=16, input_dim=2, model=model)
    controller = PopArtController(
        enabled=True,
        mode="live",
        ema_beta=0.5,
        min_samples=1,
        warmup_updates=0,
        max_rel_step=1.0,
        ev_floor=0.0,
        ret_std_band=(0.1, 5.0),
        gate_patience=1,
        holdout_loader=lambda: holdout_batch,
        logger=DummyLogger(),
    )

    returns = torch.linspace(-0.4, 0.4, steps=6)
    metrics = controller.evaluate_shadow(
        model=model,
        returns_raw=returns,
        ret_mean=model.raw_mean,
        ret_std=model.raw_std,
        explained_variance_train=0.5,
    )
    assert metrics is not None

    baseline_raw = controller._last_holdout_eval.baseline_raw.clone()
    old_mean, old_std = model.raw_mean, model.raw_std
    controller.apply_live_update(
        model=model,
        old_mean=old_mean,
        old_std=old_std,
        new_mean=metrics.mean,
        new_std=metrics.std,
    )
    model.raw_mean = metrics.mean
    model.raw_std = metrics.std

    with torch.no_grad():
        logits = policy.value_quantiles(
            holdout_batch.observations, None, holdout_batch.episode_starts
        )
        probs = torch.softmax(logits, dim=-1)
        expected_norm = (probs * policy.atoms.to(dtype=torch.float32)).sum(dim=-1, keepdim=True)
        updated_raw = model._to_raw_returns(expected_norm)

    assert torch.allclose(updated_raw, baseline_raw, atol=1e-6)
    scale = old_std / metrics.std
    assert math.isclose(float(policy.atoms[0]), scale * (-1.0) + (old_mean - metrics.mean) / metrics.std, rel_tol=1e-6, abs_tol=1e-6)
    expected_delta = (policy.v_max - policy.v_min) / max(policy.atoms.numel() - 1, 1)
    assert math.isclose(policy.delta_z, expected_delta, rel_tol=1e-6, abs_tol=1e-6)
