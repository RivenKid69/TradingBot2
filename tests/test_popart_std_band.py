from __future__ import annotations

import sys
import types
from typing import Optional

import torch

if "sb3_contrib" not in sys.modules:
    sb3_module = types.ModuleType("sb3_contrib")

    class _RecurrentPPO:  # pragma: no cover - lightweight stub for optional dependency
        pass

    sb3_module.RecurrentPPO = _RecurrentPPO
    sys.modules["sb3_contrib"] = sb3_module

    common_module = types.ModuleType("sb3_contrib.common")
    sys.modules["sb3_contrib.common"] = common_module

    recurrent_module = types.ModuleType("sb3_contrib.common.recurrent")
    sys.modules["sb3_contrib.common.recurrent"] = recurrent_module

    policies_module = types.ModuleType("sb3_contrib.common.recurrent.policies")

    class _PolicyStub:  # pragma: no cover - lightweight stub for optional dependency
        pass

    policies_module.RecurrentActorCriticPolicy = _PolicyStub
    sys.modules["sb3_contrib.common.recurrent.policies"] = policies_module

    buffers_module = types.ModuleType("sb3_contrib.common.recurrent.buffers")

    class _BufferStub:  # pragma: no cover - lightweight stub for optional dependency
        pass

    buffers_module.RecurrentRolloutBuffer = _BufferStub
    sys.modules["sb3_contrib.common.recurrent.buffers"] = buffers_module

    type_aliases_module = types.ModuleType("sb3_contrib.common.recurrent.type_aliases")
    type_aliases_module.RNNStates = tuple
    sys.modules["sb3_contrib.common.recurrent.type_aliases"] = type_aliases_module

from distributional_ppo import PopArtController, PopArtHoldoutBatch, PopArtHoldoutEvaluation


class _DummyPolicy(torch.nn.Module):
    def __init__(self, input_dim: int, num_quantiles: int) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = torch.nn.Linear(input_dim, num_quantiles, bias=True)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.uses_quantile_value_head = True
        self.recurrent_initial_state = None

    def value_quantiles(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, ...]] = None,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.linear(obs)


class _DummyModel:
    def __init__(self, policy: _DummyPolicy, ret_mean: float, ret_std: float) -> None:
        self.policy = policy
        self.device = torch.device("cpu")
        self.normalize_returns = True
        self.raw_mean = float(ret_mean)
        self.raw_std = float(ret_std)

    def _policy_value_outputs(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[tuple[torch.Tensor, ...]],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy.value_quantiles(obs, lstm_states, episode_starts)

    def _to_raw_returns(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.new_tensor(self.raw_mean)
        std = tensor.new_tensor(self.raw_std)
        return tensor * std + mean


def _make_holdout(batch_size: int, input_dim: int, model: _DummyModel) -> PopArtHoldoutBatch:
    obs = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        preds = model.policy.value_quantiles(obs, None, None)
        baseline_norm = preds.mean(dim=-1, keepdim=True)
    returns_raw = model._to_raw_returns(baseline_norm)
    episode_starts = torch.zeros(batch_size, dtype=torch.float32)
    return PopArtHoldoutBatch(
        observations=obs,
        returns_raw=returns_raw,
        episode_starts=episode_starts,
        lstm_states=None,
        mask=None,
    )


def test_popart_std_guard_accepts_small_sigma() -> None:
    torch.manual_seed(21)
    returns_raw = 0.02 * torch.randn(64, dtype=torch.float32)
    std_estimate = float(returns_raw.std(unbiased=False).item())

    policy = _DummyPolicy(input_dim=3, num_quantiles=5)
    model = _DummyModel(policy, ret_mean=0.0, ret_std=std_estimate)
    holdout_batch = _make_holdout(batch_size=64, input_dim=3, model=model)
    controller = PopArtController(
        enabled=True,
        mode="shadow",
        ema_beta=0.9,
        min_samples=32,
        warmup_updates=0,
        max_rel_step=0.04,
        ev_floor=0.0,
        ret_std_band=(0.01, 2.0),
        gate_patience=1,
        holdout_loader=lambda: holdout_batch,
        logger=types.SimpleNamespace(record=lambda *args, **kwargs: None),
    )

    def _fake_holdout(
        self,
        *,
        model,
        holdout,
        old_mean: float,
        old_std: float,
        new_mean: float,
        new_std: float,
    ) -> PopArtHoldoutEvaluation:
        zeros = torch.zeros_like(holdout.returns_raw)
        return PopArtHoldoutEvaluation(
            baseline_raw=zeros,
            candidate_raw=zeros,
            target_raw=zeros,
            mask=None,
            ev_before=0.5,
            ev_after=0.6,
            clip_fraction_before=0.0,
            clip_fraction_after=0.0,
        )

    controller._evaluate_holdout = types.MethodType(_fake_holdout, controller)

    metrics = controller.evaluate_shadow(
        model=model,
        returns_raw=returns_raw,
        ret_mean=model.raw_mean,
        ret_std=model.raw_std,
    )

    assert metrics is not None
    assert metrics.passed_guards
    assert metrics.blocked_reason is None
    assert torch.isclose(
        torch.tensor(metrics.std),
        torch.tensor(std_estimate),
        rtol=1e-6,
    )
