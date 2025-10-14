import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip(
    "sb3_contrib", reason="Custom policy depends on sb3_contrib recurrent components"
)

from custom_policy_patch1 import CustomActorCriticPolicy


def test_policy_forward_raises_on_nan_actions():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    policy = CustomActorCriticPolicy(obs_space, action_space, lr_schedule=lambda _: 1e-4)

    class _NaNDistribution:
        def get_actions(self, deterministic: bool = False):
            return torch.full((1, 1), float("nan"))

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.zeros(actions.shape[0], dtype=torch.float32)

        def entropy(self) -> torch.Tensor:
            return torch.zeros(1, dtype=torch.float32)

    def _fake_proba_distribution(mean: torch.Tensor, log_std: torch.Tensor):
        return _NaNDistribution()

    policy.action_dist.proba_distribution = _fake_proba_distribution  # type: ignore[assignment]

    obs = torch.zeros((1, obs_space.shape[0]), dtype=torch.float32)
    episode_starts = torch.zeros((1,), dtype=torch.float32)
    states = policy.recurrent_initial_state

    with pytest.raises(RuntimeError):
        policy.forward(obs, states, episode_starts)
