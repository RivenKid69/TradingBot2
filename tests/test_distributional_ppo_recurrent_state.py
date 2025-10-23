import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip(
    "sb3_contrib", reason="Custom policy depends on sb3_contrib recurrent components"
)

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def _constant_lr_schedule(_fraction: float) -> float:
    return 1e-3


def test_policy_value_outputs_accepts_recurrent_initial_state() -> None:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    policy = CustomActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=_constant_lr_schedule,
        arch_params={"critic": {"distributional": False}},
    )

    algo = DistributionalPPO.__new__(DistributionalPPO)
    algo.policy = policy

    obs = torch.zeros((1,) + obs_space.shape, dtype=torch.float32, device=policy.device)
    episode_starts = torch.zeros(obs.shape[0], dtype=torch.float32, device=policy.device)
    initial_states = policy.recurrent_initial_state

    assert isinstance(initial_states, RNNStates)

    value_outputs = DistributionalPPO._policy_value_outputs(
        algo,
        obs,
        initial_states,
        episode_starts,
    )

    assert isinstance(value_outputs, torch.Tensor)
    assert value_outputs.shape[0] == obs.shape[0]
    assert torch.isfinite(value_outputs).all()
