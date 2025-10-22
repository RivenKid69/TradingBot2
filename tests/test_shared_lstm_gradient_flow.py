from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from custom_policy_patch1 import CustomActorCriticPolicy


def _schedule(_: float) -> float:
    return 0.001


def _make_shared_lstm_policy() -> CustomActorCriticPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    policy = CustomActorCriticPolicy(
        obs_space,
        action_space,
        _schedule,
        arch_params={"hidden_dim": 8, "enable_critic_lstm": False},
    )
    # ``shared_lstm`` is toggled dynamically in production depending on architecture
    # flags.  The test forces it on to exercise the shared branch explicitly.
    policy.shared_lstm = True
    policy.set_training_mode(True)
    return policy


def test_shared_lstm_grad_flow_unblocked() -> None:
    policy = _make_shared_lstm_policy()
    policy.set_critic_gradient_blocked(False)

    obs = torch.randn(1, 4)
    episode_starts = torch.zeros(1, dtype=torch.float32)
    features = policy.extract_features(obs)
    latent_pi, latent_vf, new_states = policy._forward_recurrent(
        features, policy.recurrent_initial_state, episode_starts
    )

    assert latent_vf.requires_grad
    assert all(state.requires_grad for state in new_states.vf)

    policy.optimizer.zero_grad(set_to_none=True)
    loss = latent_vf.pow(2).sum()
    loss.backward()

    grads = [param.grad for param in policy.lstm_actor.parameters() if param.requires_grad]
    assert any(grad is not None and torch.any(grad != 0) for grad in grads)


def test_shared_lstm_grad_blocked_zero_scale() -> None:
    policy = _make_shared_lstm_policy()
    policy.set_critic_gradient_blocked(True)

    obs = torch.randn(1, 4)
    episode_starts = torch.zeros(1, dtype=torch.float32)
    features = policy.extract_features(obs)
    _, latent_vf, blocked_states = policy._forward_recurrent(
        features, policy.recurrent_initial_state, episode_starts
    )

    assert not latent_vf.requires_grad
    assert all(not state.requires_grad for state in blocked_states.vf)
