# -*- coding: utf-8 -*-
"""
Regression: production __init__ must set ``_value_target_scale_base``.

Found by the PPO-core line-by-line audit. The attribute was assigned ONLY on the
env=None test/mock path, but the non-normalize_returns branch of
``_twin_critics_vf_clipping_loss`` (distributional_ppo.py ~3498/3601) READS
``self._value_target_scale_base``. Config that reaches it: quantile twin critics +
clip_range_vf + normalize_returns=False → previously crashed with AttributeError.

This test fails before the fix (attribute missing on the production path) and
passes after assigning it in production __init__.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
gym = pytest.importorskip("gymnasium")

from stable_baselines3.common.vec_env import DummyVecEnv

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO


def _make_model(normalize_returns: bool):
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    return DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs={
            "arch_params": {
                "hidden_dim": 32,
                "critic": {
                    "distributional": True,
                    "num_quantiles": 16,
                    "use_twin_critics": True,
                },
            }
        },
        n_steps=64,
        n_epochs=1,
        batch_size=64,
        normalize_returns=normalize_returns,
        clip_range_vf=0.2,  # enables the VF-clipping path
        verbose=0,
        device="cpu",
    )


@pytest.mark.parametrize("normalize_returns", [False, True])
def test_value_target_scale_base_set_on_production_init(normalize_returns):
    model = _make_model(normalize_returns)
    # The VF-clipping non-normalize branch reads this; it must exist on the production path.
    assert hasattr(model, "_value_target_scale_base")
    assert float(model._value_target_scale_base) == pytest.approx(float(model.value_target_scale))


def test_base_scale_matches_effective_at_init():
    """Base and effective scales start from the same value_target_scale baseline."""
    model = _make_model(normalize_returns=False)
    assert float(model._value_target_scale_base) == pytest.approx(
        float(model._value_target_scale_effective)
    )
