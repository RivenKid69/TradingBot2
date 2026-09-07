# -*- coding: utf-8 -*-
"""
test_xs_rl_inference_e2e.py
===========================

E2E-мост (закрывает главный doc-vs-code разрыв): обученный **Distributional-PPO чекпоинт**
→ ``make_sb3_distributional_loader`` → ``RLInferenceAdapter`` → ``RLAlphaSignal`` →
cross-sectional сигнал по панели. Проверяет реальную загрузку весов и прогон критика
(value + CVaR-utility + conformal-shrink), а не stub-функции.

Тренируем крошечную модель на Pendulum-v1 (obs_dim=3), сохраняем, грузим БЕЗ env, строим
сигнал на синтетической панели с 3 feature-колонками (совпадает с obs-схемой обучения).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")
gym = pytest.importorskip("gymnasium")

from stable_baselines3.common.vec_env import DummyVecEnv

from custom_policy_patch1 import CustomActorCriticPolicy
from distributional_ppo import DistributionalPPO
from service_rl_inference import RLInferenceAdapter, make_sb3_distributional_loader


FEATURES = ["f0", "f1", "f2"]  # obs_dim Pendulum-v1 == 3


def _train_tiny_checkpoint(path: str) -> None:
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    model = DistributionalPPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs={
            "arch_params": {
                "hidden_dim": 32,
                "critic": {
                    "distributional": True,
                    "num_quantiles": 16,
                    "huber_kappa": 1.0,
                    "use_twin_critics": True,
                },
            }
        },
        n_steps=64,
        n_epochs=1,
        batch_size=64,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=128)
    model.save(path)
    env.close()


def _make_panel(n_ts: int = 4, n_sym: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = np.repeat(np.arange(n_ts) * 60_000 + 1_700_000_000_000, n_sym)
    syms = np.tile([f"S{i}" for i in range(n_sym)], n_ts)
    idx = pd.MultiIndex.from_arrays([ts, syms], names=["ts_ms", "symbol"])
    data = rng.standard_normal((n_ts * n_sym, len(FEATURES)))
    return pd.DataFrame(data, index=idx, columns=FEATURES)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("rl") / "tiny_ppo.zip")
    _train_tiny_checkpoint(path)
    return path


def test_loader_loads_and_runs_critic(checkpoint):
    """Загрузчик реально читает чекпоинт и выдаёт работающие value_fn/quantiles_fn."""
    loader = make_sb3_distributional_loader(feature_cols=FEATURES)
    built = loader(checkpoint, "cpu")
    assert built is not None, "загрузчик должен вернуть dict для валидного чекпоинта"
    assert callable(built["value_fn"]) and callable(built["quantiles_fn"])
    assert built["value_head"]["type"] == "quantile"

    panel = _make_panel()
    obs = built["obs_fn"](panel)
    assert obs.shape == (len(panel), len(FEATURES))

    q = built["quantiles_fn"](obs)
    v = built["value_fn"](obs)
    assert q.shape == (len(panel), 16)
    assert v.shape == (len(panel),)
    assert np.isfinite(q).all() and np.isfinite(v).all()


def test_e2e_value_signal(checkpoint):
    """from_checkpoint → available → RLAlphaSignal даёт конечный cross-sectional сигнал."""
    adapter = RLInferenceAdapter.from_checkpoint(checkpoint, feature_cols=FEATURES, utility="value")
    assert adapter.available() is True

    panel = _make_panel()
    util = adapter.utility_panel(panel)
    assert isinstance(util, pd.Series)
    assert util.index.equals(panel.index)
    assert np.isfinite(util.to_numpy()).all()

    sig = adapter.build_signal("rl_alpha").compute_panel(panel)
    assert sig.name == "rl_alpha"
    assert sig.index.equals(panel.index)
    assert np.isfinite(sig.to_numpy()).all()
    # без confidence: сигнал == utility
    np.testing.assert_allclose(sig.to_numpy(), util.to_numpy(), rtol=1e-6)


def test_e2e_cvar_utility_differs_from_value(checkpoint):
    """CVaR-utility (нижние квантили) отличается от ожидания и тоже конечна."""
    val = RLInferenceAdapter.from_checkpoint(
        checkpoint, feature_cols=FEATURES, utility="value"
    ).utility_panel(_make_panel())
    cvar = RLInferenceAdapter.from_checkpoint(
        checkpoint, feature_cols=FEATURES, utility="cvar", cvar_alpha=0.25
    ).utility_panel(_make_panel())

    assert np.isfinite(cvar.to_numpy()).all()
    # CVaR (среднее нижних квантилей) <= ожидание поквантильно (риск-aware), и не идентичен
    assert not np.allclose(val.to_numpy(), cvar.to_numpy())


def test_e2e_conformal_confidence_shrinks_signal(checkpoint):
    """conformal-уверенность из ширин шринкует |сигнал| (узкий интервал → больше доверия)."""
    panel = _make_panel()

    # широкие интервалы → низкая уверенность → шринк к нулю
    def wide_widths(p):
        return pd.Series(1.0, index=p.index)

    base = RLInferenceAdapter.from_checkpoint(checkpoint, feature_cols=FEATURES, utility="value")
    shrunk = RLInferenceAdapter.from_checkpoint(
        checkpoint,
        feature_cols=FEATURES,
        utility="value",
        widths_fn=wide_widths,
        conf_baseline_width=0.1,  # baseline << width → conf=min_conf
    )
    s_base = base.build_signal().compute_panel(panel).abs().sum()
    s_shrunk = shrunk.build_signal().compute_panel(panel).abs().sum()
    assert s_shrunk < s_base, "широкие conformal-интервалы должны уменьшать |сигнал|"


def test_user_obs_fn_overrides_loader(checkpoint):
    """Пользовательская obs_fn имеет приоритет над референсной из загрузчика."""
    calls = {"n": 0}

    def my_obs_fn(p):
        calls["n"] += 1
        return p[FEATURES].to_numpy(dtype="float32")

    adapter = RLInferenceAdapter.from_checkpoint(
        checkpoint, feature_cols=FEATURES, utility="value", obs_fn=my_obs_fn
    )
    _ = adapter.utility_panel(_make_panel())
    assert calls["n"] >= 1, "должна вызываться пользовательская obs_fn, а не лоадерная"
