# -*- coding: utf-8 -*-
"""
Stage D6 tests — RL-as-signal (service_rl_inference + kind rl_alpha). Без torch/без обучения (DI stub).

  * RLInferenceAdapter value/cvar: utility панель из stub value_fn/quantiles_fn
  * confidence из conformal-ширин → utility × confidence (шринк)
  * graceful: нет checkpoint/политики → NaN, available()=False
  * RLAlphaSignal измеримый IC рядом с факторами; интеграция в backtest пресета
  * pipeline kind 'rl_alpha' строит сигнал (нейтрален без артефакта)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL
from impl_panel import PanelBuilder
from impl_rl_signal import (
    RLAlphaSignal,
    conformal_confidence_from_widths,
    expected_utility_from_quantiles,
)
from service_rl_inference import RLInferenceAdapter

T0, STEP = 1_700_000_000, 86_400


def _panel(n=12, syms=("BTC", "ETH", "SOL")):
    frames = {}
    rng = np.random.default_rng(0)
    for k, s in enumerate(syms):
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005 * (k + 1), 0.02, n))
        frames[s] = pd.DataFrame(
            {"timestamp": [T0 + i * STEP for i in range(n)], "symbol": s, "close": close}
        )
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# value / cvar utility
# ---------------------------------------------------------------------------
def test_value_utility_from_stub():
    panel = _panel()

    # stub value_fn: «полезность» = z-score доходности (DI, без модели)
    def value_fn(p):
        ret = p["close"].groupby(level=SYMBOL_LEVEL, group_keys=False).pct_change().fillna(0.0)
        return ret.to_numpy()

    adapter = RLInferenceAdapter(value_fn=value_fn, utility="value")
    assert adapter.available()
    u = adapter.utility_panel(panel)
    assert u.index.equals(panel.index) and u.notna().any()


def test_cvar_utility_from_quantiles():
    panel = _panel(n=6, syms=("BTC", "ETH"))
    nq = 5

    def quantiles_fn(p):
        base = np.linspace(-1, 1, nq)
        return np.tile(base, (len(p), 1)) + np.arange(len(p)).reshape(-1, 1) * 0.01

    adapter = RLInferenceAdapter(quantiles_fn=quantiles_fn, utility="cvar", cvar_alpha=0.4)
    u = adapter.utility_panel(panel)
    # CVaR(0.4) = среднее нижних 40% квантилей (2 из 5) = mean(линспейс[:2]) + сдвиг
    assert u.index.equals(panel.index)
    assert u.notna().all()


def test_conformal_confidence_matches_canonical_scaling():
    """conf согласован с канонической position-scaling (impl_conformal.UncertaintyTrackerImpl)."""
    baseline, min_conf = 0.1, 0.5
    max_red = 1.0 - min_conf
    widths = pd.Series([0.05, 0.1, 0.15, 0.2, 0.5, 1.0])
    conf = conformal_confidence_from_widths(widths, baseline_width=baseline, min_conf=min_conf)

    # каноническая формула воспроизведена точечно
    def canon(w):
        if w <= baseline:
            return 1.0
        return max(min_conf, 1.0 - min((w - baseline) / baseline * max_red, max_red))

    np.testing.assert_allclose(conf.to_numpy(), [canon(w) for w in widths], rtol=1e-12)
    # инварианты: граница, floor, монотонность
    assert conf.iloc[1] == pytest.approx(1.0)  # width == baseline → conf 1
    assert conf.iloc[3] == pytest.approx(min_conf)  # width == 2*baseline → floor
    assert (conf.diff().dropna() <= 1e-12).all()  # не возрастает с ростом ширины


def test_cvar_utility_uses_accurate_integration():
    """CVaR-utility сигнала ≡ кусочно-линейному интегрированию (не наивный bottom-k)."""
    # линейные квантили q(τ)=τ → CVaR_0.5 точно = 0.25; наивный mean(bottom-half) был бы смещён
    N = 21
    taus = (np.arange(N) + 0.5) / N
    q = pd.DataFrame(np.tile(taus, (3, 1)))
    u = expected_utility_from_quantiles(q, cvar_alpha=0.5)
    np.testing.assert_allclose(u.to_numpy(), 0.25, atol=1e-6)


def test_confidence_shrinks_utility():
    panel = _panel(n=5, syms=("BTC",))

    def value_fn(p):
        return np.ones(len(p))

    def widths_fn(p):
        # узкий интервал → высокая уверенность; широкий → низкая
        return pd.Series(np.linspace(0.1, 1.0, len(p)), index=p.index)

    adapter = RLInferenceAdapter(value_fn=value_fn, widths_fn=widths_fn, conf_baseline_width=0.1)
    sig = adapter.build_signal("rl")
    out = sig.compute_panel(panel)
    # utility=1; conf=clip(0.1/width,0.5,1) убывает → сигнал убывает
    vals = out.xs("BTC", level=SYMBOL_LEVEL).to_numpy()
    assert vals[0] == pytest.approx(1.0)  # width=0.1 → conf=1
    assert vals[-1] < vals[0]  # шире интервал → ниже сигнал


# ---------------------------------------------------------------------------
# graceful
# ---------------------------------------------------------------------------
def test_graceful_no_checkpoint():
    panel = _panel()
    adapter = RLInferenceAdapter(checkpoint=None)  # нет ни fn, ни checkpoint
    assert adapter.available() is False
    u = adapter.utility_panel(panel)
    assert u.isna().all()  # нейтрально, без падения


def test_graceful_checkpoint_no_loader():
    panel = _panel()
    adapter = RLInferenceAdapter(checkpoint="some/path.zip")  # дефолтный загрузчик = no-op
    assert adapter.available() is False
    assert adapter.utility_panel(panel).isna().all()


# ---------------------------------------------------------------------------
# IC measurable + RLAlphaSignal
# ---------------------------------------------------------------------------
def test_rl_signal_ic_measurable():
    from impl_signal_diagnostics import signal_report

    panel = _panel(n=40, syms=("A", "B", "C", "D"))

    # value_fn слабо коррелирует с будущей доходностью → IC измерим
    def value_fn(p):
        return (
            p["close"]
            .groupby(level=SYMBOL_LEVEL, group_keys=False)
            .pct_change()
            .shift(0)
            .fillna(0.0)
            .to_numpy()
        )

    adapter = RLInferenceAdapter(value_fn=value_fn)
    sig = adapter.build_signal("rl_alpha").compute_panel(panel)
    fwd = PanelBuilder.add_forward_returns(panel, price_col="close")["fwd_return"]
    rep = signal_report(sig, fwd)
    assert "ic_mean" in rep  # IC посчитан (рядом с факторами)


def test_pipeline_kind_rl_alpha_neutral_without_artifact():
    from service_xs_pipeline import XSConfig, build_signal_library, load_panel

    cfg = XSConfig.model_validate(
        {
            "asset_class": "crypto",
            "data": {"source": "synthetic", "symbols": ["BTC", "ETH", "SOL"], "synthetic_bars": 30},
            "signals": [{"name": "rl", "kind": "rl_alpha"}],  # без cfg.rl/checkpoint → нейтрален
        }
    )
    lib = build_signal_library(cfg)
    assert lib.names == ["rl"]
    out = lib.compute(load_panel(cfg))
    assert out["rl"].isna().all()  # graceful нейтрален, не падает


def test_rl_signal_in_backtest_with_stub_adapter():
    # интеграция: RL-сигнал в SignalLibrary + бэктест (DI stub, без модели)
    from service_signals import SignalLibrary, MomentumSignal
    from service_xs_backtest import CrossSectionalBacktest, XSBacktestConfig
    from service_alpha import ICWeightedAlpha
    from service_risk_model import StatRiskModel
    from service_optimizer import PortfolioOptimizer, OptimizerConstraints
    from impl_universe import StaticUniverse

    panel = _panel(n=60, syms=("A", "B", "C", "D", "E"))

    def value_fn(p):
        return (
            p["close"]
            .groupby(level=SYMBOL_LEVEL, group_keys=False)
            .pct_change()
            .fillna(0.0)
            .to_numpy()
        )

    rl_sig = RLInferenceAdapter(value_fn=value_fn).build_signal("rl_alpha")

    lib = SignalLibrary()
    lib.register(MomentumSignal("mom", lookback=10), transforms=["zscore"])
    lib.register(rl_sig, transforms=["zscore"])  # RL рядом с классикой
    bt = CrossSectionalBacktest(
        universe=StaticUniverse(["A", "B", "C", "D", "E"]),
        alpha_model=ICWeightedAlpha(),
        risk_model=StatRiskModel(),
        optimizer=PortfolioOptimizer(
            objective="mean_variance",
            constraints=OptimizerConstraints(gross_max=1.0, net_target=0.0),
        ),
        signal_library=lib,
        config=XSBacktestConfig(rebalance_every=5, cov_lookback=20, min_cov_obs=10),
    )
    res = bt.run(panel)
    assert res.weights.shape[0] > 0  # отработал с RL-сигналом среди факторов
