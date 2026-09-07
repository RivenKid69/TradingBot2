# -*- coding: utf-8 -*-
"""
Stage A6 tests — service_alpha (μ-комбинация) + impl_rl_signal (RL как сигнал).

  * EqualWeightAlpha = среднее сигналов
  * ICWeightedAlpha повышает |вес| предсказательному сигналу vs шуму
  * RidgeAlpha регуляризует (||coef|| падает с ростом alpha)
  * predict_panel → MultiIndex μ
  * RLAlphaSignal: utility×confidence; conformal-вес; квантили→полезность;
    регистрируется в SignalLibrary и имеет измеримый (ненулевой) IC
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from impl_panel import PanelBuilder
from impl_signal_diagnostics import signal_report
from service_signals import SignalLibrary
from service_alpha import EqualWeightAlpha, ICWeightedAlpha, RidgeAlpha
from impl_rl_signal import (
    RLAlphaSignal,
    conformal_confidence_from_widths,
    expected_utility_from_quantiles,
)


def _signals_and_fwd(seed=0, T=20, syms=("S0", "S1", "S2", "S3", "S4")):
    rng = np.random.default_rng(seed)
    rows = []
    for ts in range(1, T + 1):
        for sym in syms:
            fwd = float(rng.normal())
            rows.append((ts, sym, fwd, fwd, float(rng.normal())))  # good = fwd, noise = random
    idx = pd.MultiIndex.from_tuples([(t, s) for t, s, *_ in rows], names=["ts_ms", "symbol"])
    signals = pd.DataFrame({"good": [r[3] for r in rows], "noise": [r[4] for r in rows]}, index=idx)
    fwd = pd.Series([r[2] for r in rows], index=idx, name="fwd_return")
    return signals, fwd


# ---------------------------------------------------------------------------
# Alpha models
# ---------------------------------------------------------------------------
def test_equal_weight_is_mean():
    signals, fwd = _signals_and_fwd()
    m = EqualWeightAlpha().fit(signals, fwd)
    assert isinstance(m, cp.AlphaModel)
    cs = signals.xs(1, level="ts_ms")
    mu = m.predict(cs)
    assert np.allclose(mu.to_numpy(), cs.mean(axis=1).to_numpy())


def test_ic_weighted_favours_predictive_signal():
    signals, fwd = _signals_and_fwd()
    m = ICWeightedAlpha().fit(signals, fwd)
    assert abs(m.weights["good"]) > abs(m.weights["noise"])
    assert m.ic["good"] == pytest.approx(1.0)  # good == fwd → IC=1
    mu = m.predict(signals.xs(1, level="ts_ms"))
    assert isinstance(mu, pd.Series) and len(mu) == 5


def test_ridge_regularizes():
    signals, fwd = _signals_and_fwd()
    small = RidgeAlpha(alpha=0.01).fit(signals, fwd)
    large = RidgeAlpha(alpha=1e4).fit(signals, fwd)
    assert np.linalg.norm(large.coef_) < np.linalg.norm(small.coef_)
    mu = small.predict(signals.xs(1, level="ts_ms"))
    assert len(mu) == 5


def test_predict_panel_multiindex():
    signals, fwd = _signals_and_fwd()
    mu_panel = EqualWeightAlpha().fit(signals, fwd).predict_panel(signals)
    assert isinstance(mu_panel.index, pd.MultiIndex)
    assert tuple(mu_panel.index.names) == ("ts_ms", "symbol")
    assert len(mu_panel) == len(signals)


# ---------------------------------------------------------------------------
# RL signal helpers
# ---------------------------------------------------------------------------
def test_expected_utility_from_quantiles():
    q = pd.DataFrame({"q0": [1.0, 4.0], "q1": [2.0, 5.0], "q2": [3.0, 6.0]})
    eu = expected_utility_from_quantiles(q)
    assert eu.tolist() == pytest.approx([2.0, 5.0])
    # cvar: нижние квантили
    cv = expected_utility_from_quantiles(q, cvar_alpha=0.34)  # k=1 → минимум строки
    assert cv.tolist() == pytest.approx([1.0, 4.0])


def test_conformal_confidence_from_widths():
    widths = pd.Series([0.1, 0.2, 0.4], index=["A", "B", "C"])
    conf = conformal_confidence_from_widths(widths, baseline_width=0.2, min_conf=0.5)
    assert conf["A"] == pytest.approx(1.0)  # узкий интервал → max доверие (clip)
    assert conf["B"] == pytest.approx(1.0)
    assert conf["C"] == pytest.approx(0.5)  # широкий → пол доверия


# ---------------------------------------------------------------------------
# RLAlphaSignal
# ---------------------------------------------------------------------------
def _price_panel():
    ts = [1_700_000_000, 1_700_086_400, 1_700_172_800]
    # разные траектории доходностей по символам → есть cross-sectional дисперсия (IC определён)
    closes = {
        "A": [100.0, 110.0, 115.5],  # r0=0.10, r1=0.05
        "B": [100.0, 120.0, 132.0],  # r0=0.20, r1=0.10
        "C": [100.0, 105.0, 120.75],  # r0=0.05, r1=0.15
    }
    frames = {
        sym: pd.DataFrame({"timestamp": ts, "symbol": sym, "close": c}) for sym, c in closes.items()
    }
    return PanelBuilder.from_frames(frames)


def test_rl_signal_utility_times_confidence():
    panel = _price_panel()
    utility = pd.Series(np.arange(len(panel), dtype="float64"), index=panel.index)
    sig = RLAlphaSignal(utility_source=utility, confidence=0.5)
    out = sig.compute_panel(panel)
    assert out.name == "rl_alpha"
    assert np.allclose(out.to_numpy(), utility.to_numpy() * 0.5)


def test_rl_signal_in_library_has_measurable_ic():
    panel = _price_panel()
    fwd = PanelBuilder.add_forward_returns(panel, price_col="close")["fwd_return"]
    # «выход RL» идеально коррелирует с будущим → IC должен быть высоким
    sig = RLAlphaSignal(utility_source=fwd, name="rl_alpha")
    lib = SignalLibrary().register(sig, transforms=["zscore"])
    sig_panel = lib.compute(panel)
    assert "rl_alpha" in sig_panel.columns
    rep = signal_report(sig_panel["rl_alpha"], fwd)
    assert rep["ic_mean"] is not None and rep["ic_mean"] > 0.5  # RL участвует измеримо
