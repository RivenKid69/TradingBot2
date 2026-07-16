# -*- coding: utf-8 -*-
"""
Stage A5 tests — service_risk_model (FactorRiskModel, StatRiskModel, LW shrinkage).

  * nearest_psd / ledoit_wolf_identity: PSD, симметрия, δ∈[0,1]
  * FactorRiskModel: на синтетике без шума восстанавливает известную ковариацию
    (factor returns и Σ); экспозиции согласованы; D≈0
  * StatRiskModel: PCA k=N воспроизводит сэмпловую Σ; PSD; specific_var≥0
  * Σ всегда симметрична и PSD
  * принимает панель (MultiIndex) и широкие доходности
  * соответствие core_portfolio.RiskModel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from service_risk_model import (
    FactorRiskModel,
    StatRiskModel,
    ledoit_wolf_identity,
    nearest_psd,
    to_wide_returns,
)


def _is_psd(m, tol=1e-8):
    w = np.linalg.eigvalsh(0.5 * (m + m.T))
    return float(w.min()) >= -tol


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_nearest_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eig = 3, -1
    out = nearest_psd(bad)
    assert np.allclose(out, out.T)
    assert _is_psd(out)


def test_ledoit_wolf_properties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    sigma, delta = ledoit_wolf_identity(X - X.mean(0))
    assert np.allclose(sigma, sigma.T)
    assert _is_psd(sigma)
    assert 0.0 <= delta <= 1.0


def test_ledoit_wolf_single_asset():
    X = np.array([[1.0], [3.0], [2.0], [4.0]])
    sigma, _ = ledoit_wolf_identity(X - X.mean(0))
    assert sigma.shape == (1, 1)
    assert sigma[0, 0] == pytest.approx(np.var(X, ddof=0))


# ---------------------------------------------------------------------------
# synthetic noise-free factor data (known covariance)
# ---------------------------------------------------------------------------
def _synthetic_factor_data():
    T, N = 30, 5
    t = np.arange(T)
    F = np.column_stack([np.sin(t / 3.0), np.cos(t / 5.0)])  # T×2, неколлинеарны
    B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, 0.5]])  # 5×2
    R = F @ B.T  # T×N, без шума → r = B f
    syms = [f"S{i}" for i in range(N)]
    r_wide = pd.DataFrame(R, index=pd.Index(range(T), name="ts_ms"), columns=pd.Index(syms, name="symbol"))
    B_df = pd.DataFrame(B, index=syms, columns=["f1", "f2"])
    return F, B, R, r_wide, B_df


def test_factor_model_recovers_known_covariance():
    F, B, R, r_wide, B_df = _synthetic_factor_data()
    model = FactorRiskModel(B_df, factor_cov_method="sample").fit(r_wide)
    assert isinstance(model, cp.RiskModel)

    # факторные доходности восстановлены точно (r = B f)
    assert np.allclose(model.factor_returns.to_numpy(), F, atol=1e-8)
    # идиосинкратическая дисперсия ≈ 0 (нет шума)
    assert np.allclose(model.specific_var().to_numpy(), 0.0, atol=1e-12)
    # экспозиции == B
    assert np.allclose(model.exposures().to_numpy(), B)

    # Σ_model == популяционная ковариация r (= B cov(F) Bᵀ)
    Rd = R - R.mean(0)
    cov_r = Rd.T @ Rd / R.shape[0]
    sigma = model.cov().to_numpy()
    assert np.allclose(sigma, cov_r, atol=1e-8)
    assert np.allclose(sigma, sigma.T)
    assert _is_psd(sigma)


def test_factor_model_with_noise_is_psd():
    rng = np.random.default_rng(2)
    _, B, R, r_wide, B_df = _synthetic_factor_data()
    noisy = r_wide + rng.normal(0, 0.01, size=r_wide.shape)
    model = FactorRiskModel(B_df, factor_cov_method="ledoit_wolf").fit(noisy)
    sigma = model.cov().to_numpy()
    assert np.allclose(sigma, sigma.T)
    assert _is_psd(sigma)
    assert (model.specific_var().to_numpy() >= 0).all()
    # tilt validator заполнен (интеграция с portfolio_constraints)
    assert model.tilt_validator is not None


# ---------------------------------------------------------------------------
# StatRiskModel
# ---------------------------------------------------------------------------
def test_stat_model_pca_full_reconstructs_sample_cov():
    _, _, R, r_wide, _ = _synthetic_factor_data()
    model = StatRiskModel(method="sample", n_factors=R.shape[1]).fit(r_wide)
    Rd = R - R.mean(0)
    cov_r = Rd.T @ Rd / R.shape[0]
    assert np.allclose(model.cov().to_numpy(), cov_r, atol=1e-8)
    assert (model.specific_var().to_numpy() >= -1e-12).all()


def test_stat_model_ledoit_wolf_psd_and_shapes():
    rng = np.random.default_rng(3)
    R = rng.normal(size=(150, 6))
    wide = pd.DataFrame(R, columns=[f"A{i}" for i in range(6)])
    model = StatRiskModel(method="ledoit_wolf", n_factors=3).fit(wide)
    assert isinstance(model, cp.RiskModel)
    sigma = model.cov().to_numpy()
    assert np.allclose(sigma, sigma.T)
    assert _is_psd(sigma)
    assert model.exposures().shape == (6, 3)
    assert model.factor_cov().shape == (3, 3)


# ---------------------------------------------------------------------------
# input formats
# ---------------------------------------------------------------------------
def test_accepts_panel_multiindex_input():
    F, B, R, r_wide, B_df = _synthetic_factor_data()
    long = r_wide.stack().rename("ret")  # MultiIndex (ts_ms, symbol)
    assert isinstance(long.index, pd.MultiIndex)
    model = FactorRiskModel(B_df, factor_cov_method="sample").fit(long)
    Rd = R - R.mean(0)
    cov_r = Rd.T @ Rd / R.shape[0]
    # колонки могут идти в другом порядке — сравним через выравнивание
    sigma = model.cov()
    cov_r_df = pd.DataFrame(cov_r, index=r_wide.columns, columns=r_wide.columns).loc[sigma.index, sigma.columns]
    assert np.allclose(sigma.to_numpy(), cov_r_df.to_numpy(), atol=1e-8)
