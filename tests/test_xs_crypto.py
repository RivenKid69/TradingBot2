# -*- coding: utf-8 -*-
"""
Stage B1 tests — крипто-вертикаль (signals/crypto_signals + risk/crypto_factors + pipeline).

  * крипто-сигналы считаются; graceful к отсутствующим колонкам (BYO-слот)
  * funding_carry/basis/size — корректный знак/значение
  * btc_beta восстанавливает известную β; build_crypto_exposures — корректная B
  * end-to-end бэктест по пресету config_xs_crypto.yaml → Trust Report (acceptance)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from impl_panel import PanelBuilder
from signals.crypto_signals import (
    Basis,
    CryptoMomentum,
    FundingCarry,
    OnChain,
    ShortTermReversal,
    Size,
)
from xs_risk.crypto_factors import btc_beta, build_crypto_exposures
from service_xs_pipeline import XSConfig, run_backtest, build_signal_library

T0, STEP = 1_700_000_000, 86_400
CRYPTO_CFG = os.path.join("configs", "config_xs_crypto.yaml")


def _crypto_panel():
    ts = [T0 + i * STEP for i in range(12)]
    spec = {
        "BTC": (50000.0, 0.0001, 0.002, 1300.0),
        "ETH": (3000.0, 0.0003, 0.001, 400.0),
        "SOL": (100.0, -0.0001, -0.001, 80.0),
    }
    frames = {}
    for sym, (base, fund, basis, mcap) in spec.items():
        n = len(ts)
        close = base * (1.0 + 0.01 * np.arange(n))  # детерминированный рост
        frames[sym] = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": sym,
                "close": close,
                "funding_rate": fund,
                "basis": basis,
                "mcap": mcap,
            }
        )
    return PanelBuilder.from_frames(frames)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def test_crypto_momentum_and_reversal():
    panel = _crypto_panel()
    mom = CryptoMomentum("m", lookback=3, skip=0).compute_panel(panel)
    rev = ShortTermReversal("r", window=3).compute_panel(panel)
    # mom[t] = close[t]/close[t-3]-1 ; rev = -mom
    bar3 = (T0 + 3 * STEP) * 1000
    expected = (1 + 0.01 * 3) / (1 + 0.01 * 0) - 1.0
    assert mom.loc[(bar3, "BTC")] == pytest.approx(expected)
    assert rev.loc[(bar3, "BTC")] == pytest.approx(-expected)


def test_funding_basis_size_signs():
    panel = _crypto_panel()
    fc = FundingCarry("fc").compute_panel(panel)
    bs = Basis("bs").compute_panel(panel)
    sz = Size("sz").compute_panel(panel)
    t0 = T0 * 1000
    assert fc.loc[(t0, "BTC")] == pytest.approx(-0.0001)  # -funding
    assert bs.loc[(t0, "BTC")] == pytest.approx(-0.002)  # -basis
    assert sz.loc[(t0, "BTC")] == pytest.approx(-np.log(1300.0))  # -log(mcap)


def test_signals_graceful_missing_column():
    # панель только с close → funding/basis/size/onchain → NaN (BYO-слот пуст)
    frame = pd.DataFrame({"timestamp": [T0, T0 + STEP], "symbol": "BTC", "close": [1.0, 2.0]})
    panel = PanelBuilder.from_frames({"BTC": frame})
    for sig in (FundingCarry("fc"), Basis("bs"), Size("sz"), OnChain("oc")):
        out = sig.compute_panel(panel)
        assert out.isna().all()  # нейтрально, без падения


# ---------------------------------------------------------------------------
# factors
# ---------------------------------------------------------------------------
def test_btc_beta_recovers_known():
    rng = np.random.default_rng(0)
    btc = rng.normal(0, 0.02, 120)
    rw = pd.DataFrame({"BTC": btc, "ALT2X": 2.0 * btc, "NOISE": rng.normal(0, 0.02, 120)})
    beta = btc_beta(rw, btc_symbol="BTC")
    assert beta["BTC"] == pytest.approx(1.0, abs=1e-9)
    assert beta["ALT2X"] == pytest.approx(2.0, abs=1e-9)


def test_build_crypto_exposures():
    rng = np.random.default_rng(1)
    rw = pd.DataFrame({s: rng.normal(0, 0.02, 100) for s in ["BTC", "ETH", "SOL"]})
    B = build_crypto_exposures(
        rw,
        sectors={"BTC": "L1", "ETH": "L1", "SOL": "DeFi"},
        mcaps={"BTC": 1300, "ETH": 400, "SOL": 80},
        btc_symbol="BTC",
    )
    assert "btc_beta" in B.columns and "size" in B.columns
    assert any(c.startswith("sector_") for c in B.columns)
    assert list(B.index) == ["BTC", "ETH", "SOL"]


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_builds_crypto_signals():
    cfg = XSConfig.model_validate(
        {
            "data": {"source": "synthetic", "symbols": ["BTC", "ETH"]},
            "signals": [
                {"name": "fc", "kind": "funding_carry"},
                {"name": "bs", "kind": "basis"},
                {"name": "mom", "kind": "crypto_momentum", "lookback": 30, "skip": 1},
            ],
        }
    )
    lib = build_signal_library(cfg)
    assert lib.names == ["fc", "bs", "mom"]


# ---------------------------------------------------------------------------
# end-to-end (acceptance)
# ---------------------------------------------------------------------------
def test_crypto_preset_end_to_end():
    import yaml

    with open(CRYPTO_CFG, "r", encoding="utf-8") as fh:
        cfg = XSConfig.model_validate(yaml.safe_load(fh))
    assert cfg.risk.type == "crypto_factor"
    out = run_backtest(cfg)
    assert out["n_rebalances"] > 0
    assert "deflated_sharpe" in out["trust_report"]
    # market-neutral long-short
    assert np.allclose(out["result"].net.to_numpy(), 0.0, atol=1e-6)
