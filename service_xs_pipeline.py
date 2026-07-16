# -*- coding: utf-8 -*-
"""
service_xs_pipeline.py
======================

Сборка cross-sectional конвейера из конфигурации (Stage A12). Pydantic-схема +
фабрики + высокоуровневый прогон. Используется и CLI (``script_xs_*``), и API
(``xs_api``) — единая точка wiring.

Поддерживает источники данных: ``synthetic`` (детерминированный, для smoke/демо без
данных), ``parquet`` (BYO), ``free`` (адаптеры). Слой ``service_``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core_portfolio import Panel
from impl_panel import PanelBuilder
from impl_universe import StaticUniverse, IndexMembershipUniverse
from impl_data_sources import ParquetPriceSource, AdapterPriceSource, build_price_panel
from service_signals import SignalLibrary, ColumnSignal, MomentumSignal
from signals.crypto_signals import build_crypto_signal, CRYPTO_SIGNAL_KINDS
from signals.equity_signals import build_equity_signal, EQUITY_SIGNAL_KINDS
from signals.futures_signals import build_futures_signal, FUTURES_SIGNAL_KINDS
from signals.forex_signals import build_forex_signal, FOREX_SIGNAL_KINDS
from signals.options_signals import build_options_signal, OPTIONS_SIGNAL_KINDS
from signals.common_signals import build_common_signal, COMMON_SIGNAL_KINDS
from service_alpha import EqualWeightAlpha, ICWeightedAlpha, RidgeAlpha
from service_risk_model import StatRiskModel, FactorRiskModel
from service_optimizer import OptimizerConstraints, PortfolioOptimizer, TCostModel, SizingConfig, RobustConfig
from service_xs_backtest import CrossSectionalBacktest, XSBacktestConfig
from service_backtest_validation import trust_report


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------
class DataCfg(BaseModel):
    source: str = "synthetic"            # synthetic | parquet | free
    symbols: List[str] = Field(default_factory=list)
    timeframe: str = "1d"
    parquet_root: Optional[str] = None
    vendor: str = "yahoo"
    synthetic_bars: int = 120
    synthetic_seed: int = 42
    # Stage D0: free-обогащение + кэш
    enrich: List[str] = Field(default_factory=list)   # имена обогатителей (D0: 'mcap'; D1-D5: funding/...)
    cache: bool = True                                # parquet-кэш баров (free rate-limit)
    cache_ttl_ms: Optional[int] = None                # TTL кэша (None = бессрочно)


class UniverseCfg(BaseModel):
    type: str = "static"                 # static | index_membership
    symbols: List[str] = Field(default_factory=list)
    index: Optional[str] = None
    membership_path: Optional[str] = None  # PIT changes-файл (date,ticker,action) → survivorship-free


class SignalCfg(BaseModel):
    name: str
    kind: str = "momentum"               # column | momentum | <crypto> | <equity>
                                         # crypto: crypto_momentum | reversal | funding_carry | basis | size | onchain
                                         # equity: equity_momentum | earnings_yield | book_to_price | fcf_yield |
                                         #         roe | accruals | low_vol | equity_size
    column: Optional[str] = None
    lookback: int = 60
    skip: int = 0
    window: int = 5
    price_col: str = "close"
    # crypto-specific columns
    funding_col: str = "funding_rate"
    basis_col: str = "basis"
    mcap_col: str = "mcap"
    onchain_column: str = "onchain"
    # equity-specific columns (фундаментал — BYO-слот)
    yield_col: Optional[str] = None      # готовая yield-колонка (ep/bp/fcf_yield) если есть
    earnings_col: str = "earnings"
    book_col: str = "book_value"
    fcf_col: str = "fcf"
    roe_col: str = "roe"
    accruals_col: str = "accruals"
    market_cap_col: str = "market_cap"
    vol_window: int = 60
    # futures-specific (Stage B3)
    vol_normalize: bool = False
    carry_col: str = "carry"
    roll_yield_col: str = "roll_yield"
    front_col: str = "front"
    back_col: str = "back"
    # forex-specific (Stage B4)
    rate_diff_col: str = "rate_diff"
    rate_base_col: str = "rate_base"
    rate_quote_col: str = "rate_quote"
    ppp_col: str = "ppp"
    reer_col: str = "reer_gap"
    terms_col: str = "terms_of_trade"
    # options-specific (Stage B5) — vol-структуры (BYO опционные данные)
    iv_col: str = "iv"
    rv_col: str = "realized_vol"
    vrp_col: str = "vrp"
    skew_col: str = "skew"
    dispersion_col: str = "dispersion"
    term_slope_col: str = "term_slope"
    transforms: List[Any] = Field(default_factory=lambda: ["zscore"])
    neutralize_by: List[str] = Field(default_factory=list)


class AlphaCfg(BaseModel):
    method: str = "equal_weight"         # equal_weight | ic_weighted | ridge
    alpha: float = 1.0


class RiskCfg(BaseModel):
    type: str = "stat"                   # stat | factor
    method: str = "ledoit_wolf"          # ledoit_wolf | sample
    n_factors: Optional[int] = None


class OptimizerCfg(BaseModel):
    objective: str = "mean_variance"
    risk_aversion: float = 5.0
    gross_max: Optional[float] = 1.0
    net_target: Optional[float] = 0.0    # market-neutral по умолчанию
    long_only: bool = False
    max_position: Optional[float] = None
    max_turnover: Optional[float] = None
    # P1: tcost В целевой функции (scipy) + сайзинг (vol-target / Kelly)
    tcost_aware: bool = False
    tcost_linear: float = 0.0008         # линейный косты на единицу оборота (8bps)
    tcost_quad: float = 0.0              # квадратичный market-impact
    tcost_coef: float = 1.0             # общий множитель κ
    sizing: Optional[str] = None        # None | vol_target | kelly
    target_vol: Optional[float] = None  # для vol_target (σ на период)
    kelly_fraction: float = 0.5         # для kelly
    max_leverage: Optional[float] = None
    # P1 #6: ранее реализованные, но недоступные из YAML возможности оптимизатора —
    # теперь конфигурируемые (sector/factor caps, robust μ, BL-views, multi-period).
    sector_caps: Optional[Dict[str, float]] = None   # {sector: gross-cap} (sector_map = cfg.sectors)
    factor_caps: Optional[Dict[str, float]] = None   # {factor: |Bᵀw| cap}; нужны exposures
    exposures: Optional[Dict[str, Dict[str, float]]] = None  # BYO B: {symbol: {factor: loading}}
    beta_neutral: bool = False           # βᵀw=0 (через factor cap≈0 на 'market'/'beta')
    beta_factor: str = "market"          # имя факторной колонки для beta_neutral
    robust: Optional[Dict[str, Any]] = None      # {enabled, kind: box|ellipsoidal, kappa, mu_uncertainty?}
    bl_views: Optional[Dict[str, Any]] = None    # {P, Q, omega?, tau?} — Black-Litterman views
    multi_period: Optional[Dict[str, Any]] = None  # {enabled, trade_rate?, trade_cost?} — Gârleanu–Pedersen


class BacktestCfg(BaseModel):
    rebalance_every: int = 1
    cov_lookback: int = 60
    min_cov_obs: int = 5
    alpha_refit_every: int = 1
    cost_bps: float = 5.0
    price_col: str = "close"
    periods_per_year: float = 252.0


class CapacityCfg(BaseModel):
    """Capacity analysis (Stage A9): AUM→Sharpe decay via √participation impact.
    The allocator's first question after Sharpe — surfaced in the Trust Report."""
    enabled: bool = True
    adv_usd: float = 10_000_000.0        # assumed per-name ADV (USD); override for a real universe
    aum_grid: List[float] = Field(
        default_factory=lambda: [1e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10])
    impact_coef: float = 0.1
    sharpe_threshold_frac: float = 0.5


class RLCfg(BaseModel):
    """RL-as-signal (Stage D6) — обученная Distributional-PPO политика как сигнал. Training не трогаем."""
    checkpoint: Optional[str] = None     # путь к артефакту (BYO/CCEA-signed); None → нейтральный сигнал
    utility: str = "value"               # value (value-head) | cvar (нижние квантили критика)
    cvar_alpha: float = 0.05
    confidence: bool = False             # шринк utility × conformal-confidence (нужен widths-источник)
    conf_baseline_width: float = 0.1


class XSConfig(BaseModel):
    mode: str = "cross_sectional"
    asset_class: str = "crypto"
    data: DataCfg = Field(default_factory=DataCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    signals: List[SignalCfg] = Field(default_factory=list)
    alpha: AlphaCfg = Field(default_factory=AlphaCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    capacity: CapacityCfg = Field(default_factory=CapacityCfg)
    n_trials: int = 1
    # crypto factor model (risk.type='crypto_factor')
    sectors: Optional[Dict[str, str]] = None
    mcaps: Optional[Dict[str, float]] = None
    btc_symbol: str = "BTC"
    # equity factor model (risk.type='equity_factor') — Barra-lite
    values: Optional[Dict[str, float]] = None       # value-скор (book-to-price/E-P), BYO
    market_symbol: Optional[str] = None             # явный индекс; иначе равновзвешенный прокси
    momentum_lookback: int = 60                     # также vol_lookback для futures-факторов
    # equity PIT-фундаментал (Stage D2, enrich: pit_fundamentals)
    fundamentals_path: Optional[str] = None         # BYO PIT parquet (publish_ts) → pit=true; иначе free снимок → none
    fundamentals_fields: Optional[List[str]] = None # колонки фундаментала (earnings/book_value/fcf/roe)
    fundamentals_publish_lag_days: int = 0          # лаг публикации (анти-look-ahead запас)
    # futures factor model (risk.type='futures_factor')
    asset_classes: Optional[Dict[str, str]] = None  # symbol → класс (equity_index/rates/energy/...)
    # forex factor model (risk.type='forex_factor')
    carries: Optional[Dict[str, float]] = None       # carry-скор (дифференциал ставок), BYO
    blocs: Optional[Dict[str, str]] = None           # symbol → блок (G10/EM/commodity)
    usd_symbol: Optional[str] = None                 # USD-индекс; иначе равновзвеш. прокси
    # forex rate-diff обогащение (Stage D3, enrich: rate_diff)
    policy_rates: Optional[Dict[str, float]] = None  # {currency: rate} (snapshot approx; BYO history → PIT)
    # options IV обогащение (Stage D5, enrich: iv/realized_vol)
    iv_vendor: str = "deribit"                       # deribit (крипто, approx) | yfinance (US EOD снимок, none)
    # RL-as-signal (Stage D6, kind: rl_alpha)
    rl: Optional[RLCfg] = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_universe(cfg: XSConfig):
    u = cfg.universe
    syms = u.symbols or cfg.data.symbols
    import logging as _logging
    if u.type == "index_membership":
        # PIT survivorship-free состав, если задан membership_path с историей членства.
        if u.membership_path:
            # P0 #2: guard against silently running on the tiny demo membership file
            # (which is survivorship-biased and not representative of a real index).
            try:
                import os as _os
                mp = str(u.membership_path)
                if "demo" in mp.lower():
                    _logging.getLogger(__name__).warning(
                        "index_membership uses a DEMO file (%s): not a real index history — "
                        "results are survivorship-biased. Supply a full membership history.", mp)
                elif _os.path.exists(mp):
                    with open(mp, "r", encoding="utf-8") as _fh:
                        _n = sum(1 for _ in _fh)
                    if _n < 50:
                        _logging.getLogger(__name__).warning(
                            "index_membership file %s has only %d rows — likely incomplete "
                            "history (survivorship risk).", mp, _n)
            except Exception:  # pragma: no cover - guard must never break the run
                pass
            try:
                from services.index_membership_loader import build_index_membership_universe
                return build_index_membership_universe(
                    u.membership_path, index=u.index or "CUSTOM",
                    name=f"index:{u.index or 'custom'}")
            except Exception as exc:  # без файла/при ошибке — honest деградация
                _logging.getLogger(__name__).warning(
                    "index_membership load failed (%s) — fallback to StaticUniverse (survivorship-biased)", exc)
        else:
            _logging.getLogger(__name__).warning(
                "universe.type=index_membership but no membership_path set — using a static "
                "(survivorship-biased) universe. Provide membership_path for PIT correctness.")
        return StaticUniverse(syms, name=f"index:{u.index or 'custom'}")
    return StaticUniverse(syms)


def build_signal_library(cfg: XSConfig) -> SignalLibrary:
    lib = SignalLibrary()
    for s in cfg.signals:
        if s.kind == "column":
            sig = ColumnSignal(s.name, s.column or s.name)
        elif s.kind == "momentum":
            sig = MomentumSignal(s.name, lookback=s.lookback, skip=s.skip, price_col=s.price_col)
        elif s.kind in CRYPTO_SIGNAL_KINDS:
            sig = _build_crypto(s)
        elif s.kind in EQUITY_SIGNAL_KINDS:
            sig = _build_equity(s)
        elif s.kind in FUTURES_SIGNAL_KINDS:
            sig = _build_futures(s)
        elif s.kind in FOREX_SIGNAL_KINDS:
            sig = _build_forex(s)
        elif s.kind in OPTIONS_SIGNAL_KINDS:
            sig = _build_options(s)
        elif s.kind in COMMON_SIGNAL_KINDS:
            sig = _build_common(s)
        elif s.kind == "rl_alpha":
            sig = _build_rl(s, cfg)
        else:
            raise ValueError(f"unknown signal kind: {s.kind!r}")
        lib.register(sig, transforms=s.transforms, neutralize_by=s.neutralize_by)
    return lib


def _build_crypto(s: "SignalCfg"):
    """Сконструировать крипто-сигнал (Stage B1) из SignalCfg."""
    if s.kind == "crypto_momentum":
        kw = dict(lookback=s.lookback or 90, skip=s.skip or 7, price_col=s.price_col)
    elif s.kind == "reversal":
        kw = dict(window=s.window, price_col=s.price_col)
    elif s.kind == "funding_carry":
        kw = dict(funding_col=s.funding_col)
    elif s.kind == "basis":
        kw = dict(basis_col=s.basis_col, spot_col=s.price_col)
    elif s.kind == "size":
        kw = dict(mcap_col=s.mcap_col)
    elif s.kind == "onchain":
        kw = dict(column=s.onchain_column)
    else:
        kw = {}
    return build_crypto_signal(s.kind, s.name, **kw)


def _build_equity(s: "SignalCfg"):
    """Сконструировать equity-сигнал (Stage B2) из SignalCfg."""
    if s.kind == "equity_momentum":
        kw = dict(lookback=s.lookback or 252, skip=s.skip or 21, price_col=s.price_col)
    elif s.kind == "earnings_yield":
        kw = dict(yield_col=s.yield_col or "ep", earnings_col=s.earnings_col, price_col=s.price_col)
    elif s.kind == "book_to_price":
        kw = dict(yield_col=s.yield_col or "bp", book_col=s.book_col, price_col=s.price_col)
    elif s.kind == "fcf_yield":
        kw = dict(yield_col=s.yield_col or "fcf_yield", fcf_col=s.fcf_col, price_col=s.price_col)
    elif s.kind == "roe":
        kw = dict(roe_col=s.roe_col)
    elif s.kind == "accruals":
        kw = dict(accruals_col=s.accruals_col)
    elif s.kind == "low_vol":
        kw = dict(window=s.vol_window, price_col=s.price_col)
    elif s.kind == "equity_size":
        kw = dict(mcap_col=s.market_cap_col)
    else:
        kw = {}
    return build_equity_signal(s.kind, s.name, **kw)


def _build_futures(s: "SignalCfg"):
    """Сконструировать futures-сигнал (Stage B3) из SignalCfg."""
    if s.kind == "trend":
        kw = dict(lookback=s.lookback or 100, price_col=s.price_col,
                  vol_normalize=s.vol_normalize, vol_window=s.vol_window)
    elif s.kind == "carry":
        kw = dict(carry_col=s.carry_col, roll_yield_col=s.roll_yield_col,
                  front_col=s.front_col, back_col=s.back_col)
    elif s.kind == "futures_value":
        kw = dict(lookback=s.lookback or 1000, price_col=s.price_col)
    elif s.kind == "inv_vol":
        kw = dict(window=s.vol_window, price_col=s.price_col)
    else:
        kw = {}
    return build_futures_signal(s.kind, s.name, **kw)


def _build_forex(s: "SignalCfg"):
    """Сконструировать forex-сигнал (Stage B4) из SignalCfg."""
    if s.kind == "fx_carry":
        kw = dict(rate_diff_col=s.rate_diff_col, carry_col=s.carry_col,
                  rate_base_col=s.rate_base_col, rate_quote_col=s.rate_quote_col)
    elif s.kind == "fx_momentum":
        kw = dict(lookback=s.lookback or 90, price_col=s.price_col)
    elif s.kind == "fx_value":
        kw = dict(ppp_col=s.ppp_col, reer_col=s.reer_col, lookback=s.lookback or 500, price_col=s.price_col)
    elif s.kind == "terms_of_trade":
        kw = dict(terms_col=s.terms_col)
    else:
        kw = {}
    return build_forex_signal(s.kind, s.name, **kw)


def _build_options(s: "SignalCfg"):
    """Сконструировать options-сигнал (Stage B5) из SignalCfg."""
    if s.kind == "vrp":
        kw = dict(vrp_col=s.vrp_col, iv_col=s.iv_col, rv_col=s.rv_col)
    elif s.kind == "skew":
        kw = dict(skew_col=s.skew_col)
    elif s.kind == "dispersion":
        kw = dict(dispersion_col=s.dispersion_col)
    elif s.kind == "term_structure":
        kw = dict(slope_col=s.term_slope_col)
    else:
        kw = {}
    return build_options_signal(s.kind, s.name, **kw)


def _build_common(s: "SignalCfg"):
    """Asset-agnostic сигналы (P2): residual momentum / seasonality / sentiment / 52w-high / idio-vol / cot."""
    if s.kind == "residual_momentum":
        kw = dict(lookback=s.lookback or 252, skip=s.skip or 21,
                  beta_window=s.vol_window or 60, price_col=s.price_col)
    elif s.kind == "seasonality":
        kw = dict(price_col=s.price_col)
    elif s.kind == "sentiment":
        kw = dict(column=s.column or "sentiment")
    elif s.kind == "high_52w":
        kw = dict(window=s.lookback or 252, price_col=s.price_col)
    elif s.kind == "idio_vol":
        kw = dict(window=s.vol_window or 60, price_col=s.price_col)
    elif s.kind == "cot":
        kw = dict(column=s.column or "cot_net")
    else:
        kw = {}
    return build_common_signal(s.kind, s.name, **kw)


def _build_rl(s: "SignalCfg", cfg: "XSConfig"):
    """RL-as-signal (Stage D6): RLInferenceAdapter из cfg.rl → RLAlphaSignal. Без checkpoint → нейтрален."""
    from service_rl_inference import RLInferenceAdapter

    rl = cfg.rl or RLCfg()
    adapter = RLInferenceAdapter(
        checkpoint=rl.checkpoint, utility=rl.utility, cvar_alpha=rl.cvar_alpha,
        conf_baseline_width=rl.conf_baseline_width,
    )
    if not adapter.available():
        import logging
        logging.getLogger(__name__).warning(
            "rl_alpha %r: нет рабочего артефакта (checkpoint=%r) → сигнал нейтрален (NaN). "
            "Подайте обученный Distributional-PPO артефакт + obs_fn (BYO/CCEA-signed).",
            s.name, rl.checkpoint,
        )
    return adapter.build_signal(s.name)


def build_alpha(cfg: XSConfig):
    a = cfg.alpha
    if a.method == "equal_weight":
        return EqualWeightAlpha()
    if a.method == "ic_weighted":
        return ICWeightedAlpha()
    if a.method == "ridge":
        return RidgeAlpha(alpha=a.alpha)
    raise ValueError(f"unknown alpha method: {a.method!r}")


def build_risk_model(cfg: XSConfig):
    r = cfg.risk
    if r.type == "stat":
        return StatRiskModel(method=r.method, n_factors=r.n_factors)
    raise ValueError(f"risk type {r.type!r} requires exposures (use 'stat' or 'crypto_factor')")


def build_crypto_factor_risk(cfg: XSConfig, panel: Panel):
    """FactorRiskModel с крипто-экспозициями (BTC-beta/size/sector) из панели (Stage B1)."""
    from xs_risk.crypto_factors import returns_wide_from_panel, build_crypto_exposures

    rw = returns_wide_from_panel(panel, price_col=cfg.backtest.price_col)
    B = build_crypto_exposures(rw, sectors=cfg.sectors, mcaps=cfg.mcaps, btc_symbol=cfg.btc_symbol)
    method = cfg.risk.method if cfg.risk.method in ("ledoit_wolf", "sample", "ewma") else "ledoit_wolf"
    return FactorRiskModel(B, factor_cov_method=method)


def _latest_cross_section(panel: Panel, col: str):
    """Latest available value per symbol for a panel column (PIT-safe: uses what's
    in the panel, which is already as-of joined). Returns {symbol: float} or None."""
    if col not in panel.columns:
        return None
    from core_portfolio import SYMBOL_LEVEL as _SYM, TS_LEVEL as _TS
    s = panel[col].dropna()
    if s.empty:
        return None
    df = s.reset_index()
    try:
        last = df.sort_values(_TS).groupby(_SYM)[col].last()
    except Exception:
        return None
    out = {str(k): float(v) for k, v in last.items() if np.isfinite(float(v))}
    return out or None


def build_equity_factor_risk(cfg: XSConfig, panel: Panel):
    """FactorRiskModel с equity-экспозициями (market-beta/size/value/quality/momentum/
    low_vol/sector) (Stage B2 + P2 #17). VALUE/QUALITY строятся из фундаментальных
    колонок панели (market_cap/earnings/book_value/roe), а не только из BYO-скора."""
    from xs_risk.equity_factors import returns_wide_from_panel, build_equity_exposures

    rw = returns_wide_from_panel(panel, price_col=cfg.backtest.price_col)
    # Pull fundamentals from the panel (PIT as-of joined upstream) to build factors.
    mcaps = cfg.mcaps or _latest_cross_section(panel, "market_cap") or _latest_cross_section(panel, "mcap")
    earnings = _latest_cross_section(panel, "earnings") or _latest_cross_section(panel, "net_income")
    book = _latest_cross_section(panel, "book_value")
    roe = _latest_cross_section(panel, "roe")
    B = build_equity_exposures(
        rw, sectors=cfg.sectors, mcaps=mcaps, values=cfg.values,
        earnings=earnings, book=book, roe=roe,
        market_symbol=cfg.market_symbol, momentum_lookback=cfg.momentum_lookback,
        vol_lookback=cfg.momentum_lookback,
    )
    method = cfg.risk.method if cfg.risk.method in ("ledoit_wolf", "sample", "ewma") else "ledoit_wolf"
    return FactorRiskModel(B, factor_cov_method=method)


def build_futures_factor_risk(cfg: XSConfig, panel: Panel):
    """FactorRiskModel с futures-экспозициями (market-beta/vol/asset-class) (Stage B3)."""
    from xs_risk.futures_factors import returns_wide_from_panel, build_futures_exposures

    rw = returns_wide_from_panel(panel, price_col=cfg.backtest.price_col)
    ac = cfg.asset_classes or cfg.sectors  # asset-class metadata (sectors как алиас)
    B = build_futures_exposures(
        rw, asset_classes=ac, market_symbol=cfg.market_symbol, vol_lookback=cfg.momentum_lookback,
    )
    method = cfg.risk.method if cfg.risk.method in ("ledoit_wolf", "sample", "ewma") else "ledoit_wolf"
    return FactorRiskModel(B, factor_cov_method=method)


def build_forex_factor_risk(cfg: XSConfig, panel: Panel):
    """FactorRiskModel с forex-экспозициями (USD-beta/carry/value) (Stage B4)."""
    from xs_risk.forex_factors import returns_wide_from_panel, build_forex_exposures

    rw = returns_wide_from_panel(panel, price_col=cfg.backtest.price_col)
    B = build_forex_exposures(
        rw, carries=cfg.carries, values=cfg.values, blocs=cfg.blocs, usd_symbol=cfg.usd_symbol,
    )
    method = cfg.risk.method if cfg.risk.method in ("ledoit_wolf", "sample", "ewma") else "ledoit_wolf"
    return FactorRiskModel(B, factor_cov_method=method)


def _build_run_risk_model(cfg: XSConfig, panel: Panel):
    """Диспетчер риск-модели (stat | crypto_factor | equity_factor | futures_factor | forex_factor)."""
    if cfg.risk.type == "crypto_factor":
        return build_crypto_factor_risk(cfg, panel)
    if cfg.risk.type == "equity_factor":
        return build_equity_factor_risk(cfg, panel)
    if cfg.risk.type == "futures_factor":
        return build_futures_factor_risk(cfg, panel)
    if cfg.risk.type == "forex_factor":
        return build_forex_factor_risk(cfg, panel)
    return build_risk_model(cfg)


def build_optimizer(cfg: XSConfig):
    """Build the portfolio optimizer from config.

    P1 #6: previously only gross/net/long-only/max_position/max_turnover were wired
    from YAML; the optimizer's sector/factor caps, robust μ-uncertainty, Black-Litterman
    views and multi-period (Gârleanu–Pedersen) were dark. They are now all configurable.
    """
    import numpy as _np
    o = cfg.optimizer

    # factor-loadings B (for factor_caps / beta_neutral): BYO static exposures, else
    # the cfg.values/betas can seed a 'market'/'value' column. Built as a DataFrame.
    exposures_df = None
    if o.exposures:
        exposures_df = pd.DataFrame(o.exposures).T.astype("float64")  # index=symbol, cols=factor

    factor_caps = dict(o.factor_caps) if o.factor_caps else None
    if o.beta_neutral:
        factor_caps = factor_caps or {}
        factor_caps.setdefault(o.beta_factor, 1e-6)   # |βᵀw| ≈ 0
        # seed a market-beta exposure column of 1.0 if none provided (cross-sectional β proxy)
        if exposures_df is None or o.beta_factor not in (exposures_df.columns if exposures_df is not None else []):
            syms = cfg.universe.symbols or cfg.data.symbols or []
            if syms:
                col = pd.Series(1.0, index=[str(s) for s in syms], name=o.beta_factor)
                exposures_df = (col.to_frame() if exposures_df is None
                                else exposures_df.join(col, how="outer"))

    cons = OptimizerConstraints(
        gross_max=o.gross_max, net_target=o.net_target, long_only=o.long_only,
        max_position=o.max_position, max_turnover=o.max_turnover,
        sector_map=cfg.sectors, sector_caps=(dict(o.sector_caps) if o.sector_caps else None),
        exposures=exposures_df, factor_caps=factor_caps,
    )
    tcost = (TCostModel(linear=o.tcost_linear, quad=o.tcost_quad, coef=o.tcost_coef)
             if o.tcost_aware else None)
    sizing = (SizingConfig(method=o.sizing, target_vol=o.target_vol,
                           kelly_fraction=o.kelly_fraction, max_leverage=o.max_leverage)
              if o.sizing else None)
    robust = None
    if o.robust and bool(o.robust.get("enabled")):
        mu_unc = o.robust.get("mu_uncertainty")
        robust = RobustConfig(
            enabled=True, kind=str(o.robust.get("kind", "box")),
            kappa=float(o.robust.get("kappa", 1.0)),
            mu_uncertainty=(_np.asarray(mu_unc, dtype="float64") if mu_unc is not None else None),
        )
    bl_views = None
    if o.bl_views and o.bl_views.get("P") is not None and o.bl_views.get("Q") is not None:
        bl_views = {
            "P": _np.asarray(o.bl_views["P"], dtype="float64"),
            "Q": _np.asarray(o.bl_views["Q"], dtype="float64"),
            "omega": (_np.asarray(o.bl_views["omega"], dtype="float64")
                      if o.bl_views.get("omega") is not None else None),
            "tau": float(o.bl_views.get("tau", 0.05)),
        }

    opt = PortfolioOptimizer(
        objective=o.objective, risk_aversion=o.risk_aversion, use_cvxpy="auto",
        constraints=cons, tcost=tcost, sizing=sizing, robust=robust, bl_views=bl_views,
    )
    # multi-period (Gârleanu–Pedersen): wrap with a single-step aim-blending optimizer.
    if o.multi_period and bool(o.multi_period.get("enabled")):
        from service_optimizer import MultiPeriodOptimizer
        tr = o.multi_period.get("trade_rate")
        return MultiPeriodOptimizer(
            opt, trade_rate=(float(tr) if tr is not None else None),
            trade_cost=float(o.multi_period.get("trade_cost", 0.001)))
    return opt


def build_backtest_config(cfg: XSConfig) -> XSBacktestConfig:
    b = cfg.backtest
    return XSBacktestConfig(
        rebalance_every=b.rebalance_every, cov_lookback=b.cov_lookback,
        min_cov_obs=b.min_cov_obs, alpha_refit_every=b.alpha_refit_every,
        cost_bps=b.cost_bps, price_col=b.price_col, periods_per_year=b.periods_per_year,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _synthetic_panel(symbols: List[str], n_bars: int, seed: int) -> Panel:
    rng = np.random.default_rng(seed)
    t0, step = 1_700_000_000, 86_400
    ts = [t0 + i * step for i in range(n_bars)]
    frames = {}
    for s in symbols:
        r = rng.normal(0.0005, 0.02, n_bars)
        close = 100.0 * np.cumprod(1.0 + r)
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": close})
    return PanelBuilder.from_frames(frames)


def load_panel(cfg: XSConfig) -> Panel:
    d = cfg.data
    syms = d.symbols or cfg.universe.symbols
    if d.source == "synthetic":
        if not syms:
            syms = [f"S{i}" for i in range(8)]
        if cfg.asset_class == "futures":
            # трендовые непрерывные серии (демонстрирует impl_continuous_futures в контуре)
            from impl_continuous_futures import synthetic_continuous_frames
            frames = synthetic_continuous_frames(syms, n_bars=d.synthetic_bars, seed=d.synthetic_seed)
            return PanelBuilder.from_frames(frames)
        return _synthetic_panel(syms, d.synthetic_bars, d.synthetic_seed)
    if d.source == "parquet":
        src = ParquetPriceSource(root=d.parquet_root)
        return build_price_panel(src, syms, d.timeframe)
    if d.source == "free":
        return assemble_free(cfg).panel
    raise ValueError(f"unknown data source: {d.source!r}")


# ---------------------------------------------------------------------------
# Free data assembly + quality (Stage D0)
# ---------------------------------------------------------------------------
def build_enrichers(cfg: XSConfig) -> List[Any]:
    """Реестр обогатителей по именам ``cfg.data.enrich`` (D1: crypto funding/basis/mcap; D2-D5 расширят)."""
    import logging
    from loaders.crypto_enrich import CRYPTO_ENRICHERS, build_crypto_enricher
    from loaders.equity_enrich import EQUITY_ENRICHERS, build_equity_enricher
    from loaders.forex_enrich import FOREX_ENRICHERS, build_forex_enricher
    from loaders.futures_enrich import FUTURES_ENRICHERS, build_futures_enricher
    from loaders.options_enrich import OPTIONS_ENRICHERS, build_options_enricher
    from loaders.altdata_enrich import ALTDATA_ENRICHERS, build_altdata_enricher

    out: List[Any] = []
    for name in cfg.data.enrich:
        enr = None
        if name in CRYPTO_ENRICHERS:
            enr = build_crypto_enricher(name, cfg)
        elif name in EQUITY_ENRICHERS:
            enr = build_equity_enricher(name, cfg)
        elif name in FOREX_ENRICHERS:
            enr = build_forex_enricher(name, cfg)
        elif name in FUTURES_ENRICHERS:
            enr = build_futures_enricher(name, cfg)
        elif name in OPTIONS_ENRICHERS:
            enr = build_options_enricher(name, cfg)
        elif name in ALTDATA_ENRICHERS:
            enr = build_altdata_enricher(name, cfg)
        if enr is not None:
            out.append(enr)
        else:
            logging.getLogger(__name__).warning(
                "enricher %r неизвестен/без данных на этой стадии — пропуск (добавится в D4-D5)", name
            )
    return out


def _price_source_for(cfg: XSConfig):
    """Прайс-источник под класс актива: futures free → continuous-прокси (yahoo ES=F), иначе AdapterPriceSource."""
    if cfg.asset_class == "futures":
        from loaders.futures_enrich import ContinuousProxySource
        return ContinuousProxySource(vendor=cfg.data.vendor or "yahoo")
    return AdapterPriceSource(vendor=cfg.data.vendor)


def assemble_free(cfg: XSConfig):
    """Собрать free-панель через DataAssembler (prices + enrichers + кэш) → AssembleResult."""
    from service_xs_data import DataAssembler
    from impl_data_cache import ParquetCache

    d = cfg.data
    syms = d.symbols or cfg.universe.symbols
    src = _price_source_for(cfg)
    cache = ParquetCache(enabled=bool(d.cache))
    assembler = DataAssembler(src, enrichers=build_enrichers(cfg), cache=cache, cache_ttl_ms=d.cache_ttl_ms)
    return assembler.assemble(syms, d.timeframe, price_col=cfg.backtest.price_col)


def _provenance_for(cfg: XSConfig, panel: Panel):
    """Провенанс колонок панели для synthetic/parquet (free даёт провенанс из assembler)."""
    from core_xs_data import ColumnProvenance, PIT_NONE, PIT_TRUE

    if cfg.data.source == "synthetic":
        return [ColumnProvenance(c, "synthetic", "synthetic", PIT_NONE, True,
                                 "Детерминированная синтетика (демо, не реальные данные).")
                for c in panel.columns]
    return [ColumnProvenance(c, "byo:parquet", "byo", PIT_TRUE, False, "BYO price files.")
            for c in panel.columns]


def _panel_with_provenance(cfg: XSConfig):
    """(panel, provenance) для любого источника: free → assembler, иначе load + провенанс по типу."""
    if cfg.data.source == "free":
        res = assemble_free(cfg)
        return res.panel, list(res.report.columns)
    panel = load_panel(cfg)
    return panel, _provenance_for(cfg, panel)


def data_quality_for_config(cfg: XSConfig):
    """DataQualityReport для любого источника (free через assembler; synthetic/parquet — провенанс по типу)."""
    from service_xs_data import build_quality_report

    if cfg.data.source == "free":
        return assemble_free(cfg).report
    panel = load_panel(cfg)
    prov = _provenance_for(cfg, panel)
    return build_quality_report(panel, prov, price_col=cfg.backtest.price_col, survivorship_biased=None)


def data_trust_for_config(cfg: XSConfig) -> Dict[str, Any]:
    """Data-Trust отчёт (Stage D7): DataQualityReport + PIT-lineage сигналов + trust_verdict."""
    from service_data_quality import data_trust_report

    panel, prov = _panel_with_provenance(cfg)
    lib = build_signal_library(cfg)
    return data_trust_report(panel, prov, signal_library=lib, price_col=cfg.backtest.price_col)


# ---------------------------------------------------------------------------
# High-level run
# ---------------------------------------------------------------------------
def run_backtest(cfg: XSConfig, panel: Optional[Panel] = None) -> Dict[str, Any]:
    """Полный прогон из конфига → result + Trust Report + Data-Trust gate (Stage D7)."""
    provenance = None
    if panel is None:
        panel, provenance = _panel_with_provenance(cfg)
    # Honesty guard (P0 #2): synthetic data must NEVER be presented as a real edge.
    # An unaware user running the default `source: synthetic` would otherwise see a
    # fabricated Sharpe. We stamp a loud flag on the result and log a warning so the
    # API / Trust Report / UI can mark it as not-real.
    is_synthetic = (str(getattr(cfg.data, "source", "")).lower() == "synthetic")
    if is_synthetic:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "XS backtest is running on SYNTHETIC data (source=synthetic): metrics are "
            "NOT a real edge and must not be interpreted as performance. Set data.source "
            "to 'free' or 'parquet' with real data for trustworthy results."
        )
    risk_model = _build_run_risk_model(cfg, panel)
    lib = build_signal_library(cfg)
    bt = CrossSectionalBacktest(
        universe=build_universe(cfg),
        alpha_model=build_alpha(cfg),
        risk_model=risk_model,
        optimizer=build_optimizer(cfg),
        signal_library=lib,
        config=build_backtest_config(cfg),
    )
    res = bt.run(panel)
    # Capacity (P1 #12): AUM→Sharpe decay via √participation impact — now actually
    # computed and folded into the Trust Report (was a no-op pass-through).
    capacity = None
    try:
        if getattr(cfg, "capacity", None) is not None and cfg.capacity.enabled and res.weights.shape[0] >= 2:
            from impl_capacity import capacity_from_result
            capacity = capacity_from_result(
                res, adv_usd=cfg.capacity.adv_usd, aum_grid=cfg.capacity.aum_grid,
                impact_coef=cfg.capacity.impact_coef,
                periods_per_year=cfg.backtest.periods_per_year,
                sharpe_threshold_frac=cfg.capacity.sharpe_threshold_frac,
            )
    except Exception as exc:  # pragma: no cover - capacity is best-effort
        import logging as _lg
        _lg.getLogger(__name__).debug("capacity computation skipped: %s", exc)
        capacity = None
    trust = trust_report(
        res.returns, n_trials=cfg.n_trials, periods_per_year=cfg.backtest.periods_per_year,
        capacity=capacity,
    )
    # Data-Trust gate: предупреждаем, если backtested-сигнал зависит от pit_quality=none колонки
    data_trust = None
    try:
        from service_data_quality import data_trust_report
        if provenance is None:
            provenance = _provenance_for(cfg, panel)
        dt = data_trust_report(panel, provenance, signal_library=lib, price_col=cfg.backtest.price_col)
        data_trust = {"trust_verdict": dt["trust_verdict"], "pit_violations": dt["pit_violations"],
                      "used_worst_pit": dt["used_worst_pit"], "signal_lineage": dt["signal_lineage"]}
        if dt["pit_violations"]:
            import logging
            logging.getLogger(__name__).warning(
                "Data-Trust: backtested-сигналы зависят от pit_quality=none колонок: %s — НЕ backtest-safe!",
                dt["pit_violations"],
            )
    except Exception as exc:  # pragma: no cover
        import logging
        logging.getLogger(__name__).debug("data_trust gate skipped: %s", exc)
    # Per-name P&L attribution (Σ_t w[t,s]·fwd_return[t,s]); additive, never breaks backtest.
    attribution = None
    attribution_ts = None
    try:
        from core_portfolio import SYMBOL_LEVEL
        pw = panel[cfg.backtest.price_col].unstack(SYMBOL_LEVEL).sort_index()
        pidx = list(pw.index)
        pos_of = {t: i for i, t in enumerate(pidx)}
        W = res.weights
        syms = list(W.columns)
        totals = {s: 0.0 for s in syms}
        series_by = {s: [] for s in syms}
        for t in W.index:
            i = pos_of.get(t)
            if i is None or i + 1 >= len(pidx):
                for s in syms:
                    series_by[s].append(0.0)
                continue
            p0 = pw.loc[pidx[i]]
            p1 = pw.loc[pidx[i + 1]]
            for s in syms:
                fwd = 0.0
                if s in pw.columns:
                    a = float(p0.get(s, float("nan")))
                    b = float(p1.get(s, float("nan")))
                    if np.isfinite(a) and np.isfinite(b) and a != 0.0:
                        fwd = b / a - 1.0
                c = float(W.loc[t, s]) * fwd
                totals[s] += c
                series_by[s].append(c)
        top = sorted(syms, key=lambda s: abs(totals[s]), reverse=True)[:12]
        attribution = {s: float(totals[s]) for s in top}
        attribution_ts = {
            "ts": [int(t) for t in W.index],
            "by": {s: [float(x) for x in series_by[s]] for s in top},
        }
    except Exception:  # pragma: no cover - attribution is best-effort
        attribution = None
        attribution_ts = None

    # Factor P&L attribution (P1 #11): decompose realized P&L against the SAME
    # risk-model factors the optimizer traded against (not an ad-hoc OLS). We fit
    # the configured risk model on full-panel returns to get exposures B, then run
    # the exact r = B·f + u decomposition (Σ factor + specific = total).
    factor_attribution_out = None
    try:
        from core_portfolio import SYMBOL_LEVEL as _SYM
        from service_attribution import factor_attribution as _fattr
        W = res.weights
        if W.shape[0] >= 1:
            pw = panel[cfg.backtest.price_col].unstack(_SYM).sort_index()
            pidx = list(pw.index)
            pos_of = {t: i for i, t in enumerate(pidx)}
            ar_rows = {}
            for t in W.index:
                i = pos_of.get(t)
                if i is None or i + 1 >= len(pidx):
                    continue
                ar_rows[int(t)] = (pw.loc[pidx[i + 1]] / pw.loc[pidx[i]] - 1.0)
            asset_returns = pd.DataFrame(ar_rows).T
            full_ret = pw.pct_change().dropna(how="all")
            risk_model.fit(full_ret)                 # same model class/params as backtest
            B = risk_model.exposures()               # exposures the optimizer used
            fa = _fattr(W, asset_returns, B)
            fa.pop("per_period", None)               # drop DataFrame for JSON
            fa["risk_model"] = cfg.risk.type         # provenance: which model produced B
            fa["factors"] = list(B.columns)
            factor_attribution_out = fa
    except Exception as exc:  # pragma: no cover - factor attribution is best-effort
        import logging as _lg
        _lg.getLogger(__name__).debug("factor attribution skipped: %s", exc)
        factor_attribution_out = None

    return {
        "result": res,
        "summary": res.summary(),
        "trust_report": trust,
        "data_trust": data_trust,
        # OOS per-period return path (for CPCV/PBO matrices across sweep variants, P1 #8)
        "returns": [float(x) for x in np.asarray(res.returns, dtype="float64").ravel()],
        "n_rebalances": int(res.weights.shape[0]),
        "attribution": attribution,
        "attribution_ts": attribution_ts,
        "factor_attribution": factor_attribution_out,
        "capacity": capacity,
        # Honesty flags (P0 #2): never let synthetic pass as real.
        "data_source": str(getattr(cfg.data, "source", "")),
        "is_synthetic": is_synthetic,
        "real_data": not is_synthetic,
        "warning": (
            "SYNTHETIC DATA — results are NOT a real edge; demo only."
            if is_synthetic else None
        ),
    }


def latest_target_weights(cfg: XSConfig, panel: Optional[Panel] = None) -> pd.Series:
    """Целевые веса последнего ребаланса (для live)."""
    out = run_backtest(cfg, panel)
    w = out["result"].weights
    if w.shape[0] == 0:
        return pd.Series(dtype="float64")
    return w.iloc[-1].dropna()


def load_config_dict(data: Dict[str, Any]) -> XSConfig:
    return XSConfig.model_validate(data)


__all__ = [
    "DataCfg", "UniverseCfg", "SignalCfg", "AlphaCfg", "RiskCfg", "OptimizerCfg",
    "BacktestCfg", "RLCfg", "XSConfig",
    "build_universe", "build_signal_library", "build_alpha", "build_risk_model",
    "build_crypto_factor_risk", "build_equity_factor_risk", "build_futures_factor_risk",
    "build_forex_factor_risk",
    "build_optimizer", "build_backtest_config", "load_panel", "run_backtest",
    "latest_target_weights", "load_config_dict",
    "build_enrichers", "assemble_free", "data_quality_for_config", "data_trust_for_config",
]
