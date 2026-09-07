# -*- coding: utf-8 -*-
"""
service_cross_asset.py
======================

Unified cross-asset portfolio (Stage C1) — ОДИН портфель поверх всех directional вертикалей
(crypto + equity + futures + forex одновременно; options остаются отдельной greeks-машинерией).

Слои объединения:
  1. **Валютная нормализация** — локальные доходности каждого класса → базовую валюту:
        r_base = (1 + r_local)(1 + r_fx) − 1   (r_fx = доходность валюты котировки vs base);
  2. **Кросс-asset ковариация** — стек всех base-доходностей в одну широкую матрицу → joint Σ
        (Ledoit-Wolf через ``StatRiskModel`` → гарантированно симметрична и **PSD**);
        захватывает кросс-asset корреляции (то, чего нет в per-class Σ);
  3. **Верхний risk-parity между классами** — аллокация риск-бюджета обратно к vol класса
        (a_c ∝ 1/vol_c), внутри класса — веса вертикали; combined w = Σ_c a_c · w_c;
  4. **Общий vol-target** — масштаб всех весов к целевой годовой волатильности портфеля.

Интеграция: поверх `service_xs_pipeline` (вертикали → веса + панель) и совместимо с
`services.unified_futures_risk.PortfolioRiskManager` (live-гарды). Слой ``service_``.

Honest-note: кросс-asset корреляции из бесплатных/synthetic данных приблизительны
(разные сессии/часовые пояса, неполный overlap) — joint Σ настолько честна, насколько
честны входные ряды; для проды подайте выровненные BYO-данные.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------
def normalize_returns_to_base(
    returns_wide: pd.DataFrame,
    *,
    currency_map: Optional[Mapping[str, str]] = None,
    fx_returns: Optional[Mapping[str, pd.Series]] = None,
    base: str = "USD",
) -> pd.DataFrame:
    """Локальные доходности → базовую валюту: r_base = (1+r_local)(1+r_fx) − 1.

    ``currency_map``: symbol → валюта котировки (по умолчанию все base).
    ``fx_returns``: валюта → ряд её доходности vs base (по тем же ts). base/неизвестная → 0.
    """
    if not currency_map or not fx_returns:
        return returns_wide.copy()
    out = returns_wide.copy()
    for sym in out.columns:
        cur = currency_map.get(sym, base)
        if cur == base or cur not in fx_returns:
            continue
        fx = fx_returns[cur].reindex(out.index).fillna(0.0).astype("float64")
        out[sym] = (1.0 + out[sym].astype("float64")) * (1.0 + fx) - 1.0
    return out


# ---------------------------------------------------------------------------
# Asset-class block
# ---------------------------------------------------------------------------
@dataclass
class AssetClassBlock:
    """Блок одного класса активов для объединения."""

    name: str  # 'crypto' | 'equity' | 'futures' | 'forex'
    weights: pd.Series  # within-class веса (index=symbol)
    returns_wide: pd.DataFrame  # index=ts, cols=symbol (ЛОКАЛЬНАЯ валюта)
    currency_map: Optional[Dict[str, str]] = None

    def normalized_weights(self) -> pd.Series:
        w = self.weights.astype("float64").dropna()
        g = float(w.abs().sum())
        return w / g if g > 0 else w


@dataclass
class CrossAssetResult:
    """Результат объединения."""

    weights: pd.Series  # унифицированные веса (все символы)
    class_allocations: Dict[str, float]  # риск-аллокация по классам (Σ=1)
    cov: pd.DataFrame  # joint Σ (PSD)
    port_vol_annual: float  # достигнутая годовая vol
    target_vol: float
    base: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "class_allocations": {k: float(v) for k, v in self.class_allocations.items()},
            "gross": float(self.weights.abs().sum()),
            "net": float(self.weights.sum()),
            "port_vol_annual": float(self.port_vol_annual),
            "target_vol": float(self.target_vol),
            "base": self.base,
            "n_names": int(len(self.weights)),
        }


# ---------------------------------------------------------------------------
# Joint covariance
# ---------------------------------------------------------------------------
def build_cross_asset_cov(
    blocks: Sequence[AssetClassBlock],
    *,
    fx_returns: Optional[Mapping[str, pd.Series]] = None,
    base: str = "USD",
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Joint Σ по всем классам (base-валюта, Ledoit-Wolf → PSD)."""
    from service_risk_model import StatRiskModel

    cols: List[pd.DataFrame] = []
    for b in blocks:
        rb = normalize_returns_to_base(
            b.returns_wide, currency_map=b.currency_map, fx_returns=fx_returns, base=base
        )
        cols.append(rb.dropna(how="all"))
    # выравниваем по общим ts (intersection) — избегаем NaN в ковариации
    joint = pd.concat(cols, axis=1, join="inner").dropna(how="any")
    if len(joint) < 10:
        # honest fallback: разные эпохи/сессии free-данных не пересекаются по ts →
        # выравниваем по ПОЗИЦИИ (последние m общих баров). Для проды подайте выровненные данные.
        m = min((len(c) for c in cols), default=0)
        if m >= 2:
            aligned = [c.iloc[-m:].reset_index(drop=True) for c in cols]
            joint = pd.concat(aligned, axis=1).dropna(how="any")
            logger.warning(
                "cross-asset: ts не пересекаются → позиционное выравнивание (%d баров)", m
            )
    # дубли символов (если есть) — оставляем первый
    joint = joint.loc[:, ~joint.columns.duplicated()]
    rm = StatRiskModel(method=method).fit(joint)
    return rm.cov()


def _annual_vol(weights: pd.Series, cov: pd.DataFrame, periods_per_year: float) -> float:
    syms = [s for s in weights.index if s in cov.index]
    if not syms:
        return 0.0
    w = weights.reindex(syms).fillna(0.0).to_numpy()
    S = cov.reindex(index=syms, columns=syms).fillna(0.0).to_numpy()
    var = float(w @ S @ w)
    var = max(var, 0.0)
    return float(np.sqrt(var) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_cross_asset(
    blocks: Sequence[AssetClassBlock],
    *,
    target_vol: float = 0.10,
    periods_per_year: float = 252.0,
    fx_returns: Optional[Mapping[str, pd.Series]] = None,
    base: str = "USD",
    method: str = "ledoit_wolf",
    class_weighting: str = "risk_parity",  # 'risk_parity' | 'equal'
) -> CrossAssetResult:
    """Объединить ≥1 классов в один портфель: класс-risk-parity + общий vol-target."""
    blocks = [b for b in blocks if len(b.normalized_weights()) > 0]
    if not blocks:
        return CrossAssetResult(
            pd.Series(dtype="float64"), {}, pd.DataFrame(), 0.0, target_vol, base
        )

    cov = build_cross_asset_cov(blocks, fx_returns=fx_returns, base=base, method=method)

    # риск каждого класса (на его within-class весах)
    class_vol: Dict[str, float] = {}
    for b in blocks:
        class_vol[b.name] = _annual_vol(b.normalized_weights(), cov, periods_per_year)

    # верхняя аллокация между классами
    if class_weighting == "equal":
        alloc = {b.name: 1.0 / len(blocks) for b in blocks}
    else:  # risk_parity: обратно к vol класса (inverse-vol)
        inv = {n: (1.0 / v if v > 1e-12 else 0.0) for n, v in class_vol.items()}
        tot = sum(inv.values())
        alloc = {n: (inv[n] / tot if tot > 0 else 1.0 / len(blocks)) for n in inv}

    # combined веса: a_c · w_c, агрегируем по символам
    combined: Dict[str, float] = {}
    for b in blocks:
        a = alloc[b.name]
        for s, w in b.normalized_weights().items():
            combined[s] = combined.get(s, 0.0) + a * float(w)
    w = pd.Series(combined, name="weight").astype("float64")

    # общий vol-target
    pv = _annual_vol(w, cov, periods_per_year)
    if pv > 1e-12:
        w = w * (target_vol / pv)
    pv_after = _annual_vol(w, cov, periods_per_year)

    return CrossAssetResult(
        weights=w,
        class_allocations=alloc,
        cov=cov,
        port_vol_annual=pv_after,
        target_vol=target_vol,
        base=base,
    )


# ---------------------------------------------------------------------------
# Convenience: block from an XS vertical config
# ---------------------------------------------------------------------------
def block_from_xs_config(cfg, *, name: Optional[str] = None) -> AssetClassBlock:
    """Прогнать directional-вертикаль (crypto/equity/futures/forex) → AssetClassBlock.

    Берёт последние целевые веса + локальные доходности из загруженной панели.
    Options не directional — для C1 не используется.
    """
    from service_xs_pipeline import load_panel, latest_target_weights
    from core_portfolio import SYMBOL_LEVEL

    panel = load_panel(cfg)
    weights = latest_target_weights(cfg, panel)
    price = panel[cfg.backtest.price_col].unstack(SYMBOL_LEVEL).sort_index()
    returns_wide = price.pct_change().dropna(how="all")
    return AssetClassBlock(
        name=name or cfg.asset_class,
        weights=weights,
        returns_wide=returns_wide,
    )


def combine_from_configs(
    configs: Mapping[str, object],
    *,
    target_vol: float = 0.10,
    periods_per_year: float = 252.0,
    class_weighting: str = "risk_parity",
    base: str = "USD",
) -> CrossAssetResult:
    """Высокоуровнево: {class_name -> XSConfig} → единый cross-asset портфель."""
    blocks = [block_from_xs_config(cfg, name=name) for name, cfg in configs.items()]
    return combine_cross_asset(
        blocks,
        target_vol=target_vol,
        periods_per_year=periods_per_year,
        base=base,
        class_weighting=class_weighting,
    )


__all__ = [
    "normalize_returns_to_base",
    "AssetClassBlock",
    "CrossAssetResult",
    "build_cross_asset_cov",
    "combine_cross_asset",
    "block_from_xs_config",
    "combine_from_configs",
]
