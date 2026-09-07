# -*- coding: utf-8 -*-
"""
signals/common_signals.py
=========================

Asset-agnostic расширение каталога сигналов (P2): residual momentum, seasonality,
sentiment (alt-data BYO), 52-week-high, idiosyncratic-vol. Работают на любом классе
(нужна цена + опц. BYO-колонка), регистрируются как ``COMMON_SIGNAL_KINDS`` в пайплайне.

Все — ``BaseSignal`` (``compute_panel(panel) -> Series`` по индексу (ts, symbol)).
PIT-безопасны: используют только прошлое (rolling/expanding + shift нормализуется в pipeline).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL, Panel
from service_signals import BaseSignal


def _nan(panel: Panel, name: str) -> pd.Series:
    return pd.Series(np.nan, index=panel.index, name=name)


def _wide(panel: Panel, col: str) -> Optional[pd.DataFrame]:
    if col not in panel.columns:
        return None
    return panel[col].astype("float64").unstack(level=SYMBOL_LEVEL).sort_index()


def _restack(wide: pd.DataFrame, panel: Panel, name: str) -> pd.Series:
    # pandas 3.0: stack() сохраняет полную сетку (ts×symbol), без dropna-аргумента
    s = wide.stack()
    s.index = s.index.set_names([TS_LEVEL, SYMBOL_LEVEL])
    return s.reindex(panel.index).rename(name)


class ResidualMomentum(BaseSignal):
    """Residual momentum: momentum остатков после удаления рыночной беты (Blitz et al.).

    market_t = средняя cross-sectional доходность; β_i — rolling cov(r_i, mkt)/var(mkt);
    residual_i = r_i − β_i·mkt; сигнал = накопленный residual за ``lookback`` (пропуск ``skip``).
    """

    def __init__(
        self,
        name: str = "resid_mom",
        *,
        lookback: int = 252,
        skip: int = 21,
        beta_window: int = 60,
        price_col: str = "close",
    ) -> None:
        self.name = name
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.beta_window = int(beta_window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        close = _wide(panel, self.price_col)
        if close is None or close.shape[1] < 2:
            return _nan(panel, self.name)
        ret = close.pct_change()
        mkt = ret.mean(axis=1)
        w = self.beta_window
        var_mkt = mkt.rolling(w).var()
        # rolling cov(r_i, mkt) = E[r_i·mkt] − E[r_i]·E[mkt]
        cov = (
            ret.mul(mkt, axis=0)
            .rolling(w)
            .mean()
            .sub(ret.rolling(w).mean().mul(mkt.rolling(w).mean(), axis=0))
        )
        beta = cov.div(var_mkt.replace(0.0, np.nan), axis=0)
        resid = ret.sub(beta.mul(mkt, axis=0))
        sig = resid.shift(self.skip).rolling(self.lookback - self.skip).sum()
        return _restack(sig, panel, self.name)


class Seasonality(BaseSignal):
    """Month-of-year seasonality (Heston-Sadka): средняя доходность символа в текущем
    календарном месяце по ПРОШЛЫМ годам (expanding, PIT-safe)."""

    def __init__(self, name: str = "seasonality", *, price_col: str = "close") -> None:
        self.name = name
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        close = _wide(panel, self.price_col)
        if close is None:
            return _nan(panel, self.name)
        ret = close.pct_change()
        months = pd.to_datetime(
            ret.index.get_level_values(0) if isinstance(ret.index, pd.MultiIndex) else ret.index,
            unit="ms",
            errors="coerce",
        ).month
        months = pd.Series(months, index=ret.index)
        out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
        for m in range(1, 13):
            mask = (months == m).to_numpy()
            if not mask.any():
                continue
            sub = ret[mask]
            # expanding mean прошлых значений того же месяца, сдвиг на 1 (без текущего)
            exp_mean = sub.expanding().mean().shift(1)
            out.loc[mask] = exp_mean.to_numpy()
        return _restack(out, panel, self.name)


class Sentiment(BaseSignal):
    """Alt-data sentiment: BYO-колонка (sentiment/social_score). Нет колонки → NaN (нейтрально)."""

    def __init__(self, name: str = "sentiment", *, column: str = "sentiment") -> None:
        self.name = name
        self.column = column

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.column not in panel.columns:
            return _nan(panel, self.name)
        return panel[self.column].astype("float64").rename(self.name)


class Week52High(BaseSignal):
    """52-week-high momentum (George & Hwang): price / rolling-max(window) − 1."""

    def __init__(
        self, name: str = "high_52w", *, window: int = 252, price_col: str = "close"
    ) -> None:
        self.name = name
        self.window = int(window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        close = _wide(panel, self.price_col)
        if close is None:
            return _nan(panel, self.name)
        roll_max = close.rolling(self.window, min_periods=max(2, self.window // 4)).max()
        prox = close.div(roll_max) - 1.0
        return _restack(prox, panel, self.name)


class IdiosyncraticVol(BaseSignal):
    """Low idiosyncratic-vol anomaly: −rolling std остатков (после удаления рыночной беты)."""

    def __init__(
        self, name: str = "idio_vol", *, window: int = 60, price_col: str = "close"
    ) -> None:
        self.name = name
        self.window = int(window)
        self.price_col = price_col

    def compute_panel(self, panel: Panel) -> pd.Series:
        close = _wide(panel, self.price_col)
        if close is None or close.shape[1] < 2:
            return _nan(panel, self.name)
        ret = close.pct_change()
        mkt = ret.mean(axis=1)
        w = self.window
        var_mkt = mkt.rolling(w).var()
        cov = (
            ret.mul(mkt, axis=0)
            .rolling(w)
            .mean()
            .sub(ret.rolling(w).mean().mul(mkt.rolling(w).mean(), axis=0))
        )
        beta = cov.div(var_mkt.replace(0.0, np.nan), axis=0)
        resid = ret.sub(beta.mul(mkt, axis=0))
        idio = resid.rolling(w).std()
        return _restack(-idio, panel, self.name)


class COTPositioning(BaseSignal):
    """COT positioning: чистая позиция крупных спекулянтов (BYO-колонка ``cot_net``;
    через ``loaders.futures_enrich`` enricher 'cot'). Нет колонки → NaN."""

    def __init__(self, name: str = "cot", *, column: str = "cot_net") -> None:
        self.name = name
        self.column = column

    def compute_panel(self, panel: Panel) -> pd.Series:
        if self.column not in panel.columns:
            return _nan(panel, self.name)
        return panel[self.column].astype("float64").rename(self.name)


_KINDS = {
    "residual_momentum": ResidualMomentum,
    "seasonality": Seasonality,
    "sentiment": Sentiment,
    "high_52w": Week52High,
    "idio_vol": IdiosyncraticVol,
    "cot": COTPositioning,
}


def build_common_signal(kind: str, name: str, **kwargs: Any) -> BaseSignal:
    if kind not in _KINDS:
        raise ValueError(f"unknown common signal kind: {kind!r}")
    return _KINDS[kind](name=name, **kwargs)


COMMON_SIGNAL_KINDS = tuple(_KINDS.keys())

__all__ = [
    "ResidualMomentum",
    "Seasonality",
    "Sentiment",
    "Week52High",
    "IdiosyncraticVol",
    "COTPositioning",
    "build_common_signal",
    "COMMON_SIGNAL_KINDS",
]
