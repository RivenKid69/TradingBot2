# -*- coding: utf-8 -*-
"""
loaders/options_enrich.py
=========================

Options free-обогатители + загрузчик бука (Stage D5) — «оживляют» VRP/skew/term сигналы и
строят реальный бук для greeks-конструктора:

  * ``IVSummaryEnricher`` — IV-сводка по андерлаю (``iv`` ATM, ``skew``, ``term_slope``)
    as-of (publish-lag). База для:
      − ``DeribitIVEnricher`` — free крипто-опционы (Deribit IV/DVOL), history → ``pit='approx'``;
      − ``YFinanceChainEnricher`` — EOD US chains, СНИМОК → ``pit='none'`` (НЕ backtest-safe).
  * ``RealizedVolEnricher`` — реализованная волатильность андерлая из close (PIT-true) →
    ``realized_vol`` (нужна VRP = IV − RV).
  * ``OptionsBookLoader`` — option chain → ``List[OptionLeg]`` для
    ``service_options_portfolio`` (greeks-нейтральный конструктор с реальными IV).

Провайдеры — DI (дефолт = deribit/yfinance, тесты = фейки без сети; сбои → graceful NaN).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL, TS_LEVEL
from core_xs_data import PIT_APPROX, PIT_NONE, PIT_TRUE
from impl_data_sources import DataSourceMeta
from service_xs_data import AsofEnricher

logger = logging.getLogger(__name__)

DEFAULT_IV_COLS = ("iv", "skew", "term_slope")


# ---------------------------------------------------------------------------
# IV summary enrichers
# ---------------------------------------------------------------------------
class IVSummaryEnricher(AsofEnricher):
    """IV-сводка (iv/skew/term_slope) по андерлаю as-of (publish-lag)."""

    def __init__(self, *, provider: Callable[[Sequence[str]], pd.DataFrame],
                 columns: Sequence[str] = DEFAULT_IV_COLS, publish_lag_ms: int = 0,
                 vendor: str = "byo", pit_quality: str = PIT_APPROX, name: str = "iv") -> None:
        super().__init__(
            provider, columns=list(columns), publish_ts_col="publish_ts",
            publish_lag_ms=publish_lag_ms,
            meta=DataSourceMeta(name=name, vendor=vendor, kind="enrich", pit_quality=pit_quality,
                                notes="Implied-vol summary (iv/skew/term)."),
        )


class DeribitIVEnricher(IVSummaryEnricher):
    """Free крипто-опционы (Deribit IV/DVOL). History → pit=approx."""

    def __init__(self, *, provider: Optional[Callable[[Sequence[str]], pd.DataFrame]] = None,
                 publish_lag_ms: int = 0, pit_quality: str = PIT_APPROX) -> None:
        super().__init__(provider=provider or _default_deribit_iv, columns=DEFAULT_IV_COLS,
                         publish_lag_ms=publish_lag_ms, vendor="deribit", pit_quality=pit_quality,
                         name="deribit:iv")


class YFinanceChainEnricher(IVSummaryEnricher):
    """EOD US chains (yfinance). СНИМОК → pit=none (НЕ backtest-safe, live-screening)."""

    def __init__(self, *, provider: Optional[Callable[[Sequence[str]], pd.DataFrame]] = None) -> None:
        super().__init__(provider=provider or _default_yfinance_iv, columns=DEFAULT_IV_COLS,
                         vendor="yfinance", pit_quality=PIT_NONE, name="yfinance:chain")


def _default_deribit_iv(symbols: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - сеть
    logger.warning("DeribitIVEnricher: live provider не настроен (BYO/DI) → пусто")
    return pd.DataFrame(columns=["publish_ts", "symbol", *DEFAULT_IV_COLS])


def _default_yfinance_iv(symbols: Sequence[str]) -> pd.DataFrame:  # pragma: no cover - сеть
    logger.warning("YFinanceChainEnricher: live provider не настроен (BYO/DI) → пусто")
    return pd.DataFrame(columns=["publish_ts", "symbol", *DEFAULT_IV_COLS])


# ---------------------------------------------------------------------------
# Realized vol (для VRP = IV − RV)
# ---------------------------------------------------------------------------
class RealizedVolEnricher:
    """Реализованная волатильность андерлая из close (annualized) → ``realized_vol`` (PIT-true)."""

    def __init__(self, *, window: int = 20, periods_per_year: float = 365.0,
                 close_col: str = "close", out_col: str = "realized_vol") -> None:
        self.window = int(window)
        self.ppy = float(periods_per_year)
        self.close_col = close_col
        self.out_col = out_col
        self.meta = DataSourceMeta(name="realized_vol", vendor="computed", kind="enrich",
                                   pit_quality=PIT_TRUE, notes="Annualized realized vol from close.")

    def columns(self) -> List[str]:
        return [self.out_col]

    def enrich(self, panel: Panel) -> Panel:
        out = panel.copy()
        if self.close_col not in out.columns:
            out[self.out_col] = np.nan
            return out
        rv = pd.Series(np.nan, index=out.index, dtype="float64")
        scale = float(np.sqrt(self.ppy))
        for sym, g in out.groupby(level=SYMBOL_LEVEL, sort=False):
            ret = g[self.close_col].astype("float64").pct_change()
            vol = ret.rolling(self.window).std() * scale
            rv.loc[g.index] = vol.to_numpy()
        out[self.out_col] = rv
        return out


# ---------------------------------------------------------------------------
# Option chain → legs
# ---------------------------------------------------------------------------
class OptionsBookLoader:
    """Option chain → ``List[OptionLeg]`` для greeks-конструктора (реальные IV)."""

    @staticmethod
    def chain_to_legs(chain: Sequence[Mapping[str, Any]], *, spot: float,
                      rate: float = 0.0, dividend_yield: float = 0.0,
                      multiplier: float = 100.0, default_alpha: float = 0.0) -> List[Any]:
        from service_options_portfolio import OptionLeg

        legs: List[Any] = []
        for i, row in enumerate(chain or []):
            tte = row.get("time_to_expiry")
            if tte is None and row.get("expiry_days") is not None:
                tte = float(row["expiry_days"]) / 365.0
            strike = row.get("strike")
            iv = row.get("iv")
            if strike is None or iv is None or tte is None:
                continue
            is_call = bool(row.get("is_call", True))
            sym = row.get("symbol") or f"OPT_{i}_{'C' if is_call else 'P'}{int(round(float(strike)))}"
            legs.append(OptionLeg(
                symbol=str(sym), spot=float(row.get("spot", spot)), strike=float(strike),
                time_to_expiry=float(tte), iv=float(iv), is_call=is_call,
                rate=float(row.get("rate", rate)), dividend_yield=float(row.get("dividend_yield", dividend_yield)),
                alpha=float(row.get("alpha", default_alpha)), multiplier=float(row.get("multiplier", multiplier)),
            ))
        return legs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
OPTIONS_ENRICHERS = ("iv", "realized_vol")


def build_options_enricher(name: str, cfg: Any) -> Optional[Any]:
    """Сконструировать options-обогатитель по имени (из cfg)."""
    if name == "realized_vol":
        return RealizedVolEnricher()
    if name == "iv":
        vendor = str(getattr(cfg, "iv_vendor", "deribit") or "deribit").lower()
        if vendor == "yfinance":
            return YFinanceChainEnricher()
        return DeribitIVEnricher()
    return None


__all__ = [
    "DEFAULT_IV_COLS", "IVSummaryEnricher", "DeribitIVEnricher", "YFinanceChainEnricher",
    "RealizedVolEnricher", "OptionsBookLoader",
    "OPTIONS_ENRICHERS", "build_options_enricher",
]
