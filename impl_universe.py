# -*- coding: utf-8 -*-
"""
impl_universe.py
================

Point-in-time (survivorship-free) состав торгуемого юниверса (Stage A3).

Реализации контракта ``core_portfolio.UniverseProvider`` поверх уже существующего
``services/survivorship.py`` (``UniverseSnapshot`` + ``DelistingTracker``):

* ``StaticUniverse`` (free) — фиксированный список символов. Честно помечен как
  **survivorship-biased** (это «сегодняшний» список без истории членства).
* ``IndexMembershipUniverse`` (BYO) — историческая реконструкция состава индекса на
  любую дату + учёт делистингов. **survivorship-free** (делистнутые-тогда-активные
  тикеры присутствуют в прошлом).
* ``ADVLiquidityFilter`` — обёртка над любым ``UniverseProvider``, отсекающая неликвид
  по trailing ADV (dollar volume) из ``Panel``.

Мост времени: Panel/контракты оперируют ``asof_ms`` (int мс), а survivorship — датами;
``ms_to_date`` конвертирует. Слой ``impl_`` (зависит от ``core_portfolio``); импорт
``services.survivorship`` — ленивый, без disk side-effects (``auto_load=False``).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import SYMBOL_LEVEL, TS_LEVEL, Panel, UniverseProvider, validate_panel

logger = logging.getLogger(__name__)


def ms_to_date(asof_ms: int) -> date:
    """ms (UTC) → календарная дата UTC."""
    return datetime.fromtimestamp(int(asof_ms) / 1000.0, tz=timezone.utc).date()


def date_to_ms(d: Any) -> int:
    """date/str/Timestamp → ms (UTC, начало дня)."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Static universe (free; survivorship-biased)
# ---------------------------------------------------------------------------
class StaticUniverse:
    """Фиксированный список символов (free).

    Не несёт истории членства → honest-флаг ``survivorship_biased=True``. Опциональное
    окно торгуемости ``[tradable_from_ms, tradable_to_ms)`` для грубого ограничения.
    """

    survivorship_biased: bool = True

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        name: str = "static",
        tradable_from_ms: Optional[int] = None,
        tradable_to_ms: Optional[int] = None,
    ) -> None:
        self.name = name
        self._symbols = list(dict.fromkeys(str(s) for s in symbols))  # сохранить порядок, без дублей
        self._set = set(self._symbols)
        self.tradable_from_ms = tradable_from_ms
        self.tradable_to_ms = tradable_to_ms
        logger.info(
            "StaticUniverse '%s': %d symbols (survivorship_biased=True)",
            name, len(self._symbols),
        )

    def _in_window(self, asof_ms: int) -> bool:
        if self.tradable_from_ms is not None and int(asof_ms) < self.tradable_from_ms:
            return False
        if self.tradable_to_ms is not None and int(asof_ms) >= self.tradable_to_ms:
            return False
        return True

    def constituents(self, asof_ms: int) -> List[str]:
        return list(self._symbols) if self._in_window(asof_ms) else []

    def is_tradable(self, symbol: str, asof_ms: int) -> bool:
        return symbol in self._set and self._in_window(asof_ms)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "static",
            "survivorship_biased": self.survivorship_biased,
            "n_symbols": len(self._symbols),
        }


# ---------------------------------------------------------------------------
# Index membership universe (BYO; survivorship-free, PIT)
# ---------------------------------------------------------------------------
class IndexMembershipUniverse:
    """PIT-юниверс по истории членства индекса (BYO) + делистинги.

    Обёртка над ``services.survivorship.UniverseSnapshot`` (реконструкция состава на
    дату) и опциональным ``DelistingTracker``. Делистнутые-позже тикеры остаются в
    прошлом → **survivorship-free**.
    """

    survivorship_biased: bool = False

    def __init__(
        self,
        index: str,
        *,
        snapshot: Any = None,
        delisting: Any = None,
        name: Optional[str] = None,
    ) -> None:
        self.index = str(index)
        self.name = name or f"index:{self.index}"
        if snapshot is None:
            from services.survivorship import UniverseSnapshot

            snapshot = UniverseSnapshot(auto_load=False)
        self._snap = snapshot
        self._delist = delisting
        logger.info(
            "IndexMembershipUniverse '%s' (survivorship_biased=False)", self.name
        )

    # ---- builders / passthrough ----
    @classmethod
    def from_baseline(
        cls,
        index: str,
        constituents: Sequence[str],
        asof: Any,
        *,
        changes: Optional[Sequence[Dict[str, Any]]] = None,
        delistings: Optional[Sequence[Dict[str, Any]]] = None,
        name: Optional[str] = None,
    ) -> "IndexMembershipUniverse":
        """Удобный конструктор: baseline + список изменений + делистинги.

        ``changes``: ``[{date, added=[], removed=[], reason=''}, ...]``.
        ``delistings``: ``[{symbol, delist_date, reason?}, ...]``.
        """
        from services.survivorship import UniverseSnapshot

        snap = UniverseSnapshot(auto_load=False)
        snap.set_baseline(index, list(constituents), asof)
        for ch in changes or []:
            snap.add_change(
                index,
                ch["date"],
                added=ch.get("added"),
                removed=ch.get("removed"),
                reason=ch.get("reason", ""),
            )
        delist = None
        if delistings:
            from services.survivorship import DelistingTracker

            delist = DelistingTracker(auto_load=False)
            for d in delistings:
                delist.add_delisting(
                    d["symbol"], d["delist_date"], reason=d.get("reason", "unknown")
                )
        return cls(index, snapshot=snap, delisting=delist, name=name)

    def add_change(self, date_: Any, added=None, removed=None, reason: str = "") -> None:
        self._snap.add_change(self.index, date_, added=added, removed=removed, reason=reason)

    def add_delisting(self, symbol: str, delist_date: Any, reason: str = "unknown") -> None:
        if self._delist is None:
            from services.survivorship import DelistingTracker

            self._delist = DelistingTracker(auto_load=False)
        self._delist.add_delisting(symbol, delist_date, reason=reason)

    # ---- UniverseProvider ----
    def constituents(self, asof_ms: int) -> List[str]:
        d = ms_to_date(asof_ms)
        try:
            members = self._snap.get_constituents(self.index, d)
        except ValueError:
            logger.warning("IndexMembershipUniverse '%s': no baseline yet", self.name)
            return []
        syms = sorted(members)
        if self._delist is not None:
            syms = [s for s in syms if self._delist.is_tradable(s, d)]
        return syms

    def is_tradable(self, symbol: str, asof_ms: int) -> bool:
        d = ms_to_date(asof_ms)
        try:
            in_index = self._snap.was_constituent(symbol, self.index, d)
        except Exception:
            in_index = True
        if not in_index:
            return False
        if self._delist is not None:
            return bool(self._delist.is_tradable(symbol, d))
        return True

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "index_membership",
            "index": self.index,
            "survivorship_biased": self.survivorship_biased,
            "has_delisting_tracker": self._delist is not None,
        }


# ---------------------------------------------------------------------------
# Liquidity filter (ADV)
# ---------------------------------------------------------------------------
class ADVLiquidityFilter:
    """Обёртка над ``UniverseProvider``, отсекающая неликвид по trailing ADV.

    ADV считается из ``Panel`` (как правило dollar volume = close × volume) на окне
    ``lookback`` баров, заканчивающемся на ``asof_ms``. Сохраняет honest-флаг базового
    провайдера.
    """

    def __init__(
        self,
        base: UniverseProvider,
        panel: Panel,
        *,
        min_adv: float,
        lookback: int = 20,
        dollar_volume: bool = True,
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        validate_panel(panel)
        if volume_col not in panel.columns:
            raise ValueError(f"ADVLiquidityFilter: panel missing volume column '{volume_col}'")
        if dollar_volume and close_col not in panel.columns:
            raise ValueError(f"ADVLiquidityFilter: panel missing close column '{close_col}'")
        self.base = base
        self.panel = panel
        self.min_adv = float(min_adv)
        self.lookback = int(lookback)
        self.dollar_volume = bool(dollar_volume)
        self.close_col = close_col
        self.volume_col = volume_col
        self.survivorship_biased = getattr(base, "survivorship_biased", None)

    def _adv_at(self, asof_ms: int) -> Dict[str, float]:
        p = self.panel
        mask = p.index.get_level_values(TS_LEVEL) <= int(asof_ms)
        sub = p.loc[mask]
        if len(sub) == 0:
            return {}
        if self.dollar_volume:
            dv = sub[self.close_col].astype("float64") * sub[self.volume_col].astype("float64")
        else:
            dv = sub[self.volume_col].astype("float64")
        dv = dv.dropna()
        if len(dv) == 0:
            return {}
        n = self.lookback
        adv = dv.groupby(level=SYMBOL_LEVEL).apply(lambda s: s.tail(n).mean())
        return {str(k): float(v) for k, v in adv.items()}

    def constituents(self, asof_ms: int) -> List[str]:
        base_syms = list(self.base.constituents(asof_ms))
        adv = self._adv_at(asof_ms)
        return [s for s in base_syms if adv.get(s, 0.0) >= self.min_adv]

    def is_tradable(self, symbol: str, asof_ms: int) -> bool:
        if not self.base.is_tradable(symbol, asof_ms):
            return False
        return self._adv_at(asof_ms).get(symbol, 0.0) >= self.min_adv

    def adv(self, asof_ms: int) -> Dict[str, float]:
        """Публичный доступ к посчитанным ADV (для диагностики/UI)."""
        return self._adv_at(asof_ms)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": f"adv_filter({getattr(self.base, 'name', 'base')})",
            "type": "adv_liquidity_filter",
            "survivorship_biased": self.survivorship_biased,
            "min_adv": self.min_adv,
            "lookback": self.lookback,
            "dollar_volume": self.dollar_volume,
        }


__all__ = [
    "ms_to_date",
    "date_to_ms",
    "StaticUniverse",
    "IndexMembershipUniverse",
    "ADVLiquidityFilter",
]
