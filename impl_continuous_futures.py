# -*- coding: utf-8 -*-
"""
impl_continuous_futures.py
==========================

Back-adjusted **непрерывные** фьючерсные серии (Stage B3) для cross-sectional контура.

Сырые фьючерсы экспирируются → наивная склейка контрактов даёт ИСКУССТВЕННЫЙ скачок цены
на дате ролла (разный уровень соседних контрактов из-за carry/контанго). CTA-портфели
торгуют **непрерывные back-adjusted серии** (Panama-canal): исторические сегменты
сдвигаются так, чтобы на роллах не было разрыва, а последний сегмент = реальные цены.

Два метода (как в индустрии):
  * ``ratio`` (пропорциональный, по умолчанию) — умножает историю на произведение будущих
    roll-факторов; **точно сохраняет доходности** (предпочтительно для возвратных серий);
  * ``diff`` (Panama/аддитивный) — прибавляет к истории сумму будущих gap'ов; сохраняет
    **уровневые приращения** (может дать отрицательные цены на длинной истории).

Опирается на тот же принцип, что и ``impl_cme_rollover.ContractRolloverManager``
(``adjustment_factor``/``cumulative_adjustment``), но векторизован и панель-дружелюбен.
Слой ``impl_`` (зависит от ``core_portfolio``/``impl_panel``). Сетевых вызовов нет.

Honest-note: бесплатные continuous-серии (yahoo ``ES=F`` / stooq) — это УЖЕ back-adjusted
прокси неизвестным методом → ``pit_quality='approx'`` (метод ролла непрозрачен). Для
точных roll-accurate серий подайте BYO контрактные данные + расписание роллов.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core_portfolio import Panel
from impl_panel import PanelBuilder

logger = logging.getLogger(__name__)

VALID_METHODS = ("ratio", "diff")

# Roll-событие: (roll_ts_ms, gap) где
#   diff:  gap = new_contract_price - old_contract_price (на момент ролла)
#   ratio: gap = new_contract_price / old_contract_price (на момент ролла)
RollEvent = Tuple[int, float]


@dataclass(frozen=True)
class ContinuousMeta:
    """Метаданные непрерывной серии — для UI/логов (continuous-contract индикатор)."""

    symbol: str
    method: str = "ratio"
    pit_quality: str = "approx"  # back-adjusted прокси непрозрачен; BYO → 'true'
    n_rolls: int = 0
    source: str = "byo"

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "method": self.method,
            "pit_quality": self.pit_quality,
            "n_rolls": self.n_rolls,
            "source": self.source,
        }


def back_adjust(
    prices: pd.Series,
    rolls: Sequence[RollEvent],
    *,
    method: str = "ratio",
) -> pd.Series:
    """Back-adjust непрерывной серии (последний сегмент не трогаем = реальные цены).

    ``prices`` индексируется по ts_ms (возрастание). ``rolls`` — список (roll_ts, gap).
    Точка в момент ``t`` корректируется накоплением ВСЕХ будущих роллов (``roll_ts > t``):
      * ratio: ``price * Π gap_future``;
      * diff:  ``price + Σ gap_future``.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    s = prices.sort_index().astype("float64")
    if s.empty or not rolls:
        return s.rename(prices.name)
    ts = s.index.to_numpy()
    if method == "ratio":
        adj = np.ones(len(s), dtype="float64")
        for rts, gap in rolls:
            adj[ts < int(rts)] *= float(gap)
        out = s.to_numpy() * adj
    else:  # diff
        adj = np.zeros(len(s), dtype="float64")
        for rts, gap in rolls:
            adj[ts < int(rts)] += float(gap)
        out = s.to_numpy() + adj
    return pd.Series(out, index=s.index, name=prices.name)


def roll_events_from_overlap(
    old_close: pd.Series,
    new_close: pd.Series,
    roll_ts: int,
    *,
    method: str = "ratio",
) -> RollEvent:
    """Вычислить gap из ПЕРЕКРЫТИЯ старого/нового контракта на дату ролла.

    Берёт цены обоих контрактов на ``roll_ts`` (или ближайшую общую ранее) → gap.
    """
    common = old_close.index.intersection(new_close.index)
    common = common[common <= int(roll_ts)]
    if len(common) == 0:
        return (int(roll_ts), 1.0 if method == "ratio" else 0.0)
    at = common.max()
    old_p = float(old_close.loc[at])
    new_p = float(new_close.loc[at])
    if method == "ratio":
        gap = (new_p / old_p) if old_p not in (0.0,) and np.isfinite(old_p) and old_p != 0 else 1.0
    else:
        gap = new_p - old_p
    return (int(roll_ts), float(gap))


def stitch_contracts(
    contracts: Sequence[Tuple[int, pd.Series]],
    *,
    method: str = "ratio",
) -> Tuple[pd.Series, List[RollEvent]]:
    """Склеить упорядоченные контракты в back-adjusted непрерывную серию.

    ``contracts`` — список ``(roll_ts, close_series)`` по возрастанию: контракт i активен
    до ``roll_ts_{i+1}``. Внутри строит roll-события из перекрытий и применяет back_adjust.
    Возвращает (continuous_close, rolls).
    """
    contracts = list(contracts)
    if not contracts:
        return pd.Series(dtype="float64"), []
    # сырой стич: каждый контракт активен в [own_roll_ts, next_roll_ts) — БЕЗ перекрытия
    # (граница сегмента ровно = ключ gap'а, иначе перекрытие даёт двойную корректировку).
    segments: List[pd.Series] = []
    rolls: List[RollEvent] = []
    for i, (rts, close) in enumerate(contracts):
        nxt = contracts[i + 1][0] if i + 1 < len(contracts) else None
        seg = close.sort_index()
        seg = seg[seg.index >= int(rts)]
        if nxt is not None:
            seg = seg[seg.index < int(nxt)]
            ev = roll_events_from_overlap(
                contracts[i][1], contracts[i + 1][1], int(nxt), method=method
            )
            rolls.append(ev)
        segments.append(seg)
    raw = pd.concat(segments).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    cont = back_adjust(raw, rolls, method=method)
    return cont, rolls


def build_continuous_panel(
    contract_map: Dict[str, Sequence[Tuple[int, pd.Series]]],
    *,
    method: str = "ratio",
    pit_quality: str = "true",
) -> Tuple[Panel, Dict[str, ContinuousMeta]]:
    """Построить Panel непрерывных закрытий из BYO контрактных данных по символам.

    ``contract_map``: ``{symbol -> [(roll_ts, close_series), ...]}``. Возвращает (Panel, meta).
    """
    frames: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, ContinuousMeta] = {}
    for sym, contracts in contract_map.items():
        cont, rolls = stitch_contracts(contracts, method=method)
        frames[sym] = pd.DataFrame(
            {
                "timestamp": cont.index.to_numpy(),
                "symbol": sym,
                "close": cont.to_numpy(),
            }
        )
        metas[sym] = ContinuousMeta(
            symbol=sym, method=method, pit_quality=pit_quality, n_rolls=len(rolls), source="byo"
        )
    panel = PanelBuilder.from_frames(frames)
    return panel, metas


# ---------------------------------------------------------------------------
# Synthetic (demo / tests / no-data smoke)
# ---------------------------------------------------------------------------
def synthetic_continuous_frames(
    symbols: Sequence[str],
    *,
    n_bars: int = 260,
    seed: int = 17,
    roll_every: int = 60,
    carry_offset: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """Синтетические УЖЕ-непрерывные closes (для пресета без данных).

    Возвращает ``{symbol -> price_frame}`` с трендовыми путями (разные дрейфы → диверсификация).
    """
    rng = np.random.default_rng(seed)
    t0, step = 1_600_000_000, 86_400
    ts = [t0 + i * step for i in range(n_bars)]
    frames: Dict[str, pd.DataFrame] = {}
    for k, s in enumerate(symbols):
        drift = 0.0004 * ((k % 5) - 2)  # разнознаковые тренды
        vol = 0.008 + 0.004 * (k % 3)
        r = rng.normal(drift, vol, n_bars)
        close = 100.0 * np.cumprod(1.0 + r)
        frames[s] = pd.DataFrame({"timestamp": ts, "symbol": s, "close": close})
    return frames


__all__ = [
    "VALID_METHODS",
    "RollEvent",
    "ContinuousMeta",
    "back_adjust",
    "roll_events_from_overlap",
    "stitch_contracts",
    "build_continuous_panel",
    "synthetic_continuous_frames",
]
