# -*- coding: utf-8 -*-
"""
service_data_quality.py
=======================

Data-Trust gate (Stage D7) — параллель «Backtest Trust Report», но про ДАННЫЕ: то, что
отличает институционал — PIT-дисциплина и видимость происхождения. Финальный про-штрих
Part D.

Содержит:
  * ``pit_leak_scan`` — строгая проверка анти-look-ahead: ни одно значение колонки не
    появляется в панели РАНЬШЕ своей публикации (``value_ts >= publish_ts``);
  * ``signal_columns`` — lineage: какие колонки панели читает сигнал (по ``*_col`` атрибутам);
  * ``data_trust_report`` — поверх ``DataQualityReport`` (D0): per-signal lineage
    (columns → worst pit_quality → backtest_safe), список **PIT-violations** (backtested-
    сигнал зависит от ``pit_quality=none`` колонки) и **trust_verdict**
    (``trusted|caution|untrusted``).

Слой ``service_`` (зависит от core_/impl_/service_xs_data; НЕ зависит от pipeline → без цикла).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core_portfolio import Panel, SYMBOL_LEVEL, TS_LEVEL
from core_xs_data import PIT_NONE, PIT_APPROX, PIT_TRUE, ColumnProvenance
from impl_panel import normalize_ts_ms
from service_xs_data import build_quality_report

logger = logging.getLogger(__name__)

_PIT_RANK = {PIT_TRUE: 2, PIT_APPROX: 1, PIT_NONE: 0}


# ---------------------------------------------------------------------------
# PIT-leak scan
# ---------------------------------------------------------------------------
def pit_leak_scan(
    panel: Panel,
    asof_long: pd.DataFrame,
    *,
    value_col: str,
    publish_ts_col: str = "publish_ts",
    symbol_col: str = "symbol",
) -> List[Dict[str, Any]]:
    """Найти look-ahead: значение ``value_col`` присутствует РАНЬШЕ первой публикации.

    Для каждого символа: первый ts с НЕ-NaN значением должен быть ≥ min(publish_ts). Иначе —
    утечка из будущего. Возвращает список нарушений (пусто = чисто).
    """
    leaks: List[Dict[str, Any]] = []
    if value_col not in panel.columns or asof_long is None or len(asof_long) == 0:
        return leaks
    pub_ms_all = normalize_ts_ms(asof_long[publish_ts_col])
    for sym, g in panel[value_col].groupby(level=SYMBOL_LEVEL):
        nonnan = g.dropna()
        if len(nonnan) == 0:
            continue
        first_value_ts = int(nonnan.index.get_level_values(TS_LEVEL).min())
        mask = asof_long[symbol_col].to_numpy() == sym
        pubs = pub_ms_all[mask]
        if len(pubs) == 0:
            leaks.append({"symbol": str(sym), "reason": "value present but no publish record"})
            continue
        min_pub = int(np.min(pubs))
        if first_value_ts < min_pub:
            leaks.append(
                {
                    "symbol": str(sym),
                    "first_value_ts": first_value_ts,
                    "min_publish_ts": min_pub,
                    "reason": "look-ahead: value before first publish",
                }
            )
    return leaks


# ---------------------------------------------------------------------------
# Signal → columns lineage
# ---------------------------------------------------------------------------
def signal_columns(signal: Any, panel: Panel) -> List[str]:
    """Колонки панели, которые читает сигнал (по атрибутам ``*_col``/``*_column``/``column``)."""
    cols = set()
    for attr in dir(signal):
        if attr.startswith("_"):
            continue
        if not (attr.endswith("_col") or attr.endswith("_column") or attr == "column"):
            continue
        try:
            v = getattr(signal, attr)
        except Exception:
            continue
        if isinstance(v, str) and v in panel.columns:
            cols.add(v)
    return sorted(cols)


def _worst_pit(pits: Sequence[str]) -> str:
    return min(pits, key=lambda p: _PIT_RANK.get(p, 2)) if pits else PIT_TRUE


# ---------------------------------------------------------------------------
# Data-Trust report
# ---------------------------------------------------------------------------
def data_trust_report(
    panel: Panel,
    provenance: Sequence[ColumnProvenance],
    *,
    signal_library: Any = None,
    price_col: str = "close",
    now_ms: Optional[int] = None,
    survivorship_biased: Optional[bool] = None,
) -> Dict[str, Any]:
    """Data-Trust: DataQualityReport + per-signal PIT-lineage + verdict (trusted|caution|untrusted)."""
    base = build_quality_report(
        panel,
        provenance,
        price_col=price_col,
        now_ms=now_ms,
        survivorship_biased=survivorship_biased,
    )
    pit_by_col = {p.column: p.pit_quality for p in provenance}

    lineage: Dict[str, Any] = {}
    pit_violations: List[str] = []
    specs = getattr(signal_library, "_specs", None) if signal_library is not None else None
    if specs:
        for spec in specs:
            name = spec.output_name
            cols = signal_columns(spec.signal, panel)
            pits = [pit_by_col.get(c, PIT_TRUE) for c in cols]
            worst = _worst_pit(pits)
            safe = worst != PIT_NONE
            lineage[name] = {"columns": cols, "worst_pit": worst, "backtest_safe": safe}
            if not safe:
                pit_violations.append(name)

    if lineage:
        used_worst = _worst_pit([l["worst_pit"] for l in lineage.values()])
    else:
        used_worst = base.worst_pit
    if pit_violations:
        trust = "untrusted"
    elif used_worst == PIT_APPROX:
        trust = "caution"
    else:
        trust = "trusted"

    out = base.to_dict()
    out["signal_lineage"] = lineage
    out["pit_violations"] = pit_violations
    out["trust_verdict"] = trust
    out["used_worst_pit"] = used_worst
    return out


__all__ = ["pit_leak_scan", "signal_columns", "data_trust_report"]
