# -*- coding: utf-8 -*-
"""
services/index_membership_loader.py
===================================

Загрузчик **истории членства в индексе** (survivorship-free, point-in-time) для
cross-sectional equity-юниверса. Закрывает половину equity-PIT блокера: код-слот
``impl_universe.IndexMembershipUniverse`` уже умеет PIT-реконструкцию состава —
не хватало ЗАГРУЗЧИКА данных и обвязки в пайплайн. Это здесь.

Почему важно: бэктест по СЕГОДНЯШНЕМУ списку индекса (как в free-пресетах) имеет
**survivorship bias** — исключённые/делистнутые тогда-активные тикеры пропадают,
и доходность завышается. PIT-членство устраняет это: на каждую дату ребаланса берётся
исторический состав индекса.

Формат файла изменений (CSV/parquet), long:
    date,ticker,action       # action ∈ {add, remove}
Самая ранняя дата = baseline (её ``add`` задают исходный состав); далее — события.

Полную историю S&P 500 (платно/скрейп) пользователь кладёт в этот формат и указывает
``universe.membership_path`` в конфиге. В репозитории — небольшой ЧЕСТНО ПОМЕЧЕННЫЙ
demo-файл с реальными якорными событиями (см. data/universe/sp500_membership_demo.csv).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_ADD = {"add", "added", "include", "1", "+"}
_REMOVE = {"remove", "removed", "delete", "drop", "0", "-"}


def load_membership_changes(path: str) -> pd.DataFrame:
    """Прочитать changes-файл → нормализованный DataFrame [date, ticker, action]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_c = cols.get("date") or cols.get("effective_date") or cols.get("effective")
    tick_c = cols.get("ticker") or cols.get("symbol")
    act_c = cols.get("action") or cols.get("event")
    if not (date_c and tick_c and act_c):
        raise ValueError(
            f"membership file must have columns date,ticker,action; got {list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_c]).dt.strftime("%Y-%m-%d"),
            "ticker": df[tick_c].astype(str).str.upper().str.strip(),
            "action": df[act_c].astype(str).str.lower().str.strip(),
        }
    )
    out["action"] = out["action"].map(
        lambda a: "add" if a in _ADD else ("remove" if a in _REMOVE else a)
    )
    out = out[out["action"].isin(["add", "remove"])].reset_index(drop=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def changes_to_baseline_and_events(changes: pd.DataFrame):
    """Из long changes → (baseline_date, baseline_symbols, events[{date,added,removed}])."""
    if changes.empty:
        return None, [], []
    dates = sorted(changes["date"].unique())
    baseline_date = dates[0]
    base_rows = changes[(changes["date"] == baseline_date) & (changes["action"] == "add")]
    baseline = sorted(base_rows["ticker"].unique().tolist())
    events: List[Dict[str, Any]] = []
    for d in dates[1:]:
        day = changes[changes["date"] == d]
        added = sorted(day[day["action"] == "add"]["ticker"].unique().tolist())
        removed = sorted(day[day["action"] == "remove"]["ticker"].unique().tolist())
        if added or removed:
            events.append(
                {"date": d, "added": added, "removed": removed, "reason": "membership change"}
            )
    return baseline_date, baseline, events


def build_index_membership_universe(
    path: str,
    *,
    index: str = "CUSTOM",
    name: Optional[str] = None,
    delistings: Optional[Sequence[Dict[str, Any]]] = None,
):
    """Построить survivorship-free ``IndexMembershipUniverse`` из changes-файла."""
    from impl_universe import IndexMembershipUniverse

    changes = load_membership_changes(path)
    baseline_date, baseline, events = changes_to_baseline_and_events(changes)
    if baseline_date is None:
        raise ValueError(f"no membership events parsed from {path}")
    uni = IndexMembershipUniverse.from_baseline(
        index,
        baseline,
        baseline_date,
        changes=events,
        delistings=list(delistings or []),
        name=name or f"index:{index}",
    )
    logger.info(
        "IndexMembershipUniverse '%s': baseline=%d @ %s, %d change-events (PIT, survivorship-free)",
        index,
        len(baseline),
        baseline_date,
        len(events),
    )
    return uni


__all__ = [
    "load_membership_changes",
    "changes_to_baseline_and_events",
    "build_index_membership_universe",
]
