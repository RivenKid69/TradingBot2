# -*- coding: utf-8 -*-
"""
services/automation/tca_reporter.py
===================================

Авто-отчёты TCA / best-execution (P2): из лога исполнений считает implementation
shortfall, slippage (bps), market-impact и агрегаты по venue/символу/стороне, плюс
сводку best-execution. Закрывает разрыв «модули есть, но отчёт не автоматизирован».

Вход — список trade-record'ов (DI), либо CSV/parquet лога. Выход — структурированный
отчёт (dict) + markdown. Слой services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _bps(x: float) -> float:
    return float(x) * 1e4


@dataclass
class TCAReport:
    n_trades: int
    total_notional: float
    avg_slippage_bps: float
    avg_impl_shortfall_bps: float
    total_cost: float
    by_venue: Dict[str, Dict[str, float]]
    by_symbol: Dict[str, Dict[str, float]]
    by_side: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "total_notional": self.total_notional,
            "avg_slippage_bps": self.avg_slippage_bps,
            "avg_impl_shortfall_bps": self.avg_impl_shortfall_bps,
            "total_cost": self.total_cost,
            "by_venue": self.by_venue,
            "by_symbol": self.by_symbol,
            "by_side": self.by_side,
        }


class TCAReporter:
    """Авто-TCA: trade-лог → метрики (slippage, implementation shortfall, by-venue)."""

    def analyze(self, trades: Sequence[Dict[str, Any]]) -> TCAReport:
        """trades: каждый — {symbol, side(BUY/SELL), qty, fill_price, arrival_price,
        benchmark_price?(vwap), venue?}. Метрики знаково корректны по стороне."""
        df = pd.DataFrame(list(trades))
        if df.empty:
            return TCAReport(0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})
        df["qty"] = df["qty"].astype(float)
        df["fill_price"] = df["fill_price"].astype(float)
        df["arrival_price"] = df["arrival_price"].astype(float)
        if "benchmark_price" not in df.columns:
            df["benchmark_price"] = df["arrival_price"]
        if "venue" not in df.columns:
            df["venue"] = "DEFAULT"
        df["side"] = df["side"].astype(str).str.upper()

        sign = np.where(df["side"] == "BUY", 1.0, -1.0)  # покупка дороже = хуже
        df["notional"] = df["qty"].abs() * df["fill_price"]
        # implementation shortfall vs arrival (decision price)
        df["is_bps"] = sign * (df["fill_price"] - df["arrival_price"]) / df["arrival_price"] * 1e4
        # slippage vs benchmark (e.g. interval VWAP)
        df["slip_bps"] = (
            sign * (df["fill_price"] - df["benchmark_price"]) / df["benchmark_price"] * 1e4
        )
        df["cost"] = df["is_bps"] / 1e4 * df["notional"]

        def _agg(g: pd.DataFrame) -> Dict[str, float]:
            notl = float(g["notional"].sum())
            wis = float((g["is_bps"] * g["notional"]).sum() / notl) if notl else 0.0
            wsl = float((g["slip_bps"] * g["notional"]).sum() / notl) if notl else 0.0
            return {
                "n": int(len(g)),
                "notional": notl,
                "impl_shortfall_bps": round(wis, 3),
                "slippage_bps": round(wsl, 3),
                "cost": round(float(g["cost"].sum()), 2),
            }

        total_notl = float(df["notional"].sum())
        return TCAReport(
            n_trades=int(len(df)),
            total_notional=total_notl,
            avg_slippage_bps=(
                round(float((df["slip_bps"] * df["notional"]).sum() / total_notl), 3)
                if total_notl
                else 0.0
            ),
            avg_impl_shortfall_bps=(
                round(float((df["is_bps"] * df["notional"]).sum() / total_notl), 3)
                if total_notl
                else 0.0
            ),
            total_cost=round(float(df["cost"].sum()), 2),
            by_venue={str(k): _agg(g) for k, g in df.groupby("venue")},
            by_symbol={str(k): _agg(g) for k, g in df.groupby("symbol")},
            by_side={str(k): _agg(g) for k, g in df.groupby("side")},
        )

    def analyze_log(self, path: str) -> TCAReport:
        if not os.path.exists(path):
            return TCAReport(0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        return self.analyze(df.to_dict("records"))

    def to_markdown(self, rep: TCAReport, *, title: str = "TCA / Best-Execution Report") -> str:
        L = [
            f"# {title}\n",
            f"- Сделок: **{rep.n_trades}** · оборот ${rep.total_notional:,.0f}",
            f"- Implementation shortfall: **{rep.avg_impl_shortfall_bps:.2f} bps** · "
            f"slippage vs benchmark: **{rep.avg_slippage_bps:.2f} bps** · издержки ${rep.total_cost:,.0f}\n",
            "## По venue\n",
            "| Venue | Сделок | Оборот | IS (bps) | Slippage (bps) |",
            "|---|---|---|---|---|",
        ]
        for v, a in sorted(rep.by_venue.items()):
            L.append(
                f"| {v} | {a['n']} | ${a['notional']:,.0f} | {a['impl_shortfall_bps']} | {a['slippage_bps']} |"
            )
        return "\n".join(L)


__all__ = ["TCAReport", "TCAReporter"]
