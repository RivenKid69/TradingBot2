# -*- coding: utf-8 -*-
"""
packages/agent/execution/smart_order_router.py
==============================================

Smart Order Routing (P2): выбор venue и **мульти-venue сплит** заявки для минимизации
ожидаемых издержек (fee + market-impact + latency). Нужно для институциональной
интеграции и масштаба (одна заявка распределяется по площадкам с лучшей ликвидностью).

Модель издержек на venue (bps): ``fee_bps + impact_coef·sqrt(notional/liquidity)·1e4 +
latency_penalty``. Сплит — «water-filling» по предельной стоимости (каждый следующий
кусок едет на venue с наименьшей маржинальной стоимостью), что снижает суммарный импакт.

DI: список ``Venue`` подаётся снаружи (Agent знает свои подключения). Слой Agent (ордера
создаются локально; SOR выбирает КУДА, не «что» — CCEA-граница соблюдена).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


@dataclass
class VenueQuote:
    """Live top-of-book snapshot from a venue (P2 #14)."""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0      # size in units (not notional)
    ask_size: float = 0.0


class LiquidityProvider(Protocol):
    """Live liquidity feed: per (venue, symbol) top-of-book."""
    def get_quote(self, venue: str, symbol: str) -> Optional[VenueQuote]: ...


@dataclass
class Venue:
    name: str
    fee_bps: float = 1.0                 # комиссия (bps от notional)
    latency_ms: float = 50.0             # латентность (информативно + штраф)
    liquidity: float = 1e7               # доступная ликвидность (notional)
    impact_coef: float = 0.1             # k в impact = k·sqrt(participation)·1e4
    latency_penalty_bps_per_100ms: float = 0.2
    min_notional: float = 0.0            # мин. размер заявки на venue
    enabled: bool = True
    # live quote (P2 #14) — refreshed from a LiquidityProvider; drives spread + cap
    quote: Optional[VenueQuote] = None
    spread_bps: float = 0.0              # half-spread cost added when live quote present

    def update_from_quote(self, q: VenueQuote, side: str) -> None:
        """Refresh available liquidity + spread from a live top-of-book snapshot."""
        self.quote = q
        mid = (q.bid + q.ask) / 2.0 if (q.bid > 0 and q.ask > 0) else max(q.bid, q.ask)
        if mid > 0 and q.ask > q.bid > 0:
            self.spread_bps = (q.ask - q.bid) / mid * 1e4 / 2.0   # half-spread
        # displayed size on the relevant side → notional capacity at the touch
        if side.upper() == "BUY" and q.ask > 0:
            self.liquidity = max(q.ask_size * q.ask, 1e-9)
        elif side.upper() == "SELL" and q.bid > 0:
            self.liquidity = max(q.bid_size * q.bid, 1e-9)

    def marginal_cost_bps(self, filled: float, add: float) -> float:
        """Маржинальная стоимость bps для добавления ``add`` notional при уже ``filled``."""
        liq = max(self.liquidity, 1e-9)
        part0 = filled / liq
        part1 = (filled + add) / liq
        # средний импакт на добавляемом куске (разница интегралов ~ sqrt)
        impact = self.impact_coef * ((math.sqrt(max(part1, 0)) + math.sqrt(max(part0, 0))) / 2.0) * 1e4
        latency = self.latency_penalty_bps_per_100ms * (self.latency_ms / 100.0)
        return self.fee_bps + self.spread_bps + impact + latency


@dataclass
class Allocation:
    venue: str
    notional: float
    est_cost_bps: float
    est_cost: float


@dataclass
class RouteResult:
    symbol: str
    side: str
    total_notional: float
    allocations: List[Allocation]
    total_est_cost: float
    total_est_cost_bps: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol, "side": self.side, "total_notional": self.total_notional,
            "total_est_cost": self.total_est_cost, "total_est_cost_bps": self.total_est_cost_bps,
            "allocations": [a.__dict__ for a in self.allocations],
        }


class SmartOrderRouter:
    def __init__(self, venues: Sequence[Venue], *, n_steps: int = 50) -> None:
        self.venues = [v for v in venues if v.enabled]
        self.n_steps = int(n_steps)

    def best_venue(self, notional: float) -> Optional[Venue]:
        """Лучший ОДИН venue для всей заявки (минимальная средняя стоимость)."""
        cands = [v for v in self.venues if notional >= v.min_notional and v.liquidity > 0]
        if not cands:
            return None
        return min(cands, key=lambda v: v.marginal_cost_bps(0.0, notional))

    def route(self, symbol: str, side: str, notional: float, *, split: bool = True) -> RouteResult:
        """Маршрутизация заявки. ``split=True`` → мульти-venue water-filling.

        Venues, чей ``min_notional`` > размер заявки, исключаются (нельзя удовлетворить минимум).
        """
        notional = float(notional)
        # upfront-фильтр: только venue, способные принять заявку целиком (min_notional)
        usable = [v for v in self.venues if v.min_notional <= notional and v.liquidity > 0]
        if not usable or notional <= 0:
            return self._result(symbol, side, notional, [])

        if not split or len(usable) == 1:
            v = min(usable, key=lambda v: v.marginal_cost_bps(0.0, notional))
            bps = v.marginal_cost_bps(0.0, notional)
            return self._result(symbol, side, notional, [Allocation(v.name, notional, bps, notional * bps / 1e4)])

        # water-filling по маржинальной стоимости
        filled: Dict[str, float] = {v.name: 0.0 for v in usable}
        vmap = {v.name: v for v in usable}
        step = notional / self.n_steps if self.n_steps > 0 else notional
        remaining = notional
        while remaining > 1e-9:
            chunk = min(step, remaining)
            best = min(usable, key=lambda v: v.marginal_cost_bps(filled[v.name], chunk))
            filled[best.name] += chunk
            remaining -= chunk

        allocs = []
        for name, notl in filled.items():
            if notl <= 1e-9:
                continue
            v = vmap[name]
            bps = v.marginal_cost_bps(0.0, notl)   # средняя стоимость аллокации
            allocs.append(Allocation(name, notl, bps, notl * bps / 1e4))
        allocs.sort(key=lambda a: -a.notional)
        return self._result(symbol, side, notional, allocs)

    @staticmethod
    def _result(symbol, side, notional, allocs: List[Allocation]) -> RouteResult:
        total_cost = sum(a.est_cost for a in allocs)
        total_bps = (total_cost / notional * 1e4) if notional > 0 else 0.0
        return RouteResult(symbol=symbol, side=side, total_notional=notional,
                           allocations=allocs, total_est_cost=total_cost,
                           total_est_cost_bps=total_bps)

    # -- live liquidity (P2 #14) -------------------------------------------
    def refresh_liquidity(self, symbol: str, side: str, provider: "LiquidityProvider") -> int:
        """Refresh each venue's available liquidity + spread from a live feed.

        Venues with no quote (or zero size on the relevant side) are disabled for
        this route so we never split onto a venue showing no liquidity. Returns the
        number of venues with usable live liquidity.
        """
        usable = 0
        for v in self.venues:
            q = None
            try:
                q = provider.get_quote(v.name, symbol)
            except Exception:
                q = None
            if q is None:
                continue
            v.update_from_quote(q, side)
            if v.liquidity > 0:
                usable += 1
        return usable

    def route_live(self, symbol: str, side: str, notional: float,
                   provider: "LiquidityProvider", *, split: bool = True) -> RouteResult:
        """Route using LIVE top-of-book liquidity (refresh then route)."""
        self.refresh_liquidity(symbol, side, provider)
        return self.route(symbol, side, notional, split=split)

    # -- dispatch (P2 #14): actually send child orders to venues -----------
    def dispatch(self, route: RouteResult, submit_fn: Callable[[str, str, str, float], Dict[str, Any]]) -> Dict[str, Any]:
        """Send each venue allocation as a child order via ``submit_fn``.

        ``submit_fn(venue, symbol, side, notional) -> result dict`` is the venue/broker
        connector. SOR decides WHERE/HOW MUCH; the connector creates the order locally
        (CCEA boundary preserved). Returns per-venue dispatch results + a rollup.
        """
        results: List[Dict[str, Any]] = []
        ok = 0
        for a in route.allocations:
            try:
                res = submit_fn(a.venue, route.symbol, route.side, a.notional)
                success = bool(res.get("success", True)) if isinstance(res, dict) else True
            except Exception as exc:  # pragma: no cover
                res, success = {"error": str(exc)}, False
            if success:
                ok += 1
            results.append({"venue": a.venue, "notional": a.notional,
                            "success": success, "result": res})
        return {"symbol": route.symbol, "side": route.side,
                "venues_dispatched": len(results), "venues_ok": ok,
                "all_ok": ok == len(results) and len(results) > 0,
                "dispatches": results}


__all__ = ["Venue", "VenueQuote", "LiquidityProvider", "Allocation", "RouteResult",
           "SmartOrderRouter"]
