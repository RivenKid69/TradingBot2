# -*- coding: utf-8 -*-
"""
packages/agent/execution/allocation.py
=======================================

Post-trade allocation, give-up/CMTA and settlement (P2 #25).

The MVP's ``/api/post_trade/clearing_approve`` only returned a canned string. This
module provides the real post-trade machinery institutional desks need:

  * **average-price allocation** — a block parent filled across many prints is
    allocated to sub-accounts at the *single blended VWAP*, fairly and to the exact
    block quantity (largest-remainder rounding, no leakage);
  * **give-up / CMTA** — record that an executing broker gives the trade up to a
    different clearing broker (Clearing Member Trade Assignment);
  * **T+1 settlement** — compute settlement dates per asset class (US equities T+1
    since 2024-05-28, most others T+2) and net each account's cash/position
    obligations.

Pure stdlib + dataclasses; deterministic and unit-testable. Agent zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


@dataclass
class Fill:
    qty: Decimal
    price: Decimal

    def __post_init__(self) -> None:
        self.qty = Decimal(str(self.qty))
        self.price = Decimal(str(self.price))


@dataclass
class SubAccountTarget:
    account: str
    qty: Decimal

    def __post_init__(self) -> None:
        self.qty = Decimal(str(self.qty))


@dataclass
class AccountAllocation:
    account: str
    qty: Decimal
    price: Decimal             # average (blended) price — same for all accounts
    notional: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {"account": self.account, "qty": str(self.qty),
                "price": str(self.price), "notional": str(self.notional)}


@dataclass
class AllocationResult:
    symbol: str
    side: str
    avg_price: Decimal
    total_filled: Decimal
    allocations: List[AccountAllocation]
    residual: Decimal = Decimal("0")
    give_up: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "side": self.side, "avg_price": str(self.avg_price),
                "total_filled": str(self.total_filled), "residual": str(self.residual),
                "allocations": [a.to_dict() for a in self.allocations],
                "give_up": self.give_up}


def average_fill_price(fills: List[Fill]) -> Decimal:
    """VWAP of the fills (the single price every sub-account is allocated at)."""
    tot_qty = sum((f.qty for f in fills), Decimal("0"))
    if tot_qty == 0:
        return Decimal("0")
    notional = sum((f.qty * f.price for f in fills), Decimal("0"))
    return (notional / tot_qty)


def average_price_allocation(
    symbol: str, side: str, fills: List[Fill], targets: List[SubAccountTarget],
    *, give_up: Optional["GiveUp"] = None,
) -> AllocationResult:
    """Allocate a block to sub-accounts at the blended VWAP.

    The sum of target quantities should equal the filled quantity; any rounding
    residual (for fractional venues) is reported. Largest-remainder rounding keeps
    the allocated total exactly equal to the block (no over/under-allocation).
    """
    total_filled = sum((f.qty for f in fills), Decimal("0"))
    avg = average_fill_price(fills)
    requested = sum((t.qty for t in targets), Decimal("0"))

    allocs: List[AccountAllocation] = []
    if requested <= 0 or total_filled <= 0:
        return AllocationResult(symbol, side, avg, total_filled, allocs, total_filled)

    # pro-rata each account's share of the actual filled quantity
    raw = [(t.account, total_filled * (t.qty / requested)) for t in targets]
    # largest-remainder integer-ish rounding to preserve the exact block total
    floor_q = [(acc, q.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)) for acc, q in raw]
    allocated = sum((q for _, q in floor_q), Decimal("0"))
    residual = total_filled - allocated
    # push residual onto the largest target (deterministic)
    if floor_q and residual != 0:
        idx = max(range(len(floor_q)), key=lambda i: floor_q[i][1])
        acc, q = floor_q[idx]
        floor_q[idx] = (acc, q + residual)
        residual = Decimal("0")

    for acc, q in floor_q:
        allocs.append(AccountAllocation(acc, q, avg, (q * avg).quantize(Decimal("0.01"))))

    gd = give_up.to_dict() if give_up is not None else None
    return AllocationResult(symbol, side, avg, total_filled, allocs, residual, give_up=gd)


@dataclass
class GiveUp:
    """Give-up / CMTA: executing broker hands the trade to a clearing broker."""
    executing_broker: str
    clearing_broker: str
    account: str = ""
    cmta_code: str = ""        # Clearing Member Trade Assignment code

    def to_dict(self) -> Dict[str, Any]:
        return {"executing_broker": self.executing_broker, "clearing_broker": self.clearing_broker,
                "account": self.account, "cmta_code": self.cmta_code, "type": "give_up_cmta"}


# Settlement cycles by asset class (US equities moved to T+1 on 2024-05-28).
_SETTLE_DAYS = {"equity": 1, "etf": 1, "bond": 1, "crypto": 0,
                "fx": 2, "futures": 1, "options": 1}

_US_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


def _add_business_days(d: date, n: int) -> date:
    cur = d
    added = 0
    while added < n:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5 and cur not in _US_HOLIDAYS_2026:
            added += 1
    return cur


def settlement_date(trade_date: date, asset_class: str = "equity") -> date:
    """Settlement date for a trade (T+1 equities, T+2 FX, T+0 crypto), business-day aware."""
    n = _SETTLE_DAYS.get(str(asset_class).lower(), 2)
    return _add_business_days(trade_date, n) if n > 0 else trade_date


def net_settlement(allocations: List[AccountAllocation], side: str) -> Dict[str, Dict[str, str]]:
    """Net cash/position obligations per account for settlement.

    BUY → account pays cash (negative), receives shares (positive). SELL → reverse.
    """
    sign = Decimal("1") if side.upper() == "BUY" else Decimal("-1")
    out: Dict[str, Dict[str, str]] = {}
    for a in allocations:
        cash = -sign * a.notional        # buy pays cash out
        shares = sign * a.qty
        out[a.account] = {"cash_delta": str(cash.quantize(Decimal("0.01"))),
                          "position_delta": str(shares)}
    return out


class ClearingEngine:
    """Orchestrates allocation → give-up → settlement for a block trade."""

    def process_block(
        self, *, symbol: str, side: str, fills: List[Fill], targets: List[SubAccountTarget],
        trade_date: date, asset_class: str = "equity", give_up: Optional[GiveUp] = None,
    ) -> Dict[str, Any]:
        result = average_price_allocation(symbol, side, fills, targets, give_up=give_up)
        settle = settlement_date(trade_date, asset_class)
        net = net_settlement(result.allocations, side)
        return {
            "allocation": result.to_dict(),
            "settlement_date": settle.isoformat(),
            "asset_class": asset_class,
            "net_obligations": net,
            "status": "allocated",
        }


__all__ = [
    "Fill", "SubAccountTarget", "AccountAllocation", "AllocationResult", "GiveUp",
    "average_fill_price", "average_price_allocation", "settlement_date",
    "net_settlement", "ClearingEngine",
]
