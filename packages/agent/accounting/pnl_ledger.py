# -*- coding: utf-8 -*-
"""
Live P&L ledger - AGENT ZONE ONLY.

Closes the P0 post-trade gap: the live Agent had no realized/unrealized P&L ledger
of its own — equity was only echoed from the broker (``position_sync`` read it
opportunistically) and the engine's ``PortfolioState.equity`` was supplied
externally. This module is the Agent's OWN books-of-record.

Responsibilities
----------------
* consume fills (from ``FillHandler`` or directly) and maintain per-symbol
  inventory with **average-cost** (default) or **FIFO** lot accounting, handling
  reductions, full closes and sign flips through zero correctly;
* compute **realized** P&L on closes, **unrealized** P&L vs live marks, fees and
  **financing/funding** accrual;
* maintain cash, equity/**NAV** = cash + Σ market_value, day-P&L and cumulative P&L;
* take **EOD NAV snapshots** (durable), roll the day, and recover on restart by
  replaying the persisted fill log (crash-safe via SQLite);
* expose a ``PortfolioState`` so the execution engine's pre-trade risk runs against
  the Agent's own equity instead of an externally-supplied number.

Accounting identity (always holds):

    equity == starting_cash + realized_pnl + unrealized_pnl − fees − financing

Decimal throughout for monetary exactness (consistent with the rest of the Agent).

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

getcontext().prec = 34  # ample precision for inventory math

_D0 = Decimal("0")


def _d(value: Any, default: Decimal = _D0) -> Decimal:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Fill:
    """A single (incremental) execution applied to the ledger."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: Decimal  # absolute, > 0
    price: Decimal
    fee: Decimal = _D0
    ts: datetime = field(default_factory=_now)
    client_order_id: Optional[str] = None
    broker_order_id: Optional[str] = None

    @property
    def signed_qty(self) -> Decimal:
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass
class _Lot:
    """FIFO lot: signed qty (sign = position direction) at a cost price."""

    qty: Decimal
    price: Decimal


@dataclass
class LedgerPosition:
    """Per-symbol inventory + realized accounting."""

    symbol: str
    quantity: Decimal = _D0  # signed (long > 0, short < 0)
    avg_cost: Decimal = _D0  # average entry (abs basis)
    realized_pnl: Decimal = _D0  # cumulative realized on this symbol (excl. fees)
    fees: Decimal = _D0
    financing: Decimal = _D0
    mark: Decimal = _D0  # last mark price
    lots: Deque[_Lot] = field(default_factory=deque)  # FIFO mode only

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.mark

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.quantity == 0:
            return _D0
        return (self.mark - self.avg_cost) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "avg_cost": float(self.avg_cost),
            "mark": float(self.mark),
            "market_value": float(self.market_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "fees": float(self.fees),
            "financing": float(self.financing),
        }


@dataclass
class NavSnapshot:
    """EOD (or ad-hoc) NAV snapshot."""

    ts: str
    nav: float
    cash: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    financing: float
    day_pnl: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    label: str = "eod"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
class PnLLedger:
    """Agent-zone realized/unrealized P&L ledger with EOD NAV snapshots.

    Parameters
    ----------
    starting_cash : opening cash balance (== opening NAV with no positions).
    method : "average" (default) or "fifo" lot accounting.
    base_currency : reporting currency label.
    db_path : optional SQLite path for durability + crash recovery (replays fills).
    account_id / strategy_id : context tags persisted with snapshots.
    """

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        method: str = "average",
        base_currency: str = "USD",
        db_path: Optional[Path] = None,
        account_id: str = "agent",
        strategy_id: str = "",
    ) -> None:
        if method not in ("average", "fifo"):
            raise ValueError("method must be 'average' or 'fifo'")
        self._lock = threading.RLock()
        self.method = method
        self.base_currency = base_currency
        self.account_id = account_id
        self.strategy_id = strategy_id

        self._starting_cash = _d(starting_cash)
        self._cash = self._starting_cash
        self._positions: Dict[str, LedgerPosition] = {}
        self._realized_cum = _D0
        self._fees_cum = _D0
        self._financing_cum = _D0
        self._fill_seq = 0

        # day tracking
        self._last_close_nav = self._starting_cash
        self._day_realized = _D0
        self._day_fees = _D0
        self._nav_snapshots: List[NavSnapshot] = []

        # persistence
        self._conn: Optional[sqlite3.Connection] = None
        self._db_path = Path(db_path) if db_path else None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._recover()

    # ------------------------------------------------------------ persistence
    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
                quantity TEXT NOT NULL, price TEXT NOT NULL, fee TEXT NOT NULL,
                client_order_id TEXT, broker_order_id TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nav_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL, label TEXT NOT NULL, payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS marks (
                symbol TEXT PRIMARY KEY, price TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        # persist starting cash once (first writer wins) so recovery is exact
        row = self._conn.execute("SELECT value FROM meta WHERE key='starting_cash'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('starting_cash', ?)",
                (str(self._starting_cash),),
            )
        else:
            self._starting_cash = _d(row["value"])
        self._conn.commit()

    def _recover(self) -> None:
        """Rebuild ledger state by replaying persisted fills + last close NAV."""
        if self._conn is None:
            return
        self._cash = self._starting_cash
        # replay fills in order (no re-persist)
        rows = self._conn.execute(
            "SELECT ts, symbol, side, quantity, price, fee, client_order_id, broker_order_id "
            "FROM fills ORDER BY seq ASC"
        ).fetchall()
        for r in rows:
            f = Fill(
                symbol=r["symbol"],
                side=r["side"],
                quantity=_d(r["quantity"]),
                price=_d(r["price"]),
                fee=_d(r["fee"]),
                client_order_id=r["client_order_id"],
                broker_order_id=r["broker_order_id"],
            )
            self._apply_fill(f, persist=False)
        # A trade price is not a market mark.  Restore the last durable marks
        # after replaying fills so NAV/unrealized P&L survive a restart exactly.
        for r in self._conn.execute("SELECT symbol, price FROM marks").fetchall():
            pos = self._positions.setdefault(r["symbol"], LedgerPosition(symbol=r["symbol"]))
            pos.mark = _d(r["price"])
        # restore last-close NAV baseline for day-P&L continuity
        snap = self._conn.execute(
            "SELECT payload FROM nav_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if snap is not None:
            try:
                self._last_close_nav = _d(json.loads(snap["payload"]).get("nav"))
                self._day_realized = _D0
                self._day_fees = _D0
            except Exception:
                pass

    # ---------------------------------------------------------------- fills
    def on_fill(
        self,
        symbol: str,
        side: str,
        quantity: Any,
        price: Any,
        *,
        fee: Any = 0,
        ts: Optional[datetime] = None,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply an INCREMENTAL fill; returns the realized-P&L delta and new state."""
        side = str(side).lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy/sell, got {side!r}")
        qty = _d(quantity)
        if qty <= 0:
            raise ValueError("quantity must be > 0")
        f = Fill(
            symbol=str(symbol),
            side=side,
            quantity=qty,
            price=_d(price),
            fee=_d(fee),
            ts=ts or _now(),
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
        )
        with self._lock:
            return self._apply_fill(f, persist=True)

    def _apply_fill(self, f: Fill, *, persist: bool) -> Dict[str, Any]:
        pos = self._positions.setdefault(f.symbol, LedgerPosition(symbol=f.symbol, mark=f.price))
        realized = self._apply_inventory(pos, f)

        # cash & cumulative bookkeeping
        self._cash -= f.signed_qty * f.price  # buy reduces cash, sell increases
        self._cash -= f.fee
        pos.fees += f.fee
        pos.realized_pnl += realized
        self._realized_cum += realized
        self._fees_cum += f.fee
        self._day_realized += realized
        self._day_fees += f.fee
        # keep mark fresh at the trade price (until a real mark arrives)
        if pos.mark == 0:
            pos.mark = f.price
        self._fill_seq += 1

        if persist and self._conn is not None:
            self._conn.execute(
                "INSERT INTO fills(ts, symbol, side, quantity, price, fee, client_order_id, broker_order_id) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    f.ts.isoformat(),
                    f.symbol,
                    f.side,
                    str(f.quantity),
                    str(f.price),
                    str(f.fee),
                    f.client_order_id,
                    f.broker_order_id,
                ),
            )
            self._conn.commit()

        return {
            "symbol": f.symbol,
            "side": f.side,
            "quantity": float(f.quantity),
            "price": float(f.price),
            "fee": float(f.fee),
            "realized_delta": float(realized),
            "position_qty": float(pos.quantity),
            "avg_cost": float(pos.avg_cost),
            "realized_pnl": float(self._realized_cum),
            "equity": float(self.equity),
        }

    def _apply_inventory(self, pos: LedgerPosition, f: Fill) -> Decimal:
        """Update inventory; return realized P&L on the closed portion (excl. fee)."""
        if self.method == "fifo":
            return self._apply_fifo(pos, f)
        return self._apply_average(pos, f)

    @staticmethod
    def _apply_average(pos: LedgerPosition, f: Fill) -> Decimal:
        signed = f.signed_qty
        prev = pos.quantity
        realized = _D0
        if prev == 0 or (prev > 0) == (signed > 0):
            # same direction (or opening): weighted-average the cost
            new_qty = prev + signed
            if new_qty != 0:
                pos.avg_cost = (pos.avg_cost * abs(prev) + f.price * abs(signed)) / abs(new_qty)
            pos.quantity = new_qty
        else:
            # opposite direction: reduce / close / possibly flip
            closing = min(abs(signed), abs(prev))
            if prev > 0:  # selling to reduce a long
                realized = (f.price - pos.avg_cost) * closing
            else:  # buying to reduce a short
                realized = (pos.avg_cost - f.price) * closing
            new_qty = prev + signed
            pos.quantity = new_qty
            if (prev > 0) != (new_qty > 0) and new_qty != 0:
                pos.avg_cost = f.price  # flipped through zero -> new basis
            elif new_qty == 0:
                pos.avg_cost = _D0
            # else: same-side remainder keeps avg_cost
        return realized

    @staticmethod
    def _apply_fifo(pos: LedgerPosition, f: Fill) -> Decimal:
        signed = f.signed_qty
        realized = _D0
        remaining = signed
        # close opposing lots FIFO
        while remaining != 0 and pos.lots and (pos.lots[0].qty > 0) != (remaining > 0):
            lot = pos.lots[0]
            close = min(abs(remaining), abs(lot.qty))
            if lot.qty > 0:  # closing long lots by selling
                realized += (f.price - lot.price) * close
            else:  # closing short lots by buying
                realized += (lot.price - f.price) * close
            lot.qty -= close if lot.qty > 0 else -close
            remaining -= -close if remaining < 0 else close
            if lot.qty == 0:
                pos.lots.popleft()
        # any remainder opens a new lot in the trade direction
        if remaining != 0:
            pos.lots.append(_Lot(qty=remaining, price=f.price))
        # recompute aggregate qty + avg from lots
        pos.quantity = sum((l.qty for l in pos.lots), _D0)
        tot = sum((abs(l.qty) for l in pos.lots), _D0)
        pos.avg_cost = sum((abs(l.qty) * l.price for l in pos.lots), _D0) / tot if tot != 0 else _D0
        return realized

    # ---------------------------------------------------------------- marks
    def mark(self, symbol: str, price: Any) -> None:
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                pos = self._positions.setdefault(symbol, LedgerPosition(symbol=symbol))
            pos.mark = _d(price)
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO marks(symbol, price, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET price=excluded.price, updated_at=excluded.updated_at",
                    (str(symbol), str(pos.mark), _now().isoformat()),
                )
                self._conn.commit()

    def mark_prices(self, prices: Dict[str, Any]) -> None:
        with self._lock:
            for s, p in (prices or {}).items():
                self.mark(s, p)

    def accrue_financing(self, symbol: str, amount: Any) -> None:
        """Book a financing/funding cost (positive = cost): reduces cash + NAV."""
        amt = _d(amount)
        with self._lock:
            pos = self._positions.setdefault(symbol, LedgerPosition(symbol=symbol))
            pos.financing += amt
            self._financing_cum += amt
            self._cash -= amt

    # ----------------------------------------------------------- aggregates
    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_cum

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self._positions.values()), _D0)

    @property
    def market_value(self) -> Decimal:
        return sum((p.market_value for p in self._positions.values()), _D0)

    @property
    def equity(self) -> Decimal:
        """NAV = cash + Σ market_value."""
        return self._cash + self.market_value

    @property
    def gross_exposure(self) -> Decimal:
        return sum((abs(p.market_value) for p in self._positions.values()), _D0)

    @property
    def net_exposure(self) -> Decimal:
        return sum((p.market_value for p in self._positions.values()), _D0)

    @property
    def day_pnl(self) -> Decimal:
        """P&L since the last EOD close (NAV delta — includes realized+unrealized−costs)."""
        return self.equity - self._last_close_nav

    def position(self, symbol: str) -> Optional[LedgerPosition]:
        return self._positions.get(symbol)

    def positions(self) -> List[LedgerPosition]:
        return [p for p in self._positions.values() if p.quantity != 0 or p.realized_pnl != 0]

    # ------------------------------------------------------------- EOD / NAV
    def eod_close(self, *, ts: Optional[datetime] = None, label: str = "eod") -> NavSnapshot:
        """Take a NAV snapshot, roll the trading day, and persist it."""
        with self._lock:
            t = (ts or _now()).isoformat()
            snap = NavSnapshot(
                ts=t,
                nav=float(self.equity),
                cash=float(self._cash),
                realized_pnl=float(self._realized_cum),
                unrealized_pnl=float(self.unrealized_pnl),
                fees=float(self._fees_cum),
                financing=float(self._financing_cum),
                day_pnl=float(self.day_pnl),
                gross_exposure=float(self.gross_exposure),
                net_exposure=float(self.net_exposure),
                n_positions=len([p for p in self._positions.values() if p.quantity != 0]),
                label=label,
            )
            self._nav_snapshots.append(snap)
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO nav_snapshots(ts, label, payload) VALUES(?,?,?)",
                    (t, label, json.dumps(snap.to_dict())),
                )
                self._conn.commit()
            # roll the day
            self._last_close_nav = self.equity
            self._day_realized = _D0
            self._day_fees = _D0
            return snap

    def nav_history(self) -> List[Dict[str, Any]]:
        if self._conn is not None:
            rows = self._conn.execute(
                "SELECT payload FROM nav_snapshots ORDER BY id ASC"
            ).fetchall()
            return [json.loads(r["payload"]) for r in rows]
        return [s.to_dict() for s in self._nav_snapshots]

    # ---------------------------------------------------------------- views
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "account_id": self.account_id,
                "strategy_id": self.strategy_id,
                "base_currency": self.base_currency,
                "method": self.method,
                "starting_cash": float(self._starting_cash),
                "cash": float(self._cash),
                "equity": float(self.equity),
                "nav": float(self.equity),
                "realized_pnl": float(self._realized_cum),
                "unrealized_pnl": float(self.unrealized_pnl),
                "total_pnl": float(self.equity - self._starting_cash),
                "fees": float(self._fees_cum),
                "financing": float(self._financing_cum),
                "day_pnl": float(self.day_pnl),
                "gross_exposure": float(self.gross_exposure),
                "net_exposure": float(self.net_exposure),
                "n_positions": len([p for p in self._positions.values() if p.quantity != 0]),
                "n_fills": self._fill_seq,
                "positions": [p.to_dict() for p in self.positions()],
                "last_close_nav": float(self._last_close_nav),
            }

    def to_portfolio_state(self) -> Any:
        """Build a ``PortfolioState`` from the ledger so the engine's pre-trade risk
        uses the Agent's OWN equity (closes 'equity supplied externally')."""
        from packages.agent.policy.risk_checker import PortfolioState

        positions = {s: p.quantity for s, p in self._positions.items() if p.quantity != 0}
        values = {s: p.market_value for s, p in self._positions.items() if p.quantity != 0}
        eq = self.equity
        return PortfolioState(
            equity=eq,
            buying_power=eq,
            margin_available=eq,
            positions=positions,
            position_values=values,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            daily_pnl=self.day_pnl,
            peak_equity=max(eq, self._last_close_nav),
        )

    def reconcile_against(self, broker_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Compare ledger qty vs broker qty per symbol; report breaks (no mutation)."""
        breaks: List[Dict[str, Any]] = []
        syms = set(self._positions) | set(broker_positions or {})
        for s in sorted(syms):
            led = float(self._positions.get(s, LedgerPosition(symbol=s)).quantity)
            brk = float(_d(broker_positions.get(s, 0)))
            if abs(led - brk) > 1e-9:
                breaks.append(
                    {"symbol": s, "ledger_qty": led, "broker_qty": brk, "diff": led - brk}
                )
        return {"reconciled": len(breaks) == 0, "breaks": breaks}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# FillHandler bridge
# ---------------------------------------------------------------------------
def ledger_fill_callback(ledger: PnLLedger) -> Callable[[Dict[str, Any]], None]:
    """Adapt ``FillHandler.on_fill`` (cumulative) payloads into ledger increments.

    ``FillHandler`` reports cumulative ``filled_qty`` per order; this callback keeps
    a per-order high-water mark and applies only the INCREMENT to the ledger, so a
    sequence of partial fills books realized P&L correctly. Prefers the explicit
    ``fill_increment`` field when the handler provides it.
    """
    seen: Dict[str, Decimal] = {}

    def _cb(payload: Dict[str, Any]) -> None:
        coid = str(payload.get("client_order_id") or "")
        symbol = payload.get("symbol")
        side = payload.get("side")
        if not symbol or side not in ("buy", "sell"):
            return
        price = payload.get("avg_fill_price")
        if price in (None, "", "None"):
            return
        inc = payload.get("fill_increment")
        if inc is not None:
            qty = _d(inc)
        else:
            cum = _d(payload.get("filled_qty"))
            prev = seen.get(coid, _D0)
            qty = cum - prev
            seen[coid] = cum
        if qty <= 0:
            return
        ledger.on_fill(
            symbol,
            side,
            qty,
            price,
            client_order_id=coid or None,
            broker_order_id=payload.get("broker_order_id"),
        )

    return _cb
