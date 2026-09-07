# -*- coding: utf-8 -*-
"""
Immutable trade blotter + cash ledger - AGENT ZONE (books-and-records).

Closes the P0 post-trade gap: there was an order journal and a MiFIR audit trail,
but no consolidated **executed-trade blotter** (trade economics: gross, fees,
financing, settlement) as the firm's official record of trades, and no
**cash ledger** (general-ledger of cash movements). This module is both, each an
**append-only, hash-chained** (tamper-evident) SQLite store.

* ``TradeBlotter`` — one immutable row per executed trade with full economics and
  the canonical instrument identity (FIGI from the instrument master). INSERT-only;
  every row commits to the previous row's hash (see ``packages.agent.audit``).
* ``CashLedger`` — append-only double-entry-style cash GL: every fill posts a TRADE
  movement (signed) plus FEE / FINANCING / DIVIDEND / INTEREST / DEPOSIT movements,
  each carrying the running balance and chained.

Books-and-records integrity is verifiable at any time via ``verify()`` (recomputes
the chain and reports the first tampered row, if any).

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.agent.audit.hash_chain import GENESIS_HASH, chain_hash, ChainRecord, verify_chain


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _f(x: Any) -> float:
    if isinstance(x, Decimal):
        return float(x)
    return float(x if x is not None else 0.0)


# Settlement convention per asset class (calendar→business-day add).
_SETTLE_DAYS = {"equity": 1, "etf": 1, "option": 1, "fx": 2, "crypto": 0, "future": 0, "index": 0}


def settlement_date(asset_class: str, trade_dt: Optional[datetime] = None) -> str:
    """T+N settlement date (business days) by asset class -> ISO date string."""
    d = (trade_dt or _now()).date()
    n = _SETTLE_DAYS.get((asset_class or "equity").lower(), 1)
    added = 0
    while added < n:
        d = d + timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d.isoformat()


# ---------------------------------------------------------------------------
# Trade blotter
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    seq: int
    ts: str
    symbol: str
    figi: Optional[str]
    side: str
    quantity: float
    price: float
    gross_notional: float
    fee: float
    financing: float
    currency: str
    asset_class: str
    strategy_id: str
    client_order_id: Optional[str]
    broker_order_id: Optional[str]
    settlement_date: str
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class TradeBlotter:
    """Append-only, hash-chained executed-trade record (books-and-records)."""

    def __init__(self, db_path: Optional[Path] = None, *, hmac_key: Optional[bytes] = None) -> None:
        self._lock = threading.RLock()
        self._key = hmac_key
        self._db_path = Path(db_path) if db_path else None
        self._mem: List[TradeRecord] = []
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL, symbol TEXT NOT NULL, figi TEXT,
                    side TEXT NOT NULL, quantity TEXT NOT NULL, price TEXT NOT NULL,
                    gross_notional TEXT NOT NULL, fee TEXT NOT NULL, financing TEXT NOT NULL,
                    currency TEXT NOT NULL, asset_class TEXT NOT NULL, strategy_id TEXT,
                    client_order_id TEXT, broker_order_id TEXT, settlement_date TEXT,
                    prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    @property
    def _head_hash(self) -> str:
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT entry_hash FROM trades ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            return row["entry_hash"] if row else GENESIS_HASH
        return self._mem[-1].entry_hash if self._mem else GENESIS_HASH

    @property
    def _count(self) -> int:
        if self._conn is not None:
            return int(self._conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"])
        return len(self._mem)

    def record_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Any,
        price: Any,
        fee: Any = 0,
        financing: Any = 0,
        figi: Optional[str] = None,
        currency: str = "USD",
        asset_class: str = "equity",
        strategy_id: str = "",
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> TradeRecord:
        with self._lock:
            t = ts or _now()
            seq = self._count + 1
            qty = _f(quantity)
            px = _f(price)
            payload = {
                "ts": t.isoformat(),
                "symbol": str(symbol),
                "figi": figi,
                "side": str(side).lower(),
                "quantity": qty,
                "price": px,
                "gross_notional": abs(qty * px),
                "fee": _f(fee),
                "financing": _f(financing),
                "currency": currency,
                "asset_class": asset_class,
                "strategy_id": strategy_id,
                "client_order_id": client_order_id,
                "broker_order_id": broker_order_id,
                "settlement_date": settlement_date(asset_class, t),
            }
            prev = self._head_hash
            h = chain_hash(prev, payload, seq, key=self._key)
            rec = TradeRecord(seq=seq, prev_hash=prev, entry_hash=h, **payload)
            if self._conn is not None:
                self._conn.execute(
                    """INSERT INTO trades(ts,symbol,figi,side,quantity,price,gross_notional,fee,
                       financing,currency,asset_class,strategy_id,client_order_id,broker_order_id,
                       settlement_date,prev_hash,entry_hash)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.ts,
                        rec.symbol,
                        rec.figi,
                        rec.side,
                        str(rec.quantity),
                        str(rec.price),
                        str(rec.gross_notional),
                        str(rec.fee),
                        str(rec.financing),
                        rec.currency,
                        rec.asset_class,
                        rec.strategy_id,
                        rec.client_order_id,
                        rec.broker_order_id,
                        rec.settlement_date,
                        rec.prev_hash,
                        rec.entry_hash,
                    ),
                )
                self._conn.commit()
            else:
                self._mem.append(rec)
            return rec

    def _all_records(self) -> List[TradeRecord]:
        if self._conn is not None:
            rows = self._conn.execute("SELECT * FROM trades ORDER BY seq ASC").fetchall()
            out = []
            for r in rows:
                out.append(
                    TradeRecord(
                        seq=r["seq"],
                        ts=r["ts"],
                        symbol=r["symbol"],
                        figi=r["figi"],
                        side=r["side"],
                        quantity=float(r["quantity"]),
                        price=float(r["price"]),
                        gross_notional=float(r["gross_notional"]),
                        fee=float(r["fee"]),
                        financing=float(r["financing"]),
                        currency=r["currency"],
                        asset_class=r["asset_class"],
                        strategy_id=r["strategy_id"],
                        client_order_id=r["client_order_id"],
                        broker_order_id=r["broker_order_id"],
                        settlement_date=r["settlement_date"],
                        prev_hash=r["prev_hash"],
                        entry_hash=r["entry_hash"],
                    )
                )
            return out
        return list(self._mem)

    def trades(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        recs = self._all_records()
        return [r.to_dict() for r in recs[-limit:]]

    def verify(self) -> Dict[str, Any]:
        recs = self._all_records()
        chain = [
            ChainRecord(
                seq=r.seq,
                payload=_blotter_payload(r),
                prev_hash=r.prev_hash,
                entry_hash=r.entry_hash,
            )
            for r in recs
        ]
        out = verify_chain(chain, key=self._key)
        out["keyed"] = self._key is not None
        return out

    def summary(self) -> Dict[str, Any]:
        recs = self._all_records()
        return {
            "n_trades": len(recs),
            "gross_traded": round(sum(r.gross_notional for r in recs), 2),
            "total_fees": round(sum(r.fee for r in recs), 2),
            "head_hash": self._head_hash,
            "integrity": self.verify(),
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def _blotter_payload(r: TradeRecord) -> Dict[str, Any]:
    return {
        "ts": r.ts,
        "symbol": r.symbol,
        "figi": r.figi,
        "side": r.side,
        "quantity": r.quantity,
        "price": r.price,
        "gross_notional": r.gross_notional,
        "fee": r.fee,
        "financing": r.financing,
        "currency": r.currency,
        "asset_class": r.asset_class,
        "strategy_id": r.strategy_id,
        "client_order_id": r.client_order_id,
        "broker_order_id": r.broker_order_id,
        "settlement_date": r.settlement_date,
    }


# ---------------------------------------------------------------------------
# Cash ledger
# ---------------------------------------------------------------------------
_CASH_TYPES = {"TRADE", "FEE", "FINANCING", "DIVIDEND", "INTEREST", "DEPOSIT", "WITHDRAWAL"}


@dataclass
class CashMovement:
    seq: int
    ts: str
    type: str
    amount: float  # signed: + increases cash
    balance: float  # running balance after this movement
    currency: str
    ref: Optional[str]
    symbol: Optional[str]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class CashLedger:
    """Append-only, hash-chained cash general-ledger with running balance."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        *,
        opening_balance: float = 0.0,
        currency: str = "USD",
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._key = hmac_key
        self._currency = currency
        self._opening = float(opening_balance)
        self._db_path = Path(db_path) if db_path else None
        self._mem: List[CashMovement] = []
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cash_movements (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL, type TEXT NOT NULL, amount TEXT NOT NULL,
                    balance TEXT NOT NULL, currency TEXT NOT NULL, ref TEXT, symbol TEXT,
                    prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cash_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            row = self._conn.execute("SELECT value FROM cash_meta WHERE key='opening'").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO cash_meta(key,value) VALUES('opening',?)", (str(self._opening),)
                )
                self._conn.commit()
            else:
                self._opening = float(row["value"])

    @property
    def _last(self) -> Optional[CashMovement]:
        if self._conn is not None:
            r = self._conn.execute(
                "SELECT * FROM cash_movements ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            if r is None:
                return None
            return CashMovement(
                seq=r["seq"],
                ts=r["ts"],
                type=r["type"],
                amount=float(r["amount"]),
                balance=float(r["balance"]),
                currency=r["currency"],
                ref=r["ref"],
                symbol=r["symbol"],
                prev_hash=r["prev_hash"],
                entry_hash=r["entry_hash"],
            )
        return self._mem[-1] if self._mem else None

    @property
    def balance(self) -> float:
        last = self._last
        return last.balance if last else self._opening

    def post(
        self,
        type_: str,
        amount: Any,
        *,
        ref: Optional[str] = None,
        symbol: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> CashMovement:
        t = str(type_).upper()
        if t not in _CASH_TYPES:
            raise ValueError(f"unknown cash movement type {type_!r}")
        with self._lock:
            last = self._last
            seq = (last.seq + 1) if last else 1
            prev = last.entry_hash if last else GENESIS_HASH
            bal = (last.balance if last else self._opening) + _f(amount)
            tt = (ts or _now()).isoformat()
            payload = {
                "ts": tt,
                "type": t,
                "amount": _f(amount),
                "balance": round(bal, 10),
                "currency": self._currency,
                "ref": ref,
                "symbol": symbol,
            }
            h = chain_hash(prev, payload, seq, key=self._key)
            mv = CashMovement(seq=seq, prev_hash=prev, entry_hash=h, **payload)
            if self._conn is not None:
                self._conn.execute(
                    """INSERT INTO cash_movements(ts,type,amount,balance,currency,ref,symbol,prev_hash,entry_hash)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        mv.ts,
                        mv.type,
                        str(mv.amount),
                        str(mv.balance),
                        mv.currency,
                        mv.ref,
                        mv.symbol,
                        mv.prev_hash,
                        mv.entry_hash,
                    ),
                )
                self._conn.commit()
            else:
                self._mem.append(mv)
            return mv

    def _all(self) -> List[CashMovement]:
        if self._conn is not None:
            rows = self._conn.execute("SELECT * FROM cash_movements ORDER BY seq ASC").fetchall()
            return [
                CashMovement(
                    seq=r["seq"],
                    ts=r["ts"],
                    type=r["type"],
                    amount=float(r["amount"]),
                    balance=float(r["balance"]),
                    currency=r["currency"],
                    ref=r["ref"],
                    symbol=r["symbol"],
                    prev_hash=r["prev_hash"],
                    entry_hash=r["entry_hash"],
                )
                for r in rows
            ]
        return list(self._mem)

    def movements(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._all()[-limit:]]

    def verify(self) -> Dict[str, Any]:
        recs = self._all()
        chain = [
            ChainRecord(
                seq=m.seq,
                payload={
                    "ts": m.ts,
                    "type": m.type,
                    "amount": m.amount,
                    "balance": m.balance,
                    "currency": m.currency,
                    "ref": m.ref,
                    "symbol": m.symbol,
                },
                prev_hash=m.prev_hash,
                entry_hash=m.entry_hash,
            )
            for m in recs
        ]
        out = verify_chain(chain, key=self._key)
        out["keyed"] = self._key is not None
        return out

    def summary(self) -> Dict[str, Any]:
        recs = self._all()
        by_type: Dict[str, float] = {}
        for m in recs:
            by_type[m.type] = by_type.get(m.type, 0.0) + m.amount
        return {
            "opening": self._opening,
            "balance": round(self.balance, 2),
            "n_movements": len(recs),
            "by_type": {k: round(v, 2) for k, v in by_type.items()},
            "integrity": self.verify(),
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


__all__ = ["TradeBlotter", "TradeRecord", "CashLedger", "CashMovement", "settlement_date"]
