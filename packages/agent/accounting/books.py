# -*- coding: utf-8 -*-
"""
Books-and-records facade - AGENT ZONE.

Single integration point that keeps the Agent's full books consistent on every
fill, tying together:

  * ``PnLLedger``      — positions + realized/unrealized P&L + NAV (source of truth);
  * ``TradeBlotter``   — immutable, hash-chained executed-trade record;
  * ``CashLedger``     — append-only, hash-chained cash general-ledger;
  * ``InstrumentMaster`` — annotates every trade with the canonical FIGI;
  * ``MarketAbuseMonitor`` — live MAR surveillance fed by the SAME order/fill flow.

``on_order`` / ``on_fill`` update all of the above atomically, so the firm's
books-of-record, P&L, and surveillance all observe the real live execution path.

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.agent.accounting.pnl_ledger import PnLLedger
from packages.agent.accounting.blotter import TradeBlotter, CashLedger


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class BooksAndRecords:
    """Owns the Agent's consistent books + live surveillance over one fill stream."""

    def __init__(
        self,
        *,
        starting_cash: float = 100_000.0,
        data_dir: Optional[Path] = None,
        account_id: str = "agent",
        strategy_id: str = "",
        base_currency: str = "USD",
        method: str = "average",
        hmac_key: Optional[bytes] = None,
        instrument_master: Any = None,
        surveillance: Any = None,
    ) -> None:
        self._lock = threading.RLock()
        self.account_id = account_id
        d = Path(data_dir) if data_dir else None
        self.ledger = PnLLedger(
            starting_cash=starting_cash,
            method=method,
            base_currency=base_currency,
            db_path=(d / "pnl_ledger.db") if d else None,
            account_id=account_id,
            strategy_id=strategy_id,
        )
        self.blotter = TradeBlotter(
            db_path=(d / "trade_blotter.db") if d else None, hmac_key=hmac_key
        )
        self.cash = CashLedger(
            db_path=(d / "cash_ledger.db") if d else None,
            opening_balance=starting_cash,
            currency=base_currency,
            hmac_key=hmac_key,
        )
        # instrument master (lazy default) + surveillance monitor (optional)
        if instrument_master is None:
            try:
                from services.instrument_master import get_default_master

                instrument_master = get_default_master()
            except Exception:
                instrument_master = None
        self.instruments = instrument_master
        if surveillance is None:
            try:
                from services.algo_integration.market_abuse import MarketAbuseMonitor

                surveillance = MarketAbuseMonitor()
            except Exception:
                surveillance = None
        self.surveillance = surveillance

    # ------------------------------------------------------------------ helpers
    def _resolve(self, symbol: str):
        if self.instruments is None:
            return None, "equity"
        rec = self.instruments.resolve(symbol)
        if rec is None:
            return None, "equity"
        return rec.figi, rec.asset_class

    # ------------------------------------------------------------------ orders
    def on_order(
        self,
        *,
        symbol: str,
        side: str,
        action: str,
        quantity: float,
        price: float,
        order_id: str,
        account: Optional[str] = None,
        mid: Optional[float] = None,
        ts_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Feed an order lifecycle event (NEW/CANCEL/MODIFY) to live surveillance."""
        if self.surveillance is None:
            return []
        from services.algo_integration.market_abuse import OrderEvent

        ev = OrderEvent(
            ts_ms=ts_ms or _now_ms(),
            symbol=str(symbol),
            account=account or self.account_id,
            side=str(side).upper(),
            action=str(action).upper(),
            qty=float(quantity),
            price=float(price),
            order_id=str(order_id),
            mid=mid,
        )
        with self._lock:
            alerts = self.surveillance.record_order(ev)
        return [a.to_dict() for a in alerts]

    # ------------------------------------------------------------------ fills
    def on_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Any,
        price: Any,
        fee: Any = 0,
        financing: Any = 0,
        strategy_id: str = "",
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        asset_class: Optional[str] = None,
        is_aggressive: bool = True,
        account: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Book a fill across ALL records + feed surveillance. Atomic under lock."""
        side = str(side).lower()
        figi, ac = self._resolve(symbol)
        asset_class = asset_class or ac
        with self._lock:
            # 1) P&L ledger (positions / realized / unrealized / cash)
            led = self.ledger.on_fill(
                symbol,
                side,
                quantity,
                price,
                fee=fee,
                ts=ts,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
            )
            qf = float(led["quantity"])
            pf = float(led["price"])
            fef = float(led["fee"])
            signed_notional = (qf if side == "buy" else -qf) * pf

            # 2) immutable trade blotter (books-and-records)
            tr = self.blotter.record_trade(
                symbol=symbol,
                side=side,
                quantity=qf,
                price=pf,
                fee=fef,
                financing=financing,
                figi=figi,
                currency=self.ledger.base_currency,
                asset_class=asset_class,
                strategy_id=strategy_id,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                ts=ts,
            )

            # 3) cash general-ledger movements (mirror the ledger cash math)
            self.cash.post("TRADE", -signed_notional, ref=client_order_id, symbol=symbol, ts=ts)
            if fef:
                self.cash.post("FEE", -fef, ref=client_order_id, symbol=symbol, ts=ts)
            if float(financing or 0):
                self.cash.post(
                    "FINANCING", -float(financing), ref=client_order_id, symbol=symbol, ts=ts
                )
                self.ledger.accrue_financing(symbol, float(financing))

            # 4) live MAR surveillance over the same fill
            alerts: List[Dict[str, Any]] = []
            if self.surveillance is not None:
                from services.algo_integration.market_abuse import TradeEvent

                tev = TradeEvent(
                    ts_ms=(int(ts.timestamp() * 1000) if ts else _now_ms()),
                    symbol=str(symbol),
                    account=account or self.account_id,
                    side=side.upper(),
                    qty=qf,
                    price=pf,
                    is_aggressive=is_aggressive,
                    order_id=str(client_order_id or broker_order_id or ""),
                )
                alerts = [a.to_dict() for a in self.surveillance.record_trade(tev)]

        return {
            "ledger": led,
            "trade": tr.to_dict(),
            "figi": figi,
            "cash_balance": self.cash.balance,
            "alerts": alerts,
        }

    # ------------------------------------------------------------------ marks / eod
    def mark(self, symbol: str, price: Any) -> None:
        self.ledger.mark(symbol, price)

    def mark_prices(self, prices: Dict[str, Any]) -> None:
        self.ledger.mark_prices(prices)

    def eod_close(self) -> Dict[str, Any]:
        return self.ledger.eod_close().to_dict()

    # ------------------------------------------------------------------ views
    def verify_integrity(self) -> Dict[str, Any]:
        bl = self.blotter.verify()
        ca = self.cash.verify()
        # books reconcile: cash-ledger balance == P&L-ledger cash
        cash_match = abs(self.cash.balance - float(self.ledger.cash)) < 1e-6
        return {
            "blotter": bl,
            "cash": ca,
            "cash_ledger_matches_pnl_cash": cash_match,
            "all_valid": bool(bl["valid"] and ca["valid"] and cash_match),
        }

    def snapshot(self) -> Dict[str, Any]:
        out = {
            "pnl": self.ledger.snapshot(),
            "blotter": self.blotter.summary(),
            "cash_ledger": self.cash.summary(),
            "integrity": self.verify_integrity(),
        }
        if self.surveillance is not None:
            out["surveillance"] = {
                "summary": self.surveillance.summary(),
                "alerts": [a.to_dict() for a in self.surveillance.get_alerts()][-50:],
            }
        return out

    def recent_trades(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return self.blotter.trades(limit=limit)

    def recent_cash(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return self.cash.movements(limit=limit)

    # ------------------------------------------------------------------ wiring
    def fill_handler_callback(self, *, strategy_id: str = ""):
        """Return a ``FillHandler.on_fill`` callback that books each INCREMENTAL fill
        across all records + surveillance (cumulative-aware, like ledger_fill_callback)."""
        from decimal import Decimal

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
                qty = Decimal(str(inc))
            else:
                cum = Decimal(str(payload.get("filled_qty") or "0"))
                qty = cum - seen.get(coid, Decimal("0"))
                seen[coid] = cum
            if qty <= 0:
                return
            self.on_fill(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                strategy_id=strategy_id,
                client_order_id=coid or None,
                broker_order_id=payload.get("broker_order_id"),
            )

        return _cb

    def close(self) -> None:
        for c in (self.ledger, self.blotter, self.cash):
            try:
                c.close()
            except Exception:
                pass


__all__ = ["BooksAndRecords"]
