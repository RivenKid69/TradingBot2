# -*- coding: utf-8 -*-
"""
services/algo_integration/market_abuse.py
=========================================

Market-abuse surveillance (P2 #20) — MAR (EU 596/2014) Art. 12/16 & RTS 6.

OTR alone is not adequate surveillance. This adds heuristic detectors for the four
classic manipulation patterns, fed real order/trade events:

  * **Spoofing** — large orders placed then cancelled fast, away from touch, never
    (or rarely) executed (intent to mislead the book).
  * **Layering** — multiple same-side orders stacked across levels, cancelled shortly
    after a fill on the OPPOSITE side.
  * **Wash trading** — near-simultaneous buy & sell of the same instrument by the same
    account at ~the same price (no change in beneficial ownership).
  * **Marking-the-close** — aggressive trades in the closing window that push the price
    in the account's favourable direction.

Deterministic, in-process, unit-testable. Emits ``MarketAbuseAlert``s. Designed to be
fed from the live execution path (order placed/cancelled/filled).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderEvent:
    ts_ms: int
    symbol: str
    account: str
    side: str                # BUY | SELL
    action: str              # NEW | CANCEL | MODIFY
    qty: float
    price: float
    order_id: str
    mid: Optional[float] = None   # prevailing mid at event time (for distance-from-touch)


@dataclass
class TradeEvent:
    ts_ms: int
    symbol: str
    account: str
    side: str
    qty: float
    price: float
    is_aggressive: bool = True
    order_id: str = ""


@dataclass
class MarketAbuseAlert:
    pattern: str             # spoofing | layering | wash_trade | marking_the_close
    symbol: str
    account: str
    severity: str            # LOW | MEDIUM | HIGH
    ts_ms: int
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"pattern": self.pattern, "symbol": self.symbol, "account": self.account,
                "severity": self.severity, "ts_ms": self.ts_ms, "detail": self.detail,
                "evidence": self.evidence}


@dataclass
class MarketAbuseConfig:
    window_ms: int = 60_000                  # sliding window for pattern detection
    # spoofing
    spoof_large_qty: float = 1000.0          # "large" order threshold
    spoof_cancel_ms: int = 5_000             # cancelled within this ⇒ fast cancel
    spoof_min_distance_bps: float = 5.0      # placed at least this far from mid
    spoof_min_events: int = 3                # need this many to alert
    # layering
    layering_min_orders: int = 4             # same-side orders stacked
    layering_cancel_ratio: float = 0.7       # fraction cancelled after opposite fill
    # wash
    wash_price_tol_bps: float = 5.0
    wash_window_ms: int = 10_000
    # marking-the-close
    close_window_ms: int = 300_000           # last 5 min before close
    market_close_ms_of_day: int = 20 * 3_600_000  # 20:00 UTC default (override per venue)
    marking_min_notional: float = 50_000.0


class MarketAbuseMonitor:
    """Heuristic MAR surveillance over a sliding window of order/trade events."""

    def __init__(self, config: Optional[MarketAbuseConfig] = None) -> None:
        self.cfg = config or MarketAbuseConfig()
        self._orders: Dict[str, Deque[OrderEvent]] = defaultdict(lambda: deque(maxlen=5000))
        self._trades: Dict[str, Deque[TradeEvent]] = defaultdict(lambda: deque(maxlen=5000))
        self._open: Dict[str, OrderEvent] = {}        # order_id -> NEW event still open
        self._filled_ids: set = set()
        self._alerts: List[MarketAbuseAlert] = []

    # ------------------------------------------------------------------
    def record_order(self, ev: OrderEvent) -> List[MarketAbuseAlert]:
        self._orders[ev.symbol].append(ev)
        if ev.action == "NEW":
            self._open[ev.order_id] = ev
        elif ev.action == "CANCEL":
            alerts = self._check_spoofing(ev)
            alerts += self._check_layering(ev)
            self._open.pop(ev.order_id, None)
            self._emit(alerts)
            return alerts
        return []

    def record_trade(self, ev: TradeEvent) -> List[MarketAbuseAlert]:
        self._trades[ev.symbol].append(ev)
        if ev.order_id:
            self._filled_ids.add(ev.order_id)
        alerts = self._check_wash(ev) + self._check_marking_the_close(ev)
        self._emit(alerts)
        return alerts

    def _emit(self, alerts: List[MarketAbuseAlert]) -> None:
        for a in alerts:
            self._alerts.append(a)
            logger.warning("MARKET ABUSE [%s] %s/%s: %s", a.pattern, a.symbol, a.account, a.detail)

    # ------------------------------------------------------------------
    def _check_spoofing(self, cancel_ev: OrderEvent) -> List[MarketAbuseAlert]:
        orig = self._open.get(cancel_ev.order_id)
        if orig is None or orig.action != "NEW":
            return []
        fast = (cancel_ev.ts_ms - orig.ts_ms) <= self.cfg.spoof_cancel_ms
        large = orig.qty >= self.cfg.spoof_large_qty
        never_filled = orig.order_id not in self._filled_ids
        far = True
        if orig.mid and orig.mid > 0:
            dist_bps = abs(orig.price - orig.mid) / orig.mid * 1e4
            far = dist_bps >= self.cfg.spoof_min_distance_bps
        if not (fast and large and never_filled and far):
            return []
        # count recent spoof-like cancels by this account/symbol
        cnt = self._recent_spoof_count(cancel_ev)
        if cnt < self.cfg.spoof_min_events:
            return []
        return [MarketAbuseAlert(
            "spoofing", cancel_ev.symbol, cancel_ev.account,
            "HIGH" if cnt >= 2 * self.cfg.spoof_min_events else "MEDIUM", cancel_ev.ts_ms,
            f"{cnt} large orders placed away from mid and cancelled within "
            f"{self.cfg.spoof_cancel_ms}ms without execution",
            {"count": cnt, "qty": orig.qty, "price": orig.price, "mid": orig.mid})]

    def _recent_spoof_count(self, ev: OrderEvent) -> int:
        cnt = 0
        seen = {}
        for o in self._orders[ev.symbol]:
            if o.account != ev.account:
                continue
            if o.action == "NEW" and o.qty >= self.cfg.spoof_large_qty:
                seen[o.order_id] = o
            elif o.action == "CANCEL" and o.order_id in seen:
                orig = seen[o.order_id]
                if (o.ts_ms - orig.ts_ms) <= self.cfg.spoof_cancel_ms and orig.order_id not in self._filled_ids:
                    if (ev.ts_ms - o.ts_ms) <= self.cfg.window_ms:
                        cnt += 1
        return cnt

    def _check_layering(self, cancel_ev: OrderEvent) -> List[MarketAbuseAlert]:
        # opposite-side fill recently, then a burst of same-side cancels
        opp = "SELL" if cancel_ev.side == "BUY" else "BUY"
        recent_opp_fill = any(
            t.account == cancel_ev.account and t.side == opp
            and (cancel_ev.ts_ms - t.ts_ms) <= self.cfg.window_ms
            for t in self._trades[cancel_ev.symbol]
        )
        if not recent_opp_fill:
            return []
        same_side_news = [o for o in self._orders[cancel_ev.symbol]
                          if o.account == cancel_ev.account and o.side == cancel_ev.side
                          and o.action == "NEW" and (cancel_ev.ts_ms - o.ts_ms) <= self.cfg.window_ms]
        same_side_cancels = [o for o in self._orders[cancel_ev.symbol]
                             if o.account == cancel_ev.account and o.side == cancel_ev.side
                             and o.action == "CANCEL" and (cancel_ev.ts_ms - o.ts_ms) <= self.cfg.window_ms]
        if len(same_side_news) < self.cfg.layering_min_orders:
            return []
        ratio = len(same_side_cancels) / max(1, len(same_side_news))
        if ratio < self.cfg.layering_cancel_ratio:
            return []
        return [MarketAbuseAlert(
            "layering", cancel_ev.symbol, cancel_ev.account, "HIGH", cancel_ev.ts_ms,
            f"{len(same_side_news)} {cancel_ev.side} orders layered then {ratio:.0%} cancelled "
            f"after an opposite-side fill",
            {"orders": len(same_side_news), "cancel_ratio": ratio})]

    def _check_wash(self, ev: TradeEvent) -> List[MarketAbuseAlert]:
        opp = "SELL" if ev.side == "BUY" else "BUY"
        for t in reversed(self._trades[ev.symbol]):
            if t is ev:
                continue
            if (ev.ts_ms - t.ts_ms) > self.cfg.wash_window_ms:
                break
            if t.account != ev.account or t.side != opp:
                continue
            price_close = (ev.price > 0 and abs(t.price - ev.price) / ev.price * 1e4 <= self.cfg.wash_price_tol_bps)
            if price_close and min(t.qty, ev.qty) > 0:
                return [MarketAbuseAlert(
                    "wash_trade", ev.symbol, ev.account, "HIGH", ev.ts_ms,
                    "same account bought and sold the instrument at ~same price within "
                    f"{self.cfg.wash_window_ms}ms (no change in beneficial ownership)",
                    {"buy_price": ev.price if ev.side == "BUY" else t.price,
                     "sell_price": t.price if ev.side == "BUY" else ev.price,
                     "qty": float(min(t.qty, ev.qty))})]
        return []

    def _check_marking_the_close(self, ev: TradeEvent) -> List[MarketAbuseAlert]:
        ms_of_day = ev.ts_ms % 86_400_000
        until_close = self.cfg.market_close_ms_of_day - ms_of_day
        if not (0 <= until_close <= self.cfg.close_window_ms):
            return []
        if not ev.is_aggressive:
            return []
        notional = ev.qty * ev.price
        if notional < self.cfg.marking_min_notional:
            return []
        return [MarketAbuseAlert(
            "marking_the_close", ev.symbol, ev.account, "MEDIUM", ev.ts_ms,
            f"aggressive {ev.side} of ${notional:,.0f} within {self.cfg.close_window_ms//1000}s "
            f"of the close — potential price marking",
            {"notional": notional, "ms_to_close": until_close})]

    # ------------------------------------------------------------------
    def get_alerts(self, *, pattern: Optional[str] = None) -> List[MarketAbuseAlert]:
        if pattern:
            return [a for a in self._alerts if a.pattern == pattern]
        return list(self._alerts)

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for a in self._alerts:
            out[a.pattern] += 1
        return dict(out)


__all__ = ["OrderEvent", "TradeEvent", "MarketAbuseAlert", "MarketAbuseConfig", "MarketAbuseMonitor"]
