# -*- coding: utf-8 -*-
"""
Live execution stack factory - AGENT ZONE ONLY.

Assembles the full XS->Agent->broker pipeline from any ``BrokerConnector``:

    LiveExecutionEngine  (journaled, idempotent OMS, policy/hard-cap/risk stack)
        |  broker_submit  -> connector.submit_order
        v
    FillHandler  <- FillSource (polling connector.get_order, or a push source)
        |  on_child_fill
        v
    ClockDrivenChildExecutor (optional: TWAP/VWAP/POV slicing + cancel-replace)
        ^
    AgentClient.send_intents(IntentBatch)   <- CrossSectionalLiveRunner (Cloud)

This is what makes the CCEA boundary call ``agent_client.send_intents`` real:
target exposures -> deltas -> local OrderIntents -> broker orders -> OMS fills.

PROHIBITED in Cloud zone.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from packages.agent.broker.protocol import (
    BrokerConnector,
    OrderRequest,
    OrderSide,
    OrderType as BkOrderType,
    TimeInForce,
)
from packages.agent.execution.engine import (
    LiveExecutionEngine,
    Order,
    OrderType as EngOrderType,
    BrokerSubmitFn,
)
from packages.agent.execution.agent_client import AgentClient
from packages.agent.execution.fill_handler import FillHandler, PollingFillSource
from packages.agent.execution.child_executor import ClockDrivenChildExecutor
from packages.agent.reconciliation.journal import OrderJournal

logger = logging.getLogger(__name__)

_ENG_TO_BK_TYPE = {
    EngOrderType.MARKET: BkOrderType.MARKET,
    EngOrderType.LIMIT: BkOrderType.LIMIT,
    EngOrderType.STOP: BkOrderType.STOP,
    EngOrderType.STOP_LIMIT: BkOrderType.STOP_LIMIT,
}

_TIF = {
    "GTC": TimeInForce.GTC,
    "DAY": TimeInForce.DAY,
    "IOC": TimeInForce.IOC,
    "FOK": TimeInForce.FOK,
    "GTD": TimeInForce.GTD,
    "OPG": TimeInForce.OPG,
    "CLS": TimeInForce.CLS,
}


def make_broker_submit(connector: BrokerConnector) -> BrokerSubmitFn:
    """Adapt an engine ``Order`` -> broker ``OrderRequest`` -> (ok, broker_id, err)."""

    def _submit(order: Order):
        req = OrderRequest(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            order_type=_ENG_TO_BK_TYPE.get(order.order_type, BkOrderType.MARKET),
            quantity=Decimal(order.quantity),
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=_TIF.get(str(order.time_in_force).upper(), TimeInForce.GTC),
            strategy_id=getattr(order, "strategy_id", None),
        )
        try:
            res = connector.submit_order(req)
        except Exception as exc:  # pragma: no cover - defensive
            return False, None, f"broker error: {exc}"
        return bool(res.success), res.broker_order_id, res.error_message

    return _submit


def make_broker_cancel(connector: BrokerConnector):
    """(client_order_id, broker_order_id) -> bool cancel adapter."""

    def _cancel(client_order_id: str, broker_order_id: Optional[str]) -> bool:
        try:
            res = connector.cancel_order(
                client_order_id=client_order_id, broker_order_id=broker_order_id
            )
            return bool(res.success)
        except Exception as exc:  # pragma: no cover
            logger.warning("cancel failed for %s: %s", client_order_id, exc)
            return False

    return _cancel


def make_broker_replace(connector: BrokerConnector):
    """(client_order_id, new_qty, new_price) -> (ok, broker_id, err) amend adapter (FIX 35=G)."""

    def _replace(client_order_id: str, new_qty: Optional[Decimal], new_price: Optional[Decimal]):
        fn = getattr(connector, "replace_order", None)
        if fn is None:
            return False, None, "broker does not support replace_order"
        try:
            res = fn(client_order_id, quantity=new_qty, limit_price=new_price)
            return bool(res.success), res.broker_order_id, res.error_message
        except Exception as exc:  # pragma: no cover
            return False, None, f"broker replace error: {exc}"

    return _replace


class ConnectorPositionProvider:
    """get_positions() -> {symbol: signed market-value notional} from the broker."""

    def __init__(self, connector: BrokerConnector) -> None:
        self._c = connector

    def get_positions(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            for pos in self._c.get_positions() or []:
                mv = pos.market_value
                if mv is None and pos.current_price is not None:
                    mv = pos.quantity * pos.current_price
                if mv is not None:
                    out[pos.symbol] = float(mv)
        except Exception as exc:  # pragma: no cover
            logger.warning("get_positions failed: %s", exc)
        return out


class ConnectorPricesProvider:
    """get_prices() -> {symbol: last price}; ``prime(symbols)`` fetches on demand."""

    def __init__(self, connector: BrokerConnector, symbols: Optional[List[str]] = None) -> None:
        self._c = connector
        self._symbols: set = set(symbols or [])
        self._cache: Dict[str, float] = {}

    def prime(self, symbols: List[str]) -> None:
        self._symbols.update(str(s) for s in symbols)

    def get_prices(self) -> Dict[str, float]:
        # include any symbols we currently hold a position in
        syms = set(self._symbols)
        try:
            for pos in self._c.get_positions() or []:
                syms.add(pos.symbol)
        except Exception:  # pragma: no cover
            pass
        out: Dict[str, float] = {}
        for s in syms:
            try:
                px = self._c.get_last_price(s)
            except Exception:  # pragma: no cover
                px = None
            if px is not None and float(px) > 0:
                out[s] = float(px)
        self._cache.update(out)
        return dict(self._cache)


def make_fill_fetch(connector: BrokerConnector):
    """fetch_order(client_order_id) -> normalized dict for PollingFillSource."""

    def _fetch(client_order_id: str) -> Optional[Dict[str, Any]]:
        info = connector.get_order(client_order_id=client_order_id)
        if info is None:
            return None
        return {
            "client_order_id": info.client_order_id,
            "broker_order_id": info.broker_order_id,
            "status": info.status.value,
            "filled_qty": str(info.filled_quantity),
            "filled_avg_price": (
                str(info.avg_fill_price) if info.avg_fill_price is not None else None
            ),
            "cumulative": True,
        }

    return _fetch


class BrokerLiquidityProvider:
    """LiquidityProvider (for SOR) backed by broker connectors' top-of-book.

    Uses a venue's order book when available, else synthesizes a tight quote from the
    last price (paper). This is the production feed that lets the SOR route on LIVE
    liquidity instead of static assumptions (P1 #7)."""

    def __init__(self, connector_map: Dict[str, BrokerConnector]) -> None:
        self._m = dict(connector_map)

    def get_quote(self, venue: str, symbol: str):
        from packages.agent.execution.smart_order_router import VenueQuote

        c = self._m.get(venue)
        if c is None:
            return None
        # prefer a real order book if the connector exposes one
        book = getattr(c, "get_order_book", None)
        if callable(book):
            try:
                ob = book(symbol)
                if ob is not None:
                    bid = float(getattr(ob, "best_bid", 0) or 0)
                    ask = float(getattr(ob, "best_ask", 0) or 0)
                    if bid > 0 and ask > 0:
                        return VenueQuote(
                            bid=bid,
                            ask=ask,
                            bid_size=float(getattr(ob, "bid_size", 1e6 / ask) or 1e6 / ask),
                            ask_size=float(getattr(ob, "ask_size", 1e6 / ask) or 1e6 / ask),
                        )
            except Exception:
                pass
        try:
            px = c.get_last_price(symbol)
        except Exception:
            px = None
        if px is None or float(px) <= 0:
            return None
        px = float(px)
        return VenueQuote(bid=px * 0.99995, ask=px * 1.00005, bid_size=1e6 / px, ask_size=1e6 / px)


def make_venue_submit(
    connector_map: Dict[str, BrokerConnector],
    price_fn: Optional[Callable[[str], Optional[float]]] = None,
):
    """submit_fn(venue, symbol, side, notional) -> dict — dispatch a child order to a venue."""
    from decimal import Decimal as _D

    def _submit(venue: str, symbol: str, side: str, notional: float) -> Dict[str, Any]:
        c = connector_map.get(venue)
        if c is None:
            return {"success": False, "error": f"no connector for venue {venue}"}
        px = price_fn(symbol) if price_fn else None
        if not px:
            try:
                px = float(c.get_last_price(symbol))
            except Exception:
                px = None
        if not px or px <= 0:
            return {"success": False, "error": "no price"}
        qty = abs(float(notional)) / float(px)
        coid = f"sor_{venue}_{symbol}_{abs(hash((venue, symbol, round(notional, 2)))) % 10**8}"
        req = OrderRequest(
            client_order_id=coid,
            symbol=symbol,
            side=OrderSide.BUY if str(side).upper() == "BUY" else OrderSide.SELL,
            order_type=BkOrderType.MARKET,
            quantity=_D(str(qty)),
        )
        try:
            res = c.submit_order(req)
            return {
                "success": bool(res.success),
                "broker_order_id": res.broker_order_id,
                "venue": venue,
                "qty": qty,
            }
        except Exception as exc:  # pragma: no cover
            return {"success": False, "error": str(exc), "venue": venue}

    return _submit


def routed_broker_submit(
    sor,
    connector_map: Dict[str, BrokerConnector],
    price_fn: Optional[Callable[[str], Optional[float]]] = None,
    provider: Any = None,
    *,
    record: Optional[List[Dict[str, Any]]] = None,
):
    """A ``BrokerSubmitFn`` that routes the order's notional across venues via the
    SmartOrderRouter and DISPATCHES child orders — making SOR part of the live path.

    SOR decides WHERE/HOW MUCH; each venue connector creates the order locally
    (CCEA boundary preserved)."""
    venue_submit = make_venue_submit(connector_map, price_fn)

    def _submit(order: Order):
        px = (
            float(order.limit_price)
            if order.limit_price
            else (price_fn(order.symbol) if price_fn else None)
        )
        if not px:
            for c in connector_map.values():
                try:
                    px = float(c.get_last_price(order.symbol))
                    break
                except Exception:
                    continue
        if not px or px <= 0:
            return False, None, "no price for routing"
        notional = abs(float(order.quantity)) * float(px)
        side = "BUY" if order.side == "buy" else "SELL"
        route = (
            sor.route_live(order.symbol, side, notional, provider)
            if provider is not None
            else sor.route(order.symbol, side, notional)
        )
        disp = sor.dispatch(route, venue_submit)
        if record is not None:
            record.append(
                {
                    "client_order_id": order.client_order_id,
                    "route": route.to_dict(),
                    "dispatch": disp,
                }
            )
        broker_id = "routed:" + ",".join(a.venue for a in route.allocations) or "routed"
        return (
            bool(disp.get("all_ok")),
            broker_id,
            (None if disp.get("all_ok") else "some venue dispatches failed"),
        )

    return _submit


def build_live_stack(
    connector: BrokerConnector,
    *,
    strategy_id: str = "xs_cross_sectional",
    n_slices: int = 1,
    slice_weights: Optional[List[float]] = None,
    min_trade_notional: float = 1.0,
    journal_path: Optional[str] = None,
    deployment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    slice_interval_s: float = 30.0,
    straggler_timeout_s: float = 60.0,
    symbols: Optional[List[str]] = None,
    use_limit_orders: bool = False,
    clock: Any = None,
    pnl_ledger: Any = None,
    sor: Any = None,
    venue_connectors: Optional[Dict[str, BrokerConnector]] = None,
    use_live_liquidity: bool = True,
) -> Dict[str, Any]:
    """Assemble and wire the complete live execution stack.

    Returns a dict with: engine, agent_client, fill_handler, fill_source,
    child_executor (or None), broker.

    If ``pnl_ledger`` (a ``PnLLedger``) is given, every processed fill is booked
    into it (realized/unrealized/fees) and the engine's pre-trade portfolio is
    driven from the ledger's OWN equity.
    """
    from pathlib import Path

    journal = OrderJournal(db_path=Path(journal_path) if journal_path else None)
    prices_provider = ConnectorPricesProvider(connector, symbols=symbols)
    position_provider = ConnectorPositionProvider(connector)

    # P1 #7: when a SmartOrderRouter is supplied, route every order across venues and
    # dispatch child orders (SOR now part of the live submission path, not dead code).
    sor_routes: List[Dict[str, Any]] = []
    if sor is not None:
        vmap = dict(venue_connectors or {})
        if not vmap:
            # single-venue: route to this connector under its broker name
            vmap = {getattr(connector, "broker_name", "broker"): connector}
        provider = BrokerLiquidityProvider(vmap) if use_live_liquidity else None
        _price_fn = lambda s: prices_provider.get_prices().get(s)  # noqa: E731
        _submit = routed_broker_submit(sor, vmap, _price_fn, provider, record=sor_routes)
    else:
        _submit = make_broker_submit(connector)

    engine = LiveExecutionEngine(
        broker_submit=_submit,
        broker_cancel=make_broker_cancel(connector),
        broker_replace=make_broker_replace(connector),
        broker_name=getattr(connector, "broker_name", "broker"),
        order_journal=journal,
        deployment_id=deployment_id,
        run_id=run_id,
    )

    child_executor = None
    if n_slices and int(n_slices) > 1:
        child_executor = ClockDrivenChildExecutor(
            engine,
            prices_provider=prices_provider,
            broker_cancel=make_broker_cancel(connector),
            strategy_id=strategy_id,
            slice_interval_s=slice_interval_s,
            straggler_timeout_s=straggler_timeout_s,
            use_limit_orders=use_limit_orders,
        )

    fill_source = PollingFillSource(fetch_order=make_fill_fetch(connector))
    on_fill_cb = None
    if pnl_ledger is not None:
        from packages.agent.accounting.pnl_ledger import ledger_fill_callback

        on_fill_cb = ledger_fill_callback(pnl_ledger)
    fill_handler = FillHandler(
        engine,
        on_fill=on_fill_cb,
        on_child_fill=(child_executor.on_child_fill if child_executor is not None else None),
    )

    agent_client = AgentClient(
        engine,
        prices_provider=prices_provider,
        position_provider=position_provider,
        strategy_id=strategy_id,
        min_trade_notional=min_trade_notional,
        child_executor=child_executor,
        n_slices=int(n_slices),
        slice_weights=slice_weights,
        fill_handler=fill_handler,
        fill_source=fill_source,
        clock=clock,
    )

    return {
        "engine": engine,
        "agent_client": agent_client,
        "fill_handler": fill_handler,
        "fill_source": fill_source,
        "child_executor": child_executor,
        "broker": connector,
        "sor": sor,
        "sor_routes": sor_routes,  # live routing+dispatch record (P1 #7)
    }


__all__ = [
    "build_live_stack",
    "make_broker_submit",
    "make_broker_cancel",
    "make_broker_replace",
    "make_fill_fetch",
    "make_venue_submit",
    "routed_broker_submit",
    "BrokerLiquidityProvider",
    "ConnectorPositionProvider",
    "ConnectorPricesProvider",
]
