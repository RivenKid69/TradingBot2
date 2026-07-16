# -*- coding: utf-8 -*-
"""
adapters/binance/order_execution.py
Binance **Spot** order execution adapter.

Closes gap P0-C: the spot ``ExchangeVendor.BINANCE`` vendor had a
``MARKET_DATA`` / ``FEE`` / ``TRADING_HOURS`` / ``EXCHANGE_INFO`` adapter but
**no** ``ORDER_EXECUTION`` adapter — so ``create_order_execution_adapter(
ExchangeVendor.BINANCE, ...)`` raised, and the MVP panic/holdings/close paths
for crypto reported "unavailable". This module registers the missing spot
execution adapter so live/panic crypto-spot works through the same registry
path used by every other asset class.

Design mirrors ``futures_order_execution.py`` (same HMAC-SHA256 signing,
``RestBudgetSession`` transport) but targets the Binance **Spot** REST API:
    - POST   /api/v3/order        submit
    - DELETE /api/v3/order        cancel
    - GET    /api/v3/order        status
    - GET    /api/v3/openOrders   open orders
    - DELETE /api/v3/openOrders   cancel-all (per symbol)
    - GET    /api/v3/account      balances

Spot vs futures semantics (intentional differences):
    - No leverage / margin / positionSide / reduceOnly (long-only cash market).
    - "Positions" are **balances**: a non-zero base-asset balance is surfaced as
      a synthetic ``Position`` keyed by ``{asset}{quote_asset}`` (default quote
      ``USDT``) so that panic-flatten / close can market-SELL it. Cost basis is
      not provided by the spot account endpoint, so ``avg_entry_price`` is 0
      (unknown) — reported honestly, never fabricated.

References:
    - Binance Spot API: https://binance-docs.github.io/apidocs/spot/en/
    - New order: https://binance-docs.github.io/apidocs/spot/en/#new-order-trade
    - Account: https://binance-docs.github.io/apidocs/spot/en/#account-information-user_data
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core_models import (
    Order,
    ExecReport,
    Position,
    Side,
    OrderType,
    TimeInForce,
    ExecStatus,
)
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor, AccountInfo, MarketType

logger = logging.getLogger(__name__)

# Quote assets that are themselves cash, not tradable "positions" to flatten.
_DEFAULT_QUOTE_ASSETS = frozenset(
    {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD"}
)
# Tiny balances (dust) below this base-asset quantity are ignored when building
# synthetic positions — selling dust just errors on the exchange (min-notional).
_DUST_EPS = Decimal("0")


class BinanceOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Binance Spot order execution adapter.

    Handles spot order submission, cancellation, balance-derived positions and
    account info. Requires API keys with **Spot trading** permission.

    Configuration:
        api_key: Binance API key (required for trading)
        api_secret: Binance API secret (required for trading)
        spot_url: API base URL (default: https://api.binance.com)
        testnet: Use spot testnet (https://testnet.binance.vision)
        recv_window: Signed-request receive window in ms (default: 5000)
        timeout: HTTP timeout seconds (default: 30)
        quote_asset: Quote used to map balances→symbols for flatten (default USDT)

    Example:
        >>> adapter = BinanceOrderExecutionAdapter(
        ...     config={"api_key": "xxx", "api_secret": "yyy"}
        ... )
        >>> adapter.connect()
        >>> res = adapter.submit_order(Order(
        ...     ts=..., symbol="BTCUSDT", side=Side.BUY,
        ...     order_type=OrderType.MARKET, quantity=Decimal("0.001")))
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        self._api_key = str(self._config.get("api_key", ""))
        self._api_secret = str(self._config.get("api_secret", ""))

        if not self._api_key or not self._api_secret:
            logger.warning("No API keys provided - adapter will work in read-only mode")

        # Base URL (Binance US uses a different host via default_config)
        self._spot_url = str(
            self._config.get("spot_url")
            or self._config.get("base_url")
            or "https://api.binance.com"
        )
        if self._config.get("testnet", False):
            self._spot_url = "https://testnet.binance.vision"

        self._recv_window = int(self._config.get("recv_window", 5000))
        self._quote_asset = str(self._config.get("quote_asset", "USDT")).upper()

        # Lazy-initialized REST session
        self._session = None

    # ------------------------------------------------------------------ transport

    def _get_session(self):
        """Lazy initialization of REST session (shared budget transport)."""
        if self._session is None:
            from services.rest_budget import RestBudgetSession
            self._session = RestBudgetSession({
                "timeout": int(self._config.get("timeout", 30)),
            })
        return self._session

    def _do_connect(self) -> None:
        self._get_session()

    def _do_disconnect(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach timestamp/recvWindow and an HMAC-SHA256 signature."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
    ) -> Any:
        """Make a request to the Binance Spot API."""
        session = self._get_session()
        if params is None:
            params = {}
        if signed:
            params = self._sign_request(params)

        url = f"{self._spot_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self._api_key}
        timeout = int(self._config.get("timeout", 30))

        try:
            method_u = method.upper()
            if method_u == "GET":
                return session.get(url, params=params, headers=headers,
                                   timeout=timeout, budget="spot_api", tokens=1.0)
            if method_u == "POST":
                return session.post(url, data=params, headers=headers,
                                    timeout=timeout, budget="spot_api", tokens=1.0)
            if method_u == "DELETE":
                return session.delete(url, params=params, headers=headers,
                                      timeout=timeout, budget="spot_api", tokens=1.0)
            raise ValueError(f"Unsupported method: {method}")
        except Exception as e:
            logger.error(f"Spot API request failed: {e}")
            raise

    # ============================================================ standard interface

    def submit_order(self, order: Order) -> OrderResult:
        """Submit a spot order (MARKET or LIMIT via core_models.Order)."""
        try:
            params: Dict[str, Any] = {
                "symbol": order.symbol,
                "side": "BUY" if order.side == Side.BUY else "SELL",
                "type": "MARKET" if order.order_type == OrderType.MARKET else "LIMIT",
                "quantity": str(order.quantity),
            }

            cid = getattr(order, "client_order_id", None)
            if cid:
                params["newClientOrderId"] = str(cid)[:36]

            if order.order_type == OrderType.LIMIT:
                if order.price is not None:
                    params["price"] = str(order.price)
                tif = getattr(order, "time_in_force", TimeInForce.GTC)
                params["timeInForce"] = tif.value if hasattr(tif, "value") else str(tif)

            response = self._request("POST", "/api/v3/order", params)
            return self._parse_order_response(response)

        except Exception as e:
            return OrderResult(success=False, error_code="SUBMISSION_FAILED",
                               error_message=str(e))

    def submit_spot_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        *,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Submit a spot order with full control (incl. STOP_LOSS / TAKE_PROFIT).

        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            side: "BUY" or "SELL"
            order_type: MARKET, LIMIT, STOP_LOSS, STOP_LOSS_LIMIT,
                TAKE_PROFIT, TAKE_PROFIT_LIMIT, LIMIT_MAKER
            quantity: Base-asset quantity
            price: Limit price (LIMIT / *_LIMIT types)
            stop_price: Trigger price (STOP_* / TAKE_PROFIT_* types)
            time_in_force: GTC, IOC, FOK (for LIMIT-family types)
            client_order_id: Optional idempotency id
        """
        try:
            otype = order_type.upper()
            params: Dict[str, Any] = {
                "symbol": symbol,
                "side": side.upper(),
                "type": otype,
                "quantity": str(quantity),
                "newClientOrderId": (client_order_id or str(uuid.uuid4()))[:36],
            }
            # LIMIT-family types require price + timeInForce.
            if otype in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"):
                if price is not None:
                    params["price"] = str(price)
                if otype != "LIMIT_MAKER":
                    params["timeInForce"] = time_in_force.upper()
            # STOP / TAKE_PROFIT families require a trigger price.
            if otype in ("STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"):
                if stop_price is not None:
                    params["stopPrice"] = str(stop_price)

            response = self._request("POST", "/api/v3/order", params)
            return self._parse_order_response(response)
        except Exception as e:
            return OrderResult(success=False, error_code="SUBMISSION_FAILED",
                               error_message=str(e))

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> bool:
        """Cancel an open spot order (Binance requires the symbol)."""
        if not symbol:
            logger.error("Symbol required for spot order cancellation")
            return False
        try:
            params: Dict[str, Any] = {"symbol": symbol}
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                logger.error("Either order_id or client_order_id required")
                return False

            response = self._request("DELETE", "/api/v3/order", params)
            if isinstance(response, dict):
                status = str(response.get("status", ""))
                # A successful DELETE echoes the cancelled order; treat presence
                # of a CANCELED status (or an orderId with no error code) as ok.
                if response.get("code"):
                    return False
                return status in ("CANCELED", "CANCELLED", "PENDING_CANCEL") or bool(
                    response.get("orderId"))
            return False
        except Exception as e:
            logger.error(f"Spot order cancellation failed: {e}")
            return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """Get current status of a spot order (symbol required)."""
        if not symbol:
            logger.error("Symbol required for spot order status")
            return None
        try:
            params: Dict[str, Any] = {"symbol": symbol}
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                return None
            response = self._request("GET", "/api/v3/order", params)
            if isinstance(response, dict) and not response.get("code"):
                return self._parse_exec_report(response)
            return None
        except Exception as e:
            logger.error(f"Failed to get spot order status: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open spot orders (optionally filtered by symbol)."""
        try:
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol
            response = self._request("GET", "/api/v3/openOrders", params)
            orders: List[Order] = []
            for item in response if isinstance(response, list) else []:
                parsed = self._parse_order(item)
                if parsed:
                    orders.append(parsed)
            return orders
        except Exception as e:
            logger.error(f"Failed to get open spot orders: {e}")
            return []

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Position]:
        """
        Get spot "positions" derived from account balances.

        A non-zero base-asset balance (free+locked) is surfaced as a synthetic
        long ``Position`` keyed by ``{asset}{quote_asset}`` so panic/close can
        market-SELL it. Quote/stable assets are excluded (they are cash, not a
        position to flatten). ``avg_entry_price`` is 0 — the spot account
        endpoint does not provide cost basis, and we never fabricate one.
        """
        try:
            response = self._request("GET", "/api/v3/account", {})
            positions: Dict[str, Position] = {}
            if not isinstance(response, dict):
                return {}
            now_ms = int(time.time() * 1000)
            for bal in response.get("balances", []):
                if not isinstance(bal, dict):
                    continue
                asset = str(bal.get("asset", "")).upper()
                if not asset or asset in _DEFAULT_QUOTE_ASSETS or asset == self._quote_asset:
                    continue
                free = Decimal(str(bal.get("free", "0")))
                locked = Decimal(str(bal.get("locked", "0")))
                total = free + locked
                if total <= _DUST_EPS:
                    continue
                symbol = f"{asset}{self._quote_asset}"
                if symbols and symbol not in symbols and asset not in symbols:
                    continue
                positions[symbol] = Position(
                    symbol=symbol,
                    qty=total,
                    avg_entry_price=Decimal("0"),   # spot: cost basis not provided
                    realized_pnl=Decimal("0"),
                    fee_paid=Decimal("0"),
                    ts=now_ms,
                    meta={
                        "asset": asset,
                        "free": str(free),
                        "locked": str(locked),
                        "market": "spot",
                        "cost_basis_available": False,
                    },
                )
            return positions
        except Exception as e:
            logger.error(f"Failed to get spot balances: {e}")
            return {}

    def get_account_info(self) -> AccountInfo:
        """Get spot account info (quote-asset cash balance)."""
        try:
            response = self._request("GET", "/api/v3/account", {})
            if not isinstance(response, dict):
                return AccountInfo(vendor=self._vendor)

            cash = Decimal("0")
            for bal in response.get("balances", []):
                if isinstance(bal, dict) and str(bal.get("asset", "")).upper() == self._quote_asset:
                    cash = Decimal(str(bal.get("free", "0"))) + Decimal(str(bal.get("locked", "0")))
                    break

            return AccountInfo(
                vendor=self._vendor,
                account_id=str(response.get("accountType", "SPOT")),
                account_type="spot",
                vip_tier=0,
                buying_power=cash,
                cash_balance=cash,
                margin_enabled=False,
                raw_data=response,
            )
        except Exception as e:
            logger.error(f"Failed to get spot account info: {e}")
            return AccountInfo(vendor=self._vendor)

    # ============================================================ spot-specific

    def close_position(self, symbol: str) -> OrderResult:
        """
        Flatten a spot holding by market-SELLing its entire base balance.

        Spot is long-only, so "close" always means SELL. Returns a no-op
        success when there is nothing to sell.
        """
        positions = self.get_positions([symbol])
        pos = positions.get(symbol)
        if not pos or pos.qty <= 0:
            return OrderResult(success=True, status="NO_POSITION")

        order = Order(
            ts=int(time.time() * 1000),
            symbol=symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=abs(pos.qty),
        )
        return self.submit_order(order)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open spot orders.

        Binance requires a symbol for the batch endpoint, so with no symbol we
        enumerate open orders and cancel per-symbol.
        """
        if symbol:
            try:
                self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol})
                return -1  # batch cancel — exact count unknown
            except Exception as e:
                logger.error(f"Failed to cancel all spot orders: {e}")
                return 0

        orders = self.get_open_orders()
        symbols = {o.symbol for o in orders}
        cancelled = 0
        for sym in symbols:
            try:
                self._request("DELETE", "/api/v3/openOrders", {"symbol": sym})
                cancelled += sum(1 for o in orders if o.symbol == sym)
            except Exception:
                pass
        return cancelled

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """Get last traded price (public endpoint, unsigned)."""
        try:
            response = self._request("GET", "/api/v3/ticker/price",
                                     {"symbol": symbol}, signed=False)
            if isinstance(response, dict) and response.get("price"):
                return Decimal(str(response["price"]))
            return None
        except Exception:
            return None

    # ============================================================ parse helpers

    def _parse_order_response(self, response: Dict[str, Any]) -> OrderResult:
        """Parse a spot order submission response."""
        if not isinstance(response, dict):
            return OrderResult(success=False, error_message="Invalid response")
        if response.get("code"):
            return OrderResult(
                success=False,
                error_code=str(response.get("code")),
                error_message=str(response.get("msg", "")),
                raw_response=response,
            )
        # Spot fills report cumulative quote qty; derive avg price when possible.
        executed_qty = Decimal(str(response.get("executedQty", "0")))
        cummulative_quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        avg_price: Optional[Decimal] = None
        if executed_qty > 0 and cummulative_quote > 0:
            avg_price = cummulative_quote / executed_qty

        return OrderResult(
            success=True,
            order_id=str(response.get("orderId", "")),
            client_order_id=str(response.get("clientOrderId", "")),
            status=str(response.get("status", "")),
            filled_qty=executed_qty,
            filled_price=avg_price,
            raw_response=response,
        )

    def _parse_exec_report(self, response: Dict[str, Any]) -> Optional[ExecReport]:
        """Parse a spot order status response into an ExecReport."""
        if not isinstance(response, dict):
            return None
        status_str = str(response.get("status", "")).upper()
        if status_str == "FILLED":
            exec_status = ExecStatus.FILLED
        elif status_str == "PARTIALLY_FILLED":
            exec_status = ExecStatus.PARTIALLY_FILLED
        elif status_str in ("NEW", "PENDING_NEW"):
            exec_status = ExecStatus.NEW
        elif status_str in ("CANCELED", "CANCELLED"):
            exec_status = ExecStatus.CANCELED
        elif status_str in ("REJECTED", "EXPIRED"):
            exec_status = ExecStatus.REJECTED
        else:
            exec_status = ExecStatus.NEW

        side = Side.BUY if str(response.get("side", "")).upper() == "BUY" else Side.SELL
        executed_qty = Decimal(str(response.get("executedQty", "0")))
        cummulative_quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        price = Decimal(str(response.get("price", "0")))
        if executed_qty > 0 and cummulative_quote > 0:
            price = cummulative_quote / executed_qty

        return ExecReport(
            ts=int(response.get("updateTime", int(time.time() * 1000))),
            run_id="binance_spot",
            symbol=str(response.get("symbol", "")),
            side=side,
            order_type=OrderType.MARKET if str(response.get("type", "")).upper() == "MARKET" else OrderType.LIMIT,
            price=price,
            quantity=executed_qty,
            fee=Decimal("0"),
            fee_asset=self._quote_asset,
            exec_status=exec_status,
            client_order_id=str(response.get("clientOrderId")),
            order_id=str(response.get("orderId")),
        )

    def _parse_order(self, response: Dict[str, Any]) -> Optional[Order]:
        """Parse open-order data into a core_models.Order."""
        if not isinstance(response, dict):
            return None
        side = Side.BUY if str(response.get("side", "")).upper() == "BUY" else Side.SELL
        otype = OrderType.MARKET if str(response.get("type", "")).upper() == "MARKET" else OrderType.LIMIT
        try:
            tif = TimeInForce(response.get("timeInForce", "GTC"))
        except ValueError:
            tif = TimeInForce.GTC
        return Order(
            ts=int(response.get("time", int(time.time() * 1000))),
            symbol=str(response.get("symbol", "")),
            side=side,
            order_type=otype,
            quantity=Decimal(str(response.get("origQty", "0"))),
            price=Decimal(str(response.get("price", "0"))) if response.get("price") else None,
            time_in_force=tif,
            client_order_id=str(response.get("clientOrderId", "")),
            meta={
                "order_id": response.get("orderId"),
                "status": response.get("status"),
                "stop_price": response.get("stopPrice"),
            },
        )

    @property
    def market_type(self) -> MarketType:
        """Return the market type this adapter serves."""
        return MarketType.CRYPTO_SPOT
