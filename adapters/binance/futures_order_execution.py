# -*- coding: utf-8 -*-
"""
adapters/binance/futures_order_execution.py
Binance Futures order execution adapter.

Provides futures-specific order execution including:
- Market and limit orders with leverage
- Stop orders (stop market, take profit)
- Position management (long/short)
- Margin mode switching (cross/isolated)
- Leverage adjustment

Note: This adapter requires API keys with Futures trading permissions.

References:
    - Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
    - Order Types: https://binance-docs.github.io/apidocs/futures/en/#new-order-trade
    - Position Mode: https://binance-docs.github.io/apidocs/futures/en/#change-position-mode
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
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
    Liquidity,
)
from core_futures import (
    FuturesPosition,
    FuturesAccountState,
    FuturesFill,
    FuturesOrder,
    MarginMode,
    PositionSide,
    OrderSide as FuturesOrderSide,
    OrderType as FuturesOrderType,
    TimeInForce as FuturesTimeInForce,
    WorkingType,
)
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor, AccountInfo, MarketType

logger = logging.getLogger(__name__)


@dataclass
class FuturesOrderResult(OrderResult):
    """Extended order result for futures."""

    position_side: Optional[str] = None
    realized_pnl: Decimal = Decimal("0")
    margin_impact: Decimal = Decimal("0")
    commission_asset: str = "USDT"
    futures_fill: Optional[FuturesFill] = None


class BinanceFuturesOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Binance Futures order execution adapter.

    Handles order submission, position management, and margin settings.
    Requires API keys with Futures trading permissions.

    Configuration:
        api_key: Binance API key (required)
        api_secret: Binance API secret (required)
        futures_url: API base URL (default: https://fapi.binance.com)
        testnet: Use testnet URL (default: False)
        recv_window: Request timeout in ms (default: 5000)
        hedge_mode: Use hedge mode (separate long/short positions)

    Example:
        >>> adapter = BinanceFuturesOrderExecutionAdapter(
        ...     config={"api_key": "xxx", "api_secret": "yyy"}
        ... )
        >>> adapter.connect()
        >>> result = adapter.submit_futures_order(
        ...     symbol="BTCUSDT",
        ...     side="BUY",
        ...     order_type="MARKET",
        ...     quantity=Decimal("0.001"),
        ...     leverage=10,
        ... )
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

        # URLs
        self._futures_url = self._config.get("futures_url", "https://fapi.binance.com")
        if self._config.get("testnet", False):
            self._futures_url = "https://testnet.binancefuture.com"

        self._recv_window = int(self._config.get("recv_window", 5000))
        self._hedge_mode = bool(self._config.get("hedge_mode", False))

        # Lazy-initialized REST session
        self._session = None

    def _get_session(self):
        """Lazy initialization of REST session."""
        if self._session is None:
            from services.rest_budget import RestBudgetSession

            self._session = RestBudgetSession(
                {
                    "timeout": int(self._config.get("timeout", 30)),
                }
            )
        return self._session

    def _do_connect(self) -> None:
        """Initialize session connection."""
        self._get_session()

    def _do_disconnect(self) -> None:
        """Close session connection."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sign request with API secret."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window

        # Create query string
        query_string = urllib.parse.urlencode(params)

        # Sign
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
        """Make authenticated request to Binance Futures API."""
        session = self._get_session()

        if params is None:
            params = {}

        if signed:
            params = self._sign_request(params)

        url = f"{self._futures_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self._api_key}

        try:
            if method.upper() == "GET":
                response = session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=int(self._config.get("timeout", 30)),
                    budget="futures_api",
                    tokens=1.0,
                )
            elif method.upper() == "POST":
                response = session.post(
                    url,
                    data=params,
                    headers=headers,
                    timeout=int(self._config.get("timeout", 30)),
                    budget="futures_api",
                    tokens=1.0,
                )
            elif method.upper() == "DELETE":
                response = session.delete(
                    url,
                    params=params,
                    headers=headers,
                    timeout=int(self._config.get("timeout", 30)),
                    budget="futures_api",
                    tokens=1.0,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            return response

        except Exception as e:
            logger.error(f"Futures API request failed: {e}")
            raise

    # ========================================================================
    # Standard OrderExecutionAdapter methods
    # ========================================================================

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to Binance Futures.

        Converts core_models.Order to Binance Futures API format.

        Args:
            order: Order to submit

        Returns:
            OrderResult with execution status
        """
        try:
            params: Dict[str, Any] = {
                "symbol": order.symbol,
                "side": "BUY" if order.side == Side.BUY else "SELL",
                "type": self._convert_order_type(order.order_type),
                "quantity": str(order.quantity),
            }

            if order.client_order_id:
                params["newClientOrderId"] = order.client_order_id

            if order.order_type == OrderType.LIMIT and order.price:
                params["price"] = str(order.price)
                params["timeInForce"] = order.time_in_force.value

            if order.reduce_only:
                params["reduceOnly"] = "true"

            if self._hedge_mode:
                # Default to LONG for BUY, SHORT for SELL in hedge mode
                if order.side == Side.BUY:
                    params["positionSide"] = "LONG"
                else:
                    params["positionSide"] = "SHORT"

            response = self._request("POST", "/fapi/v1/order", params)

            return self._parse_order_response(response)

        except Exception as e:
            return OrderResult(
                success=False,
                error_code="SUBMISSION_FAILED",
                error_message=str(e),
            )

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            symbol: Symbol (required for Binance)

        Returns:
            True if cancellation successful
        """
        if not symbol:
            logger.error("Symbol required for order cancellation")
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

            response = self._request("DELETE", "/fapi/v1/order", params)

            if isinstance(response, dict):
                status = response.get("status", "")
                return status in ("CANCELED", "CANCELLED", "PENDING_CANCEL")

            return False

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            symbol: Symbol (required for Binance)

        Returns:
            ExecReport with current status, or None if not found
        """
        if not symbol:
            logger.error("Symbol required for order status")
            return None

        try:
            params: Dict[str, Any] = {"symbol": symbol}

            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                return None

            response = self._request("GET", "/fapi/v1/order", params)

            if isinstance(response, dict):
                return self._parse_exec_report(response)

            return None

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        try:
            params: Dict[str, Any] = {}
            if symbol:
                params["symbol"] = symbol

            response = self._request("GET", "/fapi/v1/openOrders", params)

            orders = []
            for item in response if isinstance(response, list) else []:
                order = self._parse_order(item)
                if order:
                    orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Position]:
        """
        Get current positions.

        Args:
            symbols: Filter by symbols (optional, None = all)

        Returns:
            Dict mapping symbol to Position
        """
        try:
            response = self._request("GET", "/fapi/v2/positionRisk", {})

            positions = {}
            for item in response if isinstance(response, list) else []:
                if not isinstance(item, dict):
                    continue

                symbol = str(item.get("symbol", ""))
                if symbols and symbol not in symbols:
                    continue

                qty = Decimal(str(item.get("positionAmt", "0")))
                if qty == 0:
                    continue

                positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=Decimal(str(item.get("entryPrice", "0"))),
                    realized_pnl=Decimal("0"),  # Not provided by this endpoint
                    fee_paid=Decimal("0"),
                    ts=int(item.get("updateTime", int(time.time() * 1000))),
                    meta={
                        "unrealized_pnl": str(item.get("unRealizedProfit", "0")),
                        "leverage": item.get("leverage"),
                        "margin_type": item.get("marginType"),
                        "liquidation_price": item.get("liquidationPrice"),
                        "mark_price": item.get("markPrice"),
                    },
                )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        try:
            response = self._request("GET", "/fapi/v2/account", {})

            if not isinstance(response, dict):
                return AccountInfo(vendor=self._vendor)

            # Extract USDT balance
            assets = response.get("assets", [])
            usdt_balance = Decimal("0")
            for asset in assets:
                if isinstance(asset, dict) and asset.get("asset") == "USDT":
                    usdt_balance = Decimal(str(asset.get("walletBalance", "0")))
                    break

            return AccountInfo(
                vendor=self._vendor,
                account_id=str(response.get("accountId", "")),
                account_type="futures",
                vip_tier=int(response.get("feeTier", 0)),
                buying_power=Decimal(str(response.get("availableBalance", "0"))),
                cash_balance=usdt_balance,
                margin_enabled=True,
                raw_data=response,
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(vendor=self._vendor)

    # ========================================================================
    # Futures-specific methods
    # ========================================================================

    def submit_futures_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        *,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        position_side: str = "BOTH",
        working_type: str = "CONTRACT_PRICE",
        client_order_id: Optional[str] = None,
        leverage: Optional[int] = None,
    ) -> FuturesOrderResult:
        """
        Submit a futures order with full control over parameters.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            order_type: Order type (MARKET, LIMIT, STOP, TAKE_PROFIT, etc.)
            quantity: Order quantity
            price: Limit price
            stop_price: Stop trigger price
            time_in_force: GTC, IOC, FOK, GTX
            reduce_only: Only reduce position
            position_side: BOTH, LONG, SHORT (for hedge mode)
            working_type: MARK_PRICE or CONTRACT_PRICE
            client_order_id: Client order ID
            leverage: Set leverage before order (optional)

        Returns:
            FuturesOrderResult with execution details
        """
        try:
            # Set leverage if specified
            if leverage is not None:
                self.set_leverage(symbol, leverage)

            params: Dict[str, Any] = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": str(quantity),
            }

            if client_order_id:
                params["newClientOrderId"] = client_order_id
            else:
                params["newClientOrderId"] = str(uuid.uuid4())[:32]

            if order_type.upper() in ("LIMIT", "STOP", "TAKE_PROFIT"):
                if price:
                    params["price"] = str(price)
                params["timeInForce"] = time_in_force.upper()

            if order_type.upper() in (
                "STOP",
                "STOP_MARKET",
                "TAKE_PROFIT",
                "TAKE_PROFIT_MARKET",
                "TRAILING_STOP_MARKET",
            ):
                if stop_price:
                    params["stopPrice"] = str(stop_price)
                params["workingType"] = working_type.upper()

            if reduce_only:
                params["reduceOnly"] = "true"

            if self._hedge_mode or position_side.upper() != "BOTH":
                params["positionSide"] = position_side.upper()

            response = self._request("POST", "/fapi/v1/order", params)

            return self._parse_futures_order_response(response)

        except Exception as e:
            return FuturesOrderResult(
                success=False,
                error_code="SUBMISSION_FAILED",
                error_message=str(e),
            )

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage value (1-125)

        Returns:
            True if successful
        """
        try:
            params = {
                "symbol": symbol,
                "leverage": min(max(leverage, 1), 125),
            }

            response = self._request("POST", "/fapi/v1/leverage", params)

            if isinstance(response, dict):
                return response.get("leverage") == leverage

            return False

        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a symbol.

        Args:
            symbol: Trading symbol
            margin_type: "CROSSED" or "ISOLATED"

        Returns:
            True if successful
        """
        try:
            params = {
                "symbol": symbol,
                "marginType": margin_type.upper(),
            }

            self._request("POST", "/fapi/v1/marginType", params)
            return True

        except Exception as e:
            # May fail if already set to this type
            if "No need to change margin type" in str(e):
                return True
            logger.error(f"Failed to set margin type: {e}")
            return False

    def set_position_mode(self, hedge_mode: bool) -> bool:
        """
        Set position mode (one-way or hedge).

        Args:
            hedge_mode: True for hedge mode (separate long/short)

        Returns:
            True if successful
        """
        try:
            params = {
                "dualSidePosition": "true" if hedge_mode else "false",
            }

            self._request("POST", "/fapi/v1/positionSide/dual", params)
            self._hedge_mode = hedge_mode
            return True

        except Exception as e:
            if "No need to change position side" in str(e):
                self._hedge_mode = hedge_mode
                return True
            logger.error(f"Failed to set position mode: {e}")
            return False

    def get_futures_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, FuturesPosition]:
        """
        Get current futures positions with full details.

        Args:
            symbols: Filter by symbols

        Returns:
            Dict mapping symbol to FuturesPosition
        """
        try:
            response = self._request("GET", "/fapi/v2/positionRisk", {})

            positions = {}
            for item in response if isinstance(response, list) else []:
                if not isinstance(item, dict):
                    continue

                symbol = str(item.get("symbol", ""))
                if symbols and symbol not in symbols:
                    continue

                qty = Decimal(str(item.get("positionAmt", "0")))
                if qty == 0:
                    continue

                # Determine position side
                position_side_str = str(item.get("positionSide", "BOTH")).upper()
                if position_side_str == "LONG":
                    side = PositionSide.LONG
                elif position_side_str == "SHORT":
                    side = PositionSide.SHORT
                else:
                    side = PositionSide.BOTH

                # Determine margin mode
                margin_type_str = str(item.get("marginType", "cross")).lower()
                margin_mode = (
                    MarginMode.ISOLATED if margin_type_str == "isolated" else MarginMode.CROSS
                )

                positions[symbol] = FuturesPosition(
                    symbol=symbol,
                    side=side,
                    entry_price=Decimal(str(item.get("entryPrice", "0"))),
                    qty=qty,
                    leverage=int(item.get("leverage", 1)),
                    margin_mode=margin_mode,
                    unrealized_pnl=Decimal(str(item.get("unRealizedProfit", "0"))),
                    realized_pnl=Decimal("0"),
                    liquidation_price=Decimal(str(item.get("liquidationPrice", "0"))),
                    mark_price=Decimal(str(item.get("markPrice", "0"))),
                    margin=Decimal(str(item.get("isolatedMargin", "0"))),
                    maint_margin=Decimal(str(item.get("maintMargin", "0"))),
                    timestamp_ms=int(item.get("updateTime", int(time.time() * 1000))),
                    position_value=Decimal(str(item.get("notional", "0"))),
                )

            return positions

        except Exception as e:
            logger.error(f"Failed to get futures positions: {e}")
            return {}

    def get_futures_account_state(self) -> Optional[FuturesAccountState]:
        """
        Get comprehensive futures account state.

        Returns:
            FuturesAccountState with balances and positions
        """
        try:
            response = self._request("GET", "/fapi/v2/account", {})

            if not isinstance(response, dict):
                return None

            # Get positions
            positions = {}
            for pos_data in response.get("positions", []):
                if not isinstance(pos_data, dict):
                    continue

                qty = Decimal(str(pos_data.get("positionAmt", "0")))
                if qty == 0:
                    continue

                symbol = str(pos_data.get("symbol", ""))

                position_side_str = str(pos_data.get("positionSide", "BOTH")).upper()
                if position_side_str == "LONG":
                    side = PositionSide.LONG
                elif position_side_str == "SHORT":
                    side = PositionSide.SHORT
                else:
                    side = PositionSide.BOTH

                positions[symbol] = FuturesPosition(
                    symbol=symbol,
                    side=side,
                    entry_price=Decimal(str(pos_data.get("entryPrice", "0"))),
                    qty=qty,
                    leverage=int(pos_data.get("leverage", 1)),
                    margin_mode=MarginMode.CROSS,
                    unrealized_pnl=Decimal(str(pos_data.get("unrealizedProfit", "0"))),
                    timestamp_ms=int(time.time() * 1000),
                )

            return FuturesAccountState(
                timestamp_ms=int(response.get("updateTime", int(time.time() * 1000))),
                total_wallet_balance=Decimal(str(response.get("totalWalletBalance", "0"))),
                total_margin_balance=Decimal(str(response.get("totalMarginBalance", "0"))),
                total_unrealized_pnl=Decimal(str(response.get("totalUnrealizedProfit", "0"))),
                available_balance=Decimal(str(response.get("availableBalance", "0"))),
                total_initial_margin=Decimal(str(response.get("totalInitialMargin", "0"))),
                total_maint_margin=Decimal(str(response.get("totalMaintMargin", "0"))),
                total_position_initial_margin=Decimal(
                    str(response.get("totalPositionInitialMargin", "0"))
                ),
                total_open_order_initial_margin=Decimal(
                    str(response.get("totalOpenOrderInitialMargin", "0"))
                ),
                max_withdraw_amount=Decimal(str(response.get("maxWithdrawAmount", "0"))),
                positions=positions,
                asset="USDT",
            )

        except Exception as e:
            logger.error(f"Failed to get futures account state: {e}")
            return None

    def close_position(
        self,
        symbol: str,
        position_side: str = "BOTH",
    ) -> FuturesOrderResult:
        """
        Close entire position for a symbol.

        Args:
            symbol: Trading symbol
            position_side: BOTH, LONG, or SHORT

        Returns:
            FuturesOrderResult
        """
        positions = self.get_futures_positions([symbol])
        position = positions.get(symbol)

        if not position or position.qty == 0:
            return FuturesOrderResult(
                success=True,
                status="NO_POSITION",
            )

        # Determine closing side
        if position.qty > 0:
            close_side = "SELL"
        else:
            close_side = "BUY"

        return self.submit_futures_order(
            symbol=symbol,
            side=close_side,
            order_type="MARKET",
            quantity=abs(position.qty),
            reduce_only=True,
            position_side=position_side,
        )

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Filter by symbol (required for Binance)

        Returns:
            Number of orders cancelled
        """
        if symbol:
            try:
                self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})
                return -1  # Unknown count for batch cancel

            except Exception as e:
                logger.error(f"Failed to cancel all orders: {e}")
                return 0

        # Cancel for all symbols with open orders
        orders = self.get_open_orders()
        symbols = set(o.symbol for o in orders)
        cancelled = 0

        for sym in symbols:
            try:
                self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": sym})
                cancelled += sum(1 for o in orders if o.symbol == sym)
            except Exception:
                pass

        return cancelled

    # ========================================================================
    # Parse methods
    # ========================================================================

    def _convert_order_type(self, order_type: FuturesOrderType) -> str:
        """Convert core_futures.OrderType to Binance API string."""
        mapping = {
            FuturesOrderType.MARKET: "MARKET",
            FuturesOrderType.LIMIT: "LIMIT",
            FuturesOrderType.STOP: "STOP",
            FuturesOrderType.STOP_MARKET: "STOP_MARKET",
            FuturesOrderType.TAKE_PROFIT: "TAKE_PROFIT",
            FuturesOrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
            FuturesOrderType.TRAILING_STOP_MARKET: "TRAILING_STOP_MARKET",
        }
        return mapping.get(order_type, "MARKET")

    def _parse_order_response(self, response: Dict[str, Any]) -> OrderResult:
        """Parse order submission response into OrderResult."""
        if not isinstance(response, dict):
            return OrderResult(success=False, error_message="Invalid response")

        order_id = str(response.get("orderId", ""))
        client_order_id = str(response.get("clientOrderId", ""))
        status = str(response.get("status", ""))

        # Check for error
        if response.get("code"):
            return OrderResult(
                success=False,
                error_code=str(response.get("code")),
                error_message=str(response.get("msg", "")),
                raw_response=response,
            )

        return OrderResult(
            success=True,
            order_id=order_id,
            client_order_id=client_order_id,
            status=status,
            filled_qty=Decimal(str(response.get("executedQty", "0"))),
            filled_price=(
                Decimal(str(response.get("avgPrice", "0"))) if response.get("avgPrice") else None
            ),
            raw_response=response,
        )

    def _parse_futures_order_response(self, response: Dict[str, Any]) -> FuturesOrderResult:
        """Parse futures order response into FuturesOrderResult."""
        if not isinstance(response, dict):
            return FuturesOrderResult(success=False, error_message="Invalid response")

        # Check for error
        if response.get("code"):
            return FuturesOrderResult(
                success=False,
                error_code=str(response.get("code")),
                error_message=str(response.get("msg", "")),
                raw_response=response,
            )

        order_id = str(response.get("orderId", ""))
        client_order_id = str(response.get("clientOrderId", ""))
        status = str(response.get("status", ""))
        filled_qty = Decimal(str(response.get("executedQty", "0")))
        avg_price = (
            Decimal(str(response.get("avgPrice", "0"))) if response.get("avgPrice") else None
        )

        # Create FuturesFill if filled
        futures_fill = None
        if filled_qty > 0 and avg_price:
            futures_fill = FuturesFill(
                order_id=order_id,
                client_order_id=client_order_id if client_order_id else None,
                symbol=str(response.get("symbol", "")),
                side=FuturesOrderSide(str(response.get("side", "BUY")).upper()),
                filled_qty=filled_qty,
                avg_price=avg_price,
                commission=Decimal(str(response.get("commission", "0"))),
                commission_asset=str(response.get("commissionAsset", "USDT")),
                realized_pnl=Decimal(str(response.get("realizedPnl", "0"))),
                timestamp_ms=int(response.get("updateTime", int(time.time() * 1000))),
                is_maker=str(response.get("type", "")).upper() == "LIMIT",
                liquidity="MAKER" if str(response.get("type", "")).upper() == "LIMIT" else "TAKER",
            )

        return FuturesOrderResult(
            success=True,
            order_id=order_id,
            client_order_id=client_order_id,
            status=status,
            filled_qty=filled_qty,
            filled_price=avg_price,
            position_side=str(response.get("positionSide")),
            realized_pnl=Decimal(str(response.get("realizedPnl", "0"))),
            commission_asset=str(response.get("commissionAsset", "USDT")),
            futures_fill=futures_fill,
            raw_response=response,
        )

    def _parse_exec_report(self, response: Dict[str, Any]) -> Optional[ExecReport]:
        """Parse order status response into ExecReport."""
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

        side_str = str(response.get("side", "")).upper()
        side = Side.BUY if side_str == "BUY" else Side.SELL

        return ExecReport(
            ts=int(response.get("updateTime", int(time.time() * 1000))),
            run_id="binance_futures",
            symbol=str(response.get("symbol", "")),
            side=side,
            order_type=OrderType.MARKET if response.get("type") == "MARKET" else OrderType.LIMIT,
            price=Decimal(str(response.get("avgPrice", response.get("price", "0")))),
            quantity=Decimal(str(response.get("executedQty", "0"))),
            fee=Decimal("0"),  # Not provided in order status
            fee_asset="USDT",
            exec_status=exec_status,
            client_order_id=str(response.get("clientOrderId")),
            order_id=str(response.get("orderId")),
        )

    def _parse_order(self, response: Dict[str, Any]) -> Optional[Order]:
        """Parse order data into Order."""
        if not isinstance(response, dict):
            return None

        side_str = str(response.get("side", "")).upper()
        side = Side.BUY if side_str == "BUY" else Side.SELL

        type_str = str(response.get("type", "")).upper()
        if type_str == "MARKET":
            order_type = OrderType.MARKET
        elif type_str == "LIMIT":
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.LIMIT

        return Order(
            ts=int(response.get("time", int(time.time() * 1000))),
            symbol=str(response.get("symbol", "")),
            side=side,
            order_type=order_type,
            quantity=Decimal(str(response.get("origQty", "0"))),
            price=Decimal(str(response.get("price", "0"))) if response.get("price") else None,
            time_in_force=TimeInForce(response.get("timeInForce", "GTC")),
            client_order_id=str(response.get("clientOrderId", "")),
            reduce_only=response.get("reduceOnly", False),
            meta={
                "order_id": response.get("orderId"),
                "status": response.get("status"),
                "position_side": response.get("positionSide"),
            },
        )

    @property
    def market_type(self) -> MarketType:
        """Return the market type this adapter serves."""
        return MarketType.CRYPTO_FUTURES
