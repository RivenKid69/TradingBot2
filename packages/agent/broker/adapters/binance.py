# -*- coding: utf-8 -*-
"""
Binance BrokerConnector Adapter.

Wraps the Binance API for use with CCEA Agent.

Design Doc Section 4.2: Broker Connectors (only in Agent)

AGENT ZONE ONLY - Never import in Cloud zone.

Supports:
- Spot trading
- Futures trading (USDT-M)
- Testnet for paper trading

References:
- Binance API: https://binance-docs.github.io/apidocs/spot/en/
- python-binance: https://python-binance.readthedocs.io/
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from packages.agent.broker.protocol import (
    BaseBrokerConnector,
    BrokerCredentials,
    OrderRequest,
    OrderResult,
    OrderInfo,
    Position,
    AccountInfo,
    CancelResult,
    BulkCancelResult,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    PositionSide,
    ConnectionStatus,
    BrokerConnectorFactory,
)


logger = logging.getLogger(__name__)


class BinanceConnector(BaseBrokerConnector):
    """
    Binance broker connector implementation.

    Supports:
    - Spot trading
    - USDT-M Futures
    - Testnet (paper trading)

    Usage:
        creds = BrokerCredentials(
            api_key="your_key",
            api_secret="your_secret",
        )
        connector = BinanceConnector(creds, sandbox=True)
        connector.connect()

        result = connector.submit_order(OrderRequest(...))
    """

    # API endpoints
    SPOT_URL = "https://api.binance.com"
    SPOT_TESTNET_URL = "https://testnet.binance.vision"
    FUTURES_URL = "https://fapi.binance.com"
    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        credentials: BrokerCredentials,
        *,
        sandbox: bool = True,
        futures: bool = False,
        timeout_seconds: int = 30,
    ):
        """
        Initialize Binance connector.

        Args:
            credentials: API credentials from Local Vault
            sandbox: Use testnet (default: True for safety)
            futures: Use futures API (default: False = spot)
            timeout_seconds: Request timeout
        """
        super().__init__(credentials, sandbox=sandbox, timeout_seconds=timeout_seconds)

        self._futures = futures

        # Select endpoint
        if futures:
            self._base_url = self.FUTURES_TESTNET_URL if sandbox else self.FUTURES_URL
        else:
            self._base_url = self.SPOT_TESTNET_URL if sandbox else self.SPOT_URL

        self._recv_window = 5000
        self._client: Optional[Any] = None

    @property
    def broker_name(self) -> str:
        """Return broker identifier."""
        return "binance" + ("_futures" if self._futures else "")

    def _sign(self, params: Dict[str, Any]) -> str:
        """Create HMAC SHA256 signature."""
        query = urlencode(params)
        signature = hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Get server timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "X-MBX-APIKEY": self._credentials.api_key,
            "Content-Type": "application/json",
        }

    def connect(self) -> bool:
        """
        Establish connection to Binance.

        Returns:
            True if connection successful
        """
        try:
            # Try python-binance SDK first
            try:
                from binance.client import Client

                self._client = Client(
                    api_key=self._credentials.api_key,
                    api_secret=self._credentials.api_secret,
                    testnet=self._sandbox,
                )
                # Test connection
                self._client.get_account()
                self._connected = True
                self._last_heartbeat = datetime.utcnow()
                logger.info(f"Connected to Binance ({'testnet' if self._sandbox else 'live'})")
                return True

            except ImportError:
                # Fallback to requests
                import requests

                params = {"timestamp": self._get_timestamp(), "recvWindow": self._recv_window}
                params["signature"] = self._sign(params)

                response = requests.get(
                    f"{self._base_url}/api/v3/account",
                    headers=self._headers(),
                    params=params,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                self._connected = True
                self._last_heartbeat = datetime.utcnow()
                logger.info(
                    f"Connected to Binance via REST ({'testnet' if self._sandbox else 'live'})"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from Binance")

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit order to Binance."""
        if not self._connected:
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error_message="Not connected to Binance",
            )

        try:
            if self._client:
                return self._submit_order_sdk(request)
            else:
                return self._submit_order_rest(request)

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error_message=str(e),
            )

    def _submit_order_sdk(self, request: OrderRequest) -> OrderResult:
        """Submit order via SDK."""
        from binance.enums import (
            SIDE_BUY,
            SIDE_SELL,
            ORDER_TYPE_MARKET,
            ORDER_TYPE_LIMIT,
            ORDER_TYPE_STOP_LOSS,
            ORDER_TYPE_STOP_LOSS_LIMIT,
            TIME_IN_FORCE_GTC,
            TIME_IN_FORCE_IOC,
            TIME_IN_FORCE_FOK,
        )

        # Map side
        side = SIDE_BUY if request.side == OrderSide.BUY else SIDE_SELL

        # Map order type
        type_map = {
            OrderType.MARKET: ORDER_TYPE_MARKET,
            OrderType.LIMIT: ORDER_TYPE_LIMIT,
            OrderType.STOP: ORDER_TYPE_STOP_LOSS,
            OrderType.STOP_LIMIT: ORDER_TYPE_STOP_LOSS_LIMIT,
        }
        order_type = type_map.get(request.order_type, ORDER_TYPE_MARKET)

        # Map time in force
        tif_map = {
            TimeInForce.GTC: TIME_IN_FORCE_GTC,
            TimeInForce.IOC: TIME_IN_FORCE_IOC,
            TimeInForce.FOK: TIME_IN_FORCE_FOK,
        }
        tif = tif_map.get(request.time_in_force, TIME_IN_FORCE_GTC)

        # Build params
        params = {
            "symbol": request.symbol,
            "side": side,
            "type": order_type,
            "quantity": str(request.quantity),
            "newClientOrderId": request.client_order_id,
        }

        if request.order_type == OrderType.LIMIT:
            params["price"] = str(request.limit_price)
            params["timeInForce"] = tif

        if request.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            params["stopPrice"] = str(request.stop_price)
            if request.order_type == OrderType.STOP_LIMIT:
                params["price"] = str(request.limit_price)
                params["timeInForce"] = tif

        # Submit
        order = self._client.create_order(**params)

        return OrderResult(
            success=True,
            client_order_id=request.client_order_id,
            broker_order_id=str(order.get("orderId")),
            status=self._map_order_status(order.get("status", "NEW")),
            filled_quantity=Decimal(order.get("executedQty", "0")),
            avg_fill_price=Decimal(order.get("avgPrice", "0")) if order.get("avgPrice") else None,
            timestamp=datetime.utcnow(),
            raw_response=order,
        )

    def _submit_order_rest(self, request: OrderRequest) -> OrderResult:
        """Submit order via REST API."""
        import requests

        # Build params
        params = {
            "symbol": request.symbol,
            "side": request.side.value.upper(),
            "type": request.order_type.value.upper(),
            "quantity": str(request.quantity),
            "newClientOrderId": request.client_order_id,
            "timestamp": self._get_timestamp(),
            "recvWindow": self._recv_window,
        }

        if request.order_type == OrderType.LIMIT:
            params["price"] = str(request.limit_price)
            params["timeInForce"] = request.time_in_force.value.upper()

        if request.stop_price:
            params["stopPrice"] = str(request.stop_price)

        params["signature"] = self._sign(params)

        endpoint = "/fapi/v1/order" if self._futures else "/api/v3/order"
        response = requests.post(
            f"{self._base_url}{endpoint}",
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )

        if response.status_code >= 400:
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error_message=response.text,
                error_code=str(response.status_code),
            )

        data = response.json()
        return OrderResult(
            success=True,
            client_order_id=request.client_order_id,
            broker_order_id=str(data.get("orderId")),
            status=self._map_order_status(data.get("status", "NEW")),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            timestamp=datetime.utcnow(),
            raw_response=data,
        )

    def cancel_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> CancelResult:
        """Cancel order."""
        if not self._connected:
            return CancelResult(
                success=False,
                client_order_id=client_order_id or "",
                error_message="Not connected",
            )

        try:
            if self._client:
                if broker_order_id:
                    # Need symbol - this is a limitation
                    return CancelResult(
                        success=False,
                        client_order_id=client_order_id or "",
                        error_message="Binance requires symbol for cancel by order ID",
                    )

                # Cancel by client order ID requires symbol too
                return CancelResult(
                    success=False,
                    client_order_id=client_order_id or "",
                    error_message="Binance requires symbol for cancel",
                )
            else:
                return CancelResult(
                    success=False,
                    client_order_id=client_order_id or "",
                    error_message="Not implemented for REST fallback",
                )

        except Exception as e:
            return CancelResult(
                success=False,
                client_order_id=client_order_id or "",
                error_message=str(e),
            )

    def cancel_order_with_symbol(
        self,
        symbol: str,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> CancelResult:
        """Cancel order with symbol (Binance-specific)."""
        if not self._connected:
            return CancelResult(
                success=False,
                client_order_id=client_order_id or "",
                error_message="Not connected",
            )

        try:
            if self._client:
                if broker_order_id:
                    result = self._client.cancel_order(symbol=symbol, orderId=int(broker_order_id))
                else:
                    result = self._client.cancel_order(
                        symbol=symbol, origClientOrderId=client_order_id
                    )

                return CancelResult(
                    success=True,
                    client_order_id=result.get("origClientOrderId", client_order_id or ""),
                    broker_order_id=str(result.get("orderId")),
                    cancelled_at=datetime.utcnow(),
                )
            else:
                import requests

                params = {
                    "symbol": symbol,
                    "timestamp": self._get_timestamp(),
                    "recvWindow": self._recv_window,
                }
                if broker_order_id:
                    params["orderId"] = broker_order_id
                else:
                    params["origClientOrderId"] = client_order_id

                params["signature"] = self._sign(params)

                endpoint = "/fapi/v1/order" if self._futures else "/api/v3/order"
                response = requests.delete(
                    f"{self._base_url}{endpoint}",
                    headers=self._headers(),
                    params=params,
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return CancelResult(
                        success=False,
                        client_order_id=client_order_id or "",
                        error_message=response.text,
                    )

                data = response.json()
                return CancelResult(
                    success=True,
                    client_order_id=data.get("origClientOrderId", client_order_id or ""),
                    broker_order_id=str(data.get("orderId")),
                    cancelled_at=datetime.utcnow(),
                )

        except Exception as e:
            return CancelResult(
                success=False,
                client_order_id=client_order_id or "",
                error_message=str(e),
            )

    def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
        """
        Cancel all open orders.

        Note: Binance requires symbol for bulk cancel.
        """
        if not self._connected:
            return BulkCancelResult(
                total_requested=0,
                total_cancelled=0,
                total_failed=0,
                errors=["Not connected"],
            )

        if not symbol:
            # Get all open orders and cancel them
            open_orders = self.get_open_orders()
            results = []
            for order in open_orders:
                result = self.cancel_order_with_symbol(
                    symbol=order.symbol,
                    client_order_id=order.client_order_id,
                    broker_order_id=order.broker_order_id,
                )
                results.append(result)

            cancelled = sum(1 for r in results if r.success)
            return BulkCancelResult(
                total_requested=len(results),
                total_cancelled=cancelled,
                total_failed=len(results) - cancelled,
                results=results,
            )

        try:
            if self._client:
                result = self._client.cancel_open_orders(symbol=symbol)
                return BulkCancelResult(
                    total_requested=len(result),
                    total_cancelled=len(result),
                    total_failed=0,
                )
            else:
                import requests

                params = {
                    "symbol": symbol,
                    "timestamp": self._get_timestamp(),
                    "recvWindow": self._recv_window,
                }
                params["signature"] = self._sign(params)

                endpoint = "/fapi/v1/allOpenOrders" if self._futures else "/api/v3/openOrders"
                response = requests.delete(
                    f"{self._base_url}{endpoint}",
                    headers=self._headers(),
                    params=params,
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return BulkCancelResult(
                        total_requested=0,
                        total_cancelled=0,
                        total_failed=0,
                        errors=[response.text],
                    )

                data = response.json()
                return BulkCancelResult(
                    total_requested=len(data),
                    total_cancelled=len(data),
                    total_failed=0,
                )

        except Exception as e:
            return BulkCancelResult(
                total_requested=0,
                total_cancelled=0,
                total_failed=0,
                errors=[str(e)],
            )

    def get_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> Optional[OrderInfo]:
        """Get order information (requires symbol for Binance)."""
        # Note: Binance API requires symbol for order query
        # This is a limitation - would need to track symbol locally
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get all open orders."""
        if not self._connected:
            return []

        try:
            if self._client:
                if symbol:
                    orders = self._client.get_open_orders(symbol=symbol)
                else:
                    orders = self._client.get_open_orders()
                return [self._map_order_info_dict(o) for o in orders]
            else:
                import requests

                params = {"timestamp": self._get_timestamp(), "recvWindow": self._recv_window}
                if symbol:
                    params["symbol"] = symbol
                params["signature"] = self._sign(params)

                endpoint = "/fapi/v1/openOrders" if self._futures else "/api/v3/openOrders"
                response = requests.get(
                    f"{self._base_url}{endpoint}",
                    headers=self._headers(),
                    params=params,
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return []

                return [self._map_order_info_dict(o) for o in response.json()]

        except Exception:
            return []

    def get_positions(self) -> List[Position]:
        """Get all positions."""
        if not self._connected:
            return []

        try:
            if self._futures:
                return self._get_futures_positions()
            else:
                return self._get_spot_positions()
        except Exception:
            return []

    def _get_spot_positions(self) -> List[Position]:
        """Get spot positions (balances)."""
        if self._client:
            account = self._client.get_account()
            positions = []
            for balance in account.get("balances", []):
                free = Decimal(balance.get("free", "0"))
                locked = Decimal(balance.get("locked", "0"))
                total = free + locked
                if total > 0:
                    positions.append(
                        Position(
                            symbol=balance.get("asset", ""),
                            side=PositionSide.LONG,
                            quantity=total,
                            avg_entry_price=Decimal("0"),  # Not tracked in spot
                        )
                    )
            return positions
        return []

    def _get_futures_positions(self) -> List[Position]:
        """Get futures positions."""
        if self._client:
            positions_data = self._client.futures_position_information()
            positions = []
            for pos in positions_data:
                qty = Decimal(pos.get("positionAmt", "0"))
                if qty != 0:
                    positions.append(
                        Position(
                            symbol=pos.get("symbol", ""),
                            side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                            quantity=abs(qty),
                            avg_entry_price=Decimal(pos.get("entryPrice", "0")),
                            unrealized_pnl=Decimal(pos.get("unRealizedProfit", "0")),
                        )
                    )
            return positions
        return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> OrderResult:
        """Close position for symbol."""
        position = self.get_position(symbol)
        if not position:
            return OrderResult(
                success=False,
                client_order_id="",
                error_message=f"No position for {symbol}",
            )

        # Determine close side
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        close_qty = quantity or position.quantity

        request = OrderRequest(
            client_order_id=f"close_{symbol}_{int(time.time() * 1000)}",
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
        )

        return self.submit_order(request)

    def close_all_positions(self) -> List[OrderResult]:
        """Close all positions."""
        results = []
        for position in self.get_positions():
            result = self.close_position(position.symbol)
            results.append(result)
        return results

    def get_account(self) -> AccountInfo:
        """Get account information."""
        if not self._connected:
            return AccountInfo(
                account_id="",
                equity=Decimal("0"),
                cash=Decimal("0"),
                buying_power=Decimal("0"),
                status="disconnected",
            )

        try:
            if self._client:
                if self._futures:
                    account = self._client.futures_account()
                    return AccountInfo(
                        account_id="binance_futures",
                        equity=Decimal(account.get("totalWalletBalance", "0")),
                        cash=Decimal(account.get("availableBalance", "0")),
                        buying_power=Decimal(account.get("availableBalance", "0")),
                        margin_used=Decimal(account.get("totalMaintMargin", "0")),
                        unrealized_pnl=Decimal(account.get("totalUnrealizedProfit", "0")),
                        currency="USDT",
                        status="active",
                        updated_at=datetime.utcnow(),
                    )
                else:
                    account = self._client.get_account()
                    # Calculate total from balances
                    usdt_balance = Decimal("0")
                    for balance in account.get("balances", []):
                        if balance.get("asset") == "USDT":
                            usdt_balance = Decimal(balance.get("free", "0")) + Decimal(
                                balance.get("locked", "0")
                            )
                            break

                    return AccountInfo(
                        account_id="binance_spot",
                        equity=usdt_balance,
                        cash=usdt_balance,
                        buying_power=usdt_balance,
                        currency="USDT",
                        status="active",
                        updated_at=datetime.utcnow(),
                    )
            return AccountInfo(
                account_id="",
                equity=Decimal("0"),
                cash=Decimal("0"),
                buying_power=Decimal("0"),
                status="error",
            )

        except Exception:
            return AccountInfo(
                account_id="",
                equity=Decimal("0"),
                cash=Decimal("0"),
                buying_power=Decimal("0"),
                status="error",
            )

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """Get last traded price."""
        try:
            if self._client:
                ticker = self._client.get_symbol_ticker(symbol=symbol)
                return Decimal(ticker.get("price", "0"))
            return None
        except Exception:
            return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Binance order status to OrderStatus."""
        status_map = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "PENDING_CANCEL": OrderStatus.PENDING,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_map.get(status.upper(), OrderStatus.PENDING)

    def _map_order_info_dict(self, data: Dict[str, Any]) -> OrderInfo:
        """Map dict to OrderInfo."""
        return OrderInfo(
            client_order_id=data.get("clientOrderId", ""),
            broker_order_id=str(data.get("orderId")),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side", "").upper() == "BUY" else OrderSide.SELL,
            order_type=OrderType(data.get("type", "MARKET").lower()),
            quantity=Decimal(data.get("origQty", "0")),
            filled_quantity=Decimal(data.get("executedQty", "0")),
            limit_price=(
                Decimal(data["price"]) if data.get("price") and data["price"] != "0" else None
            ),
            stop_price=(
                Decimal(data["stopPrice"])
                if data.get("stopPrice") and data["stopPrice"] != "0"
                else None
            ),
            avg_fill_price=(
                Decimal(data["avgPrice"])
                if data.get("avgPrice") and data["avgPrice"] != "0"
                else None
            ),
            status=self._map_order_status(data.get("status", "NEW")),
            time_in_force=TimeInForce.GTC,
        )


# Register with factory
BrokerConnectorFactory.register("binance", BinanceConnector)
BrokerConnectorFactory.register(
    "binance_futures", lambda creds, **kw: BinanceConnector(creds, futures=True, **kw)
)
