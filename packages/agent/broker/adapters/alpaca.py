# -*- coding: utf-8 -*-
"""
Alpaca BrokerConnector Adapter.

Wraps the Alpaca API for use with CCEA Agent.

Design Doc Section 4.2: Broker Connectors (only in Agent)

AGENT ZONE ONLY - Never import in Cloud zone.

References:
- Alpaca Trading API: https://docs.alpaca.markets/docs/trading-api
- Alpaca Python SDK: https://github.com/alpacahq/alpaca-py
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

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


class AlpacaConnector(BaseBrokerConnector):
    """
    Alpaca broker connector implementation.

    Supports:
    - US Equities (stocks)
    - Paper and live trading
    - Market, limit, stop, stop-limit orders

    Usage:
        creds = BrokerCredentials(
            api_key="your_key",
            api_secret="your_secret",
        )
        connector = AlpacaConnector(creds, sandbox=True)
        connector.connect()

        result = connector.submit_order(OrderRequest(...))
    """

    # API endpoints
    LIVE_URL = "https://api.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        credentials: BrokerCredentials,
        *,
        sandbox: bool = True,
        timeout_seconds: int = 30,
    ):
        """
        Initialize Alpaca connector.

        Args:
            credentials: API credentials from Local Vault
            sandbox: Use paper trading (default: True for safety)
            timeout_seconds: Request timeout
        """
        super().__init__(credentials, sandbox=sandbox, timeout_seconds=timeout_seconds)

        self._base_url = self.PAPER_URL if sandbox else self.LIVE_URL
        self._client: Optional[Any] = None
        self._trading_client: Optional[Any] = None

    @property
    def broker_name(self) -> str:
        """Return broker identifier."""
        return "alpaca"

    def connect(self) -> bool:
        """
        Establish connection to Alpaca.

        Returns:
            True if connection successful
        """
        try:
            # Try alpaca-py SDK first
            try:
                from alpaca.trading.client import TradingClient

                self._trading_client = TradingClient(
                    api_key=self._credentials.api_key,
                    secret_key=self._credentials.api_secret,
                    paper=self._sandbox,
                )
                # Test connection
                self._trading_client.get_account()
                self._connected = True
                self._last_heartbeat = datetime.utcnow()
                logger.info(f"Connected to Alpaca ({'paper' if self._sandbox else 'live'})")
                return True

            except ImportError:
                # Fallback to requests-based client
                import requests

                headers = {
                    "APCA-API-KEY-ID": self._credentials.api_key,
                    "APCA-API-SECRET-KEY": self._credentials.api_secret,
                }
                response = requests.get(
                    f"{self._base_url}/v2/account",
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                self._client = {"headers": headers, "base_url": self._base_url}
                self._connected = True
                self._last_heartbeat = datetime.utcnow()
                logger.info(
                    f"Connected to Alpaca via REST ({'paper' if self._sandbox else 'live'})"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._trading_client = None
        self._client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit order to Alpaca.

        Args:
            request: Order request

        Returns:
            Order result
        """
        if not self._connected:
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error_message="Not connected to Alpaca",
            )

        try:
            if self._trading_client:
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
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import (
            OrderSide as AlpacaOrderSide,
            TimeInForce as AlpacaTimeInForce,
        )

        # Map side
        side = AlpacaOrderSide.BUY if request.side == OrderSide.BUY else AlpacaOrderSide.SELL

        # Map time in force
        tif_map = {
            TimeInForce.GTC: AlpacaTimeInForce.GTC,
            TimeInForce.DAY: AlpacaTimeInForce.DAY,
            TimeInForce.IOC: AlpacaTimeInForce.IOC,
            TimeInForce.FOK: AlpacaTimeInForce.FOK,
            TimeInForce.OPG: AlpacaTimeInForce.OPG,
            TimeInForce.CLS: AlpacaTimeInForce.CLS,
        }
        tif = tif_map.get(request.time_in_force, AlpacaTimeInForce.GTC)

        # Build order request based on type
        if request.order_type == OrderType.MARKET:
            order_req = MarketOrderRequest(
                symbol=request.symbol,
                qty=float(request.quantity),
                side=side,
                time_in_force=tif,
                client_order_id=request.client_order_id,
            )
        elif request.order_type == OrderType.LIMIT:
            order_req = LimitOrderRequest(
                symbol=request.symbol,
                qty=float(request.quantity),
                side=side,
                time_in_force=tif,
                limit_price=float(request.limit_price) if request.limit_price else None,
                client_order_id=request.client_order_id,
            )
        elif request.order_type == OrderType.STOP:
            order_req = StopOrderRequest(
                symbol=request.symbol,
                qty=float(request.quantity),
                side=side,
                time_in_force=tif,
                stop_price=float(request.stop_price) if request.stop_price else None,
                client_order_id=request.client_order_id,
            )
        elif request.order_type == OrderType.STOP_LIMIT:
            order_req = StopLimitOrderRequest(
                symbol=request.symbol,
                qty=float(request.quantity),
                side=side,
                time_in_force=tif,
                limit_price=float(request.limit_price) if request.limit_price else None,
                stop_price=float(request.stop_price) if request.stop_price else None,
                client_order_id=request.client_order_id,
            )
        else:
            return OrderResult(
                success=False,
                client_order_id=request.client_order_id,
                error_message=f"Unsupported order type: {request.order_type}",
            )

        # Submit
        order = self._trading_client.submit_order(order_req)

        return OrderResult(
            success=True,
            client_order_id=request.client_order_id,
            broker_order_id=str(order.id),
            status=self._map_order_status(order.status.value),
            filled_quantity=Decimal(str(order.filled_qty or 0)),
            avg_fill_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
            timestamp=datetime.utcnow(),
        )

    def _submit_order_rest(self, request: OrderRequest) -> OrderResult:
        """Submit order via REST API."""
        import requests

        # Build order payload
        payload = {
            "symbol": request.symbol,
            "qty": str(request.quantity),
            "side": request.side.value,
            "type": request.order_type.value,
            "time_in_force": request.time_in_force.value,
            "client_order_id": request.client_order_id,
        }

        if request.limit_price:
            payload["limit_price"] = str(request.limit_price)
        if request.stop_price:
            payload["stop_price"] = str(request.stop_price)

        response = requests.post(
            f"{self._client['base_url']}/v2/orders",
            headers=self._client["headers"],
            json=payload,
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
            broker_order_id=data.get("id"),
            status=self._map_order_status(data.get("status", "new")),
            filled_quantity=Decimal(data.get("filled_qty", "0")),
            avg_fill_price=(
                Decimal(data["filled_avg_price"]) if data.get("filled_avg_price") else None
            ),
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
            if self._trading_client:
                if broker_order_id:
                    self._trading_client.cancel_order_by_id(broker_order_id)
                else:
                    # Get order by client_order_id first
                    order = self._trading_client.get_order_by_client_id(client_order_id)
                    self._trading_client.cancel_order_by_id(str(order.id))

                return CancelResult(
                    success=True,
                    client_order_id=client_order_id or "",
                    broker_order_id=broker_order_id,
                    cancelled_at=datetime.utcnow(),
                )
            else:
                import requests

                order_id = broker_order_id or client_order_id
                response = requests.delete(
                    f"{self._client['base_url']}/v2/orders/{order_id}",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return CancelResult(
                        success=False,
                        client_order_id=client_order_id or "",
                        error_message=response.text,
                    )

                return CancelResult(
                    success=True,
                    client_order_id=client_order_id or "",
                    broker_order_id=broker_order_id,
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

        Design Doc Section 9.4: Kill Switch action.
        """
        if not self._connected:
            return BulkCancelResult(
                total_requested=0,
                total_cancelled=0,
                total_failed=0,
                errors=["Not connected"],
            )

        try:
            if self._trading_client:
                statuses = self._trading_client.cancel_orders()
                cancelled = sum(1 for s in statuses if s.status == 200)
                failed = len(statuses) - cancelled

                return BulkCancelResult(
                    total_requested=len(statuses),
                    total_cancelled=cancelled,
                    total_failed=failed,
                )
            else:
                import requests

                response = requests.delete(
                    f"{self._client['base_url']}/v2/orders",
                    headers=self._client["headers"],
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
        """Get order information."""
        if not self._connected:
            return None

        try:
            if self._trading_client:
                if broker_order_id:
                    order = self._trading_client.get_order_by_id(broker_order_id)
                else:
                    order = self._trading_client.get_order_by_client_id(client_order_id)

                return self._map_order_info(order)
            else:
                import requests

                order_id = broker_order_id or client_order_id
                response = requests.get(
                    f"{self._client['base_url']}/v2/orders/{order_id}",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return None

                return self._map_order_info_dict(response.json())

        except Exception:
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get all open orders."""
        if not self._connected:
            return []

        try:
            if self._trading_client:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus

                request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                orders = self._trading_client.get_orders(request)

                result = [self._map_order_info(o) for o in orders]
                if symbol:
                    result = [o for o in result if o.symbol == symbol]
                return result
            else:
                import requests

                params = {"status": "open"}
                if symbol:
                    params["symbols"] = symbol

                response = requests.get(
                    f"{self._client['base_url']}/v2/orders",
                    headers=self._client["headers"],
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
            if self._trading_client:
                positions = self._trading_client.get_all_positions()
                return [self._map_position(p) for p in positions]
            else:
                import requests

                response = requests.get(
                    f"{self._client['base_url']}/v2/positions",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return []

                return [self._map_position_dict(p) for p in response.json()]

        except Exception:
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        if not self._connected:
            return None

        try:
            if self._trading_client:
                position = self._trading_client.get_open_position(symbol)
                return self._map_position(position)
            else:
                import requests

                response = requests.get(
                    f"{self._client['base_url']}/v2/positions/{symbol}",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return None

                return self._map_position_dict(response.json())

        except Exception:
            return None

    def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> OrderResult:
        """Close position for symbol."""
        if not self._connected:
            return OrderResult(
                success=False,
                client_order_id="",
                error_message="Not connected",
            )

        try:
            if self._trading_client:
                from alpaca.trading.requests import ClosePositionRequest

                request = ClosePositionRequest()
                if quantity:
                    request = ClosePositionRequest(qty=str(quantity))

                order = self._trading_client.close_position(symbol, request)

                return OrderResult(
                    success=True,
                    client_order_id=str(order.client_order_id) if order.client_order_id else "",
                    broker_order_id=str(order.id),
                    status=self._map_order_status(order.status.value),
                )
            else:
                import requests

                params = {}
                if quantity:
                    params["qty"] = str(quantity)

                response = requests.delete(
                    f"{self._client['base_url']}/v2/positions/{symbol}",
                    headers=self._client["headers"],
                    params=params,
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return OrderResult(
                        success=False,
                        client_order_id="",
                        error_message=response.text,
                    )

                data = response.json()
                return OrderResult(
                    success=True,
                    client_order_id=data.get("client_order_id", ""),
                    broker_order_id=data.get("id"),
                )

        except Exception as e:
            return OrderResult(
                success=False,
                client_order_id="",
                error_message=str(e),
            )

    def close_all_positions(self) -> List[OrderResult]:
        """Close all positions (flatten)."""
        if not self._connected:
            return [OrderResult(success=False, client_order_id="", error_message="Not connected")]

        try:
            if self._trading_client:
                statuses = self._trading_client.close_all_positions(cancel_orders=True)
                results = []
                for status in statuses:
                    results.append(
                        OrderResult(
                            success=status.status == 200,
                            client_order_id="",
                            broker_order_id=str(status.body.get("id")) if status.body else None,
                        )
                    )
                return results
            else:
                import requests

                response = requests.delete(
                    f"{self._client['base_url']}/v2/positions",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return [
                        OrderResult(success=False, client_order_id="", error_message=response.text)
                    ]

                data = response.json()
                return [
                    OrderResult(
                        success=True,
                        client_order_id="",
                        broker_order_id=item.get("id"),
                    )
                    for item in data
                ]

        except Exception as e:
            return [OrderResult(success=False, client_order_id="", error_message=str(e))]

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
            if self._trading_client:
                account = self._trading_client.get_account()
                return AccountInfo(
                    account_id=str(account.id),
                    equity=Decimal(str(account.equity)),
                    cash=Decimal(str(account.cash)),
                    buying_power=Decimal(str(account.buying_power)),
                    unrealized_pnl=Decimal(str(account.unrealized_pl or 0)),
                    status=account.status.value if account.status else "active",
                    currency="USD",
                    updated_at=datetime.utcnow(),
                )
            else:
                import requests

                response = requests.get(
                    f"{self._client['base_url']}/v2/account",
                    headers=self._client["headers"],
                    timeout=self._timeout,
                )

                if response.status_code >= 400:
                    return AccountInfo(
                        account_id="",
                        equity=Decimal("0"),
                        cash=Decimal("0"),
                        buying_power=Decimal("0"),
                        status="error",
                    )

                data = response.json()
                return AccountInfo(
                    account_id=data.get("id", ""),
                    equity=Decimal(data.get("equity", "0")),
                    cash=Decimal(data.get("cash", "0")),
                    buying_power=Decimal(data.get("buying_power", "0")),
                    unrealized_pnl=Decimal(data.get("unrealized_pl", "0")),
                    status=data.get("status", "active"),
                    currency="USD",
                    updated_at=datetime.utcnow(),
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
            if self._trading_client:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestQuoteRequest

                data_client = StockHistoricalDataClient()
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = data_client.get_stock_latest_quote(request)
                if symbol in quotes:
                    return Decimal(str(quotes[symbol].ask_price))
            return None
        except Exception:
            return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Alpaca order status to OrderStatus."""
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
            "pending_new": OrderStatus.PENDING,
            "pending_cancel": OrderStatus.PENDING,
        }
        return status_map.get(status.lower(), OrderStatus.PENDING)

    def _map_order_info(self, order: Any) -> OrderInfo:
        """Map SDK order to OrderInfo."""
        return OrderInfo(
            client_order_id=str(order.client_order_id) if order.client_order_id else "",
            broker_order_id=str(order.id),
            symbol=order.symbol,
            side=OrderSide.BUY if order.side.value == "buy" else OrderSide.SELL,
            order_type=OrderType(order.type.value) if order.type else OrderType.MARKET,
            quantity=Decimal(str(order.qty)),
            filled_quantity=Decimal(str(order.filled_qty or 0)),
            limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
            stop_price=Decimal(str(order.stop_price)) if order.stop_price else None,
            avg_fill_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
            status=self._map_order_status(order.status.value),
            time_in_force=(
                TimeInForce(order.time_in_force.value) if order.time_in_force else TimeInForce.GTC
            ),
            created_at=order.created_at,
            updated_at=order.updated_at,
            filled_at=order.filled_at,
        )

    def _map_order_info_dict(self, data: Dict[str, Any]) -> OrderInfo:
        """Map dict to OrderInfo."""
        return OrderInfo(
            client_order_id=data.get("client_order_id", ""),
            broker_order_id=data.get("id"),
            symbol=data.get("symbol", ""),
            side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
            order_type=OrderType(data.get("type", "market")),
            quantity=Decimal(data.get("qty", "0")),
            filled_quantity=Decimal(data.get("filled_qty", "0")),
            limit_price=Decimal(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            avg_fill_price=(
                Decimal(data["filled_avg_price"]) if data.get("filled_avg_price") else None
            ),
            status=self._map_order_status(data.get("status", "new")),
            time_in_force=TimeInForce(data.get("time_in_force", "gtc")),
        )

    def _map_position(self, position: Any) -> Position:
        """Map SDK position to Position."""
        qty = Decimal(str(position.qty))
        return Position(
            symbol=position.symbol,
            side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
            quantity=abs(qty),
            avg_entry_price=Decimal(str(position.avg_entry_price)),
            current_price=Decimal(str(position.current_price)) if position.current_price else None,
            unrealized_pnl=Decimal(str(position.unrealized_pl)) if position.unrealized_pl else None,
            market_value=Decimal(str(position.market_value)) if position.market_value else None,
            cost_basis=Decimal(str(position.cost_basis)) if position.cost_basis else None,
            updated_at=datetime.utcnow(),
        )

    def _map_position_dict(self, data: Dict[str, Any]) -> Position:
        """Map dict to Position."""
        qty = Decimal(data.get("qty", "0"))
        return Position(
            symbol=data.get("symbol", ""),
            side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
            quantity=abs(qty),
            avg_entry_price=Decimal(data.get("avg_entry_price", "0")),
            current_price=Decimal(data["current_price"]) if data.get("current_price") else None,
            unrealized_pnl=Decimal(data["unrealized_pl"]) if data.get("unrealized_pl") else None,
            market_value=Decimal(data["market_value"]) if data.get("market_value") else None,
            cost_basis=Decimal(data["cost_basis"]) if data.get("cost_basis") else None,
            updated_at=datetime.utcnow(),
        )


# Register with factory
BrokerConnectorFactory.register("alpaca", AlpacaConnector)
