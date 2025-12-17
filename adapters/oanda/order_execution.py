# -*- coding: utf-8 -*-
"""
adapters/oanda/order_execution.py
OANDA order execution adapter for forex trading.

Status: Tested and operational (Phase 2 Complete)

OANDA Order Types:
- Market: Immediate execution at current price
- Limit: Execute at specified price or better
- Stop: Trigger market order when price reaches level
- Market If Touched (MIT): Like stop but for limit entries
- Take Profit: Close position at profit target
- Stop Loss: Close position at loss limit
- Trailing Stop: Dynamic stop loss

OANDA v20 API Order Endpoints:
- POST /v3/accounts/{accountID}/orders - Create order
- GET /v3/accounts/{accountID}/orders - List orders
- PUT /v3/accounts/{accountID}/orders/{orderID} - Modify order
- PUT /v3/accounts/{accountID}/orders/{orderID}/cancel - Cancel order

Last-Look Feature:
- Dealer can reject orders within time window
- Typical rejection: 1-5% of aggressive orders
- More common during high volatility/news

Usage:
    adapter = OandaOrderExecutionAdapter(config={
        "api_key": "your_key",
        "account_id": "your_account",
        "practice": True,
    })

    # Submit market order
    order = Order(
        symbol="EUR_USD",
        side=Side.BUY,
        qty=100000,
        order_type="MARKET",
    )
    result = adapter.submit_order(order)

    # Get positions
    positions = adapter.get_positions()

References:
    - OANDA Orders API: https://developer.oanda.com/rest-live-v20/order-ep/
    - OANDA Positions API: https://developer.oanda.com/rest-live-v20/position-ep/
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core_models import Bar, ExecReport, Order, Position, Side, Tick, Liquidity
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import AccountInfo, ExchangeVendor

logger = logging.getLogger(__name__)


# =========================
# OANDA Order Types
# =========================

class OandaOrderType(str, Enum):
    """OANDA order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP_LOSS = "TRAILING_STOP_LOSS"


class OandaTimeInForce(str, Enum):
    """OANDA time-in-force options."""
    GTC = "GTC"        # Good Till Cancelled
    GTD = "GTD"        # Good Till Date
    GFD = "GFD"        # Good For Day
    FOK = "FOK"        # Fill Or Kill
    IOC = "IOC"        # Immediate Or Cancel


class OandaOrderState(str, Enum):
    """OANDA order states."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"  # Last-look rejection


# =========================
# Order Configuration
# =========================

@dataclass
class OandaOrderConfig:
    """
    Extended order configuration for OANDA.

    Includes forex-specific options like take profit,
    stop loss, and trailing stop.
    """
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None  # In pips
    time_in_force: OandaTimeInForce = OandaTimeInForce.GTC
    gtd_time: Optional[str] = None  # ISO 8601 for GTD orders
    client_extensions: Dict[str, str] = field(default_factory=dict)
    position_fill: str = "DEFAULT"  # DEFAULT, OPEN_ONLY, REDUCE_FIRST, REDUCE_ONLY


class OandaOrderExecutionAdapter(OrderExecutionAdapter):
    """
    OANDA order execution adapter for forex.

    Handles order placement, cancellation, and position tracking
    via OANDA v20 REST API.

    Configuration:
        api_key: OANDA API access token (required)
        account_id: Trading account ID (required)
        practice: Use practice/demo environment (default: True)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)

    Features:
    - Market and limit orders
    - Take profit and stop loss attachments
    - Position tracking
    - Account balance queries
    - Order modification and cancellation

    Environment Variables:
        OANDA_API_KEY: API access token
        OANDA_ACCOUNT_ID: Trading account ID
        OANDA_PRACTICE: "true" for demo, "false" for live
    """

    # API endpoints
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize OANDA order execution adapter.

        Args:
            vendor: Exchange vendor (default: OANDA)
            config: Configuration dict
        """
        super().__init__(vendor, config)

        # API configuration
        self._api_key = self._config.get("api_key") or os.getenv("OANDA_API_KEY")
        self._account_id = self._config.get("account_id") or os.getenv("OANDA_ACCOUNT_ID")

        # Environment
        practice_env = os.getenv("OANDA_PRACTICE", "true").lower()
        self._practice = self._config.get("practice", practice_env == "true")

        # Set URL
        if self._practice:
            self._base_url = self._config.get("base_url", self.PRACTICE_URL)
        else:
            self._base_url = self._config.get("base_url", self.LIVE_URL)

        # Request settings
        self._timeout = self._config.get("timeout", 30)
        self._max_retries = self._config.get("max_retries", 3)

        # Session (lazy init)
        self._session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        if not self._api_key:
            raise ValueError(
                "OANDA API key is required. "
                "Set via config['api_key'] or OANDA_API_KEY environment variable."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX",
        }

    def _ensure_session(self):
        """Ensure HTTP session is initialized."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update(self._get_headers())
        return self._session

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to OANDA format."""
        return symbol.upper().replace("/", "_").replace("-", "_")

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to OANDA.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info
        """
        session = self._ensure_session()

        if not self._account_id:
            return OrderResult(
                success=False,
                error_code="NO_ACCOUNT",
                error_message="Account ID is required",
            )

        instrument = self._normalize_symbol(order.symbol)

        # Build order request
        order_data = self._build_order_request(order, instrument)

        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders"

        try:
            response = session.post(url, json=order_data, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return OrderResult(
                success=False,
                error_code="API_ERROR",
                error_message=str(e),
            )

        # Parse response
        return self._parse_order_response(data, order.client_order_id)

    def _build_order_request(
        self,
        order: Order,
        instrument: str,
    ) -> Dict[str, Any]:
        """
        Build OANDA order request body.

        Args:
            order: Order to convert
            instrument: Normalized instrument name

        Returns:
            Order request dict for OANDA API
        """
        # Determine order type
        order_type = order.order_type.upper() if order.order_type else "MARKET"
        oanda_type = OandaOrderType(order_type) if order_type in [e.value for e in OandaOrderType] else OandaOrderType.MARKET

        # Units: positive for buy, negative for sell
        is_buy = order.side == Side.BUY or str(order.side).upper() == "BUY"
        units = abs(float(order.quantity)) if is_buy else -abs(float(order.quantity))

        # Base order structure
        order_request: Dict[str, Any] = {
            "order": {
                "instrument": instrument,
                "units": str(int(units)),
                "type": oanda_type.value,
                "timeInForce": "FOK" if oanda_type == OandaOrderType.MARKET else "GTC",
            }
        }

        # Add price for limit orders
        if oanda_type == OandaOrderType.LIMIT and order.price is not None:
            order_request["order"]["price"] = str(order.price)

        # Add stop price for stop orders
        if oanda_type == OandaOrderType.STOP and order.stop_price is not None:
            order_request["order"]["price"] = str(order.stop_price)
        elif oanda_type == OandaOrderType.STOP and order.price is not None:
            order_request["order"]["price"] = str(order.price)

        # Add client order ID if provided
        if order.client_order_id:
            order_request["order"]["clientExtensions"] = {
                "id": order.client_order_id,
            }

        return order_request

    def _parse_order_response(
        self,
        data: Dict[str, Any],
        client_order_id: Optional[str],
    ) -> OrderResult:
        """
        Parse OANDA order response.

        Args:
            data: API response data
            client_order_id: Client order ID

        Returns:
            OrderResult
        """
        # Check for fill
        fill = data.get("orderFillTransaction")
        if fill:
            return OrderResult(
                success=True,
                order_id=str(fill.get("orderID", "")),
                client_order_id=client_order_id,
                status="FILLED",
                filled_qty=Decimal(str(abs(int(fill.get("units", 0))))),
                filled_price=Decimal(str(fill.get("price", "0"))),
                fee=Decimal(str(abs(float(fill.get("commission", 0))))),
                raw_response=data,
            )

        # Check for order creation
        order_create = data.get("orderCreateTransaction")
        if order_create:
            return OrderResult(
                success=True,
                order_id=str(order_create.get("id", "")),
                client_order_id=client_order_id,
                status="PENDING",
                raw_response=data,
            )

        # Check for rejection
        reject = data.get("orderRejectTransaction")
        if reject:
            return OrderResult(
                success=False,
                error_code=reject.get("rejectReason", "UNKNOWN"),
                error_message=reject.get("rejectReason", "Order rejected"),
                raw_response=data,
            )

        # Check for cancellation (last-look rejection)
        cancel = data.get("orderCancelTransaction")
        if cancel:
            return OrderResult(
                success=False,
                order_id=str(cancel.get("orderID", "")),
                error_code="LAST_LOOK_REJECTED",
                error_message=cancel.get("reason", "Order cancelled"),
                raw_response=data,
            )

        # Unknown response
        return OrderResult(
            success=False,
            error_code="UNKNOWN_RESPONSE",
            error_message="Unexpected API response",
            raw_response=data,
        )

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: OANDA order ID
            client_order_id: Client order ID (uses @client_id syntax)

        Returns:
            True if cancellation successful
        """
        session = self._ensure_session()

        if not order_id and not client_order_id:
            return False

        # Use client order ID with @ syntax if no order_id
        if order_id:
            order_specifier = order_id
        else:
            order_specifier = f"@{client_order_id}"

        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders/{order_specifier}/cancel"

        try:
            response = session.put(url, timeout=self._timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Order cancellation failed: {e}")
            return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: OANDA order ID
            client_order_id: Client order ID

        Returns:
            ExecReport with current status, or None
        """
        session = self._ensure_session()

        if not order_id and not client_order_id:
            return None

        order_specifier = order_id or f"@{client_order_id}"
        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders/{order_specifier}"

        try:
            response = session.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return None

        order_data = data.get("order", {})
        if not order_data:
            return None

        # Parse to ExecReport
        state = order_data.get("state", "PENDING")
        filled_qty = Decimal(str(abs(int(order_data.get("filledUnits", "0")))))

        return ExecReport(
            order_id=order_data.get("id", ""),
            client_order_id=order_data.get("clientExtensions", {}).get("id"),
            symbol=order_data.get("instrument", ""),
            side=Side.BUY if int(order_data.get("units", "0")) > 0 else Side.SELL,
            qty=Decimal(str(abs(int(order_data.get("units", "0"))))),
            filled_qty=filled_qty,
            status=state,
            fill_price=Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
        )

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by currency pair (optional)

        Returns:
            List of open orders
        """
        session = self._ensure_session()

        url = f"{self._base_url}/v3/accounts/{self._account_id}/pendingOrders"

        try:
            response = session.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")
            return []

        orders = []
        for order_data in data.get("orders", []):
            instrument = order_data.get("instrument", "")

            # Filter by symbol if provided
            if symbol:
                norm_symbol = self._normalize_symbol(symbol)
                if instrument != norm_symbol:
                    continue

            units = int(order_data.get("units", "0"))
            side = Side.BUY if units > 0 else Side.SELL

            order = Order(
                symbol=instrument,
                side=side,
                qty=Decimal(str(abs(units))),
                order_type=order_data.get("type", "LIMIT"),
                price=Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                client_order_id=order_data.get("clientExtensions", {}).get("id"),
            )
            orders.append(order)

        return orders

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Position]:
        """
        Get current positions.

        Args:
            symbols: Filter by symbols (optional)

        Returns:
            Dict mapping symbol to Position
        """
        session = self._ensure_session()

        url = f"{self._base_url}/v3/accounts/{self._account_id}/positions"

        try:
            response = session.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return {}

        positions: Dict[str, Position] = {}

        for pos_data in data.get("positions", []):
            instrument = pos_data.get("instrument", "")

            # Filter by symbols if provided
            if symbols:
                norm_symbols = [self._normalize_symbol(s) for s in symbols]
                if instrument not in norm_symbols:
                    continue

            long_data = pos_data.get("long", {})
            short_data = pos_data.get("short", {})

            long_units = int(long_data.get("units", "0"))
            short_units = int(short_data.get("units", "0"))
            net_units = long_units + short_units  # Short units are negative

            if net_units == 0:
                continue  # No position

            side = Side.BUY if net_units > 0 else Side.SELL
            avg_price = Decimal("0")

            if net_units > 0 and long_data.get("averagePrice"):
                avg_price = Decimal(str(long_data.get("averagePrice", "0")))
            elif net_units < 0 and short_data.get("averagePrice"):
                avg_price = Decimal(str(short_data.get("averagePrice", "0")))

            # Calculate unrealized P&L
            long_pnl = float(long_data.get("unrealizedPL", "0"))
            short_pnl = float(short_data.get("unrealizedPL", "0"))
            unrealized_pnl = long_pnl + short_pnl

            # Position uses signed qty (negative for short)
            signed_qty = Decimal(str(net_units))  # Already signed: negative for short
            positions[instrument] = Position(
                symbol=instrument,
                qty=signed_qty,
                avg_entry_price=avg_price,
                meta={"unrealized_pnl": unrealized_pnl, "side": side},
            )

        return positions

    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        session = self._ensure_session()

        url = f"{self._base_url}/v3/accounts/{self._account_id}"

        try:
            response = session.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(
                vendor=ExchangeVendor.OANDA,
                account_id=self._account_id or "",
                account_type="margin",
                margin_enabled=True,
            )

        account = data.get("account", {})

        return AccountInfo(
            vendor=ExchangeVendor.OANDA,
            account_id=account.get("id", self._account_id or ""),
            account_type="margin",
            buying_power=Decimal(str(account.get("marginAvailable", "0"))),
            cash_balance=Decimal(str(account.get("balance", "0"))),
            margin_enabled=True,
            raw_data={
                "currency": account.get("currency", "USD"),
                "margin_used": account.get("marginUsed", "0"),
                "unrealized_pnl": account.get("unrealizedPL", "0"),
            },
        )

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: OANDA order ID
            price: New limit price
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            True if modification successful
        """
        session = self._ensure_session()

        # Build modification request
        modifications: Dict[str, Any] = {"order": {}}

        if price is not None:
            modifications["order"]["price"] = str(price)

        if stop_loss is not None:
            modifications["order"]["stopLossOnFill"] = {
                "price": str(stop_loss)
            }

        if take_profit is not None:
            modifications["order"]["takeProfitOnFill"] = {
                "price": str(take_profit)
            }

        if not modifications["order"]:
            return False

        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders/{order_id}"

        try:
            response = session.put(url, json=modifications, timeout=self._timeout)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Order modification failed: {e}")
            return False

    def close_position(
        self,
        symbol: str,
        units: Optional[int] = None,
    ) -> OrderResult:
        """
        Close an open position.

        Args:
            symbol: Currency pair
            units: Number of units to close (None = close all)

        Returns:
            OrderResult with close transaction details
        """
        session = self._ensure_session()
        instrument = self._normalize_symbol(symbol)

        url = f"{self._base_url}/v3/accounts/{self._account_id}/positions/{instrument}/close"

        close_data: Dict[str, Any] = {}
        if units is not None:
            # Close specific number of units
            if units > 0:
                close_data["longUnits"] = str(units)
            else:
                close_data["shortUnits"] = str(abs(units))
        else:
            # Close all
            close_data["longUnits"] = "ALL"
            close_data["shortUnits"] = "ALL"

        try:
            response = session.put(url, json=close_data, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Position close failed: {e}")
            return OrderResult(
                success=False,
                error_code="CLOSE_FAILED",
                error_message=str(e),
            )

        # Parse close transaction
        long_close = data.get("longOrderFillTransaction")
        short_close = data.get("shortOrderFillTransaction")

        fill = long_close or short_close
        if fill:
            return OrderResult(
                success=True,
                order_id=str(fill.get("orderID", "")),
                status="FILLED",
                filled_qty=Decimal(str(abs(int(fill.get("units", 0))))),
                filled_price=Decimal(str(fill.get("price", "0"))),
                raw_response=data,
            )

        return OrderResult(
            success=False,
            error_code="NO_FILL",
            error_message="No position to close",
            raw_response=data,
        )

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get recent trade history.

        Args:
            symbol: Filter by currency pair
            limit: Maximum trades to return

        Returns:
            List of trade dicts
        """
        session = self._ensure_session()

        url = f"{self._base_url}/v3/accounts/{self._account_id}/trades"
        params = {"count": min(limit, 500)}

        if symbol:
            params["instrument"] = self._normalize_symbol(symbol)

        try:
            response = session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to get trade history: {e}")
            return []

        return data.get("trades", [])
