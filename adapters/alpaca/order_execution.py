# -*- coding: utf-8 -*-
"""
adapters/alpaca/order_execution.py
Alpaca order execution adapter for US equities.

Phase 9: Live Trading Improvements (2025-11-27)
- Enhanced order management (cancel/replace, bracket orders)
- Extended hours support
- Position tracking and synchronization

Handles order submission, cancellation, bracket orders, and position tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core_models import Order, ExecReport, Position, Side, OrderType, ExecStatus, Liquidity
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import AccountInfo, ExchangeVendor

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 9: Order Management Enhancements
# =============================================================================


class BracketOrderType(str, Enum):
    """Types of bracket orders."""

    OTO = "oto"  # One-Triggers-Other
    OCO = "oco"  # One-Cancels-Other
    OTO_OCO = "oto_oco"  # Bracket (entry + TP + SL)


@dataclass
class BracketOrderConfig:
    """Configuration for bracket orders (OCO/OTO)."""

    # Primary order
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None

    # Take profit leg
    take_profit_price: Optional[float] = None
    take_profit_limit_price: Optional[float] = None  # If None, uses market

    # Stop loss leg
    stop_loss_price: Optional[float] = None
    stop_loss_limit_price: Optional[float] = None  # If None, uses market (stop-market)

    # Options
    time_in_force: str = "DAY"
    extended_hours: bool = False
    client_order_id: Optional[str] = None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate bracket order configuration."""
        if not self.symbol:
            return False, "Symbol required"
        if not self.qty or self.qty <= 0:
            return False, "Quantity must be positive"

        # Need at least one of TP or SL for bracket
        if self.take_profit_price is None and self.stop_loss_price is None:
            return False, "At least one of take_profit_price or stop_loss_price required"

        # Validate price relationships for BUY side
        if self.side == Side.BUY:
            if self.take_profit_price is not None and self.limit_price is not None:
                if self.take_profit_price <= self.limit_price:
                    return False, "Take profit must be above entry for BUY"
            if self.stop_loss_price is not None and self.limit_price is not None:
                if self.stop_loss_price >= self.limit_price:
                    return False, "Stop loss must be below entry for BUY"
            if self.take_profit_price and self.stop_loss_price:
                if self.take_profit_price <= self.stop_loss_price:
                    return False, "Take profit must be above stop loss for BUY"

        # Validate price relationships for SELL side
        elif self.side == Side.SELL:
            if self.take_profit_price is not None and self.limit_price is not None:
                if self.take_profit_price >= self.limit_price:
                    return False, "Take profit must be below entry for SELL"
            if self.stop_loss_price is not None and self.limit_price is not None:
                if self.stop_loss_price <= self.limit_price:
                    return False, "Stop loss must be above entry for SELL"
            if self.take_profit_price and self.stop_loss_price:
                if self.take_profit_price >= self.stop_loss_price:
                    return False, "Take profit must be below stop loss for SELL"

        return True, None


@dataclass
class BracketOrderResult:
    """Result of bracket order submission."""

    success: bool
    primary_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    bracket_type: Optional[BracketOrderType] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.raw_response is None:
            self.raw_response = {}


@dataclass
class ReplaceOrderConfig:
    """Configuration for order replacement."""

    order_id: Optional[str] = None
    client_order_id: Optional[str] = None

    # New values (None = keep original)
    new_qty: Optional[float] = None
    new_limit_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    new_time_in_force: Optional[str] = None

    # Options
    new_client_order_id: Optional[str] = None


class AlpacaOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Alpaca order execution adapter for US equities.

    Handles order placement, cancellation, and position management.

    Configuration:
        api_key: Alpaca API key (required)
        api_secret: Alpaca API secret (required)
        paper: Use paper trading endpoint (default: True)
        extended_hours: Allow extended hours trading (default: False)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._client = None

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError:
                raise ImportError("Alpaca SDK not installed. Install with: pip install alpaca-py")

            api_key = self._config.get("api_key")
            api_secret = self._config.get("api_secret")
            paper = self._config.get("paper", True)

            if not api_key or not api_secret:
                raise ValueError("Alpaca API key and secret are required")

            self._client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )

        return self._client

    def _do_connect(self) -> None:
        """Initialize client connection."""
        self._get_client()

    def _do_disconnect(self) -> None:
        """Close client connection."""
        self._client = None

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to Alpaca.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Convert side
            side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

            # Convert time in force
            tif_map = {
                "GTC": TimeInForce.GTC,
                "IOC": TimeInForce.IOC,
                "FOK": TimeInForce.FOK,
                "DAY": TimeInForce.DAY,
            }
            tif = tif_map.get(order.time_in_force.value, TimeInForce.DAY)

            # Extended hours
            extended = self._config.get("extended_hours", False)

            # Create order request
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                    extended_hours=extended,
                )
            else:  # LIMIT
                if order.price is None:
                    return OrderResult(
                        success=False,
                        error_code="MISSING_PRICE",
                        error_message="Limit orders require a price",
                    )

                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.price),
                    client_order_id=order.client_order_id,
                    extended_hours=extended,
                )

            # Submit order
            alpaca_order = client.submit_order(request)

            # Parse response
            return OrderResult(
                success=True,
                order_id=str(alpaca_order.id),
                client_order_id=str(alpaca_order.client_order_id),
                status=str(alpaca_order.status.value),
                filled_qty=Decimal(str(alpaca_order.filled_qty or 0)),
                filled_price=(
                    Decimal(str(alpaca_order.filled_avg_price))
                    if alpaca_order.filled_avg_price
                    else None
                ),
                raw_response={"id": str(alpaca_order.id)},
            )

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return OrderResult(
                success=False,
                error_code="SUBMISSION_FAILED",
                error_message=str(e),
            )

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Alpaca order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        try:
            client = self._get_client()

            if order_id:
                client.cancel_order_by_id(order_id)
            elif client_order_id:
                # Need to find order by client_order_id
                orders = client.get_orders()
                for o in orders:
                    if str(o.client_order_id) == client_order_id:
                        client.cancel_order_by_id(str(o.id))
                        return True
                logger.warning(f"Order with client_order_id {client_order_id} not found")
                return False
            else:
                raise ValueError("Either order_id or client_order_id required")

            return True

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Alpaca order ID
            client_order_id: Client order ID

        Returns:
            ExecReport or None
        """
        try:
            client = self._get_client()

            if order_id:
                alpaca_order = client.get_order_by_id(order_id)
            elif client_order_id:
                alpaca_order = client.get_order_by_client_id(client_order_id)
            else:
                return None

            return self._convert_order_to_exec_report(alpaca_order)

        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return None

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol

        Returns:
            List of open orders
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            if symbol:
                request.symbols = [symbol]

            alpaca_orders = client.get_orders(request)

            orders = []
            for o in alpaca_orders:
                order = self._convert_alpaca_order(o)
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
            symbols: Filter by symbols

        Returns:
            Dict mapping symbol to Position
        """
        try:
            client = self._get_client()
            alpaca_positions = client.get_all_positions()

            positions = {}
            for p in alpaca_positions:
                if symbols and p.symbol not in symbols:
                    continue

                import time

                position = Position(
                    symbol=p.symbol,
                    qty=Decimal(str(p.qty)),
                    avg_entry_price=Decimal(str(p.avg_entry_price)),
                    realized_pnl=Decimal("0"),  # Alpaca doesn't provide this directly
                    ts=int(time.time() * 1000),
                    meta={
                        "market_value": str(p.market_value),
                        "cost_basis": str(p.cost_basis),
                        "unrealized_pl": str(p.unrealized_pl),
                        "unrealized_plpc": str(p.unrealized_plpc),
                        "current_price": str(p.current_price),
                        "side": str(p.side),
                    },
                )
                positions[p.symbol] = position

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
            client = self._get_client()
            account = client.get_account()

            return AccountInfo(
                vendor=self._vendor,
                account_id=str(account.id),
                account_type="margin" if account.multiplier > 1 else "cash",
                vip_tier=0,  # Alpaca doesn't have VIP tiers
                maker_fee_rate=0.0,
                taker_fee_rate=0.0,
                buying_power=Decimal(str(account.buying_power)),
                cash_balance=Decimal(str(account.cash)),
                margin_enabled=account.multiplier > 1,
                pattern_day_trader=account.pattern_day_trader,
                raw_data={
                    "equity": str(account.equity),
                    "portfolio_value": str(account.portfolio_value),
                    "multiplier": str(account.multiplier),
                    "status": str(account.status),
                    "daytrade_count": account.daytrade_count,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(
                vendor=self._vendor,
                raw_data={"error": str(e)},
            )

    def _convert_alpaca_order(self, alpaca_order: Any) -> Optional[Order]:
        """Convert Alpaca order to our Order model."""
        try:
            from core_models import TimeInForce

            side = Side.BUY if str(alpaca_order.side) == "buy" else Side.SELL
            order_type = OrderType.MARKET if str(alpaca_order.type) == "market" else OrderType.LIMIT

            tif_map = {
                "day": TimeInForce.GTC,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }

            return Order(
                ts=int(alpaca_order.created_at.timestamp() * 1000),
                symbol=alpaca_order.symbol,
                side=side,
                order_type=order_type,
                quantity=Decimal(str(alpaca_order.qty)),
                price=Decimal(str(alpaca_order.limit_price)) if alpaca_order.limit_price else None,
                time_in_force=tif_map.get(str(alpaca_order.time_in_force), TimeInForce.GTC),
                client_order_id=str(alpaca_order.client_order_id),
                meta={"alpaca_id": str(alpaca_order.id)},
            )
        except Exception as e:
            logger.debug(f"Order conversion error: {e}")
            return None

    def _convert_order_to_exec_report(self, alpaca_order: Any) -> ExecReport:
        """Convert Alpaca order to ExecReport."""
        import time

        side = Side.BUY if str(alpaca_order.side) == "buy" else Side.SELL
        order_type = OrderType.MARKET if str(alpaca_order.type) == "market" else OrderType.LIMIT

        # Map status
        status_map = {
            "new": ExecStatus.NEW,
            "partially_filled": ExecStatus.PARTIALLY_FILLED,
            "filled": ExecStatus.FILLED,
            "canceled": ExecStatus.CANCELED,
            "expired": ExecStatus.CANCELED,
            "rejected": ExecStatus.REJECTED,
            "pending_new": ExecStatus.NEW,
            "accepted": ExecStatus.NEW,
        }
        status = status_map.get(str(alpaca_order.status), ExecStatus.NEW)

        return ExecReport(
            ts=int(time.time() * 1000),
            run_id="",
            symbol=alpaca_order.symbol,
            side=side,
            order_type=order_type,
            price=Decimal(str(alpaca_order.filled_avg_price or alpaca_order.limit_price or 0)),
            quantity=Decimal(str(alpaca_order.filled_qty or 0)),
            fee=Decimal("0"),  # Alpaca is commission-free
            fee_asset="USD",
            exec_status=status,
            liquidity=Liquidity.UNKNOWN,
            client_order_id=str(alpaca_order.client_order_id),
            order_id=str(alpaca_order.id),
            meta={
                "original_qty": str(alpaca_order.qty),
                "status": str(alpaca_order.status),
            },
        )

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Filter by symbol

        Returns:
            Number of orders cancelled
        """
        try:
            client = self._get_client()

            if symbol:
                # Cancel orders for specific symbol
                orders = self.get_open_orders(symbol)
                cancelled = 0
                for order in orders:
                    if self.cancel_order(client_order_id=order.client_order_id):
                        cancelled += 1
                return cancelled
            else:
                # Cancel all orders
                client.cancel_orders()
                return -1  # Unknown count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    # =========================================================================
    # Phase 9: Enhanced Order Management
    # =========================================================================

    def replace_order(self, config: ReplaceOrderConfig) -> OrderResult:
        """
        Replace (modify) an existing order.

        Alpaca supports modifying limit price, stop price, quantity, and TIF
        for pending orders. This is an atomic operation.

        Args:
            config: ReplaceOrderConfig with order ID and new values

        Returns:
            OrderResult with new order details
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import ReplaceOrderRequest

            # Build replace request
            request_params: Dict[str, Any] = {}

            if config.new_qty is not None:
                request_params["qty"] = config.new_qty
            if config.new_limit_price is not None:
                request_params["limit_price"] = config.new_limit_price
            if config.new_stop_price is not None:
                request_params["stop_price"] = config.new_stop_price
            if config.new_time_in_force is not None:
                from alpaca.trading.enums import TimeInForce

                tif_map = {
                    "GTC": TimeInForce.GTC,
                    "IOC": TimeInForce.IOC,
                    "FOK": TimeInForce.FOK,
                    "DAY": TimeInForce.DAY,
                }
                request_params["time_in_force"] = tif_map.get(
                    config.new_time_in_force.upper(), TimeInForce.DAY
                )
            if config.new_client_order_id is not None:
                request_params["client_order_id"] = config.new_client_order_id

            request = ReplaceOrderRequest(**request_params)

            # Get order ID
            order_id = config.order_id
            if not order_id and config.client_order_id:
                # Find order by client_order_id
                orders = client.get_orders()
                for o in orders:
                    if str(o.client_order_id) == config.client_order_id:
                        order_id = str(o.id)
                        break

            if not order_id:
                return OrderResult(
                    success=False,
                    error_code="ORDER_NOT_FOUND",
                    error_message="Order not found for replacement",
                )

            # Replace order
            alpaca_order = client.replace_order_by_id(order_id, request)

            return OrderResult(
                success=True,
                order_id=str(alpaca_order.id),
                client_order_id=str(alpaca_order.client_order_id),
                status=str(alpaca_order.status.value),
                filled_qty=Decimal(str(alpaca_order.filled_qty or 0)),
                filled_price=(
                    Decimal(str(alpaca_order.filled_avg_price))
                    if alpaca_order.filled_avg_price
                    else None
                ),
                raw_response={"id": str(alpaca_order.id), "replaced": True},
            )

        except Exception as e:
            logger.error(f"Order replacement failed: {e}")
            return OrderResult(
                success=False,
                error_code="REPLACE_FAILED",
                error_message=str(e),
            )

    def submit_bracket_order(self, config: BracketOrderConfig) -> BracketOrderResult:
        """
        Submit a bracket order (entry with take-profit and/or stop-loss).

        Alpaca supports bracket orders as a single atomic submission with:
        - Primary order (market or limit entry)
        - Take profit leg (limit order)
        - Stop loss leg (stop or stop-limit order)

        Args:
            config: BracketOrderConfig with all order details

        Returns:
            BracketOrderResult with all order IDs
        """
        # Validate configuration
        is_valid, error_msg = config.validate()
        if not is_valid:
            return BracketOrderResult(
                success=False,
                error_code="INVALID_CONFIG",
                error_message=error_msg,
            )

        try:
            client = self._get_client()

            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

            # Convert side
            side = OrderSide.BUY if config.side == Side.BUY else OrderSide.SELL

            # Convert time in force
            tif_map = {
                "GTC": TimeInForce.GTC,
                "IOC": TimeInForce.IOC,
                "FOK": TimeInForce.FOK,
                "DAY": TimeInForce.DAY,
            }
            tif = tif_map.get(config.time_in_force.upper(), TimeInForce.DAY)

            # Build take profit request
            take_profit_req = None
            if config.take_profit_price is not None:
                take_profit_req = TakeProfitRequest(
                    limit_price=config.take_profit_limit_price or config.take_profit_price
                )

            # Build stop loss request
            stop_loss_req = None
            if config.stop_loss_price is not None:
                if config.stop_loss_limit_price is not None:
                    # Stop-limit order
                    stop_loss_req = StopLossRequest(
                        stop_price=config.stop_loss_price,
                        limit_price=config.stop_loss_limit_price,
                    )
                else:
                    # Stop-market order
                    stop_loss_req = StopLossRequest(
                        stop_price=config.stop_loss_price,
                    )

            # Determine order class
            if take_profit_req and stop_loss_req:
                order_class = OrderClass.BRACKET
                bracket_type = BracketOrderType.OTO_OCO
            elif take_profit_req or stop_loss_req:
                order_class = OrderClass.OTO
                bracket_type = BracketOrderType.OTO
            else:
                order_class = OrderClass.SIMPLE
                bracket_type = None

            # Build primary order request
            if config.order_type == OrderType.LIMIT and config.limit_price is not None:
                request = LimitOrderRequest(
                    symbol=config.symbol,
                    qty=config.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=config.limit_price,
                    client_order_id=config.client_order_id,
                    extended_hours=config.extended_hours,
                    order_class=order_class,
                    take_profit=take_profit_req,
                    stop_loss=stop_loss_req,
                )
            else:
                request = MarketOrderRequest(
                    symbol=config.symbol,
                    qty=config.qty,
                    side=side,
                    time_in_force=tif,
                    client_order_id=config.client_order_id,
                    extended_hours=config.extended_hours,
                    order_class=order_class,
                    take_profit=take_profit_req,
                    stop_loss=stop_loss_req,
                )

            # Submit order
            alpaca_order = client.submit_order(request)

            # Extract leg order IDs
            tp_order_id = None
            sl_order_id = None

            legs = getattr(alpaca_order, "legs", None) or []
            for leg in legs:
                leg_type = getattr(leg, "order_type", "")
                if "limit" in str(leg_type).lower() and take_profit_req:
                    tp_order_id = str(leg.id)
                elif "stop" in str(leg_type).lower():
                    sl_order_id = str(leg.id)

            return BracketOrderResult(
                success=True,
                primary_order_id=str(alpaca_order.id),
                take_profit_order_id=tp_order_id,
                stop_loss_order_id=sl_order_id,
                bracket_type=bracket_type,
                raw_response={
                    "id": str(alpaca_order.id),
                    "order_class": str(order_class.value),
                    "status": str(alpaca_order.status.value),
                },
            )

        except Exception as e:
            logger.error(f"Bracket order submission failed: {e}")
            return BracketOrderResult(
                success=False,
                error_code="BRACKET_FAILED",
                error_message=str(e),
            )

    def submit_oco_order(
        self,
        symbol: str,
        qty: float,
        take_profit_price: float,
        stop_loss_price: float,
        *,
        take_profit_limit_price: Optional[float] = None,
        stop_loss_limit_price: Optional[float] = None,
        side: Side = Side.SELL,  # Default to exit
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> BracketOrderResult:
        """
        Submit an OCO (One-Cancels-Other) order for exits.

        This is a convenience method for common exit patterns:
        - Sets both take profit and stop loss
        - When one fills, the other is cancelled

        Args:
            symbol: Trading symbol
            qty: Quantity to sell
            take_profit_price: Price for take profit
            stop_loss_price: Stop trigger price for stop loss
            take_profit_limit_price: Limit price for TP (None = market)
            stop_loss_limit_price: Limit price for SL (None = market)
            side: Order side (default SELL for exits)
            time_in_force: Time in force
            client_order_id: Client order ID

        Returns:
            BracketOrderResult with OCO order IDs
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import (
                LimitOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

            # Convert parameters
            order_side = OrderSide.SELL if side == Side.SELL else OrderSide.BUY
            tif_map = {
                "GTC": TimeInForce.GTC,
                "DAY": TimeInForce.DAY,
            }
            tif = tif_map.get(time_in_force.upper(), TimeInForce.GTC)

            # Build take profit leg
            take_profit_req = TakeProfitRequest(
                limit_price=take_profit_limit_price or take_profit_price
            )

            # Build stop loss leg
            if stop_loss_limit_price:
                stop_loss_req = StopLossRequest(
                    stop_price=stop_loss_price,
                    limit_price=stop_loss_limit_price,
                )
            else:
                stop_loss_req = StopLossRequest(stop_price=stop_loss_price)

            # Use high limit price to ensure fill (OCO entry)
            # For OCO without primary, we use limit at current price
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=take_profit_price,  # Will be immediately filled or queued
                client_order_id=client_order_id,
                order_class=OrderClass.OCO,
                take_profit=take_profit_req,
                stop_loss=stop_loss_req,
            )

            alpaca_order = client.submit_order(request)

            return BracketOrderResult(
                success=True,
                primary_order_id=str(alpaca_order.id),
                bracket_type=BracketOrderType.OCO,
                raw_response={"id": str(alpaca_order.id)},
            )

        except Exception as e:
            logger.error(f"OCO order submission failed: {e}")
            return BracketOrderResult(
                success=False,
                error_code="OCO_FAILED",
                error_message=str(e),
            )

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[ExecReport]:
        """
        Get order history.

        Args:
            symbol: Filter by symbol
            status: Filter by status (open, closed, all)
            limit: Maximum orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp

        Returns:
            List of ExecReport objects
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            # Map status
            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            query_status = status_map.get(status or "all", QueryOrderStatus.ALL)

            request = GetOrdersRequest(
                status=query_status,
                limit=limit,
            )

            if symbol:
                request.symbols = [symbol]
            if after:
                request.after = after
            if until:
                request.until = until

            alpaca_orders = client.get_orders(request)

            reports = []
            for order in alpaca_orders:
                report = self._convert_order_to_exec_report(order)
                reports.append(report)

            return reports

        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []

    def wait_for_fill(
        self,
        order_id: str,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> Optional[ExecReport]:
        """
        Wait for an order to fill.

        Args:
            order_id: Order ID to monitor
            timeout_s: Maximum time to wait
            poll_interval_s: Polling interval

        Returns:
            ExecReport if filled, None if timeout or cancelled
        """
        start_time = time.time()

        while time.time() - start_time < timeout_s:
            status = self.get_order_status(order_id=order_id)

            if status is None:
                return None

            if status.exec_status == ExecStatus.FILLED:
                return status
            if status.exec_status in (ExecStatus.CANCELED, ExecStatus.REJECTED):
                return status

            time.sleep(poll_interval_s)

        return self.get_order_status(order_id=order_id)

    # =========================================================================
    # Extended Hours Support (Phase 9)
    # =========================================================================

    def submit_extended_hours_order(
        self,
        order: Order,
        session: str = "any",
    ) -> OrderResult:
        """
        Submit order with explicit extended hours handling.

        Args:
            order: Order to submit
            session: Session type - "regular", "extended", "any"

        Returns:
            OrderResult with status
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

            # Extended hours only works with limit orders
            if session in ("extended", "any"):
                if order.order_type == OrderType.MARKET:
                    # Extended hours requires limit orders
                    logger.warning(
                        "Extended hours trading requires limit orders. "
                        "Converting to limit order with offset."
                    )

            # TIF for extended hours should be DAY
            tif = TimeInForce.DAY

            # Determine extended hours flag
            extended = session in ("extended", "any")

            if order.order_type == OrderType.LIMIT or extended:
                if order.price is None:
                    return OrderResult(
                        success=False,
                        error_code="MISSING_PRICE",
                        error_message="Limit price required for extended hours",
                    )

                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.price),
                    client_order_id=order.client_order_id,
                    extended_hours=extended,
                )
            else:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                    extended_hours=False,  # Market orders can't use extended hours
                )

            alpaca_order = client.submit_order(request)

            return OrderResult(
                success=True,
                order_id=str(alpaca_order.id),
                client_order_id=str(alpaca_order.client_order_id),
                status=str(alpaca_order.status.value),
                filled_qty=Decimal(str(alpaca_order.filled_qty or 0)),
                filled_price=(
                    Decimal(str(alpaca_order.filled_avg_price))
                    if alpaca_order.filled_avg_price
                    else None
                ),
                raw_response={
                    "id": str(alpaca_order.id),
                    "extended_hours": extended,
                    "session": session,
                },
            )

        except Exception as e:
            logger.error(f"Extended hours order failed: {e}")
            return OrderResult(
                success=False,
                error_code="EXTENDED_HOURS_FAILED",
                error_message=str(e),
            )

    def is_extended_hours_eligible(self, symbol: str) -> bool:
        """
        Check if symbol is eligible for extended hours trading.

        Most US equities are eligible, but some restrictions apply.

        Args:
            symbol: Trading symbol

        Returns:
            True if eligible for extended hours
        """
        try:
            client = self._get_client()
            asset = client.get_asset(symbol)

            # Check if asset is tradable and fractionable (generally EH eligible)
            return (
                asset.tradable
                and asset.status == "active"
                and asset.exchange in ("NYSE", "NASDAQ", "ARCA", "AMEX")
            )

        except Exception as e:
            logger.warning(f"Failed to check extended hours eligibility: {e}")
            return False
