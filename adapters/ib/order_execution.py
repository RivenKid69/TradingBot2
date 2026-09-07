# -*- coding: utf-8 -*-
"""
adapters/ib/order_execution.py
Interactive Brokers order execution adapter for CME Group futures.

Supports market, limit, stop, and bracket orders for CME futures.

Features:
- Market, limit, stop orders
- Bracket orders (entry + take-profit + stop-loss)
- Position queries with margin info
- Order status tracking
- Account margin information
- What-if margin calculations

Reference:
- IB Order Types: https://interactivebrokers.github.io/tws-api/available_order_types.html
- ib_insync: https://ib-insync.readthedocs.io/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor, AccountInfo
from core_models import Order, ExecReport, Position, Side

# Import core_futures models
from core_futures import (
    FuturesPosition,
    FuturesOrder,
    FuturesFill,
    PositionSide,
    MarginMode,
    OrderSide,
    OrderType as FuturesOrderType,
    TimeInForce,
    OrderStatus,
)

logger = logging.getLogger(__name__)

# Try to import ib_insync, but don't fail if not installed
try:
    from ib_insync import (
        IB,
        Future,
        ContFuture,
        MarketOrder,
        LimitOrder,
        StopOrder,
        StopLimitOrder,
        BracketOrder,
        Trade,
    )

    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    IB = None
    Future = None
    MarketOrder = None
    LimitOrder = None
    StopOrder = None


# =========================
# Order Type Mapping
# =========================


class IBOrderAction(str, Enum):
    """IB order action (direction)."""

    BUY = "BUY"
    SELL = "SELL"


# Contract mapping (shared with market_data)
CONTRACT_MAP: Dict[str, Dict[str, str]] = {
    # Equity Index
    "ES": {"exchange": "CME", "currency": "USD"},
    "NQ": {"exchange": "CME", "currency": "USD"},
    "RTY": {"exchange": "CME", "currency": "USD"},
    "YM": {"exchange": "CBOT", "currency": "USD"},
    "MES": {"exchange": "CME", "currency": "USD"},
    "MNQ": {"exchange": "CME", "currency": "USD"},
    # Metals
    "GC": {"exchange": "COMEX", "currency": "USD"},
    "SI": {"exchange": "COMEX", "currency": "USD"},
    "HG": {"exchange": "COMEX", "currency": "USD"},
    # Energy
    "CL": {"exchange": "NYMEX", "currency": "USD"},
    "NG": {"exchange": "NYMEX", "currency": "USD"},
    # Currencies
    "6E": {"exchange": "CME", "currency": "USD"},
    "6J": {"exchange": "CME", "currency": "USD"},
    "6B": {"exchange": "CME", "currency": "USD"},
    # Bonds
    "ZB": {"exchange": "CBOT", "currency": "USD"},
    "ZN": {"exchange": "CBOT", "currency": "USD"},
}


# =========================
# Bracket Order Config
# =========================


@dataclass
class IBBracketOrderConfig:
    """Configuration for IB bracket order."""

    symbol: str
    side: str  # BUY or SELL
    qty: int
    entry_price: Optional[Decimal] = None  # For limit entry
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    time_in_force: str = "GTC"


# =========================
# IB Order Execution Adapter
# =========================


class IBOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Interactive Brokers futures order execution adapter.

    Features:
    - Market, limit, stop, stop-limit orders
    - Bracket orders (entry + take-profit + stop-loss)
    - Position queries with margin info
    - Account margin and buying power info
    - What-if margin calculations

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live)
        client_id: Unique client ID
        account: Specific account to use
        default_exchange: Default exchange for symbol lookup

    Usage:
        from adapters.ib import IBOrderExecutionAdapter
        from adapters.models import ExchangeVendor

        adapter = IBOrderExecutionAdapter(
            vendor=ExchangeVendor.IB,
            config={"port": 7497}
        )
        adapter.connect()

        # Submit market order
        result = adapter.submit_market_order("ES", "BUY", 1)

        # Get positions
        positions = adapter.get_futures_positions()

        adapter.disconnect()
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.IB,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._ib: Optional[IB] = None
        self._host = self._config.get("host", "127.0.0.1")
        self._port = self._config.get("port", 7497)
        self._client_id = self._config.get("client_id", 2)  # Different from market data
        self._account = self._config.get("account")
        self._default_exchange = self._config.get("default_exchange", "CME")

    def _do_connect(self) -> None:
        """Connect to TWS/Gateway."""
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IB adapters. " "Install with: pip install ib_insync"
            )

        self._ib = IB()
        self._ib.connect(
            self._host,
            self._port,
            clientId=self._client_id,
            readonly=False,  # Need write access for orders
            account=self._account,
            timeout=self._config.get("timeout", 10.0),
        )
        logger.info(f"Connected to IB TWS for order execution at {self._host}:{self._port}")

    def _do_disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB TWS")
        self._ib = None

    def _create_contract(
        self,
        symbol: str,
        use_continuous: bool = True,
        exchange: Optional[str] = None,
    ):
        """Create IB Future contract."""
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync is required for IB adapters")

        details = CONTRACT_MAP.get(
            symbol.upper(), {"exchange": exchange or self._default_exchange, "currency": "USD"}
        )
        actual_exchange = exchange or details.get("exchange", self._default_exchange)

        if use_continuous:
            return ContFuture(symbol, exchange=actual_exchange)
        else:
            return Future(symbol, exchange=actual_exchange)

    # =========================
    # Futures-Specific Methods
    # =========================

    def get_futures_positions(self) -> List[FuturesPosition]:
        """
        Get all futures positions.

        Returns:
            List of FuturesPosition objects
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        positions = []
        for pos in self._ib.positions():
            contract = pos.contract
            if hasattr(contract, "secType") and contract.secType == "FUT":
                qty = Decimal(str(pos.position))
                if qty == 0:
                    continue

                # Get multiplier for entry price calculation
                multiplier = Decimal(str(contract.multiplier or 1))
                entry_price = Decimal(str(pos.avgCost)) / multiplier

                positions.append(
                    FuturesPosition(
                        symbol=contract.symbol,
                        side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                        entry_price=entry_price,
                        qty=abs(qty),
                        leverage=1,  # IB doesn't expose effective leverage
                        margin_mode=MarginMode.CROSS,  # IB uses portfolio margin
                        unrealized_pnl=Decimal("0"),  # Would need market data for this
                        realized_pnl=Decimal("0"),
                    )
                )

        return positions

    def get_account_margin(self) -> Dict[str, Decimal]:
        """
        Get account margin information.

        Returns:
            Dict with margin values:
            - InitMarginReq: Initial margin requirement
            - MaintMarginReq: Maintenance margin requirement
            - AvailableFunds: Available funds
            - ExcessLiquidity: Excess liquidity
            - BuyingPower: Buying power
            - NetLiquidation: Net liquidation value
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        summary = self._ib.accountSummary()
        margin_info: Dict[str, Decimal] = {}

        relevant_tags = {
            "InitMarginReq",
            "MaintMarginReq",
            "AvailableFunds",
            "ExcessLiquidity",
            "BuyingPower",
            "NetLiquidation",
            "TotalCashValue",
            "GrossPositionValue",
        }

        for item in summary:
            if item.tag in relevant_tags:
                try:
                    margin_info[item.tag] = Decimal(str(item.value))
                except (ValueError, TypeError):
                    pass

        return margin_info

    def get_margin_requirement(
        self,
        symbol: str,
        qty: int,
        side: str = "BUY",
    ) -> Dict[str, Decimal]:
        """
        Get margin requirement for a hypothetical order (what-if).

        Args:
            symbol: Futures symbol
            qty: Number of contracts
            side: BUY or SELL

        Returns:
            Dict with margin impact:
            - initial_margin: Initial margin change
            - maint_margin: Maintenance margin change
            - equity_impact: Impact on equity
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, qty)

        what_if = self._ib.whatIfOrder(contract, order)

        return {
            "initial_margin": Decimal(str(what_if.initMarginChange or 0)),
            "maint_margin": Decimal(str(what_if.maintMarginChange or 0)),
            "equity_impact": Decimal(str(what_if.equityWithLoanChange or 0)),
            "commission": Decimal(str(what_if.commission or 0)),
        }

    # =========================
    # Order Submission Methods
    # =========================

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
    ) -> FuturesOrder:
        """
        Submit market order.

        Args:
            symbol: Futures symbol
            side: BUY or SELL
            qty: Number of contracts

        Returns:
            FuturesOrder with order details
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, qty)

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade_to_futures_order(trade, symbol)

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Decimal,
        time_in_force: str = "GTC",
    ) -> FuturesOrder:
        """
        Submit limit order.

        Args:
            symbol: Futures symbol
            side: BUY or SELL
            qty: Number of contracts
            price: Limit price
            time_in_force: GTC, DAY, IOC, FOK

        Returns:
            FuturesOrder with order details
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = LimitOrder(action, qty, float(price))
        order.tif = time_in_force.upper()

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade_to_futures_order(trade, symbol)

    def submit_stop_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        stop_price: Decimal,
        time_in_force: str = "GTC",
    ) -> FuturesOrder:
        """
        Submit stop order.

        Args:
            symbol: Futures symbol
            side: BUY or SELL
            qty: Number of contracts
            stop_price: Stop trigger price
            time_in_force: GTC, DAY, IOC, FOK

        Returns:
            FuturesOrder with order details
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = StopOrder(action, qty, float(stop_price))
        order.tif = time_in_force.upper()

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade_to_futures_order(trade, symbol)

    def submit_stop_limit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        stop_price: Decimal,
        limit_price: Decimal,
        time_in_force: str = "GTC",
    ) -> FuturesOrder:
        """
        Submit stop-limit order.

        Args:
            symbol: Futures symbol
            side: BUY or SELL
            qty: Number of contracts
            stop_price: Stop trigger price
            limit_price: Limit price after trigger
            time_in_force: GTC, DAY, IOC, FOK

        Returns:
            FuturesOrder with order details
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = StopLimitOrder(action, qty, float(limit_price), float(stop_price))
        order.tif = time_in_force.upper()

        trade = self._ib.placeOrder(contract, order)
        return self._convert_trade_to_futures_order(trade, symbol)

    def submit_bracket_order(
        self,
        config: IBBracketOrderConfig,
    ) -> Dict[str, FuturesOrder]:
        """
        Submit bracket order (entry + take-profit + stop-loss).

        Args:
            config: Bracket order configuration

        Returns:
            Dict with entry, take_profit, stop_loss FuturesOrders
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        contract = self._create_contract(config.symbol, use_continuous=False)
        self._ib.qualifyContracts(contract)

        action = "BUY" if config.side.upper() == "BUY" else "SELL"
        reverse_action = "SELL" if action == "BUY" else "BUY"

        # Create bracket order
        if config.entry_price:
            # Limit entry
            entry_order = LimitOrder(action, config.qty, float(config.entry_price))
        else:
            # Market entry
            entry_order = MarketOrder(action, config.qty)

        entry_order.tif = config.time_in_force.upper()
        entry_order.transmit = False  # Don't transmit yet

        # Take profit (limit order)
        tp_order = None
        if config.take_profit_price:
            tp_order = LimitOrder(reverse_action, config.qty, float(config.take_profit_price))
            tp_order.parentId = entry_order.orderId
            tp_order.tif = config.time_in_force.upper()
            tp_order.transmit = False

        # Stop loss (stop order)
        sl_order = None
        if config.stop_loss_price:
            sl_order = StopOrder(reverse_action, config.qty, float(config.stop_loss_price))
            sl_order.parentId = entry_order.orderId
            sl_order.tif = config.time_in_force.upper()
            sl_order.transmit = True  # Transmit all orders

        # Place orders
        trades: Dict[str, FuturesOrder] = {}

        entry_trade = self._ib.placeOrder(contract, entry_order)
        trades["entry"] = self._convert_trade_to_futures_order(entry_trade, config.symbol)

        if tp_order:
            tp_trade = self._ib.placeOrder(contract, tp_order)
            trades["take_profit"] = self._convert_trade_to_futures_order(tp_trade, config.symbol)

        if sl_order:
            sl_trade = self._ib.placeOrder(contract, sl_order)
            trades["stop_loss"] = self._convert_trade_to_futures_order(sl_trade, config.symbol)

        return trades

    # =========================
    # Base Class Implementation
    # =========================

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to exchange.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status and fill info
        """
        try:
            if order.order_type.upper() == "MARKET":
                futures_order = self.submit_market_order(
                    order.symbol,
                    order.side.value if hasattr(order.side, "value") else order.side,
                    int(order.qty),
                )
            elif order.order_type.upper() == "LIMIT":
                futures_order = self.submit_limit_order(
                    order.symbol,
                    order.side.value if hasattr(order.side, "value") else order.side,
                    int(order.qty),
                    order.price,
                )
            else:
                return OrderResult(
                    success=False,
                    error_message=f"Unsupported order type: {order.order_type}",
                )

            return OrderResult(
                success=True,
                order_id=futures_order.order_id,
                client_order_id=futures_order.client_order_id,
                status=futures_order.status.value,
            )
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return OrderResult(
                success=False,
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
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        # Find the trade
        for trade in self._ib.openTrades():
            if (order_id and str(trade.order.orderId) == order_id) or (
                client_order_id and trade.order.orderRef == client_order_id
            ):
                self._ib.cancelOrder(trade.order)
                return True

        return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            ExecReport with current status, or None if not found
        """
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        for trade in self._ib.trades():
            if (order_id and str(trade.order.orderId) == order_id) or (
                client_order_id and trade.order.orderRef == client_order_id
            ):
                return ExecReport(
                    order_id=str(trade.order.orderId),
                    symbol=trade.contract.symbol,
                    side=Side.BUY if trade.order.action == "BUY" else Side.SELL,
                    qty=Decimal(str(trade.order.totalQuantity)),
                    filled_qty=Decimal(str(trade.orderStatus.filled)),
                    status=trade.orderStatus.status,
                    price=Decimal(str(trade.order.lmtPrice)) if trade.order.lmtPrice else None,
                    avg_price=(
                        Decimal(str(trade.orderStatus.avgFillPrice))
                        if trade.orderStatus.avgFillPrice
                        else None
                    ),
                )

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
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        orders = []
        for trade in self._ib.openTrades():
            if symbol and trade.contract.symbol != symbol:
                continue

            orders.append(
                Order(
                    symbol=trade.contract.symbol,
                    side=Side.BUY if trade.order.action == "BUY" else Side.SELL,
                    qty=Decimal(str(trade.order.totalQuantity)),
                    order_type=self._ib_order_type_to_str(trade.order),
                    price=Decimal(str(trade.order.lmtPrice)) if trade.order.lmtPrice else None,
                    client_order_id=trade.order.orderRef or str(trade.order.orderId),
                )
            )

        return orders

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
        if not self._ib:
            raise ConnectionError("Not connected to IB")

        positions: Dict[str, Position] = {}

        for pos in self._ib.positions():
            contract = pos.contract
            if hasattr(contract, "secType") and contract.secType == "FUT":
                if symbols and contract.symbol not in symbols:
                    continue

                qty = Decimal(str(pos.position))
                if qty == 0:
                    continue

                multiplier = Decimal(str(contract.multiplier or 1))
                entry_price = Decimal(str(pos.avgCost)) / multiplier

                positions[contract.symbol] = Position(
                    symbol=contract.symbol,
                    qty=qty,
                    side=Side.BUY if qty > 0 else Side.SELL,
                    entry_price=entry_price,
                )

        return positions

    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        margin_info = self.get_account_margin()

        return AccountInfo(
            account_id=self._account or "",
            equity=margin_info.get("NetLiquidation", Decimal("0")),
            available=margin_info.get("AvailableFunds", Decimal("0")),
            margin_used=margin_info.get("InitMarginReq", Decimal("0")),
            unrealized_pnl=Decimal("0"),  # Would need calculation
            is_active=self._ib.isConnected() if self._ib else False,
        )

    # =========================
    # Helper Methods
    # =========================

    def _convert_trade_to_futures_order(
        self,
        trade: Any,
        symbol: str,
    ) -> FuturesOrder:
        """Convert IB Trade to FuturesOrder."""
        order = trade.order
        status = trade.orderStatus

        return FuturesOrder(
            order_id=str(order.orderId),
            client_order_id=order.orderRef or str(order.orderId),
            symbol=symbol,
            side=OrderSide.BUY if order.action == "BUY" else OrderSide.SELL,
            order_type=self._ib_order_type_to_futures_type(order),
            qty=Decimal(str(order.totalQuantity)),
            filled_qty=Decimal(str(status.filled)),
            price=Decimal(str(order.lmtPrice)) if order.lmtPrice else None,
            avg_fill_price=Decimal(str(status.avgFillPrice)) if status.avgFillPrice else None,
            status=self._ib_status_to_order_status(status.status),
            time_in_force=(
                TimeInForce(order.tif)
                if order.tif in [e.value for e in TimeInForce]
                else TimeInForce.GTC
            ),
            created_at=int(time.time() * 1000),
        )

    @staticmethod
    def _ib_order_type_to_futures_type(order: Any) -> FuturesOrderType:
        """Convert IB order type to FuturesOrderType."""
        order_type = order.orderType.upper()
        if order_type == "MKT":
            return FuturesOrderType.MARKET
        elif order_type == "LMT":
            return FuturesOrderType.LIMIT
        elif order_type == "STP":
            return FuturesOrderType.STOP_MARKET
        elif order_type == "STP LMT":
            return FuturesOrderType.STOP_LIMIT
        else:
            return FuturesOrderType.LIMIT

    @staticmethod
    def _ib_order_type_to_str(order: Any) -> str:
        """Convert IB order to type string."""
        order_type = order.orderType.upper()
        if order_type == "MKT":
            return "MARKET"
        elif order_type == "LMT":
            return "LIMIT"
        elif order_type == "STP":
            return "STOP"
        elif order_type == "STP LMT":
            return "STOP_LIMIT"
        else:
            return order_type

    @staticmethod
    def _ib_status_to_order_status(ib_status: str) -> OrderStatus:
        """Convert IB status to OrderStatus."""
        status_upper = ib_status.upper()
        if status_upper in ("SUBMITTED", "PRESUBMITTED"):
            return OrderStatus.NEW
        elif status_upper == "FILLED":
            return OrderStatus.FILLED
        elif status_upper in ("CANCELLED", "INACTIVE"):
            return OrderStatus.CANCELLED
        elif status_upper == "PENDINGSUBMIT":
            return OrderStatus.NEW
        elif "PARTIAL" in status_upper:
            return OrderStatus.PARTIALLY_FILLED
        else:
            return OrderStatus.NEW
