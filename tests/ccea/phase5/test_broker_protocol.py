# -*- coding: utf-8 -*-
"""
Tests for Broker Protocol and Adapters.

Design Doc Section 4.2: Broker Connectors
- BrokerConnector protocol definition
- Adapters for different brokers (Alpaca, Binance)
- Order management, position management
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

from packages.agent.broker.protocol import (
    # Enums
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    PositionSide,
    ConnectionStatus,
    # Data classes
    BrokerCredentials,
    OrderRequest,
    OrderResult,
    OrderInfo,
    Position,
    AccountInfo,
    CancelResult,
    BulkCancelResult,
    # Protocol and base
    BrokerConnector,
    BaseBrokerConnector,
    # Factory
    BrokerConnectorFactory,
)


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_order_sides(self):
        """Test order side values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_side_from_string(self):
        """Test creating from string."""
        assert OrderSide("buy") == OrderSide.BUY
        assert OrderSide("sell") == OrderSide.SELL


class TestOrderType:
    """Tests for OrderType enum."""

    def test_order_types(self):
        """Test order type values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_all_order_types_have_values(self):
        """Test all order types have string values."""
        for order_type in OrderType:
            assert isinstance(order_type.value, str)


class TestTimeInForce:
    """Tests for TimeInForce enum."""

    def test_time_in_force_values(self):
        """Test time in force values."""
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_order_status_values(self):
        """Test order status values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.ACCEPTED.value == "accepted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestPositionSide:
    """Tests for PositionSide enum."""

    def test_position_sides(self):
        """Test position side values."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"
        assert PositionSide.FLAT.value == "flat"


class TestConnectionStatus:
    """Tests for ConnectionStatus enum."""

    def test_connection_status_values(self):
        """Test connection status values."""
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.RECONNECTING.value == "reconnecting"
        assert ConnectionStatus.ERROR.value == "error"


class TestBrokerCredentials:
    """Tests for BrokerCredentials dataclass."""

    def test_create_credentials(self):
        """Test creating credentials."""
        creds = BrokerCredentials(
            api_key="test-key",
            api_secret="test-secret",
        )

        assert creds.api_key == "test-key"
        assert creds.api_secret == "test-secret"
        assert creds.passphrase is None
        assert creds.extra == {}

    def test_credentials_with_extra(self):
        """Test credentials with extra params."""
        creds = BrokerCredentials(
            api_key="test-key",
            api_secret="test-secret",
            extra={"account_id": "12345"},
        )

        assert creds.extra["account_id"] == "12345"

    def test_credentials_with_passphrase(self):
        """Test credentials with passphrase."""
        creds = BrokerCredentials(
            api_key="test-key",
            api_secret="test-secret",
            passphrase="my-passphrase",
        )

        assert creds.passphrase == "my-passphrase"

    def test_credentials_with_subaccount(self):
        """Test credentials with subaccount."""
        creds = BrokerCredentials(
            api_key="test-key",
            api_secret="test-secret",
            subaccount="sub123",
        )

        assert creds.subaccount == "sub123"


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_create_market_order(self):
        """Test creating market order request."""
        request = OrderRequest(
            client_order_id="client-123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
        )

        assert request.symbol == "AAPL"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.MARKET
        assert request.quantity == Decimal("10")
        assert request.client_order_id == "client-123"

    def test_create_limit_order(self):
        """Test creating limit order request."""
        request = OrderRequest(
            client_order_id="client-456",
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5"),
            limit_price=Decimal("400.50"),
            time_in_force=TimeInForce.GTC,
        )

        assert request.limit_price == Decimal("400.50")
        assert request.time_in_force == TimeInForce.GTC

    def test_order_request_with_context(self):
        """Test order request with context (strategy, deployment, run)."""
        request = OrderRequest(
            client_order_id="client-789",
            symbol="GOOG",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2"),
            strategy_id="strat-001",
            deployment_id="deploy-001",
            run_id="run-001",
        )

        assert request.strategy_id == "strat-001"
        assert request.deployment_id == "deploy-001"
        assert request.run_id == "run-001"


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_successful_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            client_order_id="client-123",
            broker_order_id="broker-123",
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("10"),
            avg_fill_price=Decimal("150.25"),
        )

        assert result.success is True
        assert result.broker_order_id == "broker-123"
        assert result.status == OrderStatus.FILLED

    def test_failed_result(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            client_order_id="client-456",
            error_message="Insufficient funds",
            error_code="INSUFFICIENT_FUNDS",
        )

        assert result.success is False
        assert result.error_message == "Insufficient funds"


class TestOrderInfo:
    """Tests for OrderInfo dataclass."""

    def test_create_order_info(self):
        """Test creating order info."""
        info = OrderInfo(
            client_order_id="client-123",
            broker_order_id="broker-123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            filled_quantity=Decimal("0"),
            limit_price=Decimal("150.00"),
            stop_price=None,
            avg_fill_price=None,
            status=OrderStatus.ACCEPTED,
            time_in_force=TimeInForce.GTC,
            created_at=datetime.utcnow(),
        )

        assert info.client_order_id == "client-123"
        assert info.limit_price == Decimal("150.00")
        assert info.status == OrderStatus.ACCEPTED


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_long_position(self):
        """Test creating long position."""
        position = Position(
            symbol="AAPL",
            quantity=Decimal("100"),
            side=PositionSide.LONG,
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        assert position.symbol == "AAPL"
        assert position.side == PositionSide.LONG
        assert position.quantity == Decimal("100")

    def test_create_short_position(self):
        """Test creating short position."""
        position = Position(
            symbol="TSLA",
            quantity=Decimal("50"),
            side=PositionSide.SHORT,
            avg_entry_price=Decimal("200.00"),
            current_price=Decimal("180.00"),
        )

        assert position.side == PositionSide.SHORT
        assert position.quantity == Decimal("50")

    def test_position_with_pnl(self):
        """Test position with P&L values set."""
        position = Position(
            symbol="GOOG",
            quantity=Decimal("10"),
            side=PositionSide.LONG,
            avg_entry_price=Decimal("100.00"),
            current_price=Decimal("110.00"),
            unrealized_pnl=Decimal("100.00"),
            market_value=Decimal("1100.00"),
        )

        assert position.unrealized_pnl == Decimal("100.00")
        assert position.market_value == Decimal("1100.00")


class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_create_account_info(self):
        """Test creating account info."""
        info = AccountInfo(
            account_id="12345",
            equity=Decimal("100000.00"),
            cash=Decimal("50000.00"),
            buying_power=Decimal("200000.00"),
        )

        assert info.account_id == "12345"
        assert info.equity == Decimal("100000.00")

    def test_account_info_with_margin(self):
        """Test account info with margin values."""
        info = AccountInfo(
            account_id="12345",
            equity=Decimal("100000.00"),
            cash=Decimal("50000.00"),
            buying_power=Decimal("200000.00"),
            margin_used=Decimal("50000.00"),
            margin_available=Decimal("150000.00"),
        )

        assert info.margin_used == Decimal("50000.00")


class TestCancelResult:
    """Tests for CancelResult dataclass."""

    def test_successful_cancel(self):
        """Test successful cancel result."""
        result = CancelResult(
            success=True,
            client_order_id="client-123",
        )

        assert result.success is True
        assert result.client_order_id == "client-123"

    def test_failed_cancel(self):
        """Test failed cancel result."""
        result = CancelResult(
            success=False,
            client_order_id="client-123",
            error_message="Order already filled",
        )

        assert result.success is False
        assert result.error_message == "Order already filled"


class TestBulkCancelResult:
    """Tests for BulkCancelResult dataclass."""

    def test_bulk_cancel_result(self):
        """Test bulk cancel result."""
        result = BulkCancelResult(
            total_requested=3,
            total_cancelled=2,
            total_failed=1,
            results=[
                CancelResult(success=True, client_order_id="client-1"),
                CancelResult(success=True, client_order_id="client-2"),
                CancelResult(success=False, client_order_id="client-3", error_message="Failed"),
            ],
            errors=["Order 3 failed"],
        )

        assert result.total_requested == 3
        assert result.total_cancelled == 2
        assert result.total_failed == 1
        assert len(result.results) == 3
        assert len(result.errors) == 1


class TestBrokerConnectorFactory:
    """Tests for BrokerConnectorFactory."""

    def test_register_and_create(self):
        """Test registering and creating connector."""

        class MockConnector(BaseBrokerConnector):
            @property
            def broker_name(self) -> str:
                return "test_mock"

            def connect(self) -> bool:
                return True

            def disconnect(self) -> bool:
                return True

            def submit_order(self, request):
                pass

            def cancel_order(self, order_id=None, client_order_id=None, symbol=None):
                pass

            def cancel_all_orders(self, symbol=None):
                pass

            def get_order(self, order_id=None, client_order_id=None):
                pass

            def get_open_orders(self, symbol=None):
                return []

            def get_positions(self):
                return []

            def get_position(self, symbol):
                pass

            def close_position(self, symbol, quantity=None):
                pass

            def close_all_positions(self):
                return []

            def get_account_info(self):
                pass

            def get_account(self):
                return AccountInfo(
                    account_id="test",
                    equity=Decimal("100000.00"),
                    cash=Decimal("50000.00"),
                    buying_power=Decimal("100000.00"),
                )

        BrokerConnectorFactory.register("test_mock", MockConnector)

        assert "test_mock" in BrokerConnectorFactory.list_brokers()

        creds = BrokerCredentials(api_key="test", api_secret="secret")
        connector = BrokerConnectorFactory.create("test_mock", creds)

        assert isinstance(connector, MockConnector)

    def test_create_unknown_broker(self):
        """Test creating unknown broker raises error."""
        creds = BrokerCredentials(api_key="test", api_secret="secret")

        with pytest.raises(ValueError, match="Unknown broker"):
            BrokerConnectorFactory.create("unknown_broker_xyz", creds)


class TestBaseBrokerConnector:
    """Tests for BaseBrokerConnector abstract class."""

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # BaseBrokerConnector should not be instantiable directly
        # due to abstract methods
        with pytest.raises(TypeError):
            BaseBrokerConnector(BrokerCredentials(api_key="test", api_secret="secret"))


class TestBrokerConnectorProtocol:
    """Tests for BrokerConnector protocol compliance."""

    def test_protocol_has_required_methods(self):
        """Test that compliant connector has all required methods."""

        class CompliantConnector:
            @property
            def is_connected(self) -> bool:
                return True

            @property
            def connection_status(self) -> ConnectionStatus:
                return ConnectionStatus.CONNECTED

            def connect(self) -> bool:
                return True

            def disconnect(self) -> bool:
                return True

            def submit_order(self, request: OrderRequest) -> OrderResult:
                return OrderResult(success=True, client_order_id="test")

            def cancel_order(
                self,
                order_id: Optional[str] = None,
                client_order_id: Optional[str] = None,
                symbol: Optional[str] = None,
            ) -> CancelResult:
                return CancelResult(client_order_id=client_order_id or "", success=True)

            def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
                return BulkCancelResult(
                    total_requested=0,
                    total_cancelled=0,
                    total_failed=0,
                    results=[],
                    errors=[],
                )

            def get_order(
                self, order_id: Optional[str] = None, client_order_id: Optional[str] = None
            ) -> Optional[OrderInfo]:
                return None

            def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
                return []

            def get_positions(self) -> List[Position]:
                return []

            def get_position(self, symbol: str) -> Optional[Position]:
                return None

            def close_position(
                self, symbol: str, quantity: Optional[Decimal] = None
            ) -> OrderResult:
                return OrderResult(success=True, client_order_id="test")

            def close_all_positions(self) -> List[OrderResult]:
                return []

            def get_account_info(self) -> Optional[AccountInfo]:
                return None

        connector = CompliantConnector()
        # Verify all required methods exist
        assert hasattr(connector, "is_connected")
        assert hasattr(connector, "connection_status")
        assert hasattr(connector, "connect")
        assert hasattr(connector, "disconnect")
        assert hasattr(connector, "submit_order")
        assert hasattr(connector, "cancel_order")
        assert hasattr(connector, "cancel_all_orders")
        assert hasattr(connector, "get_order")
        assert hasattr(connector, "get_open_orders")
        assert hasattr(connector, "get_positions")
        assert hasattr(connector, "get_position")
        assert hasattr(connector, "close_position")
        assert hasattr(connector, "close_all_positions")
        assert hasattr(connector, "get_account_info")

        # Verify methods are callable
        assert callable(connector.connect)
        assert callable(connector.disconnect)
        assert callable(connector.submit_order)
        assert callable(connector.cancel_all_orders)
        assert callable(connector.get_positions)


class TestOrderRequestValidation:
    """Tests for OrderRequest validation."""

    def test_limit_order_with_price(self):
        """Test limit order with limit_price."""
        request = OrderRequest(
            client_order_id="test-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            limit_price=Decimal("150.00"),
        )
        assert request.limit_price is not None

    def test_stop_order_with_stop_price(self):
        """Test stop order structure."""
        request = OrderRequest(
            client_order_id="test-002",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("10"),
            stop_price=Decimal("140.00"),
        )
        assert request.stop_price == Decimal("140.00")

    def test_stop_limit_with_both_prices(self):
        """Test stop limit order structure."""
        request = OrderRequest(
            client_order_id="test-003",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("10"),
            stop_price=Decimal("140.00"),
            limit_price=Decimal("139.50"),
        )
        assert request.stop_price == Decimal("140.00")
        assert request.limit_price == Decimal("139.50")


class TestEnumStringConversion:
    """Tests for enum to/from string conversion."""

    def test_order_side_is_str_enum(self):
        """Test OrderSide works as string."""
        side = OrderSide.BUY
        assert str(side) == "OrderSide.BUY"
        assert side.value == "buy"
        assert side == "buy"

    def test_order_type_is_str_enum(self):
        """Test OrderType works as string."""
        otype = OrderType.MARKET
        assert otype.value == "market"
        assert otype == "market"

    def test_order_status_is_str_enum(self):
        """Test OrderStatus works as string."""
        status = OrderStatus.FILLED
        assert status.value == "filled"
        assert status == "filled"
