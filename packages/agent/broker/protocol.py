# -*- coding: utf-8 -*-
"""
BrokerConnector Protocol - Standard interface for broker integrations.

Design Doc Section 4.2/9.3:
- Broker Connectors (только тут)
- Agent единственная зона где создаются/отправляются ордера

This module defines the canonical BrokerConnector protocol that ALL
broker adapters must implement for use with CCEA Agent.

SECURITY:
- This module is AGENT ZONE ONLY
- Never import in Cloud zone (enforced by CI guardrails)
- Credentials must be retrieved from Local Vault only

References:
- Design Doc Section 9.3: Live Loop
- Design Doc Section 9.4: Kill Switch (cancel/flatten)
- Design Doc Section 9.5: Reconciliation/Idempotency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4


# =============================================================================
# Enums
# =============================================================================


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    """Time in force."""

    GTC = "gtc"  # Good til cancelled
    DAY = "day"  # Day order
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good til date
    OPG = "opg"  # At open
    CLS = "cls"  # At close


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"  # Not yet submitted
    SUBMITTED = "submitted"  # Sent to broker
    ACCEPTED = "accepted"  # Accepted by exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class PositionSide(str, Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ConnectionStatus(str, Enum):
    """Connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BrokerCredentials:
    """
    Broker credentials container.

    SECURITY: Never log or transmit these values.
    Retrieved from Local Vault only.
    """

    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # Some exchanges require this
    subaccount: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class OrderRequest:
    """
    Order request for submission.

    Design Doc Section 9.3: Order creation in Agent only.
    """

    # Identification
    client_order_id: str  # Deterministic for idempotency
    symbol: str

    # Order details
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC

    # Context
    strategy_id: Optional[str] = None
    deployment_id: Optional[str] = None
    run_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """
    Result of order submission.
    """

    success: bool
    client_order_id: str
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class OrderInfo:
    """
    Full order information.
    """

    client_order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    avg_fill_price: Optional[Decimal]
    status: OrderStatus
    time_in_force: TimeInForce
    commission: Decimal = Decimal("0")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    """
    Position information.
    """

    symbol: str
    side: PositionSide
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal("0")
    market_value: Optional[Decimal] = None
    cost_basis: Optional[Decimal] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountInfo:
    """
    Account information.
    """

    account_id: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    margin_used: Optional[Decimal] = None
    margin_available: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl_today: Decimal = Decimal("0")
    currency: str = "USD"
    status: str = "active"
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CancelResult:
    """
    Result of order cancellation.
    """

    success: bool
    client_order_id: str
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    cancelled_at: Optional[datetime] = None


@dataclass
class BulkCancelResult:
    """
    Result of bulk order cancellation.

    Design Doc Section 9.4: Kill Switch cancel all orders.
    """

    total_requested: int
    total_cancelled: int
    total_failed: int
    results: List[CancelResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class BrokerConnector(Protocol):
    """
    BrokerConnector Protocol - Standard interface for broker integrations.

    Design Doc Section 4.2: Broker Connectors (only in Agent)
    Design Doc Section 9.3: Live Loop where Orders are created

    All broker adapters MUST implement this protocol to be used with CCEA Agent.

    SECURITY:
    - Implementations must never log credentials
    - Implementations must never transmit credentials to Cloud
    - All network calls must use HTTPS/TLS

    Example:
        class AlpacaConnector:
            def __init__(self, credentials: BrokerCredentials):
                self._creds = credentials
                self._client = AlpacaClient(...)

            def submit_order(self, request: OrderRequest) -> OrderResult:
                ...

        # Usage
        connector: BrokerConnector = AlpacaConnector(creds)
        result = connector.submit_order(order_request)
    """

    # =========================================================================
    # Connection Management
    # =========================================================================

    @property
    def broker_name(self) -> str:
        """Return broker identifier (e.g., 'alpaca', 'binance')."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        ...

    @property
    def connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        ...

    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connection successful
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from broker."""
        ...

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary with at least:
            - connected: bool
            - latency_ms: float
            - last_heartbeat: datetime
        """
        ...

    # =========================================================================
    # Order Management
    # Design Doc Section 9.3: Order submission
    # =========================================================================

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit order to broker.

        IDEMPOTENCY: Uses client_order_id to prevent duplicates.

        Args:
            request: Order request

        Returns:
            Order result with status
        """
        ...

    def cancel_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> CancelResult:
        """
        Cancel a single order.

        Args:
            client_order_id: Client order ID
            broker_order_id: Broker order ID (if client_order_id not available)

        Returns:
            Cancel result
        """
        ...

    def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
        """
        Cancel all open orders.

        Design Doc Section 9.4: Kill Switch action.

        Args:
            symbol: Optional symbol filter (None = all symbols)

        Returns:
            Bulk cancel result
        """
        ...

    def get_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> Optional[OrderInfo]:
        """
        Get order information.

        Args:
            client_order_id: Client order ID
            broker_order_id: Broker order ID

        Returns:
            Order info or None if not found
        """
        ...

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        ...

    # =========================================================================
    # Position Management
    # Design Doc Section 9.5: Reconciliation
    # =========================================================================

    def get_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of current positions
        """
        ...

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Position or None if no position
        """
        ...

    def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> OrderResult:
        """
        Close position for symbol.

        Args:
            symbol: Symbol to close
            quantity: Quantity to close (None = full position)

        Returns:
            Order result for closing order
        """
        ...

    def close_all_positions(self) -> List[OrderResult]:
        """
        Close all positions (flatten).

        Design Doc Section 9.4: Kill Switch optional flatten.

        Returns:
            List of order results for closing orders
        """
        ...

    # =========================================================================
    # Account Information
    # =========================================================================

    def get_account(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            Account info with equity, cash, margin
        """
        ...

    # =========================================================================
    # Market Data (optional, for sanity checks)
    # =========================================================================

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get last traded price.

        Args:
            symbol: Symbol to query

        Returns:
            Last price or None
        """
        ...


# =============================================================================
# Base Implementation
# =============================================================================


class BaseBrokerConnector(ABC):
    """
    Abstract base class for broker connectors.

    Provides common functionality and enforces the protocol.
    Subclasses must implement abstract methods.
    """

    def __init__(
        self,
        credentials: BrokerCredentials,
        *,
        sandbox: bool = False,
        timeout_seconds: int = 30,
    ):
        """
        Initialize broker connector.

        Args:
            credentials: Broker credentials from Local Vault
            sandbox: Use sandbox/paper trading
            timeout_seconds: Request timeout
        """
        self._credentials = credentials
        self._sandbox = sandbox
        self._timeout = timeout_seconds
        self._connected = False
        self._last_heartbeat: Optional[datetime] = None

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def connection_status(self) -> ConnectionStatus:
        """Get connection status."""
        if self._connected:
            return ConnectionStatus.CONNECTED
        return ConnectionStatus.DISCONNECTED

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "broker": self.broker_name,
            "connected": self._connected,
            "sandbox": self._sandbox,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
        }

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect."""
        ...

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit order."""
        ...

    @abstractmethod
    def cancel_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> CancelResult:
        """Cancel order."""
        ...

    @abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None) -> BulkCancelResult:
        """Cancel all orders."""
        ...

    @abstractmethod
    def get_order(
        self,
        client_order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
    ) -> Optional[OrderInfo]:
        """Get order info."""
        ...

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """Get open orders."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get positions."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position."""
        ...

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> OrderResult:
        """Close position."""
        ...

    @abstractmethod
    def close_all_positions(self) -> List[OrderResult]:
        """Close all positions."""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account info."""
        ...

    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """Get last price (optional)."""
        return None


# =============================================================================
# Factory
# =============================================================================


class BrokerConnectorFactory:
    """
    Factory for creating broker connectors.

    Usage:
        factory = BrokerConnectorFactory()
        factory.register("alpaca", AlpacaConnector)

        connector = factory.create("alpaca", credentials, sandbox=True)
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, connector_class: type) -> None:
        """
        Register a connector class.

        Args:
            name: Broker name (lowercase)
            connector_class: Connector class
        """
        cls._registry[name.lower()] = connector_class

    @classmethod
    def create(
        cls,
        name: str,
        credentials: BrokerCredentials,
        **kwargs: Any,
    ) -> BrokerConnector:
        """
        Create a connector instance.

        Args:
            name: Broker name
            credentials: Broker credentials
            **kwargs: Additional arguments

        Returns:
            BrokerConnector instance

        Raises:
            ValueError: If broker not registered
        """
        connector_class = cls._registry.get(name.lower())
        if connector_class is None:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown broker '{name}'. Available: {available}")

        return connector_class(credentials, **kwargs)

    @classmethod
    def list_brokers(cls) -> List[str]:
        """List registered broker names."""
        return list(cls._registry.keys())
