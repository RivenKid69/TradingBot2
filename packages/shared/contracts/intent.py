# -*- coding: utf-8 -*-
"""
Order Intent - Strategy's expression of desired action.

OrderIntent represents what a strategy WANTS to do, not an actual order.
It is the ONLY output format strategies should produce.

Cloud Zone: Can process intents for simulation/backtest
Agent Zone: Converts intents to actual orders via execution engine

IMPORTANT: This is NOT an order. Orders are created ONLY in the Agent zone
by the execution engine after applying risk controls and policy firewall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional
from uuid import UUID, uuid4


class IntentType(str, Enum):
    """Type of intent action."""

    # Market orders
    MARKET_ENTRY = "market_entry"
    MARKET_EXIT = "market_exit"

    # Limit orders
    LIMIT_ENTRY = "limit_entry"
    LIMIT_EXIT = "limit_exit"

    # Stop orders
    STOP_ENTRY = "stop_entry"
    STOP_EXIT = "stop_exit"

    # Position management
    ADJUST_POSITION = "adjust_position"
    CLOSE_POSITION = "close_position"
    FLATTEN_ALL = "flatten_all"

    # Passive / no action
    HOLD = "hold"
    NO_ACTION = "no_action"


class IntentSide(str, Enum):
    """Side of the intent."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # For close/flatten


class IntentPriority(str, Enum):
    """Priority level for intent processing."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"  # For risk-related actions


@dataclass(frozen=True)
class OrderIntent:
    """
    Strategy's expression of desired trading action.

    This is the ONLY format strategies should use to express trading decisions.
    OrderIntent is then processed differently depending on the zone:

    - Cloud (simulation): SimExecutionEngine processes intent
    - Agent (live): RiskManager validates -> ExecutionEngine creates actual orders

    Attributes:
        intent_id: Unique identifier for this intent
        strategy_id: ID of strategy that generated this intent
        symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
        intent_type: Type of action (market_entry, limit_exit, etc.)
        side: Long, short, or flat
        target_quantity: Desired quantity (contracts, shares, etc.)
        target_notional: Alternative: desired notional value
        limit_price: Price for limit orders (optional)
        stop_price: Price for stop orders (optional)
        time_in_force: Order duration (GTC, IOC, FOK, DAY)
        urgency: Priority level for processing
        reason: Human-readable explanation
        metadata: Additional strategy-specific data
        created_at: Timestamp of creation
        expires_at: Optional expiration time

    Example:
        >>> intent = OrderIntent(
        ...     strategy_id="momentum_btc",
        ...     symbol="BTCUSDT",
        ...     intent_type=IntentType.MARKET_ENTRY,
        ...     side=IntentSide.LONG,
        ...     target_quantity=Decimal("0.1"),
        ...     reason="Momentum breakout signal",
        ... )
    """

    # Required fields
    strategy_id: str
    symbol: str
    intent_type: IntentType
    side: IntentSide

    # Quantity specification (at least one should be set for non-HOLD intents)
    target_quantity: Optional[Decimal] = None
    target_notional: Optional[Decimal] = None

    # Price constraints (for limit/stop orders)
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None

    # Order behavior
    time_in_force: str = "GTC"
    urgency: IntentPriority = IntentPriority.NORMAL

    # Documentation
    reason: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Unique identifier (generated)
    intent_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        """Validate intent after creation."""
        # Validate quantity for non-hold intents
        if self.intent_type not in (IntentType.HOLD, IntentType.NO_ACTION):
            if self.target_quantity is None and self.target_notional is None:
                # Allow for FLATTEN_ALL which doesn't need quantity
                if self.intent_type != IntentType.FLATTEN_ALL:
                    pass  # Validation can be done at processing time

        # Validate limit price for limit orders
        if self.intent_type in (IntentType.LIMIT_ENTRY, IntentType.LIMIT_EXIT):
            if self.limit_price is None:
                pass  # Can be set later or use market

        # Validate stop price for stop orders
        if self.intent_type in (IntentType.STOP_ENTRY, IntentType.STOP_EXIT):
            if self.stop_price is None:
                pass  # Can be set later

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry intent."""
        return self.intent_type in (
            IntentType.MARKET_ENTRY,
            IntentType.LIMIT_ENTRY,
            IntentType.STOP_ENTRY,
        )

    @property
    def is_exit(self) -> bool:
        """Check if this is an exit intent."""
        return self.intent_type in (
            IntentType.MARKET_EXIT,
            IntentType.LIMIT_EXIT,
            IntentType.STOP_EXIT,
            IntentType.CLOSE_POSITION,
            IntentType.FLATTEN_ALL,
        )

    @property
    def is_limit(self) -> bool:
        """Check if this requires a limit price."""
        return self.intent_type in (
            IntentType.LIMIT_ENTRY,
            IntentType.LIMIT_EXIT,
        )

    @property
    def is_stop(self) -> bool:
        """Check if this requires a stop price."""
        return self.intent_type in (
            IntentType.STOP_ENTRY,
            IntentType.STOP_EXIT,
        )

    @property
    def is_passive(self) -> bool:
        """Check if this is a passive/no-action intent."""
        return self.intent_type in (
            IntentType.HOLD,
            IntentType.NO_ACTION,
        )

    @property
    def is_expired(self) -> bool:
        """Check if intent has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent_id": str(self.intent_id),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "intent_type": self.intent_type.value,
            "side": self.side.value,
            "target_quantity": str(self.target_quantity) if self.target_quantity else None,
            "target_notional": str(self.target_notional) if self.target_notional else None,
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force,
            "urgency": self.urgency.value,
            "reason": self.reason,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderIntent:
        """Create from dictionary."""
        return cls(
            intent_id=UUID(data["intent_id"]) if "intent_id" in data else uuid4(),
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            intent_type=IntentType(data["intent_type"]),
            side=IntentSide(data["side"]),
            target_quantity=(
                Decimal(data["target_quantity"]) if data.get("target_quantity") else None
            ),
            target_notional=(
                Decimal(data["target_notional"]) if data.get("target_notional") else None
            ),
            limit_price=Decimal(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            time_in_force=data.get("time_in_force", "GTC"),
            urgency=IntentPriority(data.get("urgency", "normal")),
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.utcnow()
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
        )


# Type hints for intent lists
IntentList = List[OrderIntent]

# Constants for common intents
HOLD_INTENT: Final[IntentType] = IntentType.HOLD
NO_ACTION_INTENT: Final[IntentType] = IntentType.NO_ACTION
