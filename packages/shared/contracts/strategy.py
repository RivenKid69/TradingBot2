# -*- coding: utf-8 -*-
"""
Strategy Contract - Interface all strategies must implement.

This defines the boundary between strategies and the execution layer.
Strategies produce OrderIntents, which are then processed by the
appropriate execution engine (simulation or live).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .intent import OrderIntent


@dataclass
class MarketSnapshot:
    """
    Current market state provided to strategy.

    Contains only data that is safe to share:
    - Price data
    - Position info (without sensitive account details)
    - Time information
    """

    timestamp: datetime
    symbol: str
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None

    # Current position (managed by execution layer)
    position_qty: Decimal = Decimal("0")
    position_avg_price: Optional[Decimal] = None
    position_pnl: Optional[Decimal] = None

    # Additional data
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyContext:
    """
    Context provided to strategy for decision making.

    Contains all information the strategy needs to make decisions,
    but NO sensitive data like API keys or account credentials.
    """

    # Current market state
    market: MarketSnapshot

    # Strategy configuration (non-sensitive)
    config: Dict[str, Any] = field(default_factory=dict)

    # Historical data (if needed)
    history: List[MarketSnapshot] = field(default_factory=list)

    # Strategy state (persisted between calls)
    state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    run_id: Optional[str] = None
    is_live: bool = False
    is_paper: bool = False


@dataclass
class StrategyResult:
    """
    Result of strategy execution.

    Contains intents (what strategy wants to do) and optional metadata.
    """

    # Primary output: list of intents
    intents: List[OrderIntent] = field(default_factory=list)

    # Optional: updated state to persist
    new_state: Optional[Dict[str, Any]] = None

    # Telemetry data (will be redacted if sensitive)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    # Diagnostics
    execution_time_ms: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def has_intents(self) -> bool:
        """Check if result contains any intents."""
        return len(self.intents) > 0

    @property
    def has_action_intents(self) -> bool:
        """Check if result contains non-passive intents."""
        from .intent import IntentType

        return any(
            i.intent_type not in (IntentType.HOLD, IntentType.NO_ACTION) for i in self.intents
        )


@runtime_checkable
class StrategyContract(Protocol):
    """
    Protocol that all strategies must implement.

    This is the core interface between strategies and execution.
    Strategies receive context and return results with intents.
    """

    @property
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        ...

    @property
    def version(self) -> str:
        """Strategy version for tracking."""
        ...

    @property
    def symbols(self) -> List[str]:
        """List of symbols this strategy trades."""
        ...

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration.

        Called once when strategy is loaded.
        Should NOT access any external resources here.
        """
        ...

    def on_data(self, context: StrategyContext) -> StrategyResult:
        """
        Process new market data and generate intents.

        This is the main entry point called on each data tick.
        Strategy should analyze context and return desired intents.

        Args:
            context: Current market state and strategy context

        Returns:
            StrategyResult with intents and optional state updates
        """
        ...

    def on_fill(self, fill_info: Dict[str, Any]) -> None:
        """
        Called when an order is filled.

        Allows strategy to track fills and update internal state.
        """
        ...

    def shutdown(self) -> None:
        """
        Clean shutdown of strategy.

        Called when strategy is stopped. Should release any resources.
        """
        ...


class BaseStrategy(ABC):
    """
    Abstract base class for strategies.

    Provides default implementations for optional methods.
    Strategies can extend this instead of implementing full protocol.
    """

    _strategy_id: str = "base_strategy"
    _version: str = "1.0.0"
    _symbols: List[str] = []
    _config: Dict[str, Any] = {}
    _state: Dict[str, Any] = {}

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with config. Override if needed."""
        self._config = config
        self._symbols = config.get("symbols", [])

    @abstractmethod
    def on_data(self, context: StrategyContext) -> StrategyResult:
        """Must be implemented by subclass."""
        ...

    def on_fill(self, fill_info: Dict[str, Any]) -> None:
        """Default: do nothing. Override if needed."""
        pass

    def shutdown(self) -> None:
        """Default: do nothing. Override if needed."""
        pass

    def _create_hold_result(self) -> StrategyResult:
        """Helper to create a no-action result."""
        from .intent import IntentType, IntentSide, OrderIntent

        return StrategyResult(
            intents=[
                OrderIntent(
                    strategy_id=self.strategy_id,
                    symbol=self._symbols[0] if self._symbols else "UNKNOWN",
                    intent_type=IntentType.HOLD,
                    side=IntentSide.FLAT,
                    reason="No action required",
                )
            ]
        )
