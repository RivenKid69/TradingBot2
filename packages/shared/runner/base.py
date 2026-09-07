# -*- coding: utf-8 -*-
"""
Base Runner - Abstract base class for strategy runners.

Defines the common interface for all runners (Cloud simulation, Agent live).
Runners process strategies and route OrderIntents to appropriate execution engines.

IMPORTANT:
    - Runners receive OrderIntents from strategies
    - Cloud runners process intents via SimExecutionEngine
    - Agent runners process intents via LiveExecutionEngine with risk controls
    - Orders are NEVER created in Cloud zone
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from packages.shared.contracts.intent import OrderIntent, IntentList
from packages.shared.contracts.strategy import (
    StrategyContract,
    StrategyContext,
    StrategyResult,
    MarketSnapshot,
)
from packages.shared.contracts.config import StrategyConfig, ExecutionMode


class RunnerState(str, Enum):
    """Runner execution state."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class RunnerZone(str, Enum):
    """Runner zone classification."""

    CLOUD = "cloud"  # Simulation only
    AGENT = "agent"  # Live execution


@dataclass
class RunnerConfig:
    """
    Configuration for strategy runner.

    Zone-agnostic configuration that is processed differently
    depending on whether runner is in Cloud or Agent zone.
    """

    # Identity
    run_id: str = ""
    strategy_id: str = ""

    # Mode
    mode: ExecutionMode = ExecutionMode.PAPER
    zone: RunnerZone = RunnerZone.CLOUD

    # Symbols
    symbols: List[str] = field(default_factory=list)

    # Timing
    tick_interval_ms: int = 1000
    max_run_duration_seconds: Optional[int] = None

    # Capital (for simulation)
    initial_capital: Decimal = Decimal("100000")

    # Callbacks
    on_fill_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    on_error_callback: Optional[Callable[[str], None]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "strategy_id": self.strategy_id,
            "mode": self.mode.value,
            "zone": self.zone.value,
            "symbols": self.symbols,
            "tick_interval_ms": self.tick_interval_ms,
            "max_run_duration_seconds": self.max_run_duration_seconds,
            "initial_capital": str(self.initial_capital),
        }


@dataclass
class IntentProcessingResult:
    """Result of processing an OrderIntent."""

    intent_id: str
    processed: bool = False
    fill_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_id": self.intent_id,
            "processed": self.processed,
            "fill_info": self.fill_info,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RunnerResult:
    """
    Result of runner execution.

    Contains execution summary and all processed intents.
    """

    run_id: str = ""
    success: bool = False
    state: RunnerState = RunnerState.STOPPED

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: int = 0

    # Processing stats
    ticks_processed: int = 0
    intents_received: int = 0
    intents_processed: int = 0
    fills_count: int = 0

    # Financial
    initial_capital: Decimal = Decimal("0")
    final_equity: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    # Errors
    errors: List[str] = field(default_factory=list)

    # Processed intents
    intent_results: List[IntentProcessingResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "success": self.success,
            "state": self.state.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "ticks_processed": self.ticks_processed,
            "intents_received": self.intents_received,
            "intents_processed": self.intents_processed,
            "fills_count": self.fills_count,
            "initial_capital": str(self.initial_capital),
            "final_equity": str(self.final_equity),
            "total_pnl": str(self.total_pnl),
            "errors": self.errors,
        }


class BaseRunner(ABC):
    """
    Abstract base class for strategy runners.

    Defines the interface that all runners must implement.
    Runners process strategies and route OrderIntents appropriately.
    """

    def __init__(self, config: RunnerConfig):
        """
        Initialize runner.

        Args:
            config: Runner configuration
        """
        self._config = config
        self._state = RunnerState.IDLE
        self._strategy: Optional[StrategyContract] = None
        self._result = RunnerResult(
            run_id=config.run_id,
            initial_capital=config.initial_capital,
        )

    @property
    def state(self) -> RunnerState:
        """Get current runner state."""
        return self._state

    @property
    def zone(self) -> RunnerZone:
        """Get runner zone."""
        return self._config.zone

    @property
    def is_running(self) -> bool:
        """Check if runner is currently running."""
        return self._state == RunnerState.RUNNING

    @abstractmethod
    def initialize(self, strategy: StrategyContract) -> bool:
        """
        Initialize runner with strategy.

        Args:
            strategy: Strategy to run

        Returns:
            True if initialization successful
        """
        ...

    @abstractmethod
    def process_tick(self, market_data: Dict[str, Any]) -> StrategyResult:
        """
        Process a single market data tick.

        Args:
            market_data: Current market data

        Returns:
            Strategy result with intents
        """
        ...

    @abstractmethod
    def process_intents(self, intents: IntentList) -> List[IntentProcessingResult]:
        """
        Process OrderIntents from strategy.

        Args:
            intents: List of OrderIntents to process

        Returns:
            List of processing results
        """
        ...

    @abstractmethod
    def start(self) -> bool:
        """
        Start the runner.

        Returns:
            True if started successfully
        """
        ...

    @abstractmethod
    def stop(self, reason: str = "user_requested") -> RunnerResult:
        """
        Stop the runner.

        Args:
            reason: Reason for stopping

        Returns:
            Final runner result
        """
        ...

    @abstractmethod
    def pause(self) -> bool:
        """
        Pause the runner.

        Returns:
            True if paused successfully
        """
        ...

    @abstractmethod
    def resume(self) -> bool:
        """
        Resume paused runner.

        Returns:
            True if resumed successfully
        """
        ...

    def get_result(self) -> RunnerResult:
        """Get current result."""
        return self._result

    def _create_context(self, market_data: Dict[str, Any]) -> StrategyContext:
        """
        Create strategy context from market data.

        Args:
            market_data: Market data dictionary

        Returns:
            StrategyContext for strategy
        """
        timestamp = market_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()

        symbol = market_data.get(
            "symbol", self._config.symbols[0] if self._config.symbols else "UNKNOWN"
        )

        snapshot = MarketSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bid=Decimal(str(market_data.get("bid", 0))) if market_data.get("bid") else None,
            ask=Decimal(str(market_data.get("ask", 0))) if market_data.get("ask") else None,
            last=Decimal(str(market_data.get("close", market_data.get("last", 0)))),
            volume_24h=(
                Decimal(str(market_data.get("volume", 0))) if market_data.get("volume") else None
            ),
            position_qty=Decimal(str(market_data.get("position_qty", 0))),
            position_avg_price=(
                Decimal(str(market_data.get("position_avg_price", 0)))
                if market_data.get("position_avg_price")
                else None
            ),
            features=market_data.get("features", {}),
        )

        return StrategyContext(
            market=snapshot,
            config=self._config.to_dict(),
            run_id=self._config.run_id,
            is_live=self._config.mode == ExecutionMode.LIVE,
            is_paper=self._config.mode == ExecutionMode.PAPER,
        )
