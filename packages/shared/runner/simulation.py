# -*- coding: utf-8 -*-
"""
Simulation Runner - CLOUD ZONE ONLY.

Runs strategies in simulation mode using SimExecutionEngine.
This runner NEVER executes real trades - it simulates them.

Key Features:
- Processes OrderIntents via SimExecutionEngine
- Tracks simulated positions and P&L
- Generates simulated fills
- Safe for Cloud zone deployment

IMPORTANT:
    This module is PROHIBITED from:
    - Creating real orders
    - Connecting to brokers
    - Accessing trading APIs
    - Sending any order-like payloads
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from packages.shared.contracts.intent import OrderIntent, IntentList, IntentType
from packages.shared.contracts.strategy import (
    StrategyContract,
    StrategyContext,
    StrategyResult,
    MarketSnapshot,
)
from packages.shared.contracts.config import ExecutionMode
from packages.shared.simulation.engine import (
    SimExecutionEngine,
    SimulationConfig,
    SimulatedFill,
)
from packages.shared.runner.base import (
    BaseRunner,
    RunnerConfig,
    RunnerResult,
    RunnerState,
    RunnerZone,
    IntentProcessingResult,
)


@dataclass
class SimulationRunnerConfig(RunnerConfig):
    """
    Configuration specific to simulation runner.

    Extends base config with simulation-specific settings.
    """

    # Simulation settings
    slippage_bps: Decimal = Decimal("5")
    fill_probability: float = 0.98
    enable_market_impact: bool = True

    # Position tracking
    track_positions: bool = True
    track_equity_curve: bool = True

    def __post_init__(self):
        """Ensure zone is CLOUD."""
        self.zone = RunnerZone.CLOUD

    def to_sim_config(self) -> SimulationConfig:
        """Convert to SimulationConfig."""
        return SimulationConfig(
            slippage_bps=self.slippage_bps,
            fill_probability=self.fill_probability,
            enable_market_impact=self.enable_market_impact,
            initial_capital=self.initial_capital,
        )


class SimulationRunner(BaseRunner):
    """
    Simulation Runner for Cloud Zone.

    Processes strategies using SimExecutionEngine.
    NEVER executes real trades.

    Usage:
        config = SimulationRunnerConfig(
            run_id="sim_001",
            strategy_id="momentum",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)
        runner.initialize(strategy)
        runner.start()

        # Process market data
        for tick in market_data:
            result = runner.process_tick(tick)

        final_result = runner.stop()
    """

    def __init__(self, config: SimulationRunnerConfig):
        """
        Initialize simulation runner.

        Args:
            config: Simulation runner configuration
        """
        # Ensure zone is CLOUD
        config.zone = RunnerZone.CLOUD

        super().__init__(config)
        self._sim_config = (
            config
            if isinstance(config, SimulationRunnerConfig)
            else SimulationRunnerConfig(**config.to_dict())
        )

        # Initialize simulation engine
        self._engine = SimExecutionEngine(config=self._sim_config.to_sim_config())

        # Strategy
        self._strategy: Optional[StrategyContract] = None

        # Tracking
        self._fills: List[SimulatedFill] = []
        self._equity_curve: List[Dict[str, Any]] = []
        self._tick_count = 0

    def initialize(self, strategy: StrategyContract) -> bool:
        """
        Initialize runner with strategy.

        Args:
            strategy: Strategy to run

        Returns:
            True if initialization successful
        """
        try:
            self._state = RunnerState.INITIALIZING

            # Store strategy
            self._strategy = strategy

            # Initialize strategy
            strategy.initialize(
                {
                    "symbols": self._config.symbols,
                    "mode": self._config.mode.value,
                    "run_id": self._config.run_id,
                }
            )

            # Reset engine
            self._engine.reset()

            # Update result
            self._result.start_time = datetime.utcnow()
            self._result.initial_capital = self._config.initial_capital

            self._state = RunnerState.IDLE
            return True

        except Exception as e:
            self._state = RunnerState.ERROR
            self._result.errors.append(f"Initialization failed: {str(e)}")
            return False

    def process_tick(self, market_data: Dict[str, Any]) -> StrategyResult:
        """
        Process a single market data tick.

        Args:
            market_data: Current market data

        Returns:
            Strategy result with intents
        """
        if self._state != RunnerState.RUNNING:
            return StrategyResult()

        if self._strategy is None:
            return StrategyResult()

        self._tick_count += 1

        # Update price in engine
        symbol = market_data.get(
            "symbol", self._config.symbols[0] if self._config.symbols else "UNKNOWN"
        )
        price = Decimal(str(market_data.get("close", market_data.get("last", 0))))
        self._engine.set_price(symbol, price)

        # Get position info
        position = self._engine.get_position(symbol)

        # Update market data with position info
        market_data["position_qty"] = str(position.quantity)
        market_data["position_avg_price"] = str(position.avg_price) if position.avg_price else None

        # Create context
        context = self._create_context(market_data)

        # Get strategy decision
        start_time = time.time()
        strategy_result = self._strategy.on_data(context)
        execution_time_ms = (time.time() - start_time) * 1000

        # Update telemetry
        strategy_result.execution_time_ms = execution_time_ms

        # Process intents
        if strategy_result.has_intents:
            self._result.intents_received += len(strategy_result.intents)
            intent_results = self.process_intents(strategy_result.intents)
            self._result.intent_results.extend(intent_results)

        # Track equity if enabled
        if self._sim_config.track_equity_curve:
            self._equity_curve.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "tick": self._tick_count,
                    "equity": str(self._engine.get_equity()),
                    "total_pnl": str(self._engine.get_total_pnl()),
                }
            )

        # Update stats
        self._result.ticks_processed = self._tick_count

        return strategy_result

    def process_intents(self, intents: IntentList) -> List[IntentProcessingResult]:
        """
        Process OrderIntents via SimExecutionEngine.

        Args:
            intents: List of OrderIntents to process

        Returns:
            List of processing results
        """
        results = []

        for intent in intents:
            result = IntentProcessingResult(intent_id=str(intent.intent_id))

            try:
                # Skip passive intents
                if intent.is_passive:
                    result.processed = True
                    result.fill_info = {"status": "skipped", "reason": "passive_intent"}
                    results.append(result)
                    continue

                # Process intent through simulation engine
                fill = self._engine.process_intent(intent)

                if fill:
                    self._fills.append(fill)
                    result.processed = True
                    result.fill_info = fill.to_dict()
                    self._result.fills_count += 1
                    self._result.intents_processed += 1

                    # Notify strategy of fill
                    if self._strategy:
                        self._strategy.on_fill(fill.to_dict())

                    # Callback if provided
                    if self._config.on_fill_callback:
                        self._config.on_fill_callback(fill.to_dict())
                else:
                    result.processed = True
                    result.fill_info = {"status": "no_fill"}

            except Exception as e:
                result.error = str(e)
                self._result.errors.append(f"Intent processing error: {str(e)}")

                # Error callback if provided
                if self._config.on_error_callback:
                    self._config.on_error_callback(str(e))

            results.append(result)

        return results

    def start(self) -> bool:
        """
        Start the runner.

        Returns:
            True if started successfully
        """
        if self._state not in (RunnerState.IDLE, RunnerState.STOPPED):
            return False

        if self._strategy is None:
            self._result.errors.append("Cannot start: no strategy initialized")
            return False

        self._state = RunnerState.RUNNING
        self._result.start_time = datetime.utcnow()
        return True

    def stop(self, reason: str = "user_requested") -> RunnerResult:
        """
        Stop the runner.

        Args:
            reason: Reason for stopping

        Returns:
            Final runner result
        """
        if self._state == RunnerState.STOPPED:
            return self._result

        self._state = RunnerState.STOPPING

        # Shutdown strategy
        if self._strategy:
            try:
                self._strategy.shutdown()
            except Exception as e:
                self._result.errors.append(f"Strategy shutdown error: {str(e)}")

        # Finalize result
        self._result.end_time = datetime.utcnow()
        self._result.duration_ms = int(
            (self._result.end_time - self._result.start_time).total_seconds() * 1000
            if self._result.start_time
            else 0
        )
        self._result.final_equity = self._engine.get_equity()
        self._result.total_pnl = self._engine.get_total_pnl()
        self._result.success = len(self._result.errors) == 0
        self._result.state = RunnerState.STOPPED

        self._state = RunnerState.STOPPED
        return self._result

    def pause(self) -> bool:
        """
        Pause the runner.

        Returns:
            True if paused successfully
        """
        if self._state != RunnerState.RUNNING:
            return False

        self._state = RunnerState.PAUSED
        return True

    def resume(self) -> bool:
        """
        Resume paused runner.

        Returns:
            True if resumed successfully
        """
        if self._state != RunnerState.PAUSED:
            return False

        self._state = RunnerState.RUNNING
        return True

    def get_equity(self) -> Decimal:
        """Get current equity."""
        return self._engine.get_equity()

    def get_total_pnl(self) -> Decimal:
        """Get total P&L."""
        return self._engine.get_total_pnl()

    def get_fills(self) -> List[SimulatedFill]:
        """Get all simulated fills."""
        return list(self._fills)

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve."""
        return list(self._equity_curve)

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return {symbol: pos.to_dict() for symbol, pos in self._engine.get_positions().items()}
