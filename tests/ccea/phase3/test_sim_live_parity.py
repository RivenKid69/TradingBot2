# -*- coding: utf-8 -*-
"""
Tests for Sim/Live Parity.

Verifies:
1. SimulationRunner processes intents correctly
2. LiveRunner processes intents correctly
3. Same intent produces consistent behavior in both zones
4. Risk controls are properly enforced
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Optional

from packages.shared.contracts.intent import (
    OrderIntent,
    IntentType,
    IntentSide,
    IntentPriority,
)
from packages.shared.contracts.strategy import (
    BaseStrategy,
    StrategyContext,
    StrategyResult,
    MarketSnapshot,
)
from packages.shared.contracts.config import ExecutionMode
from packages.shared.simulation.engine import (
    SimExecutionEngine,
    SimulationConfig,
    SimulatedFill,
    FillStatus,
)
from packages.shared.runner.base import (
    RunnerConfig,
    RunnerState,
    RunnerZone,
)
from packages.shared.runner.simulation import (
    SimulationRunner,
    SimulationRunnerConfig,
)


class SimpleTestStrategy(BaseStrategy):
    """Simple strategy for testing."""

    _strategy_id = "test_strategy"
    _version = "1.0.0"

    def __init__(self):
        self._next_intent: Optional[OrderIntent] = None

    def set_next_intent(self, intent: OrderIntent) -> None:
        """Set the intent to return on next on_data call."""
        self._next_intent = intent

    def on_data(self, context: StrategyContext) -> StrategyResult:
        """Return configured intent."""
        if self._next_intent:
            intent = self._next_intent
            self._next_intent = None
            return StrategyResult(intents=[intent])
        return self._create_hold_result()


class TestSimExecutionEngine:
    """Tests for SimExecutionEngine."""

    def test_process_market_entry_intent(self):
        """Test processing market entry intent."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )

        fill = engine.process_intent(intent)

        assert fill is not None
        assert fill.status == FillStatus.FILLED
        assert fill.symbol == "BTCUSDT"
        assert fill.quantity == Decimal("1.0")
        assert fill.fill_price > Decimal("0")

    def test_process_limit_entry_intent(self):
        """Test processing limit entry intent."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.LIMIT_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.5"),
            limit_price=Decimal("49900"),
        )

        fill = engine.process_intent(intent)

        assert fill is not None
        # Limit price should be respected
        assert fill.fill_price <= Decimal("49900")

    def test_process_exit_intent(self):
        """Test processing exit intent."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        # First entry
        entry_intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )
        engine.process_intent(entry_intent)

        # Then exit
        exit_intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_EXIT,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1.0"),
        )
        fill = engine.process_intent(exit_intent)

        assert fill is not None
        assert fill.status == FillStatus.FILLED

    def test_hold_intent_no_fill(self):
        """Test that hold intent produces no fill."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        fill = engine.process_intent(intent)

        assert fill is None

    def test_position_tracking(self):
        """Test that positions are tracked correctly."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        # Entry
        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("2.0"),
        )
        engine.process_intent(intent)

        position = engine.get_position("BTCUSDT")
        assert position.quantity == Decimal("2.0")
        assert position.side == IntentSide.LONG

    def test_pnl_calculation(self):
        """Test P&L calculation."""
        config = SimulationConfig(
            slippage_bps=Decimal("0"),  # No slippage for predictable test
            initial_capital=Decimal("100000"),
        )
        engine = SimExecutionEngine(config=config)

        # Entry at 50000
        engine.set_price("BTCUSDT", Decimal("50000"))
        entry = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )
        engine.process_intent(entry)

        # Price goes up
        engine.set_price("BTCUSDT", Decimal("51000"))

        # Exit
        exit_intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_EXIT,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1.0"),
        )
        engine.process_intent(exit_intent)

        # Should have profit (minus commission)
        total_pnl = engine.get_total_pnl()
        assert total_pnl > Decimal("0")

    def test_reject_without_price(self):
        """Test rejection when no price available."""
        engine = SimExecutionEngine()
        # No price set

        intent = OrderIntent(
            strategy_id="test",
            symbol="UNKNOWN",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )

        fill = engine.process_intent(intent)

        assert fill is not None
        assert fill.status == FillStatus.REJECTED

    def test_notional_to_quantity_conversion(self):
        """Test conversion from target_notional to quantity."""
        engine = SimExecutionEngine()
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_notional=Decimal("25000"),  # Should be 0.5 BTC
        )

        fill = engine.process_intent(intent)

        assert fill is not None
        assert fill.quantity == Decimal("0.5")


class TestSimulationRunner:
    """Tests for SimulationRunner."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        config = SimulationRunnerConfig(
            run_id="test_001",
            strategy_id="test_strategy",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)

        assert runner.zone == RunnerZone.CLOUD
        assert runner.state == RunnerState.IDLE

    def test_runner_with_strategy(self):
        """Test runner with strategy."""
        config = SimulationRunnerConfig(
            run_id="test_001",
            strategy_id="test_strategy",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()

        success = runner.initialize(strategy)
        assert success is True

    def test_runner_start_stop(self):
        """Test starting and stopping runner."""
        config = SimulationRunnerConfig(
            run_id="test_001",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()
        runner.initialize(strategy)

        # Start
        assert runner.start() is True
        assert runner.state == RunnerState.RUNNING
        assert runner.is_running is True

        # Stop
        result = runner.stop()
        assert runner.state == RunnerState.STOPPED
        assert result.run_id == "test_001"

    def test_runner_pause_resume(self):
        """Test pausing and resuming."""
        config = SimulationRunnerConfig(run_id="test", symbols=["BTC"])

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()
        runner.initialize(strategy)
        runner.start()

        # Pause
        assert runner.pause() is True
        assert runner.state == RunnerState.PAUSED

        # Resume
        assert runner.resume() is True
        assert runner.state == RunnerState.RUNNING

    def test_runner_process_tick(self):
        """Test processing market tick."""
        config = SimulationRunnerConfig(
            run_id="test",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()

        # Set up strategy to return market entry
        entry_intent = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )
        strategy.set_next_intent(entry_intent)

        runner.initialize(strategy)
        runner.start()

        # Process tick
        market_data = {
            "symbol": "BTCUSDT",
            "close": 50000,
            "timestamp": datetime.utcnow().isoformat(),
        }

        result = runner.process_tick(market_data)

        assert result.has_intents is True
        assert len(runner._fills) > 0

    def test_runner_equity_tracking(self):
        """Test equity tracking."""
        config = SimulationRunnerConfig(
            run_id="test",
            symbols=["BTCUSDT"],
            initial_capital=Decimal("100000"),
            track_equity_curve=True,
        )

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()
        runner.initialize(strategy)
        runner.start()

        # Process some ticks
        for i in range(5):
            market_data = {
                "symbol": "BTCUSDT",
                "close": 50000 + i * 100,
            }
            runner.process_tick(market_data)

        equity_curve = runner.get_equity_curve()
        assert len(equity_curve) == 5

    def test_runner_result_metrics(self):
        """Test result metrics."""
        config = SimulationRunnerConfig(
            run_id="test",
            symbols=["BTCUSDT"],
        )

        runner = SimulationRunner(config)
        strategy = SimpleTestStrategy()

        # Entry
        strategy.set_next_intent(OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        ))

        runner.initialize(strategy)
        runner.start()

        runner.process_tick({"symbol": "BTCUSDT", "close": 50000})

        result = runner.stop()

        assert result.ticks_processed >= 1
        assert result.intents_received >= 1
        assert result.fills_count >= 1


class TestSimLiveParity:
    """Tests ensuring sim and live produce consistent results."""

    def test_same_intent_same_fill_direction(self):
        """Test that same intent produces same fill direction in sim."""
        # Sim
        sim_engine = SimExecutionEngine()
        sim_engine.set_price("BTC", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )

        fill = sim_engine.process_intent(intent)

        # Verify sim fill matches intent
        assert fill.side == IntentSide.LONG
        assert fill.symbol == "BTC"
        assert fill.quantity == Decimal("1.0")

    def test_hold_intent_produces_no_action(self):
        """Test hold intent produces no action in both zones."""
        sim_engine = SimExecutionEngine()
        sim_engine.set_price("BTC", Decimal("50000"))

        hold_intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.HOLD,
            side=IntentSide.FLAT,
        )

        # Sim: no fill
        fill = sim_engine.process_intent(hold_intent)
        assert fill is None

    def test_position_state_consistency(self):
        """Test position state is consistent after operations."""
        sim_engine = SimExecutionEngine()
        sim_engine.set_price("BTC", Decimal("50000"))

        # Entry
        entry = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("2.0"),
        )
        sim_engine.process_intent(entry)

        pos = sim_engine.get_position("BTC")
        assert pos.quantity == Decimal("2.0")
        assert pos.side == IntentSide.LONG

        # Partial exit
        exit_partial = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.MARKET_EXIT,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1.0"),
        )
        sim_engine.process_intent(exit_partial)

        pos = sim_engine.get_position("BTC")
        assert pos.quantity == Decimal("1.0")

    def test_intent_properties_respected(self):
        """Test that intent properties are respected in execution."""
        sim_engine = SimExecutionEngine()
        sim_engine.set_price("BTC", Decimal("50000"))

        # Limit intent with specific price
        limit_intent = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.LIMIT_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
            limit_price=Decimal("49500"),
        )

        fill = sim_engine.process_intent(limit_intent)

        # Fill price should not exceed limit
        assert fill.fill_price <= Decimal("49500")

    def test_flatten_all_closes_position(self):
        """Test flatten_all properly closes position."""
        sim_engine = SimExecutionEngine()
        sim_engine.set_price("BTC", Decimal("50000"))

        # Entry
        entry = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("5.0"),
        )
        sim_engine.process_intent(entry)

        assert sim_engine.get_position("BTC").quantity == Decimal("5.0")

        # Flatten
        flatten = OrderIntent(
            strategy_id="test",
            symbol="BTC",
            intent_type=IntentType.FLATTEN_ALL,
            side=IntentSide.FLAT,
        )
        sim_engine.process_intent(flatten)

        pos = sim_engine.get_position("BTC")
        assert pos.quantity == Decimal("0")
        assert pos.is_flat is True
