# -*- coding: utf-8 -*-
"""
Tests for packages/shared/simulation.

Phase 3 Updated: Tests for simulation execution engine aligned with actual implementation.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestSimExecutionEngine:
    """Tests for SimExecutionEngine."""

    def test_engine_creation(self):
        """Test creating simulation engine."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig

        config = SimulationConfig(
            initial_capital=Decimal("100000"),
        )
        engine = SimExecutionEngine(config=config)

        assert engine is not None
        assert engine.config.initial_capital == Decimal("100000")

    def test_process_open_intent(self):
        """Test processing open intent."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        config = SimulationConfig(initial_capital=Decimal("100000"))
        engine = SimExecutionEngine(config=config)

        # Set price
        engine.set_price("AAPL", Decimal("150"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        fill = engine.process_intent(intent)

        assert fill is not None
        assert fill.quantity == Decimal("100")

    def test_process_close_intent(self):
        """Test processing close/flatten intent."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        config = SimulationConfig(initial_capital=Decimal("100000"))
        engine = SimExecutionEngine(config=config)

        # Set price
        engine.set_price("AAPL", Decimal("150"))

        # First open a position
        open_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        engine.process_intent(open_intent)

        # Update price
        engine.set_price("AAPL", Decimal("155"))

        # Now flatten position (FLATTEN_ALL is handled by engine)
        close_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.FLATTEN_ALL,
            side=IntentSide.FLAT,
        )
        fill = engine.process_intent(close_intent)

        assert fill is not None
        position = engine.get_position("AAPL")
        assert position is None or position.is_flat

    def test_position_tracking(self):
        """Test position tracking."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        config = SimulationConfig(initial_capital=Decimal("100000"))
        engine = SimExecutionEngine(config=config)

        # Set price
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1"),
        )
        engine.process_intent(intent)

        position = engine.get_position("BTCUSDT")
        assert position is not None
        assert position.side == IntentSide.SHORT

    def test_pnl_calculation(self):
        """Test P&L calculation."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        config = SimulationConfig(
            initial_capital=Decimal("100000"),
            slippage_bps=Decimal("0"),  # No slippage for simple test
        )
        engine = SimExecutionEngine(config=config)

        # Set price and open position
        engine.set_price("AAPL", Decimal("150"))

        open_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        engine.process_intent(open_intent)

        # Update price (profit scenario)
        engine.set_price("AAPL", Decimal("155"))

        # Flatten position
        close_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.FLATTEN_ALL,
            side=IntentSide.FLAT,
        )
        engine.process_intent(close_intent)

        # Check realized P&L
        equity = engine.get_equity()
        assert equity > config.initial_capital  # Made profit

    def test_no_live_orders(self):
        """Test that simulation never sends live orders."""
        from packages.shared.simulation.engine import SimExecutionEngine, SimulationConfig
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        config = SimulationConfig(initial_capital=Decimal("100000"))
        engine = SimExecutionEngine(config=config)

        # Set price
        engine.set_price("AAPL", Decimal("150"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        # Process many intents
        for _ in range(10):
            engine.process_intent(intent)

        # Simulation engine should not have any live order methods
        assert not hasattr(engine, 'submit_live_order')


class TestSimulatedFill:
    """Tests for SimulatedFill."""

    def test_fill_creation(self):
        """Test creating simulated fill."""
        from packages.shared.simulation.engine import SimulatedFill
        from packages.shared.contracts.intent import IntentSide

        fill = SimulatedFill(
            symbol="AAPL",
            side=IntentSide.LONG,
            quantity=Decimal("100"),
            fill_price=Decimal("150.25"),
            commission=Decimal("0.15"),
        )

        assert fill.symbol == "AAPL"
        assert fill.notional == Decimal("15025")  # 100 * 150.25

    def test_fill_with_slippage(self):
        """Test fill with slippage."""
        from packages.shared.simulation.engine import SimulatedFill
        from packages.shared.contracts.intent import IntentSide

        fill = SimulatedFill(
            symbol="BTCUSDT",
            side=IntentSide.LONG,
            quantity=Decimal("1"),
            fill_price=Decimal("50100"),  # Intended 50000, got 50100
            commission=Decimal("50"),
            slippage=Decimal("100"),
        )

        assert fill.slippage == Decimal("100")


class TestSimulatedPosition:
    """Tests for SimulatedPosition."""

    def test_position_creation(self):
        """Test creating simulated position."""
        from packages.shared.simulation.engine import SimulatedPosition
        from packages.shared.contracts.intent import IntentSide

        position = SimulatedPosition(
            symbol="AAPL",
            side=IntentSide.LONG,
            quantity=Decimal("100"),
            avg_price=Decimal("150"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        from packages.shared.simulation.engine import SimulatedPosition
        from packages.shared.contracts.intent import IntentSide

        position = SimulatedPosition(
            symbol="AAPL",
            side=IntentSide.LONG,
            quantity=Decimal("100"),
            avg_price=Decimal("150"),
        )

        # Price went up
        position.update_unrealized_pnl(current_price=Decimal("155"))
        assert position.unrealized_pnl == Decimal("500")  # (155 - 150) * 100

        # Price went down
        position.update_unrealized_pnl(current_price=Decimal("145"))
        assert position.unrealized_pnl == Decimal("-500")  # (145 - 150) * 100

    def test_short_position_pnl(self):
        """Test short position P&L."""
        from packages.shared.simulation.engine import SimulatedPosition
        from packages.shared.contracts.intent import IntentSide

        position = SimulatedPosition(
            symbol="BTCUSDT",
            side=IntentSide.SHORT,
            quantity=Decimal("-1"),  # Short
            avg_price=Decimal("50000"),
        )

        # Price went down (profit for short)
        position.update_unrealized_pnl(current_price=Decimal("45000"))
        assert position.unrealized_pnl == Decimal("5000")  # (50000 - 45000) * 1
