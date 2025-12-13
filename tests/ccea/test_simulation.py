# -*- coding: utf-8 -*-
"""
Tests for packages/shared/simulation.

Phase 2 Implementation: Tests for simulation execution engine.
"""

from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestSimExecutionEngine:
    """Tests for SimExecutionEngine."""

    def test_engine_creation(self):
        """Test creating simulation engine."""
        from packages.shared.simulation.engine import SimExecutionEngine

        engine = SimExecutionEngine(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0.001"),
        )

        assert engine.capital == Decimal("100000")
        assert engine.commission_rate == Decimal("0.001")

    def test_process_open_intent(self):
        """Test processing open intent."""
        from packages.shared.simulation.engine import SimExecutionEngine
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        engine = SimExecutionEngine(initial_capital=Decimal("100000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
            limit_price=Decimal("150"),
        )

        fill = engine.process_intent(intent, market_price=Decimal("150"))

        assert fill is not None
        assert fill.quantity == Decimal("100")
        assert fill.price == Decimal("150")

    def test_process_close_intent(self):
        """Test processing close intent."""
        from packages.shared.simulation.engine import SimExecutionEngine
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        engine = SimExecutionEngine(initial_capital=Decimal("100000"))

        # First open a position
        open_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        engine.process_intent(open_intent, market_price=Decimal("150"))

        # Now close it
        close_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.CLOSE,
            side=IntentSide.FLAT,
        )
        fill = engine.process_intent(close_intent, market_price=Decimal("155"))

        assert fill is not None
        assert engine.get_position("AAPL") is None or engine.get_position("AAPL").quantity == 0

    def test_position_tracking(self):
        """Test position tracking."""
        from packages.shared.simulation.engine import SimExecutionEngine
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        engine = SimExecutionEngine(initial_capital=Decimal("100000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="BTCUSDT",
            intent_type=IntentType.OPEN,
            side=IntentSide.SHORT,
            target_quantity=Decimal("1"),
        )
        engine.process_intent(intent, market_price=Decimal("50000"))

        position = engine.get_position("BTCUSDT")
        assert position is not None
        assert position.quantity == Decimal("-1")  # Short
        assert position.side == IntentSide.SHORT

    def test_pnl_calculation(self):
        """Test P&L calculation."""
        from packages.shared.simulation.engine import SimExecutionEngine
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        engine = SimExecutionEngine(
            initial_capital=Decimal("100000"),
            commission_rate=Decimal("0"),  # No commission for simple test
        )

        # Open position at 150
        open_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )
        engine.process_intent(open_intent, market_price=Decimal("150"))

        # Close at 155 (profit of 5 * 100 = 500)
        close_intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.CLOSE,
            side=IntentSide.FLAT,
        )
        engine.process_intent(close_intent, market_price=Decimal("155"))

        assert engine.realized_pnl == Decimal("500")

    def test_no_live_orders(self):
        """Test that simulation never sends live orders."""
        from packages.shared.simulation.engine import SimExecutionEngine
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        engine = SimExecutionEngine(initial_capital=Decimal("100000"))

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        # Process many intents
        for _ in range(100):
            engine.process_intent(intent, market_price=Decimal("150"))

        # Verify no live orders were sent
        assert engine.live_orders_sent == 0
        assert engine.mode == "simulation"


class TestSimulatedFill:
    """Tests for SimulatedFill."""

    def test_fill_creation(self):
        """Test creating simulated fill."""
        from packages.shared.simulation.engine import SimulatedFill

        fill = SimulatedFill(
            symbol="AAPL",
            side="long",
            quantity=Decimal("100"),
            price=Decimal("150.25"),
            commission=Decimal("0.15"),
        )

        assert fill.symbol == "AAPL"
        assert fill.notional == Decimal("15025")  # 100 * 150.25

    def test_fill_with_slippage(self):
        """Test fill with slippage."""
        from packages.shared.simulation.engine import SimulatedFill

        fill = SimulatedFill(
            symbol="BTCUSDT",
            side="long",
            quantity=Decimal("1"),
            price=Decimal("50100"),  # Intended 50000, got 50100
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
            entry_price=Decimal("150"),
        )

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.notional == Decimal("15000")

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        from packages.shared.simulation.engine import SimulatedPosition
        from packages.shared.contracts.intent import IntentSide

        position = SimulatedPosition(
            symbol="AAPL",
            side=IntentSide.LONG,
            quantity=Decimal("100"),
            entry_price=Decimal("150"),
        )

        # Price went up
        unrealized = position.calculate_unrealized_pnl(current_price=Decimal("155"))
        assert unrealized == Decimal("500")  # (155 - 150) * 100

        # Price went down
        unrealized = position.calculate_unrealized_pnl(current_price=Decimal("145"))
        assert unrealized == Decimal("-500")  # (145 - 150) * 100

    def test_short_position_pnl(self):
        """Test short position P&L."""
        from packages.shared.simulation.engine import SimulatedPosition
        from packages.shared.contracts.intent import IntentSide

        position = SimulatedPosition(
            symbol="BTCUSDT",
            side=IntentSide.SHORT,
            quantity=Decimal("-1"),  # Short
            entry_price=Decimal("50000"),
        )

        # Price went down (profit for short)
        unrealized = position.calculate_unrealized_pnl(current_price=Decimal("45000"))
        assert unrealized == Decimal("5000")  # (50000 - 45000) * 1
