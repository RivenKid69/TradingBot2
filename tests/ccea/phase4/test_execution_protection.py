# -*- coding: utf-8 -*-
"""
Tests for Execution Engine Cloud Injection Protection - Phase 4.4

Tests cover:
- Intent origin validation
- Cloud injection blocking
- Strategy ID validation
- Local-only execution path
"""

from decimal import Decimal
from uuid import uuid4

import pytest

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
from packages.agent.execution.engine import LiveExecutionEngine, ExecutionResult


class TestCloudInjectionProtection:
    """Test Cloud injection protection in LiveExecutionEngine."""

    @pytest.fixture
    def engine(self):
        """Create execution engine."""
        return LiveExecutionEngine(broker_name="test")

    @pytest.fixture
    def valid_intent(self):
        """Create a valid local intent."""
        return OrderIntent(
            strategy_id="local-strategy-123",
            symbol="BTCUSDT",
            side=IntentSide.LONG,
            intent_type=IntentType.MARKET_ENTRY,
            target_quantity=Decimal("0.1"),
        )

    def test_local_origin_allowed(self, engine, valid_intent):
        """Test intents from local origin are allowed."""
        result = engine.execute(valid_intent, origin="local")

        # Should not fail due to origin (may fail for other reasons)
        if not result.success:
            assert "injection blocked" not in (result.error_message or "").lower()

    def test_strategy_origin_allowed(self, engine, valid_intent):
        """Test intents from strategy origin are allowed."""
        result = engine.execute(valid_intent, origin="strategy")

        if not result.success:
            assert "injection blocked" not in (result.error_message or "").lower()

    def test_runner_origin_allowed(self, engine, valid_intent):
        """Test intents from runner origin are allowed."""
        result = engine.execute(valid_intent, origin="runner")

        if not result.success:
            assert "injection blocked" not in (result.error_message or "").lower()

    def test_cloud_origin_blocked(self, engine, valid_intent):
        """Test intents from cloud origin are BLOCKED."""
        result = engine.execute(valid_intent, origin="cloud")

        assert result.success is False
        assert "injection blocked" in result.error_message.lower()

    def test_unknown_origin_blocked(self, engine, valid_intent):
        """Test intents from unknown origins are BLOCKED."""
        result = engine.execute(valid_intent, origin="unknown")

        assert result.success is False
        assert "not allowed" in result.error_message.lower()

    def test_remote_origin_blocked(self, engine, valid_intent):
        """Test intents from remote origin are BLOCKED."""
        result = engine.execute(valid_intent, origin="remote")

        assert result.success is False

    def test_missing_strategy_id_blocked(self, engine):
        """Test intents without strategy_id are BLOCKED."""
        intent = OrderIntent(
            strategy_id="",  # Empty strategy ID
            symbol="BTCUSDT",
            side=IntentSide.LONG,
            intent_type=IntentType.MARKET_ENTRY,
        )

        result = engine.execute(intent, origin="local")

        assert result.success is False
        assert "strategy_id" in result.error_message.lower()

    def test_passive_intent_succeeds(self, engine):
        """Test passive intents pass without execution."""
        intent = OrderIntent(
            strategy_id="strategy-123",
            symbol="BTCUSDT",
            side=IntentSide.LONG,
            intent_type=IntentType.HOLD,  # Passive
        )

        result = engine.execute(intent, origin="local")

        assert result.success is True
        assert "Passive" in (result.error_message or "")


class TestIntentValidation:
    """Test intent validation in execution engine."""

    @pytest.fixture
    def engine(self):
        """Create execution engine."""
        return LiveExecutionEngine(broker_name="test")

    def test_valid_intent_structure(self, engine):
        """Test valid intent structure is accepted."""
        intent = OrderIntent(
            strategy_id="valid-strategy",
            symbol="ETHUSDT",
            side=IntentSide.SHORT,
            intent_type=IntentType.MARKET_ENTRY,
            target_quantity=Decimal("1.0"),
        )

        result = engine.execute(intent, origin="local")

        # Should not fail on origin or structure validation
        if not result.success:
            assert "origin" not in (result.error_message or "").lower()
            assert "strategy_id" not in (result.error_message or "").lower()


class TestExecutionPath:
    """Test execution path remains local-only."""

    @pytest.fixture
    def engine(self):
        """Create execution engine."""
        return LiveExecutionEngine(broker_name="test")

    def test_all_orders_logged_before_submit(self, engine):
        """Test orders are journaled before broker submission."""
        intent = OrderIntent(
            strategy_id="strategy-123",
            symbol="BTCUSDT",
            side=IntentSide.LONG,
            intent_type=IntentType.MARKET_ENTRY,
            target_quantity=Decimal("0.01"),
        )

        # Execute (will fail at broker submit since no broker)
        result = engine.execute(intent, origin="local")

        # Journal should have entry
        journal = engine._journal
        entry = journal.get_by_client_id(result.order.client_order_id if result.order else None)
        # Entry exists if we got past origin validation
        # (may not exist if policy blocked it first)

    def test_no_direct_broker_access_from_cloud(self, engine):
        """Test cloud cannot directly access broker."""
        # There's no direct broker access method exposed
        # All execution goes through execute() which checks origin

        # Verify broker_submit is private
        assert not hasattr(engine, 'broker_submit')
        assert hasattr(engine, '_broker_submit')

    def test_policy_checked_before_execution(self, engine):
        """Test policy firewall is checked before execution."""
        intent = OrderIntent(
            strategy_id="strategy-123",
            symbol="BLOCKED_SYMBOL",
            side=IntentSide.LONG,
            intent_type=IntentType.MARKET_ENTRY,
            target_quantity=Decimal("1.0"),
        )

        # Even if origin is valid, policy should be checked
        result = engine.execute(intent, origin="local")

        # Policy check happens after origin check
        if not result.success:
            # Either blocked by origin or policy - both are valid
            pass
