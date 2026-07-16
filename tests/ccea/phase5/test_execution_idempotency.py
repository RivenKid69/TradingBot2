# -*- coding: utf-8 -*-
"""
Tests for Execution Engine Idempotency.

Design Doc Phase 8 WI-AGENT-06:
- client_order_id must be deterministic from broker_name|strategy_id|intent_id
- Same intent_id always produces same client_order_id (broker-level dedup)
- deployment_id/run_id/sequence stored in journal metadata for audit
- Sequence tracks order count per deployment/run for monitoring
"""

import pytest
import tempfile
from pathlib import Path
from decimal import Decimal
from uuid import uuid4

from packages.agent.execution.engine import LiveExecutionEngine, OrderStatus
from packages.agent.reconciliation.journal import OrderJournal, JournalStatus
from packages.shared.contracts.intent import (
    OrderIntent,
    IntentType,
    IntentSide,
)


class TestExecutionEngineIdempotency:
    """Tests for idempotency with deployment_id/run_id/sequence."""

    @pytest.fixture
    def temp_journal_path(self):
        """Create temporary path for journal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_journal.db"

    @pytest.fixture
    def journal(self, temp_journal_path):
        """Create test journal."""
        journal = OrderJournal(db_path=temp_journal_path)
        yield journal
        journal.close()

    @pytest.fixture
    def intent(self):
        """Create test intent."""
        return OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.1"),
        )

    def test_engine_accepts_deployment_and_run_ids(self, journal):
        """Test engine accepts deployment_id and run_id in constructor."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        assert engine.deployment_id == "deploy_123"
        assert engine.run_id == "run_456"

    def test_engine_generates_ids_if_not_provided(self, journal):
        """Test engine generates IDs if not provided."""
        engine = LiveExecutionEngine(order_journal=journal)

        assert engine.deployment_id is not None
        assert len(engine.deployment_id) > 0
        assert engine.run_id is not None
        assert len(engine.run_id) > 0

    def test_sequence_starts_at_zero(self, journal):
        """Test sequence starts at zero for new engine."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        assert engine.sequence == 0

    def test_sequence_increments_on_execute(self, journal, intent):
        """Test sequence increments with each execute call."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        assert engine.sequence == 0

        # First execute
        engine.execute(intent, origin="local")
        assert engine.sequence == 1

        # Second execute with different intent
        intent2 = OrderIntent(
            strategy_id="test_strategy",
            symbol="ETHUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )
        engine.execute(intent2, origin="local")
        assert engine.sequence == 2

    def test_same_deployment_run_produces_unique_sequential_ids(self, temp_journal_path, intent):
        """Test same deployment/run with sequence produces unique but sequential IDs."""
        # Create first engine
        journal1 = OrderJournal(db_path=temp_journal_path)
        engine1 = LiveExecutionEngine(
            order_journal=journal1,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        result1 = engine1.execute(intent, origin="local")
        client_id_1 = result1.order.client_order_id

        # Create second engine with same IDs (simulating restart)
        journal2 = OrderJournal(db_path=temp_journal_path)
        engine2 = LiveExecutionEngine(
            order_journal=journal2,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        # Sequence should recover from journal
        assert engine2.sequence == 1

        # Create new intent with same params - gets new sequence -> new ID
        intent2 = OrderIntent(
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            intent_type=intent.intent_type,
            side=intent.side,
            target_quantity=intent.target_quantity,
        )
        result2 = engine2.execute(intent2, origin="local")

        # Should succeed with NEW client_order_id (sequence=2)
        assert result2.success
        assert result2.order.client_order_id != client_id_1
        assert engine2.sequence == 2

    def test_same_intent_produces_same_client_order_id(self, temp_journal_path, intent):
        """Test same intent_id produces same client_order_id (broker-level dedup).

        Design: client_order_id = hash(broker_name|strategy_id|intent_id)
        Same intent_id should always produce the same client_order_id.
        The broker handles duplicate rejection based on client_order_id.
        """
        journal = OrderJournal(db_path=temp_journal_path)
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        # First execute
        result1 = engine.execute(intent, origin="local")
        assert result1.success
        client_id_1 = result1.order.client_order_id

        # Execute same intent again (same intent_id)
        # In real scenario, broker would reject as duplicate
        result2 = engine.execute(intent, origin="local")
        client_id_2 = result2.order.client_order_id

        # Same intent_id -> same client_order_id (deterministic)
        assert client_id_1 == client_id_2

    def test_same_intent_across_runs_produces_same_id(self, temp_journal_path, intent):
        """Test same intent produces same client_order_id across different runs.

        Design: run_id is for audit/tracking only, NOT part of client_order_id.
        Same intent_id should produce same client_order_id regardless of run.
        This enables broker-level duplicate detection across restarts.
        """
        # First run
        journal1 = OrderJournal(db_path=temp_journal_path)
        engine1 = LiveExecutionEngine(
            order_journal=journal1,
            deployment_id="deploy_123",
            run_id="run_001",
        )

        result1 = engine1.execute(intent, origin="local")
        client_id_1 = result1.order.client_order_id

        # Second run with different run_id (but same intent)
        journal2 = OrderJournal(db_path=temp_journal_path)
        engine2 = LiveExecutionEngine(
            order_journal=journal2,
            deployment_id="deploy_123",
            run_id="run_002",
        )

        result2 = engine2.execute(intent, origin="local")
        client_id_2 = result2.order.client_order_id

        # Same intent_id -> same client_order_id (run_id doesn't affect it)
        assert client_id_1 == client_id_2

    def test_sequence_recovery_on_restart(self, temp_journal_path, intent):
        """Test sequence is recovered from journal on restart."""
        # First engine - execute multiple orders
        journal1 = OrderJournal(db_path=temp_journal_path)
        engine1 = LiveExecutionEngine(
            order_journal=journal1,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        engine1.execute(intent, origin="local")

        intent2 = OrderIntent(
            strategy_id="test_strategy",
            symbol="ETHUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("1.0"),
        )
        engine1.execute(intent2, origin="local")

        intent3 = OrderIntent(
            strategy_id="test_strategy",
            symbol="BNBUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("10.0"),
        )
        engine1.execute(intent3, origin="local")

        assert engine1.sequence == 3

        # New engine (restart) - should recover sequence
        journal2 = OrderJournal(db_path=temp_journal_path)
        engine2 = LiveExecutionEngine(
            order_journal=journal2,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        # Sequence should be recovered from journal
        assert engine2.sequence == 3

    def test_get_idempotency_state(self, journal):
        """Test get_idempotency_state returns correct info."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
            broker_name="alpaca",
        )

        state = engine.get_idempotency_state()

        assert state["deployment_id"] == "deploy_123"
        assert state["run_id"] == "run_456"
        assert state["sequence"] == 0
        assert state["broker_name"] == "alpaca"

    def test_set_run_id_resets_sequence(self, journal):
        """Test set_run_id resets sequence to 0."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_001",
        )

        # Manually increment sequence (simulating orders)
        engine._sequence = 5

        # Set new run_id
        engine.set_run_id("run_002")

        assert engine.run_id == "run_002"
        assert engine.sequence == 0

    def test_journal_metadata_includes_idempotency_fields(self, temp_journal_path, intent):
        """Test journal entries include deployment_id, run_id, sequence."""
        journal = OrderJournal(db_path=temp_journal_path)
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        result = engine.execute(intent, origin="local")

        # Get journal entry
        entry = journal.get_by_client_id(result.order.client_order_id)

        assert entry is not None
        assert entry.metadata.get("deployment_id") == "deploy_123"
        assert entry.metadata.get("run_id") == "run_456"
        assert entry.metadata.get("sequence") == 1

    def test_client_order_id_format(self, journal, intent):
        """Test client_order_id has expected format."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        result = engine.execute(intent, origin="local")

        # Should start with ccea_
        assert result.order.client_order_id.startswith("ccea_")
        # Should be ccea_ + 24 char hash
        assert len(result.order.client_order_id) == 29  # "ccea_" (5) + 24 chars

    def test_different_strategy_produces_different_ids(self, journal):
        """Test different strategy_id produces different client_order_ids."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        intent1 = OrderIntent(
            strategy_id="strategy_A",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.1"),
        )

        intent2 = OrderIntent(
            strategy_id="strategy_B",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.1"),
        )

        result1 = engine.execute(intent1, origin="local")
        result2 = engine.execute(intent2, origin="local")

        assert result1.order.client_order_id != result2.order.client_order_id

    def test_different_symbol_produces_different_ids(self, journal):
        """Test different symbol produces different client_order_ids."""
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        intent1 = OrderIntent(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.1"),
        )

        intent2 = OrderIntent(
            strategy_id="test_strategy",
            symbol="ETHUSDT",
            intent_type=IntentType.MARKET_ENTRY,
            side=IntentSide.LONG,
            target_quantity=Decimal("0.1"),
        )

        result1 = engine.execute(intent1, origin="local")
        result2 = engine.execute(intent2, origin="local")

        assert result1.order.client_order_id != result2.order.client_order_id


class TestJournalGetAllEntries:
    """Tests for OrderJournal.get_all_entries method."""

    @pytest.fixture
    def temp_journal_path(self):
        """Create temporary path for journal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_journal.db"

    @pytest.fixture
    def journal(self, temp_journal_path):
        """Create test journal."""
        journal = OrderJournal(db_path=temp_journal_path)
        yield journal
        journal.close()

    def test_get_all_entries_empty(self, journal):
        """Test get_all_entries on empty journal."""
        entries = journal.get_all_entries()
        assert entries == []

    def test_get_all_entries_returns_all(self, journal):
        """Test get_all_entries returns all entries."""
        # Log multiple orders
        journal.log_order(
            client_order_id="order_1",
            intent_id="intent_1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
        )

        journal.log_order(
            client_order_id="order_2",
            intent_id="intent_2",
            symbol="ETHUSDT",
            side="sell",
            quantity=Decimal("1.0"),
            order_type="limit",
        )

        journal.log_order(
            client_order_id="order_3",
            intent_id="intent_3",
            symbol="BNBUSDT",
            side="buy",
            quantity=Decimal("10.0"),
            order_type="market",
        )

        entries = journal.get_all_entries()

        assert len(entries) == 3

        # Should be ordered by created_at ASC
        assert entries[0].client_order_id == "order_1"
        assert entries[1].client_order_id == "order_2"
        assert entries[2].client_order_id == "order_3"

    def test_get_all_entries_includes_metadata(self, journal):
        """Test get_all_entries includes metadata."""
        journal.log_order(
            client_order_id="order_1",
            intent_id="intent_1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata={
                "deployment_id": "deploy_123",
                "run_id": "run_456",
                "sequence": 1,
            },
        )

        entries = journal.get_all_entries()

        assert len(entries) == 1
        assert entries[0].metadata.get("deployment_id") == "deploy_123"
        assert entries[0].metadata.get("run_id") == "run_456"
        assert entries[0].metadata.get("sequence") == 1


class TestSequenceRecoveryEdgeCases:
    """Edge case tests for sequence recovery."""

    @pytest.fixture
    def temp_journal_path(self):
        """Create temporary path for journal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_journal.db"

    def test_recovery_ignores_other_deployments(self, temp_journal_path):
        """Test sequence recovery ignores entries from other deployments."""
        # First engine with deployment_A
        journal1 = OrderJournal(db_path=temp_journal_path)
        engine1 = LiveExecutionEngine(
            order_journal=journal1,
            deployment_id="deploy_A",
            run_id="run_001",
        )

        # Execute some orders
        for i in range(5):
            intent = OrderIntent(
                strategy_id="test_strategy",
                symbol=f"SYM{i}",
                intent_type=IntentType.MARKET_ENTRY,
                side=IntentSide.LONG,
                target_quantity=Decimal("0.1"),
            )
            engine1.execute(intent, origin="local")

        assert engine1.sequence == 5

        # New engine with deployment_B (should start at 0)
        journal2 = OrderJournal(db_path=temp_journal_path)
        engine2 = LiveExecutionEngine(
            order_journal=journal2,
            deployment_id="deploy_B",
            run_id="run_001",
        )

        assert engine2.sequence == 0

    def test_recovery_ignores_other_runs(self, temp_journal_path):
        """Test sequence recovery ignores entries from other runs."""
        # First engine with run_001
        journal1 = OrderJournal(db_path=temp_journal_path)
        engine1 = LiveExecutionEngine(
            order_journal=journal1,
            deployment_id="deploy_123",
            run_id="run_001",
        )

        # Execute some orders
        for i in range(3):
            intent = OrderIntent(
                strategy_id="test_strategy",
                symbol=f"SYM{i}",
                intent_type=IntentType.MARKET_ENTRY,
                side=IntentSide.LONG,
                target_quantity=Decimal("0.1"),
            )
            engine1.execute(intent, origin="local")

        assert engine1.sequence == 3

        # New engine with run_002 (should start at 0)
        journal2 = OrderJournal(db_path=temp_journal_path)
        engine2 = LiveExecutionEngine(
            order_journal=journal2,
            deployment_id="deploy_123",
            run_id="run_002",
        )

        assert engine2.sequence == 0

    def test_recovery_handles_gaps_in_sequence(self, temp_journal_path):
        """Test recovery finds max sequence even if there are gaps."""
        # Manually create entries with gaps in sequence
        journal = OrderJournal(db_path=temp_journal_path)

        journal.log_order(
            client_order_id="order_1",
            intent_id="intent_1",
            symbol="SYM1",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata={
                "deployment_id": "deploy_123",
                "run_id": "run_456",
                "sequence": 1,
            },
        )

        journal.log_order(
            client_order_id="order_5",
            intent_id="intent_5",
            symbol="SYM5",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata={
                "deployment_id": "deploy_123",
                "run_id": "run_456",
                "sequence": 5,  # Gap: no 2,3,4
            },
        )

        journal.log_order(
            client_order_id="order_3",
            intent_id="intent_3",
            symbol="SYM3",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata={
                "deployment_id": "deploy_123",
                "run_id": "run_456",
                "sequence": 3,
            },
        )

        # New engine should recover max sequence (5)
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        assert engine.sequence == 5

    def test_recovery_handles_missing_metadata(self, temp_journal_path):
        """Test recovery handles entries without metadata gracefully."""
        journal = OrderJournal(db_path=temp_journal_path)

        # Entry without metadata
        journal.log_order(
            client_order_id="order_1",
            intent_id="intent_1",
            symbol="SYM1",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata=None,
        )

        # Entry with partial metadata
        journal.log_order(
            client_order_id="order_2",
            intent_id="intent_2",
            symbol="SYM2",
            side="buy",
            quantity=Decimal("0.1"),
            order_type="market",
            metadata={"some_field": "value"},
        )

        # Should not crash
        engine = LiveExecutionEngine(
            order_journal=journal,
            deployment_id="deploy_123",
            run_id="run_456",
        )

        assert engine.sequence == 0  # No matching entries
