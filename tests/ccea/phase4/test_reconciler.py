# -*- coding: utf-8 -*-
"""
Tests for Position Reconciler - Phase 4.6

Tests cover:
- Position reconciliation with broker
- Order reconciliation
- Mismatch detection and handling
- Idempotency on restart
- Safe-halt on uncertainty
"""

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import pytest

from packages.agent.reconciliation.reconciler import (
    PositionReconciler,
    ReconciliationResult,
    PositionMismatch,
    MismatchAction,
)
from packages.agent.reconciliation.journal import (
    OrderJournal,
    JournalEntry,
    JournalStatus,
)


class TestPositionMismatch:
    """Test PositionMismatch dataclass."""

    def test_creation(self):
        """Test mismatch creation."""
        mismatch = PositionMismatch(
            symbol="BTCUSDT",
            local_quantity=Decimal("1.5"),
            broker_quantity=Decimal("1.0"),
            difference=Decimal("0.5"),
        )

        assert mismatch.symbol == "BTCUSDT"
        assert mismatch.difference == Decimal("0.5")

    def test_is_significant_true(self):
        """Test significant mismatch detection."""
        # > 1% difference is significant
        mismatch = PositionMismatch(
            symbol="BTCUSDT",
            local_quantity=Decimal("1.5"),
            broker_quantity=Decimal("1.0"),
            difference=Decimal("0.5"),
        )

        assert mismatch.is_significant is True

    def test_is_significant_false(self):
        """Test insignificant mismatch."""
        # < 1% difference is not significant
        mismatch = PositionMismatch(
            symbol="BTCUSDT",
            local_quantity=Decimal("1.005"),
            broker_quantity=Decimal("1.0"),
            difference=Decimal("0.005"),
        )

        assert mismatch.is_significant is False

    def test_to_dict(self):
        """Test serialization."""
        mismatch = PositionMismatch(
            symbol="ETHUSDT",
            local_quantity=Decimal("10.0"),
            broker_quantity=Decimal("9.0"),
            difference=Decimal("1.0"),
            severity="error",
        )

        data = mismatch.to_dict()

        assert data["symbol"] == "ETHUSDT"
        assert data["severity"] == "error"
        assert "timestamp" in data


class TestReconciliationResult:
    """Test ReconciliationResult dataclass."""

    def test_default_success(self):
        """Test default result is success."""
        result = ReconciliationResult()

        assert result.success is True
        assert result.halted is False
        assert result.mismatches == []

    def test_has_significant_mismatches(self):
        """Test significant mismatch detection."""
        result = ReconciliationResult(
            mismatches=[
                PositionMismatch(
                    symbol="BTCUSDT",
                    local_quantity=Decimal("2.0"),
                    broker_quantity=Decimal("1.0"),
                    difference=Decimal("1.0"),
                )
            ]
        )

        assert result.has_significant_mismatches is True

    def test_to_dict(self):
        """Test serialization."""
        result = ReconciliationResult(
            success=False,
            halted=True,
            halt_reason="Position mismatch",
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["halted"] is True
        assert "Position mismatch" in data["halt_reason"]


class TestPositionReconciler:
    """Test PositionReconciler operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def journal(self, temp_dir):
        """Create order journal."""
        journal = OrderJournal(db_path=temp_dir / "journal.db")
        yield journal
        journal.close()

    def test_reconcile_matching_positions(self):
        """Test reconciliation with matching positions."""
        local_positions = {"BTCUSDT": Decimal("1.0"), "ETHUSDT": Decimal("5.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0"), "ETHUSDT": Decimal("5.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
        )

        result = reconciler.reconcile()

        assert result.success is True
        assert result.halted is False
        assert len(result.mismatches) == 0

    def test_reconcile_within_tolerance(self):
        """Test positions within tolerance are accepted."""
        local_positions = {"BTCUSDT": Decimal("1.0001")}
        broker_positions = {"BTCUSDT": Decimal("1.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            tolerance_pct=Decimal("0.001"),  # 0.1% tolerance
        )

        result = reconciler.reconcile()

        assert result.success is True
        assert len(result.mismatches) == 0

    def test_reconcile_mismatch_halt(self):
        """Test mismatch triggers halt by default."""
        local_positions = {"BTCUSDT": Decimal("2.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            mismatch_action=MismatchAction.HALT,
        )

        result = reconciler.reconcile()

        assert result.success is False
        assert result.halted is True
        assert "mismatch" in result.halt_reason.lower()
        assert len(result.mismatches) == 1

    def test_reconcile_mismatch_accept_broker(self):
        """Test accepting broker positions on mismatch."""
        local_positions = {"BTCUSDT": Decimal("2.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            mismatch_action=MismatchAction.ACCEPT_BROKER,
        )

        result = reconciler.reconcile()

        assert result.success is True
        assert result.halted is False
        assert "BTCUSDT" in result.resolved

        # Local should now match broker
        assert reconciler.get_local_position("BTCUSDT") == Decimal("1.0")

    def test_reconcile_mismatch_alert_only(self):
        """Test alert-only mode doesn't halt."""
        local_positions = {"BTCUSDT": Decimal("2.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            mismatch_action=MismatchAction.ALERT,
        )

        result = reconciler.reconcile()

        assert result.success is True
        assert result.halted is False
        assert len(result.mismatches) == 1  # Recorded but not halted

    def test_reconcile_no_broker_connection_halts(self):
        """Test safe-halt when broker connection unavailable."""
        reconciler = PositionReconciler(
            local_positions={"BTCUSDT": Decimal("1.0")},
            fetch_broker_positions=None,  # No broker connection
        )

        result = reconciler.reconcile()

        assert result.success is False
        assert result.halted is True
        assert "No broker connection" in result.halt_reason

    def test_reconcile_broker_error_halts(self):
        """Test safe-halt on broker error."""

        def failing_fetch():
            raise ConnectionError("Broker offline")

        reconciler = PositionReconciler(
            local_positions={"BTCUSDT": Decimal("1.0")},
            fetch_broker_positions=failing_fetch,
        )

        result = reconciler.reconcile()

        assert result.success is False
        assert result.halted is True
        assert "Failed to fetch" in result.halt_reason

    def test_reconcile_extra_local_position(self):
        """Test detecting position in local but not broker."""
        local_positions = {"BTCUSDT": Decimal("1.0"), "ETHUSDT": Decimal("5.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0")}  # Missing ETHUSDT

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            mismatch_action=MismatchAction.HALT,
        )

        result = reconciler.reconcile()

        assert result.halted is True
        assert any(m.symbol == "ETHUSDT" for m in result.mismatches)

    def test_reconcile_extra_broker_position(self):
        """Test detecting position in broker but not local."""
        local_positions = {"BTCUSDT": Decimal("1.0")}
        broker_positions = {"BTCUSDT": Decimal("1.0"), "ETHUSDT": Decimal("5.0")}

        reconciler = PositionReconciler(
            local_positions=local_positions,
            fetch_broker_positions=lambda: broker_positions,
            mismatch_action=MismatchAction.HALT,
        )

        result = reconciler.reconcile()

        assert result.halted is True
        assert any(m.symbol == "ETHUSDT" for m in result.mismatches)

    def test_update_local_positions(self):
        """Test updating local positions."""
        reconciler = PositionReconciler()

        reconciler.update_local_positions({"BTCUSDT": Decimal("2.0")})

        assert reconciler.get_local_position("BTCUSDT") == Decimal("2.0")

    def test_set_local_position(self):
        """Test setting individual position."""
        reconciler = PositionReconciler()

        reconciler.set_local_position("BTCUSDT", Decimal("1.5"))

        assert reconciler.get_local_position("BTCUSDT") == Decimal("1.5")

    def test_set_local_position_zero_removes(self):
        """Test setting position to zero removes it."""
        reconciler = PositionReconciler(
            local_positions={"BTCUSDT": Decimal("1.0")}
        )

        reconciler.set_local_position("BTCUSDT", Decimal("0"))

        assert reconciler.get_local_position("BTCUSDT") == Decimal("0")


class TestOrderReconciliation:
    """Test order reconciliation on restart."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def journal(self, temp_dir):
        """Create order journal with pending orders."""
        journal = OrderJournal(db_path=temp_dir / "journal.db")
        # Add pending order
        journal.log_order(
            client_order_id="order-123",
            intent_id="intent-456",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.0"),
            order_type="market",
        )
        yield journal
        journal.close()

    def test_reconcile_orders_no_unresolved(self, temp_dir):
        """Test reconciliation with no unresolved orders."""
        journal = OrderJournal(db_path=temp_dir / "journal_empty.db")

        reconciler = PositionReconciler(journal=journal)

        result = reconciler.reconcile_orders()

        assert result.success is True
        assert result.halted is False
        journal.close()

    def test_reconcile_orders_halts_without_broker_fn(self, journal):
        """Test safe-halt when can't check broker order status."""
        reconciler = PositionReconciler(
            journal=journal,
            fetch_broker_order_status=None,  # No broker function
        )

        result = reconciler.reconcile_orders()

        assert result.success is False
        assert result.halted is True
        assert "cannot reconcile" in result.halt_reason.lower()

    def test_reconcile_orders_success(self, journal):
        """Test successful order reconciliation."""

        def mock_broker_status(entry: JournalEntry) -> Optional[JournalStatus]:
            # Broker says order is confirmed
            return JournalStatus.CONFIRMED

        reconciler = PositionReconciler(
            journal=journal,
            fetch_broker_order_status=mock_broker_status,
        )

        result = reconciler.reconcile_orders()

        assert result.success is True
        assert result.halted is False
        assert "order-123" in result.resolved

    def test_reconcile_orders_unknown_status_halts(self, journal):
        """Test safe-halt when broker returns unknown status."""

        def mock_broker_status(entry: JournalEntry) -> Optional[JournalStatus]:
            return None  # Unknown status

        reconciler = PositionReconciler(
            journal=journal,
            fetch_broker_order_status=mock_broker_status,
        )

        result = reconciler.reconcile_orders()

        assert result.success is False
        assert result.halted is True
        assert "unknown" in result.halt_reason.lower()

    def test_reconcile_orders_broker_error_halts(self, journal):
        """Test safe-halt when broker query fails."""

        def failing_broker_status(entry: JournalEntry) -> JournalStatus:
            raise ConnectionError("Broker unavailable")

        reconciler = PositionReconciler(
            journal=journal,
            fetch_broker_order_status=failing_broker_status,
        )

        result = reconciler.reconcile_orders()

        assert result.success is False
        assert result.halted is True
        assert "failed" in result.halt_reason.lower()


class TestOrderJournal:
    """Test OrderJournal for idempotency."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_idempotent_order_logging(self, temp_dir):
        """Test duplicate order detection."""
        journal = OrderJournal(db_path=temp_dir / "journal.db")

        # First order
        entry1 = journal.log_order(
            client_order_id="unique-order-1",
            intent_id="intent-1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.0"),
            order_type="market",
        )

        # Check duplicate
        assert journal.is_duplicate("unique-order-1") is True
        assert journal.is_duplicate("new-order") is False

        journal.close()

    def test_status_updates(self, temp_dir):
        """Test order status updates."""
        journal = OrderJournal(db_path=temp_dir / "journal.db")

        entry = journal.log_order(
            client_order_id="status-test",
            intent_id="intent-1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.0"),
            order_type="market",
        )

        # Update to submitted
        journal.update_status(entry.entry_id, JournalStatus.SUBMITTED)
        updated = journal.get_entry(entry.entry_id)
        assert updated.status == JournalStatus.SUBMITTED

        # Update to confirmed with fill info
        journal.update_status(
            entry.entry_id,
            JournalStatus.CONFIRMED,
            broker_order_id="broker-123",
            filled_quantity=Decimal("1.0"),
            avg_price=Decimal("50000.0"),
        )
        updated = journal.get_entry(entry.entry_id)
        assert updated.status == JournalStatus.CONFIRMED
        assert updated.broker_order_id == "broker-123"
        assert updated.filled_quantity == "1.0"

        journal.close()

    def test_get_unresolved_orders(self, temp_dir):
        """Test getting unresolved orders for restart."""
        journal = OrderJournal(db_path=temp_dir / "journal.db")

        # Add orders with different statuses
        journal.log_order(
            client_order_id="pending-1",
            intent_id="intent-1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.0"),
            order_type="market",
        )

        entry2 = journal.log_order(
            client_order_id="confirmed-1",
            intent_id="intent-2",
            symbol="ETHUSDT",
            side="sell",
            quantity=Decimal("5.0"),
            order_type="limit",
        )
        journal.update_status(entry2.entry_id, JournalStatus.CONFIRMED)

        unresolved = journal.get_unresolved_orders()

        assert len(unresolved) == 1
        assert unresolved[0].client_order_id == "pending-1"

        journal.close()

    def test_persistence_across_restart(self, temp_dir):
        """Test journal persists across restarts."""
        db_path = temp_dir / "persistent.db"

        # Create and populate journal
        journal1 = OrderJournal(db_path=db_path)
        journal1.log_order(
            client_order_id="persistent-order",
            intent_id="intent-1",
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1.0"),
            order_type="market",
        )
        journal1.close()

        # Simulate restart - create new journal instance
        journal2 = OrderJournal(db_path=db_path)

        # Order should still be there
        assert journal2.is_duplicate("persistent-order") is True
        entry = journal2.get_by_client_id("persistent-order")
        assert entry is not None
        assert entry.symbol == "BTCUSDT"

        journal2.close()
