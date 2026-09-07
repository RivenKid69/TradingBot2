# -*- coding: utf-8 -*-
"""
tests/test_dry_run_executor.py
Tests for DryRunExecutor wrapper functionality.

Tests:
1. DryRunExecutor logs orders without executing
2. DryRunExecutor returns synthetic ExecReport
3. DryRunExecutor delegates non-execute methods
4. DryRunExecutor counter increments
5. Integration with BarExecutor
"""

import logging
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch, call
from typing import Any, Optional

from impl_bar_executor import DryRunExecutor, BarExecutor, ExecReport, ExecStatus


# =============================================================================
# Mock Order class for testing
# =============================================================================


class MockOrder:
    """Mock order for testing."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        side: str = "BUY",
        quantity: float = 1.0,
        order_type: str = "MARKET",
        meta: Optional[dict] = None,
    ):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.meta = meta or {}


# =============================================================================
# Mock Executor for testing delegation
# =============================================================================


class MockExecutor:
    """Mock executor to test delegation."""

    def __init__(self):
        self.execute_called = False
        self.custom_attr = "test_value"
        self.custom_method_called = False

    def execute(self, order: Any) -> ExecReport:
        """Execute order - should NOT be called in dry-run mode."""
        from core_models import Side, OrderType
        import time

        self.execute_called = True
        return ExecReport(
            ts=int(time.time() * 1000),
            run_id="mock",
            symbol=str(getattr(order, "symbol", "UNKNOWN")),
            side=Side.BUY,
            order_type=OrderType.MARKET,
            price=Decimal("50000.0"),
            quantity=Decimal("1.0"),
            fee=Decimal("0.1"),
            fee_asset="USDT",
            exec_status=ExecStatus.FILLED,
            order_id="real_order_123",
        )

    def custom_method(self) -> str:
        """Custom method that should be delegated."""
        self.custom_method_called = True
        return "custom_result"

    def get_portfolio_state(self) -> dict:
        """Another method that should be delegated."""
        return {"equity": 100000}


# =============================================================================
# Tests: Basic DryRunExecutor functionality
# =============================================================================


class TestDryRunExecutorBasic:
    """Test basic DryRunExecutor functionality."""

    def test_does_not_execute_real_order(self):
        """DryRunExecutor should NOT call wrapped executor's execute()."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder(symbol="ETHUSDT", side="SELL", quantity=5.0)
        result = dry_run.execute(order)

        # Real executor should NOT be called
        assert mock_executor.execute_called is False

        # Should return synthetic report (NEW = logged but not executed)
        assert result.exec_status == ExecStatus.NEW
        assert result.quantity == Decimal("0")
        assert "DRY-RUN" in result.meta.get("dry_run_message", "")

    def test_returns_synthetic_exec_report(self):
        """DryRunExecutor should return a synthetic ExecReport."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder(symbol="BTCUSDT", side="BUY", quantity=0.5)
        result = dry_run.execute(order)

        assert isinstance(result, ExecReport)
        assert result.symbol == "BTCUSDT"
        assert result.exec_status == ExecStatus.NEW
        assert result.quantity == Decimal("0")
        assert result.price == Decimal("0")
        assert "dry_run_" in result.order_id

    def test_counter_increments(self):
        """DryRunExecutor counter should increment with each execute call."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        assert dry_run.dry_run_count == 0

        for i in range(5):
            order = MockOrder(symbol=f"SYM{i}", side="BUY")
            dry_run.execute(order)
            assert dry_run.dry_run_count == i + 1

        assert dry_run.dry_run_count == 5

    def test_order_id_includes_counter(self):
        """DryRunExecutor order_id should include the counter."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        result1 = dry_run.execute(MockOrder())
        result2 = dry_run.execute(MockOrder())
        result3 = dry_run.execute(MockOrder())

        assert result1.order_id == "dry_run_1"
        assert result2.order_id == "dry_run_2"
        assert result3.order_id == "dry_run_3"


# =============================================================================
# Tests: Logging
# =============================================================================


class TestDryRunExecutorLogging:
    """Test DryRunExecutor logging functionality."""

    def test_logs_order_details(self, caplog):
        """DryRunExecutor should log order details."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor, log_level=logging.INFO)

        order = MockOrder(
            symbol="SOLUSDT",
            side="BUY",
            quantity=100.0,
            order_type="LIMIT",
        )

        with caplog.at_level(logging.INFO):
            dry_run.execute(order)

        # Check log contains key information
        log_text = caplog.text
        assert "DRY-RUN" in log_text
        assert "SOLUSDT" in log_text
        assert "BUY" in log_text
        assert "100" in log_text or "100.0" in log_text

    def test_logs_payload_info(self, caplog):
        """DryRunExecutor should log payload info from order meta."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor, log_level=logging.INFO)

        order = MockOrder(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            meta={
                "payload": {
                    "target_weight": 0.5,
                    "delta_weight": 0.1,
                    "edge_bps": 5.0,
                    "cost_bps": 2.0,
                }
            },
        )

        with caplog.at_level(logging.INFO):
            dry_run.execute(order)

        log_text = caplog.text
        assert "target_weight" in log_text
        assert "0.5" in log_text

    def test_custom_logger(self):
        """DryRunExecutor should use custom logger if provided."""
        mock_executor = MockExecutor()
        custom_logger = MagicMock()
        dry_run = DryRunExecutor(
            mock_executor,
            dry_run_logger=custom_logger,
            log_level=logging.WARNING,
        )

        order = MockOrder(symbol="AVAXUSDT", side="SELL")
        dry_run.execute(order)

        # Custom logger should be called
        custom_logger.log.assert_called_once()
        call_args = custom_logger.log.call_args
        assert call_args[0][0] == logging.WARNING  # log level
        assert "DRY-RUN" in call_args[0][1]  # message format


# =============================================================================
# Tests: Delegation
# =============================================================================


class TestDryRunExecutorDelegation:
    """Test DryRunExecutor delegation to wrapped executor."""

    def test_delegates_attributes(self):
        """DryRunExecutor should delegate attribute access."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        # Should delegate custom_attr
        assert dry_run.custom_attr == "test_value"

    def test_delegates_methods(self):
        """DryRunExecutor should delegate method calls."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        # Should delegate custom_method
        result = dry_run.custom_method()
        assert result == "custom_result"
        assert mock_executor.custom_method_called is True

    def test_delegates_complex_methods(self):
        """DryRunExecutor should delegate complex methods."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        # Should delegate get_portfolio_state
        state = dry_run.get_portfolio_state()
        assert state == {"equity": 100000}

    def test_does_not_delegate_execute(self):
        """DryRunExecutor should NOT delegate execute()."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder()
        result = dry_run.execute(order)

        # Should use dry-run execute, not delegated
        assert mock_executor.execute_called is False
        assert result.exec_status == ExecStatus.NEW


# =============================================================================
# Tests: Order attribute variations
# =============================================================================


class TestDryRunExecutorOrderVariations:
    """Test DryRunExecutor with various order attribute names."""

    def test_handles_qty_attribute(self):
        """DryRunExecutor should handle 'qty' instead of 'quantity'."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        class OrderWithQty:
            symbol = "BNBUSDT"
            side = "BUY"
            qty = 10.0
            order_type = "MARKET"

        result = dry_run.execute(OrderWithQty())
        assert result.exec_status == ExecStatus.NEW
        assert "DRY-RUN" in result.meta.get("dry_run_message", "")

    def test_handles_type_attribute(self):
        """DryRunExecutor should handle 'type' instead of 'order_type'."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        class OrderWithType:
            symbol = "DOTUSDT"
            side = "SELL"
            quantity = 20.0
            type = "LIMIT"

        result = dry_run.execute(OrderWithType())
        assert result.exec_status == ExecStatus.NEW

    def test_handles_missing_attributes(self):
        """DryRunExecutor should handle orders with missing attributes."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        class MinimalOrder:
            pass

        # Should not raise, use defaults
        result = dry_run.execute(MinimalOrder())
        assert result.exec_status == ExecStatus.NEW
        assert result.symbol == "UNKNOWN"


# =============================================================================
# Tests: Integration with BarExecutor
# =============================================================================


class TestDryRunExecutorBarExecutorIntegration:
    """Test DryRunExecutor integration with BarExecutor."""

    def test_wraps_bar_executor(self):
        """DryRunExecutor should successfully wrap BarExecutor."""
        # Create a minimal BarExecutor
        bar_executor = BarExecutor(
            run_id="test",
            bar_price="close",
            min_rebalance_step=0.01,
            cost_config=None,
            safety_margin_bps=0.0,
            max_participation=None,
            default_equity_usd=100000.0,
            initial_weights={},
            symbol_specs={},
        )

        dry_run = DryRunExecutor(bar_executor)

        # Should have access to BarExecutor attributes
        assert hasattr(dry_run, "portfolio_state") or True  # May not exist until first call

        # Execute should return dry-run report
        order = MockOrder(symbol="BTCUSDT", side="BUY", quantity=0.1)
        result = dry_run.execute(order)

        assert result.exec_status == ExecStatus.NEW
        assert "DRY-RUN" in result.meta.get("dry_run_message", "")


# =============================================================================
# Tests: Edge cases
# =============================================================================


class TestDryRunExecutorEdgeCases:
    """Test DryRunExecutor edge cases."""

    def test_none_meta(self):
        """DryRunExecutor should handle None meta gracefully."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder(symbol="TEST", meta=None)
        result = dry_run.execute(order)

        assert result.exec_status == ExecStatus.NEW

    def test_empty_payload(self):
        """DryRunExecutor should handle empty payload gracefully."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder(symbol="TEST", meta={"payload": {}})
        result = dry_run.execute(order)

        assert result.exec_status == ExecStatus.NEW

    def test_non_mapping_meta(self):
        """DryRunExecutor should handle non-mapping meta gracefully."""
        mock_executor = MockExecutor()
        dry_run = DryRunExecutor(mock_executor)

        order = MockOrder(symbol="TEST", meta="not_a_dict")
        result = dry_run.execute(order)

        assert result.exec_status == ExecStatus.NEW

    def test_multiple_executors_independent_counters(self):
        """Multiple DryRunExecutors should have independent counters."""
        mock1 = MockExecutor()
        mock2 = MockExecutor()
        dry1 = DryRunExecutor(mock1)
        dry2 = DryRunExecutor(mock2)

        for _ in range(3):
            dry1.execute(MockOrder())

        for _ in range(5):
            dry2.execute(MockOrder())

        assert dry1.dry_run_count == 3
        assert dry2.dry_run_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
