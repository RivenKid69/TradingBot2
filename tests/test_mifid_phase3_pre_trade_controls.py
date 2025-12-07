# -*- coding: utf-8 -*-
"""
Tests for MiFID II RTS 6 Article 15 Pre-Trade Controls.

Comprehensive test coverage for:
    - Price collar checks
    - Fat finger protection
    - Maximum order value/volume limits
    - Message rate limiting
    - Trader authorization
    - Daily loss limits

References:
    - RTS 6 Article 15: Pre-trade risk controls
"""

import pytest
import time
import threading
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from services.compliance.pre_trade_controls import (
    RejectionReason,
    ControlSeverity,
    PreTradeCheckResult,
    PreTradeControlsConfig,
    TraderAuthorization,
    MessageRateWindow,
    PreTradeControls,
    create_pre_trade_controls,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Default configuration for testing."""
    return PreTradeControlsConfig(
        price_collar_pct=5.0,
        fat_finger_price_deviation_pct=10.0,
        max_order_value_eur=Decimal("100000"),
        max_order_volume=Decimal("1000"),
        max_messages_per_second=50,
        max_messages_per_minute=1000,
        require_trader_authorization=True,
        max_daily_loss_eur=Decimal("10000"),
    )


@pytest.fixture
def controls(config):
    """Create pre-trade controls instance."""
    return PreTradeControls(config=config)


@pytest.fixture
def authorized_trader():
    """Create an authorized trader."""
    return TraderAuthorization(
        trader_id="TRADER001",
        name="Test Trader",
        authorized_instruments={"*"},  # All instruments
        authorized_venues={"XLON", "XETR"},
    )


# =============================================================================
# Test PreTradeCheckResult
# =============================================================================

class TestPreTradeCheckResult:
    """Tests for PreTradeCheckResult data class."""

    def test_allowed_result(self):
        """Test creating an allowed result."""
        result = PreTradeCheckResult(
            allowed=True,
            message="All checks passed",
        )

        assert result.allowed is True
        assert result.rejection_reason is None
        assert result.severity == ControlSeverity.INFO

    def test_rejected_result(self):
        """Test creating a rejected result."""
        result = PreTradeCheckResult(
            allowed=False,
            rejection_reason=RejectionReason.PRICE_COLLAR_BREACH,
            severity=ControlSeverity.HARD_REJECT,
            message="Price outside collar",
            details={"deviation": 7.5},
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.PRICE_COLLAR_BREACH
        assert result.severity == ControlSeverity.HARD_REJECT

    def test_result_to_dict(self):
        """Test result serialization."""
        result = PreTradeCheckResult(
            allowed=False,
            rejection_reason=RejectionReason.MAX_ORDER_VALUE_EXCEEDED,
            message="Order too large",
        )

        data = result.to_dict()

        assert data["allowed"] is False
        assert data["rejection_reason"] == "max_order_value_exceeded"
        assert "check_timestamp_ns" in data


# =============================================================================
# Test MessageRateWindow
# =============================================================================

class TestMessageRateWindow:
    """Tests for message rate limiting window."""

    def test_rate_within_limit(self):
        """Test recording within limit."""
        window = MessageRateWindow(
            window_size_seconds=1.0,
            max_messages=10,
        )

        now = time.time()
        for i in range(10):
            assert window.record(now + i * 0.05) is True

    def test_rate_exceeds_limit(self):
        """Test exceeding limit."""
        window = MessageRateWindow(
            window_size_seconds=1.0,
            max_messages=5,
        )

        now = time.time()
        for i in range(5):
            assert window.record(now) is True

        # Next should fail
        assert window.record(now) is False

    def test_rate_window_sliding(self):
        """Test sliding window behavior."""
        window = MessageRateWindow(
            window_size_seconds=0.5,
            max_messages=3,
        )

        start = time.time()

        # Fill window
        for i in range(3):
            assert window.record(start) is True

        # Should fail
        assert window.record(start) is False

        # After window expires, should succeed
        assert window.record(start + 0.6) is True


# =============================================================================
# Test Price Collar
# =============================================================================

class TestPriceCollar:
    """Tests for price collar controls (Article 15(1))."""

    def test_price_within_collar(self, controls, authorized_trader):
        """Test price within acceptable range."""
        controls.authorize_trader(authorized_trader)

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True

    def test_price_at_collar_boundary(self, controls, authorized_trader):
        """Test price at collar boundary."""
        controls.authorize_trader(authorized_trader)

        # 5% deviation is exactly at limit
        result = controls.check_order(
            order={"price": 105.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True  # At boundary, should pass

    def test_price_outside_collar(self, controls, authorized_trader):
        """Test price outside collar."""
        controls.authorize_trader(authorized_trader)

        # 6% deviation exceeds 5% collar
        result = controls.check_order(
            order={"price": 106.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.PRICE_COLLAR_BREACH

    def test_negative_price_rejected(self, controls, authorized_trader):
        """Test negative price is rejected."""
        controls.authorize_trader(authorized_trader)

        result = controls.check_order(
            order={"price": -10.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.ZERO_OR_NEGATIVE_PRICE

    def test_zero_price_rejected(self, controls, authorized_trader):
        """Test zero price is rejected."""
        controls.authorize_trader(authorized_trader)

        result = controls.check_order(
            order={"price": 0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.ZERO_OR_NEGATIVE_PRICE


# =============================================================================
# Test Fat Finger Protection
# =============================================================================

class TestFatFingerProtection:
    """Tests for fat finger protection."""

    def test_fat_finger_triggered(self, controls, authorized_trader):
        """Test fat finger protection triggered on large deviation."""
        controls.authorize_trader(authorized_trader)

        # 15% deviation exceeds 10% fat finger threshold
        result = controls.check_order(
            order={"price": 115.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.FAT_FINGER_PRICE

    def test_fat_finger_not_triggered(self, controls, authorized_trader):
        """Test fat finger not triggered on small deviation."""
        controls.authorize_trader(authorized_trader)

        # 8% deviation is within fat finger but outside collar
        # Should be rejected by collar first
        result = controls.check_order(
            order={"price": 108.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        # Rejected by price collar (5%), not fat finger (10%)
        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.PRICE_COLLAR_BREACH


# =============================================================================
# Test Maximum Order Value
# =============================================================================

class TestMaxOrderValue:
    """Tests for maximum order value controls (Article 15(2))."""

    def test_order_value_within_limit(self, controls, authorized_trader):
        """Test order value within limit."""
        controls.authorize_trader(authorized_trader)

        # 100 * 100 = 10000 EUR < 100000 limit
        result = controls.check_order(
            order={"price": 100.0, "quantity": 100, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True

    def test_order_value_exceeds_limit(self, controls, authorized_trader):
        """Test order value exceeding limit."""
        controls.authorize_trader(authorized_trader)

        # 1000 * 200 = 200000 EUR > 100000 limit
        result = controls.check_order(
            order={"price": 200.0, "quantity": 1000, "side": "BUY"},
            reference_price=Decimal("200"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.MAX_ORDER_VALUE_EXCEEDED

    def test_trader_specific_value_limit(self, controls):
        """Test trader-specific order value limit."""
        limited_trader = TraderAuthorization(
            trader_id="LIMITED001",
            name="Limited Trader",
            authorized_instruments={"*"},
            max_order_value_eur=Decimal("5000"),  # Lower limit
        )
        controls.authorize_trader(limited_trader)

        # 100 * 100 = 10000 > 5000 trader limit
        result = controls.check_order(
            order={"price": 100.0, "quantity": 100, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="LIMITED001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.MAX_ORDER_VALUE_EXCEEDED


# =============================================================================
# Test Maximum Order Volume
# =============================================================================

class TestMaxOrderVolume:
    """Tests for maximum order volume controls (Article 15(3))."""

    def test_volume_within_limit(self, controls, authorized_trader):
        """Test volume within limit."""
        controls.authorize_trader(authorized_trader)

        result = controls.check_order(
            order={"price": 100.0, "quantity": 500, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True

    def test_volume_exceeds_limit(self, controls, authorized_trader):
        """Test volume exceeding limit."""
        controls.authorize_trader(authorized_trader)

        # 1500 > 1000 limit
        result = controls.check_order(
            order={"price": 50.0, "quantity": 1500, "side": "BUY"},
            reference_price=Decimal("50"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.MAX_ORDER_VOLUME_EXCEEDED

    def test_adv_limit_exceeded(self, controls, authorized_trader):
        """Test ADV-based volume limit exceeded."""
        controls.authorize_trader(authorized_trader)

        # Order is 15% of ADV, limit is 10%
        result = controls.check_order(
            order={"price": 100.0, "quantity": 150, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
            adv=Decimal("1000"),  # ADV = 1000, 150 = 15%
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.MAX_ORDER_VOLUME_EXCEEDED


# =============================================================================
# Test Message Rate Limits
# =============================================================================

class TestMessageRateLimits:
    """Tests for message rate limits (Article 15(4))."""

    def test_rate_within_limit(self, config, authorized_trader):
        """Test message rate within limit."""
        config.max_messages_per_second = 100
        controls = PreTradeControls(config=config)
        controls.authorize_trader(authorized_trader)

        # Send 10 orders quickly
        for _ in range(10):
            result = controls.check_order(
                order={"price": 100.0, "quantity": 1, "side": "BUY"},
                reference_price=Decimal("100"),
                trader_id="TRADER001",
            )
            assert result.allowed is True

    def test_rate_exceeded(self, authorized_trader):
        """Test message rate exceeded."""
        config = PreTradeControlsConfig(
            max_messages_per_second=5,
            max_messages_per_minute=100,
            require_trader_authorization=True,
        )
        controls = PreTradeControls(config=config)
        controls.authorize_trader(authorized_trader)

        # Send more than limit
        rejections = 0
        for _ in range(10):
            result = controls.check_order(
                order={"price": 100.0, "quantity": 1, "side": "BUY"},
                reference_price=Decimal("100"),
                trader_id="TRADER001",
            )
            if not result.allowed:
                rejections += 1
                assert result.rejection_reason == RejectionReason.MESSAGE_RATE_EXCEEDED

        assert rejections > 0


# =============================================================================
# Test Trader Authorization
# =============================================================================

class TestTraderAuthorization:
    """Tests for trader authorization (Article 15(5))."""

    def test_unauthorized_trader_rejected(self, controls):
        """Test unauthorized trader is rejected."""
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="UNKNOWN_TRADER",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.UNAUTHORIZED_TRADER

    def test_authorized_trader_allowed(self, controls, authorized_trader):
        """Test authorized trader is allowed."""
        controls.authorize_trader(authorized_trader)

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True

    def test_revoked_trader_rejected(self, controls, authorized_trader):
        """Test revoked trader is rejected."""
        controls.authorize_trader(authorized_trader)

        # Revoke authorization
        controls.revoke_trader("TRADER001")

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.UNAUTHORIZED_TRADER

    def test_expired_authorization_rejected(self, controls):
        """Test expired authorization is rejected."""
        expired_trader = TraderAuthorization(
            trader_id="EXPIRED001",
            name="Expired Trader",
            authorized_instruments={"*"},
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),  # Expired
        )
        controls.authorize_trader(expired_trader)

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="EXPIRED001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.UNAUTHORIZED_TRADER

    def test_instrument_authorization(self, controls):
        """Test instrument-specific authorization."""
        limited_trader = TraderAuthorization(
            trader_id="LIMITED001",
            name="Limited Trader",
            authorized_instruments={"ISIN001", "ISIN002"},  # Only specific instruments
            authorized_venues={"*"},
        )
        controls.authorize_trader(limited_trader)

        # Authorized instrument
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="LIMITED001",
            instrument_isin="ISIN001",
        )
        assert result.allowed is True

        # Unauthorized instrument
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="LIMITED001",
            instrument_isin="ISIN999",
        )
        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.UNAUTHORIZED_INSTRUMENT

    def test_venue_authorization(self, controls):
        """Test venue-specific authorization."""
        limited_trader = TraderAuthorization(
            trader_id="LIMITED001",
            name="Limited Trader",
            authorized_instruments={"*"},
            authorized_venues={"XLON"},  # Only LSE
        )
        controls.authorize_trader(limited_trader)

        # Authorized venue
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="LIMITED001",
            venue="XLON",
        )
        assert result.allowed is True

        # Unauthorized venue
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="LIMITED001",
            venue="XETR",
        )
        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.UNAUTHORIZED_VENUE


# =============================================================================
# Test Daily Loss Limit
# =============================================================================

class TestDailyLossLimit:
    """Tests for daily loss limit checks."""

    def test_within_daily_limit(self, controls, authorized_trader):
        """Test trading within daily limit."""
        controls.authorize_trader(authorized_trader)

        # Set small loss
        controls.update_daily_pnl(Decimal("-5000"))

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is True

    def test_daily_limit_exceeded(self, controls, authorized_trader):
        """Test trading blocked when daily limit exceeded."""
        controls.authorize_trader(authorized_trader)

        # Set loss exceeding limit
        controls.update_daily_pnl(Decimal("-15000"))  # Exceeds 10000 limit

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT


# =============================================================================
# Test Concentration Limits
# =============================================================================

class TestConcentrationLimits:
    """Tests for position concentration limits."""

    def test_concentration_within_limit(self, controls, authorized_trader):
        """Test order within concentration limit."""
        controls.authorize_trader(authorized_trader)

        # Order is 10% of portfolio
        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
            portfolio_notional=Decimal("10000"),  # 1000 / 10000 = 10%
        )

        assert result.allowed is True

    def test_concentration_exceeded(self, controls, authorized_trader):
        """Test order exceeding concentration limit."""
        controls.authorize_trader(authorized_trader)

        # Order is 30% of portfolio, exceeds 25% limit
        result = controls.check_order(
            order={"price": 100.0, "quantity": 30, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
            portfolio_notional=Decimal("10000"),  # 3000 / 10000 = 30%
        )

        assert result.allowed is False
        assert result.rejection_reason == RejectionReason.CONCENTRATION_LIMIT


# =============================================================================
# Test Statistics and Reporting
# =============================================================================

class TestStatisticsAndReporting:
    """Tests for statistics and compliance reporting."""

    def test_statistics_tracking(self, controls, authorized_trader):
        """Test statistics are tracked."""
        controls.authorize_trader(authorized_trader)

        # Submit some orders
        controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        # Submit failing order
        controls.check_order(
            order={"price": 1000.0, "quantity": 1000, "side": "BUY"},  # Too large
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        stats = controls.get_statistics()

        assert stats["checks_total"] == 2
        assert stats["checks_passed"] >= 1
        assert stats["checks_rejected"] >= 1

    def test_compliance_report(self, controls):
        """Test compliance report generation."""
        report = controls.generate_compliance_report()

        assert report["regulation"] == "RTS 6 Article 15"
        assert report["requirement"] == "Pre-Trade Risk Controls"
        assert "configuration" in report
        assert "statistics" in report
        assert "controls_enabled" in report


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_with_dict_config(self):
        """Test creating with dictionary config."""
        config = {
            "price_collar_pct": 3.0,
            "max_order_value_eur": "50000",
            "require_trader_authorization": False,
        }

        controls = create_pre_trade_controls(config=config)

        assert controls is not None

    def test_create_with_defaults(self):
        """Test creating with default config."""
        controls = create_pre_trade_controls()

        assert controls is not None


# =============================================================================
# Test Audit Callback
# =============================================================================

class TestAuditCallback:
    """Tests for audit callback on rejections."""

    def test_audit_callback_on_rejection(self, config, authorized_trader):
        """Test audit callback is called on rejection."""
        audit_mock = Mock()
        controls = PreTradeControls(config=config, audit_callback=audit_mock)
        controls.authorize_trader(authorized_trader)

        # Submit failing order
        controls.check_order(
            order={"price": 200.0, "quantity": 10, "side": "BUY"},  # Outside collar
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        audit_mock.assert_called_once()
        result = audit_mock.call_args[0][0]
        assert isinstance(result, PreTradeCheckResult)
        assert result.allowed is False


# =============================================================================
# Test Disabled Controls
# =============================================================================

class TestDisabledControls:
    """Tests for disabled controls."""

    def test_disabled_price_collar(self, authorized_trader):
        """Test disabled price collar allows any price."""
        config = PreTradeControlsConfig(
            price_collar_enabled=False,
            require_trader_authorization=True,
        )
        controls = PreTradeControls(config=config)
        controls.authorize_trader(authorized_trader)

        # Large deviation should pass
        result = controls.check_order(
            order={"price": 200.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="TRADER001",
        )

        # May still fail on fat finger
        # If fat finger is also high or disabled, would pass

    def test_disabled_authorization(self):
        """Test disabled authorization allows any trader."""
        config = PreTradeControlsConfig(
            require_trader_authorization=False,
        )
        controls = PreTradeControls(config=config)

        result = controls.check_order(
            order={"price": 100.0, "quantity": 10, "side": "BUY"},
            reference_price=Decimal("100"),
            trader_id="ANY_TRADER",
        )

        assert result.allowed is True


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_order_checks(self, controls, authorized_trader):
        """Test concurrent order checks."""
        controls.authorize_trader(authorized_trader)

        results = []

        def check_order(n):
            result = controls.check_order(
                order={"price": 100.0, "quantity": 10, "side": "BUY"},
                reference_price=Decimal("100"),
                trader_id="TRADER001",
            )
            results.append((n, result.allowed))

        threads = [threading.Thread(target=check_order, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should have completed without error
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
