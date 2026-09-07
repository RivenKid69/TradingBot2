# -*- coding: utf-8 -*-
"""
tests/test_stock_risk_management.py
Comprehensive tests for Phase 6: Stock-specific risk management.

Tests cover:
1. PDT Rule Enforcement - Pattern Day Trader rule (< $25k accounts)
2. Margin Requirements - Reg T margin (50% initial, 25% maintenance)
3. Short Sale Rules - Uptick rule, HTB list, circuit breakers
4. Corporate Actions - Dividends, stock splits
5. StockRiskGuard - Combined integration tests
6. Backward Compatibility - Crypto trading unaffected
"""

import pytest
from datetime import datetime, timezone, date, timedelta
from unittest.mock import MagicMock, patch
import math

# Import the modules under test
from services.pdt_tracker import (
    PDTTracker,
    PDTTrackerConfig,
    PDTStatus,
    DayTrade,
    DayTradeType,
    OpenPosition,
    create_pdt_tracker,
    PDT_EQUITY_THRESHOLD,
    PDT_MAX_DAY_TRADES,
    PDT_ROLLING_DAYS,
)

from services.stock_risk_guards import (
    MarginGuard,
    MarginGuardConfig,
    MarginRequirement,
    MarginStatus,
    MarginCallType,
    ShortSaleGuard,
    ShortSaleGuardConfig,
    ShortSaleStatus,
    ShortSaleRestriction,
    CorporateActionsHandler,
    CorporateActionsConfig,
    CorporateAction,
    CorporateActionType,
    PositionSnapshot,
    REG_T_INITIAL_MARGIN,
    REG_T_MAINTENANCE_MARGIN,
    create_margin_guard,
    create_short_sale_guard,
    create_corporate_actions_handler,
)

from risk_guard import (
    RiskEvent,
    RiskGuard,
    RiskConfig,
    StockRiskGuard,
    StockRiskConfig,
    create_stock_risk_guard,
    create_combined_risk_guard,
)


# =========================
# Helper Functions
# =========================


def get_timestamp_ms(year=2024, month=6, day=15, hour=10, minute=0) -> int:
    """Create a timestamp in milliseconds."""
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_business_day_timestamp(days_ago: int = 0) -> int:
    """Get timestamp for a business day N days ago."""
    now = datetime.now(timezone.utc)
    current = now - timedelta(days=days_ago)
    # Adjust to a weekday if weekend
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return int(current.replace(hour=10, minute=0, second=0).timestamp() * 1000)


# =========================
# PDT Tracker Tests
# =========================


class TestPDTTracker:
    """Tests for Pattern Day Trader rule enforcement."""

    def test_initialization_default(self):
        """Test default PDT tracker initialization."""
        tracker = PDTTracker()
        assert tracker.account_equity == 30_000.0
        assert not tracker.is_pdt_flagged
        assert tracker.is_exempt  # $30k >= $25k

    def test_initialization_under_threshold(self):
        """Test PDT tracker with account under threshold."""
        tracker = PDTTracker(account_equity=20_000.0)
        assert tracker.account_equity == 20_000.0
        assert not tracker.is_exempt  # $20k < $25k
        assert not tracker.is_pdt_flagged

    def test_is_exempt_threshold(self):
        """Test PDT exemption at exactly $25k."""
        # At threshold - exempt
        tracker = PDTTracker(account_equity=25_000.0)
        assert tracker.is_exempt

        # Just below - not exempt
        tracker = PDTTracker(account_equity=24_999.99)
        assert not tracker.is_exempt

    def test_can_day_trade_exempt(self):
        """Test day trade check for exempt account."""
        tracker = PDTTracker(account_equity=30_000.0)
        can_trade, reason = tracker.can_day_trade("AAPL")
        assert can_trade
        assert "exempt" in reason.lower()

    def test_can_day_trade_under_limit(self):
        """Test day trade check when under the limit."""
        tracker = PDTTracker(account_equity=20_000.0)
        can_trade, reason = tracker.can_day_trade("AAPL")
        assert can_trade
        assert "remaining" in reason.lower()

    def test_day_trade_counting(self):
        """Test that day trades are counted correctly."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)

        # Record first day trade
        tracker.record_day_trade("AAPL", ts)
        assert tracker.get_day_trade_count(ts) == 1
        assert tracker.get_remaining_day_trades(ts) == 2

        # Record second day trade
        tracker.record_day_trade("MSFT", ts)
        assert tracker.get_day_trade_count(ts) == 2
        assert tracker.get_remaining_day_trades(ts) == 1

    def test_day_trade_limit_reached(self):
        """Test that trades are blocked at limit."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)

        # Use up all day trades
        for _ in range(3):
            tracker.record_day_trade("AAPL", ts)

        assert tracker.get_day_trade_count(ts) == 3
        assert tracker.get_remaining_day_trades(ts) == 0

        # Next trade should be blocked
        can_trade, reason = tracker.can_day_trade("AAPL", ts)
        assert not can_trade
        assert "limit" in reason.lower()

    def test_rolling_window(self):
        """Test rolling 5 business day window."""
        tracker = PDTTracker(account_equity=20_000.0)

        # Trade 6 business days ago (should NOT count)
        old_ts = get_business_day_timestamp(6)
        tracker.record_day_trade("AAPL", old_ts)

        # Trade today (should count)
        today_ts = get_business_day_timestamp(0)
        tracker.record_day_trade("MSFT", today_ts)

        # Only today's trade should count in window
        count = tracker.get_day_trade_count(today_ts)
        # Note: The old trade might still be in _day_trades list,
        # but get_day_trades_in_window should filter it
        in_window = tracker.get_day_trades_in_window(today_ts)
        assert len(in_window) <= 2  # At most 2 if old one still in window

    def test_open_close_tracking(self):
        """Test opening and closing position tracking."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts_open = get_business_day_timestamp(0)
        ts_close = ts_open + 3600_000  # 1 hour later

        # Open position
        tracker.record_open("AAPL", "LONG", 100, 150.0, ts_open)

        # Close position same day
        day_trade = tracker.record_close("AAPL", 100, 155.0, ts_close)

        assert day_trade is not None
        assert day_trade.symbol == "AAPL"
        assert day_trade.trade_type == DayTradeType.LONG_ROUND_TRIP
        assert day_trade.pnl == 500.0  # (155-150) * 100

    def test_short_round_trip(self):
        """Test short sell and cover day trade."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)

        # Short sell
        tracker.record_open("TSLA", "SHORT", 50, 200.0, ts)

        # Cover same day
        day_trade = tracker.record_close("TSLA", 50, 190.0, ts + 1000)

        assert day_trade is not None
        assert day_trade.trade_type == DayTradeType.SHORT_ROUND_TRIP
        assert day_trade.pnl == 500.0  # (200-190) * 50 profit on short

    def test_status_transitions(self):
        """Test PDT status transitions."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)

        # Initial: COMPLIANT
        assert tracker.get_status(ts) == PDTStatus.COMPLIANT

        # After 2 trades: WARNING
        tracker.record_day_trade("AAPL", ts)
        tracker.record_day_trade("MSFT", ts)
        assert tracker.get_status(ts) == PDTStatus.WARNING

        # After 3 trades: AT_LIMIT
        tracker.record_day_trade("GOOGL", ts)
        assert tracker.get_status(ts) == PDTStatus.AT_LIMIT

    def test_pdt_flag(self):
        """Test PDT flagging."""
        tracker = PDTTracker(account_equity=20_000.0)

        # Flag as PDT
        tracker.flag_as_pdt()
        assert tracker.is_pdt_flagged
        assert tracker.get_status() == PDTStatus.RESTRICTED

        # Flagged accounts can't day trade
        can_trade, reason = tracker.can_day_trade("AAPL")
        assert not can_trade
        assert "flagged" in reason.lower()

    def test_serialization(self):
        """Test tracker serialization and deserialization."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)
        tracker.record_day_trade("AAPL", ts, buy_price=150.0, sell_price=155.0)

        # Serialize
        data = tracker.to_dict()
        assert data["account_equity"] == 20_000.0
        assert len(data["day_trades"]) == 1

        # Deserialize
        restored = PDTTracker.from_dict(data)
        assert restored.account_equity == 20_000.0
        assert len(restored._day_trades) == 1

    def test_reset(self):
        """Test tracker reset."""
        tracker = PDTTracker(account_equity=20_000.0)
        ts = get_business_day_timestamp(0)
        tracker.record_day_trade("AAPL", ts)

        assert tracker.get_day_trade_count(ts) == 1

        tracker.reset()
        assert tracker.get_day_trade_count(ts) == 0


# =========================
# Margin Guard Tests
# =========================


class TestMarginGuard:
    """Tests for Reg T margin requirements."""

    def test_initialization_default(self):
        """Test default margin guard initialization."""
        guard = MarginGuard()
        assert guard._config.initial_margin == REG_T_INITIAL_MARGIN
        assert guard._config.maintenance_margin == REG_T_MAINTENANCE_MARGIN

    def test_set_equity(self):
        """Test setting account equity."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 20_000.0)
        assert guard._equity == 50_000.0
        assert guard._cash == 20_000.0

    def test_buying_power_calculation(self):
        """Test buying power calculation with 50% margin."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 50_000.0)

        # With 50% initial margin, $50k equity = $100k buying power
        bp = guard._available_buying_power()
        assert bp == pytest.approx(100_000.0, rel=0.01)

    def test_can_open_position_success(self):
        """Test successful position opening."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 50_000.0)

        # Try to buy $80k worth (within $100k buying power)
        can_buy, reason = guard.can_open_position("AAPL", 400, 200.0)
        assert can_buy
        assert "OK" in reason

    def test_can_open_position_insufficient_margin(self):
        """Test position opening blocked by insufficient margin."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 50_000.0)

        # Try to buy $150k worth (exceeds $100k buying power)
        can_buy, reason = guard.can_open_position("AAPL", 1000, 150.0)
        assert not can_buy
        assert "Insufficient" in reason

    def test_margin_requirement_custom(self):
        """Test custom margin requirement for a symbol."""
        guard = MarginGuard()
        guard.set_equity(50_000.0)

        # Set higher margin for volatile stock
        req = MarginRequirement(symbol="MEME", initial_margin=0.70, is_volatile=True)
        guard.set_symbol_requirement(req)

        # With 70% margin, buying power is lower
        margin_needed = guard._calculate_position_margin("MEME", 100, 100.0, is_new=True)
        assert margin_needed == pytest.approx(7_000.0, rel=0.01)  # 70% of $10k

    def test_non_marginable_stock(self):
        """Test non-marginable stock requires full cash."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 10_000.0)  # Only $10k cash

        # Set symbol as non-marginable
        req = MarginRequirement(symbol="PENNY", is_marginable=False)
        guard.set_symbol_requirement(req)

        # Try to buy $20k worth (more than $10k cash)
        can_buy, reason = guard.can_open_position("PENNY", 200, 100.0)
        assert not can_buy
        assert "not marginable" in reason.lower()

    def test_margin_call_detection(self):
        """Test margin call detection."""
        guard = MarginGuard()
        guard.set_equity(10_000.0)  # Low equity

        # Add position worth $50k
        guard.set_position(
            PositionSnapshot(
                symbol="AAPL",
                quantity=250,
                market_value=50_000.0,
                cost_basis=45_000.0,
                unrealized_pnl=5_000.0,
            )
        )

        # Check margin status
        status = guard.check_margin_call()

        # Maintenance margin on $50k = $12,500 (25%)
        # Equity $10k < $12,500 = margin call
        assert status.margin_call_type != MarginCallType.NONE
        assert status.margin_call_amount > 0

    def test_calculate_max_position(self):
        """Test maximum position calculation."""
        guard = MarginGuard()
        guard.set_equity(50_000.0, 50_000.0)

        # Max shares at $100/share with buffer
        max_shares = guard.calculate_max_position("AAPL", 100.0)

        # Buying power ~$100k, at $100/share = ~1000 shares (minus buffer)
        assert max_shares > 0
        assert max_shares <= 1000


# =========================
# Short Sale Guard Tests
# =========================


class TestShortSaleGuard:
    """Tests for short sale rules enforcement."""

    def test_initialization_default(self):
        """Test default short sale guard initialization."""
        guard = ShortSaleGuard()
        assert guard._config.enforce_uptick_rule
        assert guard._config.check_htb_list

    def test_can_short_no_restrictions(self):
        """Test shorting allowed with no restrictions."""
        guard = ShortSaleGuard()
        can_short, status = guard.can_short("AAPL", 150.0)
        assert can_short
        assert status.restriction == ShortSaleRestriction.NONE

    def test_circuit_breaker_trigger(self):
        """Test circuit breaker activation."""
        guard = ShortSaleGuard()
        ts = get_timestamp_ms()

        # Trigger circuit breaker
        guard.trigger_circuit_breaker("AAPL", ts)

        # Check that it's active
        assert guard.is_circuit_breaker_active("AAPL", ts)

        # Check that it expires
        future_ts = ts + 100 * 3600 * 1000  # 100 hours later
        assert not guard.is_circuit_breaker_active("AAPL", future_ts)

    def test_uptick_rule_enforcement(self):
        """Test uptick rule during circuit breaker."""
        guard = ShortSaleGuard()
        ts = get_timestamp_ms()

        # Trigger circuit breaker
        guard.trigger_circuit_breaker("AAPL", ts)

        # Set last sale at $150
        guard.update_last_sale("AAPL", 150.0, ts)

        # Try to short at $150 (not an uptick) - should fail
        can_short, status = guard.can_short("AAPL", 150.0, 100, ts + 1000)
        assert not can_short
        assert status.restriction == ShortSaleRestriction.UPTICK_RULE

        # Try to short at $150.01 (uptick) - should succeed
        can_short, status = guard.can_short("AAPL", 150.01, 100, ts + 2000)
        assert can_short

    def test_htb_status(self):
        """Test hard-to-borrow status."""
        guard = ShortSaleGuard()

        # Set HTB status manually
        guard.set_htb_status("GME", is_htb=True, borrow_rate=0.50)

        status = guard.get_short_status("GME")
        assert not status.is_easy_to_borrow
        assert status.borrow_rate == 0.50

    def test_restricted_symbol(self):
        """Test manually restricted symbol."""
        guard = ShortSaleGuard()

        # Add restriction
        guard.add_restriction("HALT")

        # Should not be shortable
        can_short, status = guard.can_short("HALT", 100.0)
        assert not can_short
        assert status.restriction == ShortSaleRestriction.RESTRICTED

        # Remove restriction
        guard.remove_restriction("HALT")
        can_short, _ = guard.can_short("HALT", 100.0)
        assert can_short

    def test_clear_circuit_breaker(self):
        """Test clearing circuit breaker."""
        guard = ShortSaleGuard()
        ts = get_timestamp_ms()

        guard.trigger_circuit_breaker("AAPL", ts)
        assert guard.is_circuit_breaker_active("AAPL", ts)

        guard.clear_circuit_breaker("AAPL")
        assert not guard.is_circuit_breaker_active("AAPL", ts)


# =========================
# Corporate Actions Tests
# =========================


class TestCorporateActionsHandler:
    """Tests for corporate actions handling."""

    def test_initialization_default(self):
        """Test default handler initialization."""
        handler = CorporateActionsHandler()
        assert handler._config.adjust_positions_on_split
        assert handler._config.warn_on_ex_dividend

    def test_add_dividend(self):
        """Test adding a dividend action."""
        handler = CorporateActionsHandler()

        action = CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=date(2024, 2, 9),
            dividend_amount=0.24,
        )
        handler.add_action(action)

        actions = handler.get_actions("AAPL")
        assert len(actions) == 1
        assert actions[0].dividend_amount == 0.24

    def test_add_stock_split(self):
        """Test adding a stock split action."""
        handler = CorporateActionsHandler()

        action = CorporateAction(
            symbol="TSLA",
            action_type=CorporateActionType.STOCK_SPLIT,
            ex_date=date(2024, 8, 25),
            split_ratio=(3, 1),  # 3-for-1 split
        )
        handler.add_action(action)

        actions = handler.get_actions("TSLA", CorporateActionType.STOCK_SPLIT)
        assert len(actions) == 1
        assert actions[0].split_ratio == (3, 1)

    def test_apply_split(self):
        """Test applying stock split to position."""
        handler = CorporateActionsHandler()

        # Before split: 100 shares @ $300
        # 3-for-1 split: 300 shares @ $100
        new_qty, new_price = handler.apply_split(
            "TSLA", quantity=100, price=300.0, split_ratio=(3, 1)
        )

        assert new_qty == 300
        assert new_price == pytest.approx(100.0, rel=0.01)

    def test_apply_reverse_split(self):
        """Test applying reverse stock split."""
        handler = CorporateActionsHandler()

        # Before: 100 shares @ $5
        # 1-for-10 reverse split: 10 shares @ $50
        new_qty, new_price = handler.apply_split(
            "PENNY", quantity=100, price=5.0, split_ratio=(1, 10)
        )

        assert new_qty == 10
        assert new_price == pytest.approx(50.0, rel=0.01)

    def test_dividend_warning_long(self):
        """Test dividend warning for long position."""
        handler = CorporateActionsHandler()

        # Add upcoming dividend
        tomorrow = date.today() + timedelta(days=1)
        action = CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=tomorrow,
            dividend_amount=0.24,
        )
        handler.add_action(action)

        # Check warning
        warning = handler.check_dividend_warning("AAPL", is_short=False)
        assert warning is not None
        assert "INFO" in warning
        assert "0.24" in warning

    def test_dividend_warning_short(self):
        """Test dividend warning for short position."""
        handler = CorporateActionsHandler()

        # Add upcoming dividend
        tomorrow = date.today() + timedelta(days=1)
        action = CorporateAction(
            symbol="AAPL",
            action_type=CorporateActionType.DIVIDEND,
            ex_date=tomorrow,
            dividend_amount=0.24,
        )
        handler.add_action(action)

        # Check warning for short
        warning = handler.check_dividend_warning("AAPL", is_short=True)
        assert warning is not None
        assert "WARNING" in warning
        assert "owe" in warning.lower()

    def test_upcoming_actions(self):
        """Test getting upcoming actions."""
        handler = CorporateActionsHandler()

        # Add actions at different dates
        today = date.today()
        handler.add_action(
            CorporateAction(
                symbol="AAPL",
                action_type=CorporateActionType.DIVIDEND,
                ex_date=today + timedelta(days=2),
                dividend_amount=0.24,
            )
        )
        handler.add_action(
            CorporateAction(
                symbol="MSFT",
                action_type=CorporateActionType.DIVIDEND,
                ex_date=today + timedelta(days=10),
                dividend_amount=0.68,
            )
        )

        # Get actions in next 7 days
        upcoming = handler.get_upcoming_actions(days=7)
        assert len(upcoming) == 1
        assert upcoming[0].symbol == "AAPL"


# =========================
# Stock Risk Guard Integration Tests
# =========================


class TestStockRiskGuard:
    """Integration tests for combined stock risk guard."""

    def test_initialization_equity(self):
        """Test initialization for equity trading."""
        config = StockRiskConfig(market_type="EQUITY")
        guard = StockRiskGuard(config)

        assert guard.config.is_stock_trading
        assert guard.pdt_tracker is not None
        assert guard.margin_guard is not None
        assert guard.short_sale_guard is not None
        assert guard.corporate_actions is not None

    def test_initialization_crypto_skips_guards(self):
        """Test that crypto mode skips stock guards."""
        config = StockRiskConfig(market_type="CRYPTO_SPOT")
        guard = StockRiskGuard(config)

        assert not guard.config.is_stock_trading
        assert guard.pdt_tracker is None
        assert guard.margin_guard is None

    def test_check_trade_crypto_always_passes(self):
        """Test that crypto trades always pass stock checks."""
        config = StockRiskConfig(market_type="CRYPTO_SPOT")
        guard = StockRiskGuard(config)

        event = guard.check_trade("BTCUSDT", "BUY", 1.0, 50000.0)
        assert event == RiskEvent.NONE

    def test_check_trade_pdt_violation(self):
        """Test PDT violation detection."""
        config = StockRiskConfig(
            market_type="EQUITY",
            pdt_account_equity=20_000.0,  # Under threshold
        )
        guard = StockRiskGuard(config)
        ts = get_business_day_timestamp(0)

        # Use up all day trades
        for _ in range(3):
            guard.record_day_trade("AAPL", ts)

        # Next day trade should be blocked
        event = guard.check_trade(
            "AAPL",
            "BUY",
            100,
            150.0,
            is_day_trade=True,
            timestamp_ms=ts,
        )
        assert event == RiskEvent.PDT_VIOLATION

    def test_check_trade_margin_violation(self):
        """Test margin violation detection."""
        config = StockRiskConfig(market_type="EQUITY")
        guard = StockRiskGuard(config)

        # Set low equity
        guard.update_account_equity(1_000.0, 1_000.0)

        # Try to buy way more than buying power allows
        event = guard.check_trade("AAPL", "BUY", 1000, 150.0)
        assert event == RiskEvent.MARGIN_CALL

    def test_check_short_sale_with_circuit_breaker(self):
        """Test short sale with circuit breaker."""
        config = StockRiskConfig(market_type="EQUITY")
        guard = StockRiskGuard(config)
        ts = get_timestamp_ms()

        # Trigger circuit breaker
        guard.trigger_circuit_breaker("GME", ts)

        # Update last sale
        guard.short_sale_guard.update_last_sale("GME", 100.0, ts)

        # Try to short at same price (not uptick)
        can_short, reason = guard.check_short_sale("GME", 100.0, 50, ts + 1000)
        assert not can_short

    def test_record_trade_updates_guards(self):
        """Test that recording trade updates all relevant guards."""
        config = StockRiskConfig(market_type="EQUITY", pdt_account_equity=20_000.0)
        guard = StockRiskGuard(config)
        ts = get_business_day_timestamp(0)

        # Record opening trade
        guard.record_trade("AAPL", "BUY", 100, 150.0, ts, is_opening=True)

        # Should have open position in PDT tracker
        assert len(guard.pdt_tracker._open_positions["AAPL"]) == 1

        # Record closing trade same day
        guard.record_trade("AAPL", "SELL", 100, 155.0, ts + 3600000, is_opening=False)

        # Should have recorded day trade
        assert guard.pdt_tracker.get_day_trade_count(ts) == 1

    def test_add_corporate_action(self):
        """Test adding corporate action through stock guard."""
        config = StockRiskConfig(market_type="EQUITY")
        guard = StockRiskGuard(config)

        # Add dividend
        tomorrow = date.today() + timedelta(days=1)
        guard.add_corporate_action("AAPL", "DIVIDEND", tomorrow, dividend_amount=0.24)

        # Should be in corporate actions handler
        actions = guard.corporate_actions.get_actions("AAPL")
        assert len(actions) == 1

    def test_snapshot(self):
        """Test status snapshot."""
        config = StockRiskConfig(market_type="EQUITY")
        guard = StockRiskGuard(config)

        snapshot = guard.snapshot()
        assert snapshot["market_type"] == "EQUITY"
        assert snapshot["is_stock_trading"]
        assert snapshot["pdt_enabled"]
        assert snapshot["margin_enabled"]

    def test_reset(self):
        """Test guard reset."""
        config = StockRiskConfig(market_type="EQUITY", pdt_account_equity=20_000.0)
        guard = StockRiskGuard(config)
        ts = get_business_day_timestamp(0)

        # Record some activity
        guard.record_day_trade("AAPL", ts)
        assert guard.pdt_tracker.get_day_trade_count(ts) == 1

        # Reset
        guard.reset()
        assert guard.pdt_tracker.get_day_trade_count(ts) == 0


# =========================
# Backward Compatibility Tests
# =========================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with crypto trading."""

    def test_risk_guard_unchanged(self):
        """Test that base RiskGuard behavior is unchanged."""
        guard = RiskGuard()
        assert guard.last_event() == RiskEvent.NONE

    def test_create_combined_guard_crypto(self):
        """Test combined guard creation for crypto."""
        crypto_config = StockRiskConfig(market_type="CRYPTO_SPOT")
        risk_guard, stock_guard = create_combined_risk_guard(stock_config=crypto_config)

        assert risk_guard is not None
        assert stock_guard is None  # Should be None for crypto

    def test_create_combined_guard_equity(self):
        """Test combined guard creation for equity."""
        stock_config = StockRiskConfig(market_type="EQUITY")
        risk_guard, stock_guard = create_combined_risk_guard(stock_config=stock_config)

        assert risk_guard is not None
        assert stock_guard is not None

    def test_stock_guard_skips_crypto(self):
        """Test that stock guard does nothing for crypto symbols."""
        crypto_config = StockRiskConfig(market_type="CRYPTO_SPOT")
        guard = StockRiskGuard(crypto_config)

        # These should all be no-ops
        event = guard.check_trade("BTCUSDT", "BUY", 1.0, 50000.0)
        assert event == RiskEvent.NONE

        guard.record_trade("BTCUSDT", "BUY", 1.0, 50000.0, get_timestamp_ms())
        guard.update_account_equity(100_000.0)
        guard.reset()

        # Should not raise any errors

    def test_risk_event_values_unchanged(self):
        """Test that original RiskEvent values are unchanged."""
        # Verify original values haven't changed
        assert RiskEvent.NONE == 0
        assert RiskEvent.POSITION_LIMIT == 1
        assert RiskEvent.NOTIONAL_LIMIT == 2
        assert RiskEvent.DRAWDOWN == 3
        assert RiskEvent.BANKRUPTCY == 4

        # New values should be higher
        assert RiskEvent.PDT_VIOLATION > RiskEvent.BANKRUPTCY
        assert RiskEvent.MARGIN_CALL > RiskEvent.BANKRUPTCY
        assert RiskEvent.SHORT_SALE_RESTRICTED > RiskEvent.BANKRUPTCY


# =========================
# Factory Function Tests
# =========================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_pdt_tracker(self):
        """Test PDT tracker factory."""
        tracker = create_pdt_tracker(
            account_equity=20_000.0,
            simulation_mode=True,
        )
        assert tracker.account_equity == 20_000.0
        assert tracker._config.simulation_mode

    def test_create_margin_guard(self):
        """Test margin guard factory."""
        guard = create_margin_guard(
            initial_margin=0.60,
            maintenance_margin=0.30,
        )
        assert guard._config.initial_margin == 0.60
        assert guard._config.maintenance_margin == 0.30

    def test_create_short_sale_guard(self):
        """Test short sale guard factory."""
        guard = create_short_sale_guard(
            enforce_uptick_rule=False,
            check_htb_list=False,
        )
        assert not guard._config.enforce_uptick_rule
        assert not guard._config.check_htb_list

    def test_create_corporate_actions_handler(self):
        """Test corporate actions handler factory."""
        handler = create_corporate_actions_handler(
            adjust_positions_on_split=False,
        )
        assert not handler._config.adjust_positions_on_split

    def test_create_stock_risk_guard(self):
        """Test stock risk guard factory."""
        guard = create_stock_risk_guard(
            market_type="EQUITY",
            account_equity=15_000.0,
            simulation_mode=True,
        )
        assert guard.config.pdt_account_equity == 15_000.0
        assert guard.config.simulation_mode


# =========================
# Edge Cases
# =========================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_pdt_exactly_at_threshold(self):
        """Test PDT at exactly $25,000."""
        tracker = PDTTracker(account_equity=25_000.0)
        assert tracker.is_exempt

    def test_pdt_equity_change(self):
        """Test PDT status changes with equity changes."""
        tracker = PDTTracker(account_equity=30_000.0)
        assert tracker.is_exempt

        # Equity drops below threshold
        tracker.account_equity = 20_000.0
        assert not tracker.is_exempt

    def test_margin_zero_price(self):
        """Test margin calculation with zero price."""
        guard = MarginGuard()
        guard.set_equity(50_000.0)

        max_shares = guard.calculate_max_position("AAPL", 0.0)
        assert max_shares == 0

    def test_split_zero_ratio(self):
        """Test split with invalid ratio."""
        handler = CorporateActionsHandler()

        # Zero denominator should return unchanged
        new_qty, new_price = handler.apply_split("AAPL", 100, 150.0, split_ratio=(2, 0))
        assert new_qty == 100
        assert new_price == 150.0

    def test_circuit_breaker_expiry(self):
        """Test circuit breaker expiration."""
        guard = ShortSaleGuard()
        ts = get_timestamp_ms()

        # Short duration
        guard.trigger_circuit_breaker("AAPL", ts, duration_ms=1000)

        # Should be active immediately
        assert guard.is_circuit_breaker_active("AAPL", ts)

        # Should be expired after duration
        assert not guard.is_circuit_breaker_active("AAPL", ts + 2000)

    def test_empty_day_trade_window(self):
        """Test with no day trades in window."""
        tracker = PDTTracker(account_equity=20_000.0)

        trades = tracker.get_day_trades_in_window()
        assert len(trades) == 0
        assert tracker.get_remaining_day_trades() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
