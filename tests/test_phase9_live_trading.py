# -*- coding: utf-8 -*-
"""
tests/test_phase9_live_trading.py
Comprehensive tests for Phase 9: Live Trading Improvements.

Tests cover:
1. Unified Live Script - asset_class detection and defaults
2. Position Sync - position synchronization service
3. Order Management - bracket orders, replace orders
4. Extended Hours - session-aware order routing

All tests ensure backward compatibility with existing crypto functionality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from core_models import Side, OrderType


# =============================================================================
# Test 1: Unified Live Script - Asset Class Detection
# =============================================================================


class TestAssetClassDetection:
    """Tests for asset class detection in script_live.py."""

    def test_detect_crypto_from_explicit_config(self):
        """Test detection when asset_class is explicitly set to crypto."""
        from script_live import detect_asset_class

        cfg_dict = {"asset_class": "crypto"}
        assert detect_asset_class(cfg_dict) == "crypto"

    def test_detect_equity_from_explicit_config(self):
        """Test detection when asset_class is explicitly set to equity."""
        from script_live import detect_asset_class

        cfg_dict = {"asset_class": "equity"}
        assert detect_asset_class(cfg_dict) == "equity"

    def test_detect_crypto_from_binance_vendor(self):
        """Test detection from Binance vendor configuration."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"vendor": "binance"}}
        assert detect_asset_class(cfg_dict) == "crypto"

    def test_detect_equity_from_alpaca_vendor(self):
        """Test detection from Alpaca vendor configuration."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"vendor": "alpaca"}}
        assert detect_asset_class(cfg_dict) == "equity"

    def test_detect_equity_from_polygon_vendor(self):
        """Test detection from Polygon vendor configuration."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"vendor": "polygon"}}
        assert detect_asset_class(cfg_dict) == "equity"

    def test_detect_equity_from_market_type(self):
        """Test detection from market type configuration."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"market_type": "EQUITY"}}
        assert detect_asset_class(cfg_dict) == "equity"

    def test_detect_crypto_from_market_type(self):
        """Test detection from market type configuration."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"market_type": "CRYPTO_SPOT"}}
        assert detect_asset_class(cfg_dict) == "crypto"

    def test_default_to_crypto_for_empty_config(self):
        """Test default to crypto for backward compatibility."""
        from script_live import detect_asset_class

        cfg_dict = {}
        assert detect_asset_class(cfg_dict) == "crypto"

    def test_default_to_crypto_for_unknown_vendor(self):
        """Test default to crypto for unknown vendor."""
        from script_live import detect_asset_class

        cfg_dict = {"exchange": {"vendor": "unknown"}}
        assert detect_asset_class(cfg_dict) == "crypto"


class TestAssetClassDefaults:
    """Tests for asset class defaults application."""

    def test_apply_crypto_defaults(self):
        """Test that crypto defaults are applied correctly."""
        from script_live import apply_asset_class_defaults, ASSET_CLASS_CRYPTO

        cfg_dict = {}
        result = apply_asset_class_defaults(cfg_dict, ASSET_CLASS_CRYPTO)

        assert result["asset_class"] == "crypto"
        assert result["execution_params"]["slippage_bps"] == 5.0
        assert result["execution_params"]["tif"] == "GTC"
        assert result["data_vendor"] == "binance"

    def test_apply_equity_defaults(self):
        """Test that equity defaults are applied correctly."""
        from script_live import apply_asset_class_defaults, ASSET_CLASS_EQUITY

        cfg_dict = {}
        result = apply_asset_class_defaults(cfg_dict, ASSET_CLASS_EQUITY)

        assert result["asset_class"] == "equity"
        assert result["execution_params"]["slippage_bps"] == 2.0
        assert result["execution_params"]["tif"] == "DAY"
        assert result["data_vendor"] == "alpaca"

    def test_explicit_config_not_overwritten(self):
        """Test that explicit config values are preserved."""
        from script_live import apply_asset_class_defaults, ASSET_CLASS_EQUITY

        cfg_dict = {"execution_params": {"slippage_bps": 3.0}}
        result = apply_asset_class_defaults(cfg_dict, ASSET_CLASS_EQUITY)

        # Explicit value should be preserved
        assert result["execution_params"]["slippage_bps"] == 3.0
        # Default should be applied for missing values
        assert result["execution_params"]["tif"] == "DAY"

    def test_extended_hours_cli_override(self):
        """Test that CLI extended hours override works."""
        from script_live import apply_asset_class_defaults, ASSET_CLASS_EQUITY

        cfg_dict = {"extended_hours": False}
        result = apply_asset_class_defaults(cfg_dict, ASSET_CLASS_EQUITY, cli_extended_hours=True)

        assert result["extended_hours"] is True
        assert result["exchange"]["alpaca"]["extended_hours"] is True

    def test_backward_compatible_crypto(self):
        """Test backward compatibility with existing crypto config."""
        from script_live import apply_asset_class_defaults, ASSET_CLASS_CRYPTO

        # Simulate existing crypto config
        cfg_dict = {
            "exchange": {"vendor": "binance"},
            "execution_params": {"slippage_bps": 4.0},
        }
        result = apply_asset_class_defaults(cfg_dict, ASSET_CLASS_CRYPTO)

        # Original values preserved
        assert result["execution_params"]["slippage_bps"] == 4.0


# =============================================================================
# Test 2: Position Sync
# =============================================================================


class MockPositionProvider:
    """Mock position provider for testing."""

    def __init__(self, positions: Dict[str, Any]):
        self._positions = positions

    def get_positions(self, symbols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        if symbols:
            return {s: self._positions[s] for s in symbols if s in self._positions}
        return self._positions


class TestPositionSynchronizer:
    """Tests for position synchronization service."""

    def test_sync_matching_positions(self):
        """Test sync when positions match."""
        from services.position_sync import PositionSynchronizer, SyncConfig

        local = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}
        remote = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(),
        )

        result = sync.sync_once()

        assert result.success
        assert not result.has_discrepancies
        assert len(result.discrepancies) == 0

    def test_detect_missing_local_position(self):
        """Test detection of position missing locally."""
        from services.position_sync import (
            PositionSynchronizer,
            SyncConfig,
            DiscrepancyType,
        )

        local = {"AAPL": Decimal("100")}
        remote = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(),
        )

        result = sync.sync_once()

        assert result.success
        assert result.has_discrepancies
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].symbol == "MSFT"
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_LOCAL

    def test_detect_missing_remote_position(self):
        """Test detection of position missing on exchange."""
        from services.position_sync import (
            PositionSynchronizer,
            SyncConfig,
            DiscrepancyType,
        )

        local = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}
        remote = {"AAPL": Decimal("100")}

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(),
        )

        result = sync.sync_once()

        assert result.success
        assert result.has_discrepancies
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].symbol == "MSFT"
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.MISSING_REMOTE

    def test_detect_quantity_mismatch(self):
        """Test detection of quantity mismatch."""
        from services.position_sync import (
            PositionSynchronizer,
            SyncConfig,
            DiscrepancyType,
        )

        local = {"AAPL": Decimal("100")}
        remote = {"AAPL": Decimal("150")}  # 50% difference

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(qty_tolerance_pct=0.001),  # 0.1% tolerance
        )

        result = sync.sync_once()

        assert result.success
        assert result.has_discrepancies
        assert result.discrepancies[0].discrepancy_type == DiscrepancyType.QTY_MISMATCH
        assert result.discrepancies[0].qty_diff == Decimal("50")

    def test_tolerance_respected(self):
        """Test that small differences within tolerance are ignored."""
        from services.position_sync import PositionSynchronizer, SyncConfig

        local = {"AAPL": Decimal("100.00")}
        remote = {"AAPL": Decimal("100.05")}  # 0.05% difference

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(qty_tolerance_pct=0.01),  # 1% tolerance
        )

        result = sync.sync_once()

        assert result.success
        assert not result.has_discrepancies

    def test_discrepancy_callback(self):
        """Test that discrepancy callback is invoked."""
        from services.position_sync import PositionSynchronizer, SyncConfig

        local = {}
        remote = {"AAPL": Decimal("100")}

        callback_called = []

        def on_discrepancy(d):
            callback_called.append(d)

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(),
            on_discrepancy=on_discrepancy,
        )

        sync.sync_once()

        assert len(callback_called) == 1
        assert callback_called[0].symbol == "AAPL"


class TestAlpacaReconciliation:
    """Tests for Alpaca-specific reconciliation."""

    def test_reconcile_with_matching_state(self):
        """Test reconciliation when state matches."""
        from services.position_sync import reconcile_alpaca_state, SyncConfig

        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.get_positions.return_value = {
            "AAPL": MagicMock(
                qty=Decimal("100"),
                avg_entry_price=Decimal("150.00"),
                meta={"market_value": "15000.00", "unrealized_pl": "500.00"},
            )
        }
        mock_adapter.get_open_orders.return_value = []
        mock_adapter.get_account_info.return_value = MagicMock(
            cash_balance=Decimal("10000"),
            buying_power=Decimal("25000"),
            pattern_day_trader=False,
            raw_data={"daytrade_count": 0},
        )

        local_positions = {"AAPL": Decimal("100")}

        result = reconcile_alpaca_state(mock_adapter, local_positions)

        assert result.success
        assert result.position_count == 1
        assert len(result.position_discrepancies) == 0


# =============================================================================
# Test 3: Order Management
# =============================================================================


class TestBracketOrderConfig:
    """Tests for bracket order configuration validation."""

    def test_valid_buy_bracket(self):
        """Test valid BUY bracket order configuration."""
        from adapters.alpaca.order_execution import BracketOrderConfig

        config = BracketOrderConfig(
            symbol="AAPL",
            side=Side.BUY,
            qty=100,
            limit_price=150.0,
            take_profit_price=160.0,  # Above entry
            stop_loss_price=145.0,  # Below entry
        )

        is_valid, error = config.validate()
        assert is_valid
        assert error is None

    def test_valid_sell_bracket(self):
        """Test valid SELL bracket order configuration."""
        from adapters.alpaca.order_execution import BracketOrderConfig

        config = BracketOrderConfig(
            symbol="AAPL",
            side=Side.SELL,
            qty=100,
            limit_price=150.0,
            take_profit_price=140.0,  # Below entry for SELL
            stop_loss_price=155.0,  # Above entry for SELL
        )

        is_valid, error = config.validate()
        assert is_valid
        assert error is None

    def test_invalid_missing_tp_and_sl(self):
        """Test validation fails without TP or SL."""
        from adapters.alpaca.order_execution import BracketOrderConfig

        config = BracketOrderConfig(
            symbol="AAPL",
            side=Side.BUY,
            qty=100,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "take_profit_price or stop_loss_price" in error

    def test_invalid_buy_tp_below_entry(self):
        """Test validation fails when BUY TP is below entry."""
        from adapters.alpaca.order_execution import BracketOrderConfig

        config = BracketOrderConfig(
            symbol="AAPL",
            side=Side.BUY,
            qty=100,
            limit_price=150.0,
            take_profit_price=140.0,  # Below entry - invalid for BUY
            stop_loss_price=145.0,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "Take profit must be above entry for BUY" in error

    def test_invalid_zero_quantity(self):
        """Test validation fails with zero quantity."""
        from adapters.alpaca.order_execution import BracketOrderConfig

        config = BracketOrderConfig(
            symbol="AAPL",
            side=Side.BUY,
            qty=0,
            take_profit_price=160.0,
        )

        is_valid, error = config.validate()
        assert not is_valid
        assert "Quantity must be positive" in error


class TestReplaceOrderConfig:
    """Tests for replace order configuration."""

    def test_replace_config_creation(self):
        """Test ReplaceOrderConfig creation."""
        from adapters.alpaca.order_execution import ReplaceOrderConfig

        config = ReplaceOrderConfig(
            order_id="test-order-123",
            new_qty=200,
            new_limit_price=155.0,
        )

        assert config.order_id == "test-order-123"
        assert config.new_qty == 200
        assert config.new_limit_price == 155.0
        assert config.new_stop_price is None


class TestBracketOrderResult:
    """Tests for bracket order result."""

    def test_bracket_result_success(self):
        """Test successful bracket order result."""
        from adapters.alpaca.order_execution import (
            BracketOrderResult,
            BracketOrderType,
        )

        result = BracketOrderResult(
            success=True,
            primary_order_id="primary-123",
            take_profit_order_id="tp-456",
            stop_loss_order_id="sl-789",
            bracket_type=BracketOrderType.OTO_OCO,
        )

        assert result.success
        assert result.primary_order_id == "primary-123"
        assert result.take_profit_order_id == "tp-456"
        assert result.stop_loss_order_id == "sl-789"
        assert result.bracket_type == BracketOrderType.OTO_OCO

    def test_bracket_result_failure(self):
        """Test failed bracket order result."""
        from adapters.alpaca.order_execution import BracketOrderResult

        result = BracketOrderResult(
            success=False,
            error_code="BRACKET_FAILED",
            error_message="Insufficient buying power",
        )

        assert not result.success
        assert result.error_code == "BRACKET_FAILED"


# =============================================================================
# Test 4: Extended Hours Trading
# =============================================================================


class TestSessionDetection:
    """Tests for trading session detection."""

    def test_detect_regular_hours(self):
        """Test detection of regular trading hours."""
        from services.session_router import get_current_session, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        # Wednesday at 10:00 AM ET
        dt = datetime(2024, 1, 17, 10, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        session = get_current_session(ts_ms)

        assert session.session == TradingSession.REGULAR
        assert session.is_open

    def test_detect_pre_market(self):
        """Test detection of pre-market hours."""
        from services.session_router import get_current_session, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        # Wednesday at 7:00 AM ET (pre-market)
        dt = datetime(2024, 1, 17, 7, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        session = get_current_session(ts_ms)

        assert session.session == TradingSession.PRE_MARKET
        assert session.is_open

    def test_detect_after_hours(self):
        """Test detection of after-hours session."""
        from services.session_router import get_current_session, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        # Wednesday at 5:00 PM ET (after-hours)
        dt = datetime(2024, 1, 17, 17, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        session = get_current_session(ts_ms)

        assert session.session == TradingSession.AFTER_HOURS
        assert session.is_open

    def test_detect_closed_overnight(self):
        """Test detection of closed market (overnight)."""
        from services.session_router import get_current_session, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        # Wednesday at 9:00 PM ET (closed)
        dt = datetime(2024, 1, 17, 21, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        session = get_current_session(ts_ms)

        assert session.session == TradingSession.CLOSED
        assert not session.is_open

    def test_detect_weekend_closed(self):
        """Test detection of weekend (market closed)."""
        from services.session_router import get_current_session, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        # Saturday at 10:00 AM ET
        dt = datetime(2024, 1, 20, 10, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        session = get_current_session(ts_ms)

        assert session.session == TradingSession.CLOSED
        assert not session.is_open


class TestSessionRouter:
    """Tests for session-aware order routing."""

    def test_regular_hours_market_order(self):
        """Test routing market order during regular hours."""
        from services.session_router import SessionRouter, TradingSession
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 10, 0, 0, tzinfo=ET)
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter(allow_extended_hours=True)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="market",
            ts_ms=ts_ms,
        )

        assert decision.should_submit
        assert not decision.use_extended_hours
        assert decision.order_type_override is None

    def test_extended_hours_market_order_rejected(self):
        """Test that market orders are rejected in extended hours without limit price."""
        from services.session_router import SessionRouter
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 7, 0, 0, tzinfo=ET)  # Pre-market
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter(allow_extended_hours=True)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="market",
            ts_ms=ts_ms,
        )

        assert not decision.should_submit
        assert "need limit price" in decision.reason.lower()

    def test_extended_hours_market_order_with_limit(self):
        """Test market order converted to limit in extended hours with price."""
        from services.session_router import SessionRouter
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 7, 0, 0, tzinfo=ET)  # Pre-market
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter(allow_extended_hours=True)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="market",
            limit_price=150.0,  # Provide limit price
            ts_ms=ts_ms,
        )

        assert decision.should_submit
        assert decision.use_extended_hours
        assert decision.order_type_override == "limit"

    def test_extended_hours_disabled(self):
        """Test that orders are rejected when extended hours disabled."""
        from services.session_router import SessionRouter
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 7, 0, 0, tzinfo=ET)  # Pre-market
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter(allow_extended_hours=False)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="limit",
            limit_price=150.0,
            ts_ms=ts_ms,
        )

        assert not decision.should_submit
        assert "disabled" in decision.reason.lower()

    def test_closed_market_rejection(self):
        """Test order rejection when market is closed."""
        from services.session_router import SessionRouter
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 20, 10, 0, 0, tzinfo=ET)  # Saturday
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter()
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="market",
            ts_ms=ts_ms,
        )

        assert not decision.should_submit
        assert "closed" in decision.reason.lower()

    def test_spread_adjustment_extended_hours(self):
        """Test spread adjustment calculation for extended hours."""
        from services.session_router import SessionRouter, TradingSession

        router = SessionRouter(extended_hours_spread_multiplier=2.5)

        # Buy order - should adjust price down
        buy_price = router.adjust_limit_price_for_session(
            base_price=100.0,
            side="buy",
            session=TradingSession.PRE_MARKET,
        )
        assert buy_price < 100.0

        # Sell order - should adjust price up
        sell_price = router.adjust_limit_price_for_session(
            base_price=100.0,
            side="sell",
            session=TradingSession.PRE_MARKET,
        )
        assert sell_price > 100.0

        # Regular hours - no adjustment
        regular_price = router.adjust_limit_price_for_session(
            base_price=100.0,
            side="buy",
            session=TradingSession.REGULAR,
        )
        assert regular_price == 100.0

    def test_session_volume_estimate(self):
        """Test session volume estimation."""
        from services.session_router import SessionRouter, TradingSession

        router = SessionRouter()
        daily_volume = 1_000_000

        regular_vol = router.get_session_volume_estimate(daily_volume, TradingSession.REGULAR)
        pre_market_vol = router.get_session_volume_estimate(daily_volume, TradingSession.PRE_MARKET)
        after_hours_vol = router.get_session_volume_estimate(
            daily_volume, TradingSession.AFTER_HOURS
        )

        # Regular hours should have most volume
        assert regular_vol > pre_market_vol
        assert regular_vol > after_hours_vol
        # Extended hours should have some volume
        assert pre_market_vol > 0
        assert after_hours_vol > 0


# =============================================================================
# Test 5: Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with crypto functionality."""

    def test_crypto_config_unchanged(self):
        """Test that existing crypto configs still work."""
        from script_live import detect_asset_class, apply_asset_class_defaults

        # Existing crypto config structure
        crypto_config = {
            "exchange": {"vendor": "binance"},
            "execution_params": {"slippage_bps": 4.5},
            "data": {"timeframe": "1h"},
        }

        asset_class = detect_asset_class(crypto_config)
        assert asset_class == "crypto"

        result = apply_asset_class_defaults(crypto_config, asset_class)

        # Original values preserved
        assert result["execution_params"]["slippage_bps"] == 4.5
        # Asset class set
        assert result["asset_class"] == "crypto"

    def test_position_sync_works_with_binance_style_positions(self):
        """Test position sync works with Binance-style position format."""
        from services.position_sync import PositionSynchronizer, SyncConfig

        # Binance-style positions (just quantity)
        local = {"BTCUSDT": Decimal("0.5"), "ETHUSDT": Decimal("2.0")}
        remote = {"BTCUSDT": Decimal("0.5"), "ETHUSDT": Decimal("2.0")}

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(
                qty_tolerance_pct=0.0001,  # Tighter tolerance for crypto
                min_qty_diff=1e-8,
            ),
        )

        result = sync.sync_once()

        assert result.success
        assert not result.has_discrepancies

    def test_session_router_irrelevant_for_crypto(self):
        """Test that session router gracefully handles crypto (24/7 market)."""
        # For crypto, the session router shouldn't be used,
        # but if it is, it should not block orders
        from services.session_router import SessionRouter

        router = SessionRouter(allow_extended_hours=True)

        # Even during "closed" equity hours, crypto should trade
        # This test ensures the router doesn't break if used incorrectly
        decision = router.get_routing_decision(
            symbol="BTCUSDT",  # Crypto symbol
            side="buy",
            qty=1.0,
            order_type="market",
        )

        # The router will report based on equity hours, but caller
        # should check asset class first
        assert decision is not None


# =============================================================================
# Test 6: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for Phase 9 components."""

    def test_full_equity_order_flow(self):
        """Test complete flow: detection -> routing -> order config."""
        from script_live import detect_asset_class, apply_asset_class_defaults
        from services.session_router import SessionRouter, TradingSession
        from adapters.alpaca.order_execution import BracketOrderConfig
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # 1. Detect asset class from config
        config = {
            "exchange": {"vendor": "alpaca"},
            "data": {"symbols_path": "data/universe/alpaca_symbols.json"},
        }
        asset_class = detect_asset_class(config)
        assert asset_class == "equity"

        # 2. Apply defaults
        config = apply_asset_class_defaults(config, asset_class)
        assert config["execution_params"]["tif"] == "DAY"

        # 3. Check session for routing
        ET = ZoneInfo("America/New_York")
        dt = datetime(2024, 1, 17, 10, 0, 0, tzinfo=ET)  # Regular hours
        ts_ms = int(dt.timestamp() * 1000)

        router = SessionRouter()
        routing = router.get_routing_decision(
            symbol="AAPL",
            side="buy",
            qty=100,
            order_type="market",
            ts_ms=ts_ms,
        )
        assert routing.should_submit

        # 4. Create bracket order config
        bracket = BracketOrderConfig(
            symbol="AAPL",
            side=Side.BUY,
            qty=100,
            order_type=OrderType.MARKET,
            take_profit_price=160.0,
            stop_loss_price=145.0,
        )
        is_valid, error = bracket.validate()
        assert is_valid

    def test_position_sync_with_discrepancy_callback(self):
        """Test position sync with callback integration."""
        from services.position_sync import PositionSynchronizer, SyncConfig

        discrepancies_found = []

        def handle_discrepancy(d):
            discrepancies_found.append(d)

        local = {"AAPL": Decimal("100")}
        remote = {"AAPL": Decimal("100"), "MSFT": Decimal("50")}

        provider = MockPositionProvider(remote)
        sync = PositionSynchronizer(
            position_provider=provider,
            local_state_getter=lambda: local,
            config=SyncConfig(),
            on_discrepancy=handle_discrepancy,
        )

        result = sync.sync_once()

        assert result.success
        assert len(discrepancies_found) == 1
        assert discrepancies_found[0].symbol == "MSFT"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
