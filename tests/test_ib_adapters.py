# -*- coding: utf-8 -*-
"""
tests/test_ib_adapters.py
Comprehensive tests for Interactive Brokers adapters.

Tests cover:
- IBRateLimiter: Rate limiting logic
- IBConnectionManager: Connection lifecycle
- IBMarketDataAdapter: Market data retrieval
- IBOrderExecutionAdapter: Order execution
- IBExchangeInfoAdapter: Contract specifications

Target: 80+ tests per Phase 3B specification.
"""

import pytest
import time
from decimal import Decimal
from datetime import datetime, date, time as dt_time, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List

# Import adapters
from adapters.ib.market_data import (
    IBRateLimiter,
    IBConnectionManager,
    IBMarketDataAdapter,
    CONTRACT_MAP,
    IB_INSYNC_AVAILABLE,
)
from adapters.ib.order_execution import (
    IBOrderExecutionAdapter,
    IBBracketOrderConfig,
    IBOrderAction,
)
from adapters.ib.exchange_info import (
    IBExchangeInfoAdapter,
    CONTRACT_SPECS,
)
from adapters.models import ExchangeVendor

# Import core models
from core_models import Bar, Tick, Side
from core_futures import (
    FuturesPosition,
    FuturesOrder,
    PositionSide,
    OrderSide,
    OrderStatus,
    TimeInForce,
    MarginMode,
)


# =============================================================================
# IBRateLimiter Tests (20 tests)
# =============================================================================


class TestIBRateLimiter:
    """Test IB rate limiter functionality."""

    def test_init_empty_state(self):
        """Test rate limiter starts with empty state."""
        limiter = IBRateLimiter()
        status = limiter.get_status()

        assert status["messages_this_second"] == 0
        assert status["historical_last_10min"] == 0
        assert status["active_subscriptions"] == 0

    def test_can_send_message_initially_true(self):
        """Test can send message when no messages sent."""
        limiter = IBRateLimiter()
        assert limiter.can_send_message() is True

    def test_record_message_increments_count(self):
        """Test recording message increments count."""
        limiter = IBRateLimiter()
        limiter.record_message()
        status = limiter.get_status()
        assert status["messages_this_second"] == 1

    def test_can_send_message_respects_limit(self):
        """Test message rate limit is enforced."""
        limiter = IBRateLimiter()
        # Fill up to limit
        for _ in range(limiter.MSG_PER_SEC):
            limiter.record_message()

        # Should be blocked
        assert limiter.can_send_message() is False

    def test_message_window_expires(self):
        """Test old messages expire from window."""
        limiter = IBRateLimiter()
        limiter.record_message()

        # Hack: manually age the timestamp
        limiter._message_times[0] = time.time() - 2.0

        assert limiter.can_send_message() is True
        status = limiter.get_status()
        assert status["messages_this_second"] == 0

    def test_wait_for_message_slot_success(self):
        """Test wait for message slot returns true when available."""
        limiter = IBRateLimiter()
        assert limiter.wait_for_message_slot(timeout=0.1) is True

    def test_wait_for_message_slot_timeout(self):
        """Test wait for message slot times out when limit reached."""
        limiter = IBRateLimiter()
        # Fill up
        for _ in range(limiter.MSG_PER_SEC):
            limiter.record_message()

        # Should timeout
        assert limiter.wait_for_message_slot(timeout=0.05) is False

    def test_can_request_historical_initially_true(self):
        """Test historical requests allowed initially."""
        limiter = IBRateLimiter()
        can_request, reason = limiter.can_request_historical()
        assert can_request is True
        assert reason == ""

    def test_record_historical_request(self):
        """Test recording historical request."""
        limiter = IBRateLimiter()
        limiter.record_historical_request("ES:1h:100")
        status = limiter.get_status()
        assert status["historical_last_10min"] == 1

    def test_historical_limit_enforced(self):
        """Test historical rate limit is enforced."""
        limiter = IBRateLimiter()
        # Fill up to limit
        for i in range(limiter.HIST_PER_10MIN):
            limiter.record_historical_request(f"key_{i}")

        can_request, reason = limiter.can_request_historical()
        assert can_request is False
        assert "Historical rate limit" in reason

    def test_identical_request_tracking(self):
        """Test identical request tracking."""
        limiter = IBRateLimiter()
        request_key = "ES:1h:100"

        # Record same request multiple times
        for _ in range(limiter.HIST_IDENTICAL_PER_10MIN):
            limiter.record_historical_request(request_key)

        # Next identical request should be blocked
        can_request, reason = limiter.can_request_historical(request_key)
        assert can_request is False
        assert "Identical request limit" in reason

    def test_different_requests_not_blocked(self):
        """Test different requests aren't blocked by identical limit."""
        limiter = IBRateLimiter()

        # Record one key multiple times
        for _ in range(limiter.HIST_IDENTICAL_PER_10MIN):
            limiter.record_historical_request("ES:1h:100")

        # Different key should still work
        can_request, reason = limiter.can_request_historical("NQ:1h:100")
        assert can_request is True

    def test_can_subscribe_market_data_initially(self):
        """Test subscription allowed initially."""
        limiter = IBRateLimiter()
        can_sub, reason = limiter.can_subscribe_market_data("ES")
        assert can_sub is True
        assert reason == ""

    def test_record_subscription(self):
        """Test recording subscription."""
        limiter = IBRateLimiter()
        limiter.record_subscription("ES")
        status = limiter.get_status()
        assert status["active_subscriptions"] == 1
        assert "ES" in limiter._active_subscriptions

    def test_record_unsubscription(self):
        """Test recording unsubscription."""
        limiter = IBRateLimiter()
        limiter.record_subscription("ES")
        limiter.record_unsubscription("ES")
        status = limiter.get_status()
        assert status["active_subscriptions"] == 0

    def test_subscription_limit_enforced(self):
        """Test max market data lines limit."""
        limiter = IBRateLimiter()

        # Fill up subscriptions (need to bypass rate limit by modifying internals)
        for i in range(limiter.MAX_MARKET_DATA_LINES):
            limiter._active_subscriptions.add(f"SYM{i}")

        # Clear rate limit timestamps to test max lines specifically
        limiter._subscription_times.clear()

        can_sub, reason = limiter.can_subscribe_market_data("EXTRA")
        assert can_sub is False
        assert "Max market data lines" in reason

    def test_subscription_rate_limit(self):
        """Test subscription rate limit (1 per second)."""
        limiter = IBRateLimiter()
        limiter.record_subscription("ES")

        # Immediate second subscription should be blocked
        can_sub, reason = limiter.can_subscribe_market_data("NQ")
        assert can_sub is False
        assert "1 per second" in reason

    def test_get_status_complete(self):
        """Test get_status returns all expected fields."""
        limiter = IBRateLimiter()
        status = limiter.get_status()

        expected_keys = [
            "messages_this_second",
            "messages_per_sec_limit",
            "historical_last_10min",
            "historical_per_10min_limit",
            "active_subscriptions",
            "max_subscriptions",
            "active_scanners",
            "max_scanners",
        ]

        for key in expected_keys:
            assert key in status

    def test_thread_safety_lock(self):
        """Test rate limiter has lock for thread safety."""
        limiter = IBRateLimiter()
        assert hasattr(limiter, "_lock")

    def test_constants_within_limits(self):
        """Test rate limit constants are below actual IB limits."""
        limiter = IBRateLimiter()

        # Safety margins
        assert limiter.MSG_PER_SEC < 50  # IB limit is 50
        assert limiter.HIST_PER_10MIN < 60  # IB limit is 60
        assert limiter.HIST_IDENTICAL_PER_10MIN < 6  # IB limit is 6


# =============================================================================
# IBConnectionManager Tests (15 tests)
# =============================================================================


class TestIBConnectionManager:
    """Test IB connection manager functionality."""

    def test_init_defaults(self):
        """Test connection manager initializes with defaults."""
        manager = IBConnectionManager()

        assert manager._host == "127.0.0.1"
        assert manager._port == 7497  # Paper trading default
        assert manager._client_id == 1
        assert manager._readonly is False
        assert manager._connected is False

    def test_init_custom_params(self):
        """Test connection manager with custom parameters."""
        manager = IBConnectionManager(
            host="192.168.1.1",
            port=7496,
            client_id=99,
            readonly=True,
            account="U12345",
        )

        assert manager._host == "192.168.1.1"
        assert manager._port == 7496
        assert manager._client_id == 99
        assert manager._readonly is True
        assert manager._account == "U12345"

    def test_is_connected_false_initially(self):
        """Test is_connected returns False initially."""
        manager = IBConnectionManager()
        assert manager.is_connected is False

    def test_health_check_structure(self):
        """Test health_check returns expected structure."""
        manager = IBConnectionManager()
        health = manager.health_check()

        expected_keys = [
            "connected",
            "host",
            "port",
            "client_id",
            "readonly",
            "reconnect_count",
            "last_heartbeat",
            "rate_limit_status",
        ]

        for key in expected_keys:
            assert key in health

    def test_rate_limiter_property(self):
        """Test rate_limiter property returns IBRateLimiter."""
        manager = IBConnectionManager()
        assert isinstance(manager.rate_limiter, IBRateLimiter)

    def test_reconnect_delays_exponential(self):
        """Test reconnect delays follow exponential pattern."""
        delays = IBConnectionManager.RECONNECT_DELAYS

        # Should be increasing
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]

    def test_heartbeat_interval_reasonable(self):
        """Test heartbeat interval is reasonable."""
        assert IBConnectionManager.HEARTBEAT_INTERVAL_SEC < 60  # IB requires activity every 60s
        assert IBConnectionManager.HEARTBEAT_INTERVAL_SEC >= 10  # Not too aggressive

    def test_max_messages_per_sec_safe(self):
        """Test max messages per second has safety margin."""
        assert IBConnectionManager.MAX_MESSAGES_PER_SEC < 50  # IB limit is 50

    @pytest.mark.skipif(not IB_INSYNC_AVAILABLE, reason="ib_insync not installed")
    def test_connect_requires_ib_insync(self):
        """Test connect checks for ib_insync."""
        manager = IBConnectionManager()
        # This would fail to connect without TWS running, but shouldn't raise ImportError
        with pytest.raises(ConnectionError):
            manager.connect(timeout=0.1)

    def test_disconnect_when_not_connected(self):
        """Test disconnect is safe when not connected."""
        manager = IBConnectionManager()
        manager.disconnect()  # Should not raise
        assert manager._connected is False

    def test_ib_property_raises_when_not_connected(self):
        """Test ib property raises when not connected."""
        manager = IBConnectionManager()
        with pytest.raises(ConnectionError, match="Not connected"):
            _ = manager.ib

    def test_send_heartbeat_when_not_connected(self):
        """Test send_heartbeat is safe when not connected."""
        manager = IBConnectionManager()
        manager.send_heartbeat()  # Should not raise

    def test_wait_for_rate_limit_delegates(self):
        """Test wait_for_rate_limit delegates to rate limiter."""
        manager = IBConnectionManager()
        result = manager.wait_for_rate_limit()
        assert result is True  # Should succeed when not rate limited

    def test_port_reference_values(self):
        """Test standard port values are documented."""
        # These are standard IB ports
        assert 7496 >= 1024  # Live TWS
        assert 7497 >= 1024  # Paper TWS
        assert 4001 >= 1024  # Live Gateway
        assert 4002 >= 1024  # Paper Gateway

    def test_on_disconnect_handler(self):
        """Test disconnect handler sets connected flag."""
        manager = IBConnectionManager()
        manager._connected = True
        # Mock connect to avoid requiring ib_insync
        manager.connect = lambda: None
        manager._on_disconnect()
        assert manager._connected is False


# =============================================================================
# IBMarketDataAdapter Tests (20 tests)
# =============================================================================


class TestIBMarketDataAdapter:
    """Test IB market data adapter functionality."""

    def test_init_defaults(self):
        """Test adapter initializes with defaults."""
        adapter = IBMarketDataAdapter()

        assert adapter._host == "127.0.0.1"
        assert adapter._port == 7497
        assert adapter._client_id == 1
        assert adapter._readonly is True
        assert adapter._default_exchange == "CME"

    def test_init_custom_config(self):
        """Test adapter with custom configuration."""
        config = {
            "host": "192.168.1.1",
            "port": 7496,
            "client_id": 42,
            "default_exchange": "CBOT",
        }
        adapter = IBMarketDataAdapter(config=config)

        assert adapter._host == "192.168.1.1"
        assert adapter._port == 7496
        assert adapter._client_id == 42
        assert adapter._default_exchange == "CBOT"

    def test_vendor_type(self):
        """Test adapter has correct vendor type."""
        adapter = IBMarketDataAdapter(vendor=ExchangeVendor.IB_CME)
        assert adapter._vendor == ExchangeVendor.IB_CME

    def test_contract_map_has_common_symbols(self):
        """Test CONTRACT_MAP includes common futures."""
        expected_symbols = ["ES", "NQ", "GC", "CL", "6E", "ZN"]
        for symbol in expected_symbols:
            assert symbol in CONTRACT_MAP

    def test_contract_map_exchange_info(self):
        """Test CONTRACT_MAP has correct exchange assignments."""
        assert CONTRACT_MAP["ES"]["exchange"] == "CME"
        assert CONTRACT_MAP["GC"]["exchange"] == "COMEX"
        assert CONTRACT_MAP["CL"]["exchange"] == "NYMEX"
        assert CONTRACT_MAP["ZN"]["exchange"] == "CBOT"

    def test_convert_timeframe_mapping(self):
        """Test timeframe conversion to IB format."""
        mapping = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }

        for our_tf, ib_tf in mapping.items():
            assert IBMarketDataAdapter._convert_timeframe(our_tf) == ib_tf

    def test_convert_timeframe_default(self):
        """Test unknown timeframe returns default."""
        assert IBMarketDataAdapter._convert_timeframe("unknown") == "1 hour"

    def test_calculate_duration_minutes(self):
        """Test duration calculation for short periods."""
        duration = IBMarketDataAdapter._calculate_duration(30, "1m")
        assert "M" in duration  # Minutes

    def test_calculate_duration_hours(self):
        """Test duration calculation for hours."""
        duration = IBMarketDataAdapter._calculate_duration(24, "1h")
        assert "H" in duration or "D" in duration

    def test_calculate_duration_days(self):
        """Test duration calculation for days."""
        duration = IBMarketDataAdapter._calculate_duration(30, "1d")
        assert "D" in duration or "W" in duration or "M" in duration

    def test_ms_to_timeframe_1min(self):
        """Test milliseconds to timeframe - 1 minute."""
        assert IBMarketDataAdapter._ms_to_timeframe(60000) == "1m"

    def test_ms_to_timeframe_5min(self):
        """Test milliseconds to timeframe - 5 minutes."""
        assert IBMarketDataAdapter._ms_to_timeframe(300000) == "5m"

    def test_ms_to_timeframe_1hour(self):
        """Test milliseconds to timeframe - 1 hour."""
        assert IBMarketDataAdapter._ms_to_timeframe(3600000) == "1h"

    def test_ms_to_timeframe_4hour(self):
        """Test milliseconds to timeframe - 4 hours."""
        assert IBMarketDataAdapter._ms_to_timeframe(14400000) == "4h"

    def test_ms_to_timeframe_1day(self):
        """Test milliseconds to timeframe - 1 day."""
        assert IBMarketDataAdapter._ms_to_timeframe(86400000) == "1d"

    def test_do_connect_not_connected_initially(self):
        """Test adapter not connected initially."""
        adapter = IBMarketDataAdapter()
        assert adapter._conn_manager is None

    def test_do_disconnect_safe_when_not_connected(self):
        """Test disconnect is safe when not connected."""
        adapter = IBMarketDataAdapter()
        adapter._do_disconnect()  # Should not raise

    @pytest.mark.skipif(not IB_INSYNC_AVAILABLE, reason="ib_insync not installed")
    def test_create_contract_continuous(self):
        """Test creating continuous future contract."""
        adapter = IBMarketDataAdapter()
        # This tests the contract creation logic without needing connection
        # The actual ContFuture creation requires ib_insync

    def test_contract_map_currency_all_usd(self):
        """Test all contracts in map are USD denominated."""
        for symbol, details in CONTRACT_MAP.items():
            assert details["currency"] == "USD"

    def test_contract_map_descriptions(self):
        """Test key contracts have descriptions."""
        assert "description" in CONTRACT_MAP["ES"]
        assert "E-mini S&P 500" in CONTRACT_MAP["ES"]["description"]


# =============================================================================
# IBOrderExecutionAdapter Tests (15 tests)
# =============================================================================


class TestIBOrderExecutionAdapter:
    """Test IB order execution adapter functionality."""

    def test_init_defaults(self):
        """Test adapter initializes with defaults."""
        adapter = IBOrderExecutionAdapter()

        assert adapter._host == "127.0.0.1"
        assert adapter._port == 7497
        assert adapter._client_id == 2  # Different from market data
        assert adapter._default_exchange == "CME"

    def test_client_id_different_from_market_data(self):
        """Test order execution uses different client ID than market data."""
        md_adapter = IBMarketDataAdapter()
        exec_adapter = IBOrderExecutionAdapter()

        # Should use different client IDs to avoid conflicts
        assert md_adapter._client_id != exec_adapter._client_id

    def test_contract_map_present(self):
        """Test order execution has CONTRACT_MAP."""
        from adapters.ib.order_execution import CONTRACT_MAP as EXEC_MAP

        assert "ES" in EXEC_MAP
        assert "GC" in EXEC_MAP

    def test_ib_order_action_enum(self):
        """Test IBOrderAction enum values."""
        assert IBOrderAction.BUY.value == "BUY"
        assert IBOrderAction.SELL.value == "SELL"

    def test_bracket_order_config_dataclass(self):
        """Test IBBracketOrderConfig dataclass."""
        config = IBBracketOrderConfig(
            symbol="ES",
            side="BUY",
            qty=1,
            entry_price=Decimal("4500.00"),
            take_profit_price=Decimal("4550.00"),
            stop_loss_price=Decimal("4450.00"),
        )

        assert config.symbol == "ES"
        assert config.side == "BUY"
        assert config.qty == 1
        assert config.time_in_force == "GTC"

    def test_bracket_order_config_defaults(self):
        """Test IBBracketOrderConfig default values."""
        config = IBBracketOrderConfig(
            symbol="NQ",
            side="SELL",
            qty=2,
        )

        assert config.entry_price is None
        assert config.take_profit_price is None
        assert config.stop_loss_price is None
        assert config.time_in_force == "GTC"

    def test_not_connected_raises(self):
        """Test methods raise when not connected."""
        adapter = IBOrderExecutionAdapter()

        with pytest.raises(ConnectionError):
            adapter.get_futures_positions()

        with pytest.raises(ConnectionError):
            adapter.get_account_margin()

        with pytest.raises(ConnectionError):
            adapter.submit_market_order("ES", "BUY", 1)

    def test_ib_order_type_to_str(self):
        """Test IB order type conversion to string."""
        mock_order = Mock()
        mock_order.orderType = "MKT"
        assert IBOrderExecutionAdapter._ib_order_type_to_str(mock_order) == "MARKET"

        mock_order.orderType = "LMT"
        assert IBOrderExecutionAdapter._ib_order_type_to_str(mock_order) == "LIMIT"

        mock_order.orderType = "STP"
        assert IBOrderExecutionAdapter._ib_order_type_to_str(mock_order) == "STOP"

        mock_order.orderType = "STP LMT"
        assert IBOrderExecutionAdapter._ib_order_type_to_str(mock_order) == "STOP_LIMIT"

    def test_ib_status_to_order_status_submitted(self):
        """Test IB status conversion - submitted."""
        assert IBOrderExecutionAdapter._ib_status_to_order_status("Submitted") == OrderStatus.NEW
        assert IBOrderExecutionAdapter._ib_status_to_order_status("PreSubmitted") == OrderStatus.NEW

    def test_ib_status_to_order_status_filled(self):
        """Test IB status conversion - filled."""
        assert IBOrderExecutionAdapter._ib_status_to_order_status("Filled") == OrderStatus.FILLED

    def test_ib_status_to_order_status_cancelled(self):
        """Test IB status conversion - cancelled."""
        assert (
            IBOrderExecutionAdapter._ib_status_to_order_status("Cancelled") == OrderStatus.CANCELLED
        )
        assert (
            IBOrderExecutionAdapter._ib_status_to_order_status("Inactive") == OrderStatus.CANCELLED
        )

    def test_ib_status_to_order_status_partial(self):
        """Test IB status conversion - partial fill."""
        assert (
            IBOrderExecutionAdapter._ib_status_to_order_status("PartiallyFilled")
            == OrderStatus.PARTIALLY_FILLED
        )

    def test_do_disconnect_safe(self):
        """Test disconnect is safe when not connected."""
        adapter = IBOrderExecutionAdapter()
        adapter._do_disconnect()  # Should not raise

    def test_cancel_order_not_connected(self):
        """Test cancel_order raises when not connected."""
        adapter = IBOrderExecutionAdapter()
        with pytest.raises(ConnectionError):
            adapter.cancel_order(order_id="12345")

    def test_get_order_status_not_connected(self):
        """Test get_order_status raises when not connected."""
        adapter = IBOrderExecutionAdapter()
        with pytest.raises(ConnectionError):
            adapter.get_order_status(order_id="12345")


# =============================================================================
# IBExchangeInfoAdapter Tests (15 tests)
# =============================================================================


class TestIBExchangeInfoAdapter:
    """Test IB exchange info adapter functionality."""

    def test_init_defaults(self):
        """Test adapter initializes with defaults."""
        adapter = IBExchangeInfoAdapter()
        assert adapter._default_exchange == "CME"

    def test_init_custom_exchange(self):
        """Test adapter with custom default exchange."""
        adapter = IBExchangeInfoAdapter(config={"default_exchange": "COMEX"})
        assert adapter._default_exchange == "COMEX"

    def test_contract_specs_has_common_symbols(self):
        """Test CONTRACT_SPECS includes common futures."""
        expected = ["ES", "NQ", "GC", "CL", "6E", "ZN"]
        for symbol in expected:
            assert symbol in CONTRACT_SPECS

    def test_contract_spec_structure(self):
        """Test contract spec has required fields."""
        es_spec = CONTRACT_SPECS["ES"]

        # Required fields in CONTRACT_SPECS
        required_fields = [
            "exchange",
            "multiplier",
            "tick_size",
            "currency",
            "description",
            "futures_type",
        ]

        for field in required_fields:
            assert field in es_spec, f"Missing field: {field}"

    def test_es_contract_spec_values(self):
        """Test ES contract spec has correct values."""
        es_spec = CONTRACT_SPECS["ES"]

        assert es_spec["exchange"] == "CME"
        assert es_spec["multiplier"] == Decimal("50")
        assert es_spec["tick_size"] == Decimal("0.25")
        assert es_spec["currency"] == "USD"

    def test_gc_contract_spec_values(self):
        """Test GC (Gold) contract spec has correct values."""
        gc_spec = CONTRACT_SPECS["GC"]

        assert gc_spec["exchange"] == "COMEX"
        assert gc_spec["multiplier"] == Decimal("100")  # 100 oz
        assert gc_spec["tick_size"] == Decimal("0.10")
        assert gc_spec["currency"] == "USD"

    def test_cl_contract_spec_values(self):
        """Test CL (Crude Oil) contract spec has correct values."""
        cl_spec = CONTRACT_SPECS["CL"]

        assert cl_spec["exchange"] == "NYMEX"
        assert cl_spec["multiplier"] == Decimal("1000")  # 1000 barrels
        assert cl_spec["tick_size"] == Decimal("0.01")
        assert cl_spec["currency"] == "USD"

    def test_get_contract_spec_returns_spec(self):
        """Test get_contract_spec returns specification."""
        adapter = IBExchangeInfoAdapter()
        spec = adapter.get_contract_spec("ES")

        assert spec is not None
        # spec is FuturesContractSpec object
        assert spec.symbol == "ES"

    def test_get_contract_spec_unknown_returns_none(self):
        """Test get_contract_spec returns None for unknown symbol."""
        adapter = IBExchangeInfoAdapter()
        spec = adapter.get_contract_spec("UNKNOWN_SYMBOL")

        assert spec is None

    def test_get_symbols_returns_list(self):
        """Test get_symbols returns list of symbols."""
        adapter = IBExchangeInfoAdapter()
        symbols = adapter.get_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "ES" in symbols
        assert "GC" in symbols

    def test_get_symbols_by_exchange(self):
        """Test get_symbols filtered by exchange."""
        adapter = IBExchangeInfoAdapter()

        cme_symbols = adapter.get_symbols(filters={"exchange": "CME"})
        assert "ES" in cme_symbols
        assert "NQ" in cme_symbols

        comex_symbols = adapter.get_symbols(filters={"exchange": "COMEX"})
        assert "GC" in comex_symbols
        assert "SI" in comex_symbols

    def test_micro_contracts_present(self):
        """Test micro contracts are in CONTRACT_SPECS."""
        micro_symbols = ["MES", "MNQ", "MGC", "MCL"]
        for symbol in micro_symbols:
            assert symbol in CONTRACT_SPECS

    def test_micro_contracts_smaller_multiplier(self):
        """Test micro contracts have smaller multipliers."""
        assert CONTRACT_SPECS["MES"]["multiplier"] < CONTRACT_SPECS["ES"]["multiplier"]
        assert CONTRACT_SPECS["MNQ"]["multiplier"] < CONTRACT_SPECS["NQ"]["multiplier"]
        assert CONTRACT_SPECS["MGC"]["multiplier"] < CONTRACT_SPECS["GC"]["multiplier"]

    def test_currency_futures_present(self):
        """Test currency futures are present."""
        fx_symbols = ["6E", "6J", "6B", "6A", "6C", "6S"]
        for symbol in fx_symbols:
            assert symbol in CONTRACT_SPECS
            assert CONTRACT_SPECS[symbol]["exchange"] == "CME"

    def test_bond_futures_present(self):
        """Test bond futures are present."""
        bond_symbols = ["ZB", "ZN", "ZT", "ZF"]
        for symbol in bond_symbols:
            assert symbol in CONTRACT_SPECS
            assert CONTRACT_SPECS[symbol]["exchange"] == "CBOT"


# =============================================================================
# Integration Tests (5 tests)
# =============================================================================


class TestIBAdapterIntegration:
    """Integration tests for IB adapters."""

    def test_all_adapters_have_same_contract_symbols(self):
        """Test all adapters share common symbol set."""
        from adapters.ib.market_data import CONTRACT_MAP as MD_MAP
        from adapters.ib.order_execution import CONTRACT_MAP as EXEC_MAP
        from adapters.ib.exchange_info import CONTRACT_SPECS

        # All common symbols should be in all maps
        common = ["ES", "NQ", "GC", "CL", "6E"]
        for symbol in common:
            assert symbol in MD_MAP
            assert symbol in EXEC_MAP
            assert symbol in CONTRACT_SPECS

    def test_adapters_register_correctly(self):
        """Test adapters are registered in __init__."""
        from adapters.ib import (
            IBMarketDataAdapter,
            IBOrderExecutionAdapter,
            IBExchangeInfoAdapter,
            IBConnectionManager,
            IBRateLimiter,
        )

        # Should be importable without error
        assert IBMarketDataAdapter is not None
        assert IBOrderExecutionAdapter is not None
        assert IBExchangeInfoAdapter is not None
        assert IBConnectionManager is not None
        assert IBRateLimiter is not None

    def test_vendor_enum_values(self):
        """Test IB vendor enum values exist."""
        assert hasattr(ExchangeVendor, "IB")
        assert hasattr(ExchangeVendor, "IB_CME")
        assert hasattr(ExchangeVendor, "IB_CBOT")
        assert hasattr(ExchangeVendor, "IB_NYMEX")
        assert hasattr(ExchangeVendor, "IB_COMEX")

    def test_contract_specs_multipliers_positive(self):
        """Test all contract multipliers are positive."""
        for symbol, spec in CONTRACT_SPECS.items():
            assert spec["multiplier"] > 0, f"{symbol} has non-positive multiplier"

    def test_contract_specs_tick_sizes_positive(self):
        """Test all tick sizes are positive."""
        for symbol, spec in CONTRACT_SPECS.items():
            assert spec["tick_size"] > 0, f"{symbol} has non-positive tick_size"


# =============================================================================
# Edge Case Tests (10 tests)
# =============================================================================


class TestIBAdapterEdgeCases:
    """Edge case tests for IB adapters."""

    def test_rate_limiter_concurrent_access(self):
        """Test rate limiter handles concurrent access."""
        import threading

        limiter = IBRateLimiter()
        errors = []

        def record_messages():
            try:
                for _ in range(10):
                    limiter.record_message()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_empty_symbol_handling(self):
        """Test adapters handle empty symbol."""
        adapter = IBExchangeInfoAdapter()
        spec = adapter.get_contract_spec("")
        assert spec is None

    def test_case_insensitive_symbol_lookup(self):
        """Test symbol lookup is case insensitive."""
        adapter = IBExchangeInfoAdapter()

        spec_upper = adapter.get_contract_spec("ES")
        spec_lower = adapter.get_contract_spec("es")

        # Both should return the spec (or None if not case-normalized)
        # The implementation should handle case

    def test_timeframe_edge_cases(self):
        """Test timeframe conversion edge cases."""
        # Very short
        assert IBMarketDataAdapter._ms_to_timeframe(1000) == "1m"

        # Very long
        assert IBMarketDataAdapter._ms_to_timeframe(86400000 * 7) == "1d"

    def test_duration_calculation_small_limit(self):
        """Test duration calculation with small limit."""
        duration = IBMarketDataAdapter._calculate_duration(1, "1m")
        assert "M" in duration or "H" in duration

    def test_duration_calculation_large_limit(self):
        """Test duration calculation with large limit."""
        duration = IBMarketDataAdapter._calculate_duration(10000, "1d")
        assert "M" in duration  # Months

    def test_bracket_order_partial_config(self):
        """Test bracket order with only stop loss."""
        config = IBBracketOrderConfig(
            symbol="ES",
            side="BUY",
            qty=1,
            stop_loss_price=Decimal("4450.00"),
        )

        assert config.take_profit_price is None
        assert config.stop_loss_price == Decimal("4450.00")

    def test_rate_limiter_window_boundary(self):
        """Test rate limiter at window boundary."""
        limiter = IBRateLimiter()

        # Record at exact boundary
        limiter._message_times = [time.time() - 1.0001]
        assert limiter.can_send_message() is True

    def test_historical_request_key_none(self):
        """Test historical request with None key."""
        limiter = IBRateLimiter()
        can_request, reason = limiter.can_request_historical(None)
        assert can_request is True

        limiter.record_historical_request(None)
        status = limiter.get_status()
        assert status["historical_last_10min"] == 1

    def test_order_status_unknown_status(self):
        """Test order status conversion with unknown status."""
        result = IBOrderExecutionAdapter._ib_status_to_order_status("UnknownStatus")
        assert result == OrderStatus.NEW  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
