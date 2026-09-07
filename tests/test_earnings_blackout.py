# -*- coding: utf-8 -*-
"""
Tests for earnings-based trading filters.

Verifies:
1. EarningsConfig model validation
2. Earnings blackout window computation
3. Earnings mask calculation
4. Integration with no_trade_mask
5. Backward compatibility (crypto not affected)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from no_trade_config import EarningsConfig, NoTradeConfig


# ==============================================================================
# Test EarningsConfig Model
# ==============================================================================


class TestEarningsConfig:
    """Tests for EarningsConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = EarningsConfig()
        assert cfg.enabled is False
        assert cfg.pre_earnings_bars == 2
        assert cfg.post_earnings_bars == 1
        assert cfg.symbols == []
        assert cfg.cache_ttl_sec == 3600

    def test_custom_values(self):
        """Test custom configuration values."""
        cfg = EarningsConfig(
            enabled=True,
            pre_earnings_bars=5,
            post_earnings_bars=3,
            symbols=["AAPL", "MSFT"],
            cache_ttl_sec=7200,
        )
        assert cfg.enabled is True
        assert cfg.pre_earnings_bars == 5
        assert cfg.post_earnings_bars == 3
        assert cfg.symbols == ["AAPL", "MSFT"]
        assert cfg.cache_ttl_sec == 7200

    def test_no_trade_config_includes_earnings(self):
        """Test that NoTradeConfig includes earnings section."""
        cfg = NoTradeConfig()
        assert hasattr(cfg, "earnings")
        assert isinstance(cfg.earnings, EarningsConfig)

    def test_no_trade_config_with_earnings_dict(self):
        """Test NoTradeConfig with earnings dict."""
        cfg = NoTradeConfig(
            earnings={
                "enabled": True,
                "pre_earnings_bars": 3,
            }
        )
        assert cfg.earnings.enabled is True
        assert cfg.earnings.pre_earnings_bars == 3


# ==============================================================================
# Test Blackout Window Building
# ==============================================================================


class TestBuildEarningsBlackoutWindows:
    """Tests for _build_earnings_blackout_windows function."""

    def test_empty_events(self):
        """Test with no earnings events."""
        from no_trade import _build_earnings_blackout_windows

        result = _build_earnings_blackout_windows({}, 2, 1)
        assert result == []

    def test_single_event(self):
        """Test with a single earnings event."""
        from no_trade import _build_earnings_blackout_windows

        events = {
            "AAPL": [
                {
                    "symbol": "AAPL",
                    "report_date": "2025-01-15",
                    "is_confirmed": True,
                }
            ]
        }

        bar_duration_ms = 4 * 60 * 60 * 1000  # 4 hours
        windows = _build_earnings_blackout_windows(events, 2, 1, bar_duration_ms)

        assert len(windows) == 1
        window = windows[0]
        assert window["symbol"] == "AAPL"
        assert window["reason"] == "earnings_blackout"
        assert window["report_date"] == "2025-01-15"
        assert window["start_ts_ms"] < window["end_ts_ms"]

    def test_multiple_events_same_symbol(self):
        """Test with multiple earnings events for same symbol."""
        from no_trade import _build_earnings_blackout_windows

        events = {
            "AAPL": [
                {"symbol": "AAPL", "report_date": "2025-01-15"},
                {"symbol": "AAPL", "report_date": "2025-04-15"},
            ]
        }

        windows = _build_earnings_blackout_windows(events, 2, 1)
        assert len(windows) == 2

    def test_multiple_symbols(self):
        """Test with multiple symbols."""
        from no_trade import _build_earnings_blackout_windows

        events = {
            "AAPL": [{"symbol": "AAPL", "report_date": "2025-01-15"}],
            "MSFT": [{"symbol": "MSFT", "report_date": "2025-01-20"}],
        }

        windows = _build_earnings_blackout_windows(events, 2, 1)
        assert len(windows) == 2

        symbols = {w["symbol"] for w in windows}
        assert symbols == {"AAPL", "MSFT"}

    def test_window_duration(self):
        """Test that window duration is correct."""
        from no_trade import _build_earnings_blackout_windows

        events = {"AAPL": [{"symbol": "AAPL", "report_date": "2025-01-15"}]}

        bar_duration_ms = 4 * 60 * 60 * 1000  # 4 hours
        pre_bars = 3
        post_bars = 2
        windows = _build_earnings_blackout_windows(events, pre_bars, post_bars, bar_duration_ms)

        window = windows[0]
        expected_duration = (pre_bars + post_bars) * bar_duration_ms
        actual_duration = window["end_ts_ms"] - window["start_ts_ms"]
        assert actual_duration == expected_duration

    def test_invalid_date_format(self):
        """Test handling of invalid date format."""
        from no_trade import _build_earnings_blackout_windows

        events = {"AAPL": [{"symbol": "AAPL", "report_date": "invalid-date"}]}

        # Should not raise, just skip invalid entries
        windows = _build_earnings_blackout_windows(events, 2, 1)
        assert len(windows) == 0

    def test_missing_report_date(self):
        """Test handling of missing report_date."""
        from no_trade import _build_earnings_blackout_windows

        events = {"AAPL": [{"symbol": "AAPL", "report_date": None}]}

        windows = _build_earnings_blackout_windows(events, 2, 1)
        assert len(windows) == 0


# ==============================================================================
# Test Earnings Blackout Mask
# ==============================================================================


class TestInEarningsBlackout:
    """Tests for _in_earnings_blackout function."""

    def test_empty_timestamps(self):
        """Test with empty timestamps."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([], dtype=np.int64)
        mask = _in_earnings_blackout(ts_ms, None, [])
        assert len(mask) == 0

    def test_no_windows(self):
        """Test with no blackout windows."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000000, 2000000, 3000000], dtype=np.int64)
        mask = _in_earnings_blackout(ts_ms, None, [])
        assert mask.all() == False

    def test_single_window_global(self):
        """Test with a single global (no symbol) window."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        windows = [{"start_ts_ms": 2000, "end_ts_ms": 4000}]

        mask = _in_earnings_blackout(ts_ms, None, windows)

        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_per_symbol_window(self):
        """Test with per-symbol windows."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
        symbols = pd.Series(["AAPL", "MSFT", "AAPL", "MSFT"])
        windows = [
            {"start_ts_ms": 1500, "end_ts_ms": 2500, "symbol": "AAPL"},
        ]

        mask = _in_earnings_blackout(ts_ms, symbols, windows)

        # Only AAPL at ts=2000 should be blocked
        expected = np.array([False, False, False, False])
        # Wait - ts 2000 has symbol MSFT, ts 3000 has symbol AAPL
        # Let's verify: AAPL rows are at indices 0, 2
        # ts[0]=1000 (AAPL), window 1500-2500 -> 1000 not in range
        # ts[2]=3000 (AAPL), window 1500-2500 -> 3000 not in range
        # So no rows blocked
        np.testing.assert_array_equal(mask, expected)

    def test_per_symbol_window_blocked(self):
        """Test with per-symbol windows where rows are blocked."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
        symbols = pd.Series(["AAPL", "MSFT", "AAPL", "MSFT"])
        windows = [
            {"start_ts_ms": 2500, "end_ts_ms": 3500, "symbol": "AAPL"},
        ]

        mask = _in_earnings_blackout(ts_ms, symbols, windows)

        # AAPL at index 2 (ts=3000) should be blocked
        expected = np.array([False, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_multiple_windows(self):
        """Test with multiple windows."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int64)
        windows = [
            {"start_ts_ms": 1500, "end_ts_ms": 2500},
            {"start_ts_ms": 4500, "end_ts_ms": 5500},
        ]

        mask = _in_earnings_blackout(ts_ms, None, windows)

        expected = np.array([False, True, False, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_symbol_case_insensitivity(self):
        """Test that symbol matching is case-insensitive."""
        from no_trade import _in_earnings_blackout

        ts_ms = np.array([1000, 2000], dtype=np.int64)
        symbols = pd.Series(["aapl", "AAPL"])  # Different case
        windows = [
            {"start_ts_ms": 500, "end_ts_ms": 1500, "symbol": "AAPL"},
        ]

        mask = _in_earnings_blackout(ts_ms, symbols, windows)

        # Both should be treated as AAPL, only first row in window
        expected = np.array([True, False])
        np.testing.assert_array_equal(mask, expected)


# ==============================================================================
# Test Integration with No-Trade Mask
# ==============================================================================


class TestEarningsIntegration:
    """Tests for earnings integration with compute_no_trade_mask."""

    def test_earnings_disabled_by_default(self):
        """Test that earnings filter is disabled by default."""
        from no_trade import _compute_no_trade_components, NO_TRADE_FEATURES_DISABLED

        if NO_TRADE_FEATURES_DISABLED:
            pytest.skip("NO_TRADE_FEATURES_DISABLED is True")

        df = pd.DataFrame(
            {
                "ts_ms": [1000, 2000, 3000],
                "symbol": ["AAPL", "AAPL", "AAPL"],
            }
        )

        cfg = NoTradeConfig()
        assert cfg.earnings.enabled is False

        mask, reasons, meta, state, labels = _compute_no_trade_components(df, cfg)

        # No earnings blackout should be applied
        assert "earnings_blackout" in reasons.columns
        assert not reasons["earnings_blackout"].any()

    def test_crypto_symbols_not_affected(self):
        """Test that crypto symbols (ending in USDT/USD) are not affected."""
        # This is tested via the filter in _compute_no_trade_components
        # Crypto symbols should be filtered out before fetching earnings

        from no_trade import _compute_no_trade_components, NO_TRADE_FEATURES_DISABLED

        if NO_TRADE_FEATURES_DISABLED:
            pytest.skip("NO_TRADE_FEATURES_DISABLED is True")

        df = pd.DataFrame(
            {
                "ts_ms": [1000, 2000, 3000],
                "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSD"],
            }
        )

        cfg = NoTradeConfig(earnings={"enabled": True})

        # Should not raise and should not block any rows
        # (crypto symbols are filtered out)
        mask, reasons, meta, state, labels = _compute_no_trade_components(df, cfg)

        # Crypto symbols should not have earnings blackout
        assert not reasons.get("earnings_blackout", pd.Series(False)).any()

    def test_reason_label_exists(self):
        """Test that earnings_blackout reason label exists."""
        from no_trade import _compute_no_trade_components, NO_TRADE_FEATURES_DISABLED

        if NO_TRADE_FEATURES_DISABLED:
            pytest.skip("NO_TRADE_FEATURES_DISABLED is True")

        df = pd.DataFrame(
            {
                "ts_ms": [1000],
                "symbol": ["AAPL"],
            }
        )

        cfg = NoTradeConfig()
        _, _, _, _, labels = _compute_no_trade_components(df, cfg)

        assert "earnings_blackout" in labels
        assert labels["earnings_blackout"] == "Earnings blackout period"


# ==============================================================================
# Test YAML Config Loading
# ==============================================================================


class TestYAMLConfigLoading:
    """Tests for loading earnings config from YAML."""

    def test_load_earnings_from_yaml_dict(self):
        """Test loading earnings config from dict representation."""
        from no_trade_config import _normalise_no_trade_payload

        raw = {
            "earnings": {
                "enabled": True,
                "pre_earnings_bars": 5,
                "post_earnings_bars": 2,
                "symbols": ["AAPL", "MSFT"],
            }
        }

        result = _normalise_no_trade_payload(raw)

        assert "earnings" in result
        assert result["earnings"]["enabled"] is True
        assert result["earnings"]["pre_earnings_bars"] == 5
        assert result["earnings"]["post_earnings_bars"] == 2
        assert result["earnings"]["symbols"] == ["AAPL", "MSFT"]

    def test_default_earnings_config(self):
        """Test default earnings config when not specified."""
        from no_trade_config import _normalise_no_trade_payload

        raw = {}
        result = _normalise_no_trade_payload(raw)

        assert "earnings" in result
        assert result["earnings"]["enabled"] is False
        assert result["earnings"]["pre_earnings_bars"] == 2
        assert result["earnings"]["post_earnings_bars"] == 1


# ==============================================================================
# Test Mock Earnings Adapter
# ==============================================================================


class TestEarningsAdapterIntegration:
    """Tests for integration with Yahoo earnings adapter (mocked)."""

    def test_get_earnings_events_caching(self):
        """Test that earnings events are cached."""
        from no_trade import _get_earnings_events, _earnings_cache

        # Clear cache first
        _earnings_cache.clear()

        # Mock at the location where it's imported
        with patch("adapters.yahoo.earnings.YahooEarningsAdapter") as MockAdapter:
            mock_instance = MagicMock()
            MockAdapter.return_value = mock_instance

            # Create mock earnings events
            mock_event = MagicMock()
            mock_event.symbol = "AAPL"
            mock_event.report_date = "2025-01-15"
            mock_event.is_confirmed = True
            mock_instance.get_earnings_history.return_value = [mock_event]

            # First call
            result1 = _get_earnings_events(["AAPL"], "2025-01-01", "2025-03-01", cache_ttl_sec=3600)

            # Second call (should use cache)
            result2 = _get_earnings_events(["AAPL"], "2025-01-01", "2025-03-01", cache_ttl_sec=3600)

            # Adapter should only be called once (due to caching)
            assert mock_instance.get_earnings_history.call_count == 1

            # Results should be the same
            assert result1 == result2

    def test_get_earnings_events_import_error_handling(self):
        """Test handling of errors when fetching earnings."""
        from no_trade import _get_earnings_events, _earnings_cache

        _earnings_cache.clear()

        # Test that function handles missing adapter gracefully
        # by trying to fetch and catching errors
        with patch.dict("sys.modules", {"adapters.yahoo.earnings": None}):
            # When module is None, import will fail
            result = _get_earnings_events(["AAPL"], "2025-01-01", "2025-03-01", cache_ttl_sec=3600)
            # Should either return empty or have cached result
            assert isinstance(result, dict)


# ==============================================================================
# Test Backward Compatibility
# ==============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_existing_crypto_workflow(self):
        """Test that existing crypto workflow is not affected."""
        from no_trade import compute_no_trade_mask, NO_TRADE_FEATURES_DISABLED

        if NO_TRADE_FEATURES_DISABLED:
            pytest.skip("NO_TRADE_FEATURES_DISABLED is True")

        # Simulate crypto trading data
        df = pd.DataFrame(
            {
                "ts_ms": [1000, 2000, 3000],
                "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            }
        )

        cfg = NoTradeConfig(earnings={"enabled": True})

        # Should not raise and should work normally
        mask = compute_no_trade_mask(df, config=cfg)
        assert isinstance(mask, pd.Series)
        assert len(mask) == len(df)

    def test_no_trade_config_without_earnings(self):
        """Test NoTradeConfig works without explicit earnings config."""
        cfg = NoTradeConfig(
            daily_utc=["00:00-00:05"],
            funding_buffer_min=5,
        )

        # Should have default earnings config
        assert hasattr(cfg, "earnings")
        assert cfg.earnings.enabled is False

    def test_existing_config_loading(self):
        """Test that existing configs without earnings still work."""
        from no_trade_config import get_no_trade_config
        import tempfile
        import os

        # Create a minimal config without earnings
        config_content = """
no_trade:
  maintenance:
    funding_buffer_min: 5
    daily_utc:
      - "00:00-00:05"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            cfg = get_no_trade_config(temp_path)
            # Should have default earnings config
            assert hasattr(cfg, "earnings")
            assert cfg.earnings.enabled is False
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
