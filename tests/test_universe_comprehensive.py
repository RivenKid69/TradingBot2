"""Comprehensive unit tests for services/universe.py - Symbol management."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from services.universe import (
    run,
    get_symbols,
    _throttled_get,
    _ensure_dir,
    _is_stale,
    _DEFAULT_TTL_SECONDS,
)


class TestThrottledGet:
    """Test _throttled_get helper function."""

    @patch("requests.get")
    def test_throttled_get_success(self, mock_get):
        """Test successful throttled request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_get.return_value = mock_response

        result = _throttled_get("http://example.com/api")
        assert result.json() == {"data": "test"}
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_throttled_get_with_params(self, mock_get):
        """Test throttled request with parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        _throttled_get("http://example.com/api", params={"key": "value"})
        mock_get.assert_called_once_with(
            "http://example.com/api", params={"key": "value"}, timeout=20
        )

    @patch("requests.get")
    def test_throttled_get_respects_throttle(self, mock_get):
        """Test throttling delay between requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # First request
        start = time.time()
        _throttled_get("http://example.com/api")

        # Second request should be throttled
        _throttled_get("http://example.com/api")
        elapsed = time.time() - start

        # Should have some delay (at least _REQUEST_THROTTLE_SECONDS)
        assert elapsed >= 0.2  # _REQUEST_THROTTLE_SECONDS = 0.2

    @patch("requests.get")
    def test_throttled_get_retry_on_failure(self, mock_get):
        """Test retries on failure."""
        mock_get.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            Mock(status_code=200),  # Success on third attempt
        ]

        # Should succeed after retries
        result = _throttled_get("http://example.com/api", max_attempts=3)
        assert mock_get.call_count == 3

    @patch("requests.get")
    def test_throttled_get_raises_after_max_attempts(self, mock_get):
        """Test raises exception after max attempts."""
        mock_get.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            _throttled_get("http://example.com/api", max_attempts=2)


class TestEnsureDir:
    """Test _ensure_dir helper function."""

    def test_ensure_dir_creates_directory(self):
        """Test _ensure_dir creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "subdir", "file.txt")
            _ensure_dir(test_path)
            assert os.path.exists(os.path.dirname(test_path))

    def test_ensure_dir_existing_directory(self):
        """Test _ensure_dir with existing directory doesn't error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "file.txt")
            _ensure_dir(test_path)
            # Call again - should not error
            _ensure_dir(test_path)
            assert os.path.exists(tmpdir)

    def test_ensure_dir_with_no_directory(self):
        """Test _ensure_dir with filename only."""
        _ensure_dir("file.txt")  # Should not error


class TestIsStale:
    """Test _is_stale helper function."""

    def test_is_stale_missing_file(self):
        """Test _is_stale returns True for missing file."""
        assert _is_stale("/nonexistent/file.json", ttl=3600) is True

    def test_is_stale_fresh_file(self):
        """Test _is_stale returns False for fresh file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            # File just created - should not be stale
            assert _is_stale(path, ttl=3600) is False
        finally:
            os.unlink(path)

    def test_is_stale_old_file(self):
        """Test _is_stale returns True for old file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            # Set mtime to past
            old_time = time.time() - 7200  # 2 hours ago
            os.utime(path, (old_time, old_time))

            # Should be stale with 1 hour TTL
            assert _is_stale(path, ttl=3600) is True
        finally:
            os.unlink(path)


class TestRunFunction:
    """Test run() function."""

    @patch("services.universe._throttled_get")
    def test_run_basic_success(self, mock_get):
        """Test run() fetches and saves symbols successfully."""
        # Mock exchangeInfo response
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Verify symbols
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols

            # Verify file saved
            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                saved = json.load(f)
            assert saved == symbols

    @patch("services.universe._throttled_get")
    def test_run_filters_non_trading(self, mock_get):
        """Test run() filters non-trading symbols."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "DELISTED",
                    "status": "BREAK",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Only trading symbol should be included
            assert "BTCUSDT" in symbols
            assert "DELISTED" not in symbols

    @patch("services.universe._throttled_get")
    def test_run_filters_non_usdt(self, mock_get):
        """Test run() filters non-USDT pairs."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "BTCBUSD",
                    "status": "TRADING",
                    "quoteAsset": "BUSD",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Only USDT pair should be included
            assert "BTCUSDT" in symbols
            assert "BTCBUSD" not in symbols

    @patch("services.universe._throttled_get")
    def test_run_filters_non_spot(self, mock_get):
        """Test run() filters non-spot permissions."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "BTCUSDT_FUTURES",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["FUTURES"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Only spot symbol should be included
            assert "BTCUSDT" in symbols
            assert "BTCUSDT_FUTURES" not in symbols

    @patch("services.universe._throttled_get")
    def test_run_with_liquidity_threshold(self, mock_get):
        """Test run() filters by liquidity threshold."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "LOWLIQ",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        ticker_24hr = [
            {"symbol": "BTCUSDT", "quoteVolume": "1000000.0"},
            {"symbol": "LOWLIQ", "quoteVolume": "100.0"},
        ]

        mock_get.side_effect = [
            Mock(json=lambda: exchange_info),
            Mock(json=lambda: ticker_24hr),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path, liquidity_threshold=1000.0)

            # Only high liquidity symbol should be included
            assert "BTCUSDT" in symbols
            assert "LOWLIQ" not in symbols

    @patch("services.universe._throttled_get")
    def test_run_sorts_symbols(self, mock_get):
        """Test run() returns sorted symbols."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "ZZUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "AAUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "MMUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Should be sorted
            assert symbols == sorted(symbols)
            assert symbols[0] == "AAUSDT"
            assert symbols[-1] == "ZZUSDT"


class TestGetSymbolsFunction:
    """Test get_symbols() function."""

    @patch("services.universe.run")
    def test_get_symbols_returns_cached(self, mock_run):
        """Test get_symbols returns cached symbols when fresh."""
        # Create fresh cache file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["BTCUSDT", "ETHUSDT"], f)
            cache_path = f.name

        try:
            symbols = get_symbols(out=cache_path, force=False)

            # Should use cache, not call run()
            mock_run.assert_not_called()
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols
        finally:
            os.unlink(cache_path)

    @patch("services.universe.run")
    def test_get_symbols_refreshes_stale_cache(self, mock_run):
        """Test get_symbols refreshes stale cache."""
        mock_run.return_value = ["NEWBTC", "NEWETH"]

        # Create old cache file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["OLDBTC"], f)
            cache_path = f.name

        try:
            # Set mtime to past
            old_time = time.time() - _DEFAULT_TTL_SECONDS - 1
            os.utime(cache_path, (old_time, old_time))

            symbols = get_symbols(out=cache_path)

            # Should call run() to refresh
            mock_run.assert_called_once()
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    @patch("services.universe.run")
    def test_get_symbols_force_refresh(self, mock_run):
        """Test get_symbols with force=True always refreshes."""
        mock_run.return_value = ["BTCUSDT"]

        # Create fresh cache file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["OLDBTC"], f)
            cache_path = f.name

        try:
            symbols = get_symbols(out=cache_path, force=True)

            # Should call run() even with fresh cache
            mock_run.assert_called_once()
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    @patch("services.universe.run")
    def test_get_symbols_creates_missing_file(self, mock_run):
        """Test get_symbols creates file if missing."""
        mock_run.return_value = ["BTCUSDT"]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "symbols.json")

            symbols = get_symbols(out=cache_path)

            # Should call run() for missing file
            mock_run.assert_called_once()
            assert os.path.exists(cache_path)

    def test_get_symbols_with_liquidity_threshold(self):
        """Test get_symbols passes liquidity_threshold to run()."""
        with patch("services.universe.run") as mock_run:
            mock_run.return_value = ["BTCUSDT"]

            with tempfile.TemporaryDirectory() as tmpdir:
                cache_path = os.path.join(tmpdir, "symbols.json")

                get_symbols(out=cache_path, liquidity_threshold=1000000.0)

                mock_run.assert_called_once_with(cache_path, liquidity_threshold=1000000.0)


class TestIntegration:
    """Integration tests for universe module."""

    @patch("services.universe._throttled_get")
    def test_full_workflow_no_threshold(self, mock_get):
        """Test complete workflow without liquidity filter."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "symbols.json")

            # First call - should fetch from API
            symbols1 = get_symbols(out=cache_path, ttl=3600)
            assert len(symbols1) == 2

            # Second call - should use cache
            mock_get.reset_mock()
            symbols2 = get_symbols(out=cache_path, ttl=3600)
            assert symbols1 == symbols2
            mock_get.assert_not_called()

    @patch("services.universe._throttled_get")
    def test_full_workflow_with_threshold(self, mock_get):
        """Test complete workflow with liquidity filter."""
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
                {
                    "symbol": "LOWLIQ",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        ticker_24hr = [
            {"symbol": "BTCUSDT", "quoteVolume": "1000000.0"},
            {"symbol": "LOWLIQ", "quoteVolume": "100.0"},
        ]

        mock_get.side_effect = [
            Mock(json=lambda: exchange_info),
            Mock(json=lambda: ticker_24hr),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "symbols.json")

            symbols = get_symbols(out=cache_path, liquidity_threshold=10000.0, force=True)

            # Only BTCUSDT should pass threshold
            assert "BTCUSDT" in symbols
            assert "LOWLIQ" not in symbols


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("services.universe._throttled_get")
    def test_empty_symbols_list(self, mock_get):
        """Test handling of empty symbols list."""
        exchange_info = {"symbols": []}

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            assert symbols == []

    @patch("services.universe._throttled_get")
    def test_malformed_response(self, mock_get):
        """Test handling of malformed API response."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Should handle gracefully (empty list)
            assert symbols == []

    @patch("services.universe._throttled_get")
    def test_missing_fields_in_symbol(self, mock_get):
        """Test handling of symbols with missing fields."""
        exchange_info = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},  # Missing quoteAsset and permissions
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                },
            ]
        }

        mock_response = Mock()
        mock_response.json.return_value = exchange_info
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "symbols.json")
            symbols = run(out=out_path)

            # Only valid symbol should be included
            assert "ETHUSDT" in symbols
            assert "BTCUSDT" not in symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
