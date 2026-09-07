# -*- coding: utf-8 -*-
"""
Tests for stock universe auto-update service.

Tests cover:
- Popular symbols preset functions
- Symbol fetching and caching
- ETF and sector filtering
- Module-level convenience functions
- Crypto backward compatibility

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

import pytest
import tempfile
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestPopularSymbols:
    """Tests for popular symbols preset."""

    def test_get_popular_symbols_returns_list(self):
        """Test get_popular_symbols returns a list."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_popular_symbols_contains_expected(self):
        """Test popular symbols contains expected tickers."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols()

        # Should contain major tech stocks
        expected = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        for symbol in expected:
            assert symbol in symbols

    def test_popular_symbols_contains_etfs(self):
        """Test popular symbols contains major ETFs."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols(include_etfs=True)

        # Should contain major ETFs
        expected_etfs = ["SPY", "QQQ"]
        for etf in expected_etfs:
            assert etf in symbols

    def test_popular_symbols_excludes_etfs_when_disabled(self):
        """Test popular symbols excludes ETFs when include_etfs=False."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols(include_etfs=False)

        # Should NOT contain ETFs
        etfs = ["SPY", "QQQ", "GLD", "SLV"]
        for etf in etfs:
            assert etf not in symbols

    def test_popular_symbols_contains_metals(self):
        """Test popular symbols contains precious metals ETFs."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols(include_etfs=True)

        # Should contain precious metals
        metals = ["GLD", "SLV"]
        for metal in metals:
            assert metal in symbols

    def test_popular_symbols_sorted(self):
        """Test popular symbols are sorted."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols()
        assert symbols == sorted(symbols)


class TestHighlyLiquidSymbols:
    """Tests for highly liquid symbols."""

    def test_get_highly_liquid_returns_list(self):
        """Test highly liquid symbols filter returns a list."""
        from services.universe_stocks import get_popular_symbols

        symbols = get_popular_symbols(highly_liquid_only=True)

        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_highly_liquid_smaller_than_popular(self):
        """Test highly liquid is smaller than all popular."""
        from services.universe_stocks import get_popular_symbols

        popular = get_popular_symbols(highly_liquid_only=False)
        liquid = get_popular_symbols(highly_liquid_only=True)

        assert len(liquid) < len(popular)

    def test_highly_liquid_contains_major_stocks(self):
        """Test highly liquid contains major stocks."""
        from services.universe_stocks import get_popular_symbols

        liquid = get_popular_symbols(highly_liquid_only=True)

        # Should contain major stocks
        expected = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
        for symbol in expected:
            assert symbol in liquid


class TestSectorSymbols:
    """Tests for sector-based symbol filtering."""

    def test_get_tech_sector(self):
        """Test getting tech sector symbols."""
        from services.universe_stocks import get_sector_symbols

        symbols = get_sector_symbols("tech")

        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "NVDA" in symbols

    def test_get_finance_sector(self):
        """Test getting finance sector symbols."""
        from services.universe_stocks import get_sector_symbols

        symbols = get_sector_symbols("finance")

        assert len(symbols) > 0
        assert "JPM" in symbols
        assert "BAC" in symbols

    def test_get_healthcare_sector(self):
        """Test getting healthcare sector symbols."""
        from services.universe_stocks import get_sector_symbols

        symbols = get_sector_symbols("healthcare")

        assert len(symbols) > 0
        assert "JNJ" in symbols
        assert "UNH" in symbols

    def test_get_unknown_sector(self):
        """Test getting unknown sector returns empty list."""
        from services.universe_stocks import get_sector_symbols

        symbols = get_sector_symbols("unknown_sector")

        assert symbols == []


class TestETFSymbols:
    """Tests for ETF symbol functions."""

    def test_get_all_etfs(self):
        """Test getting all ETF symbols."""
        from services.universe_stocks import get_etf_symbols

        etfs = get_etf_symbols()

        assert len(etfs) > 0
        assert "SPY" in etfs
        assert "QQQ" in etfs
        assert "GLD" in etfs

    def test_get_index_etfs(self):
        """Test getting index ETFs."""
        from services.universe_stocks import get_etf_symbols

        etfs = get_etf_symbols(category="index")

        assert "SPY" in etfs
        assert "QQQ" in etfs
        assert "IWM" in etfs

    def test_get_commodity_etfs(self):
        """Test getting commodity ETFs."""
        from services.universe_stocks import get_etf_symbols

        etfs = get_etf_symbols(category="commodity")

        assert "GLD" in etfs
        assert "SLV" in etfs

    def test_get_bond_etfs(self):
        """Test getting bond ETFs."""
        from services.universe_stocks import get_etf_symbols

        etfs = get_etf_symbols(category="bond")

        assert "TLT" in etfs
        assert "BND" in etfs

    def test_get_unknown_category(self):
        """Test getting unknown category returns empty list."""
        from services.universe_stocks import get_etf_symbols

        etfs = get_etf_symbols(category="unknown")

        assert etfs == []


class TestSymbolFiltering:
    """Tests for symbol filtering functions."""

    def test_filter_by_prefix(self):
        """Test filtering symbols by prefix."""
        from services.universe_stocks import filter_symbols_by_prefix

        symbols = ["AAPL", "AMZN", "AMD", "MSFT", "META"]
        filtered = filter_symbols_by_prefix(symbols, "A")

        assert "AAPL" in filtered
        assert "AMZN" in filtered
        assert "AMD" in filtered
        assert "MSFT" not in filtered
        assert "META" not in filtered

    def test_filter_by_prefix_case_insensitive(self):
        """Test filtering is case insensitive for prefix."""
        from services.universe_stocks import filter_symbols_by_prefix

        symbols = ["AAPL", "AMZN", "MSFT"]
        filtered = filter_symbols_by_prefix(symbols, "a")

        assert "AAPL" in filtered
        assert "AMZN" in filtered


class TestSymbolCaching:
    """Tests for symbol caching functionality."""

    def test_get_symbols_from_cache(self):
        """Test getting symbols from cache file."""
        from services.universe_stocks import get_symbols

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file
            cache_data = {
                "vendor": "alpaca",
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "count": 3,
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            # Get symbols (should use cache)
            symbols = get_symbols(out=cache_file, ttl=3600)

            assert "AAPL" in symbols
            assert "MSFT" in symbols
            assert "GOOGL" in symbols

    def test_get_universe_metadata(self):
        """Test getting universe metadata."""
        from services.universe_stocks import get_universe_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file
            cache_data = {
                "vendor": "alpaca",
                "asset_class": "us_equity",
                "generated_at": "2024-01-01T00:00:00Z",
                "count": 100,
                "symbols": ["AAPL"] * 100,
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            metadata = get_universe_metadata(out=cache_file)

            assert metadata["exists"] is True
            assert metadata["count"] == 100
            assert metadata["vendor"] == "alpaca"

    def test_get_universe_metadata_nonexistent(self):
        """Test metadata for nonexistent file."""
        from services.universe_stocks import get_universe_metadata

        metadata = get_universe_metadata(out="/nonexistent/path.json")

        assert metadata["exists"] is False
        assert metadata["count"] == 0


class TestAssetDetails:
    """Tests for asset details functions."""

    def test_get_assets_from_cache(self):
        """Test getting assets with metadata."""
        from services.universe_stocks import get_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file with assets
            cache_data = {
                "vendor": "alpaca",
                "symbols": ["AAPL", "MSFT"],
                "assets": [
                    {"symbol": "AAPL", "exchange": "NASDAQ", "tradable": True},
                    {"symbol": "MSFT", "exchange": "NASDAQ", "tradable": True},
                ],
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            assets = get_assets(out=cache_file, ttl=3600)

            assert len(assets) == 2
            assert assets[0]["symbol"] == "AAPL"

    def test_get_assets_filter_by_exchange(self):
        """Test filtering assets by exchange."""
        from services.universe_stocks import get_assets

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file with mixed exchanges
            cache_data = {
                "symbols": ["AAPL", "BAC"],
                "assets": [
                    {"symbol": "AAPL", "exchange": "NASDAQ", "tradable": True},
                    {"symbol": "BAC", "exchange": "NYSE", "tradable": True},
                ],
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            assets = get_assets(out=cache_file, ttl=3600, exchange="NYSE")

            assert len(assets) == 1
            assert assets[0]["symbol"] == "BAC"


class TestConstants:
    """Tests for module constants."""

    def test_popular_symbols_constant(self):
        """Test POPULAR_SYMBOLS constant."""
        from services.universe_stocks import POPULAR_SYMBOLS

        assert isinstance(POPULAR_SYMBOLS, set)
        assert len(POPULAR_SYMBOLS) > 50  # Should have many symbols
        assert "AAPL" in POPULAR_SYMBOLS

    def test_highly_liquid_constant(self):
        """Test HIGHLY_LIQUID_SYMBOLS constant."""
        from services.universe_stocks import HIGHLY_LIQUID_SYMBOLS

        assert isinstance(HIGHLY_LIQUID_SYMBOLS, set)
        assert len(HIGHLY_LIQUID_SYMBOLS) > 10
        assert "AAPL" in HIGHLY_LIQUID_SYMBOLS

    def test_highly_liquid_subset_of_popular(self):
        """Test HIGHLY_LIQUID is subset of POPULAR."""
        from services.universe_stocks import POPULAR_SYMBOLS, HIGHLY_LIQUID_SYMBOLS

        assert HIGHLY_LIQUID_SYMBOLS.issubset(POPULAR_SYMBOLS)


class TestCryptoBackwardCompatibility:
    """Tests to ensure crypto functionality is not affected."""

    def test_stock_universe_independent_of_crypto(self):
        """Test stock universe doesn't contain crypto symbols."""
        from services.universe_stocks import get_popular_symbols, POPULAR_SYMBOLS

        symbols = get_popular_symbols()

        # Should not contain crypto symbols
        crypto_symbols = ["BTCUSDT", "ETHUSDT", "BTC/USDT", "BTC-USD"]
        for crypto in crypto_symbols:
            assert crypto not in symbols
            assert crypto not in POPULAR_SYMBOLS

    def test_existing_crypto_universe_service_unchanged(self):
        """Test that existing services/universe.py is unchanged."""
        # Import existing crypto universe service
        try:
            from services.universe import get_symbols as get_crypto_symbols

            # Should work independently
            # (This is a basic import test - the service may need API keys)
            assert callable(get_crypto_symbols)
        except ImportError:
            pytest.skip("Crypto universe service not available")


class TestStockUniverseConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from services.universe_stocks import StockUniverseConfig

        config = StockUniverseConfig()

        assert config.ttl_seconds == 24 * 60 * 60
        assert config.paper is True
        assert config.include_etfs is True

    def test_custom_config(self):
        """Test custom configuration."""
        from services.universe_stocks import StockUniverseConfig

        config = StockUniverseConfig(
            ttl_seconds=3600,
            paper=False,
            default_exchange="NYSE",
            fractionable_only=True,
        )

        assert config.ttl_seconds == 3600
        assert config.paper is False
        assert config.default_exchange == "NYSE"
        assert config.fractionable_only is True


class TestSymbolTradability:
    """Tests for symbol tradability check."""

    def test_is_symbol_tradable_from_cache(self):
        """Test checking symbol tradability from cache."""
        from services.universe_stocks import is_symbol_tradable

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file
            cache_data = {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            assert is_symbol_tradable("AAPL", out=cache_file) is True
            assert is_symbol_tradable("MSFT", out=cache_file) is True
            assert is_symbol_tradable("INVALID", out=cache_file) is False

    def test_is_symbol_tradable_case_insensitive(self):
        """Test tradability check is case insensitive."""
        from services.universe_stocks import is_symbol_tradable

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            # Create cache file
            cache_data = {
                "symbols": ["AAPL", "MSFT"],
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            # Uppercase should work
            assert is_symbol_tradable("aapl", out=cache_file) is True


class TestRefreshUniverse:
    """Tests for universe refresh functionality."""

    @patch("services.universe_stocks.fetch_from_alpaca")
    def test_refresh_universe_success(self, mock_fetch):
        """Test successful universe refresh."""
        from services.universe_stocks import refresh_universe

        mock_fetch.return_value = [
            {"symbol": "AAPL", "name": "Apple"},
            {"symbol": "MSFT", "name": "Microsoft"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            result = refresh_universe(out=cache_file)

            assert result is True
            assert os.path.exists(cache_file)

    @patch("services.universe_stocks.fetch_from_alpaca")
    def test_refresh_universe_failure(self, mock_fetch):
        """Test universe refresh failure handling."""
        from services.universe_stocks import refresh_universe

        mock_fetch.side_effect = Exception("API Error")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_universe.json")

            result = refresh_universe(out=cache_file)

            assert result is False
