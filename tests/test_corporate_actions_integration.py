# -*- coding: utf-8 -*-
"""
Tests for corporate actions integration in data_loader_multi_asset.py
"""

import numpy as np
import pandas as pd
import pytest

# Skip entire module if pyarrow not available (tests use parquet files)
pytest.importorskip("pyarrow", reason="pyarrow required for parquet support")

from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from data_loader_multi_asset import (
    load_multi_asset_data,
    load_from_adapter,
    load_stock_data,
    load_alpaca_data,
    load_polygon_data,
    AssetClass,
    DataVendor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_equity_df():
    """Create sample equity DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=100, freq="4h")
    return pd.DataFrame(
        {
            "timestamp": [int(ts.timestamp()) for ts in timestamps],
            "symbol": "AAPL",
            "open": np.random.uniform(150, 160, 100),
            "high": np.random.uniform(160, 170, 100),
            "low": np.random.uniform(140, 150, 100),
            "close": np.random.uniform(150, 160, 100),
            "volume": np.random.uniform(1000000, 5000000, 100),
        }
    )


@pytest.fixture
def sample_parquet_file(sample_equity_df, tmp_path):
    """Create temporary parquet file with sample data."""
    file_path = tmp_path / "AAPL.parquet"
    sample_equity_df.to_parquet(file_path)
    return str(file_path)


# =============================================================================
# Test load_multi_asset_data with corporate actions
# =============================================================================


class TestLoadMultiAssetDataCorporateActions:
    """Tests for load_multi_asset_data corporate actions integration."""

    def test_equity_loads_with_corporate_actions_default(self, sample_parquet_file):
        """Test that equity data loads with corporate actions by default."""
        with patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust:
            mock_adjust.side_effect = lambda df, symbol: df  # Pass through

            all_dfs, _ = load_multi_asset_data(
                paths=[sample_parquet_file],
                asset_class=AssetClass.EQUITY,
            )

            # Should be called since adjust_corporate_actions defaults to True
            mock_adjust.assert_called_once()
            assert "AAPL" in all_dfs

    def test_equity_skips_corporate_actions_when_disabled(self, sample_parquet_file):
        """Test that corporate actions can be disabled."""
        with patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust:
            all_dfs, _ = load_multi_asset_data(
                paths=[sample_parquet_file],
                asset_class=AssetClass.EQUITY,
                adjust_corporate_actions=False,
            )

            # Should NOT be called
            mock_adjust.assert_not_called()
            assert "AAPL" in all_dfs

    def test_crypto_skips_corporate_actions(self, sample_parquet_file):
        """Test that crypto data never applies corporate actions."""
        with patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust:
            # Even with adjust_corporate_actions=True, crypto should skip it
            all_dfs, _ = load_multi_asset_data(
                paths=[sample_parquet_file],
                asset_class=AssetClass.CRYPTO,
                adjust_corporate_actions=True,
                merge_fear_greed=False,
            )

            mock_adjust.assert_not_called()

    def test_add_corp_features_when_enabled(self, sample_parquet_file):
        """Test that corporate action features are added when enabled."""
        with (
            patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust,
            patch("data_loader_multi_asset.add_corporate_action_features") as mock_features,
        ):
            mock_adjust.side_effect = lambda df, symbol: df
            mock_features.side_effect = lambda df, symbol: df

            all_dfs, _ = load_multi_asset_data(
                paths=[sample_parquet_file],
                asset_class=AssetClass.EQUITY,
                add_corp_features=True,
            )

            mock_adjust.assert_called_once()
            mock_features.assert_called_once()


# =============================================================================
# Test load_stock_data convenience function
# =============================================================================


class TestLoadStockDataCorporateActions:
    """Tests for load_stock_data corporate actions integration."""

    def test_passes_corporate_actions_param(self, sample_parquet_file):
        """Test that corporate actions params are passed through."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_stock_data(
                paths=[sample_parquet_file],
                adjust_corporate_actions=True,
                add_corp_features=True,
            )

            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args.kwargs
            assert call_kwargs["adjust_corporate_actions"] is True
            assert call_kwargs["add_corp_features"] is True

    def test_defaults_to_adjust_true(self, sample_parquet_file):
        """Test that adjust_corporate_actions defaults to True."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_stock_data(paths=[sample_parquet_file])

            call_kwargs = mock_load.call_args.kwargs
            assert call_kwargs["adjust_corporate_actions"] is True


# =============================================================================
# Test load_from_adapter with corporate actions
# =============================================================================


class TestLoadFromAdapterCorporateActions:
    """Tests for load_from_adapter corporate actions integration."""

    def test_alpaca_applies_corporate_actions(self):
        """Test that Alpaca data gets corporate actions applied."""
        mock_bar = MagicMock()
        mock_bar.ts = 1704067200000  # 2024-01-01 in ms
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 148.0
        mock_bar.close = 153.0
        mock_bar.volume_base = 1000000
        mock_bar.volume_quote = 153000000
        mock_bar.trades = 10000
        mock_bar.vwap = 152.0

        with (
            patch("adapters.registry.create_market_data_adapter") as mock_create,
            patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust,
        ):
            mock_adapter = MagicMock()
            mock_adapter.get_bars.return_value = [mock_bar]
            mock_create.return_value = mock_adapter
            mock_adjust.side_effect = lambda df, symbol: df

            all_dfs, _ = load_from_adapter(
                vendor=DataVendor.ALPACA,
                symbols=["AAPL"],
                adjust_corporate_actions=True,
            )

            mock_adjust.assert_called_once()
            assert "AAPL" in all_dfs

    def test_binance_skips_corporate_actions(self):
        """Test that Binance (crypto) skips corporate actions."""
        mock_bar = MagicMock()
        mock_bar.ts = 1704067200000
        mock_bar.open = 42000.0
        mock_bar.high = 43000.0
        mock_bar.low = 41000.0
        mock_bar.close = 42500.0
        mock_bar.volume_base = 100.0
        mock_bar.volume_quote = 4250000.0
        mock_bar.trades = 50000
        mock_bar.vwap = 42200.0

        with (
            patch("adapters.registry.create_market_data_adapter") as mock_create,
            patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust,
        ):
            mock_adapter = MagicMock()
            mock_adapter.get_bars.return_value = [mock_bar]
            mock_create.return_value = mock_adapter

            all_dfs, _ = load_from_adapter(
                vendor=DataVendor.BINANCE,
                symbols=["BTCUSDT"],
                adjust_corporate_actions=True,  # Should be ignored for crypto
            )

            mock_adjust.assert_not_called()


# =============================================================================
# Test load_alpaca_data and load_polygon_data
# =============================================================================


class TestAlpacaPolygonCorporateActions:
    """Tests for Alpaca and Polygon convenience functions."""

    def test_load_alpaca_data_passes_params(self):
        """Test that load_alpaca_data passes corporate actions params."""
        with patch("data_loader_multi_asset.load_from_adapter") as mock_load:
            mock_load.return_value = ({}, {})

            load_alpaca_data(
                symbols=["AAPL"],
                adjust_corporate_actions=True,
                add_corp_features=True,
            )

            call_kwargs = mock_load.call_args.kwargs
            assert call_kwargs["adjust_corporate_actions"] is True
            assert call_kwargs["add_corp_features"] is True

    def test_load_polygon_data_passes_params(self):
        """Test that load_polygon_data passes corporate actions params."""
        with patch("data_loader_multi_asset.load_from_adapter") as mock_load:
            mock_load.return_value = ({}, {})

            load_polygon_data(
                symbols=["MSFT"],
                adjust_corporate_actions=False,
                add_corp_features=True,
            )

            call_kwargs = mock_load.call_args.kwargs
            assert call_kwargs["adjust_corporate_actions"] is False
            assert call_kwargs["add_corp_features"] is True


# =============================================================================
# Integration test with actual CorporateActionsService
# =============================================================================


class TestCorporateActionsServiceIntegration:
    """Integration tests with actual CorporateActionsService."""

    def test_apply_split_adjustment_integration(self, sample_equity_df):
        """Test apply_split_adjustment with service."""
        from data_loader_multi_asset import apply_split_adjustment

        # Mock the service to avoid external API calls
        with patch("services.corporate_actions.get_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.adjust_prices.return_value = sample_equity_df.copy()
            mock_get_service.return_value = mock_service

            result = apply_split_adjustment(sample_equity_df, "AAPL")

            # Should have called the service
            mock_service.adjust_prices.assert_called_once()
            assert len(result) == len(sample_equity_df)

    def test_add_corporate_action_features_integration(self, sample_equity_df):
        """Test add_corporate_action_features with service."""
        from data_loader_multi_asset import add_corporate_action_features

        # Mock the service
        with patch("services.corporate_actions.get_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.compute_gap_features.return_value = sample_equity_df.copy()
            mock_service.get_dividends.return_value = []
            mock_service.get_earnings.return_value = []
            mock_get_service.return_value = mock_service

            result = add_corporate_action_features(sample_equity_df, "AAPL")

            # Should have called compute_gap_features
            mock_service.compute_gap_features.assert_called_once()


# =============================================================================
# Test backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_load_multi_asset_data_default_params(self, sample_parquet_file):
        """Test that default params maintain backward compatibility."""
        with patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust:
            mock_adjust.side_effect = lambda df, symbol: df

            # Call without new params - should still work
            all_dfs, all_obs = load_multi_asset_data(
                paths=[sample_parquet_file],
                asset_class=AssetClass.EQUITY,
            )

            assert isinstance(all_dfs, dict)
            assert isinstance(all_obs, dict)

    def test_crypto_unchanged(self, tmp_path):
        """Test that crypto loading is unchanged."""
        # Create crypto-like data
        timestamps = pd.date_range("2024-01-01", periods=100, freq="4h")
        crypto_df = pd.DataFrame(
            {
                "timestamp": [int(ts.timestamp()) for ts in timestamps],
                "symbol": "BTCUSDT",
                "open": np.random.uniform(40000, 45000, 100),
                "high": np.random.uniform(45000, 50000, 100),
                "low": np.random.uniform(35000, 40000, 100),
                "close": np.random.uniform(40000, 45000, 100),
                "volume": np.random.uniform(100, 500, 100),
            }
        )

        file_path = tmp_path / "BTCUSDT.parquet"
        crypto_df.to_parquet(file_path)

        with patch("data_loader_multi_asset.apply_split_adjustment") as mock_adjust:
            all_dfs, _ = load_multi_asset_data(
                paths=[str(file_path)],
                asset_class=AssetClass.CRYPTO,
                merge_fear_greed=False,
            )

            # Corporate actions should NOT be called for crypto
            mock_adjust.assert_not_called()
            assert "BTCUSDT" in all_dfs
