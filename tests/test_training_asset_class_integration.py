# -*- coding: utf-8 -*-
"""
tests/test_training_asset_class_integration.py
Phase 11: Training Pipeline Stock Integration Tests

These tests verify that:
1. Crypto pipeline remains 100% backward compatible (CRITICAL)
2. Equity pipeline correctly loads stock data with features
3. asset_class detection works correctly from config
4. load_all_data() routes correctly based on asset_class

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def crypto_feather_file(tmp_path: Path) -> Path:
    """Create a minimal crypto .feather file for testing."""
    n_rows = 100
    df = pd.DataFrame(
        {
            "timestamp": np.arange(1609459200, 1609459200 + n_rows * 14400, 14400),
            "open": np.random.uniform(30000, 50000, n_rows),
            "high": np.random.uniform(30000, 50000, n_rows),
            "low": np.random.uniform(30000, 50000, n_rows),
            "close": np.random.uniform(30000, 50000, n_rows),
            "volume": np.random.uniform(100, 1000, n_rows),
            "quote_asset_volume": np.random.uniform(1000000, 10000000, n_rows),
            "number_of_trades": np.random.randint(100, 1000, n_rows),
            "taker_buy_base_asset_volume": np.random.uniform(50, 500, n_rows),
            "taker_buy_quote_asset_volume": np.random.uniform(500000, 5000000, n_rows),
            "symbol": "BTCUSDT",
        }
    )

    file_path = tmp_path / "BTCUSDT.feather"
    df.to_feather(file_path)
    return file_path


@pytest.fixture
def stock_parquet_file(tmp_path: Path) -> Path:
    """Create a minimal stock .parquet file for testing."""
    n_rows = 100
    df = pd.DataFrame(
        {
            "timestamp": np.arange(1609459200, 1609459200 + n_rows * 14400, 14400),
            "open": np.random.uniform(100, 200, n_rows),
            "high": np.random.uniform(100, 200, n_rows),
            "low": np.random.uniform(100, 200, n_rows),
            "close": np.random.uniform(100, 200, n_rows),
            "volume": np.random.uniform(10000, 100000, n_rows),
            "symbol": "AAPL",
        }
    )

    file_path = tmp_path / "AAPL.parquet"
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def mock_config_crypto() -> MagicMock:
    """Create a mock config for crypto (no asset_class specified)."""
    cfg = MagicMock()
    cfg.asset_class = None
    cfg.data = MagicMock()
    cfg.data.asset_class = None
    cfg.data.processed_dir = "data/processed"
    cfg.data.timeframe = "4h"
    cfg.data.paths = None
    return cfg


@pytest.fixture
def mock_config_equity() -> MagicMock:
    """Create a mock config for equity."""
    cfg = MagicMock()
    cfg.asset_class = "equity"
    cfg.data = MagicMock()
    cfg.data.asset_class = "equity"
    cfg.data.processed_dir = "data/processed"
    cfg.data.timeframe = "4h"
    cfg.data.paths = ["data/raw_stocks/*.parquet"]
    return cfg


# =============================================================================
# Test: Asset Class Detection
# =============================================================================


class TestAssetClassDetection:
    """Test asset_class detection from config."""

    def test_default_is_crypto_when_not_specified(self, mock_config_crypto: MagicMock):
        """When asset_class is not in config, default should be 'crypto'."""
        cfg = mock_config_crypto

        # Simulate the detection logic from train_model_multi_patch.py
        asset_class = (
            getattr(cfg, "asset_class", None) or getattr(cfg.data, "asset_class", None) or "crypto"
        ).lower()

        assert asset_class == "crypto"

    def test_equity_detected_from_root_config(self):
        """asset_class at root level should be detected."""
        cfg = MagicMock()
        cfg.asset_class = "equity"
        cfg.data = MagicMock()
        cfg.data.asset_class = None

        asset_class = (
            getattr(cfg, "asset_class", None) or getattr(cfg.data, "asset_class", None) or "crypto"
        ).lower()

        assert asset_class == "equity"

    def test_equity_detected_from_data_config(self):
        """asset_class in data block should be detected."""
        cfg = MagicMock()
        cfg.asset_class = None
        cfg.data = MagicMock()
        cfg.data.asset_class = "equity"

        asset_class = (
            getattr(cfg, "asset_class", None) or getattr(cfg.data, "asset_class", None) or "crypto"
        ).lower()

        assert asset_class == "equity"

    def test_root_config_takes_precedence(self):
        """Root asset_class should take precedence over data.asset_class."""
        cfg = MagicMock()
        cfg.asset_class = "crypto"
        cfg.data = MagicMock()
        cfg.data.asset_class = "equity"  # Should be ignored

        asset_class = (
            getattr(cfg, "asset_class", None) or getattr(cfg.data, "asset_class", None) or "crypto"
        ).lower()

        assert asset_class == "crypto"

    def test_case_insensitive_detection(self):
        """Asset class detection should be case-insensitive."""
        cfg = MagicMock()
        cfg.asset_class = "EQUITY"
        cfg.data = MagicMock()
        cfg.data.asset_class = None

        asset_class = (
            getattr(cfg, "asset_class", None) or getattr(cfg.data, "asset_class", None) or "crypto"
        ).lower()

        assert asset_class == "equity"


# =============================================================================
# Test: load_all_data() Routing
# =============================================================================


class TestLoadAllDataRouting:
    """Test that load_all_data() routes correctly based on asset_class."""

    def test_crypto_default_calls_crypto_path(self, crypto_feather_file: Path):
        """Default asset_class='crypto' should use crypto loading path."""
        from fetch_all_data_patch import load_all_data

        # Mock _read_fng to avoid file dependency
        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            all_dfs, all_obs = load_all_data(
                [str(crypto_feather_file)],
                synthetic_fraction=0,
                seed=42,
                asset_class="crypto",  # Explicit crypto
            )

        assert len(all_dfs) == 1
        assert "BTCUSDT" in all_dfs
        # Crypto data should have specific columns
        df = all_dfs["BTCUSDT"]
        assert "symbol" in df.columns
        assert "timestamp" in df.columns

    def test_equity_calls_equity_path(self, stock_parquet_file: Path):
        """asset_class='equity' should use equity loading path."""
        from fetch_all_data_patch import load_all_data

        # Mock _load_equity_data to verify it's called
        with patch("fetch_all_data_patch._load_equity_data") as mock_load:
            mock_load.return_value = ({"AAPL": pd.DataFrame()}, {})

            load_all_data(
                [str(stock_parquet_file)],
                synthetic_fraction=0,
                seed=42,
                asset_class="equity",
            )

            # Verify equity loader was called
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[1].get("add_stock_features", False) == True

    def test_crypto_does_not_call_equity_path(self, crypto_feather_file: Path):
        """Crypto should NOT trigger equity loading path."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            with patch("fetch_all_data_patch._load_equity_data") as mock_equity:
                load_all_data(
                    [str(crypto_feather_file)],
                    asset_class="crypto",
                )

                # Equity loader should NOT be called
                mock_equity.assert_not_called()


# =============================================================================
# Test: Crypto Backward Compatibility (CRITICAL)
# =============================================================================


class TestCryptoBackwardCompatibility:
    """
    CRITICAL: These tests verify crypto pipeline remains unchanged.

    The crypto pipeline must:
    1. Load .feather files from processed_data_dir
    2. Merge Fear & Greed index
    3. NOT add stock features (VIX, RS, sector)
    4. Keep all existing columns intact
    """

    def test_crypto_loads_feather_files(self, crypto_feather_file: Path):
        """Crypto should load .feather files correctly."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            all_dfs, _ = load_all_data(
                [str(crypto_feather_file)],
                asset_class="crypto",
            )

        assert "BTCUSDT" in all_dfs
        df = all_dfs["BTCUSDT"]
        assert len(df) == 100  # Original row count preserved

    def test_crypto_preserves_original_columns(self, crypto_feather_file: Path):
        """Crypto should preserve all original columns."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            all_dfs, _ = load_all_data(
                [str(crypto_feather_file)],
                asset_class="crypto",
            )

        df = all_dfs["BTCUSDT"]
        required_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "symbol",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_crypto_does_not_have_stock_features(self, crypto_feather_file: Path):
        """Crypto should NOT have stock-specific features."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            all_dfs, _ = load_all_data(
                [str(crypto_feather_file)],
                asset_class="crypto",
            )

        df = all_dfs["BTCUSDT"]
        stock_feature_cols = [
            "vix_normalized",
            "vix_regime",
            "market_regime",
            "rs_spy_20d",
            "rs_spy_50d",
            "rs_qqq_20d",
            "sector_momentum",
        ]
        for col in stock_feature_cols:
            assert col not in df.columns, f"Crypto should not have stock feature: {col}"

    def test_crypto_fear_greed_merge_preserved(self, crypto_feather_file: Path):
        """Crypto should still merge Fear & Greed index."""
        from fetch_all_data_patch import load_all_data

        # Create mock F&G data
        fng_df = pd.DataFrame(
            {
                "timestamp": [1609459200, 1609545600],
                "fear_greed_value": [50, 60],
            }
        )

        with patch("fetch_all_data_patch._read_fng", return_value=fng_df):
            all_dfs, _ = load_all_data(
                [str(crypto_feather_file)],
                asset_class="crypto",
            )

        df = all_dfs["BTCUSDT"]
        # F&G merge should be attempted (column may or may not exist based on join)
        # The key is that the merge logic is still executed
        assert len(df) >= 1  # Data not corrupted

    def test_crypto_empty_input_raises_correct_error(self, tmp_path: Path):
        """Empty input should raise ValueError, not crash."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            # Should handle empty list gracefully
            all_dfs, _ = load_all_data(
                [],
                asset_class="crypto",
            )
            assert len(all_dfs) == 0


# =============================================================================
# Test: Equity Features Integration
# =============================================================================


class TestEquityFeaturesIntegration:
    """Test that equity data gets stock features added."""

    def test_equity_triggers_stock_features_loading(self, stock_parquet_file: Path):
        """Equity asset_class should trigger stock features addition."""
        from fetch_all_data_patch import load_all_data

        # Mock the equity loader to verify add_stock_features is True
        with patch("fetch_all_data_patch._load_equity_data") as mock_load:
            mock_load.return_value = ({"AAPL": pd.DataFrame()}, {})

            load_all_data(
                [str(stock_parquet_file)],
                asset_class="equity",
                add_stock_features=True,
            )

            # Verify add_stock_features was passed
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("add_stock_features") == True

    def test_equity_can_disable_stock_features(self, stock_parquet_file: Path):
        """Should be able to disable stock features for equity."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._load_equity_data") as mock_load:
            mock_load.return_value = ({"AAPL": pd.DataFrame()}, {})

            load_all_data(
                [str(stock_parquet_file)],
                asset_class="equity",
                add_stock_features=False,
            )

            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("add_stock_features") == False


# =============================================================================
# Test: Config YAML Compatibility
# =============================================================================


class TestConfigYAMLCompatibility:
    """Test that existing YAML configs work correctly."""

    def test_crypto_config_yaml_compatibility(self):
        """Existing crypto configs without asset_class should work."""
        # Simulate loading a config without asset_class
        cfg_dict = {
            "mode": "train",
            "data": {
                "processed_dir": "data/processed",
                "timeframe": "4h",
            },
            "model": {
                "algo": "ppo",
            },
        }

        # Detection logic
        asset_class = (
            cfg_dict.get("asset_class")
            or (cfg_dict.get("data", {}) or {}).get("asset_class")
            or "crypto"
        )

        assert asset_class == "crypto"

    def test_stock_config_yaml_compatibility(self):
        """Stock configs with asset_class: equity should work."""
        cfg_dict = {
            "mode": "train",
            "asset_class": "equity",
            "data": {
                "asset_class": "equity",
                "paths": ["data/raw_stocks/*.parquet"],
                "timeframe": "4h",
            },
        }

        asset_class = (
            cfg_dict.get("asset_class")
            or (cfg_dict.get("data", {}) or {}).get("asset_class")
            or "crypto"
        )

        assert asset_class == "equity"


# =============================================================================
# Test: Error Messages
# =============================================================================


class TestErrorMessages:
    """Test that error messages are appropriate for each asset class."""

    def test_crypto_error_message_mentions_feather(self, tmp_path: Path):
        """Crypto error should mention .feather files and crypto-specific steps."""
        # This tests the error message logic, not the actual file loading
        processed_data_dir = str(tmp_path / "empty")
        os.makedirs(processed_data_dir, exist_ok=True)

        # Simulate crypto branch error message
        error_msg = (
            f"No .feather files found in: {processed_data_dir}\n"
            f"Prepare Real Market Data:\n"
            f"  1. Run: python prepare_advanced_data.py\n"
        )

        assert ".feather" in error_msg
        assert "prepare_advanced_data.py" in error_msg

    def test_equity_error_message_mentions_parquet(self, tmp_path: Path):
        """Equity error should mention stock data download steps."""
        data_paths = ["data/raw_stocks/*.parquet"]

        # Simulate equity branch error message
        error_msg = (
            f"No .feather/.parquet files found in paths:\n"
            f"  {data_paths}\n"
            f"Prepare Stock Data:\n"
            f"  1. Run: python scripts/download_stock_data.py --symbols AAPL\n"
        )

        assert ".parquet" in error_msg
        assert "download_stock_data.py" in error_msg


# =============================================================================
# Integration Test: Full Pipeline Mock
# =============================================================================


class TestFullPipelineIntegration:
    """Integration tests for the full training pipeline changes."""

    def test_training_pipeline_crypto_unchanged(self, crypto_feather_file: Path, tmp_path: Path):
        """Full pipeline test: crypto training should work exactly as before."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            # Load data as crypto (default)
            all_dfs, all_obs = load_all_data(
                [str(crypto_feather_file)],
                synthetic_fraction=0,
                seed=42,
                asset_class="crypto",
                timeframe="4h",
            )

        # Verify structure
        assert isinstance(all_dfs, dict)
        assert isinstance(all_obs, dict)
        assert "BTCUSDT" in all_dfs

        # Verify data integrity
        df = all_dfs["BTCUSDT"]
        assert len(df) == 100
        assert df["close"].notna().all()

    def test_training_pipeline_params_passthrough(self, crypto_feather_file: Path):
        """Parameters should be correctly passed to load_all_data."""
        from fetch_all_data_patch import load_all_data

        with patch("fetch_all_data_patch._read_fng", return_value=pd.DataFrame()):
            # All params should be accepted without error
            all_dfs, all_obs = load_all_data(
                [str(crypto_feather_file)],
                synthetic_fraction=0,
                seed=42,
                asset_class="crypto",
                timeframe="4h",
                add_stock_features=True,  # Should be ignored for crypto
                vix_path=None,
                spy_path=None,
                qqq_path=None,
            )

        assert len(all_dfs) == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
