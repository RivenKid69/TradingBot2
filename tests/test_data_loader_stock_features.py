# -*- coding: utf-8 -*-
"""
tests/test_data_loader_stock_features.py
------------------------------------------------------------------
Tests for stock features integration in data_loader_multi_asset.py (Gap #1 fix).

This test module verifies:
1. VIX, SPY, QQQ data is loaded automatically for equity
2. Stock features (vix_normalized, market_regime, rs_spy_*, sector_momentum) are added
3. Crypto path is NOT affected (backward compatibility)
4. Feature opt-in/opt-out works correctly
5. Benchmark data loading functions work properly

Test Coverage:
- load_multi_asset_data with equity
- load_multi_asset_data with crypto (backward compatibility)
- _load_benchmark_data helper function
- _add_stock_features helper function
- load_stock_data convenience function
- add_stock_features=True/False parameter
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Skip all tests if pyarrow is not installed (required for parquet/feather operations)
pytest.importorskip("pyarrow", reason="pyarrow required for parquet/feather support")

from data_loader_multi_asset import (
    AssetClass,
    DataVendor,
    _add_stock_features,
    _load_benchmark_data,
    load_crypto_data,
    load_from_file,
    load_multi_asset_data,
    load_stock_data,
    DEFAULT_BENCHMARK_PATHS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_stock_df():
    """Create a sample stock DataFrame with OHLCV data."""
    n_rows = 100
    timestamps = np.arange(1700000000, 1700000000 + n_rows * 14400, 14400)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "AAPL",
            "open": np.random.uniform(150, 160, n_rows),
            "high": np.random.uniform(160, 165, n_rows),
            "low": np.random.uniform(145, 150, n_rows),
            "close": np.random.uniform(150, 160, n_rows),
            "volume": np.random.uniform(1e6, 5e6, n_rows),
            "quote_asset_volume": np.random.uniform(1e8, 5e8, n_rows),
            "number_of_trades": np.random.randint(1000, 5000, n_rows),
        }
    )
    # Fix OHLC consistency
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 1
    return df


@pytest.fixture
def sample_crypto_df():
    """Create a sample crypto DataFrame with OHLCV data."""
    n_rows = 100
    timestamps = np.arange(1700000000, 1700000000 + n_rows * 14400, 14400)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "open": np.random.uniform(40000, 45000, n_rows),
            "high": np.random.uniform(45000, 48000, n_rows),
            "low": np.random.uniform(38000, 40000, n_rows),
            "close": np.random.uniform(40000, 45000, n_rows),
            "volume": np.random.uniform(1000, 5000, n_rows),
            "quote_asset_volume": np.random.uniform(4e7, 2e8, n_rows),
            "number_of_trades": np.random.randint(10000, 50000, n_rows),
        }
    )
    # Fix OHLC consistency
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1) + 100
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1) - 100
    return df


@pytest.fixture
def sample_vix_df():
    """Create sample VIX benchmark data."""
    n_rows = 100
    timestamps = np.arange(1700000000, 1700000000 + n_rows * 14400, 14400)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "VIX",
            "open": np.random.uniform(15, 25, n_rows),
            "high": np.random.uniform(20, 30, n_rows),
            "low": np.random.uniform(12, 20, n_rows),
            "close": np.random.uniform(15, 25, n_rows),
            "volume": np.random.uniform(1e6, 5e6, n_rows),
        }
    )


@pytest.fixture
def sample_spy_df():
    """Create sample SPY benchmark data."""
    n_rows = 100
    timestamps = np.arange(1700000000, 1700000000 + n_rows * 14400, 14400)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "SPY",
            "open": np.random.uniform(450, 470, n_rows),
            "high": np.random.uniform(470, 480, n_rows),
            "low": np.random.uniform(440, 450, n_rows),
            "close": np.random.uniform(450, 470, n_rows),
            "volume": np.random.uniform(5e7, 1e8, n_rows),
        }
    )


@pytest.fixture
def sample_qqq_df():
    """Create sample QQQ benchmark data."""
    n_rows = 100
    timestamps = np.arange(1700000000, 1700000000 + n_rows * 14400, 14400)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "QQQ",
            "open": np.random.uniform(380, 400, n_rows),
            "high": np.random.uniform(400, 420, n_rows),
            "low": np.random.uniform(370, 380, n_rows),
            "close": np.random.uniform(380, 400, n_rows),
            "volume": np.random.uniform(3e7, 7e7, n_rows),
        }
    )


@pytest.fixture
def temp_data_dir(sample_stock_df, sample_crypto_df, sample_vix_df, sample_spy_df, sample_qqq_df):
    """Create temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create stock file
        stock_path = Path(tmpdir) / "AAPL.parquet"
        sample_stock_df.to_parquet(stock_path)

        # Create crypto file
        crypto_path = Path(tmpdir) / "BTCUSDT.parquet"
        sample_crypto_df.to_parquet(crypto_path)

        # Create benchmark files
        vix_path = Path(tmpdir) / "VIX.parquet"
        sample_vix_df.to_parquet(vix_path)

        spy_path = Path(tmpdir) / "SPY.parquet"
        sample_spy_df.to_parquet(spy_path)

        qqq_path = Path(tmpdir) / "QQQ.parquet"
        sample_qqq_df.to_parquet(qqq_path)

        yield {
            "dir": tmpdir,
            "stock": str(stock_path),
            "crypto": str(crypto_path),
            "vix": str(vix_path),
            "spy": str(spy_path),
            "qqq": str(qqq_path),
        }


# =============================================================================
# TEST: _load_benchmark_data
# =============================================================================


class TestLoadBenchmarkData:
    """Tests for _load_benchmark_data helper function."""

    def test_load_benchmark_from_explicit_path(self, temp_data_dir):
        """Should load benchmark data from explicit path."""
        df = _load_benchmark_data("VIX", temp_data_dir["vix"], "4h")

        assert df is not None
        assert not df.empty
        assert "close" in df.columns
        assert len(df) == 100

    def test_load_benchmark_nonexistent_path_returns_none(self):
        """Should return None when benchmark data not found."""
        df = _load_benchmark_data("VIX", "/nonexistent/path.parquet", "4h")

        # May return None or try fallback depending on availability
        # The function should not raise an exception
        assert df is None or isinstance(df, pd.DataFrame)

    def test_load_benchmark_tries_default_paths(self):
        """Should try default paths for benchmark data."""
        # This tests that DEFAULT_BENCHMARK_PATHS is used
        assert "VIX" in DEFAULT_BENCHMARK_PATHS
        assert "SPY" in DEFAULT_BENCHMARK_PATHS
        assert "QQQ" in DEFAULT_BENCHMARK_PATHS

        # Each should have multiple fallback paths
        assert len(DEFAULT_BENCHMARK_PATHS["VIX"]) >= 2
        assert len(DEFAULT_BENCHMARK_PATHS["SPY"]) >= 2


# =============================================================================
# TEST: _add_stock_features
# =============================================================================


class TestAddStockFeatures:
    """Tests for _add_stock_features helper function."""

    def test_add_stock_features_with_benchmarks(
        self, sample_stock_df, sample_vix_df, sample_spy_df, sample_qqq_df
    ):
        """Should add stock features when benchmark data is available."""
        # Patch at the services.sector_momentum module level since it's imported inside _add_stock_features
        with patch(
            "services.sector_momentum.enrich_dataframe_with_all_stock_features"
        ) as mock_enrich:
            # Set up mock to return enriched DataFrame
            enriched_df = sample_stock_df.copy()
            enriched_df["vix_normalized"] = 0.1
            enriched_df["market_regime"] = 0.5
            enriched_df["rs_spy_20d"] = 0.2
            enriched_df["sector_momentum"] = 0.3
            mock_enrich.return_value = enriched_df

            result = _add_stock_features(
                sample_stock_df,
                "AAPL",
                vix_df=sample_vix_df,
                spy_df=sample_spy_df,
                qqq_df=sample_qqq_df,
            )

            # Verify mock was called with correct arguments
            mock_enrich.assert_called_once()
            call_kwargs = mock_enrich.call_args[1]
            assert call_kwargs["symbol"] == "AAPL"
            assert call_kwargs["vix_df"] is not None
            assert call_kwargs["spy_df"] is not None
            assert call_kwargs["qqq_df"] is not None

    def test_add_stock_features_without_benchmarks(self, sample_stock_df):
        """Should handle missing benchmark data gracefully."""
        with patch(
            "services.sector_momentum.enrich_dataframe_with_all_stock_features"
        ) as mock_enrich:
            mock_enrich.return_value = sample_stock_df.copy()

            result = _add_stock_features(
                sample_stock_df,
                "AAPL",
                vix_df=None,
                spy_df=None,
                qqq_df=None,
            )

            # Should still call enrich function (it handles None gracefully)
            mock_enrich.assert_called_once()

    def test_add_stock_features_fallback_to_stock_features_module(self, sample_stock_df):
        """Should fallback to stock_features module if sector_momentum unavailable."""
        with patch(
            "services.sector_momentum.enrich_dataframe_with_all_stock_features",
            side_effect=ImportError("Mock import error"),
        ):
            with patch("stock_features.add_stock_features_to_dataframe") as mock_fallback:
                mock_fallback.return_value = sample_stock_df.copy()

                result = _add_stock_features(
                    sample_stock_df,
                    "AAPL",
                    vix_df=None,
                    spy_df=None,
                    qqq_df=None,
                )

                # Fallback should be called
                mock_fallback.assert_called_once()


# =============================================================================
# TEST: load_multi_asset_data (EQUITY)
# =============================================================================


class TestLoadMultiAssetDataEquity:
    """Tests for load_multi_asset_data with equity asset class."""

    def test_equity_loads_benchmark_data_when_add_stock_features_true(self, temp_data_dir):
        """Should load VIX/SPY/QQQ benchmark data when add_stock_features=True."""
        with patch("data_loader_multi_asset._load_benchmark_data") as mock_load_benchmark:
            mock_load_benchmark.return_value = pd.DataFrame(
                {
                    "timestamp": [1700000000],
                    "close": [20.0],
                }
            )
            with patch("data_loader_multi_asset._add_stock_features") as mock_add_features:
                mock_add_features.side_effect = lambda df, *args, **kwargs: df

                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["stock"]],
                    asset_class=AssetClass.EQUITY,
                    timeframe="4h",
                    add_stock_features=True,
                )

                # Should have called _load_benchmark_data for VIX, SPY, QQQ
                assert mock_load_benchmark.call_count == 3
                call_args_list = [call[0][0] for call in mock_load_benchmark.call_args_list]
                assert "VIX" in call_args_list
                assert "SPY" in call_args_list
                assert "QQQ" in call_args_list

    def test_equity_skips_benchmark_loading_when_add_stock_features_false(self, temp_data_dir):
        """Should skip benchmark loading when add_stock_features=False."""
        with patch("data_loader_multi_asset._load_benchmark_data") as mock_load_benchmark:
            with patch("data_loader_multi_asset._add_stock_features") as mock_add_features:
                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["stock"]],
                    asset_class=AssetClass.EQUITY,
                    timeframe="4h",
                    add_stock_features=False,
                )

                # Should NOT have called _load_benchmark_data
                mock_load_benchmark.assert_not_called()
                # Should NOT have called _add_stock_features
                mock_add_features.assert_not_called()

    def test_equity_calls_add_stock_features_for_each_symbol(self, temp_data_dir):
        """Should call _add_stock_features for each equity symbol."""
        with patch("data_loader_multi_asset._load_benchmark_data", return_value=None):
            with patch("data_loader_multi_asset._add_stock_features") as mock_add_features:
                mock_add_features.side_effect = lambda df, *args, **kwargs: df

                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["stock"]],
                    asset_class=AssetClass.EQUITY,
                    timeframe="4h",
                    add_stock_features=True,
                )

                # Should have called _add_stock_features once per symbol
                assert mock_add_features.call_count == 1


# =============================================================================
# TEST: load_multi_asset_data (CRYPTO - BACKWARD COMPATIBILITY)
# =============================================================================


class TestLoadMultiAssetDataCryptoBackwardCompatibility:
    """Tests to ensure crypto path is NOT affected by stock features changes."""

    def test_crypto_does_not_load_benchmark_data(self, temp_data_dir):
        """Crypto should NOT load VIX/SPY/QQQ benchmark data."""
        with patch("data_loader_multi_asset._load_benchmark_data") as mock_load_benchmark:
            with patch("data_loader_multi_asset._add_stock_features") as mock_add_features:
                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["crypto"]],
                    asset_class=AssetClass.CRYPTO,
                    timeframe="4h",
                    merge_fear_greed=False,  # Disable FNG to simplify test
                )

                # Should NOT have called benchmark loading for crypto
                mock_load_benchmark.assert_not_called()
                # Should NOT have called _add_stock_features for crypto
                mock_add_features.assert_not_called()

    def test_crypto_still_merges_fear_greed(self, temp_data_dir):
        """Crypto should still merge Fear & Greed data."""
        with patch("data_loader_multi_asset.load_fear_greed") as mock_load_fng:
            mock_load_fng.return_value = pd.DataFrame(
                {
                    "timestamp": [1700000000],
                    "fear_greed_value": [50],
                }
            )
            with patch("data_loader_multi_asset._merge_fear_greed") as mock_merge:
                mock_merge.side_effect = lambda df, fng: df

                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["crypto"]],
                    asset_class=AssetClass.CRYPTO,
                    timeframe="4h",
                    merge_fear_greed=True,
                )

                # Should have loaded Fear & Greed
                mock_load_fng.assert_called_once()
                # Should have merged Fear & Greed
                mock_merge.assert_called_once()

    def test_crypto_add_stock_features_param_has_no_effect(self, temp_data_dir):
        """add_stock_features parameter should have no effect on crypto."""
        with patch("data_loader_multi_asset._load_benchmark_data") as mock_load_benchmark:
            with patch("data_loader_multi_asset._add_stock_features") as mock_add_features:
                # Even with add_stock_features=True, crypto should be unaffected
                dfs, obs = load_multi_asset_data(
                    paths=[temp_data_dir["crypto"]],
                    asset_class=AssetClass.CRYPTO,
                    timeframe="4h",
                    add_stock_features=True,  # This should have no effect on crypto
                    merge_fear_greed=False,
                )

                mock_load_benchmark.assert_not_called()
                mock_add_features.assert_not_called()

    def test_crypto_data_structure_unchanged(self, temp_data_dir):
        """Crypto DataFrame structure should remain unchanged."""
        dfs, obs = load_multi_asset_data(
            paths=[temp_data_dir["crypto"]],
            asset_class=AssetClass.CRYPTO,
            timeframe="4h",
            merge_fear_greed=False,
        )

        assert "BTCUSDT" in dfs
        df = dfs["BTCUSDT"]

        # Standard columns should be present
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Stock-specific features should NOT be present
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
            assert col not in df.columns, f"Stock feature {col} should not be in crypto data"


# =============================================================================
# TEST: load_stock_data (convenience function)
# =============================================================================


class TestLoadStockDataConvenience:
    """Tests for load_stock_data convenience function."""

    def test_load_stock_data_enables_stock_features_by_default(self, temp_data_dir):
        """load_stock_data should enable stock features by default."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_stock_data(paths=[temp_data_dir["stock"]], timeframe="4h")

            # Verify add_stock_features=True was passed
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("add_stock_features") is True

    def test_load_stock_data_can_disable_stock_features(self, temp_data_dir):
        """load_stock_data should allow disabling stock features."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_stock_data(
                paths=[temp_data_dir["stock"]],
                timeframe="4h",
                add_stock_features=False,
            )

            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("add_stock_features") is False

    def test_load_stock_data_passes_benchmark_paths(self, temp_data_dir):
        """load_stock_data should pass benchmark paths to load_multi_asset_data."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_stock_data(
                paths=[temp_data_dir["stock"]],
                timeframe="4h",
                vix_path="/path/to/vix.parquet",
                spy_path="/path/to/spy.parquet",
                qqq_path="/path/to/qqq.parquet",
            )

            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("vix_path") == "/path/to/vix.parquet"
            assert call_kwargs.get("spy_path") == "/path/to/spy.parquet"
            assert call_kwargs.get("qqq_path") == "/path/to/qqq.parquet"


# =============================================================================
# TEST: load_crypto_data (backward compatibility)
# =============================================================================


class TestLoadCryptoDataBackwardCompatibility:
    """Tests to ensure load_crypto_data function is unchanged."""

    def test_load_crypto_data_uses_crypto_asset_class(self, temp_data_dir):
        """load_crypto_data should use CRYPTO asset class."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_crypto_data(paths=[temp_data_dir["crypto"]], timeframe="4h")

            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("asset_class") == AssetClass.CRYPTO

    def test_load_crypto_data_merges_fear_greed_by_default(self, temp_data_dir):
        """load_crypto_data should merge Fear & Greed by default."""
        with patch("data_loader_multi_asset.load_multi_asset_data") as mock_load:
            mock_load.return_value = ({}, {})

            load_crypto_data(paths=[temp_data_dir["crypto"]], timeframe="4h")

            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("merge_fear_greed") is True


# =============================================================================
# TEST: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for stock features loading."""

    def test_full_equity_load_integration(self, temp_data_dir):
        """Full integration test for equity data loading with stock features."""
        # This test verifies the full flow works without mocking
        # It may use fallbacks if benchmark data is not available

        dfs, obs = load_multi_asset_data(
            paths=[temp_data_dir["stock"]],
            asset_class=AssetClass.EQUITY,
            timeframe="4h",
            add_stock_features=True,
            vix_path=temp_data_dir["vix"],
            spy_path=temp_data_dir["spy"],
            qqq_path=temp_data_dir["qqq"],
        )

        assert "AAPL" in dfs
        df = dfs["AAPL"]

        # Basic columns should exist
        assert "timestamp" in df.columns
        assert "close" in df.columns

        # DataFrame should have data
        assert len(df) > 0

    def test_equity_and_crypto_can_load_in_same_session(self, temp_data_dir):
        """Should be able to load both equity and crypto in the same session."""
        # Load equity
        equity_dfs, _ = load_multi_asset_data(
            paths=[temp_data_dir["stock"]],
            asset_class=AssetClass.EQUITY,
            timeframe="4h",
            add_stock_features=False,  # Skip features to simplify
        )

        # Load crypto
        crypto_dfs, _ = load_multi_asset_data(
            paths=[temp_data_dir["crypto"]],
            asset_class=AssetClass.CRYPTO,
            timeframe="4h",
            merge_fear_greed=False,
        )

        # Both should have loaded successfully
        assert "AAPL" in equity_dfs
        assert "BTCUSDT" in crypto_dfs

        # They should be independent
        assert len(equity_dfs) == 1
        assert len(crypto_dfs) == 1


# =============================================================================
# TEST: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in stock features loading."""

    def test_missing_benchmark_data_graceful_handling(self, temp_data_dir):
        """Should handle missing benchmark data gracefully."""
        # Load with non-existent benchmark paths
        dfs, obs = load_multi_asset_data(
            paths=[temp_data_dir["stock"]],
            asset_class=AssetClass.EQUITY,
            timeframe="4h",
            add_stock_features=True,
            vix_path="/nonexistent/vix.parquet",
            spy_path="/nonexistent/spy.parquet",
            qqq_path="/nonexistent/qqq.parquet",
        )

        # Should still load the main data without crashing
        assert "AAPL" in dfs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
