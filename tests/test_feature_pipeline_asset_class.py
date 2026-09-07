# -*- coding: utf-8 -*-
"""
tests/test_feature_pipeline_asset_class.py
Tests for FeaturePipeline asset_class awareness fix.

FIX (2025-11-28): Tests for Issue #5 "FeaturePipeline не asset_class aware"
Reference: docs/PLATFORM_REFERENCE.md → Issue #5

These tests verify:
1. FeaturePipeline accepts asset_class parameter
2. Stock features are automatically added for equity
3. Crypto remains unchanged (backward compatibility)
4. Save/load preserves asset_class configuration
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features_pipeline import FeaturePipeline, add_stock_features


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 100

    timestamps = [1700000000 + i * 14400 for i in range(n)]  # 4h intervals
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * n,
        "open": prices + np.random.randn(n) * 0.1,
        "high": prices + abs(np.random.randn(n) * 0.3),
        "low": prices - abs(np.random.randn(n) * 0.3),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })

    return df


@pytest.fixture
def sample_crypto_df():
    """Create sample crypto DataFrame."""
    np.random.seed(42)
    n = 100

    timestamps = [1700000000 + i * 14400 for i in range(n)]
    base_price = 50000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["BTCUSDT"] * n,
        "open": prices + np.random.randn(n) * 10,
        "high": prices + abs(np.random.randn(n) * 30),
        "low": prices - abs(np.random.randn(n) * 30),
        "close": prices,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })

    return df


@pytest.fixture
def fitted_pipeline():
    """Create a fitted FeaturePipeline."""
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        "timestamp": range(n),
        "symbol": ["TEST"] * n,
        "close": 100.0 + np.random.randn(n),
        "volume": 1000.0 + np.random.randn(n) * 100,
    })

    pipe = FeaturePipeline()
    pipe.fit({"TEST": df})
    return pipe


# =============================================================================
# TEST: ASSET CLASS PARAMETER
# =============================================================================


class TestAssetClassParameter:
    """Tests for asset_class parameter in FeaturePipeline."""

    def test_default_asset_class_is_none(self):
        """Default asset_class should be None for backward compatibility."""
        pipe = FeaturePipeline()
        assert pipe.asset_class is None

    def test_asset_class_crypto(self):
        """Asset class can be set to crypto."""
        pipe = FeaturePipeline(asset_class="crypto")
        assert pipe.asset_class == "crypto"

    def test_asset_class_equity(self):
        """Asset class can be set to equity."""
        pipe = FeaturePipeline(asset_class="equity")
        assert pipe.asset_class == "equity"

    def test_asset_class_case_insensitive(self):
        """Asset class should be case-insensitive."""
        pipe = FeaturePipeline(asset_class="EQUITY")
        assert pipe.asset_class == "equity"

        pipe2 = FeaturePipeline(asset_class="Crypto")
        assert pipe2.asset_class == "crypto"

    def test_invalid_asset_class_raises(self):
        """Invalid asset class should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid asset_class"):
            FeaturePipeline(asset_class="invalid")

    def test_auto_stock_features_default_true(self):
        """auto_stock_features should default to True."""
        pipe = FeaturePipeline()
        assert pipe.auto_stock_features is True

    def test_auto_stock_features_can_be_disabled(self):
        """auto_stock_features can be disabled."""
        pipe = FeaturePipeline(asset_class="equity", auto_stock_features=False)
        assert pipe.auto_stock_features is False


# =============================================================================
# TEST: AUTO STOCK FEATURES
# =============================================================================


class TestAutoStockFeatures:
    """Tests for automatic stock feature addition."""

    def test_equity_adds_gap_features(self, sample_ohlcv_df):
        """Equity pipeline should add gap features."""
        pipe = FeaturePipeline(asset_class="equity")
        # Need to fit first
        pipe.fit({"AAPL": sample_ohlcv_df})

        result = pipe.transform_df(sample_ohlcv_df)

        # Should have gap_pct (added by auto_stock_features)
        assert "gap_pct" in result.columns

    def test_crypto_does_not_add_gap_features(self, sample_crypto_df):
        """Crypto pipeline should NOT add gap features automatically."""
        pipe = FeaturePipeline(asset_class="crypto")
        pipe.fit({"BTCUSDT": sample_crypto_df})

        result = pipe.transform_df(sample_crypto_df)

        # Should NOT have gap_pct
        assert "gap_pct" not in result.columns

    def test_none_asset_class_does_not_add_gap_features(self, sample_ohlcv_df):
        """None asset_class (default) should NOT add gap features."""
        pipe = FeaturePipeline(asset_class=None)
        pipe.fit({"AAPL": sample_ohlcv_df})

        result = pipe.transform_df(sample_ohlcv_df)

        assert "gap_pct" not in result.columns

    def test_auto_stock_features_disabled(self, sample_ohlcv_df):
        """Disabled auto_stock_features should not add features."""
        pipe = FeaturePipeline(asset_class="equity", auto_stock_features=False)
        pipe.fit({"AAPL": sample_ohlcv_df})

        result = pipe.transform_df(sample_ohlcv_df)

        # gap_pct should NOT be present
        assert "gap_pct" not in result.columns

    def test_features_not_duplicated(self, sample_ohlcv_df):
        """If gap_pct already exists, should not be duplicated."""
        # Add gap features manually first
        df_with_gaps = add_stock_features(sample_ohlcv_df.copy(), "AAPL")

        pipe = FeaturePipeline(asset_class="equity")
        pipe.fit({"AAPL": df_with_gaps})

        result = pipe.transform_df(df_with_gaps)

        # Should still have exactly one gap_pct column
        gap_cols = [c for c in result.columns if c == "gap_pct"]
        assert len(gap_cols) == 1


# =============================================================================
# TEST: SAVE/LOAD PERSISTENCE
# =============================================================================


class TestSaveLoadPersistence:
    """Tests for save/load of asset_class configuration."""

    def test_save_preserves_asset_class(self, fitted_pipeline):
        """Save should preserve asset_class in config."""
        pipe = FeaturePipeline(asset_class="equity")
        pipe.fit({"TEST": pd.DataFrame({
            "timestamp": range(10),
            "symbol": ["TEST"] * 10,
            "close": [100.0] * 10,
        })})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            pipe.save(path)

            # Read JSON and verify
            with open(path, "r") as f:
                data = json.load(f)

            assert data["config"]["asset_class"] == "equity"
            assert data["config"]["auto_stock_features"] is True
        finally:
            os.unlink(path)

    def test_load_restores_asset_class(self):
        """Load should restore asset_class from config."""
        # Create and save pipeline
        pipe = FeaturePipeline(asset_class="equity", auto_stock_features=False)
        pipe.fit({"TEST": pd.DataFrame({
            "timestamp": range(10),
            "symbol": ["TEST"] * 10,
            "close": [100.0] * 10,
        })})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            pipe.save(path)

            # Load and verify
            loaded = FeaturePipeline.load(path)

            assert loaded.asset_class == "equity"
            assert loaded.auto_stock_features is False
        finally:
            os.unlink(path)

    def test_load_legacy_file_defaults_to_none(self):
        """Loading legacy file without asset_class should default to None."""
        # Create minimal legacy JSON (no asset_class in config)
        legacy_data = {
            "stats": {"close": {"mean": 100.0, "std": 1.0}},
            "metadata": {},
            "config": {
                "enable_winsorization": True,
                "winsorize_percentiles": [1.0, 99.0],
                "strict_idempotency": True,
                "preserve_close_orig": True,
                # Note: no asset_class or auto_stock_features
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            path = f.name

        try:
            loaded = FeaturePipeline.load(path)

            # Should default to None
            assert loaded.asset_class is None
            # auto_stock_features should default to True
            assert loaded.auto_stock_features is True
        finally:
            os.unlink(path)


# =============================================================================
# TEST: BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with crypto."""

    def test_crypto_pipeline_unchanged(self, sample_crypto_df):
        """Crypto pipeline behavior should be unchanged."""
        # Old behavior (no asset_class)
        pipe_old = FeaturePipeline()
        pipe_old.fit({"BTCUSDT": sample_crypto_df})
        result_old = pipe_old.transform_df(sample_crypto_df.copy())

        # New behavior (explicit crypto)
        pipe_new = FeaturePipeline(asset_class="crypto")
        pipe_new.fit({"BTCUSDT": sample_crypto_df})
        result_new = pipe_new.transform_df(sample_crypto_df.copy())

        # Should have same columns
        assert set(result_old.columns) == set(result_new.columns)

    def test_existing_code_works_without_changes(self, sample_ohlcv_df):
        """Existing code without asset_class should work unchanged."""
        # This is how existing code uses FeaturePipeline
        pipe = FeaturePipeline()
        pipe.fit({"AAPL": sample_ohlcv_df})
        result = pipe.transform_df(sample_ohlcv_df)

        # Should work without errors
        assert len(result) == len(sample_ohlcv_df)

        # Should have normalized columns
        assert "close_z" in result.columns

    def test_fit_transform_works_for_equity(self, sample_ohlcv_df):
        """Full fit/transform cycle should work for equity."""
        pipe = FeaturePipeline(asset_class="equity")
        pipe.fit({"AAPL": sample_ohlcv_df})
        result = pipe.transform_df(sample_ohlcv_df)

        assert len(result) == len(sample_ohlcv_df)
        assert "close_z" in result.columns
        assert "gap_pct" in result.columns  # Auto-added for equity


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Empty DataFrame should not crash."""
        pipe = FeaturePipeline(asset_class="equity")

        # Create minimal stats for transform
        pipe.stats = {"close": {"mean": 100.0, "std": 1.0}}

        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close", "open", "high", "low", "volume"])

        result = pipe.transform_df(empty_df)
        assert len(result) == 0

    def test_missing_symbol_column(self):
        """DataFrame without symbol column should still work."""
        df = pd.DataFrame({
            "timestamp": range(10),
            "close": [100.0] * 10,
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "volume": [1000.0] * 10,
        })

        pipe = FeaturePipeline(asset_class="equity")
        pipe.fit({"TEST": df})

        # Should work, symbol defaults to "UNKNOWN"
        result = pipe.transform_df(df)
        assert len(result) == 10

    def test_transform_dict_with_equity(self, sample_ohlcv_df):
        """transform_dict should apply stock features to all DataFrames."""
        pipe = FeaturePipeline(asset_class="equity")

        # Create dict with multiple symbols
        dfs = {
            "AAPL": sample_ohlcv_df.copy(),
            "MSFT": sample_ohlcv_df.copy(),
        }
        dfs["MSFT"]["symbol"] = "MSFT"

        pipe.fit(dfs)
        result = pipe.transform_dict(dfs)

        # Both should have gap_pct
        assert "gap_pct" in result["AAPL"].columns
        assert "gap_pct" in result["MSFT"].columns


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
