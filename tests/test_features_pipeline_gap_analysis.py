# -*- coding: utf-8 -*-
"""
Tests for gap analysis features in features_pipeline.py.

Tests cover:
- compute_gap_features()
- Gap percentage calculation
- Gap direction and magnitude
- Gap fill detection
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestComputeGapFeatures:
    """Tests for compute_gap_features function."""

    def test_basic_gap_calculation(self):
        """Test basic gap percentage calculation."""
        from features_pipeline import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 110.0, 105.0],
                "high": [101.0, 112.0, 108.0],
                "low": [99.0, 108.0, 103.0],
                "close": [100.0, 111.0, 106.0],
            }
        )

        result = compute_gap_features(df)

        # First row: no previous close, should be NaN or 0
        assert pd.isna(result.iloc[0]["gap_pct"]) or result.iloc[0]["gap_pct"] == 0

        # Second row: open=110, prev_close=100 → 10% gap
        assert result.iloc[1]["gap_pct"] == pytest.approx(10.0, rel=0.1)

    def test_gap_magnitude_classification(self):
        """Test gap magnitude classification."""
        from features_pipeline import compute_gap_features

        # Create gaps of different sizes
        df = pd.DataFrame(
            {
                "open": [100.0, 100.3, 102.5, 106.0, 115.0],
                "high": [101.0, 101.0, 103.0, 107.0, 116.0],
                "low": [99.0, 99.0, 101.0, 105.0, 114.0],
                "close": [100.0, 100.0, 102.0, 106.0, 115.0],
            }
        )

        result = compute_gap_features(df)

        # Magnitudes should be present
        assert "gap_magnitude" in result.columns

    def test_gap_filled_detection(self):
        """Test gap filled detection."""
        from features_pipeline import compute_gap_features

        # Gap up that gets filled
        df_filled = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [101.0, 112.0],
                "low": [99.0, 99.0],  # Low reaches below previous close → filled
                "close": [100.0, 108.0],
            }
        )

        result_filled = compute_gap_features(df_filled)
        assert "gap_filled" in result_filled.columns

    def test_gap_fill_ratio(self):
        """Test gap fill ratio calculation."""
        from features_pipeline import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [101.0, 112.0],
                "low": [99.0, 105.0],
                "close": [100.0, 108.0],
            }
        )

        result = compute_gap_features(df)
        assert "gap_fill_ratio" in result.columns

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        from features_pipeline import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
                "volume": [1000, 1100],
                "custom": ["a", "b"],
            }
        )

        result = compute_gap_features(df)

        assert "volume" in result.columns
        assert "custom" in result.columns


class TestGapFeatureNormalization:
    """Tests for gap feature normalization."""

    def test_gap_pct_zscore(self):
        """Test that gap_pct can be z-score normalized."""
        from features_pipeline import compute_gap_features

        # Create data with known gap distribution
        np.random.seed(42)
        n = 100
        opens = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        closes = opens + np.random.randn(n) * 0.3
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.2
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.2

        # Add some gaps
        opens[1:] = closes[:-1] * (1 + np.random.randn(n - 1) * 0.01)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )

        result = compute_gap_features(df)

        # Gap percentages should be reasonable
        gap_pcts = result["gap_pct"].dropna()
        assert gap_pcts.std() < 10.0  # Std reasonable


class TestEdgeCases:
    """Tests for edge cases in gap analysis."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        from features_pipeline import compute_gap_features

        df = pd.DataFrame(columns=["open", "high", "low", "close"])

        result = compute_gap_features(df)

        assert len(result) == 0
        assert "gap_pct" in result.columns


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_original_columns_unchanged(self):
        """Test that original columns are not modified."""
        from features_pipeline import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
            }
        )

        original_close = df["close"].copy()

        result = compute_gap_features(df)

        pd.testing.assert_series_equal(result["close"], original_close)

    def test_index_preserved(self):
        """Test that index is preserved."""
        from features_pipeline import compute_gap_features

        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
            },
            index=dates,
        )

        result = compute_gap_features(df)

        pd.testing.assert_index_equal(result.index, df.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
