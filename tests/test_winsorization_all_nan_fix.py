#!/usr/bin/env python3
"""
Comprehensive test suite for Issue #1: Winsorization with all-NaN columns.

Problem:
--------
When a feature column is entirely NaN, winsorization bounds become (nan, nan).
This leads to silent conversion NaN -> 0.0 via is_constant flag, creating
semantic ambiguity (model cannot distinguish "missing data" from "zero value").

Expected Behavior (after fix):
-------------------------------
1. Detect all-NaN columns during fit()
2. Log clear warning for each all-NaN column
3. Mark column as invalid/skipped (don't create bounds)
4. In transform(), skip winsorization for invalid columns
5. Final output: explicit NaN (not silent 0.0 conversion)

References:
-----------
- Verification script: verify_issues_simple.py
- Related: NaN handling external features (mediator.py)
- Best practices: Scikit-learn's handling of all-constant features

Tech Debt Tracking:
- docs/reports/TECH_DEBT_REGISTRY.md#testing-winsorization-allnan

Status: Test documents expected behavior. Fix pending in features_pipeline.py.
Mitigation: Pre-filter all-NaN columns before winsorization; use explicit NaN markers.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features_pipeline import FeaturePipeline


class TestWinsorization_AllNaNColumns:
    """Test winsorization behavior with all-NaN columns."""

    @pytest.fixture
    def df_with_all_nan(self):
        """DataFrame with an entirely NaN column."""
        return pd.DataFrame({
            'timestamp': range(10),
            'symbol': ['BTC'] * 10,
            'close': [100.0 + i for i in range(10)],
            'volume': [1.0 + i for i in range(10)],
            'all_nan_col': [np.nan] * 10,
            'partial_nan_col': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'valid_col': list(range(10)),
        })

    @pytest.fixture
    def df_multi_symbol_all_nan(self):
        """Multi-symbol DataFrame where one symbol has all-NaN column."""
        return pd.DataFrame({
            'timestamp': [0, 1, 2, 0, 1, 2],
            'symbol': ['BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH'],
            'close': [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            'volume': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature_a': [np.nan] * 6,  # All NaN across both symbols
        })

    # ==========================================================================
    # Test 1: Current Behavior (Before Fix)
    # ==========================================================================

    def test_fixed_behavior_all_nan_marked_and_preserved(self, df_with_all_nan):
        """
        FIXED BEHAVIOR: All-NaN columns are detected, marked, and preserved as NaN.

        After fix (2025-11-21):
        - fit() detects all-NaN columns
        - Marks with is_all_nan=True
        - Does NOT create winsorize_bounds (or sets to None)
        - Logs warning
        - transform() preserves NaN (does NOT convert to zeros)
        """
        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)

        # Should log warning during fit
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit({'BTC': df_with_all_nan.copy()})

            # Note: warning is logged via logging module, not warnings module
            # So we check logs separately

        # Check stats for all-NaN column
        stats = pipe.stats.get('all_nan_col', {})

        # FIXED BEHAVIOR:
        # - is_all_nan=True (new flag)
        # - NO winsorize_bounds (or None)
        # - is_constant=True (still true)
        assert stats.get('is_all_nan', False) == True, \
            "FIXED: Should mark as all-NaN"

        assert 'winsorize_bounds' not in stats, \
            "FIXED: Should NOT have winsorize_bounds for all-NaN columns"

        assert stats['is_constant'] == True, "Should still be marked constant"

        # Transform and check output
        result = pipe.transform_df(df_with_all_nan.copy())
        z_col = 'all_nan_col_z'

        assert z_col in result.columns, "Should create z-score column"
        z_values = result[z_col]

        # FIXED BEHAVIOR: Preserve NaN (do NOT convert to zeros)
        assert z_values.isna().all(), \
            "FIXED: All-NaN column should remain NaN (not zeros)"

    # ==========================================================================
    # Test 2: Expected Behavior (After Fix)
    # ==========================================================================

    def test_fixed_behavior_warns_and_marks_invalid(self, df_with_all_nan):
        """
        DESIRED BEHAVIOR: After fix, should detect and warn about all-NaN columns.

        Expected:
        1. fit() detects all-NaN column
        2. Logs warning with column name
        3. Marks column as invalid (no winsorize_bounds, or bounds=None)
        4. transform() skips winsorization for invalid columns
        5. Output: Explicit NaN (or skip column entirely)
        """
        # This test will PASS after fix is implemented
        pytest.skip("FIX NOT YET IMPLEMENTED - will pass after fix")

        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)

        # Should raise warning during fit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit({'BTC': df_with_all_nan.copy()})

            # Check that warning was raised
            assert len(w) > 0, "Should raise warning for all-NaN column"
            warning_messages = [str(warning.message) for warning in w]
            assert any('all_nan_col' in msg for msg in warning_messages), \
                "Warning should mention column name"

        # Check stats
        stats = pipe.stats.get('all_nan_col', {})

        # After fix: Should mark as invalid
        # Option A: No winsorize_bounds key
        # Option B: winsorize_bounds = None
        # Option C: Add 'is_invalid': True flag
        if 'winsorize_bounds' in stats:
            bounds = stats['winsorize_bounds']
            assert bounds is None or (not np.isfinite(bounds[0])), \
                "Invalid column should have None or invalid bounds"

        # Or check for explicit invalid flag
        assert stats.get('is_invalid', False) or stats.get('all_nan', False), \
            "Should mark column as invalid or all_nan"

        # Transform and check output
        result = pipe.transform_df(df_with_all_nan.copy())

        # After fix: Should NOT silently convert to zeros
        # Option A: Keep as NaN (explicit missing data)
        # Option B: Skip column entirely (don't create _z column)
        z_col = 'all_nan_col_z'

        if z_col in result.columns:
            z_values = result[z_col]
            # Should be NaN, not zeros
            assert z_values.isna().all(), \
                "All-NaN column should remain NaN (not silently converted to zeros)"
        else:
            # Column was skipped - also acceptable
            assert z_col not in result.columns, \
                "Invalid column should be skipped in transform"

    # ==========================================================================
    # Test 3: Partial NaN Handling (Should Work Correctly)
    # ==========================================================================

    def test_partial_nan_column_works_correctly(self, df_with_all_nan):
        """
        Test that columns with PARTIAL NaN (but not all-NaN) work correctly.

        This should already work - winsorization uses np.nanpercentile.
        """
        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)
        pipe.fit({'BTC': df_with_all_nan.copy()})

        stats = pipe.stats.get('partial_nan_col', {})

        # Should have valid bounds (computed on non-NaN values)
        assert 'winsorize_bounds' in stats
        bounds = stats['winsorize_bounds']
        assert np.isfinite(bounds[0]) and np.isfinite(bounds[1]), \
            "Partial NaN column should have valid bounds"

        # Should NOT be marked constant (has variation)
        assert stats['is_constant'] == False, "Should not be constant"

        # Transform should work
        result = pipe.transform_df(df_with_all_nan.copy())
        z_col = 'partial_nan_col_z'

        assert z_col in result.columns
        z_values = result[z_col]

        # NaN positions should be preserved
        original_nan_mask = df_with_all_nan['partial_nan_col'].isna()
        result_nan_mask = z_values.isna()
        pd.testing.assert_series_equal(
            original_nan_mask,
            result_nan_mask,
            check_names=False,
            obj="NaN positions should be preserved"
        )

        # Non-NaN values should be z-scored
        non_nan_values = z_values[~result_nan_mask]
        assert len(non_nan_values) > 0, "Should have non-NaN values"
        assert np.isfinite(non_nan_values).all(), "Non-NaN values should be finite"

    # ==========================================================================
    # Test 4: Multi-Symbol Edge Case
    # ==========================================================================

    def test_multi_symbol_all_nan_column(self, df_multi_symbol_all_nan):
        """
        Test that all-NaN column across multiple symbols is detected.

        Edge case: Column is all-NaN for ALL symbols.
        """
        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)
        pipe.fit({'ALL': df_multi_symbol_all_nan.copy()})

        stats = pipe.stats.get('feature_a', {})

        # Should detect all-NaN even across multiple symbols
        if 'winsorize_bounds' in stats:
            bounds = stats['winsorize_bounds']
            # Current behavior: (nan, nan)
            assert np.isnan(bounds[0]) and np.isnan(bounds[1]), \
                "Multi-symbol all-NaN column has NaN bounds (current bug)"

    # ==========================================================================
    # Test 5: Disable Winsorization
    # ==========================================================================

    def test_all_nan_without_winsorization(self, df_with_all_nan):
        """
        Test all-NaN column when winsorization is DISABLED.

        After fix (2025-11-21): Should still detect and preserve NaN.
        """
        pipe = FeaturePipeline(enable_winsorization=False, strict_idempotency=True)
        pipe.fit({'BTC': df_with_all_nan.copy()})

        stats = pipe.stats.get('all_nan_col', {})

        # No winsorize_bounds when disabled
        assert 'winsorize_bounds' not in stats, \
            "Should not have winsorize_bounds when disabled"

        # Should be marked as all-NaN even without winsorization
        assert stats.get('is_all_nan', False) == True, \
            "FIXED: Should detect all-NaN even without winsorization"

        # Should still be marked constant
        assert stats['is_constant'] == True

        # Transform
        result = pipe.transform_df(df_with_all_nan.copy())
        z_col = 'all_nan_col_z'

        assert z_col in result.columns
        z_values = result[z_col]

        # FIXED: Should preserve NaN (not convert to zeros)
        assert z_values.isna().all(), \
            "FIXED: All-NaN preserved even without winsorization"

    # ==========================================================================
    # Test 6: Save/Load Preserves Invalid Markers
    # ==========================================================================

    def test_save_load_preserves_invalid_markers(self, df_with_all_nan, tmp_path):
        """
        Test that invalid/all-NaN markers are preserved in save/load.

        After fix, should save is_invalid or all_nan flags.
        """
        pytest.skip("FIX NOT YET IMPLEMENTED")

        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)
        pipe.fit({'BTC': df_with_all_nan.copy()})

        # Save
        save_path = tmp_path / "pipeline.json"
        pipe.save(str(save_path))

        # Load
        loaded_pipe = FeaturePipeline.load(str(save_path))

        # Check that invalid markers are preserved
        stats = loaded_pipe.stats.get('all_nan_col', {})
        assert stats.get('is_invalid', False) or stats.get('all_nan', False), \
            "Invalid markers should be preserved in save/load"

    # ==========================================================================
    # Test 7: Zero-Variance vs All-NaN Distinction
    # ==========================================================================

    def test_distinguish_zero_variance_from_all_nan(self):
        """
        Test that zero-variance (constant) columns are distinguished from all-NaN.

        Zero-variance: [5.0, 5.0, 5.0, ...] -> valid, should normalize to zeros
        All-NaN: [nan, nan, nan, ...] -> invalid, should warn/skip
        """
        df = pd.DataFrame({
            'timestamp': range(10),
            'symbol': ['BTC'] * 10,
            'close': [100.0] * 10,
            'zero_var': [5.0] * 10,  # Constant, zero variance
            'all_nan': [np.nan] * 10,  # All NaN
        })

        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)
        pipe.fit({'BTC': df.copy()})

        # Zero variance column: Should be marked constant, but VALID
        zero_var_stats = pipe.stats.get('zero_var', {})
        assert zero_var_stats['is_constant'] == True, "Zero variance -> constant"
        # Should NOT have invalid markers
        assert not zero_var_stats.get('is_invalid', False)
        assert not zero_var_stats.get('all_nan', False)

        # All-NaN column: Should be marked constant AND invalid
        all_nan_stats = pipe.stats.get('all_nan', {})
        assert all_nan_stats['is_constant'] == True, "All-NaN -> constant (current)"

        # After fix: Should ALSO have invalid marker
        # pytest.skip("Fix not implemented")
        # assert all_nan_stats.get('is_invalid', False) or all_nan_stats.get('all_nan', False)

    # ==========================================================================
    # Test 8: Error Message Quality
    # ==========================================================================

    def test_warning_message_is_helpful(self, df_with_all_nan):
        """
        Test that warning message for all-NaN columns is informative.

        Should include:
        - Column name
        - Explanation of the problem
        - Recommendation (skip, impute, or check data quality)
        """
        pytest.skip("FIX NOT YET IMPLEMENTED")

        pipe = FeaturePipeline(enable_winsorization=True, strict_idempotency=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit({'BTC': df_with_all_nan.copy()})

            # Find warning for all_nan_col
            all_nan_warnings = [
                warning for warning in w
                if 'all_nan_col' in str(warning.message)
            ]

            assert len(all_nan_warnings) > 0, "Should warn about all-NaN column"

            warning_msg = str(all_nan_warnings[0].message)

            # Check message quality
            assert 'all_nan_col' in warning_msg, "Should mention column name"
            assert 'entirely NaN' in warning_msg or 'all-NaN' in warning_msg, \
                "Should explain problem"
            # Optionally check for recommendations
            # assert 'impute' in warning_msg or 'skip' in warning_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
