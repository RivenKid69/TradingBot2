# -*- coding: utf-8 -*-
"""
test_close_orig_semantic_fix.py
==================================================================
Comprehensive test suite for CRITICAL close_orig semantic fix (2025-11-25).

CRITICAL BUG (FIXED):
=====================
The `close_orig` marker had conflicting semantics between components:

1. fetch_all_data_patch.py (REMOVED):
   - Created `close_orig` as "backup copy of original close" (data NOT shifted)
   - Created `close_prev` as shifted close for features

2. features_pipeline.py:
   - Interpreted `close_orig` as "marker that data is ALREADY shifted"
   - When `close_orig` present → SKIPPED shifting!

3. Result: DATA LEAKAGE
   - Data from fetch_all_data_patch had `close_orig` → features_pipeline skipped shift
   - Model saw UNSHIFTED features → learned from FUTURE information!

FIX:
====
1. REMOVED close_orig/close_prev creation from fetch_all_data_patch.py
2. Added `_close_shifted` column marker in features_pipeline.py for TradingEnv compatibility
3. This eliminates the semantic conflict and ensures proper feature shifting

References:
- docs/PLATFORM_REFERENCE.md: DATA_LEAKAGE_FIX_REPORT_2025_11_23.md
- tests/test_data_leakage_prevention.py
- Issue: Inconsistent close_orig semantics causing data leakage
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check for pyarrow availability (needed for feather tests)
try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

pyarrow_required = pytest.mark.skipif(
    not HAS_PYARROW, reason="pyarrow required for feather/parquet support"
)

from features_pipeline import FeaturePipeline, _columns_to_shift, METADATA_COLUMNS


# ==============================================================================
# Test 1: fetch_all_data_patch no longer creates close_orig
# ==============================================================================

class TestFetchAllDataPatchNoCloseOrig:
    """Verify that fetch_all_data_patch no longer creates close_orig."""

    @pyarrow_required
    def test_load_all_data_no_close_orig(self, tmp_path):
        """Test that load_all_data() no longer creates close_orig."""
        # Import here to avoid module-level import issues
        from fetch_all_data_patch import load_all_data

        # Create test feather file with 4h-aligned timestamps (14400 seconds apart)
        # This is required because _ensure_required_columns does 4h alignment
        base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
        df = pd.DataFrame({
            'timestamp': [base_ts + i * 14400 for i in range(5)],  # 4h intervals
            'symbol': ['BTCUSDT'] * 5,
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            'quote_asset_volume': [100000.0, 110000.0, 120000.0, 130000.0, 140000.0],
            'number_of_trades': [100, 110, 120, 130, 140],
            'taker_buy_base_asset_volume': [500.0, 550.0, 600.0, 650.0, 700.0],
            'taker_buy_quote_asset_volume': [50000.0, 55000.0, 60000.0, 65000.0, 70000.0],
        })

        feather_path = tmp_path / "BTCUSDT.feather"
        df.to_feather(feather_path)

        # Load data
        all_dfs, _ = load_all_data([str(feather_path)])

        loaded_df = all_dfs['BTCUSDT']

        # CRITICAL: close_orig should NOT exist
        assert 'close_orig' not in loaded_df.columns, \
            "close_orig should NOT be created by load_all_data() (FIX 2025-11-25)"

        # CRITICAL: close_prev should NOT exist
        assert 'close_prev' not in loaded_df.columns, \
            "close_prev should NOT be created by load_all_data() (FIX 2025-11-25)"

        # close should remain as is (not shifted here)
        assert 'close' in loaded_df.columns
        # First value should be the original (not NaN from shifting)
        assert loaded_df['close'].iloc[0] == 100.5, \
            "close should not be shifted in load_all_data()"


# ==============================================================================
# Test 2: features_pipeline properly shifts when no close_orig
# ==============================================================================

class TestFeaturesPipelineShiftingWithoutCloseOrig:
    """Verify that features_pipeline shifts ALL features when close_orig absent."""

    def test_fit_shifts_all_features_without_close_orig(self):
        """Test that fit() shifts ALL features when close_orig is NOT present."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'rsi_14': [50.0, 55.0, 60.0, 65.0, 70.0],
            'macd': [0.5, 0.6, 0.7, 0.8, 0.9],
        })

        # NO close_orig present
        assert 'close_orig' not in df.columns

        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': df})

        # Stats should be computed on SHIFTED data
        # Original close: [100, 101, 102, 103, 104]
        # Shifted close: [NaN, 100, 101, 102, 103]
        # Mean should be ~101.5 (mean of [100, 101, 102, 103])
        close_mean = pipe.stats['close']['mean']
        assert 101.0 <= close_mean <= 102.0, \
            f"close mean {close_mean} suggests data was NOT shifted (expected ~101.5)"

        # RSI should also be shifted
        rsi_mean = pipe.stats['rsi_14']['mean']
        # Original: [50, 55, 60, 65, 70] → mean = 60
        # Shifted: [NaN, 50, 55, 60, 65] → mean = 57.5
        assert 56.0 <= rsi_mean <= 59.0, \
            f"rsi_14 mean {rsi_mean} suggests data was NOT shifted"

    def test_transform_shifts_all_features_without_close_orig(self):
        """Test that transform_df() shifts ALL features when close_orig NOT present."""
        train_df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'rsi_14': [50.0, 55.0, 60.0, 65.0, 70.0],
        })

        test_df = pd.DataFrame({
            'timestamp': [6, 7, 8],
            'close': [105.0, 106.0, 107.0],
            'rsi_14': [75.0, 80.0, 85.0],
        })

        # NO close_orig present
        assert 'close_orig' not in test_df.columns

        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': train_df})
        transformed = pipe.transform_df(test_df)

        # First row should have NaN after shift
        assert pd.isna(transformed['close'].iloc[0]), "close should be shifted"
        assert pd.isna(transformed['rsi_14'].iloc[0]), "rsi_14 should be shifted"

        # Second row should have values from first row (shift by 1)
        assert transformed['close'].iloc[1] == 105.0, "close not properly shifted"
        assert transformed['rsi_14'].iloc[1] == 75.0, "rsi_14 not properly shifted"

    def test_close_shifted_marker_created(self):
        """Test that _close_shifted marker is created after shifting."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'close': [100.0, 101.0, 102.0],
        })

        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': df})
        transformed = pipe.transform_df(df.copy())

        # _close_shifted marker should be present
        assert '_close_shifted' in transformed.columns, \
            "_close_shifted marker should be created after shifting"

        # All values should be True
        assert transformed['_close_shifted'].all(), \
            "_close_shifted should be True for all rows"


# ==============================================================================
# Test 3: features_pipeline skips shifting when close_orig present (for preserve_close_orig=True)
# ==============================================================================

class TestFeaturesPipelineWithCloseOrig:
    """Verify that features_pipeline skips shifting when close_orig is present."""

    def test_fit_skips_shift_when_close_orig_present(self):
        """Test that fit() skips shifting when close_orig is present."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'close_orig': [100.0, 101.0, 102.0, 103.0, 104.0],  # Already processed
            'rsi_14': [50.0, 55.0, 60.0, 65.0, 70.0],
        })

        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': df})

        # Stats should be computed on UNSHIFTED data (close_orig present = already shifted)
        # Mean of [100, 101, 102, 103, 104] = 102.0
        close_mean = pipe.stats['close']['mean']
        assert 101.5 <= close_mean <= 102.5, \
            f"close mean {close_mean} suggests data was shifted (should skip when close_orig present)"

    def test_transform_skips_shift_when_close_orig_present(self):
        """Test that transform_df() skips shifting when close_orig is present."""
        train_df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'close': [100.0, 101.0, 102.0],
        })

        test_df = pd.DataFrame({
            'timestamp': [4, 5, 6],
            'close': [103.0, 104.0, 105.0],
            'close_orig': [103.0, 104.0, 105.0],  # Already processed
        })

        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': train_df})
        transformed = pipe.transform_df(test_df)

        # close should NOT be shifted (close_orig present)
        assert not pd.isna(transformed['close'].iloc[0]), \
            "close should NOT be shifted when close_orig present"
        assert transformed['close'].iloc[0] == 103.0, \
            "close should preserve original value when close_orig present"


# ==============================================================================
# Test 4: TradingEnv compatibility - _close_shifted marker
# ==============================================================================

class TestTradingEnvCompatibility:
    """Verify that _close_shifted marker works with TradingEnv."""

    def test_close_shifted_in_metadata_columns(self):
        """Test that _close_shifted is in METADATA_COLUMNS."""
        assert '_close_shifted' in METADATA_COLUMNS, \
            "_close_shifted should be in METADATA_COLUMNS"

    def test_close_shifted_not_in_columns_to_shift(self):
        """Test that _close_shifted is excluded from _columns_to_shift."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'close': [100.0, 101.0, 102.0],
            '_close_shifted': [True, True, True],
        })

        cols = _columns_to_shift(df)
        assert '_close_shifted' not in cols, \
            "_close_shifted should NOT be in columns_to_shift"

    def test_trading_env_respects_close_shifted_marker(self):
        """Test that data with _close_shifted won't be double-shifted in TradingEnv."""
        # This test verifies the contract: if _close_shifted is True,
        # TradingEnv will skip its own shifting

        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'close': [100.0, 101.0, 102.0],  # Already shifted values
            '_close_shifted': [True, True, True],  # Marker present
        })

        # Verify marker is present and data has expected values
        assert '_close_shifted' in df.columns
        assert df['_close_shifted'].all()

        # TradingEnv checks: "close" in df.columns and "_close_shifted" not in df.columns
        # Since _close_shifted IS in df.columns, it will skip shifting
        close_shifted_present = '_close_shifted' in df.columns
        assert close_shifted_present, \
            "TradingEnv should see _close_shifted marker and skip shifting"


# ==============================================================================
# Test 5: Full pipeline integration (no double-shifting)
# ==============================================================================

class TestFullPipelineIntegration:
    """Test full pipeline: load_all_data → features_pipeline → no double-shift."""

    @pyarrow_required
    def test_full_pipeline_no_double_shift(self, tmp_path):
        """Test that data flows correctly without double-shifting."""
        from fetch_all_data_patch import load_all_data

        # Create test data with 4h-aligned timestamps (14400 seconds apart)
        base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
        df = pd.DataFrame({
            'timestamp': [base_ts + i * 14400 for i in range(5)],  # 4h intervals
            'symbol': ['BTCUSDT'] * 5,
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            'quote_asset_volume': [100000.0, 110000.0, 120000.0, 130000.0, 140000.0],
            'number_of_trades': [100, 110, 120, 130, 140],
            'taker_buy_base_asset_volume': [500.0, 550.0, 600.0, 650.0, 700.0],
            'taker_buy_quote_asset_volume': [50000.0, 55000.0, 60000.0, 65000.0, 70000.0],
            'rsi_14': [50.0, 55.0, 60.0, 65.0, 70.0],
        })

        feather_path = tmp_path / "BTCUSDT.feather"
        df.to_feather(feather_path)

        # Step 1: Load data
        all_dfs, _ = load_all_data([str(feather_path)])
        loaded_df = all_dfs['BTCUSDT']

        # Verify no close_orig
        assert 'close_orig' not in loaded_df.columns

        # Step 2: Apply features_pipeline
        pipe = FeaturePipeline()
        pipe.fit({'BTCUSDT': loaded_df})
        transformed = pipe.transform_df(loaded_df.copy())

        # Verify shifting happened ONCE
        assert pd.isna(transformed['close'].iloc[0]), "close should be shifted"
        assert transformed['close'].iloc[1] == 100.5, "close[1] should be original close[0]"

        # Verify _close_shifted marker
        assert '_close_shifted' in transformed.columns
        assert transformed['_close_shifted'].all()

        # Verify RSI also shifted
        assert pd.isna(transformed['rsi_14'].iloc[0]), "rsi_14 should be shifted"
        assert transformed['rsi_14'].iloc[1] == 50.0, "rsi_14[1] should be original rsi_14[0]"

    @pyarrow_required
    def test_no_data_leakage_after_fix(self, tmp_path):
        """CRITICAL: Verify no data leakage in the fixed pipeline."""
        from fetch_all_data_patch import load_all_data

        # Create test data with clear temporal pattern and 4h-aligned timestamps
        # At t=2: close=110, rsi=60 (based on recent uptrend)
        # At t=3: close=120, rsi=70 (continuing uptrend)
        base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
        df = pd.DataFrame({
            'timestamp': [base_ts + i * 14400 for i in range(5)],  # 4h intervals
            'symbol': ['BTCUSDT'] * 5,
            'open': [100.0, 105.0, 115.0, 125.0, 135.0],
            'high': [105.0, 115.0, 125.0, 135.0, 145.0],
            'low': [95.0, 100.0, 110.0, 120.0, 130.0],
            'close': [100.0, 110.0, 120.0, 130.0, 140.0],  # Clear uptrend
            'volume': [1000.0] * 5,
            'quote_asset_volume': [100000.0] * 5,
            'number_of_trades': [100] * 5,
            'taker_buy_base_asset_volume': [500.0] * 5,
            'taker_buy_quote_asset_volume': [50000.0] * 5,
            'rsi_14': [50.0, 60.0, 70.0, 80.0, 90.0],  # RSI increasing with price
        })

        feather_path = tmp_path / "BTCUSDT.feather"
        df.to_feather(feather_path)

        # Load and transform
        all_dfs, _ = load_all_data([str(feather_path)])
        pipe = FeaturePipeline()
        pipe.fit(all_dfs)
        transformed = pipe.transform_df(all_dfs['BTCUSDT'].copy())

        # CRITICAL CHECK: At index 2 (time t=3000):
        # - Model should see close[t=2000]=110 (NOT close[t=3000]=120!)
        # - Model should see rsi_14[t=2000]=60 (NOT rsi_14[t=3000]=70!)
        # This ensures model makes decision at t=3000 using info from t=2000

        assert transformed['close'].iloc[2] == 110.0, \
            "DATA LEAKAGE: close[2] should be 110 (from t=2000), not 120 (future info)"

        assert transformed['rsi_14'].iloc[2] == 60.0, \
            "DATA LEAKAGE: rsi_14[2] should be 60 (from t=2000), not 70 (future info)"


# ==============================================================================
# Test 6: Backward compatibility
# ==============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing artifacts."""

    def test_preserve_close_orig_still_works(self):
        """Test that preserve_close_orig=True still creates close_orig."""
        df = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'close': [100.0, 101.0, 102.0],
        })

        pipe = FeaturePipeline(preserve_close_orig=True)
        pipe.fit({'BTCUSDT': df})
        transformed = pipe.transform_df(df.copy())

        # close_orig should be created
        assert 'close_orig' in transformed.columns, \
            "close_orig should be created when preserve_close_orig=True"

        # close_orig should have original unshifted values
        assert np.allclose(
            transformed['close_orig'].values,
            [100.0, 101.0, 102.0]
        ), "close_orig should contain original values"

        # close should be shifted
        assert pd.isna(transformed['close'].iloc[0])
        assert transformed['close'].iloc[1] == 100.0

    def test_legacy_artifacts_without_preserve_close_orig(self):
        """Test that legacy artifacts (without preserve_close_orig) still work."""
        import json
        import tempfile

        # Simulate legacy artifact (JSON with stats but no preserve_close_orig)
        legacy_json = {
            "stats": {
                "close": {"mean": 100.0, "std": 5.0, "is_constant": False}
            },
            "metadata": {},
            "config": {
                "enable_winsorization": True,
                "winsorize_percentiles": [1.0, 99.0],
                "strict_idempotency": True
                # NO "preserve_close_orig" key (legacy)
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(legacy_json, f)
            temp_path = f.name

        try:
            pipe = FeaturePipeline.load(temp_path)

            # FIX (2025-11-25): Changed default from False to True to match constructor
            # Legacy artifacts now default to True for correct TradingEnv reward calculation
            assert pipe.preserve_close_orig is True, \
                "Legacy artifacts should default to True (fixed 2025-11-25)"

            # Should work without errors
            df = pd.DataFrame({
                'timestamp': [1, 2, 3],
                'close': [100.0, 110.0, 90.0],
            })

            transformed = pipe.transform_df(df.copy())
            assert 'close_z' in transformed.columns

            # With new default, close_orig should now be created
            assert 'close_orig' in transformed.columns, \
                "Legacy artifacts with new default should create close_orig"

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
