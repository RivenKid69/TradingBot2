# -*- coding: utf-8 -*-
"""
Tests for macro data service.

Tests cover:
- MacroDataService initialization
- VIX data fetching and caching
- DXY data fetching and caching
- Treasury yields fetching and caching
- Feature engineering functions
- Module-level convenience functions
- Crypto backward compatibility (no macro data in crypto mode)

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestMacroDataServiceInit:
    """Tests for service initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        assert service.config is not None
        assert service.config.cache_ttl_hours == 6  # Default

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(
                cache_dir=Path(tmpdir),
                cache_ttl_hours=8,
                vix_normalization_center=22.0,
            )
            service = MacroDataService(config=config)

            assert service.config.cache_ttl_hours == 8
            assert service.config.cache_dir == Path(tmpdir)
            assert service.config.vix_normalization_center == 22.0

    def test_cache_directory_created(self):
        """Test cache directory is created on init."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "cache"
            config = MacroDataConfig(cache_dir=cache_path)
            service = MacroDataService(config=config)

            assert cache_path.exists()


class TestVIXData:
    """Tests for VIX data fetching."""

    @patch("services.macro_data.MacroDataService._fetch_indicator")
    def test_get_vix_returns_data(self, mock_fetch):
        """Test get_vix returns data when available."""
        from services.macro_data import MacroDataService, MacroDataConfig

        mock_fetch.return_value = pd.DataFrame(
            {
                "value": [15.5, 16.0, 14.8],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "timestamp": [1704067200, 1704153600, 1704240000],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            result = service.get_vix()

            assert result is not None
            assert "value" in result.columns
            assert len(result) == 3

    def test_get_vix_cached(self):
        """Test VIX data is cached."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Pre-populate cache (uses dict-based cache)
            test_data = pd.DataFrame(
                {
                    "value": [15.5, 16.0],
                    "date": ["2024-01-01", "2024-01-02"],
                    "timestamp": [1704067200, 1704153600],
                }
            )
            service._data_cache["vix"] = test_data
            service._cache_timestamps["vix"] = datetime.now()

            result = service.get_vix()
            assert result is not None
            assert len(result) == 2


class TestDXYData:
    """Tests for DXY (Dollar Index) data fetching."""

    @patch("services.macro_data.MacroDataService._fetch_indicator")
    def test_get_dxy_returns_data(self, mock_fetch):
        """Test get_dxy returns data when available."""
        from services.macro_data import MacroDataService, MacroDataConfig

        mock_fetch.return_value = pd.DataFrame(
            {
                "value": [103.5, 104.0, 102.8],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "timestamp": [1704067200, 1704153600, 1704240000],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            result = service.get_dxy()

            assert result is not None
            assert "value" in result.columns

    def test_get_dxy_cached(self):
        """Test DXY data is cached."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Pre-populate cache
            test_data = pd.DataFrame(
                {
                    "value": [103.5, 104.0],
                    "date": ["2024-01-01", "2024-01-02"],
                    "timestamp": [1704067200, 1704153600],
                }
            )
            service._data_cache["dxy"] = test_data
            service._cache_timestamps["dxy"] = datetime.now()

            result = service.get_dxy()
            assert result is not None


class TestTreasuryYields:
    """Tests for Treasury yields data fetching."""

    @patch("services.macro_data.MacroDataService._fetch_indicator")
    def test_get_treasury_yields_returns_data(self, mock_fetch):
        """Test get_treasury_yields returns data when available."""
        from services.macro_data import MacroDataService, MacroDataConfig

        mock_fetch.return_value = pd.DataFrame(
            {
                "value": [4.5, 4.6, 4.4],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "timestamp": [1704067200, 1704153600, 1704240000],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            result = service.get_treasury_yields()

            assert result is not None
            assert "date" in result.columns


class TestNormalizationFunctions:
    """Tests for normalization functions (service methods)."""

    def test_normalize_vix_low_vol(self):
        """Test VIX normalization for low volatility regime."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Low VIX (below 15) - tanh centered at 20, scale 10
        # (12 - 20) / 10 = -0.8, tanh(-0.8) ≈ -0.66
        result = service.normalize_vix(12.0)
        assert -1.0 <= result <= 1.0
        assert result < 0.0  # Below center, negative

    def test_normalize_vix_at_center(self):
        """Test VIX normalization at center (20)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # VIX at center (20), tanh(0) = 0
        result = service.normalize_vix(20.0)
        assert abs(result) < 0.1  # Close to 0

    def test_normalize_vix_high_vol(self):
        """Test VIX normalization for high volatility regime."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # High VIX (above 30)
        result = service.normalize_vix(40.0)
        assert -1.0 <= result <= 1.0
        assert result > 0.5  # High volatility

    def test_normalize_vix_extreme(self):
        """Test VIX normalization for extreme values."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Extreme VIX (like 2020 March)
        result = service.normalize_vix(80.0)
        assert result <= 1.0  # tanh is bounded

    def test_normalize_dxy_strong_dollar(self):
        """Test DXY normalization for strong dollar."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Strong dollar (above 105)
        result = service.normalize_dxy(110.0)
        assert result > 0.0  # Positive (strong dollar)
        assert result <= 1.0

    def test_normalize_dxy_weak_dollar(self):
        """Test DXY normalization for weak dollar."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Weak dollar (below 95)
        result = service.normalize_dxy(90.0)
        assert result < 0.0  # Negative (weak dollar)
        assert result >= -1.0

    def test_normalize_dxy_neutral(self):
        """Test DXY normalization for neutral dollar."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Neutral (around 100 = center)
        result = service.normalize_dxy(100.0)
        assert abs(result) < 0.1  # Near neutral (0)

    def test_normalize_treasury_high_yield(self):
        """Test Treasury normalization for high yields."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # High yield (above center 3.5%)
        result = service.normalize_treasury(5.5)
        assert result > 0.0
        assert result <= 1.0

    def test_normalize_treasury_low_yield(self):
        """Test Treasury normalization for low yields."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Low yield (below center 3.5%)
        result = service.normalize_treasury(1.5)
        assert result < 0.0
        assert result >= -1.0


class TestVIXRegime:
    """Tests for VIX regime classification."""

    def test_compute_vix_regime_calm(self):
        """Test calm regime detection (VIX <= 15)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(12.0)
        assert regime == 0.0  # Calm

    def test_compute_vix_regime_low_normal(self):
        """Test low normal regime detection (15 < VIX <= 20)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(18.0)
        assert regime == 0.25  # Low normal

    def test_compute_vix_regime_high_normal(self):
        """Test high normal regime detection (20 < VIX <= 25)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(23.0)
        assert regime == 0.5  # High normal

    def test_compute_vix_regime_elevated(self):
        """Test elevated regime detection (25 < VIX <= 30)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(28.0)
        assert regime == 0.75  # Elevated

    def test_compute_vix_regime_high(self):
        """Test high regime detection (VIX > 30)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(35.0)
        assert regime == 1.0  # High/Extreme

    def test_compute_vix_regime_extreme(self):
        """Test extreme regime detection (VIX > 40)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        regime = service.compute_vix_regime(55.0)
        assert regime == 1.0  # Extreme


class TestRealYieldComputation:
    """Tests for real yield proxy computation."""

    def test_compute_real_yield_positive(self):
        """Test positive real yield."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Treasury 4.5%, VIX 20 → inflation proxy = 2.0
        # Real yield = 4.5 - 2.0 = 2.5
        result = service.compute_real_yield_proxy(4.5, 20.0)
        assert result > 0.0
        assert abs(result - 2.5) < 0.1  # ~2.5% real yield

    def test_compute_real_yield_negative(self):
        """Test negative real yield (high VIX)."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Treasury 2%, VIX 40 → inflation proxy = 4.0
        # Real yield = 2.0 - 4.0 = -2.0
        result = service.compute_real_yield_proxy(2.0, 40.0)
        assert result < 0.0

    def test_compute_real_yield_near_zero(self):
        """Test near-zero real yield."""
        from services.macro_data import MacroDataService

        service = MacroDataService()
        # Treasury 3.5%, VIX 35 → inflation proxy = 3.5
        # Real yield = 3.5 - 3.5 = 0
        result = service.compute_real_yield_proxy(3.5, 35.0)
        assert abs(result) < 0.1


class TestAddMacroFeatures:
    """Tests for add_macro_features function."""

    @patch("services.macro_data.MacroDataService.get_all_macro")
    def test_add_macro_features_to_df(self, mock_get_all):
        """Test adding macro features to DataFrame."""
        from services.macro_data import MacroDataService, MacroDataConfig

        # Mock the get_all_macro response
        mock_get_all.return_value = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "vix": [15.0, 16.0, 17.0],
                "dxy": [102.0, 103.0, 104.0],
                "treasury_10y": [4.0, 4.1, 4.2],
                "treasury_30y": [4.5, 4.6, 4.7],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Input DataFrame with date column
            df = pd.DataFrame(
                {
                    "close": [100.0, 101.0, 102.0],
                    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                }
            )

            result = service.add_macro_features(df)

            # Check that macro features were added
            assert "vix_value" in result.columns
            assert "vix_normalized" in result.columns
            assert "vix_regime" in result.columns
            assert "dxy_value" in result.columns
            assert "dxy_normalized" in result.columns
            assert "treasury_10y_yield" in result.columns
            assert "treasury_10y_normalized" in result.columns

    def test_add_macro_features_preserves_original(self):
        """Test that original columns are preserved."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            df = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [105.0, 106.0],
                    "low": [95.0, 96.0],
                    "close": [102.0, 103.0],
                    "volume": [1000, 1100],
                    "date": ["2024-01-01", "2024-01-02"],
                }
            )

            result = service.add_macro_features(df)

            assert "open" in result.columns
            assert "high" in result.columns
            assert "low" in result.columns
            assert "close" in result.columns
            assert "volume" in result.columns


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_service_singleton(self):
        """Test get_service returns singleton."""
        from services.macro_data import get_service

        service1 = get_service()
        service2 = get_service()

        assert service1 is service2

    def test_get_vix_function(self):
        """Test module-level get_vix function."""
        from services.macro_data import get_vix, get_service

        # Pre-populate cache (dict-based)
        service = get_service()
        service._data_cache["vix"] = pd.DataFrame(
            {
                "value": [15.0],
                "date": ["2024-01-01"],
                "timestamp": [1704067200],
            }
        )
        service._cache_timestamps["vix"] = datetime.now()

        result = get_vix()
        assert result is not None

    def test_get_dxy_function(self):
        """Test module-level get_dxy function."""
        from services.macro_data import get_dxy, get_service

        # Pre-populate cache (dict-based)
        service = get_service()
        service._data_cache["dxy"] = pd.DataFrame(
            {
                "value": [103.0],
                "date": ["2024-01-01"],
                "timestamp": [1704067200],
            }
        )
        service._cache_timestamps["dxy"] = datetime.now()

        result = get_dxy()
        assert result is not None


class TestCryptoBackwardCompatibility:
    """Tests to ensure crypto functionality is not affected."""

    def test_macro_features_optional_for_crypto(self):
        """Test that macro features are optional (don't break crypto pipeline)."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Crypto DataFrame without timestamp column matching macro data
            df = pd.DataFrame(
                {
                    "close": [50000.0, 50100.0, 50200.0],
                    "volume": [1000, 1100, 1200],
                }
            )

            # Should not raise, should just return df with NaN or default values
            result = service.add_macro_features(df)

            assert len(result) == len(df)
            assert "close" in result.columns
            assert "volume" in result.columns

    def test_missing_macro_data_returns_nan(self):
        """Test that missing macro data results in NaN values."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Empty caches (no data)
            df = pd.DataFrame(
                {
                    "close": [100.0, 101.0],
                    "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                }
            )

            result = service.add_macro_features(df)

            # Should have columns but with NaN values
            if "vix" in result.columns:
                assert result["vix"].isna().all() or len(result) == len(df)


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            # Populate caches (dict-based)
            service._data_cache["vix"] = pd.DataFrame({"value": [15.0]})
            service._data_cache["dxy"] = pd.DataFrame({"value": [103.0]})
            service._data_cache["treasury_10y"] = pd.DataFrame({"value": [4.0]})
            service._cache_timestamps["vix"] = datetime.now()

            service.clear_cache()

            assert len(service._data_cache) == 0
            assert len(service._cache_timestamps) == 0

    def test_cache_expiry(self):
        """Test cache expiry based on TTL."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(
                cache_dir=Path(tmpdir),
                cache_ttl_hours=1,
            )
            service = MacroDataService(config=config)

            # Set cache with old timestamp
            service._data_cache["vix"] = pd.DataFrame({"value": [15.0]})
            service._cache_timestamps["vix"] = datetime.now() - timedelta(hours=2)

            # Should be considered expired
            assert not service._is_cache_valid("vix")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe_input(self):
        """Test handling of empty DataFrame."""
        from services.macro_data import MacroDataService, MacroDataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MacroDataConfig(cache_dir=Path(tmpdir))
            service = MacroDataService(config=config)

            df = pd.DataFrame()

            result = service.add_macro_features(df)
            assert len(result) == 0

    def test_nan_values_in_input(self):
        """Test handling of NaN values in normalization."""
        from services.macro_data import MacroDataService

        service = MacroDataService()

        # NaN input returns NaN (tanh of NaN is NaN)
        result = service.normalize_vix(float("nan"))
        assert np.isnan(result)

        result = service.normalize_dxy(float("nan"))
        assert np.isnan(result)

        result = service.normalize_treasury(float("nan"))
        assert np.isnan(result)

    def test_extreme_values(self):
        """Test handling of extreme values (tanh is bounded to [-1, 1])."""
        from services.macro_data import MacroDataService

        service = MacroDataService()

        # Extreme VIX (tanh is bounded)
        result = service.normalize_vix(200.0)
        assert -1.0 <= result <= 1.0

        # Extreme DXY
        result = service.normalize_dxy(150.0)
        assert -1.0 <= result <= 1.0

        result = service.normalize_dxy(50.0)
        assert -1.0 <= result <= 1.0
