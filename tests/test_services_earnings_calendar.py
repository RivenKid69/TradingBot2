# -*- coding: utf-8 -*-
"""
Tests for earnings calendar service.

Tests cover:
- EarningsCalendarService initialization
- Earnings date fetching and caching
- Bulk preloading operations
- Feature engineering functions
- Vectorized DataFrame operations
- Module-level convenience functions
- Crypto backward compatibility

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# For test setup - datetime is imported above


class TestEarningsCalendarServiceInit:
    """Tests for service initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from services.earnings_calendar import EarningsCalendarService

        service = EarningsCalendarService()
        assert service.config is not None
        assert service.config.cache_ttl_hours == 24  # Default

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(
                cache_dir=Path(tmpdir),
                cache_ttl_hours=48,
                blackout_days=21,
                drift_days=5,
            )
            service = EarningsCalendarService(config=config)

            assert service.config.cache_ttl_hours == 48
            assert service.config.blackout_days == 21
            assert service.config.drift_days == 5

    def test_cache_directory_created(self):
        """Test cache directory is created on init."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "cache"
            config = EarningsCalendarConfig(cache_dir=cache_path)
            service = EarningsCalendarService(config=config)

            assert cache_path.exists()


class TestEarningsFetching:
    """Tests for earnings data fetching."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_earnings_returns_data(self, mock_fetch):
        """Test get_earnings returns data when available."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "eps_actual": 2.18, "eps_estimate": 2.10},
            {"report_date": "2024-04-25", "eps_actual": 2.05, "eps_estimate": 2.00},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.get_earnings("AAPL")

            assert len(result) == 2
            assert result[0]["report_date"] == "2024-01-25"

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_earnings_uses_cache(self, mock_fetch):
        """Test get_earnings uses cache when valid."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            # Pre-populate cache
            cached_data = [{"report_date": "2024-01-25"}]
            service._earnings_cache["AAPL"] = cached_data
            service._cache_timestamps["AAPL"] = datetime.now()

            result = service.get_earnings("AAPL")

            # Should not call fetch (cache hit)
            assert not mock_fetch.called
            assert result == cached_data


class TestEarningsDates:
    """Tests for earnings dates retrieval."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_earnings_dates(self, mock_fetch):
        """Test get_earnings_dates returns list of dates."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25"},
            {"report_date": "2024-04-25"},
            {"report_date": "2024-07-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            dates = service.get_earnings_dates("AAPL")

            assert len(dates) == 3
            assert "2024-01-25" in dates
            assert dates == sorted(dates)  # Should be sorted

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_earnings_dates_filter_by_year(self, mock_fetch):
        """Test filtering earnings dates by year."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2023-10-25"},
            {"report_date": "2024-01-25"},
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            dates = service.get_earnings_dates("AAPL", year=2024)

            assert len(dates) == 2
            assert all(d.startswith("2024") for d in dates)


class TestNextLastEarnings:
    """Tests for next/last earnings retrieval."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_next_earnings(self, mock_fetch):
        """Test get_next_earnings returns next earnings event."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25"},
            {"report_date": "2024-04-25"},
            {"report_date": "2024-07-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.get_next_earnings("AAPL", "2024-03-01")

            assert result is not None
            assert result["report_date"] == "2024-04-25"

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_last_earnings(self, mock_fetch):
        """Test get_last_earnings returns most recent past earnings."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "surprise_pct": 5.0},
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.get_last_earnings("AAPL", "2024-03-01")

            assert result is not None
            assert result["report_date"] == "2024-01-25"
            assert result["surprise_pct"] == 5.0

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_next_earnings_none(self, mock_fetch):
        """Test get_next_earnings returns None when no future earnings."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.get_next_earnings("AAPL", "2024-12-01")

            assert result is None


class TestBulkOperations:
    """Tests for bulk preloading operations."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_preload_earnings_multiple_symbols(self, mock_fetch):
        """Test preloading earnings for multiple symbols."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.side_effect = [
            [{"report_date": "2024-01-25"}],  # AAPL
            [{"report_date": "2024-01-30"}, {"report_date": "2024-04-30"}],  # MSFT
            [],  # GOOGL (no earnings)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.preload_earnings(["AAPL", "MSFT", "GOOGL"])

            assert result["AAPL"] == 1
            assert result["MSFT"] == 2
            assert result["GOOGL"] == 0

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_get_upcoming_earnings(self, mock_fetch):
        """Test getting upcoming earnings for multiple symbols."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")

        mock_fetch.side_effect = [
            [{"report_date": future}],  # AAPL has upcoming
            [],  # MSFT no upcoming
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            result = service.get_upcoming_earnings(["AAPL", "MSFT"], days_ahead=30)

            assert "AAPL" in result
            assert "MSFT" not in result


class TestFeatureComputation:
    """Tests for earnings feature computation."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_compute_earnings_features_basic(self, mock_fetch):
        """Test basic earnings feature computation."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "surprise_pct": 5.0},
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            features = service.compute_earnings_features("AAPL", "2024-03-01")

            assert "days_until_earnings" in features
            assert "days_since_earnings" in features
            assert "last_earnings_surprise" in features
            assert "in_earnings_blackout" in features
            assert "post_earnings" in features

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_days_until_earnings(self, mock_fetch):
        """Test days_until_earnings calculation."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            features = service.compute_earnings_features("AAPL", "2024-04-15")

            # 10 days until earnings
            assert features["days_until_earnings"] == 10.0

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_in_blackout_period(self, mock_fetch):
        """Test blackout period detection."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(
                cache_dir=Path(tmpdir),
                blackout_days=14,
            )
            service = EarningsCalendarService(config=config)

            # Within blackout (14 days before)
            features = service.compute_earnings_features("AAPL", "2024-04-15")
            assert features["in_earnings_blackout"] == 1

            # Outside blackout (20 days before)
            features = service.compute_earnings_features("AAPL", "2024-04-05")
            assert features["in_earnings_blackout"] == 0

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_post_earnings_period(self, mock_fetch):
        """Test post-earnings drift period detection."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "surprise_pct": 5.0},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(
                cache_dir=Path(tmpdir),
                drift_days=3,
            )
            service = EarningsCalendarService(config=config)

            # Within drift period (2 days after)
            features = service.compute_earnings_features("AAPL", "2024-01-27")
            assert features["post_earnings"] == 1

            # Outside drift period (5 days after)
            features = service.compute_earnings_features("AAPL", "2024-01-30")
            assert features["post_earnings"] == 0


class TestDataFrameOperations:
    """Tests for DataFrame operations."""

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_add_earnings_to_df(self, mock_fetch):
        """Test adding earnings features to DataFrame."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "surprise_pct": 5.0},
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame({
                "close": [100.0, 101.0, 102.0],
                "date": ["2024-01-20", "2024-01-25", "2024-01-30"],
            })

            result = service.add_earnings_to_df(df, "AAPL")

            assert "days_until_earnings" in result.columns
            assert "days_since_earnings" in result.columns
            assert "last_earnings_surprise" in result.columns
            assert "in_earnings_blackout" in result.columns
            assert "post_earnings" in result.columns

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_add_earnings_to_df_vectorized(self, mock_fetch):
        """Test vectorized earnings features addition."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        mock_fetch.return_value = [
            {"report_date": "2024-01-25", "surprise_pct": 5.0},
            {"report_date": "2024-04-25"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame({
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "date": pd.date_range("2024-01-20", periods=5).astype(str),
            })

            result = service.add_earnings_to_df_vectorized(df, "AAPL")

            assert "days_until_earnings" in result.columns
            assert len(result) == len(df)

    def test_add_earnings_preserves_columns(self):
        """Test that original columns are preserved."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame({
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [95.0, 96.0],
                "close": [102.0, 103.0],
                "volume": [1000, 1100],
                "date": ["2024-01-20", "2024-01-21"],
            })

            result = service.add_earnings_to_df(df, "AAPL")

            assert "open" in result.columns
            assert "high" in result.columns
            assert "low" in result.columns
            assert "close" in result.columns
            assert "volume" in result.columns


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_service_singleton(self):
        """Test get_service returns singleton."""
        from services.earnings_calendar import get_service

        service1 = get_service()
        service2 = get_service()

        assert service1 is service2

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_preload_earnings_function(self, mock_fetch):
        """Test module-level preload_earnings function."""
        from services.earnings_calendar import preload_earnings, get_service

        mock_fetch.return_value = [{"report_date": "2024-01-25"}]

        result = preload_earnings(["AAPL"])

        assert "AAPL" in result

    def test_get_earnings_dates_function(self):
        """Test module-level get_earnings_dates function."""
        from services.earnings_calendar import get_earnings_dates, get_service

        # Pre-populate cache directly
        service = get_service()
        service._earnings_cache["AAPL"] = [
            {"report_date": "2024-01-25"},
            {"report_date": "2024-04-25"},
        ]
        service._cache_timestamps["AAPL"] = datetime.now()

        dates = get_earnings_dates("AAPL")

        assert len(dates) == 2

    @patch("services.earnings_calendar.EarningsCalendarService._fetch_earnings")
    def test_add_earnings_to_df_function(self, mock_fetch):
        """Test module-level add_earnings_to_df function."""
        from services.earnings_calendar import add_earnings_to_df

        mock_fetch.return_value = []

        df = pd.DataFrame({
            "close": [100.0],
            "date": ["2024-01-20"],
        })

        result = add_earnings_to_df(df, "AAPL")

        assert "days_until_earnings" in result.columns


class TestCryptoBackwardCompatibility:
    """Tests to ensure crypto functionality is not affected."""

    def test_service_does_not_affect_crypto(self):
        """Test earnings calendar doesn't affect crypto pipeline."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            # Crypto DataFrame (no earnings concept)
            df = pd.DataFrame({
                "close": [50000.0, 50100.0],
                "volume": [1000, 1100],
            })

            # Should not raise, should just return df unchanged or with defaults
            result = service.add_earnings_to_df(df, "BTCUSD")

            assert len(result) == len(df)
            # Original columns preserved
            assert "close" in result.columns
            assert "volume" in result.columns

    def test_missing_earnings_returns_defaults(self):
        """Test missing earnings returns default values."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame({
                "close": [100.0],
                "date": ["2024-01-20"],
            })

            result = service.add_earnings_to_df(df, "NONEXISTENT")

            # Should have default values
            assert result["days_until_earnings"].iloc[0] == 90.0
            assert result["in_earnings_blackout"].iloc[0] == 0


class TestCacheManagement:
    """Tests for cache management."""

    def test_save_to_file_cache(self):
        """Test saving earnings to file cache."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            earnings = [{"report_date": "2024-01-25", "surprise_pct": 5.0}]
            service._save_to_cache("AAPL", earnings)

            # Check file exists
            cache_file = Path(tmpdir) / "AAPL_earnings.json"
            assert cache_file.exists()

    def test_load_from_file_cache(self):
        """Test loading earnings from file cache."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file
            cache_file = Path(tmpdir) / "AAPL_earnings.json"
            with open(cache_file, "w") as f:
                json.dump([{"report_date": "2024-01-25"}], f)

            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            loaded = service._load_from_cache("AAPL")

            assert loaded is not None
            assert len(loaded) == 1
            assert loaded[0]["report_date"] == "2024-01-25"

    def test_clear_cache(self):
        """Test clearing all caches."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            # Populate caches
            service._earnings_cache["AAPL"] = [{"report_date": "2024-01-25"}]
            service._save_to_cache("AAPL", [{"report_date": "2024-01-25"}])

            # Clear all
            service.clear_cache()

            assert len(service._earnings_cache) == 0
            assert not (Path(tmpdir) / "AAPL_earnings.json").exists()

    def test_clear_cache_single_symbol(self):
        """Test clearing cache for single symbol."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            # Populate caches for multiple symbols
            service._earnings_cache["AAPL"] = [{"report_date": "2024-01-25"}]
            service._earnings_cache["MSFT"] = [{"report_date": "2024-01-30"}]
            service._save_to_cache("AAPL", [{"report_date": "2024-01-25"}])
            service._save_to_cache("MSFT", [{"report_date": "2024-01-30"}])

            # Clear only AAPL
            service.clear_cache("AAPL")

            assert "AAPL" not in service._earnings_cache
            assert "MSFT" in service._earnings_cache


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame()
            result = service.add_earnings_to_df(df, "AAPL")

            assert len(result) == 0

    def test_no_date_column(self):
        """Test handling of DataFrame without date column."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            df = pd.DataFrame({
                "close": [100.0, 101.0],
                "volume": [1000, 1100],
            })

            # Should not raise, should return df unchanged
            result = service.add_earnings_to_df(df, "AAPL")

            assert len(result) == len(df)

    def test_days_capped_at_90(self):
        """Test days_until_earnings is capped at 90."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            # Pre-populate with far future earnings
            service._earnings_cache["AAPL"] = [
                {"report_date": "2025-12-25"}  # Far in future
            ]
            service._cache_timestamps["AAPL"] = datetime.now()

            features = service.compute_earnings_features("AAPL", "2024-01-01")

            assert features["days_until_earnings"] == 90.0  # Capped

    def test_surprise_percentage_extraction(self):
        """Test extraction of earnings surprise percentage."""
        from services.earnings_calendar import (
            EarningsCalendarService,
            EarningsCalendarConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = EarningsCalendarConfig(cache_dir=Path(tmpdir))
            service = EarningsCalendarService(config=config)

            service._earnings_cache["AAPL"] = [
                {"report_date": "2024-01-25", "surprise_pct": -3.5}  # Beat by 3.5%
            ]
            service._cache_timestamps["AAPL"] = datetime.now()

            features = service.compute_earnings_features("AAPL", "2024-02-01")

            assert features["last_earnings_surprise"] == -3.5
