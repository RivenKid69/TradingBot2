# -*- coding: utf-8 -*-
"""
Tests for corporate actions service layer.

Tests cover:
- CorporateActionsService initialization
- Gap features computation
- Module-level convenience functions
"""

import pytest
import tempfile
import os
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class TestCorporateActionsServiceInit:
    """Tests for service initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()
        assert service.config is not None
        assert service._adapter is None  # Lazy loaded

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        from services.corporate_actions import (
            CorporateActionsService,
            CorporateActionsConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CorporateActionsConfig(
                cache_dir=Path(tmpdir),
                cache_ttl_hours=48,
            )
            service = CorporateActionsService(config=config)

            assert service.config.cache_ttl_hours == 48
            assert service.config.cache_dir == Path(tmpdir)


class TestCorporateActionsServiceGapFeatures:
    """Tests for gap features computation via service."""

    def test_compute_gap_features_basic(self):
        """Test basic gap feature computation."""
        from services.corporate_actions import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0, 98.0],
                "close": [100.0, 106.0, 99.0],
                "high": [101.0, 107.0, 100.0],
                "low": [99.0, 104.0, 97.0],
            }
        )

        result = compute_gap_features(df)

        assert "gap_pct" in result.columns
        assert "gap_direction" in result.columns
        assert "gap_magnitude" in result.columns

    def test_compute_gap_features_preserves_columns(self):
        """Test that original columns are preserved."""
        from services.corporate_actions import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "close": [100.0, 106.0],
                "high": [101.0, 107.0],
                "low": [99.0, 104.0],
                "volume": [1000, 1100],
            }
        )

        result = compute_gap_features(df)

        assert "volume" in result.columns
        assert "close" in result.columns


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_service_singleton(self):
        """Test get_service returns singleton."""
        from services.corporate_actions import get_service

        service1 = get_service()
        service2 = get_service()

        assert service1 is service2

    def test_compute_gap_features_function(self):
        """Test module-level compute_gap_features."""
        from services.corporate_actions import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "close": [100.0, 106.0],
                "high": [101.0, 107.0],
                "low": [99.0, 104.0],
            }
        )

        result = compute_gap_features(df)

        assert "gap_pct" in result.columns


class TestCacheManagement:
    """Tests for cache management."""

    def test_clear_cache(self):
        """Test cache clearing."""
        from services.corporate_actions import CorporateActionsService

        service = CorporateActionsService()

        # Clear should not raise
        service.clear_cache()
        service.clear_cache("AAPL")


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_service_works_without_network(self):
        """Test that service can be instantiated without network."""
        from services.corporate_actions import CorporateActionsService

        # Should not raise - adapters are lazy loaded
        service = CorporateActionsService()
        assert service is not None

    def test_gap_features_only_require_ohlc(self):
        """Test gap features work with minimal OHLC data."""
        from services.corporate_actions import compute_gap_features

        df = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [101.0, 106.0],
                "low": [99.0, 104.0],
                "close": [100.0, 105.0],
            }
        )

        # Should not raise
        result = compute_gap_features(df)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
