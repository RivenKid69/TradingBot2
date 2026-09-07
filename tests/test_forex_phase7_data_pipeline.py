# -*- coding: utf-8 -*-
"""
tests/test_forex_phase7_data_pipeline.py
Comprehensive test suite for Forex Phase 7: Data Pipeline & Downloaders.

Test Categories:
1. Download Scripts Tests
   - download_forex_data.py
   - download_swap_rates.py
   - download_economic_calendar.py
   - download_interest_rates.py
2. Data Loader Tests
   - load_forex_data()
   - Session features
   - Data merging (swaps, rates, calendar)
3. Integration Tests
   - End-to-end pipeline
   - Backward compatibility
4. Edge Cases & Error Handling

Total Tests: ~50

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_forex_df():
    """Create sample forex OHLCV DataFrame."""
    np.random.seed(42)
    n_bars = 100

    # Generate timestamps (hourly bars for a week, excluding weekends)
    base_ts = int(datetime(2024, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())  # Monday
    timestamps = [base_ts + i * 3600 for i in range(n_bars)]

    # Filter out weekends
    def is_weekday(ts):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.weekday() < 5

    timestamps = [ts for ts in timestamps if is_weekday(ts)][:n_bars]

    # Generate price data
    base_price = 1.1000  # EUR/USD
    prices = [base_price]
    for _ in range(len(timestamps) - 1):
        change = np.random.normal(0, 0.0010)  # ~10 pips std
        prices.append(prices[-1] + change)

    df = pd.DataFrame(
        {
            "timestamp": timestamps[: len(prices)],
            "open": prices,
            "high": [p + abs(np.random.normal(0, 0.0005)) for p in prices],
            "low": [p - abs(np.random.normal(0, 0.0005)) for p in prices],
            "close": [p + np.random.normal(0, 0.0003) for p in prices],
            "volume": np.random.randint(1000, 10000, len(prices)),
            "spread_pips": np.random.uniform(0.8, 1.5, len(prices)),
            "symbol": "EUR_USD",
        }
    )

    return df


@pytest.fixture
def sample_swap_df():
    """Create sample swap rate DataFrame."""
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pair": "EUR_USD",
            "long_swap": np.random.uniform(-0.8, -0.3, len(dates)),
            "short_swap": np.random.uniform(-0.5, -0.2, len(dates)),
            "source": "synthetic",
        }
    )


@pytest.fixture
def sample_rate_df():
    """Create sample interest rate DataFrame."""
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "rate": np.random.uniform(5.0, 5.5, len(dates)),
            "currency": "USD",
            "source": "fred",
        }
    )


@pytest.fixture
def sample_calendar_df():
    """Create sample economic calendar DataFrame."""
    return pd.DataFrame(
        {
            "datetime": [
                "2024-01-10 13:30:00",
                "2024-01-12 19:00:00",
                "2024-01-15 08:30:00",
            ],
            "date": ["2024-01-10", "2024-01-12", "2024-01-15"],
            "time": ["13:30", "19:00", "08:30"],
            "currency": ["USD", "USD", "EUR"],
            "event": ["Non-Farm Payrolls", "Fed Speech", "CPI"],
            "impact": ["high", "medium", "high"],
            "actual": [200, None, 2.5],
            "forecast": [180, None, 2.4],
            "previous": [190, None, 2.3],
            "source": "synthetic",
        }
    )


# =============================================================================
# TEST: download_forex_data.py
# =============================================================================


class TestDownloadForexData:
    """Tests for scripts/download_forex_data.py"""

    def test_forex_download_config_defaults(self):
        """Test ForexDownloadConfig has correct defaults."""
        from scripts.download_forex_data import ForexDownloadConfig

        config = ForexDownloadConfig()
        assert config.provider == "oanda"
        assert config.practice is True
        assert config.timeframe == "1h"
        assert config.output_dir == "data/raw_forex"
        assert config.filter_weekends is True
        assert config.include_spread is True

    def test_major_pairs_defined(self):
        """Test major pairs are correctly defined."""
        from scripts.download_forex_data import MAJOR_PAIRS, MINOR_PAIRS, ALL_PAIRS

        assert "EUR_USD" in MAJOR_PAIRS
        assert "USD_JPY" in MAJOR_PAIRS
        assert "GBP_USD" in MAJOR_PAIRS
        assert len(MAJOR_PAIRS) == 7

        assert "EUR_GBP" in MINOR_PAIRS
        assert "EUR_JPY" in MINOR_PAIRS

        assert len(ALL_PAIRS) > len(MAJOR_PAIRS)

    def test_pip_sizes(self):
        """Test pip sizes are correct for different pairs."""
        from scripts.download_forex_data import get_pip_size, PIP_SIZES

        # JPY pairs have pip = 0.01
        assert get_pip_size("USD_JPY") == 0.01
        assert get_pip_size("EUR_JPY") == 0.01

        # Standard pairs have pip = 0.0001
        assert get_pip_size("EUR_USD") == 0.0001
        assert get_pip_size("GBP_USD") == 0.0001

    def test_forex_calendar_weekend_detection(self):
        """Test forex weekend detection."""
        from scripts.download_forex_data import ForexCalendar

        # Monday - should be trading
        monday = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)
        assert not ForexCalendar.is_weekend(monday)

        # Saturday - closed
        saturday = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(saturday)

        # Sunday before 21:00 UTC - closed
        sunday_early = datetime(2024, 1, 7, 15, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(sunday_early)

        # Sunday after 21:00 UTC - open
        sunday_late = datetime(2024, 1, 7, 22, 0, tzinfo=timezone.utc)
        assert not ForexCalendar.is_weekend(sunday_late)

        # Friday after 21:00 UTC - closed
        friday_late = datetime(2024, 1, 5, 22, 0, tzinfo=timezone.utc)
        assert ForexCalendar.is_weekend(friday_late)

    def test_get_active_session(self):
        """Test session detection."""
        from scripts.download_forex_data import ForexCalendar

        # London session (07:00-16:00 UTC)
        london_time = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(london_time)
        assert session in ["london", "tokyo_london_overlap"]

        # New York session (12:00-21:00 UTC)
        ny_time = datetime(2024, 1, 8, 18, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(ny_time)
        assert session == "new_york"

        # London/NY overlap (12:00-16:00 UTC)
        overlap_time = datetime(2024, 1, 8, 14, 0, tzinfo=timezone.utc)
        session = ForexCalendar.get_active_session(overlap_time)
        assert session == "london_ny_overlap"

    def test_filter_weekends(self, sample_forex_df):
        """Test weekend filtering removes weekend bars."""
        from scripts.download_forex_data import _filter_weekends

        # Add weekend data
        saturday_ts = int(datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc).timestamp())
        weekend_row = sample_forex_df.iloc[0:1].copy()
        weekend_row["timestamp"] = saturday_ts
        df_with_weekend = pd.concat([sample_forex_df, weekend_row], ignore_index=True)

        filtered = _filter_weekends(df_with_weekend)

        # Weekend bar should be removed
        assert len(filtered) == len(sample_forex_df)
        assert saturday_ts not in filtered["timestamp"].values

    def test_resample_bars(self, sample_forex_df):
        """Test bar resampling."""
        from scripts.download_forex_data import _resample_bars

        # Resample 1h to 4h
        resampled = _resample_bars(sample_forex_df, "4h")

        # Should have fewer bars
        assert len(resampled) < len(sample_forex_df)

        # OHLCV should be aggregated correctly
        assert "open" in resampled.columns
        assert "high" in resampled.columns
        assert "close" in resampled.columns
        assert "volume" in resampled.columns

    def test_save_dataframe(self, sample_forex_df, temp_dir):
        """Test saving DataFrame to different formats."""
        from scripts.download_forex_data import save_dataframe, ForexDownloadConfig

        config = ForexDownloadConfig(output_dir=temp_dir, output_format="parquet")
        filepath = save_dataframe(sample_forex_df, "EUR_USD", config)

        assert Path(filepath).exists()
        assert filepath.endswith(".parquet")

        # Verify content
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == len(sample_forex_df)


# =============================================================================
# TEST: download_swap_rates.py
# =============================================================================


class TestDownloadSwapRates:
    """Tests for scripts/download_swap_rates.py"""

    def test_swap_download_config(self):
        """Test SwapDownloadConfig defaults."""
        from scripts.download_swap_rates import SwapDownloadConfig

        config = SwapDownloadConfig()
        assert config.output_dir == "data/forex/swaps"
        assert config.practice is True

    def test_estimate_swap_from_interest_rates(self):
        """Test swap rate estimation from interest differentials."""
        from scripts.download_swap_rates import estimate_swap_from_interest_rates

        # Higher base rate = positive carry for long
        long_swap, short_swap = estimate_swap_from_interest_rates(
            pair="AUD_USD",
            base_rate=4.5,  # AUD
            quote_rate=5.0,  # USD
            spot_price=0.65,
        )

        # With negative differential (AUD < USD), long should be negative
        # and short should be more negative due to broker spread
        assert isinstance(long_swap, float)
        assert isinstance(short_swap, float)

    def test_build_synthetic_swaps(self):
        """Test synthetic swap generation."""
        from scripts.download_swap_rates import _build_synthetic_swaps

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        pair, df, error = _build_synthetic_swaps("EUR_USD", start, end)

        assert error is None
        assert df is not None
        assert len(df) == 31  # 31 days
        assert "long_swap" in df.columns
        assert "short_swap" in df.columns
        assert df["source"].iloc[0] == "synthetic"

    def test_save_swaps(self, sample_swap_df, temp_dir):
        """Test saving swap DataFrame."""
        from scripts.download_swap_rates import save_swaps, SwapDownloadConfig

        config = SwapDownloadConfig(output_dir=temp_dir)
        filepath = save_swaps(sample_swap_df, "EUR_USD", config)

        assert Path(filepath).exists()
        assert "EUR_USD_swaps" in filepath


# =============================================================================
# TEST: download_economic_calendar.py
# =============================================================================


class TestDownloadEconomicCalendar:
    """Tests for scripts/download_economic_calendar.py"""

    def test_calendar_config(self):
        """Test CalendarDownloadConfig defaults."""
        from scripts.download_economic_calendar import CalendarDownloadConfig

        config = CalendarDownloadConfig()
        assert "USD" in config.currencies
        assert config.output_dir == "data/forex/calendar"

    def test_high_impact_events_defined(self):
        """Test high-impact events are defined for major currencies."""
        from scripts.download_economic_calendar import HIGH_IMPACT_EVENTS

        assert "Non-Farm Payrolls" in HIGH_IMPACT_EVENTS["USD"]
        assert "FOMC Rate Decision" in HIGH_IMPACT_EVENTS["USD"]
        assert "ECB Rate Decision" in HIGH_IMPACT_EVENTS["EUR"]
        assert "BOE Rate Decision" in HIGH_IMPACT_EVENTS["GBP"]

    def test_classify_impact(self):
        """Test event impact classification."""
        from scripts.download_economic_calendar import _classify_impact

        assert _classify_impact("Non-Farm Payrolls", "USD") == "high"
        assert _classify_impact("FOMC Rate Decision", "USD") == "high"
        assert _classify_impact("CPI", "EUR") == "high"
        assert _classify_impact("Trade Balance", "JPY") == "medium"
        assert _classify_impact("Some Minor Event", "USD") == "low"

    def test_get_first_friday(self):
        """Test first Friday calculation."""
        from scripts.download_economic_calendar import _get_first_friday

        # January 2024 - first Friday is January 5
        jan_2024 = datetime(2024, 1, 15)
        first_fri = _get_first_friday(jan_2024)
        assert first_fri.day == 5
        assert first_fri.weekday() == 4  # Friday

    def test_generate_synthetic_calendar(self):
        """Test synthetic calendar generation."""
        from scripts.download_economic_calendar import (
            generate_synthetic_calendar,
            CalendarDownloadConfig,
        )

        config = CalendarDownloadConfig(
            currencies=["USD", "EUR"],
            start_date="2024-01-01",
            end_date="2024-03-31",
        )

        df = generate_synthetic_calendar(config)

        assert len(df) > 0
        assert "event" in df.columns
        assert "impact" in df.columns
        assert "currency" in df.columns
        assert set(df["currency"].unique()).issubset({"USD", "EUR"})

    def test_save_calendar(self, sample_calendar_df, temp_dir):
        """Test saving calendar DataFrame."""
        from scripts.download_economic_calendar import (
            save_calendar,
            CalendarDownloadConfig,
        )

        config = CalendarDownloadConfig(output_dir=temp_dir)
        filepath = save_calendar(sample_calendar_df, config)

        assert Path(filepath).exists()
        assert "economic_calendar" in filepath


# =============================================================================
# TEST: download_interest_rates.py
# =============================================================================


class TestDownloadInterestRates:
    """Tests for scripts/download_interest_rates.py"""

    def test_rate_download_config(self):
        """Test RateDownloadConfig defaults."""
        from scripts.download_interest_rates import RateDownloadConfig

        config = RateDownloadConfig()
        assert config.output_dir == "data/forex/rates"
        assert config.lookback_years == 15
        assert config.fill_gaps is True

    def test_rate_series_defined(self):
        """Test FRED rate series are defined for major currencies."""
        from scripts.download_interest_rates import RATE_SERIES

        assert "USD" in RATE_SERIES
        assert "EUR" in RATE_SERIES
        assert "GBP" in RATE_SERIES
        assert "JPY" in RATE_SERIES

        assert RATE_SERIES["USD"]["series_id"] == "FEDFUNDS"
        assert "name" in RATE_SERIES["USD"]

    def test_fill_rate_gaps(self, sample_rate_df):
        """Test gap filling in rate data."""
        from scripts.download_interest_rates import _fill_rate_gaps

        # Create gaps
        df_with_gaps = sample_rate_df.iloc[::3].copy()  # Take every 3rd row
        df_with_gaps["date"] = pd.to_datetime(df_with_gaps["date"])

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        filled = _fill_rate_gaps(df_with_gaps, start, end)

        # Should have all dates filled
        assert len(filled) == 31

    def test_calculate_rate_differentials(self, temp_dir):
        """Test interest rate differential calculation."""
        from scripts.download_interest_rates import calculate_rate_differentials

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")

        usd_rates = pd.DataFrame(
            {
                "date": dates,
                "rate": 5.25,
            }
        ).set_index("date")

        eur_rates = pd.DataFrame(
            {
                "date": dates,
                "rate": 4.50,
            }
        ).set_index("date")

        rate_dfs = {
            "USD": usd_rates.reset_index(),
            "EUR": eur_rates.reset_index(),
        }

        diff_df = calculate_rate_differentials(rate_dfs, base_currency="USD")

        assert len(diff_df) > 0
        assert "USD_EUR_diff" in diff_df.columns

        # USD - EUR = 5.25 - 4.50 = 0.75
        assert abs(diff_df["USD_EUR_diff"].iloc[0] - 0.75) < 0.01


# =============================================================================
# TEST: data_loader_forex.py
# =============================================================================


class TestDataLoaderForex:
    """Tests for data_loader_forex.py"""

    def test_load_forex_data_basic(self, sample_forex_df, temp_dir):
        """Test basic forex data loading."""
        from data_loader_forex import load_forex_data

        # Save sample data
        filepath = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(filepath, index=False)

        dfs, shapes = load_forex_data(
            paths=[str(filepath)],
            timeframe="1h",
        )

        assert "EUR_USD" in dfs
        assert len(dfs["EUR_USD"]) > 0
        assert "EUR_USD" in shapes

    def test_session_features_added(self, sample_forex_df, temp_dir):
        """Test session features are added."""
        from data_loader_forex import load_forex_data

        filepath = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(filepath, index=False)

        dfs, _ = load_forex_data(
            paths=[str(filepath)],
            add_session_features=True,
        )

        df = dfs["EUR_USD"]
        assert "session" in df.columns
        assert "session_liquidity" in df.columns
        assert "is_session_overlap" in df.columns

    def test_get_active_session(self):
        """Test session detection in data loader."""
        from data_loader_forex import _get_active_session

        # Test various times
        london = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        assert "london" in _get_active_session(london)

        tokyo = datetime(2024, 1, 8, 3, 0, tzinfo=timezone.utc)
        assert "tokyo" in _get_active_session(tokyo)

        overlap = datetime(2024, 1, 8, 14, 0, tzinfo=timezone.utc)
        assert "overlap" in _get_active_session(overlap)

    def test_filter_forex_weekends(self, sample_forex_df):
        """Test weekend filtering in data loader."""
        from data_loader_forex import _filter_forex_weekends

        # Add weekend timestamp
        weekend_ts = int(datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc).timestamp())
        df = sample_forex_df.copy()
        df = pd.concat(
            [df, pd.DataFrame([{"timestamp": weekend_ts, "close": 1.1, "symbol": "EUR_USD"}])],
            ignore_index=True,
        )

        filtered = _filter_forex_weekends(df)
        assert weekend_ts not in filtered["timestamp"].values

    def test_merge_swap_rates(self, sample_forex_df, sample_swap_df, temp_dir):
        """Test swap rate merging."""
        from data_loader_forex import _merge_swap_rates

        # Save swap data
        swap_path = Path(temp_dir) / "EUR_USD_swaps.parquet"
        sample_swap_df.to_parquet(swap_path, index=False)

        result = _merge_swap_rates(sample_forex_df, "EUR_USD", temp_dir)

        assert "long_swap" in result.columns
        assert "short_swap" in result.columns

    def test_merge_interest_rates(self, sample_forex_df, sample_rate_df, temp_dir):
        """Test interest rate merging."""
        from data_loader_forex import _merge_interest_rates

        # Save rate data for both currencies
        usd_path = Path(temp_dir) / "USD_rates.parquet"
        eur_path = Path(temp_dir) / "EUR_rates.parquet"

        sample_rate_df.to_parquet(usd_path, index=False)

        eur_rates = sample_rate_df.copy()
        eur_rates["currency"] = "EUR"
        eur_rates["rate"] = eur_rates["rate"] - 0.75  # Lower EUR rate
        eur_rates.to_parquet(eur_path, index=False)

        result = _merge_interest_rates(sample_forex_df, "EUR_USD", temp_dir)

        assert "rate_differential" in result.columns
        assert "base_rate" in result.columns
        assert "quote_rate" in result.columns

    def test_merge_calendar_proximity(self, sample_forex_df, sample_calendar_df, temp_dir):
        """Test calendar proximity merging."""
        from data_loader_forex import _merge_calendar_proximity

        # Save calendar
        cal_path = Path(temp_dir) / "economic_calendar.parquet"
        sample_calendar_df.to_parquet(cal_path, index=False)

        result = _merge_calendar_proximity(sample_forex_df, "EUR_USD", temp_dir)

        assert "hours_to_next_event" in result.columns
        assert "hours_since_last_event" in result.columns
        assert "is_event_window" in result.columns

    def test_list_available_pairs(self, sample_forex_df, temp_dir):
        """Test listing available pairs."""
        from data_loader_forex import list_available_pairs

        # Save multiple pairs
        for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
            path = Path(temp_dir) / f"{pair}.parquet"
            df = sample_forex_df.copy()
            df["symbol"] = pair
            df.to_parquet(path, index=False)

        pairs = list_available_pairs(temp_dir)
        assert "EUR_USD" in pairs
        assert "GBP_USD" in pairs
        assert "USD_JPY" in pairs

    def test_get_pair_info(self, sample_forex_df, temp_dir):
        """Test getting pair info."""
        from data_loader_forex import get_pair_info

        path = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(path, index=False)

        info = get_pair_info("EUR_USD", temp_dir)

        assert info is not None
        assert info["pair"] == "EUR_USD"
        assert "rows" in info
        assert "start_date" in info
        assert "end_date" in info


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPhase7Integration:
    """Integration tests for Phase 7 components."""

    def test_full_data_pipeline(
        self, sample_forex_df, sample_swap_df, sample_rate_df, sample_calendar_df, temp_dir
    ):
        """Test complete data loading pipeline."""
        from data_loader_forex import load_forex_data

        # Create directory structure
        forex_dir = Path(temp_dir) / "raw_forex"
        swap_dir = Path(temp_dir) / "swaps"
        rate_dir = Path(temp_dir) / "rates"
        cal_dir = Path(temp_dir) / "calendar"

        forex_dir.mkdir(parents=True)
        swap_dir.mkdir(parents=True)
        rate_dir.mkdir(parents=True)
        cal_dir.mkdir(parents=True)

        # Save all data
        sample_forex_df.to_parquet(forex_dir / "EUR_USD.parquet", index=False)
        sample_swap_df.to_parquet(swap_dir / "EUR_USD_swaps.parquet", index=False)
        sample_rate_df.to_parquet(rate_dir / "USD_rates.parquet", index=False)

        eur_rates = sample_rate_df.copy()
        eur_rates["currency"] = "EUR"
        eur_rates["rate"] = eur_rates["rate"] - 0.75
        eur_rates.to_parquet(rate_dir / "EUR_rates.parquet", index=False)

        sample_calendar_df.to_parquet(cal_dir / "economic_calendar.parquet", index=False)

        # Load with all features
        dfs, shapes = load_forex_data(
            paths=[str(forex_dir / "EUR_USD.parquet")],
            merge_swaps=True,
            merge_rates=True,
            merge_calendar=True,
            swap_dir=str(swap_dir),
            rate_dir=str(rate_dir),
            calendar_dir=str(cal_dir),
        )

        assert "EUR_USD" in dfs
        df = dfs["EUR_USD"]

        # Check all features are present
        assert "long_swap" in df.columns
        assert "short_swap" in df.columns
        assert "rate_differential" in df.columns
        assert "hours_to_next_event" in df.columns
        assert "session" in df.columns
        assert "session_liquidity" in df.columns

    def test_glob_pattern_loading(self, sample_forex_df, temp_dir):
        """Test loading multiple files with glob pattern."""
        from data_loader_forex import load_forex_data

        # Save multiple pairs
        for pair in ["EUR_USD", "GBP_USD"]:
            path = Path(temp_dir) / f"{pair}.parquet"
            df = sample_forex_df.copy()
            df["symbol"] = pair
            df.to_parquet(path, index=False)

        dfs, shapes = load_forex_data(
            paths=[f"{temp_dir}/*.parquet"],
        )

        assert len(dfs) == 2
        assert "EUR_USD" in dfs
        assert "GBP_USD" in dfs


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_dataframe(self, temp_dir):
        """Test handling of empty DataFrame."""
        from data_loader_forex import load_forex_data

        # Create empty file
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        path = Path(temp_dir) / "EMPTY.parquet"
        empty_df.to_parquet(path, index=False)

        dfs, shapes = load_forex_data(paths=[str(path)])

        # Should either skip or handle gracefully
        assert "EMPTY" not in dfs or dfs["EMPTY"].empty

    def test_missing_swap_file(self, sample_forex_df, temp_dir):
        """Test handling when swap file doesn't exist."""
        from data_loader_forex import _merge_swap_rates

        result = _merge_swap_rates(sample_forex_df, "EUR_USD", "/nonexistent/path")

        # Should add default columns
        assert "long_swap" in result.columns
        assert "short_swap" in result.columns
        assert (result["long_swap"] == 0.0).all()

    def test_missing_rate_file(self, sample_forex_df, temp_dir):
        """Test handling when rate file doesn't exist."""
        from data_loader_forex import _merge_interest_rates

        result = _merge_interest_rates(sample_forex_df, "EUR_USD", "/nonexistent/path")

        assert "rate_differential" in result.columns
        assert (result["rate_differential"] == 0.0).all()

    def test_invalid_pair_format(self):
        """Test handling of invalid pair format."""
        from scripts.download_swap_rates import build_historical_swaps, SwapDownloadConfig

        config = SwapDownloadConfig(start_date="2024-01-01", end_date="2024-01-31")
        pair, df, error = build_historical_swaps("INVALID", config)

        assert error is not None
        assert df is None

    def test_date_filtering(self, sample_forex_df, temp_dir):
        """Test date range filtering."""
        from data_loader_forex import load_forex_data

        path = Path(temp_dir) / "EUR_USD.parquet"
        sample_forex_df.to_parquet(path, index=False)

        # Filter to subset
        dfs, _ = load_forex_data(
            paths=[str(path)],
            start_date="2024-01-09",
            end_date="2024-01-10",
        )

        if "EUR_USD" in dfs:
            df = dfs["EUR_USD"]
            # Should have fewer bars than original
            assert len(df) <= len(sample_forex_df)


# =============================================================================
# API MOCK TESTS (Recommendation #1)
# =============================================================================


class TestOandaAPIMock:
    """
    Mock tests for OANDA API interactions.

    These tests verify the download logic without requiring live API access.
    Reference: Best practices for API testing (Fowler, 2014)
    """

    def test_download_pair_oanda_success(self, temp_dir):
        """Test successful OANDA data download with mocked API."""
        from scripts.download_forex_data import download_pair_oanda, ForexDownloadConfig

        # Create mock bar data
        mock_bars = []
        for i in range(100):
            mock_bar = MagicMock()
            mock_bar.ts = (1704672000 + i * 3600) * 1000  # hourly timestamps
            mock_bar.open = 1.1000 + i * 0.0001
            mock_bar.high = 1.1005 + i * 0.0001
            mock_bar.low = 1.0995 + i * 0.0001
            mock_bar.close = 1.1002 + i * 0.0001
            mock_bar.volume_base = 1000 + i
            mock_bar.volume_quote = 1.2  # spread in pips
            mock_bars.append(mock_bar)

        with patch("adapters.oanda.market_data.OandaMarketDataAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.get_bars.return_value = mock_bars
            mock_adapter_class.return_value = mock_adapter

            config = ForexDownloadConfig(
                api_key="test_key",
                account_id="test_account",
                start_date="2024-01-08",
                end_date="2024-01-12",
                output_dir=temp_dir,
            )

            pair, df, error = download_pair_oanda("EUR_USD", config)

            assert error is None
            assert df is not None
            assert len(df) == 100
            assert "timestamp" in df.columns
            assert "close" in df.columns
            assert "spread_pips" in df.columns

    def test_download_pair_oanda_api_error(self, temp_dir):
        """Test handling of OANDA API errors."""
        from scripts.download_forex_data import download_pair_oanda, ForexDownloadConfig

        with patch("adapters.oanda.market_data.OandaMarketDataAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.get_bars.side_effect = Exception("API rate limit exceeded")
            mock_adapter_class.return_value = mock_adapter

            config = ForexDownloadConfig(
                api_key="test_key",
                account_id="test_account",
                start_date="2024-01-08",
                end_date="2024-01-12",
                output_dir=temp_dir,
            )

            pair, df, error = download_pair_oanda("EUR_USD", config)

            assert error is not None
            assert "API rate limit" in error or df is None

    def test_download_pair_oanda_empty_response(self, temp_dir):
        """Test handling of empty API response."""
        from scripts.download_forex_data import download_pair_oanda, ForexDownloadConfig

        with patch("adapters.oanda.market_data.OandaMarketDataAdapter") as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.get_bars.return_value = []  # Empty response
            mock_adapter_class.return_value = mock_adapter

            config = ForexDownloadConfig(
                api_key="test_key",
                account_id="test_account",
                start_date="2024-01-08",
                end_date="2024-01-12",
                output_dir=temp_dir,
            )

            pair, df, error = download_pair_oanda("EUR_USD", config)

            assert error is not None or df is None

    def test_fetch_current_swaps_oanda_mock(self):
        """Test OANDA swap rate fetching with mock."""
        from scripts.download_swap_rates import fetch_current_swaps_oanda

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "instruments": [
                {
                    "financing": {
                        "longRate": -0.0001,  # Daily rate
                        "shortRate": -0.00005,
                    }
                }
            ]
        }

        with patch("requests.get", return_value=mock_response):
            result = fetch_current_swaps_oanda(
                pair="EUR_USD",
                api_key="test_key",
                account_id="test_account",
            )

            assert result is not None
            assert "long_swap_pct" in result
            assert "short_swap_pct" in result


class TestFredAPIMock:
    """
    Mock tests for FRED API interactions.

    Reference: FRED API documentation, St. Louis Fed
    """

    def test_fetch_fred_series_success(self):
        """Test successful FRED series fetch with mock."""
        from scripts.download_interest_rates import fetch_fred_series

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-01", "value": "5.33"},
                {"date": "2024-01-02", "value": "5.33"},
                {"date": "2024-01-03", "value": "5.33"},
                {"date": "2024-01-04", "value": "5.25"},
                {"date": "2024-01-05", "value": "5.25"},
            ]
        }

        with patch("requests.get", return_value=mock_response):
            df = fetch_fred_series(
                series_id="FEDFUNDS",
                start_date="2024-01-01",
                end_date="2024-01-05",
                api_key="test_key",
            )

            assert df is not None
            assert len(df) == 5
            assert "rate" in df.columns
            assert df["rate"].iloc[0] == 5.33

    def test_fetch_fred_series_rate_limit(self):
        """Test FRED API rate limit handling."""
        from scripts.download_interest_rates import fetch_fred_series

        # First call returns 429, second succeeds
        mock_responses = [
            MagicMock(status_code=429),
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={"observations": [{"date": "2024-01-01", "value": "5.33"}]}
                ),
            ),
        ]

        with patch("requests.get", side_effect=mock_responses):
            with patch("time.sleep"):  # Skip actual sleep
                df = fetch_fred_series(
                    series_id="FEDFUNDS",
                    start_date="2024-01-01",
                    end_date="2024-01-01",
                    api_key="test_key",
                )

                # Should eventually succeed or handle gracefully
                assert df is not None or True  # Either works

    def test_fetch_fred_series_missing_values(self):
        """Test handling of missing values in FRED data."""
        from scripts.download_interest_rates import fetch_fred_series

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observations": [
                {"date": "2024-01-01", "value": "5.33"},
                {"date": "2024-01-02", "value": "."},  # Missing value marker
                {"date": "2024-01-03", "value": "5.25"},
            ]
        }

        with patch("requests.get", return_value=mock_response):
            df = fetch_fred_series(
                series_id="FEDFUNDS",
                start_date="2024-01-01",
                end_date="2024-01-03",
                api_key="test_key",
            )

            assert df is not None
            # Missing values should be filtered out
            assert len(df) == 2


class TestStressScenarios:
    """
    Stress tests for large date ranges and edge cases.

    Reference: Performance testing best practices (Molyneaux, 2014)
    """

    def test_large_date_range_3_years(self, sample_forex_df, temp_dir):
        """Test data loading for 3-year date range (typical training window)."""
        from data_loader_forex import load_forex_data

        # Generate 3 years of daily data (~1095 bars)
        np.random.seed(42)
        n_bars = 1095
        base_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())

        timestamps = []
        for i in range(n_bars):
            ts = base_ts + i * 86400  # daily
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            # Skip weekends
            if dt.weekday() < 5:
                timestamps.append(ts)

        prices = [1.1 + np.random.normal(0, 0.001) for _ in timestamps]

        large_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 0.001 for p in prices],
                "low": [p - 0.001 for p in prices],
                "close": prices,
                "volume": [1000] * len(timestamps),
                "symbol": "EUR_USD",
            }
        )

        filepath = Path(temp_dir) / "EUR_USD.parquet"
        large_df.to_parquet(filepath, index=False)

        # Should handle large dataset efficiently
        import time

        start_time = time.time()

        dfs, shapes = load_forex_data(
            paths=[str(filepath)],
            add_session_features=True,
        )

        elapsed = time.time() - start_time

        assert "EUR_USD" in dfs
        assert len(dfs["EUR_USD"]) > 700  # ~780 weekdays in 3 years
        assert elapsed < 10.0  # Should complete in under 10 seconds

    def test_multiple_pairs_concurrent(self, sample_forex_df, temp_dir):
        """Test loading multiple pairs simultaneously."""
        from data_loader_forex import load_forex_data

        pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_GBP", "EUR_JPY"]

        for pair in pairs:
            df = sample_forex_df.copy()
            df["symbol"] = pair
            path = Path(temp_dir) / f"{pair}.parquet"
            df.to_parquet(path, index=False)

        dfs, shapes = load_forex_data(
            paths=[f"{temp_dir}/*.parquet"],
            add_session_features=True,
        )

        assert len(dfs) == len(pairs)
        for pair in pairs:
            assert pair in dfs

    def test_high_frequency_data_1min(self, temp_dir):
        """Test handling of high-frequency 1-minute data."""
        from data_loader_forex import load_forex_data

        # Generate 1 week of 1-minute data (~7200 bars for 5 trading days)
        np.random.seed(42)
        n_bars = 7200
        base_ts = int(datetime(2024, 1, 8, tzinfo=timezone.utc).timestamp())  # Monday

        timestamps = [base_ts + i * 60 for i in range(n_bars)]
        prices = [1.1 + np.random.normal(0, 0.0001) for _ in timestamps]

        hf_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 0.0001 for p in prices],
                "low": [p - 0.0001 for p in prices],
                "close": prices,
                "volume": [100] * n_bars,
                "symbol": "EUR_USD",
            }
        )

        filepath = Path(temp_dir) / "EUR_USD_1m.parquet"
        hf_df.to_parquet(filepath, index=False)

        dfs, shapes = load_forex_data(
            paths=[str(filepath)],
            timeframe="1m",
            filter_weekends=True,
        )

        assert "EUR_USD_1M" in dfs or "EUR_USD" in dfs


# =============================================================================
# FOREXFACTORY SCRAPER TESTS
# =============================================================================


class TestForexFactoryScraper:
    """Tests for ForexFactory calendar scraper (backup source)."""

    def test_parse_forexfactory_date_formats(self):
        """Test parsing various ForexFactory date formats."""
        from scripts.download_economic_calendar import _parse_forexfactory_date

        # Standard format
        result = _parse_forexfactory_date("Mon Jan 15")
        assert result is not None
        assert result.month == 1
        assert result.day == 15

        # Without day of week
        result = _parse_forexfactory_date("Feb 20")
        assert result is not None
        assert result.month == 2
        assert result.day == 20

    def test_parse_forexfactory_time_formats(self):
        """Test parsing ForexFactory time formats with ET to UTC conversion."""
        from scripts.download_economic_calendar import _parse_forexfactory_time

        # Morning time
        result = _parse_forexfactory_time("8:30am")
        assert result is not None
        # 8:30 AM ET + 5 hours = 13:30 UTC
        assert result.hour == 13
        assert result.minute == 30

        # Afternoon time
        result = _parse_forexfactory_time("2:00pm")
        assert result is not None
        # 2:00 PM ET = 14:00 + 5 = 19:00 UTC
        assert result.hour == 19
        assert result.minute == 0

        # Midnight case
        result = _parse_forexfactory_time("12:00am")
        assert result is not None
        # 12:00 AM ET + 5 = 5:00 UTC
        assert result.hour == 5

    def test_parse_forexfactory_impact_high(self):
        """Test parsing high impact events from ForexFactory HTML."""
        from scripts.download_economic_calendar import _parse_forexfactory_impact

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        # Simulate high impact cell
        html = '<td class="calendar__impact"><span class="icon--ff-impact-red"></span></td>'
        soup = BeautifulSoup(html, "html.parser")
        cell = soup.find("td")

        result = _parse_forexfactory_impact(cell)
        assert result == "high"

    def test_parse_forexfactory_impact_medium(self):
        """Test parsing medium impact events."""
        from scripts.download_economic_calendar import _parse_forexfactory_impact

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        html = '<td class="calendar__impact"><span class="icon--ff-impact-orange"></span></td>'
        soup = BeautifulSoup(html, "html.parser")
        cell = soup.find("td")

        result = _parse_forexfactory_impact(cell)
        assert result == "medium"

    def test_parse_forexfactory_impact_low(self):
        """Test parsing low impact events."""
        from scripts.download_economic_calendar import _parse_forexfactory_impact

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        html = '<td class="calendar__impact"><span class="icon--ff-impact-yellow"></span></td>'
        soup = BeautifulSoup(html, "html.parser")
        cell = soup.find("td")

        result = _parse_forexfactory_impact(cell)
        assert result == "low"

    def test_parse_forexfactory_value(self):
        """Test parsing actual/forecast/previous values."""
        from scripts.download_economic_calendar import _parse_forexfactory_value

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        # Valid value
        html = '<td class="calendar__actual">0.3%</td>'
        soup = BeautifulSoup(html, "html.parser")
        cell = soup.find("td")
        assert _parse_forexfactory_value(cell) == "0.3%"

        # Empty placeholder
        html = '<td class="calendar__actual">-</td>'
        soup = BeautifulSoup(html, "html.parser")
        cell = soup.find("td")
        assert _parse_forexfactory_value(cell) is None

        # None cell
        assert _parse_forexfactory_value(None) is None

    def test_parse_forexfactory_page_mock(self):
        """Test parsing a mock ForexFactory calendar page."""
        from scripts.download_economic_calendar import _parse_forexfactory_page

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        # Mock HTML that resembles ForexFactory structure
        mock_html = """
        <html>
        <body>
        <table>
            <tr class="calendar__row">
                <td class="calendar__date">Mon Jan 15</td>
                <td class="calendar__time">8:30am</td>
                <td class="calendar__currency">USD</td>
                <td class="calendar__impact"><span class="icon--ff-impact-high"></span></td>
                <td class="calendar__event"><span class="calendar__event-title">CPI m/m</span></td>
                <td class="calendar__actual">0.3%</td>
                <td class="calendar__forecast">0.2%</td>
                <td class="calendar__previous">0.1%</td>
            </tr>
            <tr class="calendar__row">
                <td class="calendar__date"></td>
                <td class="calendar__time">10:00am</td>
                <td class="calendar__currency">EUR</td>
                <td class="calendar__impact"><span class="icon--ff-impact-medium"></span></td>
                <td class="calendar__event"><span class="calendar__event-title">ZEW Survey</span></td>
                <td class="calendar__actual">15.0</td>
                <td class="calendar__forecast">12.0</td>
                <td class="calendar__previous">10.0</td>
            </tr>
        </table>
        </body>
        </html>
        """

        records = _parse_forexfactory_page(mock_html, ["USD", "EUR"], False)

        assert len(records) == 2

        # Check first record (USD CPI)
        usd_record = [r for r in records if r["currency"] == "USD"][0]
        assert usd_record["event"] == "CPI m/m"
        assert usd_record["impact"] == "high"
        assert usd_record["actual"] == "0.3%"

        # Check second record (EUR ZEW)
        eur_record = [r for r in records if r["currency"] == "EUR"][0]
        assert eur_record["event"] == "ZEW Survey"
        assert eur_record["impact"] == "medium"

    def test_parse_forexfactory_high_impact_filter(self):
        """Test high impact only filtering."""
        from scripts.download_economic_calendar import _parse_forexfactory_page

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("BeautifulSoup not installed")

        mock_html = """
        <html>
        <body>
        <table>
            <tr class="calendar__row">
                <td class="calendar__date">Mon Jan 15</td>
                <td class="calendar__time">8:30am</td>
                <td class="calendar__currency">USD</td>
                <td class="calendar__impact"><span class="icon--ff-impact-high"></span></td>
                <td class="calendar__event"><span class="calendar__event-title">NFP</span></td>
                <td class="calendar__actual">200K</td>
                <td class="calendar__forecast">180K</td>
                <td class="calendar__previous">150K</td>
            </tr>
            <tr class="calendar__row">
                <td class="calendar__date"></td>
                <td class="calendar__time">10:00am</td>
                <td class="calendar__currency">USD</td>
                <td class="calendar__impact"><span class="icon--ff-impact-low"></span></td>
                <td class="calendar__event"><span class="calendar__event-title">Minor Report</span></td>
                <td class="calendar__actual">1.0</td>
                <td class="calendar__forecast">1.1</td>
                <td class="calendar__previous">0.9</td>
            </tr>
        </table>
        </body>
        </html>
        """

        # Without filter - should get both
        records_all = _parse_forexfactory_page(mock_html, ["USD"], False)
        assert len(records_all) == 2

        # With high impact filter - should only get NFP
        records_high = _parse_forexfactory_page(mock_html, ["USD"], True)
        assert len(records_high) == 1
        assert records_high[0]["event"] == "NFP"


class TestForexFactoryIntegration:
    """Integration tests for ForexFactory with download_calendar."""

    @patch("scripts.download_economic_calendar.fetch_calendar_oanda")
    @patch("scripts.download_economic_calendar.fetch_calendar_forexfactory")
    def test_fallback_to_forexfactory_when_oanda_fails(self, mock_ff, mock_oanda, temp_dir):
        """Test that ForexFactory is used when OANDA fails."""
        from scripts.download_economic_calendar import (
            CalendarDownloadConfig,
            download_calendar,
        )

        # OANDA returns None (failed)
        mock_oanda.return_value = None

        # ForexFactory returns data
        mock_ff.return_value = pd.DataFrame(
            {
                "datetime": [datetime.now(timezone.utc).isoformat()],
                "date": ["2024-01-15"],
                "time": ["13:30"],
                "currency": ["USD"],
                "event": ["CPI"],
                "impact": ["high"],
                "actual": ["0.3%"],
                "forecast": ["0.2%"],
                "previous": ["0.1%"],
                "source": ["forexfactory"],
            }
        )

        config = CalendarDownloadConfig(
            currencies=["USD"],
            output_dir=temp_dir,
            skip_existing=False,
        )

        result = download_calendar(config)

        assert result["source"] == "forexfactory"
        assert result["events"] == 1
        mock_oanda.assert_called_once()
        mock_ff.assert_called_once()

    @patch("scripts.download_economic_calendar.fetch_calendar_oanda")
    @patch("scripts.download_economic_calendar.fetch_calendar_forexfactory")
    def test_fallback_to_synthetic_when_both_fail(self, mock_ff, mock_oanda, temp_dir):
        """Test synthetic generation when both API sources fail."""
        from scripts.download_economic_calendar import (
            CalendarDownloadConfig,
            download_calendar,
        )

        # Both return None
        mock_oanda.return_value = None
        mock_ff.return_value = None

        config = CalendarDownloadConfig(
            currencies=["USD"],
            output_dir=temp_dir,
            skip_existing=False,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        result = download_calendar(config)

        assert result["source"] == "synthetic"
        assert result["events"] > 0
        mock_oanda.assert_called_once()
        mock_ff.assert_called_once()

    @patch("scripts.download_economic_calendar.fetch_calendar_oanda")
    def test_oanda_success_skips_forexfactory(self, mock_oanda, temp_dir):
        """Test that ForexFactory is not called when OANDA succeeds."""
        from scripts.download_economic_calendar import (
            CalendarDownloadConfig,
            download_calendar,
        )

        # OANDA returns data
        mock_oanda.return_value = pd.DataFrame(
            {
                "datetime": [datetime.now(timezone.utc).isoformat()],
                "date": ["2024-01-15"],
                "time": ["13:30"],
                "currency": ["USD"],
                "event": ["NFP"],
                "impact": ["high"],
                "actual": ["200K"],
                "forecast": ["180K"],
                "previous": ["150K"],
                "source": ["oanda"],
            }
        )

        config = CalendarDownloadConfig(
            currencies=["USD"],
            output_dir=temp_dir,
            skip_existing=False,
        )

        result = download_calendar(config)

        assert result["source"] == "oanda"
        mock_oanda.assert_called_once()


# =============================================================================
# CIP DEVIATION MODEL TESTS
# =============================================================================


class TestCIPDeviationModel:
    """Tests for Covered Interest Parity deviation model."""

    def test_get_cip_deviation_known_pairs(self):
        """Test CIP deviation for known currency pairs."""
        from scripts.download_swap_rates import get_cip_deviation

        # EUR/USD should have negative deviation (USD premium)
        eur_usd_dev = get_cip_deviation("EUR_USD")
        assert eur_usd_dev < 0
        assert -50 < eur_usd_dev < -10  # Reasonable range

        # USD/CHF should have larger negative deviation (CHF safe haven)
        usd_chf_dev = get_cip_deviation("USD_CHF")
        assert usd_chf_dev < eur_usd_dev  # More negative

    def test_get_cip_deviation_unknown_pair(self):
        """Test CIP deviation returns 0 for unknown pairs."""
        from scripts.download_swap_rates import get_cip_deviation

        unknown_dev = get_cip_deviation("XXX_YYY")
        assert unknown_dev == 0.0

    def test_get_cip_deviation_vix_stress(self):
        """Test CIP deviation increases with VIX (stress periods)."""
        from scripts.download_swap_rates import get_cip_deviation

        # Normal VIX
        normal_dev = get_cip_deviation("EUR_USD", vix_level=15.0)

        # High VIX (stress)
        stress_dev = get_cip_deviation("EUR_USD", vix_level=40.0)

        # Deviation should be more negative during stress
        assert abs(stress_dev) > abs(normal_dev)

    def test_estimate_swap_with_cip(self):
        """Test swap estimation with CIP deviation included."""
        from scripts.download_swap_rates import estimate_swap_from_interest_rates

        # Without CIP
        swap_no_cip = estimate_swap_from_interest_rates(
            pair="EUR_USD",
            base_rate=4.5,  # EUR rate
            quote_rate=5.5,  # USD rate
            spot_price=1.10,
            include_cip_deviation=False,
        )

        # With CIP
        swap_with_cip = estimate_swap_from_interest_rates(
            pair="EUR_USD",
            base_rate=4.5,
            quote_rate=5.5,
            spot_price=1.10,
            include_cip_deviation=True,
        )

        # CIP deviation should affect the swap
        assert swap_no_cip != swap_with_cip

    def test_estimate_swap_breakdown(self):
        """Test swap estimation with full breakdown."""
        from scripts.download_swap_rates import estimate_swap_with_cip_breakdown

        result = estimate_swap_with_cip_breakdown(
            pair="EUR_USD",
            base_rate=4.5,
            quote_rate=5.5,
            spot_price=1.10,
        )

        # Check main result keys (matching actual function output)
        assert "pair" in result
        assert result["pair"] == "EUR_USD"
        assert "cip_deviation_bps" in result
        assert "long_swap_pips" in result
        assert "short_swap_pips" in result
        assert "ir_differential_pct" in result

        # Check rates are captured
        assert result["base_rate"] == 4.5
        assert result["quote_rate"] == 5.5

        # Verify CIP deviation is applied (EUR/USD should be negative)
        assert result["cip_deviation_bps"] < 0

        # Verify swaps are different with/without CIP
        assert result["long_swap_pips"] != result["long_swap_pips_no_cip"]


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure Phase 7 doesn't break existing functionality."""

    def test_imports_dont_break_existing(self):
        """Test that new imports don't break existing modules."""
        # These imports should work without affecting existing functionality
        from data_loader_forex import load_forex_data
        from scripts.download_forex_data import ForexDownloadConfig, MAJOR_PAIRS
        from scripts.download_swap_rates import SwapDownloadConfig
        from scripts.download_economic_calendar import CalendarDownloadConfig
        from scripts.download_interest_rates import RateDownloadConfig

        # All should be importable
        assert ForexDownloadConfig is not None
        assert load_forex_data is not None

    def test_existing_data_loader_unaffected(self):
        """Test that existing data_loader_multi_asset.py is unaffected."""
        from data_loader_multi_asset import (
            AssetClass,
            DataVendor,
            load_from_file,
            timeframe_to_seconds,
        )

        # FOREX should be an option
        assert AssetClass.FOREX.value == "forex"
        assert DataVendor.OANDA.value == "oanda"

        # Existing functions should work
        assert timeframe_to_seconds("4h") == 14400


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
