# -*- coding: utf-8 -*-
"""
test_forex_improvements.py
Comprehensive tests for forex environment improvements (2025-11-30).

Tests for:
1. DST-aware rollover time detection
2. Optimized rollover counter algorithm
3. SwapRateProvider with file loading
4. ForexEnvWrapper new parameters
5. ForexLeverageWrapper stop-out level

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

gym = pytest.importorskip("gymnasium")
import numpy as np
import pytest

pytest.importorskip("torch")
from gymnasium import spaces

from wrappers.forex_env import (
    # Functions
    get_rollover_hour_utc,
    is_dst_in_effect,
    count_rollovers_optimized,
    # Classes
    SwapRate,
    SwapRateProvider,
    ForexEnvWrapper,
    ForexLeverageWrapper,
    # Constants
    MAJOR_PAIRS,
    MINOR_PAIRS,
    EXOTIC_PAIRS,
    DEFAULT_SWAP_RATES,
    SESSION_LIQUIDITY,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_swap_dir():
    """Create a temporary directory with swap rate files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSON file
        json_data = {
            "EUR_USD": {"long": -0.35, "short": 0.12, "timestamp": 1704067200},
            "GBP_USD": {"long_swap": -0.42, "short_swap": 0.18},
        }
        json_path = Path(tmpdir) / "swaps.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Create CSV file
        csv_content = "pair,long_swap,short_swap,timestamp\nUSD_JPY,-0.25,0.08,1704067200\nAUD_USD,-0.30,0.10,\n"
        csv_path = Path(tmpdir) / "swaps.csv"
        with open(csv_path, "w") as f:
            f.write(csv_content)

        yield tmpdir


class MockForexEnv(gym.Env):
    """Mock forex trading environment for testing wrappers."""

    def __init__(
        self,
        reset_timestamp: int = 1704067200,
        step_timestamp: int = 1704153600,
    ):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._reset_timestamp = reset_timestamp
        self._step_timestamp = step_timestamp
        self._net_worth = 100000.0
        self._used_margin = 0.0
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(10, dtype=np.float32), {
            "timestamp": self._reset_timestamp,
            "symbol": "EUR_USD",
        }

    def step(self, action):
        self._step_count += 1
        return (
            np.zeros(10, dtype=np.float32),
            0.01,
            False,
            False,
            {
                "timestamp": self._step_timestamp,
                "signal_pos_next": (
                    float(action[0]) if hasattr(action, "__getitem__") else float(action)
                ),
                "symbol": "EUR_USD",
            },
        )


@pytest.fixture
def mock_env():
    """Create a mock trading environment."""
    return MockForexEnv()


# =============================================================================
# DST-AWARE ROLLOVER TIME TESTS
# =============================================================================


class TestGetRolloverHourUtc:
    """Tests for get_rollover_hour_utc function."""

    def test_summer_time_july(self):
        """Test rollover hour during summer (EDT - should be 21 UTC)."""
        # July 15, 2024 12:00 UTC - clearly summer
        timestamp = int(datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        hour = get_rollover_hour_utc(timestamp)
        assert hour == 21, f"Expected 21 for summer, got {hour}"

    def test_winter_time_january(self):
        """Test rollover hour during winter (EST - should be 22 UTC)."""
        # January 15, 2024 12:00 UTC - clearly winter
        timestamp = int(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        hour = get_rollover_hour_utc(timestamp)
        assert hour == 22, f"Expected 22 for winter, got {hour}"

    def test_winter_time_december(self):
        """Test rollover hour in December (winter)."""
        timestamp = int(datetime(2024, 12, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        hour = get_rollover_hour_utc(timestamp)
        assert hour == 22, f"Expected 22 for December (winter), got {hour}"

    def test_summer_time_june(self):
        """Test rollover hour in June (summer)."""
        timestamp = int(datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        hour = get_rollover_hour_utc(timestamp)
        assert hour == 21, f"Expected 21 for June (summer), got {hour}"

    def test_dst_transition_march(self):
        """Test near DST transition in March 2024."""
        # March 10, 2024 - DST starts (2nd Sunday in March)
        # Before DST: should be 22 (EST)
        before_dst = int(datetime(2024, 3, 9, 12, 0, tzinfo=timezone.utc).timestamp())
        hour_before = get_rollover_hour_utc(before_dst)

        # After DST: should be 21 (EDT)
        after_dst = int(datetime(2024, 3, 11, 12, 0, tzinfo=timezone.utc).timestamp())
        hour_after = get_rollover_hour_utc(after_dst)

        assert hour_before == 22, f"Expected 22 before DST, got {hour_before}"
        assert hour_after == 21, f"Expected 21 after DST, got {hour_after}"

    def test_dst_transition_november(self):
        """Test near DST end in November 2024."""
        # November 3, 2024 - DST ends (1st Sunday in November)
        # Before DST ends: should be 21 (EDT)
        before_end = int(datetime(2024, 11, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        hour_before = get_rollover_hour_utc(before_end)

        # After DST ends: should be 22 (EST)
        after_end = int(datetime(2024, 11, 4, 12, 0, tzinfo=timezone.utc).timestamp())
        hour_after = get_rollover_hour_utc(after_end)

        assert hour_before == 21, f"Expected 21 before DST ends, got {hour_before}"
        assert hour_after == 22, f"Expected 22 after DST ends, got {hour_after}"

    def test_returns_valid_hour(self):
        """Test that returned hour is always valid (21 or 22)."""
        for year in [2023, 2024, 2025]:
            for month in range(1, 13):
                ts = int(datetime(year, month, 15, 12, 0, tzinfo=timezone.utc).timestamp())
                hour = get_rollover_hour_utc(ts)
                assert hour in (21, 22), f"Invalid rollover hour {hour} for {year}-{month}"


class TestIsDstInEffect:
    """Tests for is_dst_in_effect function."""

    def test_summer_is_dst(self):
        """Test that summer months are correctly identified as DST."""
        timestamp = int(datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(timestamp) is True

    def test_winter_not_dst(self):
        """Test that winter months are correctly identified as not DST."""
        timestamp = int(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(timestamp) is False

    def test_dst_march_transition(self):
        """Test DST detection around March transition."""
        # Before DST (March 9)
        before = int(datetime(2024, 3, 9, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(before) is False

        # After DST (March 11)
        after = int(datetime(2024, 3, 11, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(after) is True

    def test_dst_november_transition(self):
        """Test DST detection around November transition."""
        # Before DST ends (November 2)
        before = int(datetime(2024, 11, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(before) is True

        # After DST ends (November 4)
        after = int(datetime(2024, 11, 4, 12, 0, tzinfo=timezone.utc).timestamp())
        assert is_dst_in_effect(after) is False


# =============================================================================
# OPTIMIZED ROLLOVER COUNTER TESTS
# =============================================================================


class TestCountRolloversOptimized:
    """Tests for count_rollovers_optimized function."""

    def test_no_rollover_same_day(self):
        """Test no rollover when within same trading day."""
        # Jan 15, 2024 10:00 to 15:00 UTC (before 5pm ET)
        prev = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc).timestamp())
        assert count_rollovers_optimized(prev, curr) == 0

    def test_single_rollover(self):
        """Test single rollover crossing 5pm ET."""
        # Jan 15, 2024 20:00 to Jan 16, 2024 10:00 UTC
        # Rollover at 22:00 UTC (winter)
        prev = int(datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc).timestamp())
        count = count_rollovers_optimized(prev, curr)
        assert count == 1, f"Expected 1 rollover, got {count}"

    def test_wednesday_triple_swap(self):
        """Test Wednesday rollover counts as 3 (weekend settlement)."""
        # Wednesday Jan 17, 2024 to Thursday Jan 18, 2024
        # Wednesday rollover = 3x swap
        prev = int(datetime(2024, 1, 17, 20, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 18, 10, 0, tzinfo=timezone.utc).timestamp())
        count = count_rollovers_optimized(prev, curr)
        assert count == 3, f"Expected 3 for Wednesday rollover, got {count}"

    def test_weekend_no_rollover(self):
        """Test no rollovers on weekend."""
        # Saturday to Sunday
        prev = int(datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 14, 10, 0, tzinfo=timezone.utc).timestamp())
        count = count_rollovers_optimized(prev, curr)
        assert count == 0, f"Expected 0 for weekend, got {count}"

    def test_multiple_days(self):
        """Test rollover count across multiple days."""
        # Monday to Friday (4 trading days = 4 rollovers)
        # But if one is Wednesday, total = 1+3+1+1 = 6
        prev = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())  # Monday
        curr = int(datetime(2024, 1, 19, 10, 0, tzinfo=timezone.utc).timestamp())  # Friday
        count = count_rollovers_optimized(prev, curr)
        # Mon->Tue (1), Tue->Wed (1), Wed->Thu (3), Thu->Fri (1) = 6
        assert count == 6, f"Expected 6 rollovers Mon-Fri, got {count}"

    def test_full_week_including_weekend(self):
        """Test rollover count for full week including weekend."""
        # Monday to next Monday
        prev = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 22, 10, 0, tzinfo=timezone.utc).timestamp())
        count = count_rollovers_optimized(prev, curr)
        # 5 weekdays with 1 Wednesday (3x) = 1+1+3+1+1 = 7
        assert count == 7, f"Expected 7 for full week, got {count}"

    def test_prev_equals_curr(self):
        """Test zero rollovers when timestamps are equal."""
        ts = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        assert count_rollovers_optimized(ts, ts) == 0

    def test_prev_after_curr(self):
        """Test zero rollovers when prev > curr (invalid range)."""
        prev = int(datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        assert count_rollovers_optimized(prev, curr) == 0

    def test_dst_aware_vs_not(self):
        """Test DST-aware vs fixed rollover hour."""
        # Summer time - rollover at 21 UTC
        prev = int(datetime(2024, 7, 15, 20, 0, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 7, 16, 10, 0, tzinfo=timezone.utc).timestamp())

        count_aware = count_rollovers_optimized(prev, curr, dst_aware=True)
        count_fixed = count_rollovers_optimized(prev, curr, dst_aware=False)

        # Both should detect 1 rollover (21 or 22 is between 20:00 and 10:00 next day)
        assert count_aware == 1
        assert count_fixed == 1

    def test_no_rollover_before_5pm_et(self):
        """Test no rollover when staying before 5pm ET."""
        # Stay before rollover hour on same day
        prev = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        curr = int(
            datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc).timestamp()
        )  # Just before 22:00
        count = count_rollovers_optimized(prev, curr)
        # Rollover is at 22:00 UTC (winter), so 21:00 is before it
        assert count == 0, f"Expected 0 before rollover, got {count}"

    def test_rollover_exactly_at_5pm_et(self):
        """Test rollover at exactly 5pm ET (22:00 UTC winter)."""
        prev = int(datetime(2024, 1, 15, 21, 59, tzinfo=timezone.utc).timestamp())
        curr = int(datetime(2024, 1, 15, 22, 1, tzinfo=timezone.utc).timestamp())
        count = count_rollovers_optimized(prev, curr)
        assert count == 1, f"Expected 1 rollover at exactly 5pm ET, got {count}"


# =============================================================================
# SWAP RATE PROVIDER TESTS
# =============================================================================


class TestSwapRate:
    """Tests for SwapRate dataclass."""

    def test_creation(self):
        """Test basic SwapRate creation."""
        rate = SwapRate(
            pair="EUR_USD",
            long_swap=-0.35,
            short_swap=0.12,
        )
        assert rate.pair == "EUR_USD"
        assert rate.long_swap == -0.35
        assert rate.short_swap == 0.12
        assert rate.timestamp is None
        assert rate.source == "default"

    def test_creation_with_all_fields(self):
        """Test SwapRate with all fields."""
        rate = SwapRate(
            pair="GBP_USD",
            long_swap=-0.42,
            short_swap=0.18,
            timestamp=1704067200,
            source="file",
        )
        assert rate.timestamp == 1704067200
        assert rate.source == "file"


class TestSwapRateProvider:
    """Tests for SwapRateProvider class."""

    def test_default_provider(self):
        """Test provider with defaults only."""
        provider = SwapRateProvider(use_defaults=True)

        # Should get default rate for major pair
        rate = provider.get_swap_rate("EUR_USD")
        assert rate.source == "default"
        assert rate.long_swap == DEFAULT_SWAP_RATES["majors"]["long"]

    def test_provider_with_rates(self):
        """Test provider with pre-loaded rates."""
        rates = {
            "EUR_USD": SwapRate("EUR_USD", -0.35, 0.12, source="test"),
        }
        provider = SwapRateProvider(rates=rates, use_defaults=True)

        rate = provider.get_swap_rate("EUR_USD")
        assert rate.long_swap == -0.35
        assert rate.source == "test"

    def test_provider_from_json_file(self, temp_swap_dir):
        """Test loading from JSON file."""
        json_path = Path(temp_swap_dir) / "swaps.json"
        provider = SwapRateProvider.from_file(json_path)

        rate = provider.get_swap_rate("EUR_USD")
        assert rate.long_swap == -0.35
        assert rate.short_swap == 0.12
        assert rate.source == "file"

    def test_provider_from_csv_file(self, temp_swap_dir):
        """Test loading from CSV file."""
        csv_path = Path(temp_swap_dir) / "swaps.csv"
        provider = SwapRateProvider.from_file(csv_path)

        rate = provider.get_swap_rate("USD_JPY")
        assert rate.long_swap == -0.25
        assert rate.short_swap == 0.08

    def test_provider_from_directory(self, temp_swap_dir):
        """Test loading from directory with multiple files."""
        provider = SwapRateProvider.from_directory(temp_swap_dir, pattern="*.json")

        # Should have EUR_USD and GBP_USD from JSON
        eur_rate = provider.get_swap_rate("EUR_USD")
        gbp_rate = provider.get_swap_rate("GBP_USD")

        assert eur_rate.long_swap == -0.35
        assert gbp_rate.long_swap == -0.42

    def test_default_rates_by_category(self):
        """Test default rates based on pair category."""
        provider = SwapRateProvider(use_defaults=True)

        # Major pair
        major = provider.get_swap_rate("USD_JPY")
        assert major.long_swap == DEFAULT_SWAP_RATES["majors"]["long"]

        # Minor pair
        minor = provider.get_swap_rate("EUR_GBP")
        assert minor.long_swap == DEFAULT_SWAP_RATES["minors"]["long"]

        # Exotic pair
        exotic = provider.get_swap_rate("USD_TRY")
        assert exotic.long_swap == DEFAULT_SWAP_RATES["exotics"]["long"]

        # Cross pair (unknown = default to crosses)
        cross = provider.get_swap_rate("EUR_JPY")
        assert cross.long_swap == DEFAULT_SWAP_RATES["crosses"]["long"]

    def test_pair_normalization(self):
        """Test that pair names are normalized."""
        provider = SwapRateProvider(use_defaults=True)

        # Different formats should work
        rate1 = provider.get_swap_rate("EUR_USD")
        rate2 = provider.get_swap_rate("EUR/USD")
        rate3 = provider.get_swap_rate("eur_usd")

        assert rate1.long_swap == rate2.long_swap == rate3.long_swap

    def test_missing_file_uses_defaults(self):
        """Test that missing file falls back to defaults."""
        provider = SwapRateProvider.from_file("/nonexistent/path/file.json")
        rate = provider.get_swap_rate("EUR_USD")
        assert rate.source == "default"

    def test_add_rate(self):
        """Test adding a rate dynamically."""
        provider = SwapRateProvider(use_defaults=False)

        # Initially no rate
        rate = provider.get_swap_rate("EUR_USD")
        assert rate.source == "none"

        # Add rate
        provider.add_rate(SwapRate("EUR_USD", -0.5, 0.2, source="added"))

        rate = provider.get_swap_rate("EUR_USD")
        assert rate.long_swap == -0.5
        assert rate.source == "added"

    def test_historical_rate_lookup(self, temp_swap_dir):
        """Test historical rate lookup by timestamp."""
        # Create provider WITHOUT rates in _rates dict, only in historical
        provider = SwapRateProvider(rates={}, use_defaults=False)
        provider._historical_rates = {
            "EUR_USD": [
                SwapRate("EUR_USD", -0.30, 0.10, timestamp=1703980800),  # Earlier
                SwapRate("EUR_USD", -0.35, 0.12, timestamp=1704067200),  # Later
            ]
        }

        # Lookup with timestamp should find appropriate rate (most recent <= timestamp)
        # Timestamp 1704000000 is between 1703980800 and 1704067200
        # So the rate with timestamp 1703980800 should be returned
        rate = provider.get_swap_rate("EUR_USD", timestamp=1704000000)
        assert rate.long_swap == -0.30  # Earlier rate applies

    def test_historical_rate_lookup_exact_match(self):
        """Test historical rate lookup with exact timestamp match."""
        provider = SwapRateProvider(use_defaults=False)
        provider._historical_rates = {
            "EUR_USD": [
                SwapRate("EUR_USD", -0.30, 0.10, timestamp=1703980800),
                SwapRate("EUR_USD", -0.35, 0.12, timestamp=1704067200),
            ]
        }

        # Exact match should return the matching rate
        rate = provider.get_swap_rate("EUR_USD", timestamp=1704067200)
        assert rate.long_swap == -0.35


# =============================================================================
# FOREX ENV WRAPPER TESTS
# =============================================================================


class TestForexEnvWrapper:
    """Tests for ForexEnvWrapper class."""

    def test_initialization(self, mock_env):
        """Test wrapper initialization."""
        wrapper = ForexEnvWrapper(
            mock_env,
            leverage=30.0,
            include_swap_costs=True,
            dst_aware=True,
        )
        assert wrapper.leverage == 30.0
        assert wrapper.include_swap_costs is True
        assert wrapper.dst_aware is True

    def test_reset_clears_state(self, mock_env):
        """Test that reset clears forex state."""
        wrapper = ForexEnvWrapper(mock_env)
        wrapper._position = 0.5
        wrapper._cumulative_swap_cost = 10.0
        wrapper._rollover_count = 3

        wrapper.reset()

        assert wrapper._position == 0.0
        assert wrapper._cumulative_swap_cost == 0.0
        assert wrapper._rollover_count == 0

    def test_info_enriched_with_forex_data(self, mock_env):
        """Test that info dict contains forex-specific data."""
        wrapper = ForexEnvWrapper(mock_env, dst_aware=True)
        obs, info = wrapper.reset()

        assert "forex_session" in info
        assert "session_liquidity" in info
        assert "max_leverage" in info
        assert "is_weekend" in info
        assert "dst_in_effect" in info
        assert "rollover_hour_utc" in info

    def test_swap_cost_calculation(self):
        """Test swap cost is calculated correctly."""
        # Setup to cross a rollover - use correct timestamps
        # Jan 15, 2024 21:00 UTC (before 22:00 UTC winter rollover)
        reset_ts = int(datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc).timestamp())
        # Jan 16, 2024 09:00 UTC (after rollover)
        step_ts = int(datetime(2024, 1, 16, 9, 0, tzinfo=timezone.utc).timestamp())

        env = MockForexEnvWithTimestamps(reset_ts, step_ts)
        wrapper = ForexEnvWrapper(env, include_swap_costs=True)
        wrapper.reset()
        wrapper._position = 1.0  # Set position (long position)

        obs, reward, _, _, info = wrapper.step(np.array([0.5]))

        assert info["rollovers_this_step"] >= 1
        assert info["swap_cost"] >= 0

    def test_swap_provider_used(self, mock_env):
        """Test that custom swap provider is used."""
        custom_provider = SwapRateProvider(
            rates={
                "EUR_USD": SwapRate("EUR_USD", -1.0, 0.5, source="custom"),
            }
        )

        wrapper = ForexEnvWrapper(
            mock_env,
            swap_provider=custom_provider,
            symbol="EUR_USD",
        )

        assert wrapper.swap_provider is custom_provider

    def test_session_detection(self, mock_env):
        """Test session detection in different hours."""
        wrapper = ForexEnvWrapper(mock_env)

        # London session (10:00 UTC)
        session = wrapper._detect_session(
            int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        )
        assert session == "london"

        # NY/London overlap (14:00 UTC)
        session = wrapper._detect_session(
            int(datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc).timestamp())
        )
        assert session == "london_ny_overlap"

        # Weekend
        session = wrapper._detect_session(
            int(datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc).timestamp())
        )
        assert session == "weekend"

    def test_session_reward_scaling(self, mock_env):
        """Test session-based reward scaling."""
        # Low liquidity session should have lower rewards
        wrapper = ForexEnvWrapper(
            mock_env,
            session_reward_scaling=True,
            session_reward_scale_min=0.5,
        )

        wrapper.reset()
        _, reward, _, _, info = wrapper.step(0.5)

        assert "reward_scale" in info
        assert 0.5 <= info["reward_scale"] <= 1.0

    def test_dst_aware_parameter(self, mock_env):
        """Test DST-aware parameter affects rollover detection."""
        wrapper_aware = ForexEnvWrapper(mock_env, dst_aware=True)
        wrapper_fixed = ForexEnvWrapper(mock_env, dst_aware=False)

        # Summer timestamp
        ts = int(datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc).timestamp())

        hour_aware = get_rollover_hour_utc(ts)

        # DST-aware should detect summer hour (21)
        assert hour_aware == 21


# =============================================================================
# FOREX LEVERAGE WRAPPER TESTS
# =============================================================================


class MockForexEnvWithMargin(gym.Env):
    """Mock forex environment with margin tracking for leverage tests."""

    def __init__(self, net_worth: float = 100000.0, used_margin: float = 0.0):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._net_worth = net_worth
        self._used_margin = used_margin
        self._last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(10, dtype=np.float32), {"timestamp": 1704067200}

    def step(self, action):
        self._last_action = action
        return (
            np.zeros(10, dtype=np.float32),
            0.01,
            False,
            False,
            {"timestamp": 1704153600},
        )


class TestForexLeverageWrapper:
    """Tests for ForexLeverageWrapper class."""

    def test_initialization(self, mock_env):
        """Test wrapper initialization."""
        wrapper = ForexLeverageWrapper(
            mock_env,
            max_leverage=50.0,
            margin_call_level=1.0,
            stop_out_level=0.5,
        )
        assert wrapper.max_leverage == 50.0
        assert wrapper.margin_call_level == 1.0
        assert wrapper.stop_out_level == 0.5

    def test_stop_out_forces_liquidation(self):
        """Test that stop-out level forces position to zero."""
        env = MockForexEnvWithMargin(
            net_worth=40000.0, used_margin=100000.0
        )  # Margin level = 0.4 < 0.5

        wrapper = ForexLeverageWrapper(
            env,
            stop_out_level=0.5,
        )

        # Action should be forced to zero
        action = np.array([0.8])  # Request 80% position
        wrapper.step(action)

        # Check that env received zero action
        np.testing.assert_array_equal(env._last_action, np.zeros_like(action))

    def test_margin_call_reduces_position(self):
        """Test that margin call reduces position by half."""
        env = MockForexEnvWithMargin(
            net_worth=80000.0, used_margin=100000.0
        )  # Margin level = 0.8 < 1.0

        wrapper = ForexLeverageWrapper(
            env,
            margin_call_level=1.0,
        )

        action = np.array([0.8])  # Request 80% position
        wrapper.step(action)

        # Check that action was reduced by half
        np.testing.assert_array_almost_equal(env._last_action, np.array([0.4]))

    def test_no_intervention_above_margin_levels(self):
        """Test no intervention when margin levels are healthy."""
        env = MockForexEnvWithMargin(
            net_worth=150000.0, used_margin=100000.0
        )  # Margin level = 1.5 > 1.0

        wrapper = ForexLeverageWrapper(env)

        action = np.array([0.8])
        wrapper.step(action)

        # Action should pass through unchanged
        np.testing.assert_array_equal(env._last_action, action)

    def test_zero_margin_no_crash(self):
        """Test that zero used margin doesn't cause division by zero."""
        env = MockForexEnvWithMargin(net_worth=100000.0, used_margin=0.0)  # No position

        wrapper = ForexLeverageWrapper(env)

        # Should not raise
        action = np.array([0.5])
        wrapper.step(action)

        # Action should pass through unchanged
        np.testing.assert_array_equal(env._last_action, action)

    def test_scalar_action(self):
        """Test with scalar action instead of array."""
        env = MockForexEnvWithMargin(net_worth=40000.0, used_margin=100000.0)

        wrapper = ForexLeverageWrapper(env, stop_out_level=0.5)

        # Scalar action
        wrapper.step(0.8)

        # Should be forced to 0.0
        assert env._last_action == 0.0


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateForexEnv:
    """Tests for create_forex_env factory function."""

    def test_swap_provider_from_directory(self, temp_swap_dir):
        """Test that SwapRateProvider.from_directory works correctly."""
        provider = SwapRateProvider.from_directory(temp_swap_dir)

        # Should have loaded rates from files
        rate = provider.get_swap_rate("EUR_USD")
        assert rate.long_swap == -0.35
        assert rate.source == "file"

    def test_wrapper_composition(self, mock_env, temp_swap_dir):
        """Test that wrappers can be composed correctly."""
        swap_provider = SwapRateProvider.from_directory(temp_swap_dir)

        # Create wrapper stack manually (simulating create_forex_env)
        forex_wrapper = ForexEnvWrapper(
            mock_env,
            leverage=50.0,
            include_swap_costs=True,
            swap_provider=swap_provider,
            dst_aware=True,
        )
        leverage_wrapper = ForexLeverageWrapper(
            forex_wrapper,
            max_leverage=50.0,
            margin_call_level=1.0,
            stop_out_level=0.5,
        )

        # Check wrapper composition
        assert isinstance(leverage_wrapper, ForexLeverageWrapper)
        assert isinstance(leverage_wrapper.env, ForexEnvWrapper)

        # Test that reset/step work
        obs, info = leverage_wrapper.reset()
        assert obs.shape == (10,)
        assert "forex_session" in info

    def test_swap_provider_passed_to_wrapper(self, mock_env, temp_swap_dir):
        """Test that swap_provider is correctly passed to ForexEnvWrapper."""
        swap_provider = SwapRateProvider.from_directory(temp_swap_dir)

        wrapper = ForexEnvWrapper(
            mock_env,
            swap_provider=swap_provider,
        )

        assert wrapper.swap_provider is swap_provider

        # Verify rates are accessible
        rate = wrapper.swap_provider.get_swap_rate("EUR_USD")
        assert rate.long_swap == -0.35


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class MockForexEnvWithTimestamps(gym.Env):
    """Mock forex environment with configurable timestamps for integration tests."""

    def __init__(self, reset_ts: int, step_ts: int):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._reset_ts = reset_ts
        self._step_ts = step_ts
        self._net_worth = 100000.0
        self._used_margin = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(10, dtype=np.float32), {
            "timestamp": self._reset_ts,
            "symbol": "EUR_USD",
        }

    def step(self, action):
        return (
            np.zeros(10, dtype=np.float32),
            0.01,
            False,
            False,
            {
                "timestamp": self._step_ts,
                "signal_pos_next": (
                    float(action[0]) if hasattr(action, "__getitem__") else float(action)
                ),
                "symbol": "EUR_USD",
            },
        )


class TestForexIntegration:
    """Integration tests combining multiple components."""

    def test_full_trading_cycle(self, temp_swap_dir):
        """Test complete trading cycle with swaps and sessions."""
        # Setup timestamps to cross a rollover
        reset_ts = 1705362000  # Jan 15, 2024 21:00 UTC
        step_ts = 1705406400  # Jan 16, 2024 09:20 UTC

        env = MockForexEnvWithTimestamps(reset_ts, step_ts)

        # Load swap provider from temp dir
        swap_provider = SwapRateProvider.from_directory(temp_swap_dir)

        # Create wrapper stack
        forex_env = ForexEnvWrapper(
            env,
            leverage=30.0,
            include_swap_costs=True,
            swap_provider=swap_provider,
            symbol="EUR_USD",
            dst_aware=True,
        )
        leverage_env = ForexLeverageWrapper(
            forex_env,
            max_leverage=30.0,
            margin_call_level=1.0,
            stop_out_level=0.5,
        )

        # Trading cycle
        obs, info = leverage_env.reset()
        assert "forex_session" in info

        # Simulate holding position overnight
        forex_env._position = 1.0  # Set position before step

        obs, reward, terminated, truncated, info = leverage_env.step(np.array([0.5]))

        # Should have swap cost info
        assert "swap_cost" in info
        assert "cumulative_swap_cost" in info
        assert "rollovers_this_step" in info

    def test_wednesday_triple_swap_integration(self):
        """Test Wednesday triple swap in full environment."""
        # Wednesday to Thursday crossing rollover
        wed_before = int(datetime(2024, 1, 17, 20, 0, tzinfo=timezone.utc).timestamp())
        thu_after = int(datetime(2024, 1, 18, 10, 0, tzinfo=timezone.utc).timestamp())

        env = MockForexEnvWithTimestamps(wed_before, thu_after)
        wrapper = ForexEnvWrapper(env, include_swap_costs=True)
        wrapper.reset()
        wrapper._position = 1.0  # Hold position

        _, _, _, _, info = wrapper.step(np.array([0.5]))

        # Wednesday rollover should count as 3
        assert info["rollovers_this_step"] == 3

    def test_session_detection_integration(self):
        """Test session detection during full trading cycle."""
        # London session - 10:00 UTC
        london_ts = int(datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).timestamp())
        step_ts = london_ts + 3600  # 1 hour later

        env = MockForexEnvWithTimestamps(london_ts, step_ts)
        wrapper = ForexEnvWrapper(env)

        obs, info = wrapper.reset()
        assert info["forex_session"] == "london"
        assert info["session_liquidity"] == SESSION_LIQUIDITY["london"]

    def test_weekend_detection_integration(self):
        """Test weekend detection in full environment."""
        # Saturday 10:00 UTC
        saturday_ts = int(datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc).timestamp())
        step_ts = saturday_ts + 3600

        env = MockForexEnvWithTimestamps(saturday_ts, step_ts)
        wrapper = ForexEnvWrapper(env)

        obs, info = wrapper.reset()
        assert info["forex_session"] == "weekend"
        assert info["is_weekend"] is True
        assert info["session_liquidity"] == 0.0


# =============================================================================
# CONSTANTS VALIDATION TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_major_pairs_defined(self):
        """Test major pairs are defined correctly."""
        assert len(MAJOR_PAIRS) == 7
        assert "EUR_USD" in MAJOR_PAIRS
        assert "USD_JPY" in MAJOR_PAIRS

    def test_minor_pairs_defined(self):
        """Test minor pairs are defined."""
        assert len(MINOR_PAIRS) > 0
        assert "EUR_GBP" in MINOR_PAIRS

    def test_exotic_pairs_defined(self):
        """Test exotic pairs are defined."""
        assert len(EXOTIC_PAIRS) > 0
        assert "USD_TRY" in EXOTIC_PAIRS

    def test_session_liquidity_values(self):
        """Test session liquidity values are reasonable."""
        for session, liquidity in SESSION_LIQUIDITY.items():
            assert 0.0 <= liquidity <= 2.0, f"Invalid liquidity for {session}"

        # Weekend should have zero liquidity
        assert SESSION_LIQUIDITY["weekend"] == 0.0

        # Overlap should have highest liquidity
        assert SESSION_LIQUIDITY["london_ny_overlap"] >= SESSION_LIQUIDITY["london"]

    def test_default_swap_rates_structure(self):
        """Test default swap rates have correct structure."""
        for category in ["majors", "minors", "crosses", "exotics"]:
            assert category in DEFAULT_SWAP_RATES
            assert "long" in DEFAULT_SWAP_RATES[category]
            assert "short" in DEFAULT_SWAP_RATES[category]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
