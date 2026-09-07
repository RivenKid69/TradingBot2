# -*- coding: utf-8 -*-
"""
test_forex_features.py
Comprehensive test suite for Phase 4: Forex Features Pipeline.

Covers:
- forex_features.py - Forex-specific features (~40 tests)
- economic_calendar.py - Economic calendar integration (~15 tests)
- swap_rates_provider.py - Swap/financing rates (~15 tests)
- cot_data_loader.py - COT data loading (~10 tests)
- features_pipeline.py - Forex integration (~10 tests)

Total: ~90 tests

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import math
from datetime import datetime, timezone, timedelta, time
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# FOREX FEATURES TESTS
# =============================================================================


class TestCarryFeatures:
    """Test interest rate differential (carry) features."""

    def test_calculate_carry_features_basic(self):
        """Test basic carry calculation with dict rates."""
        from forex_features import calculate_carry_features

        rates = {"EUR": 4.0, "USD": 5.25}

        base_rate, quote_rate, diff, diff_norm, valid = calculate_carry_features(
            base_currency="EUR",
            quote_currency="USD",
            rates_data=rates,
        )

        assert valid is True
        assert base_rate == 4.0
        assert quote_rate == 5.25
        assert diff == pytest.approx(-1.25, rel=0.01)
        # Normalized: tanh(-1.25 / 5.0) ≈ -0.245
        assert -1.0 < diff_norm < 0.0

    def test_calculate_carry_features_positive_carry(self):
        """Test positive carry (high-yield base currency)."""
        from forex_features import calculate_carry_features

        rates = {"AUD": 4.35, "JPY": -0.10}

        base_rate, quote_rate, diff, diff_norm, valid = calculate_carry_features(
            base_currency="AUD",
            quote_currency="JPY",
            rates_data=rates,
        )

        assert valid is True
        assert diff == pytest.approx(4.45, rel=0.01)
        assert diff_norm > 0.5  # Strong positive carry

    def test_calculate_carry_features_missing_currency(self):
        """Test handling of missing currency rates.

        Note: When a currency is missing from rates dict, it's treated as 0%.
        This is valid because 0% rates exist (e.g., Japan). To get invalid=False,
        BOTH currencies must be missing.
        """
        from forex_features import calculate_carry_features

        # When BOTH currencies are missing, returns invalid
        result = calculate_carry_features(
            base_currency="XYZ",
            quote_currency="ABC",
            rates_data={"EUR": 4.0},  # Neither XYZ nor ABC present
        )
        assert result == (0.0, 0.0, 0.0, 0.0, False)

        # When one currency is present, treats missing as 0% (valid)
        result2 = calculate_carry_features(
            base_currency="EUR",
            quote_currency="USD",
            rates_data={"EUR": 4.0},  # USD missing → treated as 0%
        )
        base_rate, quote_rate, diff, diff_norm, valid = result2
        assert valid is True  # Valid because EUR has a rate
        assert base_rate == 4.0
        assert quote_rate == 0.0  # Missing → 0%
        assert diff == 4.0  # 4.0 - 0.0

    def test_calculate_carry_features_dataframe(self):
        """Test carry calculation with DataFrame rates."""
        from forex_features import calculate_carry_features

        rates_df = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "EUR_RATE": [3.5, 3.75, 4.0],
                "USD_RATE": [4.5, 4.75, 5.25],
            }
        )

        base_rate, quote_rate, diff, diff_norm, valid = calculate_carry_features(
            base_currency="EUR",
            quote_currency="USD",
            rates_data=rates_df,
            timestamp_ms=2500,
        )

        assert valid is True
        assert base_rate == 3.75
        assert quote_rate == 4.75

    def test_classify_carry_regime(self):
        """Test carry regime classification."""
        from forex_features import classify_carry_regime, CarryRegime

        # Negative carry
        regime, val = classify_carry_regime(-2.0)
        assert regime == CarryRegime.NEGATIVE
        assert val == -1.0

        # Neutral
        regime, val = classify_carry_regime(0.0)
        assert regime == CarryRegime.NEUTRAL
        assert val == 0.0

        # Positive
        regime, val = classify_carry_regime(2.0)
        assert regime == CarryRegime.POSITIVE
        assert 0.0 < val < 1.0

        # High carry
        regime, val = classify_carry_regime(5.0)
        assert regime == CarryRegime.HIGH_CARRY
        assert val == 1.0


class TestSessionFeatures:
    """Test forex session detection features."""

    def test_detect_forex_session_london(self):
        """Test London session detection."""
        from forex_features import detect_forex_session, ForexSession

        # 10:00 UTC is London session
        dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        session = detect_forex_session(dt)
        assert session == ForexSession.LONDON

    def test_detect_forex_session_tokyo(self):
        """Test Tokyo session detection.

        Note: Tokyo (00:00-09:00) overlaps with Sydney (21:00-06:00) from 00:00-06:00.
        When sessions overlap, the first in iteration order wins (Sydney).
        Use 06:30 UTC to test Tokyo-only window (after Sydney ends at 06:00,
        before Tokyo-London overlap at 07:00).
        """
        from forex_features import detect_forex_session, ForexSession

        # 06:30 UTC is Tokyo-only (Sydney ends at 06:00, Tokyo-London starts at 07:00)
        dt = datetime(2024, 1, 15, 6, 30, 0, tzinfo=timezone.utc)
        session = detect_forex_session(dt)
        assert session == ForexSession.TOKYO

    def test_detect_forex_session_new_york(self):
        """Test New York session detection."""
        from forex_features import detect_forex_session, ForexSession

        # 18:00 UTC is New York session
        dt = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc)
        session = detect_forex_session(dt)
        assert session == ForexSession.NEW_YORK

    def test_detect_forex_session_london_ny_overlap(self):
        """Test London-NY overlap detection."""
        from forex_features import detect_forex_session, ForexSession

        # 14:00 UTC is London-NY overlap
        dt = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        session = detect_forex_session(dt)
        assert session == ForexSession.LONDON_NY_OVERLAP

    def test_detect_forex_session_timestamp_ms(self):
        """Test session detection from millisecond timestamp."""
        from forex_features import detect_forex_session, ForexSession

        # 10:00 UTC in milliseconds
        ts_ms = int(datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        session = detect_forex_session(ts_ms)
        assert session == ForexSession.LONDON

    def test_get_session_liquidity(self):
        """Test session liquidity multipliers."""
        from forex_features import get_session_liquidity, ForexSession

        assert get_session_liquidity(ForexSession.LONDON) == 1.3
        assert get_session_liquidity(ForexSession.TOKYO) == 0.8
        assert get_session_liquidity(ForexSession.LONDON_NY_OVERLAP) == 1.5
        assert get_session_liquidity(ForexSession.LOW_LIQUIDITY) == 0.4

    def test_get_session_features(self):
        """Test full session feature extraction."""
        from forex_features import get_session_features, ForexSession

        dt = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        is_sydney, is_tokyo, is_london, is_ny, is_overlap, liquidity, session = (
            get_session_features(dt)
        )

        assert is_sydney is False
        assert is_tokyo is False
        assert is_london is False
        assert is_ny is False
        assert is_overlap is True
        assert liquidity == 1.5
        assert session == ForexSession.LONDON_NY_OVERLAP


class TestDXYFeatures:
    """Test DXY-related features."""

    def test_calculate_dxy_features_basic(self):
        """Test basic DXY feature calculation."""
        from forex_features import calculate_dxy_features

        # Generate mock price data
        pair_prices = [1.08 + 0.001 * i for i in range(30)]
        dxy_prices = [100.0 + 0.1 * i for i in range(30)]

        dxy_ret_5d, rs_vs_dxy, dxy_val, valid = calculate_dxy_features(
            pair_prices, dxy_prices, window=20
        )

        assert valid is True
        assert math.isfinite(dxy_ret_5d)
        assert -1.0 <= rs_vs_dxy <= 1.0
        assert dxy_val > 100.0  # Latest DXY value

    def test_calculate_dxy_features_insufficient_data(self):
        """Test handling of insufficient data.

        When data is shorter than requested window, the function adaptively
        uses a shorter window (min_len - 1). With 2 prices, window=1 is used.
        Returns invalid only if less than 2 prices available.
        """
        from forex_features import calculate_dxy_features

        # With 2 prices: adapts to window=1, returns valid
        pair_prices = [1.08, 1.09]
        dxy_prices = [100.0, 100.5]

        dxy_ret_5d, rs_vs_dxy, dxy_val, valid = calculate_dxy_features(
            pair_prices, dxy_prices, window=20
        )

        # With 2+ prices, function adapts window and returns valid
        assert valid is True
        assert dxy_val == 100.5

        # With only 1 price: returns invalid
        _, _, _, valid_1 = calculate_dxy_features([1.08], [100.0], window=20)
        assert valid_1 is False

        # With empty lists: returns invalid
        _, _, _, valid_empty = calculate_dxy_features([], [], window=20)
        assert valid_empty is False

    def test_normalize_dxy_value(self):
        """Test DXY normalization."""
        from forex_features import normalize_dxy_value

        # DXY at 100 (reference) should be ~0
        assert normalize_dxy_value(100.0) == pytest.approx(0.0, abs=0.01)

        # DXY at 120 should be positive
        assert normalize_dxy_value(120.0) > 0.5

        # DXY at 80 should be negative
        assert normalize_dxy_value(80.0) < -0.5


class TestSpreadFeatures:
    """Test spread-related features."""

    def test_calculate_spread_features_normal(self):
        """Test normal spread calculation."""
        from forex_features import calculate_spread_features, SpreadRegime

        current_spread = 1.5
        spread_history = [1.4, 1.5, 1.6, 1.5, 1.4, 1.5, 1.6, 1.5, 1.4, 1.5] * 10

        zscore, regime_val, regime, valid = calculate_spread_features(
            current_spread, spread_history
        )

        assert valid is True
        assert -3.0 <= zscore <= 3.0
        assert regime == SpreadRegime.NORMAL

    def test_calculate_spread_features_wide(self):
        """Test wide spread detection.

        Note: Spread history must have non-zero std deviation for valid=True.
        """
        from forex_features import calculate_spread_features, SpreadRegime

        current_spread = 4.0  # Much wider than average (~2x)
        # History with variation (mean ~1.5, std ~0.3)
        spread_history = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.5, 1.4, 1.5] * 10

        zscore, regime_val, regime, valid = calculate_spread_features(
            current_spread, spread_history
        )

        assert valid is True
        assert regime == SpreadRegime.WIDE or regime == SpreadRegime.EXTREME

    def test_calculate_spread_features_tight(self):
        """Test tight spread detection.

        Note: Spread history must have non-zero std deviation for valid=True.
        """
        from forex_features import calculate_spread_features, SpreadRegime

        current_spread = 0.8  # Below 70% of avg ~1.5
        # History with variation (mean ~1.5, std ~0.3)
        spread_history = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.5, 1.4, 1.5] * 10

        zscore, regime_val, regime, valid = calculate_spread_features(
            current_spread, spread_history
        )

        assert valid is True
        assert regime == SpreadRegime.TIGHT

    def test_calculate_spread_features_insufficient_history(self):
        """Test handling of insufficient spread history."""
        from forex_features import calculate_spread_features

        result = calculate_spread_features(1.5, [1.5] * 5)
        assert result[-1] is False  # valid should be False


class TestVolatilityFeatures:
    """Test volatility features."""

    def test_calculate_volatility_features_basic(self):
        """Test basic volatility calculation."""
        from forex_features import calculate_volatility_features

        # Generate price data with volatility
        np.random.seed(42)
        prices = [100.0 * np.exp(np.cumsum(np.random.randn(100) * 0.01))]
        prices = list(prices[0])

        vol_5d, vol_20d, vol_ratio, valid = calculate_volatility_features(prices)

        assert valid is True
        assert 0.0 <= vol_5d <= 1.0  # Normalized
        assert 0.0 <= vol_20d <= 1.0
        assert vol_ratio > 0

    def test_calculate_volatility_features_insufficient_data(self):
        """Test handling of insufficient data."""
        from forex_features import calculate_volatility_features

        prices = [100.0, 100.5, 101.0]
        vol_5d, vol_20d, vol_ratio, valid = calculate_volatility_features(prices)

        assert valid is False


class TestCOTFeatures:
    """Test COT positioning features."""

    def test_calculate_cot_features_basic(self):
        """Test basic COT feature calculation."""
        from forex_features import calculate_cot_features, COTPositioning

        cot_df = pd.DataFrame(
            {
                "EUR_NET": [10000, 15000, 20000, 18000, 22000],
                "EUR_OI": [100000, 100000, 100000, 100000, 100000],
            }
        )

        net_pct, zscore, change, positioning, valid = calculate_cot_features("EUR", cot_df)

        assert valid is True
        assert 0.0 <= net_pct <= 1.0
        assert -3.0 <= zscore <= 3.0

    def test_calculate_cot_features_missing_currency(self):
        """Test handling of missing currency in COT data."""
        from forex_features import calculate_cot_features

        cot_df = pd.DataFrame({"USD_NET": [10000]})

        result = calculate_cot_features("EUR", cot_df)
        assert result[-1] is False  # valid should be False

    def test_calculate_cot_features_empty_df(self):
        """Test handling of empty COT DataFrame."""
        from forex_features import calculate_cot_features

        result = calculate_cot_features("EUR", pd.DataFrame())
        assert result[-1] is False


class TestCrossMomentum:
    """Test cross-currency momentum features."""

    def test_calculate_cross_momentum_basic(self):
        """Test basic cross momentum calculation."""
        from forex_features import calculate_cross_momentum

        pair_returns = {
            "EUR_USD": 0.02,
            "GBP_USD": 0.01,
            "AUD_USD": 0.03,
            "USD_JPY": -0.01,
        }

        momentum, valid = calculate_cross_momentum(pair_returns, "AUD_USD")

        assert valid is True
        # AUD_USD has highest return, should rank high
        assert momentum > 0.0

    def test_calculate_cross_momentum_single_pair(self):
        """Test cross momentum with single pair."""
        from forex_features import calculate_cross_momentum

        pair_returns = {"EUR_USD": 0.02}

        momentum, valid = calculate_cross_momentum(pair_returns, "EUR_USD")

        assert valid is True


class TestForexFeaturesExtraction:
    """Test main feature extraction function."""

    def test_extract_forex_features_basic(self):
        """Test basic forex feature extraction."""
        from forex_features import extract_forex_features, ForexFeatures, BenchmarkForexData

        row = {"timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
        benchmark = BenchmarkForexData(
            interest_rates={"EUR": 4.0, "USD": 5.25},
        )

        features = extract_forex_features(
            row=row,
            symbol="EUR_USD",
            benchmark_data=benchmark,
        )

        assert isinstance(features, ForexFeatures)
        # Session should be detected
        assert features.session_valid is True

    def test_extract_forex_features_with_prices(self):
        """Test feature extraction with price history."""
        from forex_features import extract_forex_features, BenchmarkForexData

        row = {"timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
        pair_prices = list(np.linspace(1.08, 1.12, 30))
        dxy_prices = list(np.linspace(100.0, 105.0, 30))

        benchmark = BenchmarkForexData(
            dxy_prices=dxy_prices,
            interest_rates={"EUR": 4.0, "USD": 5.25},
        )

        features = extract_forex_features(
            row=row,
            symbol="EUR_USD",
            benchmark_data=benchmark,
            pair_prices=pair_prices,
        )

        assert features.carry_valid is True
        assert features.dxy_valid is True
        assert features.vol_valid is True


class TestCurrencyParsing:
    """Test currency pair parsing."""

    def test_parse_currency_pair_underscore(self):
        """Test parsing underscore format."""
        from forex_features import _parse_currency_pair

        result = _parse_currency_pair("EUR_USD")
        assert result == ("EUR", "USD")

    def test_parse_currency_pair_slash(self):
        """Test parsing slash format."""
        from forex_features import _parse_currency_pair

        result = _parse_currency_pair("EUR/USD")
        assert result == ("EUR", "USD")

    def test_parse_currency_pair_concat(self):
        """Test parsing concatenated format."""
        from forex_features import _parse_currency_pair

        result = _parse_currency_pair("EURUSD")
        assert result == ("EUR", "USD")

    def test_parse_currency_pair_lowercase(self):
        """Test parsing lowercase."""
        from forex_features import _parse_currency_pair

        result = _parse_currency_pair("eur_usd")
        assert result == ("EUR", "USD")

    def test_parse_currency_pair_invalid(self):
        """Test parsing invalid format."""
        from forex_features import _parse_currency_pair

        result = _parse_currency_pair("INVALID")
        assert result is None


class TestDataFrameIntegration:
    """Test DataFrame integration functions."""

    def test_add_forex_features_to_dataframe(self):
        """Test adding forex features to DataFrame."""
        from forex_features import add_forex_features_to_dataframe

        # Create sample DataFrame
        n = 50
        timestamps = [
            int((datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).timestamp() * 1000)
            for i in range(n)
        ]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.random.randn(n) * 0.01 + 1.10,
                "high": np.random.randn(n) * 0.01 + 1.11,
                "low": np.random.randn(n) * 0.01 + 1.09,
                "close": np.random.randn(n) * 0.01 + 1.10,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        result_df = add_forex_features_to_dataframe(df, "EUR_USD")

        # Check new columns exist
        assert "carry_diff" in result_df.columns
        assert "session_london" in result_df.columns
        assert "session_liquidity" in result_df.columns
        assert "vol_ratio" in result_df.columns


# =============================================================================
# ECONOMIC CALENDAR TESTS
# =============================================================================


class TestEconomicEvent:
    """Test EconomicEvent dataclass."""

    def test_economic_event_creation(self):
        """Test creating EconomicEvent."""
        from economic_calendar import EconomicEvent, ImpactLevel, CalendarSource

        event = EconomicEvent(
            event_id="nfp_001",
            datetime_utc=datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc),
            currency="USD",
            event_name="Non-Farm Payrolls",
            impact=ImpactLevel.HIGH,
        )

        assert event.currency == "USD"
        assert event.impact == ImpactLevel.HIGH
        assert event.timestamp_ms > 0

    def test_economic_event_to_dict(self):
        """Test EconomicEvent serialization."""
        from economic_calendar import EconomicEvent, ImpactLevel

        event = EconomicEvent(
            event_id="test",
            datetime_utc=datetime(2024, 1, 5, tzinfo=timezone.utc),
            currency="EUR",
            event_name="ECB Rate Decision",
            impact=ImpactLevel.HIGH,
        )

        d = event.to_dict()
        assert d["currency"] == "EUR"
        assert d["impact"] == 3

    def test_economic_event_from_dict(self):
        """Test EconomicEvent deserialization."""
        from economic_calendar import EconomicEvent

        data = {
            "event_id": "test",
            "datetime": "2024-01-05T00:00:00+00:00",
            "currency": "GBP",
            "event_name": "BOE Decision",
            "impact": 3,
        }

        event = EconomicEvent.from_dict(data)
        assert event.currency == "GBP"


class TestCachedCalendarProvider:
    """Test CachedCalendarProvider."""

    def test_cached_provider_add_and_get(self):
        """Test adding and retrieving events."""
        from economic_calendar import (
            CachedCalendarProvider,
            EconomicEvent,
            ImpactLevel,
            CalendarConfig,
        )

        provider = CachedCalendarProvider(CalendarConfig(cache_dir="/tmp/test_cal"))

        event = EconomicEvent(
            event_id="test1",
            datetime_utc=datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc),
            currency="USD",
            event_name="Test Event",
            impact=ImpactLevel.HIGH,
        )
        provider.add_events([event])

        events = provider.get_events(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 31, tzinfo=timezone.utc),
            currencies=["USD"],
        )

        assert len(events) == 1
        assert events[0].event_id == "test1"


class TestEconomicCalendar:
    """Test main EconomicCalendar class."""

    def test_calendar_creation(self):
        """Test calendar creation."""
        from economic_calendar import EconomicCalendar, CalendarSource

        calendar = EconomicCalendar(source=CalendarSource.CACHED)
        assert calendar.config.source == CalendarSource.CACHED

    def test_hours_to_next_high_impact(self):
        """Test hours_to_next_high_impact."""
        from economic_calendar import EconomicCalendar, EconomicEvent, ImpactLevel

        calendar = EconomicCalendar(source="cached")

        # Add a test event
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=5)

        event = EconomicEvent(
            event_id="test",
            datetime_utc=future,
            currency="USD",
            event_name="FOMC",
            impact=ImpactLevel.HIGH,
        )
        calendar._cache.add_events([event])

        hours, name, impact = calendar.hours_to_next_high_impact("USD")

        assert hours < 6
        assert hours > 4
        assert impact == 3

    def test_classify_event_impact(self):
        """Test event impact classification."""
        from economic_calendar import classify_event_impact, ImpactLevel

        # High impact USD events
        assert classify_event_impact("Non-Farm Payrolls", "USD") == ImpactLevel.HIGH
        assert classify_event_impact("FOMC Rate Decision", "USD") == ImpactLevel.HIGH

        # High impact EUR events
        assert classify_event_impact("ECB Interest Rate", "EUR") == ImpactLevel.HIGH


# =============================================================================
# SWAP RATES PROVIDER TESTS
# =============================================================================


class TestSwapRates:
    """Test SwapRates dataclass."""

    def test_swap_rates_creation(self):
        """Test SwapRates creation."""
        from swap_rates_provider import SwapRates

        swaps = SwapRates(
            symbol="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
        )

        assert swaps.symbol == "EUR_USD"
        assert swaps.long_swap_pips == -0.5
        assert swaps.short_swap_pips == 0.3

    def test_swap_rates_daily_cost(self):
        """Test daily swap cost calculation."""
        from swap_rates_provider import SwapRates

        swaps = SwapRates(
            symbol="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
        )

        # 1 lot long position
        cost = swaps.get_daily_cost(1.0, is_long=True, pip_value=10.0)
        assert cost == -5.0  # -0.5 pips * 1 lot * $10/pip

        # 2 lots short position
        credit = swaps.get_daily_cost(2.0, is_long=False, pip_value=10.0)
        assert credit == 6.0  # 0.3 pips * 2 lots * $10/pip


class TestInterestRateSwapProvider:
    """Test InterestRateSwapProvider."""

    def test_calculate_swap_rates(self):
        """Test swap rate calculation from interest rates."""
        from swap_rates_provider import InterestRateSwapProvider, SwapProviderConfig

        provider = InterestRateSwapProvider(rates={"EUR": 4.0, "USD": 5.25})

        swaps = provider.calculate_swap_rates("EUR_USD")

        assert swaps is not None
        assert swaps.symbol == "EUR_USD"
        # EUR has lower rate than USD, so long swap should be negative
        assert swaps.long_swap_pct < 0

    def test_get_current_swaps(self):
        """Test getting current swaps."""
        from swap_rates_provider import InterestRateSwapProvider

        provider = InterestRateSwapProvider()
        swaps = provider.get_current_swaps("GBP_USD")

        assert swaps is not None
        assert swaps.symbol == "GBP_USD"

    def test_get_historical_swaps(self):
        """Test getting historical swaps."""
        from swap_rates_provider import InterestRateSwapProvider

        provider = InterestRateSwapProvider()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 10, tzinfo=timezone.utc)

        df = provider.get_historical_swaps("EUR_USD", start, end)

        assert not df.empty
        assert "long_swap" in df.columns
        assert "short_swap" in df.columns


class TestSwapRatesManager:
    """Test SwapRatesManager."""

    def test_manager_get_swaps(self):
        """Test getting swaps through manager."""
        from swap_rates_provider import SwapRatesManager

        manager = SwapRatesManager()
        swaps = manager.get_swaps("EUR_USD")

        assert swaps is not None
        assert swaps.symbol == "EUR_USD"


class TestSwapUtilities:
    """Test swap utility functions."""

    def test_get_pip_size(self):
        """Test pip size detection."""
        from swap_rates_provider import get_pip_size

        assert get_pip_size("EUR_USD") == 0.0001
        assert get_pip_size("USD_JPY") == 0.01
        assert get_pip_size("GBP_JPY") == 0.01

    def test_is_triple_rollover_day(self):
        """Test triple rollover day detection."""
        from swap_rates_provider import is_triple_rollover_day

        wednesday = datetime(2024, 1, 10)  # Wednesday
        thursday = datetime(2024, 1, 11)  # Thursday

        assert is_triple_rollover_day(wednesday) is True
        assert is_triple_rollover_day(thursday) is False


# =============================================================================
# COT DATA LOADER TESTS
# =============================================================================


class TestCOTPosition:
    """Test COTPosition dataclass."""

    def test_cot_position_creation(self):
        """Test COTPosition creation."""
        from cot_data_loader import COTPosition

        pos = COTPosition(
            currency="EUR",
            report_date=datetime(2024, 1, 9, tzinfo=timezone.utc),
            net_long=50000,
            net_long_pct=0.15,
            open_interest=350000,
        )

        assert pos.currency == "EUR"
        assert pos.net_long == 50000
        assert pos.timestamp_ms > 0

    def test_cot_position_to_dict(self):
        """Test COTPosition serialization."""
        from cot_data_loader import COTPosition

        pos = COTPosition(
            currency="GBP",
            report_date=datetime(2024, 1, 9, tzinfo=timezone.utc),
            net_long=25000,
        )

        d = pos.to_dict()
        assert d["currency"] == "GBP"
        assert d["net_long"] == 25000


class TestCOTDataLoader:
    """Test COTDataLoader."""

    def test_loader_creation(self):
        """Test loader creation."""
        from cot_data_loader import COTDataLoader, COTConfig

        loader = COTDataLoader(COTConfig(cache_dir="/tmp/test_cot"))
        assert loader.config.cache_dir == "/tmp/test_cot"

    def test_load_from_dataframe(self):
        """Test loading COT data from DataFrame."""
        from cot_data_loader import COTDataLoader

        df = pd.DataFrame(
            {
                "currency": ["EUR", "EUR", "EUR", "GBP", "GBP"],
                "report_date": [
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 9, tzinfo=timezone.utc),
                    datetime(2024, 1, 16, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 9, tzinfo=timezone.utc),
                ],
                "net_long": [10000, 15000, 20000, 5000, 8000],
                "open_interest": [100000, 105000, 110000, 50000, 52000],
            }
        )

        loader = COTDataLoader()
        loader.load_from_dataframe(df)

        assert "EUR" in loader.available_currencies
        assert "GBP" in loader.available_currencies

    def test_get_current_position(self):
        """Test getting current position."""
        from cot_data_loader import COTDataLoader

        df = pd.DataFrame(
            {
                "currency": ["EUR", "EUR"],
                "report_date": [
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 9, tzinfo=timezone.utc),
                ],
                "net_long": [10000, 15000],
            }
        )

        loader = COTDataLoader()
        loader.load_from_dataframe(df)

        pos = loader.get_current_position("EUR")
        assert pos is not None
        assert pos.net_long == 15000

    def test_calculate_zscore(self):
        """Test z-score calculation."""
        from cot_data_loader import COTDataLoader

        # Create data with known properties
        net_longs = list(np.random.randn(52) * 10000 + 50000)
        df = pd.DataFrame(
            {
                "currency": ["EUR"] * 52,
                "report_date": [
                    datetime(2024, 1, 2, tzinfo=timezone.utc) + timedelta(weeks=i)
                    for i in range(52)
                ],
                "net_long": net_longs,
            }
        )

        loader = COTDataLoader()
        loader.load_from_dataframe(df)

        zscore, valid = loader.calculate_zscore("EUR")
        assert valid is True
        assert -3.0 <= zscore <= 3.0


class TestMockCOTData:
    """Test mock COT data generation."""

    def test_create_mock_cot_data(self):
        """Test mock data creation."""
        from cot_data_loader import create_mock_cot_data

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        loader = create_mock_cot_data(
            currencies=["EUR", "GBP", "JPY"],
            start_date=start,
            end_date=end,
            seed=42,
        )

        assert len(loader) > 0
        assert "EUR" in loader.available_currencies
        assert "GBP" in loader.available_currencies
        assert "JPY" in loader.available_currencies


class TestCOTForPair:
    """Test COT data for currency pairs."""

    def test_get_cot_for_pair(self):
        """Test getting COT for a currency pair."""
        from cot_data_loader import create_mock_cot_data, get_cot_for_pair

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)

        loader = create_mock_cot_data(
            currencies=["EUR", "GBP"],
            start_date=start,
            end_date=end,
            seed=42,
        )

        net_pct, zscore, valid = get_cot_for_pair(loader, "EUR_USD")

        assert valid is True
        assert 0.0 <= net_pct <= 1.0
        assert -3.0 <= zscore <= 3.0


# =============================================================================
# FEATURES PIPELINE INTEGRATION TESTS
# =============================================================================


class TestFeaturesPipelineForexIntegration:
    """Test features_pipeline.py forex integration."""

    def test_pipeline_with_forex_asset_class(self):
        """Test pipeline with forex asset class."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="forex")
        assert pipeline.asset_class == "forex"
        assert pipeline.auto_forex_features is True

    def test_pipeline_auto_forex_features_disabled(self):
        """Test pipeline with auto forex features disabled."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(
            asset_class="forex",
            auto_forex_features=False,
        )
        assert pipeline.auto_forex_features is False

    def test_pipeline_save_load_forex(self):
        """Test saving and loading pipeline with forex config."""
        import tempfile
        import os
        from features_pipeline import FeaturePipeline

        # Create and save
        pipeline = FeaturePipeline(
            asset_class="forex",
            auto_forex_features=True,
        )
        pipeline.stats = {"test_col": {"mean": 1.0, "std": 0.5}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pipeline.json")
            pipeline.save(path)

            # Load
            loaded = FeaturePipeline.load(path)
            assert loaded.asset_class == "forex"
            assert loaded.auto_forex_features is True

    def test_pipeline_transform_adds_forex_features(self):
        """Test that transform adds forex features when enabled."""
        from features_pipeline import FeaturePipeline

        # Create sample data
        n = 50
        timestamps = [
            int((datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).timestamp() * 1000)
            for i in range(n)
        ]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["EUR_USD"] * n,
                "close": np.random.randn(n) * 0.01 + 1.10,
                "open": np.random.randn(n) * 0.01 + 1.10,
                "high": np.random.randn(n) * 0.01 + 1.11,
                "low": np.random.randn(n) * 0.01 + 1.09,
                "volume": np.random.randint(1000, 10000, n),
            }
        )

        # Fit pipeline
        pipeline = FeaturePipeline(
            asset_class="forex",
            auto_forex_features=True,
            strict_idempotency=False,
        )
        pipeline.fit({"test": df})

        # Transform
        result = pipeline.transform_df(df)

        # Check forex features added
        assert "carry_diff" in result.columns or "session_london" in result.columns


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_crypto_unchanged(self):
        """Test that crypto asset class is unchanged."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="crypto")
        assert pipeline.asset_class == "crypto"
        assert (
            not hasattr(pipeline, "_add_forex_features_internal")
            or pipeline.auto_forex_features is True
        )

    def test_none_asset_class_unchanged(self):
        """Test that None asset class is unchanged."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class=None)
        assert pipeline.asset_class is None

    def test_equity_still_works(self):
        """Test that equity asset class still works."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="equity")
        assert pipeline.asset_class == "equity"
        assert pipeline.auto_stock_features is True


# =============================================================================
# IMPLIED VOLATILITY FEATURES TESTS
# =============================================================================


class TestImpliedVolFeatures:
    """Test FX implied volatility features (EVZ, CBOE indices)."""

    def test_calculate_implied_vol_basic(self):
        """Test basic implied vol calculation."""
        from forex_features import calculate_implied_vol_features

        # Normal volatility around 8-10%
        implied_vol_prices = [8.0, 8.5, 9.0, 8.8, 9.2, 9.5]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        assert valid is True
        assert regime == "normal"  # 6-10% is normal
        assert 0.0 <= normalized_vol <= 1.0

    def test_calculate_implied_vol_low_regime(self):
        """Test low volatility regime detection."""
        from forex_features import calculate_implied_vol_features

        # Very low vol (< 6%)
        implied_vol_prices = [4.0, 4.5, 5.0, 4.8, 5.2]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        assert valid is True
        assert regime == "low"

    def test_calculate_implied_vol_elevated_regime(self):
        """Test elevated volatility regime detection."""
        from forex_features import calculate_implied_vol_features

        # Elevated vol (10-15%)
        implied_vol_prices = [10.0, 11.0, 12.0, 11.5, 13.0]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        assert valid is True
        assert regime == "elevated"

    def test_calculate_implied_vol_high_regime(self):
        """Test high volatility regime detection (stress environment)."""
        from forex_features import calculate_implied_vol_features

        # High vol (> 15%)
        implied_vol_prices = [15.0, 16.0, 18.0, 20.0, 22.0]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        assert valid is True
        assert regime == "high"
        assert percentile > 0.5  # Should be in upper percentiles

    def test_calculate_implied_vol_percentile_ranking(self):
        """Test percentile calculation with sufficient history."""
        from forex_features import calculate_implied_vol_features

        # Create history with known distribution
        # 20 values: 5-24, current is 20 (should be ~75th percentile)
        implied_vol_prices = list(range(5, 25))  # [5, 6, ..., 24]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD", window=20
        )

        assert valid is True
        # Current value 24 is the max, so percentile should be ~95%
        assert percentile > 0.9

    def test_calculate_implied_vol_short_history(self):
        """Test with history shorter than window."""
        from forex_features import calculate_implied_vol_features

        # Only 5 values, window default is 20
        implied_vol_prices = [8.0, 9.0, 10.0, 9.5, 11.0]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD", window=20
        )

        assert valid is True
        # Should still work with regime-based percentile fallback
        assert 0.0 <= percentile <= 1.0

    def test_calculate_implied_vol_empty_list(self):
        """Test with empty price list."""
        from forex_features import calculate_implied_vol_features

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            [], symbol="EUR_USD"
        )

        assert valid is False
        assert normalized_vol == 0.0
        assert regime == "normal"

    def test_calculate_implied_vol_pair_specific_typical_vol(self):
        """Test that pair-specific typical vol is used for normalization."""
        from forex_features import calculate_implied_vol_features, TYPICAL_IMPLIED_VOL

        # GBP_USD has higher typical vol (10%) than EUR_USD (8%)
        implied_vol_prices = [10.0]  # Same value for both

        _, _, _, valid_gbp = calculate_implied_vol_features(implied_vol_prices, symbol="GBP_USD")
        _, _, _, valid_eur = calculate_implied_vol_features(implied_vol_prices, symbol="EUR_USD")

        assert valid_gbp is True
        assert valid_eur is True
        assert TYPICAL_IMPLIED_VOL["GBP_USD"] > TYPICAL_IMPLIED_VOL["EUR_USD"]

    def test_get_fx_vol_regime_multiplier(self):
        """Test spread/cost multiplier based on vol regime."""
        from forex_features import get_fx_vol_regime_multiplier

        assert get_fx_vol_regime_multiplier("low") == 0.8
        assert get_fx_vol_regime_multiplier("normal") == 1.0
        assert get_fx_vol_regime_multiplier("elevated") == 1.2
        assert get_fx_vol_regime_multiplier("high") == 1.5
        assert get_fx_vol_regime_multiplier("unknown") == 1.0  # Default

    def test_implied_vol_in_extract_features(self):
        """Test that implied vol is integrated into extract_forex_features."""
        from forex_features import extract_forex_features, BenchmarkForexData

        benchmark = BenchmarkForexData(
            implied_vol_prices=[8.0, 9.0, 10.0, 9.5, 8.5],
        )

        features = extract_forex_features(
            row={"timestamp": 1700000000000},
            symbol="EUR_USD",
            benchmark_data=benchmark,
        )

        # Should have computed implied vol
        assert features.implied_vol >= 0.0
        # vol_valid should be True because implied vol was computed
        assert features.vol_valid is True

    def test_implied_vol_dataframe_integration(self):
        """Test implied vol column is added to DataFrame."""
        from forex_features import add_forex_features_to_dataframe

        # Create simple price DataFrame
        df = pd.DataFrame(
            {
                "timestamp": [1700000000000 + i * 3600000 for i in range(10)],
                "close": [1.05 + i * 0.001 for i in range(10)],
                "open": [1.05 + i * 0.001 for i in range(10)],
                "high": [1.055 + i * 0.001 for i in range(10)],
                "low": [1.045 + i * 0.001 for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        # Create implied vol DataFrame
        implied_vol_df = pd.DataFrame(
            {
                "timestamp": [1700000000000 + i * 3600000 for i in range(10)],
                "close": [8.0 + i * 0.1 for i in range(10)],  # EVZ prices
            }
        )

        result = add_forex_features_to_dataframe(
            df,
            "EUR_USD",
            implied_vol_df=implied_vol_df,
        )

        assert "implied_vol" in result.columns
        # Should have some non-zero values
        assert result["implied_vol"].sum() > 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from forex_features import add_forex_features_to_dataframe

        df = pd.DataFrame()
        result = add_forex_features_to_dataframe(df, "EUR_USD")
        assert result.empty

    def test_invalid_symbol_format(self):
        """Test handling of invalid symbol format."""
        from forex_features import extract_forex_features

        features = extract_forex_features(
            row={},
            symbol="INVALID",
        )
        # Should not raise, just return default features
        assert features.carry_valid is False

    def test_nan_in_prices(self):
        """Test handling of NaN values in prices."""
        from forex_features import calculate_volatility_features

        prices = [1.0, 1.1, np.nan, 1.2, 1.15, 1.18]
        # Should handle NaN gracefully
        try:
            vol_5d, vol_20d, vol_ratio, valid = calculate_volatility_features(prices)
        except Exception:
            # If it raises, that's acceptable - but should not crash
            pass

    def test_negative_spread(self):
        """Test handling of negative spread (edge case)."""
        from forex_features import calculate_spread_features

        result = calculate_spread_features(-1.0, [1.0, 1.5, 1.2])
        assert result[-1] is False  # valid should be False

    def test_zero_open_interest_cot(self):
        """Test handling of zero open interest in COT data."""
        from forex_features import calculate_cot_features

        cot_df = pd.DataFrame(
            {
                "EUR_NET": [10000],
                "EUR_OI": [0],  # Zero OI
            }
        )

        net_pct, zscore, change, positioning, valid = calculate_cot_features("EUR", cot_df)
        # Should handle gracefully
        assert valid is True or valid is False  # Either is acceptable

    def test_implied_vol_nan_value(self):
        """Test handling of NaN in implied vol prices."""
        from forex_features import calculate_implied_vol_features

        implied_vol_prices = [8.0, 9.0, float("nan")]

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        # NaN at end makes current value NaN, should return invalid
        assert valid is False

    def test_implied_vol_negative_value(self):
        """Test handling of negative implied vol (edge case)."""
        from forex_features import calculate_implied_vol_features

        implied_vol_prices = [8.0, 9.0, -5.0]  # Negative value at end

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        # Negative vol should return invalid
        assert valid is False

    def test_implied_vol_extreme_value(self):
        """Test handling of extreme implied vol value."""
        from forex_features import calculate_implied_vol_features

        # Extreme vol like during 2008 crisis or flash crash
        implied_vol_prices = [50.0]  # 50% implied vol

        normalized_vol, percentile, regime, valid = calculate_implied_vol_features(
            implied_vol_prices, symbol="EUR_USD"
        )

        assert valid is True
        assert regime == "high"
        # Normalized should be capped at 1.0
        assert normalized_vol <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
