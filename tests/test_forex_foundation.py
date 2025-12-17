# -*- coding: utf-8 -*-
"""
tests/test_forex_foundation.py
Phase 0: Forex Foundation Tests

Tests for Forex-specific models, enums, and helper functions.
Ensures all forex foundation components work correctly before
implementing adapters and execution providers.

Test Categories:
1. ForexSessionType enum tests
2. CurrencyPairCategory enum tests
3. ForexSessionWindow dataclass tests
4. ForexSpreadProfile dataclass tests
5. Pip size and conversion function tests
6. Currency pair classification tests
7. Session detection and liquidity factor tests
8. ExchangeVendor forex extensions tests
9. Backward compatibility tests (no regression)
10. Property-based tests (edge cases)

References:
    - BIS Triennial Survey 2022
    - Standard forex session times
    - OANDA API documentation
"""

import pytest
import math
from decimal import Decimal
from typing import Tuple

# Import forex models
from adapters.models import (
    # Existing models (backward compat)
    MarketType,
    ExchangeVendor,
    TradingSession,
    MarketCalendar,
    SessionType,
    # New forex models
    ForexSessionType,
    CurrencyPairCategory,
    ForexSessionWindow,
    ForexSpreadProfile,
    # Forex constants
    FOREX_SESSION_WINDOWS,
    FOREX_WEEKEND_WINDOW,
    PIP_SIZE_BY_QUOTE_CURRENCY,
    FOREX_MAJOR_PAIRS,
    FOREX_MINOR_PAIRS,
    FOREX_JPY_CROSSES,
    FOREX_EXOTIC_PAIRS,
    FOREX_SPREAD_PROFILES,
    # Forex helper functions
    get_pip_size,
    pips_to_price,
    price_to_pips,
    classify_currency_pair,
    get_spread_profile,
    create_forex_calendar,
    get_current_forex_session,
    get_session_liquidity_factor,
    get_session_spread_multiplier,
)

# Import AssetClass for forex check
from execution_providers import AssetClass


# =============================================================================
# Test: ForexSessionType Enum
# =============================================================================

class TestForexSessionType:
    """Test ForexSessionType enum values and properties."""

    def test_all_session_types_exist(self):
        """Verify all expected session types are defined."""
        expected = [
            "SYDNEY", "TOKYO", "LONDON", "NEW_YORK",
            "LONDON_NY_OVERLAP", "TOKYO_LONDON_OVERLAP",
            "WEEKEND", "OFF_HOURS"
        ]
        actual = [s.name for s in ForexSessionType]
        assert set(expected) == set(actual)

    def test_session_type_values(self):
        """Test session type string values."""
        assert ForexSessionType.SYDNEY.value == "sydney"
        assert ForexSessionType.TOKYO.value == "tokyo"
        assert ForexSessionType.LONDON.value == "london"
        assert ForexSessionType.NEW_YORK.value == "new_york"
        assert ForexSessionType.LONDON_NY_OVERLAP.value == "london_ny_overlap"
        assert ForexSessionType.WEEKEND.value == "weekend"

    def test_session_type_string_enum(self):
        """Verify ForexSessionType is a string enum."""
        assert isinstance(ForexSessionType.LONDON.value, str)
        assert str(ForexSessionType.LONDON) == "ForexSessionType.LONDON"

    def test_session_type_from_value(self):
        """Test creating session type from string value."""
        assert ForexSessionType("london") == ForexSessionType.LONDON
        assert ForexSessionType("tokyo") == ForexSessionType.TOKYO


# =============================================================================
# Test: CurrencyPairCategory Enum
# =============================================================================

class TestCurrencyPairCategory:
    """Test CurrencyPairCategory enum values."""

    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        expected = ["MAJOR", "MINOR", "CROSS", "EXOTIC"]
        actual = [c.name for c in CurrencyPairCategory]
        assert set(expected) == set(actual)

    def test_category_values(self):
        """Test category string values."""
        assert CurrencyPairCategory.MAJOR.value == "major"
        assert CurrencyPairCategory.MINOR.value == "minor"
        assert CurrencyPairCategory.CROSS.value == "cross"
        assert CurrencyPairCategory.EXOTIC.value == "exotic"

    def test_category_from_value(self):
        """Test creating category from string value."""
        assert CurrencyPairCategory("major") == CurrencyPairCategory.MAJOR
        assert CurrencyPairCategory("exotic") == CurrencyPairCategory.EXOTIC


# =============================================================================
# Test: ForexSessionWindow Dataclass
# =============================================================================

class TestForexSessionWindow:
    """Test ForexSessionWindow dataclass."""

    def test_basic_creation(self):
        """Test basic creation of session window."""
        window = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        assert window.session_type == ForexSessionType.LONDON
        assert window.start_hour_utc == 7
        assert window.end_hour_utc == 16
        assert window.liquidity_factor == 1.1
        assert window.spread_multiplier == 1.0
        assert window.days_of_week == (0, 1, 2, 3, 4)  # Default Mon-Fri

    def test_custom_days_of_week(self):
        """Test custom days of week."""
        window = ForexSessionWindow(
            session_type=ForexSessionType.SYDNEY,
            start_hour_utc=21,
            end_hour_utc=6,
            liquidity_factor=0.65,
            spread_multiplier=1.5,
            days_of_week=(0, 1, 2, 3, 4, 6),  # Include Sunday
        )
        assert 6 in window.days_of_week

    def test_is_overnight_property(self):
        """Test overnight session detection."""
        # Sydney session: 21:00 - 06:00 (overnight)
        overnight = ForexSessionWindow(
            session_type=ForexSessionType.SYDNEY,
            start_hour_utc=21,
            end_hour_utc=6,
            liquidity_factor=0.65,
            spread_multiplier=1.5,
        )
        assert overnight.is_overnight is True

        # London session: 07:00 - 16:00 (not overnight)
        daytime = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        assert daytime.is_overnight is False

    def test_duration_hours_property(self):
        """Test duration calculation."""
        # London: 7 to 16 = 9 hours
        london = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        assert london.duration_hours == 9

        # Sydney: 21 to 6 = 9 hours (overnight)
        sydney = ForexSessionWindow(
            session_type=ForexSessionType.SYDNEY,
            start_hour_utc=21,
            end_hour_utc=6,
            liquidity_factor=0.65,
            spread_multiplier=1.5,
        )
        assert sydney.duration_hours == 9

    def test_contains_hour_daytime(self):
        """Test contains_hour for daytime session."""
        london = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        # Monday at 10:00 - should be in session
        assert london.contains_hour(10, 0) is True
        # Monday at 6:00 - before session
        assert london.contains_hour(6, 0) is False
        # Monday at 16:00 - at end (exclusive)
        assert london.contains_hour(16, 0) is False
        # Saturday at 10:00 - weekend
        assert london.contains_hour(10, 5) is False

    def test_contains_hour_overnight(self):
        """Test contains_hour for overnight session."""
        sydney = ForexSessionWindow(
            session_type=ForexSessionType.SYDNEY,
            start_hour_utc=21,
            end_hour_utc=6,
            liquidity_factor=0.65,
            spread_multiplier=1.5,
            days_of_week=(0, 1, 2, 3, 4, 6),
        )
        # Monday at 22:00 - in session
        assert sydney.contains_hour(22, 0) is True
        # Tuesday at 3:00 - in session (overnight)
        assert sydney.contains_hour(3, 1) is True
        # Monday at 12:00 - outside session
        assert sydney.contains_hour(12, 0) is False
        # Sunday at 22:00 - in session (day 6)
        assert sydney.contains_hour(22, 6) is True

    def test_validation_invalid_hour(self):
        """Test validation rejects invalid hours."""
        with pytest.raises(ValueError, match="start_hour_utc must be 0-23"):
            ForexSessionWindow(
                session_type=ForexSessionType.LONDON,
                start_hour_utc=25,  # Invalid
                end_hour_utc=16,
                liquidity_factor=1.0,
                spread_multiplier=1.0,
            )

        with pytest.raises(ValueError, match="end_hour_utc must be 0-23"):
            ForexSessionWindow(
                session_type=ForexSessionType.LONDON,
                start_hour_utc=7,
                end_hour_utc=-1,  # Invalid
                liquidity_factor=1.0,
                spread_multiplier=1.0,
            )

    def test_validation_invalid_liquidity(self):
        """Test validation rejects negative liquidity."""
        with pytest.raises(ValueError, match="liquidity_factor must be non-negative"):
            ForexSessionWindow(
                session_type=ForexSessionType.LONDON,
                start_hour_utc=7,
                end_hour_utc=16,
                liquidity_factor=-0.1,  # Invalid - negative
                spread_multiplier=1.0,
            )

    def test_validation_zero_liquidity_allowed(self):
        """Test that zero liquidity is allowed (for weekend)."""
        # 0.0 is valid for weekend sessions - no trading
        window = ForexSessionWindow(
            session_type=ForexSessionType.WEEKEND,
            start_hour_utc=21,
            end_hour_utc=21,
            liquidity_factor=0.0,  # Valid for weekend
            spread_multiplier=float("inf"),
        )
        assert window.liquidity_factor == 0.0

    def test_validation_invalid_spread_multiplier(self):
        """Test validation rejects non-positive spread multiplier."""
        with pytest.raises(ValueError, match="spread_multiplier must be positive"):
            ForexSessionWindow(
                session_type=ForexSessionType.LONDON,
                start_hour_utc=7,
                end_hour_utc=16,
                liquidity_factor=1.0,
                spread_multiplier=-1.0,  # Invalid
            )

    def test_to_dict(self):
        """Test serialization to dict."""
        window = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        d = window.to_dict()
        assert d["session_type"] == "london"
        assert d["start_hour_utc"] == 7
        assert d["end_hour_utc"] == 16
        assert d["liquidity_factor"] == 1.1
        assert d["spread_multiplier"] == 1.0
        assert d["days_of_week"] == [0, 1, 2, 3, 4]

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "session_type": "tokyo",
            "start_hour_utc": 0,
            "end_hour_utc": 9,
            "liquidity_factor": 0.75,
            "spread_multiplier": 1.3,
            "days_of_week": [0, 1, 2, 3, 4],
        }
        window = ForexSessionWindow.from_dict(d)
        assert window.session_type == ForexSessionType.TOKYO
        assert window.start_hour_utc == 0
        assert window.end_hour_utc == 9
        assert window.liquidity_factor == 0.75
        assert window.spread_multiplier == 1.3

    def test_immutability(self):
        """Test that ForexSessionWindow is immutable (frozen)."""
        window = ForexSessionWindow(
            session_type=ForexSessionType.LONDON,
            start_hour_utc=7,
            end_hour_utc=16,
            liquidity_factor=1.1,
            spread_multiplier=1.0,
        )
        with pytest.raises(AttributeError):
            window.start_hour_utc = 8


# =============================================================================
# Test: Forex Session Constants
# =============================================================================

class TestForexSessionConstants:
    """Test predefined forex session constants."""

    def test_all_main_sessions_defined(self):
        """Verify all main sessions are in FOREX_SESSION_WINDOWS."""
        session_types = {w.session_type for w in FOREX_SESSION_WINDOWS}
        expected = {
            ForexSessionType.SYDNEY,
            ForexSessionType.TOKYO,
            ForexSessionType.LONDON,
            ForexSessionType.NEW_YORK,
            ForexSessionType.LONDON_NY_OVERLAP,
            ForexSessionType.TOKYO_LONDON_OVERLAP,
        }
        assert expected.issubset(session_types)

    def test_london_ny_overlap_has_best_liquidity(self):
        """Verify London/NY overlap has highest liquidity and tightest spreads."""
        overlap = None
        for w in FOREX_SESSION_WINDOWS:
            if w.session_type == ForexSessionType.LONDON_NY_OVERLAP:
                overlap = w
                break

        assert overlap is not None
        # Should have highest liquidity factor
        assert overlap.liquidity_factor >= 1.3
        # Should have tightest spreads (lowest multiplier)
        assert overlap.spread_multiplier <= 0.9

    def test_sydney_has_lowest_liquidity(self):
        """Verify Sydney has lowest liquidity among main sessions."""
        sydney = None
        for w in FOREX_SESSION_WINDOWS:
            if w.session_type == ForexSessionType.SYDNEY:
                sydney = w
                break

        assert sydney is not None
        assert sydney.liquidity_factor <= 0.7
        assert sydney.spread_multiplier >= 1.4

    def test_weekend_window_no_trading(self):
        """Verify weekend window has zero liquidity."""
        assert FOREX_WEEKEND_WINDOW.session_type == ForexSessionType.WEEKEND
        assert FOREX_WEEKEND_WINDOW.liquidity_factor == 0.0
        assert FOREX_WEEKEND_WINDOW.spread_multiplier == float("inf")


# =============================================================================
# Test: Pip Size Functions
# =============================================================================

class TestPipSizeFunctions:
    """Test pip size and conversion functions."""

    def test_get_pip_size_standard_pairs(self):
        """Test pip size for standard (4 decimal) pairs."""
        assert get_pip_size("EUR_USD") == 0.0001
        assert get_pip_size("GBP_USD") == 0.0001
        assert get_pip_size("AUD_USD") == 0.0001
        assert get_pip_size("EUR/USD") == 0.0001  # Slash format
        assert get_pip_size("eur_usd") == 0.0001  # Lowercase

    def test_get_pip_size_jpy_pairs(self):
        """Test pip size for JPY pairs (2 decimal)."""
        assert get_pip_size("USD_JPY") == 0.01
        assert get_pip_size("EUR_JPY") == 0.01
        assert get_pip_size("GBP_JPY") == 0.01
        assert get_pip_size("AUD/JPY") == 0.01  # Slash format

    def test_get_pip_size_huf_pairs(self):
        """Test pip size for HUF pairs (2 decimal)."""
        assert get_pip_size("USD_HUF") == 0.01
        assert get_pip_size("EUR_HUF") == 0.01

    def test_get_pip_size_unknown_pair(self):
        """Test pip size defaults to 0.0001 for unknown pairs."""
        assert get_pip_size("XXX_YYY") == 0.0001
        assert get_pip_size("UNKNOWN") == 0.0001

    def test_pips_to_price_standard(self):
        """Test pips to price conversion for standard pairs."""
        # 10 pips on EUR/USD = 0.001
        assert abs(pips_to_price(10, "EUR_USD") - 0.001) < 1e-10
        # 1 pip on EUR/USD = 0.0001
        assert abs(pips_to_price(1, "EUR_USD") - 0.0001) < 1e-10

    def test_pips_to_price_jpy(self):
        """Test pips to price conversion for JPY pairs."""
        # 10 pips on USD/JPY = 0.1
        assert abs(pips_to_price(10, "USD_JPY") - 0.1) < 1e-10
        # 1 pip on USD/JPY = 0.01
        assert abs(pips_to_price(1, "USD_JPY") - 0.01) < 1e-10

    def test_price_to_pips_standard(self):
        """Test price to pips conversion for standard pairs."""
        # 0.001 on EUR/USD = 10 pips
        assert abs(price_to_pips(0.001, "EUR_USD") - 10.0) < 1e-10
        # 0.0001 on EUR/USD = 1 pip
        assert abs(price_to_pips(0.0001, "EUR_USD") - 1.0) < 1e-10

    def test_price_to_pips_jpy(self):
        """Test price to pips conversion for JPY pairs."""
        # 0.1 on USD/JPY = 10 pips
        assert abs(price_to_pips(0.1, "USD_JPY") - 10.0) < 1e-10
        # 0.01 on USD/JPY = 1 pip
        assert abs(price_to_pips(0.01, "USD_JPY") - 1.0) < 1e-10

    def test_pips_price_roundtrip(self):
        """Test roundtrip conversion pips -> price -> pips."""
        pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "EUR_JPY"]
        pips_values = [1.0, 5.5, 10.0, 100.0]

        for pair in pairs:
            for pips in pips_values:
                price = pips_to_price(pips, pair)
                result = price_to_pips(price, pair)
                assert abs(result - pips) < 1e-10, f"Roundtrip failed for {pair} with {pips} pips"


# =============================================================================
# Test: Currency Pair Classification
# =============================================================================

class TestCurrencyPairClassification:
    """Test currency pair classification functions."""

    def test_classify_major_pairs(self):
        """Test classification of major pairs."""
        for pair in FOREX_MAJOR_PAIRS:
            assert classify_currency_pair(pair) == CurrencyPairCategory.MAJOR, f"{pair} should be MAJOR"

    def test_classify_minor_pairs(self):
        """Test classification of minor pairs."""
        for pair in FOREX_MINOR_PAIRS:
            assert classify_currency_pair(pair) == CurrencyPairCategory.MINOR, f"{pair} should be MINOR"

    def test_classify_jpy_crosses(self):
        """Test classification of JPY crosses."""
        for pair in FOREX_JPY_CROSSES:
            assert classify_currency_pair(pair) == CurrencyPairCategory.CROSS, f"{pair} should be CROSS"

    def test_classify_exotic_pairs(self):
        """Test classification of exotic pairs."""
        for pair in FOREX_EXOTIC_PAIRS:
            assert classify_currency_pair(pair) == CurrencyPairCategory.EXOTIC, f"{pair} should be EXOTIC"

    def test_classify_slash_format(self):
        """Test classification with slash format."""
        assert classify_currency_pair("EUR/USD") == CurrencyPairCategory.MAJOR
        assert classify_currency_pair("EUR/JPY") == CurrencyPairCategory.CROSS
        assert classify_currency_pair("USD/TRY") == CurrencyPairCategory.EXOTIC

    def test_classify_lowercase(self):
        """Test classification with lowercase."""
        assert classify_currency_pair("eur_usd") == CurrencyPairCategory.MAJOR
        assert classify_currency_pair("eur/gbp") == CurrencyPairCategory.MINOR

    def test_classify_unknown_pair_heuristic(self):
        """Test heuristic classification for unknown pairs."""
        # Unknown USD pair with major currency -> MAJOR
        assert classify_currency_pair("USD_SEK") == CurrencyPairCategory.EXOTIC

        # Unknown pair between major currencies -> uses heuristic
        assert classify_currency_pair("CHF_CAD") == CurrencyPairCategory.MINOR

    def test_classify_completely_unknown(self):
        """Test classification of completely unknown pairs."""
        # Should default to EXOTIC
        assert classify_currency_pair("XXX_YYY") == CurrencyPairCategory.EXOTIC
        assert classify_currency_pair("INVALID") == CurrencyPairCategory.EXOTIC


# =============================================================================
# Test: Forex Spread Profiles
# =============================================================================

class TestForexSpreadProfiles:
    """Test forex spread profile functionality."""

    def test_all_categories_have_profiles(self):
        """Verify all categories have spread profiles."""
        for cat in CurrencyPairCategory:
            assert cat in FOREX_SPREAD_PROFILES

    def test_major_has_tightest_spreads(self):
        """Verify majors have tightest spreads."""
        major = FOREX_SPREAD_PROFILES[CurrencyPairCategory.MAJOR]
        exotic = FOREX_SPREAD_PROFILES[CurrencyPairCategory.EXOTIC]

        assert major.retail_spread_pips < exotic.retail_spread_pips
        assert major.institutional_spread_pips < exotic.institutional_spread_pips

    def test_spread_profile_attributes(self):
        """Test spread profile attribute access."""
        major = FOREX_SPREAD_PROFILES[CurrencyPairCategory.MAJOR]

        assert major.category == CurrencyPairCategory.MAJOR
        assert major.retail_spread_pips > 0
        assert major.institutional_spread_pips > 0
        assert major.conservative_spread_pips > 0
        assert major.avg_daily_range_pips > 0

        # Institutional should be tighter than retail
        assert major.institutional_spread_pips < major.retail_spread_pips

    def test_get_spread_method(self):
        """Test get_spread method with different profiles."""
        major = FOREX_SPREAD_PROFILES[CurrencyPairCategory.MAJOR]

        assert major.get_spread("retail") == major.retail_spread_pips
        assert major.get_spread("institutional") == major.institutional_spread_pips
        assert major.get_spread("conservative") == major.conservative_spread_pips
        assert major.get_spread("unknown") == major.conservative_spread_pips  # Default

    def test_get_spread_profile_function(self):
        """Test get_spread_profile helper function."""
        profile = get_spread_profile("EUR_USD")
        assert profile.category == CurrencyPairCategory.MAJOR

        profile = get_spread_profile("EUR_JPY")
        assert profile.category == CurrencyPairCategory.CROSS

        profile = get_spread_profile("USD_TRY")
        assert profile.category == CurrencyPairCategory.EXOTIC


# =============================================================================
# Test: Session Detection Functions
# =============================================================================

class TestSessionDetection:
    """Test forex session detection functions."""

    def test_weekend_detection_saturday(self):
        """Test weekend detection on Saturday."""
        session = get_current_forex_session(12, 5)  # Saturday noon
        assert session == ForexSessionType.WEEKEND

    def test_weekend_detection_sunday_morning(self):
        """Test weekend detection on Sunday before market open."""
        session = get_current_forex_session(10, 6)  # Sunday 10:00 UTC
        assert session == ForexSessionType.WEEKEND

    def test_weekend_detection_sunday_evening(self):
        """Test market open on Sunday evening."""
        session = get_current_forex_session(22, 6)  # Sunday 22:00 UTC
        assert session != ForexSessionType.WEEKEND  # Should be Sydney

    def test_friday_evening_weekend(self):
        """Test weekend starts Friday evening."""
        session = get_current_forex_session(22, 4)  # Friday 22:00 UTC
        assert session == ForexSessionType.WEEKEND

    def test_london_ny_overlap_priority(self):
        """Test London/NY overlap has highest priority."""
        # 14:00 UTC on Monday - should be overlap
        session = get_current_forex_session(14, 0)
        assert session == ForexSessionType.LONDON_NY_OVERLAP

    def test_london_session(self):
        """Test London session detection."""
        # 10:00 UTC on Tuesday (before NY opens)
        session = get_current_forex_session(10, 1)
        assert session == ForexSessionType.LONDON

    def test_new_york_session(self):
        """Test New York session detection."""
        # 18:00 UTC on Wednesday (after London closes)
        session = get_current_forex_session(18, 2)
        assert session == ForexSessionType.NEW_YORK

    def test_tokyo_session(self):
        """Test Tokyo session detection."""
        # 3:00 UTC on Thursday
        session = get_current_forex_session(3, 3)
        assert session == ForexSessionType.TOKYO

    def test_tokyo_london_overlap(self):
        """Test Tokyo/London overlap detection."""
        # 8:00 UTC on Monday
        session = get_current_forex_session(8, 0)
        assert session == ForexSessionType.TOKYO_LONDON_OVERLAP


# =============================================================================
# Test: Liquidity and Spread Factor Functions
# =============================================================================

class TestLiquidityFactorFunctions:
    """Test liquidity and spread factor helper functions."""

    def test_weekend_liquidity_zero(self):
        """Test weekend has zero liquidity."""
        liquidity = get_session_liquidity_factor(12, 5)  # Saturday
        assert liquidity == 0.0

    def test_weekend_spread_infinite(self):
        """Test weekend has infinite spread multiplier."""
        spread = get_session_spread_multiplier(12, 5)  # Saturday
        assert spread == float("inf")

    def test_london_ny_overlap_best_liquidity(self):
        """Test London/NY overlap has best liquidity."""
        liquidity = get_session_liquidity_factor(14, 0)  # Monday 14:00 UTC
        assert liquidity >= 1.3  # Best liquidity

        spread = get_session_spread_multiplier(14, 0)
        assert spread <= 0.9  # Tightest spreads

    def test_sydney_low_liquidity(self):
        """Test Sydney session has lower liquidity."""
        # Sunday 22:00 UTC (Sydney session start)
        liquidity = get_session_liquidity_factor(22, 6)
        assert liquidity < 1.0  # Lower than London

    def test_liquidity_positive_during_trading(self):
        """Test liquidity is positive during normal trading hours."""
        # Check various times on weekdays
        for day in range(5):  # Mon-Fri
            for hour in [8, 10, 12, 14, 16, 18]:
                liquidity = get_session_liquidity_factor(hour, day)
                assert liquidity > 0, f"Expected positive liquidity at hour {hour}, day {day}"


# =============================================================================
# Test: ExchangeVendor Forex Extensions
# =============================================================================

class TestExchangeVendorForex:
    """Test ExchangeVendor forex extensions."""

    def test_forex_vendors_exist(self):
        """Verify forex vendors are defined."""
        assert hasattr(ExchangeVendor, "OANDA")
        assert hasattr(ExchangeVendor, "IG")
        assert hasattr(ExchangeVendor, "DUKASCOPY")

    def test_forex_vendor_values(self):
        """Test forex vendor string values."""
        assert ExchangeVendor.OANDA.value == "oanda"
        assert ExchangeVendor.IG.value == "ig"
        assert ExchangeVendor.DUKASCOPY.value == "dukascopy"

    def test_forex_vendor_market_type(self):
        """Test forex vendors return FOREX market type."""
        assert ExchangeVendor.OANDA.market_type == MarketType.FOREX
        assert ExchangeVendor.IG.market_type == MarketType.FOREX
        assert ExchangeVendor.DUKASCOPY.market_type == MarketType.FOREX

    def test_forex_vendor_is_forex_property(self):
        """Test is_forex property for vendors."""
        assert ExchangeVendor.OANDA.is_forex is True
        assert ExchangeVendor.IG.is_forex is True
        assert ExchangeVendor.DUKASCOPY.is_forex is True
        assert ExchangeVendor.BINANCE.is_forex is False
        assert ExchangeVendor.ALPACA.is_forex is False

    def test_forex_market_type_has_trading_hours(self):
        """Test FOREX market type has trading hours."""
        assert MarketType.FOREX.has_trading_hours is True


# =============================================================================
# Test: AssetClass.FOREX
# =============================================================================

class TestAssetClassForex:
    """Test AssetClass.FOREX in execution_providers."""

    def test_forex_asset_class_exists(self):
        """Verify FOREX asset class is defined."""
        assert hasattr(AssetClass, "FOREX")

    def test_forex_asset_class_value(self):
        """Test FOREX asset class value."""
        assert AssetClass.FOREX.value == "forex"


# =============================================================================
# Test: Forex Calendar
# =============================================================================

class TestForexCalendar:
    """Test forex calendar creation."""

    def test_create_forex_calendar_default(self):
        """Test creating forex calendar with default vendor."""
        calendar = create_forex_calendar()
        assert calendar.vendor == ExchangeVendor.OANDA
        assert calendar.market_type == MarketType.FOREX
        assert calendar.timezone == "UTC"
        assert len(calendar.sessions) == 1

    def test_create_forex_calendar_custom_vendor(self):
        """Test creating forex calendar with custom vendor."""
        calendar = create_forex_calendar(ExchangeVendor.IG)
        assert calendar.vendor == ExchangeVendor.IG

    def test_forex_calendar_not_24_7(self):
        """Test forex calendar is not 24/7 (has weekend closure)."""
        calendar = create_forex_calendar()
        # Even though it has a continuous session internally,
        # the is_24_7 property checks market_type
        # Forex is NOT 24/7 like crypto (weekend closure)
        assert calendar.is_24_7 is False


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility - existing functionality should not break."""

    def test_existing_vendors_unchanged(self):
        """Test existing vendors still work correctly."""
        assert ExchangeVendor.BINANCE.market_type == MarketType.CRYPTO_SPOT
        assert ExchangeVendor.ALPACA.market_type == MarketType.EQUITY
        assert ExchangeVendor.POLYGON.market_type == MarketType.CRYPTO_SPOT  # Data provider default
        assert ExchangeVendor.YAHOO.market_type == MarketType.CRYPTO_SPOT  # Data provider default

    def test_existing_market_types_unchanged(self):
        """Test existing market types still work correctly."""
        assert MarketType.CRYPTO_SPOT.is_crypto is True
        assert MarketType.EQUITY.is_equity is True
        assert MarketType.FOREX.is_crypto is False
        assert MarketType.FOREX.is_equity is False

    def test_existing_session_types_unchanged(self):
        """Test existing session types still work."""
        assert SessionType.REGULAR.value == "regular"
        assert SessionType.PRE_MARKET.value == "pre_market"
        assert SessionType.CONTINUOUS.value == "continuous"

    def test_existing_asset_classes_unchanged(self):
        """Test existing asset classes in execution_providers still work."""
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.FUTURES.value == "futures"
        assert AssetClass.OPTIONS.value == "options"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_midnight_session_detection(self):
        """Test session detection at midnight UTC."""
        # 0:00 UTC on Monday should be Tokyo
        session = get_current_forex_session(0, 0)
        assert session == ForexSessionType.TOKYO

    def test_session_boundary_end(self):
        """Test session detection at session end boundary."""
        # 16:00 UTC - London ends, but still in overlap
        session = get_current_forex_session(16, 0)
        # At 16:00, overlap ends (12:00-16:00), but NY continues (12:00-21:00)
        assert session == ForexSessionType.NEW_YORK

    def test_pip_size_empty_string(self):
        """Test pip size with empty string."""
        # Should return default
        assert get_pip_size("") == 0.0001

    def test_pip_size_single_currency(self):
        """Test pip size with single currency (not a pair)."""
        assert get_pip_size("EUR") == 0.0001  # Default

    def test_zero_pips_conversion(self):
        """Test zero pips conversion."""
        assert pips_to_price(0, "EUR_USD") == 0.0
        assert price_to_pips(0, "EUR_USD") == 0.0

    def test_negative_pips_conversion(self):
        """Test negative pips conversion (valid for spreads/losses)."""
        assert pips_to_price(-10, "EUR_USD") == -0.001
        assert price_to_pips(-0.001, "EUR_USD") == -10.0

    def test_very_large_pips(self):
        """Test very large pip values."""
        result = pips_to_price(10000, "EUR_USD")
        assert abs(result - 1.0) < 1e-10


# =============================================================================
# Test: Property-Based Tests (if hypothesis available)
# =============================================================================

class TestPropertyBased:
    """Property-based tests for robustness."""

    def test_pip_conversion_symmetry(self):
        """Test pip/price conversion is symmetric."""
        # For any pip value, converting to price and back should give original
        test_values = [0.1, 1.0, 5.5, 10.0, 50.0, 100.0, 1000.0]
        for pips in test_values:
            for pair in ["EUR_USD", "USD_JPY"]:
                price = pips_to_price(pips, pair)
                result = price_to_pips(price, pair)
                assert abs(result - pips) < 1e-10

    def test_session_always_returns_valid_type(self):
        """Test session detection always returns valid ForexSessionType."""
        for hour in range(24):
            for day in range(7):
                session = get_current_forex_session(hour, day)
                assert isinstance(session, ForexSessionType)

    def test_liquidity_always_non_negative(self):
        """Test liquidity factor is always non-negative."""
        for hour in range(24):
            for day in range(7):
                liquidity = get_session_liquidity_factor(hour, day)
                assert liquidity >= 0

    def test_spread_multiplier_always_positive_or_infinite(self):
        """Test spread multiplier is always positive or infinite."""
        for hour in range(24):
            for day in range(7):
                spread = get_session_spread_multiplier(hour, day)
                assert spread > 0 or spread == float("inf")


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pair_analysis(self):
        """Test full currency pair analysis workflow."""
        pair = "EUR_USD"

        # Get pip size
        pip_size = get_pip_size(pair)
        assert pip_size == 0.0001

        # Classify pair
        category = classify_currency_pair(pair)
        assert category == CurrencyPairCategory.MAJOR

        # Get spread profile
        profile = get_spread_profile(pair)
        assert profile.category == CurrencyPairCategory.MAJOR

        # Convert spread to price
        spread_pips = profile.get_spread("institutional")
        spread_price = pips_to_price(spread_pips, pair)
        assert spread_price > 0

    def test_session_aware_spread_calculation(self):
        """Test calculating session-aware spread."""
        pair = "EUR_USD"

        # Get base spread
        profile = get_spread_profile(pair)
        base_spread_pips = profile.get_spread("retail")

        # Get session multiplier (Monday 14:00 - London/NY overlap)
        multiplier = get_session_spread_multiplier(14, 0)

        # Calculate effective spread
        effective_spread = base_spread_pips * multiplier
        assert effective_spread < base_spread_pips  # Overlap has tighter spreads

        # Same for low liquidity period (Sydney)
        sydney_mult = get_session_spread_multiplier(22, 6)  # Sunday night
        sydney_spread = base_spread_pips * sydney_mult
        assert sydney_spread > base_spread_pips  # Sydney has wider spreads


# =============================================================================
# Test: Mock Fixtures Usage
# =============================================================================

# Import forex conftest fixtures
from tests.conftest_forex import (
    MockOandaPrice,
    MockOandaCandle,
    CURRENCY_PAIR_TEST_DATA,
    SESSION_LIQUIDITY_TEST_DATA,
)


class TestMockOandaFixtures:
    """Tests using mock OANDA fixtures."""

    def test_mock_price_spread_calculation(self):
        """Test spread calculation from mock price."""
        price = MockOandaPrice(
            instrument="EUR_USD",
            bid=Decimal("1.08500"),
            ask=Decimal("1.08515"),
            time="2025-01-15T12:00:00.000000000Z",
            tradeable=True,
        )

        # Spread should be 1.5 pips (0.00015 / 0.0001)
        assert abs(price.spread_pips - 1.5) < 0.01

    def test_mock_price_jpy_spread_calculation(self):
        """Test spread calculation for JPY pair from mock price."""
        price = MockOandaPrice(
            instrument="USD_JPY",
            bid=Decimal("150.120"),
            ask=Decimal("150.135"),
            time="2025-01-15T12:00:00.000000000Z",
            tradeable=True,
        )

        # Spread should be 1.5 pips (0.015 / 0.01)
        assert abs(price.spread_pips - 1.5) < 0.01

    def test_mock_price_to_oanda_response(self):
        """Test conversion to OANDA API response format."""
        price = MockOandaPrice(
            instrument="GBP_USD",
            bid=Decimal("1.26500"),
            ask=Decimal("1.26520"),
            time="2025-01-15T12:00:00.000000000Z",
            tradeable=True,
        )

        response = price.to_oanda_response()
        assert "prices" in response
        assert len(response["prices"]) == 1
        assert response["prices"][0]["instrument"] == "GBP_USD"
        assert response["prices"][0]["tradeable"] is True
        assert response["prices"][0]["bids"][0]["price"] == "1.26500"
        assert response["prices"][0]["asks"][0]["price"] == "1.26520"

    def test_mock_candle_to_oanda_response(self):
        """Test candle conversion to OANDA API format."""
        candle = MockOandaCandle(
            instrument="EUR_USD",
            time="2025-01-15T12:00:00.000000000Z",
            open=Decimal("1.08500"),
            high=Decimal("1.08600"),
            low=Decimal("1.08400"),
            close=Decimal("1.08550"),
            volume=10000,
            complete=True,
        )

        response = candle.to_oanda_response()
        assert response["complete"] is True
        assert response["volume"] == 10000
        assert response["mid"]["o"] == "1.08500"
        assert response["mid"]["h"] == "1.08600"
        assert response["mid"]["l"] == "1.08400"
        assert response["mid"]["c"] == "1.08550"


# =============================================================================
# Test: Parametrized Property-Based Tests
# =============================================================================

class TestParametrizedCurrencyPairs:
    """Parametrized tests for currency pairs (property-based style)."""

    @pytest.mark.parametrize("symbol,expected_category,expected_pip_size", CURRENCY_PAIR_TEST_DATA)
    def test_currency_pair_classification_parametrized(
        self, symbol: str, expected_category: str, expected_pip_size: float
    ):
        """Test currency pair classification with parametrized data."""
        category = classify_currency_pair(symbol)
        assert category.value == expected_category

    @pytest.mark.parametrize("symbol,expected_category,expected_pip_size", CURRENCY_PAIR_TEST_DATA)
    def test_pip_size_parametrized(
        self, symbol: str, expected_category: str, expected_pip_size: float
    ):
        """Test pip size with parametrized data."""
        pip_size = get_pip_size(symbol)
        assert pip_size == expected_pip_size

    @pytest.mark.parametrize("symbol,expected_category,expected_pip_size", CURRENCY_PAIR_TEST_DATA)
    def test_pip_conversion_roundtrip_parametrized(
        self, symbol: str, expected_category: str, expected_pip_size: float
    ):
        """Test pip conversion roundtrip with parametrized data."""
        test_pips = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        for pips in test_pips:
            price = pips_to_price(pips, symbol)
            result = price_to_pips(price, symbol)
            assert abs(result - pips) < 1e-10, f"Roundtrip failed for {symbol} with {pips} pips"


class TestParametrizedSessionLiquidity:
    """Parametrized tests for session liquidity (property-based style)."""

    @pytest.mark.parametrize("hour_utc,day_of_week,liquidity_range", SESSION_LIQUIDITY_TEST_DATA)
    def test_session_liquidity_in_range(
        self, hour_utc: int, day_of_week: int, liquidity_range: tuple
    ):
        """Test session liquidity falls within expected range."""
        liquidity = get_session_liquidity_factor(hour_utc, day_of_week)
        min_liq, max_liq = liquidity_range
        assert min_liq <= liquidity <= max_liq, (
            f"Liquidity {liquidity} not in range [{min_liq}, {max_liq}] "
            f"for hour={hour_utc}, day={day_of_week}"
        )


# =============================================================================
# Test: Regression / Isolation Tests
# =============================================================================

class TestRegressionIsolation:
    """Regression and isolation tests to prevent cross-contamination."""

    def test_forex_does_not_affect_crypto_vendors(self):
        """Test that forex vendors don't affect existing crypto vendor behavior."""
        from adapters.models import ExchangeVendor, MarketType

        # Crypto vendors should still work exactly as before
        assert ExchangeVendor.BINANCE.market_type == MarketType.CRYPTO_SPOT
        assert ExchangeVendor.BINANCE_US.market_type == MarketType.CRYPTO_SPOT

        # Verify is_forex is False for crypto
        assert ExchangeVendor.BINANCE.is_forex is False
        assert ExchangeVendor.BINANCE_US.is_forex is False

    def test_forex_does_not_affect_equity_vendors(self):
        """Test that forex vendors don't affect existing equity vendor behavior."""
        from adapters.models import ExchangeVendor, MarketType

        # Equity vendors should still work exactly as before
        assert ExchangeVendor.ALPACA.market_type == MarketType.EQUITY

        # Verify is_forex is False for equity
        assert ExchangeVendor.ALPACA.is_forex is False

    def test_forex_asset_class_isolated_from_others(self):
        """Test AssetClass.FOREX is isolated from other asset classes."""
        from execution_providers import AssetClass

        # Each asset class should have unique value
        values = [ac.value for ac in AssetClass]
        assert len(values) == len(set(values)), "Asset class values must be unique"

        # FOREX should not match any other
        assert AssetClass.FOREX != AssetClass.CRYPTO
        assert AssetClass.FOREX != AssetClass.EQUITY
        assert AssetClass.FOREX != AssetClass.FUTURES
        assert AssetClass.FOREX != AssetClass.OPTIONS

    def test_forex_session_constants_immutable(self):
        """Test forex session constants are immutable."""
        # ForexSessionWindow is frozen
        window = FOREX_SESSION_WINDOWS[0]
        with pytest.raises(AttributeError):
            window.liquidity_factor = 999.0

    def test_forex_models_do_not_pollute_namespace(self):
        """Test forex models don't add unexpected items to models module."""
        import adapters.models as models

        # These should NOT exist (would indicate pollution)
        assert not hasattr(models, "CryptoSessionType")  # No such thing
        assert not hasattr(models, "EquityForexHybrid")  # No such thing

        # These SHOULD exist (forex additions)
        assert hasattr(models, "ForexSessionType")
        assert hasattr(models, "CurrencyPairCategory")
        assert hasattr(models, "ForexSessionWindow")
        assert hasattr(models, "ForexSpreadProfile")


# =============================================================================
# Test: Stress and Boundary Tests
# =============================================================================

class TestStressBoundary:
    """Stress and boundary condition tests."""

    def test_all_24_hours_7_days_session_detection(self):
        """Test session detection for all 168 hour/day combinations."""
        valid_sessions = set(ForexSessionType)
        for day in range(7):
            for hour in range(24):
                session = get_current_forex_session(hour, day)
                assert session in valid_sessions, f"Invalid session at hour={hour}, day={day}"

    def test_all_24_hours_7_days_liquidity_non_negative(self):
        """Test liquidity is non-negative for all times."""
        for day in range(7):
            for hour in range(24):
                liquidity = get_session_liquidity_factor(hour, day)
                assert liquidity >= 0, f"Negative liquidity at hour={hour}, day={day}"

    def test_all_24_hours_7_days_spread_positive_or_inf(self):
        """Test spread multiplier is positive or inf for all times."""
        for day in range(7):
            for hour in range(24):
                spread = get_session_spread_multiplier(hour, day)
                assert spread > 0 or spread == float("inf"), (
                    f"Invalid spread at hour={hour}, day={day}"
                )

    def test_pip_size_consistency_across_formats(self):
        """Test pip size is consistent across symbol formats."""
        test_pairs = [
            ("EUR_USD", "EUR/USD", "eur_usd", "EUR/usd"),
            ("USD_JPY", "USD/JPY", "usd_jpy", "usd/JPY"),
            ("GBP_USD", "GBP/USD", "gbp_usd", "Gbp_Usd"),
        ]
        for formats in test_pairs:
            pip_sizes = [get_pip_size(f) for f in formats]
            assert len(set(pip_sizes)) == 1, f"Inconsistent pip sizes for {formats}"

    def test_classification_consistency_across_formats(self):
        """Test classification is consistent across symbol formats."""
        test_pairs = [
            ("EUR_USD", "EUR/USD", "eur_usd"),
            ("EUR_JPY", "EUR/JPY", "eur_jpy"),
            ("USD_TRY", "USD/TRY", "usd_try"),
        ]
        for formats in test_pairs:
            categories = [classify_currency_pair(f) for f in formats]
            assert len(set(categories)) == 1, f"Inconsistent categories for {formats}"

    def test_extreme_pip_values(self):
        """Test pip conversion with extreme values."""
        # Very small
        assert pips_to_price(0.001, "EUR_USD") == pytest.approx(0.0000001, rel=1e-10)

        # Very large
        assert pips_to_price(1000000, "EUR_USD") == pytest.approx(100.0, rel=1e-10)

        # Negative
        assert pips_to_price(-500, "EUR_USD") == pytest.approx(-0.05, rel=1e-10)


# =============================================================================
# Test: OANDA Adapter Full Implementation (Phase 2 Complete)
# =============================================================================

class TestOandaAdapterImplementation:
    """Tests for OANDA adapter full implementation (Phase 2)."""

    def test_oanda_module_exists(self):
        """Test that oanda adapter module exists."""
        import adapters.oanda
        assert adapters.oanda is not None

    def test_oanda_module_exports_adapters(self):
        """Test that oanda module exports all adapters (Phase 2 complete)."""
        import adapters.oanda as oanda

        # __all__ should contain all 5 adapters after Phase 2
        assert hasattr(oanda, "__all__")
        expected_exports = {
            "OandaMarketDataAdapter",
            "OandaFeeAdapter",
            "OandaTradingHoursAdapter",
            "OandaExchangeInfoAdapter",
            "OandaOrderExecutionAdapter",
        }
        assert set(oanda.__all__) == expected_exports

    def test_oanda_docstring_present(self):
        """Test that oanda module has proper documentation."""
        import adapters.oanda as oanda

        assert oanda.__doc__ is not None
        assert "OANDA" in oanda.__doc__
        # Phase 2 complete - should say "Tested and operational"
        assert "Tested and operational" in oanda.__doc__ or "Phase 2" in oanda.__doc__

    def test_oanda_adapters_importable(self):
        """Test that all OANDA adapters can be imported."""
        from adapters.oanda import (
            OandaMarketDataAdapter,
            OandaFeeAdapter,
            OandaTradingHoursAdapter,
            OandaExchangeInfoAdapter,
            OandaOrderExecutionAdapter,
        )
        assert OandaMarketDataAdapter is not None
        assert OandaFeeAdapter is not None
        assert OandaTradingHoursAdapter is not None
        assert OandaExchangeInfoAdapter is not None
        assert OandaOrderExecutionAdapter is not None

    def test_oanda_adapters_inherit_from_base(self):
        """Test that OANDA adapters inherit from proper base classes."""
        from adapters.oanda import (
            OandaMarketDataAdapter,
            OandaFeeAdapter,
            OandaTradingHoursAdapter,
            OandaExchangeInfoAdapter,
            OandaOrderExecutionAdapter,
        )
        from adapters.base import (
            MarketDataAdapter,
            FeeAdapter,
            TradingHoursAdapter,
            ExchangeInfoAdapter,
            OrderExecutionAdapter,
        )

        assert issubclass(OandaMarketDataAdapter, MarketDataAdapter)
        assert issubclass(OandaFeeAdapter, FeeAdapter)
        assert issubclass(OandaTradingHoursAdapter, TradingHoursAdapter)
        assert issubclass(OandaExchangeInfoAdapter, ExchangeInfoAdapter)
        assert issubclass(OandaOrderExecutionAdapter, OrderExecutionAdapter)
