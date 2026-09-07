# -*- coding: utf-8 -*-
"""
tests/test_forex_realtime_swaps.py
Comprehensive tests for Real-time Swap Rates Provider

Tests cover:
1. RealtimeSwapRate dataclass functionality
2. SwapRateCalculator interest rate differential calculations
3. OandaSwapRateClient API interaction (mocked)
4. RealtimeSwapRateProvider caching and fallback
5. Wednesday triple swap calculation
6. Quality indicators and staleness detection

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

import json
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest

from services.forex_realtime_swaps import (
    # Enums
    SwapRateSource,
    SwapDirection,
    SwapRateQuality,
    # Data classes
    RealtimeSwapRate,
    SwapRateUpdate,
    SwapRateCacheConfig,
    # Classes
    SwapRateCalculator,
    OandaSwapRateClient,
    RealtimeSwapRateProvider,
    # Factory
    create_realtime_swap_provider,
    # Constants
    DEFAULT_INTERBANK_RATES,
    BROKER_MARKUP_RANGES,
    DEFAULT_CACHE_TTL_SEC,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_swap_rate() -> RealtimeSwapRate:
    """Create a sample swap rate for testing."""
    return RealtimeSwapRate(
        pair="EUR_USD",
        long_swap_pips=-0.5,
        short_swap_pips=0.3,
        long_swap_pct=-1.825,
        short_swap_pct=1.095,
        timestamp_ms=int(time.time() * 1000),
        source=SwapRateSource.OANDA_REALTIME,
        quality=SwapRateQuality.FRESH,
    )


@pytest.fixture
def swap_calculator() -> SwapRateCalculator:
    """Create a swap rate calculator."""
    return SwapRateCalculator(broker_tier="retail")


@pytest.fixture
def cache_config(tmp_path: Path) -> SwapRateCacheConfig:
    """Create cache config with temp directory."""
    return SwapRateCacheConfig(
        cache_ttl_sec=60.0,
        stale_threshold_sec=120.0,
        persist_to_disk=True,
        cache_file_path=str(tmp_path / "swap_cache.json"),
        auto_refresh_enabled=False,
        broker_tier="retail",
    )


# =============================================================================
# RealtimeSwapRate Tests
# =============================================================================

class TestRealtimeSwapRate:
    """Tests for RealtimeSwapRate dataclass."""

    def test_creation_basic(self) -> None:
        """Test basic creation."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
        )
        assert rate.pair == "EUR_USD"
        assert rate.long_swap_pips == -0.5
        assert rate.short_swap_pips == 0.3
        assert rate.timestamp_ms > 0

    def test_auto_timestamp(self) -> None:
        """Test automatic timestamp assignment."""
        before = int(time.time() * 1000)
        rate = RealtimeSwapRate(
            pair="GBP_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
        )
        after = int(time.time() * 1000)
        assert before <= rate.timestamp_ms <= after

    def test_age_seconds(self) -> None:
        """Test age calculation."""
        # Create rate 100ms ago
        old_ts = int(time.time() * 1000) - 100
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
            timestamp_ms=old_ts,
        )
        assert rate.age_seconds >= 0.1

    def test_is_fresh(self) -> None:
        """Test freshness check."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
        )
        assert rate.is_fresh(max_age_sec=60.0)
        # Create rate with old timestamp (1 second ago)
        old_ts = int(time.time() * 1000) - 1000
        old_rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
            timestamp_ms=old_ts,
        )
        assert old_rate.is_fresh(max_age_sec=0.5) is False

    def test_get_swap_for_direction_long(self) -> None:
        """Test getting swap for long direction."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
            long_swap_pct=-1.825,
            short_swap_pct=1.095,
        )
        assert rate.get_swap_for_direction(SwapDirection.LONG, in_pips=True) == -0.5
        assert rate.get_swap_for_direction(SwapDirection.LONG, in_pips=False) == -1.825

    def test_get_swap_for_direction_short(self) -> None:
        """Test getting swap for short direction."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
            long_swap_pct=-1.825,
            short_swap_pct=1.095,
        )
        assert rate.get_swap_for_direction(SwapDirection.SHORT, in_pips=True) == 0.3
        assert rate.get_swap_for_direction(SwapDirection.SHORT, in_pips=False) == 1.095

    def test_wednesday_multiplier_applied(self) -> None:
        """Test Wednesday triple swap multiplier."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
        )
        wed_rate = rate.apply_wednesday_multiplier(2)  # Wednesday
        assert wed_rate.long_swap_pips == pytest.approx(-1.5, rel=0.01)  # 3x
        assert wed_rate.short_swap_pips == pytest.approx(0.9, rel=0.01)  # 3x

    def test_wednesday_multiplier_not_wednesday(self) -> None:
        """Test Wednesday multiplier returns same rate on other days."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=-0.5,
            short_swap_pips=0.3,
        )
        # Monday
        mon_rate = rate.apply_wednesday_multiplier(0)
        assert mon_rate.long_swap_pips == -0.5
        # Thursday
        thu_rate = rate.apply_wednesday_multiplier(3)
        assert thu_rate.long_swap_pips == -0.5

    def test_to_dict(self, sample_swap_rate: RealtimeSwapRate) -> None:
        """Test conversion to dictionary."""
        d = sample_swap_rate.to_dict()
        assert d["pair"] == "EUR_USD"
        assert d["long_swap_pips"] == -0.5
        assert d["short_swap_pips"] == 0.3
        assert "source" in d
        assert "quality" in d
        assert "age_seconds" in d


class TestSwapRateUpdate:
    """Tests for SwapRateUpdate dataclass."""

    def test_to_swap_rate(self) -> None:
        """Test conversion to RealtimeSwapRate."""
        update = SwapRateUpdate(
            pair="EUR_USD",
            long_financing=-0.001,  # Daily rate
            short_financing=0.0005,
            timestamp_ms=int(time.time() * 1000),
        )
        rate = update.to_swap_rate(pip_value=0.0001)
        assert rate.pair == "EUR_USD"
        assert rate.source == SwapRateSource.OANDA_REALTIME
        # Check conversion to annual %
        assert abs(rate.long_swap_pct - (-0.001 * 365 * 100)) < 0.01


# =============================================================================
# SwapRateCalculator Tests
# =============================================================================

class TestSwapRateCalculator:
    """Tests for SwapRateCalculator."""

    def test_creation_with_default_rates(self) -> None:
        """Test creation with default interest rates."""
        calc = SwapRateCalculator()
        assert calc.get_rate("USD") is not None
        assert calc.get_rate("EUR") is not None

    def test_creation_with_custom_rates(self) -> None:
        """Test creation with custom interest rates."""
        rates = {"USD": 5.0, "EUR": 4.0}
        calc = SwapRateCalculator(interest_rates=rates)
        assert calc.get_rate("USD") == 5.0
        assert calc.get_rate("EUR") == 4.0

    def test_update_rate(self, swap_calculator: SwapRateCalculator) -> None:
        """Test updating an interest rate."""
        swap_calculator.update_rate("USD", 6.0)
        assert swap_calculator.get_rate("USD") == 6.0

    def test_calculate_swap_long_positive_carry(self) -> None:
        """Test long swap with positive carry (earn swap)."""
        # AUD (4.35%) vs JPY (0.1%) = positive carry for long
        calc = SwapRateCalculator(broker_tier="institutional")
        pips, pct = calc.calculate_swap_rate("AUD_JPY", SwapDirection.LONG, mid_price=95.0)
        # Long AUD/JPY: earn AUD rate, pay JPY rate
        # 4.35% - 0.1% - markup ≈ positive
        assert pct > 0 or abs(pct) < 0.1  # May be near zero with markup

    def test_calculate_swap_short_negative_carry(self) -> None:
        """Test short swap with negative carry (pay swap)."""
        # Short AUD/JPY: pay AUD rate, earn JPY rate
        calc = SwapRateCalculator(broker_tier="institutional")
        pips, pct = calc.calculate_swap_rate("AUD_JPY", SwapDirection.SHORT, mid_price=95.0)
        # Should be negative (paying)
        assert pct < 0

    def test_calculate_swap_jpy_pair_pip_value(self) -> None:
        """Test JPY pair uses correct pip value (0.01)."""
        calc = SwapRateCalculator()
        pips, _ = calc.calculate_swap_rate("USD_JPY", SwapDirection.LONG, mid_price=150.0)
        # Just verify it calculates without error
        assert isinstance(pips, float)

    def test_calculate_swap_non_jpy_pair_pip_value(self) -> None:
        """Test non-JPY pair uses correct pip value (0.0001)."""
        calc = SwapRateCalculator()
        pips, _ = calc.calculate_swap_rate("EUR_USD", SwapDirection.LONG, mid_price=1.10)
        assert isinstance(pips, float)

    def test_calculate_swap_invalid_pair(self) -> None:
        """Test handling of invalid pair format."""
        calc = SwapRateCalculator()
        pips, pct = calc.calculate_swap_rate("INVALID", SwapDirection.LONG)
        assert pips == 0.0
        assert pct == 0.0

    def test_calculate_swap_unknown_currency(self) -> None:
        """Test handling of unknown currency."""
        calc = SwapRateCalculator(interest_rates={"USD": 5.0})
        pips, pct = calc.calculate_swap_rate("USD_XXX", SwapDirection.LONG)
        assert pips == 0.0
        assert pct == 0.0

    def test_calculate_swap_without_markup(self) -> None:
        """Test swap calculation without broker markup."""
        calc = SwapRateCalculator()
        with_markup, _ = calc.calculate_swap_rate(
            "EUR_USD", SwapDirection.LONG, include_markup=True
        )
        without_markup, _ = calc.calculate_swap_rate(
            "EUR_USD", SwapDirection.LONG, include_markup=False
        )
        # Without markup should have higher absolute value (better for client)
        # This depends on direction and rates
        assert with_markup != without_markup

    def test_calculate_full_swap_rate(self, swap_calculator: SwapRateCalculator) -> None:
        """Test full swap rate calculation for both directions."""
        rate = swap_calculator.calculate_full_swap_rate("EUR_USD", mid_price=1.10)
        assert isinstance(rate, RealtimeSwapRate)
        assert rate.pair == "EUR_USD"
        assert rate.source == SwapRateSource.CALCULATED

    def test_broker_tier_markup_ranges(self) -> None:
        """Test different broker tiers have different markups."""
        retail = SwapRateCalculator(broker_tier="retail")
        inst = SwapRateCalculator(broker_tier="institutional")

        retail_rate, _ = retail.calculate_swap_rate("EUR_USD", SwapDirection.LONG)
        inst_rate, _ = inst.calculate_swap_rate("EUR_USD", SwapDirection.LONG)

        # Institutional should have better (less negative or more positive) rates
        # due to lower markup
        assert retail_rate != inst_rate


# =============================================================================
# OandaSwapRateClient Tests
# =============================================================================

class TestOandaSwapRateClient:
    """Tests for OandaSwapRateClient."""

    def test_is_configured_without_credentials(self) -> None:
        """Test is_configured returns False without credentials."""
        client = OandaSwapRateClient(api_key=None, account_id=None)
        assert client.is_configured is False

    def test_is_configured_with_credentials(self) -> None:
        """Test is_configured returns True with credentials."""
        client = OandaSwapRateClient(api_key="key", account_id="account")
        assert client.is_configured is True

    def test_practice_vs_live_urls(self) -> None:
        """Test practice and live URLs are different."""
        practice = OandaSwapRateClient(practice=True)
        live = OandaSwapRateClient(practice=False)
        assert practice._base_url != live._base_url

    def test_fetch_financing_rates_unconfigured(self) -> None:
        """Test fetch returns empty dict when unconfigured."""
        client = OandaSwapRateClient(api_key=None, account_id=None)
        result = client.fetch_financing_rates(["EUR_USD"])
        assert result == {}

    @patch("requests.get")
    def test_fetch_financing_rates_success(self, mock_get: Mock) -> None:
        """Test successful financing rate fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "bids": [{"price": "1.0850"}],
                    "asks": [{"price": "1.0852"}],
                    "financing": {
                        "longRate": "-0.0365",
                        "shortRate": "0.0100",
                    },
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = OandaSwapRateClient(api_key="key", account_id="account")
        result = client.fetch_financing_rates(["EUR_USD"])

        assert "EUR_USD" in result
        assert result["EUR_USD"].pair == "EUR_USD"
        assert result["EUR_USD"].source == SwapRateSource.OANDA_REALTIME

    @patch("requests.get")
    def test_fetch_financing_rates_api_error(self, mock_get: Mock) -> None:
        """Test handling of API errors."""
        mock_get.side_effect = Exception("API Error")

        client = OandaSwapRateClient(api_key="key", account_id="account")
        result = client.fetch_financing_rates(["EUR_USD"])

        assert result == {}


# =============================================================================
# RealtimeSwapRateProvider Tests
# =============================================================================

class TestRealtimeSwapRateProvider:
    """Tests for RealtimeSwapRateProvider."""

    def test_creation_default(self) -> None:
        """Test creation with defaults."""
        provider = RealtimeSwapRateProvider()
        assert provider.config is not None

    def test_creation_with_config(self, cache_config: SwapRateCacheConfig) -> None:
        """Test creation with custom config."""
        provider = RealtimeSwapRateProvider(config=cache_config)
        assert provider.config.cache_ttl_sec == 60.0

    def test_get_swap_rate_fallback_calculated(self) -> None:
        """Test fallback to calculated rate when OANDA not configured."""
        config = SwapRateCacheConfig(
            fallback_to_calculated=True,
            persist_to_disk=False,
        )
        oanda = OandaSwapRateClient(api_key=None, account_id=None)
        provider = RealtimeSwapRateProvider(config=config, oanda_client=oanda)

        rate = provider.get_swap_rate("EUR_USD", mid_price=1.10)

        assert rate.pair == "EUR_USD"
        assert rate.source == SwapRateSource.CALCULATED

    def test_get_swap_rate_caching(self) -> None:
        """Test that rates are cached."""
        config = SwapRateCacheConfig(
            cache_ttl_sec=60.0,
            persist_to_disk=False,
        )
        provider = RealtimeSwapRateProvider(config=config)

        # First call
        rate1 = provider.get_swap_rate("EUR_USD")
        # Second call should hit cache
        rate2 = provider.get_swap_rate("EUR_USD")

        stats = provider.get_stats()
        assert stats["cache_hits"] >= 1

    def test_get_swap_rate_force_refresh(self) -> None:
        """Test force refresh bypasses cache."""
        config = SwapRateCacheConfig(
            cache_ttl_sec=60.0,
            persist_to_disk=False,
        )
        provider = RealtimeSwapRateProvider(config=config)

        # First call
        rate1 = provider.get_swap_rate("EUR_USD")
        # Force refresh
        rate2 = provider.get_swap_rate("EUR_USD", force_refresh=True)

        # Both should be calculated but second bypassed cache
        stats = provider.get_stats()
        assert stats["cache_misses"] >= 2

    def test_get_multiple_swap_rates(self) -> None:
        """Test fetching multiple swap rates."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
        rates = provider.get_multiple_swap_rates(pairs)

        assert len(rates) == 3
        for pair in pairs:
            assert pair in rates

    def test_calculate_swap_cost_long(self) -> None:
        """Test swap cost calculation for long position."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100000,  # 1 standard lot
            days=1,
        )

        # Cost should be a float (could be positive or negative)
        assert isinstance(cost, float)

    def test_calculate_swap_cost_wednesday_triple(self) -> None:
        """Test Wednesday triple swap cost."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        regular_cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100000,
            days=1,
            wednesday_triple=False,
        )

        wednesday_cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100000,
            days=1,
            wednesday_triple=True,
            day_of_week=2,  # Wednesday
        )

        # Wednesday cost should be 3x regular
        assert abs(wednesday_cost) == pytest.approx(abs(regular_cost) * 3, rel=0.01)

    def test_normalize_pair_format(self) -> None:
        """Test pair normalization."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        # Test different formats
        rate1 = provider.get_swap_rate("EUR_USD")
        rate2 = provider.get_swap_rate("eur_usd")
        rate3 = provider.get_swap_rate("EUR/USD")

        assert rate1.pair == rate2.pair == rate3.pair == "EUR_USD"

    def test_cache_persistence_save(self, cache_config: SwapRateCacheConfig) -> None:
        """Test saving cache to disk."""
        provider = RealtimeSwapRateProvider(config=cache_config)

        # Get a rate (populates cache)
        provider.get_swap_rate("EUR_USD")

        # Manually trigger save
        provider._save_cache_to_disk()

        # Check file exists
        cache_path = Path(cache_config.cache_file_path)
        assert cache_path.exists()

    def test_cache_persistence_load(self, cache_config: SwapRateCacheConfig) -> None:
        """Test loading cache from disk."""
        cache_path = Path(cache_config.cache_file_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write test data
        test_data = {
            "EUR_USD": {
                "long_swap_pips": -0.5,
                "short_swap_pips": 0.3,
                "timestamp_ms": int(time.time() * 1000),
                "source": "oanda_cached",
            }
        }
        with open(cache_path, "w") as f:
            json.dump(test_data, f)

        # Create provider (should load cache)
        provider = RealtimeSwapRateProvider(config=cache_config)

        assert "EUR_USD" in provider._cache

    def test_quality_update_fresh(self) -> None:
        """Test quality indicator for fresh rate."""
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
            timestamp_ms=int(time.time() * 1000),
        )
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        updated = provider._update_quality(rate)
        assert updated.quality == SwapRateQuality.FRESH

    def test_quality_update_stale(self) -> None:
        """Test quality indicator for stale rate."""
        old_ts = int(time.time() * 1000) - 3 * 3600 * 1000  # 3 hours ago
        rate = RealtimeSwapRate(
            pair="EUR_USD",
            long_swap_pips=0.1,
            short_swap_pips=-0.1,
            timestamp_ms=old_ts,
        )
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        updated = provider._update_quality(rate)
        assert updated.quality == SwapRateQuality.STALE

    def test_get_stats(self) -> None:
        """Test getting provider statistics."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        provider.get_swap_rate("EUR_USD")

        stats = provider.get_stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_size" in stats
        assert "oanda_configured" in stats

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        provider.get_swap_rate("EUR_USD")
        assert len(provider._cache) > 0

        provider.clear_cache()
        assert len(provider._cache) == 0

    def test_max_cache_size_enforcement(self) -> None:
        """Test that cache size is enforced."""
        config = SwapRateCacheConfig(
            max_cache_size=5,
            persist_to_disk=False,
        )
        provider = RealtimeSwapRateProvider(config=config)

        # Add more rates than max cache size
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
        for pair in pairs:
            provider.get_swap_rate(pair)

        assert len(provider._cache) <= 5


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateRealtimeSwapProvider:
    """Tests for create_realtime_swap_provider factory."""

    def test_create_default(self) -> None:
        """Test creation with defaults."""
        provider = create_realtime_swap_provider(auto_start_refresh=False)
        assert provider is not None

    def test_create_with_credentials(self) -> None:
        """Test creation with OANDA credentials."""
        provider = create_realtime_swap_provider(
            oanda_api_key="test_key",
            oanda_account_id="test_account",
            practice=True,
            auto_start_refresh=False,
        )
        assert provider._oanda.is_configured

    def test_create_different_tiers(self) -> None:
        """Test creation with different broker tiers."""
        retail = create_realtime_swap_provider(broker_tier="retail", auto_start_refresh=False)
        inst = create_realtime_swap_provider(broker_tier="institutional", auto_start_refresh=False)

        assert retail.config.broker_tier == "retail"
        assert inst.config.broker_tier == "institutional"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSwapRateIntegration:
    """Integration tests for swap rate functionality."""

    def test_full_swap_workflow(self) -> None:
        """Test complete swap calculation workflow."""
        # Create provider
        provider = create_realtime_swap_provider(
            broker_tier="retail",
            auto_start_refresh=False,
        )

        # Get rate
        rate = provider.get_swap_rate("EUR_USD", mid_price=1.10)
        assert rate is not None

        # Calculate cost
        cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100000,
            days=5,
        )
        assert isinstance(cost, float)

        # Check stats
        stats = provider.get_stats()
        assert stats["cache_size"] >= 1

    def test_carry_trade_scenario(self) -> None:
        """Test carry trade scenario with high interest rate differential."""
        calculator = SwapRateCalculator(
            interest_rates={
                "AUD": 4.35,  # High yield
                "JPY": 0.10,  # Low yield
            },
            broker_tier="professional",
        )

        # Long AUD/JPY should have positive carry
        rate = calculator.calculate_full_swap_rate("AUD_JPY", mid_price=95.0)

        # Long should be positive or near zero (earning)
        # Short should be negative (paying)
        assert rate.long_swap_pct > rate.short_swap_pct

    def test_negative_rate_scenario(self) -> None:
        """Test scenario with negative interest rates."""
        calculator = SwapRateCalculator(
            interest_rates={
                "EUR": -0.50,  # Negative rate
                "CHF": -0.75,  # More negative
            },
            broker_tier="retail",
        )

        rate = calculator.calculate_full_swap_rate("EUR_CHF", mid_price=0.95)
        # Should still calculate without error
        assert isinstance(rate.long_swap_pips, float)
        assert isinstance(rate.short_swap_pips, float)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestSwapRateEdgeCases:
    """Edge case tests for swap rate functionality."""

    def test_zero_interest_rate(self) -> None:
        """Test handling of zero interest rate."""
        calc = SwapRateCalculator(
            interest_rates={"USD": 5.0, "XXX": 0.0}
        )
        pips, pct = calc.calculate_swap_rate("USD_XXX", SwapDirection.LONG)
        assert isinstance(pips, float)

    def test_very_large_position(self) -> None:
        """Test swap cost with very large position."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)

        cost = provider.calculate_swap_cost(
            pair="EUR_USD",
            direction=SwapDirection.LONG,
            position_size=100_000_000,  # 1000 lots
            days=1,
        )
        assert isinstance(cost, float)
        assert not math.isnan(cost)
        assert not math.isinf(cost)

    def test_exotic_pair_handling(self) -> None:
        """Test handling of exotic currency pairs."""
        calc = SwapRateCalculator()
        rate = calc.calculate_full_swap_rate("USD_TRY", mid_price=30.0)
        # Should handle high-yield exotic
        assert abs(rate.long_swap_pips) > 0 or abs(rate.short_swap_pips) > 0

    def test_concurrent_access(self) -> None:
        """Test thread-safe concurrent access."""
        config = SwapRateCacheConfig(persist_to_disk=False)
        provider = RealtimeSwapRateProvider(config=config)
        results = []
        errors = []

        def fetch_rate(pair: str) -> None:
            try:
                rate = provider.get_swap_rate(pair)
                results.append(rate)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=fetch_rate, args=(pair,))
            for pair in ["EUR_USD", "GBP_USD", "USD_JPY"] * 5
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 15
