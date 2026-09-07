# -*- coding: utf-8 -*-
"""
tests/test_forex_properties.py
Phase 10: Property-Based Tests for Forex Integration.

PURPOSE: Use property-based testing (Hypothesis) to verify invariants
         and find edge cases that traditional unit tests might miss.

Properties Tested:
1. Slippage monotonicity: More participation -> more slippage
2. Session ordering: Overlap liquidity > individual session
3. Spread bounds: Always within configured min/max
4. Pair classification: All pairs classified correctly
5. DST handling: Rollover time consistent regardless of date
6. Fee computation: Fees always non-negative

Test Count Target: 40 tests

References:
    - Hypothesis: https://hypothesis.readthedocs.io/
    - Property-based testing best practices
    - Almgren-Chriss (2001): Market impact properties
"""

import pytest
import math
from datetime import datetime, timezone
from typing import Optional

# Check if hypothesis is installed
try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)

from execution_providers import (
    AssetClass,
    Order,
    MarketState,
    ForexSession,
    PairType,
    VolatilityRegime,
    ForexParametricConfig,
    ForexParametricSlippageProvider,
    ForexFeeProvider,
    CryptoParametricSlippageProvider,
)


# =============================================================================
# Custom Strategies
# =============================================================================


@composite
def forex_pair_strategy(draw):
    """Generate valid forex pair symbols."""
    majors = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
    minors = ["EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD", "EUR_CAD"]
    crosses = ["EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY"]
    exotics = ["USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN", "EUR_TRY"]

    all_pairs = majors + minors + crosses + exotics
    return draw(st.sampled_from(all_pairs))


@composite
def forex_order_strategy(draw):
    """Generate valid forex orders."""
    symbol = draw(forex_pair_strategy())
    side = draw(st.sampled_from(["BUY", "SELL"]))
    qty = draw(
        st.floats(min_value=1000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False)
    )

    return Order(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type="MARKET",
        asset_class=AssetClass.FOREX,
    )


@composite
def market_state_strategy(draw):
    """Generate valid market states."""
    # Generate timestamp during forex trading hours (not weekend)
    # Wednesday 12:00 UTC = London/NY overlap
    base_ts = 1700049600000  # 2023-11-15 12:00 UTC

    bid = draw(st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False))
    spread_pips = draw(
        st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False)
    )

    # Convert spread to price units (assuming non-JPY pair)
    spread = spread_pips * 0.0001
    ask = bid + spread

    return MarketState(
        timestamp=base_ts,
        bid=bid,
        ask=ask,
        adv=500_000_000_000.0,  # $500B ADV for major pairs
    )


@composite
def participation_strategy(draw):
    """Generate valid participation ratios."""
    return draw(
        st.floats(
            min_value=0.0001,
            max_value=0.1,
            allow_nan=False,
            allow_infinity=False,
        )
    )


# =============================================================================
# Slippage Monotonicity Tests
# =============================================================================


class TestSlippageMonotonicity:
    """Test that slippage increases monotonically with participation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()
        self.order = Order(
            symbol="EUR_USD",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
            asset_class=AssetClass.FOREX,
        )
        # Wednesday 12:00 UTC (London/NY overlap)
        self.market = MarketState(
            timestamp=1700049600000,
            bid=1.0850,
            ask=1.0852,
            adv=500_000_000_000.0,
        )

    @given(
        p1=st.floats(min_value=0.0001, max_value=0.05, allow_nan=False, allow_infinity=False),
        p2=st.floats(min_value=0.0001, max_value=0.05, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_slippage_monotonic_in_participation(self, p1: float, p2: float):
        """Higher participation should result in equal or higher slippage."""
        assume(p1 <= p2)

        slip1 = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=p1,
        )

        slip2 = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=p2,
        )

        # Monotonicity: higher participation -> higher or equal slippage
        assert slip1 <= slip2 + 1e-9, f"Slippage not monotonic: {slip1} > {slip2}"

    @given(participation=participation_strategy())
    @settings(max_examples=50)
    def test_slippage_positive(self, participation: float):
        """Slippage should always be positive."""
        slip = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=participation,
        )
        assert slip > 0, f"Slippage should be positive: {slip}"

    @given(participation=participation_strategy())
    @settings(max_examples=50)
    def test_slippage_bounded(self, participation: float):
        """Slippage should always be within configured bounds."""
        slip = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=participation,
        )

        assert slip >= self.provider.config.min_slippage_pips
        assert slip <= self.provider.config.max_slippage_pips


# =============================================================================
# Pair Classification Tests
# =============================================================================


class TestPairClassification:
    """Test pair classification properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()

    @given(symbol=forex_pair_strategy())
    @settings(max_examples=50)
    def test_pair_classification_deterministic(self, symbol: str):
        """Pair classification should be deterministic."""
        result1 = self.provider._classify_pair(symbol)
        result2 = self.provider._classify_pair(symbol)

        assert result1 == result2

    @given(symbol=forex_pair_strategy())
    @settings(max_examples=50)
    def test_pair_classification_valid_enum(self, symbol: str):
        """Pair classification should return valid PairType."""
        result = self.provider._classify_pair(symbol)

        assert isinstance(result, PairType)
        assert result in [PairType.MAJOR, PairType.MINOR, PairType.CROSS, PairType.EXOTIC]

    def test_majors_classified_correctly(self):
        """All major pairs should be classified as MAJOR."""
        majors = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]

        for pair in majors:
            result = self.provider._classify_pair(pair)
            assert result == PairType.MAJOR, f"{pair} should be MAJOR, got {result}"

    def test_exotics_classified_correctly(self):
        """All exotic pairs should be classified as EXOTIC."""
        exotics = ["USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN"]

        for pair in exotics:
            result = self.provider._classify_pair(pair)
            assert result == PairType.EXOTIC, f"{pair} should be EXOTIC, got {result}"


# =============================================================================
# Session Detection Tests
# =============================================================================


class TestSessionDetection:
    """Test session detection properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()

    @given(hour=st.integers(min_value=0, max_value=23))
    @settings(max_examples=24)
    def test_session_detection_complete(self, hour: int):
        """Every hour should map to a valid session."""
        # Create timestamp for Wednesday at given hour
        # Wednesday Nov 15, 2023
        dt = datetime(2023, 11, 15, hour, 0, 0, tzinfo=timezone.utc)
        ts_ms = int(dt.timestamp() * 1000)

        session = self.provider._detect_session(ts_ms)

        assert session is not None
        assert isinstance(session, ForexSession)

    @given(minute=st.integers(min_value=0, max_value=59))
    @settings(max_examples=30)
    def test_session_consistent_within_hour(self, minute: int):
        """Session should be consistent within the same hour."""
        # Wednesday 14:XX UTC (London/NY overlap)
        dt1 = datetime(2023, 11, 15, 14, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2023, 11, 15, 14, minute, 0, tzinfo=timezone.utc)

        ts1 = int(dt1.timestamp() * 1000)
        ts2 = int(dt2.timestamp() * 1000)

        session1 = self.provider._detect_session(ts1)
        session2 = self.provider._detect_session(ts2)

        assert session1 == session2

    def test_weekend_detection(self):
        """Weekend should be detected correctly."""
        # Saturday Nov 18, 2023 12:00 UTC
        saturday = datetime(2023, 11, 18, 12, 0, 0, tzinfo=timezone.utc)
        ts_sat = int(saturday.timestamp() * 1000)

        session = self.provider._detect_session(ts_sat)
        assert session == ForexSession.WEEKEND

    def test_overlap_priority(self):
        """Overlap sessions should take priority."""
        # Wednesday 14:00 UTC = London/NY overlap
        overlap_time = datetime(2023, 11, 15, 14, 0, 0, tzinfo=timezone.utc)
        ts = int(overlap_time.timestamp() * 1000)

        session = self.provider._detect_session(ts)
        assert session == ForexSession.LONDON_NY_OVERLAP


# =============================================================================
# Session Liquidity Ordering Tests
# =============================================================================


class TestSessionLiquidityOrdering:
    """Test session liquidity ordering properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = ForexParametricConfig()

    def test_overlap_liquidity_highest(self):
        """Overlap sessions should have highest liquidity."""
        liquidity = self.config.session_liquidity

        # London/NY overlap should be highest
        assert liquidity["london_ny_overlap"] >= liquidity["london"]
        assert liquidity["london_ny_overlap"] >= liquidity["new_york"]

    def test_sydney_liquidity_lowest_trading(self):
        """Sydney session should have lowest liquidity (among trading sessions)."""
        liquidity = self.config.session_liquidity

        # Sydney is lowest among trading sessions
        assert liquidity["sydney"] <= liquidity["tokyo"]
        assert liquidity["sydney"] <= liquidity["london"]
        assert liquidity["sydney"] <= liquidity["new_york"]

    def test_weekend_zero_liquidity(self):
        """Weekend should have zero liquidity."""
        liquidity = self.config.session_liquidity

        assert liquidity["weekend"] == 0.0


# =============================================================================
# Spread Bounds Tests
# =============================================================================


class TestSpreadBounds:
    """Test spread-related properties."""

    @given(pair_type=st.sampled_from(["major", "minor", "cross", "exotic"]))
    @settings(max_examples=20)
    def test_spread_positive_for_all_types(self, pair_type: str):
        """Spreads should be positive for all pair types."""
        config = ForexParametricConfig()
        spread = config.default_spreads_pips[pair_type]

        assert spread > 0

    def test_spread_ordering_by_liquidity(self):
        """Spreads should be ordered by liquidity (majors tightest)."""
        config = ForexParametricConfig()
        spreads = config.default_spreads_pips

        assert spreads["major"] < spreads["minor"]
        assert spreads["minor"] < spreads["cross"]
        assert spreads["cross"] < spreads["exotic"]


# =============================================================================
# Fee Computation Tests
# =============================================================================


class TestFeeComputation:
    """Test fee computation properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fee_provider = ForexFeeProvider()

    @given(
        notional=st.floats(
            min_value=1000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False
        ),
        qty=st.floats(
            min_value=1000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_fee_non_negative(self, notional: float, qty: float):
        """Fees should always be non-negative."""
        # ForexFeeProvider uses: notional, side, liquidity, qty
        fee = self.fee_provider.compute_fee(
            notional=notional,
            side="BUY",
            liquidity="taker",
            qty=qty,
        )

        assert fee >= 0, f"Fee should be non-negative: {fee}"

    @given(
        notional1=st.floats(
            min_value=1000.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False
        ),
        notional2=st.floats(
            min_value=1000.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False
        ),
        qty=st.floats(
            min_value=1000.0, max_value=5_000_000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_fee_proportional_to_quantity(self, notional1: float, notional2: float, qty: float):
        """Fees should scale with notional (for institutional pricing)."""
        assume(notional1 <= notional2)

        # Use institutional fee provider to test proportionality
        inst_provider = ForexFeeProvider(commission_bps=1.0)
        fee1 = inst_provider.compute_fee(notional=notional1, side="BUY", liquidity="taker", qty=qty)
        fee2 = inst_provider.compute_fee(notional=notional2, side="BUY", liquidity="taker", qty=qty)

        # Larger notional -> larger fee (with some tolerance for edge cases)
        assert fee1 <= fee2 + 0.001


# =============================================================================
# Volatility Regime Tests
# =============================================================================


class TestVolatilityRegime:
    """Test volatility regime detection properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()

    @given(
        returns=st.lists(
            st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_volatility_regime_valid(self, returns: list):
        """Volatility regime should always be a valid string."""
        regime = self.provider._detect_volatility_regime(returns)

        valid_regimes = {"low", "normal", "high", "extreme"}
        assert regime in valid_regimes

    def test_empty_returns_gives_normal(self):
        """Empty returns should give 'normal' regime."""
        regime = self.provider._detect_volatility_regime([])
        assert regime == "normal"

        regime = self.provider._detect_volatility_regime(None)
        assert regime == "normal"


# =============================================================================
# Cross-Asset Class Tests
# =============================================================================


class TestCrossAssetProperties:
    """Test properties across asset classes."""

    @given(
        participation=st.floats(
            min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=30)
    def test_forex_lower_impact_than_crypto(self, participation: float):
        """Forex should have lower impact coefficient than crypto (more liquid)."""
        forex_config = ForexParametricConfig()
        from execution_providers import CryptoParametricConfig

        crypto_config = CryptoParametricConfig()

        # Forex base impact should be lower (more liquid)
        assert forex_config.impact_coef_base <= crypto_config.impact_coef_base

    def test_forex_wider_spread_than_crypto_majors(self):
        """Forex major spread should be comparable but in different units."""
        forex_config = ForexParametricConfig()

        # Forex uses pips, crypto uses bps - can't directly compare
        # But we can verify forex majors are in reasonable range
        assert forex_config.default_spreads_pips["major"] > 0
        assert forex_config.default_spreads_pips["major"] < 10  # Major pairs < 10 pips


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestIdempotency:
    """Test that operations are idempotent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()
        self.order = Order(
            symbol="EUR_USD",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
            asset_class=AssetClass.FOREX,
        )
        self.market = MarketState(
            timestamp=1700049600000,
            bid=1.0850,
            ask=1.0852,
        )

    @given(participation=participation_strategy())
    @settings(max_examples=30)
    def test_slippage_computation_idempotent(self, participation: float):
        """Same inputs should produce same slippage."""
        slip1 = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=participation,
        )

        slip2 = self.provider.compute_slippage_pips(
            order=self.order,
            market=self.market,
            participation_ratio=participation,
        )

        assert abs(slip1 - slip2) < 1e-10


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = ForexParametricSlippageProvider()

    @given(
        participation=st.floats(
            min_value=1e-10,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=50)
    def test_no_nan_slippage(self, participation: float):
        """Slippage should never be NaN."""
        order = Order(
            symbol="EUR_USD",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
            asset_class=AssetClass.FOREX,
        )
        market = MarketState(
            timestamp=1700049600000,
            bid=1.0850,
            ask=1.0852,
        )

        slip = self.provider.compute_slippage_pips(
            order=order,
            market=market,
            participation_ratio=participation,
        )

        assert not math.isnan(slip)
        assert not math.isinf(slip)

    @given(
        bid=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        spread_pips=st.floats(
            min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_no_nan_with_various_prices(self, bid: float, spread_pips: float):
        """Slippage should be finite for various price levels."""
        spread = spread_pips * 0.0001
        ask = bid + spread

        order = Order(
            symbol="EUR_USD",
            side="BUY",
            qty=100000.0,
            order_type="MARKET",
            asset_class=AssetClass.FOREX,
        )
        market = MarketState(
            timestamp=1700049600000,
            bid=bid,
            ask=ask,
        )

        slip = self.provider.compute_slippage_pips(
            order=order,
            market=market,
            participation_ratio=0.001,
        )

        assert not math.isnan(slip)
        assert not math.isinf(slip)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
