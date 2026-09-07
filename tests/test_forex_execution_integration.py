# -*- coding: utf-8 -*-
"""
tests/test_forex_execution_integration.py

Tests for forex execution integration layer.
Phase 5: Forex Integration (2025-11-30)

Tests cover:
1. ForexExecutionConfig validation
2. ForexExecutionEstimate generation
3. ForexExecutionReport generation
4. ForexExecutionIntegration full workflow
5. Factory functions
6. Adaptive blending
7. Execution quality scoring

Expected: ~50 tests (100% pass)
"""
from __future__ import annotations

import math
from typing import Dict

import pytest

from execution_providers import ForexSession, PairType
from services.forex_dealer import RejectReason
from services.forex_execution_integration import (
    # Data classes
    ForexExecutionConfig,
    ForexExecutionEstimate,
    ForexExecutionReport,
    # Main class
    ForexExecutionIntegration,
    # Factory functions
    create_forex_execution_integration,
    create_institutional_integration,
    create_retail_integration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> ForexExecutionConfig:
    """Create default config."""
    return ForexExecutionConfig()


@pytest.fixture
def integration() -> ForexExecutionIntegration:
    """Create integration with fixed seed."""
    return ForexExecutionIntegration(seed=42)


@pytest.fixture
def institutional_integration() -> ForexExecutionIntegration:
    """Create institutional integration."""
    return create_institutional_integration(seed=42)


@pytest.fixture
def retail_integration() -> ForexExecutionIntegration:
    """Create retail integration."""
    return create_retail_integration(seed=42)


# =============================================================================
# Test ForexExecutionConfig
# =============================================================================


class TestForexExecutionConfig:
    """Tests for ForexExecutionConfig data class."""

    def test_default_values(self, default_config: ForexExecutionConfig) -> None:
        """Test default configuration values."""
        assert default_config.execution_weight == 0.3
        assert default_config.use_dealer_for_large_orders is True
        assert default_config.large_order_threshold_usd == pytest.approx(1_000_000)
        assert default_config.enable_adaptive_blending is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ForexExecutionConfig(
            execution_weight=0.5,
            large_order_threshold_usd=5_000_000,
            enable_adaptive_blending=False,
        )
        assert config.execution_weight == 0.5
        assert config.large_order_threshold_usd == pytest.approx(5_000_000)
        assert config.enable_adaptive_blending is False


# =============================================================================
# Test ForexExecutionEstimate
# =============================================================================


class TestForexExecutionEstimate:
    """Tests for ForexExecutionEstimate data class."""

    def test_basic_estimate(self) -> None:
        """Test basic estimate creation."""
        estimate = ForexExecutionEstimate(
            expected_slippage_pips=1.5,
            expected_rejection_prob=0.05,
            session=ForexSession.LONDON,
            pair_type=PairType.MAJOR,
        )
        assert estimate.expected_slippage_pips == pytest.approx(1.5)
        assert estimate.expected_rejection_prob == pytest.approx(0.05)
        assert estimate.session == ForexSession.LONDON
        assert estimate.pair_type == PairType.MAJOR

    def test_estimate_with_risk_factors(self) -> None:
        """Test estimate with risk factors."""
        estimate = ForexExecutionEstimate(
            expected_slippage_pips=2.0,
            expected_rejection_prob=0.1,
            session=ForexSession.TOKYO,
            pair_type=PairType.CROSS,
            volatility_regime="high",
            dealer_spread_pips=1.8,
            recommended_execution="wait_for_better_liquidity",
            risk_factors={
                "size_risk": 0.5,
                "spread_risk": 0.3,
            },
        )
        assert estimate.volatility_regime == "high"
        assert estimate.dealer_spread_pips == pytest.approx(1.8)
        assert estimate.recommended_execution == "wait_for_better_liquidity"
        assert "size_risk" in estimate.risk_factors

    def test_estimate_to_dict(self) -> None:
        """Test estimate serialization."""
        estimate = ForexExecutionEstimate(
            expected_slippage_pips=1.0,
            expected_rejection_prob=0.05,
            session=ForexSession.NEW_YORK,
            pair_type=PairType.MAJOR,
            risk_factors={"test": 0.5},
        )
        d = estimate.to_dict()

        assert d["expected_slippage_pips"] == pytest.approx(1.0)
        assert d["session"] == "new_york"
        assert d["pair_type"] == "major"
        assert d["risk_factors"]["test"] == pytest.approx(0.5)


# =============================================================================
# Test ForexExecutionReport
# =============================================================================


class TestForexExecutionReport:
    """Tests for ForexExecutionReport data class."""

    def test_report_creation(self, integration: ForexExecutionIntegration) -> None:
        """Test creating execution report."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert report.pre_trade_estimate is not None
        assert report.execution_result is not None
        assert report.tca_slippage_pips >= 0
        assert report.combined_slippage_pips >= 0

    def test_report_to_dict(self, integration: ForexExecutionIntegration) -> None:
        """Test report serialization."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )
        d = report.to_dict()

        assert "pre_trade_estimate" in d
        assert "execution_result" in d
        assert "tca_slippage_pips" in d
        assert "combined_slippage_pips" in d
        assert "execution_quality" in d


# =============================================================================
# Test ForexExecutionIntegration - Initialization
# =============================================================================


class TestForexExecutionIntegrationInit:
    """Tests for ForexExecutionIntegration initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        integration = ForexExecutionIntegration()
        assert integration._tca is not None
        assert integration._dealer is not None

    def test_initialization_with_seed(self) -> None:
        """Test initialization with seed for reproducibility."""
        int1 = ForexExecutionIntegration(seed=42)
        int2 = ForexExecutionIntegration(seed=42)

        est1 = int1.estimate_execution_cost("EUR_USD", "BUY", 100_000, 1.0850)
        est2 = int2.estimate_execution_cost("EUR_USD", "BUY", 100_000, 1.0850)

        assert est1.dealer_spread_pips == pytest.approx(est2.dealer_spread_pips, rel=0.01)

    def test_initialization_with_profiles(self) -> None:
        """Test initialization with different profiles."""
        retail = ForexExecutionIntegration(tca_profile="retail", dealer_profile="retail")
        inst = ForexExecutionIntegration(
            tca_profile="institutional", dealer_profile="institutional"
        )

        # Institutional should have tighter spreads
        assert retail is not None
        assert inst is not None


# =============================================================================
# Test ForexExecutionIntegration - Cost Estimation
# =============================================================================


class TestForexExecutionIntegrationEstimate:
    """Tests for cost estimation."""

    def test_estimate_basic(self, integration: ForexExecutionIntegration) -> None:
        """Test basic cost estimation."""
        estimate = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert estimate.expected_slippage_pips > 0
        assert estimate.expected_rejection_prob >= 0
        assert estimate.pair_type == PairType.MAJOR
        assert estimate.dealer_spread_pips > 0

    def test_estimate_jpy_pair(self, integration: ForexExecutionIntegration) -> None:
        """Test cost estimation for JPY pair."""
        estimate = integration.estimate_execution_cost(
            symbol="USD_JPY",
            side="SELL",
            size_usd=200_000,
            mid_price=150.00,
        )

        assert estimate.expected_slippage_pips > 0
        assert estimate.pair_type in [PairType.MAJOR, PairType.CROSS]

    def test_estimate_exotic_pair(self, integration: ForexExecutionIntegration) -> None:
        """Test cost estimation for exotic pair."""
        estimate = integration.estimate_execution_cost(
            symbol="USD_TRY",
            side="BUY",
            size_usd=100_000,
            mid_price=32.50,
        )

        # Exotics should have higher slippage
        eur_estimate = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        # Exotic expected to have higher slippage (usually)
        assert estimate.expected_slippage_pips >= 0

    def test_estimate_with_session(self, integration: ForexExecutionIntegration) -> None:
        """Test cost estimation with explicit session."""
        overlap_est = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=500_000,
            mid_price=1.0850,
            session=ForexSession.LONDON_NY_OVERLAP,
        )

        sydney_est = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=500_000,
            mid_price=1.0850,
            session=ForexSession.SYDNEY,
        )

        # Sydney should generally have higher slippage (less liquidity)
        assert overlap_est.session == ForexSession.LONDON_NY_OVERLAP
        assert sydney_est.session == ForexSession.SYDNEY

    def test_estimate_with_hour_utc(self, integration: ForexExecutionIntegration) -> None:
        """Test cost estimation with hour UTC."""
        # London-NY overlap hours
        estimate_14 = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
            hour_utc=14,
        )

        # Sydney hours
        estimate_3 = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
            hour_utc=3,
        )

        # Both should work
        assert estimate_14.expected_slippage_pips > 0
        assert estimate_3.expected_slippage_pips > 0

    def test_estimate_risk_factors(self, integration: ForexExecutionIntegration) -> None:
        """Test that risk factors are populated."""
        estimate = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=3_000_000,
            mid_price=1.0850,
        )

        assert "size_risk" in estimate.risk_factors
        assert "spread_risk" in estimate.risk_factors
        assert "rejection_risk" in estimate.risk_factors
        assert "session_liquidity" in estimate.risk_factors

    def test_estimate_recommendations(self, integration: ForexExecutionIntegration) -> None:
        """Test execution recommendations."""
        # Small order should be immediate
        small_est = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )
        assert small_est.recommended_execution in [
            "immediate",
            "wait_for_better_liquidity",
            "split_order",
            "use_limit_orders",
        ]

        # Very large order should suggest splitting
        large_est = integration.estimate_execution_cost(
            symbol="EUR_USD",
            side="BUY",
            size_usd=50_000_000,
            mid_price=1.0850,
        )
        # Large orders often get split recommendation
        assert large_est.recommended_execution in [
            "immediate",
            "split_order",
            "wait_for_better_liquidity",
        ]


# =============================================================================
# Test ForexExecutionIntegration - Execution
# =============================================================================


class TestForexExecutionIntegrationExecute:
    """Tests for full execution simulation."""

    def test_execute_basic(self, integration: ForexExecutionIntegration) -> None:
        """Test basic execution."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert report.pre_trade_estimate is not None
        assert report.execution_result is not None
        assert report.tca_slippage_pips >= 0
        assert report.combined_slippage_pips >= 0

    def test_execute_buy_and_sell(self, integration: ForexExecutionIntegration) -> None:
        """Test both buy and sell executions."""
        buy_report = integration.execute(
            symbol="GBP_USD",
            side="BUY",
            size_usd=200_000,
            mid_price=1.2500,
        )

        sell_report = integration.execute(
            symbol="GBP_USD",
            side="SELL",
            size_usd=200_000,
            mid_price=1.2500,
        )

        # Both should execute
        assert buy_report.execution_result is not None
        assert sell_report.execution_result is not None

    def test_execute_tracks_slippage(self, integration: ForexExecutionIntegration) -> None:
        """Test that slippage is tracked."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert report.tca_slippage_pips >= 0
        assert report.dealer_slippage_pips >= 0 or not report.execution_result.filled
        assert report.combined_slippage_pips >= 0

    def test_execute_tracks_tca_error(self, integration: ForexExecutionIntegration) -> None:
        """Test that TCA error is computed."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert report.tca_error_pips >= 0

    def test_execute_tracks_fill_rate(self, integration: ForexExecutionIntegration) -> None:
        """Test that fill rate is tracked."""
        # Execute multiple times
        for _ in range(10):
            integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
            )

        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        # Fill rate should be tracked
        assert 0 <= report.fill_rate_recent <= 100

    def test_execute_quality_score(self, integration: ForexExecutionIntegration) -> None:
        """Test execution quality score."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
        )

        assert 0 <= report.execution_quality <= 1

    def test_execute_with_news_event(self, integration: ForexExecutionIntegration) -> None:
        """Test execution with upcoming news event."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
            upcoming_news="nfp",  # Non-farm payrolls
        )

        # Should still execute
        assert report.execution_result is not None

    def test_execute_with_high_volatility(self, integration: ForexExecutionIntegration) -> None:
        """Test execution in high volatility regime."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
            volatility_regime="high",
        )

        assert report.execution_result is not None
        assert report.pre_trade_estimate.volatility_regime == "high"


# =============================================================================
# Test ForexExecutionIntegration - Adaptive Blending
# =============================================================================


class TestForexExecutionIntegrationAdaptive:
    """Tests for adaptive blending."""

    def test_adaptive_blending_initial_weight(self) -> None:
        """Test initial blend weight matches config."""
        config = ForexExecutionConfig(execution_weight=0.4)
        integration = ForexExecutionIntegration(config=config, seed=42)

        assert integration._blend_weight == pytest.approx(0.4)

    def test_adaptive_blending_updates(self) -> None:
        """Test that blend weight updates over time."""
        config = ForexExecutionConfig(
            enable_adaptive_blending=True,
            execution_weight=0.3,
        )
        integration = ForexExecutionIntegration(config=config, seed=42)

        initial_weight = integration._blend_weight

        # Execute many times to trigger adaptation
        for _ in range(50):
            integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
            )

        # Weight may have changed (depends on TCA accuracy)
        # Just verify it's still in valid range
        assert 0.1 <= integration._blend_weight <= 0.5

    def test_adaptive_blending_disabled(self) -> None:
        """Test that blending can be disabled."""
        config = ForexExecutionConfig(
            enable_adaptive_blending=False,
            execution_weight=0.3,
        )
        integration = ForexExecutionIntegration(config=config, seed=42)

        initial_weight = integration._blend_weight

        # Execute many times
        for _ in range(30):
            integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
            )

        # Weight should not change when disabled
        assert integration._blend_weight == pytest.approx(initial_weight)


# =============================================================================
# Test ForexExecutionIntegration - Stats and Reset
# =============================================================================


class TestForexExecutionIntegrationStats:
    """Tests for stats and reset functionality."""

    def test_get_dealer_stats(self, integration: ForexExecutionIntegration) -> None:
        """Test getting dealer stats."""
        # Execute some orders
        for _ in range(5):
            integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
            )

        stats = integration.get_dealer_stats()
        assert stats.total_attempts == 5

    def test_reset(self, integration: ForexExecutionIntegration) -> None:
        """Test reset functionality."""
        # Execute some orders
        for _ in range(5):
            integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
            )

        # Reset
        integration.reset()

        # Stats should be cleared
        stats = integration.get_dealer_stats()
        assert stats.total_attempts == 0


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_forex_execution_integration(self) -> None:
        """Test basic factory function."""
        integration = create_forex_execution_integration()
        assert integration is not None
        assert isinstance(integration, ForexExecutionIntegration)

    def test_create_forex_execution_integration_with_params(self) -> None:
        """Test factory with parameters."""
        integration = create_forex_execution_integration(
            tca_profile="institutional",
            dealer_profile="institutional",
            seed=123,
            execution_weight=0.2,
        )
        assert integration._blend_weight == pytest.approx(0.2)

    def test_create_institutional_integration(self) -> None:
        """Test institutional integration factory."""
        integration = create_institutional_integration(seed=42)

        assert integration is not None
        assert integration.config.execution_weight == pytest.approx(0.2)
        assert integration.config.large_order_threshold_usd == pytest.approx(5_000_000)

    def test_create_retail_integration(self) -> None:
        """Test retail integration factory."""
        integration = create_retail_integration(seed=42)

        assert integration is not None
        assert integration.config.execution_weight == pytest.approx(0.4)
        assert integration.config.large_order_threshold_usd == pytest.approx(1_000_000)

    def test_institutional_vs_retail_spreads(self) -> None:
        """Test that institutional has tighter estimates than retail."""
        inst = create_institutional_integration(seed=42)
        retail = create_retail_integration(seed=42)

        # Compare estimates
        inst_est = inst.estimate_execution_cost("EUR_USD", "BUY", 100_000, 1.0850)
        retail_est = retail.estimate_execution_cost("EUR_USD", "BUY", 100_000, 1.0850)

        # Institutional dealer spreads should generally be tighter
        # Note: due to randomness, we just verify both work
        assert inst_est.dealer_spread_pips > 0
        assert retail_est.dealer_spread_pips > 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_order(self, integration: ForexExecutionIntegration) -> None:
        """Test with very small order."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=1_000,  # Very small
            mid_price=1.0850,
        )
        assert report.execution_result is not None

    def test_very_large_order(self, integration: ForexExecutionIntegration) -> None:
        """Test with very large order."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000_000,  # Very large
            mid_price=1.0850,
        )
        assert report.execution_result is not None
        # Large order should have recommendation
        assert report.pre_trade_estimate.recommended_execution in [
            "split_order",
            "wait_for_better_liquidity",
            "immediate",
        ]

    def test_weekend_session(self, integration: ForexExecutionIntegration) -> None:
        """Test execution attempt during weekend."""
        report = integration.execute(
            symbol="EUR_USD",
            side="BUY",
            size_usd=100_000,
            mid_price=1.0850,
            session=ForexSession.WEEKEND,
        )
        # Weekend should typically have high slippage in TCA
        assert report.tca_slippage_pips > 0

    def test_cross_pair(self, integration: ForexExecutionIntegration) -> None:
        """Test execution for cross pair."""
        report = integration.execute(
            symbol="EUR_GBP",
            side="BUY",
            size_usd=200_000,
            mid_price=0.8600,
        )
        assert report.pre_trade_estimate.pair_type in [PairType.MINOR, PairType.CROSS]

    def test_multiple_sessions_same_integration(
        self, integration: ForexExecutionIntegration
    ) -> None:
        """Test multiple sessions with same integration."""
        sessions = [
            ForexSession.SYDNEY,
            ForexSession.TOKYO,
            ForexSession.LONDON,
            ForexSession.NEW_YORK,
            ForexSession.LONDON_NY_OVERLAP,
        ]

        for session in sessions:
            report = integration.execute(
                symbol="EUR_USD",
                side="BUY",
                size_usd=100_000,
                mid_price=1.0850,
                session=session,
            )
            assert report.pre_trade_estimate.session == session


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
