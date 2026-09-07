# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 5: TCA Compliance Wrapper.

This module provides comprehensive tests for the tca_compliance module:
- TCAMetricType, TCABenchmark, CostCategory, ExecutionStrategy enums
- PreTradeEstimate, PostTradeAnalysis dataclasses
- TCAConfig, TCAAggregateMetrics dataclasses
- TCAComplianceWrapper class
- Factory functions

Target: 100% test coverage
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
import time

from services.algo_integration.tca_compliance import (
    # Enums
    TCAMetricType,
    TCABenchmark,
    CostCategory,
    ExecutionStrategy,
    # Data classes
    PreTradeEstimate,
    PostTradeAnalysis,
    TCAConfig,
    TCAAggregateMetrics,
    # Protocol
    SlippageProvider,
    # Main class
    TCAComplianceWrapper,
    # Factory
    create_tca_wrapper,
)


# =============================================================================
# Test: Enums
# =============================================================================


class TestTCAMetricType:
    """Tests for TCAMetricType enum."""

    def test_metric_types_defined(self):
        """Verify TCA metric types."""
        assert TCAMetricType.IMPLEMENTATION_SHORTFALL.value == "implementation_shortfall"
        assert TCAMetricType.VWAP_SLIPPAGE.value == "vwap_slippage"
        assert TCAMetricType.ARRIVAL_PRICE_SLIPPAGE.value == "arrival_price_slippage"
        assert TCAMetricType.MARKET_IMPACT.value == "market_impact"


class TestTCABenchmark:
    """Tests for TCABenchmark enum."""

    def test_benchmarks_defined(self):
        """Verify benchmark types."""
        assert TCABenchmark.ARRIVAL_PRICE.value == "arrival_price"
        assert TCABenchmark.VWAP.value == "vwap"
        assert TCABenchmark.TWAP.value == "twap"
        assert TCABenchmark.CLOSE.value == "close"


class TestCostCategory:
    """Tests for CostCategory enum."""

    def test_cost_categories_defined(self):
        """Verify cost categories."""
        assert CostCategory.EXPLICIT.value == "explicit"
        assert CostCategory.IMPLICIT.value == "implicit"
        assert CostCategory.TOTAL.value == "total"


class TestExecutionStrategy:
    """Tests for ExecutionStrategy enum."""

    def test_strategies_defined(self):
        """Verify execution strategies."""
        assert ExecutionStrategy.MARKET.value == "market"
        assert ExecutionStrategy.VWAP.value == "vwap"
        assert ExecutionStrategy.TWAP.value == "twap"
        assert ExecutionStrategy.IS.value == "is"


# =============================================================================
# Test: PreTradeEstimate
# =============================================================================


class TestPreTradeEstimate:
    """Tests for PreTradeEstimate dataclass."""

    def test_create_estimate(self):
        """Test creating pre-trade estimate."""
        estimate = PreTradeEstimate(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side="BUY",
            quantity=Decimal("1000"),
            notional_value=Decimal("100000"),
        )

        assert estimate.order_id == "ORD-001"
        assert estimate.estimate_id is not None
        assert estimate.timestamp_ns > 0

    def test_estimate_with_costs(self):
        """Test estimate with cost projections."""
        estimate = PreTradeEstimate(
            estimated_impact_bps=Decimal("5.5"),
            estimated_spread_cost_bps=Decimal("2.5"),
            estimated_timing_cost_bps=Decimal("1.0"),
            estimated_total_cost_bps=Decimal("9.0"),
        )

        assert estimate.estimated_impact_bps == Decimal("5.5")
        assert estimate.estimated_total_cost_bps == Decimal("9.0")

    def test_estimate_with_recommendations(self):
        """Test estimate with strategy recommendations."""
        estimate = PreTradeEstimate(
            recommended_strategy=ExecutionStrategy.VWAP,
            recommended_urgency="low",
            participation_rate_pct=5.0,
            expected_duration_seconds=3600.0,
        )

        assert estimate.recommended_strategy == ExecutionStrategy.VWAP
        assert estimate.recommended_urgency == "low"
        assert estimate.participation_rate_pct == 5.0

    def test_estimate_to_dict(self):
        """Test estimate serialization."""
        estimate = PreTradeEstimate(
            order_id="ORD-002",
            arrival_price=Decimal("100.50"),
            estimated_total_cost_bps=Decimal("10.0"),
        )

        d = estimate.to_dict()

        assert d["order_id"] == "ORD-002"
        assert d["arrival_price"] == "100.50"
        assert d["estimated_total_cost_bps"] == "10.0"


# =============================================================================
# Test: PostTradeAnalysis
# =============================================================================


class TestPostTradeAnalysis:
    """Tests for PostTradeAnalysis dataclass."""

    def test_create_analysis(self):
        """Test creating post-trade analysis."""
        analysis = PostTradeAnalysis(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side="BUY",
            quantity=Decimal("1000"),
            filled_quantity=Decimal("1000"),
        )

        assert analysis.order_id == "ORD-001"
        assert analysis.analysis_id is not None
        assert analysis.fill_rate == 0.0  # Not calculated yet

    def test_analysis_with_costs(self):
        """Test analysis with actual costs."""
        analysis = PostTradeAnalysis(
            arrival_slippage_bps=Decimal("3.5"),
            vwap_slippage_bps=Decimal("2.0"),
            implementation_shortfall_bps=Decimal("5.0"),
            explicit_cost_bps=Decimal("1.5"),
            total_cost_bps=Decimal("6.5"),
        )

        assert analysis.implementation_shortfall_bps == Decimal("5.0")
        assert analysis.total_cost_bps == Decimal("6.5")

    def test_analysis_vs_estimate(self):
        """Test analysis with estimate comparison."""
        analysis = PostTradeAnalysis(
            estimated_cost_bps=Decimal("10.0"),
            total_cost_bps=Decimal("8.0"),
            cost_vs_estimate_bps=Decimal("-2.0"),  # 2bps better than estimate
            within_estimate_range=True,
        )

        assert analysis.cost_vs_estimate_bps == Decimal("-2.0")
        assert analysis.within_estimate_range is True

    def test_analysis_execution_quality(self):
        """Test analysis with execution quality metrics."""
        analysis = PostTradeAnalysis(
            fill_rate=0.95,
            execution_duration_seconds=120.0,
            num_fills=5,
            avg_fill_size=Decimal("200"),
            strategy_used=ExecutionStrategy.TWAP,
        )

        assert analysis.fill_rate == 0.95
        assert analysis.num_fills == 5
        assert analysis.strategy_used == ExecutionStrategy.TWAP

    def test_analysis_to_dict(self):
        """Test analysis serialization."""
        analysis = PostTradeAnalysis(
            order_id="ORD-003",
            execution_price=Decimal("100.25"),
            total_cost_bps=Decimal("7.5"),
        )

        d = analysis.to_dict()

        assert d["order_id"] == "ORD-003"
        assert d["execution_price"] == "100.25"


# =============================================================================
# Test: TCAConfig
# =============================================================================


class TestTCAConfig:
    """Tests for TCAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TCAConfig()

        assert config.impact_model == "almgren_chriss"
        assert config.default_confidence_level == 0.95
        assert config.permanent_impact_coefficient == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = TCAConfig(
            firm_lei="5493001KJTIIGC8Y1R12",
            impact_model="square_root",
            permanent_impact_coefficient=0.15,
            alert_threshold_bps=Decimal("75"),
        )

        assert config.firm_lei == "5493001KJTIIGC8Y1R12"
        assert config.impact_model == "square_root"
        assert config.alert_threshold_bps == Decimal("75")


# =============================================================================
# Test: TCAAggregateMetrics
# =============================================================================


class TestTCAAggregateMetrics:
    """Tests for TCAAggregateMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating aggregate metrics."""
        now = datetime.utcnow()
        metrics = TCAAggregateMetrics(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_orders=100,
            total_notional=Decimal("10000000"),
        )

        assert metrics.total_orders == 100
        assert metrics.total_notional == Decimal("10000000")

    def test_metrics_with_averages(self):
        """Test metrics with calculated averages."""
        now = datetime.utcnow()
        metrics = TCAAggregateMetrics(
            period_start=now - timedelta(days=30),
            period_end=now,
            avg_implementation_shortfall_bps=Decimal("5.5"),
            avg_arrival_slippage_bps=Decimal("3.0"),
            avg_total_cost_bps=Decimal("8.0"),
            pct_within_estimate=85.0,
        )

        assert metrics.avg_implementation_shortfall_bps == Decimal("5.5")
        assert metrics.pct_within_estimate == 85.0


# =============================================================================
# Test: TCAComplianceWrapper
# =============================================================================


class TestTCAComplianceWrapper:
    """Tests for TCAComplianceWrapper class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TCAConfig(
            firm_lei="5493001KJTIIGC8Y1R12",
            impact_model="almgren_chriss",
        )

    @pytest.fixture
    def wrapper(self, config):
        """Create test wrapper."""
        return TCAComplianceWrapper(config)

    # === Pre-Trade Estimation Tests ===

    def test_estimate_pre_trade_basic(self, wrapper):
        """Test basic pre-trade estimation."""
        order = {
            "order_id": "ORD-001",
            "instrument_isin": "US0378331005",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        market_data = {
            "bid": Decimal("99.50"),
            "ask": Decimal("100.50"),
            "adv": Decimal("100000"),  # ADV
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.order_id == "ORD-001"
        assert estimate.arrival_price == Decimal("100.00")  # Mid price
        assert estimate.estimated_spread_cost_bps > Decimal("0")
        assert estimate.estimated_total_cost_bps > Decimal("0")
        assert estimate.recommended_strategy is not None

    def test_estimate_pre_trade_no_adv(self, wrapper):
        """Test estimation without ADV."""
        order = {
            "order_id": "ORD-002",
            "side": "SELL",
            "quantity": Decimal("500"),
        }

        market_data = {
            "bid": Decimal("50.00"),
            "ask": Decimal("50.10"),
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.estimated_impact_bps == Decimal("5")  # Default

    def test_estimate_pre_trade_with_notional(self, wrapper):
        """Test estimation with explicit notional."""
        order = {
            "order_id": "ORD-003",
            "side": "BUY",
            "quantity": Decimal("1000"),
            "notional": Decimal("100000"),
        }

        market_data = {
            "mid": Decimal("100.00"),
            "adv": Decimal("1000000"),
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.notional_value == Decimal("100000")

    def test_estimate_pre_trade_with_volatility(self, wrapper):
        """Test estimation with volatility input."""
        order = {
            "order_id": "ORD-004",
            "side": "BUY",
            "quantity": Decimal("5000"),
        }

        market_data = {
            "bid": Decimal("100.00"),
            "ask": Decimal("100.20"),
            "adv": Decimal("50000"),
            "volatility": 0.03,  # 3% volatility
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.volatility == 0.03

    def test_estimate_strategy_recommendation_small_order(self, wrapper):
        """Test strategy recommendation for small order."""
        order = {
            "order_id": "SMALL",
            "side": "BUY",
            "quantity": Decimal("50"),  # Very small order
        }

        market_data = {
            "bid": Decimal("100.00"),
            "ask": Decimal("100.10"),
            "adv": Decimal("1000000"),  # Very large ADV -> participation = 0.5% < 1%
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        # Small order (< 1% ADV) should recommend MARKET strategy
        assert estimate.recommended_strategy == ExecutionStrategy.MARKET

    def test_estimate_strategy_recommendation_large_order(self, wrapper):
        """Test strategy recommendation for large order."""
        order = {
            "order_id": "LARGE",
            "side": "BUY",
            "quantity": Decimal("100000"),
        }

        market_data = {
            "bid": Decimal("100.00"),
            "ask": Decimal("100.10"),
            "adv": Decimal("500000"),  # 20% of ADV
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        # Large order should recommend IS strategy
        assert estimate.recommended_strategy == ExecutionStrategy.IS
        assert estimate.recommended_urgency == "low"

    def test_estimate_confidence_range(self, wrapper):
        """Test confidence range calculation."""
        order = {
            "order_id": "RANGE",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        market_data = {
            "bid": Decimal("100.00"),
            "ask": Decimal("100.10"),
            "adv": Decimal("100000"),
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.cost_range_low_bps < estimate.estimated_total_cost_bps
        assert estimate.cost_range_high_bps > estimate.estimated_total_cost_bps

    # === Post-Trade Analysis Tests ===

    def test_analyze_post_trade_basic(self, wrapper):
        """Test basic post-trade analysis."""
        order = {
            "order_id": "ORD-PT-001",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        fills = [
            {
                "price": Decimal("100.05"),
                "quantity": Decimal("1000"),
                "timestamp_ms": 1000050,
                "commission": Decimal("5.00"),
                "fees": Decimal("1.00"),
                "venue_mic": "XLON",
            }
        ]

        market_data = {
            "arrival_price": Decimal("100.00"),
            "vwap": Decimal("100.02"),
            "close": Decimal("100.10"),
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        assert analysis.order_id == "ORD-PT-001"
        assert analysis.execution_price == Decimal("100.05")
        assert analysis.filled_quantity == Decimal("1000")
        assert analysis.fill_rate == 1.0

    def test_analyze_post_trade_multiple_fills(self, wrapper):
        """Test analysis with multiple fills."""
        order = {
            "order_id": "ORD-MULTI",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        fills = [
            {"price": Decimal("100.00"), "quantity": Decimal("300"), "timestamp_ms": 1000},
            {"price": Decimal("100.05"), "quantity": Decimal("300"), "timestamp_ms": 2000},
            {"price": Decimal("100.10"), "quantity": Decimal("400"), "timestamp_ms": 3000},
        ]

        market_data = {
            "arrival_price": Decimal("100.00"),
            "vwap": Decimal("100.05"),
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        assert analysis.num_fills == 3
        assert analysis.execution_duration_seconds == 2.0  # 3000 - 1000 ms
        # Weighted avg: (300*100 + 300*100.05 + 400*100.10) / 1000 = 100.055
        assert analysis.execution_price == Decimal("100.055")

    def test_analyze_post_trade_partial_fill(self, wrapper):
        """Test analysis with partial fill."""
        order = {
            "order_id": "ORD-PARTIAL",
            "side": "SELL",
            "quantity": Decimal("1000"),
        }

        fills = [
            {"price": Decimal("99.95"), "quantity": Decimal("600"), "timestamp_ms": 1000},
        ]

        market_data = {
            "arrival_price": Decimal("100.00"),
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        assert analysis.filled_quantity == Decimal("600")
        assert analysis.fill_rate == 0.6

    def test_analyze_post_trade_no_fills(self, wrapper):
        """Test analysis with no fills."""
        order = {
            "order_id": "ORD-NOFILL",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        fills = []

        market_data = {}

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        assert "No fills received" in analysis.notes

    def test_analyze_post_trade_with_estimate(self, wrapper):
        """Test analysis compared to pre-trade estimate."""
        order = {
            "order_id": "ORD-EST",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        market_data_pre = {
            "bid": Decimal("99.95"),
            "ask": Decimal("100.05"),
            "adv": Decimal("100000"),
        }

        # Get estimate first
        estimate = wrapper.estimate_pre_trade(order, market_data_pre)

        fills = [
            {"price": Decimal("100.02"), "quantity": Decimal("1000"), "timestamp_ms": 1000},
        ]

        market_data_post = {
            "arrival_price": Decimal("100.00"),
            "vwap": Decimal("100.01"),
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data_post, estimate)

        assert analysis.estimate_id == estimate.estimate_id
        assert analysis.estimated_cost_bps is not None
        assert analysis.cost_vs_estimate_bps is not None

    def test_analyze_post_trade_slippage_calculation(self, wrapper):
        """Test slippage calculation."""
        order = {
            "order_id": "ORD-SLIP",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        fills = [
            {"price": Decimal("100.10"), "quantity": Decimal("100"), "timestamp_ms": 1000},
        ]

        market_data = {
            "arrival_price": Decimal("100.00"),  # Bought 10bps higher
            "vwap": Decimal("100.05"),
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        # For buy, paying more than arrival = positive slippage (bad)
        assert analysis.arrival_slippage_bps > Decimal("0")

    def test_analyze_post_trade_sell_slippage(self, wrapper):
        """Test slippage calculation for sell orders."""
        order = {
            "order_id": "ORD-SELL-SLIP",
            "side": "SELL",
            "quantity": Decimal("100"),
        }

        fills = [
            {"price": Decimal("99.90"), "quantity": Decimal("100"), "timestamp_ms": 1000},
        ]

        market_data = {
            "arrival_price": Decimal("100.00"),  # Sold 10bps lower
        }

        analysis = wrapper.analyze_post_trade(order, fills, market_data)

        # For sell, receiving less than arrival = positive slippage (bad)
        assert analysis.arrival_slippage_bps > Decimal("0")

    # === Integration Tests ===

    def test_get_estimate(self, wrapper):
        """Test getting estimate by ID."""
        order = {"order_id": "GET-EST", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"mid": Decimal("100")}

        estimate = wrapper.estimate_pre_trade(order, market_data)
        retrieved = wrapper.get_estimate(estimate.estimate_id)

        assert retrieved == estimate

    def test_get_estimates_for_order(self, wrapper):
        """Test getting all estimates for an order."""
        order = {"order_id": "MULTI-EST", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"mid": Decimal("100")}

        # Create multiple estimates for same order
        wrapper.estimate_pre_trade(order, market_data)
        wrapper.estimate_pre_trade(order, market_data)

        estimates = wrapper.get_estimates_for_order("MULTI-EST")
        assert len(estimates) == 2

    def test_get_analyses_date_filter(self, wrapper):
        """Test getting analyses with date filter."""
        order = {"order_id": "DATE-FILTER", "side": "BUY", "quantity": Decimal("100")}
        fills = [{"price": Decimal("100"), "quantity": Decimal("100"), "timestamp_ms": 1000}]
        market_data = {}

        wrapper.analyze_post_trade(order, fills, market_data)

        now = datetime.utcnow()
        analyses = wrapper.get_analyses(
            start_time=now - timedelta(hours=1), end_time=now + timedelta(hours=1)
        )

        assert len(analyses) == 1

    def test_get_aggregate_metrics(self, wrapper):
        """Test aggregate metrics calculation."""
        # Add multiple analyses
        for i in range(5):
            order = {"order_id": f"AGG-{i}", "side": "BUY", "quantity": Decimal("100")}
            fills = [{"price": Decimal("100"), "quantity": Decimal("100"), "timestamp_ms": 1000}]
            market_data = {"arrival_price": Decimal("99.95"), "vwap": Decimal("99.98")}

            # Create estimate and analysis
            estimate = wrapper.estimate_pre_trade(
                order, {"mid": Decimal("99.975"), "adv": Decimal("10000")}
            )
            wrapper.analyze_post_trade(order, fills, market_data, estimate)

        now = datetime.utcnow()
        metrics = wrapper.get_aggregate_metrics(
            start_time=now - timedelta(hours=1), end_time=now + timedelta(hours=1)
        )

        assert metrics.total_orders == 5
        assert metrics.total_notional > Decimal("0")

    def test_get_aggregate_metrics_empty(self, wrapper):
        """Test aggregate metrics with no data."""
        now = datetime.utcnow()
        metrics = wrapper.get_aggregate_metrics(
            start_time=now - timedelta(hours=1), end_time=now + timedelta(hours=1)
        )

        assert metrics.total_orders == 0

    def test_clear_history(self, wrapper):
        """Test clearing history."""
        order = {"order_id": "CLEAR", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"mid": Decimal("100")}
        fills = [{"price": Decimal("100"), "quantity": Decimal("100"), "timestamp_ms": 1000}]

        wrapper.estimate_pre_trade(order, market_data)
        wrapper.analyze_post_trade(order, fills, {})

        estimates_cleared, analyses_cleared = wrapper.clear_history()

        assert estimates_cleared == 1
        assert analyses_cleared == 1

    # === Audit Callback Tests ===

    def test_audit_callback_pre_trade(self, config):
        """Test audit callback for pre-trade estimate."""
        callback_data = []

        def callback(data):
            callback_data.append(data)

        wrapper = TCAComplianceWrapper(config, audit_callback=callback)

        order = {"order_id": "AUDIT-PRE", "side": "BUY", "quantity": Decimal("100")}
        wrapper.estimate_pre_trade(order, {"mid": Decimal("100")})

        assert len(callback_data) == 1
        assert callback_data[0]["type"] == "pre_trade_estimate"

    def test_audit_callback_post_trade(self, config):
        """Test audit callback for post-trade analysis."""
        callback_data = []

        def callback(data):
            callback_data.append(data)

        wrapper = TCAComplianceWrapper(config, audit_callback=callback)

        order = {"order_id": "AUDIT-POST", "side": "BUY", "quantity": Decimal("100")}
        fills = [{"price": Decimal("100"), "quantity": Decimal("100"), "timestamp_ms": 1000}]
        wrapper.analyze_post_trade(order, fills, {})

        assert len(callback_data) == 1
        assert callback_data[0]["type"] == "post_trade_analysis"

    # === External Slippage Provider Tests ===

    def test_external_slippage_provider(self, config):
        """Test using external slippage provider."""

        class MockSlippageProvider:
            def estimate_slippage_bps(self, notional, adv, side, volatility=None, spread_bps=None):
                return 15.0  # Fixed 15bps

        provider = MockSlippageProvider()
        wrapper = TCAComplianceWrapper(config, slippage_provider=provider)

        order = {"order_id": "EXT-SLIP", "side": "BUY", "quantity": Decimal("1000")}
        market_data = {"mid": Decimal("100"), "adv": Decimal("100000")}

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.estimated_impact_bps == Decimal("15")

    # === Impact Model Tests ===

    def test_linear_impact_model(self):
        """Test linear impact model."""
        config = TCAConfig(impact_model="linear")
        wrapper = TCAComplianceWrapper(config)

        order = {"order_id": "LINEAR", "side": "BUY", "quantity": Decimal("10000")}
        market_data = {"mid": Decimal("100"), "adv": Decimal("100000")}

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.estimated_impact_bps > Decimal("0")
        assert estimate.model_name == "linear"

    def test_square_root_impact_model(self):
        """Test square root impact model."""
        config = TCAConfig(impact_model="square_root")
        wrapper = TCAComplianceWrapper(config)

        order = {"order_id": "SQRT", "side": "BUY", "quantity": Decimal("10000")}
        market_data = {"mid": Decimal("100"), "adv": Decimal("100000")}

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.estimated_impact_bps > Decimal("0")
        assert estimate.model_name == "square_root"


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tca_wrapper_default(self):
        """Test creating wrapper with defaults."""
        wrapper = create_tca_wrapper()

        assert wrapper is not None
        assert wrapper.config.impact_model == "almgren_chriss"

    def test_create_tca_wrapper_with_lei(self):
        """Test creating wrapper with LEI."""
        wrapper = create_tca_wrapper(firm_lei="5493001KJTIIGC8Y1R12")

        assert wrapper.config.firm_lei == "5493001KJTIIGC8Y1R12"

    def test_create_tca_wrapper_with_callback(self):
        """Test creating wrapper with callback."""
        callback = Mock()
        wrapper = create_tca_wrapper(audit_callback=callback)

        assert wrapper.audit_callback == callback
        assert wrapper.config.enable_audit_trail is True

    def test_create_tca_wrapper_with_model(self):
        """Test creating wrapper with specific model."""
        wrapper = create_tca_wrapper(impact_model="linear")

        assert wrapper.config.impact_model == "linear"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_estimate_with_zero_spread(self):
        """Test estimation with zero spread."""
        wrapper = create_tca_wrapper()

        order = {"order_id": "ZERO-SPREAD", "side": "BUY", "quantity": Decimal("100")}
        market_data = {
            "bid": Decimal("100"),
            "ask": Decimal("100"),  # Zero spread
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.spread_bps == Decimal("0")

    def test_estimate_with_missing_bid_ask(self):
        """Test estimation with only mid price."""
        wrapper = create_tca_wrapper()

        order = {"order_id": "MID-ONLY", "side": "BUY", "quantity": Decimal("100")}
        market_data = {
            "mid": Decimal("100"),
            "spread_bps": Decimal("10"),
        }

        estimate = wrapper.estimate_pre_trade(order, market_data)

        assert estimate.arrival_price == Decimal("100")

    def test_analysis_with_multiple_venues(self):
        """Test analysis with fills from multiple venues."""
        wrapper = create_tca_wrapper()

        order = {"order_id": "MULTI-VENUE", "side": "BUY", "quantity": Decimal("1000")}
        fills = [
            {
                "price": Decimal("100"),
                "quantity": Decimal("500"),
                "timestamp_ms": 1000,
                "venue_mic": "XLON",
            },
            {
                "price": Decimal("100.05"),
                "quantity": Decimal("500"),
                "timestamp_ms": 2000,
                "venue_mic": "XETR",
            },
        ]

        analysis = wrapper.analyze_post_trade(order, fills, {})

        assert "XLON" in analysis.venue_mic
        assert "XETR" in analysis.venue_mic

    def test_concurrent_estimates(self):
        """Test concurrent estimate generation."""
        import threading

        wrapper = create_tca_wrapper()
        results = []
        errors = []

        def generate_estimate(i):
            try:
                order = {"order_id": f"CONC-{i}", "side": "BUY", "quantity": Decimal("100")}
                estimate = wrapper.estimate_pre_trade(order, {"mid": Decimal("100")})
                results.append(estimate)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate_estimate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=services.compliance.tca_compliance"])
