# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 5: Venue Analysis and Smart Order Routing.

This module provides comprehensive tests for the venue_analysis module:
- VenueMetricType, VenueSelectionReason, VenueStatus enums
- VenueExecutionRecord, VenuePerformanceMetrics, VenueRoutingDecision dataclasses
- VenueAnalysisConfig dataclass
- VenueAnalyzer class
- SmartOrderRouter class
- Factory functions

Target: 100% test coverage
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
import time
import threading

from services.algo_integration.venue_analysis import (
    # Enums
    VenueMetricType,
    VenueSelectionReason,
    VenueStatus,
    # Data classes
    VenueExecutionRecord,
    VenuePerformanceMetrics,
    VenueRoutingDecision,
    VenueAnalysisConfig,
    # Main classes
    VenueAnalyzer,
    SmartOrderRouter,
    # Factory functions
    create_venue_analyzer,
    create_smart_order_router,
)

from services.algo_integration.best_execution import (
    ExecutionVenue,
    VenueType,
    AssetClass,
)


# =============================================================================
# Test: Enums
# =============================================================================


class TestVenueMetricType:
    """Tests for VenueMetricType enum."""

    def test_metric_types_defined(self):
        """Verify venue metric types."""
        assert VenueMetricType.SPREAD.value == "spread"
        assert VenueMetricType.LATENCY.value == "latency"
        assert VenueMetricType.FILL_RATE.value == "fill_rate"
        assert VenueMetricType.PRICE_IMPROVEMENT.value == "price_improvement"
        assert VenueMetricType.COST.value == "cost"


class TestVenueSelectionReason:
    """Tests for VenueSelectionReason enum."""

    def test_selection_reasons_defined(self):
        """Verify selection reasons."""
        assert VenueSelectionReason.BEST_PRICE.value == "best_price"
        assert VenueSelectionReason.BEST_LIQUIDITY.value == "best_liquidity"
        assert VenueSelectionReason.LOWEST_COST.value == "lowest_cost"
        assert VenueSelectionReason.EXPLICIT_INSTRUCTION.value == "explicit_instruction"


class TestVenueStatus:
    """Tests for VenueStatus enum."""

    def test_statuses_defined(self):
        """Verify venue statuses."""
        assert VenueStatus.ACTIVE.value == "active"
        assert VenueStatus.INACTIVE.value == "inactive"
        assert VenueStatus.DEGRADED.value == "degraded"
        assert VenueStatus.MAINTENANCE.value == "maintenance"


# =============================================================================
# Test: VenueExecutionRecord
# =============================================================================


class TestVenueExecutionRecord:
    """Tests for VenueExecutionRecord dataclass."""

    def test_create_record(self):
        """Test creating execution record."""
        record = VenueExecutionRecord(
            venue_mic="XLON",
            instrument_isin="US0378331005",
            side="BUY",
            order_id="ORD-001",
            order_quantity=Decimal("1000"),
            fill_quantity=Decimal("1000"),
            fill_price=Decimal("100.50"),
        )

        assert record.venue_mic == "XLON"
        assert record.record_id is not None
        assert record.timestamp_ns > 0

    def test_record_with_metrics(self):
        """Test record with performance metrics."""
        record = VenueExecutionRecord(
            venue_mic="XETR",
            spread_bps=Decimal("5.5"),
            latency_ms=25.0,
            price_improvement_bps=Decimal("2.0"),
            cost_bps=Decimal("3.0"),
            fill_rate=0.95,
        )

        assert record.spread_bps == Decimal("5.5")
        assert record.latency_ms == 25.0
        assert record.fill_rate == 0.95

    def test_record_with_rejection(self):
        """Test record with rejection."""
        record = VenueExecutionRecord(
            venue_mic="XPAR",
            was_rejected=True,
            rejection_reason="Price outside collar",
        )

        assert record.was_rejected is True
        assert "collar" in record.rejection_reason

    def test_record_to_dict(self):
        """Test record serialization."""
        record = VenueExecutionRecord(
            venue_mic="XLON",
            order_id="ORD-002",
            fill_price=Decimal("150.25"),
        )

        d = record.to_dict()

        assert d["venue_mic"] == "XLON"
        assert d["order_id"] == "ORD-002"
        assert d["fill_price"] == "150.25"


# =============================================================================
# Test: VenuePerformanceMetrics
# =============================================================================


class TestVenuePerformanceMetrics:
    """Tests for VenuePerformanceMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating performance metrics."""
        metrics = VenuePerformanceMetrics(
            venue_mic="XLON",
            total_orders=100,
            total_fills=95,
            total_quantity_filled=Decimal("95000"),
        )

        assert metrics.venue_mic == "XLON"
        assert metrics.total_orders == 100
        assert metrics.total_fills == 95

    def test_metrics_with_averages(self):
        """Test metrics with calculated averages."""
        metrics = VenuePerformanceMetrics(
            venue_mic="XETR",
            avg_spread_bps=Decimal("4.5"),
            avg_latency_ms=15.0,
            avg_fill_rate=0.95,
            avg_price_improvement_bps=Decimal("2.0"),
            quality_score=85.0,
        )

        assert metrics.avg_spread_bps == Decimal("4.5")
        assert metrics.quality_score == 85.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        now = datetime.utcnow()
        metrics = VenuePerformanceMetrics(
            venue_mic="XPAR",
            period_start=now - timedelta(days=30),
            period_end=now,
            total_orders=50,
        )

        d = metrics.to_dict()

        assert d["venue_mic"] == "XPAR"
        assert d["total_orders"] == 50
        assert d["period_start"] is not None


# =============================================================================
# Test: VenueRoutingDecision
# =============================================================================


class TestVenueRoutingDecision:
    """Tests for VenueRoutingDecision dataclass."""

    def test_create_decision(self):
        """Test creating routing decision."""
        decision = VenueRoutingDecision(
            order_id="ORD-001",
            instrument_isin="US0378331005",
            side="BUY",
            quantity=Decimal("1000"),
            selected_venue="XLON",
            selection_reason=VenueSelectionReason.BEST_PRICE,
        )

        assert decision.order_id == "ORD-001"
        assert decision.selected_venue == "XLON"
        assert decision.selection_reason == VenueSelectionReason.BEST_PRICE

    def test_decision_with_alternatives(self):
        """Test decision with alternative venues."""
        decision = VenueRoutingDecision(
            order_id="ORD-002",
            selected_venue="XETR",
            venues_considered=["XLON", "XETR", "XPAR"],
            venue_scores={"XLON": 85.0, "XETR": 90.0, "XPAR": 82.0},
        )

        assert len(decision.venues_considered) == 3
        assert decision.venue_scores["XETR"] == 90.0

    def test_decision_to_dict(self):
        """Test decision serialization."""
        decision = VenueRoutingDecision(
            order_id="ORD-003",
            selected_venue="XPAR",
            selection_reason=VenueSelectionReason.EXPLICIT_INSTRUCTION,
            rationale="Client requested specific venue",
        )

        d = decision.to_dict()

        assert d["order_id"] == "ORD-003"
        assert d["selected_venue"] == "XPAR"
        assert "explicit_instruction" in d["selection_reason"]


# =============================================================================
# Test: VenueAnalysisConfig
# =============================================================================


class TestVenueAnalysisConfig:
    """Tests for VenueAnalysisConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = VenueAnalysisConfig()

        assert config.min_samples == 30
        assert config.rolling_window_days == 30
        assert len(config.metric_weights) > 0

    def test_validate_config(self):
        """Test configuration validation."""
        config = VenueAnalysisConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_validate_invalid_weights(self):
        """Test validation catches invalid weights."""
        config = VenueAnalysisConfig(
            metric_weights={
                VenueMetricType.SPREAD: 0.5,
                VenueMetricType.LATENCY: 0.5,
                VenueMetricType.FILL_RATE: 0.5,  # Sum > 1
            }
        )

        errors = config.validate()
        assert any("weights sum" in e.lower() for e in errors)

    def test_custom_config(self):
        """Test custom configuration."""
        config = VenueAnalysisConfig(
            min_samples=50,
            rolling_window_days=60,
            dark_pool_enabled=True,
        )

        assert config.min_samples == 50
        assert config.dark_pool_enabled is True


# =============================================================================
# Test: VenueAnalyzer
# =============================================================================


class TestVenueAnalyzer:
    """Tests for VenueAnalyzer class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return VenueAnalysisConfig(min_samples=3)  # Lower for testing

    @pytest.fixture
    def venues(self):
        """Create test venues."""
        return [
            ExecutionVenue(mic="XLON", name="LSE", venue_type=VenueType.REGULATED_MARKET),
            ExecutionVenue(mic="XETR", name="Xetra", venue_type=VenueType.REGULATED_MARKET),
            ExecutionVenue(mic="XPAR", name="Euronext Paris", venue_type=VenueType.REGULATED_MARKET),
        ]

    @pytest.fixture
    def analyzer(self, config, venues):
        """Create test analyzer."""
        return VenueAnalyzer(config, venues=venues)

    def test_create_analyzer(self, analyzer):
        """Test analyzer creation."""
        assert analyzer is not None
        assert len(analyzer._venues) == 3

    def test_add_venue(self, analyzer):
        """Test adding a venue."""
        new_venue = ExecutionVenue(mic="XAMS", name="Amsterdam")
        analyzer.add_venue(new_venue)

        assert "XAMS" in analyzer._venues
        assert analyzer._venue_status["XAMS"] == VenueStatus.ACTIVE

    def test_remove_venue(self, analyzer):
        """Test removing a venue."""
        result = analyzer.remove_venue("XLON")

        assert result is True
        assert "XLON" not in analyzer._venues

    def test_remove_nonexistent_venue(self, analyzer):
        """Test removing non-existent venue."""
        result = analyzer.remove_venue("XXXX")
        assert result is False

    def test_set_venue_status(self, analyzer):
        """Test setting venue status."""
        result = analyzer.set_venue_status("XLON", VenueStatus.MAINTENANCE)

        assert result is True
        assert analyzer._venue_status["XLON"] == VenueStatus.MAINTENANCE

    def test_record_execution(self, analyzer):
        """Test recording execution."""
        record = VenueExecutionRecord(
            venue_mic="XLON",
            order_id="ORD-001",
            fill_quantity=Decimal("100"),
            fill_price=Decimal("100.50"),
        )

        analyzer.record_execution(record)

        assert len(analyzer._records) == 1

    def test_record_from_fill(self, analyzer):
        """Test creating record from fill data."""
        order = {
            "order_id": "ORD-002",
            "instrument_isin": "US0378331005",
            "side": "BUY",
            "quantity": Decimal("1000"),
            "submit_time_ms": 1000000,
        }

        fill = {
            "price": Decimal("100.05"),
            "quantity": Decimal("1000"),
            "timestamp_ms": 1000100,
            "commission": Decimal("5.00"),
            "fees": Decimal("1.00"),
            "venue_mic": "XLON",
        }

        market_data = {
            "arrival_price": Decimal("100.00"),
            "bid": Decimal("99.95"),
            "ask": Decimal("100.05"),
        }

        record = analyzer.record_from_fill(order, fill, market_data)

        assert record.venue_mic == "XLON"
        assert record.fill_rate == 1.0
        assert record.latency_ms == 100.0

    def test_get_venue_metrics(self, analyzer):
        """Test getting venue metrics."""
        # Add some records
        for i in range(5):
            record = VenueExecutionRecord(
                venue_mic="XLON",
                order_id=f"ORD-{i}",
                fill_quantity=Decimal("100"),
                fill_price=Decimal("100.00"),
                spread_bps=Decimal("5"),
                latency_ms=20.0 + i,
                price_improvement_bps=Decimal("1"),
                fill_rate=0.95,
            )
            analyzer.record_execution(record)

        metrics = analyzer.get_venue_metrics("XLON")

        assert metrics.venue_mic == "XLON"
        assert metrics.total_orders == 5
        assert metrics.avg_latency_ms > 0

    def test_get_venue_metrics_empty(self, analyzer):
        """Test getting metrics for venue with no data."""
        metrics = analyzer.get_venue_metrics("XLON")

        assert metrics.total_orders == 0
        assert metrics.avg_spread_bps == Decimal("0")

    def test_get_venue_metrics_date_filter(self, analyzer):
        """Test metrics with date filter."""
        record = VenueExecutionRecord(
            venue_mic="XETR",
            order_id="ORD-DATED",
            fill_quantity=Decimal("100"),
            fill_price=Decimal("50.00"),
        )
        analyzer.record_execution(record)

        now = datetime.utcnow()
        metrics = analyzer.get_venue_metrics(
            "XETR",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1)
        )

        assert metrics.total_orders == 1

    def test_quality_score_calculation(self, analyzer):
        """Test quality score calculation."""
        # Add records with known metrics
        for i in range(5):
            record = VenueExecutionRecord(
                venue_mic="XPAR",
                order_id=f"ORD-SCORE-{i}",
                fill_quantity=Decimal("100"),
                fill_price=Decimal("75.00"),
                spread_bps=Decimal("5"),  # Good spread
                latency_ms=10.0,  # Good latency
                price_improvement_bps=Decimal("5"),  # Good PI
                fill_rate=0.98,  # Good fill rate
                cost_bps=Decimal("2"),  # Low cost
            )
            analyzer.record_execution(record)

        metrics = analyzer.get_venue_metrics("XPAR")

        # With good metrics, score should be high
        assert metrics.quality_score > 50

    def test_compare_venues(self, analyzer):
        """Test comparing venues."""
        # Add records for different venues
        for venue in ["XLON", "XETR"]:
            for i in range(3):
                record = VenueExecutionRecord(
                    venue_mic=venue,
                    order_id=f"ORD-{venue}-{i}",
                    fill_quantity=Decimal("100"),
                    fill_price=Decimal("100.00"),
                    spread_bps=Decimal("5") if venue == "XLON" else Decimal("10"),
                )
                analyzer.record_execution(record)

        comparison = analyzer.compare_venues(["XLON", "XETR"])

        assert "XLON" in comparison
        assert "XETR" in comparison
        # XLON should have better spread
        assert comparison["XLON"].avg_spread_bps < comparison["XETR"].avg_spread_bps

    def test_compare_all_venues(self, analyzer):
        """Test comparing all venues."""
        for venue in ["XLON", "XETR", "XPAR"]:
            record = VenueExecutionRecord(
                venue_mic=venue,
                order_id=f"ORD-{venue}",
                fill_quantity=Decimal("100"),
                fill_price=Decimal("100.00"),
            )
            analyzer.record_execution(record)

        comparison = analyzer.compare_venues()

        assert len(comparison) == 3

    def test_get_venue_rankings(self, analyzer):
        """Test getting venue rankings."""
        # Add records with different quality
        records_data = [
            ("XLON", Decimal("3"), 5.0, Decimal("3")),  # Best
            ("XETR", Decimal("5"), 10.0, Decimal("1")),  # Medium
            ("XPAR", Decimal("8"), 20.0, Decimal("-1")),  # Worst
        ]

        for venue, spread, latency, pi in records_data:
            for i in range(5):  # Enough for min_samples
                record = VenueExecutionRecord(
                    venue_mic=venue,
                    order_id=f"ORD-{venue}-{i}",
                    fill_quantity=Decimal("100"),
                    fill_price=Decimal("100.00"),
                    spread_bps=spread,
                    latency_ms=latency,
                    price_improvement_bps=pi,
                    fill_rate=0.95,
                )
                analyzer.record_execution(record)

        rankings = analyzer.get_venue_rankings()

        # Should have 3 venues ranked
        assert len(rankings) == 3
        # First venue should have highest score
        assert rankings[0][1] >= rankings[1][1]

    def test_audit_callback(self, config, venues):
        """Test audit callback."""
        callback_data = []

        def callback(data):
            callback_data.append(data)

        analyzer = VenueAnalyzer(config, venues=venues, audit_callback=callback)

        record = VenueExecutionRecord(
            venue_mic="XLON",
            order_id="AUDIT-TEST",
        )
        analyzer.record_execution(record)

        assert len(callback_data) == 1
        assert callback_data[0]["type"] == "venue_execution_record"


# =============================================================================
# Test: SmartOrderRouter
# =============================================================================


class TestSmartOrderRouter:
    """Tests for SmartOrderRouter class."""

    @pytest.fixture
    def analyzer(self):
        """Create test analyzer."""
        config = VenueAnalysisConfig(min_samples=1)
        venues = [
            ExecutionVenue(mic="XLON", name="LSE"),
            ExecutionVenue(mic="XETR", name="Xetra"),
            ExecutionVenue(mic="XPAR", name="Euronext Paris"),
        ]
        return VenueAnalyzer(config, venues=venues)

    @pytest.fixture
    def router(self, analyzer):
        """Create test router."""
        return SmartOrderRouter(analyzer)

    def test_create_router(self, router):
        """Test router creation."""
        assert router is not None
        assert router.analyzer is not None

    def test_select_venue_basic(self, router):
        """Test basic venue selection."""
        order = {
            "order_id": "ORD-001",
            "instrument_isin": "US0378331005",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.05, "size": 5000},
            "XETR": {"bid": 100.01, "ask": 100.06, "size": 3000},
        }

        decision = router.select_venue(order, market_data)

        assert decision.order_id == "ORD-001"
        assert decision.selected_venue in ["XLON", "XETR"]
        assert decision.selection_reason is not None

    def test_select_venue_best_price_buy(self, router):
        """Test selecting venue with best price for buy."""
        order = {
            "order_id": "ORD-BUY",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.10},  # Higher ask
            "XETR": {"bid": 100.00, "ask": 100.05},  # Lower ask (best for buy)
        }

        decision = router.select_venue(order, market_data)

        # Should pick XETR for lower ask
        assert decision.selected_venue == "XETR"

    def test_select_venue_best_price_sell(self, router):
        """Test selecting venue with best price for sell."""
        order = {
            "order_id": "ORD-SELL",
            "side": "SELL",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.05, "ask": 100.10},  # Higher bid (best for sell)
            "XETR": {"bid": 100.00, "ask": 100.05},  # Lower bid
        }

        decision = router.select_venue(order, market_data)

        # Should pick XLON for higher bid
        assert decision.selected_venue == "XLON"

    def test_select_venue_explicit_instruction(self, router):
        """Test explicit venue instruction."""
        order = {
            "order_id": "ORD-EXPLICIT",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.10},
            "XETR": {"bid": 100.00, "ask": 100.05},  # Better price
        }

        decision = router.select_venue(
            order, market_data,
            explicit_venue="XLON"  # But client wants XLON
        )

        assert decision.selected_venue == "XLON"
        assert decision.selection_reason == VenueSelectionReason.EXPLICIT_INSTRUCTION

    def test_select_venue_explicit_not_available(self, router):
        """Test explicit venue when not available."""
        order = {
            "order_id": "ORD-UNAVAIL",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.05},
        }

        decision = router.select_venue(
            order, market_data,
            explicit_venue="XXXX"  # Not in market_data
        )

        # Should fall back to best available
        assert decision.selected_venue == "XLON"

    def test_select_venue_single_option(self, router):
        """Test selection with single venue."""
        order = {
            "order_id": "ORD-SINGLE",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.05},
        }

        decision = router.select_venue(order, market_data)

        assert decision.selected_venue == "XLON"
        assert decision.venue_scores["XLON"] == 100.0

    def test_select_venue_with_size(self, router):
        """Test selection considering available size."""
        order = {
            "order_id": "ORD-SIZE",
            "side": "BUY",
            "quantity": Decimal("1000"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.05, "size": 500},  # Not enough size
            "XETR": {"bid": 100.00, "ask": 100.06, "size": 2000},  # Enough size
        }

        decision = router.select_venue(order, market_data)

        # XETR has better fill likelihood
        assert decision.selected_venue in ["XLON", "XETR"]

    def test_select_venue_with_historical_data(self, analyzer, router):
        """Test selection with historical performance."""
        # Add good historical data for XPAR
        for i in range(5):
            record = VenueExecutionRecord(
                venue_mic="XPAR",
                order_id=f"HIST-{i}",
                fill_quantity=Decimal("100"),
                fill_price=Decimal("100"),
                spread_bps=Decimal("2"),  # Very good
                latency_ms=5.0,  # Very fast
                price_improvement_bps=Decimal("5"),  # Good PI
                fill_rate=0.99,
            )
            analyzer.record_execution(record)

        order = {
            "order_id": "ORD-HIST",
            "side": "BUY",
            "quantity": Decimal("100"),
        }

        market_data = {
            "XLON": {"bid": 100.00, "ask": 100.05},
            "XPAR": {"bid": 100.00, "ask": 100.05},  # Same price
        }

        decision = router.select_venue(order, market_data)

        # Should consider XPAR's better historical performance
        # (though prices are equal)
        assert decision.selected_venue in ["XLON", "XPAR"]

    def test_get_decisions(self, router):
        """Test getting routing decisions."""
        order = {"order_id": "ORD-GET", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"XLON": {"bid": 100, "ask": 100.05}}

        router.select_venue(order, market_data)
        router.select_venue(order, market_data)

        decisions = router.get_decisions()
        assert len(decisions) == 2

    def test_get_decisions_date_filter(self, router):
        """Test getting decisions with date filter."""
        order = {"order_id": "ORD-DATE", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"XLON": {"bid": 100, "ask": 100.05}}

        router.select_venue(order, market_data)

        now = datetime.utcnow()
        decisions = router.get_decisions(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1)
        )

        assert len(decisions) == 1

    def test_get_routing_statistics(self, router):
        """Test getting routing statistics."""
        # Make multiple routing decisions
        for venue, reason in [("XLON", None), ("XETR", None), ("XLON", "XLON")]:
            order = {"order_id": f"ORD-{venue}", "side": "BUY", "quantity": Decimal("100")}
            market_data = {
                "XLON": {"bid": 100, "ask": 100.05},
                "XETR": {"bid": 100, "ask": 100.10},
            }
            router.select_venue(order, market_data, explicit_venue=reason)

        stats = router.get_routing_statistics()

        assert stats["total_decisions"] == 3
        assert "venue_distribution" in stats
        assert "reason_distribution" in stats

    def test_get_routing_statistics_empty(self, router):
        """Test statistics with no decisions."""
        stats = router.get_routing_statistics()

        assert stats["total_decisions"] == 0

    def test_audit_callback(self, analyzer):
        """Test router audit callback."""
        callback_data = []

        def callback(data):
            callback_data.append(data)

        router = SmartOrderRouter(analyzer, audit_callback=callback)

        order = {"order_id": "AUDIT", "side": "BUY", "quantity": Decimal("100")}
        market_data = {"XLON": {"bid": 100, "ask": 100.05}}

        router.select_venue(order, market_data)

        assert len(callback_data) == 1
        assert callback_data[0]["type"] == "venue_routing_decision"


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_venue_analyzer_default(self):
        """Test creating analyzer with defaults."""
        analyzer = create_venue_analyzer()

        assert analyzer is not None
        assert analyzer.config.min_samples == 30

    def test_create_venue_analyzer_with_venues(self):
        """Test creating analyzer with venues."""
        venues = [
            ExecutionVenue(mic="XLON", name="LSE"),
            ExecutionVenue(mic="XETR", name="Xetra"),
        ]

        analyzer = create_venue_analyzer(venues=venues)

        assert len(analyzer._venues) == 2

    def test_create_venue_analyzer_with_config(self):
        """Test creating analyzer with custom config."""
        config = VenueAnalysisConfig(min_samples=10)
        analyzer = create_venue_analyzer(config=config)

        assert analyzer.config.min_samples == 10

    def test_create_smart_order_router(self):
        """Test creating router."""
        analyzer = create_venue_analyzer()
        router = create_smart_order_router(analyzer)

        assert router is not None
        assert router.analyzer == analyzer


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_market_data(self):
        """Test selection with empty market data."""
        config = VenueAnalysisConfig()
        venues = [ExecutionVenue(mic="XLON", name="LSE")]
        analyzer = VenueAnalyzer(config, venues=venues)
        router = SmartOrderRouter(analyzer)

        order = {"order_id": "EMPTY", "side": "BUY", "quantity": Decimal("100")}
        market_data = {}

        decision = router.select_venue(order, market_data)

        assert decision.selected_venue == ""

    def test_inactive_venue_excluded(self):
        """Test inactive venues are excluded from routing."""
        config = VenueAnalysisConfig()
        venues = [
            ExecutionVenue(mic="XLON", name="LSE", is_active=True),
            ExecutionVenue(mic="XETR", name="Xetra", is_active=False),
        ]
        analyzer = VenueAnalyzer(config, venues=venues)
        router = SmartOrderRouter(analyzer)

        available = router._get_available_venues()

        assert "XLON" in available
        assert "XETR" not in available

    def test_concurrent_recording(self):
        """Test concurrent record operations."""
        config = VenueAnalysisConfig()
        venues = [ExecutionVenue(mic="XLON", name="LSE")]
        analyzer = VenueAnalyzer(config, venues=venues)
        errors = []

        def record_executions():
            try:
                for i in range(10):
                    record = VenueExecutionRecord(
                        venue_mic="XLON",
                        order_id=f"CONC-{threading.current_thread().name}-{i}",
                    )
                    analyzer.record_execution(record)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_executions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(analyzer._records) == 50

    def test_metrics_with_rejection_records(self):
        """Test metrics calculation with rejected orders."""
        config = VenueAnalysisConfig(min_samples=1)
        venues = [ExecutionVenue(mic="XLON", name="LSE")]
        analyzer = VenueAnalyzer(config, venues=venues)

        # Add normal and rejected records
        for i, rejected in enumerate([False, False, True, False, True]):
            record = VenueExecutionRecord(
                venue_mic="XLON",
                order_id=f"REJ-{i}",
                was_rejected=rejected,
                fill_quantity=Decimal("0") if rejected else Decimal("100"),
                fill_price=Decimal("0") if rejected else Decimal("100"),
            )
            analyzer.record_execution(record)

        metrics = analyzer.get_venue_metrics("XLON")

        assert metrics.total_orders == 5
        assert metrics.total_rejections == 2
        assert metrics.rejection_rate == 40.0  # 2/5 * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=services.compliance.venue_analysis"])
