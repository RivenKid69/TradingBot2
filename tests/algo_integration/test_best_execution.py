# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 5: Best Execution Policy (Article 27).

This module provides comprehensive tests for the best_execution module:
- ExecutionFactor enum
- ExecutionVenue dataclass
- FactorWeights dataclass
- ExecutionAnalysis dataclass
- BestExecutionPolicyConfig dataclass
- BestExecutionPolicy class
- BestExecutionAnalyzer class
- Factory functions

Target: 100% test coverage
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
import json
import time

from services.algo_integration.best_execution import (
    # Enums
    ExecutionFactor,
    AssetClass,
    OrderCategory,
    VenueType,
    ExecutionQualityLevel,
    # Data classes
    ExecutionVenue,
    FactorWeights,
    ExecutionAnalysis,
    BestExecutionPolicyConfig,
    # Main classes
    BestExecutionPolicy,
    BestExecutionAnalyzer,
    # Factory functions
    create_best_execution_policy,
    create_best_execution_analyzer,
    # Utilities
    get_standard_eu_venues,
)


# =============================================================================
# Test: Enums
# =============================================================================


class TestExecutionFactor:
    """Tests for ExecutionFactor enum."""

    def test_all_seven_factors_defined(self):
        """Verify all 7 MiFID II factors are defined."""
        assert len(ExecutionFactor) == 7
        assert ExecutionFactor.PRICE.value == "price"
        assert ExecutionFactor.COST.value == "cost"
        assert ExecutionFactor.SPEED.value == "speed"
        assert ExecutionFactor.LIKELIHOOD.value == "likelihood"
        assert ExecutionFactor.SETTLEMENT.value == "settlement"
        assert ExecutionFactor.SIZE.value == "size"
        assert ExecutionFactor.NATURE.value == "nature"


class TestAssetClass:
    """Tests for AssetClass enum."""

    def test_asset_classes_defined(self):
        """Verify asset classes."""
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.FIXED_INCOME.value == "fixed_income"
        assert AssetClass.DERIVATIVES.value == "derivatives"
        assert AssetClass.FX.value == "fx"
        assert AssetClass.COMMODITIES.value == "commodities"
        assert AssetClass.CRYPTO.value == "crypto"


class TestOrderCategory:
    """Tests for OrderCategory enum."""

    def test_order_categories_defined(self):
        """Verify order categories."""
        assert OrderCategory.SMALL.value == "small"
        assert OrderCategory.LARGE.value == "large"
        assert OrderCategory.BLOCK.value == "block"
        assert OrderCategory.ALGORITHMIC.value == "algorithmic"


class TestVenueType:
    """Tests for VenueType enum."""

    def test_venue_types_defined(self):
        """Verify venue types per MiFID II."""
        assert VenueType.REGULATED_MARKET.value == "regulated_market"
        assert VenueType.MTF.value == "mtf"
        assert VenueType.OTF.value == "otf"
        assert VenueType.SYSTEMATIC_INTERNALISER.value == "si"
        assert VenueType.DARK_POOL.value == "dark_pool"


class TestExecutionQualityLevel:
    """Tests for ExecutionQualityLevel enum."""

    def test_quality_levels_defined(self):
        """Verify quality levels."""
        assert ExecutionQualityLevel.EXCELLENT.value == "excellent"
        assert ExecutionQualityLevel.GOOD.value == "good"
        assert ExecutionQualityLevel.ACCEPTABLE.value == "acceptable"
        assert ExecutionQualityLevel.BELOW_STANDARD.value == "below_standard"
        assert ExecutionQualityLevel.POOR.value == "poor"


# =============================================================================
# Test: ExecutionVenue
# =============================================================================


class TestExecutionVenue:
    """Tests for ExecutionVenue dataclass."""

    def test_create_venue(self):
        """Test creating execution venue."""
        venue = ExecutionVenue(
            mic="XLON",
            name="London Stock Exchange",
            venue_type=VenueType.REGULATED_MARKET,
            country="GB",
            currency="GBP",
            ranking=1,
        )

        assert venue.mic == "XLON"
        assert venue.name == "London Stock Exchange"
        assert venue.venue_type == VenueType.REGULATED_MARKET
        assert venue.country == "GB"
        assert venue.currency == "GBP"
        assert venue.ranking == 1
        assert venue.is_active is True

    def test_venue_with_metrics(self):
        """Test venue with performance metrics."""
        venue = ExecutionVenue(
            mic="XETR",
            name="Xetra",
            avg_spread_bps=Decimal("4.5"),
            avg_latency_ms=5.0,
            fill_rate_pct=98.5,
            cost_bps=Decimal("2.5"),
            avg_price_improvement_bps=Decimal("1.5"),
        )

        assert venue.avg_spread_bps == Decimal("4.5")
        assert venue.avg_latency_ms == 5.0
        assert venue.fill_rate_pct == 98.5
        assert venue.cost_bps == Decimal("2.5")
        assert venue.avg_price_improvement_bps == Decimal("1.5")

    def test_venue_to_dict(self):
        """Test venue serialization."""
        venue = ExecutionVenue(
            mic="XPAR",
            name="Euronext Paris",
            ranking=2,
            last_review_date=date(2024, 1, 15),
        )

        d = venue.to_dict()

        assert d["mic"] == "XPAR"
        assert d["name"] == "Euronext Paris"
        assert d["ranking"] == 2
        assert d["last_review_date"] == "2024-01-15"

    def test_venue_from_dict(self):
        """Test venue deserialization."""
        data = {
            "mic": "XAMS",
            "name": "Euronext Amsterdam",
            "venue_type": "mtf",
            "country": "NL",
            "avg_spread_bps": "5.0",
            "fill_rate_pct": 97.0,
        }

        venue = ExecutionVenue.from_dict(data)

        assert venue.mic == "XAMS"
        assert venue.venue_type == VenueType.MTF
        assert venue.avg_spread_bps == Decimal("5.0")

    def test_venue_immutable(self):
        """Test venue is frozen (immutable)."""
        venue = ExecutionVenue(mic="XLON", name="LSE")

        with pytest.raises(Exception):  # FrozenInstanceError
            venue.mic = "XETR"


# =============================================================================
# Test: FactorWeights
# =============================================================================


class TestFactorWeights:
    """Tests for FactorWeights dataclass."""

    def test_default_weights_sum_to_one(self):
        """Test default weights sum to 1.0."""
        weights = FactorWeights()

        total = (
            weights.price
            + weights.cost
            + weights.speed
            + weights.likelihood
            + weights.settlement
            + weights.size
            + weights.nature
        )

        assert abs(total - 1.0) < 0.001

    def test_custom_weights_validation(self):
        """Test custom weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            FactorWeights(price=0.5, cost=0.5, speed=0.5)  # Sum > 1.0

    def test_validate_method(self):
        """Test validation returns errors list."""
        weights = FactorWeights()
        # Manually override to test validation
        object.__setattr__(weights, "price", 2.0)

        errors = weights.validate()
        assert len(errors) > 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        weights = FactorWeights()
        d = weights.to_dict()

        assert ExecutionFactor.PRICE in d
        assert d[ExecutionFactor.PRICE] == weights.price

    def test_get_weight(self):
        """Test getting weight by factor."""
        weights = FactorWeights(
            price=0.40,
            cost=0.20,
            speed=0.15,
            likelihood=0.15,
            settlement=0.05,
            size=0.03,
            nature=0.02,
        )

        assert weights.get_weight(ExecutionFactor.PRICE) == 0.40
        assert weights.get_weight(ExecutionFactor.COST) == 0.20

    def test_for_retail_client(self):
        """Test retail client weights prioritize cost."""
        weights = FactorWeights.for_retail_client()

        # Retail: cost should be weighted higher
        assert weights.cost > weights.speed
        total = sum(
            [
                weights.price,
                weights.cost,
                weights.speed,
                weights.likelihood,
                weights.settlement,
                weights.size,
                weights.nature,
            ]
        )
        assert abs(total - 1.0) < 0.001

    def test_for_professional_client(self):
        """Test professional client weights."""
        weights = FactorWeights.for_professional_client()

        total = sum(
            [
                weights.price,
                weights.cost,
                weights.speed,
                weights.likelihood,
                weights.settlement,
                weights.size,
                weights.nature,
            ]
        )
        assert abs(total - 1.0) < 0.001

    def test_for_proprietary_trading(self):
        """Test proprietary trading weights prioritize price."""
        weights = FactorWeights.for_proprietary_trading()

        assert weights.price >= 0.40  # Price most important
        total = sum(
            [
                weights.price,
                weights.cost,
                weights.speed,
                weights.likelihood,
                weights.settlement,
                weights.size,
                weights.nature,
            ]
        )
        assert abs(total - 1.0) < 0.001


# =============================================================================
# Test: ExecutionAnalysis
# =============================================================================


class TestExecutionAnalysis:
    """Tests for ExecutionAnalysis dataclass."""

    def test_create_analysis(self):
        """Test creating execution analysis."""
        analysis = ExecutionAnalysis(
            order_id="ORD-123",
            instrument_isin="US0378331005",
            venue_mic="XLON",
            side="BUY",
        )

        assert analysis.order_id == "ORD-123"
        assert analysis.execution_id is not None
        assert analysis.analysis_timestamp_ns > 0

    def test_analysis_with_scores(self):
        """Test analysis with calculated scores."""
        analysis = ExecutionAnalysis(
            price_score=0.8,
            cost_score=0.7,
            speed_score=0.9,
            likelihood_score=0.95,
            settlement_score=1.0,
            size_score=0.85,
            nature_score=1.0,
            weighted_score=0.82,
            quality_level=ExecutionQualityLevel.EXCELLENT,
        )

        assert analysis.price_score == 0.8
        assert analysis.weighted_score == 0.82
        assert analysis.quality_level == ExecutionQualityLevel.EXCELLENT

    def test_analysis_to_dict(self):
        """Test analysis serialization."""
        analysis = ExecutionAnalysis(
            order_id="ORD-456",
            reference_price=Decimal("100.50"),
            execution_price=Decimal("100.48"),
            price_improvement_bps=Decimal("1.99"),
        )

        d = analysis.to_dict()

        assert d["order_id"] == "ORD-456"
        assert d["reference_price"] == "100.50"
        assert d["execution_price"] == "100.48"

    def test_analysis_from_dict(self):
        """Test analysis deserialization."""
        data = {
            "order_id": "ORD-789",
            "execution_price": "150.25",
            "weighted_score": 0.75,
            "quality_level": "good",
        }

        analysis = ExecutionAnalysis.from_dict(data)

        assert analysis.order_id == "ORD-789"
        assert analysis.execution_price == Decimal("150.25")
        assert analysis.quality_level == ExecutionQualityLevel.GOOD

    def test_analysis_notes(self):
        """Test analysis with notes."""
        analysis = ExecutionAnalysis()
        analysis.notes.append("Price improvement below threshold")
        analysis.notes.append("High latency detected")

        assert len(analysis.notes) == 2


# =============================================================================
# Test: BestExecutionPolicyConfig
# =============================================================================


class TestBestExecutionPolicyConfig:
    """Tests for BestExecutionPolicyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = BestExecutionPolicyConfig()

        assert config.policy_version == "1.0.0"
        assert config.default_factor_weights is not None
        assert config.review_frequency_days == 365

    def test_config_with_custom_weights(self):
        """Test config with custom factor weights."""
        weights = FactorWeights.for_retail_client()
        config = BestExecutionPolicyConfig(
            default_factor_weights=weights,
            firm_lei="5493001KJTIIGC8Y1R12",
        )

        assert config.default_factor_weights == weights
        assert config.firm_lei == "5493001KJTIIGC8Y1R12"

    def test_config_with_asset_class_weights(self):
        """Test config with asset class specific weights."""
        equity_weights = FactorWeights.for_proprietary_trading()
        fx_weights = FactorWeights(
            price=0.30,
            cost=0.20,
            speed=0.25,
            likelihood=0.15,
            settlement=0.05,
            size=0.03,
            nature=0.02,
        )

        config = BestExecutionPolicyConfig(
            asset_class_weights={
                AssetClass.EQUITY: equity_weights,
                AssetClass.FX: fx_weights,
            }
        )

        assert AssetClass.EQUITY in config.asset_class_weights
        assert AssetClass.FX in config.asset_class_weights

    def test_validate_config(self):
        """Test configuration validation."""
        config = BestExecutionPolicyConfig(
            approved_by="",  # Missing required field
        )

        errors = config.validate()
        assert any("approved_by" in err for err in errors)

    def test_validate_review_date(self):
        """Test review date validation."""
        config = BestExecutionPolicyConfig(
            approved_by="Compliance Officer",
            effective_date=date(2024, 1, 1),
            review_date=date(2023, 1, 1),  # Before effective
        )

        errors = config.validate()
        assert any("review_date must be after" in e for e in errors)

    def test_get_factor_weights_default(self):
        """Test getting default factor weights."""
        config = BestExecutionPolicyConfig()
        weights = config.get_factor_weights()

        assert weights == config.default_factor_weights

    def test_get_factor_weights_by_asset_class(self):
        """Test getting weights by asset class."""
        equity_weights = FactorWeights.for_proprietary_trading()
        config = BestExecutionPolicyConfig(asset_class_weights={AssetClass.EQUITY: equity_weights})

        weights = config.get_factor_weights(asset_class=AssetClass.EQUITY)
        assert weights == equity_weights

    def test_get_factor_weights_by_order_category(self):
        """Test order category overrides asset class."""
        block_weights = FactorWeights(
            price=0.25,
            cost=0.20,
            speed=0.10,
            likelihood=0.20,
            settlement=0.10,
            size=0.10,
            nature=0.05,
        )
        config = BestExecutionPolicyConfig(
            order_category_weights={OrderCategory.BLOCK: block_weights}
        )

        weights = config.get_factor_weights(
            asset_class=AssetClass.EQUITY, order_category=OrderCategory.BLOCK
        )
        assert weights == block_weights

    def test_config_to_dict(self):
        """Test config serialization."""
        config = BestExecutionPolicyConfig(
            policy_version="2.0.0",
            firm_lei="TEST123",
        )

        d = config.to_dict()

        assert d["policy_version"] == "2.0.0"
        assert d["firm_lei"] == "TEST123"


# =============================================================================
# Test: BestExecutionPolicy
# =============================================================================


class TestBestExecutionPolicy:
    """Tests for BestExecutionPolicy class."""

    @pytest.fixture
    def policy(self):
        """Create test policy."""
        config = BestExecutionPolicyConfig(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Trading Firm",
            approved_by="Compliance Officer",
            effective_date=date.today(),
            review_date=date.today() + timedelta(days=365),
        )
        return BestExecutionPolicy(config)

    def test_create_policy(self, policy):
        """Test policy creation."""
        assert policy is not None
        assert policy.config.firm_lei == "5493001KJTIIGC8Y1R12"
        assert policy.policy_hash is not None

    def test_validate_policy(self, policy):
        """Test policy validation."""
        errors = policy.validate()
        assert len(errors) == 0
        assert policy.is_valid()

    def test_is_review_due(self, policy):
        """Test review due check."""
        # Review date is in the future
        assert not policy.is_review_due()

        # Modify to past date
        policy.config.review_date = date.today() - timedelta(days=1)
        assert policy.is_review_due()

    def test_days_until_review(self, policy):
        """Test days until review calculation."""
        days = policy.days_until_review()
        assert days > 0
        assert days <= 365

    def test_get_factor_weights(self, policy):
        """Test getting factor weights."""
        weights = policy.get_factor_weights()
        assert isinstance(weights, FactorWeights)

    def test_add_venue(self, policy):
        """Test adding venue to policy."""
        venue = ExecutionVenue(
            mic="XLON",
            name="London Stock Exchange",
            ranking=1,
        )

        policy.add_venue(AssetClass.EQUITY, venue)

        venues = policy.get_venues(AssetClass.EQUITY)
        assert len(venues) == 1
        assert venues[0].mic == "XLON"

    def test_get_top_venue(self, policy):
        """Test getting top venue."""
        venue1 = ExecutionVenue(mic="XETR", name="Xetra", ranking=2)
        venue2 = ExecutionVenue(mic="XLON", name="LSE", ranking=1)

        policy.add_venue(AssetClass.EQUITY, venue1)
        policy.add_venue(AssetClass.EQUITY, venue2)

        top = policy.get_top_venue(AssetClass.EQUITY)
        assert top.mic == "XLON"  # Ranking 1 is best

    def test_update_venue_ranking(self, policy):
        """Test updating venue ranking."""
        venue = ExecutionVenue(mic="XPAR", name="Euronext Paris", ranking=5)
        policy.add_venue(AssetClass.EQUITY, venue)

        result = policy.update_venue_ranking(AssetClass.EQUITY, "XPAR", 2)

        assert result is True
        updated = policy.get_venues(AssetClass.EQUITY)[0]
        assert updated.ranking == 2

    def test_update_venue_ranking_not_found(self, policy):
        """Test updating non-existent venue."""
        result = policy.update_venue_ranking(AssetClass.EQUITY, "XXXX", 1)
        assert result is False

    def test_mark_reviewed(self, policy):
        """Test marking policy as reviewed."""
        old_hash = policy.policy_hash

        policy.mark_reviewed("John Smith")

        assert policy.config.approved_by == "John Smith"
        assert policy.config.review_date > date.today()
        assert policy.policy_hash != old_hash  # Hash should change

    def test_increment_version(self, policy):
        """Test incrementing policy version."""
        policy.increment_version("2.0.0")
        assert policy.version == "2.0.0"

    def test_generate_policy_document(self, policy):
        """Test generating formal policy document."""
        doc = policy.generate_policy_document()

        assert "BEST EXECUTION POLICY" in doc
        assert policy.config.firm_lei in doc
        assert "Price:" in doc
        assert "Cost:" in doc
        assert policy.policy_hash in doc

    def test_policy_hash_changes_on_modification(self, policy):
        """Test policy hash changes when content changes."""
        old_hash = policy.policy_hash

        venue = ExecutionVenue(mic="XMIL", name="Borsa Italiana", ranking=5)
        policy.add_venue(AssetClass.EQUITY, venue)

        assert policy.policy_hash != old_hash


# =============================================================================
# Test: BestExecutionAnalyzer
# =============================================================================


class TestBestExecutionAnalyzer:
    """Tests for BestExecutionAnalyzer class."""

    @pytest.fixture
    def policy(self):
        """Create test policy."""
        config = BestExecutionPolicyConfig(
            firm_lei="5493001KJTIIGC8Y1R12",
            approved_by="Compliance",
        )
        return BestExecutionPolicy(config)

    @pytest.fixture
    def analyzer(self, policy):
        """Create test analyzer."""
        return BestExecutionAnalyzer(policy)

    def test_create_analyzer(self, analyzer):
        """Test analyzer creation."""
        assert analyzer is not None
        assert analyzer.policy is not None

    def test_analyze_execution_buy_order(self, analyzer):
        """Test analyzing a buy order execution."""
        order = {
            "order_id": "ORD-001",
            "side": "BUY",
            "quantity": Decimal("100"),
            "submit_time_ms": 1000000,
            "instrument_isin": "US0378331005",
        }

        fill = {
            "price": Decimal("100.50"),
            "quantity": Decimal("100"),
            "fill_time_ms": 1000100,  # 100ms later
            "commission": Decimal("1.00"),
            "fees": Decimal("0.50"),
            "venue_mic": "XLON",
        }

        market_data = {
            "bid": Decimal("100.45"),
            "ask": Decimal("100.55"),
            "spread_bps": Decimal("10"),
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        assert analysis.order_id == "ORD-001"
        assert analysis.execution_price == Decimal("100.50")
        assert analysis.latency_ms == 100.0
        assert analysis.fill_rate == 1.0
        assert analysis.quality_level is not None

    def test_analyze_execution_sell_order(self, analyzer):
        """Test analyzing a sell order execution."""
        order = {
            "order_id": "ORD-002",
            "side": "SELL",
            "quantity": Decimal("50"),
            "submit_time_ms": 2000000,
        }

        fill = {
            "price": Decimal("100.00"),  # Selling at ask price = good execution
            "quantity": Decimal("50"),
            "fill_time_ms": 2000050,
            "venue_mic": "XETR",
        }

        market_data = {
            "bid": Decimal("99.90"),
            "ask": Decimal("100.00"),
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        assert analysis.side == "SELL"
        # Mid = 99.95, sell at 100.00 = positive price improvement
        assert analysis.price_improvement_bps > Decimal("0")

    def test_analyze_execution_partial_fill(self, analyzer):
        """Test analyzing partial fill."""
        order = {
            "order_id": "ORD-003",
            "side": "BUY",
            "quantity": Decimal("1000"),
            "submit_time_ms": 3000000,
        }

        fill = {
            "price": Decimal("50.00"),
            "quantity": Decimal("500"),  # Only 50% filled
            "fill_time_ms": 3000500,
        }

        market_data = {
            "mid": Decimal("50.00"),
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        assert analysis.fill_rate == 0.5
        assert analysis.likelihood_score == 0.5

    def test_analyze_execution_with_adv(self, analyzer):
        """Test analyzing execution with ADV for market impact."""
        order = {
            "order_id": "ORD-004",
            "side": "BUY",
            "quantity": Decimal("10000"),
            "submit_time_ms": 4000000,
        }

        fill = {
            "price": Decimal("25.00"),
            "quantity": Decimal("10000"),
            "fill_time_ms": 4000100,
        }

        market_data = {
            "bid": Decimal("24.95"),
            "ask": Decimal("25.05"),
            "adv": Decimal("100000"),  # ADV = 100k, order = 10% of ADV
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        # Should have market impact calculated
        assert analysis.market_impact_bps >= Decimal("0")

    def test_analyze_execution_with_asset_class(self, analyzer):
        """Test analyzing execution with specific asset class."""
        order = {
            "order_id": "ORD-005",
            "side": "BUY",
            "quantity": Decimal("100"),
            "submit_time_ms": 5000000,
        }

        fill = {
            "price": Decimal("150.00"),
            "quantity": Decimal("100"),
            "fill_time_ms": 5000050,
        }

        market_data = {
            "mid": Decimal("150.00"),
        }

        analysis = analyzer.analyze_execution(
            order, fill, market_data, asset_class=AssetClass.EQUITY
        )

        assert analysis is not None

    def test_analyze_execution_with_order_category(self, analyzer):
        """Test analyzing execution with order category."""
        order = {
            "order_id": "ORD-006",
            "side": "SELL",
            "quantity": Decimal("100000"),
            "submit_time_ms": 6000000,
            "category": "block",
        }

        fill = {
            "price": Decimal("75.00"),
            "quantity": Decimal("100000"),
            "fill_time_ms": 6001000,
        }

        market_data = {
            "mid": Decimal("75.00"),
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        assert analysis.order_category == OrderCategory.BLOCK

    def test_quality_level_excellent(self, analyzer):
        """Test excellent quality classification."""
        order = {
            "order_id": "ORD-EX",
            "side": "BUY",
            "quantity": Decimal("100"),
            "submit_time_ms": 0,
        }

        fill = {
            "price": Decimal("99.00"),  # Below mid = good for buy
            "quantity": Decimal("100"),
            "fill_time_ms": 10,  # Very fast
        }

        market_data = {
            "bid": Decimal("99.00"),
            "ask": Decimal("101.00"),  # Wide spread, buying at bid = excellent
        }

        analysis = analyzer.analyze_execution(order, fill, market_data)

        # With significant price improvement, should be excellent
        assert analysis.weighted_score >= 0.5

    def test_get_aggregate_metrics(self, analyzer):
        """Test getting aggregate metrics."""
        # Add some analyses
        for i in range(5):
            order = {
                "order_id": f"ORD-{i}",
                "side": "BUY",
                "quantity": Decimal("100"),
                "submit_time_ms": i * 1000,
            }
            fill = {
                "price": Decimal("100.00"),
                "quantity": Decimal("100"),
                "fill_time_ms": i * 1000 + 50,
                "venue_mic": "XLON",
            }
            market_data = {"mid": Decimal("100.00")}
            analyzer.analyze_execution(order, fill, market_data)

        metrics = analyzer.get_aggregate_metrics()

        assert metrics["count"] == 5
        assert "avg_weighted_score" in metrics
        assert "quality_distribution" in metrics

    def test_get_aggregate_metrics_empty(self, analyzer):
        """Test aggregate metrics with no data."""
        metrics = analyzer.get_aggregate_metrics()

        assert metrics["count"] == 0
        assert metrics["avg_weighted_score"] == 0.0

    def test_get_aggregate_metrics_filtered(self, analyzer):
        """Test filtered aggregate metrics."""
        # Add analyses at different venues
        for venue in ["XLON", "XETR", "XLON"]:
            order = {
                "order_id": f"ORD-{venue}",
                "side": "BUY",
                "quantity": Decimal("100"),
                "submit_time_ms": 0,
            }
            fill = {
                "price": Decimal("100.00"),
                "quantity": Decimal("100"),
                "fill_time_ms": 50,
                "venue_mic": venue,
            }
            analyzer.analyze_execution(order, fill, {"mid": Decimal("100.00")})

        metrics = analyzer.get_aggregate_metrics(venue_mic="XLON")

        assert metrics["count"] == 2  # Only XLON orders

    def test_compare_venues(self, analyzer):
        """Test comparing venues."""
        # Add analyses at different venues
        for venue in ["XLON", "XETR"]:
            for _ in range(3):
                order = {
                    "order_id": f"ORD-{venue}",
                    "side": "BUY",
                    "quantity": Decimal("100"),
                    "submit_time_ms": 0,
                }
                fill = {
                    "price": Decimal("100.00"),
                    "quantity": Decimal("100"),
                    "fill_time_ms": 50,
                    "venue_mic": venue,
                }
                analyzer.analyze_execution(order, fill, {"mid": Decimal("100.00")})

        comparison = analyzer.compare_venues(["XLON", "XETR"])

        assert "XLON" in comparison
        assert "XETR" in comparison

    def test_get_analyses(self, analyzer):
        """Test getting all analyses."""
        order = {"order_id": "TEST", "side": "BUY", "quantity": Decimal("100"), "submit_time_ms": 0}
        fill = {"price": Decimal("100"), "quantity": Decimal("100"), "fill_time_ms": 10}
        analyzer.analyze_execution(order, fill, {"mid": Decimal("100")})

        analyses = analyzer.get_analyses()

        assert len(analyses) == 1
        assert analyses[0].order_id == "TEST"

    def test_clear_analyses(self, analyzer):
        """Test clearing analyses."""
        order = {"order_id": "TEST", "side": "BUY", "quantity": Decimal("100"), "submit_time_ms": 0}
        fill = {"price": Decimal("100"), "quantity": Decimal("100"), "fill_time_ms": 10}
        analyzer.analyze_execution(order, fill, {"mid": Decimal("100")})

        count = analyzer.clear_analyses()

        assert count == 1
        assert len(analyzer.get_analyses()) == 0

    def test_audit_callback(self, policy):
        """Test audit callback is called."""
        callback_called = []

        def callback(analysis):
            callback_called.append(analysis)

        analyzer = BestExecutionAnalyzer(policy, audit_callback=callback)

        order = {
            "order_id": "AUDIT",
            "side": "BUY",
            "quantity": Decimal("100"),
            "submit_time_ms": 0,
        }
        fill = {"price": Decimal("100"), "quantity": Decimal("100"), "fill_time_ms": 10}
        analyzer.analyze_execution(order, fill, {"mid": Decimal("100")})

        assert len(callback_called) == 1


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_best_execution_policy(self):
        """Test policy factory."""
        policy = create_best_execution_policy(
            firm_lei="5493001KJTIIGC8Y1R12",
            firm_name="Test Firm",
            approved_by="Test Approver",
        )

        assert policy is not None
        assert policy.config.firm_lei == "5493001KJTIIGC8Y1R12"
        assert policy.is_valid()

    def test_create_best_execution_policy_with_weights(self):
        """Test policy factory with custom weights."""
        weights = FactorWeights.for_retail_client()
        policy = create_best_execution_policy(
            factor_weights=weights,
        )

        assert policy.get_factor_weights() == weights

    def test_create_best_execution_analyzer(self):
        """Test analyzer factory."""
        policy = create_best_execution_policy()
        analyzer = create_best_execution_analyzer(policy)

        assert analyzer is not None
        assert analyzer.policy == policy

    def test_create_best_execution_analyzer_with_callback(self):
        """Test analyzer factory with callback."""
        policy = create_best_execution_policy()
        callback = Mock()
        analyzer = create_best_execution_analyzer(policy, audit_callback=callback)

        assert analyzer.audit_callback == callback


# =============================================================================
# Test: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_standard_eu_venues(self):
        """Test getting standard EU venues."""
        venues = get_standard_eu_venues()

        assert len(venues) >= 5  # At least 5 major venues
        mics = [v.mic for v in venues]
        assert "XLON" in mics
        assert "XETR" in mics
        assert "XPAR" in mics

    def test_standard_venues_have_metrics(self):
        """Test standard venues have performance metrics."""
        venues = get_standard_eu_venues()

        for venue in venues:
            assert venue.avg_spread_bps >= Decimal("0")
            assert venue.avg_latency_ms >= 0
            assert venue.fill_rate_pct > 0


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_analysis_with_zero_price(self):
        """Test analysis with zero reference price."""
        policy = create_best_execution_policy()
        analyzer = BestExecutionAnalyzer(policy)

        order = {"order_id": "ZERO", "side": "BUY", "quantity": Decimal("100"), "submit_time_ms": 0}
        fill = {"price": Decimal("100"), "quantity": Decimal("100"), "fill_time_ms": 10}
        market_data = {"mid": Decimal("0")}  # Zero price

        analysis = analyzer.analyze_execution(order, fill, market_data)

        # Should handle gracefully
        assert analysis is not None

    def test_analysis_with_zero_quantity(self):
        """Test analysis with zero quantity."""
        policy = create_best_execution_policy()
        analyzer = BestExecutionAnalyzer(policy)

        order = {
            "order_id": "ZERO-QTY",
            "side": "BUY",
            "quantity": Decimal("0"),
            "submit_time_ms": 0,
        }
        fill = {"price": Decimal("100"), "quantity": Decimal("0"), "fill_time_ms": 10}
        market_data = {"mid": Decimal("100")}

        analysis = analyzer.analyze_execution(order, fill, market_data)

        assert analysis.fill_rate == 1.0  # Division by zero handled

    def test_policy_with_empty_venues(self):
        """Test policy with no venues."""
        policy = create_best_execution_policy()

        venues = policy.get_venues(AssetClass.EQUITY)
        assert venues == []

        top = policy.get_top_venue(AssetClass.EQUITY)
        assert top is None

    def test_concurrent_analysis(
        self,
    ):
        """Test concurrent analysis operations."""
        import threading

        policy = create_best_execution_policy()
        analyzer = BestExecutionAnalyzer(policy)
        errors = []

        def analyze_orders():
            try:
                for i in range(10):
                    order = {
                        "order_id": f"CONC-{i}",
                        "side": "BUY",
                        "quantity": Decimal("100"),
                        "submit_time_ms": i,
                    }
                    fill = {
                        "price": Decimal("100"),
                        "quantity": Decimal("100"),
                        "fill_time_ms": i + 10,
                    }
                    analyzer.analyze_execution(order, fill, {"mid": Decimal("100")})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyze_orders) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(analyzer.get_analyses()) == 50  # 5 threads * 10 orders


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=services.compliance.best_execution"])
