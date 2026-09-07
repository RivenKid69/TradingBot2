# -*- coding: utf-8 -*-
"""
MiFID II Best Execution Policy - Article 27 Compliance.

This module implements best execution requirements per MiFID II Article 27:

"Investment firms must take all sufficient steps to obtain the best possible
result for their clients taking into account price, costs, speed, likelihood
of execution and settlement, size, nature or any other consideration relevant
to the execution of the order."

Key Components:
    - ExecutionFactor: The 7 best execution factors defined by MiFID II
    - ExecutionVenue: Trading venue metrics and rankings
    - BestExecutionPolicy: Documented execution policy
    - BestExecutionAnalyzer: Execution quality analysis
    - BestExecutionMonitor: Real-time execution monitoring

Requirements:
    1. Document execution policy with factor weightings
    2. Review policy at least annually
    3. Monitor execution quality continuously
    4. Demonstrate best execution to regulators

References:
    - MiFID II Article 27: Best Execution
        https://www.esma.europa.eu/sites/default/files/library/esma35-43-3088_final_report_review_of_mifid_ii_framework_on_best_execution_reports.pdf
    - ESMA Best Execution Guidelines (ESMA35-43-620)
    - RTS 27 (suspended) - Execution venue quality data
    - RTS 28 (repealed) - Annual execution quality reports
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Union,
    Sequence,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ExecutionFactor(Enum):
    """
    Best execution factors per MiFID II Article 27.

    These are the seven factors that must be considered when
    determining the best possible result for order execution.
    """

    PRICE = "price"
    """Price of execution - the actual price achieved."""

    COST = "cost"
    """Total costs including commission, fees, and implicit costs."""

    SPEED = "speed"
    """Speed of execution - time to fill."""

    LIKELIHOOD = "likelihood"
    """Likelihood of execution and settlement."""

    SETTLEMENT = "settlement"
    """Settlement certainty and timing."""

    SIZE = "size"
    """Ability to handle order size without market impact."""

    NATURE = "nature"
    """Nature of the order (e.g., algorithmic, program trade)."""


class AssetClass(Enum):
    """Asset classes for execution policy purposes."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    DERIVATIVES = "derivatives"
    FX = "fx"
    COMMODITIES = "commodities"
    STRUCTURED_PRODUCTS = "structured_products"
    ETF = "etf"
    CRYPTO = "crypto"  # For proprietary trading


class OrderCategory(Enum):
    """Order categories that may have different execution strategies."""

    SMALL = "small"  # Below 25% EBBO
    MEDIUM = "medium"  # 25-100% EBBO
    LARGE = "large"  # Above EBBO
    BLOCK = "block"  # Block trade size
    ALGORITHMIC = "algorithmic"  # Algo-generated orders
    CLIENT_LIMIT = "client_limit"  # Specific price instructions


class VenueType(Enum):
    """Types of execution venues per MiFID II."""

    REGULATED_MARKET = "regulated_market"  # RM
    MTF = "mtf"  # Multilateral Trading Facility
    OTF = "otf"  # Organised Trading Facility
    SYSTEMATIC_INTERNALISER = "si"  # SI
    MARKET_MAKER = "mm"  # Market Maker
    OTC = "otc"  # Over-the-Counter
    DARK_POOL = "dark_pool"  # Dark liquidity pool


class ExecutionQualityLevel(Enum):
    """Execution quality assessment levels."""

    EXCELLENT = "excellent"  # Top quartile
    GOOD = "good"  # Above median
    ACCEPTABLE = "acceptable"  # Median
    BELOW_STANDARD = "below_standard"  # Below median
    POOR = "poor"  # Bottom quartile


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class ExecutionVenue:
    """
    Execution venue with quality metrics.

    Captures venue characteristics and historical performance
    metrics used for best execution analysis.

    Attributes:
        mic: Market Identifier Code (ISO 10383)
        name: Venue name
        venue_type: Type of venue (RM, MTF, SI, etc.)
        country: Country code (ISO 3166-1 alpha-2)
        currency: Primary trading currency
        ranking: Firm's ranking for this venue (1 = best)
        avg_spread_bps: Average spread in basis points
        avg_latency_ms: Average order-to-fill latency
        fill_rate_pct: Historical fill rate percentage
        cost_bps: Average total cost in basis points
        avg_price_improvement_bps: Average price improvement
        market_share_pct: Venue's market share
        is_active: Whether venue is currently active
        last_review_date: Date of last venue review
        notes: Additional notes about venue
    """

    mic: str
    name: str
    venue_type: VenueType = VenueType.REGULATED_MARKET
    country: str = ""
    currency: str = "EUR"
    ranking: int = 0
    avg_spread_bps: Decimal = Decimal("0")
    avg_latency_ms: float = 0.0
    fill_rate_pct: float = 100.0
    cost_bps: Decimal = Decimal("0")
    avg_price_improvement_bps: Decimal = Decimal("0")
    market_share_pct: float = 0.0
    is_active: bool = True
    last_review_date: Optional[date] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mic": self.mic,
            "name": self.name,
            "venue_type": self.venue_type.value,
            "country": self.country,
            "currency": self.currency,
            "ranking": self.ranking,
            "avg_spread_bps": str(self.avg_spread_bps),
            "avg_latency_ms": self.avg_latency_ms,
            "fill_rate_pct": self.fill_rate_pct,
            "cost_bps": str(self.cost_bps),
            "avg_price_improvement_bps": str(self.avg_price_improvement_bps),
            "market_share_pct": self.market_share_pct,
            "is_active": self.is_active,
            "last_review_date": (
                self.last_review_date.isoformat() if self.last_review_date else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionVenue":
        """Create from dictionary."""
        return cls(
            mic=data["mic"],
            name=data["name"],
            venue_type=VenueType(data.get("venue_type", "regulated_market")),
            country=data.get("country", ""),
            currency=data.get("currency", "EUR"),
            ranking=data.get("ranking", 0),
            avg_spread_bps=Decimal(data.get("avg_spread_bps", "0")),
            avg_latency_ms=float(data.get("avg_latency_ms", 0)),
            fill_rate_pct=float(data.get("fill_rate_pct", 100)),
            cost_bps=Decimal(data.get("cost_bps", "0")),
            avg_price_improvement_bps=Decimal(data.get("avg_price_improvement_bps", "0")),
            market_share_pct=float(data.get("market_share_pct", 0)),
            is_active=data.get("is_active", True),
            last_review_date=(
                date.fromisoformat(data["last_review_date"])
                if data.get("last_review_date")
                else None
            ),
            notes=data.get("notes", ""),
        )


@dataclass
class FactorWeights:
    """
    Weights for best execution factors.

    Per MiFID II, factors must be prioritized based on the characteristics
    of the client, order, instrument, and execution venue.

    Weights must sum to 1.0.

    Default weights reflect typical prioritization for proprietary trading:
    - Price and cost are most important (0.35 + 0.25 = 0.60)
    - Speed and likelihood are secondary (0.15 + 0.15 = 0.30)
    - Settlement, size, nature are tertiary (0.05 + 0.03 + 0.02 = 0.10)
    """

    price: float = 0.35
    cost: float = 0.25
    speed: float = 0.15
    likelihood: float = 0.15
    settlement: float = 0.05
    size: float = 0.03
    nature: float = 0.02

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        self._validate()

    def _validate(self) -> None:
        """Validate that weights sum to 1.0."""
        total = (
            self.price
            + self.cost
            + self.speed
            + self.likelihood
            + self.settlement
            + self.size
            + self.nature
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")

    def validate(self) -> List[str]:
        """Validate and return list of errors."""
        errors = []
        total = (
            self.price
            + self.cost
            + self.speed
            + self.likelihood
            + self.settlement
            + self.size
            + self.nature
        )
        if abs(total - 1.0) > 0.001:
            errors.append(f"Factor weights sum to {total}, must be 1.0")

        for factor in ["price", "cost", "speed", "likelihood", "settlement", "size", "nature"]:
            value = getattr(self, factor)
            if value < 0 or value > 1:
                errors.append(f"Factor {factor} weight {value} must be between 0 and 1")

        return errors

    def to_dict(self) -> Dict[ExecutionFactor, float]:
        """Convert to dictionary keyed by ExecutionFactor."""
        return {
            ExecutionFactor.PRICE: self.price,
            ExecutionFactor.COST: self.cost,
            ExecutionFactor.SPEED: self.speed,
            ExecutionFactor.LIKELIHOOD: self.likelihood,
            ExecutionFactor.SETTLEMENT: self.settlement,
            ExecutionFactor.SIZE: self.size,
            ExecutionFactor.NATURE: self.nature,
        }

    def get_weight(self, factor: ExecutionFactor) -> float:
        """Get weight for a specific factor."""
        mapping = {
            ExecutionFactor.PRICE: self.price,
            ExecutionFactor.COST: self.cost,
            ExecutionFactor.SPEED: self.speed,
            ExecutionFactor.LIKELIHOOD: self.likelihood,
            ExecutionFactor.SETTLEMENT: self.settlement,
            ExecutionFactor.SIZE: self.size,
            ExecutionFactor.NATURE: self.nature,
        }
        return mapping.get(factor, 0.0)

    @classmethod
    def for_retail_client(cls) -> "FactorWeights":
        """Default weights for retail clients (total consideration)."""
        return cls(
            price=0.30,
            cost=0.35,  # Higher weight on cost for retail
            speed=0.10,
            likelihood=0.15,
            settlement=0.05,
            size=0.03,
            nature=0.02,
        )

    @classmethod
    def for_professional_client(cls) -> "FactorWeights":
        """Default weights for professional clients."""
        return cls(
            price=0.35,
            cost=0.20,
            speed=0.20,  # Higher weight on speed
            likelihood=0.15,
            settlement=0.05,
            size=0.03,
            nature=0.02,
        )

    @classmethod
    def for_proprietary_trading(cls) -> "FactorWeights":
        """Default weights for proprietary trading."""
        return cls(
            price=0.40,  # Price most important
            cost=0.20,
            speed=0.20,
            likelihood=0.10,
            settlement=0.05,
            size=0.03,
            nature=0.02,
        )


@dataclass
class ExecutionAnalysis:
    """
    Analysis of a single execution against best execution criteria.

    Provides detailed breakdown of execution quality by factor.
    """

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    analysis_timestamp_ns: int = field(default_factory=time.time_ns)

    # Price analysis
    reference_price: Decimal = Decimal("0")
    execution_price: Decimal = Decimal("0")
    price_improvement_bps: Decimal = Decimal("0")
    price_score: float = 0.0

    # Cost analysis
    explicit_cost_bps: Decimal = Decimal("0")
    implicit_cost_bps: Decimal = Decimal("0")
    total_cost_bps: Decimal = Decimal("0")
    cost_score: float = 0.0

    # Speed analysis
    order_submit_time_ms: int = 0
    fill_time_ms: int = 0
    latency_ms: float = 0.0
    speed_score: float = 0.0

    # Likelihood analysis
    fill_rate: float = 0.0
    partial_fills: int = 0
    likelihood_score: float = 0.0

    # Settlement analysis
    expected_settlement_date: Optional[date] = None
    settlement_certainty: float = 1.0
    settlement_score: float = 0.0

    # Size analysis
    order_size: Decimal = Decimal("0")
    filled_size: Decimal = Decimal("0")
    market_impact_bps: Decimal = Decimal("0")
    size_score: float = 0.0

    # Nature analysis
    order_category: OrderCategory = OrderCategory.SMALL
    nature_score: float = 0.0

    # Overall scores
    weighted_score: float = 0.0
    quality_level: ExecutionQualityLevel = ExecutionQualityLevel.ACCEPTABLE

    # Metadata
    venue_mic: str = ""
    instrument_isin: str = ""
    side: str = ""
    currency: str = "EUR"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/reporting."""
        return {
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "analysis_timestamp_ns": self.analysis_timestamp_ns,
            "reference_price": str(self.reference_price),
            "execution_price": str(self.execution_price),
            "price_improvement_bps": str(self.price_improvement_bps),
            "price_score": self.price_score,
            "explicit_cost_bps": str(self.explicit_cost_bps),
            "implicit_cost_bps": str(self.implicit_cost_bps),
            "total_cost_bps": str(self.total_cost_bps),
            "cost_score": self.cost_score,
            "latency_ms": self.latency_ms,
            "speed_score": self.speed_score,
            "fill_rate": self.fill_rate,
            "likelihood_score": self.likelihood_score,
            "settlement_certainty": self.settlement_certainty,
            "settlement_score": self.settlement_score,
            "order_size": str(self.order_size),
            "filled_size": str(self.filled_size),
            "market_impact_bps": str(self.market_impact_bps),
            "size_score": self.size_score,
            "order_category": self.order_category.value,
            "nature_score": self.nature_score,
            "weighted_score": self.weighted_score,
            "quality_level": self.quality_level.value,
            "venue_mic": self.venue_mic,
            "instrument_isin": self.instrument_isin,
            "side": self.side,
            "currency": self.currency,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionAnalysis":
        """Create from dictionary."""
        return cls(
            execution_id=data.get("execution_id", str(uuid.uuid4())),
            order_id=data.get("order_id", ""),
            analysis_timestamp_ns=data.get("analysis_timestamp_ns", time.time_ns()),
            reference_price=Decimal(data.get("reference_price", "0")),
            execution_price=Decimal(data.get("execution_price", "0")),
            price_improvement_bps=Decimal(data.get("price_improvement_bps", "0")),
            price_score=float(data.get("price_score", 0)),
            explicit_cost_bps=Decimal(data.get("explicit_cost_bps", "0")),
            implicit_cost_bps=Decimal(data.get("implicit_cost_bps", "0")),
            total_cost_bps=Decimal(data.get("total_cost_bps", "0")),
            cost_score=float(data.get("cost_score", 0)),
            latency_ms=float(data.get("latency_ms", 0)),
            speed_score=float(data.get("speed_score", 0)),
            fill_rate=float(data.get("fill_rate", 0)),
            likelihood_score=float(data.get("likelihood_score", 0)),
            settlement_certainty=float(data.get("settlement_certainty", 1)),
            settlement_score=float(data.get("settlement_score", 0)),
            order_size=Decimal(data.get("order_size", "0")),
            filled_size=Decimal(data.get("filled_size", "0")),
            market_impact_bps=Decimal(data.get("market_impact_bps", "0")),
            size_score=float(data.get("size_score", 0)),
            order_category=OrderCategory(data.get("order_category", "small")),
            nature_score=float(data.get("nature_score", 0)),
            weighted_score=float(data.get("weighted_score", 0)),
            quality_level=ExecutionQualityLevel(data.get("quality_level", "acceptable")),
            venue_mic=data.get("venue_mic", ""),
            instrument_isin=data.get("instrument_isin", ""),
            side=data.get("side", ""),
            currency=data.get("currency", "EUR"),
            notes=data.get("notes", []),
        )


@dataclass
class BestExecutionPolicyConfig:
    """
    Configuration for best execution policy.

    Attributes:
        policy_version: Version of the policy
        effective_date: When policy became effective
        review_date: Next scheduled review date
        approved_by: Person who approved the policy
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        factor_weights: Weights for execution factors
        asset_class_weights: Different weights per asset class
        order_category_weights: Different weights per order category
        venue_rankings: Ranked list of venues per asset class
        routing_rules: Rules for order routing
        conflict_of_interest_policy: CoI policy statement
        monitoring_frequency: How often to monitor execution quality
        review_frequency_days: How often to review policy
    """

    policy_version: str = "1.0.0"
    effective_date: Optional[date] = None
    review_date: Optional[date] = None
    approved_by: str = ""
    firm_lei: str = ""
    firm_name: str = ""

    # Factor weights
    default_factor_weights: FactorWeights = field(default_factory=FactorWeights)
    asset_class_weights: Dict[AssetClass, FactorWeights] = field(default_factory=dict)
    order_category_weights: Dict[OrderCategory, FactorWeights] = field(default_factory=dict)

    # Venue configuration
    venue_rankings: Dict[AssetClass, List[ExecutionVenue]] = field(default_factory=dict)

    # Routing rules
    routing_rules: Dict[str, Any] = field(default_factory=dict)

    # Policies
    order_handling_policy: str = ""
    conflict_of_interest_policy: str = ""

    # Monitoring
    monitoring_frequency_seconds: float = 60.0
    review_frequency_days: int = 365

    # Thresholds for execution quality alerts
    price_improvement_threshold_bps: Decimal = Decimal("-10")  # Alert if worse than -10bps
    cost_threshold_bps: Decimal = Decimal("50")  # Alert if cost > 50bps
    latency_threshold_ms: float = 1000.0  # Alert if latency > 1s
    fill_rate_threshold_pct: float = 90.0  # Alert if fill rate < 90%

    def validate(self) -> List[str]:
        """Validate policy configuration."""
        errors = []

        if not self.policy_version:
            errors.append("Missing policy_version")

        if not self.approved_by:
            errors.append("Missing approved_by (required per MiFID II)")

        if self.effective_date and self.review_date:
            if self.review_date <= self.effective_date:
                errors.append("review_date must be after effective_date")

        errors.extend(self.default_factor_weights.validate())

        for asset_class, weights in self.asset_class_weights.items():
            ac_errors = weights.validate()
            for err in ac_errors:
                errors.append(f"{asset_class.value}: {err}")

        return errors

    def get_factor_weights(
        self,
        asset_class: Optional[AssetClass] = None,
        order_category: Optional[OrderCategory] = None,
    ) -> FactorWeights:
        """
        Get appropriate factor weights for given context.

        Priority: order_category > asset_class > default
        """
        if order_category and order_category in self.order_category_weights:
            return self.order_category_weights[order_category]
        if asset_class and asset_class in self.asset_class_weights:
            return self.asset_class_weights[asset_class]
        return self.default_factor_weights

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_version": self.policy_version,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "approved_by": self.approved_by,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "default_factor_weights": asdict(self.default_factor_weights),
            "asset_class_weights": {
                ac.value: asdict(weights) for ac, weights in self.asset_class_weights.items()
            },
            "order_category_weights": {
                oc.value: asdict(weights) for oc, weights in self.order_category_weights.items()
            },
            "venue_rankings": {
                ac.value: [v.to_dict() for v in venues]
                for ac, venues in self.venue_rankings.items()
            },
            "routing_rules": self.routing_rules,
            "order_handling_policy": self.order_handling_policy,
            "conflict_of_interest_policy": self.conflict_of_interest_policy,
            "monitoring_frequency_seconds": self.monitoring_frequency_seconds,
            "review_frequency_days": self.review_frequency_days,
            "price_improvement_threshold_bps": str(self.price_improvement_threshold_bps),
            "cost_threshold_bps": str(self.cost_threshold_bps),
            "latency_threshold_ms": self.latency_threshold_ms,
            "fill_rate_threshold_pct": self.fill_rate_threshold_pct,
        }


# =============================================================================
# Best Execution Policy
# =============================================================================


class BestExecutionPolicy:
    """
    MiFID II Article 27 Best Execution Policy.

    Implements a documented best execution policy that:
    - Defines factor weights and priorities
    - Maintains venue rankings per asset class
    - Tracks policy versions and reviews
    - Generates policy documentation

    Per MiFID II:
    - Policy must be documented
    - Policy must be reviewed at least annually
    - Changes must be notified to clients (if applicable)
    """

    def __init__(self, config: BestExecutionPolicyConfig):
        """
        Initialize best execution policy.

        Args:
            config: Policy configuration
        """
        self.config = config
        self._lock = threading.Lock()
        self._policy_hash: Optional[str] = None

        # Set defaults if not provided
        if self.config.effective_date is None:
            self.config.effective_date = date.today()
        if self.config.review_date is None:
            self.config.review_date = self.config.effective_date + timedelta(days=365)

        # Compute initial hash
        self._update_hash()

    def _update_hash(self) -> None:
        """Update policy hash for change detection."""
        data = json.dumps(self.config.to_dict(), sort_keys=True)
        self._policy_hash = hashlib.sha256(data.encode()).hexdigest()

    def validate(self) -> List[str]:
        """Validate policy."""
        return self.config.validate()

    def is_valid(self) -> bool:
        """Check if policy is valid."""
        return len(self.validate()) == 0

    def is_review_due(self) -> bool:
        """Check if policy review is due."""
        if self.config.review_date is None:
            return True
        return date.today() >= self.config.review_date

    def days_until_review(self) -> int:
        """Get days until next review."""
        if self.config.review_date is None:
            return 0
        delta = self.config.review_date - date.today()
        return max(0, delta.days)

    def get_factor_weights(
        self,
        asset_class: Optional[AssetClass] = None,
        order_category: Optional[OrderCategory] = None,
    ) -> FactorWeights:
        """Get factor weights for given context."""
        return self.config.get_factor_weights(asset_class, order_category)

    def get_venues(self, asset_class: AssetClass) -> List[ExecutionVenue]:
        """Get ranked venues for asset class."""
        venues = self.config.venue_rankings.get(asset_class, [])
        return sorted(venues, key=lambda v: v.ranking)

    def get_top_venue(self, asset_class: AssetClass) -> Optional[ExecutionVenue]:
        """Get top-ranked venue for asset class."""
        venues = self.get_venues(asset_class)
        return venues[0] if venues else None

    def add_venue(self, asset_class: AssetClass, venue: ExecutionVenue) -> None:
        """Add venue to rankings."""
        with self._lock:
            if asset_class not in self.config.venue_rankings:
                self.config.venue_rankings[asset_class] = []
            self.config.venue_rankings[asset_class].append(venue)
            self._update_hash()

    def update_venue_ranking(
        self,
        asset_class: AssetClass,
        mic: str,
        new_ranking: int,
    ) -> bool:
        """Update venue ranking."""
        with self._lock:
            venues = self.config.venue_rankings.get(asset_class, [])
            for i, venue in enumerate(venues):
                if venue.mic == mic:
                    # Create new venue with updated ranking (frozen dataclass)
                    updated = ExecutionVenue(
                        mic=venue.mic,
                        name=venue.name,
                        venue_type=venue.venue_type,
                        country=venue.country,
                        currency=venue.currency,
                        ranking=new_ranking,
                        avg_spread_bps=venue.avg_spread_bps,
                        avg_latency_ms=venue.avg_latency_ms,
                        fill_rate_pct=venue.fill_rate_pct,
                        cost_bps=venue.cost_bps,
                        avg_price_improvement_bps=venue.avg_price_improvement_bps,
                        market_share_pct=venue.market_share_pct,
                        is_active=venue.is_active,
                        last_review_date=date.today(),
                        notes=venue.notes,
                    )
                    self.config.venue_rankings[asset_class][i] = updated
                    self._update_hash()
                    return True
            return False

    def mark_reviewed(self, reviewer: str) -> None:
        """Mark policy as reviewed."""
        with self._lock:
            self.config.review_date = date.today() + timedelta(
                days=self.config.review_frequency_days
            )
            self.config.approved_by = reviewer
            self._update_hash()
            logger.info(f"Policy reviewed by {reviewer}, next review: {self.config.review_date}")

    def increment_version(self, new_version: str) -> None:
        """Increment policy version."""
        with self._lock:
            self.config.policy_version = new_version
            self._update_hash()

    def generate_policy_document(self) -> str:
        """
        Generate formal policy document for compliance records.

        Returns:
            Formatted policy document text.
        """
        doc = f"""
================================================================================
BEST EXECUTION POLICY
================================================================================

Version: {self.config.policy_version}
Effective Date: {self.config.effective_date}
Next Review Date: {self.config.review_date}
Approved By: {self.config.approved_by}

Firm: {self.config.firm_name}
LEI: {self.config.firm_lei}

--------------------------------------------------------------------------------
1. SCOPE AND APPLICABILITY
--------------------------------------------------------------------------------

This Best Execution Policy applies to all order execution activities conducted
by {self.config.firm_name or 'the firm'} in accordance with MiFID II Article 27.

For proprietary trading, the firm takes all sufficient steps to obtain the best
possible result taking into account the factors specified below.

--------------------------------------------------------------------------------
2. EXECUTION FACTORS AND WEIGHTING
--------------------------------------------------------------------------------

Per MiFID II Article 27, the following factors are considered when determining
the best possible result:

Default Factor Weights:
  - Price:       {self.config.default_factor_weights.price * 100:.1f}%
  - Cost:        {self.config.default_factor_weights.cost * 100:.1f}%
  - Speed:       {self.config.default_factor_weights.speed * 100:.1f}%
  - Likelihood:  {self.config.default_factor_weights.likelihood * 100:.1f}%
  - Settlement:  {self.config.default_factor_weights.settlement * 100:.1f}%
  - Size:        {self.config.default_factor_weights.size * 100:.1f}%
  - Nature:      {self.config.default_factor_weights.nature * 100:.1f}%

"""

        # Add asset class specific weights
        if self.config.asset_class_weights:
            doc += "Asset Class Specific Weights:\n"
            for ac, weights in self.config.asset_class_weights.items():
                doc += f"\n  {ac.value.upper()}:\n"
                doc += f"    Price: {weights.price * 100:.1f}%, Cost: {weights.cost * 100:.1f}%, "
                doc += f"Speed: {weights.speed * 100:.1f}%, Likelihood: {weights.likelihood * 100:.1f}%\n"

        doc += """
--------------------------------------------------------------------------------
3. EXECUTION VENUES
--------------------------------------------------------------------------------

The firm has assessed and ranked the following execution venues based on
historical execution quality metrics:

"""

        # Add venue rankings
        for ac, venues in self.config.venue_rankings.items():
            doc += f"\n{ac.value.upper()}:\n"
            for venue in sorted(venues, key=lambda v: v.ranking):
                doc += f"  #{venue.ranking} {venue.name} ({venue.mic})"
                doc += f" - Spread: {venue.avg_spread_bps}bps, Latency: {venue.avg_latency_ms}ms\n"

        doc += f"""
--------------------------------------------------------------------------------
4. ORDER HANDLING POLICY
--------------------------------------------------------------------------------

{self.config.order_handling_policy or 'Orders are handled promptly and fairly, with priority given to client orders over proprietary orders where applicable.'}

--------------------------------------------------------------------------------
5. CONFLICTS OF INTEREST
--------------------------------------------------------------------------------

{self.config.conflict_of_interest_policy or 'The firm maintains policies to identify, prevent, and manage conflicts of interest that may arise in order execution.'}

--------------------------------------------------------------------------------
6. MONITORING AND REVIEW
--------------------------------------------------------------------------------

Execution quality is monitored on a continuous basis with formal reviews
conducted at intervals not exceeding {self.config.review_frequency_days} days.

Monitoring includes:
  - Price improvement vs. reference price
  - Total cost of execution
  - Execution latency
  - Fill rates
  - Market impact

Thresholds triggering review:
  - Price deterioration > {abs(self.config.price_improvement_threshold_bps)}bps
  - Total cost > {self.config.cost_threshold_bps}bps
  - Latency > {self.config.latency_threshold_ms}ms
  - Fill rate < {self.config.fill_rate_threshold_pct}%

--------------------------------------------------------------------------------
7. DOCUMENT CONTROL
--------------------------------------------------------------------------------

Policy Hash: {self._policy_hash}
Generated: {datetime.utcnow().isoformat()}Z

This document is controlled and maintained in accordance with MiFID II
Article 27 requirements for best execution policy documentation.

================================================================================
"""
        return doc

    @property
    def policy_hash(self) -> str:
        """Get current policy hash."""
        return self._policy_hash or ""

    @property
    def version(self) -> str:
        """Get policy version."""
        return self.config.policy_version


# =============================================================================
# Best Execution Analyzer
# =============================================================================


class BestExecutionAnalyzer:
    """
    Analyzes execution quality against best execution criteria.

    Provides:
    - Single execution analysis
    - Aggregate period analysis
    - Venue performance comparison
    - Factor-weighted scoring
    """

    # Scoring parameters (configurable)
    PRICE_SCORE_SCALE_BPS = 50.0  # ±50bps maps to ±1.0 score
    COST_SCORE_SCALE_BPS = 100.0  # 100bps cost = 0 score
    SPEED_SCORE_SCALE_MS = 1000.0  # 1000ms latency = 0 score

    def __init__(
        self,
        policy: BestExecutionPolicy,
        audit_callback: Optional[Callable[[ExecutionAnalysis], None]] = None,
    ):
        """
        Initialize analyzer.

        Args:
            policy: Best execution policy for factor weights
            audit_callback: Optional callback for audit trail integration
        """
        self.policy = policy
        self.audit_callback = audit_callback
        self._analyses: List[ExecutionAnalysis] = []
        self._lock = threading.Lock()

    def analyze_execution(
        self,
        order: Dict[str, Any],
        fill: Dict[str, Any],
        market_data: Dict[str, Any],
        asset_class: Optional[AssetClass] = None,
    ) -> ExecutionAnalysis:
        """
        Analyze a single execution against best execution criteria.

        Args:
            order: Order data with keys:
                - order_id: str
                - side: str ("BUY" or "SELL")
                - quantity: Decimal
                - submit_time_ms: int
                - category: str (optional)
                - instrument_isin: str (optional)
            fill: Fill data with keys:
                - price: Decimal
                - quantity: Decimal
                - commission: Decimal (optional)
                - fees: Decimal (optional)
                - fill_time_ms: int
                - venue_mic: str (optional)
            market_data: Market data with keys:
                - bid: Decimal
                - ask: Decimal
                - mid: Decimal (optional)
                - spread_bps: Decimal (optional)

        Returns:
            ExecutionAnalysis with detailed breakdown.
        """
        analysis = ExecutionAnalysis(
            order_id=order.get("order_id", ""),
            instrument_isin=order.get("instrument_isin", ""),
            venue_mic=fill.get("venue_mic", ""),
            side=order.get("side", ""),
        )

        # Get factor weights
        order_category = None
        if "category" in order:
            try:
                order_category = OrderCategory(order["category"])
                analysis.order_category = order_category
            except ValueError:
                pass

        weights = self.policy.get_factor_weights(asset_class, order_category)

        # Calculate reference price
        mid_price = market_data.get("mid")
        if mid_price is None:
            bid = Decimal(str(market_data.get("bid", 0)))
            ask = Decimal(str(market_data.get("ask", 0)))
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal("0")
        else:
            mid_price = Decimal(str(mid_price))

        analysis.reference_price = mid_price
        analysis.execution_price = Decimal(str(fill.get("price", 0)))
        analysis.order_size = Decimal(str(order.get("quantity", 0)))
        analysis.filled_size = Decimal(str(fill.get("quantity", 0)))

        # === Price Analysis ===
        if mid_price > 0:
            side = order.get("side", "BUY").upper()
            if side == "BUY":
                # For buys, negative price improvement = paying more than mid
                price_diff = mid_price - analysis.execution_price
            else:
                # For sells, negative price improvement = receiving less than mid
                price_diff = analysis.execution_price - mid_price

            analysis.price_improvement_bps = (price_diff / mid_price) * Decimal("10000")

            # Score: +1.0 for good execution, -1.0 for bad
            price_score = float(analysis.price_improvement_bps) / self.PRICE_SCORE_SCALE_BPS
            analysis.price_score = max(-1.0, min(1.0, price_score))

        # === Cost Analysis ===
        commission = Decimal(str(fill.get("commission", 0)))
        fees = Decimal(str(fill.get("fees", 0)))
        notional = analysis.execution_price * analysis.filled_size

        analysis.explicit_cost_bps = Decimal("0")
        if notional > 0:
            analysis.explicit_cost_bps = ((commission + fees) / notional) * Decimal("10000")

        # Implicit cost = slippage
        spread_bps = market_data.get("spread_bps", Decimal("5"))
        analysis.implicit_cost_bps = Decimal(str(spread_bps)) / 2  # Half spread

        analysis.total_cost_bps = analysis.explicit_cost_bps + analysis.implicit_cost_bps

        # Cost score: lower is better
        cost_score = 1.0 - (float(analysis.total_cost_bps) / self.COST_SCORE_SCALE_BPS)
        analysis.cost_score = max(0.0, min(1.0, cost_score))

        # === Speed Analysis ===
        submit_time = order.get("submit_time_ms", 0)
        fill_time = fill.get("fill_time_ms", 0)
        analysis.order_submit_time_ms = submit_time
        analysis.fill_time_ms = fill_time
        analysis.latency_ms = float(fill_time - submit_time) if fill_time > submit_time else 0.0

        # Speed score: lower latency is better
        speed_score = 1.0 - (analysis.latency_ms / self.SPEED_SCORE_SCALE_MS)
        analysis.speed_score = max(0.0, min(1.0, speed_score))

        # === Likelihood Analysis ===
        if analysis.order_size > 0:
            analysis.fill_rate = float(analysis.filled_size / analysis.order_size)
        else:
            analysis.fill_rate = 1.0

        analysis.likelihood_score = analysis.fill_rate

        # === Settlement Analysis ===
        analysis.settlement_certainty = fill.get("settlement_certainty", 1.0)
        analysis.settlement_score = analysis.settlement_certainty

        # === Size Analysis ===
        # Market impact estimation (simplified)
        adv = Decimal(str(market_data.get("adv", 0)))
        if adv > 0:
            participation = analysis.filled_size / adv
            # Simple square root impact model
            impact_bps = float(participation) ** 0.5 * 100  # 100bps at 100% ADV
            analysis.market_impact_bps = Decimal(str(min(impact_bps, 500)))

        # Size score: lower impact is better
        size_score = 1.0 - (float(analysis.market_impact_bps) / 100.0)
        analysis.size_score = max(0.0, min(1.0, size_score))

        # === Nature Analysis ===
        # Default full score for nature (depends on order handling)
        analysis.nature_score = 1.0

        # === Weighted Score ===
        analysis.weighted_score = (
            analysis.price_score * weights.price
            + analysis.cost_score * weights.cost
            + analysis.speed_score * weights.speed
            + analysis.likelihood_score * weights.likelihood
            + analysis.settlement_score * weights.settlement
            + analysis.size_score * weights.size
            + analysis.nature_score * weights.nature
        )

        # === Quality Level ===
        if analysis.weighted_score >= 0.8:
            analysis.quality_level = ExecutionQualityLevel.EXCELLENT
        elif analysis.weighted_score >= 0.6:
            analysis.quality_level = ExecutionQualityLevel.GOOD
        elif analysis.weighted_score >= 0.4:
            analysis.quality_level = ExecutionQualityLevel.ACCEPTABLE
        elif analysis.weighted_score >= 0.2:
            analysis.quality_level = ExecutionQualityLevel.BELOW_STANDARD
        else:
            analysis.quality_level = ExecutionQualityLevel.POOR

        # Add notes
        if analysis.price_improvement_bps < self.policy.config.price_improvement_threshold_bps:
            analysis.notes.append(
                f"Price improvement below threshold: {analysis.price_improvement_bps}bps"
            )
        if analysis.total_cost_bps > self.policy.config.cost_threshold_bps:
            analysis.notes.append(f"Cost above threshold: {analysis.total_cost_bps}bps")
        if analysis.latency_ms > self.policy.config.latency_threshold_ms:
            analysis.notes.append(f"Latency above threshold: {analysis.latency_ms}ms")
        if analysis.fill_rate < self.policy.config.fill_rate_threshold_pct / 100:
            analysis.notes.append(f"Fill rate below threshold: {analysis.fill_rate * 100:.1f}%")

        # Store and callback
        with self._lock:
            self._analyses.append(analysis)

        if self.audit_callback:
            self.audit_callback(analysis)

        return analysis

    def get_aggregate_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        venue_mic: Optional[str] = None,
        instrument_isin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregate execution metrics for a period.

        Args:
            start_time: Start of period (inclusive)
            end_time: End of period (inclusive)
            venue_mic: Filter by venue
            instrument_isin: Filter by instrument

        Returns:
            Dictionary with aggregate metrics.
        """
        with self._lock:
            analyses = self._analyses.copy()

        # Filter
        if start_time:
            start_ns = int(start_time.timestamp() * 1e9)
            analyses = [a for a in analyses if a.analysis_timestamp_ns >= start_ns]
        if end_time:
            end_ns = int(end_time.timestamp() * 1e9)
            analyses = [a for a in analyses if a.analysis_timestamp_ns <= end_ns]
        if venue_mic:
            analyses = [a for a in analyses if a.venue_mic == venue_mic]
        if instrument_isin:
            analyses = [a for a in analyses if a.instrument_isin == instrument_isin]

        if not analyses:
            return {
                "count": 0,
                "avg_weighted_score": 0.0,
                "avg_price_improvement_bps": 0.0,
                "avg_total_cost_bps": 0.0,
                "avg_latency_ms": 0.0,
                "avg_fill_rate": 0.0,
                "quality_distribution": {},
            }

        # Calculate aggregates
        price_improvements = [float(a.price_improvement_bps) for a in analyses]
        costs = [float(a.total_cost_bps) for a in analyses]
        latencies = [a.latency_ms for a in analyses]
        fill_rates = [a.fill_rate for a in analyses]
        scores = [a.weighted_score for a in analyses]

        # Quality distribution
        quality_dist = {}
        for level in ExecutionQualityLevel:
            quality_dist[level.value] = sum(1 for a in analyses if a.quality_level == level)

        return {
            "count": len(analyses),
            "avg_weighted_score": statistics.mean(scores),
            "median_weighted_score": statistics.median(scores),
            "std_weighted_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "avg_price_improvement_bps": statistics.mean(price_improvements),
            "median_price_improvement_bps": statistics.median(price_improvements),
            "avg_total_cost_bps": statistics.mean(costs),
            "median_total_cost_bps": statistics.median(costs),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_fill_rate": statistics.mean(fill_rates),
            "min_fill_rate": min(fill_rates),
            "quality_distribution": quality_dist,
            "excellent_pct": quality_dist.get("excellent", 0) / len(analyses) * 100,
            "poor_pct": quality_dist.get("poor", 0) / len(analyses) * 100,
        }

    def compare_venues(
        self,
        venue_mics: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare execution quality across venues.

        Args:
            venue_mics: List of venue MICs to compare
            start_time: Start of period
            end_time: End of period

        Returns:
            Dictionary of metrics per venue.
        """
        return {
            mic: self.get_aggregate_metrics(start_time, end_time, venue_mic=mic)
            for mic in venue_mics
        }

    def get_analyses(self) -> List[ExecutionAnalysis]:
        """Get all stored analyses."""
        with self._lock:
            return self._analyses.copy()

    def clear_analyses(self) -> int:
        """Clear stored analyses. Returns count cleared."""
        with self._lock:
            count = len(self._analyses)
            self._analyses.clear()
            return count


# =============================================================================
# Factory Functions
# =============================================================================


def create_best_execution_policy(
    firm_lei: str = "",
    firm_name: str = "",
    approved_by: str = "Compliance Officer",
    factor_weights: Optional[FactorWeights] = None,
) -> BestExecutionPolicy:
    """
    Factory function to create a best execution policy.

    Args:
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        approved_by: Person approving the policy
        factor_weights: Custom factor weights (default: proprietary trading)

    Returns:
        Configured BestExecutionPolicy instance.
    """
    if factor_weights is None:
        factor_weights = FactorWeights.for_proprietary_trading()

    config = BestExecutionPolicyConfig(
        firm_lei=firm_lei,
        firm_name=firm_name,
        approved_by=approved_by,
        effective_date=date.today(),
        review_date=date.today() + timedelta(days=365),
        default_factor_weights=factor_weights,
    )

    return BestExecutionPolicy(config)


def create_best_execution_analyzer(
    policy: BestExecutionPolicy,
    audit_callback: Optional[Callable[[ExecutionAnalysis], None]] = None,
) -> BestExecutionAnalyzer:
    """
    Factory function to create a best execution analyzer.

    Args:
        policy: Best execution policy
        audit_callback: Optional callback for audit trail

    Returns:
        Configured BestExecutionAnalyzer instance.
    """
    return BestExecutionAnalyzer(policy, audit_callback)


# =============================================================================
# Standard Venue Data
# =============================================================================


def get_standard_eu_venues() -> List[ExecutionVenue]:
    """Get standard EU trading venues."""
    return [
        ExecutionVenue(
            mic="XLON",
            name="London Stock Exchange",
            venue_type=VenueType.REGULATED_MARKET,
            country="GB",
            currency="GBP",
            ranking=1,
            avg_spread_bps=Decimal("3.5"),
            avg_latency_ms=5.0,
            fill_rate_pct=98.5,
            cost_bps=Decimal("2.0"),
        ),
        ExecutionVenue(
            mic="XETR",
            name="Deutsche Börse Xetra",
            venue_type=VenueType.REGULATED_MARKET,
            country="DE",
            currency="EUR",
            ranking=2,
            avg_spread_bps=Decimal("4.0"),
            avg_latency_ms=6.0,
            fill_rate_pct=98.0,
            cost_bps=Decimal("2.5"),
        ),
        ExecutionVenue(
            mic="XPAR",
            name="Euronext Paris",
            venue_type=VenueType.REGULATED_MARKET,
            country="FR",
            currency="EUR",
            ranking=3,
            avg_spread_bps=Decimal("4.5"),
            avg_latency_ms=7.0,
            fill_rate_pct=97.5,
            cost_bps=Decimal("3.0"),
        ),
        ExecutionVenue(
            mic="XAMS",
            name="Euronext Amsterdam",
            venue_type=VenueType.REGULATED_MARKET,
            country="NL",
            currency="EUR",
            ranking=4,
            avg_spread_bps=Decimal("5.0"),
            avg_latency_ms=8.0,
            fill_rate_pct=97.0,
            cost_bps=Decimal("3.5"),
        ),
        ExecutionVenue(
            mic="XMIL",
            name="Borsa Italiana",
            venue_type=VenueType.REGULATED_MARKET,
            country="IT",
            currency="EUR",
            ranking=5,
            avg_spread_bps=Decimal("5.5"),
            avg_latency_ms=9.0,
            fill_rate_pct=96.5,
            cost_bps=Decimal("4.0"),
        ),
        ExecutionVenue(
            mic="XOFF",
            name="Off-Exchange",
            venue_type=VenueType.OTC,
            country="XX",
            currency="EUR",
            ranking=99,
            avg_spread_bps=Decimal("10.0"),
            avg_latency_ms=50.0,
            fill_rate_pct=90.0,
            cost_bps=Decimal("5.0"),
        ),
    ]


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ExecutionFactor",
    "AssetClass",
    "OrderCategory",
    "VenueType",
    "ExecutionQualityLevel",
    # Data classes
    "ExecutionVenue",
    "FactorWeights",
    "ExecutionAnalysis",
    "BestExecutionPolicyConfig",
    # Main classes
    "BestExecutionPolicy",
    "BestExecutionAnalyzer",
    # Factory functions
    "create_best_execution_policy",
    "create_best_execution_analyzer",
    # Utilities
    "get_standard_eu_venues",
]
