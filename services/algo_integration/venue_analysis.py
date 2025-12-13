# -*- coding: utf-8 -*-
"""
MiFID II Venue Analysis and Smart Order Routing.

This module provides venue-level execution analysis and smart order routing
capabilities for best execution compliance per MiFID II Article 27.

Key Features:
    - Venue performance metrics tracking
    - Venue quality scoring
    - Venue comparison analysis
    - Smart order routing recommendations
    - Venue ranking updates based on performance

The module integrates with the best_execution policy to maintain
accurate venue rankings based on actual execution quality.

References:
    - MiFID II Article 27: Best Execution
    - RTS 27 (suspended): Execution venue quality data
    - ESMA Guidelines on venue selection
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
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
    Set,
)

from services.algo_integration.best_execution import (
    ExecutionVenue,
    VenueType,
    AssetClass,
    ExecutionQualityLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class VenueMetricType(Enum):
    """Types of venue performance metrics."""

    SPREAD = "spread"
    LATENCY = "latency"
    FILL_RATE = "fill_rate"
    PRICE_IMPROVEMENT = "price_improvement"
    REJECTION_RATE = "rejection_rate"
    MARKET_IMPACT = "market_impact"
    COST = "cost"
    LIQUIDITY = "liquidity"


class VenueSelectionReason(Enum):
    """Reasons for venue selection in SOR."""

    BEST_PRICE = "best_price"
    BEST_LIQUIDITY = "best_liquidity"
    LOWEST_COST = "lowest_cost"
    FASTEST = "fastest"
    HIGHEST_FILL_RATE = "highest_fill_rate"
    BEST_OVERALL_SCORE = "best_overall_score"
    EXPLICIT_INSTRUCTION = "explicit_instruction"
    DARK_LIQUIDITY = "dark_liquidity"
    REFERENCE_PRICE_WAIVER = "reference_price_waiver"


class VenueStatus(Enum):
    """Venue operational status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"  # Issues detected
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"  # Manually disabled


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VenueExecutionRecord:
    """
    Record of a single execution at a venue.

    Used to track venue performance over time.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=time.time_ns)

    venue_mic: str = ""
    instrument_isin: str = ""
    side: str = ""

    # Order details
    order_id: str = ""
    order_quantity: Decimal = Decimal("0")
    order_price: Optional[Decimal] = None

    # Fill details
    fill_quantity: Decimal = Decimal("0")
    fill_price: Decimal = Decimal("0")
    fill_rate: float = 0.0

    # Reference prices
    arrival_price: Decimal = Decimal("0")
    nbbo_bid: Optional[Decimal] = None
    nbbo_ask: Optional[Decimal] = None

    # Metrics
    spread_bps: Decimal = Decimal("0")
    latency_ms: float = 0.0
    price_improvement_bps: Decimal = Decimal("0")
    cost_bps: Decimal = Decimal("0")
    market_impact_bps: Decimal = Decimal("0")

    # Status
    was_rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "timestamp_ns": self.timestamp_ns,
            "venue_mic": self.venue_mic,
            "instrument_isin": self.instrument_isin,
            "side": self.side,
            "order_id": self.order_id,
            "order_quantity": str(self.order_quantity),
            "fill_quantity": str(self.fill_quantity),
            "fill_price": str(self.fill_price),
            "fill_rate": self.fill_rate,
            "arrival_price": str(self.arrival_price),
            "spread_bps": str(self.spread_bps),
            "latency_ms": self.latency_ms,
            "price_improvement_bps": str(self.price_improvement_bps),
            "cost_bps": str(self.cost_bps),
            "market_impact_bps": str(self.market_impact_bps),
            "was_rejected": self.was_rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class VenuePerformanceMetrics:
    """
    Aggregate performance metrics for a venue over a period.
    """

    venue_mic: str = ""
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Volume statistics
    total_orders: int = 0
    total_fills: int = 0
    total_rejections: int = 0
    total_quantity_filled: Decimal = Decimal("0")
    total_notional: Decimal = Decimal("0")

    # Spread metrics
    avg_spread_bps: Decimal = Decimal("0")
    median_spread_bps: Decimal = Decimal("0")
    min_spread_bps: Decimal = Decimal("0")
    max_spread_bps: Decimal = Decimal("0")

    # Latency metrics
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Fill rate metrics
    avg_fill_rate: float = 0.0
    rejection_rate: float = 0.0

    # Price improvement metrics
    avg_price_improvement_bps: Decimal = Decimal("0")
    median_price_improvement_bps: Decimal = Decimal("0")
    price_improvement_rate: float = 0.0  # % of orders with positive improvement

    # Cost metrics
    avg_cost_bps: Decimal = Decimal("0")
    avg_market_impact_bps: Decimal = Decimal("0")

    # Liquidity metrics
    avg_available_size: Decimal = Decimal("0")
    liquidity_score: float = 0.0

    # Overall quality score (0-100)
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "venue_mic": self.venue_mic,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_rejections": self.total_rejections,
            "total_quantity_filled": str(self.total_quantity_filled),
            "total_notional": str(self.total_notional),
            "avg_spread_bps": str(self.avg_spread_bps),
            "median_spread_bps": str(self.median_spread_bps),
            "avg_latency_ms": self.avg_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "avg_fill_rate": self.avg_fill_rate,
            "rejection_rate": self.rejection_rate,
            "avg_price_improvement_bps": str(self.avg_price_improvement_bps),
            "price_improvement_rate": self.price_improvement_rate,
            "avg_cost_bps": str(self.avg_cost_bps),
            "quality_score": self.quality_score,
        }


@dataclass
class VenueRoutingDecision:
    """
    Smart order routing decision.

    Documents the venue selection for compliance purposes.
    """

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=time.time_ns)

    order_id: str = ""
    instrument_isin: str = ""
    side: str = ""
    quantity: Decimal = Decimal("0")

    # Selected venue
    selected_venue: str = ""
    selection_reason: VenueSelectionReason = VenueSelectionReason.BEST_OVERALL_SCORE

    # Alternative venues considered
    venues_considered: List[str] = field(default_factory=list)
    venue_scores: Dict[str, float] = field(default_factory=dict)

    # Market data at decision time
    venue_quotes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Rationale
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit trail."""
        return {
            "decision_id": self.decision_id,
            "timestamp_ns": self.timestamp_ns,
            "order_id": self.order_id,
            "instrument_isin": self.instrument_isin,
            "side": self.side,
            "quantity": str(self.quantity),
            "selected_venue": self.selected_venue,
            "selection_reason": self.selection_reason.value,
            "venues_considered": self.venues_considered,
            "venue_scores": self.venue_scores,
            "venue_quotes": self.venue_quotes,
            "rationale": self.rationale,
        }


@dataclass
class VenueAnalysisConfig:
    """
    Configuration for venue analysis.

    Attributes:
        min_samples: Minimum executions required for reliable metrics
        rolling_window_days: Days of history to consider for metrics
        metric_weights: Weights for scoring venues (must sum to 1.0)
        auto_update_rankings: Automatically update venue rankings
        dark_pool_enabled: Include dark pools in routing
        spread_threshold_bps: Alert if spread exceeds threshold
        latency_threshold_ms: Alert if latency exceeds threshold
        rejection_rate_threshold_pct: Alert if rejection rate exceeds threshold
    """

    min_samples: int = 30
    rolling_window_days: int = 30

    # Weights for venue quality scoring
    metric_weights: Dict[VenueMetricType, float] = field(default_factory=lambda: {
        VenueMetricType.SPREAD: 0.20,
        VenueMetricType.LATENCY: 0.15,
        VenueMetricType.FILL_RATE: 0.20,
        VenueMetricType.PRICE_IMPROVEMENT: 0.25,
        VenueMetricType.REJECTION_RATE: 0.10,
        VenueMetricType.COST: 0.10,
    })

    auto_update_rankings: bool = True
    dark_pool_enabled: bool = False

    # Alert thresholds
    spread_threshold_bps: Decimal = Decimal("20")
    latency_threshold_ms: float = 500.0
    rejection_rate_threshold_pct: float = 10.0

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []

        total_weight = sum(self.metric_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            errors.append(f"Metric weights sum to {total_weight}, must be 1.0")

        if self.min_samples < 1:
            errors.append("min_samples must be at least 1")

        if self.rolling_window_days < 1:
            errors.append("rolling_window_days must be at least 1")

        return errors


# =============================================================================
# Venue Analyzer
# =============================================================================


class VenueAnalyzer:
    """
    Analyzes venue execution quality and maintains performance metrics.

    Per MiFID II, firms must monitor execution quality across venues
    and update venue rankings based on actual performance.
    """

    def __init__(
        self,
        config: VenueAnalysisConfig,
        venues: Optional[List[ExecutionVenue]] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize venue analyzer.

        Args:
            config: Analysis configuration
            venues: Initial list of venues
            audit_callback: Callback for audit trail
        """
        self.config = config
        self.audit_callback = audit_callback

        self._venues: Dict[str, ExecutionVenue] = {}
        self._venue_status: Dict[str, VenueStatus] = {}
        self._records: List[VenueExecutionRecord] = []
        self._lock = threading.Lock()

        # Initialize venues
        if venues:
            for venue in venues:
                self._venues[venue.mic] = venue
                self._venue_status[venue.mic] = (
                    VenueStatus.ACTIVE if venue.is_active else VenueStatus.INACTIVE
                )

    def add_venue(self, venue: ExecutionVenue) -> None:
        """Add a venue to tracking."""
        with self._lock:
            self._venues[venue.mic] = venue
            self._venue_status[venue.mic] = (
                VenueStatus.ACTIVE if venue.is_active else VenueStatus.INACTIVE
            )

    def remove_venue(self, mic: str) -> bool:
        """Remove a venue from tracking."""
        with self._lock:
            if mic in self._venues:
                del self._venues[mic]
                del self._venue_status[mic]
                return True
            return False

    def set_venue_status(self, mic: str, status: VenueStatus) -> bool:
        """Set venue operational status."""
        with self._lock:
            if mic in self._venue_status:
                self._venue_status[mic] = status
                logger.info(f"Venue {mic} status set to {status.value}")
                return True
            return False

    def record_execution(self, record: VenueExecutionRecord) -> None:
        """
        Record an execution for venue analysis.

        Args:
            record: Execution record to store
        """
        with self._lock:
            self._records.append(record)

        # Callback for audit
        if self.audit_callback:
            self.audit_callback({
                "type": "venue_execution_record",
                "record": record.to_dict(),
            })

    def record_from_fill(
        self,
        order: Dict[str, Any],
        fill: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> VenueExecutionRecord:
        """
        Create and record execution from order/fill data.

        Args:
            order: Order data
            fill: Fill data
            market_data: Market data at execution time

        Returns:
            Created VenueExecutionRecord
        """
        record = VenueExecutionRecord(
            venue_mic=fill.get("venue_mic", ""),
            instrument_isin=order.get("instrument_isin", ""),
            side=order.get("side", ""),
            order_id=order.get("order_id", ""),
            order_quantity=Decimal(str(order.get("quantity", 0))),
            fill_quantity=Decimal(str(fill.get("quantity", 0))),
            fill_price=Decimal(str(fill.get("price", 0))),
        )

        # Fill rate
        if record.order_quantity > 0:
            record.fill_rate = float(record.fill_quantity / record.order_quantity)

        # Reference prices
        record.arrival_price = Decimal(str(market_data.get("arrival_price", 0)))
        if "bid" in market_data and "ask" in market_data:
            record.nbbo_bid = Decimal(str(market_data["bid"]))
            record.nbbo_ask = Decimal(str(market_data["ask"]))

        # Spread
        if record.nbbo_bid and record.nbbo_ask:
            mid = (record.nbbo_bid + record.nbbo_ask) / 2
            if mid > 0:
                spread = record.nbbo_ask - record.nbbo_bid
                record.spread_bps = (spread / mid) * Decimal("10000")

        # Latency
        submit_time = order.get("submit_time_ms", 0)
        fill_time = fill.get("timestamp_ms", 0)
        if fill_time > submit_time:
            record.latency_ms = float(fill_time - submit_time)

        # Price improvement
        if record.arrival_price > 0 and record.fill_price > 0:
            side = record.side.upper()
            if side == "BUY":
                improvement = record.arrival_price - record.fill_price
            else:
                improvement = record.fill_price - record.arrival_price
            record.price_improvement_bps = (improvement / record.arrival_price) * Decimal("10000")

        # Cost
        commission = Decimal(str(fill.get("commission", 0)))
        fees = Decimal(str(fill.get("fees", 0)))
        notional = record.fill_price * record.fill_quantity
        if notional > 0:
            record.cost_bps = ((commission + fees) / notional) * Decimal("10000")

        # Record
        self.record_execution(record)
        return record

    def get_venue_metrics(
        self,
        venue_mic: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> VenuePerformanceMetrics:
        """
        Calculate performance metrics for a venue.

        Args:
            venue_mic: Venue MIC code
            start_time: Period start (default: rolling window)
            end_time: Period end (default: now)

        Returns:
            VenuePerformanceMetrics for the period
        """
        # Use time.time() for consistency with time.time_ns() used in records
        # Note: datetime.utcnow().timestamp() has timezone issues on non-UTC systems
        # We calculate an offset to correct for this
        current_time_ns = time.time_ns()
        current_time_sec = current_time_ns / 1e9

        def datetime_to_ns(dt: datetime) -> int:
            """Convert datetime to nanoseconds, correcting for UTC naive datetime."""
            # Assume dt is meant to be UTC (from datetime.utcnow())
            # Calculate offset between timestamp interpretation and actual UTC
            ref_utc = datetime.utcnow()
            offset_sec = ref_utc.timestamp() - current_time_sec
            return int((dt.timestamp() - offset_sec) * 1e9)

        if end_time is None:
            end_ns = current_time_ns
            end_time = datetime.utcfromtimestamp(current_time_sec)
        else:
            end_ns = datetime_to_ns(end_time)

        if start_time is None:
            start_time = end_time - timedelta(days=self.config.rolling_window_days)
            start_ns = end_ns - int(self.config.rolling_window_days * 86400 * 1e9)
        else:
            start_ns = datetime_to_ns(start_time)

        with self._lock:
            records = [
                r for r in self._records
                if r.venue_mic == venue_mic
                and start_ns <= r.timestamp_ns <= end_ns
            ]

        metrics = VenuePerformanceMetrics(
            venue_mic=venue_mic,
            period_start=start_time,
            period_end=end_time,
        )

        if not records:
            return metrics

        # Volume statistics
        metrics.total_orders = len(records)
        metrics.total_fills = sum(1 for r in records if r.fill_quantity > 0)
        metrics.total_rejections = sum(1 for r in records if r.was_rejected)
        metrics.total_quantity_filled = sum(r.fill_quantity for r in records)
        metrics.total_notional = sum(r.fill_quantity * r.fill_price for r in records)

        # Spread metrics
        spreads = [float(r.spread_bps) for r in records if r.spread_bps > 0]
        if spreads:
            metrics.avg_spread_bps = Decimal(str(statistics.mean(spreads)))
            metrics.median_spread_bps = Decimal(str(statistics.median(spreads)))
            metrics.min_spread_bps = Decimal(str(min(spreads)))
            metrics.max_spread_bps = Decimal(str(max(spreads)))

        # Latency metrics
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]
        if latencies:
            metrics.avg_latency_ms = statistics.mean(latencies)
            metrics.median_latency_ms = statistics.median(latencies)
            sorted_lat = sorted(latencies)
            metrics.p95_latency_ms = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0
            metrics.p99_latency_ms = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 0 else 0
            metrics.min_latency_ms = min(latencies)
            metrics.max_latency_ms = max(latencies)

        # Fill rate
        fill_rates = [r.fill_rate for r in records]
        if fill_rates:
            metrics.avg_fill_rate = statistics.mean(fill_rates)

        # Rejection rate
        if metrics.total_orders > 0:
            metrics.rejection_rate = metrics.total_rejections / metrics.total_orders * 100

        # Price improvement
        price_improvements = [float(r.price_improvement_bps) for r in records]
        if price_improvements:
            metrics.avg_price_improvement_bps = Decimal(str(statistics.mean(price_improvements)))
            metrics.median_price_improvement_bps = Decimal(str(statistics.median(price_improvements)))
            metrics.price_improvement_rate = sum(1 for p in price_improvements if p > 0) / len(price_improvements) * 100

        # Cost
        costs = [float(r.cost_bps) for r in records if r.cost_bps > 0]
        if costs:
            metrics.avg_cost_bps = Decimal(str(statistics.mean(costs)))

        # Market impact
        impacts = [float(r.market_impact_bps) for r in records if r.market_impact_bps > 0]
        if impacts:
            metrics.avg_market_impact_bps = Decimal(str(statistics.mean(impacts)))

        # Quality score
        metrics.quality_score = self._calculate_quality_score(metrics)

        return metrics

    def _calculate_quality_score(self, metrics: VenuePerformanceMetrics) -> float:
        """
        Calculate overall quality score (0-100) for a venue.

        Higher is better.
        """
        scores = {}

        # Spread score (lower is better)
        # Normalize: 0bps = 100, 20bps = 0
        spread_score = max(0, 100 - float(metrics.avg_spread_bps) * 5)
        scores[VenueMetricType.SPREAD] = spread_score

        # Latency score (lower is better)
        # Normalize: 0ms = 100, 500ms = 0
        latency_score = max(0, 100 - metrics.avg_latency_ms / 5)
        scores[VenueMetricType.LATENCY] = latency_score

        # Fill rate score (higher is better)
        # Already 0-1, multiply by 100
        fill_rate_score = metrics.avg_fill_rate * 100
        scores[VenueMetricType.FILL_RATE] = fill_rate_score

        # Price improvement score
        # Normalize: -50bps = 0, +50bps = 100
        pi_bps = float(metrics.avg_price_improvement_bps)
        pi_score = max(0, min(100, 50 + pi_bps))
        scores[VenueMetricType.PRICE_IMPROVEMENT] = pi_score

        # Rejection rate score (lower is better)
        # 0% = 100, 100% = 0
        rejection_score = max(0, 100 - metrics.rejection_rate)
        scores[VenueMetricType.REJECTION_RATE] = rejection_score

        # Cost score (lower is better)
        # Normalize: 0bps = 100, 100bps = 0
        cost_score = max(0, 100 - float(metrics.avg_cost_bps))
        scores[VenueMetricType.COST] = cost_score

        # Weighted average
        total_score = 0.0
        for metric_type, weight in self.config.metric_weights.items():
            if metric_type in scores:
                total_score += scores[metric_type] * weight

        return total_score

    def compare_venues(
        self,
        venue_mics: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, VenuePerformanceMetrics]:
        """
        Compare performance across venues.

        Args:
            venue_mics: List of venues to compare (default: all)
            start_time: Period start
            end_time: Period end

        Returns:
            Dictionary of metrics per venue
        """
        if venue_mics is None:
            with self._lock:
                venue_mics = list(self._venues.keys())

        return {
            mic: self.get_venue_metrics(mic, start_time, end_time)
            for mic in venue_mics
        }

    def get_venue_rankings(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get venues ranked by quality score.

        Returns:
            List of (venue_mic, score) tuples, highest first
        """
        metrics = self.compare_venues(start_time=start_time, end_time=end_time)

        rankings = [
            (mic, m.quality_score)
            for mic, m in metrics.items()
            if m.total_orders >= self.config.min_samples
        ]

        return sorted(rankings, key=lambda x: x[1], reverse=True)


# =============================================================================
# Smart Order Router
# =============================================================================


class SmartOrderRouter:
    """
    Smart Order Router (SOR) for best execution compliance.

    Selects optimal venue based on:
    - Current market data (prices, liquidity)
    - Historical venue performance
    - Order characteristics
    - Execution policy requirements
    """

    def __init__(
        self,
        venue_analyzer: VenueAnalyzer,
        config: Optional[VenueAnalysisConfig] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize SOR.

        Args:
            venue_analyzer: Venue analyzer for performance data
            config: Configuration (uses analyzer's config if None)
            audit_callback: Callback for audit trail
        """
        self.analyzer = venue_analyzer
        self.config = config or venue_analyzer.config
        self.audit_callback = audit_callback

        self._decisions: List[VenueRoutingDecision] = []
        self._lock = threading.Lock()

    def select_venue(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Dict[str, Any]],
        asset_class: Optional[AssetClass] = None,
        explicit_venue: Optional[str] = None,
    ) -> VenueRoutingDecision:
        """
        Select optimal venue for an order.

        Args:
            order: Order details:
                - order_id: str
                - instrument_isin: str
                - side: str
                - quantity: Decimal
            market_data: Market data per venue:
                {
                    "XLON": {"bid": 100.0, "ask": 100.05, "size": 1000},
                    "XETR": {"bid": 100.01, "ask": 100.04, "size": 800},
                }
            asset_class: Asset class for venue filtering
            explicit_venue: If specified, use this venue (client instruction)

        Returns:
            VenueRoutingDecision documenting selection
        """
        decision = VenueRoutingDecision(
            order_id=order.get("order_id", ""),
            instrument_isin=order.get("instrument_isin", ""),
            side=order.get("side", ""),
            quantity=Decimal(str(order.get("quantity", 0))),
        )

        # Get available venues
        available_venues = self._get_available_venues(asset_class)

        # Filter to venues with market data
        venues_with_data = [v for v in available_venues if v in market_data]
        decision.venues_considered = venues_with_data
        decision.venue_quotes = market_data

        # Handle explicit instruction
        if explicit_venue:
            if explicit_venue in venues_with_data:
                decision.selected_venue = explicit_venue
                decision.selection_reason = VenueSelectionReason.EXPLICIT_INSTRUCTION
                decision.rationale = f"Client explicit instruction to route to {explicit_venue}"
            else:
                # Fall back to smart selection
                logger.warning(f"Explicit venue {explicit_venue} not available, using SOR")
                decision.selected_venue, decision.selection_reason, decision.venue_scores = (
                    self._smart_select(
                        order=order,
                        venues=venues_with_data,
                        market_data=market_data,
                    )
                )
        else:
            # Smart selection
            decision.selected_venue, decision.selection_reason, decision.venue_scores = (
                self._smart_select(
                    order=order,
                    venues=venues_with_data,
                    market_data=market_data,
                )
            )

        # Generate rationale
        if not decision.rationale:
            decision.rationale = self._generate_rationale(decision)

        # Store decision
        with self._lock:
            self._decisions.append(decision)

        # Audit callback
        if self.audit_callback:
            self.audit_callback({
                "type": "venue_routing_decision",
                "decision": decision.to_dict(),
            })

        return decision

    def _get_available_venues(
        self,
        asset_class: Optional[AssetClass] = None,
    ) -> List[str]:
        """Get list of available venues."""
        with self.analyzer._lock:
            venues = []
            for mic, status in self.analyzer._venue_status.items():
                if status == VenueStatus.ACTIVE:
                    venue = self.analyzer._venues.get(mic)
                    if venue:
                        # Filter by asset class if specified
                        if asset_class is None or venue.venue_type != VenueType.DARK_POOL or self.config.dark_pool_enabled:
                            venues.append(mic)
            return venues

    def _smart_select(
        self,
        order: Dict[str, Any],
        venues: List[str],
        market_data: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, VenueSelectionReason, Dict[str, float]]:
        """
        Select venue using smart logic.

        Returns:
            Tuple of (selected_venue, reason, scores)
        """
        if not venues:
            return "", VenueSelectionReason.BEST_OVERALL_SCORE, {}

        if len(venues) == 1:
            return venues[0], VenueSelectionReason.BEST_OVERALL_SCORE, {venues[0]: 100.0}

        side = order.get("side", "BUY").upper()
        scores: Dict[str, float] = {}

        for venue in venues:
            quote = market_data.get(venue, {})
            score = 0.0

            # Price score (most important for best execution)
            # For buys, lower ask is better; for sells, higher bid is better
            bid = float(quote.get("bid", 0))
            ask = float(quote.get("ask", 0))

            if side == "BUY" and ask > 0:
                # Normalize: assume price range of 0.1% around midpoint
                mid = (bid + ask) / 2 if bid > 0 else ask
                price_score = max(0, 100 - (ask - mid) / mid * 10000)  # 100bps deviation = 0 score
                score += price_score * 0.40
            elif side == "SELL" and bid > 0:
                mid = (bid + ask) / 2 if ask > 0 else bid
                price_score = max(0, 100 - (mid - bid) / mid * 10000)
                score += price_score * 0.40

            # Spread score
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_bps = (ask - bid) / mid * 10000
                spread_score = max(0, 100 - spread_bps * 5)  # 20bps = 0 score
                score += spread_score * 0.20

            # Size score
            size = float(quote.get("size", quote.get("bid_size", 0)))
            order_qty = float(order.get("quantity", 0))
            if order_qty > 0:
                fill_likelihood = min(1.0, size / order_qty)
                score += fill_likelihood * 100 * 0.20

            # Historical performance score
            hist_metrics = self.analyzer.get_venue_metrics(venue)
            if hist_metrics.total_orders >= self.config.min_samples:
                score += hist_metrics.quality_score * 0.20

            scores[venue] = score

        # Select best venue
        best_venue = max(scores.keys(), key=lambda v: scores[v])

        # Determine primary reason
        quote = market_data.get(best_venue, {})
        if side == "BUY":
            best_ask = min(float(market_data[v].get("ask", float("inf"))) for v in venues if market_data[v].get("ask"))
            if float(quote.get("ask", float("inf"))) == best_ask:
                reason = VenueSelectionReason.BEST_PRICE
            else:
                reason = VenueSelectionReason.BEST_OVERALL_SCORE
        else:
            best_bid = max(float(market_data[v].get("bid", 0)) for v in venues)
            if float(quote.get("bid", 0)) == best_bid:
                reason = VenueSelectionReason.BEST_PRICE
            else:
                reason = VenueSelectionReason.BEST_OVERALL_SCORE

        return best_venue, reason, scores

    def _generate_rationale(self, decision: VenueRoutingDecision) -> str:
        """Generate human-readable rationale."""
        reason = decision.selection_reason

        if reason == VenueSelectionReason.EXPLICIT_INSTRUCTION:
            return f"Order routed to {decision.selected_venue} per explicit client instruction."

        if reason == VenueSelectionReason.BEST_PRICE:
            quote = decision.venue_quotes.get(decision.selected_venue, {})
            if decision.side == "BUY":
                price = quote.get("ask", "N/A")
                return f"Routed to {decision.selected_venue} for best ask price ({price})."
            else:
                price = quote.get("bid", "N/A")
                return f"Routed to {decision.selected_venue} for best bid price ({price})."

        if reason == VenueSelectionReason.BEST_LIQUIDITY:
            return f"Routed to {decision.selected_venue} for best liquidity."

        # Default
        score = decision.venue_scores.get(decision.selected_venue, 0)
        return (
            f"Routed to {decision.selected_venue} based on overall score ({score:.1f}) "
            f"considering price, spread, liquidity, and historical performance."
        )

    def get_decisions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[VenueRoutingDecision]:
        """Get routing decisions for a period."""
        with self._lock:
            decisions = self._decisions.copy()

        # Convert datetime to nanoseconds with UTC correction
        current_time_ns = time.time_ns()
        current_time_sec = current_time_ns / 1e9

        def datetime_to_ns(dt: datetime) -> int:
            """Convert datetime to nanoseconds, correcting for UTC naive datetime."""
            ref_utc = datetime.utcnow()
            offset_sec = ref_utc.timestamp() - current_time_sec
            return int((dt.timestamp() - offset_sec) * 1e9)

        if start_time:
            start_ns = datetime_to_ns(start_time)
            decisions = [d for d in decisions if d.timestamp_ns >= start_ns]
        if end_time:
            end_ns = datetime_to_ns(end_time)
            decisions = [d for d in decisions if d.timestamp_ns <= end_ns]

        return decisions

    def get_routing_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get routing statistics for compliance reporting."""
        decisions = self.get_decisions(start_time, end_time)

        if not decisions:
            return {
                "total_decisions": 0,
                "venue_distribution": {},
                "reason_distribution": {},
            }

        # Venue distribution
        venue_counts: Dict[str, int] = {}
        reason_counts: Dict[str, int] = {}

        for d in decisions:
            venue_counts[d.selected_venue] = venue_counts.get(d.selected_venue, 0) + 1
            reason_counts[d.selection_reason.value] = reason_counts.get(d.selection_reason.value, 0) + 1

        total = len(decisions)

        return {
            "total_decisions": total,
            "venue_distribution": {v: c / total * 100 for v, c in venue_counts.items()},
            "reason_distribution": {r: c / total * 100 for r, c in reason_counts.items()},
            "venues_used": list(venue_counts.keys()),
            "most_common_venue": max(venue_counts.keys(), key=lambda v: venue_counts[v]),
            "explicit_instruction_pct": reason_counts.get("explicit_instruction", 0) / total * 100,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_venue_analyzer(
    venues: Optional[List[ExecutionVenue]] = None,
    config: Optional[VenueAnalysisConfig] = None,
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> VenueAnalyzer:
    """
    Factory function to create venue analyzer.

    Args:
        venues: Initial list of venues
        config: Analysis configuration
        audit_callback: Callback for audit trail

    Returns:
        Configured VenueAnalyzer instance.
    """
    if config is None:
        config = VenueAnalysisConfig()

    return VenueAnalyzer(
        config=config,
        venues=venues,
        audit_callback=audit_callback,
    )


def create_smart_order_router(
    venue_analyzer: VenueAnalyzer,
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SmartOrderRouter:
    """
    Factory function to create smart order router.

    Args:
        venue_analyzer: Venue analyzer for performance data
        audit_callback: Callback for audit trail

    Returns:
        Configured SmartOrderRouter instance.
    """
    return SmartOrderRouter(
        venue_analyzer=venue_analyzer,
        audit_callback=audit_callback,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "VenueMetricType",
    "VenueSelectionReason",
    "VenueStatus",
    # Data classes
    "VenueExecutionRecord",
    "VenuePerformanceMetrics",
    "VenueRoutingDecision",
    "VenueAnalysisConfig",
    # Main classes
    "VenueAnalyzer",
    "SmartOrderRouter",
    # Factory functions
    "create_venue_analyzer",
    "create_smart_order_router",
]
