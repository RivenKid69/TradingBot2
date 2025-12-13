# -*- coding: utf-8 -*-
"""
MiFID II Transaction Cost Analysis (TCA) Compliance Wrapper.

This module provides a compliance-focused wrapper for TCA functionality,
integrating transaction cost analysis with best execution requirements
per MiFID II Article 27.

Key Features:
    - Pre-trade cost estimation
    - Post-trade execution analysis
    - Slippage vs. estimate comparison
    - Implementation shortfall calculation
    - Audit trail integration
    - Best execution compliance reporting

The TCA wrapper bridges the existing execution_providers.py models
with MiFID II compliance requirements.

References:
    - MiFID II Article 27: Best Execution
    - ESMA Guidelines on TCA for Best Execution
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Callable,
    Protocol,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class TCAMetricType(Enum):
    """Types of TCA metrics."""

    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    VWAP_SLIPPAGE = "vwap_slippage"
    TWAP_SLIPPAGE = "twap_slippage"
    ARRIVAL_PRICE_SLIPPAGE = "arrival_price_slippage"
    INTERVAL_VWAP = "interval_vwap"
    REVERSION = "reversion"
    MARKET_IMPACT = "market_impact"
    TIMING_COST = "timing_cost"


class TCABenchmark(Enum):
    """Benchmark types for TCA."""

    ARRIVAL_PRICE = "arrival_price"  # Decision price
    VWAP = "vwap"  # Volume-weighted average price
    TWAP = "twap"  # Time-weighted average price
    OPEN = "open"  # Opening price
    CLOSE = "close"  # Closing price
    MIDPOINT = "midpoint"  # Mid-market price
    PREVIOUS_CLOSE = "previous_close"


class CostCategory(Enum):
    """Categories of execution costs."""

    EXPLICIT = "explicit"  # Commission, fees, taxes
    IMPLICIT = "implicit"  # Spread, market impact, timing
    TOTAL = "total"


class ExecutionStrategy(Enum):
    """Execution strategies."""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"  # Percentage of volume
    IS = "is"  # Implementation shortfall
    ARRIVAL = "arrival"  # Arrival price


# =============================================================================
# Protocol Definitions
# =============================================================================


@runtime_checkable
class SlippageProvider(Protocol):
    """Protocol for slippage estimation providers."""

    def estimate_slippage_bps(
        self,
        notional: float,
        adv: float,
        side: str,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None,
    ) -> float:
        """Estimate slippage in basis points."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PreTradeEstimate:
    """
    Pre-trade cost and impact estimation.

    Used to provide execution guidance before order submission
    per best execution requirements.
    """

    estimate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    timestamp_ns: int = field(default_factory=time.time_ns)

    # Order details
    instrument_isin: str = ""
    side: str = ""
    quantity: Decimal = Decimal("0")
    notional_value: Decimal = Decimal("0")
    currency: str = "EUR"

    # Market context
    arrival_price: Decimal = Decimal("0")
    bid_price: Decimal = Decimal("0")
    ask_price: Decimal = Decimal("0")
    spread_bps: Decimal = Decimal("0")
    volatility: Optional[float] = None
    adv: Optional[Decimal] = None  # Average daily volume

    # Estimates
    estimated_impact_bps: Decimal = Decimal("0")
    estimated_spread_cost_bps: Decimal = Decimal("0")
    estimated_timing_cost_bps: Decimal = Decimal("0")
    estimated_total_cost_bps: Decimal = Decimal("0")
    estimated_execution_price: Decimal = Decimal("0")

    # Confidence
    confidence_level: float = 0.95  # 95% confidence
    cost_range_low_bps: Decimal = Decimal("0")
    cost_range_high_bps: Decimal = Decimal("0")

    # Recommendations
    recommended_strategy: ExecutionStrategy = ExecutionStrategy.MARKET
    recommended_urgency: str = "normal"  # low, normal, high
    participation_rate_pct: Optional[float] = None
    expected_duration_seconds: Optional[float] = None

    # Metadata
    model_name: str = ""
    model_version: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimate_id": self.estimate_id,
            "order_id": self.order_id,
            "timestamp_ns": self.timestamp_ns,
            "instrument_isin": self.instrument_isin,
            "side": self.side,
            "quantity": str(self.quantity),
            "notional_value": str(self.notional_value),
            "currency": self.currency,
            "arrival_price": str(self.arrival_price),
            "spread_bps": str(self.spread_bps),
            "estimated_impact_bps": str(self.estimated_impact_bps),
            "estimated_spread_cost_bps": str(self.estimated_spread_cost_bps),
            "estimated_timing_cost_bps": str(self.estimated_timing_cost_bps),
            "estimated_total_cost_bps": str(self.estimated_total_cost_bps),
            "estimated_execution_price": str(self.estimated_execution_price),
            "confidence_level": self.confidence_level,
            "cost_range_low_bps": str(self.cost_range_low_bps),
            "cost_range_high_bps": str(self.cost_range_high_bps),
            "recommended_strategy": self.recommended_strategy.value,
            "recommended_urgency": self.recommended_urgency,
            "participation_rate_pct": self.participation_rate_pct,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "notes": self.notes,
        }


@dataclass
class PostTradeAnalysis:
    """
    Post-trade execution analysis.

    Compares actual execution against pre-trade estimates
    for best execution assessment.
    """

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    estimate_id: Optional[str] = None  # Link to pre-trade estimate
    timestamp_ns: int = field(default_factory=time.time_ns)

    # Order details
    instrument_isin: str = ""
    side: str = ""
    quantity: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")
    currency: str = "EUR"

    # Prices
    arrival_price: Decimal = Decimal("0")  # Decision price
    execution_price: Decimal = Decimal("0")  # Weighted avg fill price
    vwap_benchmark: Decimal = Decimal("0")  # Market VWAP
    close_price: Decimal = Decimal("0")

    # Actual costs
    commission: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    explicit_cost_bps: Decimal = Decimal("0")

    # Slippage analysis
    arrival_slippage_bps: Decimal = Decimal("0")
    vwap_slippage_bps: Decimal = Decimal("0")
    implementation_shortfall_bps: Decimal = Decimal("0")

    # Market impact
    permanent_impact_bps: Decimal = Decimal("0")
    temporary_impact_bps: Decimal = Decimal("0")
    timing_cost_bps: Decimal = Decimal("0")
    reversion_bps: Decimal = Decimal("0")

    # Total cost
    total_cost_bps: Decimal = Decimal("0")
    total_cost_currency: Decimal = Decimal("0")

    # vs. Pre-trade estimate
    estimated_cost_bps: Optional[Decimal] = None
    cost_vs_estimate_bps: Optional[Decimal] = None
    within_estimate_range: bool = True

    # Execution quality
    fill_rate: float = 0.0
    execution_duration_seconds: float = 0.0
    num_fills: int = 0
    avg_fill_size: Decimal = Decimal("0")

    # Metadata
    venue_mic: str = ""
    strategy_used: ExecutionStrategy = ExecutionStrategy.MARKET
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "order_id": self.order_id,
            "estimate_id": self.estimate_id,
            "timestamp_ns": self.timestamp_ns,
            "instrument_isin": self.instrument_isin,
            "side": self.side,
            "quantity": str(self.quantity),
            "filled_quantity": str(self.filled_quantity),
            "currency": self.currency,
            "arrival_price": str(self.arrival_price),
            "execution_price": str(self.execution_price),
            "vwap_benchmark": str(self.vwap_benchmark),
            "commission": str(self.commission),
            "fees": str(self.fees),
            "explicit_cost_bps": str(self.explicit_cost_bps),
            "arrival_slippage_bps": str(self.arrival_slippage_bps),
            "vwap_slippage_bps": str(self.vwap_slippage_bps),
            "implementation_shortfall_bps": str(self.implementation_shortfall_bps),
            "permanent_impact_bps": str(self.permanent_impact_bps),
            "temporary_impact_bps": str(self.temporary_impact_bps),
            "timing_cost_bps": str(self.timing_cost_bps),
            "total_cost_bps": str(self.total_cost_bps),
            "total_cost_currency": str(self.total_cost_currency),
            "estimated_cost_bps": str(self.estimated_cost_bps) if self.estimated_cost_bps else None,
            "cost_vs_estimate_bps": str(self.cost_vs_estimate_bps) if self.cost_vs_estimate_bps else None,
            "within_estimate_range": self.within_estimate_range,
            "fill_rate": self.fill_rate,
            "execution_duration_seconds": self.execution_duration_seconds,
            "num_fills": self.num_fills,
            "venue_mic": self.venue_mic,
            "strategy_used": self.strategy_used.value,
            "notes": self.notes,
        }


@dataclass
class TCAConfig:
    """
    Configuration for TCA compliance wrapper.

    Attributes:
        firm_lei: Firm's LEI for audit trail
        default_confidence_level: Default confidence for estimates
        impact_model: Name of market impact model to use
        enable_audit_trail: Log all TCA analyses to audit trail
        alert_threshold_bps: Trigger alert if cost exceeds estimate by this much
    """

    firm_lei: str = ""
    default_confidence_level: float = 0.95
    impact_model: str = "almgren_chriss"  # or "linear", "square_root"
    enable_audit_trail: bool = True
    alert_threshold_bps: Decimal = Decimal("50")

    # Model parameters (Almgren-Chriss style)
    permanent_impact_coefficient: float = 0.1
    temporary_impact_coefficient: float = 0.1
    volatility_scaling: float = 1.0

    # Timing cost parameters
    alpha_decay: float = 0.5  # Half-life for alpha decay

    # Strategy recommendation thresholds
    low_urgency_participation_pct: float = 5.0
    normal_urgency_participation_pct: float = 10.0
    high_urgency_participation_pct: float = 20.0


@dataclass
class TCAAggregateMetrics:
    """Aggregate TCA metrics for a period."""

    period_start: datetime
    period_end: datetime

    # Volume metrics
    total_orders: int = 0
    total_notional: Decimal = Decimal("0")
    total_fills: int = 0

    # Cost metrics (bps)
    avg_implementation_shortfall_bps: Decimal = Decimal("0")
    median_implementation_shortfall_bps: Decimal = Decimal("0")
    avg_arrival_slippage_bps: Decimal = Decimal("0")
    avg_vwap_slippage_bps: Decimal = Decimal("0")
    avg_explicit_cost_bps: Decimal = Decimal("0")
    avg_total_cost_bps: Decimal = Decimal("0")

    # Estimate accuracy
    avg_cost_vs_estimate_bps: Decimal = Decimal("0")
    pct_within_estimate: float = 0.0
    estimate_accuracy_score: float = 0.0

    # Execution quality
    avg_fill_rate: float = 0.0
    avg_execution_duration_seconds: float = 0.0

    # Breakdown by strategy
    metrics_by_strategy: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Breakdown by venue
    metrics_by_venue: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# TCA Compliance Wrapper
# =============================================================================


class TCAComplianceWrapper:
    """
    MiFID II compliant TCA wrapper.

    Provides pre-trade and post-trade analysis with audit trail
    integration for best execution compliance.

    Usage:
        tca = TCAComplianceWrapper(config)

        # Pre-trade estimation
        estimate = tca.estimate_pre_trade(order, market_data)

        # After execution
        analysis = tca.analyze_post_trade(order, fills, market_data, estimate)
    """

    def __init__(
        self,
        config: TCAConfig,
        slippage_provider: Optional[SlippageProvider] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize TCA wrapper.

        Args:
            config: TCA configuration
            slippage_provider: Optional external slippage model
            audit_callback: Callback for audit trail integration
        """
        self.config = config
        self.slippage_provider = slippage_provider
        self.audit_callback = audit_callback

        self._estimates: Dict[str, PreTradeEstimate] = {}
        self._analyses: List[PostTradeAnalysis] = []
        self._lock = threading.Lock()

    def estimate_pre_trade(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> PreTradeEstimate:
        """
        Generate pre-trade cost estimate.

        Per MiFID II Article 27: Firms must take "all sufficient steps"
        to obtain best execution. Pre-trade analysis informs this.

        Args:
            order: Order details:
                - order_id: str
                - instrument_isin: str
                - side: str ("BUY" or "SELL")
                - quantity: Decimal
                - notional: Decimal (optional)
            market_data: Market context:
                - bid: Decimal
                - ask: Decimal
                - mid: Decimal (optional)
                - spread_bps: Decimal (optional)
                - volatility: float (optional)
                - adv: Decimal (optional, average daily volume)

        Returns:
            PreTradeEstimate with cost projections.
        """
        estimate = PreTradeEstimate(
            order_id=order.get("order_id", ""),
            instrument_isin=order.get("instrument_isin", ""),
            side=order.get("side", ""),
            quantity=Decimal(str(order.get("quantity", 0))),
            currency=order.get("currency", "EUR"),
            model_name=self.config.impact_model,
            model_version="1.0",
            confidence_level=self.config.default_confidence_level,
        )

        # Extract market data
        bid = Decimal(str(market_data.get("bid", 0)))
        ask = Decimal(str(market_data.get("ask", 0)))
        mid = market_data.get("mid")
        if mid is None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        else:
            mid = Decimal(str(mid or 0))

        estimate.bid_price = bid
        estimate.ask_price = ask
        estimate.arrival_price = mid

        # Calculate spread
        if mid > 0:
            spread = ask - bid
            estimate.spread_bps = (spread / mid) * Decimal("10000")
        else:
            estimate.spread_bps = Decimal(str(market_data.get("spread_bps", 10)))

        # Get ADV and volatility
        adv = market_data.get("adv")
        if adv is not None:
            estimate.adv = Decimal(str(adv))

        estimate.volatility = market_data.get("volatility")

        # Calculate notional
        notional = order.get("notional")
        if notional is None and mid > 0:
            notional = mid * estimate.quantity
        estimate.notional_value = Decimal(str(notional or 0))

        # === Cost Estimation ===

        # 1. Spread cost (half spread for crossing)
        estimate.estimated_spread_cost_bps = estimate.spread_bps / 2

        # 2. Market impact
        if self.slippage_provider and estimate.adv:
            # Use external provider
            impact_bps = self.slippage_provider.estimate_slippage_bps(
                notional=float(estimate.notional_value),
                adv=float(estimate.adv),
                side=estimate.side,
                volatility=estimate.volatility,
                spread_bps=float(estimate.spread_bps),
            )
            estimate.estimated_impact_bps = Decimal(str(impact_bps))
        elif estimate.adv and estimate.adv > 0:
            # Built-in Almgren-Chriss style model
            participation = float(estimate.notional_value / estimate.adv)

            if self.config.impact_model == "linear":
                impact = self.config.permanent_impact_coefficient * participation * 10000
            elif self.config.impact_model == "square_root":
                impact = self.config.permanent_impact_coefficient * (participation ** 0.5) * 10000
            else:  # almgren_chriss
                # Combined permanent + temporary impact
                sigma = estimate.volatility if estimate.volatility else 0.02  # Default 2% vol
                permanent = self.config.permanent_impact_coefficient * participation
                temporary = self.config.temporary_impact_coefficient * (participation ** 0.5)
                impact = (permanent + temporary) * sigma * 10000 * self.config.volatility_scaling

            estimate.estimated_impact_bps = Decimal(str(min(impact, 500)))  # Cap at 500bps
        else:
            # Fallback: 5bps default impact
            estimate.estimated_impact_bps = Decimal("5")

        # 3. Timing cost (opportunity cost of not executing immediately)
        # Simplified: based on alpha decay
        estimate.estimated_timing_cost_bps = Decimal("2")  # Default 2bps

        # 4. Total estimated cost
        estimate.estimated_total_cost_bps = (
            estimate.estimated_spread_cost_bps +
            estimate.estimated_impact_bps +
            estimate.estimated_timing_cost_bps
        )

        # 5. Estimated execution price
        side_multiplier = 1 if estimate.side.upper() == "BUY" else -1
        price_adjustment = float(estimate.arrival_price) * float(estimate.estimated_total_cost_bps) / 10000
        estimate.estimated_execution_price = estimate.arrival_price + Decimal(str(side_multiplier * price_adjustment))

        # 6. Confidence range (simplified: ±50% of estimate)
        range_factor = Decimal("0.5")
        estimate.cost_range_low_bps = estimate.estimated_total_cost_bps * (1 - range_factor)
        estimate.cost_range_high_bps = estimate.estimated_total_cost_bps * (1 + range_factor)

        # === Strategy Recommendation ===
        estimate.recommended_strategy, estimate.recommended_urgency = self._recommend_strategy(
            estimate
        )

        if estimate.adv and estimate.adv > 0:
            if estimate.recommended_urgency == "low":
                rate = self.config.low_urgency_participation_pct
            elif estimate.recommended_urgency == "high":
                rate = self.config.high_urgency_participation_pct
            else:
                rate = self.config.normal_urgency_participation_pct
            estimate.participation_rate_pct = rate

            # Expected duration
            participation = float(estimate.notional_value / estimate.adv)
            if rate > 0:
                estimate.expected_duration_seconds = (participation / (rate / 100)) * 6.5 * 3600  # 6.5hr trading day

        # Store estimate
        with self._lock:
            self._estimates[estimate.estimate_id] = estimate

        # Audit callback
        if self.audit_callback:
            self.audit_callback({
                "type": "pre_trade_estimate",
                "estimate_id": estimate.estimate_id,
                "order_id": estimate.order_id,
                "timestamp_ns": estimate.timestamp_ns,
                "estimated_cost_bps": str(estimate.estimated_total_cost_bps),
                "recommended_strategy": estimate.recommended_strategy.value,
            })

        return estimate

    def analyze_post_trade(
        self,
        order: Dict[str, Any],
        fills: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        estimate: Optional[PreTradeEstimate] = None,
    ) -> PostTradeAnalysis:
        """
        Analyze execution quality post-trade.

        Args:
            order: Original order details
            fills: List of fill data:
                - price: Decimal
                - quantity: Decimal
                - timestamp_ms: int
                - commission: Decimal (optional)
                - fees: Decimal (optional)
                - venue_mic: str (optional)
            market_data: Market context including:
                - arrival_price: Decimal
                - vwap: Decimal (market VWAP during execution)
                - close: Decimal
            estimate: Optional pre-trade estimate for comparison

        Returns:
            PostTradeAnalysis with execution breakdown.
        """
        analysis = PostTradeAnalysis(
            order_id=order.get("order_id", ""),
            estimate_id=estimate.estimate_id if estimate else None,
            instrument_isin=order.get("instrument_isin", ""),
            side=order.get("side", ""),
            quantity=Decimal(str(order.get("quantity", 0))),
            currency=order.get("currency", "EUR"),
        )

        # Process fills
        if not fills:
            analysis.notes.append("No fills received")
            return analysis

        total_notional = Decimal("0")
        total_quantity = Decimal("0")
        total_commission = Decimal("0")
        total_fees = Decimal("0")
        first_fill_ts = None
        last_fill_ts = None
        venues = set()

        for fill in fills:
            qty = Decimal(str(fill.get("quantity", 0)))
            price = Decimal(str(fill.get("price", 0)))

            total_quantity += qty
            total_notional += qty * price
            total_commission += Decimal(str(fill.get("commission", 0)))
            total_fees += Decimal(str(fill.get("fees", 0)))

            ts = fill.get("timestamp_ms", 0)
            if first_fill_ts is None or ts < first_fill_ts:
                first_fill_ts = ts
            if last_fill_ts is None or ts > last_fill_ts:
                last_fill_ts = ts

            if fill.get("venue_mic"):
                venues.add(fill["venue_mic"])

        analysis.filled_quantity = total_quantity
        analysis.num_fills = len(fills)

        if analysis.filled_quantity > 0:
            analysis.fill_rate = float(analysis.filled_quantity / analysis.quantity)
            analysis.execution_price = total_notional / analysis.filled_quantity
            analysis.avg_fill_size = analysis.filled_quantity / analysis.num_fills

        if first_fill_ts and last_fill_ts:
            analysis.execution_duration_seconds = (last_fill_ts - first_fill_ts) / 1000

        if venues:
            analysis.venue_mic = ",".join(sorted(venues))

        # Costs
        analysis.commission = total_commission
        analysis.fees = total_fees

        if total_notional > 0:
            explicit_cost = total_commission + total_fees
            analysis.explicit_cost_bps = (explicit_cost / total_notional) * Decimal("10000")

        # Benchmark prices
        arrival_price = market_data.get("arrival_price")
        if arrival_price is None and estimate:
            arrival_price = estimate.arrival_price
        analysis.arrival_price = Decimal(str(arrival_price or 0))

        analysis.vwap_benchmark = Decimal(str(market_data.get("vwap", 0)))
        analysis.close_price = Decimal(str(market_data.get("close", 0)))

        # === Slippage Calculations ===

        if analysis.arrival_price > 0 and analysis.execution_price > 0:
            # Arrival price slippage (Implementation Shortfall basis)
            side_mult = 1 if analysis.side.upper() == "BUY" else -1
            price_diff = analysis.execution_price - analysis.arrival_price
            analysis.arrival_slippage_bps = (
                side_mult * (price_diff / analysis.arrival_price) * Decimal("10000")
            )

        if analysis.vwap_benchmark > 0 and analysis.execution_price > 0:
            # VWAP slippage
            side_mult = 1 if analysis.side.upper() == "BUY" else -1
            price_diff = analysis.execution_price - analysis.vwap_benchmark
            analysis.vwap_slippage_bps = (
                side_mult * (price_diff / analysis.vwap_benchmark) * Decimal("10000")
            )

        # Implementation shortfall (total)
        analysis.implementation_shortfall_bps = (
            analysis.arrival_slippage_bps + analysis.explicit_cost_bps
        )

        # Total cost
        analysis.total_cost_bps = analysis.implementation_shortfall_bps
        analysis.total_cost_currency = (
            total_notional * analysis.total_cost_bps / Decimal("10000")
        )

        # === Comparison with Pre-Trade Estimate ===

        if estimate:
            analysis.estimated_cost_bps = estimate.estimated_total_cost_bps
            analysis.cost_vs_estimate_bps = (
                analysis.total_cost_bps - estimate.estimated_total_cost_bps
            )

            # Check if within range
            analysis.within_estimate_range = (
                estimate.cost_range_low_bps <= analysis.total_cost_bps <= estimate.cost_range_high_bps
            )

            if not analysis.within_estimate_range:
                analysis.notes.append(
                    f"Cost {analysis.total_cost_bps}bps outside estimate range "
                    f"[{estimate.cost_range_low_bps}, {estimate.cost_range_high_bps}]"
                )

            # Alert if significantly worse
            if analysis.cost_vs_estimate_bps and analysis.cost_vs_estimate_bps > self.config.alert_threshold_bps:
                analysis.notes.append(
                    f"WARNING: Cost exceeded estimate by {analysis.cost_vs_estimate_bps}bps"
                )

        # Store analysis
        with self._lock:
            self._analyses.append(analysis)

        # Audit callback
        if self.audit_callback:
            self.audit_callback({
                "type": "post_trade_analysis",
                "analysis_id": analysis.analysis_id,
                "order_id": analysis.order_id,
                "timestamp_ns": analysis.timestamp_ns,
                "total_cost_bps": str(analysis.total_cost_bps),
                "implementation_shortfall_bps": str(analysis.implementation_shortfall_bps),
                "within_estimate_range": analysis.within_estimate_range,
            })

        return analysis

    def _recommend_strategy(
        self,
        estimate: PreTradeEstimate,
    ) -> tuple[ExecutionStrategy, str]:
        """
        Recommend execution strategy based on order characteristics.

        Returns:
            Tuple of (strategy, urgency)
        """
        # Calculate participation rate if ADV available
        if estimate.adv and estimate.adv > 0:
            participation = float(estimate.notional_value / estimate.adv)

            if participation < 0.01:  # < 1% ADV
                return ExecutionStrategy.MARKET, "normal"
            elif participation < 0.05:  # 1-5% ADV
                return ExecutionStrategy.VWAP, "normal"
            elif participation < 0.15:  # 5-15% ADV
                return ExecutionStrategy.TWAP, "normal"
            else:  # > 15% ADV
                return ExecutionStrategy.IS, "low"  # Minimize impact

        # Default
        return ExecutionStrategy.MARKET, "normal"

    def get_estimate(self, estimate_id: str) -> Optional[PreTradeEstimate]:
        """Get pre-trade estimate by ID."""
        with self._lock:
            return self._estimates.get(estimate_id)

    def get_estimates_for_order(self, order_id: str) -> List[PreTradeEstimate]:
        """Get all estimates for an order."""
        with self._lock:
            return [e for e in self._estimates.values() if e.order_id == order_id]

    def get_analyses(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PostTradeAnalysis]:
        """Get post-trade analyses for a period."""
        with self._lock:
            analyses = self._analyses.copy()

        # Convert datetime to nanoseconds with UTC correction
        # Note: datetime.utcnow().timestamp() has timezone issues on non-UTC systems
        current_time_ns = time.time_ns()
        current_time_sec = current_time_ns / 1e9

        def datetime_to_ns(dt: datetime) -> int:
            """Convert datetime to nanoseconds, correcting for UTC naive datetime."""
            ref_utc = datetime.utcnow()
            offset_sec = ref_utc.timestamp() - current_time_sec
            return int((dt.timestamp() - offset_sec) * 1e9)

        if start_time:
            start_ns = datetime_to_ns(start_time)
            analyses = [a for a in analyses if a.timestamp_ns >= start_ns]
        if end_time:
            end_ns = datetime_to_ns(end_time)
            analyses = [a for a in analyses if a.timestamp_ns <= end_ns]

        return analyses

    def get_aggregate_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> TCAAggregateMetrics:
        """
        Get aggregate TCA metrics for a period.

        Args:
            start_time: Period start
            end_time: Period end

        Returns:
            TCAAggregateMetrics with summary statistics.
        """
        analyses = self.get_analyses(start_time, end_time)

        metrics = TCAAggregateMetrics(
            period_start=start_time,
            period_end=end_time,
            total_orders=len(analyses),
        )

        if not analyses:
            return metrics

        # Aggregate calculations
        is_costs = []
        arrival_costs = []
        vwap_costs = []
        explicit_costs = []
        total_costs = []
        estimate_diffs = []
        within_estimate = 0
        fill_rates = []
        durations = []
        total_notional = Decimal("0")
        total_fills = 0

        strategy_data: Dict[str, List[PostTradeAnalysis]] = {}
        venue_data: Dict[str, List[PostTradeAnalysis]] = {}

        for a in analyses:
            is_costs.append(float(a.implementation_shortfall_bps))
            arrival_costs.append(float(a.arrival_slippage_bps))
            vwap_costs.append(float(a.vwap_slippage_bps))
            explicit_costs.append(float(a.explicit_cost_bps))
            total_costs.append(float(a.total_cost_bps))

            if a.cost_vs_estimate_bps is not None:
                estimate_diffs.append(float(a.cost_vs_estimate_bps))
            if a.within_estimate_range:
                within_estimate += 1

            fill_rates.append(a.fill_rate)
            durations.append(a.execution_duration_seconds)

            total_notional += a.filled_quantity * a.execution_price
            total_fills += a.num_fills

            # Group by strategy
            strategy_key = a.strategy_used.value
            if strategy_key not in strategy_data:
                strategy_data[strategy_key] = []
            strategy_data[strategy_key].append(a)

            # Group by venue
            if a.venue_mic:
                if a.venue_mic not in venue_data:
                    venue_data[a.venue_mic] = []
                venue_data[a.venue_mic].append(a)

        # Calculate averages
        import statistics

        metrics.total_notional = total_notional
        metrics.total_fills = total_fills

        metrics.avg_implementation_shortfall_bps = Decimal(str(statistics.mean(is_costs)))
        metrics.median_implementation_shortfall_bps = Decimal(str(statistics.median(is_costs)))
        metrics.avg_arrival_slippage_bps = Decimal(str(statistics.mean(arrival_costs)))
        metrics.avg_vwap_slippage_bps = Decimal(str(statistics.mean(vwap_costs)))
        metrics.avg_explicit_cost_bps = Decimal(str(statistics.mean(explicit_costs)))
        metrics.avg_total_cost_bps = Decimal(str(statistics.mean(total_costs)))

        if estimate_diffs:
            metrics.avg_cost_vs_estimate_bps = Decimal(str(statistics.mean(estimate_diffs)))

        metrics.pct_within_estimate = (within_estimate / len(analyses)) * 100
        metrics.avg_fill_rate = statistics.mean(fill_rates)
        metrics.avg_execution_duration_seconds = statistics.mean(durations)

        # Estimate accuracy score (100 = perfect, lower if estimates are off)
        if estimate_diffs:
            avg_abs_error = statistics.mean(abs(d) for d in estimate_diffs)
            metrics.estimate_accuracy_score = max(0, 100 - avg_abs_error)

        # Strategy breakdown
        for strategy, strategy_analyses in strategy_data.items():
            s_costs = [float(a.total_cost_bps) for a in strategy_analyses]
            metrics.metrics_by_strategy[strategy] = {
                "count": len(strategy_analyses),
                "avg_cost_bps": statistics.mean(s_costs),
                "median_cost_bps": statistics.median(s_costs),
            }

        # Venue breakdown
        for venue, venue_analyses in venue_data.items():
            v_costs = [float(a.total_cost_bps) for a in venue_analyses]
            metrics.metrics_by_venue[venue] = {
                "count": len(venue_analyses),
                "avg_cost_bps": statistics.mean(v_costs),
                "median_cost_bps": statistics.median(v_costs),
            }

        return metrics

    def clear_history(self) -> tuple[int, int]:
        """Clear stored estimates and analyses. Returns (estimates_cleared, analyses_cleared)."""
        with self._lock:
            e_count = len(self._estimates)
            a_count = len(self._analyses)
            self._estimates.clear()
            self._analyses.clear()
            return e_count, a_count


# =============================================================================
# Factory Functions
# =============================================================================


def create_tca_wrapper(
    firm_lei: str = "",
    slippage_provider: Optional[SlippageProvider] = None,
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    impact_model: str = "almgren_chriss",
) -> TCAComplianceWrapper:
    """
    Factory function to create TCA compliance wrapper.

    Args:
        firm_lei: Firm's LEI for audit trail
        slippage_provider: Optional external slippage model
        audit_callback: Callback for audit trail integration
        impact_model: Market impact model ("almgren_chriss", "linear", "square_root")

    Returns:
        Configured TCAComplianceWrapper instance.
    """
    config = TCAConfig(
        firm_lei=firm_lei,
        impact_model=impact_model,
        enable_audit_trail=audit_callback is not None,
    )

    return TCAComplianceWrapper(
        config=config,
        slippage_provider=slippage_provider,
        audit_callback=audit_callback,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "TCAMetricType",
    "TCABenchmark",
    "CostCategory",
    "ExecutionStrategy",
    # Data classes
    "PreTradeEstimate",
    "PostTradeAnalysis",
    "TCAConfig",
    "TCAAggregateMetrics",
    # Protocols
    "SlippageProvider",
    # Main class
    "TCAComplianceWrapper",
    # Factory
    "create_tca_wrapper",
]
