# -*- coding: utf-8 -*-
"""
services/forex_execution_integration.py

Integration layer combining ForexParametricSlippageProvider (L2+ TCA model)
with ForexDealerSimulator (OTC dealer execution simulation).

This module provides a unified interface for forex execution simulation that:
1. Uses parametric TCA for expected cost estimation (pre-trade)
2. Uses dealer simulation for realistic execution (trade time)
3. Combines results for accurate transaction cost analysis

Integration Architecture:
    ┌───────────────────────────────────┐
    │     ForexExecutionIntegration     │
    │  (Unified Forex Execution API)    │
    └─────────────┬─────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────────┐    ┌─────────────────────┐
│ForexParametric  │    │ ForexDealerSimulator│
│SlippageProvider │    │ (OTC Dealer Quotes) │
│ (L2+ TCA Model) │    │                     │
└─────────────────┘    └─────────────────────┘
    │                           │
    ▼                           ▼
Pre-trade Cost            Execution Result
Estimation                with Last-look

References:
- Evans & Lyons (2002): Order flow and exchange rate dynamics
- King et al. (2012): FX market structure
- Oomen (2017): Last-look in FX markets
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from execution_providers import (
    ForexParametricConfig,
    ForexParametricSlippageProvider,
    ForexSession,
    MarketState,
    Order,
    PairType,
)
from services.forex_dealer import (
    AggregatedQuote,
    DealerTier,
    ExecutionResult,
    ExecutionStats,
    ForexDealerConfig,
    ForexDealerSimulator,
    RejectReason,
    combine_with_parametric_slippage,
    create_forex_dealer_simulator,
    estimate_rejection_probability,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ForexExecutionConfig:
    """
    Configuration for integrated forex execution.

    Combines configurations for both TCA model and dealer simulation.

    Attributes:
        tca_config: ForexParametricConfig for TCA model
        dealer_config: ForexDealerConfig for dealer simulation
        execution_weight: Weight for execution slippage vs TCA estimate (0-1)
        use_dealer_for_large_orders: Use dealer sim for orders above threshold
        large_order_threshold_usd: Threshold for large order handling
        enable_adaptive_blending: Adapt blending based on fill quality
    """

    tca_config: Optional[ForexParametricConfig] = None
    dealer_config: Optional[ForexDealerConfig] = None
    execution_weight: float = 0.3
    use_dealer_for_large_orders: bool = True
    large_order_threshold_usd: float = 1_000_000.0
    enable_adaptive_blending: bool = True


@dataclass
class ForexExecutionEstimate:
    """
    Pre-trade cost estimation result.

    Contains comprehensive pre-trade analysis for a forex order.

    Attributes:
        expected_slippage_pips: Expected slippage from TCA model
        expected_rejection_prob: Estimated rejection probability
        session: Current forex session
        pair_type: Pair classification
        volatility_regime: Current volatility regime
        dealer_spread_pips: Expected dealer spread
        recommended_execution: Execution recommendation
        risk_factors: Dictionary of risk factors
    """

    expected_slippage_pips: float
    expected_rejection_prob: float
    session: ForexSession
    pair_type: PairType
    volatility_regime: str = "normal"
    dealer_spread_pips: float = 0.0
    recommended_execution: str = "immediate"
    risk_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expected_slippage_pips": self.expected_slippage_pips,
            "expected_rejection_prob": self.expected_rejection_prob,
            "session": self.session.value,
            "pair_type": self.pair_type.value,
            "volatility_regime": self.volatility_regime,
            "dealer_spread_pips": self.dealer_spread_pips,
            "recommended_execution": self.recommended_execution,
            "risk_factors": self.risk_factors,
        }


@dataclass
class ForexExecutionReport:
    """
    Full execution report combining TCA and dealer simulation.

    Provides a comprehensive view of the execution outcome.

    Attributes:
        pre_trade_estimate: Pre-trade cost estimation
        execution_result: Actual execution result from dealer sim
        tca_slippage_pips: Slippage predicted by TCA model
        dealer_slippage_pips: Actual slippage from dealer execution
        combined_slippage_pips: Weighted combination of TCA and dealer
        tca_error_pips: Difference between TCA prediction and actual
        fill_rate_recent: Recent fill rate percentage
        total_latency_ms: Total execution latency
        execution_quality: Quality score (0-1, higher is better)
    """

    pre_trade_estimate: ForexExecutionEstimate
    execution_result: ExecutionResult
    tca_slippage_pips: float
    dealer_slippage_pips: float
    combined_slippage_pips: float
    tca_error_pips: float = 0.0
    fill_rate_recent: float = 0.0
    total_latency_ms: float = 0.0
    execution_quality: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pre_trade_estimate": self.pre_trade_estimate.to_dict(),
            "execution_result": self.execution_result.to_dict(),
            "tca_slippage_pips": self.tca_slippage_pips,
            "dealer_slippage_pips": self.dealer_slippage_pips,
            "combined_slippage_pips": self.combined_slippage_pips,
            "tca_error_pips": self.tca_error_pips,
            "fill_rate_recent": self.fill_rate_recent,
            "total_latency_ms": self.total_latency_ms,
            "execution_quality": self.execution_quality,
        }


# =============================================================================
# Main Integration Class
# =============================================================================


class ForexExecutionIntegration:
    """
    Integrated forex execution combining TCA model and dealer simulation.

    This class provides a unified interface for forex execution that:
    1. Computes pre-trade cost estimates using ForexParametricSlippageProvider
    2. Simulates realistic execution using ForexDealerSimulator
    3. Tracks and adapts based on fill quality

    The integration handles the fundamental difference between:
    - TCA Model: Statistical expected cost (pre-trade planning)
    - Dealer Sim: Realistic execution with last-look, rejection, etc.

    Usage:
        >>> integration = ForexExecutionIntegration()
        >>> # Pre-trade estimation
        >>> estimate = integration.estimate_execution_cost(
        ...     symbol="EUR_USD",
        ...     side="BUY",
        ...     size_usd=500_000,
        ...     mid_price=1.0850,
        ... )
        >>> print(f"Expected slippage: {estimate.expected_slippage_pips:.2f} pips")
        >>>
        >>> # Full execution simulation
        >>> report = integration.execute(
        ...     symbol="EUR_USD",
        ...     side="BUY",
        ...     size_usd=500_000,
        ...     mid_price=1.0850,
        ... )
        >>> if report.execution_result.filled:
        ...     print(f"Filled at {report.execution_result.fill_price}")
        ... else:
        ...     print(f"Rejected: {report.execution_result.reject_reason}")
    """

    def __init__(
        self,
        config: Optional[ForexExecutionConfig] = None,
        seed: Optional[int] = None,
        tca_profile: str = "retail",
        dealer_profile: str = "retail",
    ) -> None:
        """
        Initialize forex execution integration.

        Args:
            config: Full configuration (or use defaults)
            seed: Random seed for reproducibility
            tca_profile: Profile for TCA model
            dealer_profile: Profile for dealer simulator
        """
        self.config = config or ForexExecutionConfig()

        # Initialize TCA model
        self._tca = ForexParametricSlippageProvider(
            config=self.config.tca_config,
            spread_profile=tca_profile,
        )

        # Initialize dealer simulator
        self._dealer = create_forex_dealer_simulator(
            config=(self.config.dealer_config.__dict__ if self.config.dealer_config else None),
            seed=seed,
            profile=dealer_profile,
        )

        # Adaptive blending tracking
        self._blend_weight = self.config.execution_weight
        self._tca_error_history: List[float] = []
        self._max_error_history = 100

    def estimate_execution_cost(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        mid_price: float,
        *,
        session: Optional[ForexSession] = None,
        hour_utc: Optional[int] = None,
        interest_rate_diff: Optional[float] = None,
        volatility_regime: str = "normal",
        upcoming_news: Optional[str] = None,
    ) -> ForexExecutionEstimate:
        """
        Estimate execution cost before trading.

        Combines TCA model prediction with dealer spread estimation
        and rejection probability.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            side: "BUY" or "SELL"
            size_usd: Order size in USD
            mid_price: Current mid price
            session: Forex session (auto-detected if None)
            hour_utc: Current hour UTC
            interest_rate_diff: Interest rate differential
            volatility_regime: Current volatility regime
            upcoming_news: Upcoming economic event

        Returns:
            ForexExecutionEstimate with comprehensive analysis
        """
        # Create order and market state for TCA
        order = Order(
            symbol=symbol,
            side=side.upper(),
            qty=size_usd,
            order_type="MARKET",
        )

        # Determine pip size for spread conversion
        is_jpy = "JPY" in symbol.upper()
        pip_size = 0.01 if is_jpy else 0.0001

        # Get dealer quote for spread estimation
        session_factor = self._get_session_factor(session, hour_utc)
        dealer_quote = self._dealer.get_aggregated_quote(
            symbol=symbol,
            mid_price=mid_price,
            session_factor=session_factor,
            order_size_usd=size_usd,
            hour_utc=hour_utc,
        )
        dealer_spread_pips = dealer_quote.spread / pip_size

        # Market state for TCA
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=dealer_quote.best_bid,
            ask=dealer_quote.best_ask,
            adv=size_usd * 1000,  # Rough ADV estimate
        )

        # Estimate participation ratio (rough)
        # Typical daily EUR/USD volume is ~$200B
        session_volume_estimate = 5e9 if session_factor >= 1.0 else 2e9
        participation = size_usd / session_volume_estimate

        # TCA slippage estimate
        tca_slippage = self._tca.compute_slippage_pips(
            order=order,
            market=market,
            participation_ratio=participation,
            session=session,
            interest_rate_diff=interest_rate_diff,
            upcoming_news=upcoming_news,
        )

        # Rejection probability estimate
        rejection_prob = estimate_rejection_probability(
            size_usd=size_usd,
            session_factor=session_factor,
            volatility_regime=volatility_regime,
        )

        # Classify pair
        pair_type = self._tca._classify_pair(symbol)

        # Determine session
        detected_session = session or self._tca._detect_session(market.timestamp)

        # Execution recommendation
        recommendation = self._get_execution_recommendation(
            size_usd=size_usd,
            rejection_prob=rejection_prob,
            dealer_spread_pips=dealer_spread_pips,
            session_factor=session_factor,
        )

        # Risk factors
        risk_factors = {
            "size_risk": min(1.0, size_usd / 5_000_000),
            "spread_risk": min(1.0, dealer_spread_pips / 5.0),
            "rejection_risk": rejection_prob,
            "session_liquidity": session_factor,
        }

        return ForexExecutionEstimate(
            expected_slippage_pips=tca_slippage,
            expected_rejection_prob=rejection_prob,
            session=detected_session,
            pair_type=pair_type,
            volatility_regime=volatility_regime,
            dealer_spread_pips=dealer_spread_pips,
            recommended_execution=recommendation,
            risk_factors=risk_factors,
        )

    def execute(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        mid_price: float,
        *,
        session: Optional[ForexSession] = None,
        hour_utc: Optional[int] = None,
        interest_rate_diff: Optional[float] = None,
        volatility_regime: str = "normal",
        upcoming_news: Optional[str] = None,
    ) -> ForexExecutionReport:
        """
        Execute forex order with full simulation.

        Performs both TCA estimation and dealer execution simulation,
        then combines results for comprehensive reporting.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            side: "BUY" or "SELL"
            size_usd: Order size in USD
            mid_price: Current mid price
            session: Forex session (auto-detected if None)
            hour_utc: Current hour UTC
            interest_rate_diff: Interest rate differential
            volatility_regime: Current volatility regime
            upcoming_news: Upcoming economic event

        Returns:
            ForexExecutionReport with full analysis
        """
        # Get pre-trade estimate
        estimate = self.estimate_execution_cost(
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            mid_price=mid_price,
            session=session,
            hour_utc=hour_utc,
            interest_rate_diff=interest_rate_diff,
            volatility_regime=volatility_regime,
            upcoming_news=upcoming_news,
        )

        # Get session factor for dealer simulation
        session_factor = self._get_session_factor(session, hour_utc)

        # Get dealer quote
        quote = self._dealer.get_aggregated_quote(
            symbol=symbol,
            mid_price=mid_price,
            session_factor=session_factor,
            order_size_usd=size_usd,
            hour_utc=hour_utc,
        )

        # Simulate small price movement for last-look
        # This simulates the market moving slightly during the last-look window
        price_drift = (0.0001 if "JPY" not in symbol.upper() else 0.01) * 0.5
        if side.upper() == "BUY":
            # For buyers, adverse move is price going up
            current_mid = mid_price + price_drift * (
                0.5 - 1.0 * (1.0 - estimate.expected_rejection_prob)
            )
        else:
            # For sellers, adverse move is price going down
            current_mid = mid_price - price_drift * (
                0.5 - 1.0 * (1.0 - estimate.expected_rejection_prob)
            )

        # Execute through dealer simulator
        execution_result = self._dealer.attempt_execution(
            is_buy=(side.upper() == "BUY"),
            size_usd=size_usd,
            quote=quote,
            current_mid=current_mid,
            symbol=symbol,
        )

        # Calculate dealer slippage
        dealer_slippage = execution_result.slippage_pips if execution_result.filled else 0.0

        # Combined slippage
        combined_slippage = combine_with_parametric_slippage(
            parametric_slippage_pips=estimate.expected_slippage_pips,
            execution_result=execution_result,
            weight_execution=self._blend_weight,
        )

        # TCA error (for adaptive learning)
        tca_error = (
            abs(dealer_slippage - estimate.expected_slippage_pips)
            if execution_result.filled
            else 0.0
        )

        # Update adaptive blending if enabled
        if self.config.enable_adaptive_blending and execution_result.filled:
            self._update_adaptive_blend(
                tca_predicted=estimate.expected_slippage_pips,
                actual=dealer_slippage,
            )

        # Update TCA model
        if execution_result.filled:
            self._tca.update_fill_quality(
                predicted_slippage_pips=estimate.expected_slippage_pips,
                actual_slippage_pips=dealer_slippage,
            )

        # Calculate execution quality (0-1, higher is better)
        quality = self._calculate_execution_quality(
            estimate=estimate,
            result=execution_result,
        )

        return ForexExecutionReport(
            pre_trade_estimate=estimate,
            execution_result=execution_result,
            tca_slippage_pips=estimate.expected_slippage_pips,
            dealer_slippage_pips=dealer_slippage,
            combined_slippage_pips=combined_slippage,
            tca_error_pips=tca_error,
            fill_rate_recent=self._dealer.get_recent_fill_rate(),
            total_latency_ms=execution_result.latency_ms,
            execution_quality=quality,
        )

    def get_dealer_stats(self) -> ExecutionStats:
        """Get dealer execution statistics."""
        return self._dealer.get_stats()

    def reset(self) -> None:
        """Reset all state."""
        self._dealer.reset_state()
        self._tca_error_history.clear()
        self._blend_weight = self.config.execution_weight

    def _get_session_factor(
        self,
        session: Optional[ForexSession],
        hour_utc: Optional[int],
    ) -> float:
        """Get session liquidity factor."""
        if session is not None:
            return self._tca.config.session_liquidity.get(session.value, 1.0)

        # Estimate from hour
        if hour_utc is not None:
            # London-NY overlap (12-16 UTC) is best
            if 12 <= hour_utc < 16:
                return 1.5
            # London/NY sessions
            elif 7 <= hour_utc < 21:
                return 1.0
            # Sydney/Tokyo
            elif hour_utc < 7 or hour_utc >= 21:
                return 0.7

        return 1.0

    def _get_execution_recommendation(
        self,
        size_usd: float,
        rejection_prob: float,
        dealer_spread_pips: float,
        session_factor: float,
    ) -> str:
        """
        Get execution recommendation based on conditions.

        Returns:
            Recommendation string: "immediate", "wait_for_better_liquidity",
            "split_order", or "use_limit_orders"
        """
        # High rejection probability
        if rejection_prob > 0.3:
            if size_usd > 5_000_000:
                return "split_order"
            elif session_factor < 0.8:
                return "wait_for_better_liquidity"

        # Wide spreads
        if dealer_spread_pips > 3.0:
            return "wait_for_better_liquidity"

        # Large orders
        if size_usd > 10_000_000:
            return "split_order"

        return "immediate"

    def _update_adaptive_blend(
        self,
        tca_predicted: float,
        actual: float,
    ) -> None:
        """Update adaptive blend weight based on TCA accuracy."""
        if tca_predicted <= 0:
            return

        error = abs(actual - tca_predicted)
        self._tca_error_history.append(error)

        if len(self._tca_error_history) > self._max_error_history:
            self._tca_error_history = self._tca_error_history[-self._max_error_history // 2 :]

        # Adjust blend weight based on recent accuracy
        if len(self._tca_error_history) >= 20:
            avg_error = sum(self._tca_error_history) / len(self._tca_error_history)
            avg_predicted = tca_predicted  # Use most recent

            if avg_predicted > 0:
                error_ratio = avg_error / avg_predicted

                # If TCA is accurate (error ratio < 0.3), reduce dealer weight
                # If TCA is inaccurate (error ratio > 0.5), increase dealer weight
                if error_ratio < 0.3:
                    self._blend_weight = max(0.1, self._blend_weight - 0.01)
                elif error_ratio > 0.5:
                    self._blend_weight = min(0.5, self._blend_weight + 0.01)

    def _calculate_execution_quality(
        self,
        estimate: ForexExecutionEstimate,
        result: ExecutionResult,
    ) -> float:
        """
        Calculate execution quality score.

        Quality = 1.0 is best (filled with no slippage, price improvement)
        Quality = 0.0 is worst (rejected or maximum slippage)
        """
        if not result.filled:
            return 0.0

        quality = 0.5  # Base quality for filled order

        # Better than expected slippage
        if result.slippage_pips < estimate.expected_slippage_pips * 0.8:
            quality += 0.2
        elif result.slippage_pips < estimate.expected_slippage_pips:
            quality += 0.1

        # Price improvement bonus
        if result.price_improvement > 0:
            quality += 0.15

        # Fast execution bonus
        if result.latency_ms < 50:
            quality += 0.1
        elif result.latency_ms < 100:
            quality += 0.05

        # Penalty for partial fill
        if result.partial_fill:
            quality -= 0.1

        # Penalty for high slippage
        if result.slippage_pips > estimate.expected_slippage_pips * 1.5:
            quality -= 0.15

        return max(0.0, min(1.0, quality))


# =============================================================================
# Factory Functions
# =============================================================================


def create_forex_execution_integration(
    tca_profile: str = "retail",
    dealer_profile: str = "retail",
    seed: Optional[int] = None,
    execution_weight: float = 0.3,
) -> ForexExecutionIntegration:
    """
    Create forex execution integration with common configurations.

    Args:
        tca_profile: TCA model profile ("retail", "institutional", "conservative")
        dealer_profile: Dealer simulator profile
        seed: Random seed for reproducibility
        execution_weight: Weight for execution slippage vs TCA

    Returns:
        Configured ForexExecutionIntegration
    """
    config = ForexExecutionConfig(execution_weight=execution_weight)

    return ForexExecutionIntegration(
        config=config,
        seed=seed,
        tca_profile=tca_profile,
        dealer_profile=dealer_profile,
    )


def create_institutional_integration(
    seed: Optional[int] = None,
) -> ForexExecutionIntegration:
    """
    Create integration configured for institutional trading.

    Institutional configuration features:
    - Tighter spreads in TCA model
    - More dealers in simulator
    - Lower execution weight (trust TCA more)
    """
    dealer_config = ForexDealerConfig(
        num_dealers=10,
        base_spread_pips=0.3,
        last_look_enabled=True,
        size_impact_threshold_usd=5_000_000,
    )

    config = ForexExecutionConfig(
        dealer_config=dealer_config,
        execution_weight=0.2,
        large_order_threshold_usd=5_000_000,
    )

    return ForexExecutionIntegration(
        config=config,
        seed=seed,
        tca_profile="institutional",
        dealer_profile="institutional",
    )


def create_retail_integration(
    seed: Optional[int] = None,
) -> ForexExecutionIntegration:
    """
    Create integration configured for retail trading.

    Retail configuration features:
    - Wider spreads in TCA model
    - Fewer dealers in simulator
    - Higher execution weight (dealer more realistic for retail)
    """
    dealer_config = ForexDealerConfig(
        num_dealers=5,
        base_spread_pips=1.2,
        last_look_enabled=True,
        size_impact_threshold_usd=500_000,
    )

    config = ForexExecutionConfig(
        dealer_config=dealer_config,
        execution_weight=0.4,
        large_order_threshold_usd=1_000_000,
    )

    return ForexExecutionIntegration(
        config=config,
        seed=seed,
        tca_profile="retail",
        dealer_profile="retail",
    )
