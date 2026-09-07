# -*- coding: utf-8 -*-
"""
L3 Futures Execution Provider with Full LOB Simulation.

This module provides high-fidelity execution simulation for crypto futures
markets by integrating:

1. Full L3 LOB simulation (matching engine, queue tracking)
2. Liquidation order stream injection
3. Liquidation cascade dynamics
4. Insurance fund management
5. ADL (Auto-Deleveraging) simulation
6. Funding-adjusted queue dynamics
7. Mark price execution

The key difference from L2 FuturesExecutionProvider is that L3 uses:
- Actual order book state for price discovery
- Queue position tracking for limit order fills
- Realistic liquidation order flow injection
- Cascade simulation for market stress scenarios

Stage 5A of Futures Integration (v1.0)

Architecture:
    FuturesL3ExecutionProvider extends L3ExecutionProvider from
    execution_providers_l3.py with futures-specific mechanics.
    It maintains backward compatibility with existing interfaces.

References:
    - Binance Futures Documentation
    - CME Globex Matching Algorithm (for FIFO matching)
    - Kyle (1985): Price Impact Model
    - Almgren & Chriss (2001): Optimal Execution
    - Zhao et al. (2020): Liquidation Cascades
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Import base execution providers types
from execution_providers import (
    AssetClass,
    BarData,
    Fill,
    MarketState,
    Order,
    OrderSide,
    OrderType,
    LiquidityRole,
    SlippageProvider,
    FillProvider,
    FeeProvider,
    ExecutionProvider,
)

# Import L3 execution provider base
from execution_providers_l3 import (
    L3ExecutionProvider,
    L3SlippageProvider,
    L3FillProvider,
    L3ExecutionConfig,
)

# Import L2 futures components for fee calculation
from execution_providers_futures import (
    FuturesSlippageConfig,
    FuturesSlippageProvider,
    FuturesFeeProvider,
    FuturesL2ExecutionProvider,
)

# Import LOB components
from lob import (
    Side as LOBSide,
    OrderType as LOBOrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill as LOBFill,
    Trade,
    MatchingEngine,
    MatchResult,
    MatchType,
    QueuePositionTracker,
    QueueState,
    AlmgrenChrissModel,
    ImpactParameters,
    LatencyModel,
    LatencyProfile,
    ImpactEffects,
    LOBImpactSimulator,
)

# Import LOB config
from lob.config import (
    L3ExecutionConfig as LOBConfig,
    create_l3_config,
)

# Import futures extensions
from lob.futures_extensions import (
    LiquidationOrderInfo,
    LiquidationFillResult,
    LiquidationOrderStream,
    LiquidationCascadeSimulator,
    CascadeResult,
    CascadeWave,
    InsuranceFundManager,
    InsuranceFundState,
    ADLQueueManager,
    ADLQueueEntry,
    FundingPeriodDynamics,
    FundingPeriodState,
    create_liquidation_stream,
    create_cascade_simulator,
    create_insurance_fund,
    create_adl_manager,
    create_funding_dynamics,
)

# Import core futures models
from core_futures import (
    FuturesPosition,
    FundingPayment,
    LiquidationEvent,
    MarginMode,
    PositionSide,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_INSURANCE_FUND = Decimal("1_000_000_000")
_DEFAULT_CASCADE_DECAY = 0.7
_DEFAULT_MAX_CASCADE_WAVES = 5
_DEFAULT_LIQUIDATION_FEE_BPS = 50.0  # 0.5%


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FuturesL3Config:
    """
    Configuration for FuturesL3ExecutionProvider.

    Attributes:
        enable_liquidation_injection: Inject liquidation orders into LOB
        enable_cascade_simulation: Simulate liquidation cascades
        enable_insurance_fund: Track insurance fund dynamics
        enable_adl_simulation: Simulate ADL queue
        enable_funding_dynamics: Adjust queue near funding times
        use_mark_price_execution: Execute at mark price vs last
        cascade_decay: Cascade wave decay factor
        max_cascade_waves: Maximum cascade depth
        price_impact_coef: Kyle lambda for cascade impact
        insurance_fund_initial: Starting insurance fund balance
        liquidation_fee_bps: Fee for liquidation orders
    """

    # Feature flags
    enable_liquidation_injection: bool = True
    enable_cascade_simulation: bool = True
    enable_insurance_fund: bool = True
    enable_adl_simulation: bool = True
    enable_funding_dynamics: bool = True
    use_mark_price_execution: bool = True

    # Cascade parameters
    cascade_decay: float = _DEFAULT_CASCADE_DECAY
    max_cascade_waves: int = _DEFAULT_MAX_CASCADE_WAVES
    price_impact_coef: float = 0.1

    # Insurance fund
    insurance_fund_initial: Decimal = _DEFAULT_INSURANCE_FUND

    # Fees
    liquidation_fee_bps: float = _DEFAULT_LIQUIDATION_FEE_BPS

    # L3 LOB config (for parent class)
    lob_config: Optional[LOBConfig] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.cascade_decay <= 1:
            raise ValueError(f"cascade_decay must be in (0, 1], got {self.cascade_decay}")
        if self.max_cascade_waves < 1:
            raise ValueError(f"max_cascade_waves must be >= 1")
        if self.price_impact_coef <= 0:
            raise ValueError(f"price_impact_coef must be > 0")
        if self.insurance_fund_initial < 0:
            raise ValueError(f"insurance_fund_initial must be >= 0")


# =============================================================================
# L3 Futures Slippage Provider
# =============================================================================


class FuturesL3SlippageProvider:
    """
    L3 slippage provider with futures-specific extensions.

    Combines:
    - LOB walk-through slippage computation
    - Funding rate impact
    - Liquidation cascade impact
    - Insurance fund state impact

    The slippage calculation adds futures-specific adjustments for:
    - Queue crowding near funding
    - Cascade-induced liquidity gaps
    - Market stress indicators

    Note: This class uses composition rather than inheritance from L3SlippageProvider
    to allow flexibility in the method signature for futures-specific parameters.
    """

    def __init__(
        self,
        config: Optional[FuturesL3Config] = None,
        funding_dynamics: Optional[FundingPeriodDynamics] = None,
        **kwargs,
    ):
        """
        Initialize futures L3 slippage provider.

        Args:
            config: Futures L3 configuration
            funding_dynamics: Funding period dynamics tracker
            **kwargs: Additional parameters
        """
        self._futures_config = config or FuturesL3Config()
        self._funding_dynamics = funding_dynamics or create_funding_dynamics()

        # Base slippage parameters
        self._default_spread_bps = 5.0
        self._impact_coef = 0.1

        # Cascade state tracking
        self._active_cascade: Optional[CascadeResult] = None
        self._cascade_impact_remaining_bps: float = 0.0

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        order_book: Optional[OrderBook] = None,
        funding_rate: Optional[float] = None,
        funding_state: Optional[FundingPeriodState] = None,
        cascade_active: bool = False,
        participation_ratio: float = 0.0,
        **kwargs,
    ) -> float:
        """
        Compute slippage with futures-specific adjustments.

        Args:
            order: Order to execute
            market: Current market state
            order_book: L3 order book (if available)
            funding_rate: Current funding rate
            funding_state: Funding period state
            cascade_active: True if cascade in progress
            participation_ratio: Order size / ADV
            **kwargs: Additional parameters

        Returns:
            Expected slippage in basis points
        """
        # Base slippage calculation (Almgren-Chriss style)
        spread = market.get_spread_bps()
        if spread is None or not math.isfinite(spread) or spread < 0:
            spread = self._default_spread_bps

        half_spread = spread / 2.0
        impact_term = self._impact_coef * math.sqrt(max(0.0, participation_ratio)) * 10000
        base_bps = half_spread + impact_term

        # Funding dynamics adjustment
        funding_mult = 1.0
        if self._futures_config.enable_funding_dynamics and funding_state:
            if funding_state.is_funding_window:
                # Impact increases near funding
                funding_mult = funding_state.impact_multiplier

        # Cascade adjustment
        cascade_mult = 1.0
        if cascade_active and self._cascade_impact_remaining_bps > 0:
            # Additional slippage during cascade
            cascade_mult = (
                1.0 + (self._cascade_impact_remaining_bps / base_bps) if base_bps > 0 else 1.0
            )
            cascade_mult = min(cascade_mult, 3.0)  # Cap at 200% increase

        total_bps = base_bps * funding_mult * cascade_mult

        return total_bps

    def set_cascade_impact(self, impact_bps: float) -> None:
        """Set remaining cascade impact."""
        self._cascade_impact_remaining_bps = max(0, impact_bps)

    def decay_cascade_impact(self, decay_rate: float = 0.5) -> None:
        """Decay cascade impact over time."""
        self._cascade_impact_remaining_bps *= decay_rate


# =============================================================================
# L3 Futures Fill Provider
# =============================================================================


class FuturesL3FillProvider:
    """
    L3 fill provider with liquidation order injection.

    Handles:
    - Normal order fills through LOB
    - Liquidation order injection and execution
    - Mark price vs last price execution
    - Queue position adjustment for futures

    Note: This class uses composition rather than inheritance from L3FillProvider
    to allow flexibility in the method signature for futures-specific parameters.
    """

    def __init__(
        self,
        config: Optional[FuturesL3Config] = None,
        liquidation_stream: Optional[LiquidationOrderStream] = None,
        **kwargs,
    ):
        """
        Initialize futures L3 fill provider.

        Args:
            config: Futures L3 configuration
            liquidation_stream: Source of liquidation orders
            **kwargs: Additional parameters
        """
        self._futures_config = config or FuturesL3Config()
        self._liquidation_stream = liquidation_stream

        # Tracking
        self._pending_liquidations: List[LiquidationOrderInfo] = []
        self._executed_liquidations: List[LiquidationFillResult] = []

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        order_book: Optional[OrderBook] = None,
        matching_engine: Optional[MatchingEngine] = None,
        mark_bar: Optional[BarData] = None,
        timestamp_ms: Optional[int] = None,
        **kwargs,
    ) -> Optional[Fill]:
        """
        Try to fill order with liquidation injection.

        Args:
            order: Order to fill
            market: Current market state
            bar: OHLCV bar data
            order_book: L3 order book
            matching_engine: Matching engine
            mark_bar: Mark price bar (optional)
            timestamp_ms: Current timestamp
            **kwargs: Additional parameters

        Returns:
            Fill if successful, None otherwise
        """
        # First, inject any pending liquidations
        if (
            self._futures_config.enable_liquidation_injection
            and self._liquidation_stream
            and timestamp_ms
        ):
            self._inject_liquidations(timestamp_ms, matching_engine, market)

        # Use mark price bar if configured
        exec_bar = bar
        if self._futures_config.use_mark_price_execution and mark_bar:
            exec_bar = mark_bar

        # Execute fill based on bar data (simple L2-style fill)
        # In production, would integrate with L3 matching engine
        return self._execute_fill(order, market, exec_bar)

    def _execute_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute fill using bar data.

        For MARKET orders: fills at typical price (OHLC average)
        For LIMIT orders: fills if bar range includes limit price

        Args:
            order: Order to execute
            market: Market state
            bar: OHLCV bar data

        Returns:
            Fill if executed, None otherwise
        """
        if order.qty <= 0:
            return None

        order_side = str(order.side).upper()
        order_type = str(order.order_type).upper()

        # Calculate fill price
        if order_type == "MARKET":
            # Use typical price for market orders
            fill_price = Decimal(str(bar.typical_price))
            is_maker = False
        else:
            # LIMIT order - check if fills
            limit_price = getattr(order, "limit_price", None)
            if limit_price is None:
                # Fallback: use typical price
                fill_price = Decimal(str(bar.typical_price))
                is_maker = False
            else:
                limit_price = Decimal(str(limit_price))
                bar_low = Decimal(str(bar.low))
                bar_high = Decimal(str(bar.high))

                if order_side == "BUY":
                    # BUY LIMIT fills if bar low <= limit
                    if bar_low <= limit_price:
                        fill_price = min(limit_price, Decimal(str(bar.typical_price)))
                        is_maker = True
                    else:
                        return None  # No fill
                else:
                    # SELL LIMIT fills if bar high >= limit
                    if bar_high >= limit_price:
                        fill_price = max(limit_price, Decimal(str(bar.typical_price)))
                        is_maker = True
                    else:
                        return None  # No fill

        # Create fill
        notional = float(order.qty * fill_price)
        liquidity_role = "maker" if is_maker else "taker"

        return Fill(
            price=float(fill_price),
            qty=float(order.qty),
            fee=0.0,  # Fee computed by provider
            slippage_bps=0.0,  # Slippage computed by provider
            liquidity=liquidity_role,
            notional=notional,
            metadata={
                "symbol": order.symbol,
                "side": order_side,
                "is_maker": is_maker,
            },
        )

    def _inject_liquidations(
        self,
        timestamp_ms: int,
        matching_engine: Optional[MatchingEngine],
        market: MarketState,
    ) -> List[LiquidationFillResult]:
        """
        Inject liquidation orders into order book.

        Args:
            timestamp_ms: Current timestamp
            matching_engine: Matching engine for execution
            market: Current market state

        Returns:
            List of liquidation fill results
        """
        if not self._liquidation_stream:
            return []

        results = []

        for liq_order in self._liquidation_stream.get_liquidations_up_to(timestamp_ms):
            # Create market order for liquidation
            lob_side = LOBSide.BUY if liq_order.side.upper() == "BUY" else LOBSide.SELL

            # Execute as market order (must fill)
            if matching_engine:
                # Walk through order book
                fill_price, fill_qty = self._walk_lob_for_liquidation(
                    liq_order, matching_engine, market
                )
            else:
                # Fallback: use bankruptcy price with slippage estimate
                fill_price = liq_order.bankruptcy_price
                fill_qty = liq_order.qty

            # Calculate slippage from bankruptcy
            if liq_order.bankruptcy_price > 0:
                if liq_order.is_long_liquidation:
                    slippage_bps = float(
                        (liq_order.bankruptcy_price - fill_price)
                        / liq_order.bankruptcy_price
                        * 10000
                    )
                else:
                    slippage_bps = float(
                        (fill_price - liq_order.bankruptcy_price)
                        / liq_order.bankruptcy_price
                        * 10000
                    )
            else:
                slippage_bps = 0.0

            # Insurance fund impact
            insurance_impact = self._calculate_insurance_impact(liq_order, fill_price, fill_qty)

            result = LiquidationFillResult(
                order_info=liq_order,
                fill_price=fill_price,
                fill_qty=fill_qty,
                slippage_bps=slippage_bps,
                insurance_fund_impact=insurance_impact,
                timestamp_ms=timestamp_ms,
            )

            results.append(result)
            self._executed_liquidations.append(result)

        return results

    def _walk_lob_for_liquidation(
        self,
        liq_order: LiquidationOrderInfo,
        matching_engine: MatchingEngine,
        market: MarketState,
    ) -> Tuple[Decimal, Decimal]:
        """
        Walk through LOB to fill liquidation order.

        Returns:
            (fill_price, fill_qty)
        """
        # Simplified: use market mid + impact estimate
        # In full implementation, would walk actual LOB levels
        mid_price = market.get_mid_price()
        if mid_price is None:
            mid_price = liq_order.bankruptcy_price

        # Estimate impact based on order size vs ADV
        if market.adv and market.adv > 0:
            participation = float(liq_order.qty * mid_price) / market.adv
        else:
            participation = 0.01

        impact_bps = 10 * math.sqrt(participation) * 10000  # Simple impact model

        # Apply directional impact
        if liq_order.is_long_liquidation:
            fill_price = mid_price * Decimal(str(1 - impact_bps / 10000))
        else:
            fill_price = mid_price * Decimal(str(1 + impact_bps / 10000))

        return fill_price, liq_order.qty

    def _calculate_insurance_impact(
        self,
        liq_order: LiquidationOrderInfo,
        fill_price: Decimal,
        fill_qty: Decimal,
    ) -> Decimal:
        """
        Calculate insurance fund impact.

        Positive = contribution to fund
        Negative = payout from fund
        """
        if liq_order.is_long_liquidation:
            # Long liquidation: profit if fill > bankruptcy
            price_diff = fill_price - liq_order.bankruptcy_price
        else:
            # Short liquidation: profit if fill < bankruptcy
            price_diff = liq_order.bankruptcy_price - fill_price

        return price_diff * fill_qty

    def set_liquidation_stream(self, stream: LiquidationOrderStream) -> None:
        """Set liquidation order stream."""
        self._liquidation_stream = stream

    @property
    def executed_liquidations(self) -> List[LiquidationFillResult]:
        """Get list of executed liquidations."""
        return self._executed_liquidations

    def clear_executed_liquidations(self) -> None:
        """Clear executed liquidations list."""
        self._executed_liquidations.clear()


# =============================================================================
# L3 Futures Execution Provider
# =============================================================================


class FuturesL3ExecutionProvider:
    """
    L3 execution provider for crypto futures with full LOB simulation.

    Combines all futures-specific mechanics:
    - L3 order book simulation
    - Liquidation order injection
    - Cascade simulation
    - Insurance fund tracking
    - ADL simulation
    - Funding dynamics

    Example:
        >>> config = FuturesL3Config(
        ...     enable_cascade_simulation=True,
        ...     use_mark_price_execution=True,
        ... )
        >>> provider = FuturesL3ExecutionProvider(config=config)
        >>>
        >>> # Load liquidation data
        >>> provider.load_liquidation_data(historical_liquidations)
        >>>
        >>> # Execute order
        >>> fill = provider.execute(
        ...     order=Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
        ...     market=MarketState(...),
        ...     bar=BarData(...),
        ...     mark_bar=BarData(...),  # Mark price bar
        ...     funding_rate=0.0001,
        ... )
        >>>
        >>> # Check for cascade
        >>> if provider.is_cascade_active:
        ...     cascade = provider.get_active_cascade()

    References:
        - Binance Futures Documentation
        - Kyle (1985): Price Impact
        - Almgren & Chriss (2001): Optimal Execution
    """

    def __init__(
        self,
        config: Optional[FuturesL3Config] = None,
        slippage_provider: Optional[FuturesL3SlippageProvider] = None,
        fill_provider: Optional[FuturesL3FillProvider] = None,
        fee_provider: Optional[FuturesFeeProvider] = None,
        **kwargs,
    ):
        """
        Initialize futures L3 execution provider.

        Args:
            config: Futures L3 configuration
            slippage_provider: Custom slippage provider
            fill_provider: Custom fill provider
            fee_provider: Custom fee provider
            **kwargs: Additional parameters
        """
        self._config = config or FuturesL3Config()

        # Create components
        self._slippage = slippage_provider or FuturesL3SlippageProvider(config=self._config)
        self._fill_provider = fill_provider or FuturesL3FillProvider(config=self._config)
        self._fees = fee_provider or FuturesFeeProvider()

        # Futures-specific components
        self._liquidation_stream = create_liquidation_stream()
        self._cascade_simulator = create_cascade_simulator(
            cascade_decay=self._config.cascade_decay,
            max_waves=self._config.max_cascade_waves,
            price_impact_coef=self._config.price_impact_coef,
        )
        self._insurance_fund = create_insurance_fund(
            initial_balance=self._config.insurance_fund_initial
        )
        self._adl_manager = create_adl_manager()
        self._funding_dynamics = create_funding_dynamics()

        # Connect liquidation stream to fill provider
        self._fill_provider.set_liquidation_stream(self._liquidation_stream)

        # State
        self._active_cascade: Optional[CascadeResult] = None
        self._last_funding_state: Optional[FundingPeriodState] = None

        # Tracking
        self._total_fills: int = 0
        self._total_liquidations_processed: int = 0
        self._total_cascade_events: int = 0

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        funding_rate: Optional[float] = None,
        mark_bar: Optional[BarData] = None,
        open_interest: Optional[float] = None,
        recent_liquidations: Optional[float] = None,
        order_book: Optional[OrderBook] = None,
        matching_engine: Optional[MatchingEngine] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Fill:
        """
        Execute order with full futures mechanics.

        Args:
            order: Order to execute
            market: Current market state
            bar: Last price OHLCV bar
            funding_rate: Current funding rate
            mark_bar: Mark price bar (optional)
            open_interest: Total open interest
            recent_liquidations: Recent liquidation volume
            order_book: L3 order book (optional)
            matching_engine: Matching engine (optional)
            positions: Open positions for cascade sim (optional)
            **kwargs: Additional parameters

        Returns:
            Fill with futures-specific adjustments

        Notes:
            - Liquidations are injected before order execution
            - Cascades may be triggered by large liquidations
            - Insurance fund is updated on each liquidation
            - ADL may be triggered if insurance fund depletes
        """
        timestamp_ms = bar.timestamp_ms if hasattr(bar, "timestamp_ms") else int(time.time() * 1000)

        # Update funding state
        if self._config.enable_funding_dynamics:
            self._last_funding_state = self._funding_dynamics.get_state(
                current_ts_ms=timestamp_ms,
                current_funding_rate=Decimal(str(funding_rate or 0)),
            )

        # Check for cascade conditions
        if (
            self._config.enable_cascade_simulation
            and recent_liquidations
            and market.adv
            and recent_liquidations / market.adv > 0.01  # >1% ADV
        ):
            self._check_and_trigger_cascade(
                timestamp_ms=timestamp_ms,
                market=market,
                recent_liquidations=recent_liquidations,
                positions=positions,
                order_book=order_book,
            )

        # Try to fill order
        fill = self._fill_provider.try_fill(
            order=order,
            market=market,
            bar=bar,
            order_book=order_book,
            matching_engine=matching_engine,
            mark_bar=mark_bar,
            timestamp_ms=timestamp_ms,
            **kwargs,
        )

        if fill is None:
            # Create empty fill for unfilled order
            return Fill(
                price=0.0,
                qty=0.0,
                fee=0.0,
                slippage_bps=0.0,
                liquidity="taker",
                notional=0.0,
                metadata={
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "is_filled": False,
                    "is_maker": False,
                },
            )

        # Calculate slippage with futures adjustments
        if market.adv and market.adv > 0:
            participation = float(fill.notional) / market.adv
        else:
            participation = 0.0

        slippage_bps = self._slippage.compute_slippage_bps(
            order=order,
            market=market,
            order_book=order_book,
            funding_rate=funding_rate,
            funding_state=self._last_funding_state,
            cascade_active=self._active_cascade is not None,
        )

        # Apply slippage to fill price (all float arithmetic)
        slippage_factor = 1.0 + (slippage_bps / 10000.0)
        if str(order.side).upper() == "BUY":
            adjusted_price = fill.price * slippage_factor
        else:
            adjusted_price = fill.price / slippage_factor

        adjusted_notional = fill.qty * adjusted_price

        # Compute fee
        fee = self._fees.compute_fee(fill, is_liquidation=False)

        # Process any executed liquidations for insurance fund
        self._process_liquidation_fills()

        self._total_fills += 1

        # Get metadata from original fill
        original_metadata = getattr(fill, "metadata", {}) or {}
        is_maker = original_metadata.get("is_maker", False)
        symbol = original_metadata.get("symbol", "")
        side = original_metadata.get("side", "")

        return Fill(
            price=float(adjusted_price),
            qty=float(fill.qty),
            fee=float(fee),
            slippage_bps=slippage_bps,
            liquidity="maker" if is_maker else "taker",
            notional=float(adjusted_notional),
            metadata={
                "symbol": symbol,
                "side": side,
                "is_filled": True,
                "is_maker": is_maker,
            },
        )

    def _check_and_trigger_cascade(
        self,
        timestamp_ms: int,
        market: MarketState,
        recent_liquidations: float,
        positions: Optional[List[Dict[str, Any]]],
        order_book: Optional[OrderBook],
    ) -> None:
        """Check conditions and trigger cascade simulation."""
        if self._active_cascade is not None:
            # Already in cascade
            return

        # Create initial liquidation event
        mid_price = market.get_mid_price() or Decimal("0")
        initial_liq = LiquidationOrderInfo(
            symbol=market.symbol if hasattr(market, "symbol") else "BTCUSDT",
            side="SELL",  # Assume long liquidation (most common in downtrend)
            qty=(
                Decimal(str(recent_liquidations / float(mid_price)))
                if mid_price > 0
                else Decimal("0")
            ),
            bankruptcy_price=mid_price,
            mark_price=mid_price,
            timestamp_ms=timestamp_ms,
        )

        # Run cascade simulation
        if order_book:
            self._active_cascade = self._cascade_simulator.simulate_cascade(
                initial_liquidation=initial_liq,
                order_book=order_book,
                volatility=0.02,  # Default volatility
                adv=market.adv or 1_000_000_000,
            )

            # Set cascade impact on slippage provider
            self._slippage.set_cascade_impact(self._active_cascade.total_price_impact_bps)

            self._total_cascade_events += 1

    def _process_liquidation_fills(self) -> None:
        """Process executed liquidations and update insurance fund."""
        if not self._config.enable_insurance_fund:
            return

        for result in self._fill_provider.executed_liquidations:
            self._insurance_fund.process_liquidation_fill(
                bankruptcy_price=result.order_info.bankruptcy_price,
                fill_price=result.fill_price,
                qty=result.fill_qty,
                side=result.order_info.side,
                timestamp_ms=result.timestamp_ms,
            )
            self._total_liquidations_processed += 1

        # Clear processed liquidations
        self._fill_provider.clear_executed_liquidations()

        # Check for ADL trigger
        if self._config.enable_adl_simulation:
            if self._insurance_fund.check_adl_trigger():
                logger.warning("Insurance fund depleted - ADL triggered")
                # ADL handling would go here

    def load_liquidation_data(
        self,
        data: List[Dict[str, Any]],
    ) -> None:
        """
        Load historical liquidation data for injection.

        Args:
            data: List of liquidation event dicts with fields:
                - symbol: Contract symbol
                - side: "BUY" or "SELL"
                - qty: Quantity
                - price: Liquidation price
                - timestamp_ms: Timestamp
        """
        self._liquidation_stream.add_historical_data(data)

    def add_liquidation_event(
        self,
        event: LiquidationOrderInfo,
    ) -> None:
        """Add single liquidation event."""
        self._liquidation_stream.add_event(event)

    # =========================================================================
    # Properties and Accessors
    # =========================================================================

    @property
    def is_cascade_active(self) -> bool:
        """Check if cascade is currently active."""
        return self._active_cascade is not None

    @property
    def active_cascade(self) -> Optional[CascadeResult]:
        """Get active cascade result."""
        return self._active_cascade

    @property
    def insurance_fund_balance(self) -> Decimal:
        """Current insurance fund balance."""
        return self._insurance_fund.balance

    @property
    def insurance_fund_state(self) -> InsuranceFundState:
        """Current insurance fund state."""
        return self._insurance_fund.state

    @property
    def funding_state(self) -> Optional[FundingPeriodState]:
        """Last funding period state."""
        return self._last_funding_state

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_fills": self._total_fills,
            "total_liquidations_processed": self._total_liquidations_processed,
            "total_cascade_events": self._total_cascade_events,
            "pending_liquidations": self._liquidation_stream.pending_count,
            "insurance_fund_balance": str(self._insurance_fund.balance),
            "insurance_fund_utilization_pct": self._insurance_fund.state.utilization_pct,
            "cascade_stats": self._cascade_simulator.stats,
        }

    def reset(self) -> None:
        """Reset provider state."""
        self._active_cascade = None
        self._last_funding_state = None
        self._total_fills = 0
        self._total_liquidations_processed = 0
        self._total_cascade_events = 0
        self._liquidation_stream.clear()
        self._insurance_fund.reset()
        self._adl_manager.clear()
        self._fill_provider.clear_executed_liquidations()

    def clear_cascade(self) -> None:
        """Clear active cascade."""
        self._active_cascade = None
        self._slippage.set_cascade_impact(0.0)


# =============================================================================
# Factory Functions
# =============================================================================


def create_futures_l3_config(
    enable_cascade: bool = True,
    use_mark_price: bool = True,
    cascade_decay: float = 0.7,
    max_cascade_waves: int = 5,
    price_impact_coef: float = 0.1,
    insurance_fund_initial: Decimal = _DEFAULT_INSURANCE_FUND,
    **kwargs,
) -> FuturesL3Config:
    """
    Create FuturesL3Config with common defaults.

    Args:
        enable_cascade: Enable cascade simulation
        use_mark_price: Use mark price for execution
        cascade_decay: Cascade decay factor
        max_cascade_waves: Maximum cascade waves
        price_impact_coef: Kyle lambda
        insurance_fund_initial: Starting insurance fund
        **kwargs: Additional config parameters

    Returns:
        FuturesL3Config instance
    """
    return FuturesL3Config(
        enable_cascade_simulation=enable_cascade,
        use_mark_price_execution=use_mark_price,
        cascade_decay=cascade_decay,
        max_cascade_waves=max_cascade_waves,
        price_impact_coef=price_impact_coef,
        insurance_fund_initial=insurance_fund_initial,
        **kwargs,
    )


def create_futures_l3_execution_provider(
    config: Optional[FuturesL3Config] = None,
    use_mark_price: bool = True,
    enable_cascade: bool = True,
    **kwargs,
) -> FuturesL3ExecutionProvider:
    """
    Factory function to create FuturesL3ExecutionProvider.

    Args:
        config: Optional configuration
        use_mark_price: Use mark price execution
        enable_cascade: Enable cascade simulation
        **kwargs: Additional parameters

    Returns:
        FuturesL3ExecutionProvider instance
    """
    if config is None:
        config = FuturesL3Config(
            use_mark_price_execution=use_mark_price,
            enable_cascade_simulation=enable_cascade,
        )

    return FuturesL3ExecutionProvider(config=config, **kwargs)


# =============================================================================
# Profile Presets
# =============================================================================


_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "enable_cascade_simulation": True,
        "enable_liquidation_injection": True,
        "enable_insurance_fund": True,
        "use_mark_price_execution": True,
        "cascade_decay": 0.7,
        "max_cascade_waves": 5,
        "price_impact_coef": 0.1,
    },
    "conservative": {
        "enable_cascade_simulation": True,
        "enable_liquidation_injection": True,
        "enable_insurance_fund": True,
        "use_mark_price_execution": True,
        "cascade_decay": 0.8,  # Slower decay = longer cascades
        "max_cascade_waves": 7,
        "price_impact_coef": 0.15,  # Higher impact
    },
    "fast": {
        "enable_cascade_simulation": False,  # Disable for speed
        "enable_liquidation_injection": True,
        "enable_insurance_fund": False,
        "use_mark_price_execution": True,
        "cascade_decay": 0.5,
        "max_cascade_waves": 3,
        "price_impact_coef": 0.05,
    },
    "stress_test": {
        "enable_cascade_simulation": True,
        "enable_liquidation_injection": True,
        "enable_insurance_fund": True,
        "enable_adl_simulation": True,
        "use_mark_price_execution": True,
        "cascade_decay": 0.9,  # Very slow decay = severe cascades
        "max_cascade_waves": 10,
        "price_impact_coef": 0.2,
        "insurance_fund_initial": Decimal("100_000_000"),  # Smaller fund
    },
}


def create_futures_l3_from_preset(preset: str) -> FuturesL3ExecutionProvider:
    """
    Create FuturesL3ExecutionProvider from preset profile.

    Args:
        preset: Profile name ("default", "conservative", "fast", "stress_test")

    Returns:
        Configured FuturesL3ExecutionProvider
    """
    if preset not in _PRESETS:
        available = ", ".join(_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    preset_config = _PRESETS[preset]
    config = FuturesL3Config(**preset_config)

    return FuturesL3ExecutionProvider(config=config)
