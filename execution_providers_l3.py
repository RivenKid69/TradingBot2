# -*- coding: utf-8 -*-
"""
L3 Execution Provider with Full LOB Simulation.

This module provides a high-fidelity execution simulation using the complete
L3 LOB infrastructure. It integrates all LOB components:

- LOB state management (OrderBook, PriceLevel, LimitOrder)
- FIFO matching engine with self-trade prevention
- Queue position tracking (MBP/MBO estimation)
- Fill probability models (Poisson, Queue-Reactive)
- Queue value computation (Moallemi & Yuan)
- Market impact models (Kyle, Almgren-Chriss, Gatheral)
- Latency simulation (feed/order/exchange/fill)
- Hidden liquidity detection (iceberg orders)
- Dark pool simulation (optional)

Architecture:
    L3ExecutionProvider implements the ExecutionProvider protocol from
    execution_providers.py, allowing seamless integration with existing
    backtesting and training pipelines.

    The key difference from L2 is that L3 uses actual order book state
    and queue position tracking rather than statistical models.

Design Principles:
    1. Protocol-compatible with L2ExecutionProvider
    2. Backward compatible - crypto paths unaffected
    3. Configurable fidelity (can disable subsystems)
    4. Research-grade accuracy with deployment-ready performance

Stage 7 of L3 LOB Simulation (v7.0)

References:
    - CME Globex Matching Algorithm
    - Erik Rigtorp: Queue Position Estimation
    - Moallemi & Yuan (2017): Queue Position Valuation
    - Almgren & Chriss (2001): Optimal Execution
    - Gatheral (2010): Transient Impact
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
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
    StatisticalSlippageProvider,
    OHLCVFillProvider,
    CryptoFeeProvider,
    EquityFeeProvider,
    ZeroFeeProvider,
)

# Import LOB components
from lob import (
    # Core structures
    Side as LOBSide,
    OrderType as LOBOrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill as LOBFill,
    Trade,
    # Matching
    MatchingEngine,
    MatchResult,
    MatchType,
    STPAction,
    # Queue tracking
    QueuePositionTracker,
    QueueState,
    FillProbability,
    PositionEstimationMethod,
    # Order management
    OrderManager,
    ManagedOrder,
    TimeInForce as LOBTimeInForce,
    # Fill probability
    FillProbabilityModel,
    QueueReactiveModel,
    AnalyticalPoissonModel,
    LOBState,
    FillProbabilityResult,
    create_fill_probability_model,
    # Queue value
    QueueValueModel,
    QueueValueResult,
    create_queue_value_model,
    # Market impact
    MarketImpactModel,
    AlmgrenChrissModel,
    GatheralModel,
    KyleLambdaModel,
    ImpactParameters,
    ImpactResult,
    create_impact_model,
    # Impact effects
    ImpactEffects,
    LOBImpactSimulator,
    QuoteShiftResult,
    create_lob_impact_simulator,
    # Latency
    LatencyModel,
    LatencyProfile,
    create_latency_model,
    # Event scheduler
    EventScheduler,
    SimulationClock,
    create_event_scheduler,
    # Hidden liquidity
    IcebergDetector,
    HiddenLiquidityEstimator,
    create_iceberg_detector,
    create_hidden_liquidity_estimator,
    # Dark pools
    DarkPoolSimulator,
    DarkPoolFill,
    create_dark_pool_simulator,
    create_default_dark_pool_simulator,
)

# Import L3 configuration
from lob.config import (
    L3ExecutionConfig,
    LatencyConfig,
    FillProbabilityConfig,
    QueueValueConfig,
    MarketImpactConfig,
    HiddenLiquidityConfig,
    DarkPoolsConfig,
    LatencyProfileType,
    FillProbabilityModelType,
    ImpactModelType,
    create_l3_config,
    latency_config_to_dataclass,
    impact_config_to_parameters,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_MIN_PRICE = 1e-8
_MAX_SLIPPAGE_BPS = 500.0
_DEFAULT_ADV = 1_000_000.0
_DEFAULT_VOLATILITY = 0.02
_TICK_SIZE_EQUITY = 0.01
_TICK_SIZE_CRYPTO = 1e-8


# =============================================================================
# L3 Slippage Provider
# =============================================================================

class L3SlippageProvider:
    """
    L3: LOB walk-through slippage provider.

    Simulates walking through the order book to compute actual execution
    price based on available liquidity at each price level.

    Unlike L2's statistical model, this uses actual LOB depth when available.

    Attributes:
        asset_class: Asset class for defaults
        impact_model: Market impact model for large orders
        config: L3 execution configuration
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.EQUITY,
        impact_model: Optional[MarketImpactModel] = None,
        config: Optional[L3ExecutionConfig] = None,
    ) -> None:
        """
        Initialize L3 slippage provider.

        Args:
            asset_class: Asset class for defaults
            impact_model: Market impact model (default: Almgren-Chriss)
            config: L3 execution configuration
        """
        self._asset_class = asset_class
        self._config = config or L3ExecutionConfig.for_equity()

        # Create impact model
        if impact_model is not None:
            self._impact_model = impact_model
        elif self._config.market_impact.enabled:
            impact_params = impact_config_to_parameters(self._config.market_impact)
            model_map = {
                ImpactModelType.KYLE: "kyle",
                ImpactModelType.ALMGREN_CHRISS: "almgren_chriss",
                ImpactModelType.GATHERAL: "gatheral",
            }
            model_name = model_map.get(
                self._config.market_impact.model, "almgren_chriss"
            )
            self._impact_model = create_impact_model(model_name, params=impact_params)
        else:
            self._impact_model = None

        # Fallback to statistical model
        self._fallback = StatisticalSlippageProvider(
            impact_coef=0.05 if asset_class == AssetClass.EQUITY else 0.1,
            spread_bps=2.0 if asset_class == AssetClass.EQUITY else 5.0,
        )

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute slippage from LOB walk-through.

        If LOB depth is available, walks through the book to compute
        actual execution price. Otherwise falls back to statistical model.

        Args:
            order: Order to execute
            market: Market state with optional LOB depth
            participation_ratio: Order size / ADV

        Returns:
            Slippage in basis points
        """
        # Check if we have LOB depth
        has_lob = (
            market.bid_depth is not None
            and market.ask_depth is not None
            and len(market.bid_depth) > 0
            and len(market.ask_depth) > 0
        )

        if not has_lob:
            # Fall back to statistical model
            return self._fallback.compute_slippage_bps(
                order, market, participation_ratio
            )

        # Get reference price
        mid_price = market.get_mid_price()
        if mid_price is None or mid_price <= 0:
            return self._fallback.compute_slippage_bps(
                order, market, participation_ratio
            )

        # Walk through the book
        is_buy = str(order.side).upper() == "BUY"
        depth = market.ask_depth if is_buy else market.bid_depth

        # Simulate execution
        remaining_qty = order.qty
        total_cost = 0.0

        for price, size in depth:
            if remaining_qty <= 0:
                break
            fill_qty = min(remaining_qty, size)
            total_cost += fill_qty * price
            remaining_qty -= fill_qty

        # If we couldn't fill completely, use worst price for remainder
        if remaining_qty > 0 and len(depth) > 0:
            worst_price = depth[-1][0]
            total_cost += remaining_qty * worst_price

        # Compute average execution price
        avg_price = total_cost / order.qty if order.qty > 0 else mid_price

        # Compute slippage vs mid price
        if is_buy:
            slippage_bps = (avg_price - mid_price) / mid_price * 10000.0
        else:
            slippage_bps = (mid_price - avg_price) / mid_price * 10000.0

        # Add market impact for large orders
        if self._impact_model is not None and participation_ratio > 0.001:
            adv = market.adv or _DEFAULT_ADV
            volatility = market.volatility or _DEFAULT_VOLATILITY
            notional = order.qty * mid_price

            impact_result = self._impact_model.compute_total_impact(
                order_qty=notional,
                adv=adv,
                volatility=volatility,
                mid_price=mid_price,
            )
            slippage_bps += impact_result.temporary_impact_bps

        # Ensure non-negative and capped
        return max(0.0, min(slippage_bps, _MAX_SLIPPAGE_BPS))

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        volatility: float = _DEFAULT_VOLATILITY,
    ) -> Dict[str, float]:
        """
        Pre-trade impact cost estimation.

        Args:
            notional: Planned trade notional
            adv: Average daily volume
            volatility: Expected volatility

        Returns:
            Dict with cost breakdown
        """
        if self._impact_model is None:
            participation = notional / adv if adv > 0 else 0.01
            return {
                "participation": participation,
                "temporary_impact_bps": 0.0,
                "permanent_impact_bps": 0.0,
                "total_impact_bps": 0.0,
            }

        result = self._impact_model.compute_total_impact(
            order_qty=notional,
            adv=adv,
            volatility=volatility,
            mid_price=100.0,  # Normalized
        )

        return {
            "participation": notional / adv if adv > 0 else 0.0,
            "temporary_impact_bps": result.temporary_impact_bps,
            "permanent_impact_bps": result.permanent_impact_bps,
            "total_impact_bps": result.total_impact_bps,
            "impact_cost": result.impact_cost,
        }


# =============================================================================
# L3 Fill Provider
# =============================================================================

class L3FillProvider:
    """
    L3: Matching engine fill provider.

    Uses FIFO matching engine with queue position tracking to determine
    if and when limit orders fill.

    Features:
        - FIFO price-time priority matching
        - Queue position tracking (MBP/MBO estimation)
        - Fill probability estimation
        - Market impact application
        - Hidden liquidity detection

    Attributes:
        matching_engine: FIFO matching engine
        queue_tracker: Queue position tracker
        order_manager: Order lifecycle manager
        fill_prob_model: Fill probability model
        slippage_provider: Slippage provider
        fee_provider: Fee provider
        config: L3 execution configuration
    """

    def __init__(
        self,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        config: Optional[L3ExecutionConfig] = None,
        asset_class: AssetClass = AssetClass.EQUITY,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize L3 fill provider.

        Args:
            slippage_provider: Slippage provider (default: L3SlippageProvider)
            fee_provider: Fee provider (default: asset-class specific)
            config: L3 execution configuration
            asset_class: Asset class for defaults
            seed: Random seed for reproducible backtests (None = non-deterministic)
            **kwargs: Additional configuration
        """
        self._asset_class = asset_class
        self._config = config or L3ExecutionConfig.for_equity()

        # Seedable RNG for reproducible backtests
        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Initialize providers
        if slippage_provider is not None:
            self.slippage = slippage_provider
        else:
            self.slippage = L3SlippageProvider(
                asset_class=asset_class,
                config=self._config,
            )

        if fee_provider is not None:
            self.fees = fee_provider
        else:
            if asset_class == AssetClass.EQUITY:
                self.fees = EquityFeeProvider()
            else:
                self.fees = CryptoFeeProvider()

        # Initialize LOB components
        self._matching_engine = MatchingEngine()
        self._queue_tracker = QueuePositionTracker()
        self._order_manager: Dict[str, OrderManager] = {}  # Per-symbol

        # Fill probability model
        if self._config.fill_probability.enabled:
            fp_config = self._config.fill_probability
            model_map = {
                FillProbabilityModelType.POISSON: "poisson",
                FillProbabilityModelType.QUEUE_REACTIVE: "queue_reactive",
                FillProbabilityModelType.HISTORICAL: "historical",
            }
            model_name = model_map.get(fp_config.model, "queue_reactive")
            self._fill_prob_model = create_fill_probability_model(
                model_name,
                base_rate=fp_config.base_rate,
            )
        else:
            self._fill_prob_model = None

        # Market impact simulator
        if self._config.market_impact.enabled:
            impact_params = impact_config_to_parameters(self._config.market_impact)
            model_map = {
                ImpactModelType.ALMGREN_CHRISS: "almgren_chriss",
                ImpactModelType.GATHERAL: "gatheral",
                ImpactModelType.KYLE: "kyle",
            }
            model_name = model_map.get(
                self._config.market_impact.model, "almgren_chriss"
            )
            impact_model = create_impact_model(model_name, params=impact_params)
            self._impact_simulator = create_lob_impact_simulator(
                impact_model=impact_model
            )
        else:
            self._impact_simulator = None

        # Iceberg detector
        if self._config.hidden_liquidity.enabled:
            iceberg_config = self._config.hidden_liquidity.iceberg
            self._iceberg_detector = create_iceberg_detector(
                min_refills_to_confirm=iceberg_config.min_refills_to_confirm,
                lookback_window_sec=iceberg_config.lookback_window_sec,
            )
        else:
            self._iceberg_detector = None

        # Dark pool simulator
        if self._config.dark_pools.enabled:
            self._dark_pool = create_default_dark_pool_simulator()
        else:
            self._dark_pool = None

        # Latency model
        if self._config.latency.enabled:
            profile_map = {
                LatencyProfileType.COLOCATED: "colocated",
                LatencyProfileType.PROXIMITY: "proximity",
                LatencyProfileType.RETAIL: "retail",
                LatencyProfileType.INSTITUTIONAL: "institutional",
            }
            profile_name = profile_map.get(
                self._config.latency.profile, "institutional"
            )
            self._latency_model = create_latency_model(profile_name)
        else:
            self._latency_model = None

        # Fallback to OHLCV provider
        self._fallback = OHLCVFillProvider(
            slippage_provider=StatisticalSlippageProvider(
                impact_coef=0.05 if asset_class == AssetClass.EQUITY else 0.1,
                spread_bps=2.0 if asset_class == AssetClass.EQUITY else 5.0,
            ),
            fee_provider=self.fees,
        )

        # Tracking
        self._order_counter = 0

    def _get_order_manager(self, symbol: str) -> OrderManager:
        """Get or create order manager for symbol."""
        if symbol not in self._order_manager:
            tick_size = _TICK_SIZE_EQUITY if self._asset_class == AssetClass.EQUITY else _TICK_SIZE_CRYPTO
            self._order_manager[symbol] = OrderManager(
                symbol=symbol,
                tick_size=tick_size,
            )
        return self._order_manager[symbol]

    def _convert_order_to_lob(self, order: Order, timestamp_ns: int) -> LimitOrder:
        """Convert execution_providers.Order to LOB LimitOrder."""
        self._order_counter += 1
        order_id = f"L3_{self._order_counter}_{timestamp_ns}"

        side = LOBSide.BUY if order.is_buy else LOBSide.SELL
        price = order.limit_price if order.limit_price else 0.0

        return LimitOrder(
            order_id=order_id,
            price=price,
            qty=order.qty,
            remaining_qty=order.qty,
            timestamp_ns=timestamp_ns,
            side=side,
            is_own=True,  # Mark as our order for tracking
        )

    def _convert_lob_side(self, side: str) -> LOBSide:
        """Convert string side to LOB Side."""
        return LOBSide.BUY if str(side).upper() == "BUY" else LOBSide.SELL

    def _build_lob_state(
        self,
        market: MarketState,
        bar: BarData,
    ) -> LOBState:
        """Build LOBState from MarketState and BarData."""
        mid = market.get_mid_price() or bar.close
        spread = market.get_spread_bps() or 5.0
        volatility = market.volatility or _DEFAULT_VOLATILITY

        # Calculate depth from market state if available
        bid_depth = 0.0
        ask_depth = 0.0
        if market.bid_depth:
            bid_depth = sum(size for _, size in market.bid_depth)
        if market.ask_depth:
            ask_depth = sum(size for _, size in market.ask_depth)

        # Imbalance
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Volume rate from bar
        timeframe_ms = bar.timeframe_ms or 3600_000  # Default 1 hour
        timeframe_sec = timeframe_ms / 1000.0
        volume_rate = (bar.volume or 0.0) / timeframe_sec if timeframe_sec > 0 else 100.0

        return LOBState(
            timestamp_ns=market.timestamp * 1_000_000 if market.timestamp else int(time.time_ns()),
            mid_price=mid,
            spread=mid * spread / 10000.0 if mid > 0 else 0.0,
            spread_bps=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance,
            volatility=volatility,
            volume_rate=volume_rate,
        )

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt to fill order using L3 LOB simulation.

        Process:
        1. Check if L3 data available, else fall back to L2
        2. For MARKET orders: immediate fill at market
        3. For LIMIT orders:
           a. Check if crosses spread (immediate TAKER fill)
           b. Otherwise, estimate queue position and fill probability
           c. Simulate fill based on bar volume and queue position

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data (OHLCV)

        Returns:
            Fill if order executed, None if not filled
        """
        # Check if we have sufficient L3 data
        has_lob = (
            market.bid_depth is not None
            and market.ask_depth is not None
            and len(market.bid_depth or []) > 0
            and len(market.ask_depth or []) > 0
        )

        # For MARKET orders, always fill
        order_type = str(order.order_type).upper()
        is_market = order_type == "MARKET"

        if is_market:
            return self._fill_market_order(order, market, bar, has_lob)
        else:
            return self._fill_limit_order(order, market, bar, has_lob)

    def _fill_market_order(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        has_lob: bool,
    ) -> Optional[Fill]:
        """Fill market order (always fills)."""
        mid = market.get_mid_price() or bar.close
        if mid is None or mid <= 0:
            mid = bar.close

        is_buy = order.is_buy
        spread_bps = market.get_spread_bps() or 5.0
        half_spread = mid * spread_bps / 20000.0  # Half spread in price terms

        # Base price from quote
        if is_buy:
            base_price = market.ask if market.ask else mid + half_spread
        else:
            base_price = market.bid if market.bid else mid - half_spread

        # Compute slippage
        adv = market.adv or _DEFAULT_ADV
        participation = (order.qty * base_price) / adv if adv > 0 else 0.01
        slippage_bps = self.slippage.compute_slippage_bps(order, market, participation)

        # Apply slippage to get fill price
        slippage_mult = slippage_bps / 10000.0
        if is_buy:
            fill_price = base_price * (1.0 + slippage_mult)
        else:
            fill_price = base_price * (1.0 - slippage_mult)

        # Ensure fill price is within bar range
        fill_price = max(bar.low, min(bar.high, fill_price))

        # Compute fee
        notional = fill_price * order.qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=str(order.side),
            liquidity="taker",
            qty=order.qty,
        )

        # Apply latency if enabled
        timestamp = market.timestamp
        if self._latency_model is not None:
            latency_ns = self._latency_model.sample_round_trip()
            timestamp = (timestamp or int(time.time() * 1000)) + latency_ns // 1_000_000

        return Fill(
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity="taker",
            timestamp=timestamp,
            notional=notional,
            metadata={
                "fill_type": "market",
                "has_lob": has_lob,
                "participation": participation,
            },
        )

    def _fill_limit_order(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        has_lob: bool,
    ) -> Optional[Fill]:
        """Fill limit order (may not fill)."""
        if order.limit_price is None:
            logger.warning("LIMIT order without limit_price, treating as MARKET")
            return self._fill_market_order(order, market, bar, has_lob)

        limit_price = order.limit_price
        is_buy = order.is_buy
        mid = market.get_mid_price() or bar.close

        # Get quotes
        ask = market.ask if market.ask and math.isfinite(market.ask) else mid * 1.0005
        bid = market.bid if market.bid and math.isfinite(market.bid) else mid * 0.9995

        # Check for immediate fill (crosses spread) - TAKER
        crosses_spread = (is_buy and limit_price >= ask) or (not is_buy and limit_price <= bid)

        if crosses_spread:
            # Immediate fill at limit or better
            if is_buy:
                fill_price = min(limit_price, ask)
            else:
                fill_price = max(limit_price, bid)

            # Compute slippage from mid
            if mid > 0:
                slippage_bps = abs(fill_price - mid) / mid * 10000.0
            else:
                slippage_bps = 0.0

            # Taker fee
            notional = fill_price * order.qty
            fee = self.fees.compute_fee(
                notional=notional,
                side=str(order.side),
                liquidity="taker",
                qty=order.qty,
            )

            return Fill(
                price=fill_price,
                qty=order.qty,
                fee=fee,
                slippage_bps=slippage_bps,
                liquidity="taker",
                timestamp=market.timestamp,
                notional=notional,
                metadata={
                    "fill_type": "limit_aggressive",
                    "has_lob": has_lob,
                    "crosses_spread": True,
                },
            )

        # Passive limit order - check if bar range touches limit price
        bar_low = bar.low
        bar_high = bar.high

        # Price tolerance for fill determination
        tolerance = limit_price * 1e-6

        # Check if limit price touched during bar
        if is_buy:
            fills = bar_low <= limit_price + tolerance
        else:
            fills = bar_high >= limit_price - tolerance

        if not fills:
            # Limit not touched, no fill
            return None

        # Limit was touched - determine if we fill based on queue position
        fill_price = limit_price  # Maker fills at limit

        # If fill probability model enabled, use it
        if self._fill_prob_model is not None and has_lob:
            lob_state = self._build_lob_state(market, bar)

            # Estimate queue position (pessimistic)
            if is_buy and market.bid_depth:
                # For buy limit below best ask, queue behind existing bids at same level
                qty_ahead = sum(
                    size for price, size in market.bid_depth if price >= limit_price
                )
            elif not is_buy and market.ask_depth:
                qty_ahead = sum(
                    size for price, size in market.ask_depth if price <= limit_price
                )
            else:
                qty_ahead = 0.0

            # Compute fill probability
            timeframe_sec = (bar.timeframe_ms or 3600_000) / 1000.0
            prob_result = self._fill_prob_model.compute_fill_probability(
                queue_position=max(1, int(qty_ahead / order.qty) + 1),
                qty_ahead=qty_ahead,
                order_qty=order.qty,
                time_horizon_sec=timeframe_sec,
                market_state=lob_state,
            )

            # Stochastic fill based on probability (uses seedable RNG)
            if self._rng.random() > prob_result.prob_fill:
                # Did not fill this bar
                return None

        # Fill at maker price
        notional = fill_price * order.qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=str(order.side),
            liquidity="maker",
            qty=order.qty,
        )

        # Slippage is typically positive for maker (better than taker)
        if mid > 0:
            if is_buy:
                slippage_bps = (mid - fill_price) / mid * 10000.0  # Positive if below mid
            else:
                slippage_bps = (fill_price - mid) / mid * 10000.0  # Positive if above mid
        else:
            slippage_bps = 0.0

        return Fill(
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage_bps=max(0.0, -slippage_bps),  # Report slippage as cost (positive)
            liquidity="maker",
            timestamp=market.timestamp,
            notional=notional,
            metadata={
                "fill_type": "limit_passive",
                "has_lob": has_lob,
                "queue_based": self._fill_prob_model is not None,
            },
        )

    def get_fill_probability(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        time_horizon_sec: float = 60.0,
    ) -> Optional[FillProbabilityResult]:
        """
        Estimate fill probability for a limit order.

        Args:
            order: Limit order to evaluate
            market: Current market state
            bar: Current bar data
            time_horizon_sec: Time horizon for probability estimation

        Returns:
            FillProbabilityResult or None if not available
        """
        if self._fill_prob_model is None:
            return None

        if order.limit_price is None:
            return None

        lob_state = self._build_lob_state(market, bar)
        is_buy = order.is_buy

        # Estimate queue position
        if is_buy and market.bid_depth:
            qty_ahead = sum(
                size for price, size in market.bid_depth if price >= order.limit_price
            )
        elif not is_buy and market.ask_depth:
            qty_ahead = sum(
                size for price, size in market.ask_depth if price <= order.limit_price
            )
        else:
            qty_ahead = 0.0

        return self._fill_prob_model.compute_fill_probability(
            queue_position=max(1, int(qty_ahead / order.qty) + 1),
            qty_ahead=qty_ahead,
            order_qty=order.qty,
            time_horizon_sec=time_horizon_sec,
            market_state=lob_state,
        )


# =============================================================================
# L3 Execution Provider
# =============================================================================

class L3ExecutionProvider:
    """
    Full L3 execution provider with LOB simulation.

    Combines all L3 components into a unified execution interface that
    implements the ExecutionProvider protocol.

    Components:
        - L3SlippageProvider: LOB walk-through slippage
        - L3FillProvider: FIFO matching with queue tracking
        - Market impact models (Almgren-Chriss, Gatheral)
        - Latency simulation
        - Hidden liquidity detection
        - Dark pool routing (optional)

    Attributes:
        asset_class: Asset class (EQUITY or CRYPTO)
        config: L3 execution configuration
        slippage: L3 slippage provider
        fill: L3 fill provider
        fees: Fee provider
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.EQUITY,
        config: Optional[L3ExecutionConfig] = None,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize L3 execution provider.

        Args:
            asset_class: Asset class (EQUITY, CRYPTO)
            config: L3 execution configuration
            slippage_provider: Custom slippage provider (default: L3SlippageProvider)
            fee_provider: Custom fee provider (default: asset-class specific)
            **kwargs: Additional configuration
        """
        self._asset_class = asset_class

        # Load configuration
        if config is not None:
            self._config = config
        elif "config_path" in kwargs:
            self._config = L3ExecutionConfig.from_yaml(kwargs["config_path"])
        else:
            if asset_class == AssetClass.EQUITY:
                self._config = L3ExecutionConfig.for_equity()
            else:
                self._config = L3ExecutionConfig.for_crypto()

        # Initialize slippage provider
        if slippage_provider is not None:
            self.slippage = slippage_provider
        else:
            self.slippage = L3SlippageProvider(
                asset_class=asset_class,
                config=self._config,
            )

        # Initialize fee provider
        if fee_provider is not None:
            self.fees = fee_provider
        else:
            if asset_class == AssetClass.EQUITY:
                self.fees = EquityFeeProvider()
            else:
                self.fees = CryptoFeeProvider()

        # Initialize fill provider
        self.fill = L3FillProvider(
            slippage_provider=self.slippage,
            fee_provider=self.fees,
            config=self._config,
            asset_class=asset_class,
            **kwargs,
        )

        # Latency model for round-trip timing
        if self._config.latency.enabled:
            profile_map = {
                LatencyProfileType.COLOCATED: "colocated",
                LatencyProfileType.PROXIMITY: "proximity",
                LatencyProfileType.RETAIL: "retail",
                LatencyProfileType.INSTITUTIONAL: "institutional",
            }
            profile_name = profile_map.get(
                self._config.latency.profile, "institutional"
            )
            self._latency_model = create_latency_model(profile_name)
        else:
            self._latency_model = None

        # Dark pool simulator
        if self._config.dark_pools.enabled:
            self._dark_pool = create_default_dark_pool_simulator()
        else:
            self._dark_pool = None

        # Statistics
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "taker_fills": 0,
            "maker_fills": 0,
            "dark_pool_fills": 0,
        }

        # Counter for unique dark pool order IDs
        self._dark_pool_order_counter = 0

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        return self._asset_class

    @property
    def config(self) -> L3ExecutionConfig:
        """L3 execution configuration."""
        return self._config

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute an order with full L3 simulation.

        Process:
        1. Apply feed latency to market state (if enabled)
        2. Optionally try dark pool fill first
        3. Execute via L3 fill provider (matching engine + queue)
        4. Apply market impact to LOB state
        5. Return fill with correct timing

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data

        Returns:
            Fill with slippage and fees, or None if not filled
        """
        self._stats["total_orders"] += 1

        # Try dark pool first if enabled and order is large enough
        if self._dark_pool is not None and self._config.dark_pools.enabled:
            dark_fill = self._try_dark_pool_fill(order, market, bar)
            if dark_fill is not None:
                self._stats["filled_orders"] += 1
                self._stats["dark_pool_fills"] += 1
                return dark_fill

        # Execute via L3 fill provider
        fill = self.fill.try_fill(order, market, bar)

        if fill is not None:
            self._stats["filled_orders"] += 1
            if fill.liquidity == "maker":
                self._stats["maker_fills"] += 1
            else:
                self._stats["taker_fills"] += 1

        return fill

    def _try_dark_pool_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """Try to fill order in dark pool."""
        if self._dark_pool is None:
            return None

        mid = market.get_mid_price() or bar.close
        if mid is None or mid <= 0:
            return None

        spread = market.get_spread_bps() or 5.0
        adv = market.adv or _DEFAULT_ADV
        volatility = market.volatility or _DEFAULT_VOLATILITY

        # Convert to LOB order for dark pool (unique order_id)
        timestamp_ns = (market.timestamp or int(time.time() * 1000)) * 1_000_000
        self._dark_pool_order_counter += 1
        lob_order = LimitOrder(
            order_id=f"dark_{timestamp_ns}_{self._dark_pool_order_counter}",
            price=order.limit_price or mid,
            qty=order.qty,
            remaining_qty=order.qty,
            timestamp_ns=timestamp_ns,
            side=LOBSide.BUY if order.is_buy else LOBSide.SELL,
        )

        # Try dark pool fill
        dark_fill = self._dark_pool.attempt_dark_fill(
            order=lob_order,
            lit_mid_price=mid,
            lit_spread=mid * spread / 10000.0,
            adv=adv,
            volatility=volatility,
        )

        if dark_fill is None or not dark_fill.is_filled:
            return None

        # Convert dark pool fill to execution_providers.Fill
        notional = dark_fill.fill_price * dark_fill.filled_qty

        # Dark pool typically has zero explicit fee (built into spread)
        fee = self.fees.compute_fee(
            notional=notional,
            side=str(order.side),
            liquidity="taker",
            qty=dark_fill.filled_qty,
        )

        # Slippage is typically zero for dark pools (mid-price execution)
        slippage_bps = abs(dark_fill.fill_price - mid) / mid * 10000.0 if mid > 0 else 0.0

        return Fill(
            price=dark_fill.fill_price,
            qty=dark_fill.filled_qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity="taker",  # Dark pools are typically taker-style
            timestamp=market.timestamp,
            notional=notional,
            metadata={
                "fill_type": "dark_pool",
                "venue_id": dark_fill.venue_id,
                "info_leakage": dark_fill.info_leakage is not None,
            },
        )

    def estimate_execution_cost(
        self,
        notional: float,
        adv: float,
        side: str = "BUY",
        volatility: float = _DEFAULT_VOLATILITY,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Args:
            notional: Planned trade notional
            adv: Average daily volume
            side: Trade side
            volatility: Expected volatility

        Returns:
            Dict with cost breakdown
        """
        # Get impact estimate
        if isinstance(self.slippage, L3SlippageProvider):
            impact_est = self.slippage.estimate_impact_cost(notional, adv, volatility)
        else:
            participation = notional / adv if adv > 0 else 0.01
            impact_est = {
                "participation": participation,
                "total_impact_bps": 0.0,
            }

        # Fee estimate
        fee = self.fees.compute_fee(
            notional=notional,
            side=side,
            liquidity="taker",
            qty=notional / 100.0,
        )

        # Combine
        slippage_bps = impact_est.get("total_impact_bps", 0.0)
        slippage_cost = notional * slippage_bps / 10000.0

        return {
            "participation": impact_est.get("participation", 0.0),
            "slippage_bps": slippage_bps,
            "slippage_cost": slippage_cost,
            "fee": fee,
            "total_cost": slippage_cost + fee,
            "total_bps": (slippage_cost + fee) / notional * 10000.0 if notional > 0 else 0.0,
            "temporary_impact_bps": impact_est.get("temporary_impact_bps", 0.0),
            "permanent_impact_bps": impact_est.get("permanent_impact_bps", 0.0),
        }

    def get_fill_probability(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        time_horizon_sec: float = 60.0,
    ) -> Optional[FillProbabilityResult]:
        """
        Estimate fill probability for a limit order.

        Args:
            order: Limit order to evaluate
            market: Current market state
            bar: Current bar data
            time_horizon_sec: Time horizon for probability estimation

        Returns:
            FillProbabilityResult or None if not available
        """
        return self.fill.get_fill_probability(order, market, bar, time_horizon_sec)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self._stats["total_orders"]
        filled = self._stats["filled_orders"]

        return {
            **self._stats,
            "fill_rate": filled / total if total > 0 else 0.0,
            "maker_rate": self._stats["maker_fills"] / filled if filled > 0 else 0.0,
            "taker_rate": self._stats["taker_fills"] / filled if filled > 0 else 0.0,
            "dark_pool_rate": self._stats["dark_pool_fills"] / filled if filled > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "taker_fills": 0,
            "maker_fills": 0,
            "dark_pool_fills": 0,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_l3_execution_provider(
    asset_class: AssetClass = AssetClass.EQUITY,
    config: Optional[L3ExecutionConfig] = None,
    **kwargs: Any,
) -> L3ExecutionProvider:
    """
    Factory function to create L3 execution provider.

    Args:
        asset_class: Asset class (EQUITY, CRYPTO)
        config: L3 execution configuration
        **kwargs: Additional configuration

    Returns:
        L3ExecutionProvider instance
    """
    return L3ExecutionProvider(
        asset_class=asset_class,
        config=config,
        **kwargs,
    )


def create_l3_slippage_provider(
    asset_class: AssetClass = AssetClass.EQUITY,
    config: Optional[L3ExecutionConfig] = None,
    **kwargs: Any,
) -> L3SlippageProvider:
    """
    Factory function to create L3 slippage provider.

    Args:
        asset_class: Asset class
        config: L3 execution configuration
        **kwargs: Additional configuration

    Returns:
        L3SlippageProvider instance
    """
    return L3SlippageProvider(
        asset_class=asset_class,
        config=config,
        **kwargs,
    )


def create_l3_fill_provider(
    asset_class: AssetClass = AssetClass.EQUITY,
    config: Optional[L3ExecutionConfig] = None,
    slippage_provider: Optional[SlippageProvider] = None,
    fee_provider: Optional[FeeProvider] = None,
    **kwargs: Any,
) -> L3FillProvider:
    """
    Factory function to create L3 fill provider.

    Args:
        asset_class: Asset class
        config: L3 execution configuration
        slippage_provider: Custom slippage provider
        fee_provider: Custom fee provider
        **kwargs: Additional configuration

    Returns:
        L3FillProvider instance
    """
    return L3FillProvider(
        asset_class=asset_class,
        config=config,
        slippage_provider=slippage_provider,
        fee_provider=fee_provider,
        **kwargs,
    )


__all__ = [
    # Providers
    "L3SlippageProvider",
    "L3FillProvider",
    "L3ExecutionProvider",
    # Factory Functions
    "create_l3_execution_provider",
    "create_l3_slippage_provider",
    "create_l3_fill_provider",
]
