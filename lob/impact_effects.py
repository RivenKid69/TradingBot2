# -*- coding: utf-8 -*-
"""
Impact Effects Module for L3 LOB Simulation.

Applies market impact to order book state, simulating:
- Quote shifting after aggressive orders
- Liquidity replenishment (market maker reaction)
- Momentum detection (informed trading patterns)
- Adverse selection indicators

This module bridges market impact models with order book mechanics,
providing realistic simulation of market microstructure dynamics.

References:
    - Lehalle & Laruelle (2018): "Market Microstructure in Practice"
    - Cont & Kukanov (2017): "Optimal Order Placement in LOB"
    - Moallemi & Sağlam (2013): "Or Wait: Optimal Market Making"
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderBook,
    OrderType,
    PriceLevel,
    Side,
    Trade,
)
from lob.market_impact import (
    AlmgrenChrissModel,
    ImpactParameters,
    ImpactResult,
    ImpactState,
    ImpactTracker,
    MarketImpactModel,
    create_impact_model,
)


# ==============================================================================
# Enums and Constants
# ==============================================================================


class QuoteShiftType(IntEnum):
    """Type of quote shift after impact."""

    SYMMETRIC = 1  # Both bid and ask shift equally
    ASYMMETRIC = 2  # Only aggressed side shifts
    WIDENING = 3  # Spread widens (both move outward)
    NARROWING = 4  # Spread narrows (recovery)


class LiquidityReaction(IntEnum):
    """Market maker reaction type."""

    REPLENISH = 1  # Add liquidity at new levels
    WITHDRAW = 2  # Remove liquidity (adverse selection)
    NEUTRAL = 3  # No change
    INFORMED = 4  # Detected informed flow - adjust quotes


class MomentumSignal(IntEnum):
    """Momentum detection signal."""

    NONE = 0
    WEAK_CONTINUATION = 1
    STRONG_CONTINUATION = 2
    REVERSAL_EXPECTED = 3
    PRICE_DISCOVERY = 4


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class QuoteShiftResult:
    """
    Result of quote shift operation.

    Attributes:
        new_bid: New best bid price after shift
        new_ask: New best ask price after shift
        bid_shift_bps: Bid price shift in basis points
        ask_shift_bps: Ask price shift in basis points
        new_spread_bps: New spread in basis points
        shift_type: Type of shift applied
    """

    new_bid: Optional[float] = None
    new_ask: Optional[float] = None
    bid_shift_bps: float = 0.0
    ask_shift_bps: float = 0.0
    new_spread_bps: float = 0.0
    shift_type: QuoteShiftType = QuoteShiftType.SYMMETRIC


@dataclass
class LiquidityReactionResult:
    """
    Result of liquidity reaction simulation.

    Attributes:
        reaction_type: Type of market maker reaction
        orders_added: Number of orders added to book
        orders_removed: Number of orders removed
        qty_added: Total quantity added
        qty_removed: Total quantity removed
        new_depth_bid: New bid depth (top N levels qty)
        new_depth_ask: New ask depth
        imbalance_change: Change in book imbalance
    """

    reaction_type: LiquidityReaction = LiquidityReaction.NEUTRAL
    orders_added: int = 0
    orders_removed: int = 0
    qty_added: float = 0.0
    qty_removed: float = 0.0
    new_depth_bid: float = 0.0
    new_depth_ask: float = 0.0
    imbalance_change: float = 0.0


@dataclass
class MomentumResult:
    """
    Result of momentum analysis.

    Attributes:
        signal: Momentum signal detected
        continuation_probability: P(price continues in same direction)
        reversal_probability: P(price reverses)
        confidence: Confidence level [0, 1]
        recent_trade_imbalance: Net buy/sell imbalance in recent trades
        price_velocity: Rate of price change
    """

    signal: MomentumSignal = MomentumSignal.NONE
    continuation_probability: float = 0.5
    reversal_probability: float = 0.5
    confidence: float = 0.5
    recent_trade_imbalance: float = 0.0
    price_velocity: float = 0.0


@dataclass
class AdverseSelectionResult:
    """
    Adverse selection analysis result.

    Attributes:
        adverse_selection_bps: Estimated adverse selection cost in bps
        information_probability: P(informed trader)
        toxic_flow_indicator: Toxicity score [0, 1]
        recommended_spread_bps: Recommended minimum spread
        confidence: Confidence level
    """

    adverse_selection_bps: float = 0.0
    information_probability: float = 0.0
    toxic_flow_indicator: float = 0.0
    recommended_spread_bps: float = 0.0
    confidence: float = 0.5


@dataclass
class ImpactEffectConfig:
    """
    Configuration for impact effects.

    Attributes:
        quote_shift_fraction: Fraction of impact to apply as quote shift
        replenish_rate: Rate of liquidity replenishment (qty/sec)
        replenish_delay_ms: Delay before replenishment starts
        momentum_lookback_trades: Number of recent trades for momentum
        adverse_selection_threshold: Threshold for adverse selection detection
        spread_widen_factor: Factor to widen spread during impact
        tick_size: Minimum price increment
    """

    quote_shift_fraction: float = 0.5  # 50% of impact shifts quotes
    replenish_rate: float = 100.0  # qty per second
    replenish_delay_ms: int = 100  # 100ms delay
    momentum_lookback_trades: int = 10  # Last 10 trades
    adverse_selection_threshold: float = 0.3  # 30% threshold
    spread_widen_factor: float = 1.5  # Spread widens by 50%
    tick_size: float = 0.01  # Default tick size


# ==============================================================================
# Impact Effects Engine
# ==============================================================================


class ImpactEffects:
    """
    Applies market impact effects to order book state.

    This class provides methods for simulating realistic market
    microstructure effects following trades:
    - Quote adjustment by market makers
    - Liquidity replenishment dynamics
    - Momentum pattern detection
    - Adverse selection indicators
    """

    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        config: Optional[ImpactEffectConfig] = None,
    ) -> None:
        """
        Initialize impact effects engine.

        Args:
            impact_model: Market impact model to use
            config: Configuration parameters
        """
        self._model = impact_model or create_impact_model("almgren_chriss")
        self._config = config or ImpactEffectConfig()

        # Tracking state
        self._recent_trades: Deque[Tuple[int, float, Side]] = deque(
            maxlen=self._config.momentum_lookback_trades * 2
        )
        self._impact_tracker = ImpactTracker(model=self._model)
        self._last_quote_shift_ms: int = 0

    @property
    def config(self) -> ImpactEffectConfig:
        """Get configuration."""
        return self._config

    @property
    def impact_model(self) -> MarketImpactModel:
        """Get impact model."""
        return self._model

    def apply_temporary_impact(
        self,
        order_book: OrderBook,
        impact_bps: float,
        side: Side,
        timestamp_ms: Optional[int] = None,
    ) -> QuoteShiftResult:
        """
        Apply temporary impact by shifting quotes.

        Temporary impact causes quotes to move away from the aggressor:
        - BUY aggression: quotes shift up (ask moves up, bid may follow)
        - SELL aggression: quotes shift down (bid moves down, ask may follow)

        Args:
            order_book: Order book to modify
            impact_bps: Impact in basis points to apply
            side: Side of aggressive order
            timestamp_ms: Current timestamp

        Returns:
            QuoteShiftResult with new quote levels
        """
        if timestamp_ms is not None:
            self._last_quote_shift_ms = timestamp_ms

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is None or best_ask is None:
            return QuoteShiftResult()

        mid_price = (best_bid + best_ask) / 2.0
        current_spread = best_ask - best_bid
        current_spread_bps = (current_spread / mid_price) * 10000.0

        # Calculate shift amount
        shift_fraction = self._config.quote_shift_fraction
        shift_bps = impact_bps * shift_fraction
        shift_price = mid_price * shift_bps / 10000.0

        # Determine shift type and apply
        if side == Side.BUY:
            # Buy aggression - prices shift up
            new_ask = best_ask + shift_price
            # Bid shifts up but less (spread may widen)
            widen_factor = self._config.spread_widen_factor
            bid_shift = shift_price / widen_factor
            new_bid = best_bid + bid_shift

            bid_shift_bps = (bid_shift / mid_price) * 10000.0
            ask_shift_bps = shift_bps

        else:
            # Sell aggression - prices shift down
            new_bid = best_bid - shift_price
            # Ask shifts down but less (spread may widen)
            widen_factor = self._config.spread_widen_factor
            ask_shift = shift_price / widen_factor
            new_ask = best_ask - ask_shift

            bid_shift_bps = -shift_bps
            ask_shift_bps = -(ask_shift / mid_price) * 10000.0

        # Round to tick size
        tick = self._config.tick_size
        new_bid = round(new_bid / tick) * tick
        new_ask = round(new_ask / tick) * tick

        # Ensure no crossing
        if new_bid >= new_ask:
            new_ask = new_bid + tick

        new_spread = new_ask - new_bid
        new_spread_bps = (new_spread / ((new_bid + new_ask) / 2.0)) * 10000.0

        return QuoteShiftResult(
            new_bid=new_bid,
            new_ask=new_ask,
            bid_shift_bps=bid_shift_bps,
            ask_shift_bps=ask_shift_bps,
            new_spread_bps=new_spread_bps,
            shift_type=QuoteShiftType.ASYMMETRIC if shift_bps > 2.0 else QuoteShiftType.SYMMETRIC,
        )

    def simulate_liquidity_reaction(
        self,
        order_book: OrderBook,
        our_order: LimitOrder,
        fill: Fill,
        adv: float,
        timestamp_ms: Optional[int] = None,
    ) -> LiquidityReactionResult:
        """
        Simulate market maker reaction to our trade.

        Market makers may:
        - Replenish liquidity after our trade consumes it
        - Withdraw liquidity if they detect informed flow
        - Adjust quotes based on new information

        Args:
            order_book: Current order book
            our_order: Our order that was filled
            fill: Fill result
            adv: Average daily volume
            timestamp_ms: Current timestamp

        Returns:
            LiquidityReactionResult with reaction details
        """
        # Calculate participation
        participation = fill.total_qty / max(adv, 1.0)

        # Get current depth
        bids, asks = order_book.get_depth(n_levels=10)
        current_bid_depth = sum(qty for _, qty in bids)
        current_ask_depth = sum(qty for _, qty in asks)

        # Determine reaction type based on fill characteristics
        reaction_type = self._determine_reaction_type(
            participation=participation,
            fill_qty=fill.total_qty,
            side=our_order.side,
            order_book=order_book,
        )

        # Calculate quantities to add/remove
        qty_added = 0.0
        qty_removed = 0.0
        orders_added = 0
        orders_removed = 0

        if reaction_type == LiquidityReaction.REPLENISH:
            # Normal replenishment - add back consumed liquidity
            replenish_qty = fill.total_qty * 0.8  # 80% replenishment
            qty_added = replenish_qty
            orders_added = max(1, int(replenish_qty / 100))

        elif reaction_type == LiquidityReaction.WITHDRAW:
            # Adverse selection detected - remove liquidity
            withdraw_qty = fill.total_qty * 0.3
            qty_removed = withdraw_qty
            orders_removed = max(1, int(withdraw_qty / 100))

        elif reaction_type == LiquidityReaction.INFORMED:
            # Adjust to new information level
            # Replenish but at worse prices
            qty_added = fill.total_qty * 0.5
            orders_added = max(1, int(qty_added / 100))

        # Calculate new depth
        if our_order.side == Side.BUY:
            # We consumed ask liquidity
            new_depth_ask = max(0, current_ask_depth - fill.total_qty + qty_added)
            new_depth_bid = current_bid_depth
        else:
            # We consumed bid liquidity
            new_depth_bid = max(0, current_bid_depth - fill.total_qty + qty_added)
            new_depth_ask = current_ask_depth

        # Calculate imbalance change
        old_total = current_bid_depth + current_ask_depth
        new_total = new_depth_bid + new_depth_ask
        if old_total > 0 and new_total > 0:
            old_imb = (current_bid_depth - current_ask_depth) / old_total
            new_imb = (new_depth_bid - new_depth_ask) / new_total
            imb_change = new_imb - old_imb
        else:
            imb_change = 0.0

        return LiquidityReactionResult(
            reaction_type=reaction_type,
            orders_added=orders_added,
            orders_removed=orders_removed,
            qty_added=qty_added,
            qty_removed=qty_removed,
            new_depth_bid=new_depth_bid,
            new_depth_ask=new_depth_ask,
            imbalance_change=imb_change,
        )

    def _determine_reaction_type(
        self,
        participation: float,
        fill_qty: float,
        side: Side,
        order_book: OrderBook,
    ) -> LiquidityReaction:
        """Determine MM reaction type based on trade characteristics."""
        # High participation suggests informed trading
        if participation > 0.05:  # > 5% of ADV
            return LiquidityReaction.INFORMED

        # Check recent trade pattern
        if len(self._recent_trades) >= 5:
            recent_sides = [s for _, _, s in list(self._recent_trades)[-5:]]
            same_side_count = sum(1 for s in recent_sides if s == side)
            if same_side_count >= 4:
                # Many trades in same direction - could be informed
                return LiquidityReaction.WITHDRAW

        # Normal trading - standard replenishment
        return LiquidityReaction.REPLENISH

    def detect_momentum(
        self,
        order_book: OrderBook,
        recent_trades: Optional[List[Trade]] = None,
    ) -> MomentumResult:
        """
        Detect momentum patterns from recent trading activity.

        Analyzes:
        - Trade direction imbalance
        - Price velocity
        - Order flow toxicity

        Args:
            order_book: Current order book
            recent_trades: Optional list of recent trades

        Returns:
            MomentumResult with momentum analysis
        """
        # Use internal recent trades if not provided
        if recent_trades:
            for t in recent_trades:
                side = t.aggressor_side or Side.BUY
                self._recent_trades.append((t.timestamp_ns, t.price, side))

        if len(self._recent_trades) < 3:
            return MomentumResult(signal=MomentumSignal.NONE)

        trades_list = list(self._recent_trades)

        # Calculate trade imbalance
        buy_volume = sum(1 for _, _, s in trades_list if s == Side.BUY)
        sell_volume = sum(1 for _, _, s in trades_list if s == Side.SELL)
        total = buy_volume + sell_volume

        if total == 0:
            return MomentumResult(signal=MomentumSignal.NONE)

        imbalance = (buy_volume - sell_volume) / total

        # Calculate price velocity
        if len(trades_list) >= 2:
            first_price = trades_list[0][1]
            last_price = trades_list[-1][1]
            first_ts = trades_list[0][0]
            last_ts = trades_list[-1][0]

            if last_ts > first_ts and first_price > 0:
                dt_sec = (last_ts - first_ts) / 1e9  # ns to sec
                price_change = (last_price - first_price) / first_price
                velocity = price_change / max(dt_sec, 0.001)  # Per second
            else:
                velocity = 0.0
        else:
            velocity = 0.0

        # Determine signal
        signal = MomentumSignal.NONE
        continuation_prob = 0.5
        reversal_prob = 0.5

        abs_imbalance = abs(imbalance)
        if abs_imbalance > 0.6:
            # Strong imbalance
            if abs_imbalance > 0.8:
                signal = MomentumSignal.STRONG_CONTINUATION
                continuation_prob = 0.7
            else:
                signal = MomentumSignal.WEAK_CONTINUATION
                continuation_prob = 0.6

            # Very strong imbalance often leads to reversal
            if abs_imbalance > 0.9:
                signal = MomentumSignal.REVERSAL_EXPECTED
                reversal_prob = 0.65
                continuation_prob = 0.35
        elif abs(velocity) > 0.001:  # 0.1% per second is significant
            signal = MomentumSignal.PRICE_DISCOVERY

        reversal_prob = 1.0 - continuation_prob

        return MomentumResult(
            signal=signal,
            continuation_probability=continuation_prob,
            reversal_probability=reversal_prob,
            confidence=min(0.9, 0.5 + abs_imbalance * 0.4),
            recent_trade_imbalance=imbalance,
            price_velocity=velocity,
        )

    def estimate_adverse_selection(
        self,
        order_book: OrderBook,
        our_order: LimitOrder,
        fill: Optional[Fill] = None,
        post_trade_mid: Optional[float] = None,
    ) -> AdverseSelectionResult:
        """
        Estimate adverse selection cost for a trade.

        Adverse selection occurs when our passive orders are picked off
        by informed traders who know the price is about to move against us.

        Args:
            order_book: Order book state
            our_order: Our order
            fill: Fill result (if available)
            post_trade_mid: Mid price after trade (for realized measure)

        Returns:
            AdverseSelectionResult with adverse selection analysis
        """
        mid_price = order_book.mid_price or our_order.price
        spread = order_book.spread or 0.0
        spread_bps = order_book.spread_bps or 0.0

        # Calculate basic adverse selection from price movement
        adverse_bps = 0.0
        if post_trade_mid is not None and fill is not None:
            # Realized adverse selection
            if our_order.side == Side.BUY:
                # We bought - adverse if price dropped after
                price_move = (mid_price - post_trade_mid) / mid_price * 10000.0
                adverse_bps = max(0, price_move)
            else:
                # We sold - adverse if price rose after
                price_move = (post_trade_mid - mid_price) / mid_price * 10000.0
                adverse_bps = max(0, price_move)

        # Estimate information probability from order characteristics
        info_prob = 0.0

        # Large orders more likely informed
        if fill is not None:
            size_ratio = fill.total_qty / max(order_book.best_bid_qty, 1.0)
            info_prob += min(0.3, size_ratio * 0.1)

        # Orders hitting spread aggressively more likely informed
        if fill is not None and fill.is_complete:
            info_prob += 0.1

        # Book imbalance suggests direction
        bids, asks = order_book.get_depth(5)
        bid_depth = sum(q for _, q in bids)
        ask_depth = sum(q for _, q in asks)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
            # Trading against imbalance more likely informed
            if our_order.side == Side.BUY and imbalance < -0.2:
                info_prob += 0.1
            elif our_order.side == Side.SELL and imbalance > 0.2:
                info_prob += 0.1

        info_prob = min(0.8, info_prob)

        # Calculate toxicity indicator
        toxicity = info_prob * 0.5 + (adverse_bps / max(spread_bps, 1.0)) * 0.5
        toxicity = min(1.0, max(0.0, toxicity))

        # Recommended spread to cover adverse selection
        recommended_spread_bps = spread_bps * (1.0 + toxicity * 0.5)

        return AdverseSelectionResult(
            adverse_selection_bps=adverse_bps,
            information_probability=info_prob,
            toxic_flow_indicator=toxicity,
            recommended_spread_bps=recommended_spread_bps,
            confidence=0.6 if fill is not None else 0.4,
        )

    def record_trade(
        self,
        trade: Trade,
    ) -> None:
        """Record a trade for momentum tracking."""
        side = trade.aggressor_side or Side.BUY
        self._recent_trades.append((trade.timestamp_ns, trade.price, side))

    def record_trade_details(
        self,
        timestamp_ns: int,
        price: float,
        side: Side,
    ) -> None:
        """Record trade details for momentum tracking."""
        self._recent_trades.append((timestamp_ns, price, side))

    def clear_history(self) -> None:
        """Clear trade history."""
        self._recent_trades.clear()


# ==============================================================================
# LOB Impact Simulator
# ==============================================================================


class LOBImpactSimulator:
    """
    High-level simulator that combines impact models with LOB state.

    Provides complete simulation of market impact including:
    - Impact computation
    - Quote adjustment
    - Liquidity dynamics
    - State tracking over time
    """

    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        config: Optional[ImpactEffectConfig] = None,
    ) -> None:
        """
        Initialize LOB impact simulator.

        Args:
            impact_model: Market impact model
            config: Configuration
        """
        self._model = impact_model or create_impact_model("almgren_chriss")
        self._config = config or ImpactEffectConfig()
        self._effects = ImpactEffects(impact_model=self._model, config=self._config)
        self._tracker = ImpactTracker(model=self._model)

        # State
        self._cumulative_impact: ImpactState = ImpactState()

    @property
    def effects(self) -> ImpactEffects:
        """Get effects engine."""
        return self._effects

    @property
    def tracker(self) -> ImpactTracker:
        """Get impact tracker."""
        return self._tracker

    @property
    def cumulative_impact_bps(self) -> float:
        """Get current cumulative impact."""
        return self._tracker.current_impact_bps

    def simulate_trade_impact(
        self,
        order_book: OrderBook,
        order: LimitOrder,
        fill: Fill,
        adv: float,
        volatility: float,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[ImpactResult, QuoteShiftResult, LiquidityReactionResult]:
        """
        Simulate complete impact of a trade.

        Args:
            order_book: Current order book
            order: Order that was executed
            fill: Fill result
            adv: Average daily volume
            volatility: Current volatility
            timestamp_ms: Timestamp

        Returns:
            Tuple of (ImpactResult, QuoteShiftResult, LiquidityReactionResult)
        """
        ts = timestamp_ms or int(time.time() * 1000)

        # Compute impact
        impact = self._tracker.record_trade(
            timestamp_ms=ts,
            order_qty=fill.total_qty,
            adv=adv,
            volatility=volatility,
        )

        # Apply quote shift
        quote_shift = self._effects.apply_temporary_impact(
            order_book=order_book,
            impact_bps=impact.temporary_impact_bps,
            side=order.side,
            timestamp_ms=ts,
        )

        # Simulate liquidity reaction
        liquidity_reaction = self._effects.simulate_liquidity_reaction(
            order_book=order_book,
            our_order=order,
            fill=fill,
            adv=adv,
            timestamp_ms=ts,
        )

        # Record trade for momentum
        self._effects.record_trade_details(
            timestamp_ns=ts * 1_000_000,  # ms to ns
            price=fill.avg_price,
            side=order.side,
        )

        return impact, quote_shift, liquidity_reaction

    def get_impact_at_time(self, timestamp_ms: int) -> float:
        """Get estimated impact at a specific time."""
        return self._tracker.get_impact_at_time(timestamp_ms)

    def reset(self) -> None:
        """Reset simulator state."""
        self._tracker.reset()
        self._effects.clear_history()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_impact_effects(
    model_type: str = "almgren_chriss",
    asset_class: str = "equity",
    **kwargs,
) -> ImpactEffects:
    """
    Factory function to create impact effects engine.

    Args:
        model_type: Market impact model type
        asset_class: Asset class
        **kwargs: Model and config parameters

    Returns:
        ImpactEffects instance
    """
    model = create_impact_model(model_type, asset_class, **kwargs)

    # Extract config parameters
    config_keys = [
        "quote_shift_fraction",
        "replenish_rate",
        "replenish_delay_ms",
        "momentum_lookback_trades",
        "adverse_selection_threshold",
        "spread_widen_factor",
        "tick_size",
    ]
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_keys}
    config = ImpactEffectConfig(**config_kwargs) if config_kwargs else None

    return ImpactEffects(impact_model=model, config=config)


def create_lob_impact_simulator(
    model_type: str = "almgren_chriss",
    asset_class: str = "equity",
    **kwargs,
) -> LOBImpactSimulator:
    """
    Factory function to create LOB impact simulator.

    Args:
        model_type: Market impact model type
        asset_class: Asset class
        **kwargs: Model and config parameters

    Returns:
        LOBImpactSimulator instance
    """
    model = create_impact_model(model_type, asset_class, **kwargs)

    config_keys = [
        "quote_shift_fraction",
        "replenish_rate",
        "replenish_delay_ms",
        "momentum_lookback_trades",
        "adverse_selection_threshold",
        "spread_widen_factor",
        "tick_size",
    ]
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_keys}
    config = ImpactEffectConfig(**config_kwargs) if config_kwargs else None

    return LOBImpactSimulator(impact_model=model, config=config)
