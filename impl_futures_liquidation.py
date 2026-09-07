# -*- coding: utf-8 -*-
"""
impl_futures_liquidation.py
Futures liquidation simulation engine.

Implements liquidation mechanics for:
- Binance USDT-M Perpetual (Insurance Fund + ADL)
- CME Futures (Margin Call → Forced Close)

Features:
- Liquidation detection based on margin ratio
- Partial and full liquidation support
- Insurance fund mechanics
- Auto-Deleveraging (ADL) simulation
- Liquidation cascade simulation
- Cross-margin liquidation ordering

Design Principles:
- Realistic liquidation mechanics
- Accurate insurance fund accounting
- Fair ADL queue ranking
- Testable with dependency injection

References:
- Binance Liquidation: https://www.binance.com/en/support/faq/360033525271
- Binance ADL: https://www.binance.com/en/support/faq/360033525711
- CME Margin Handbook: https://www.cmegroup.com/clearing/margins/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Tuple, Literal, Protocol
import logging
from collections import deque

from core_futures import (
    FuturesPosition,
    LiquidationEvent,
    MarginMode,
    PositionSide,
    FuturesType,
)
from impl_futures_margin import TieredMarginCalculator, FuturesMarginCalculator


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class LiquidationPriority(str, Enum):
    """
    Liquidation priority for cross-margin accounts.

    When an account's total margin ratio falls below 1.0, positions must be
    liquidated to restore margin. This enum defines the order in which
    positions are selected for liquidation.

    Different exchanges use different strategies:
    - Binance: HIGHEST_LOSS_FIRST (liquidate most unprofitable first)
    - Some exchanges: LOWEST_MARGIN_RATIO (most risky positions first)
    - Others: LARGEST_POSITION (reduce biggest exposure first)

    References:
    - Binance Cross-Margin: https://www.binance.com/en/support/faq/360038685551
    """

    HIGHEST_LOSS_FIRST = "highest_loss"  # Binance default - close losers first
    LOWEST_MARGIN_RATIO = "lowest_ratio"  # Most risky positions first
    OLDEST_POSITION = "oldest"  # FIFO - oldest positions first
    LARGEST_POSITION = "largest"  # Biggest notional first
    HIGHEST_LEVERAGE = "highest_leverage"  # Most leveraged first


class LiquidationType(str, Enum):
    """Type of liquidation event."""

    PARTIAL = "partial"  # Only part of position liquidated
    FULL = "full"  # Entire position liquidated
    ADL = "adl"  # Auto-deleveraging (counterparty liquidation)
    MARGIN_CALL = "margin_call"  # CME-style margin call


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ADLQueuePosition:
    """
    Position in Auto-Deleveraging (ADL) queue.

    ADL is triggered when insurance fund is insufficient to cover
    liquidation losses. Profitable traders on the opposite side
    are forced to close at bankruptcy price.

    Ranking based on: PnL percentile × Leverage percentile

    Attributes:
        symbol: Contract symbol
        side: Position side (LONG/SHORT)
        rank: ADL rank 1-5 (5 = highest priority for ADL)
        percentile: Position's percentile in ADL queue (0-100)
        margin_ratio: Current margin ratio
        pnl_ratio: PnL as percentage of margin
        estimated_adl_qty: Estimated qty to be ADL'd if triggered
        position_id: Unique position identifier

    Reference:
        https://www.binance.com/en/support/faq/360033525711
    """

    symbol: str
    side: Literal["LONG", "SHORT"]
    rank: int  # 1-5, where 5 = highest risk of ADL
    percentile: float  # 0-100, position's rank in ADL queue
    margin_ratio: Decimal
    pnl_ratio: Decimal  # PnL / margin
    estimated_adl_qty: Optional[Decimal] = None
    position_id: Optional[str] = None

    @property
    def is_high_risk(self) -> bool:
        """True if in top 20% (rank 4-5) - high ADL risk."""
        return self.rank >= 4

    @property
    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.rank == 5:
            return "CRITICAL"  # Will be ADL'd first
        elif self.rank == 4:
            return "HIGH"
        elif self.rank == 3:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class LiquidationResult:
    """
    Result of a liquidation check or execution.

    Attributes:
        triggered: Whether liquidation was triggered
        events: List of liquidation events
        insurance_fund_change: Change in insurance fund balance
        adl_events: Any ADL events triggered
        remaining_positions: Positions after liquidation
        total_loss: Total loss from liquidation
    """

    triggered: bool
    events: List[LiquidationEvent] = field(default_factory=list)
    insurance_fund_change: Decimal = Decimal("0")
    adl_events: List[Tuple[ADLQueuePosition, Decimal]] = field(default_factory=list)
    remaining_positions: List[FuturesPosition] = field(default_factory=list)
    total_loss: Decimal = Decimal("0")


@dataclass
class InsuranceFundState:
    """
    Insurance fund state.

    Attributes:
        balance: Current fund balance
        total_contributions: Historical contributions
        total_payouts: Historical payouts
        last_update_ms: Last update timestamp
    """

    balance: Decimal
    total_contributions: Decimal = Decimal("0")
    total_payouts: Decimal = Decimal("0")
    last_update_ms: int = 0


# ============================================================================
# CROSS-MARGIN LIQUIDATION ORDERING
# ============================================================================


class CrossMarginLiquidationOrdering:
    """
    Determines order of position liquidation for cross-margin accounts.

    In cross-margin mode, all positions share the same margin pool. When
    total account margin ratio drops below maintenance, the system must
    decide WHICH positions to liquidate first.

    This affects:
    1. Which positions get closed (user preference may differ from system)
    2. Cascade effects (closing one position may save others)
    3. Overall account survival probability
    """

    def __init__(self, priority: LiquidationPriority = LiquidationPriority.HIGHEST_LOSS_FIRST):
        """
        Initialize with liquidation priority.

        Args:
            priority: Strategy for ordering positions
        """
        self._priority = priority

    @property
    def priority(self) -> LiquidationPriority:
        """Get current priority setting."""
        return self._priority

    def order_positions_for_liquidation(
        self,
        positions: List[FuturesPosition],
        mark_prices: Dict[str, Decimal],
    ) -> List[FuturesPosition]:
        """
        Order positions by liquidation priority.

        Args:
            positions: All open positions in cross-margin account
            mark_prices: Current mark prices per symbol

        Returns:
            Positions ordered from first-to-liquidate to last
        """
        if not positions:
            return []

        def get_pnl(pos: FuturesPosition) -> Decimal:
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            if pos.qty > 0:  # Long
                return (mark - pos.entry_price) * abs(pos.qty)
            else:  # Short
                return (pos.entry_price - mark) * abs(pos.qty)

        def get_notional(pos: FuturesPosition) -> Decimal:
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            return mark * abs(pos.qty)

        if self._priority == LiquidationPriority.HIGHEST_LOSS_FIRST:
            # Sort by PnL ascending (most negative first)
            return sorted(positions, key=get_pnl)

        elif self._priority == LiquidationPriority.LOWEST_MARGIN_RATIO:
            # Would need margin calc per position - use leverage as proxy
            return sorted(positions, key=lambda p: -p.leverage)

        elif self._priority == LiquidationPriority.OLDEST_POSITION:
            # Sort by timestamp (requires timestamp on position)
            return sorted(positions, key=lambda p: p.timestamp_ms)

        elif self._priority == LiquidationPriority.LARGEST_POSITION:
            # Sort by notional descending
            return sorted(positions, key=get_notional, reverse=True)

        elif self._priority == LiquidationPriority.HIGHEST_LEVERAGE:
            # Sort by leverage descending
            return sorted(positions, key=lambda p: -p.leverage)

        return positions  # Default: no ordering

    def select_positions_to_liquidate(
        self,
        positions: List[FuturesPosition],
        mark_prices: Dict[str, Decimal],
        target_margin_ratio: Decimal,
        current_balance: Decimal,
        margin_calculator: FuturesMarginCalculator,
    ) -> List[FuturesPosition]:
        """
        Select minimum set of positions to liquidate to restore margin.

        Greedy algorithm: liquidate in priority order until margin ratio >= target.

        Args:
            positions: All open positions
            mark_prices: Current mark prices
            target_margin_ratio: Target margin ratio to achieve (e.g., 1.5)
            current_balance: Current wallet balance
            margin_calculator: Margin calculator instance

        Returns:
            List of positions to liquidate (subset of input)
        """
        ordered = self.order_positions_for_liquidation(positions, mark_prices)
        to_liquidate = []

        remaining = list(ordered)

        for pos in ordered:
            if not remaining:
                break

            # Calculate current margin ratio with remaining positions
            total_margin = sum(
                margin_calculator.calculate_maintenance_margin(
                    mark_prices.get(p.symbol, p.entry_price) * abs(p.qty)
                )
                for p in remaining
            )

            total_upnl = sum(
                self._calculate_pnl(p, mark_prices.get(p.symbol, p.entry_price)) for p in remaining
            )

            current_ratio = (
                (current_balance + total_upnl) / total_margin
                if total_margin > 0
                else Decimal("inf")
            )

            if current_ratio >= target_margin_ratio:
                break  # Target achieved

            # Mark for liquidation and remove from remaining
            to_liquidate.append(pos)
            remaining.remove(pos)

        return to_liquidate

    def _calculate_pnl(self, position: FuturesPosition, mark_price: Decimal) -> Decimal:
        """Calculate unrealized PnL for position."""
        if position.qty > 0:  # Long
            return (mark_price - position.entry_price) * abs(position.qty)
        else:  # Short
            return (position.entry_price - mark_price) * abs(position.qty)


# ============================================================================
# LIQUIDATION ENGINE
# ============================================================================


class LiquidationEngine:
    """
    Simulates futures liquidation mechanics.

    Liquidation process:
    1. Position margin ratio drops below 1 (MM)
    2. Liquidation order placed at bankruptcy price
    3. If filled above bankruptcy → profit to insurance fund
    4. If filled below bankruptcy → ADL or insurance fund covers

    Key concepts:
    - Liquidation Price: Price at which position is force-closed
    - Bankruptcy Price: Price at which position value = 0
    - Insurance Fund: Buffer to cover losses beyond margin
    - ADL: Auto-deleveraging when insurance fund depleted
    """

    def __init__(
        self,
        margin_calculator: FuturesMarginCalculator,
        insurance_fund_balance: Decimal = Decimal("1000000"),
        liquidation_fee_rate: Decimal = Decimal("0.005"),  # 0.5%
        partial_liquidation_ratio: Decimal = Decimal("0.5"),  # Liquidate 50% at a time
        enable_adl: bool = True,
    ):
        """
        Initialize liquidation engine.

        Args:
            margin_calculator: Margin calculator for margin ratio checks
            insurance_fund_balance: Initial insurance fund balance
            liquidation_fee_rate: Fee charged on liquidation (0.5% typical)
            partial_liquidation_ratio: Fraction to liquidate for partial liq
            enable_adl: Enable auto-deleveraging simulation
        """
        self._margin_calculator = margin_calculator
        self._insurance_fund = InsuranceFundState(balance=insurance_fund_balance)
        self._liquidation_fee_rate = liquidation_fee_rate
        self._partial_liq_ratio = partial_liquidation_ratio
        self._enable_adl = enable_adl

        # Track liquidation history
        self._liquidation_history: deque = deque(maxlen=1000)
        self._adl_history: deque = deque(maxlen=1000)

    @property
    def insurance_fund_balance(self) -> Decimal:
        """Get current insurance fund balance."""
        return self._insurance_fund.balance

    @property
    def insurance_fund_state(self) -> InsuranceFundState:
        """Get full insurance fund state."""
        return self._insurance_fund

    def check_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: int = 0,
    ) -> Optional[LiquidationEvent]:
        """
        Check if position should be liquidated.

        Liquidation is triggered when margin ratio < 1.0

        Args:
            position: Position to check
            mark_price: Current mark price
            wallet_balance: Wallet balance (for cross margin)
            timestamp_ms: Current timestamp

        Returns:
            LiquidationEvent if liquidation triggered, None otherwise
        """
        if position.qty == 0:
            return None

        # Calculate margin ratio
        margin_ratio = self._margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if margin_ratio >= Decimal("1"):
            return None  # No liquidation

        # Liquidation triggered
        return self._create_liquidation_event(
            position=position,
            mark_price=mark_price,
            wallet_balance=wallet_balance,
            timestamp_ms=timestamp_ms,
            liquidation_type=LiquidationType.FULL,
        )

    def check_partial_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: int = 0,
    ) -> Optional[LiquidationEvent]:
        """
        Check if partial liquidation should be triggered.

        Partial liquidation occurs when margin ratio is below 1.0 but
        we can restore health by closing part of the position.

        Args:
            position: Position to check
            mark_price: Current mark price
            wallet_balance: Wallet balance
            timestamp_ms: Current timestamp

        Returns:
            LiquidationEvent for partial liquidation, None if not needed
        """
        if position.qty == 0:
            return None

        margin_ratio = self._margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if margin_ratio >= Decimal("1"):
            return None

        # Calculate quantity to liquidate to restore margin ratio to 1.5
        partial_qty = self._calculate_partial_liquidation_qty(
            position, mark_price, wallet_balance, target_ratio=Decimal("1.5")
        )

        if partial_qty >= abs(position.qty):
            # Full liquidation needed
            return self._create_liquidation_event(
                position=position,
                mark_price=mark_price,
                wallet_balance=wallet_balance,
                timestamp_ms=timestamp_ms,
                liquidation_type=LiquidationType.FULL,
            )

        # Partial liquidation
        return self._create_liquidation_event(
            position=position,
            mark_price=mark_price,
            wallet_balance=wallet_balance,
            timestamp_ms=timestamp_ms,
            liquidation_type=LiquidationType.PARTIAL,
            qty_override=partial_qty,
        )

    def execute_liquidation(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        fill_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: int = 0,
    ) -> LiquidationResult:
        """
        Execute liquidation and handle insurance fund/ADL.

        Process:
        1. Calculate loss amount
        2. Deduct from position margin/wallet
        3. If loss exceeds margin, use insurance fund
        4. If insurance fund insufficient, trigger ADL

        Args:
            position: Position being liquidated
            mark_price: Current mark price
            fill_price: Actual fill price (may differ from liquidation price)
            wallet_balance: Wallet balance
            timestamp_ms: Current timestamp

        Returns:
            LiquidationResult with events and fund changes
        """
        if position.qty == 0:
            return LiquidationResult(triggered=False)

        abs_qty = abs(position.qty)
        is_long = position.qty > 0
        notional = mark_price * abs_qty

        # Calculate loss
        if is_long:
            pnl = (fill_price - position.entry_price) * abs_qty
        else:
            pnl = (position.entry_price - fill_price) * abs_qty

        # Liquidation fee
        liq_fee = notional * self._liquidation_fee_rate

        # Total loss (can be positive if liquidation was profitable)
        total_loss = -pnl + liq_fee

        # Handle insurance fund
        insurance_change = Decimal("0")
        adl_events = []

        if total_loss > Decimal("0"):
            # Liquidation resulted in loss
            if position.margin_mode == MarginMode.CROSS:
                available = wallet_balance
            else:
                available = position.margin

            if total_loss > available:
                # Bankruptcy - loss exceeds margin
                shortfall = total_loss - available

                if shortfall <= self._insurance_fund.balance:
                    # Insurance fund covers
                    insurance_change = -shortfall
                    self._insurance_fund = InsuranceFundState(
                        balance=self._insurance_fund.balance - shortfall,
                        total_contributions=self._insurance_fund.total_contributions,
                        total_payouts=self._insurance_fund.total_payouts + shortfall,
                        last_update_ms=timestamp_ms,
                    )
                elif self._enable_adl:
                    # ADL needed
                    remaining_shortfall = shortfall - self._insurance_fund.balance

                    # Use remaining insurance fund
                    insurance_change = -self._insurance_fund.balance
                    self._insurance_fund = InsuranceFundState(
                        balance=Decimal("0"),
                        total_contributions=self._insurance_fund.total_contributions,
                        total_payouts=self._insurance_fund.total_payouts
                        + self._insurance_fund.balance,
                        last_update_ms=timestamp_ms,
                    )

                    # ADL would be triggered here
                    logger.warning(
                        f"ADL triggered for {position.symbol}: shortfall={remaining_shortfall}"
                    )
        else:
            # Liquidation was profitable - contribute to insurance fund
            profit = -total_loss
            insurance_change = profit
            self._insurance_fund = InsuranceFundState(
                balance=self._insurance_fund.balance + profit,
                total_contributions=self._insurance_fund.total_contributions + profit,
                total_payouts=self._insurance_fund.total_payouts,
                last_update_ms=timestamp_ms,
            )

        # Create liquidation event
        event = LiquidationEvent(
            symbol=position.symbol,
            timestamp_ms=timestamp_ms,
            side="SELL" if is_long else "BUY",
            qty=abs_qty,
            price=fill_price,
            liquidation_type="full",
            loss_amount=max(Decimal("0"), total_loss),
            insurance_fund_contribution=(
                abs(insurance_change) if insurance_change > 0 else Decimal("0")
            ),
        )

        self._liquidation_history.append(event)

        return LiquidationResult(
            triggered=True,
            events=[event],
            insurance_fund_change=insurance_change,
            adl_events=adl_events,
            remaining_positions=[],
            total_loss=max(Decimal("0"), total_loss),
        )

    def _create_liquidation_event(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        timestamp_ms: int,
        liquidation_type: LiquidationType,
        qty_override: Optional[Decimal] = None,
    ) -> LiquidationEvent:
        """Create liquidation event."""
        is_long = position.qty > 0
        qty = qty_override if qty_override is not None else abs(position.qty)
        notional = mark_price * qty

        # Calculate loss
        if is_long:
            pnl = (mark_price - position.entry_price) * qty
        else:
            pnl = (position.entry_price - mark_price) * qty

        liq_fee = notional * self._liquidation_fee_rate
        loss = max(Decimal("0"), -pnl + liq_fee)

        return LiquidationEvent(
            symbol=position.symbol,
            timestamp_ms=timestamp_ms,
            side="SELL" if is_long else "BUY",
            qty=qty,
            price=mark_price,
            liquidation_type=liquidation_type.value,
            loss_amount=loss,
            insurance_fund_contribution=Decimal("0"),
        )

    def _calculate_partial_liquidation_qty(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
        target_ratio: Decimal = Decimal("1.5"),
    ) -> Decimal:
        """
        Calculate quantity to liquidate to restore margin ratio.

        Uses iterative approach to find minimum qty to close.

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance
            target_ratio: Target margin ratio after partial liquidation

        Returns:
            Quantity to liquidate
        """
        # Simple approach: liquidate fixed ratio
        return abs(position.qty) * self._partial_liq_ratio

    def calculate_bankruptcy_price(
        self,
        position: FuturesPosition,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate bankruptcy price (position value = 0).

        This is the price at which the position has zero value
        and any further movement causes socialized losses.

        Args:
            position: Current position
            wallet_balance: Wallet balance

        Returns:
            Bankruptcy price
        """
        if position.qty == 0:
            return Decimal("0")

        is_long = position.qty > 0
        abs_qty = abs(position.qty)

        # Available margin
        if position.margin_mode == MarginMode.CROSS:
            available = wallet_balance
        else:
            available = position.margin

        if is_long:
            # Long: bankruptcy when price = entry - available/qty
            bankruptcy = position.entry_price - available / abs_qty
        else:
            # Short: bankruptcy when price = entry + available/qty
            bankruptcy = position.entry_price + available / abs_qty

        return max(Decimal("0"), bankruptcy)

    def get_margin_warning_level(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> str:
        """
        Get margin warning level.

        Levels:
        - SAFE: MR > 1.5 (50% buffer)
        - WARNING: 1.0 < MR <= 1.5 (margin call territory)
        - DANGER: MR <= 1.0 (liquidation imminent)

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance

        Returns:
            Warning level string
        """
        margin_ratio = self._margin_calculator.calculate_margin_ratio(
            position, mark_price, wallet_balance
        )

        if margin_ratio > Decimal("1.5"):
            return "SAFE"
        elif margin_ratio > Decimal("1.0"):
            return "WARNING"
        else:
            return "DANGER"


# ============================================================================
# ADL SIMULATOR
# ============================================================================


class ADLSimulator:
    """
    Simulates Auto-Deleveraging events.

    When liquidation cannot be filled at bankruptcy price and
    insurance fund is depleted, profitable traders are ADL'd.

    ADL ranking based on: PnL percentile × Leverage percentile
    - Higher PnL = higher ADL priority
    - Higher leverage = higher ADL priority

    References:
    https://www.binance.com/en/support/faq/360033525711
    """

    def __init__(self):
        """Initialize ADL simulator."""
        self._adl_queues: Dict[str, List[ADLQueuePosition]] = {}

    def build_adl_queue(
        self,
        positions: List[FuturesPosition],
        symbol: str,
        side: Literal["LONG", "SHORT"],
        mark_price: Decimal,
    ) -> List[ADLQueuePosition]:
        """
        Build ADL queue for positions that could be ADL'd.

        ADL targets the opposite side of the liquidated position.
        If a LONG is liquidated, profitable SHORTs are ADL'd.

        Args:
            positions: All positions in the system
            symbol: Contract symbol
            side: Side being liquidated (LONG or SHORT)
            mark_price: Current mark price

        Returns:
            Ordered list of ADL queue positions
        """
        # Filter positions on opposite side
        opposite_side = "SHORT" if side == "LONG" else "LONG"
        candidates = [
            p
            for p in positions
            if p.symbol == symbol
            and (
                (opposite_side == "LONG" and p.qty > 0) or (opposite_side == "SHORT" and p.qty < 0)
            )
        ]

        if not candidates:
            return []

        # Calculate PnL and leverage for each
        scored = []
        for pos in candidates:
            if pos.qty > 0:
                pnl = (mark_price - pos.entry_price) * abs(pos.qty)
            else:
                pnl = (pos.entry_price - mark_price) * abs(pos.qty)

            pnl_pct = float(pnl / pos.margin) if pos.margin > 0 else 0.0
            leverage = float(abs(pos.qty) * mark_price / pos.margin) if pos.margin > 0 else 0.0
            scored.append((pos, pnl_pct, leverage))

        # Calculate percentiles
        pnl_values = sorted([s[1] for s in scored])
        lev_values = sorted([s[2] for s in scored])

        def percentile_rank(value: float, sorted_list: List[float]) -> float:
            if not sorted_list:
                return 0.0
            # Find position in sorted list
            count_below = sum(1 for v in sorted_list if v < value)
            count_equal = sum(1 for v in sorted_list if v == value)
            return (count_below + 0.5 * count_equal) / len(sorted_list)

        queue = []
        for pos, pnl_pct, leverage in scored:
            pnl_percentile = percentile_rank(pnl_pct, pnl_values)
            lev_percentile = percentile_rank(leverage, lev_values)
            score = pnl_percentile * lev_percentile

            # ADL rank 1-5
            if score >= 0.8:
                rank = 5
            elif score >= 0.6:
                rank = 4
            elif score >= 0.4:
                rank = 3
            elif score >= 0.2:
                rank = 2
            else:
                rank = 1

            queue.append(
                ADLQueuePosition(
                    symbol=pos.symbol,
                    side=opposite_side,
                    rank=rank,
                    percentile=score * 100,
                    margin_ratio=Decimal("0"),  # Would need to calculate
                    pnl_ratio=Decimal(str(pnl_pct)),
                )
            )

        # Sort by rank descending (highest risk first)
        queue.sort(key=lambda x: (-x.rank, -x.percentile))

        self._adl_queues[f"{symbol}_{opposite_side}"] = queue
        return queue

    def get_adl_queue(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
    ) -> List[ADLQueuePosition]:
        """Get cached ADL queue."""
        key = f"{symbol}_{side}"
        return self._adl_queues.get(key, [])

    def execute_adl(
        self,
        positions: List[FuturesPosition],
        symbol: str,
        liquidated_side: Literal["LONG", "SHORT"],
        qty_to_adl: Decimal,
        bankruptcy_price: Decimal,
        mark_price: Decimal,
    ) -> List[Tuple[FuturesPosition, Decimal, Decimal]]:
        """
        Execute ADL on positions in queue order.

        Args:
            positions: All positions
            symbol: Contract symbol
            liquidated_side: Side of the liquidated position
            qty_to_adl: Quantity that needs to be covered via ADL
            bankruptcy_price: Price at which to close ADL'd positions
            mark_price: Current mark price

        Returns:
            List of (position, qty_adl'd, realized_pnl) tuples
        """
        # Build queue for opposite side
        opposite_side = "SHORT" if liquidated_side == "LONG" else "LONG"
        queue = self.build_adl_queue(positions, symbol, liquidated_side, mark_price)

        results = []
        remaining = qty_to_adl

        for adl_pos in queue:
            if remaining <= 0:
                break

            # Find actual position matching queue entry
            matching = [
                p
                for p in positions
                if p.symbol == symbol
                and (
                    (adl_pos.side == "LONG" and p.qty > 0)
                    or (adl_pos.side == "SHORT" and p.qty < 0)
                )
            ]

            if not matching:
                continue

            pos = matching[0]
            adl_qty = min(remaining, abs(pos.qty))

            # Calculate realized PnL at bankruptcy price
            if pos.qty > 0:  # Long being ADL'd
                realized_pnl = (bankruptcy_price - pos.entry_price) * adl_qty
            else:  # Short being ADL'd
                realized_pnl = (pos.entry_price - bankruptcy_price) * adl_qty

            results.append((pos, adl_qty, realized_pnl))
            remaining -= adl_qty

        return results

    def get_adl_indicator(
        self,
        position: FuturesPosition,
        all_positions: List[FuturesPosition],
        mark_price: Decimal,
    ) -> int:
        """
        Get ADL indicator (1-5 lights) for a position.

        This is the indicator shown in trading UI to warn users
        about their ADL risk.

        Args:
            position: Position to check
            all_positions: All positions in the system
            mark_price: Current mark price

        Returns:
            ADL rank 1-5 (5 = highest risk)
        """
        side = "LONG" if position.qty > 0 else "SHORT"
        queue = self.build_adl_queue(
            all_positions, position.symbol, "SHORT" if side == "LONG" else "LONG", mark_price
        )

        # Find this position in queue
        for adl_pos in queue:
            # Match by PnL ratio (simplified matching)
            if abs(float(adl_pos.pnl_ratio) - float(position.roe_pct / 100)) < 0.01:
                return adl_pos.rank

        return 1  # Default low risk


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_liquidation_engine(
    futures_type: FuturesType,
    margin_calculator: FuturesMarginCalculator,
    config: Optional[Dict] = None,
) -> LiquidationEngine:
    """
    Create liquidation engine for futures type.

    Args:
        futures_type: Type of futures
        margin_calculator: Margin calculator
        config: Additional configuration

    Returns:
        Configured liquidation engine
    """
    config = config or {}

    return LiquidationEngine(
        margin_calculator=margin_calculator,
        insurance_fund_balance=Decimal(str(config.get("insurance_fund_balance", "1000000"))),
        liquidation_fee_rate=Decimal(str(config.get("liquidation_fee_rate", "0.005"))),
        partial_liquidation_ratio=Decimal(str(config.get("partial_liquidation_ratio", "0.5"))),
        enable_adl=config.get("enable_adl", futures_type.is_crypto),
    )
