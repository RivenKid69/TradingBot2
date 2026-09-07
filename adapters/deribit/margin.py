# -*- coding: utf-8 -*-
"""
adapters/deribit/margin.py

Deribit Inverse Margin Calculator.

This module implements margin calculations for Deribit crypto options,
which use inverse settlement (P&L in underlying crypto, not USD).

Key Concepts:

1. **Inverse Settlement**:
   - All P&L is denominated in the underlying cryptocurrency (BTC or ETH)
   - Call payoff: max(0, S-K) / S  (in crypto units)
   - Put payoff: max(0, K-S) / S   (in crypto units)

2. **Double-Whammy Risk**:
   - For SHORT positions, as crypto price drops:
     a) The option may become more valuable (loss in crypto terms)
     b) The USD value of your crypto margin also drops
   - This creates additional risk not present in traditional options

3. **Margin Requirements**:
   - Initial Margin (IM): Required to open a position
   - Maintenance Margin (MM): Minimum to keep position open
   - Deribit uses portfolio margin with cross-margining

4. **Mark Price**:
   - Used for margin calculations (not last traded price)
   - Based on index price and theoretical value

References:
    - Deribit Margin Documentation: https://www.deribit.com/pages/docs/margins
    - Inverse Derivatives: "Understanding Crypto Derivatives" (Deribit research)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Deribit margin parameters (as of 2024)
# These are approximate and may change - always verify with Deribit
INITIAL_MARGIN_FACTOR = Decimal("0.10")  # 10% for options
MAINTENANCE_MARGIN_FACTOR = Decimal("0.075")  # 7.5% for options

# Maximum leverage by underlying
MAX_LEVERAGE = {
    "BTC": Decimal("10"),  # 10x max
    "ETH": Decimal("10"),  # 10x max
}

# Liquidation buffer (margin called before liquidation)
LIQUIDATION_BUFFER = Decimal("0.005")  # 0.5%

# Minimum margin requirements (in underlying)
MIN_MARGIN = {
    "BTC": Decimal("0.001"),  # 0.001 BTC
    "ETH": Decimal("0.01"),  # 0.01 ETH
}


# =============================================================================
# Enums
# =============================================================================


class MarginMode(str, Enum):
    """Margin mode for Deribit accounts."""

    CROSS = "cross"  # Portfolio margin (cross-margining)
    ISOLATED = "isolated"  # Isolated margin per position (not supported for options)


class MarginCallLevel(str, Enum):
    """Margin call severity levels."""

    NONE = "none"
    WARNING = "warning"  # Approaching maintenance
    MARGIN_CALL = "margin_call"  # Below maintenance
    LIQUIDATION = "liquidation"  # Force liquidation


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class InversePayoff:
    """
    Payoff for an inverse-settled option.

    All values are in the underlying cryptocurrency units.
    """

    payoff_crypto: Decimal  # Payoff in crypto units
    payoff_usd: Decimal  # Payoff in USD (for reference)
    spot_price: Decimal  # Spot price at calculation
    strike: Decimal  # Strike price
    is_call: bool  # True if call option
    is_itm: bool  # True if in-the-money

    @property
    def intrinsic_value_usd(self) -> Decimal:
        """Intrinsic value in USD."""
        if self.is_call:
            return max(Decimal("0"), self.spot_price - self.strike)
        else:
            return max(Decimal("0"), self.strike - self.spot_price)


@dataclass
class DeribitMarginResult:
    """
    Result of margin calculation for a position or portfolio.
    """

    initial_margin: Decimal  # Required to open (in crypto)
    maintenance_margin: Decimal  # Required to maintain (in crypto)
    mark_value: Decimal  # Current mark-to-market value (in crypto)
    available_margin: Decimal  # Margin available for new trades
    margin_balance: Decimal  # Total margin balance
    margin_ratio: Decimal  # Margin used / total margin
    margin_call_level: MarginCallLevel
    liquidation_price: Optional[Decimal] = None
    currency: str = "BTC"

    # Portfolio-level values
    delta_total: Decimal = Decimal("0")
    gamma_total: Decimal = Decimal("0")
    vega_total: Decimal = Decimal("0")
    theta_total: Decimal = Decimal("0")

    @property
    def excess_margin(self) -> Decimal:
        """Margin above maintenance requirement."""
        return max(Decimal("0"), self.margin_balance - self.maintenance_margin)

    @property
    def is_margin_call(self) -> bool:
        """True if below maintenance margin."""
        return self.margin_call_level in (
            MarginCallLevel.MARGIN_CALL,
            MarginCallLevel.LIQUIDATION,
        )

    def to_usd(self, crypto_price: Decimal) -> Dict[str, Decimal]:
        """Convert margin values to USD."""
        return {
            "initial_margin_usd": self.initial_margin * crypto_price,
            "maintenance_margin_usd": self.maintenance_margin * crypto_price,
            "mark_value_usd": self.mark_value * crypto_price,
            "available_margin_usd": self.available_margin * crypto_price,
            "margin_balance_usd": self.margin_balance * crypto_price,
        }


@dataclass
class PositionForMargin:
    """
    Position data required for margin calculation.
    """

    instrument_name: str
    size: Decimal  # Positive = long, negative = short
    mark_price: Decimal  # Mark price in crypto
    index_price: Decimal  # Underlying index price in USD
    strike: Decimal  # Strike price
    is_call: bool  # True if call
    delta: Decimal = Decimal("0")
    gamma: Decimal = Decimal("0")
    vega: Decimal = Decimal("0")
    theta: Decimal = Decimal("0")
    time_to_expiry: Decimal = Decimal("0")  # Years to expiry

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def abs_size(self) -> Decimal:
        return abs(self.size)

    @property
    def notional_crypto(self) -> Decimal:
        """Notional value in crypto units."""
        return self.abs_size * self.mark_price

    @property
    def notional_usd(self) -> Decimal:
        """Notional value in USD."""
        return self.abs_size * self.index_price


# =============================================================================
# Inverse Settlement Calculator
# =============================================================================


class InverseSettlementCalculator:
    """
    Calculator for inverse-settled option payoffs.

    In inverse settlement, payoffs are denominated in the underlying
    cryptocurrency rather than USD. This creates unique characteristics:

    1. Call payoff: max(0, S-K) / S  (crypto units)
       - As S increases, payoff in crypto DECREASES (but USD value increases)
       - Maximum payoff approaches 1 crypto unit as S → ∞

    2. Put payoff: max(0, K-S) / S   (crypto units)
       - As S decreases, payoff in crypto INCREASES dramatically
       - Can exceed strike/S ratio significantly for deep ITM

    The "double-whammy" risk for sellers:
       - Short call: If S rises, owe more crypto + crypto is worth more USD
       - Short put: If S falls, owe more crypto but crypto worth less USD
         (This partially hedges the loss)
    """

    def calculate_call_payoff(
        self,
        spot: Union[Decimal, float],
        strike: Union[Decimal, float],
    ) -> InversePayoff:
        """
        Calculate inverse call option payoff.

        Formula: payoff_crypto = max(0, S - K) / S

        Args:
            spot: Spot price at expiration
            strike: Strike price

        Returns:
            InversePayoff with all relevant values
        """
        spot = Decimal(str(spot))
        strike = Decimal(str(strike))

        if spot <= 0:
            return InversePayoff(
                payoff_crypto=Decimal("0"),
                payoff_usd=Decimal("0"),
                spot_price=spot,
                strike=strike,
                is_call=True,
                is_itm=False,
            )

        intrinsic_usd = max(Decimal("0"), spot - strike)
        payoff_crypto = intrinsic_usd / spot
        is_itm = spot > strike

        return InversePayoff(
            payoff_crypto=payoff_crypto,
            payoff_usd=intrinsic_usd,
            spot_price=spot,
            strike=strike,
            is_call=True,
            is_itm=is_itm,
        )

    def calculate_put_payoff(
        self,
        spot: Union[Decimal, float],
        strike: Union[Decimal, float],
    ) -> InversePayoff:
        """
        Calculate inverse put option payoff.

        Formula: payoff_crypto = max(0, K - S) / S

        Args:
            spot: Spot price at expiration
            strike: Strike price

        Returns:
            InversePayoff with all relevant values
        """
        spot = Decimal(str(spot))
        strike = Decimal(str(strike))

        if spot <= 0:
            # Edge case: if spot = 0, put payoff is infinite in theory
            # In practice, this shouldn't happen with proper risk management
            return InversePayoff(
                payoff_crypto=strike,  # Approximate as strike
                payoff_usd=strike,
                spot_price=spot,
                strike=strike,
                is_call=False,
                is_itm=True,
            )

        intrinsic_usd = max(Decimal("0"), strike - spot)
        payoff_crypto = intrinsic_usd / spot
        is_itm = spot < strike

        return InversePayoff(
            payoff_crypto=payoff_crypto,
            payoff_usd=intrinsic_usd,
            spot_price=spot,
            strike=strike,
            is_call=False,
            is_itm=is_itm,
        )

    def calculate_payoff(
        self,
        spot: Union[Decimal, float],
        strike: Union[Decimal, float],
        is_call: bool,
    ) -> InversePayoff:
        """
        Calculate inverse option payoff for either call or put.

        Args:
            spot: Spot price at expiration
            strike: Strike price
            is_call: True for call, False for put

        Returns:
            InversePayoff with all relevant values
        """
        if is_call:
            return self.calculate_call_payoff(spot, strike)
        else:
            return self.calculate_put_payoff(spot, strike)

    def calculate_pnl(
        self,
        entry_price: Decimal,
        exit_price: Decimal,
        size: Decimal,
        is_long: bool,
    ) -> Decimal:
        """
        Calculate P&L in crypto units.

        Args:
            entry_price: Entry price in crypto
            exit_price: Exit price in crypto
            size: Position size (always positive)
            is_long: True if long position

        Returns:
            P&L in crypto units (positive = profit)
        """
        price_diff = exit_price - entry_price
        if is_long:
            return price_diff * size
        else:
            return -price_diff * size


# =============================================================================
# Margin Calculator
# =============================================================================


class DeribitMarginCalculator:
    """
    Margin calculator for Deribit options positions.

    Implements Deribit's portfolio margin methodology:
    1. Mark-to-market all positions
    2. Calculate initial margin (IM) for each position
    3. Calculate maintenance margin (MM) for each position
    4. Apply portfolio benefits (hedging offsets)
    5. Determine margin call level

    Margin Requirements:
    - Long options: Premium paid (no margin required beyond this)
    - Short options: Based on potential loss + volatility
    - Portfolio margin reduces requirements for hedged positions
    """

    def __init__(
        self,
        currency: str = "BTC",
        im_factor: Decimal = INITIAL_MARGIN_FACTOR,
        mm_factor: Decimal = MAINTENANCE_MARGIN_FACTOR,
    ):
        """
        Initialize margin calculator.

        Args:
            currency: Base currency ("BTC" or "ETH")
            im_factor: Initial margin factor
            mm_factor: Maintenance margin factor
        """
        self._currency = currency.upper()
        self._im_factor = im_factor
        self._mm_factor = mm_factor
        self._settlement_calc = InverseSettlementCalculator()

    @property
    def currency(self) -> str:
        return self._currency

    def calculate_single_position_margin(
        self,
        position: PositionForMargin,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate margin for a single position.

        Args:
            position: Position details

        Returns:
            Tuple of (initial_margin, maintenance_margin) in crypto
        """
        abs_size = position.abs_size
        mark_price = position.mark_price
        index_price = position.index_price

        if abs_size == 0:
            return Decimal("0"), Decimal("0")

        # Long positions: margin = premium paid
        if position.is_long:
            im = abs_size * mark_price
            mm = im * Decimal("0.5")  # Lower maintenance for long
            return im, mm

        # Short positions: need additional margin for potential loss
        # Base margin on potential adverse move
        strike = position.strike
        is_call = position.is_call

        # Calculate intrinsic value
        if is_call:
            intrinsic = max(Decimal("0"), (index_price - strike) / index_price)
        else:
            intrinsic = max(Decimal("0"), (strike - index_price) / index_price)

        # Time value component (simplified - actual uses vol model)
        time_value = mark_price - intrinsic
        if time_value < 0:
            time_value = Decimal("0")

        # Volatility margin: higher for more volatile underlying
        vol_margin = abs_size * self._im_factor

        # Initial margin = max(intrinsic, time_value) + vol margin
        base_margin = max(intrinsic * abs_size, time_value * abs_size)
        im = base_margin + vol_margin

        # Maintenance margin is lower
        mm = im * (self._mm_factor / self._im_factor)

        # Minimum margin requirement
        min_margin = MIN_MARGIN.get(self._currency, Decimal("0.001"))
        im = max(im, min_margin)
        mm = max(mm, min_margin * Decimal("0.5"))

        return im, mm

    def calculate_portfolio_margin(
        self,
        positions: List[PositionForMargin],
        margin_balance: Decimal,
    ) -> DeribitMarginResult:
        """
        Calculate portfolio margin for multiple positions.

        Applies cross-margining benefits for hedged positions.

        Args:
            positions: List of positions
            margin_balance: Current margin balance in crypto

        Returns:
            DeribitMarginResult with full margin details
        """
        if not positions:
            return DeribitMarginResult(
                initial_margin=Decimal("0"),
                maintenance_margin=Decimal("0"),
                mark_value=Decimal("0"),
                available_margin=margin_balance,
                margin_balance=margin_balance,
                margin_ratio=Decimal("0"),
                margin_call_level=MarginCallLevel.NONE,
                currency=self._currency,
            )

        # Calculate individual margins
        total_im = Decimal("0")
        total_mm = Decimal("0")
        total_mark_value = Decimal("0")
        delta_total = Decimal("0")
        gamma_total = Decimal("0")
        vega_total = Decimal("0")
        theta_total = Decimal("0")

        for pos in positions:
            im, mm = self.calculate_single_position_margin(pos)
            total_im += im
            total_mm += mm

            # Mark value (positive for long, negative for short)
            mark_val = pos.size * pos.mark_price
            total_mark_value += mark_val

            # Aggregate Greeks
            delta_total += pos.delta * pos.size
            gamma_total += pos.gamma * pos.size
            vega_total += pos.vega * pos.size
            theta_total += pos.theta * pos.size

        # Apply portfolio margin benefit for hedged positions
        # Simplified: reduce margin if delta is hedged
        hedge_benefit = self._calculate_hedge_benefit(positions, delta_total)
        adjusted_im = max(total_im * (Decimal("1") - hedge_benefit), total_im * Decimal("0.3"))
        adjusted_mm = max(total_mm * (Decimal("1") - hedge_benefit), total_mm * Decimal("0.3"))

        # Calculate margin ratio and available margin
        if margin_balance > 0:
            margin_ratio = adjusted_mm / margin_balance
        else:
            margin_ratio = Decimal("999")  # Max ratio if no balance

        available_margin = margin_balance - adjusted_im

        # Determine margin call level
        margin_call_level = self._determine_margin_call_level(
            margin_balance=margin_balance,
            maintenance_margin=adjusted_mm,
        )

        return DeribitMarginResult(
            initial_margin=adjusted_im,
            maintenance_margin=adjusted_mm,
            mark_value=total_mark_value,
            available_margin=available_margin,
            margin_balance=margin_balance,
            margin_ratio=margin_ratio,
            margin_call_level=margin_call_level,
            currency=self._currency,
            delta_total=delta_total,
            gamma_total=gamma_total,
            vega_total=vega_total,
            theta_total=theta_total,
        )

    def _calculate_hedge_benefit(
        self,
        positions: List[PositionForMargin],
        net_delta: Decimal,
    ) -> Decimal:
        """
        Calculate portfolio margin benefit from hedging.

        A perfectly hedged portfolio (delta = 0) gets maximum benefit.
        """
        if not positions:
            return Decimal("0")

        # Calculate gross delta exposure
        gross_delta = sum(abs(p.delta * p.size) for p in positions)

        if gross_delta == 0:
            return Decimal("0")

        # Hedge ratio: how much of gross exposure is netted out
        net_delta_abs = abs(net_delta)
        hedge_ratio = Decimal("1") - (net_delta_abs / gross_delta)
        hedge_ratio = max(Decimal("0"), min(hedge_ratio, Decimal("1")))

        # Maximum benefit is 50% margin reduction
        max_benefit = Decimal("0.5")
        return hedge_ratio * max_benefit

    def _determine_margin_call_level(
        self,
        margin_balance: Decimal,
        maintenance_margin: Decimal,
    ) -> MarginCallLevel:
        """Determine margin call level based on current margin."""
        if margin_balance <= 0:
            return MarginCallLevel.LIQUIDATION

        ratio = maintenance_margin / margin_balance if margin_balance > 0 else Decimal("999")

        if ratio >= Decimal("1.0") - LIQUIDATION_BUFFER:
            return MarginCallLevel.LIQUIDATION
        elif ratio >= Decimal("0.9"):
            return MarginCallLevel.MARGIN_CALL
        elif ratio >= Decimal("0.75"):
            return MarginCallLevel.WARNING
        else:
            return MarginCallLevel.NONE

    def estimate_liquidation_price(
        self,
        position: PositionForMargin,
        margin_balance: Decimal,
    ) -> Optional[Decimal]:
        """
        Estimate liquidation price for a single position.

        This is a simplified estimate - actual liquidation depends on
        portfolio margin and other factors.

        Args:
            position: Position to analyze
            margin_balance: Current margin balance

        Returns:
            Estimated liquidation price or None if not applicable
        """
        if position.is_long:
            # Long positions don't have forced liquidation
            return None

        # For short positions, estimate where margin would be exhausted
        abs_size = position.abs_size
        if abs_size == 0:
            return None

        index_price = position.index_price
        strike = position.strike
        mark_price = position.mark_price

        # Estimate max loss that would trigger liquidation
        max_loss = margin_balance * (Decimal("1") - LIQUIDATION_BUFFER)

        # For short call: loss increases as price rises
        if position.is_call:
            # Rough estimate: how much can price rise before margin exhausted
            # Loss per unit of price increase ≈ abs_size / index_price
            price_move = max_loss * index_price / abs_size
            liquidation_price = index_price + price_move
            return liquidation_price

        # For short put: loss increases as price falls
        else:
            # Loss per unit of price decrease ≈ abs_size * strike / index_price^2
            # This is more complex for puts due to inverse settlement
            price_move = max_loss * index_price * index_price / (abs_size * strike)
            liquidation_price = max(Decimal("0"), index_price - price_move)
            return liquidation_price

    def calculate_max_position_size(
        self,
        margin_available: Decimal,
        instrument_mark_price: Decimal,
        index_price: Decimal,
        strike: Decimal,
        is_call: bool,
        is_long: bool,
    ) -> Decimal:
        """
        Calculate maximum position size given available margin.

        Args:
            margin_available: Available margin in crypto
            instrument_mark_price: Mark price of the option
            index_price: Current underlying price
            strike: Strike price
            is_call: True for call
            is_long: True for long position

        Returns:
            Maximum position size (in contracts)
        """
        if margin_available <= 0:
            return Decimal("0")

        if is_long:
            # Long positions: margin = premium
            if instrument_mark_price <= 0:
                return Decimal("0")
            return margin_available / instrument_mark_price

        # Short positions: need margin for potential loss
        # Use simplified margin calculation
        position = PositionForMargin(
            instrument_name="",
            size=Decimal("-1"),  # Short 1 contract
            mark_price=instrument_mark_price,
            index_price=index_price,
            strike=strike,
            is_call=is_call,
        )

        im_per_contract, _ = self.calculate_single_position_margin(position)

        if im_per_contract <= 0:
            return Decimal("0")

        return margin_available / im_per_contract


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_inverse_call_payoff(
    spot: Union[Decimal, float],
    strike: Union[Decimal, float],
) -> Decimal:
    """
    Calculate inverse call option payoff in crypto units.

    Formula: max(0, S - K) / S

    Args:
        spot: Spot price at expiration
        strike: Strike price

    Returns:
        Payoff in cryptocurrency units
    """
    calc = InverseSettlementCalculator()
    return calc.calculate_call_payoff(spot, strike).payoff_crypto


def calculate_inverse_put_payoff(
    spot: Union[Decimal, float],
    strike: Union[Decimal, float],
) -> Decimal:
    """
    Calculate inverse put option payoff in crypto units.

    Formula: max(0, K - S) / S

    Args:
        spot: Spot price at expiration
        strike: Strike price

    Returns:
        Payoff in cryptocurrency units
    """
    calc = InverseSettlementCalculator()
    return calc.calculate_put_payoff(spot, strike).payoff_crypto


def create_deribit_margin_calculator(
    currency: str = "BTC",
    im_factor: Optional[Decimal] = None,
    mm_factor: Optional[Decimal] = None,
) -> DeribitMarginCalculator:
    """
    Create a Deribit margin calculator.

    Args:
        currency: Base currency ("BTC" or "ETH")
        im_factor: Optional initial margin factor override
        mm_factor: Optional maintenance margin factor override

    Returns:
        Configured margin calculator

    Example:
        >>> calc = create_deribit_margin_calculator("BTC")
        >>> position = PositionForMargin(
        ...     instrument_name="BTC-28MAR25-100000-C",
        ...     size=Decimal("-0.5"),  # Short 0.5 contracts
        ...     mark_price=Decimal("0.02"),
        ...     index_price=Decimal("50000"),
        ...     strike=Decimal("100000"),
        ...     is_call=True,
        ... )
        >>> im, mm = calc.calculate_single_position_margin(position)
        >>> print(f"Initial Margin: {im} BTC")
    """
    kwargs = {"currency": currency}
    if im_factor is not None:
        kwargs["im_factor"] = im_factor
    if mm_factor is not None:
        kwargs["mm_factor"] = mm_factor
    return DeribitMarginCalculator(**kwargs)
