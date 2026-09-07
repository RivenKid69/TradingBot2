# -*- coding: utf-8 -*-
"""
lob/us_market_structure.py
US Equity Market Structure Rules for L3 LOB Simulation.

This module implements key US equity market structure regulations:
- Tick size enforcement (Reg NMS Rule 612)
- Odd lot handling (SEC Rule 600)
- NBBO protection (Reg NMS Rule 611 - Order Protection Rule)

These rules are critical for realistic equity simulation:
- Prices must be in valid tick increments
- Odd lots (< 100 shares) have different execution properties
- Orders must not trade through protected quotes (NBBO)

References:
    - SEC Reg NMS (2005): https://www.sec.gov/rules/final/34-51808.pdf
    - Rule 611: Order Protection Rule (Trade-Through)
    - Rule 612: Sub-Penny Rule
    - Rule 600: Definitions (odd lot, round lot)
    - NYSE Pillar Gateway Spec: Tick size and lot classifications

FIX (2025-11-28): Added as part of Issue #7 "L3 LOB: US Market Structure" fix.
Reference: docs/PLATFORM_REFERENCE.md → "L3 LOB: US Market Structure"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard tick sizes
TICK_SIZE_PENNY = 0.01  # Standard tick size for stocks >= $1.00
TICK_SIZE_SUB_PENNY = 0.0001  # Sub-penny for stocks < $1.00

# Round lot size
ROUND_LOT_SIZE = 100

# SEC tick size pilot thresholds (historical, for reference)
TICK_PILOT_THRESHOLD = 1.00  # Stocks below $1 can trade sub-penny


# =============================================================================
# ENUMS
# =============================================================================


class LotType(IntEnum):
    """Order lot classification per SEC Rule 600."""
    ODD_LOT = 1       # < 100 shares
    ROUND_LOT = 2     # Exactly 100 shares or multiples thereof
    MIXED_LOT = 3     # Combination: round lots + odd lot remainder


class TradeThrough(IntEnum):
    """Trade-through violation type per Reg NMS Rule 611."""
    NONE = 0          # No violation
    BID_THROUGH = 1   # Sell below protected bid
    ASK_THROUGH = 2   # Buy above protected ask


class OddLotHandling(str, Enum):
    """How to handle odd lot orders."""
    ALLOW = "allow"                  # Allow all odd lots (default)
    REJECT = "reject"                # Reject odd lots
    COMBINE_ONLY = "combine_only"    # Only accept as part of mixed lot
    SEGREGATE = "segregate"          # Route to odd-lot only book


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NBBO:
    """
    National Best Bid and Offer.

    The NBBO is the best available bid and ask prices across all
    protected exchanges. All orders must respect NBBO under Reg NMS.

    Attributes:
        bid: Best (highest) bid price across all venues
        bid_size: Total size at best bid
        bid_exchange: Exchange with best bid
        ask: Best (lowest) ask price across all venues
        ask_size: Total size at best ask
        ask_exchange: Exchange with best ask
        timestamp_ns: NBBO timestamp in nanoseconds
    """
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    bid_exchange: str = ""
    ask_exchange: str = ""
    timestamp_ns: int = 0

    @property
    def spread(self) -> float:
        """Bid-ask spread in price units."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        mid = (self.bid + self.ask) / 2.0
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000.0

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2.0

    @property
    def is_locked(self) -> bool:
        """Check if market is locked (bid == ask)."""
        return abs(self.bid - self.ask) < 1e-9

    @property
    def is_crossed(self) -> bool:
        """Check if market is crossed (bid > ask)."""
        return self.bid > self.ask + 1e-9


@dataclass
class ValidationResult:
    """
    Result of order validation against market structure rules.

    Attributes:
        valid: Whether order passes all validations
        errors: List of error messages (if invalid)
        warnings: List of warning messages
        adjusted_price: Price adjusted to valid tick (if auto-adjust enabled)
        lot_type: Classified lot type
        trade_through: Trade-through violation type (if any)
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_price: Optional[float] = None
    lot_type: LotType = LotType.ROUND_LOT
    trade_through: TradeThrough = TradeThrough.NONE


# =============================================================================
# TICK SIZE ENFORCER
# =============================================================================


class TickSizeEnforcer:
    """
    Enforce tick size rules per SEC Reg NMS Rule 612.

    Rule 612 (Sub-Penny Rule):
    - Stocks priced >= $1.00: minimum tick size $0.01
    - Stocks priced < $1.00: minimum tick size $0.0001
    - Midpoint pegs and certain other orders may use sub-penny

    Usage:
        enforcer = TickSizeEnforcer()

        # Validate price
        if enforcer.is_valid_price(150.005, reference_price=150.00):
            ...

        # Round to valid tick
        valid_price = enforcer.round_to_tick(150.005, direction="down")

        # Get tick size for price
        tick = enforcer.get_tick_size(150.00)

    References:
        SEC Rule 612: https://www.ecfr.gov/current/title-17/section-242.612
    """

    def __init__(
        self,
        default_tick_size: float = TICK_SIZE_PENNY,
        sub_penny_threshold: float = TICK_PILOT_THRESHOLD,
        sub_penny_tick: float = TICK_SIZE_SUB_PENNY,
        allow_midpoint: bool = True,
    ) -> None:
        """
        Initialize tick size enforcer.

        Args:
            default_tick_size: Default tick size for stocks >= $1.00
            sub_penny_threshold: Price below which sub-penny is allowed
            sub_penny_tick: Tick size for sub-penny stocks
            allow_midpoint: Allow midpoint prices (half-tick)
        """
        self.default_tick_size = default_tick_size
        self.sub_penny_threshold = sub_penny_threshold
        self.sub_penny_tick = sub_penny_tick
        self.allow_midpoint = allow_midpoint

    def get_tick_size(self, price: float) -> float:
        """
        Get the tick size for a given price.

        Args:
            price: Reference price

        Returns:
            Applicable tick size
        """
        if price < self.sub_penny_threshold:
            return self.sub_penny_tick
        return self.default_tick_size

    def is_valid_price(
        self,
        price: float,
        reference_price: Optional[float] = None,
        is_midpoint: bool = False,
    ) -> bool:
        """
        Check if price is on a valid tick increment.

        Args:
            price: Price to validate
            reference_price: Reference price for tick determination (default: price itself)
            is_midpoint: Whether this is a midpoint order (allows half-tick)

        Returns:
            True if price is on valid tick
        """
        ref_price = reference_price if reference_price is not None else price
        tick = self.get_tick_size(ref_price)

        if is_midpoint and self.allow_midpoint:
            # Midpoint can be at half-tick
            tick = tick / 2.0

        # Check if price is on tick
        # Use modulo with tolerance for floating point
        remainder = abs(price % tick)
        tolerance = tick * 1e-9

        return remainder < tolerance or abs(remainder - tick) < tolerance

    def round_to_tick(
        self,
        price: float,
        reference_price: Optional[float] = None,
        direction: str = "nearest",
    ) -> float:
        """
        Round price to valid tick increment.

        Args:
            price: Price to round
            reference_price: Reference price for tick determination
            direction: Rounding direction ("nearest", "up", "down")

        Returns:
            Price rounded to valid tick

        Raises:
            ValueError: If invalid direction
        """
        ref_price = reference_price if reference_price is not None else price
        tick = self.get_tick_size(ref_price)

        # Use Decimal for precision
        price_dec = Decimal(str(price))
        tick_dec = Decimal(str(tick))

        if direction == "nearest":
            rounded = (price_dec / tick_dec).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_dec
        elif direction == "down":
            rounded = (price_dec / tick_dec).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_dec
        elif direction == "up":
            # Round up: if not on tick, go to next tick above
            base = (price_dec / tick_dec).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_dec
            if base < price_dec:
                rounded = base + tick_dec
            else:
                rounded = base
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'nearest', 'up', or 'down'.")

        return float(rounded)

    def validate_and_adjust(
        self,
        price: float,
        reference_price: Optional[float] = None,
        auto_adjust: bool = True,
    ) -> Tuple[bool, float, str]:
        """
        Validate price and optionally adjust to valid tick.

        Args:
            price: Price to validate
            reference_price: Reference price for tick determination
            auto_adjust: If True, return adjusted price

        Returns:
            Tuple of (is_valid, adjusted_price, message)
        """
        ref_price = reference_price if reference_price is not None else price

        if self.is_valid_price(price, ref_price):
            return True, price, ""

        if auto_adjust:
            adjusted = self.round_to_tick(price, ref_price, "nearest")
            tick = self.get_tick_size(ref_price)
            return False, adjusted, f"Price {price} adjusted to {adjusted} (tick={tick})"

        tick = self.get_tick_size(ref_price)
        return False, price, f"Price {price} not on valid tick ({tick})"


# =============================================================================
# ODD LOT CLASSIFIER
# =============================================================================


class OddLotClassifier:
    """
    Classify and handle odd lot orders per SEC Rule 600.

    Definitions:
    - Round lot: 100 shares (or multiples thereof)
    - Odd lot: < 100 shares
    - Mixed lot: Round lots + odd lot remainder (e.g., 350 shares)

    Odd lot handling considerations:
    - Odd lots don't contribute to NBBO quote
    - Odd lots may have different execution priority
    - Odd lots may have higher per-share costs
    - Some venues don't display odd lots

    Usage:
        classifier = OddLotClassifier()

        # Classify order
        lot_type = classifier.classify(qty=75)
        assert lot_type == LotType.ODD_LOT

        # Get round/odd breakdown
        round_qty, odd_qty = classifier.split_mixed(350)
        assert round_qty == 300 and odd_qty == 50

    References:
        SEC Rule 600(b)(49): https://www.ecfr.gov/current/title-17/section-242.600
    """

    def __init__(
        self,
        round_lot_size: int = ROUND_LOT_SIZE,
        handling: OddLotHandling = OddLotHandling.ALLOW,
    ) -> None:
        """
        Initialize odd lot classifier.

        Args:
            round_lot_size: Size of a round lot (default: 100)
            handling: How to handle odd lot orders
        """
        self.round_lot_size = round_lot_size
        self.handling = handling

    def classify(self, qty: float) -> LotType:
        """
        Classify order by lot type.

        Args:
            qty: Order quantity

        Returns:
            LotType classification
        """
        qty_int = int(qty)

        if qty_int < self.round_lot_size:
            return LotType.ODD_LOT
        elif qty_int % self.round_lot_size == 0:
            return LotType.ROUND_LOT
        else:
            return LotType.MIXED_LOT

    def split_mixed(self, qty: float) -> Tuple[int, int]:
        """
        Split mixed lot into round and odd components.

        Args:
            qty: Total quantity

        Returns:
            Tuple of (round_lot_qty, odd_lot_qty)
        """
        qty_int = int(qty)
        round_qty = (qty_int // self.round_lot_size) * self.round_lot_size
        odd_qty = qty_int % self.round_lot_size
        return round_qty, odd_qty

    def is_odd_lot(self, qty: float) -> bool:
        """Check if quantity is an odd lot."""
        return self.classify(qty) == LotType.ODD_LOT

    def is_round_lot(self, qty: float) -> bool:
        """Check if quantity is a round lot."""
        return self.classify(qty) == LotType.ROUND_LOT

    def validate(self, qty: float) -> Tuple[bool, str]:
        """
        Validate order quantity against odd lot handling rules.

        Args:
            qty: Order quantity

        Returns:
            Tuple of (is_valid, message)
        """
        lot_type = self.classify(qty)

        if self.handling == OddLotHandling.ALLOW:
            return True, ""

        elif self.handling == OddLotHandling.REJECT:
            if lot_type == LotType.ODD_LOT:
                return False, f"Odd lots rejected: {qty} < {self.round_lot_size}"
            return True, ""

        elif self.handling == OddLotHandling.COMBINE_ONLY:
            if lot_type == LotType.ODD_LOT:
                return False, f"Standalone odd lot rejected: {qty}"
            return True, ""

        elif self.handling == OddLotHandling.SEGREGATE:
            # All lots valid, but odd lots routed separately
            if lot_type == LotType.ODD_LOT:
                return True, "Routed to odd-lot book"
            return True, ""

        return True, ""


# =============================================================================
# REG NMS VALIDATOR
# =============================================================================


class RegNMSValidator:
    """
    Validate orders against Reg NMS Rule 611 (Order Protection Rule).

    Rule 611 prohibits trade-throughs:
    - Buying above the protected (best) offer
    - Selling below the protected (best) bid

    Exceptions to trade-through rule:
    - Intermarket sweep orders (ISO)
    - Flickering quotes (quote changed within 1 second)
    - Benchmark/VWAP orders
    - Stopped orders

    Usage:
        validator = RegNMSValidator()

        # Validate order against NBBO
        nbbo = NBBO(bid=100.00, bid_size=1000, ask=100.05, ask_size=500)
        result = validator.validate_order(
            side="BUY",
            price=100.10,  # Above ask
            nbbo=nbbo,
        )

        if not result.valid:
            print(f"Trade-through violation: {result.errors}")

    References:
        SEC Rule 611: https://www.ecfr.gov/current/title-17/section-242.611
    """

    def __init__(
        self,
        enforce_trade_through: bool = True,
        allow_iso: bool = True,
        flickering_quote_ms: int = 1000,
    ) -> None:
        """
        Initialize Reg NMS validator.

        Args:
            enforce_trade_through: If True, reject trade-through orders
            allow_iso: If True, allow intermarket sweep orders
            flickering_quote_ms: Time window for flickering quote exception (ms)
        """
        self.enforce_trade_through = enforce_trade_through
        self.allow_iso = allow_iso
        self.flickering_quote_ms = flickering_quote_ms

    def check_trade_through(
        self,
        side: str,
        price: float,
        nbbo: NBBO,
    ) -> TradeThrough:
        """
        Check if order would trade through NBBO.

        Args:
            side: Order side ("BUY" or "SELL")
            price: Order price
            nbbo: Current NBBO

        Returns:
            TradeThrough type (NONE if no violation)
        """
        side_upper = side.upper()

        if side_upper in ("BUY", "B", "BID"):
            # Buy order: cannot trade above protected ask
            if price > nbbo.ask + 1e-9:
                return TradeThrough.ASK_THROUGH
        else:
            # Sell order: cannot trade below protected bid
            if price < nbbo.bid - 1e-9:
                return TradeThrough.BID_THROUGH

        return TradeThrough.NONE

    def validate_order(
        self,
        side: str,
        price: float,
        qty: float,
        nbbo: NBBO,
        is_iso: bool = False,
        is_market: bool = False,
    ) -> ValidationResult:
        """
        Validate order against Reg NMS rules.

        Args:
            side: Order side
            price: Order price
            qty: Order quantity
            nbbo: Current NBBO
            is_iso: Is this an intermarket sweep order
            is_market: Is this a market order

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        # Market orders are typically allowed at NBBO
        if is_market:
            return result

        # ISO orders bypass trade-through
        if is_iso and self.allow_iso:
            result.warnings.append("ISO order: trade-through allowed")
            return result

        # Check for trade-through
        tt = self.check_trade_through(side, price, nbbo)
        result.trade_through = tt

        if tt != TradeThrough.NONE and self.enforce_trade_through:
            result.valid = False
            if tt == TradeThrough.ASK_THROUGH:
                result.errors.append(
                    f"Trade-through: BUY at {price} > protected ask {nbbo.ask}"
                )
            else:
                result.errors.append(
                    f"Trade-through: SELL at {price} < protected bid {nbbo.bid}"
                )

        return result

    def get_compliant_price(
        self,
        side: str,
        desired_price: float,
        nbbo: NBBO,
    ) -> float:
        """
        Get a Reg NMS compliant price for the order.

        Args:
            side: Order side
            desired_price: Desired price
            nbbo: Current NBBO

        Returns:
            Price adjusted to comply with trade-through rule
        """
        side_upper = side.upper()

        if side_upper in ("BUY", "B", "BID"):
            # Buy: cannot exceed ask
            return min(desired_price, nbbo.ask)
        else:
            # Sell: cannot be below bid
            return max(desired_price, nbbo.bid)


# =============================================================================
# UNIFIED MARKET STRUCTURE VALIDATOR
# =============================================================================


class USMarketStructureValidator:
    """
    Unified validator for all US equity market structure rules.

    Combines:
    - Tick size enforcement (Rule 612)
    - Odd lot classification (Rule 600)
    - NBBO protection (Rule 611)

    Usage:
        validator = USMarketStructureValidator()

        # Validate order
        result = validator.validate_order(
            side="BUY",
            price=150.005,
            qty=75,
            nbbo=nbbo,
        )

        if result.valid:
            # Order passes all market structure rules
            final_price = result.adjusted_price or price
        else:
            # Handle violations
            for error in result.errors:
                print(error)

    This is the recommended entry point for market structure validation.
    """

    def __init__(
        self,
        tick_enforcer: Optional[TickSizeEnforcer] = None,
        lot_classifier: Optional[OddLotClassifier] = None,
        nbbo_validator: Optional[RegNMSValidator] = None,
        auto_adjust_price: bool = True,
    ) -> None:
        """
        Initialize unified validator.

        Args:
            tick_enforcer: TickSizeEnforcer instance (created if None)
            lot_classifier: OddLotClassifier instance (created if None)
            nbbo_validator: RegNMSValidator instance (created if None)
            auto_adjust_price: If True, adjust prices to valid ticks
        """
        self.tick_enforcer = tick_enforcer or TickSizeEnforcer()
        self.lot_classifier = lot_classifier or OddLotClassifier()
        self.nbbo_validator = nbbo_validator or RegNMSValidator()
        self.auto_adjust_price = auto_adjust_price

    def validate_order(
        self,
        side: str,
        price: float,
        qty: float,
        nbbo: Optional[NBBO] = None,
        is_iso: bool = False,
        is_market: bool = False,
        reference_price: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate order against all market structure rules.

        Args:
            side: Order side ("BUY" or "SELL")
            price: Order price
            qty: Order quantity
            nbbo: Current NBBO (optional for tick/lot validation only)
            is_iso: Is this an intermarket sweep order
            is_market: Is this a market order
            reference_price: Reference price for tick determination

        Returns:
            ValidationResult with comprehensive validation
        """
        result = ValidationResult()

        # 1. Classify lot type
        result.lot_type = self.lot_classifier.classify(qty)
        lot_valid, lot_msg = self.lot_classifier.validate(qty)
        if not lot_valid:
            result.valid = False
            result.errors.append(lot_msg)
        elif lot_msg:
            result.warnings.append(lot_msg)

        # 2. Validate and adjust tick
        ref_price = reference_price or price
        tick_valid, adjusted_price, tick_msg = self.tick_enforcer.validate_and_adjust(
            price, ref_price, auto_adjust=self.auto_adjust_price
        )

        if not tick_valid:
            if self.auto_adjust_price:
                result.adjusted_price = adjusted_price
                result.warnings.append(tick_msg)
            else:
                result.valid = False
                result.errors.append(tick_msg)

        # 3. Validate NBBO (if provided)
        effective_price = result.adjusted_price or price

        if nbbo is not None:
            nbbo_result = self.nbbo_validator.validate_order(
                side=side,
                price=effective_price,
                qty=qty,
                nbbo=nbbo,
                is_iso=is_iso,
                is_market=is_market,
            )

            result.trade_through = nbbo_result.trade_through
            result.errors.extend(nbbo_result.errors)
            result.warnings.extend(nbbo_result.warnings)

            if not nbbo_result.valid:
                result.valid = False

        return result

    def create_compliant_order(
        self,
        side: str,
        price: float,
        qty: float,
        nbbo: Optional[NBBO] = None,
    ) -> Tuple[float, float, List[str]]:
        """
        Create a market structure compliant order.

        Adjusts price to:
        1. Valid tick increment
        2. Within NBBO (no trade-through)

        Args:
            side: Order side
            price: Desired price
            qty: Order quantity
            nbbo: Current NBBO

        Returns:
            Tuple of (compliant_price, qty, warnings)
        """
        warnings: List[str] = []

        # Round to valid tick
        adjusted_price = self.tick_enforcer.round_to_tick(price, price, "nearest")
        if abs(adjusted_price - price) > 1e-9:
            warnings.append(f"Price adjusted from {price} to {adjusted_price}")

        # Adjust for NBBO
        if nbbo is not None:
            compliant_price = self.nbbo_validator.get_compliant_price(
                side, adjusted_price, nbbo
            )
            if abs(compliant_price - adjusted_price) > 1e-9:
                warnings.append(
                    f"Price adjusted from {adjusted_price} to {compliant_price} for NBBO compliance"
                )
                adjusted_price = compliant_price

        return adjusted_price, qty, warnings


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_tick_enforcer(
    default_tick: float = TICK_SIZE_PENNY,
    allow_sub_penny: bool = True,
) -> TickSizeEnforcer:
    """Create configured tick size enforcer."""
    return TickSizeEnforcer(
        default_tick_size=default_tick,
        sub_penny_threshold=TICK_PILOT_THRESHOLD if allow_sub_penny else 0.0,
    )


def create_odd_lot_classifier(
    handling: Union[str, OddLotHandling] = OddLotHandling.ALLOW,
) -> OddLotClassifier:
    """Create configured odd lot classifier."""
    if isinstance(handling, str):
        handling = OddLotHandling(handling.lower())
    return OddLotClassifier(handling=handling)


def create_nbbo_validator(
    enforce: bool = True,
    allow_iso: bool = True,
) -> RegNMSValidator:
    """Create configured NBBO validator."""
    return RegNMSValidator(
        enforce_trade_through=enforce,
        allow_iso=allow_iso,
    )


def create_market_structure_validator(
    auto_adjust: bool = True,
    enforce_nbbo: bool = True,
    odd_lot_handling: str = "allow",
) -> USMarketStructureValidator:
    """
    Create fully configured market structure validator.

    Args:
        auto_adjust: Automatically adjust prices to valid ticks
        enforce_nbbo: Enforce trade-through rule
        odd_lot_handling: How to handle odd lots

    Returns:
        Configured USMarketStructureValidator
    """
    return USMarketStructureValidator(
        tick_enforcer=create_tick_enforcer(),
        lot_classifier=create_odd_lot_classifier(odd_lot_handling),
        nbbo_validator=create_nbbo_validator(enforce=enforce_nbbo),
        auto_adjust_price=auto_adjust,
    )
