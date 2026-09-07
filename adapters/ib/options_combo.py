# -*- coding: utf-8 -*-
"""
adapters/ib/options_combo.py
Interactive Brokers combo/spread order support for options.

Phase 2: US Exchange Adapters

Supports multi-leg options strategies:
- Vertical spreads (bull call, bear put, etc.)
- Calendar/horizontal spreads
- Diagonal spreads
- Straddles and strangles
- Iron condors
- Butterflies
- Custom combo orders

IB ComboLeg Specifics:
    - Each leg has: conId, ratio, action, exchange, openClose
    - Combo orders require a "BAG" contract type
    - Delta-neutral combos supported
    - Exchange can be SMART for best execution

Rate Limits:
    - Combo orders count against the 50/sec order limit
    - No additional rate limiting beyond standard orders

References:
    - IB Combo Orders: https://interactivebrokers.github.io/tws-api/combination_orders.html
    - IB Spread Orders: https://interactivebrokers.github.io/tws-api/spread_orders.html
    - ib_insync combos: https://ib-insync.readthedocs.io/api.html#ib_insync.contract.ComboLeg
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from adapters.base import OrderResult
from adapters.models import ExchangeVendor
from adapters.ib.market_data import IB_INSYNC_AVAILABLE
from adapters.ib.options import (
    IBOptionsOrderExecutionAdapter,
    OptionsContractSpec,
    OptionsOrder,
    OptionsOrderResult,
)
from core_options import (
    OptionType,
    ExerciseStyle,
    SettlementType,
    GreeksResult,
    OPTIONS_CONTRACT_MULTIPLIER,
)

logger = logging.getLogger(__name__)

# Conditional imports for ib_insync
if IB_INSYNC_AVAILABLE:
    from ib_insync import (
        IB,
        Option,
        Contract,
        ComboLeg,
        Order as IBOrder,
        MarketOrder,
        LimitOrder,
        Trade,
    )
else:
    IB = None
    Option = None
    Contract = None
    ComboLeg = None
    IBOrder = None


# =============================================================================
# Combo Strategy Types
# =============================================================================


class ComboStrategy(str, Enum):
    """
    Pre-defined options combo strategies.

    Based on standard CBOE multi-leg order types.
    """

    # Two-leg strategies
    VERTICAL_SPREAD = "vertical_spread"  # Same expiry, different strikes
    CALENDAR_SPREAD = "calendar_spread"  # Same strike, different expiries (horizontal)
    DIAGONAL_SPREAD = "diagonal_spread"  # Different strike and expiry
    STRADDLE = "straddle"  # Same strike call + put
    STRANGLE = "strangle"  # Different strike call + put
    COVERED_CALL = "covered_call"  # Long stock + short call
    PROTECTIVE_PUT = "protective_put"  # Long stock + long put
    COLLAR = "collar"  # Long stock + put + short call

    # Three-leg strategies
    BUTTERFLY = "butterfly"  # 1-2-1 ratio at 3 strikes
    RATIO_SPREAD = "ratio_spread"  # Unequal ratios (e.g., 1x2)

    # Four-leg strategies
    IRON_CONDOR = "iron_condor"  # Put spread + call spread
    IRON_BUTTERFLY = "iron_butterfly"  # Straddle + strangle wings
    BOX_SPREAD = "box_spread"  # Call spread + put spread at same strikes
    DOUBLE_DIAGONAL = "double_diagonal"  # Two diagonal spreads

    # Custom
    CUSTOM = "custom"  # User-defined legs


class LegAction(str, Enum):
    """Action for a combo leg."""

    BUY = "BUY"
    SELL = "SELL"


class OpenClose(str, Enum):
    """
    Open/Close designation for legs.

    Required for regulatory reporting (exchange rules).
    """

    OPEN = "O"  # Opening new position
    CLOSE = "C"  # Closing existing position
    SAME = "S"  # Same as previous (auto-determined)


# =============================================================================
# Combo Leg Definition
# =============================================================================


@dataclass
class OptionsComboLeg:
    """
    Single leg of a combo/spread order.

    Attributes:
        contract: Options contract specification
        action: BUY or SELL
        ratio: Number of contracts for this leg (usually 1)
        exchange: Exchange for this leg (default: SMART)
        open_close: Open/Close designation
        con_id: IB contract ID (filled automatically)
    """

    contract: OptionsContractSpec
    action: LegAction
    ratio: int = 1
    exchange: str = "SMART"
    open_close: OpenClose = OpenClose.SAME
    con_id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.action, str):
            self.action = LegAction(self.action.upper())
        if isinstance(self.open_close, str):
            self.open_close = OpenClose(self.open_close.upper())

    @property
    def is_long(self) -> bool:
        """Check if leg is long."""
        return self.action == LegAction.BUY

    @property
    def is_short(self) -> bool:
        """Check if leg is short."""
        return self.action == LegAction.SELL

    def to_ib_combo_leg(self, con_id: int) -> "ComboLeg":
        """
        Convert to IB ComboLeg.

        Args:
            con_id: IB contract ID for this leg

        Returns:
            IB ComboLeg object
        """
        if ComboLeg is None:
            raise ImportError("ib_insync not available")

        return ComboLeg(
            conId=con_id,
            ratio=self.ratio,
            action=self.action.value,
            exchange=self.exchange,
            openClose=self.open_close.value,
        )


# =============================================================================
# Combo Order Definition
# =============================================================================


@dataclass
class OptionsComboOrder:
    """
    Multi-leg combo/spread order.

    Attributes:
        legs: List of combo legs
        strategy: Named strategy type
        order_type: MARKET or LIMIT
        limit_price: Net limit price for the combo (credit or debit)
        underlying: Underlying symbol
        time_in_force: Order duration (DAY, GTC, etc.)
        exchange: Exchange routing (default: SMART)
        client_order_id: Optional client order ID
    """

    legs: List[OptionsComboLeg]
    strategy: ComboStrategy = ComboStrategy.CUSTOM
    order_type: str = "LIMIT"
    limit_price: Optional[Decimal] = None
    underlying: str = ""
    time_in_force: str = "DAY"
    exchange: str = "SMART"
    client_order_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.strategy, str):
            self.strategy = ComboStrategy(self.strategy.lower())
        if isinstance(self.limit_price, (int, float)):
            self.limit_price = Decimal(str(self.limit_price))

        # Infer underlying from legs if not set
        if not self.underlying and self.legs:
            self.underlying = self.legs[0].contract.underlying

    @property
    def num_legs(self) -> int:
        """Number of legs in the combo."""
        return len(self.legs)

    @property
    def total_contracts(self) -> int:
        """Total contracts across all legs."""
        return sum(leg.ratio for leg in self.legs)

    @property
    def is_credit(self) -> bool:
        """Check if combo is a net credit (receiving premium)."""
        if self.limit_price is None:
            return False
        return self.limit_price > 0

    @property
    def is_debit(self) -> bool:
        """Check if combo is a net debit (paying premium)."""
        if self.limit_price is None:
            return False
        return self.limit_price < 0

    @property
    def expirations(self) -> List[date]:
        """Get unique expiration dates."""
        return sorted(set(leg.contract.expiration for leg in self.legs))

    @property
    def strikes(self) -> List[Decimal]:
        """Get unique strike prices."""
        return sorted(set(leg.contract.strike for leg in self.legs))

    def get_net_delta(self) -> Optional[float]:
        """
        Calculate net position delta.

        Returns None if Greeks not available.
        """
        total_delta = 0.0
        for leg in self.legs:
            if leg.contract.delta is None:
                return None
            sign = 1.0 if leg.is_long else -1.0
            total_delta += sign * leg.contract.delta * leg.ratio
        return total_delta

    def get_max_profit(self) -> Optional[Decimal]:
        """
        Calculate maximum profit potential.

        Returns None if cannot be determined.

        Note:
            Currently implemented for IRON_CONDOR only.
            Other strategies return None (conservative - no false profit estimates).
            This is tracked as low-priority tech debt.

        Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#options-max-profit
        """
        # Strategy-specific calculations
        if self.strategy == ComboStrategy.IRON_CONDOR:
            # Max profit is net credit received
            return self.limit_price if self.is_credit else None
        # Other strategies: return None (conservative)
        # Tracking: Low priority - additional strategies can be implemented on demand
        # Control: Returning None is safe (no false profit estimates)
        return None

    def get_max_loss(self) -> Optional[Decimal]:
        """
        Calculate maximum loss potential.

        Returns None if cannot be determined.
        """
        # Strategy-specific calculations
        if self.strategy == ComboStrategy.IRON_CONDOR:
            if len(self.strikes) != 4:
                return None
            strikes_sorted = sorted(self.strikes)
            # Width between put strikes or call strikes, minus credit
            width = strikes_sorted[1] - strikes_sorted[0]
            credit = self.limit_price if self.limit_price else Decimal(0)
            return (width * OPTIONS_CONTRACT_MULTIPLIER) - credit
        return None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate combo order configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.legs:
            return False, "At least one leg required"

        if len(self.legs) > 8:
            return False, "Maximum 8 legs supported by IB"

        if self.order_type == "LIMIT" and self.limit_price is None:
            return False, "Limit price required for limit orders"

        # Validate each leg
        for i, leg in enumerate(self.legs):
            if leg.ratio <= 0:
                return False, f"Leg {i+1}: ratio must be positive"
            if not leg.contract.underlying:
                return False, f"Leg {i+1}: underlying required"

        # Check underlying consistency
        underlyings = set(leg.contract.underlying for leg in self.legs)
        if len(underlyings) > 1:
            return False, f"All legs must have same underlying, found: {underlyings}"

        return True, None


@dataclass
class OptionsComboOrderResult:
    """
    Result of combo order submission.

    Attributes:
        success: Whether order was submitted successfully
        order_id: IB order ID
        status: Order status
        filled_qty: Number of combos filled
        avg_fill_price: Average fill price (net credit/debit)
        leg_fills: Individual leg fill details
        commission: Total commission
        error_message: Error message if failed
    """

    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: str = "UNKNOWN"
    filled_qty: int = 0
    avg_fill_price: Optional[Decimal] = None
    leg_fills: List[Dict[str, Any]] = field(default_factory=list)
    commission: Decimal = Decimal(0)
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Strategy Builders
# =============================================================================


class ComboStrategyBuilder:
    """
    Factory for creating standard combo strategies.

    Each method returns a list of OptionsComboLeg objects
    configured for the specific strategy.
    """

    @staticmethod
    def vertical_spread(
        underlying: str,
        expiration: date,
        option_type: OptionType,
        long_strike: Decimal,
        short_strike: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a vertical spread (same expiry, different strikes).

        Bull Call Spread: Buy lower strike call, sell higher strike call
        Bear Put Spread: Buy higher strike put, sell lower strike put

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            option_type: CALL or PUT
            long_strike: Strike price to buy
            short_strike: Strike price to sell
            qty: Number of spreads

        Returns:
            List of OptionsComboLeg
        """
        long_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=long_strike,
            option_type=option_type,
        )
        short_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=short_strike,
            option_type=option_type,
        )

        return [
            OptionsComboLeg(contract=long_contract, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=short_contract, action=LegAction.SELL, ratio=qty),
        ]

    @staticmethod
    def calendar_spread(
        underlying: str,
        strike: Decimal,
        option_type: OptionType,
        near_expiration: date,
        far_expiration: date,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a calendar (horizontal) spread.

        Sell near-term, buy far-term at same strike.

        Args:
            underlying: Underlying symbol
            strike: Strike price
            option_type: CALL or PUT
            near_expiration: Near-term expiration (sell)
            far_expiration: Far-term expiration (buy)
            qty: Number of spreads

        Returns:
            List of OptionsComboLeg
        """
        near_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=near_expiration,
            strike=strike,
            option_type=option_type,
        )
        far_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=far_expiration,
            strike=strike,
            option_type=option_type,
        )

        return [
            OptionsComboLeg(contract=near_contract, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=far_contract, action=LegAction.BUY, ratio=qty),
        ]

    @staticmethod
    def diagonal_spread(
        underlying: str,
        option_type: OptionType,
        near_expiration: date,
        near_strike: Decimal,
        far_expiration: date,
        far_strike: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a diagonal spread.

        Combines features of vertical and calendar spreads.

        Args:
            underlying: Underlying symbol
            option_type: CALL or PUT
            near_expiration: Near-term expiration (sell)
            near_strike: Near-term strike (sell)
            far_expiration: Far-term expiration (buy)
            far_strike: Far-term strike (buy)
            qty: Number of spreads

        Returns:
            List of OptionsComboLeg
        """
        near_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=near_expiration,
            strike=near_strike,
            option_type=option_type,
        )
        far_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=far_expiration,
            strike=far_strike,
            option_type=option_type,
        )

        return [
            OptionsComboLeg(contract=near_contract, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=far_contract, action=LegAction.BUY, ratio=qty),
        ]

    @staticmethod
    def straddle(
        underlying: str,
        expiration: date,
        strike: Decimal,
        direction: str = "long",
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a straddle (call + put at same strike).

        Long straddle: Buy call + buy put (volatility play)
        Short straddle: Sell call + sell put (collect premium)

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price (usually ATM)
            direction: "long" or "short"
            qty: Number of straddles

        Returns:
            List of OptionsComboLeg
        """
        call_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=OptionType.CALL,
        )
        put_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=OptionType.PUT,
        )

        action = LegAction.BUY if direction.lower() == "long" else LegAction.SELL

        return [
            OptionsComboLeg(contract=call_contract, action=action, ratio=qty),
            OptionsComboLeg(contract=put_contract, action=action, ratio=qty),
        ]

    @staticmethod
    def strangle(
        underlying: str,
        expiration: date,
        call_strike: Decimal,
        put_strike: Decimal,
        direction: str = "long",
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a strangle (call + put at different strikes).

        Long strangle: Buy OTM call + buy OTM put
        Short strangle: Sell OTM call + sell OTM put

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            call_strike: Call strike price (above current price)
            put_strike: Put strike price (below current price)
            direction: "long" or "short"
            qty: Number of strangles

        Returns:
            List of OptionsComboLeg
        """
        call_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=call_strike,
            option_type=OptionType.CALL,
        )
        put_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=put_strike,
            option_type=OptionType.PUT,
        )

        action = LegAction.BUY if direction.lower() == "long" else LegAction.SELL

        return [
            OptionsComboLeg(contract=call_contract, action=action, ratio=qty),
            OptionsComboLeg(contract=put_contract, action=action, ratio=qty),
        ]

    @staticmethod
    def iron_condor(
        underlying: str,
        expiration: date,
        put_long_strike: Decimal,
        put_short_strike: Decimal,
        call_short_strike: Decimal,
        call_long_strike: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create an iron condor (4-leg neutral strategy).

        Structure:
        - Buy OTM put (protection)
        - Sell OTM put (credit)
        - Sell OTM call (credit)
        - Buy OTM call (protection)

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            put_long_strike: Long put strike (lowest)
            put_short_strike: Short put strike
            call_short_strike: Short call strike
            call_long_strike: Long call strike (highest)
            qty: Number of iron condors

        Returns:
            List of OptionsComboLeg
        """
        put_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=put_long_strike,
            option_type=OptionType.PUT,
        )
        put_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=put_short_strike,
            option_type=OptionType.PUT,
        )
        call_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=call_short_strike,
            option_type=OptionType.CALL,
        )
        call_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=call_long_strike,
            option_type=OptionType.CALL,
        )

        return [
            OptionsComboLeg(contract=put_long, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=put_short, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=call_short, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=call_long, action=LegAction.BUY, ratio=qty),
        ]

    @staticmethod
    def iron_butterfly(
        underlying: str,
        expiration: date,
        center_strike: Decimal,
        wing_width: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create an iron butterfly.

        Structure (short iron butterfly):
        - Buy OTM put (wing)
        - Sell ATM put (body)
        - Sell ATM call (body)
        - Buy OTM call (wing)

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            center_strike: ATM strike for body
            wing_width: Distance to wing strikes
            qty: Number of iron butterflies

        Returns:
            List of OptionsComboLeg
        """
        put_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=center_strike - wing_width,
            option_type=OptionType.PUT,
        )
        put_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=center_strike,
            option_type=OptionType.PUT,
        )
        call_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=center_strike,
            option_type=OptionType.CALL,
        )
        call_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=center_strike + wing_width,
            option_type=OptionType.CALL,
        )

        return [
            OptionsComboLeg(contract=put_long, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=put_short, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=call_short, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=call_long, action=LegAction.BUY, ratio=qty),
        ]

    @staticmethod
    def butterfly(
        underlying: str,
        expiration: date,
        option_type: OptionType,
        lower_strike: Decimal,
        middle_strike: Decimal,
        upper_strike: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a butterfly spread (3-strike, 1-2-1 ratio).

        Long butterfly:
        - Buy 1 lower strike
        - Sell 2 middle strike
        - Buy 1 upper strike

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            option_type: CALL or PUT
            lower_strike: Lower wing strike
            middle_strike: Body strike (usually ATM)
            upper_strike: Upper wing strike
            qty: Number of butterflies

        Returns:
            List of OptionsComboLeg
        """
        lower_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=lower_strike,
            option_type=option_type,
        )
        middle_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=middle_strike,
            option_type=option_type,
        )
        upper_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=upper_strike,
            option_type=option_type,
        )

        return [
            OptionsComboLeg(contract=lower_contract, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=middle_contract, action=LegAction.SELL, ratio=qty * 2),
            OptionsComboLeg(contract=upper_contract, action=LegAction.BUY, ratio=qty),
        ]

    @staticmethod
    def box_spread(
        underlying: str,
        expiration: date,
        lower_strike: Decimal,
        upper_strike: Decimal,
        qty: int = 1,
    ) -> List[OptionsComboLeg]:
        """
        Create a box spread (arbitrage strategy).

        Structure:
        - Bull call spread (buy lower call, sell upper call)
        - Bear put spread (buy upper put, sell lower put)

        The value at expiration = strike difference × 100.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            lower_strike: Lower strike
            upper_strike: Upper strike
            qty: Number of boxes

        Returns:
            List of OptionsComboLeg
        """
        call_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=lower_strike,
            option_type=OptionType.CALL,
        )
        call_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=upper_strike,
            option_type=OptionType.CALL,
        )
        put_long = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=upper_strike,
            option_type=OptionType.PUT,
        )
        put_short = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=lower_strike,
            option_type=OptionType.PUT,
        )

        return [
            OptionsComboLeg(contract=call_long, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=call_short, action=LegAction.SELL, ratio=qty),
            OptionsComboLeg(contract=put_long, action=LegAction.BUY, ratio=qty),
            OptionsComboLeg(contract=put_short, action=LegAction.SELL, ratio=qty),
        ]

    @staticmethod
    def ratio_spread(
        underlying: str,
        expiration: date,
        option_type: OptionType,
        long_strike: Decimal,
        short_strike: Decimal,
        long_qty: int = 1,
        short_qty: int = 2,
    ) -> List[OptionsComboLeg]:
        """
        Create a ratio spread (unequal leg ratios).

        Example 1x2: Buy 1 at lower strike, sell 2 at higher strike.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            option_type: CALL or PUT
            long_strike: Long leg strike
            short_strike: Short leg strike
            long_qty: Quantity to buy
            short_qty: Quantity to sell

        Returns:
            List of OptionsComboLeg
        """
        long_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=long_strike,
            option_type=option_type,
        )
        short_contract = OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=short_strike,
            option_type=option_type,
        )

        return [
            OptionsComboLeg(contract=long_contract, action=LegAction.BUY, ratio=long_qty),
            OptionsComboLeg(contract=short_contract, action=LegAction.SELL, ratio=short_qty),
        ]


# =============================================================================
# IB Options Combo Execution Adapter
# =============================================================================


class IBOptionsComboAdapter:
    """
    IB adapter for combo/spread order execution.

    Extends IBOptionsOrderExecutionAdapter with multi-leg capabilities.
    """

    def __init__(
        self,
        execution_adapter: IBOptionsOrderExecutionAdapter,
    ) -> None:
        """
        Initialize combo adapter.

        Args:
            execution_adapter: Base IB options execution adapter
        """
        self._adapter = execution_adapter
        self._ib = None

    @property
    def ib(self):
        """Get IB connection from base adapter."""
        if self._ib is None:
            self._ib = self._adapter._ib
        return self._ib

    def _ensure_connected(self) -> None:
        """Ensure connection to IB is active."""
        if self.ib is None or not self.ib.isConnected():
            self._adapter._do_connect()
            self._ib = self._adapter._ib

    def _get_contract_con_id(self, spec: OptionsContractSpec) -> Optional[int]:
        """
        Get IB contract ID for an options contract.

        Args:
            spec: Options contract specification

        Returns:
            IB contract ID or None if not found
        """
        if not IB_INSYNC_AVAILABLE:
            return None

        self._ensure_connected()

        try:
            contract = Option(
                symbol=spec.underlying,
                lastTradeDateOrContractMonth=spec.expiration.strftime("%Y%m%d"),
                strike=float(spec.strike),
                right="C" if spec.option_type == OptionType.CALL else "P",
                exchange="SMART",
            )

            # Qualify contract to get conId
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                return qualified[0].conId
            return None

        except Exception as e:
            logger.error(f"Failed to get contract ID: {e}")
            return None

    def _build_combo_contract(
        self,
        legs: List[OptionsComboLeg],
        underlying: str,
    ) -> Optional["Contract"]:
        """
        Build IB BAG contract for combo order.

        Args:
            legs: List of combo legs
            underlying: Underlying symbol

        Returns:
            IB Contract (BAG type) or None if failed
        """
        if not IB_INSYNC_AVAILABLE:
            return None

        combo_legs = []

        for leg in legs:
            con_id = leg.con_id or self._get_contract_con_id(leg.contract)
            if con_id is None:
                logger.error(f"Failed to get conId for leg: {leg.contract}")
                return None

            combo_legs.append(leg.to_ib_combo_leg(con_id))

        # Create BAG contract
        contract = Contract()
        contract.symbol = underlying
        contract.secType = "BAG"
        contract.currency = "USD"
        contract.exchange = "SMART"
        contract.comboLegs = combo_legs

        return contract

    def submit_combo_order(
        self,
        combo_order: OptionsComboOrder,
    ) -> OptionsComboOrderResult:
        """
        Submit a multi-leg combo order to IB.

        Args:
            combo_order: Combo order specification

        Returns:
            OptionsComboOrderResult with fill details
        """
        # Validate order
        is_valid, error = combo_order.validate()
        if not is_valid:
            return OptionsComboOrderResult(
                success=False,
                error_message=error,
            )

        if not IB_INSYNC_AVAILABLE:
            return OptionsComboOrderResult(
                success=False,
                error_message="ib_insync not available",
            )

        self._ensure_connected()

        try:
            # Build combo contract
            contract = self._build_combo_contract(
                combo_order.legs,
                combo_order.underlying,
            )

            if contract is None:
                return OptionsComboOrderResult(
                    success=False,
                    error_message="Failed to build combo contract",
                )

            # Create order
            if combo_order.order_type.upper() == "MARKET":
                ib_order = MarketOrder(
                    action="BUY",  # Combo direction determined by legs
                    totalQuantity=1,
                    tif=combo_order.time_in_force,
                )
            else:
                ib_order = LimitOrder(
                    action="BUY",
                    totalQuantity=1,
                    lmtPrice=float(combo_order.limit_price),
                    tif=combo_order.time_in_force,
                )

            if combo_order.client_order_id:
                ib_order.orderRef = combo_order.client_order_id

            # Submit order
            trade = self.ib.placeOrder(contract, ib_order)

            # Wait for acknowledgment
            self.ib.sleep(0.5)

            return OptionsComboOrderResult(
                success=True,
                order_id=str(trade.order.orderId),
                client_order_id=combo_order.client_order_id,
                status=trade.orderStatus.status,
                filled_qty=int(trade.orderStatus.filled),
                avg_fill_price=(
                    Decimal(str(trade.orderStatus.avgFillPrice))
                    if trade.orderStatus.avgFillPrice
                    else None
                ),
                commission=(
                    Decimal(str(trade.orderStatus.commission))
                    if trade.orderStatus.commission
                    else Decimal(0)
                ),
                raw_response={"trade": str(trade)},
            )

        except Exception as e:
            logger.error(f"Failed to submit combo order: {e}")
            return OptionsComboOrderResult(
                success=False,
                error_message=str(e),
            )

    def submit_vertical_spread(
        self,
        underlying: str,
        expiration: date,
        option_type: OptionType,
        long_strike: Decimal,
        short_strike: Decimal,
        qty: int = 1,
        limit_price: Optional[Decimal] = None,
    ) -> OptionsComboOrderResult:
        """
        Submit a vertical spread order.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            option_type: CALL or PUT
            long_strike: Strike to buy
            short_strike: Strike to sell
            qty: Number of spreads
            limit_price: Net debit/credit limit

        Returns:
            OptionsComboOrderResult
        """
        legs = ComboStrategyBuilder.vertical_spread(
            underlying=underlying,
            expiration=expiration,
            option_type=option_type,
            long_strike=long_strike,
            short_strike=short_strike,
            qty=qty,
        )

        order = OptionsComboOrder(
            legs=legs,
            strategy=ComboStrategy.VERTICAL_SPREAD,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
            underlying=underlying,
        )

        return self.submit_combo_order(order)

    def submit_iron_condor(
        self,
        underlying: str,
        expiration: date,
        put_long_strike: Decimal,
        put_short_strike: Decimal,
        call_short_strike: Decimal,
        call_long_strike: Decimal,
        qty: int = 1,
        limit_price: Optional[Decimal] = None,
    ) -> OptionsComboOrderResult:
        """
        Submit an iron condor order.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            put_long_strike: Long put strike
            put_short_strike: Short put strike
            call_short_strike: Short call strike
            call_long_strike: Long call strike
            qty: Number of iron condors
            limit_price: Net credit limit

        Returns:
            OptionsComboOrderResult
        """
        legs = ComboStrategyBuilder.iron_condor(
            underlying=underlying,
            expiration=expiration,
            put_long_strike=put_long_strike,
            put_short_strike=put_short_strike,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            qty=qty,
        )

        order = OptionsComboOrder(
            legs=legs,
            strategy=ComboStrategy.IRON_CONDOR,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
            underlying=underlying,
        )

        return self.submit_combo_order(order)

    def submit_straddle(
        self,
        underlying: str,
        expiration: date,
        strike: Decimal,
        direction: str = "long",
        qty: int = 1,
        limit_price: Optional[Decimal] = None,
    ) -> OptionsComboOrderResult:
        """
        Submit a straddle order.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            direction: "long" or "short"
            qty: Number of straddles
            limit_price: Net limit price

        Returns:
            OptionsComboOrderResult
        """
        legs = ComboStrategyBuilder.straddle(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            direction=direction,
            qty=qty,
        )

        order = OptionsComboOrder(
            legs=legs,
            strategy=ComboStrategy.STRADDLE,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
            underlying=underlying,
        )

        return self.submit_combo_order(order)

    def submit_butterfly(
        self,
        underlying: str,
        expiration: date,
        option_type: OptionType,
        lower_strike: Decimal,
        middle_strike: Decimal,
        upper_strike: Decimal,
        qty: int = 1,
        limit_price: Optional[Decimal] = None,
    ) -> OptionsComboOrderResult:
        """
        Submit a butterfly spread order.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            option_type: CALL or PUT
            lower_strike: Lower wing strike
            middle_strike: Body strike
            upper_strike: Upper wing strike
            qty: Number of butterflies
            limit_price: Net debit limit

        Returns:
            OptionsComboOrderResult
        """
        legs = ComboStrategyBuilder.butterfly(
            underlying=underlying,
            expiration=expiration,
            option_type=option_type,
            lower_strike=lower_strike,
            middle_strike=middle_strike,
            upper_strike=upper_strike,
            qty=qty,
        )

        order = OptionsComboOrder(
            legs=legs,
            strategy=ComboStrategy.BUTTERFLY,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
            underlying=underlying,
        )

        return self.submit_combo_order(order)

    def get_combo_margin_requirement(
        self,
        combo_order: OptionsComboOrder,
    ) -> Optional[Dict[str, Any]]:
        """
        Get margin requirement for a combo order (What-If).

        Args:
            combo_order: Combo order to check

        Returns:
            Dictionary with margin details or None
        """
        if not IB_INSYNC_AVAILABLE:
            return None

        self._ensure_connected()

        try:
            contract = self._build_combo_contract(
                combo_order.legs,
                combo_order.underlying,
            )

            if contract is None:
                return None

            # Create a hypothetical order for what-if
            ib_order = LimitOrder(
                action="BUY",
                totalQuantity=1,
                lmtPrice=float(combo_order.limit_price or 0),
            )
            ib_order.whatIf = True

            # Place what-if order
            trade = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(0.5)

            if trade.orderStatus:
                return {
                    "initial_margin": trade.orderStatus.initMarginChange,
                    "maintenance_margin": trade.orderStatus.maintMarginChange,
                    "equity_with_loan": trade.orderStatus.equityWithLoanChange,
                    "commission": trade.orderStatus.commission,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get combo margin: {e}")
            return None

    def cancel_combo_order(self, order_id: str) -> bool:
        """
        Cancel a combo order.

        Args:
            order_id: IB order ID

        Returns:
            True if cancelled, False otherwise
        """
        self._ensure_connected()

        try:
            # Find the trade by order ID
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel combo order: {e}")
            return False


# =============================================================================
# Factory Functions
# =============================================================================


def create_ib_options_combo_adapter(
    execution_adapter: IBOptionsOrderExecutionAdapter,
) -> IBOptionsComboAdapter:
    """
    Create IB options combo adapter.

    Args:
        execution_adapter: Base execution adapter

    Returns:
        IBOptionsComboAdapter instance
    """
    return IBOptionsComboAdapter(execution_adapter)


def create_combo_order(
    strategy: ComboStrategy,
    legs: List[OptionsComboLeg],
    limit_price: Optional[Decimal] = None,
    time_in_force: str = "DAY",
) -> OptionsComboOrder:
    """
    Create a combo order from legs.

    Args:
        strategy: Strategy type
        legs: List of combo legs
        limit_price: Net limit price
        time_in_force: Order duration

    Returns:
        OptionsComboOrder instance
    """
    underlying = legs[0].contract.underlying if legs else ""

    return OptionsComboOrder(
        legs=legs,
        strategy=strategy,
        order_type="LIMIT" if limit_price else "MARKET",
        limit_price=limit_price,
        underlying=underlying,
        time_in_force=time_in_force,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ComboStrategy",
    "LegAction",
    "OpenClose",
    # Data classes
    "OptionsComboLeg",
    "OptionsComboOrder",
    "OptionsComboOrderResult",
    # Strategy builder
    "ComboStrategyBuilder",
    # Adapter
    "IBOptionsComboAdapter",
    # Factory functions
    "create_ib_options_combo_adapter",
    "create_combo_order",
]
