# -*- coding: utf-8 -*-
"""
impl_futures_margin.py
Futures margin calculation engine.

Implements margin calculation for:
- Binance USDT-M Perpetual/Quarterly (Tiered brackets)
- CME Index Futures (Simplified SPAN)
- Commodity & Currency Futures

Design Principles:
- Unified interface (FuturesMarginCalculator protocol)
- Vendor-specific implementations
- Decimal precision for financial calculations
- Thread-safe (immutable results)

Key Formulas (Binance):
- Initial Margin (IM) = Notional / Leverage
- Maintenance Margin (MM) = Notional * MMR (from bracket)
- Liquidation Price (Long) = Entry * (1 - IM% + MM%)
- Liquidation Price (Short) = Entry * (1 + IM% - MM%)

Key Formulas (CME - Simplified SPAN):
- Initial Margin = Fixed amount per contract (from exchange)
- Maintenance Margin = ~90% of Initial
- Portfolio offsets for correlated positions

References:
- Binance Leverage & Margin: https://www.binance.com/en/support/faq/360033162192
- Binance Liquidation: https://www.binance.com/en/support/faq/360033525271
- CME SPAN: https://www.cmegroup.com/clearing/span-methodology.html
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from enum import Enum
from typing import List, Optional, Dict, Any, Protocol, Tuple
import json
import logging

from core_futures import (
    FuturesType,
    FuturesContractSpec,
    FuturesPosition,
    LeverageBracket,
    MarginRequirement,
    MarginMode,
    PositionSide,
)


logger = logging.getLogger(__name__)


# ============================================================================
# PROTOCOL INTERFACE
# ============================================================================

class FuturesMarginCalculator(Protocol):
    """
    Protocol for margin calculation providers.

    Implementations:
    - TieredMarginCalculator: Binance tiered brackets
    - CMEMarginCalculator: CME SPAN-like margin
    - SimpleMarginCalculator: Fixed percentage margin
    """

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """
        Calculate initial margin requirement.

        Args:
            notional: Position notional value
            leverage: Leverage multiplier

        Returns:
            Initial margin amount
        """
        ...

    def calculate_maintenance_margin(
        self,
        notional: Decimal,
    ) -> Decimal:
        """
        Calculate maintenance margin requirement.

        Args:
            notional: Position notional value

        Returns:
            Maintenance margin amount
        """
        ...

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: MarginMode,
        isolated_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate liquidation price for position.

        Args:
            entry_price: Position entry price
            qty: Position quantity (positive=long, negative=short)
            leverage: Position leverage
            wallet_balance: Total wallet balance (cross margin)
            margin_mode: Cross or isolated margin
            isolated_margin: Isolated margin amount (if applicable)

        Returns:
            Liquidation price
        """
        ...

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate current margin ratio.

        MR < 1 = Liquidation zone

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance

        Returns:
            Margin ratio (> 1 = safe, < 1 = liquidation)
        """
        ...

    def get_max_leverage(self, notional: Decimal) -> int:
        """
        Get maximum allowed leverage for notional size.

        Args:
            notional: Position notional value

        Returns:
            Maximum leverage allowed
        """
        ...


# ============================================================================
# TIERED MARGIN CALCULATOR (Binance)
# ============================================================================

class TieredMarginCalculator:
    """
    Binance-style tiered margin calculator.

    Uses leverage brackets where larger positions require more margin.
    Each bracket specifies:
    - notional_cap: Maximum notional for this bracket
    - maint_margin_rate: Maintenance margin rate
    - max_leverage: Maximum leverage allowed

    Example BTCUSDT brackets:
    - Bracket 1: <$50K, MMR=0.4%, leverage=125x
    - Bracket 2: <$250K, MMR=0.5%, leverage=100x
    - ...
    - Bracket 10: >$300M, MMR=50%, leverage=1x

    Reference:
    https://www.binance.com/en/support/faq/leverage-margin-and-maintenance-margin-ratio-360033162192
    """

    def __init__(
        self,
        brackets: List[LeverageBracket],
        liquidation_fee_rate: Decimal = Decimal("0.005"),
    ):
        """
        Initialize with leverage brackets.

        Args:
            brackets: List of leverage brackets (sorted by notional_cap)
            liquidation_fee_rate: Liquidation fee rate (default 0.5%)
        """
        if not brackets:
            raise ValueError("At least one leverage bracket required")

        # Sort brackets by notional cap ascending
        self._brackets = sorted(brackets, key=lambda x: x.notional_cap)
        self._liquidation_fee_rate = liquidation_fee_rate

        # Pre-compute cumulative maintenance amounts for efficient lookup
        self._precompute_cumulative_maintenance()

    def _precompute_cumulative_maintenance(self) -> None:
        """
        Precompute cumulative maintenance for each bracket.

        Cumulative maintenance is used in the Binance liquidation formula:
        cum_maint[i] = sum(prev_bracket_size * prev_bracket_mmr)
        """
        self._cum_maintenance: List[Decimal] = []
        prev_cap = Decimal("0")
        cum = Decimal("0")

        for bracket in self._brackets:
            self._cum_maintenance.append(cum)
            bracket_size = bracket.notional_cap - prev_cap
            cum += bracket_size * bracket.maint_margin_rate
            prev_cap = bracket.notional_cap

    def _find_bracket(self, notional: Decimal) -> Tuple[int, LeverageBracket]:
        """
        Find applicable bracket for notional.

        Args:
            notional: Position notional

        Returns:
            (bracket_index, bracket)
        """
        for i, bracket in enumerate(self._brackets):
            if notional <= bracket.notional_cap:
                return i, bracket
        # Return last bracket for very large positions
        return len(self._brackets) - 1, self._brackets[-1]

    def get_maintenance_margin_rate(self, notional: Decimal) -> Decimal:
        """
        Get maintenance margin rate for notional value.

        Tiered system - higher notional = higher MMR.

        Args:
            notional: Position notional

        Returns:
            Maintenance margin rate (e.g., 0.004 = 0.4%)
        """
        _, bracket = self._find_bracket(abs(notional))
        return bracket.maint_margin_rate

    def get_max_leverage(self, notional: Decimal) -> int:
        """
        Get maximum leverage for notional size.

        Args:
            notional: Position notional

        Returns:
            Maximum leverage allowed
        """
        _, bracket = self._find_bracket(abs(notional))
        return bracket.max_leverage

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """
        Calculate initial margin requirement.

        IM = Notional / Leverage

        Args:
            notional: Position notional value
            leverage: Leverage multiplier

        Returns:
            Initial margin amount
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        abs_notional = abs(notional)

        # Check max leverage
        max_lev = self.get_max_leverage(abs_notional)
        effective_leverage = min(leverage, max_lev)

        if effective_leverage != leverage:
            logger.warning(
                f"Leverage {leverage}x exceeds max {max_lev}x for notional {abs_notional}, "
                f"using {effective_leverage}x"
            )

        return abs_notional / Decimal(effective_leverage)

    def calculate_maintenance_margin(
        self,
        notional: Decimal,
    ) -> Decimal:
        """
        Calculate maintenance margin requirement.

        MM = Notional * MMR

        For accurate Binance-style calculation with cumulative:
        MM = Notional * MMR - Cumulative

        Args:
            notional: Position notional value

        Returns:
            Maintenance margin amount
        """
        abs_notional = abs(notional)
        idx, bracket = self._find_bracket(abs_notional)

        # Simple calculation: Notional * MMR
        # Note: Binance uses more complex formula with cumulative for exact liquidation
        # For simplicity, we use the simpler formula which is slightly conservative
        return abs_notional * bracket.maint_margin_rate

    def calculate_maintenance_margin_exact(
        self,
        notional: Decimal,
    ) -> Decimal:
        """
        Calculate exact maintenance margin with cumulative adjustment.

        Binance formula:
        MM = Notional * MMR - Cumulative Maintenance

        This is more accurate for exact liquidation price calculation.

        Args:
            notional: Position notional value

        Returns:
            Exact maintenance margin amount
        """
        abs_notional = abs(notional)
        idx, bracket = self._find_bracket(abs_notional)

        # Exact Binance formula with cumulative
        raw_mm = abs_notional * bracket.maint_margin_rate
        cum_maint = self._cum_maintenance[idx] if idx < len(self._cum_maintenance) else Decimal("0")

        return max(Decimal("0"), raw_mm - cum_maint)

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: MarginMode,
        isolated_margin: Decimal = Decimal("0"),
        cum_pnl: Decimal = Decimal("0"),
        cum_funding: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate liquidation price.

        Binance Liquidation Formula:

        For CROSS margin:
            Long: LP = (Notional - AvailableMargin - cumMaint) / (Qty * (1 - MMR - LiqFee))
            Short: LP = (Notional + AvailableMargin + cumMaint) / (abs_qty * (1 + MMR + LiqFee))

        For ISOLATED margin:
            Use isolated_margin instead of wallet_balance.

        Args:
            entry_price: Position entry price
            qty: Position quantity (positive=long, negative=short)
            leverage: Position leverage
            wallet_balance: Total wallet balance (cross margin)
            margin_mode: Cross or isolated margin
            isolated_margin: Isolated margin amount (if applicable)
            cum_pnl: Cumulative realized PnL
            cum_funding: Cumulative funding payments

        Returns:
            Estimated liquidation price
        """
        if qty == 0:
            return Decimal("0")

        is_long = qty > 0
        abs_qty = abs(qty)
        notional = entry_price * abs_qty

        # Get MMR for position size
        mmr = self.get_maintenance_margin_rate(notional)

        # Get index and bracket info to find cumulative maintenance (cum_maint)
        idx, _ = self._find_bracket(notional)
        cum_maint = self._cum_maintenance[idx] if idx < len(self._cum_maintenance) else Decimal("0")

        # Determine available margin
        if margin_mode == MarginMode.CROSS:
            available_margin = wallet_balance + cum_pnl + cum_funding
        else:
            available_margin = isolated_margin

        # Unified mathematically exact formula
        if is_long:
            # Long liquidation: price drops
            # LP = (Notional - AvailableMargin - cumMaint) / (Qty * (1 - MMR - LiqFee))
            denominator = abs_qty * (Decimal("1") - mmr - self._liquidation_fee_rate)
            if denominator <= 0:
                return Decimal("0")
            liq_price = (notional - available_margin - cum_maint) / denominator
        else:
            # Short liquidation: price rises
            # LP = (Notional + AvailableMargin + cumMaint) / (abs_qty * (1 + MMR + LiqFee))
            denominator = abs_qty * (Decimal("1") + mmr + self._liquidation_fee_rate)
            liq_price = (notional + available_margin + cum_maint) / denominator

        # Liquidation price can't be negative
        return max(Decimal("0"), liq_price)

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate current margin ratio.

        Margin Ratio = (Wallet Balance + Unrealized PnL) / Maintenance Margin

        MR > 1: Safe
        MR = 1: At maintenance (margin call)
        MR < 1: Liquidation zone

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance (for cross margin)

        Returns:
            Margin ratio
        """
        if position.qty == 0:
            return Decimal("inf")

        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        # Calculate unrealized PnL
        if position.qty > 0:  # Long
            upnl = (mark_price - position.entry_price) * abs_qty
        else:  # Short
            upnl = (position.entry_price - mark_price) * abs_qty

        # Calculate maintenance margin
        mm = self.calculate_maintenance_margin(notional)

        if mm == 0:
            return Decimal("inf")

        # Determine effective balance
        if position.margin_mode == MarginMode.CROSS:
            effective_balance = wallet_balance + upnl
        else:
            effective_balance = position.margin + upnl

        return effective_balance / mm

    def calculate_margin_requirement(
        self,
        contract: FuturesContractSpec,
        qty: Decimal,
        price: Decimal,
        leverage: int,
    ) -> MarginRequirement:
        """
        Calculate full margin requirement for a trade.

        Args:
            contract: Contract specification
            qty: Order quantity
            price: Order price
            leverage: Desired leverage

        Returns:
            MarginRequirement with initial, maintenance, and available
        """
        notional = abs(qty) * price * contract.multiplier

        initial = self.calculate_initial_margin(notional, leverage)
        maintenance = self.calculate_maintenance_margin(notional)

        return MarginRequirement(
            initial=initial,
            maintenance=maintenance,
            variation=Decimal("0"),
            available=initial - maintenance,
        )

    def get_effective_leverage(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Calculate effective leverage for position.

        Effective Leverage = Position Value / Margin

        Args:
            position: Current position
            mark_price: Current mark price

        Returns:
            Effective leverage (can exceed nominal due to PnL)
        """
        if position.margin == 0:
            return Decimal("0")

        position_value = abs(position.qty) * mark_price
        return position_value / position.margin

    def estimate_liquidation_distance_pct(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Estimate distance to liquidation as percentage of current price.

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance

        Returns:
            Percentage distance (e.g., 5.0 = 5% away from liquidation)
        """
        liq_price = self.calculate_liquidation_price(
            entry_price=position.entry_price,
            qty=position.qty,
            leverage=position.leverage,
            wallet_balance=wallet_balance,
            margin_mode=position.margin_mode,
            isolated_margin=position.margin,
        )

        if mark_price == 0 or liq_price == 0:
            return Decimal("100")  # Safe (no position or invalid price)

        distance = abs(mark_price - liq_price) / mark_price * 100
        return distance


# ============================================================================
# CME MARGIN CALCULATOR (Simplified SPAN)
# ============================================================================

@dataclass(frozen=True)
class CMEMarginRate:
    """
    CME margin rates for a contract.

    CME uses fixed dollar amounts per contract, not percentages.
    These are typically published by the exchange.

    Attributes:
        symbol: Contract symbol (ES, NQ, GC, etc.)
        initial_margin: Initial margin per contract (USD)
        maintenance_margin: Maintenance margin per contract (USD)
        day_trading_margin: Intraday margin (optional, lower)
    """
    symbol: str
    initial_margin: Decimal
    maintenance_margin: Decimal
    day_trading_margin: Optional[Decimal] = None


# Default CME margin rates (approximate, updated periodically)
DEFAULT_CME_MARGINS: Dict[str, CMEMarginRate] = {
    "ES": CMEMarginRate(
        symbol="ES",
        initial_margin=Decimal("15180"),
        maintenance_margin=Decimal("13800"),
        day_trading_margin=Decimal("500"),
    ),
    "NQ": CMEMarginRate(
        symbol="NQ",
        initial_margin=Decimal("21120"),
        maintenance_margin=Decimal("19200"),
        day_trading_margin=Decimal("500"),
    ),
    "MES": CMEMarginRate(
        symbol="MES",
        initial_margin=Decimal("1518"),
        maintenance_margin=Decimal("1380"),
        day_trading_margin=Decimal("50"),
    ),
    "GC": CMEMarginRate(
        symbol="GC",
        initial_margin=Decimal("11000"),
        maintenance_margin=Decimal("10000"),
        day_trading_margin=Decimal("1000"),
    ),
    "CL": CMEMarginRate(
        symbol="CL",
        initial_margin=Decimal("7700"),
        maintenance_margin=Decimal("7000"),
        day_trading_margin=Decimal("500"),
    ),
    "6E": CMEMarginRate(
        symbol="6E",
        initial_margin=Decimal("2420"),
        maintenance_margin=Decimal("2200"),
        day_trading_margin=Decimal("500"),
    ),
}


class CMEMarginCalculator:
    """
    CME-style margin calculator.

    Uses fixed dollar amounts per contract rather than tiered percentages.
    Supports basic SPAN-like concepts (but not full portfolio offsets).

    Key differences from Binance:
    - Fixed margin per contract (not percentage)
    - No tiered brackets based on position size
    - Margin requirements published by exchange
    - Day trading margins available

    Reference:
    https://www.cmegroup.com/clearing/margins/outright-vol-background.html
    """

    def __init__(
        self,
        margin_rates: Optional[Dict[str, CMEMarginRate]] = None,
        use_day_trading_margin: bool = False,
    ):
        """
        Initialize with margin rates.

        Args:
            margin_rates: Dict of symbol to margin rates
            use_day_trading_margin: Use lower day trading margins
        """
        self._rates = margin_rates or DEFAULT_CME_MARGINS
        self._use_day_margin = use_day_trading_margin

    def get_margin_rate(self, symbol: str) -> Optional[CMEMarginRate]:
        """Get margin rate for symbol."""
        # Try exact match first
        if symbol in self._rates:
            return self._rates[symbol]

        # Try base symbol (e.g., ESM24 -> ES)
        base_symbol = "".join(c for c in symbol if c.isalpha())[:2]
        return self._rates.get(base_symbol)

    def calculate_initial_margin(
        self,
        symbol: str,
        qty: Decimal,
        price: Decimal = Decimal("0"),  # Not used, but kept for interface
        leverage: int = 1,  # Not used for CME
    ) -> Decimal:
        """
        Calculate initial margin for CME futures.

        IM = Number of contracts * Initial margin per contract

        Args:
            symbol: Contract symbol
            qty: Number of contracts
            price: Not used (CME uses fixed amounts)
            leverage: Not used (CME determines leverage via margin)

        Returns:
            Initial margin amount in USD
        """
        rate = self.get_margin_rate(symbol)
        if not rate:
            # Fallback: 10% of notional if margin rate unknown
            logger.warning(f"No margin rate for {symbol}, using fallback")
            return Decimal("10000") * abs(qty)  # Rough estimate

        per_contract = rate.initial_margin
        if self._use_day_margin and rate.day_trading_margin:
            per_contract = rate.day_trading_margin

        return per_contract * abs(qty)

    def calculate_maintenance_margin(
        self,
        symbol: str,
        qty: Decimal,
    ) -> Decimal:
        """
        Calculate maintenance margin for CME futures.

        MM = Number of contracts * Maintenance margin per contract

        Args:
            symbol: Contract symbol
            qty: Number of contracts

        Returns:
            Maintenance margin amount in USD
        """
        rate = self.get_margin_rate(symbol)
        if not rate:
            logger.warning(f"No margin rate for {symbol}, using fallback")
            return Decimal("9000") * abs(qty)

        return rate.maintenance_margin * abs(qty)

    def calculate_max_position_size(
        self,
        symbol: str,
        available_margin: Decimal,
        use_day_margin: bool = False,
    ) -> int:
        """
        Calculate maximum position size given available margin.

        Args:
            symbol: Contract symbol
            available_margin: Available margin in USD
            use_day_margin: Use day trading margins

        Returns:
            Maximum number of contracts
        """
        rate = self.get_margin_rate(symbol)
        if not rate:
            return 0

        margin_per = rate.initial_margin
        if use_day_margin and rate.day_trading_margin:
            margin_per = rate.day_trading_margin

        if margin_per <= 0:
            return 0

        return int(available_margin / margin_per)

    def get_implied_leverage(
        self,
        symbol: str,
        price: Decimal,
        contract: FuturesContractSpec,
    ) -> Decimal:
        """
        Calculate implied leverage based on margin requirement.

        Leverage = Notional / Initial Margin

        Args:
            symbol: Contract symbol
            price: Current price
            contract: Contract specification

        Returns:
            Implied leverage
        """
        rate = self.get_margin_rate(symbol)
        if not rate or rate.initial_margin == 0:
            return Decimal("1")

        notional = price * contract.multiplier
        return notional / rate.initial_margin


# ============================================================================
# SIMPLE MARGIN CALCULATOR
# ============================================================================

class SimpleMarginCalculator:
    """
    Simple fixed-percentage margin calculator.

    Uses fixed initial and maintenance margin percentages.
    Suitable for:
    - Testing and development
    - Simple backtesting
    - Exchanges without tiered brackets

    Example:
        >>> calc = SimpleMarginCalculator(initial_pct=5.0, maintenance_pct=4.0)
        >>> calc.calculate_initial_margin(Decimal("100000"), leverage=20)
        Decimal('5000')
    """

    def __init__(
        self,
        initial_pct: float = 5.0,
        maintenance_pct: float = 4.0,
        max_leverage: int = 20,
        liquidation_fee_pct: float = 0.5,
    ):
        """
        Initialize with margin percentages.

        Args:
            initial_pct: Initial margin as percentage of notional
            maintenance_pct: Maintenance margin as percentage
            max_leverage: Maximum allowed leverage
            liquidation_fee_pct: Liquidation fee percentage
        """
        self._initial_rate = Decimal(str(initial_pct)) / 100
        self._maintenance_rate = Decimal(str(maintenance_pct)) / 100
        self._max_leverage = max(1, max_leverage)
        self._liq_fee_rate = Decimal(str(liquidation_fee_pct)) / 100

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """Calculate initial margin."""
        effective_leverage = max(1, min(leverage, self._max_leverage))
        return abs(notional) / Decimal(effective_leverage)

    def calculate_maintenance_margin(
        self,
        notional: Decimal,
    ) -> Decimal:
        """Calculate maintenance margin."""
        return abs(notional) * self._maintenance_rate

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        qty: Decimal,
        leverage: int,
        wallet_balance: Decimal,
        margin_mode: MarginMode,
        isolated_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """Calculate liquidation price."""
        if qty == 0:
            return Decimal("0")

        is_long = qty > 0
        effective_leverage = max(1, min(leverage, self._max_leverage))
        imr = Decimal("1") / Decimal(effective_leverage)
        mmr = self._maintenance_rate

        if is_long:
            liq_price = entry_price * (1 - imr + mmr + self._liq_fee_rate)
        else:
            liq_price = entry_price * (1 + imr - mmr - self._liq_fee_rate)

        return max(Decimal("0"), liq_price)

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """Calculate margin ratio."""
        if position.qty == 0:
            return Decimal("inf")

        abs_qty = abs(position.qty)
        notional = mark_price * abs_qty

        if position.qty > 0:
            upnl = (mark_price - position.entry_price) * abs_qty
        else:
            upnl = (position.entry_price - mark_price) * abs_qty

        mm = self.calculate_maintenance_margin(notional)

        if mm == 0:
            return Decimal("inf")

        if position.margin_mode == MarginMode.CROSS:
            effective_balance = wallet_balance + upnl
        else:
            effective_balance = position.margin + upnl

        return effective_balance / mm

    def get_max_leverage(self, notional: Decimal) -> int:
        """Get maximum leverage."""
        return self._max_leverage


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_margin_calculator(
    futures_type: FuturesType,
    brackets: Optional[List[LeverageBracket]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> FuturesMarginCalculator:
    """
    Create appropriate margin calculator for futures type.

    Args:
        futures_type: Type of futures contract
        brackets: Leverage brackets (for crypto)
        config: Additional configuration

    Returns:
        Margin calculator instance
    """
    config = config or {}

    if futures_type.is_crypto:
        if brackets:
            return TieredMarginCalculator(
                brackets=brackets,
                liquidation_fee_rate=Decimal(str(config.get("liquidation_fee_rate", "0.005"))),
            )
        else:
            # Use default brackets
            return TieredMarginCalculator(
                brackets=get_default_btc_brackets(),
                liquidation_fee_rate=Decimal(str(config.get("liquidation_fee_rate", "0.005"))),
            )

    elif futures_type in (FuturesType.INDEX_FUTURES, FuturesType.COMMODITY_FUTURES,
                          FuturesType.CURRENCY_FUTURES, FuturesType.BOND_FUTURES):
        return CMEMarginCalculator(
            margin_rates=config.get("margin_rates"),
            use_day_trading_margin=config.get("use_day_trading_margin", False),
        )

    else:
        # Fallback to simple calculator
        return SimpleMarginCalculator(
            initial_pct=config.get("initial_margin_pct", 5.0),
            maintenance_pct=config.get("maintenance_margin_pct", 4.0),
            max_leverage=config.get("max_leverage", 20),
        )


def get_default_btc_brackets() -> List[LeverageBracket]:
    """
    Get default BTCUSDT leverage brackets.

    Based on Binance USDT-M Futures brackets as of 2024.

    Returns:
        List of LeverageBracket
    """
    return [
        LeverageBracket(bracket=1, notional_cap=Decimal("50000"), maint_margin_rate=Decimal("0.004"), max_leverage=125),
        LeverageBracket(bracket=2, notional_cap=Decimal("250000"), maint_margin_rate=Decimal("0.005"), max_leverage=100),
        LeverageBracket(bracket=3, notional_cap=Decimal("1000000"), maint_margin_rate=Decimal("0.01"), max_leverage=50),
        LeverageBracket(bracket=4, notional_cap=Decimal("10000000"), maint_margin_rate=Decimal("0.025"), max_leverage=20),
        LeverageBracket(bracket=5, notional_cap=Decimal("20000000"), maint_margin_rate=Decimal("0.05"), max_leverage=10),
        LeverageBracket(bracket=6, notional_cap=Decimal("50000000"), maint_margin_rate=Decimal("0.10"), max_leverage=5),
        LeverageBracket(bracket=7, notional_cap=Decimal("100000000"), maint_margin_rate=Decimal("0.125"), max_leverage=4),
        LeverageBracket(bracket=8, notional_cap=Decimal("200000000"), maint_margin_rate=Decimal("0.15"), max_leverage=3),
        LeverageBracket(bracket=9, notional_cap=Decimal("300000000"), maint_margin_rate=Decimal("0.25"), max_leverage=2),
        LeverageBracket(bracket=10, notional_cap=Decimal("9223372036854775807"), maint_margin_rate=Decimal("0.50"), max_leverage=1),
    ]


def get_default_eth_brackets() -> List[LeverageBracket]:
    """
    Get default ETHUSDT leverage brackets.

    Returns:
        List of LeverageBracket
    """
    return [
        LeverageBracket(bracket=1, notional_cap=Decimal("10000"), maint_margin_rate=Decimal("0.005"), max_leverage=100),
        LeverageBracket(bracket=2, notional_cap=Decimal("100000"), maint_margin_rate=Decimal("0.0065"), max_leverage=75),
        LeverageBracket(bracket=3, notional_cap=Decimal("500000"), maint_margin_rate=Decimal("0.01"), max_leverage=50),
        LeverageBracket(bracket=4, notional_cap=Decimal("1000000"), maint_margin_rate=Decimal("0.02"), max_leverage=25),
        LeverageBracket(bracket=5, notional_cap=Decimal("2000000"), maint_margin_rate=Decimal("0.05"), max_leverage=10),
        LeverageBracket(bracket=6, notional_cap=Decimal("5000000"), maint_margin_rate=Decimal("0.10"), max_leverage=5),
        LeverageBracket(bracket=7, notional_cap=Decimal("10000000"), maint_margin_rate=Decimal("0.125"), max_leverage=4),
        LeverageBracket(bracket=8, notional_cap=Decimal("20000000"), maint_margin_rate=Decimal("0.15"), max_leverage=3),
        LeverageBracket(bracket=9, notional_cap=Decimal("9223372036854775807"), maint_margin_rate=Decimal("0.25"), max_leverage=2),
    ]


def load_brackets_from_json(filepath: str) -> Dict[str, List[LeverageBracket]]:
    """
    Load leverage brackets from JSON file.

    Expected format:
    {
        "BTCUSDT": [
            {"bracket": 1, "notionalCap": 50000, "maintMarginRate": 0.004, "maxLeverage": 125},
            ...
        ]
    }

    Args:
        filepath: Path to JSON file

    Returns:
        Dict of symbol to list of brackets
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    result = {}
    for symbol, brackets_data in data.items():
        brackets = []
        for i, b in enumerate(brackets_data):
            brackets.append(LeverageBracket(
                bracket=b.get("bracket", i + 1),
                notional_cap=Decimal(str(b.get("notionalCap", b.get("notional_cap", "0")))),
                maint_margin_rate=Decimal(str(b.get("maintMarginRate", b.get("maint_margin_rate", "0.01")))),
                max_leverage=int(b.get("maxLeverage", b.get("max_leverage", 1))),
            ))
        result[symbol] = brackets

    return result


# Alias for compatibility with execution_providers_futures_base.py
load_leverage_brackets = load_brackets_from_json
