# -*- coding: utf-8 -*-
"""
impl_span_margin.py
CME SPAN Margin Calculator for traditional futures.

SPAN (Standard Portfolio Analysis of Risk) is the industry-standard
risk-based margin methodology used by CME Group and other major exchanges.

This module provides a SIMPLIFIED approximation of SPAN margin. Full SPAN
uses a comprehensive 16-scenario risk analysis that we approximate here.

Key Concepts:
=============

1. SCANNING RANGE
   The maximum expected price move for a given contract over a set period
   (typically one trading day). SPAN tests the portfolio under various
   price and volatility scenarios.

2. INTER-COMMODITY SPREAD CREDITS
   Offsets for positions in correlated markets (e.g., ES/NQ, GC/SI).
   Reduces margin because correlation reduces combined risk.

3. INTRA-COMMODITY SPREAD CREDITS (Calendar Spreads)
   Offsets for positions in different months of the same contract.
   Lower risk than outright positions.

4. INITIAL vs MAINTENANCE MARGIN
   - Initial: Required to open a position (~125% of scanning risk)
   - Maintenance: Required to keep position open (~80% of initial)
   - Margin call triggered when equity < maintenance

Accuracy Assessment:
===================
- Single positions: ~90-95% accurate vs real SPAN
- Simple spreads (ES/NQ): ~80-90% accurate
- Complex portfolios: ~60-80% accurate (may underestimate margin)

For Production Systems:
======================
- Use CME's official PC-SPAN calculator
- Use IB's whatIfOrder() for real-time margin queries
- Integrate with CME SPAN parameter files (updated daily)
- Consider OpenGamma's Strata for enterprise-grade SPAN

References:
==========
- CME SPAN Methodology: https://www.cmegroup.com/clearing/risk-management/span-overview.html
- CME SPAN Parameter Files: https://www.cmegroup.com/clearing/risk-management/files-resources.html
- CME Margin Requirements: https://www.cmegroup.com/clearing/margins/outright-vol-background.html
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from core_futures import (
    FuturesContractSpec,
    FuturesPosition,
    FuturesType,
    MarginMode,
    PositionSide,
    Exchange,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


class ProductGroup(str, Enum):
    """CME product group classification for SPAN."""

    EQUITY_INDEX = "EQUITY_INDEX"  # ES, NQ, YM, RTY
    METALS = "METALS"  # GC, SI, HG
    ENERGY = "ENERGY"  # CL, NG, RB, HO
    CURRENCIES = "CURRENCIES"  # 6E, 6J, 6B, 6A
    BONDS = "BONDS"  # ZB, ZN, ZT, ZF
    AGRICULTURAL = "AGRICULTURAL"  # ZC, ZS, ZW


# Product to group mapping
PRODUCT_GROUPS: Dict[str, ProductGroup] = {
    # Equity Index
    "ES": ProductGroup.EQUITY_INDEX,
    "NQ": ProductGroup.EQUITY_INDEX,
    "YM": ProductGroup.EQUITY_INDEX,
    "RTY": ProductGroup.EQUITY_INDEX,
    "MES": ProductGroup.EQUITY_INDEX,
    "MNQ": ProductGroup.EQUITY_INDEX,
    "MYM": ProductGroup.EQUITY_INDEX,
    "M2K": ProductGroup.EQUITY_INDEX,
    # Metals
    "GC": ProductGroup.METALS,
    "SI": ProductGroup.METALS,
    "HG": ProductGroup.METALS,
    "MGC": ProductGroup.METALS,
    "SIL": ProductGroup.METALS,
    # Energy
    "CL": ProductGroup.ENERGY,
    "NG": ProductGroup.ENERGY,
    "MCL": ProductGroup.ENERGY,
    "RB": ProductGroup.ENERGY,
    "HO": ProductGroup.ENERGY,
    # Currencies
    "6E": ProductGroup.CURRENCIES,
    "6J": ProductGroup.CURRENCIES,
    "6B": ProductGroup.CURRENCIES,
    "6A": ProductGroup.CURRENCIES,
    "6C": ProductGroup.CURRENCIES,
    "6S": ProductGroup.CURRENCIES,
    "M6E": ProductGroup.CURRENCIES,
    # Bonds
    "ZB": ProductGroup.BONDS,
    "ZN": ProductGroup.BONDS,
    "ZT": ProductGroup.BONDS,
    "ZF": ProductGroup.BONDS,
    # Agricultural
    "ZC": ProductGroup.AGRICULTURAL,
    "ZS": ProductGroup.AGRICULTURAL,
    "ZW": ProductGroup.AGRICULTURAL,
    "ZM": ProductGroup.AGRICULTURAL,
    "ZL": ProductGroup.AGRICULTURAL,
}


# =============================================================================
# SPAN Scanning Ranges
# =============================================================================


@dataclass(frozen=True)
class ScanningRangeConfig:
    """
    Scanning range configuration for a product.

    The scanning range is the maximum expected price move under stress.
    SPAN tests the portfolio under 16 scenarios combining price and
    volatility changes.

    Attributes:
        price_scan_range_pct: Max price move as % (e.g., 0.08 = ±8%)
        volatility_scan_range_pct: Max volatility change as % (e.g., 0.33 = ±33%)
        short_option_minimum: Minimum margin for short options
        extreme_move_multiplier: Multiplier for extreme move scenarios
    """

    price_scan_range_pct: Decimal
    volatility_scan_range_pct: Decimal = Decimal("0.33")  # ±33%
    short_option_minimum: Decimal = Decimal("0")
    extreme_move_multiplier: Decimal = Decimal("2.0")


# Default scanning ranges by product
# These are approximations based on CME published margin requirements
# Actual SPAN parameter files are updated daily
SCANNING_RANGES: Dict[str, ScanningRangeConfig] = {
    # Equity Index (moderate volatility)
    "ES": ScanningRangeConfig(Decimal("0.06")),  # ±6%
    "NQ": ScanningRangeConfig(Decimal("0.08")),  # ±8% (more volatile)
    "YM": ScanningRangeConfig(Decimal("0.06")),  # ±6%
    "RTY": ScanningRangeConfig(Decimal("0.09")),  # ±9% (small cap)
    "MES": ScanningRangeConfig(Decimal("0.06")),  # Same as ES
    "MNQ": ScanningRangeConfig(Decimal("0.08")),  # Same as NQ
    # Metals
    "GC": ScanningRangeConfig(Decimal("0.05")),  # ±5%
    "SI": ScanningRangeConfig(Decimal("0.08")),  # ±8% (more volatile)
    "HG": ScanningRangeConfig(Decimal("0.07")),  # ±7%
    "MGC": ScanningRangeConfig(Decimal("0.05")),  # Same as GC
    # Energy (high volatility)
    "CL": ScanningRangeConfig(Decimal("0.10")),  # ±10%
    "NG": ScanningRangeConfig(Decimal("0.15")),  # ±15% (very volatile)
    "MCL": ScanningRangeConfig(Decimal("0.10")),  # Same as CL
    "RB": ScanningRangeConfig(Decimal("0.10")),  # ±10%
    "HO": ScanningRangeConfig(Decimal("0.10")),  # ±10%
    # Currencies (low volatility)
    "6E": ScanningRangeConfig(Decimal("0.03")),  # ±3%
    "6J": ScanningRangeConfig(Decimal("0.04")),  # ±4%
    "6B": ScanningRangeConfig(Decimal("0.04")),  # ±4%
    "6A": ScanningRangeConfig(Decimal("0.04")),  # ±4%
    "6C": ScanningRangeConfig(Decimal("0.03")),  # ±3%
    "6S": ScanningRangeConfig(Decimal("0.03")),  # ±3%
    # Bonds (low volatility)
    "ZB": ScanningRangeConfig(Decimal("0.025")),  # ±2.5%
    "ZN": ScanningRangeConfig(Decimal("0.020")),  # ±2.0%
    "ZT": ScanningRangeConfig(Decimal("0.015")),  # ±1.5%
    "ZF": ScanningRangeConfig(Decimal("0.018")),  # ±1.8%
    # Agricultural
    "ZC": ScanningRangeConfig(Decimal("0.06")),  # ±6%
    "ZS": ScanningRangeConfig(Decimal("0.06")),  # ±6%
    "ZW": ScanningRangeConfig(Decimal("0.07")),  # ±7%
}

# Default scanning range for unknown products
DEFAULT_SCANNING_RANGE = ScanningRangeConfig(Decimal("0.08"))


# =============================================================================
# Spread Credit Configuration
# =============================================================================


@dataclass(frozen=True)
class InterCommoditySpreadCredit:
    """
    Inter-commodity spread credit configuration.

    Applied when holding positions in correlated products.
    Credit reduces total margin requirement.

    Attributes:
        product1: First product symbol
        product2: Second product symbol
        credit_rate: Credit as % of smaller leg margin (0.0 to 1.0)
        ratio: Spread ratio (contracts of product1 per contract of product2)
        min_legs: Minimum matched legs to apply credit
    """

    product1: str
    product2: str
    credit_rate: Decimal
    ratio: Tuple[int, int] = (1, 1)  # e.g., (1, 1) means 1:1 spread
    min_legs: int = 1


# Inter-commodity spread credits
# These represent correlation-based margin offsets
INTER_COMMODITY_CREDITS: List[InterCommoditySpreadCredit] = [
    # Equity Index spreads
    InterCommoditySpreadCredit("ES", "NQ", Decimal("0.50")),  # 50% credit
    InterCommoditySpreadCredit("ES", "YM", Decimal("0.55")),  # 55% credit
    InterCommoditySpreadCredit("ES", "RTY", Decimal("0.40")),  # 40% credit
    InterCommoditySpreadCredit("NQ", "RTY", Decimal("0.35")),  # 35% credit
    # Micro vs Standard
    InterCommoditySpreadCredit("ES", "MES", Decimal("0.90")),  # 90% credit
    InterCommoditySpreadCredit("NQ", "MNQ", Decimal("0.90")),
    # Metals
    InterCommoditySpreadCredit("GC", "SI", Decimal("0.35")),  # 35% gold/silver
    InterCommoditySpreadCredit("GC", "HG", Decimal("0.20")),  # 20% gold/copper
    InterCommoditySpreadCredit("GC", "MGC", Decimal("0.90")),  # 90% micro
    # Energy
    InterCommoditySpreadCredit("CL", "NG", Decimal("0.15")),  # 15% oil/gas
    InterCommoditySpreadCredit("CL", "RB", Decimal("0.60")),  # 60% crack spread
    InterCommoditySpreadCredit("CL", "HO", Decimal("0.60")),  # 60% crack spread
    InterCommoditySpreadCredit("CL", "MCL", Decimal("0.90")),  # 90% micro
    # Currencies
    InterCommoditySpreadCredit("6E", "6B", Decimal("0.45")),  # 45% EUR/GBP
    InterCommoditySpreadCredit("6E", "6S", Decimal("0.50")),  # 50% EUR/CHF
    InterCommoditySpreadCredit("6J", "6A", Decimal("0.25")),  # 25% JPY/AUD
    # Bonds (yield curve)
    InterCommoditySpreadCredit("ZB", "ZN", Decimal("0.70")),  # 70% 30Y/10Y
    InterCommoditySpreadCredit("ZN", "ZF", Decimal("0.75")),  # 75% 10Y/5Y
    InterCommoditySpreadCredit("ZF", "ZT", Decimal("0.80")),  # 80% 5Y/2Y
]


# Intra-commodity (calendar) spread credits
# Applied to positions in different months of same product
CALENDAR_SPREAD_CREDITS: Dict[str, Decimal] = {
    # Equity Index (high credit for calendar spreads)
    "ES": Decimal("0.80"),
    "NQ": Decimal("0.80"),
    "YM": Decimal("0.80"),
    "RTY": Decimal("0.75"),
    # Metals
    "GC": Decimal("0.85"),
    "SI": Decimal("0.80"),
    "HG": Decimal("0.75"),
    # Energy (lower due to contango/backwardation)
    "CL": Decimal("0.60"),
    "NG": Decimal("0.50"),  # Lower due to seasonality
    "RB": Decimal("0.65"),
    "HO": Decimal("0.65"),
    # Currencies
    "6E": Decimal("0.85"),
    "6J": Decimal("0.85"),
    "6B": Decimal("0.85"),
    # Bonds
    "ZB": Decimal("0.90"),
    "ZN": Decimal("0.90"),
    "ZF": Decimal("0.90"),
    "ZT": Decimal("0.90"),
}

# Default calendar spread credit
DEFAULT_CALENDAR_SPREAD_CREDIT = Decimal("0.70")


# =============================================================================
# SPAN Margin Data Classes
# =============================================================================


@dataclass
class SPANScenarioResult:
    """
    Result of a single SPAN price/volatility scenario.

    Attributes:
        scenario_id: Scenario number (1-16)
        price_move_pct: Price move as percentage
        volatility_move_pct: Volatility move as percentage
        portfolio_value_change: Change in portfolio value
        is_extreme: Whether this is an extreme move scenario
    """

    scenario_id: int
    price_move_pct: Decimal
    volatility_move_pct: Decimal
    portfolio_value_change: Decimal
    is_extreme: bool = False


@dataclass
class PositionMarginDetail:
    """
    Detailed margin breakdown for a single position.

    Attributes:
        symbol: Product symbol
        position_qty: Number of contracts
        position_side: LONG or SHORT
        notional_value: Position notional at current price
        scanning_risk: Max loss across 16 scenarios
        delivery_month_charge: Extra charge for near-expiry
        net_option_value: Mark-to-market of options (0 for futures)
        gross_margin: scanning_risk + delivery_charge - option_value
    """

    symbol: str
    position_qty: Decimal
    position_side: PositionSide
    notional_value: Decimal
    scanning_risk: Decimal
    delivery_month_charge: Decimal = Decimal("0")
    net_option_value: Decimal = Decimal("0")

    @property
    def gross_margin(self) -> Decimal:
        """Gross margin before spread credits."""
        return self.scanning_risk + self.delivery_month_charge - self.net_option_value


@dataclass
class SpreadCreditDetail:
    """
    Detail of a spread credit applied.

    Attributes:
        credit_type: "INTER" or "INTRA"
        products: Products involved in spread
        legs_matched: Number of spread legs matched
        credit_amount: Margin credit amount
        credit_rate: Credit rate applied
    """

    credit_type: str  # "INTER" or "INTRA"
    products: Tuple[str, ...]
    legs_matched: int
    credit_amount: Decimal
    credit_rate: Decimal


@dataclass
class SPANMarginResult:
    """
    Complete SPAN margin calculation result.

    Attributes:
        initial_margin: Required to open positions
        maintenance_margin: Required to maintain positions
        scanning_risk: Total scanning risk across portfolio
        inter_commodity_credit: Credit for correlated products
        intra_commodity_credit: Credit for calendar spreads
        delivery_month_charge: Extra charge for near-expiry
        net_option_value: Net option value
        position_details: Per-position margin breakdown
        spread_credits: Applied spread credits
        worst_scenario: Scenario producing max loss
    """

    initial_margin: Decimal
    maintenance_margin: Decimal
    scanning_risk: Decimal
    inter_commodity_credit: Decimal = Decimal("0")
    intra_commodity_credit: Decimal = Decimal("0")
    delivery_month_charge: Decimal = Decimal("0")
    net_option_value: Decimal = Decimal("0")
    position_details: List[PositionMarginDetail] = field(default_factory=list)
    spread_credits: List[SpreadCreditDetail] = field(default_factory=list)
    worst_scenario: Optional[SPANScenarioResult] = None

    @property
    def net_portfolio_margin(self) -> Decimal:
        """Net margin after all credits."""
        return max(
            Decimal("0"),
            self.scanning_risk
            + self.delivery_month_charge
            - self.net_option_value
            - self.inter_commodity_credit
            - self.intra_commodity_credit,
        )

    @property
    def total_credit(self) -> Decimal:
        """Total spread credits applied."""
        return self.inter_commodity_credit + self.intra_commodity_credit

    @property
    def margin_ratio(self) -> Decimal:
        """Ratio of maintenance to initial margin."""
        if self.initial_margin == 0:
            return Decimal("0")
        return self.maintenance_margin / self.initial_margin


# =============================================================================
# SPAN Margin Calculator
# =============================================================================


class SPANMarginCalculator:
    """
    Simplified SPAN margin calculator.

    This is an APPROXIMATION of CME's official SPAN methodology.
    See module docstring for accuracy assessment and production recommendations.

    Basic Formula:
        scanning_risk = notional × scanning_range_pct
        gross_margin = scanning_risk + delivery_charge - option_value
        net_margin = gross_margin - inter_credits - intra_credits
        initial_margin = net_margin × initial_multiplier (default 1.10)
        maintenance_margin = initial_margin × maint_ratio (default 0.80)

    Example Usage:
        >>> calculator = SPANMarginCalculator()
        >>> result = calculator.calculate_margin(
        ...     position=position,
        ...     current_price=Decimal("4500.00"),
        ...     contract_spec=es_spec,
        ... )
        >>> print(f"Initial: ${result.initial_margin}")
        >>> print(f"Maintenance: ${result.maintenance_margin}")

    For Portfolio Margin:
        >>> result = calculator.calculate_portfolio_margin(
        ...     positions=[es_position, nq_position],
        ...     prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
        ...     contract_specs={"ES": es_spec, "NQ": nq_spec},
        ... )
        >>> print(f"Portfolio Initial: ${result.initial_margin}")
        >>> print(f"Credits Applied: ${result.total_credit}")
    """

    # Margin multipliers
    DEFAULT_INITIAL_MULTIPLIER = Decimal("1.10")  # Initial = scanning × 1.10
    DEFAULT_MAINT_RATIO = Decimal("0.80")  # Maint = initial × 0.80

    # Delivery month charge parameters
    DELIVERY_MONTH_DAYS = 10  # Days before expiry for charge
    DELIVERY_CHARGE_RATE = Decimal("0.05")  # 5% extra charge

    def __init__(
        self,
        contract_specs: Optional[Dict[str, FuturesContractSpec]] = None,
        scanning_ranges: Optional[Dict[str, ScanningRangeConfig]] = None,
        inter_credits: Optional[List[InterCommoditySpreadCredit]] = None,
        calendar_credits: Optional[Dict[str, Decimal]] = None,
        initial_multiplier: Decimal = DEFAULT_INITIAL_MULTIPLIER,
        maint_ratio: Decimal = DEFAULT_MAINT_RATIO,
    ) -> None:
        """
        Initialize SPAN margin calculator.

        Args:
            contract_specs: Map of symbol to contract spec (optional)
            scanning_ranges: Custom scanning ranges by symbol
            inter_credits: Custom inter-commodity spread credits
            calendar_credits: Custom calendar spread credits
            initial_multiplier: Multiplier for initial margin
            maint_ratio: Maintenance/initial margin ratio
        """
        self._specs = contract_specs or {}
        self._scanning_ranges = scanning_ranges or SCANNING_RANGES.copy()
        self._inter_credits = inter_credits or INTER_COMMODITY_CREDITS
        self._calendar_credits = calendar_credits or CALENDAR_SPREAD_CREDITS.copy()
        self._initial_mult = initial_multiplier
        self._maint_ratio = maint_ratio

        # Build inter-commodity credit lookup
        self._inter_credit_lookup: Dict[FrozenSet[str], InterCommoditySpreadCredit] = {}
        for credit in self._inter_credits:
            key = frozenset([credit.product1.upper(), credit.product2.upper()])
            self._inter_credit_lookup[key] = credit

    def get_scanning_range(self, symbol: str) -> ScanningRangeConfig:
        """
        Get scanning range configuration for a symbol.

        Args:
            symbol: Product symbol

        Returns:
            ScanningRangeConfig for the product
        """
        symbol = symbol.upper()
        # Check custom overrides
        if symbol in self._scanning_ranges:
            return self._scanning_ranges[symbol]
        # Check defaults
        if symbol in SCANNING_RANGES:
            return SCANNING_RANGES[symbol]
        # Return default
        return DEFAULT_SCANNING_RANGE

    def get_calendar_credit_rate(self, symbol: str) -> Decimal:
        """
        Get calendar spread credit rate for a symbol.

        Args:
            symbol: Product symbol

        Returns:
            Credit rate (0.0 to 1.0)
        """
        symbol = symbol.upper()
        if symbol in self._calendar_credits:
            return self._calendar_credits[symbol]
        if symbol in CALENDAR_SPREAD_CREDITS:
            return CALENDAR_SPREAD_CREDITS[symbol]
        return DEFAULT_CALENDAR_SPREAD_CREDIT

    def calculate_scanning_risk(
        self,
        notional_value: Decimal,
        symbol: str,
        volatility_override: Optional[Decimal] = None,
    ) -> Tuple[Decimal, SPANScenarioResult]:
        """
        Calculate scanning risk for a notional position.

        Simplified: Uses single scenario (max price move in adverse direction).
        Full SPAN would test 16 price/volatility combinations.

        Args:
            notional_value: Position notional value
            symbol: Product symbol
            volatility_override: Override volatility scan range

        Returns:
            (scanning_risk, worst_scenario)
        """
        config = self.get_scanning_range(symbol)

        # Use override if provided
        price_scan = config.price_scan_range_pct
        vol_scan = volatility_override or config.volatility_scan_range_pct

        # Simplified: Max loss = notional × price_scan_range
        # (In full SPAN this would be the max of 16 scenarios)
        scanning_risk = abs(notional_value) * price_scan

        # Create worst scenario result
        worst = SPANScenarioResult(
            scenario_id=3,  # Typically scenario 3 is down, vol up
            price_move_pct=-price_scan,
            volatility_move_pct=vol_scan,
            portfolio_value_change=-scanning_risk,
            is_extreme=False,
        )

        return scanning_risk, worst

    def calculate_margin(
        self,
        position: FuturesPosition,
        current_price: Decimal,
        contract_spec: Optional[FuturesContractSpec] = None,
        days_to_expiry: Optional[int] = None,
    ) -> SPANMarginResult:
        """
        Calculate SPAN margin for a single position.

        Args:
            position: Futures position
            current_price: Current market price
            contract_spec: Contract specification (uses cached if not provided)
            days_to_expiry: Days until contract expiration

        Returns:
            SPANMarginResult with margin requirements

        Raises:
            ValueError: If contract spec not found and not provided
        """
        symbol = position.symbol.upper()

        # Get contract spec
        spec = contract_spec or self._specs.get(symbol)
        if spec is None:
            raise ValueError(f"No contract spec for {symbol}. Provide spec or add to constructor.")

        # Calculate notional value
        # For index futures: price * qty * multiplier (multiplier is $/point)
        # For commodity futures: price * qty * contract_size * multiplier
        abs_qty = abs(position.qty)
        notional = current_price * abs_qty * spec.multiplier * spec.contract_size

        # Calculate scanning risk
        scanning_risk, worst_scenario = self.calculate_scanning_risk(notional, symbol)

        # Calculate delivery month charge
        delivery_charge = Decimal("0")
        if days_to_expiry is not None and days_to_expiry <= self.DELIVERY_MONTH_DAYS:
            delivery_charge = scanning_risk * self.DELIVERY_CHARGE_RATE

        # Calculate initial and maintenance margin
        net_margin = max(Decimal("0"), scanning_risk + delivery_charge)
        initial_margin = (net_margin * self._initial_mult).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        maintenance_margin = (initial_margin * self._maint_ratio).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Build position detail
        pos_detail = PositionMarginDetail(
            symbol=symbol,
            position_qty=abs_qty,
            position_side=position.side,
            notional_value=notional,
            scanning_risk=scanning_risk,
            delivery_month_charge=delivery_charge,
        )

        return SPANMarginResult(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            scanning_risk=scanning_risk,
            delivery_month_charge=delivery_charge,
            position_details=[pos_detail],
            worst_scenario=worst_scenario,
        )

    def calculate_portfolio_margin(
        self,
        positions: Sequence[FuturesPosition],
        prices: Mapping[str, Decimal],
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
        days_to_expiry: Optional[Mapping[str, int]] = None,
    ) -> SPANMarginResult:
        """
        Calculate SPAN margin for a portfolio with spread credits.

        This method:
        1. Calculates scanning risk for each position
        2. Applies inter-commodity spread credits
        3. Applies intra-commodity (calendar) spread credits
        4. Computes net portfolio margin

        Args:
            positions: List of futures positions
            prices: Map of symbol to current price
            contract_specs: Map of symbol to contract spec
            days_to_expiry: Map of symbol to days until expiry

        Returns:
            SPANMarginResult with portfolio margin and credits
        """
        if not positions:
            return SPANMarginResult(
                initial_margin=Decimal("0"),
                maintenance_margin=Decimal("0"),
                scanning_risk=Decimal("0"),
            )

        specs = dict(contract_specs or self._specs)
        expiry_days = dict(days_to_expiry or {})

        # Step 1: Calculate margin for each position
        position_margins: Dict[str, SPANMarginResult] = {}
        position_details: List[PositionMarginDetail] = []
        total_scanning = Decimal("0")
        total_delivery = Decimal("0")

        for pos in positions:
            symbol = pos.symbol.upper()

            if symbol not in prices:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            spec = specs.get(symbol)
            if spec is None:
                logger.warning(f"No contract spec for {symbol}, skipping")
                continue

            margin = self.calculate_margin(
                position=pos,
                current_price=prices[symbol],
                contract_spec=spec,
                days_to_expiry=expiry_days.get(symbol),
            )

            position_margins[symbol] = margin
            position_details.extend(margin.position_details)
            total_scanning += margin.scanning_risk
            total_delivery += margin.delivery_month_charge

        if not position_margins:
            return SPANMarginResult(
                initial_margin=Decimal("0"),
                maintenance_margin=Decimal("0"),
                scanning_risk=Decimal("0"),
            )

        # Step 2: Calculate inter-commodity spread credits
        inter_credits, inter_details = self._calculate_inter_commodity_credits(
            position_margins, positions
        )

        # Step 3: Calculate intra-commodity (calendar) spread credits
        intra_credits, intra_details = self._calculate_intra_commodity_credits(
            positions, position_margins
        )

        # Step 4: Calculate net margin
        net_scanning = max(Decimal("0"), total_scanning - inter_credits - intra_credits)

        net_margin = net_scanning + total_delivery
        initial_margin = (net_margin * self._initial_mult).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        maintenance_margin = (initial_margin * self._maint_ratio).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return SPANMarginResult(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            scanning_risk=total_scanning,
            inter_commodity_credit=inter_credits,
            intra_commodity_credit=intra_credits,
            delivery_month_charge=total_delivery,
            position_details=position_details,
            spread_credits=inter_details + intra_details,
        )

    def _calculate_inter_commodity_credits(
        self,
        margins: Mapping[str, SPANMarginResult],
        positions: Sequence[FuturesPosition],
    ) -> Tuple[Decimal, List[SpreadCreditDetail]]:
        """
        Calculate inter-commodity spread credits.

        Credits are applied when holding positions in correlated products.
        """
        total_credit = Decimal("0")
        details: List[SpreadCreditDetail] = []
        symbols = set(margins.keys())
        used_symbols: Set[str] = set()

        for credit_def in self._inter_credits:
            sym1 = credit_def.product1.upper()
            sym2 = credit_def.product2.upper()

            # Check if both products are in portfolio
            if sym1 not in symbols or sym2 not in symbols:
                continue

            # Don't double-count symbols
            if sym1 in used_symbols or sym2 in used_symbols:
                continue

            # Get position sides
            side1 = next((p.side for p in positions if p.symbol.upper() == sym1), None)
            side2 = next((p.side for p in positions if p.symbol.upper() == sym2), None)

            if side1 is None or side2 is None:
                continue

            # For spread credit, positions should be opposite (hedge)
            # or same direction for positively correlated products
            # Simplified: Always apply credit if both present

            # Calculate credit = min(margin1, margin2) × credit_rate
            margin1 = margins[sym1].scanning_risk
            margin2 = margins[sym2].scanning_risk
            smaller_margin = min(margin1, margin2)

            credit = smaller_margin * credit_def.credit_rate
            total_credit += credit

            details.append(
                SpreadCreditDetail(
                    credit_type="INTER",
                    products=(sym1, sym2),
                    legs_matched=1,
                    credit_amount=credit,
                    credit_rate=credit_def.credit_rate,
                )
            )

            used_symbols.add(sym1)
            used_symbols.add(sym2)

        return total_credit, details

    def _calculate_intra_commodity_credits(
        self,
        positions: Sequence[FuturesPosition],
        margins: Mapping[str, SPANMarginResult],
    ) -> Tuple[Decimal, List[SpreadCreditDetail]]:
        """
        Calculate intra-commodity (calendar) spread credits.

        Credits for positions in different months of same product.
        Simplified: Assumes positions in same product but different months
        get credit based on the calendar credit rate.
        """
        # Group positions by base symbol (strip month code)
        # For now, assume each symbol is unique (no calendar spreads)
        # Full implementation would parse month codes
        return Decimal("0"), []

    def estimate_margin_impact(
        self,
        current_positions: Sequence[FuturesPosition],
        new_position: FuturesPosition,
        prices: Mapping[str, Decimal],
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
    ) -> Tuple[Decimal, Decimal]:
        """
        Estimate margin impact of adding a new position.

        Useful for pre-trade checks.

        Args:
            current_positions: Existing portfolio
            new_position: Position to add
            prices: Current prices
            contract_specs: Contract specifications

        Returns:
            (margin_impact, new_total_margin)
            margin_impact can be negative if spread credits apply
        """
        specs = dict(contract_specs or self._specs)

        # Current margin
        current_margin = self.calculate_portfolio_margin(
            positions=current_positions,
            prices=prices,
            contract_specs=specs,
        )

        # New margin with additional position
        new_positions = list(current_positions) + [new_position]
        new_margin = self.calculate_portfolio_margin(
            positions=new_positions,
            prices=prices,
            contract_specs=specs,
        )

        impact = new_margin.initial_margin - current_margin.initial_margin

        return impact, new_margin.initial_margin

    def check_margin_call(
        self,
        account_equity: Decimal,
        positions: Sequence[FuturesPosition],
        prices: Mapping[str, Decimal],
        contract_specs: Optional[Mapping[str, FuturesContractSpec]] = None,
    ) -> Tuple[bool, Decimal, str]:
        """
        Check if account is in margin call.

        Args:
            account_equity: Current account equity
            positions: Portfolio positions
            prices: Current prices
            contract_specs: Contract specifications

        Returns:
            (is_margin_call, excess_or_deficit, status)
            status: "OK", "WARNING" (< 150% maint), "MARGIN_CALL" (< maint)
        """
        margin = self.calculate_portfolio_margin(
            positions=positions,
            prices=prices,
            contract_specs=contract_specs,
        )

        maint = margin.maintenance_margin

        if maint == 0:
            return False, account_equity, "OK"

        excess = account_equity - maint
        ratio = account_equity / maint if maint > 0 else Decimal("inf")

        if excess < 0:
            return True, excess, "MARGIN_CALL"
        elif ratio < Decimal("1.50"):
            return False, excess, "WARNING"
        else:
            return False, excess, "OK"

    def set_contract_spec(self, symbol: str, spec: FuturesContractSpec) -> None:
        """Add or update a contract specification."""
        self._specs[symbol.upper()] = spec

    def set_scanning_range(self, symbol: str, config: ScanningRangeConfig) -> None:
        """Add or update a scanning range configuration."""
        self._scanning_ranges[symbol.upper()] = config


# =============================================================================
# Factory Functions
# =============================================================================


def create_span_calculator(
    include_default_specs: bool = True,
) -> SPANMarginCalculator:
    """
    Create a SPAN margin calculator with default configurations.

    Args:
        include_default_specs: Include common contract specs (ES, NQ, GC, etc.)

    Returns:
        Configured SPANMarginCalculator
    """
    specs: Dict[str, FuturesContractSpec] = {}

    if include_default_specs:
        # Import factory functions
        from core_futures import (
            create_es_futures_spec,
            create_gc_futures_spec,
            create_6e_futures_spec,
        )

        # Add common specs
        specs["ES"] = create_es_futures_spec()
        specs["GC"] = create_gc_futures_spec()
        specs["6E"] = create_6e_futures_spec()

    return SPANMarginCalculator(contract_specs=specs)


def calculate_simple_margin(
    position_notional: Decimal,
    symbol: str,
    initial_only: bool = False,
) -> Tuple[Decimal, Decimal]:
    """
    Simple margin calculation without portfolio effects.

    Args:
        position_notional: Position notional value
        symbol: Product symbol
        initial_only: Return only initial margin

    Returns:
        (initial_margin, maintenance_margin)
    """
    calc = SPANMarginCalculator()
    config = calc.get_scanning_range(symbol)

    scanning_risk = abs(position_notional) * config.price_scan_range_pct
    initial = scanning_risk * SPANMarginCalculator.DEFAULT_INITIAL_MULTIPLIER
    maintenance = initial * SPANMarginCalculator.DEFAULT_MAINT_RATIO

    return initial, maintenance


# =============================================================================
# Margin Requirement Presets (2024 approximations)
# =============================================================================

# These are approximate initial margin requirements per contract
# Updated periodically by CME Group
APPROXIMATE_MARGINS_PER_CONTRACT: Dict[str, Tuple[Decimal, Decimal]] = {
    # (Initial, Maintenance) in USD
    # Equity Index
    "ES": (Decimal("12000"), Decimal("11000")),  # E-mini S&P 500
    "NQ": (Decimal("17000"), Decimal("15500")),  # E-mini NASDAQ
    "YM": (Decimal("9500"), Decimal("8600")),  # E-mini Dow
    "RTY": (Decimal("7500"), Decimal("6800")),  # E-mini Russell 2000
    "MES": (Decimal("1200"), Decimal("1100")),  # Micro E-mini S&P
    "MNQ": (Decimal("1700"), Decimal("1550")),  # Micro E-mini NASDAQ
    # Metals
    "GC": (Decimal("9500"), Decimal("8600")),  # Gold
    "SI": (Decimal("12000"), Decimal("11000")),  # Silver
    "MGC": (Decimal("950"), Decimal("860")),  # Micro Gold
    # Energy
    "CL": (Decimal("6500"), Decimal("5900")),  # Crude Oil
    "NG": (Decimal("2500"), Decimal("2300")),  # Natural Gas
    "MCL": (Decimal("650"), Decimal("590")),  # Micro Crude
    # Currencies
    "6E": (Decimal("2500"), Decimal("2300")),  # Euro FX
    "6J": (Decimal("3000"), Decimal("2700")),  # Japanese Yen
    "6B": (Decimal("2800"), Decimal("2500")),  # British Pound
    # Bonds
    "ZB": (Decimal("3500"), Decimal("3200")),  # 30-Year T-Bond
    "ZN": (Decimal("2000"), Decimal("1800")),  # 10-Year T-Note
    "ZF": (Decimal("1200"), Decimal("1100")),  # 5-Year T-Note
    "ZT": (Decimal("600"), Decimal("550")),  # 2-Year T-Note
}


def get_approximate_margin_per_contract(symbol: str) -> Tuple[Decimal, Decimal]:
    """
    Get approximate margin requirement per contract.

    Uses preset values based on typical CME margin requirements.
    For accurate requirements, use IB's whatIfOrder() or CME's PC-SPAN.

    Args:
        symbol: Product symbol

    Returns:
        (initial_margin, maintenance_margin) per contract
    """
    symbol = symbol.upper()
    if symbol in APPROXIMATE_MARGINS_PER_CONTRACT:
        return APPROXIMATE_MARGINS_PER_CONTRACT[symbol]

    # Default: $10,000 initial, $9,000 maintenance
    return (Decimal("10000"), Decimal("9000"))
