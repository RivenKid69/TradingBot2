# -*- coding: utf-8 -*-
"""
execution_providers_cme.py
CME futures execution providers for traditional index/commodity/currency futures.

CME index futures (ES, NQ) have high liquidity and tight spreads.
Commodity futures (GC, CL) have seasonal and time-of-day patterns.
Currency futures (6E, 6J) correlate with forex spot markets.

This module provides:
1. CMESlippageProvider - L2+ slippage model with CME-specific factors
2. CMEFeeProvider - Exchange fees including NFA, clearing, etc.
3. CMEL2ExecutionProvider - Combined execution provider for CME futures

Key CME Microstructure Factors:
==============================

1. SESSION AWARENESS
   - RTH (Regular Trading Hours): 8:30 AM - 3:15 PM CT for equity index
   - ETH (Electronic Trading Hours): 5:00 PM - 4:00 PM CT (next day)
   - ETH spreads are typically 50-100% wider than RTH

2. SETTLEMENT TIME EFFECTS
   - Daily settlement at product-specific times (see impl_cme_settlement.py)
   - Increased volatility and widened spreads 15-30 min before settlement
   - Equity: 2:30 PM CT, Metals: 1:30 PM CT, Energy: 2:30 PM CT

3. CIRCUIT BREAKER INTEGRATION
   - Equity index: 7%, 13%, 20% levels
   - Commodity daily limits: vary by product
   - When active, liquidity dries up significantly

4. ROLL PERIOD EFFECTS
   - Front month liquidity shifts ~8 days before expiry
   - Back month volume increases as roll approaches
   - Calendar spreads tighten during roll

References:
==========
- CME Globex Market Maker Program: https://www.cmegroup.com/education/market-making.html
- CME Market Data Feed Specifications
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2013): "Optimal Trading Strategies"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from execution_providers import (
    AssetClass,
    BarData,
    Fill,
    MarketState,
    Order,
    SlippageProvider,
    FeeProvider,
    ExecutionProvider,
)

from impl_span_margin import ProductGroup, PRODUCT_GROUPS

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


class CMETradingSession(str, Enum):
    """CME trading session types."""

    RTH = "RTH"  # Regular Trading Hours
    ETH = "ETH"  # Electronic Trading Hours (overnight)
    PRE_OPEN = "PRE_OPEN"  # Pre-open auction period
    CLOSE = "CLOSE"  # Closing period


class CircuitBreakerState(str, Enum):
    """Circuit breaker state."""

    NORMAL = "NORMAL"
    LEVEL_1 = "LEVEL_1"  # -7% for equity index
    LEVEL_2 = "LEVEL_2"  # -13% for equity index
    LEVEL_3 = "LEVEL_3"  # -20% for equity index (day halt)
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    VELOCITY_PAUSE = "VELOCITY_PAUSE"


# Default tick sizes by product
TICK_SIZES: Dict[str, Decimal] = {
    # Equity Index
    "ES": Decimal("0.25"),  # $12.50 per tick
    "NQ": Decimal("0.25"),  # $5.00 per tick
    "YM": Decimal("1.0"),  # $5.00 per tick
    "RTY": Decimal("0.10"),  # $5.00 per tick
    "MES": Decimal("0.25"),  # $1.25 per tick
    "MNQ": Decimal("0.25"),  # $0.50 per tick
    # Metals
    "GC": Decimal("0.10"),  # $10.00 per tick
    "SI": Decimal("0.005"),  # $25.00 per tick
    "HG": Decimal("0.0005"),  # $12.50 per tick
    "MGC": Decimal("0.10"),  # $1.00 per tick
    # Energy
    "CL": Decimal("0.01"),  # $10.00 per tick
    "NG": Decimal("0.001"),  # $10.00 per tick
    "MCL": Decimal("0.01"),  # $1.00 per tick
    # Currencies
    "6E": Decimal("0.00005"),  # $6.25 per tick
    "6J": Decimal("0.0000005"),  # $6.25 per tick
    "6B": Decimal("0.0001"),  # $6.25 per tick
    "6A": Decimal("0.0001"),  # $10.00 per tick
    # Bonds
    "ZB": Decimal("0.03125"),  # $31.25 per tick (1/32nd)
    "ZN": Decimal("0.015625"),  # $15.625 per tick (1/64th)
    "ZF": Decimal("0.0078125"),  # $7.8125 per tick
    "ZT": Decimal("0.0078125"),  # $15.625 per tick
}

DEFAULT_TICK_SIZE = Decimal("0.01")


# =============================================================================
# CME Slippage Configuration
# =============================================================================


@dataclass
class CMESlippageConfig:
    """
    CME-specific slippage configuration.

    Attributes:
        symbol_spreads: Map of symbol to typical spread in ticks
        symbol_impacts: Map of symbol to impact coefficient
        rth_spread_multiplier: RTH spread multiplier (1.0 = baseline)
        eth_spread_multiplier: ETH spread multiplier (wider)
        settlement_premium_max: Max premium near settlement (%)
        settlement_window_minutes: Minutes before settlement for premium
        circuit_breaker_multiplier: Multiplier when circuit breaker near
        roll_period_spread_multiplier: Spread increase during roll
        min_slippage_bps: Minimum slippage floor
        max_slippage_bps: Maximum slippage cap
    """

    # Default spreads in ticks
    symbol_spreads: Dict[str, Decimal] = field(
        default_factory=lambda: {
            # Equity Index (very liquid)
            "ES": Decimal("0.25"),  # 1 tick
            "NQ": Decimal("0.25"),  # 1 tick
            "YM": Decimal("1.0"),  # 1 tick
            "RTY": Decimal("0.20"),  # 2 ticks
            "MES": Decimal("0.25"),  # 1 tick
            "MNQ": Decimal("0.25"),  # 1 tick
            # Metals
            "GC": Decimal("0.10"),  # 1 tick
            "SI": Decimal("0.005"),  # 1 tick
            "HG": Decimal("0.0005"),  # 1 tick
            "MGC": Decimal("0.10"),  # 1 tick
            # Energy (can widen)
            "CL": Decimal("0.01"),  # 1 tick
            "NG": Decimal("0.002"),  # 2 ticks (more volatile)
            "MCL": Decimal("0.01"),  # 1 tick
            # Currencies
            "6E": Decimal("0.00005"),  # 1 tick
            "6J": Decimal("0.0000010"),  # 2 ticks
            "6B": Decimal("0.0001"),  # 1 tick
            # Bonds
            "ZB": Decimal("0.03125"),  # 1 tick
            "ZN": Decimal("0.015625"),  # 1 tick
            "ZF": Decimal("0.0078125"),  # 1 tick
        }
    )

    # Impact coefficients (lower = more liquid)
    symbol_impacts: Dict[str, float] = field(
        default_factory=lambda: {
            # Equity Index (very liquid)
            "ES": 0.03,  # E-mini S&P 500 - highest liquidity
            "NQ": 0.04,  # E-mini NASDAQ
            "YM": 0.05,  # E-mini Dow
            "RTY": 0.06,  # E-mini Russell 2000
            "MES": 0.035,  # Micro E-mini (slightly wider)
            "MNQ": 0.045,
            # Metals
            "GC": 0.05,  # Gold - moderate liquidity
            "SI": 0.07,  # Silver - less liquid
            "HG": 0.06,  # Copper
            "MGC": 0.055,  # Micro Gold
            # Energy
            "CL": 0.04,  # Crude Oil - very liquid
            "NG": 0.08,  # Natural Gas - more volatile
            "MCL": 0.045,  # Micro Crude
            # Currencies
            "6E": 0.04,  # Euro FX - very liquid
            "6J": 0.05,  # Japanese Yen
            "6B": 0.05,  # British Pound
            "6A": 0.06,  # Australian Dollar
            # Bonds
            "ZB": 0.03,  # 30-Year Bond - very liquid
            "ZN": 0.025,  # 10-Year Note - most liquid
            "ZF": 0.03,  # 5-Year Note
            "ZT": 0.04,  # 2-Year Note
        }
    )

    # Session multipliers
    rth_spread_multiplier: float = 1.0
    eth_spread_multiplier: float = 1.5  # 50% wider in ETH

    # Settlement effects
    settlement_premium_max: float = 0.30  # Up to 30% increase
    settlement_window_minutes: int = 30  # 30 min before settlement

    # Circuit breaker
    circuit_breaker_multiplier: float = 5.0  # 5x spread when near limit

    # Roll period
    roll_period_spread_multiplier: float = 1.2  # 20% wider during roll

    # Bounds
    min_slippage_bps: float = 0.5
    max_slippage_bps: float = 200.0

    # Default impact coefficient
    default_impact: float = 0.05
    default_spread_bps: float = 2.0

    def get_spread_ticks(self, symbol: str) -> Decimal:
        """Get spread in ticks for a symbol."""
        return self.symbol_spreads.get(symbol.upper(), Decimal("1"))

    def get_impact_coef(self, symbol: str) -> float:
        """Get impact coefficient for a symbol."""
        return self.symbol_impacts.get(symbol.upper(), self.default_impact)


# Predefined profiles
CME_SLIPPAGE_PROFILES: Dict[str, CMESlippageConfig] = {
    "default": CMESlippageConfig(),
    "conservative": CMESlippageConfig(
        eth_spread_multiplier=2.0,
        settlement_premium_max=0.40,
        min_slippage_bps=1.0,
    ),
    "aggressive": CMESlippageConfig(
        eth_spread_multiplier=1.3,
        settlement_premium_max=0.20,
        min_slippage_bps=0.3,
    ),
    "equity_index": CMESlippageConfig(
        symbol_impacts={
            "ES": 0.025,
            "NQ": 0.03,
            "YM": 0.04,
            "RTY": 0.05,
            "MES": 0.03,
            "MNQ": 0.035,
            "MYM": 0.045,
            "M2K": 0.055,
        },
        min_slippage_bps=0.3,
    ),
    "metals": CMESlippageConfig(
        symbol_impacts={"GC": 0.05, "SI": 0.07, "HG": 0.06, "MGC": 0.055},
        eth_spread_multiplier=1.8,
    ),
    "energy": CMESlippageConfig(
        symbol_impacts={"CL": 0.04, "NG": 0.09, "MCL": 0.045, "RB": 0.06, "HO": 0.06},
        eth_spread_multiplier=1.6,
    ),
}


# =============================================================================
# CME Slippage Provider
# =============================================================================


class CMESlippageProvider:
    """
    L2+ CME futures slippage model.

    Extends basic √participation model with CME-specific factors:
    - Trading session awareness (RTH vs ETH)
    - Settlement time premium
    - Circuit breaker integration
    - Roll period effects
    - Product-specific spreads and impact coefficients

    Formula:
        base_spread_bps = spread_ticks / mid_price * 10000
        base_slippage = base_spread_bps / 2 + k * sqrt(participation) * 10000
        total_slippage = base_slippage * session_mult * settlement_mult * cb_mult

    Example Usage:
        >>> provider = CMESlippageProvider()
        >>> slippage = provider.compute_slippage_bps(
        ...     order=Order("ES", "BUY", 5, "MARKET"),
        ...     market=MarketState(timestamp=0, bid=4500.0, ask=4500.25, adv=2e9),
        ...     participation_ratio=0.001,
        ...     is_rth=True,
        ... )
        >>> print(f"Slippage: {slippage:.2f} bps")

    References:
        - Almgren & Chriss (2001): "Optimal Execution"
        - Kissell (2013): "Optimal Trading Strategies"
        - CME Globex Trading Hours: https://www.cmegroup.com/trading-hours.html
    """

    def __init__(
        self,
        config: Optional[CMESlippageConfig] = None,
    ) -> None:
        """
        Initialize CME slippage provider.

        Args:
            config: Slippage configuration (uses default if None)
        """
        self._config = config or CMESlippageConfig()
        self._circuit_breaker_state = CircuitBreakerState.NORMAL
        self._is_roll_period = False

    @classmethod
    def from_profile(cls, profile: str) -> "CMESlippageProvider":
        """
        Create provider from a predefined profile.

        Args:
            profile: Profile name (default, conservative, aggressive,
                     equity_index, metals, energy)

        Returns:
            Configured CMESlippageProvider

        Raises:
            ValueError: If profile not found
        """
        if profile not in CME_SLIPPAGE_PROFILES:
            available = ", ".join(CME_SLIPPAGE_PROFILES.keys())
            raise ValueError(f"Unknown profile: {profile}. Available: {available}")
        return cls(config=CME_SLIPPAGE_PROFILES[profile])

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        is_rth: bool = True,
        minutes_to_settlement: Optional[int] = None,
        is_roll_period: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Compute CME futures slippage in basis points.

        Args:
            order: Order to execute
            market: Current market state (bid/ask/adv)
            participation_ratio: Order notional / ADV
            is_rth: True if Regular Trading Hours
            minutes_to_settlement: Minutes until daily settlement
            is_roll_period: True if within 8 days of expiry
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            Expected slippage in basis points

        Notes:
            - Returns max_slippage_bps if circuit breaker active (Level 2+)
            - Returns elevated slippage if velocity pause active
        """
        # Handle circuit breaker states
        if self._circuit_breaker_state in (
            CircuitBreakerState.LEVEL_2,
            CircuitBreakerState.LEVEL_3,
            CircuitBreakerState.LIMIT_UP,
            CircuitBreakerState.LIMIT_DOWN,
        ):
            return self._config.max_slippage_bps

        if self._circuit_breaker_state == CircuitBreakerState.VELOCITY_PAUSE:
            # Still tradeable but very wide spreads
            return self._config.max_slippage_bps / 2

        symbol = order.symbol.upper()

        # Get mid price
        mid_price = market.get_mid_price()
        if mid_price is None or mid_price <= 0:
            mid_price = 1.0  # Fallback

        # 1. Calculate base spread in bps
        spread_ticks = float(self._config.get_spread_ticks(symbol))
        tick_size = float(TICK_SIZES.get(symbol, DEFAULT_TICK_SIZE))
        spread_value = spread_ticks * tick_size

        # Handle percentage-style spreads for currencies
        if spread_value < 0.0001:
            # Currency futures: spread is already in price terms
            base_spread_bps = (spread_value / mid_price) * 10000
        else:
            base_spread_bps = (spread_value / mid_price) * 10000

        # 2. Calculate market impact
        impact_k = self._config.get_impact_coef(symbol)
        participation = max(0, min(1, participation_ratio))  # Clamp to [0, 1]

        # √participation impact model
        impact_bps = impact_k * math.sqrt(participation) * 10000

        # 3. Base slippage (half spread + impact)
        base_slippage = base_spread_bps / 2 + impact_bps

        # 4. Apply session multiplier
        if is_rth:
            session_mult = self._config.rth_spread_multiplier
        else:
            session_mult = self._config.eth_spread_multiplier

        # 5. Apply settlement time premium
        settlement_mult = 1.0
        if minutes_to_settlement is not None and minutes_to_settlement > 0:
            if minutes_to_settlement <= self._config.settlement_window_minutes:
                # Linear increase as settlement approaches
                fraction = 1.0 - (minutes_to_settlement / self._config.settlement_window_minutes)
                settlement_mult = 1.0 + (self._config.settlement_premium_max * fraction)

        # 6. Apply roll period multiplier
        roll_mult = 1.0
        if is_roll_period or self._is_roll_period:
            roll_mult = self._config.roll_period_spread_multiplier

        # 7. Apply circuit breaker multiplier (for Level 1 / near-limit)
        cb_mult = 1.0
        if self._circuit_breaker_state == CircuitBreakerState.LEVEL_1:
            cb_mult = self._config.circuit_breaker_multiplier

        # Total slippage
        total_slippage = base_slippage * session_mult * settlement_mult * roll_mult * cb_mult

        # Apply bounds
        return float(
            max(self._config.min_slippage_bps, min(self._config.max_slippage_bps, total_slippage))
        )

    def set_circuit_breaker_state(self, state: CircuitBreakerState) -> None:
        """
        Set circuit breaker state.

        Args:
            state: New circuit breaker state
        """
        self._circuit_breaker_state = state

    def set_roll_period(self, is_roll: bool) -> None:
        """
        Set roll period flag.

        Args:
            is_roll: True if within roll period (typically 8 days before expiry)
        """
        self._is_roll_period = is_roll

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        symbol: str,
        is_rth: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate impact cost for a given order size.

        Args:
            notional: Order notional value
            adv: Average Daily Volume in notional
            symbol: Product symbol
            is_rth: True if Regular Trading Hours

        Returns:
            Dictionary with impact_bps, impact_cost, and recommendation
        """
        if adv <= 0:
            return {
                "impact_bps": self._config.max_slippage_bps,
                "impact_cost": notional * self._config.max_slippage_bps / 10000,
                "recommendation": "Cannot estimate - no ADV provided",
            }

        participation = notional / adv

        # Create synthetic order and market state
        order = Order(symbol=symbol, side="BUY", qty=1.0, order_type="MARKET")
        market = MarketState(timestamp=0, bid=100.0, ask=100.01, adv=adv)

        impact_bps = self.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation,
            is_rth=is_rth,
        )

        impact_cost = notional * impact_bps / 10000

        # Generate recommendation
        if participation > 0.10:
            rec = "CRITICAL: >10% of ADV. Consider multi-day execution."
        elif participation > 0.05:
            rec = "HIGH: 5-10% of ADV. Consider TWAP over session."
        elif participation > 0.01:
            rec = "MODERATE: 1-5% of ADV. Consider VWAP."
        else:
            rec = "LOW: <1% of ADV. Market order acceptable."

        return {
            "impact_bps": impact_bps,
            "impact_cost": impact_cost,
            "participation_ratio": participation,
            "recommendation": rec,
        }

    @property
    def config(self) -> CMESlippageConfig:
        """Get current configuration."""
        return self._config


# =============================================================================
# CME Fee Provider
# =============================================================================


@dataclass
class CMEFeeConfig:
    """
    CME fee configuration.

    CME futures fees include:
    - Exchange fee (CME/CBOT/NYMEX/COMEX)
    - Clearing fee
    - NFA fee (National Futures Association)
    - Technology fee

    Fees are per-contract (not percentage).

    Attributes:
        exchange_fees: Map of symbol to exchange fee per contract
        clearing_fee: Clearing fee per contract
        nfa_fee: NFA regulatory fee per contract
        tech_fee: Technology/platform fee per contract
    """

    # Exchange fees per contract (2024 approximations)
    exchange_fees: Dict[str, Decimal] = field(
        default_factory=lambda: {
            # Equity Index (CME)
            "ES": Decimal("1.18"),
            "NQ": Decimal("1.18"),
            "YM": Decimal("1.18"),
            "RTY": Decimal("1.18"),
            "MES": Decimal("0.20"),
            "MNQ": Decimal("0.20"),
            # Metals (COMEX)
            "GC": Decimal("1.50"),
            "SI": Decimal("1.50"),
            "HG": Decimal("1.50"),
            "MGC": Decimal("0.50"),
            # Energy (NYMEX)
            "CL": Decimal("1.45"),
            "NG": Decimal("1.45"),
            "MCL": Decimal("0.25"),
            # Currencies (CME)
            "6E": Decimal("0.85"),
            "6J": Decimal("0.85"),
            "6B": Decimal("0.85"),
            # Bonds (CBOT)
            "ZB": Decimal("0.85"),
            "ZN": Decimal("0.85"),
            "ZF": Decimal("0.85"),
            "ZT": Decimal("0.85"),
        }
    )

    # Clearing fee per contract
    clearing_fee: Decimal = Decimal("0.10")

    # NFA regulatory fee per contract
    nfa_fee: Decimal = Decimal("0.02")

    # Technology/platform fee per contract
    tech_fee: Decimal = Decimal("0.25")

    # Default exchange fee if symbol not found
    default_exchange_fee: Decimal = Decimal("1.50")

    def get_exchange_fee(self, symbol: str) -> Decimal:
        """Get exchange fee for a symbol."""
        return self.exchange_fees.get(symbol.upper(), self.default_exchange_fee)

    def get_total_fee_per_contract(self, symbol: str) -> Decimal:
        """Get total fee per contract."""
        return self.get_exchange_fee(symbol) + self.clearing_fee + self.nfa_fee + self.tech_fee


class CMEFeeProvider:
    """
    CME futures fee provider.

    Calculates per-contract fees for CME Group futures including:
    - Exchange fees (varies by product and exchange)
    - Clearing fees
    - NFA regulatory fees
    - Technology fees

    Note: CME fees are per-contract, not percentage-based like crypto.

    Example:
        >>> provider = CMEFeeProvider()
        >>> fee = provider.compute_fee(
        ...     notional=225000.0,  # 1 ES at 4500
        ...     side="BUY",
        ...     liquidity="taker",
        ...     qty=1.0,
        ...     symbol="ES",
        ... )
        >>> print(f"Fee: ${fee:.2f}")  # ~$1.55
    """

    def __init__(self, config: Optional[CMEFeeConfig] = None) -> None:
        """
        Initialize CME fee provider.

        Args:
            config: Fee configuration (uses default if None)
        """
        self._config = config or CMEFeeConfig()

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
        symbol: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute CME futures fee.

        Args:
            notional: Trade notional (not used, fees are per-contract)
            side: Trade side ("BUY" or "SELL") - not used
            liquidity: "maker" or "taker" - not used (same fee)
            qty: Number of contracts
            symbol: Product symbol
            **kwargs: Additional arguments (ignored)

        Returns:
            Total fee amount in USD

        Note:
            CME fees are the same for maker/taker unlike crypto exchanges.
        """
        if symbol is None:
            symbol = "ES"  # Default

        fee_per_contract = float(self._config.get_total_fee_per_contract(symbol))
        return fee_per_contract * abs(qty)

    def get_fee_breakdown(self, symbol: str, qty: float) -> Dict[str, float]:
        """
        Get detailed fee breakdown.

        Args:
            symbol: Product symbol
            qty: Number of contracts

        Returns:
            Dictionary with exchange, clearing, nfa, tech fees
        """
        qty_abs = abs(qty)
        return {
            "exchange_fee": float(self._config.get_exchange_fee(symbol)) * qty_abs,
            "clearing_fee": float(self._config.clearing_fee) * qty_abs,
            "nfa_fee": float(self._config.nfa_fee) * qty_abs,
            "tech_fee": float(self._config.tech_fee) * qty_abs,
            "total_fee": float(self._config.get_total_fee_per_contract(symbol)) * qty_abs,
        }


# =============================================================================
# CME L2 Execution Provider
# =============================================================================


class CMEL2ExecutionProvider:
    """
    Combined L2 execution provider for CME futures.

    Integrates CMESlippageProvider and CMEFeeProvider for full
    execution simulation.

    Example:
        >>> provider = CMEL2ExecutionProvider()
        >>> fill = provider.execute(
        ...     order=Order("ES", "BUY", 5, "MARKET"),
        ...     market=MarketState(timestamp=0, bid=4500, ask=4500.25, adv=2e9),
        ...     bar=BarData(open=4500, high=4510, low=4490, close=4505, volume=100000),
        ... )
        >>> print(f"Filled at {fill.price} with {fill.slippage_bps:.2f} bps slippage")
    """

    def __init__(
        self,
        slippage_config: Optional[CMESlippageConfig] = None,
        fee_config: Optional[CMEFeeConfig] = None,
    ) -> None:
        """
        Initialize CME execution provider.

        Args:
            slippage_config: Slippage configuration
            fee_config: Fee configuration
        """
        self._slippage = CMESlippageProvider(config=slippage_config)
        self._fees = CMEFeeProvider(config=fee_config)

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        return AssetClass.FUTURES

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        is_rth: bool = True,
        minutes_to_settlement: Optional[int] = None,
        is_roll_period: bool = False,
        **kwargs: Any,
    ) -> Optional[Fill]:
        """
        Execute an order with CME-specific slippage and fees.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data
            is_rth: True if Regular Trading Hours
            minutes_to_settlement: Minutes until daily settlement
            is_roll_period: True if within roll period
            **kwargs: Additional arguments

        Returns:
            Fill object with execution details, or None if not filled

        Notes:
            - MARKET orders always fill (at worst price in bar)
            - LIMIT orders fill if price touches limit
            - Slippage and fees are computed based on CME model
        """
        # Calculate participation ratio
        participation = 0.0
        if market.adv is not None and market.adv > 0:
            mid_price = market.get_mid_price() or bar.typical_price
            order_notional = order.qty * mid_price
            participation = order_notional / market.adv

        # Compute slippage
        slippage_bps = self._slippage.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation,
            is_rth=is_rth,
            minutes_to_settlement=minutes_to_settlement,
            is_roll_period=is_roll_period,
        )

        # Determine fill price
        mid_price = market.get_mid_price() or bar.typical_price
        slippage_fraction = slippage_bps / 10000

        if order.order_type.upper() == "MARKET":
            if order.side.upper() == "BUY":
                # Buy: price increases with slippage
                fill_price = mid_price * (1 + slippage_fraction)
                # Cap at bar high
                fill_price = min(fill_price, bar.high)
            else:
                # Sell: price decreases with slippage
                fill_price = mid_price * (1 - slippage_fraction)
                # Cap at bar low
                fill_price = max(fill_price, bar.low)

            filled = True
            liquidity = "taker"

        elif order.order_type.upper() == "LIMIT":
            limit_price = order.limit_price
            if limit_price is None:
                return None

            if order.side.upper() == "BUY":
                # Buy limit fills if bar low <= limit
                if bar.low <= limit_price:
                    fill_price = min(limit_price, bar.low)
                    filled = True
                    # Check if aggressive (crossing spread)
                    if limit_price >= market.ask:
                        liquidity = "taker"
                        fill_price = market.ask * (1 + slippage_fraction / 2)
                    else:
                        liquidity = "maker"
                else:
                    filled = False
            else:
                # Sell limit fills if bar high >= limit
                if bar.high >= limit_price:
                    fill_price = max(limit_price, bar.high)
                    filled = True
                    # Check if aggressive
                    if limit_price <= market.bid:
                        liquidity = "taker"
                        fill_price = market.bid * (1 - slippage_fraction / 2)
                    else:
                        liquidity = "maker"
                else:
                    filled = False

            if not filled:
                return None
        else:
            # Unknown order type
            logger.warning(f"Unknown order type: {order.order_type}")
            return None

        # Compute fee
        notional = fill_price * order.qty
        fee = self._fees.compute_fee(
            notional=notional,
            side=order.side,
            liquidity=liquidity,
            qty=order.qty,
            symbol=order.symbol,
        )

        return Fill(
            price=fill_price,
            qty=order.qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity=liquidity,
            timestamp=market.timestamp,
            notional=notional,
            metadata={
                "symbol": order.symbol,
                "side": order.side,
                "client_order_id": order.client_order_id,
            },
        )

    def set_circuit_breaker_state(self, state: CircuitBreakerState) -> None:
        """Set circuit breaker state."""
        self._slippage.set_circuit_breaker_state(state)

    def set_roll_period(self, is_roll: bool) -> None:
        """Set roll period flag."""
        self._slippage.set_roll_period(is_roll)

    def estimate_cost(
        self,
        symbol: str,
        qty: float,
        price: float,
        adv: float,
        is_rth: bool = True,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Args:
            symbol: Product symbol
            qty: Number of contracts
            price: Current price
            adv: Average daily volume (notional)
            is_rth: True if RTH

        Returns:
            Dictionary with slippage_cost, fee_cost, total_cost
        """
        notional = qty * price

        impact = self._slippage.estimate_impact_cost(
            notional=notional,
            adv=adv,
            symbol=symbol,
            is_rth=is_rth,
        )

        fee_breakdown = self._fees.get_fee_breakdown(symbol, qty)

        return {
            "slippage_bps": impact["impact_bps"],
            "slippage_cost": impact["impact_cost"],
            "fee_cost": fee_breakdown["total_fee"],
            "total_cost": impact["impact_cost"] + fee_breakdown["total_fee"],
            "participation_ratio": impact["participation_ratio"],
            "recommendation": impact["recommendation"],
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_cme_slippage_provider(
    profile: str = "default",
) -> CMESlippageProvider:
    """
    Create CME slippage provider from profile.

    Args:
        profile: Profile name (default, conservative, aggressive,
                 equity_index, metals, energy)

    Returns:
        Configured CMESlippageProvider
    """
    return CMESlippageProvider.from_profile(profile)


def create_cme_execution_provider(
    profile: str = "default",
) -> CMEL2ExecutionProvider:
    """
    Create CME execution provider.

    Args:
        profile: Slippage profile name

    Returns:
        Configured CMEL2ExecutionProvider
    """
    slippage_config = CME_SLIPPAGE_PROFILES.get(profile, CME_SLIPPAGE_PROFILES["default"])
    return CMEL2ExecutionProvider(slippage_config=slippage_config)


def get_tick_size(symbol: str) -> Decimal:
    """
    Get tick size for a CME product.

    Args:
        symbol: Product symbol

    Returns:
        Tick size as Decimal
    """
    return TICK_SIZES.get(symbol.upper(), DEFAULT_TICK_SIZE)


def calculate_spread_in_bps(
    symbol: str,
    spread_ticks: float,
    mid_price: float,
) -> float:
    """
    Calculate spread in basis points.

    Args:
        symbol: Product symbol
        spread_ticks: Spread in number of ticks
        mid_price: Current mid price

    Returns:
        Spread in basis points
    """
    tick_size = float(get_tick_size(symbol))
    spread_value = spread_ticks * tick_size
    return (spread_value / mid_price) * 10000 if mid_price > 0 else 0
