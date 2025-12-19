# -*- coding: utf-8 -*-
"""
execution_providers.py
Multi-asset execution provider interfaces and implementations.

Execution provider abstractions and default implementations.

This module contains protocol-style interfaces (``SlippageProvider``,
``FillProvider``, ``FeeProvider``) together with the in-tree reference
implementations used by the crypto (Binance-style), equity, futures and
forex fee/impact models. Level-3 depth-aware simulation lives in ``lob/`` and
``execution_providers_l3.py``; here we focus on the L1/L2 models that are wired
into the Python stack by default.

References:
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
"""

from __future__ import annotations

import enum
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AssetClass(enum.Enum):
    """Asset class enumeration for execution providers."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"  # Full FX support via parametric slippage/fee models


class OrderSide(enum.Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class LiquidityRole(enum.Enum):
    """Liquidity role (maker/taker) for fee calculation."""
    MAKER = "maker"
    TAKER = "taker"
    UNKNOWN = "unknown"


# Default constants
_DEFAULT_SPREAD_BPS = 5.0
_DEFAULT_IMPACT_COEF = 0.1  # Almgren-Chriss style
_DEFAULT_VOLATILITY_SCALE = 1.0
_MIN_PARTICIPATION = 1e-12
_MAX_SLIPPAGE_BPS = 500.0  # Safety cap


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class MarketState:
    """
    Market snapshot for execution decisions.

    Captures the current market state including quotes, liquidity,
    and optional order book depth (L3).

    Attributes:
        timestamp: Unix timestamp in milliseconds
        bid: Best bid price
        ask: Best ask price
        bid_size: Size at best bid
        ask_size: Size at best ask
        last_price: Last traded price
        mid_price: Mid-market price (computed if None)
        spread_bps: Spread in basis points (computed if None)
        adv: Average daily volume (optional)
        volatility: Current volatility estimate (optional)
        bid_depth: L3 bid depth [(price, size), ...]
        ask_depth: L3 ask depth [(price, size), ...]
    """
    timestamp: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    last_price: Optional[float] = None
    mid_price: Optional[float] = None
    spread_bps: Optional[float] = None
    adv: Optional[float] = None
    volatility: Optional[float] = None
    # L3 extension points
    bid_depth: Optional[List[Tuple[float, float]]] = None
    ask_depth: Optional[List[Tuple[float, float]]] = None

    def get_mid_price(self) -> Optional[float]:
        """Get mid-price, computing from bid/ask if not provided."""
        if self.mid_price is not None and math.isfinite(self.mid_price):
            return self.mid_price
        if self.bid is not None and self.ask is not None:
            if math.isfinite(self.bid) and math.isfinite(self.ask):
                return (self.bid + self.ask) / 2.0
        if self.last_price is not None and math.isfinite(self.last_price):
            return self.last_price
        return None

    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points, computing if not provided."""
        if self.spread_bps is not None and math.isfinite(self.spread_bps):
            return self.spread_bps
        mid = self.get_mid_price()
        if mid is None or mid <= 0:
            return None
        if self.bid is not None and self.ask is not None:
            if math.isfinite(self.bid) and math.isfinite(self.ask):
                spread = self.ask - self.bid
                return (spread / mid) * 10000.0
        return None

    def get_reference_price(self, side: Union[str, OrderSide]) -> Optional[float]:
        """Get reference price for a given side (bid for sell, ask for buy)."""
        side_str = side.value if isinstance(side, OrderSide) else str(side).upper()
        if side_str == "BUY":
            if self.ask is not None and math.isfinite(self.ask):
                return self.ask
        elif side_str == "SELL":
            if self.bid is not None and math.isfinite(self.bid):
                return self.bid
        return self.get_mid_price()


@dataclass(frozen=True)
class Order:
    """
    Order for execution.

    Represents an order to be executed in the simulation.

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
        side: Order side ("BUY" or "SELL")
        qty: Quantity to trade
        order_type: Order type ("MARKET" or "LIMIT")
        limit_price: Limit price (required for LIMIT orders)
        notional: Optional notional value (computed if not provided)
        asset_class: Asset class for this order
        client_order_id: Optional client-provided order ID
        time_in_force: Time in force (GTC, IOC, FOK)
    """
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: float
    order_type: str  # "MARKET" | "LIMIT"
    limit_price: Optional[float] = None
    notional: Optional[float] = None
    asset_class: AssetClass = AssetClass.CRYPTO
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"

    def __post_init__(self) -> None:
        # Validation (for frozen dataclass, we just log warnings)
        if self.qty <= 0:
            logger.warning("Order qty should be positive: %s", self.qty)
        if self.order_type == "LIMIT" and self.limit_price is None:
            logger.warning("LIMIT order missing limit_price")

    def get_notional(self, price: float) -> float:
        """Compute notional value at given price."""
        if self.notional is not None and self.notional > 0:
            return self.notional
        return abs(self.qty) * price

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return str(self.side).upper() == "BUY"


@dataclass
class Fill:
    """
    Execution result (fill).

    Represents the result of an order execution attempt.

    Attributes:
        price: Fill price
        qty: Filled quantity
        fee: Fee amount
        slippage_bps: Slippage in basis points
        liquidity: Liquidity role ("maker" or "taker")
        timestamp: Fill timestamp (optional)
        notional: Trade notional value
        fee_breakdown: Detailed fee breakdown (optional)
        metadata: Additional execution metadata
    """
    price: float
    qty: float
    fee: float
    slippage_bps: float
    liquidity: str  # "maker" | "taker"
    timestamp: Optional[int] = None
    notional: Optional[float] = None
    fee_breakdown: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.notional is None:
            self.notional = self.price * self.qty

    @property
    def total_cost(self) -> float:
        """Total execution cost (slippage + fee)."""
        notional = self.notional or (self.price * self.qty)
        slippage_cost = notional * self.slippage_bps / 10000.0
        return slippage_cost + self.fee

    @property
    def total_cost_bps(self) -> float:
        """Total execution cost in basis points."""
        notional = self.notional or (self.price * self.qty)
        if notional <= 0:
            return 0.0
        return (self.total_cost / notional) * 10000.0


@dataclass
class BarData:
    """
    OHLCV bar data for fill simulation.

    Used by L2 providers for intrabar price interpolation.

    Attributes:
        open: Bar open price
        high: Bar high price
        low: Bar low price
        close: Bar close price
        volume: Bar volume
        timestamp: Bar start timestamp
        timeframe_ms: Bar timeframe in milliseconds
    """
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    timestamp: Optional[int] = None
    timeframe_ms: Optional[int] = None

    def contains_price(self, price: float, tolerance: float = 0.0) -> bool:
        """Check if price is within bar's high-low range."""
        return (self.low - tolerance) <= price <= (self.high + tolerance)

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def bar_range(self) -> float:
        """Bar price range (high - low)."""
        return self.high - self.low


# =============================================================================
# Protocol Definitions (Interfaces)
# =============================================================================

@runtime_checkable
class SlippageProvider(Protocol):
    """
    Abstract slippage computation protocol.

    Implementations compute expected slippage based on order details,
    market state, and participation ratio.

    L2 Example: √participation model (Almgren-Chriss)
    L3 Example: LOB walk-through simulation
    """

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute expected slippage in basis points.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / ADV (or bar volume)

        Returns:
            Expected slippage in basis points (always non-negative)
        """
        ...


@runtime_checkable
class FillProvider(Protocol):
    """
    Abstract fill logic protocol.

    Implementations determine if/how an order fills based on
    order type, market state, and bar data.

    L2 Example: OHLCV bar-based fills (check if limit touched)
    L3 Example: Matching engine simulation with queue position
    """

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt to fill order.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data (OHLCV)

        Returns:
            Fill if order executed, None if not filled
        """
        ...


@runtime_checkable
class FeeProvider(Protocol):
    """
    Abstract fee computation protocol.

    Implementations compute trading fees based on notional,
    side, liquidity role, and asset-specific rules.

    Examples:
    - Crypto: Maker/taker fee tiers (0.02%/0.04%)
    - Equity: Commission-free with regulatory fees (SEC, TAF)
    """

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute fee amount.

        Args:
            notional: Trade notional value (price * qty)
            side: Trade side ("BUY" or "SELL")
            liquidity: Liquidity role ("maker" or "taker")
            qty: Trade quantity

        Returns:
            Fee amount in quote currency
        """
        ...


@runtime_checkable
class ExecutionProvider(Protocol):
    """
    Combined execution provider protocol.

    High-level interface that combines slippage, fill, and fee
    computation into a single execution workflow.
    """

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute an order with full slippage/fee calculation.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data

        Returns:
            Fill with slippage and fees, or None if not filled
        """
        ...

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        ...


# =============================================================================
# L2 Implementations (Statistical Models)
# =============================================================================

class StatisticalSlippageProvider:
    """
    L2: Statistical slippage model (Almgren-Chriss style).

    Implements square-root market impact model:
        slippage = half_spread + impact_coef * sqrt(participation) * volatility_scale

    Based on empirical research showing market impact scales with
    the square root of participation ratio.

    References:
        - Almgren & Chriss (2001): sqrt impact scaling
        - Kyle (1985): Lambda model for market impact
        - Gatheral (2010): Transient vs permanent impact

    Attributes:
        impact_coef: Market impact coefficient (default: 0.1)
        spread_bps: Default half-spread in basis points
        volatility_scale: Volatility scaling factor
        min_slippage_bps: Minimum slippage floor
        max_slippage_bps: Maximum slippage cap
    """

    def __init__(
        self,
        impact_coef: float = _DEFAULT_IMPACT_COEF,
        spread_bps: float = _DEFAULT_SPREAD_BPS,
        volatility_scale: float = _DEFAULT_VOLATILITY_SCALE,
        min_slippage_bps: float = 0.0,
        max_slippage_bps: float = _MAX_SLIPPAGE_BPS,
    ) -> None:
        """
        Initialize statistical slippage provider.

        Args:
            impact_coef: Market impact coefficient (k in k*sqrt(v))
            spread_bps: Default half-spread in basis points
            volatility_scale: Volatility adjustment factor
            min_slippage_bps: Minimum slippage floor
            max_slippage_bps: Maximum slippage cap (safety limit)
        """
        self.impact_coef = float(impact_coef)
        self.spread_bps = float(spread_bps)
        self.volatility_scale = float(volatility_scale)
        self.min_slippage_bps = float(min_slippage_bps)
        self.max_slippage_bps = float(max_slippage_bps)

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute expected slippage using √participation model.

        Formula:
            slippage_bps = half_spread + k * sqrt(participation) * vol_scale * 10000

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / reference volume

        Returns:
            Expected slippage in basis points
        """
        # Get spread from market or use default
        spread = market.get_spread_bps()
        if spread is None or not math.isfinite(spread) or spread < 0:
            half_spread = self.spread_bps / 2.0
        else:
            half_spread = spread / 2.0

        # Sanitize participation ratio
        participation = max(_MIN_PARTICIPATION, abs(participation_ratio))

        # Volatility adjustment
        vol_factor = self.volatility_scale
        if market.volatility is not None and math.isfinite(market.volatility):
            vol_factor *= market.volatility

        # Square-root market impact (Almgren-Chriss)
        impact_bps = self.impact_coef * math.sqrt(participation) * vol_factor * 10000.0

        # Total slippage
        total_slippage = half_spread + impact_bps

        # Apply bounds
        total_slippage = max(self.min_slippage_bps, total_slippage)
        total_slippage = min(self.max_slippage_bps, total_slippage)

        return float(total_slippage)

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """
        Estimate market impact cost for a given trade size.

        Useful for pre-trade analytics and optimal execution planning.

        Args:
            notional: Trade notional value
            adv: Average daily volume (in quote currency)
            volatility: Annualized volatility (default: 2%)

        Returns:
            Dict with impact breakdown
        """
        if adv <= 0:
            return {"participation": 0.0, "impact_bps": 0.0, "impact_cost": 0.0}

        participation = notional / adv
        impact_bps = self.compute_slippage_bps(
            Order("", "BUY", 1.0, "MARKET"),
            MarketState(0, volatility=volatility),
            participation,
        )
        impact_cost = notional * impact_bps / 10000.0

        return {
            "participation": participation,
            "impact_bps": impact_bps,
            "impact_cost": impact_cost,
        }

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any],
        asset_class: Optional[AssetClass] = None,
    ) -> "StatisticalSlippageProvider":
        """
        Create SlippageProvider from configuration dict or object.

        Supports loading from:
        - Dict/mapping with slippage parameters
        - Pydantic config objects
        - YAML profile structure (profiles.equity, profiles.crypto)

        Args:
            config: Configuration dict or object
            asset_class: Optional asset class for profile selection

        Returns:
            StatisticalSlippageProvider instance
        """
        if config is None:
            # Return asset-class-specific defaults
            if asset_class == AssetClass.EQUITY:
                return cls(impact_coef=0.05, spread_bps=2.0)
            return cls()  # Crypto defaults

        # Extract parameters from config
        params: Dict[str, Any] = {}

        def _extract_value(cfg: Any, *keys: str, default: Any = None) -> Any:
            """Extract value from config trying multiple key names."""
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        # Check for profile-based config (profiles.equity or profiles.crypto)
        profiles = _extract_value(config, "profiles")
        if profiles is not None:
            profile_name = None
            if asset_class == AssetClass.EQUITY:
                profile_name = "equity"
            elif asset_class == AssetClass.CRYPTO:
                profile_name = "crypto"
            elif asset_class == AssetClass.FUTURES:
                profile_name = "crypto_futures"

            if profile_name:
                profile = _extract_value(profiles, profile_name, "default")
                if profile is not None:
                    config = profile

        # Extract slippage parameters
        params["impact_coef"] = float(
            _extract_value(config, "impact_coef", "k", "impact_coefficient", default=0.1)
        )
        params["spread_bps"] = float(
            _extract_value(config, "spread_bps", "default_spread_bps", "half_spread_bps", default=5.0)
        )
        params["volatility_scale"] = float(
            _extract_value(config, "volatility_scale", "vol_scale", default=1.0)
        )
        params["min_slippage_bps"] = float(
            _extract_value(config, "min_slippage_bps", "min_bps", default=0.0)
        )
        params["max_slippage_bps"] = float(
            _extract_value(config, "max_slippage_bps", "max_bps", default=500.0)
        )

        return cls(**params)

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
        profiles_config: Optional[Mapping[str, Any]] = None,
    ) -> "StatisticalSlippageProvider":
        """
        Create SlippageProvider from named profile.

        Args:
            profile_name: Profile name ("equity", "crypto", "crypto_futures", etc.)
            profiles_config: Optional profiles configuration dict

        Returns:
            StatisticalSlippageProvider instance
        """
        # Default profiles (calibrated values)
        default_profiles = {
            "equity": {
                "spread_bps": 2.0,
                "impact_coef": 0.05,
                "volatility_scale": 1.0,
                "min_slippage_bps": 0.5,
                "max_slippage_bps": 200.0,
            },
            "crypto": {
                "spread_bps": 5.0,
                "impact_coef": 0.10,
                "volatility_scale": 1.0,
                "min_slippage_bps": 1.0,
                "max_slippage_bps": 500.0,
            },
            "crypto_futures": {
                "spread_bps": 4.0,
                "impact_coef": 0.09,
                "volatility_scale": 1.2,
                "min_slippage_bps": 0.5,
                "max_slippage_bps": 500.0,
            },
            "default": {
                "spread_bps": 5.0,
                "impact_coef": 0.10,
                "volatility_scale": 1.0,
                "min_slippage_bps": 0.0,
                "max_slippage_bps": 500.0,
            },
        }

        # Merge with provided profiles
        if profiles_config:
            for name, profile in profiles_config.items():
                if name in default_profiles and isinstance(profile, Mapping):
                    default_profiles[name].update(profile)
                elif isinstance(profile, Mapping):
                    default_profiles[name] = dict(profile)

        profile = default_profiles.get(profile_name, default_profiles["default"])

        # Filter out unknown keys (e.g., 'tiers' from YAML config)
        valid_keys = {
            "impact_coef",
            "spread_bps",
            "volatility_scale",
            "min_slippage_bps",
            "max_slippage_bps",
        }
        filtered_profile = {k: v for k, v in profile.items() if k in valid_keys}

        return cls(**filtered_profile)


# =============================================================================
# Crypto Parametric TCA Model
# =============================================================================

class VolatilityRegime(enum.Enum):
    """Volatility regime classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# =============================================================================
# Forex Parametric TCA Model Enums
# =============================================================================

class ForexSession(enum.Enum):
    """
    Forex trading session classification.

    Sessions are based on major financial centers:
    - Sydney: 21:00-06:00 UTC (lowest liquidity for majors)
    - Tokyo: 00:00-09:00 UTC (JPY pairs active)
    - London: 07:00-16:00 UTC (highest liquidity)
    - New York: 12:00-21:00 UTC (USD pairs active)
    - London/NY overlap: 12:00-16:00 UTC (peak liquidity)
    - Tokyo/London overlap: 07:00-09:00 UTC (moderate liquidity)

    References:
        - BIS Triennial Central Bank Survey (2022)
        - King, Osler, Rime (2012): "Foreign Exchange Market Structure"
    """
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    OFF_HOURS = "off_hours"
    WEEKEND = "weekend"


class PairType(enum.Enum):
    """
    Currency pair classification by liquidity and spread characteristics.

    Classification:
    - MAJOR: G7 pairs with USD (highest liquidity)
    - MINOR: Cross pairs without USD (EUR/GBP, EUR/CHF, etc.)
    - CROSS: JPY crosses and other liquid crosses
    - EXOTIC: Emerging market currencies (higher spreads)

    References:
        - BIS Triennial Survey (2022): Currency pair turnover data
        - Berger et al. (2008): "The Development of the Global FX Market"
    """
    MAJOR = "major"
    MINOR = "minor"
    CROSS = "cross"
    EXOTIC = "exotic"


@dataclass
class CryptoParametricConfig:
    """
    Configuration for CryptoParametricSlippageProvider.

    All parameters are configurable with research-backed defaults.

    Attributes:
        impact_coef_base: Base market impact coefficient (k in √participation)
        impact_coef_range: (min, max) range for adaptive impact
        spread_bps: Default half-spread in basis points

        vol_regime_multipliers: Multipliers for each volatility regime
        vol_lookback_periods: Number of periods for volatility regime detection
        vol_regime_thresholds: (low, high) percentile thresholds for regime classification

        imbalance_penalty_max: Maximum penalty for order book imbalance
        funding_stress_sensitivity: Multiplier for funding rate stress

        tod_curve: Time-of-day liquidity curve (24 hours, UTC)
        btc_correlation_decay_factor: Decay factor for BTC correlation

        whale_threshold: Participation ratio threshold for whale detection
        whale_twap_adjustment: TWAP-style adjustment factor for whale orders

        asymmetric_sell_premium: Extra cost for sells in downtrend
        downtrend_threshold: Return threshold to detect downtrend

        min_slippage_bps: Minimum slippage floor
        max_slippage_bps: Maximum slippage cap
    """
    # Base impact parameters (Almgren-Chriss)
    impact_coef_base: float = 0.10
    impact_coef_range: Tuple[float, float] = (0.05, 0.15)
    spread_bps: float = 5.0

    # Volatility regime (Cont 2001)
    vol_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.8,
        "normal": 1.0,
        "high": 1.5,
    })
    vol_lookback_periods: int = 20
    vol_regime_thresholds: Tuple[float, float] = (25.0, 75.0)  # Percentiles

    # Order book imbalance (Cont et al. 2014)
    imbalance_penalty_max: float = 0.3

    # Funding rate stress (perp-specific)
    funding_stress_sensitivity: float = 10.0

    # Time-of-day curve (Binance research: Asia/EU/US sessions)
    # Values represent liquidity multipliers (lower = less liquidity = more slippage)
    tod_curve: Dict[int, float] = field(default_factory=lambda: {
        # Asia session (00:00-08:00 UTC)
        0: 0.85, 1: 0.80, 2: 0.75, 3: 0.70, 4: 0.75, 5: 0.80, 6: 0.85, 7: 0.90,
        # EU session (08:00-16:00 UTC)
        8: 0.95, 9: 1.00, 10: 1.05, 11: 1.05, 12: 1.00, 13: 1.05, 14: 1.10, 15: 1.10,
        # US session (16:00-24:00 UTC) - overlap with EU is peak
        16: 1.15, 17: 1.15, 18: 1.10, 19: 1.05, 20: 1.00, 21: 0.95, 22: 0.90, 23: 0.85,
    })

    # BTC correlation decay (altcoin liquidity fragmentation)
    btc_correlation_decay_factor: float = 0.5

    # Whale detection
    whale_threshold: float = 0.01  # 1% of ADV
    whale_twap_adjustment: float = 0.7  # Reduce impact as if TWAP'd

    # Asymmetric slippage
    asymmetric_sell_premium: float = 0.2  # 20% extra cost for panic sells
    downtrend_threshold: float = -0.02  # -2% recent return = downtrend

    # Bounds
    min_slippage_bps: float = 1.0
    max_slippage_bps: float = 500.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.impact_coef_base <= 0:
            raise ValueError("impact_coef_base must be positive")
        if self.impact_coef_range[0] >= self.impact_coef_range[1]:
            raise ValueError("impact_coef_range must have min < max")
        if self.whale_threshold <= 0:
            raise ValueError("whale_threshold must be positive")
        if self.vol_lookback_periods < 2:
            raise ValueError("vol_lookback_periods must be >= 2")


class CryptoParametricSlippageProvider:
    """
    L2+: Smart parametric TCA model for cryptocurrency markets.

    Extends the basic √participation model (Almgren-Chriss) with multiple
    crypto-specific factors that capture real market microstructure effects.

    Total Slippage Formula:
        slippage = half_spread
            × (1 + k × √participation)
            × vol_regime_mult
            × (1 + imbalance_penalty × sign(side))
            × funding_stress
            × (1 / tod_factor)  # Inverted: low liquidity = high slippage
            × correlation_decay
            × asymmetric_adjustment

    Smart Features:
        - **Regime Detection**: Auto-detects low/normal/high volatility from returns
        - **Adaptive Impact**: Coefficient k adjusts based on trailing fill quality
        - **Asymmetric Slippage**: Sells in downtrend cost more (panic liquidity)
        - **Whale Detection**: Large orders (Q/ADV > 1%) get TWAP-adjusted model

    References:
        - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
        - Cont (2001): "Empirical Properties of Asset Returns: Stylized Facts"
        - Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
        - Kyle (1985): "Continuous Auctions and Insider Trading"
        - Cartea, Jaimungal, Penalva (2015): "Algorithmic and HF Trading", Ch. 10

    Attributes:
        config: CryptoParametricConfig with all tunable parameters
        _adaptive_k: Current adaptive impact coefficient
        _fill_quality_history: Recent fill quality for adaptive adjustment
    """

    def __init__(
        self,
        config: Optional[CryptoParametricConfig] = None,
        *,
        # Convenience overrides for common parameters
        impact_coef: Optional[float] = None,
        spread_bps: Optional[float] = None,
        min_slippage_bps: Optional[float] = None,
        max_slippage_bps: Optional[float] = None,
    ) -> None:
        """
        Initialize crypto parametric slippage provider.

        Args:
            config: Full configuration object (uses defaults if None)
            impact_coef: Override base impact coefficient
            spread_bps: Override default spread
            min_slippage_bps: Override minimum slippage
            max_slippage_bps: Override maximum slippage
        """
        self.config = config or CryptoParametricConfig()

        # Apply convenience overrides
        if impact_coef is not None:
            self.config = CryptoParametricConfig(
                **{**self.config.__dict__, "impact_coef_base": float(impact_coef)}
            )
        if spread_bps is not None:
            self.config = CryptoParametricConfig(
                **{**self.config.__dict__, "spread_bps": float(spread_bps)}
            )
        if min_slippage_bps is not None:
            self.config = CryptoParametricConfig(
                **{**self.config.__dict__, "min_slippage_bps": float(min_slippage_bps)}
            )
        if max_slippage_bps is not None:
            self.config = CryptoParametricConfig(
                **{**self.config.__dict__, "max_slippage_bps": float(max_slippage_bps)}
            )

        # Adaptive impact coefficient (starts at base)
        self._adaptive_k: float = self.config.impact_coef_base

        # Fill quality history for adaptive adjustment
        # Stores (predicted_slippage, actual_slippage) tuples
        self._fill_quality_history: List[Tuple[float, float]] = []
        self._max_history_size: int = 100

        # Volatility history for regime detection
        self._volatility_history: List[float] = []

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        *,
        # Extended parameters for full model
        funding_rate: Optional[float] = None,
        btc_correlation: Optional[float] = None,
        hour_utc: Optional[int] = None,
        recent_returns: Optional[Sequence[float]] = None,
        bid_depth_total: Optional[float] = None,
        ask_depth_total: Optional[float] = None,
    ) -> float:
        """
        Compute expected slippage using the full parametric model.

        The model combines multiple factors to produce a realistic
        slippage estimate that accounts for crypto-specific dynamics.

        Formula:
            slippage = half_spread × (1 + k × √participation) × Π(factors)

        where Π(factors) includes volatility regime, imbalance, funding,
        time-of-day, BTC correlation, and asymmetric adjustments.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / ADV (or bar volume)
            funding_rate: Perpetual funding rate (optional, -0.01 to +0.01 typical)
            btc_correlation: Correlation with BTC (0.0 to 1.0, optional)
            hour_utc: Current hour in UTC (0-23, optional)
            recent_returns: Recent price returns for regime detection
            bid_depth_total: Total bid depth for imbalance calculation
            ask_depth_total: Total ask depth for imbalance calculation

        Returns:
            Expected slippage in basis points (always non-negative)

        Example:
            >>> provider = CryptoParametricSlippageProvider()
            >>> slippage = provider.compute_slippage_bps(
            ...     order=Order("ETHUSDT", "BUY", 10.0, "MARKET"),
            ...     market=MarketState(timestamp=0, bid=2000.0, ask=2001.0, adv=50_000_000),
            ...     participation_ratio=0.005,
            ...     funding_rate=0.0003,  # Slightly positive
            ...     btc_correlation=0.85,
            ...     hour_utc=14,  # EU session
            ...     recent_returns=[-0.01, 0.005, -0.008, 0.003],  # Recent returns
            ... )
        """
        # =====================================================================
        # 1. Base spread component
        # =====================================================================
        spread = market.get_spread_bps()
        if spread is None or not math.isfinite(spread) or spread < 0:
            half_spread = self.config.spread_bps / 2.0
        else:
            half_spread = spread / 2.0

        # =====================================================================
        # 2. √Participation impact (Almgren-Chriss 2001)
        # =====================================================================
        participation = max(_MIN_PARTICIPATION, abs(participation_ratio))

        # Check for whale orders
        is_whale = participation >= self.config.whale_threshold
        k_effective = self._adaptive_k

        if is_whale:
            # Whale orders get TWAP-adjusted impact (as if slicing the order)
            # Reference: Cartea et al. (2015), optimal execution theory
            k_effective *= self.config.whale_twap_adjustment
            # Also reduce effective participation as if TWAP'd over time
            participation = participation * self.config.whale_twap_adjustment

        impact_bps = k_effective * math.sqrt(participation) * 10000.0

        # =====================================================================
        # 3. Volatility regime multiplier (Cont 2001)
        # =====================================================================
        vol_regime = self.detect_volatility_regime(recent_returns)
        vol_mult = self.config.vol_regime_multipliers.get(vol_regime.value, 1.0)

        # Also incorporate real-time volatility from market state if available
        if market.volatility is not None and math.isfinite(market.volatility):
            # Scale based on deviation from typical crypto volatility (~60% annualized ≈ 0.04 daily)
            vol_baseline = 0.04
            vol_scale = market.volatility / vol_baseline if vol_baseline > 0 else 1.0
            vol_mult *= min(2.0, max(0.5, vol_scale))  # Clamp to [0.5, 2.0]

        # =====================================================================
        # 4. Order book imbalance penalty (Cont et al. 2014)
        # =====================================================================
        imbalance_factor = 1.0
        imbalance = self._compute_order_book_imbalance(
            market, bid_depth_total, ask_depth_total
        )

        if imbalance is not None:
            # Imbalance is in [-1, 1]: positive = more bids, negative = more asks
            # For BUY: negative imbalance (more asks) is favorable
            # For SELL: positive imbalance (more bids) is favorable
            is_buy = str(order.side).upper() == "BUY"

            # Penalty when trading against imbalance
            if is_buy:
                # Buying when asks are thin (positive imbalance) costs more
                penalty = max(0.0, imbalance) * self.config.imbalance_penalty_max
            else:
                # Selling when bids are thin (negative imbalance) costs more
                penalty = max(0.0, -imbalance) * self.config.imbalance_penalty_max

            imbalance_factor = 1.0 + penalty

        # =====================================================================
        # 5. Funding rate stress (perp-specific, empirical)
        # =====================================================================
        funding_factor = 1.0
        if funding_rate is not None and math.isfinite(funding_rate):
            # High absolute funding = crowded trade = stress
            # Reference: Empirical observation on Binance/FTX perps
            funding_factor = 1.0 + abs(funding_rate) * self.config.funding_stress_sensitivity

        # =====================================================================
        # 6. Time-of-day liquidity curve (Binance research)
        # =====================================================================
        tod_factor = self.get_time_of_day_factor(hour_utc)
        # Invert: high liquidity (factor > 1) reduces slippage
        tod_adjustment = 1.0 / max(0.5, tod_factor)  # Clamp to prevent extreme values

        # =====================================================================
        # 7. BTC correlation decay (altcoin fragmentation)
        # =====================================================================
        correlation_decay = 1.0
        if btc_correlation is not None and math.isfinite(btc_correlation):
            # Lower correlation with BTC = less liquidity spillover = more slippage
            # Reference: Empirical observation of altcoin order book depth
            corr_clamped = max(0.0, min(1.0, btc_correlation))
            correlation_decay = 1.0 + (1.0 - corr_clamped) * self.config.btc_correlation_decay_factor

        # =====================================================================
        # 8. Asymmetric slippage for panic sells
        # =====================================================================
        asymmetric_factor = 1.0
        is_sell = str(order.side).upper() == "SELL"

        if is_sell and recent_returns is not None and len(recent_returns) > 0:
            # Check if we're in a downtrend
            returns_array = list(recent_returns)[-self.config.vol_lookback_periods:]
            if len(returns_array) > 0:
                cumulative_return = sum(returns_array)
                if cumulative_return < self.config.downtrend_threshold:
                    # Panic selling premium: liquidity dries up when everyone exits
                    # Reference: Brunnermeier & Pedersen (2009), liquidity spirals
                    asymmetric_factor = 1.0 + self.config.asymmetric_sell_premium

        # =====================================================================
        # 9. Combine all factors
        # =====================================================================
        total_slippage = (
            half_spread
            * (1.0 + impact_bps / half_spread if half_spread > 0 else 1.0 + impact_bps)
            * vol_mult
            * imbalance_factor
            * funding_factor
            * tod_adjustment
            * correlation_decay
            * asymmetric_factor
        )

        # Apply bounds
        total_slippage = max(self.config.min_slippage_bps, total_slippage)
        total_slippage = min(self.config.max_slippage_bps, total_slippage)

        return float(total_slippage)

    def detect_volatility_regime(
        self,
        returns: Optional[Sequence[float]],
    ) -> VolatilityRegime:
        """
        Detect current volatility regime from recent returns.

        Uses rolling volatility percentile to classify into
        LOW, NORMAL, or HIGH regime.

        Method:
            1. Compute realized volatility from returns
            2. Update volatility history
            3. Compare to historical percentiles

        Args:
            returns: Sequence of recent price returns

        Returns:
            VolatilityRegime classification

        Reference:
            - Cont (2001): Volatility clustering in asset returns
        """
        if returns is None or len(returns) < 2:
            return VolatilityRegime.NORMAL

        # Compute realized volatility (standard deviation of returns)
        returns_list = list(returns)[-self.config.vol_lookback_periods:]
        if len(returns_list) < 2:
            return VolatilityRegime.NORMAL

        mean_ret = sum(returns_list) / len(returns_list)
        variance = sum((r - mean_ret) ** 2 for r in returns_list) / (len(returns_list) - 1)
        current_vol = math.sqrt(variance) if variance > 0 else 0.0

        if current_vol == 0:
            return VolatilityRegime.NORMAL

        # Update history
        self._volatility_history.append(current_vol)
        if len(self._volatility_history) > 500:  # Keep bounded
            self._volatility_history = self._volatility_history[-250:]

        # Need sufficient history for percentile comparison
        if len(self._volatility_history) < 20:
            return VolatilityRegime.NORMAL

        # Compute percentile rank
        sorted_vols = sorted(self._volatility_history)
        rank = sum(1 for v in sorted_vols if v < current_vol)
        percentile = (rank / len(sorted_vols)) * 100.0

        low_threshold, high_threshold = self.config.vol_regime_thresholds

        if percentile < low_threshold:
            return VolatilityRegime.LOW
        elif percentile > high_threshold:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL

    def get_time_of_day_factor(self, hour_utc: Optional[int]) -> float:
        """
        Get liquidity factor based on time of day.

        The factor represents relative liquidity at each hour,
        calibrated from Binance trading activity patterns:
        - Asia session (00:00-08:00 UTC): Lower liquidity
        - EU session (08:00-16:00 UTC): Increasing liquidity
        - US session (16:00-24:00 UTC): Peak during EU/US overlap

        Args:
            hour_utc: Current hour in UTC (0-23)

        Returns:
            Liquidity factor (higher = more liquidity = less slippage)

        Reference:
            - Binance trading volume analysis by hour
            - Pagano & Schwartz (2003): Time-of-day effects in equity markets
        """
        if hour_utc is None:
            return 1.0  # Default to neutral

        hour_clamped = int(hour_utc) % 24
        return self.config.tod_curve.get(hour_clamped, 1.0)

    def _compute_order_book_imbalance(
        self,
        market: MarketState,
        bid_depth_total: Optional[float] = None,
        ask_depth_total: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute order book imbalance.

        Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

        Returns value in [-1, 1]:
        - Positive: More bids than asks (bullish pressure)
        - Negative: More asks than bids (bearish pressure)

        Args:
            market: Market state (may have bid_size, ask_size)
            bid_depth_total: Optional pre-computed total bid depth
            ask_depth_total: Optional pre-computed total ask depth

        Returns:
            Imbalance in [-1, 1], or None if unavailable

        Reference:
            - Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
        """
        # Try explicit depth first
        bid_depth = bid_depth_total
        ask_depth = ask_depth_total

        # Fall back to top-of-book from market state
        if bid_depth is None and market.bid_size is not None:
            bid_depth = market.bid_size
        if ask_depth is None and market.ask_size is not None:
            ask_depth = market.ask_size

        # Fall back to L3 depth if available
        if bid_depth is None and market.bid_depth is not None:
            bid_depth = sum(size for _, size in market.bid_depth)
        if ask_depth is None and market.ask_depth is not None:
            ask_depth = sum(size for _, size in market.ask_depth)

        # Compute imbalance if both available
        if bid_depth is not None and ask_depth is not None:
            total = bid_depth + ask_depth
            if total > 0:
                return (bid_depth - ask_depth) / total

        return None

    def update_fill_quality(
        self,
        predicted_slippage_bps: float,
        actual_slippage_bps: float,
    ) -> None:
        """
        Update adaptive impact coefficient based on fill quality.

        Tracks the ratio of actual to predicted slippage and adjusts
        the impact coefficient k to minimize prediction error.

        This implements a simple adaptive learning rule:
            k_new = k_old × (1 + α × (actual/predicted - 1))

        where α is a learning rate that decays with history size.

        Args:
            predicted_slippage_bps: Predicted slippage at order time
            actual_slippage_bps: Realized slippage after fill

        Reference:
            - Online learning for execution algorithms
            - Almgren (2003): "Optimal Execution with Nonlinear Impact Functions"
        """
        if predicted_slippage_bps <= 0 or actual_slippage_bps < 0:
            return

        # Add to history
        self._fill_quality_history.append((predicted_slippage_bps, actual_slippage_bps))

        # Trim history
        if len(self._fill_quality_history) > self._max_history_size:
            self._fill_quality_history = self._fill_quality_history[-self._max_history_size:]

        # Compute adjustment based on recent prediction errors
        if len(self._fill_quality_history) >= 10:
            recent = self._fill_quality_history[-20:]  # Look at last 20
            ratios = [actual / pred if pred > 0 else 1.0 for pred, actual in recent]
            avg_ratio = sum(ratios) / len(ratios)

            # Learning rate decays with history size
            learning_rate = 0.1 / math.sqrt(len(self._fill_quality_history))

            # Update adaptive k
            adjustment = 1.0 + learning_rate * (avg_ratio - 1.0)
            new_k = self._adaptive_k * adjustment

            # Clamp to configured range
            min_k, max_k = self.config.impact_coef_range
            self._adaptive_k = max(min_k, min(max_k, new_k))

    def reset_adaptive_state(self) -> None:
        """Reset adaptive state to initial values."""
        self._adaptive_k = self.config.impact_coef_base
        self._fill_quality_history.clear()
        self._volatility_history.clear()

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        side: str = "BUY",
        volatility: Optional[float] = None,
        funding_rate: Optional[float] = None,
        btc_correlation: Optional[float] = None,
        hour_utc: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Pre-trade cost estimation with full model.

        Useful for optimal execution planning and trade scheduling.

        Args:
            notional: Planned trade notional value
            adv: Average daily volume
            side: Trade side ("BUY" or "SELL")
            volatility: Expected volatility
            funding_rate: Current funding rate
            btc_correlation: BTC correlation for altcoins
            hour_utc: Planned execution hour (UTC)

        Returns:
            Dict with detailed cost breakdown and recommendations
        """
        if adv <= 0:
            return {
                "participation": 0.0,
                "impact_bps": 0.0,
                "impact_cost": 0.0,
                "is_whale": False,
                "recommendation": "Unable to estimate: ADV is zero",
            }

        participation = notional / adv
        is_whale = participation >= self.config.whale_threshold

        # Create dummy order and market for estimation
        dummy_order = Order("ESTIMATE", side, 1.0, "MARKET")
        dummy_market = MarketState(
            timestamp=0,
            volatility=volatility,
            adv=adv,
        )

        # Compute slippage
        impact_bps = self.compute_slippage_bps(
            dummy_order,
            dummy_market,
            participation,
            funding_rate=funding_rate,
            btc_correlation=btc_correlation,
            hour_utc=hour_utc,
        )

        impact_cost = notional * impact_bps / 10000.0

        # Generate recommendation
        recommendation = self._generate_execution_recommendation(
            participation, is_whale, hour_utc, funding_rate
        )

        return {
            "participation": participation,
            "participation_pct": participation * 100,
            "impact_bps": impact_bps,
            "impact_cost": impact_cost,
            "is_whale": is_whale,
            "vol_regime": self.detect_volatility_regime(None).value,
            "tod_factor": self.get_time_of_day_factor(hour_utc),
            "recommendation": recommendation,
        }

    def _generate_execution_recommendation(
        self,
        participation: float,
        is_whale: bool,
        hour_utc: Optional[int],
        funding_rate: Optional[float],
    ) -> str:
        """Generate execution recommendation based on analysis."""
        recommendations = []

        if is_whale:
            recommendations.append(
                f"Large order ({participation*100:.2f}% ADV). "
                "Consider TWAP/VWAP over 2-4 hours."
            )
        elif participation > 0.005:
            recommendations.append(
                "Moderate size order. Consider splitting into 2-3 tranches."
            )

        if hour_utc is not None:
            tod_factor = self.get_time_of_day_factor(hour_utc)
            if tod_factor < 0.85:
                recommendations.append(
                    f"Low liquidity hour (UTC {hour_utc}). "
                    "Consider delaying to EU/US overlap (14:00-18:00 UTC)."
                )
            elif tod_factor > 1.1:
                recommendations.append(
                    f"High liquidity hour (UTC {hour_utc}). Good execution window."
                )

        if funding_rate is not None and abs(funding_rate) > 0.001:
            direction = "long" if funding_rate > 0 else "short"
            recommendations.append(
                f"High funding ({funding_rate*100:.3f}%). "
                f"Crowded {direction} trade may see additional slippage."
            )

        if not recommendations:
            return "Standard execution conditions."

        return " ".join(recommendations)

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any, None],
        **kwargs: Any,
    ) -> "CryptoParametricSlippageProvider":
        """
        Create provider from configuration dict or object.

        Args:
            config: Configuration dict, Pydantic model, or None
            **kwargs: Override parameters

        Returns:
            CryptoParametricSlippageProvider instance
        """
        if config is None:
            return cls(**kwargs)

        # Extract parameters
        params: Dict[str, Any] = {}

        def _extract(cfg: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        # Map config keys to CryptoParametricConfig fields
        if _extract(config, "impact_coef", "impact_coef_base", "k") is not None:
            params["impact_coef"] = float(_extract(config, "impact_coef", "impact_coef_base", "k"))
        if _extract(config, "spread_bps", "default_spread_bps") is not None:
            params["spread_bps"] = float(_extract(config, "spread_bps", "default_spread_bps"))
        if _extract(config, "min_slippage_bps", "min_bps") is not None:
            params["min_slippage_bps"] = float(_extract(config, "min_slippage_bps", "min_bps"))
        if _extract(config, "max_slippage_bps", "max_bps") is not None:
            params["max_slippage_bps"] = float(_extract(config, "max_slippage_bps", "max_bps"))

        # Allow kwargs to override
        params.update(kwargs)

        return cls(**params)

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
    ) -> "CryptoParametricSlippageProvider":
        """
        Create provider from named profile.

        Available profiles:
        - "conservative": Higher impact estimates (safer)
        - "aggressive": Lower impact estimates (tighter)
        - "default": Standard parameters

        Args:
            profile_name: Profile name

        Returns:
            CryptoParametricSlippageProvider instance
        """
        profiles: Dict[str, Dict[str, Any]] = {
            "conservative": {
                "impact_coef": 0.12,
                "spread_bps": 6.0,
                "min_slippage_bps": 2.0,
            },
            "aggressive": {
                "impact_coef": 0.08,
                "spread_bps": 4.0,
                "min_slippage_bps": 0.5,
            },
            "default": {},  # Use defaults
            "altcoin": {
                "impact_coef": 0.15,
                "spread_bps": 10.0,
                "min_slippage_bps": 3.0,
            },
            "stablecoin": {
                "impact_coef": 0.05,
                "spread_bps": 1.0,
                "min_slippage_bps": 0.1,
                "max_slippage_bps": 50.0,
            },
        }

        profile = profiles.get(profile_name.lower(), profiles["default"])
        return cls(**profile)


# =============================================================================
# Forex Parametric TCA Model
# =============================================================================

@dataclass
class ForexParametricConfig:
    """
    Configuration for ForexParametricSlippageProvider.

    Forex-specific parametric TCA model with 8 factors:
    1. √Participation impact (Almgren-Chriss base)
    2. Session liquidity (Sydney/Tokyo/London/NY/overlaps)
    3. Spread regime (tight/normal/wide based on volatility)
    4. Interest rate differential (carry trade effect)
    5. Volatility regime (ATR-based)
    6. News event proximity (economic calendar)
    7. DXY correlation (for non-USD pairs)
    8. Pair type multiplier (major/minor/cross/exotic)

    IMPORTANT: All spreads and slippage values are in PIPS, not basis points.
    1 pip = 0.0001 for most pairs, 0.01 for JPY pairs.
    For rough conversion: 1 pip ≈ 1 bps for EUR/USD at ~1.10

    References:
        - Lyons (2001): "The Microstructure Approach to Exchange Rates"
        - Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
        - King, Osler, Rime (2012): "Foreign Exchange Market Structure"
        - BIS (2022): Triennial Central Bank Survey of FX Markets
        - Chaboud et al. (2014): "Rise of the Machines" (Algorithmic FX Trading)

    Attributes:
        impact_coef_base: Base market impact coefficient (lower than crypto due to FX liquidity)
        impact_coef_range: (min, max) range for adaptive impact
        default_spreads_pips: Default spreads by pair type in pips
        spread_profiles: Spread profiles for different client types
        session_liquidity: Liquidity multipliers by session
        carry_sensitivity: Impact of interest rate differential
        dxy_correlation_decay: Decay factor for non-USD pairs
        news_event_multipliers: Spread multipliers for economic events
        pair_type_multipliers: Slippage multipliers by pair classification
        vol_regime_multipliers: Volatility regime adjustments
        min_slippage_pips: Minimum slippage floor in pips
        max_slippage_pips: Maximum slippage cap in pips
    """
    # Base impact parameters (Almgren-Chriss)
    # Lower than crypto (0.10) and equity (0.05) due to higher FX liquidity
    impact_coef_base: float = 0.03
    impact_coef_range: Tuple[float, float] = (0.02, 0.05)

    # Default spreads by pair type (pips) - RETAIL profile
    default_spreads_pips: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.2,    # EUR/USD, GBP/USD, USD/JPY, etc.
        "minor": 2.0,    # EUR/GBP, EUR/CHF, etc.
        "cross": 3.0,    # EUR/JPY, GBP/JPY, etc.
        "exotic": 25.0,  # USD/TRY, USD/ZAR, etc.
    })

    # Spread profiles for different client types
    spread_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "institutional": {"major": 0.3, "minor": 0.8, "cross": 1.5, "exotic": 8.0},
        "retail": {"major": 1.2, "minor": 2.0, "cross": 3.0, "exotic": 25.0},
        "conservative": {"major": 1.8, "minor": 3.0, "cross": 4.5, "exotic": 40.0},
    })

    # Session liquidity factors (1.0 = normal liquidity)
    # Higher value = more liquidity = lower slippage
    session_liquidity: Dict[str, float] = field(default_factory=lambda: {
        "sydney": 0.65,           # Lowest for majors
        "tokyo": 0.75,            # JPY pairs more active
        "london": 1.10,           # Highest single-session liquidity
        "new_york": 1.05,         # Second highest
        "london_ny_overlap": 1.35,  # Peak liquidity (12:00-16:00 UTC)
        "tokyo_london_overlap": 0.90,  # Moderate
        "off_hours": 0.50,        # Very low
        "weekend": 0.0,           # Market closed
    })

    # Interest rate differential sensitivity
    # Positive carry = more market makers = tighter spreads
    # 3% slippage adjustment per 1% rate differential
    carry_sensitivity: float = 0.03

    # DXY correlation decay for non-USD pairs
    # Low correlation with DXY = less USD liquidity spillover = more slippage
    dxy_correlation_decay: float = 0.25

    # News event impact multipliers
    # Reference: ForexFactory impact ratings, OANDA volatility studies
    news_event_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "nfp": 3.0,           # Non-Farm Payrolls (highest impact)
        "fomc": 2.5,          # Fed decisions
        "ecb": 2.0,           # ECB decisions
        "boe": 1.8,           # Bank of England
        "boj": 1.8,           # Bank of Japan
        "rba": 1.5,           # Reserve Bank of Australia
        "cpi": 1.8,           # Inflation data
        "gdp": 1.5,           # GDP releases
        "pmi": 1.3,           # PMI data
        "retail_sales": 1.2,  # Retail sales
        "employment": 1.5,    # Employment data (non-NFP)
        "trade_balance": 1.2, # Trade data
        "other": 1.1,         # Minor events
    })

    # Pair type slippage multipliers
    pair_type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "major": 1.0,   # Baseline
        "minor": 1.4,   # 40% more slippage
        "cross": 1.8,   # 80% more slippage
        "exotic": 3.5,  # 250% more slippage
    })

    # Volatility regime multipliers
    vol_regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.80,      # Calm markets (VIX < 12 equivalent)
        "normal": 1.00,   # Normal conditions (VIX 12-20)
        "high": 1.50,     # Elevated volatility (VIX 20-30)
        "extreme": 2.50,  # Crisis/flash crash (VIX > 30)
    })

    # Volatility regime detection parameters
    vol_lookback_periods: int = 20
    vol_regime_thresholds: Tuple[float, float] = (25.0, 75.0)  # Percentiles

    # Bounds (in pips)
    min_slippage_pips: float = 0.05   # Minimum realistic slippage
    max_slippage_pips: float = 150.0  # For exotic pairs during extreme events

    # Adaptive coefficient learning rate
    adaptive_learning_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.impact_coef_base <= 0:
            raise ValueError("impact_coef_base must be positive")
        if self.impact_coef_range[0] >= self.impact_coef_range[1]:
            raise ValueError("impact_coef_range must have min < max")
        if self.vol_lookback_periods < 2:
            raise ValueError("vol_lookback_periods must be >= 2")
        if self.min_slippage_pips < 0:
            raise ValueError("min_slippage_pips must be non-negative")
        if self.max_slippage_pips <= self.min_slippage_pips:
            raise ValueError("max_slippage_pips must be greater than min_slippage_pips")


class ForexParametricSlippageProvider:
    """
    L2+: Smart parametric TCA model for forex OTC markets.

    Extends the basic √participation model (Almgren-Chriss) with 8
    forex-specific factors that capture OTC market microstructure.

    IMPORTANT: Forex is an OTC (Over-The-Counter) market with dealer quotes,
    NOT an exchange with a central limit order book. This means:
    - No queue position concept
    - Spread is quoted by dealers, not derived from order book
    - Last-look rejection is simulated separately
    - Liquidity varies significantly by session

    Total Slippage Formula (in pips):
        slippage_pips = half_spread_pips
            × (1 + k × √participation)       # Almgren-Chriss impact
            × (1 / session_liquidity_factor) # Session adjustment (inverted)
            × volatility_regime_mult         # Vol regime
            × (1 + carry_adjustment)         # Interest rate differential
            × dxy_correlation_factor         # USD liquidity spillover
            × news_event_factor              # Economic calendar
            × pair_type_multiplier           # Major/Minor/Cross/Exotic

    Key Differences from Crypto/Equity:
        - All values in PIPS, not basis points
        - Session-based liquidity (overlaps are best, Sydney is worst)
        - Carry trade considerations (interest rate differential)
        - No central LOB (OTC dealer market)
        - Last-look rejection handled separately in forex_dealer.py

    Smart Features:
        - **Session Detection**: Auto-detects current forex session from timestamp
        - **Pair Classification**: Classifies pairs as major/minor/cross/exotic
        - **Volatility Regime**: Detects low/normal/high/extreme volatility
        - **Adaptive Impact**: Coefficient k adjusts based on fill quality
        - **News Awareness**: Adjusts for upcoming economic events

    References:
        - Lyons (2001): "The Microstructure Approach to Exchange Rates"
        - Evans & Lyons (2002): "Order Flow and Exchange Rate Dynamics"
        - Berger et al. (2008): "The Development of the Global FX Market"
        - King, Osler, Rime (2012): "Foreign Exchange Market Structure"
        - BIS (2022): Triennial Central Bank Survey of FX Markets
        - Chaboud et al. (2014): "Rise of the Machines"
        - Oomen (2017): "Last Look" in FX Markets

    Attributes:
        config: ForexParametricConfig with all tunable parameters
        spread_profile: Active spread profile ("retail", "institutional", etc.)
        _adaptive_k: Current adaptive impact coefficient
        _fill_quality_history: Recent fill quality for adaptive adjustment
        _volatility_history: Volatility history for regime detection
    """

    # Currency pair classifications (standard forex notation)
    # Using underscore format for consistency (EUR_USD, not EURUSD)
    MAJORS: frozenset = frozenset({
        "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
    })

    MINORS: frozenset = frozenset({
        "EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD",
        "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD",
        "AUD_NZD", "CHF_JPY",
    })

    EXOTICS: frozenset = frozenset({
        "USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN",
        "USD_HUF", "USD_CZK", "USD_SGD", "USD_HKD",
        "USD_NOK", "USD_SEK", "USD_DKK", "EUR_TRY",
        "EUR_ZAR", "EUR_NOK", "EUR_SEK", "EUR_PLN",
    })

    # JPY pairs (different pip size: 0.01 instead of 0.0001)
    JPY_PAIRS: frozenset = frozenset({
        "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY",
        "NZD_JPY", "CAD_JPY", "CHF_JPY",
    })

    def __init__(
        self,
        config: Optional[ForexParametricConfig] = None,
        spread_profile: str = "retail",
        *,
        # Convenience overrides for common parameters
        impact_coef: Optional[float] = None,
        min_slippage_pips: Optional[float] = None,
        max_slippage_pips: Optional[float] = None,
    ) -> None:
        """
        Initialize forex parametric slippage provider.

        Args:
            config: Full configuration object (uses defaults if None)
            spread_profile: Spread profile name ("retail", "institutional", "conservative")
            impact_coef: Override base impact coefficient
            min_slippage_pips: Override minimum slippage
            max_slippage_pips: Override maximum slippage
        """
        self.config = config or ForexParametricConfig()
        self.spread_profile = spread_profile

        # Apply convenience overrides by creating new config
        overrides = {}
        if impact_coef is not None:
            overrides["impact_coef_base"] = float(impact_coef)
        if min_slippage_pips is not None:
            overrides["min_slippage_pips"] = float(min_slippage_pips)
        if max_slippage_pips is not None:
            overrides["max_slippage_pips"] = float(max_slippage_pips)

        if overrides:
            # Create new config with overrides
            config_dict = {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith("_")
            }
            config_dict.update(overrides)
            self.config = ForexParametricConfig(**config_dict)

        # Adaptive impact coefficient (starts at base)
        self._adaptive_k: float = self.config.impact_coef_base

        # Fill quality history for adaptive adjustment
        self._fill_quality_history: List[Tuple[float, float]] = []
        self._max_history_size: int = 100

        # Volatility history for regime detection
        self._volatility_history: List[float] = []

    def compute_slippage_pips(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        *,
        session: Optional[ForexSession] = None,
        pair_type: Optional[PairType] = None,
        interest_rate_diff: Optional[float] = None,
        dxy_correlation: Optional[float] = None,
        upcoming_news: Optional[str] = None,
        recent_returns: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Compute expected slippage in pips using the full parametric model.

        The model combines multiple factors to produce a realistic
        slippage estimate that accounts for forex OTC market dynamics.

        Formula:
            slippage = half_spread × (1 + k × √participation) × Π(factors)

        where Π(factors) includes session liquidity, volatility regime,
        carry trade, DXY correlation, news events, and pair type.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / estimated session volume
            session: Current forex session (auto-detected if None)
            pair_type: Currency pair classification (auto-detected if None)
            interest_rate_diff: Base rate - Quote rate (annual %)
            dxy_correlation: Correlation with Dollar Index (-1 to 1)
            upcoming_news: Type of upcoming economic event
            recent_returns: Recent price returns for regime detection

        Returns:
            Expected slippage in pips (always positive)

        Example:
            >>> provider = ForexParametricSlippageProvider()
            >>> slippage = provider.compute_slippage_pips(
            ...     order=Order("EUR_USD", "BUY", 100000, "MARKET"),
            ...     market=MarketState(timestamp=1700000000000, bid=1.0850, ask=1.0852),
            ...     participation_ratio=0.001,
            ...     session=ForexSession.LONDON_NY_OVERLAP,
            ...     interest_rate_diff=-0.50,  # EUR rates lower than USD
            ... )
        """
        from datetime import datetime, timezone

        # =====================================================================
        # 1. Determine pair type and get base spread
        # =====================================================================
        pair_type = pair_type or self._classify_pair(order.symbol)
        spreads = self.config.spread_profiles.get(
            self.spread_profile,
            self.config.default_spreads_pips
        )
        base_spread_pips = spreads.get(pair_type.value, 1.5)
        half_spread = base_spread_pips / 2.0

        # Override with market spread if available
        # Market spread is in bps, convert to pips (rough approximation)
        if market.spread_bps is not None and math.isfinite(market.spread_bps):
            # 1 pip ≈ 1 bps for most pairs at typical rates
            half_spread = market.spread_bps / 10.0 / 2.0

        # =====================================================================
        # 2. √Participation impact (Almgren-Chriss 2001)
        # =====================================================================
        participation = max(_MIN_PARTICIPATION, abs(participation_ratio))
        impact = self._adaptive_k * math.sqrt(participation) * 100.0  # Scale for pips

        # =====================================================================
        # 3. Session liquidity adjustment
        # =====================================================================
        session = session or self._detect_session(market.timestamp)
        session_factor = self.config.session_liquidity.get(session.value, 1.0)

        # Handle weekend (market closed)
        if session == ForexSession.WEEKEND or session_factor <= 0:
            # Market closed - return maximum slippage as signal
            return self.config.max_slippage_pips

        # Invert: low liquidity (factor < 1) = higher slippage
        session_adjustment = 1.0 / max(0.3, session_factor)

        # =====================================================================
        # 4. Volatility regime multiplier
        # =====================================================================
        vol_regime = self._detect_volatility_regime(recent_returns)
        vol_mult = self.config.vol_regime_multipliers.get(vol_regime, 1.0)

        # =====================================================================
        # 5. Carry adjustment (interest rate differential)
        # =====================================================================
        carry_factor = 1.0
        if interest_rate_diff is not None and math.isfinite(interest_rate_diff):
            # Positive carry (long high-yield) = more liquidity = lower slippage
            # Negative carry = less interest from dealers = wider spreads
            carry_factor = 1.0 - interest_rate_diff * self.config.carry_sensitivity
            carry_factor = max(0.7, min(1.4, carry_factor))

        # =====================================================================
        # 6. DXY correlation factor (for non-USD pairs)
        # =====================================================================
        dxy_factor = 1.0
        if dxy_correlation is not None and not self._has_usd(order.symbol):
            # Lower correlation with USD = less liquidity spillover = more slippage
            corr_clamped = max(0.0, min(1.0, abs(dxy_correlation)))
            dxy_factor = 1.0 + (1.0 - corr_clamped) * self.config.dxy_correlation_decay

        # =====================================================================
        # 7. News event impact
        # =====================================================================
        news_factor = 1.0
        if upcoming_news is not None:
            news_factor = self.config.news_event_multipliers.get(
                upcoming_news.lower(), 1.0
            )

        # =====================================================================
        # 8. Pair type multiplier
        # =====================================================================
        pair_mult = self.config.pair_type_multipliers.get(pair_type.value, 1.0)

        # =====================================================================
        # 9. Combine all factors
        # =====================================================================
        total_slippage = (
            half_spread
            * (1.0 + impact / half_spread if half_spread > 0 else 1.0 + impact)
            * session_adjustment
            * vol_mult
            * carry_factor
            * dxy_factor
            * news_factor
            * pair_mult
        )

        # Apply bounds
        total_slippage = max(self.config.min_slippage_pips, total_slippage)
        total_slippage = min(self.config.max_slippage_pips, total_slippage)

        return float(total_slippage)

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute slippage in basis points for compatibility with existing interfaces.

        This is a convenience method that converts pip-based slippage to bps.
        For accurate forex slippage, use compute_slippage_pips() directly.

        Conversion: 1 pip ≈ 1 bps (rough approximation for EUR/USD at ~1.10)
        More precise: bps = pips × 10 × pip_size / mid_price × 10000

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / volume
            **kwargs: Additional parameters for compute_slippage_pips

        Returns:
            Expected slippage in basis points
        """
        slippage_pips = self.compute_slippage_pips(
            order, market, participation_ratio, **kwargs
        )

        # Convert pips to bps (rough approximation)
        # For most pairs: 1 pip ≈ 10 bps at mid-price around 1.0
        # For JPY pairs: 1 pip ≈ 10 bps as well
        # This is approximate; exact conversion depends on price
        mid_price = market.get_mid_price() or 1.0
        pip_size = 0.01 if self._is_jpy_pair(order.symbol) else 0.0001

        # bps = (slippage_pips * pip_size / mid_price) * 10000
        slippage_bps = (slippage_pips * pip_size / mid_price) * 10000.0

        return float(slippage_bps)

    def _classify_pair(self, symbol: str) -> PairType:
        """
        Classify currency pair by type.

        Args:
            symbol: Currency pair symbol (e.g., "EUR_USD", "EURUSD", "EUR/USD")

        Returns:
            PairType classification
        """
        # Normalize symbol format
        norm = symbol.replace("/", "_").replace("-", "_").upper()
        # Handle formats like "EURUSD" -> "EUR_USD"
        if len(norm) == 6 and "_" not in norm:
            norm = f"{norm[:3]}_{norm[3:]}"

        if norm in self.MAJORS:
            return PairType.MAJOR
        if norm in self.MINORS:
            return PairType.MINOR
        if norm in self.EXOTICS:
            return PairType.EXOTIC
        # Default to CROSS for unclassified pairs (like EUR/JPY, GBP/JPY)
        return PairType.CROSS

    def _has_usd(self, symbol: str) -> bool:
        """Check if pair contains USD."""
        norm = symbol.upper().replace("/", "_").replace("-", "_")
        return "USD" in norm

    def _is_jpy_pair(self, symbol: str) -> bool:
        """Check if pair is a JPY pair (different pip size)."""
        norm = symbol.replace("/", "_").replace("-", "_").upper()
        if len(norm) == 6 and "_" not in norm:
            norm = f"{norm[:3]}_{norm[3:]}"
        return norm in self.JPY_PAIRS or "JPY" in norm

    def _detect_session(self, timestamp: int) -> ForexSession:
        """
        Detect current forex session from timestamp.

        Sessions are based on UTC times:
        - Sydney: 21:00-06:00 UTC
        - Tokyo: 00:00-09:00 UTC
        - London: 07:00-16:00 UTC
        - New York: 12:00-21:00 UTC
        - London/NY overlap: 12:00-16:00 UTC (priority)
        - Tokyo/London overlap: 07:00-09:00 UTC (priority)

        Args:
            timestamp: Unix timestamp in milliseconds

        Returns:
            ForexSession classification
        """
        from datetime import datetime, timezone

        try:
            # Handle both milliseconds and seconds timestamps
            ts_sec = timestamp / 1000.0 if timestamp > 1e12 else float(timestamp)
            dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return ForexSession.OFF_HOURS

        hour = dt.hour
        weekday = dt.weekday()

        # Weekend check (Fri 21:00 UTC - Sun 21:00 UTC)
        # Friday after 21:00 UTC or Saturday or Sunday before 21:00 UTC
        if weekday == 5:  # Saturday
            return ForexSession.WEEKEND
        if weekday == 6:  # Sunday
            if hour < 21:  # Before Sydney open
                return ForexSession.WEEKEND
        if weekday == 4 and hour >= 21:  # Friday after NY close
            return ForexSession.WEEKEND

        # Check overlaps first (higher priority)
        if 12 <= hour < 16:
            return ForexSession.LONDON_NY_OVERLAP
        if 7 <= hour < 9:
            return ForexSession.TOKYO_LONDON_OVERLAP

        # Individual sessions
        if 21 <= hour or hour < 6:
            return ForexSession.SYDNEY
        if 0 <= hour < 9:
            return ForexSession.TOKYO
        if 7 <= hour < 16:
            return ForexSession.LONDON
        if 12 <= hour < 21:
            return ForexSession.NEW_YORK

        return ForexSession.OFF_HOURS

    def _detect_volatility_regime(
        self,
        returns: Optional[Sequence[float]],
    ) -> str:
        """
        Detect volatility regime from recent returns.

        Uses rolling volatility percentile to classify into
        low/normal/high/extreme regime.

        Args:
            returns: Sequence of recent price returns

        Returns:
            Volatility regime string ("low", "normal", "high", "extreme")
        """
        if returns is None or len(returns) < 2:
            return "normal"

        returns_list = list(returns)[-self.config.vol_lookback_periods:]
        if len(returns_list) < 2:
            return "normal"

        # Compute realized volatility
        mean_ret = sum(returns_list) / len(returns_list)
        variance = sum((r - mean_ret) ** 2 for r in returns_list) / (len(returns_list) - 1)
        current_vol = math.sqrt(variance) if variance > 0 else 0.0

        if current_vol == 0:
            return "normal"

        # Update history
        self._volatility_history.append(current_vol)
        if len(self._volatility_history) > 500:
            self._volatility_history = self._volatility_history[-250:]

        # Need sufficient history for regime classification
        if len(self._volatility_history) < 20:
            return "normal"

        # Compute percentile rank
        sorted_vols = sorted(self._volatility_history)
        rank = sum(1 for v in sorted_vols if v < current_vol)
        percentile = (rank / len(sorted_vols)) * 100.0

        low_threshold, high_threshold = self.config.vol_regime_thresholds

        # Extreme is above 95th percentile (hardcoded threshold)
        if percentile > 95.0:
            return "extreme"
        elif percentile > high_threshold:
            return "high"
        elif percentile < low_threshold:
            return "low"
        else:
            return "normal"

    def update_fill_quality(
        self,
        predicted_slippage_pips: float,
        actual_slippage_pips: float,
    ) -> None:
        """
        Update adaptive impact coefficient based on fill quality.

        Tracks the ratio of actual to predicted slippage and adjusts
        the impact coefficient k to minimize prediction error.

        Args:
            predicted_slippage_pips: Predicted slippage at order time (pips)
            actual_slippage_pips: Realized slippage after fill (pips)
        """
        if predicted_slippage_pips <= 0 or actual_slippage_pips < 0:
            return

        # Add to history
        self._fill_quality_history.append((predicted_slippage_pips, actual_slippage_pips))

        # Trim history
        if len(self._fill_quality_history) > self._max_history_size:
            self._fill_quality_history = self._fill_quality_history[-self._max_history_size:]

        # Compute adjustment based on recent prediction errors
        if len(self._fill_quality_history) >= 10:
            recent = self._fill_quality_history[-20:]
            ratios = [actual / pred if pred > 0 else 1.0 for pred, actual in recent]
            avg_ratio = sum(ratios) / len(ratios)

            # Learning rate decays with history size
            learning_rate = self.config.adaptive_learning_rate / math.sqrt(
                len(self._fill_quality_history)
            )

            # Update adaptive k
            adjustment = 1.0 + learning_rate * (avg_ratio - 1.0)
            new_k = self._adaptive_k * adjustment

            # Clamp to configured range
            min_k, max_k = self.config.impact_coef_range
            self._adaptive_k = max(min_k, min(max_k, new_k))

    def reset_adaptive_state(self) -> None:
        """Reset adaptive state to initial values."""
        self._adaptive_k = self.config.impact_coef_base
        self._fill_quality_history.clear()
        self._volatility_history.clear()

    def estimate_impact_cost(
        self,
        notional: float,
        adv: float,
        symbol: str = "EUR_USD",
        side: str = "BUY",
        session: Optional[ForexSession] = None,
        hour_utc: Optional[int] = None,
        interest_rate_diff: Optional[float] = None,
        upcoming_news: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pre-trade cost estimation with full model.

        Useful for optimal execution planning and trade scheduling.

        Args:
            notional: Planned trade notional value (in quote currency)
            adv: Average daily volume
            symbol: Currency pair symbol
            side: Trade side ("BUY" or "SELL")
            session: Forex session (auto-detected from hour_utc if None)
            hour_utc: Planned execution hour (UTC)
            interest_rate_diff: Interest rate differential
            upcoming_news: Expected news event type

        Returns:
            Dict with detailed cost breakdown and recommendations

        Example:
            >>> provider = ForexParametricSlippageProvider()
            >>> estimate = provider.estimate_impact_cost(
            ...     notional=10_000_000,  # $10M
            ...     adv=500_000_000_000,  # $500B EUR/USD daily volume
            ...     symbol="EUR_USD",
            ...     hour_utc=14,  # London/NY overlap
            ... )
        """
        if adv <= 0:
            return {
                "participation": 0.0,
                "impact_pips": 0.0,
                "impact_cost": 0.0,
                "pair_type": "unknown",
                "session": "unknown",
                "recommendation": "Unable to estimate: ADV is zero",
            }

        participation = notional / adv
        pair_type = self._classify_pair(symbol)

        # Determine session from hour if not provided
        if session is None and hour_utc is not None:
            # Create fake timestamp for session detection
            import time
            fake_ts = int(time.time() * 1000)
            # Adjust to the specified hour (rough approximation)
            session = self._detect_session(fake_ts)
        elif session is None:
            session = ForexSession.LONDON  # Default to highest liquidity

        # Create dummy order and market for estimation
        dummy_order = Order(symbol, side, 1.0, "MARKET")
        dummy_market = MarketState(
            timestamp=int(hour_utc * 3600 * 1000) if hour_utc else 0,
            adv=adv,
        )

        # Compute slippage
        impact_pips = self.compute_slippage_pips(
            dummy_order,
            dummy_market,
            participation,
            session=session,
            pair_type=pair_type,
            interest_rate_diff=interest_rate_diff,
            upcoming_news=upcoming_news,
        )

        # Convert to cost
        pip_size = 0.01 if self._is_jpy_pair(symbol) else 0.0001
        impact_cost = notional * impact_pips * pip_size

        # Generate recommendation
        recommendation = self._generate_execution_recommendation(
            participation, session, hour_utc, pair_type, upcoming_news
        )

        return {
            "participation": participation,
            "participation_pct": participation * 100,
            "impact_pips": impact_pips,
            "impact_cost": impact_cost,
            "pair_type": pair_type.value,
            "session": session.value,
            "liquidity_factor": self.config.session_liquidity.get(session.value, 1.0),
            "recommendation": recommendation,
        }

    def _generate_execution_recommendation(
        self,
        participation: float,
        session: ForexSession,
        hour_utc: Optional[int],
        pair_type: PairType,
        upcoming_news: Optional[str],
    ) -> str:
        """Generate execution recommendation based on analysis."""
        recommendations = []

        # Participation-based recommendations
        if participation > 0.01:  # > 1% of ADV is large for FX
            recommendations.append(
                f"Large order ({participation*100:.3f}% ADV). "
                "Consider TWAP over multiple sessions."
            )
        elif participation > 0.001:
            recommendations.append(
                "Moderate size order. Consider splitting across session overlap."
            )

        # Session-based recommendations
        if session == ForexSession.WEEKEND:
            recommendations.append(
                "⚠️ Market closed (weekend). Delay execution to Sunday 21:00 UTC."
            )
        elif session == ForexSession.SYDNEY:
            recommendations.append(
                "Low liquidity session (Sydney). Consider delaying to London or overlap."
            )
        elif session == ForexSession.LONDON_NY_OVERLAP:
            recommendations.append("Optimal execution window (London/NY overlap).")

        # Pair type recommendations
        if pair_type == PairType.EXOTIC:
            recommendations.append(
                "Exotic pair - expect wider spreads. Consider limit orders."
            )

        # News recommendations
        if upcoming_news:
            news_mult = self.config.news_event_multipliers.get(upcoming_news.lower(), 1.0)
            if news_mult > 2.0:
                recommendations.append(
                    f"⚠️ High-impact event ({upcoming_news}) expected. "
                    "Spreads may be {news_mult:.1f}x wider."
                )

        if not recommendations:
            return "Standard execution conditions."

        return " ".join(recommendations)

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any, None],
        **kwargs: Any,
    ) -> "ForexParametricSlippageProvider":
        """
        Create provider from configuration dict or object.

        Args:
            config: Configuration dict, Pydantic model, or None
            **kwargs: Override parameters

        Returns:
            ForexParametricSlippageProvider instance
        """
        if config is None:
            return cls(**kwargs)

        # Extract parameters
        params: Dict[str, Any] = {}

        def _extract(cfg: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        # Map config keys to init parameters
        if _extract(config, "impact_coef", "impact_coef_base", "k") is not None:
            params["impact_coef"] = float(_extract(config, "impact_coef", "impact_coef_base", "k"))
        if _extract(config, "spread_profile", "profile") is not None:
            params["spread_profile"] = str(_extract(config, "spread_profile", "profile"))
        if _extract(config, "min_slippage_pips", "min_pips") is not None:
            params["min_slippage_pips"] = float(_extract(config, "min_slippage_pips", "min_pips"))
        if _extract(config, "max_slippage_pips", "max_pips") is not None:
            params["max_slippage_pips"] = float(_extract(config, "max_slippage_pips", "max_pips"))

        # Allow kwargs to override
        params.update(kwargs)

        return cls(**params)

    @classmethod
    def from_profile(
        cls,
        profile_name: str,
    ) -> "ForexParametricSlippageProvider":
        """
        Create provider from named profile.

        Available profiles:
        - "retail": Standard retail spreads (default)
        - "institutional": Tight institutional spreads
        - "conservative": Wider spreads for safety margin
        - "major_only": Optimized for major pairs
        - "exotic": Wider spreads for exotic pairs

        Args:
            profile_name: Profile name

        Returns:
            ForexParametricSlippageProvider instance
        """
        profiles: Dict[str, Dict[str, Any]] = {
            "retail": {
                "spread_profile": "retail",
            },
            "institutional": {
                "impact_coef": 0.02,
                "spread_profile": "institutional",
                "min_slippage_pips": 0.02,
            },
            "conservative": {
                "impact_coef": 0.04,
                "spread_profile": "conservative",
                "min_slippage_pips": 0.1,
                "max_slippage_pips": 200.0,
            },
            "major_only": {
                "impact_coef": 0.025,
                "spread_profile": "institutional",
                "min_slippage_pips": 0.01,
                "max_slippage_pips": 50.0,
            },
            "exotic": {
                "impact_coef": 0.05,
                "spread_profile": "conservative",
                "min_slippage_pips": 1.0,
                "max_slippage_pips": 200.0,
            },
        }

        profile = profiles.get(profile_name.lower(), profiles["retail"])
        return cls(**profile)


class OHLCVFillProvider:
    """
    L2: Fill provider based on OHLCV bar data.

    Determines order fills based on whether the bar's price range
    would trigger the order:
    - MARKET: Always fills at reference price + slippage
    - LIMIT BUY: Fills if bar_low <= limit_price
    - LIMIT SELL: Fills if bar_high >= limit_price

    Uses pluggable SlippageProvider and FeeProvider for cost calculation.

    Attributes:
        slippage: SlippageProvider for slippage calculation
        fees: FeeProvider for fee calculation
        fill_at_limit: If True, limit orders fill at limit price; else at touch
        partial_fills: If True, allow partial fills based on liquidity
    """

    def __init__(
        self,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        fill_at_limit: bool = True,
        partial_fills: bool = False,
    ) -> None:
        """
        Initialize OHLCV fill provider.

        Args:
            slippage_provider: Provider for slippage calculation
            fee_provider: Provider for fee calculation
            fill_at_limit: Fill limit orders at limit price (True) or touch price
            partial_fills: Allow partial fills based on available liquidity
        """
        self.slippage = slippage_provider or StatisticalSlippageProvider()
        self.fees = fee_provider or ZeroFeeProvider()
        self.fill_at_limit = fill_at_limit
        self.partial_fills = partial_fills

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt to fill order based on bar data.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data (OHLCV)

        Returns:
            Fill if order executed, None if not filled
        """
        if order.qty <= 0:
            return None

        is_buy = order.is_buy
        order_type = str(order.order_type).upper()

        # Determine fill price and whether order fills
        filled = False
        fill_price: Optional[float] = None
        liquidity_role = LiquidityRole.TAKER

        if order_type == "MARKET":
            # Market orders always fill
            filled = True
            liquidity_role = LiquidityRole.TAKER
            # Get reference price from market
            ref_price = market.get_reference_price(order.side)
            if ref_price is None:
                ref_price = bar.open  # Fallback to bar open
            fill_price = ref_price

        elif order_type == "LIMIT":
            limit_price = order.limit_price
            if limit_price is None:
                logger.warning("LIMIT order missing limit_price")
                return None

            # First check: Immediate execution (crossing spread) → TAKER fill
            # A buy limit above the ask crosses the spread and fills immediately
            # A sell limit below the bid crosses the spread and fills immediately
            ref_price = market.get_reference_price(order.side)
            if ref_price is not None:
                if is_buy and limit_price >= ref_price:
                    # Buy limit at or above ask → immediate taker fill
                    filled = True
                    liquidity_role = LiquidityRole.TAKER
                    fill_price = ref_price
                elif not is_buy and limit_price <= ref_price:
                    # Sell limit at or below bid → immediate taker fill
                    filled = True
                    liquidity_role = LiquidityRole.TAKER
                    fill_price = ref_price

            # Second check: Passive fill based on bar range → MAKER fill
            # Only if not already filled as taker
            if not filled:
                if is_buy:
                    # Buy limit fills if bar_low <= limit_price
                    if bar.low <= limit_price:
                        filled = True
                        liquidity_role = LiquidityRole.MAKER
                        fill_price = limit_price if self.fill_at_limit else bar.low
                else:
                    # Sell limit fills if bar_high >= limit_price
                    if bar.high >= limit_price:
                        filled = True
                        liquidity_role = LiquidityRole.MAKER
                        fill_price = limit_price if self.fill_at_limit else bar.high

        if not filled or fill_price is None:
            return None

        # Calculate participation ratio for slippage
        participation = self._compute_participation(order, market, bar)

        # Calculate slippage
        slippage_bps = self.slippage.compute_slippage_bps(order, market, participation)

        # Apply slippage to fill price
        if liquidity_role == LiquidityRole.TAKER:
            slippage_mult = 1.0 + (slippage_bps / 10000.0) * (1 if is_buy else -1)
            adjusted_price = fill_price * slippage_mult
        else:
            # Maker orders have reduced/no slippage
            adjusted_price = fill_price
            slippage_bps = 0.0  # Maker fills at limit price

        # Clip to bar range (sanity check)
        adjusted_price = max(bar.low, min(bar.high, adjusted_price))

        # Calculate fill quantity (with partial fill support)
        fill_qty = order.qty
        if self.partial_fills and market.bid_size is not None and market.ask_size is not None:
            available = market.ask_size if is_buy else market.bid_size
            fill_qty = min(fill_qty, available)

        # Calculate notional and fee
        notional = adjusted_price * fill_qty
        fee = self.fees.compute_fee(
            notional=notional,
            side=order.side,
            liquidity=liquidity_role.value,
            qty=fill_qty,
        )

        return Fill(
            price=adjusted_price,
            qty=fill_qty,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidity=liquidity_role.value,
            timestamp=market.timestamp,
            notional=notional,
            metadata={
                "original_price": fill_price,
                "participation": participation,
                "bar_range": (bar.low, bar.high),
            },
        )

    def _compute_participation(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> float:
        """Compute participation ratio for slippage calculation."""
        # Try ADV first
        if market.adv is not None and market.adv > 0:
            ref_price = market.get_mid_price() or bar.typical_price
            order_notional = order.get_notional(ref_price)
            return order_notional / market.adv

        # Fallback to bar volume
        if bar.volume is not None and bar.volume > 0:
            return order.qty / bar.volume

        # Default small participation
        return 0.001


class ZeroFeeProvider:
    """
    Fee provider returning zero fees.

    Useful as default/fallback or for testing.
    """

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """Return zero fee."""
        return 0.0


class CryptoFeeProvider:
    """
    L2: Crypto exchange fee provider (Binance-style).

    Implements tiered maker/taker fee structure common to
    cryptocurrency exchanges.

    Attributes:
        maker_bps: Maker fee in basis points (default: 2.0 = 0.02%)
        taker_bps: Taker fee in basis points (default: 4.0 = 0.04%)
        discount_rate: BNB discount rate (default: 0.75 = 25% off)
        use_discount: Whether to apply discount
    """

    def __init__(
        self,
        maker_bps: float = 2.0,
        taker_bps: float = 4.0,
        discount_rate: float = 0.75,
        use_discount: bool = False,
    ) -> None:
        """
        Initialize crypto fee provider.

        Args:
            maker_bps: Maker fee in basis points
            taker_bps: Taker fee in basis points
            discount_rate: Discount multiplier (e.g., 0.75 for 25% off)
            use_discount: Whether to apply discount (e.g., BNB payment)
        """
        self.maker_bps = float(maker_bps)
        self.taker_bps = float(taker_bps)
        self.discount_rate = float(discount_rate)
        self.use_discount = bool(use_discount)

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute trading fee based on liquidity role.

        Args:
            notional: Trade notional value
            side: Trade side (not used for crypto)
            liquidity: "maker" or "taker"
            qty: Trade quantity (not used for crypto)

        Returns:
            Fee amount in quote currency
        """
        liquidity_norm = str(liquidity).lower()

        if liquidity_norm == "maker":
            rate_bps = self.maker_bps
        else:
            rate_bps = self.taker_bps

        if self.use_discount:
            rate_bps *= self.discount_rate

        fee = abs(notional) * rate_bps / 10000.0
        return float(fee)

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any, None],
    ) -> "CryptoFeeProvider":
        """
        Create CryptoFeeProvider from configuration dict or object.

        Args:
            config: Configuration dict or object with fee parameters

        Returns:
            CryptoFeeProvider instance
        """
        if config is None:
            return cls()

        def _extract_value(cfg: Any, *keys: str, default: Any = None) -> Any:
            """Extract value from config trying multiple key names."""
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        maker_bps = _extract_value(
            config, "maker_bps", "maker_rate_bps", "maker_fee_bps", default=2.0
        )
        taker_bps = _extract_value(
            config, "taker_bps", "taker_rate_bps", "taker_fee_bps", default=4.0
        )
        discount_rate = _extract_value(
            config, "discount_rate", "bnb_discount", default=0.75
        )
        use_discount = _extract_value(
            config, "use_discount", "use_bnb_discount", "bnb_settlement", default=False
        )

        return cls(
            maker_bps=float(maker_bps),
            taker_bps=float(taker_bps),
            discount_rate=float(discount_rate),
            use_discount=bool(use_discount),
        )


class EquityFeeProvider:
    """
    L2: US Equity fee provider (Alpaca-style).

    Implements commission-free trading with regulatory fees:
    - SEC fee: ~$0.0000278 per dollar of sale proceeds
    - TAF fee: $0.000166 per share sold (max $8.30)

    These fees only apply to SELL orders.

    Attributes:
        sec_fee_rate: SEC fee per dollar (sell only)
        taf_fee_rate: TAF fee per share (sell only)
        taf_max_fee: Maximum TAF fee per trade
        include_regulatory: Whether to include regulatory fees
    """

    # Regulatory fee rates (2024 values)
    SEC_FEE_RATE = 0.0000278
    TAF_FEE_RATE = 0.000166
    TAF_MAX_FEE = 8.30

    def __init__(
        self,
        sec_fee_rate: Optional[float] = None,
        taf_fee_rate: Optional[float] = None,
        taf_max_fee: Optional[float] = None,
        include_regulatory: bool = True,
    ) -> None:
        """
        Initialize equity fee provider.

        Args:
            sec_fee_rate: SEC fee per dollar (default: current rate)
            taf_fee_rate: TAF fee per share (default: current rate)
            taf_max_fee: Maximum TAF fee (default: $8.30)
            include_regulatory: Whether to include regulatory fees
        """
        self.sec_fee_rate = sec_fee_rate if sec_fee_rate is not None else self.SEC_FEE_RATE
        self.taf_fee_rate = taf_fee_rate if taf_fee_rate is not None else self.TAF_FEE_RATE
        self.taf_max_fee = taf_max_fee if taf_max_fee is not None else self.TAF_MAX_FEE
        self.include_regulatory = include_regulatory

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute trading fee (regulatory fees on sells).

        Alpaca is commission-free, but regulatory fees apply to sales.

        Args:
            notional: Trade notional value
            side: Trade side ("BUY" or "SELL")
            liquidity: Liquidity role (not used for equity)
            qty: Number of shares

        Returns:
            Fee amount in USD
        """
        # Commission-free for buys
        # Handle both string and OrderSide enum
        side_str = side.value if isinstance(side, OrderSide) else str(side)
        if side_str.upper() != "SELL":
            return 0.0

        if not self.include_regulatory:
            return 0.0

        fee = 0.0

        # SEC fee (on sale proceeds)
        sec_fee = abs(notional) * self.sec_fee_rate
        fee += sec_fee

        # TAF fee (per share sold)
        taf_fee = min(abs(qty) * self.taf_fee_rate, self.taf_max_fee)
        fee += taf_fee

        return round(fee, 4)

    def estimate_regulatory_breakdown(
        self,
        notional: float,
        qty: float,
    ) -> Dict[str, float]:
        """
        Get detailed regulatory fee breakdown.

        Args:
            notional: Trade notional value
            qty: Number of shares

        Returns:
            Dict with fee breakdown
        """
        sec_fee = abs(notional) * self.sec_fee_rate
        taf_fee = min(abs(qty) * self.taf_fee_rate, self.taf_max_fee)

        return {
            "sec_fee": round(sec_fee, 4),
            "taf_fee": round(taf_fee, 4),
            "total": round(sec_fee + taf_fee, 4),
        }

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any, None],
    ) -> "EquityFeeProvider":
        """
        Create EquityFeeProvider from configuration dict or object.

        Args:
            config: Configuration dict or object with fee parameters

        Returns:
            EquityFeeProvider instance
        """
        if config is None:
            return cls()

        def _extract_value(cfg: Any, *keys: str, default: Any = None) -> Any:
            """Extract value from config trying multiple key names."""
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        sec_fee_rate = _extract_value(
            config, "sec_fee_rate", "sec_fee_per_dollar", "sec_fee", default=None
        )
        taf_fee_rate = _extract_value(
            config, "taf_fee_rate", "taf_fee_per_share", "taf_fee", default=None
        )
        taf_max_fee = _extract_value(
            config, "taf_max_fee", "taf_max", "max_taf", default=None
        )
        include_regulatory = _extract_value(
            config, "include_regulatory", "regulatory_fees", "include_fees", default=True
        )

        return cls(
            sec_fee_rate=float(sec_fee_rate) if sec_fee_rate is not None else None,
            taf_fee_rate=float(taf_fee_rate) if taf_fee_rate is not None else None,
            taf_max_fee=float(taf_max_fee) if taf_max_fee is not None else None,
            include_regulatory=bool(include_regulatory),
        )


class ForexFeeProvider:
    """
    Forex fee provider (spread-based, zero commission).

    In forex markets, transaction costs are embedded in the bid-ask spread.
    There are no separate maker/taker fees or regulatory fees like in equity.

    This provider always returns 0 for compute_fee() because:
    1. Spread cost is already captured in the slippage model
    2. Retail forex is typically commission-free
    3. Swap/rollover costs are handled separately (overnight positions)

    For institutional pricing with commission-based models,
    use the commission_bps parameter.

    Attributes:
        commission_bps: Optional commission in basis points (institutional)
        include_swap: Whether to estimate swap cost (for overnight holds)
    """

    def __init__(
        self,
        commission_bps: float = 0.0,
        include_swap: bool = False,
    ) -> None:
        """
        Initialize forex fee provider.

        Args:
            commission_bps: Commission per side in bps (0 for retail, ~0.5-1 for institutional)
            include_swap: Whether to include swap cost placeholder
        """
        self.commission_bps = float(commission_bps)
        self.include_swap = include_swap

    def compute_fee(
        self,
        notional: float,
        side: str,
        liquidity: str,
        qty: float,
    ) -> float:
        """
        Compute trading fee.

        For retail forex, this returns 0 (cost is in spread).
        For institutional, may include commission.

        Args:
            notional: Trade notional value
            side: Trade side ("BUY" or "SELL")
            liquidity: Liquidity role (not used for forex)
            qty: Trade quantity (not used for forex)

        Returns:
            Fee amount (0 for retail, commission for institutional)
        """
        if self.commission_bps <= 0:
            return 0.0

        # Institutional commission
        fee = abs(notional) * self.commission_bps / 10000.0
        return float(fee)

    def estimate_swap_cost(
        self,
        notional: float,
        side: str,
        holding_days: int = 1,
        swap_points_long: float = 0.0,
        swap_points_short: float = 0.0,
    ) -> float:
        """
        Estimate swap/rollover cost for overnight positions.

        Forex positions held overnight incur swap costs based on
        interest rate differential between currencies.

        Args:
            notional: Position notional value
            side: Position side ("BUY" or "SELL")
            holding_days: Number of nights position is held
            swap_points_long: Swap points for long positions (pips/day)
            swap_points_short: Swap points for short positions (pips/day)

        Returns:
            Estimated swap cost (negative = cost, positive = credit)

        Note:
            Wednesday positions incur 3x swap (weekend rollover).
        """
        if not self.include_swap:
            return 0.0

        side_str = str(side).upper()
        swap_points = swap_points_long if side_str == "BUY" else swap_points_short

        # Convert pips to cost (assuming standard pip size)
        pip_value = notional * 0.0001  # Simplified
        swap_cost = swap_points * pip_value * holding_days

        return float(swap_cost)

    @classmethod
    def from_config(
        cls,
        config: Union[Mapping[str, Any], Any, None],
    ) -> "ForexFeeProvider":
        """
        Create ForexFeeProvider from configuration dict or object.

        Args:
            config: Configuration dict or object with fee parameters

        Returns:
            ForexFeeProvider instance
        """
        if config is None:
            return cls()

        def _extract_value(cfg: Any, *keys: str, default: Any = None) -> Any:
            for key in keys:
                if isinstance(cfg, Mapping) and key in cfg:
                    return cfg[key]
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        commission_bps = _extract_value(
            config, "commission_bps", "commission", "fee_bps", default=0.0
        )
        include_swap = _extract_value(
            config, "include_swap", "swap", "include_rollover", default=False
        )

        return cls(
            commission_bps=float(commission_bps) if commission_bps else 0.0,
            include_swap=bool(include_swap),
        )


# =============================================================================
# L2 Combined Execution Provider
# =============================================================================

class L2ExecutionProvider:
    """
    L2: Combined execution provider using statistical models.

    Combines SlippageProvider, FillProvider, and FeeProvider into
    a complete execution simulation workflow.

    Supports both crypto and equity asset classes with appropriate
    default providers.

    Attributes:
        asset_class: Asset class (CRYPTO or EQUITY)
        slippage: SlippageProvider instance
        fill: FillProvider instance (typically OHLCVFillProvider)
        fees: FeeProvider instance
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.CRYPTO,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize L2 execution provider.

        Args:
            asset_class: Asset class to configure defaults for (CRYPTO, EQUITY, FUTURES, FOREX)
            slippage_provider: Custom slippage provider
            fee_provider: Custom fee provider
            **kwargs: Additional configuration
        """
        self._asset_class = asset_class

        # Default slippage provider
        if slippage_provider is not None:
            self.slippage = slippage_provider
        else:
            # Asset-class specific defaults
            if asset_class == AssetClass.EQUITY:
                # Equities typically have tighter spreads
                self.slippage = StatisticalSlippageProvider(
                    impact_coef=0.05,
                    spread_bps=2.0,
                )
            elif asset_class == AssetClass.FUTURES:
                # Futures: use FuturesSlippageProvider
                try:
                    from execution_providers_futures import FuturesSlippageProvider
                    self.slippage = FuturesSlippageProvider()
                except ImportError:
                    logger.warning(
                        "FuturesSlippageProvider not available, using crypto defaults"
                    )
                    self.slippage = StatisticalSlippageProvider(
                        impact_coef=0.1,
                        spread_bps=5.0,
                    )
            else:
                # Crypto has wider spreads
                self.slippage = StatisticalSlippageProvider(
                    impact_coef=0.1,
                    spread_bps=5.0,
                )

        # Default fee provider
        if fee_provider is not None:
            self.fees = fee_provider
        else:
            if asset_class == AssetClass.EQUITY:
                self.fees = EquityFeeProvider()
            elif asset_class == AssetClass.FUTURES:
                # Futures: use FuturesFeeProvider
                try:
                    from execution_providers_futures import FuturesFeeProvider
                    self.fees = FuturesFeeProvider()
                except ImportError:
                    logger.warning(
                        "FuturesFeeProvider not available, using crypto defaults"
                    )
                    self.fees = CryptoFeeProvider()
            else:
                self.fees = CryptoFeeProvider()

        # Fill provider with injected slippage and fees
        self.fill = OHLCVFillProvider(
            slippage_provider=self.slippage,
            fee_provider=self.fees,
            fill_at_limit=kwargs.get("fill_at_limit", True),
            partial_fills=kwargs.get("partial_fills", False),
        )

    @property
    def asset_class(self) -> AssetClass:
        """Asset class this provider handles."""
        return self._asset_class

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Execute an order with full slippage/fee calculation.

        Args:
            order: Order to execute
            market: Current market state
            bar: Current bar data

        Returns:
            Fill with slippage and fees, or None if not filled
        """
        return self.fill.try_fill(order, market, bar)

    def estimate_execution_cost(
        self,
        notional: float,
        adv: float,
        side: str = "BUY",
        volatility: float = 0.02,
    ) -> Dict[str, float]:
        """
        Pre-trade cost estimation.

        Useful for optimal execution planning.

        Args:
            notional: Planned trade notional
            adv: Average daily volume
            side: Trade side
            volatility: Expected volatility

        Returns:
            Dict with cost breakdown
        """
        # Participation
        participation = notional / adv if adv > 0 else 0.01

        # Slippage estimate
        dummy_order = Order("", side, 1.0, "MARKET")
        dummy_market = MarketState(0, volatility=volatility, adv=adv)
        slippage_bps = self.slippage.compute_slippage_bps(
            dummy_order, dummy_market, participation
        )
        slippage_cost = notional * slippage_bps / 10000.0

        # Fee estimate (taker worst case)
        fee = self.fees.compute_fee(
            notional=notional,
            side=side,
            liquidity="taker",
            qty=notional / 100.0,  # Approximate qty
        )

        return {
            "participation": participation,
            "slippage_bps": slippage_bps,
            "slippage_cost": slippage_cost,
            "fee": fee,
            "total_cost": slippage_cost + fee,
            "total_bps": (slippage_cost + fee) / notional * 10000.0 if notional > 0 else 0.0,
        }


# =============================================================================
# L3 Stubs (Future LOB-based implementations)
# =============================================================================

class LOBSlippageProvider:
    """
    L3: Order book based slippage provider (FUTURE).

    Will simulate walking through the order book to compute
    actual execution price based on available liquidity.

    Features (planned):
    - Walk-through LOB simulation
    - Queue position estimation
    - Hidden liquidity modeling
    - Cross-venue routing optimization

    Status: STUB - not yet implemented
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize LOB slippage provider (stub)."""
        self._config = kwargs
        logger.warning(
            "LOBSlippageProvider is a stub. Use StatisticalSlippageProvider for now."
        )

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
    ) -> float:
        """
        Compute slippage from order book walk-through.

        IMPLEMENTATION STATUS: STUB (spread-based fallback)
        ---------------------------------------------------
        This method uses a simplified spread-based estimate instead of
        full order book walk-through simulation. This is a known limitation
        that may affect simulation accuracy for large orders.

        Sim-to-Live Gap Risk:
        - Underestimates slippage for orders consuming multiple price levels
        - Does not account for hidden liquidity or queue position
        - May overestimate available liquidity at best bid/ask

        Mitigation:
        - Use StatisticalSlippageProvider with calibrated historical data
        - Apply conservative slippage multipliers for live trading
        - Monitor sim-to-live slippage divergence in production

        Future Implementation (LOB walk-through) will:
        - Walk through bid/ask levels to compute cumulative slippage
        - Model partial fills at each price level
        - Account for order size vs available depth

        See: docs/SIMULATION_LIMITATIONS.md for validation status
        """
        # Check if L3 data available
        if market.bid_depth is None or market.ask_depth is None:
            logger.debug("LOB depth not available, using spread estimate")
            spread = market.get_spread_bps() or 10.0
            return spread / 2.0

        # STUB: Spread-based estimate (conservative upper bound)
        # Full LOB walk-through requires implementing depth consumption logic
        spread = market.get_spread_bps() or 10.0
        return spread / 2.0


class LOBFillProvider:
    """
    L3: Matching engine simulation (FUTURE).

    Will simulate a full matching engine with queue position
    tracking and time priority.

    Features (planned):
    - Time-price priority matching
    - Queue position tracking
    - Partial fill simulation
    - Hidden order modeling

    Status: STUB - not yet implemented
    """

    def __init__(
        self,
        slippage_provider: Optional[SlippageProvider] = None,
        fee_provider: Optional[FeeProvider] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LOB fill provider (stub)."""
        self.slippage = slippage_provider or StatisticalSlippageProvider()
        self.fees = fee_provider or ZeroFeeProvider()
        self._config = kwargs
        logger.warning(
            "LOBFillProvider is a stub. Use OHLCVFillProvider for now."
        )

    def try_fill(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
    ) -> Optional[Fill]:
        """
        Attempt fill with matching engine simulation.

        STUB: Delegates to OHLCV fill logic.
        """
        # Fall back to OHLCV fill provider
        ohlcv_fill = OHLCVFillProvider(
            slippage_provider=self.slippage,
            fee_provider=self.fees,
        )
        return ohlcv_fill.try_fill(order, market, bar)


# =============================================================================
# Factory Functions
# =============================================================================

def create_slippage_provider(
    level: str = "L2",
    asset_class: AssetClass = AssetClass.CRYPTO,
    **kwargs: Any,
) -> SlippageProvider:
    """
    Factory function to create slippage provider.

    Args:
        level: Fidelity level ("L1", "L2", "L3")
        asset_class: Asset class for defaults
        **kwargs: Provider-specific configuration

    Returns:
        SlippageProvider instance
    """
    level_upper = str(level).upper()

    if level_upper == "L3":
        # Import L3 provider (avoid circular import)
        try:
            from execution_providers_l3 import L3SlippageProvider
            return L3SlippageProvider(asset_class=asset_class, **kwargs)
        except ImportError:
            logger.warning(
                "L3 providers not available, falling back to L2. "
                "Ensure execution_providers_l3.py is present."
            )
            # Fall back to LOB stub
            return LOBSlippageProvider(**kwargs)

    # L1/L2: Statistical model
    if asset_class == AssetClass.FOREX:
        # FOREX: Use ForexParametricSlippageProvider (L2+)
        profile = kwargs.pop("profile", None)
        if profile:
            return ForexParametricSlippageProvider.from_profile(profile)
        config_dict = kwargs.pop("config", None)
        if config_dict:
            return ForexParametricSlippageProvider.from_config(config_dict)
        return ForexParametricSlippageProvider(**kwargs)
    elif asset_class == AssetClass.EQUITY:
        defaults = {"impact_coef": 0.05, "spread_bps": 2.0}
    else:
        defaults = {"impact_coef": 0.1, "spread_bps": 5.0}

    defaults.update(kwargs)
    return StatisticalSlippageProvider(**defaults)


def create_fee_provider(
    asset_class: AssetClass = AssetClass.CRYPTO,
    **kwargs: Any,
) -> FeeProvider:
    """
    Factory function to create fee provider.

    Args:
        asset_class: Asset class for fee structure
        **kwargs: Provider-specific configuration

    Returns:
        FeeProvider instance
    """
    if asset_class == AssetClass.EQUITY:
        return EquityFeeProvider(**kwargs)
    elif asset_class == AssetClass.CRYPTO:
        return CryptoFeeProvider(**kwargs)
    elif asset_class == AssetClass.FOREX:
        config_dict = kwargs.pop("config", None)
        if config_dict:
            return ForexFeeProvider.from_config(config_dict)
        return ForexFeeProvider(**kwargs)
    else:
        return ZeroFeeProvider()


def create_fill_provider(
    level: str = "L2",
    asset_class: AssetClass = AssetClass.CRYPTO,
    slippage_provider: Optional[SlippageProvider] = None,
    fee_provider: Optional[FeeProvider] = None,
    **kwargs: Any,
) -> FillProvider:
    """
    Factory function to create fill provider.

    Args:
        level: Fidelity level ("L2", "L3")
        asset_class: Asset class for defaults
        slippage_provider: Custom slippage provider
        fee_provider: Custom fee provider
        **kwargs: Provider-specific configuration

    Returns:
        FillProvider instance
    """
    level_upper = str(level).upper()

    # Create default providers if not provided
    if slippage_provider is None:
        slippage_provider = create_slippage_provider(level, asset_class)
    if fee_provider is None:
        fee_provider = create_fee_provider(asset_class)

    if level_upper == "L3":
        # Import L3 provider (avoid circular import)
        try:
            from execution_providers_l3 import L3FillProvider
            return L3FillProvider(
                slippage_provider=slippage_provider,
                fee_provider=fee_provider,
                asset_class=asset_class,
                **kwargs,
            )
        except ImportError:
            logger.warning(
                "L3 providers not available, falling back to LOB stub. "
                "Ensure execution_providers_l3.py is present."
            )
            return LOBFillProvider(
                slippage_provider=slippage_provider,
                fee_provider=fee_provider,
                **kwargs,
            )

    # L2: OHLCV-based fills
    return OHLCVFillProvider(
        slippage_provider=slippage_provider,
        fee_provider=fee_provider,
        **kwargs,
    )


def create_execution_provider(
    asset_class: AssetClass = AssetClass.CRYPTO,
    level: str = "L2",
    **kwargs: Any,
) -> ExecutionProvider:
    """
    Factory function to create combined execution provider.

    Args:
        asset_class: Asset class (CRYPTO, EQUITY, FUTURES, or FOREX)
        level: Fidelity level ("L2", "L3")
        **kwargs: Provider-specific configuration

    Returns:
        ExecutionProvider instance

    Example:
        # L2 provider (statistical models)
        >>> provider = create_execution_provider(AssetClass.EQUITY, level="L2")

        # L3 provider (full LOB simulation)
        >>> provider = create_execution_provider(AssetClass.EQUITY, level="L3")

        # L3 with custom config
        >>> from lob.config import L3ExecutionConfig
        >>> config = L3ExecutionConfig.for_equity()
        >>> provider = create_execution_provider(AssetClass.EQUITY, level="L3", config=config)

        # FUTURES provider (L2 with funding/OI/liquidation factors)
        >>> provider = create_execution_provider(AssetClass.FUTURES, level="L2")
        >>> # Uses FuturesSlippageProvider + FuturesFeeProvider

        # FOREX provider (L2+ parametric TCA)
        >>> provider = create_execution_provider(AssetClass.FOREX, level="L2")
        >>> provider = create_execution_provider(AssetClass.FOREX, profile="conservative")
    """
    level_upper = str(level).upper()

    if level_upper == "L3":
        # Import L3 provider (avoid circular import)
        try:
            from execution_providers_l3 import L3ExecutionProvider
            return L3ExecutionProvider(
                asset_class=asset_class,
                config=kwargs.pop("config", None),
                slippage_provider=kwargs.pop("slippage_provider", None),
                fee_provider=kwargs.pop("fee_provider", None),
                **kwargs,
            )
        except ImportError as e:
            logger.warning(
                "L3 providers not available, falling back to L2. "
                "Ensure execution_providers_l3.py is present. Error: %s", e
            )
            # Fall through to L2

    # L2: Statistical models (default)
    return L2ExecutionProvider(
        asset_class=asset_class,
        slippage_provider=kwargs.pop("slippage_provider", None),
        fee_provider=kwargs.pop("fee_provider", None),
        **kwargs,
    )


def load_slippage_profile(
    profile_name: str,
    config_path: Optional[str] = None,
) -> StatisticalSlippageProvider:
    """
    Load slippage provider from YAML config profile.

    Loads calibrated slippage parameters from configs/slippage.yaml
    or a custom config file.

    Args:
        profile_name: Profile name ("equity", "crypto", "crypto_futures", etc.)
        config_path: Optional path to YAML config file

    Returns:
        StatisticalSlippageProvider instance

    Example:
        >>> provider = load_slippage_profile("equity")
        >>> provider.spread_bps  # 2.0
        >>> provider.impact_coef  # 0.05
    """
    import os

    # Default config path
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "slippage.yaml",
        )
        if not os.path.exists(config_path):
            # Try relative to current working directory
            config_path = "configs/slippage.yaml"

    profiles_config: Optional[Mapping[str, Any]] = None

    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config and isinstance(config, Mapping):
                slippage_cfg = config.get("slippage", config)
                if isinstance(slippage_cfg, Mapping):
                    profiles_config = slippage_cfg.get("profiles")
        except ImportError:
            logger.warning("PyYAML not installed, using default profiles")
        except Exception as e:
            logger.warning("Failed to load slippage config from %s: %s", config_path, e)

    return StatisticalSlippageProvider.from_profile(profile_name, profiles_config)


def create_providers_from_asset_class(
    asset_class: AssetClass,
    slippage_config: Optional[Mapping[str, Any]] = None,
    fee_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[SlippageProvider, FeeProvider]:
    """
    Create slippage and fee providers based on asset class.

    Convenience function to create correctly configured providers
    for a given asset class.

    Args:
        asset_class: Asset class (CRYPTO, EQUITY, FUTURES)
        slippage_config: Optional slippage configuration override
        fee_config: Optional fee configuration override

    Returns:
        Tuple of (SlippageProvider, FeeProvider)

    Example:
        >>> slippage, fees = create_providers_from_asset_class(AssetClass.EQUITY)
        >>> slippage.spread_bps  # 2.0 (equity default)
    """
    # Slippage provider
    if slippage_config is not None:
        slippage = StatisticalSlippageProvider.from_config(slippage_config, asset_class)
    else:
        profile_name = {
            AssetClass.EQUITY: "equity",
            AssetClass.CRYPTO: "crypto",
            AssetClass.FUTURES: "crypto_futures",
        }.get(asset_class, "default")
        slippage = StatisticalSlippageProvider.from_profile(profile_name)

    # Fee provider
    if asset_class == AssetClass.EQUITY:
        fees: FeeProvider = (
            EquityFeeProvider.from_config(fee_config)
            if fee_config
            else EquityFeeProvider()
        )
    else:
        fees = (
            CryptoFeeProvider.from_config(fee_config)
            if fee_config
            else CryptoFeeProvider()
        )

    return slippage, fees


# =============================================================================
# Backward Compatibility: Integration with existing execution_sim.py
# =============================================================================

def wrap_legacy_slippage_config(config: Any) -> StatisticalSlippageProvider:
    """
    Create SlippageProvider from legacy SlippageConfig.

    Provides backward compatibility with existing slippage configuration.

    Args:
        config: Legacy SlippageConfig or dict

    Returns:
        StatisticalSlippageProvider instance
    """
    if config is None:
        return StatisticalSlippageProvider()

    # Extract parameters from legacy config
    if hasattr(config, "k"):
        impact_coef = float(getattr(config, "k", 0.1))
    elif isinstance(config, Mapping) and "k" in config:
        impact_coef = float(config.get("k", 0.1))
    else:
        impact_coef = 0.1

    if hasattr(config, "default_spread_bps"):
        spread_bps = float(getattr(config, "default_spread_bps", 5.0))
    elif isinstance(config, Mapping) and "default_spread_bps" in config:
        spread_bps = float(config.get("default_spread_bps", 5.0))
    else:
        spread_bps = 5.0

    return StatisticalSlippageProvider(
        impact_coef=impact_coef,
        spread_bps=spread_bps,
    )


def wrap_legacy_fees_model(model: Any) -> FeeProvider:
    """
    Create FeeProvider from legacy FeesModel.

    Provides backward compatibility with existing fees configuration.

    Args:
        model: Legacy FeesModel or dict

    Returns:
        FeeProvider instance
    """
    if model is None:
        return CryptoFeeProvider()

    # Extract parameters from legacy model
    if hasattr(model, "maker_rate_bps"):
        maker_bps = float(getattr(model, "maker_rate_bps", 2.0))
    elif isinstance(model, Mapping) and "maker_rate_bps" in model:
        maker_bps = float(model.get("maker_rate_bps", 2.0))
    else:
        maker_bps = 2.0

    if hasattr(model, "taker_rate_bps"):
        taker_bps = float(getattr(model, "taker_rate_bps", 4.0))
    elif isinstance(model, Mapping) and "taker_rate_bps" in model:
        taker_bps = float(model.get("taker_rate_bps", 4.0))
    else:
        taker_bps = 4.0

    return CryptoFeeProvider(
        maker_bps=maker_bps,
        taker_bps=taker_bps,
    )
