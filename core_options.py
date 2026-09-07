# -*- coding: utf-8 -*-
"""
core_options.py
Options trading core models and data structures.

Phase 1: Core Models & Data Structures

This module provides foundational options data structures used throughout
the platform. It integrates with existing infrastructure:
- AssetClass.OPTIONS from execution_providers.py
- OptionType, OptionContract from adapters/alpaca/options_execution.py
- BotError from core_errors.py

References:
    - Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
    - Merton (1973): "Theory of Rational Option Pricing"
    - Hull (2018): "Options, Futures, and Other Derivatives" (10th ed.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Import existing types from Alpaca adapter (DO NOT redefine!)
from adapters.alpaca.options_execution import (
    OptionType,
    OptionStrategy,
    OptionOrderType,
    OptionContract,
    OptionChain,
    OPTIONS_CONTRACT_MULTIPLIER,
)

# Import existing asset class enum
from execution_providers import AssetClass

# Import base error class
from core_errors import BotError


# =============================================================================
# Options-Specific Enums (New, not in Alpaca adapter)
# =============================================================================


class ExerciseStyle(str, Enum):
    """
    Exercise style for options contracts.

    AMERICAN: Can be exercised any time before expiration
    EUROPEAN: Can only be exercised at expiration
    BERMUDAN: Can be exercised on specific dates (rare, not commonly supported)
    """

    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"


class SettlementType(str, Enum):
    """
    Settlement type for options contracts.

    PHYSICAL: Delivery of underlying asset (most equity options)
    CASH: Cash settlement (SPX, VIX, most index options)
    """

    PHYSICAL = "physical"
    CASH = "cash"


class ExerciseDecision(str, Enum):
    """
    Decision for early exercise analysis.

    Reference: Broadie & Detemple (1996) "American Option Valuation"
    """

    HOLD = "hold"  # Continue holding the option
    EXERCISE = "exercise"  # Exercise immediately
    UNCERTAIN = "uncertain"  # Near decision boundary


class VolatilityType(str, Enum):
    """Type of volatility measure."""

    IMPLIED = "implied"  # Market-implied volatility
    HISTORICAL = "historical"  # Realized historical volatility
    REALIZED = "realized"  # Same as historical, different naming
    FORWARD = "forward"  # Forward-starting implied volatility


class MoneynessBucket(str, Enum):
    """
    Moneyness classification for options.

    Based on delta or strike-to-spot ratio.
    """

    DEEP_ITM = "deep_itm"  # |delta| > 0.80
    ITM = "itm"  # 0.55 < |delta| <= 0.80
    ATM = "atm"  # 0.45 <= |delta| <= 0.55
    OTM = "otm"  # 0.20 <= |delta| < 0.45
    DEEP_OTM = "deep_otm"  # |delta| < 0.20


# =============================================================================
# Options Contract Specification
# =============================================================================


@dataclass
class OptionsContractSpec:
    """
    Extended options contract specification.

    Extends the existing OptionContract from adapters/alpaca/options_execution.py
    with additional fields for multi-exchange support and advanced pricing.

    Attributes:
        symbol: Full OCC symbol (e.g., "AAPL240315C00175000")
        underlying: Underlying symbol (e.g., "AAPL")
        option_type: CALL or PUT
        strike: Strike price
        expiration: Expiration date
        exercise_style: AMERICAN or EUROPEAN
        settlement: PHYSICAL or CASH
        multiplier: Contract multiplier (100 for US equities)
        tick_size: Minimum price increment
        exchange: Primary exchange
        asset_class: Asset class (always OPTIONS)
        root_symbol: Options root symbol (may differ from underlying)
        underlying_type: Type of underlying (equity, index, etf, futures)
    """

    symbol: str
    underlying: str
    option_type: OptionType
    strike: Decimal
    expiration: date
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN
    settlement: SettlementType = SettlementType.PHYSICAL
    multiplier: int = OPTIONS_CONTRACT_MULTIPLIER
    tick_size: Decimal = Decimal("0.01")
    exchange: str = "CBOE"
    asset_class: AssetClass = AssetClass.OPTIONS
    root_symbol: Optional[str] = None
    underlying_type: str = "equity"

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        # Convert string types to enums if needed
        if isinstance(self.option_type, str):
            self.option_type = OptionType(self.option_type.lower())
        if isinstance(self.exercise_style, str):
            self.exercise_style = ExerciseStyle(self.exercise_style.lower())
        if isinstance(self.settlement, str):
            self.settlement = SettlementType(self.settlement.lower())

        # Ensure Decimal types
        if not isinstance(self.strike, Decimal):
            self.strike = Decimal(str(self.strike))
        if not isinstance(self.tick_size, Decimal):
            self.tick_size = Decimal(str(self.tick_size))

        # Set root symbol if not provided
        if self.root_symbol is None:
            self.root_symbol = self.underlying

        # Validate strike is positive
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")

        # Validate multiplier is positive
        if self.multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {self.multiplier}")

    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT

    @property
    def is_american(self) -> bool:
        """Check if option has American exercise style."""
        return self.exercise_style == ExerciseStyle.AMERICAN

    @property
    def is_european(self) -> bool:
        """Check if option has European exercise style."""
        return self.exercise_style == ExerciseStyle.EUROPEAN

    @property
    def is_cash_settled(self) -> bool:
        """Check if option is cash settled."""
        return self.settlement == SettlementType.CASH

    @property
    def strike_float(self) -> float:
        """Get strike as float for calculations."""
        return float(self.strike)

    def time_to_expiry(self, valuation_date: Optional[date] = None) -> float:
        """
        Calculate time to expiration in years.

        Args:
            valuation_date: Date to calculate from (default: today)

        Returns:
            Time to expiration in years (ACT/365)
        """
        if valuation_date is None:
            valuation_date = date.today()
        days = (self.expiration - valuation_date).days
        return max(0.0, days / 365.0)

    def days_to_expiry(self, valuation_date: Optional[date] = None) -> int:
        """Get days until expiration."""
        if valuation_date is None:
            valuation_date = date.today()
        return (self.expiration - valuation_date).days

    def is_expired(self, valuation_date: Optional[date] = None) -> bool:
        """Check if option is expired."""
        return self.days_to_expiry(valuation_date) < 0

    def moneyness(self, spot: float) -> float:
        """
        Calculate simple moneyness (Strike/Spot).

        Args:
            spot: Current underlying price

        Returns:
            Moneyness ratio (K/S)
        """
        if spot <= 0:
            raise ValueError(f"Spot must be positive, got {spot}")
        return self.strike_float / spot

    def log_moneyness(self, spot: float) -> float:
        """
        Calculate log-moneyness ln(K/S).

        Args:
            spot: Current underlying price

        Returns:
            Log-moneyness
        """
        import math

        return math.log(self.moneyness(spot))

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value.

        Args:
            spot: Current underlying price

        Returns:
            Intrinsic value (>=0)
        """
        if self.is_call:
            return max(0.0, spot - self.strike_float)
        else:
            return max(0.0, self.strike_float - spot)

    def is_itm(self, spot: float) -> bool:
        """Check if option is in-the-money."""
        return self.intrinsic_value(spot) > 0

    def is_otm(self, spot: float) -> bool:
        """Check if option is out-of-the-money."""
        return self.intrinsic_value(spot) == 0

    def is_atm(self, spot: float, tolerance: float = 0.02) -> bool:
        """
        Check if option is at-the-money.

        Args:
            spot: Current underlying price
            tolerance: Tolerance for ATM classification (default 2%)

        Returns:
            True if strike is within tolerance of spot
        """
        return abs(self.moneyness(spot) - 1.0) <= tolerance

    def classify_moneyness(self, spot: float, delta: Optional[float] = None) -> MoneynessBucket:
        """
        Classify option moneyness into bucket.

        Args:
            spot: Current underlying price
            delta: Option delta (if available, preferred method)

        Returns:
            MoneynessBucket classification
        """
        if delta is not None:
            abs_delta = abs(delta)
            if abs_delta > 0.80:
                return MoneynessBucket.DEEP_ITM
            elif abs_delta > 0.55:
                return MoneynessBucket.ITM
            elif abs_delta >= 0.45:
                return MoneynessBucket.ATM
            elif abs_delta >= 0.20:
                return MoneynessBucket.OTM
            else:
                return MoneynessBucket.DEEP_OTM

        # Fallback to moneyness ratio
        m = self.moneyness(spot)
        # For calls: ITM if K < S (m < 1), OTM if K > S (m > 1)
        # For puts: ITM if K > S (m > 1), OTM if K < S (m < 1)
        if self.is_call:
            if m < 0.85:
                return MoneynessBucket.DEEP_ITM
            elif m < 0.95:
                return MoneynessBucket.ITM
            elif m <= 1.05:
                return MoneynessBucket.ATM
            elif m <= 1.15:
                return MoneynessBucket.OTM
            else:
                return MoneynessBucket.DEEP_OTM
        else:
            if m > 1.15:
                return MoneynessBucket.DEEP_ITM
            elif m > 1.05:
                return MoneynessBucket.ITM
            elif m >= 0.95:
                return MoneynessBucket.ATM
            elif m >= 0.85:
                return MoneynessBucket.OTM
            else:
                return MoneynessBucket.DEEP_OTM

    def to_occ_symbol(self) -> str:
        """
        Generate OCC option symbol.

        Format: SYMBOL (6 padded) + YYMMDD + C/P + Strike (8 digits, 5.3)
        Example: "AAPL  241220C00200000" = AAPL Dec 20, 2024 $200 Call
        """
        symbol_padded = self.underlying.ljust(6)
        date_str = self.expiration.strftime("%y%m%d")
        opt_char = "C" if self.is_call else "P"
        # Strike in 5.3 format (5 whole digits, 3 decimal)
        strike_int = int(self.strike * 1000)
        strike_str = f"{strike_int:08d}"
        return f"{symbol_padded}{date_str}{opt_char}{strike_str}"

    @classmethod
    def from_occ_symbol(cls, occ_symbol: str, **kwargs) -> "OptionsContractSpec":
        """
        Parse OCC symbol into OptionsContractSpec.

        Args:
            occ_symbol: OCC format symbol
            **kwargs: Additional fields (exercise_style, settlement, etc.)

        Returns:
            OptionsContractSpec instance
        """
        # Use existing parser from OptionContract
        basic_contract = OptionContract.from_occ_symbol(occ_symbol)

        return cls(
            symbol=occ_symbol.replace(" ", ""),
            underlying=basic_contract.symbol,
            option_type=basic_contract.option_type,
            strike=Decimal(str(basic_contract.strike_price)),
            expiration=basic_contract.expiration_date,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": str(self.strike),
            "expiration": self.expiration.isoformat(),
            "exercise_style": self.exercise_style.value,
            "settlement": self.settlement.value,
            "multiplier": self.multiplier,
            "tick_size": str(self.tick_size),
            "exchange": self.exchange,
            "asset_class": self.asset_class.value,
            "root_symbol": self.root_symbol,
            "underlying_type": self.underlying_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptionsContractSpec":
        """Create from dictionary representation."""
        return cls(
            symbol=data["symbol"],
            underlying=data["underlying"],
            option_type=OptionType(data["option_type"]),
            strike=Decimal(data["strike"]),
            expiration=date.fromisoformat(data["expiration"]),
            exercise_style=ExerciseStyle(data.get("exercise_style", "american")),
            settlement=SettlementType(data.get("settlement", "physical")),
            multiplier=data.get("multiplier", OPTIONS_CONTRACT_MULTIPLIER),
            tick_size=Decimal(data.get("tick_size", "0.01")),
            exchange=data.get("exchange", "CBOE"),
            root_symbol=data.get("root_symbol"),
            underlying_type=data.get("underlying_type", "equity"),
        )


# =============================================================================
# Greeks Result
# =============================================================================


@dataclass
class GreeksResult:
    """
    Complete Greeks result for an option.

    Contains all 12 Greeks (first, second, and third order derivatives).

    First-Order Greeks:
        delta: ∂V/∂S - Directional exposure
        gamma: ∂²V/∂S² - Convexity risk (rate of delta change)
        theta: ∂V/∂t - Time decay (per day, NOT per year)
        vega: ∂V/∂σ - Volatility exposure (per 1% vol = 0.01)
        rho: ∂V/∂r - Interest rate sensitivity (per 1% rate)

    Second-Order Greeks:
        vanna: ∂²V/∂S∂σ - Delta-vol correlation (skew risk)
        volga: ∂²V/∂σ² - Volatility-of-volatility (Vomma)
        charm: ∂²V/∂S∂t - Delta decay rate (Delta bleed)

    Third-Order Greeks (critical for market makers):
        speed: ∂³V/∂S³ - Gamma convexity (rate of gamma change)
        color: ∂³V/∂S²∂t - Gamma decay rate
        zomma: ∂³V/∂S²∂σ - Gamma-vol sensitivity
        ultima: ∂³V/∂σ³ - Volga-vol sensitivity (Vol-of-vol-of-vol)

    References:
        - Taleb (1997): "Dynamic Hedging"
        - Haug (2007): "The Complete Guide to Option Pricing Formulas"
    """

    # First-order Greeks
    delta: float
    gamma: float
    theta: float  # Per day (NOT per year)
    vega: float  # Per 1% vol (0.01 absolute)
    rho: float  # Per 1% rate

    # Second-order Greeks
    vanna: float  # ∂Δ/∂σ = ∂ν/∂S
    volga: float  # ∂ν/∂σ (Vomma)
    charm: float  # ∂Δ/∂t (Delta bleed)

    # Third-order Greeks (critical for MM risk)
    speed: float  # ∂Γ/∂S
    color: float  # ∂Γ/∂t
    zomma: float  # ∂Γ/∂σ
    ultima: float  # ∂Volga/∂σ

    # Metadata
    timestamp_ns: int = 0
    spot: Optional[float] = None
    strike: Optional[float] = None
    time_to_expiry: Optional[float] = None
    volatility: Optional[float] = None
    rate: Optional[float] = None
    dividend_yield: Optional[float] = None

    def to_dollar_terms(self, multiplier: int = 100) -> "GreeksResult":
        """
        Convert Greeks to dollar terms.

        Dollar delta = delta * spot * multiplier
        Dollar gamma = 0.5 * gamma * spot^2 * multiplier (1% move)
        Dollar theta = theta * multiplier
        Dollar vega = vega * multiplier

        Args:
            multiplier: Contract multiplier (default 100)

        Returns:
            New GreeksResult with dollar-scaled Greeks
        """
        spot = self.spot or 100.0  # Default if not set

        return GreeksResult(
            # First-order (dollar scaled)
            delta=self.delta * spot * multiplier,
            gamma=0.5 * self.gamma * spot * spot * multiplier * 0.01,  # For 1% move
            theta=self.theta * multiplier,
            vega=self.vega * multiplier,
            rho=self.rho * multiplier,
            # Second-order (dollar scaled)
            vanna=self.vanna * spot * multiplier,
            volga=self.volga * multiplier,
            charm=self.charm * spot * multiplier,
            # Third-order (dollar scaled)
            speed=self.speed * spot * spot * multiplier,
            color=self.color * spot * spot * multiplier,
            zomma=self.zomma * spot * multiplier,
            ultima=self.ultima * multiplier,
            # Metadata
            timestamp_ns=self.timestamp_ns,
            spot=self.spot,
            strike=self.strike,
            time_to_expiry=self.time_to_expiry,
            volatility=self.volatility,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "speed": self.speed,
            "color": self.color,
            "zomma": self.zomma,
            "ultima": self.ultima,
            "timestamp_ns": self.timestamp_ns,
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "volatility": self.volatility,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GreeksResult":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def zero(cls) -> "GreeksResult":
        """Create a zero Greeks result (for expired options)."""
        return cls(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            vanna=0.0,
            volga=0.0,
            charm=0.0,
            speed=0.0,
            color=0.0,
            zomma=0.0,
            ultima=0.0,
        )


# =============================================================================
# Pricing Result
# =============================================================================


@dataclass
class PricingResult:
    """
    Result of options pricing calculation.

    Contains price, Greeks, and metadata about the pricing.
    """

    price: float
    greeks: GreeksResult

    # Pricing metadata
    model: str = "black_scholes"  # "black_scholes", "binomial", "monte_carlo", etc.
    iterations: int = 0
    convergence_error: float = 0.0
    computation_time_us: int = 0

    # Inputs used
    spot: float = 0.0
    strike: float = 0.0
    time_to_expiry: float = 0.0
    volatility: float = 0.0
    rate: float = 0.0
    dividend_yield: float = 0.0
    is_call: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": self.price,
            "greeks": self.greeks.to_dict(),
            "model": self.model,
            "iterations": self.iterations,
            "convergence_error": self.convergence_error,
            "computation_time_us": self.computation_time_us,
            "spot": self.spot,
            "strike": self.strike,
            "time_to_expiry": self.time_to_expiry,
            "volatility": self.volatility,
            "rate": self.rate,
            "dividend_yield": self.dividend_yield,
            "is_call": self.is_call,
        }


# =============================================================================
# IV Result
# =============================================================================


@dataclass
class IVResult:
    """
    Result of implied volatility calculation.

    Contains IV value and solver metadata.
    """

    implied_volatility: float
    converged: bool
    iterations: int
    error: float
    method: str = "hybrid"  # "newton", "brent", "hybrid", "jaeckel"

    # Pricing verification
    model_price: Optional[float] = None
    market_price: Optional[float] = None
    price_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "implied_volatility": self.implied_volatility,
            "converged": self.converged,
            "iterations": self.iterations,
            "error": self.error,
            "method": self.method,
            "model_price": self.model_price,
            "market_price": self.market_price,
            "price_error": self.price_error,
        }


# =============================================================================
# Early Exercise
# =============================================================================


@dataclass
class EarlyExerciseResult:
    """
    Result of early exercise analysis.

    Reference: Broadie & Detemple (1996) "American Option Valuation"
    """

    decision: ExerciseDecision
    exercise_probability: float  # Probability of early exercise
    optimal_boundary: float  # Optimal exercise boundary (spot level)
    continuation_value: float  # Value of continuing to hold
    exercise_value: float  # Value of immediate exercise

    # Additional info
    time_to_expiry: float = 0.0
    dividend_effect: bool = False  # True if dividend-driven

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "exercise_probability": self.exercise_probability,
            "optimal_boundary": self.optimal_boundary,
            "continuation_value": self.continuation_value,
            "exercise_value": self.exercise_value,
            "time_to_expiry": self.time_to_expiry,
            "dividend_effect": self.dividend_effect,
        }


# =============================================================================
# Dividend Model
# =============================================================================


@dataclass
class Dividend:
    """
    Dividend payment information.

    Used for discrete dividend handling in American options pricing.
    """

    ex_date: date
    amount: float
    declared_date: Optional[date] = None
    record_date: Optional[date] = None
    payment_date: Optional[date] = None

    def present_value(self, valuation_date: date, rate: float) -> float:
        """
        Calculate present value of dividend.

        Args:
            valuation_date: Date to discount to
            rate: Risk-free rate (annualized)

        Returns:
            Present value of dividend
        """
        import math

        days = (self.ex_date - valuation_date).days
        if days <= 0:
            return 0.0
        t = days / 365.0
        return self.amount * math.exp(-rate * t)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ex_date": self.ex_date.isoformat(),
            "amount": self.amount,
            "declared_date": self.declared_date.isoformat() if self.declared_date else None,
            "record_date": self.record_date.isoformat() if self.record_date else None,
            "payment_date": self.payment_date.isoformat() if self.payment_date else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dividend":
        """Create from dictionary."""
        return cls(
            ex_date=date.fromisoformat(data["ex_date"]),
            amount=data["amount"],
            declared_date=(
                date.fromisoformat(data["declared_date"]) if data.get("declared_date") else None
            ),
            record_date=(
                date.fromisoformat(data["record_date"]) if data.get("record_date") else None
            ),
            payment_date=(
                date.fromisoformat(data["payment_date"]) if data.get("payment_date") else None
            ),
        )


# =============================================================================
# Jump Parameters
# =============================================================================


@dataclass
class JumpParams:
    """
    Merton jump-diffusion model parameters.

    Reference: Merton (1976) "Option pricing when underlying stock returns
               are discontinuous"

    Model:
        dS/S = (μ - λk)dt + σdW + (J-1)dN

        where:
        - λ = jump intensity (jumps per year)
        - k = E[J-1] = expected relative jump size
        - J = jump multiplier (lognormal)
        - N = Poisson process
    """

    lambda_intensity: float  # Jumps per year (λ)
    mu_jump: float  # Mean log jump size (μ_J)
    sigma_jump: float  # Jump size std dev (σ_J)
    calibration_method: str = "earnings"
    calibration_date: Optional[date] = None
    confidence_interval: Optional[Tuple[float, float]] = None  # 95% CI for λ

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.lambda_intensity < 0:
            raise ValueError(f"Jump intensity must be non-negative, got {self.lambda_intensity}")
        if self.sigma_jump < 0:
            raise ValueError(f"Jump volatility must be non-negative, got {self.sigma_jump}")

    @property
    def expected_jump_size(self) -> float:
        """Expected relative jump size k = E[J-1] = exp(μ_J + σ_J²/2) - 1."""
        import math

        return math.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1

    @property
    def jump_variance_contribution(self) -> float:
        """Variance contribution from jumps per year."""
        import math

        k = self.expected_jump_size
        return self.lambda_intensity * (self.sigma_jump**2 + k**2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lambda_intensity": self.lambda_intensity,
            "mu_jump": self.mu_jump,
            "sigma_jump": self.sigma_jump,
            "calibration_method": self.calibration_method,
            "calibration_date": (
                self.calibration_date.isoformat() if self.calibration_date else None
            ),
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JumpParams":
        """Create from dictionary."""
        return cls(
            lambda_intensity=data["lambda_intensity"],
            mu_jump=data["mu_jump"],
            sigma_jump=data["sigma_jump"],
            calibration_method=data.get("calibration_method", "earnings"),
            calibration_date=(
                date.fromisoformat(data["calibration_date"])
                if data.get("calibration_date")
                else None
            ),
            confidence_interval=(
                tuple(data["confidence_interval"]) if data.get("confidence_interval") else None
            ),
        )

    @classmethod
    def default_equity(cls) -> "JumpParams":
        """Default jump parameters for equity options."""
        return cls(
            lambda_intensity=1.0,  # 1 jump per year
            mu_jump=-0.05,  # 5% down on average
            sigma_jump=0.15,  # 15% jump volatility
            calibration_method="default",
        )

    @classmethod
    def high_volatility(cls) -> "JumpParams":
        """Jump parameters for high volatility regimes (earnings, M&A)."""
        return cls(
            lambda_intensity=3.0,  # 3 jumps per year
            mu_jump=-0.08,  # 8% down on average
            sigma_jump=0.25,  # 25% jump volatility
            calibration_method="high_vol",
        )


# =============================================================================
# Variance Swap
# =============================================================================


@dataclass
class VarianceSwapQuote:
    """
    Variance swap fair strike calculation result.

    Reference: Carr & Madan (1998) "Towards a theory of volatility trading"

    Fair variance strike:
        K_var² = (2/T) × [∫₀^F (1/K²)P(K)dK + ∫_F^∞ (1/K²)C(K)dK]
    """

    variance_strike: float  # K_var (in volatility terms, e.g., 0.20 for 20%)
    variance_strike_squared: float  # K_var² (in variance terms)
    forward_price: float
    time_to_expiry: float
    num_strikes_used: int

    # Components
    otm_put_contribution: float  # ∫₀^F contribution
    otm_call_contribution: float  # ∫_F^∞ contribution

    # For variance_swap_value calculation
    vega_notional: float = 1.0  # Vega notional (defaults to 1.0)

    @property
    def strike_variance(self) -> float:
        """Alias for variance_strike_squared for API compatibility."""
        return self.variance_strike_squared

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variance_strike": self.variance_strike,
            "variance_strike_squared": self.variance_strike_squared,
            "forward_price": self.forward_price,
            "time_to_expiry": self.time_to_expiry,
            "num_strikes_used": self.num_strikes_used,
            "otm_put_contribution": self.otm_put_contribution,
            "otm_call_contribution": self.otm_call_contribution,
            "vega_notional": self.vega_notional,
        }


# =============================================================================
# Re-export existing types for convenience
# =============================================================================

__all__ = [
    # From Alpaca adapter (re-exported)
    "OptionType",
    "OptionStrategy",
    "OptionOrderType",
    "OptionContract",
    "OptionChain",
    "OPTIONS_CONTRACT_MULTIPLIER",
    # From execution_providers
    "AssetClass",
    # New enums
    "ExerciseStyle",
    "SettlementType",
    "ExerciseDecision",
    "VolatilityType",
    "MoneynessBucket",
    # New data classes
    "OptionsContractSpec",
    "GreeksResult",
    "PricingResult",
    "IVResult",
    "EarlyExerciseResult",
    "Dividend",
    "JumpParams",
    "VarianceSwapQuote",
]
