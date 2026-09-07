# -*- coding: utf-8 -*-
"""
impl_greeks.py
Black-Scholes Greeks implementation with all 12 derivatives.

Phase 1: Core Models & Data Structures

This module provides analytical Black-Scholes Greeks calculations for
European options. All 12 Greeks are implemented:

First-Order Greeks:
    - Delta (∂V/∂S): Directional exposure
    - Gamma (∂²V/∂S²): Convexity risk
    - Theta (∂V/∂t): Time decay
    - Vega (∂V/∂σ): Volatility exposure
    - Rho (∂V/∂r): Interest rate sensitivity

Second-Order Greeks:
    - Vanna (∂²V/∂S∂σ): Delta-vol correlation
    - Volga/Vomma (∂²V/∂σ²): Vol-of-vol sensitivity
    - Charm (∂²V/∂S∂t): Delta decay

Third-Order Greeks:
    - Speed (∂³V/∂S³): Gamma convexity
    - Color (∂³V/∂S²∂t): Gamma decay
    - Zomma (∂³V/∂S²∂σ): Gamma-vol sensitivity
    - Ultima (∂³V/∂σ³): Volga-vol sensitivity

References:
    - Black & Scholes (1973): "The Pricing of Options"
    - Merton (1973): "Theory of Rational Option Pricing"
    - Taleb (1997): "Dynamic Hedging"
    - Haug (2007): "The Complete Guide to Option Pricing Formulas"
    - Gatheral (2006): "The Volatility Surface"
"""

from __future__ import annotations

import math
import time
from datetime import date
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from core_options import GreeksResult, OptionsContractSpec, OptionType
from core_errors import GreeksCalculationError


# =============================================================================
# Constants
# =============================================================================

# Mathematical constants
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

# Numerical tolerances
_MIN_TIME = 1e-10  # Minimum time to avoid division by zero
_MIN_VOLATILITY = 1e-10  # Minimum volatility to avoid division by zero
_MIN_SPOT = 1e-10  # Minimum spot to avoid division by zero
_MIN_STRIKE = 1e-10  # Minimum strike


# =============================================================================
# Standard Normal Distribution Functions
# =============================================================================


def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Uses the error function for accuracy.

    Args:
        x: Input value

    Returns:
        N(x) = P(Z <= x)
    """
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


def _norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    Args:
        x: Input value

    Returns:
        n(x) = (1/√2π) * exp(-x²/2)
    """
    return math.exp(-0.5 * x * x) / _SQRT_2PI


# =============================================================================
# Black-Scholes d1, d2 Calculation
# =============================================================================


def _compute_d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> Tuple[float, float]:
    """
    Compute Black-Scholes d1 and d2.

    d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate (annualized)
        dividend_yield: Continuous dividend yield (annualized)
        volatility: Annualized volatility (e.g., 0.20 for 20%)

    Returns:
        Tuple of (d1, d2)
    """
    if time_to_expiry < _MIN_TIME:
        time_to_expiry = _MIN_TIME
    if volatility < _MIN_VOLATILITY:
        volatility = _MIN_VOLATILITY
    if spot < _MIN_SPOT:
        spot = _MIN_SPOT
    if strike < _MIN_STRIKE:
        strike = _MIN_STRIKE

    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t

    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return d1, d2


# =============================================================================
# First-Order Greeks
# =============================================================================


def compute_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> float:
    """
    Compute option delta (∂V/∂S).

    Delta represents the option's sensitivity to changes in the underlying price.

    Call: Δ = e^(-qT) × N(d1)
    Put:  Δ = e^(-qT) × [N(d1) - 1] = -e^(-qT) × N(-d1)

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put

    Returns:
        Delta value (-1 to 1 for most options)
    """
    if time_to_expiry < _MIN_TIME:
        # At expiration: delta is 1 or 0 for ITM/OTM
        if is_call:
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    d1, _ = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return exp_div * _norm_cdf(d1)
    else:
        return exp_div * (_norm_cdf(d1) - 1.0)


def compute_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option gamma (∂²V/∂S²).

    Gamma represents the rate of change of delta with respect to spot.
    Same for calls and puts.

    Γ = e^(-qT) × n(d1) / (S × σ × √T)

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Gamma value (always positive)
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY or spot < _MIN_SPOT:
        return 0.0

    d1, _ = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    return exp_div * _norm_pdf(d1) / (spot * volatility * sqrt_t)


def compute_theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    per_day: bool = True,
) -> float:
    """
    Compute option theta (∂V/∂t).

    Theta represents time decay - the rate at which option value decreases
    as time passes.

    Call: Θ = -[S × e^(-qT) × n(d1) × σ / (2√T)]
             - r × K × e^(-rT) × N(d2)
             + q × S × e^(-qT) × N(d1)

    Put:  Θ = -[S × e^(-qT) × n(d1) × σ / (2√T)]
             + r × K × e^(-rT) × N(-d2)
             - q × S × e^(-qT) × N(-d1)

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        per_day: If True, return theta per day; else per year

    Returns:
        Theta value (typically negative, representing time decay)
    """
    if time_to_expiry < _MIN_TIME:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)
    exp_rate = math.exp(-rate * time_to_expiry)
    n_d1 = _norm_pdf(d1)

    # First term: common to both
    term1 = -(spot * exp_div * n_d1 * volatility) / (2.0 * sqrt_t)

    if is_call:
        term2 = -rate * strike * exp_rate * _norm_cdf(d2)
        term3 = dividend_yield * spot * exp_div * _norm_cdf(d1)
    else:
        term2 = rate * strike * exp_rate * _norm_cdf(-d2)
        term3 = -dividend_yield * spot * exp_div * _norm_cdf(-d1)

    theta = term1 + term2 + term3

    if per_day:
        return theta / 365.0
    return theta


def compute_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option vega (∂V/∂σ).

    Vega represents sensitivity to changes in implied volatility.
    Same for calls and puts.

    ν = S × e^(-qT) × √T × n(d1)

    Note: Result is per 1% volatility change (0.01 absolute).

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Vega value (per 1% vol change)
    """
    if time_to_expiry < _MIN_TIME:
        return 0.0

    d1, _ = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    # Vega per 1.0 volatility change
    vega = spot * exp_div * sqrt_t * _norm_pdf(d1)

    # Return per 0.01 (1%) volatility change
    return vega * 0.01


def compute_rho(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> float:
    """
    Compute option rho (∂V/∂r).

    Rho represents sensitivity to changes in interest rates.

    Call: ρ = K × T × e^(-rT) × N(d2)
    Put:  ρ = -K × T × e^(-rT) × N(-d2)

    Note: Result is per 1% rate change (0.01 absolute).

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put

    Returns:
        Rho value (per 1% rate change)
    """
    if time_to_expiry < _MIN_TIME:
        return 0.0

    _, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    exp_rate = math.exp(-rate * time_to_expiry)

    if is_call:
        rho = strike * time_to_expiry * exp_rate * _norm_cdf(d2)
    else:
        rho = -strike * time_to_expiry * exp_rate * _norm_cdf(-d2)

    # Return per 0.01 (1%) rate change
    return rho * 0.01


# =============================================================================
# Second-Order Greeks
# =============================================================================


def compute_vanna(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option vanna (∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S).

    Vanna represents the sensitivity of delta to volatility changes,
    or equivalently, the sensitivity of vega to spot changes.
    Critical for managing skew risk.

    Vanna = -e^(-qT) × n(d1) × d2 / σ

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Vanna value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    return -exp_div * _norm_pdf(d1) * d2 / volatility


def compute_volga(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option volga/vomma (∂²V/∂σ²).

    Volga (also called Vomma) represents the sensitivity of vega to
    volatility changes. Critical for vol-of-vol exposure.

    Volga = ν × d1 × d2 / σ

    where ν is vega (per unit vol change, not per 1%)

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Volga value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    # Vega per unit vol
    vega_unit = spot * exp_div * sqrt_t * _norm_pdf(d1)

    return vega_unit * d1 * d2 / volatility


def compute_charm(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    per_day: bool = True,
) -> float:
    """
    Compute option charm (∂²V/∂S∂t = ∂Δ/∂t).

    Charm represents the rate at which delta changes as time passes
    (delta bleed). Important for hedging adjustment timing.

    Charm = -e^(-qT) × n(d1) × [q + (2(r-q)T - d2×σ×√T) / (2T×σ×√T)]

    For calls: charm is typically negative near ATM (delta decreases)
    For puts: charm is typically positive near ATM (delta increases toward -1)

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        per_day: If True, return charm per day; else per year

    Returns:
        Charm value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)
    n_d1 = _norm_pdf(d1)

    # Compute the charm formula
    term = 2.0 * (rate - dividend_yield) * time_to_expiry - d2 * volatility * sqrt_t
    charm = -exp_div * n_d1 * (dividend_yield + term / (2.0 * time_to_expiry * volatility * sqrt_t))

    # Adjust sign for puts
    if not is_call:
        # For puts, add the dividend adjustment
        charm = charm + dividend_yield * exp_div * _norm_cdf(-d1)

    if per_day:
        return charm / 365.0
    return charm


# =============================================================================
# Third-Order Greeks
# =============================================================================


def compute_speed(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option speed (∂³V/∂S³ = ∂Γ/∂S).

    Speed represents how gamma changes as spot moves. Critical for
    large position management and market maker risk.

    Speed = -Γ × (1 + d1/(σ√T)) / S

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Speed value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY or spot < _MIN_SPOT:
        return 0.0

    gamma = compute_gamma(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    d1, _ = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)

    return -gamma * (1.0 + d1 / (volatility * sqrt_t)) / spot


def compute_color(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    per_day: bool = True,
) -> float:
    """
    Compute option color (∂³V/∂S²∂t = ∂Γ/∂t).

    Color represents how gamma changes as time passes. Important for
    understanding gamma decay as expiration approaches.

    Color = -e^(-qT) × n(d1) / (2 × S × T × σ × √T) ×
            [2qT + 1 + d1 × (2(r-q)T - d2×σ×√T) / (σ×√T)]

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        per_day: If True, return color per day; else per year

    Returns:
        Color value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY or spot < _MIN_SPOT:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)
    n_d1 = _norm_pdf(d1)

    vol_sqrt_t = volatility * sqrt_t

    term1 = 2.0 * dividend_yield * time_to_expiry + 1.0
    term2 = d1 * (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * vol_sqrt_t) / vol_sqrt_t

    color = -exp_div * n_d1 * (term1 + term2) / (2.0 * spot * time_to_expiry * vol_sqrt_t)

    if per_day:
        return color / 365.0
    return color


def compute_zomma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option zomma (∂³V/∂S²∂σ = ∂Γ/∂σ).

    Zomma represents how gamma changes as volatility changes.
    Important for understanding gamma risk in volatile markets.

    Zomma = Γ × (d1×d2 - 1) / σ

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Zomma value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY:
        return 0.0

    gamma = compute_gamma(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)

    return gamma * (d1 * d2 - 1.0) / volatility


def compute_ultima(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    """
    Compute option ultima (∂³V/∂σ³ = ∂Volga/∂σ).

    Ultima represents how volga changes as volatility changes.
    Third derivative with respect to volatility.

    Ultima = -ν × [d1×d2×(1-d1×d2) + d1² + d2²] / σ²

    where ν is vega per unit vol change.

    Same for calls and puts.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility

    Returns:
        Ultima value
    """
    if time_to_expiry < _MIN_TIME or volatility < _MIN_VOLATILITY:
        return 0.0

    d1, d2 = _compute_d1_d2(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    exp_div = math.exp(-dividend_yield * time_to_expiry)

    # Vega per unit vol
    vega_unit = spot * exp_div * sqrt_t * _norm_pdf(d1)

    d1d2 = d1 * d2
    term = d1d2 * (1.0 - d1d2) + d1 * d1 + d2 * d2

    return -vega_unit * term / (volatility * volatility)


# =============================================================================
# Combined Greeks Calculation
# =============================================================================


def compute_all_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> GreeksResult:
    """
    Compute all 12 Greeks for an option.

    This is the main entry point for Greeks calculation. It computes
    all first, second, and third-order Greeks in a single call.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate (annualized)
        dividend_yield: Continuous dividend yield (annualized)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        is_call: True for call option, False for put option

    Returns:
        GreeksResult with all 12 Greeks

    Raises:
        GreeksCalculationError: If inputs are invalid
    """
    # Validate inputs
    if spot <= 0:
        raise GreeksCalculationError(f"Spot must be positive, got {spot}")
    if strike <= 0:
        raise GreeksCalculationError(f"Strike must be positive, got {strike}")
    if time_to_expiry < 0:
        raise GreeksCalculationError(f"Time to expiry cannot be negative, got {time_to_expiry}")
    if volatility < 0:
        raise GreeksCalculationError(f"Volatility cannot be negative, got {volatility}")

    # Handle expired option
    if time_to_expiry < _MIN_TIME:
        return GreeksResult.zero()

    # Handle zero volatility edge case
    if volatility < _MIN_VOLATILITY:
        volatility = _MIN_VOLATILITY

    timestamp_ns = time.time_ns()

    # Compute all Greeks
    delta = compute_delta(spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call)
    gamma = compute_gamma(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    theta = compute_theta(
        spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call, per_day=True
    )
    vega = compute_vega(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    rho = compute_rho(spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call)

    vanna = compute_vanna(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    volga = compute_volga(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    charm = compute_charm(
        spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call, per_day=True
    )

    speed = compute_speed(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    color = compute_color(
        spot, strike, time_to_expiry, rate, dividend_yield, volatility, per_day=True
    )
    zomma = compute_zomma(spot, strike, time_to_expiry, rate, dividend_yield, volatility)
    ultima = compute_ultima(spot, strike, time_to_expiry, rate, dividend_yield, volatility)

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        vanna=vanna,
        volga=volga,
        charm=charm,
        speed=speed,
        color=color,
        zomma=zomma,
        ultima=ultima,
        timestamp_ns=timestamp_ns,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        rate=rate,
        dividend_yield=dividend_yield,
    )


def compute_greeks_for_contract(
    contract: OptionsContractSpec,
    spot: float,
    volatility: float,
    rate: float,
    dividend_yield: float = 0.0,
    valuation_date: Optional["date"] = None,
) -> GreeksResult:
    """
    Compute all Greeks for an options contract.

    Convenience function that extracts parameters from OptionsContractSpec.

    Args:
        contract: Options contract specification
        spot: Current underlying price
        volatility: Annualized implied volatility
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        valuation_date: Date for valuation (default: today)

    Returns:
        GreeksResult with all 12 Greeks
    """
    time_to_expiry = contract.time_to_expiry(valuation_date)
    is_call = contract.is_call

    return compute_all_greeks(
        spot=spot,
        strike=contract.strike_float,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
    )


# =============================================================================
# Greeks Validation / Numerical Differentiation
# =============================================================================


def validate_greeks_numerically(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    h_spot: float = 0.01,  # Bump size for spot (as fraction)
    h_vol: float = 0.001,  # Bump size for vol (absolute)
    h_time: float = 1 / 365,  # Bump size for time (1 day)
    h_rate: float = 0.0001,  # Bump size for rate (1 bp)
    tolerance: float = 0.01,  # Relative tolerance (1%)
) -> dict:
    """
    Validate analytical Greeks against numerical differentiation.

    Uses central difference approximation to verify analytical formulas.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call option
        h_spot: Bump size for spot (as fraction)
        h_vol: Bump size for volatility (absolute)
        h_time: Bump size for time (in years)
        h_rate: Bump size for rate (absolute)
        tolerance: Maximum acceptable relative error

    Returns:
        Dictionary with validation results for each Greek
    """
    from impl_pricing import black_scholes_price

    results = {}

    # Compute analytical Greeks
    greeks = compute_all_greeks(
        spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call
    )

    # Helper for central difference
    def central_diff(f, x, h):
        return (f(x + h) - f(x - h)) / (2.0 * h)

    # Validate Delta
    def price_at_spot(s):
        return black_scholes_price(
            s, strike, time_to_expiry, rate, dividend_yield, volatility, is_call
        )

    delta_h = spot * h_spot
    delta_num = central_diff(price_at_spot, spot, delta_h)
    delta_err = abs(greeks.delta - delta_num) / max(abs(delta_num), 1e-10)
    results["delta"] = {
        "analytical": greeks.delta,
        "numerical": delta_num,
        "error": delta_err,
        "passed": delta_err < tolerance,
    }

    # Validate Gamma (second derivative)
    gamma_num = (
        price_at_spot(spot + delta_h) - 2 * price_at_spot(spot) + price_at_spot(spot - delta_h)
    ) / (delta_h**2)
    gamma_err = abs(greeks.gamma - gamma_num) / max(abs(gamma_num), 1e-10)
    results["gamma"] = {
        "analytical": greeks.gamma,
        "numerical": gamma_num,
        "error": gamma_err,
        "passed": gamma_err < tolerance,
    }

    # Validate Vega
    def price_at_vol(v):
        return black_scholes_price(spot, strike, time_to_expiry, rate, dividend_yield, v, is_call)

    # Vega analytical is per 0.01, numerical is per h_vol
    vega_num = central_diff(price_at_vol, volatility, h_vol) * 0.01
    vega_err = abs(greeks.vega - vega_num) / max(abs(vega_num), 1e-10)
    results["vega"] = {
        "analytical": greeks.vega,
        "numerical": vega_num,
        "error": vega_err,
        "passed": vega_err < tolerance,
    }

    # Validate Theta
    def price_at_time(t):
        if t < _MIN_TIME:
            t = _MIN_TIME
        return black_scholes_price(spot, strike, t, rate, dividend_yield, volatility, is_call)

    # Theta analytical is per day, need to convert
    theta_num = -central_diff(price_at_time, time_to_expiry, h_time) / 365.0
    theta_err = abs(greeks.theta - theta_num) / max(abs(theta_num), 1e-10)
    results["theta"] = {
        "analytical": greeks.theta,
        "numerical": theta_num,
        "error": theta_err,
        "passed": theta_err < tolerance,
    }

    # Validate Rho
    def price_at_rate(r):
        return black_scholes_price(
            spot, strike, time_to_expiry, r, dividend_yield, volatility, is_call
        )

    # Rho analytical is per 0.01
    rho_num = central_diff(price_at_rate, rate, h_rate) * 0.01
    rho_err = abs(greeks.rho - rho_num) / max(abs(rho_num), 1e-10)
    results["rho"] = {
        "analytical": greeks.rho,
        "numerical": rho_num,
        "error": rho_err,
        "passed": rho_err < tolerance,
    }

    return results


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # First-order Greeks
    "compute_delta",
    "compute_gamma",
    "compute_theta",
    "compute_vega",
    "compute_rho",
    # Second-order Greeks
    "compute_vanna",
    "compute_volga",
    "compute_charm",
    # Third-order Greeks
    "compute_speed",
    "compute_color",
    "compute_zomma",
    "compute_ultima",
    # Combined calculation
    "compute_all_greeks",
    "compute_greeks_for_contract",
    # Validation
    "validate_greeks_numerically",
]
