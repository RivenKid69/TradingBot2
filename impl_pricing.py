# -*- coding: utf-8 -*-
"""
impl_pricing.py
Options pricing models: Black-Scholes-Merton, Leisen-Reimer Binomial, Jump-Diffusion.

Phase 1: Core Models & Data Structures

This module provides multiple pricing models for options valuation:

1. Black-Scholes-Merton (European options)
   - Closed-form solution for European calls/puts
   - Supports continuous dividend yield
   - Standard model for liquid European options

2. Leisen-Reimer Binomial Tree (American options)
   - Second-order convergence for faster accuracy
   - Handles early exercise premium
   - Optimal for American options pricing

3. Merton Jump-Diffusion (Fat-tailed distributions)
   - Adds jump component to Black-Scholes
   - Captures sudden price movements
   - Better fit for smile/skew in short-term options

4. Variance Swap Replication
   - Model-free variance pricing
   - Uses discrete strip of options
   - Log-contract replication approach

References:
    - Black & Scholes (1973): "The Pricing of Options"
    - Merton (1976): "Option Pricing When Underlying Stock Returns Are Discontinuous"
    - Leisen & Reimer (1996): "Binomial Models for Option Valuation"
    - Cox, Ross & Rubinstein (1979): "Option Pricing: A Simplified Approach"
    - Demeterfi et al. (1999): "A Guide to Volatility and Variance Swaps"
"""

from __future__ import annotations

from datetime import date

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.special import factorial

from core_options import (
    ExerciseStyle,
    GreeksResult,
    JumpParams,
    OptionsContractSpec,
    OptionType,
    PricingResult,
    VarianceSwapQuote,
)
from core_errors import PricingError
from impl_greeks import compute_all_greeks


# =============================================================================
# Constants
# =============================================================================

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

# Numerical tolerances
_MIN_TIME = 1e-10
_MIN_VOLATILITY = 1e-10
_MIN_SPOT = 1e-10
_MIN_STRIKE = 1e-10

# Default pricing parameters
_DEFAULT_BINOMIAL_STEPS = 201  # Odd number for Leisen-Reimer
_MAX_BINOMIAL_STEPS = 10001
_JUMP_DIFFUSION_MAX_TERMS = 50  # Poisson truncation


# =============================================================================
# Normal Distribution Functions
# =============================================================================


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


# =============================================================================
# Black-Scholes-Merton Model (European Options)
# =============================================================================


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> float:
    """
    Black-Scholes-Merton option pricing for European options.

    The classic closed-form solution for European option pricing.

    Call: C = S × e^(-qT) × N(d1) - K × e^(-rT) × N(d2)
    Put:  P = K × e^(-rT) × N(-d2) - S × e^(-qT) × N(-d1)

    where:
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate (annualized)
        dividend_yield: Continuous dividend yield (annualized)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        is_call: True for call, False for put

    Returns:
        Option price

    Raises:
        PricingError: If inputs are invalid
    """
    # Validate inputs
    if spot <= 0:
        raise PricingError(f"Spot must be positive, got {spot}")
    if strike <= 0:
        raise PricingError(f"Strike must be positive, got {strike}")
    if time_to_expiry < 0:
        raise PricingError(f"Time to expiry cannot be negative, got {time_to_expiry}")
    if volatility < 0:
        raise PricingError(f"Volatility cannot be negative, got {volatility}")

    # Handle expiration
    if time_to_expiry < _MIN_TIME:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic

    # Handle zero volatility
    if volatility < _MIN_VOLATILITY:
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        df = math.exp(-rate * time_to_expiry)
        if is_call:
            return df * max(forward - strike, 0.0)
        else:
            return df * max(strike - forward, 0.0)

    # Compute d1, d2
    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t

    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # Discount factors
    exp_div = math.exp(-dividend_yield * time_to_expiry)
    exp_rate = math.exp(-rate * time_to_expiry)

    if is_call:
        price = spot * exp_div * _norm_cdf(d1) - strike * exp_rate * _norm_cdf(d2)
    else:
        price = strike * exp_rate * _norm_cdf(-d2) - spot * exp_div * _norm_cdf(-d1)

    return max(price, 0.0)  # Ensure non-negative


def black_scholes_price_vec(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    volatility: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Black-Scholes-Merton pricing.

    All inputs should be NumPy arrays of the same shape.

    Returns:
        Array of option prices
    """
    from scipy.special import erf

    # Apply minimums
    time_to_expiry = np.maximum(time_to_expiry, _MIN_TIME)
    volatility = np.maximum(volatility, _MIN_VOLATILITY)
    spot = np.maximum(spot, _MIN_SPOT)
    strike = np.maximum(strike, _MIN_STRIKE)

    sqrt_t = np.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t

    d1 = (
        np.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # Normal CDF vectorized
    n_d1 = 0.5 * (1.0 + erf(d1 * _INV_SQRT_2))
    n_d2 = 0.5 * (1.0 + erf(d2 * _INV_SQRT_2))
    n_neg_d1 = 0.5 * (1.0 + erf(-d1 * _INV_SQRT_2))
    n_neg_d2 = 0.5 * (1.0 + erf(-d2 * _INV_SQRT_2))

    exp_div = np.exp(-dividend_yield * time_to_expiry)
    exp_rate = np.exp(-rate * time_to_expiry)

    call_price = spot * exp_div * n_d1 - strike * exp_rate * n_d2
    put_price = strike * exp_rate * n_neg_d2 - spot * exp_div * n_neg_d1

    price = np.where(is_call, call_price, put_price)
    return np.maximum(price, 0.0)


# =============================================================================
# Leisen-Reimer Binomial Tree (American Options)
# =============================================================================


def _leisen_reimer_inversion(z: float, n: int) -> float:
    """
    Peizer-Pratt inversion formula for Leisen-Reimer tree.

    This provides second-order convergence for binomial trees.

    h(z) = 0.5 + sign(z) × 0.5 × √(1 - exp(-((z/(n+1/3+0.1/(n+1)))²) × (n+1/6)))

    Args:
        z: The d1 or d2 value
        n: Number of steps

    Returns:
        Probability value for the tree
    """
    if abs(z) < 1e-10:
        return 0.5

    n_plus = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    factor = n + 1.0 / 6.0

    arg = -((z / n_plus) ** 2) * factor
    inner = 1.0 - math.exp(arg)

    if inner < 0:
        inner = 0.0

    result = 0.5 + (1.0 if z > 0 else -1.0) * 0.5 * math.sqrt(inner)
    return max(0.0, min(1.0, result))


def leisen_reimer_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    is_american: bool = True,
    n_steps: int = _DEFAULT_BINOMIAL_STEPS,
) -> float:
    """
    Leisen-Reimer binomial tree option pricing.

    Second-order convergence binomial model, optimal for American options.

    The Leisen-Reimer tree uses the Peizer-Pratt inversion to determine
    tree probabilities, providing faster convergence than CRR trees.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        is_american: True for American style, False for European
        n_steps: Number of tree steps (must be odd)

    Returns:
        Option price

    Raises:
        PricingError: If inputs are invalid

    References:
        Leisen & Reimer (1996): "Binomial Models for Option Valuation"
    """
    # Validate inputs
    if spot <= 0:
        raise PricingError(f"Spot must be positive, got {spot}")
    if strike <= 0:
        raise PricingError(f"Strike must be positive, got {strike}")
    if time_to_expiry < 0:
        raise PricingError(f"Time to expiry cannot be negative, got {time_to_expiry}")
    if volatility < 0:
        raise PricingError(f"Volatility cannot be negative, got {volatility}")

    # Ensure odd number of steps
    if n_steps % 2 == 0:
        n_steps += 1
    n_steps = max(3, min(n_steps, _MAX_BINOMIAL_STEPS))

    # Handle expiration
    if time_to_expiry < _MIN_TIME:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic

    # Handle zero volatility - use forward price
    if volatility < _MIN_VOLATILITY:
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        df = math.exp(-rate * time_to_expiry)
        if is_call:
            return df * max(forward - strike, 0.0)
        else:
            return df * max(strike - forward, 0.0)

    # Compute d1, d2
    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t

    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # Leisen-Reimer probabilities
    p_d1 = _leisen_reimer_inversion(d1, n_steps)
    p_d2 = _leisen_reimer_inversion(d2, n_steps)

    # Handle edge cases where probabilities are extreme (fallback to Black-Scholes)
    if p_d2 <= 1e-10 or p_d2 >= 1.0 - 1e-10 or p_d1 <= 1e-10 or p_d1 >= 1.0 - 1e-10:
        # Fall back to Black-Scholes for European or CRR for American
        if is_american:
            return crr_binomial_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                is_call=is_call,
                is_american=is_american,
                n_steps=n_steps,
            )
        else:
            return black_scholes_price(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                is_call=is_call,
            )

    # Tree parameters
    dt = time_to_expiry / n_steps
    drift = math.exp((rate - dividend_yield) * dt)
    discount = math.exp(-rate * dt)

    # Up and down factors
    u = drift * p_d1 / p_d2
    d = (drift - p_d2 * u) / (1.0 - p_d2)

    # Risk-neutral probability (use p_d2)
    p = p_d2

    # Build terminal payoffs
    option_values = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        final_spot = spot * (u ** (n_steps - j)) * (d**j)
        if is_call:
            option_values[j] = max(final_spot - strike, 0.0)
        else:
            option_values[j] = max(strike - final_spot, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option_values[j] + (1.0 - p) * option_values[j + 1])

            if is_american:
                current_spot = spot * (u ** (i - j)) * (d**j)
                if is_call:
                    intrinsic = max(current_spot - strike, 0.0)
                else:
                    intrinsic = max(strike - current_spot, 0.0)
                option_values[j] = max(continuation, intrinsic)
            else:
                option_values[j] = continuation

    return max(option_values[0], 0.0)


def crr_binomial_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    is_american: bool = True,
    n_steps: int = 200,
) -> float:
    """
    Cox-Ross-Rubinstein binomial tree option pricing.

    Classic binomial model with first-order convergence.

    u = e^(σ√Δt)
    d = 1/u
    p = (e^((r-q)Δt) - d) / (u - d)

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        is_american: True for American style
        n_steps: Number of tree steps

    Returns:
        Option price

    References:
        Cox, Ross & Rubinstein (1979): "Option Pricing"
    """
    # Validate inputs
    if spot <= 0:
        raise PricingError(f"Spot must be positive, got {spot}")
    if strike <= 0:
        raise PricingError(f"Strike must be positive, got {strike}")
    if time_to_expiry < 0:
        raise PricingError(f"Time to expiry cannot be negative, got {time_to_expiry}")
    if volatility < 0:
        raise PricingError(f"Volatility cannot be negative, got {volatility}")

    n_steps = max(1, min(n_steps, _MAX_BINOMIAL_STEPS))

    # Handle expiration
    if time_to_expiry < _MIN_TIME:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic

    # Handle zero volatility
    if volatility < _MIN_VOLATILITY:
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        df = math.exp(-rate * time_to_expiry)
        if is_call:
            return df * max(forward - strike, 0.0)
        else:
            return df * max(strike - forward, 0.0)

    # CRR parameters
    dt = time_to_expiry / n_steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u

    # CRITICAL FIX: Prevent division by zero if u - d is extremely small
    # This can occur if volatility is very small (near _MIN_VOLATILITY)
    # and time step dt is also small, leading to u ≈ d ≈ 1.0.
    if u - d < 1e-12:
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        df = math.exp(-rate * time_to_expiry)
        if is_call:
            return df * max(forward - strike, 0.0)
        else:
            return df * max(strike - forward, 0.0)

    drift = math.exp((rate - dividend_yield) * dt)
    p = (drift - d) / (u - d)
    discount = math.exp(-rate * dt)

    # Ensure valid probability
    if p < 0.0 or p > 1.0:
        # Fall back to adjusted parameters
        p = max(0.0001, min(0.9999, p))

    # Build terminal payoffs
    option_values = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        final_spot = spot * (u ** (n_steps - j)) * (d**j)
        if is_call:
            option_values[j] = max(final_spot - strike, 0.0)
        else:
            option_values[j] = max(strike - final_spot, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option_values[j] + (1.0 - p) * option_values[j + 1])

            if is_american:
                current_spot = spot * (u ** (i - j)) * (d**j)
                if is_call:
                    intrinsic = max(current_spot - strike, 0.0)
                else:
                    intrinsic = max(strike - current_spot, 0.0)
                option_values[j] = max(continuation, intrinsic)
            else:
                option_values[j] = continuation

    return max(option_values[0], 0.0)


# =============================================================================
# Merton Jump-Diffusion Model
# =============================================================================


def merton_jump_diffusion_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    jump_params: JumpParams,
    max_terms: int = _JUMP_DIFFUSION_MAX_TERMS,
) -> float:
    """
    Merton (1976) jump-diffusion option pricing.

    Extends Black-Scholes with a compound Poisson jump process to capture
    sudden large price movements (fat tails).

    The price is a sum of Black-Scholes prices weighted by Poisson probabilities:

    V = Σ (e^(-λ'T) × (λ'T)^n / n!) × BSM(S, K, T, r_n, q, σ_n)

    where:
        λ' = λ(1 + m) is the risk-neutral jump intensity
        r_n = r - λm + n×ln(1+m)/T
        σ_n² = σ² + n×δ²/T
        m = E[Y-1] = exp(μ_J + δ²/2) - 1 is the mean jump size

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Diffusion volatility (excluding jumps)
        is_call: True for call, False for put
        jump_params: Jump process parameters
        max_terms: Maximum Poisson terms to sum

    Returns:
        Option price

    Raises:
        PricingError: If inputs are invalid

    References:
        Merton (1976): "Option Pricing When Underlying Stock Returns"
    """
    # Validate inputs
    if spot <= 0:
        raise PricingError(f"Spot must be positive, got {spot}")
    if strike <= 0:
        raise PricingError(f"Strike must be positive, got {strike}")
    if time_to_expiry < 0:
        raise PricingError(f"Time to expiry cannot be negative, got {time_to_expiry}")
    if volatility < 0:
        raise PricingError(f"Volatility cannot be negative, got {volatility}")

    # Handle expiration
    if time_to_expiry < _MIN_TIME:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return intrinsic

    # Extract jump parameters
    jump_intensity = jump_params.lambda_intensity  # λ
    jump_mean = jump_params.mu_jump  # μ_J (log)
    jump_vol = jump_params.sigma_jump  # δ

    # If no jumps, reduce to Black-Scholes
    if jump_intensity < 1e-10:
        return black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
        )

    # Mean relative jump size: E[Y] - 1 where Y = e^J
    # m = exp(μ_J + δ²/2) - 1
    m = math.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0

    # Risk-neutral jump intensity
    lambda_prime = jump_intensity * (1.0 + m)

    # If risk-neutral intensity is effectively zero, reduce to Black-Scholes
    if lambda_prime < 1e-10:
        return black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
        )

    # Sum over Poisson terms
    total_price = 0.0

    for n in range(max_terms):
        # Poisson weight
        if n == 0:
            poisson_weight = math.exp(-lambda_prime * time_to_expiry)
        else:
            log_weight = -lambda_prime * time_to_expiry + n * math.log(
                max(lambda_prime * time_to_expiry, 1e-300)
            )
            # Use log of factorial for numerical stability
            log_factorial = sum(math.log(i) for i in range(1, n + 1))
            log_weight -= log_factorial
            poisson_weight = math.exp(log_weight)

        if poisson_weight < 1e-20:
            break  # Negligible contribution

        # Adjusted parameters for n jumps
        if time_to_expiry > _MIN_TIME:
            r_n = rate - jump_intensity * m + n * math.log(1.0 + m) / time_to_expiry
            sigma_n_sq = volatility * volatility + n * jump_vol * jump_vol / time_to_expiry
            sigma_n = math.sqrt(max(sigma_n_sq, _MIN_VOLATILITY))
        else:
            r_n = rate
            sigma_n = volatility

        # Black-Scholes price with adjusted parameters
        bs_price = black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=r_n,
            dividend_yield=dividend_yield,
            volatility=sigma_n,
            is_call=is_call,
        )

        total_price += poisson_weight * bs_price

    return max(total_price, 0.0)


def merton_jump_diffusion_price_vec(
    spot: np.ndarray,
    strike: np.ndarray,
    time_to_expiry: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    volatility: np.ndarray,
    is_call: np.ndarray,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    max_terms: int = _JUMP_DIFFUSION_MAX_TERMS,
) -> np.ndarray:
    """
    Vectorized Merton jump-diffusion pricing.

    Uses the same jump parameters for all options (common for a single underlying).

    Returns:
        Array of option prices
    """
    n_options = len(spot)
    total_price = np.zeros(n_options)

    # Mean relative jump size
    m = math.exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0
    lambda_prime = jump_intensity * (1.0 + m)

    for n in range(max_terms):
        # Poisson weight (vector)
        if n == 0:
            poisson_weight = np.exp(-lambda_prime * time_to_expiry)
        else:
            log_weight = -lambda_prime * time_to_expiry + n * np.log(
                np.maximum(lambda_prime * time_to_expiry, 1e-300)
            )
            log_factorial = sum(math.log(i) for i in range(1, n + 1))
            log_weight -= log_factorial
            poisson_weight = np.exp(log_weight)

        if np.all(poisson_weight < 1e-20):
            break

        # Adjusted parameters
        safe_time = np.maximum(time_to_expiry, _MIN_TIME)
        r_n = rate - jump_intensity * m + n * np.log(1.0 + m) / safe_time
        sigma_n_sq = volatility * volatility + n * jump_vol * jump_vol / safe_time
        sigma_n = np.sqrt(np.maximum(sigma_n_sq, _MIN_VOLATILITY))

        # Vectorized BS price
        bs_price = black_scholes_price_vec(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=r_n,
            dividend_yield=dividend_yield,
            volatility=sigma_n,
            is_call=is_call,
        )

        total_price += poisson_weight * bs_price

    return np.maximum(total_price, 0.0)


# =============================================================================
# Variance Swap Pricing
# =============================================================================


def variance_swap_strike(
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    call_strikes: np.ndarray,
    put_strikes: np.ndarray,
    forward: float,
    rate: float,
    time_to_expiry: float,
) -> float:
    """
    Calculate fair variance swap strike using option replication.

    The variance swap strike (K_var) is computed via the model-free
    replication approach:

    K²_var = (2/T) × e^(rT) × [ ∫_{0}^{F} P(K)/K² dK + ∫_{F}^{∞} C(K)/K² dK ]

    Discretized using the trapezoidal rule with available OTM options.

    Args:
        call_prices: OTM call prices (K > F)
        put_prices: OTM put prices (K < F)
        call_strikes: Call strike prices
        put_strikes: Put strike prices
        forward: Forward price
        rate: Risk-free rate
        time_to_expiry: Time to expiration

    Returns:
        Fair variance swap strike (annualized variance)

    References:
        Demeterfi et al. (1999): "A Guide to Volatility and Variance Swaps"
    """
    if time_to_expiry <= 0:
        return 0.0

    # Sort by strike
    put_order = np.argsort(put_strikes)
    call_order = np.argsort(call_strikes)

    put_strikes = put_strikes[put_order]
    put_prices = put_prices[put_order]
    call_strikes = call_strikes[call_order]
    call_prices = call_prices[call_order]

    # Filter OTM options and ensure strikes are positive to avoid division by zero
    put_mask = (put_strikes < forward) & (put_strikes > 0.0)
    call_mask = (call_strikes > forward) & (call_strikes > 0.0)

    otm_put_strikes = put_strikes[put_mask]
    otm_put_prices = put_prices[put_mask]
    otm_call_strikes = call_strikes[call_mask]
    otm_call_prices = call_prices[call_mask]

    # Integrate using trapezoidal rule
    put_integral = 0.0
    if len(otm_put_strikes) > 1:
        for i in range(len(otm_put_strikes) - 1):
            k1, k2 = otm_put_strikes[i], otm_put_strikes[i + 1]
            p1, p2 = otm_put_prices[i], otm_put_prices[i + 1]
            put_integral += 0.5 * (p1 / (k1 * k1) + p2 / (k2 * k2)) * (k2 - k1)

    call_integral = 0.0
    if len(otm_call_strikes) > 1:
        for i in range(len(otm_call_strikes) - 1):
            k1, k2 = otm_call_strikes[i], otm_call_strikes[i + 1]
            c1, c2 = otm_call_prices[i], otm_call_prices[i + 1]
            call_integral += 0.5 * (c1 / (k1 * k1) + c2 / (k2 * k2)) * (k2 - k1)

    # Fair variance
    exp_rt = math.exp(rate * time_to_expiry)
    variance = (2.0 / time_to_expiry) * exp_rt * (put_integral + call_integral)

    return max(variance, 0.0)


def variance_swap_value(
    quote: VarianceSwapQuote,
    realized_variance: float,
    current_time: float,
) -> float:
    """
    Mark-to-market value of a variance swap position.

    Value = Notional × (Realized_Var - Strike²) × (T_elapsed / T_total)
            + Notional × (Expected_Future_Var - Strike²) × (T_remaining / T_total)

    For simplicity, we assume expected future variance equals current strike.

    Args:
        quote: Variance swap quote with strike and notional
        realized_variance: Annualized realized variance to date
        current_time: Current time as fraction of swap tenor

    Returns:
        Mark-to-market value
    """
    if current_time >= 1.0:
        # At expiry, value is just the payoff
        return quote.vega_notional * (realized_variance - quote.strike_variance)

    if current_time <= 0.0:
        return 0.0

    # Accrued value
    accrued = quote.vega_notional * (realized_variance - quote.strike_variance) * current_time

    return accrued


def compute_variance_swap_value(
    realized_variance: float,
    variance_strike: float,
    notional: float,
    time_to_expiry: float,
    time_elapsed: float = 0.0,
    rate: float = 0.0,
) -> float:
    """
    Simple variance swap value calculation for convenience.

    Value = Notional × (Realized_Var - Strike) × discount

    Args:
        realized_variance: Annualized realized variance
        variance_strike: Fair variance strike (in variance terms, e.g., 0.04 for 20% vol)
        notional: Notional amount
        time_to_expiry: Time to expiry in years (remaining)
        time_elapsed: Time elapsed since inception (fraction of total tenor)
        rate: Risk-free rate for discounting

    Returns:
        Mark-to-market value of variance swap position
    """
    import math

    # At expiry or full elapsed time
    if time_elapsed >= 1.0 or time_to_expiry <= 0.0:
        return notional * (realized_variance - variance_strike)

    # Discount factor
    discount = math.exp(-rate * time_to_expiry)

    # Value = Notional × (Realized - Strike) × discount
    value = notional * (realized_variance - variance_strike) * discount

    return value


# =============================================================================
# Unified Pricing Function
# =============================================================================


def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    model: str = "black_scholes",
    jump_params: Optional[JumpParams] = None,
    n_steps: int = _DEFAULT_BINOMIAL_STEPS,
) -> PricingResult:
    """
    Unified option pricing with model selection.

    Main entry point for pricing that selects the appropriate model
    based on inputs.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        exercise_style: EUROPEAN or AMERICAN
        model: Pricing model ("black_scholes", "leisen_reimer", "crr", "merton_jd")
        jump_params: Jump parameters for Merton model
        n_steps: Number of steps for binomial trees

    Returns:
        PricingResult with price and metadata

    Raises:
        PricingError: If inputs or model are invalid
    """
    timestamp_ns = time.time_ns()

    # Validate model choice
    valid_models = ["black_scholes", "leisen_reimer", "crr", "merton_jd"]
    if model not in valid_models:
        raise PricingError(f"Unknown model: {model}. Valid: {valid_models}")

    # Select pricing function
    if model == "black_scholes":
        if exercise_style == ExerciseStyle.AMERICAN and is_call and dividend_yield == 0:
            # American call without dividends = European call
            price = black_scholes_price(
                spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call
            )
        elif exercise_style == ExerciseStyle.AMERICAN:
            # Use Leisen-Reimer for American options
            price = leisen_reimer_price(
                spot,
                strike,
                time_to_expiry,
                rate,
                dividend_yield,
                volatility,
                is_call,
                is_american=True,
                n_steps=n_steps,
            )
        else:
            price = black_scholes_price(
                spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call
            )

    elif model == "leisen_reimer":
        is_american = exercise_style == ExerciseStyle.AMERICAN
        price = leisen_reimer_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            dividend_yield,
            volatility,
            is_call,
            is_american=is_american,
            n_steps=n_steps,
        )

    elif model == "crr":
        is_american = exercise_style == ExerciseStyle.AMERICAN
        price = crr_binomial_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            dividend_yield,
            volatility,
            is_call,
            is_american=is_american,
            n_steps=n_steps,
        )

    elif model == "merton_jd":
        if jump_params is None:
            jump_params = JumpParams()  # Use defaults
        price = merton_jump_diffusion_price(
            spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call, jump_params
        )

    else:
        raise PricingError(f"Model not implemented: {model}")

    # Compute Greeks using the scalar Greeks function
    greeks = compute_all_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
    )

    # Compute timing for metadata
    computation_time_us = (time.time_ns() - timestamp_ns) // 1000

    return PricingResult(
        price=price,
        greeks=greeks,
        model=model,
        iterations=n_steps if model in ["leisen_reimer", "crr"] else 0,
        convergence_error=0.0,
        computation_time_us=computation_time_us,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
    )


def price_contract(
    contract: OptionsContractSpec,
    spot: float,
    volatility: float,
    rate: float,
    dividend_yield: float = 0.0,
    valuation_date: Optional["date"] = None,
    model: str = "auto",
    jump_params: Optional[JumpParams] = None,
    n_steps: int = _DEFAULT_BINOMIAL_STEPS,
) -> PricingResult:
    """
    Price an options contract.

    Convenience function that extracts parameters from contract spec.

    Args:
        contract: Options contract specification
        spot: Current underlying price
        volatility: Implied volatility
        rate: Risk-free rate
        dividend_yield: Dividend yield
        valuation_date: Valuation date (default: today)
        model: Pricing model ("auto", "black_scholes", "leisen_reimer", etc.)
        jump_params: Jump parameters for Merton model
        n_steps: Tree steps for binomial

    Returns:
        PricingResult with price and metadata
    """
    time_to_expiry = contract.time_to_expiry(valuation_date)
    is_call = contract.is_call

    # Auto-select model
    if model == "auto":
        if contract.exercise_style == ExerciseStyle.AMERICAN:
            model = "leisen_reimer"
        else:
            model = "black_scholes"

    return price_option(
        spot=spot,
        strike=contract.strike_float,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
        exercise_style=contract.exercise_style,
        model=model,
        jump_params=jump_params,
        n_steps=n_steps,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Black-Scholes
    "black_scholes_price",
    "black_scholes_price_vec",
    # Binomial Trees
    "leisen_reimer_price",
    "crr_binomial_price",
    # Jump-Diffusion
    "merton_jump_diffusion_price",
    "merton_jump_diffusion_price_vec",
    # Variance Swaps
    "variance_swap_strike",
    "variance_swap_value",
    # Unified Interface
    "price_option",
    "price_contract",
]
