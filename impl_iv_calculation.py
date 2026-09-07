# -*- coding: utf-8 -*-
"""
impl_iv_calculation.py
Implied Volatility Calculation with Hybrid Solver.

Phase 1: Core Models & Data Structures

This module provides robust implied volatility (IV) calculation using a
hybrid approach combining Newton-Raphson for speed and Brent's method
for reliability.

Solver Strategy:
1. Initial guess using rational approximation (Corrado-Miller formula)
2. Newton-Raphson iteration with vega for fast convergence near solution
3. Automatic fallback to Brent's method if NR fails to converge
4. Bisection as final fallback for extreme cases

Features:
    - European IV via Black-Scholes inversion
    - American IV via binomial tree inversion
    - Vectorized batch IV calculation
    - Robust handling of edge cases (deep ITM/OTM)
    - Convergence diagnostics

References:
    - Corrado & Miller (1996): "A Note on a Simple, Accurate Formula"
    - Jaeckel (2015): "Let's Be Rational" (most accurate)
    - Manaster & Koehler (1982): "The Calculation of Implied Variances"
    - Brenner & Subrahmanyam (1988): "A Simple Approach"
"""

from __future__ import annotations

from datetime import date

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq, bisect

from core_options import (
    ExerciseStyle,
    IVResult,
    OptionsContractSpec,
    OptionType,
)
from core_errors import IVConvergenceError, PricingError
from impl_pricing import (
    black_scholes_price,
    leisen_reimer_price,
)
from impl_greeks import compute_vega


# =============================================================================
# Constants
# =============================================================================

# Volatility bounds
_MIN_IV = 0.0001  # 0.01%
_MAX_IV = 10.0  # 1000%
_DEFAULT_IV = 0.20  # 20% starting guess

# Convergence parameters
_DEFAULT_TOL = 1e-8  # Price tolerance
_DEFAULT_VOL_TOL = 1e-6  # Volatility tolerance
_MAX_ITERATIONS = 100
_NR_MAX_ITERATIONS = 50  # Max Newton-Raphson iterations

# Numerical constants
_MIN_TIME = 1e-10
_MIN_PRICE = 1e-10


# =============================================================================
# Initial Guess Methods
# =============================================================================


def _brenner_subrahmanyam_guess(
    spot: float,
    strike: float,
    time_to_expiry: float,
    price: float,
) -> float:
    """
    Brenner-Subrahmanyam (1988) ATM approximation.

    σ ≈ √(2π/T) × C/S

    Works best for ATM options.
    """
    if time_to_expiry < _MIN_TIME or spot < _MIN_PRICE:
        return _DEFAULT_IV

    guess = math.sqrt(2.0 * math.pi / time_to_expiry) * price / spot
    return max(_MIN_IV, min(guess, _MAX_IV))


def _corrado_miller_guess(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    price: float,
    is_call: bool,
) -> float:
    """
    Corrado-Miller (1996) rational approximation.

    More accurate than Brenner-Subrahmanyam across strike range.

    σ ≈ √(2π/T) × [C - (S-K)/2] / [S + K + √((S+K)² - 4×C²)] × 2
    """
    if time_to_expiry < _MIN_TIME:
        return _DEFAULT_IV

    # Forward price
    forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
    df = math.exp(-rate * time_to_expiry)

    # Undiscounted price
    c_undiscounted = price / df if df > 0 else price

    # For puts, convert to call via put-call parity
    if not is_call:
        # P = C - (F - K) → C = P + F - K
        c_undiscounted = c_undiscounted + forward - strike

    # Corrado-Miller formula
    x = forward - strike
    y = c_undiscounted - x / 2.0

    if y <= 0:
        return _DEFAULT_IV

    # Quadratic discriminant
    discriminant = (forward + strike) ** 2 - 4.0 * c_undiscounted * c_undiscounted

    if discriminant <= 0:
        # Use simpler formula
        return _brenner_subrahmanyam_guess(spot, strike, time_to_expiry, price)

    denom = (forward + strike + math.sqrt(discriminant)) / 2.0

    if denom <= 0:
        return _DEFAULT_IV

    sigma = math.sqrt(2.0 * math.pi / time_to_expiry) * y / denom

    return max(_MIN_IV, min(sigma, _MAX_IV))


def _initial_guess(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    price: float,
    is_call: bool,
) -> float:
    """
    Get best initial guess for IV solver.

    Uses Corrado-Miller with fallback to Brenner-Subrahmanyam.
    """
    try:
        guess = _corrado_miller_guess(
            spot, strike, time_to_expiry, rate, dividend_yield, price, is_call
        )
        if _MIN_IV < guess < _MAX_IV:
            return guess
    except (ValueError, ZeroDivisionError):
        pass

    try:
        guess = _brenner_subrahmanyam_guess(spot, strike, time_to_expiry, price)
        if _MIN_IV < guess < _MAX_IV:
            return guess
    except (ValueError, ZeroDivisionError):
        pass

    return _DEFAULT_IV


# =============================================================================
# Newton-Raphson Solver
# =============================================================================


def _newton_raphson_iv(
    target_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    initial_guess: float,
    tol: float = _DEFAULT_TOL,
    max_iterations: int = _NR_MAX_ITERATIONS,
) -> Tuple[float, int, bool]:
    """
    Newton-Raphson IV solver.

    Uses vega (derivative of price w.r.t. volatility) for quadratic convergence.

    σ_{n+1} = σ_n - (Price(σ_n) - Target) / Vega(σ_n)

    Args:
        target_price: Market option price to match
        spot: Underlying price
        strike: Strike price
        time_to_expiry: Time to expiry
        rate: Risk-free rate
        dividend_yield: Dividend yield
        is_call: True for call
        initial_guess: Starting volatility guess
        tol: Price tolerance
        max_iterations: Maximum iterations

    Returns:
        Tuple of (IV, iterations, converged)
    """
    sigma = initial_guess

    for iteration in range(max_iterations):
        # Compute price and vega
        try:
            price = black_scholes_price(
                spot, strike, time_to_expiry, rate, dividend_yield, sigma, is_call
            )
            vega = compute_vega(spot, strike, time_to_expiry, rate, dividend_yield, sigma)
        except (ValueError, PricingError):
            return sigma, iteration, False

        # Check convergence
        diff = price - target_price
        if abs(diff) < tol:
            return sigma, iteration + 1, True

        # Vega too small - can't update reliably
        if abs(vega) < 1e-15:
            return sigma, iteration + 1, False

        # Newton-Raphson update (vega is per 0.01, so scale)
        sigma_new = sigma - diff / (vega * 100.0)

        # Bound check
        sigma_new = max(_MIN_IV, min(sigma_new, _MAX_IV))

        # Check volatility convergence
        if abs(sigma_new - sigma) < _DEFAULT_VOL_TOL:
            return sigma_new, iteration + 1, True

        sigma = sigma_new

    return sigma, max_iterations, False


# =============================================================================
# Brent's Method Solver
# =============================================================================


def _brent_iv(
    target_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    tol: float = _DEFAULT_TOL,
    vol_lo: float = _MIN_IV,
    vol_hi: float = _MAX_IV,
) -> Tuple[float, int, bool]:
    """
    Brent's method IV solver.

    Uses scipy.optimize.brentq for guaranteed convergence (given bracketing).

    Args:
        target_price: Market option price
        spot: Underlying price
        strike: Strike price
        time_to_expiry: Time to expiry
        rate: Risk-free rate
        dividend_yield: Dividend yield
        is_call: True for call
        tol: Tolerance
        vol_lo: Lower volatility bound
        vol_hi: Upper volatility bound

    Returns:
        Tuple of (IV, iterations, converged)
    """

    def price_diff(sigma: float) -> float:
        try:
            return (
                black_scholes_price(
                    spot, strike, time_to_expiry, rate, dividend_yield, sigma, is_call
                )
                - target_price
            )
        except (ValueError, PricingError):
            return float("inf")

    # Check if bracketing is valid
    try:
        f_lo = price_diff(vol_lo)
        f_hi = price_diff(vol_hi)
    except Exception:
        return _DEFAULT_IV, 0, False

    # If not bracketed, try to find brackets
    if f_lo * f_hi > 0:
        # Both same sign - try to adjust bounds
        if f_lo > 0:
            # Price at low vol is too high - price below intrinsic
            return _MIN_IV, 0, False
        else:
            # Price at high vol is too low - no solution
            return _MAX_IV, 0, False

    try:
        result = brentq(
            price_diff, vol_lo, vol_hi, xtol=tol, maxiter=_MAX_ITERATIONS, full_output=True
        )
        sigma = result[0]
        info = result[1]
        iterations = info.iterations
        return sigma, iterations, True
    except ValueError:
        return _DEFAULT_IV, 0, False


# =============================================================================
# Bisection Fallback
# =============================================================================


def _bisection_iv(
    target_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    tol: float = _DEFAULT_TOL,
    vol_lo: float = _MIN_IV,
    vol_hi: float = _MAX_IV,
    max_iterations: int = _MAX_ITERATIONS,
) -> Tuple[float, int, bool]:
    """
    Bisection method as final fallback.

    Slow but guaranteed to converge if bracketed.
    """

    def price_diff(sigma: float) -> float:
        try:
            return (
                black_scholes_price(
                    spot, strike, time_to_expiry, rate, dividend_yield, sigma, is_call
                )
                - target_price
            )
        except (ValueError, PricingError):
            return float("inf")

    try:
        result = bisect(
            price_diff, vol_lo, vol_hi, xtol=tol, maxiter=max_iterations, full_output=True
        )
        sigma = result[0]
        info = result[1]
        iterations = info.iterations
        return sigma, iterations, True
    except ValueError:
        return _DEFAULT_IV, 0, False


# =============================================================================
# Hybrid IV Solver (Main)
# =============================================================================


def calculate_iv(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    tol: float = _DEFAULT_TOL,
    max_iterations: int = _MAX_ITERATIONS,
) -> IVResult:
    """
    Calculate implied volatility using hybrid solver.

    Strategy:
    1. Compute initial guess (Corrado-Miller)
    2. Try Newton-Raphson for fast convergence
    3. Fall back to Brent's method if NR fails
    4. Final fallback to bisection

    Args:
        market_price: Observed option market price
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        is_call: True for call, False for put
        tol: Price tolerance for convergence
        max_iterations: Maximum solver iterations

    Returns:
        IVResult with IV and convergence metadata

    Raises:
        IVConvergenceError: If all methods fail to converge
    """
    timestamp_ns = time.time_ns()

    # Input validation
    if spot <= 0:
        raise IVConvergenceError("Spot must be positive", 0, 0.0, market_price, 0.0)
    if strike <= 0:
        raise IVConvergenceError("Strike must be positive", 0, 0.0, market_price, 0.0)
    if time_to_expiry < 0:
        raise IVConvergenceError("Time to expiry cannot be negative", 0, 0.0, market_price, 0.0)
    if market_price < 0:
        raise IVConvergenceError("Market price cannot be negative", 0, 0.0, market_price, 0.0)

    # Check intrinsic value bounds
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    df = math.exp(-rate * time_to_expiry)
    intrinsic_pv = intrinsic * df

    if market_price < intrinsic_pv - tol:
        # Price below intrinsic - arbitrage violation
        return IVResult(
            implied_volatility=0.0,
            converged=False,
            iterations=0,
            error=intrinsic_pv - market_price,
            method="intrinsic_violation",
            model_price=intrinsic_pv,
        )

    # Handle at-expiry case
    if time_to_expiry < _MIN_TIME:
        if abs(market_price - intrinsic) < tol:
            return IVResult(
                implied_volatility=0.0,
                converged=True,
                iterations=0,
                error=0.0,
                method="at_expiry",
                model_price=intrinsic,
                market_price=market_price,
                price_error=abs(market_price - intrinsic),
            )
        raise IVConvergenceError(
            "At expiry with non-intrinsic price", 0, 0.0, market_price, intrinsic
        )

    # Get initial guess
    initial = _initial_guess(
        spot, strike, time_to_expiry, rate, dividend_yield, market_price, is_call
    )

    # Try Newton-Raphson first
    iv, iterations, converged = _newton_raphson_iv(
        target_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        initial_guess=initial,
        tol=tol,
        max_iterations=min(max_iterations // 2, _NR_MAX_ITERATIONS),
    )

    method = "newton_raphson"

    if converged:
        # Verify convergence
        model_price = black_scholes_price(
            spot, strike, time_to_expiry, rate, dividend_yield, iv, is_call
        )
        price_error = abs(model_price - market_price)

        if price_error < tol:
            return IVResult(
                implied_volatility=iv,
                converged=True,
                iterations=iterations,
                error=price_error,
                method=method,
                model_price=model_price,
                market_price=market_price,
                price_error=price_error,
            )

    # Fall back to Brent's method
    iv_brent, iterations_brent, converged_brent = _brent_iv(
        target_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        tol=tol,
    )

    if converged_brent:
        model_price = black_scholes_price(
            spot, strike, time_to_expiry, rate, dividend_yield, iv_brent, is_call
        )
        price_error = abs(model_price - market_price)

        return IVResult(
            implied_volatility=iv_brent,
            converged=True,
            iterations=iterations + iterations_brent,
            error=price_error,
            method="brent",
            model_price=model_price,
            market_price=market_price,
            price_error=price_error,
        )

    # Final fallback: bisection
    iv_bisect, iterations_bisect, converged_bisect = _bisection_iv(
        target_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        is_call=is_call,
        tol=tol,
        max_iterations=max_iterations,
    )

    if converged_bisect:
        model_price = black_scholes_price(
            spot, strike, time_to_expiry, rate, dividend_yield, iv_bisect, is_call
        )
        price_error = abs(model_price - market_price)

        return IVResult(
            implied_volatility=iv_bisect,
            converged=True,
            iterations=iterations + iterations_brent + iterations_bisect,
            error=price_error,
            method="bisection",
            model_price=model_price,
            market_price=market_price,
            price_error=price_error,
        )

    # All methods failed
    best_iv = iv_bisect if converged_bisect else (iv_brent if converged_brent else iv)
    model_price = black_scholes_price(
        spot, strike, time_to_expiry, rate, dividend_yield, best_iv, is_call
    )

    raise IVConvergenceError(
        message=f"IV solver failed to converge after {iterations + iterations_brent + iterations_bisect} iterations",
        iterations=iterations + iterations_brent + iterations_bisect,
        last_estimate=best_iv,
        target_price=market_price,
        model_price=model_price,
    )


def calculate_iv_american(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    tol: float = _DEFAULT_TOL,
    max_iterations: int = _MAX_ITERATIONS,
    n_tree_steps: int = 201,
) -> IVResult:
    """
    Calculate implied volatility for American options.

    Uses Leisen-Reimer binomial tree for pricing, with bisection search
    for the IV.

    Args:
        market_price: Observed market price
        spot: Underlying price
        strike: Strike price
        time_to_expiry: Time to expiry
        rate: Risk-free rate
        dividend_yield: Dividend yield
        is_call: True for call
        tol: Price tolerance
        max_iterations: Max iterations
        n_tree_steps: Binomial tree steps

    Returns:
        IVResult with IV and metadata

    Raises:
        IVConvergenceError: If solver fails
    """
    timestamp_ns = time.time_ns()

    # For American calls without dividends, use European solver
    if is_call and dividend_yield <= 0:
        return calculate_iv(
            market_price,
            spot,
            strike,
            time_to_expiry,
            rate,
            dividend_yield,
            is_call,
            tol,
            max_iterations,
        )

    def price_diff(sigma: float) -> float:
        try:
            return (
                leisen_reimer_price(
                    spot,
                    strike,
                    time_to_expiry,
                    rate,
                    dividend_yield,
                    sigma,
                    is_call,
                    is_american=True,
                    n_steps=n_tree_steps,
                )
                - market_price
            )
        except (ValueError, PricingError):
            return float("inf")

    # Use bisection for American options (more robust)
    try:
        # Find brackets
        vol_lo, vol_hi = _MIN_IV, _MAX_IV

        f_lo = price_diff(vol_lo)
        f_hi = price_diff(vol_hi)

        if f_lo * f_hi > 0:
            raise IVConvergenceError(
                "Cannot bracket IV solution", 0, _DEFAULT_IV, market_price, 0.0
            )

        result = brentq(
            price_diff, vol_lo, vol_hi, xtol=tol, maxiter=max_iterations, full_output=True
        )
        iv = result[0]
        iterations = result[1].iterations

        model_price = leisen_reimer_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            dividend_yield,
            iv,
            is_call,
            is_american=True,
            n_steps=n_tree_steps,
        )
        price_error = abs(model_price - market_price)

        return IVResult(
            implied_volatility=iv,
            converged=True,
            iterations=iterations,
            error=price_error,
            method="american_brent",
            model_price=model_price,
            market_price=market_price,
            price_error=price_error,
        )

    except ValueError as e:
        raise IVConvergenceError(
            f"American IV solver failed: {e}", max_iterations, _DEFAULT_IV, market_price, 0.0
        )


# =============================================================================
# Vectorized IV Calculation
# =============================================================================


def calculate_iv_batch(
    market_prices: np.ndarray,
    spots: np.ndarray,
    strikes: np.ndarray,
    times_to_expiry: np.ndarray,
    rates: np.ndarray,
    dividend_yields: np.ndarray,
    is_calls: np.ndarray,
    tol: float = _DEFAULT_TOL,
    max_iterations: int = _MAX_ITERATIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate implied volatility for a batch of options.

    Processes each option sequentially with individual error handling.

    Args:
        market_prices: Array of market prices
        spots: Array of underlying prices
        strikes: Array of strikes
        times_to_expiry: Array of times to expiry
        rates: Array of risk-free rates
        dividend_yields: Array of dividend yields
        is_calls: Boolean array
        tol: Price tolerance
        max_iterations: Max iterations per option

    Returns:
        Tuple of (IVs, converged_flags, iterations)
    """
    n = len(market_prices)

    ivs = np.zeros(n, dtype=np.float64)
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int32)

    for i in range(n):
        try:
            result = calculate_iv(
                market_price=float(market_prices[i]),
                spot=float(spots[i]),
                strike=float(strikes[i]),
                time_to_expiry=float(times_to_expiry[i]),
                rate=float(rates[i]),
                dividend_yield=float(dividend_yields[i]),
                is_call=bool(is_calls[i]),
                tol=tol,
                max_iterations=max_iterations,
            )
            ivs[i] = result.implied_volatility
            converged[i] = result.converged
            iterations[i] = result.iterations
        except IVConvergenceError as e:
            ivs[i] = e.last_estimate
            converged[i] = False
            iterations[i] = e.iterations

    return ivs, converged, iterations


def calculate_iv_for_chain(
    contracts: list,
    market_prices: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    valuation_date: Optional["date"] = None,
    tol: float = _DEFAULT_TOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate IV for an option chain.

    Convenience function for processing option chains.

    Args:
        contracts: List of OptionsContractSpec
        market_prices: Array of market prices (same order as contracts)
        spot: Current underlying price
        rate: Risk-free rate
        dividend_yield: Dividend yield
        valuation_date: Valuation date
        tol: Tolerance

    Returns:
        Tuple of (IVs, converged_flags, iterations)
    """
    n = len(contracts)

    spots = np.full(n, spot)
    strikes = np.array([c.strike_float for c in contracts])
    times = np.array([c.time_to_expiry(valuation_date) for c in contracts])
    rates = np.full(n, rate)
    divs = np.full(n, dividend_yield)
    is_calls = np.array([c.is_call for c in contracts])

    return calculate_iv_batch(
        market_prices=market_prices,
        spots=spots,
        strikes=strikes,
        times_to_expiry=times,
        rates=rates,
        dividend_yields=divs,
        is_calls=is_calls,
        tol=tol,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def iv_from_delta(
    delta: float,
    spot: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    is_call: bool,
    strike_guess: Optional[float] = None,
    vol_guess: float = 0.20,
) -> Tuple[float, float]:
    """
    Find strike and IV for a given delta.

    Iteratively solves for the strike that produces the target delta,
    then returns the corresponding IV.

    Useful for delta-based quoting (25-delta puts, etc.).

    Args:
        delta: Target delta (e.g., 0.25 for 25-delta call)
        spot: Underlying price
        time_to_expiry: Time to expiry
        rate: Risk-free rate
        dividend_yield: Dividend yield
        is_call: True for call
        strike_guess: Initial strike guess (default: ATM)
        vol_guess: Volatility guess

    Returns:
        Tuple of (strike, implied_vol)
    """
    from impl_greeks import compute_delta

    if strike_guess is None:
        # Start with ATM
        strike_guess = spot * math.exp((rate - dividend_yield) * time_to_expiry)

    # Adjust delta sign for puts
    target_delta = abs(delta) if is_call else -abs(delta)

    def delta_diff(k: float) -> float:
        computed = compute_delta(spot, k, time_to_expiry, rate, dividend_yield, vol_guess, is_call)
        return computed - target_delta

    # Find strike using bisection
    k_lo = spot * 0.5
    k_hi = spot * 2.0

    try:
        strike = brentq(delta_diff, k_lo, k_hi, xtol=0.01)
    except ValueError:
        # Use guess if bisection fails
        strike = strike_guess

    return strike, vol_guess


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main IV calculation
    "calculate_iv",
    "calculate_iv_american",
    # Batch processing
    "calculate_iv_batch",
    "calculate_iv_for_chain",
    # Utilities
    "iv_from_delta",
]
