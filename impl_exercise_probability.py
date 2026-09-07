# -*- coding: utf-8 -*-
"""
impl_exercise_probability.py
Early exercise probability using Longstaff-Schwartz Monte Carlo.

Phase 1: Core Models & Data Structures

This module implements the Longstaff-Schwartz (LS) algorithm for:
1. American option pricing via regression-based optimal stopping
2. Exercise boundary estimation
3. Early exercise probability computation
4. Exercise timing distribution

The LS algorithm uses least-squares regression to approximate the
continuation value at each time step, enabling efficient backward
induction for American option valuation.

Key Features:
- Multiple basis function choices (Laguerre, Hermite, power)
- Variance reduction (antithetic, control variate)
- Delta/Gamma computation via pathwise sensitivities
- Exercise boundary extraction
- Probability of exercise at each monitoring date

References:
    - Longstaff & Schwartz (2001): "Valuing American Options by Simulation"
    - Clement, Lamberton & Protter (2002): "Analysis of the LS Algorithm"
    - Glasserman & Yu (2004): "Number of Paths vs Basis Functions"
    - Rogers (2002): "Monte Carlo Valuation of American Options"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.polynomial import laguerre, hermite

from core_options import ExerciseStyle, OptionType
from core_errors import PricingError, ExerciseError


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_N_PATHS = 100_000
_DEFAULT_N_STEPS = 100
_DEFAULT_DEGREE = 3  # Polynomial degree for regression
_MIN_CONTINUATION_PATHS = 100  # Minimum ITM paths for regression

# Numerical constants
_MIN_TIME = 1e-10
_MIN_SPOT = 1e-10
_MIN_VOLATILITY = 1e-10


# =============================================================================
# Enums and Data Classes
# =============================================================================


class BasisFunctions(Enum):
    """Choice of basis functions for regression."""

    POWER = "power"  # 1, x, x^2, x^3, ...
    LAGUERRE = "laguerre"  # Laguerre polynomials (bounded on [0, inf))
    HERMITE = "hermite"  # Hermite polynomials (standard in finance)
    CHEBYSHEV = "chebyshev"  # Chebyshev polynomials


class VarianceReduction(Enum):
    """Variance reduction technique."""

    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    BOTH = "both"


@dataclass
class ExerciseBoundary:
    """
    Exercise boundary for American option.

    The boundary S*(t) defines the critical stock price at each time t:
    - Call: Exercise if S > S*(t)
    - Put: Exercise if S < S*(t)
    """

    times: np.ndarray  # Monitoring times (years from now)
    boundary_prices: np.ndarray  # S*(t) at each time
    exercise_probabilities: np.ndarray  # P(exercise at t | not exercised before)
    cumulative_exercise_prob: np.ndarray  # P(exercise by t)
    is_call: bool


@dataclass
class LSResult:
    """Result of Longstaff-Schwartz simulation."""

    # Option value
    price: float
    standard_error: float

    # Greeks (if computed)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    delta_se: Optional[float] = None
    gamma_se: Optional[float] = None

    # Exercise analysis
    exercise_boundary: Optional[ExerciseBoundary] = None
    expected_exercise_time: Optional[float] = None
    prob_early_exercise: float = 0.0
    prob_exercise_at_expiry: float = 0.0
    prob_expire_worthless: float = 0.0

    # Simulation details
    n_paths: int = 0
    n_steps: int = 0
    basis_functions: str = "power"
    variance_reduction: str = "none"
    computation_time_ms: float = 0.0

    # Regression diagnostics
    r_squared_avg: Optional[float] = None
    n_itm_paths_avg: Optional[float] = None


@dataclass
class ExerciseProbability:
    """
    Probability distribution of early exercise.

    For each monitoring date, gives the probability of:
    - Exercising at that date (conditional on not exercised before)
    - Having exercised by that date (cumulative)
    """

    times: np.ndarray
    conditional_probs: np.ndarray  # P(exercise at t | not exercised before)
    cumulative_probs: np.ndarray  # P(exercise by t)
    expected_time: float  # E[exercise time | exercised early]
    prob_early: float  # P(exercise before expiry)


# =============================================================================
# Basis Functions
# =============================================================================


def _evaluate_basis(
    x: np.ndarray,
    degree: int,
    basis_type: BasisFunctions,
) -> np.ndarray:
    """
    Evaluate basis functions at given points.

    Args:
        x: Points at which to evaluate (N,)
        degree: Maximum polynomial degree
        basis_type: Type of basis functions

    Returns:
        Design matrix (N, degree+1)
    """
    n = len(x)
    X = np.zeros((n, degree + 1))

    if basis_type == BasisFunctions.POWER:
        for i in range(degree + 1):
            X[:, i] = x**i

    elif basis_type == BasisFunctions.LAGUERRE:
        # Weighted Laguerre: L_n(x) * exp(-x/2)
        # Normalized to stock price
        x_normalized = x / np.mean(x) if np.mean(x) > 0 else x
        for i in range(degree + 1):
            coeffs = np.zeros(i + 1)
            coeffs[i] = 1
            X[:, i] = laguerre.lagval(x_normalized, coeffs) * np.exp(-x_normalized / 2)

    elif basis_type == BasisFunctions.HERMITE:
        # Probabilist's Hermite polynomials
        x_standardized = (x - np.mean(x)) / (np.std(x) + 1e-10)
        for i in range(degree + 1):
            coeffs = np.zeros(i + 1)
            coeffs[i] = 1
            X[:, i] = hermite.hermval(x_standardized, coeffs)

    elif basis_type == BasisFunctions.CHEBYSHEV:
        # Map to [-1, 1] for Chebyshev
        x_min, x_max = np.min(x), np.max(x)
        if x_max > x_min:
            x_mapped = 2 * (x - x_min) / (x_max - x_min) - 1
        else:
            x_mapped = np.zeros_like(x)
        for i in range(degree + 1):
            X[:, i] = np.cos(i * np.arccos(np.clip(x_mapped, -1, 1)))

    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    return X


# =============================================================================
# Path Generation
# =============================================================================


def _generate_paths(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_expiry: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GBM paths for Monte Carlo simulation.

    Args:
        spot: Initial stock price
        rate: Risk-free rate
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        time_to_expiry: Time to expiration (years)
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        antithetic: Use antithetic variates
        random_seed: Random seed for reproducibility

    Returns:
        (paths, times) where paths is (n_paths, n_steps+1) and times is (n_steps+1,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = time_to_expiry / n_steps
    times = np.linspace(0, time_to_expiry, n_steps + 1)

    # Drift and diffusion
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * math.sqrt(dt)

    # Generate random increments
    if antithetic:
        # Half paths, then mirror
        half_paths = n_paths // 2
        Z = np.random.standard_normal((half_paths, n_steps))
        Z = np.vstack([Z, -Z])
        if n_paths % 2 == 1:
            Z = np.vstack([Z, np.random.standard_normal((1, n_steps))])
    else:
        Z = np.random.standard_normal((n_paths, n_steps))

    # Build paths
    log_returns = drift + diffusion * Z
    log_prices = np.zeros((n_paths, n_steps + 1))
    log_prices[:, 0] = math.log(spot)
    log_prices[:, 1:] = np.cumsum(log_returns, axis=1) + math.log(spot)

    paths = np.exp(log_prices)

    return paths, times


# =============================================================================
# Longstaff-Schwartz Algorithm
# =============================================================================


def longstaff_schwartz(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    n_paths: int = _DEFAULT_N_PATHS,
    n_steps: int = _DEFAULT_N_STEPS,
    basis: BasisFunctions = BasisFunctions.LAGUERRE,
    degree: int = _DEFAULT_DEGREE,
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC,
    compute_greeks: bool = False,
    extract_boundary: bool = True,
    random_seed: Optional[int] = None,
) -> LSResult:
    """
    Price American option using Longstaff-Schwartz algorithm.

    Args:
        spot: Current stock price
        strike: Option strike price
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate (annualized)
        dividend_yield: Continuous dividend yield
        volatility: Annualized volatility
        is_call: True for call, False for put
        n_paths: Number of Monte Carlo paths
        n_steps: Number of exercise dates
        basis: Type of basis functions for regression
        degree: Polynomial degree for regression
        variance_reduction: Variance reduction technique
        compute_greeks: Compute delta and gamma
        extract_boundary: Extract exercise boundary
        random_seed: Random seed for reproducibility

    Returns:
        LSResult with price, Greeks, and exercise analysis
    """
    start_time = time.perf_counter()

    # Input validation
    if time_to_expiry <= _MIN_TIME:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return LSResult(
            price=intrinsic,
            standard_error=0.0,
            n_paths=0,
            n_steps=0,
            computation_time_ms=0.0,
        )

    if volatility <= _MIN_VOLATILITY:
        volatility = _MIN_VOLATILITY

    # Generate paths
    use_antithetic = variance_reduction in (
        VarianceReduction.ANTITHETIC,
        VarianceReduction.BOTH,
    )
    paths, times = _generate_paths(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=use_antithetic,
        random_seed=random_seed,
    )

    dt = time_to_expiry / n_steps
    discount = math.exp(-rate * dt)

    # Initialize cash flows (exercise value at expiry)
    if is_call:
        payoffs = np.maximum(paths[:, -1] - strike, 0)
    else:
        payoffs = np.maximum(strike - paths[:, -1], 0)

    cash_flows = payoffs.copy()
    exercise_times = np.full(n_paths, n_steps, dtype=np.int32)

    # Track exercise boundary and probabilities
    boundary_prices = np.zeros(n_steps)
    exercise_counts = np.zeros(n_steps + 1)
    r_squared_values = []
    n_itm_paths_values = []

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]

        # Exercise value at time t
        if is_call:
            exercise_value = np.maximum(S_t - strike, 0)
        else:
            exercise_value = np.maximum(strike - S_t, 0)

        # ITM paths (only regress on ITM paths)
        itm_mask = exercise_value > 0

        if np.sum(itm_mask) >= _MIN_CONTINUATION_PATHS:
            # Discounted future cash flows for ITM paths
            future_cf = cash_flows[itm_mask] * (discount ** (exercise_times[itm_mask] - t))

            # Build design matrix
            S_itm = S_t[itm_mask]
            X = _evaluate_basis(S_itm, degree, basis)

            # Regression: E[continuation | S_t]
            try:
                coeffs, residuals, rank, singular = np.linalg.lstsq(X, future_cf, rcond=None)

                # Compute R-squared
                predicted = X @ coeffs
                ss_res = np.sum((future_cf - predicted) ** 2)
                ss_tot = np.sum((future_cf - np.mean(future_cf)) ** 2)
                r_squared = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0
                r_squared_values.append(r_squared)
                n_itm_paths_values.append(np.sum(itm_mask))

                # Continuation value for ITM paths
                continuation = X @ coeffs

                # Exercise decision: exercise if exercise_value > continuation
                exercise_mask_itm = exercise_value[itm_mask] > continuation

                # Update cash flows and exercise times for paths that exercise
                itm_indices = np.where(itm_mask)[0]
                exercise_indices = itm_indices[exercise_mask_itm]

                cash_flows[exercise_indices] = exercise_value[exercise_indices]
                exercise_times[exercise_indices] = t

                # Track exercise boundary (critical price)
                if np.any(exercise_mask_itm):
                    if is_call:
                        # Boundary is minimum price at which we exercise
                        boundary_prices[t] = np.min(S_itm[exercise_mask_itm])
                    else:
                        # Boundary is maximum price at which we exercise
                        boundary_prices[t] = np.max(S_itm[exercise_mask_itm])

            except np.linalg.LinAlgError:
                # Regression failed - don't exercise at this step
                pass

    # Count exercises at each time
    for i, t in enumerate(exercise_times):
        exercise_counts[t] += 1

    # Discount cash flows to present
    discounted_cf = cash_flows * np.exp(-rate * times[exercise_times])

    # Price estimate
    price = float(np.mean(discounted_cf))
    standard_error = float(np.std(discounted_cf, ddof=1) / math.sqrt(n_paths))

    # Exercise probabilities
    exercise_probs = exercise_counts / n_paths
    cumulative_probs = np.cumsum(exercise_probs)

    prob_early = float(cumulative_probs[-2]) if len(cumulative_probs) > 1 else 0.0
    prob_at_expiry = float(exercise_probs[-1])
    prob_worthless = 1.0 - prob_early - prob_at_expiry

    # Expected exercise time (conditional on exercising early)
    early_mask = exercise_times < n_steps
    if np.sum(early_mask) > 0:
        expected_time = float(np.mean(times[exercise_times[early_mask]]))
    else:
        expected_time = time_to_expiry

    # Extract exercise boundary
    exercise_boundary = None
    if extract_boundary:
        # Interpolate missing boundary points
        valid_boundary = boundary_prices > 0
        if np.any(valid_boundary):
            exercise_boundary = ExerciseBoundary(
                times=times[1:-1],
                boundary_prices=boundary_prices[1:],
                exercise_probabilities=exercise_probs[1:-1],
                cumulative_exercise_prob=cumulative_probs[1:-1],
                is_call=is_call,
            )

    # Compute Greeks via bump-and-revalue
    delta, gamma, delta_se, gamma_se = None, None, None, None
    if compute_greeks:
        bump = spot * 0.01  # 1% bump

        # Delta
        result_up = longstaff_schwartz(
            spot=spot + bump,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
            n_paths=n_paths,
            n_steps=n_steps,
            basis=basis,
            degree=degree,
            variance_reduction=variance_reduction,
            compute_greeks=False,
            extract_boundary=False,
            random_seed=random_seed,
        )
        result_down = longstaff_schwartz(
            spot=spot - bump,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
            n_paths=n_paths,
            n_steps=n_steps,
            basis=basis,
            degree=degree,
            variance_reduction=variance_reduction,
            compute_greeks=False,
            extract_boundary=False,
            random_seed=random_seed,
        )

        delta = (result_up.price - result_down.price) / (2 * bump)
        gamma = (result_up.price - 2 * price + result_down.price) / (bump**2)

        # Standard errors via propagation
        delta_se = math.sqrt(result_up.standard_error**2 + result_down.standard_error**2) / (
            2 * bump
        )
        gamma_se = math.sqrt(
            result_up.standard_error**2 + 4 * standard_error**2 + result_down.standard_error**2
        ) / (bump**2)

    elapsed = (time.perf_counter() - start_time) * 1000

    return LSResult(
        price=price,
        standard_error=standard_error,
        delta=delta,
        gamma=gamma,
        delta_se=delta_se,
        gamma_se=gamma_se,
        exercise_boundary=exercise_boundary,
        expected_exercise_time=expected_time,
        prob_early_exercise=prob_early,
        prob_exercise_at_expiry=prob_at_expiry,
        prob_expire_worthless=prob_worthless,
        n_paths=n_paths,
        n_steps=n_steps,
        basis_functions=basis.value,
        variance_reduction=variance_reduction.value,
        computation_time_ms=elapsed,
        r_squared_avg=float(np.mean(r_squared_values)) if r_squared_values else None,
        n_itm_paths_avg=float(np.mean(n_itm_paths_values)) if n_itm_paths_values else None,
    )


# =============================================================================
# Exercise Probability Analysis
# =============================================================================


def compute_exercise_probability(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    n_paths: int = _DEFAULT_N_PATHS,
    n_steps: int = _DEFAULT_N_STEPS,
    random_seed: Optional[int] = None,
) -> ExerciseProbability:
    """
    Compute detailed early exercise probability distribution.

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate
        dividend_yield: Dividend yield
        volatility: Implied volatility
        is_call: True for call, False for put
        n_paths: Number of Monte Carlo paths
        n_steps: Number of monitoring dates
        random_seed: Random seed

    Returns:
        ExerciseProbability with detailed distribution
    """
    result = longstaff_schwartz(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
        n_paths=n_paths,
        n_steps=n_steps,
        extract_boundary=True,
        random_seed=random_seed,
    )

    if result.exercise_boundary is not None:
        times = result.exercise_boundary.times
        conditional_probs = result.exercise_boundary.exercise_probabilities
        cumulative_probs = result.exercise_boundary.cumulative_exercise_prob
    else:
        dt = time_to_expiry / n_steps
        times = np.arange(1, n_steps) * dt
        conditional_probs = np.zeros(n_steps - 1)
        cumulative_probs = np.zeros(n_steps - 1)

    return ExerciseProbability(
        times=times,
        conditional_probs=conditional_probs,
        cumulative_probs=cumulative_probs,
        expected_time=result.expected_exercise_time or time_to_expiry,
        prob_early=result.prob_early_exercise,
    )


def should_exercise_early(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    threshold: float = 0.5,
    n_paths: int = 50000,
    n_steps: int = 50,
) -> Tuple[bool, float, float]:
    """
    Determine if early exercise is likely optimal.

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration
        rate: Risk-free rate
        dividend_yield: Dividend yield
        volatility: Volatility
        is_call: True for call, False for put
        threshold: Probability threshold for "should exercise"
        n_paths: Monte Carlo paths
        n_steps: Number of steps

    Returns:
        (should_exercise, probability, exercise_value_vs_continuation)
    """
    result = longstaff_schwartz(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
        n_paths=n_paths,
        n_steps=n_steps,
    )

    # Intrinsic value
    if is_call:
        intrinsic = max(0.0, spot - strike)
    else:
        intrinsic = max(0.0, strike - spot)

    # Early exercise is likely if prob > threshold
    should = result.prob_early_exercise > threshold

    # Ratio of intrinsic to option value
    ratio = intrinsic / (result.price + 1e-10) if result.price > 0 else 0.0

    return should, result.prob_early_exercise, ratio


# =============================================================================
# Early Exercise Premium
# =============================================================================


def compute_early_exercise_premium(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    n_paths: int = _DEFAULT_N_PATHS,
    n_steps: int = _DEFAULT_N_STEPS,
) -> Tuple[float, float, float]:
    """
    Compute early exercise premium (American - European value).

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration
        rate: Risk-free rate
        dividend_yield: Dividend yield
        volatility: Volatility
        is_call: True for call, False for put
        n_paths: Monte Carlo paths
        n_steps: Number of steps

    Returns:
        (american_price, european_price, early_exercise_premium)
    """
    # American price via LS
    american_result = longstaff_schwartz(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
        n_paths=n_paths,
        n_steps=n_steps,
    )

    # European price via Black-Scholes
    european_price = _black_scholes_internal(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
    )

    premium = american_result.price - european_price

    return american_result.price, european_price, premium


# =============================================================================
# Analytic Approximations
# =============================================================================


def barone_adesi_whaley(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """
    Barone-Adesi Whaley analytic approximation for American options.

    Fast closed-form approximation that works well for:
    - Short to medium term options
    - Moderate dividend yields

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration
        rate: Risk-free rate
        dividend_yield: Dividend yield
        volatility: Volatility
        is_call: True for call, False for put
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        (american_price, early_exercise_premium)

    References:
        - Barone-Adesi & Whaley (1987)
    """
    if time_to_expiry <= _MIN_TIME:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return intrinsic, 0.0

    # European price
    european = _black_scholes_internal(
        spot, strike, time_to_expiry, rate, dividend_yield, volatility, is_call
    )

    # For calls with no dividends, American = European
    if is_call and dividend_yield <= 0:
        return european, 0.0

    # Parameters - use a safe rate floor to avoid division-by-zero when rate is 0.0
    rate_safe = max(rate, 1e-10)
    sigma2 = volatility * volatility
    h = 1.0 - math.exp(-rate_safe * time_to_expiry)

    # M and N parameters
    M = 2 * rate_safe / sigma2
    N = 2 * (rate_safe - dividend_yield) / sigma2

    k = 1.0 - h
    q = 0.5 * (-(N - 1) + math.sqrt((N - 1) ** 2 + 4 * M / h))

    if is_call:
        # Find critical price S* for call
        S_star = strike
        for _ in range(max_iter):
            d1 = (
                math.log(S_star / strike) + (rate - dividend_yield + 0.5 * sigma2) * time_to_expiry
            ) / (volatility * math.sqrt(time_to_expiry))

            nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

            bs_call = _black_scholes_internal(
                S_star, strike, time_to_expiry, rate, dividend_yield, volatility, True
            )

            lhs = S_star - strike
            rhs = bs_call + (1 - math.exp(-dividend_yield * time_to_expiry) * nd1) * S_star / q

            if abs(lhs - rhs) < tol:
                break

            # Newton-Raphson update
            deriv = 1 - (1 - math.exp(-dividend_yield * time_to_expiry) * nd1) / q
            if abs(deriv) < 1e-10:
                break
            S_star = S_star - (lhs - rhs) / deriv
            S_star = max(strike * 1.0001, S_star)

        if spot >= S_star:
            return spot - strike, spot - strike - european

        A = (S_star / q) * (1 - math.exp(-dividend_yield * time_to_expiry) * nd1)
        american = european + A * (spot / S_star) ** q
        premium = american - european

    else:
        # For puts, similar logic with different q
        q = 0.5 * (-(N - 1) - math.sqrt((N - 1) ** 2 + 4 * M / h))

        S_star = strike
        for _ in range(max_iter):
            d1 = (
                math.log(S_star / strike) + (rate - dividend_yield + 0.5 * sigma2) * time_to_expiry
            ) / (volatility * math.sqrt(time_to_expiry))

            nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

            bs_put = _black_scholes_internal(
                S_star, strike, time_to_expiry, rate, dividend_yield, volatility, False
            )

            lhs = strike - S_star
            rhs = bs_put - (1 - math.exp(-dividend_yield * time_to_expiry) * (1 - nd1)) * S_star / q

            if abs(lhs - rhs) < tol:
                break

            deriv = -1 + (1 - math.exp(-dividend_yield * time_to_expiry) * (1 - nd1)) / q
            if abs(deriv) < 1e-10:
                break
            S_star = S_star - (lhs - rhs) / deriv
            S_star = max(0.01, min(strike * 0.9999, S_star))

        if spot <= S_star:
            return strike - spot, strike - spot - european

        A = -(S_star / q) * (1 - math.exp(-dividend_yield * time_to_expiry) * (1 - nd1))
        american = european + A * (spot / S_star) ** q
        premium = american - european

    return max(european, american), max(0.0, premium)


# =============================================================================
# Internal Functions
# =============================================================================


def _black_scholes_internal(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> float:
    """Internal Black-Scholes for comparisons."""
    if time_to_expiry <= _MIN_TIME:
        if is_call:
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    if volatility <= _MIN_VOLATILITY:
        df = math.exp(-rate * time_to_expiry)
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        if is_call:
            return max(0.0, forward - strike) * df
        return max(0.0, strike - forward) * df

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    df_rate = math.exp(-rate * time_to_expiry)
    df_div = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return spot * df_div * nd1 - strike * df_rate * nd2
    else:
        return strike * df_rate * (1 - nd2) - spot * df_div * (1 - nd1)


# =============================================================================
# Factory Functions
# =============================================================================


def create_exercise_analyzer(
    n_paths: int = _DEFAULT_N_PATHS,
    n_steps: int = _DEFAULT_N_STEPS,
    basis: BasisFunctions = BasisFunctions.LAGUERRE,
    degree: int = _DEFAULT_DEGREE,
) -> Callable[..., LSResult]:
    """
    Create a configured exercise probability analyzer.

    Returns:
        Function that takes option parameters and returns LSResult
    """

    def analyze(
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        dividend_yield: float,
        volatility: float,
        is_call: bool,
        random_seed: Optional[int] = None,
    ) -> LSResult:
        return longstaff_schwartz(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            is_call=is_call,
            n_paths=n_paths,
            n_steps=n_steps,
            basis=basis,
            degree=degree,
            compute_greeks=True,
            extract_boundary=True,
            random_seed=random_seed,
        )

    return analyze


def quick_exercise_check(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float = 0.05,
    dividend_yield: float = 0.02,
    volatility: float = 0.25,
    is_call: bool = True,
) -> dict:
    """
    Quick analysis of early exercise potential.

    Returns dictionary with key metrics for exercise decision.
    """
    result = longstaff_schwartz(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        is_call=is_call,
        n_paths=50000,
        n_steps=50,
    )

    # Intrinsic value
    if is_call:
        intrinsic = max(0.0, spot - strike)
    else:
        intrinsic = max(0.0, strike - spot)

    return {
        "price": result.price,
        "intrinsic_value": intrinsic,
        "time_value": result.price - intrinsic,
        "prob_early_exercise": result.prob_early_exercise,
        "expected_exercise_time": result.expected_exercise_time,
        "prob_expire_worthless": result.prob_expire_worthless,
        "recommendation": (
            "Consider exercising"
            if result.prob_early_exercise > 0.7
            else "Hold" if result.prob_early_exercise < 0.3 else "Monitor closely"
        ),
    }
