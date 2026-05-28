# -*- coding: utf-8 -*-
"""
impl_jump_calibration.py
Jump-diffusion model parameter calibration.

Phase 1: Core Models & Data Structures

This module provides calibration methods for Merton jump-diffusion model parameters:
- λ (lambda): Jump intensity (average number of jumps per year)
- μ_J (mu_J): Mean jump size in log space
- σ_J (sigma_J): Jump size volatility

Calibration Methods:
1. Option Price Calibration (implied parameters)
   - Minimizes squared pricing error across option chain
   - Uses gradient-based optimization (L-BFGS-B)
   - Supports multiple maturities simultaneously

2. Historical Returns Calibration (realized parameters)
   - Method of moments estimation
   - Maximum likelihood estimation (MLE)
   - Threshold-based jump detection

3. Hybrid Calibration
   - Combines historical and option-implied estimates
   - Weighted average with user-defined weights

References:
    - Merton (1976): "Option Pricing When Underlying Stock Returns Are Discontinuous"
    - Cont & Tankov (2004): "Financial Modelling with Jump Processes"
    - Honoré (1998): "Pitfalls in Estimating Jump-Diffusion Models"
    - Andersen et al. (2002): "Empirical Analysis of Jump Diffusion"
    - Carr & Wu (2003): "The Finite Moment Log Stable Process"
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize
from scipy import stats
from scipy.special import factorial

from core_options import JumpParams
from core_errors import CalibrationError


# =============================================================================
# Constants
# =============================================================================

_MIN_LAMBDA = 1e-6
_MAX_LAMBDA = 50.0  # Max 50 jumps per year
_MIN_SIGMA_J = 1e-6
_MAX_SIGMA_J = 2.0  # Max 200% jump volatility
_MIN_MU_J = -2.0  # Min -200% mean jump (log space)
_MAX_MU_J = 2.0  # Max +200% mean jump (log space)

# Convergence tolerances
_DEFAULT_TOL = 1e-8
_DEFAULT_MAX_ITER = 500

# Jump detection thresholds
_DEFAULT_JUMP_THRESHOLD_SIGMAS = 3.0


# =============================================================================
# Enums and Data Classes
# =============================================================================

class CalibrationMethod(Enum):
    """Calibration method selection."""
    OPTION_PRICES = "option_prices"
    HISTORICAL_MOMENTS = "historical_moments"
    HISTORICAL_MLE = "historical_mle"
    HYBRID = "hybrid"


class OptimizationMethod(Enum):
    """Optimization algorithm selection."""
    L_BFGS_B = "L-BFGS-B"
    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    SLSQP = "SLSQP"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


@dataclass
class CalibrationInput:
    """Input data for jump calibration."""
    # Option data (for option price calibration)
    option_prices: Optional[np.ndarray] = None
    strikes: Optional[np.ndarray] = None
    maturities: Optional[np.ndarray] = None
    is_calls: Optional[np.ndarray] = None
    spot: Optional[float] = None
    rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    base_volatility: Optional[float] = None  # Diffusion component

    # Historical returns (for historical calibration)
    returns: Optional[np.ndarray] = None
    dt: Optional[float] = None  # Time step (e.g., 1/252 for daily)

    # Weights for each option (for weighted calibration)
    weights: Optional[np.ndarray] = None


@dataclass
class CalibrationResult:
    """Result of jump parameter calibration."""
    # Calibrated parameters
    jump_params: JumpParams

    # Calibration quality metrics
    rmse: float  # Root mean squared error
    mape: float  # Mean absolute percentage error
    r_squared: float  # R-squared (coefficient of determination)

    # Optimization details
    n_iterations: int
    converged: bool
    method: CalibrationMethod
    optimization_method: str

    # Diagnostics
    residuals: Optional[np.ndarray] = None
    jacobian_norm: Optional[float] = None
    hessian_condition: Optional[float] = None

    # Timing
    computation_time_ms: float = 0.0

    # Parameter uncertainty (if available)
    lambda_std: Optional[float] = None
    mu_j_std: Optional[float] = None
    sigma_j_std: Optional[float] = None


@dataclass
class JumpDetectionResult:
    """Result of jump detection from historical returns."""
    jump_times: np.ndarray  # Indices of detected jumps
    jump_sizes: np.ndarray  # Sizes of detected jumps
    n_jumps: int
    threshold: float
    diffusion_returns: np.ndarray  # Returns with jumps removed


# =============================================================================
# Jump-Diffusion Pricing (internal, for calibration)
# =============================================================================

def _merton_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
    jump_params: JumpParams,
    max_terms: int = 50,
) -> float:
    """
    Merton jump-diffusion price for calibration.

    Uses Poisson-weighted sum of Black-Scholes prices.
    """
    if time_to_expiry <= 0 or spot <= 0 or strike <= 0:
        return 0.0

    lambda_ = jump_params.lambda_intensity
    mu_j = jump_params.mu_jump
    sigma_j = jump_params.sigma_jump

    # Mean jump factor
    k = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0

    # Compensator for drift
    lambda_prime = lambda_ * (1.0 + k)

    price = 0.0
    poisson_prob_sum = 0.0

    for n in range(max_terms):
        # Poisson probability
        poisson_prob = (
            math.exp(-lambda_prime * time_to_expiry)
            * (lambda_prime * time_to_expiry) ** n
            / math.factorial(n)
        )

        if poisson_prob < 1e-15 and n > 5:
            break

        poisson_prob_sum += poisson_prob

        # Adjusted volatility and rate for n jumps
        sigma_n = math.sqrt(
            volatility * volatility + n * sigma_j * sigma_j / time_to_expiry
        )

        # Risk-neutral drift adjustment
        r_n = (
            rate
            - dividend_yield
            - lambda_ * k
            + n * (mu_j + 0.5 * sigma_j * sigma_j) / time_to_expiry
        )

        # Black-Scholes price with adjusted parameters
        bs_price = _black_scholes_internal(
            spot, strike, time_to_expiry, r_n + dividend_yield,
            dividend_yield, sigma_n, is_call
        )

        price += poisson_prob * bs_price

    return price


def _black_scholes_internal(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    is_call: bool,
) -> float:
    """Internal Black-Scholes for calibration."""
    if time_to_expiry <= 1e-10:
        if is_call:
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    if volatility <= 1e-10:
        df = math.exp(-rate * time_to_expiry)
        forward = spot * math.exp((rate - dividend_yield) * time_to_expiry)
        if is_call:
            return max(0.0, forward - strike) * df
        return max(0.0, strike - forward) * df

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    # Standard normal CDF
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    df_rate = math.exp(-rate * time_to_expiry)
    df_div = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return spot * df_div * nd1 - strike * df_rate * nd2
    else:
        return strike * df_rate * (1.0 - nd2) - spot * df_div * (1.0 - nd1)


# =============================================================================
# Jump Detection from Historical Returns
# =============================================================================

def detect_jumps(
    returns: np.ndarray,
    threshold_sigmas: float = _DEFAULT_JUMP_THRESHOLD_SIGMAS,
    robust: bool = True,
) -> JumpDetectionResult:
    """
    Detect jumps in historical returns using threshold method.

    Args:
        returns: Array of log returns
        threshold_sigmas: Number of standard deviations for jump detection
        robust: Use robust estimators (median, MAD) instead of mean/std

    Returns:
        JumpDetectionResult with detected jumps and cleaned returns

    References:
        - Lee & Mykland (2008): "Jumps in Financial Markets"
        - Barndorff-Nielsen & Shephard (2006): "Econometrics of Testing for Jumps"
    """
    returns = np.asarray(returns)
    n = len(returns)

    if n < 10:
        raise CalibrationError("Need at least 10 returns for jump detection")

    if robust:
        # Robust location and scale estimates
        center = np.median(returns)
        # MAD scaled to be consistent with std for normal distribution
        scale = 1.4826 * np.median(np.abs(returns - center))
    else:
        center = np.mean(returns)
        scale = np.std(returns, ddof=1)

    if scale < 1e-10:
        scale = 1e-10

    # Standardized returns
    standardized = (returns - center) / scale

    # Detect jumps
    threshold = threshold_sigmas
    jump_mask = np.abs(standardized) > threshold

    jump_times = np.where(jump_mask)[0]
    jump_sizes = returns[jump_mask]
    n_jumps = len(jump_times)

    # Create diffusion returns (with jumps removed)
    diffusion_returns = returns.copy()
    if n_jumps > 0:
        # Replace jumps with median
        diffusion_returns[jump_mask] = center

    return JumpDetectionResult(
        jump_times=jump_times,
        jump_sizes=jump_sizes,
        n_jumps=n_jumps,
        threshold=threshold * scale,
        diffusion_returns=diffusion_returns,
    )


# =============================================================================
# Historical Calibration Methods
# =============================================================================

def calibrate_from_moments(
    returns: np.ndarray,
    dt: float,
    jump_detection: Optional[JumpDetectionResult] = None,
    threshold_sigmas: float = _DEFAULT_JUMP_THRESHOLD_SIGMAS,
) -> CalibrationResult:
    """
    Calibrate jump parameters using method of moments.

    Uses the first four moments of returns to estimate:
    - σ (diffusion volatility)
    - λ (jump intensity)
    - μ_J (mean jump size)
    - σ_J (jump volatility)

    Args:
        returns: Array of log returns
        dt: Time step (e.g., 1/252 for daily returns)
        jump_detection: Optional pre-computed jump detection result
        threshold_sigmas: Jump detection threshold if not pre-computed

    Returns:
        CalibrationResult with estimated parameters

    References:
        - Press (1967): "A Compound Events Model for Security Prices"
        - Honoré (1998): "Pitfalls in Estimating Jump-Diffusion Models"
    """
    start_time = time.perf_counter()

    returns = np.asarray(returns)
    n = len(returns)

    if dt <= 0:
        raise CalibrationError(f"Time step dt must be positive, got {dt}")

    if n < 30:
        raise CalibrationError("Need at least 30 returns for moment calibration")

    # Detect jumps if not provided
    if jump_detection is None:
        jump_detection = detect_jumps(returns, threshold_sigmas)

    # Estimate diffusion volatility from cleaned returns
    diffusion_vol = np.std(jump_detection.diffusion_returns, ddof=1) / math.sqrt(dt)

    # Estimate jump intensity
    # λ = (number of jumps) / (total time)
    total_time = n * dt
    lambda_est = jump_detection.n_jumps / total_time
    lambda_est = max(_MIN_LAMBDA, min(_MAX_LAMBDA, lambda_est))

    # Estimate jump size distribution
    if jump_detection.n_jumps >= 3:
        # Mean and std of jump sizes
        mu_j_est = float(np.mean(jump_detection.jump_sizes))
        sigma_j_est = float(np.std(jump_detection.jump_sizes, ddof=1))
    else:
        # Use higher moments if few jumps detected
        # Estimate from excess kurtosis
        kurtosis = stats.kurtosis(returns, fisher=True)
        skewness = stats.skew(returns)

        # Simple approximation
        sigma_j_est = 0.1  # 10% default
        mu_j_est = skewness * sigma_j_est / 3.0

    # Bound estimates
    mu_j_est = max(_MIN_MU_J, min(_MAX_MU_J, mu_j_est))
    sigma_j_est = max(_MIN_SIGMA_J, min(_MAX_SIGMA_J, sigma_j_est))

    # Calculate fit quality metrics
    # Compare empirical and model moments
    model_mean = mu_j_est * lambda_est * dt
    model_var = (diffusion_vol ** 2 + lambda_est * (sigma_j_est ** 2 + mu_j_est ** 2)) * dt

    empirical_mean = float(np.mean(returns))
    empirical_var = float(np.var(returns, ddof=1))

    # Simple R² approximation
    ss_res = (empirical_mean - model_mean) ** 2 + (empirical_var - model_var) ** 2
    ss_tot = empirical_mean ** 2 + empirical_var ** 2 + 1e-10
    r_squared = max(0.0, 1.0 - ss_res / ss_tot)

    # RMSE on standardized returns
    rmse = math.sqrt(ss_res / 2)
    mape = abs(ss_res / ss_tot)

    elapsed = (time.perf_counter() - start_time) * 1000

    return CalibrationResult(
        jump_params=JumpParams(
            lambda_intensity=lambda_est,
            mu_jump=mu_j_est,
            sigma_jump=sigma_j_est,
        ),
        rmse=rmse,
        mape=mape,
        r_squared=r_squared,
        n_iterations=1,
        converged=True,
        method=CalibrationMethod.HISTORICAL_MOMENTS,
        optimization_method="moments",
        computation_time_ms=elapsed,
    )


def calibrate_from_mle(
    returns: np.ndarray,
    dt: float,
    initial_guess: Optional[JumpParams] = None,
    max_iterations: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> CalibrationResult:
    """
    Calibrate jump parameters using Maximum Likelihood Estimation.

    Maximizes the log-likelihood of the Merton jump-diffusion model.

    Args:
        returns: Array of log returns
        dt: Time step (e.g., 1/252 for daily returns)
        initial_guess: Optional initial parameter guess
        max_iterations: Maximum optimization iterations
        tol: Convergence tolerance

    Returns:
        CalibrationResult with MLE estimates

    References:
        - Honoré (1998): "Pitfalls in Estimating Jump-Diffusion Models"
        - Cont & Tankov (2004): "Financial Modelling with Jump Processes"
    """
    start_time = time.perf_counter()

    returns = np.asarray(returns)
    n = len(returns)

    if dt <= 0:
        raise CalibrationError(f"Time step dt must be positive, got {dt}")

    if n < 50:
        raise CalibrationError("Need at least 50 returns for MLE calibration")

    # Get initial guess
    if initial_guess is None:
        # Use moment method for initial guess
        moment_result = calibrate_from_moments(returns, dt)
        initial_guess = moment_result.jump_params

    # Initial parameters: [sigma, lambda, mu_j, sigma_j]
    sigma_init = float(np.std(returns, ddof=1) / math.sqrt(dt))
    x0 = np.array([
        sigma_init,
        initial_guess.lambda_intensity,
        initial_guess.mu_jump,
        initial_guess.sigma_jump,
    ])

    def negative_log_likelihood(params: np.ndarray) -> float:
        """Negative log-likelihood for optimization."""
        sigma, lambda_, mu_j, sigma_j = params

        # Parameter bounds check
        if sigma <= 0 or lambda_ < 0 or sigma_j <= 0:
            return 1e10

        total_ll = 0.0
        max_jumps = 10  # Truncate Poisson sum

        for r in returns:
            # Log-likelihood as Poisson-weighted mixture
            prob_sum = 0.0

            for k in range(max_jumps + 1):
                # Poisson probability
                poisson_prob = (
                    math.exp(-lambda_ * dt)
                    * (lambda_ * dt) ** k
                    / math.factorial(k)
                )

                if poisson_prob < 1e-15:
                    break

                # Conditional density given k jumps
                # Mean: k * mu_j
                # Variance: sigma^2 * dt + k * sigma_j^2
                cond_mean = k * mu_j
                cond_var = sigma * sigma * dt + k * sigma_j * sigma_j

                if cond_var <= 0:
                    continue

                cond_std = math.sqrt(cond_var)

                # Normal density
                z = (r - cond_mean) / cond_std
                pdf = math.exp(-0.5 * z * z) / (cond_std * math.sqrt(2 * math.pi))

                prob_sum += poisson_prob * pdf

            if prob_sum > 1e-100:
                total_ll += math.log(prob_sum)
            else:
                total_ll -= 100  # Penalty for zero likelihood

        return -total_ll

    # Bounds
    bounds = [
        (1e-6, 2.0),  # sigma
        (_MIN_LAMBDA, _MAX_LAMBDA),  # lambda
        (_MIN_MU_J, _MAX_MU_J),  # mu_j
        (_MIN_SIGMA_J, _MAX_SIGMA_J),  # sigma_j
    ]

    # Optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize(
            negative_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": tol},
        )

    sigma_opt, lambda_opt, mu_j_opt, sigma_j_opt = result.x

    # Calculate metrics
    n_params = 4
    ll = -result.fun

    # Approximate R² using pseudo-R²
    # Compare to null model (no jumps)
    null_ll = -0.5 * n * (
        math.log(2 * math.pi * sigma_opt ** 2 * dt)
        + np.sum((returns - np.mean(returns)) ** 2) / (sigma_opt ** 2 * dt * n)
    )

    pseudo_r2 = max(0.0, 1.0 - ll / (null_ll - 1e-10)) if null_ll < 0 else 0.0

    # RMSE from model fit
    model_mean = mu_j_opt * lambda_opt * dt
    model_std = math.sqrt(sigma_opt ** 2 * dt + lambda_opt * (sigma_j_opt ** 2 + mu_j_opt ** 2) * dt)

    empirical_mean = float(np.mean(returns))
    empirical_std = float(np.std(returns, ddof=1))

    rmse = math.sqrt(
        (empirical_mean - model_mean) ** 2 + (empirical_std - model_std) ** 2
    ) / 2

    mape = abs((empirical_std - model_std) / (empirical_std + 1e-10))

    elapsed = (time.perf_counter() - start_time) * 1000

    return CalibrationResult(
        jump_params=JumpParams(
            lambda_intensity=lambda_opt,
            mu_jump=mu_j_opt,
            sigma_jump=sigma_j_opt,
        ),
        rmse=rmse,
        mape=mape,
        r_squared=pseudo_r2,
        n_iterations=result.nit,
        converged=result.success,
        method=CalibrationMethod.HISTORICAL_MLE,
        optimization_method="L-BFGS-B",
        computation_time_ms=elapsed,
    )


# =============================================================================
# Option Price Calibration
# =============================================================================

def calibrate_from_options(
    option_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    is_calls: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
    base_volatility: float,
    initial_guess: Optional[JumpParams] = None,
    weights: Optional[np.ndarray] = None,
    optimization_method: OptimizationMethod = OptimizationMethod.L_BFGS_B,
    max_iterations: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> CalibrationResult:
    """
    Calibrate jump parameters by fitting to market option prices.

    Minimizes weighted sum of squared pricing errors between model
    and market prices.

    Args:
        option_prices: Market option prices
        strikes: Option strike prices
        maturities: Time to expiration (years)
        is_calls: True for call options, False for puts
        spot: Current underlying price
        rate: Risk-free rate (annualized)
        dividend_yield: Continuous dividend yield
        base_volatility: Base diffusion volatility
        initial_guess: Optional initial parameter guess
        weights: Optional weights for each option
        optimization_method: Optimization algorithm
        max_iterations: Maximum iterations
        tol: Convergence tolerance

    Returns:
        CalibrationResult with calibrated parameters

    References:
        - Cont & Tankov (2004): "Calibration of Jump-Diffusion Option Pricing Models"
        - Andersen & Andreasen (2000): "Jump-Diffusion Processes"
    """
    start_time = time.perf_counter()

    # Validate inputs
    n_options = len(option_prices)
    if n_options < 3:
        raise CalibrationError("Need at least 3 options for calibration")
    if spot <= 0:
        raise CalibrationError(f"Spot must be positive, got {spot}")
    if base_volatility <= 0:
        raise CalibrationError(f"Base volatility must be positive, got {base_volatility}")

    option_prices = np.asarray(option_prices, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    maturities = np.asarray(maturities, dtype=np.float64)
    is_calls = np.asarray(is_calls, dtype=bool)

    if np.any(option_prices < 0):
        raise CalibrationError("Option prices cannot be negative")
    if np.any(strikes <= 0):
        raise CalibrationError("Strikes must be positive")
    if np.any(maturities <= 0):
        raise CalibrationError("Maturities must be positive")

    if weights is None:
        # Vega-weighted by default (ATM options weighted more)
        moneyness = spot / strikes
        weights = np.exp(-0.5 * (np.log(moneyness)) ** 2 / (0.1 ** 2))
        sum_weights = np.sum(weights)
        if sum_weights <= 0.0:
            raise CalibrationError("Sum of calculated weights is zero or negative")
        weights = weights / sum_weights
    else:
        weights = np.asarray(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        if sum_weights <= 0.0:
            raise CalibrationError("Sum of provided weights must be positive")
        weights = weights / sum_weights

    # Initial guess
    if initial_guess is None:
        initial_guess = JumpParams(
            lambda_intensity=1.0,
            mu_jump=-0.05,
            sigma_jump=0.15,
        )

    x0 = np.array([
        initial_guess.lambda_intensity,
        initial_guess.mu_jump,
        initial_guess.sigma_jump,
    ])

    def objective(params: np.ndarray) -> float:
        """Weighted sum of squared errors."""
        lambda_, mu_j, sigma_j = params

        jump_params = JumpParams(
            lambda_intensity=lambda_,
            mu_jump=mu_j,
            sigma_jump=sigma_j,
        )

        total_error = 0.0

        for i in range(n_options):
            model_price = _merton_price(
                spot=spot,
                strike=strikes[i],
                time_to_expiry=maturities[i],
                rate=rate,
                dividend_yield=dividend_yield,
                volatility=base_volatility,
                is_call=bool(is_calls[i]),
                jump_params=jump_params,
            )

            error = (model_price - option_prices[i]) ** 2
            total_error += weights[i] * error

        return total_error

    # Bounds
    bounds = [
        (_MIN_LAMBDA, _MAX_LAMBDA),
        (_MIN_MU_J, _MAX_MU_J),
        (_MIN_SIGMA_J, _MAX_SIGMA_J),
    ]

    # Select optimization method
    if optimization_method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.differential_evolution(
                objective,
                bounds,
                maxiter=max_iterations,
                tol=tol,
                seed=42,
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                objective,
                x0,
                method=optimization_method.value,
                bounds=bounds,
                options={"maxiter": max_iterations},
                tol=tol,
            )

    lambda_opt, mu_j_opt, sigma_j_opt = result.x

    # Calculate residuals and metrics
    jump_params_opt = JumpParams(
        lambda_intensity=lambda_opt,
        mu_jump=mu_j_opt,
        sigma_jump=sigma_j_opt,
    )

    residuals = np.zeros(n_options)
    for i in range(n_options):
        model_price = _merton_price(
            spot=spot,
            strike=strikes[i],
            time_to_expiry=maturities[i],
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=base_volatility,
            is_call=bool(is_calls[i]),
            jump_params=jump_params_opt,
        )
        residuals[i] = model_price - option_prices[i]

    # Metrics
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mape = float(np.mean(np.abs(residuals) / (option_prices + 1e-10)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((option_prices - np.mean(option_prices)) ** 2))
    r_squared = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

    elapsed = (time.perf_counter() - start_time) * 1000

    return CalibrationResult(
        jump_params=jump_params_opt,
        rmse=rmse,
        mape=mape,
        r_squared=r_squared,
        n_iterations=result.nit if hasattr(result, "nit") else 0,
        converged=result.success,
        method=CalibrationMethod.OPTION_PRICES,
        optimization_method=optimization_method.value,
        residuals=residuals,
        computation_time_ms=elapsed,
    )


# =============================================================================
# Hybrid Calibration
# =============================================================================

def calibrate_hybrid(
    calibration_input: CalibrationInput,
    historical_weight: float = 0.3,
    option_weight: float = 0.7,
    use_mle: bool = True,
) -> CalibrationResult:
    """
    Hybrid calibration combining historical and option-implied parameters.

    Weighted average of parameters from historical data and option prices.

    Args:
        calibration_input: Input data containing both returns and option prices
        historical_weight: Weight for historical estimates (0-1)
        option_weight: Weight for option-implied estimates (0-1)
        use_mle: Use MLE for historical calibration (vs moments)

    Returns:
        CalibrationResult with combined parameters
    """
    start_time = time.perf_counter()

    # Validate inputs
    has_historical = (
        calibration_input.returns is not None
        and calibration_input.dt is not None
    )
    has_options = (
        calibration_input.option_prices is not None
        and calibration_input.strikes is not None
        and calibration_input.maturities is not None
        and calibration_input.is_calls is not None
        and calibration_input.spot is not None
        and calibration_input.rate is not None
        and calibration_input.base_volatility is not None
    )

    if not has_historical and not has_options:
        raise CalibrationError("Need either historical returns or option prices")

    # Normalize weights
    total_weight = historical_weight + option_weight
    if total_weight <= 0:
        raise CalibrationError("Weights must be positive")

    historical_weight = historical_weight / total_weight
    option_weight = option_weight / total_weight

    results = []
    weights_used = []

    # Historical calibration
    if has_historical:
        if use_mle:
            hist_result = calibrate_from_mle(
                calibration_input.returns,
                calibration_input.dt,
            )
        else:
            hist_result = calibrate_from_moments(
                calibration_input.returns,
                calibration_input.dt,
            )
        results.append(hist_result)
        weights_used.append(historical_weight if has_options else 1.0)

    # Option price calibration
    if has_options:
        initial_guess = results[0].jump_params if results else None

        opt_result = calibrate_from_options(
            option_prices=calibration_input.option_prices,
            strikes=calibration_input.strikes,
            maturities=calibration_input.maturities,
            is_calls=calibration_input.is_calls,
            spot=calibration_input.spot,
            rate=calibration_input.rate,
            dividend_yield=calibration_input.dividend_yield or 0.0,
            base_volatility=calibration_input.base_volatility,
            weights=calibration_input.weights,
            initial_guess=initial_guess,
        )
        results.append(opt_result)
        weights_used.append(option_weight if has_historical else 1.0)

    # Combine parameters
    if len(results) == 1:
        combined_result = results[0]
        combined_result.method = CalibrationMethod.HYBRID
    else:
        # Weighted average
        combined_lambda = sum(
            w * r.jump_params.lambda_intensity
            for w, r in zip(weights_used, results)
        )
        combined_mu_j = sum(
            w * r.jump_params.mu_jump
            for w, r in zip(weights_used, results)
        )
        combined_sigma_j = sum(
            w * r.jump_params.sigma_jump
            for w, r in zip(weights_used, results)
        )

        combined_rmse = sum(w * r.rmse for w, r in zip(weights_used, results))
        combined_mape = sum(w * r.mape for w, r in zip(weights_used, results))
        combined_r2 = sum(w * r.r_squared for w, r in zip(weights_used, results))

        elapsed = (time.perf_counter() - start_time) * 1000

        combined_result = CalibrationResult(
            jump_params=JumpParams(
                lambda_intensity=combined_lambda,
                mu_jump=combined_mu_j,
                sigma_jump=combined_sigma_j,
            ),
            rmse=combined_rmse,
            mape=combined_mape,
            r_squared=combined_r2,
            n_iterations=sum(r.n_iterations for r in results),
            converged=all(r.converged for r in results),
            method=CalibrationMethod.HYBRID,
            optimization_method="hybrid",
            computation_time_ms=elapsed,
        )

    return combined_result


# =============================================================================
# Calibration Diagnostics
# =============================================================================

def compute_calibration_diagnostics(
    result: CalibrationResult,
    option_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    is_calls: np.ndarray,
    spot: float,
    rate: float,
    dividend_yield: float,
    base_volatility: float,
) -> Dict[str, float]:
    """
    Compute detailed diagnostics for calibration result.

    Returns:
        Dictionary with diagnostic metrics:
        - max_error: Maximum absolute pricing error
        - mean_error: Mean absolute pricing error
        - median_error: Median absolute pricing error
        - skewness_error: Skewness of error distribution
        - kurtosis_error: Kurtosis of error distribution
        - atm_error: Error for near-ATM options
        - otm_call_error: Error for OTM calls
        - otm_put_error: Error for OTM puts
    """
    jump_params = result.jump_params
    n_options = len(option_prices)

    errors = np.zeros(n_options)
    model_prices = np.zeros(n_options)

    for i in range(n_options):
        model_prices[i] = _merton_price(
            spot=spot,
            strike=strikes[i],
            time_to_expiry=maturities[i],
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=base_volatility,
            is_call=bool(is_calls[i]),
            jump_params=jump_params,
        )
        errors[i] = model_prices[i] - option_prices[i]

    abs_errors = np.abs(errors)
    rel_errors = abs_errors / (option_prices + 1e-10)

    # Moneyness categories
    moneyness = spot / strikes
    atm_mask = (moneyness > 0.95) & (moneyness < 1.05)
    otm_call_mask = is_calls & (moneyness < 0.95)
    otm_put_mask = ~is_calls & (moneyness > 1.05)

    diagnostics = {
        "max_abs_error": float(np.max(abs_errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
        "error_skewness": float(stats.skew(errors)),
        "error_kurtosis": float(stats.kurtosis(errors)),
    }

    if np.any(atm_mask):
        diagnostics["atm_mean_error"] = float(np.mean(abs_errors[atm_mask]))

    if np.any(otm_call_mask):
        diagnostics["otm_call_mean_error"] = float(np.mean(abs_errors[otm_call_mask]))

    if np.any(otm_put_mask):
        diagnostics["otm_put_mean_error"] = float(np.mean(abs_errors[otm_put_mask]))

    return diagnostics


# =============================================================================
# High-Level Calibration Interface
# =============================================================================

class JumpCalibrator:
    """
    High-level interface for jump parameter calibration.

    Supports multiple calibration methods and maintains calibration history.

    Example:
        >>> calibrator = JumpCalibrator()
        >>> result = calibrator.calibrate(
        ...     method=CalibrationMethod.OPTION_PRICES,
        ...     option_prices=prices,
        ...     strikes=strikes,
        ...     maturities=maturities,
        ...     is_calls=is_calls,
        ...     spot=100.0,
        ...     rate=0.05,
        ...     base_volatility=0.2,
        ... )
        >>> print(f"λ = {result.jump_params.lambda_intensity:.4f}")
        >>> print(f"μ_J = {result.jump_params.mu_jump:.4f}")
        >>> print(f"σ_J = {result.jump_params.sigma_jump:.4f}")
    """

    def __init__(self):
        """Initialize calibrator."""
        self._history: List[CalibrationResult] = []
        self._last_result: Optional[CalibrationResult] = None

    def calibrate(
        self,
        method: CalibrationMethod = CalibrationMethod.OPTION_PRICES,
        **kwargs,
    ) -> CalibrationResult:
        """
        Calibrate jump parameters using specified method.

        Args:
            method: Calibration method to use
            **kwargs: Method-specific arguments

        Returns:
            CalibrationResult with calibrated parameters
        """
        if method == CalibrationMethod.OPTION_PRICES:
            result = calibrate_from_options(**kwargs)
        elif method == CalibrationMethod.HISTORICAL_MOMENTS:
            result = calibrate_from_moments(**kwargs)
        elif method == CalibrationMethod.HISTORICAL_MLE:
            result = calibrate_from_mle(**kwargs)
        elif method == CalibrationMethod.HYBRID:
            result = calibrate_hybrid(**kwargs)
        else:
            raise CalibrationError(f"Unknown calibration method: {method}")

        self._last_result = result
        self._history.append(result)

        return result

    def get_last_result(self) -> Optional[CalibrationResult]:
        """Get most recent calibration result."""
        return self._last_result

    def get_history(self) -> List[CalibrationResult]:
        """Get calibration history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear calibration history."""
        self._history.clear()
        self._last_result = None


# =============================================================================
# Factory Functions
# =============================================================================

def create_calibrator() -> JumpCalibrator:
    """Create a new JumpCalibrator instance."""
    return JumpCalibrator()


def quick_calibrate_from_options(
    option_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    is_calls: np.ndarray,
    spot: float,
    rate: float = 0.05,
    dividend_yield: float = 0.0,
    base_volatility: float = 0.2,
) -> JumpParams:
    """
    Quick calibration from option prices with sensible defaults.

    Returns:
        JumpParams with calibrated parameters
    """
    result = calibrate_from_options(
        option_prices=option_prices,
        strikes=strikes,
        maturities=maturities,
        is_calls=is_calls,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        base_volatility=base_volatility,
    )
    return result.jump_params


def quick_calibrate_from_returns(
    returns: np.ndarray,
    dt: float = 1 / 252,
    use_mle: bool = False,
) -> JumpParams:
    """
    Quick calibration from historical returns with sensible defaults.

    Args:
        returns: Array of log returns
        dt: Time step (default: daily = 1/252)
        use_mle: Use MLE instead of moments

    Returns:
        JumpParams with calibrated parameters
    """
    if use_mle:
        result = calibrate_from_mle(returns, dt)
    else:
        result = calibrate_from_moments(returns, dt)
    return result.jump_params
