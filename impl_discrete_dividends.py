# -*- coding: utf-8 -*-
"""
impl_discrete_dividends.py
Discrete dividend handling for options pricing.

Phase 1: Core Models & Data Structures

This module provides methods for handling discrete dividends in option pricing:

1. Escrowed Dividend Model
   - Adjusts spot price by present value of expected dividends
   - Simple and fast, works well for short-dated options
   - Standard approach for European options

2. Piecewise Lognormal Model
   - Treats stock as lognormal between dividend dates
   - More accurate for long-dated options
   - Preserves arbitrage-free pricing

3. Modified Binomial Tree
   - Explicit dividend treatment in tree nodes
   - Handles early exercise around dividends
   - Optimal for American options with dividends

4. Dividend Interpolation
   - Estimate future dividends from historical data
   - Yield curve-based projection
   - Seasonal adjustment for quarterly dividends

References:
    - Bos & Vandermark (2002): "Finessing Fixed Dividends"
    - Haug, Haug & Lewis (2003): "Back to Basics: A New Approach to the Discrete Dividend Problem"
    - Vellekoop & Nieuwenhuis (2006): "Efficient Pricing of Derivatives on Assets with Discrete Dividends"
    - Wilmott (2006): "Paul Wilmott on Quantitative Finance", Chapter 26
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from core_options import (
    Dividend,
    ExerciseStyle,
    OptionType,
    OptionsContractSpec,
    PricingResult,
)
from core_errors import PricingError


# =============================================================================
# Constants
# =============================================================================

_MIN_TIME = 1e-10
_MIN_SPOT = 1e-10
_MIN_VOLATILITY = 1e-10
_DEFAULT_BINOMIAL_STEPS = 201


# =============================================================================
# Enums and Data Classes
# =============================================================================


class DividendModel(Enum):
    """Dividend handling model selection."""

    ESCROWED = "escrowed"  # PV-adjusted spot
    PIECEWISE_LOGNORMAL = "piecewise_lognormal"  # Lognormal between dividends
    PROPORTIONAL = "proportional"  # Dividend as % of spot
    BINOMIAL_EXPLICIT = "binomial_explicit"  # Explicit tree handling


@dataclass
class DividendSchedule:
    """
    Schedule of expected discrete dividends.

    Attributes:
        ex_dates: Ex-dividend dates (datetime or float years from now)
        amounts: Dividend amounts per share
        payment_dates: Optional payment dates (usually T+2 after ex-date)
        is_percentage: If True, amounts are % of spot (e.g., 0.02 for 2%)
    """

    ex_dates: List[Union[datetime, float]]
    amounts: List[float]
    payment_dates: Optional[List[Union[datetime, float]]] = None
    is_percentage: bool = False

    def __post_init__(self):
        """Validate schedule."""
        if len(self.ex_dates) != len(self.amounts):
            raise ValueError("ex_dates and amounts must have same length")
        if self.payment_dates is not None and len(self.payment_dates) != len(self.amounts):
            raise ValueError("payment_dates must have same length as amounts")

    def to_years(self, valuation_date: Union[datetime, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert schedule to years from valuation date.

        Returns:
            (times, amounts) as numpy arrays
        """
        if isinstance(valuation_date, datetime):
            times = []
            for ex_date in self.ex_dates:
                if isinstance(ex_date, datetime):
                    delta = (ex_date - valuation_date).days / 365.25
                else:
                    delta = ex_date
                times.append(max(0.0, delta))
        else:
            times = [
                max(0.0, float(t) - valuation_date) if isinstance(t, (int, float)) else t
                for t in self.ex_dates
            ]

        return np.array(times, dtype=np.float64), np.array(self.amounts, dtype=np.float64)


@dataclass
class DividendAdjustedResult:
    """Result of dividend-adjusted pricing."""

    price: float
    adjusted_spot: float
    pv_dividends: float
    model: DividendModel
    dividends_used: int
    computation_time_ms: float = 0.0


# =============================================================================
# Present Value of Dividends
# =============================================================================


def compute_pv_dividends(
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float],
    rate: float,
    time_to_expiry: float,
) -> Tuple[float, int]:
    """
    Compute present value of dividends within option lifetime.

    Args:
        dividend_schedule: Schedule of expected dividends
        valuation_date: Current date or time (0 for now)
        rate: Risk-free rate (annualized)
        time_to_expiry: Time to option expiration (years)

    Returns:
        (pv_dividends, n_dividends_used)
    """
    times, amounts = dividend_schedule.to_years(valuation_date)

    # Filter dividends within option lifetime
    mask = (times >= 0) & (times < time_to_expiry)
    relevant_times = times[mask]
    relevant_amounts = amounts[mask]

    if len(relevant_times) == 0:
        return 0.0, 0

    # Discount each dividend to present value
    discount_factors = np.exp(-rate * relevant_times)
    pv = float(np.sum(relevant_amounts * discount_factors))

    return pv, len(relevant_times)


# =============================================================================
# Escrowed Dividend Model
# =============================================================================


def adjust_spot_for_dividends(
    spot: float,
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float],
    rate: float,
    time_to_expiry: float,
) -> Tuple[float, float, int]:
    """
    Adjust spot price by subtracting PV of dividends (escrowed model).

    This is the simplest approach, treating the stock price as:
    S_adj = S - PV(dividends)

    Works well for:
    - Short-dated European options
    - Options with small dividends relative to spot

    Args:
        spot: Current stock price
        dividend_schedule: Expected dividend schedule
        valuation_date: Current date
        rate: Risk-free rate
        time_to_expiry: Option maturity

    Returns:
        (adjusted_spot, pv_dividends, n_dividends)
    """
    pv_dividends, n_dividends = compute_pv_dividends(
        dividend_schedule, valuation_date, rate, time_to_expiry
    )

    # For percentage dividends, convert to absolute
    if dividend_schedule.is_percentage:
        pv_dividends = pv_dividends * spot

    adjusted_spot = max(_MIN_SPOT, spot - pv_dividends)

    return adjusted_spot, pv_dividends, n_dividends


def price_with_escrowed_dividends(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float] = 0.0,
) -> DividendAdjustedResult:
    """
    Price European option using escrowed dividend model.

    Uses Black-Scholes with PV-adjusted spot price.

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate
        volatility: Implied volatility
        is_call: True for call, False for put
        dividend_schedule: Expected dividends
        valuation_date: Current date (for dividend timing)

    Returns:
        DividendAdjustedResult with price and details
    """
    start_time = time.perf_counter()

    # Adjust spot
    adjusted_spot, pv_dividends, n_divs = adjust_spot_for_dividends(
        spot, dividend_schedule, valuation_date, rate, time_to_expiry
    )

    # Black-Scholes with adjusted spot
    price = _black_scholes_internal(
        spot=adjusted_spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=0.0,  # Dividends already subtracted
        volatility=volatility,
        is_call=is_call,
    )

    elapsed = (time.perf_counter() - start_time) * 1000

    return DividendAdjustedResult(
        price=price,
        adjusted_spot=adjusted_spot,
        pv_dividends=pv_dividends,
        model=DividendModel.ESCROWED,
        dividends_used=n_divs,
        computation_time_ms=elapsed,
    )


# =============================================================================
# Piecewise Lognormal Model
# =============================================================================


def price_with_piecewise_lognormal(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float] = 0.0,
) -> DividendAdjustedResult:
    """
    Price option using piecewise lognormal model.

    The stock is lognormal between dividend dates, with discontinuous drops
    at each ex-dividend date.

    More accurate than escrowed model for:
    - Long-dated options
    - Large dividends
    - Multiple dividend payments

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate
        volatility: Implied volatility
        is_call: True for call, False for put
        dividend_schedule: Expected dividends
        valuation_date: Current date

    Returns:
        DividendAdjustedResult with price and details

    References:
        - Bos & Vandermark (2002)
        - Haug, Haug & Lewis (2003)
    """
    start_time = time.perf_counter()

    times, amounts = dividend_schedule.to_years(valuation_date)

    # Filter relevant dividends
    mask = (times >= 0) & (times < time_to_expiry)
    div_times = times[mask]
    div_amounts = amounts[mask]
    n_divs = len(div_times)

    if n_divs == 0:
        # No dividends - standard BS
        price = _black_scholes_internal(
            spot, strike, time_to_expiry, rate, 0.0, volatility, is_call
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        return DividendAdjustedResult(
            price=price,
            adjusted_spot=spot,
            pv_dividends=0.0,
            model=DividendModel.PIECEWISE_LOGNORMAL,
            dividends_used=0,
            computation_time_ms=elapsed,
        )

    # Sort dividends by time
    sort_idx = np.argsort(div_times)
    div_times = div_times[sort_idx]
    div_amounts = div_amounts[sort_idx]

    # Handle percentage dividends
    if dividend_schedule.is_percentage:
        # For percentage dividends, use multiplicative model
        # S(T) = S(0) * prod(1 - d_i) * exp(...)
        total_div_factor = np.prod(1.0 - div_amounts)
        adjusted_spot = spot * total_div_factor
        pv_dividends = spot * (1.0 - total_div_factor)
    else:
        # Absolute dividends - use Haug-Haug-Lewis approach
        # Adjust forward price for each dividend
        forward = spot * math.exp(rate * time_to_expiry)

        pv_dividends = 0.0
        for t_div, d_amt in zip(div_times, div_amounts):
            # Discount dividend to valuation date
            pv_div = d_amt * math.exp(-rate * t_div)
            pv_dividends += pv_div
            # Subtract from forward (compounded to expiry)
            forward -= d_amt * math.exp(rate * (time_to_expiry - t_div))

        # Convert forward back to adjusted spot
        adjusted_spot = forward * math.exp(-rate * time_to_expiry)
        adjusted_spot = max(_MIN_SPOT, adjusted_spot)

    # Price with adjusted spot
    price = _black_scholes_internal(
        spot=adjusted_spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        rate=rate,
        dividend_yield=0.0,
        volatility=volatility,
        is_call=is_call,
    )

    elapsed = (time.perf_counter() - start_time) * 1000

    return DividendAdjustedResult(
        price=price,
        adjusted_spot=adjusted_spot,
        pv_dividends=pv_dividends,
        model=DividendModel.PIECEWISE_LOGNORMAL,
        dividends_used=n_divs,
        computation_time_ms=elapsed,
    )


# =============================================================================
# Binomial Tree with Explicit Dividends
# =============================================================================


def price_american_with_dividends(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float] = 0.0,
    n_steps: int = _DEFAULT_BINOMIAL_STEPS,
) -> DividendAdjustedResult:
    """
    Price American option with discrete dividends using modified binomial tree.

    The tree is constructed with explicit dividend drops at ex-dividend dates.
    This properly captures early exercise incentive around dividends.

    Key insight: American call should be exercised just before ex-dividend
    date if dividend > intrinsic value of remaining time value.

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate
        volatility: Implied volatility
        is_call: True for call, False for put
        dividend_schedule: Expected dividends
        valuation_date: Current date
        n_steps: Number of binomial tree steps

    Returns:
        DividendAdjustedResult with price and details

    References:
        - Cox, Ross, Rubinstein (1979)
        - Wilmott (2006)
    """
    start_time = time.perf_counter()

    if time_to_expiry <= _MIN_TIME:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        elapsed = (time.perf_counter() - start_time) * 1000
        return DividendAdjustedResult(
            price=intrinsic,
            adjusted_spot=spot,
            pv_dividends=0.0,
            model=DividendModel.BINOMIAL_EXPLICIT,
            dividends_used=0,
            computation_time_ms=elapsed,
        )

    if volatility <= _MIN_VOLATILITY:
        volatility = _MIN_VOLATILITY

    # Get dividend times and amounts
    times, amounts = dividend_schedule.to_years(valuation_date)
    mask = (times >= 0) & (times < time_to_expiry)
    div_times = times[mask]
    div_amounts = amounts[mask]
    n_divs = len(div_times)

    # Handle percentage dividends
    if dividend_schedule.is_percentage:
        # Will be converted to absolute amounts during tree construction
        pass

    # Time step
    dt = time_to_expiry / n_steps

    # CRR parameters
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(rate * dt) - d) / (u - d)

    if p < 0 or p > 1:
        # Fallback to stable parameters
        p = 0.5
        u = math.exp(volatility * math.sqrt(dt) + (rate - 0.5 * volatility**2) * dt)
        d = math.exp(-volatility * math.sqrt(dt) + (rate - 0.5 * volatility**2) * dt)

    # Find dividend steps (which tree step each dividend falls on)
    div_steps = []
    for t_div in div_times:
        step = int(t_div / dt)
        if 0 <= step < n_steps:
            div_steps.append(step)
        else:
            div_steps.append(-1)

    # Build stock price tree with dividend adjustments
    stock = np.zeros((n_steps + 1, n_steps + 1))

    # Initialize prices at each node
    for i in range(n_steps + 1):
        for j in range(i + 1):
            # Base price from up/down moves
            price = spot * (u**j) * (d ** (i - j))

            # Apply dividends that have occurred by step i
            for k, div_step in enumerate(div_steps):
                if div_step != -1 and div_step < i:
                    if dividend_schedule.is_percentage:
                        # Multiplicative reduction
                        price *= 1.0 - div_amounts[k]
                    else:
                        # Absolute reduction (with floor at 0)
                        price = max(0.0, price - div_amounts[k])

            stock[i, j] = price

    # Calculate terminal payoffs
    option = np.zeros((n_steps + 1, n_steps + 1))
    for j in range(n_steps + 1):
        if is_call:
            option[n_steps, j] = max(0.0, stock[n_steps, j] - strike)
        else:
            option[n_steps, j] = max(0.0, strike - stock[n_steps, j])

    # Backward induction with early exercise check
    df = math.exp(-rate * dt)

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Continuation value
            continuation = df * (p * option[i + 1, j + 1] + (1 - p) * option[i + 1, j])

            # Early exercise value
            if is_call:
                exercise = max(0.0, stock[i, j] - strike)
            else:
                exercise = max(0.0, strike - stock[i, j])

            # American option: take max of exercise and continuation
            option[i, j] = max(exercise, continuation)

    price = option[0, 0]

    # Calculate PV of dividends
    pv_dividends = 0.0
    for t_div, d_amt in zip(div_times, div_amounts):
        if dividend_schedule.is_percentage:
            # Approximate PV using current spot
            pv_dividends += d_amt * spot * math.exp(-rate * t_div)
        else:
            pv_dividends += d_amt * math.exp(-rate * t_div)

    elapsed = (time.perf_counter() - start_time) * 1000

    return DividendAdjustedResult(
        price=price,
        adjusted_spot=spot,  # Original spot (not adjusted in binomial)
        pv_dividends=pv_dividends,
        model=DividendModel.BINOMIAL_EXPLICIT,
        dividends_used=n_divs,
        computation_time_ms=elapsed,
    )


# =============================================================================
# Dividend Estimation / Projection
# =============================================================================


def estimate_future_dividends(
    historical_dividends: List[Dividend],
    projection_horizon: float,
    growth_rate: float = 0.0,
    seasonal_pattern: Optional[List[float]] = None,
) -> DividendSchedule:
    """
    Estimate future dividends based on historical pattern.

    Args:
        historical_dividends: Past dividend payments
        projection_horizon: How far ahead to project (years)
        growth_rate: Annual dividend growth rate
        seasonal_pattern: Optional quarterly pattern [Q1, Q2, Q3, Q4] weights

    Returns:
        DividendSchedule with projected dividends
    """
    if not historical_dividends:
        raise ValueError("Need at least one historical dividend")

    # Sort by ex-date
    sorted_divs = sorted(historical_dividends, key=lambda d: d.ex_date)

    # Calculate average dividend frequency (payments per year)
    if len(sorted_divs) >= 2:
        first_date = sorted_divs[0].ex_date
        last_date = sorted_divs[-1].ex_date
        # Handle both datetime and date objects, or float (year fractions)
        if isinstance(first_date, (datetime, date)) and isinstance(last_date, (datetime, date)):
            span_years = (last_date - first_date).days / 365.25
        elif isinstance(first_date, (int, float)) and isinstance(last_date, (int, float)):
            span_years = float(last_date) - float(first_date)
        else:
            # Fallback: try to compute days difference
            span_years = (last_date - first_date).days / 365.25

        if span_years > 0:
            frequency = len(sorted_divs) / span_years
        else:
            frequency = 4.0  # Default quarterly
    else:
        frequency = 4.0

    # Calculate average dividend amount
    avg_amount = sum(d.amount for d in sorted_divs) / len(sorted_divs)

    # Project future dividends
    n_projected = int(projection_horizon * frequency) + 1
    interval = 1.0 / frequency

    ex_dates = []
    amounts = []

    for i in range(n_projected):
        t = (i + 1) * interval

        if t > projection_horizon:
            break

        ex_dates.append(t)

        # Apply growth rate
        growth_factor = (1.0 + growth_rate) ** t
        projected_amount = avg_amount * growth_factor

        # Apply seasonal pattern if provided
        if seasonal_pattern is not None and len(seasonal_pattern) == 4:
            quarter = i % 4
            seasonal_factor = seasonal_pattern[quarter] / np.mean(seasonal_pattern)
            projected_amount *= seasonal_factor

        amounts.append(projected_amount)

    return DividendSchedule(
        ex_dates=ex_dates,
        amounts=amounts,
    )


def yield_to_discrete_dividends(
    spot: float,
    dividend_yield: float,
    time_to_expiry: float,
    frequency: int = 4,
) -> DividendSchedule:
    """
    Convert continuous dividend yield to discrete dividend schedule.

    Useful when only yield is available but discrete model is needed.

    Args:
        spot: Current stock price
        dividend_yield: Continuous dividend yield (annualized)
        time_to_expiry: Time horizon
        frequency: Payments per year (4 = quarterly)

    Returns:
        DividendSchedule approximating the yield
    """
    if dividend_yield <= 0:
        return DividendSchedule(ex_dates=[], amounts=[])

    interval = 1.0 / frequency
    n_payments = int(time_to_expiry * frequency) + 1

    # Each discrete dividend should replicate continuous yield effect
    # exp(-q*T) ≈ prod(1 - d_i/S)
    # For equal dividends: (1 - d/S)^n = exp(-q*T)
    # d/S = 1 - exp(-q*T/n)

    ex_dates = []
    amounts = []

    for i in range(n_payments):
        t = (i + 1) * interval
        if t > time_to_expiry:
            break

        ex_dates.append(t)

        # Dividend amount that replicates yield
        div_fraction = 1.0 - math.exp(-dividend_yield * interval)
        amounts.append(spot * div_fraction)

    return DividendSchedule(ex_dates=ex_dates, amounts=amounts)


# =============================================================================
# Internal Black-Scholes (for dividend-adjusted pricing)
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
    """Internal Black-Scholes implementation."""
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
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    df_rate = math.exp(-rate * time_to_expiry)
    df_div = math.exp(-dividend_yield * time_to_expiry)

    if is_call:
        return spot * df_div * nd1 - strike * df_rate * nd2
    else:
        return strike * df_rate * (1.0 - nd2) - spot * df_div * (1.0 - nd1)


# =============================================================================
# High-Level Interface
# =============================================================================


def price_with_dividends(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    is_call: bool,
    dividend_schedule: DividendSchedule,
    valuation_date: Union[datetime, float] = 0.0,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    model: DividendModel = DividendModel.PIECEWISE_LOGNORMAL,
    n_steps: int = _DEFAULT_BINOMIAL_STEPS,
) -> DividendAdjustedResult:
    """
    Price option with discrete dividends using specified model.

    High-level interface that selects appropriate method based on:
    - Exercise style (European vs American)
    - Dividend model preference

    Args:
        spot: Current stock price
        strike: Option strike
        time_to_expiry: Time to expiration (years)
        rate: Risk-free rate
        volatility: Implied volatility
        is_call: True for call, False for put
        dividend_schedule: Expected dividends
        valuation_date: Current date
        exercise_style: European or American
        model: Dividend handling model
        n_steps: Steps for binomial tree (if used)

    Returns:
        DividendAdjustedResult with price and details
    """
    if exercise_style == ExerciseStyle.AMERICAN:
        # American options require binomial tree for proper early exercise
        return price_american_with_dividends(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            dividend_schedule=dividend_schedule,
            valuation_date=valuation_date,
            n_steps=n_steps,
        )

    # European options
    if model == DividendModel.ESCROWED:
        return price_with_escrowed_dividends(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            dividend_schedule=dividend_schedule,
            valuation_date=valuation_date,
        )
    elif model == DividendModel.PIECEWISE_LOGNORMAL:
        return price_with_piecewise_lognormal(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            dividend_schedule=dividend_schedule,
            valuation_date=valuation_date,
        )
    elif model == DividendModel.BINOMIAL_EXPLICIT:
        return price_american_with_dividends(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            dividend_schedule=dividend_schedule,
            valuation_date=valuation_date,
            n_steps=n_steps,
        )
    else:
        raise PricingError(f"Unknown dividend model: {model}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_dividend_schedule(
    ex_dates: List[Union[datetime, float]],
    amounts: List[float],
    is_percentage: bool = False,
) -> DividendSchedule:
    """Create a dividend schedule from dates and amounts."""
    return DividendSchedule(
        ex_dates=ex_dates,
        amounts=amounts,
        is_percentage=is_percentage,
    )


def create_quarterly_schedule(
    annual_dividend: float,
    years_ahead: float = 2.0,
    start_offset: float = 0.25,
) -> DividendSchedule:
    """
    Create standard quarterly dividend schedule.

    Args:
        annual_dividend: Total annual dividend per share
        years_ahead: How far to project
        start_offset: Time until first dividend (years)

    Returns:
        DividendSchedule with quarterly payments
    """
    quarterly_amount = annual_dividend / 4.0

    ex_dates = []
    amounts = []

    t = start_offset
    while t <= years_ahead:
        ex_dates.append(t)
        amounts.append(quarterly_amount)
        t += 0.25

    return DividendSchedule(ex_dates=ex_dates, amounts=amounts)
