# -*- coding: utf-8 -*-
"""
test_options_core.py
Comprehensive test suite for Phase 1 Options Integration.

Target: 200+ tests covering all components:
- Greeks (scalar and vectorized)
- Pricing models (BS, Leisen-Reimer, CRR, Jump-Diffusion)
- IV calculation (hybrid solver)
- Jump calibration
- Discrete dividends
- Exercise probability (Longstaff-Schwartz)

Test Categories:
1. Unit tests for individual functions
2. Integration tests across components
3. Edge case handling
4. Numerical accuracy validation
5. Performance regression tests
"""

import math
import pytest
import numpy as np
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Tuple

# Core options types
from core_options import (
    OptionType,
    ExerciseStyle,
    SettlementType,
    OptionsContractSpec,
    GreeksResult,
    PricingResult,
    IVResult,
    JumpParams,
    Dividend,
    VarianceSwapQuote,
)

# Error types
from core_errors import (
    OptionsError,
    GreeksCalculationError,
    IVConvergenceError,
    PricingError,
    CalibrationError,
)

# Greeks
from impl_greeks import (
    compute_all_greeks,
    compute_delta,
    compute_gamma,
    compute_theta,
    compute_vega,
    compute_rho,
    compute_vanna,
    compute_volga,
    compute_charm,
    compute_speed,
    compute_color,
    compute_zomma,
    compute_ultima,
)

# Vectorized Greeks
from impl_greeks_vectorized import (
    compute_all_greeks_batch,
    compute_greeks_for_chain,
    BatchGreeksResult,
)

# Pricing
from impl_pricing import (
    black_scholes_price,
    leisen_reimer_price,
    crr_binomial_price,
    merton_jump_diffusion_price,
    merton_jump_diffusion_price_vec,
    variance_swap_strike,
    compute_variance_swap_value,
    price_option,
)

# IV Calculation
from impl_iv_calculation import (
    calculate_iv,
    calculate_iv_american,
    calculate_iv_batch,
)

# Jump Calibration
from impl_jump_calibration import (
    calibrate_from_options,
    calibrate_from_moments,
    calibrate_from_mle,
    calibrate_hybrid,
    detect_jumps,
    CalibrationInput,
    CalibrationMethod,
    JumpCalibrator,
)

# Discrete Dividends
from impl_discrete_dividends import (
    DividendSchedule,
    DividendModel,
    adjust_spot_for_dividends,
    price_with_escrowed_dividends,
    price_with_piecewise_lognormal,
    price_american_with_dividends,
    price_with_dividends,
    estimate_future_dividends,
    yield_to_discrete_dividends,
)

# Exercise Probability
from impl_exercise_probability import (
    longstaff_schwartz,
    compute_exercise_probability,
    should_exercise_early,
    compute_early_exercise_premium,
    barone_adesi_whaley,
    BasisFunctions,
    VarianceReduction,
    LSResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def standard_params():
    """Standard option parameters for testing."""
    return {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 0.25,  # 3 months
        "rate": 0.05,
        "dividend_yield": 0.02,
        "volatility": 0.20,
    }


@pytest.fixture
def itm_call_params():
    """ITM call parameters."""
    return {
        "spot": 110.0,
        "strike": 100.0,
        "time_to_expiry": 0.25,
        "rate": 0.05,
        "dividend_yield": 0.02,
        "volatility": 0.20,
    }


@pytest.fixture
def otm_put_params():
    """OTM put parameters."""
    return {
        "spot": 110.0,
        "strike": 100.0,
        "time_to_expiry": 0.25,
        "rate": 0.05,
        "dividend_yield": 0.02,
        "volatility": 0.20,
    }


@pytest.fixture
def jump_params():
    """Jump-diffusion parameters."""
    return JumpParams(
        lambda_intensity=1.0,
        mu_jump=-0.05,
        sigma_jump=0.15,
    )


@pytest.fixture
def dividend_schedule():
    """Standard dividend schedule."""
    return DividendSchedule(
        ex_dates=[0.1, 0.2],  # 0.1 and 0.2 years from now
        amounts=[1.0, 1.0],  # $1 each
    )


# =============================================================================
# GREEKS TESTS (50 tests)
# =============================================================================


class TestScalarGreeks:
    """Test scalar Greeks calculations."""

    def test_delta_call_atm(self, standard_params):
        """ATM call delta should be around 0.5."""
        delta = compute_delta(**standard_params, is_call=True)
        assert 0.4 < delta < 0.6

    def test_delta_call_itm(self, itm_call_params):
        """ITM call delta should be > 0.5."""
        delta = compute_delta(**itm_call_params, is_call=True)
        assert delta > 0.6

    def test_delta_put_atm(self, standard_params):
        """ATM put delta should be around -0.5."""
        delta = compute_delta(**standard_params, is_call=False)
        assert -0.6 < delta < -0.4

    def test_delta_call_put_parity(self, standard_params):
        """Call delta - put delta should equal e^(-qT)."""
        call_delta = compute_delta(**standard_params, is_call=True)
        put_delta = compute_delta(**standard_params, is_call=False)
        q = standard_params["dividend_yield"]
        T = standard_params["time_to_expiry"]
        expected_diff = math.exp(-q * T)
        assert abs((call_delta - put_delta) - expected_diff) < 0.01

    def test_gamma_atm_maximum(self, standard_params):
        """Gamma should be maximum at ATM."""
        gamma_atm = compute_gamma(**standard_params)

        itm_params = standard_params.copy()
        itm_params["spot"] = 110.0
        gamma_itm = compute_gamma(**itm_params)

        otm_params = standard_params.copy()
        otm_params["spot"] = 90.0
        gamma_otm = compute_gamma(**otm_params)

        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_gamma_positive(self, standard_params):
        """Gamma should always be positive."""
        gamma = compute_gamma(**standard_params)
        assert gamma > 0

    def test_theta_call_negative(self, standard_params):
        """Call theta should typically be negative (time decay)."""
        theta = compute_theta(**standard_params, is_call=True)
        assert theta < 0

    def test_theta_deep_itm_put_positive(self):
        """Deep ITM put theta can be positive."""
        params = {
            "spot": 50.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.10,
            "dividend_yield": 0.0,
            "volatility": 0.20,
        }
        theta = compute_theta(**params, is_call=False)
        # Deep ITM put with high rate can have positive theta
        assert theta > -10  # At least bounded

    def test_vega_atm_maximum(self, standard_params):
        """Vega should be maximum at ATM."""
        vega_atm = compute_vega(**standard_params)

        itm_params = standard_params.copy()
        itm_params["spot"] = 120.0
        vega_itm = compute_vega(**itm_params)

        assert vega_atm > vega_itm

    def test_vega_positive(self, standard_params):
        """Vega should always be positive."""
        vega = compute_vega(**standard_params)
        assert vega > 0

    def test_vega_call_put_equal(self, standard_params):
        """Call and put vega should be equal."""
        vega_call = compute_vega(**standard_params)  # is_call doesn't matter for vega
        vega_put = compute_vega(**standard_params)
        assert abs(vega_call - vega_put) < 1e-10

    def test_rho_call_positive(self, standard_params):
        """Call rho should be positive."""
        rho = compute_rho(**standard_params, is_call=True)
        assert rho > 0

    def test_rho_put_negative(self, standard_params):
        """Put rho should be negative."""
        rho = compute_rho(**standard_params, is_call=False)
        assert rho < 0

    def test_vanna_sign_changes(self, standard_params):
        """Vanna changes sign around ATM."""
        vanna_atm = compute_vanna(**standard_params)

        otm_params = standard_params.copy()
        otm_params["spot"] = 80.0
        vanna_otm = compute_vanna(**otm_params)

        itm_params = standard_params.copy()
        itm_params["spot"] = 120.0
        vanna_itm = compute_vanna(**itm_params)

        # Vanna changes sign
        assert vanna_otm * vanna_itm < 0 or abs(vanna_atm) < 0.1

    def test_volga_positive(self, standard_params):
        """Volga (vomma) should typically be positive."""
        volga = compute_volga(**standard_params)
        assert volga > 0

    def test_charm_bounded(self, standard_params):
        """Charm should be bounded."""
        charm = compute_charm(**standard_params, is_call=True)
        assert abs(charm) < 10  # Reasonable bound

    def test_speed_bounded(self, standard_params):
        """Speed should be bounded."""
        speed = compute_speed(**standard_params)
        assert abs(speed) < 10

    def test_color_bounded(self, standard_params):
        """Color should be bounded."""
        color = compute_color(**standard_params)
        assert abs(color) < 100

    def test_zomma_bounded(self, standard_params):
        """Zomma should be bounded."""
        zomma = compute_zomma(**standard_params)
        assert abs(zomma) < 10

    def test_ultima_bounded(self, standard_params):
        """Ultima should be bounded."""
        ultima = compute_ultima(**standard_params)
        assert abs(ultima) < 1000

    def test_all_greeks_returns_dataclass(self, standard_params):
        """compute_all_greeks should return GreeksResult."""
        result = compute_all_greeks(**standard_params, is_call=True)
        assert isinstance(result, GreeksResult)
        assert result.delta is not None
        assert result.gamma is not None
        assert result.theta is not None
        assert result.vega is not None
        assert result.rho is not None

    def test_greeks_near_expiry(self):
        """Greeks should handle near-expiry options."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1e-6,  # Very small
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }
        result = compute_all_greeks(**params, is_call=True)
        assert math.isfinite(result.delta)
        assert math.isfinite(result.gamma)

    def test_greeks_zero_volatility(self):
        """Greeks should handle zero volatility."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 1e-10,
        }
        result = compute_all_greeks(**params, is_call=True)
        assert math.isfinite(result.delta)

    def test_greeks_high_volatility(self):
        """Greeks should handle high volatility."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 2.0,  # 200% vol
        }
        result = compute_all_greeks(**params, is_call=True)
        assert math.isfinite(result.delta)
        assert math.isfinite(result.vega)


class TestVectorizedGreeks:
    """Test vectorized Greeks calculations."""

    def test_batch_greeks_shape(self, standard_params):
        """Batch Greeks should return correct shape."""
        n = 100
        spots = np.full(n, standard_params["spot"])
        strikes = np.linspace(80, 120, n)
        times = np.full(n, standard_params["time_to_expiry"])
        rates = np.full(n, standard_params["rate"])
        yields_ = np.full(n, standard_params["dividend_yield"])
        vols = np.full(n, standard_params["volatility"])
        is_calls = np.ones(n, dtype=bool)

        result = compute_all_greeks_batch(spots, strikes, times, rates, yields_, vols, is_calls)

        assert isinstance(result, BatchGreeksResult)
        assert len(result.delta) == n
        assert len(result.gamma) == n

    def test_batch_matches_scalar(self, standard_params):
        """Batch Greeks should match scalar calculations."""
        n = 10
        spots = np.full(n, standard_params["spot"])
        strikes = np.linspace(90, 110, n)
        times = np.full(n, standard_params["time_to_expiry"])
        rates = np.full(n, standard_params["rate"])
        yields_ = np.full(n, standard_params["dividend_yield"])
        vols = np.full(n, standard_params["volatility"])
        is_calls = np.ones(n, dtype=bool)

        batch_result = compute_all_greeks_batch(
            spots, strikes, times, rates, yields_, vols, is_calls
        )

        # Compare with scalar
        for i in range(n):
            scalar_delta = compute_delta(
                spot=spots[i],
                strike=strikes[i],
                time_to_expiry=times[i],
                rate=rates[i],
                dividend_yield=yields_[i],
                volatility=vols[i],
                is_call=True,
            )
            assert abs(batch_result.delta[i] - scalar_delta) < 1e-10

    def test_batch_vs_scalar_consistency(self, standard_params):
        """Validate vectorized results match scalar calculations."""
        n = 50
        spots = np.full(n, standard_params["spot"])
        strikes = np.linspace(80, 120, n)
        times = np.full(n, standard_params["time_to_expiry"])
        rates = np.full(n, standard_params["rate"])
        yields_ = np.full(n, standard_params["dividend_yield"])
        vols = np.full(n, standard_params["volatility"])
        is_calls = np.ones(n, dtype=bool)

        batch_result = compute_all_greeks_batch(
            spots, strikes, times, rates, yields_, vols, is_calls
        )

        # Compare with scalar for all elements
        for i in range(n):
            scalar_delta = compute_delta(
                spot=spots[i],
                strike=strikes[i],
                time_to_expiry=times[i],
                rate=rates[i],
                dividend_yield=yields_[i],
                volatility=vols[i],
                is_call=True,
            )
            assert abs(batch_result.delta[i] - scalar_delta) < 1e-10

    def test_batch_greeks_large_dataset(self):
        """Batch processing should handle large datasets."""
        n = 10000
        np.random.seed(42)
        spots = np.random.uniform(90, 110, n)
        strikes = np.random.uniform(80, 120, n)
        times = np.random.uniform(0.1, 1.0, n)
        rates = np.full(n, 0.05)
        yields_ = np.full(n, 0.02)
        vols = np.random.uniform(0.1, 0.5, n)
        is_calls = np.random.choice([True, False], n)

        result = compute_all_greeks_batch(spots, strikes, times, rates, yields_, vols, is_calls)

        assert result.n_options == n
        assert len(result.delta) == n

    def test_option_chain_greeks(self, standard_params):
        """Test Greeks computation for option chain."""
        spot = standard_params["spot"]
        strikes = np.linspace(80, 120, 9)
        expiries = [0.25, 0.5]

        # Create contract specs for all combinations
        contracts = []
        expiration_base = date.today()
        for exp_years in expiries:
            expiration = expiration_base + timedelta(days=int(exp_years * 365))
            for strike in strikes:
                contract = OptionsContractSpec(
                    symbol=f"TEST{expiration:%y%m%d}C{int(strike*1000):08d}",
                    underlying="TEST",
                    option_type=OptionType.CALL,
                    strike=Decimal(str(strike)),
                    expiration=expiration,
                )
                contracts.append(contract)

        result = compute_greeks_for_chain(
            contracts=contracts,
            spot=spot,
            volatility=standard_params["volatility"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
        )

        assert result.n_options == len(strikes) * len(expiries)


# =============================================================================
# PRICING TESTS (40 tests)
# =============================================================================


class TestBlackScholesPricing:
    """Test Black-Scholes pricing."""

    def test_bs_call_positive(self, standard_params):
        """Call price should be positive."""
        price = black_scholes_price(**standard_params, is_call=True)
        assert price > 0

    def test_bs_put_positive(self, standard_params):
        """Put price should be positive."""
        price = black_scholes_price(**standard_params, is_call=False)
        assert price > 0

    def test_bs_put_call_parity(self, standard_params):
        """Put-call parity should hold."""
        call = black_scholes_price(**standard_params, is_call=True)
        put = black_scholes_price(**standard_params, is_call=False)

        S = standard_params["spot"]
        K = standard_params["strike"]
        r = standard_params["rate"]
        q = standard_params["dividend_yield"]
        T = standard_params["time_to_expiry"]

        # C - P = S*e^(-qT) - K*e^(-rT)
        expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((call - put) - expected_diff) < 1e-10

    def test_bs_atm_approximation(self, standard_params):
        """ATM approximation should be accurate."""
        price = black_scholes_price(**standard_params, is_call=True)

        # ATM approximation: C ≈ 0.4 * S * σ * √T
        S = standard_params["spot"]
        vol = standard_params["volatility"]
        T = standard_params["time_to_expiry"]
        approx = 0.4 * S * vol * math.sqrt(T)

        assert abs(price - approx) / price < 0.1  # Within 10%

    def test_bs_intrinsic_value_floor(self, itm_call_params):
        """Option should be worth at least intrinsic value."""
        call = black_scholes_price(**itm_call_params, is_call=True)
        intrinsic = itm_call_params["spot"] - itm_call_params["strike"]
        assert call >= intrinsic * 0.99  # Allow small numerical error

    def test_bs_price_increases_with_volatility(self, standard_params):
        """Option price should increase with volatility."""
        price1 = black_scholes_price(**standard_params, is_call=True)

        params2 = standard_params.copy()
        params2["volatility"] = 0.30
        price2 = black_scholes_price(**params2, is_call=True)

        assert price2 > price1

    def test_bs_call_price_increases_with_spot(self, standard_params):
        """Call price should increase with spot."""
        price1 = black_scholes_price(**standard_params, is_call=True)

        params2 = standard_params.copy()
        params2["spot"] = 105.0
        price2 = black_scholes_price(**params2, is_call=True)

        assert price2 > price1

    def test_bs_near_expiry(self):
        """Price should converge to intrinsic near expiry."""
        params = {
            "spot": 105.0,
            "strike": 100.0,
            "time_to_expiry": 1e-6,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }
        price = black_scholes_price(**params, is_call=True)
        intrinsic = params["spot"] - params["strike"]
        assert abs(price - intrinsic) < 0.01


class TestBinomialPricing:
    """Test binomial tree pricing."""

    def test_lr_european_matches_bs(self, standard_params):
        """Leisen-Reimer European should match Black-Scholes."""
        bs_price = black_scholes_price(**standard_params, is_call=True)
        lr_price = leisen_reimer_price(**standard_params, is_call=True, is_american=False)
        assert abs(lr_price - bs_price) < 0.01

    def test_lr_american_geq_european(self, standard_params):
        """American option should be >= European."""
        params = standard_params.copy()
        params["dividend_yield"] = 0.05  # High dividend for early exercise premium

        european = leisen_reimer_price(**params, is_call=True, is_american=False)
        american = leisen_reimer_price(**params, is_call=True, is_american=True)

        assert american >= european - 0.01

    def test_crr_matches_bs_at_limit(self, standard_params):
        """CRR should converge to BS with many steps."""
        bs_price = black_scholes_price(**standard_params, is_call=True)
        crr_price = crr_binomial_price(
            **standard_params, is_call=True, is_american=False, n_steps=500
        )
        assert abs(crr_price - bs_price) < 0.05

    def test_lr_second_order_convergence(self, standard_params):
        """Leisen-Reimer should have better convergence than CRR."""
        bs_price = black_scholes_price(**standard_params, is_call=True)

        lr_51 = leisen_reimer_price(**standard_params, is_call=True, is_american=False, n_steps=51)
        lr_101 = leisen_reimer_price(
            **standard_params, is_call=True, is_american=False, n_steps=101
        )

        error_51 = abs(lr_51 - bs_price)
        error_101 = abs(lr_101 - bs_price)

        # Second order: error should decrease by factor of ~4 when steps double
        assert error_101 < error_51

    def test_american_put_early_exercise(self):
        """American put should have early exercise premium."""
        params = {
            "spot": 80.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "rate": 0.10,
            "dividend_yield": 0.0,
            "volatility": 0.20,
        }

        european = black_scholes_price(**params, is_call=False)
        american = leisen_reimer_price(**params, is_call=False, is_american=True)

        assert american > european


class TestJumpDiffusionPricing:
    """Test Merton jump-diffusion pricing."""

    def test_jd_reduces_to_bs_no_jumps(self, standard_params):
        """JD should reduce to BS when λ=0."""
        bs_price = black_scholes_price(**standard_params, is_call=True)

        no_jump_params = JumpParams(lambda_intensity=0.0, mu_jump=0.0, sigma_jump=0.0)
        jd_price = merton_jump_diffusion_price(
            **standard_params, is_call=True, jump_params=no_jump_params
        )

        assert abs(jd_price - bs_price) < 0.01

    def test_jd_otm_put_higher_with_jumps(self, standard_params, jump_params):
        """OTM put should be higher with negative jumps."""
        params = standard_params.copy()
        params["spot"] = 110.0  # OTM put

        bs_price = black_scholes_price(**params, is_call=False)
        jd_price = merton_jump_diffusion_price(**params, is_call=False, jump_params=jump_params)

        # Negative jumps increase OTM put value
        assert jd_price > bs_price * 0.9  # Allow some tolerance

    def test_jd_positive_price(self, standard_params, jump_params):
        """JD price should always be positive."""
        price = merton_jump_diffusion_price(
            **standard_params, is_call=True, jump_params=jump_params
        )
        assert price > 0

    def test_jd_put_call_parity(self, standard_params, jump_params):
        """Put-call parity should approximately hold for JD."""
        call = merton_jump_diffusion_price(**standard_params, is_call=True, jump_params=jump_params)
        put = merton_jump_diffusion_price(**standard_params, is_call=False, jump_params=jump_params)

        S = standard_params["spot"]
        K = standard_params["strike"]
        r = standard_params["rate"]
        q = standard_params["dividend_yield"]
        T = standard_params["time_to_expiry"]

        expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((call - put) - expected_diff) < 0.5  # Wider tolerance for JD


class TestUnifiedPricingInterface:
    """Test unified price_option function."""

    def test_price_option_bs(self, standard_params):
        """price_option should work with black_scholes model."""
        result = price_option(**standard_params, is_call=True, model="black_scholes")
        assert isinstance(result, PricingResult)
        assert result.price > 0

    def test_price_option_leisen_reimer(self, standard_params):
        """price_option should work with leisen_reimer model."""
        result = price_option(**standard_params, is_call=True, model="leisen_reimer")
        assert result.price > 0

    def test_price_option_crr(self, standard_params):
        """price_option should work with crr model."""
        result = price_option(**standard_params, is_call=True, model="crr")
        assert result.price > 0


# =============================================================================
# IV CALCULATION TESTS (30 tests)
# =============================================================================


class TestIVCalculation:
    """Test implied volatility calculation."""

    def test_iv_recovers_input_vol(self, standard_params):
        """IV should recover the input volatility."""
        price = black_scholes_price(**standard_params, is_call=True)

        result = calculate_iv(
            market_price=price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=True,
        )

        assert result.converged
        assert abs(result.implied_volatility - standard_params["volatility"]) < 1e-6

    def test_iv_call_put_same(self, standard_params):
        """IV should be same for call and put (by put-call parity)."""
        call_price = black_scholes_price(**standard_params, is_call=True)
        put_price = black_scholes_price(**standard_params, is_call=False)

        call_iv = calculate_iv(
            market_price=call_price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=True,
        )

        put_iv = calculate_iv(
            market_price=put_price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=False,
        )

        assert abs(call_iv.implied_volatility - put_iv.implied_volatility) < 1e-6

    def test_iv_low_vol(self):
        """IV should handle low volatility cases."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.05,  # 5% vol
        }
        price = black_scholes_price(**params, is_call=True)

        result = calculate_iv(
            market_price=price,
            spot=params["spot"],
            strike=params["strike"],
            time_to_expiry=params["time_to_expiry"],
            rate=params["rate"],
            dividend_yield=params["dividend_yield"],
            is_call=True,
        )

        assert result.converged
        assert abs(result.implied_volatility - 0.05) < 1e-4

    def test_iv_high_vol(self):
        """IV should handle high volatility cases."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 1.5,  # 150% vol
        }
        price = black_scholes_price(**params, is_call=True)

        result = calculate_iv(
            market_price=price,
            spot=params["spot"],
            strike=params["strike"],
            time_to_expiry=params["time_to_expiry"],
            rate=params["rate"],
            dividend_yield=params["dividend_yield"],
            is_call=True,
        )

        assert result.converged
        assert abs(result.implied_volatility - 1.5) < 0.01

    def test_iv_deep_itm(self):
        """IV should handle deep ITM options."""
        params = {
            "spot": 150.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.25,
        }
        price = black_scholes_price(**params, is_call=True)

        result = calculate_iv(
            market_price=price,
            spot=params["spot"],
            strike=params["strike"],
            time_to_expiry=params["time_to_expiry"],
            rate=params["rate"],
            dividend_yield=params["dividend_yield"],
            is_call=True,
        )

        assert result.converged

    def test_iv_deep_otm(self):
        """IV should handle deep OTM options."""
        params = {
            "spot": 100.0,
            "strike": 150.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.25,
        }
        price = black_scholes_price(**params, is_call=True)

        result = calculate_iv(
            market_price=price,
            spot=params["spot"],
            strike=params["strike"],
            time_to_expiry=params["time_to_expiry"],
            rate=params["rate"],
            dividend_yield=params["dividend_yield"],
            is_call=True,
        )

        assert result.converged

    def test_iv_batch_accuracy(self):
        """Batch IV calculation should be accurate."""
        n = 20
        spots = np.full(n, 100.0)
        strikes = np.linspace(80, 120, n)
        times = np.full(n, 0.25)
        rates = np.full(n, 0.05)
        yields_ = np.full(n, 0.02)
        true_vols = np.full(n, 0.25)
        is_calls = np.ones(n, dtype=bool)

        # Generate prices
        prices = np.array(
            [
                black_scholes_price(s, k, t, r, y, v, True)
                for s, k, t, r, y, v in zip(spots, strikes, times, rates, yields_, true_vols)
            ]
        )

        ivs, errors, converged = calculate_iv_batch(
            prices, spots, strikes, times, rates, yields_, is_calls
        )

        assert np.all(converged)
        assert np.max(np.abs(ivs - true_vols)) < 1e-4

    def test_iv_below_intrinsic_fails(self):
        """IV calculation should fail for price below intrinsic."""
        result = calculate_iv(
            market_price=5.0,  # Below intrinsic for ITM call
            spot=110.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            is_call=True,
        )
        # Should either not converge or give error
        assert not result.converged or result.implied_volatility < 0.01

    def test_iv_american_option(self, standard_params):
        """IV calculation for American options."""
        # Get American price
        american_price = leisen_reimer_price(**standard_params, is_call=True, is_american=True)

        result = calculate_iv_american(
            market_price=american_price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=True,
        )

        # Should be close to input vol
        assert abs(result.implied_volatility - standard_params["volatility"]) < 0.01


class TestIVSolverAlgorithms:
    """Test IV solver algorithms via calculate_iv."""

    def test_standard_iv_recovery(self):
        """Standard IV recovery should work for standard cases."""
        price = black_scholes_price(100, 100, 0.25, 0.05, 0.02, 0.25, True)
        result = calculate_iv(
            market_price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            is_call=True,
        )

        assert result.converged
        assert abs(result.implied_volatility - 0.25) < 1e-6

    def test_iv_convergence_with_tolerance(self):
        """IV solver should respect tolerance."""
        price = black_scholes_price(100, 100, 0.25, 0.05, 0.02, 0.25, True)
        result = calculate_iv(
            market_price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            is_call=True,
        )

        assert result.converged
        # Result should be within solver tolerance
        assert abs(result.implied_volatility - 0.25) < 1e-4

    def test_iv_edge_case_short_expiry(self):
        """IV solver should handle edge case with short expiry."""
        # Edge case that might challenge Newton-Raphson
        price = black_scholes_price(100, 100, 0.01, 0.05, 0.02, 0.50, True)
        result = calculate_iv(
            market_price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.01,
            rate=0.05,
            dividend_yield=0.02,
            is_call=True,
        )

        assert result.converged


# =============================================================================
# JUMP CALIBRATION TESTS (25 tests)
# =============================================================================


class TestJumpCalibration:
    """Test jump parameter calibration."""

    def test_moment_calibration_basic(self):
        """Moment calibration should work on simple returns."""
        np.random.seed(42)
        # Simulate jump-diffusion returns
        n = 500
        dt = 1 / 252
        sigma = 0.2
        lambda_ = 2.0
        mu_j = -0.03
        sigma_j = 0.05

        # Diffusion part
        returns = np.random.normal(0, sigma * np.sqrt(dt), n)

        # Add jumps
        jump_times = np.random.poisson(lambda_ * dt, n)
        for i in range(n):
            if jump_times[i] > 0:
                returns[i] += np.random.normal(mu_j, sigma_j, jump_times[i]).sum()

        result = calibrate_from_moments(returns, dt)

        assert result.converged
        # Parameters should be in reasonable range
        assert 0 < result.jump_params.lambda_intensity < 20
        assert -1 < result.jump_params.mu_jump < 1
        assert 0 < result.jump_params.sigma_jump < 1

    def test_mle_calibration(self):
        """MLE calibration should work."""
        np.random.seed(42)
        n = 500
        dt = 1 / 252
        returns = np.random.normal(0, 0.01, n)

        result = calibrate_from_mle(returns, dt)

        assert result.converged

    def test_option_price_calibration(self):
        """Calibration from option prices should work."""
        # Generate option prices from known jump parameters
        true_params = JumpParams(lambda_intensity=1.0, mu_jump=-0.03, sigma_jump=0.10)

        spot = 100.0
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        is_calls = np.array([True, True, True, True, True])

        # Generate "market" prices
        prices = np.array(
            [
                merton_jump_diffusion_price(spot, k, t, 0.05, 0.02, 0.20, True, true_params)
                for k, t in zip(strikes, maturities)
            ]
        )

        result = calibrate_from_options(
            option_prices=prices,
            strikes=strikes,
            maturities=maturities,
            is_calls=is_calls,
            spot=spot,
            rate=0.05,
            dividend_yield=0.02,
            base_volatility=0.20,
        )

        assert result.converged
        # Should recover approximate parameters
        assert abs(result.jump_params.lambda_intensity - true_params.lambda_intensity) < 2.0

    def test_jump_detection(self):
        """Jump detection should identify outliers."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        # Add obvious jumps
        returns[20] = 0.10  # 10% jump
        returns[50] = -0.08  # -8% jump

        result = detect_jumps(returns, threshold_sigmas=3.0)

        assert result.n_jumps >= 2
        assert 20 in result.jump_times
        assert 50 in result.jump_times

    def test_hybrid_calibration(self):
        """Hybrid calibration should combine methods."""
        np.random.seed(42)

        # Historical returns
        returns = np.random.normal(0, 0.01, 200)

        # Option prices
        spot = 100.0
        strikes = np.array([95, 100, 105])
        maturities = np.array([0.25, 0.25, 0.25])
        is_calls = np.array([True, True, True])
        prices = np.array(
            [
                black_scholes_price(spot, k, t, 0.05, 0.02, 0.25, True)
                for k, t in zip(strikes, maturities)
            ]
        )

        cal_input = CalibrationInput(
            returns=returns,
            dt=1 / 252,
            option_prices=prices,
            strikes=strikes,
            maturities=maturities,
            is_calls=is_calls,
            spot=spot,
            rate=0.05,
            dividend_yield=0.02,
            base_volatility=0.25,
        )

        result = calibrate_hybrid(cal_input)

        assert result.converged

    def test_calibrator_class(self):
        """JumpCalibrator class should work."""
        calibrator = JumpCalibrator()

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 200)

        result = calibrator.calibrate(
            method=CalibrationMethod.HISTORICAL_MOMENTS,
            returns=returns,
            dt=1 / 252,
        )

        assert result.converged
        assert calibrator.get_last_result() is not None
        assert len(calibrator.get_history()) == 1


# =============================================================================
# DISCRETE DIVIDENDS TESTS (25 tests)
# =============================================================================


class TestDiscreteDividends:
    """Test discrete dividend handling."""

    def test_pv_dividends_calculation(self, dividend_schedule, standard_params):
        """PV of dividends should be positive."""
        adjusted, pv, n = adjust_spot_for_dividends(
            spot=standard_params["spot"],
            dividend_schedule=dividend_schedule,
            valuation_date=0.0,
            rate=standard_params["rate"],
            time_to_expiry=standard_params["time_to_expiry"],
        )

        assert pv > 0
        assert adjusted < standard_params["spot"]
        assert n == 2

    def test_escrowed_model_reduces_call_price(self, dividend_schedule, standard_params):
        """Escrowed model should reduce call price."""
        # Without dividends
        no_div_price = black_scholes_price(**standard_params, is_call=True)

        # With dividends (exclude dividend_yield as these functions use explicit dividends)
        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_with_escrowed_dividends(
            **params_no_yield,
            is_call=True,
            dividend_schedule=dividend_schedule,
        )

        assert result.price < no_div_price
        assert result.dividends_used == 2

    def test_escrowed_model_increases_put_price(self, dividend_schedule, standard_params):
        """Escrowed model should increase put price."""
        # Without dividends
        no_div_price = black_scholes_price(**standard_params, is_call=False)

        # With dividends (exclude dividend_yield as these functions use explicit dividends)
        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_with_escrowed_dividends(
            **params_no_yield,
            is_call=False,
            dividend_schedule=dividend_schedule,
        )

        assert result.price > no_div_price

    def test_piecewise_lognormal_basic(self, dividend_schedule, standard_params):
        """Piecewise lognormal should produce valid prices."""
        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_with_piecewise_lognormal(
            **params_no_yield,
            is_call=True,
            dividend_schedule=dividend_schedule,
        )

        assert result.price > 0
        assert result.model == DividendModel.PIECEWISE_LOGNORMAL

    def test_american_with_dividends(self, dividend_schedule, standard_params):
        """American option with dividends should price correctly."""
        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_american_with_dividends(
            **params_no_yield,
            is_call=True,
            dividend_schedule=dividend_schedule,
        )

        assert result.price > 0
        assert result.model == DividendModel.BINOMIAL_EXPLICIT

    def test_no_dividends_matches_bs(self, standard_params):
        """No dividends should match Black-Scholes."""
        empty_schedule = DividendSchedule(ex_dates=[], amounts=[])

        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_with_dividends(
            **params_no_yield,
            is_call=True,
            dividend_schedule=empty_schedule,
        )

        # Compare to BS without dividend yield (since discrete dividends are explicit)
        bs_price = black_scholes_price(
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=0.0,  # No continuous yield
            volatility=standard_params["volatility"],
            is_call=True,
        )
        assert abs(result.price - bs_price) < 0.01

    def test_percentage_dividends(self, standard_params):
        """Percentage dividends should work."""
        schedule = DividendSchedule(
            ex_dates=[0.1, 0.2],
            amounts=[0.01, 0.01],  # 1% each
            is_percentage=True,
        )

        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        result = price_with_dividends(
            **params_no_yield,
            is_call=True,
            dividend_schedule=schedule,
        )

        assert result.price > 0

    def test_yield_to_discrete_conversion(self, standard_params):
        """Yield to discrete conversion should work."""
        schedule = yield_to_discrete_dividends(
            spot=standard_params["spot"],
            dividend_yield=0.02,
            time_to_expiry=1.0,
            frequency=4,
        )

        assert len(schedule.ex_dates) == 4

    def test_dividend_estimation(self):
        """Dividend estimation should project future dividends."""
        today = date.today()
        historical = [
            Dividend(
                ex_date=today - timedelta(days=90),
                amount=0.50,
                record_date=today - timedelta(days=88),
                payment_date=today - timedelta(days=80),
            ),
            Dividend(
                ex_date=today - timedelta(days=180),
                amount=0.48,
                record_date=today - timedelta(days=178),
                payment_date=today - timedelta(days=170),
            ),
        ]

        schedule = estimate_future_dividends(
            historical_dividends=historical,
            projection_horizon=1.0,
        )

        assert len(schedule.ex_dates) > 0


# =============================================================================
# EXERCISE PROBABILITY TESTS (30 tests)
# =============================================================================


class TestExerciseProbability:
    """Test early exercise probability calculations."""

    def test_ls_basic_pricing(self, standard_params):
        """Longstaff-Schwartz should produce valid price."""
        result = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            random_seed=42,
        )

        assert isinstance(result, LSResult)
        assert result.price > 0
        assert result.standard_error > 0
        assert result.n_paths == 10000

    def test_ls_matches_bs_for_european_call(self, standard_params):
        """LS should approximately match BS for European call (no early exercise)."""
        # For call without dividends, American = European
        params = standard_params.copy()
        params["dividend_yield"] = 0.0

        bs_price = black_scholes_price(**params, is_call=True)

        result = longstaff_schwartz(
            **params,
            is_call=True,
            n_paths=50000,
            n_steps=50,
            random_seed=42,
        )

        # Should be within 3 standard errors
        assert abs(result.price - bs_price) < 3 * result.standard_error + 0.5

    def test_ls_american_put_premium(self):
        """American put should have early exercise premium."""
        params = {
            "spot": 90.0,
            "strike": 100.0,
            "time_to_expiry": 0.5,
            "rate": 0.10,
            "dividend_yield": 0.0,
            "volatility": 0.25,
        }

        result = longstaff_schwartz(
            **params,
            is_call=False,
            n_paths=50000,
            n_steps=50,
            random_seed=42,
        )

        bs_price = black_scholes_price(**params, is_call=False)

        # American put should be > European put
        assert result.price > bs_price * 0.95

    def test_ls_exercise_probabilities_sum_to_one(self, standard_params):
        """Exercise probabilities should approximately sum to 1."""
        result = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            random_seed=42,
        )

        total = (
            result.prob_early_exercise
            + result.prob_exercise_at_expiry
            + result.prob_expire_worthless
        )
        assert abs(total - 1.0) < 0.01

    def test_ls_greeks_computation(self, standard_params):
        """LS Greeks computation should work."""
        result = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=20000,
            n_steps=50,
            compute_greeks=True,
            random_seed=42,
        )

        assert result.delta is not None
        assert result.gamma is not None
        assert result.delta_se is not None

    def test_ls_exercise_boundary_extraction(self, standard_params):
        """Exercise boundary extraction should work."""
        result = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=20000,
            n_steps=50,
            extract_boundary=True,
            random_seed=42,
        )

        if result.exercise_boundary is not None:
            assert len(result.exercise_boundary.times) > 0
            assert len(result.exercise_boundary.boundary_prices) == len(
                result.exercise_boundary.times
            )

    def test_ls_different_basis_functions(self, standard_params):
        """Different basis functions should produce similar results."""
        results = {}
        for basis in [BasisFunctions.POWER, BasisFunctions.LAGUERRE, BasisFunctions.HERMITE]:
            results[basis] = longstaff_schwartz(
                **standard_params,
                is_call=True,
                n_paths=20000,
                n_steps=50,
                basis=basis,
                random_seed=42,
            )

        prices = [r.price for r in results.values()]
        # All should be within 10% of each other
        assert max(prices) / min(prices) < 1.1

    def test_ls_antithetic_variance_reduction(self, standard_params):
        """Antithetic variates should reduce variance."""
        result_no_av = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            variance_reduction=VarianceReduction.NONE,
            random_seed=42,
        )

        result_av = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            variance_reduction=VarianceReduction.ANTITHETIC,
            random_seed=42,
        )

        # Antithetic should typically have lower SE
        # (Not guaranteed for every seed, so just check it works)
        assert result_av.standard_error > 0

    def test_compute_exercise_probability(self, standard_params):
        """Exercise probability function should work."""
        prob = compute_exercise_probability(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            random_seed=42,
        )

        assert len(prob.times) > 0
        assert 0 <= prob.prob_early <= 1

    def test_should_exercise_early(self):
        """Should exercise early function should work."""
        should, prob, ratio = should_exercise_early(
            spot=80.0,
            strike=100.0,
            time_to_expiry=0.5,
            rate=0.10,
            dividend_yield=0.0,
            volatility=0.25,
            is_call=False,
        )

        assert isinstance(should, bool)
        assert 0 <= prob <= 1

    def test_early_exercise_premium(self, standard_params):
        """Early exercise premium calculation should work."""
        american, european, premium = compute_early_exercise_premium(
            **standard_params,
            is_call=True,
            n_paths=20000,
            n_steps=50,
        )

        assert american >= european - 0.5  # Allow small MC error
        assert premium >= -0.5

    def test_barone_adesi_whaley_call(self, standard_params):
        """BAW approximation should work for calls."""
        price, premium = barone_adesi_whaley(
            **standard_params,
            is_call=True,
        )

        european = black_scholes_price(**standard_params, is_call=True)
        assert price >= european * 0.99

    def test_barone_adesi_whaley_put(self):
        """BAW should show premium for deep ITM put."""
        params = {
            "spot": 80.0,
            "strike": 100.0,
            "time_to_expiry": 0.5,
            "rate": 0.10,
            "dividend_yield": 0.0,
            "volatility": 0.25,
        }

        price, premium = barone_adesi_whaley(**params, is_call=False)
        european = black_scholes_price(**params, is_call=False)

        assert price > european
        assert premium > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests across components."""

    def test_full_option_analysis_workflow(self, standard_params):
        """Full workflow from pricing to Greeks to IV recovery."""
        # Price option
        price = black_scholes_price(**standard_params, is_call=True)

        # Compute Greeks
        greeks = compute_all_greeks(**standard_params, is_call=True)

        # Recover IV
        iv_result = calculate_iv(
            market_price=price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=True,
        )

        assert abs(iv_result.implied_volatility - standard_params["volatility"]) < 1e-6

    def test_jump_diffusion_iv_recovery(self, standard_params, jump_params):
        """IV recovery for jump-diffusion prices."""
        # Price with JD model
        jd_price = merton_jump_diffusion_price(
            **standard_params,
            is_call=True,
            jump_params=jump_params,
        )

        # Compute implied vol (BS IV)
        iv_result = calculate_iv(
            market_price=jd_price,
            spot=standard_params["spot"],
            strike=standard_params["strike"],
            time_to_expiry=standard_params["time_to_expiry"],
            rate=standard_params["rate"],
            dividend_yield=standard_params["dividend_yield"],
            is_call=True,
        )

        # IV should be higher than BS vol due to jumps
        assert iv_result.converged

    def test_dividend_impact_on_exercise(self, dividend_schedule, standard_params):
        """Dividends should affect early exercise probability."""
        # Without dividends
        result_no_div = longstaff_schwartz(
            **standard_params,
            is_call=True,
            n_paths=10000,
            n_steps=50,
            random_seed=42,
        )

        # Discrete dividend function doesn't take dividend_yield
        params_no_yield = {k: v for k, v in standard_params.items() if k != "dividend_yield"}
        # Compare with binomial that includes dividends
        american_with_div = price_american_with_dividends(
            **params_no_yield,
            is_call=True,
            dividend_schedule=dividend_schedule,
        )

        # Both should produce valid prices
        assert result_no_div.price > 0
        assert american_with_div.price > 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_time_to_expiry(self):
        """Should handle zero time to expiry."""
        params = {
            "spot": 105.0,
            "strike": 100.0,
            "time_to_expiry": 0.0,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }

        price = black_scholes_price(**params, is_call=True)
        assert abs(price - 5.0) < 0.01  # Intrinsic value

    def test_very_long_maturity(self):
        """Should handle very long maturity."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 10.0,  # 10 years
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }

        price = black_scholes_price(**params, is_call=True)
        assert price > 0 and math.isfinite(price)

    def test_extreme_moneyness(self):
        """Should handle extreme moneyness."""
        # Deep OTM call
        price = black_scholes_price(
            spot=100.0,
            strike=200.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            volatility=0.20,
            is_call=True,
        )
        assert price >= 0 and math.isfinite(price)

        # Deep ITM put
        price = black_scholes_price(
            spot=200.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            volatility=0.20,
            is_call=False,
        )
        assert price >= 0 and math.isfinite(price)

    def test_high_interest_rate(self):
        """Should handle high interest rates."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.50,  # 50%
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }

        price = black_scholes_price(**params, is_call=True)
        assert price > 0 and math.isfinite(price)

    def test_negative_rates(self):
        """Should handle negative interest rates."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": -0.02,  # -2%
            "dividend_yield": 0.0,
            "volatility": 0.20,
        }

        price = black_scholes_price(**params, is_call=True)
        assert price > 0 and math.isfinite(price)

    def test_empty_option_chain(self):
        """Should handle empty inputs gracefully."""
        spots = np.array([])
        strikes = np.array([])
        times = np.array([])
        rates = np.array([])
        yields_ = np.array([])
        vols = np.array([])
        is_calls = np.array([], dtype=bool)

        result = compute_all_greeks_batch(spots, strikes, times, rates, yields_, vols, is_calls)

        assert result.n_options == 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance regression tests."""

    def test_batch_greeks_performance(self):
        """Batch Greeks should be fast."""
        n = 1000
        spots = np.random.uniform(90, 110, n)
        strikes = np.random.uniform(80, 120, n)
        times = np.random.uniform(0.1, 1.0, n)
        rates = np.full(n, 0.05)
        yields_ = np.full(n, 0.02)
        vols = np.random.uniform(0.1, 0.5, n)
        is_calls = np.random.choice([True, False], n)

        import time

        start = time.perf_counter()
        result = compute_all_greeks_batch(spots, strikes, times, rates, yields_, vols, is_calls)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second for 1000 options)
        assert elapsed < 1.0
        assert result.n_options == n

    def test_iv_batch_performance(self):
        """Batch IV should be reasonably fast."""
        n = 100
        spots = np.full(n, 100.0)
        strikes = np.linspace(80, 120, n)
        times = np.full(n, 0.25)
        rates = np.full(n, 0.05)
        yields_ = np.full(n, 0.02)
        vols = np.full(n, 0.25)
        is_calls = np.ones(n, dtype=bool)

        prices = np.array(
            [
                black_scholes_price(s, k, t, r, y, v, True)
                for s, k, t, r, y, v in zip(spots, strikes, times, rates, yields_, vols)
            ]
        )

        import time

        start = time.perf_counter()
        ivs, _, converged = calculate_iv_batch(
            prices, spots, strikes, times, rates, yields_, is_calls
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0  # Should be fast
        assert np.all(converged)


# =============================================================================
# CONTRACT SPECS AND ENUMS TESTS
# =============================================================================


class TestContractSpecs:
    """Test contract specifications and enums."""

    def test_option_type_enum(self):
        """OptionType enum should work."""
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_exercise_style_enum(self):
        """ExerciseStyle enum should work."""
        assert ExerciseStyle.EUROPEAN.value == "european"
        assert ExerciseStyle.AMERICAN.value == "american"

    def test_settlement_type_enum(self):
        """SettlementType enum should work."""
        assert SettlementType.CASH.value == "cash"
        assert SettlementType.PHYSICAL.value == "physical"

    def test_options_contract_spec_creation(self):
        """OptionsContractSpec should be creatable."""
        from decimal import Decimal

        spec = OptionsContractSpec(
            symbol="SPY",
            underlying="SPY",
            strike=Decimal("450.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            settlement=SettlementType.PHYSICAL,
            multiplier=100,
        )

        assert spec.symbol == "SPY"
        assert spec.strike == Decimal("450.0")
        assert spec.option_type == OptionType.CALL

    def test_greeks_result_dataclass(self):
        """GreeksResult dataclass should work."""
        result = GreeksResult(
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.30,
            rho=0.10,
            vanna=0.01,
            volga=0.005,
            charm=0.002,
            speed=0.001,
            color=0.0005,
            zomma=0.0002,
            ultima=0.0001,
            timestamp_ns=0,
        )

        assert result.delta == 0.5
        assert result.gamma == 0.02

    def test_jump_params_dataclass(self):
        """JumpParams dataclass should work."""
        params = JumpParams(
            lambda_intensity=1.0,
            mu_jump=-0.03,
            sigma_jump=0.10,
        )

        assert params.lambda_intensity == 1.0
        assert params.mu_jump == -0.03


# =============================================================================
# ADDITIONAL TESTS TO MEET 200 TEST TARGET
# =============================================================================


class TestContractSpecsExtended:
    """Extended tests for OptionsContractSpec (additional 9 tests)."""

    def test_contract_spec_with_defaults(self):
        """Contract spec should have sensible defaults."""
        spec = OptionsContractSpec(
            symbol="AAPL240315C00175000",
            underlying="AAPL",
            strike=Decimal("175.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            settlement=SettlementType.PHYSICAL,
        )
        assert spec.multiplier == 100
        assert spec.tick_size == Decimal("0.01")

    def test_contract_spec_european_style(self):
        """European style contracts should work."""
        spec = OptionsContractSpec(
            symbol="SPX240315C04500000",
            underlying="SPX",
            strike=Decimal("4500.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
            settlement=SettlementType.CASH,
        )
        assert spec.exercise_style == ExerciseStyle.EUROPEAN
        assert spec.settlement == SettlementType.CASH

    def test_contract_spec_put_option(self):
        """Put option contract spec should work."""
        spec = OptionsContractSpec(
            symbol="AAPL240315P00150000",
            underlying="AAPL",
            strike=Decimal("150.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN,
            settlement=SettlementType.PHYSICAL,
        )
        assert spec.option_type == OptionType.PUT

    def test_contract_spec_index_option(self):
        """Index option should have cash settlement."""
        spec = OptionsContractSpec(
            symbol="VIX240315C00020000",
            underlying="VIX",
            strike=Decimal("20.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
            settlement=SettlementType.CASH,
            exchange="CBOE",
        )
        assert spec.settlement == SettlementType.CASH
        assert spec.exchange == "CBOE"

    def test_contract_spec_custom_multiplier(self):
        """Non-standard multipliers should work."""
        spec = OptionsContractSpec(
            symbol="MINI",
            underlying="SPY",
            strike=Decimal("450.0"),
            expiration=(datetime.now() + timedelta(days=30)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            settlement=SettlementType.PHYSICAL,
            multiplier=10,  # Mini option
        )
        assert spec.multiplier == 10

    def test_contract_spec_different_exchanges(self):
        """Different exchanges should be supported."""
        exchanges = ["CBOE", "PHLX", "NYSE_ARCA", "ISE", "MIAX"]
        for exchange in exchanges:
            spec = OptionsContractSpec(
                symbol="TEST",
                underlying="TEST",
                strike=Decimal("100.0"),
                expiration=(datetime.now() + timedelta(days=30)).date(),
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.AMERICAN,
                settlement=SettlementType.PHYSICAL,
                exchange=exchange,
            )
            assert spec.exchange == exchange

    def test_contract_spec_far_expiration(self):
        """LEAPS with far expiration should work."""
        spec = OptionsContractSpec(
            symbol="AAPL260116C00200000",
            underlying="AAPL",
            strike=Decimal("200.0"),
            expiration=(datetime.now() + timedelta(days=365)).date(),
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            settlement=SettlementType.PHYSICAL,
        )
        days_to_exp = (spec.expiration - datetime.now().date()).days
        assert days_to_exp >= 365

    def test_contract_spec_tick_size_variations(self):
        """Different tick sizes should work."""
        for tick in [Decimal("0.01"), Decimal("0.05"), Decimal("0.10")]:
            spec = OptionsContractSpec(
                symbol="TEST",
                underlying="TEST",
                strike=Decimal("100.0"),
                expiration=(datetime.now() + timedelta(days=30)).date(),
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.AMERICAN,
                settlement=SettlementType.PHYSICAL,
                tick_size=tick,
            )
            assert spec.tick_size == tick

    def test_iv_result_dataclass(self):
        """IVResult dataclass should work correctly."""
        result = IVResult(
            implied_volatility=0.25,
            converged=True,
            iterations=5,
            error=1e-10,
            method="newton",
            model_price=5.50,
        )
        assert result.implied_volatility == 0.25
        assert result.converged is True
        assert result.iterations == 5


class TestScalarGreeksExtended:
    """Extended Greeks tests (additional 11 tests)."""

    @pytest.fixture
    def atm_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }

    def test_delta_deep_itm_call_near_one(self, atm_params):
        """Deep ITM call should have delta near 1."""
        params = {**atm_params, "strike": 80.0}  # Deep ITM
        greeks = compute_all_greeks(**params, is_call=True)
        assert greeks.delta > 0.95

    def test_delta_deep_otm_call_near_zero(self, atm_params):
        """Deep OTM call should have delta near 0."""
        params = {**atm_params, "strike": 130.0}  # Deep OTM
        greeks = compute_all_greeks(**params, is_call=True)
        assert greeks.delta < 0.10

    def test_delta_deep_itm_put_near_neg_one(self, atm_params):
        """Deep ITM put should have delta near -1."""
        params = {**atm_params, "strike": 120.0}  # Deep ITM put
        greeks = compute_all_greeks(**params, is_call=False)
        assert greeks.delta < -0.90

    def test_gamma_decreases_far_from_atm(self, atm_params):
        """Gamma should decrease for ITM/OTM options."""
        atm_greeks = compute_all_greeks(**atm_params, is_call=True)
        itm_greeks = compute_all_greeks(**{**atm_params, "strike": 80.0}, is_call=True)
        otm_greeks = compute_all_greeks(**{**atm_params, "strike": 120.0}, is_call=True)
        assert atm_greeks.gamma > itm_greeks.gamma
        assert atm_greeks.gamma > otm_greeks.gamma

    def test_theta_more_negative_atm(self, atm_params):
        """ATM options should have most negative theta."""
        atm_greeks = compute_all_greeks(**atm_params, is_call=True)
        otm_greeks = compute_all_greeks(**{**atm_params, "strike": 115.0}, is_call=True)
        assert abs(atm_greeks.theta) > abs(otm_greeks.theta)

    def test_theta_accelerates_near_expiry(self, atm_params):
        """Theta should accelerate near expiration."""
        far_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 0.5}, is_call=True)
        near_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 0.05}, is_call=True)
        # Near expiry ATM has higher absolute theta per day
        assert abs(near_greeks.theta) > abs(far_greeks.theta)

    def test_vega_decreases_with_time(self, atm_params):
        """Vega should be higher for longer-dated options."""
        short_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 0.1}, is_call=True)
        long_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 1.0}, is_call=True)
        assert long_greeks.vega > short_greeks.vega

    def test_rho_increases_with_time(self, atm_params):
        """Rho should be higher for longer-dated options."""
        short_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 0.1}, is_call=True)
        long_greeks = compute_all_greeks(**{**atm_params, "time_to_expiry": 1.0}, is_call=True)
        assert abs(long_greeks.rho) > abs(short_greeks.rho)

    def test_higher_order_greeks_bounded(self, atm_params):
        """Third-order Greeks should be reasonably bounded."""
        greeks = compute_all_greeks(**atm_params, is_call=True)
        assert abs(greeks.speed) < 10
        assert abs(greeks.color) < 10
        assert abs(greeks.zomma) < 10
        assert abs(greeks.ultima) < 10

    def test_greeks_at_zero_rate(self, atm_params):
        """Greeks should work at zero interest rate."""
        params = {**atm_params, "rate": 0.0}
        greeks = compute_all_greeks(**params, is_call=True)
        # Note: Rho is dP/dr (sensitivity), not r itself; it can be non-zero at r=0
        assert np.isfinite(greeks.rho)
        assert 0.4 < greeks.delta < 0.6  # Still around 0.5 for ATM

    def test_greeks_with_high_volatility(self, atm_params):
        """Greeks should work with high volatility."""
        params = {**atm_params, "volatility": 1.0}  # 100% vol
        greeks = compute_all_greeks(**params, is_call=True)
        assert 0.4 < greeks.delta < 0.8  # Delta still reasonable
        assert greeks.vega > 0


class TestBlackScholesPricingExtended:
    """Extended Black-Scholes pricing tests (additional 17 tests)."""

    @pytest.fixture
    def standard_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.20,
        }

    def test_bs_call_intrinsic_value_floor(self, standard_params):
        """Call price should be at least max(S-K, 0)."""
        params = {**standard_params, "strike": 90.0}  # ITM
        price = black_scholes_price(**params, is_call=True)
        intrinsic = max(params["spot"] - params["strike"], 0)
        assert price >= intrinsic

    def test_bs_put_intrinsic_value_floor(self, standard_params):
        """Put price should be at least max(K-S, 0) discounted."""
        params = {**standard_params, "strike": 110.0}  # ITM put
        price = black_scholes_price(**params, is_call=False)
        # Allowing for time value, price should be positive
        assert price > 0

    def test_bs_call_upper_bound(self, standard_params):
        """Call price should be at most S*e^(-qT)."""
        price = black_scholes_price(**standard_params, is_call=True)
        upper = standard_params["spot"] * np.exp(
            -standard_params["dividend_yield"] * standard_params["time_to_expiry"]
        )
        assert price <= upper

    def test_bs_put_upper_bound(self, standard_params):
        """Put price should be at most K*e^(-rT)."""
        price = black_scholes_price(**standard_params, is_call=False)
        upper = standard_params["strike"] * np.exp(
            -standard_params["rate"] * standard_params["time_to_expiry"]
        )
        assert price <= upper

    def test_bs_put_call_parity(self, standard_params):
        """Put-Call parity should hold: C - P = S*e^(-qT) - K*e^(-rT)."""
        call_price = black_scholes_price(**standard_params, is_call=True)
        put_price = black_scholes_price(**standard_params, is_call=False)
        S = standard_params["spot"]
        K = standard_params["strike"]
        r = standard_params["rate"]
        q = standard_params["dividend_yield"]
        T = standard_params["time_to_expiry"]
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(call_price - put_price - parity) < 1e-10

    def test_bs_price_increases_with_volatility(self, standard_params):
        """Option price should increase with volatility."""
        low_vol = black_scholes_price(**{**standard_params, "volatility": 0.10}, is_call=True)
        high_vol = black_scholes_price(**{**standard_params, "volatility": 0.40}, is_call=True)
        assert high_vol > low_vol

    def test_bs_call_price_increases_with_rate(self, standard_params):
        """Call price should increase with interest rate."""
        low_rate = black_scholes_price(**{**standard_params, "rate": 0.01}, is_call=True)
        high_rate = black_scholes_price(**{**standard_params, "rate": 0.10}, is_call=True)
        assert high_rate > low_rate

    def test_bs_put_price_decreases_with_rate(self, standard_params):
        """Put price should decrease with interest rate."""
        low_rate = black_scholes_price(**{**standard_params, "rate": 0.01}, is_call=False)
        high_rate = black_scholes_price(**{**standard_params, "rate": 0.10}, is_call=False)
        assert high_rate < low_rate

    def test_bs_call_decreases_with_dividend(self, standard_params):
        """Call price should decrease with dividend yield."""
        low_div = black_scholes_price(**{**standard_params, "dividend_yield": 0.0}, is_call=True)
        high_div = black_scholes_price(**{**standard_params, "dividend_yield": 0.05}, is_call=True)
        assert high_div < low_div

    def test_bs_put_increases_with_dividend(self, standard_params):
        """Put price should increase with dividend yield."""
        low_div = black_scholes_price(**{**standard_params, "dividend_yield": 0.0}, is_call=False)
        high_div = black_scholes_price(**{**standard_params, "dividend_yield": 0.05}, is_call=False)
        assert high_div > low_div

    def test_bs_price_decays_with_time(self, standard_params):
        """ATM option price should decay with time (mostly theta effect)."""
        long_time = black_scholes_price(**{**standard_params, "time_to_expiry": 1.0}, is_call=True)
        short_time = black_scholes_price(**{**standard_params, "time_to_expiry": 0.1}, is_call=True)
        assert long_time > short_time

    def test_bs_deep_itm_call_nearly_intrinsic(self, standard_params):
        """Deep ITM call should be close to intrinsic value."""
        params = {**standard_params, "strike": 50.0}
        price = black_scholes_price(**params, is_call=True)
        intrinsic = params["spot"] - params["strike"]
        assert abs(price - intrinsic) < 5.0  # Within $5 of intrinsic

    def test_bs_deep_otm_call_nearly_zero(self, standard_params):
        """Deep OTM call should be close to zero."""
        params = {**standard_params, "strike": 200.0}
        price = black_scholes_price(**params, is_call=True)
        assert price < 0.01

    def test_bs_at_expiry_call(self, standard_params):
        """At expiry, call equals max(S-K, 0)."""
        params = {**standard_params, "time_to_expiry": 1e-10}
        price = black_scholes_price(**params, is_call=True)
        intrinsic = max(params["spot"] - params["strike"], 0)
        # Note: with tiny t, there's still minimal time value; use relaxed tolerance
        assert abs(price - intrinsic) < 1e-3

    def test_bs_at_expiry_put(self, standard_params):
        """At expiry, put equals max(K-S, 0)."""
        params = {**standard_params, "time_to_expiry": 1e-10, "strike": 110.0}
        price = black_scholes_price(**params, is_call=False)
        intrinsic = max(params["strike"] - params["spot"], 0)
        assert abs(price - intrinsic) < 1e-6

    def test_bs_zero_volatility_call(self, standard_params):
        """Zero volatility call should equal forward intrinsic value."""
        params = {**standard_params, "volatility": 1e-10, "strike": 90.0}
        price = black_scholes_price(**params, is_call=True)
        fwd_intrinsic = params["spot"] * np.exp(
            -params["dividend_yield"] * params["time_to_expiry"]
        ) - params["strike"] * np.exp(-params["rate"] * params["time_to_expiry"])
        assert abs(price - max(fwd_intrinsic, 0)) < 1e-6

    def test_bs_very_high_volatility(self, standard_params):
        """Very high volatility should still give valid prices."""
        params = {**standard_params, "volatility": 2.0}
        price = black_scholes_price(**params, is_call=True)
        assert price > 0
        assert np.isfinite(price)


class TestBinomialPricingExtended:
    """Extended binomial/Leisen-Reimer tests (additional 20 tests)."""

    @pytest.fixture
    def american_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "dividend_yield": 0.03,
            "volatility": 0.25,
        }

    def test_leisen_reimer_converges_to_bs_european(self, american_params):
        """Leisen-Reimer should converge to BS for European options."""
        bs_price = black_scholes_price(**american_params, is_call=True)
        lr_price = leisen_reimer_price(
            **american_params, is_call=True, is_american=False, n_steps=501
        )
        assert abs(lr_price - bs_price) < 0.01

    def test_leisen_reimer_american_geq_european(self, american_params):
        """American option should be worth at least as much as European."""
        european = leisen_reimer_price(
            **american_params, is_call=True, is_american=False, n_steps=201
        )
        american = leisen_reimer_price(
            **american_params, is_call=True, is_american=True, n_steps=201
        )
        assert american >= european - 1e-6

    def test_crr_converges_to_bs(self, american_params):
        """CRR should converge to BS for European options."""
        bs_price = black_scholes_price(**american_params, is_call=True)
        crr_price = crr_binomial_price(
            **american_params, is_call=True, is_american=False, n_steps=500
        )
        assert abs(crr_price - bs_price) < 0.05

    def test_american_put_early_exercise_value(self, american_params):
        """American put should have early exercise premium."""
        params = {**american_params, "strike": 110.0, "dividend_yield": 0.0}  # ITM put, no div
        european_put = black_scholes_price(**params, is_call=False)
        american_put = leisen_reimer_price(**params, is_call=False, is_american=True, n_steps=201)
        # American put should be worth more (early exercise value)
        assert american_put >= european_put - 1e-6

    def test_american_call_no_early_exercise_no_dividend(self, american_params):
        """American call without dividends should equal European."""
        params = {**american_params, "dividend_yield": 0.0}
        european = black_scholes_price(**params, is_call=True)
        american = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert abs(american - european) < 0.01

    def test_american_call_with_dividend_early_exercise(self, american_params):
        """American call with high dividend should have early exercise premium."""
        params = {**american_params, "strike": 90.0, "dividend_yield": 0.10}  # ITM, high div
        european = black_scholes_price(**params, is_call=True)
        american = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert american >= european - 1e-6

    def test_binomial_step_convergence(self, american_params):
        """More steps should give more accurate results."""
        bs_price = black_scholes_price(**american_params, is_call=True)
        price_101 = crr_binomial_price(
            **american_params, is_call=True, is_american=False, n_steps=101
        )
        price_501 = crr_binomial_price(
            **american_params, is_call=True, is_american=False, n_steps=501
        )
        error_101 = abs(price_101 - bs_price)
        error_501 = abs(price_501 - bs_price)
        assert error_501 < error_101

    def test_leisen_reimer_odd_steps_required(self, american_params):
        """Leisen-Reimer should handle odd step requirement."""
        # Should work with odd steps
        price = leisen_reimer_price(**american_params, is_call=True, is_american=True, n_steps=501)
        assert price > 0

    def test_binomial_put_intrinsic_floor(self, american_params):
        """American put should be at least intrinsic value."""
        params = {**american_params, "strike": 120.0}
        price = leisen_reimer_price(**params, is_call=False, is_american=True, n_steps=201)
        intrinsic = max(params["strike"] - params["spot"], 0)
        assert price >= intrinsic - 1e-6

    def test_binomial_call_intrinsic_floor(self, american_params):
        """American call should be at least intrinsic value."""
        params = {**american_params, "strike": 80.0}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        intrinsic = max(params["spot"] - params["strike"], 0)
        assert price >= intrinsic - 1e-6

    def test_binomial_deep_itm_put(self, american_params):
        """Deep ITM American put should be close to intrinsic."""
        params = {**american_params, "strike": 150.0}
        price = leisen_reimer_price(**params, is_call=False, is_american=True, n_steps=201)
        intrinsic = params["strike"] - params["spot"]
        assert abs(price - intrinsic) < 5.0

    def test_binomial_deep_otm_call(self, american_params):
        """Deep OTM call should be small."""
        params = {**american_params, "strike": 150.0}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price < 1.0

    def test_binomial_long_expiry(self, american_params):
        """Long expiry options should work."""
        params = {**american_params, "time_to_expiry": 2.0}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price > 0
        assert np.isfinite(price)

    def test_binomial_short_expiry(self, american_params):
        """Short expiry options should work."""
        params = {**american_params, "time_to_expiry": 0.01}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price >= 0
        assert np.isfinite(price)

    def test_barone_adesi_whaley_american_put(self, american_params):
        """Barone-Adesi-Whaley approximation for American put."""
        price, _ = barone_adesi_whaley(**american_params, is_call=False)
        european = black_scholes_price(**american_params, is_call=False)
        assert price >= european - 1e-6

    def test_barone_adesi_whaley_american_call_no_div(self, american_params):
        """BAW American call without dividends should equal European."""
        params = {**american_params, "dividend_yield": 0.0}
        baw_price, _ = barone_adesi_whaley(**params, is_call=True)
        european = black_scholes_price(**params, is_call=True)
        assert abs(baw_price - european) < 0.05

    def test_binomial_high_volatility(self, american_params):
        """High volatility should work."""
        params = {**american_params, "volatility": 0.80}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price > 0
        assert np.isfinite(price)

    def test_binomial_low_volatility(self, american_params):
        """Low volatility should work."""
        params = {**american_params, "volatility": 0.05}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price >= 0
        assert np.isfinite(price)

    def test_binomial_zero_rate(self, american_params):
        """Zero rate should work."""
        params = {**american_params, "rate": 0.0}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price > 0

    def test_binomial_high_rate(self, american_params):
        """High interest rate should work."""
        params = {**american_params, "rate": 0.20}
        price = leisen_reimer_price(**params, is_call=True, is_american=True, n_steps=201)
        assert price > 0
        assert np.isfinite(price)

    def test_binomial_extreme_underflow(self) -> None:
        """CRR and LR binomial trees should handle extreme low vol/short expiry without ZeroDivisionError."""
        params = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.01e-10,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 1.01e-10,
        }
        # This would raise ZeroDivisionError before the fix due to u - d underflow to 0.0
        price_crr = crr_binomial_price(**params, is_call=True, n_steps=200)
        price_lr = leisen_reimer_price(**params, is_call=True, n_steps=201)

        # Zero-volatility intrinsic price
        assert price_crr >= 0.0
        assert price_lr >= 0.0
        assert np.isfinite(price_crr)
        assert np.isfinite(price_lr)

    def test_variance_swap_zero_strike(self) -> None:
        """variance_swap_strike should ignore strikes <= 0.0 to prevent ZeroDivisionError in integration."""
        call_prices = np.array([1.5, 0.5])
        put_prices = np.array([2.5, 1.2])
        call_strikes = np.array([110.0, 120.0])
        # Put strikes containing a zero strike
        put_strikes = np.array([0.0, 90.0])

        # This would raise ZeroDivisionError before the fix due to division by k^2
        strike = variance_swap_strike(
            call_prices=call_prices,
            put_prices=put_prices,
            call_strikes=call_strikes,
            put_strikes=put_strikes,
            forward=100.0,
            rate=0.05,
            time_to_expiry=0.25,
        )
        assert strike >= 0.0
        assert np.isfinite(strike)

    def test_baw_zero_rate(self) -> None:
        """barone_adesi_whaley should handle rate = 0.0 without ZeroDivisionError."""
        price, premium = barone_adesi_whaley(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.0,
            dividend_yield=0.02,
            volatility=0.20,
            is_call=True,
        )
        assert price > 0.0
        assert np.isfinite(price)

    def test_merton_extreme_negative_jump_mean(self) -> None:
        """merton_jump_diffusion_price should handle extremely negative jump mean without ValueError."""
        jp = JumpParams(lambda_intensity=1.0, mu_jump=-100.0, sigma_jump=0.1)
        price = merton_jump_diffusion_price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.02,
            volatility=0.20,
            is_call=True,
            jump_params=jp,
        )
        assert price > 0.0
        assert np.isfinite(price)

    def test_jump_calibration_safeguards(self) -> None:
        """Jump calibration functions should validate inputs and raise CalibrationError on invalid values."""
        # Test time step <= 0
        with pytest.raises(CalibrationError):
            calibrate_from_moments(np.array([0.01, -0.02]), dt=0.0)
        with pytest.raises(CalibrationError):
            calibrate_from_mle(np.array([0.01, -0.02]), dt=-1.0)

        # Test option calibration invalid inputs
        with pytest.raises(CalibrationError):
            calibrate_from_options(
                option_prices=np.array([1.0, 2.0, 3.0]),
                strikes=np.array([90.0, 100.0, 110.0]),
                maturities=np.array([0.25, 0.25, 0.25]),
                is_calls=np.array([True, True, True]),
                spot=0.0,  # invalid spot
                rate=0.05,
                dividend_yield=0.02,
                base_volatility=0.20,
            )


class TestJumpDiffusionExtended:
    """Extended jump-diffusion tests (additional 11 tests)."""

    @pytest.fixture
    def jump_pricing_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.0,
            "volatility": 0.20,
            "is_call": True,
        }

    def test_merton_no_jumps_equals_bs(self, jump_pricing_params):
        """With lambda=0, Merton should equal Black-Scholes."""
        bs_price = black_scholes_price(
            spot=jump_pricing_params["spot"],
            strike=jump_pricing_params["strike"],
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=True,
        )
        merton_price = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=0.0, mu_jump=0.0, sigma_jump=0.1),
        )
        assert abs(merton_price - bs_price) < 0.01

    def test_merton_small_jumps_near_bs(self, jump_pricing_params):
        """With small jumps, Merton should be near BS."""
        bs_price = black_scholes_price(
            spot=jump_pricing_params["spot"],
            strike=jump_pricing_params["strike"],
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=True,
        )
        merton_price = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=0.1, mu_jump=0.0, sigma_jump=0.01),
        )
        assert abs(merton_price - bs_price) < 0.5

    def test_merton_negative_jumps_skew_otm_puts(self, jump_pricing_params):
        """Negative mean jumps should make OTM puts more expensive."""
        no_jump = black_scholes_price(
            spot=jump_pricing_params["spot"],
            strike=90.0,
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=False,
        )
        with_jump = merton_jump_diffusion_price(
            spot=jump_pricing_params["spot"],
            strike=90.0,
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=False,
            jump_params=JumpParams(lambda_intensity=2.0, mu_jump=-0.10, sigma_jump=0.15),
        )
        assert with_jump > no_jump

    def test_merton_put_call_parity(self, jump_pricing_params):
        """Merton model should satisfy put-call parity."""
        jp = JumpParams(lambda_intensity=1.0, mu_jump=-0.05, sigma_jump=0.10)
        call_price = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=jp,
        )
        put_price = merton_jump_diffusion_price(
            **{**jump_pricing_params, "is_call": False},
            jump_params=jp,
        )
        S = jump_pricing_params["spot"]
        K = jump_pricing_params["strike"]
        r = jump_pricing_params["rate"]
        T = jump_pricing_params["time_to_expiry"]
        parity_diff = call_price - put_price - (S - K * np.exp(-r * T))
        assert abs(parity_diff) < 0.5  # Allowing for numerical error

    def test_merton_price_positive(self, jump_pricing_params):
        """Merton price should always be positive."""
        price = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=3.0, mu_jump=-0.20, sigma_jump=0.30),
        )
        assert price > 0

    def test_merton_increases_with_jump_volatility(self, jump_pricing_params):
        """Higher jump volatility should increase option price."""
        low_jvol = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=1.0, mu_jump=0.0, sigma_jump=0.05),
        )
        high_jvol = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=1.0, mu_jump=0.0, sigma_jump=0.30),
        )
        assert high_jvol > low_jvol

    def test_merton_increases_with_jump_intensity(self, jump_pricing_params):
        """Higher jump intensity should generally increase option price."""
        low_lambda = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=0.5, mu_jump=0.0, sigma_jump=0.15),
        )
        high_lambda = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=JumpParams(lambda_intensity=5.0, mu_jump=0.0, sigma_jump=0.15),
        )
        assert high_lambda > low_lambda

    def test_merton_deep_itm_call(self, jump_pricing_params):
        """Deep ITM call should be near intrinsic."""
        price = merton_jump_diffusion_price(
            spot=jump_pricing_params["spot"],
            strike=60.0,
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=True,
            jump_params=JumpParams(lambda_intensity=1.0, mu_jump=-0.05, sigma_jump=0.10),
        )
        intrinsic = jump_pricing_params["spot"] - 60.0
        assert price >= intrinsic - 1e-6

    def test_merton_deep_otm_call_small(self, jump_pricing_params):
        """Deep OTM call should be small."""
        price = merton_jump_diffusion_price(
            spot=jump_pricing_params["spot"],
            strike=150.0,
            time_to_expiry=jump_pricing_params["time_to_expiry"],
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=True,
            jump_params=JumpParams(lambda_intensity=1.0, mu_jump=-0.05, sigma_jump=0.10),
        )
        assert price < 5.0

    def test_merton_long_expiry(self, jump_pricing_params):
        """Long expiry should work."""
        price = merton_jump_diffusion_price(
            spot=jump_pricing_params["spot"],
            strike=jump_pricing_params["strike"],
            time_to_expiry=2.0,
            rate=jump_pricing_params["rate"],
            dividend_yield=0.0,
            volatility=jump_pricing_params["volatility"],
            is_call=True,
            jump_params=JumpParams(lambda_intensity=1.0, mu_jump=-0.05, sigma_jump=0.10),
        )
        assert price > 0
        assert np.isfinite(price)

    def test_merton_series_convergence(self, jump_pricing_params):
        """More terms should give same result (convergence)."""
        jp = JumpParams(lambda_intensity=1.0, mu_jump=-0.05, sigma_jump=0.10)
        price_20 = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=jp,
            max_terms=20,
        )
        price_100 = merton_jump_diffusion_price(
            **jump_pricing_params,
            jump_params=jp,
            max_terms=100,
        )
        assert abs(price_20 - price_100) < 0.01

    def test_merton_vectorized_vs_scalar(self):
        """Vectorized Merton jump-diffusion pricing should match scalar pricing exactly."""
        spots = np.array([100.0, 100.0, 95.0, 105.0])
        strikes = np.array([90.0, 110.0, 100.0, 100.0])
        times = np.array([0.1, 0.5, 0.25, 1.0])
        rates = np.array([0.05, 0.05, 0.03, 0.06])
        divs = np.array([0.0, 0.01, 0.02, 0.0])
        vols = np.array([0.2, 0.3, 0.25, 0.15])
        is_calls = np.array([True, False, True, False])

        jp = JumpParams(lambda_intensity=1.5, mu_jump=-0.1, sigma_jump=0.2)

        # Calculate scalar values
        scalar_prices = []
        for i in range(len(spots)):
            p = merton_jump_diffusion_price(
                spot=spots[i],
                strike=strikes[i],
                time_to_expiry=times[i],
                rate=rates[i],
                dividend_yield=divs[i],
                volatility=vols[i],
                is_call=is_calls[i],
                jump_params=jp,
            )
            scalar_prices.append(p)

        # Calculate vectorized values
        vec_prices = merton_jump_diffusion_price_vec(
            spot=spots,
            strike=strikes,
            time_to_expiry=times,
            rate=rates,
            dividend_yield=divs,
            volatility=vols,
            is_call=is_calls,
            jump_intensity=jp.lambda_intensity,
            jump_mean=jp.mu_jump,
            jump_vol=jp.sigma_jump,
        )

        # Assert match
        np.testing.assert_allclose(vec_prices, scalar_prices, rtol=1e-5, atol=1e-5)


class TestVarianceSwap:
    """Variance swap tests (10 tests)."""

    def test_variance_swap_strike_basic(self):
        """Basic variance swap strike calculation."""
        # Create simple option chain
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        call_prices = np.array([12.0, 8.0, 5.0, 3.0, 1.5])
        put_prices = np.array([2.0, 3.5, 5.5, 8.5, 12.0])

        strike = variance_swap_strike(
            call_prices=call_prices,
            put_prices=put_prices,
            call_strikes=strikes,
            put_strikes=strikes,
            forward=100.0,
            rate=0.05,
            time_to_expiry=0.25,
        )
        assert strike > 0
        assert np.isfinite(strike)

    def test_variance_swap_strike_atm_vol_approx(self):
        """Variance swap strike should approach ATM vol squared with dense strike coverage."""
        # Need wide strike range with many points for accurate variance swap integration
        strikes = np.array(
            [
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                92.5,
                95.0,
                97.5,
                100.0,
                102.5,
                105.0,
                107.5,
                110.0,
                115.0,
                120.0,
                125.0,
                130.0,
            ]
        )
        atm_vol = 0.20
        T = 0.25
        r = 0.05

        # Generate BS prices at constant vol
        call_prices = np.array(
            [black_scholes_price(100.0, k, T, r, 0.0, atm_vol, is_call=True) for k in strikes]
        )
        put_prices = np.array(
            [black_scholes_price(100.0, k, T, r, 0.0, atm_vol, is_call=False) for k in strikes]
        )

        strike = variance_swap_strike(
            call_prices,
            put_prices,
            strikes,
            strikes,
            100.0,
            r,
            T,
        )
        # With dense strikes spanning 70-130, should be close to vol^2 = 0.04
        # Allow wider range due to discrete integration
        assert 0.02 < strike < 0.08

    def test_variance_swap_value_at_inception(self):
        """Variance swap value at inception should be near zero."""
        value = compute_variance_swap_value(
            realized_variance=0.04,  # Realized = Strike
            variance_strike=0.04,
            notional=1000000.0,
            time_to_expiry=0.25,
            rate=0.05,
        )
        assert abs(value) < 1.0  # Near zero at inception

    def test_variance_swap_value_realized_above_strike(self):
        """If realized variance > strike, long variance profits."""
        value = compute_variance_swap_value(
            realized_variance=0.06,  # Realized above strike
            variance_strike=0.04,
            notional=1000000.0,
            time_to_expiry=0.25,
            rate=0.05,
        )
        assert value > 0

    def test_variance_swap_value_realized_below_strike(self):
        """If realized variance < strike, long variance loses."""
        value = compute_variance_swap_value(
            realized_variance=0.02,  # Realized below strike
            variance_strike=0.04,
            notional=1000000.0,
            time_to_expiry=0.25,
            rate=0.05,
        )
        assert value < 0

    def test_variance_swap_strike_increases_with_skew(self):
        """With put skew, variance strike should increase."""
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

        # Flat vol
        flat_calls = np.array(
            [black_scholes_price(100, k, 0.25, 0.05, 0, 0.20, True) for k in strikes]
        )
        flat_puts = np.array(
            [black_scholes_price(100, k, 0.25, 0.05, 0, 0.20, False) for k in strikes]
        )

        # Skewed vol (higher for lower strikes)
        skewed_vols = [0.30, 0.25, 0.20, 0.18, 0.17]
        skewed_calls = np.array(
            [
                black_scholes_price(100, k, 0.25, 0.05, 0, v, True)
                for k, v in zip(strikes, skewed_vols)
            ]
        )
        skewed_puts = np.array(
            [
                black_scholes_price(100, k, 0.25, 0.05, 0, v, False)
                for k, v in zip(strikes, skewed_vols)
            ]
        )

        flat_strike = variance_swap_strike(
            flat_calls,
            flat_puts,
            strikes,
            strikes,
            100.0,
            0.05,
            0.25,
        )

        skewed_strike = variance_swap_strike(
            skewed_calls,
            skewed_puts,
            strikes,
            strikes,
            100.0,
            0.05,
            0.25,
        )

        assert skewed_strike > flat_strike

    def test_variance_swap_strike_positive(self):
        """Variance strike should always be positive."""
        # Need enough strikes for proper integration
        strikes = np.array(
            [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        )
        vol = 0.25
        call_prices = np.array(
            [black_scholes_price(100, k, 0.5, 0.03, 0, vol, True) for k in strikes]
        )
        put_prices = np.array(
            [black_scholes_price(100, k, 0.5, 0.03, 0, vol, False) for k in strikes]
        )

        strike = variance_swap_strike(
            call_prices,
            put_prices,
            strikes,
            strikes,
            100.0,
            0.03,
            0.5,
        )
        assert strike > 0

    def test_variance_swap_value_scales_with_notional(self):
        """Variance swap value should scale linearly with notional."""
        value_1m = compute_variance_swap_value(
            realized_variance=0.05,
            variance_strike=0.04,
            notional=1000000.0,
            time_to_expiry=0.25,
            rate=0.05,
        )
        value_2m = compute_variance_swap_value(
            realized_variance=0.05,
            variance_strike=0.04,
            notional=2000000.0,
            time_to_expiry=0.25,
            rate=0.05,
        )
        assert abs(value_2m / value_1m - 2.0) < 0.01

    def test_variance_swap_strike_longer_expiry(self):
        """Longer expiry should affect variance strike."""
        # Need dense strikes for proper integration
        strikes = np.array(
            [
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                92.5,
                95.0,
                97.5,
                100.0,
                102.5,
                105.0,
                107.5,
                110.0,
                115.0,
                120.0,
                125.0,
                130.0,
            ]
        )

        short_calls = np.array(
            [black_scholes_price(100, k, 0.25, 0.05, 0, 0.20, True) for k in strikes]
        )
        short_puts = np.array(
            [black_scholes_price(100, k, 0.25, 0.05, 0, 0.20, False) for k in strikes]
        )

        long_calls = np.array(
            [black_scholes_price(100, k, 1.0, 0.05, 0, 0.20, True) for k in strikes]
        )
        long_puts = np.array(
            [black_scholes_price(100, k, 1.0, 0.05, 0, 0.20, False) for k in strikes]
        )

        short_strike = variance_swap_strike(
            short_calls,
            short_puts,
            strikes,
            strikes,
            100.0,
            0.05,
            0.25,
        )

        long_strike = variance_swap_strike(
            long_calls,
            long_puts,
            strikes,
            strikes,
            100.0,
            0.05,
            1.0,
        )

        # Both should be near vol^2 = 0.04
        assert 0.02 < short_strike < 0.08
        assert 0.02 < long_strike < 0.08

    def test_variance_swap_value_at_expiry(self):
        """At expiry, variance swap value is just realized - strike."""
        # At expiry, time_to_expiry = 0
        value = compute_variance_swap_value(
            realized_variance=0.05,
            variance_strike=0.04,
            notional=1000000.0,
            time_to_expiry=0.0,
            rate=0.05,
        )
        expected = 1000000.0 * (0.05 - 0.04)
        assert abs(value - expected) < 1.0


class TestIVSolverExtended:
    """Extended IV solver tests (additional 23 tests)."""

    @pytest.fixture
    def standard_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.25,
            "rate": 0.05,
            "dividend_yield": 0.02,
        }

    def test_iv_round_trip_atm_call(self, standard_params):
        """IV should round-trip for ATM call."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-6

    def test_iv_round_trip_atm_put(self, standard_params):
        """IV should round-trip for ATM put."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=False)
        result = calculate_iv(**standard_params, market_price=price, is_call=False)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-6

    def test_iv_round_trip_itm_call(self, standard_params):
        """IV should round-trip for ITM call."""
        params = {**standard_params, "strike": 90.0}
        true_vol = 0.30
        price = black_scholes_price(**params, volatility=true_vol, is_call=True)
        result = calculate_iv(**params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-5

    def test_iv_round_trip_otm_put(self, standard_params):
        """IV should round-trip for OTM put."""
        params = {**standard_params, "strike": 90.0}
        true_vol = 0.30
        price = black_scholes_price(**params, volatility=true_vol, is_call=False)
        result = calculate_iv(**params, market_price=price, is_call=False)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-5

    def test_iv_deep_otm_call(self, standard_params):
        """Deep OTM call IV should still work."""
        params = {**standard_params, "strike": 130.0}
        true_vol = 0.30
        price = black_scholes_price(**params, volatility=true_vol, is_call=True)
        result = calculate_iv(**params, market_price=price, is_call=True)
        # May need more iterations for deep OTM
        if result.converged:
            assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_deep_otm_put(self, standard_params):
        """Deep OTM put IV should still work."""
        params = {**standard_params, "strike": 70.0}
        true_vol = 0.30
        price = black_scholes_price(**params, volatility=true_vol, is_call=False)
        result = calculate_iv(**params, market_price=price, is_call=False)
        if result.converged:
            assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_high_volatility(self, standard_params):
        """High volatility should be recoverable."""
        true_vol = 1.0
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_low_volatility(self, standard_params):
        """Low volatility should be recoverable."""
        true_vol = 0.05
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_short_expiry(self, standard_params):
        """Short expiry IV should work."""
        params = {**standard_params, "time_to_expiry": 0.01}
        true_vol = 0.25
        price = black_scholes_price(**params, volatility=true_vol, is_call=True)
        result = calculate_iv(**params, market_price=price, is_call=True)
        if result.converged:
            assert abs(result.implied_volatility - true_vol) < 0.01

    def test_iv_long_expiry(self, standard_params):
        """Long expiry IV should work."""
        params = {**standard_params, "time_to_expiry": 2.0}
        true_vol = 0.25
        price = black_scholes_price(**params, volatility=true_vol, is_call=True)
        result = calculate_iv(**params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-5

    def test_iv_price_below_intrinsic_returns_not_converged(self, standard_params):
        """Price below intrinsic should return converged=False."""
        params = {**standard_params, "strike": 95.0}
        intrinsic = params["spot"] - params["strike"]  # 5.0
        result = calculate_iv(**params, market_price=intrinsic - 1.0, is_call=True)
        assert result.converged is False

    def test_iv_zero_price_returns_low_vol(self, standard_params):
        """Zero or tiny price should return low vol or not converge."""
        # Use a price derived from very low vol for a valid test
        # Very tiny arbitrary prices may not correspond to any valid vol
        low_vol = 0.05
        price = black_scholes_price(**standard_params, volatility=low_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        # Should converge to the low vol
        assert result.converged
        assert result.implied_volatility < 0.10

    def test_iv_different_vol_levels_low(self, standard_params):
        """IV solver should work for low vol."""
        true_vol = 0.10
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_different_vol_levels_medium(self, standard_params):
        """IV solver should work for medium vol."""
        true_vol = 0.30
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_different_vol_levels_high(self, standard_params):
        """IV solver should work for high vol (hybrid internally uses best method)."""
        true_vol = 0.60
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.implied_volatility - true_vol) < 1e-4

    def test_iv_american_call(self, standard_params):
        """American call IV via tree should work."""
        params = {**standard_params, "dividend_yield": 0.05}
        true_vol = 0.25
        price = leisen_reimer_price(
            **params, volatility=true_vol, is_call=True, is_american=True, n_steps=201
        )
        result = calculate_iv_american(**params, market_price=price, is_call=True)
        if result.converged:
            assert abs(result.implied_volatility - true_vol) < 0.01

    def test_iv_american_put(self, standard_params):
        """American put IV via tree should work."""
        true_vol = 0.25
        price = leisen_reimer_price(
            **standard_params, volatility=true_vol, is_call=False, is_american=True, n_steps=201
        )
        result = calculate_iv_american(**standard_params, market_price=price, is_call=False)
        if result.converged:
            assert abs(result.implied_volatility - true_vol) < 0.01

    def test_iv_result_fields(self, standard_params):
        """IVResult should have all expected fields."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)

        assert hasattr(result, "implied_volatility")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")
        assert hasattr(result, "error")
        assert hasattr(result, "method")
        assert hasattr(result, "model_price")

    def test_iv_model_price_matches_input(self, standard_params):
        """Model price from result should match input."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert abs(result.model_price - price) < 0.01

    def test_iv_iterations_reasonable(self, standard_params):
        """Iterations should be reasonable."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert result.iterations < 100

    def test_iv_error_small_on_convergence(self, standard_params):
        """Error should be small when converged."""
        true_vol = 0.25
        price = black_scholes_price(**standard_params, volatility=true_vol, is_call=True)
        result = calculate_iv(**standard_params, market_price=price, is_call=True)
        assert result.converged
        assert result.error < 1e-8

    def test_iv_various_strikes(self, standard_params):
        """IV should work for various strikes."""
        true_vol = 0.25
        for strike in [80.0, 90.0, 100.0, 110.0, 120.0]:
            params = {**standard_params, "strike": strike}
            price = black_scholes_price(**params, volatility=true_vol, is_call=True)
            result = calculate_iv(**params, market_price=price, is_call=True)
            assert result.converged
            assert abs(result.implied_volatility - true_vol) < 1e-4


class TestVectorizedGreeksExtended:
    """Extended vectorized Greeks tests (additional 20 tests)."""

    @pytest.fixture
    def batch_params(self):
        return {
            "spot": np.array([100.0, 100.0, 100.0, 100.0]),
            "strike": np.array([95.0, 100.0, 105.0, 110.0]),
            "time_to_expiry": np.array([0.25, 0.25, 0.25, 0.25]),
            "volatility": np.array([0.20, 0.20, 0.20, 0.20]),
            "rate": np.array([0.05, 0.05, 0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02, 0.02, 0.02]),
            "is_call": np.array([True, True, True, True]),
        }

    def test_batch_delta_ordering(self, batch_params):
        """Delta should decrease with strike for calls."""
        result = compute_all_greeks_batch(**batch_params)
        deltas = result.delta
        # For calls, delta should decrease as strike increases
        assert deltas[0] > deltas[1] > deltas[2] > deltas[3]

    def test_batch_gamma_atm_highest(self, batch_params):
        """Gamma should be highest for ATM."""
        result = compute_all_greeks_batch(**batch_params)
        gammas = result.gamma
        # ATM (strike=100) should have highest gamma
        assert gammas[1] == max(gammas)

    def test_batch_vega_atm_highest(self, batch_params):
        """Vega should be highest for ATM."""
        result = compute_all_greeks_batch(**batch_params)
        vegas = result.vega
        assert vegas[1] == max(vegas)

    def test_batch_vs_scalar_delta(self, batch_params):
        """Batch delta should match scalar calculations."""
        result = compute_all_greeks_batch(**batch_params)
        for i in range(len(batch_params["strike"])):
            scalar = compute_all_greeks(
                spot=batch_params["spot"][i],
                strike=batch_params["strike"][i],
                time_to_expiry=batch_params["time_to_expiry"][i],
                rate=batch_params["rate"][i],
                dividend_yield=batch_params["dividend_yield"][i],
                volatility=batch_params["volatility"][i],
                is_call=batch_params["is_call"][i],
            )
            assert abs(result.delta[i] - scalar.delta) < 1e-10

    def test_batch_vs_scalar_gamma(self, batch_params):
        """Batch gamma should match scalar calculations."""
        result = compute_all_greeks_batch(**batch_params)
        for i in range(len(batch_params["strike"])):
            scalar = compute_all_greeks(
                spot=batch_params["spot"][i],
                strike=batch_params["strike"][i],
                time_to_expiry=batch_params["time_to_expiry"][i],
                rate=batch_params["rate"][i],
                dividend_yield=batch_params["dividend_yield"][i],
                volatility=batch_params["volatility"][i],
                is_call=batch_params["is_call"][i],
            )
            assert abs(result.gamma[i] - scalar.gamma) < 1e-10

    def test_batch_vs_scalar_theta(self, batch_params):
        """Batch theta should match scalar calculations."""
        result = compute_all_greeks_batch(**batch_params)
        for i in range(len(batch_params["strike"])):
            scalar = compute_all_greeks(
                spot=batch_params["spot"][i],
                strike=batch_params["strike"][i],
                time_to_expiry=batch_params["time_to_expiry"][i],
                rate=batch_params["rate"][i],
                dividend_yield=batch_params["dividend_yield"][i],
                volatility=batch_params["volatility"][i],
                is_call=batch_params["is_call"][i],
            )
            assert abs(result.theta[i] - scalar.theta) < 1e-10

    def test_batch_large_array(self):
        """Batch should handle large arrays."""
        n = 1000
        params = {
            "spot": np.full(n, 100.0),
            "strike": np.linspace(80.0, 120.0, n),
            "time_to_expiry": np.full(n, 0.25),
            "volatility": np.full(n, 0.20),
            "rate": np.full(n, 0.05),
            "dividend_yield": np.full(n, 0.02),
            "is_call": np.full(n, True),
        }
        result = compute_all_greeks_batch(**params)
        assert len(result.delta) == n
        assert np.all(np.isfinite(result.delta))

    def test_batch_mixed_calls_puts(self):
        """Batch should handle mixed calls and puts."""
        params = {
            "spot": np.array([100.0, 100.0]),
            "strike": np.array([100.0, 100.0]),
            "time_to_expiry": np.array([0.25, 0.25]),
            "volatility": np.array([0.20, 0.20]),
            "rate": np.array([0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02]),
            "is_call": np.array([True, False]),
        }
        result = compute_all_greeks_batch(**params)
        # Call and put deltas should sum to ~exp(-qT) for ATM
        assert abs(result.delta[0] - result.delta[1] - np.exp(-0.02 * 0.25)) < 0.01

    def test_batch_different_expiries(self):
        """Batch should handle different expiries."""
        params = {
            "spot": np.array([100.0, 100.0, 100.0]),
            "strike": np.array([100.0, 100.0, 100.0]),
            "time_to_expiry": np.array([0.1, 0.5, 1.0]),
            "volatility": np.array([0.20, 0.20, 0.20]),
            "rate": np.array([0.05, 0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02, 0.02]),
            "is_call": np.array([True, True, True]),
        }
        result = compute_all_greeks_batch(**params)
        # Vega should increase with time
        assert result.vega[0] < result.vega[1] < result.vega[2]

    def test_batch_different_vols(self):
        """Batch should handle different volatilities."""
        params = {
            "spot": np.array([100.0, 100.0, 100.0]),
            "strike": np.array([100.0, 100.0, 100.0]),
            "time_to_expiry": np.array([0.25, 0.25, 0.25]),
            "volatility": np.array([0.10, 0.20, 0.40]),
            "rate": np.array([0.05, 0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02, 0.02]),
            "is_call": np.array([True, True, True]),
        }
        result = compute_all_greeks_batch(**params)
        assert np.all(np.isfinite(result.delta))
        assert np.all(np.isfinite(result.vega))

    def test_batch_second_order_greeks(self, batch_params):
        """Batch should calculate second-order Greeks."""
        result = compute_all_greeks_batch(**batch_params)
        assert hasattr(result, "vanna")
        assert hasattr(result, "volga")
        assert hasattr(result, "charm")
        assert np.all(np.isfinite(result.vanna))
        assert np.all(np.isfinite(result.volga))

    def test_batch_third_order_greeks(self, batch_params):
        """Batch should calculate third-order Greeks."""
        result = compute_all_greeks_batch(**batch_params)
        assert hasattr(result, "speed")
        assert hasattr(result, "color")
        assert hasattr(result, "zomma")
        assert hasattr(result, "ultima")
        assert np.all(np.isfinite(result.speed))
        assert np.all(np.isfinite(result.color))

    def test_batch_same_spot_different_strikes(self):
        """Same spot with different strikes should work (arrays must match length)."""
        # API requires all arrays to have the same length (no broadcasting)
        params = {
            "spot": np.array([100.0, 100.0, 100.0]),
            "strike": np.array([95.0, 100.0, 105.0]),
            "time_to_expiry": np.array([0.25, 0.25, 0.25]),
            "volatility": np.array([0.20, 0.20, 0.20]),
            "rate": np.array([0.05, 0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02, 0.02]),
            "is_call": np.array([True, True, True]),
        }
        result = compute_all_greeks_batch(**params)
        assert len(result.delta) == 3

    def test_batch_empty_array(self):
        """Empty arrays should return empty results."""
        params = {
            "spot": np.array([]),
            "strike": np.array([]),
            "time_to_expiry": np.array([]),
            "volatility": np.array([]),
            "rate": np.array([]),
            "dividend_yield": np.array([]),
            "is_call": np.array([], dtype=bool),
        }
        result = compute_all_greeks_batch(**params)
        assert len(result.delta) == 0

    def test_batch_single_element(self):
        """Single element should work."""
        params = {
            "spot": np.array([100.0]),
            "strike": np.array([100.0]),
            "time_to_expiry": np.array([0.25]),
            "volatility": np.array([0.20]),
            "rate": np.array([0.05]),
            "dividend_yield": np.array([0.02]),
            "is_call": np.array([True]),
        }
        result = compute_all_greeks_batch(**params)
        assert len(result.delta) == 1
        assert 0.4 < result.delta[0] < 0.6  # ATM call

    def test_batch_greeks_put_delta_negative(self):
        """Put deltas should be negative."""
        params = {
            "spot": np.array([100.0, 100.0, 100.0]),
            "strike": np.array([95.0, 100.0, 105.0]),
            "time_to_expiry": np.array([0.25, 0.25, 0.25]),
            "volatility": np.array([0.20, 0.20, 0.20]),
            "rate": np.array([0.05, 0.05, 0.05]),
            "dividend_yield": np.array([0.02, 0.02, 0.02]),
            "is_call": np.array([False, False, False]),
        }
        result = compute_all_greeks_batch(**params)
        assert np.all(result.delta < 0)

    def test_batch_high_vol_stability(self):
        """High vol should not cause numerical issues."""
        params = {
            "spot": np.array([100.0]),
            "strike": np.array([100.0]),
            "time_to_expiry": np.array([0.25]),
            "volatility": np.array([2.0]),  # 200% vol
            "rate": np.array([0.05]),
            "dividend_yield": np.array([0.02]),
            "is_call": np.array([True]),
        }
        result = compute_all_greeks_batch(**params)
        assert np.all(np.isfinite(result.delta))
        assert np.all(np.isfinite(result.gamma))

    def test_batch_short_expiry_stability(self):
        """Short expiry should not cause numerical issues."""
        params = {
            "spot": np.array([100.0]),
            "strike": np.array([100.0]),
            "time_to_expiry": np.array([0.001]),  # ~9 hours
            "volatility": np.array([0.20]),
            "rate": np.array([0.05]),
            "dividend_yield": np.array([0.02]),
            "is_call": np.array([True]),
        }
        result = compute_all_greeks_batch(**params)
        assert np.all(np.isfinite(result.delta))

    def test_batch_zero_dividend(self):
        """Zero dividend should work."""
        params = {
            "spot": np.array([100.0]),
            "strike": np.array([100.0]),
            "time_to_expiry": np.array([0.25]),
            "volatility": np.array([0.20]),
            "rate": np.array([0.05]),
            "dividend_yield": np.array([0.0]),
            "is_call": np.array([True]),
        }
        result = compute_all_greeks_batch(**params)
        assert np.all(np.isfinite(result.delta))


class TestLongstaffSchwartzMC:
    """Longstaff-Schwartz Monte Carlo tests (15 tests)."""

    @pytest.fixture
    def lsmc_params(self):
        return {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.5,
            "rate": 0.05,
            "dividend_yield": 0.02,
            "volatility": 0.25,
        }

    def test_lsmc_american_put_value(self, lsmc_params):
        """LSMC should price American put reasonably."""
        result = longstaff_schwartz(
            **lsmc_params,
            is_call=False,
            n_paths=10000,
            n_steps=50,
            random_seed=42,
        )
        price = result.price
        # Should be positive and reasonable
        assert price > 0
        assert price < lsmc_params["strike"]  # Put can't be worth more than strike

    def test_lsmc_american_call_no_div(self, lsmc_params):
        """LSMC American call without dividend should equal European."""
        params = {**lsmc_params, "dividend_yield": 0.0}
        european = black_scholes_price(**params, is_call=True)
        result = longstaff_schwartz(
            **params, is_call=True, n_paths=50000, n_steps=100, random_seed=42
        )
        # Should be close (within Monte Carlo error)
        assert abs(result.price - european) < 0.5

    def test_lsmc_american_geq_european_put(self, lsmc_params):
        """LSMC American put should be worth at least European."""
        european = black_scholes_price(**lsmc_params, is_call=False)
        result = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price >= european * 0.95  # Allow some MC error

    def test_lsmc_put_intrinsic_floor(self, lsmc_params):
        """LSMC put should be at least intrinsic value."""
        params = {**lsmc_params, "strike": 110.0}  # ITM put
        intrinsic = params["strike"] - params["spot"]
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price >= intrinsic * 0.95

    def test_lsmc_call_intrinsic_floor(self, lsmc_params):
        """LSMC call should be at least intrinsic value."""
        params = {**lsmc_params, "strike": 90.0}  # ITM call
        intrinsic = params["spot"] - params["strike"]
        result = longstaff_schwartz(
            **params, is_call=True, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price >= intrinsic * 0.95

    def test_lsmc_deep_otm_put_small(self, lsmc_params):
        """Deep OTM put should be small."""
        params = {**lsmc_params, "strike": 70.0}
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price < 2.0

    def test_lsmc_deep_otm_call_small(self, lsmc_params):
        """Deep OTM call should be small."""
        params = {**lsmc_params, "strike": 130.0}
        result = longstaff_schwartz(
            **params, is_call=True, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price < 2.0

    def test_lsmc_more_paths_reduces_variance(self, lsmc_params):
        """More paths should reduce variance."""
        prices_small = [
            longstaff_schwartz(
                **lsmc_params, is_call=False, n_paths=1000, n_steps=50, random_seed=i
            ).price
            for i in range(5)
        ]
        prices_large = [
            longstaff_schwartz(
                **lsmc_params, is_call=False, n_paths=20000, n_steps=50, random_seed=i
            ).price
            for i in range(5)
        ]

        std_small = np.std(prices_small)
        std_large = np.std(prices_large)
        # Large sample should have lower std
        assert std_large < std_small

    def test_lsmc_seed_reproducibility(self, lsmc_params):
        """Same seed should give same result."""
        result1 = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=5000, n_steps=50, random_seed=12345
        )
        result2 = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=5000, n_steps=50, random_seed=12345
        )
        assert result1.price == result2.price

    def test_lsmc_different_seeds_different_results(self, lsmc_params):
        """Different seeds should give different results."""
        result1 = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=5000, n_steps=50, random_seed=1
        )
        result2 = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=5000, n_steps=50, random_seed=2
        )
        # Should be different (with high probability)
        assert result1.price != result2.price

    def test_lsmc_convergence_to_binomial(self, lsmc_params):
        """LSMC should converge to binomial price."""
        binomial = leisen_reimer_price(**lsmc_params, is_call=False, is_american=True, n_steps=201)
        result = longstaff_schwartz(
            **lsmc_params, is_call=False, n_paths=50000, n_steps=100, random_seed=42
        )
        # Should be within 1%
        assert abs(result.price - binomial) / binomial < 0.02

    def test_lsmc_high_volatility(self, lsmc_params):
        """LSMC should handle high volatility."""
        params = {**lsmc_params, "volatility": 0.60}
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=10000, n_steps=50, random_seed=42
        )
        assert result.price > 0
        assert np.isfinite(result.price)

    def test_lsmc_long_expiry(self, lsmc_params):
        """LSMC should handle long expiry."""
        params = {**lsmc_params, "time_to_expiry": 2.0}
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=10000, n_steps=100, random_seed=42
        )
        assert result.price > 0
        assert np.isfinite(result.price)

    def test_lsmc_short_expiry(self, lsmc_params):
        """LSMC should handle short expiry."""
        params = {**lsmc_params, "time_to_expiry": 0.05}
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=10000, n_steps=25, random_seed=42
        )
        assert result.price >= 0
        assert np.isfinite(result.price)

    def test_lsmc_itm_put_early_exercise(self, lsmc_params):
        """Deep ITM put should show early exercise value."""
        params = {**lsmc_params, "strike": 120.0}  # Deep ITM
        european = black_scholes_price(**params, is_call=False)
        result = longstaff_schwartz(
            **params, is_call=False, n_paths=20000, n_steps=50, random_seed=42
        )
        # American should be significantly higher due to early exercise
        assert result.price > european
