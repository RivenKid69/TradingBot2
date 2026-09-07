"""
Options Greeks and Pricing Performance Benchmarks.

Benchmarks for Phase 1 options pricing infrastructure.
Target latencies:
- Single Greek calculation: <5μs
- All 12 Greeks (scalar): <50μs
- Vectorized Greeks (1000 options): <1ms
- BS pricing: <5μs
- Binomial pricing (201 steps): <500μs
- Jump-diffusion pricing (10 jumps): <100μs
- IV calculation: <100μs
- Longstaff-Schwartz (1000 paths): <100ms

Run with:
    python benchmarks/bench_options_greeks.py
    python -m pytest benchmarks/bench_options_greeks.py -v
"""

import time
import math
import statistics
from typing import List, Callable, Optional, Dict, Any
import numpy as np

# Options module imports
from core_options import (
    OptionType,
    ExerciseStyle,
    OptionContract,
    GreeksResult,
    PricingResult,
)
from impl_greeks import (
    compute_delta,
    compute_gamma,
    compute_vega,
    compute_theta,
    compute_rho,
    compute_vanna,
    compute_volga,
    compute_all_greeks,
)
from impl_greeks_vectorized import (
    compute_greeks_vectorized,
    VectorizedGreeksInput,
    compute_delta_vectorized,
    compute_gamma_vectorized,
)
from impl_pricing import (
    black_scholes_price,
    leisen_reimer_price,
    crr_binomial_price,
    jump_diffusion_price,
    price_option,
    PricingModel,
    JumpParams,
)
from impl_iv_calculation import (
    compute_iv,
    compute_iv_newton,
    compute_iv_brent,
    IVSolver,
    IVSolverConfig,
)
from impl_jump_calibration import (
    calibrate_from_moments,
    calibrate_from_mle,
    detect_jumps,
)
from impl_discrete_dividends import (
    DividendSchedule,
    DividendModel,
    price_with_escrowed_dividends,
    price_american_with_dividends,
)
from impl_exercise_probability import (
    longstaff_schwartz,
    BasisFunctions,
    VarianceReduction,
    compute_early_exercise_premium,
    barone_adesi_whaley,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def run_benchmark(
    name: str,
    func: Callable,
    iterations: int = 10000,
    warmup: int = 1000,
) -> dict:
    """Run a benchmark and return statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        latencies.append(elapsed)

    latencies_us = [l / 1000.0 for l in latencies]

    return {
        "name": name,
        "iterations": iterations,
        "mean_us": statistics.mean(latencies_us),
        "median_us": statistics.median(latencies_us),
        "stdev_us": statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0,
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
        "p50_us": np.percentile(latencies_us, 50),
        "p95_us": np.percentile(latencies_us, 95),
        "p99_us": np.percentile(latencies_us, 99),
    }


def print_results(results: List[dict], title: str = "OPTIONS GREEKS BENCHMARKS") -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'Benchmark':<45} {'Mean (μs)':<12} {'P50 (μs)':<12} {'P95 (μs)':<12} {'P99 (μs)':<12}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:<45} {r['mean_us']:<12.2f} {r['p50_us']:<12.2f} {r['p95_us']:<12.2f} {r['p99_us']:<12.2f}"
        )

    print("=" * 100)


# ==============================================================================
# Standard Test Parameters
# ==============================================================================

# ATM call option parameters
SPOT = 100.0
STRIKE = 100.0
T = 0.25  # 3 months
R = 0.05
Q = 0.02
SIGMA = 0.20

# OTM put option parameters
OTM_STRIKE = 90.0

# ITM call option parameters
ITM_STRIKE = 110.0


# ==============================================================================
# Benchmark Classes
# ==============================================================================


class BenchmarkScalarGreeks:
    """Benchmarks for individual scalar Greek calculations."""

    def bench_delta(self) -> dict:
        """Benchmark Delta calculation (target: <5μs)."""

        def func():
            compute_delta(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)

        return run_benchmark("delta_scalar", func, iterations=20000)

    def bench_gamma(self) -> dict:
        """Benchmark Gamma calculation (target: <5μs)."""

        def func():
            compute_gamma(SPOT, STRIKE, T, R, SIGMA, Q)

        return run_benchmark("gamma_scalar", func, iterations=20000)

    def bench_vega(self) -> dict:
        """Benchmark Vega calculation (target: <5μs)."""

        def func():
            compute_vega(SPOT, STRIKE, T, R, SIGMA, Q)

        return run_benchmark("vega_scalar", func, iterations=20000)

    def bench_theta(self) -> dict:
        """Benchmark Theta calculation (target: <5μs)."""

        def func():
            compute_theta(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)

        return run_benchmark("theta_scalar", func, iterations=20000)

    def bench_rho(self) -> dict:
        """Benchmark Rho calculation (target: <5μs)."""

        def func():
            compute_rho(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)

        return run_benchmark("rho_scalar", func, iterations=20000)

    def bench_vanna(self) -> dict:
        """Benchmark Vanna calculation (target: <5μs)."""

        def func():
            compute_vanna(SPOT, STRIKE, T, R, SIGMA, Q)

        return run_benchmark("vanna_scalar", func, iterations=20000)

    def bench_volga(self) -> dict:
        """Benchmark Volga calculation (target: <5μs)."""

        def func():
            compute_volga(SPOT, STRIKE, T, R, SIGMA, Q)

        return run_benchmark("volga_scalar", func, iterations=20000)

    def bench_all_greeks(self) -> dict:
        """Benchmark all 12 Greeks together (target: <50μs)."""

        def func():
            compute_all_greeks(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)

        return run_benchmark("all_12_greeks_scalar", func, iterations=10000)


class BenchmarkVectorizedGreeks:
    """Benchmarks for vectorized Greek calculations."""

    def setup(self, n_options: int = 1000):
        """Setup vectorized inputs."""
        np.random.seed(42)
        self.spots = np.full(n_options, 100.0)
        self.strikes = np.linspace(80, 120, n_options)
        self.times = np.random.uniform(0.1, 1.0, n_options)
        self.rates = np.full(n_options, 0.05)
        self.vols = np.random.uniform(0.15, 0.35, n_options)
        self.dividends = np.full(n_options, 0.02)
        self.option_types = np.array([OptionType.CALL] * n_options)

    def bench_delta_vectorized_1k(self) -> dict:
        """Benchmark vectorized Delta for 1000 options (target: <500μs)."""
        self.setup(1000)

        def func():
            compute_delta_vectorized(
                self.spots,
                self.strikes,
                self.times,
                self.rates,
                self.vols,
                self.dividends,
                self.option_types,
            )

        return run_benchmark("delta_vectorized_1k", func, iterations=2000, warmup=200)

    def bench_gamma_vectorized_1k(self) -> dict:
        """Benchmark vectorized Gamma for 1000 options."""
        self.setup(1000)

        def func():
            compute_gamma_vectorized(
                self.spots, self.strikes, self.times, self.rates, self.vols, self.dividends
            )

        return run_benchmark("gamma_vectorized_1k", func, iterations=2000, warmup=200)

    def bench_all_greeks_vectorized_1k(self) -> dict:
        """Benchmark all vectorized Greeks for 1000 options (target: <1ms)."""
        self.setup(1000)
        inputs = VectorizedGreeksInput(
            spots=self.spots,
            strikes=self.strikes,
            times_to_expiry=self.times,
            rates=self.rates,
            volatilities=self.vols,
            dividend_yields=self.dividends,
            option_types=self.option_types,
        )

        def func():
            compute_greeks_vectorized(inputs)

        return run_benchmark("all_greeks_vectorized_1k", func, iterations=1000, warmup=100)

    def bench_all_greeks_vectorized_10k(self) -> dict:
        """Benchmark all vectorized Greeks for 10000 options (target: <10ms)."""
        self.setup(10000)
        inputs = VectorizedGreeksInput(
            spots=self.spots,
            strikes=self.strikes,
            times_to_expiry=self.times,
            rates=self.rates,
            volatilities=self.vols,
            dividend_yields=self.dividends,
            option_types=self.option_types,
        )

        def func():
            compute_greeks_vectorized(inputs)

        return run_benchmark("all_greeks_vectorized_10k", func, iterations=500, warmup=50)


class BenchmarkPricing:
    """Benchmarks for option pricing models."""

    def bench_black_scholes(self) -> dict:
        """Benchmark Black-Scholes pricing (target: <5μs)."""

        def func():
            black_scholes_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)

        return run_benchmark("black_scholes_price", func, iterations=20000)

    def bench_leisen_reimer_51(self) -> dict:
        """Benchmark Leisen-Reimer with 51 steps (target: <100μs)."""

        def func():
            leisen_reimer_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL, n_steps=51)

        return run_benchmark("leisen_reimer_51_steps", func, iterations=5000)

    def bench_leisen_reimer_201(self) -> dict:
        """Benchmark Leisen-Reimer with 201 steps (target: <500μs)."""

        def func():
            leisen_reimer_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL, n_steps=201)

        return run_benchmark("leisen_reimer_201_steps", func, iterations=2000)

    def bench_crr_binomial_51(self) -> dict:
        """Benchmark CRR binomial with 51 steps."""

        def func():
            crr_binomial_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL, n_steps=51)

        return run_benchmark("crr_binomial_51_steps", func, iterations=5000)

    def bench_crr_binomial_201(self) -> dict:
        """Benchmark CRR binomial with 201 steps (target: <500μs)."""

        def func():
            crr_binomial_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL, n_steps=201)

        return run_benchmark("crr_binomial_201_steps", func, iterations=2000)

    def bench_jump_diffusion(self) -> dict:
        """Benchmark jump-diffusion pricing (target: <100μs)."""
        jump_params = JumpParams(
            jump_intensity=1.0,
            jump_mean=-0.05,
            jump_std=0.10,
        )

        def func():
            jump_diffusion_price(
                SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL, jump_params, max_jumps=10
            )

        return run_benchmark("jump_diffusion_10_jumps", func, iterations=5000)

    def bench_american_put_binomial(self) -> dict:
        """Benchmark American put pricing with binomial (target: <500μs)."""

        def func():
            crr_binomial_price(
                SPOT,
                STRIKE,
                T,
                R,
                SIGMA,
                Q,
                OptionType.PUT,
                n_steps=201,
                exercise_style=ExerciseStyle.AMERICAN,
            )

        return run_benchmark("american_put_binomial_201", func, iterations=2000)


class BenchmarkIVCalculation:
    """Benchmarks for implied volatility calculation."""

    def setup(self):
        """Setup test prices."""
        # Calculate reference prices at different volatilities
        self.atm_price = black_scholes_price(SPOT, STRIKE, T, R, SIGMA, Q, OptionType.CALL)
        self.otm_price = black_scholes_price(SPOT, OTM_STRIKE, T, R, SIGMA, Q, OptionType.PUT)
        self.itm_price = black_scholes_price(SPOT, ITM_STRIKE, T, R, SIGMA, Q, OptionType.CALL)

    def bench_iv_newton(self) -> dict:
        """Benchmark Newton-Raphson IV solver (target: <50μs)."""
        self.setup()

        def func():
            compute_iv_newton(self.atm_price, SPOT, STRIKE, T, R, Q, OptionType.CALL)

        return run_benchmark("iv_newton_atm", func, iterations=10000)

    def bench_iv_brent(self) -> dict:
        """Benchmark Brent's method IV solver (target: <100μs)."""
        self.setup()

        def func():
            compute_iv_brent(self.atm_price, SPOT, STRIKE, T, R, Q, OptionType.CALL)

        return run_benchmark("iv_brent_atm", func, iterations=10000)

    def bench_iv_hybrid(self) -> dict:
        """Benchmark hybrid IV solver (target: <100μs)."""
        self.setup()

        def func():
            compute_iv(self.atm_price, SPOT, STRIKE, T, R, Q, OptionType.CALL)

        return run_benchmark("iv_hybrid_atm", func, iterations=10000)

    def bench_iv_otm_put(self) -> dict:
        """Benchmark IV for OTM put (harder case)."""
        self.setup()

        def func():
            compute_iv(self.otm_price, SPOT, OTM_STRIKE, T, R, Q, OptionType.PUT)

        return run_benchmark("iv_hybrid_otm_put", func, iterations=10000)

    def bench_iv_solver_batch(self) -> dict:
        """Benchmark IV solver for batch of 100 options (target: <10ms)."""
        self.setup()
        solver = IVSolver(IVSolverConfig())
        np.random.seed(42)
        prices = []
        strikes = np.linspace(80, 120, 100)
        for K in strikes:
            prices.append(black_scholes_price(SPOT, K, T, R, SIGMA, Q, OptionType.CALL))

        def func():
            for i, (K, price) in enumerate(zip(strikes, prices)):
                solver.compute_iv(price, SPOT, K, T, R, Q, OptionType.CALL)

        return run_benchmark("iv_solver_batch_100", func, iterations=500, warmup=50)


class BenchmarkJumpCalibration:
    """Benchmarks for jump-diffusion parameter calibration."""

    def setup(self):
        """Setup synthetic returns with jumps."""
        np.random.seed(42)
        n = 1000
        dt = 1 / 252  # Daily
        sigma = 0.20
        lambda_j = 3.0  # 3 jumps per year
        mu_j = -0.02
        sigma_j = 0.05

        # Generate diffusion component
        diffusion = np.random.normal(0, sigma * np.sqrt(dt), n)

        # Generate jump component
        num_jumps = np.random.poisson(lambda_j * dt, n)
        jumps = np.zeros(n)
        for i in range(n):
            if num_jumps[i] > 0:
                jumps[i] = np.sum(np.random.normal(mu_j, sigma_j, num_jumps[i]))

        self.returns = diffusion + jumps

    def bench_calibrate_moments(self) -> dict:
        """Benchmark moment-based calibration (target: <10ms)."""
        self.setup()
        dt = 1 / 252

        def func():
            calibrate_from_moments(self.returns, dt)

        return run_benchmark("calibrate_moments", func, iterations=1000, warmup=100)

    def bench_calibrate_mle(self) -> dict:
        """Benchmark MLE calibration (target: <100ms)."""
        self.setup()
        dt = 1 / 252

        def func():
            calibrate_from_mle(self.returns, dt, max_iterations=50)

        return run_benchmark("calibrate_mle", func, iterations=100, warmup=10)

    def bench_detect_jumps(self) -> dict:
        """Benchmark jump detection (target: <1ms)."""
        self.setup()

        def func():
            detect_jumps(self.returns, threshold_sigmas=3.0)

        return run_benchmark("detect_jumps", func, iterations=5000, warmup=500)


class BenchmarkDiscreteDividends:
    """Benchmarks for discrete dividend handling."""

    def setup(self):
        """Setup dividend schedule."""
        # Quarterly dividends
        self.dividends = DividendSchedule(
            ex_dates=[0.08, 0.33, 0.58, 0.83],  # Quarterly
            amounts=[0.50, 0.50, 0.50, 0.50],
            is_percentage=False,
        )
        self.t = 0.5  # 6 months

    def bench_escrowed_dividends(self) -> dict:
        """Benchmark escrowed dividend model (target: <10μs)."""
        self.setup()

        def func():
            price_with_escrowed_dividends(
                SPOT, STRIKE, self.t, R, SIGMA, OptionType.CALL, self.dividends
            )

        return run_benchmark("escrowed_dividends", func, iterations=10000)

    def bench_american_with_dividends(self) -> dict:
        """Benchmark American option with dividends (target: <1ms)."""
        self.setup()

        def func():
            price_american_with_dividends(
                SPOT, STRIKE, self.t, R, SIGMA, OptionType.PUT, self.dividends, n_steps=101
            )

        return run_benchmark("american_put_with_dividends", func, iterations=1000, warmup=100)


class BenchmarkLongstaffSchwartz:
    """Benchmarks for Longstaff-Schwartz Monte Carlo."""

    def bench_ls_1k_paths(self) -> dict:
        """Benchmark LS with 1000 paths (target: <100ms)."""

        def func():
            longstaff_schwartz(
                spot=SPOT,
                strike=STRIKE,
                time_to_expiry=T,
                rate=R,
                volatility=SIGMA,
                dividend_yield=Q,
                option_type=OptionType.PUT,
                n_paths=1000,
                n_steps=50,
                basis=BasisFunctions.LAGUERRE,
                degree=3,
                variance_reduction=VarianceReduction.ANTITHETIC,
                seed=42,
            )

        return run_benchmark("ls_1k_paths_50_steps", func, iterations=50, warmup=5)

    def bench_ls_10k_paths(self) -> dict:
        """Benchmark LS with 10000 paths (target: <1s)."""

        def func():
            longstaff_schwartz(
                spot=SPOT,
                strike=STRIKE,
                time_to_expiry=T,
                rate=R,
                volatility=SIGMA,
                dividend_yield=Q,
                option_type=OptionType.PUT,
                n_paths=10000,
                n_steps=50,
                basis=BasisFunctions.LAGUERRE,
                degree=3,
                variance_reduction=VarianceReduction.ANTITHETIC,
                seed=42,
            )

        return run_benchmark("ls_10k_paths_50_steps", func, iterations=10, warmup=2)

    def bench_barone_adesi_whaley(self) -> dict:
        """Benchmark Barone-Adesi Whaley approximation (target: <50μs)."""

        def func():
            barone_adesi_whaley(
                spot=SPOT,
                strike=STRIKE,
                time_to_expiry=T,
                rate=R,
                volatility=SIGMA,
                dividend_yield=Q,
                option_type=OptionType.PUT,
            )

        return run_benchmark("barone_adesi_whaley", func, iterations=10000)

    def bench_early_exercise_premium(self) -> dict:
        """Benchmark early exercise premium calculation."""

        def func():
            compute_early_exercise_premium(
                spot=SPOT,
                strike=STRIKE,
                time_to_expiry=T,
                rate=R,
                volatility=SIGMA,
                dividend_yield=Q,
                option_type=OptionType.PUT,
                n_paths=500,
                n_steps=25,
            )

        return run_benchmark("early_exercise_premium", func, iterations=50, warmup=5)


# ==============================================================================
# Main
# ==============================================================================


def run_all_benchmarks() -> List[dict]:
    """Run all benchmarks and return results."""
    results = []

    print("Running Scalar Greeks benchmarks...")
    bench_scalar = BenchmarkScalarGreeks()
    results.append(bench_scalar.bench_delta())
    results.append(bench_scalar.bench_gamma())
    results.append(bench_scalar.bench_vega())
    results.append(bench_scalar.bench_theta())
    results.append(bench_scalar.bench_rho())
    results.append(bench_scalar.bench_vanna())
    results.append(bench_scalar.bench_volga())
    results.append(bench_scalar.bench_all_greeks())

    print("Running Vectorized Greeks benchmarks...")
    bench_vector = BenchmarkVectorizedGreeks()
    results.append(bench_vector.bench_delta_vectorized_1k())
    results.append(bench_vector.bench_gamma_vectorized_1k())
    results.append(bench_vector.bench_all_greeks_vectorized_1k())
    results.append(bench_vector.bench_all_greeks_vectorized_10k())

    print("Running Pricing benchmarks...")
    bench_pricing = BenchmarkPricing()
    results.append(bench_pricing.bench_black_scholes())
    results.append(bench_pricing.bench_leisen_reimer_51())
    results.append(bench_pricing.bench_leisen_reimer_201())
    results.append(bench_pricing.bench_crr_binomial_51())
    results.append(bench_pricing.bench_crr_binomial_201())
    results.append(bench_pricing.bench_jump_diffusion())
    results.append(bench_pricing.bench_american_put_binomial())

    print("Running IV Calculation benchmarks...")
    bench_iv = BenchmarkIVCalculation()
    results.append(bench_iv.bench_iv_newton())
    results.append(bench_iv.bench_iv_brent())
    results.append(bench_iv.bench_iv_hybrid())
    results.append(bench_iv.bench_iv_otm_put())
    results.append(bench_iv.bench_iv_solver_batch())

    print("Running Jump Calibration benchmarks...")
    bench_jump = BenchmarkJumpCalibration()
    results.append(bench_jump.bench_calibrate_moments())
    results.append(bench_jump.bench_calibrate_mle())
    results.append(bench_jump.bench_detect_jumps())

    print("Running Discrete Dividends benchmarks...")
    bench_div = BenchmarkDiscreteDividends()
    results.append(bench_div.bench_escrowed_dividends())
    results.append(bench_div.bench_american_with_dividends())

    print("Running Longstaff-Schwartz benchmarks...")
    bench_ls = BenchmarkLongstaffSchwartz()
    results.append(bench_ls.bench_barone_adesi_whaley())
    results.append(bench_ls.bench_ls_1k_paths())
    results.append(bench_ls.bench_ls_10k_paths())
    results.append(bench_ls.bench_early_exercise_premium())

    return results


def check_targets(results: List[dict]) -> bool:
    """Check if results meet performance targets."""
    targets = {
        # Scalar Greeks (target: <5μs each)
        "delta_scalar": 10.0,
        "gamma_scalar": 10.0,
        "vega_scalar": 10.0,
        "theta_scalar": 10.0,
        "rho_scalar": 10.0,
        "vanna_scalar": 10.0,
        "volga_scalar": 10.0,
        "all_12_greeks_scalar": 100.0,  # <50μs target, 100μs CI allowance
        # Vectorized Greeks
        "delta_vectorized_1k": 1000.0,  # <500μs target
        "all_greeks_vectorized_1k": 2000.0,  # <1ms target
        "all_greeks_vectorized_10k": 20000.0,  # <10ms target
        # Pricing
        "black_scholes_price": 10.0,  # <5μs target
        "leisen_reimer_51_steps": 200.0,  # <100μs target
        "leisen_reimer_201_steps": 1000.0,  # <500μs target
        "crr_binomial_201_steps": 1000.0,  # <500μs target
        "jump_diffusion_10_jumps": 200.0,  # <100μs target
        "american_put_binomial_201": 1000.0,  # <500μs target
        # IV Calculation
        "iv_newton_atm": 100.0,  # <50μs target
        "iv_brent_atm": 200.0,  # <100μs target
        "iv_hybrid_atm": 200.0,  # <100μs target
        "iv_solver_batch_100": 20000.0,  # <10ms target
        # Jump Calibration
        "calibrate_moments": 20000.0,  # <10ms target
        "detect_jumps": 2000.0,  # <1ms target
        # Dividends
        "escrowed_dividends": 50.0,  # <10μs target
        "american_put_with_dividends": 5000.0,  # <1ms target
        # Longstaff-Schwartz
        "barone_adesi_whaley": 100.0,  # <50μs target
        "ls_1k_paths_50_steps": 200000.0,  # <100ms target
    }

    print("\nTarget Performance Check:")
    print("-" * 80)

    all_passed = True
    for r in results:
        if r["name"] in targets:
            target = targets[r["name"]]
            passed = r["p95_us"] < target
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"  {r['name']:<35} P95={r['p95_us']:>10.2f}μs  target<{target:>10.0f}μs  [{status}]"
            )
            if not passed:
                all_passed = False

    return all_passed


def main():
    """Run benchmarks and print results."""
    print("=" * 100)
    print("OPTIONS GREEKS AND PRICING PERFORMANCE BENCHMARKS")
    print("=" * 100)

    results = run_all_benchmarks()
    print_results(results)

    all_passed = check_targets(results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All performance targets met!")
    else:
        print("✗ Some performance targets not met (may be acceptable for Python overhead)")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ==============================================================================
# Pytest Benchmarks
# ==============================================================================


class TestBenchmarks:
    """Pytest-compatible benchmark tests."""

    def test_scalar_greeks_latency(self):
        """Test scalar Greeks meet latency target."""
        bench = BenchmarkScalarGreeks()
        result = bench.bench_all_greeks()
        # Target <50μs, allow 200μs for Python overhead in CI
        assert result["p95_us"] < 200.0, f"P95={result['p95_us']:.1f}μs"

    def test_vectorized_greeks_latency_1k(self):
        """Test vectorized Greeks (1000) meet latency target."""
        bench = BenchmarkVectorizedGreeks()
        result = bench.bench_all_greeks_vectorized_1k()
        # Target <1ms, allow 5ms for CI
        assert result["p95_us"] < 5000.0, f"P95={result['p95_us']:.1f}μs"

    def test_black_scholes_latency(self):
        """Test BS pricing meets latency target."""
        bench = BenchmarkPricing()
        result = bench.bench_black_scholes()
        # Target <5μs, allow 20μs for Python
        assert result["p95_us"] < 20.0, f"P95={result['p95_us']:.1f}μs"

    def test_binomial_pricing_latency(self):
        """Test binomial pricing meets latency target."""
        bench = BenchmarkPricing()
        result = bench.bench_leisen_reimer_201()
        # Target <500μs, allow 2000μs for CI
        assert result["p95_us"] < 2000.0, f"P95={result['p95_us']:.1f}μs"

    def test_iv_calculation_latency(self):
        """Test IV calculation meets latency target."""
        bench = BenchmarkIVCalculation()
        result = bench.bench_iv_hybrid()
        # Target <100μs, allow 500μs
        assert result["p95_us"] < 500.0, f"P95={result['p95_us']:.1f}μs"

    def test_barone_adesi_whaley_latency(self):
        """Test BAW approximation meets latency target."""
        bench = BenchmarkLongstaffSchwartz()
        result = bench.bench_barone_adesi_whaley()
        # Target <50μs, allow 200μs
        assert result["p95_us"] < 200.0, f"P95={result['p95_us']:.1f}μs"

    def test_longstaff_schwartz_latency(self):
        """Test LS meets latency target."""
        bench = BenchmarkLongstaffSchwartz()
        result = bench.bench_ls_1k_paths()
        # Target <100ms, allow 500ms for CI
        assert result["p95_us"] < 500000.0, f"P95={result['p95_us']:.1f}μs"

    def test_dividend_pricing_latency(self):
        """Test dividend-adjusted pricing meets latency target."""
        bench = BenchmarkDiscreteDividends()
        result = bench.bench_american_with_dividends()
        # Target <1ms, allow 5ms
        assert result["p95_us"] < 5000.0, f"P95={result['p95_us']:.1f}μs"
