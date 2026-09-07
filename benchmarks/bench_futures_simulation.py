"""
Phase 10: Futures Simulation Benchmarks

Performance benchmarks for futures simulation components:
- L2 Execution Provider
- L3 LOB Simulation (Crypto Futures)
- L3 LOB Simulation (CME Futures)
- Margin Calculation
- Funding Rate Computation
- Liquidation Engine
- Risk Guards

Target Performance:
- L2 execution: <100 μs per order
- L3 execution: <500 μs per order
- Margin calculation: <50 μs per position
- Funding rate: <10 μs per computation
- Risk guard check: <20 μs per check

Usage:
    python benchmarks/bench_futures_simulation.py
    python benchmarks/bench_futures_simulation.py --component l2_execution
    python benchmarks/bench_futures_simulation.py --iterations 10000
"""

import argparse
import time
import statistics
from decimal import Decimal
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    mean_time_us: float
    median_time_us: float
    min_time_us: float
    max_time_us: float
    std_time_us: float
    p95_time_us: float
    p99_time_us: float
    ops_per_sec: float
    target_us: Optional[float] = None
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 2),
            "mean_time_us": round(self.mean_time_us, 2),
            "median_time_us": round(self.median_time_us, 2),
            "min_time_us": round(self.min_time_us, 2),
            "max_time_us": round(self.max_time_us, 2),
            "std_time_us": round(self.std_time_us, 2),
            "p95_time_us": round(self.p95_time_us, 2),
            "p99_time_us": round(self.p99_time_us, 2),
            "ops_per_sec": round(self.ops_per_sec, 0),
            "target_us": self.target_us,
            "passed": self.passed,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time_sec: float = 0.0

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> Dict[str, Any]:
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        return {
            "suite": self.name,
            "total_benchmarks": len(self.results),
            "passed": passed,
            "failed": failed,
            "total_time_sec": round(self.total_time_sec, 2),
        }


def run_benchmark(
    name: str,
    func: Callable,
    iterations: int = 1000,
    warmup: int = 100,
    target_us: Optional[float] = None,
    setup: Optional[Callable] = None,
) -> BenchmarkResult:
    """
    Run a benchmark and collect timing statistics.

    Args:
        name: Benchmark name
        func: Function to benchmark (should take no arguments)
        iterations: Number of iterations
        warmup: Number of warmup iterations
        target_us: Target time in microseconds (for pass/fail)
        setup: Optional setup function called before each iteration

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup phase
    for _ in range(warmup):
        if setup:
            setup()
        func()

    # Timed phase
    times_ns: List[int] = []
    start_total = time.perf_counter_ns()

    for _ in range(iterations):
        if setup:
            setup()
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    end_total = time.perf_counter_ns()
    total_ns = end_total - start_total

    # Convert to microseconds
    times_us = [t / 1000.0 for t in times_ns]
    times_us_sorted = sorted(times_us)

    # Calculate statistics
    mean_us = statistics.mean(times_us)
    median_us = statistics.median(times_us)
    min_us = min(times_us)
    max_us = max(times_us)
    std_us = statistics.stdev(times_us) if len(times_us) > 1 else 0.0

    # Percentiles
    p95_idx = int(0.95 * len(times_us_sorted))
    p99_idx = int(0.99 * len(times_us_sorted))
    p95_us = times_us_sorted[min(p95_idx, len(times_us_sorted) - 1)]
    p99_us = times_us_sorted[min(p99_idx, len(times_us_sorted) - 1)]

    # Operations per second
    ops_per_sec = iterations / (total_ns / 1e9)

    # Pass/fail based on target
    passed = True
    if target_us is not None:
        passed = p95_us <= target_us

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_ns / 1e6,
        mean_time_us=mean_us,
        median_time_us=median_us,
        min_time_us=min_us,
        max_time_us=max_us,
        std_time_us=std_us,
        p95_time_us=p95_us,
        p99_time_us=p99_us,
        ops_per_sec=ops_per_sec,
        target_us=target_us,
        passed=passed,
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result in a formatted way."""
    status = "PASS" if result.passed else "FAIL"
    target_str = f" (target: {result.target_us} μs)" if result.target_us else ""

    print(f"\n{'='*60}")
    print(f"Benchmark: {result.name}")
    print(f"{'='*60}")
    print(f"  Iterations:    {result.iterations:,}")
    print(f"  Total time:    {result.total_time_ms:.2f} ms")
    print(f"  Mean:          {result.mean_time_us:.2f} μs")
    print(f"  Median:        {result.median_time_us:.2f} μs")
    print(f"  Min:           {result.min_time_us:.2f} μs")
    print(f"  Max:           {result.max_time_us:.2f} μs")
    print(f"  Std:           {result.std_time_us:.2f} μs")
    print(f"  P95:           {result.p95_time_us:.2f} μs{target_str}")
    print(f"  P99:           {result.p99_time_us:.2f} μs")
    print(f"  Ops/sec:       {result.ops_per_sec:,.0f}")
    print(f"  Status:        [{status}]")


# =============================================================================
# L2 Execution Provider Benchmarks
# =============================================================================


def bench_l2_crypto_futures_slippage(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark L2 crypto futures slippage calculation."""
    try:
        from execution_providers_futures import (
            FuturesSlippageProvider,
            FuturesSlippageConfig,
        )
        from execution_providers import Order, MarketState

        config = FuturesSlippageConfig()
        provider = FuturesSlippageProvider(config=config)

        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
        )
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=50000.0,
            ask=50001.0,
            adv=1e9,
        )

        def benchmark_func():
            provider.compute_slippage_bps(
                order=order,
                market=market,
                participation_ratio=0.001,
                funding_rate=0.0001,
                open_interest=2e9,
                recent_liquidations=1e7,
            )

        return run_benchmark(
            name="L2 Crypto Futures Slippage",
            func=benchmark_func,
            iterations=iterations,
            target_us=100.0,
        )
    except ImportError as e:
        print(f"Skipping L2 crypto futures slippage benchmark: {e}")
        return BenchmarkResult(
            name="L2 Crypto Futures Slippage",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=100.0,
            passed=False,
        )


def bench_l2_cme_futures_slippage(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark L2 CME futures slippage calculation."""
    try:
        from execution_providers_cme import CMESlippageProvider
        from execution_providers import Order, MarketState

        provider = CMESlippageProvider.from_profile("default")

        order = Order(
            symbol="ES",
            side="BUY",
            qty=5.0,
            order_type="MARKET",
        )
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=4500.0,
            ask=4500.25,
            adv=2e9,
        )

        def benchmark_func():
            provider.compute_slippage_bps(
                order=order,
                market=market,
                participation_ratio=0.001,
            )

        return run_benchmark(
            name="L2 CME Futures Slippage",
            func=benchmark_func,
            iterations=iterations,
            target_us=100.0,
        )
    except ImportError as e:
        print(f"Skipping L2 CME futures slippage benchmark: {e}")
        return BenchmarkResult(
            name="L2 CME Futures Slippage",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=100.0,
            passed=False,
        )


# =============================================================================
# L3 LOB Benchmarks
# =============================================================================


def bench_l3_crypto_futures_matching(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark L3 crypto futures matching engine."""
    try:
        from lob.matching_engine import MatchingEngine
        from lob.data_structures import LimitOrder, Side, OrderType

        engine = MatchingEngine(symbol="BTCUSDT")

        # Setup: Add resting orders
        for i in range(100):
            engine.add_resting_order(
                LimitOrder(
                    order_id=f"rest_buy_{i}",
                    price=49990.0 - i * 0.5,
                    qty=0.1,
                    remaining_qty=0.1,
                    timestamp_ns=i,
                    side=Side.BUY,
                )
            )
            engine.add_resting_order(
                LimitOrder(
                    order_id=f"rest_sell_{i}",
                    price=50010.0 + i * 0.5,
                    qty=0.1,
                    remaining_qty=0.1,
                    timestamp_ns=i,
                    side=Side.SELL,
                )
            )

        order_counter = [0]

        def benchmark_func():
            order_counter[0] += 1
            aggressive = LimitOrder(
                order_id=f"aggr_{order_counter[0]}",
                price=50010.0,  # Cross spread
                qty=0.05,
                remaining_qty=0.05,
                timestamp_ns=order_counter[0] * 1000,
                side=Side.BUY,
                order_type=OrderType.MARKET,
            )
            engine.match(aggressive)

        return run_benchmark(
            name="L3 Crypto Futures Matching",
            func=benchmark_func,
            iterations=iterations,
            target_us=500.0,
        )
    except ImportError as e:
        print(f"Skipping L3 crypto futures matching benchmark: {e}")
        return BenchmarkResult(
            name="L3 Crypto Futures Matching",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=500.0,
            passed=False,
        )


def bench_l3_cme_futures_matching(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark L3 CME futures matching engine (Globex)."""
    try:
        from lob.cme_matching import GlobexMatchingEngine
        from lob.data_structures import LimitOrder, Side, OrderType

        engine = GlobexMatchingEngine(symbol="ES", tick_size=0.25, protection_points=6)

        # Setup: Add resting orders
        for i in range(100):
            engine.add_resting_order(
                LimitOrder(
                    order_id=f"rest_buy_{i}",
                    price=4499.0 - i * 0.25,
                    qty=5.0,
                    remaining_qty=5.0,
                    timestamp_ns=i,
                    side=Side.BUY,
                )
            )
            engine.add_resting_order(
                LimitOrder(
                    order_id=f"rest_sell_{i}",
                    price=4501.0 + i * 0.25,
                    qty=5.0,
                    remaining_qty=5.0,
                    timestamp_ns=i,
                    side=Side.SELL,
                )
            )

        order_counter = [0]

        def benchmark_func():
            order_counter[0] += 1
            aggressive = LimitOrder(
                order_id=f"aggr_{order_counter[0]}",
                price=4501.0,  # Cross spread
                qty=2.0,
                remaining_qty=2.0,
                timestamp_ns=order_counter[0] * 1000,
                side=Side.BUY,
                order_type=OrderType.MARKET,
            )
            engine.match(aggressive)

        return run_benchmark(
            name="L3 CME Futures Matching (Globex)",
            func=benchmark_func,
            iterations=iterations,
            target_us=500.0,
        )
    except ImportError as e:
        print(f"Skipping L3 CME futures matching benchmark: {e}")
        return BenchmarkResult(
            name="L3 CME Futures Matching (Globex)",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=500.0,
            passed=False,
        )


# =============================================================================
# Margin Calculation Benchmarks
# =============================================================================


def bench_crypto_margin_calculation(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark crypto futures margin calculation."""
    try:
        from impl_futures_margin import TieredMarginCalculator
        from core_futures import FuturesPosition, PositionSide, MarginMode

        calculator = TieredMarginCalculator()

        position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("0.5"),
            entry_price=Decimal("50000"),
            side=PositionSide.LONG,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
        )

        def benchmark_func():
            calculator.calculate_initial_margin(
                position=position,
                current_price=Decimal("50500"),
            )

        return run_benchmark(
            name="Crypto Futures Margin Calculation",
            func=benchmark_func,
            iterations=iterations,
            target_us=50.0,
        )
    except ImportError as e:
        print(f"Skipping crypto margin calculation benchmark: {e}")
        return BenchmarkResult(
            name="Crypto Futures Margin Calculation",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=50.0,
            passed=False,
        )


def bench_span_margin_calculation(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark CME SPAN margin calculation."""
    try:
        from impl_span_margin import SPANMarginCalculator
        from core_futures import FuturesPosition, PositionSide, MarginMode

        calculator = SPANMarginCalculator()

        position = FuturesPosition(
            symbol="ES",
            qty=Decimal("5"),
            entry_price=Decimal("4500"),
            side=PositionSide.LONG,
            leverage=1,
            margin_mode=MarginMode.SPAN,
        )

        def benchmark_func():
            calculator.calculate_margin(
                position=position,
                current_price=Decimal("4520"),
            )

        return run_benchmark(
            name="CME SPAN Margin Calculation",
            func=benchmark_func,
            iterations=iterations,
            target_us=100.0,
        )
    except ImportError as e:
        print(f"Skipping SPAN margin calculation benchmark: {e}")
        return BenchmarkResult(
            name="CME SPAN Margin Calculation",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=100.0,
            passed=False,
        )


# =============================================================================
# Funding Rate Benchmarks
# =============================================================================


def bench_funding_rate_calculation(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark funding rate calculation."""
    try:
        from execution_providers_futures import FuturesFeeProvider

        provider = FuturesFeeProvider()

        def benchmark_func():
            provider.compute_funding_payment(
                position_notional=100000.0,
                funding_rate=0.0001,
                is_long=True,
            )

        return run_benchmark(
            name="Funding Rate Calculation",
            func=benchmark_func,
            iterations=iterations,
            target_us=10.0,
        )
    except ImportError as e:
        print(f"Skipping funding rate calculation benchmark: {e}")
        return BenchmarkResult(
            name="Funding Rate Calculation",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=10.0,
            passed=False,
        )


# =============================================================================
# Liquidation Engine Benchmarks
# =============================================================================


def bench_liquidation_price_calculation(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark liquidation price calculation."""
    try:
        from impl_futures_margin import TieredMarginCalculator
        from core_futures import FuturesPosition, PositionSide, MarginMode

        calculator = TieredMarginCalculator()

        position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("0.5"),
            entry_price=Decimal("50000"),
            side=PositionSide.LONG,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
        )

        def benchmark_func():
            calculator.calculate_liquidation_price(
                position=position,
                wallet_balance=Decimal("2500"),
            )

        return run_benchmark(
            name="Liquidation Price Calculation",
            func=benchmark_func,
            iterations=iterations,
            target_us=50.0,
        )
    except ImportError as e:
        print(f"Skipping liquidation price calculation benchmark: {e}")
        return BenchmarkResult(
            name="Liquidation Price Calculation",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=50.0,
            passed=False,
        )


def bench_liquidation_cascade_simulation(iterations: int = 500) -> BenchmarkResult:
    """Benchmark liquidation cascade simulation."""
    try:
        from lob.futures_extensions import LiquidationCascadeSimulator

        simulator = LiquidationCascadeSimulator(
            price_impact_coef=0.5,
            cascade_decay=0.7,
            max_waves=5,
        )

        def benchmark_func():
            simulator.simulate_cascade(
                initial_liquidation_volume=1_000_000,
                market_price=50000.0,
                adv=500_000_000,
            )

        return run_benchmark(
            name="Liquidation Cascade Simulation",
            func=benchmark_func,
            iterations=iterations,
            target_us=200.0,
        )
    except ImportError as e:
        print(f"Skipping liquidation cascade simulation benchmark: {e}")
        return BenchmarkResult(
            name="Liquidation Cascade Simulation",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=200.0,
            passed=False,
        )


# =============================================================================
# Risk Guard Benchmarks
# =============================================================================


def bench_leverage_guard_check(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark leverage guard check."""
    try:
        from services.futures_risk_guards import FuturesLeverageGuard
        from core_futures import FuturesPosition, PositionSide, MarginMode

        guard = FuturesLeverageGuard()

        position = FuturesPosition(
            symbol="BTCUSDT",
            qty=Decimal("0.5"),
            entry_price=Decimal("50000"),
            side=PositionSide.LONG,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
        )

        def benchmark_func():
            guard.validate_new_position(
                proposed_position=position,
                current_positions=[],
                account_balance=Decimal("10000"),
            )

        return run_benchmark(
            name="Leverage Guard Check",
            func=benchmark_func,
            iterations=iterations,
            target_us=20.0,
        )
    except ImportError as e:
        print(f"Skipping leverage guard check benchmark: {e}")
        return BenchmarkResult(
            name="Leverage Guard Check",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=20.0,
            passed=False,
        )


def bench_margin_guard_check(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark margin guard check."""
    try:
        from services.futures_risk_guards import FuturesMarginGuard

        guard = FuturesMarginGuard()

        def benchmark_func():
            guard.check_margin_ratio(
                margin_ratio=1.35,
                account_equity=10000.0,
                total_margin_used=7407.0,
                symbol="BTCUSDT",
            )

        return run_benchmark(
            name="Margin Guard Check",
            func=benchmark_func,
            iterations=iterations,
            target_us=20.0,
        )
    except ImportError as e:
        print(f"Skipping margin guard check benchmark: {e}")
        return BenchmarkResult(
            name="Margin Guard Check",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=20.0,
            passed=False,
        )


def bench_unified_risk_guard_check(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark unified futures risk guard check."""
    try:
        from services.unified_futures_risk import UnifiedFuturesRiskGuard

        guard = UnifiedFuturesRiskGuard()

        def benchmark_func():
            guard.check_trade(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.1,
                leverage=10,
                account_equity=Decimal("10000"),
                mark_price=Decimal("50000"),
            )

        return run_benchmark(
            name="Unified Risk Guard Check",
            func=benchmark_func,
            iterations=iterations,
            target_us=50.0,
        )
    except ImportError as e:
        print(f"Skipping unified risk guard check benchmark: {e}")
        return BenchmarkResult(
            name="Unified Risk Guard Check",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=50.0,
            passed=False,
        )


# =============================================================================
# Circuit Breaker Benchmarks
# =============================================================================


def bench_circuit_breaker_check(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark circuit breaker check."""
    try:
        from impl_circuit_breaker import CMECircuitBreaker

        cb = CMECircuitBreaker(symbol="ES", reference_price=Decimal("4500"))

        def benchmark_func():
            cb.check_circuit_breaker(
                current_price=Decimal("4300"),
                timestamp_ms=int(time.time() * 1000),
                is_rth=True,
            )

        return run_benchmark(
            name="Circuit Breaker Check",
            func=benchmark_func,
            iterations=iterations,
            target_us=20.0,
        )
    except ImportError as e:
        print(f"Skipping circuit breaker check benchmark: {e}")
        return BenchmarkResult(
            name="Circuit Breaker Check",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=20.0,
            passed=False,
        )


# =============================================================================
# Full Execution Flow Benchmarks
# =============================================================================


def bench_full_l2_execution_crypto(iterations: int = 500) -> BenchmarkResult:
    """Benchmark full L2 execution flow for crypto futures."""
    try:
        from execution_providers_futures import create_futures_execution_provider
        from execution_providers import Order, MarketState, BarData

        provider = create_futures_execution_provider()

        order = Order(
            symbol="BTCUSDT",
            side="BUY",
            qty=0.1,
            order_type="MARKET",
        )
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=50000.0,
            ask=50001.0,
            adv=1e9,
        )
        bar = BarData(
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=10000.0,
        )

        def benchmark_func():
            provider.execute(
                order=order,
                market=market,
                bar=bar,
                funding_rate=0.0001,
                open_interest=2e9,
                recent_liquidations=1e7,
            )

        return run_benchmark(
            name="Full L2 Execution (Crypto Futures)",
            func=benchmark_func,
            iterations=iterations,
            target_us=150.0,
        )
    except ImportError as e:
        print(f"Skipping full L2 execution crypto benchmark: {e}")
        return BenchmarkResult(
            name="Full L2 Execution (Crypto Futures)",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=150.0,
            passed=False,
        )


def bench_full_l2_execution_cme(iterations: int = 500) -> BenchmarkResult:
    """Benchmark full L2 execution flow for CME futures."""
    try:
        from execution_providers_cme import create_cme_execution_provider
        from execution_providers import Order, MarketState, BarData

        provider = create_cme_execution_provider()

        order = Order(
            symbol="ES",
            side="BUY",
            qty=5.0,
            order_type="MARKET",
        )
        market = MarketState(
            timestamp=int(time.time() * 1000),
            bid=4500.0,
            ask=4500.25,
            adv=2e9,
        )
        bar = BarData(
            open=4500.0,
            high=4510.0,
            low=4490.0,
            close=4505.0,
            volume=100000.0,
        )

        def benchmark_func():
            provider.execute(
                order=order,
                market=market,
                bar=bar,
            )

        return run_benchmark(
            name="Full L2 Execution (CME Futures)",
            func=benchmark_func,
            iterations=iterations,
            target_us=150.0,
        )
    except ImportError as e:
        print(f"Skipping full L2 execution CME benchmark: {e}")
        return BenchmarkResult(
            name="Full L2 Execution (CME Futures)",
            iterations=0,
            total_time_ms=0,
            mean_time_us=0,
            median_time_us=0,
            min_time_us=0,
            max_time_us=0,
            std_time_us=0,
            p95_time_us=0,
            p99_time_us=0,
            ops_per_sec=0,
            target_us=150.0,
            passed=False,
        )


# =============================================================================
# Main
# =============================================================================


def run_all_benchmarks(iterations: int = 1000) -> BenchmarkSuite:
    """Run all futures simulation benchmarks."""
    suite = BenchmarkSuite(name="Futures Simulation Benchmarks")
    start_time = time.perf_counter()

    benchmarks = [
        # L2 Execution
        (bench_l2_crypto_futures_slippage, iterations),
        (bench_l2_cme_futures_slippage, iterations),
        # L3 LOB
        (bench_l3_crypto_futures_matching, iterations),
        (bench_l3_cme_futures_matching, iterations),
        # Margin Calculation
        (bench_crypto_margin_calculation, iterations),
        (bench_span_margin_calculation, iterations),
        # Funding Rate
        (bench_funding_rate_calculation, iterations),
        # Liquidation
        (bench_liquidation_price_calculation, iterations),
        (bench_liquidation_cascade_simulation, iterations // 2),
        # Risk Guards
        (bench_leverage_guard_check, iterations),
        (bench_margin_guard_check, iterations),
        (bench_unified_risk_guard_check, iterations),
        (bench_circuit_breaker_check, iterations),
        # Full Execution Flow
        (bench_full_l2_execution_crypto, iterations // 2),
        (bench_full_l2_execution_cme, iterations // 2),
    ]

    for bench_func, iters in benchmarks:
        print(f"\nRunning: {bench_func.__name__}...")
        result = bench_func(iterations=iters)
        suite.add_result(result)
        print_result(result)

    suite.total_time_sec = time.perf_counter() - start_time

    return suite


def run_component_benchmark(component: str, iterations: int = 1000) -> BenchmarkSuite:
    """Run benchmarks for a specific component."""
    suite = BenchmarkSuite(name=f"Futures {component} Benchmarks")
    start_time = time.perf_counter()

    component_map = {
        "l2_execution": [bench_l2_crypto_futures_slippage, bench_l2_cme_futures_slippage],
        "l3_lob": [bench_l3_crypto_futures_matching, bench_l3_cme_futures_matching],
        "margin": [bench_crypto_margin_calculation, bench_span_margin_calculation],
        "funding": [bench_funding_rate_calculation],
        "liquidation": [bench_liquidation_price_calculation, bench_liquidation_cascade_simulation],
        "risk_guards": [
            bench_leverage_guard_check,
            bench_margin_guard_check,
            bench_unified_risk_guard_check,
            bench_circuit_breaker_check,
        ],
        "full_execution": [bench_full_l2_execution_crypto, bench_full_l2_execution_cme],
    }

    benchmarks = component_map.get(component, [])
    if not benchmarks:
        print(f"Unknown component: {component}")
        print(f"Available components: {list(component_map.keys())}")
        return suite

    for bench_func in benchmarks:
        print(f"\nRunning: {bench_func.__name__}...")
        result = bench_func(iterations=iterations)
        suite.add_result(result)
        print_result(result)

    suite.total_time_sec = time.perf_counter() - start_time

    return suite


def print_summary(suite: BenchmarkSuite):
    """Print benchmark suite summary."""
    summary = suite.summary()

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Suite: {summary['suite']}")
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total Time: {summary['total_time_sec']:.2f} sec")
    print("=" * 60)

    # Print failed benchmarks
    failed = [r for r in suite.results if not r.passed]
    if failed:
        print("\nFailed Benchmarks:")
        for r in failed:
            print(f"  - {r.name}: P95={r.p95_time_us:.2f}μs (target: {r.target_us}μs)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Futures Simulation Benchmarks")
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Run benchmarks for a specific component (l2_execution, l3_lob, margin, funding, liquidation, risk_guards, full_execution)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FUTURES SIMULATION BENCHMARKS")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")

    if args.component:
        suite = run_component_benchmark(args.component, args.iterations)
    else:
        suite = run_all_benchmarks(args.iterations)

    print_summary(suite)

    # Exit with non-zero if any benchmark failed
    failed_count = sum(1 for r in suite.results if not r.passed)
    if failed_count > 0:
        print(f"\n{failed_count} benchmark(s) failed to meet targets.")
        sys.exit(1)


if __name__ == "__main__":
    main()
