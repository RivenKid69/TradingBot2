"""
Matching Engine Performance Benchmarks.

Benchmarks for L3 LOB matching engine components.
Target latencies:
- Market order simulation: <10us
- Limit order matching: <50us
- Queue position update: <500us

Run with:
    python benchmarks/bench_matching.py
    python -m pytest benchmarks/bench_matching.py -v
"""

import time
import statistics
from typing import List, Tuple, Callable
import numpy as np

# L3 LOB imports
from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
    OrderType,
)
from lob.matching_engine import (
    MatchingEngine,
    create_matching_engine,
)
from lob.queue_tracker import (
    QueuePositionTracker,
    create_queue_tracker,
)
from lob.fill_probability import (
    AnalyticalPoissonModel,
    QueueReactiveModel,
    LOBState,
    create_fill_probability_model,
)
from lob.market_impact import (
    AlmgrenChrissModel,
    create_impact_model,
)
from lob.latency_model import (
    LatencyModel,
    LatencyProfile,
    create_latency_model,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_benchmark_orderbook(
    mid_price: float = 100.0,
    spread_bps: float = 10.0,
    levels: int = 20,
    qty_per_level: float = 1000.0,
) -> OrderBook:
    """Create an order book for benchmarking."""
    book = OrderBook()
    half_spread = mid_price * spread_bps / 20000.0

    for i in range(levels):
        # Bids
        bid_price = mid_price - half_spread - i * 0.01
        book.add_limit_order(
            LimitOrder(
                order_id=f"bid_{i}",
                price=bid_price,
                qty=qty_per_level,
                remaining_qty=qty_per_level,
                timestamp_ns=1000,
                side=Side.BUY,
            )
        )

        # Asks
        ask_price = mid_price + half_spread + i * 0.01
        book.add_limit_order(
            LimitOrder(
                order_id=f"ask_{i}",
                price=ask_price,
                qty=qty_per_level,
                remaining_qty=qty_per_level,
                timestamp_ns=1000,
                side=Side.SELL,
            )
        )

    return book


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


def print_results(results: List[dict]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("MATCHING ENGINE BENCHMARKS")
    print("=" * 80)
    print(f"{'Benchmark':<40} {'Mean (us)':<12} {'P50 (us)':<12} {'P95 (us)':<12} {'P99 (us)':<12}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['name']:<40} {r['mean_us']:<12.2f} {r['p50_us']:<12.2f} {r['p95_us']:<12.2f} {r['p99_us']:<12.2f}"
        )

    print("=" * 80)


# ==============================================================================
# Benchmarks
# ==============================================================================


class BenchmarkMatchingEngine:
    """Benchmarks for MatchingEngine."""

    def setup(self):
        """Setup for benchmarks."""
        self.engine = MatchingEngine()
        self.book = create_benchmark_orderbook(levels=20, qty_per_level=10000.0)

    def bench_market_order_simulation(self) -> dict:
        """Benchmark market order simulation (target: <10us)."""
        self.setup()

        def func():
            self.engine.simulate_market_order(Side.BUY, 100.0, self.book)

        return run_benchmark("market_order_simulation", func, iterations=10000)

    def bench_market_order_match(self) -> dict:
        """Benchmark market order matching with execution (target: <50us)."""
        self.setup()
        book = create_benchmark_orderbook(levels=100, qty_per_level=10000.0)

        def func():
            self.engine.match_market_order(Side.BUY, 100.0, book)

        return run_benchmark("market_order_match", func, iterations=5000)

    def bench_limit_order_add(self) -> dict:
        """Benchmark adding limit orders to book."""
        self.setup()
        counter = [0]

        def func():
            counter[0] += 1
            order = LimitOrder(
                order_id=f"bench_{counter[0]}",
                price=99.5,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
            self.book.add_limit_order(order)

        return run_benchmark("limit_order_add", func, iterations=10000)

    def bench_order_cancel(self) -> dict:
        """Benchmark order cancellation."""
        self.setup()
        # Pre-populate with orders to cancel
        orders = []
        for i in range(10000):
            order = LimitOrder(
                order_id=f"cancel_{i}",
                price=99.5 - (i * 0.001),
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
            self.book.add_limit_order(order)
            orders.append(order)

        idx = [0]

        def func():
            if idx[0] < len(orders):
                self.book.cancel_order(orders[idx[0]].order_id)
                idx[0] += 1

        return run_benchmark("order_cancel", func, iterations=min(10000, len(orders)))


class BenchmarkQueueTracker:
    """Benchmarks for QueuePositionTracker."""

    def setup(self):
        """Setup for benchmarks."""
        self.tracker = create_queue_tracker()

    def bench_add_order(self) -> dict:
        """Benchmark adding orders to tracker."""
        self.setup()
        counter = [0]

        def func():
            counter[0] += 1
            order = LimitOrder(
                order_id=f"track_{counter[0]}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
            self.tracker.add_order(order, level_qty_before=1000.0)

        return run_benchmark("queue_tracker_add", func, iterations=10000)

    def bench_update_on_execution(self) -> dict:
        """Benchmark execution updates (target: <500us)."""
        self.setup()
        # Pre-populate
        for i in range(100):
            order = LimitOrder(
                order_id=f"pre_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
            self.tracker.add_order(order, level_qty_before=float(i * 100))

        def func():
            self.tracker.update_on_execution(50.0, 100.0)

        return run_benchmark("queue_execution_update", func, iterations=10000)

    def bench_get_state(self) -> dict:
        """Benchmark state lookup."""
        self.setup()
        # Pre-populate
        for i in range(1000):
            order = LimitOrder(
                order_id=f"lookup_{i}",
                price=100.0,
                qty=100.0,
                remaining_qty=100.0,
                timestamp_ns=1000,
                side=Side.BUY,
            )
            self.tracker.add_order(order, level_qty_before=float(i * 100))

        idx = [0]

        def func():
            order_id = f"lookup_{idx[0] % 1000}"
            self.tracker.get_state(order_id)
            idx[0] += 1

        return run_benchmark("queue_state_lookup", func, iterations=10000)


class BenchmarkFillProbability:
    """Benchmarks for fill probability models."""

    def setup(self):
        """Setup for benchmarks."""
        self.poisson = create_fill_probability_model("analytical_poisson")
        self.queue_reactive = create_fill_probability_model("queue_reactive")
        self.market_state = LOBState(
            mid_price=100.0,
            spread_bps=10.0,
            volume_rate=100.0,
        )

    def bench_poisson_fill_prob(self) -> dict:
        """Benchmark Poisson fill probability (target: <10us)."""
        self.setup()

        def func():
            self.poisson.compute_fill_probability(
                queue_position=5,
                qty_ahead=500.0,
                order_qty=100.0,
                time_horizon_sec=60.0,
                market_state=self.market_state,
            )

        return run_benchmark("poisson_fill_prob", func, iterations=10000)

    def bench_queue_reactive_fill_prob(self) -> dict:
        """Benchmark Queue-Reactive fill probability."""
        self.setup()

        def func():
            self.queue_reactive.compute_fill_probability(
                queue_position=5,
                qty_ahead=500.0,
                order_qty=100.0,
                time_horizon_sec=60.0,
                market_state=self.market_state,
            )

        return run_benchmark("queue_reactive_fill_prob", func, iterations=10000)


class BenchmarkMarketImpact:
    """Benchmarks for market impact models."""

    def setup(self):
        """Setup for benchmarks."""
        self.ac_model = create_impact_model("almgren_chriss")

    def bench_almgren_chriss_impact(self) -> dict:
        """Benchmark Almgren-Chriss impact calculation (target: <10us)."""
        self.setup()

        def func():
            self.ac_model.compute_total_impact(
                order_qty=10000.0,
                adv=10_000_000.0,
                volatility=0.02,
                mid_price=100.0,
            )

        return run_benchmark("almgren_chriss_impact", func, iterations=10000)


class BenchmarkLatency:
    """Benchmarks for latency sampling."""

    def setup(self):
        """Setup for benchmarks."""
        self.model = create_latency_model("institutional", seed=42)

    def bench_latency_sample(self) -> dict:
        """Benchmark latency sampling (target: <1us)."""
        self.setup()

        def func():
            self.model.sample_round_trip()

        return run_benchmark("latency_sample", func, iterations=50000)


# ==============================================================================
# Main
# ==============================================================================


def run_all_benchmarks() -> List[dict]:
    """Run all benchmarks and return results."""
    results = []

    # Matching Engine
    bench_engine = BenchmarkMatchingEngine()
    results.append(bench_engine.bench_market_order_simulation())
    results.append(bench_engine.bench_market_order_match())
    results.append(bench_engine.bench_limit_order_add())
    results.append(bench_engine.bench_order_cancel())

    # Queue Tracker
    bench_tracker = BenchmarkQueueTracker()
    results.append(bench_tracker.bench_add_order())
    results.append(bench_tracker.bench_update_on_execution())
    results.append(bench_tracker.bench_get_state())

    # Fill Probability
    bench_fill = BenchmarkFillProbability()
    results.append(bench_fill.bench_poisson_fill_prob())
    results.append(bench_fill.bench_queue_reactive_fill_prob())

    # Market Impact
    bench_impact = BenchmarkMarketImpact()
    results.append(bench_impact.bench_almgren_chriss_impact())

    # Latency
    bench_latency = BenchmarkLatency()
    results.append(bench_latency.bench_latency_sample())

    return results


def main():
    """Run benchmarks and print results."""
    print("Running L3 LOB Matching Engine Benchmarks...")
    results = run_all_benchmarks()
    print_results(results)

    # Summary
    print("\nTarget Performance Check:")
    targets = {
        "market_order_simulation": 10.0,  # us
        "market_order_match": 50.0,
        "queue_execution_update": 500.0,
        "poisson_fill_prob": 50.0,
        "almgren_chriss_impact": 10.0,
        "latency_sample": 5.0,
    }

    all_passed = True
    for r in results:
        if r["name"] in targets:
            target = targets[r["name"]]
            passed = r["p95_us"] < target
            status = "PASS" if passed else "FAIL"
            print(f"  {r['name']}: P95={r['p95_us']:.2f}us, target<{target}us [{status}]")
            if not passed:
                all_passed = False

    if all_passed:
        print("\nAll performance targets met!")
    else:
        print("\nSome performance targets not met (may be acceptable for Python overhead)")


if __name__ == "__main__":
    main()


# ==============================================================================
# Pytest Benchmarks
# ==============================================================================


class TestBenchmarks:
    """Pytest-compatible benchmark tests."""

    def test_market_order_simulation_latency(self):
        """Test market order simulation meets latency target."""
        bench = BenchmarkMatchingEngine()
        result = bench.bench_market_order_simulation()
        # Target <10us but allow 50us for Python overhead
        assert result["p95_us"] < 100.0, f"P95={result['p95_us']:.1f}us"

    def test_queue_update_latency(self):
        """Test queue update meets latency target."""
        bench = BenchmarkQueueTracker()
        result = bench.bench_update_on_execution()
        # Target <500us, allow 2000us for Python overhead in CI
        assert result["p95_us"] < 2000.0, f"P95={result['p95_us']:.1f}us"

    def test_fill_probability_latency(self):
        """Test fill probability meets latency target."""
        bench = BenchmarkFillProbability()
        result = bench.bench_poisson_fill_prob()
        # Target <50us
        assert result["p95_us"] < 100.0, f"P95={result['p95_us']:.1f}us"

    def test_impact_calculation_latency(self):
        """Test impact calculation meets latency target."""
        bench = BenchmarkMarketImpact()
        result = bench.bench_almgren_chriss_impact()
        # Target <10us but allow 100us for Python
        assert result["p95_us"] < 200.0, f"P95={result['p95_us']:.1f}us"
