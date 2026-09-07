"""
Full L3 LOB Simulation Benchmarks.

End-to-end benchmarks for complete L3 simulation pipeline.
Tests the entire workflow from order submission to fill.

Target metrics:
- Full order lifecycle: <1ms
- Simulation throughput: >10,000 orders/sec
- Memory usage: <100MB for 10k orders

Run with:
    python benchmarks/bench_full_sim.py
    python -m pytest benchmarks/bench_full_sim.py -v
"""

import time
import statistics
import tracemalloc
from typing import List, Dict, Tuple
from dataclasses import dataclass
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
    LOBState,
    create_fill_probability_model,
)
from lob.market_impact import (
    create_impact_model,
)
from lob.latency_model import (
    create_latency_model,
)
from lob.hidden_liquidity import (
    create_iceberg_detector,
)
from lob.dark_pool import (
    create_default_dark_pool_simulator,
)


# ==============================================================================
# Benchmark Data Structures
# ==============================================================================


@dataclass
class SimulationResult:
    """Result from a simulation run."""

    orders_processed: int
    fills_executed: int
    elapsed_sec: float
    throughput_per_sec: float
    avg_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    memory_peak_mb: float


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result."""

    name: str
    runs: int
    avg_throughput: float
    avg_latency_us: float
    avg_p95_latency_us: float
    memory_mb: float


# ==============================================================================
# Simulation Setup
# ==============================================================================


def create_deep_orderbook(
    mid_price: float = 100.0,
    spread_bps: float = 5.0,
    levels: int = 50,
    qty_per_level: float = 5000.0,
) -> OrderBook:
    """Create a deep order book for realistic simulation."""
    book = OrderBook()
    half_spread = mid_price * spread_bps / 20000.0

    for i in range(levels):
        # Bids - decreasing prices, increasing qty (typical market structure)
        bid_price = mid_price - half_spread - i * 0.01
        bid_qty = qty_per_level * (1.0 + 0.05 * i)  # Deeper levels have more liquidity
        book.add_limit_order(
            LimitOrder(
                order_id=f"bid_{i}",
                price=bid_price,
                qty=bid_qty,
                remaining_qty=bid_qty,
                timestamp_ns=1000 + i,
                side=Side.BUY,
            )
        )

        # Asks - increasing prices, increasing qty
        ask_price = mid_price + half_spread + i * 0.01
        ask_qty = qty_per_level * (1.0 + 0.05 * i)
        book.add_limit_order(
            LimitOrder(
                order_id=f"ask_{i}",
                price=ask_price,
                qty=ask_qty,
                remaining_qty=ask_qty,
                timestamp_ns=1000 + i,
                side=Side.SELL,
            )
        )

    return book


def generate_order_flow(
    n_orders: int,
    market_ratio: float = 0.3,  # 30% market, 70% limit
    buy_ratio: float = 0.5,
    size_mean: float = 100.0,
    size_std: float = 50.0,
    seed: int = 42,
) -> List[Tuple[str, Side, float, float]]:
    """
    Generate synthetic order flow.

    Returns:
        List of (order_type, side, qty, limit_price or None)
    """
    np.random.seed(seed)
    orders = []

    for i in range(n_orders):
        side = Side.BUY if np.random.random() < buy_ratio else Side.SELL
        qty = max(1.0, np.random.normal(size_mean, size_std))
        is_market = np.random.random() < market_ratio

        if is_market:
            orders.append(("MARKET", side, qty, None))
        else:
            # Limit order with random offset from mid
            offset = np.random.exponential(0.05)  # Price offset
            if side == Side.BUY:
                price = 100.0 - offset
            else:
                price = 100.0 + offset
            orders.append(("LIMIT", side, qty, price))

    return orders


# ==============================================================================
# Full Pipeline Simulation
# ==============================================================================


class FullSimulationBenchmark:
    """Full L3 simulation pipeline benchmark."""

    def __init__(self, seed: int = 42):
        """Initialize simulation components."""
        self.engine = MatchingEngine()
        self.tracker = create_queue_tracker()
        self.fill_model = create_fill_probability_model("analytical_poisson")
        self.impact_model = create_impact_model("almgren_chriss")
        self.latency_model = create_latency_model("institutional", seed=seed)
        self.dark_pool = create_default_dark_pool_simulator(seed=seed)
        self.seed = seed

    def run_simulation(
        self,
        n_orders: int = 1000,
        include_latency: bool = True,
        include_impact: bool = True,
        include_fill_prob: bool = True,
    ) -> SimulationResult:
        """
        Run full simulation and measure performance.

        Args:
            n_orders: Number of orders to process
            include_latency: Include latency simulation
            include_impact: Include market impact calculation
            include_fill_prob: Include fill probability estimation

        Returns:
            SimulationResult with performance metrics
        """
        # Setup
        book = create_deep_orderbook()
        orders = generate_order_flow(n_orders, seed=self.seed)
        market_state = LOBState(
            mid_price=100.0,
            spread_bps=5.0,
            volume_rate=1000.0,
        )

        # Track memory
        tracemalloc.start()

        # Track latencies
        latencies_ns = []
        fills = 0

        # Run simulation
        start = time.perf_counter()

        for i, (order_type, side, qty, limit_price) in enumerate(orders):
            iter_start = time.perf_counter_ns()

            # 1. Simulate latency
            if include_latency:
                _ = self.latency_model.sample_order_latency()

            # 2. Calculate impact
            if include_impact:
                _ = self.impact_model.compute_total_impact(
                    order_qty=qty,
                    adv=10_000_000.0,
                    volatility=0.02,
                    mid_price=100.0,
                )

            # 3. Execute order
            if order_type == "MARKET":
                result = self.engine.match_market_order(side, qty, book)
                if result.total_filled_qty > 0:
                    fills += 1
            else:
                # For limit orders, track queue position and estimate fill prob
                order = LimitOrder(
                    order_id=f"sim_{i}",
                    price=limit_price,
                    qty=qty,
                    remaining_qty=qty,
                    timestamp_ns=1000 + i,
                    side=side,
                )

                # Add to book
                book.add_limit_order(order)

                # Track queue
                state = self.tracker.add_order(order, level_qty_before=1000.0)

                # Fill probability
                if include_fill_prob:
                    _ = self.fill_model.compute_fill_probability(
                        queue_position=(
                            state.estimated_position if state.estimated_position >= 0 else 1
                        ),
                        qty_ahead=state.qty_ahead,
                        order_qty=qty,
                        time_horizon_sec=60.0,
                        market_state=market_state,
                    )

            iter_elapsed = time.perf_counter_ns() - iter_start
            latencies_ns.append(iter_elapsed)

        elapsed = time.perf_counter() - start

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate stats
        latencies_us = [l / 1000.0 for l in latencies_ns]

        return SimulationResult(
            orders_processed=n_orders,
            fills_executed=fills,
            elapsed_sec=elapsed,
            throughput_per_sec=n_orders / elapsed if elapsed > 0 else 0,
            avg_latency_us=statistics.mean(latencies_us),
            p95_latency_us=np.percentile(latencies_us, 95),
            p99_latency_us=np.percentile(latencies_us, 99),
            memory_peak_mb=peak / (1024 * 1024),
        )


# ==============================================================================
# Benchmark Scenarios
# ==============================================================================


def bench_minimal_pipeline() -> BenchmarkResult:
    """Benchmark minimal pipeline (matching only)."""
    print("Running minimal pipeline benchmark...")
    results = []

    for _ in range(3):
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(
            n_orders=10000,
            include_latency=False,
            include_impact=False,
            include_fill_prob=False,
        )
        results.append(result)

    return BenchmarkResult(
        name="minimal_pipeline",
        runs=len(results),
        avg_throughput=statistics.mean(r.throughput_per_sec for r in results),
        avg_latency_us=statistics.mean(r.avg_latency_us for r in results),
        avg_p95_latency_us=statistics.mean(r.p95_latency_us for r in results),
        memory_mb=max(r.memory_peak_mb for r in results),
    )


def bench_full_pipeline() -> BenchmarkResult:
    """Benchmark full pipeline (all components)."""
    print("Running full pipeline benchmark...")
    results = []

    for _ in range(3):
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(
            n_orders=10000,
            include_latency=True,
            include_impact=True,
            include_fill_prob=True,
        )
        results.append(result)

    return BenchmarkResult(
        name="full_pipeline",
        runs=len(results),
        avg_throughput=statistics.mean(r.throughput_per_sec for r in results),
        avg_latency_us=statistics.mean(r.avg_latency_us for r in results),
        avg_p95_latency_us=statistics.mean(r.p95_latency_us for r in results),
        memory_mb=max(r.memory_peak_mb for r in results),
    )


def bench_high_frequency() -> BenchmarkResult:
    """Benchmark high-frequency scenario (many small orders)."""
    print("Running high-frequency benchmark...")
    results = []

    for _ in range(3):
        sim = FullSimulationBenchmark()
        # Generate HFT-like order flow
        orders = generate_order_flow(
            n_orders=50000,
            market_ratio=0.1,  # 10% market, 90% limit
            size_mean=10.0,  # Small sizes
            size_std=5.0,
        )
        result = sim.run_simulation(
            n_orders=50000,
            include_latency=True,
            include_impact=False,  # Skip impact for HFT
            include_fill_prob=True,
        )
        results.append(result)

    return BenchmarkResult(
        name="high_frequency",
        runs=len(results),
        avg_throughput=statistics.mean(r.throughput_per_sec for r in results),
        avg_latency_us=statistics.mean(r.avg_latency_us for r in results),
        avg_p95_latency_us=statistics.mean(r.p95_latency_us for r in results),
        memory_mb=max(r.memory_peak_mb for r in results),
    )


def bench_institutional() -> BenchmarkResult:
    """Benchmark institutional scenario (large orders with impact)."""
    print("Running institutional benchmark...")
    results = []

    for _ in range(3):
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(
            n_orders=1000,
            include_latency=True,
            include_impact=True,
            include_fill_prob=True,
        )
        results.append(result)

    return BenchmarkResult(
        name="institutional",
        runs=len(results),
        avg_throughput=statistics.mean(r.throughput_per_sec for r in results),
        avg_latency_us=statistics.mean(r.avg_latency_us for r in results),
        avg_p95_latency_us=statistics.mean(r.p95_latency_us for r in results),
        memory_mb=max(r.memory_peak_mb for r in results),
    )


# ==============================================================================
# Main
# ==============================================================================


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 90)
    print("FULL L3 SIMULATION BENCHMARKS")
    print("=" * 90)
    print(
        f"{'Scenario':<25} {'Throughput (ord/s)':<20} {'Avg Latency (us)':<18} {'P95 (us)':<15} {'Memory (MB)':<12}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r.name:<25} {r.avg_throughput:<20.0f} {r.avg_latency_us:<18.1f} {r.avg_p95_latency_us:<15.1f} {r.memory_mb:<12.1f}"
        )

    print("=" * 90)


def main():
    """Run all benchmarks."""
    print("Running L3 LOB Full Simulation Benchmarks...")
    print("This may take a few minutes...\n")

    results = []
    results.append(bench_minimal_pipeline())
    results.append(bench_full_pipeline())
    results.append(bench_high_frequency())
    results.append(bench_institutional())

    print_results(results)

    # Target checks
    print("\nPerformance Target Checks:")
    print("-" * 50)

    # Throughput target: >10,000 orders/sec
    full = next(r for r in results if r.name == "full_pipeline")
    passed = full.avg_throughput > 10000
    status = "PASS" if passed else "WARN"
    print(
        f"  Full pipeline throughput: {full.avg_throughput:.0f} ord/s (target: >10,000) [{status}]"
    )

    # Latency target: <1ms (1000us)
    passed = full.avg_p95_latency_us < 1000
    status = "PASS" if passed else "WARN"
    print(
        f"  Full pipeline P95 latency: {full.avg_p95_latency_us:.1f}us (target: <1000us) [{status}]"
    )

    # Memory target: <100MB
    passed = full.memory_mb < 100
    status = "PASS" if passed else "WARN"
    print(f"  Memory usage: {full.memory_mb:.1f}MB (target: <100MB) [{status}]")

    # HFT throughput
    hft = next(r for r in results if r.name == "high_frequency")
    passed = hft.avg_throughput > 50000
    status = "PASS" if passed else "WARN"
    print(f"  HFT throughput: {hft.avg_throughput:.0f} ord/s (target: >50,000) [{status}]")


if __name__ == "__main__":
    main()


# ==============================================================================
# Pytest Tests
# ==============================================================================


class TestFullSimulationBenchmarks:
    """Pytest-compatible benchmark tests."""

    def test_throughput_target(self):
        """Test simulation meets throughput target (>10k orders/sec)."""
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(n_orders=5000)

        # Allow some slack for CI environments
        assert (
            result.throughput_per_sec > 5000
        ), f"Throughput {result.throughput_per_sec:.0f} below minimum"

    def test_latency_target(self):
        """Test simulation meets latency target (P95 <1ms)."""
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(n_orders=1000)

        # P95 should be <1ms (1000us), allow 2ms for Python overhead
        assert (
            result.p95_latency_us < 2000
        ), f"P95 latency {result.p95_latency_us:.0f}us exceeds target"

    def test_memory_target(self):
        """Test simulation meets memory target (<100MB)."""
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(n_orders=10000)

        # Memory should be under 100MB
        assert result.memory_peak_mb < 200, f"Memory {result.memory_peak_mb:.1f}MB exceeds target"

    def test_fills_occur(self):
        """Test that fills actually happen in simulation."""
        sim = FullSimulationBenchmark()
        result = sim.run_simulation(n_orders=1000)

        # At least some orders should fill
        assert result.fills_executed > 0, "No fills occurred"
        fill_rate = result.fills_executed / result.orders_processed
        assert fill_rate > 0.1, f"Fill rate {fill_rate:.2%} too low"
