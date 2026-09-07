#!/usr/bin/env python3
"""
Memory Architecture Benchmarks for Options LOB (Phase 0.5).

This module benchmarks the memory-efficient options LOB architecture:
- LazyMultiSeriesLOBManager: lazy instantiation, LRU eviction, disk persistence
- RingBufferOrderBook: fixed-depth order book with aggregation
- EventDrivenLOBCoordinator: O(N log N) event propagation

Target metrics from OPTIONS_INTEGRATION_PLAN.md Phase 0.5:
- Peak memory for SPY full chain: < 4 GB
- LOB access latency: < 1 ms
- Event propagation: < 100 μs per tick

Usage:
    python benchmarks/bench_options_memory.py
    python benchmarks/bench_options_memory.py --full  # Full SPY chain simulation

References:
- OPTIONS_INTEGRATION_PLAN.md Phase 0.5
- SPY has ~960 option series (480 strikes × 2 types), each LOB ~500KB-50MB
- Naive: 960 × 50MB = 48GB → Lazy: 50 × 50MB = 2.5GB
"""

import argparse
import gc
import os
import sys
import tempfile
import threading
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lob.lazy_multi_series import (
    EvictionPolicy,
    LazyMultiSeriesLOBManager,
    create_lazy_lob_manager,
    create_options_lob_manager,
)
from lob.ring_buffer_orderbook import (
    RingBufferOrderBook,
    create_ring_buffer_book,
    create_options_book,
)
from lob.event_coordinator import (
    EventDrivenLOBCoordinator,
    OptionsEvent,
    OptionsEventType,
    PropagationScope,
    create_event_coordinator,
    create_options_coordinator,
)
from lob.data_structures import Side


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    peak_memory_mb: float
    avg_latency_us: float
    p99_latency_us: float
    operations: int
    duration_sec: float
    passed: bool
    target: str
    notes: str = ""


class OptionsMemoryBenchmark:
    """Benchmark suite for options memory architecture."""

    def __init__(self, temp_dir: str, verbose: bool = True):
        self.temp_dir = temp_dir
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def run_all(self, full_chain: bool = False) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.log("=" * 60)
        self.log("Options Memory Architecture Benchmarks (Phase 0.5)")
        self.log("=" * 60)

        # Memory benchmarks
        self.bench_lazy_lob_memory()
        self.bench_ring_buffer_memory()
        self.bench_coordinator_memory()

        # Latency benchmarks
        self.bench_lob_access_latency()
        self.bench_event_propagation_latency()
        self.bench_disk_persistence_latency()

        # Scalability benchmarks
        self.bench_eviction_throughput()
        self.bench_concurrent_access()

        if full_chain:
            self.bench_full_spy_chain()

        self.print_summary()

        return self.results

    def bench_lazy_lob_memory(self):
        """Benchmark: Memory usage of LazyMultiSeriesLOBManager."""
        self.log("\n--- Benchmark: Lazy LOB Manager Memory ---")

        gc.collect()
        tracemalloc.start()

        manager = create_lazy_lob_manager(
            max_active_lobs=50,
            persist_dir=self.temp_dir,
            eviction_policy=EvictionPolicy.LRU,
        )

        # Create 50 LOBs with 1000 orders each
        for i in range(50):
            key = f"SPY_241220_C_{400 + i * 2}"
            lob = manager.get_or_create(key)
            for j in range(1000):
                lob.add_order(
                    side=Side.BUY if j % 2 == 0 else Side.SELL,
                    price=Decimal(str(5.0 + (j % 100) * 0.05)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        target_mb = 4000  # 4GB target

        result = BenchmarkResult(
            name="Lazy LOB Manager Memory",
            peak_memory_mb=peak_mb,
            avg_latency_us=0,
            p99_latency_us=0,
            operations=50,
            duration_sec=0,
            passed=peak_mb < target_mb,
            target=f"< {target_mb} MB",
            notes=f"50 LOBs × 1000 orders each",
        )

        self.results.append(result)
        self.log(f"Peak memory: {peak_mb:.2f} MB (target: < {target_mb} MB)")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_ring_buffer_memory(self):
        """Benchmark: Memory usage of RingBufferOrderBook."""
        self.log("\n--- Benchmark: Ring Buffer Order Book Memory ---")

        gc.collect()
        tracemalloc.start()

        books = []
        for i in range(100):
            book = create_ring_buffer_book(max_depth=20, tick_size=Decimal("0.01"))
            # Add 10000 orders (simulating high activity)
            for j in range(10000):
                book.add_order(
                    side=Side.BUY if j % 2 == 0 else Side.SELL,
                    price=Decimal(str(100.0 + (j % 50) * 0.01)),
                    qty=Decimal("10"),
                    order_id=f"order_{i}_{j}",
                )
            books.append(book)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        # With depth=20, memory should be constant regardless of order volume
        target_mb = 50  # Very conservative target

        result = BenchmarkResult(
            name="Ring Buffer Memory (100 books)",
            peak_memory_mb=peak_mb,
            avg_latency_us=0,
            p99_latency_us=0,
            operations=100 * 10000,
            duration_sec=0,
            passed=peak_mb < target_mb,
            target=f"< {target_mb} MB",
            notes=f"100 books × 10000 orders, depth=20",
        )

        self.results.append(result)
        self.log(f"Peak memory: {peak_mb:.2f} MB (target: < {target_mb} MB)")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_coordinator_memory(self):
        """Benchmark: Memory usage of EventDrivenLOBCoordinator."""
        self.log("\n--- Benchmark: Event Coordinator Memory ---")

        gc.collect()
        tracemalloc.start()

        coordinator = create_event_coordinator(
            underlying="SPY",
            bucket_width=Decimal("5.0"),
        )

        # Register 1000 series
        for strike in range(300, 600):
            for opt_type in ["C", "P"]:
                key = f"SPY_241220_{opt_type}_{strike}"
                coordinator.register_series(key, Decimal(str(strike)))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        target_mb = 100  # Very conservative

        result = BenchmarkResult(
            name="Event Coordinator Memory",
            peak_memory_mb=peak_mb,
            avg_latency_us=0,
            p99_latency_us=0,
            operations=600,
            duration_sec=0,
            passed=peak_mb < target_mb,
            target=f"< {target_mb} MB",
            notes=f"600 registered series",
        )

        self.results.append(result)
        self.log(f"Peak memory: {peak_mb:.2f} MB (target: < {target_mb} MB)")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_lob_access_latency(self):
        """Benchmark: LOB access latency."""
        self.log("\n--- Benchmark: LOB Access Latency ---")

        manager = create_lazy_lob_manager(
            max_active_lobs=100,
            persist_dir=self.temp_dir,
        )

        # Create some LOBs first
        keys = [f"SPY_241220_C_{400 + i * 2}" for i in range(50)]
        for key in keys:
            manager.get_or_create(key)

        # Measure access latency
        latencies = []
        iterations = 10000

        for i in range(iterations):
            key = keys[i % len(keys)]
            start = time.perf_counter_ns()
            manager.get_or_create(key)
            latencies.append(time.perf_counter_ns() - start)

        avg_latency_ns = sum(latencies) / len(latencies)
        avg_latency_us = avg_latency_ns / 1000
        sorted_latencies = sorted(latencies)
        p99_latency_us = sorted_latencies[int(len(latencies) * 0.99)] / 1000

        target_us = 1000  # 1ms = 1000us

        result = BenchmarkResult(
            name="LOB Access Latency",
            peak_memory_mb=0,
            avg_latency_us=avg_latency_us,
            p99_latency_us=p99_latency_us,
            operations=iterations,
            duration_sec=sum(latencies) / 1e9,
            passed=avg_latency_us < target_us,
            target=f"< {target_us} μs",
            notes=f"Hit ratio: 100% (cached)",
        )

        self.results.append(result)
        self.log(f"Avg latency: {avg_latency_us:.2f} μs (target: < {target_us} μs)")
        self.log(f"P99 latency: {p99_latency_us:.2f} μs")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_event_propagation_latency(self):
        """Benchmark: Event propagation latency."""
        self.log("\n--- Benchmark: Event Propagation Latency ---")

        coordinator = create_event_coordinator(
            underlying="SPY",
            bucket_width=Decimal("5.0"),
            max_propagation_depth=3,
        )

        # Register 200 series (like a medium-sized chain)
        for strike in range(400, 600):
            key = f"SPY_241220_C_{strike}"
            coordinator.register_series(key, Decimal(str(strike)))

        # Measure propagation latency
        latencies = []
        iterations = 1000

        for i in range(iterations):
            event = OptionsEvent(
                event_type=OptionsEventType.UNDERLYING_TICK,
                underlying_price=Decimal(str(500.0 + (i % 100) * 0.01)),
                timestamp=datetime.now(),
            )

            start = time.perf_counter_ns()
            coordinator.propagate(event, scope=PropagationScope.ATM_ONLY)
            latencies.append(time.perf_counter_ns() - start)

        avg_latency_ns = sum(latencies) / len(latencies)
        avg_latency_us = avg_latency_ns / 1000
        sorted_latencies = sorted(latencies)
        p99_latency_us = sorted_latencies[int(len(latencies) * 0.99)] / 1000

        target_us = 100  # 100μs target

        result = BenchmarkResult(
            name="Event Propagation Latency",
            peak_memory_mb=0,
            avg_latency_us=avg_latency_us,
            p99_latency_us=p99_latency_us,
            operations=iterations,
            duration_sec=sum(latencies) / 1e9,
            passed=avg_latency_us < target_us,
            target=f"< {target_us} μs",
            notes=f"200 series, ATM_ONLY scope",
        )

        self.results.append(result)
        self.log(f"Avg latency: {avg_latency_us:.2f} μs (target: < {target_us} μs)")
        self.log(f"P99 latency: {p99_latency_us:.2f} μs")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_disk_persistence_latency(self):
        """Benchmark: Disk persistence latency."""
        self.log("\n--- Benchmark: Disk Persistence Latency ---")

        manager = create_lazy_lob_manager(
            max_active_lobs=50,
            persist_dir=self.temp_dir,
            enable_compression=True,
        )

        # Create LOB with realistic content
        key = "SPY_241220_C_500"
        lob = manager.get_or_create(key)
        for i in range(1000):
            lob.add_order(
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                price=Decimal(str(5.0 + (i % 100) * 0.01)),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        # Measure persist latency
        latencies = []
        iterations = 100

        for _ in range(iterations):
            start = time.perf_counter_ns()
            manager.persist_to_disk(key)
            latencies.append(time.perf_counter_ns() - start)

        avg_latency_ns = sum(latencies) / len(latencies)
        avg_latency_ms = avg_latency_ns / 1_000_000
        sorted_latencies = sorted(latencies)
        p99_latency_ms = sorted_latencies[int(len(latencies) * 0.99)] / 1_000_000

        target_ms = 50  # 50ms target for persist

        result = BenchmarkResult(
            name="Disk Persistence Latency",
            peak_memory_mb=0,
            avg_latency_us=avg_latency_ns / 1000,
            p99_latency_us=sorted_latencies[int(len(latencies) * 0.99)] / 1000,
            operations=iterations,
            duration_sec=sum(latencies) / 1e9,
            passed=avg_latency_ms < target_ms,
            target=f"< {target_ms} ms",
            notes=f"1000 orders, gzip compression",
        )

        self.results.append(result)
        self.log(f"Avg latency: {avg_latency_ms:.2f} ms (target: < {target_ms} ms)")
        self.log(f"P99 latency: {p99_latency_ms:.2f} ms")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_eviction_throughput(self):
        """Benchmark: Eviction throughput."""
        self.log("\n--- Benchmark: Eviction Throughput ---")

        manager = create_lazy_lob_manager(
            max_active_lobs=10,
            persist_dir=self.temp_dir,
            persist_on_evict=True,
        )

        start = time.perf_counter()

        # Create 100 LOBs, triggering 90 evictions
        for i in range(100):
            key = f"SPY_241220_C_{400 + i * 2}"
            lob = manager.get_or_create(key)
            lob.add_order(
                side=Side.BUY,
                price=Decimal("5.00"),
                qty=Decimal("100"),
                order_id=f"order_{i}",
            )

        elapsed = time.perf_counter() - start

        stats = manager.get_stats()
        evictions_per_sec = stats.evictions / elapsed if elapsed > 0 else 0

        target_eps = 50  # 50 evictions/sec

        result = BenchmarkResult(
            name="Eviction Throughput",
            peak_memory_mb=0,
            avg_latency_us=0,
            p99_latency_us=0,
            operations=stats.evictions,
            duration_sec=elapsed,
            passed=evictions_per_sec > target_eps,
            target=f"> {target_eps} evictions/sec",
            notes=f"{stats.evictions} evictions in {elapsed:.2f}s",
        )

        self.results.append(result)
        self.log(f"Eviction rate: {evictions_per_sec:.1f}/sec (target: > {target_eps}/sec)")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_concurrent_access(self):
        """Benchmark: Concurrent access performance."""
        self.log("\n--- Benchmark: Concurrent Access ---")

        manager = create_lazy_lob_manager(
            max_active_lobs=50,
            persist_dir=self.temp_dir,
        )

        errors = []
        latencies = []
        lock = threading.Lock()

        def worker(worker_id: int, iterations: int):
            local_latencies = []
            try:
                for i in range(iterations):
                    key = f"SPY_241220_C_{400 + (worker_id * 10 + i % 10) * 2}"
                    start = time.perf_counter_ns()
                    lob = manager.get_or_create(key)
                    lob.add_order(
                        side=Side.BUY,
                        price=Decimal("5.00"),
                        qty=Decimal("10"),
                        order_id=f"order_{worker_id}_{i}",
                    )
                    local_latencies.append(time.perf_counter_ns() - start)
            except Exception as e:
                with lock:
                    errors.append(e)
            with lock:
                latencies.extend(local_latencies)

        num_threads = 8
        iterations_per_thread = 500

        start = time.perf_counter()

        threads = [
            threading.Thread(target=worker, args=(i, iterations_per_thread))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.perf_counter() - start

        total_ops = num_threads * iterations_per_thread
        ops_per_sec = total_ops / elapsed if elapsed > 0 else 0
        avg_latency_us = (sum(latencies) / len(latencies) / 1000) if latencies else 0

        target_ops = 1000  # 1000 ops/sec

        result = BenchmarkResult(
            name="Concurrent Access",
            peak_memory_mb=0,
            avg_latency_us=avg_latency_us,
            p99_latency_us=0,
            operations=total_ops,
            duration_sec=elapsed,
            passed=len(errors) == 0 and ops_per_sec > target_ops,
            target=f"> {target_ops} ops/sec, 0 errors",
            notes=f"{num_threads} threads, {len(errors)} errors",
        )

        self.results.append(result)
        self.log(f"Throughput: {ops_per_sec:.1f} ops/sec (target: > {target_ops})")
        self.log(f"Errors: {len(errors)}")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def bench_full_spy_chain(self):
        """Benchmark: Full SPY chain simulation (960 series)."""
        self.log("\n--- Benchmark: Full SPY Chain (960 series) ---")

        gc.collect()
        tracemalloc.start()

        manager = create_lazy_lob_manager(
            max_active_lobs=50,  # Only keep 50 active
            persist_dir=self.temp_dir,
            persist_on_evict=True,
            eviction_policy=EvictionPolicy.LRU,
        )

        coordinator = create_event_coordinator(
            underlying="SPY",
            bucket_width=Decimal("5.0"),
        )

        # Generate 960 series keys (480 strikes × 2 types)
        series_keys = []
        for strike in range(300, 540):  # 240 strikes
            for expiry in ["241220", "250117"]:  # 2 expiries
                for opt_type in ["C", "P"]:  # Call and Put
                    key = f"SPY_{expiry}_{opt_type}_{strike}"
                    series_keys.append((key, Decimal(str(strike))))
                    coordinator.register_series(key, Decimal(str(strike)))

        self.log(f"Registered {len(series_keys)} series")

        # Simulate access pattern (ATM more frequent)
        start = time.perf_counter()
        atm_strike = 420  # Assume SPY at 420

        for _ in range(1000):
            # Access ATM and nearby strikes more frequently
            if len(series_keys) > 0:
                # Weight towards ATM
                idx = min(
                    int(abs(atm_strike - 300 + (time.perf_counter() * 100) % 50)),
                    len(series_keys) - 1,
                )
                key, strike = series_keys[idx % len(series_keys)]

                lob = manager.get_or_create(key)
                lob.add_order(
                    side=Side.BUY,
                    price=Decimal("1.00"),
                    qty=Decimal("10"),
                    order_id=f"order_{time.perf_counter_ns()}",
                )

        elapsed = time.perf_counter() - start

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        target_mb = 4000  # 4GB target

        result = BenchmarkResult(
            name="Full SPY Chain Simulation",
            peak_memory_mb=peak_mb,
            avg_latency_us=0,
            p99_latency_us=0,
            operations=1000,
            duration_sec=elapsed,
            passed=peak_mb < target_mb,
            target=f"< {target_mb} MB",
            notes=f"{len(series_keys)} series, 50 active max",
        )

        self.results.append(result)
        self.log(f"Peak memory: {peak_mb:.2f} MB (target: < {target_mb} MB)")
        self.log(f"Active LOBs: {manager.get_active_count()} (max: 50)")
        self.log(f"Evictions: {manager.get_stats().evictions}")
        self.log(f"Result: {'PASS' if result.passed else 'FAIL'}")

    def print_summary(self):
        """Print benchmark summary."""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK SUMMARY")
        self.log("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            self.log(f"\n{status}: {r.name}")
            if r.peak_memory_mb > 0:
                self.log(f"  Memory: {r.peak_memory_mb:.2f} MB")
            if r.avg_latency_us > 0:
                self.log(f"  Avg Latency: {r.avg_latency_us:.2f} μs")
            if r.p99_latency_us > 0:
                self.log(f"  P99 Latency: {r.p99_latency_us:.2f} μs")
            self.log(f"  Target: {r.target}")
            if r.notes:
                self.log(f"  Notes: {r.notes}")

        self.log("\n" + "-" * 60)
        self.log(f"Results: {passed}/{total} passed")

        if passed == total:
            self.log("\n✓ ALL BENCHMARKS PASSED - Phase 0.5 targets met!")
        else:
            self.log(f"\n✗ {total - passed} benchmarks failed")


def main():
    parser = argparse.ArgumentParser(description="Options Memory Architecture Benchmarks")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full SPY chain simulation (slower)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark = OptionsMemoryBenchmark(
            temp_dir=temp_dir,
            verbose=not args.quiet,
        )
        results = benchmark.run_all(full_chain=args.full)

        # Exit with error code if any benchmark failed
        sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
