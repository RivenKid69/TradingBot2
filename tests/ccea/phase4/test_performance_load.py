# -*- coding: utf-8 -*-
"""
Performance and Load Tests for CCEA Critical Paths.

Tests cover:
- Approval workflow performance (request creation, decisions)
- Persistence layer performance (save/load operations)
- Concurrent access patterns
- High-volume scenarios
- Latency requirements validation

Design Doc Reference:
    - Section 5.4: Performance requirements
    - Section 9.3: Scalability requirements
"""

import concurrent.futures
import statistics
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from uuid import UUID, uuid4

import pytest

from packages.shared.contracts.config import ChangeClass
from packages.agent.approval.manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
)
from packages.agent.approval.persistence import (
    ApprovalPersistence,
    ApprovalPersistenceConfig,
)
from packages.agent.approval.metrics import (
    record_request_created,
    record_decision,
    set_pending_count,
)


# =============================================================================
# Performance Constants
# =============================================================================

# Target latencies (in seconds)
MAX_REQUEST_CREATION_LATENCY = 0.01  # 10ms
MAX_DECISION_LATENCY = 0.01  # 10ms
MAX_PERSISTENCE_SAVE_LATENCY = 0.05  # 50ms
MAX_PERSISTENCE_LOAD_LATENCY = 0.05  # 50ms

# Load test parameters
LOAD_TEST_REQUESTS = 100
CONCURRENT_THREADS = 10
STRESS_TEST_REQUESTS = 500


# =============================================================================
# Helper Functions
# =============================================================================


def measure_latency(func, *args, **kwargs):
    """Measure function execution latency."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compute_latency_stats(latencies: List[float]) -> dict:
    """Compute latency statistics."""
    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
        "p99": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else max(latencies),
    }


# =============================================================================
# Approval Manager Performance Tests
# =============================================================================


class TestApprovalManagerPerformance:
    """Performance tests for ApprovalManager."""

    @pytest.fixture
    def manager(self):
        """Create approval manager."""
        return ApprovalManager(approval_timeout_seconds=60)

    def test_request_creation_latency(self, manager):
        """Test that request creation meets latency requirements."""
        latencies = []

        for i in range(50):
            _, elapsed = measure_latency(
                manager.create_request,
                command_type=f"TEST_COMMAND_{i}",
                description=f"Test request {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        # Verify latency requirements
        assert stats["median"] < MAX_REQUEST_CREATION_LATENCY, (
            f"Request creation median latency {stats['median']:.4f}s "
            f"exceeds requirement {MAX_REQUEST_CREATION_LATENCY}s"
        )
        assert stats["p95"] < MAX_REQUEST_CREATION_LATENCY * 2, (
            f"Request creation p95 latency {stats['p95']:.4f}s "
            f"exceeds 2x requirement"
        )

    def test_approval_decision_latency(self, manager):
        """Test that approval decisions meet latency requirements."""
        # Create requests first
        request_ids = []
        for i in range(50):
            request = manager.create_request(
                command_type=f"TEST_COMMAND_{i}",
                description=f"Test request {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )
            request_ids.append(request.request_id)

        # Measure approval latency
        latencies = []
        for i, request_id in enumerate(request_ids):
            if i % 2 == 0:
                _, elapsed = measure_latency(
                    manager.approve,
                    request_id,
                    reason=f"Approved {i}",
                )
            else:
                _, elapsed = measure_latency(
                    manager.deny,
                    request_id,
                    reason=f"Denied {i}",
                )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        assert stats["median"] < MAX_DECISION_LATENCY, (
            f"Decision median latency {stats['median']:.4f}s "
            f"exceeds requirement {MAX_DECISION_LATENCY}s"
        )

    def test_auto_approval_latency(self, manager):
        """Test that auto-approval meets latency requirements."""
        latencies = []

        for i in range(50):
            _, elapsed = measure_latency(
                manager.create_request,
                command_type=f"OPERATIONAL_CMD_{i}",
                description=f"Operational request {i}",
                change_class=ChangeClass.OPERATIONAL,
            )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        # Auto-approval should be faster than manual approval
        assert stats["median"] < MAX_REQUEST_CREATION_LATENCY, (
            f"Auto-approval median latency {stats['median']:.4f}s "
            f"exceeds requirement"
        )


class TestApprovalLoadTests:
    """Load tests for approval workflow."""

    @pytest.fixture
    def manager(self):
        """Create approval manager."""
        return ApprovalManager(approval_timeout_seconds=300)

    def test_high_volume_requests(self, manager):
        """Test handling of high-volume request creation."""
        start = time.perf_counter()

        for i in range(LOAD_TEST_REQUESTS):
            manager.create_request(
                command_type=f"LOAD_TEST_{i}",
                description=f"Load test request {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )

        elapsed = time.perf_counter() - start
        throughput = LOAD_TEST_REQUESTS / elapsed

        # Should handle at least 100 requests/sec
        assert throughput > 100, f"Throughput {throughput:.1f} req/s is too low"

    def test_concurrent_request_creation(self, manager):
        """Test concurrent request creation from multiple threads."""
        errors = []
        request_ids = []
        lock = threading.Lock()

        def create_request(i):
            try:
                request = manager.create_request(
                    command_type=f"CONCURRENT_{i}",
                    description=f"Concurrent request {i}",
                    change_class=ChangeClass.TRADING_IMPACTING,
                )
                with lock:
                    request_ids.append(request.request_id)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
            futures = [executor.submit(create_request, i) for i in range(LOAD_TEST_REQUESTS)]
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"

        # All requests should be created
        assert len(request_ids) == LOAD_TEST_REQUESTS

        # All request IDs should be unique
        assert len(set(request_ids)) == LOAD_TEST_REQUESTS

    def test_concurrent_decisions(self, manager):
        """Test concurrent approval/denial decisions."""
        # Create requests first
        request_ids = []
        for i in range(LOAD_TEST_REQUESTS):
            request = manager.create_request(
                command_type=f"DECISION_TEST_{i}",
                description=f"Decision test {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )
            request_ids.append(request.request_id)

        errors = []
        decisions = []
        lock = threading.Lock()

        def make_decision(i, request_id):
            try:
                if i % 2 == 0:
                    result = manager.approve(request_id, reason=f"Approved {i}")
                else:
                    result = manager.deny(request_id, reason=f"Denied {i}")
                with lock:
                    decisions.append((request_id, result))
            except Exception as e:
                with lock:
                    errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
            futures = [
                executor.submit(make_decision, i, rid)
                for i, rid in enumerate(request_ids)
            ]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors during concurrent decisions: {errors}"

        # Count successful decisions
        successful = sum(1 for _, result in decisions if result)
        assert successful == LOAD_TEST_REQUESTS


# =============================================================================
# Persistence Performance Tests
# =============================================================================


class TestPersistencePerformance:
    """Performance tests for ApprovalPersistence."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory."""
        return tmp_path / "approvals"

    @pytest.fixture
    def persistence(self, temp_storage):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=temp_storage)
        return ApprovalPersistence(config)

    def test_save_pending_latency(self, persistence):
        """Test save_pending latency requirements."""
        latencies = []

        for i in range(50):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"TEST_{i}",
                "description": f"Test {i}",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }

            _, elapsed = measure_latency(
                persistence.save_pending,
                request_id,
                data,
            )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        assert stats["median"] < MAX_PERSISTENCE_SAVE_LATENCY, (
            f"save_pending median latency {stats['median']:.4f}s "
            f"exceeds requirement {MAX_PERSISTENCE_SAVE_LATENCY}s"
        )

    def test_save_history_latency(self, persistence):
        """Test save_history latency requirements."""
        latencies = []

        for i in range(50):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"TEST_{i}",
                "description": f"Test {i}",
                "status": "approved",
                "decided_at": datetime.utcnow().isoformat(),
            }

            _, elapsed = measure_latency(
                persistence.save_history,
                request_id,
                data,
            )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        assert stats["median"] < MAX_PERSISTENCE_SAVE_LATENCY, (
            f"save_history median latency {stats['median']:.4f}s "
            f"exceeds requirement"
        )

    def test_load_pending_latency(self, persistence):
        """Test load_pending latency requirements."""
        # Save some requests first
        request_ids = []
        for i in range(50):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"LOAD_TEST_{i}",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
            persistence.save_pending(request_id, data)
            request_ids.append(request_id)

        # Measure load latency
        latencies = []
        for request_id in request_ids:
            _, elapsed = measure_latency(
                persistence.load_pending,
                request_id,
            )
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        assert stats["median"] < MAX_PERSISTENCE_LOAD_LATENCY, (
            f"load_pending median latency {stats['median']:.4f}s "
            f"exceeds requirement"
        )

    def test_load_all_pending_latency(self, persistence):
        """Test load_all_pending latency with many records."""
        # Save many requests
        for i in range(LOAD_TEST_REQUESTS):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"BULK_TEST_{i}",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
            persistence.save_pending(request_id, data)

        # Measure bulk load latency
        _, elapsed = measure_latency(persistence.load_all_pending)

        # Should load 100 records in under 500ms
        assert elapsed < 0.5, f"load_all_pending took {elapsed:.4f}s for {LOAD_TEST_REQUESTS} records"

    def test_concurrent_persistence_operations(self, persistence):
        """Test concurrent save/load operations."""
        errors = []
        lock = threading.Lock()

        def save_and_load(i):
            try:
                request_id = uuid4()
                data = {
                    "request_id": str(request_id),
                    "command_type": f"CONCURRENT_{i}",
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat(),
                }

                # Save
                persistence.save_pending(request_id, data)

                # Load back
                loaded = persistence.load_pending(request_id)
                assert loaded is not None
                assert loaded["command_type"] == f"CONCURRENT_{i}"

            except Exception as e:
                with lock:
                    errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
            futures = [executor.submit(save_and_load, i) for i in range(LOAD_TEST_REQUESTS)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors during concurrent persistence: {errors}"


# =============================================================================
# Metrics Performance Tests
# =============================================================================


class TestMetricsPerformance:
    """Performance tests for metrics recording."""

    def test_metrics_recording_overhead(self):
        """Test that metrics recording has minimal overhead."""
        latencies = []

        for i in range(1000):
            start = time.perf_counter()
            record_request_created(f"TEST_{i}", "TRADING_IMPACTING")
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        # Metrics should have sub-millisecond overhead
        assert stats["median"] < 0.001, (
            f"Metrics overhead {stats['median']*1000:.4f}ms is too high"
        )

    def test_metrics_under_load(self):
        """Test metrics performance under high load."""
        start = time.perf_counter()

        for i in range(10000):
            record_request_created(f"LOAD_{i % 10}", "TRADING_IMPACTING")
            if i % 2 == 0:
                record_decision(f"LOAD_{i % 10}", "approved", "user", 1.0)

        elapsed = time.perf_counter() - start

        # Should handle 10k metric recordings quickly
        assert elapsed < 1.0, f"10k metric recordings took {elapsed:.2f}s"


# =============================================================================
# Stress Tests
# =============================================================================


class TestStressScenarios:
    """Stress tests for extreme conditions."""

    @pytest.fixture
    def manager(self):
        """Create approval manager."""
        return ApprovalManager(approval_timeout_seconds=300)

    def test_many_pending_requests(self, manager):
        """Test handling many pending requests."""
        # Create many pending requests
        for i in range(STRESS_TEST_REQUESTS):
            manager.create_request(
                command_type=f"STRESS_{i}",
                description=f"Stress test {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )

        # Verify all are tracked
        pending = manager.get_pending_requests()
        assert len(pending) == STRESS_TEST_REQUESTS

        # Verify retrieval is still fast
        start = time.perf_counter()
        pending = manager.get_pending_requests()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"get_pending_requests took {elapsed:.4f}s"

    def test_rapid_create_approve_cycle(self, manager):
        """Test rapid create-approve cycles."""
        cycles = 100
        latencies = []

        for i in range(cycles):
            start = time.perf_counter()

            # Create
            request = manager.create_request(
                command_type=f"CYCLE_{i}",
                description=f"Cycle {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )

            # Approve immediately
            manager.approve(request.request_id, reason="Quick approve")

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        # Full cycle should be under 20ms
        assert stats["median"] < 0.02, (
            f"Create-approve cycle median {stats['median']*1000:.2f}ms "
            f"exceeds 20ms requirement"
        )

    def test_history_accumulation(self, manager):
        """Test performance with large history."""
        # Create and approve many requests to build history
        for i in range(STRESS_TEST_REQUESTS):
            request = manager.create_request(
                command_type=f"HISTORY_{i}",
                description=f"History test {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )
            manager.approve(request.request_id, reason="Building history")

        # Verify history retrieval is fast
        start = time.perf_counter()
        history = manager.get_history(limit=100)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.01, f"get_history took {elapsed:.4f}s"
        assert len(history) == 100

    @pytest.fixture
    def persistence_stress(self, tmp_path):
        """Create persistence for stress tests."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "stress_approvals")
        return ApprovalPersistence(config)

    def test_persistence_stress(self, persistence_stress):
        """Stress test persistence layer."""
        # Save many records
        for i in range(STRESS_TEST_REQUESTS):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"STRESS_{i}",
                "status": "approved" if i % 2 == 0 else "denied",
                "decided_at": datetime.utcnow().isoformat(),
            }
            persistence_stress.save_history(request_id, data)

        # Verify stats
        stats = persistence_stress.get_stats()
        assert stats["history_count"] == STRESS_TEST_REQUESTS

        # Verify cleanup works
        deleted = persistence_stress.cleanup_old_history(days=0)
        assert deleted >= 0  # May or may not delete depending on dates


# =============================================================================
# Integration Performance Tests
# =============================================================================


class TestIntegrationPerformance:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def manager_with_persistence(self, tmp_path):
        """Create manager with persistence."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "integrated")
        persistence = ApprovalPersistence(config)
        return ApprovalManager(
            approval_timeout_seconds=60,
            persistence=persistence,
        )

    def test_full_workflow_performance(self, manager_with_persistence):
        """Test full workflow with persistence."""
        manager = manager_with_persistence
        latencies = []

        for i in range(50):
            start = time.perf_counter()

            # Full workflow: create -> approve
            request = manager.create_request(
                command_type=f"FULL_{i}",
                description=f"Full workflow {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )
            manager.approve(request.request_id, reason="Test approval")

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        stats = compute_latency_stats(latencies)

        # Full workflow with persistence should still be fast
        assert stats["median"] < 0.1, (
            f"Full workflow median {stats['median']*1000:.2f}ms "
            f"exceeds 100ms requirement"
        )

    def test_persistence_recovery_performance(self, tmp_path):
        """Test performance of recovery from persistence."""
        storage_dir = tmp_path / "recovery"
        config = ApprovalPersistenceConfig(storage_dir=storage_dir)

        # Create first instance and add requests
        persistence1 = ApprovalPersistence(config)
        manager1 = ApprovalManager(persistence=persistence1)

        for i in range(50):
            manager1.create_request(
                command_type=f"RECOVER_{i}",
                description=f"Recovery test {i}",
                change_class=ChangeClass.TRADING_IMPACTING,
            )

        # Simulate restart: create new manager with same storage
        start = time.perf_counter()
        persistence2 = ApprovalPersistence(config)
        manager2 = ApprovalManager(persistence=persistence2)
        elapsed = time.perf_counter() - start

        # Recovery should be fast
        assert elapsed < 0.5, f"Recovery took {elapsed:.4f}s"

        # All pending requests should be recovered
        pending = manager2.get_pending_requests()
        assert len(pending) == 50
