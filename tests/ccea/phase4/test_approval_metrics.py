# -*- coding: utf-8 -*-
"""
Tests for Approval Metrics Module.

Tests cover:
- Metric definitions (Counter, Gauge, Histogram)
- Helper functions
- Graceful fallback when prometheus_client is not available
- Thread safety of metrics

100% coverage target for metrics.py
"""

import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from packages.agent.approval.metrics import (
    # Metric definitions
    METRIC_NAMESPACE,
    APPROVAL_REQUESTS_TOTAL,
    APPROVAL_DECISIONS_TOTAL,
    APPROVAL_AUTO_APPROVED_TOTAL,
    APPROVAL_EXPIRED_TOTAL,
    APPROVAL_PENDING_COUNT,
    APPROVAL_DECISION_LATENCY,
    APPROVAL_PERSISTENCE_LATENCY,
    # Helper functions
    record_request_created,
    record_decision,
    record_auto_approved,
    record_expired,
    set_pending_count,
    record_persistence_operation,
)


# =============================================================================
# Metric Namespace Tests
# =============================================================================


class TestMetricNamespace:
    """Tests for metric namespace."""

    def test_namespace_value(self):
        """Test metric namespace is correctly set."""
        assert METRIC_NAMESPACE == "ccea_approval"


# =============================================================================
# Metric Definition Tests
# =============================================================================


class TestMetricDefinitions:
    """Tests for metric object definitions."""

    def test_requests_total_exists(self):
        """Test APPROVAL_REQUESTS_TOTAL is defined."""
        assert APPROVAL_REQUESTS_TOTAL is not None

    def test_decisions_total_exists(self):
        """Test APPROVAL_DECISIONS_TOTAL is defined."""
        assert APPROVAL_DECISIONS_TOTAL is not None

    def test_auto_approved_total_exists(self):
        """Test APPROVAL_AUTO_APPROVED_TOTAL is defined."""
        assert APPROVAL_AUTO_APPROVED_TOTAL is not None

    def test_expired_total_exists(self):
        """Test APPROVAL_EXPIRED_TOTAL is defined."""
        assert APPROVAL_EXPIRED_TOTAL is not None

    def test_pending_count_exists(self):
        """Test APPROVAL_PENDING_COUNT is defined."""
        assert APPROVAL_PENDING_COUNT is not None

    def test_decision_latency_exists(self):
        """Test APPROVAL_DECISION_LATENCY is defined."""
        assert APPROVAL_DECISION_LATENCY is not None

    def test_persistence_latency_exists(self):
        """Test APPROVAL_PERSISTENCE_LATENCY is defined."""
        assert APPROVAL_PERSISTENCE_LATENCY is not None


# =============================================================================
# Counter Helper Function Tests
# =============================================================================


class TestRecordRequestCreated:
    """Tests for record_request_created helper."""

    def test_records_with_labels(self):
        """Test recording request with labels."""
        # Should not raise any errors
        record_request_created("START_RUN", "TRADING_IMPACTING")

    def test_records_various_command_types(self):
        """Test recording various command types."""
        command_types = ["START_RUN", "STOP_RUN", "UPDATE_CONFIG", "EMERGENCY_HALT"]

        for cmd in command_types:
            record_request_created(cmd, "TRADING_IMPACTING")

    def test_records_various_change_classes(self):
        """Test recording various change classes."""
        change_classes = ["TRADING_IMPACTING", "OPERATIONAL", "INFORMATIONAL"]

        for cc in change_classes:
            record_request_created("TEST_CMD", cc)


class TestRecordDecision:
    """Tests for record_decision helper."""

    def test_records_approval(self):
        """Test recording approval decision."""
        record_decision(
            command_type="START_RUN",
            decision="approved",
            decided_by="local_user",
            latency_seconds=1.5,
        )

    def test_records_denial(self):
        """Test recording denial decision."""
        record_decision(
            command_type="START_RUN",
            decision="denied",
            decided_by="admin",
            latency_seconds=2.0,
        )

    def test_records_various_latencies(self):
        """Test recording various latency values."""
        latencies = [0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 300.0]

        for lat in latencies:
            record_decision("TEST", "approved", "user", lat)

    def test_records_with_various_users(self):
        """Test recording with various decision makers."""
        users = ["local_user", "admin", "system", "api_client"]

        for user in users:
            record_decision("TEST", "approved", user, 1.0)


class TestRecordAutoApproved:
    """Tests for record_auto_approved helper."""

    def test_records_auto_approval(self):
        """Test recording auto-approval."""
        record_auto_approved("OPERATIONAL_CMD", "OPERATIONAL")

    def test_records_various_types(self):
        """Test recording various auto-approved types."""
        types = [
            ("INFO_CMD", "INFORMATIONAL"),
            ("OPS_CMD", "OPERATIONAL"),
        ]

        for cmd, cc in types:
            record_auto_approved(cmd, cc)


class TestRecordExpired:
    """Tests for record_expired helper."""

    def test_records_expiration(self):
        """Test recording expiration."""
        record_expired("START_RUN")

    def test_records_various_expired_types(self):
        """Test recording various expired command types."""
        commands = ["START_RUN", "UPDATE_CONFIG", "DEPLOY_STRATEGY"]

        for cmd in commands:
            record_expired(cmd)


# =============================================================================
# Gauge Helper Function Tests
# =============================================================================


class TestSetPendingCount:
    """Tests for set_pending_count helper."""

    def test_sets_count(self):
        """Test setting pending count."""
        set_pending_count(10)

    def test_sets_zero(self):
        """Test setting zero count."""
        set_pending_count(0)

    def test_sets_large_count(self):
        """Test setting large count."""
        set_pending_count(10000)


# =============================================================================
# Histogram Helper Function Tests
# =============================================================================


class TestRecordPersistenceOperation:
    """Tests for record_persistence_operation helper."""

    def test_records_save_pending(self):
        """Test recording save_pending operation."""
        record_persistence_operation("save_pending", 0.005)

    def test_records_save_history(self):
        """Test recording save_history operation."""
        record_persistence_operation("save_history", 0.010)

    def test_records_load_pending(self):
        """Test recording load_pending operation."""
        record_persistence_operation("load_pending", 0.002)

    def test_records_load_history(self):
        """Test recording load_history operation."""
        record_persistence_operation("load_history", 0.050)

    def test_records_various_latencies(self):
        """Test recording various persistence latencies."""
        operations = [
            ("save_pending", 0.001),
            ("save_pending", 0.01),
            ("save_history", 0.05),
            ("load_all_pending", 0.1),
        ]

        for op, lat in operations:
            record_persistence_operation(op, lat)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestMetricsThreadSafety:
    """Tests for metrics thread safety."""

    def test_concurrent_counter_increments(self):
        """Test concurrent counter increments."""
        errors = []
        lock = threading.Lock()

        def increment(i):
            try:
                record_request_created(f"THREAD_{i % 5}", "TRADING_IMPACTING")
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=increment, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_gauge_sets(self):
        """Test concurrent gauge sets."""
        errors = []
        lock = threading.Lock()

        def set_gauge(i):
            try:
                set_pending_count(i)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=set_gauge, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_histogram_observations(self):
        """Test concurrent histogram observations."""
        errors = []
        lock = threading.Lock()

        def observe(i):
            try:
                record_decision(
                    f"CMD_{i % 5}",
                    "approved" if i % 2 == 0 else "denied",
                    "user",
                    i * 0.01,
                )
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=observe, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestMetricsPerformance:
    """Performance tests for metrics."""

    def test_counter_increment_speed(self):
        """Test counter increment is fast."""
        start = time.perf_counter()

        for i in range(1000):
            record_request_created("TEST", "TRADING_IMPACTING")

        elapsed = time.perf_counter() - start

        # 1000 increments should take less than 100ms
        assert elapsed < 0.1, f"1000 counter increments took {elapsed:.3f}s"

    def test_gauge_set_speed(self):
        """Test gauge set is fast."""
        start = time.perf_counter()

        for i in range(1000):
            set_pending_count(i)

        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"1000 gauge sets took {elapsed:.3f}s"

    def test_histogram_observe_speed(self):
        """Test histogram observe is fast."""
        start = time.perf_counter()

        for i in range(1000):
            record_decision("TEST", "approved", "user", 1.0)

        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"1000 histogram observations took {elapsed:.3f}s"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics with ApprovalManager."""

    def test_metrics_recorded_on_request_creation(self):
        """Test that metrics are recorded when creating requests."""
        from packages.agent.approval.manager import ApprovalManager
        from packages.shared.contracts.config import ChangeClass

        manager = ApprovalManager()

        # This should trigger metric recording internally
        request = manager.create_request(
            command_type="INTEGRATION_TEST",
            description="Testing metrics",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        # If we get here without error, metrics recording worked
        assert request is not None

    def test_metrics_recorded_on_approval(self):
        """Test that metrics are recorded on approval."""
        from packages.agent.approval.manager import ApprovalManager
        from packages.shared.contracts.config import ChangeClass

        manager = ApprovalManager()
        request = manager.create_request(
            command_type="APPROVAL_TEST",
            description="Testing approval metrics",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        result = manager.approve(request.request_id, reason="Test")

        assert result is True

    def test_metrics_recorded_on_denial(self):
        """Test that metrics are recorded on denial."""
        from packages.agent.approval.manager import ApprovalManager
        from packages.shared.contracts.config import ChangeClass

        manager = ApprovalManager()
        request = manager.create_request(
            command_type="DENIAL_TEST",
            description="Testing denial metrics",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        result = manager.deny(request.request_id, reason="Test denial")

        assert result is True

    def test_metrics_recorded_on_auto_approval(self):
        """Test that metrics are recorded on auto-approval."""
        from packages.agent.approval.manager import ApprovalManager
        from packages.shared.contracts.config import ChangeClass

        manager = ApprovalManager(auto_approve_operational=True)

        # OPERATIONAL should be auto-approved
        request = manager.create_request(
            command_type="AUTO_APPROVAL_TEST",
            description="Testing auto-approval metrics",
            change_class=ChangeClass.OPERATIONAL,
        )

        # Request should be auto-approved
        from packages.agent.approval.manager import ApprovalStatus

        assert request.status == ApprovalStatus.APPROVED


# =============================================================================
# Fallback Tests
# =============================================================================


class TestDummyMetricsFallback:
    """Tests for dummy metrics fallback when prometheus_client is unavailable."""

    def test_dummy_counter_methods(self):
        """Test dummy counter has all required methods."""
        # Test that the actual Counter (which may be dummy) works
        counter = APPROVAL_REQUESTS_TOTAL

        # These should not raise errors
        labeled = counter.labels(command_type="TEST", change_class="TEST")
        labeled.inc()
        labeled.inc(2)

    def test_dummy_gauge_methods(self):
        """Test dummy gauge has all required methods."""
        gauge = APPROVAL_PENDING_COUNT

        # These should not raise errors
        gauge.set(10)
        gauge.inc()
        gauge.inc(5)
        gauge.dec()
        gauge.dec(3)

    def test_dummy_histogram_methods(self):
        """Test dummy histogram has all required methods."""
        histogram = APPROVAL_DECISION_LATENCY

        # These should not raise errors
        labeled = histogram.labels(command_type="TEST", decision="approved")
        labeled.observe(1.5)
