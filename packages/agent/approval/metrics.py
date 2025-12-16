# -*- coding: utf-8 -*-
"""
Approval Metrics - Prometheus metrics for approval workflow.

AGENT ZONE ONLY.

Provides observability for the approval system with graceful fallback
when prometheus_client is not available.
"""

from __future__ import annotations

from typing import Final

# Graceful import with fallback for environments without prometheus_client
try:  # pragma: no cover
    from prometheus_client import Counter, Gauge, Histogram
except ImportError:  # pragma: no cover

    class _DummyCounter:
        """Fallback counter when prometheus_client is not available."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyCounter":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

    class _DummyGauge:
        """Fallback gauge when prometheus_client is not available."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyGauge":
            return self

        def set(self, value: float) -> None:
            pass

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

    class _DummyHistogram:
        """Fallback histogram when prometheus_client is not available."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyHistogram":
            return self

        def observe(self, value: float) -> None:
            pass

    Counter = _DummyCounter  # type: ignore
    Gauge = _DummyGauge  # type: ignore
    Histogram = _DummyHistogram  # type: ignore


# ============================================================================
# Metric Definitions
# ============================================================================

# Namespace for all approval metrics
METRIC_NAMESPACE: Final[str] = "ccea_approval"

# Request counters
APPROVAL_REQUESTS_TOTAL = Counter(
    f"{METRIC_NAMESPACE}_requests_total",
    "Total number of approval requests created",
    ["command_type", "change_class"],
)

APPROVAL_DECISIONS_TOTAL = Counter(
    f"{METRIC_NAMESPACE}_decisions_total",
    "Total number of approval decisions made",
    ["command_type", "decision", "decided_by"],
)

APPROVAL_AUTO_APPROVED_TOTAL = Counter(
    f"{METRIC_NAMESPACE}_auto_approved_total",
    "Total number of auto-approved requests",
    ["command_type", "change_class"],
)

APPROVAL_EXPIRED_TOTAL = Counter(
    f"{METRIC_NAMESPACE}_expired_total",
    "Total number of expired approval requests",
    ["command_type"],
)

# Gauges for current state
APPROVAL_PENDING_COUNT = Gauge(
    f"{METRIC_NAMESPACE}_pending_count",
    "Current number of pending approval requests",
)

# Latency histograms
APPROVAL_DECISION_LATENCY = Histogram(
    f"{METRIC_NAMESPACE}_decision_latency_seconds",
    "Time taken for approval decisions (from request creation to decision)",
    ["command_type", "decision"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
)

APPROVAL_PERSISTENCE_LATENCY = Histogram(
    f"{METRIC_NAMESPACE}_persistence_latency_seconds",
    "Time taken for persistence operations",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)


# ============================================================================
# Helper Functions
# ============================================================================


def record_request_created(command_type: str, change_class: str) -> None:
    """Record that a new approval request was created."""
    APPROVAL_REQUESTS_TOTAL.labels(
        command_type=command_type,
        change_class=change_class,
    ).inc()


def record_decision(
    command_type: str,
    decision: str,
    decided_by: str,
    latency_seconds: float,
) -> None:
    """Record an approval decision."""
    APPROVAL_DECISIONS_TOTAL.labels(
        command_type=command_type,
        decision=decision,
        decided_by=decided_by,
    ).inc()

    APPROVAL_DECISION_LATENCY.labels(
        command_type=command_type,
        decision=decision,
    ).observe(latency_seconds)


def record_auto_approved(command_type: str, change_class: str) -> None:
    """Record an auto-approved request."""
    APPROVAL_AUTO_APPROVED_TOTAL.labels(
        command_type=command_type,
        change_class=change_class,
    ).inc()


def record_expired(command_type: str) -> None:
    """Record an expired approval request."""
    APPROVAL_EXPIRED_TOTAL.labels(command_type=command_type).inc()


def set_pending_count(count: int) -> None:
    """Set the current pending approval count."""
    APPROVAL_PENDING_COUNT.set(count)


def record_persistence_operation(operation: str, latency_seconds: float) -> None:
    """Record persistence operation latency."""
    APPROVAL_PERSISTENCE_LATENCY.labels(operation=operation).observe(latency_seconds)


__all__ = [
    "METRIC_NAMESPACE",
    "APPROVAL_REQUESTS_TOTAL",
    "APPROVAL_DECISIONS_TOTAL",
    "APPROVAL_AUTO_APPROVED_TOTAL",
    "APPROVAL_EXPIRED_TOTAL",
    "APPROVAL_PENDING_COUNT",
    "APPROVAL_DECISION_LATENCY",
    "APPROVAL_PERSISTENCE_LATENCY",
    "record_request_created",
    "record_decision",
    "record_auto_approved",
    "record_expired",
    "set_pending_count",
    "record_persistence_operation",
]
