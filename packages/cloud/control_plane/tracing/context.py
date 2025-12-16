# -*- coding: utf-8 -*-
"""
Trace Context Propagation - CLOUD ZONE ONLY.

W3C Trace Context propagation utilities for distributed tracing.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
    - Cross-service trace correlation

Features:
    - W3C Trace Context extraction/injection
    - Baggage propagation
    - Context correlation with workspace/tenant
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from .tracer import OPENTELEMETRY_AVAILABLE

if OPENTELEMETRY_AVAILABLE:
    from opentelemetry import trace, baggage
    from opentelemetry.context import Context
    from opentelemetry.propagate import extract, inject
    from opentelemetry.trace import SpanContext, TraceFlags

logger = logging.getLogger(__name__)


# ============================================================================
# Trace Context Data Class
# ============================================================================


@dataclass
class TraceContext:
    """
    Encapsulates trace context for propagation.

    Contains trace ID, span ID, and custom baggage items.
    """
    trace_id: str = ""
    span_id: str = ""
    trace_flags: int = 0
    trace_state: str = ""

    # Custom context
    workspace_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None

    # Baggage items
    baggage: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """
        Extract trace context from HTTP headers.

        Args:
            headers: HTTP headers dictionary

        Returns:
            TraceContext with extracted values
        """
        context = cls()

        if not OPENTELEMETRY_AVAILABLE:
            # Parse W3C traceparent manually if OTel not available
            traceparent = headers.get("traceparent", "")
            if traceparent:
                parts = traceparent.split("-")
                if len(parts) >= 4:
                    context.trace_id = parts[1]
                    context.span_id = parts[2]
                    context.trace_flags = int(parts[3], 16)

            context.trace_state = headers.get("tracestate", "")

            # Extract custom context from headers
            context.workspace_id = headers.get("x-workspace-id")
            context.tenant_id = headers.get("x-tenant-id")
            context.user_id = headers.get("x-user-id")
            context.request_id = headers.get("x-request-id")
            return context

        # Extract using OpenTelemetry
        otel_context = extract(headers)
        span_context = trace.get_current_span(otel_context).get_span_context()

        if span_context and span_context.is_valid:
            context.trace_id = format(span_context.trace_id, "032x")
            context.span_id = format(span_context.span_id, "016x")
            context.trace_flags = span_context.trace_flags

        # Extract baggage
        context.baggage = dict(baggage.get_all(otel_context))

        # Extract custom context from headers
        context.workspace_id = headers.get("x-workspace-id")
        context.tenant_id = headers.get("x-tenant-id")
        context.user_id = headers.get("x-user-id")
        context.request_id = headers.get("x-request-id")

        return context

    def to_headers(self) -> Dict[str, str]:
        """
        Convert trace context to HTTP headers.

        Returns:
            Dictionary of headers for propagation
        """
        headers: Dict[str, str] = {}

        if not OPENTELEMETRY_AVAILABLE:
            # Generate W3C traceparent manually
            if self.trace_id and self.span_id:
                traceparent = f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"
                headers["traceparent"] = traceparent

            if self.trace_state:
                headers["tracestate"] = self.trace_state

        else:
            # Inject using OpenTelemetry
            inject(headers)

        # Add custom context
        if self.workspace_id:
            headers["x-workspace-id"] = self.workspace_id
        if self.tenant_id:
            headers["x-tenant-id"] = self.tenant_id
        if self.user_id:
            headers["x-user-id"] = self.user_id
        if self.request_id:
            headers["x-request-id"] = self.request_id

        return headers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "baggage": self.baggage,
        }


# ============================================================================
# Context Propagation Functions
# ============================================================================


def extract_context(carrier: Dict[str, str]) -> "Context":
    """
    Extract trace context from a carrier (headers).

    Args:
        carrier: Dictionary containing trace context headers

    Returns:
        OpenTelemetry Context or empty context
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    return extract(carrier)


def inject_context(carrier: Dict[str, str], context: Optional["Context"] = None) -> None:
    """
    Inject trace context into a carrier (headers).

    Args:
        carrier: Dictionary to inject headers into
        context: Optional context (uses current if not provided)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    inject(carrier, context=context)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID.

    Returns:
        32-character hex trace ID or None
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context and span_context.is_valid:
            return format(span_context.trace_id, "032x")

    return None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID.

    Returns:
        16-character hex span ID or None
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context and span_context.is_valid:
            return format(span_context.span_id, "016x")

    return None


def get_current_span() -> Any:
    """
    Get the current span.

    Returns:
        Current span or no-op span
    """
    if not OPENTELEMETRY_AVAILABLE:
        from .tracer import _NoOpSpan
        return _NoOpSpan()

    return trace.get_current_span()


def set_span_attribute(key: str, value: Any) -> None:
    """
    Set an attribute on the current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def add_span_event(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def record_exception(
    exception: Exception,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an exception on the current span.

    Args:
        exception: Exception to record
        attributes: Additional attributes
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception, attributes=attributes)


# ============================================================================
# Baggage Operations
# ============================================================================


def set_baggage(key: str, value: str) -> "Context":
    """
    Set a baggage item in the current context.

    Args:
        key: Baggage key
        value: Baggage value

    Returns:
        Updated context
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    return baggage.set_baggage(key, value)


def get_baggage(key: str) -> Optional[str]:
    """
    Get a baggage item from the current context.

    Args:
        key: Baggage key

    Returns:
        Baggage value or None
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None

    return baggage.get_baggage(key)


def get_all_baggage() -> Dict[str, str]:
    """
    Get all baggage items from the current context.

    Returns:
        Dictionary of baggage items
    """
    if not OPENTELEMETRY_AVAILABLE:
        return {}

    return dict(baggage.get_all())


# ============================================================================
# Context Correlation
# ============================================================================


def correlate_with_workspace(workspace_id: str) -> None:
    """
    Correlate current span with a workspace.

    Args:
        workspace_id: Workspace ID to correlate
    """
    set_span_attribute("ccea.workspace_id", workspace_id)
    set_baggage("ccea.workspace_id", workspace_id)


def correlate_with_tenant(tenant_id: str) -> None:
    """
    Correlate current span with a tenant.

    Args:
        tenant_id: Tenant ID to correlate
    """
    set_span_attribute("ccea.tenant_id", tenant_id)
    set_baggage("ccea.tenant_id", tenant_id)


def correlate_with_agent(agent_id: str) -> None:
    """
    Correlate current span with an agent.

    Args:
        agent_id: Agent ID to correlate
    """
    set_span_attribute("ccea.agent_id", agent_id)
    set_baggage("ccea.agent_id", agent_id)


def correlate_with_job(job_id: str) -> None:
    """
    Correlate current span with a job.

    Args:
        job_id: Job ID to correlate
    """
    set_span_attribute("ccea.job_id", job_id)
    set_baggage("ccea.job_id", job_id)


# ============================================================================
# Logging Integration
# ============================================================================


class TraceLoggingFilter(logging.Filter):
    """
    Logging filter that adds trace context to log records.

    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(TraceLoggingFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace context to log record."""
        record.trace_id = get_current_trace_id() or ""
        record.span_id = get_current_span_id() or ""
        record.workspace_id = get_baggage("ccea.workspace_id") or ""
        return True


def setup_trace_logging(logger: logging.Logger) -> None:
    """
    Setup trace context logging for a logger.

    Adds trace_id and span_id to all log records.

    Args:
        logger: Logger to configure
    """
    for handler in logger.handlers:
        handler.addFilter(TraceLoggingFilter())

    # Update formatter to include trace context
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [trace_id=%(trace_id)s] "
        "[span_id=%(span_id)s] %(name)s: %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)
