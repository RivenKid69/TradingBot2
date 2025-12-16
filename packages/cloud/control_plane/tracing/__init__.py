# -*- coding: utf-8 -*-
"""
Distributed Tracing Package - CLOUD ZONE ONLY.

OpenTelemetry-based distributed tracing for Control Plane.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
    - Section 13: "Telemetry: уровни чувствительности"

Components:
    - tracer: Core tracing configuration
    - middleware: FastAPI middleware for automatic tracing
    - context: Context propagation utilities
    - exporters: Trace exporters (Jaeger, OTLP, etc.)
"""

from .tracer import (
    TracingConfig,
    TracingService,
    get_tracer,
    init_tracing,
    shutdown_tracing,
)
from .middleware import (
    TracingMiddleware,
    trace_route,
)
from .context import (
    TraceContext,
    extract_context,
    inject_context,
    get_current_trace_id,
    get_current_span_id,
)

__all__ = [
    "TracingConfig",
    "TracingService",
    "get_tracer",
    "init_tracing",
    "shutdown_tracing",
    "TracingMiddleware",
    "trace_route",
    "TraceContext",
    "extract_context",
    "inject_context",
    "get_current_trace_id",
    "get_current_span_id",
]
