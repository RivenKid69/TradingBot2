# -*- coding: utf-8 -*-
"""
Tracing Middleware - CLOUD ZONE ONLY.

FastAPI middleware for automatic request tracing.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracer import get_tracer, OPENTELEMETRY_AVAILABLE

if OPENTELEMETRY_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


# ============================================================================
# Middleware Configuration
# ============================================================================


class TracingMiddlewareConfig:
    """Configuration for tracing middleware."""

    def __init__(
        self,
        exclude_paths: Optional[Set[str]] = None,
        exclude_methods: Optional[Set[str]] = None,
        record_request_body: bool = False,
        record_response_body: bool = False,
        max_body_size: int = 1024,
        record_headers: bool = True,
        sensitive_headers: Optional[Set[str]] = None,
    ):
        self.exclude_paths = exclude_paths or {"/health", "/ready", "/metrics"}
        self.exclude_methods = exclude_methods or set()
        self.record_request_body = record_request_body
        self.record_response_body = record_response_body
        self.max_body_size = max_body_size
        self.record_headers = record_headers
        self.sensitive_headers = sensitive_headers or {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
        }


# ============================================================================
# Tracing Middleware
# ============================================================================


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request tracing.

    Creates spans for each HTTP request with:
    - Request attributes (method, path, query params)
    - Response attributes (status code, content type)
    - Timing information
    - Error recording

    Usage:
        app = FastAPI()
        app.add_middleware(TracingMiddleware)
    """

    def __init__(
        self,
        app: Any,
        config: Optional[TracingMiddlewareConfig] = None,
    ):
        super().__init__(app)
        self.config = config or TracingMiddlewareConfig()
        self._tracer = get_tracer("ccea.http")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with tracing."""
        # Skip excluded paths
        if self._should_skip(request):
            return await call_next(request)

        if not OPENTELEMETRY_AVAILABLE:
            return await call_next(request)

        # Create span name
        span_name = f"{request.method} {self._get_route_pattern(request)}"

        # Extract attributes
        attributes = self._extract_request_attributes(request)

        # Start span
        with self._tracer.start_as_current_span(
            span_name,
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            start_time = time.perf_counter()

            try:
                # Process request
                response = await call_next(request)

                # Record response attributes
                self._record_response_attributes(span, response)

                # Set success status
                if response.status_code < 400:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.ERROR))

                return response

            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            finally:
                # Record duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("http.duration_ms", duration_ms)

    def _should_skip(self, request: Request) -> bool:
        """Check if request should skip tracing."""
        if request.url.path in self.config.exclude_paths:
            return True
        if request.method in self.config.exclude_methods:
            return True
        return False

    def _get_route_pattern(self, request: Request) -> str:
        """Get route pattern for span name."""
        # Try to get the matched route pattern
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return route.path

        # Fall back to URL path
        return request.url.path

    def _extract_request_attributes(self, request: Request) -> Dict[str, Any]:
        """Extract span attributes from request."""
        attributes: Dict[str, Any] = {
            SpanAttributes.HTTP_METHOD: request.method,
            SpanAttributes.HTTP_URL: str(request.url),
            SpanAttributes.HTTP_TARGET: request.url.path,
            SpanAttributes.HTTP_SCHEME: request.url.scheme,
            SpanAttributes.HTTP_HOST: request.url.hostname or "",
            SpanAttributes.NET_HOST_PORT: request.url.port or 80,
        }

        # Client info
        if request.client:
            attributes[SpanAttributes.NET_PEER_IP] = request.client.host
            attributes[SpanAttributes.NET_PEER_PORT] = request.client.port

        # User agent
        user_agent = request.headers.get("user-agent")
        if user_agent:
            attributes[SpanAttributes.HTTP_USER_AGENT] = user_agent

        # Request ID
        request_id = request.headers.get("x-request-id")
        if request_id:
            attributes["http.request_id"] = request_id

        # Workspace ID (from path or header)
        workspace_id = request.path_params.get("workspace_id") or \
                       request.headers.get("x-workspace-id")
        if workspace_id:
            attributes["ccea.workspace_id"] = str(workspace_id)

        # Record headers (excluding sensitive)
        if self.config.record_headers:
            for key, value in request.headers.items():
                if key.lower() not in self.config.sensitive_headers:
                    attributes[f"http.request.header.{key}"] = value

        return attributes

    def _record_response_attributes(
        self,
        span: Any,
        response: Response,
    ) -> None:
        """Record response attributes on span."""
        span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)

        # Content type
        content_type = response.headers.get("content-type")
        if content_type:
            span.set_attribute("http.response.content_type", content_type)

        # Response size
        content_length = response.headers.get("content-length")
        if content_length:
            span.set_attribute(
                SpanAttributes.HTTP_RESPONSE_CONTENT_LENGTH,
                int(content_length),
            )


# ============================================================================
# Route Decorator
# ============================================================================


def trace_route(
    name: Optional[str] = None,
    kind: str = "internal",
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to trace a route handler or function.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind (client, server, internal, etc.)
        attributes: Additional span attributes

    Usage:
        @app.get("/items/{item_id}")
        @trace_route(name="get_item")
        async def get_item(item_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = get_tracer()
            span_name = name or func.__name__

            # Map kind string to SpanKind
            kind_map = {
                "client": trace.SpanKind.CLIENT,
                "server": trace.SpanKind.SERVER,
                "producer": trace.SpanKind.PRODUCER,
                "consumer": trace.SpanKind.CONSUMER,
                "internal": trace.SpanKind.INTERNAL,
            }
            span_kind = kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=attributes or {},
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)

            tracer = get_tracer()
            span_name = name or func.__name__

            kind_map = {
                "client": trace.SpanKind.CLIENT,
                "server": trace.SpanKind.SERVER,
                "producer": trace.SpanKind.PRODUCER,
                "consumer": trace.SpanKind.CONSUMER,
                "internal": trace.SpanKind.INTERNAL,
            }
            span_kind = kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
                attributes=attributes or {},
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================================
# Database Tracing
# ============================================================================


def trace_db_query(
    operation: str,
    query: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Context manager for tracing database queries.

    Args:
        operation: Database operation (SELECT, INSERT, etc.)
        query: SQL query (will be truncated for safety)
        parameters: Query parameters (will be redacted)

    Usage:
        with trace_db_query("SELECT", "SELECT * FROM users WHERE id = ?"):
            result = await session.execute(query)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = get_tracer("ccea.database")

            attributes = {
                "db.system": "postgresql",
                "db.operation": operation,
            }

            if query:
                # Truncate query to avoid huge spans
                truncated = query[:500] + "..." if len(query) > 500 else query
                attributes["db.statement"] = truncated

            with tracer.start_as_current_span(
                f"db.{operation.lower()}",
                kind=trace.SpanKind.CLIENT,
                attributes=attributes,
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)

            tracer = get_tracer("ccea.database")

            attributes = {
                "db.system": "postgresql",
                "db.operation": operation,
            }

            if query:
                truncated = query[:500] + "..." if len(query) > 500 else query
                attributes["db.statement"] = truncated

            with tracer.start_as_current_span(
                f"db.{operation.lower()}",
                kind=trace.SpanKind.CLIENT,
                attributes=attributes,
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
