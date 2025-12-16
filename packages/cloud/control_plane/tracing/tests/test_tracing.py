# -*- coding: utf-8 -*-
"""
Tests for Tracing Package.

Tests for:
- TracingService
- TracingMiddleware
- Context propagation
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from packages.cloud.control_plane.tracing import (
    TracingConfig,
    TracingService,
    get_tracer,
    init_tracing,
    shutdown_tracing,
    TracingMiddleware,
    trace_route,
    TraceContext,
    extract_context,
    inject_context,
    get_current_trace_id,
    get_current_span_id,
)
from packages.cloud.control_plane.tracing.tracer import (
    ExporterType,
    SamplingStrategy,
    _NoOpTracer,
    _NoOpSpan,
)


# ============================================================================
# TracingConfig Tests
# ============================================================================


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TracingConfig()

        assert config.service_name == "ccea-control-plane"
        assert config.exporter_type == ExporterType.CONSOLE
        assert config.sampling_strategy == SamplingStrategy.PARENT_BASED
        assert config.sample_rate == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracingConfig(
            service_name="test-service",
            exporter_type=ExporterType.JAEGER,
            sample_rate=0.5,
        )

        assert config.service_name == "test-service"
        assert config.exporter_type == ExporterType.JAEGER
        assert config.sample_rate == 0.5

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TracingConfig(
            service_name="test",
            environment="production",
        )

        data = config.to_dict()

        assert data["service_name"] == "test"
        assert data["environment"] == "production"


# ============================================================================
# TracingService Tests
# ============================================================================


class TestTracingService:
    """Tests for TracingService."""

    def test_singleton_pattern(self):
        """Test that TracingService is singleton."""
        # Reset singleton
        TracingService._instance = None
        TracingService._initialized = False

        service1 = TracingService()
        service2 = TracingService()

        assert service1 is service2

    def test_initialize(self):
        """Test service initialization."""
        TracingService._instance = None
        TracingService._initialized = False

        config = TracingConfig(exporter_type=ExporterType.CONSOLE)
        service = TracingService(config)

        # Initialize returns bool based on OTel availability
        result = service.initialize()
        assert isinstance(result, bool)

    def test_get_tracer(self):
        """Test getting tracer instance."""
        TracingService._instance = None
        TracingService._initialized = False

        service = TracingService()
        service.initialize()

        tracer = service.get_tracer()

        # Should return either real tracer or no-op
        assert tracer is not None

    def test_create_span(self):
        """Test creating a span."""
        TracingService._instance = None
        TracingService._initialized = False

        service = TracingService()
        service.initialize()

        span = service.create_span("test_operation")

        # Should return context manager
        assert hasattr(span, "__enter__") or hasattr(span, "__aenter__")

    def test_sensitive_attribute_redaction(self):
        """Test that sensitive attributes are redacted."""
        TracingService._instance = None
        TracingService._initialized = False

        config = TracingConfig(redact_sensitive_attributes=True)
        service = TracingService(config)

        attributes = {
            "username": "john",
            "password": "secret123",
            "api_key": "abc123",
            "normal_field": "value",
        }

        redacted = service._redact_sensitive(attributes)

        assert redacted["username"] == "john"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["normal_field"] == "value"


# ============================================================================
# TraceContext Tests
# ============================================================================


class TestTraceContext:
    """Tests for TraceContext."""

    def test_from_empty_headers(self):
        """Test extracting context from empty headers."""
        context = TraceContext.from_headers({})

        assert context.trace_id == ""
        assert context.span_id == ""

    def test_from_headers_with_custom_context(self):
        """Test extracting custom context from headers."""
        headers = {
            "x-workspace-id": "ws-123",
            "x-tenant-id": "tenant-456",
            "x-request-id": "req-789",
        }

        context = TraceContext.from_headers(headers)

        assert context.workspace_id == "ws-123"
        assert context.tenant_id == "tenant-456"
        assert context.request_id == "req-789"

    def test_to_headers(self):
        """Test converting context to headers."""
        context = TraceContext(
            trace_id="abc123",
            span_id="def456",
            workspace_id="ws-123",
            request_id="req-789",
        )

        headers = context.to_headers()

        assert headers.get("x-workspace-id") == "ws-123"
        assert headers.get("x-request-id") == "req-789"

    def test_to_dict(self):
        """Test context serialization."""
        context = TraceContext(
            trace_id="abc",
            span_id="def",
            workspace_id="ws-1",
        )

        data = context.to_dict()

        assert data["trace_id"] == "abc"
        assert data["workspace_id"] == "ws-1"


# ============================================================================
# NoOp Implementation Tests
# ============================================================================


class TestNoOpImplementations:
    """Tests for no-op implementations."""

    def test_noop_tracer(self):
        """Test NoOpTracer methods."""
        tracer = _NoOpTracer()

        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

        span = tracer.start_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_noop_span_context_manager(self):
        """Test NoOpSpan as context manager."""
        span = _NoOpSpan()

        with span:
            pass  # Should not raise

    def test_noop_span_methods(self):
        """Test NoOpSpan methods don't raise."""
        span = _NoOpSpan()

        span.set_attribute("key", "value")
        span.set_attributes({"a": 1, "b": 2})
        span.add_event("event_name")
        span.record_exception(Exception("test"))
        span.set_status(None)
        span.end()

        assert span.is_recording() is False


# ============================================================================
# Module-level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_init_tracing(self):
        """Test init_tracing function."""
        # Reset global
        import packages.cloud.control_plane.tracing.tracer as tracer_module
        tracer_module._tracing_service = None

        service = init_tracing()

        assert service is not None

    def test_get_tracer_initializes_if_needed(self):
        """Test that get_tracer initializes service if needed."""
        import packages.cloud.control_plane.tracing.tracer as tracer_module
        tracer_module._tracing_service = None

        tracer = get_tracer()

        assert tracer is not None

    def test_shutdown_tracing(self):
        """Test shutdown_tracing function."""
        import packages.cloud.control_plane.tracing.tracer as tracer_module

        init_tracing()
        shutdown_tracing()

        assert tracer_module._tracing_service is None


# ============================================================================
# Context Functions Tests
# ============================================================================


class TestContextFunctions:
    """Tests for context propagation functions."""

    def test_get_current_trace_id_without_span(self):
        """Test getting trace ID when no span active."""
        trace_id = get_current_trace_id()

        # Should return None when no active span
        assert trace_id is None or isinstance(trace_id, str)

    def test_get_current_span_id_without_span(self):
        """Test getting span ID when no span active."""
        span_id = get_current_span_id()

        assert span_id is None or isinstance(span_id, str)

    def test_inject_extract_context(self):
        """Test context injection and extraction."""
        carrier = {}

        inject_context(carrier)

        # Should not raise
        context = extract_context(carrier)

        # Context may be None if OTel not available


# ============================================================================
# Middleware Tests
# ============================================================================


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        async def app(scope, receive, send):
            pass
        return app

    def test_middleware_creation(self, mock_app):
        """Test middleware creation."""
        middleware = TracingMiddleware(mock_app)

        assert middleware.app is mock_app

    def test_middleware_config(self, mock_app):
        """Test middleware with custom config."""
        from packages.cloud.control_plane.tracing.middleware import TracingMiddlewareConfig

        config = TracingMiddlewareConfig(
            exclude_paths={"/health", "/metrics"},
            record_headers=False,
        )

        middleware = TracingMiddleware(mock_app, config)

        assert "/health" in middleware.config.exclude_paths


# ============================================================================
# trace_route Decorator Tests
# ============================================================================


class TestTraceRouteDecorator:
    """Tests for trace_route decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decorated(self):
        """Test decorating async function."""
        @trace_route(name="test_operation")
        async def async_handler():
            return "result"

        result = await async_handler()

        assert result == "result"

    def test_sync_function_decorated(self):
        """Test decorating sync function."""
        @trace_route(name="test_sync")
        def sync_handler():
            return "sync_result"

        result = sync_handler()

        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_decorator_with_exception(self):
        """Test decorator handles exceptions."""
        @trace_route(name="failing")
        async def failing_handler():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await failing_handler()

    @pytest.mark.asyncio
    async def test_decorator_with_attributes(self):
        """Test decorator with custom attributes."""
        @trace_route(
            name="custom",
            kind="client",
            attributes={"custom.attr": "value"}
        )
        async def handler():
            return "done"

        result = await handler()

        assert result == "done"
