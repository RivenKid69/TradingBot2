# -*- coding: utf-8 -*-
"""
Distributed Tracing Core - CLOUD ZONE ONLY.

OpenTelemetry tracer configuration and management.

Design Doc Reference:
    - Section 4.1: "Telemetry Ingestion & Monitoring"

Features:
    - W3C Trace Context propagation
    - Multiple exporter support (Jaeger, OTLP, Zipkin)
    - Sampling strategies
    - Resource attributes
    - Baggage propagation
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import (
        Sampler,
        TraceIdRatioBased,
        ParentBased,
        ALWAYS_ON,
        ALWAYS_OFF,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not installed. Tracing disabled.")


# ============================================================================
# Constants
# ============================================================================

SERVICE_VERSION: Final[str] = "1.0.0"
DEFAULT_SAMPLE_RATE: Final[float] = 1.0  # 100% sampling in dev
PRODUCTION_SAMPLE_RATE: Final[float] = 0.1  # 10% in production


class ExporterType(str, Enum):
    """Trace exporter types."""

    CONSOLE = "console"
    JAEGER = "jaeger"
    OTLP = "otlp"
    ZIPKIN = "zipkin"
    NONE = "none"


class SamplingStrategy(str, Enum):
    """Sampling strategies."""

    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    RATIO = "ratio"
    PARENT_BASED = "parent_based"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TracingConfig:
    """
    Tracing configuration.

    Supports multiple exporters and sampling strategies.
    """

    # Service identification
    service_name: str = "ccea-control-plane"
    service_version: str = SERVICE_VERSION
    environment: str = "development"

    # Exporter settings
    exporter_type: ExporterType = ExporterType.CONSOLE
    exporter_endpoint: Optional[str] = None

    # Jaeger settings
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    jaeger_collector_endpoint: Optional[str] = None

    # OTLP settings
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True

    # Sampling
    sampling_strategy: SamplingStrategy = SamplingStrategy.PARENT_BASED
    sample_rate: float = DEFAULT_SAMPLE_RATE

    # Batch processing
    batch_export: bool = True
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000

    # Resource attributes
    resource_attributes: Dict[str, str] = field(default_factory=dict)

    # Security
    redact_sensitive_attributes: bool = True
    sensitive_attribute_patterns: List[str] = field(
        default_factory=lambda: [
            "password",
            "secret",
            "token",
            "key",
            "credential",
            "auth",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "exporter_type": self.exporter_type.value,
            "sampling_strategy": self.sampling_strategy.value,
            "sample_rate": self.sample_rate,
        }


# ============================================================================
# Tracing Service
# ============================================================================


class TracingService:
    """
    Manages distributed tracing infrastructure.

    Thread-safe singleton for application-wide tracing.
    """

    _instance: Optional["TracingService"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[TracingConfig] = None):
        if self._initialized:
            return

        self.config = config or TracingConfig()
        self._provider: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._processors: List[Any] = []

        TracingService._initialized = True

    def initialize(self) -> bool:
        """
        Initialize tracing infrastructure.

        Returns:
            True if successfully initialized
        """
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return False

        try:
            # Create resource
            resource = self._create_resource()

            # Create sampler
            sampler = self._create_sampler()

            # Create provider
            self._provider = TracerProvider(
                resource=resource,
                sampler=sampler,
            )

            # Add exporters
            self._setup_exporters()

            # Set as global provider
            trace.set_tracer_provider(self._provider)

            # Setup propagators
            self._setup_propagators()

            # Create tracer
            self._tracer = trace.get_tracer(
                self.config.service_name,
                self.config.service_version,
            )

            logger.info(
                f"Tracing initialized: service={self.config.service_name}, "
                f"exporter={self.config.exporter_type.value}"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize tracing: {e}")
            return False

    def _create_resource(self) -> "Resource":
        """Create resource with service attributes."""
        attributes = {
            SERVICE_NAME: self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
            "service.namespace": "ccea",
        }

        # Add custom attributes
        attributes.update(self.config.resource_attributes)

        # Add environment-based attributes
        if os.getenv("K8S_POD_NAME"):
            attributes["k8s.pod.name"] = os.getenv("K8S_POD_NAME", "")
        if os.getenv("K8S_NAMESPACE"):
            attributes["k8s.namespace.name"] = os.getenv("K8S_NAMESPACE", "")
        if os.getenv("K8S_NODE_NAME"):
            attributes["k8s.node.name"] = os.getenv("K8S_NODE_NAME", "")

        return Resource(attributes=attributes)

    def _create_sampler(self) -> "Sampler":
        """Create sampler based on configuration."""
        if self.config.sampling_strategy == SamplingStrategy.ALWAYS_ON:
            return ALWAYS_ON
        elif self.config.sampling_strategy == SamplingStrategy.ALWAYS_OFF:
            return ALWAYS_OFF
        elif self.config.sampling_strategy == SamplingStrategy.RATIO:
            return TraceIdRatioBased(self.config.sample_rate)
        else:  # PARENT_BASED
            return ParentBased(root=TraceIdRatioBased(self.config.sample_rate))

    def _setup_exporters(self) -> None:
        """Setup trace exporters."""
        if self.config.exporter_type == ExporterType.NONE:
            return

        processor = None

        if self.config.exporter_type == ExporterType.CONSOLE:
            exporter = ConsoleSpanExporter()
            processor = SimpleSpanProcessor(exporter)

        elif self.config.exporter_type == ExporterType.JAEGER:
            processor = self._create_jaeger_processor()

        elif self.config.exporter_type == ExporterType.OTLP:
            processor = self._create_otlp_processor()

        elif self.config.exporter_type == ExporterType.ZIPKIN:
            processor = self._create_zipkin_processor()

        if processor and self._provider:
            self._provider.add_span_processor(processor)
            self._processors.append(processor)

    def _create_jaeger_processor(self) -> Optional["SpanProcessor"]:
        """Create Jaeger exporter processor."""
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
                collector_endpoint=self.config.jaeger_collector_endpoint,
            )

            if self.config.batch_export:
                return BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_millis,
                )
            return SimpleSpanProcessor(exporter)

        except ImportError:
            logger.warning("Jaeger exporter not installed, falling back to console")
            return SimpleSpanProcessor(ConsoleSpanExporter())

    def _create_otlp_processor(self) -> Optional["SpanProcessor"]:
        """Create OTLP exporter processor."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
            )

            if self.config.batch_export:
                return BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_millis,
                )
            return SimpleSpanProcessor(exporter)

        except ImportError:
            logger.warning("OTLP exporter not installed, falling back to console")
            return SimpleSpanProcessor(ConsoleSpanExporter())

    def _create_zipkin_processor(self) -> Optional["SpanProcessor"]:
        """Create Zipkin exporter processor."""
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            endpoint = self.config.exporter_endpoint or "http://localhost:9411/api/v2/spans"
            exporter = ZipkinExporter(endpoint=endpoint)

            if self.config.batch_export:
                return BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.max_queue_size,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_millis,
                )
            return SimpleSpanProcessor(exporter)

        except ImportError:
            logger.warning("Zipkin exporter not installed, falling back to console")
            return SimpleSpanProcessor(ConsoleSpanExporter())

    def _setup_propagators(self) -> None:
        """Setup context propagators."""
        # W3C Trace Context + Baggage
        propagator = CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
        set_global_textmap(propagator)

    def get_tracer(self, name: Optional[str] = None) -> Any:
        """
        Get a tracer instance.

        Args:
            name: Optional tracer name (defaults to service name)

        Returns:
            Tracer instance or no-op tracer if not initialized
        """
        if not OPENTELEMETRY_AVAILABLE or not self._tracer:
            return _NoOpTracer()

        if name:
            return trace.get_tracer(name, self.config.service_version)

        return self._tracer

    def shutdown(self) -> None:
        """Shutdown tracing and flush pending spans."""
        if self._provider:
            try:
                self._provider.shutdown()
                logger.info("Tracing shutdown complete")
            except Exception as e:
                logger.warning(f"Error during tracing shutdown: {e}")

    def create_span(
        self,
        name: str,
        kind: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new span.

        Args:
            name: Span name
            kind: Span kind (client, server, producer, consumer, internal)
            attributes: Span attributes

        Returns:
            Span context manager
        """
        tracer = self.get_tracer()

        if not OPENTELEMETRY_AVAILABLE:
            return _NoOpSpan()

        span_kind = trace.SpanKind.INTERNAL
        if kind:
            kind_map = {
                "client": trace.SpanKind.CLIENT,
                "server": trace.SpanKind.SERVER,
                "producer": trace.SpanKind.PRODUCER,
                "consumer": trace.SpanKind.CONSUMER,
                "internal": trace.SpanKind.INTERNAL,
            }
            span_kind = kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

        # Redact sensitive attributes
        safe_attributes = self._redact_sensitive(attributes or {})

        return tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=safe_attributes,
        )

    def _redact_sensitive(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive attribute values."""
        if not self.config.redact_sensitive_attributes:
            return attributes

        result = {}
        for key, value in attributes.items():
            key_lower = key.lower()
            is_sensitive = any(
                pattern in key_lower for pattern in self.config.sensitive_attribute_patterns
            )
            result[key] = "[REDACTED]" if is_sensitive else value

        return result


# ============================================================================
# No-Op Implementations
# ============================================================================


class _NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, *args, **kwargs):
        return _NoOpSpan()

    def start_span(self, *args, **kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def is_recording(self) -> bool:
        return False


# ============================================================================
# Module-level Functions
# ============================================================================

_tracing_service: Optional[TracingService] = None


def init_tracing(config: Optional[TracingConfig] = None) -> TracingService:
    """
    Initialize global tracing service.

    Args:
        config: Optional tracing configuration

    Returns:
        TracingService instance
    """
    global _tracing_service

    if _tracing_service is None:
        _tracing_service = TracingService(config)
        _tracing_service.initialize()

    return _tracing_service


def get_tracer(name: Optional[str] = None) -> Any:
    """
    Get a tracer instance.

    Args:
        name: Optional tracer name

    Returns:
        Tracer instance
    """
    if _tracing_service is None:
        init_tracing()

    return _tracing_service.get_tracer(name) if _tracing_service else _NoOpTracer()


def shutdown_tracing() -> None:
    """Shutdown tracing service."""
    global _tracing_service

    if _tracing_service:
        _tracing_service.shutdown()
        _tracing_service = None
