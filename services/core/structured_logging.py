# -*- coding: utf-8 -*-
"""
Structured Logging with Correlation IDs (Block 2.3).

Implements structured logging for distributed tracing:
- JSON-formatted log entries
- Correlation ID propagation
- Context-aware logging
- DORA-aligned audit trail (evidence-friendly)

DORA References:
    - Article 9: Protection and Prevention (logging requirements)
    - Article 10: Detection (anomaly detection through logs)
    - Article 12: Backup (log retention)
    - RTS CDR 2024/1774: ICT Risk Management Framework

Best Practices:
    - OpenTelemetry Logging Specification
    - ELK Stack Best Practices
    - Cloud Native Logging Guidelines
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

# Context variable for additional context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


# =============================================================================
# Enumerations
# =============================================================================

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel."""
        return cls[level.upper()]


class LogCategory(Enum):
    """Log categories for classification."""
    APPLICATION = "application"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    INCIDENT = "incident"
    COMPLIANCE = "compliance"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CorrelationContext:
    """Correlation context for distributed tracing."""
    correlation_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""

    # Request context
    request_id: str = ""
    session_id: str = ""
    user_id: str = ""

    # Service context
    service_name: str = ""
    service_version: str = ""
    instance_id: str = ""

    # Additional context
    tenant_id: str = ""
    environment: str = ""
    region: str = ""

    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if not self.trace_id:
            self.trace_id = uuid.uuid4().hex[:16]
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:8]

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class StructuredLogEntry:
    """Structured log entry."""
    # Core fields
    timestamp: str = ""
    level: str = "INFO"
    message: str = ""
    logger_name: str = ""

    # Correlation
    correlation_id: str = ""
    trace_id: str = ""
    span_id: str = ""

    # Classification
    category: str = "application"
    event_type: str = ""

    # Context
    service: str = ""
    version: str = ""
    environment: str = ""
    instance_id: str = ""

    # Request context
    request_id: str = ""
    user_id: str = ""
    session_id: str = ""

    # Error information
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""

    # Performance
    duration_ms: Optional[float] = None
    latency_ms: Optional[float] = None

    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)

    # Source location
    file_name: str = ""
    line_number: int = 0
    function_name: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {k: v for k, v in asdict(self).items() if v or v == 0}
        return json.dumps(data, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v or v == 0}


@dataclass
class LoggingConfig:
    """Configuration for StructuredLogger."""
    # Service identification
    service_name: str = "quantitative-research-platform"
    service_version: str = "1.0.0"
    environment: str = "production"
    instance_id: str = ""

    # Log level
    level: LogLevel = LogLevel.INFO

    # Output
    output_format: str = "json"  # json, text
    output_destination: str = "stdout"  # stdout, file, both
    log_file_path: str = "logs/app.log"
    max_file_size_mb: int = 100
    backup_count: int = 10

    # Formatting
    include_timestamp: bool = True
    include_correlation: bool = True
    include_source_location: bool = True
    include_stack_trace: bool = True

    # Performance
    async_logging: bool = False
    buffer_size: int = 1000

    # Retention
    retention_days: int = 90

    # Sensitive data masking
    mask_sensitive_fields: bool = True
    sensitive_field_patterns: List[str] = field(default_factory=lambda: [
        "password", "secret", "token", "api_key", "auth",
        "credit_card", "ssn", "private_key",
    ])

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = uuid.uuid4().hex[:8]


# =============================================================================
# Context Management
# =============================================================================

def get_correlation_id() -> str:
    """Get current correlation ID."""
    cid = _correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID."""
    _correlation_id.set(correlation_id)


def get_log_context() -> Dict[str, Any]:
    """Get current log context."""
    return _log_context.get().copy()


def set_log_context(context: Dict[str, Any]) -> None:
    """Set log context."""
    _log_context.set(context)


def update_log_context(**kwargs: Any) -> None:
    """Update log context with additional fields."""
    current = _log_context.get().copy()
    current.update(kwargs)
    _log_context.set(current)


def clear_log_context() -> None:
    """Clear log context."""
    _log_context.set({})


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    **extra_context: Any,
):
    """
    Context manager for correlation ID and context.

    Usage:
        with correlation_context(correlation_id="abc123", user_id="user1"):
            logger.info("Processing request")
            # All logs in this block will have the correlation_id and user_id
    """
    # Save current state
    old_correlation_id = _correlation_id.get()
    old_context = _log_context.get().copy()

    try:
        # Set new state
        new_correlation_id = correlation_id or str(uuid.uuid4())
        _correlation_id.set(new_correlation_id)

        new_context = old_context.copy()
        new_context.update(extra_context)
        _log_context.set(new_context)

        yield new_correlation_id
    finally:
        # Restore state
        _correlation_id.set(old_correlation_id)
        _log_context.set(old_context)


# =============================================================================
# JSON Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, config: LoggingConfig):
        super().__init__()
        self.config = config
        self._sensitive_patterns = [p.lower() for p in config.sensitive_field_patterns]

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get correlation ID and context
        correlation_id = get_correlation_id()
        context = get_log_context()

        # Build entry
        entry = StructuredLogEntry(
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            correlation_id=correlation_id,
            service=self.config.service_name,
            version=self.config.service_version,
            environment=self.config.environment,
            instance_id=self.config.instance_id,
        )

        # Add context fields
        for key, value in context.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
            else:
                entry.extra[key] = value

        # Add source location
        if self.config.include_source_location:
            entry.file_name = record.filename
            entry.line_number = record.lineno
            entry.function_name = record.funcName

        # Add exception info
        if record.exc_info and self.config.include_stack_trace:
            entry.error_type = record.exc_info[0].__name__ if record.exc_info[0] else ""
            entry.error_message = str(record.exc_info[1]) if record.exc_info[1] else ""
            entry.stack_trace = "".join(traceback.format_exception(*record.exc_info))

        # Add extra fields from record
        if hasattr(record, "extra"):
            entry.extra.update(record.extra)

        # Mask sensitive data
        if self.config.mask_sensitive_fields:
            entry.extra = self._mask_sensitive(entry.extra)

        return entry.to_json()

    def _mask_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields."""
        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in self._sensitive_patterns):
                masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        return masked


# =============================================================================
# Main Logger Class
# =============================================================================

class StructuredLogger:
    """
    Structured Logger per DORA requirements.

    Features:
    - JSON-formatted output
    - Correlation ID propagation
    - Context-aware logging
    - Sensitive data masking
    - Multi-destination output

    Usage:
        config = LoggingConfig(service_name="trading-service")
        logger = StructuredLogger(config)

        # Simple logging
        logger.info("Processing order")

        # With context
        with correlation_context(user_id="user123"):
            logger.info("Order processed", order_id="ORD-001", amount=100.0)

        # With category
        logger.security("Login attempt", user_id="user123", ip="192.168.1.1")

        # With performance metrics
        logger.info("API call completed", duration_ms=45.5, endpoint="/api/orders")
    """

    def __init__(self, config: Optional[LoggingConfig] = None):
        """Initialize Structured Logger."""
        self.config = config or LoggingConfig()

        # Create logger
        self._logger = logging.getLogger(self.config.service_name)
        self._logger.setLevel(getattr(logging, self.config.level.value))

        # Remove existing handlers
        self._logger.handlers = []

        # Add formatter
        self._formatter = StructuredFormatter(self.config)

        # Add handlers
        self._setup_handlers()

        # Thread safety
        self._lock = threading.RLock()

    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # Console handler
        if self.config.output_destination in ("stdout", "both"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._formatter)
            self._logger.addHandler(console_handler)

        # File handler
        if self.config.output_destination in ("file", "both"):
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count,
            )
            file_handler.setFormatter(self._formatter)
            self._logger.addHandler(file_handler)

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def _log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.APPLICATION,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Internal log method."""
        # Update context with extra fields
        extra = {"extra": kwargs, "category": category.value}

        if "event_type" in kwargs:
            extra["event_type"] = kwargs.pop("event_type")

        log_level = getattr(logging, level.value)
        self._logger.log(log_level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with stack trace."""
        self._log(LogLevel.ERROR, message, exc_info=True, **kwargs)

    # =========================================================================
    # Category-Specific Methods
    # =========================================================================

    def security(self, message: str, **kwargs: Any) -> None:
        """Log security event."""
        self._log(LogLevel.INFO, message, category=LogCategory.SECURITY, **kwargs)

    def audit(self, message: str, **kwargs: Any) -> None:
        """Log audit event."""
        self._log(LogLevel.INFO, message, category=LogCategory.AUDIT, **kwargs)

    def performance(self, message: str, **kwargs: Any) -> None:
        """Log performance metric."""
        self._log(LogLevel.INFO, message, category=LogCategory.PERFORMANCE, **kwargs)

    def business(self, message: str, **kwargs: Any) -> None:
        """Log business event."""
        self._log(LogLevel.INFO, message, category=LogCategory.BUSINESS, **kwargs)

    def incident(self, message: str, **kwargs: Any) -> None:
        """Log incident event."""
        self._log(LogLevel.WARNING, message, category=LogCategory.INCIDENT, **kwargs)

    def compliance(self, message: str, **kwargs: Any) -> None:
        """Log compliance event."""
        self._log(LogLevel.INFO, message, category=LogCategory.COMPLIANCE, **kwargs)

    # =========================================================================
    # Contextual Logging
    # =========================================================================

    def with_context(self, **context: Any) -> "ContextualLogger":
        """
        Create a logger with additional context.

        Usage:
            order_logger = logger.with_context(order_id="ORD-001")
            order_logger.info("Processing")  # Will include order_id
        """
        return ContextualLogger(self, context)

    def bind(self, **context: Any) -> "ContextualLogger":
        """Alias for with_context."""
        return self.with_context(**context)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_level(self, level: Union[str, LogLevel]) -> None:
        """Set log level."""
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        self._logger.setLevel(getattr(logging, level.value))
        self.config.level = level

    def get_level(self) -> LogLevel:
        """Get current log level."""
        return self.config.level


class ContextualLogger:
    """Logger with bound context."""

    def __init__(self, parent: StructuredLogger, context: Dict[str, Any]):
        self._parent = parent
        self._context = context

    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge bound context with call-time kwargs."""
        merged = self._context.copy()
        merged.update(kwargs)
        return merged

    def debug(self, message: str, **kwargs: Any) -> None:
        self._parent.debug(message, **self._merge_context(kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._parent.info(message, **self._merge_context(kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._parent.warning(message, **self._merge_context(kwargs))

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._parent.error(message, exc_info=exc_info, **self._merge_context(kwargs))

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._parent.critical(message, exc_info=exc_info, **self._merge_context(kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        self._parent.exception(message, **self._merge_context(kwargs))

    def security(self, message: str, **kwargs: Any) -> None:
        self._parent.security(message, **self._merge_context(kwargs))

    def audit(self, message: str, **kwargs: Any) -> None:
        self._parent.audit(message, **self._merge_context(kwargs))

    def with_context(self, **context: Any) -> "ContextualLogger":
        """Add more context."""
        merged = self._context.copy()
        merged.update(context)
        return ContextualLogger(self._parent, merged)


# =============================================================================
# Factory Functions
# =============================================================================

def create_structured_logger(
    config: Optional[LoggingConfig] = None,
) -> StructuredLogger:
    """
    Create a StructuredLogger instance.

    Args:
        config: Optional configuration

    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(config=config)


# =============================================================================
# Global Logger Instance
# =============================================================================

_global_logger: Optional[StructuredLogger] = None
_global_lock = threading.Lock()


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get the global structured logger.

    Args:
        name: Optional logger name (for namespacing)

    Returns:
        StructuredLogger instance
    """
    global _global_logger

    with _global_lock:
        if _global_logger is None:
            _global_logger = create_structured_logger()

    if name:
        return _global_logger.with_context(logger_name=name)

    return _global_logger


def configure_logging(config: LoggingConfig) -> StructuredLogger:
    """
    Configure global logging.

    Args:
        config: Logging configuration

    Returns:
        Configured StructuredLogger
    """
    global _global_logger

    with _global_lock:
        _global_logger = create_structured_logger(config)

    return _global_logger
