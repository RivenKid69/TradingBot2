# -*- coding: utf-8 -*-
"""
CCEA Telemetry Module.

Provides:
- Telemetry event collection
- MANDATORY redaction (cannot be disabled)
- Multi-level telemetry (aggregated/detailed/raw)
- Secure export

Security (Design Doc 13-14):
- Redaction is ALWAYS enabled
- No secrets in telemetry
- No account identifiers exposed
- Privacy by design
"""

from ccea.telemetry.collector import (
    TelemetryCollector,
    TelemetryEvent,
    TelemetryLevel,
)

from ccea.telemetry.redaction import (
    RedactionMiddleware,
    RedactionRule,
    MANDATORY_REDACTION_RULES,
    redact_data,
)

from ccea.telemetry.exporter import (
    TelemetryExporter,
    TelemetryBatch,
)

__all__ = [
    # Collector
    "TelemetryCollector",
    "TelemetryEvent",
    "TelemetryLevel",
    # Redaction
    "RedactionMiddleware",
    "RedactionRule",
    "MANDATORY_REDACTION_RULES",
    "redact_data",
    # Exporter
    "TelemetryExporter",
    "TelemetryBatch",
]
