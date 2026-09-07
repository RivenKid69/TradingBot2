# -*- coding: utf-8 -*-
"""
Agent Telemetry Module.

CCEA Phase 8 Implementation: Telemetry + Privacy/GDPR + Residency

This module provides:
    - TelemetryRedactionMiddleware: Mandatory redaction layer
    - DLPService: Data Loss Prevention for telemetry
    - TelemetryLevelManager: Controls telemetry verbosity
    - TelemetryExporter: Exports telemetry with privacy controls

Design Doc Reference:
    - Phase 8: "Telemetry + Privacy/GDPR + Residency + Access Controls"
    - Mandatory redaction + DLP (13.2)
    - Telemetry levels and defaults (13.1)
"""

from .redaction import (
    TelemetryRedactionMiddleware,
    RedactionConfig,
    RedactionPattern,
    RedactionResult,
    RedactionStats,
)
from .dlp import (
    DLPService,
    DLPConfig,
    DLPRule,
    DLPViolation,
    DLPAction,
    SensitivityLevel,
)
from .level_manager import (
    TelemetryLevelManager,
    TelemetryLevelConfig,
    TelemetryMode,
)

# Exporter requires cryptography, make it optional
try:
    from .exporter import (
        TelemetryExporter,
        ExportConfig,
        ExportFormat,
        ExportResult,
    )

    _HAS_EXPORTER = True
except ImportError:
    _HAS_EXPORTER = False
    TelemetryExporter = None
    ExportConfig = None
    ExportFormat = None
    ExportResult = None

__all__ = [
    # Redaction
    "TelemetryRedactionMiddleware",
    "RedactionConfig",
    "RedactionPattern",
    "RedactionResult",
    "RedactionStats",
    # DLP
    "DLPService",
    "DLPConfig",
    "DLPRule",
    "DLPViolation",
    "DLPAction",
    "SensitivityLevel",
    # Level Manager
    "TelemetryLevelManager",
    "TelemetryLevelConfig",
    "TelemetryMode",
    # Exporter
    "TelemetryExporter",
    "ExportConfig",
    "ExportFormat",
    "ExportResult",
]
