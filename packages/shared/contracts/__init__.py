# -*- coding: utf-8 -*-
"""
CCEA Shared Contracts.

Core contracts and interfaces shared between Cloud and Agent zones.
These define the API boundary and data exchange formats.

Key Types:
- OrderIntent: Strategy's expression of desired action (NOT an order)
- StrategyContract: Interface that all strategies must implement
- TelemetryEvent: Standard telemetry format
- ConfigContract: Configuration structure
"""

from .intent import OrderIntent, IntentType, IntentSide
from .strategy import StrategyContract, StrategyResult, StrategyContext
from .config import (
    StrategyConfig,
    RiskConfig,
    ExecutionConfig,
    ConfigContract,
)
from .telemetry import (
    TelemetryEvent,
    TelemetryLevel,
    MetricType,
)
from .manifest import (
    ArtifactManifest,
    ManifestSchema,
    Provenance,
    RuntimeRequirements,
)

__all__ = [
    # Intent
    "OrderIntent",
    "IntentType",
    "IntentSide",
    # Strategy
    "StrategyContract",
    "StrategyResult",
    "StrategyContext",
    # Config
    "StrategyConfig",
    "RiskConfig",
    "ExecutionConfig",
    "ConfigContract",
    # Telemetry
    "TelemetryEvent",
    "TelemetryLevel",
    "MetricType",
    # Manifest
    "ArtifactManifest",
    "ManifestSchema",
    "Provenance",
    "RuntimeRequirements",
]
