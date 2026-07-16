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
from .strategy import StrategyContract, StrategyResult, StrategyContext, MarketSnapshot
from .config import (
    StrategyConfig,
    RiskConfig,
    ExecutionConfig,
    ConfigContract,
    ExecutionMode,
    RiskLevel,
    ChangeClass,
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
    "MarketSnapshot",
    # Config
    "StrategyConfig",
    "RiskConfig",
    "ExecutionConfig",
    "ConfigContract",
    "ExecutionMode",
    "RiskLevel",
    "ChangeClass",
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
