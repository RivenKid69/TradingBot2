# -*- coding: utf-8 -*-
"""
CCEA Shared Models.

Re-exports core models that are safe for both Cloud and Agent zones.
These models define data structures without any execution logic.
"""

from typing import Final

ZONE: Final[str] = "shared"

# Re-export from core_models
from core_models import (
    OrderIntent,
    TimeFrame,
    Bar,
    OrderSide,
    OrderType,
    PositionSide,
)

# Re-export contracts
from packages.shared.contracts import (
    # Intent
    IntentType,
    IntentSide,
    # Strategy
    StrategyContract,
    StrategyResult,
    StrategyContext,
    MarketSnapshot,
    # Config
    StrategyConfig,
    RiskConfig,
    ExecutionConfig,
    ExecutionMode,
    RiskLevel,
    ChangeClass,
    # Telemetry
    TelemetryEvent,
    TelemetryLevel,
    MetricType,
    # Manifest
    ArtifactManifest,
    Provenance,
    RuntimeRequirements,
)

__all__ = [
    # Core models
    "OrderIntent",
    "TimeFrame",
    "Bar",
    "OrderSide",
    "OrderType",
    "PositionSide",
    # Intent
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
    "ExecutionMode",
    "RiskLevel",
    "ChangeClass",
    # Telemetry
    "TelemetryEvent",
    "TelemetryLevel",
    "MetricType",
    # Manifest
    "ArtifactManifest",
    "Provenance",
    "RuntimeRequirements",
]
