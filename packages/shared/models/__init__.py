# -*- coding: utf-8 -*-
"""
CCEA Shared Models.

Re-exports core models that are safe for both Cloud and Agent zones.
These models define data structures without any execution logic.
"""

from typing import Final

ZONE: Final[str] = "shared"

# Re-export from core_models (base data layer; TimeFrame added for P0-A closure)
from core_models import (
    OrderIntent,
    TimeFrame,
    Bar,
    OrderType,
)

# OrderSide / PositionSide are canonically defined in core_futures (a zone-safe
# core_* module). They were never in core_models — importing them from there was
# the P0-A ImportError. Re-export from their real home.
from core_futures import (
    OrderSide,
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
