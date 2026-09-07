# -*- coding: utf-8 -*-
"""
Adapter Zone Mapping - Defines which adapters belong to which zone.

Phase 2 Implementation: Hard separation of Cloud/Agent/Shared zones.

DATA-ONLY adapters (SHARED zone - safe for Cloud and Agent):
- market_data.py - Public market data
- fees.py - Fee calculations
- trading_hours.py - Market hours
- exchange_info.py - Symbol info, trading rules

TRADING-ONLY adapters (AGENT zone - never in Cloud):
- order_execution.py - Live order submission
- options_execution.py - Options order submission
- Any private trading API access

This module provides:
1. Zone classification for each adapter module
2. Validation functions for CI/CD
3. Import checking utilities
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Final, FrozenSet, List, Set


class AdapterZone(str, Enum):
    """Zone classification for adapters."""

    SHARED = "shared"  # Data-only, safe for Cloud and Agent
    AGENT = "agent"  # Trading-only, never in Cloud
    INTERNAL = "internal"  # Internal utilities


# ============================================================================
# Data-Only Adapters (SHARED zone)
# ============================================================================

DATA_ONLY_MODULES: Final[FrozenSet[str]] = frozenset(
    [
        # Market data modules
        "adapters.alpaca.market_data",
        "adapters.binance.market_data",
        "adapters.binance.futures_market_data",
        "adapters.oanda.market_data",
        "adapters.ib.market_data",
        "adapters.polygon.market_data",
        "adapters.yahoo.market_data",
        "adapters.deribit.websocket",
        # Fee modules
        "adapters.alpaca.fees",
        "adapters.binance.fees",
        "adapters.oanda.fees",
        # Trading hours modules
        "adapters.alpaca.trading_hours",
        "adapters.binance.trading_hours",
        "adapters.oanda.trading_hours",
        "adapters.polygon.trading_hours",
        # Exchange info modules
        "adapters.alpaca.exchange_info",
        "adapters.binance.exchange_info",
        "adapters.binance.futures_exchange_info",
        "adapters.oanda.exchange_info",
        "adapters.ib.exchange_info",
        "adapters.polygon.exchange_info",
        # Options data (read-only)
        "adapters.ib.options",
        "adapters.polygon.options",
        "adapters.deribit.options",
        "adapters.theta_data.options",
        # Other data modules
        "adapters.yahoo.earnings",
        "adapters.yahoo.corporate_actions",
        "adapters.deribit.margin",
        # Base and models (shared infrastructure)
        "adapters.base",
        "adapters.models",
        "adapters.registry",
        "adapters.config",
        "adapters.websocket_base",
    ]
)

# ============================================================================
# Trading-Only Adapters (AGENT zone)
# ============================================================================

TRADING_ONLY_MODULES: Final[FrozenSet[str]] = frozenset(
    [
        # Order execution modules
        "adapters.alpaca.order_execution",
        "adapters.alpaca.options_execution",
        "adapters.binance.futures_order_execution",
        "adapters.binance_spot_private",
        "adapters.oanda.order_execution",
        "adapters.ib.order_execution",
        "adapters.ib.options_combo",
        # Live execution providers
        "execution_providers",
        "execution_providers_l3",
        "execution_providers_futures",
        "execution_providers_futures_l3",
        "execution_providers_cme",
        "execution_providers_cme_l3",
        # Live signal runner
        "service_signal_runner",
        # Live scripts
        "script_live",
        "script_futures_live",
    ]
)

# ============================================================================
# Module -> Zone Mapping
# ============================================================================


def get_adapter_zone(module_name: str) -> AdapterZone:
    """
    Get zone for adapter module.

    Args:
        module_name: Full module name (e.g., 'adapters.alpaca.order_execution')

    Returns:
        AdapterZone classification
    """
    if module_name in TRADING_ONLY_MODULES:
        return AdapterZone.AGENT

    if module_name in DATA_ONLY_MODULES:
        return AdapterZone.SHARED

    # Check patterns
    if any(
        pattern in module_name for pattern in ["order_execution", "options_execution", "_private"]
    ):
        return AdapterZone.AGENT

    if any(
        pattern in module_name
        for pattern in ["market_data", "fees", "trading_hours", "exchange_info"]
    ):
        return AdapterZone.SHARED

    # Default to internal
    return AdapterZone.INTERNAL


def is_cloud_safe(module_name: str) -> bool:
    """
    Check if module is safe for Cloud zone.

    Args:
        module_name: Module to check

    Returns:
        True if module can be imported in Cloud
    """
    zone = get_adapter_zone(module_name)
    return zone in (AdapterZone.SHARED, AdapterZone.INTERNAL)


def is_agent_only(module_name: str) -> bool:
    """
    Check if module is Agent-only.

    Args:
        module_name: Module to check

    Returns:
        True if module must not be in Cloud
    """
    return get_adapter_zone(module_name) == AdapterZone.AGENT


def get_cloud_prohibited_modules() -> FrozenSet[str]:
    """Get all modules prohibited in Cloud."""
    return TRADING_ONLY_MODULES


def get_shared_modules() -> FrozenSet[str]:
    """Get all modules safe for sharing."""
    return DATA_ONLY_MODULES


def validate_cloud_imports(imported_modules: Set[str]) -> List[str]:
    """
    Validate that imported modules are Cloud-safe.

    Args:
        imported_modules: Set of imported module names

    Returns:
        List of violations (module names that shouldn't be in Cloud)
    """
    violations = []
    for module in imported_modules:
        if is_agent_only(module):
            violations.append(module)
    return violations


def validate_zone_separation() -> Dict[str, List[str]]:
    """
    Validate zone separation is properly configured.

    Returns:
        Dict with any issues found
    """
    issues = {}

    # Check for overlaps
    overlaps = DATA_ONLY_MODULES & TRADING_ONLY_MODULES
    if overlaps:
        issues["overlapping_modules"] = list(overlaps)

    return issues


# ============================================================================
# Documentation
# ============================================================================

ZONE_DOCUMENTATION: Final[
    str
] = """
CCEA Adapter Zone Separation (Phase 2)
=====================================

SHARED Zone (Data-Only):
- Can be imported by Cloud and Agent
- Contains: market_data, fees, trading_hours, exchange_info
- NO order submission capabilities
- NO private trading API access

AGENT Zone (Trading-Only):
- Can ONLY be imported by Agent
- Contains: order_execution, options_execution
- Has order submission capabilities
- Has private trading API access

Zone Rules:
1. Cloud MUST NOT import AGENT zone modules
2. Agent CAN import all zones
3. Shared modules cannot depend on Cloud or Agent

CI Enforcement:
- importlinter.ini enforces import boundaries
- ccea/guardrails/import_check.py validates builds
- Pre-commit hooks check changes
"""
