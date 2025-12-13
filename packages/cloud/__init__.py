# -*- coding: utf-8 -*-
"""
CCEA Cloud Package - Research and Control Plane Zone.

Contains components for:
- Research and backtesting
- Strategy builder and registry
- Control plane (deployment, commands, telemetry)
- Governance (RBAC, retention, residency)

This package contains NO trading code.
NO order execution, NO broker credentials, NO live trading.

Key Principle:
    Cloud = research/build/monitoring/control plane (lifecycle requests)

Cloud NEVER:
    - Stores broker API keys
    - Generates or sends orders
    - Has access to trading endpoints
    - Contains order-like payloads in protocol
    - Can raise Agent's risk limits

Security Guarantees:
    - No trading client libraries
    - No order execution code
    - No access to broker APIs
    - Protocol enforces no order-like payloads
"""

from typing import Final, List

__version__ = "2.0.0"

# Zone identifier
ZONE: Final[str] = "cloud"

# Components provided by this package
CLOUD_COMPONENTS: Final[List[str]] = [
    # Control Plane
    "ControlPlane",
    "CommandDispatcher",
    "TelemetryIngester",
    # Builder
    "ArtifactBuilder",
    "StrategyRegistry",
    # Research
    "BacktestRunner",
    "ResearchJobRunner",
    # Governance
    "GovernanceManager",
    "RBACManager",
]

# Modules that MUST NOT be imported by Cloud
PROHIBITED_IMPORTS: Final[List[str]] = [
    "packages.agent.*",
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "execution_providers*",
    "service_signal_runner",
    "risk_guard",
]

# Third-party packages prohibited in Cloud
PROHIBITED_PACKAGES: Final[List[str]] = [
    # When used for order submission
    # Note: Some packages like ccxt can be used for market data,
    # but submission endpoints must not be called
]
