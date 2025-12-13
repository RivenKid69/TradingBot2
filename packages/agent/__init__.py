# -*- coding: utf-8 -*-
"""
CCEA Agent Package - Live Execution Zone.

Contains all components that require access to:
- Broker API keys and credentials (Local Vault)
- Order creation and submission (Execution Engine)
- Live trading connections (Broker Connectors)
- Risk enforcement (Policy Firewall)
- Position reconciliation

This package MUST NEVER be imported by Cloud zone.
Cloud zone imports from packages/shared only.

Security Architecture:
- Local Vault: Encrypted storage for broker credentials
- Policy Firewall: Enforces hard caps, Cloud cannot override
- Approval Workflow: Local approval for TRADING_IMPACTING changes
- Kill Switch: Automatic halt on risk thresholds
- Reconciliation: Position sync and idempotent orders

Key Principle:
    Agent = secrets + live loop + risk enforce + order creation/sending
    Agent NEVER exposes credentials to Cloud
"""

from typing import Final, List

__version__ = "2.0.0"

# Zone identifier
ZONE: Final[str] = "agent"

# Components provided by this package
AGENT_COMPONENTS: Final[List[str]] = [
    # Vault
    "LocalVault",
    "CredentialManager",
    # Policy
    "PolicyFirewall",
    "HardCapEnforcer",
    "RiskChecker",
    # Execution
    "LiveExecutionEngine",
    "BrokerConnector",
    "OrderRouter",
    # Reconciliation
    "PositionReconciler",
    "OrderJournal",
    # Approval
    "ApprovalManager",
    "LocalApprovalUI",
]

# Modules that belong to AGENT zone only
AGENT_ONLY_MODULES: Final[List[str]] = [
    "packages.agent.*",
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "execution_providers*",
    "service_signal_runner",
    "risk_guard",
]

# Modules that MUST NOT be imported BY agent (from Cloud internals)
PROHIBITED_IMPORTS: Final[List[str]] = [
    "packages.cloud.internal.*",
]
