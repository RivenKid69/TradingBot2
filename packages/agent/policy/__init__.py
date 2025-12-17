# -*- coding: utf-8 -*-
"""
CCEA Agent Policy Module.

Enforces risk limits and trading policies LOCALLY.
Cloud cannot override these limits - local policy has PRIORITY.

Key Components:
- PolicyFirewall: Main policy enforcement
- HardCapEnforcer: Absolute limits that cannot be exceeded
- RiskChecker: Pre-trade risk validation

Security Design Commitments:
- Local hard caps cannot be raised by Cloud
- All trades validated before submission
- Kill switch triggers on policy violations
"""

from typing import Final

ZONE: Final[str] = "agent"

from .firewall import (
    PolicyFirewall,
    PolicyConfig,
    PolicyViolation,
    PolicyResult,
)

from .hard_caps import (
    HardCapEnforcer,
    HardCaps,
    HardCapViolation,
)

from .risk_checker import (
    RiskChecker,
    RiskCheckResult,
    PreTradeCheck,
)

__all__ = [
    # Firewall
    "PolicyFirewall",
    "PolicyConfig",
    "PolicyViolation",
    "PolicyResult",
    # Hard Caps
    "HardCapEnforcer",
    "HardCaps",
    "HardCapViolation",
    # Risk Checker
    "RiskChecker",
    "RiskCheckResult",
    "PreTradeCheck",
]
