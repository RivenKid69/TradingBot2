# -*- coding: utf-8 -*-
"""
CCEA Agent Execution Module.

Live order execution - AGENT ZONE ONLY.
Contains execution engine primitives and (optionally) order routing.

This module is PROHIBITED in Cloud zone.

Key Components:
- LiveExecutionEngine: Converts intents to orders
- OrderRouter: Routes orders to an execution backend (optional)
"""

from __future__ import annotations

from typing import Final, List

ZONE: Final[str] = "agent"

# Components that are AGENT-ONLY
AGENT_ONLY_COMPONENTS: Final[List[str]] = [
    "LiveExecutionEngine",
    "OrderRouter",
    "OrderManager",
]

from .engine import (
    LiveExecutionEngine,
    ExecutionResult,
    OrderStatus,
)

__all__ = [
    "LiveExecutionEngine",
    "ExecutionResult",
    "OrderStatus",
]

# Order routing is optional in OSS splits (broker integrations/routing rules may be proprietary plugins).
try:
    from .router import (  # noqa: F401
        OrderRouter,
        RoutingRule,
        RoutingResult,
    )

    __all__.extend(["OrderRouter", "RoutingRule", "RoutingResult"])
except ImportError:
    pass
