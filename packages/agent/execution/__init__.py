# -*- coding: utf-8 -*-
"""
CCEA Agent Execution Module.

Live order execution - AGENT ZONE ONLY.
Contains broker connectors and order routing.

This module is PROHIBITED in Cloud zone.

Key Components:
- LiveExecutionEngine: Converts intents to orders
- BrokerConnector: Broker API connections (uses vault for credentials)
- OrderRouter: Routes orders to appropriate broker
"""

from typing import Final, List

ZONE: Final[str] = "agent"

# Components that are AGENT-ONLY
AGENT_ONLY_COMPONENTS: Final[List[str]] = [
    "LiveExecutionEngine",
    "BrokerConnector",
    "OrderRouter",
    "OrderManager",
]

from .engine import (
    LiveExecutionEngine,
    ExecutionResult,
    OrderStatus,
)

from .router import (
    OrderRouter,
    RoutingRule,
    RoutingResult,
)

__all__ = [
    "LiveExecutionEngine",
    "ExecutionResult",
    "OrderStatus",
    "OrderRouter",
    "RoutingRule",
    "RoutingResult",
]
