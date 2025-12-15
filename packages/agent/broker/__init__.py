# -*- coding: utf-8 -*-
"""
CCEA Agent Broker Module.

Provides BrokerConnector protocol and adapters for live trading.

AGENT ZONE ONLY - Never import in Cloud zone.

Design Doc References:
- Section 4.2: Broker Connectors
- Section 9.3: Live Loop
- Section 9.4: Kill Switch
"""

from packages.agent.broker.protocol import (
    # Enums
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    PositionSide,
    ConnectionStatus,
    # Data classes
    BrokerCredentials,
    OrderRequest,
    OrderResult,
    OrderInfo,
    Position,
    AccountInfo,
    CancelResult,
    BulkCancelResult,
    # Protocol
    BrokerConnector,
    BaseBrokerConnector,
    # Factory
    BrokerConnectorFactory,
)

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "OrderStatus",
    "PositionSide",
    "ConnectionStatus",
    # Data classes
    "BrokerCredentials",
    "OrderRequest",
    "OrderResult",
    "OrderInfo",
    "Position",
    "AccountInfo",
    "CancelResult",
    "BulkCancelResult",
    # Protocol
    "BrokerConnector",
    "BaseBrokerConnector",
    # Factory
    "BrokerConnectorFactory",
]
