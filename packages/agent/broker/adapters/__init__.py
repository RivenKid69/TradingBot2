# -*- coding: utf-8 -*-
"""
CCEA Agent Broker Adapters.

Provides broker-specific implementations of BrokerConnector protocol.

AGENT ZONE ONLY - Never import in Cloud zone.

Available Adapters:
- alpaca: US Equities via Alpaca
- binance: Crypto via Binance
- ib:     CME Futures via Interactive Brokers (P2 #26)
- oanda:  Forex via OANDA (P2 #26)
- sim:    In-process paper broker

Usage:
    from packages.agent.broker import BrokerConnectorFactory, BrokerCredentials

    creds = BrokerCredentials(api_key="...", api_secret="...")
    connector = BrokerConnectorFactory.create("alpaca", creds, sandbox=True)
"""

# Import adapters to trigger registration
from packages.agent.broker.adapters import alpaca
from packages.agent.broker.adapters import binance

# IB / OANDA Agent connectors (P2 #26) — import lazily-safe (delegating base only).
from packages.agent.broker.adapters.ib import IBConnector
from packages.agent.broker.adapters.oanda import OANDAConnector

__all__ = ["alpaca", "binance", "IBConnector", "OANDAConnector"]
