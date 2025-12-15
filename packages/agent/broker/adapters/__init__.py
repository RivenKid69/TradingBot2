# -*- coding: utf-8 -*-
"""
CCEA Agent Broker Adapters.

Provides broker-specific implementations of BrokerConnector protocol.

AGENT ZONE ONLY - Never import in Cloud zone.

Available Adapters:
- alpaca: US Equities via Alpaca
- binance: Crypto via Binance
- oanda: Forex via OANDA

Usage:
    from packages.agent.broker import BrokerConnectorFactory, BrokerCredentials

    creds = BrokerCredentials(api_key="...", api_secret="...")
    connector = BrokerConnectorFactory.create("alpaca", creds, sandbox=True)
"""

# Import adapters to trigger registration
from packages.agent.broker.adapters import alpaca
from packages.agent.broker.adapters import binance

__all__ = ["alpaca", "binance"]
