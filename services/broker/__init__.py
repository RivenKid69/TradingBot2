"""
Broker Services Module.

Provides broker-related compliance services including terms acknowledgment
and rate limiting for API protection.

Components:
    - BrokerTermsService: Broker API terms acknowledgment tracking
    - SupportedBroker: Enumeration of supported brokers

References:
    - Interactive Brokers API Agreement Section 5 (Third-Party Access)
    - Alpaca Platform Agreement and API Terms
    - Binance API Terms of Use
"""

from services.broker.terms_compliance import (
    BrokerTermsService,
    SupportedBroker,
    BrokerTermsAcknowledgment,
    BrokerTermsNotAcknowledgedError,
    BrokerRateLimitInfo,
    BROKER_TERMS_WARNINGS,
    BROKER_RATE_LIMITS,
)

# Alias for convenience
BROKER_WARNINGS = BROKER_TERMS_WARNINGS

__all__ = [
    "BrokerTermsService",
    "SupportedBroker",
    "BrokerTermsAcknowledgment",
    "BrokerTermsNotAcknowledgedError",
    "BrokerRateLimitInfo",
    "BROKER_WARNINGS",
    "BROKER_TERMS_WARNINGS",
    "BROKER_RATE_LIMITS",
]
