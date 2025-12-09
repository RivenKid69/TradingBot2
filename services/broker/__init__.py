"""
Broker Services Module.

Provides broker-related compliance services including terms acknowledgment
and rate limiting for API protection.

Components:
    - BrokerTermsService: Broker API terms acknowledgment tracking
    - SupportedBroker: Enumeration of supported brokers
    - BrokerRateLimiter: Pre-broker rate limiting with circuit breaker
    - RunawayDetector: Runaway strategy detection and prevention

References:
    - Interactive Brokers API Agreement Section 5 (Third-Party Access)
    - Alpaca Platform Agreement and API Terms
    - Binance API Terms of Use
    - RFC 6585: Token Bucket Rate Limiting
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

from services.broker.rate_limiter import (
    BrokerRateLimiter,
    BrokerRateLimits,
    RateLimitStatus,
    RateLimitCheck,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    RunawayDetector,
    RunawayStrategyError,
    BROKER_LIMITS,
)

# Alias for convenience
BROKER_WARNINGS = BROKER_TERMS_WARNINGS

__all__ = [
    # Terms compliance
    "BrokerTermsService",
    "SupportedBroker",
    "BrokerTermsAcknowledgment",
    "BrokerTermsNotAcknowledgedError",
    "BrokerRateLimitInfo",
    "BROKER_WARNINGS",
    "BROKER_TERMS_WARNINGS",
    "BROKER_RATE_LIMITS",
    # Rate limiting
    "BrokerRateLimiter",
    "BrokerRateLimits",
    "RateLimitStatus",
    "RateLimitCheck",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",
    "RunawayDetector",
    "RunawayStrategyError",
    "BROKER_LIMITS",
]
