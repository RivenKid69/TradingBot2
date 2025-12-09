"""
Broker API Terms Compliance Tracking.

Ensures users acknowledge broker-specific terms before submitting
API credentials. This protects both users and the platform by
ensuring compliance with broker API usage policies.

Features:
    - Broker-specific terms warnings
    - Acknowledgment tracking with version control
    - Rate limit information per broker
    - Enforcement mechanism for credential submission

Supported Brokers:
    - Interactive Brokers (IB)
    - Alpaca
    - Binance
    - Coinbase
    - Kraken
    - OANDA

References:
    - Interactive Brokers API Agreement Section 5 (Third-Party Access)
    - Alpaca Platform Agreement, API Terms
    - Binance API Terms of Use
    - Coinbase API User Agreement
    - Kraken API Terms of Service

Example:
    >>> service = BrokerTermsService(storage)
    >>> # Before user submits API keys:
    >>> if not service.has_valid_acknowledgment("user_001", SupportedBroker.ALPACA):
    ...     warning = service.get_broker_warning(SupportedBroker.ALPACA)
    ...     # Show warning to user...
    ...     service.record_acknowledgment("user_001", SupportedBroker.ALPACA, "1.2.3.4")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
import secrets


class SupportedBroker(Enum):
    """Enumeration of supported broker integrations."""

    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    OANDA = "oanda"
    DERIBIT = "deribit"


# ============================================================================
# BROKER-SPECIFIC WARNINGS
# ============================================================================

BROKER_TERMS_WARNINGS: Dict[SupportedBroker, str] = {
    SupportedBroker.INTERACTIVE_BROKERS: """
INTERACTIVE BROKERS API USAGE NOTICE

By connecting your Interactive Brokers account, you acknowledge:

1. THIRD-PARTY ACCESS
   You are using a third-party application to access IB's API.
   Interactive Brokers API Agreement Section 5 applies to this usage.

2. YOUR RESPONSIBILITIES
   - Ensure compliance with IB's Acceptable Use Policy
   - Monitor your account activity regularly
   - Report any suspicious activity immediately

3. PLATFORM CAPABILITIES
   This platform will send orders on your behalf using your credentials.
   The platform does NOT have withdrawal access to your account.

4. RATE LIMITS
   - 50 messages per second maximum
   - Excessive requests may result in connection throttling

5. PAPER TRADING RECOMMENDED
   Test all strategies in IB's paper trading environment before going live.

IB API Documentation: https://interactivebrokers.github.io/
IB API Terms: Contact your IB representative

By proceeding, you confirm that you have read IB's API Agreement
and authorize this platform to execute orders on your behalf.
""",

    SupportedBroker.ALPACA: """
ALPACA API USAGE NOTICE

By connecting your Alpaca account, you acknowledge:

1. THIRD-PARTY PLATFORM
   You are using a third-party platform to access Alpaca's API.
   Alpaca's Platform Agreement and API Terms apply.

2. TRADING AUTHORITY
   You authorize this platform to submit, modify, and cancel orders
   on your behalf using your API credentials.

3. NO WITHDRAWAL ACCESS
   Your API keys should be configured for trading only.
   This platform does NOT require or support withdrawal capabilities.

4. RATE LIMITS
   - 200 API calls per minute for trading endpoints
   - Exceeding limits may result in temporary 429 errors

5. PAPER TRADING
   Alpaca provides paper trading. Test strategies thoroughly
   before deploying to your live account.

6. MARKET DATA
   Access to market data is subject to Alpaca's data agreements.
   Ensure you have appropriate data subscriptions.

Alpaca Terms: https://alpaca.markets/terms-and-conditions
Alpaca API Docs: https://alpaca.markets/docs/api-references/

By proceeding, you confirm acceptance of Alpaca's terms and
authorize this platform to trade on your behalf.
""",

    SupportedBroker.BINANCE: """
BINANCE API USAGE NOTICE

By connecting your Binance account, you acknowledge:

1. API TERMS ACCEPTANCE
   You accept Binance's API Terms of Use and all related policies.
   Automated trading must comply with Binance's guidelines.

2. API KEY SECURITY
   - IP whitelisting is STRONGLY RECOMMENDED
   - Use API keys with trading permissions only
   - Do NOT enable withdrawal permissions

3. RATE LIMITS
   - Weight-based limits (varies by endpoint)
   - Order limits: 10 orders per second, 100,000 per 24 hours
   - Exceeding limits may result in temporary IP ban

4. GEOGRAPHIC RESTRICTIONS
   Binance services are not available in certain jurisdictions.
   Ensure you are eligible to use Binance in your location.

5. FUTURES TRADING (if applicable)
   Futures trading involves leverage and substantial risk.
   Ensure you understand margin and liquidation mechanics.

Binance API Terms: https://www.binance.com/en/terms-api
Binance API Docs: https://binance-docs.github.io/apidocs/

By proceeding, you confirm you are in a permitted jurisdiction
and authorize this platform to trade on your behalf.
""",

    SupportedBroker.COINBASE: """
COINBASE API USAGE NOTICE

By connecting your Coinbase account, you acknowledge:

1. API USER AGREEMENT
   Coinbase's API User Agreement applies to this integration.
   Review the agreement at coinbase.com.

2. PERMISSIONS
   Grant only the minimum permissions required:
   - Trade: Required for order execution
   - View: Required for portfolio monitoring
   - Transfer: NOT required - do NOT enable

3. RATE LIMITS
   - 10 requests per second
   - 10,000 requests per hour
   - Rate limit headers provided in responses

4. ADVANCED TRADE API
   This platform uses Coinbase's Advanced Trade API.
   Different from the legacy Pro API.

5. CUSTODY
   Your assets remain in Coinbase's custody.
   This platform does not have access to your funds directly.

Coinbase API Terms: https://www.coinbase.com/legal/user_agreement
Coinbase API Docs: https://docs.cloud.coinbase.com/

By proceeding, you accept Coinbase's terms and authorize
this platform to execute trades on your behalf.
""",

    SupportedBroker.KRAKEN: """
KRAKEN API USAGE NOTICE

By connecting your Kraken account, you acknowledge:

1. API TERMS OF SERVICE
   Kraken's API Terms of Service apply to all API usage.

2. API KEY CONFIGURATION
   Configure your API key with appropriate permissions:
   - Query Funds: For balance checking
   - Query Orders: For order status
   - Create/Cancel Orders: For trading
   - Do NOT enable: Withdraw Funds

3. RATE LIMITS
   - Private endpoints: varies by tier
   - Public endpoints: more generous limits
   - Counter-based system (not simple rate)

4. TWO-FACTOR AUTHENTICATION
   Consider enabling 2FA on your Kraken account for additional security.

5. FUNDING TIER
   Your funding tier affects withdrawal limits (not applicable here)
   but also some API rate limits.

Kraken Terms: https://www.kraken.com/legal
Kraken API Docs: https://docs.kraken.com/rest/

By proceeding, you accept Kraken's terms and authorize
this platform to trade on your behalf.
""",

    SupportedBroker.OANDA: """
OANDA API USAGE NOTICE

By connecting your OANDA account, you acknowledge:

1. API ACCESS AGREEMENT
   OANDA's API Access Agreement governs this integration.
   Available in your OANDA account settings.

2. ACCOUNT TYPE
   Ensure you have an OANDA Practice or Live account with API access.
   API access may require specific account verification.

3. API KEY PERMISSIONS
   Configure your API key with trading permissions only.
   This platform does NOT require withdrawal or transfer capabilities.

4. RATE LIMITS
   - 120 requests per second for most endpoints
   - Streaming connections are limited
   - Exceeding limits results in 429 responses

5. FOREX LEVERAGE
   Forex trading involves leverage and substantial risk.
   Leverage limits vary by jurisdiction and account type.

6. REGULATORY COMPLIANCE
   OANDA is regulated in multiple jurisdictions.
   Ensure you are trading through the appropriate OANDA entity.

OANDA Terms: https://www.oanda.com/legal/
OANDA API Docs: https://developer.oanda.com/rest-live-v20/

By proceeding, you accept OANDA's terms and authorize
this platform to trade forex on your behalf.
""",

    SupportedBroker.DERIBIT: """
DERIBIT API USAGE NOTICE

By connecting your Deribit account, you acknowledge:

1. API TERMS
   Deribit's Terms of Service apply to all API usage.
   Deribit is registered in Panama.

2. DERIVATIVES TRADING
   Deribit specializes in crypto derivatives (options, futures).
   These instruments are complex and involve substantial risk.

3. API KEY SETUP
   - Enable only necessary permissions
   - Consider using sub-accounts for isolation
   - IP whitelisting recommended

4. RATE LIMITS
   - Non-matching engine: 20 requests/second
   - Matching engine: 10 requests/second
   - WebSocket preferred for real-time data

5. MARGIN AND LIQUIDATION
   Understand margin requirements and liquidation mechanics.
   Cross-margin and portfolio margin have different risks.

6. API KEY PERMISSIONS
   This platform does NOT require withdrawal or transfer capabilities.
   Your API key should have TRADING permissions only.
   Do NOT enable withdrawal permissions.

Deribit Terms: https://www.deribit.com/pages/information/terms
Deribit API Docs: https://docs.deribit.com/

By proceeding, you accept Deribit's terms and acknowledge
the risks of derivative trading.
""",
}


# ============================================================================
# BROKER RATE LIMITS
# ============================================================================

@dataclass(frozen=True)
class BrokerRateLimitInfo:
    """Rate limit information for a broker."""

    orders_per_second: float
    api_calls_per_minute: int
    description: str
    documentation_url: str


BROKER_RATE_LIMITS: Dict[SupportedBroker, BrokerRateLimitInfo] = {
    SupportedBroker.INTERACTIVE_BROKERS: BrokerRateLimitInfo(
        orders_per_second=50.0,
        api_calls_per_minute=3000,
        description="50 messages/second sustained",
        documentation_url="https://interactivebrokers.github.io/",
    ),
    SupportedBroker.ALPACA: BrokerRateLimitInfo(
        orders_per_second=3.3,  # 200/min
        api_calls_per_minute=200,
        description="200 API calls per minute",
        documentation_url="https://alpaca.markets/docs/api-references/",
    ),
    SupportedBroker.BINANCE: BrokerRateLimitInfo(
        orders_per_second=10.0,
        api_calls_per_minute=1200,
        description="Weight-based, ~1200/min typical",
        documentation_url="https://binance-docs.github.io/apidocs/",
    ),
    SupportedBroker.COINBASE: BrokerRateLimitInfo(
        orders_per_second=10.0,
        api_calls_per_minute=600,
        description="10 requests/second, 10K/hour",
        documentation_url="https://docs.cloud.coinbase.com/",
    ),
    SupportedBroker.KRAKEN: BrokerRateLimitInfo(
        orders_per_second=1.0,  # Conservative
        api_calls_per_minute=60,
        description="Counter-based, varies by endpoint",
        documentation_url="https://docs.kraken.com/rest/",
    ),
    SupportedBroker.OANDA: BrokerRateLimitInfo(
        orders_per_second=120.0,
        api_calls_per_minute=7200,
        description="120 requests/second",
        documentation_url="https://developer.oanda.com/",
    ),
    SupportedBroker.DERIBIT: BrokerRateLimitInfo(
        orders_per_second=10.0,
        api_calls_per_minute=600,
        description="10-20 req/sec depending on endpoint",
        documentation_url="https://docs.deribit.com/",
    ),
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BrokerTermsAcknowledgment:
    """
    Records a user's acknowledgment of broker-specific terms.

    Attributes:
        acknowledgment_id: Unique identifier
        user_id: User who acknowledged
        broker: Broker whose terms were acknowledged
        terms_version: Version of the terms text
        acknowledged_at: Timestamp of acknowledgment
        ip_address: IP address at time of acknowledgment
        metadata: Additional context
    """

    acknowledgment_id: str
    user_id: str
    broker: SupportedBroker
    terms_version: str
    acknowledged_at: datetime
    ip_address: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "acknowledgment_id": self.acknowledgment_id,
            "user_id": self.user_id,
            "broker": self.broker.value,
            "terms_version": self.terms_version,
            "acknowledged_at": self.acknowledged_at.isoformat(),
            "ip_address": self.ip_address,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrokerTermsAcknowledgment":
        """Deserialize from dictionary."""
        return cls(
            acknowledgment_id=data["acknowledgment_id"],
            user_id=data["user_id"],
            broker=SupportedBroker(data["broker"]),
            terms_version=data["terms_version"],
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]),
            ip_address=data["ip_address"],
            metadata=data.get("metadata", {}),
        )


class BrokerTermsNotAcknowledgedError(Exception):
    """Raised when broker terms have not been acknowledged."""

    def __init__(self, broker: SupportedBroker, message: Optional[str] = None):
        self.broker = broker
        self.message = message or f"Please review and accept {broker.value} API terms before submitting credentials"
        super().__init__(self.message)


# ============================================================================
# STORAGE PROTOCOL
# ============================================================================

class BrokerTermsStorageProtocol(Protocol):
    """Protocol for broker terms acknowledgment storage."""

    def save(self, acknowledgment: BrokerTermsAcknowledgment) -> None:
        """Save an acknowledgment."""
        ...

    def get_latest(
        self,
        user_id: str,
        broker: SupportedBroker
    ) -> Optional[BrokerTermsAcknowledgment]:
        """Get the latest acknowledgment for a user and broker."""
        ...

    def get_all_for_user(self, user_id: str) -> List[BrokerTermsAcknowledgment]:
        """Get all acknowledgments for a user."""
        ...

    def delete_for_user(self, user_id: str) -> int:
        """Delete all acknowledgments for a user (GDPR deletion)."""
        ...


class InMemoryBrokerTermsStorage:
    """In-memory implementation for testing and development."""

    def __init__(self):
        self._acknowledgments: List[BrokerTermsAcknowledgment] = []

    def save(self, acknowledgment: BrokerTermsAcknowledgment) -> None:
        """Save an acknowledgment."""
        self._acknowledgments.append(acknowledgment)

    def get_latest(
        self,
        user_id: str,
        broker: SupportedBroker
    ) -> Optional[BrokerTermsAcknowledgment]:
        """Get the latest acknowledgment for a user and broker."""
        matching = [
            a for a in self._acknowledgments
            if a.user_id == user_id and a.broker == broker
        ]
        if not matching:
            return None
        return max(matching, key=lambda a: a.acknowledged_at)

    def get_all_for_user(self, user_id: str) -> List[BrokerTermsAcknowledgment]:
        """Get all acknowledgments for a user."""
        return [a for a in self._acknowledgments if a.user_id == user_id]

    def delete_for_user(self, user_id: str) -> int:
        """Delete all acknowledgments for a user."""
        original = len(self._acknowledgments)
        self._acknowledgments = [a for a in self._acknowledgments if a.user_id != user_id]
        return original - len(self._acknowledgments)


# ============================================================================
# SERVICE CLASS
# ============================================================================

class BrokerTermsService:
    """
    Manage broker-specific terms acknowledgments.

    Ensures users understand and accept broker API terms before
    submitting credentials for each broker integration.

    Example:
        >>> storage = InMemoryBrokerTermsStorage()
        >>> service = BrokerTermsService(storage)
        >>>
        >>> # Before user submits Alpaca API keys:
        >>> try:
        ...     service.require_acknowledgment_before_key_submission(
        ...         "user_001", SupportedBroker.ALPACA
        ...     )
        ... except BrokerTermsNotAcknowledgedError:
        ...     warning = service.get_broker_warning(SupportedBroker.ALPACA)
        ...     # Show warning to user...
        ...     service.record_acknowledgment("user_001", SupportedBroker.ALPACA, "1.2.3.4")
    """

    def __init__(self, storage: BrokerTermsStorageProtocol):
        """
        Initialize the broker terms service.

        Args:
            storage: Backend for storing acknowledgments
        """
        self._storage = storage

        # Current versions of broker terms
        # Increment when terms text changes significantly
        self._current_versions: Dict[SupportedBroker, str] = {
            SupportedBroker.INTERACTIVE_BROKERS: "2024.1",
            SupportedBroker.ALPACA: "2024.1",
            SupportedBroker.BINANCE: "2024.1",
            SupportedBroker.COINBASE: "2024.1",
            SupportedBroker.KRAKEN: "2024.1",
            SupportedBroker.OANDA: "2024.1",
            SupportedBroker.DERIBIT: "2024.1",
        }

    def get_supported_brokers(self) -> List[SupportedBroker]:
        """Get list of all supported brokers."""
        return list(SupportedBroker)

    def get_broker_warning(self, broker: SupportedBroker) -> str:
        """
        Get the warning text for a specific broker.

        Args:
            broker: Broker to get warning for

        Returns:
            Warning text to display to users
        """
        return BROKER_TERMS_WARNINGS.get(broker, "")

    def get_broker_rate_limits(self, broker: SupportedBroker) -> Optional[BrokerRateLimitInfo]:
        """
        Get rate limit information for a broker.

        Args:
            broker: Broker to query

        Returns:
            Rate limit information or None if not available
        """
        return BROKER_RATE_LIMITS.get(broker)

    def get_current_version(self, broker: SupportedBroker) -> str:
        """
        Get the current terms version for a broker.

        Args:
            broker: Broker to query

        Returns:
            Version string
        """
        return self._current_versions.get(broker, "2024.1")

    def record_acknowledgment(
        self,
        user_id: str,
        broker: SupportedBroker,
        ip_address: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BrokerTermsAcknowledgment:
        """
        Record a user's acknowledgment of broker terms.

        Args:
            user_id: User acknowledging the terms
            broker: Broker whose terms are being acknowledged
            ip_address: IP address of the request
            metadata: Additional context

        Returns:
            The created acknowledgment record
        """
        ack = BrokerTermsAcknowledgment(
            acknowledgment_id=secrets.token_hex(8),
            user_id=user_id,
            broker=broker,
            terms_version=self._current_versions.get(broker, "2024.1"),
            acknowledged_at=datetime.now(timezone.utc),
            ip_address=ip_address,
            metadata=metadata or {},
        )

        self._storage.save(ack)
        return ack

    def has_valid_acknowledgment(
        self,
        user_id: str,
        broker: SupportedBroker
    ) -> bool:
        """
        Check if user has a valid (current version) acknowledgment.

        Args:
            user_id: User to check
            broker: Broker to check

        Returns:
            True if user has acknowledged current version
        """
        latest = self._storage.get_latest(user_id, broker)

        if latest is None:
            return False

        current_version = self._current_versions.get(broker, "2024.1")
        return latest.terms_version == current_version

    def require_acknowledgment_before_key_submission(
        self,
        user_id: str,
        broker: SupportedBroker
    ) -> None:
        """
        Enforce acknowledgment before API key can be submitted.

        Args:
            user_id: User submitting credentials
            broker: Broker being connected

        Raises:
            BrokerTermsNotAcknowledgedError: If not acknowledged
        """
        if not self.has_valid_acknowledgment(user_id, broker):
            raise BrokerTermsNotAcknowledgedError(
                broker,
                f"Please review and accept {broker.value} API terms before submitting credentials. "
                f"Current terms version: {self._current_versions.get(broker, '2024.1')}"
            )

    def get_pending_acknowledgments(
        self,
        user_id: str,
        brokers: Optional[List[SupportedBroker]] = None
    ) -> List[SupportedBroker]:
        """
        Get list of brokers requiring acknowledgment.

        Args:
            user_id: User to check
            brokers: Specific brokers to check (or all if None)

        Returns:
            List of brokers requiring acknowledgment
        """
        to_check = brokers or list(SupportedBroker)
        pending = []

        for broker in to_check:
            if not self.has_valid_acknowledgment(user_id, broker):
                pending.append(broker)

        return pending

    def get_user_acknowledgments(
        self,
        user_id: str
    ) -> List[BrokerTermsAcknowledgment]:
        """
        Get all acknowledgments for a user.

        Useful for GDPR data access requests.

        Args:
            user_id: User to query

        Returns:
            List of all acknowledgments
        """
        return self._storage.get_all_for_user(user_id)

    def delete_user_acknowledgments(self, user_id: str) -> int:
        """
        Delete all acknowledgments for a user.

        For GDPR right to erasure compliance.

        Args:
            user_id: User to delete

        Returns:
            Number of acknowledgments deleted
        """
        return self._storage.delete_for_user(user_id)

    def update_terms_version(
        self,
        broker: SupportedBroker,
        new_version: str
    ) -> None:
        """
        Update the terms version for a broker.

        After updating, users will need to re-acknowledge before
        submitting new credentials.

        Args:
            broker: Broker to update
            new_version: New version string
        """
        self._current_versions[broker] = new_version

    def get_acknowledgment_status(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get acknowledgment status for all brokers.

        Args:
            user_id: User to check

        Returns:
            Status dictionary for all brokers
        """
        status = {
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "brokers": {},
        }

        for broker in SupportedBroker:
            latest = self._storage.get_latest(user_id, broker)
            current_version = self._current_versions.get(broker, "2024.1")

            if latest:
                is_current = latest.terms_version == current_version
                status["brokers"][broker.value] = {
                    "acknowledged": True,
                    "acknowledged_at": latest.acknowledged_at.isoformat(),
                    "acknowledged_version": latest.terms_version,
                    "current_version": current_version,
                    "is_current": is_current,
                    "requires_reacknowledgment": not is_current,
                }
            else:
                status["brokers"][broker.value] = {
                    "acknowledged": False,
                    "acknowledged_at": None,
                    "acknowledged_version": None,
                    "current_version": current_version,
                    "is_current": False,
                    "requires_reacknowledgment": True,
                }

        return status
