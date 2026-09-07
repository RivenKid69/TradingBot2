"""
User acknowledgment tracking for legal disclaimers.

Ensures users explicitly acknowledge important risk warnings and legal
notices before accessing specific platform features.

References:
    - MiFID II Article 24(4): Fair, clear, not misleading communications
    - ESMA Guidelines on Product Governance
    - EU E-Commerce Directive 2000/31/EC Art. 5, 6

Features:
    - Version-controlled disclaimers
    - Acknowledgment tracking with audit trail
    - Enforcement mechanism for feature access
    - Multi-disclaimer support (ToS, risk warnings, etc.)

Example:
    >>> service = DisclaimerService(storage)
    >>> # Check if user can access live trading
    >>> if not service.has_valid_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING):
    ...     # Show disclaimer and require acceptance
    ...     text = service.get_disclaimer_text(DisclaimerType.PRE_LIVE_TRADING)
    ...     # After user accepts:
    ...     service.record_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING, "1.2.3.4", "Mozilla...")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class DisclaimerType(Enum):
    """Types of disclaimers that users must acknowledge."""

    PRE_LIVE_TRADING = "pre_live_trading"
    BACKTEST_RESULTS = "backtest_results"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    RISK_WARNING = "risk_warning"
    STRATEGY_DEPLOYMENT = "strategy_deployment"


# ============================================================================
# DISCLAIMER TEXTS
# ============================================================================

PRE_LIVE_TRADING_DISCLAIMER = """
IMPORTANT: READ BEFORE ENABLING LIVE TRADING

1. RISK WARNING
   Trading financial instruments involves substantial risk of loss.
   You may lose more than your initial investment. Past performance
   does NOT guarantee future results.

2. NO INVESTMENT ADVICE
   This platform is a software tool. It does NOT provide:
   - Investment advice or recommendations
   - Portfolio management services
   - Personalized financial guidance

   You are solely responsible for your trading decisions.

3. YOUR RESPONSIBILITY
   By enabling live trading, you confirm that:
   - You are using YOUR OWN broker account and API keys
   - You understand the strategies you deploy
   - You can afford to lose the capital you are risking
   - You will monitor your positions and risk exposure

4. BROKER CREDENTIALS
   Your broker API credentials will be used to execute orders
   on your behalf, exactly as your strategies dictate. The platform:
   - Does NOT modify your strategies
   - Does NOT have withdrawal access to your funds
   - Executes orders as programmed by YOU

5. TECHNICAL RISKS
   Live trading involves technical risks including:
   - Software bugs that may cause unintended orders
   - Network failures affecting order execution
   - Market conditions differing from backtests
   - Slippage and execution quality variations

6. SIMULATED VS REAL PERFORMANCE
   Backtest results are simulations based on historical data.
   Real trading results may differ significantly due to:
   - Market impact and liquidity constraints
   - Execution timing and slippage
   - Fee structures and borrowing costs
   - Market regime changes

By clicking "I Understand and Accept", you confirm that:
- You have read and understood the above warnings
- You are 18 years or older (or legal age in your jurisdiction)
- You accept full responsibility for your trading activities
- You understand this is software, not financial advice

THIS ACKNOWLEDGMENT IS LEGALLY BINDING.
"""

BACKTEST_RESULTS_DISCLAIMER = """
BACKTEST RESULTS DISCLAIMER

The results shown are based on HISTORICAL SIMULATION and do NOT
represent actual trading results.

IMPORTANT LIMITATIONS:
1. Past performance does NOT guarantee future results
2. Simulated results have inherent limitations
3. These results do not account for all real-world factors

SIMULATION ASSUMPTIONS:
- Perfect execution (may not occur in live trading)
- Historical data accuracy (may contain errors)
- No market impact (large orders affect prices)
- Idealized slippage models

WHAT BACKTESTS CANNOT PREDICT:
- Future market conditions
- Liquidity during your actual trades
- Black swan events and market crashes
- Changes in market structure or regulations

USE OF RESULTS:
Backtest results should be used ONLY as one input in your
research process, not as predictions of future performance.

RISK WARNING:
Trading involves substantial risk of loss. Only trade with
capital you can afford to lose.
"""

TERMS_OF_SERVICE_DISCLAIMER = """
TERMS OF SERVICE ACKNOWLEDGMENT

By using this platform, you agree to be bound by our Terms of Service.

KEY POINTS:
1. This platform provides software tools, NOT investment advice
2. You are responsible for your own trading decisions
3. Liability is limited as described in the Terms
4. Your data is handled according to our Privacy Policy
5. You must comply with applicable laws and regulations

Please review the full Terms of Service and Privacy Policy
documents accessible from your account settings.

By clicking "I Accept", you confirm that you have read,
understood, and agree to be bound by our Terms of Service.
"""

RISK_WARNING_GENERAL = """
GENERAL RISK WARNING

TRADING FINANCIAL INSTRUMENTS CARRIES SIGNIFICANT RISKS.

Before trading, you should:
1. Understand the nature of the instruments you trade
2. Be aware you can lose more than your initial investment
3. Consider your financial situation and risk tolerance
4. Seek independent advice if unsure

This platform does not provide investment advice.
All trading decisions are yours alone.
"""

STRATEGY_DEPLOYMENT_DISCLAIMER = """
STRATEGY DEPLOYMENT WARNING

You are about to deploy a trading strategy to live markets.

PLEASE CONFIRM:
1. You have thoroughly tested this strategy in paper trading
2. You understand the strategy's logic and risk profile
3. You have appropriate risk management in place
4. You can afford potential losses from this strategy

DEPLOYMENT RISKS:
- Strategy may behave differently in live markets
- Market conditions may have changed since backtesting
- Technical issues may affect execution
- You are responsible for monitoring the strategy

This action will execute real trades with real money.
Proceed only if you accept these risks.
"""

# Map of disclaimer types to their text content
DISCLAIMER_TEXTS: Dict[DisclaimerType, str] = {
    DisclaimerType.PRE_LIVE_TRADING: PRE_LIVE_TRADING_DISCLAIMER,
    DisclaimerType.BACKTEST_RESULTS: BACKTEST_RESULTS_DISCLAIMER,
    DisclaimerType.TERMS_OF_SERVICE: TERMS_OF_SERVICE_DISCLAIMER,
    DisclaimerType.RISK_WARNING: RISK_WARNING_GENERAL,
    DisclaimerType.STRATEGY_DEPLOYMENT: STRATEGY_DEPLOYMENT_DISCLAIMER,
}


@dataclass
class DisclaimerAcknowledgment:
    """
    Records a user's acknowledgment of a disclaimer.

    Attributes:
        acknowledgment_id: Unique identifier
        user_id: User who acknowledged
        disclaimer_type: Type of disclaimer acknowledged
        disclaimer_version: Version of the disclaimer text
        acknowledged_at: Timestamp of acknowledgment
        ip_address: IP address at time of acknowledgment
        user_agent: Browser/client user agent
        consent_text_hash: Hash of the exact text shown (for verification)
        metadata: Additional context
    """

    acknowledgment_id: str
    user_id: str
    disclaimer_type: DisclaimerType
    disclaimer_version: str
    acknowledged_at: datetime
    ip_address: str
    user_agent: str
    consent_text_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "acknowledgment_id": self.acknowledgment_id,
            "user_id": self.user_id,
            "disclaimer_type": self.disclaimer_type.value,
            "disclaimer_version": self.disclaimer_version,
            "acknowledged_at": self.acknowledged_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "consent_text_hash": self.consent_text_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisclaimerAcknowledgment":
        """Deserialize from dictionary."""
        return cls(
            acknowledgment_id=data["acknowledgment_id"],
            user_id=data["user_id"],
            disclaimer_type=DisclaimerType(data["disclaimer_type"]),
            disclaimer_version=data["disclaimer_version"],
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]),
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            consent_text_hash=data.get("consent_text_hash"),
            metadata=data.get("metadata", {}),
        )


class DisclaimerNotAcknowledgedError(Exception):
    """Raised when a required disclaimer has not been acknowledged."""

    def __init__(self, disclaimer_type: DisclaimerType, message: Optional[str] = None):
        self.disclaimer_type = disclaimer_type
        self.message = message or f"User must acknowledge {disclaimer_type.value} before proceeding"
        super().__init__(self.message)


class DisclaimerStorageProtocol(Protocol):
    """Protocol for disclaimer acknowledgment storage backends."""

    def save(self, acknowledgment: DisclaimerAcknowledgment) -> None:
        """Save an acknowledgment."""
        ...

    def get_latest(
        self, user_id: str, disclaimer_type: DisclaimerType
    ) -> Optional[DisclaimerAcknowledgment]:
        """Get the latest acknowledgment for a user and disclaimer type."""
        ...

    def get_all_for_user(self, user_id: str) -> List[DisclaimerAcknowledgment]:
        """Get all acknowledgments for a user."""
        ...

    def delete_for_user(self, user_id: str) -> int:
        """Delete all acknowledgments for a user (GDPR deletion)."""
        ...


class InMemoryDisclaimerStorage:
    """
    In-memory implementation of disclaimer storage.

    Suitable for testing and development.
    """

    def __init__(self):
        self._acknowledgments: List[DisclaimerAcknowledgment] = []

    def save(self, acknowledgment: DisclaimerAcknowledgment) -> None:
        """Save an acknowledgment."""
        self._acknowledgments.append(acknowledgment)

    def get_latest(
        self, user_id: str, disclaimer_type: DisclaimerType
    ) -> Optional[DisclaimerAcknowledgment]:
        """Get the latest acknowledgment for a user and disclaimer type."""
        matching = [
            a
            for a in self._acknowledgments
            if a.user_id == user_id and a.disclaimer_type == disclaimer_type
        ]
        if not matching:
            return None
        return max(matching, key=lambda a: a.acknowledged_at)

    def get_all_for_user(self, user_id: str) -> List[DisclaimerAcknowledgment]:
        """Get all acknowledgments for a user."""
        return [a for a in self._acknowledgments if a.user_id == user_id]

    def delete_for_user(self, user_id: str) -> int:
        """Delete all acknowledgments for a user."""
        original = len(self._acknowledgments)
        self._acknowledgments = [a for a in self._acknowledgments if a.user_id != user_id]
        return original - len(self._acknowledgments)


class DisclaimerService:
    """
    Service for managing user acknowledgments of legal disclaimers.

    Tracks which users have acknowledged which disclaimers, enforces
    acknowledgment requirements, and supports version-controlled
    disclaimer updates.

    Example:
        >>> storage = InMemoryDisclaimerStorage()
        >>> service = DisclaimerService(storage)
        >>>
        >>> # Check before allowing live trading
        >>> try:
        ...     service.require_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING)
        ... except DisclaimerNotAcknowledgedError:
        ...     text = service.get_disclaimer_text(DisclaimerType.PRE_LIVE_TRADING)
        ...     # Show text to user...
        ...     service.record_acknowledgment(
        ...         "user_001", DisclaimerType.PRE_LIVE_TRADING,
        ...         "1.2.3.4", "Mozilla/5.0..."
        ...     )
    """

    def __init__(self, storage: DisclaimerStorageProtocol):
        """
        Initialize the disclaimer service.

        Args:
            storage: Backend for storing acknowledgments
        """
        self._storage = storage

        # Current versions of each disclaimer
        # Increment when disclaimer text changes significantly
        self._current_versions: Dict[DisclaimerType, str] = {
            DisclaimerType.PRE_LIVE_TRADING: "1.0.0",
            DisclaimerType.BACKTEST_RESULTS: "1.0.0",
            DisclaimerType.TERMS_OF_SERVICE: "1.0.0",
            DisclaimerType.PRIVACY_POLICY: "1.0.0",
            DisclaimerType.RISK_WARNING: "1.0.0",
            DisclaimerType.STRATEGY_DEPLOYMENT: "1.0.0",
        }

    def get_disclaimer_text(self, disclaimer_type: DisclaimerType) -> str:
        """
        Get the current text for a disclaimer.

        Args:
            disclaimer_type: Type of disclaimer

        Returns:
            The disclaimer text to display to users
        """
        return DISCLAIMER_TEXTS.get(disclaimer_type, "")

    def get_disclaimer_version(self, disclaimer_type: DisclaimerType) -> str:
        """
        Get the current version for a disclaimer.

        Args:
            disclaimer_type: Type of disclaimer

        Returns:
            Version string (e.g., "1.0.0")
        """
        return self._current_versions.get(disclaimer_type, "1.0.0")

    def record_acknowledgment(
        self,
        user_id: str,
        disclaimer_type: DisclaimerType,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DisclaimerAcknowledgment:
        """
        Record a user's acknowledgment of a disclaimer.

        Args:
            user_id: User acknowledging the disclaimer
            disclaimer_type: Type of disclaimer being acknowledged
            ip_address: IP address of the request
            user_agent: User agent string
            metadata: Additional context

        Returns:
            The created acknowledgment record
        """
        import hashlib
        import secrets

        # Get current disclaimer text and hash it
        text = self.get_disclaimer_text(disclaimer_type)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        # Create acknowledgment record
        ack = DisclaimerAcknowledgment(
            acknowledgment_id=secrets.token_hex(8),
            user_id=user_id,
            disclaimer_type=disclaimer_type,
            disclaimer_version=self._current_versions[disclaimer_type],
            acknowledged_at=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
            consent_text_hash=text_hash,
            metadata=metadata or {},
        )

        self._storage.save(ack)
        return ack

    def has_valid_acknowledgment(self, user_id: str, disclaimer_type: DisclaimerType) -> bool:
        """
        Check if user has a valid (current version) acknowledgment.

        Args:
            user_id: User to check
            disclaimer_type: Type of disclaimer

        Returns:
            True if user has acknowledged the current version
        """
        latest = self._storage.get_latest(user_id, disclaimer_type)

        if latest is None:
            return False

        # Check if the acknowledged version matches current version
        current_version = self._current_versions.get(disclaimer_type, "1.0.0")
        return latest.disclaimer_version == current_version

    def require_acknowledgment(self, user_id: str, disclaimer_type: DisclaimerType) -> None:
        """
        Require that a user has acknowledged a disclaimer.

        Raises an exception if the user has not acknowledged
        the current version of the disclaimer.

        Args:
            user_id: User to check
            disclaimer_type: Type of disclaimer required

        Raises:
            DisclaimerNotAcknowledgedError: If user hasn't acknowledged
        """
        if not self.has_valid_acknowledgment(user_id, disclaimer_type):
            raise DisclaimerNotAcknowledgedError(
                disclaimer_type,
                f"User must acknowledge {disclaimer_type.value} disclaimer (version "
                f"{self._current_versions.get(disclaimer_type, '1.0.0')}) before proceeding",
            )

    def get_pending_disclaimers(
        self, user_id: str, required_types: Optional[List[DisclaimerType]] = None
    ) -> List[DisclaimerType]:
        """
        Get list of disclaimers the user needs to acknowledge.

        Args:
            user_id: User to check
            required_types: Specific types to check (or all if None)

        Returns:
            List of disclaimer types requiring acknowledgment
        """
        types_to_check = required_types or list(DisclaimerType)
        pending = []

        for dtype in types_to_check:
            if not self.has_valid_acknowledgment(user_id, dtype):
                pending.append(dtype)

        return pending

    def get_user_acknowledgments(self, user_id: str) -> List[DisclaimerAcknowledgment]:
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

    def update_disclaimer_version(self, disclaimer_type: DisclaimerType, new_version: str) -> None:
        """
        Update the version of a disclaimer.

        After updating, users will need to re-acknowledge the disclaimer.

        Args:
            disclaimer_type: Type of disclaimer to update
            new_version: New version string
        """
        self._current_versions[disclaimer_type] = new_version

    def get_acknowledgment_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a report of user's acknowledgment status.

        Args:
            user_id: User to report on

        Returns:
            Report dictionary with status for each disclaimer type
        """
        report = {
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimers": {},
        }

        for dtype in DisclaimerType:
            latest = self._storage.get_latest(user_id, dtype)
            current_version = self._current_versions.get(dtype, "1.0.0")

            if latest:
                is_current = latest.disclaimer_version == current_version
                report["disclaimers"][dtype.value] = {
                    "acknowledged": True,
                    "acknowledged_at": latest.acknowledged_at.isoformat(),
                    "acknowledged_version": latest.disclaimer_version,
                    "current_version": current_version,
                    "is_current": is_current,
                    "requires_reacknowledgment": not is_current,
                }
            else:
                report["disclaimers"][dtype.value] = {
                    "acknowledged": False,
                    "acknowledged_at": None,
                    "acknowledged_version": None,
                    "current_version": current_version,
                    "is_current": False,
                    "requires_reacknowledgment": True,
                }

        return report


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================


def require_live_trading_acknowledgment(
    disclaimer_service: DisclaimerService, user_id: str
) -> None:
    """
    Helper function to enforce live trading disclaimer.

    Use this before executing live orders.

    Args:
        disclaimer_service: Disclaimer service instance
        user_id: User attempting to trade

    Raises:
        DisclaimerNotAcknowledgedError: If not acknowledged
    """
    disclaimer_service.require_acknowledgment(user_id, DisclaimerType.PRE_LIVE_TRADING)


def require_strategy_deployment_acknowledgment(
    disclaimer_service: DisclaimerService, user_id: str
) -> None:
    """
    Helper function to enforce strategy deployment disclaimer.

    Use this before deploying a strategy to live markets.

    Args:
        disclaimer_service: Disclaimer service instance
        user_id: User deploying strategy

    Raises:
        DisclaimerNotAcknowledgedError: If not acknowledged
    """
    disclaimer_service.require_acknowledgment(user_id, DisclaimerType.STRATEGY_DEPLOYMENT)
