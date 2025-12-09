"""
API Services Module.

Provides API-level services for compliance, disclaimers, and user interactions.

Components:
    - DisclaimerService: User acknowledgment tracking for legal disclaimers

References:
    - MiFID II Article 24(4): Fair, clear, not misleading communications
    - ESMA Guidelines on Product Governance
"""

from services.api.disclaimer_service import (
    DisclaimerService,
    DisclaimerType,
    DisclaimerAcknowledgment,
    DisclaimerNotAcknowledgedError,
    PRE_LIVE_TRADING_DISCLAIMER,
    BACKTEST_RESULTS_DISCLAIMER,
    TERMS_OF_SERVICE_DISCLAIMER,
)

__all__ = [
    "DisclaimerService",
    "DisclaimerType",
    "DisclaimerAcknowledgment",
    "DisclaimerNotAcknowledgedError",
    "PRE_LIVE_TRADING_DISCLAIMER",
    "BACKTEST_RESULTS_DISCLAIMER",
    "TERMS_OF_SERVICE_DISCLAIMER",
]
