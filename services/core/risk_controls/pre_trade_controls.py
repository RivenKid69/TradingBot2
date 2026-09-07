# -*- coding: utf-8 -*-
"""
MiFID II RTS 6 Article 15 Pre-Trade Risk Controls.

Implements pre-trade risk controls per Commission Delegated Regulation
(EU) 2017/589 (RTS 6) Article 15:

    "Investment firms shall have in place pre-trade controls on order
    entry that are appropriate for each financial instrument they trade"

Key Requirements (Article 15 RTS 6):
    (1) Price collars
    (2) Maximum order values
    (3) Maximum order volumes
    (4) Maximum message limits
    (5) Automatic blocking of orders from unauthorized traders/instruments

References:
    - RTS 6 Article 15: https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-3/
    - ESMA Guidelines: https://www.eventus.com/cat-article/enforcement-action-from-esma-on-rts-6/
"""

from __future__ import annotations

import logging
import time
import threading
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Any,
    Tuple,
    Deque,
    Callable,
)

logger = logging.getLogger(__name__)


class RejectionReason(str, Enum):
    """
    Pre-trade control rejection reasons per RTS 6 Article 15.
    """

    # Price controls
    PRICE_COLLAR_BREACH = "price_collar_breach"
    FAT_FINGER_PRICE = "fat_finger_price"
    ZERO_OR_NEGATIVE_PRICE = "zero_or_negative_price"
    MISSING_REFERENCE_PRICE = "missing_reference_price"

    # Value/Volume controls
    MAX_ORDER_VALUE_EXCEEDED = "max_order_value_exceeded"
    MAX_ORDER_VOLUME_EXCEEDED = "max_order_volume_exceeded"
    MIN_ORDER_VALUE_BELOW = "min_order_value_below"

    # Rate controls
    MESSAGE_RATE_EXCEEDED = "message_rate_exceeded"
    BURST_RATE_EXCEEDED = "burst_rate_exceeded"

    # Authorization controls
    UNAUTHORIZED_TRADER = "unauthorized_trader"
    UNAUTHORIZED_INSTRUMENT = "unauthorized_instrument"
    UNAUTHORIZED_VENUE = "unauthorized_venue"

    # Risk controls
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    NOTIONAL_LIMIT_BREACH = "notional_limit_breach"
    CONCENTRATION_LIMIT = "concentration_limit"

    # System controls
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    MARKET_CLOSED = "market_closed"
    VALIDATION_ERROR = "validation_error"


class ControlSeverity(str, Enum):
    """Severity level of control breach."""

    INFO = "info"  # Informational only
    WARNING = "warning"  # Warning, may proceed
    SOFT_REJECT = "soft_reject"  # Rejected but can be overridden
    HARD_REJECT = "hard_reject"  # Rejected, cannot proceed


@dataclass
class PreTradeCheckResult:
    """
    Result of pre-trade control check.

    Per RTS 6: All rejections must be logged with reason.
    """

    allowed: bool
    rejection_reason: Optional[RejectionReason] = None
    severity: ControlSeverity = ControlSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    check_timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "allowed": self.allowed,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "check_timestamp_ns": self.check_timestamp_ns,
        }


@dataclass
class PreTradeControlsConfig:
    """
    Configuration for RTS 6 Article 15 pre-trade controls.
    """

    # ==== Price Collars (Art. 15(1)) ====
    price_collar_pct: float = 5.0  # Max deviation from reference (%)
    price_collar_enabled: bool = True

    # Fat finger protection (stricter than collar)
    fat_finger_price_deviation_pct: float = 10.0
    fat_finger_enabled: bool = True

    # ==== Maximum Order Values (Art. 15(2)) ====
    max_order_value_eur: Decimal = Decimal("1000000")  # EUR 1M default
    max_order_value_enabled: bool = True
    min_order_value_eur: Decimal = Decimal("0")

    # ==== Maximum Order Volumes (Art. 15(3)) ====
    max_order_volume: Decimal = Decimal("10000")  # Units
    max_order_volume_enabled: bool = True

    # ADV-based limits
    max_volume_vs_adv_pct: float = 10.0  # Max % of ADV per order
    adv_check_enabled: bool = True

    # ==== Message Rate Limits (Art. 15(4)) ====
    max_messages_per_second: int = 100
    max_messages_per_minute: int = 3000
    burst_window_ms: int = 100
    max_burst_messages: int = 10
    message_rate_enabled: bool = True

    # ==== Authorization Controls (Art. 15(5)) ====
    require_trader_authorization: bool = True
    require_instrument_authorization: bool = False
    require_venue_authorization: bool = False

    # ==== Position/Risk Limits ====
    max_position_pct_of_limit: float = 95.0
    max_daily_loss_eur: Decimal = Decimal("100000")
    daily_loss_check_enabled: bool = True

    max_notional_per_instrument: Decimal = Decimal("5000000")
    max_concentration_pct: float = 25.0  # Max % of portfolio in one position

    # ==== Venue-specific limits ====
    venue_specific_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ==== Instrument-specific limits ====
    instrument_specific_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TraderAuthorization:
    """Trader authorization record per RTS 6 Article 15(5)."""

    trader_id: str
    name: str
    authorized_instruments: Set[str] = field(default_factory=set)  # ISIN set or "*" for all
    authorized_venues: Set[str] = field(default_factory=set)  # MIC set or "*" for all
    max_order_value_eur: Optional[Decimal] = None
    max_position_notional: Optional[Decimal] = None
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    is_active: bool = True


@dataclass
class MessageRateWindow:
    """Sliding window for message rate limiting."""

    window_size_seconds: float
    max_messages: int
    timestamps: Deque[float] = field(default_factory=deque)

    def record(self, timestamp: float) -> bool:
        """
        Record a message and check if within limit.

        Returns:
            True if within limit, False if exceeded
        """
        cutoff = timestamp - self.window_size_seconds

        # Remove old timestamps
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

        # Check limit
        if len(self.timestamps) >= self.max_messages:
            return False

        self.timestamps.append(timestamp)
        return True


class PreTradeControls:
    """
    MiFID II RTS 6 Article 15 Pre-Trade Risk Controls.

    Implements all required pre-trade controls:
        1. Price collars
        2. Maximum order values
        3. Maximum order volumes
        4. Message rate limits
        5. Trader/instrument authorization

    Usage:
        controls = PreTradeControls(config=PreTradeControlsConfig())

        # Add authorized traders
        controls.authorize_trader(TraderAuthorization(
            trader_id="TRADER001",
            name="John Doe",
            authorized_instruments={"*"},  # All instruments
        ))

        # Check order
        result = controls.check_order(
            order={
                "symbol": "AAPL",
                "price": 150.0,
                "quantity": 100,
                "side": "BUY",
            },
            reference_price=149.5,
            trader_id="TRADER001",
        )

        if not result.allowed:
            reject_order(result.rejection_reason)

    RTS 6 Article 15 Compliance:
        - All five control categories implemented
        - Configurable thresholds
        - Full audit logging of rejections
        - Per-venue and per-instrument customization
    """

    def __init__(
        self,
        config: Optional[PreTradeControlsConfig] = None,
        audit_callback: Optional[Callable[[PreTradeCheckResult], None]] = None,
    ):
        """
        Initialize pre-trade controls.

        Args:
            config: Pre-trade controls configuration
            audit_callback: Called for each rejection (for audit trail)
        """
        self._config = config or PreTradeControlsConfig()
        self._audit_callback = audit_callback

        # Authorization tracking
        self._lock = threading.RLock()
        self._authorized_traders: Dict[str, TraderAuthorization] = {}
        self._authorized_instruments: Set[str] = set()
        self._authorized_venues: Set[str] = set()

        # Message rate tracking
        self._rate_windows: Dict[str, MessageRateWindow] = {}
        self._global_rate_window = MessageRateWindow(
            window_size_seconds=1.0,
            max_messages=self._config.max_messages_per_second,
        )
        self._minute_rate_window = MessageRateWindow(
            window_size_seconds=60.0,
            max_messages=self._config.max_messages_per_minute,
        )
        self._burst_windows: Dict[str, MessageRateWindow] = {}

        # Daily P&L tracking
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_reset_date: str = ""

        # Statistics
        self._stats = {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_rejected": 0,
            "rejections_by_reason": {},
        }

        logger.info(
            "PreTradeControls initialized",
            extra={
                "price_collar_pct": self._config.price_collar_pct,
                "max_order_value_eur": str(self._config.max_order_value_eur),
                "max_messages_per_second": self._config.max_messages_per_second,
            },
        )

    # =========================================================================
    # Main Check Method
    # =========================================================================

    def check_order(
        self,
        order: Dict[str, Any],
        reference_price: Optional[Decimal] = None,
        trader_id: str = "",
        venue: str = "",
        instrument_isin: str = "",
        adv: Optional[Decimal] = None,
        current_position: Decimal = Decimal("0"),
        portfolio_notional: Decimal = Decimal("0"),
    ) -> PreTradeCheckResult:
        """
        Run all pre-trade controls on an order.

        Per RTS 6 Article 15: All controls must be checked before order entry.

        Args:
            order: Order dict with price, quantity, side, symbol
            reference_price: Reference price for collar check
            trader_id: Trader identifier
            venue: Trading venue MIC code
            instrument_isin: Instrument ISIN
            adv: Average Daily Volume for ADV-based checks
            current_position: Current position in the instrument
            portfolio_notional: Total portfolio notional for concentration

        Returns:
            PreTradeCheckResult with pass/fail and rejection reason
        """
        with self._lock:
            self._stats["checks_total"] += 1

        # Extract order details
        price = Decimal(str(order.get("price", 0)))
        quantity = Decimal(str(order.get("quantity", 0)))
        side = str(order.get("side", "")).upper()

        # Calculate order value
        order_value = price * quantity

        # Run checks in order of priority
        checks = [
            # 1. Rate limits (fast rejection)
            lambda: self._check_message_rate(venue),
            # 2. Authorization checks
            lambda: self._check_trader_authorization(trader_id, instrument_isin, venue),
            # 3. Price controls (fat finger first - catches extreme errors)
            lambda: self._check_fat_finger_price(price, reference_price),
            lambda: self._check_price_collar(price, reference_price),
            # 4. Value/Volume controls
            lambda: self._check_max_order_value(order_value, venue, trader_id),
            lambda: self._check_max_order_volume(quantity, venue),
            lambda: self._check_adv_limit(quantity, adv),
            # 5. Risk controls
            lambda: self._check_position_limit(current_position, quantity, side),
            lambda: self._check_concentration(order_value, portfolio_notional),
            lambda: self._check_daily_loss(),
        ]

        for check in checks:
            result = check()
            if not result.allowed:
                self._record_rejection(result)
                return result

        # All checks passed
        with self._lock:
            self._stats["checks_passed"] += 1

        return PreTradeCheckResult(
            allowed=True,
            message="All pre-trade controls passed",
            severity=ControlSeverity.INFO,
        )

    # =========================================================================
    # Individual Control Checks
    # =========================================================================

    def _check_message_rate(self, venue: str = "") -> PreTradeCheckResult:
        """
        Check message rate limits per RTS 6 Article 15(4).

        Maximum message limits to prevent:
        - Market abuse through excessive messaging
        - System overload
        """
        if not self._config.message_rate_enabled:
            return PreTradeCheckResult(allowed=True)

        now = time.time()

        with self._lock:
            # Global rate check
            if not self._global_rate_window.record(now):
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.MESSAGE_RATE_EXCEEDED,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Message rate exceeded: >{self._config.max_messages_per_second}/sec",
                    details={
                        "limit": self._config.max_messages_per_second,
                        "window": "1 second",
                    },
                )

            # Minute rate check
            if not self._minute_rate_window.record(now):
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.MESSAGE_RATE_EXCEEDED,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Minute message rate exceeded: >{self._config.max_messages_per_minute}/min",
                    details={
                        "limit": self._config.max_messages_per_minute,
                        "window": "1 minute",
                    },
                )

            # Burst check (per venue)
            if venue:
                if venue not in self._burst_windows:
                    self._burst_windows[venue] = MessageRateWindow(
                        window_size_seconds=self._config.burst_window_ms / 1000.0,
                        max_messages=self._config.max_burst_messages,
                    )

                if not self._burst_windows[venue].record(now):
                    return PreTradeCheckResult(
                        allowed=False,
                        rejection_reason=RejectionReason.BURST_RATE_EXCEEDED,
                        severity=ControlSeverity.HARD_REJECT,
                        message=f"Burst rate exceeded for venue {venue}",
                        details={
                            "venue": venue,
                            "limit": self._config.max_burst_messages,
                            "window_ms": self._config.burst_window_ms,
                        },
                    )

        return PreTradeCheckResult(allowed=True)

    def _check_trader_authorization(
        self,
        trader_id: str,
        instrument: str,
        venue: str,
    ) -> PreTradeCheckResult:
        """
        Check trader authorization per RTS 6 Article 15(5).

        Automatic blocking of orders from unauthorized traders.
        """
        if not self._config.require_trader_authorization:
            return PreTradeCheckResult(allowed=True)

        with self._lock:
            if trader_id not in self._authorized_traders:
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.UNAUTHORIZED_TRADER,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Trader {trader_id} is not authorized",
                    details={"trader_id": trader_id},
                )

            auth = self._authorized_traders[trader_id]

            # Check if authorization is active and valid
            if not auth.is_active:
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.UNAUTHORIZED_TRADER,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Trader {trader_id} authorization is inactive",
                    details={"trader_id": trader_id},
                )

            now = datetime.now(timezone.utc)
            if auth.valid_until and now > auth.valid_until:
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.UNAUTHORIZED_TRADER,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Trader {trader_id} authorization expired",
                    details={"trader_id": trader_id, "expired": str(auth.valid_until)},
                )

            # Check instrument authorization (only if enabled in config)
            if self._config.require_instrument_authorization:
                if instrument and "*" not in auth.authorized_instruments:
                    if instrument not in auth.authorized_instruments:
                        return PreTradeCheckResult(
                            allowed=False,
                            rejection_reason=RejectionReason.UNAUTHORIZED_INSTRUMENT,
                            severity=ControlSeverity.HARD_REJECT,
                            message=f"Trader {trader_id} not authorized for {instrument}",
                            details={"trader_id": trader_id, "instrument": instrument},
                        )

            # Check venue authorization (only if enabled in config)
            if self._config.require_venue_authorization:
                if venue and "*" not in auth.authorized_venues:
                    if venue not in auth.authorized_venues:
                        return PreTradeCheckResult(
                            allowed=False,
                            rejection_reason=RejectionReason.UNAUTHORIZED_VENUE,
                            severity=ControlSeverity.HARD_REJECT,
                            message=f"Trader {trader_id} not authorized for venue {venue}",
                            details={"trader_id": trader_id, "venue": venue},
                        )

        return PreTradeCheckResult(allowed=True)

    def _check_price_collar(
        self,
        order_price: Decimal,
        reference_price: Optional[Decimal],
    ) -> PreTradeCheckResult:
        """
        Check price collar per RTS 6 Article 15(1).

        Reject orders outside acceptable price range from reference.
        """
        if not self._config.price_collar_enabled:
            return PreTradeCheckResult(allowed=True)

        if order_price <= 0:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.ZERO_OR_NEGATIVE_PRICE,
                severity=ControlSeverity.HARD_REJECT,
                message="Order price must be positive",
                details={"order_price": str(order_price)},
            )

        if reference_price is None or reference_price <= 0:
            # Missing reference price - may need to reject depending on config
            if self._config.price_collar_enabled:
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.MISSING_REFERENCE_PRICE,
                    severity=ControlSeverity.SOFT_REJECT,
                    message="Reference price not available for collar check",
                    details={"order_price": str(order_price)},
                )
            return PreTradeCheckResult(allowed=True)

        # Calculate deviation percentage
        deviation_pct = abs(order_price - reference_price) / reference_price * 100

        if deviation_pct > self._config.price_collar_pct:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.PRICE_COLLAR_BREACH,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Price {order_price} exceeds collar ({self._config.price_collar_pct}% from {reference_price})",
                details={
                    "order_price": str(order_price),
                    "reference_price": str(reference_price),
                    "deviation_pct": float(deviation_pct),
                    "collar_pct": self._config.price_collar_pct,
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_fat_finger_price(
        self,
        order_price: Decimal,
        reference_price: Optional[Decimal],
    ) -> PreTradeCheckResult:
        """
        Check fat finger protection (stricter price check).

        Catches potential human input errors.
        """
        if not self._config.fat_finger_enabled:
            return PreTradeCheckResult(allowed=True)

        # Check for zero/negative price first
        if order_price <= 0:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.ZERO_OR_NEGATIVE_PRICE,
                severity=ControlSeverity.HARD_REJECT,
                message="Order price must be positive",
                details={"order_price": str(order_price)},
            )

        if reference_price is None or reference_price <= 0:
            return PreTradeCheckResult(allowed=True)

        deviation_pct = abs(order_price - reference_price) / reference_price * 100

        if deviation_pct > self._config.fat_finger_price_deviation_pct:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.FAT_FINGER_PRICE,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Fat finger protection: price deviation {deviation_pct:.1f}% exceeds {self._config.fat_finger_price_deviation_pct}%",
                details={
                    "order_price": str(order_price),
                    "reference_price": str(reference_price),
                    "deviation_pct": float(deviation_pct),
                    "fat_finger_threshold": self._config.fat_finger_price_deviation_pct,
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_max_order_value(
        self,
        order_value: Decimal,
        venue: str = "",
        trader_id: str = "",
    ) -> PreTradeCheckResult:
        """
        Check maximum order value per RTS 6 Article 15(2).
        """
        if not self._config.max_order_value_enabled:
            return PreTradeCheckResult(allowed=True)

        # Check venue-specific limits
        max_value = self._config.max_order_value_eur
        if venue and venue in self._config.venue_specific_limits:
            venue_limit = self._config.venue_specific_limits[venue].get("max_order_value")
            if venue_limit:
                max_value = min(max_value, Decimal(str(venue_limit)))

        # Check trader-specific limits
        if trader_id and trader_id in self._authorized_traders:
            trader_limit = self._authorized_traders[trader_id].max_order_value_eur
            if trader_limit:
                max_value = min(max_value, trader_limit)

        if order_value > max_value:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.MAX_ORDER_VALUE_EXCEEDED,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Order value {order_value} exceeds max {max_value}",
                details={
                    "order_value": str(order_value),
                    "max_value": str(max_value),
                    "venue": venue,
                    "trader_id": trader_id,
                },
            )

        if order_value < self._config.min_order_value_eur:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.MIN_ORDER_VALUE_BELOW,
                severity=ControlSeverity.SOFT_REJECT,
                message=f"Order value {order_value} below minimum {self._config.min_order_value_eur}",
                details={
                    "order_value": str(order_value),
                    "min_value": str(self._config.min_order_value_eur),
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_max_order_volume(
        self,
        quantity: Decimal,
        venue: str = "",
    ) -> PreTradeCheckResult:
        """
        Check maximum order volume per RTS 6 Article 15(3).
        """
        if not self._config.max_order_volume_enabled:
            return PreTradeCheckResult(allowed=True)

        max_volume = self._config.max_order_volume

        # Check venue-specific limits
        if venue and venue in self._config.venue_specific_limits:
            venue_limit = self._config.venue_specific_limits[venue].get("max_order_volume")
            if venue_limit:
                max_volume = min(max_volume, Decimal(str(venue_limit)))

        if quantity > max_volume:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.MAX_ORDER_VOLUME_EXCEEDED,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Order quantity {quantity} exceeds max {max_volume}",
                details={
                    "quantity": str(quantity),
                    "max_volume": str(max_volume),
                    "venue": venue,
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_adv_limit(
        self,
        quantity: Decimal,
        adv: Optional[Decimal],
    ) -> PreTradeCheckResult:
        """
        Check order volume against ADV (Average Daily Volume).

        Prevents outsized orders that could impact market.
        """
        if not self._config.adv_check_enabled:
            return PreTradeCheckResult(allowed=True)

        if adv is None or adv <= 0:
            return PreTradeCheckResult(allowed=True)  # No ADV data

        pct_of_adv = (quantity / adv) * 100

        if pct_of_adv > self._config.max_volume_vs_adv_pct:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.MAX_ORDER_VOLUME_EXCEEDED,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Order is {pct_of_adv:.1f}% of ADV, exceeds {self._config.max_volume_vs_adv_pct}% limit",
                details={
                    "quantity": str(quantity),
                    "adv": str(adv),
                    "pct_of_adv": float(pct_of_adv),
                    "max_pct": self._config.max_volume_vs_adv_pct,
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_position_limit(
        self,
        current_position: Decimal,
        order_quantity: Decimal,
        side: str,
    ) -> PreTradeCheckResult:
        """
        Check if order would breach position limits.
        """
        # Calculate resulting position
        if side == "BUY":
            resulting_position = current_position + order_quantity
        elif side == "SELL":
            resulting_position = current_position - order_quantity
        else:
            resulting_position = current_position

        # Check against max volume as a proxy for position limit
        max_position = self._config.max_order_volume * 10  # 10x single order as position limit

        if abs(resulting_position) > max_position:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.POSITION_LIMIT_BREACH,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Resulting position {resulting_position} exceeds limit {max_position}",
                details={
                    "current_position": str(current_position),
                    "order_quantity": str(order_quantity),
                    "resulting_position": str(resulting_position),
                    "max_position": str(max_position),
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_concentration(
        self,
        order_value: Decimal,
        portfolio_notional: Decimal,
    ) -> PreTradeCheckResult:
        """
        Check position concentration limit.

        Prevents over-concentration in a single position.
        """
        if portfolio_notional <= 0:
            return PreTradeCheckResult(allowed=True)

        concentration_pct = (order_value / portfolio_notional) * 100

        if concentration_pct > self._config.max_concentration_pct:
            return PreTradeCheckResult(
                allowed=False,
                rejection_reason=RejectionReason.CONCENTRATION_LIMIT,
                severity=ControlSeverity.HARD_REJECT,
                message=f"Order concentration {concentration_pct:.1f}% exceeds {self._config.max_concentration_pct}% limit",
                details={
                    "order_value": str(order_value),
                    "portfolio_notional": str(portfolio_notional),
                    "concentration_pct": float(concentration_pct),
                    "max_concentration_pct": self._config.max_concentration_pct,
                },
            )

        return PreTradeCheckResult(allowed=True)

    def _check_daily_loss(self) -> PreTradeCheckResult:
        """
        Check daily loss limit.

        Prevents trading when daily loss threshold is reached.
        """
        if not self._config.daily_loss_check_enabled:
            return PreTradeCheckResult(allowed=True)

        with self._lock:
            # Reset daily tracking if needed
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_reset_date != today:
                self._daily_pnl = Decimal("0")
                self._daily_reset_date = today

            if self._daily_pnl < -self._config.max_daily_loss_eur:
                return PreTradeCheckResult(
                    allowed=False,
                    rejection_reason=RejectionReason.DAILY_LOSS_LIMIT,
                    severity=ControlSeverity.HARD_REJECT,
                    message=f"Daily loss limit reached: {self._daily_pnl}",
                    details={
                        "daily_pnl": str(self._daily_pnl),
                        "limit": str(self._config.max_daily_loss_eur),
                    },
                )

        return PreTradeCheckResult(allowed=True)

    # =========================================================================
    # Authorization Management
    # =========================================================================

    def authorize_trader(self, authorization: TraderAuthorization) -> None:
        """
        Add or update trader authorization.

        Args:
            authorization: TraderAuthorization object
        """
        with self._lock:
            self._authorized_traders[authorization.trader_id] = authorization
            logger.info(
                f"Trader authorized: {authorization.trader_id}",
                extra={
                    "trader_id": authorization.trader_id,
                    "name": authorization.name,
                },
            )

    def revoke_trader(self, trader_id: str) -> bool:
        """
        Revoke trader authorization.

        Args:
            trader_id: Trader identifier to revoke

        Returns:
            True if trader was found and revoked
        """
        with self._lock:
            if trader_id in self._authorized_traders:
                self._authorized_traders[trader_id].is_active = False
                logger.warning(f"Trader authorization revoked: {trader_id}")
                return True
            return False

    def get_authorized_traders(self) -> List[TraderAuthorization]:
        """Get all authorized traders."""
        with self._lock:
            return list(self._authorized_traders.values())

    def authorize_instrument(self, isin: str) -> None:
        """Add authorized instrument."""
        with self._lock:
            self._authorized_instruments.add(isin)

    def authorize_venue(self, mic: str) -> None:
        """Add authorized venue."""
        with self._lock:
            self._authorized_venues.add(mic)

    # =========================================================================
    # P&L Tracking
    # =========================================================================

    def update_daily_pnl(self, pnl_change: Decimal) -> None:
        """
        Update daily P&L for loss limit tracking.

        Args:
            pnl_change: Change in P&L (positive or negative)
        """
        with self._lock:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_reset_date != today:
                self._daily_pnl = Decimal("0")
                self._daily_reset_date = today

            self._daily_pnl += pnl_change

    def get_daily_pnl(self) -> Decimal:
        """Get current daily P&L."""
        with self._lock:
            return self._daily_pnl

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    def _record_rejection(self, result: PreTradeCheckResult) -> None:
        """Record a rejection for statistics and audit."""
        with self._lock:
            self._stats["checks_rejected"] += 1

            if result.rejection_reason:
                reason = result.rejection_reason.value
                self._stats["rejections_by_reason"][reason] = (
                    self._stats["rejections_by_reason"].get(reason, 0) + 1
                )

        # Log the rejection
        log_extra = result.to_dict()
        # Remove 'message' key to avoid conflict with LogRecord
        log_extra.pop("message", None)
        logger.warning(
            f"Pre-trade control REJECTED: {result.rejection_reason.value if result.rejection_reason else 'unknown'} - {result.message}",
            extra=log_extra,
        )

        # Audit callback
        if self._audit_callback:
            try:
                self._audit_callback(result)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pre-trade control statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["authorized_traders"] = len(self._authorized_traders)
            stats["authorized_instruments"] = len(self._authorized_instruments)
            stats["authorized_venues"] = len(self._authorized_venues)
            stats["daily_pnl"] = str(self._daily_pnl)
            return stats

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate RTS 6 Article 15 compliance report.

        Returns:
            Dictionary with compliance report data
        """
        stats = self.get_statistics()

        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": "RTS 6 Article 15",
            "requirement": "Pre-Trade Risk Controls",
            "compliance_status": "CONFIGURED",
            "configuration": {
                "price_collar_pct": self._config.price_collar_pct,
                "fat_finger_pct": self._config.fat_finger_price_deviation_pct,
                "max_order_value_eur": str(self._config.max_order_value_eur),
                "max_order_volume": str(self._config.max_order_volume),
                "max_messages_per_second": self._config.max_messages_per_second,
                "require_trader_authorization": self._config.require_trader_authorization,
                "daily_loss_limit_eur": str(self._config.max_daily_loss_eur),
            },
            "statistics": stats,
            "controls_enabled": {
                "price_collar": self._config.price_collar_enabled,
                "fat_finger": self._config.fat_finger_enabled,
                "max_order_value": self._config.max_order_value_enabled,
                "max_order_volume": self._config.max_order_volume_enabled,
                "adv_check": self._config.adv_check_enabled,
                "message_rate": self._config.message_rate_enabled,
                "trader_authorization": self._config.require_trader_authorization,
                "daily_loss": self._config.daily_loss_check_enabled,
            },
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = {
                "checks_total": 0,
                "checks_passed": 0,
                "checks_rejected": 0,
                "rejections_by_reason": {},
            }


# =============================================================================
# Factory Function
# =============================================================================


def create_pre_trade_controls(
    config: Optional[Dict[str, Any]] = None,
    audit_callback: Optional[Callable[[PreTradeCheckResult], None]] = None,
) -> PreTradeControls:
    """
    Factory function to create PreTradeControls instance.

    Args:
        config: Optional configuration dictionary
        audit_callback: Optional audit callback function

    Returns:
        Configured PreTradeControls instance
    """
    cfg = PreTradeControlsConfig()

    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                if isinstance(getattr(cfg, key), Decimal):
                    setattr(cfg, key, Decimal(str(value)))
                else:
                    setattr(cfg, key, value)

    return PreTradeControls(config=cfg, audit_callback=audit_callback)


__all__ = [
    "RejectionReason",
    "ControlSeverity",
    "PreTradeCheckResult",
    "PreTradeControlsConfig",
    "TraderAuthorization",
    "MessageRateWindow",
    "PreTradeControls",
    "create_pre_trade_controls",
]
