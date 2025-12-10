# -*- coding: utf-8 -*-
"""
MiFID II Order-to-Trade Ratio (OTR) Monitoring.

Implements OTR monitoring per Commission Delegated Regulation (EU) 2017/589
(RTS 6) and RTS 9 requirements:

    Order-to-Trade Ratio (OTR) = Number of Orders / Number of Trades

High OTR can indicate:
    - Market manipulation (quote stuffing, spoofing)
    - Inefficient order management
    - System malfunction

Regulatory Requirements:
    - RTS 6: Investment firms must monitor OTR
    - RTS 9: Trading venues must implement OTR limits
    - Venues may charge fees or throttle based on OTR

References:
    - RTS 6 Article 13: https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-3/
    - RTS 9: Ratio of unexecuted orders to transactions
    - ESMA Guidelines on algorithmic trading
"""

from __future__ import annotations

import logging
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Any,
    Tuple,
    Callable,
    Deque,
    NamedTuple,
)

logger = logging.getLogger(__name__)


class OrderEvent(str, Enum):
    """Types of order events tracked for OTR calculation."""

    NEW_ORDER = "new_order"                 # New order submitted
    MODIFY_ORDER = "modify_order"           # Order modification
    CANCEL_ORDER = "cancel_order"           # Order cancellation
    PARTIAL_FILL = "partial_fill"           # Partial execution
    FULL_FILL = "full_fill"                 # Full execution
    REJECT = "reject"                       # Order rejection


class OTRBucket(NamedTuple):
    """Time bucket for OTR calculation."""

    timestamp: float
    orders: int
    trades: int
    cancellations: int
    modifications: int
    rejections: int


class OTRLevel(str, Enum):
    """OTR threshold levels."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


@dataclass
class OTRMetrics:
    """
    Order-to-Trade Ratio metrics snapshot.

    Provides comprehensive OTR metrics for regulatory reporting.
    """

    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Core OTR metrics
    otr_current: float = 0.0                # Current OTR
    otr_rolling_1min: float = 0.0           # 1-minute rolling OTR
    otr_rolling_5min: float = 0.0           # 5-minute rolling OTR
    otr_rolling_1hour: float = 0.0          # 1-hour rolling OTR
    otr_daily: float = 0.0                  # Daily OTR

    # Raw counts
    orders_total: int = 0                   # Total orders
    trades_total: int = 0                   # Total trades (fills)
    cancellations_total: int = 0            # Total cancellations
    modifications_total: int = 0            # Total modifications
    rejections_total: int = 0               # Total rejections

    # Rates
    orders_per_second: float = 0.0
    trades_per_second: float = 0.0
    cancellation_rate_pct: float = 0.0      # Cancellations / Orders %

    # Level and status
    level: OTRLevel = OTRLevel.NORMAL
    is_within_limits: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "otr_current": round(self.otr_current, 2),
            "otr_rolling_1min": round(self.otr_rolling_1min, 2),
            "otr_rolling_5min": round(self.otr_rolling_5min, 2),
            "otr_rolling_1hour": round(self.otr_rolling_1hour, 2),
            "otr_daily": round(self.otr_daily, 2),
            "orders_total": self.orders_total,
            "trades_total": self.trades_total,
            "cancellations_total": self.cancellations_total,
            "modifications_total": self.modifications_total,
            "rejections_total": self.rejections_total,
            "orders_per_second": round(self.orders_per_second, 2),
            "trades_per_second": round(self.trades_per_second, 2),
            "cancellation_rate_pct": round(self.cancellation_rate_pct, 2),
            "level": self.level.value,
            "is_within_limits": self.is_within_limits,
        }


@dataclass
class OTRBreachEvent:
    """Record of an OTR limit breach."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    otr_value: float = 0.0
    otr_limit: float = 0.0
    level: OTRLevel = OTRLevel.BREACH
    window_seconds: int = 0

    venue: Optional[str] = None
    algorithm_id: Optional[str] = None
    instrument: Optional[str] = None

    orders_count: int = 0
    trades_count: int = 0

    action_taken: str = ""
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "otr_value": self.otr_value,
            "otr_limit": self.otr_limit,
            "level": self.level.value,
            "window_seconds": self.window_seconds,
            "venue": self.venue,
            "algorithm_id": self.algorithm_id,
            "instrument": self.instrument,
            "orders_count": self.orders_count,
            "trades_count": self.trades_count,
            "action_taken": self.action_taken,
            "resolved": self.resolved,
        }


@dataclass
class OTRMonitorConfig:
    """Configuration for OTR monitoring."""

    # OTR thresholds (orders per trade)
    otr_normal_max: float = 10.0            # Normal: up to 10:1
    otr_elevated_threshold: float = 20.0    # Elevated: 20:1
    otr_warning_threshold: float = 50.0     # Warning: 50:1
    otr_critical_threshold: float = 100.0   # Critical: 100:1
    otr_breach_threshold: float = 200.0     # Breach: 200:1 (may trigger kill switch)

    # Venue-specific thresholds
    venue_otr_limits: Dict[str, float] = field(default_factory=dict)

    # Monitoring windows
    bucket_size_seconds: int = 1            # Granularity of tracking
    window_1min: int = 60                   # 1-minute window
    window_5min: int = 300                  # 5-minute window
    window_1hour: int = 3600                # 1-hour window

    # Cancellation rate thresholds
    cancellation_warning_pct: float = 50.0  # Warning if >50% cancelled
    cancellation_critical_pct: float = 80.0 # Critical if >80% cancelled

    # Response actions
    throttle_on_warning: bool = True
    throttle_delay_ms: int = 100            # Delay per order when throttling
    block_on_critical: bool = True
    trigger_kill_switch_on_breach: bool = True

    # Logging and persistence
    log_all_events: bool = False
    persist_breach_events: bool = True
    breach_log_path: str = "state/compliance/otr_breaches.jsonl"


@dataclass
class PerVenueOTR:
    """Per-venue OTR tracking."""

    venue_mic: str
    orders: int = 0
    trades: int = 0
    cancellations: int = 0
    modifications: int = 0
    last_update: float = 0.0

    @property
    def otr(self) -> float:
        """Calculate current OTR."""
        if self.trades == 0:
            return float(self.orders) if self.orders > 0 else 0.0
        return self.orders / self.trades


@dataclass
class PerAlgorithmOTR:
    """Per-algorithm OTR tracking."""

    algorithm_id: str
    orders: int = 0
    trades: int = 0
    cancellations: int = 0
    modifications: int = 0
    last_update: float = 0.0

    @property
    def otr(self) -> float:
        """Calculate current OTR."""
        if self.trades == 0:
            return float(self.orders) if self.orders > 0 else 0.0
        return self.orders / self.trades


class OTRMonitor:
    """
    Order-to-Trade Ratio Monitor per MiFID II RTS 6/9.

    Provides:
        - Real-time OTR calculation across multiple time windows
        - Per-venue and per-algorithm OTR tracking
        - Automatic throttling and blocking on threshold breach
        - Integration with kill switch for severe breaches
        - Comprehensive audit trail

    Usage:
        monitor = OTRMonitor(
            config=OTRMonitorConfig(),
            breach_callback=handle_breach,
        )

        # Record events
        monitor.record_order(venue="XLON", algorithm_id="ALGO001")
        monitor.record_trade(venue="XLON", algorithm_id="ALGO001")
        monitor.record_cancellation(venue="XLON")

        # Get metrics
        metrics = monitor.get_metrics()
        print(f"Current OTR: {metrics.otr_current}")

        # Check if trading allowed
        can_trade, reason = monitor.can_submit_order(venue="XLON")

    Regulatory Compliance:
        - RTS 6: Real-time monitoring requirement
        - RTS 9: OTR limits per venue
        - Automatic response to threshold breaches
    """

    def __init__(
        self,
        config: Optional[OTRMonitorConfig] = None,
        breach_callback: Optional[Callable[[OTRBreachEvent], None]] = None,
        throttle_callback: Optional[Callable[[str, int], None]] = None,
        kill_switch_callback: Optional[Callable[[OTRBreachEvent], None]] = None,
    ):
        """
        Initialize OTR Monitor.

        Args:
            config: OTR monitoring configuration
            breach_callback: Called on OTR threshold breach
            throttle_callback: Called when throttling is applied (venue, delay_ms)
            kill_switch_callback: Called for severe breaches
        """
        self._config = config or OTRMonitorConfig()
        self._breach_callback = breach_callback
        self._throttle_callback = throttle_callback
        self._kill_switch_callback = kill_switch_callback

        # State
        self._lock = threading.RLock()

        # Time-series buckets for rolling calculations
        self._buckets: Deque[OTRBucket] = deque(
            maxlen=self._config.window_1hour // self._config.bucket_size_seconds + 1
        )

        # Per-venue tracking
        self._venue_otr: Dict[str, PerVenueOTR] = {}

        # Per-algorithm tracking
        self._algo_otr: Dict[str, PerAlgorithmOTR] = {}

        # Daily totals (reset at midnight)
        self._daily_orders: int = 0
        self._daily_trades: int = 0
        self._daily_cancellations: int = 0
        self._daily_modifications: int = 0
        self._daily_reset_date: str = ""

        # Breach tracking
        self._breaches: List[OTRBreachEvent] = []
        self._last_breach_time: float = 0.0
        self._is_throttled: bool = False
        self._is_blocked: bool = False

        # Statistics
        self._stats = {
            "total_orders": 0,
            "total_trades": 0,
            "total_cancellations": 0,
            "total_modifications": 0,
            "total_rejections": 0,
            "total_breaches": 0,
            "throttle_events": 0,
            "block_events": 0,
        }

        logger.info(
            "OTRMonitor initialized",
            extra={
                "warning_threshold": self._config.otr_warning_threshold,
                "critical_threshold": self._config.otr_critical_threshold,
                "breach_threshold": self._config.otr_breach_threshold,
            },
        )

    # =========================================================================
    # Event Recording
    # =========================================================================

    def record_order(
        self,
        venue: str = "",
        algorithm_id: str = "",
        instrument: str = "",
        quantity: float = 0.0,
        is_modification: bool = False,
    ) -> None:
        """
        Record a new order or modification.

        Args:
            venue: Trading venue MIC code
            algorithm_id: Algorithm identifier
            instrument: Instrument ISIN
            quantity: Order quantity
            is_modification: True if this is an order modification
        """
        now = time.time()

        with self._lock:
            self._ensure_daily_reset()
            bucket = self._get_or_create_bucket(now)

            if is_modification:
                self._daily_modifications += 1
                self._stats["total_modifications"] += 1
                # Track modification in bucket
                new_bucket = OTRBucket(
                    timestamp=bucket.timestamp,
                    orders=bucket.orders,
                    trades=bucket.trades,
                    cancellations=bucket.cancellations,
                    modifications=bucket.modifications + 1,
                    rejections=bucket.rejections,
                )
            else:
                self._daily_orders += 1
                self._stats["total_orders"] += 1
                new_bucket = OTRBucket(
                    timestamp=bucket.timestamp,
                    orders=bucket.orders + 1,
                    trades=bucket.trades,
                    cancellations=bucket.cancellations,
                    modifications=bucket.modifications,
                    rejections=bucket.rejections,
                )

            # Update bucket
            self._update_bucket(new_bucket)

            # Update per-venue
            if venue:
                if venue not in self._venue_otr:
                    self._venue_otr[venue] = PerVenueOTR(venue_mic=venue)
                self._venue_otr[venue].orders += 1
                if is_modification:
                    self._venue_otr[venue].modifications += 1
                self._venue_otr[venue].last_update = now

            # Update per-algorithm
            if algorithm_id:
                if algorithm_id not in self._algo_otr:
                    self._algo_otr[algorithm_id] = PerAlgorithmOTR(algorithm_id=algorithm_id)
                self._algo_otr[algorithm_id].orders += 1
                if is_modification:
                    self._algo_otr[algorithm_id].modifications += 1
                self._algo_otr[algorithm_id].last_update = now

        # Check thresholds
        self._check_thresholds(venue=venue, algorithm_id=algorithm_id)

        if self._config.log_all_events:
            logger.debug(
                f"Order recorded: venue={venue}, algo={algorithm_id}, mod={is_modification}"
            )

    def record_trade(
        self,
        venue: str = "",
        algorithm_id: str = "",
        instrument: str = "",
        quantity: float = 0.0,
        is_partial: bool = False,
    ) -> None:
        """
        Record a trade (execution).

        Args:
            venue: Trading venue MIC code
            algorithm_id: Algorithm identifier
            instrument: Instrument ISIN
            quantity: Executed quantity
            is_partial: True if partial fill
        """
        now = time.time()

        with self._lock:
            self._ensure_daily_reset()
            bucket = self._get_or_create_bucket(now)

            self._daily_trades += 1
            self._stats["total_trades"] += 1

            new_bucket = OTRBucket(
                timestamp=bucket.timestamp,
                orders=bucket.orders,
                trades=bucket.trades + 1,
                cancellations=bucket.cancellations,
                modifications=bucket.modifications,
                rejections=bucket.rejections,
            )
            self._update_bucket(new_bucket)

            # Update per-venue
            if venue:
                if venue not in self._venue_otr:
                    self._venue_otr[venue] = PerVenueOTR(venue_mic=venue)
                self._venue_otr[venue].trades += 1
                self._venue_otr[venue].last_update = now

            # Update per-algorithm
            if algorithm_id:
                if algorithm_id not in self._algo_otr:
                    self._algo_otr[algorithm_id] = PerAlgorithmOTR(algorithm_id=algorithm_id)
                self._algo_otr[algorithm_id].trades += 1
                self._algo_otr[algorithm_id].last_update = now

        if self._config.log_all_events:
            logger.debug(f"Trade recorded: venue={venue}, algo={algorithm_id}")

    def record_cancellation(
        self,
        venue: str = "",
        algorithm_id: str = "",
        instrument: str = "",
    ) -> None:
        """
        Record an order cancellation.

        Args:
            venue: Trading venue MIC code
            algorithm_id: Algorithm identifier
            instrument: Instrument ISIN
        """
        now = time.time()

        with self._lock:
            self._ensure_daily_reset()
            bucket = self._get_or_create_bucket(now)

            self._daily_cancellations += 1
            self._stats["total_cancellations"] += 1

            new_bucket = OTRBucket(
                timestamp=bucket.timestamp,
                orders=bucket.orders,
                trades=bucket.trades,
                cancellations=bucket.cancellations + 1,
                modifications=bucket.modifications,
                rejections=bucket.rejections,
            )
            self._update_bucket(new_bucket)

            if venue:
                if venue not in self._venue_otr:
                    self._venue_otr[venue] = PerVenueOTR(venue_mic=venue)
                self._venue_otr[venue].cancellations += 1
                self._venue_otr[venue].last_update = now

            if algorithm_id:
                if algorithm_id not in self._algo_otr:
                    self._algo_otr[algorithm_id] = PerAlgorithmOTR(algorithm_id=algorithm_id)
                self._algo_otr[algorithm_id].cancellations += 1
                self._algo_otr[algorithm_id].last_update = now

        if self._config.log_all_events:
            logger.debug(f"Cancellation recorded: venue={venue}, algo={algorithm_id}")

    def record_rejection(
        self,
        venue: str = "",
        algorithm_id: str = "",
        reason: str = "",
    ) -> None:
        """
        Record an order rejection.

        Args:
            venue: Trading venue MIC code
            algorithm_id: Algorithm identifier
            reason: Rejection reason
        """
        now = time.time()

        with self._lock:
            bucket = self._get_or_create_bucket(now)

            self._stats["total_rejections"] += 1

            new_bucket = OTRBucket(
                timestamp=bucket.timestamp,
                orders=bucket.orders,
                trades=bucket.trades,
                cancellations=bucket.cancellations,
                modifications=bucket.modifications,
                rejections=bucket.rejections + 1,
            )
            self._update_bucket(new_bucket)

        if self._config.log_all_events:
            logger.debug(f"Rejection recorded: venue={venue}, reason={reason}")

    # =========================================================================
    # Trading Permission Checks
    # =========================================================================

    def can_submit_order(
        self,
        venue: str = "",
        algorithm_id: str = "",
    ) -> Tuple[bool, str]:
        """
        Check if order submission is allowed based on OTR status.

        Args:
            venue: Trading venue
            algorithm_id: Algorithm identifier

        Returns:
            (can_submit, reason) tuple
        """
        with self._lock:
            # Check if blocked
            if self._is_blocked:
                return False, "Order submission blocked due to OTR breach"

            # Check current OTR level
            metrics = self._calculate_metrics()

            if metrics.level == OTRLevel.BREACH:
                return False, f"OTR breach: {metrics.otr_current:.1f} exceeds {self._config.otr_breach_threshold}"

            if metrics.level == OTRLevel.CRITICAL and self._config.block_on_critical:
                return False, f"OTR critical: {metrics.otr_current:.1f} exceeds {self._config.otr_critical_threshold}"

            # Check venue-specific limits
            if venue and venue in self._config.venue_otr_limits:
                venue_limit = self._config.venue_otr_limits[venue]
                venue_otr = self.get_venue_otr(venue)
                if venue_otr > venue_limit:
                    return False, f"Venue {venue} OTR {venue_otr:.1f} exceeds limit {venue_limit}"

            # Throttling warning
            if metrics.level == OTRLevel.WARNING and self._config.throttle_on_warning:
                # Allow but may apply delay
                return True, f"OTR elevated ({metrics.otr_current:.1f}), throttling may apply"

        return True, "OTR within limits"

    def get_throttle_delay(self, venue: str = "") -> int:
        """
        Get throttle delay in milliseconds for current OTR level.

        Args:
            venue: Trading venue

        Returns:
            Delay in milliseconds (0 if no throttling)
        """
        if not self._config.throttle_on_warning:
            return 0

        with self._lock:
            metrics = self._calculate_metrics()

            if metrics.level == OTRLevel.WARNING:
                return self._config.throttle_delay_ms
            elif metrics.level == OTRLevel.CRITICAL:
                return self._config.throttle_delay_ms * 2
            elif metrics.level == OTRLevel.ELEVATED:
                return self._config.throttle_delay_ms // 2

        return 0

    # =========================================================================
    # Metrics Calculation
    # =========================================================================

    def get_metrics(self) -> OTRMetrics:
        """Get current OTR metrics."""
        with self._lock:
            return self._calculate_metrics()

    def _calculate_metrics(self) -> OTRMetrics:
        """Calculate current OTR metrics (internal, must hold lock)."""
        now = time.time()

        # Calculate rolling OTRs
        orders_1min = 0
        trades_1min = 0
        orders_5min = 0
        trades_5min = 0
        orders_1hour = 0
        trades_1hour = 0

        cutoff_1min = now - self._config.window_1min
        cutoff_5min = now - self._config.window_5min
        cutoff_1hour = now - self._config.window_1hour

        for bucket in self._buckets:
            if bucket.timestamp >= cutoff_1min:
                orders_1min += bucket.orders
                trades_1min += bucket.trades
            if bucket.timestamp >= cutoff_5min:
                orders_5min += bucket.orders
                trades_5min += bucket.trades
            if bucket.timestamp >= cutoff_1hour:
                orders_1hour += bucket.orders
                trades_1hour += bucket.trades

        # Calculate OTRs
        otr_1min = orders_1min / max(trades_1min, 1)
        otr_5min = orders_5min / max(trades_5min, 1)
        otr_1hour = orders_1hour / max(trades_1hour, 1)
        otr_daily = self._daily_orders / max(self._daily_trades, 1)

        # Use 1-minute OTR as current
        otr_current = otr_1min

        # Calculate rates
        orders_per_second = orders_1min / self._config.window_1min if orders_1min > 0 else 0
        trades_per_second = trades_1min / self._config.window_1min if trades_1min > 0 else 0

        # Cancellation rate
        cancellation_rate = (
            self._daily_cancellations / max(self._daily_orders, 1) * 100
        )

        # Determine level
        level = self._determine_level(otr_current)

        return OTRMetrics(
            otr_current=otr_current,
            otr_rolling_1min=otr_1min,
            otr_rolling_5min=otr_5min,
            otr_rolling_1hour=otr_1hour,
            otr_daily=otr_daily,
            orders_total=self._daily_orders,
            trades_total=self._daily_trades,
            cancellations_total=self._daily_cancellations,
            modifications_total=self._daily_modifications,
            rejections_total=self._stats.get("total_rejections", 0),
            orders_per_second=orders_per_second,
            trades_per_second=trades_per_second,
            cancellation_rate_pct=cancellation_rate,
            level=level,
            is_within_limits=level not in (OTRLevel.CRITICAL, OTRLevel.BREACH),
        )

    def _determine_level(self, otr: float) -> OTRLevel:
        """Determine OTR level from value."""
        if otr >= self._config.otr_breach_threshold:
            return OTRLevel.BREACH
        elif otr >= self._config.otr_critical_threshold:
            return OTRLevel.CRITICAL
        elif otr >= self._config.otr_warning_threshold:
            return OTRLevel.WARNING
        elif otr >= self._config.otr_elevated_threshold:
            return OTRLevel.ELEVATED
        else:
            return OTRLevel.NORMAL

    def get_venue_otr(self, venue: str) -> float:
        """Get OTR for a specific venue."""
        with self._lock:
            if venue in self._venue_otr:
                return self._venue_otr[venue].otr
            return 0.0

    def get_algorithm_otr(self, algorithm_id: str) -> float:
        """Get OTR for a specific algorithm."""
        with self._lock:
            if algorithm_id in self._algo_otr:
                return self._algo_otr[algorithm_id].otr
            return 0.0

    def get_all_venue_otrs(self) -> Dict[str, float]:
        """Get OTR for all tracked venues."""
        with self._lock:
            return {
                venue: data.otr
                for venue, data in self._venue_otr.items()
            }

    def get_all_algorithm_otrs(self) -> Dict[str, float]:
        """Get OTR for all tracked algorithms."""
        with self._lock:
            return {
                algo: data.otr
                for algo, data in self._algo_otr.items()
            }

    # =========================================================================
    # Threshold Checking
    # =========================================================================

    def _check_thresholds(
        self,
        venue: str = "",
        algorithm_id: str = "",
    ) -> None:
        """Check OTR thresholds and take action if needed."""
        with self._lock:
            metrics = self._calculate_metrics()

            # Check for breach
            if metrics.level == OTRLevel.BREACH:
                self._handle_breach(metrics, venue, algorithm_id)
            elif metrics.level == OTRLevel.CRITICAL:
                self._handle_critical(metrics, venue, algorithm_id)
            elif metrics.level == OTRLevel.WARNING:
                self._handle_warning(metrics, venue, algorithm_id)

            # Check cancellation rate
            if metrics.cancellation_rate_pct >= self._config.cancellation_critical_pct:
                logger.warning(
                    f"Cancellation rate critical: {metrics.cancellation_rate_pct:.1f}%"
                )
            elif metrics.cancellation_rate_pct >= self._config.cancellation_warning_pct:
                logger.info(
                    f"Cancellation rate elevated: {metrics.cancellation_rate_pct:.1f}%"
                )

    def _handle_breach(
        self,
        metrics: OTRMetrics,
        venue: str,
        algorithm_id: str,
    ) -> None:
        """Handle OTR breach."""
        breach = OTRBreachEvent(
            otr_value=metrics.otr_current,
            otr_limit=self._config.otr_breach_threshold,
            level=OTRLevel.BREACH,
            window_seconds=self._config.window_1min,
            venue=venue or None,
            algorithm_id=algorithm_id or None,
            orders_count=metrics.orders_total,
            trades_count=metrics.trades_total,
            action_taken="",
        )

        self._breaches.append(breach)
        self._stats["total_breaches"] += 1
        self._last_breach_time = time.time()

        logger.critical(
            f"OTR BREACH: {metrics.otr_current:.1f} exceeds {self._config.otr_breach_threshold}",
            extra=breach.to_dict(),
        )

        # Block further orders
        if self._config.block_on_critical:
            self._is_blocked = True
            breach.action_taken = "blocked"
            self._stats["block_events"] += 1
            logger.warning("Order submission BLOCKED due to OTR breach")

        # Trigger kill switch
        if self._config.trigger_kill_switch_on_breach:
            if self._kill_switch_callback:
                try:
                    self._kill_switch_callback(breach)
                except Exception as e:
                    logger.error(f"Kill switch callback error: {e}")

        # Notify callback
        if self._breach_callback:
            try:
                self._breach_callback(breach)
            except Exception as e:
                logger.error(f"Breach callback error: {e}")

        # Persist
        if self._config.persist_breach_events:
            self._persist_breach(breach)

    def _handle_critical(
        self,
        metrics: OTRMetrics,
        venue: str,
        algorithm_id: str,
    ) -> None:
        """Handle critical OTR level."""
        logger.error(
            f"OTR CRITICAL: {metrics.otr_current:.1f} exceeds {self._config.otr_critical_threshold}"
        )

        if self._config.block_on_critical:
            self._is_blocked = True
            self._stats["block_events"] += 1
            logger.warning("Order submission BLOCKED due to critical OTR")

    def _handle_warning(
        self,
        metrics: OTRMetrics,
        venue: str,
        algorithm_id: str,
    ) -> None:
        """Handle warning OTR level."""
        logger.warning(
            f"OTR WARNING: {metrics.otr_current:.1f} exceeds {self._config.otr_warning_threshold}"
        )

        if self._config.throttle_on_warning:
            self._is_throttled = True
            self._stats["throttle_events"] += 1

            if self._throttle_callback:
                try:
                    self._throttle_callback(venue, self._config.throttle_delay_ms)
                except Exception as e:
                    logger.error(f"Throttle callback error: {e}")

    # =========================================================================
    # State Management
    # =========================================================================

    def reset_block(self) -> None:
        """Reset blocking state."""
        with self._lock:
            self._is_blocked = False
            logger.info("OTR blocking reset")

    def reset_throttle(self) -> None:
        """Reset throttling state."""
        with self._lock:
            self._is_throttled = False
            logger.info("OTR throttling reset")

    def reset_daily(self) -> None:
        """Reset daily counters."""
        with self._lock:
            self._daily_orders = 0
            self._daily_trades = 0
            self._daily_cancellations = 0
            self._daily_modifications = 0
            self._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._venue_otr.clear()
            self._algo_otr.clear()
            self._is_blocked = False
            self._is_throttled = False
            logger.info("OTR daily counters reset")

    def _ensure_daily_reset(self) -> None:
        """Ensure daily reset has occurred."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_orders = 0
            self._daily_trades = 0
            self._daily_cancellations = 0
            self._daily_modifications = 0
            self._daily_reset_date = today

    # =========================================================================
    # Bucket Management
    # =========================================================================

    def _get_or_create_bucket(self, timestamp: float) -> OTRBucket:
        """Get or create bucket for timestamp."""
        bucket_time = int(timestamp // self._config.bucket_size_seconds) * self._config.bucket_size_seconds

        # Find existing bucket
        for bucket in self._buckets:
            if bucket.timestamp == bucket_time:
                return bucket

        # Create new bucket
        new_bucket = OTRBucket(
            timestamp=bucket_time,
            orders=0,
            trades=0,
            cancellations=0,
            modifications=0,
            rejections=0,
        )
        self._buckets.append(new_bucket)
        return new_bucket

    def _update_bucket(self, bucket: OTRBucket) -> None:
        """Update bucket in deque."""
        for i, b in enumerate(self._buckets):
            if b.timestamp == bucket.timestamp:
                self._buckets[i] = bucket
                return
        self._buckets.append(bucket)

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist_breach(self, breach: OTRBreachEvent) -> None:
        """Persist breach event to disk."""
        try:
            import os
            from pathlib import Path
            import json

            path = Path(self._config.breach_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(breach.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist OTR breach: {e}")

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_breaches(self, limit: int = 100) -> List[OTRBreachEvent]:
        """Get recent breach events."""
        with self._lock:
            return self._breaches[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get OTR monitoring statistics."""
        with self._lock:
            metrics = self._calculate_metrics()
            return {
                **self._stats,
                "current_otr": metrics.otr_current,
                "otr_level": metrics.level.value,
                "is_blocked": self._is_blocked,
                "is_throttled": self._is_throttled,
                "venues_tracked": len(self._venue_otr),
                "algorithms_tracked": len(self._algo_otr),
                "breaches_today": sum(
                    1 for b in self._breaches
                    if datetime.now(timezone.utc).strftime("%Y-%m-%d") in b.timestamp_utc
                ),
            }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate OTR compliance report for regulatory purposes.

        Returns:
            Dictionary with compliance report data
        """
        stats = self.get_statistics()
        metrics = self.get_metrics()

        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": "RTS 6 Article 13 / RTS 9",
            "requirement": "Order-to-Trade Ratio Monitoring",
            "compliance_status": "WITHIN_LIMITS" if metrics.is_within_limits else "BREACH",
            "configuration": {
                "warning_threshold": self._config.otr_warning_threshold,
                "critical_threshold": self._config.otr_critical_threshold,
                "breach_threshold": self._config.otr_breach_threshold,
                "cancellation_warning_pct": self._config.cancellation_warning_pct,
                "throttle_on_warning": self._config.throttle_on_warning,
                "block_on_critical": self._config.block_on_critical,
            },
            "current_metrics": metrics.to_dict(),
            "statistics": stats,
            "venue_otrs": self.get_all_venue_otrs(),
            "algorithm_otrs": self.get_all_algorithm_otrs(),
            "recent_breaches": [b.to_dict() for b in self.get_breaches(limit=10)],
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_otr_monitor(
    config: Optional[Dict[str, Any]] = None,
    breach_callback: Optional[Callable[[OTRBreachEvent], None]] = None,
    kill_switch_callback: Optional[Callable[[OTRBreachEvent], None]] = None,
) -> OTRMonitor:
    """
    Factory function to create OTRMonitor instance.

    Args:
        config: Optional configuration dictionary
        breach_callback: Optional breach callback
        kill_switch_callback: Optional kill switch callback

    Returns:
        Configured OTRMonitor instance
    """
    cfg = OTRMonitorConfig()

    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return OTRMonitor(
        config=cfg,
        breach_callback=breach_callback,
        kill_switch_callback=kill_switch_callback,
    )


__all__ = [
    "OrderEvent",
    "OTRBucket",
    "OTRLevel",
    "OTRMetrics",
    "OTRBreachEvent",
    "OTRMonitorConfig",
    "PerVenueOTR",
    "PerAlgorithmOTR",
    "OTRMonitor",
    "create_otr_monitor",
]
