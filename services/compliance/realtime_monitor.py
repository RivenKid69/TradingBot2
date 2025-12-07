# -*- coding: utf-8 -*-
"""
MiFID II RTS 6 Article 17 Real-Time Monitoring.

Implements real-time monitoring requirements per Commission Delegated
Regulation (EU) 2017/589 (RTS 6) Article 17:

    "Real-time alerts shall be generated within five seconds after
    the relevant event"

Key Requirements (Article 17 RTS 6):
    1. Real-time monitoring of all algorithmic trading activity
    2. Alerts generated within 5 seconds of events
    3. Monitoring of order-to-trade ratios
    4. P&L monitoring and thresholds
    5. Position limit monitoring
    6. System health monitoring

References:
    - RTS 6 Article 17: https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-4/
    - ESMA Guidelines: https://www.esma.europa.eu/sites/default/files/library/esma70-156-4572_mifid_ii_final_report_on_algorithmic_trading.pdf
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
import uuid
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
    Callable,
    Deque,
    Awaitable,
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"               # Informational
    WARNING = "warning"         # Attention needed
    CRITICAL = "critical"       # Immediate action required
    EMERGENCY = "emergency"     # Emergency - trigger kill switch


class AlertCategory(str, Enum):
    """Categories of compliance alerts per RTS 6."""

    # Order/Trade monitoring
    ORDER_TO_TRADE_RATIO = "order_to_trade_ratio"
    MESSAGE_RATE = "message_rate"
    EXECUTION_QUALITY = "execution_quality"

    # Risk monitoring
    PNL_THRESHOLD = "pnl_threshold"
    POSITION_LIMIT = "position_limit"
    RISK_LIMIT = "risk_limit"
    CONCENTRATION = "concentration"

    # Price monitoring
    PRICE_DEVIATION = "price_deviation"
    MARKET_CIRCUIT_BREAKER = "market_circuit_breaker"

    # System monitoring
    LATENCY = "latency"
    CONNECTIVITY = "connectivity"
    CLOCK_DRIFT = "clock_drift"
    SYSTEM_ERROR = "system_error"
    CAPACITY = "capacity"

    # Compliance monitoring
    KILL_SWITCH = "kill_switch"
    UNAUTHORIZED_ACTIVITY = "unauthorized_activity"
    REGULATORY = "regulatory"


@dataclass
class ComplianceAlert:
    """
    Real-time compliance alert per RTS 6 Article 17.

    Per regulation: Alerts must be generated within 5 seconds of event.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory = AlertCategory.SYSTEM_ERROR
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Context
    algorithm_id: Optional[str] = None
    venue: Optional[str] = None
    instrument: Optional[str] = None

    # State
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[int] = None
    resolved: bool = False
    resolved_at: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "algorithm_id": self.algorithm_id,
            "venue": self.venue,
            "instrument": self.instrument,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
        }


@dataclass
class MonitoringThreshold:
    """Configurable threshold for monitoring."""

    name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    unit: str = ""
    description: str = ""


@dataclass
class RealTimeMonitorConfig:
    """
    Configuration for RTS 6 Article 17 real-time monitoring.
    """

    # RTS 6 Article 17 requirement: alerts within 5 seconds
    max_alert_delay_seconds: float = 5.0

    # Monitoring intervals
    check_interval_seconds: float = 1.0
    aggregation_window_seconds: float = 60.0

    # Order-to-Trade Ratio thresholds
    otr_warning_threshold: float = 50.0       # 50:1
    otr_critical_threshold: float = 100.0     # 100:1
    otr_emergency_threshold: float = 200.0    # 200:1 - trigger kill switch

    # P&L thresholds (as % of daily limit)
    pnl_warning_pct: float = 50.0             # 50% of daily limit
    pnl_critical_pct: float = 75.0            # 75% of daily limit
    pnl_emergency_pct: float = 90.0           # 90% - near kill switch

    # Position limit thresholds (as % of max)
    position_warning_pct: float = 80.0
    position_critical_pct: float = 90.0
    position_emergency_pct: float = 95.0

    # Message rate thresholds (as % of max)
    message_rate_warning_pct: float = 70.0
    message_rate_critical_pct: float = 90.0

    # Latency thresholds (milliseconds)
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    latency_emergency_ms: float = 1000.0

    # Clock drift thresholds (milliseconds)
    clock_drift_warning_ms: float = 50.0
    clock_drift_critical_ms: float = 100.0
    clock_drift_emergency_ms: float = 1000.0

    # Price deviation (% from reference)
    price_deviation_warning_pct: float = 3.0
    price_deviation_critical_pct: float = 5.0

    # System capacity
    cpu_warning_pct: float = 70.0
    cpu_critical_pct: float = 90.0
    memory_warning_pct: float = 80.0
    memory_critical_pct: float = 95.0

    # Alert management
    max_alerts_per_category_per_minute: int = 10
    alert_deduplication_window_seconds: float = 60.0
    persist_alerts: bool = True
    alert_log_path: str = "logs/compliance/realtime_alerts.jsonl"


@dataclass
class MonitoringMetrics:
    """Current monitoring metrics snapshot."""

    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    # Order metrics
    orders_submitted: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    order_to_trade_ratio: float = 0.0

    # P&L metrics
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    daily_pnl_limit: Decimal = Decimal("100000")

    # Position metrics
    total_position_value: Decimal = Decimal("0")
    max_position_value: Decimal = Decimal("0")
    position_utilization_pct: float = 0.0

    # Rate metrics
    messages_per_second: float = 0.0
    max_messages_per_second: int = 100

    # Latency metrics
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # System metrics
    cpu_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    disk_usage_pct: float = 0.0

    # Clock metrics
    clock_offset_ms: float = 0.0
    clock_synced: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "orders_submitted": self.orders_submitted,
            "orders_cancelled": self.orders_cancelled,
            "orders_filled": self.orders_filled,
            "order_to_trade_ratio": self.order_to_trade_ratio,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "daily_pnl": str(self.daily_pnl),
            "position_utilization_pct": self.position_utilization_pct,
            "messages_per_second": self.messages_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "cpu_usage_pct": self.cpu_usage_pct,
            "memory_usage_pct": self.memory_usage_pct,
            "clock_offset_ms": self.clock_offset_ms,
        }


class RealTimeMonitor:
    """
    MiFID II RTS 6 Article 17 Real-Time Monitoring.

    Provides:
        - Real-time monitoring of all algorithmic trading activity
        - Alert generation within 5 seconds (RTS 6 requirement)
        - OTR, P&L, position, and system monitoring
        - Escalation to kill switch for emergency alerts
        - Full audit trail of all alerts

    Usage:
        monitor = RealTimeMonitor(
            config=RealTimeMonitorConfig(),
            alert_callback=handle_alert,
            kill_switch_callback=trigger_kill_switch,
        )

        # Start monitoring
        await monitor.start()

        # Update metrics
        monitor.update_order_metrics(submitted=10, cancelled=2, filled=8)
        monitor.update_pnl(daily_pnl=Decimal("-5000"))

        # Manual alert
        monitor.generate_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK_LIMIT,
            message="Risk limit approaching",
        )

        # Stop monitoring
        await monitor.stop()

    RTS 6 Article 17 Compliance:
        - Alert deadline: 5 seconds (configurable, default compliant)
        - All metric categories monitored
        - Automatic escalation for emergency events
        - Full audit trail
    """

    ALERT_DEADLINE_SEC = 5.0  # RTS 6 Article 17 requirement

    def __init__(
        self,
        config: Optional[RealTimeMonitorConfig] = None,
        alert_callback: Optional[Callable[[ComplianceAlert], None]] = None,
        async_alert_callback: Optional[Callable[[ComplianceAlert], Awaitable[None]]] = None,
        kill_switch_callback: Optional[Callable[[ComplianceAlert], None]] = None,
        escalation_callback: Optional[Callable[[ComplianceAlert], None]] = None,
    ):
        """
        Initialize Real-Time Monitor.

        Args:
            config: Monitoring configuration
            alert_callback: Sync callback for each alert
            async_alert_callback: Async callback for each alert
            kill_switch_callback: Called for emergency alerts
            escalation_callback: Called for critical+ alerts
        """
        self._config = config or RealTimeMonitorConfig()
        self._alert_callback = alert_callback
        self._async_alert_callback = async_alert_callback
        self._kill_switch_callback = kill_switch_callback
        self._escalation_callback = escalation_callback

        # State
        self._lock = threading.RLock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics = MonitoringMetrics()
        self._metrics_history: Deque[MonitoringMetrics] = deque(maxlen=1000)

        # Alerts
        self._alerts: List[ComplianceAlert] = []
        self._alert_counts: Dict[str, Deque[float]] = {}  # For rate limiting
        self._alert_hashes: Deque[Tuple[str, float]] = deque(maxlen=1000)  # For dedup

        # Order tracking for OTR
        self._order_timestamps: Deque[Tuple[float, str]] = deque(maxlen=10000)

        # Latency tracking
        self._latencies: Deque[float] = deque(maxlen=1000)

        # Custom thresholds
        self._custom_thresholds: Dict[str, MonitoringThreshold] = {}

        # Statistics
        self._stats = {
            "checks_performed": 0,
            "alerts_generated": 0,
            "alerts_by_severity": {},
            "alerts_by_category": {},
            "escalations": 0,
            "kill_switch_triggers": 0,
        }

        logger.info(
            "RealTimeMonitor initialized",
            extra={
                "check_interval": self._config.check_interval_seconds,
                "alert_deadline": self._config.max_alert_delay_seconds,
            },
        )

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Real-time monitoring started")

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Real-time monitoring stopped")

    def start_sync(self) -> None:
        """Start monitoring in a background thread (for sync code)."""
        if self._running:
            return

        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._monitor_loop())

        thread = threading.Thread(target=run_loop, daemon=True, name="realtime-monitor")
        self._running = True
        thread.start()
        logger.info("Real-time monitoring started (background thread)")

    def stop_sync(self) -> None:
        """Stop background monitoring."""
        self._running = False

    # =========================================================================
    # Main Monitoring Loop
    # =========================================================================

    async def _monitor_loop(self) -> None:
        """
        Main monitoring loop.

        Per RTS 6 Article 17: Check metrics and generate alerts within deadline.
        """
        while self._running:
            try:
                start_time = time.time()

                # Run all checks
                await self._check_all_metrics()

                # Track check time (must be < 5 seconds per RTS 6)
                check_duration = time.time() - start_time
                if check_duration > self._config.max_alert_delay_seconds:
                    logger.warning(
                        f"Monitoring check took {check_duration:.2f}s, "
                        f"exceeds RTS 6 deadline of {self._config.max_alert_delay_seconds}s"
                    )

                with self._lock:
                    self._stats["checks_performed"] += 1

                # Wait for next check
                await asyncio.sleep(self._config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)

    async def _check_all_metrics(self) -> None:
        """Check all monitored metrics."""
        # Take snapshot
        metrics = self._get_current_metrics()

        # Store in history
        with self._lock:
            self._metrics = metrics
            self._metrics_history.append(metrics)

        # Run checks
        await self._check_order_to_trade_ratio(metrics)
        await self._check_pnl_thresholds(metrics)
        await self._check_position_limits(metrics)
        await self._check_message_rate(metrics)
        await self._check_latency(metrics)
        await self._check_clock_drift(metrics)
        await self._check_system_health(metrics)

    def _get_current_metrics(self) -> MonitoringMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return MonitoringMetrics(
                orders_submitted=self._metrics.orders_submitted,
                orders_cancelled=self._metrics.orders_cancelled,
                orders_filled=self._metrics.orders_filled,
                order_to_trade_ratio=self._calculate_otr(),
                realized_pnl=self._metrics.realized_pnl,
                unrealized_pnl=self._metrics.unrealized_pnl,
                daily_pnl=self._metrics.daily_pnl,
                daily_pnl_limit=self._metrics.daily_pnl_limit,
                total_position_value=self._metrics.total_position_value,
                max_position_value=self._metrics.max_position_value,
                position_utilization_pct=self._calculate_position_utilization(),
                messages_per_second=self._calculate_message_rate(),
                max_messages_per_second=self._metrics.max_messages_per_second,
                avg_latency_ms=self._calculate_avg_latency(),
                max_latency_ms=self._metrics.max_latency_ms,
                p99_latency_ms=self._calculate_p99_latency(),
                cpu_usage_pct=self._metrics.cpu_usage_pct,
                memory_usage_pct=self._metrics.memory_usage_pct,
                clock_offset_ms=self._metrics.clock_offset_ms,
                clock_synced=self._metrics.clock_synced,
            )

    # =========================================================================
    # Individual Checks
    # =========================================================================

    async def _check_order_to_trade_ratio(self, metrics: MonitoringMetrics) -> None:
        """Check order-to-trade ratio."""
        otr = metrics.order_to_trade_ratio

        if otr >= self._config.otr_emergency_threshold:
            await self._generate_alert_async(
                severity=AlertSeverity.EMERGENCY,
                category=AlertCategory.ORDER_TO_TRADE_RATIO,
                message=f"OTR {otr:.1f} exceeds emergency threshold {self._config.otr_emergency_threshold}",
                details={"otr": otr, "threshold": self._config.otr_emergency_threshold},
            )
        elif otr >= self._config.otr_critical_threshold:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ORDER_TO_TRADE_RATIO,
                message=f"OTR {otr:.1f} exceeds critical threshold {self._config.otr_critical_threshold}",
                details={"otr": otr, "threshold": self._config.otr_critical_threshold},
            )
        elif otr >= self._config.otr_warning_threshold:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.ORDER_TO_TRADE_RATIO,
                message=f"OTR {otr:.1f} exceeds warning threshold {self._config.otr_warning_threshold}",
                details={"otr": otr, "threshold": self._config.otr_warning_threshold},
            )

    async def _check_pnl_thresholds(self, metrics: MonitoringMetrics) -> None:
        """Check P&L against thresholds."""
        if metrics.daily_pnl_limit <= 0:
            return

        pnl_pct = abs(float(metrics.daily_pnl)) / float(metrics.daily_pnl_limit) * 100

        if pnl_pct >= self._config.pnl_emergency_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.EMERGENCY,
                category=AlertCategory.PNL_THRESHOLD,
                message=f"Daily P&L at {pnl_pct:.1f}% of limit - emergency threshold",
                details={
                    "daily_pnl": str(metrics.daily_pnl),
                    "limit": str(metrics.daily_pnl_limit),
                    "pct_of_limit": pnl_pct,
                },
            )
        elif pnl_pct >= self._config.pnl_critical_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.PNL_THRESHOLD,
                message=f"Daily P&L at {pnl_pct:.1f}% of limit",
                details={
                    "daily_pnl": str(metrics.daily_pnl),
                    "limit": str(metrics.daily_pnl_limit),
                    "pct_of_limit": pnl_pct,
                },
            )
        elif pnl_pct >= self._config.pnl_warning_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PNL_THRESHOLD,
                message=f"Daily P&L at {pnl_pct:.1f}% of limit",
                details={
                    "daily_pnl": str(metrics.daily_pnl),
                    "limit": str(metrics.daily_pnl_limit),
                    "pct_of_limit": pnl_pct,
                },
            )

    async def _check_position_limits(self, metrics: MonitoringMetrics) -> None:
        """Check position utilization against limits."""
        util_pct = metrics.position_utilization_pct

        if util_pct >= self._config.position_emergency_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.EMERGENCY,
                category=AlertCategory.POSITION_LIMIT,
                message=f"Position utilization at {util_pct:.1f}% - emergency threshold",
                details={"utilization_pct": util_pct},
            )
        elif util_pct >= self._config.position_critical_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.POSITION_LIMIT,
                message=f"Position utilization at {util_pct:.1f}%",
                details={"utilization_pct": util_pct},
            )
        elif util_pct >= self._config.position_warning_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.POSITION_LIMIT,
                message=f"Position utilization at {util_pct:.1f}%",
                details={"utilization_pct": util_pct},
            )

    async def _check_message_rate(self, metrics: MonitoringMetrics) -> None:
        """Check message rate against limits."""
        if metrics.max_messages_per_second <= 0:
            return

        rate_pct = metrics.messages_per_second / metrics.max_messages_per_second * 100

        if rate_pct >= self._config.message_rate_critical_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.MESSAGE_RATE,
                message=f"Message rate at {rate_pct:.1f}% of limit",
                details={
                    "messages_per_second": metrics.messages_per_second,
                    "max": metrics.max_messages_per_second,
                },
            )
        elif rate_pct >= self._config.message_rate_warning_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.MESSAGE_RATE,
                message=f"Message rate at {rate_pct:.1f}% of limit",
                details={
                    "messages_per_second": metrics.messages_per_second,
                    "max": metrics.max_messages_per_second,
                },
            )

    async def _check_latency(self, metrics: MonitoringMetrics) -> None:
        """Check latency against thresholds."""
        latency = metrics.p99_latency_ms

        if latency >= self._config.latency_emergency_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.EMERGENCY,
                category=AlertCategory.LATENCY,
                message=f"P99 latency {latency:.1f}ms exceeds emergency threshold",
                details={"p99_latency_ms": latency, "avg_latency_ms": metrics.avg_latency_ms},
            )
        elif latency >= self._config.latency_critical_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.LATENCY,
                message=f"P99 latency {latency:.1f}ms exceeds critical threshold",
                details={"p99_latency_ms": latency, "avg_latency_ms": metrics.avg_latency_ms},
            )
        elif latency >= self._config.latency_warning_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.LATENCY,
                message=f"P99 latency {latency:.1f}ms exceeds warning threshold",
                details={"p99_latency_ms": latency, "avg_latency_ms": metrics.avg_latency_ms},
            )

    async def _check_clock_drift(self, metrics: MonitoringMetrics) -> None:
        """Check clock drift per RTS 25."""
        drift = abs(metrics.clock_offset_ms)

        if drift >= self._config.clock_drift_emergency_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.EMERGENCY,
                category=AlertCategory.CLOCK_DRIFT,
                message=f"Clock drift {drift:.1f}ms exceeds emergency threshold (RTS 25 violation)",
                details={"clock_offset_ms": metrics.clock_offset_ms},
            )
        elif drift >= self._config.clock_drift_critical_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.CLOCK_DRIFT,
                message=f"Clock drift {drift:.1f}ms exceeds RTS 25 threshold",
                details={"clock_offset_ms": metrics.clock_offset_ms},
            )
        elif drift >= self._config.clock_drift_warning_ms:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CLOCK_DRIFT,
                message=f"Clock drift {drift:.1f}ms approaching threshold",
                details={"clock_offset_ms": metrics.clock_offset_ms},
            )

    async def _check_system_health(self, metrics: MonitoringMetrics) -> None:
        """Check system resource usage."""
        # CPU check
        if metrics.cpu_usage_pct >= self._config.cpu_critical_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.CAPACITY,
                message=f"CPU usage at {metrics.cpu_usage_pct:.1f}%",
                details={"cpu_usage_pct": metrics.cpu_usage_pct},
            )
        elif metrics.cpu_usage_pct >= self._config.cpu_warning_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CAPACITY,
                message=f"CPU usage at {metrics.cpu_usage_pct:.1f}%",
                details={"cpu_usage_pct": metrics.cpu_usage_pct},
            )

        # Memory check
        if metrics.memory_usage_pct >= self._config.memory_critical_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.CAPACITY,
                message=f"Memory usage at {metrics.memory_usage_pct:.1f}%",
                details={"memory_usage_pct": metrics.memory_usage_pct},
            )
        elif metrics.memory_usage_pct >= self._config.memory_warning_pct:
            await self._generate_alert_async(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CAPACITY,
                message=f"Memory usage at {metrics.memory_usage_pct:.1f}%",
                details={"memory_usage_pct": metrics.memory_usage_pct},
            )

    # =========================================================================
    # Metric Updates
    # =========================================================================

    def update_order_metrics(
        self,
        submitted: int = 0,
        cancelled: int = 0,
        filled: int = 0,
    ) -> None:
        """
        Update order metrics for OTR calculation.

        Args:
            submitted: Number of orders submitted
            cancelled: Number of orders cancelled
            filled: Number of orders filled
        """
        with self._lock:
            self._metrics.orders_submitted += submitted
            self._metrics.orders_cancelled += cancelled
            self._metrics.orders_filled += filled

            # Track for rolling OTR
            now = time.time()
            for _ in range(submitted):
                self._order_timestamps.append((now, "submitted"))
            for _ in range(filled):
                self._order_timestamps.append((now, "filled"))

    def update_pnl(
        self,
        realized_pnl: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
        daily_pnl: Optional[Decimal] = None,
        daily_pnl_limit: Optional[Decimal] = None,
    ) -> None:
        """Update P&L metrics."""
        with self._lock:
            if realized_pnl is not None:
                self._metrics.realized_pnl = realized_pnl
            if unrealized_pnl is not None:
                self._metrics.unrealized_pnl = unrealized_pnl
            if daily_pnl is not None:
                self._metrics.daily_pnl = daily_pnl
            if daily_pnl_limit is not None:
                self._metrics.daily_pnl_limit = daily_pnl_limit

    def update_position(
        self,
        total_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None,
    ) -> None:
        """Update position metrics."""
        with self._lock:
            if total_value is not None:
                self._metrics.total_position_value = total_value
            if max_value is not None:
                self._metrics.max_position_value = max_value

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._latencies.append(latency_ms)
            if latency_ms > self._metrics.max_latency_ms:
                self._metrics.max_latency_ms = latency_ms

    def record_message(self) -> None:
        """Record a message for rate tracking."""
        with self._lock:
            now = time.time()
            self._order_timestamps.append((now, "message"))

    def update_clock_status(self, offset_ms: float, synced: bool = True) -> None:
        """Update clock synchronization status."""
        with self._lock:
            self._metrics.clock_offset_ms = offset_ms
            self._metrics.clock_synced = synced

    def update_system_metrics(
        self,
        cpu_pct: Optional[float] = None,
        memory_pct: Optional[float] = None,
        disk_pct: Optional[float] = None,
    ) -> None:
        """Update system resource metrics."""
        with self._lock:
            if cpu_pct is not None:
                self._metrics.cpu_usage_pct = cpu_pct
            if memory_pct is not None:
                self._metrics.memory_usage_pct = memory_pct
            if disk_pct is not None:
                self._metrics.disk_usage_pct = disk_pct

    # =========================================================================
    # Calculations
    # =========================================================================

    def _calculate_otr(self) -> float:
        """Calculate order-to-trade ratio."""
        with self._lock:
            trades = max(self._metrics.orders_filled, 1)
            return self._metrics.orders_submitted / trades

    def _calculate_position_utilization(self) -> float:
        """Calculate position utilization percentage."""
        with self._lock:
            if self._metrics.max_position_value <= 0:
                return 0.0
            return float(
                self._metrics.total_position_value / self._metrics.max_position_value * 100
            )

    def _calculate_message_rate(self) -> float:
        """Calculate messages per second (rolling)."""
        with self._lock:
            now = time.time()
            cutoff = now - 1.0  # Last 1 second

            count = sum(
                1 for ts, _ in self._order_timestamps
                if ts > cutoff
            )
            return float(count)

    def _calculate_avg_latency(self) -> float:
        """Calculate average latency."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    def _calculate_p99_latency(self) -> float:
        """Calculate P99 latency."""
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_latencies = sorted(self._latencies)
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    # =========================================================================
    # Alert Generation
    # =========================================================================

    def generate_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        algorithm_id: Optional[str] = None,
        venue: Optional[str] = None,
        instrument: Optional[str] = None,
    ) -> Optional[ComplianceAlert]:
        """
        Generate a compliance alert (synchronous).

        Args:
            severity: Alert severity level
            category: Alert category
            message: Alert message
            details: Additional details
            algorithm_id: Related algorithm
            venue: Related venue
            instrument: Related instrument

        Returns:
            ComplianceAlert if generated (may be suppressed by rate limiting)
        """
        # Check rate limiting
        if not self._check_rate_limit(category):
            return None

        # Check deduplication
        if self._check_duplicate(category, message):
            return None

        alert = ComplianceAlert(
            severity=severity,
            category=category,
            message=message,
            details=details or {},
            algorithm_id=algorithm_id,
            venue=venue,
            instrument=instrument,
        )

        self._process_alert(alert)
        return alert

    async def _generate_alert_async(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        algorithm_id: Optional[str] = None,
        venue: Optional[str] = None,
        instrument: Optional[str] = None,
    ) -> Optional[ComplianceAlert]:
        """Generate alert (async version)."""
        alert = self.generate_alert(
            severity=severity,
            category=category,
            message=message,
            details=details,
            algorithm_id=algorithm_id,
            venue=venue,
            instrument=instrument,
        )

        if alert and self._async_alert_callback:
            try:
                await self._async_alert_callback(alert)
            except Exception as e:
                logger.error(f"Async alert callback error: {e}")

        return alert

    def _process_alert(self, alert: ComplianceAlert) -> None:
        """Process and dispatch an alert."""
        with self._lock:
            self._alerts.append(alert)
            self._stats["alerts_generated"] += 1
            self._stats["alerts_by_severity"][alert.severity.value] = \
                self._stats["alerts_by_severity"].get(alert.severity.value, 0) + 1
            self._stats["alerts_by_category"][alert.category.value] = \
                self._stats["alerts_by_category"].get(alert.category.value, 0) + 1

        # Log the alert
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.warning)

        alert_data = alert.to_dict()
        # Remove 'message' key to avoid conflict with LogRecord
        alert_data.pop('message', None)
        log_method(
            f"ALERT [{alert.severity.value.upper()}] {alert.category.value}: {alert.message}",
            extra=alert_data,
        )

        # Sync callback
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Escalation
        if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
            with self._lock:
                self._stats["escalations"] += 1

            if self._escalation_callback:
                try:
                    self._escalation_callback(alert)
                except Exception as e:
                    logger.error(f"Escalation callback error: {e}")

        # Kill switch for emergency
        if alert.severity == AlertSeverity.EMERGENCY:
            with self._lock:
                self._stats["kill_switch_triggers"] += 1

            if self._kill_switch_callback:
                try:
                    self._kill_switch_callback(alert)
                except Exception as e:
                    logger.error(f"Kill switch callback error: {e}")

        # Persist
        if self._config.persist_alerts:
            self._persist_alert(alert)

    def _check_rate_limit(self, category: AlertCategory) -> bool:
        """Check if alert is within rate limits."""
        now = time.time()
        key = category.value

        with self._lock:
            if key not in self._alert_counts:
                self._alert_counts[key] = deque(maxlen=100)

            # Clean old entries
            cutoff = now - 60.0
            while self._alert_counts[key] and self._alert_counts[key][0] < cutoff:
                self._alert_counts[key].popleft()

            # Check limit
            if len(self._alert_counts[key]) >= self._config.max_alerts_per_category_per_minute:
                return False

            self._alert_counts[key].append(now)
            return True

    def _check_duplicate(self, category: AlertCategory, message: str) -> bool:
        """Check if alert is a duplicate within dedup window."""
        now = time.time()
        alert_hash = f"{category.value}:{message}"

        with self._lock:
            # Clean old entries
            cutoff = now - self._config.alert_deduplication_window_seconds
            while self._alert_hashes and self._alert_hashes[0][1] < cutoff:
                self._alert_hashes.popleft()

            # Check for duplicate
            for h, _ in self._alert_hashes:
                if h == alert_hash:
                    return True

            self._alert_hashes.append((alert_hash, now))
            return False

    def _persist_alert(self, alert: ComplianceAlert) -> None:
        """Persist alert to disk."""
        try:
            import os
            from pathlib import Path
            import json

            path = Path(self._config.alert_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")

    # =========================================================================
    # Alert Management
    # =========================================================================

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = time.time_ns()
                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = time.time_ns()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
        return False

    def get_alerts(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
        unacknowledged_only: bool = False,
        unresolved_only: bool = False,
    ) -> List[ComplianceAlert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = self._alerts.copy()

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return alerts[-limit:]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_current_metrics(self) -> MonitoringMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return self._metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            return {
                **self._stats,
                "running": self._running,
                "total_alerts": len(self._alerts),
                "unacknowledged_alerts": sum(
                    1 for a in self._alerts if not a.acknowledged
                ),
                "current_metrics": self._metrics.to_dict(),
            }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate RTS 6 Article 17 compliance report.

        Returns:
            Dictionary with compliance report data
        """
        stats = self.get_statistics()
        metrics = self.get_current_metrics()

        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": "RTS 6 Article 17",
            "requirement": "Real-Time Monitoring",
            "compliance_status": "ACTIVE" if self._running else "INACTIVE",
            "alert_deadline_requirement": "5 seconds",
            "alert_deadline_configured": f"{self._config.max_alert_delay_seconds} seconds",
            "configuration": {
                "check_interval": self._config.check_interval_seconds,
                "otr_warning": self._config.otr_warning_threshold,
                "otr_critical": self._config.otr_critical_threshold,
                "pnl_warning_pct": self._config.pnl_warning_pct,
                "latency_warning_ms": self._config.latency_warning_ms,
            },
            "current_metrics": metrics.to_dict(),
            "statistics": stats,
            "recent_alerts": [a.to_dict() for a in self.get_alerts(limit=10)],
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_realtime_monitor(
    config: Optional[Dict[str, Any]] = None,
    alert_callback: Optional[Callable[[ComplianceAlert], None]] = None,
    kill_switch_callback: Optional[Callable[[ComplianceAlert], None]] = None,
) -> RealTimeMonitor:
    """
    Factory function to create RealTimeMonitor instance.

    Args:
        config: Optional configuration dictionary
        alert_callback: Optional alert callback
        kill_switch_callback: Optional kill switch callback

    Returns:
        Configured RealTimeMonitor instance
    """
    cfg = RealTimeMonitorConfig()

    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return RealTimeMonitor(
        config=cfg,
        alert_callback=alert_callback,
        kill_switch_callback=kill_switch_callback,
    )


__all__ = [
    "AlertSeverity",
    "AlertCategory",
    "ComplianceAlert",
    "MonitoringThreshold",
    "RealTimeMonitorConfig",
    "MonitoringMetrics",
    "RealTimeMonitor",
    "create_realtime_monitor",
]
