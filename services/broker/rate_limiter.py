"""
Pre-broker rate limiting to protect user accounts.

Prevents users from exceeding broker rate limits which could:
- Lock their API access temporarily or permanently
- Cause order rejections at critical moments
- Result in account restrictions from the broker

Features:
    - Per-user, per-broker rate tracking
    - Warning before hitting limits
    - Automatic throttling at limit
    - Circuit breaker for runaway strategies
    - Configurable broker-specific limits

References:
    - Token bucket algorithm (RFC 6585)
    - Circuit breaker pattern (Release It!, Nygard)
    - Broker API documentation (IB, Alpaca, Binance)

Broker Rate Limits (conservative):
    - Interactive Brokers: 45/sec (actual: 50 msg/sec)
    - Alpaca: 3/sec orders, 180/min API (actual: 200/min)
    - Binance: 8/sec orders, 1000/min API (actual: 10/sec, 1200/min)
    - Coinbase: 8/sec (actual: 10/sec)

Example:
    >>> limiter = BrokerRateLimiter()
    >>> check = limiter.check_and_consume("user_001", "alpaca", "order")
    >>> if check.status == RateLimitStatus.THROTTLED:
    ...     await asyncio.sleep(check.wait_seconds)
    >>> elif check.status == RateLimitStatus.BLOCKED:
    ...     raise CircuitBreakerOpenError(check.message)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set

logger = logging.getLogger("broker.rate_limiter")


class RateLimitStatus(Enum):
    """Status of a rate limit check."""

    OK = "ok"  # Request allowed
    WARNING = "warning"  # Approaching limit
    THROTTLED = "throttled"  # At limit, should wait
    BLOCKED = "blocked"  # Circuit breaker open


@dataclass
class BrokerRateLimits:
    """
    Rate limit configuration per broker.

    Attributes:
        orders_per_second: Maximum orders per second
        api_calls_per_minute: Maximum API calls per minute
        warning_threshold: Percentage of limit to trigger warning (0.8 = 80%)
        cooldown_seconds: Cooldown period after hitting limit
        burst_multiplier: Allow burst up to this multiple briefly
    """

    orders_per_second: float
    api_calls_per_minute: int
    warning_threshold: float = 0.8
    cooldown_seconds: int = 60
    burst_multiplier: float = 1.5

    def __post_init__(self):
        if not 0 < self.warning_threshold <= 1:
            raise ValueError("warning_threshold must be between 0 and 1")


# Conservative limits (below actual broker limits for safety margin)
BROKER_LIMITS: Dict[str, BrokerRateLimits] = {
    "interactive_brokers": BrokerRateLimits(
        orders_per_second=45.0,  # Actual: 50 msg/sec
        api_calls_per_minute=2700,  # Conservative
        warning_threshold=0.8,
    ),
    "alpaca": BrokerRateLimits(
        orders_per_second=3.0,  # 200/min = 3.3/sec
        api_calls_per_minute=180,  # Conservative vs 200
        warning_threshold=0.8,
    ),
    "binance": BrokerRateLimits(
        orders_per_second=8.0,  # Actual: 10/sec
        api_calls_per_minute=1000,  # Actual: 1200
        warning_threshold=0.8,
    ),
    "coinbase": BrokerRateLimits(
        orders_per_second=8.0,  # Actual: 10/sec
        api_calls_per_minute=8000,  # Actual: 10,000/hour
        warning_threshold=0.8,
    ),
    "kraken": BrokerRateLimits(
        orders_per_second=1.0,  # Conservative
        api_calls_per_minute=15,  # Very conservative
        warning_threshold=0.7,
    ),
}


@dataclass
class RateLimitCheck:
    """
    Result of a rate limit check.

    Attributes:
        status: Current status
        current_rate: Current request rate
        limit: Maximum allowed rate
        wait_seconds: Seconds to wait if throttled
        message: Human-readable message
        utilization: Current utilization percentage
    """

    status: RateLimitStatus
    current_rate: float
    limit: float
    wait_seconds: float = 0.0
    message: str = ""
    utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "current_rate": self.current_rate,
            "limit": self.limit,
            "wait_seconds": self.wait_seconds,
            "message": self.message,
            "utilization": self.utilization,
        }


@dataclass
class CircuitBreakerState:
    """
    State of a circuit breaker.

    Attributes:
        is_open: Whether the circuit is open (blocking)
        opened_at: When the circuit was opened
        closes_at: When the circuit will close
        reason: Why the circuit was opened
        failure_count: Number of failures that triggered opening
    """

    is_open: bool
    opened_at: Optional[datetime] = None
    closes_at: Optional[datetime] = None
    reason: Optional[str] = None
    failure_count: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, wait_seconds: float):
        self.message = message
        self.wait_seconds = wait_seconds
        super().__init__(message)


class RunawayStrategyError(Exception):
    """Raised when a strategy is detected as runaway."""

    pass


class BrokerRateLimiter:
    """
    Rate limiter with circuit breaker for broker API protection.

    Features:
        - Per-user, per-broker rate tracking using sliding windows
        - Warning alerts before hitting limits
        - Automatic throttling with calculated wait times
        - Circuit breaker for runaway strategies
        - Thread-safe operations

    Example:
        >>> limiter = BrokerRateLimiter()
        >>>
        >>> async def execute_order(user_id: str, broker: str, order):
        ...     check = limiter.check_and_consume(user_id, broker, "order")
        ...
        ...     if check.status == RateLimitStatus.BLOCKED:
        ...         raise CircuitBreakerOpenError(check.message, check.wait_seconds)
        ...
        ...     if check.status == RateLimitStatus.THROTTLED:
        ...         await asyncio.sleep(check.wait_seconds)
        ...         check = limiter.check_and_consume(user_id, broker, "order")
        ...
        ...     if check.status == RateLimitStatus.WARNING:
        ...         logger.warning(f"Approaching rate limit: {check.message}")
        ...
        ...     return await broker_client.submit_order(order)
    """

    def __init__(
        self,
        custom_limits: Optional[Dict[str, BrokerRateLimits]] = None,
        on_warning: Optional[Callable[[str, str, RateLimitCheck], None]] = None,
        on_circuit_open: Optional[Callable[[str, str, str], None]] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            custom_limits: Custom broker limits (merged with defaults)
            on_warning: Callback when warning threshold reached
            on_circuit_open: Callback when circuit breaker opens
        """
        self._limits = {**BROKER_LIMITS}
        if custom_limits:
            self._limits.update(custom_limits)

        # Sliding windows: user:broker -> deque of timestamps
        self._windows: Dict[str, Deque[float]] = {}

        # Circuit breakers: user:broker -> CircuitBreakerState
        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Callbacks
        self._on_warning = on_warning
        self._on_circuit_open = on_circuit_open

        # Thread safety
        self._lock = threading.RLock()

    def _get_key(self, user_id: str, broker: str) -> str:
        """Generate cache key for user:broker pair."""
        return f"{user_id}:{broker}"

    def check_and_consume(
        self, user_id: str, broker: str, request_type: str = "order"
    ) -> RateLimitCheck:
        """
        Check rate limit and consume a token if allowed.

        This is the main entry point for rate limiting.

        Args:
            user_id: User making the request
            broker: Target broker
            request_type: Type of request ("order" or "api_call")

        Returns:
            RateLimitCheck with status and wait time if throttled
        """
        key = self._get_key(user_id, broker)
        limits = self._limits.get(broker)

        # No limits configured for this broker
        if limits is None:
            return RateLimitCheck(
                status=RateLimitStatus.OK,
                current_rate=0,
                limit=0,
                message=f"No rate limits configured for {broker}",
            )

        with self._lock:
            # Check circuit breaker first
            cb_state = self._circuit_breakers.get(key)
            if cb_state and cb_state.is_open:
                now = datetime.now(timezone.utc)
                if cb_state.closes_at and now < cb_state.closes_at:
                    wait_seconds = (cb_state.closes_at - now).total_seconds()
                    return RateLimitCheck(
                        status=RateLimitStatus.BLOCKED,
                        current_rate=0,
                        limit=limits.orders_per_second,
                        wait_seconds=wait_seconds,
                        message=f"Circuit breaker open: {cb_state.reason}",
                    )
                else:
                    # Circuit breaker has expired, close it
                    self._circuit_breakers[key] = CircuitBreakerState(is_open=False)

            # Initialize window if needed
            if key not in self._windows:
                self._windows[key] = deque()

            window = self._windows[key]
            now = time.time()
            window_start = now - 1.0  # 1-second sliding window

            # Remove old entries
            while window and window[0] < window_start:
                window.popleft()

            current_rate = len(window)
            limit = limits.orders_per_second
            utilization = (current_rate / limit) * 100 if limit > 0 else 0

            # Check if at limit
            if current_rate >= limit:
                # Calculate wait time until oldest request expires
                wait_time = max(0.0, window[0] - window_start + 0.1) if window else 0.1

                logger.warning(
                    f"RATE_LIMIT_THROTTLED | user={user_id} | "
                    f"broker={broker} | rate={current_rate}/{limit}/sec"
                )

                return RateLimitCheck(
                    status=RateLimitStatus.THROTTLED,
                    current_rate=current_rate,
                    limit=limit,
                    wait_seconds=wait_time,
                    message=f"Rate limit reached ({current_rate}/{limit}/sec). Wait {wait_time:.2f}s",
                    utilization=utilization,
                )

            # Check if approaching limit (warning)
            if current_rate >= limit * limits.warning_threshold:
                window.append(now)
                new_rate = current_rate + 1
                new_utilization = (new_rate / limit) * 100 if limit > 0 else 0

                check = RateLimitCheck(
                    status=RateLimitStatus.WARNING,
                    current_rate=new_rate,
                    limit=limit,
                    message=f"Approaching rate limit: {new_rate}/{limit}/sec ({new_utilization:.1f}%)",
                    utilization=new_utilization,
                )

                if self._on_warning:
                    self._on_warning(user_id, broker, check)

                logger.info(
                    f"RATE_LIMIT_WARNING | user={user_id} | "
                    f"broker={broker} | rate={new_rate}/{limit}/sec"
                )

                return check

            # OK to proceed
            window.append(now)
            new_rate = current_rate + 1
            new_utilization = (new_rate / limit) * 100 if limit > 0 else 0

            return RateLimitCheck(
                status=RateLimitStatus.OK,
                current_rate=new_rate,
                limit=limit,
                utilization=new_utilization,
            )

    def trigger_circuit_breaker(
        self, user_id: str, broker: str, reason: str, duration_seconds: int = 60
    ) -> None:
        """
        Open circuit breaker to stop all requests.

        Use for:
            - Runaway strategy detection
            - Broker error responses indicating issues
            - Manual emergency stop

        Args:
            user_id: User to block
            broker: Broker to block
            reason: Reason for opening circuit
            duration_seconds: How long to keep circuit open
        """
        key = self._get_key(user_id, broker)
        now = datetime.now(timezone.utc)

        with self._lock:
            self._circuit_breakers[key] = CircuitBreakerState(
                is_open=True,
                opened_at=now,
                closes_at=now + timedelta(seconds=duration_seconds),
                reason=reason,
            )

        logger.warning(
            f"CIRCUIT_BREAKER_OPENED | user={user_id} | "
            f"broker={broker} | reason={reason} | duration={duration_seconds}s"
        )

        if self._on_circuit_open:
            self._on_circuit_open(user_id, broker, reason)

    def close_circuit_breaker(self, user_id: str, broker: str) -> bool:
        """
        Manually close a circuit breaker.

        Args:
            user_id: User
            broker: Broker

        Returns:
            True if circuit was closed, False if not open
        """
        key = self._get_key(user_id, broker)

        with self._lock:
            if key in self._circuit_breakers and self._circuit_breakers[key].is_open:
                self._circuit_breakers[key] = CircuitBreakerState(is_open=False)
                logger.info(f"CIRCUIT_BREAKER_CLOSED | user={user_id} | broker={broker}")
                return True
            return False

    def get_circuit_breaker_state(self, user_id: str, broker: str) -> CircuitBreakerState:
        """
        Get current circuit breaker state.

        Args:
            user_id: User
            broker: Broker

        Returns:
            CircuitBreakerState
        """
        key = self._get_key(user_id, broker)

        with self._lock:
            return self._circuit_breakers.get(key, CircuitBreakerState(is_open=False))

    def get_user_status(self, user_id: str, broker: str) -> Dict[str, Any]:
        """
        Get current rate limit status for a user.

        Args:
            user_id: User to check
            broker: Broker to check

        Returns:
            Status dictionary
        """
        key = self._get_key(user_id, broker)
        limits = self._limits.get(broker)

        with self._lock:
            window = self._windows.get(key, deque())
            now = time.time()
            recent = sum(1 for t in window if t > now - 1.0)

            cb_state = self._circuit_breakers.get(key, CircuitBreakerState(is_open=False))

            return {
                "user_id": user_id,
                "broker": broker,
                "current_rate_per_second": recent,
                "limit_per_second": limits.orders_per_second if limits else None,
                "utilization_percent": (recent / limits.orders_per_second * 100) if limits else 0,
                "circuit_breaker_active": cb_state.is_open,
                "circuit_breaker_reason": cb_state.reason if cb_state.is_open else None,
                "circuit_breaker_closes_at": (
                    cb_state.closes_at.isoformat() if cb_state.closes_at else None
                ),
            }

    def reset_user(self, user_id: str, broker: str) -> None:
        """
        Reset rate limit state for a user.

        Args:
            user_id: User to reset
            broker: Broker to reset
        """
        key = self._get_key(user_id, broker)

        with self._lock:
            if key in self._windows:
                self._windows[key].clear()
            if key in self._circuit_breakers:
                self._circuit_breakers[key] = CircuitBreakerState(is_open=False)

        logger.info(f"RATE_LIMIT_RESET | user={user_id} | broker={broker}")

    def get_all_active_circuit_breakers(self) -> List[Dict[str, Any]]:
        """
        Get all currently active circuit breakers.

        Returns:
            List of active circuit breaker states
        """
        now = datetime.now(timezone.utc)
        active = []

        with self._lock:
            for key, state in self._circuit_breakers.items():
                if state.is_open and state.closes_at and state.closes_at > now:
                    user_id, broker = key.split(":", 1)
                    active.append(
                        {
                            "user_id": user_id,
                            "broker": broker,
                            "opened_at": state.opened_at.isoformat() if state.opened_at else None,
                            "closes_at": state.closes_at.isoformat(),
                            "reason": state.reason,
                        }
                    )

        return active


class RunawayDetector:
    """
    Detects and stops runaway trading strategies.

    A runaway strategy is one that places orders at an abnormally
    high rate, potentially due to bugs or infinite loops.

    Features:
        - Per-strategy order counting
        - Automatic circuit breaker triggering
        - Configurable thresholds

    Example:
        >>> detector = RunawayDetector(rate_limiter)
        >>> try:
        ...     detector.check_strategy(user_id, strategy_id, broker)
        ... except RunawayStrategyError:
        ...     # Strategy has been stopped
        ...     notify_user(user_id, "Strategy stopped due to excessive order rate")
    """

    # Default threshold: 100 orders in rapid succession = runaway
    DEFAULT_THRESHOLD = 100

    # Default cooldown when runaway detected
    DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        rate_limiter: BrokerRateLimiter,
        threshold: int = DEFAULT_THRESHOLD,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        window_seconds: float = 60.0,
    ):
        """
        Initialize the runaway detector.

        Args:
            rate_limiter: Rate limiter to trigger circuit breaker on
            threshold: Number of orders to trigger runaway detection
            cooldown_seconds: How long to block after runaway detected
            window_seconds: Time window for counting orders
        """
        self._limiter = rate_limiter
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._window = window_seconds
        self._order_counts: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _get_key(self, user_id: str, strategy_id: str) -> str:
        """Generate key for user:strategy pair."""
        return f"{user_id}:{strategy_id}"

    def check_strategy(self, user_id: str, strategy_id: str, broker: str) -> None:
        """
        Check if a strategy is placing orders too rapidly.

        Call this before each order execution.

        Args:
            user_id: User running the strategy
            strategy_id: Strategy identifier
            broker: Target broker

        Raises:
            RunawayStrategyError: If strategy is detected as runaway
        """
        key = self._get_key(user_id, strategy_id)
        now = time.time()

        with self._lock:
            if key not in self._order_counts:
                self._order_counts[key] = deque()

            window = self._order_counts[key]
            window_start = now - self._window

            # Remove old entries
            while window and window[0] < window_start:
                window.popleft()

            # Add current order
            window.append(now)

            # Check threshold
            if len(window) > self._threshold:
                # Trigger circuit breaker
                self._limiter.trigger_circuit_breaker(
                    user_id,
                    broker,
                    reason=f"Runaway strategy detected: {strategy_id} ({len(window)} orders in {self._window}s)",
                    duration_seconds=self._cooldown,
                )

                # Clear the count
                window.clear()

                logger.error(
                    f"RUNAWAY_STRATEGY_DETECTED | user={user_id} | "
                    f"strategy={strategy_id} | broker={broker} | "
                    f"orders={len(window)} | window={self._window}s"
                )

                raise RunawayStrategyError(
                    f"Strategy {strategy_id} stopped: excessive order rate "
                    f"(>{self._threshold} orders in {self._window}s)"
                )

    def get_strategy_order_count(self, user_id: str, strategy_id: str) -> int:
        """
        Get current order count for a strategy.

        Args:
            user_id: User
            strategy_id: Strategy

        Returns:
            Number of orders in the current window
        """
        key = self._get_key(user_id, strategy_id)
        now = time.time()

        with self._lock:
            window = self._order_counts.get(key, deque())
            window_start = now - self._window
            return sum(1 for t in window if t >= window_start)

    def reset_strategy(self, user_id: str, strategy_id: str) -> None:
        """
        Reset order count for a strategy.

        Args:
            user_id: User
            strategy_id: Strategy
        """
        key = self._get_key(user_id, strategy_id)

        with self._lock:
            if key in self._order_counts:
                self._order_counts[key].clear()
