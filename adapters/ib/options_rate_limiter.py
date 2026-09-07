# -*- coding: utf-8 -*-
"""
adapters/ib/options_rate_limiter.py
IB Options-specific rate limit manager with intelligent caching and priority queue.

Phase 2: US Exchange Adapters

Problem:
    SPY has 24 expirations × 20 strikes = 480 series.
    At IB's 10 chains/min limit, full chain refresh = 48 minutes.

Solution:
    - Intelligent caching with configurable TTL
    - Priority queue for rate-limited requests
    - Front-month prioritization
    - Incremental delta updates

IB Options Rate Limits:
    ─────────────────────────────────────────────────────────────────────────────
    | Type                    | Limit                  | Our Limit  | Strategy   |
    |-------------------------|------------------------|------------|------------|
    | Option chains           | 10/min                 | 8/min      | Cache      |
    | Quote requests          | 100/sec                | 80/sec     | Throttle   |
    | Order submissions       | 50/sec                 | 40/sec     | Block      |
    | Market data lines       | 100 concurrent         | 100        | LRU evict  |
    ─────────────────────────────────────────────────────────────────────────────

References:
    - IB TWS API: https://interactivebrokers.github.io/tws-api/
    - IB Market Data: https://interactivebrokers.github.io/tws-api/market_data.html
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Priority Levels
# =============================================================================


class RequestPriority(IntEnum):
    """
    Priority levels for rate-limited requests.

    Lower value = higher priority.
    """

    ORDER_EXECUTION = 0  # Highest: order execution
    RISK_UPDATE = 1  # Position risk updates
    FRONT_MONTH = 2  # Front-month chain refresh
    ACTIVE_UNDERLYING = 3  # Active underlyings we're trading
    BACKGROUND_REFRESH = 4  # Background chain updates
    BACKFILL = 9  # Lowest: historical backfill


# =============================================================================
# Cached Option Chain
# =============================================================================


@dataclass
class CachedChain:
    """
    Cached option chain with TTL.

    Attributes:
        underlying: Underlying symbol
        expiration: Expiration date
        chain_data: The cached chain data
        timestamp: When the cache was created
        ttl_sec: Time-to-live in seconds
        access_count: Number of times accessed (for LFU)
        last_access: Last access timestamp
    """

    underlying: str
    expiration: date
    chain_data: Any
    timestamp: float
    ttl_sec: float = 300.0  # 5-minute default TTL
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > self.ttl_sec

    def is_stale(self, max_age_sec: float) -> bool:
        """Check if cache entry is stale (older than max_age)."""
        return time.time() - self.timestamp > max_age_sec

    def touch(self) -> None:
        """Update access count and time."""
        self.access_count += 1
        self.last_access = time.time()


# =============================================================================
# Options Chain Cache
# =============================================================================


class OptionsChainCache:
    """
    LRU cache for option chains with configurable TTL.

    Features:
        - Separate TTL for front-month vs back-month options
        - LRU eviction when at capacity
        - Thread-safe operations
        - Statistics tracking

    Example:
        cache = OptionsChainCache(max_chains=100, default_ttl_sec=300.0)

        # Store chain
        cache.put("AAPL", date(2024, 12, 20), chain_data)

        # Retrieve chain
        chain = cache.get("AAPL", date(2024, 12, 20))
        if chain is not None:
            # Use cached chain
            ...
    """

    def __init__(
        self,
        max_chains: int = 100,
        default_ttl_sec: float = 300.0,  # 5 minutes
        front_month_ttl_sec: float = 60.0,  # 1 minute for front month
        front_month_days: int = 35,  # Days to consider front month
    ) -> None:
        """
        Initialize cache.

        Args:
            max_chains: Maximum number of chains to cache
            default_ttl_sec: Default TTL for back-month options
            front_month_ttl_sec: TTL for front-month options (shorter)
            front_month_days: Days until expiration to consider front month
        """
        self._cache: OrderedDict[str, CachedChain] = OrderedDict()
        self._max_chains = max_chains
        self._default_ttl = default_ttl_sec
        self._front_month_ttl = front_month_ttl_sec
        self._front_month_days = front_month_days
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _make_key(self, underlying: str, expiration: date) -> str:
        """Create cache key from underlying and expiration."""
        return f"{underlying}:{expiration.isoformat()}"

    def _is_front_month(self, expiration: date) -> bool:
        """Check if expiration is front-month."""
        days_to_expiry = (expiration - date.today()).days
        return 0 <= days_to_expiry <= self._front_month_days

    def get(
        self,
        underlying: str,
        expiration: date,
        max_age_sec: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Get cached chain if not expired.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            max_age_sec: Optional maximum age (overrides TTL)

        Returns:
            Cached chain data or None if not found/expired
        """
        key = self._make_key(underlying, expiration)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            cached = self._cache[key]

            # Check expiration
            if cached.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Check max age if specified
            if max_age_sec is not None and cached.is_stale(max_age_sec):
                self._misses += 1
                return None

            # Update access and move to end (LRU)
            cached.touch()
            self._cache.move_to_end(key)

            self._hits += 1
            return cached.chain_data

    def put(
        self,
        underlying: str,
        expiration: date,
        chain_data: Any,
        ttl_sec: Optional[float] = None,
    ) -> None:
        """
        Cache chain with appropriate TTL.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            chain_data: Chain data to cache
            ttl_sec: Optional custom TTL (overrides default)
        """
        key = self._make_key(underlying, expiration)

        # Determine TTL
        if ttl_sec is None:
            ttl_sec = (
                self._front_month_ttl if self._is_front_month(expiration) else self._default_ttl
            )

        with self._lock:
            self._cache[key] = CachedChain(
                underlying=underlying,
                expiration=expiration,
                chain_data=chain_data,
                timestamp=time.time(),
                ttl_sec=ttl_sec,
            )
            self._cache.move_to_end(key)

            # Evict oldest if over capacity
            while len(self._cache) > self._max_chains:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Evicted chain cache entry: {evicted_key}")

    def invalidate(self, underlying: str, expiration: Optional[date] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            underlying: Underlying symbol
            expiration: Optional specific expiration (None = all for underlying)

        Returns:
            Number of entries invalidated
        """
        count = 0

        with self._lock:
            if expiration is not None:
                # Invalidate specific entry
                key = self._make_key(underlying, expiration)
                if key in self._cache:
                    del self._cache[key]
                    count = 1
            else:
                # Invalidate all entries for underlying
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{underlying}:")]
                for key in keys_to_remove:
                    del self._cache[key]
                    count += 1

        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        count = 0

        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]
                count += 1

        return count

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_chains,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Prioritized Request
# =============================================================================


@dataclass(order=True)
class PrioritizedRequest:
    """
    Priority queue item for rate-limited requests.

    Ordering is by priority (lower = higher priority), then by timestamp.
    """

    priority: int
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    request_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Optional[Callable[[Any], None]] = field(compare=False, default=None)
    error_callback: Optional[Callable[[Exception], None]] = field(compare=False, default=None)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# =============================================================================
# IB Options Rate Limit Manager
# =============================================================================


class IBOptionsRateLimitManager:
    """
    Rate limit manager with priority queue for IB options requests.

    Handles:
        - Chain requests with caching
        - Quote requests with throttling
        - Order submissions with blocking
        - Market data subscription management

    Priority levels:
        0: Order execution (highest)
        1: Position risk updates
        2: Front-month chain refresh
        3: Active underlyings
        4: Background chain refresh
        9: Backfill requests (lowest)

    Example:
        manager = IBOptionsRateLimitManager()

        # Request chain with priority
        manager.request_chain(
            underlying="AAPL",
            expiration=date(2024, 12, 20),
            callback=process_chain,
            priority=RequestPriority.FRONT_MONTH,
        )

        # Process queue (call periodically)
        processed = manager.process_queue()

    References:
        - Existing IBRateLimiter in adapters/ib/market_data.py
    """

    # Rate limits (with safety margins)
    CHAIN_LIMIT_PER_MIN = 8  # IB limit: 10, safety: 8
    QUOTE_LIMIT_PER_SEC = 80  # IB limit: 100, safety: 80
    ORDER_LIMIT_PER_SEC = 40  # IB limit: 50, safety: 40
    MAX_MARKET_DATA_LINES = 100  # Hard limit

    def __init__(
        self,
        chain_limit_per_min: int = CHAIN_LIMIT_PER_MIN,
        quote_limit_per_sec: int = QUOTE_LIMIT_PER_SEC,
        order_limit_per_sec: int = ORDER_LIMIT_PER_SEC,
        cache_max_chains: int = 100,
        cache_default_ttl_sec: float = 300.0,
        cache_front_month_ttl_sec: float = 60.0,
    ) -> None:
        """
        Initialize rate limit manager.

        Args:
            chain_limit_per_min: Chain requests per minute limit
            quote_limit_per_sec: Quote requests per second limit
            order_limit_per_sec: Order submissions per second limit
            cache_max_chains: Maximum chains to cache
            cache_default_ttl_sec: Default cache TTL
            cache_front_month_ttl_sec: Front-month cache TTL
        """
        # Rate limits
        self._chain_limit = chain_limit_per_min
        self._quote_limit = quote_limit_per_sec
        self._order_limit = order_limit_per_sec

        # Chain cache
        self._chain_cache = OptionsChainCache(
            max_chains=cache_max_chains,
            default_ttl_sec=cache_default_ttl_sec,
            front_month_ttl_sec=cache_front_month_ttl_sec,
        )

        # Request queue
        self._request_queue: List[PrioritizedRequest] = []

        # Rate tracking
        self._chain_requests_this_minute: int = 0
        self._minute_reset_time: float = time.time()
        self._quote_requests_this_second: int = 0
        self._second_reset_time: float = time.time()
        self._order_requests_this_second: int = 0
        self._order_second_reset_time: float = time.time()

        # Market data subscriptions
        self._active_subscriptions: Set[str] = set()

        # Threading
        self._lock = threading.Lock()

        # Statistics
        self._chains_requested = 0
        self._chains_from_cache = 0
        self._queue_peak_size = 0

    def _reset_minute_counter_if_needed(self) -> None:
        """Reset minute counter if a minute has passed."""
        now = time.time()
        if now - self._minute_reset_time >= 60:
            self._chain_requests_this_minute = 0
            self._minute_reset_time = now

    def _reset_second_counter_if_needed(self) -> None:
        """Reset second counter if a second has passed."""
        now = time.time()
        if now - self._second_reset_time >= 1.0:
            self._quote_requests_this_second = 0
            self._second_reset_time = now

    def _reset_order_counter_if_needed(self) -> None:
        """Reset order counter if a second has passed."""
        now = time.time()
        if now - self._order_second_reset_time >= 1.0:
            self._order_requests_this_second = 0
            self._order_second_reset_time = now

    def can_request_chain(self) -> bool:
        """Check if chain request can be made."""
        with self._lock:
            self._reset_minute_counter_if_needed()
            return self._chain_requests_this_minute < self._chain_limit

    def can_request_quote(self) -> bool:
        """Check if quote request can be made."""
        with self._lock:
            self._reset_second_counter_if_needed()
            return self._quote_requests_this_second < self._quote_limit

    def can_submit_order(self) -> bool:
        """Check if order can be submitted."""
        with self._lock:
            self._reset_order_counter_if_needed()
            return self._order_requests_this_second < self._order_limit

    def can_subscribe_market_data(self) -> bool:
        """Check if market data subscription can be added."""
        return len(self._active_subscriptions) < self.MAX_MARKET_DATA_LINES

    def record_chain_request(self) -> None:
        """Record a chain request being made."""
        with self._lock:
            self._reset_minute_counter_if_needed()
            self._chain_requests_this_minute += 1
            self._chains_requested += 1

    def record_quote_request(self) -> None:
        """Record a quote request being made."""
        with self._lock:
            self._reset_second_counter_if_needed()
            self._quote_requests_this_second += 1

    def record_order_request(self) -> None:
        """Record an order submission."""
        with self._lock:
            self._reset_order_counter_if_needed()
            self._order_requests_this_second += 1

    def request_chain(
        self,
        underlying: str,
        expiration: date,
        callback: Callable[[Any], None],
        error_callback: Optional[Callable[[Exception], None]] = None,
        priority: int = RequestPriority.BACKGROUND_REFRESH,
        force_refresh: bool = False,
    ) -> bool:
        """
        Request option chain with priority queueing.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            callback: Function to call with chain data
            error_callback: Function to call on error
            priority: Request priority (lower = higher)
            force_refresh: Force refresh even if cached

        Returns:
            True if request queued, False if served from cache
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self._chain_cache.get(underlying, expiration)
            if cached is not None:
                self._chains_from_cache += 1
                callback(cached)
                return False

        # Queue request
        request_id = f"chain:{underlying}:{expiration.isoformat()}"
        request = PrioritizedRequest(
            priority=priority,
            timestamp=time.time(),
            request_id=request_id,
            request_type="chain",
            payload={"underlying": underlying, "expiration": expiration},
            callback=callback,
            error_callback=error_callback,
        )

        with self._lock:
            heapq.heappush(self._request_queue, request)
            self._queue_peak_size = max(self._queue_peak_size, len(self._request_queue))

        return True

    def request_quotes(
        self,
        contracts: List[str],
        callback: Callable[[Any], None],
        error_callback: Optional[Callable[[Exception], None]] = None,
        priority: int = RequestPriority.ACTIVE_UNDERLYING,
    ) -> str:
        """
        Request option quotes with priority queueing.

        Args:
            contracts: List of OCC symbols
            callback: Function to call with quote data
            error_callback: Function to call on error
            priority: Request priority

        Returns:
            Request ID
        """
        request_id = f"quotes:{int(time.time() * 1000)}"
        request = PrioritizedRequest(
            priority=priority,
            timestamp=time.time(),
            request_id=request_id,
            request_type="quotes",
            payload={"contracts": contracts},
            callback=callback,
            error_callback=error_callback,
        )

        with self._lock:
            heapq.heappush(self._request_queue, request)
            self._queue_peak_size = max(self._queue_peak_size, len(self._request_queue))

        return request_id

    def process_queue(
        self,
        executor: Optional[Callable[[PrioritizedRequest], Any]] = None,
        max_requests: int = 10,
    ) -> int:
        """
        Process pending requests within rate limits.

        Args:
            executor: Function to execute requests (if None, just dequeues)
            max_requests: Maximum requests to process in this call

        Returns:
            Number of requests processed
        """
        processed = 0

        with self._lock:
            self._reset_minute_counter_if_needed()
            self._reset_second_counter_if_needed()

            while self._request_queue and processed < max_requests:
                # Peek at next request
                if not self._request_queue:
                    break

                request = self._request_queue[0]

                # Check if we can process based on request type
                can_process = False
                if request.request_type == "chain":
                    can_process = self._chain_requests_this_minute < self._chain_limit
                    if can_process:
                        self._chain_requests_this_minute += 1
                elif request.request_type == "quotes":
                    can_process = self._quote_requests_this_second < self._quote_limit
                    if can_process:
                        self._quote_requests_this_second += 1
                elif request.request_type == "order":
                    can_process = self._order_requests_this_second < self._order_limit
                    if can_process:
                        self._order_requests_this_second += 1
                else:
                    can_process = True

                if not can_process:
                    # Can't process more of this type, but might process other types
                    # For simplicity, break here (more sophisticated would skip)
                    break

                # Pop and process
                request = heapq.heappop(self._request_queue)

        # Execute outside lock
        if executor is not None and request is not None:
            try:
                result = executor(request)

                # Cache chain results
                if request.request_type == "chain" and result is not None:
                    self._chain_cache.put(
                        request.payload["underlying"],
                        request.payload["expiration"],
                        result,
                    )

                if request.callback is not None:
                    request.callback(result)

            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
                if request.error_callback is not None:
                    request.error_callback(e)

        processed += 1
        return processed

    def cache_chain(
        self,
        underlying: str,
        expiration: date,
        chain_data: Any,
        ttl_sec: Optional[float] = None,
    ) -> None:
        """
        Manually cache chain data.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            chain_data: Chain data to cache
            ttl_sec: Optional custom TTL
        """
        self._chain_cache.put(underlying, expiration, chain_data, ttl_sec)

    def get_cached_chain(
        self,
        underlying: str,
        expiration: date,
    ) -> Optional[Any]:
        """
        Get cached chain without queueing.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            Cached chain or None
        """
        return self._chain_cache.get(underlying, expiration)

    def invalidate_cache(
        self,
        underlying: str,
        expiration: Optional[date] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            underlying: Underlying symbol
            expiration: Optional specific expiration

        Returns:
            Number of entries invalidated
        """
        return self._chain_cache.invalidate(underlying, expiration)

    def add_subscription(self, symbol: str) -> bool:
        """
        Register market data subscription.

        Args:
            symbol: Symbol being subscribed

        Returns:
            True if added, False if at limit
        """
        if len(self._active_subscriptions) >= self.MAX_MARKET_DATA_LINES:
            return False
        self._active_subscriptions.add(symbol)
        return True

    def remove_subscription(self, symbol: str) -> bool:
        """
        Remove market data subscription.

        Args:
            symbol: Symbol to unsubscribe

        Returns:
            True if removed, False if not found
        """
        if symbol in self._active_subscriptions:
            self._active_subscriptions.remove(symbol)
            return True
        return False

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return len(self._request_queue)

    @property
    def subscription_count(self) -> int:
        """Current number of subscriptions."""
        return len(self._active_subscriptions)

    @property
    def cache_hit_rate(self) -> float:
        """Chain cache hit rate."""
        return self._chain_cache.hit_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "queue_size": len(self._request_queue),
            "queue_peak_size": self._queue_peak_size,
            "chains_requested": self._chains_requested,
            "chains_from_cache": self._chains_from_cache,
            "chain_cache_hit_rate": self._chains_from_cache / max(1, self._chains_requested),
            "active_subscriptions": len(self._active_subscriptions),
            "chains_this_minute": self._chain_requests_this_minute,
            "quotes_this_second": self._quote_requests_this_second,
            "cache_stats": self._chain_cache.get_stats(),
        }

    def wait_for_chain_slot(self, timeout: float = 60.0) -> bool:
        """
        Wait until chain request can be made.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if slot available, False if timeout
        """
        start = time.time()
        while not self.can_request_chain():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def wait_for_quote_slot(self, timeout: float = 5.0) -> bool:
        """
        Wait until quote request can be made.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if slot available, False if timeout
        """
        start = time.time()
        while not self.can_request_quote():
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        return True

    def wait_for_order_slot(self, timeout: float = 5.0) -> bool:
        """
        Wait until order can be submitted.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if slot available, False if timeout
        """
        start = time.time()
        while not self.can_submit_order():
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        return True


# =============================================================================
# Factory Functions
# =============================================================================


def create_options_rate_limiter(
    profile: str = "default",
    **kwargs: Any,
) -> IBOptionsRateLimitManager:
    """
    Create options rate limit manager with predefined profile.

    Args:
        profile: Configuration profile name
        **kwargs: Override any profile settings

    Returns:
        Configured rate limit manager

    Profiles:
        - "default": Standard settings for normal usage
        - "conservative": Lower limits for safety
        - "aggressive": Higher throughput (use with caution)
        - "backtest": Minimal caching for backtesting
    """
    profiles = {
        "default": {
            "chain_limit_per_min": 8,
            "quote_limit_per_sec": 80,
            "order_limit_per_sec": 40,
            "cache_max_chains": 100,
            "cache_default_ttl_sec": 300.0,
            "cache_front_month_ttl_sec": 60.0,
        },
        "conservative": {
            "chain_limit_per_min": 5,
            "quote_limit_per_sec": 50,
            "order_limit_per_sec": 25,
            "cache_max_chains": 200,
            "cache_default_ttl_sec": 600.0,
            "cache_front_month_ttl_sec": 120.0,
        },
        "aggressive": {
            "chain_limit_per_min": 10,
            "quote_limit_per_sec": 100,
            "order_limit_per_sec": 50,
            "cache_max_chains": 50,
            "cache_default_ttl_sec": 180.0,
            "cache_front_month_ttl_sec": 30.0,
        },
        "backtest": {
            "chain_limit_per_min": 100,  # No real limit for backtest
            "quote_limit_per_sec": 1000,
            "order_limit_per_sec": 1000,
            "cache_max_chains": 10,
            "cache_default_ttl_sec": 60.0,
            "cache_front_month_ttl_sec": 60.0,
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(profiles.keys())}")

    config = dict(profiles[profile])
    config.update(kwargs)

    return IBOptionsRateLimitManager(**config)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias for tests expecting OptionsRateLimiter
OptionsRateLimiter = IBOptionsRateLimitManager


__all__ = [
    "RequestPriority",
    "CachedChain",
    "OptionsChainCache",
    "PrioritizedRequest",
    "IBOptionsRateLimitManager",
    "create_options_rate_limiter",
    # Backward compatibility
    "OptionsRateLimiter",
]
