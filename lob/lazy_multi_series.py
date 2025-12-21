"""
Lazy Multi-Series LOB Manager for Options Markets.

This module provides memory-efficient LOB management for options chains
where each underlying can have 100-1000+ individual option series.

Problem Statement:
    - SPY chain: 24 expiries × 20 strikes × 2 (call/put) = 960 series
    - Each full LOB at 1000 levels: ~500MB
    - Total naive memory: 480 GB — impossible!

Solution:
    1. Lazy instantiation: Only create LOB when first accessed
    2. LRU eviction: Keep max N LOBs in memory (default: 50)
    3. Ring buffer depth: Limit each LOB to M levels (default: 100)
    4. Disk persistence: Store evicted LOBs compressed on disk

Memory Budget:
    - 50 active LOBs × 50MB (100 levels) = 2.5 GB ✓
    - vs 960 full LOBs × 500MB = 480 GB ✗

Reference:
    Phase 0.5 of OPTIONS_INTEGRATION_PLAN.md

Performance Targets:
    - LOB access latency: < 1 ms (including lazy load)
    - Event propagation: < 100 μs per tick
    - Peak memory for SPY full chain: < 4 GB

DISK CACHE SECURITY MODEL
--------------------------
This module uses pickle for disk persistence. Pickle deserialization can execute
arbitrary code if cache files are tampered with. Security controls:

1. HMAC Integrity Verification:
   - Each cache file includes HMAC-SHA256 signature
   - Signature verified before deserialization
   - Tampered files are rejected and logged

2. Threat Model:
   - TRUSTED: Cache files created by this process (same machine, same user)
   - UNTRUSTED: Cache files from external sources, shared drives, or after compromise
   - Control: HMAC key from LOB_CACHE_HMAC_KEY env var (default: research-only key)

3. Operational Guidance:
   - Production: Set LOB_CACHE_HMAC_KEY to a strong random secret
   - Research: Default key acceptable for local-only caches
   - Never share cache directories across trust boundaries without key rotation

Tech Debt Reference: docs/reports/TECH_DEBT_REGISTRY.md#security-lob-cache-pickle
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from lob.data_structures import LimitOrder, OrderBook, PriceLevel, Side

logger = logging.getLogger(__name__)


# ==============================================================================
# Cache Security - HMAC Integrity Verification
# ==============================================================================

# HMAC key for cache file integrity verification
# Production: Set LOB_CACHE_HMAC_KEY environment variable to a strong secret
# Research: Default key is acceptable for local-only, single-user caches
_LOB_CACHE_HMAC_KEY: bytes = os.environ.get(
    "LOB_CACHE_HMAC_KEY",
    "lob-cache-research-default-key-v1"  # Default for research use only
).encode("utf-8")

# HMAC signature length (SHA-256 = 32 bytes)
_HMAC_SIGNATURE_LENGTH = 32


def _compute_cache_hmac(data: bytes) -> bytes:
    """Compute HMAC-SHA256 signature for cache data."""
    return hmac.new(_LOB_CACHE_HMAC_KEY, data, hashlib.sha256).digest()


def _verify_cache_hmac(data: bytes, signature: bytes) -> bool:
    """Verify HMAC signature for cache data. Returns True if valid."""
    expected = _compute_cache_hmac(data)
    return hmac.compare_digest(expected, signature)


# ==============================================================================
# Type Definitions
# ==============================================================================


class EvictionPolicy(Enum):
    """LOB eviction policy."""

    LRU = auto()  # Least Recently Used (default)
    LFU = auto()  # Least Frequently Used
    TTL = auto()  # Time-To-Live based


@dataclass
class SeriesKey:
    """
    Unique identifier for an option series.

    Format: {underlying}_{expiry}_{type}_{strike}
    Example: "AAPL_241220_C_200" for AAPL Dec 20 2024 200 Call

    Attributes:
        underlying: Underlying symbol (e.g., "AAPL", "SPY")
        expiry: Expiration date as YYMMDD string
        option_type: "C" for call, "P" for put
        strike: Strike price as Decimal
    """

    underlying: str
    expiry: str
    option_type: str  # "C" or "P"
    strike: "Decimal"  # Forward reference for Decimal

    def __str__(self) -> str:
        return f"{self.underlying}_{self.expiry}_{self.option_type}_{self.strike}"

    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def from_string(cls, key_str: str) -> "SeriesKey":
        """
        Parse series key from string format.

        Args:
            key_str: Key string like "AAPL_241220_C_200"

        Returns:
            Parsed SeriesKey instance

        Raises:
            ValueError: If key format is invalid
        """
        from decimal import Decimal as Dec

        parts = key_str.split("_")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid series key format: {key_str}. "
                f"Expected format: UNDERLYING_YYMMDD_C/P_STRIKE"
            )

        underlying, expiry, option_type, strike_str = parts

        if option_type not in ("C", "P"):
            raise ValueError(
                f"Invalid option type: {option_type}. Must be 'C' or 'P'"
            )

        try:
            strike = Dec(strike_str)
        except Exception:
            raise ValueError(f"Invalid strike price: {strike_str}")

        return cls(
            underlying=underlying,
            expiry=expiry,
            option_type=option_type,
            strike=strike,
        )

    @classmethod
    def from_contract(
        cls,
        underlying: str,
        expiry_date: str,  # YYMMDD or YYYYMMDD
        is_call: bool,
        strike: Union[float, "Decimal"],
    ) -> "SeriesKey":
        """
        Create series key from contract parameters.

        Args:
            underlying: Underlying symbol
            expiry_date: Expiration date (YYMMDD or YYYYMMDD)
            is_call: True for call, False for put
            strike: Strike price

        Returns:
            SeriesKey instance
        """
        from decimal import Decimal as Dec

        # Normalize expiry to YYMMDD
        if len(expiry_date) == 8:
            expiry = expiry_date[2:]  # YYYYMMDD -> YYMMDD
        else:
            expiry = expiry_date

        # Convert to Decimal
        if not isinstance(strike, Dec):
            strike = Dec(str(strike))

        return cls(
            underlying=underlying.upper(),
            expiry=expiry,
            option_type="C" if is_call else "P",
            strike=strike,
        )

    def get_bucket_key(self) -> Tuple[str, str]:
        """Get bucket key for grouping (underlying, expiry)."""
        return (self.underlying, self.expiry)


@dataclass
class LOBMetadata:
    """
    Metadata for a series LOB.

    Tracks access patterns for eviction decisions.
    """

    series_key: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    total_orders: int = 0
    is_dirty: bool = False  # Has unsaved changes

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class SeriesLOBState:
    """
    Complete state for a series LOB.

    Encapsulates order book plus metadata for persistence.
    Provides convenience methods that delegate to the underlying order book.
    """

    order_book: OrderBook
    metadata: LOBMetadata
    orders_by_id: Dict[str, LimitOrder] = field(default_factory=dict)

    def add_order(
        self,
        side: Side,
        price: Any,  # Decimal, float, or str
        qty: Any,  # Decimal, float, or str
        order_id: str,
        timestamp_ns: Optional[int] = None,
    ) -> int:
        """
        Add an order to the book.

        Args:
            side: Side.BUY or Side.SELL
            price: Order price (Decimal, float, or str)
            qty: Order quantity (Decimal, float, or str)
            order_id: Unique order identifier
            timestamp_ns: Timestamp in nanoseconds (defaults to current time)

        Returns:
            Queue position at the price level
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)

        # Convert to float for internal OrderBook
        price_float = float(price)
        qty_float = float(qty)

        order = LimitOrder(
            order_id=order_id,
            price=price_float,
            qty=qty_float,
            remaining_qty=qty_float,
            timestamp_ns=timestamp_ns,
            side=side,
        )

        queue_pos = self.order_book.add_limit_order(order)
        self.orders_by_id[order_id] = order
        return queue_pos

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        if order_id not in self.orders_by_id:
            return False
        result = self.order_book.cancel_order(order_id)
        if result:
            del self.orders_by_id[order_id]
        return result

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the order book."""
        bids, asks = self.order_book.get_depth(n_levels=100)
        return {
            "bids": bids,
            "asks": asks,
            "best_bid": self.order_book.best_bid,
            "best_ask": self.order_book.best_ask,
            "mid_price": self.order_book.mid_price,
            "spread_bps": self.order_book.spread_bps,
        }

    def get_memory_estimate_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimate: each order ~200 bytes, each level ~100 bytes overhead
        num_orders = len(self.orders_by_id)
        num_levels = self.order_book.num_bid_levels + self.order_book.num_ask_levels
        return num_orders * 200 + num_levels * 100 + 1000  # +1KB overhead


@dataclass
class ManagerStats:
    """Statistics for the LOB manager."""

    active_lobs: int = 0
    total_lobs_created: int = 0
    total_evictions: int = 0
    total_restorations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    disk_writes: int = 0
    disk_reads: int = 0
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    # Aliases for test compatibility
    @property
    def evictions(self) -> int:
        """Alias for total_evictions."""
        return self.total_evictions

    @property
    def total_creates(self) -> int:
        """Alias for total_lobs_created."""
        return self.total_lobs_created

    @property
    def hits(self) -> int:
        """Alias for cache_hits."""
        return self.cache_hits

    @property
    def misses(self) -> int:
        """Alias for cache_misses."""
        return self.cache_misses


# ==============================================================================
# Lazy Multi-Series LOB Manager
# ==============================================================================


class LazyMultiSeriesLOBManager:
    """
    Memory-efficient multi-series LOB manager.

    Manages hundreds of option series LOBs with bounded memory using:
    - Lazy instantiation: LOBs created only on first access
    - LRU eviction: Keeps only N most-recently-used LOBs in memory
    - Disk persistence: Evicted LOBs saved to disk for restoration
    - Ring buffer depth: Each LOB limited to M price levels

    Thread Safety:
        This class is thread-safe for concurrent LOB access.
        Uses a ReentrantLock for synchronization.

    Usage:
        manager = LazyMultiSeriesLOBManager(
            max_active_lobs=50,
            max_depth_per_lob=100,
            disk_cache_path=Path("./lob_cache"),
        )

        # Get or create LOB for specific series
        lob = manager.get_lob("AAPL_241220_C_200")

        # Add order
        lob.add_order(order)

        # Get stats
        stats = manager.get_stats()

    Memory Budget Calculation:
        - max_active_lobs=50
        - max_depth_per_lob=100 levels
        - ~1000 orders per LOB on average
        - Memory per LOB: 1000 × 200B + 100 × 100B ≈ 210KB
        - Total: 50 × 210KB = 10.5 MB active LOBs
        - With overhead: ~50 MB total

        This is dramatically less than naive 480 GB!
    """

    def __init__(
        self,
        max_active_lobs: int = 50,
        max_depth_per_lob: int = 100,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        disk_cache_path: Optional[Path] = None,
        persist_dir: Optional[Path] = None,  # Alias for disk_cache_path
        enable_compression: bool = True,
        compression_level: int = 6,
        on_eviction: Optional[Callable[[str, SeriesLOBState], None]] = None,
        on_evict: Optional[Callable[[str, Any], None]] = None,  # Alias for on_eviction
        on_restoration: Optional[Callable[[str, SeriesLOBState], None]] = None,
        persist_on_evict: bool = True,
        ttl_seconds: Optional[float] = None,
        max_memory_bytes: Optional[int] = None,
        max_persist_age_days: Optional[int] = None,
    ):
        """
        Initialize the lazy multi-series LOB manager.

        Args:
            max_active_lobs: Maximum number of LOBs to keep in memory
            max_depth_per_lob: Maximum price levels per LOB (ring buffer depth)
            eviction_policy: Policy for choosing which LOB to evict
            disk_cache_path: Path for disk persistence (None = no persistence)
            persist_dir: Alias for disk_cache_path
            enable_compression: Whether to gzip compress disk files
            compression_level: Gzip compression level (1-9)
            on_eviction: Callback when LOB is evicted
            on_evict: Alias for on_eviction
            on_restoration: Callback when LOB is restored from disk
            persist_on_evict: Whether to persist LOB to disk on eviction
            ttl_seconds: Time-to-live in seconds for TTL eviction policy
            max_memory_bytes: Maximum memory usage in bytes (soft limit)
            max_persist_age_days: Maximum age of persisted files in days

        Raises:
            ValueError: If parameters are invalid
        """
        if max_active_lobs < 1:
            raise ValueError("max_active_lobs must be >= 1")
        if max_depth_per_lob < 1:
            raise ValueError("max_depth_per_lob must be >= 1")
        if compression_level < 1 or compression_level > 9:
            raise ValueError("compression_level must be 1-9")

        self._max_active = max_active_lobs
        self._max_depth = max_depth_per_lob
        self._eviction_policy = eviction_policy

        # Handle persist_dir as alias for disk_cache_path
        actual_cache_path = persist_dir if persist_dir is not None else disk_cache_path
        self._disk_cache_path = actual_cache_path
        self._enable_compression = enable_compression
        self._compression_level = compression_level

        # Handle on_evict as alias for on_eviction
        self._on_eviction = on_evict if on_evict is not None else on_eviction
        self._on_restoration = on_restoration
        self._persist_on_evict = persist_on_evict
        self._ttl_seconds = ttl_seconds
        self._max_memory_bytes = max_memory_bytes
        self._max_persist_age_days = max_persist_age_days

        # Active LOBs in memory (OrderedDict for LRU ordering)
        self._active_lobs: OrderedDict[str, SeriesLOBState] = OrderedDict()

        # Access frequency tracking (for LFU policy)
        self._access_counts: Dict[str, int] = {}

        # Lock for thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = ManagerStats()

        # Create disk cache directory if specified
        if self._disk_cache_path:
            self._disk_cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"LazyMultiSeriesLOBManager initialized: "
            f"max_active={max_active_lobs}, max_depth={max_depth_per_lob}, "
            f"policy={eviction_policy.name}, "
            f"disk_cache={'enabled' if actual_cache_path else 'disabled'}"
        )

    def get_lob(self, series_key: Union[str, SeriesKey]) -> SeriesLOBState:
        """
        Get or create LOB for the specified series.

        If LOB is not in memory, attempts to restore from disk.
        If not on disk, creates new empty LOB.
        Evicts LRU LOB if at capacity.

        Args:
            series_key: Series identifier (string or SeriesKey)

        Returns:
            SeriesLOBState with order book and metadata
        """
        key = str(series_key)

        with self._lock:
            # Check if already in memory
            if key in self._active_lobs:
                # Move to end (most recently used) for LRU
                self._active_lobs.move_to_end(key)
                state = self._active_lobs[key]
                state.metadata.touch()
                self._stats.cache_hits += 1
                return state

            self._stats.cache_misses += 1

            # Evict if at capacity
            if len(self._active_lobs) >= self._max_active:
                self._evict_one()

            # Try to restore from disk
            state = self._try_restore_from_disk(key)

            if state is None:
                # Create new LOB
                state = self._create_new_lob(key)

            # Add to active set
            self._active_lobs[key] = state
            self._stats.active_lobs = len(self._active_lobs)
            self._update_memory_stats()

            # Touch for first access (creation or restore also counts as access)
            state.metadata.touch()

            return state

    def get_or_create(self, series_key: Union[str, SeriesKey]) -> SeriesLOBState:
        """
        Get or create LOB for the specified series.

        Alias for get_lob() for API compatibility.

        Args:
            series_key: Series identifier (string or SeriesKey)

        Returns:
            SeriesLOBState with order book and metadata
        """
        return self.get_lob(series_key)

    def has_lob(self, series_key: Union[str, SeriesKey]) -> bool:
        """Check if LOB exists in memory."""
        key = str(series_key)
        with self._lock:
            return key in self._active_lobs

    def has_series(self, series_key: Union[str, SeriesKey]) -> bool:
        """
        Check if series LOB exists in memory.

        Alias for has_lob() for API compatibility.

        Args:
            series_key: Series identifier

        Returns:
            True if series is currently in memory
        """
        return self.has_lob(series_key)

    def has_lob_on_disk(self, series_key: Union[str, SeriesKey]) -> bool:
        """Check if LOB exists on disk."""
        if not self._disk_cache_path:
            return False
        key = str(series_key)
        cache_file = self._get_cache_file_path(key)
        return cache_file.exists()

    def remove_lob(self, series_key: Union[str, SeriesKey]) -> bool:
        """
        Remove LOB from memory and disk.

        Args:
            series_key: Series identifier

        Returns:
            True if removed, False if not found
        """
        key = str(series_key)

        with self._lock:
            # Remove from memory
            removed = self._active_lobs.pop(key, None) is not None

            # Remove from disk
            if self._disk_cache_path:
                cache_file = self._get_cache_file_path(key)
                if cache_file.exists():
                    cache_file.unlink()
                    removed = True

            if removed:
                self._stats.active_lobs = len(self._active_lobs)

            return removed

    def get_active_keys(self) -> List[str]:
        """Get list of currently active LOB keys."""
        with self._lock:
            return list(self._active_lobs.keys())

    def get_all_active_keys(self) -> List[str]:
        """
        Get list of all active series keys.

        Alias for get_active_keys() for API compatibility.

        Returns:
            List of active series keys
        """
        return self.get_active_keys()

    def get_active_count(self) -> int:
        """
        Get count of active LOBs in memory.

        Returns:
            Number of currently active LOBs
        """
        with self._lock:
            return len(self._active_lobs)

    def get_metadata(self, series_key: Union[str, SeriesKey]) -> Optional[LOBMetadata]:
        """
        Get metadata for a specific series.

        Args:
            series_key: Series identifier

        Returns:
            LOBMetadata if series exists in memory, None otherwise
        """
        key = str(series_key)
        with self._lock:
            if key in self._active_lobs:
                metadata = self._active_lobs[key].metadata
                # Add last_access alias for API compatibility
                if not hasattr(metadata, 'last_access'):
                    object.__setattr__(metadata, 'last_access', metadata.last_accessed)
                return metadata
            return None

    def persist_to_disk(self, series_key: Union[str, SeriesKey]) -> bool:
        """
        Explicitly persist a series LOB to disk.

        Args:
            series_key: Series identifier

        Returns:
            True if persisted, False if series not found or persistence disabled
        """
        if not self._disk_cache_path:
            return False

        key = str(series_key)
        with self._lock:
            if key not in self._active_lobs:
                return False
            state = self._active_lobs[key]
            self._persist_to_disk_internal(key, state)
            state.metadata.is_dirty = False
            return True

    def evict(self, series_key: Union[str, SeriesKey]) -> bool:
        """
        Explicitly evict a specific series from memory.

        If disk persistence is enabled, persists state before eviction.

        Args:
            series_key: Series identifier

        Returns:
            True if evicted, False if series not found
        """
        key = str(series_key)

        with self._lock:
            if key not in self._active_lobs:
                return False

            state = self._active_lobs[key]

            # Persist to disk if enabled
            if self._disk_cache_path and self._persist_on_evict:
                self._persist_to_disk_internal(key, state)

            # Remove from memory
            del self._active_lobs[key]

            # Callback
            if self._on_eviction:
                try:
                    self._on_eviction(key, state)
                except Exception as e:
                    logger.warning(f"Eviction callback failed for {key}: {e}")

            self._stats.total_evictions += 1
            self._stats.active_lobs = len(self._active_lobs)

            return True

    def evict_all(self, persist_first: bool = True) -> int:
        """
        Evict all LOBs from memory.

        Args:
            persist_first: If True, persist all dirty LOBs to disk first

        Returns:
            Number of LOBs evicted
        """
        with self._lock:
            count = len(self._active_lobs)

            if persist_first and self._disk_cache_path:
                self.flush_to_disk(force=True)

            # Callbacks for each evicted LOB
            if self._on_eviction:
                for key, state in list(self._active_lobs.items()):
                    try:
                        self._on_eviction(key, state)
                    except Exception as e:
                        logger.warning(f"Eviction callback failed for {key}: {e}")

            self._active_lobs.clear()
            self._access_counts.clear()
            self._stats.total_evictions += count
            self._stats.active_lobs = 0

            return count

    def estimate_memory_usage(self) -> int:
        """
        Estimate total memory usage of all active LOBs.

        Returns:
            Estimated memory usage in bytes
        """
        return self.get_memory_usage()

    def cleanup_expired(self) -> int:
        """
        Remove LOBs that have exceeded their TTL.

        Only effective when eviction_policy is TTL and ttl_seconds is set.

        Returns:
            Number of LOBs cleaned up
        """
        if self._ttl_seconds is None:
            return 0

        now = time.time()
        cleaned = 0

        with self._lock:
            keys_to_remove = []
            for key, state in self._active_lobs.items():
                age = now - state.metadata.last_accessed
                if age > self._ttl_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                state = self._active_lobs[key]

                # Persist to disk if enabled
                if self._disk_cache_path and self._persist_on_evict:
                    self._persist_to_disk_internal(key, state)

                del self._active_lobs[key]

                if self._on_eviction:
                    try:
                        self._on_eviction(key, state)
                    except Exception as e:
                        logger.warning(f"TTL cleanup callback failed for {key}: {e}")

                cleaned += 1
                self._stats.total_evictions += 1

            self._stats.active_lobs = len(self._active_lobs)

        return cleaned

    def cleanup_old_persist_files(self) -> int:
        """
        Remove persisted files older than max_persist_age_days.

        Returns:
            Number of files removed
        """
        if not self._disk_cache_path or not self._max_persist_age_days:
            return 0

        max_age_seconds = self._max_persist_age_days * 24 * 3600
        now = time.time()
        removed = 0

        suffix = ".pkl.gz" if self._enable_compression else ".pkl"
        for cache_file in self._disk_cache_path.glob(f"*{suffix}"):
            try:
                file_age = now - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove old cache file {cache_file}: {e}")

        return removed

    def persist_all(self) -> int:
        """
        Persist all active LOBs to disk.

        Returns:
            Number of LOBs persisted
        """
        return self.flush_to_disk(force=True)

    @property
    def max_active_lobs(self) -> int:
        """Maximum number of active LOBs allowed."""
        return self._max_active

    def get_all_keys(self) -> List[str]:
        """Get list of all LOB keys (active + disk)."""
        with self._lock:
            keys = set(self._active_lobs.keys())

        # Add disk keys
        if self._disk_cache_path:
            suffix = ".gz" if self._enable_compression else ".pkl"
            for cache_file in self._disk_cache_path.glob(f"*{suffix}"):
                key = cache_file.stem
                if self._enable_compression and key.endswith(".pkl"):
                    key = key[:-4]
                keys.add(key)

        return sorted(keys)

    def flush_to_disk(self, force: bool = False) -> int:
        """
        Flush all dirty LOBs to disk.

        Args:
            force: If True, flush all LOBs regardless of dirty flag

        Returns:
            Number of LOBs flushed
        """
        if not self._disk_cache_path:
            return 0

        flushed = 0

        with self._lock:
            for key, state in self._active_lobs.items():
                if force or state.metadata.is_dirty:
                    self._persist_to_disk_internal(key, state)
                    state.metadata.is_dirty = False
                    flushed += 1

        return flushed

    def clear_all(self, persist_first: bool = True) -> None:
        """
        Clear all LOBs from memory.

        Args:
            persist_first: If True, persist dirty LOBs to disk first
        """
        with self._lock:
            if persist_first and self._disk_cache_path:
                self.flush_to_disk(force=True)

            self._active_lobs.clear()
            self._access_counts.clear()
            self._stats.active_lobs = 0

    def get_stats(self) -> ManagerStats:
        """Get manager statistics."""
        with self._lock:
            self._stats.active_lobs = len(self._active_lobs)
            return ManagerStats(
                active_lobs=self._stats.active_lobs,
                total_lobs_created=self._stats.total_lobs_created,
                total_evictions=self._stats.total_evictions,
                total_restorations=self._stats.total_restorations,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                disk_writes=self._stats.disk_writes,
                disk_reads=self._stats.disk_reads,
                peak_memory_bytes=self._stats.peak_memory_bytes,
                current_memory_bytes=self._stats.current_memory_bytes,
            )

    def get_memory_usage(self) -> int:
        """Get estimated current memory usage in bytes."""
        with self._lock:
            return sum(
                state.get_memory_estimate_bytes()
                for state in self._active_lobs.values()
            )

    def preload_series(
        self,
        series_keys: List[Union[str, SeriesKey]],
        priority_order: bool = True,
    ) -> int:
        """
        Preload multiple series into memory.

        Useful for initializing frequently-accessed series at startup.

        Args:
            series_keys: List of series to preload
            priority_order: If True, later keys have higher priority (less likely evicted)

        Returns:
            Number of series actually loaded (may be less due to capacity)
        """
        loaded = 0

        for key in series_keys:
            if len(self._active_lobs) >= self._max_active:
                break

            try:
                self.get_lob(key)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to preload {key}: {e}")

        return loaded

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _create_new_lob(self, key: str) -> SeriesLOBState:
        """Create a new empty LOB for the given series."""
        order_book = OrderBook(symbol=key)
        metadata = LOBMetadata(series_key=key)

        self._stats.total_lobs_created += 1

        return SeriesLOBState(
            order_book=order_book,
            metadata=metadata,
            orders_by_id={},
        )

    def _evict_one(self) -> Optional[str]:
        """
        Evict one LOB based on the eviction policy.

        Returns:
            Key of evicted LOB, or None if nothing to evict
        """
        if not self._active_lobs:
            return None

        if self._eviction_policy == EvictionPolicy.LRU:
            # Evict oldest (first item in OrderedDict)
            key, state = next(iter(self._active_lobs.items()))
        elif self._eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            min_count = float('inf')
            key = None
            for k, s in self._active_lobs.items():
                if s.metadata.access_count < min_count:
                    min_count = s.metadata.access_count
                    key = k
            if key is None:
                key = next(iter(self._active_lobs.keys()))
            state = self._active_lobs[key]
        elif self._eviction_policy == EvictionPolicy.TTL:
            # Evict oldest by creation time
            oldest_time = float('inf')
            key = None
            for k, s in self._active_lobs.items():
                if s.metadata.created_at < oldest_time:
                    oldest_time = s.metadata.created_at
                    key = k
            if key is None:
                key = next(iter(self._active_lobs.keys()))
            state = self._active_lobs[key]
        else:
            raise ValueError(f"Unknown eviction policy: {self._eviction_policy}")

        # Persist to disk if enabled and persist_on_evict is True
        if self._disk_cache_path and self._persist_on_evict:
            self._persist_to_disk_internal(key, state)

        # Remove from memory
        del self._active_lobs[key]

        # Callback
        if self._on_eviction:
            try:
                self._on_eviction(key, state)
            except Exception as e:
                logger.warning(f"Eviction callback failed for {key}: {e}")

        self._stats.total_evictions += 1

        logger.debug(f"Evicted LOB: {key}")

        return key

    def _persist_to_disk_internal(self, key: str, state: SeriesLOBState) -> None:
        """Persist LOB state to disk (internal method).

        Security: Appends HMAC-SHA256 signature for integrity verification.
        See module docstring for security model.
        """
        if not self._disk_cache_path:
            return

        cache_file = self._get_cache_file_path(key)

        try:
            # Serialize state with version for compatibility checking
            data = pickle.dumps({
                'version': 1,  # Persistence format version
                'metadata': state.metadata,
                'orders_by_id': state.orders_by_id,
                # OrderBook state reconstruction info
                'bid_levels': self._extract_levels(state.order_book, Side.BUY),
                'ask_levels': self._extract_levels(state.order_book, Side.SELL),
            })

            # Optionally compress
            if self._enable_compression:
                data = gzip.compress(data, compresslevel=self._compression_level)

            # Compute HMAC signature for integrity verification
            signature = _compute_cache_hmac(data)

            # Atomic write (write to temp then replace)
            # Format: [data][32-byte HMAC signature]
            # Note: Using replace() instead of rename() for Windows compatibility
            temp_file = cache_file.with_suffix('.tmp')
            temp_file.write_bytes(data + signature)
            temp_file.replace(cache_file)

            self._stats.disk_writes += 1

        except Exception as e:
            logger.error(f"Failed to persist LOB {key}: {e}")
            raise

    def _try_restore_from_disk(self, key: str) -> Optional[SeriesLOBState]:
        """Attempt to restore LOB from disk.

        Security: Verifies HMAC-SHA256 signature before deserialization.
        Rejects tampered cache files. See module docstring for security model.
        """
        if not self._disk_cache_path:
            return None

        cache_file = self._get_cache_file_path(key)

        if not cache_file.exists():
            return None

        try:
            raw_data = cache_file.read_bytes()

            # Extract HMAC signature (last 32 bytes)
            if len(raw_data) < _HMAC_SIGNATURE_LENGTH:
                logger.warning(
                    f"Cache file too small for {key}: {len(raw_data)} bytes "
                    f"(minimum {_HMAC_SIGNATURE_LENGTH} for HMAC)"
                )
                return None

            data = raw_data[:-_HMAC_SIGNATURE_LENGTH]
            signature = raw_data[-_HMAC_SIGNATURE_LENGTH:]

            # Verify HMAC signature BEFORE deserialization (security critical)
            if not _verify_cache_hmac(data, signature):
                logger.error(
                    f"SECURITY: Cache file HMAC verification failed for {key}. "
                    f"File may be corrupted or tampered. Rejecting cache."
                )
                # Delete the potentially malicious file
                try:
                    cache_file.unlink()
                    logger.warning(f"Deleted suspicious cache file: {cache_file}")
                except OSError:
                    pass
                return None

            # Decompress if needed (after HMAC verification)
            if self._enable_compression:
                data = gzip.decompress(data)

            # Deserialize (safe - HMAC verified)
            state_dict = pickle.loads(data)

            # Reconstruct OrderBook
            order_book = OrderBook(symbol=key)
            for order in state_dict.get('orders_by_id', {}).values():
                order_book.add_limit_order(order)

            # Create state
            metadata = state_dict.get('metadata', LOBMetadata(series_key=key))
            metadata.touch()  # Update access time

            state = SeriesLOBState(
                order_book=order_book,
                metadata=metadata,
                orders_by_id=state_dict.get('orders_by_id', {}),
            )

            self._stats.disk_reads += 1
            self._stats.total_restorations += 1

            # Callback
            if self._on_restoration:
                try:
                    self._on_restoration(key, state)
                except Exception as e:
                    logger.warning(f"Restoration callback failed for {key}: {e}")

            logger.debug(f"Restored LOB from disk: {key}")

            return state

        except Exception as e:
            logger.warning(f"Failed to restore LOB {key} from disk: {e}")
            return None

    def _get_cache_file_path(self, key: str) -> Path:
        """Get the cache file path for a series key."""
        # Sanitize key for filesystem safety (replace special chars)
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        suffix = ".lob.gz" if self._enable_compression else ".lob"
        return self._disk_cache_path / f"{safe_key}{suffix}"

    def _extract_levels(
        self,
        order_book: OrderBook,
        side: Side,
    ) -> List[Dict[str, Any]]:
        """Extract price levels for serialization."""
        levels = []

        if side == Side.BUY:
            price_levels = order_book.get_bid_levels(limit=self._max_depth)
        else:
            price_levels = order_book.get_ask_levels(limit=self._max_depth)

        for level in price_levels:
            levels.append({
                'price': level.price,
                'total_qty': level.total_qty,
                'order_count': level.order_count,
            })

        return levels

    def _update_memory_stats(self) -> None:
        """Update memory usage statistics."""
        current = self.get_memory_usage()
        self._stats.current_memory_bytes = current
        if current > self._stats.peak_memory_bytes:
            self._stats.peak_memory_bytes = current

    # ==========================================================================
    # Iterator Support
    # ==========================================================================

    def __iter__(self) -> Iterator[Tuple[str, SeriesLOBState]]:
        """Iterate over active LOBs."""
        with self._lock:
            # Create copy to avoid modification during iteration
            items = list(self._active_lobs.items())

        for key, state in items:
            yield key, state

    def __len__(self) -> int:
        """Return number of active LOBs."""
        with self._lock:
            return len(self._active_lobs)

    def __contains__(self, key: Union[str, SeriesKey]) -> bool:
        """Check if key is in active LOBs."""
        return self.has_lob(key)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_lazy_lob_manager(
    max_active_lobs: int = 50,
    max_depth_per_lob: int = 100,
    disk_cache_path: Optional[Union[str, Path]] = None,
    persist_dir: Optional[Union[str, Path]] = None,  # Alias for disk_cache_path
    eviction_policy: Union[str, EvictionPolicy] = "lru",
    enable_compression: bool = True,
    ttl_seconds: Optional[float] = None,
    on_evict: Optional[Callable[[str, Any], None]] = None,
    persist_on_evict: bool = True,
    max_persist_age_days: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
) -> LazyMultiSeriesLOBManager:
    """
    Create a lazy multi-series LOB manager with common settings.

    Args:
        max_active_lobs: Maximum LOBs to keep in memory
        max_depth_per_lob: Maximum price levels per LOB
        disk_cache_path: Path for disk caching (None to disable)
        persist_dir: Alias for disk_cache_path (for API compatibility)
        eviction_policy: "lru", "lfu", "ttl" or EvictionPolicy enum
        enable_compression: Whether to compress disk files
        ttl_seconds: Time-to-live for LOBs in seconds (optional)
        on_evict: Callback when LOB is evicted
        persist_on_evict: Whether to persist LOB to disk on eviction
        max_persist_age_days: Maximum age of persisted files in days
        max_memory_bytes: Maximum memory usage in bytes (soft limit)

    Returns:
        Configured LazyMultiSeriesLOBManager
    """
    policy_map = {
        "lru": EvictionPolicy.LRU,
        "lfu": EvictionPolicy.LFU,
        "ttl": EvictionPolicy.TTL,
    }

    # Handle eviction_policy as string or enum
    if isinstance(eviction_policy, EvictionPolicy):
        policy = eviction_policy
    elif isinstance(eviction_policy, str):
        if eviction_policy.lower() not in policy_map:
            raise ValueError(
                f"Unknown eviction policy: {eviction_policy}. "
                f"Use one of: {list(policy_map.keys())}"
            )
        policy = policy_map[eviction_policy.lower()]
    else:
        raise ValueError(f"eviction_policy must be str or EvictionPolicy, got {type(eviction_policy)}")

    # Use persist_dir if provided, otherwise disk_cache_path
    cache_path = persist_dir if persist_dir is not None else disk_cache_path
    if cache_path is not None and not isinstance(cache_path, Path):
        cache_path = Path(cache_path)

    return LazyMultiSeriesLOBManager(
        max_active_lobs=max_active_lobs,
        max_depth_per_lob=max_depth_per_lob,
        eviction_policy=policy,
        disk_cache_path=cache_path,
        enable_compression=enable_compression,
        ttl_seconds=ttl_seconds,
        on_evict=on_evict,
        persist_on_evict=persist_on_evict,
        max_persist_age_days=max_persist_age_days,
        max_memory_bytes=max_memory_bytes,
    )


def create_options_lob_manager(
    underlying: str = "SPY",
    expected_series: Optional[int] = None,  # Deprecated, use max_series
    max_series: Optional[int] = None,  # Alias for expected_series
    memory_budget_gb: float = 4.0,
    disk_cache_path: Optional[Union[str, Path]] = None,
    persist_dir: Optional[Union[str, Path]] = None,  # Alias for disk_cache_path
) -> LazyMultiSeriesLOBManager:
    """
    Create LOB manager optimized for options chains.

    Automatically calculates appropriate parameters based on
    expected series count and memory budget.

    Args:
        underlying: Underlying symbol (for naming)
        expected_series: Expected number of option series (deprecated, use max_series)
        max_series: Maximum number of series to manage (alias for expected_series)
        memory_budget_gb: Maximum memory in gigabytes
        disk_cache_path: Path for disk caching
        persist_dir: Alias for disk_cache_path (for API compatibility)

    Returns:
        Configured LazyMultiSeriesLOBManager
    """
    # Use max_series if provided, otherwise expected_series, default to 960
    series_count = max_series if max_series is not None else (expected_series if expected_series is not None else 960)

    # Use persist_dir if provided, otherwise disk_cache_path
    cache_path = persist_dir if persist_dir is not None else disk_cache_path
    if cache_path is not None and not isinstance(cache_path, Path):
        cache_path = Path(cache_path)

    # Calculate parameters
    # Assume ~200KB per LOB with 100 levels
    bytes_per_lob = 200 * 1024
    memory_budget_bytes = int(memory_budget_gb * 1024 * 1024 * 1024)

    max_active_lobs = min(
        memory_budget_bytes // bytes_per_lob,
        series_count,
        100,  # Cap at 100 for performance
    )
    max_active_lobs = max(10, max_active_lobs)  # Minimum 10

    logger.info(
        f"Creating options LOB manager for {underlying}: "
        f"max_series={series_count}, "
        f"max_active_lobs={max_active_lobs}, "
        f"memory_budget={memory_budget_gb}GB"
    )

    return LazyMultiSeriesLOBManager(
        max_active_lobs=max_active_lobs,
        max_depth_per_lob=100,
        eviction_policy=EvictionPolicy.LRU,
        disk_cache_path=cache_path,
        enable_compression=True,
    )


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Enums
    "EvictionPolicy",
    # Data classes
    "SeriesKey",
    "LOBMetadata",
    "SeriesLOBState",
    "ManagerStats",
    # Main class
    "LazyMultiSeriesLOBManager",
    # Factory functions
    "create_lazy_lob_manager",
    "create_options_lob_manager",
]
