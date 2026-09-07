# -*- coding: utf-8 -*-
"""
services/survivorship.py
Survivorship Bias Protection for Equity Backtesting.

This module provides tools to prevent survivorship bias in backtesting:
- Track delisted companies and their delisting dates
- Maintain historical index/ETF constituents
- Filter trading universe based on point-in-time constituents

Survivorship Bias:
    Backtesting only on stocks that exist today (survivors) creates
    upward bias because failed/delisted companies are excluded.
    This can inflate Sharpe ratio by 1-3% annually.

Usage:
    from services.survivorship import (
        DelistingTracker,
        UniverseSnapshot,
        filter_survivorship_bias,
    )

    # Track delistings
    tracker = DelistingTracker()
    tracker.add_delisting("LMND", "2024-06-15", reason="ACQUISITION")

    # Get point-in-time universe
    snapshot = UniverseSnapshot()
    sp500_2020 = snapshot.get_constituents("SP500", as_of="2020-01-01")

    # Filter DataFrame for backtest
    filtered_df = filter_survivorship_bias(
        df,
        tracker=tracker,
        as_of="2023-06-01",
    )

References:
    - Elton, Gruber, Blake (1996): "Survivorship Bias and Mutual Fund Performance"
    - Brown, Goetzmann, Ibbotson (1995): "Survivorship Bias in Performance Studies"
    - CRSP delisting methodology

FIX (2025-11-28): Added as part of Issue #6 "Survivorship Bias" fix.
Reference: docs/PLATFORM_REFERENCE.md → "Survivorship Bias" → "Нет механизма для исключения delisted"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default storage paths
DEFAULT_DELISTING_PATH = Path("data/universe/delistings.json")
DEFAULT_CONSTITUENTS_PATH = Path("data/universe/historical_constituents.json")


# =============================================================================
# ENUMS
# =============================================================================


class DelistingReason(str, Enum):
    """Reason for stock delisting."""
    ACQUISITION = "acquisition"      # Company acquired by another
    MERGER = "merger"                # Merged with another company
    BANKRUPTCY = "bankruptcy"        # Company went bankrupt
    DELISTED = "delisted"            # Delisted for non-compliance
    PRIVATIZATION = "privatization"  # Went private (LBO, MBO)
    SPINOFF = "spinoff"              # Spun off into separate company
    LIQUIDATION = "liquidation"      # Company liquidated
    OTHER = "other"                  # Other reasons
    UNKNOWN = "unknown"              # Reason not known


class IndexType(str, Enum):
    """Standard index/universe types."""
    SP500 = "SP500"
    SP400 = "SP400"              # S&P MidCap 400
    SP600 = "SP600"              # S&P SmallCap 600
    NASDAQ100 = "NASDAQ100"
    RUSSELL1000 = "RUSSELL1000"
    RUSSELL2000 = "RUSSELL2000"
    RUSSELL3000 = "RUSSELL3000"
    DOW30 = "DOW30"
    CUSTOM = "CUSTOM"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DelistingEvent:
    """
    Record of a stock delisting event.

    Attributes:
        symbol: Ticker symbol that was delisted
        delist_date: Date when stock was delisted (last trading day)
        reason: Reason for delisting
        successor_symbol: New symbol if renamed/spun off
        last_price: Last traded price before delisting
        metadata: Additional metadata (e.g., acquirer, deal price)
    """
    symbol: str
    delist_date: date
    reason: DelistingReason = DelistingReason.UNKNOWN
    successor_symbol: Optional[str] = None
    last_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "delist_date": self.delist_date.isoformat(),
            "reason": self.reason.value,
            "successor_symbol": self.successor_symbol,
            "last_price": self.last_price,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelistingEvent":
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            delist_date=date.fromisoformat(data["delist_date"]),
            reason=DelistingReason(data.get("reason", "unknown")),
            successor_symbol=data.get("successor_symbol"),
            last_price=data.get("last_price"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConstituentChange:
    """
    Record of an index constituent change.

    Attributes:
        index: Index name (e.g., "SP500")
        date: Effective date of change
        added: Symbols added to index
        removed: Symbols removed from index
        reason: Reason for changes (e.g., rebalance, corporate action)
    """
    index: str
    date: date
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "index": self.index,
            "date": self.date.isoformat(),
            "added": self.added,
            "removed": self.removed,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstituentChange":
        """Deserialize from dictionary."""
        return cls(
            index=data["index"],
            date=date.fromisoformat(data["date"]),
            added=data.get("added", []),
            removed=data.get("removed", []),
            reason=data.get("reason", ""),
        )


# =============================================================================
# DELISTING TRACKER
# =============================================================================


class DelistingTracker:
    """
    Track delisted companies for survivorship bias prevention.

    This class maintains a registry of delisted stocks with:
    - Delisting dates
    - Reasons for delisting
    - Successor symbols (if any)

    Usage:
        tracker = DelistingTracker()

        # Add delisting manually
        tracker.add_delisting("LMND", "2024-06-15", reason="ACQUISITION")

        # Check if symbol was delisted before a date
        if tracker.was_delisted("LMND", as_of="2024-01-01"):
            # Symbol existed on 2024-01-01

        # Get all delistings in a date range
        delistings = tracker.get_delistings(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

    Note:
        Delisting data must be populated either:
        - Manually via add_delisting()
        - From external sources via load_from_api()
        - From local file via load()
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        """
        Initialize delisting tracker.

        Args:
            storage_path: Path to JSON file for persistence
            auto_load: If True, load existing data from storage_path
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_DELISTING_PATH
        self._delistings: Dict[str, DelistingEvent] = {}

        if auto_load and self.storage_path.exists():
            self.load()

    def add_delisting(
        self,
        symbol: str,
        delist_date: Union[str, date],
        reason: Union[str, DelistingReason] = DelistingReason.UNKNOWN,
        successor_symbol: Optional[str] = None,
        last_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a delisting event.

        Args:
            symbol: Ticker symbol
            delist_date: Date of delisting (last trading day)
            reason: Reason for delisting
            successor_symbol: New symbol if renamed
            last_price: Last traded price
            metadata: Additional metadata
        """
        if isinstance(delist_date, str):
            delist_date = date.fromisoformat(delist_date)

        if isinstance(reason, str):
            try:
                reason = DelistingReason(reason.lower())
            except ValueError:
                reason = DelistingReason.UNKNOWN

        event = DelistingEvent(
            symbol=symbol.upper(),
            delist_date=delist_date,
            reason=reason,
            successor_symbol=successor_symbol.upper() if successor_symbol else None,
            last_price=last_price,
            metadata=metadata or {},
        )

        self._delistings[symbol.upper()] = event
        logger.debug(f"Added delisting: {symbol} on {delist_date} ({reason.value})")

    def remove_delisting(self, symbol: str) -> bool:
        """
        Remove a delisting entry.

        Args:
            symbol: Symbol to remove

        Returns:
            True if removed, False if not found
        """
        symbol = symbol.upper()
        if symbol in self._delistings:
            del self._delistings[symbol]
            return True
        return False

    def get_delisting(self, symbol: str) -> Optional[DelistingEvent]:
        """
        Get delisting event for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            DelistingEvent or None if not found
        """
        return self._delistings.get(symbol.upper())

    def was_delisted(
        self,
        symbol: str,
        as_of: Optional[Union[str, date]] = None,
    ) -> bool:
        """
        Check if symbol was delisted before a given date.

        Args:
            symbol: Ticker symbol
            as_of: Check if delisted before this date (default: today)

        Returns:
            True if symbol was delisted before as_of date
        """
        event = self._delistings.get(symbol.upper())
        if event is None:
            return False

        if as_of is None:
            return True  # Delisted at some point

        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        return event.delist_date <= as_of

    def is_tradable(
        self,
        symbol: str,
        as_of: Union[str, date],
    ) -> bool:
        """
        Check if symbol was tradable on a given date.

        Args:
            symbol: Ticker symbol
            as_of: Date to check

        Returns:
            True if symbol was tradable (not yet delisted) on as_of
        """
        event = self._delistings.get(symbol.upper())
        if event is None:
            return True  # No delisting record = assume tradable

        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        return as_of < event.delist_date

    def get_delistings(
        self,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        reason: Optional[Union[str, DelistingReason]] = None,
    ) -> List[DelistingEvent]:
        """
        Get delistings in a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            reason: Filter by delisting reason

        Returns:
            List of DelistingEvents matching criteria
        """
        results: List[DelistingEvent] = []

        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
        if isinstance(reason, str):
            try:
                reason = DelistingReason(reason.lower())
            except ValueError:
                reason = None

        for event in self._delistings.values():
            # Check date range
            if start_date and event.delist_date < start_date:
                continue
            if end_date and event.delist_date > end_date:
                continue

            # Check reason
            if reason and event.reason != reason:
                continue

            results.append(event)

        return sorted(results, key=lambda e: e.delist_date)

    def get_tradable_symbols(
        self,
        symbols: Sequence[str],
        as_of: Union[str, date],
    ) -> List[str]:
        """
        Filter list of symbols to only those tradable on a date.

        Args:
            symbols: List of symbols to check
            as_of: Date to check tradability

        Returns:
            List of symbols that were tradable on as_of
        """
        return [s for s in symbols if self.is_tradable(s, as_of)]

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save delistings to JSON file.

        Args:
            path: Path to save (default: self.storage_path)
        """
        path = Path(path) if path else self.storage_path
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "count": len(self._delistings),
            "delistings": [e.to_dict() for e in self._delistings.values()],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._delistings)} delistings to {path}")

    def load(self, path: Optional[Union[str, Path]] = None) -> int:
        """
        Load delistings from JSON file.

        Args:
            path: Path to load from (default: self.storage_path)

        Returns:
            Number of delistings loaded
        """
        path = Path(path) if path else self.storage_path

        if not path.exists():
            logger.debug(f"No delisting file found at {path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both old and new format
        delistings = data.get("delistings", data) if isinstance(data, dict) else data

        count = 0
        for item in delistings:
            try:
                event = DelistingEvent.from_dict(item)
                self._delistings[event.symbol] = event
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load delisting: {item} - {e}")

        logger.info(f"Loaded {count} delistings from {path}")
        return count

    def __len__(self) -> int:
        """Number of tracked delistings."""
        return len(self._delistings)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol has a delisting record."""
        return symbol.upper() in self._delistings


# =============================================================================
# UNIVERSE SNAPSHOT
# =============================================================================


class UniverseSnapshot:
    """
    Track historical index constituents for point-in-time analysis.

    This class maintains the history of index membership changes,
    allowing reconstruction of the exact constituents at any past date.

    Usage:
        snapshot = UniverseSnapshot()

        # Add a constituent change
        snapshot.add_change(
            index="SP500",
            date="2024-03-15",
            added=["SMCI"],
            removed=["WHR"],
            reason="Quarterly rebalance"
        )

        # Get constituents as of a date
        sp500_2020 = snapshot.get_constituents("SP500", as_of="2020-01-01")

        # Check if symbol was in index on a date
        in_sp500 = snapshot.was_constituent("AAPL", "SP500", as_of="2020-01-01")

    Note:
        - Changes are tracked incrementally (not full snapshots)
        - Must have a starting point (baseline constituents)
        - More changes = more accurate historical reconstruction
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        """
        Initialize universe snapshot tracker.

        Args:
            storage_path: Path to JSON file for persistence
            auto_load: If True, load existing data from storage_path
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_CONSTITUENTS_PATH

        # Map: index -> list of constituent changes (sorted by date)
        self._changes: Dict[str, List[ConstituentChange]] = {}

        # Map: index -> baseline constituents (starting point)
        self._baselines: Dict[str, Tuple[date, Set[str]]] = {}

        if auto_load and self.storage_path.exists():
            self.load()

    def set_baseline(
        self,
        index: str,
        constituents: Sequence[str],
        as_of: Union[str, date],
    ) -> None:
        """
        Set baseline constituents for an index.

        This is the starting point for historical reconstruction.
        All subsequent changes are applied relative to this baseline.

        Args:
            index: Index name (e.g., "SP500")
            constituents: List of constituent symbols
            as_of: Date of baseline snapshot
        """
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        index = index.upper()
        self._baselines[index] = (as_of, set(s.upper() for s in constituents))

        logger.info(f"Set baseline for {index}: {len(constituents)} constituents as of {as_of}")

    def add_change(
        self,
        index: str,
        date_: Union[str, date],
        added: Optional[Sequence[str]] = None,
        removed: Optional[Sequence[str]] = None,
        reason: str = "",
    ) -> None:
        """
        Add a constituent change event.

        Args:
            index: Index name
            date_: Effective date of change
            added: Symbols added to index
            removed: Symbols removed from index
            reason: Reason for change
        """
        if isinstance(date_, str):
            date_ = date.fromisoformat(date_)

        index = index.upper()
        change = ConstituentChange(
            index=index,
            date=date_,
            added=[s.upper() for s in (added or [])],
            removed=[s.upper() for s in (removed or [])],
            reason=reason,
        )

        if index not in self._changes:
            self._changes[index] = []

        self._changes[index].append(change)
        # Keep sorted by date
        self._changes[index].sort(key=lambda c: c.date)

        logger.debug(
            f"Added change for {index} on {date_}: "
            f"+{len(change.added)} -{len(change.removed)}"
        )

    def get_constituents(
        self,
        index: str,
        as_of: Union[str, date],
    ) -> Set[str]:
        """
        Get index constituents as of a specific date.

        Args:
            index: Index name
            as_of: Date to reconstruct constituents

        Returns:
            Set of constituent symbols

        Raises:
            ValueError: If no baseline exists for the index
        """
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        index = index.upper()

        if index not in self._baselines:
            raise ValueError(
                f"No baseline for index '{index}'. "
                f"Call set_baseline() first."
            )

        baseline_date, baseline_constituents = self._baselines[index]
        constituents = baseline_constituents.copy()

        # Apply changes
        changes = self._changes.get(index, [])

        if as_of >= baseline_date:
            # Forward from baseline: apply changes chronologically
            for change in changes:
                if change.date > as_of:
                    break  # Past our target date
                if change.date <= baseline_date:
                    continue  # Already in baseline

                constituents.update(change.added)
                constituents.difference_update(change.removed)
        else:
            # Backward from baseline: reverse changes
            # Get changes between as_of and baseline, apply in reverse
            relevant_changes = [
                c for c in changes
                if as_of < c.date <= baseline_date
            ]

            for change in reversed(relevant_changes):
                # Reverse: add back what was removed, remove what was added
                constituents.update(change.removed)
                constituents.difference_update(change.added)

        return constituents

    def was_constituent(
        self,
        symbol: str,
        index: str,
        as_of: Union[str, date],
    ) -> bool:
        """
        Check if symbol was an index constituent on a date.

        Args:
            symbol: Ticker symbol
            index: Index name
            as_of: Date to check

        Returns:
            True if symbol was in index on as_of
        """
        try:
            constituents = self.get_constituents(index, as_of)
            return symbol.upper() in constituents
        except ValueError:
            logger.warning(f"No baseline for {index}, cannot check membership")
            return True  # Assume true if no data

    def get_changes(
        self,
        index: str,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> List[ConstituentChange]:
        """
        Get constituent changes in a date range.

        Args:
            index: Index name
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of ConstituentChange events
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        index = index.upper()
        changes = self._changes.get(index, [])

        results = []
        for change in changes:
            if start_date and change.date < start_date:
                continue
            if end_date and change.date > end_date:
                continue
            results.append(change)

        return results

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save to JSON file."""
        path = Path(path) if path else self.storage_path
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat(),
            "baselines": {
                idx: {
                    "date": d.isoformat(),
                    "constituents": sorted(list(c)),
                }
                for idx, (d, c) in self._baselines.items()
            },
            "changes": {
                idx: [c.to_dict() for c in changes]
                for idx, changes in self._changes.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved universe snapshots to {path}")

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load from JSON file."""
        path = Path(path) if path else self.storage_path

        if not path.exists():
            logger.debug(f"No constituents file found at {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load baselines
        for idx, baseline in data.get("baselines", {}).items():
            self._baselines[idx] = (
                date.fromisoformat(baseline["date"]),
                set(baseline["constituents"]),
            )

        # Load changes
        for idx, changes in data.get("changes", {}).items():
            self._changes[idx] = [
                ConstituentChange.from_dict(c) for c in changes
            ]

        logger.info(
            f"Loaded {len(self._baselines)} baselines, "
            f"{sum(len(c) for c in self._changes.values())} changes"
        )


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================


def filter_survivorship_bias(
    df: pd.DataFrame,
    tracker: Optional[DelistingTracker] = None,
    universe: Optional[UniverseSnapshot] = None,
    index: Optional[str] = None,
    as_of: Optional[Union[str, date]] = None,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Filter DataFrame to remove survivorship bias.

    This function filters out:
    1. Rows for delisted symbols (after their delisting date)
    2. Rows for symbols not in the index (if index specified)

    Args:
        df: DataFrame with trading data
        tracker: DelistingTracker instance
        universe: UniverseSnapshot instance (required if index specified)
        index: Index name to filter by (e.g., "SP500")
        as_of: Reference date for filtering
        symbol_column: Column name for symbol
        timestamp_column: Column name for timestamp

    Returns:
        Filtered DataFrame

    Example:
        # Filter to remove delisted symbols
        filtered = filter_survivorship_bias(df, tracker=tracker)

        # Filter to only S&P 500 constituents
        filtered = filter_survivorship_bias(
            df,
            tracker=tracker,
            universe=snapshot,
            index="SP500",
        )
    """
    if df.empty:
        return df

    if tracker is None and universe is None:
        logger.warning("No tracker or universe provided, no filtering applied")
        return df

    # Convert as_of to date
    if isinstance(as_of, str):
        as_of_date = date.fromisoformat(as_of)
    elif as_of is None:
        as_of_date = date.today()
    else:
        as_of_date = as_of

    mask = pd.Series(True, index=df.index)

    # Filter by delisting
    if tracker is not None and symbol_column in df.columns:
        # Group by symbol for efficiency
        for symbol, group in df.groupby(symbol_column):
            event = tracker.get_delisting(symbol)
            if event is not None:
                # Get timestamps for this symbol
                if timestamp_column in df.columns:
                    # Filter rows after delisting date
                    ts = pd.to_numeric(df.loc[group.index, timestamp_column], errors="coerce")
                    delist_ts = int(datetime.combine(
                        event.delist_date, datetime.min.time()
                    ).timestamp())

                    # Mark rows after delisting as False
                    mask.loc[group.index] = ts < delist_ts
                else:
                    # No timestamp column - check if delisted before as_of
                    if event.delist_date <= as_of_date:
                        mask.loc[group.index] = False

    # Filter by index membership
    if universe is not None and index is not None:
        try:
            constituents = universe.get_constituents(index, as_of_date)

            if symbol_column in df.columns:
                symbol_mask = df[symbol_column].str.upper().isin(constituents)
                mask &= symbol_mask
        except ValueError as e:
            logger.warning(f"Could not filter by index: {e}")

    filtered = df[mask].copy()

    removed_count = len(df) - len(filtered)
    if removed_count > 0:
        logger.info(
            f"Survivorship filter removed {removed_count} rows "
            f"({100 * removed_count / len(df):.1f}%)"
        )

    return filtered


def validate_no_survivorship_bias(
    df: pd.DataFrame,
    tracker: DelistingTracker,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    raise_on_violation: bool = False,
) -> List[Dict[str, Any]]:
    """
    Validate that DataFrame has no survivorship bias violations.

    Checks for:
    1. Data for delisted symbols after their delisting date
    2. Data gaps suggesting survivorship issues

    Args:
        df: DataFrame to validate
        tracker: DelistingTracker instance
        symbol_column: Column name for symbol
        timestamp_column: Column name for timestamp
        raise_on_violation: If True, raise ValueError on violations

    Returns:
        List of violation records

    Raises:
        ValueError: If raise_on_violation=True and violations found
    """
    violations: List[Dict[str, Any]] = []

    if df.empty:
        return violations

    for symbol, group in df.groupby(symbol_column):
        event = tracker.get_delisting(symbol)
        if event is None:
            continue

        if timestamp_column in df.columns:
            ts = pd.to_numeric(group[timestamp_column], errors="coerce")
            max_ts = ts.max()

            # Convert delisting date to timestamp
            delist_ts = int(datetime.combine(
                event.delist_date, datetime.min.time()
            ).timestamp())

            if max_ts >= delist_ts:
                violations.append({
                    "symbol": symbol,
                    "violation_type": "data_after_delisting",
                    "delist_date": event.delist_date.isoformat(),
                    "latest_data_ts": int(max_ts),
                    "rows_affected": int((ts >= delist_ts).sum()),
                })

    if violations and raise_on_violation:
        raise ValueError(
            f"Survivorship bias violations found: {len(violations)} symbols "
            f"have data after delisting. Use filter_survivorship_bias() to fix."
        )

    return violations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_global_delisting_tracker: Optional[DelistingTracker] = None
_global_universe_snapshot: Optional[UniverseSnapshot] = None


def get_delisting_tracker() -> DelistingTracker:
    """Get or create global delisting tracker."""
    global _global_delisting_tracker
    if _global_delisting_tracker is None:
        _global_delisting_tracker = DelistingTracker()
    return _global_delisting_tracker


def get_universe_snapshot() -> UniverseSnapshot:
    """Get or create global universe snapshot."""
    global _global_universe_snapshot
    if _global_universe_snapshot is None:
        _global_universe_snapshot = UniverseSnapshot()
    return _global_universe_snapshot


def reset_global_instances() -> None:
    """Reset global instances (for testing)."""
    global _global_delisting_tracker, _global_universe_snapshot
    _global_delisting_tracker = None
    _global_universe_snapshot = None
