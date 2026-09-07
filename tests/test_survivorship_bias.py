# -*- coding: utf-8 -*-
"""
tests/test_survivorship_bias.py
Tests for survivorship bias protection module.

FIX (2025-11-28): Tests for Issue #6 "Survivorship Bias"
Reference: docs/PLATFORM_REFERENCE.md → Issue #6

These tests verify:
1. DelistingTracker functionality
2. UniverseSnapshot functionality
3. Filter functions work correctly
4. Save/load persistence works
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.survivorship import (
    DelistingTracker,
    DelistingEvent,
    DelistingReason,
    UniverseSnapshot,
    ConstituentChange,
    IndexType,
    filter_survivorship_bias,
    validate_no_survivorship_bias,
    get_delisting_tracker,
    get_universe_snapshot,
    reset_global_instances,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    import tempfile
    import shutil

    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_tracker(temp_dir):
    """Create a DelistingTracker with sample data."""
    tracker = DelistingTracker(storage_path=temp_dir / "delistings.json", auto_load=False)

    # Add some delistings
    tracker.add_delisting("LMND", "2024-06-15", reason="acquisition", last_price=15.50)
    tracker.add_delisting("WHR", "2024-03-01", reason="merger")
    tracker.add_delisting("BANKRUPT", "2023-12-31", reason="bankruptcy")

    return tracker


@pytest.fixture
def sample_snapshot(temp_dir):
    """Create a UniverseSnapshot with sample data."""
    snapshot = UniverseSnapshot(storage_path=temp_dir / "constituents.json", auto_load=False)

    # Set baseline for S&P 500
    baseline_constituents = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "LMND", "WHR"]
    snapshot.set_baseline("SP500", baseline_constituents, "2024-01-01")

    # Add changes
    snapshot.add_change("SP500", "2024-03-15", added=["SMCI"], removed=["WHR"], reason="Rebalance")
    snapshot.add_change(
        "SP500", "2024-06-20", added=["UBER"], removed=["LMND"], reason="Acquisition"
    )

    return snapshot


@pytest.fixture
def sample_trading_df():
    """Create sample trading DataFrame."""
    np.random.seed(42)
    n_days = 100

    symbols = ["AAPL", "MSFT", "LMND", "WHR", "BANKRUPT"]
    rows = []

    for symbol in symbols:
        for i in range(n_days):
            ts = 1704067200 + i * 86400  # 2024-01-01 + i days
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "close": 100.0 + np.random.randn(),
                    "volume": 1000.0,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# TEST: DELISTING TRACKER
# =============================================================================


class TestDelistingTracker:
    """Tests for DelistingTracker."""

    def test_create_tracker(self, temp_dir):
        """Can create a new tracker."""
        tracker = DelistingTracker(storage_path=temp_dir / "test.json", auto_load=False)
        assert len(tracker) == 0

    def test_add_delisting(self, temp_dir):
        """Can add delisting events."""
        tracker = DelistingTracker(storage_path=temp_dir / "test.json", auto_load=False)
        tracker.add_delisting("TEST", "2024-01-15", reason="acquisition")

        assert len(tracker) == 1
        assert "TEST" in tracker

    def test_add_delisting_with_metadata(self, temp_dir):
        """Can add delisting with metadata."""
        tracker = DelistingTracker(storage_path=temp_dir / "test.json", auto_load=False)
        tracker.add_delisting(
            "ACME",
            "2024-06-01",
            reason="acquisition",
            successor_symbol="NEWCO",
            last_price=50.25,
            metadata={"acquirer": "BigCorp", "deal_price": 55.00},
        )

        event = tracker.get_delisting("ACME")
        assert event is not None
        assert event.successor_symbol == "NEWCO"
        assert event.last_price == 50.25
        assert event.metadata["acquirer"] == "BigCorp"

    def test_symbol_case_insensitive(self, temp_dir):
        """Symbol lookup should be case-insensitive."""
        tracker = DelistingTracker(storage_path=temp_dir / "test.json", auto_load=False)
        tracker.add_delisting("test", "2024-01-15")

        assert "TEST" in tracker
        assert "test" in tracker
        assert tracker.get_delisting("Test") is not None

    def test_was_delisted(self, sample_tracker):
        """was_delisted should check date correctly."""
        # LMND delisted 2024-06-15
        assert sample_tracker.was_delisted("LMND", "2024-06-15") is True
        assert sample_tracker.was_delisted("LMND", "2024-06-14") is False
        assert sample_tracker.was_delisted("LMND", "2024-07-01") is True

    def test_is_tradable(self, sample_tracker):
        """is_tradable should be inverse of was_delisted."""
        # LMND delisted 2024-06-15
        assert sample_tracker.is_tradable("LMND", "2024-06-14") is True
        assert sample_tracker.is_tradable("LMND", "2024-06-15") is False

        # Non-delisted symbol
        assert sample_tracker.is_tradable("AAPL", "2024-06-15") is True

    def test_get_delistings_date_range(self, sample_tracker):
        """get_delistings should filter by date range."""
        # All delistings
        all_events = sample_tracker.get_delistings()
        assert len(all_events) == 3

        # Q1 2024 only
        q1_events = sample_tracker.get_delistings(start_date="2024-01-01", end_date="2024-03-31")
        assert len(q1_events) == 1
        assert q1_events[0].symbol == "WHR"

    def test_get_delistings_by_reason(self, sample_tracker):
        """get_delistings should filter by reason."""
        acquisitions = sample_tracker.get_delistings(reason="acquisition")
        assert len(acquisitions) == 1
        assert acquisitions[0].symbol == "LMND"

    def test_get_tradable_symbols(self, sample_tracker):
        """get_tradable_symbols should filter symbol list."""
        all_symbols = ["AAPL", "MSFT", "LMND", "WHR", "BANKRUPT"]

        # As of 2024-01-01:
        # - BANKRUPT delisted 2023-12-31 (NOT tradable)
        # - WHR delisted 2024-03-01 (tradable)
        # - LMND delisted 2024-06-15 (tradable)
        # - AAPL, MSFT (always tradable)
        tradable = sample_tracker.get_tradable_symbols(all_symbols, "2024-01-01")
        assert len(tradable) == 4  # AAPL, MSFT, LMND, WHR (BANKRUPT already delisted)

        # As of 2024-07-01 (after all delistings)
        tradable = sample_tracker.get_tradable_symbols(all_symbols, "2024-07-01")
        assert len(tradable) == 2  # AAPL, MSFT

    def test_save_load(self, temp_dir):
        """Save and load should preserve data."""
        path = temp_dir / "delistings.json"

        tracker = DelistingTracker(storage_path=path, auto_load=False)
        tracker.add_delisting("TEST1", "2024-01-15", reason="acquisition")
        tracker.add_delisting("TEST2", "2024-02-20", reason="bankruptcy")
        tracker.save()

        # Load into new tracker
        tracker2 = DelistingTracker(storage_path=path, auto_load=True)
        assert len(tracker2) == 2
        assert tracker2.get_delisting("TEST1") is not None
        assert tracker2.get_delisting("TEST2").reason == DelistingReason.BANKRUPTCY

    def test_remove_delisting(self, sample_tracker):
        """Can remove delisting entries."""
        assert "LMND" in sample_tracker
        removed = sample_tracker.remove_delisting("LMND")
        assert removed is True
        assert "LMND" not in sample_tracker

        # Non-existent symbol
        removed = sample_tracker.remove_delisting("NONEXISTENT")
        assert removed is False


# =============================================================================
# TEST: UNIVERSE SNAPSHOT
# =============================================================================


class TestUniverseSnapshot:
    """Tests for UniverseSnapshot."""

    def test_create_snapshot(self, temp_dir):
        """Can create a new snapshot."""
        snapshot = UniverseSnapshot(storage_path=temp_dir / "test.json", auto_load=False)
        assert snapshot is not None

    def test_set_baseline(self, temp_dir):
        """Can set baseline constituents."""
        snapshot = UniverseSnapshot(storage_path=temp_dir / "test.json", auto_load=False)
        snapshot.set_baseline("SP500", ["AAPL", "MSFT", "GOOGL"], "2024-01-01")

        constituents = snapshot.get_constituents("SP500", "2024-01-01")
        assert len(constituents) == 3
        assert "AAPL" in constituents

    def test_get_constituents_without_baseline_raises(self, temp_dir):
        """get_constituents without baseline should raise."""
        snapshot = UniverseSnapshot(storage_path=temp_dir / "test.json", auto_load=False)

        with pytest.raises(ValueError, match="No baseline"):
            snapshot.get_constituents("SP500", "2024-01-01")

    def test_add_change_forward(self, sample_snapshot):
        """Changes should be applied forward from baseline."""
        # Baseline 2024-01-01: AAPL, MSFT, GOOGL, AMZN, META, LMND, WHR
        # Change 2024-03-15: +SMCI, -WHR
        # Change 2024-06-20: +UBER, -LMND

        # On baseline
        jan_constituents = sample_snapshot.get_constituents("SP500", "2024-01-01")
        assert "WHR" in jan_constituents
        assert "LMND" in jan_constituents

        # After first change
        apr_constituents = sample_snapshot.get_constituents("SP500", "2024-04-01")
        assert "WHR" not in apr_constituents
        assert "SMCI" in apr_constituents
        assert "LMND" in apr_constituents

        # After both changes
        jul_constituents = sample_snapshot.get_constituents("SP500", "2024-07-01")
        assert "WHR" not in jul_constituents
        assert "LMND" not in jul_constituents
        assert "SMCI" in jul_constituents
        assert "UBER" in jul_constituents

    def test_was_constituent(self, sample_snapshot):
        """was_constituent should check membership correctly."""
        # LMND was constituent until 2024-06-20
        assert sample_snapshot.was_constituent("LMND", "SP500", "2024-01-01") is True
        assert sample_snapshot.was_constituent("LMND", "SP500", "2024-06-19") is True
        assert sample_snapshot.was_constituent("LMND", "SP500", "2024-06-21") is False

    def test_get_changes(self, sample_snapshot):
        """get_changes should return changes in range."""
        all_changes = sample_snapshot.get_changes("SP500")
        assert len(all_changes) == 2

        q1_changes = sample_snapshot.get_changes(
            "SP500", start_date="2024-01-01", end_date="2024-03-31"
        )
        assert len(q1_changes) == 1

    def test_save_load(self, temp_dir):
        """Save and load should preserve data."""
        path = temp_dir / "constituents.json"

        snapshot = UniverseSnapshot(storage_path=path, auto_load=False)
        snapshot.set_baseline("TEST", ["A", "B", "C"], "2024-01-01")
        snapshot.add_change("TEST", "2024-02-01", added=["D"], removed=["A"])
        snapshot.save()

        # Load into new snapshot
        snapshot2 = UniverseSnapshot(storage_path=path, auto_load=True)
        constituents = snapshot2.get_constituents("TEST", "2024-03-01")
        assert "A" not in constituents
        assert "D" in constituents


# =============================================================================
# TEST: FILTER FUNCTIONS
# =============================================================================


class TestFilterFunctions:
    """Tests for filter functions."""

    def test_filter_removes_delisted_symbols(self, sample_tracker, sample_trading_df):
        """filter_survivorship_bias should remove delisted symbols."""
        # BANKRUPT delisted 2023-12-31 (before all data)
        # WHR delisted 2024-03-01
        # LMND delisted 2024-06-15

        filtered = filter_survivorship_bias(
            sample_trading_df,
            tracker=sample_tracker,
        )

        # Should have removed data for delisted symbols after their delist date
        original_count = len(sample_trading_df)
        assert len(filtered) < original_count

        # BANKRUPT should be completely gone (delisted before data starts)
        assert len(filtered[filtered["symbol"] == "BANKRUPT"]) == 0

    def test_filter_with_index(self, sample_tracker, sample_snapshot, sample_trading_df):
        """Filter should also filter by index membership."""
        # Only include SP500 constituents
        filtered = filter_survivorship_bias(
            sample_trading_df,
            tracker=sample_tracker,
            universe=sample_snapshot,
            index="SP500",
            as_of="2024-07-01",  # After all changes
        )

        # AAPL and MSFT should be present
        symbols_in_result = filtered["symbol"].unique()
        assert "AAPL" in symbols_in_result
        assert "MSFT" in symbols_in_result

        # BANKRUPT was never in index
        assert "BANKRUPT" not in symbols_in_result

    def test_filter_empty_dataframe(self, sample_tracker):
        """Filter should handle empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
        filtered = filter_survivorship_bias(empty_df, tracker=sample_tracker)
        assert len(filtered) == 0

    def test_filter_no_tracker_or_universe(self, sample_trading_df):
        """Filter with no tracker/universe should return original."""
        filtered = filter_survivorship_bias(sample_trading_df)
        assert len(filtered) == len(sample_trading_df)


class TestValidation:
    """Tests for validation functions."""

    def test_validate_finds_violations(self, sample_tracker, sample_trading_df):
        """validate_no_survivorship_bias should find violations."""
        violations = validate_no_survivorship_bias(
            sample_trading_df,
            tracker=sample_tracker,
        )

        # Should find violations for LMND, WHR, BANKRUPT
        assert len(violations) >= 1

    def test_validate_raises_on_violation(self, sample_tracker, sample_trading_df):
        """validate_no_survivorship_bias should raise if requested."""
        with pytest.raises(ValueError, match="Survivorship bias violations"):
            validate_no_survivorship_bias(
                sample_trading_df,
                tracker=sample_tracker,
                raise_on_violation=True,
            )

    def test_validate_clean_data(self, sample_tracker):
        """Validation should pass for clean data."""
        # Data only for non-delisted symbols
        clean_df = pd.DataFrame(
            {
                "timestamp": [1704067200, 1704153600],  # 2024-01-01, 2024-01-02
                "symbol": ["AAPL", "AAPL"],
                "close": [150.0, 151.0],
            }
        )

        violations = validate_no_survivorship_bias(clean_df, tracker=sample_tracker)
        assert len(violations) == 0


# =============================================================================
# TEST: GLOBAL INSTANCES
# =============================================================================


class TestGlobalInstances:
    """Tests for global instance management."""

    def test_get_delisting_tracker(self):
        """get_delisting_tracker should return same instance."""
        reset_global_instances()

        tracker1 = get_delisting_tracker()
        tracker2 = get_delisting_tracker()

        assert tracker1 is tracker2

    def test_get_universe_snapshot(self):
        """get_universe_snapshot should return same instance."""
        reset_global_instances()

        snapshot1 = get_universe_snapshot()
        snapshot2 = get_universe_snapshot()

        assert snapshot1 is snapshot2

    def test_reset_global_instances(self):
        """reset_global_instances should clear instances."""
        reset_global_instances()

        tracker1 = get_delisting_tracker()
        reset_global_instances()
        tracker2 = get_delisting_tracker()

        assert tracker1 is not tracker2


# =============================================================================
# TEST: DATA CLASSES
# =============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_delisting_event_to_dict(self):
        """DelistingEvent should serialize to dict."""
        event = DelistingEvent(
            symbol="TEST",
            delist_date=date(2024, 1, 15),
            reason=DelistingReason.ACQUISITION,
            successor_symbol="NEWTEST",
            last_price=100.0,
            metadata={"acquirer": "BigCorp"},
        )

        d = event.to_dict()
        assert d["symbol"] == "TEST"
        assert d["delist_date"] == "2024-01-15"
        assert d["reason"] == "acquisition"
        assert d["successor_symbol"] == "NEWTEST"

    def test_delisting_event_from_dict(self):
        """DelistingEvent should deserialize from dict."""
        d = {
            "symbol": "TEST",
            "delist_date": "2024-01-15",
            "reason": "bankruptcy",
            "last_price": 0.01,
        }

        event = DelistingEvent.from_dict(d)
        assert event.symbol == "TEST"
        assert event.delist_date == date(2024, 1, 15)
        assert event.reason == DelistingReason.BANKRUPTCY

    def test_constituent_change_to_dict(self):
        """ConstituentChange should serialize to dict."""
        change = ConstituentChange(
            index="SP500",
            date=date(2024, 3, 15),
            added=["SMCI"],
            removed=["WHR"],
            reason="Rebalance",
        )

        d = change.to_dict()
        assert d["index"] == "SP500"
        assert d["date"] == "2024-03-15"
        assert "SMCI" in d["added"]


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
