# -*- coding: utf-8 -*-
"""
Test for stale bar temporal causality fix.

Verifies that when data degradation returns a stale bar, it uses the CURRENT
timestamp (not the previous bar's timestamp) to preserve temporal causality.
"""
import io
from decimal import Decimal

import pandas as pd
import pytest

from config import DataDegradationConfig
from impl_offline_data import OfflineCSVBarSource, OfflineCSVConfig


def test_stale_bar_uses_current_timestamp():
    """Verify that stale bars are returned with the current timestamp, not the previous one."""
    # Create test data: 3 bars at t=0, t=60000, t=120000 (1 minute intervals)
    # Note: Use 'ts' as column name to match OfflineCSVConfig default
    csv_data = """ts,symbol,open,high,low,close,volume
0,BTCUSDT,100.0,101.0,99.0,100.5,1000.0
60000,BTCUSDT,100.5,102.0,100.0,101.0,1100.0
120000,BTCUSDT,101.0,103.0,100.5,102.0,1200.0
"""

    # Write to temporary CSV
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        # Configure data degradation with 100% stale probability
        degradation = DataDegradationConfig(
            stale_prob=1.0,  # Always return stale data
            drop_prob=0.0,
            dropout_prob=0.0,
            max_delay_ms=0,
            seed=42,
        )

        cfg = OfflineCSVConfig(
            paths=[temp_path],
            timeframe="1m",
            enforce_closed_bars=False,
        )

        source = OfflineCSVBarSource(cfg, data_degradation=degradation)
        bars = list(source.stream_bars(["BTCUSDT"], interval_ms=60000))

        # We should get 3 bars
        assert len(bars) == 3, f"Expected 3 bars, got {len(bars)}"

        # First bar is always fresh (no previous bar to copy from)
        assert bars[0].ts == 0
        assert float(bars[0].close) == 100.5

        # Second bar should be STALE (copies data from first bar)
        # but uses CURRENT timestamp (60000), not previous timestamp (0)
        assert (
            bars[1].ts == 60000
        ), f"Stale bar should use current timestamp 60000, got {bars[1].ts}"
        assert (
            float(bars[1].close) == 100.5
        ), f"Stale bar should copy previous close price 100.5, got {bars[1].close}"

        # Third bar should be STALE (copies data from second bar which was stale itself)
        # and uses CURRENT timestamp (120000)
        assert (
            bars[2].ts == 120000
        ), f"Stale bar should use current timestamp 120000, got {bars[2].ts}"
        # Since bar[1] was stale and copied bar[0]'s data, bar[2] also gets bar[0]'s data
        assert (
            float(bars[2].close) == 100.5
        ), f"Stale bar should copy previous close price 100.5, got {bars[2].close}"

    finally:
        # Cleanup
        os.unlink(temp_path)


def test_stale_bar_preserves_symbol():
    """Verify that stale bars preserve the correct symbol."""
    csv_data = """ts,symbol,open,high,low,close,volume
0,BTCUSDT,100.0,101.0,99.0,100.5,1000.0
60000,BTCUSDT,100.5,102.0,100.0,101.0,1100.0
"""

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        degradation = DataDegradationConfig(
            stale_prob=1.0,
            drop_prob=0.0,
            dropout_prob=0.0,
            max_delay_ms=0,
            seed=42,
        )

        cfg = OfflineCSVConfig(
            paths=[temp_path],
            timeframe="1m",
            enforce_closed_bars=False,
        )

        source = OfflineCSVBarSource(cfg, data_degradation=degradation)
        bars = list(source.stream_bars(["BTCUSDT"], interval_ms=60000))

        # All bars should have correct symbol
        for bar in bars:
            assert bar.symbol == "BTCUSDT", f"Bar at {bar.ts} has wrong symbol: {bar.symbol}"

    finally:
        os.unlink(temp_path)


def test_no_stale_bar_normal_operation():
    """Verify that with stale_prob=0, bars use correct timestamps and data."""
    csv_data = """ts,symbol,open,high,low,close,volume
0,BTCUSDT,100.0,101.0,99.0,100.5,1000.0
60000,BTCUSDT,100.5,102.0,100.0,101.0,1100.0
120000,BTCUSDT,101.0,103.0,100.5,102.0,1200.0
"""

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        degradation = DataDegradationConfig(
            stale_prob=0.0,  # No stale data
            drop_prob=0.0,
            dropout_prob=0.0,
            max_delay_ms=0,
            seed=42,
        )

        cfg = OfflineCSVConfig(
            paths=[temp_path],
            timeframe="1m",
            enforce_closed_bars=False,
        )

        source = OfflineCSVBarSource(cfg, data_degradation=degradation)
        bars = list(source.stream_bars(["BTCUSDT"], interval_ms=60000))

        assert len(bars) == 3

        # All bars should have correct timestamps and data
        assert bars[0].ts == 0
        assert float(bars[0].close) == 100.5

        assert bars[1].ts == 60000
        assert float(bars[1].close) == 101.0

        assert bars[2].ts == 120000
        assert float(bars[2].close) == 102.0

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
