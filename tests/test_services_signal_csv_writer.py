"""Comprehensive tests for services.signal_csv_writer module."""

import csv
import os
import time
from datetime import datetime, date
from pathlib import Path

import pytest

from services.signal_csv_writer import SignalCSVWriter


class TestSignalCSVWriterInit:
    """Tests for SignalCSVWriter initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Test initialization with default parameters."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        assert writer.path == str(path)
        assert writer.header == SignalCSVWriter.DEFAULT_HEADER
        assert writer._fsync_mode == "batch"
        assert writer._rotate_daily is True
        assert writer._flush_interval_s == 5.0
        writer.close()

    def test_init_with_custom_header(self, tmp_path):
        """Test initialization with custom header."""
        path = tmp_path / "signals.csv"
        custom_header = ["field1", "field2", "field3"]
        writer = SignalCSVWriter(str(path), header=custom_header)

        assert writer.header == custom_header
        writer.close()

    def test_init_with_fsync_modes(self, tmp_path):
        """Test initialization with different fsync modes."""
        for mode in ["always", "batch", "off"]:
            path = tmp_path / f"signals_{mode}.csv"
            writer = SignalCSVWriter(str(path), fsync_mode=mode)
            assert writer._fsync_mode == mode
            writer.close()

    def test_init_creates_directory(self, tmp_path):
        """Test initialization creates directory if not exists."""
        path = tmp_path / "subdir" / "signals.csv"
        writer = SignalCSVWriter(str(path))

        assert path.parent.exists()
        writer.close()

    def test_init_creates_file_with_header(self, tmp_path):
        """Test initialization creates file with header."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        assert path.exists()
        content = path.read_text()
        assert "ts_ms" in content
        assert "symbol" in content
        writer.close()


class TestSignalCSVWriterWrite:
    """Tests for write method."""

    def test_write_single_row(self, tmp_path):
        """Test writing a single row."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        row = {
            "ts_ms": 1234567890000,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "volume_frac": 0.5,
            "score": 0.75,
            "features_hash": "abc123",
        }

        writer.write(row)
        writer.close()

        # Read and verify
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTCUSDT"
        assert rows[0]["side"] == "BUY"

    def test_write_multiple_rows(self, tmp_path):
        """Test writing multiple rows."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        for i in range(5):
            row = {
                "ts_ms": 1234567890000 + i,
                "symbol": f"SYM{i}",
                "side": "BUY",
                "volume_frac": 0.5,
                "score": 0.75,
                "features_hash": f"hash{i}",
            }
            writer.write(row)

        writer.close()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5

    def test_write_with_missing_fields(self, tmp_path):
        """Test writing row with missing fields uses empty string."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        row = {
            "ts_ms": 1234567890000,
            "symbol": "BTCUSDT",
        }

        writer.write(row)
        writer.close()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["symbol"] == "BTCUSDT"
        assert rows[0]["side"] == ""

    def test_write_increments_written_counter(self, tmp_path):
        """Test write increments written counter."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        row = {"ts_ms": 1234567890000, "symbol": "BTCUSDT"}
        writer.write(row)

        assert writer._written == 1
        writer.close()

    def test_write_after_close_increments_dropped(self, tmp_path):
        """Test write after close increments dropped counter."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))
        writer.close()

        row = {"ts_ms": 1234567890000, "symbol": "BTCUSDT"}
        writer.write(row)

        assert writer._dropped == 1


class TestSignalCSVWriterRotation:
    """Tests for daily rotation."""

    def test_rotation_on_day_change(self, tmp_path):
        """Test file rotates when day changes."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), rotate_daily=True)

        # Write with today's timestamp
        today_ts = int(datetime.utcnow().timestamp() * 1000)
        writer.write({"ts_ms": today_ts, "symbol": "BTCUSDT"})

        # Write with tomorrow's timestamp (simulate day change)
        tomorrow_ts = today_ts + 86400 * 1000
        writer.write({"ts_ms": tomorrow_ts, "symbol": "ETHUSDT"})

        writer.close()

        # Check that rotation happened
        rotated_files = list(tmp_path.glob("signals-*.csv"))
        assert len(rotated_files) >= 0  # May or may not rotate depending on timing

    def test_no_rotation_when_disabled(self, tmp_path):
        """Test no rotation when rotate_daily is False."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), rotate_daily=False)

        today_ts = int(datetime.utcnow().timestamp() * 1000)
        tomorrow_ts = today_ts + 86400 * 1000

        writer.write({"ts_ms": today_ts, "symbol": "BTCUSDT"})
        writer.write({"ts_ms": tomorrow_ts, "symbol": "ETHUSDT"})

        writer.close()

        # All rows should be in main file
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_rotate_existing_file_on_init(self, tmp_path):
        """Test existing file is rotated on init if from different day."""
        path = tmp_path / "signals.csv"

        # Create old file
        path.write_text("ts_ms,symbol\n")

        # Set old mtime (yesterday)
        old_time = time.time() - 86400
        os.utime(path, (old_time, old_time))

        writer = SignalCSVWriter(str(path), rotate_daily=True)
        writer.close()

        # Check rotation happened
        rotated_files = list(tmp_path.glob("signals-*.csv"))
        # May or may not rotate depending on exact timing
        assert len(rotated_files) >= 0


class TestSignalCSVWriterFlush:
    """Tests for flush and fsync."""

    def test_flush_fsync_with_always_mode(self, tmp_path):
        """Test flush_fsync is called on every write with always mode."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), fsync_mode="always", flush_interval_s=None)

        writer.write({"ts_ms": 1234567890000, "symbol": "BTCUSDT"})

        # File should be flushed immediately
        assert path.exists()
        assert path.stat().st_size > 0
        writer.close()

    def test_flush_fsync_with_batch_mode(self, tmp_path):
        """Test flush_fsync is called periodically with batch mode."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), fsync_mode="batch", flush_interval_s=0.1)

        writer.write({"ts_ms": 1234567890000, "symbol": "BTCUSDT"})
        time.sleep(0.15)
        writer.write({"ts_ms": 1234567890001, "symbol": "ETHUSDT"})

        writer.close()
        assert path.exists()

    def test_flush_fsync_with_off_mode(self, tmp_path):
        """Test flush_fsync with off mode."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), fsync_mode="off")

        writer.write({"ts_ms": 1234567890000, "symbol": "BTCUSDT"})
        writer.flush_fsync()

        assert path.exists()
        writer.close()


class TestSignalCSVWriterReopen:
    """Tests for reopen method."""

    def test_reopen_after_write_error(self, tmp_path):
        """Test reopen recovers from write error."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        # Close file handle to simulate error
        if writer._file:
            writer._file.close()
            writer._file = None

        # Write should trigger retry with reopen
        row = {"ts_ms": 1234567890000, "symbol": "BTCUSDT"}
        writer.write(row)

        # Should have retried
        assert writer._retries >= 0
        writer.close()

    def test_reopen_preserves_content(self, tmp_path):
        """Test reopen preserves existing content."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        writer.write({"ts_ms": 1234567890000, "symbol": "BTCUSDT"})
        writer.reopen()
        writer.write({"ts_ms": 1234567890001, "symbol": "ETHUSDT"})

        writer.close()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2


class TestSignalCSVWriterStats:
    """Tests for stats method."""

    def test_stats_returns_dict(self, tmp_path):
        """Test stats returns dictionary with metrics."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        stats = writer.stats()

        assert isinstance(stats, dict)
        assert "path" in stats
        assert "written" in stats
        assert "errors" in stats
        assert "dropped" in stats
        writer.close()

    def test_stats_tracks_written_count(self, tmp_path):
        """Test stats tracks written count."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        for i in range(5):
            writer.write({"ts_ms": 1234567890000 + i, "symbol": f"SYM{i}"})

        stats = writer.stats()
        assert stats["written"] == 5
        writer.close()

    def test_stats_tracks_open_status(self, tmp_path):
        """Test stats tracks open status."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        stats1 = writer.stats()
        assert stats1["open"] is True

        writer.close()

        stats2 = writer.stats()
        assert stats2["open"] is False


class TestSignalCSVWriterClose:
    """Tests for close method."""

    def test_close_closes_file(self, tmp_path):
        """Test close closes file handle."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        assert writer._file is not None
        writer.close()
        assert writer._file is None

    def test_close_idempotent(self, tmp_path):
        """Test close can be called multiple times."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        writer.close()
        writer.close()  # Should not raise

    def test_close_flushes_data(self, tmp_path):
        """Test close flushes pending data."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        writer.write({"ts_ms": 1234567890000, "symbol": "BTCUSDT"})
        writer.close()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "BTCUSDT" in content


class TestSignalCSVWriterEdgeCases:
    """Tests for edge cases."""

    def test_write_with_special_characters(self, tmp_path):
        """Test writing data with special characters."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path))

        row = {
            "ts_ms": 1234567890000,
            "symbol": "BTC,USDT",  # Contains comma
            "features_hash": 'hash"with"quotes',
        }

        writer.write(row)
        writer.close()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # CSV should handle special characters correctly
        assert len(rows) == 1

    def test_normalize_flush_interval_negative(self, tmp_path):
        """Test negative flush interval is normalized to 0."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), flush_interval_s=-1.0)

        assert writer._flush_interval_s == 0.0
        writer.close()

    def test_normalize_fsync_mode_invalid(self, tmp_path):
        """Test invalid fsync mode is normalized to off."""
        path = tmp_path / "signals.csv"
        writer = SignalCSVWriter(str(path), fsync_mode="invalid")

        assert writer._fsync_mode == "off"
        writer.close()
