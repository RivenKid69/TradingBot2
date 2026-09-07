"""Comprehensive tests for services.signal_bus module."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services import signal_bus
from api.spot_signals import SpotSignalEnvelope, SpotSignalTargetWeightPayload, SpotSignalEconomics


def _make_valid_payload() -> dict:
    """Create a valid payload structure for SpotSignalEnvelope."""
    return {
        "kind": "target_weight",
        "target_weight": 0.5,
        "economics": {
            "edge_bps": 10.0,
            "cost_bps": 2.0,
            "net_bps": 8.0,
            "turnover_usd": 1000.0,
            "act_now": True,
            "impact": 0.0,
            "impact_mode": "none",
        },
    }


@pytest.fixture(autouse=True)
def reset_signal_bus():
    """Reset signal_bus global state before each test."""
    signal_bus._SEEN.clear()
    signal_bus._loaded = False
    signal_bus.dropped_by_reason.clear()
    signal_bus.config.enabled = True
    signal_bus._SIGNING_SECRET = None
    signal_bus.OUT_CSV = None
    signal_bus.DROPS_CSV = None
    yield
    signal_bus._SEEN.clear()
    signal_bus._loaded = False
    signal_bus.dropped_by_reason.clear()
    signal_bus.config.enabled = True
    signal_bus._SIGNING_SECRET = None
    signal_bus.OUT_CSV = None
    signal_bus.DROPS_CSV = None


class TestSignalId:
    """Tests for signal_id function."""

    def test_signal_id_generation(self):
        """Test signal ID generation with valid inputs."""
        sid = signal_bus.signal_id("BTCUSDT", 1234567890000)
        assert sid == "BTCUSDT:1234567890000"

    def test_signal_id_different_symbols(self):
        """Test signal IDs are unique for different symbols."""
        sid1 = signal_bus.signal_id("BTCUSDT", 1234567890000)
        sid2 = signal_bus.signal_id("ETHUSDT", 1234567890000)
        assert sid1 != sid2

    def test_signal_id_different_timestamps(self):
        """Test signal IDs are unique for different timestamps."""
        sid1 = signal_bus.signal_id("BTCUSDT", 1234567890000)
        sid2 = signal_bus.signal_id("BTCUSDT", 1234567890001)
        assert sid1 != sid2


class TestConfigureSigning:
    """Tests for configure_signing function."""

    def test_configure_signing_with_bytes(self):
        """Test configuring signing with bytes secret."""
        secret = b"test_secret"
        signal_bus.configure_signing(secret)
        assert signal_bus._SIGNING_SECRET == secret

    def test_configure_signing_with_string(self):
        """Test configuring signing with string secret."""
        secret = "test_secret"
        signal_bus.configure_signing(secret)
        assert signal_bus._SIGNING_SECRET == b"test_secret"

    def test_configure_signing_with_none(self):
        """Test configuring signing with None disables signing."""
        signal_bus.configure_signing("secret")
        assert signal_bus._SIGNING_SECRET is not None
        signal_bus.configure_signing(None)
        assert signal_bus._SIGNING_SECRET is None

    def test_configure_signing_with_empty_string(self):
        """Test configuring signing with empty string."""
        signal_bus.configure_signing("")
        assert signal_bus._SIGNING_SECRET is None


class TestStateManagement:
    """Tests for state loading and saving."""

    def test_load_state_from_nonexistent_file(self, tmp_path):
        """Test loading state from non-existent file creates empty state."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)
        assert len(signal_bus._SEEN) == 0
        assert signal_bus._loaded is True

    def test_load_state_from_valid_file(self, tmp_path):
        """Test loading state from valid JSON file."""
        state_file = tmp_path / "state.json"
        now_ms = int(time.time() * 1000)
        future_ms = now_ms + 60000
        data = {
            "signal1": future_ms,
            "signal2": future_ms + 1000,
        }
        state_file.write_text(json.dumps(data))

        signal_bus.load_state(state_file)
        assert len(signal_bus._SEEN) == 2
        assert signal_bus._SEEN["signal1"] == future_ms

    def test_load_state_purges_expired_entries(self, tmp_path):
        """Test loading state purges expired entries."""
        state_file = tmp_path / "state.json"
        now_ms = int(time.time() * 1000)
        past_ms = now_ms - 60000
        future_ms = now_ms + 60000

        data = {
            "expired_signal": past_ms,
            "valid_signal": future_ms,
        }
        state_file.write_text(json.dumps(data))

        signal_bus.load_state(state_file)
        assert "expired_signal" not in signal_bus._SEEN
        assert "valid_signal" in signal_bus._SEEN

    def test_load_state_handles_corrupt_file(self, tmp_path):
        """Test loading state from corrupt file handles gracefully."""
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json {{{")

        signal_bus.load_state(state_file)
        assert len(signal_bus._SEEN) == 0
        assert signal_bus._loaded is True

    def test_flush_state_creates_file(self, tmp_path):
        """Test flushing state creates JSON file."""
        state_file = tmp_path / "state" / "seen_signals.json"
        signal_bus._STATE_PATH = state_file
        signal_bus._SEEN["test"] = 12345
        signal_bus._loaded = True

        signal_bus.flush_state()
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["test"] == 12345

    def test_flush_state_atomic_write(self, tmp_path):
        """Test flush_state uses atomic write."""
        state_file = tmp_path / "state.json"
        signal_bus._STATE_PATH = state_file
        signal_bus._SEEN["test"] = 12345
        signal_bus._loaded = True

        signal_bus.flush_state()

        # Check that temp file doesn't exist
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0


class TestAlreadyEmitted:
    """Tests for already_emitted function."""

    def test_already_emitted_for_new_signal(self):
        """Test already_emitted returns False for new signal."""
        signal_bus.load_state()
        assert signal_bus.already_emitted("new_signal") is False

    def test_already_emitted_for_existing_valid_signal(self):
        """Test already_emitted returns True for existing valid signal."""
        signal_bus.load_state()
        now_ms = int(time.time() * 1000)
        future_ms = now_ms + 60000
        signal_bus._SEEN["existing_signal"] = future_ms

        assert signal_bus.already_emitted("existing_signal", now_ms=now_ms) is True

    def test_already_emitted_for_expired_signal(self):
        """Test already_emitted returns False for expired signal."""
        signal_bus.load_state()
        now_ms = int(time.time() * 1000)
        past_ms = now_ms - 60000
        signal_bus._SEEN["expired_signal"] = past_ms

        assert signal_bus.already_emitted("expired_signal", now_ms=now_ms) is False
        # Should be removed from _SEEN
        assert "expired_signal" not in signal_bus._SEEN


class TestMarkEmitted:
    """Tests for mark_emitted function."""

    def test_mark_emitted_adds_signal(self, tmp_path):
        """Test mark_emitted adds signal to state."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        now_ms = int(time.time() * 1000)
        future_ms = now_ms + 60000

        signal_bus.mark_emitted("test_signal", future_ms, now_ms=now_ms)
        assert signal_bus._SEEN["test_signal"] == future_ms

    def test_mark_emitted_purges_expired(self, tmp_path):
        """Test mark_emitted purges expired entries."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        now_ms = int(time.time() * 1000)
        past_ms = now_ms - 60000
        future_ms = now_ms + 60000

        signal_bus._SEEN["expired"] = past_ms
        signal_bus.mark_emitted("new_signal", future_ms, now_ms=now_ms)

        assert "expired" not in signal_bus._SEEN
        assert "new_signal" in signal_bus._SEEN


class TestLogDrop:
    """Tests for log_drop function."""

    def test_log_drop_without_csv(self):
        """Test log_drop without CSV file configured."""
        envelope = SpotSignalEnvelope(
            symbol="BTCUSDT",
            bar_close_ms=1234567890000,
            expires_at_ms=1234567950000,
            payload=_make_valid_payload(),
        )

        signal_bus.log_drop(envelope, "duplicate")
        assert signal_bus.dropped_by_reason["duplicate"] == 1

    def test_log_drop_with_csv(self, tmp_path):
        """Test log_drop writes to CSV file."""
        csv_file = tmp_path / "drops.csv"
        signal_bus.DROPS_CSV = str(csv_file)

        envelope = SpotSignalEnvelope(
            symbol="BTCUSDT",
            bar_close_ms=1234567890000,
            expires_at_ms=1234567950000,
            payload=_make_valid_payload(),
        )

        signal_bus.log_drop(envelope, "expired")
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "BTCUSDT" in content
        assert "expired" in content

    @patch("services.ops_kill_switch.record_duplicate")
    def test_log_drop_duplicate_records_to_kill_switch(self, mock_record):
        """Test log_drop with duplicate reason records to kill switch."""
        envelope = SpotSignalEnvelope(
            symbol="BTCUSDT",
            bar_close_ms=1234567890000,
            expires_at_ms=1234567950000,
            payload=_make_valid_payload(),
        )

        signal_bus.log_drop(envelope, "duplicate")
        mock_record.assert_called_once()


class TestPublishSignal:
    """Tests for publish_signal function."""

    def test_publish_signal_success(self, tmp_path):
        """Test successful signal publishing."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert result is True
        send_fn.assert_called_once()
        sid = signal_bus.signal_id("BTCUSDT", now_ms)
        assert sid in signal_bus._SEEN

    def test_publish_signal_when_disabled(self):
        """Test publish_signal when config is disabled."""
        signal_bus.config.enabled = False
        signal_bus.load_state()

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert result is False
        send_fn.assert_not_called()
        assert signal_bus.dropped_by_reason["disabled"] == 1

    def test_publish_signal_duplicate(self, tmp_path):
        """Test publishing duplicate signal."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        # First publish
        result1 = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        # Second publish (duplicate)
        result2 = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert result1 is True
        assert result2 is False
        assert send_fn.call_count == 1
        assert signal_bus.dropped_by_reason["duplicate"] == 1

    def test_publish_signal_expired(self, tmp_path):
        """Test publishing expired signal."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        # Signal was created in the past with a short validity
        bar_close_ms = now_ms - 5000  # Bar closed 5 seconds ago
        expires_at_ms = now_ms - 1000  # Expired 1 second ago (but after bar_close)

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=bar_close_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert result is False
        send_fn.assert_not_called()
        assert signal_bus.dropped_by_reason["expired"] == 1

    def test_publish_signal_with_valid_until(self, tmp_path):
        """Test publishing signal with valid_until constraint."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000
        valid_until_ms = now_ms + 30000

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            valid_until_ms=valid_until_ms,
            now_ms=now_ms,
        )

        assert result is True
        # Should use min(expires_at_ms, valid_until_ms)
        sid = signal_bus.signal_id("BTCUSDT", now_ms)
        assert signal_bus._SEEN[sid] == valid_until_ms

    def test_publish_signal_valid_until_expired(self, tmp_path):
        """Test publishing signal with expired valid_until."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000
        valid_until_ms = now_ms - 1000  # Already expired

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            valid_until_ms=valid_until_ms,
            now_ms=now_ms,
        )

        assert result is False
        send_fn.assert_not_called()
        assert signal_bus.dropped_by_reason["valid_until_expired"] == 1

    def test_publish_signal_with_signing(self, tmp_path):
        """Test publishing signal with HMAC signing enabled."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)
        signal_bus.configure_signing("test_secret")

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert result is True
        send_fn.assert_called_once()
        # Check that signature was added
        call_args = send_fn.call_args[0][0]
        assert "signature" in call_args

    def test_publish_signal_with_custom_dedup_key(self, tmp_path):
        """Test publishing signal with custom deduplication key."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        result = signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
            dedup_key="custom_key",
        )

        assert result is True
        assert "custom_key" in signal_bus._SEEN

    def test_publish_signal_writes_to_csv(self, tmp_path):
        """Test publish_signal writes to OUT_CSV when configured."""
        state_file = tmp_path / "state.json"
        csv_file = tmp_path / "signals.csv"
        signal_bus.load_state(state_file)
        signal_bus.OUT_CSV = str(csv_file)

        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        signal_bus.publish_signal(
            symbol="BTCUSDT",
            bar_close_ms=now_ms,
            payload=_make_valid_payload(),
            send_fn=send_fn,
            expires_at_ms=expires_at_ms,
            now_ms=now_ms,
        )

        assert csv_file.exists()
        content = csv_file.read_text()
        assert "BTCUSDT" in content


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_publish_signals(self, tmp_path):
        """Test concurrent signal publishing is thread-safe."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        results = []
        send_fn = MagicMock()
        now_ms = int(time.time() * 1000)
        expires_at_ms = now_ms + 60000

        def publish():
            result = signal_bus.publish_signal(
                symbol="BTCUSDT",
                bar_close_ms=now_ms,
                payload=_make_valid_payload(),
                send_fn=send_fn,
                expires_at_ms=expires_at_ms,
                now_ms=now_ms,
            )
            results.append(result)

        threads = [threading.Thread(target=publish) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed due to deduplication
        assert sum(results) == 1
        assert send_fn.call_count == 1

    def test_concurrent_mark_emitted(self, tmp_path):
        """Test concurrent mark_emitted calls are thread-safe."""
        state_file = tmp_path / "state.json"
        signal_bus.load_state(state_file)

        now_ms = int(time.time() * 1000)
        future_ms = now_ms + 60000

        def mark():
            for i in range(100):
                signal_bus.mark_emitted(f"signal_{i}", future_ms, now_ms=now_ms)

        threads = [threading.Thread(target=mark) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All signals should be marked
        assert len(signal_bus._SEEN) == 100


class TestTimestampCoercion:
    """Tests for _coerce_timestamp_ms function."""

    def test_coerce_none(self):
        """Test coercing None returns None."""
        assert signal_bus._coerce_timestamp_ms(None) is None

    def test_coerce_int(self):
        """Test coercing int returns int."""
        assert signal_bus._coerce_timestamp_ms(1234567890) == 1234567890

    def test_coerce_float(self):
        """Test coercing float returns int."""
        assert signal_bus._coerce_timestamp_ms(1234567890.5) == 1234567890

    def test_coerce_string_int(self):
        """Test coercing string integer."""
        assert signal_bus._coerce_timestamp_ms("1234567890") == 1234567890

    def test_coerce_bool(self):
        """Test coercing bool returns None."""
        assert signal_bus._coerce_timestamp_ms(True) is None
        assert signal_bus._coerce_timestamp_ms(False) is None

    def test_coerce_inf(self):
        """Test coercing infinity returns None."""
        assert signal_bus._coerce_timestamp_ms(float("inf")) is None
        assert signal_bus._coerce_timestamp_ms(float("-inf")) is None

    def test_coerce_nan(self):
        """Test coercing NaN returns None."""
        assert signal_bus._coerce_timestamp_ms(float("nan")) is None
