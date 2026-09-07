"""Comprehensive tests for services.ops_kill_switch module."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from services import ops_kill_switch


@pytest.fixture(autouse=True)
def reset_kill_switch(tmp_path):
    """Reset kill switch state before each test."""
    ops_kill_switch._counters = {"rest": 0, "ws": 0, "duplicates": 0, "stale": 0}
    ops_kill_switch._last_ts = {"rest": 0.0, "ws": 0.0, "duplicates": 0.0, "stale": 0.0}
    ops_kill_switch._tripped = False
    ops_kill_switch._limits = {"rest": 0, "ws": 0, "duplicates": 0, "stale": 0}
    ops_kill_switch._state_path = tmp_path / "ops_state.json"
    ops_kill_switch._flag_path = tmp_path / "ops_kill_switch.flag"
    ops_kill_switch._alert_command = None
    ops_kill_switch._reset_cooldown_sec = 60.0
    yield
    ops_kill_switch._counters = {"rest": 0, "ws": 0, "duplicates": 0, "stale": 0}
    ops_kill_switch._last_ts = {"rest": 0.0, "ws": 0.0, "duplicates": 0.0, "stale": 0.0}
    ops_kill_switch._tripped = False


class TestInit:
    """Tests for init function."""

    def test_init_with_limits(self, tmp_path):
        """Test init with error limits."""
        cfg = {
            "rest_limit": 10,
            "ws_limit": 5,
            "duplicate_limit": 3,
            "stale_limit": 2,
            "reset_cooldown_sec": 120.0,
        }

        ops_kill_switch.init(cfg)

        assert ops_kill_switch._limits["rest"] == 10
        assert ops_kill_switch._limits["ws"] == 5
        assert ops_kill_switch._limits["duplicates"] == 3
        assert ops_kill_switch._limits["stale"] == 2
        assert ops_kill_switch._reset_cooldown_sec == 120.0

    def test_init_with_legacy_config_keys(self):
        """Test init with legacy configuration keys."""
        cfg = {
            "rest_error_limit": 15,
            "ws_error_limit": 8,
        }

        ops_kill_switch.init(cfg)

        assert ops_kill_switch._limits["rest"] == 15
        assert ops_kill_switch._limits["ws"] == 8

    def test_init_with_custom_paths(self, tmp_path):
        """Test init with custom state and flag paths."""
        state_path = tmp_path / "custom_state.json"
        flag_path = tmp_path / "custom_flag.txt"

        cfg = {
            "state_path": str(state_path),
            "flag_path": str(flag_path),
        }

        ops_kill_switch.init(cfg)

        assert ops_kill_switch._state_path == state_path
        assert ops_kill_switch._flag_path == flag_path

    def test_init_with_alert_command(self):
        """Test init with alert command."""
        cfg = {
            "alert_command": ["echo", "Alert!"],
        }

        ops_kill_switch.init(cfg)

        assert ops_kill_switch._alert_command == ["echo", "Alert!"]

    def test_init_loads_existing_state(self, tmp_path):
        """Test init loads existing state from file."""
        state_path = tmp_path / "state.json"
        state_data = {
            "counters": {"rest": 5, "ws": 3},
            "last_ts": {"rest": 100.0, "ws": 200.0},
            "tripped": True,
        }
        state_path.write_text(json.dumps(state_data))

        cfg = {"state_path": str(state_path)}
        ops_kill_switch.init(cfg)

        assert ops_kill_switch._counters["rest"] == 5
        assert ops_kill_switch._counters["ws"] == 3
        assert ops_kill_switch._tripped is True


class TestRecordError:
    """Tests for record_error function."""

    def test_record_rest_error(self):
        """Test recording REST error."""
        ops_kill_switch.init({"rest_limit": 0})

        ops_kill_switch.record_error("rest")

        assert ops_kill_switch._counters["rest"] == 1

    def test_record_ws_error(self):
        """Test recording WebSocket error."""
        ops_kill_switch.init({"ws_limit": 0})

        ops_kill_switch.record_error("ws")

        assert ops_kill_switch._counters["ws"] == 1

    def test_record_error_invalid_kind(self):
        """Test recording error with invalid kind raises ValueError."""
        ops_kill_switch.init({})

        with pytest.raises(ValueError, match="kind must be"):
            ops_kill_switch.record_error("invalid")

    def test_record_error_trips_when_limit_reached(self, tmp_path):
        """Test recording error trips kill switch when limit is reached."""
        ops_kill_switch.init({"rest_limit": 3, "state_path": str(tmp_path / "state.json")})

        ops_kill_switch.record_error("rest")
        ops_kill_switch.record_error("rest")
        assert ops_kill_switch._tripped is False

        ops_kill_switch.record_error("rest")
        assert ops_kill_switch._tripped is True

    def test_record_error_resets_counter_after_cooldown(self):
        """Test counter resets after cooldown period."""
        ops_kill_switch.init({"rest_limit": 0, "reset_cooldown_sec": 0.1})

        ops_kill_switch.record_error("rest")
        assert ops_kill_switch._counters["rest"] == 1

        time.sleep(0.15)
        ops_kill_switch.record_error("rest")

        # Counter should be reset and then incremented
        assert ops_kill_switch._counters["rest"] == 1

    @patch("subprocess.run")
    def test_record_error_executes_alert_command(self, mock_run):
        """Test alert command is executed when tripped."""
        mock_run.return_value = MagicMock(returncode=0)
        ops_kill_switch.init(
            {
                "rest_limit": 1,
                "alert_command": ["echo", "Alert!"],
            }
        )

        ops_kill_switch.record_error("rest")

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["echo", "Alert!"]


class TestRecordDuplicate:
    """Tests for record_duplicate function."""

    def test_record_duplicate(self):
        """Test recording duplicate message."""
        ops_kill_switch.init({"duplicate_limit": 0})

        ops_kill_switch.record_duplicate()

        assert ops_kill_switch._counters["duplicates"] == 1

    def test_record_duplicate_trips_when_limit_reached(self, tmp_path):
        """Test recording duplicate trips when limit reached.

        Note: The kill switch uses >= comparison, so with limit=3,
        trip occurs when counter reaches 3 (after 3rd call).
        """
        # Use tmp_path to ensure clean state with fresh paths
        state_path = tmp_path / "test_dup_state.json"
        flag_path = tmp_path / "test_dup_flag.txt"

        ops_kill_switch.init(
            {
                "duplicate_limit": 3,  # Trip when counter >= 3
                "state_path": str(state_path),
                "flag_path": str(flag_path),
            }
        )

        ops_kill_switch.record_duplicate()  # counter=1
        ops_kill_switch.record_duplicate()  # counter=2
        assert ops_kill_switch._tripped is False

        ops_kill_switch.record_duplicate()  # counter=3, 3 >= 3 -> trip
        assert ops_kill_switch._tripped is True


class TestResetDuplicates:
    """Tests for reset_duplicates function."""

    def test_reset_duplicates(self):
        """Test resetting duplicate counter."""
        ops_kill_switch.init({})
        ops_kill_switch._counters["duplicates"] = 10

        ops_kill_switch.reset_duplicates()

        assert ops_kill_switch._counters["duplicates"] == 0

    def test_reset_duplicates_updates_timestamp(self):
        """Test reset_duplicates updates last_ts."""
        ops_kill_switch.init({})
        ops_kill_switch._last_ts["duplicates"] = 0.0

        before = time.time()
        ops_kill_switch.reset_duplicates()
        after = time.time()

        assert before <= ops_kill_switch._last_ts["duplicates"] <= after


class TestRecordStale:
    """Tests for record_stale function."""

    def test_record_stale(self):
        """Test recording stale interval event."""
        ops_kill_switch.init({"stale_limit": 0})

        ops_kill_switch.record_stale()

        assert ops_kill_switch._counters["stale"] == 1


class TestTripped:
    """Tests for tripped function."""

    def test_tripped_initially_false(self):
        """Test tripped is initially false."""
        ops_kill_switch.init({})

        assert ops_kill_switch.tripped() is False

    def test_tripped_after_limit_reached(self):
        """Test tripped returns true after limit reached."""
        ops_kill_switch.init({"rest_limit": 1})

        ops_kill_switch.record_error("rest")

        assert ops_kill_switch.tripped() is True

    def test_tripped_reads_flag_file(self, tmp_path):
        """Test tripped reads flag file on init."""
        flag_path = tmp_path / "flag.txt"
        flag_path.write_text("1")

        ops_kill_switch.init({"flag_path": str(flag_path)})

        assert ops_kill_switch.tripped() is True


class TestManualReset:
    """Tests for manual_reset function."""

    def test_manual_reset_clears_counters(self):
        """Test manual reset clears all counters."""
        ops_kill_switch.init({})
        ops_kill_switch._counters["rest"] = 5
        ops_kill_switch._counters["ws"] = 3
        ops_kill_switch._tripped = True

        ops_kill_switch.manual_reset()

        assert ops_kill_switch._counters["rest"] == 0
        assert ops_kill_switch._counters["ws"] == 0
        assert ops_kill_switch._tripped is False

    def test_manual_reset_removes_flag_file(self, tmp_path):
        """Test manual reset removes flag file."""
        flag_path = tmp_path / "flag.txt"
        flag_path.write_text("1")
        ops_kill_switch.init({"flag_path": str(flag_path)})

        ops_kill_switch.manual_reset()

        assert not flag_path.exists()

    def test_manual_reset_saves_state(self, tmp_path):
        """Test manual reset saves state."""
        state_path = tmp_path / "state.json"
        ops_kill_switch.init({"state_path": str(state_path)})
        ops_kill_switch._counters["rest"] = 5

        ops_kill_switch.manual_reset()

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["counters"]["rest"] == 0


class TestTick:
    """Tests for tick function."""

    def test_tick_resets_counters_after_cooldown(self):
        """Test tick resets counters after cooldown."""
        ops_kill_switch.init({"reset_cooldown_sec": 0.1})
        ops_kill_switch._counters["rest"] = 5
        ops_kill_switch._last_ts["rest"] = time.time() - 0.2

        ops_kill_switch.tick()

        assert ops_kill_switch._counters["rest"] == 0

    def test_tick_saves_state(self, tmp_path):
        """Test tick saves state."""
        state_path = tmp_path / "state.json"
        ops_kill_switch.init({"state_path": str(state_path)})

        ops_kill_switch.tick()

        assert state_path.exists()


class TestStatePersistence:
    """Tests for state persistence."""

    def test_state_saved_on_record_error(self, tmp_path):
        """Test state is saved when recording error."""
        state_path = tmp_path / "state.json"
        ops_kill_switch.init({"state_path": str(state_path)})

        ops_kill_switch.record_error("rest")

        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["counters"]["rest"] == 1

    def test_flag_file_created_when_tripped(self, tmp_path):
        """Test flag file is created when tripped."""
        flag_path = tmp_path / "flag.txt"
        ops_kill_switch.init({"flag_path": str(flag_path), "rest_limit": 1})

        ops_kill_switch.record_error("rest")

        assert flag_path.exists()
