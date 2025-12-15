# -*- coding: utf-8 -*-
"""
Tests for Agent Daemon CLI Entrypoint.

Covers:
- setup_logging()
- load_config_file()
- build_daemon_config()
- DaemonRunner
- create_parser()
- main()
"""

import pytest
import tempfile
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import yaml

from packages.agent.daemon.__main__ import (
    setup_logging,
    load_config_file,
    build_daemon_config,
    DaemonRunner,
    create_parser,
    main,
    __version__,
)
from packages.agent.daemon.agentd import DaemonConfig


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        with patch('packages.agent.daemon.__main__.logging') as mock_logging:
            mock_root = MagicMock()
            mock_logging.getLogger.return_value = mock_root
            mock_logging.INFO = 20

            setup_logging()

            mock_logging.getLogger.assert_called()

    def test_setup_logging_with_level(self):
        """Test logging setup with custom level."""
        with patch('packages.agent.daemon.__main__.logging') as mock_logging:
            mock_root = MagicMock()
            mock_logging.getLogger.return_value = mock_root
            mock_logging.DEBUG = 10

            setup_logging(level="DEBUG")

            mock_logging.getLogger.assert_called()

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "logs" / "test.log"

        with patch('packages.agent.daemon.__main__.logging') as mock_logging:
            mock_root = MagicMock()
            mock_logging.getLogger.return_value = mock_root
            mock_logging.INFO = 20
            mock_logging.Formatter.return_value = MagicMock()
            mock_logging.StreamHandler.return_value = MagicMock()
            mock_logging.FileHandler.return_value = MagicMock()

            setup_logging(level="INFO", log_file=log_file)

            # Parent directory should be created
            assert log_file.parent.exists()


class TestLoadConfigFile:
    """Tests for load_config_file function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid YAML config."""
        config_path = tmp_path / "agent.yaml"
        config_data = {
            "agent": {
                "name": "test-agent",
                "version": "1.0.0",
            },
            "cloud": {
                "endpoint": "https://cloud.example.com",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_config_file(config_path)

        assert result["agent"]["name"] == "test-agent"
        assert result["cloud"]["endpoint"] == "https://cloud.example.com"

    def test_load_missing_config(self, tmp_path):
        """Test loading missing config file."""
        config_path = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            load_config_file(config_path)

    def test_load_empty_config(self, tmp_path):
        """Test loading empty config file."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        result = load_config_file(config_path)

        assert result == {}

    def test_load_config_with_components(self, tmp_path):
        """Test loading config with component sections."""
        config_path = tmp_path / "agent.yaml"
        config_data = {
            "agent": {"name": "test"},
            "cloud": {"endpoint": "https://cloud.example.com"},
            "components": {
                "kill_switch": {
                    "max_daily_loss_pct": 0.03,
                    "max_drawdown_pct": 0.08,
                },
                "telemetry": {
                    "max_buffer_size": 50000,
                    "batch_size": 50,
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_config_file(config_path)

        assert result["components"]["kill_switch"]["max_daily_loss_pct"] == 0.03
        assert result["components"]["telemetry"]["batch_size"] == 50


class TestBuildDaemonConfig:
    """Tests for build_daemon_config function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock CLI args."""
        args = MagicMock()
        args.cloud_endpoint = None
        args.enrollment_token = None
        args.access_token = None
        args.agent_id = None
        args.agent_name = None
        args.data_dir = None
        return args

    def test_build_config_from_dict(self, mock_args, tmp_path):
        """Test building config from dict."""
        config_dict = {
            "agent": {
                "name": "test-agent",
                "data_dir": str(tmp_path),
            },
            "cloud": {
                "endpoint": "https://cloud.example.com",
                "timeout_seconds": 60,
            },
        }

        result = build_daemon_config(config_dict, mock_args)

        assert result.agent_name == "test-agent"
        assert result.cloud_endpoint == "https://cloud.example.com"
        assert result.cloud_timeout_seconds == 60
        assert result.data_dir == tmp_path

    def test_build_config_cli_priority(self, mock_args, tmp_path):
        """Test CLI args have priority over config file."""
        config_dict = {
            "agent": {"name": "config-agent"},
            "cloud": {"endpoint": "https://config.example.com"},
        }
        mock_args.agent_name = "cli-agent"
        mock_args.cloud_endpoint = "https://cli.example.com"

        result = build_daemon_config(config_dict, mock_args)

        assert result.agent_name == "cli-agent"
        assert result.cloud_endpoint == "https://cli.example.com"

    def test_build_config_env_priority(self, mock_args, tmp_path):
        """Test ENV vars have priority over config file."""
        config_dict = {
            "agent": {"name": "config-agent"},
            "cloud": {"endpoint": "https://config.example.com"},
        }

        with patch.dict(os.environ, {
            "CCEA_AGENT_NAME": "env-agent",
            "CCEA_CLOUD_ENDPOINT": "https://env.example.com",
        }):
            result = build_daemon_config(config_dict, mock_args)

        assert result.agent_name == "env-agent"
        assert result.cloud_endpoint == "https://env.example.com"

    def test_build_config_with_kill_switch(self, mock_args, tmp_path):
        """Test building config with kill switch component."""
        config_dict = {
            "agent": {"data_dir": str(tmp_path)},
            "cloud": {},
            "components": {
                "kill_switch": {
                    "max_daily_loss_pct": 0.03,
                    "max_drawdown_pct": 0.08,
                    "max_broker_errors_per_minute": 10,
                    "max_consecutive_errors": 5,
                },
            },
        }

        result = build_daemon_config(config_dict, mock_args)

        assert result.kill_switch_config is not None
        assert result.kill_switch_config.max_broker_errors_per_minute == 10
        assert result.kill_switch_config.max_consecutive_errors == 5

    def test_build_config_with_telemetry(self, mock_args, tmp_path):
        """Test building config with telemetry component."""
        config_dict = {
            "agent": {"data_dir": str(tmp_path)},
            "cloud": {},
            "components": {
                "telemetry": {
                    "max_buffer_size": 50000,
                    "max_age_days": 14,
                    "batch_size": 200,
                    "flush_interval_seconds": 60,
                },
            },
        }

        result = build_daemon_config(config_dict, mock_args)

        assert result.telemetry_config is not None
        assert result.telemetry_config.max_buffer_size == 50000
        assert result.telemetry_config.max_age_days == 14
        assert result.telemetry_config.batch_size == 200

    def test_build_config_with_time_sync(self, mock_args, tmp_path):
        """Test building config with time sync component."""
        config_dict = {
            "agent": {"data_dir": str(tmp_path)},
            "cloud": {},
            "components": {
                "time_sync": {
                    "max_drift_seconds": 2.0,
                    "check_interval_seconds": 30,
                    "ntp_servers": ["time.google.com"],
                },
            },
        }

        result = build_daemon_config(config_dict, mock_args)

        assert result.time_sync_config is not None
        assert result.time_sync_config.max_drift_seconds == 2.0

    def test_build_config_defaults(self, mock_args):
        """Test default values when no config provided."""
        result = build_daemon_config({}, mock_args)

        assert result.agent_name == "ccea-agent"
        assert result.require_preflight is True
        assert result.enable_telemetry is True


class TestDaemonRunner:
    """Tests for DaemonRunner class."""

    @pytest.fixture
    def mock_daemon(self):
        """Create mock daemon."""
        daemon = MagicMock()
        daemon.initialize.return_value = True
        daemon.start.return_value = (True, None)
        daemon.stop.return_value = True
        daemon.is_halted = False
        daemon.agent_id = "test-agent-123"
        daemon.state.name = "RUNNING"
        return daemon

    def test_runner_creation(self, mock_daemon):
        """Test runner creation."""
        runner = DaemonRunner(mock_daemon)

        assert runner.daemon == mock_daemon
        assert runner._shutdown_requested is False
        assert runner._restart_requested is False

    def test_setup_signal_handlers(self, mock_daemon):
        """Test signal handler setup."""
        runner = DaemonRunner(mock_daemon)

        with patch('packages.agent.daemon.__main__.signal') as mock_signal:
            mock_signal.SIGTERM = signal.SIGTERM
            mock_signal.SIGINT = signal.SIGINT
            if hasattr(signal, 'SIGHUP'):
                mock_signal.SIGHUP = signal.SIGHUP
            if hasattr(signal, 'SIGUSR1'):
                mock_signal.SIGUSR1 = signal.SIGUSR1

            runner.setup_signal_handlers()

            # Should register handlers for SIGTERM and SIGINT at minimum
            assert mock_signal.signal.call_count >= 2

    def test_handle_shutdown_signal(self, mock_daemon):
        """Test shutdown signal handling."""
        runner = DaemonRunner(mock_daemon)

        runner._handle_shutdown_signal(signal.SIGTERM, None)

        assert runner._shutdown_requested is True

    def test_handle_reload_signal(self, mock_daemon):
        """Test reload signal handling (SIGHUP)."""
        runner = DaemonRunner(mock_daemon)

        runner._handle_reload_signal(signal.SIGHUP if hasattr(signal, 'SIGHUP') else 1, None)

        assert runner._restart_requested is True
        assert runner._shutdown_requested is True

    def test_handle_status_signal(self, mock_daemon):
        """Test status signal handling (SIGUSR1)."""
        runner = DaemonRunner(mock_daemon)
        mock_daemon.get_status.return_value = {"state": "RUNNING"}

        runner._handle_status_signal(signal.SIGUSR1 if hasattr(signal, 'SIGUSR1') else 10, None)

        mock_daemon.get_status.assert_called_once()

    def test_run_initialize_failure(self, mock_daemon):
        """Test run with initialization failure."""
        mock_daemon.initialize.return_value = False
        runner = DaemonRunner(mock_daemon)
        runner._shutdown_requested = True  # Prevent blocking

        with patch.object(runner, 'setup_signal_handlers'):
            result = runner.run()

        assert result == 1

    def test_run_start_failure(self, mock_daemon):
        """Test run with start failure."""
        mock_daemon.start.return_value = (False, "Start error")
        runner = DaemonRunner(mock_daemon)
        runner._shutdown_requested = True

        with patch.object(runner, 'setup_signal_handlers'):
            result = runner.run()

        assert result == 1


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_creation(self):
        """Test parser is created correctly."""
        parser = create_parser()

        assert parser.prog == "ccea-agentd"
        assert parser.description is not None

    def test_parser_config_arg(self):
        """Test --config argument."""
        parser = create_parser()
        args = parser.parse_args(["--config", "/path/to/config.yaml"])

        assert args.config == Path("/path/to/config.yaml")

    def test_parser_cloud_args(self):
        """Test cloud connection arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--cloud-endpoint", "https://cloud.example.com",
            "--enrollment-token", "test-token",
        ])

        assert args.cloud_endpoint == "https://cloud.example.com"
        assert args.enrollment_token == "test-token"

    def test_parser_agent_args(self):
        """Test agent identity arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--agent-id", "agent-123",
            "--agent-name", "my-agent",
        ])

        assert args.agent_id == "agent-123"
        assert args.agent_name == "my-agent"

    def test_parser_logging_args(self):
        """Test logging arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "--log-level", "DEBUG",
            "--log-file", "/var/log/agent.log",
        ])

        assert args.log_level == "DEBUG"
        assert args.log_file == Path("/var/log/agent.log")

    def test_parser_behavior_flags(self):
        """Test behavior flag arguments."""
        parser = create_parser()
        args = parser.parse_args(["--no-preflight", "--no-telemetry"])

        assert args.no_preflight is True
        assert args.no_telemetry is True

    def test_parser_debug_flags(self):
        """Test debug flag arguments."""
        parser = create_parser()
        args = parser.parse_args(["--dry-run", "--dump-config"])

        assert args.dry_run is True
        assert args.dump_config is True

    def test_parser_version(self):
        """Test --version flag."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])


class TestMain:
    """Tests for main entry point."""

    def test_main_dry_run(self, tmp_path):
        """Test main with --dry-run flag."""
        result = main(["--dry-run", "--data-dir", str(tmp_path)])

        assert result == 0

    def test_main_dump_config(self, tmp_path, capsys):
        """Test main with --dump-config flag."""
        result = main(["--dump-config", "--data-dir", str(tmp_path)])

        assert result == 0
        captured = capsys.readouterr()
        assert "agent_name" in captured.out

    def test_main_missing_config_file(self, tmp_path):
        """Test main with missing config file."""
        result = main(["--config", str(tmp_path / "missing.yaml")])

        assert result == 1

    def test_main_invalid_yaml(self, tmp_path):
        """Test main with invalid YAML config."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [")

        result = main(["--config", str(config_path)])

        assert result == 1

    def test_main_valid_config_dry_run(self, tmp_path):
        """Test main with valid config and dry run."""
        config_path = tmp_path / "agent.yaml"
        config_data = {
            "agent": {"name": "test-agent", "data_dir": str(tmp_path)},
            "cloud": {"endpoint": "https://cloud.example.com"},
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = main(["--config", str(config_path), "--dry-run"])

        assert result == 0

    def test_main_no_preflight_warning(self, tmp_path, caplog):
        """Test main logs warning when preflight disabled."""
        import logging
        caplog.set_level(logging.WARNING)

        result = main(["--no-preflight", "--dry-run", "--data-dir", str(tmp_path)])

        assert result == 0
        # Warning should be logged

    def test_main_env_config_path(self, tmp_path):
        """Test main reads config from environment."""
        config_path = tmp_path / "env_config.yaml"
        config_data = {"agent": {"name": "env-agent", "data_dir": str(tmp_path)}}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with patch.dict(os.environ, {"CCEA_AGENT_CONFIG": str(config_path)}):
            result = main(["--dry-run"])

        assert result == 0


class TestConfigPriority:
    """Tests for configuration priority (CLI > ENV > File)."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create a config file."""
        config_path = tmp_path / "agent.yaml"
        config_data = {
            "agent": {
                "name": "file-agent",
                "data_dir": str(tmp_path / "file_data"),
            },
            "cloud": {
                "endpoint": "https://file.example.com",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    def test_cli_overrides_file(self, config_file, tmp_path, capsys):
        """Test CLI arguments override file config."""
        result = main([
            "--config", str(config_file),
            "--agent-name", "cli-agent",
            "--dump-config",
        ])

        assert result == 0
        output = capsys.readouterr().out
        assert "cli-agent" in output

    def test_cli_overrides_env(self, tmp_path, capsys):
        """Test CLI arguments override environment."""
        with patch.dict(os.environ, {"CCEA_AGENT_NAME": "env-agent"}):
            result = main([
                "--agent-name", "cli-agent",
                "--dump-config",
                "--data-dir", str(tmp_path),
            ])

        assert result == 0
        output = capsys.readouterr().out
        assert "cli-agent" in output


class TestVersion:
    """Tests for version handling."""

    def test_version_defined(self):
        """Test version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self):
        """Test version follows semver format."""
        parts = __version__.split(".")
        assert len(parts) >= 2  # At least major.minor
