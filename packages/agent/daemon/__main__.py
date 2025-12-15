# -*- coding: utf-8 -*-
"""
CCEA Agent Daemon CLI Entrypoint.

Design Doc compliant daemon runner with:
- YAML/ENV config loading
- Graceful shutdown (SIGTERM, SIGINT)
- Heartbeat/poll/telemetry/reconciliation threads
- Signal handling for production deployment

Usage:
    python -m packages.agent.daemon --config configs/agent.yaml
    python -m packages.agent.daemon --cloud-endpoint https://cloud.example.com --enrollment-token TOKEN

Environment Variables:
    CCEA_AGENT_CONFIG - Path to config file
    CCEA_CLOUD_ENDPOINT - Cloud control plane URL
    CCEA_ENROLLMENT_TOKEN - One-time enrollment token
    CCEA_ACCESS_TOKEN - Access token (after enrollment)
    CCEA_DATA_DIR - Data directory path
    CCEA_LOG_LEVEL - Logging level (DEBUG, INFO, WARNING, ERROR)

References:
    - Design Doc Section 4.2: Agent Components
    - Design Doc Section 9: Agent runtime
    - Design Doc Section 22.1: Enrollment sequence
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from packages.agent.daemon.agentd import (
    AgentDaemon,
    DaemonConfig,
    DaemonState,
    HaltReason,
)
from packages.agent.daemon.kill_switch import (
    KillSwitchConfig,
    HaltReasonType,
    HaltSeverity,
    HaltAction,
)
from packages.agent.daemon.time_sync import TimeSyncConfig
from packages.agent.daemon.preflight import PreflightConfig
from packages.agent.daemon.degraded_mode import DegradedModeConfig, DegradedMode, DegradedModeAction
from packages.agent.daemon.telemetry_buffer import TelemetryBufferConfig
from packages.agent.daemon.keychain import KeychainConfig
from packages.agent.daemon.sandbox import SandboxConfig, SandboxType


# Version
__version__ = "1.0.0"

# Logger setup
logger = logging.getLogger("ccea.agentd")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup logging for daemon.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    logger.info(f"Loaded config from {config_path}")
    return config


def build_daemon_config(
    config_dict: Dict[str, Any],
    cli_args: argparse.Namespace,
) -> DaemonConfig:
    """
    Build DaemonConfig from config dict and CLI args.

    Priority: CLI args > ENV vars > Config file > Defaults

    Args:
        config_dict: Config from YAML file
        cli_args: Parsed CLI arguments

    Returns:
        DaemonConfig instance
    """
    # Helper to get value with priority
    def get_value(key: str, env_var: str, default: Any = None) -> Any:
        # CLI args have highest priority
        cli_value = getattr(cli_args, key.replace("-", "_"), None)
        if cli_value is not None:
            return cli_value
        # Then ENV vars
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return env_value
        # Then config file
        return config_dict.get(key, default)

    # Extract agent section
    agent_config = config_dict.get("agent", {})
    cloud_config = config_dict.get("cloud", {})
    components_config = config_dict.get("components", {})

    # Build data_dir Path
    data_dir_str = get_value(
        "data_dir",
        "CCEA_DATA_DIR",
        agent_config.get("data_dir", str(Path.home() / ".ccea")),
    )
    data_dir = Path(data_dir_str)

    # Build component configs
    kill_switch_cfg = None
    ks_config = components_config.get("kill_switch", {})
    if ks_config:
        kill_switch_cfg = KillSwitchConfig(
            max_daily_loss_pct=ks_config.get("max_daily_loss_pct", 0.30),
            max_drawdown_pct=ks_config.get("max_drawdown_pct", 0.50),
            max_broker_errors_per_minute=ks_config.get("max_broker_errors_per_minute", 5),
            max_broker_errors_per_hour=ks_config.get("max_broker_errors_per_hour", 20),
            max_consecutive_errors=ks_config.get("max_consecutive_errors", 3),
        )

    time_sync_cfg = None
    ts_config = components_config.get("time_sync", {})
    if ts_config:
        time_sync_cfg = TimeSyncConfig(
            max_drift_seconds=ts_config.get("max_drift_seconds", 5.0),
            check_interval_seconds=ts_config.get("check_interval_seconds", 60),
            ntp_servers=ts_config.get("ntp_servers", ["pool.ntp.org"]),
        )

    degraded_cfg = None
    dg_config = components_config.get("degraded_mode", {})
    if dg_config:
        degraded_cfg = DegradedModeConfig(
            cloud_unreachable_threshold_seconds=dg_config.get("cloud_unreachable_threshold_seconds", 120),
            data_feed_stale_threshold_seconds=dg_config.get("data_feed_stale_threshold_seconds", 30),
            auto_recover=dg_config.get("auto_recover", True),
        )

    telemetry_cfg = None
    tel_config = components_config.get("telemetry", {})
    if tel_config:
        telemetry_cfg = TelemetryBufferConfig(
            max_buffer_size=tel_config.get("max_buffer_size", 100000),
            max_age_days=tel_config.get("max_age_days", 7),
            batch_size=tel_config.get("batch_size", 100),
            flush_interval_seconds=tel_config.get("flush_interval_seconds", 30),
        )

    sandbox_cfg = None
    sb_config = components_config.get("sandbox", {})
    if sb_config:
        sandbox_type_str = sb_config.get("type", "process").upper()
        try:
            sandbox_type = SandboxType[sandbox_type_str]
        except KeyError:
            sandbox_type = SandboxType.PROCESS
        sandbox_cfg = SandboxConfig(
            sandbox_type=sandbox_type,
            cpu_limit=sb_config.get("cpu_limit"),
            memory_limit_mb=sb_config.get("memory_limit_mb"),
            timeout_seconds=sb_config.get("timeout_seconds", 3600),
            readonly_fs=sb_config.get("readonly_fs", False),
            egress_allowlist=sb_config.get("egress_allowlist", []),
        )

    keychain_cfg = None
    kc_config = components_config.get("keychain", {})
    if kc_config:
        keychain_cfg = KeychainConfig(
            use_keychain=kc_config.get("use_keychain", True),
            fallback_to_env=kc_config.get("fallback_to_env", True),
            fallback_to_file=kc_config.get("fallback_to_file", True),
            require_keychain_for_live=kc_config.get("require_keychain_for_live", False),
        )

    # Build main config
    return DaemonConfig(
        agent_id=get_value("agent_id", "CCEA_AGENT_ID", agent_config.get("agent_id")),
        agent_name=get_value("agent_name", "CCEA_AGENT_NAME", agent_config.get("name", "ccea-agent")),
        agent_version=agent_config.get("version", __version__),
        cloud_endpoint=get_value("cloud_endpoint", "CCEA_CLOUD_ENDPOINT", cloud_config.get("endpoint")),
        cloud_timeout_seconds=cloud_config.get("timeout_seconds", 30),
        heartbeat_interval_seconds=cloud_config.get("heartbeat_interval_seconds", 30),
        cloud_enrollment_token=get_value("enrollment_token", "CCEA_ENROLLMENT_TOKEN", cloud_config.get("enrollment_token")),
        cloud_access_token=get_value("access_token", "CCEA_ACCESS_TOKEN", cloud_config.get("access_token")),
        data_dir=data_dir,
        state_file=agent_config.get("state_file", "daemon_state.json"),
        kill_switch_config=kill_switch_cfg,
        time_sync_config=time_sync_cfg,
        degraded_mode_config=degraded_cfg,
        telemetry_config=telemetry_cfg,
        keychain_config=keychain_cfg,
        sandbox_config=sandbox_cfg,
        auto_recover=agent_config.get("auto_recover", True),
        require_preflight=agent_config.get("require_preflight", True),
        enable_telemetry=agent_config.get("enable_telemetry", True),
        safe_shutdown_timeout_seconds=agent_config.get("safe_shutdown_timeout_seconds", 30),
    )


class DaemonRunner:
    """
    Daemon runner with signal handling and graceful shutdown.

    Implements Design Doc Section 9.7/9.8:
    - Safe degraded modes
    - Graceful shutdown
    - Signal handling
    """

    def __init__(self, daemon: AgentDaemon):
        self.daemon = daemon
        self._shutdown_requested = False
        self._restart_requested = False

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        # SIGHUP for reload (Unix only)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle_reload_signal)

        # SIGUSR1 for status dump (Unix only)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._handle_status_signal)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

    def _handle_reload_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGHUP for config reload."""
        logger.info("Received SIGHUP, requesting restart with new config...")
        self._restart_requested = True
        self._shutdown_requested = True

    def _handle_status_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGUSR1 for status dump."""
        logger.info("Received SIGUSR1, dumping status...")
        status = self.daemon.get_status()
        logger.info(f"Daemon Status: {status}")

    def run(self) -> int:
        """
        Run the daemon main loop.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        self.setup_signal_handlers()

        # Initialize daemon
        logger.info("Initializing agent daemon...")
        if not self.daemon.initialize():
            logger.error("Failed to initialize daemon")
            return 1

        # Start daemon
        logger.info("Starting agent daemon...")
        success, error = self.daemon.start()
        if not success:
            logger.error(f"Failed to start daemon: {error}")
            return 1

        logger.info(f"Agent daemon started (agent_id={self.daemon.agent_id})")
        logger.info(f"State: {self.daemon.state.name}")

        # Main loop - wait for shutdown signal
        try:
            while not self._shutdown_requested:
                # Check daemon health
                if self.daemon.is_halted:
                    logger.warning("Daemon is in HALTED state")
                    # Wait for manual intervention or auto-recovery

                # Sleep with interrupt check
                for _ in range(10):  # 10 x 0.5s = 5s between checks
                    if self._shutdown_requested:
                        break
                    time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        # Graceful shutdown
        logger.info("Stopping agent daemon...")
        if self.daemon.stop(reason="shutdown_requested"):
            logger.info("Agent daemon stopped successfully")
            return 0 if not self._restart_requested else 75  # EX_TEMPFAIL for restart
        else:
            logger.error("Failed to stop daemon gracefully")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ccea-agentd",
        description="CCEA Agent Daemon - Local execution runtime for cloud-controlled strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with config file
  python -m packages.agent.daemon --config configs/agent.yaml

  # Start with inline options
  python -m packages.agent.daemon \\
    --cloud-endpoint https://cloud.example.com \\
    --enrollment-token YOUR_TOKEN

  # Start with environment variables
  CCEA_CLOUD_ENDPOINT=https://cloud.example.com \\
  CCEA_ENROLLMENT_TOKEN=YOUR_TOKEN \\
  python -m packages.agent.daemon

Environment Variables:
  CCEA_AGENT_CONFIG       Path to config file
  CCEA_CLOUD_ENDPOINT     Cloud control plane URL
  CCEA_ENROLLMENT_TOKEN   One-time enrollment token
  CCEA_ACCESS_TOKEN       Access token (after enrollment)
  CCEA_DATA_DIR           Data directory path
  CCEA_LOG_LEVEL          Logging level

For more information, see: docs/agent/INSTALLATION.md
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Config file
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: from CCEA_AGENT_CONFIG env)",
    )

    # Cloud connection
    parser.add_argument(
        "--cloud-endpoint",
        type=str,
        default=None,
        help="Cloud control plane endpoint URL",
    )
    parser.add_argument(
        "--enrollment-token",
        type=str,
        default=None,
        help="One-time enrollment token (TTL-limited)",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help="Access token (if already enrolled)",
    )

    # Agent identity
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Agent ID (generated if not provided)",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default=None,
        help="Agent name for identification",
    )

    # Data directory
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory for state/logs/artifacts (default: ~/.ccea)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: stdout only)",
    )

    # Behavior flags
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight checks (NOT recommended for production)",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable telemetry (local journal still maintained)",
    )

    # Debugging
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without starting",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Dump effective config and exit",
    )

    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for agent daemon.

    Args:
        args: Command line arguments (default: sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging
    log_level = parsed_args.log_level or os.environ.get("CCEA_LOG_LEVEL", "INFO")
    setup_logging(level=log_level, log_file=parsed_args.log_file)

    logger.info(f"CCEA Agent Daemon v{__version__}")

    # Load config file
    config_dict: Dict[str, Any] = {}
    config_path = parsed_args.config or os.environ.get("CCEA_AGENT_CONFIG")
    if config_path:
        config_path = Path(config_path)
        try:
            config_dict = load_config_file(config_path)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            return 1

    # Build daemon config
    try:
        daemon_config = build_daemon_config(config_dict, parsed_args)
    except Exception as e:
        logger.error(f"Failed to build config: {e}")
        return 1

    # Apply CLI overrides
    if parsed_args.no_preflight:
        daemon_config.require_preflight = False
        logger.warning("Preflight checks disabled - NOT recommended for production")

    if parsed_args.no_telemetry:
        daemon_config.enable_telemetry = False
        logger.warning("Telemetry disabled")

    # Dump config if requested
    if parsed_args.dump_config:
        import json
        print(json.dumps(daemon_config.to_dict(), indent=2, default=str))
        return 0

    # Validate minimum config
    if not daemon_config.cloud_endpoint:
        logger.warning("No cloud endpoint configured - running in standalone mode")

    # Dry run
    if parsed_args.dry_run:
        logger.info("Dry run - config validation successful")
        logger.info(f"Config: {daemon_config.to_dict()}")
        return 0

    # Create and run daemon
    daemon = AgentDaemon(config=daemon_config)
    runner = DaemonRunner(daemon)

    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
