#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script_futures_live.py
Entry point for futures live trading.

Supports:
- Crypto perpetual futures (Binance USDT-M)
- CME futures (ES, NQ, GC, CL via Interactive Brokers)

Features:
- Unified configuration for all futures types
- Auto-detection of futures type from config
- Position synchronization with exchange
- Real-time margin monitoring
- Funding rate tracking (crypto)
- ADL detection (crypto)
- Circuit breaker handling (CME)

Usage:
    # Crypto perpetual futures (Binance)
    python script_futures_live.py --config configs/config_live_futures.yaml

    # CME futures (Interactive Brokers)
    python script_futures_live.py --config configs/config_live_cme.yaml

    # Paper trading mode
    python script_futures_live.py --config configs/config_live_futures.yaml --paper

    # Specific symbols
    python script_futures_live.py --config configs/config_live_futures.yaml --symbols BTCUSDT ETHUSDT

Author: Trading Bot Team
Date: 2025-12-02
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_futures import FuturesType
from services.futures_live_runner import (
    FuturesLiveRunner,
    FuturesLiveRunnerConfig,
    LiveRunnerState,
    LiveRunnerEvent,
    create_crypto_futures_runner,
    create_cme_futures_runner,
)
from services.futures_position_sync import (
    FuturesPositionSynchronizer,
    FuturesSyncConfig,
)
from services.futures_funding_tracker import (
    FundingTrackerService,
    FundingTrackerConfig,
    create_funding_service,
)
from services.futures_margin_monitor import (
    FuturesMarginMonitor,
    MarginMonitorConfig,
    MarginLevel,
    MarginAlert,
)
from services.unified_futures_risk import (
    UnifiedFuturesRiskGuard,
    UnifiedRiskConfig,
    create_unified_risk_guard,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG_PATH = "configs/config_live_futures.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def detect_futures_type(config: Dict[str, Any]) -> FuturesType:
    """
    Detect futures type from configuration.

    Priority:
    1. Explicit futures_type field
    2. Exchange/vendor field
    3. Symbol patterns
    4. Default to crypto perpetual

    Args:
        config: Configuration dictionary

    Returns:
        FuturesType enum
    """
    # Check explicit futures_type
    futures_type_str = config.get("futures_type", "").upper()
    if futures_type_str:
        try:
            return FuturesType[futures_type_str]
        except KeyError:
            pass

    # Check exchange/vendor
    exchange = config.get("exchange", config.get("vendor", "")).lower()
    if exchange in ("binance", "binance_futures"):
        return FuturesType.CRYPTO_PERPETUAL
    elif exchange in ("ib", "interactive_brokers", "cme"):
        return FuturesType.INDEX_FUTURES

    # Check symbols for pattern matching
    symbols = config.get("symbols", [])
    if symbols:
        first_symbol = symbols[0].upper()
        # Crypto patterns: BTCUSDT, ETHUSDT, etc.
        if first_symbol.endswith(("USDT", "BUSD", "USD")):
            return FuturesType.CRYPTO_PERPETUAL
        # CME patterns: ES, NQ, GC, CL, etc.
        if first_symbol in (
            "ES",
            "NQ",
            "YM",
            "RTY",
            "GC",
            "SI",
            "CL",
            "NG",
            "6E",
            "6J",
            "ZN",
            "ZB",
        ):
            return FuturesType.INDEX_FUTURES

    # Default to crypto perpetual
    return FuturesType.CRYPTO_PERPETUAL


def apply_defaults(config: Dict[str, Any], futures_type: FuturesType) -> Dict[str, Any]:
    """
    Apply default values based on futures type.

    Args:
        config: Configuration dictionary
        futures_type: Futures type

    Returns:
        Updated configuration
    """
    defaults = {
        FuturesType.CRYPTO_PERPETUAL: {
            "exchange": "binance",
            "position_sync_interval_sec": 5.0,
            "margin_check_interval_sec": 10.0,
            "funding_check_interval_sec": 60.0,
            "enable_funding_tracking": True,
            "enable_adl_monitoring": True,
            "enable_circuit_breaker_monitoring": False,
            "max_leverage": 10,
        },
        FuturesType.INDEX_FUTURES: {
            "exchange": "ib",
            "position_sync_interval_sec": 5.0,
            "margin_check_interval_sec": 30.0,
            "funding_check_interval_sec": 0,  # CME doesn't have funding
            "enable_funding_tracking": False,
            "enable_adl_monitoring": False,
            "enable_circuit_breaker_monitoring": True,
            "max_leverage": 20,
        },
    }

    type_defaults = defaults.get(futures_type, {})

    # Apply defaults for missing keys
    for key, value in type_defaults.items():
        if key not in config:
            config[key] = value

    return config


# ============================================================================
# ADAPTER CREATION
# ============================================================================


def create_market_data_adapter(config: Dict[str, Any], futures_type: FuturesType):
    """
    Create market data adapter based on configuration.

    Args:
        config: Configuration dictionary
        futures_type: Futures type

    Returns:
        Market data adapter instance
    """
    exchange = config.get("exchange", "").lower()

    if futures_type == FuturesType.CRYPTO_PERPETUAL or exchange in ("binance", "binance_futures"):
        # Create Binance futures market data adapter
        from adapters.binance.market_data import BinanceMarketDataAdapter
        from adapters.models import ExchangeVendor

        api_key = config.get("api_key", os.environ.get("BINANCE_API_KEY", ""))
        api_secret = config.get("api_secret", os.environ.get("BINANCE_API_SECRET", ""))

        return BinanceMarketDataAdapter(
            vendor=ExchangeVendor.BINANCE,
            config={
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": config.get("paper_trading", True),
            },
        )

    elif futures_type == FuturesType.INDEX_FUTURES or exchange in ("ib", "interactive_brokers"):
        # Create IB market data adapter
        from adapters.ib.market_data import IBMarketDataAdapter
        from adapters.models import ExchangeVendor

        return IBMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={
                "host": config.get("ib_host", "127.0.0.1"),
                "port": config.get("ib_port", 7497 if config.get("paper_trading", True) else 7496),
                "client_id": config.get("ib_client_id", 1),
            },
        )

    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


def create_order_executor(config: Dict[str, Any], futures_type: FuturesType):
    """
    Create order execution adapter based on configuration.

    Args:
        config: Configuration dictionary
        futures_type: Futures type

    Returns:
        Order execution adapter instance
    """
    exchange = config.get("exchange", "").lower()

    if futures_type == FuturesType.CRYPTO_PERPETUAL or exchange in ("binance", "binance_futures"):
        # Create Binance futures order execution adapter
        from adapters.binance.futures_order_execution import BinanceFuturesOrderExecutionAdapter
        from adapters.models import ExchangeVendor

        api_key = config.get("api_key", os.environ.get("BINANCE_API_KEY", ""))
        api_secret = config.get("api_secret", os.environ.get("BINANCE_API_SECRET", ""))

        return BinanceFuturesOrderExecutionAdapter(
            vendor=ExchangeVendor.BINANCE,
            config={
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": config.get("paper_trading", True),
            },
        )

    elif futures_type == FuturesType.INDEX_FUTURES or exchange in ("ib", "interactive_brokers"):
        # Create IB order execution adapter
        from adapters.ib.order_execution import IBOrderExecutionAdapter
        from adapters.models import ExchangeVendor

        return IBOrderExecutionAdapter(
            vendor=ExchangeVendor.IB,
            config={
                "host": config.get("ib_host", "127.0.0.1"),
                "port": config.get("ib_port", 7497 if config.get("paper_trading", True) else 7496),
                "client_id": config.get("ib_client_id", 2),
            },
        )

    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


def create_signal_provider(config: Dict[str, Any]):
    """
    Create signal provider (ML model) based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Signal provider instance or None
    """
    model_path = config.get("model_path")
    if not model_path:
        logger.info("No model path specified, running without signal provider")
        return None

    # Load model and create signal provider
    # This is a placeholder - actual implementation depends on model type
    try:
        from service_signal_runner import create_signal_runner

        return create_signal_runner(
            model_path=model_path,
            config=config.get("model_config", {}),
        )
    except ImportError:
        logger.warning("Signal runner not available")
        return None
    except Exception as e:
        logger.error(f"Error creating signal provider: {e}")
        return None


# ============================================================================
# CALLBACKS
# ============================================================================


def create_event_callbacks(config: Dict[str, Any]) -> Dict[LiveRunnerEvent, List]:
    """Create event callbacks based on configuration."""
    callbacks = {}

    # Logging callbacks
    def log_event(event: LiveRunnerEvent, data: Dict):
        logger.info(f"Event: {event.name} - {data}")

    for event in LiveRunnerEvent:
        callbacks[event] = [log_event]

    # Add specific callbacks
    def on_margin_warning(event: LiveRunnerEvent, data: Dict):
        logger.warning(f"MARGIN WARNING: {data}")
        # Could send notification here

    def on_margin_critical(event: LiveRunnerEvent, data: Dict):
        logger.error(f"MARGIN CRITICAL: {data}")
        # Could trigger emergency actions here

    def on_adl_warning(event: LiveRunnerEvent, data: Dict):
        logger.warning(f"ADL WARNING: {data}")

    callbacks[LiveRunnerEvent.MARGIN_WARNING].append(on_margin_warning)
    callbacks[LiveRunnerEvent.MARGIN_CRITICAL].append(on_margin_critical)
    callbacks[LiveRunnerEvent.ADL_WARNING].append(on_adl_warning)

    return callbacks


# ============================================================================
# MAIN RUNNER
# ============================================================================


class FuturesLiveApplication:
    """Main application class for futures live trading."""

    def __init__(
        self,
        config: Dict[str, Any],
        paper_trading: bool = True,
    ):
        """
        Initialize application.

        Args:
            config: Configuration dictionary
            paper_trading: Enable paper trading mode
        """
        self._config = config
        self._paper_trading = paper_trading
        self._runner: Optional[FuturesLiveRunner] = None
        self._margin_monitor: Optional[FuturesMarginMonitor] = None
        self._shutdown_requested = False

        # Detect futures type
        self._futures_type = detect_futures_type(config)
        logger.info(f"Detected futures type: {self._futures_type.name}")

        # Apply defaults
        self._config = apply_defaults(config, self._futures_type)

        # Override paper trading if specified
        self._config["paper_trading"] = paper_trading

    def setup(self) -> None:
        """Set up all components."""
        logger.info("Setting up futures live application...")

        # Create adapters
        try:
            market_data = create_market_data_adapter(self._config, self._futures_type)
            order_executor = create_order_executor(self._config, self._futures_type)
            signal_provider = create_signal_provider(self._config)
        except Exception as e:
            logger.error(f"Error creating adapters: {e}")
            raise

        # Create runner config
        symbols = self._config.get("symbols", [])
        if not symbols:
            raise ValueError("No symbols specified in configuration")

        callbacks = create_event_callbacks(self._config)

        runner_config = FuturesLiveRunnerConfig(
            futures_type=self._futures_type,
            symbols=symbols,
            main_loop_interval_sec=float(self._config.get("main_loop_interval_sec", 1.0)),
            position_sync_interval_sec=float(self._config.get("position_sync_interval_sec", 5.0)),
            margin_check_interval_sec=float(self._config.get("margin_check_interval_sec", 10.0)),
            funding_check_interval_sec=float(self._config.get("funding_check_interval_sec", 60.0)),
            enable_position_sync=bool(self._config.get("enable_position_sync", True)),
            enable_margin_monitoring=bool(self._config.get("enable_margin_monitoring", True)),
            enable_funding_tracking=bool(self._config.get("enable_funding_tracking", True)),
            enable_adl_monitoring=bool(self._config.get("enable_adl_monitoring", True)),
            enable_circuit_breaker_monitoring=bool(
                self._config.get("enable_circuit_breaker_monitoring", True)
            ),
            max_reconnect_attempts=int(self._config.get("max_reconnect_attempts", 10)),
            strict_mode=bool(self._config.get("strict_mode", True)),
            max_leverage=int(self._config.get("max_leverage", 10)),
            paper_trading=self._paper_trading,
            event_callbacks=callbacks,
        )

        # Create runner
        self._runner = FuturesLiveRunner(
            config=runner_config,
            market_data=market_data,
            order_executor=order_executor,
            signal_provider=signal_provider,
        )

        logger.info(f"Futures live application setup complete: symbols={symbols}")

    def run(self) -> None:
        """Run the application."""
        if not self._runner:
            raise RuntimeError("Application not set up. Call setup() first.")

        logger.info("Starting futures live trading...")

        # Start runner
        self._runner.start()

        try:
            # Run main loop
            self._runner.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    async def run_async(self) -> None:
        """Run the application asynchronously."""
        if not self._runner:
            raise RuntimeError("Application not set up. Call setup() first.")

        logger.info("Starting async futures live trading...")

        # Start runner
        self._runner.start()

        try:
            # Run main loop
            await self._runner.run_async()
        except asyncio.CancelledError:
            logger.info("Async run cancelled")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shut down the application."""
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        logger.info("Shutting down futures live application...")

        if self._runner:
            self._runner.stop()

        if self._margin_monitor:
            self._margin_monitor.stop_background_monitoring()

        logger.info("Futures live application shut down complete")

    def get_status(self) -> Dict[str, Any]:
        """Get current application status."""
        if not self._runner:
            return {"status": "not_initialized"}

        health = self._runner.get_health_status()
        stats = self._runner.get_stats()

        return {
            "status": "running" if self._runner.is_running else "stopped",
            "state": health.state.name,
            "is_healthy": health.is_healthy,
            "open_positions": health.open_positions,
            "active_orders": health.active_orders,
            "margin_status": health.margin_status.name,
            "total_orders": stats.total_orders_submitted,
            "total_fills": stats.total_orders_filled,
            "total_syncs": stats.total_position_syncs,
            "warnings": health.warnings,
            "errors": health.errors,
        }


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

_app: Optional[FuturesLiveApplication] = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global _app
    logger.info(f"Received signal {signum}")
    if _app:
        _app.shutdown()
    sys.exit(0)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Futures Live Trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Override symbols from config",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading mode (disables paper)",
    )

    parser.add_argument(
        "--futures-type",
        type=str,
        choices=["crypto", "cme"],
        help="Override futures type detection",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async mode",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    """Main entry point."""
    global _app

    args = parse_args()

    # Set up logging
    setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Futures Live Trading")
    logger.info("=" * 60)

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration: {e}")
        return 1

    # Override config with CLI args
    if args.symbols:
        config["symbols"] = args.symbols

    if args.futures_type:
        if args.futures_type == "crypto":
            config["futures_type"] = "CRYPTO_PERPETUAL"
        elif args.futures_type == "cme":
            config["futures_type"] = "INDEX_FUTURES"

    # Determine paper/live mode
    paper_trading = True
    if args.live:
        paper_trading = False
    elif args.paper:
        paper_trading = True
    else:
        paper_trading = config.get("paper_trading", True)

    logger.info(f"Mode: {'PAPER' if paper_trading else 'LIVE'}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Symbols: {config.get('symbols', [])}")

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run application
    try:
        _app = FuturesLiveApplication(config, paper_trading=paper_trading)
        _app.setup()

        if args.use_async:
            asyncio.run(_app.run_async())
        else:
            _app.run()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        if _app:
            _app.shutdown()


if __name__ == "__main__":
    sys.exit(main())
