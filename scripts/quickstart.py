#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/quickstart.py
Quick Start CLI for TradingBot2.

Provides easy setup and verification for new users.

Usage:
    python scripts/quickstart.py check                    # Check setup
    python scripts/quickstart.py check --asset crypto     # Check crypto setup
    python scripts/quickstart.py list                     # List presets
    python scripts/quickstart.py info <preset>            # Show preset info
    python scripts/quickstart.py run <preset>             # Run backtest
    python scripts/quickstart.py train <preset>           # Train model

Presets:
    crypto_momentum    - Crypto spot momentum (Binance)
    equity_swing       - US Equity swing trading (Alpaca)
    forex_carry        - Forex carry + momentum (OANDA)
    crypto_perp        - Crypto perpetual futures (Binance)
    cme_index          - CME equity index futures (IB)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
QUICKSTART_DIR = PROJECT_ROOT / "configs" / "quickstart"

PRESETS = {
    "crypto_momentum": {
        "name": "Crypto Spot Momentum",
        "asset_class": "crypto",
        "config": "crypto_momentum.yaml",
        "description": "Trend-following momentum strategy on 4H crypto data (BTC/ETH)",
        "difficulty": "⭐⭐ Beginner",
        "trading_hours": "24/7",
        "data_vendor": "binance",
        "fees": "Maker 2bps / Taker 4bps",
        "required_env": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
        "data_path": "data/train/crypto/*.parquet",
        "sample_symbols": ["BTCUSDT", "ETHUSDT"],
    },
    "equity_swing": {
        "name": "US Equity Swing",
        "asset_class": "equity",
        "config": "equity_swing.yaml",
        "description": "Mean-reversion strategy on US equities (S&P 500 stocks)",
        "difficulty": "⭐⭐ Beginner",
        "trading_hours": "NYSE 9:30-16:00 ET",
        "data_vendor": "alpaca",
        "fees": "$0 commission + regulatory",
        "required_env": ["ALPACA_API_KEY", "ALPACA_API_SECRET"],
        "data_path": "data/raw_stocks/*.parquet",
        "sample_symbols": ["SPY", "AAPL", "MSFT"],
    },
    "forex_carry": {
        "name": "Forex Carry + Momentum",
        "asset_class": "forex",
        "config": "forex_carry.yaml",
        "description": "Carry trade with momentum on major currency pairs",
        "difficulty": "⭐⭐⭐ Intermediate",
        "trading_hours": "Sun 5pm - Fri 5pm ET",
        "data_vendor": "oanda",
        "fees": "Spread-only (no commission)",
        "required_env": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
        "data_path": "data/forex/*.parquet",
        "sample_symbols": ["EUR_USD", "GBP_USD", "USD_JPY"],
    },
    "crypto_perp": {
        "name": "Crypto Perpetual Futures",
        "asset_class": "futures",
        "config": "crypto_perp.yaml",
        "description": "Funding rate arbitrage + momentum on crypto perpetuals",
        "difficulty": "⭐⭐⭐⭐ Advanced",
        "trading_hours": "24/7",
        "data_vendor": "binance",
        "fees": "Maker 2bps / Taker 4bps + funding",
        "required_env": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
        "data_path": "data/futures/crypto/*.parquet",
        "sample_symbols": ["BTCUSDT", "ETHUSDT"],
    },
    "cme_index": {
        "name": "CME Equity Index Futures",
        "asset_class": "futures",
        "config": "cme_index.yaml",
        "description": "Momentum on E-mini S&P 500 and NASDAQ 100 futures",
        "difficulty": "⭐⭐⭐⭐⭐ Expert",
        "trading_hours": "Globex Sun 18:00 - Fri 17:00 ET",
        "data_vendor": "ib",
        "fees": "$1.29/contract + broker",
        "required_env": [],  # IB uses local connection
        "data_path": "data/futures/cme/*.parquet",
        "sample_symbols": ["ES", "NQ"],
    },
}

ASSET_CLASSES = {
    "crypto": ["crypto_momentum"],
    "equity": ["equity_swing"],
    "forex": ["forex_carry"],
    "futures": ["crypto_perp", "cme_index"],
}


# =============================================================================
# Color Output
# =============================================================================


class Colors:
    """ANSI color codes."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = ""
        cls.BOLD = cls.DIM = cls.RESET = ""


# =============================================================================
# Check Functions
# =============================================================================


@dataclass
class QuickCheckResult:
    """Result of a quick check."""

    name: str
    passed: bool
    message: str
    fix_hint: Optional[str] = None


def check_config_exists(preset_name: str) -> QuickCheckResult:
    """Check if preset config exists."""
    preset = PRESETS.get(preset_name)
    if not preset:
        return QuickCheckResult(
            name="Config File",
            passed=False,
            message=f"Unknown preset: {preset_name}",
        )

    config_path = QUICKSTART_DIR / preset["config"]
    if config_path.exists():
        return QuickCheckResult(
            name="Config File",
            passed=True,
            message=f"Found {preset['config']}",
        )
    else:
        return QuickCheckResult(
            name="Config File",
            passed=False,
            message=f"Missing {preset['config']}",
            fix_hint="Check configs/quickstart/ directory",
        )


def check_data_exists(preset_name: str) -> QuickCheckResult:
    """Check if training data exists for preset."""
    preset = PRESETS.get(preset_name)
    if not preset:
        return QuickCheckResult(
            name="Training Data",
            passed=False,
            message=f"Unknown preset: {preset_name}",
        )

    data_pattern = preset["data_path"]
    data_dir = PROJECT_ROOT / Path(data_pattern).parent

    if not data_dir.exists():
        return QuickCheckResult(
            name="Training Data",
            passed=False,
            message=f"Data directory not found: {data_dir.relative_to(PROJECT_ROOT)}",
            fix_hint=get_data_download_hint(preset_name),
        )

    # Check for parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        return QuickCheckResult(
            name="Training Data",
            passed=False,
            message=f"No .parquet files in {data_dir.relative_to(PROJECT_ROOT)}",
            fix_hint=get_data_download_hint(preset_name),
        )

    return QuickCheckResult(
        name="Training Data",
        passed=True,
        message=f"Found {len(parquet_files)} data files",
    )


def check_env_vars(preset_name: str) -> QuickCheckResult:
    """Check if required environment variables are set."""
    preset = PRESETS.get(preset_name)
    if not preset:
        return QuickCheckResult(
            name="API Keys",
            passed=False,
            message=f"Unknown preset: {preset_name}",
        )

    required = preset.get("required_env", [])
    if not required:
        return QuickCheckResult(
            name="API Keys",
            passed=True,
            message="No API keys required (local connection)",
        )

    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        return QuickCheckResult(
            name="API Keys",
            passed=False,
            message=f"Missing: {', '.join(missing)}",
            fix_hint=f"Set environment variables: {', '.join(missing)}",
        )

    return QuickCheckResult(
        name="API Keys",
        passed=True,
        message=f"All {len(required)} keys configured",
    )


def check_dependencies(preset_name: str) -> QuickCheckResult:
    """Check if required packages are installed."""
    preset = PRESETS.get(preset_name)
    if not preset:
        return QuickCheckResult(
            name="Dependencies",
            passed=False,
            message=f"Unknown preset: {preset_name}",
        )

    vendor = preset.get("data_vendor", "")
    packages_to_check = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("stable_baselines3", "stable-baselines3"),
    ]

    # Add vendor-specific packages
    if vendor == "binance":
        packages_to_check.append(("binance", "python-binance"))
    elif vendor == "alpaca":
        packages_to_check.append(("alpaca_py", "alpaca-py"))
    elif vendor == "oanda":
        packages_to_check.append(("oandapyV20", "oandapyV20"))

    missing = []
    for import_name, pkg_name in packages_to_check:
        try:
            __import__(import_name.split(".")[0])
        except ImportError:
            missing.append(pkg_name)

    if missing:
        return QuickCheckResult(
            name="Dependencies",
            passed=False,
            message=f"Missing: {', '.join(missing)}",
            fix_hint=f"pip install {' '.join(missing)}",
        )

    return QuickCheckResult(
        name="Dependencies",
        passed=True,
        message="All core packages installed",
    )


def get_data_download_hint(preset_name: str) -> str:
    """Get data download command hint for preset."""
    hints = {
        "crypto_momentum": "python scripts/prepare_training_data.py --preset crypto_starter",
        "equity_swing": "python scripts/download_stock_data.py --symbols SPY AAPL MSFT --start 2023-01-01",
        "forex_carry": "python scripts/download_forex_data.py --pairs EURUSD GBPUSD --start 2023-01-01",
        "crypto_perp": "python scripts/download_funding_history.py --symbols BTCUSDT ETHUSDT --days 365",
        "cme_index": "python scripts/download_cme_data.py --symbols ES NQ --days 365",
    }
    return hints.get(preset_name, "Download data for this preset")


# =============================================================================
# Commands
# =============================================================================


def cmd_check(args: argparse.Namespace) -> int:
    """Run quick checks for a preset or asset class."""
    presets_to_check = []

    if args.preset:
        if args.preset not in PRESETS:
            print(f"{Colors.RED}Error: Unknown preset '{args.preset}'{Colors.RESET}")
            print(f"Available presets: {', '.join(PRESETS.keys())}")
            return 1
        presets_to_check = [args.preset]
    elif args.asset:
        if args.asset not in ASSET_CLASSES:
            print(f"{Colors.RED}Error: Unknown asset class '{args.asset}'{Colors.RESET}")
            print(f"Available: {', '.join(ASSET_CLASSES.keys())}")
            return 1
        presets_to_check = ASSET_CLASSES[args.asset]
    else:
        presets_to_check = list(PRESETS.keys())

    all_passed = True

    for preset_name in presets_to_check:
        preset = PRESETS[preset_name]
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}  {preset['name']} ({preset_name}){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")

        checks = [
            check_config_exists(preset_name),
            check_dependencies(preset_name),
            check_env_vars(preset_name),
            check_data_exists(preset_name),
        ]

        for check in checks:
            if check.passed:
                status = f"{Colors.GREEN}✓{Colors.RESET}"
            else:
                status = f"{Colors.RED}✗{Colors.RESET}"
                all_passed = False

            print(f"  {status} {check.name}: {check.message}")
            if not check.passed and check.fix_hint:
                print(f"      {Colors.YELLOW}Hint: {check.fix_hint}{Colors.RESET}")

    # Summary
    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.RESET}")
        print(f"\n{Colors.CYAN}Next steps:{Colors.RESET}")
        if len(presets_to_check) == 1:
            print(
                f"  python script_backtest.py --config configs/quickstart/{PRESETS[presets_to_check[0]]['config']}"
            )
        else:
            print("  python scripts/quickstart.py run <preset>")
        return 0
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}✗ Some checks failed. Please fix issues above.{Colors.RESET}"
        )
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available presets."""
    print(f"\n{Colors.BOLD}Available Quick Start Presets:{Colors.RESET}\n")

    # Group by asset class
    for asset_class, preset_names in ASSET_CLASSES.items():
        print(f"{Colors.BLUE}{Colors.BOLD}{asset_class.upper()}{Colors.RESET}")
        print(f"{'─' * 40}")

        for preset_name in preset_names:
            preset = PRESETS[preset_name]
            print(f"  {Colors.CYAN}{preset_name}{Colors.RESET}")
            print(f"    {preset['description']}")
            print(f"    Difficulty: {preset['difficulty']}")
            print()

    print(f"{Colors.DIM}Use 'python scripts/quickstart.py info <preset>' for details{Colors.RESET}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info about a preset."""
    preset_name = args.preset
    if preset_name not in PRESETS:
        print(f"{Colors.RED}Error: Unknown preset '{preset_name}'{Colors.RESET}")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        return 1

    preset = PRESETS[preset_name]

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {preset['name']}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")

    print(f"{Colors.BOLD}Description:{Colors.RESET}")
    print(f"  {preset['description']}\n")

    print(f"{Colors.BOLD}Details:{Colors.RESET}")
    print(f"  Asset Class:    {preset['asset_class']}")
    print(f"  Difficulty:     {preset['difficulty']}")
    print(f"  Trading Hours:  {preset['trading_hours']}")
    print(f"  Data Vendor:    {preset['data_vendor']}")
    print(f"  Fee Structure:  {preset['fees']}")
    print(f"  Symbols:        {', '.join(preset['sample_symbols'])}")
    print()

    print(f"{Colors.BOLD}Config File:{Colors.RESET}")
    print(f"  configs/quickstart/{preset['config']}")
    print()

    if preset.get("required_env"):
        print(f"{Colors.BOLD}Required Environment Variables:{Colors.RESET}")
        for var in preset["required_env"]:
            is_set = "✓" if os.environ.get(var) else "✗"
            color = Colors.GREEN if os.environ.get(var) else Colors.RED
            print(f"  {color}{is_set}{Colors.RESET} {var}")
        print()

    print(f"{Colors.BOLD}Quick Commands:{Colors.RESET}")
    print(f"  # Download data")
    print(f"  {get_data_download_hint(preset_name)}")
    print()
    print(f"  # Run backtest")
    print(f"  python script_backtest.py --config configs/quickstart/{preset['config']}")
    print()
    print(f"  # Train model")
    print(f"  python train_model_multi_patch.py --config configs/quickstart/{preset['config']}")
    print()

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run backtest for a preset."""
    preset_name = args.preset
    if preset_name not in PRESETS:
        print(f"{Colors.RED}Error: Unknown preset '{preset_name}'{Colors.RESET}")
        return 1

    preset = PRESETS[preset_name]
    config_path = QUICKSTART_DIR / preset["config"]

    if not config_path.exists():
        print(f"{Colors.RED}Error: Config not found: {config_path}{Colors.RESET}")
        return 1

    print(f"\n{Colors.CYAN}Running backtest for {preset['name']}...{Colors.RESET}\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "script_backtest.py"),
        "--config",
        str(config_path),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        return 130


def cmd_train(args: argparse.Namespace) -> int:
    """Train model for a preset."""
    preset_name = args.preset
    if preset_name not in PRESETS:
        print(f"{Colors.RED}Error: Unknown preset '{preset_name}'{Colors.RESET}")
        return 1

    preset = PRESETS[preset_name]
    config_path = QUICKSTART_DIR / preset["config"]

    if not config_path.exists():
        print(f"{Colors.RED}Error: Config not found: {config_path}{Colors.RESET}")
        return 1

    print(f"\n{Colors.CYAN}Training model for {preset['name']}...{Colors.RESET}\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train_model_multi_patch.py"),
        "--config",
        str(config_path),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
        return 130


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick Start CLI for TradingBot2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/quickstart.py check                    # Check all presets
    python scripts/quickstart.py check --asset crypto     # Check crypto presets
    python scripts/quickstart.py check --preset equity_swing  # Check specific preset
    python scripts/quickstart.py list                     # List available presets
    python scripts/quickstart.py info crypto_momentum     # Show preset details
    python scripts/quickstart.py run crypto_momentum      # Run backtest
    python scripts/quickstart.py train crypto_momentum    # Train model

Presets:
    crypto_momentum  - Crypto spot momentum (Binance, 4H)
    equity_swing     - US Equity swing trading (Alpaca)
    forex_carry      - Forex carry + momentum (OANDA)
    crypto_perp      - Crypto perpetual futures (Binance)
    cme_index        - CME equity index futures (IB)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # check command
    check_parser = subparsers.add_parser("check", help="Check setup for presets")
    check_parser.add_argument("--preset", "-p", help="Check specific preset")
    check_parser.add_argument(
        "--asset",
        "-a",
        choices=list(ASSET_CLASSES.keys()),
        help="Check all presets for asset class",
    )

    # list command
    subparsers.add_parser("list", help="List available presets")

    # info command
    info_parser = subparsers.add_parser("info", help="Show preset details")
    info_parser.add_argument("preset", choices=list(PRESETS.keys()), help="Preset name")

    # run command
    run_parser = subparsers.add_parser("run", help="Run backtest for preset")
    run_parser.add_argument("preset", choices=list(PRESETS.keys()), help="Preset name")

    # train command
    train_parser = subparsers.add_parser("train", help="Train model for preset")
    train_parser.add_argument("preset", choices=list(PRESETS.keys()), help="Preset name")

    args = parser.parse_args()

    # Disable colors for non-TTY
    if not sys.stdout.isatty():
        Colors.disable()

    if args.command == "check":
        return cmd_check(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "train":
        return cmd_train(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
