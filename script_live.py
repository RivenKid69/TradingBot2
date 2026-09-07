"""Run realtime signaler using :mod:`service_signal_runner`.

This script is the unified entry point for live trading, supporting
crypto (Binance), equity (Alpaca), and forex (OANDA) markets through
the --asset-class option.

Phase 9: Live Trading Improvements (2025-11-27)
- Explicit asset_class CLI support
- Auto-detection from config
- Asset-class-specific defaults

Phase 6: Forex Integration (2025-11-30)
- Added forex asset class support
- OANDA broker integration
- Forex session-aware routing
- Position synchronization
- Swap cost tracking
"""

from __future__ import annotations

import argparse
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from pydantic import BaseModel

from services.universe import get_symbols
from core_config import StateConfig, load_config
from service_signal_runner import from_config
from runtime_trade_defaults import (
    DEFAULT_RUNTIME_TRADE_PATH,
    load_runtime_trade_defaults,
    merge_runtime_trade_defaults,
)

# Forex services (Phase 6 integration)
try:
    from services.forex_risk_guards import (
        ForexMarginGuard,
        ForexLeverageGuard,
        SwapCostTracker,
        create_forex_margin_guard,
        create_forex_leverage_guard,
        create_swap_cost_tracker,
    )
    from services.forex_position_sync import (
        ForexPositionSynchronizer,
        SyncConfig,
        SyncResult,
        ReconciliationExecutor,
    )
    from services.forex_session_router import (
        ForexSessionRouter,
        ForexSessionType,
        get_current_forex_session,
        is_forex_market_open,
        create_forex_session_router,
        ROLLOVER_HOUR_ET,
        ROLLOVER_KEEPOUT_MINUTES,
    )

    FOREX_SERVICES_AVAILABLE = True
except ImportError:
    FOREX_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Asset Class Constants and Defaults
# =============================================================================

ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASS_EQUITY = "equity"
ASSET_CLASS_FOREX = "forex"
VALID_ASSET_CLASSES = (ASSET_CLASS_CRYPTO, ASSET_CLASS_EQUITY, ASSET_CLASS_FOREX)

# Asset-class-specific execution defaults
ASSET_CLASS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    ASSET_CLASS_CRYPTO: {
        "slippage_bps": 5.0,
        "limit_offset_bps": 10.0,
        "tif": "GTC",
        "extended_hours": False,  # Not applicable (24/7)
        "default_vendor": "binance",
    },
    ASSET_CLASS_EQUITY: {
        "slippage_bps": 2.0,  # Tighter spreads in regulated markets
        "limit_offset_bps": 5.0,
        "tif": "DAY",
        "extended_hours": False,  # Default to regular hours
        "default_vendor": "alpaca",
    },
    ASSET_CLASS_FOREX: {
        "slippage_bps": 0.5,  # Very tight spreads for major pairs (EUR/USD ~0.1-0.5 pip)
        "limit_offset_bps": 2.0,  # Conservative offset for limit orders
        "tif": "GTC",  # Good-til-cancelled (forex 24/5)
        "extended_hours": True,  # Always 24/5 trading
        "default_vendor": "oanda",
        # Forex-specific defaults (CFTC/NFA US rules)
        "max_leverage": 50,  # 50:1 for major pairs under CFTC
        "margin_call_level": 0.5,  # 50% margin call
        "rollover_time_utc": 21,  # 5pm ET = 21:00 UTC (winter), 22:00 (summer)
        "sync_interval_sec": 30.0,  # Position sync interval
        "auto_reconcile": True,  # Enable auto-reconciliation
    },
}

# Vendor to asset class mapping for auto-detection
VENDOR_TO_ASSET_CLASS: Dict[str, str] = {
    "binance": ASSET_CLASS_CRYPTO,
    "alpaca": ASSET_CLASS_EQUITY,
    "polygon": ASSET_CLASS_EQUITY,
    "oanda": ASSET_CLASS_FOREX,
}

try:
    from box import Box  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Box = None  # type: ignore


# =============================================================================
# Asset Class Detection and Application
# =============================================================================


def detect_asset_class(cfg_dict: Dict[str, Any]) -> str:
    """
    Auto-detect asset class from configuration.

    Priority:
    1. Explicit asset_class field in config
    2. exchange.vendor mapping
    3. exchange.market_type
    4. Default to 'crypto' for backward compatibility

    Args:
        cfg_dict: Configuration dictionary

    Returns:
        Asset class string ('crypto', 'equity', or 'forex')
    """
    # Priority 1: Explicit asset_class
    asset_class = cfg_dict.get("asset_class")
    if asset_class and asset_class in VALID_ASSET_CLASSES:
        return asset_class

    # Priority 2: Exchange vendor mapping
    exchange = cfg_dict.get("exchange", {})
    vendor = exchange.get("vendor", "").lower()
    if vendor in VENDOR_TO_ASSET_CLASS:
        return VENDOR_TO_ASSET_CLASS[vendor]

    # Priority 3: Market type
    market_type = exchange.get("market_type", "").upper()
    if market_type in ("EQUITY", "STOCK"):
        return ASSET_CLASS_EQUITY
    if market_type in ("CRYPTO", "CRYPTO_SPOT", "CRYPTO_FUTURES"):
        return ASSET_CLASS_CRYPTO
    if market_type in ("FOREX", "FX", "CURRENCY"):
        return ASSET_CLASS_FOREX

    # Default: crypto for backward compatibility
    return ASSET_CLASS_CRYPTO


def apply_asset_class_defaults(
    cfg_dict: Dict[str, Any],
    asset_class: str,
    cli_extended_hours: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply asset-class-specific defaults to configuration.

    These defaults are applied only if the corresponding field is not
    already set in the config, ensuring explicit config values take precedence.

    Args:
        cfg_dict: Configuration dictionary
        asset_class: Asset class ('crypto' or 'equity')
        cli_extended_hours: Extended hours override from CLI (optional)

    Returns:
        Updated configuration dictionary
    """
    defaults = ASSET_CLASS_DEFAULTS.get(asset_class, ASSET_CLASS_DEFAULTS[ASSET_CLASS_CRYPTO])

    # Set asset_class in config
    cfg_dict["asset_class"] = asset_class

    # Apply execution defaults
    exec_block = dict(cfg_dict.get("execution", {}) or {})
    exec_params = dict(cfg_dict.get("execution_params", {}) or {})

    # slippage_bps
    if "slippage_bps" not in exec_params:
        exec_params["slippage_bps"] = defaults["slippage_bps"]

    # limit_offset_bps
    if "limit_offset_bps" not in exec_params:
        exec_params["limit_offset_bps"] = defaults["limit_offset_bps"]

    # tif (time in force)
    if "tif" not in exec_params:
        exec_params["tif"] = defaults["tif"]

    cfg_dict["execution_params"] = exec_params

    # Extended hours handling (equity only)
    if asset_class == ASSET_CLASS_EQUITY:
        # CLI override takes precedence
        if cli_extended_hours is not None:
            cfg_dict["extended_hours"] = cli_extended_hours
        elif "extended_hours" not in cfg_dict:
            cfg_dict["extended_hours"] = defaults["extended_hours"]

        # Also set in exchange config for adapter
        exchange = dict(cfg_dict.get("exchange", {}) or {})
        alpaca_cfg = dict(exchange.get("alpaca", {}) or {})

        if cli_extended_hours is not None:
            alpaca_cfg["extended_hours"] = cli_extended_hours
        elif "extended_hours" not in alpaca_cfg:
            alpaca_cfg["extended_hours"] = cfg_dict.get("extended_hours", False)

        exchange["alpaca"] = alpaca_cfg
        cfg_dict["exchange"] = exchange

    # Forex-specific configuration
    elif asset_class == ASSET_CLASS_FOREX:
        # Forex is always 24/5 (Sunday 5pm ET to Friday 5pm ET)
        cfg_dict["extended_hours"] = True

        # Set up OANDA exchange config
        exchange = dict(cfg_dict.get("exchange", {}) or {})
        oanda_cfg = dict(exchange.get("oanda", {}) or {})

        # Apply forex-specific defaults
        if "max_leverage" not in oanda_cfg:
            oanda_cfg["max_leverage"] = defaults.get("max_leverage", 50)
        if "margin_call_level" not in oanda_cfg:
            oanda_cfg["margin_call_level"] = defaults.get("margin_call_level", 0.5)
        if "rollover_time_utc" not in oanda_cfg:
            oanda_cfg["rollover_time_utc"] = defaults.get("rollover_time_utc", 21)

        exchange["oanda"] = oanda_cfg
        cfg_dict["exchange"] = exchange

        # Forex position sync configuration
        forex_cfg = dict(cfg_dict.get("forex", {}) or {})
        if "sync_interval_sec" not in forex_cfg:
            forex_cfg["sync_interval_sec"] = defaults.get("sync_interval_sec", 30.0)
        if "auto_reconcile" not in forex_cfg:
            forex_cfg["auto_reconcile"] = defaults.get("auto_reconcile", True)
        cfg_dict["forex"] = forex_cfg

    # Set data_vendor if not specified
    if not cfg_dict.get("data_vendor"):
        cfg_dict["data_vendor"] = defaults["default_vendor"]

    return cfg_dict


def _get_symbols_for_asset_class(
    args: argparse.Namespace,
    cfg_dict: Dict[str, Any],
    asset_class: str,
) -> list:
    """
    Get symbols list based on asset class and config.

    Args:
        args: CLI arguments
        cfg_dict: Configuration dictionary
        asset_class: Asset class

    Returns:
        List of symbols
    """
    # CLI override
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Try to get from config data section
    data_cfg = cfg_dict.get("data", {}) or {}
    symbols_path = data_cfg.get("symbols_path")

    if symbols_path:
        try:
            from pathlib import Path
            import json

            symbols_file = Path(symbols_path)
            if symbols_file.exists():
                with open(symbols_file, "r", encoding="utf-8") as f:
                    symbols_data = json.load(f)
                    if isinstance(symbols_data, list):
                        return [s.upper() for s in symbols_data]
                    if isinstance(symbols_data, dict) and "symbols" in symbols_data:
                        return [s.upper() for s in symbols_data["symbols"]]
        except Exception as e:
            logger.warning(f"Failed to load symbols from {symbols_path}: {e}")

    # Fallback to universe service
    return get_symbols()


def _apply_runtime_overrides(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI-provided runtime overrides to a config mapping."""

    def _require_non_negative(value: float, label: str) -> float:
        if value < 0:
            raise SystemExit(f"{label} must be non-negative")
        return float(value)

    exec_block = dict(cfg_dict.get("execution") or {})
    exec_changed = False

    if args.execution_mode:
        exec_block["mode"] = str(args.execution_mode).strip().lower()
        exec_changed = True

    if args.execution_bar_price is not None:
        bar_price = str(args.execution_bar_price or "").strip()
        if bar_price:
            exec_block["bar_price"] = bar_price
        else:
            exec_block.pop("bar_price", None)
        exec_changed = True

    if args.execution_min_step is not None:
        exec_block["min_rebalance_step"] = _require_non_negative(
            args.execution_min_step, "execution-min-step"
        )
        exec_changed = True

    if args.execution_safety_margin_bps is not None:
        exec_block["safety_margin_bps"] = _require_non_negative(
            args.execution_safety_margin_bps, "execution-safety-margin-bps"
        )
        exec_changed = True

    if args.execution_max_participation is not None:
        exec_block["max_participation"] = _require_non_negative(
            args.execution_max_participation, "execution-max-participation"
        )
        exec_changed = True

    if args.portfolio_equity_usd is not None:
        equity = _require_non_negative(args.portfolio_equity_usd, "portfolio-equity-usd")
        portfolio_block = dict(cfg_dict.get("portfolio") or {})
        portfolio_block["equity_usd"] = equity
        cfg_dict["portfolio"] = portfolio_block
        exec_portfolio = dict(exec_block.get("portfolio") or {})
        exec_portfolio["equity_usd"] = equity
        exec_block["portfolio"] = exec_portfolio
        exec_changed = True

    if any(
        value is not None
        for value in (
            args.costs_taker_fee_bps,
            args.costs_half_spread_bps,
            args.costs_impact_sqrt,
            args.costs_impact_linear,
            args.costs_turnover_cap_symbol_bps,
            args.costs_turnover_cap_symbol_usd,
            args.costs_turnover_cap_portfolio_bps,
            args.costs_turnover_cap_portfolio_usd,
            args.costs_turnover_cap_symbol_daily_bps,
            args.costs_turnover_cap_symbol_daily_usd,
            args.costs_turnover_cap_portfolio_daily_bps,
            args.costs_turnover_cap_portfolio_daily_usd,
        )
    ):
        costs_block = dict(cfg_dict.get("costs") or {})
        exec_costs = dict(exec_block.get("costs") or {})
        impact_block = dict(costs_block.get("impact") or {})
        exec_impact = dict(exec_costs.get("impact") or {})
        turnover_caps_block = dict(costs_block.get("turnover_caps") or {})
        exec_turnover_caps = dict(exec_costs.get("turnover_caps") or {})
        symbol_caps_block = dict(turnover_caps_block.get("per_symbol") or {})
        exec_symbol_caps_block = dict(exec_turnover_caps.get("per_symbol") or {})
        portfolio_caps_block = dict(turnover_caps_block.get("portfolio") or {})
        exec_portfolio_caps_block = dict(exec_turnover_caps.get("portfolio") or {})

        if args.costs_taker_fee_bps is not None:
            fee = _require_non_negative(args.costs_taker_fee_bps, "costs-taker-fee-bps")
            costs_block["taker_fee_bps"] = fee
            exec_costs["taker_fee_bps"] = fee

        if args.costs_half_spread_bps is not None:
            half = _require_non_negative(args.costs_half_spread_bps, "costs-half-spread-bps")
            costs_block["half_spread_bps"] = half
            exec_costs["half_spread_bps"] = half

        if args.costs_impact_sqrt is not None:
            sqrt_coeff = _require_non_negative(args.costs_impact_sqrt, "costs-impact-sqrt")
            impact_block["sqrt_coeff"] = sqrt_coeff
            exec_impact["sqrt_coeff"] = sqrt_coeff

        if args.costs_impact_linear is not None:
            linear_coeff = _require_non_negative(args.costs_impact_linear, "costs-impact-linear")
            impact_block["linear_coeff"] = linear_coeff
            exec_impact["linear_coeff"] = linear_coeff

        if args.costs_turnover_cap_symbol_bps is not None:
            symbol_caps_block["bps"] = _require_non_negative(
                args.costs_turnover_cap_symbol_bps, "costs-turnover-cap-symbol-bps"
            )
            exec_symbol_caps_block["bps"] = symbol_caps_block["bps"]

        if args.costs_turnover_cap_symbol_usd is not None:
            symbol_caps_block["usd"] = _require_non_negative(
                args.costs_turnover_cap_symbol_usd, "costs-turnover-cap-symbol-usd"
            )
            exec_symbol_caps_block["usd"] = symbol_caps_block["usd"]

        if args.costs_turnover_cap_symbol_daily_bps is not None:
            symbol_caps_block["daily_bps"] = _require_non_negative(
                args.costs_turnover_cap_symbol_daily_bps,
                "costs-turnover-cap-symbol-daily-bps",
            )
            exec_symbol_caps_block["daily_bps"] = symbol_caps_block["daily_bps"]

        if args.costs_turnover_cap_symbol_daily_usd is not None:
            symbol_caps_block["daily_usd"] = _require_non_negative(
                args.costs_turnover_cap_symbol_daily_usd,
                "costs-turnover-cap-symbol-daily-usd",
            )
            exec_symbol_caps_block["daily_usd"] = symbol_caps_block["daily_usd"]

        if args.costs_turnover_cap_portfolio_bps is not None:
            portfolio_caps_block["bps"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_bps,
                "costs-turnover-cap-portfolio-bps",
            )
            exec_portfolio_caps_block["bps"] = portfolio_caps_block["bps"]

        if args.costs_turnover_cap_portfolio_usd is not None:
            portfolio_caps_block["usd"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_usd,
                "costs-turnover-cap-portfolio-usd",
            )
            exec_portfolio_caps_block["usd"] = portfolio_caps_block["usd"]

        if args.costs_turnover_cap_portfolio_daily_bps is not None:
            portfolio_caps_block["daily_bps"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_daily_bps,
                "costs-turnover-cap-portfolio-daily-bps",
            )
            exec_portfolio_caps_block["daily_bps"] = portfolio_caps_block["daily_bps"]

        if args.costs_turnover_cap_portfolio_daily_usd is not None:
            portfolio_caps_block["daily_usd"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_daily_usd,
                "costs-turnover-cap-portfolio-daily-usd",
            )
            exec_portfolio_caps_block["daily_usd"] = portfolio_caps_block["daily_usd"]

        if impact_block:
            costs_block["impact"] = impact_block
        else:
            costs_block.pop("impact", None)

        if exec_impact:
            exec_costs["impact"] = exec_impact
        else:
            exec_costs.pop("impact", None)

        if symbol_caps_block:
            turnover_caps_block["per_symbol"] = symbol_caps_block
            exec_turnover_caps["per_symbol"] = exec_symbol_caps_block
        else:
            turnover_caps_block.pop("per_symbol", None)
            exec_turnover_caps.pop("per_symbol", None)

        if portfolio_caps_block:
            turnover_caps_block["portfolio"] = portfolio_caps_block
            exec_turnover_caps["portfolio"] = exec_portfolio_caps_block
        else:
            turnover_caps_block.pop("portfolio", None)
            exec_turnover_caps.pop("portfolio", None)

        if turnover_caps_block:
            costs_block["turnover_caps"] = turnover_caps_block
        else:
            costs_block.pop("turnover_caps", None)

        if exec_turnover_caps:
            exec_costs["turnover_caps"] = exec_turnover_caps
        else:
            exec_costs.pop("turnover_caps", None)

        cfg_dict["costs"] = costs_block
        if exec_costs:
            exec_block["costs"] = exec_costs
        else:
            exec_block.pop("costs", None)
        exec_changed = True

    if exec_changed:
        cfg_dict["execution"] = exec_block

    return cfg_dict


def _merge_state_config(state_obj: Any, payload: Mapping[str, Any]) -> Any:
    if not payload:
        return state_obj
    if isinstance(state_obj, BaseModel):
        return state_obj.copy(update=payload)
    if state_obj is None:
        try:
            return StateConfig.parse_obj(payload)
        except Exception:
            return payload
    if Box is not None and isinstance(state_obj, Box):
        state_obj.update(payload)
        return state_obj
    if isinstance(state_obj, dict):
        state_obj.update(payload)
        return state_obj
    for key, value in payload.items():
        try:
            setattr(state_obj, key, value)
        except Exception:
            continue
    return state_obj


def _reset_state_files(state_obj: Any) -> None:
    path_value = getattr(state_obj, "path", None)
    if path_value:
        p = Path(path_value)
        with suppress(Exception):
            p.unlink()
        for backup in p.parent.glob(f"{p.name}.bak*"):
            with suppress(Exception):
                backup.unlink()
        plain_backup = p.with_name(p.name + ".bak")
        with suppress(Exception):
            if plain_backup.exists():
                plain_backup.unlink()
        derived_lock = p.with_suffix(p.suffix + ".lock")
        with suppress(Exception):
            if derived_lock.exists():
                derived_lock.unlink()
    lock_value = getattr(state_obj, "lock_path", None)
    if lock_value:
        lock_path = Path(lock_value)
        with suppress(Exception):
            if lock_path.exists():
                lock_path.unlink()


def _ensure_state_dir(state_obj: Any) -> None:
    target_dir = getattr(state_obj, "dir", None)
    if not target_dir:
        path_value = getattr(state_obj, "path", None)
        if path_value:
            target_dir = Path(path_value).parent
    if not target_dir:
        return
    Path(target_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Unified live trading script for crypto, equity, and forex markets.\n\n"
            "Supports:\n"
            "  - Crypto: Binance (default)\n"
            "  - Equity: Alpaca (US stocks)\n"
            "  - Forex: OANDA (currency pairs)\n\n"
            "The asset class is auto-detected from config or can be explicitly set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ==========================================================================
    # Core arguments
    # ==========================================================================
    p.add_argument(
        "--config",
        default="configs/config_live.yaml",
        help="Path to YAML config file (default: configs/config_live.yaml)",
    )
    p.add_argument(
        "--state-config",
        default="configs/state.yaml",
        help="Path to state YAML config file",
    )
    p.add_argument(
        "--reset-state",
        action="store_true",
        help="Clear state files before starting",
    )
    p.add_argument(
        "--symbols",
        default="",
        help="Comma-separated list of symbols (overrides config)",
    )

    # ==========================================================================
    # Asset class arguments (Phase 9)
    # ==========================================================================
    asset_group = p.add_argument_group(
        "Asset class options",
        "Control asset-class-specific behavior for unified live trading",
    )
    asset_group.add_argument(
        "--asset-class",
        choices=["crypto", "equity", "forex"],
        default=None,
        help=(
            "Asset class to trade. If not specified, auto-detected from config. "
            "crypto=Binance, equity=Alpaca, forex=OANDA"
        ),
    )
    asset_group.add_argument(
        "--extended-hours",
        action="store_true",
        default=None,
        help=(
            "Enable extended hours trading (equity only). "
            "Pre-market: 4:00-9:30 AM ET, After-hours: 4:00-8:00 PM ET"
        ),
    )
    asset_group.add_argument(
        "--no-extended-hours",
        action="store_true",
        help="Disable extended hours trading (equity only). Regular hours: 9:30 AM-4:00 PM ET",
    )
    asset_group.add_argument(
        "--paper",
        action="store_true",
        default=None,
        help="Use paper trading (Alpaca only, default: True)",
    )
    asset_group.add_argument(
        "--live",
        action="store_true",
        help="Use live trading (Alpaca/OANDA, overrides --paper)",
    )
    asset_group.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Dry-run mode: validate config, connect to exchange, generate signals, "
            "but DO NOT execute any orders. Useful for testing setup safely."
        ),
    )

    # ==========================================================================
    # Forex-specific arguments (Phase 6)
    # ==========================================================================
    forex_group = p.add_argument_group(
        "Forex options",
        "Control forex-specific behavior for OANDA trading",
    )
    forex_group.add_argument(
        "--forex-sync-interval",
        type=float,
        default=None,
        help="Position sync interval in seconds (default: 30.0)",
    )
    forex_group.add_argument(
        "--forex-auto-reconcile",
        action="store_true",
        default=None,
        help="Enable automatic position reconciliation (default: True)",
    )
    forex_group.add_argument(
        "--forex-no-auto-reconcile",
        action="store_true",
        help="Disable automatic position reconciliation",
    )
    forex_group.add_argument(
        "--forex-max-leverage",
        type=int,
        default=None,
        help="Maximum leverage (CFTC default: 50 for majors, 20 for minors)",
    )
    forex_group.add_argument(
        "--forex-rollover-keepout-minutes",
        type=int,
        default=None,
        help="Minutes to avoid trading around 5pm ET rollover (default: 5)",
    )
    forex_group.add_argument(
        "--forex-session-filter",
        choices=["sydney", "tokyo", "london", "new_york", "overlap", "all"],
        default=None,
        help="Only trade during specific sessions (default: all)",
    )

    runtime_group = p.add_argument_group("Runtime overrides")
    runtime_group.add_argument(
        "--execution-mode",
        choices=["order", "bar"],
        help="Override execution.mode (order/bar)",
    )
    runtime_group.add_argument(
        "--execution-bar-price",
        help="Override execution.bar_price (empty string to clear)",
    )
    runtime_group.add_argument(
        "--execution-min-step",
        type=float,
        help="Override execution.min_rebalance_step (fraction, >=0)",
    )
    runtime_group.add_argument(
        "--execution-safety-margin-bps",
        type=float,
        help="Override execution.safety_margin_bps used by the bar executor",
    )
    runtime_group.add_argument(
        "--execution-max-participation",
        type=float,
        help="Override execution.max_participation (fraction of ADV, >=0)",
    )
    runtime_group.add_argument(
        "--portfolio-equity-usd",
        type=float,
        help="Override portfolio.equity_usd assumption (>=0)",
    )
    runtime_group.add_argument(
        "--costs-taker-fee-bps",
        type=float,
        help="Override costs.taker_fee_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-half-spread-bps",
        type=float,
        help="Override costs.half_spread_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-impact-sqrt",
        type=float,
        help="Override costs.impact.sqrt_coeff (>=0)",
    )
    runtime_group.add_argument(
        "--costs-impact-linear",
        type=float,
        help="Override costs.impact.linear_coeff (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-bps",
        type=float,
        help="Override costs.turnover_caps.per_symbol.bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-usd",
        type=float,
        help="Override costs.turnover_caps.per_symbol.usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-daily-bps",
        type=float,
        help="Override costs.turnover_caps.per_symbol.daily_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-daily-usd",
        type=float,
        help="Override costs.turnover_caps.per_symbol.daily_usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-bps",
        type=float,
        help="Override costs.turnover_caps.portfolio.bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-usd",
        type=float,
        help="Override costs.turnover_caps.portfolio.usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-daily-bps",
        type=float,
        help="Override costs.turnover_caps.portfolio.daily_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-daily-usd",
        type=float,
        help="Override costs.turnover_caps.portfolio.daily_usd (>=0)",
    )
    p.add_argument(
        "--runtime-trade-config",
        default=DEFAULT_RUNTIME_TRADE_PATH,
        help="Path to runtime_trade.yaml with execution defaults",
    )
    args = p.parse_args()

    # ==========================================================================
    # CCEA Architecture Runtime Guard (Phase 1.5)
    # ==========================================================================
    # This script is for DEVELOPMENT/TESTING ONLY.
    # Production CCEA deployments MUST use the Agent daemon:
    #   python -m packages.agent.daemon.agentd --config configs/agent.yaml
    #
    # See: docs/archive/LEGACY_STACKS.md
    # See: ARCHITECTURE.md (CCEA Protocol section)
    # ==========================================================================

    import os
    import sys

    # Check for production mode environment variable
    _ccea_mode = os.environ.get("CCEA_PRODUCTION_MODE", "").lower()
    _allow_legacy = os.environ.get("CCEA_ALLOW_LEGACY_LIVE", "").lower() in ("1", "true", "yes")

    if _ccea_mode in ("1", "true", "yes") and not _allow_legacy:
        sys.stderr.write(
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  ERROR: script_live.py is NOT for production CCEA deployments   ║\n"
            "║                                                                  ║\n"
            "║  Production live trading MUST use the Agent daemon:             ║\n"
            "║    python -m packages.agent.daemon.agentd --config agent.yaml   ║\n"
            "║                                                                  ║\n"
            "║  This script is retained for development/testing only.          ║\n"
            "║  Set CCEA_ALLOW_LEGACY_LIVE=1 to override (NOT recommended).    ║\n"
            "║                                                                  ║\n"
            "║  See: docs/archive/LEGACY_STACKS.md                             ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n"
            "\n"
        )
        sys.exit(1)

    # Always emit deprecation warning for awareness
    import warnings

    warnings.warn(
        "script_live.py is deprecated for production use. "
        "Use 'python -m packages.agent.daemon.agentd' for CCEA production deployments. "
        "See docs/archive/LEGACY_STACKS.md for migration guide.",
        DeprecationWarning,
        stacklevel=1,
    )

    # ==========================================================================
    # Load and prepare configuration
    # ==========================================================================

    try:
        with open(args.state_config, "r", encoding="utf-8") as f:
            state_data_raw = yaml.safe_load(f) or {}
    except Exception:
        state_data_raw = {}
    state_data = state_data_raw if isinstance(state_data_raw, Mapping) else {}

    cfg = load_config(args.config)
    cfg_dict = cfg.dict()

    # ==========================================================================
    # Phase 9: Asset class detection and defaults
    # ==========================================================================

    # Determine asset class (CLI override > config detection)
    if args.asset_class:
        asset_class = args.asset_class
        logger.info(f"Asset class from CLI: {asset_class}")
    else:
        asset_class = detect_asset_class(cfg_dict)
        logger.info(f"Auto-detected asset class: {asset_class}")

    # Determine extended hours setting
    cli_extended_hours: Optional[bool] = None
    if args.no_extended_hours:
        cli_extended_hours = False
    elif args.extended_hours:
        cli_extended_hours = True

    # Apply asset-class-specific defaults
    cfg_dict = apply_asset_class_defaults(cfg_dict, asset_class, cli_extended_hours)

    # Handle paper/live trading (Alpaca)
    if asset_class == ASSET_CLASS_EQUITY:
        exchange = dict(cfg_dict.get("exchange", {}) or {})
        alpaca_cfg = dict(exchange.get("alpaca", {}) or {})

        if args.live:
            alpaca_cfg["paper"] = False
            logger.warning("LIVE TRADING MODE enabled - real money at risk!")
        elif args.paper:
            alpaca_cfg["paper"] = True
            logger.info("Paper trading mode enabled")

        exchange["alpaca"] = alpaca_cfg
        cfg_dict["exchange"] = exchange

    # Handle forex-specific configuration (OANDA)
    elif asset_class == ASSET_CLASS_FOREX:
        if not FOREX_SERVICES_AVAILABLE:
            raise SystemExit(
                "Forex services not available. Ensure forex_risk_guards.py, "
                "forex_position_sync.py, and forex_session_router.py are present."
            )

        exchange = dict(cfg_dict.get("exchange", {}) or {})
        oanda_cfg = dict(exchange.get("oanda", {}) or {})
        forex_cfg = dict(cfg_dict.get("forex", {}) or {})

        # Paper/live mode for OANDA
        if args.live:
            oanda_cfg["practice"] = False
            logger.warning("FOREX LIVE TRADING MODE enabled - real money at risk!")
        elif args.paper:
            oanda_cfg["practice"] = True
            logger.info("Forex practice mode enabled")

        # CLI overrides for forex-specific settings
        if args.forex_sync_interval is not None:
            forex_cfg["sync_interval_sec"] = args.forex_sync_interval

        if args.forex_no_auto_reconcile:
            forex_cfg["auto_reconcile"] = False
        elif args.forex_auto_reconcile:
            forex_cfg["auto_reconcile"] = True

        if args.forex_max_leverage is not None:
            oanda_cfg["max_leverage"] = args.forex_max_leverage
            forex_cfg["max_leverage"] = args.forex_max_leverage

        if args.forex_rollover_keepout_minutes is not None:
            forex_cfg["rollover_keepout_minutes"] = args.forex_rollover_keepout_minutes

        if args.forex_session_filter is not None:
            forex_cfg["session_filter"] = args.forex_session_filter

        exchange["oanda"] = oanda_cfg
        cfg_dict["exchange"] = exchange
        cfg_dict["forex"] = forex_cfg

        # Get current session info
        session_info = get_current_forex_session()

        # Check rollover window at startup
        if session_info.in_rollover_window:
            logger.warning(
                f"Currently within rollover keepout window (around 5pm ET). "
                "Trading may be restricted."
            )

        # Check if forex market is open
        if not is_forex_market_open():
            logger.warning(
                "Forex market is currently CLOSED (weekend). " "Trading will resume Sunday 5pm ET."
            )

        # Log current session
        session_filter = forex_cfg.get("session_filter", "all")
        logger.info(
            f"Current forex session: {session_info.session.name}, "
            f"liquidity_factor={session_info.liquidity_factor:.2f}, "
            f"spread_mult={session_info.spread_multiplier:.2f}, "
            f"session_filter={session_filter}"
        )

        # Log forex-specific settings
        logger.info(
            f"Forex config: sync_interval={forex_cfg.get('sync_interval_sec', 30.0)}s, "
            f"auto_reconcile={forex_cfg.get('auto_reconcile', True)}, "
            f"max_leverage={oanda_cfg.get('max_leverage', 50)}"
        )

    # Get symbols (using asset-class-aware function)
    symbols = _get_symbols_for_asset_class(args, cfg_dict, asset_class)

    # ==========================================================================
    # Handle dry-run mode
    # ==========================================================================
    if args.dry_run:
        cfg_dict["dry_run"] = True
        # Set execution to dry-run mode (no actual orders)
        exec_block = dict(cfg_dict.get("execution", {}) or {})
        exec_block["dry_run"] = True
        cfg_dict["execution"] = exec_block
        logger.warning(
            "\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║                       DRY-RUN MODE ENABLED                       ║\n"
            "║                                                                  ║\n"
            "║  Signals will be generated and logged, but NO ORDERS will be    ║\n"
            "║  executed. Use this to validate your configuration safely.      ║\n"
            "║                                                                  ║\n"
            "║  To run with real orders, remove the --dry-run flag.            ║\n"
            "╚══════════════════════════════════════════════════════════════════╝"
        )
    else:
        cfg_dict["dry_run"] = False

    # Log asset class summary
    mode_str = "DRY-RUN" if args.dry_run else "LIVE"
    logger.info(
        f"{mode_str} trading starting: asset_class={asset_class}, "
        f"symbols={len(symbols)}, extended_hours={cfg_dict.get('extended_hours', False)}"
    )

    # ==========================================================================
    # Continue with standard configuration flow
    # ==========================================================================

    # Ensure symbols are set in data config
    data_cfg = dict(cfg_dict.get("data", {}) or {})
    data_cfg["symbols"] = symbols
    cfg_dict["data"] = data_cfg

    # Apply runtime trade defaults
    runtime_trade_defaults = load_runtime_trade_defaults(args.runtime_trade_config)
    cfg_dict = merge_runtime_trade_defaults(cfg_dict, runtime_trade_defaults)

    # Apply CLI overrides
    cfg_dict = _apply_runtime_overrides(cfg_dict, args)

    # Rebuild config from dict
    cfg = cfg.__class__.parse_obj(cfg_dict)
    cfg.data.symbols = symbols

    try:
        cfg.components.executor.params["symbol"] = symbols[0] if symbols else ""
    except Exception:
        pass

    # Handle state config
    if state_data:
        merged_state = _merge_state_config(cfg.state, state_data)
        if merged_state is not cfg.state:
            cfg.state = merged_state
    state_cfg = cfg.state

    if args.reset_state:
        _reset_state_files(state_cfg)

    if getattr(state_cfg, "enabled", False):
        _ensure_state_dir(state_cfg)

    # ==========================================================================
    # Start signal runner
    # ==========================================================================

    for report in from_config(cfg, snapshot_config_path=args.config):
        print(report)


if __name__ == "__main__":
    main()
