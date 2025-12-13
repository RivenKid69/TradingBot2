# -*- coding: utf-8 -*-
"""
Configuration Contracts.

Defines configuration structures shared between Cloud and Agent.
These are the non-sensitive configuration options that control
strategy and execution behavior.

IMPORTANT: This module MUST NOT contain any secret/credential handling.
Secrets are managed ONLY in the Agent zone's Local Vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set


class ExecutionMode(str, Enum):
    """Execution mode for strategy."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class RiskLevel(str, Enum):
    """Risk level classification."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ChangeClass(str, Enum):
    """
    Classification of configuration changes.

    From Design Doc 7.1 - determines approval requirements.
    """

    # Trading-impacting - requires local approval
    TRADING_IMPACTING = "trading_impacting"

    # Operational - may be applied remotely
    OPERATIONAL = "operational"

    # Informational - no approval needed
    INFORMATIONAL = "informational"


# Fields that are TRADING_IMPACTING (require local approval)
TRADING_IMPACTING_FIELDS: Final[Set[str]] = frozenset(
    [
        "symbols",
        "universe",
        "risk_limits",
        "max_position_size",
        "max_daily_loss",
        "execution_mode",
        "broker_account",
        "strategy_params",
        "model_version",
        "schedule",
    ]
)


@dataclass
class RiskConfig:
    """
    Risk configuration parameters.

    These define the risk boundaries for strategy execution.
    In Agent zone, these are enforced by Policy Firewall.
    """

    # Position limits
    max_position_size: Decimal = Decimal("100000")
    max_position_pct: Decimal = Decimal("0.1")  # 10% of portfolio

    # Loss limits
    max_daily_loss: Decimal = Decimal("1000")
    max_daily_loss_pct: Decimal = Decimal("0.02")  # 2%
    max_drawdown_pct: Decimal = Decimal("0.1")  # 10%

    # Order limits
    max_order_size: Decimal = Decimal("10000")
    max_orders_per_minute: int = 10
    max_orders_per_day: int = 100

    # Exposure limits
    max_gross_exposure: Decimal = Decimal("1000000")
    max_net_exposure: Decimal = Decimal("500000")

    # Concentration limits
    max_single_symbol_pct: Decimal = Decimal("0.25")  # 25%

    # Risk level
    risk_level: RiskLevel = RiskLevel.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_position_size": str(self.max_position_size),
            "max_position_pct": str(self.max_position_pct),
            "max_daily_loss": str(self.max_daily_loss),
            "max_daily_loss_pct": str(self.max_daily_loss_pct),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "max_order_size": str(self.max_order_size),
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_orders_per_day": self.max_orders_per_day,
            "max_gross_exposure": str(self.max_gross_exposure),
            "max_net_exposure": str(self.max_net_exposure),
            "max_single_symbol_pct": str(self.max_single_symbol_pct),
            "risk_level": self.risk_level.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskConfig:
        """Create from dictionary."""
        return cls(
            max_position_size=Decimal(str(data.get("max_position_size", "100000"))),
            max_position_pct=Decimal(str(data.get("max_position_pct", "0.1"))),
            max_daily_loss=Decimal(str(data.get("max_daily_loss", "1000"))),
            max_daily_loss_pct=Decimal(str(data.get("max_daily_loss_pct", "0.02"))),
            max_drawdown_pct=Decimal(str(data.get("max_drawdown_pct", "0.1"))),
            max_order_size=Decimal(str(data.get("max_order_size", "10000"))),
            max_orders_per_minute=int(data.get("max_orders_per_minute", 10)),
            max_orders_per_day=int(data.get("max_orders_per_day", 100)),
            max_gross_exposure=Decimal(str(data.get("max_gross_exposure", "1000000"))),
            max_net_exposure=Decimal(str(data.get("max_net_exposure", "500000"))),
            max_single_symbol_pct=Decimal(str(data.get("max_single_symbol_pct", "0.25"))),
            risk_level=RiskLevel(data.get("risk_level", "moderate")),
        )


@dataclass
class ExecutionConfig:
    """
    Execution configuration parameters.

    Controls HOW orders are executed (not IF - that's risk config).
    """

    # Slippage and fees
    expected_slippage_bps: Decimal = Decimal("5")  # 5 basis points
    max_slippage_bps: Decimal = Decimal("50")  # 50 basis points

    # Order execution
    use_limit_orders: bool = True
    limit_offset_bps: Decimal = Decimal("2")
    time_in_force: str = "GTC"
    retry_on_reject: bool = True
    max_retries: int = 3

    # Timing
    min_order_interval_ms: int = 100
    order_timeout_ms: int = 30000

    # TWAP/VWAP
    use_algo_execution: bool = False
    algo_duration_minutes: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expected_slippage_bps": str(self.expected_slippage_bps),
            "max_slippage_bps": str(self.max_slippage_bps),
            "use_limit_orders": self.use_limit_orders,
            "limit_offset_bps": str(self.limit_offset_bps),
            "time_in_force": self.time_in_force,
            "retry_on_reject": self.retry_on_reject,
            "max_retries": self.max_retries,
            "min_order_interval_ms": self.min_order_interval_ms,
            "order_timeout_ms": self.order_timeout_ms,
            "use_algo_execution": self.use_algo_execution,
            "algo_duration_minutes": self.algo_duration_minutes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionConfig:
        """Create from dictionary."""
        return cls(
            expected_slippage_bps=Decimal(str(data.get("expected_slippage_bps", "5"))),
            max_slippage_bps=Decimal(str(data.get("max_slippage_bps", "50"))),
            use_limit_orders=bool(data.get("use_limit_orders", True)),
            limit_offset_bps=Decimal(str(data.get("limit_offset_bps", "2"))),
            time_in_force=str(data.get("time_in_force", "GTC")),
            retry_on_reject=bool(data.get("retry_on_reject", True)),
            max_retries=int(data.get("max_retries", 3)),
            min_order_interval_ms=int(data.get("min_order_interval_ms", 100)),
            order_timeout_ms=int(data.get("order_timeout_ms", 30000)),
            use_algo_execution=bool(data.get("use_algo_execution", False)),
            algo_duration_minutes=int(data.get("algo_duration_minutes", 5)),
        )


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration.

    Combines all configuration aspects for a strategy deployment.
    """

    # Identity
    strategy_id: str
    strategy_version: str

    # Trading universe
    symbols: List[str] = field(default_factory=list)

    # Risk and execution
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Mode
    mode: ExecutionMode = ExecutionMode.PAPER

    # Strategy-specific parameters (non-sensitive)
    params: Dict[str, Any] = field(default_factory=dict)

    # Schedule (cron expression or always-on)
    schedule: str = "always"

    # Enabled state
    enabled: bool = True

    def get_change_class(self, field_name: str) -> ChangeClass:
        """Determine change class for a field."""
        if field_name in TRADING_IMPACTING_FIELDS:
            return ChangeClass.TRADING_IMPACTING
        return ChangeClass.OPERATIONAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "symbols": self.symbols,
            "risk": self.risk.to_dict(),
            "execution": self.execution.to_dict(),
            "mode": self.mode.value,
            "params": self.params,
            "schedule": self.schedule,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyConfig:
        """Create from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            strategy_version=data.get("strategy_version", "1.0.0"),
            symbols=data.get("symbols", []),
            risk=RiskConfig.from_dict(data.get("risk", {})),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            mode=ExecutionMode(data.get("mode", "paper")),
            params=data.get("params", {}),
            schedule=data.get("schedule", "always"),
            enabled=data.get("enabled", True),
        )


# Type alias for config contract
ConfigContract = StrategyConfig
