# -*- coding: utf-8 -*-
"""
Policy Firewall - Core policy enforcement.

Validates all trading actions against local policies.
Cloud configurations are applied ONLY if they don't violate local policy.

Key Principle:
    Local hard caps have PRIORITY over Cloud config.
    Cloud cannot raise risk limits beyond local caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set

from packages.shared.contracts.intent import OrderIntent, IntentType
from packages.shared.contracts.config import RiskConfig, ChangeClass


class ViolationType(str, Enum):
    """Types of policy violations."""

    HARD_CAP_EXCEEDED = "hard_cap_exceeded"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    PROHIBITED_SYMBOL = "prohibited_symbol"
    PROHIBITED_ACTION = "prohibited_action"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_CHANGE = "unauthorized_change"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class PolicyViolation:
    """
    Represents a policy violation.
    """

    violation_type: ViolationType
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    limit: Optional[Any] = None
    severity: str = "error"  # warning, error, critical
    timestamp: datetime = dataclass_field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_type": self.violation_type.value,
            "message": self.message,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "limit": str(self.limit) if self.limit is not None else None,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PolicyResult:
    """
    Result of policy check.
    """

    allowed: bool = True
    violations: List[PolicyViolation] = dataclass_field(default_factory=list)
    warnings: List[PolicyViolation] = dataclass_field(default_factory=list)
    modified_intent: Optional[OrderIntent] = None  # If intent was clamped

    def add_violation(self, violation: PolicyViolation) -> None:
        """Add violation and mark as not allowed."""
        if violation.severity == "warning":
            self.warnings.append(violation)
        else:
            self.violations.append(violation)
            self.allowed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "has_modified_intent": self.modified_intent is not None,
        }


@dataclass
class PolicyConfig:
    """
    Local policy configuration.

    These are the LOCAL limits that Cloud cannot override.
    """

    # Hard position limits (absolute maximums)
    max_position_size: Decimal = Decimal("1000000")
    max_position_pct: Decimal = Decimal("1.0")  # 100% - unlimited

    # Hard loss limits
    max_daily_loss: Decimal = Decimal("100000")
    max_drawdown_pct: Decimal = Decimal("0.5")  # 50%

    # Order limits
    max_order_size: Decimal = Decimal("100000")
    max_orders_per_minute: int = 60
    max_orders_per_day: int = 10000

    # Symbol restrictions
    allowed_symbols: Optional[Set[str]] = None  # None = all allowed
    prohibited_symbols: Set[str] = dataclass_field(default_factory=set)

    # Action restrictions
    allow_short: bool = True
    allow_leverage: bool = True
    allow_live_trading: bool = True

    # Auto-approve settings (LOCAL only)
    auto_approve_operational: bool = True
    auto_approve_threshold: Decimal = Decimal("0")  # 0 = never auto-approve trading_impacting

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_position_size": str(self.max_position_size),
            "max_position_pct": str(self.max_position_pct),
            "max_daily_loss": str(self.max_daily_loss),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "max_order_size": str(self.max_order_size),
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_orders_per_day": self.max_orders_per_day,
            "allowed_symbols": list(self.allowed_symbols) if self.allowed_symbols else None,
            "prohibited_symbols": list(self.prohibited_symbols),
            "allow_short": self.allow_short,
            "allow_leverage": self.allow_leverage,
            "allow_live_trading": self.allow_live_trading,
            "auto_approve_operational": self.auto_approve_operational,
            "auto_approve_threshold": str(self.auto_approve_threshold),
        }


class PolicyFirewall:
    """
    Policy Firewall - Main policy enforcement.

    Validates intents and configurations against local policy.
    Cloud cannot override local hard caps.

    Usage:
        firewall = PolicyFirewall(local_policy)

        # Check intent
        result = firewall.check_intent(intent, current_state)
        if not result.allowed:
            print(result.violations)

        # Check config change from Cloud
        result = firewall.check_config_change(cloud_config)
        if not result.allowed:
            reject_cloud_config()
    """

    def __init__(
        self,
        local_policy: Optional[PolicyConfig] = None,
        cloud_config: Optional[RiskConfig] = None,
    ):
        """
        Initialize policy firewall.

        Args:
            local_policy: Local policy (has priority)
            cloud_config: Cloud-provided risk config (must not exceed local)
        """
        self._local_policy = local_policy or PolicyConfig()
        self._cloud_config = cloud_config
        self._effective_config = self._compute_effective_config()

        # Rate limiting state
        self._orders_this_minute: int = 0
        self._orders_today: int = 0
        self._minute_start: datetime = datetime.utcnow()
        self._day_start: datetime = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Daily P&L tracking
        self._daily_pnl: Decimal = Decimal("0")

    def check_intent(
        self,
        intent: OrderIntent,
        current_position: Decimal = Decimal("0"),
        current_exposure: Decimal = Decimal("0"),
        account_equity: Decimal = Decimal("100000"),
    ) -> PolicyResult:
        """
        Check if intent is allowed by policy.

        Args:
            intent: OrderIntent to validate
            current_position: Current position in symbol
            current_exposure: Current total exposure
            account_equity: Current account equity

        Returns:
            PolicyResult with violations if any
        """
        result = PolicyResult()

        # Skip validation for passive intents
        if intent.is_passive:
            return result

        # Check symbol restrictions
        self._check_symbol(intent.symbol, result)

        # Check action restrictions
        self._check_action(intent, result)

        # Check quantity limits
        if intent.target_quantity:
            self._check_quantity(intent.target_quantity, result)

        # Check position limits
        self._check_position_limit(intent, current_position, account_equity, result)

        # Check rate limits
        self._check_rate_limits(result)

        return result

    def check_config_change(
        self,
        new_config: RiskConfig,
        change_class: ChangeClass = ChangeClass.TRADING_IMPACTING,
    ) -> PolicyResult:
        """
        Check if config change from Cloud is allowed.

        Cloud cannot raise limits above local policy.

        Args:
            new_config: New risk config from Cloud
            change_class: Classification of change

        Returns:
            PolicyResult with violations if config would exceed local limits
        """
        result = PolicyResult()

        local = self._local_policy

        # Check position size
        if new_config.max_position_size > local.max_position_size:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.HARD_CAP_EXCEEDED,
                    message="Cloud config exceeds local max_position_size",
                    field="max_position_size",
                    value=new_config.max_position_size,
                    limit=local.max_position_size,
                )
            )

        # Check daily loss
        if new_config.max_daily_loss > local.max_daily_loss:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.HARD_CAP_EXCEEDED,
                    message="Cloud config exceeds local max_daily_loss",
                    field="max_daily_loss",
                    value=new_config.max_daily_loss,
                    limit=local.max_daily_loss,
                )
            )

        # Check order size
        if new_config.max_order_size > local.max_order_size:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.HARD_CAP_EXCEEDED,
                    message="Cloud config exceeds local max_order_size",
                    field="max_order_size",
                    value=new_config.max_order_size,
                    limit=local.max_order_size,
                )
            )

        # Check if approval required
        if (
            change_class == ChangeClass.TRADING_IMPACTING
            and not self._local_policy.auto_approve_threshold > Decimal("0")
        ):
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.APPROVAL_REQUIRED,
                    message="Trading-impacting change requires local approval",
                    severity="warning",  # Not blocking, but needs approval
                )
            )

        return result

    def apply_cloud_config(self, cloud_config: RiskConfig) -> bool:
        """
        Apply Cloud config if it passes validation.

        Args:
            cloud_config: New risk config from Cloud

        Returns:
            True if applied, False if rejected
        """
        result = self.check_config_change(cloud_config)
        if not result.allowed:
            return False

        self._cloud_config = cloud_config
        self._effective_config = self._compute_effective_config()
        return True

    def update_daily_pnl(self, pnl_change: Decimal) -> PolicyResult:
        """
        Update daily P&L and check limits.

        Args:
            pnl_change: Change in P&L

        Returns:
            PolicyResult with kill switch trigger if needed
        """
        result = PolicyResult()
        self._daily_pnl += pnl_change

        # Check daily loss limit
        if self._daily_pnl < -self._local_policy.max_daily_loss:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.RISK_LIMIT_EXCEEDED,
                    message="Daily loss limit exceeded - KILL SWITCH",
                    field="daily_pnl",
                    value=self._daily_pnl,
                    limit=self._local_policy.max_daily_loss,
                    severity="critical",
                )
            )

        return result

    def record_order(self) -> None:
        """Record an order for rate limiting."""
        now = datetime.utcnow()

        # Reset minute counter if new minute
        if (now - self._minute_start).total_seconds() >= 60:
            self._orders_this_minute = 0
            self._minute_start = now

        # Reset daily counter if new day
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_start > self._day_start:
            self._orders_today = 0
            self._day_start = today_start
            self._daily_pnl = Decimal("0")

        self._orders_this_minute += 1
        self._orders_today += 1

    def get_effective_limits(self) -> Dict[str, Any]:
        """Get effective limits (min of local and cloud)."""
        return self._effective_config

    def _compute_effective_config(self) -> Dict[str, Any]:
        """Compute effective config as minimum of local and cloud."""
        local = self._local_policy
        cloud = self._cloud_config

        if cloud is None:
            return local.to_dict()

        return {
            "max_position_size": str(min(local.max_position_size, cloud.max_position_size)),
            "max_daily_loss": str(min(local.max_daily_loss, cloud.max_daily_loss)),
            "max_order_size": str(min(local.max_order_size, cloud.max_order_size)),
            "max_orders_per_minute": min(local.max_orders_per_minute, cloud.max_orders_per_minute),
            "max_orders_per_day": min(local.max_orders_per_day, cloud.max_orders_per_day),
        }

    def _check_symbol(self, symbol: str, result: PolicyResult) -> None:
        """Check symbol restrictions."""
        if symbol in self._local_policy.prohibited_symbols:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.PROHIBITED_SYMBOL,
                    message=f"Symbol {symbol} is prohibited",
                    field="symbol",
                    value=symbol,
                )
            )

        if self._local_policy.allowed_symbols is not None:
            if symbol not in self._local_policy.allowed_symbols:
                result.add_violation(
                    PolicyViolation(
                        violation_type=ViolationType.PROHIBITED_SYMBOL,
                        message=f"Symbol {symbol} not in allowed list",
                        field="symbol",
                        value=symbol,
                    )
                )

    def _check_action(self, intent: OrderIntent, result: PolicyResult) -> None:
        """Check action restrictions."""
        from packages.shared.contracts.intent import IntentSide

        if not self._local_policy.allow_short:
            if intent.side == IntentSide.SHORT:
                result.add_violation(
                    PolicyViolation(
                        violation_type=ViolationType.PROHIBITED_ACTION,
                        message="Short selling is not allowed",
                        field="side",
                        value=intent.side.value,
                    )
                )

        if not self._local_policy.allow_live_trading:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.PROHIBITED_ACTION,
                    message="Live trading is disabled",
                )
            )

    def _check_quantity(self, quantity: Decimal, result: PolicyResult) -> None:
        """Check quantity limits."""
        effective = self._effective_config
        max_order = Decimal(effective["max_order_size"])

        if quantity > max_order:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.RISK_LIMIT_EXCEEDED,
                    message="Order size exceeds limit",
                    field="quantity",
                    value=quantity,
                    limit=max_order,
                )
            )

    def _check_position_limit(
        self,
        intent: OrderIntent,
        current_position: Decimal,
        account_equity: Decimal,
        result: PolicyResult,
    ) -> None:
        """Check position limits."""
        from packages.shared.contracts.intent import IntentSide

        effective = self._effective_config
        max_position = Decimal(effective["max_position_size"])

        # Calculate new position
        quantity = intent.target_quantity or Decimal("0")
        if intent.side == IntentSide.LONG:
            new_position = current_position + quantity
        elif intent.side == IntentSide.SHORT:
            new_position = current_position - quantity
        else:
            new_position = Decimal("0")

        if abs(new_position) > max_position:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.RISK_LIMIT_EXCEEDED,
                    message="Position would exceed limit",
                    field="position",
                    value=abs(new_position),
                    limit=max_position,
                )
            )

    def _check_rate_limits(self, result: PolicyResult) -> None:
        """Check rate limits."""
        effective = self._effective_config

        if self._orders_this_minute >= effective["max_orders_per_minute"]:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.RATE_LIMIT_EXCEEDED,
                    message="Orders per minute limit exceeded",
                    field="orders_per_minute",
                    value=self._orders_this_minute,
                    limit=effective["max_orders_per_minute"],
                )
            )

        if self._orders_today >= effective["max_orders_per_day"]:
            result.add_violation(
                PolicyViolation(
                    violation_type=ViolationType.RATE_LIMIT_EXCEEDED,
                    message="Orders per day limit exceeded",
                    field="orders_per_day",
                    value=self._orders_today,
                    limit=effective["max_orders_per_day"],
                )
            )
