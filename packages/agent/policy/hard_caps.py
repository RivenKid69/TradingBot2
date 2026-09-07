# -*- coding: utf-8 -*-
"""
Hard Caps Enforcer - Absolute limits that cannot be exceeded.

Hard caps are LOCAL limits that Cloud can NEVER override.
They represent the absolute maximum risk the agent will accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Final, List, Optional


@dataclass
class HardCaps:
    """
    Hard caps configuration.

    These are ABSOLUTE limits that cannot be exceeded.
    Cloud cannot raise these limits - they are local only.
    """

    # Position hard caps
    absolute_max_position: Decimal = Decimal("10000000")  # 10M
    absolute_max_position_pct: Decimal = Decimal("1.0")  # 100% of equity

    # Loss hard caps
    absolute_max_daily_loss: Decimal = Decimal("1000000")  # 1M
    absolute_max_daily_loss_pct: Decimal = Decimal("0.5")  # 50% of equity
    absolute_max_drawdown_pct: Decimal = Decimal("0.75")  # 75%

    # Order hard caps
    absolute_max_order_size: Decimal = Decimal("1000000")  # 1M per order
    absolute_max_orders_per_second: int = 10
    absolute_max_orders_per_day: int = 100000

    # Exposure hard caps
    absolute_max_gross_exposure: Decimal = Decimal("100000000")  # 100M
    absolute_max_leverage: Decimal = Decimal("10")  # 10x

    # Kill switch thresholds
    kill_switch_loss_pct: Decimal = Decimal("0.3")  # 30% loss triggers kill
    kill_switch_error_count: int = 10  # 10 errors triggers kill
    kill_switch_latency_ms: int = 5000  # 5 second latency triggers kill

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "absolute_max_position": str(self.absolute_max_position),
            "absolute_max_position_pct": str(self.absolute_max_position_pct),
            "absolute_max_daily_loss": str(self.absolute_max_daily_loss),
            "absolute_max_daily_loss_pct": str(self.absolute_max_daily_loss_pct),
            "absolute_max_drawdown_pct": str(self.absolute_max_drawdown_pct),
            "absolute_max_order_size": str(self.absolute_max_order_size),
            "absolute_max_orders_per_second": self.absolute_max_orders_per_second,
            "absolute_max_orders_per_day": self.absolute_max_orders_per_day,
            "absolute_max_gross_exposure": str(self.absolute_max_gross_exposure),
            "absolute_max_leverage": str(self.absolute_max_leverage),
            "kill_switch_loss_pct": str(self.kill_switch_loss_pct),
            "kill_switch_error_count": self.kill_switch_error_count,
            "kill_switch_latency_ms": self.kill_switch_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardCaps:
        """Create from dictionary."""
        return cls(
            absolute_max_position=Decimal(str(data.get("absolute_max_position", "10000000"))),
            absolute_max_position_pct=Decimal(str(data.get("absolute_max_position_pct", "1.0"))),
            absolute_max_daily_loss=Decimal(str(data.get("absolute_max_daily_loss", "1000000"))),
            absolute_max_daily_loss_pct=Decimal(
                str(data.get("absolute_max_daily_loss_pct", "0.5"))
            ),
            absolute_max_drawdown_pct=Decimal(str(data.get("absolute_max_drawdown_pct", "0.75"))),
            absolute_max_order_size=Decimal(str(data.get("absolute_max_order_size", "1000000"))),
            absolute_max_orders_per_second=int(data.get("absolute_max_orders_per_second", 10)),
            absolute_max_orders_per_day=int(data.get("absolute_max_orders_per_day", 100000)),
            absolute_max_gross_exposure=Decimal(
                str(data.get("absolute_max_gross_exposure", "100000000"))
            ),
            absolute_max_leverage=Decimal(str(data.get("absolute_max_leverage", "10"))),
            kill_switch_loss_pct=Decimal(str(data.get("kill_switch_loss_pct", "0.3"))),
            kill_switch_error_count=int(data.get("kill_switch_error_count", 10)),
            kill_switch_latency_ms=int(data.get("kill_switch_latency_ms", 5000)),
        )


@dataclass
class HardCapViolation:
    """
    Represents a hard cap violation.

    Hard cap violations ALWAYS trigger rejection or kill switch.
    """

    cap_name: str
    message: str
    current_value: Any
    hard_cap: Any
    action: str = "reject"  # reject, kill_switch, halt
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cap_name": self.cap_name,
            "message": self.message,
            "current_value": str(self.current_value),
            "hard_cap": str(self.hard_cap),
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
        }


class HardCapEnforcer:
    """
    Enforces hard caps that cannot be exceeded.

    Hard caps are the FINAL line of defense. If a hard cap is
    violated, the action is REJECTED or the kill switch is triggered.

    Usage:
        enforcer = HardCapEnforcer(hard_caps)

        # Check order
        violation = enforcer.check_order_size(order_size)
        if violation:
            reject_order(violation)

        # Check daily loss
        violation = enforcer.check_daily_loss(daily_pnl, equity)
        if violation:
            trigger_kill_switch(violation)
    """

    def __init__(self, hard_caps: Optional[HardCaps] = None):
        """Initialize enforcer."""
        self._caps = hard_caps or HardCaps()
        self._violations: List[HardCapViolation] = []

        # Tracking state
        self._orders_this_second: int = 0
        self._orders_today: int = 0
        self._errors_today: int = 0
        self._second_start: datetime = datetime.utcnow()
        self._day_start: datetime = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def check_order_size(self, order_size: Decimal) -> Optional[HardCapViolation]:
        """
        Check if order size exceeds hard cap.

        Args:
            order_size: Size of order

        Returns:
            Violation if exceeded, None if OK
        """
        if order_size > self._caps.absolute_max_order_size:
            return HardCapViolation(
                cap_name="absolute_max_order_size",
                message="Order size exceeds absolute maximum",
                current_value=order_size,
                hard_cap=self._caps.absolute_max_order_size,
                action="reject",
            )
        return None

    def check_position_size(
        self,
        position_size: Decimal,
        equity: Optional[Decimal] = None,
    ) -> Optional[HardCapViolation]:
        """
        Check if position size exceeds hard cap.

        Args:
            position_size: Current or proposed position size
            equity: Account equity for percentage check

        Returns:
            Violation if exceeded, None if OK
        """
        if abs(position_size) > self._caps.absolute_max_position:
            return HardCapViolation(
                cap_name="absolute_max_position",
                message="Position exceeds absolute maximum",
                current_value=position_size,
                hard_cap=self._caps.absolute_max_position,
                action="reject",
            )

        if equity and equity > 0:
            position_pct = abs(position_size) / equity
            if position_pct > self._caps.absolute_max_position_pct:
                return HardCapViolation(
                    cap_name="absolute_max_position_pct",
                    message="Position exceeds maximum percentage of equity",
                    current_value=position_pct,
                    hard_cap=self._caps.absolute_max_position_pct,
                    action="reject",
                )

        return None

    def check_daily_loss(
        self,
        daily_pnl: Decimal,
        equity: Decimal,
    ) -> Optional[HardCapViolation]:
        """
        Check if daily loss exceeds hard cap (triggers kill switch).

        Args:
            daily_pnl: Daily P&L (negative for loss)
            equity: Account equity

        Returns:
            Violation if exceeded, None if OK
        """
        if daily_pnl < -self._caps.absolute_max_daily_loss:
            return HardCapViolation(
                cap_name="absolute_max_daily_loss",
                message="Daily loss exceeds absolute maximum - KILL SWITCH",
                current_value=daily_pnl,
                hard_cap=self._caps.absolute_max_daily_loss,
                action="kill_switch",
            )

        if equity > 0:
            loss_pct = -daily_pnl / equity
            if loss_pct > self._caps.absolute_max_daily_loss_pct:
                return HardCapViolation(
                    cap_name="absolute_max_daily_loss_pct",
                    message="Daily loss percentage exceeds maximum - KILL SWITCH",
                    current_value=loss_pct,
                    hard_cap=self._caps.absolute_max_daily_loss_pct,
                    action="kill_switch",
                )

            if loss_pct > self._caps.kill_switch_loss_pct:
                return HardCapViolation(
                    cap_name="kill_switch_loss_pct",
                    message="Loss threshold reached - KILL SWITCH",
                    current_value=loss_pct,
                    hard_cap=self._caps.kill_switch_loss_pct,
                    action="kill_switch",
                )

        return None

    def check_drawdown(
        self,
        current_equity: Decimal,
        peak_equity: Decimal,
    ) -> Optional[HardCapViolation]:
        """
        Check if drawdown exceeds hard cap.

        Args:
            current_equity: Current account equity
            peak_equity: Peak account equity

        Returns:
            Violation if exceeded, None if OK
        """
        if peak_equity <= 0:
            return None

        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown > self._caps.absolute_max_drawdown_pct:
            return HardCapViolation(
                cap_name="absolute_max_drawdown_pct",
                message="Drawdown exceeds absolute maximum - KILL SWITCH",
                current_value=drawdown,
                hard_cap=self._caps.absolute_max_drawdown_pct,
                action="kill_switch",
            )

        return None

    def check_order_rate(self) -> Optional[HardCapViolation]:
        """
        Check if order rate exceeds hard cap.

        Returns:
            Violation if exceeded, None if OK
        """
        now = datetime.utcnow()

        # Reset second counter
        if (now - self._second_start).total_seconds() >= 1:
            self._orders_this_second = 0
            self._second_start = now

        # Reset day counter
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_start > self._day_start:
            self._orders_today = 0
            self._day_start = today_start

        if self._orders_this_second >= self._caps.absolute_max_orders_per_second:
            return HardCapViolation(
                cap_name="absolute_max_orders_per_second",
                message="Order rate exceeds maximum",
                current_value=self._orders_this_second,
                hard_cap=self._caps.absolute_max_orders_per_second,
                action="reject",
            )

        if self._orders_today >= self._caps.absolute_max_orders_per_day:
            return HardCapViolation(
                cap_name="absolute_max_orders_per_day",
                message="Daily order count exceeds maximum",
                current_value=self._orders_today,
                hard_cap=self._caps.absolute_max_orders_per_day,
                action="halt",
            )

        return None

    def check_leverage(
        self,
        gross_exposure: Decimal,
        equity: Decimal,
    ) -> Optional[HardCapViolation]:
        """
        Check if leverage exceeds hard cap.

        Args:
            gross_exposure: Total gross exposure
            equity: Account equity

        Returns:
            Violation if exceeded, None if OK
        """
        if equity <= 0:
            return HardCapViolation(
                cap_name="equity",
                message="Invalid equity value",
                current_value=equity,
                hard_cap=Decimal("0"),
                action="halt",
            )

        leverage = gross_exposure / equity

        if leverage > self._caps.absolute_max_leverage:
            return HardCapViolation(
                cap_name="absolute_max_leverage",
                message="Leverage exceeds maximum",
                current_value=leverage,
                hard_cap=self._caps.absolute_max_leverage,
                action="reject",
            )

        if gross_exposure > self._caps.absolute_max_gross_exposure:
            return HardCapViolation(
                cap_name="absolute_max_gross_exposure",
                message="Gross exposure exceeds maximum",
                current_value=gross_exposure,
                hard_cap=self._caps.absolute_max_gross_exposure,
                action="reject",
            )

        return None

    def check_error_count(self, error_count: int) -> Optional[HardCapViolation]:
        """
        Check if error count exceeds threshold.

        Args:
            error_count: Number of errors today

        Returns:
            Violation if exceeded, None if OK
        """
        if error_count >= self._caps.kill_switch_error_count:
            return HardCapViolation(
                cap_name="kill_switch_error_count",
                message="Error count threshold reached - KILL SWITCH",
                current_value=error_count,
                hard_cap=self._caps.kill_switch_error_count,
                action="kill_switch",
            )
        return None

    def check_latency(self, latency_ms: int) -> Optional[HardCapViolation]:
        """
        Check if latency exceeds threshold.

        Args:
            latency_ms: Current latency in milliseconds

        Returns:
            Violation if exceeded, None if OK
        """
        if latency_ms >= self._caps.kill_switch_latency_ms:
            return HardCapViolation(
                cap_name="kill_switch_latency_ms",
                message="Latency threshold reached - KILL SWITCH",
                current_value=latency_ms,
                hard_cap=self._caps.kill_switch_latency_ms,
                action="kill_switch",
            )
        return None

    def record_order(self) -> None:
        """Record an order for rate limiting."""
        self._orders_this_second += 1
        self._orders_today += 1

    def record_error(self) -> None:
        """Record an error."""
        self._errors_today += 1

    def get_violations(self) -> List[HardCapViolation]:
        """Get list of recent violations."""
        return list(self._violations)

    def get_caps(self) -> HardCaps:
        """Get current hard caps."""
        return self._caps
