# -*- coding: utf-8 -*-
"""
Order Router - Routes orders to appropriate broker.

AGENT ZONE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Callable

from .engine import Order


class RoutingCriteria(str, Enum):
    """Criteria for routing decisions."""

    SYMBOL = "symbol"
    ASSET_CLASS = "asset_class"
    ORDER_SIZE = "order_size"
    TIME_OF_DAY = "time_of_day"


@dataclass
class RoutingRule:
    """
    Rule for order routing.
    """

    name: str
    broker: str
    priority: int = 0

    # Matching criteria
    symbols: Optional[List[str]] = None  # Match specific symbols
    symbol_prefix: Optional[str] = None  # Match symbol prefix
    asset_class: Optional[str] = None  # crypto, equity, forex
    min_order_size: Optional[Decimal] = None
    max_order_size: Optional[Decimal] = None

    # Rule state
    enabled: bool = True

    def matches(self, order: Order) -> bool:
        """Check if order matches this rule."""
        if not self.enabled:
            return False

        # Symbol matching
        if self.symbols and order.symbol not in self.symbols:
            return False

        if self.symbol_prefix and not order.symbol.startswith(self.symbol_prefix):
            return False

        # Size matching
        if self.min_order_size and order.quantity < self.min_order_size:
            return False

        if self.max_order_size and order.quantity > self.max_order_size:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "broker": self.broker,
            "priority": self.priority,
            "symbols": self.symbols,
            "symbol_prefix": self.symbol_prefix,
            "asset_class": self.asset_class,
            "min_order_size": str(self.min_order_size) if self.min_order_size else None,
            "max_order_size": str(self.max_order_size) if self.max_order_size else None,
            "enabled": self.enabled,
        }


@dataclass
class RoutingResult:
    """
    Result of routing decision.
    """

    success: bool
    broker: Optional[str] = None
    rule_name: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "broker": self.broker,
            "rule_name": self.rule_name,
            "error_message": self.error_message,
        }


class OrderRouter:
    """
    Routes orders to appropriate broker.

    Uses configurable rules to determine which broker should
    receive each order.

    Usage:
        router = OrderRouter(default_broker="binance")
        router.add_rule(RoutingRule(
            name="crypto_orders",
            broker="binance",
            symbol_prefix="BTC",
        ))

        result = router.route(order)
        if result.success:
            submit_to_broker(result.broker, order)
    """

    def __init__(
        self,
        default_broker: str = "default",
        rules: Optional[List[RoutingRule]] = None,
    ):
        """
        Initialize router.

        Args:
            default_broker: Default broker when no rules match
            rules: Initial routing rules
        """
        self._default_broker = default_broker
        self._rules: List[RoutingRule] = rules or []

        # Sort rules by priority (higher first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        """Remove routing rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def route(self, order: Order) -> RoutingResult:
        """
        Route order to appropriate broker.

        Args:
            order: Order to route

        Returns:
            RoutingResult with broker selection
        """
        # Check rules in priority order
        for rule in self._rules:
            if rule.matches(order):
                return RoutingResult(
                    success=True,
                    broker=rule.broker,
                    rule_name=rule.name,
                )

        # Use default broker
        return RoutingResult(
            success=True,
            broker=self._default_broker,
            rule_name="default",
        )

    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules."""
        return list(self._rules)

    def get_brokers(self) -> List[str]:
        """Get list of all brokers in rules."""
        brokers = {self._default_broker}
        for rule in self._rules:
            brokers.add(rule.broker)
        return sorted(brokers)
