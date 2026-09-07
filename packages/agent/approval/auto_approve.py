# -*- coding: utf-8 -*-
"""
Auto-Approve Policy Rules - Local policies for automatic approval.

AGENT ZONE ONLY.

Manages rules that allow automatic approval of certain commands
without manual intervention, while maintaining security.

Design Doc Phase 4.11:
- Local auto-approve policies: allowlist ресурсов/команд
- Cloud cannot set auto-approve rules - LOCAL ONLY
- All auto-approved actions are logged for audit
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Pattern, Set, Tuple
from uuid import uuid4

from packages.shared.contracts.config import ChangeClass


# Rule types
class RuleType(Enum):
    """Types of auto-approve rules."""
    COMMAND_TYPE = auto()  # Match by command type
    SYMBOL_PATTERN = auto()  # Match by trading symbol
    STRATEGY_ID = auto()  # Match by strategy ID
    DEPLOYMENT_ID = auto()  # Match by deployment ID
    CONFIG_PATH = auto()  # Match by config path pattern
    TIME_WINDOW = auto()  # Match by time window
    COMPOSITE = auto()  # Combination of rules (AND)


class RuleAction(Enum):
    """Action to take when rule matches."""
    AUTO_APPROVE = "auto_approve"  # Automatically approve
    AUTO_DENY = "auto_deny"  # Automatically deny (blocklist)
    REQUIRE_APPROVAL = "require_approval"  # Force manual approval


@dataclass
class AutoApproveRule:
    """
    Single auto-approve rule.
    """
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    rule_type: RuleType = RuleType.COMMAND_TYPE
    action: RuleAction = RuleAction.AUTO_APPROVE
    enabled: bool = True

    # Matching criteria
    command_types: Set[str] = field(default_factory=set)  # Command types to match
    symbol_patterns: List[str] = field(default_factory=list)  # Regex patterns for symbols
    strategy_ids: Set[str] = field(default_factory=set)  # Strategy IDs
    deployment_ids: Set[str] = field(default_factory=set)  # Deployment IDs
    config_path_patterns: List[str] = field(default_factory=list)  # Config path patterns

    # Time-based criteria
    time_window_start: Optional[str] = None  # HH:MM format
    time_window_end: Optional[str] = None  # HH:MM format
    weekdays_only: bool = False  # Only Mon-Fri

    # Limits
    max_position_value: Optional[float] = None  # Max position for auto-approve
    max_order_value: Optional[float] = None  # Max order value
    max_daily_approvals: Optional[int] = None  # Daily limit

    # Change class restrictions
    allowed_change_classes: Set[ChangeClass] = field(
        default_factory=lambda: {ChangeClass.OPERATIONAL, ChangeClass.INFORMATIONAL}
    )

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "local_user"
    last_matched_at: Optional[datetime] = None
    match_count: int = 0

    # Compiled patterns (not serialized)
    _compiled_patterns: List[Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compile regex patterns."""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns = []
        for pattern in self.symbol_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass  # Invalid pattern, skip

    def matches_command_type(self, command_type: str) -> bool:
        """Check if rule matches command type."""
        if not self.command_types:
            return True  # No restriction
        return command_type in self.command_types

    def matches_symbol(self, symbol: str) -> bool:
        """Check if rule matches symbol."""
        if not self._compiled_patterns:
            return True  # No restriction

        for pattern in self._compiled_patterns:
            if pattern.search(symbol):
                return True
        return False

    def matches_strategy(self, strategy_id: str) -> bool:
        """Check if rule matches strategy."""
        if not self.strategy_ids:
            return True  # No restriction
        return strategy_id in self.strategy_ids

    def matches_deployment(self, deployment_id: str) -> bool:
        """Check if rule matches deployment."""
        if not self.deployment_ids:
            return True  # No restriction
        return deployment_id in self.deployment_ids

    def matches_time_window(self) -> bool:
        """Check if current time is within allowed window."""
        if not self.time_window_start or not self.time_window_end:
            return True  # No restriction

        now = datetime.utcnow()

        # Check weekday
        if self.weekdays_only and now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Parse time window
        try:
            start_hour, start_min = map(int, self.time_window_start.split(":"))
            end_hour, end_min = map(int, self.time_window_end.split(":"))

            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            current_minutes = now.hour * 60 + now.minute

            # Handle overnight windows
            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes <= end_minutes
            else:
                return current_minutes >= start_minutes or current_minutes <= end_minutes
        except (ValueError, AttributeError):
            return True  # Invalid format, allow

    def matches_change_class(self, change_class: ChangeClass) -> bool:
        """Check if rule allows this change class."""
        return change_class in self.allowed_change_classes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.name,
            "action": self.action.value,
            "enabled": self.enabled,
            "command_types": list(self.command_types),
            "symbol_patterns": self.symbol_patterns,
            "strategy_ids": list(self.strategy_ids),
            "deployment_ids": list(self.deployment_ids),
            "config_path_patterns": self.config_path_patterns,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "weekdays_only": self.weekdays_only,
            "max_position_value": self.max_position_value,
            "max_order_value": self.max_order_value,
            "max_daily_approvals": self.max_daily_approvals,
            "allowed_change_classes": [c.value for c in self.allowed_change_classes],
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "match_count": self.match_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoApproveRule":
        """Create from dictionary."""
        allowed_classes = {
            ChangeClass(c) for c in data.get("allowed_change_classes", ["operational", "informational"])
        }

        rule = cls(
            rule_id=data.get("rule_id", str(uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            rule_type=RuleType[data.get("rule_type", "COMMAND_TYPE")],
            action=RuleAction(data.get("action", "auto_approve")),
            enabled=data.get("enabled", True),
            command_types=set(data.get("command_types", [])),
            symbol_patterns=data.get("symbol_patterns", []),
            strategy_ids=set(data.get("strategy_ids", [])),
            deployment_ids=set(data.get("deployment_ids", [])),
            config_path_patterns=data.get("config_path_patterns", []),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            weekdays_only=data.get("weekdays_only", False),
            max_position_value=data.get("max_position_value"),
            max_order_value=data.get("max_order_value"),
            max_daily_approvals=data.get("max_daily_approvals"),
            allowed_change_classes=allowed_classes,
            created_by=data.get("created_by", "local_user"),
            match_count=data.get("match_count", 0),
        )

        if data.get("created_at"):
            rule.created_at = datetime.fromisoformat(data["created_at"])

        return rule


@dataclass
class AutoApproveResult:
    """
    Result of auto-approve evaluation.
    """
    should_auto_approve: bool = False
    matched_rule: Optional[AutoApproveRule] = None
    reason: str = ""
    evidence_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_auto_approve": self.should_auto_approve,
            "matched_rule_id": self.matched_rule.rule_id if self.matched_rule else None,
            "matched_rule_name": self.matched_rule.name if self.matched_rule else None,
            "reason": self.reason,
            "evidence_hash": self.evidence_hash,
        }


class AutoApprovePolicyManager:
    """
    Manages local auto-approve rules.

    SECURITY CRITICAL:
    - Rules can ONLY be set locally
    - Cloud cannot add/modify auto-approve rules
    - All auto-approvals are logged with evidence
    - TRADING_IMPACTING changes require explicit local rule

    Usage:
        manager = AutoApprovePolicyManager()

        # Add rule for OPERATIONAL changes
        manager.add_rule(AutoApproveRule(
            name="Operational Changes",
            command_types={"UPDATE_CONFIG"},
            allowed_change_classes={ChangeClass.OPERATIONAL},
        ))

        # Add symbol-specific rule
        manager.add_rule(AutoApproveRule(
            name="BTC Trading",
            symbol_patterns=["^BTC"],
            max_position_value=10000,
        ))

        # Evaluate request
        result = manager.evaluate(command_type, change_class, details)
        if result.should_auto_approve:
            # Auto-approve with evidence
            approve(evidence_hash=result.evidence_hash)
    """

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        on_auto_approve: Optional[Callable[[AutoApproveResult], None]] = None,
    ):
        """
        Initialize manager.

        Args:
            rules_path: Path to persist rules
            on_auto_approve: Callback when auto-approval occurs
        """
        self._rules_path = rules_path or Path.home() / ".ccea" / "auto_approve_rules.json"
        self._on_auto_approve = on_auto_approve

        # Rules storage
        self._rules: Dict[str, AutoApproveRule] = {}

        # Daily counters for limits
        self._daily_counts: Dict[str, int] = {}
        self._count_date: Optional[datetime] = None

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Load persisted rules
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from disk."""
        if not self._rules_path.exists():
            return

        try:
            with open(self._rules_path) as f:
                data = json.load(f)

            for rule_data in data.get("rules", []):
                rule = AutoApproveRule.from_dict(rule_data)
                self._rules[rule.rule_id] = rule

        except Exception:
            pass  # Invalid file, start fresh

    def _save_rules(self) -> None:
        """Save rules to disk."""
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "rules": [r.to_dict() for r in self._rules.values()],
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(self._rules_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_rule(self, rule: AutoApproveRule) -> str:
        """
        Add auto-approve rule.

        Args:
            rule: Rule to add

        Returns:
            Rule ID
        """
        self._rules[rule.rule_id] = rule
        self._save_rules()
        return rule.rule_id

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing rule.

        Args:
            rule_id: Rule ID
            updates: Fields to update

        Returns:
            True if updated
        """
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]

        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        # Recompile patterns if changed
        if "symbol_patterns" in updates:
            rule._compile_patterns()

        self._save_rules()
        return True

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if removed
        """
        if rule_id not in self._rules:
            return False

        del self._rules[rule_id]
        self._save_rules()
        return True

    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        return self.update_rule(rule_id, {"enabled": True})

    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        return self.update_rule(rule_id, {"enabled": False})

    def evaluate(
        self,
        command_type: str,
        change_class: ChangeClass,
        details: Optional[Dict[str, Any]] = None,
    ) -> AutoApproveResult:
        """
        Evaluate if request should be auto-approved.

        Args:
            command_type: Type of command
            change_class: Change classification
            details: Additional details (symbol, strategy_id, etc.)

        Returns:
            AutoApproveResult
        """
        details = details or {}

        # Reset daily counts if new day
        self._check_daily_reset()

        # TRADING_IMPACTING requires explicit rule
        if change_class == ChangeClass.TRADING_IMPACTING:
            result = self._evaluate_trading_impacting(command_type, details)
        else:
            result = self._evaluate_other(command_type, change_class, details)

        # Log evaluation
        self._log_evaluation(command_type, change_class, details, result)

        # Callback
        if result.should_auto_approve and self._on_auto_approve:
            self._on_auto_approve(result)

        return result

    def _evaluate_trading_impacting(
        self,
        command_type: str,
        details: Dict[str, Any],
    ) -> AutoApproveResult:
        """Evaluate TRADING_IMPACTING changes (strictest)."""
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.action != RuleAction.AUTO_APPROVE:
                continue

            # Must explicitly allow TRADING_IMPACTING
            if ChangeClass.TRADING_IMPACTING not in rule.allowed_change_classes:
                continue

            if self._rule_matches(rule, command_type, ChangeClass.TRADING_IMPACTING, details):
                # Check limits
                if not self._check_rule_limits(rule, details):
                    continue

                # Increment daily count
                self._increment_daily_count(rule.rule_id)

                # Update rule stats
                rule.last_matched_at = datetime.utcnow()
                rule.match_count += 1

                return AutoApproveResult(
                    should_auto_approve=True,
                    matched_rule=rule,
                    reason=f"Matched rule: {rule.name}",
                    evidence_hash=self._compute_evidence_hash(rule, command_type, details),
                )

        return AutoApproveResult(
            should_auto_approve=False,
            reason="No matching auto-approve rule for TRADING_IMPACTING change",
        )

    def _evaluate_other(
        self,
        command_type: str,
        change_class: ChangeClass,
        details: Dict[str, Any],
    ) -> AutoApproveResult:
        """Evaluate OPERATIONAL/INFORMATIONAL changes."""
        # Check for deny rules first
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.action == RuleAction.AUTO_DENY:
                if self._rule_matches(rule, command_type, change_class, details):
                    return AutoApproveResult(
                        should_auto_approve=False,
                        matched_rule=rule,
                        reason=f"Blocked by rule: {rule.name}",
                    )

        # Check for approve rules
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if rule.action != RuleAction.AUTO_APPROVE:
                continue

            if self._rule_matches(rule, command_type, change_class, details):
                if not self._check_rule_limits(rule, details):
                    continue

                self._increment_daily_count(rule.rule_id)

                rule.last_matched_at = datetime.utcnow()
                rule.match_count += 1

                return AutoApproveResult(
                    should_auto_approve=True,
                    matched_rule=rule,
                    reason=f"Matched rule: {rule.name}",
                    evidence_hash=self._compute_evidence_hash(rule, command_type, details),
                )

        # Default behavior for non-TRADING_IMPACTING
        if change_class == ChangeClass.INFORMATIONAL:
            return AutoApproveResult(
                should_auto_approve=True,
                reason="INFORMATIONAL changes auto-approved by default",
                evidence_hash=self._compute_evidence_hash(None, command_type, details),
            )

        return AutoApproveResult(
            should_auto_approve=False,
            reason="No matching auto-approve rule",
        )

    def _rule_matches(
        self,
        rule: AutoApproveRule,
        command_type: str,
        change_class: ChangeClass,
        details: Dict[str, Any],
    ) -> bool:
        """Check if rule matches the request."""
        # Change class check
        if not rule.matches_change_class(change_class):
            return False

        # Command type check
        if not rule.matches_command_type(command_type):
            return False

        # Time window check
        if not rule.matches_time_window():
            return False

        # Symbol check
        symbol = details.get("symbol", "")
        if symbol and not rule.matches_symbol(symbol):
            return False

        # Strategy check
        strategy_id = details.get("strategy_id", "")
        if strategy_id and not rule.matches_strategy(strategy_id):
            return False

        # Deployment check
        deployment_id = details.get("deployment_id", "")
        if deployment_id and not rule.matches_deployment(deployment_id):
            return False

        return True

    def _check_rule_limits(
        self,
        rule: AutoApproveRule,
        details: Dict[str, Any],
    ) -> bool:
        """Check if rule limits are not exceeded."""
        # Position limit
        if rule.max_position_value is not None:
            position_value = details.get("position_value", 0)
            if position_value > rule.max_position_value:
                return False

        # Order limit
        if rule.max_order_value is not None:
            order_value = details.get("order_value", 0)
            if order_value > rule.max_order_value:
                return False

        # Daily approval limit
        if rule.max_daily_approvals is not None:
            daily_count = self._daily_counts.get(rule.rule_id, 0)
            if daily_count >= rule.max_daily_approvals:
                return False

        return True

    def _check_daily_reset(self) -> None:
        """Reset daily counts if new day."""
        today = datetime.utcnow().date()
        if self._count_date != today:
            self._daily_counts.clear()
            self._count_date = today

    def _increment_daily_count(self, rule_id: str) -> None:
        """Increment daily count for rule."""
        self._daily_counts[rule_id] = self._daily_counts.get(rule_id, 0) + 1

    def _compute_evidence_hash(
        self,
        rule: Optional[AutoApproveRule],
        command_type: str,
        details: Dict[str, Any],
    ) -> str:
        """Compute evidence hash for audit trail."""
        evidence = {
            "rule_id": rule.rule_id if rule else None,
            "rule_name": rule.name if rule else None,
            "command_type": command_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }

        canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _log_evaluation(
        self,
        command_type: str,
        change_class: ChangeClass,
        details: Dict[str, Any],
        result: AutoApproveResult,
    ) -> None:
        """Log evaluation for audit."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "command_type": command_type,
            "change_class": change_class.value,
            "auto_approved": result.should_auto_approve,
            "matched_rule": result.matched_rule.name if result.matched_rule else None,
            "reason": result.reason,
            "evidence_hash": result.evidence_hash,
        }

        self._audit_log.append(entry)

        # Trim log
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def get_rules(self) -> List[AutoApproveRule]:
        """Get all rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[AutoApproveRule]:
        """Get rule by ID."""
        return self._rules.get(rule_id)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "daily_auto_approvals": sum(self._daily_counts.values()),
            "audit_log_size": len(self._audit_log),
            "rules_path": str(self._rules_path),
        }


# Convenience function to create common rules
def create_operational_auto_approve_rule(
    name: str = "Auto-approve OPERATIONAL changes",
    command_types: Optional[Set[str]] = None,
) -> AutoApproveRule:
    """Create rule for OPERATIONAL changes."""
    return AutoApproveRule(
        name=name,
        description="Auto-approve operational (non-trading) changes",
        rule_type=RuleType.COMMAND_TYPE,
        action=RuleAction.AUTO_APPROVE,
        command_types=command_types or set(),
        allowed_change_classes={ChangeClass.OPERATIONAL, ChangeClass.INFORMATIONAL},
    )


def create_symbol_allowlist_rule(
    name: str,
    symbols: List[str],
    max_position_value: Optional[float] = None,
) -> AutoApproveRule:
    """Create rule for specific symbols."""
    patterns = [f"^{re.escape(s)}$" for s in symbols]

    return AutoApproveRule(
        name=name,
        description=f"Auto-approve for symbols: {', '.join(symbols)}",
        rule_type=RuleType.SYMBOL_PATTERN,
        action=RuleAction.AUTO_APPROVE,
        symbol_patterns=patterns,
        max_position_value=max_position_value,
        allowed_change_classes={ChangeClass.OPERATIONAL},
    )
