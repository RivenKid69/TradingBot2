# -*- coding: utf-8 -*-
"""
Tests for Auto-Approve Policy Rules - Phase 4.11

Tests cover:
- Rule creation and matching
- Change class restrictions
- Time window validation
- Limit enforcement
- Audit logging
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from packages.shared.contracts.config import ChangeClass
from packages.agent.approval.auto_approve import (
    AutoApprovePolicyManager,
    AutoApproveRule,
    AutoApproveResult,
    RuleType,
    RuleAction,
    create_operational_auto_approve_rule,
    create_symbol_allowlist_rule,
)


class TestAutoApproveRule:
    """Test AutoApproveRule creation and matching."""

    def test_rule_creation_defaults(self):
        """Test rule creation with defaults."""
        rule = AutoApproveRule(name="test")

        assert rule.name == "test"
        assert rule.enabled is True
        assert rule.action == RuleAction.AUTO_APPROVE
        assert ChangeClass.OPERATIONAL in rule.allowed_change_classes
        assert ChangeClass.INFORMATIONAL in rule.allowed_change_classes
        # TRADING_IMPACTING requires explicit inclusion
        assert ChangeClass.TRADING_IMPACTING not in rule.allowed_change_classes

    def test_matches_command_type_empty(self):
        """Test matching when no command types specified (allow all)."""
        rule = AutoApproveRule()

        assert rule.matches_command_type("ANY_COMMAND") is True

    def test_matches_command_type_specific(self):
        """Test matching specific command types."""
        rule = AutoApproveRule(command_types={"UPDATE_CONFIG", "RESTART"})

        assert rule.matches_command_type("UPDATE_CONFIG") is True
        assert rule.matches_command_type("STOP") is False

    def test_matches_symbol_pattern(self):
        """Test symbol pattern matching."""
        rule = AutoApproveRule(symbol_patterns=["^BTC", "^ETH"])

        assert rule.matches_symbol("BTCUSDT") is True
        assert rule.matches_symbol("ETHUSDT") is True
        assert rule.matches_symbol("XRPUSDT") is False

    def test_matches_strategy_id(self):
        """Test strategy ID matching."""
        rule = AutoApproveRule(strategy_ids={"strategy-1", "strategy-2"})

        assert rule.matches_strategy("strategy-1") is True
        assert rule.matches_strategy("strategy-3") is False

    def test_matches_deployment_id(self):
        """Test deployment ID matching."""
        rule = AutoApproveRule(deployment_ids={"deploy-a"})

        assert rule.matches_deployment("deploy-a") is True
        assert rule.matches_deployment("deploy-b") is False

    def test_matches_change_class(self):
        """Test change class matching."""
        rule = AutoApproveRule(
            allowed_change_classes={ChangeClass.OPERATIONAL}
        )

        assert rule.matches_change_class(ChangeClass.OPERATIONAL) is True
        assert rule.matches_change_class(ChangeClass.TRADING_IMPACTING) is False

    def test_matches_time_window_empty(self):
        """Test time window when not specified (allow all)."""
        rule = AutoApproveRule()

        assert rule.matches_time_window() is True

    def test_to_dict_and_from_dict(self):
        """Test rule serialization and deserialization."""
        rule = AutoApproveRule(
            name="test",
            command_types={"CMD1"},
            symbol_patterns=["^BTC"],
            max_position_value=10000,
        )

        data = rule.to_dict()
        restored = AutoApproveRule.from_dict(data)

        assert restored.name == rule.name
        assert restored.command_types == rule.command_types
        assert restored.symbol_patterns == rule.symbol_patterns
        assert restored.max_position_value == rule.max_position_value


class TestAutoApprovePolicyManager:
    """Test AutoApprovePolicyManager operations."""

    @pytest.fixture
    def temp_rules_path(self):
        """Create temporary rules file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "rules.json"

    @pytest.fixture
    def manager(self, temp_rules_path):
        """Create manager with temp path."""
        return AutoApprovePolicyManager(rules_path=temp_rules_path)

    def test_add_rule(self, manager):
        """Test adding a rule."""
        rule = AutoApproveRule(name="test")

        rule_id = manager.add_rule(rule)

        assert rule_id == rule.rule_id
        assert manager.get_rule(rule_id) is not None

    def test_remove_rule(self, manager):
        """Test removing a rule."""
        rule = AutoApproveRule(name="test")
        rule_id = manager.add_rule(rule)

        success = manager.remove_rule(rule_id)

        assert success is True
        assert manager.get_rule(rule_id) is None

    def test_enable_disable_rule(self, manager):
        """Test enabling and disabling rules."""
        rule = AutoApproveRule(name="test", enabled=True)
        rule_id = manager.add_rule(rule)

        manager.disable_rule(rule_id)
        assert manager.get_rule(rule_id).enabled is False

        manager.enable_rule(rule_id)
        assert manager.get_rule(rule_id).enabled is True

    def test_evaluate_informational_auto_approved(self, manager):
        """Test INFORMATIONAL changes auto-approved by default."""
        result = manager.evaluate(
            command_type="GET_STATUS",
            change_class=ChangeClass.INFORMATIONAL,
        )

        assert result.should_auto_approve is True

    def test_evaluate_trading_impacting_requires_rule(self, manager):
        """Test TRADING_IMPACTING requires explicit rule."""
        result = manager.evaluate(
            command_type="START_RUN",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        assert result.should_auto_approve is False

    def test_evaluate_trading_impacting_with_rule(self, manager):
        """Test TRADING_IMPACTING with matching rule."""
        rule = AutoApproveRule(
            name="Allow trading",
            command_types={"START_RUN"},
            allowed_change_classes={ChangeClass.TRADING_IMPACTING},
        )
        manager.add_rule(rule)

        result = manager.evaluate(
            command_type="START_RUN",
            change_class=ChangeClass.TRADING_IMPACTING,
        )

        assert result.should_auto_approve is True
        assert result.matched_rule is not None

    def test_evaluate_deny_rule_blocks(self, manager):
        """Test AUTO_DENY rule blocks approval."""
        rule = AutoApproveRule(
            name="Block risky commands",
            command_types={"DANGEROUS_CMD"},
            action=RuleAction.AUTO_DENY,
        )
        manager.add_rule(rule)

        result = manager.evaluate(
            command_type="DANGEROUS_CMD",
            change_class=ChangeClass.OPERATIONAL,
        )

        assert result.should_auto_approve is False
        assert "Blocked" in result.reason

    def test_evaluate_with_position_limit(self, manager):
        """Test evaluation with position value limit."""
        rule = AutoApproveRule(
            name="Small positions only",
            allowed_change_classes={ChangeClass.TRADING_IMPACTING},
            max_position_value=10000,
        )
        manager.add_rule(rule)

        # Within limit
        result1 = manager.evaluate(
            command_type="TRADE",
            change_class=ChangeClass.TRADING_IMPACTING,
            details={"position_value": 5000},
        )
        assert result1.should_auto_approve is True

        # Exceeds limit
        result2 = manager.evaluate(
            command_type="TRADE",
            change_class=ChangeClass.TRADING_IMPACTING,
            details={"position_value": 15000},
        )
        assert result2.should_auto_approve is False

    def test_evaluate_with_daily_limit(self, manager):
        """Test daily approval limit enforcement."""
        rule = AutoApproveRule(
            name="Limited approvals",
            allowed_change_classes={ChangeClass.TRADING_IMPACTING},
            max_daily_approvals=2,
        )
        manager.add_rule(rule)

        # First two should pass
        result1 = manager.evaluate("CMD", ChangeClass.TRADING_IMPACTING)
        result2 = manager.evaluate("CMD", ChangeClass.TRADING_IMPACTING)
        assert result1.should_auto_approve is True
        assert result2.should_auto_approve is True

        # Third should fail (exceeds daily limit)
        result3 = manager.evaluate("CMD", ChangeClass.TRADING_IMPACTING)
        assert result3.should_auto_approve is False

    def test_evaluate_disabled_rule_ignored(self, manager):
        """Test disabled rules are ignored."""
        rule = AutoApproveRule(
            name="Disabled rule",
            allowed_change_classes={ChangeClass.TRADING_IMPACTING},
            enabled=False,
        )
        manager.add_rule(rule)

        result = manager.evaluate("CMD", ChangeClass.TRADING_IMPACTING)

        assert result.should_auto_approve is False

    def test_audit_log(self, manager):
        """Test audit log is populated."""
        manager.evaluate("CMD", ChangeClass.OPERATIONAL)
        manager.evaluate("CMD2", ChangeClass.INFORMATIONAL)

        log = manager.get_audit_log()

        assert len(log) >= 2

    def test_get_status(self, manager):
        """Test getting manager status."""
        rule = AutoApproveRule(name="test")
        manager.add_rule(rule)

        status = manager.get_status()

        assert status["total_rules"] == 1
        assert status["enabled_rules"] == 1


class TestConvenienceFunctions:
    """Test convenience functions for rule creation."""

    def test_create_operational_rule(self):
        """Test creating operational auto-approve rule."""
        rule = create_operational_auto_approve_rule(
            name="Ops rule",
            command_types={"UPDATE_CONFIG"},
        )

        assert rule.name == "Ops rule"
        assert "UPDATE_CONFIG" in rule.command_types
        assert ChangeClass.OPERATIONAL in rule.allowed_change_classes
        assert ChangeClass.TRADING_IMPACTING not in rule.allowed_change_classes

    def test_create_symbol_allowlist_rule(self):
        """Test creating symbol allowlist rule."""
        rule = create_symbol_allowlist_rule(
            name="BTC only",
            symbols=["BTC", "ETH"],
            max_position_value=50000,
        )

        assert rule.name == "BTC only"
        assert len(rule.symbol_patterns) == 2
        assert rule.max_position_value == 50000
