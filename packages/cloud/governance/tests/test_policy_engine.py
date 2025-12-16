# -*- coding: utf-8 -*-
"""
Tests for Policy Engine.

Tests for:
- PolicyEngine
- Policy
- PolicyRule
- PolicyCondition
"""

import asyncio
import pytest
from uuid import uuid4

from packages.cloud.governance import (
    PolicyEngine,
    Policy,
    PolicyRule,
    PolicyCondition,
    PolicyType,
    PolicyEffect,
    PolicyStatus,
    PolicyEvaluationContext,
    PolicyEvaluationReport,
    EvaluationResult,
    ComparisonOperator,
    create_default_risk_policy,
    create_default_deployment_policy,
)


# ============================================================================
# PolicyCondition Tests
# ============================================================================


class TestPolicyCondition:
    """Tests for PolicyCondition."""

    def test_greater_than(self):
        """Test greater than comparison."""
        condition = PolicyCondition(
            field="value",
            operator=ComparisonOperator.GREATER_THAN,
            value=100,
        )

        assert condition.evaluate({"value": 150}) is True
        assert condition.evaluate({"value": 50}) is False
        assert condition.evaluate({"value": 100}) is False

    def test_greater_or_equal(self):
        """Test greater or equal comparison."""
        condition = PolicyCondition(
            field="value",
            operator=ComparisonOperator.GREATER_OR_EQUAL,
            value=100,
        )

        assert condition.evaluate({"value": 100}) is True
        assert condition.evaluate({"value": 150}) is True
        assert condition.evaluate({"value": 50}) is False

    def test_less_than(self):
        """Test less than comparison."""
        condition = PolicyCondition(
            field="value",
            operator=ComparisonOperator.LESS_THAN,
            value=100,
        )

        assert condition.evaluate({"value": 50}) is True
        assert condition.evaluate({"value": 150}) is False

    def test_equals(self):
        """Test equals comparison."""
        condition = PolicyCondition(
            field="status",
            operator=ComparisonOperator.EQUALS,
            value="active",
        )

        assert condition.evaluate({"status": "active"}) is True
        assert condition.evaluate({"status": "inactive"}) is False

    def test_not_equals(self):
        """Test not equals comparison."""
        condition = PolicyCondition(
            field="type",
            operator=ComparisonOperator.NOT_EQUALS,
            value="test",
        )

        assert condition.evaluate({"type": "prod"}) is True
        assert condition.evaluate({"type": "test"}) is False

    def test_in_list(self):
        """Test IN list comparison."""
        condition = PolicyCondition(
            field="region",
            operator=ComparisonOperator.IN,
            value=["us-east", "us-west"],
        )

        assert condition.evaluate({"region": "us-east"}) is True
        assert condition.evaluate({"region": "eu-west"}) is False

    def test_not_in_list(self):
        """Test NOT IN list comparison."""
        condition = PolicyCondition(
            field="env",
            operator=ComparisonOperator.NOT_IN,
            value=["dev", "test"],
        )

        assert condition.evaluate({"env": "prod"}) is True
        assert condition.evaluate({"env": "dev"}) is False

    def test_contains(self):
        """Test CONTAINS comparison."""
        condition = PolicyCondition(
            field="tags",
            operator=ComparisonOperator.CONTAINS,
            value="important",
        )

        assert condition.evaluate({"tags": ["important", "urgent"]}) is True
        assert condition.evaluate({"tags": ["normal"]}) is False

    def test_matches_regex(self):
        """Test MATCHES regex comparison."""
        condition = PolicyCondition(
            field="name",
            operator=ComparisonOperator.MATCHES,
            value=r"^strat-\d+$",
        )

        assert condition.evaluate({"name": "strat-123"}) is True
        assert condition.evaluate({"name": "strategy-123"}) is False

    def test_exists(self):
        """Test EXISTS operator."""
        condition = PolicyCondition(
            field="optional_field",
            operator=ComparisonOperator.EXISTS,
            value=None,
        )

        assert condition.evaluate({"optional_field": "value"}) is True
        assert condition.evaluate({"other_field": "value"}) is False

    def test_not_exists(self):
        """Test NOT EXISTS operator."""
        condition = PolicyCondition(
            field="forbidden_field",
            operator=ComparisonOperator.NOT_EXISTS,
            value=None,
        )

        assert condition.evaluate({"other_field": "value"}) is True
        assert condition.evaluate({"forbidden_field": "value"}) is False

    def test_nested_field(self):
        """Test nested field access."""
        condition = PolicyCondition(
            field="trade.quantity",
            operator=ComparisonOperator.GREATER_THAN,
            value=1000,
        )

        assert condition.evaluate({"trade": {"quantity": 1500}}) is True
        assert condition.evaluate({"trade": {"quantity": 500}}) is False

    def test_missing_field(self):
        """Test handling missing field."""
        condition = PolicyCondition(
            field="nonexistent",
            operator=ComparisonOperator.EQUALS,
            value="value",
        )

        assert condition.evaluate({"other": "data"}) is False

    def test_to_dict(self):
        """Test condition serialization."""
        condition = PolicyCondition(
            field="value",
            operator=ComparisonOperator.GREATER_THAN,
            value=100,
            description="Test condition",
        )

        data = condition.to_dict()

        assert data["field"] == "value"
        assert data["operator"] == "greater_than"
        assert data["value"] == 100

    def test_from_dict(self):
        """Test condition deserialization."""
        data = {
            "field": "amount",
            "operator": "less_than",
            "value": 500,
        }

        condition = PolicyCondition.from_dict(data)

        assert condition.field == "amount"
        assert condition.operator == ComparisonOperator.LESS_THAN


# ============================================================================
# PolicyRule Tests
# ============================================================================


class TestPolicyRule:
    """Tests for PolicyRule."""

    def test_rule_with_single_condition(self):
        """Test rule evaluation with single condition."""
        rule = PolicyRule(
            name="High Value Check",
            conditions=[
                PolicyCondition(
                    field="value",
                    operator=ComparisonOperator.GREATER_THAN,
                    value=1000,
                )
            ],
            effect=PolicyEffect.DENY,
        )

        effect = rule.evaluate({"value": 1500})
        assert effect == PolicyEffect.DENY

        effect = rule.evaluate({"value": 500})
        assert effect is None  # No match

    def test_rule_with_multiple_conditions_and_logic(self):
        """Test rule with multiple conditions (AND logic)."""
        rule = PolicyRule(
            name="Complex Check",
            conditions=[
                PolicyCondition(
                    field="value",
                    operator=ComparisonOperator.GREATER_THAN,
                    value=1000,
                ),
                PolicyCondition(
                    field="type",
                    operator=ComparisonOperator.EQUALS,
                    value="market",
                ),
            ],
            effect=PolicyEffect.DENY,
        )

        # Both conditions match
        effect = rule.evaluate({"value": 1500, "type": "market"})
        assert effect == PolicyEffect.DENY

        # Only first condition matches
        effect = rule.evaluate({"value": 1500, "type": "limit"})
        assert effect is None

        # Only second condition matches
        effect = rule.evaluate({"value": 500, "type": "market"})
        assert effect is None

    def test_rule_priority(self):
        """Test rule priority ordering."""
        rule1 = PolicyRule(name="Low Priority", priority=10)
        rule2 = PolicyRule(name="High Priority", priority=100)

        assert rule2.priority > rule1.priority


# ============================================================================
# Policy Tests
# ============================================================================


class TestPolicy:
    """Tests for Policy."""

    def test_policy_evaluation_default_allow(self):
        """Test policy evaluation with default ALLOW."""
        policy = Policy(
            name="Test Policy",
            rules=[
                PolicyRule(
                    name="Block High Value",
                    conditions=[
                        PolicyCondition(
                            field="value",
                            operator=ComparisonOperator.GREATER_THAN,
                            value=10000,
                        )
                    ],
                    effect=PolicyEffect.DENY,
                )
            ],
            default_effect=PolicyEffect.ALLOW,
        )

        # Should deny high value
        effect = policy.evaluate({"value": 15000})
        assert effect == PolicyEffect.DENY

        # Should allow normal value (default)
        effect = policy.evaluate({"value": 5000})
        assert effect == PolicyEffect.ALLOW

    def test_policy_evaluation_default_deny(self):
        """Test policy evaluation with default DENY."""
        policy = Policy(
            name="Strict Policy",
            rules=[
                PolicyRule(
                    name="Allow Known Type",
                    conditions=[
                        PolicyCondition(
                            field="type",
                            operator=ComparisonOperator.IN,
                            value=["limit", "market"],
                        )
                    ],
                    effect=PolicyEffect.ALLOW,
                )
            ],
            default_effect=PolicyEffect.DENY,
        )

        # Should allow known types
        effect = policy.evaluate({"type": "limit"})
        assert effect == PolicyEffect.ALLOW

        # Should deny unknown types
        effect = policy.evaluate({"type": "stop_loss"})
        assert effect == PolicyEffect.DENY

    def test_policy_rule_priority_order(self):
        """Test that rules are evaluated by priority."""
        policy = Policy(
            name="Priority Test",
            rules=[
                PolicyRule(
                    name="Low Priority Allow",
                    conditions=[
                        PolicyCondition(
                            field="status",
                            operator=ComparisonOperator.EQUALS,
                            value="active",
                        )
                    ],
                    effect=PolicyEffect.ALLOW,
                    priority=10,
                ),
                PolicyRule(
                    name="High Priority Deny",
                    conditions=[
                        PolicyCondition(
                            field="status",
                            operator=ComparisonOperator.EQUALS,
                            value="active",
                        )
                    ],
                    effect=PolicyEffect.DENY,
                    priority=100,
                ),
            ],
        )

        # High priority DENY should win
        effect = policy.evaluate({"status": "active"})
        assert effect == PolicyEffect.DENY

    def test_policy_content_hash(self):
        """Test content hash is deterministic."""
        policy = Policy(
            name="Policy1",
            rules=[
                PolicyRule(
                    rule_id="fixed-rule-id",
                    name="Rule1",
                    conditions=[
                        PolicyCondition(
                            field="value",
                            operator=ComparisonOperator.GREATER_THAN,
                            value=100,
                        )
                    ],
                )
            ],
        )

        # Hash should be deterministic for the same policy
        hash1 = policy.get_content_hash()
        hash2 = policy.get_content_hash()
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


# ============================================================================
# PolicyEngine Tests
# ============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    @pytest.fixture
    def engine(self):
        """Create PolicyEngine instance."""
        return PolicyEngine()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_create_policy(self, engine, workspace_id):
        """Test creating a policy."""
        policy = Policy(
            name="Test Policy",
            workspace_id=workspace_id,
            policy_type=PolicyType.RISK_LIMIT,
        )

        created = await engine.create_policy(policy)

        assert created is not None
        assert created.workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_get_policy(self, engine, workspace_id):
        """Test getting policy by ID."""
        policy = Policy(name="Test", workspace_id=workspace_id)
        created = await engine.create_policy(policy)

        found = await engine.get_policy(created.policy_id)

        assert found is not None
        assert found.policy_id == created.policy_id

    @pytest.mark.asyncio
    async def test_update_policy(self, engine, workspace_id):
        """Test updating a policy."""
        policy = Policy(
            name="Original",
            workspace_id=workspace_id,
            status=PolicyStatus.DRAFT,
        )
        created = await engine.create_policy(policy)

        updated = await engine.update_policy(
            created.policy_id,
            {"name": "Updated", "status": PolicyStatus.ACTIVE}
        )

        assert updated.name == "Updated"
        assert updated.status == PolicyStatus.ACTIVE
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_delete_policy(self, engine, workspace_id):
        """Test deleting a policy."""
        policy = Policy(name="ToDelete", workspace_id=workspace_id)
        created = await engine.create_policy(policy)

        deleted = await engine.delete_policy(created.policy_id)

        assert deleted is True
        assert await engine.get_policy(created.policy_id) is None

    @pytest.mark.asyncio
    async def test_list_policies(self, engine, workspace_id):
        """Test listing policies by workspace."""
        await engine.create_policy(Policy(name="P1", workspace_id=workspace_id))
        await engine.create_policy(Policy(name="P2", workspace_id=workspace_id))

        policies = await engine.list_policies(workspace_id=workspace_id)

        assert len(policies) == 2

    @pytest.mark.asyncio
    async def test_evaluate_policy(self, engine, workspace_id):
        """Test policy evaluation."""
        policy = Policy(
            name="Risk Limit",
            workspace_id=workspace_id,
            status=PolicyStatus.ACTIVE,
            rules=[
                PolicyRule(
                    name="Max Value",
                    conditions=[
                        PolicyCondition(
                            field="trade_value",
                            operator=ComparisonOperator.GREATER_THAN,
                            value=100000,
                        )
                    ],
                    effect=PolicyEffect.DENY,
                )
            ],
        )
        await engine.create_policy(policy)

        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="trade",
            action="execute",
            data={"trade_value": 150000},
        )

        report = await engine.evaluate(context)

        assert report.result == EvaluationResult.DENIED

    @pytest.mark.asyncio
    async def test_evaluate_allowed(self, engine, workspace_id):
        """Test policy evaluation when allowed."""
        policy = Policy(
            name="Risk Limit",
            workspace_id=workspace_id,
            status=PolicyStatus.ACTIVE,
            rules=[
                PolicyRule(
                    name="Max Value",
                    conditions=[
                        PolicyCondition(
                            field="trade_value",
                            operator=ComparisonOperator.GREATER_THAN,
                            value=100000,
                        )
                    ],
                    effect=PolicyEffect.DENY,
                )
            ],
            default_effect=PolicyEffect.ALLOW,
        )
        await engine.create_policy(policy)

        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="trade",
            action="execute",
            data={"trade_value": 50000},
        )

        report = await engine.evaluate(context)

        assert report.result == EvaluationResult.ALLOWED

    @pytest.mark.asyncio
    async def test_evaluate_with_warning(self, engine, workspace_id):
        """Test policy evaluation with warning."""
        policy = Policy(
            name="Warning Policy",
            workspace_id=workspace_id,
            status=PolicyStatus.ACTIVE,
            rules=[
                PolicyRule(
                    name="Large Trade Warning",
                    conditions=[
                        PolicyCondition(
                            field="trade_value",
                            operator=ComparisonOperator.GREATER_THAN,
                            value=50000,
                        )
                    ],
                    effect=PolicyEffect.WARN,
                )
            ],
        )
        await engine.create_policy(policy)

        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="trade",
            action="execute",
            data={"trade_value": 75000},
        )

        report = await engine.evaluate(context)

        assert report.result == EvaluationResult.WARNING
        assert len(report.warnings) > 0

    @pytest.mark.asyncio
    async def test_dry_run_evaluation(self, engine, workspace_id):
        """Test dry run evaluation."""
        policy = Policy(
            name="Test",
            workspace_id=workspace_id,
            status=PolicyStatus.ACTIVE,
            rules=[
                PolicyRule(
                    name="Block All",
                    conditions=[
                        PolicyCondition(
                            field="action",
                            operator=ComparisonOperator.EQUALS,
                            value="trade",
                        )
                    ],
                    effect=PolicyEffect.DENY,
                )
            ],
        )
        await engine.create_policy(policy)

        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="trade",
            action="trade",
        )

        report = await engine.evaluate(context, dry_run=True)

        assert report.dry_run is True


# ============================================================================
# Default Policy Tests
# ============================================================================


class TestDefaultPolicies:
    """Tests for default policy creation."""

    def test_create_default_risk_policy(self):
        """Test creating default risk policy."""
        workspace_id = uuid4()
        policy = create_default_risk_policy(workspace_id)

        assert policy.workspace_id == workspace_id
        assert policy.policy_type == PolicyType.RISK_LIMIT
        assert len(policy.rules) > 0

    def test_create_default_deployment_policy(self):
        """Test creating default deployment policy."""
        workspace_id = uuid4()
        policy = create_default_deployment_policy(workspace_id)

        assert policy.workspace_id == workspace_id
        assert policy.policy_type == PolicyType.DEPLOYMENT
        assert len(policy.rules) > 0
