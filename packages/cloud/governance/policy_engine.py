# -*- coding: utf-8 -*-
"""
Policy Engine - CLOUD ZONE ONLY.

Centralized policy management and enforcement for governance.

Design Doc Reference:
    - Section 4.1: "Governance Portal"
    - Section 5.3.7: "Risk limits enforcement"
    - Section 10.3: "Config Schema (JSON) с versioning"

Features:
    - Policy definition and versioning
    - Rule evaluation engine
    - Policy enforcement points
    - Audit trail
    - Policy simulation (dry-run)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set, Union
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MAX_POLICIES_PER_WORKSPACE: Final[int] = 100
MAX_RULES_PER_POLICY: Final[int] = 50
POLICY_VERSION_LIMIT: Final[int] = 100  # Max versions to retain


# ============================================================================
# Enums
# ============================================================================


class PolicyType(str, Enum):
    """Types of policies."""

    RISK_LIMIT = "risk_limit"
    POSITION_LIMIT = "position_limit"
    TRADE_LIMIT = "trade_limit"
    ACCESS_CONTROL = "access_control"
    DATA_GOVERNANCE = "data_governance"
    COMPLIANCE = "compliance"
    DEPLOYMENT = "deployment"
    RESOURCE = "resource"
    CUSTOM = "custom"


class PolicyEffect(str, Enum):
    """Policy effect when matched."""

    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"  # Log but don't block
    WARN = "warn"  # Allow but warn


class PolicyStatus(str, Enum):
    """Policy lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    DISABLED = "disabled"
    ARCHIVED = "archived"


class ComparisonOperator(str, Enum):
    """Comparison operators for conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_THAN = "less_than"
    LESS_OR_EQUAL = "less_or_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # Regex
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class EvaluationResult(str, Enum):
    """Result of policy evaluation."""

    ALLOWED = "allowed"
    DENIED = "denied"
    AUDIT_ONLY = "audit_only"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PolicyCondition:
    """
    A single condition in a policy rule.

    Example: position_value > 1000000
    """

    field: str
    operator: ComparisonOperator
    value: Any
    description: Optional[str] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Args:
            context: Dictionary of values to evaluate

        Returns:
            True if condition matches
        """
        # Navigate nested fields (e.g., "trade.quantity")
        actual_value = self._get_field_value(context, self.field)

        # Handle NOT_EXISTS
        if self.operator == ComparisonOperator.NOT_EXISTS:
            return actual_value is None

        # Handle EXISTS
        if self.operator == ComparisonOperator.EXISTS:
            return actual_value is not None

        # If field doesn't exist and operator requires value
        if actual_value is None:
            return False

        # Evaluate based on operator
        return self._compare(actual_value, self.value)

    def _get_field_value(self, context: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested field path."""
        parts = field_path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def _compare(self, actual: Any, expected: Any) -> bool:
        """Compare values based on operator."""
        op = self.operator

        if op == ComparisonOperator.EQUALS:
            return actual == expected
        elif op == ComparisonOperator.NOT_EQUALS:
            return actual != expected
        elif op == ComparisonOperator.GREATER_THAN:
            return actual > expected
        elif op == ComparisonOperator.GREATER_OR_EQUAL:
            return actual >= expected
        elif op == ComparisonOperator.LESS_THAN:
            return actual < expected
        elif op == ComparisonOperator.LESS_OR_EQUAL:
            return actual <= expected
        elif op == ComparisonOperator.IN:
            return actual in expected
        elif op == ComparisonOperator.NOT_IN:
            return actual not in expected
        elif op == ComparisonOperator.CONTAINS:
            return expected in actual if isinstance(actual, (str, list)) else False
        elif op == ComparisonOperator.NOT_CONTAINS:
            return expected not in actual if isinstance(actual, (str, list)) else True
        elif op == ComparisonOperator.MATCHES:
            return bool(re.match(expected, str(actual))) if actual else False

        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCondition":
        return cls(
            field=data["field"],
            operator=ComparisonOperator(data["operator"]),
            value=data["value"],
            description=data.get("description"),
        )


@dataclass
class PolicyRule:
    """
    A rule within a policy.

    Rules consist of conditions and an effect.
    All conditions must match (AND logic).
    """

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    conditions: List[PolicyCondition] = field(default_factory=list)
    effect: PolicyEffect = PolicyEffect.DENY
    priority: int = 0  # Higher = evaluated first
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Dict[str, Any]) -> Optional[PolicyEffect]:
        """
        Evaluate rule against context.

        Args:
            context: Dictionary of values to evaluate

        Returns:
            PolicyEffect if all conditions match, None otherwise
        """
        # All conditions must match (AND logic)
        for condition in self.conditions:
            if not condition.evaluate(context):
                return None

        return self.effect

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "effect": self.effect.value,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRule":
        return cls(
            rule_id=data.get("rule_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            conditions=[PolicyCondition.from_dict(c) for c in data.get("conditions", [])],
            effect=PolicyEffect(data.get("effect", "deny")),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Policy:
    """
    A governance policy.

    Contains rules, targeting criteria, and lifecycle metadata.
    """

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: Optional[UUID] = None
    name: str = ""
    description: str = ""
    policy_type: PolicyType = PolicyType.CUSTOM
    version: int = 1
    status: PolicyStatus = PolicyStatus.DRAFT

    # Rules
    rules: List[PolicyRule] = field(default_factory=list)

    # Default effect when no rules match
    default_effect: PolicyEffect = PolicyEffect.ALLOW

    # Targeting (which resources/actions this policy applies to)
    targets: Dict[str, Any] = field(default_factory=dict)
    # Example: {"resource_type": "trade", "strategy_ids": ["strat-123"]}

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    def evaluate(self, context: Dict[str, Any]) -> PolicyEffect:
        """
        Evaluate policy against context.

        Args:
            context: Dictionary of values to evaluate

        Returns:
            PolicyEffect based on rule evaluation
        """
        # Sort rules by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            effect = rule.evaluate(context)
            if effect is not None:
                return effect

        return self.default_effect

    def get_content_hash(self) -> str:
        """Get hash of policy content for versioning."""
        content = json.dumps(
            {
                "rules": [r.to_dict() for r in self.rules],
                "default_effect": self.default_effect.value,
                "targets": self.targets,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "version": self.version,
            "status": self.status.value,
            "rules": [r.to_dict() for r in self.rules],
            "default_effect": self.default_effect.value,
            "targets": self.targets,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content_hash": self.get_content_hash(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        workspace_id = None
        if data.get("workspace_id"):
            workspace_id = (
                UUID(data["workspace_id"])
                if isinstance(data["workspace_id"], str)
                else data["workspace_id"]
            )

        return cls(
            policy_id=data.get("policy_id", str(uuid.uuid4())),
            workspace_id=workspace_id,
            name=data.get("name", ""),
            description=data.get("description", ""),
            policy_type=PolicyType(data.get("policy_type", "custom")),
            version=data.get("version", 1),
            status=PolicyStatus(data.get("status", "draft")),
            rules=[PolicyRule.from_dict(r) for r in data.get("rules", [])],
            default_effect=PolicyEffect(data.get("default_effect", "allow")),
            targets=data.get("targets", {}),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
        )


@dataclass
class PolicyEvaluationContext:
    """Context for policy evaluation."""

    workspace_id: UUID
    resource_type: str
    action: str
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    strategy_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": str(self.workspace_id),
            "resource_type": self.resource_type,
            "action": self.action,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "strategy_id": self.strategy_id,
            **self.data,
        }


@dataclass
class PolicyEvaluationReport:
    """Report of policy evaluation."""

    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: PolicyEvaluationContext = field(
        default_factory=lambda: PolicyEvaluationContext(UUID(int=0), "", "")
    )
    result: EvaluationResult = EvaluationResult.NOT_APPLICABLE
    matched_policies: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    deny_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    audit_entries: List[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "context": self.context.to_dict(),
            "result": self.result.value,
            "matched_policies": self.matched_policies,
            "matched_rules": self.matched_rules,
            "deny_reason": self.deny_reason,
            "warnings": self.warnings,
            "audit_entries": self.audit_entries,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "dry_run": self.dry_run,
        }


# ============================================================================
# Policy Engine
# ============================================================================


class PolicyEngine:
    """
    Central policy evaluation engine.

    Manages policies and provides enforcement points.

    SECURITY: Respects workspace isolation.
    """

    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._workspace_policies: Dict[UUID, Set[str]] = {}
        self._global_policies: Set[str] = set()
        self._policy_versions: Dict[str, List[Policy]] = {}
        self._lock = asyncio.Lock()
        self._evaluation_hooks: List[Callable] = []

        logger.info("PolicyEngine initialized")

    # ========================================================================
    # Policy Management
    # ========================================================================

    async def create_policy(self, policy: Policy, user_id: Optional[str] = None) -> Policy:
        """
        Create a new policy.

        Args:
            policy: Policy to create
            user_id: User creating the policy

        Returns:
            Created policy with assigned ID
        """
        async with self._lock:
            # Check workspace limits
            if policy.workspace_id:
                workspace_policies = self._workspace_policies.get(policy.workspace_id, set())
                if len(workspace_policies) >= MAX_POLICIES_PER_WORKSPACE:
                    raise ValueError(
                        f"Maximum policies per workspace ({MAX_POLICIES_PER_WORKSPACE}) exceeded"
                    )

            # Validate
            self._validate_policy(policy)

            # Set metadata
            policy.created_by = user_id
            policy.updated_by = user_id
            policy.version = 1

            # Store
            self._policies[policy.policy_id] = policy

            if policy.workspace_id:
                if policy.workspace_id not in self._workspace_policies:
                    self._workspace_policies[policy.workspace_id] = set()
                self._workspace_policies[policy.workspace_id].add(policy.policy_id)
            else:
                self._global_policies.add(policy.policy_id)

            # Store version
            self._policy_versions[policy.policy_id] = [policy]

        logger.info(f"Created policy: {policy.name} ({policy.policy_id})")
        return policy

    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Optional[Policy]:
        """
        Update an existing policy (creates new version).

        Args:
            policy_id: Policy ID to update
            updates: Dictionary of updates
            user_id: User making the update

        Returns:
            Updated policy or None if not found
        """
        async with self._lock:
            policy = self._policies.get(policy_id)
            if not policy:
                return None

            # Create copy for versioning
            import copy

            old_version = copy.deepcopy(policy)

            # Apply updates
            for key, value in updates.items():
                if key == "rules":
                    policy.rules = [PolicyRule.from_dict(r) for r in value]
                elif key == "targets":
                    policy.targets = value
                elif key == "default_effect":
                    policy.default_effect = PolicyEffect(value)
                elif key == "status":
                    policy.status = PolicyStatus(value)
                elif hasattr(policy, key):
                    setattr(policy, key, value)

            # Update metadata
            policy.version += 1
            policy.updated_at = datetime.now(timezone.utc)
            policy.updated_by = user_id

            # Validate
            self._validate_policy(policy)

            # Store version history
            versions = self._policy_versions.get(policy_id, [])
            versions.append(old_version)
            if len(versions) > POLICY_VERSION_LIMIT:
                versions = versions[-POLICY_VERSION_LIMIT:]
            self._policy_versions[policy_id] = versions

        logger.info(f"Updated policy: {policy.name} v{policy.version}")
        return policy

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        async with self._lock:
            policy = self._policies.pop(policy_id, None)
            if not policy:
                return False

            if policy.workspace_id and policy.workspace_id in self._workspace_policies:
                self._workspace_policies[policy.workspace_id].discard(policy_id)

            self._global_policies.discard(policy_id)
            self._policy_versions.pop(policy_id, None)

        logger.info(f"Deleted policy: {policy_id}")
        return True

    async def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID."""
        return self._policies.get(policy_id)

    async def get_policy_version(self, policy_id: str, version: int) -> Optional[Policy]:
        """Get specific version of a policy."""
        versions = self._policy_versions.get(policy_id, [])
        for policy in versions:
            if policy.version == version:
                return policy
        return None

    async def list_policies(
        self,
        workspace_id: Optional[UUID] = None,
        policy_type: Optional[PolicyType] = None,
        status: Optional[PolicyStatus] = None,
        include_global: bool = True,
    ) -> List[Policy]:
        """
        List policies with filtering.

        Args:
            workspace_id: Filter by workspace
            policy_type: Filter by type
            status: Filter by status
            include_global: Include global policies

        Returns:
            List of matching policies
        """
        policies = []

        for policy in self._policies.values():
            # Workspace filter
            if workspace_id:
                if policy.workspace_id != workspace_id:
                    if not include_global or policy.workspace_id is not None:
                        continue

            # Type filter
            if policy_type and policy.policy_type != policy_type:
                continue

            # Status filter
            if status and policy.status != status:
                continue

            policies.append(policy)

        return policies

    def _validate_policy(self, policy: Policy) -> None:
        """Validate policy structure."""
        if len(policy.rules) > MAX_RULES_PER_POLICY:
            raise ValueError(f"Maximum rules per policy ({MAX_RULES_PER_POLICY}) exceeded")

        for rule in policy.rules:
            if not rule.conditions:
                raise ValueError(f"Rule {rule.name} has no conditions")

    # ========================================================================
    # Policy Evaluation
    # ========================================================================

    async def evaluate(
        self,
        context: PolicyEvaluationContext,
        dry_run: bool = False,
    ) -> PolicyEvaluationReport:
        """
        Evaluate all applicable policies against context.

        Args:
            context: Evaluation context
            dry_run: If True, don't enforce, just report

        Returns:
            PolicyEvaluationReport with results
        """
        import time

        start_time = time.perf_counter()

        report = PolicyEvaluationReport(
            context=context,
            dry_run=dry_run,
        )

        # Get applicable policies
        policies = await self._get_applicable_policies(context)

        if not policies:
            report.result = EvaluationResult.ALLOWED
            return report

        # Evaluate each policy
        context_dict = context.to_dict()

        for policy in policies:
            if policy.status != PolicyStatus.ACTIVE:
                continue

            effect = policy.evaluate(context_dict)

            # Record matched policy
            if effect != policy.default_effect:
                report.matched_policies.append(policy.policy_id)

            # Process effect
            if effect == PolicyEffect.DENY:
                report.result = EvaluationResult.DENIED
                report.deny_reason = f"Denied by policy: {policy.name}"
                break

            elif effect == PolicyEffect.AUDIT:
                report.audit_entries.append(
                    f"Audit: {policy.name} matched for {context.action} on {context.resource_type}"
                )

            elif effect == PolicyEffect.WARN:
                report.warnings.append(f"Warning from policy: {policy.name}")

        # Set final result if not already denied
        if report.result not in (EvaluationResult.DENIED, EvaluationResult.ERROR):
            if report.audit_entries:
                report.result = EvaluationResult.AUDIT_ONLY
            elif report.warnings:
                report.result = EvaluationResult.WARNING
            else:
                report.result = EvaluationResult.ALLOWED

        # Record timing
        report.evaluation_time_ms = (time.perf_counter() - start_time) * 1000

        # Run hooks
        for hook in self._evaluation_hooks:
            try:
                hook(report)
            except Exception as e:
                logger.warning(f"Evaluation hook error: {e}")

        return report

    async def _get_applicable_policies(
        self,
        context: PolicyEvaluationContext,
    ) -> List[Policy]:
        """Get policies applicable to the context."""
        policies = []

        # Get workspace policies
        workspace_policy_ids = self._workspace_policies.get(context.workspace_id, set())
        for policy_id in workspace_policy_ids:
            policy = self._policies.get(policy_id)
            if policy and self._policy_matches_context(policy, context):
                policies.append(policy)

        # Get global policies
        for policy_id in self._global_policies:
            policy = self._policies.get(policy_id)
            if policy and self._policy_matches_context(policy, context):
                policies.append(policy)

        return policies

    def _policy_matches_context(
        self,
        policy: Policy,
        context: PolicyEvaluationContext,
    ) -> bool:
        """Check if policy targets match context."""
        targets = policy.targets

        if not targets:
            return True  # Policy applies to everything

        # Check resource type
        if "resource_type" in targets:
            if targets["resource_type"] != context.resource_type:
                return False

        # Check action
        if "actions" in targets:
            if context.action not in targets["actions"]:
                return False

        # Check strategy IDs
        if "strategy_ids" in targets and context.strategy_id:
            if context.strategy_id not in targets["strategy_ids"]:
                return False

        # Check agent IDs
        if "agent_ids" in targets and context.agent_id:
            if context.agent_id not in targets["agent_ids"]:
                return False

        return True

    def register_evaluation_hook(self, hook: Callable[[PolicyEvaluationReport], None]) -> None:
        """Register a hook to be called after each evaluation."""
        self._evaluation_hooks.append(hook)

    # ========================================================================
    # Enforcement Points
    # ========================================================================

    async def enforce_trade_limit(
        self,
        workspace_id: UUID,
        trade_data: Dict[str, Any],
    ) -> PolicyEvaluationReport:
        """
        Enforce trade limit policies.

        Args:
            workspace_id: Workspace ID
            trade_data: Trade data to evaluate

        Returns:
            Evaluation report
        """
        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="trade",
            action="execute",
            strategy_id=trade_data.get("strategy_id"),
            agent_id=trade_data.get("agent_id"),
            data=trade_data,
        )

        return await self.evaluate(context)

    async def enforce_position_limit(
        self,
        workspace_id: UUID,
        position_data: Dict[str, Any],
    ) -> PolicyEvaluationReport:
        """Enforce position limit policies."""
        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="position",
            action="open",
            strategy_id=position_data.get("strategy_id"),
            agent_id=position_data.get("agent_id"),
            data=position_data,
        )

        return await self.evaluate(context)

    async def enforce_deployment_policy(
        self,
        workspace_id: UUID,
        deployment_data: Dict[str, Any],
    ) -> PolicyEvaluationReport:
        """Enforce deployment policies."""
        context = PolicyEvaluationContext(
            workspace_id=workspace_id,
            resource_type="deployment",
            action=deployment_data.get("action", "deploy"),
            strategy_id=deployment_data.get("strategy_id"),
            agent_id=deployment_data.get("agent_id"),
            data=deployment_data,
        )

        return await self.evaluate(context)


# ============================================================================
# Default Policies
# ============================================================================


def create_default_risk_policy(workspace_id: UUID) -> Policy:
    """Create a default risk limit policy."""
    return Policy(
        workspace_id=workspace_id,
        name="Default Risk Limits",
        description="Default position and trade risk limits",
        policy_type=PolicyType.RISK_LIMIT,
        status=PolicyStatus.ACTIVE,
        rules=[
            PolicyRule(
                name="Max Position Value",
                description="Reject positions exceeding max value",
                conditions=[
                    PolicyCondition(
                        field="position_value",
                        operator=ComparisonOperator.GREATER_THAN,
                        value=1_000_000,  # $1M max
                    )
                ],
                effect=PolicyEffect.DENY,
                priority=100,
            ),
            PolicyRule(
                name="Max Daily Loss",
                description="Halt trading if daily loss exceeds limit",
                conditions=[
                    PolicyCondition(
                        field="daily_pnl",
                        operator=ComparisonOperator.LESS_THAN,
                        value=-50_000,  # -$50K
                    )
                ],
                effect=PolicyEffect.DENY,
                priority=100,
            ),
            PolicyRule(
                name="Large Trade Warning",
                description="Warn on large trades",
                conditions=[
                    PolicyCondition(
                        field="trade_value",
                        operator=ComparisonOperator.GREATER_THAN,
                        value=100_000,  # $100K
                    )
                ],
                effect=PolicyEffect.WARN,
                priority=50,
            ),
        ],
        default_effect=PolicyEffect.ALLOW,
        targets={"resource_type": "trade"},
    )


def create_default_deployment_policy(workspace_id: UUID) -> Policy:
    """Create a default deployment policy."""
    return Policy(
        workspace_id=workspace_id,
        name="Default Deployment Policy",
        description="Require approvals for production deployments",
        policy_type=PolicyType.DEPLOYMENT,
        status=PolicyStatus.ACTIVE,
        rules=[
            PolicyRule(
                name="Production Approval Required",
                description="Require approval for production deployments",
                conditions=[
                    PolicyCondition(
                        field="environment",
                        operator=ComparisonOperator.EQUALS,
                        value="production",
                    ),
                    PolicyCondition(
                        field="approval_count",
                        operator=ComparisonOperator.LESS_THAN,
                        value=2,
                    ),
                ],
                effect=PolicyEffect.DENY,
                priority=100,
            ),
            PolicyRule(
                name="Audit All Deployments",
                description="Log all deployment activities",
                conditions=[
                    PolicyCondition(
                        field="action",
                        operator=ComparisonOperator.IN,
                        value=["deploy", "rollback", "scale"],
                    )
                ],
                effect=PolicyEffect.AUDIT,
                priority=10,
            ),
        ],
        default_effect=PolicyEffect.ALLOW,
        targets={"resource_type": "deployment"},
    )
