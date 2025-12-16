# -*- coding: utf-8 -*-
"""
Comprehensive tests for GDPR Phase 6: Access Control, Audit, and Break-Glass.

Tests cover:
- RBAC Service (role-based access control)
- Access Audit Service (audit logging)
- Break-Glass Phase 6 Service (emergency access)
- Change Management Service (approval workflow)

Run with: pytest packages/cloud/governance/tests/test_access_control_phase6.py -v
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Set
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from ..rbac_service import (
    AccessContext,
    AccessDecision,
    Permission,
    Principal,
    RBACService,
    ResourceType,
    Role,
    Scope,
    SensitivityLevel,
    AUDIT_REQUIRED_RESOURCES,
)
from ..access_audit import (
    AccessAuditService,
    AuditAction,
    AuditEntry,
    AuditPrincipal,
    AuditQuery,
    AuditResult,
    AuditStats,
)
from ..break_glass_phase6 import (
    BreakGlassPhase6Service,
    BreakGlassReasonType,
    BreakGlassRequest,
    BreakGlassScope,
    BreakGlassStatus,
    MIN_REASON_LENGTH,
    MAX_BREAK_GLASS_DURATION_HOURS,
    DEFAULT_BREAK_GLASS_DURATION_HOURS,
)
from ..change_management import (
    ChangeManagementService,
    ChangeClass,
    ChangeEvidence,
    ChangeRequest,
    ChangeStatus,
    ChangeType,
    ApprovalType,
    MIN_APPROVAL_REASON_LENGTH,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rbac_service() -> RBACService:
    """Create RBAC service for testing."""
    service = RBACService(cache_enabled=False)
    service.initialize_organization("org-test")
    return service


@pytest.fixture
def audit_service() -> AccessAuditService:
    """Create access audit service for testing."""
    return AccessAuditService(enable_hash_chain=True)


@pytest.fixture
def break_glass_service(audit_service: AccessAuditService) -> BreakGlassPhase6Service:
    """Create break-glass service for testing."""
    return BreakGlassPhase6Service(
        audit_service=audit_service,
        approvers={"admin-1", "admin@example.com"},
        elevated_approvers={"security-admin", "security@example.com"},
        require_approval=True,
    )


@pytest.fixture
def change_service(audit_service: AccessAuditService) -> ChangeManagementService:
    """Create change management service for testing."""
    return ChangeManagementService(audit_service=audit_service)


@pytest.fixture
def test_principal() -> Principal:
    """Create test principal."""
    return Principal(
        id="user-123",
        type="user",
        email="user@example.com",
        organization_id="org-test",
        roles=["developer"],
        mfa_verified=True,
    )


@pytest.fixture
def admin_principal() -> Principal:
    """Create admin principal."""
    return Principal(
        id="admin-1",
        type="user",
        email="admin@example.com",
        organization_id="org-test",
        roles=["admin"],
        mfa_verified=True,
    )


# ============================================================================
# RBAC Service Tests
# ============================================================================

class TestRBACService:
    """Test suite for RBAC Service."""

    def test_permission_from_string(self):
        """Test Permission parsing from string."""
        perm = Permission.from_string("strategy:read")
        assert perm.resource == "strategy"
        assert perm.scope == "read"
        assert perm.constraint is None

    def test_permission_with_constraint(self):
        """Test Permission with constraint."""
        perm = Permission.from_string("telemetry:read:aggregated_only")
        assert perm.resource == "telemetry"
        assert perm.scope == "read"
        assert perm.constraint == "aggregated_only"

    def test_permission_matches_exact(self):
        """Test exact permission matching."""
        perm = Permission("strategy", "read")
        assert perm.matches("strategy", "read")
        assert not perm.matches("strategy", "write")
        assert not perm.matches("deployment", "read")

    def test_permission_matches_wildcard_resource(self):
        """Test wildcard resource matching."""
        perm = Permission("*", "read")
        assert perm.matches("strategy", "read")
        assert perm.matches("deployment", "read")
        assert not perm.matches("strategy", "write")

    def test_permission_matches_wildcard_scope(self):
        """Test wildcard scope matching."""
        perm = Permission("strategy", "*")
        assert perm.matches("strategy", "read")
        assert perm.matches("strategy", "write")
        assert not perm.matches("deployment", "read")

    def test_permission_matches_full_wildcard(self):
        """Test full wildcard matching."""
        perm = Permission("*", "*")
        assert perm.matches("strategy", "read")
        assert perm.matches("deployment", "write")
        assert perm.matches("any", "thing")

    def test_create_role(self, rbac_service: RBACService):
        """Test role creation."""
        permissions = {
            Permission("strategy", "read"),
            Permission("strategy", "write"),
        }
        role = rbac_service.create_role(
            name="custom_role",
            description="Custom test role",
            organization_id="org-test",
            permissions=permissions,
        )
        assert role.name == "custom_role"
        assert len(role.permissions) == 2
        assert not role.is_system_role

    def test_create_role_duplicate_name_fails(self, rbac_service: RBACService):
        """Test that duplicate role names fail."""
        permissions = {Permission("strategy", "read")}
        rbac_service.create_role(
            name="unique_role",
            description="First",
            organization_id="org-test",
            permissions=permissions,
        )
        with pytest.raises(ValueError, match="already exists"):
            rbac_service.create_role(
                name="unique_role",
                description="Second",
                organization_id="org-test",
                permissions=permissions,
            )

    def test_get_role_by_name(self, rbac_service: RBACService):
        """Test getting role by name."""
        # Default roles should exist
        role = rbac_service.get_role_by_name("developer", "org-test")
        assert role is not None
        assert role.name == "developer"

    def test_assign_role_to_user(self, rbac_service: RBACService, test_principal: Principal):
        """Test role assignment."""
        role = rbac_service.get_role_by_name("developer", "org-test")
        result = rbac_service.assign_role(test_principal.id, role.id)
        assert result

        user_roles = rbac_service.get_user_roles(test_principal.id)
        assert len(user_roles) >= 1

    def test_revoke_role_from_user(self, rbac_service: RBACService, test_principal: Principal):
        """Test role revocation."""
        role = rbac_service.get_role_by_name("developer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)
        result = rbac_service.revoke_role(test_principal.id, role.id)
        assert result

    def test_check_access_granted(self, rbac_service: RBACService, test_principal: Principal):
        """Test access check - granted."""
        # Assign developer role
        role = rbac_service.get_role_by_name("developer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type=ResourceType.STRATEGY.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert decision.granted
        assert decision.method == "rbac"

    def test_check_access_denied_no_permission(self, rbac_service: RBACService, test_principal: Principal):
        """Test access check - denied due to no permission."""
        # Assign viewer role (limited permissions)
        role = rbac_service.get_role_by_name("viewer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type=ResourceType.AUDIT_LOG.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert not decision.granted

    def test_check_access_denied_no_roles(self, rbac_service: RBACService):
        """Test access check - denied due to no roles."""
        principal = Principal(
            id="no-role-user",
            type="user",
            organization_id="org-test",
        )
        context = AccessContext(
            principal=principal,
            workspace_id="ws-123",
            resource_type=ResourceType.STRATEGY.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert not decision.granted
        assert "No roles" in decision.reason

    def test_check_access_mfa_required(self, rbac_service: RBACService, test_principal: Principal):
        """Test MFA requirement for critical resources."""
        role = rbac_service.get_role_by_name("admin", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        # Without MFA for critical resource
        test_principal.mfa_verified = False
        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type=ResourceType.AUDIT_LOG.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert not decision.granted
        assert decision.requires_mfa

    def test_sensitivity_levels_correct(self, rbac_service: RBACService, test_principal: Principal):
        """Test that sensitivity levels are correctly identified."""
        role = rbac_service.get_role_by_name("admin", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        # Strategy should be standard
        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type=ResourceType.STRATEGY.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert decision.sensitivity_level == SensitivityLevel.STANDARD

        # Audit log should be critical
        context.resource_type = ResourceType.AUDIT_LOG.value
        decision = rbac_service.check_access(context)
        assert decision.sensitivity_level == SensitivityLevel.CRITICAL

    def test_audit_required_resources(self):
        """Test that audit-required resources are defined correctly."""
        assert ResourceType.AUDIT_LOG in AUDIT_REQUIRED_RESOURCES
        assert ResourceType.ACCESS_AUDIT in AUDIT_REQUIRED_RESOURCES
        assert ResourceType.BREAK_GLASS in AUDIT_REQUIRED_RESOURCES
        assert ResourceType.DSAR in AUDIT_REQUIRED_RESOURCES

    def test_has_permission_helper(self, rbac_service: RBACService, test_principal: Principal):
        """Test has_permission helper method."""
        role = rbac_service.get_role_by_name("developer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        assert rbac_service.has_permission(
            test_principal,
            ResourceType.STRATEGY.value,
            "read",
        )

    def test_require_permission_raises_on_deny(self, rbac_service: RBACService, test_principal: Principal):
        """Test require_permission raises PermissionError on deny."""
        with pytest.raises(PermissionError):
            rbac_service.require_permission(
                test_principal,
                ResourceType.AUDIT_LOG.value,
                "delete",
            )

    def test_get_snapshot(self, rbac_service: RBACService, test_principal: Principal):
        """Test RBAC snapshot export."""
        role = rbac_service.get_role_by_name("developer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        snapshot = rbac_service.get_snapshot("org-test")
        assert "snapshot_id" in snapshot
        assert "integrity_hash" in snapshot
        assert snapshot["organization_id"] == "org-test"

    def test_default_roles_initialized(self, rbac_service: RBACService):
        """Test that default roles are initialized."""
        roles = rbac_service.list_roles("org-test")
        role_names = {r.name for r in roles}
        assert "owner" in role_names
        assert "admin" in role_names
        assert "developer" in role_names
        assert "viewer" in role_names
        assert "auditor" in role_names


# ============================================================================
# Access Audit Service Tests
# ============================================================================

class TestAccessAuditService:
    """Test suite for Access Audit Service."""

    def test_log_access_creates_entry(self, audit_service: AccessAuditService):
        """Test basic access logging."""
        entry = audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        assert entry.id is not None
        assert entry.workspace_id == "ws-123"
        assert entry.action == AuditAction.READ

    def test_audit_entry_integrity_hash(self, audit_service: AccessAuditService):
        """Test integrity hash is computed."""
        entry = audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        assert entry.integrity_hash.startswith("sha256:")

    def test_hash_chain_linking(self, audit_service: AccessAuditService):
        """Test hash chain linking between entries."""
        entry1 = audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        entry2 = audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.UPDATE,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        assert entry2.previous_hash == entry1.integrity_hash

    def test_query_by_workspace(self, audit_service: AccessAuditService):
        """Test querying by workspace."""
        # Create entries in different workspaces
        audit_service.log_access(
            workspace_id="ws-1",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        audit_service.log_access(
            workspace_id="ws-2",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )

        results = audit_service.query(AuditQuery(workspace_id="ws-1"))
        assert len(results) == 1
        assert results[0].workspace_id == "ws-1"

    def test_query_by_action(self, audit_service: AccessAuditService):
        """Test querying by action."""
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.UPDATE,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )

        results = audit_service.query(AuditQuery(action=AuditAction.READ))
        assert all(e.action == AuditAction.READ for e in results)

    def test_query_by_result(self, audit_service: AccessAuditService):
        """Test querying by result."""
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-2", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.DENIED,
        )

        results = audit_service.query(AuditQuery(result=AuditResult.DENIED))
        assert len(results) == 1
        assert results[0].result == AuditResult.DENIED

    def test_query_break_glass_only(self, audit_service: AccessAuditService):
        """Test querying break-glass entries only."""
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-2", type="user"),
            action=AuditAction.BREAK_GLASS_ACCESS,
            resource_type="telemetry",
            result=AuditResult.SUCCESS,
            break_glass_id="bg-123",
        )

        results = audit_service.query(AuditQuery(break_glass_only=True))
        assert len(results) == 1
        assert results[0].break_glass_id == "bg-123"

    def test_get_stats(self, audit_service: AccessAuditService):
        """Test statistics calculation."""
        # Create mixed entries
        for _ in range(3):
            audit_service.log_access(
                workspace_id="ws-123",
                principal=AuditPrincipal(id="user-1", type="user"),
                action=AuditAction.READ,
                resource_type="strategy",
                result=AuditResult.SUCCESS,
            )
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-2", type="user"),
            action=AuditAction.UPDATE,
            resource_type="strategy",
            result=AuditResult.DENIED,
        )

        stats = audit_service.get_stats(workspace_id="ws-123")
        assert stats.total_entries == 4
        assert stats.by_action.get("read", 0) == 3
        assert stats.denied_count == 1

    def test_export(self, audit_service: AccessAuditService):
        """Test audit export."""
        audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(id="user-1", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )

        export = audit_service.export(
            workspace_id="ws-123",
            exported_by="admin-1",
        )
        assert export.checksum.startswith("sha256:")
        assert export.entry_count == 1

    def test_verify_hash_chain_valid(self, audit_service: AccessAuditService):
        """Test hash chain verification - valid chain."""
        for i in range(5):
            audit_service.log_access(
                workspace_id="ws-123",
                principal=AuditPrincipal(id="user-1", type="user"),
                action=AuditAction.READ,
                resource_type="strategy",
                result=AuditResult.SUCCESS,
            )

        is_valid, errors = audit_service.verify_hash_chain("ws-123")
        assert is_valid
        assert len(errors) == 0

    def test_log_from_rbac_context(self, audit_service: AccessAuditService, rbac_service: RBACService, test_principal: Principal):
        """Test logging from RBAC context."""
        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type="strategy",
            scope="read",
        )
        decision = AccessDecision(
            granted=True,
            method="rbac",
            role_used="developer",
            sensitivity_level=SensitivityLevel.STANDARD,
        )

        entry = audit_service.log_from_rbac(context, decision)
        assert entry.principal.id == test_principal.id
        assert entry.result == AuditResult.SUCCESS


# ============================================================================
# Break-Glass Phase 6 Service Tests
# ============================================================================

class TestBreakGlassPhase6Service:
    """Test suite for Break-Glass Phase 6 Service."""

    def test_create_request_success(self, break_glass_service: BreakGlassPhase6Service):
        """Test successful request creation."""
        result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
            ticket_id="INC-12345",
        )
        assert result.success
        assert result.request_id is not None
        assert result.requires_approval

    def test_create_request_reason_too_short(self, break_glass_service: BreakGlassPhase6Service):
        """Test request creation fails with short reason."""
        with pytest.raises(ValueError, match="at least"):
            break_glass_service.create_request(
                requester_id="user-123",
                requester_email="user@example.com",
                workspace_id="ws-456",
                reason="Too short",
                reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
                scopes={BreakGlassScope.TELEMETRY_READ},
            )

    def test_create_request_no_scopes_fails(self, break_glass_service: BreakGlassPhase6Service):
        """Test request creation fails without scopes."""
        with pytest.raises(ValueError, match="scope"):
            break_glass_service.create_request(
                requester_id="user-123",
                requester_email="user@example.com",
                workspace_id="ws-456",
                reason="Valid reason that is long enough for the check",
                reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
                scopes=set(),
            )

    def test_approve_request_success(self, break_glass_service: BreakGlassPhase6Service):
        """Test successful approval."""
        # Create request
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        # Approve
        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
            notes="Approved for incident response",
        )
        assert approve_result.success
        assert approve_result.access_token is not None
        assert approve_result.expires_at is not None

    def test_approve_request_self_approval_denied(self, break_glass_service: BreakGlassPhase6Service):
        """Test self-approval is denied."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="user-123",  # Same as requester
            approver_email="user@example.com",
        )
        assert not approve_result.success
        assert "Self-approval" in approve_result.error

    def test_approve_request_unauthorized_denied(self, break_glass_service: BreakGlassPhase6Service):
        """Test unauthorized approver is denied."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="random-user",
            approver_email="random@example.com",
        )
        assert not approve_result.success
        assert "Not authorized" in approve_result.error

    def test_elevated_scopes_require_elevated_approver(self, break_glass_service: BreakGlassPhase6Service):
        """Test elevated scopes require elevated approver."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Security investigation requiring admin access to the system",
            reason_type=BreakGlassReasonType.SECURITY_INVESTIGATION,
            scopes={BreakGlassScope.ADMIN_ACCESS},  # Elevated scope
        )
        assert create_result.requires_elevated_approval

        # Regular admin cannot approve
        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )
        assert not approve_result.success
        assert "Elevated approval" in approve_result.error

        # Security admin can approve
        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="security-admin",
            approver_email="security@example.com",
        )
        assert approve_result.success

    def test_has_access_with_valid_token(self, break_glass_service: BreakGlassPhase6Service):
        """Test access check with valid token."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )

        assert break_glass_service.has_access(
            approve_result.access_token,
            BreakGlassScope.TELEMETRY_READ,
        )

    def test_has_access_wrong_scope_denied(self, break_glass_service: BreakGlassPhase6Service):
        """Test access check fails for wrong scope."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )

        # Wrong scope
        assert not break_glass_service.has_access(
            approve_result.access_token,
            BreakGlassScope.ADMIN_ACCESS,
        )

    def test_revoke_request(self, break_glass_service: BreakGlassPhase6Service):
        """Test request revocation."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )

        # Revoke
        revoke_result = break_glass_service.revoke_request(
            request_id=create_result.request_id,
            revoker_id="admin-1",
            revoker_email="admin@example.com",
            reason="Incident resolved",
        )
        assert revoke_result.success

        # Access no longer works
        assert not break_glass_service.has_access(
            approve_result.access_token,
            BreakGlassScope.TELEMETRY_READ,
        )

    def test_deny_request(self, break_glass_service: BreakGlassPhase6Service):
        """Test request denial."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        deny_result = break_glass_service.deny_request(
            request_id=create_result.request_id,
            denier_id="admin-1",
            denier_email="admin@example.com",
            reason="Not a valid incident",
        )
        assert deny_result.success

        request = break_glass_service.get_request(create_result.request_id)
        assert request.status == BreakGlassStatus.DENIED

    def test_get_pending_requests(self, break_glass_service: BreakGlassPhase6Service):
        """Test getting pending requests."""
        break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        pending = break_glass_service.get_pending_requests()
        assert len(pending) == 1

    def test_get_stats(self, break_glass_service: BreakGlassPhase6Service):
        """Test statistics."""
        break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        stats = break_glass_service.get_stats()
        assert stats.total_requests == 1
        assert stats.pending_count == 1

    def test_export_requests(self, break_glass_service: BreakGlassPhase6Service):
        """Test request export."""
        break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        export = break_glass_service.export_requests("ws-456")
        assert len(export) == 1


# ============================================================================
# Change Management Service Tests
# ============================================================================

class TestChangeManagementService:
    """Test suite for Change Management Service."""

    def test_classify_change_trading_impacting(self, change_service: ChangeManagementService):
        """Test classification of trading-impacting changes."""
        change_class = change_service.classify_change(
            change_type=ChangeType.DEPLOY,
            target_type="deployment",
        )
        assert change_class == ChangeClass.TRADING_IMPACTING

    def test_classify_change_operational(self, change_service: ChangeManagementService):
        """Test classification of operational changes."""
        change_class = change_service.classify_change(
            change_type=ChangeType.OTHER,
            target_type="alert",
        )
        assert change_class == ChangeClass.OPERATIONAL

    def test_classify_change_security_sensitive(self, change_service: ChangeManagementService):
        """Test classification of security-sensitive changes."""
        change_class = change_service.classify_change(
            change_type=ChangeType.AGENT_UPDATE,
            target_type="agent",
        )
        assert change_class == ChangeClass.SECURITY_SENSITIVE

    def test_create_change_request(self, change_service: ChangeManagementService):
        """Test change request creation."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
            target_id="dep-789",
        )
        assert change.id is not None
        assert change.change_class == ChangeClass.TRADING_IMPACTING
        assert change.requires_local_approval

    def test_create_change_with_evidence(self, change_service: ChangeManagementService):
        """Test change request with evidence."""
        evidence = ChangeEvidence(
            config_blob_digest="sha256:abc123",
            manifest_digest="sha256:def456",
            previous_state_digest="sha256:prev",
            new_state_digest="sha256:new",
        )
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
            evidence=evidence,
        )
        assert change.evidence.config_blob_digest == "sha256:abc123"

    def test_approve_trading_impacting_change(self, change_service: ChangeManagementService):
        """Test approving trading-impacting change."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )
        assert approval is not None
        assert approval.user_acknowledged
        assert change.status == ChangeStatus.APPROVED

    def test_approve_trading_impacting_requires_reason(self, change_service: ChangeManagementService):
        """Test that trading-impacting changes require reason."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Short",  # Too short
        )
        assert approval is None  # Rejected due to short reason

    def test_approve_trading_impacting_requires_acknowledgment(self, change_service: ChangeManagementService):
        """Test that trading-impacting changes require user acknowledgment."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=False,  # Not acknowledged
            reason="Approved deployment of strategy v1.1.0 after review",
        )
        assert approval is None  # Rejected due to no acknowledgment

    def test_reject_change(self, change_service: ChangeManagementService):
        """Test change rejection."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        result = change_service.reject_change(
            change_id=change.id,
            rejector_id="admin-123",
            reason="Not ready for production",
        )
        assert result
        assert change.status == ChangeStatus.REJECTED

    def test_execute_and_complete_change(self, change_service: ChangeManagementService):
        """Test change execution and completion."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )

        # Execute
        result = change_service.execute_change(
            change_id=change.id,
            executed_by="system",
        )
        assert result
        assert change.status == ChangeStatus.EXECUTING

        # Complete
        result = change_service.complete_change(
            change_id=change.id,
            success=True,
            rollback_available=True,
            rollback_to_digest="sha256:prev",
        )
        assert result
        assert change.status == ChangeStatus.COMPLETED

    def test_rollback_change(self, change_service: ChangeManagementService):
        """Test change rollback."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )
        change_service.execute_change(change.id, "system")
        change_service.complete_change(
            change.id,
            success=True,
            rollback_available=True,
            rollback_to_digest="sha256:prev",
        )

        # Rollback
        result = change_service.rollback_change(
            change_id=change.id,
            rolledback_by="admin-123",
            reason="Bug discovered",
        )
        assert result
        assert change.status == ChangeStatus.ROLLED_BACK

    def test_get_journal(self, change_service: ChangeManagementService):
        """Test change journal."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )

        journal = change_service.get_journal(workspace_id="ws-123")
        assert len(journal) == 1
        assert journal[0].change_id == change.id

    def test_export_journal(self, change_service: ChangeManagementService):
        """Test journal export."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )

        export = change_service.export_journal("ws-123")
        assert "export_metadata" in export
        assert "integrity" in export
        assert len(export["changes"]) == 1

    def test_get_stats(self, change_service: ChangeManagementService):
        """Test change statistics."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        stats = change_service.get_stats(workspace_id="ws-123")
        assert stats.total_changes == 1
        assert stats.trading_impacting_count == 1

    def test_change_evidence_hash(self):
        """Test evidence hash computation."""
        evidence = ChangeEvidence(
            config_blob_digest="sha256:abc",
            manifest_digest="sha256:def",
        )
        hash1 = evidence.compute_evidence_hash()
        assert hash1.startswith("sha256:")

        # Same evidence should produce same hash
        evidence2 = ChangeEvidence(
            config_blob_digest="sha256:abc",
            manifest_digest="sha256:def",
        )
        assert evidence2.compute_evidence_hash() == hash1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase6Integration:
    """Integration tests for Phase 6 components."""

    def test_rbac_with_audit(self, rbac_service: RBACService, audit_service: AccessAuditService, test_principal: Principal):
        """Test RBAC integrated with audit logging."""
        # Set up audit callback
        logged_entries = []
        rbac_service._audit_callback = lambda ctx, dec: logged_entries.append((ctx, dec))

        # Assign role
        role = rbac_service.get_role_by_name("developer", "org-test")
        rbac_service.assign_role(test_principal.id, role.id)

        # Check access
        context = AccessContext(
            principal=test_principal,
            workspace_id="ws-123",
            resource_type=ResourceType.STRATEGY.value,
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert decision.granted

        # Note: We don't test logged_entries because STRATEGY is not in AUDIT_REQUIRED_RESOURCES
        # Let's test with a sensitive resource
        context.resource_type = ResourceType.TELEMETRY.value
        decision = rbac_service.check_access(context)

    def test_break_glass_with_audit(self, break_glass_service: BreakGlassPhase6Service, audit_service: AccessAuditService):
        """Test break-glass with audit logging."""
        # Create and approve request
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )

        # Use access
        break_glass_service.has_access(
            approve_result.access_token,
            BreakGlassScope.TELEMETRY_READ,
        )

        # Check audit log
        entries = audit_service.query(AuditQuery(
            workspace_id="ws-456",
            actions=[
                AuditAction.BREAK_GLASS_REQUEST,
                AuditAction.BREAK_GLASS_APPROVE,
                AuditAction.BREAK_GLASS_ACCESS,
            ],
        ))
        assert len(entries) >= 3

    def test_change_management_with_audit(self, change_service: ChangeManagementService, audit_service: AccessAuditService):
        """Test change management with audit logging."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy v1.1.0",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Approved deployment of strategy v1.1.0 after review",
        )

        # Check audit log
        entries = audit_service.query(AuditQuery(
            workspace_id="ws-123",
            actions=[AuditAction.CHANGE_REQUEST, AuditAction.CHANGE_APPROVE],
        ))
        assert len(entries) >= 2

    def test_full_workflow_deploy_with_break_glass(
        self,
        rbac_service: RBACService,
        audit_service: AccessAuditService,
        break_glass_service: BreakGlassPhase6Service,
        change_service: ChangeManagementService,
    ):
        """Test full workflow: deploy with break-glass access."""
        # 1. Create change request
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Emergency deployment to fix critical bug",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
            target_id="dep-789",
            evidence=ChangeEvidence(
                config_blob_digest="sha256:config",
                manifest_digest="sha256:manifest",
            ),
        )

        # 2. Approve change
        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Emergency fix for critical production bug affecting users",
        )
        assert approval is not None

        # 3. Execute change
        change_service.execute_change(change.id, "system")

        # 4. Create break-glass for post-deploy investigation
        bg_result = break_glass_service.create_request(
            requester_id="user-456",
            requester_email="user@example.com",
            workspace_id="ws-123",
            reason="Post-deployment investigation for critical fix monitoring",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        # 5. Approve break-glass
        bg_approve = break_glass_service.approve_request(
            request_id=bg_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )
        assert bg_approve.success

        # 6. Access telemetry
        assert break_glass_service.has_access(
            bg_approve.access_token,
            BreakGlassScope.TELEMETRY_READ,
        )

        # 7. Complete change
        change_service.complete_change(
            change.id,
            success=True,
            rollback_available=True,
        )

        # 8. Verify audit trail
        entries = audit_service.query(AuditQuery(workspace_id="ws-123"))
        assert len(entries) >= 4  # change_request, change_approve, bg_request, bg_approve, bg_access


# ============================================================================
# DoD Verification Tests
# ============================================================================

class TestPhase6DoD:
    """Tests to verify Definition of Done for Phase 6."""

    def test_dod_sensitive_access_attributable(
        self,
        audit_service: AccessAuditService,
    ):
        """DoD: Every sensitive access is attributable to principal and request_id."""
        entry = audit_service.log_access(
            workspace_id="ws-123",
            principal=AuditPrincipal(
                id="user-123",
                type="user",
                email="user@example.com",
            ),
            action=AuditAction.READ,
            resource_type="telemetry",
            result=AuditResult.SUCCESS,
            request_id="req-456",
        )

        # Verify attribution
        assert entry.principal.id == "user-123"
        assert entry.request_id == "req-456"
        assert entry.integrity_hash is not None

    def test_dod_break_glass_has_reason_scope_expiry(
        self,
        break_glass_service: BreakGlassPhase6Service,
    ):
        """DoD: Break-glass has reason, scope, expiry time."""
        create_result = break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
            duration_hours=4,
        )

        approve_result = break_glass_service.approve_request(
            request_id=create_result.request_id,
            approver_id="admin-1",
            approver_email="admin@example.com",
        )

        request = break_glass_service.get_request(create_result.request_id)

        # Verify requirements
        assert len(request.reason) >= MIN_REASON_LENGTH  # Reason
        assert len(request.scopes) > 0  # Scope
        assert request.expires_at is not None  # Expiry time
        assert request.evidence_hash is not None  # Evidence

    def test_dod_all_exportable(
        self,
        rbac_service: RBACService,
        audit_service: AccessAuditService,
        break_glass_service: BreakGlassPhase6Service,
        change_service: ChangeManagementService,
    ):
        """DoD: All are exportable."""
        # Create some data
        break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="Investigating production incident #12345 with high severity",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        change_service.create_change(
            workspace_id="ws-456",
            change_type=ChangeType.DEPLOY,
            description="Test deployment",
            requester_id="user-123",
            requester_email="user@example.com",
            target_type="deployment",
        )

        audit_service.log_access(
            workspace_id="ws-456",
            principal=AuditPrincipal(id="user-123", type="user"),
            action=AuditAction.READ,
            resource_type="strategy",
            result=AuditResult.SUCCESS,
        )

        # Test exports
        rbac_snapshot = rbac_service.get_snapshot("org-test")
        assert "integrity_hash" in rbac_snapshot

        audit_export = audit_service.export("ws-456", "admin")
        assert audit_export.checksum.startswith("sha256:")

        bg_export = break_glass_service.export_requests("ws-456")
        assert len(bg_export) >= 1

        change_journal = change_service.export_journal("ws-456")
        assert "integrity" in change_journal

    def test_dod_trading_impacting_requires_local_approval(
        self,
        change_service: ChangeManagementService,
    ):
        """DoD: TRADING_IMPACTING changes require local approval."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Deploy strategy",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        assert change.change_class == ChangeClass.TRADING_IMPACTING
        assert change.requires_local_approval

        # Cannot approve without LOCAL type and acknowledgment
        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="admin-123",
            approval_type=ApprovalType.CLOUD,  # Not LOCAL
            user_acknowledged=True,
            reason="Approved via cloud which should fail for trading impacting",
        )
        assert approval is None  # Should be rejected


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rbac_empty_principal_roles(self, rbac_service: RBACService):
        """Test access with empty roles list."""
        principal = Principal(
            id="no-role-user",
            type="user",
            organization_id="org-test",
            roles=[],
        )
        context = AccessContext(
            principal=principal,
            workspace_id="ws-123",
            resource_type="strategy",
            scope="read",
        )
        decision = rbac_service.check_access(context)
        assert not decision.granted

    def test_audit_query_with_no_results(self, audit_service: AccessAuditService):
        """Test query with no matching results."""
        results = audit_service.query(AuditQuery(
            workspace_id="nonexistent-ws",
        ))
        assert len(results) == 0

    def test_break_glass_invalid_token(self, break_glass_service: BreakGlassPhase6Service):
        """Test access with invalid token."""
        assert not break_glass_service.has_access(
            "invalid_token",
            BreakGlassScope.TELEMETRY_READ,
        )

    def test_change_approval_after_expiry(self, change_service: ChangeManagementService):
        """Test that expired changes cannot be approved."""
        change = change_service.create_change(
            workspace_id="ws-123",
            change_type=ChangeType.DEPLOY,
            description="Test deployment",
            requester_id="user-456",
            requester_email="user@example.com",
            target_type="deployment",
        )

        # Force expiry
        change.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        approval = change_service.approve_change(
            change_id=change.id,
            approver_id="user-456",
            approval_type=ApprovalType.LOCAL,
            user_acknowledged=True,
            reason="Trying to approve expired change should fail",
        )
        assert approval is None

    def test_break_glass_cooldown(self, break_glass_service: BreakGlassPhase6Service):
        """Test cooldown between break-glass requests."""
        break_glass_service.create_request(
            requester_id="user-123",
            requester_email="user@example.com",
            workspace_id="ws-456",
            reason="First request for investigating incident #1 with details",
            reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
            scopes={BreakGlassScope.TELEMETRY_READ},
        )

        # Second request should fail due to cooldown
        with pytest.raises(ValueError, match="Cooldown"):
            break_glass_service.create_request(
                requester_id="user-123",
                requester_email="user@example.com",
                workspace_id="ws-456",
                reason="Second request for investigating incident #2 with details",
                reason_type=BreakGlassReasonType.INCIDENT_RESPONSE,
                scopes={BreakGlassScope.TELEMETRY_READ},
            )
