# -*- coding: utf-8 -*-
"""
Phase 5 Enterprise Tests - Comprehensive test coverage for Phase 5 implementation.

Tests cover:
- Enterprise API endpoints (Evidence Pack, Agent Updates, Break-glass)
- TUF Repository functionality
- Staged rollout mechanism
- Version pinning
- Break-glass access control

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - CCEA_100_PERCENT_ALIGNMENT_PLAN.md - Phase 5
"""

import asyncio
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest

# =============================================================================
# Test Evidence Pack
# =============================================================================


class TestEvidencePackExporter:
    """Tests for Evidence Pack Exporter functionality."""

    @pytest.fixture
    def evidence_config(self):
        """Create evidence pack config."""
        from packages.cloud.enterprise.evidence_pack import (
            EvidencePackConfig,
            EvidenceType,
            ExportDestination,
        )

        return EvidencePackConfig(
            evidence_types=[
                EvidenceType.ARTIFACT_DIGESTS,
                EvidenceType.APPROVAL_RECORDS,
                EvidenceType.COMMAND_LOGS,
            ],
            days_back=7,
            destination=ExportDestination.LOCAL,
            include_sensitive=False,
            redact_pii=True,
            sign_pack=False,
            compress=False,
        )

    @pytest.fixture
    def exporter(self, evidence_config):
        """Create evidence pack exporter."""
        from packages.cloud.enterprise.evidence_pack import EvidencePackExporter

        return EvidencePackExporter(evidence_config)

    def test_evidence_config_creation(self, evidence_config):
        """Test evidence config creation."""
        from packages.cloud.enterprise.evidence_pack import (
            EvidenceType,
            ExportDestination,
        )

        assert len(evidence_config.evidence_types) == 3
        assert EvidenceType.ARTIFACT_DIGESTS in evidence_config.evidence_types
        assert evidence_config.days_back == 7
        assert evidence_config.destination == ExportDestination.LOCAL
        assert evidence_config.redact_pii is True

    def test_evidence_type_coverage(self):
        """Test all evidence types are defined."""
        from packages.cloud.enterprise.evidence_pack import EvidenceType

        expected_types = [
            "ARTIFACT_DIGESTS",
            "ARTIFACT_SIGNATURES",
            "ARTIFACT_SBOM",
            "ARTIFACT_PROVENANCE",
            "DEPLOYMENT_LOGS",
            "UPGRADE_LOGS",
            "APPROVAL_RECORDS",
            "COMMAND_LOGS",
            "HALT_REASONS",
            "INCIDENT_LOGS",
            "KILL_SWITCH_EVENTS",
            "TELEMETRY_AGGREGATED",
            "TELEMETRY_DETAILED",
            "ACCESS_AUDIT",
            "BREAK_GLASS_EVENTS",
            "CONFIG_CHANGES",
            "AGENT_HEALTH",
            "SYSTEM_METRICS",
        ]

        for type_name in expected_types:
            assert hasattr(EvidenceType, type_name), f"Missing EvidenceType: {type_name}"

    def test_export_destination_types(self):
        """Test export destination types."""
        from packages.cloud.enterprise.evidence_pack import ExportDestination

        assert ExportDestination.LOCAL.value == "local"
        assert ExportDestination.S3.value == "s3"
        assert ExportDestination.GCS.value == "gcs"
        assert ExportDestination.AZURE.value == "azure"

    @pytest.mark.asyncio
    async def test_exporter_initialization(self, exporter):
        """Test exporter initialization."""
        assert exporter is not None
        assert exporter.config is not None

    @pytest.mark.asyncio
    async def test_export_creates_pack_id(self, exporter):
        """Test export creates unique pack ID."""
        result = await exporter.export(
            description="Test export",
            created_by="test_user",
        )

        assert result is not None
        assert result.id is not None
        # Pack ID should be UUID
        try:
            UUID(str(result.id))
        except ValueError:
            pytest.fail("Pack ID is not a valid UUID")


# =============================================================================
# Test TUF Repository
# =============================================================================


class TestTUFRepository:
    """Tests for TUF Repository functionality."""

    @pytest.fixture
    def tuf_config(self, tmp_path):
        """Create TUF config with temp path."""
        from packages.cloud.enterprise.tuf_repository import TUFConfig

        return TUFConfig(
            repository_path=tmp_path / "tuf",
            timestamp_expiry_days=1,
            snapshot_expiry_days=7,
            targets_expiry_days=365,
            root_expiry_days=365,
            root_threshold=1,  # Single key for testing
            consistent_snapshot=True,
        )

    @pytest.fixture
    def tuf_repo(self, tuf_config):
        """Create TUF repository."""
        from packages.cloud.enterprise.tuf_repository import TUFRepository

        return TUFRepository(tuf_config)

    def test_tuf_role_types(self):
        """Test TUF role types are defined."""
        from packages.cloud.enterprise.tuf_repository import TUFRole

        assert TUFRole.ROOT.value == "root"
        assert TUFRole.TIMESTAMP.value == "timestamp"
        assert TUFRole.SNAPSHOT.value == "snapshot"
        assert TUFRole.TARGETS.value == "targets"

    def test_tuf_key_types(self):
        """Test TUF key types."""
        from packages.cloud.enterprise.tuf_repository import KeyType

        assert KeyType.ED25519.value == "ed25519"
        assert KeyType.RSA.value == "rsa-pkcs1v15-sha256"
        assert KeyType.ECDSA.value == "ecdsa-sha2-nistp256"

    def test_tuf_key_creation(self):
        """Test TUF key dataclass."""
        from packages.cloud.enterprise.tuf_repository import KeyType, TUFKey

        key = TUFKey(
            key_id="abc123",
            key_type=KeyType.ED25519,
            public_key="base64encodedkey==",
            scheme="ed25519",
        )

        assert key.key_id == "abc123"
        assert key.key_type == KeyType.ED25519

        key_dict = key.to_dict()
        assert key_dict["keytype"] == "ed25519"
        assert "keyval" in key_dict
        assert key_dict["keyval"]["public"] == "base64encodedkey=="

    def test_tuf_signature_creation(self):
        """Test TUF signature dataclass."""
        from packages.cloud.enterprise.tuf_repository import TUFSignature

        sig = TUFSignature(
            key_id="key123",
            signature="sig_base64==",
        )

        sig_dict = sig.to_dict()
        assert sig_dict["keyid"] == "key123"
        assert sig_dict["sig"] == "sig_base64=="

    def test_target_info_creation(self):
        """Test target info dataclass."""
        from packages.cloud.enterprise.tuf_repository import TargetInfo

        target = TargetInfo(
            length=1024,
            hashes={
                "sha256": "abc123",
                "sha512": "def456",
            },
            custom={"version": "1.0.0"},
        )

        target_dict = target.to_dict()
        assert target_dict["length"] == 1024
        assert "sha256" in target_dict["hashes"]
        assert target_dict["custom"]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_tuf_repo_initialization(self, tuf_repo, tuf_config):
        """Test TUF repository initialization."""
        # Generate test keys
        root_keys = [(secrets.token_bytes(32), secrets.token_bytes(64))]
        targets_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        snapshot_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        timestamp_key = (secrets.token_bytes(32), secrets.token_bytes(64))

        result = await tuf_repo.initialize(
            root_keys=root_keys,
            targets_key=targets_key,
            snapshot_key=snapshot_key,
            timestamp_key=timestamp_key,
        )

        assert result is True
        assert tuf_repo.is_initialized is True

    @pytest.mark.asyncio
    async def test_tuf_add_target(self, tuf_repo, tuf_config):
        """Test adding target to TUF repo."""
        # Initialize first
        root_keys = [(secrets.token_bytes(32), secrets.token_bytes(64))]
        targets_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        snapshot_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        timestamp_key = (secrets.token_bytes(32), secrets.token_bytes(64))

        await tuf_repo.initialize(
            root_keys=root_keys,
            targets_key=targets_key,
            snapshot_key=snapshot_key,
            timestamp_key=timestamp_key,
        )

        # Add target
        content = b"test agent update content"
        result = await tuf_repo.add_target(
            name="agent-1.0.0.tar.gz",
            content=content,
            custom={"version": "1.0.0", "channel": "stable"},
        )

        assert result is True

        # Verify target
        target_info = tuf_repo.get_target_info("agent-1.0.0.tar.gz")
        assert target_info is not None
        assert target_info.length == len(content)

    @pytest.mark.asyncio
    async def test_tuf_verify_target(self, tuf_repo, tuf_config):
        """Test target verification."""
        # Initialize and add target
        root_keys = [(secrets.token_bytes(32), secrets.token_bytes(64))]
        targets_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        snapshot_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        timestamp_key = (secrets.token_bytes(32), secrets.token_bytes(64))

        await tuf_repo.initialize(
            root_keys=root_keys,
            targets_key=targets_key,
            snapshot_key=snapshot_key,
            timestamp_key=timestamp_key,
        )

        content = b"test content for verification"
        await tuf_repo.add_target("test-target.bin", content)
        await tuf_repo.publish()

        # Verify with correct hash
        content_hash = hashlib.sha256(content).hexdigest()
        is_valid, error = await tuf_repo.verify_target(
            "test-target.bin",
            content_hash,
            len(content),
        )

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_tuf_verify_target_wrong_hash(self, tuf_repo, tuf_config):
        """Test target verification fails with wrong hash."""
        # Initialize and add target
        root_keys = [(secrets.token_bytes(32), secrets.token_bytes(64))]
        targets_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        snapshot_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        timestamp_key = (secrets.token_bytes(32), secrets.token_bytes(64))

        await tuf_repo.initialize(
            root_keys=root_keys,
            targets_key=targets_key,
            snapshot_key=snapshot_key,
            timestamp_key=timestamp_key,
        )

        content = b"original content"
        await tuf_repo.add_target("test-target.bin", content)
        await tuf_repo.publish()

        # Verify with wrong hash
        wrong_hash = hashlib.sha256(b"tampered content").hexdigest()
        is_valid, error = await tuf_repo.verify_target(
            "test-target.bin",
            wrong_hash,
            len(content),
        )

        assert is_valid is False
        assert "mismatch" in error.lower()

    def test_tuf_metadata_versions(self, tuf_repo):
        """Test metadata version tracking."""
        versions = tuf_repo.get_metadata_versions()

        assert "root" in versions
        assert "timestamp" in versions
        assert "snapshot" in versions
        assert "targets" in versions


# =============================================================================
# Test Break-Glass Access
# =============================================================================


class TestBreakGlassController:
    """Tests for break-glass access controller."""

    @pytest.fixture
    def controller(self):
        """Create break-glass controller."""
        from packages.cloud.governance.break_glass import BreakGlassController

        return BreakGlassController(
            approvers={"admin@company.com", "security@company.com"},
            require_approval=True,
        )

    def test_break_glass_reason_types(self):
        """Test break-glass reason types."""
        from packages.cloud.governance.break_glass import BreakGlassReason

        expected_reasons = [
            "INCIDENT_RESPONSE",
            "SECURITY_INVESTIGATION",
            "COMPLIANCE_AUDIT",
            "DATA_RECOVERY",
            "SYSTEM_FAILURE",
            "CUSTOMER_EMERGENCY",
            "OTHER",
        ]

        for reason in expected_reasons:
            assert hasattr(BreakGlassReason, reason)

    def test_break_glass_scope_types(self):
        """Test break-glass scope types."""
        from packages.cloud.governance.break_glass import BreakGlassScope

        assert BreakGlassScope.TELEMETRY_READ is not None
        assert BreakGlassScope.AUDIT_READ is not None
        assert BreakGlassScope.CONFIG_READ is not None
        assert BreakGlassScope.ADMIN_ACCESS is not None
        assert BreakGlassScope.DATA_EXPORT is not None

    def test_create_request_validates_reason_length(self, controller):
        """Test request creation validates reason length."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        # Short reason should fail
        with pytest.raises(ValueError, match="at least"):
            controller.create_request(
                requester_id="user123",
                requester_email="user@company.com",
                reason="too short",  # Less than 20 chars
                reason_type=BreakGlassReason.INCIDENT_RESPONSE,
                workspace_id="ws123",
                scope={BreakGlassScope.TELEMETRY_READ},
            )

    def test_create_request_success(self, controller):
        """Test successful request creation."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Investigating production incident #12345 affecting customer data",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ, BreakGlassScope.AUDIT_READ},
            duration_hours=4,
        )

        assert request is not None
        assert request.requester_id == "user123"
        assert request.approved is False
        assert len(request.scope) == 2
        assert request.evidence_hash != ""

    def test_request_caps_duration(self, controller):
        """Test request caps duration at max."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
            MAX_BREAK_GLASS_DURATION_HOURS,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Long duration request for extended investigation",
            reason_type=BreakGlassReason.SECURITY_INVESTIGATION,
            workspace_id="ws123",
            scope={BreakGlassScope.ADMIN_ACCESS},
            duration_hours=100,  # Way over max
        )

        assert request.duration_hours == MAX_BREAK_GLASS_DURATION_HOURS

    def test_approve_request(self, controller):
        """Test request approval."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        # Create request
        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Production incident requiring immediate access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Approve
        result = controller.approve_request(
            request_id=request.id,
            approved_by="admin@company.com",
            notes="Approved for incident investigation",
        )

        assert result.success is True
        assert result.access_token is not None
        assert result.expires_at is not None

    def test_unauthorized_approver_denied(self, controller):
        """Test unauthorized approver is denied."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Production incident requiring immediate access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Try unauthorized approval
        result = controller.approve_request(
            request_id=request.id,
            approved_by="random@company.com",  # Not in approvers list
        )

        assert result.success is False
        assert "not authorized" in result.error.lower()

    def test_revoke_request(self, controller):
        """Test request revocation."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Production incident requiring immediate access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        # Approve first
        controller.approve_request(
            request_id=request.id,
            approved_by="admin@company.com",
        )

        # Then revoke
        revoked = controller.revoke_request(
            request_id=request.id,
            revoked_by="security@company.com",
            reason="Investigation completed",
        )

        assert revoked is True

        # Verify request is no longer active
        updated = controller.get_request(request.id)
        assert updated.is_active is False
        assert updated.revoked_at is not None

    def test_has_access_checks_scope(self, controller):
        """Test access check validates scope."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Production incident requiring immediate access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},  # Only telemetry
        )

        controller.approve_request(
            request_id=request.id,
            approved_by="admin@company.com",
        )

        # Should have access to telemetry
        assert controller.has_access(request.id, BreakGlassScope.TELEMETRY_READ) is True

        # Should NOT have access to admin
        assert controller.has_access(request.id, BreakGlassScope.ADMIN_ACCESS) is False

    def test_audit_log_recorded(self, controller):
        """Test audit log is recorded."""
        from packages.cloud.governance.break_glass import (
            BreakGlassReason,
            BreakGlassScope,
        )

        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Production incident requiring immediate access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},
        )

        controller.approve_request(
            request_id=request.id,
            approved_by="admin@company.com",
        )

        # Check audit log
        audit_log = controller.get_audit_log(request_id=request.id)

        assert len(audit_log) >= 2  # At least create and approve
        actions = [entry["action"] for entry in audit_log]
        assert "request_created" in actions
        assert "request_approved" in actions


# =============================================================================
# Test Agent Updates
# =============================================================================


class TestAgentUpdateManager:
    """Tests for Agent Update Manager."""

    @pytest.fixture
    def update_manager(self):
        """Create agent update manager."""
        from packages.cloud.enterprise.agent_updates import AgentUpdateManager

        return AgentUpdateManager()

    def test_update_channel_types(self):
        """Test update channel types."""
        from packages.cloud.enterprise.agent_updates import UpdateChannel

        assert UpdateChannel.STABLE.value == "stable"
        assert UpdateChannel.BETA.value == "beta"
        assert UpdateChannel.CANARY.value == "canary"
        assert UpdateChannel.ENTERPRISE.value == "enterprise"

    def test_update_state_types(self):
        """Test update state types."""
        from packages.cloud.enterprise.agent_updates import UpdateState

        # Check that UpdateState enum exists
        assert UpdateState is not None

    def test_create_update(self, update_manager):
        """Test creating agent update."""
        from packages.cloud.enterprise.agent_updates import UpdateChannel

        update = update_manager.create_update(
            version="1.2.0",
            channel=UpdateChannel.STABLE,
            artifact_url="https://releases.ccea.io/agent-1.2.0.tar.gz",
            artifact_digest="sha256:" + "a" * 64,
            artifact_size_bytes=1024 * 1024,
            release_notes="Bug fixes and improvements",
        )

        assert update is not None
        assert update.version == "1.2.0"
        assert update.channel == UpdateChannel.STABLE


# =============================================================================
# Test Version Pinning
# =============================================================================


class TestVersionPinning:
    """Tests for version pinning functionality."""

    @pytest.fixture
    def pin_manager(self):
        """Create version pin manager."""
        from packages.cloud.enterprise.version_pinning import VersionPinningManager

        return VersionPinningManager()

    def test_constraint_types(self):
        """Test version constraint types."""
        from packages.cloud.enterprise.version_pinning import PinType

        assert PinType.EXACT.value == "exact"
        assert PinType.MAJOR.value == "major"
        assert PinType.MINOR.value == "minor"
        assert PinType.RANGE.value == "range"
        assert PinType.LATEST.value == "latest"

    def test_pin_scope_types(self):
        """Test pin scope types."""
        from packages.cloud.enterprise.version_pinning import PinScope

        assert PinScope.GLOBAL.value == "global"
        assert PinScope.ORGANIZATION.value == "organization"
        assert PinScope.WORKSPACE.value == "workspace"
        assert PinScope.AGENT.value == "agent"

    def test_create_constraint(self, pin_manager):
        """Test creating version constraint."""
        from packages.cloud.enterprise.version_pinning import (
            PinType,
            PinScope,
        )

        constraint = pin_manager.create_constraint(
            name="Production Pin",
            pin_type=PinType.EXACT,
            pin_scope=PinScope.ORGANIZATION,
            organization_id=uuid4(),
            version="1.2.3",
            reason="Production stability requirement",
        )

        assert constraint is not None
        assert constraint.version == "1.2.3"
        assert constraint.pin_type == PinType.EXACT

    def test_version_check_exact(self, pin_manager):
        """Test version check for exact constraint."""
        from packages.cloud.enterprise.version_pinning import (
            PinType,
            PinScope,
        )

        agent_id = uuid4()

        pin_manager.create_constraint(
            name="Exact Pin",
            pin_type=PinType.EXACT,
            pin_scope=PinScope.GLOBAL,
            version="1.2.3",
            reason="Test",
        )

        # Exact version should be allowed
        result = pin_manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.2.3",
        )
        assert result.is_allowed is True

        # Different version should be blocked
        result = pin_manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.2.4",
        )
        assert result.is_allowed is False


# =============================================================================
# Test Staged Rollout
# =============================================================================


class TestStagedRollout:
    """Tests for staged rollout functionality."""

    @pytest.fixture
    def rollout_manager(self):
        """Create staged rollout manager."""
        from packages.cloud.enterprise.staged_rollout import StagedRolloutManager

        return StagedRolloutManager()

    def test_rollout_stage_definition(self):
        """Test rollout stage dataclass."""
        from packages.cloud.enterprise.staged_rollout import RolloutStage

        stage = RolloutStage(
            name="canary",
            percentage=5.0,
            duration_hours=24,
            min_success_rate=0.99,
        )

        assert stage.name == "canary"
        assert stage.percentage == 5.0
        assert stage.duration_hours == 24
        assert stage.min_success_rate == 0.99

    def test_rollout_create(self, rollout_manager):
        """Test creating rollout."""
        rollout = rollout_manager.create_rollout(
            name="Test Rollout",
            target_version="1.2.0",
            artifact_digest="sha256:" + "a" * 64,
        )

        assert rollout is not None
        assert rollout.name == "Test Rollout"
        assert rollout.target_version == "1.2.0"

    def test_agent_in_rollout_deterministic(self, rollout_manager):
        """Test agent rollout targeting is deterministic."""
        rollout = rollout_manager.create_rollout(
            name="Deterministic Test",
            target_version="1.0.0",
        )

        agent_id = uuid4()

        # Should be deterministic
        result1 = rollout_manager.is_agent_in_rollout(agent_id, rollout.id)
        result2 = rollout_manager.is_agent_in_rollout(agent_id, rollout.id)

        assert result1 == result2


# =============================================================================
# Test Enterprise API Router
# =============================================================================


class TestEnterpriseRouter:
    """Tests for enterprise API router."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI

            from packages.cloud.control_plane.routers.enterprise import router

            app = FastAPI()
            app.include_router(router, prefix="/api/v1")

            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_evidence_pack_endpoint_types(self, client):
        """Test evidence pack endpoint accepts valid types."""
        response = client.post(
            "/api/v1/enterprise/evidence-packs",
            json={
                "description": "Test export",
                "evidence_types": ["artifact_digests", "approval_records"],
                "days_back": 7,
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    def test_list_evidence_packs(self, client):
        """Test listing evidence packs."""
        response = client.get("/api/v1/enterprise/evidence-packs")

        assert response.status_code == 200
        data = response.json()
        assert "packs" in data
        assert "total" in data
        assert "page" in data

    def test_agent_update_creation(self, client):
        """Test agent update creation."""
        response = client.post(
            "/api/v1/enterprise/agent-updates",
            json={
                "version": "1.2.0",
                "channel": "stable",
                "title": "Bug fixes",
                "description": "Various bug fixes",
                "priority": "normal",
                "artifact_digest": "sha256:" + "a" * 64,
                "artifact_url": "https://releases.ccea.io/agent-1.2.0.tar.gz",
                "artifact_size_bytes": 1024 * 1024,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["version"] == "1.2.0"
        assert data["channel"] == "stable"

    def test_version_pin_creation(self, client):
        """Test version pin creation."""
        response = client.post(
            "/api/v1/enterprise/version-pins",
            json={
                "scope": "organization",
                "scope_id": str(uuid4()),
                "constraint_type": "exact",
                "version": "1.2.3",
                "reason": "Production stability requirement for compliance",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["version"] == "1.2.3"
        assert data["is_active"] is True

    def test_break_glass_creation(self, client):
        """Test break-glass request creation."""
        response = client.post(
            "/api/v1/enterprise/break-glass",
            json={
                "reason": "Investigating production incident #12345 affecting customer data flow",
                "reason_type": "incident_response",
                "workspace_id": str(uuid4()),
                "scope": ["telemetry_read"],
                "duration_hours": 4,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["approved"] is False
        assert "evidence_hash" in data
        assert len(data["evidence_hash"]) == 64  # SHA256 hex

    def test_break_glass_reason_validation(self, client):
        """Test break-glass reason length validation."""
        response = client.post(
            "/api/v1/enterprise/break-glass",
            json={
                "reason": "too short",  # Less than 20 chars
                "reason_type": "incident_response",
                "workspace_id": str(uuid4()),
                "scope": ["telemetry_read"],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_enterprise_statistics(self, client):
        """Test enterprise statistics endpoint."""
        response = client.get("/api/v1/enterprise/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "evidence_packs" in data
        assert "agent_updates" in data
        assert "version_pins" in data
        assert "break_glass" in data


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase5Integration:
    """Integration tests for Phase 5 components."""

    @pytest.mark.asyncio
    async def test_full_evidence_export_workflow(self, tmp_path):
        """Test complete evidence export workflow."""
        from packages.cloud.enterprise.evidence_pack import (
            EvidencePackConfig,
            EvidencePackExporter,
            EvidenceType,
            ExportDestination,
        )

        config = EvidencePackConfig(
            evidence_types=[EvidenceType.ARTIFACT_DIGESTS],
            days_back=1,
            destination=ExportDestination.LOCAL,
            destination_config={"path": str(tmp_path)},
        )

        exporter = EvidencePackExporter(config)
        result = await exporter.export(
            description="Integration test export",
            created_by="test",
        )

        assert result is not None
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_full_tuf_publish_workflow(self, tmp_path):
        """Test complete TUF publish workflow."""
        from packages.cloud.enterprise.tuf_repository import TUFConfig, TUFRepository

        config = TUFConfig(
            repository_path=tmp_path / "tuf",
            root_threshold=1,
        )

        repo = TUFRepository(config)

        # Initialize
        root_keys = [(secrets.token_bytes(32), secrets.token_bytes(64))]
        targets_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        snapshot_key = (secrets.token_bytes(32), secrets.token_bytes(64))
        timestamp_key = (secrets.token_bytes(32), secrets.token_bytes(64))

        await repo.initialize(
            root_keys=root_keys,
            targets_key=targets_key,
            snapshot_key=snapshot_key,
            timestamp_key=timestamp_key,
        )

        # Add target
        await repo.add_target("agent-1.0.0.tar.gz", b"agent content")

        # Publish
        result = await repo.publish()
        assert result is True

        # Verify files created
        assert (tmp_path / "tuf" / "root.json").exists()
        assert (tmp_path / "tuf" / "timestamp.json").exists()
        assert (tmp_path / "tuf" / "snapshot.json").exists()
        assert (tmp_path / "tuf" / "targets.json").exists()

    def test_full_break_glass_workflow(self):
        """Test complete break-glass workflow."""
        from packages.cloud.governance.break_glass import (
            BreakGlassController,
            BreakGlassReason,
            BreakGlassScope,
        )

        controller = BreakGlassController(
            approvers={"admin@company.com"},
            require_approval=True,
        )

        # 1. Create request
        request = controller.create_request(
            requester_id="user123",
            requester_email="user@company.com",
            reason="Investigating production incident requiring telemetry access",
            reason_type=BreakGlassReason.INCIDENT_RESPONSE,
            workspace_id="ws123",
            scope={BreakGlassScope.TELEMETRY_READ},
            duration_hours=2,
        )

        assert request.approved is False

        # 2. Approve request
        result = controller.approve_request(
            request_id=request.id,
            approved_by="admin@company.com",
            notes="Approved for incident #123",
        )

        assert result.success is True
        assert result.access_token is not None

        # 3. Use access
        has_access = controller.has_access(
            request.id,
            BreakGlassScope.TELEMETRY_READ,
        )
        assert has_access is True

        # 4. Verify audit trail
        audit = controller.get_audit_log(request_id=request.id)
        assert len(audit) >= 2  # create, approve

        # 5. Revoke access
        revoked = controller.revoke_request(
            request_id=request.id,
            revoked_by="admin@company.com",
            reason="Investigation complete",
        )
        assert revoked is True

        # 6. Verify no longer has access
        has_access = controller.has_access(
            request.id,
            BreakGlassScope.TELEMETRY_READ,
        )
        assert has_access is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
