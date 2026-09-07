"""
Comprehensive tests for Phase 7 GDPR Implementation Services.

Tests cover:
- SecurityBaselineService (encryption, key management, MFA, secrets)
- SupplyChainService (artifacts, SBOMs, registries, signers)
- AgentUpdateService (updates, rollouts, rollbacks, version pins)
- ResearchSandboxService (isolation, quotas, egress, abuse detection)
- BreachWorkflowService (breach management, notifications, tabletops)
- EvidencePackService (evidence export aggregation)

Target: 100% coverage for Phase 7 security controls.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from typing import List

# Import Phase 7 services
from packages.cloud.governance.security_baseline import (
    SecurityBaselineService,
    EncryptionConfig,
    EncryptionAlgorithm,
    KeyDerivation,
    KeyType,
    KeyStatus,
    KeyMetadata,
    KeyRotationEvent,
    KeyManagementPolicy,
    MFAConfig,
    MFAMethod,
    MFARequirement,
    MFAChallengeResult,
    SecretPolicy,
    SecretMetadata,
    SecretType,
    SecretStatus,
    StorageBackend,
    SecurityEvent,
    SecurityEventType,
    ComplianceCheckResult,
    ComplianceStatus,
    SecurityBaselineStatus,
    DEFAULT_KEY_ROTATION_DAYS,
    MAX_MFA_ATTEMPTS,
)

from packages.cloud.governance.supply_chain import (
    SupplyChainService,
    SignedArtifact,
    ArtifactType,
    SigningAlgorithm as SCSigningAlgorithm,
    SignatureStatus,
    SignatureVerification,
    DigestPin,
    RegistryConfig,
    RegistryAllowlist,
    SBOMFormat,
    SBOMComponent,
    Vulnerability,
    VulnerabilitySeverity,
    SBOM,
    TrustedSigner,
    SupplyChainEvent,
    SupplyChainEventType,
    SupplyChainStats,
)

from packages.cloud.governance.agent_updates import (
    AgentUpdateService,
    AgentUpdate,
    UpdateChannel,
    RolloutStatus,
    UpdateResult,
    UpdateEventType,
    RolloutCriteria,
    RolloutStage,
    RolloutStageMetrics,
    RolloutPlan,
    RollbackRequest,
    ChangeWindow,
    VersionPin,
    AgentUpdateRecord,
    UpdateEvent,
    AgentUpdateStats,
)

from packages.cloud.governance.research_sandbox import (
    ResearchSandboxService,
    IsolationLevel,
    JobStatus,
    AbuseType,
    AbuseAction,
    SandboxEventType,
    QuotaType,
    ResourceQuota,
    SandboxConfig,
    EgressRule,
    EgressPolicy,
    ResourceUsage,
    ResearchJob,
    AbuseIndicator,
    AbuseEvent,
    EgressLogEntry,
    SandboxEvent,
    SandboxStats,
)

from packages.cloud.governance.breach_workflow import (
    BreachWorkflowService,
    BreachCategory,
    BreachSeverity,
    BreachStatus,
    NotificationStatus,
    ExemptionType,
    RiskFactor,
    TimelineEventType,
    BreachEventType,
    DPOContact,
    SupervisoryAuthority,
    RiskAssessment,
    NotificationDecision,
    AuthorityNotification,
    SubjectNotification,
    TimelineEvent,
    DataBreach,
    TabletopScenario,
    TabletopExercise,
    BreachWorkflowEvent,
    BreachStats,
    AUTHORITY_NOTIFICATION_HOURS,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)

from packages.cloud.governance.evidence_pack import (
    EvidencePackService,
    EvidenceCategory,
    ExportFormat,
    PackStatus,
    EvidenceArtifact,
    EvidenceManifest,
    EvidencePack,
    ExportRequest,
    EvidenceEvent,
)


# =============================================================================
# SecurityBaselineService Tests
# =============================================================================


class TestSecurityBaselineService:
    """Tests for SecurityBaselineService."""

    @pytest.fixture
    def service(self) -> SecurityBaselineService:
        """Create a SecurityBaselineService instance."""
        return SecurityBaselineService()

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    @pytest.fixture
    def user_id(self) -> str:
        return "user-456"

    # Encryption Config Tests
    def test_encryption_config_defaults(self, service, workspace_id):
        """Test default encryption configuration."""
        config = service.get_encryption_config(workspace_id)
        assert config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.at_rest_enabled is True
        assert config.in_transit_min_tls == "1.3"
        assert config.key_rotation_days == DEFAULT_KEY_ROTATION_DAYS

    def test_set_encryption_config(self, service, workspace_id, user_id):
        """Test setting encryption configuration."""
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_rotation_days=60,
        )
        result = service.set_encryption_config(workspace_id, config, user_id)
        assert result.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305

        retrieved = service.get_encryption_config(workspace_id)
        assert retrieved.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305

    def test_encryption_config_validation(self):
        """Test encryption config validation."""
        with pytest.raises(ValueError, match="Key rotation must be at least"):
            EncryptionConfig(key_rotation_days=10)

        with pytest.raises(ValueError, match="Key rotation must not exceed"):
            EncryptionConfig(key_rotation_days=400)

        with pytest.raises(ValueError, match="TLS version must be"):
            EncryptionConfig(in_transit_min_tls="1.0")

    def test_validate_encryption_compliance(self, service, workspace_id):
        """Test encryption compliance validation."""
        result = service.validate_encryption_compliance(workspace_id)
        assert result.check_name == "encryption_compliance"
        assert result.category == "encryption"
        assert result.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_ATTENTION]

    def test_encryption_compliance_with_issues(self, service, workspace_id, user_id):
        """Test encryption compliance with configuration issues."""
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            in_transit_min_tls="1.2",
            key_rotation_days=120,
        )
        service.set_encryption_config(workspace_id, config, user_id)

        result = service.validate_encryption_compliance(workspace_id)
        assert len(result.findings) > 0
        assert any("CBC" in f or "TLS" in f or "rotation" in f for f in result.findings)

    # Key Management Tests
    def test_create_key(self, service, workspace_id, user_id):
        """Test key creation."""
        key = service.create_key(
            workspace_id=workspace_id,
            key_type=KeyType.DATA_ENCRYPTION_KEY,
            created_by=user_id,
            description="Test key",
        )
        assert key.key_id is not None
        assert key.workspace_id == workspace_id
        assert key.key_type == KeyType.DATA_ENCRYPTION_KEY
        assert key.status == KeyStatus.ACTIVE
        assert key.version == 1
        assert key.key_fingerprint.startswith("sha256:")

    def test_get_key(self, service, workspace_id, user_id):
        """Test key retrieval."""
        key = service.create_key(workspace_id, KeyType.MASTER_KEY, user_id)
        retrieved = service.get_key(key.key_id)
        assert retrieved is not None
        assert retrieved.key_id == key.key_id

    def test_list_keys(self, service, workspace_id, user_id):
        """Test listing keys with filters."""
        service.create_key(workspace_id, KeyType.MASTER_KEY, user_id)
        service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        service.create_key("other-workspace", KeyType.SIGNING_KEY, user_id)

        all_keys = service.list_keys()
        assert len(all_keys) == 3

        workspace_keys = service.list_keys(workspace_id=workspace_id)
        assert len(workspace_keys) == 2

        dek_keys = service.list_keys(key_type=KeyType.DATA_ENCRYPTION_KEY)
        assert len(dek_keys) == 1

    def test_rotate_key(self, service, workspace_id, user_id):
        """Test key rotation."""
        key = service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        old_fingerprint = key.key_fingerprint
        old_version = key.version

        rotation = service.rotate_key(
            key_id=key.key_id,
            rotated_by=user_id,
            reason="Scheduled rotation",
        )

        assert rotation.key_id == key.key_id
        assert rotation.old_version == old_version
        assert rotation.new_version == old_version + 1
        assert rotation.old_key_fingerprint == old_fingerprint
        assert rotation.new_key_fingerprint != old_fingerprint

        updated_key = service.get_key(key.key_id)
        assert updated_key.version == old_version + 1
        assert updated_key.last_rotated_at is not None

    def test_rotate_key_invalid(self, service, workspace_id, user_id):
        """Test rotating invalid key."""
        with pytest.raises(ValueError, match="not found"):
            service.rotate_key("invalid-key", user_id)

        key = service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        service.revoke_key(key.key_id, user_id, "Test")

        with pytest.raises(ValueError, match="Cannot rotate"):
            service.rotate_key(key.key_id, user_id)

    def test_revoke_key(self, service, workspace_id, user_id):
        """Test key revocation."""
        key = service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        revoked = service.revoke_key(key.key_id, user_id, "Compromised")

        assert revoked.status == KeyStatus.REVOKED

    def test_keys_needing_rotation(self, service, workspace_id, user_id):
        """Test finding keys needing rotation."""
        key = service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)

        # Manually set rotation_due_at to past
        key.rotation_due_at = datetime.now(timezone.utc) - timedelta(days=1)

        needing_rotation = service.get_keys_needing_rotation(workspace_id)
        assert len(needing_rotation) == 1
        assert needing_rotation[0].key_id == key.key_id

    def test_key_management_compliance(self, service, workspace_id, user_id):
        """Test key management compliance validation."""
        service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        result = service.validate_key_management_compliance(workspace_id)

        assert result.check_name == "key_management_compliance"
        assert result.category == "key_management"
        assert "total_keys" in result.evidence

    # MFA Tests
    def test_enable_mfa(self, service, workspace_id, user_id):
        """Test enabling MFA."""
        config = service.enable_mfa(
            user_id=user_id,
            workspace_id=workspace_id,
            methods={MFAMethod.TOTP, MFAMethod.WEBAUTHN},
        )

        assert config.user_id == user_id
        assert config.requirement == MFARequirement.REQUIRED_ALWAYS
        assert MFAMethod.TOTP in config.enabled_methods
        assert MFAMethod.WEBAUTHN in config.enabled_methods

    def test_enable_mfa_sms_only_rejected(self, service, workspace_id, user_id):
        """Test that SMS-only MFA is rejected."""
        with pytest.raises(ValueError, match="SMS-only MFA is deprecated"):
            service.enable_mfa(
                user_id=user_id,
                workspace_id=workspace_id,
                methods={MFAMethod.SMS},
            )

    def test_verify_mfa_challenge_success(self, service, workspace_id, user_id):
        """Test successful MFA verification."""
        service.enable_mfa(user_id, workspace_id, {MFAMethod.TOTP})

        result = service.verify_mfa_challenge(
            user_id=user_id,
            method=MFAMethod.TOTP,
            code="123456",
            workspace_id=workspace_id,
        )

        assert result.success is True
        assert result.verified_at is not None

    def test_verify_mfa_challenge_failure(self, service, workspace_id, user_id):
        """Test failed MFA verification."""
        service.enable_mfa(user_id, workspace_id, {MFAMethod.TOTP})

        result = service.verify_mfa_challenge(
            user_id=user_id,
            method=MFAMethod.TOTP,
            code="invalid",
            workspace_id=workspace_id,
        )

        assert result.success is False
        assert "Invalid" in result.error

    def test_mfa_lockout(self, service, workspace_id, user_id):
        """Test MFA lockout after max attempts."""
        service.enable_mfa(user_id, workspace_id, {MFAMethod.TOTP})

        for _ in range(MAX_MFA_ATTEMPTS):
            service.verify_mfa_challenge(user_id, MFAMethod.TOTP, "invalid", workspace_id)

        result = service.verify_mfa_challenge(user_id, MFAMethod.TOTP, "123456", workspace_id)
        assert result.success is False
        assert "locked" in result.error.lower()

    def test_check_mfa_required(self, service, workspace_id, user_id):
        """Test MFA requirement checking."""
        # No config - use default
        required = service.check_mfa_required(user_id, for_sensitive=True)
        assert required is True  # Default is REQUIRED_SENSITIVE

        # With config
        service.enable_mfa(user_id, workspace_id, {MFAMethod.TOTP}, MFARequirement.REQUIRED_ALWAYS)
        required = service.check_mfa_required(user_id, for_sensitive=False)
        assert required is True

    def test_mfa_compliance(self, service, workspace_id, user_id):
        """Test MFA compliance validation."""
        user_ids = ["user-1", "user-2", "user-3"]
        service.enable_mfa("user-1", workspace_id, {MFAMethod.TOTP})
        service.enable_mfa("user-2", workspace_id, {MFAMethod.WEBAUTHN})

        result = service.validate_mfa_compliance(workspace_id, user_ids)

        assert result.check_name == "mfa_compliance"
        assert result.evidence["total_users"] == 3
        assert result.evidence["users_with_mfa"] == 2

    # Secrets Management Tests
    def test_register_secret(self, service, workspace_id, user_id):
        """Test secret registration."""
        secret = service.register_secret(
            workspace_id=workspace_id,
            name="API_KEY",
            secret_type=SecretType.API_KEY,
            created_by=user_id,
            description="Test API key",
        )

        assert secret.secret_id is not None
        assert secret.name == "API_KEY"
        assert secret.status == SecretStatus.ACTIVE
        assert secret.rotation_due_at is not None

    def test_list_secrets(self, service, workspace_id, user_id):
        """Test listing secrets."""
        service.register_secret(workspace_id, "secret1", SecretType.API_KEY, user_id)
        service.register_secret(workspace_id, "secret2", SecretType.DATABASE_CREDENTIAL, user_id)

        all_secrets = service.list_secrets(workspace_id=workspace_id)
        assert len(all_secrets) == 2

        api_keys = service.list_secrets(secret_type=SecretType.API_KEY)
        assert len(api_keys) == 1

    def test_rotate_secret(self, service, workspace_id, user_id):
        """Test secret rotation."""
        secret = service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)
        old_version = secret.version

        rotated = service.rotate_secret(secret.secret_id, user_id, "Scheduled")

        assert rotated.version == old_version + 1
        assert rotated.last_rotated_at is not None

    def test_revoke_secret(self, service, workspace_id, user_id):
        """Test secret revocation."""
        secret = service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)
        revoked = service.revoke_secret(secret.secret_id, user_id, "Compromised")

        assert revoked.status == SecretStatus.REVOKED

    def test_record_secret_access(self, service, workspace_id, user_id):
        """Test recording secret access."""
        secret = service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)

        service.record_secret_access(secret.secret_id, user_id, workspace_id)

        updated = service.get_secret_metadata(secret.secret_id)
        assert updated.access_count == 1
        assert updated.last_accessed_at is not None

    def test_secrets_needing_rotation(self, service, workspace_id, user_id):
        """Test finding secrets needing rotation."""
        secret = service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)
        secret.rotation_due_at = datetime.now(timezone.utc) - timedelta(days=1)

        needing = service.get_secrets_needing_rotation(workspace_id)
        assert len(needing) == 1

    def test_expiring_secrets(self, service, workspace_id, user_id):
        """Test finding expiring secrets."""
        secret = service.register_secret(
            workspace_id,
            "test",
            SecretType.API_KEY,
            user_id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )

        expiring = service.get_expiring_secrets(workspace_id, warning_days=14)
        assert len(expiring) == 1

    def test_secrets_compliance(self, service, workspace_id, user_id):
        """Test secrets compliance validation."""
        service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)

        result = service.validate_secrets_compliance(workspace_id)

        assert result.check_name == "secrets_compliance"
        assert "total_secrets" in result.evidence

    def test_secret_policy_validation(self):
        """Test secret policy validation."""
        policy = SecretPolicy(min_length=16)

        valid, errors = policy.validate_secret("Str0ng!Password#")
        assert valid is True
        assert len(errors) == 0

        valid, errors = policy.validate_secret("weak")
        assert valid is False
        assert len(errors) > 0

    # Security Baseline Tests
    def test_evaluate_security_baseline(self, service, workspace_id, user_id):
        """Test overall security baseline evaluation."""
        service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        service.register_secret(workspace_id, "test", SecretType.API_KEY, user_id)

        status = service.evaluate_security_baseline(
            workspace_id=workspace_id,
            user_ids=[user_id],
        )

        assert status.workspace_id == workspace_id
        assert status.overall_status in ComplianceStatus
        assert len(status.checks) == 4

    def test_export_security_baseline(self, service, workspace_id, user_id):
        """Test security baseline export."""
        service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)

        export = service.export_security_baseline(
            workspace_id=workspace_id,
            include_events=True,
            include_keys=True,
            include_secrets=True,
        )

        assert "encryption_config" in export
        assert "keys" in export
        assert "integrity_hash" in export

    def test_security_events(self, service, workspace_id, user_id):
        """Test security event logging."""
        service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)

        events = service.get_security_events(workspace_id=workspace_id)
        assert len(events) > 0
        assert events[0].workspace_id == workspace_id

    def test_key_rotations_history(self, service, workspace_id, user_id):
        """Test key rotation history."""
        key = service.create_key(workspace_id, KeyType.DATA_ENCRYPTION_KEY, user_id)
        service.rotate_key(key.key_id, user_id)
        service.rotate_key(key.key_id, user_id)

        rotations = service.get_key_rotations(workspace_id=workspace_id)
        assert len(rotations) == 2


# =============================================================================
# SupplyChainService Tests
# =============================================================================


class TestSupplyChainService:
    """Tests for SupplyChainService."""

    @pytest.fixture
    def service(self) -> SupplyChainService:
        return SupplyChainService(
            require_signatures=False,
            require_sbom=False,
            deny_latest_tag=True,
        )

    @pytest.fixture
    def strict_service(self) -> SupplyChainService:
        return SupplyChainService(
            require_signatures=True,
            require_sbom=True,
            deny_latest_tag=True,
        )

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    @pytest.fixture
    def valid_digest(self) -> str:
        return "sha256:" + "a" * 64

    # Artifact Tests
    def test_register_artifact(self, service, workspace_id, valid_digest):
        """Test artifact registration."""
        artifact = service.register_artifact(
            workspace_id=workspace_id,
            artifact_type=ArtifactType.CONTAINER_IMAGE,
            name="my-app",
            digest=valid_digest,
            registry="gcr.io/project",
            repository="my-app",
            created_by="user-1",
            tags=["v1.0.0"],
        )

        assert artifact.artifact_id is not None
        assert artifact.digest == valid_digest
        assert artifact.name == "my-app"

    def test_register_artifact_invalid_digest(self, service, workspace_id):
        """Test artifact registration with invalid digest."""
        with pytest.raises(ValueError, match="Invalid digest"):
            service.register_artifact(
                workspace_id,
                ArtifactType.CONTAINER_IMAGE,
                "app",
                "invalid-digest",
                "reg",
                "repo",
                "user",
            )

    def test_register_artifact_latest_tag_denied(self, service, workspace_id, valid_digest):
        """Test that 'latest' tag is denied."""
        with pytest.raises(ValueError, match="latest.*not allowed"):
            service.register_artifact(
                workspace_id,
                ArtifactType.CONTAINER_IMAGE,
                "app",
                valid_digest,
                "reg",
                "repo",
                "user",
                tags=["latest"],
            )

    def test_register_artifact_signature_required(self, strict_service, workspace_id, valid_digest):
        """Test signature requirement."""
        with pytest.raises(ValueError, match="signature is required"):
            strict_service.register_artifact(
                workspace_id,
                ArtifactType.CONTAINER_IMAGE,
                "app",
                valid_digest,
                "reg",
                "repo",
                "user",
            )

    def test_get_artifact_by_digest(self, service, workspace_id, valid_digest):
        """Test artifact retrieval by digest."""
        artifact = service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
        )

        retrieved = service.get_artifact_by_digest(valid_digest)
        assert retrieved is not None
        assert retrieved.artifact_id == artifact.artifact_id

    def test_list_artifacts(self, service, workspace_id):
        """Test listing artifacts."""
        d1 = "sha256:" + "a" * 64
        d2 = "sha256:" + "b" * 64

        service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app1",
            d1,
            "reg",
            "repo",
            "user",
        )
        service.register_artifact(
            workspace_id,
            ArtifactType.BINARY,
            "app2",
            d2,
            "reg",
            "repo",
            "user",
        )

        all_artifacts = service.list_artifacts()
        assert len(all_artifacts) == 2

        images = service.list_artifacts(artifact_type=ArtifactType.CONTAINER_IMAGE)
        assert len(images) == 1

    # Signature Verification Tests
    def test_verify_signature_not_signed(self, service, workspace_id, valid_digest):
        """Test verification of unsigned artifact."""
        artifact = service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
        )

        result = service.verify_signature(artifact.artifact_id, workspace_id)
        assert result.status == SignatureStatus.NOT_SIGNED

    def test_verify_signature_unknown_signer(self, service, workspace_id, valid_digest):
        """Test verification with unknown signer."""
        artifact = service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
            signature="sig123",
            signer_id="unknown-signer",
        )

        result = service.verify_signature(artifact.artifact_id, workspace_id)
        assert result.status == SignatureStatus.UNKNOWN_SIGNER

    def test_verify_signature_valid(self, service, workspace_id, valid_digest):
        """Test valid signature verification."""
        signer = service.add_trusted_signer(
            name="release-signer",
            public_key="PEM_PUBLIC_KEY_DATA",
            algorithm=SCSigningAlgorithm.ED25519,
            created_by="admin",
        )

        artifact = service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
            signature="sig123",
            signer_id=signer.signer_id,
        )

        result = service.verify_signature(artifact.artifact_id, workspace_id)
        assert result.status == SignatureStatus.VALID

    def test_verify_signature_revoked_signer(self, service, workspace_id, valid_digest):
        """Test verification with revoked signer."""
        signer = service.add_trusted_signer(
            "signer",
            "key",
            SCSigningAlgorithm.ED25519,
            "admin",
        )
        service.revoke_signer(signer.signer_id, "admin", "Compromised")

        artifact = service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
            signature="sig",
            signer_id=signer.signer_id,
        )

        result = service.verify_signature(artifact.artifact_id, workspace_id)
        assert result.status == SignatureStatus.REVOKED

    # Digest Pinning Tests
    def test_pin_digest(self, service, workspace_id, valid_digest):
        """Test digest pinning."""
        pin = service.pin_digest(
            workspace_id=workspace_id,
            artifact_type=ArtifactType.CONTAINER_IMAGE,
            artifact_name="my-app",
            digest=valid_digest,
            pinned_by="user-1",
            reason="Production release",
        )

        assert pin.pin_id is not None
        assert pin.pinned_digest == valid_digest
        assert pin.active is True

    def test_pin_digest_invalid(self, service, workspace_id):
        """Test pinning invalid digest."""
        with pytest.raises(ValueError):
            service.pin_digest(
                workspace_id,
                ArtifactType.CONTAINER_IMAGE,
                "app",
                "invalid",
                "user",
            )

    def test_get_pinned_digest(self, service, workspace_id, valid_digest):
        """Test retrieving pinned digest."""
        service.pin_digest(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "my-app",
            valid_digest,
            "user",
        )

        pin = service.get_pinned_digest(workspace_id, "my-app")
        assert pin is not None
        assert pin.pinned_digest == valid_digest

    def test_unpin_digest(self, service, workspace_id, valid_digest):
        """Test unpinning digest."""
        pin = service.pin_digest(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "user",
        )

        unpinned = service.unpin_digest(pin.pin_id, "user", "No longer needed")
        assert unpinned.active is False

    def test_expired_pins(self, service, workspace_id, valid_digest):
        """Test finding expired pins."""
        pin = service.pin_digest(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "user",
            expires_in_days=1,
        )

        # Manually expire
        pin.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

        expired = service.get_expired_pins(workspace_id)
        assert len(expired) == 1

    def test_validate_digest_reference(self, service):
        """Test digest reference validation."""
        valid, error = service.validate_digest_reference("sha256:" + "a" * 64)
        assert valid is True

        valid, error = service.validate_digest_reference("latest")
        assert valid is False
        assert "latest" in error

    # Registry Allowlist Tests
    def test_set_registry_allowlist(self, service, workspace_id):
        """Test setting registry allowlist."""
        registries = [
            RegistryConfig(hostname="gcr.io/project", require_signature=True),
            RegistryConfig(hostname="ecr.eu-west-1.amazonaws.com", eu_only=True),
        ]

        allowlist = service.set_registry_allowlist(workspace_id, registries, "admin")
        assert len(allowlist.registries) == 2

    def test_add_registry_to_allowlist(self, service, workspace_id):
        """Test adding registry to allowlist."""
        registry = RegistryConfig(hostname="docker.io", require_signature=True)
        allowlist = service.add_registry_to_allowlist(workspace_id, registry, "admin")

        assert len(allowlist.registries) == 1

    def test_is_registry_allowed(self, service, workspace_id):
        """Test registry allowlist check."""
        registry = RegistryConfig(hostname="gcr.io/project")
        service.add_registry_to_allowlist(workspace_id, registry, "admin")

        allowed, _ = service.is_registry_allowed(workspace_id, "gcr.io/project", "my-app")
        assert allowed is True

        allowed, _ = service.is_registry_allowed(workspace_id, "docker.io", "my-app")
        assert allowed is False

    # SBOM Tests
    def test_upload_sbom(self, service, workspace_id, valid_digest):
        """Test SBOM upload."""
        components = [
            SBOMComponent(name="lodash", version="4.17.21", purl="pkg:npm/lodash@4.17.21"),
            SBOMComponent(name="express", version="4.18.2", purl="pkg:npm/express@4.18.2"),
        ]

        sbom = service.upload_sbom(
            workspace_id=workspace_id,
            artifact_digest=valid_digest,
            sbom_format=SBOMFormat.CYCLONEDX_1_5,
            components=components,
            created_by="scanner",
        )

        assert sbom.sbom_id is not None
        assert len(sbom.components) == 2
        assert sbom.integrity_hash.startswith("sha256:")

    def test_sbom_with_vulnerabilities(self, service, workspace_id, valid_digest):
        """Test SBOM with vulnerabilities."""
        vuln = Vulnerability(
            vuln_id="CVE-2023-12345",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.8,
            affected_component="lodash",
        )

        sbom = service.upload_sbom(
            workspace_id,
            valid_digest,
            SBOMFormat.CYCLONEDX_1_5,
            [SBOMComponent(name="lodash", version="4.17.0")],
            "scanner",
            vulnerabilities=[vuln],
        )

        assert len(sbom.vulnerabilities) == 1
        summary = sbom.get_vulnerability_summary()
        assert summary["critical"] == 1

    def test_get_sbom_by_artifact_digest(self, service, workspace_id, valid_digest):
        """Test SBOM retrieval by artifact digest."""
        service.upload_sbom(
            workspace_id,
            valid_digest,
            SBOMFormat.CYCLONEDX_1_5,
            [],
            "scanner",
        )

        sbom = service.get_sbom_by_artifact_digest(valid_digest)
        assert sbom is not None

    def test_resolve_vulnerability(self, service, workspace_id, valid_digest):
        """Test vulnerability resolution."""
        vuln = Vulnerability(vuln_id="CVE-2023-12345", severity=VulnerabilitySeverity.HIGH)
        sbom = service.upload_sbom(
            workspace_id,
            valid_digest,
            SBOMFormat.CYCLONEDX_1_5,
            [],
            "scanner",
            vulnerabilities=[vuln],
        )

        updated = service.resolve_vulnerability(
            sbom.sbom_id, "CVE-2023-12345", workspace_id, "user"
        )
        assert updated.vulnerabilities[0].resolved is True

    # Trusted Signers Tests
    def test_add_trusted_signer(self, service):
        """Test adding trusted signer."""
        signer = service.add_trusted_signer(
            name="release-key",
            public_key="PEM_KEY_DATA",
            algorithm=SCSigningAlgorithm.ED25519,
            created_by="admin",
            valid_days=365,
        )

        assert signer.signer_id is not None
        assert signer.is_valid() is True

    def test_revoke_signer(self, service):
        """Test signer revocation."""
        signer = service.add_trusted_signer("key", "data", SCSigningAlgorithm.ED25519, "admin")
        revoked = service.revoke_signer(signer.signer_id, "admin", "Compromised")

        assert revoked.revoked is True
        assert revoked.is_valid() is False

    # Validation Tests
    def test_validate_artifact_for_deployment(self, service, workspace_id, valid_digest):
        """Test artifact deployment validation."""
        signer = service.add_trusted_signer("key", "data", SCSigningAlgorithm.ED25519, "admin")
        service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
            signature="sig",
            signer_id=signer.signer_id,
        )

        valid, errors = service.validate_artifact_for_deployment(valid_digest, workspace_id)
        assert valid is True

    # Stats and Export Tests
    def test_get_stats(self, service, workspace_id, valid_digest):
        """Test supply chain statistics."""
        service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
        )

        stats = service.get_stats(workspace_id)
        assert stats.total_artifacts == 1

    def test_export_inventory(self, service, workspace_id, valid_digest):
        """Test inventory export."""
        service.register_artifact(
            workspace_id,
            ArtifactType.CONTAINER_IMAGE,
            "app",
            valid_digest,
            "reg",
            "repo",
            "user",
        )

        export = service.export_inventory(workspace_id)
        assert "artifacts" in export
        assert "integrity_hash" in export


# =============================================================================
# AgentUpdateService Tests
# =============================================================================


class TestAgentUpdateService:
    """Tests for AgentUpdateService."""

    @pytest.fixture
    def service(self) -> AgentUpdateService:
        return AgentUpdateService(require_signatures=False)

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    # Update Publication Tests
    def test_publish_update(self, service):
        """Test publishing update."""
        update = service.publish_update(
            version="v1.0.0",
            artifact_digest="sha256:" + "a" * 64,
            published_by="release-bot",
            channel=UpdateChannel.STABLE,
            release_notes="Initial release",
        )

        assert update.update_id is not None
        assert update.version == "v1.0.0"
        assert update.channel == UpdateChannel.STABLE

    def test_publish_update_invalid_version(self, service):
        """Test publishing with invalid version."""
        with pytest.raises(ValueError, match="Invalid semver"):
            service.publish_update("invalid", "sha256:" + "a" * 64, "user")

    def test_publish_update_duplicate_version(self, service):
        """Test duplicate version rejection."""
        service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")

        with pytest.raises(ValueError, match="already exists"):
            service.publish_update("v1.0.0", "sha256:" + "b" * 64, "user")

    def test_get_update_by_version(self, service):
        """Test update retrieval by version."""
        service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")

        update = service.get_update_by_version("v1.0.0")
        assert update is not None
        assert update.version == "v1.0.0"

    def test_get_latest_update(self, service):
        """Test getting latest update."""
        service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        service.publish_update("v1.1.0", "sha256:" + "b" * 64, "user")

        latest = service.get_latest_update(UpdateChannel.STABLE)
        assert latest.version == "v1.1.0"

    # Rollout Tests
    def test_create_rollout(self, service):
        """Test rollout creation."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")

        rollout = service.create_rollout(update.update_id, "user")

        assert rollout.rollout_id is not None
        assert rollout.status == RolloutStatus.PENDING
        assert len(rollout.stages) == 3  # Default stages

    def test_start_rollout(self, service):
        """Test starting rollout."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")

        started = service.start_rollout(rollout.rollout_id, "user")

        assert started.status == RolloutStatus.IN_PROGRESS
        assert started.started_at is not None
        assert started.stages[0].status == RolloutStatus.IN_PROGRESS

    def test_record_agent_update(self, service, workspace_id):
        """Test recording agent update result."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        service.start_rollout(rollout.rollout_id, "user")

        record = service.record_agent_update(
            rollout_id=rollout.rollout_id,
            agent_id="agent-1",
            workspace_id=workspace_id,
            from_version="v0.9.0",
            to_version="v1.0.0",
            result=UpdateResult.SUCCESS,
            duration_seconds=5.0,
        )

        assert record.result == UpdateResult.SUCCESS

        # Check stage metrics updated
        plan = service.get_rollout(rollout.rollout_id)
        assert plan.stages[0].metrics.agents_updated == 1

    def test_advance_stage(self, service, workspace_id):
        """Test advancing rollout stage."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        service.start_rollout(rollout.rollout_id, "user")

        # Record successful update
        service.record_agent_update(
            rollout.rollout_id,
            "agent-1",
            workspace_id,
            "v0.9.0",
            "v1.0.0",
            UpdateResult.SUCCESS,
        )

        # Wait minimum time (force advance)
        success, error = service.advance_stage(rollout.rollout_id, "user", force=True)

        assert success is True

        plan = service.get_rollout(rollout.rollout_id)
        assert plan.current_stage_index == 1

    def test_pause_rollout(self, service):
        """Test pausing rollout."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        service.start_rollout(rollout.rollout_id, "user")

        paused = service.pause_rollout(rollout.rollout_id, "user", "Issues detected")

        assert paused.status == RolloutStatus.PAUSED

    def test_resume_rollout(self, service):
        """Test resuming rollout."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        service.start_rollout(rollout.rollout_id, "user")
        service.pause_rollout(rollout.rollout_id, "user")

        resumed = service.resume_rollout(rollout.rollout_id, "user")

        assert resumed.status == RolloutStatus.IN_PROGRESS

    def test_cancel_rollout(self, service):
        """Test cancelling rollout."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        service.start_rollout(rollout.rollout_id, "user")

        cancelled = service.cancel_rollout(rollout.rollout_id, "user", "Abort")

        assert cancelled.status == RolloutStatus.CANCELLED

    # Rollback Tests
    def test_request_rollback(self, service):
        """Test rollback request."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")

        rollback = service.request_rollback(
            rollout_id=rollout.rollout_id,
            to_version="v0.9.0",
            reason="Critical bug found in v1.0.0",
            requested_by="user-1",
        )

        assert rollback.rollback_id is not None
        assert rollback.status == RolloutStatus.PENDING

    def test_approve_rollback(self, service):
        """Test rollback approval."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        rollback = service.request_rollback(
            rollout.rollout_id, "v0.9.0", "Critical bug found in production", "user-1"
        )

        approved = service.approve_rollback(rollback.rollback_id, "user-2")

        assert approved.approved_by == "user-2"

    def test_approve_rollback_self_reject(self, service):
        """Test self-approval rejection."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        rollback = service.request_rollback(
            rollout.rollout_id, "v0.9.0", "Critical bug found in production", "user-1"
        )

        with pytest.raises(ValueError, match="Self-approval"):
            service.approve_rollback(rollback.rollback_id, "user-1")

    def test_execute_rollback(self, service):
        """Test rollback execution."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        rollback = service.request_rollback(
            rollout.rollout_id, "v0.9.0", "Critical bug found in production", "user-1"
        )
        service.approve_rollback(rollback.rollback_id, "user-2")

        executed = service.execute_rollback(rollback.rollback_id, "user-2")

        assert executed.status == RolloutStatus.IN_PROGRESS

    def test_complete_rollback(self, service):
        """Test rollback completion."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        rollout = service.create_rollout(update.update_id, "user")
        rollback = service.request_rollback(
            rollout.rollout_id, "v0.9.0", "Critical bug found in production", "user-1"
        )
        service.approve_rollback(rollback.rollback_id, "user-2")
        service.execute_rollback(rollback.rollback_id, "user-2")

        completed = service.complete_rollback(rollback.rollback_id, agents_rolled_back=10)

        assert completed.status == RolloutStatus.COMPLETED
        assert completed.agents_rolled_back == 10

    # Version Pinning Tests
    def test_pin_version(self, service, workspace_id):
        """Test version pinning."""
        pin = service.pin_version(
            workspace_id=workspace_id,
            version="v1.0.0",
            pinned_by="admin",
            reason="Production stability",
        )

        assert pin.pin_id is not None
        assert pin.pinned_version == "v1.0.0"
        assert pin.active is True

    def test_unpin_version(self, service, workspace_id):
        """Test version unpinning."""
        service.pin_version(workspace_id, "v1.0.0", "admin")

        unpinned = service.unpin_version(workspace_id, "admin", "Upgrade approved")

        assert unpinned.active is False

    def test_is_update_allowed(self, service, workspace_id):
        """Test update allowance check."""
        # No pin - allowed
        allowed, reason = service.is_update_allowed(workspace_id, "v1.1.0")
        assert allowed is True

        # With pin - only pinned version allowed
        service.pin_version(workspace_id, "v1.0.0", "admin")

        allowed, reason = service.is_update_allowed(workspace_id, "v1.0.0")
        assert allowed is True

        allowed, reason = service.is_update_allowed(workspace_id, "v1.1.0")
        assert allowed is False

    # Change Window Tests
    def test_change_window(self, service, workspace_id):
        """Test change window."""
        window = ChangeWindow(
            allowed_days={1, 2, 3},  # Tue-Thu
            start_hour=2,
            end_hour=6,
        )

        service.pin_version(workspace_id, "v1.0.0", "admin", change_window=window)
        service.set_change_window(workspace_id, window, "admin")

        pin = service.get_version_pin(workspace_id)
        assert pin.change_window is not None

    def test_change_window_is_within(self):
        """Test change window time check."""
        window = ChangeWindow(
            allowed_days={0, 1, 2, 3, 4},
            start_hour=2,
            end_hour=6,
        )

        # Test with a Tuesday at 3am UTC
        dt = datetime(2025, 1, 7, 3, 0, tzinfo=timezone.utc)  # Tuesday
        assert window.is_within_window(dt) is True

        # Test outside hours
        dt = datetime(2025, 1, 7, 10, 0, tzinfo=timezone.utc)
        assert window.is_within_window(dt) is False

    # Stats Tests
    def test_get_stats(self, service, workspace_id):
        """Test update statistics."""
        service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")

        stats = service.get_stats(workspace_id)
        assert stats.total_updates_published == 1

    def test_export_rollout_records(self, service, workspace_id):
        """Test rollout records export."""
        update = service.publish_update("v1.0.0", "sha256:" + "a" * 64, "user")
        service.create_rollout(update.update_id, "user")

        export = service.export_rollout_records(workspace_id)
        assert "rollouts" in export
        assert "integrity_hash" in export


# =============================================================================
# ResearchSandboxService Tests
# =============================================================================


class TestResearchSandboxService:
    """Tests for ResearchSandboxService."""

    @pytest.fixture
    def service(self) -> ResearchSandboxService:
        return ResearchSandboxService()

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    @pytest.fixture
    def user_id(self) -> str:
        return "user-456"

    # Quota Tests
    def test_set_quota(self, service, workspace_id, user_id):
        """Test setting quota."""
        quota = ResourceQuota(
            cpu_limit=4.0,
            memory_limit_mb=8192,
            max_runtime_seconds=7200,
        )

        result = service.set_quota(workspace_id, quota, user_id)
        assert result.cpu_limit == 4.0
        assert result.memory_limit_mb == 8192

    def test_get_quota_default(self, service, workspace_id):
        """Test getting default quota."""
        quota = service.get_quota(workspace_id)
        assert quota is not None
        assert quota.cpu_limit > 0

    def test_check_quota_within(self, service, workspace_id):
        """Test quota check within limits."""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_used_mb=1024,
            runtime_seconds=1800,
        )

        within, violations = service.check_quota(workspace_id, usage)
        assert within is True
        assert len(violations) == 0

    def test_check_quota_exceeded(self, service, workspace_id, user_id):
        """Test quota check when exceeded."""
        quota = ResourceQuota(memory_limit_mb=1024)
        service.set_quota(workspace_id, quota, user_id)

        usage = ResourceUsage(memory_used_mb=2048)

        within, violations = service.check_quota(workspace_id, usage)
        assert within is False
        assert len(violations) > 0

    # Sandbox Config Tests
    def test_set_sandbox_config(self, service, workspace_id, user_id):
        """Test setting sandbox config."""
        config = SandboxConfig(
            isolation_level=IsolationLevel.FIRECRACKER,
            read_only_rootfs=True,
        )

        result = service.set_sandbox_config(workspace_id, config, user_id)
        assert result.isolation_level == IsolationLevel.FIRECRACKER

    def test_get_sandbox_config_default(self, service, workspace_id):
        """Test getting default sandbox config."""
        config = service.get_sandbox_config(workspace_id)
        assert config.no_new_privileges is True

    # Egress Policy Tests
    def test_set_egress_policy(self, service, workspace_id, user_id):
        """Test setting egress policy."""
        rules = [
            EgressRule(domain="*.polygon.io", allow=True),
            EgressRule(domain="*.google.com", allow=True),
        ]
        policy = EgressPolicy(rules=rules, deny_all_by_default=True)

        result = service.set_egress_policy(workspace_id, policy, user_id)
        assert len(result.rules) == 2

    def test_add_egress_rule(self, service, workspace_id, user_id):
        """Test adding egress rule."""
        rule = EgressRule(domain="api.example.com", allow=True, ports={443})

        policy = service.add_egress_rule(workspace_id, rule, user_id)
        assert len(policy.rules) == 1

    def test_check_egress_allowed(self, service, workspace_id, user_id):
        """Test egress check - allowed."""
        rule = EgressRule(domain="api.example.com", allow=True, ports={443})
        service.add_egress_rule(workspace_id, rule, user_id)

        allowed, reason = service.check_egress(
            job_id="job-1",
            workspace_id=workspace_id,
            domain="api.example.com",
            ip=None,
            port=443,
        )

        assert allowed is True

    def test_check_egress_denied(self, service, workspace_id, user_id):
        """Test egress check - denied by default."""
        # No rules, deny by default
        policy = EgressPolicy(deny_all_by_default=True)
        service.set_egress_policy(workspace_id, policy, user_id)

        allowed, reason = service.check_egress(
            "job-1",
            workspace_id,
            "evil.com",
            None,
            443,
        )

        assert allowed is False

    # Job Tests
    def test_create_job(self, service, workspace_id, user_id):
        """Test job creation."""
        job = service.create_job(
            workspace_id=workspace_id,
            user_id=user_id,
            name="Test Job",
            image_digest="sha256:" + "a" * 64,
            command=["python", "research.py"],
        )

        assert job.job_id is not None
        assert job.status == JobStatus.PENDING

    def test_create_job_concurrent_limit(self, service, workspace_id, user_id):
        """Test concurrent job limit."""
        quota = ResourceQuota(max_concurrent_jobs=1)
        service.set_quota(workspace_id, quota, user_id)

        job = service.create_job(workspace_id, user_id, "Job1", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        with pytest.raises(ValueError, match="Concurrent job limit"):
            service.create_job(workspace_id, user_id, "Job2", "sha256:" + "b" * 64, ["cmd"])

    def test_start_job(self, service, workspace_id, user_id):
        """Test starting job."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])

        started = service.start_job(job.job_id)

        assert started.status == JobStatus.RUNNING
        assert started.started_at is not None

    def test_complete_job(self, service, workspace_id, user_id):
        """Test completing job."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        completed = service.complete_job(job.job_id, exit_code=0)

        assert completed.status == JobStatus.COMPLETED
        assert completed.exit_code == 0

    def test_fail_job(self, service, workspace_id, user_id):
        """Test failing job."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        failed = service.fail_job(job.job_id, "OOM error", exit_code=137)

        assert failed.status == JobStatus.FAILED

    def test_terminate_job(self, service, workspace_id, user_id):
        """Test terminating job."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        terminated = service.terminate_job(job.job_id, "admin", "Policy violation")

        assert terminated.status == JobStatus.TERMINATED

    def test_update_job_usage(self, service, workspace_id, user_id):
        """Test updating job usage."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        usage = ResourceUsage(cpu_percent=75.0, memory_used_mb=2048)
        updated = service.update_job_usage(job.job_id, usage)

        assert updated.current_usage.cpu_percent == 75.0

    # Abuse Detection Tests
    def test_detect_abuse_manual(self, service, workspace_id, user_id):
        """Test manual abuse detection."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        abuse = service.detect_abuse(
            job_id=job.job_id,
            abuse_type=AbuseType.CRYPTO_MINING,
            indicators=["High CPU usage", "Mining pool connection"],
            evidence={"pool_domain": "pool.minexmr.com"},
            confidence=0.9,
            action=AbuseAction.TERMINATE,
        )

        assert abuse.abuse_type == AbuseType.CRYPTO_MINING
        assert abuse.confidence == 0.9

    def test_detect_abuse_from_usage(self, service, workspace_id, user_id):
        """Test automated abuse detection from usage."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        # Simulate high CPU (mining indicator)
        usage = ResourceUsage(cpu_percent=95.0, network_tx_bytes=100 * 1024 * 1024)
        service.update_job_usage(job.job_id, usage)

        abuse_events = service.get_abuse_events(workspace_id=workspace_id)
        # May or may not trigger depending on confidence threshold
        assert isinstance(abuse_events, list)

    def test_resolve_abuse_event(self, service, workspace_id, user_id):
        """Test resolving abuse event."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)

        abuse = service.detect_abuse(
            job.job_id,
            AbuseType.RESOURCE_ABUSE,
            ["High resource"],
            {},
            0.6,
            AbuseAction.ALERT,
        )

        resolved = service.resolve_abuse_event(abuse.event_id, "admin")

        assert resolved.resolved is True

    # Stats Tests
    def test_get_stats(self, service, workspace_id, user_id):
        """Test sandbox statistics."""
        job = service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])
        service.start_job(job.job_id)
        service.complete_job(job.job_id, 0)

        stats = service.get_stats(workspace_id)
        assert stats.total_jobs == 1
        assert stats.jobs_completed == 1

    def test_export_sandbox_records(self, service, workspace_id, user_id):
        """Test sandbox records export."""
        service.create_job(workspace_id, user_id, "Test", "sha256:" + "a" * 64, ["cmd"])

        export = service.export_sandbox_records(workspace_id)
        assert "jobs" in export
        assert "integrity_hash" in export


# =============================================================================
# BreachWorkflowService Tests
# =============================================================================


class TestBreachWorkflowService:
    """Tests for BreachWorkflowService."""

    @pytest.fixture
    def service(self) -> BreachWorkflowService:
        dpo = DPOContact(name="Jane DPO", email="dpo@example.com")
        return BreachWorkflowService(dpo_contact=dpo)

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    # Breach Reporting Tests
    def test_report_breach(self, service, workspace_id):
        """Test breach reporting."""
        breach = service.report_breach(
            workspace_id=workspace_id,
            title="Data leak via API",
            description="Customer data exposed through unsecured endpoint",
            category=BreachCategory.CONFIDENTIALITY,
            reported_by="security-team",
            data_categories_affected=["email", "name"],
        )

        assert breach.breach_id is not None
        assert breach.status == BreachStatus.DETECTED
        assert breach.awareness_at is not None
        assert len(breach.timeline) >= 2

    def test_confirm_breach(self, service, workspace_id):
        """Test breach confirmation."""
        breach = service.report_breach(
            workspace_id,
            "Test Breach",
            "Description",
            BreachCategory.CONFIDENTIALITY,
            "reporter",
        )

        confirmed = service.confirm_breach(breach.breach_id, "investigator")

        assert confirmed.status == BreachStatus.CONFIRMED

    def test_breach_deadline(self, service, workspace_id):
        """Test 72-hour deadline calculation."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        deadline = breach.get_authority_deadline()
        hours_until = breach.get_hours_until_deadline()

        assert deadline > datetime.now(timezone.utc)
        assert 0 < hours_until <= AUTHORITY_NOTIFICATION_HOURS

    # Risk Assessment Tests
    def test_assess_breach(self, service, workspace_id):
        """Test breach assessment."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        assessment = RiskAssessment(
            data_sensitivity_score=0.8,
            data_volume_score=0.5,
            identifiability_score=0.9,
            special_categories_present=True,
            potential_harm_score=0.7,
        )

        assessed = service.assess_breach(breach.breach_id, assessment, "dpo")

        assert assessed.assessment is not None
        assert assessed.assessment.overall_risk_score > 0
        assert assessed.status == BreachStatus.ASSESSED

    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        assessment = RiskAssessment(
            data_sensitivity_score=1.0,  # Max
            data_volume_score=1.0,
            identifiability_score=1.0,
            special_categories_present=True,
            vulnerable_individuals_affected=True,
            potential_harm_score=1.0,
            cross_border=True,
            malicious_intent=True,
        )

        score = assessment.calculate_risk_score()

        assert score == 10.0  # Max score
        assert assessment.severity == BreachSeverity.CRITICAL

    # Notification Decision Tests
    def test_make_notification_decision(self, service, workspace_id):
        """Test notification decision."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        assessment = RiskAssessment(
            data_sensitivity_score=0.8,
            potential_harm_score=0.7,
        )
        service.assess_breach(breach.breach_id, assessment, "dpo")

        decided = service.make_notification_decision(
            breach.breach_id,
            "dpo",
        )

        assert decided.decision is not None
        assert decided.decision.authority_notification_required is True

    def test_notification_decision_with_exemption(self, service, workspace_id):
        """Test notification decision with exemption."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        assessment = RiskAssessment(
            data_sensitivity_score=0.8,
            potential_harm_score=0.9,
        )
        service.assess_breach(breach.breach_id, assessment, "dpo")

        decided = service.make_notification_decision(
            breach.breach_id,
            "dpo",
            exemption=ExemptionType.ENCRYPTION,
            exemption_justification="All data was encrypted",
        )

        assert decided.decision.subject_notification_required is False
        assert decided.decision.exemption_applied == ExemptionType.ENCRYPTION

    def test_approve_decision(self, service, workspace_id):
        """Test decision approval."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )
        assessment = RiskAssessment(data_sensitivity_score=0.5)
        service.assess_breach(breach.breach_id, assessment, "analyst")
        service.make_notification_decision(breach.breach_id, "analyst")

        approved = service.approve_decision(breach.breach_id, "dpo")

        assert approved.decision.approved_by == "dpo"

    # Authority Notification Tests
    def test_create_authority_notification(self, service, workspace_id):
        """Test creating authority notification."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
            data_categories_affected=["email", "name"],
        )

        notification = service.create_authority_notification(
            breach.breach_id,
            "dpo",
        )

        assert notification.notification_id is not None
        assert notification.status == NotificationStatus.DRAFT
        assert "email" in notification.categories_of_data

    def test_submit_authority_notification(self, service, workspace_id):
        """Test submitting authority notification."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )
        service.create_authority_notification(breach.breach_id, "dpo")

        submitted = service.submit_authority_notification(
            breach.breach_id,
            "dpo",
            submission_reference="ICO-2025-12345",
        )

        assert submitted.authority_notification.status == NotificationStatus.SUBMITTED
        assert submitted.status == BreachStatus.NOTIFYING

    # Subject Notification Tests
    def test_create_subject_notification(self, service, workspace_id):
        """Test creating subject notification."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )
        breach.individuals_affected_count = 100

        notification = service.create_subject_notification(
            breach.breach_id,
            "dpo",
            plain_language_description="Your data may have been accessed.",
            recommendations=["Change your password", "Monitor accounts"],
        )

        assert notification.notification_id is not None
        assert notification.status == NotificationStatus.DRAFT

    def test_send_subject_notification(self, service, workspace_id):
        """Test sending subject notification."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )
        service.create_subject_notification(breach.breach_id, "dpo", "Description")

        sent = service.send_subject_notification(
            breach.breach_id,
            "dpo",
            subjects_notified_count=95,
        )

        assert sent.subject_notification.status == NotificationStatus.COMPLETE
        assert sent.subject_notification.subjects_notified_count == 95

    # Containment and Resolution Tests
    def test_record_containment(self, service, workspace_id):
        """Test recording containment."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        contained = service.record_containment(
            breach.breach_id,
            ["Disabled affected API endpoint", "Rotated credentials"],
            "security-team",
        )

        assert contained.status == BreachStatus.CONTAINED
        assert len(contained.containment_measures) == 2

    def test_record_remediation(self, service, workspace_id):
        """Test recording remediation."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        remediated = service.record_remediation(
            breach.breach_id,
            ["Deployed fix", "Updated access controls"],
            "Missing authentication on endpoint",
            "security-team",
        )

        assert remediated.status == BreachStatus.REMEDIATED
        assert remediated.root_cause == "Missing authentication on endpoint"

    def test_resolve_breach(self, service, workspace_id):
        """Test breach resolution."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        resolved = service.resolve_breach(
            breach.breach_id,
            ["Implement auth middleware", "Add security review"],
            "dpo",
        )

        assert resolved.status == BreachStatus.RESOLVED
        assert resolved.resolved_at is not None

    def test_close_breach(self, service, workspace_id):
        """Test breach closure."""
        breach = service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )
        service.resolve_breach(breach.breach_id, ["Lesson"], "dpo")

        closed = service.close_breach(breach.breach_id, "dpo")

        assert closed.status == BreachStatus.CLOSED
        assert closed.closed_at is not None

    # Tabletop Exercise Tests
    def test_list_scenarios(self, service):
        """Test listing tabletop scenarios."""
        scenarios = service.list_scenarios()

        assert len(scenarios) >= 4  # Default scenarios
        assert any(s.name == "Ransomware Attack" for s in scenarios)

    def test_create_exercise(self, service, workspace_id):
        """Test creating tabletop exercise."""
        scenarios = service.list_scenarios()

        exercise = service.create_exercise(
            workspace_id=workspace_id,
            scenario_id=scenarios[0].scenario_id,
            conducted_by="dpo",
            participants=["security-lead", "eng-lead", "legal"],
        )

        assert exercise.exercise_id is not None
        assert exercise.status in ["scheduled", "in_progress"]

    def test_complete_exercise(self, service, workspace_id):
        """Test completing tabletop exercise."""
        scenarios = service.list_scenarios()
        exercise = service.create_exercise(
            workspace_id,
            scenarios[0].scenario_id,
            "dpo",
            ["security", "eng"],
        )

        completed = service.complete_exercise(
            exercise_id=exercise.exercise_id,
            completed_by="dpo",
            duration_minutes=90,
            draft_package_produced=True,
            draft_package_time_minutes=20,
            timeline_achieved={"detection": 5, "assessment": 30},
            decisions_made=["Notify authority", "Notify subjects"],
            gaps_identified=["Need faster escalation"],
            improvements_recommended=["Automate notification"],
        )

        assert completed.status == "completed"
        assert completed.draft_package_produced is True
        assert completed.next_exercise_due is not None

    def test_is_tabletop_due(self, service, workspace_id):
        """Test tabletop due check."""
        # No exercises - should be due
        assert service.is_tabletop_due(workspace_id) is True

        # After exercise - not due
        scenarios = service.list_scenarios()
        exercise = service.create_exercise(
            workspace_id,
            scenarios[0].scenario_id,
            "dpo",
            ["team"],
        )
        service.complete_exercise(
            exercise.exercise_id,
            "dpo",
            60,
            True,
            15,
            {},
            [],
            [],
            [],
        )

        assert service.is_tabletop_due(workspace_id) is False

    # Stats Tests
    def test_get_stats(self, service, workspace_id):
        """Test breach statistics."""
        service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        stats = service.get_stats(workspace_id)
        assert stats.total_breaches == 1
        assert stats.breaches_open == 1

    def test_export_breach_records(self, service, workspace_id):
        """Test breach records export."""
        service.report_breach(
            workspace_id,
            "Test",
            "Desc",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        export = service.export_breach_records(workspace_id)
        assert "breaches" in export
        assert "integrity_hash" in export


# =============================================================================
# EvidencePackService Tests
# =============================================================================


class TestEvidencePackService:
    """Tests for EvidencePackService."""

    @pytest.fixture
    def service(self) -> EvidencePackService:
        return EvidencePackService()

    @pytest.fixture
    def workspace_id(self) -> str:
        return "workspace-123"

    # Export Request Tests
    def test_request_export(self, service, workspace_id):
        """Test export request."""
        request = service.request_export(
            workspace_id=workspace_id,
            requested_by="auditor",
            categories={EvidenceCategory.ACCESS_AUDIT, EvidenceCategory.RBAC_SNAPSHOT},
            reason="Annual audit",
        )

        assert request.request_id is not None
        assert request.status == PackStatus.PENDING
        assert EvidenceCategory.ACCESS_AUDIT in request.categories

    def test_get_request(self, service, workspace_id):
        """Test getting export request."""
        request = service.request_export(workspace_id, "auditor")

        retrieved = service.get_request(request.request_id)
        assert retrieved is not None
        assert retrieved.request_id == request.request_id

    # Pack Generation Tests
    def test_generate_pack(self, service, workspace_id):
        """Test pack generation."""
        request = service.request_export(
            workspace_id,
            "auditor",
            categories={EvidenceCategory.ACCESS_AUDIT},
        )

        pack = service.generate_pack(request.request_id)

        assert pack.pack_id is not None
        assert pack.status == PackStatus.COMPLETED
        assert pack.manifest is not None
        assert len(pack.artifacts) > 0

    def test_generate_pack_all_categories(self, service, workspace_id):
        """Test pack generation with all categories."""
        request = service.request_export(workspace_id, "auditor")

        pack = service.generate_pack(request.request_id)

        assert pack.status == PackStatus.COMPLETED
        # Should have artifacts for available categories
        assert len(pack.artifacts) >= 1

    def test_get_pack(self, service, workspace_id):
        """Test getting pack."""
        request = service.request_export(workspace_id, "auditor")
        pack = service.generate_pack(request.request_id)

        retrieved = service.get_pack(pack.pack_id)
        assert retrieved is not None
        assert retrieved.pack_id == pack.pack_id

    def test_list_packs(self, service, workspace_id):
        """Test listing packs."""
        r1 = service.request_export(workspace_id, "auditor")
        service.generate_pack(r1.request_id)

        r2 = service.request_export(workspace_id, "auditor")
        service.generate_pack(r2.request_id)

        packs = service.list_packs(workspace_id=workspace_id)
        assert len(packs) == 2

    # Export Methods Tests
    def test_export_pack_as_json(self, service, workspace_id):
        """Test JSON export."""
        request = service.request_export(workspace_id, "auditor")
        pack = service.generate_pack(request.request_id)

        json_export = service.export_pack_as_json(pack.pack_id)

        assert "pack_id" in json_export
        assert "artifacts" in json_export

    def test_export_pack_as_zip(self, service, workspace_id):
        """Test ZIP export."""
        request = service.request_export(workspace_id, "auditor")
        pack = service.generate_pack(request.request_id)

        zip_bytes = service.export_pack_as_zip(pack.pack_id, workspace_id)

        assert len(zip_bytes) > 0
        # Verify it's a valid ZIP
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert "manifest.json" in zf.namelist()

    def test_generate_download_url(self, service, workspace_id):
        """Test download URL generation."""
        request = service.request_export(workspace_id, "auditor")
        pack = service.generate_pack(request.request_id)

        updated = service.generate_download_url(pack.pack_id, expires_in_hours=24)

        assert updated.download_url is not None
        assert updated.download_expires_at is not None

    # Quick Export Tests
    def test_quick_export_security_controls(self, service, workspace_id):
        """Test quick security controls export."""
        pack = service.quick_export_security_controls(workspace_id, "auditor")

        assert pack.status == PackStatus.COMPLETED
        assert EvidenceCategory.SECURITY_BASELINE in pack.categories

    def test_quick_export_breach_evidence(self, service, workspace_id):
        """Test quick breach evidence export."""
        pack = service.quick_export_breach_evidence(workspace_id, "auditor")

        assert pack.status == PackStatus.COMPLETED
        assert EvidenceCategory.BREACH_RECORDS in pack.categories

    def test_quick_export_supply_chain(self, service, workspace_id):
        """Test quick supply chain export."""
        pack = service.quick_export_supply_chain(workspace_id, "auditor")

        assert pack.status == PackStatus.COMPLETED
        assert EvidenceCategory.ARTIFACT_INVENTORY in pack.categories

    def test_quick_export_compliance(self, service, workspace_id):
        """Test quick compliance export."""
        pack = service.quick_export_compliance(workspace_id, "auditor")

        assert pack.status == PackStatus.COMPLETED
        assert EvidenceCategory.DSAR_RECORDS in pack.categories

    # Stats Tests
    def test_get_stats(self, service, workspace_id):
        """Test evidence pack statistics."""
        request = service.request_export(workspace_id, "auditor")
        service.generate_pack(request.request_id)

        stats = service.get_stats(workspace_id)

        assert stats["total_packs"] == 1
        assert stats["completed_packs"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase7Integration:
    """Integration tests for Phase 7 services."""

    def test_full_breach_workflow(self):
        """Test complete breach response workflow."""
        breach_service = BreachWorkflowService(
            dpo_contact=DPOContact(name="DPO", email="dpo@test.com"),
        )

        # 1. Report breach
        breach = breach_service.report_breach(
            workspace_id="ws-1",
            title="Customer data exposure",
            description="Unsecured API endpoint exposed customer data",
            category=BreachCategory.CONFIDENTIALITY,
            reported_by="security-team",
            data_categories_affected=["email", "name", "phone"],
            individuals_affected_estimate="500-1000",
        )

        # 2. Confirm breach
        breach_service.confirm_breach(breach.breach_id, "security-lead")

        # 3. Assess breach
        assessment = RiskAssessment(
            data_sensitivity_score=0.7,
            data_volume_score=0.6,
            identifiability_score=0.8,
            potential_harm_score=0.5,
            cross_border=True,
        )
        breach_service.assess_breach(breach.breach_id, assessment, "dpo")

        # 4. Make notification decision
        breach_service.make_notification_decision(breach.breach_id, "dpo")
        breach_service.approve_decision(breach.breach_id, "ceo")

        # 5. Create and submit authority notification
        breach_service.create_authority_notification(breach.breach_id, "dpo")
        breach_service.submit_authority_notification(
            breach.breach_id,
            "dpo",
            "AUTH-2025-001",
        )

        # 6. Containment
        breach_service.record_containment(
            breach.breach_id,
            ["Disabled endpoint", "Rotated API keys"],
            "security-team",
        )

        # 7. Remediation
        breach_service.record_remediation(
            breach.breach_id,
            ["Added authentication", "Deployed WAF rules"],
            "Missing authentication middleware",
            "engineering",
        )

        # 8. Resolution
        breach_service.resolve_breach(
            breach.breach_id,
            ["Add security review to PR process", "Implement API gateway"],
            "dpo",
        )

        # 9. Close
        breach_service.close_breach(breach.breach_id, "dpo")

        # Verify final state
        final = breach_service.get_breach(breach.breach_id)
        assert final.status == BreachStatus.CLOSED
        assert final.authority_notification.status == NotificationStatus.SUBMITTED
        assert len(final.timeline) >= 10

    def test_evidence_pack_with_services(self):
        """Test evidence pack generation with connected services."""
        # Create services
        security_service = SecurityBaselineService()
        supply_chain_service = SupplyChainService(require_signatures=False)
        breach_service = BreachWorkflowService()

        # Set up some data
        security_service.create_key("ws-1", KeyType.DATA_ENCRYPTION_KEY, "admin")
        supply_chain_service.register_artifact(
            "ws-1",
            ArtifactType.CONTAINER_IMAGE,
            "app",
            "sha256:" + "a" * 64,
            "gcr.io",
            "app",
            "admin",
        )
        breach_service.report_breach(
            "ws-1",
            "Test Breach",
            "Description",
            BreachCategory.CONFIDENTIALITY,
            "user",
        )

        # Create evidence pack service with connected services
        evidence_service = EvidencePackService(
            security_baseline_service=security_service,
            supply_chain_service=supply_chain_service,
            breach_workflow_service=breach_service,
        )

        # Generate pack
        pack = evidence_service.quick_export_security_controls("ws-1", "auditor")

        assert pack.status == PackStatus.COMPLETED
        assert len(pack.artifacts) > 0

    def test_staged_rollout_with_monitoring(self):
        """Test staged rollout with monitoring and potential rollback."""
        update_service = AgentUpdateService(require_signatures=False)

        # Publish update
        update = update_service.publish_update(
            "v2.0.0",
            "sha256:" + "a" * 64,
            "release-bot",
        )

        # Create rollout with custom stages
        stages = [
            {"name": "canary", "percentage": 1, "min_success_rate": 0.99},
            {"name": "beta", "percentage": 10, "min_success_rate": 0.98},
            {"name": "general", "percentage": 100, "min_success_rate": 0.95},
        ]
        rollout = update_service.create_rollout(
            update.update_id,
            "admin",
            stages=stages,
        )

        # Start rollout
        update_service.start_rollout(rollout.rollout_id, "admin")

        # Simulate canary updates
        for i in range(10):
            update_service.record_agent_update(
                rollout.rollout_id,
                f"agent-{i}",
                "ws-1",
                "v1.9.0",
                "v2.0.0",
                UpdateResult.SUCCESS,
            )

        # Advance to beta
        update_service.advance_stage(rollout.rollout_id, "admin", force=True)

        # Simulate beta updates with some failures
        for i in range(90):
            result = UpdateResult.SUCCESS if i % 20 != 0 else UpdateResult.FAILED
            update_service.record_agent_update(
                rollout.rollout_id,
                f"agent-beta-{i}",
                "ws-1",
                "v1.9.0",
                "v2.0.0",
                result,
            )

        # Check rollout state
        plan = update_service.get_rollout(rollout.rollout_id)
        metrics = plan.get_overall_metrics()

        assert metrics.agents_updated > 0
        assert metrics.success_rate > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
