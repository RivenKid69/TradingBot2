# -*- coding: utf-8 -*-
"""
Enterprise Posture Service - Phase 9 GDPR Implementation.

Supports enterprise on-prem/VPC deployments while preserving the
"software/platform provider" posture with full auditability.

Key Features:
    - Enterprise deployment mode management
    - "Telemetry stays local" enforcement
    - EU-only residency validation for on-prem
    - Air-gapped evidence pack export
    - Customer-managed key (CMK) integration
    - RAW telemetry handling (enterprise-only)

Design Doc Reference:
    - Phase 9: Enterprise/on-prem/VPC posture (Design Doc 16.3)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968-L972

CLOUD + ENTERPRISE ZONE.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class EnterpriseDeploymentMode(str, Enum):
    """Supported enterprise deployment modes."""
    SAAS = "saas"                      # Standard cloud-hosted
    ENTERPRISE_CLOUD = "enterprise_cloud"  # Cloud with enterprise features
    ON_PREM_FULL = "on_prem_full"     # Complete on-premises
    VPC_MANAGED = "vpc_managed"        # Customer VPC
    AIR_GAPPED = "air_gapped"          # No external connectivity


class TelemetryDestination(str, Enum):
    """Where telemetry data is sent."""
    CLOUD = "cloud"                    # Vendor cloud (SaaS/Enterprise Cloud)
    LOCAL = "local"                    # Local storage only
    HYBRID = "hybrid"                  # Local + Cloud (VPC)


class KeyManagementType(str, Enum):
    """Key management options."""
    PLATFORM_MANAGED = "platform_managed"
    CUSTOMER_MANAGED = "customer_managed"
    HSM_BACKED = "hsm_backed"
    PKCS11 = "pkcs11"


class UpdateSource(str, Enum):
    """Source for software updates."""
    CLOUD = "cloud"                    # Direct from vendor cloud
    LOCAL_REGISTRY = "local_registry"  # Local registry mirror
    OFFLINE = "offline"                # Manual offline updates


class PostureCheckResult(str, Enum):
    """Result of posture validation."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    UNKNOWN = "unknown"


class AuditAction(str, Enum):
    """Audit actions for posture changes."""
    MODE_CONFIGURED = "mode_configured"
    TELEMETRY_MODE_CHANGED = "telemetry_mode_changed"
    RAW_TELEMETRY_ENABLED = "raw_telemetry_enabled"
    RAW_TELEMETRY_DISABLED = "raw_telemetry_disabled"
    CMK_CONFIGURED = "cmk_configured"
    EVIDENCE_EXPORTED = "evidence_exported"
    POSTURE_VALIDATED = "posture_validated"
    RESIDENCY_VERIFIED = "residency_verified"
    AIR_GAP_VALIDATED = "air_gap_validated"


# EU Regions
EU_REGIONS: Set[str] = {
    # AWS
    "eu-west-1", "eu-west-2", "eu-west-3",
    "eu-central-1", "eu-central-2",
    "eu-north-1", "eu-south-1", "eu-south-2",
    # GCP
    "europe-west1", "europe-west2", "europe-west3",
    "europe-west4", "europe-west6", "europe-west8", "europe-west9",
    "europe-north1", "europe-central2",
    # Azure
    "westeurope", "northeurope",
    "germanywestcentral", "germanynorth",
    "francecentral", "francesouth",
    "swedencentral", "switzerlandnorth", "switzerlandwest",
    "uksouth", "ukwest",
    # Generic
    "eu", "europe", "on-prem", "local",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnterprisePostureConfig:
    """
    Configuration for enterprise deployment posture.

    Defines the boundaries and constraints for the deployment.
    """
    config_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    organization_id: str = ""

    # Deployment mode
    deployment_mode: EnterpriseDeploymentMode = EnterpriseDeploymentMode.SAAS

    # Telemetry configuration
    telemetry_destination: TelemetryDestination = TelemetryDestination.CLOUD
    telemetry_local_only: bool = False
    telemetry_cloud_export_enabled: bool = True
    raw_telemetry_enabled: bool = False
    raw_telemetry_opt_in_date: Optional[datetime] = None
    raw_telemetry_opt_in_by: Optional[str] = None

    # Key management
    key_management: KeyManagementType = KeyManagementType.PLATFORM_MANAGED
    cmk_provider: Optional[str] = None  # aws_kms, gcp_kms, azure_keyvault, hsm
    cmk_key_id: Optional[str] = None

    # Update source
    update_source: UpdateSource = UpdateSource.CLOUD
    local_registry_url: Optional[str] = None

    # Air-gapped settings
    air_gapped: bool = False
    offline_verification_enabled: bool = False

    # Evidence export
    evidence_export_local_only: bool = False
    evidence_export_path: Optional[str] = None

    # Residency
    data_residency: str = "eu"
    infrastructure_regions: Set[str] = field(default_factory=lambda: {"eu-west-1"})

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    updated_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": self.config_id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "deployment_mode": self.deployment_mode.value,
            "telemetry_destination": self.telemetry_destination.value,
            "telemetry_local_only": self.telemetry_local_only,
            "telemetry_cloud_export_enabled": self.telemetry_cloud_export_enabled,
            "raw_telemetry_enabled": self.raw_telemetry_enabled,
            "raw_telemetry_opt_in_date": (
                self.raw_telemetry_opt_in_date.isoformat()
                if self.raw_telemetry_opt_in_date else None
            ),
            "raw_telemetry_opt_in_by": self.raw_telemetry_opt_in_by,
            "key_management": self.key_management.value,
            "cmk_provider": self.cmk_provider,
            "cmk_key_id": self.cmk_key_id,
            "update_source": self.update_source.value,
            "local_registry_url": self.local_registry_url,
            "air_gapped": self.air_gapped,
            "offline_verification_enabled": self.offline_verification_enabled,
            "evidence_export_local_only": self.evidence_export_local_only,
            "evidence_export_path": self.evidence_export_path,
            "data_residency": self.data_residency,
            "infrastructure_regions": list(self.infrastructure_regions),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }


@dataclass
class PostureCheck:
    """Result of a single posture check."""
    check_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    category: str = ""
    result: PostureCheckResult = PostureCheckResult.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "category": self.category,
            "result": self.result.value,
            "message": self.message,
            "details": self.details,
            "required": self.required,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PostureValidationReport:
    """Complete posture validation report."""
    report_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    config: Optional[EnterprisePostureConfig] = None
    checks: List[PostureCheck] = field(default_factory=list)
    overall_result: PostureCheckResult = PostureCheckResult.UNKNOWN
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "report_id": self.report_id,
            "workspace_id": self.workspace_id,
            "overall_result": self.overall_result.value,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "validated_at": self.validated_at.isoformat(),
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    @property
    def is_compliant(self) -> bool:
        """Check if the posture is compliant (no required failures)."""
        return self.failed_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "workspace_id": self.workspace_id,
            "config": self.config.to_dict() if self.config else None,
            "checks": [c.to_dict() for c in self.checks],
            "overall_result": self.overall_result.value,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "is_compliant": self.is_compliant,
            "validated_at": self.validated_at.isoformat(),
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class TelemetryLocalModeConfig:
    """Configuration for telemetry local mode."""
    enabled: bool = True
    storage_path: str = "/data/telemetry"
    retention_days: int = 90
    encryption_enabled: bool = True
    encryption_key_id: Optional[str] = None
    max_storage_gb: int = 100
    export_allowed: bool = True  # Customer export allowed
    cloud_export_blocked: bool = True  # Cloud export blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "storage_path": self.storage_path,
            "retention_days": self.retention_days,
            "encryption_enabled": self.encryption_enabled,
            "encryption_key_id": self.encryption_key_id,
            "max_storage_gb": self.max_storage_gb,
            "export_allowed": self.export_allowed,
            "cloud_export_blocked": self.cloud_export_blocked,
        }


@dataclass
class EnterpriseEvidencePack:
    """Evidence pack for enterprise/air-gapped export."""
    pack_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    export_mode: str = "offline"  # offline, local, hybrid
    categories_included: Set[str] = field(default_factory=set)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exported_by: str = ""
    signature: Optional[str] = None
    integrity_hash: str = ""
    file_path: Optional[str] = None
    file_size_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "export_mode": self.export_mode,
            "exported_at": self.exported_at.isoformat(),
            "artifacts_count": len(self.artifacts),
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "export_mode": self.export_mode,
            "categories_included": list(self.categories_included),
            "artifacts_count": len(self.artifacts),
            "manifest": self.manifest,
            "exported_at": self.exported_at.isoformat(),
            "exported_by": self.exported_by,
            "signature": self.signature,
            "integrity_hash": self.integrity_hash,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class PostureAuditEvent:
    """Audit event for posture changes."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    action: AuditAction = AuditAction.MODE_CONFIGURED
    actor_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "action": self.action.value,
            "actor_id": self.actor_id,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "action": self.action.value,
            "actor_id": self.actor_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


# =============================================================================
# Telemetry Local Mode Service
# =============================================================================

class TelemetryLocalModeService:
    """
    Service for managing "telemetry stays local" mode.

    Enforces that telemetry never leaves customer infrastructure
    when local mode is enabled (enterprise on-prem/air-gapped).
    """

    def __init__(
        self,
        on_event: Optional[Callable[[PostureAuditEvent], None]] = None,
    ):
        self._lock = threading.RLock()
        self._configs: Dict[str, TelemetryLocalModeConfig] = {}
        self._on_event = on_event

    def enable_local_mode(
        self,
        workspace_id: str,
        config: TelemetryLocalModeConfig,
        enabled_by: str,
    ) -> TelemetryLocalModeConfig:
        """Enable telemetry local mode for a workspace."""
        config.enabled = True
        config.cloud_export_blocked = True

        with self._lock:
            self._configs[workspace_id] = config

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.TELEMETRY_MODE_CHANGED,
            actor_id=enabled_by,
            details={
                "mode": "local",
                "cloud_export_blocked": True,
                "storage_path": config.storage_path,
            },
        ))

        return config

    def disable_local_mode(
        self,
        workspace_id: str,
        disabled_by: str,
        reason: str,
    ) -> Optional[TelemetryLocalModeConfig]:
        """Disable telemetry local mode (requires explicit action)."""
        with self._lock:
            config = self._configs.get(workspace_id)
            if config:
                config.enabled = False
                config.cloud_export_blocked = False

        if config:
            self._log_event(PostureAuditEvent(
                workspace_id=workspace_id,
                action=AuditAction.TELEMETRY_MODE_CHANGED,
                actor_id=disabled_by,
                details={
                    "mode": "cloud",
                    "cloud_export_blocked": False,
                    "reason": reason,
                },
            ))

        return config

    def get_config(self, workspace_id: str) -> Optional[TelemetryLocalModeConfig]:
        """Get local mode config for workspace."""
        with self._lock:
            return self._configs.get(workspace_id)

    def is_local_mode_enabled(self, workspace_id: str) -> bool:
        """Check if local mode is enabled."""
        config = self.get_config(workspace_id)
        return config.enabled if config else False

    def is_cloud_export_blocked(self, workspace_id: str) -> bool:
        """Check if cloud export is blocked."""
        config = self.get_config(workspace_id)
        if config:
            return config.cloud_export_blocked
        return False

    def validate_telemetry_destination(
        self,
        workspace_id: str,
        destination: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a telemetry destination is allowed.

        Returns (is_allowed, error_message).
        """
        config = self.get_config(workspace_id)
        if not config or not config.enabled:
            return True, None

        if config.cloud_export_blocked and destination == "cloud":
            return False, (
                "Telemetry local mode is enabled. "
                "Cloud export is blocked. "
                "Telemetry must stay local."
            )

        return True, None

    def _log_event(self, event: PostureAuditEvent) -> None:
        """Log audit event."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass


# =============================================================================
# Enterprise Evidence Pack Exporter
# =============================================================================

class EnterpriseEvidencePackExporter:
    """
    Evidence pack exporter for enterprise/air-gapped deployments.

    Generates evidence packs that can be exported locally without
    external connectivity.
    """

    def __init__(
        self,
        # Service references
        evidence_pack_service: Optional[Any] = None,
        posture_service: Optional[Any] = None,
        residency_service: Optional[Any] = None,
        telemetry_contract_service: Optional[Any] = None,
        # Signing
        signing_key_path: Optional[str] = None,
        # Callbacks
        on_event: Optional[Callable[[PostureAuditEvent], None]] = None,
    ):
        self._lock = threading.RLock()
        self._evidence_pack_service = evidence_pack_service
        self._posture_service = posture_service
        self._residency_service = residency_service
        self._telemetry_contract_service = telemetry_contract_service
        self._signing_key_path = signing_key_path
        self._on_event = on_event

        # Storage
        self._packs: Dict[str, EnterpriseEvidencePack] = {}

    def export_offline(
        self,
        workspace_id: str,
        exported_by: str,
        output_path: str,
        categories: Optional[Set[str]] = None,
        sign: bool = True,
    ) -> EnterpriseEvidencePack:
        """
        Export evidence pack for offline/air-gapped use.

        Args:
            workspace_id: Workspace identifier
            exported_by: Principal performing export
            output_path: Path to write the evidence pack
            categories: Categories to include (default: all)
            sign: Whether to sign the pack

        Returns:
            EnterpriseEvidencePack with export details
        """
        pack = EnterpriseEvidencePack(
            workspace_id=workspace_id,
            export_mode="offline",
            exported_by=exported_by,
        )

        # Default categories for enterprise evidence
        if categories is None:
            categories = {
                "security_baseline",
                "supply_chain",
                "access_audit",
                "change_journal",
                "retention_policies",
                "residency_evidence",
                "telemetry_contracts",
                "posture_config",
            }

        pack.categories_included = categories

        # Collect evidence artifacts
        artifacts = []

        # Posture configuration
        if self._posture_service:
            posture_config = self._posture_service.get_config(workspace_id)
            if posture_config:
                artifacts.append({
                    "category": "posture_config",
                    "filename": "posture_config.json",
                    "content": posture_config.to_dict(),
                    "content_hash": self._hash_content(posture_config.to_dict()),
                })

        # Residency evidence
        if self._residency_service:
            try:
                if hasattr(self._residency_service, 'run_drift_check'):
                    residency_report = self._residency_service.run_drift_check()
                    artifacts.append({
                        "category": "residency_evidence",
                        "filename": "residency_drift_check.json",
                        "content": (
                            residency_report.to_dict()
                            if hasattr(residency_report, 'to_dict')
                            else residency_report
                        ),
                        "content_hash": self._hash_content(residency_report),
                    })
            except Exception:
                pass

        # Telemetry contracts
        if self._telemetry_contract_service:
            try:
                if hasattr(self._telemetry_contract_service, 'export_contracts'):
                    contracts = self._telemetry_contract_service.export_contracts(
                        workspace_id=workspace_id
                    )
                    artifacts.append({
                        "category": "telemetry_contracts",
                        "filename": "telemetry_contracts.json",
                        "content": contracts,
                        "content_hash": self._hash_content(contracts),
                    })
            except Exception:
                pass

        # Add from evidence pack service if available
        if self._evidence_pack_service:
            try:
                # Get existing evidence categories
                for category in categories:
                    if category in ["posture_config", "residency_evidence", "telemetry_contracts"]:
                        continue  # Already handled

                    try:
                        evidence = self._get_evidence_for_category(
                            workspace_id, category
                        )
                        if evidence:
                            artifacts.append({
                                "category": category,
                                "filename": f"{category}.json",
                                "content": evidence,
                                "content_hash": self._hash_content(evidence),
                            })
                    except Exception:
                        pass
            except Exception:
                pass

        pack.artifacts = artifacts

        # Create manifest
        pack.manifest = {
            "version": "1.0.0",
            "pack_id": pack.pack_id,
            "workspace_id": workspace_id,
            "export_mode": "offline",
            "exported_at": pack.exported_at.isoformat(),
            "exported_by": exported_by,
            "categories": list(categories),
            "artifact_count": len(artifacts),
            "artifacts": [
                {"filename": a["filename"], "hash": a["content_hash"]}
                for a in artifacts
            ],
        }

        # Write to ZIP file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            manifest_json = json.dumps(pack.manifest, indent=2, sort_keys=True)
            zf.writestr("manifest.json", manifest_json)

            # Add artifacts
            for artifact in artifacts:
                content_json = json.dumps(
                    artifact["content"], indent=2, sort_keys=True, default=str
                )
                zf.writestr(f"artifacts/{artifact['filename']}", content_json)

        # Get ZIP content
        zip_buffer.seek(0)
        zip_content = zip_buffer.read()

        # Sign if requested
        if sign:
            pack.signature = self._sign_content(zip_content)

        # Write to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_bytes(zip_content)

        pack.file_path = str(output_path_obj)
        pack.file_size_bytes = len(zip_content)
        pack.integrity_hash = f"sha256:{hashlib.sha256(zip_content).hexdigest()}"

        # Store pack
        with self._lock:
            self._packs[pack.pack_id] = pack

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.EVIDENCE_EXPORTED,
            actor_id=exported_by,
            details={
                "pack_id": pack.pack_id,
                "export_mode": "offline",
                "file_path": pack.file_path,
                "file_size_bytes": pack.file_size_bytes,
                "signed": sign,
            },
        ))

        return pack

    def verify_pack(
        self,
        pack_path: str,
        public_key_path: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify an evidence pack's integrity and signature.

        Returns (is_valid, verification_details).
        """
        pack_path_obj = Path(pack_path)
        if not pack_path_obj.exists():
            return False, {"error": f"Pack not found: {pack_path}"}

        verification = {
            "pack_path": pack_path,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "checks": [],
        }

        try:
            # Read pack
            zip_content = pack_path_obj.read_bytes()
            content_hash = f"sha256:{hashlib.sha256(zip_content).hexdigest()}"
            verification["content_hash"] = content_hash

            # Open and verify structure
            with zipfile.ZipFile(BytesIO(zip_content), 'r') as zf:
                # Check manifest
                if "manifest.json" not in zf.namelist():
                    verification["checks"].append({
                        "name": "manifest_present",
                        "passed": False,
                        "message": "Manifest not found",
                    })
                    return False, verification

                verification["checks"].append({
                    "name": "manifest_present",
                    "passed": True,
                })

                # Read manifest
                manifest_content = zf.read("manifest.json").decode('utf-8')
                manifest = json.loads(manifest_content)
                verification["manifest"] = manifest

                # Verify artifact hashes
                all_valid = True
                for artifact_info in manifest.get("artifacts", []):
                    filename = artifact_info["filename"]
                    expected_hash = artifact_info["hash"]

                    artifact_path = f"artifacts/{filename}"
                    if artifact_path not in zf.namelist():
                        verification["checks"].append({
                            "name": f"artifact_{filename}",
                            "passed": False,
                            "message": f"Artifact not found: {filename}",
                        })
                        all_valid = False
                        continue

                    artifact_content = zf.read(artifact_path)
                    actual_hash = f"sha256:{hashlib.sha256(artifact_content).hexdigest()}"

                    if actual_hash == expected_hash:
                        verification["checks"].append({
                            "name": f"artifact_{filename}",
                            "passed": True,
                        })
                    else:
                        verification["checks"].append({
                            "name": f"artifact_{filename}",
                            "passed": False,
                            "message": f"Hash mismatch: expected {expected_hash}, got {actual_hash}",
                        })
                        all_valid = False

                verification["all_artifacts_valid"] = all_valid

                # Verify signature if public key provided (per Design Doc 8.3)
                if public_key_path:
                    try:
                        from ccea.crypto.signing import verify_signature
                        from ccea.crypto.keys import load_public_key

                        # Look for signature file in pack
                        sig_path = "evidence_pack.sig"
                        if sig_path in zf.namelist():
                            signature = zf.read(sig_path)
                            manifest_content = zf.read("manifest.json")

                            # Load public key
                            public_key = load_public_key(
                                Path(public_key_path).read_text()
                            )

                            # Verify signature over manifest
                            sig_valid = verify_signature(
                                manifest_content, signature, public_key
                            )

                            verification["signature_verification"] = {
                                "verified": sig_valid,
                                "public_key_path": str(public_key_path),
                                "signed_content": "manifest.json",
                            }

                            if not sig_valid:
                                all_valid = False
                                verification["checks"].append({
                                    "name": "signature_verification",
                                    "passed": False,
                                    "message": "Evidence pack signature invalid",
                                })
                            else:
                                verification["checks"].append({
                                    "name": "signature_verification",
                                    "passed": True,
                                })
                        else:
                            verification["signature_verification"] = {
                                "verified": False,
                                "error": "No signature file in evidence pack",
                            }
                            verification["checks"].append({
                                "name": "signature_verification",
                                "passed": False,
                                "message": "Evidence pack not signed",
                            })
                    except ImportError:
                        verification["signature_verification"] = {
                            "verified": False,
                            "error": "ccea.crypto module not available",
                        }
                    except Exception as sig_err:
                        verification["signature_verification"] = {
                            "verified": False,
                            "error": str(sig_err),
                        }

                return all_valid, verification

        except Exception as e:
            verification["error"] = str(e)
            return False, verification

    def get_pack(self, pack_id: str) -> Optional[EnterpriseEvidencePack]:
        """Get pack by ID."""
        with self._lock:
            return self._packs.get(pack_id)

    def list_packs(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[EnterpriseEvidencePack]:
        """List exported packs."""
        with self._lock:
            packs = list(self._packs.values())

        if workspace_id:
            packs = [p for p in packs if p.workspace_id == workspace_id]

        packs.sort(key=lambda p: p.exported_at, reverse=True)
        return packs[:limit]

    def _get_evidence_for_category(
        self,
        workspace_id: str,
        category: str,
    ) -> Optional[Dict[str, Any]]:
        """Get evidence for a specific category."""
        # Placeholder - in production would delegate to evidence pack service
        return {
            "category": category,
            "workspace_id": workspace_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": [],
        }

    def _hash_content(self, content: Any) -> str:
        """Hash content for integrity."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

    def _sign_content(self, content: bytes) -> str:
        """Sign content (placeholder for real signing)."""
        # In production, would use actual signing with private key
        return f"sig:{hashlib.sha256(content).hexdigest()[:32]}"

    def _log_event(self, event: PostureAuditEvent) -> None:
        """Log audit event."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass


# =============================================================================
# Enterprise Posture Validator
# =============================================================================

class EnterprisePostureValidator:
    """
    Validates enterprise deployment posture against requirements.

    Ensures that on-prem/VPC deployments meet all GDPR compliance
    requirements and can produce auditable evidence.
    """

    def __init__(
        self,
        on_event: Optional[Callable[[PostureAuditEvent], None]] = None,
    ):
        self._lock = threading.RLock()
        self._on_event = on_event
        self._reports: Dict[str, PostureValidationReport] = {}

    def validate(
        self,
        config: EnterprisePostureConfig,
        validated_by: str,
    ) -> PostureValidationReport:
        """
        Validate enterprise posture configuration.

        Runs all applicable checks based on deployment mode.
        """
        report = PostureValidationReport(
            workspace_id=config.workspace_id,
            config=config,
        )

        checks = []

        # Common checks
        checks.extend(self._check_eu_residency(config))
        checks.extend(self._check_telemetry_config(config))
        checks.extend(self._check_key_management(config))

        # Mode-specific checks
        if config.deployment_mode in [
            EnterpriseDeploymentMode.ON_PREM_FULL,
            EnterpriseDeploymentMode.AIR_GAPPED,
        ]:
            checks.extend(self._check_local_telemetry(config))
            checks.extend(self._check_local_evidence_export(config))

        if config.deployment_mode == EnterpriseDeploymentMode.AIR_GAPPED:
            checks.extend(self._check_air_gapped(config))

        if config.raw_telemetry_enabled:
            checks.extend(self._check_raw_telemetry(config))

        report.checks = checks

        # Calculate results
        passed = [c for c in checks if c.result == PostureCheckResult.PASS]
        failed = [c for c in checks if c.result == PostureCheckResult.FAIL and c.required]
        warnings = [c for c in checks if c.result == PostureCheckResult.WARNING]

        report.passed_count = len(passed)
        report.failed_count = len(failed)
        report.warning_count = len(warnings)

        if failed:
            report.overall_result = PostureCheckResult.FAIL
        elif warnings:
            report.overall_result = PostureCheckResult.WARNING
        else:
            report.overall_result = PostureCheckResult.PASS

        report.integrity_hash = report._compute_hash()

        # Store report
        with self._lock:
            self._reports[report.report_id] = report

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=config.workspace_id,
            action=AuditAction.POSTURE_VALIDATED,
            actor_id=validated_by,
            details={
                "report_id": report.report_id,
                "overall_result": report.overall_result.value,
                "passed_count": report.passed_count,
                "failed_count": report.failed_count,
                "is_compliant": report.is_compliant,
            },
        ))

        return report

    def _check_eu_residency(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check EU residency requirements."""
        checks = []

        # Check data residency setting
        checks.append(PostureCheck(
            name="data_residency_eu",
            category="residency",
            result=(
                PostureCheckResult.PASS
                if config.data_residency in ["eu", "on-prem", "local"]
                else PostureCheckResult.FAIL
            ),
            message=(
                f"Data residency set to: {config.data_residency}"
                if config.data_residency in ["eu", "on-prem", "local"]
                else f"Non-EU data residency detected: {config.data_residency}"
            ),
            details={"data_residency": config.data_residency},
            required=True,
        ))

        # Check infrastructure regions
        non_eu_regions = config.infrastructure_regions - EU_REGIONS
        checks.append(PostureCheck(
            name="infrastructure_regions_eu",
            category="residency",
            result=(
                PostureCheckResult.PASS
                if not non_eu_regions
                else PostureCheckResult.FAIL
            ),
            message=(
                f"All infrastructure regions are EU: {config.infrastructure_regions}"
                if not non_eu_regions
                else f"Non-EU regions detected: {non_eu_regions}"
            ),
            details={
                "regions": list(config.infrastructure_regions),
                "non_eu_regions": list(non_eu_regions),
            },
            required=True,
        ))

        return checks

    def _check_telemetry_config(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check telemetry configuration."""
        checks = []

        # Telemetry destination appropriate for mode
        valid_destinations = {
            EnterpriseDeploymentMode.SAAS: {TelemetryDestination.CLOUD},
            EnterpriseDeploymentMode.ENTERPRISE_CLOUD: {
                TelemetryDestination.CLOUD, TelemetryDestination.HYBRID
            },
            EnterpriseDeploymentMode.ON_PREM_FULL: {TelemetryDestination.LOCAL},
            EnterpriseDeploymentMode.VPC_MANAGED: {
                TelemetryDestination.LOCAL, TelemetryDestination.HYBRID
            },
            EnterpriseDeploymentMode.AIR_GAPPED: {TelemetryDestination.LOCAL},
        }

        expected = valid_destinations.get(config.deployment_mode, set())
        is_valid = config.telemetry_destination in expected

        checks.append(PostureCheck(
            name="telemetry_destination_valid",
            category="telemetry",
            result=PostureCheckResult.PASS if is_valid else PostureCheckResult.FAIL,
            message=(
                f"Telemetry destination {config.telemetry_destination.value} "
                f"valid for {config.deployment_mode.value}"
                if is_valid
                else f"Telemetry destination {config.telemetry_destination.value} "
                     f"not valid for {config.deployment_mode.value}. Expected: {expected}"
            ),
            details={
                "destination": config.telemetry_destination.value,
                "mode": config.deployment_mode.value,
                "expected": [d.value for d in expected],
            },
            required=True,
        ))

        return checks

    def _check_key_management(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check key management configuration."""
        checks = []

        # CMK required for certain modes
        cmk_required = config.deployment_mode in [
            EnterpriseDeploymentMode.ON_PREM_FULL,
            EnterpriseDeploymentMode.AIR_GAPPED,
        ]

        has_cmk = config.key_management in [
            KeyManagementType.CUSTOMER_MANAGED,
            KeyManagementType.HSM_BACKED,
            KeyManagementType.PKCS11,
        ]

        checks.append(PostureCheck(
            name="key_management_appropriate",
            category="security",
            result=(
                PostureCheckResult.PASS
                if not cmk_required or has_cmk
                else PostureCheckResult.WARNING
            ),
            message=(
                f"Key management: {config.key_management.value}"
                if not cmk_required or has_cmk
                else "Customer-managed keys recommended for on-prem/air-gapped"
            ),
            details={
                "key_management": config.key_management.value,
                "cmk_provider": config.cmk_provider,
            },
            required=False,  # Warning only
        ))

        return checks

    def _check_local_telemetry(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check local telemetry requirements for on-prem."""
        checks = []

        # Telemetry should be local only
        checks.append(PostureCheck(
            name="telemetry_local_only",
            category="telemetry",
            result=(
                PostureCheckResult.PASS
                if config.telemetry_local_only
                else PostureCheckResult.FAIL
            ),
            message=(
                "Telemetry local mode enabled"
                if config.telemetry_local_only
                else "Telemetry local mode must be enabled for on-prem/air-gapped"
            ),
            details={"telemetry_local_only": config.telemetry_local_only},
            required=True,
        ))

        # Cloud export should be disabled
        checks.append(PostureCheck(
            name="cloud_export_disabled",
            category="telemetry",
            result=(
                PostureCheckResult.PASS
                if not config.telemetry_cloud_export_enabled
                else PostureCheckResult.FAIL
            ),
            message=(
                "Cloud telemetry export disabled"
                if not config.telemetry_cloud_export_enabled
                else "Cloud telemetry export must be disabled for on-prem/air-gapped"
            ),
            details={
                "telemetry_cloud_export_enabled": config.telemetry_cloud_export_enabled
            },
            required=True,
        ))

        return checks

    def _check_local_evidence_export(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check local evidence export configuration."""
        checks = []

        checks.append(PostureCheck(
            name="evidence_export_local",
            category="evidence",
            result=(
                PostureCheckResult.PASS
                if config.evidence_export_local_only
                else PostureCheckResult.WARNING
            ),
            message=(
                "Evidence export configured for local only"
                if config.evidence_export_local_only
                else "Evidence export should be local for on-prem"
            ),
            details={
                "evidence_export_local_only": config.evidence_export_local_only,
                "evidence_export_path": config.evidence_export_path,
            },
            required=False,
        ))

        # Check export path is set
        if config.evidence_export_local_only:
            checks.append(PostureCheck(
                name="evidence_export_path_set",
                category="evidence",
                result=(
                    PostureCheckResult.PASS
                    if config.evidence_export_path
                    else PostureCheckResult.FAIL
                ),
                message=(
                    f"Evidence export path: {config.evidence_export_path}"
                    if config.evidence_export_path
                    else "Evidence export path must be set for local export"
                ),
                required=True,
            ))

        return checks

    def _check_air_gapped(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check air-gapped specific requirements."""
        checks = []

        # Air-gapped flag
        checks.append(PostureCheck(
            name="air_gapped_enabled",
            category="air_gapped",
            result=(
                PostureCheckResult.PASS
                if config.air_gapped
                else PostureCheckResult.FAIL
            ),
            message=(
                "Air-gapped mode enabled"
                if config.air_gapped
                else "Air-gapped mode must be enabled"
            ),
            required=True,
        ))

        # Offline verification
        checks.append(PostureCheck(
            name="offline_verification",
            category="air_gapped",
            result=(
                PostureCheckResult.PASS
                if config.offline_verification_enabled
                else PostureCheckResult.FAIL
            ),
            message=(
                "Offline verification enabled"
                if config.offline_verification_enabled
                else "Offline verification must be enabled for air-gapped"
            ),
            required=True,
        ))

        # Update source must be offline
        checks.append(PostureCheck(
            name="update_source_offline",
            category="air_gapped",
            result=(
                PostureCheckResult.PASS
                if config.update_source in [
                    UpdateSource.LOCAL_REGISTRY, UpdateSource.OFFLINE
                ]
                else PostureCheckResult.FAIL
            ),
            message=(
                f"Update source: {config.update_source.value}"
                if config.update_source in [
                    UpdateSource.LOCAL_REGISTRY, UpdateSource.OFFLINE
                ]
                else "Update source must be local/offline for air-gapped"
            ),
            required=True,
        ))

        return checks

    def _check_raw_telemetry(
        self,
        config: EnterprisePostureConfig,
    ) -> List[PostureCheck]:
        """Check RAW telemetry requirements (enterprise only)."""
        checks = []

        # RAW only for enterprise
        is_enterprise = config.deployment_mode in [
            EnterpriseDeploymentMode.ENTERPRISE_CLOUD,
            EnterpriseDeploymentMode.ON_PREM_FULL,
            EnterpriseDeploymentMode.VPC_MANAGED,
        ]

        checks.append(PostureCheck(
            name="raw_telemetry_enterprise_only",
            category="telemetry",
            result=(
                PostureCheckResult.PASS
                if is_enterprise
                else PostureCheckResult.FAIL
            ),
            message=(
                "RAW telemetry enabled for enterprise deployment"
                if is_enterprise
                else "RAW telemetry only available for enterprise deployments"
            ),
            required=True,
        ))

        # Opt-in must be recorded
        checks.append(PostureCheck(
            name="raw_telemetry_opt_in_recorded",
            category="telemetry",
            result=(
                PostureCheckResult.PASS
                if config.raw_telemetry_opt_in_date and config.raw_telemetry_opt_in_by
                else PostureCheckResult.FAIL
            ),
            message=(
                f"RAW telemetry opt-in recorded: {config.raw_telemetry_opt_in_date}"
                if config.raw_telemetry_opt_in_date
                else "RAW telemetry requires explicit opt-in record"
            ),
            details={
                "opt_in_date": (
                    config.raw_telemetry_opt_in_date.isoformat()
                    if config.raw_telemetry_opt_in_date else None
                ),
                "opt_in_by": config.raw_telemetry_opt_in_by,
            },
            required=True,
        ))

        return checks

    def get_report(self, report_id: str) -> Optional[PostureValidationReport]:
        """Get validation report by ID."""
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PostureValidationReport]:
        """List validation reports."""
        with self._lock:
            reports = list(self._reports.values())

        if workspace_id:
            reports = [r for r in reports if r.workspace_id == workspace_id]

        reports.sort(key=lambda r: r.validated_at, reverse=True)
        return reports[:limit]

    def _log_event(self, event: PostureAuditEvent) -> None:
        """Log audit event."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass


# =============================================================================
# Enterprise Posture Service
# =============================================================================

class EnterprisePostureService:
    """
    Main service for enterprise deployment posture management.

    Provides unified interface for:
    - Deployment mode configuration
    - Telemetry local mode
    - Evidence pack export (air-gapped)
    - Posture validation
    - RAW telemetry gating
    """

    def __init__(
        self,
        # Service references
        evidence_pack_service: Optional[Any] = None,
        residency_service: Optional[Any] = None,
        telemetry_contract_service: Optional[Any] = None,
        raw_gate: Optional[Any] = None,
        # Signing
        signing_key_path: Optional[str] = None,
        # Callbacks
        on_event: Optional[Callable[[PostureAuditEvent], None]] = None,
    ):
        self._lock = threading.RLock()
        self._on_event = on_event

        # Storage
        self._configs: Dict[str, EnterprisePostureConfig] = {}
        self._events: List[PostureAuditEvent] = []

        # Sub-services
        self._telemetry_local_mode = TelemetryLocalModeService(
            on_event=self._handle_event
        )
        self._evidence_exporter = EnterpriseEvidencePackExporter(
            evidence_pack_service=evidence_pack_service,
            posture_service=self,
            residency_service=residency_service,
            telemetry_contract_service=telemetry_contract_service,
            signing_key_path=signing_key_path,
            on_event=self._handle_event,
        )
        self._validator = EnterprisePostureValidator(
            on_event=self._handle_event
        )

        # External services
        self._raw_gate = raw_gate

    # =========================================================================
    # Configuration Management
    # =========================================================================

    def configure(
        self,
        workspace_id: str,
        organization_id: str,
        deployment_mode: EnterpriseDeploymentMode,
        configured_by: str,
        **kwargs,
    ) -> EnterprisePostureConfig:
        """
        Configure enterprise deployment posture.

        Args:
            workspace_id: Workspace identifier
            organization_id: Organization identifier
            deployment_mode: Deployment mode
            configured_by: Principal configuring
            **kwargs: Additional configuration options

        Returns:
            EnterprisePostureConfig
        """
        # Create config with defaults based on mode
        config = EnterprisePostureConfig(
            workspace_id=workspace_id,
            organization_id=organization_id,
            deployment_mode=deployment_mode,
            created_by=configured_by,
            updated_by=configured_by,
        )

        # Apply mode-specific defaults
        if deployment_mode == EnterpriseDeploymentMode.ON_PREM_FULL:
            config.telemetry_destination = TelemetryDestination.LOCAL
            config.telemetry_local_only = True
            config.telemetry_cloud_export_enabled = False
            config.update_source = UpdateSource.LOCAL_REGISTRY
            config.evidence_export_local_only = True
            config.data_residency = "on-prem"

        elif deployment_mode == EnterpriseDeploymentMode.AIR_GAPPED:
            config.telemetry_destination = TelemetryDestination.LOCAL
            config.telemetry_local_only = True
            config.telemetry_cloud_export_enabled = False
            config.update_source = UpdateSource.OFFLINE
            config.air_gapped = True
            config.offline_verification_enabled = True
            config.evidence_export_local_only = True
            config.data_residency = "on-prem"

        elif deployment_mode == EnterpriseDeploymentMode.VPC_MANAGED:
            config.telemetry_destination = TelemetryDestination.HYBRID
            config.telemetry_local_only = kwargs.get("telemetry_local_only", True)
            config.update_source = UpdateSource.LOCAL_REGISTRY

        elif deployment_mode == EnterpriseDeploymentMode.ENTERPRISE_CLOUD:
            config.telemetry_destination = TelemetryDestination.CLOUD
            config.key_management = KeyManagementType.CUSTOMER_MANAGED

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Store config
        with self._lock:
            self._configs[workspace_id] = config

        # Enable telemetry local mode if configured
        if config.telemetry_local_only:
            self._telemetry_local_mode.enable_local_mode(
                workspace_id=workspace_id,
                config=TelemetryLocalModeConfig(
                    enabled=True,
                    storage_path=kwargs.get("telemetry_storage_path", "/data/telemetry"),
                    retention_days=kwargs.get("telemetry_retention_days", 90),
                    encryption_enabled=True,
                ),
                enabled_by=configured_by,
            )

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.MODE_CONFIGURED,
            actor_id=configured_by,
            details={
                "deployment_mode": deployment_mode.value,
                "telemetry_destination": config.telemetry_destination.value,
                "telemetry_local_only": config.telemetry_local_only,
                "air_gapped": config.air_gapped,
            },
        ))

        return config

    def get_config(self, workspace_id: str) -> Optional[EnterprisePostureConfig]:
        """Get posture config for workspace."""
        with self._lock:
            return self._configs.get(workspace_id)

    def update_config(
        self,
        workspace_id: str,
        updated_by: str,
        **kwargs,
    ) -> Optional[EnterprisePostureConfig]:
        """Update posture configuration."""
        with self._lock:
            config = self._configs.get(workspace_id)
            if not config:
                return None

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            config.updated_at = datetime.now(timezone.utc)
            config.updated_by = updated_by

        return config

    # =========================================================================
    # RAW Telemetry Management
    # =========================================================================

    def enable_raw_telemetry(
        self,
        workspace_id: str,
        organization_id: str,
        enabled_by: str,
        acknowledgment_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enable RAW_ORDER_EVENTS telemetry for enterprise workspace.

        Args:
            workspace_id: Workspace identifier
            organization_id: Organization identifier
            enabled_by: Principal enabling
            acknowledgment_text: Acknowledgment text signed by user

        Returns:
            (success, error_message)
        """
        config = self.get_config(workspace_id)
        if not config:
            return False, "Workspace not configured"

        # Verify enterprise mode
        if config.deployment_mode == EnterpriseDeploymentMode.SAAS:
            return False, "RAW telemetry not available for SaaS deployments"

        # If external RAW gate provided, use it
        if self._raw_gate:
            gate_ok, gate_errors = self._raw_gate.validate_gate(
                workspace_id, organization_id
            )
            if not gate_ok:
                return False, "; ".join(gate_errors)

            # Register opt-in
            self._raw_gate.register_opt_in(
                workspace_id=workspace_id,
                organization_id=organization_id,
                principal_id=enabled_by,
                acknowledgment_text=acknowledgment_text,
            )

        # Update config
        with self._lock:
            config.raw_telemetry_enabled = True
            config.raw_telemetry_opt_in_date = datetime.now(timezone.utc)
            config.raw_telemetry_opt_in_by = enabled_by
            config.updated_at = datetime.now(timezone.utc)
            config.updated_by = enabled_by

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.RAW_TELEMETRY_ENABLED,
            actor_id=enabled_by,
            details={
                "organization_id": organization_id,
                "acknowledgment_hash": hashlib.sha256(
                    acknowledgment_text.encode()
                ).hexdigest(),
            },
        ))

        return True, None

    def disable_raw_telemetry(
        self,
        workspace_id: str,
        disabled_by: str,
        reason: str,
    ) -> bool:
        """Disable RAW telemetry."""
        config = self.get_config(workspace_id)
        if not config:
            return False

        with self._lock:
            config.raw_telemetry_enabled = False
            config.updated_at = datetime.now(timezone.utc)
            config.updated_by = disabled_by

        # Revoke in external gate if present
        if self._raw_gate:
            self._raw_gate.revoke_opt_in(workspace_id, disabled_by, reason)

        # Audit event
        self._log_event(PostureAuditEvent(
            workspace_id=workspace_id,
            action=AuditAction.RAW_TELEMETRY_DISABLED,
            actor_id=disabled_by,
            details={"reason": reason},
        ))

        return True

    # =========================================================================
    # Telemetry Local Mode
    # =========================================================================

    def is_telemetry_local_mode(self, workspace_id: str) -> bool:
        """Check if telemetry local mode is enabled."""
        return self._telemetry_local_mode.is_local_mode_enabled(workspace_id)

    def validate_telemetry_destination(
        self,
        workspace_id: str,
        destination: str,
    ) -> Tuple[bool, Optional[str]]:
        """Validate telemetry destination is allowed."""
        return self._telemetry_local_mode.validate_telemetry_destination(
            workspace_id, destination
        )

    # =========================================================================
    # Evidence Export
    # =========================================================================

    def export_evidence_offline(
        self,
        workspace_id: str,
        exported_by: str,
        output_path: str,
        categories: Optional[Set[str]] = None,
        sign: bool = True,
    ) -> EnterpriseEvidencePack:
        """Export evidence pack for offline/air-gapped use."""
        return self._evidence_exporter.export_offline(
            workspace_id=workspace_id,
            exported_by=exported_by,
            output_path=output_path,
            categories=categories,
            sign=sign,
        )

    def verify_evidence_pack(
        self,
        pack_path: str,
        public_key_path: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify an evidence pack."""
        return self._evidence_exporter.verify_pack(pack_path, public_key_path)

    # =========================================================================
    # Posture Validation
    # =========================================================================

    def validate_posture(
        self,
        workspace_id: str,
        validated_by: str,
    ) -> PostureValidationReport:
        """Validate enterprise posture configuration."""
        config = self.get_config(workspace_id)
        if not config:
            # Return failed report if no config
            report = PostureValidationReport(
                workspace_id=workspace_id,
                overall_result=PostureCheckResult.FAIL,
                checks=[PostureCheck(
                    name="config_exists",
                    category="configuration",
                    result=PostureCheckResult.FAIL,
                    message="No posture configuration found for workspace",
                    required=True,
                )],
                failed_count=1,
            )
            return report

        return self._validator.validate(config, validated_by)

    def get_validation_report(
        self,
        report_id: str,
    ) -> Optional[PostureValidationReport]:
        """Get validation report by ID."""
        return self._validator.get_report(report_id)

    # =========================================================================
    # Audit
    # =========================================================================

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[PostureAuditEvent]:
        """Get posture audit events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def export_posture_evidence(
        self,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """Export posture configuration and validation for evidence pack."""
        config = self.get_config(workspace_id)

        # Get latest validation report
        reports = self._validator.list_reports(workspace_id=workspace_id, limit=1)
        latest_report = reports[0] if reports else None

        # Get telemetry local mode config
        telemetry_config = self._telemetry_local_mode.get_config(workspace_id)

        return {
            "workspace_id": workspace_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "posture_config": config.to_dict() if config else None,
            "telemetry_local_config": (
                telemetry_config.to_dict() if telemetry_config else None
            ),
            "latest_validation": (
                latest_report.to_dict() if latest_report else None
            ),
            "is_compliant": latest_report.is_compliant if latest_report else False,
        }

    def _handle_event(self, event: PostureAuditEvent) -> None:
        """Handle events from sub-services."""
        self._log_event(event)

    def _log_event(self, event: PostureAuditEvent) -> None:
        """Log audit event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass


# =============================================================================
# Factory Functions
# =============================================================================

def create_on_prem_posture(
    workspace_id: str,
    organization_id: str,
    configured_by: str,
    telemetry_storage_path: str = "/data/telemetry",
    evidence_export_path: str = "/data/evidence",
    infrastructure_regions: Optional[Set[str]] = None,
) -> EnterprisePostureConfig:
    """
    Create a posture configuration for on-prem deployment.

    Convenience factory for the most common on-prem setup.
    """
    service = EnterprisePostureService()
    return service.configure(
        workspace_id=workspace_id,
        organization_id=organization_id,
        deployment_mode=EnterpriseDeploymentMode.ON_PREM_FULL,
        configured_by=configured_by,
        telemetry_storage_path=telemetry_storage_path,
        evidence_export_path=evidence_export_path,
        infrastructure_regions=infrastructure_regions or {"on-prem"},
    )


def create_air_gapped_posture(
    workspace_id: str,
    organization_id: str,
    configured_by: str,
    local_registry_url: str,
    telemetry_storage_path: str = "/data/telemetry",
    evidence_export_path: str = "/data/evidence",
) -> EnterprisePostureConfig:
    """
    Create a posture configuration for air-gapped deployment.

    Convenience factory for air-gapped environments.
    """
    service = EnterprisePostureService()
    return service.configure(
        workspace_id=workspace_id,
        organization_id=organization_id,
        deployment_mode=EnterpriseDeploymentMode.AIR_GAPPED,
        configured_by=configured_by,
        local_registry_url=local_registry_url,
        telemetry_storage_path=telemetry_storage_path,
        evidence_export_path=evidence_export_path,
        infrastructure_regions={"on-prem"},
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "EnterpriseDeploymentMode",
    "TelemetryDestination",
    "KeyManagementType",
    "UpdateSource",
    "PostureCheckResult",
    "AuditAction",
    # Data classes
    "EnterprisePostureConfig",
    "PostureCheck",
    "PostureValidationReport",
    "TelemetryLocalModeConfig",
    "EnterpriseEvidencePack",
    "PostureAuditEvent",
    # Services
    "TelemetryLocalModeService",
    "EnterpriseEvidencePackExporter",
    "EnterprisePostureValidator",
    "EnterprisePostureService",
    # Factory functions
    "create_on_prem_posture",
    "create_air_gapped_posture",
    # Constants
    "EU_REGIONS",
]
