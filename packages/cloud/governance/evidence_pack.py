"""
Evidence Pack Service - Phase 7 GDPR Implementation

Generates comprehensive evidence packs for audit and compliance purposes.
Aggregates data from all Phase 7 services:
- Security baseline (encryption, keys, MFA, secrets)
- Supply chain (artifacts, SBOMs, signatures)
- Agent updates (rollouts, rollbacks, version pins)
- Research sandbox (policies, violations, abuse events)
- Breach workflow (incidents, notifications, tabletops)

Plus existing governance services:
- Access audit logs
- Change management journal
- RBAC snapshots
- DSAR records
- Retention policies

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L945-L958
"""

from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class EvidenceCategory(str, Enum):
    """Categories of evidence in the pack."""
    # Phase 7 categories
    SECURITY_BASELINE = "security_baseline"
    ENCRYPTION_CONFIG = "encryption_config"
    KEY_MANAGEMENT = "key_management"
    MFA_STATUS = "mfa_status"
    SECRETS_INVENTORY = "secrets_inventory"
    ARTIFACT_INVENTORY = "artifact_inventory"
    SBOM = "sbom"
    DIGEST_PINS = "digest_pins"
    REGISTRY_ALLOWLIST = "registry_allowlist"
    TRUSTED_SIGNERS = "trusted_signers"
    ROLLOUT_RECORDS = "rollout_records"
    ROLLBACK_RECORDS = "rollback_records"
    VERSION_PINS = "version_pins"
    SANDBOX_POLICIES = "sandbox_policies"
    SANDBOX_VIOLATIONS = "sandbox_violations"
    ABUSE_EVENTS = "abuse_events"
    EGRESS_LOGS = "egress_logs"
    BREACH_RECORDS = "breach_records"
    TABLETOP_REPORTS = "tabletop_reports"

    # Existing governance categories
    ACCESS_AUDIT = "access_audit"
    CHANGE_JOURNAL = "change_journal"
    RBAC_SNAPSHOT = "rbac_snapshot"
    DSAR_RECORDS = "dsar_records"
    RETENTION_POLICIES = "retention_policies"
    LEGAL_HOLDS = "legal_holds"
    CONSENT_RECORDS = "consent_records"
    RESIDENCY_EVIDENCE = "residency_evidence"
    TELEMETRY_CONTRACTS = "telemetry_contracts"


class ExportFormat(str, Enum):
    """Export format for evidence pack."""
    JSON = "json"
    ZIP = "zip"
    CSV = "csv"


class PackStatus(str, Enum):
    """Status of evidence pack generation."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvidenceArtifact:
    """An individual artifact in the evidence pack."""
    artifact_id: str = field(default_factory=lambda: str(uuid4()))
    category: EvidenceCategory = EvidenceCategory.ACCESS_AUDIT
    filename: str = ""
    content_type: str = "application/json"
    size_bytes: int = 0
    content_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    record_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "category": self.category.value,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "record_count": self.record_count,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceManifest:
    """Manifest for an evidence pack."""
    manifest_version: str = "1.0.0"
    pack_id: str = ""
    workspace_id: str = ""
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exported_by: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    categories_included: Set[EvidenceCategory] = field(default_factory=set)
    artifact_count: int = 0
    artifacts: List[Dict[str, str]] = field(default_factory=list)  # id -> hash mapping
    total_size_bytes: int = 0
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        if not self.manifest_hash:
            self.manifest_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "exported_at": self.exported_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "artifact_count": self.artifact_count,
            "artifacts": self.artifacts,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "exported_at": self.exported_at.isoformat(),
            "exported_by": self.exported_by,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "categories_included": [c.value for c in self.categories_included],
            "artifact_count": self.artifact_count,
            "artifacts": self.artifacts,
            "total_size_bytes": self.total_size_bytes,
            "manifest_hash": self.manifest_hash,
        }


@dataclass
class EvidencePack:
    """A complete evidence pack for audit."""
    pack_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    title: str = ""
    description: str = ""
    exported_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exported_by: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    categories: Set[EvidenceCategory] = field(default_factory=set)
    artifacts: List[EvidenceArtifact] = field(default_factory=list)
    manifest: Optional[EvidenceManifest] = None
    status: PackStatus = PackStatus.PENDING
    format: ExportFormat = ExportFormat.JSON
    integrity_hash: str = ""
    signature: Optional[str] = None
    download_url: Optional[str] = None
    download_expires_at: Optional[datetime] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "exported_at": self.exported_at.isoformat(),
            "artifact_count": len(self.artifacts),
            "categories": [c.value for c in self.categories],
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "description": self.description,
            "exported_at": self.exported_at.isoformat(),
            "exported_by": self.exported_by,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "categories": [c.value for c in self.categories],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "status": self.status.value,
            "format": self.format.value,
            "integrity_hash": self.integrity_hash,
            "signature": self.signature,
            "download_url": self.download_url,
            "download_expires_at": self.download_expires_at.isoformat() if self.download_expires_at else None,
            "error": self.error,
        }


@dataclass
class ExportRequest:
    """Request to generate an evidence pack."""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    requested_by: str = ""
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    categories: Set[EvidenceCategory] = field(default_factory=lambda: set(EvidenceCategory))
    format: ExportFormat = ExportFormat.JSON
    include_pii: bool = False  # Whether to include PII (requires authorization)
    reason: str = ""
    status: PackStatus = PackStatus.PENDING
    pack_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "workspace_id": self.workspace_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "categories": [c.value for c in self.categories],
            "format": self.format.value,
            "include_pii": self.include_pii,
            "reason": self.reason,
            "status": self.status.value,
            "pack_id": self.pack_id,
        }


@dataclass
class EvidenceEvent:
    """Event in the evidence pack audit log."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    pack_id: Optional[str] = None
    event_type: str = "export_requested"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
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
            "pack_id": self.pack_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "pack_id": self.pack_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


# =============================================================================
# Evidence Pack Service
# =============================================================================

class EvidencePackService:
    """
    Evidence Pack Service.

    Generates comprehensive evidence packs for audit and compliance.
    Aggregates data from all governance services into exportable packages.

    Reference: Design Doc Section 16.1
    """

    def __init__(
        self,
        # Phase 7 service references
        security_baseline_service: Optional[Any] = None,
        supply_chain_service: Optional[Any] = None,
        agent_update_service: Optional[Any] = None,
        research_sandbox_service: Optional[Any] = None,
        breach_workflow_service: Optional[Any] = None,
        # Existing governance service references
        access_audit_service: Optional[Any] = None,
        change_management_service: Optional[Any] = None,
        rbac_service: Optional[Any] = None,
        dsar_service: Optional[Any] = None,
        retention_service: Optional[Any] = None,
        consent_service: Optional[Any] = None,
        residency_service: Optional[Any] = None,
        telemetry_contract_service: Optional[Any] = None,
        # Callbacks
        on_event: Optional[Callable[[EvidenceEvent], None]] = None,
        on_pack_generated: Optional[Callable[[EvidencePack], None]] = None,
    ):
        """
        Initialize Evidence Pack Service.

        Args:
            security_baseline_service: Security baseline service
            supply_chain_service: Supply chain service
            agent_update_service: Agent update service
            research_sandbox_service: Research sandbox service
            breach_workflow_service: Breach workflow service
            access_audit_service: Access audit service
            change_management_service: Change management service
            rbac_service: RBAC service
            dsar_service: DSAR service
            retention_service: Retention service
            consent_service: Consent service
            residency_service: Residency drift checker service
            telemetry_contract_service: Telemetry contract service
            on_event: Callback for evidence events
            on_pack_generated: Callback when pack is generated
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Service references
        self._security_baseline = security_baseline_service
        self._supply_chain = supply_chain_service
        self._agent_updates = agent_update_service
        self._research_sandbox = research_sandbox_service
        self._breach_workflow = breach_workflow_service
        self._access_audit = access_audit_service
        self._change_management = change_management_service
        self._rbac = rbac_service
        self._dsar = dsar_service
        self._retention = retention_service
        self._consent = consent_service
        self._residency = residency_service
        self._telemetry_contract = telemetry_contract_service

        # Callbacks
        self._on_event = on_event
        self._on_pack_generated = on_pack_generated

        # Storage
        self._packs: Dict[str, EvidencePack] = {}  # pack_id -> pack
        self._requests: Dict[str, ExportRequest] = {}  # request_id -> request
        self._events: List[EvidenceEvent] = []

    # =========================================================================
    # Export Request Management
    # =========================================================================

    def request_export(
        self,
        workspace_id: str,
        requested_by: str,
        categories: Optional[Set[EvidenceCategory]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        format: ExportFormat = ExportFormat.JSON,
        include_pii: bool = False,
        reason: str = "",
    ) -> ExportRequest:
        """Request an evidence pack export."""
        request = ExportRequest(
            workspace_id=workspace_id,
            requested_by=requested_by,
            period_start=period_start or (datetime.now(timezone.utc) - timedelta(days=30)),
            period_end=period_end or datetime.now(timezone.utc),
            categories=categories or set(EvidenceCategory),
            format=format,
            include_pii=include_pii,
            reason=reason,
        )

        with self._lock:
            self._requests[request.request_id] = request

        # Log event
        event = EvidenceEvent(
            workspace_id=workspace_id,
            event_type="export_requested",
            actor_id=requested_by,
            details={
                "request_id": request.request_id,
                "categories_count": len(request.categories),
                "format": format.value,
                "include_pii": include_pii,
            },
        )
        self._log_event(event)

        return request

    def get_request(self, request_id: str) -> Optional[ExportRequest]:
        """Get export request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    # =========================================================================
    # Evidence Pack Generation
    # =========================================================================

    def generate_pack(
        self,
        request_id: str,
    ) -> EvidencePack:
        """Generate evidence pack from a request."""
        request = self.get_request(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        # Create pack
        pack = EvidencePack(
            workspace_id=request.workspace_id,
            title=f"Evidence Pack - {request.workspace_id}",
            description=request.reason or "Compliance evidence export",
            exported_by=request.requested_by,
            period_start=request.period_start,
            period_end=request.period_end,
            categories=request.categories,
            format=request.format,
            status=PackStatus.GENERATING,
        )

        with self._lock:
            self._packs[pack.pack_id] = pack
            request.pack_id = pack.pack_id
            request.status = PackStatus.GENERATING

        try:
            # Generate artifacts for each category
            for category in request.categories:
                artifact = self._generate_artifact(
                    workspace_id=request.workspace_id,
                    category=category,
                    period_start=request.period_start,
                    period_end=request.period_end,
                    include_pii=request.include_pii,
                )
                if artifact:
                    pack.artifacts.append(artifact)

            # Create manifest
            pack.manifest = EvidenceManifest(
                pack_id=pack.pack_id,
                workspace_id=request.workspace_id,
                exported_at=pack.exported_at,
                exported_by=pack.exported_by,
                period_start=pack.period_start,
                period_end=pack.period_end,
                categories_included=pack.categories,
                artifact_count=len(pack.artifacts),
                artifacts=[{"id": a.artifact_id, "hash": a.content_hash} for a in pack.artifacts],
                total_size_bytes=sum(a.size_bytes for a in pack.artifacts),
            )

            # Update status
            pack.status = PackStatus.COMPLETED
            pack.integrity_hash = pack._compute_hash()

            with self._lock:
                request.status = PackStatus.COMPLETED

            # Log event
            event = EvidenceEvent(
                workspace_id=request.workspace_id,
                pack_id=pack.pack_id,
                event_type="pack_generated",
                actor_id=request.requested_by,
                details={
                    "artifact_count": len(pack.artifacts),
                    "total_size_bytes": pack.manifest.total_size_bytes,
                },
            )
            self._log_event(event)

            if self._on_pack_generated:
                try:
                    self._on_pack_generated(pack)
                except Exception:
                    pass

        except Exception as e:
            pack.status = PackStatus.FAILED
            pack.error = str(e)

            with self._lock:
                request.status = PackStatus.FAILED

            # Log event
            event = EvidenceEvent(
                workspace_id=request.workspace_id,
                pack_id=pack.pack_id,
                event_type="pack_generation_failed",
                actor_id=request.requested_by,
                details={"error": str(e)},
                result="failure",
            )
            self._log_event(event)

        return pack

    def _generate_artifact(
        self,
        workspace_id: str,
        category: EvidenceCategory,
        period_start: datetime,
        period_end: datetime,
        include_pii: bool = False,
    ) -> Optional[EvidenceArtifact]:
        """Generate a single artifact for a category."""
        content: Dict[str, Any] = {}
        record_count = 0

        try:
            # Phase 7 categories
            if category == EvidenceCategory.SECURITY_BASELINE and self._security_baseline:
                content = self._security_baseline.export_security_baseline(
                    workspace_id=workspace_id,
                    include_events=True,
                    include_keys=True,
                    include_secrets=True,
                )
                record_count = len(content.get("keys", [])) + len(content.get("secrets", []))

            elif category == EvidenceCategory.ENCRYPTION_CONFIG and self._security_baseline:
                config = self._security_baseline.get_encryption_config(workspace_id)
                content = {"encryption_config": config.to_dict()}
                record_count = 1

            elif category == EvidenceCategory.KEY_MANAGEMENT and self._security_baseline:
                content = {
                    "keys": [k.to_dict() for k in self._security_baseline.list_keys(workspace_id=workspace_id)],
                    "rotations": [r.to_dict() for r in self._security_baseline.get_key_rotations(workspace_id=workspace_id)],
                }
                record_count = len(content.get("keys", []))

            elif category == EvidenceCategory.ARTIFACT_INVENTORY and self._supply_chain:
                content = self._supply_chain.export_inventory(
                    workspace_id=workspace_id,
                    include_sbom=False,
                    include_vulnerabilities=False,
                )
                record_count = len(content.get("artifacts", []))

            elif category == EvidenceCategory.SBOM and self._supply_chain:
                content = self._supply_chain.export_inventory(
                    workspace_id=workspace_id,
                    include_sbom=True,
                    include_vulnerabilities=True,
                )
                record_count = len(content.get("sboms", []))

            elif category == EvidenceCategory.DIGEST_PINS and self._supply_chain:
                pins = self._supply_chain.list_pins(workspace_id=workspace_id)
                content = {"digest_pins": [p.to_dict() for p in pins]}
                record_count = len(pins)

            elif category == EvidenceCategory.REGISTRY_ALLOWLIST and self._supply_chain:
                allowlist = self._supply_chain.get_registry_allowlist(workspace_id)
                content = {"registry_allowlist": allowlist.to_dict() if allowlist else None}
                record_count = 1 if allowlist else 0

            elif category == EvidenceCategory.TRUSTED_SIGNERS and self._supply_chain:
                signers = self._supply_chain.list_signers()
                content = {"trusted_signers": [s.to_dict() for s in signers]}
                record_count = len(signers)

            elif category == EvidenceCategory.ROLLOUT_RECORDS and self._agent_updates:
                content = self._agent_updates.export_rollout_records(
                    workspace_id=workspace_id,
                    include_events=True,
                )
                record_count = len(content.get("rollouts", []))

            elif category == EvidenceCategory.ROLLBACK_RECORDS and self._agent_updates:
                rollbacks = self._agent_updates.list_rollbacks()
                content = {"rollbacks": [r.to_dict() for r in rollbacks]}
                record_count = len(rollbacks)

            elif category == EvidenceCategory.VERSION_PINS and self._agent_updates:
                pins = self._agent_updates.list_version_pins()
                content = {"version_pins": [p.to_dict() for p in pins]}
                record_count = len(pins)

            elif category == EvidenceCategory.SANDBOX_POLICIES and self._research_sandbox:
                content = self._research_sandbox.export_sandbox_records(
                    workspace_id=workspace_id,
                    include_events=False,
                    include_abuse=False,
                    include_egress_logs=False,
                )
                record_count = 1

            elif category == EvidenceCategory.SANDBOX_VIOLATIONS and self._research_sandbox:
                abuse_events = self._research_sandbox.get_abuse_events(workspace_id=workspace_id)
                content = {"violations": [e.to_dict() for e in abuse_events]}
                record_count = len(abuse_events)

            elif category == EvidenceCategory.ABUSE_EVENTS and self._research_sandbox:
                abuse_events = self._research_sandbox.get_abuse_events(workspace_id=workspace_id)
                content = {"abuse_events": [e.to_dict() for e in abuse_events]}
                record_count = len(abuse_events)

            elif category == EvidenceCategory.EGRESS_LOGS and self._research_sandbox:
                logs = self._research_sandbox.get_egress_logs(workspace_id=workspace_id)
                content = {"egress_logs": [l.to_dict() for l in logs]}
                record_count = len(logs)

            elif category == EvidenceCategory.BREACH_RECORDS and self._breach_workflow:
                content = self._breach_workflow.export_breach_records(
                    workspace_id=workspace_id,
                    include_events=True,
                    include_exercises=False,
                )
                record_count = len(content.get("breaches", []))

            elif category == EvidenceCategory.TABLETOP_REPORTS and self._breach_workflow:
                content = self._breach_workflow.export_breach_records(
                    workspace_id=workspace_id,
                    include_events=False,
                    include_exercises=True,
                )
                record_count = len(content.get("tabletop_exercises", []))

            # Existing governance categories
            elif category == EvidenceCategory.ACCESS_AUDIT and self._access_audit:
                if hasattr(self._access_audit, 'export_audit_logs'):
                    content = self._access_audit.export_audit_logs(
                        workspace_id=workspace_id,
                        start_time=period_start,
                        end_time=period_end,
                    )
                else:
                    entries = self._access_audit.query(
                        workspace_id=workspace_id,
                        start_time=period_start,
                        end_time=period_end,
                    )
                    content = {"audit_entries": [e.to_dict() for e in entries]}
                record_count = len(content.get("audit_entries", content.get("entries", [])))

            elif category == EvidenceCategory.CHANGE_JOURNAL and self._change_management:
                if hasattr(self._change_management, 'export_journal'):
                    content = self._change_management.export_journal(workspace_id=workspace_id)
                else:
                    journal = self._change_management.get_journal(workspace_id=workspace_id)
                    content = {"change_journal": [j.to_dict() for j in journal]}
                record_count = len(content.get("change_journal", content.get("journal", [])))

            elif category == EvidenceCategory.RBAC_SNAPSHOT and self._rbac:
                if hasattr(self._rbac, 'export_snapshot'):
                    content = self._rbac.export_snapshot(workspace_id=workspace_id)
                else:
                    roles = self._rbac.list_roles()
                    content = {"roles": [r.to_dict() for r in roles]}
                record_count = len(content.get("roles", []))

            elif category == EvidenceCategory.DSAR_RECORDS and self._dsar:
                if hasattr(self._dsar, 'export_records'):
                    content = self._dsar.export_records(workspace_id=workspace_id)
                else:
                    requests = self._dsar.list_requests(workspace_id=workspace_id)
                    content = {"dsar_requests": [r.to_dict() for r in requests]}
                record_count = len(content.get("dsar_requests", content.get("requests", [])))

            elif category == EvidenceCategory.RETENTION_POLICIES and self._retention:
                if hasattr(self._retention, 'export_policies'):
                    content = self._retention.export_policies(workspace_id=workspace_id)
                else:
                    policies = self._retention.get_workspace_policies(workspace_id=workspace_id)
                    content = {"retention_policies": [p.to_dict() for p in policies]}
                record_count = len(content.get("retention_policies", content.get("policies", [])))

            elif category == EvidenceCategory.LEGAL_HOLDS and self._retention:
                if hasattr(self._retention, 'list_legal_holds'):
                    holds = self._retention.list_legal_holds(workspace_id=workspace_id)
                    content = {"legal_holds": [h.to_dict() for h in holds]}
                    record_count = len(holds)

            elif category == EvidenceCategory.CONSENT_RECORDS and self._consent:
                if hasattr(self._consent, 'export_records'):
                    content = self._consent.export_records(workspace_id=workspace_id)
                else:
                    records = self._consent.list_records(workspace_id=workspace_id)
                    content = {"consent_records": [r.to_dict() for r in records]}
                record_count = len(content.get("consent_records", content.get("records", [])))

            elif category == EvidenceCategory.RESIDENCY_EVIDENCE and self._residency:
                if hasattr(self._residency, 'export_evidence'):
                    content = self._residency.export_evidence(workspace_id=workspace_id)
                else:
                    report = self._residency.run_drift_check()
                    content = {"residency_report": report.to_dict() if hasattr(report, 'to_dict') else report}
                record_count = 1

            elif category == EvidenceCategory.TELEMETRY_CONTRACTS and self._telemetry_contract:
                if hasattr(self._telemetry_contract, 'export_contracts'):
                    content = self._telemetry_contract.export_contracts(workspace_id=workspace_id)
                else:
                    content = {"telemetry_contracts": "See telemetry contract documentation"}
                record_count = 1

            else:
                # Category not supported or service not available
                content = {
                    "category": category.value,
                    "message": "Service not available or category not supported",
                }
                record_count = 0

        except Exception as e:
            content = {
                "category": category.value,
                "error": str(e),
            }
            record_count = 0

        # Add metadata
        content["_metadata"] = {
            "category": category.value,
            "workspace_id": workspace_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "include_pii": include_pii,
        }

        # Serialize content
        content_json = json.dumps(content, sort_keys=True, default=str, indent=2)
        content_bytes = content_json.encode('utf-8')
        content_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"

        artifact = EvidenceArtifact(
            category=category,
            filename=f"{category.value}.json",
            content_type="application/json",
            size_bytes=len(content_bytes),
            content_hash=content_hash,
            record_count=record_count,
            period_start=period_start,
            period_end=period_end,
            metadata={"workspace_id": workspace_id},
        )

        return artifact

    def get_pack(self, pack_id: str) -> Optional[EvidencePack]:
        """Get evidence pack by ID."""
        with self._lock:
            return self._packs.get(pack_id)

    def list_packs(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[PackStatus] = None,
        limit: int = 100,
    ) -> List[EvidencePack]:
        """List evidence packs."""
        with self._lock:
            packs = list(self._packs.values())

        if workspace_id:
            packs = [p for p in packs if p.workspace_id == workspace_id]
        if status:
            packs = [p for p in packs if p.status == status]

        packs.sort(key=lambda p: p.exported_at, reverse=True)
        return packs[:limit]

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_pack_as_json(
        self,
        pack_id: str,
    ) -> Dict[str, Any]:
        """Export pack as JSON dictionary."""
        pack = self.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Pack {pack_id} not found")

        if pack.status != PackStatus.COMPLETED:
            raise ValueError(f"Pack is in {pack.status.value} status")

        return pack.to_dict()

    def export_pack_as_zip(
        self,
        pack_id: str,
        workspace_id: str,
    ) -> bytes:
        """Export pack as ZIP archive."""
        pack = self.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Pack {pack_id} not found")

        if pack.status != PackStatus.COMPLETED:
            raise ValueError(f"Pack is in {pack.status.value} status")

        # Create in-memory ZIP
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            manifest_content = json.dumps(pack.manifest.to_dict(), indent=2, default=str)
            zf.writestr("manifest.json", manifest_content)

            # Add each artifact
            for artifact in pack.artifacts:
                # Regenerate artifact content
                artifact_content = self._generate_artifact(
                    workspace_id=workspace_id,
                    category=artifact.category,
                    period_start=pack.period_start,
                    period_end=pack.period_end,
                )
                if artifact_content:
                    # Get the actual content (we stored metadata in artifact)
                    content_data = {
                        "artifact_id": artifact.artifact_id,
                        "category": artifact.category.value,
                        "period_start": artifact.period_start.isoformat() if artifact.period_start else None,
                        "period_end": artifact.period_end.isoformat() if artifact.period_end else None,
                        "record_count": artifact.record_count,
                        "content_hash": artifact.content_hash,
                    }
                    content_json = json.dumps(content_data, indent=2, default=str)
                    zf.writestr(f"artifacts/{artifact.filename}", content_json)

            # Add pack metadata
            pack_content = json.dumps(pack.to_dict(), indent=2, default=str)
            zf.writestr("pack.json", pack_content)

        zip_buffer.seek(0)
        return zip_buffer.read()

    def generate_download_url(
        self,
        pack_id: str,
        expires_in_hours: int = 24,
    ) -> EvidencePack:
        """Generate a download URL for a pack (placeholder for actual storage integration)."""
        with self._lock:
            pack = self._packs.get(pack_id)
            if not pack:
                raise ValueError(f"Pack {pack_id} not found")

            # In production, this would upload to secure storage and generate signed URL
            pack.download_url = f"/api/v1/evidence/packs/{pack_id}/download"
            pack.download_expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        # Log event
        event = EvidenceEvent(
            workspace_id=pack.workspace_id,
            pack_id=pack_id,
            event_type="download_url_generated",
            actor_id="system",
            details={"expires_in_hours": expires_in_hours},
        )
        self._log_event(event)

        return pack

    # =========================================================================
    # Quick Export Methods
    # =========================================================================

    def quick_export_security_controls(
        self,
        workspace_id: str,
        exported_by: str,
    ) -> EvidencePack:
        """Quick export of security-related evidence."""
        security_categories = {
            EvidenceCategory.SECURITY_BASELINE,
            EvidenceCategory.ENCRYPTION_CONFIG,
            EvidenceCategory.KEY_MANAGEMENT,
            EvidenceCategory.MFA_STATUS,
            EvidenceCategory.ARTIFACT_INVENTORY,
            EvidenceCategory.SBOM,
            EvidenceCategory.ACCESS_AUDIT,
        }

        request = self.request_export(
            workspace_id=workspace_id,
            requested_by=exported_by,
            categories=security_categories,
            reason="Security controls evidence export",
        )

        return self.generate_pack(request.request_id)

    def quick_export_breach_evidence(
        self,
        workspace_id: str,
        exported_by: str,
        breach_id: Optional[str] = None,
    ) -> EvidencePack:
        """Quick export of breach-related evidence."""
        breach_categories = {
            EvidenceCategory.BREACH_RECORDS,
            EvidenceCategory.TABLETOP_REPORTS,
            EvidenceCategory.ACCESS_AUDIT,
            EvidenceCategory.CHANGE_JOURNAL,
        }

        request = self.request_export(
            workspace_id=workspace_id,
            requested_by=exported_by,
            categories=breach_categories,
            reason=f"Breach evidence export{f' for {breach_id}' if breach_id else ''}",
        )

        return self.generate_pack(request.request_id)

    def quick_export_supply_chain(
        self,
        workspace_id: str,
        exported_by: str,
    ) -> EvidencePack:
        """Quick export of supply chain evidence."""
        supply_chain_categories = {
            EvidenceCategory.ARTIFACT_INVENTORY,
            EvidenceCategory.SBOM,
            EvidenceCategory.DIGEST_PINS,
            EvidenceCategory.REGISTRY_ALLOWLIST,
            EvidenceCategory.TRUSTED_SIGNERS,
            EvidenceCategory.ROLLOUT_RECORDS,
            EvidenceCategory.ROLLBACK_RECORDS,
            EvidenceCategory.VERSION_PINS,
        }

        request = self.request_export(
            workspace_id=workspace_id,
            requested_by=exported_by,
            categories=supply_chain_categories,
            reason="Supply chain evidence export",
        )

        return self.generate_pack(request.request_id)

    def quick_export_compliance(
        self,
        workspace_id: str,
        exported_by: str,
    ) -> EvidencePack:
        """Quick export of compliance-related evidence (DSAR, retention, consent)."""
        compliance_categories = {
            EvidenceCategory.DSAR_RECORDS,
            EvidenceCategory.RETENTION_POLICIES,
            EvidenceCategory.LEGAL_HOLDS,
            EvidenceCategory.CONSENT_RECORDS,
            EvidenceCategory.RESIDENCY_EVIDENCE,
            EvidenceCategory.TELEMETRY_CONTRACTS,
        }

        request = self.request_export(
            workspace_id=workspace_id,
            requested_by=exported_by,
            categories=compliance_categories,
            reason="Compliance evidence export",
        )

        return self.generate_pack(request.request_id)

    # =========================================================================
    # Audit and Statistics
    # =========================================================================

    def _log_event(self, event: EvidenceEvent) -> None:
        """Log an evidence event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        pack_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EvidenceEvent]:
        """Get evidence events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if pack_id:
            events = [e for e in events if e.pack_id == pack_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get evidence pack statistics."""
        packs = self.list_packs(workspace_id=workspace_id, limit=10000)
        completed = [p for p in packs if p.status == PackStatus.COMPLETED]

        total_size = sum(p.manifest.total_size_bytes if p.manifest else 0 for p in completed)
        total_artifacts = sum(len(p.artifacts) for p in completed)

        return {
            "workspace_id": workspace_id,
            "total_packs": len(packs),
            "completed_packs": len(completed),
            "failed_packs": len([p for p in packs if p.status == PackStatus.FAILED]),
            "total_size_bytes": total_size,
            "total_artifacts": total_artifacts,
            "categories_covered": list(set(c.value for p in completed for c in p.categories)),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
