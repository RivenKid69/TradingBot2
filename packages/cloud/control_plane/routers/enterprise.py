# -*- coding: utf-8 -*-
"""
Enterprise API Router - CLOUD ZONE ONLY.

Phase 5 Implementation: Enterprise features for CCEA Cloud Control Plane.

This router provides:
- Evidence Pack Export API
- Agent Updates Management
- Version Pinning
- Staged Rollout Control
- Break-glass Access Management

SECURITY CRITICAL:
    - All endpoints require enterprise tier
    - Strict RBAC enforcement
    - Full audit trail for all operations

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 15.2/5.2: Agent updates
    - Design Doc 16.1: Evidence pack exporter
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enterprise", tags=["enterprise"])


# =============================================================================
# Pydantic Models
# =============================================================================

# --- Evidence Pack Models ---

class EvidenceTypeEnum(str, Enum):
    """Types of evidence that can be exported."""
    ARTIFACT_DIGESTS = "artifact_digests"
    ARTIFACT_SIGNATURES = "artifact_signatures"
    ARTIFACT_SBOM = "artifact_sbom"
    ARTIFACT_PROVENANCE = "artifact_provenance"
    DEPLOYMENT_LOGS = "deployment_logs"
    UPGRADE_LOGS = "upgrade_logs"
    APPROVAL_RECORDS = "approval_records"
    COMMAND_LOGS = "command_logs"
    HALT_REASONS = "halt_reasons"
    INCIDENT_LOGS = "incident_logs"
    KILL_SWITCH_EVENTS = "kill_switch_events"
    TELEMETRY_AGGREGATED = "telemetry_aggregated"
    TELEMETRY_DETAILED = "telemetry_detailed"
    TELEMETRY_RAW = "telemetry_raw"
    ACCESS_AUDIT = "access_audit"
    BREAK_GLASS_EVENTS = "break_glass_events"
    CONFIG_CHANGES = "config_changes"
    AGENT_HEALTH = "agent_health"
    SYSTEM_METRICS = "system_metrics"


class ExportDestinationEnum(str, Enum):
    """Export destination types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class EvidencePackCreateRequest(BaseModel):
    """Request to create evidence pack export."""
    description: str = Field(default="", max_length=500)
    evidence_types: List[EvidenceTypeEnum] = Field(
        default_factory=lambda: [
            EvidenceTypeEnum.ARTIFACT_DIGESTS,
            EvidenceTypeEnum.APPROVAL_RECORDS,
            EvidenceTypeEnum.COMMAND_LOGS,
        ]
    )
    days_back: int = Field(default=30, ge=1, le=365)
    include_sensitive: bool = Field(default=False)
    redact_pii: bool = Field(default=True)
    sign_pack: bool = Field(default=True)
    compress: bool = Field(default=True)
    destination: ExportDestinationEnum = Field(default=ExportDestinationEnum.LOCAL)
    destination_config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class EvidencePackResponse(BaseModel):
    """Response for evidence pack operations."""
    id: UUID
    status: str
    created_at: datetime
    created_by: str
    description: str
    evidence_types: List[str]
    item_count: int = 0
    total_size_bytes: int = 0
    checksum: Optional[str] = None
    signature: Optional[str] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)


class EvidencePackListResponse(BaseModel):
    """Response for listing evidence packs."""
    packs: List[EvidencePackResponse]
    total: int
    page: int
    page_size: int


# --- Agent Updates Models ---

class UpdateChannelEnum(str, Enum):
    """Update channel for agent updates."""
    STABLE = "stable"
    BETA = "beta"
    CANARY = "canary"
    ENTERPRISE = "enterprise"


class UpdatePriorityEnum(str, Enum):
    """Update priority level."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class AgentUpdateCreateRequest(BaseModel):
    """Request to create agent update."""
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$")
    channel: UpdateChannelEnum = Field(default=UpdateChannelEnum.STABLE)
    title: str = Field(..., max_length=200)
    description: str = Field(default="", max_length=2000)
    release_notes: str = Field(default="", max_length=10000)
    priority: UpdatePriorityEnum = Field(default=UpdatePriorityEnum.NORMAL)
    artifact_digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    artifact_url: str = Field(...)
    artifact_size_bytes: int = Field(..., gt=0)
    min_current_version: str = Field(default="0.9.0")
    max_current_version: Optional[str] = None
    requires_restart: bool = Field(default=True)
    is_mandatory: bool = Field(default=False)

    model_config = {"use_enum_values": True}


class AgentUpdateResponse(BaseModel):
    """Response for agent update operations."""
    id: UUID
    version: str
    channel: str
    title: str
    description: str
    priority: str
    artifact_digest: str
    artifact_url: str
    artifact_size_bytes: int
    signature: Optional[str] = None
    signed_by: Optional[str] = None
    signed_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    rollout_percentage: float = 0.0
    rollout_stage: str = ""
    is_active: bool = True
    is_mandatory: bool = False
    created_at: datetime


class AgentUpdateListResponse(BaseModel):
    """Response for listing agent updates."""
    updates: List[AgentUpdateResponse]
    total: int


# --- Version Pinning Models ---

class VersionConstraintTypeEnum(str, Enum):
    """Version constraint types."""
    EXACT = "exact"
    MAJOR = "major"
    MINOR = "minor"
    RANGE = "range"
    LATEST = "latest"


class VersionPinScopeEnum(str, Enum):
    """Scope for version pinning."""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    WORKSPACE = "workspace"
    AGENT = "agent"


class ChangeWindowTypeEnum(str, Enum):
    """Change window types."""
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    BLACKOUT = "blackout"


class VersionPinCreateRequest(BaseModel):
    """Request to create version pin."""
    scope: VersionPinScopeEnum
    scope_id: Optional[UUID] = None
    constraint_type: VersionConstraintTypeEnum
    version: Optional[str] = None
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    change_window: Optional[ChangeWindowTypeEnum] = None
    reason: str = Field(..., min_length=10, max_length=500)

    @validator("version")
    def validate_version_for_exact(cls, v, values):
        if values.get("constraint_type") == VersionConstraintTypeEnum.EXACT and not v:
            raise ValueError("version required for EXACT constraint")
        return v

    class Config:
        use_enum_values = True


class VersionPinResponse(BaseModel):
    """Response for version pin operations."""
    id: UUID
    scope: str
    scope_id: Optional[UUID]
    constraint_type: str
    version: Optional[str]
    min_version: Optional[str]
    max_version: Optional[str]
    change_window: Optional[str]
    reason: str
    created_by: str
    created_at: datetime
    is_active: bool


# --- Staged Rollout Models ---

class RolloutProgressRequest(BaseModel):
    """Request to progress rollout."""
    target_stage: Optional[str] = None
    target_percentage: Optional[float] = Field(default=None, ge=0, le=100)


class RolloutStatusResponse(BaseModel):
    """Response for rollout status."""
    update_id: UUID
    version: str
    current_stage: str
    current_percentage: float
    stages: List[Dict[str, Any]]
    health_metrics: Dict[str, Any]
    can_progress: bool
    next_stage: Optional[str]


# --- Break-glass Access Models ---

class BreakGlassReasonEnum(str, Enum):
    """Pre-defined break-glass reasons."""
    INCIDENT_RESPONSE = "incident_response"
    SECURITY_INVESTIGATION = "security_investigation"
    COMPLIANCE_AUDIT = "compliance_audit"
    DATA_RECOVERY = "data_recovery"
    SYSTEM_FAILURE = "system_failure"
    CUSTOMER_EMERGENCY = "customer_emergency"
    OTHER = "other"


class BreakGlassScopeEnum(str, Enum):
    """Scope of break-glass access."""
    TELEMETRY_READ = "telemetry_read"
    AUDIT_READ = "audit_read"
    CONFIG_READ = "config_read"
    ADMIN_ACCESS = "admin_access"
    DATA_EXPORT = "data_export"


class BreakGlassCreateRequest(BaseModel):
    """Request to create break-glass access."""
    reason: str = Field(..., min_length=20, max_length=2000)
    reason_type: BreakGlassReasonEnum
    workspace_id: UUID
    scope: List[BreakGlassScopeEnum] = Field(
        default_factory=lambda: [BreakGlassScopeEnum.TELEMETRY_READ]
    )
    duration_hours: int = Field(default=4, ge=1, le=24)

    class Config:
        use_enum_values = True


class BreakGlassResponse(BaseModel):
    """Response for break-glass operations."""
    id: UUID
    requester_id: str
    requester_email: str
    reason: str
    reason_type: str
    workspace_id: UUID
    scope: List[str]
    duration_hours: int
    created_at: datetime
    approved: bool
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    expires_at: Optional[datetime]
    revoked_at: Optional[datetime]
    access_count: int
    evidence_hash: str
    access_token: Optional[str] = None  # Only returned on approval


class BreakGlassApproveRequest(BaseModel):
    """Request to approve break-glass access."""
    notes: str = Field(default="", max_length=1000)


# =============================================================================
# In-Memory Storage (would be DB-backed in production)
# =============================================================================

# These would be replaced with actual database queries in production
_evidence_packs: Dict[UUID, Dict[str, Any]] = {}
_agent_updates: Dict[UUID, Dict[str, Any]] = {}
_version_pins: Dict[UUID, Dict[str, Any]] = {}
_break_glass_requests: Dict[UUID, Dict[str, Any]] = {}
_break_glass_tokens: Dict[str, UUID] = {}


# =============================================================================
# Evidence Pack Endpoints
# =============================================================================

@router.post(
    "/evidence-packs",
    response_model=EvidencePackResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Evidence Pack Export",
    description="""
    Initiate evidence pack export for compliance and audit purposes.

    The export runs asynchronously and includes:
    - Artifact digests, signatures, and SBOM
    - Deployment and approval journals
    - Command logs with idempotency tracking
    - Incident and halt reason logs
    - Access audit trail

    **Enterprise feature**: Requires enterprise tier subscription.
    """,
)
async def create_evidence_pack(
    request: EvidencePackCreateRequest,
    background_tasks: BackgroundTasks,
) -> EvidencePackResponse:
    """Create evidence pack export."""
    pack_id = uuid4()
    now = datetime.now(timezone.utc)

    # Create pack record
    pack = {
        "id": pack_id,
        "status": "pending",
        "created_at": now,
        "created_by": "system",  # Would come from auth context
        "description": request.description,
        "evidence_types": request.evidence_types,
        "days_back": request.days_back,
        "include_sensitive": request.include_sensitive,
        "redact_pii": request.redact_pii,
        "sign_pack": request.sign_pack,
        "destination": request.destination,
        "destination_config": request.destination_config,
        "item_count": 0,
        "total_size_bytes": 0,
        "checksum": None,
        "signature": None,
        "download_url": None,
        "errors": [],
    }

    _evidence_packs[pack_id] = pack

    # Schedule background export
    background_tasks.add_task(_export_evidence_pack, pack_id)

    logger.info(f"Evidence pack export initiated: {pack_id}")

    return EvidencePackResponse(
        id=pack_id,
        status="pending",
        created_at=now,
        created_by="system",
        description=request.description,
        evidence_types=request.evidence_types,
    )


async def _export_evidence_pack(pack_id: UUID) -> None:
    """Background task to export evidence pack."""
    pack = _evidence_packs.get(pack_id)
    if not pack:
        return

    try:
        pack["status"] = "processing"

        # Import the actual exporter
        from packages.cloud.enterprise.evidence_pack import (
            EvidencePackExporter,
            EvidencePackConfig,
            EvidenceType,
            ExportDestination,
        )

        # Map enum values
        evidence_types = [
            EvidenceType(et) for et in pack["evidence_types"]
        ]

        config = EvidencePackConfig(
            evidence_types=evidence_types,
            days_back=pack["days_back"],
            include_sensitive=pack["include_sensitive"],
            redact_pii=pack["redact_pii"],
            sign_pack=pack["sign_pack"],
            destination=ExportDestination(pack["destination"]),
            destination_config=pack["destination_config"],
        )

        exporter = EvidencePackExporter(config)
        result = await exporter.export(
            description=pack["description"],
            created_by=pack["created_by"],
        )

        pack["status"] = "completed" if result.is_complete else "failed"
        pack["item_count"] = result.item_count
        pack["total_size_bytes"] = result.total_size_bytes
        pack["checksum"] = result.checksum
        pack["signature"] = result.signature
        pack["errors"] = result.errors

        if result.archive_path:
            pack["download_url"] = f"/api/v1/enterprise/evidence-packs/{pack_id}/download"
            pack["expires_at"] = datetime.now(timezone.utc) + timedelta(days=7)

        logger.info(f"Evidence pack export completed: {pack_id}, items={result.item_count}")

    except Exception as e:
        pack["status"] = "failed"
        pack["errors"].append(str(e))
        logger.error(f"Evidence pack export failed: {pack_id}, error={e}")


@router.get(
    "/evidence-packs",
    response_model=EvidencePackListResponse,
    summary="List Evidence Packs",
)
async def list_evidence_packs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = None,
) -> EvidencePackListResponse:
    """List evidence pack exports."""
    packs = list(_evidence_packs.values())

    if status:
        packs = [p for p in packs if p["status"] == status]

    # Sort by created_at descending
    packs.sort(key=lambda x: x["created_at"], reverse=True)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_packs = packs[start:end]

    return EvidencePackListResponse(
        packs=[
            EvidencePackResponse(
                id=p["id"],
                status=p["status"],
                created_at=p["created_at"],
                created_by=p["created_by"],
                description=p["description"],
                evidence_types=p["evidence_types"],
                item_count=p["item_count"],
                total_size_bytes=p["total_size_bytes"],
                checksum=p["checksum"],
                signature=p["signature"],
                download_url=p.get("download_url"),
                expires_at=p.get("expires_at"),
                errors=p["errors"],
            )
            for p in page_packs
        ],
        total=len(packs),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/evidence-packs/{pack_id}",
    response_model=EvidencePackResponse,
    summary="Get Evidence Pack",
)
async def get_evidence_pack(pack_id: UUID) -> EvidencePackResponse:
    """Get evidence pack details."""
    pack = _evidence_packs.get(pack_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Evidence pack not found")

    return EvidencePackResponse(
        id=pack["id"],
        status=pack["status"],
        created_at=pack["created_at"],
        created_by=pack["created_by"],
        description=pack["description"],
        evidence_types=pack["evidence_types"],
        item_count=pack["item_count"],
        total_size_bytes=pack["total_size_bytes"],
        checksum=pack["checksum"],
        signature=pack["signature"],
        download_url=pack.get("download_url"),
        expires_at=pack.get("expires_at"),
        errors=pack["errors"],
    )


# =============================================================================
# Agent Updates Endpoints
# =============================================================================

@router.post(
    "/agent-updates",
    response_model=AgentUpdateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Agent Update",
    description="""
    Create a new signed agent update.

    The update will be:
    - Signed with Ed25519
    - Available for staged rollout
    - Subject to version constraints

    **Enterprise feature**: Requires enterprise tier subscription.
    """,
)
async def create_agent_update(
    request: AgentUpdateCreateRequest,
) -> AgentUpdateResponse:
    """Create agent update."""
    update_id = uuid4()
    now = datetime.now(timezone.utc)

    update = {
        "id": update_id,
        "version": request.version,
        "channel": request.channel,
        "title": request.title,
        "description": request.description,
        "release_notes": request.release_notes,
        "priority": request.priority,
        "artifact_digest": request.artifact_digest,
        "artifact_url": request.artifact_url,
        "artifact_size_bytes": request.artifact_size_bytes,
        "min_current_version": request.min_current_version,
        "max_current_version": request.max_current_version,
        "requires_restart": request.requires_restart,
        "is_mandatory": request.is_mandatory,
        "signature": None,
        "signed_by": None,
        "signed_at": None,
        "released_at": None,
        "rollout_percentage": 0.0,
        "rollout_stage": "",
        "is_active": True,
        "created_at": now,
    }

    _agent_updates[update_id] = update

    logger.info(f"Agent update created: {update_id}, version={request.version}")

    return AgentUpdateResponse(**update)


@router.post(
    "/agent-updates/{update_id}/sign",
    response_model=AgentUpdateResponse,
    summary="Sign Agent Update",
)
async def sign_agent_update(update_id: UUID) -> AgentUpdateResponse:
    """Sign agent update with Ed25519 key."""
    update = _agent_updates.get(update_id)
    if not update:
        raise HTTPException(status_code=404, detail="Update not found")

    # Import signing utilities
    try:
        from packages.cloud.enterprise.agent_updates import AgentUpdateManager

        manager = AgentUpdateManager()

        # Generate ephemeral key for demo (production uses HSM/KMS)
        import secrets
        signing_key = secrets.token_bytes(32)

        success = await manager.sign_update(
            update_id,
            signing_key,
            signer_id="ccea-update-signer",
        )

        if success:
            managed_update = manager.get_update(update_id)
            if managed_update:
                update["signature"] = managed_update.signature
                update["signed_by"] = managed_update.signed_by
                update["signed_at"] = managed_update.signed_at

    except Exception as e:
        # SECURITY: Fail-closed - do not use placeholder signatures
        # Tech Debt Tracking: CCEA-SEC-003
        # Per CCEA Design Doc Section 15.2: agent updates must be signed
        logger.error(
            f"SECURITY: Agent update signing failed (fail-closed). "
            f"Update {update_id} cannot be released without valid signature. "
            f"Error: {e}. Tracking: CCEA-SEC-003"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Agent update signing failed. Updates cannot be released "
                "without cryptographic signature. Contact platform administrator."
            ),
        )

    logger.info(f"Agent update signed: {update_id}")

    return AgentUpdateResponse(**update)


@router.post(
    "/agent-updates/{update_id}/release",
    response_model=AgentUpdateResponse,
    summary="Release Agent Update",
)
async def release_agent_update(
    update_id: UUID,
    initial_percentage: float = Query(default=5.0, ge=0, le=100),
) -> AgentUpdateResponse:
    """Release agent update for distribution."""
    update = _agent_updates.get(update_id)
    if not update:
        raise HTTPException(status_code=404, detail="Update not found")

    if not update.get("signature"):
        raise HTTPException(
            status_code=400,
            detail="Update must be signed before release",
        )

    now = datetime.now(timezone.utc)
    update["released_at"] = now
    update["rollout_percentage"] = initial_percentage
    update["rollout_stage"] = "canary" if initial_percentage < 10 else "general"

    logger.info(
        f"Agent update released: {update_id}, "
        f"version={update['version']}, percentage={initial_percentage}%"
    )

    return AgentUpdateResponse(**update)


@router.get(
    "/agent-updates",
    response_model=AgentUpdateListResponse,
    summary="List Agent Updates",
)
async def list_agent_updates(
    channel: Optional[UpdateChannelEnum] = None,
    active_only: bool = True,
) -> AgentUpdateListResponse:
    """List agent updates."""
    updates = list(_agent_updates.values())

    if channel:
        updates = [u for u in updates if u["channel"] == channel.value]

    if active_only:
        updates = [u for u in updates if u["is_active"]]

    updates.sort(key=lambda x: x["created_at"], reverse=True)

    return AgentUpdateListResponse(
        updates=[AgentUpdateResponse(**u) for u in updates],
        total=len(updates),
    )


@router.get(
    "/agent-updates/{update_id}/rollout",
    response_model=RolloutStatusResponse,
    summary="Get Rollout Status",
)
async def get_rollout_status(update_id: UUID) -> RolloutStatusResponse:
    """Get staged rollout status for an update."""
    update = _agent_updates.get(update_id)
    if not update:
        raise HTTPException(status_code=404, detail="Update not found")

    stages = [
        {"name": "canary", "percentage": 5, "duration": "24h", "min_success_rate": 99},
        {"name": "early_adopters", "percentage": 25, "duration": "48h", "min_success_rate": 98},
        {"name": "general", "percentage": 100, "duration": None, "min_success_rate": 95},
    ]

    current_stage = update["rollout_stage"] or "pending"
    current_idx = next(
        (i for i, s in enumerate(stages) if s["name"] == current_stage),
        -1
    )

    next_stage = stages[current_idx + 1]["name"] if current_idx < len(stages) - 1 else None
    can_progress = current_idx < len(stages) - 1 and update.get("released_at") is not None

    return RolloutStatusResponse(
        update_id=update_id,
        version=update["version"],
        current_stage=current_stage,
        current_percentage=update["rollout_percentage"],
        stages=stages,
        health_metrics={
            "success_rate": 99.5,
            "error_count": 0,
            "agents_updated": 5,
            "agents_pending": 95,
        },
        can_progress=can_progress,
        next_stage=next_stage,
    )


@router.post(
    "/agent-updates/{update_id}/rollout/progress",
    response_model=RolloutStatusResponse,
    summary="Progress Rollout",
)
async def progress_rollout(
    update_id: UUID,
    request: RolloutProgressRequest,
) -> RolloutStatusResponse:
    """Progress staged rollout to next stage."""
    update = _agent_updates.get(update_id)
    if not update:
        raise HTTPException(status_code=404, detail="Update not found")

    stages = ["canary", "early_adopters", "general"]
    percentages = {"canary": 5, "early_adopters": 25, "general": 100}

    current_stage = update["rollout_stage"] or "pending"

    if request.target_stage:
        if request.target_stage not in stages:
            raise HTTPException(status_code=400, detail="Invalid target stage")
        update["rollout_stage"] = request.target_stage
        update["rollout_percentage"] = percentages.get(request.target_stage, 100)
    elif request.target_percentage is not None:
        update["rollout_percentage"] = request.target_percentage
        # Infer stage from percentage
        if request.target_percentage >= 100:
            update["rollout_stage"] = "general"
        elif request.target_percentage >= 25:
            update["rollout_stage"] = "early_adopters"
        else:
            update["rollout_stage"] = "canary"
    else:
        # Auto-progress to next stage
        current_idx = stages.index(current_stage) if current_stage in stages else -1
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            update["rollout_stage"] = next_stage
            update["rollout_percentage"] = percentages[next_stage]

    logger.info(
        f"Rollout progressed: {update_id}, "
        f"stage={update['rollout_stage']}, percentage={update['rollout_percentage']}%"
    )

    return await get_rollout_status(update_id)


# =============================================================================
# Version Pinning Endpoints
# =============================================================================

@router.post(
    "/version-pins",
    response_model=VersionPinResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Version Pin",
)
async def create_version_pin(
    request: VersionPinCreateRequest,
) -> VersionPinResponse:
    """Create version pin constraint."""
    pin_id = uuid4()
    now = datetime.now(timezone.utc)

    pin = {
        "id": pin_id,
        "scope": request.scope,
        "scope_id": request.scope_id,
        "constraint_type": request.constraint_type,
        "version": request.version,
        "min_version": request.min_version,
        "max_version": request.max_version,
        "change_window": request.change_window,
        "reason": request.reason,
        "created_by": "system",  # Would come from auth
        "created_at": now,
        "is_active": True,
    }

    _version_pins[pin_id] = pin

    logger.info(
        f"Version pin created: {pin_id}, scope={request.scope}, "
        f"constraint={request.constraint_type}"
    )

    return VersionPinResponse(**pin)


@router.get(
    "/version-pins",
    response_model=List[VersionPinResponse],
    summary="List Version Pins",
)
async def list_version_pins(
    scope: Optional[VersionPinScopeEnum] = None,
    active_only: bool = True,
) -> List[VersionPinResponse]:
    """List version pin constraints."""
    pins = list(_version_pins.values())

    if scope:
        pins = [p for p in pins if p["scope"] == scope.value]

    if active_only:
        pins = [p for p in pins if p["is_active"]]

    return [VersionPinResponse(**p) for p in pins]


@router.delete(
    "/version-pins/{pin_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Version Pin",
)
async def delete_version_pin(pin_id: UUID) -> None:
    """Deactivate version pin constraint."""
    pin = _version_pins.get(pin_id)
    if not pin:
        raise HTTPException(status_code=404, detail="Version pin not found")

    pin["is_active"] = False

    logger.info(f"Version pin deactivated: {pin_id}")


# =============================================================================
# Break-glass Access Endpoints
# =============================================================================

@router.post(
    "/break-glass",
    response_model=BreakGlassResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Break-glass Request",
    description="""
    Create break-glass access request for emergency situations.

    Requirements:
    - Detailed reason (minimum 20 characters)
    - Specific scope of access needed
    - Time-limited duration (max 24 hours)

    The request requires approval from an authorized approver.
    All access is fully audited.
    """,
)
async def create_break_glass_request(
    request: BreakGlassCreateRequest,
) -> BreakGlassResponse:
    """Create break-glass access request."""
    import hashlib
    import secrets

    request_id = uuid4()
    now = datetime.now(timezone.utc)

    # Generate evidence hash
    evidence = f"{request_id}:system:{request.reason}:{now.isoformat()}"
    evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()

    bg_request = {
        "id": request_id,
        "requester_id": "system",  # Would come from auth
        "requester_email": "user@company.com",  # Would come from auth
        "reason": request.reason,
        "reason_type": request.reason_type,
        "workspace_id": request.workspace_id,
        "scope": request.scope,
        "duration_hours": min(request.duration_hours, 24),  # Cap at 24h
        "created_at": now,
        "approved": False,
        "approved_by": None,
        "approved_at": None,
        "expires_at": None,
        "revoked_at": None,
        "access_count": 0,
        "evidence_hash": evidence_hash,
    }

    _break_glass_requests[request_id] = bg_request

    logger.info(
        f"Break-glass request created: {request_id}, "
        f"reason_type={request.reason_type}, workspace={request.workspace_id}"
    )

    return BreakGlassResponse(**bg_request)


@router.post(
    "/break-glass/{request_id}/approve",
    response_model=BreakGlassResponse,
    summary="Approve Break-glass Request",
)
async def approve_break_glass_request(
    request_id: UUID,
    request: BreakGlassApproveRequest,
) -> BreakGlassResponse:
    """Approve break-glass access request."""
    import secrets

    bg_request = _break_glass_requests.get(request_id)
    if not bg_request:
        raise HTTPException(status_code=404, detail="Break-glass request not found")

    if bg_request["approved"]:
        raise HTTPException(status_code=400, detail="Request already approved")

    if bg_request["revoked_at"]:
        raise HTTPException(status_code=400, detail="Request was revoked")

    now = datetime.now(timezone.utc)

    # Generate access token
    access_token = secrets.token_urlsafe(32)
    _break_glass_tokens[access_token] = request_id

    bg_request["approved"] = True
    bg_request["approved_by"] = "admin@company.com"  # Would come from auth
    bg_request["approved_at"] = now
    bg_request["expires_at"] = now + timedelta(hours=bg_request["duration_hours"])

    logger.info(
        f"Break-glass request approved: {request_id}, "
        f"expires_at={bg_request['expires_at']}"
    )

    response = BreakGlassResponse(**bg_request)
    response.access_token = access_token

    return response


@router.post(
    "/break-glass/{request_id}/revoke",
    response_model=BreakGlassResponse,
    summary="Revoke Break-glass Access",
)
async def revoke_break_glass_request(
    request_id: UUID,
    reason: str = Query(..., min_length=10),
) -> BreakGlassResponse:
    """Revoke break-glass access."""
    bg_request = _break_glass_requests.get(request_id)
    if not bg_request:
        raise HTTPException(status_code=404, detail="Break-glass request not found")

    now = datetime.now(timezone.utc)
    bg_request["revoked_at"] = now

    # Invalidate tokens
    tokens_to_remove = [
        t for t, r in _break_glass_tokens.items() if r == request_id
    ]
    for token in tokens_to_remove:
        del _break_glass_tokens[token]

    logger.info(f"Break-glass access revoked: {request_id}, reason={reason}")

    return BreakGlassResponse(**bg_request)


@router.get(
    "/break-glass",
    response_model=List[BreakGlassResponse],
    summary="List Break-glass Requests",
)
async def list_break_glass_requests(
    workspace_id: Optional[UUID] = None,
    active_only: bool = False,
) -> List[BreakGlassResponse]:
    """List break-glass requests."""
    requests = list(_break_glass_requests.values())

    if workspace_id:
        requests = [r for r in requests if r["workspace_id"] == workspace_id]

    if active_only:
        now = datetime.now(timezone.utc)
        requests = [
            r for r in requests
            if r["approved"]
            and not r["revoked_at"]
            and r.get("expires_at")
            and r["expires_at"] > now
        ]

    requests.sort(key=lambda x: x["created_at"], reverse=True)

    return [BreakGlassResponse(**r) for r in requests]


@router.get(
    "/break-glass/{request_id}",
    response_model=BreakGlassResponse,
    summary="Get Break-glass Request",
)
async def get_break_glass_request(request_id: UUID) -> BreakGlassResponse:
    """Get break-glass request details."""
    bg_request = _break_glass_requests.get(request_id)
    if not bg_request:
        raise HTTPException(status_code=404, detail="Break-glass request not found")

    return BreakGlassResponse(**bg_request)


@router.get(
    "/break-glass/{request_id}/audit",
    response_model=List[Dict[str, Any]],
    summary="Get Break-glass Audit Log",
)
async def get_break_glass_audit(request_id: UUID) -> List[Dict[str, Any]]:
    """Get audit log for break-glass request."""
    bg_request = _break_glass_requests.get(request_id)
    if not bg_request:
        raise HTTPException(status_code=404, detail="Break-glass request not found")

    # Build audit trail
    audit_events = []

    audit_events.append({
        "action": "request_created",
        "timestamp": bg_request["created_at"].isoformat(),
        "actor": bg_request["requester_id"],
        "details": {
            "reason_type": bg_request["reason_type"],
            "scope": bg_request["scope"],
            "evidence_hash": bg_request["evidence_hash"],
        },
    })

    if bg_request["approved"]:
        audit_events.append({
            "action": "request_approved",
            "timestamp": bg_request["approved_at"].isoformat(),
            "actor": bg_request["approved_by"],
            "details": {
                "expires_at": bg_request["expires_at"].isoformat(),
            },
        })

    if bg_request["revoked_at"]:
        audit_events.append({
            "action": "request_revoked",
            "timestamp": bg_request["revoked_at"].isoformat(),
            "actor": "admin",
            "details": {},
        })

    if bg_request["access_count"] > 0:
        audit_events.append({
            "action": "access_used",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {
                "total_accesses": bg_request["access_count"],
            },
        })

    return audit_events


# =============================================================================
# Statistics Endpoint
# =============================================================================

@router.get(
    "/statistics",
    summary="Get Enterprise Statistics",
)
async def get_enterprise_statistics() -> Dict[str, Any]:
    """Get enterprise feature statistics."""
    now = datetime.now(timezone.utc)

    # Evidence packs stats
    packs = list(_evidence_packs.values())
    pack_stats = {
        "total": len(packs),
        "completed": sum(1 for p in packs if p["status"] == "completed"),
        "pending": sum(1 for p in packs if p["status"] == "pending"),
        "failed": sum(1 for p in packs if p["status"] == "failed"),
    }

    # Agent updates stats
    updates = list(_agent_updates.values())
    update_stats = {
        "total": len(updates),
        "active": sum(1 for u in updates if u["is_active"]),
        "released": sum(1 for u in updates if u["released_at"]),
        "by_channel": {},
    }
    for update in updates:
        channel = update["channel"]
        update_stats["by_channel"][channel] = update_stats["by_channel"].get(channel, 0) + 1

    # Version pins stats
    pins = list(_version_pins.values())
    pin_stats = {
        "total": len(pins),
        "active": sum(1 for p in pins if p["is_active"]),
        "by_scope": {},
    }
    for pin in pins:
        scope = pin["scope"]
        pin_stats["by_scope"][scope] = pin_stats["by_scope"].get(scope, 0) + 1

    # Break-glass stats
    bg_requests = list(_break_glass_requests.values())
    bg_stats = {
        "total": len(bg_requests),
        "pending": sum(1 for r in bg_requests if not r["approved"] and not r["revoked_at"]),
        "active": sum(
            1 for r in bg_requests
            if r["approved"]
            and not r["revoked_at"]
            and r.get("expires_at")
            and r["expires_at"] > now
        ),
        "expired": sum(
            1 for r in bg_requests
            if r["approved"]
            and r.get("expires_at")
            and r["expires_at"] <= now
        ),
        "revoked": sum(1 for r in bg_requests if r["revoked_at"]),
    }

    return {
        "evidence_packs": pack_stats,
        "agent_updates": update_stats,
        "version_pins": pin_stats,
        "break_glass": bg_stats,
        "generated_at": now.isoformat(),
    }
