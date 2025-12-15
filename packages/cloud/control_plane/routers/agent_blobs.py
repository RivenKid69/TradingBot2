# -*- coding: utf-8 -*-
"""
Agent Blob/Artifact Access Router.

Design Doc 12.2 + 22.2: Agent-authenticated endpoints for blob and artifact access.

CLOUD ZONE ONLY.

Provides agent-authenticated endpoints for:
- Config blob lookup by digest
- Artifact download URL generation
- Payload reference resolution

The full flow per Design Doc 22.2:
1. Cloud sends command with payload_ref (digest) to Agent
2. Agent polls commands, receives payload_ref
3. Agent calls these endpoints to resolve payload_ref to content/download URL
4. Agent downloads artifact, verifies digest, applies

Security:
- All endpoints require AgentDep authentication
- Agent must be enrolled
- Workspace scoping enforced
- Presigned URLs are short-lived (5 minutes)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..database import get_session
from ..dependencies import AgentDep, CurrentAgent
from ..models import Agent, Artifact, ConfigBlob, TrustState

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Constants
# ============================================================================

# Presigned URL validity (short-lived for security)
PRESIGNED_URL_TTL_SECONDS = 300  # 5 minutes

# Environment variable for artifact storage base URL
ENV_ARTIFACT_STORAGE_URL = "CCEA_ARTIFACT_STORAGE_URL"
DEFAULT_ARTIFACT_STORAGE_URL = "http://localhost:9000/artifacts"

# Signing secret for presigned URLs
ENV_PRESIGN_SECRET = "CCEA_PRESIGN_SECRET"


# ============================================================================
# Request/Response Models
# ============================================================================

class ConfigBlobLookupResponse(BaseModel):
    """Response from config blob lookup."""
    id: UUID
    digest: str
    content: Dict[str, Any]
    size_bytes: int
    config_type: str
    schema_version: str
    created_at: datetime


class ArtifactDownloadResponse(BaseModel):
    """
    Response from artifact download request.

    Design Doc 22.2: Contains presigned URL for artifact download.
    """
    artifact_id: UUID
    artifact_name: str
    digest: str
    size_bytes: int
    download_url: str
    download_url_expires_at: datetime
    signature_info: Optional[Dict[str, Any]] = None


class PayloadRefResolution(BaseModel):
    """
    Resolution of payload_ref to actual content.

    payload_ref can point to:
    - Config blob (config:sha256:xxx)
    - Artifact (artifact:sha256:xxx)
    """
    payload_ref: str
    ref_type: str  # "config" or "artifact"
    resolved: bool
    content: Optional[Dict[str, Any]] = None  # For config blobs
    download_url: Optional[str] = None  # For artifacts
    download_url_expires_at: Optional[datetime] = None
    digest: str
    size_bytes: int


class BatchPayloadRefRequest(BaseModel):
    """Request to resolve multiple payload refs."""
    refs: List[str] = Field(..., max_length=10)


class BatchPayloadRefResponse(BaseModel):
    """Response with resolved payload refs."""
    resolutions: List[PayloadRefResolution]


# ============================================================================
# Helper Functions
# ============================================================================

async def verify_agent_enrolled(
    session,
    agent: CurrentAgent,
) -> Agent:
    """Verify agent exists and is enrolled."""
    result = await session.execute(
        select(Agent).where(
            Agent.id == agent.id,
            Agent.workspace_id == agent.workspace_id,
            Agent.deleted_at.is_(None),
        )
    )
    db_agent = result.scalar_one_or_none()

    if db_agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    if db_agent.trust_state != TrustState.ENROLLED.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent not enrolled. Current state: {db_agent.trust_state}",
        )

    return db_agent


def generate_presigned_url(
    base_url: str,
    path: str,
    expires_at: datetime,
    secret: Optional[str] = None,
) -> str:
    """
    Generate a presigned URL for secure download.

    Uses HMAC signature for URL authentication.

    Args:
        base_url: Base storage URL
        path: Path to artifact
        expires_at: Expiration timestamp
        secret: Signing secret (from env if not provided)

    Returns:
        Presigned URL string
    """
    secret = secret or os.environ.get(ENV_PRESIGN_SECRET, "dev-secret-change-me")
    expires_ts = int(expires_at.timestamp())

    # Create signature
    to_sign = f"{path}:{expires_ts}"
    signature = hmac.new(
        secret.encode("utf-8"),
        to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]

    return f"{base_url.rstrip('/')}/{path.lstrip('/')}?expires={expires_ts}&sig={signature}"


# ============================================================================
# Config Blob Endpoints
# ============================================================================

@router.get(
    "/config-blobs/by-digest/{digest}",
    response_model=ConfigBlobLookupResponse,
    summary="Get config blob by digest",
    description="Lookup config blob content by its digest (Design Doc 12.2).",
)
async def get_config_blob_by_digest(
    digest: str,
    current_agent: AgentDep,
) -> ConfigBlobLookupResponse:
    """
    Get config blob content by digest.

    Design Doc 12.2: Agent can retrieve config blobs by digest.
    This is how commands with config_ref payloads are resolved.

    Args:
        digest: Config blob digest (sha256:xxx format)
        current_agent: Authenticated agent

    Returns:
        Config blob with content

    Raises:
        404: Blob not found
        403: Agent not enrolled
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Lookup blob by digest in agent's workspace
        result = await session.execute(
            select(ConfigBlob).where(
                ConfigBlob.workspace_id == current_agent.workspace_id,
                ConfigBlob.digest == digest,
            )
        )
        blob = result.scalar_one_or_none()

        if blob is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Config blob with digest '{digest}' not found",
            )

        return ConfigBlobLookupResponse(
            id=blob.id,
            digest=blob.digest,
            content=blob.content,
            size_bytes=blob.size_bytes,
            config_type=blob.config_type,
            schema_version=blob.schema_version,
            created_at=blob.created_at,
        )


# ============================================================================
# Artifact Endpoints
# ============================================================================

@router.get(
    "/artifacts/by-digest/{digest}",
    response_model=ArtifactDownloadResponse,
    summary="Get artifact download URL",
    description="Get presigned download URL for artifact by digest (Design Doc 22.2).",
)
async def get_artifact_download_url(
    digest: str,
    current_agent: AgentDep,
) -> ArtifactDownloadResponse:
    """
    Get artifact download URL by digest.

    Design Doc 22.2 sequence step 8:
        Agent → Registry: pull artifact by digest

    This endpoint returns a presigned URL for the agent to download
    the artifact. The URL is short-lived (5 minutes) for security.

    Args:
        digest: Artifact digest (sha256:xxx format)
        current_agent: Authenticated agent

    Returns:
        Download URL and artifact metadata

    Raises:
        404: Artifact not found
        403: Agent not enrolled
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Lookup artifact by digest in agent's workspace
        result = await session.execute(
            select(Artifact).where(
                Artifact.workspace_id == current_agent.workspace_id,
                Artifact.digest == digest,
            )
        )
        artifact = result.scalar_one_or_none()

        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact with digest '{digest}' not found",
            )

        # Generate presigned download URL
        base_url = os.environ.get(
            ENV_ARTIFACT_STORAGE_URL,
            DEFAULT_ARTIFACT_STORAGE_URL,
        )
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=PRESIGNED_URL_TTL_SECONDS)

        # Path includes workspace and digest for proper isolation
        artifact_path = f"{current_agent.workspace_id}/{artifact.digest.replace(':', '_')}"
        download_url = generate_presigned_url(base_url, artifact_path, expires_at)

        # Signature info for verification (Design Doc Phase 4)
        # Uses signature_ref and signature_algorithm from Artifact model
        signature_info = None
        if artifact.is_signed and artifact.signature_ref:
            signature_info = {
                "algorithm": artifact.signature_algorithm or "sigstore",
                "signature_ref": artifact.signature_ref,
                "signed_digest": artifact.digest,
            }

        return ArtifactDownloadResponse(
            artifact_id=artifact.id,
            artifact_name=artifact.name,
            digest=artifact.digest,
            size_bytes=artifact.size_bytes or 0,
            download_url=download_url,
            download_url_expires_at=expires_at,
            signature_info=signature_info,
        )


# ============================================================================
# Payload Reference Resolution
# ============================================================================

@router.get(
    "/payload-refs/resolve",
    response_model=PayloadRefResolution,
    summary="Resolve payload reference",
    description="Resolve payload_ref from command to actual content/URL.",
)
async def resolve_payload_ref(
    payload_ref: str = Query(..., description="Payload reference to resolve"),
    current_agent: AgentDep = None,
) -> PayloadRefResolution:
    """
    Resolve a payload reference to its content or download URL.

    payload_ref format:
    - config:sha256:xxx - Points to a config blob
    - artifact:sha256:xxx - Points to an artifact

    Design Doc 22.2: This is how Agent resolves command payloads.

    Args:
        payload_ref: Reference string
        current_agent: Authenticated agent

    Returns:
        Resolved payload with content or download URL
    """
    async with get_session() as session:
        # Verify agent
        await verify_agent_enrolled(session, current_agent)

        # Parse ref type and digest
        if payload_ref.startswith("config:"):
            ref_type = "config"
            digest = payload_ref[7:]  # Remove "config:" prefix
        elif payload_ref.startswith("artifact:"):
            ref_type = "artifact"
            digest = payload_ref[9:]  # Remove "artifact:" prefix
        else:
            # Assume artifact if no prefix (backwards compatibility)
            ref_type = "artifact"
            digest = payload_ref

        if ref_type == "config":
            # Lookup config blob
            result = await session.execute(
                select(ConfigBlob).where(
                    ConfigBlob.workspace_id == current_agent.workspace_id,
                    ConfigBlob.digest == digest,
                )
            )
            blob = result.scalar_one_or_none()

            if blob is None:
                return PayloadRefResolution(
                    payload_ref=payload_ref,
                    ref_type=ref_type,
                    resolved=False,
                    digest=digest,
                    size_bytes=0,
                )

            return PayloadRefResolution(
                payload_ref=payload_ref,
                ref_type=ref_type,
                resolved=True,
                content=blob.content,
                digest=blob.digest,
                size_bytes=blob.size_bytes,
            )

        else:  # artifact
            # Lookup artifact
            result = await session.execute(
                select(Artifact).where(
                    Artifact.workspace_id == current_agent.workspace_id,
                    Artifact.digest == digest,
                )
            )
            artifact = result.scalar_one_or_none()

            if artifact is None:
                return PayloadRefResolution(
                    payload_ref=payload_ref,
                    ref_type=ref_type,
                    resolved=False,
                    digest=digest,
                    size_bytes=0,
                )

            # Generate download URL
            base_url = os.environ.get(
                ENV_ARTIFACT_STORAGE_URL,
                DEFAULT_ARTIFACT_STORAGE_URL,
            )
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=PRESIGNED_URL_TTL_SECONDS)
            artifact_path = f"{current_agent.workspace_id}/{artifact.digest.replace(':', '_')}"
            download_url = generate_presigned_url(base_url, artifact_path, expires_at)

            return PayloadRefResolution(
                payload_ref=payload_ref,
                ref_type=ref_type,
                resolved=True,
                download_url=download_url,
                download_url_expires_at=expires_at,
                digest=artifact.digest,
                size_bytes=artifact.size_bytes,
            )


@router.post(
    "/payload-refs/batch-resolve",
    response_model=BatchPayloadRefResponse,
    summary="Batch resolve payload references",
    description="Resolve multiple payload references in one request.",
)
async def batch_resolve_payload_refs(
    request: BatchPayloadRefRequest,
    current_agent: AgentDep,
) -> BatchPayloadRefResponse:
    """
    Batch resolve multiple payload references.

    Efficient way to resolve multiple refs in one round-trip.

    Args:
        request: List of payload refs to resolve
        current_agent: Authenticated agent

    Returns:
        List of resolutions
    """
    resolutions = []
    for ref in request.refs:
        # Use the single resolve function for each
        resolution = await resolve_payload_ref(ref, current_agent)
        resolutions.append(resolution)

    return BatchPayloadRefResponse(resolutions=resolutions)
