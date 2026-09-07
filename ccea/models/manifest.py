# -*- coding: utf-8 -*-
"""
CCEA Artifact Manifest Models.

Pydantic models for artifact manifest per artifact_manifest.schema.json.
Defines strategy/model artifact metadata with signature and provenance.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Patterns
# ============================================================================

SEMVER_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$")
DIGEST_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")


# ============================================================================
# Enums
# ============================================================================


class ArtifactType(str, Enum):
    """Type of artifact."""

    STRATEGY = "strategy"
    MODEL = "model"
    FEATURE_PIPELINE = "feature_pipeline"
    ENSEMBLE = "ensemble"


class SandboxType(str, Enum):
    """Type of sandbox for artifact execution."""

    PROCESS = "process"  # Default process-level isolation
    CONTAINER = "container"  # Container-based isolation
    VM = "vm"  # VM-level isolation (enterprise)
    WASM = "wasm"  # WebAssembly sandbox


class Timeframe(str, Enum):
    """Trading timeframe."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class Platform(str, Enum):
    """Target platform."""

    LINUX = "linux"
    DARWIN = "darwin"
    WINDOWS = "windows"
    ANY = "any"


class ChangeClass(str, Enum):
    """Change classification."""

    TRADING_IMPACTING = "TRADING_IMPACTING"
    NON_TRADING_IMPACTING = "NON_TRADING_IMPACTING"
    DOCUMENTATION_ONLY = "DOCUMENTATION_ONLY"


class SignatureAlgorithm(str, Enum):
    """Signature algorithm types."""

    SIGSTORE = "sigstore"
    GPG = "gpg"
    X509 = "x509"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa-p256"


class Broker(str, Enum):
    """Supported brokers."""

    BINANCE = "binance"
    ALPACA = "alpaca"
    OANDA = "oanda"
    IB = "ib"
    DERIBIT = "deribit"


class MarketDataType(str, Enum):
    """Market data types."""

    BARS = "bars"
    TICKS = "ticks"
    ORDERBOOK = "orderbook"
    TRADES = "trades"


# ============================================================================
# Sub-models
# ============================================================================


class Entrypoint(BaseModel):
    """Artifact entrypoint definition."""

    module: str = Field(..., description="Python module path")
    class_name: str = Field(..., alias="class", description="Strategy/Model class name")
    function: Optional[str] = Field(None, description="Optional factory function")

    model_config = {"extra": "forbid", "populate_by_name": True}


class Runtime(BaseModel):
    """Runtime requirements."""

    python_version: str = Field(
        ..., pattern=r"^[0-9]+\.[0-9]+(\.[0-9]+)?$", description="Required Python version"
    )
    platform: Platform = Platform.ANY
    gpu_required: bool = False
    min_memory_mb: int = Field(1024, ge=256)

    model_config = {"extra": "forbid"}


class ModelRef(BaseModel):
    """Reference to a model file."""

    name: str = Field(..., description="Model file name")
    digest: str = Field(
        ..., pattern=r"^sha256:[a-f0-9]{64}$", description="SHA256 digest of model file"
    )
    size_bytes: Optional[int] = Field(None, ge=0)

    model_config = {"extra": "forbid"}


class DataContract(BaseModel):
    """Data contract specification."""

    input_schema: Optional[Dict[str, Any]] = Field(None, description="Expected input data schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output data schema")
    required_features: Optional[List[str]] = Field(
        None, description="List of required feature names"
    )
    supported_instruments: Optional[List[str]] = Field(
        None, description="Supported instrument types"
    )

    model_config = {"extra": "forbid"}


class FilesystemPermissions(BaseModel):
    """Filesystem permissions."""

    read: List[str] = Field(default_factory=list)
    write: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class NetworkPermissions(BaseModel):
    """Network permissions."""

    egress_allowed: bool = False
    egress_allowlist: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class Permissions(BaseModel):
    """Artifact permissions (sandbox constraints)."""

    filesystem: Optional[FilesystemPermissions] = None
    network: Optional[NetworkPermissions] = None

    model_config = {"extra": "forbid"}


class RiskProfileSuggested(BaseModel):
    """
    SUGGESTED risk profile - can be overridden by local policy.

    IMPORTANT: These are suggestions only. Local policy has priority.
    """

    max_position_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Suggested max position as % of portfolio"
    )
    max_daily_loss_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Suggested max daily loss as % of portfolio"
    )
    max_orders_per_minute: Optional[int] = Field(None, ge=1, description="Suggested max order rate")
    note: Literal["These are SUGGESTIONS only. Local policy has priority."] = Field(
        "These are SUGGESTIONS only. Local policy has priority.",
        description="Reminder that local policy overrides",
    )

    model_config = {"extra": "forbid"}


class LiveCapabilities(BaseModel):
    """Required capabilities for live trading."""

    requires_broker_access: bool = False
    supported_brokers: Optional[List[Broker]] = None
    required_market_data: Optional[List[MarketDataType]] = None
    min_latency_ms: Optional[int] = Field(None, ge=0)

    model_config = {"extra": "forbid"}


class DatasetRef(BaseModel):
    """Training dataset reference."""

    name: str
    digest: str = Field(..., pattern=r"^sha256:[a-f0-9]{64}$")
    date_range: Optional[str] = None

    model_config = {"extra": "forbid"}


class Provenance(BaseModel):
    """Artifact provenance information."""

    git_sha: str = Field(..., pattern=r"^[a-f0-9]{40}$", description="Git commit SHA")
    git_branch: Optional[str] = None
    git_repo: Optional[str] = None
    build_id: Optional[str] = None
    builder_version: Optional[str] = None
    dataset_refs: Optional[List[DatasetRef]] = None
    training_run_id: Optional[str] = None
    params_hash: Optional[str] = Field(None, pattern=r"^sha256:[a-f0-9]{64}$")

    model_config = {"extra": "forbid"}


class ManifestSignature(BaseModel):
    """Artifact signature."""

    algorithm: SignatureAlgorithm
    signature_value: str
    certificate: Optional[str] = None
    timestamp: Optional[datetime] = None
    key_id: Optional[str] = Field(None, description="Key identifier used for signing")
    signed_digest: Optional[str] = Field(
        None, pattern=r"^sha256:[a-f0-9]{64}$", description="Digest that was signed"
    )

    model_config = {"extra": "forbid"}


# ============================================================================
# Main Manifest Model
# ============================================================================


class ArtifactManifest(BaseModel):
    """
    CCEA Artifact Manifest.

    Defines strategy/model artifact metadata per Design Doc.
    Used for digest-pinning, signature verification, and deployment.
    """

    # Required fields
    schema_version: str = Field(
        ...,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$",
        description="Semantic version of this manifest schema",
        examples=["1.0.0"],
    )
    artifact_id: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]+$",
        min_length=1,
        max_length=128,
        description="Unique identifier for this artifact",
    )
    artifact_type: ArtifactType
    entrypoint: Entrypoint
    runtime: Runtime
    deps_lock_digest: str = Field(
        ..., pattern=r"^sha256:[a-f0-9]{64}$", description="SHA256 digest of locked dependencies"
    )
    created_at: datetime

    # Optional fields
    name: Optional[str] = Field(None, max_length=256)
    description: Optional[str] = Field(None, max_length=4096)
    version: Optional[str] = Field(None, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$")

    # Strategy reference (Design Doc 8.2)
    strategy_id: Optional[str] = Field(
        None,
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=128,
        description="Associated strategy identifier",
    )

    # Sandbox configuration (Design Doc 8.2)
    sandbox_type: SandboxType = Field(
        SandboxType.PROCESS, description="Type of sandbox for execution"
    )

    # Build timestamp (Design Doc 8.2)
    built_at: Optional[datetime] = Field(None, description="When the artifact was built")

    # Trading timeframe (Design Doc 8.2)
    timeframe: Optional[Timeframe] = Field(None, description="Trading timeframe for the strategy")

    model_refs: Optional[List[ModelRef]] = None
    data_contract: Optional[DataContract] = None
    permissions: Optional[Permissions] = None
    risk_profile_suggested: Optional[RiskProfileSuggested] = None
    live_capabilities: Optional[LiveCapabilities] = None
    telemetry_schema_version: Optional[str] = Field(None, pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    change_class: Optional[ChangeClass] = None
    provenance: Optional[Provenance] = None
    sbom_ref: Optional[str] = None
    signature: Optional[ManifestSignature] = None
    created_by: Optional[str] = None
    labels: Optional[Dict[str, str]] = None

    model_config = {"extra": "forbid"}

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if not SEMVER_PATTERN.match(v):
            raise ValueError("schema_version must be a valid semver (e.g., 1.0.0)")
        return v

    def compute_digest(self) -> str:
        """
        Compute SHA256 digest of manifest content.

        Returns:
            Digest string in format sha256:<hex>
        """
        import hashlib
        import json

        # Exclude signature from digest computation
        data = self.model_dump(exclude={"signature"}, mode="json")
        content = json.dumps(data, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(content.encode()).hexdigest()
        return f"sha256:{digest}"
